"""
Overdamped Langevin Dynamics — 3D Superellipsoid Granules + Cell Tracking
==========================================================================
Numba-accelerated version. All inner loops JIT-compiled.

EQUATIONS OF MOTION:
    γ_i dx_i/dt = Σ_j F_ij^contact + Σ_j F_ij^cell + F_i^wall + F_i^noise

Usage: python overdamped_langevin.py config.json
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path
import json, sys, time as timer

try:
    from numba import njit, prange, float64, int32, boolean
    HAS_NUMBA = True; print("[Accel] Numba detected — JIT kernels active")
except ImportError:
    HAS_NUMBA = False; print("[Accel] Numba not found — pure NumPy fallback")
try:
    from scipy.spatial import cKDTree; HAS_SCIPY = True
except ImportError: HAS_SCIPY = False
try:
    from scipy.ndimage import label as sp_label; HAS_LABEL = True
except ImportError: HAS_LABEL = False
try:
    from tqdm import tqdm; HAS_TQDM = True
except ImportError: HAS_TQDM = False
try:
    import matplotlib.pyplot as plt; HAS_PLT = True
except ImportError: HAS_PLT = False


# =============================================================================
# NUMBA KERNELS
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _quat_to_rot(q):
        """Quaternion [w,x,y,z] → 3×3 rotation matrix."""
        w,x,y,z = q[0],q[1],q[2],q[3]
        R = np.empty((3,3), dtype=float64)
        R[0,0]=1-2*y*y-2*z*z; R[0,1]=2*x*y-2*w*z;   R[0,2]=2*x*z+2*w*y
        R[1,0]=2*x*y+2*w*z;   R[1,1]=1-2*x*x-2*z*z; R[1,2]=2*y*z-2*w*x
        R[2,0]=2*x*z-2*w*y;   R[2,1]=2*y*z+2*w*x;   R[2,2]=1-2*x*x-2*y*y
        return R

    @njit(cache=True)
    def _support_radius(sp, R, d):
        """Superellipsoid support function: surface distance along direction d."""
        a,b,c,n = sp[0],sp[1],sp[2],sp[3]
        # Transform direction to body frame
        db0 = R[0,0]*d[0]+R[1,0]*d[1]+R[2,0]*d[2]
        db1 = R[0,1]*d[0]+R[1,1]*d[1]+R[2,1]*d[2]
        db2 = R[0,2]*d[0]+R[1,2]*d[1]+R[2,2]*d[2]
        v = abs(db0/a)**n + abs(db1/b)**n + abs(db2/c)**n
        if v < 1e-30:
            return max(a, max(b, c))
        return 1.0 / v**(1.0/n)

    @njit(cache=True)
    def _bounding_r(sp):
        return max(sp[0], max(sp[1], sp[2]))

    @njit(cache=True)
    def _point_inside_se(px, py, pz, pos, sp, R):
        """Test if world point is inside superellipsoid."""
        rx = px-pos[0]; ry = py-pos[1]; rz = pz-pos[2]
        bx = R[0,0]*rx+R[1,0]*ry+R[2,0]*rz
        by = R[0,1]*rx+R[1,1]*ry+R[2,1]*rz
        bz = R[0,2]*rx+R[1,2]*ry+R[2,2]*rz
        return abs(bx/sp[0])**sp[3] + abs(by/sp[1])**sp[3] + abs(bz/sp[2])**sp[3] <= 1.0

    # ── Packing force kernel ──
    @njit(parallel=True, cache=True)
    def _packing_forces(pos, sp, q, dom, sc, k, mf):
        """Granule-granule + wall repulsion during growth packing."""
        n = pos.shape[0]
        f = np.zeros((n, 3), dtype=float64)
        max_ov = np.zeros(1, dtype=float64)
        for i in prange(n):
            Ri = _quat_to_rot(q[i])
            si = np.empty(4, dtype=float64)
            si[0]=sp[i,0]*sc; si[1]=sp[i,1]*sc; si[2]=sp[i,2]*sc; si[3]=sp[i,3]
            rbi = _bounding_r(si)
            for j in range(i+1, n):
                dx = pos[j,0]-pos[i,0]; dy = pos[j,1]-pos[i,1]; dz = pos[j,2]-pos[i,2]
                d = np.sqrt(dx*dx + dy*dy + dz*dz)
                if d < 1e-6: continue
                sj = np.empty(4, dtype=float64)
                sj[0]=sp[j,0]*sc; sj[1]=sp[j,1]*sc; sj[2]=sp[j,2]*sc; sj[3]=sp[j,3]
                if d > rbi + _bounding_r(sj): continue
                inv_d = 1.0/d
                dv = np.array([dx*inv_d, dy*inv_d, dz*inv_d])
                nd = np.array([-dv[0], -dv[1], -dv[2]])
                ri = _support_radius(si, Ri, dv)
                Rj = _quat_to_rot(q[j])
                rj = _support_radius(sj, Rj, nd)
                ov = ri + rj - d
                if ov > 0:
                    F = min(k*ov, mf)
                    for c in range(3):
                        f[i,c] -= F*dv[c]; f[j,c] += F*dv[c]
        return f

    @njit(parallel=True, cache=True)
    def _wall_forces_packing(pos, sp, q, dom, sc, k, mf):
        """Wall repulsion during packing."""
        n = pos.shape[0]
        f = np.zeros((n, 3), dtype=float64)
        for i in prange(n):
            Ri = _quat_to_rot(q[i])
            si = np.empty(4, dtype=float64)
            si[0]=sp[i,0]*sc; si[1]=sp[i,1]*sc; si[2]=sp[i,2]*sc; si[3]=sp[i,3]
            for d in range(3):
                dl = np.zeros(3, dtype=float64); dl[d] = 1.0
                rl = _support_radius(si, Ri, dl)
                if pos[i,d] < rl:
                    f[i,d] += min(k*(rl - pos[i,d]), mf)
                dh = np.zeros(3, dtype=float64); dh[d] = -1.0
                rh = _support_radius(si, Ri, dh)
                if pos[i,d] > dom[d] - rh:
                    f[i,d] -= min(k*(pos[i,d] - (dom[d]-rh)), mf)
        return f

    # ── Contact & bridge detection ──
    @njit(cache=True)
    def _find_contacts_bridges(pos, sp, q, types, n_cells, pi, pj, n_pairs, max_bg):
        """Find contacts and bridges from precomputed pair list."""
        # Preallocate max possible
        ci = np.empty(n_pairs, dtype=int32); cj = np.empty(n_pairs, dtype=int32)
        co = np.empty(n_pairs, dtype=float64)
        cnx = np.empty(n_pairs, dtype=float64); cny = np.empty(n_pairs, dtype=float64)
        cnz = np.empty(n_pairs, dtype=float64)
        cri = np.empty(n_pairs, dtype=float64); crj = np.empty(n_pairs, dtype=float64)
        nc = 0

        bi = np.empty(n_pairs, dtype=int32); bj = np.empty(n_pairs, dtype=int32)
        bnc = np.empty(n_pairs, dtype=int32)
        bd = np.empty(n_pairs, dtype=float64)
        bnx = np.empty(n_pairs, dtype=float64); bny = np.empty(n_pairs, dtype=float64)
        bnz = np.empty(n_pairs, dtype=float64)
        bri = np.empty(n_pairs, dtype=float64); brj = np.empty(n_pairs, dtype=float64)
        nb = 0

        for idx in range(n_pairs):
            i = pi[idx]; j = pj[idx]
            dx = pos[j,0]-pos[i,0]; dy = pos[j,1]-pos[i,1]; dz = pos[j,2]-pos[i,2]
            dt = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dt < 1e-6: continue
            inv_d = 1.0/dt
            dvx = dx*inv_d; dvy = dy*inv_d; dvz = dz*inv_d
            dv = np.array([dvx, dvy, dvz])
            nd = np.array([-dvx, -dvy, -dvz])
            Ri = _quat_to_rot(q[i]); ri = _support_radius(sp[i], Ri, dv)
            Rj = _quat_to_rot(q[j]); rj = _support_radius(sp[j], Rj, nd)
            ov = ri + rj - dt

            if ov > 0:
                ci[nc]=i; cj[nc]=j; co[nc]=ov
                cnx[nc]=dvx; cny[nc]=dvy; cnz[nc]=dvz
                cri[nc]=ri; crj[nc]=rj; nc += 1

            if types[i] == 0 and types[j] == 0:
                gap = dt - ri - rj
                if gap < max_bg:
                    pr = max(0.0, 1.0 - max(0.0, gap)/max_bg)
                    nb_ = max(1, int(np.sqrt(float(n_cells[i])*float(n_cells[j]))*pr))
                    bi[nb]=i; bj[nb]=j; bnc[nb]=nb_; bd[nb]=dt
                    bnx[nb]=dvx; bny[nb]=dvy; bnz[nb]=dvz
                    bri[nb]=ri; brj[nb]=rj; nb += 1

        return (ci[:nc], cj[:nc], co[:nc], cnx[:nc], cny[:nc], cnz[:nc], cri[:nc], crj[:nc],
                bi[:nb], bj[:nb], bnc[:nb], bd[:nb], bnx[:nb], bny[:nb], bnz[:nb], bri[:nb], brj[:nb])

    # ── Force computation ──
    @njit(parallel=True, cache=True)
    def _compute_forces(ng, pos, vel, sp, q, types, n_cells, dom,
                        ci, cj, co, cnx, cny, cnz, cri, crj, n_con,
                        bi, bj, bnc, bd, bdx, bdy, bdz, bri_, brj, n_br,
                        kr, kw, da, kc, cd, mf):
        """All forces: contact + cell bridge + wall + cap."""
        f = np.zeros((ng, 3), dtype=float64)

        # Contact repulsion (Hertz-like)
        for idx in range(n_con):
            i = ci[idx]; j = cj[idx]; ov = co[idx]
            nx = cnx[idx]; ny = cny[idx]; nz = cnz[idx]
            ri = cri[idx]; rj = crj[idx]
            re = np.sqrt(ri * rj)
            Fm = kr * re/50.0 * ov * (1.0 + 2.0*ov/max(re, 1.0))
            if Fm > mf: Fm = mf
            vn = (vel[j,0]-vel[i,0])*nx + (vel[j,1]-vel[i,1])*ny + (vel[j,2]-vel[i,2])*nz
            Ft = Fm - da*0.5*vn
            if Ft < 0: Ft = 0.0
            f[i,0] -= Ft*nx; f[i,1] -= Ft*ny; f[i,2] -= Ft*nz
            f[j,0] += Ft*nx; f[j,1] += Ft*ny; f[j,2] += Ft*nz

        # Cell bridge forces
        for idx in range(n_br):
            i = bi[idx]; j = bj[idx]
            gap = bd[idx] - bri_[idx] - brj[idx]
            ext = gap - cd
            F = kc * bnc[idx] * ext
            if ext < 0: F *= 0.1
            if F > mf*0.5: F = mf*0.5
            elif F < -mf*0.5: F = -mf*0.5
            f[i,0] += F*bdx[idx]; f[i,1] += F*bdy[idx]; f[i,2] += F*bdz[idx]
            f[j,0] -= F*bdx[idx]; f[j,1] -= F*bdy[idx]; f[j,2] -= F*bdz[idx]

        # Wall forces (stiffening contact)
        for i in prange(ng):
            Ri = _quat_to_rot(q[i])
            for d in range(3):
                dl = np.zeros(3, dtype=float64); dl[d] = 1.0
                rl = _support_radius(sp[i], Ri, dl)
                if pos[i,d] < rl:
                    ov = rl - pos[i,d]
                    ff = kw * ov * (1.0 + ov/max(rl, 1.0))
                    f[i,d] += min(ff, mf)
                dh = np.zeros(3, dtype=float64); dh[d] = -1.0
                rh = _support_radius(sp[i], Ri, dh)
                if pos[i,d] > dom[d] - rh:
                    ov = pos[i,d] - (dom[d] - rh)
                    ff = kw * ov * (1.0 + ov/max(rh, 1.0))
                    f[i,d] -= min(ff, mf)
            # Force cap per particle
            fx = f[i,0]; fy = f[i,1]; fz = f[i,2]
            fm = np.sqrt(fx*fx + fy*fy + fz*fz)
            if fm > mf:
                s = mf/fm; f[i,0]*=s; f[i,1]*=s; f[i,2]*=s
        return f

    # ── Slice rendering ──
    @njit(parallel=True, cache=True)
    def _render_slice_kernel(pos, sp, q, types, dom_d1, dom_d2, slice_pos,
                             axis, d1, d2, res):
        """Rasterise granule cross-sections onto 2D grid."""
        dx = dom_d1 / res; dy = dom_d2 / res
        phi_f = np.zeros((res, res), dtype=float64)
        phi_i = np.zeros((res, res), dtype=float64)
        ng = pos.shape[0]

        for ix in prange(res):
            px_d1 = (ix + 0.5) * dx
            for iy in range(res):
                py_d2 = (iy + 0.5) * dy
                for g in range(ng):
                    br = _bounding_r(sp[g])
                    dist_plane = abs(pos[g, axis] - slice_pos)
                    if dist_plane > br: continue
                    # Build 3D point
                    pt = np.zeros(3, dtype=float64)
                    pt[d1] = px_d1; pt[d2] = py_d2; pt[axis] = slice_pos
                    # Bounding sphere quick reject
                    ddx = pt[0]-pos[g,0]; ddy = pt[1]-pos[g,1]; ddz = pt[2]-pos[g,2]
                    if ddx*ddx+ddy*ddy+ddz*ddz > br*br: continue
                    R = _quat_to_rot(q[g])
                    if _point_inside_se(pt[0], pt[1], pt[2], pos[g], sp[g], R):
                        if types[g] == 0:
                            phi_f[ix, iy] = 1.0
                        else:
                            phi_i[ix, iy] = 1.0
                        break  # pixel claimed by first granule found
        return phi_f, phi_i

    # ── Relaxation kernel ──
    @njit(cache=True)
    def _relax_kernel(pos, sp, q, dom, sc, k, damp, mf, max_iter, tol):
        """Iterative overlap relaxation."""
        n = pos.shape[0]; mo = 1e6
        for it in range(max_iter):
            f = np.zeros((n, 3), dtype=float64); mo = 0.0
            for i in range(n):
                Ri = _quat_to_rot(q[i])
                si = np.empty(4, dtype=float64)
                si[0]=sp[i,0]*sc; si[1]=sp[i,1]*sc; si[2]=sp[i,2]*sc; si[3]=sp[i,3]
                rbi = _bounding_r(si)
                for j in range(i+1, n):
                    dx=pos[j,0]-pos[i,0]; dy=pos[j,1]-pos[i,1]; dz=pos[j,2]-pos[i,2]
                    d = np.sqrt(dx*dx+dy*dy+dz*dz)
                    if d < 1e-6: continue
                    sj = np.empty(4, dtype=float64)
                    sj[0]=sp[j,0]*sc; sj[1]=sp[j,1]*sc; sj[2]=sp[j,2]*sc; sj[3]=sp[j,3]
                    if d > rbi + _bounding_r(sj): continue
                    inv_d=1.0/d; dv=np.array([dx*inv_d,dy*inv_d,dz*inv_d])
                    nd=np.array([-dv[0],-dv[1],-dv[2]])
                    ri=_support_radius(si,Ri,dv)
                    Rj=_quat_to_rot(q[j]); rj=_support_radius(sj,Rj,nd)
                    ov=ri+rj-d
                    if ov>0:
                        if ov>mo: mo=ov
                        F=min(k*ov,mf)
                        for c in range(3): f[i,c]-=F*dv[c]; f[j,c]+=F*dv[c]
            # Walls
            for i in range(n):
                Ri=_quat_to_rot(q[i])
                si=np.empty(4,dtype=float64)
                si[0]=sp[i,0]*sc;si[1]=sp[i,1]*sc;si[2]=sp[i,2]*sc;si[3]=sp[i,3]
                for d in range(3):
                    dl=np.zeros(3,dtype=float64);dl[d]=1.0
                    rl=_support_radius(si,Ri,dl)
                    if pos[i,d]<rl: f[i,d]+=min(k*(rl-pos[i,d]),mf)
                    dh=np.zeros(3,dtype=float64);dh[d]=-1.0
                    rh=_support_radius(si,Ri,dh)
                    if pos[i,d]>dom[d]-rh: f[i,d]-=min(k*(pos[i,d]-(dom[d]-rh)),mf)
            # Apply with cap
            for i in range(n):
                fx=f[i,0];fy=f[i,1];fz=f[i,2]
                fm=np.sqrt(fx*fx+fy*fy+fz*fz)
                if fm>mf: s=mf/fm;f[i,0]*=s;f[i,1]*=s;f[i,2]*=s
            pos += f * 0.3 / damp
            # Clamp
            for i in range(n):
                si=np.empty(4,dtype=float64)
                si[0]=sp[i,0]*sc;si[1]=sp[i,1]*sc;si[2]=sp[i,2]*sc;si[3]=sp[i,3]
                br=_bounding_r(si)
                for d in range(3):
                    if pos[i,d]<br+0.5: pos[i,d]=br+0.5
                    if pos[i,d]>dom[d]-br-0.5: pos[i,d]=dom[d]-br-0.5
            if mo < tol: break
        return mo

else:
    # ── Pure NumPy fallbacks ──
    def _quat_to_rot(q):
        w,x,y,z = q; R = np.empty((3,3))
        R[0,0]=1-2*y*y-2*z*z;R[0,1]=2*x*y-2*w*z;R[0,2]=2*x*z+2*w*y
        R[1,0]=2*x*y+2*w*z;R[1,1]=1-2*x*x-2*z*z;R[1,2]=2*y*z-2*w*x
        R[2,0]=2*x*z-2*w*y;R[2,1]=2*y*z+2*w*x;R[2,2]=1-2*x*x-2*y*y
        return R

    def _support_radius(sp, R, d):
        db = R.T @ d; a,b,c,n = sp
        v = abs(db[0]/a)**n + abs(db[1]/b)**n + abs(db[2]/c)**n
        return max(a,b,c) if v < 1e-30 else 1.0/v**(1.0/n)

    def _bounding_r(sp): return max(sp[0], sp[1], sp[2])


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULTS = {
    "simulation_name": "sim",
    "domain": {"side_length_um": 500.0, "target_packing_fraction": 0.55},
    "granule_ratio": {"functional_fraction": 0.5},
    "functional_granules": {"radius_mean_um":40,"radius_std_um":8,
        "aspect_ratio_range":[1,1.5],"roundness_range":[2,3],"roughness_range":[0.1,0.3]},
    "inert_granules": {"radius_mean_um":50,"radius_std_um":10,
        "aspect_ratio_range":[1,1.3],"roundness_range":[2,2.5],"roughness_range":[0,0.2]},
    "cell_properties": {"diameter_um":20,"attachment_area_fraction":0.5,
        "force_per_cell_nN":5,"max_bridge_gap_um":50},
    "mechanics": {"repulsion_stiffness":1,"damping":1.5,"wall_stiffness":2},
    "time": {"total_hours":72,"save_interval_hours":2,
        "dt_initial_hours":0.01,"dt_min_hours":0.001,"dt_max_hours":0.5},
    "output": {"base_directory":"./simulations","save_true_shapes":True,
        "render_resolution":150}
}

def load_config(fp):
    with open(fp) as f: cfg = json.load(f)
    def merge(a,b):
        for k,v in b.items():
            if k not in a: a[k]=v
            elif isinstance(v,dict) and isinstance(a[k],dict): merge(a[k],v)
    merge(cfg, DEFAULTS); return cfg

def select_config():
    try:
        import tkinter as tk; from tkinter import filedialog
        r=tk.Tk();r.withdraw();r.attributes('-topmost',True)
        fp=filedialog.askopenfilename(title="Config",filetypes=[("JSON","*.json")])
        r.destroy(); return fp or None
    except: return None

def calc_granule_counts(cfg):
    s=cfg["domain"]["side_length_um"];phi=cfg["domain"]["target_packing_fraction"]
    ff=cfg["granule_ratio"]["functional_fraction"];V=s**3*phi
    rf=cfg["functional_granules"]["radius_mean_um"];ri=cfg["inert_granules"]["radius_mean_um"]
    return max(1,round(V*ff/((4/3)*np.pi*rf**3))), max(1,round(V*(1-ff)/((4/3)*np.pi*ri**3)))

def calc_cells(sa, cfg):
    cd=cfg["cell_properties"]["diameter_um"];af=cfg["cell_properties"]["attachment_area_fraction"]
    return max(1, int(np.ceil(sa*af/(np.pi*(cd/2)**2))))


# =============================================================================
# GEOMETRY
# =============================================================================

class Quaternion:
    __slots__ = ['q']
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w,x,y,z], dtype=np.float64)
        n = np.linalg.norm(self.q)
        if n > 1e-10: self.q /= n
    @classmethod
    def random(cls):
        u = np.random.random(3)
        return cls(np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),
                   np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                   np.sqrt(u[0])*np.sin(2*np.pi*u[2]),
                   np.sqrt(u[0])*np.cos(2*np.pi*u[2]))
    def matrix(self):
        w,x,y,z = self.q
        return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                         [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                         [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])


@dataclass
class Shape:
    a: float; b: float; c: float; n: float = 2.0; roughness: float = 0.0
    @property
    def params(self): return np.array([self.a,self.b,self.c,self.n])
    @property
    def volume(self): return (4/3)*np.pi*self.a*self.b*self.c
    @property
    def eq_radius(self): return (self.a*self.b*self.c)**(1/3)
    @property
    def bounding_radius(self): return max(self.a,self.b,self.c)
    @property
    def surface_area(self):
        p=1.6075; ap,bp,cp = self.a**p,self.b**p,self.c**p
        return 4*np.pi*((ap*bp+ap*cp+bp*cp)/3)**(1/p)
    def to_dict(self):
        return dict(a=self.a,b=self.b,c=self.c,n=self.n,roughness=self.roughness)


def create_shape(mr, csec):
    ar=np.random.uniform(*csec["aspect_ratio_range"])
    n=np.random.uniform(*csec["roundness_range"])
    ro=np.random.uniform(*csec["roughness_range"])
    c=mr*ar**(1/3); ab=mr/ar**(1/6)
    asy=np.random.uniform(0.95,1.05); a=ab*asy; b=ab/asy
    tv=(4/3)*np.pi*mr**3; cv=(4/3)*np.pi*a*b*c
    if cv>0: s=(tv/cv)**(1/3); a*=s; b*=s; c*=s
    return Shape(a,b,c,n,ro)


# =============================================================================
# CELL TRACKER
# =============================================================================

def fibonacci_sphere(n):
    pts = np.empty((n,3))
    golden = np.pi*(3-np.sqrt(5))
    for i in range(n):
        y = 1-2*i/(n-1) if n>1 else 0.0
        r = np.sqrt(max(0,1-y*y)); th = golden*i
        pts[i] = [r*np.cos(th), y, r*np.sin(th)]
    return pts

def se_surface_point(d, a, b, c, n_exp):
    d = d/np.linalg.norm(d)
    v = abs(d[0]/a)**n_exp + abs(d[1]/b)**n_exp + abs(d[2]/c)**n_exp
    if v<1e-30: return d*max(a,b,c)
    return d/v**(1.0/n_exp)


class CellTracker:
    def __init__(self, ng, nc_arr, shapes, cfg):
        self.ng = ng
        self.cell_diam = cfg["cell_properties"]["diameter_um"]
        self.k_cell = cfg["cell_properties"]["force_per_cell_nN"]
        self.max_gap = cfg["cell_properties"]["max_bridge_gap_um"]
        self.cell_vol = (4/3)*np.pi*(self.cell_diam/2)**3

        self.body_pos = []; self.offsets = np.zeros(ng+1, dtype=int)
        total = 0
        for g in range(ng):
            nc = int(nc_arr[g])
            if nc > 0:
                sh = shapes[g]; dirs = fibonacci_sphere(nc)
                pts = np.array([se_surface_point(dirs[k],sh.a,sh.b,sh.c,sh.n) for k in range(nc)])
                self.body_pos.append(pts)
            else: self.body_pos.append(np.empty((0,3)))
            self.offsets[g+1] = self.offsets[g]+nc; total += nc

        self.total_cells = int(total)
        self.world_pos = np.zeros((self.total_cells,3))
        self.is_bridging = np.zeros(self.total_cells, dtype=bool)
        self.bridge_target = -np.ones(self.total_cells, dtype=int)
        self.gap = np.zeros(self.total_cells)
        self.stress = np.zeros(self.total_cells)
        self.aspect_ratio = np.ones(self.total_cells)
        self.parent = np.zeros(self.total_cells, dtype=int)
        for g in range(ng):
            s,e = self.offsets[g], self.offsets[g+1]; self.parent[s:e] = g

    def update(self, positions, orientations, bridges):
        self.is_bridging[:]=False; self.bridge_target[:]=-1
        self.stress[:]=0; self.aspect_ratio[:]=1; self.gap[:]=0
        for g in range(self.ng):
            s,e = self.offsets[g], self.offsets[g+1]
            if e<=s: continue
            R = orientations[g].matrix()
            self.world_pos[s:e] = (R @ self.body_pos[g].T).T + positions[g]
        b_i,b_j,b_nc,b_d,b_nx,b_ny,b_nz,b_ri,b_rj = bridges
        if len(b_i)==0: return
        for idx in range(len(b_i)):
            gi,gj = int(b_i[idx]),int(b_j[idx]); n_br = int(b_nc[idx])
            dirn = np.array([b_nx[idx],b_ny[idx],b_nz[idx]])
            gap = max(0.0, float(b_d[idx]-b_ri[idx]-b_rj[idx]))
            si,ei = self.offsets[gi],self.offsets[gi+1]
            sj,ej = self.offsets[gj],self.offsets[gj+1]
            if ei<=si or ej<=sj: continue
            ci=self.world_pos[si:ei]; cj=self.world_pos[sj:ej]
            ct_i=np.mean(ci,axis=0); ct_j=np.mean(cj,axis=0)
            sc_i=np.sum((ci-ct_i)*dirn,axis=1); sc_j=np.sum((cj-ct_j)*(-dirn),axis=1)
            n_pick = min(n_br, ei-si, ej-sj)
            top_i = np.argsort(sc_i)[-n_pick:]; top_j = np.argsort(sc_j)[-n_pick:]
            for k in range(n_pick):
                ci_g=si+top_i[k]; cj_g=sj+top_j[k]
                self.is_bridging[ci_g]=self.is_bridging[cj_g]=True
                self.bridge_target[ci_g]=gj; self.bridge_target[cj_g]=gi
                self.gap[ci_g]=self.gap[cj_g]=gap
                L=max(self.cell_diam,gap); w=np.sqrt(4*self.cell_vol/(np.pi*L))
                ar=L/max(w,1e-6); self.aspect_ratio[ci_g]=self.aspect_ratio[cj_g]=ar
                ext=max(0,gap-self.cell_diam); F=self.k_cell*ext
                self.stress[ci_g]=self.stress[cj_g]=F

    def get_frame_data(self):
        return {"total_cells":int(self.total_cells),"parent":self.parent.tolist(),
                "world_pos":self.world_pos.tolist(),"is_bridging":self.is_bridging.tolist(),
                "gap_um":self.gap.tolist(),"stress_nN":self.stress.tolist(),
                "aspect_ratio":self.aspect_ratio.tolist()}


# =============================================================================
# PACKING GENERATOR
# =============================================================================

class PackingGenerator:
    def __init__(self, ds, target, verbose=True):
        self.dom = np.array([ds]*3); self.target = target; self.verbose = verbose

    def generate(self, n, shapes, orients):
        if self.verbose: print(f"\n  Growing packing for {n} superellipsoids...")
        dom = self.dom.copy()
        sp = np.array([s.params for s in shapes])
        q = np.array([o.q for o in orients])

        # Grid init
        pos = np.zeros((n,3)); ns = int(np.ceil(n**(1/3))); g = min(dom)/(ns+1); idx=0
        for ix in range(ns):
            for iy in range(ns):
                for iz in range(ns):
                    if idx>=n: break
                    pos[idx]=[(ix+1)*g+np.random.uniform(-g*.1,g*.1),
                              (iy+1)*g+np.random.uniform(-g*.1,g*.1),
                              (iz+1)*g+np.random.uniform(-g*.1,g*.1)]
                    pos[idx]=np.clip(pos[idx],20,dom-20); idx+=1
                if idx>=n: break
            if idx>=n: break

        sc=0.05; bgr=0.0005; k=1.0; mf=50.0; phi=0.0
        if self.verbose and HAS_TQDM: pb=tqdm(total=self.target,desc="  Growing",unit="φ"); lp=0

        for it in range(400000):
            if phi >= self.target: break
            pr = phi/max(self.target, 1e-6)
            gr = bgr*(0.02 if pr>.8 else 0.1 if pr>.6 else 0.5 if pr>.3 else 1)

            if HAS_NUMBA:
                f = _packing_forces(pos, sp, q, dom, sc, k, mf)
                f += _wall_forces_packing(pos, sp, q, dom, sc, k, mf)
            else:
                f = self._forces_np(pos, sp, q, dom, sc, k, mf)

            fm = np.linalg.norm(f, axis=1, keepdims=True)
            big = (fm>mf).flatten()
            if np.any(big): f[big] *= mf/fm[big]
            pos += f * 0.4 / 2.0

            nan = np.any(np.isnan(pos), axis=1)
            if np.any(nan): pos[nan] = dom/2+np.random.randn(int(np.sum(nan)),3)*10
            br = np.max(sp[:,:3], axis=1)*sc
            for d in range(3): pos[:,d] = np.clip(pos[:,d], br+1, dom[d]-br-1)

            sc += gr
            vols = (4/3)*np.pi*sp[:,0]*sp[:,1]*sp[:,2]*sc**3
            phi = np.sum(vols)/np.prod(dom)

            if it%200==0 and it>0:
                if HAS_NUMBA:
                    mr = np.min(sp[:,:3])*sc
                    _relax_kernel(pos, sp, q, dom, sc, k*2, 2.0, mf, 50, mr*0.3)
                else:
                    self._relax_np(pos, sp, q, dom, sc, k*2, 2.0, mf, 50)

            if self.verbose and HAS_TQDM and phi-lp>0.005: pb.update(phi-lp); lp=phi

        if self.verbose and HAS_TQDM: pb.close()

        # Multi-phase relaxation
        mr = np.min(sp[:,:3])*sc; tol = mr*0.05
        if self.verbose: print(f"  Relaxing (tol={tol:.2f}µm)...")
        for phase, (kf, df) in enumerate([(4,2),(16,0.5),(64,0.25)]):
            if HAS_NUMBA:
                ov = _relax_kernel(pos, sp, q, dom, sc, k*kf, df, mf*(phase+1)*2, 10000, tol)
            else:
                ov = self._relax_np(pos, sp, q, dom, sc, k*kf, df, mf*(phase+1)*2, 10000)
            if self.verbose: print(f"    Phase {phase+1}: overlap={ov:.3f}µm")
            if ov < tol: break

        for sh in shapes: sh.a*=sc; sh.b*=sc; sh.c*=sc
        if self.verbose: print(f"  Final φ: {phi:.3f}")
        return pos, orients

    def _forces_np(self, pos, sp, q, dom, sc, k, mf):
        """NumPy fallback for packing forces."""
        n=len(pos); f=np.zeros((n,3))
        for i in range(n):
            Ri=_quat_to_rot(q[i]); si=sp[i].copy(); si[:3]*=sc; rbi=max(si[:3])
            for j in range(i+1,n):
                rij=pos[j]-pos[i]; d=np.linalg.norm(rij)
                if d<1e-6: continue
                sj=sp[j].copy(); sj[:3]*=sc
                if d>rbi+max(sj[:3]): continue
                dv=rij/d; ri=_support_radius(si,Ri,dv)
                Rj=_quat_to_rot(q[j]); rj=_support_radius(sj,Rj,-dv)
                ov=ri+rj-d
                if ov>0: F=min(k*ov,mf)*dv; f[i]-=F; f[j]+=F
        for i in range(n):
            Ri=_quat_to_rot(q[i]); si=sp[i].copy(); si[:3]*=sc
            for ax in range(3):
                dl=np.zeros(3);dl[ax]=1; rl=_support_radius(si,Ri,dl)
                if pos[i,ax]<rl: f[i,ax]+=min(k*(rl-pos[i,ax]),mf)
                dh=np.zeros(3);dh[ax]=-1; rh=_support_radius(si,Ri,dh)
                if pos[i,ax]>dom[ax]-rh: f[i,ax]-=min(k*(pos[i,ax]-(dom[ax]-rh)),mf)
        return f

    def _relax_np(self, pos, sp, q, dom, sc, k, damp, mf, iters):
        mo = 1e6
        for it in range(iters):
            f = self._forces_np(pos, sp, q, dom, sc, k, mf)
            fm=np.linalg.norm(f,axis=1,keepdims=True); big=(fm>mf).flatten()
            if np.any(big): f[big]*=mf/fm[big]
            pos += f*0.3/damp
            br=np.max(sp[:,:3],axis=1)*sc
            for d in range(3): pos[:,d]=np.clip(pos[:,d],br+0.5,dom[d]-br-0.5)
        return 0.0  # no overlap tracking in fallback


# =============================================================================
# CONTACT/BRIDGE DETECTION (dispatched)
# =============================================================================

def find_contacts_bridges(pos, sp, q, types, n_cells, max_bg):
    n = len(pos); max_r = np.max(sp[:,:3]); cutoff = 2*max_r + max_bg
    if HAS_SCIPY:
        tree = cKDTree(pos); pairs = tree.query_pairs(cutoff, output_type='ndarray')
        if len(pairs) == 0:
            pi = np.array([], dtype=np.int32); pj = np.array([], dtype=np.int32)
        else:
            pi = pairs[:,0].astype(np.int32); pj = pairs[:,1].astype(np.int32)
    else:
        pi = np.array([i for i in range(n) for j in range(i+1,n)], dtype=np.int32)
        pj = np.array([j for i in range(n) for j in range(i+1,n)], dtype=np.int32)

    np_ = len(pi)
    if np_ == 0:
        ec = tuple(np.array([],dtype=t) for t in [np.int32]*2+[np.float64]*6)
        eb = tuple(np.array([],dtype=t) for t in [np.int32]*3+[np.float64]*6)
        return ec, eb

    if HAS_NUMBA:
        r = _find_contacts_bridges(pos, sp, q, types, n_cells, pi, pj, np_, max_bg)
        return r[:8], r[8:]
    else:
        return _find_cb_numpy(pos, sp, q, types, n_cells, pi, pj, np_, max_bg)


def _find_cb_numpy(pos, sp, q, types, n_cells, pi, pj, np_, max_bg):
    """NumPy fallback for contact/bridge detection."""
    cl, bl = [], []
    for idx in range(np_):
        i,j = int(pi[idx]),int(pj[idx])
        rij=pos[j]-pos[i]; d=np.linalg.norm(rij)
        if d<1e-6: continue
        dv=rij/d; Ri=_quat_to_rot(q[i]); ri=_support_radius(sp[i],Ri,dv)
        Rj=_quat_to_rot(q[j]); rj=_support_radius(sp[j],Rj,-dv)
        ov=ri+rj-d
        if ov>0: cl.append((i,j,ov,dv[0],dv[1],dv[2],ri,rj))
        if types[i]==0 and types[j]==0:
            gap=d-ri-rj
            if gap<max_bg:
                pr=max(0,1-max(0,gap)/max_bg)
                nb=max(1,int(np.sqrt(n_cells[i]*n_cells[j])*pr))
                bl.append((i,j,nb,d,dv[0],dv[1],dv[2],ri,rj))
    def pk(lst, dts):
        if not lst: return tuple(np.array([],dtype=t) for t in dts)
        return tuple(np.array(c,dtype=t) for c,t in zip(zip(*lst),dts))
    return pk(cl,[np.int32]*2+[np.float64]*6), pk(bl,[np.int32]*3+[np.float64]*6)


# =============================================================================
# FORCE DISPATCH
# =============================================================================

def compute_all_forces(nt, pos, vel, sp, q, types, n_cells, dom, contacts, bridges, cfg):
    kr = cfg["mechanics"]["repulsion_stiffness"]
    kw = cfg["mechanics"]["wall_stiffness"]
    da = cfg["mechanics"]["damping"]
    kc = cfg["cell_properties"]["force_per_cell_nN"]*0.03
    cd = cfg["cell_properties"]["diameter_um"]; mf = 100.0
    ci,cj,co,cnx,cny,cnz,cri,crj = contacts
    bi,bj,bnc,bd,bdx,bdy,bdz,bri_,brj = bridges

    if HAS_NUMBA:
        f = _compute_forces(nt, pos, vel, sp, q, types, n_cells, dom,
                            ci,cj,co,cnx,cny,cnz,cri,crj,len(ci),
                            bi,bj,bnc,bd,bdx,bdy,bdz,bri_,brj,len(bi),
                            kr,kw,da,kc,cd,mf)
    else:
        f = _forces_numpy(nt,pos,vel,sp,q,types,dom,contacts,bridges,kr,kw,da,kc,cd,mf)

    # Activity noise on functional
    func = types == 0
    noise = 0.03*np.sqrt(n_cells[func]+1)
    f[func] += noise[:,np.newaxis]*np.random.randn(np.sum(func),3)
    return f


def _forces_numpy(n,pos,vel,sp,q,types,dom,con,bri,kr,kw,da,kc,cd,mf):
    """NumPy fallback."""
    f=np.zeros((n,3)); ci,cj,co,cnx,cny,cnz,cri,crj=con
    for idx in range(len(ci)):
        i,j=int(ci[idx]),int(cj[idx]);ov=co[idx]
        nm=np.array([cnx[idx],cny[idx],cnz[idx]])
        ri,rj=cri[idx],crj[idx]; re=np.sqrt(ri*rj)
        Fm=min(kr*re/50*ov*(1+2*ov/max(re,1)),mf)
        vn=np.dot(vel[j]-vel[i],nm); Ft=max(0,Fm-da*0.5*vn)
        f[i]-=Ft*nm; f[j]+=Ft*nm
    bi,bj,bnc,bd,bdx,bdy,bdz,bri_,brj=bri
    for idx in range(len(bi)):
        i,j=int(bi[idx]),int(bj[idx])
        gap=bd[idx]-bri_[idx]-brj[idx]; ext=gap-cd
        F=kc*bnc[idx]*ext
        if ext<0: F*=0.1
        F=np.clip(F,-mf/2,mf/2); d=np.array([bdx[idx],bdy[idx],bdz[idx]])
        f[i]+=F*d; f[j]-=F*d
    for i in range(n):
        Ri=_quat_to_rot(q[i])
        for ax in range(3):
            dl=np.zeros(3);dl[ax]=1;rl=_support_radius(sp[i],Ri,dl)
            if pos[i,ax]<rl:ov=rl-pos[i,ax];f[i,ax]+=min(kw*ov*(1+ov/max(rl,1)),mf)
            dh=np.zeros(3);dh[ax]=-1;rh=_support_radius(sp[i],Ri,dh)
            if pos[i,ax]>dom[ax]-rh:ov=pos[i,ax]-(dom[ax]-rh);f[i,ax]-=min(kw*ov*(1+ov/max(rh,1)),mf)
    fm=np.linalg.norm(f,axis=1,keepdims=True);big=(fm>mf).flatten()
    if np.any(big):f[big]*=mf/fm[big]
    return f


# =============================================================================
# PHASE FIELD RENDERING
# =============================================================================

def render_slice(pos, sp, q, types, dom, axis=2, frac=0.5, res=150):
    slice_pos = dom[axis]*frac
    if axis==2: d1,d2=0,1
    elif axis==1: d1,d2=0,2
    else: d1,d2=1,2

    if HAS_NUMBA:
        pf, pi = _render_slice_kernel(pos, sp, q, types, dom[d1], dom[d2],
                                       slice_pos, axis, d1, d2, res)
    else:
        pf, pi = _render_slice_np(pos, sp, q, types, dom, axis, d1, d2, slice_pos, res)
    return pf, pi, 1.0-pf-pi


def _render_slice_np(pos, sp, q, types, dom, axis, d1, d2, sp_, res):
    dx=dom[d1]/res; dy=dom[d2]/res
    pf=np.zeros((res,res)); pi=np.zeros((res,res))
    for ix in range(res):
        for iy in range(res):
            pt=np.zeros(3); pt[d1]=(ix+0.5)*dx; pt[d2]=(iy+0.5)*dy; pt[axis]=sp_
            for g in range(len(pos)):
                br=max(sp[g,:3])
                if abs(pos[g,axis]-sp_)>br: continue
                dd=pt-pos[g]
                if np.dot(dd,dd)>br*br: continue
                R=_quat_to_rot(q[g]); b=R.T@dd
                v=abs(b[0]/sp[g,0])**sp[g,3]+abs(b[1]/sp[g,1])**sp[g,3]+abs(b[2]/sp[g,2])**sp[g,3]
                if v<=1:
                    if types[g]==0: pf[ix,iy]=1
                    else: pi[ix,iy]=1
                    break
    return pf, pi


# =============================================================================
# METRICS
# =============================================================================

def phase_connectivity(field, ts=0.3):
    if not HAS_LABEL: return 0, 0.0, 0.0
    thr = np.mean(field)+ts*np.std(field)
    b = (field>thr).astype(int); lab,nc = sp_label(b)
    if nc==0 or b.sum()==0: return 0,0.0,0.0
    sz = np.array([np.sum(lab==l) for l in range(1,nc+1)])
    return nc, float(sz.max()/b.sum()), float(b.sum()/b.size)


def compute_metrics(pos, vel, pos0, types, n_cells, dom, cfg, t,
                    contacts, bridges, cell_tracker, rendered=None):
    ci,cj,co = contacts[0],contacts[1],contacts[2]; bi = bridges[0]
    m = {'time_hours':t, 'n_contacts':len(ci), 'n_bridges':len(bi)}
    coord = np.zeros(len(pos))
    for idx in range(len(ci)): coord[int(ci[idx])]+=1; coord[int(cj[idx])]+=1
    m['mean_coordination'] = float(np.mean(coord))
    m['max_overlap_um'] = float(np.max(co)) if len(co)>0 else 0.0
    m['max_velocity'] = float(np.max(np.linalg.norm(vel, axis=1)))
    disp = np.linalg.norm(pos-pos0, axis=1)
    func=types==0; inert=types==1
    m['disp_func'] = float(np.mean(disp[func])) if np.any(func) else 0.0
    m['disp_inert'] = float(np.mean(disp[inert])) if np.any(inert) else 0.0
    ct = cell_tracker; bridging = ct.is_bridging; n_br = int(np.sum(bridging))
    m['n_bridging_cells'] = n_br
    m['mean_stress'] = float(np.mean(ct.stress[bridging])) if n_br>0 else 0.0
    m['max_stress'] = float(np.max(ct.stress[bridging])) if n_br>0 else 0.0
    m['mean_cell_ar'] = float(np.mean(ct.aspect_ratio[bridging])) if n_br>0 else 1.0
    if rendered is not None:
        pf,pi,pv = rendered
        fn,fl,fc = phase_connectivity(pf); m.update(func_nc=fn,func_lf=fl,func_cov=fc)
        vn,vl,vc = phase_connectivity(pv); m.update(void_nc=vn,void_lf=vl,void_cov=vc)
        m['tissue_frac'] = float(np.mean(pf>0.5))
        if HAS_LABEL:
            thr=np.mean(pf)+0.3*np.std(pf); lab,nc=sp_label((pf>thr).astype(int))
            dx=dom[0]/pf.shape[0]
            m['func_max_area'] = float(np.max([np.sum(lab==l) for l in range(1,nc+1)])*dx**2) if nc>0 else 0.0
        else: m['func_max_area']=0.0
    return m


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    def __init__(self, cfg, cfg_path):
        self.cfg = cfg; self.name = Path(cfg_path).stem
        self.od = Path(cfg["output"]["base_directory"])/self.name
        self.od.mkdir(parents=True, exist_ok=True)
        with open(self.od/f"{self.name}_config.json",'w') as f: json.dump(cfg,f,indent=2)
        self.ds = cfg["domain"]["side_length_um"]; self.dom = np.array([self.ds]*3)
        self.nf, self.ni = calc_granule_counts(cfg); self.nt = self.nf+self.ni
        print(f"\n  {self.name}: {self.ds:.0f}³µm, {self.nf}f+{self.ni}i={self.nt}")
        self.pos=self.vel=self.pos0=None; self.shapes=[]; self.orientations=[]
        self.types=self.n_cells=self.drag=self.sp=self.q=None; self.cell_tracker=None
        self.t=0.0; self.dt=cfg["time"]["dt_initial_hours"]; self.sc=0
        self.history=[]; self.snapshots=[]

    def setup(self):
        c=self.cfg; n=self.nt; shapes=[]; ty=[]; nc=[]
        for _ in range(self.nf):
            r=np.clip(np.random.normal(c["functional_granules"]["radius_mean_um"],
                c["functional_granules"]["radius_std_um"]),15,c["functional_granules"]["radius_mean_um"]*2)
            sh=create_shape(r,c["functional_granules"]); shapes.append(sh); ty.append(0)
            nc.append(calc_cells(sh.surface_area,c))
        for _ in range(self.ni):
            r=np.clip(np.random.normal(c["inert_granules"]["radius_mean_um"],
                c["inert_granules"]["radius_std_um"]),15,c["inert_granules"]["radius_mean_um"]*2)
            sh=create_shape(r,c["inert_granules"]); shapes.append(sh); ty.append(1); nc.append(0)
        ix=np.random.permutation(n)
        self.shapes=[shapes[i] for i in ix]; self.types=np.array([ty[i] for i in ix],dtype=np.int32)
        self.n_cells=np.array([nc[i] for i in ix],dtype=np.int32)
        self.orientations=[Quaternion.random() for _ in range(n)]
        pk = PackingGenerator(self.ds, c["domain"]["target_packing_fraction"])
        self.pos, self.orientations = pk.generate(n, self.shapes, self.orientations)
        self.pos0=self.pos.copy(); self.vel=np.zeros((n,3))
        self.sp=np.array([s.params for s in self.shapes])
        self.q=np.array([o.q for o in self.orientations])
        self.drag=np.array([c["mechanics"]["damping"]*s.eq_radius/50 for s in self.shapes])
        self.cell_tracker=CellTracker(n,self.n_cells,self.shapes,c)
        print(f"  Cells: {self.cell_tracker.total_cells}, "
              f"mean R: {np.mean([s.eq_radius for s in self.shapes]):.1f}µm")

    def _sync(self):
        self.sp = np.array([s.params for s in self.shapes])
        self.q = np.array([o.q for o in self.orientations])

    def step_forward(self):
        self._sync()
        con, bri = find_contacts_bridges(self.pos, self.sp, self.q, self.types,
                                          self.n_cells, self.cfg["cell_properties"]["max_bridge_gap_um"])
        self.cell_tracker.update(self.pos, self.orientations, bri)
        F = compute_all_forces(self.nt, self.pos, self.vel, self.sp, self.q,
                               self.types, self.n_cells, self.dom, con, bri, self.cfg)
        vel = F / self.drag[:,np.newaxis]
        vm = np.linalg.norm(vel, axis=1, keepdims=True)
        fast = (vm>100).flatten()
        if np.any(fast): vel[fast]*=100/vm[fast]
        mv = float(np.max(vm))
        if mv>1e-10:
            mr=min(s.eq_radius for s in self.shapes)
            self.dt=np.clip(0.03*mr/mv, self.cfg["time"]["dt_min_hours"],
                            self.cfg["time"]["dt_max_hours"])
        else: self.dt=self.cfg["time"]["dt_max_hours"]
        self.pos += vel*self.dt; self.vel = vel
        br=np.array([s.bounding_radius for s in self.shapes])
        for d in range(3): self.pos[:,d]=np.clip(self.pos[:,d],br+1,self.dom[d]-br-1)
        self.t+=self.dt; self.sc+=1
        return con, bri

    def save_frame(self, label=None):
        if label is None: label=f"t{self.t:.1f}h"
        fn=self.od/f"{self.name}_frame_{label}.json"
        data={'time_hours':self.t,'config_name':self.name,'n_granules':self.nt,
              'domain':self.dom.tolist(),'positions':self.pos.tolist(),
              'orientations':[q.q.tolist() for q in self.orientations],
              'shapes':[s.to_dict() for s in self.shapes],
              'types':self.types.tolist(),'n_cells':self.n_cells.tolist(),
              'cell_data':self.cell_tracker.get_frame_data()}
        with open(fn,'w') as f: json.dump(data,f)

    def run(self):
        if self.pos is None: self.setup()
        cfg=self.cfg; t_total=cfg["time"]["total_hours"]; si=cfg["time"]["save_interval_hours"]
        res=cfg["output"].get("render_resolution",150)
        print(f"\n  Running {t_total:.0f}h...")
        t0=timer.time(); last_save=0.0; last_rec=0.0

        # Warmup Numba on first call
        if HAS_NUMBA:
            print("  Warming up JIT kernels...")
            self._sync()
            _ = find_contacts_bridges(self.pos, self.sp, self.q, self.types,
                                       self.n_cells, cfg["cell_properties"]["max_bridge_gap_um"])
            print("  JIT ready.")

        self.save_frame("initial")
        pf,pi,pv = render_slice(self.pos, self.sp, self.q, self.types, self.dom, res=res)
        c0,b0 = find_contacts_bridges(self.pos, self.sp, self.q, self.types,
                                       self.n_cells, cfg["cell_properties"]["max_bridge_gap_um"])
        m = compute_metrics(self.pos,self.vel,self.pos0,self.types,self.n_cells,self.dom,
                            cfg,0,c0,b0,self.cell_tracker,(pf,pi,pv))
        self.history.append(m)
        self.snapshots.append((pf.copy(),pi.copy(),pv.copy(),self.pos.copy(),self.types.copy()))

        hdr=(f"  {'t':>6} {'cont':>5} {'brdg':>5} {'f_cl':>5} {'v_cl':>5} "
             f"{'disp':>6} {'br_c':>5} {'σ':>6}")
        print(hdr)
        print(f"  {0:6.1f} {m['n_contacts']:5d} {m['n_bridges']:5d} "
              f"{m.get('func_nc',0):5d} {m.get('void_nc',0):5d} "
              f"{m['disp_func']:6.1f} {m['n_bridging_cells']:5d} {m['mean_stress']:6.2f}")

        if HAS_TQDM: pb=tqdm(total=t_total,desc="  Sim",unit="hr"); pt=0

        while self.t < t_total:
            con, bri = self.step_forward()
            if self.t - last_rec >= 0.2:
                do_render = (self.t - last_save >= si)
                rendered = None
                if do_render:
                    self._sync()
                    rendered = render_slice(self.pos,self.sp,self.q,self.types,self.dom,res=res)
                    self.snapshots.append((rendered[0].copy(),rendered[1].copy(),
                                           rendered[2].copy(),self.pos.copy(),self.types.copy()))
                    self.save_frame(); last_save=self.t
                m = compute_metrics(self.pos,self.vel,self.pos0,self.types,self.n_cells,
                                    self.dom,cfg,self.t,con,bri,self.cell_tracker,rendered)
                self.history.append(m); last_rec=self.t
                if do_render:
                    print(f"  {self.t:6.1f} {m['n_contacts']:5d} {m['n_bridges']:5d} "
                          f"{m.get('func_nc',0):5d} {m.get('void_nc',0):5d} "
                          f"{m['disp_func']:6.1f} {m['n_bridging_cells']:5d} {m['mean_stress']:6.2f}")
            if HAS_TQDM: pb.update(self.t-pt); pt=self.t

        if HAS_TQDM: pb.close()
        self.save_frame("final")
        with open(self.od/f"{self.name}_history.json",'w') as f: json.dump(self.history,f)
        el=timer.time()-t0
        print(f"\n  Done: {el:.1f}s ({self.sc} steps) → {self.od}")
        return self.history, self.snapshots


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*65)
    print("  Overdamped Langevin — Numba-Accelerated 3D Superellipsoid")
    print("="*65)
    if len(sys.argv)>1: cp=sys.argv[1]
    else: cp=select_config()
    if not cp: print("  No config."); return
    print(f"  Config: {cp}")
    cfg=load_config(cp); sim=Simulation(cfg,cp)
    hist, snaps = sim.run()

    h0,hf=hist[0],hist[-1]
    print("\n"+"="*65)
    print("  SUMMARY")
    print("="*65)
    if 'func_nc' in hf:
        ft='CONTINUOUS' if hf.get('func_lf',0)>.8 else 'FEW CLUSTERS' if hf.get('func_nc',99)<6 else 'ISLANDS'
        vt='CONTINUOUS' if hf.get('void_lf',0)>.8 else 'POCKETS' if hf.get('void_nc',99)<6 else 'MANY POCKETS'
        print(f"  Func: {h0.get('func_nc','?')}→{hf['func_nc']} ({ft})")
        print(f"  Void: {h0.get('void_nc','?')}→{hf['void_nc']} ({vt})")
    print(f"  Disp: f={hf['disp_func']:.1f}µm i={hf['disp_inert']:.1f}µm")
    print(f"  Bridges: {hf['n_bridging_cells']}/{sim.cell_tracker.total_cells} cells bridging")
    print(f"  Stress: {hf['mean_stress']:.2f} nN mean")

if __name__ == "__main__": main()