"""
Coarse-Grained DEM Simulation — Superellipsoid Contacts
========================================================

Key physics:
  - Contact detection uses true superellipsoid geometry: |x/a|^n + |y/b|^n + |z/c|^n = 1
  - Packing fraction measured from voxelization (border-excluded), not volume ratio
  - Direction-dependent effective radius for all force calculations

Performance: Numba JIT with automatic NumPy fallback.

Usage:
    python dem_simulation.py config.json
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json
import sys
import time as time_module

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from numba import njit, prange, float64, int32, int64
    HAS_NUMBA = True
    print("[Accel] Numba detected — JIT-compiling superellipsoid kernels")
except ImportError:
    HAS_NUMBA = False
    print("[Accel] Numba not found — using NumPy fallback")

from multiprocessing import cpu_count
N_WORKERS = max(1, cpu_count() - 1)


# =============================================================================
# NUMBA KERNELS — superellipsoid geometry
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _quat_to_matrix(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = np.empty((3, 3), dtype=float64)
        R[0,0]=1-2*y*y-2*z*z; R[0,1]=2*x*y-2*w*z;     R[0,2]=2*x*z+2*w*y
        R[1,0]=2*x*y+2*w*z;   R[1,1]=1-2*x*x-2*z*z;   R[1,2]=2*y*z-2*w*x
        R[2,0]=2*x*z-2*w*y;   R[2,1]=2*y*z+2*w*x;      R[2,2]=1-2*x*x-2*y*y
        return R

    @njit(cache=True)
    def _se_radius(sp, R, d):
        """
        Superellipsoid effective radius along world-frame direction d.
        sp = [a, b, c, n_exp].
        For |x/a|^n + |y/b|^n + |z/c|^n = 1, the surface point along
        body-frame direction d_b is at distance:
            r = 1 / (|db0/a|^n + |db1/b|^n + |db2/c|^n)^(1/n)
        """
        a, b, c, n = sp[0], sp[1], sp[2], sp[3]
        # R^T @ d  (body frame direction)
        db0 = R[0,0]*d[0] + R[1,0]*d[1] + R[2,0]*d[2]
        db1 = R[0,1]*d[0] + R[1,1]*d[1] + R[2,1]*d[2]
        db2 = R[0,2]*d[0] + R[1,2]*d[1] + R[2,2]*d[2]
        val = abs(db0/a)**n + abs(db1/b)**n + abs(db2/c)**n
        if val < 1e-30:
            return max(a, max(b, c))
        return 1.0 / val**(1.0/n)

    @njit(cache=True)
    def _bounding_radius(sp):
        return max(sp[0], max(sp[1], sp[2]))

    @njit(cache=True)
    def _point_inside_se(px, py, pz, pos, sp, R):
        """Check if world point (px,py,pz) is inside superellipsoid."""
        rx = px - pos[0]; ry = py - pos[1]; rz = pz - pos[2]
        bx = R[0,0]*rx + R[1,0]*ry + R[2,0]*rz
        by = R[0,1]*rx + R[1,1]*ry + R[2,1]*rz
        bz = R[0,2]*rx + R[1,2]*ry + R[2,2]*rz
        a, b, c, n = sp[0], sp[1], sp[2], sp[3]
        return abs(bx/a)**n + abs(by/b)**n + abs(bz/c)**n <= 1.0

    @njit(parallel=True, cache=True)
    def _packing_forces_numba(positions, shape_params, quats, domain, scale,
                              stiffness, max_force):
        n = positions.shape[0]
        forces = np.zeros((n, 3), dtype=float64)
        max_overlap = 0.0
        for i in prange(n):
            Ri = _quat_to_matrix(quats[i])
            sp_i = shape_params[i].copy(); sp_i[0]*=scale; sp_i[1]*=scale; sp_i[2]*=scale
            rb_i = _bounding_radius(sp_i)
            for j in range(i+1, n):
                dx=positions[j,0]-positions[i,0]
                dy=positions[j,1]-positions[i,1]
                dz=positions[j,2]-positions[i,2]
                dist=np.sqrt(dx*dx+dy*dy+dz*dz)
                if dist < 1e-6: continue
                sp_j = shape_params[j].copy(); sp_j[0]*=scale; sp_j[1]*=scale; sp_j[2]*=scale
                rb_j = _bounding_radius(sp_j)
                if dist > rb_i + rb_j: continue
                inv_d = 1.0/dist
                d_vec = np.array([dx*inv_d, dy*inv_d, dz*inv_d])
                neg_d = np.array([-d_vec[0], -d_vec[1], -d_vec[2]])
                ri = _se_radius(sp_i, Ri, d_vec)
                Rj = _quat_to_matrix(quats[j])
                rj = _se_radius(sp_j, Rj, neg_d)
                overlap = ri + rj - dist
                if overlap > 0:
                    if overlap > max_overlap: max_overlap = overlap
                    F = min(stiffness * overlap, max_force)
                    for k in range(3):
                        forces[i,k] -= F*d_vec[k]
                        forces[j,k] += F*d_vec[k]
        return forces, max_overlap

    @njit(parallel=True, cache=True)
    def _wall_forces_numba(positions, shape_params, quats, domain, scale,
                           stiffness, max_force):
        n = positions.shape[0]
        forces = np.zeros((n, 3), dtype=float64)
        for i in prange(n):
            Ri = _quat_to_matrix(quats[i])
            sp_i = shape_params[i].copy(); sp_i[0]*=scale; sp_i[1]*=scale; sp_i[2]*=scale
            for d in range(3):
                d_lo = np.zeros(3, dtype=float64); d_lo[d] = 1.0
                r_lo = _se_radius(sp_i, Ri, d_lo)
                if positions[i,d] < r_lo:
                    ov = r_lo - positions[i,d]
                    forces[i,d] += min(stiffness*ov, max_force)
                d_hi = np.zeros(3, dtype=float64); d_hi[d] = -1.0
                r_hi = _se_radius(sp_i, Ri, d_hi)
                if positions[i,d] > domain[d] - r_hi:
                    ov = positions[i,d] - (domain[d] - r_hi)
                    forces[i,d] -= min(stiffness*ov, max_force)
        return forces

    @njit(cache=True)
    def _find_contacts_numba(positions, shape_params, quats, types, n_cells,
                             pair_i, pair_j, n_pairs, max_bridge_gap):
        c_i=np.empty(n_pairs,dtype=int32); c_j=np.empty(n_pairs,dtype=int32)
        c_ov=np.empty(n_pairs,dtype=float64)
        c_nx=np.empty(n_pairs,dtype=float64); c_ny=np.empty(n_pairs,dtype=float64)
        c_nz=np.empty(n_pairs,dtype=float64)
        c_ri=np.empty(n_pairs,dtype=float64); c_rj=np.empty(n_pairs,dtype=float64)
        nc=0
        b_i=np.empty(n_pairs,dtype=int32); b_j=np.empty(n_pairs,dtype=int32)
        b_nc=np.empty(n_pairs,dtype=int32)
        b_dist=np.empty(n_pairs,dtype=float64)
        b_dx=np.empty(n_pairs,dtype=float64); b_dy=np.empty(n_pairs,dtype=float64)
        b_dz=np.empty(n_pairs,dtype=float64)
        b_ri=np.empty(n_pairs,dtype=float64); b_rj=np.empty(n_pairs,dtype=float64)
        nb=0
        for idx in range(n_pairs):
            i=pair_i[idx]; j=pair_j[idx]
            dx=positions[j,0]-positions[i,0]
            dy=positions[j,1]-positions[i,1]
            dz=positions[j,2]-positions[i,2]
            dist=np.sqrt(dx*dx+dy*dy+dz*dz)
            if dist < 1e-6: continue
            inv_d=1.0/dist
            d_vec=np.array([dx*inv_d,dy*inv_d,dz*inv_d])
            neg_d=np.array([-d_vec[0],-d_vec[1],-d_vec[2]])
            Ri=_quat_to_matrix(quats[i])
            ri=_se_radius(shape_params[i], Ri, d_vec)
            Rj=_quat_to_matrix(quats[j])
            rj=_se_radius(shape_params[j], Rj, neg_d)
            overlap=ri+rj-dist
            if overlap > 0:
                c_i[nc]=i; c_j[nc]=j; c_ov[nc]=overlap
                c_nx[nc]=d_vec[0]; c_ny[nc]=d_vec[1]; c_nz[nc]=d_vec[2]
                c_ri[nc]=ri; c_rj[nc]=rj; nc+=1
            if types[i]==0 and types[j]==0:
                gap=dist-ri-rj
                if gap < max_bridge_gap:
                    prox=max(0.0, 1.0-max(0.0,gap)/max_bridge_gap)
                    n_br=max(1, int(np.sqrt(float(n_cells[i])*float(n_cells[j]))*prox))
                    b_i[nb]=i; b_j[nb]=j; b_nc[nb]=n_br; b_dist[nb]=dist
                    b_dx[nb]=d_vec[0]; b_dy[nb]=d_vec[1]; b_dz[nb]=d_vec[2]
                    b_ri[nb]=ri; b_rj[nb]=rj; nb+=1
        return (c_i[:nc],c_j[:nc],c_ov[:nc],c_nx[:nc],c_ny[:nc],c_nz[:nc],c_ri[:nc],c_rj[:nc],
                b_i[:nb],b_j[:nb],b_nc[:nb],b_dist[:nb],b_dx[:nb],b_dy[:nb],b_dz[:nb],b_ri[:nb],b_rj[:nb])

    @njit(parallel=True, cache=True)
    def _compute_forces_numba(
        n_g, positions, velocities, shape_params, quats, types, n_cells_arr, domain,
        c_i,c_j,c_ov,c_nx,c_ny,c_nz,c_ri,c_rj,n_c,
        b_i,b_j,b_nc,b_dist,b_dx,b_dy,b_dz,b_ri,b_rj,n_b,
        k_rep,k_wall,damping,k_cell,cell_diam,max_force):
        forces=np.zeros((n_g,3),dtype=float64)
        for idx in range(n_c):
            i=c_i[idx]; j=c_j[idx]; ov=c_ov[idx]
            nx,ny,nz=c_nx[idx],c_ny[idx],c_nz[idx]
            ri,rj=c_ri[idx],c_rj[idx]; r_eff=np.sqrt(ri*rj)
            F_mag=k_rep*r_eff/50.0*ov*(1.0+2.0*ov/max(r_eff,1.0))
            if F_mag>max_force: F_mag=max_force
            vn=(velocities[j,0]-velocities[i,0])*nx+(velocities[j,1]-velocities[i,1])*ny+(velocities[j,2]-velocities[i,2])*nz
            F_t=F_mag-damping*0.5*vn
            if F_t<0: F_t=0.0
            forces[i,0]-=F_t*nx; forces[i,1]-=F_t*ny; forces[i,2]-=F_t*nz
            forces[j,0]+=F_t*nx; forces[j,1]+=F_t*ny; forces[j,2]+=F_t*nz
        for idx in range(n_b):
            i=b_i[idx]; j=b_j[idx]; gap=b_dist[idx]-b_ri[idx]-b_rj[idx]
            ext=gap-cell_diam; F=k_cell*b_nc[idx]*ext
            if ext<0: F*=0.1
            if F>max_force*0.5: F=max_force*0.5
            elif F<-max_force*0.5: F=-max_force*0.5
            forces[i,0]+=F*b_dx[idx]; forces[i,1]+=F*b_dy[idx]; forces[i,2]+=F*b_dz[idx]
            forces[j,0]-=F*b_dx[idx]; forces[j,1]-=F*b_dy[idx]; forces[j,2]-=F*b_dz[idx]
        for i in prange(n_g):
            Ri=_quat_to_matrix(quats[i])
            for d in range(3):
                d_lo=np.zeros(3,dtype=float64); d_lo[d]=1.0
                r_lo=_se_radius(shape_params[i], Ri, d_lo)
                if positions[i,d]<r_lo:
                    ov=r_lo-positions[i,d]
                    f=k_wall*ov*(1.0+ov/max(r_lo,1.0))
                    if f>max_force: f=max_force
                    forces[i,d]+=f
                d_hi=np.zeros(3,dtype=float64); d_hi[d]=-1.0
                r_hi=_se_radius(shape_params[i], Ri, d_hi)
                if positions[i,d]>domain[d]-r_hi:
                    ov=positions[i,d]-(domain[d]-r_hi)
                    f=k_wall*ov*(1.0+ov/max(r_hi,1.0))
                    if f>max_force: f=max_force
                    forces[i,d]-=f
            fx,fy,fz=forces[i,0],forces[i,1],forces[i,2]
            fm=np.sqrt(fx*fx+fy*fy+fz*fz)
            if fm>max_force:
                s=max_force/fm
                forces[i,0]*=s; forces[i,1]*=s; forces[i,2]*=s
        return forces

    @njit(parallel=True, cache=True)
    def _voxel_packing_numba(positions, shape_params, quats, grid_min, grid_max, res):
        """Count filled voxels inside the border-excluded region."""
        nx=res; ny=res; nz=res
        dx_v=(grid_max[0]-grid_min[0])/nx
        dy_v=(grid_max[1]-grid_min[1])/ny
        dz_v=(grid_max[2]-grid_min[2])/nz
        n_g=positions.shape[0]
        filled=np.zeros(nx, dtype=int64)  # per-slice count for parallel reduction
        for ix in prange(nx):
            px=grid_min[0]+(ix+0.5)*dx_v
            count=0
            for iy in range(ny):
                py=grid_min[1]+(iy+0.5)*dy_v
                for iz in range(nz):
                    pz=grid_min[2]+(iz+0.5)*dz_v
                    inside=False
                    for g in range(n_g):
                        rb=_bounding_radius(shape_params[g])
                        d2=(px-positions[g,0])**2+(py-positions[g,1])**2+(pz-positions[g,2])**2
                        if d2 > rb*rb: continue
                        if _point_inside_se(px,py,pz,positions[g],shape_params[g],
                                           _quat_to_matrix(quats[g])):
                            inside=True; break
                    if inside: count+=1
            filled[ix]=count
        total=nx*ny*nz
        return int(np.sum(filled)), total


# =============================================================================
# NUMPY FALLBACKS
# =============================================================================

def _batch_rot(quats):
    w,x,y,z = quats[:,0],quats[:,1],quats[:,2],quats[:,3]
    R=np.empty((len(quats),3,3))
    R[:,0,0]=1-2*y*y-2*z*z; R[:,0,1]=2*x*y-2*w*z;     R[:,0,2]=2*x*z+2*w*y
    R[:,1,0]=2*x*y+2*w*z;   R[:,1,1]=1-2*x*x-2*z*z;   R[:,1,2]=2*y*z-2*w*x
    R[:,2,0]=2*x*z-2*w*y;   R[:,2,1]=2*y*z+2*w*x;      R[:,2,2]=1-2*x*x-2*y*y
    return R

def _se_radius_np(sp, R, d):
    a,b,c,n = sp
    db = R.T @ d
    val = abs(db[0]/a)**n + abs(db[1]/b)**n + abs(db[2]/c)**n
    if val < 1e-30: return max(a,b,c)
    return 1.0 / val**(1.0/n)

def _voxel_packing_np(positions, shape_params, quats, grid_min, grid_max, res):
    """Numpy fallback for voxel packing (coarser grid)."""
    Rs = _batch_rot(quats)
    n_g = len(positions)
    xs = np.linspace(grid_min[0], grid_max[0], res)
    ys = np.linspace(grid_min[1], grid_max[1], res)
    zs = np.linspace(grid_min[2], grid_max[2], res)
    filled = 0
    total = res**3
    for ix, px in enumerate(xs):
        for iy, py in enumerate(ys):
            for iz, pz in enumerate(zs):
                p = np.array([px, py, pz])
                for g in range(n_g):
                    rb = max(shape_params[g, :3])
                    if np.sum((p - positions[g])**2) > rb*rb:
                        continue
                    rel = p - positions[g]
                    body = Rs[g].T @ rel
                    sp = shape_params[g]
                    val = abs(body[0]/sp[0])**sp[3] + abs(body[1]/sp[1])**sp[3] + abs(body[2]/sp[2])**sp[3]
                    if val <= 1.0:
                        filled += 1; break
    return filled, total


# =============================================================================
# CONFIG
# =============================================================================

def select_config_file():
    if not HAS_TK: return None
    root=tk.Tk(); root.withdraw(); root.attributes('-topmost',True)
    fp=filedialog.askopenfilename(title="Select Config",
        filetypes=[("JSON","*.json"),("All","*.*")], initialdir=".")
    root.destroy()
    return fp if fp else None

def load_config(filepath):
    with open(filepath) as f: config=json.load(f)
    defaults={"simulation_name":"unnamed","domain":{"side_length_um":500.0,"target_packing_fraction":0.55},
        "granule_ratio":{"functional_fraction":0.5},
        "functional_granules":{"radius_mean_um":40.0,"radius_std_um":8.0,"aspect_ratio_range":[1.0,1.5],"roundness_range":[2.0,3.0],"roughness_range":[0.1,0.3]},
        "inert_granules":{"radius_mean_um":50.0,"radius_std_um":10.0,"aspect_ratio_range":[1.0,1.3],"roundness_range":[2.0,2.5],"roughness_range":[0.0,0.2]},
        "cell_properties":{"diameter_um":20.0,"attachment_area_fraction":0.5,"force_per_cell_nN":5.0,"max_bridge_gap_um":50.0},
        "mechanics":{"repulsion_stiffness":1.0,"damping":1.5,"wall_stiffness":2.0},
        "time":{"total_hours":72.0,"save_interval_hours":2.0,"dt_initial_hours":0.01,"dt_min_hours":0.001,"dt_max_hours":0.5},
        "output":{"base_directory":"./simulations","save_true_shapes":True}}
    def merge(c,d):
        for k,v in d.items():
            if k not in c: c[k]=v
            elif isinstance(v,dict) and isinstance(c[k],dict): merge(c[k],v)
    merge(config,defaults)
    return config

def calc_granule_counts(config):
    s=config["domain"]["side_length_um"]; phi=config["domain"]["target_packing_fraction"]
    ff=config["granule_ratio"]["functional_fraction"]; V=s**3*phi
    rf=config["functional_granules"]["radius_mean_um"]
    ri=config["inert_granules"]["radius_mean_um"]
    return max(1,int(np.round(V*ff/((4/3)*np.pi*rf**3)))), max(1,int(np.round(V*(1-ff)/((4/3)*np.pi*ri**3))))

def calc_cells(area, config):
    cd=config["cell_properties"]["diameter_um"]
    af=config["cell_properties"]["attachment_area_fraction"]
    return max(1, int(np.ceil(area*af/(np.pi*(cd/2)**2))))


# =============================================================================
# SHAPES
# =============================================================================

class GranuleType(Enum):
    FUNCTIONAL=0; INERT=1

@dataclass
class GranuleShape:
    a:float; b:float; c:float; n:float=2.0; roughness:float=0.0
    @property
    def equivalent_radius(self): return (self.a*self.b*self.c)**(1/3)
    @property
    def volume(self): return (4/3)*np.pi*self.a*self.b*self.c
    @property
    def surface_area(self):
        p=1.6075; ap,bp,cp=self.a**p,self.b**p,self.c**p
        return 4*np.pi*((ap*bp+ap*cp+bp*cp)/3)**(1/p)
    @property
    def bounding_radius(self): return max(self.a,self.b,self.c)
    @property
    def params(self): return np.array([self.a,self.b,self.c,self.n])
    def to_dict(self): return {"a":self.a,"b":self.b,"c":self.c,"n":self.n,"roughness":self.roughness}

@dataclass
class TrueShape:
    a:float; b:float; c:float; n:float; roughness:float
    surface_points:Optional[np.ndarray]=None
    def generate_surface_mesh(self, resolution=20):
        th=np.linspace(0,2*np.pi,resolution); ph=np.linspace(0,np.pi,resolution//2)
        th,ph=np.meshgrid(th,ph)
        def spow(x,p): return np.sign(x)*np.abs(x)**p
        e=2.0/self.n
        x=self.a*spow(np.cos(th),e)*spow(np.sin(ph),e)
        y=self.b*spow(np.sin(th),e)*spow(np.sin(ph),e)
        z=self.c*spow(np.cos(ph),e)
        if self.roughness>0:
            noise=np.random.randn(*x.shape)*self.roughness*min(self.a,self.b,self.c)*0.1
            r=np.sqrt(x**2+y**2+z**2)
            x+=noise*x/(r+1e-6); y+=noise*y/(r+1e-6); z+=noise*z/(r+1e-6)
        self.surface_points=np.stack([x,y,z],axis=-1); return self.surface_points
    def to_dict(self):
        return {"a":self.a,"b":self.b,"c":self.c,"n":self.n,"roughness":self.roughness}

def create_granule_shapes(mean_r, cfg):
    ar=np.random.uniform(*cfg["aspect_ratio_range"])
    n=np.random.uniform(*cfg["roundness_range"])
    rough=np.random.uniform(*cfg["roughness_range"])
    c=mean_r*ar**(1/3); ab=mean_r/ar**(1/6)
    asym=np.random.uniform(0.95,1.05); a=ab*asym; b=ab/asym
    tv=(4/3)*np.pi*mean_r**3; cv=(4/3)*np.pi*a*b*c
    if cv>0: s=(tv/cv)**(1/3); a*=s; b*=s; c*=s
    return GranuleShape(a,b,c,n,rough), TrueShape(a,b,c,n,rough)


# =============================================================================
# QUATERNION
# =============================================================================

class Quaternion:
    __slots__=['q']
    def __init__(self,w=1.0,x=0.0,y=0.0,z=0.0):
        self.q=np.array([w,x,y,z],dtype=np.float64)
        nm=np.linalg.norm(self.q)
        if nm>1e-10: self.q/=nm
    @classmethod
    def random(cls):
        u=np.random.random(3)
        return cls(np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                   np.sqrt(u[0])*np.sin(2*np.pi*u[2]),np.sqrt(u[0])*np.cos(2*np.pi*u[2]))
    def to_matrix(self):
        w,x,y,z=self.q
        return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                         [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                         [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])


# =============================================================================
# GRANULE SYSTEM
# =============================================================================

class GranuleSystem:
    def __init__(self, n):
        self.n=n
        self.positions=np.zeros((n,3))
        self.velocities=np.zeros((n,3))
        self.orientations=[Quaternion() for _ in range(n)]
        self.sim_shapes:List[GranuleShape]=[]
        self.true_shapes:List[TrueShape]=[]
        self.types=np.zeros(n,dtype=np.int32)
        self.n_cells=np.zeros(n,dtype=np.int32)
        self.drag_coeffs=np.ones(n)
        self._sp:Optional[np.ndarray]=None  # (n,4) [a,b,c,n_exp]
        self._quats:Optional[np.ndarray]=None

    def sync_arrays(self):
        self._sp=np.array([s.params for s in self.sim_shapes])
        self._quats=np.array([q.q for q in self.orientations])
    @property
    def shape_params(self):
        if self._sp is None: self.sync_arrays()
        return self._sp
    @property
    def quats(self):
        if self._quats is None: self.sync_arrays()
        return self._quats


# =============================================================================
# VOXEL PACKING FRACTION
# =============================================================================

def compute_voxel_packing(positions, shape_params, quats, domain, margin, res=40):
    """
    Compute packing fraction from voxelization, excluding border region.
    margin: distance from each wall to exclude (typically 1 mean granule radius).
    """
    grid_min = np.array([margin, margin, margin], dtype=np.float64)
    grid_max = np.array([domain[0]-margin, domain[1]-margin, domain[2]-margin], dtype=np.float64)
    if np.any(grid_max <= grid_min):
        return 0.0  # domain too small for margin
    if HAS_NUMBA:
        filled, total = _voxel_packing_numba(positions, shape_params, quats,
                                              grid_min, grid_max, res)
    else:
        filled, total = _voxel_packing_np(positions, shape_params, quats,
                                           grid_min, grid_max, max(15, res//2))
    return filled / max(total, 1)


# =============================================================================
# PACKING GENERATOR
# =============================================================================

class PackingGenerator:
    def __init__(self, domain_size, target_phi, verbose=True):
        self.domain=np.array([domain_size]*3)
        self.target_phi=target_phi
        self.verbose=verbose

    def _relax(self, pos, sp, quats, domain, scale, stiff, damp, mf, max_iters, tol):
        max_ov=1e6
        for it in range(max_iters):
            if HAS_NUMBA:
                f, max_ov = _packing_forces_numba(pos, sp, quats, domain, scale, stiff, mf)
                f += _wall_forces_numba(pos, sp, quats, domain, scale, stiff, mf)
            else:
                f = self._forces_np(pos, sp, quats, domain, scale, stiff, mf)
                max_ov = 0
            fm = np.linalg.norm(f, axis=1, keepdims=True)
            big = (fm > mf).flatten()
            if np.any(big): f[big] *= mf / fm[big]
            pos += f * 0.3 / damp
            nan = np.any(np.isnan(pos), axis=1)
            if np.any(nan): pos[nan] = domain/2 + np.random.randn(int(np.sum(nan)),3)*10
            br = np.max(sp[:,:3], axis=1) * scale
            for d in range(3): pos[:,d] = np.clip(pos[:,d], br+0.5, domain[d]-br-0.5)
            if max_ov < tol: break
        return max_ov

    def _forces_np(self, pos, sp, quats, domain, scale, stiff, mf):
        n = len(pos); forces = np.zeros((n,3)); Rs = _batch_rot(quats)
        for i in range(n):
            sp_i = sp[i].copy(); sp_i[:3]*=scale; rb_i = max(sp_i[:3])
            for j in range(i+1,n):
                rij=pos[j]-pos[i]; dist=np.linalg.norm(rij)
                if dist<1e-6: continue
                sp_j=sp[j].copy(); sp_j[:3]*=scale
                if dist>rb_i+max(sp_j[:3]): continue
                d=rij/dist
                ri=_se_radius_np(sp_i,Rs[i],d); rj=_se_radius_np(sp_j,Rs[j],-d)
                ov=ri+rj-dist
                if ov>0:
                    F=min(stiff*ov,mf)*d; forces[i]-=F; forces[j]+=F
        for i in range(n):
            sp_i=sp[i].copy(); sp_i[:3]*=scale
            for ax in range(3):
                d_lo=np.zeros(3); d_lo[ax]=1.0
                r_lo=_se_radius_np(sp_i,Rs[i],d_lo)
                if pos[i,ax]<r_lo: forces[i,ax]+=min(stiff*(r_lo-pos[i,ax]),mf)
                d_hi=np.zeros(3); d_hi[ax]=-1.0
                r_hi=_se_radius_np(sp_i,Rs[i],d_hi)
                if pos[i,ax]>domain[ax]-r_hi: forces[i,ax]-=min(stiff*(pos[i,ax]-(domain[ax]-r_hi)),mf)
        return forces

    def generate(self, n, shapes, orientations):
        if self.verbose:
            print(f"\nGenerating jammed packing for {n} superellipsoid granules...")
        pos=np.zeros((n,3))
        ns=int(np.ceil(n**(1/3))); sp_grid=min(self.domain)/(ns+1)
        idx=0
        for ix in range(ns):
            for iy in range(ns):
                for iz in range(ns):
                    if idx>=n: break
                    pos[idx]=[(ix+1)*sp_grid+np.random.uniform(-sp_grid*0.1,sp_grid*0.1),
                              (iy+1)*sp_grid+np.random.uniform(-sp_grid*0.1,sp_grid*0.1),
                              (iz+1)*sp_grid+np.random.uniform(-sp_grid*0.1,sp_grid*0.1)]
                    pos[idx]=np.clip(pos[idx],20,self.domain-20); idx+=1
                if idx>=n: break
            if idx>=n: break

        sp=np.array([s.params for s in shapes])  # (n,4) unscaled
        quats=np.array([q.q for q in orientations])
        domain=self.domain.copy()

        scale=0.05; base_gr=0.0005; stiff=1.0; damp=2.0; mf=50.0; cur_phi=0.0
        relax_every=200; relax_sub=50

        if self.verbose and HAS_TQDM:
            pbar=tqdm(total=self.target_phi,desc="Growing",unit="φ"); last_phi=0.0

        for it in range(400000):
            if cur_phi>=self.target_phi: break
            prog=cur_phi/max(self.target_phi,1e-6)
            if prog<0.3: gr=base_gr
            elif prog<0.6: gr=base_gr*0.5
            elif prog<0.8: gr=base_gr*0.1
            else: gr=base_gr*0.02

            if HAS_NUMBA:
                f,_=_packing_forces_numba(pos,sp,quats,domain,scale,stiff,mf)
                f+=_wall_forces_numba(pos,sp,quats,domain,scale,stiff,mf)
            else:
                f=self._forces_np(pos,sp,quats,domain,scale,stiff,mf)
            fm=np.linalg.norm(f,axis=1,keepdims=True)
            big=(fm>mf).flatten()
            if np.any(big): f[big]*=mf/fm[big]
            pos+=f*0.4/damp
            nan=np.any(np.isnan(pos),axis=1)
            if np.any(nan): pos[nan]=domain/2+np.random.randn(int(np.sum(nan)),3)*10
            br=np.max(sp[:,:3],axis=1)*scale
            for d in range(3): pos[:,d]=np.clip(pos[:,d],br+1,domain[d]-br-1)
            scale+=gr
            vols=(4/3)*np.pi*sp[:,0]*sp[:,1]*sp[:,2]*scale**3
            cur_phi=np.sum(vols)/np.prod(domain)

            if it%relax_every==0 and it>0:
                min_r=np.min(sp[:,:3])*scale
                self._relax(pos,sp,quats,domain,scale,stiff*2,damp,mf,relax_sub,min_r*0.3)

            if self.verbose and HAS_TQDM and cur_phi-last_phi>0.005:
                pbar.update(cur_phi-last_phi); last_phi=cur_phi

        if self.verbose and HAS_TQDM: pbar.close()

        min_r=np.min(sp[:,:3])*scale; tol=min_r*0.05
        if self.verbose: print(f"Final relaxation (tol={tol:.2f} μm)...")
        ov=self._relax(pos,sp,quats,domain,scale,stiff*4,damp,mf*2,10000,tol)
        if self.verbose: print(f"  Phase 1: max overlap={ov:.3f}")
        if ov>tol:
            ov=self._relax(pos,sp,quats,domain,scale,stiff*16,damp*0.5,mf*4,10000,tol)
            if self.verbose: print(f"  Phase 2: max overlap={ov:.3f}")
        if ov>tol:
            ov=self._relax(pos,sp,quats,domain,scale,stiff*64,damp*0.25,mf*8,20000,tol)
            if self.verbose: print(f"  Phase 3: max overlap={ov:.3f}")
        if ov>tol and self.verbose:
            print(f"  WARNING: residual overlap {ov:.3f} > tol {tol:.3f}")

        for s in shapes: s.a*=scale; s.b*=scale; s.c*=scale
        if self.verbose: print(f"  Final analytic φ: {cur_phi:.3f}")
        return pos, orientations


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    def __init__(self, config, config_filepath):
        self.config=config; self.config_name=Path(config_filepath).stem
        base=Path(config["output"]["base_directory"])
        self.output_dir=base/self.config_name; self.output_dir.mkdir(parents=True,exist_ok=True)
        with open(self.output_dir/f"{self.config_name}_config.json",'w') as f:
            json.dump(config,f,indent=2)
        self.domain_size=config["domain"]["side_length_um"]
        self.domain=np.array([self.domain_size]*3)
        self.n_functional,self.n_inert=calc_granule_counts(config)
        self.n_total=self.n_functional+self.n_inert
        print(f"\nSimulation: {self.config_name}")
        print(f"  Domain: {self.domain_size:.0f}³ μm")
        print(f"  Granules: {self.n_functional} func + {self.n_inert} inert = {self.n_total}")
        self.granules:Optional[GranuleSystem]=None
        self.time=0.0; self.step_count=0
        self.dt=config["time"]["dt_initial_hours"]
        self.mean_granule_radius = 0.0  # set in setup
        self.voxel_res = 40 if HAS_NUMBA else 20
        self.history={'time_hours':[],'n_contacts':[],'n_bridges':[],
                      'mean_coordination':[],'max_velocity':[],
                      'packing_fraction':[],'max_overlap_um':[]}

    def setup(self):
        cfg=self.config; g=GranuleSystem(self.n_total); self.granules=g
        all_s,all_t,all_ty,all_nc=[],[],[],[]
        for _ in range(self.n_functional):
            r=np.clip(np.random.normal(cfg["functional_granules"]["radius_mean_um"],
                cfg["functional_granules"]["radius_std_um"]),15.0,cfg["functional_granules"]["radius_mean_um"]*2)
            ss,ts=create_granule_shapes(r,cfg["functional_granules"])
            all_s.append(ss); all_t.append(ts); all_ty.append(0)
            all_nc.append(calc_cells(ss.surface_area,cfg))
        for _ in range(self.n_inert):
            r=np.clip(np.random.normal(cfg["inert_granules"]["radius_mean_um"],
                cfg["inert_granules"]["radius_std_um"]),15.0,cfg["inert_granules"]["radius_mean_um"]*2)
            ss,ts=create_granule_shapes(r,cfg["inert_granules"])
            all_s.append(ss); all_t.append(ts); all_ty.append(1); all_nc.append(0)
        idx=np.random.permutation(self.n_total)
        g.sim_shapes=[all_s[i] for i in idx]; g.true_shapes=[all_t[i] for i in idx]
        g.types=np.array([all_ty[i] for i in idx],dtype=np.int32)
        g.n_cells=np.array([all_nc[i] for i in idx],dtype=np.int32)
        g.orientations=[Quaternion.random() for _ in range(self.n_total)]

        packer=PackingGenerator(self.domain_size,cfg["domain"]["target_packing_fraction"])
        pos,orient=packer.generate(self.n_total,g.sim_shapes,g.orientations)
        g.positions=pos; g.orientations=orient
        for i in range(self.n_total):
            ts=g.true_shapes[i]; ss=g.sim_shapes[i]
            ts.a,ts.b,ts.c=ss.a,ss.b,ss.c
        for i in range(self.n_total):
            g.drag_coeffs[i]=cfg["mechanics"]["damping"]*g.sim_shapes[i].equivalent_radius/50.0
        g.sync_arrays()

        self.mean_granule_radius=np.mean([s.equivalent_radius for s in g.sim_shapes])
        print(f"  Mean granule radius: {self.mean_granule_radius:.1f} μm")

        if cfg["output"]["save_true_shapes"]:
            print("Generating surface meshes...")
            for ts in g.true_shapes: ts.generate_surface_mesh(16)

        tc=np.sum(g.n_cells)
        print(f"  Total cells: {tc}, mean/func: {tc/max(1,self.n_functional):.1f}")

        # Initial voxel packing
        phi_v = compute_voxel_packing(g.positions, g.shape_params, g.quats,
                                       self.domain, self.mean_granule_radius, self.voxel_res)
        print(f"  Initial voxel packing fraction: {phi_v:.3f} "
              f"(border margin = {self.mean_granule_radius:.1f} μm)")
        self._report_overlaps("After packing")

    def _report_overlaps(self, label=""):
        sp=self.granules.shape_params; pos=self.granules.positions
        quats=self.granules.quats; Rs=_batch_rot(quats); n=self.n_total
        overlaps=[]
        for i in range(n):
            for j in range(i+1,n):
                rij=pos[j]-pos[i]; dist=np.linalg.norm(rij)
                if dist<1e-6: continue
                if dist>max(sp[i,:3])+max(sp[j,:3]): continue
                d=rij/dist
                ri=_se_radius_np(sp[i],Rs[i],d)
                rj=_se_radius_np(sp[j],Rs[j],-d)
                ov=ri+rj-dist
                if ov>0: overlaps.append(ov)
        if overlaps:
            print(f"  {label}: {len(overlaps)} overlapping pairs, max={max(overlaps):.2f}, mean={np.mean(overlaps):.2f} μm")
        else:
            print(f"  {label}: no overlaps")

    def _find_contacts_and_bridges(self):
        pos=self.granules.positions; sp=self.granules.shape_params
        quats=self.granules.quats; types=self.granules.types; n=self.n_total
        nan_m=np.any(np.isnan(pos)|np.isinf(pos),axis=1)
        if np.any(nan_m):
            for i in np.where(nan_m)[0]:
                pos[i]=self.domain/2+np.random.randn(3)*10; self.granules.velocities[i]=0
        max_r=np.max(sp[:,:3]); mbg=self.config["cell_properties"]["max_bridge_gap_um"]
        cutoff=2*max_r+mbg
        if HAS_SCIPY:
            try:
                tree=cKDTree(pos); ps=tree.query_pairs(cutoff)
                pi=np.array([p[0] for p in ps],dtype=np.int32)
                pj=np.array([p[1] for p in ps],dtype=np.int32)
            except:
                pi=np.array([i for i in range(n) for j in range(i+1,n)],dtype=np.int32)
                pj=np.array([j for i in range(n) for j in range(i+1,n)],dtype=np.int32)
        else:
            pi=np.array([i for i in range(n) for j in range(i+1,n)],dtype=np.int32)
            pj=np.array([j for i in range(n) for j in range(i+1,n)],dtype=np.int32)
        np_=len(pi)
        if np_==0: return self._empty_contacts(), self._empty_bridges()
        if HAS_NUMBA:
            r=_find_contacts_numba(pos,sp,quats,types,self.granules.n_cells,pi,pj,np_,mbg)
            return r[:8], r[8:]
        else:
            return self._find_contacts_np(pi, pj)

    def _empty_contacts(self):
        e=np.array([],dtype=np.float64); ei=np.array([],dtype=np.int32)
        return (ei,ei,e,e,e,e,e,e)
    def _empty_bridges(self):
        e=np.array([],dtype=np.float64); ei=np.array([],dtype=np.int32)
        return (ei,ei,ei,e,e,e,e,e,e)

    def _find_contacts_np(self, pi, pj):
        pos=self.granules.positions; sp=self.granules.shape_params
        Rs=_batch_rot(self.granules.quats); types=self.granules.types; nc=self.granules.n_cells
        mbg=self.config["cell_properties"]["max_bridge_gap_um"]
        cl,bl=[],[]
        for idx in range(len(pi)):
            i,j=pi[idx],pj[idx]; rij=pos[j]-pos[i]; dist=np.linalg.norm(rij)
            if dist<1e-6: continue
            d=rij/dist
            ri=_se_radius_np(sp[i],Rs[i],d); rj=_se_radius_np(sp[j],Rs[j],-d)
            ov=ri+rj-dist
            if ov>0: cl.append((i,j,ov,d[0],d[1],d[2],ri,rj))
            if types[i]==0 and types[j]==0:
                gap=dist-ri-rj
                if gap<mbg:
                    prox=max(0,1-max(0,gap)/mbg)
                    nb=max(1,int(np.sqrt(nc[i]*nc[j])*prox))
                    bl.append((i,j,nb,dist,d[0],d[1],d[2],ri,rj))
        def pack(lst,dtypes):
            if not lst: return tuple(np.array([],dtype=t) for t in dtypes)
            cols=list(zip(*lst)); return tuple(np.array(c,dtype=t) for c,t in zip(cols,dtypes))
        contacts=pack(cl,[np.int32]*2+[np.float64]*6)
        bridges=pack(bl,[np.int32]*3+[np.float64]*6)
        return contacts, bridges

    def _compute_forces(self, contacts, bridges):
        cfg=self.config; n=self.n_total
        k_rep=cfg["mechanics"]["repulsion_stiffness"]
        k_wall=cfg["mechanics"]["wall_stiffness"]
        damp=cfg["mechanics"]["damping"]
        k_cell=cfg["cell_properties"]["force_per_cell_nN"]*0.03
        cd=cfg["cell_properties"]["diameter_um"]; mf=100.0
        c_i,c_j,c_ov,c_nx,c_ny,c_nz,c_ri,c_rj=contacts
        b_i,b_j,b_nc,b_dist,b_dx,b_dy,b_dz,b_ri,b_rj=bridges
        if HAS_NUMBA:
            forces=_compute_forces_numba(n,self.granules.positions,self.granules.velocities,
                self.granules.shape_params,self.granules.quats,
                self.granules.types,self.granules.n_cells,self.domain,
                c_i,c_j,c_ov,c_nx,c_ny,c_nz,c_ri,c_rj,len(c_i),
                b_i,b_j,b_nc,b_dist,b_dx,b_dy,b_dz,b_ri,b_rj,len(b_i),
                k_rep,k_wall,damp,k_cell,cd,mf)
        else:
            forces=self._forces_np_sim(contacts,bridges,k_rep,k_wall,damp,k_cell,cd,mf)
        fm=self.granules.types==0
        noise=0.03*np.sqrt(self.granules.n_cells[fm]+1)
        forces[fm]+=noise[:,np.newaxis]*np.random.randn(np.sum(fm),3)
        return forces

    def _forces_np_sim(self, contacts, bridges, k_rep, k_wall, damp, k_cell, cd, mf):
        n=self.n_total; forces=np.zeros((n,3))
        pos=self.granules.positions; vel=self.granules.velocities
        sp=self.granules.shape_params; Rs=_batch_rot(self.granules.quats)
        c_i,c_j,c_ov,c_nx,c_ny,c_nz,c_ri,c_rj=contacts
        for idx in range(len(c_i)):
            i,j=int(c_i[idx]),int(c_j[idx]); ov=c_ov[idx]
            nm=np.array([c_nx[idx],c_ny[idx],c_nz[idx]])
            ri,rj=c_ri[idx],c_rj[idx]; re=np.sqrt(ri*rj)
            F=min(k_rep*re/50*ov*(1+2*ov/max(re,1)),mf)
            vn=np.dot(vel[j]-vel[i],nm)
            Ft=max(0,F-damp*0.5*vn)
            forces[i]-=Ft*nm; forces[j]+=Ft*nm
        b_i,b_j,b_nc,b_dist,b_dx,b_dy,b_dz,b_ri,b_rj=bridges
        for idx in range(len(b_i)):
            i,j=int(b_i[idx]),int(b_j[idx])
            d=np.array([b_dx[idx],b_dy[idx],b_dz[idx]])
            gap=b_dist[idx]-b_ri[idx]-b_rj[idx]; ext=gap-cd
            F=k_cell*b_nc[idx]*ext
            if ext<0: F*=0.1
            F=np.clip(F,-mf/2,mf/2)
            forces[i]+=F*d; forces[j]-=F*d
        for i in range(n):
            for ax in range(3):
                d_lo=np.zeros(3); d_lo[ax]=1.0
                r_lo=_se_radius_np(sp[i],Rs[i],d_lo)
                if pos[i,ax]<r_lo:
                    ov=r_lo-pos[i,ax]; forces[i,ax]+=min(k_wall*ov*(1+ov/max(r_lo,1)),mf)
                d_hi=np.zeros(3); d_hi[ax]=-1.0
                r_hi=_se_radius_np(sp[i],Rs[i],d_hi)
                if pos[i,ax]>self.domain[ax]-r_hi:
                    ov=pos[i,ax]-(self.domain[ax]-r_hi); forces[i,ax]-=min(k_wall*ov*(1+ov/max(r_hi,1)),mf)
        fm=np.linalg.norm(forces,axis=1,keepdims=True)
        big=(fm>mf).flatten()
        if np.any(big): forces[big]*=mf/fm[big]
        return forces

    def step(self):
        self.granules.sync_arrays()
        contacts,bridges=self._find_contacts_and_bridges()
        forces=self._compute_forces(contacts,bridges)
        vel=forces/self.granules.drag_coeffs[:,np.newaxis]
        vm=np.linalg.norm(vel,axis=1,keepdims=True)
        fast=(vm>100).flatten()
        if np.any(fast): vel[fast]*=100.0/vm[fast]
        max_vel=float(np.max(vm))
        if max_vel>1e-10:
            min_r=min(s.equivalent_radius for s in self.granules.sim_shapes)
            self.dt=np.clip(0.03*min_r/max_vel,
                self.config["time"]["dt_min_hours"],self.config["time"]["dt_max_hours"])
        else:
            self.dt=self.config["time"]["dt_max_hours"]
        self.granules.positions+=vel*self.dt
        self.granules.velocities=vel
        br=np.array([s.bounding_radius for s in self.granules.sim_shapes])
        for d in range(3):
            self.granules.positions[:,d]=np.clip(self.granules.positions[:,d],br+1,self.domain[d]-br-1)
        self.time+=self.dt; self.step_count+=1
        return contacts,bridges

    def _record_history(self, contacts, bridges, include_voxel_packing=False):
        c_i,c_j,c_ov=contacts[0],contacts[1],contacts[2]
        nc=len(c_i); nb=len(bridges[0])
        coord=np.zeros(self.n_total); max_ov=0.0
        for idx in range(nc):
            coord[int(c_i[idx])]+=1; coord[int(c_j[idx])]+=1
            if c_ov[idx]>max_ov: max_ov=c_ov[idx]

        if include_voxel_packing:
            self.granules.sync_arrays()
            phi=compute_voxel_packing(self.granules.positions, self.granules.shape_params,
                                      self.granules.quats, self.domain,
                                      self.mean_granule_radius, self.voxel_res)
        else:
            # Use last known value or 0
            phi = self.history['packing_fraction'][-1] if self.history['packing_fraction'] else 0.0

        self.history['time_hours'].append(self.time)
        self.history['n_contacts'].append(nc)
        self.history['n_bridges'].append(nb)
        self.history['mean_coordination'].append(float(np.mean(coord)))
        self.history['max_velocity'].append(float(np.max(np.linalg.norm(self.granules.velocities,axis=1))))
        self.history['packing_fraction'].append(float(phi))
        self.history['max_overlap_um'].append(float(max_ov))

    def _save_history(self):
        target=self.output_dir/f"{self.config_name}_history.json"
        tmp=target.with_suffix('.json.tmp')
        with open(tmp,'w') as f: json.dump(self.history,f)
        tmp.replace(target)

    def save_frame(self, label=None):
        if label is None: label=f"t{self.time:.1f}h"
        fn=self.output_dir/f"{self.config_name}_frame_{label}.json"
        data={'time_hours':self.time,'config_name':self.config_name,
              'n_granules':self.n_total,'domain':self.domain.tolist(),
              'positions':self.granules.positions.tolist(),
              'orientations':[q.q.tolist() for q in self.granules.orientations],
              'sim_shapes':[s.to_dict() for s in self.granules.sim_shapes],
              'true_shapes':[s.to_dict() for s in self.granules.true_shapes],
              'types':self.granules.types.tolist(),
              'n_cells':self.granules.n_cells.tolist()}
        with open(fn,'w') as f: json.dump(data,f)

    def run(self):
        if self.granules is None: self.setup()
        total_h=self.config["time"]["total_hours"]
        save_int=self.config["time"]["save_interval_hours"]
        print(f"\nRunning for {total_h:.0f}h (voxel packing every {save_int:.0f}h)...")
        t0=time_module.time(); last_save=last_rec=0.0
        self.save_frame("initial"); self._save_history()
        if HAS_TQDM: pbar=tqdm(total=total_h,desc="Simulating",unit="hr"); pt=0.0

        while self.time<total_h:
            contacts,bridges=self.step()
            if self.time-last_rec>=0.1:
                # Voxel packing only at frame saves for performance
                self._record_history(contacts,bridges,include_voxel_packing=False)
                last_rec=self.time
            if self.time-last_save>=save_int:
                # Compute true voxel packing at frame save points
                self.granules.sync_arrays()
                phi=compute_voxel_packing(self.granules.positions,self.granules.shape_params,
                                          self.granules.quats,self.domain,
                                          self.mean_granule_radius,self.voxel_res)
                # Update the last packing_fraction entry
                if self.history['packing_fraction']:
                    self.history['packing_fraction'][-1] = float(phi)
                self.save_frame(); self._save_history()
                last_save=self.time
            if HAS_TQDM: pbar.update(self.time-pt); pt=self.time

        if HAS_TQDM: pbar.close()
        # Final voxel packing
        self.granules.sync_arrays()
        phi_final=compute_voxel_packing(self.granules.positions,self.granules.shape_params,
                                        self.granules.quats,self.domain,
                                        self.mean_granule_radius,self.voxel_res)
        self.save_frame("final"); self._save_history()
        elapsed=time_module.time()-t0
        print(f"\nDone! {self.time:.1f}h in {elapsed:.1f}s wall time")
        print(f"  Final voxel packing fraction: {phi_final:.3f}")
        print(f"  Output: {self.output_dir}")
        self._report_overlaps("Final state")
        return self.history


def main():
    print("="*60+"\nDEM Simulation — Superellipsoid Contacts\n"+"="*60)
    if len(sys.argv)>1: cp=sys.argv[1]
    else: print("\nSelect config..."); cp=select_config_file()
    if not cp: print("No config. Exiting."); return
    print(f"\nConfig: {cp}")
    Simulation(load_config(cp), cp).run()

if __name__=="__main__": main()