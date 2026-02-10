"""
DEM Simulation — Superellipsoid Contacts + Cell Tracking
=========================================================

Tracks individual cells on functional granule surfaces.
Cells that bridge between granules stretch (volume-conserved),
generating measurable stress and aspect ratio changes.

Usage: python dem_simulation.py config.json
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json, sys, time as time_module

try:
    import tkinter as tk; from tkinter import filedialog; HAS_TK=True
except ImportError: HAS_TK=False
try:
    from tqdm import tqdm; HAS_TQDM=True
except ImportError: HAS_TQDM=False
try:
    from scipy.spatial import cKDTree; HAS_SCIPY=True
except ImportError: HAS_SCIPY=False
try:
    from numba import njit, prange, float64, int32, int64; HAS_NUMBA=True
    print("[Accel] Numba detected")
except ImportError: HAS_NUMBA=False; print("[Accel] NumPy fallback")


# =============================================================================
# NUMBA KERNELS
# =============================================================================
if HAS_NUMBA:
    @njit(cache=True)
    def _qr(q):
        w,x,y,z=q[0],q[1],q[2],q[3]; R=np.empty((3,3),dtype=float64)
        R[0,0]=1-2*y*y-2*z*z;R[0,1]=2*x*y-2*w*z;R[0,2]=2*x*z+2*w*y
        R[1,0]=2*x*y+2*w*z;R[1,1]=1-2*x*x-2*z*z;R[1,2]=2*y*z-2*w*x
        R[2,0]=2*x*z-2*w*y;R[2,1]=2*y*z+2*w*x;R[2,2]=1-2*x*x-2*y*y
        return R
    @njit(cache=True)
    def _ser(sp,R,d):
        a,b,c,n=sp[0],sp[1],sp[2],sp[3]
        db0=R[0,0]*d[0]+R[1,0]*d[1]+R[2,0]*d[2]
        db1=R[0,1]*d[0]+R[1,1]*d[1]+R[2,1]*d[2]
        db2=R[0,2]*d[0]+R[1,2]*d[1]+R[2,2]*d[2]
        v=abs(db0/a)**n+abs(db1/b)**n+abs(db2/c)**n
        return max(a,max(b,c)) if v<1e-30 else 1.0/v**(1.0/n)
    @njit(cache=True)
    def _br(sp): return max(sp[0],max(sp[1],sp[2]))
    @njit(cache=True)
    def _pise(px,py,pz,pos,sp,R):
        rx=px-pos[0];ry=py-pos[1];rz=pz-pos[2]
        bx=R[0,0]*rx+R[1,0]*ry+R[2,0]*rz
        by=R[0,1]*rx+R[1,1]*ry+R[2,1]*rz
        bz=R[0,2]*rx+R[1,2]*ry+R[2,2]*rz
        return abs(bx/sp[0])**sp[3]+abs(by/sp[1])**sp[3]+abs(bz/sp[2])**sp[3]<=1.0

    @njit(parallel=True,cache=True)
    def _pack_f(pos,sp,q,dom,sc,k,mf):
        n=pos.shape[0];f=np.zeros((n,3),dtype=float64);mo=0.0
        for i in prange(n):
            Ri=_qr(q[i]);si=sp[i].copy();si[0]*=sc;si[1]*=sc;si[2]*=sc;rbi=_br(si)
            for j in range(i+1,n):
                dx=pos[j,0]-pos[i,0];dy=pos[j,1]-pos[i,1];dz=pos[j,2]-pos[i,2]
                d=np.sqrt(dx*dx+dy*dy+dz*dz)
                if d<1e-6:continue
                sj=sp[j].copy();sj[0]*=sc;sj[1]*=sc;sj[2]*=sc
                if d>rbi+_br(sj):continue
                iv=1.0/d;dv=np.array([dx*iv,dy*iv,dz*iv]);nd=np.array([-dv[0],-dv[1],-dv[2]])
                ri=_ser(si,Ri,dv);Rj=_qr(q[j]);rj=_ser(sj,Rj,nd);ov=ri+rj-d
                if ov>0:
                    if ov>mo:mo=ov
                    F=min(k*ov,mf)
                    for c in range(3):f[i,c]-=F*dv[c];f[j,c]+=F*dv[c]
        return f,mo
    @njit(parallel=True,cache=True)
    def _wall_f(pos,sp,q,dom,sc,k,mf):
        n=pos.shape[0];f=np.zeros((n,3),dtype=float64)
        for i in prange(n):
            Ri=_qr(q[i]);si=sp[i].copy();si[0]*=sc;si[1]*=sc;si[2]*=sc
            for d in range(3):
                dl=np.zeros(3,dtype=float64);dl[d]=1.0;rl=_ser(si,Ri,dl)
                if pos[i,d]<rl:f[i,d]+=min(k*(rl-pos[i,d]),mf)
                dh=np.zeros(3,dtype=float64);dh[d]=-1.0;rh=_ser(si,Ri,dh)
                if pos[i,d]>dom[d]-rh:f[i,d]-=min(k*(pos[i,d]-(dom[d]-rh)),mf)
        return f
    @njit(cache=True)
    def _find_c(pos,sp,q,ty,nc,pi,pj,np_,mbg):
        ci=np.empty(np_,dtype=int32);cj=np.empty(np_,dtype=int32)
        co=np.empty(np_,dtype=float64);cx=np.empty(np_,dtype=float64)
        cy_=np.empty(np_,dtype=float64);cz=np.empty(np_,dtype=float64)
        cri=np.empty(np_,dtype=float64);crj=np.empty(np_,dtype=float64);nc_=0
        bi=np.empty(np_,dtype=int32);bj=np.empty(np_,dtype=int32)
        bnc=np.empty(np_,dtype=int32);bd=np.empty(np_,dtype=float64)
        bx=np.empty(np_,dtype=float64);by=np.empty(np_,dtype=float64)
        bz=np.empty(np_,dtype=float64);bri=np.empty(np_,dtype=float64)
        brj=np.empty(np_,dtype=float64);nb=0
        for idx in range(np_):
            i=pi[idx];j=pj[idx]
            dx=pos[j,0]-pos[i,0];dy=pos[j,1]-pos[i,1];dz=pos[j,2]-pos[i,2]
            dt=np.sqrt(dx*dx+dy*dy+dz*dz)
            if dt<1e-6:continue
            iv=1.0/dt;dv=np.array([dx*iv,dy*iv,dz*iv]);nd=np.array([-dv[0],-dv[1],-dv[2]])
            Ri=_qr(q[i]);ri=_ser(sp[i],Ri,dv);Rj=_qr(q[j]);rj=_ser(sp[j],Rj,nd)
            ov=ri+rj-dt
            if ov>0:
                ci[nc_]=i;cj[nc_]=j;co[nc_]=ov;cx[nc_]=dv[0];cy_[nc_]=dv[1];cz[nc_]=dv[2]
                cri[nc_]=ri;crj[nc_]=rj;nc_+=1
            if ty[i]==0 and ty[j]==0:
                gap=dt-ri-rj
                if gap<mbg:
                    pr=max(0.0,1.0-max(0.0,gap)/mbg)
                    nb_=max(1,int(np.sqrt(float(nc[i])*float(nc[j]))*pr))
                    bi[nb]=i;bj[nb]=j;bnc[nb]=nb_;bd[nb]=dt;bx[nb]=dv[0];by[nb]=dv[1];bz[nb]=dv[2]
                    bri[nb]=ri;brj[nb]=rj;nb+=1
        return (ci[:nc_],cj[:nc_],co[:nc_],cx[:nc_],cy_[:nc_],cz[:nc_],cri[:nc_],crj[:nc_],
                bi[:nb],bj[:nb],bnc[:nb],bd[:nb],bx[:nb],by[:nb],bz[:nb],bri[:nb],brj[:nb])
    @njit(parallel=True,cache=True)
    def _comp_f(ng,pos,vel,sp,q,ty,nca,dom,ci,cj,co,cnx,cny,cnz,cri,crj,nc,
                bi,bj,bnc,bd,bdx,bdy,bdz,bri,brj,nb,kr,kw,da,kc,cd,mf):
        f=np.zeros((ng,3),dtype=float64)
        for idx in range(nc):
            i=ci[idx];j=cj[idx];ov=co[idx];nx=cnx[idx];ny=cny[idx];nz=cnz[idx]
            ri=cri[idx];rj=crj[idx];re=np.sqrt(ri*rj)
            Fm=kr*re/50.0*ov*(1.0+2.0*ov/max(re,1.0))
            if Fm>mf:Fm=mf
            vn=(vel[j,0]-vel[i,0])*nx+(vel[j,1]-vel[i,1])*ny+(vel[j,2]-vel[i,2])*nz
            Ft=Fm-da*0.5*vn
            if Ft<0:Ft=0.0
            f[i,0]-=Ft*nx;f[i,1]-=Ft*ny;f[i,2]-=Ft*nz
            f[j,0]+=Ft*nx;f[j,1]+=Ft*ny;f[j,2]+=Ft*nz
        for idx in range(nb):
            i=bi[idx];j=bj[idx];gap=bd[idx]-bri[idx]-brj[idx]
            ext=gap-cd;F=kc*bnc[idx]*ext
            if ext<0:F*=0.1
            if F>mf*0.5:F=mf*0.5
            elif F<-mf*0.5:F=-mf*0.5
            f[i,0]+=F*bdx[idx];f[i,1]+=F*bdy[idx];f[i,2]+=F*bdz[idx]
            f[j,0]-=F*bdx[idx];f[j,1]-=F*bdy[idx];f[j,2]-=F*bdz[idx]
        for i in prange(ng):
            Ri=_qr(q[i])
            for d in range(3):
                dl=np.zeros(3,dtype=float64);dl[d]=1.0;rl=_ser(sp[i],Ri,dl)
                if pos[i,d]<rl:ov=rl-pos[i,d];ff=kw*ov*(1+ov/max(rl,1));f[i,d]+=min(ff,mf)
                dh=np.zeros(3,dtype=float64);dh[d]=-1.0;rh=_ser(sp[i],Ri,dh)
                if pos[i,d]>dom[d]-rh:ov=pos[i,d]-(dom[d]-rh);ff=kw*ov*(1+ov/max(rh,1));f[i,d]-=min(ff,mf)
            fx,fy,fz=f[i,0],f[i,1],f[i,2];fm=np.sqrt(fx*fx+fy*fy+fz*fz)
            if fm>mf:s=mf/fm;f[i,0]*=s;f[i,1]*=s;f[i,2]*=s
        return f
    @njit(parallel=True,cache=True)
    def _vox_pack(pos,sp,q,gmin,gmax,res):
        nx=res;dx=(gmax[0]-gmin[0])/nx;dy=(gmax[1]-gmin[1])/nx;dz=(gmax[2]-gmin[2])/nx
        ng=pos.shape[0];filled=np.zeros(nx,dtype=int64)
        for ix in prange(nx):
            px=gmin[0]+(ix+0.5)*dx;cnt=0
            for iy in range(nx):
                py=gmin[1]+(iy+0.5)*dy
                for iz in range(nx):
                    pz=gmin[2]+(iz+0.5)*dz;ins=False
                    for g in range(ng):
                        if (px-pos[g,0])**2+(py-pos[g,1])**2+(pz-pos[g,2])**2>_br(sp[g])**2:continue
                        if _pise(px,py,pz,pos[g],sp[g],_qr(q[g])):ins=True;break
                    if ins:cnt+=1
            filled[ix]=cnt
        return int(np.sum(filled)),nx**3

# NumPy fallbacks
def _batch_rot(q):
    w,x,y,z=q[:,0],q[:,1],q[:,2],q[:,3];R=np.empty((len(q),3,3))
    R[:,0,0]=1-2*y*y-2*z*z;R[:,0,1]=2*x*y-2*w*z;R[:,0,2]=2*x*z+2*w*y
    R[:,1,0]=2*x*y+2*w*z;R[:,1,1]=1-2*x*x-2*z*z;R[:,1,2]=2*y*z-2*w*x
    R[:,2,0]=2*x*z-2*w*y;R[:,2,1]=2*y*z+2*w*x;R[:,2,2]=1-2*x*x-2*y*y
    return R
def _ser_np(sp,R,d):
    a,b,c,n=sp;db=R.T@d;v=abs(db[0]/a)**n+abs(db[1]/b)**n+abs(db[2]/c)**n
    return max(a,b,c) if v<1e-30 else 1.0/v**(1.0/n)

def voxel_packing(pos,sp,q,dom,margin,res=40):
    gmin=np.array([margin]*3,dtype=np.float64);gmax=dom-margin
    if np.any(gmax<=gmin):return 0.0
    if HAS_NUMBA:f,t=_vox_pack(pos,sp,q,gmin,gmax,res)
    else:
        Rs=_batch_rot(q);xs=np.linspace(gmin[0],gmax[0],res);f=0;t=res**3
        for px in xs:
            for py in np.linspace(gmin[1],gmax[1],res):
                for pz in np.linspace(gmin[2],gmax[2],res):
                    p=np.array([px,py,pz])
                    for g in range(len(pos)):
                        if np.sum((p-pos[g])**2)>max(sp[g,:3])**2:continue
                        b=Rs[g].T@(p-pos[g]);s=sp[g]
                        if abs(b[0]/s[0])**s[3]+abs(b[1]/s[1])**s[3]+abs(b[2]/s[2])**s[3]<=1:f+=1;break
    return f/max(t,1)

# Config
def sel_cfg():
    if not HAS_TK:return None
    r=tk.Tk();r.withdraw();r.attributes('-topmost',True)
    fp=filedialog.askopenfilename(title="Config",filetypes=[("JSON","*.json")],initialdir=".")
    r.destroy();return fp if fp else None
def load_cfg(fp):
    with open(fp) as f:c=json.load(f)
    d={"simulation_name":"sim","domain":{"side_length_um":500,"target_packing_fraction":0.55},
       "granule_ratio":{"functional_fraction":0.5},
       "functional_granules":{"radius_mean_um":40,"radius_std_um":8,"aspect_ratio_range":[1,1.5],"roundness_range":[2,3],"roughness_range":[0.1,0.3]},
       "inert_granules":{"radius_mean_um":50,"radius_std_um":10,"aspect_ratio_range":[1,1.3],"roundness_range":[2,2.5],"roughness_range":[0,0.2]},
       "cell_properties":{"diameter_um":20,"attachment_area_fraction":0.5,"force_per_cell_nN":5,"max_bridge_gap_um":50},
       "mechanics":{"repulsion_stiffness":1,"damping":1.5,"wall_stiffness":2},
       "time":{"total_hours":72,"save_interval_hours":2,"dt_initial_hours":0.01,"dt_min_hours":0.001,"dt_max_hours":0.5},
       "output":{"base_directory":"./simulations","save_true_shapes":True}}
    def m(a,b):
        for k,v in b.items():
            if k not in a:a[k]=v
            elif isinstance(v,dict)and isinstance(a[k],dict):m(a[k],v)
    m(c,d);return c
def calc_gc(c):
    s=c["domain"]["side_length_um"];p=c["domain"]["target_packing_fraction"];ff=c["granule_ratio"]["functional_fraction"]
    V=s**3*p;rf=c["functional_granules"]["radius_mean_um"];ri=c["inert_granules"]["radius_mean_um"]
    return max(1,round(V*ff/((4/3)*np.pi*rf**3))),max(1,round(V*(1-ff)/((4/3)*np.pi*ri**3)))
def calc_nc(area,c):
    cd=c["cell_properties"]["diameter_um"];af=c["cell_properties"]["attachment_area_fraction"]
    return max(1,int(np.ceil(area*af/(np.pi*(cd/2)**2))))

# =============================================================================
# SHAPES
# =============================================================================
class GT(Enum):
    F=0;I=1
@dataclass
class GS:
    a:float;b:float;c:float;n:float=2.0;roughness:float=0.0
    @property
    def eqr(self):return(self.a*self.b*self.c)**(1/3)
    @property
    def vol(self):return(4/3)*np.pi*self.a*self.b*self.c
    @property
    def sa(self):
        p=1.6075;return 4*np.pi*((self.a**p*self.b**p+self.a**p*self.c**p+self.b**p*self.c**p)/3)**(1/p)
    @property
    def br(self):return max(self.a,self.b,self.c)
    @property
    def params(self):return np.array([self.a,self.b,self.c,self.n])
    def to_dict(self):return{"a":self.a,"b":self.b,"c":self.c,"n":self.n,"roughness":self.roughness}
@dataclass
class TS:
    a:float;b:float;c:float;n:float;roughness:float
    def to_dict(self):return{"a":self.a,"b":self.b,"c":self.c,"n":self.n,"roughness":self.roughness}
def mk_shapes(mr,cfg):
    ar=np.random.uniform(*cfg["aspect_ratio_range"]);n=np.random.uniform(*cfg["roundness_range"])
    ro=np.random.uniform(*cfg["roughness_range"]);c=mr*ar**(1/3);ab=mr/ar**(1/6)
    asy=np.random.uniform(0.95,1.05);a=ab*asy;b=ab/asy
    tv=(4/3)*np.pi*mr**3;cv=(4/3)*np.pi*a*b*c
    if cv>0:s=(tv/cv)**(1/3);a*=s;b*=s;c*=s
    return GS(a,b,c,n,ro),TS(a,b,c,n,ro)

class Quaternion:
    __slots__=['q']
    def __init__(s,w=1,x=0,y=0,z=0):
        s.q=np.array([w,x,y,z],dtype=np.float64);nm=np.linalg.norm(s.q)
        if nm>1e-10:s.q/=nm
    @classmethod
    def random(c):
        u=np.random.random(3)
        return c(np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                 np.sqrt(u[0])*np.sin(2*np.pi*u[2]),np.sqrt(u[0])*np.cos(2*np.pi*u[2]))
    def to_matrix(s):
        w,x,y,z=s.q
        return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                         [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                         [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])

# =============================================================================
# CELL TRACKER
# =============================================================================

def fibonacci_sphere(n):
    """Generate n roughly-uniform points on unit sphere."""
    pts=np.empty((n,3))
    golden=np.pi*(3-np.sqrt(5))
    for i in range(n):
        y=1-2*i/(n-1) if n>1 else 0
        r=np.sqrt(max(0,1-y*y))
        th=golden*i
        pts[i]=[r*np.cos(th),y,r*np.sin(th)]
    return pts

def se_surface_point(body_dir, a, b, c, n_exp):
    """Project unit direction onto superellipsoid surface."""
    d=body_dir/np.linalg.norm(body_dir)
    v=abs(d[0]/a)**n_exp+abs(d[1]/b)**n_exp+abs(d[2]/c)**n_exp
    if v<1e-30:return d*max(a,b,c)
    r=1.0/v**(1.0/n_exp)
    return d*r

class CellTracker:
    """Track individual cells on functional granule surfaces."""

    def __init__(self, n_granules, n_cells_arr, shapes, config):
        self.n_granules = n_granules
        self.cell_diameter = config["cell_properties"]["diameter_um"]
        self.k_cell = config["cell_properties"]["force_per_cell_nN"]
        self.max_bridge_gap = config["cell_properties"]["max_bridge_gap_um"]

        # Cell volume (sphere approx)
        self.cell_volume = (4/3)*np.pi*(self.cell_diameter/2)**3

        # Generate cell positions in body frame for each functional granule
        self.cell_body_pos = []   # list of (n_cells_i, 3) arrays
        self.cell_parent = []     # flat array: parent granule index
        self.cell_offsets = np.zeros(n_granules+1, dtype=int)  # cumsum for indexing

        total = 0
        for g in range(n_granules):
            nc = int(n_cells_arr[g])
            if nc > 0:
                shape = shapes[g]
                dirs = fibonacci_sphere(nc)
                body_pts = np.array([se_surface_point(dirs[k], shape.a, shape.b, shape.c, shape.n)
                                     for k in range(nc)])
                self.cell_body_pos.append(body_pts)
            else:
                self.cell_body_pos.append(np.empty((0, 3)))
            self.cell_offsets[g+1] = self.cell_offsets[g] + n_cells_arr[g]
            total += n_cells_arr[g]

        self.total_cells = int(total)

        # Per-cell state (flat arrays indexed by global cell ID)
        self.world_pos = np.zeros((self.total_cells, 3))
        self.is_bridging = np.zeros(self.total_cells, dtype=bool)
        self.bridge_target_granule = -np.ones(self.total_cells, dtype=int)
        self.bridge_endpoint = np.zeros((self.total_cells, 3))  # world pos of far end
        self.gap = np.zeros(self.total_cells)
        self.stress = np.zeros(self.total_cells)     # force per cell in nN
        self.aspect_ratio = np.ones(self.total_cells) # L/w ratio
        self.cell_parent_flat = np.zeros(self.total_cells, dtype=int)

        # Build parent index
        for g in range(n_granules):
            s, e = self.cell_offsets[g], self.cell_offsets[g+1]
            self.cell_parent_flat[s:e] = g

    def update(self, positions, orientations, bridges_tuple):
        """
        Update cell world positions and bridge assignments.
        bridges_tuple = (b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj)
        """
        # Reset bridge state
        self.is_bridging[:] = False
        self.bridge_target_granule[:] = -1
        self.stress[:] = 0.0
        self.aspect_ratio[:] = 1.0
        self.gap[:] = 0.0

        # Update world positions: body_pos → world via R @ body + center
        for g in range(self.n_granules):
            s, e = self.cell_offsets[g], self.cell_offsets[g+1]
            if e <= s:
                continue
            R = orientations[g].to_matrix()
            body = self.cell_body_pos[g]
            self.world_pos[s:e] = (R @ body.T).T + positions[g]

        # Process bridges
        b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj = bridges_tuple
        if len(b_i) == 0:
            return

        d_cell = self.cell_diameter
        V_cell = self.cell_volume

        for idx in range(len(b_i)):
            gi, gj = int(b_i[idx]), int(b_j[idx])
            n_bridge = int(b_nc[idx])
            direction = np.array([b_dx[idx], b_dy[idx], b_dz[idx]])
            gap = max(0.0, float(b_dist[idx] - b_ri[idx] - b_rj[idx]))

            # Get cell indices for each granule
            si, ei = self.cell_offsets[gi], self.cell_offsets[gi+1]
            sj, ej = self.cell_offsets[gj], self.cell_offsets[gj+1]
            if ei <= si or ej <= sj:
                continue

            # Score cells by how much they face the partner granule
            cells_i = self.world_pos[si:ei]
            cells_j = self.world_pos[sj:ej]

            # Direction from gi center to gj center
            normals_i = cells_i - np.mean(cells_i, axis=0)  # approximate outward direction
            # Better: direction from granule center to cell
            center_i = np.mean(cells_i, axis=0)
            center_j = np.mean(cells_j, axis=0)

            # Score: dot product of (cell - granule_center) with bridge direction
            scores_i = np.sum((cells_i - center_i) * direction, axis=1)
            scores_j = np.sum((cells_j - center_j) * (-direction), axis=1)

            # Pick top n_bridge cells from each side
            n_pick = min(n_bridge, ei-si, ej-sj)
            top_i = np.argsort(scores_i)[-n_pick:]
            top_j = np.argsort(scores_j)[-n_pick:]

            # Per-cell mechanics
            for k in range(n_pick):
                ci_global = si + top_i[k]
                cj_global = sj + top_j[k]

                self.is_bridging[ci_global] = True
                self.is_bridging[cj_global] = True
                self.bridge_target_granule[ci_global] = gj
                self.bridge_target_granule[cj_global] = gi
                self.bridge_endpoint[ci_global] = self.world_pos[cj_global]
                self.bridge_endpoint[cj_global] = self.world_pos[ci_global]
                self.gap[ci_global] = gap
                self.gap[cj_global] = gap

                # Cell length when spanning gap
                L = max(d_cell, gap)

                # Volume-conserved width: V = π/4 * w² * L → w = sqrt(4V/(πL))
                w = np.sqrt(4 * V_cell / (np.pi * L))

                # Aspect ratio
                ar = L / max(w, 1e-6)
                self.aspect_ratio[ci_global] = ar
                self.aspect_ratio[cj_global] = ar

                # Stress: force per cell = k * extension
                extension = max(0.0, gap - d_cell)
                F = self.k_cell * extension  # nN
                self.stress[ci_global] = F
                self.stress[cj_global] = F

    def get_frame_data(self):
        """Return cell data dict for saving in frame JSON."""
        return {
            "total_cells": int(self.total_cells),
            "cell_diameter_um": float(self.cell_diameter),
            "parent": self.cell_parent_flat.tolist(),
            "world_pos": self.world_pos.tolist(),
            "is_bridging": self.is_bridging.tolist(),
            "bridge_target": self.bridge_target_granule.tolist(),
            "bridge_endpoint": self.bridge_endpoint.tolist(),
            "gap_um": self.gap.tolist(),
            "stress_nN": self.stress.tolist(),
            "aspect_ratio": self.aspect_ratio.tolist()
        }

# =============================================================================
# GRANULE SYSTEM
# =============================================================================
class GSys:
    def __init__(s,n):
        s.n=n;s.pos=np.zeros((n,3));s.vel=np.zeros((n,3))
        s.orient=[Quaternion() for _ in range(n)]
        s.sim_shapes=[];s.true_shapes=[];s.types=np.zeros(n,dtype=np.int32)
        s.n_cells=np.zeros(n,dtype=np.int32);s.drag=np.ones(n)
        s._sp=None;s._q=None
    def sync(s):
        s._sp=np.array([sh.params for sh in s.sim_shapes])
        s._q=np.array([q.q for q in s.orient])
    @property
    def sp(s):
        if s._sp is None:s.sync()
        return s._sp
    @property
    def qu(s):
        if s._q is None:s.sync()
        return s._q

# =============================================================================
# PACKING
# =============================================================================
class Packer:
    def __init__(s,ds,tp,v=True):s.dom=np.array([ds]*3);s.tp=tp;s.v=v
    def _relax(s,p,sp,q,dom,sc,k,da,mf,mi,tol):
        mo=1e6
        for it in range(mi):
            if HAS_NUMBA:f,mo=_pack_f(p,sp,q,dom,sc,k,mf);f+=_wall_f(p,sp,q,dom,sc,k,mf)
            else:f=s._fnp(p,sp,q,dom,sc,k,mf);mo=0
            fm=np.linalg.norm(f,axis=1,keepdims=True);big=(fm>mf).flatten()
            if np.any(big):f[big]*=mf/fm[big]
            p+=f*0.3/da
            nan=np.any(np.isnan(p),axis=1)
            if np.any(nan):p[nan]=dom/2+np.random.randn(int(np.sum(nan)),3)*10
            br=np.max(sp[:,:3],axis=1)*sc
            for d in range(3):p[:,d]=np.clip(p[:,d],br+0.5,dom[d]-br-0.5)
            if mo<tol:break
        return mo
    def _fnp(s,p,sp,q,dom,sc,k,mf):
        n=len(p);f=np.zeros((n,3));Rs=_batch_rot(q)
        for i in range(n):
            si=sp[i].copy();si[:3]*=sc;rbi=max(si[:3])
            for j in range(i+1,n):
                rij=p[j]-p[i];d=np.linalg.norm(rij)
                if d<1e-6:continue
                sj=sp[j].copy();sj[:3]*=sc
                if d>rbi+max(sj[:3]):continue
                dv=rij/d;ri=_ser_np(si,Rs[i],dv);rj=_ser_np(sj,Rs[j],-dv);ov=ri+rj-d
                if ov>0:F=min(k*ov,mf)*dv;f[i]-=F;f[j]+=F
        for i in range(n):
            si=sp[i].copy();si[:3]*=sc
            for ax in range(3):
                dl=np.zeros(3);dl[ax]=1;rl=_ser_np(si,Rs[i],dl)
                if p[i,ax]<rl:f[i,ax]+=min(k*(rl-p[i,ax]),mf)
                dh=np.zeros(3);dh[ax]=-1;rh=_ser_np(si,Rs[i],dh)
                if p[i,ax]>dom[ax]-rh:f[i,ax]-=min(k*(p[i,ax]-(dom[ax]-rh)),mf)
        return f
    def generate(s,n,shapes,orients):
        if s.v:print(f"\nPacking {n} superellipsoids...")
        p=np.zeros((n,3));ns=int(np.ceil(n**(1/3)));g=min(s.dom)/(ns+1);idx=0
        for ix in range(ns):
            for iy in range(ns):
                for iz in range(ns):
                    if idx>=n:break
                    p[idx]=[(ix+1)*g+np.random.uniform(-g*.1,g*.1),(iy+1)*g+np.random.uniform(-g*.1,g*.1),(iz+1)*g+np.random.uniform(-g*.1,g*.1)]
                    p[idx]=np.clip(p[idx],20,s.dom-20);idx+=1
                if idx>=n:break
            if idx>=n:break
        sp=np.array([sh.params for sh in shapes]);q=np.array([o.q for o in orients]);dom=s.dom.copy()
        sc=0.05;bgr=0.0005;k=1.0;da=2.0;mf=50.0;cp=0.0
        if s.v and HAS_TQDM:pb=tqdm(total=s.tp,desc="Growing",unit="φ");lp=0
        for it in range(400000):
            if cp>=s.tp:break
            pr=cp/max(s.tp,1e-6)
            gr=bgr*(0.02 if pr>0.8 else 0.1 if pr>0.6 else 0.5 if pr>0.3 else 1)
            if HAS_NUMBA:f,_=_pack_f(p,sp,q,dom,sc,k,mf);f+=_wall_f(p,sp,q,dom,sc,k,mf)
            else:f=s._fnp(p,sp,q,dom,sc,k,mf)
            fm=np.linalg.norm(f,axis=1,keepdims=True);big=(fm>mf).flatten()
            if np.any(big):f[big]*=mf/fm[big]
            p+=f*0.4/da
            nan=np.any(np.isnan(p),axis=1)
            if np.any(nan):p[nan]=dom/2+np.random.randn(int(np.sum(nan)),3)*10
            br=np.max(sp[:,:3],axis=1)*sc
            for d in range(3):p[:,d]=np.clip(p[:,d],br+1,dom[d]-br-1)
            sc+=gr;vols=(4/3)*np.pi*sp[:,0]*sp[:,1]*sp[:,2]*sc**3;cp=np.sum(vols)/np.prod(dom)
            if it%200==0 and it>0:
                mr=np.min(sp[:,:3])*sc;s._relax(p,sp,q,dom,sc,k*2,da,mf,50,mr*0.3)
            if s.v and HAS_TQDM and cp-lp>0.005:pb.update(cp-lp);lp=cp
        if s.v and HAS_TQDM:pb.close()
        mr=np.min(sp[:,:3])*sc;tol=mr*0.05
        if s.v:print(f"Relaxing (tol={tol:.2f}μm)...")
        ov=s._relax(p,sp,q,dom,sc,k*4,da,mf*2,10000,tol)
        if s.v:print(f"  Phase1: {ov:.3f}")
        if ov>tol:ov=s._relax(p,sp,q,dom,sc,k*16,da*0.5,mf*4,10000,tol);
        if s.v and ov>tol:print(f"  Phase2: {ov:.3f}")
        if ov>tol:ov=s._relax(p,sp,q,dom,sc,k*64,da*0.25,mf*8,20000,tol)
        if s.v and ov>tol:print(f"  Phase3: {ov:.3f}")
        for sh in shapes:sh.a*=sc;sh.b*=sc;sh.c*=sc
        if s.v:print(f"  φ_analytic: {cp:.3f}")
        return p,orients

# =============================================================================
# SIMULATION
# =============================================================================
class Simulation:
    def __init__(s,cfg,cfp):
        s.cfg=cfg;s.cn=Path(cfp).stem
        s.od=Path(cfg["output"]["base_directory"])/s.cn;s.od.mkdir(parents=True,exist_ok=True)
        with open(s.od/f"{s.cn}_config.json",'w') as f:json.dump(cfg,f,indent=2)
        s.ds=cfg["domain"]["side_length_um"];s.dom=np.array([s.ds]*3)
        s.nf,s.ni=calc_gc(cfg);s.nt=s.nf+s.ni
        print(f"\n{s.cn}: {s.ds:.0f}³μm, {s.nf}f+{s.ni}i={s.nt}")
        s.g=None;s.t=0.0;s.sc=0;s.dt=cfg["time"]["dt_initial_hours"]
        s.mgr=0.0;s.cell_tracker=None
        s.hist={'time_hours':[],'n_contacts':[],'n_bridges':[],'mean_coordination':[],
                'max_velocity':[],'packing_fraction':[],'max_overlap_um':[],
                'mean_stress_nN':[],'max_stress_nN':[],'mean_ar':[],'max_ar':[],
                'n_bridging_cells':[],'total_cells':0}

    def setup(s):
        c=s.cfg;g=GSys(s.nt);s.g=g
        a_s,a_t,a_ty,a_nc=[],[],[],[]
        for _ in range(s.nf):
            r=np.clip(np.random.normal(c["functional_granules"]["radius_mean_um"],c["functional_granules"]["radius_std_um"]),15,c["functional_granules"]["radius_mean_um"]*2)
            ss,ts=mk_shapes(r,c["functional_granules"]);a_s.append(ss);a_t.append(ts);a_ty.append(0)
            a_nc.append(calc_nc(ss.sa,c))
        for _ in range(s.ni):
            r=np.clip(np.random.normal(c["inert_granules"]["radius_mean_um"],c["inert_granules"]["radius_std_um"]),15,c["inert_granules"]["radius_mean_um"]*2)
            ss,ts=mk_shapes(r,c["inert_granules"]);a_s.append(ss);a_t.append(ts);a_ty.append(1);a_nc.append(0)
        ix=np.random.permutation(s.nt)
        g.sim_shapes=[a_s[i] for i in ix];g.true_shapes=[a_t[i] for i in ix]
        g.types=np.array([a_ty[i] for i in ix],dtype=np.int32)
        g.n_cells=np.array([a_nc[i] for i in ix],dtype=np.int32)
        g.orient=[Quaternion.random() for _ in range(s.nt)]
        pk=Packer(s.ds,c["domain"]["target_packing_fraction"])
        pos,ori=pk.generate(s.nt,g.sim_shapes,g.orient)
        g.pos=pos;g.orient=ori
        for i in range(s.nt):ts=g.true_shapes[i];ss=g.sim_shapes[i];ts.a,ts.b,ts.c=ss.a,ss.b,ss.c
        for i in range(s.nt):g.drag[i]=c["mechanics"]["damping"]*g.sim_shapes[i].eqr/50
        g.sync()
        s.mgr=np.mean([sh.eqr for sh in g.sim_shapes])

        # Initialize cell tracker
        s.cell_tracker = CellTracker(s.nt, g.n_cells, g.sim_shapes, c)
        s.hist['total_cells'] = s.cell_tracker.total_cells
        print(f"  Cells: {s.cell_tracker.total_cells} on {s.nf} functional granules")
        print(f"  Mean radius: {s.mgr:.1f}μm")

    def _find_cb(s):
        pos=s.g.pos;sp=s.g.sp;q=s.g.qu;ty=s.g.types;n=s.nt
        nan=np.any(np.isnan(pos)|np.isinf(pos),axis=1)
        if np.any(nan):
            for i in np.where(nan)[0]:pos[i]=s.dom/2+np.random.randn(3)*10;s.g.vel[i]=0
        mr=np.max(sp[:,:3]);mbg=s.cfg["cell_properties"]["max_bridge_gap_um"];co=2*mr+mbg
        if HAS_SCIPY:
            try:t=cKDTree(pos);ps=t.query_pairs(co);pi=np.array([p[0] for p in ps],dtype=np.int32);pj=np.array([p[1] for p in ps],dtype=np.int32)
            except:pi=np.array([i for i in range(n) for j in range(i+1,n)],dtype=np.int32);pj=np.array([j for i in range(n) for j in range(i+1,n)],dtype=np.int32)
        else:pi=np.array([i for i in range(n) for j in range(i+1,n)],dtype=np.int32);pj=np.array([j for i in range(n) for j in range(i+1,n)],dtype=np.int32)
        np_=len(pi)
        if np_==0:
            ec=tuple(np.array([],dtype=t) for t in [np.int32]*2+[np.float64]*6)
            eb=tuple(np.array([],dtype=t) for t in [np.int32]*3+[np.float64]*6)
            return ec,eb
        if HAS_NUMBA:
            r=_find_c(pos,sp,q,ty,s.g.n_cells,pi,pj,np_,mbg);return r[:8],r[8:]
        else:
            Rs=_batch_rot(q);cl,bl=[],[]
            for idx in range(np_):
                i,j=pi[idx],pj[idx];rij=pos[j]-pos[i];d=np.linalg.norm(rij)
                if d<1e-6:continue
                dv=rij/d;ri=_ser_np(sp[i],Rs[i],dv);rj=_ser_np(sp[j],Rs[j],-dv);ov=ri+rj-d
                if ov>0:cl.append((i,j,ov,dv[0],dv[1],dv[2],ri,rj))
                if ty[i]==0 and ty[j]==0:
                    gap=d-ri-rj
                    if gap<mbg:
                        pr=max(0,1-max(0,gap)/mbg);nb=max(1,int(np.sqrt(s.g.n_cells[i]*s.g.n_cells[j])*pr))
                        bl.append((i,j,nb,d,dv[0],dv[1],dv[2],ri,rj))
            def pk(l,dt):
                if not l:return tuple(np.array([],dtype=t) for t in dt)
                return tuple(np.array(c,dtype=t) for c,t in zip(zip(*l),dt))
            return pk(cl,[np.int32]*2+[np.float64]*6),pk(bl,[np.int32]*3+[np.float64]*6)

    def _forces(s,con,bri):
        c=s.cfg;n=s.nt;kr=c["mechanics"]["repulsion_stiffness"];kw=c["mechanics"]["wall_stiffness"]
        da=c["mechanics"]["damping"];kc=c["cell_properties"]["force_per_cell_nN"]*0.03
        cd=c["cell_properties"]["diameter_um"];mf=100.0
        ci,cj,co,cnx,cny,cnz,cri,crj=con;bi,bj,bnc,bd,bdx,bdy,bdz,bri_,brj=bri
        if HAS_NUMBA:
            f=_comp_f(n,s.g.pos,s.g.vel,s.g.sp,s.g.qu,s.g.types,s.g.n_cells,s.dom,
                      ci,cj,co,cnx,cny,cnz,cri,crj,len(ci),bi,bj,bnc,bd,bdx,bdy,bdz,bri_,brj,len(bi),
                      kr,kw,da,kc,cd,mf)
        else:
            f=np.zeros((n,3));pos=s.g.pos;vel=s.g.vel;sp=s.g.sp;Rs=_batch_rot(s.g.qu)
            for idx in range(len(ci)):
                i,j=int(ci[idx]),int(cj[idx]);ov=co[idx];nm=np.array([cnx[idx],cny[idx],cnz[idx]])
                ri,rj=cri[idx],crj[idx];re=np.sqrt(ri*rj)
                F=min(kr*re/50*ov*(1+2*ov/max(re,1)),mf)
                vn=np.dot(vel[j]-vel[i],nm);Ft=max(0,F-da*0.5*vn)
                f[i]-=Ft*nm;f[j]+=Ft*nm
            for idx in range(len(bi)):
                i,j=int(bi[idx]),int(bj[idx]);d=np.array([bdx[idx],bdy[idx],bdz[idx]])
                gap=bd[idx]-bri_[idx]-brj[idx];ext=gap-cd;F=kc*bnc[idx]*ext
                if ext<0:F*=0.1
                F=np.clip(F,-mf/2,mf/2);f[i]+=F*d;f[j]-=F*d
            for i in range(n):
                for ax in range(3):
                    dl=np.zeros(3);dl[ax]=1;rl=_ser_np(sp[i],Rs[i],dl)
                    if pos[i,ax]<rl:ov=rl-pos[i,ax];f[i,ax]+=min(kw*ov*(1+ov/max(rl,1)),mf)
                    dh=np.zeros(3);dh[ax]=-1;rh=_ser_np(sp[i],Rs[i],dh)
                    if pos[i,ax]>s.dom[ax]-rh:ov=pos[i,ax]-(s.dom[ax]-rh);f[i,ax]-=min(kw*ov*(1+ov/max(rh,1)),mf)
            fm=np.linalg.norm(f,axis=1,keepdims=True);big=(fm>mf).flatten()
            if np.any(big):f[big]*=mf/fm[big]
        fm=s.g.types==0;noise=0.03*np.sqrt(s.g.n_cells[fm]+1)
        f[fm]+=noise[:,np.newaxis]*np.random.randn(np.sum(fm),3)
        return f

    def step(s):
        s.g.sync();con,bri=s._find_cb()
        # Update cell tracker with bridge data
        s.cell_tracker.update(s.g.pos, s.g.orient, bri)
        f=s._forces(con,bri)
        vel=f/s.g.drag[:,np.newaxis]
        vm=np.linalg.norm(vel,axis=1,keepdims=True);fast=(vm>100).flatten()
        if np.any(fast):vel[fast]*=100/vm[fast]
        mv=float(np.max(vm))
        if mv>1e-10:
            mr=min(sh.eqr for sh in s.g.sim_shapes)
            s.dt=np.clip(0.03*mr/mv,s.cfg["time"]["dt_min_hours"],s.cfg["time"]["dt_max_hours"])
        else:s.dt=s.cfg["time"]["dt_max_hours"]
        s.g.pos+=vel*s.dt;s.g.vel=vel
        br=np.array([sh.br for sh in s.g.sim_shapes])
        for d in range(3):s.g.pos[:,d]=np.clip(s.g.pos[:,d],br+1,s.dom[d]-br-1)
        s.t+=s.dt;s.sc+=1
        return con,bri

    def _rec(s,con,bri,voxel=False):
        ci,cj,co=con[0],con[1],con[2];nc=len(ci);nb=len(bri[0])
        coord=np.zeros(s.nt);mo=0.0
        for idx in range(nc):coord[int(ci[idx])]+=1;coord[int(cj[idx])]+=1;mo=max(mo,co[idx])
        if voxel:
            s.g.sync();phi=voxel_packing(s.g.pos,s.g.sp,s.g.qu,s.dom,s.mgr,40)
        else:phi=s.hist['packing_fraction'][-1] if s.hist['packing_fraction'] else 0
        # Cell metrics
        ct = s.cell_tracker
        bridging = ct.is_bridging
        n_br_cells = int(np.sum(bridging))
        mean_stress = float(np.mean(ct.stress[bridging])) if n_br_cells > 0 else 0.0
        max_stress = float(np.max(ct.stress[bridging])) if n_br_cells > 0 else 0.0
        mean_ar = float(np.mean(ct.aspect_ratio[bridging])) if n_br_cells > 0 else 1.0
        max_ar = float(np.max(ct.aspect_ratio[bridging])) if n_br_cells > 0 else 1.0

        s.hist['time_hours'].append(s.t);s.hist['n_contacts'].append(nc)
        s.hist['n_bridges'].append(nb);s.hist['mean_coordination'].append(float(np.mean(coord)))
        s.hist['max_velocity'].append(float(np.max(np.linalg.norm(s.g.vel,axis=1))))
        s.hist['packing_fraction'].append(float(phi));s.hist['max_overlap_um'].append(float(mo))
        s.hist['mean_stress_nN'].append(mean_stress);s.hist['max_stress_nN'].append(max_stress)
        s.hist['mean_ar'].append(mean_ar);s.hist['max_ar'].append(max_ar)
        s.hist['n_bridging_cells'].append(n_br_cells)

    def _save_hist(s):
        tgt=s.od/f"{s.cn}_history.json";tmp=tgt.with_suffix('.json.tmp')
        with open(tmp,'w') as f:json.dump(s.hist,f)
        tmp.replace(tgt)

    def save_frame(s,label=None):
        if label is None:label=f"t{s.t:.1f}h"
        fn=s.od/f"{s.cn}_frame_{label}.json"
        data={'time_hours':s.t,'config_name':s.cn,'n_granules':s.nt,
              'domain':s.dom.tolist(),'positions':s.g.pos.tolist(),
              'orientations':[q.q.tolist() for q in s.g.orient],
              'sim_shapes':[sh.to_dict() for sh in s.g.sim_shapes],
              'true_shapes':[ts.to_dict() for ts in s.g.true_shapes],
              'types':s.g.types.tolist(),'n_cells':s.g.n_cells.tolist(),
              'cell_data':s.cell_tracker.get_frame_data()}
        with open(fn,'w') as f:json.dump(data,f)

    def run(s):
        if s.g is None:s.setup()
        th=s.cfg["time"]["total_hours"];si=s.cfg["time"]["save_interval_hours"]
        print(f"\nRunning {th:.0f}h...");t0=time_module.time();ls=lr=0.0
        s.save_frame("initial");s._save_hist()
        if HAS_TQDM:pb=tqdm(total=th,desc="Sim",unit="hr");pt=0
        while s.t<th:
            con,bri=s.step()
            if s.t-lr>=0.1:s._rec(con,bri);lr=s.t
            if s.t-ls>=si:
                s.g.sync();phi=voxel_packing(s.g.pos,s.g.sp,s.g.qu,s.dom,s.mgr,40)
                if s.hist['packing_fraction']:s.hist['packing_fraction'][-1]=float(phi)
                s.save_frame();s._save_hist();ls=s.t
            if HAS_TQDM:pb.update(s.t-pt);pt=s.t
        if HAS_TQDM:pb.close()
        s.save_frame("final");s._save_hist()
        el=time_module.time()-t0;print(f"\nDone! {s.t:.1f}h in {el:.1f}s → {s.od}")

def main():
    print("="*60+"\nDEM + Cell Tracking\n"+"="*60)
    if len(sys.argv)>1:cp=sys.argv[1]
    else:cp=sel_cfg()
    if not cp:print("No config.");return
    Simulation(load_cfg(cp),cp).run()

if __name__=="__main__":main()