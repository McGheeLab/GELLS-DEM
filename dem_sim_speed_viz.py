"""
Coarse-Grained DEM Simulation — Accelerated
=============================================

Performance tiers (automatic fallback):
  1. Numba JIT  — ~50-100x on pairwise kernels
  2. NumPy vectorized — batch rotation matrices, distances
  3. Multiprocessing — parallel packing relaxation for large n

Requirements:
    pip install numpy scipy tqdm
    pip install numba          # optional, large speedup
    pip install multiprocessing  # stdlib, used automatically

Usage:
    python dem_simulation.py                    # Opens file dialog
    python dem_simulation.py config.json        # Uses specified config
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json
import sys
import os
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

# ---------------------------------------------------------------------------
# Numba acceleration
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange, float64, int32
    HAS_NUMBA = True
    print("[Accel] Numba detected — JIT-compiling kernels")
except ImportError:
    HAS_NUMBA = False
    print("[Accel] Numba not found — using NumPy vectorized fallback")

# Multiprocessing for packing
from multiprocessing import Pool, cpu_count

N_WORKERS = max(1, cpu_count() - 1)


# =============================================================================
# NUMBA KERNELS (compiled once, called many times)
# =============================================================================

if HAS_NUMBA:
    @njit(cache=True)
    def _quat_to_matrix(q):
        """Quaternion [w,x,y,z] -> 3x3 rotation matrix."""
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = np.empty((3, 3), dtype=float64)
        R[0, 0] = 1 - 2*y*y - 2*z*z;  R[0, 1] = 2*x*y - 2*w*z;      R[0, 2] = 2*x*z + 2*w*y
        R[1, 0] = 2*x*y + 2*w*z;      R[1, 1] = 1 - 2*x*x - 2*z*z;  R[1, 2] = 2*y*z - 2*w*x
        R[2, 0] = 2*x*z - 2*w*y;      R[2, 1] = 2*y*z + 2*w*x;      R[2, 2] = 1 - 2*x*x - 2*y*y
        return R

    @njit(cache=True)
    def _effective_radius(semi, R, d):
        """Effective radius of ellipsoid(semi) with rotation R along direction d."""
        db = np.empty(3, dtype=float64)
        for k in range(3):
            db[k] = R[0, k]*d[0] + R[1, k]*d[1] + R[2, k]*d[2]  # R^T @ d
        inv_sq = (db[0]/semi[0])**2 + (db[1]/semi[1])**2 + (db[2]/semi[2])**2
        if inv_sq < 1e-20:
            return max(semi[0], semi[1], semi[2])
        return 1.0 / np.sqrt(inv_sq)

    @njit(cache=True)
    def _bounding_radius(semi):
        return max(semi[0], max(semi[1], semi[2]))

    @njit(parallel=True, cache=True)
    def _packing_forces_numba(positions, semi_axes, quats, domain, scale,
                              stiffness, max_force):
        """
        Compute packing repulsion forces for all granules in parallel.
        positions:  (n, 3)
        semi_axes:  (n, 3) — unscaled semi-axes
        quats:      (n, 4) — quaternions
        Returns forces (n, 3) and max_overlap scalar.
        """
        n = positions.shape[0]
        forces = np.zeros((n, 3), dtype=float64)
        max_overlap = 0.0

        for i in prange(n):
            Ri = _quat_to_matrix(quats[i])
            rb_i = _bounding_radius(semi_axes[i]) * scale

            for j in range(i + 1, n):
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < 1e-6:
                    continue

                rb_j = _bounding_radius(semi_axes[j]) * scale
                if dist > rb_i + rb_j:
                    continue

                inv_d = 1.0 / dist
                d_vec = np.array([dx * inv_d, dy * inv_d, dz * inv_d])
                neg_d = np.array([-d_vec[0], -d_vec[1], -d_vec[2]])

                ri = _effective_radius(semi_axes[i], Ri, d_vec) * scale
                Rj = _quat_to_matrix(quats[j])
                rj = _effective_radius(semi_axes[j], Rj, neg_d) * scale

                overlap = ri + rj - dist
                if overlap > 0:
                    if overlap > max_overlap:
                        max_overlap = overlap
                    F_mag = min(stiffness * overlap, max_force)
                    for k in range(3):
                        forces[i, k] -= F_mag * d_vec[k]
                        forces[j, k] += F_mag * d_vec[k]

        return forces, max_overlap

    @njit(parallel=True, cache=True)
    def _wall_forces_numba(positions, semi_axes, quats, domain, scale,
                           stiffness, max_force):
        """Compute wall forces for all granules in parallel."""
        n = positions.shape[0]
        forces = np.zeros((n, 3), dtype=float64)

        for i in prange(n):
            Ri = _quat_to_matrix(quats[i])
            for d in range(3):
                # Low wall — direction toward wall is +axis
                d_lo = np.zeros(3, dtype=float64)
                d_lo[d] = 1.0
                r_lo = _effective_radius(semi_axes[i], Ri, d_lo) * scale
                if positions[i, d] < r_lo:
                    ov = r_lo - positions[i, d]
                    forces[i, d] += min(stiffness * ov, max_force)
                # High wall
                d_hi = np.zeros(3, dtype=float64)
                d_hi[d] = -1.0
                r_hi = _effective_radius(semi_axes[i], Ri, d_hi) * scale
                if positions[i, d] > domain[d] - r_hi:
                    ov = positions[i, d] - (domain[d] - r_hi)
                    forces[i, d] -= min(stiffness * ov, max_force)

        return forces

    @njit(cache=True)
    def _find_contacts_numba(positions, semi_axes, quats, types, n_cells,
                             pair_i, pair_j, n_pairs,
                             max_bridge_gap):
        """
        Process candidate pairs to find contacts and bridges.
        Returns arrays of contact/bridge data.
        """
        # Pre-allocate output arrays (worst case: all pairs are contacts)
        c_i = np.empty(n_pairs, dtype=int32)
        c_j = np.empty(n_pairs, dtype=int32)
        c_overlap = np.empty(n_pairs, dtype=float64)
        c_nx = np.empty(n_pairs, dtype=float64)
        c_ny = np.empty(n_pairs, dtype=float64)
        c_nz = np.empty(n_pairs, dtype=float64)
        c_ri = np.empty(n_pairs, dtype=float64)
        c_rj = np.empty(n_pairs, dtype=float64)
        n_contacts = 0

        b_i = np.empty(n_pairs, dtype=int32)
        b_j = np.empty(n_pairs, dtype=int32)
        b_n_cells = np.empty(n_pairs, dtype=int32)
        b_dist = np.empty(n_pairs, dtype=float64)
        b_dx = np.empty(n_pairs, dtype=float64)
        b_dy = np.empty(n_pairs, dtype=float64)
        b_dz = np.empty(n_pairs, dtype=float64)
        b_ri = np.empty(n_pairs, dtype=float64)
        b_rj = np.empty(n_pairs, dtype=float64)
        n_bridges = 0

        for idx in range(n_pairs):
            i = pair_i[idx]
            j = pair_j[idx]

            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            if dist < 1e-6:
                continue

            inv_d = 1.0 / dist
            d_vec = np.array([dx*inv_d, dy*inv_d, dz*inv_d])
            neg_d = np.array([-d_vec[0], -d_vec[1], -d_vec[2]])

            Ri = _quat_to_matrix(quats[i])
            ri = _effective_radius(semi_axes[i], Ri, d_vec)
            Rj = _quat_to_matrix(quats[j])
            rj = _effective_radius(semi_axes[j], Rj, neg_d)

            overlap = ri + rj - dist

            if overlap > 0:
                c_i[n_contacts] = i
                c_j[n_contacts] = j
                c_overlap[n_contacts] = overlap
                c_nx[n_contacts] = d_vec[0]
                c_ny[n_contacts] = d_vec[1]
                c_nz[n_contacts] = d_vec[2]
                c_ri[n_contacts] = ri
                c_rj[n_contacts] = rj
                n_contacts += 1

            if types[i] == 0 and types[j] == 0:
                gap = dist - ri - rj
                if gap < max_bridge_gap:
                    proximity = max(0.0, 1.0 - max(0.0, gap) / max_bridge_gap)
                    nc = max(1, int(np.sqrt(float(n_cells[i]) * float(n_cells[j])) * proximity))
                    b_i[n_bridges] = i
                    b_j[n_bridges] = j
                    b_n_cells[n_bridges] = nc
                    b_dist[n_bridges] = dist
                    b_dx[n_bridges] = d_vec[0]
                    b_dy[n_bridges] = d_vec[1]
                    b_dz[n_bridges] = d_vec[2]
                    b_ri[n_bridges] = ri
                    b_rj[n_bridges] = rj
                    n_bridges += 1

        return (c_i[:n_contacts], c_j[:n_contacts], c_overlap[:n_contacts],
                c_nx[:n_contacts], c_ny[:n_contacts], c_nz[:n_contacts],
                c_ri[:n_contacts], c_rj[:n_contacts],
                b_i[:n_bridges], b_j[:n_bridges], b_n_cells[:n_bridges],
                b_dist[:n_bridges], b_dx[:n_bridges], b_dy[:n_bridges], b_dz[:n_bridges],
                b_ri[:n_bridges], b_rj[:n_bridges])

    @njit(parallel=True, cache=True)
    def _compute_contact_forces_numba(
        n_granules, positions, velocities, semi_axes, quats, types, n_cells_arr,
        domain,
        c_i, c_j, c_overlap, c_nx, c_ny, c_nz, c_ri, c_rj, n_contacts,
        b_i, b_j, b_ncells, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj, n_bridges,
        k_rep, k_wall, damping, cell_force_scale, cell_diameter, max_force,
        noise_scale
    ):
        """Compute all forces: contacts, bridges, walls, noise."""
        forces = np.zeros((n_granules, 3), dtype=float64)

        # --- Contact forces (serial to avoid race conditions on shared indices) ---
        for idx in range(n_contacts):
            i = c_i[idx]; j = c_j[idx]
            ov = c_overlap[idx]
            nx, ny, nz = c_nx[idx], c_ny[idx], c_nz[idx]
            ri, rj = c_ri[idx], c_rj[idx]
            r_eff = np.sqrt(ri * rj)

            F_mag = k_rep * r_eff / 50.0 * ov * (1.0 + 2.0 * ov / max(r_eff, 1.0))
            if F_mag > max_force:
                F_mag = max_force

            vx = velocities[j, 0] - velocities[i, 0]
            vy = velocities[j, 1] - velocities[i, 1]
            vz = velocities[j, 2] - velocities[i, 2]
            v_n = vx*nx + vy*ny + vz*nz
            F_damp = -damping * 0.5 * v_n

            F_total = F_mag + F_damp
            if F_total < 0:
                F_total = 0.0

            forces[i, 0] -= F_total * nx
            forces[i, 1] -= F_total * ny
            forces[i, 2] -= F_total * nz
            forces[j, 0] += F_total * nx
            forces[j, 1] += F_total * ny
            forces[j, 2] += F_total * nz

        # --- Bridge forces ---
        for idx in range(n_bridges):
            i = b_i[idx]; j = b_j[idx]
            nc = b_ncells[idx]
            ri, rj = b_ri[idx], b_rj[idx]
            gap = b_dist[idx] - ri - rj
            extension = gap - cell_diameter
            F_mag = cell_force_scale * nc * extension
            if extension < 0:
                F_mag *= 0.1
            if F_mag > max_force * 0.5:
                F_mag = max_force * 0.5
            elif F_mag < -max_force * 0.5:
                F_mag = -max_force * 0.5

            forces[i, 0] += F_mag * b_dx[idx]
            forces[i, 1] += F_mag * b_dy[idx]
            forces[i, 2] += F_mag * b_dz[idx]
            forces[j, 0] -= F_mag * b_dx[idx]
            forces[j, 1] -= F_mag * b_dy[idx]
            forces[j, 2] -= F_mag * b_dz[idx]

        # --- Wall forces (parallel over granules) ---
        for i in prange(n_granules):
            Ri = _quat_to_matrix(quats[i])
            for d in range(3):
                d_lo = np.zeros(3, dtype=float64); d_lo[d] = 1.0
                r_lo = _effective_radius(semi_axes[i], Ri, d_lo)
                if positions[i, d] < r_lo:
                    ov = r_lo - positions[i, d]
                    f = k_wall * ov * (1.0 + ov / max(r_lo, 1.0))
                    if f > max_force: f = max_force
                    forces[i, d] += f

                d_hi = np.zeros(3, dtype=float64); d_hi[d] = -1.0
                r_hi = _effective_radius(semi_axes[i], Ri, d_hi)
                if positions[i, d] > domain[d] - r_hi:
                    ov = positions[i, d] - (domain[d] - r_hi)
                    f = k_wall * ov * (1.0 + ov / max(r_hi, 1.0))
                    if f > max_force: f = max_force
                    forces[i, d] -= f

        # --- Cap forces ---
        for i in prange(n_granules):
            fx, fy, fz = forces[i, 0], forces[i, 1], forces[i, 2]
            f_mag = np.sqrt(fx*fx + fy*fy + fz*fz)
            if f_mag > max_force:
                s = max_force / f_mag
                forces[i, 0] *= s; forces[i, 1] *= s; forces[i, 2] *= s

        return forces


# =============================================================================
# NUMPY VECTORIZED FALLBACKS
# =============================================================================

def _batch_rotation_matrices(quats: np.ndarray) -> np.ndarray:
    """Convert (n, 4) quaternion array to (n, 3, 3) rotation matrices."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.empty((len(quats), 3, 3))
    R[:, 0, 0] = 1 - 2*y*y - 2*z*z; R[:, 0, 1] = 2*x*y - 2*w*z;     R[:, 0, 2] = 2*x*z + 2*w*y
    R[:, 1, 0] = 2*x*y + 2*w*z;     R[:, 1, 1] = 1 - 2*x*x - 2*z*z; R[:, 1, 2] = 2*y*z - 2*w*x
    R[:, 2, 0] = 2*x*z - 2*w*y;     R[:, 2, 1] = 2*y*z + 2*w*x;     R[:, 2, 2] = 1 - 2*x*x - 2*y*y
    return R


def _effective_radius_np(semi: np.ndarray, R: np.ndarray, d: np.ndarray) -> float:
    """Single ellipsoid effective radius (numpy fallback)."""
    d_body = R.T @ d
    inv_sq = (d_body[0]/semi[0])**2 + (d_body[1]/semi[1])**2 + (d_body[2]/semi[2])**2
    if inv_sq < 1e-20:
        return np.max(semi)
    return 1.0 / np.sqrt(inv_sq)


# =============================================================================
# CONFIGURATION
# =============================================================================

def select_config_file() -> Optional[str]:
    if not HAS_TK:
        print("tkinter not available. Please specify config file as argument.")
        return None
    root = tk.Tk(); root.withdraw(); root.attributes('-topmost', True)
    fp = filedialog.askopenfilename(
        title="Select Simulation Config File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=".")
    root.destroy()
    return fp if fp else None


def load_config(filepath: str) -> Dict[str, Any]:
    with open(filepath) as f:
        config = json.load(f)

    defaults = {
        "simulation_name": "unnamed_sim",
        "domain": {"side_length_um": 500.0, "target_packing_fraction": 0.55},
        "granule_ratio": {"functional_fraction": 0.5},
        "functional_granules": {
            "radius_mean_um": 40.0, "radius_std_um": 8.0,
            "aspect_ratio_range": [1.0, 1.5], "roundness_range": [2.0, 3.0],
            "roughness_range": [0.1, 0.3]},
        "inert_granules": {
            "radius_mean_um": 50.0, "radius_std_um": 10.0,
            "aspect_ratio_range": [1.0, 1.3], "roundness_range": [2.0, 2.5],
            "roughness_range": [0.0, 0.2]},
        "cell_properties": {
            "diameter_um": 20.0, "attachment_area_fraction": 0.5,
            "force_per_cell_nN": 5.0, "max_bridge_gap_um": 50.0},
        "mechanics": {"repulsion_stiffness": 1.0, "damping": 1.5, "wall_stiffness": 2.0},
        "time": {
            "total_hours": 72.0, "save_interval_hours": 2.0,
            "dt_initial_hours": 0.01, "dt_min_hours": 0.001, "dt_max_hours": 0.5},
        "output": {"base_directory": "./simulations", "save_true_shapes": True}
    }

    def merge(cfg, defs):
        for k, v in defs.items():
            if k not in cfg:
                cfg[k] = v
            elif isinstance(v, dict) and isinstance(cfg[k], dict):
                merge(cfg[k], v)
    merge(config, defaults)
    return config


def calculate_granule_counts(config):
    s = config["domain"]["side_length_um"]
    phi = config["domain"]["target_packing_fraction"]
    ff = config["granule_ratio"]["functional_fraction"]
    V = s**3 * phi
    r_f = config["functional_granules"]["radius_mean_um"]
    r_i = config["inert_granules"]["radius_mean_um"]
    return (max(1, int(np.round(V * ff / ((4/3)*np.pi*r_f**3)))),
            max(1, int(np.round(V * (1-ff) / ((4/3)*np.pi*r_i**3)))))


def calculate_cells_on_granule(area, config):
    cd = config["cell_properties"]["diameter_um"]
    af = config["cell_properties"]["attachment_area_fraction"]
    return max(1, int(np.ceil(area * af / (np.pi * (cd/2)**2))))


# =============================================================================
# SHAPE CLASSES
# =============================================================================

class GranuleType(Enum):
    FUNCTIONAL = 0
    INERT = 1


@dataclass
class GranuleShape:
    a: float; b: float; c: float; n: float = 2.0; roughness: float = 0.0

    @property
    def equivalent_radius(self):
        return ((3/(4*np.pi)) * self.a * self.b * self.c * (4/3)*np.pi) ** (1/3)
        # Simplified: (a*b*c)^(1/3)
    @property
    def volume(self): return (4/3)*np.pi*self.a*self.b*self.c
    @property
    def surface_area(self):
        p = 1.6075
        ap, bp, cp = self.a**p, self.b**p, self.c**p
        return 4*np.pi*((ap*bp + ap*cp + bp*cp)/3)**(1/p)
    @property
    def bounding_radius(self): return max(self.a, self.b, self.c)
    @property
    def semi_axes(self): return np.array([self.a, self.b, self.c])

    def to_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c, "n": self.n, "roughness": self.roughness}


@dataclass
class TrueShape:
    a: float; b: float; c: float; n: float; roughness: float
    surface_points: Optional[np.ndarray] = None

    def generate_surface_mesh(self, resolution=20):
        th = np.linspace(0, 2*np.pi, resolution)
        ph = np.linspace(0, np.pi, resolution//2)
        th, ph = np.meshgrid(th, ph)
        def spow(x, p): return np.sign(x)*np.abs(x)**p
        e = 2.0/self.n
        x = self.a*spow(np.cos(th), e)*spow(np.sin(ph), e)
        y = self.b*spow(np.sin(th), e)*spow(np.sin(ph), e)
        z = self.c*spow(np.cos(ph), e)
        if self.roughness > 0:
            noise = np.random.randn(*x.shape)*self.roughness*min(self.a,self.b,self.c)*0.1
            r = np.sqrt(x**2+y**2+z**2)
            x += noise*x/(r+1e-6); y += noise*y/(r+1e-6); z += noise*z/(r+1e-6)
        self.surface_points = np.stack([x,y,z], axis=-1)
        return self.surface_points

    def to_dict(self):
        return {"a": self.a, "b": self.b, "c": self.c,
                "n": self.n, "roughness": self.roughness,
                "has_surface_mesh": self.surface_points is not None}


def create_granule_shapes(mean_r, cfg_section):
    ar = np.random.uniform(*cfg_section["aspect_ratio_range"])
    n = np.random.uniform(*cfg_section["roundness_range"])
    rough = np.random.uniform(*cfg_section["roughness_range"])
    c = mean_r * ar**(1/3); ab = mean_r / ar**(1/6)
    asym = np.random.uniform(0.95, 1.05)
    a, b = ab*asym, ab/asym
    tv = (4/3)*np.pi*mean_r**3; cv = (4/3)*np.pi*a*b*c
    if cv > 0:
        s = (tv/cv)**(1/3); a *= s; b *= s; c *= s
    return GranuleShape(a,b,c,n,rough), TrueShape(a,b,c,n,rough)


# =============================================================================
# QUATERNION
# =============================================================================

class Quaternion:
    __slots__ = ['q']
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w,x,y,z], dtype=np.float64)
        norm = np.linalg.norm(self.q)
        if norm > 1e-10: self.q /= norm

    @classmethod
    def random(cls):
        u = np.random.random(3)
        return cls(np.sqrt(1-u[0])*np.sin(2*np.pi*u[1]),
                   np.sqrt(1-u[0])*np.cos(2*np.pi*u[1]),
                   np.sqrt(u[0])*np.sin(2*np.pi*u[2]),
                   np.sqrt(u[0])*np.cos(2*np.pi*u[2]))

    def to_matrix(self):
        w,x,y,z = self.q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
            [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


# =============================================================================
# GRANULE SYSTEM (Array-of-Struct → Struct-of-Arrays for vectorization)
# =============================================================================

class GranuleSystem:
    def __init__(self, n):
        self.n = n
        self.positions = np.zeros((n, 3))
        self.velocities = np.zeros((n, 3))
        self.orientations: List[Quaternion] = [Quaternion() for _ in range(n)]
        self.sim_shapes: List[GranuleShape] = []
        self.true_shapes: List[TrueShape] = []
        self.types = np.zeros(n, dtype=np.int32)
        self.n_cells = np.zeros(n, dtype=np.int32)
        self.drag_coeffs = np.ones(n)

        # Pre-allocated contiguous arrays for numba kernels
        self._semi_axes: Optional[np.ndarray] = None
        self._quats: Optional[np.ndarray] = None

    def sync_arrays(self):
        """Sync object data into contiguous arrays for kernels."""
        self._semi_axes = np.array([s.semi_axes for s in self.sim_shapes])
        self._quats = np.array([q.q for q in self.orientations])

    @property
    def semi_axes(self):
        if self._semi_axes is None:
            self.sync_arrays()
        return self._semi_axes

    @property
    def quats(self):
        if self._quats is None:
            self.sync_arrays()
        return self._quats


# =============================================================================
# PACKING GENERATOR (accelerated)
# =============================================================================

class PackingGenerator:
    def __init__(self, domain_size, target_phi, verbose=True):
        self.domain = np.array([domain_size]*3)
        self.target_phi = target_phi
        self.verbose = verbose

    def _relax(self, positions, semi_axes, quats, domain, scale,
               stiffness, damping_coeff, max_force, orientations, shapes,
               max_iters, tol):
        """
        Run overlap-relaxation sub-loop until max overlap < tol or max_iters.
        Returns final max_overlap.
        """
        max_ov = 1e6
        for it in range(max_iters):
            if HAS_NUMBA:
                forces, max_ov = _packing_forces_numba(
                    positions, semi_axes, quats, domain, scale, stiffness, max_force)
                forces += _wall_forces_numba(
                    positions, semi_axes, quats, domain, scale, stiffness, max_force)
            else:
                forces = self._forces_numpy(
                    positions, semi_axes, quats, domain,
                    scale, stiffness, max_force, orientations, shapes)
                max_ov = 0

            f_mags = np.linalg.norm(forces, axis=1, keepdims=True)
            mask = (f_mags > max_force).flatten()
            if np.any(mask):
                forces[mask] *= max_force / f_mags[mask]

            positions += forces * 0.3 / damping_coeff

            nan_mask = np.any(np.isnan(positions), axis=1)
            if np.any(nan_mask):
                positions[nan_mask] = domain / 2 + np.random.randn(int(np.sum(nan_mask)), 3) * 10
            br = np.max(semi_axes, axis=1) * scale
            for d in range(3):
                positions[:, d] = np.clip(positions[:, d], br + 0.5, domain[d] - br - 0.5)

            if max_ov < tol:
                break
        return max_ov

    def generate(self, n, shapes, orientations):
        if self.verbose:
            print(f"\nGenerating jammed packing for {n} granules...")
            if HAS_NUMBA:
                print("  Using Numba-accelerated packing kernels")

        # Grid init with extra spacing to start overlap-free
        positions = np.zeros((n, 3))
        ns = int(np.ceil(n**(1/3)))
        sp = min(self.domain) / (ns + 1)
        idx = 0
        for ix in range(ns):
            for iy in range(ns):
                for iz in range(ns):
                    if idx >= n: break
                    positions[idx] = [(ix+1)*sp + np.random.uniform(-sp*0.1, sp*0.1),
                                      (iy+1)*sp + np.random.uniform(-sp*0.1, sp*0.1),
                                      (iz+1)*sp + np.random.uniform(-sp*0.1, sp*0.1)]
                    positions[idx] = np.clip(positions[idx], 20, self.domain - 20)
                    idx += 1
                if idx >= n: break
            if idx >= n: break

        semi_axes = np.array([s.semi_axes for s in shapes])
        quats = np.array([q.q for q in orientations])
        domain = self.domain.copy()

        # --- Growth phase with interleaved relaxation ---
        scale = 0.05
        stiffness, damping_coeff = 1.0, 2.0
        max_force = 50.0
        current_phi = 0.0

        # Adaptive growth: fast early, very slow near target
        base_growth_rate = 0.0005
        relax_every = 200        # relax sub-cycle every N growth steps
        relax_sub_iters = 50     # iterations per sub-cycle
        min_radius_scaled = np.min(semi_axes) * scale

        if self.verbose and HAS_TQDM:
            pbar = tqdm(total=self.target_phi, desc="Growing packing", unit="φ")
            last_phi = 0.0

        for iteration in range(400000):
            if current_phi >= self.target_phi:
                break

            # Growth rate decays as we approach target
            progress = current_phi / max(self.target_phi, 1e-6)
            if progress < 0.3:
                growth_rate = base_growth_rate
            elif progress < 0.6:
                growth_rate = base_growth_rate * 0.5
            elif progress < 0.8:
                growth_rate = base_growth_rate * 0.1
            else:
                growth_rate = base_growth_rate * 0.02

            # One growth + force step
            if HAS_NUMBA:
                forces, _ = _packing_forces_numba(
                    positions, semi_axes, quats, domain, scale, stiffness, max_force)
                forces += _wall_forces_numba(
                    positions, semi_axes, quats, domain, scale, stiffness, max_force)
            else:
                forces = self._forces_numpy(
                    positions, semi_axes, quats, domain,
                    scale, stiffness, max_force, orientations, shapes)

            f_mags = np.linalg.norm(forces, axis=1, keepdims=True)
            mask = (f_mags > max_force).flatten()
            if np.any(mask):
                forces[mask] *= max_force / f_mags[mask]

            positions += forces * 0.4 / damping_coeff

            nan_mask = np.any(np.isnan(positions), axis=1)
            if np.any(nan_mask):
                positions[nan_mask] = domain / 2 + np.random.randn(int(np.sum(nan_mask)), 3) * 10
            br = np.max(semi_axes, axis=1) * scale
            for d in range(3):
                positions[:, d] = np.clip(positions[:, d], br + 1, domain[d] - br - 1)

            scale += growth_rate
            vols = (4/3) * np.pi * semi_axes[:, 0] * semi_axes[:, 1] * semi_axes[:, 2] * scale**3
            current_phi = np.sum(vols) / np.prod(domain)
            min_radius_scaled = np.min(semi_axes) * scale

            # Interleaved relaxation sub-cycle to prevent overlap accumulation
            if iteration % relax_every == 0 and iteration > 0:
                sub_tol = min_radius_scaled * 0.3  # generous during growth
                self._relax(positions, semi_axes, quats, domain, scale,
                            stiffness * 2, damping_coeff, max_force,
                            orientations, shapes, relax_sub_iters, sub_tol)

            if self.verbose and HAS_TQDM and current_phi - last_phi > 0.005:
                pbar.update(current_phi - last_phi)
                last_phi = current_phi

        if self.verbose and HAS_TQDM:
            pbar.close()

        # --- Final relaxation: aggressive, tight tolerance ---
        min_r = np.min(semi_axes) * scale
        overlap_tol = min_r * 0.05  # 5% of smallest semi-axis

        if self.verbose:
            print(f"Final relaxation (tolerance = {overlap_tol:.2f} μm)...")

        # Phase 1: high stiffness, many iterations
        max_ov = self._relax(
            positions, semi_axes, quats, domain, scale,
            stiffness * 4, damping_coeff, max_force * 2,
            orientations, shapes,
            max_iters=10000, tol=overlap_tol)

        if self.verbose:
            print(f"  Phase 1 done: max overlap = {max_ov:.3f} μm")

        # Phase 2: even stiffer if still overlapping
        if max_ov > overlap_tol:
            max_ov = self._relax(
                positions, semi_axes, quats, domain, scale,
                stiffness * 16, damping_coeff * 0.5, max_force * 4,
                orientations, shapes,
                max_iters=10000, tol=overlap_tol)
            if self.verbose:
                print(f"  Phase 2 done: max overlap = {max_ov:.3f} μm")

        # Phase 3: last resort — micro-steps with very high stiffness
        if max_ov > overlap_tol:
            max_ov = self._relax(
                positions, semi_axes, quats, domain, scale,
                stiffness * 64, damping_coeff * 0.25, max_force * 8,
                orientations, shapes,
                max_iters=20000, tol=overlap_tol)
            if self.verbose:
                print(f"  Phase 3 done: max overlap = {max_ov:.3f} μm")

        if max_ov > overlap_tol and self.verbose:
            print(f"  WARNING: residual overlap {max_ov:.3f} μm > tolerance {overlap_tol:.3f} μm")

        # Scale shapes
        for s in shapes:
            s.a *= scale; s.b *= scale; s.c *= scale
        if self.verbose:
            print(f"  Final packing fraction: {current_phi:.3f}")
        return positions, orientations

    def _forces_numpy(self, positions, semi_axes, quats, domain, scale,
                      stiffness, max_force, orientations, shapes):
        """Pure NumPy fallback for packing forces."""
        n = len(positions)
        forces = np.zeros((n, 3))
        Rs = _batch_rotation_matrices(quats)

        for i in range(n):
            rb_i = np.max(semi_axes[i]) * scale
            for j in range(i+1, n):
                r_ij = positions[j] - positions[i]
                dist = np.linalg.norm(r_ij)
                if dist < 1e-6: continue
                rb_j = np.max(semi_axes[j]) * scale
                if dist > rb_i + rb_j: continue

                d = r_ij / dist
                ri = _effective_radius_np(semi_axes[i]*scale, Rs[i], d)
                rj = _effective_radius_np(semi_axes[j]*scale, Rs[j], -d)

                ov = ri + rj - dist
                if ov > 0:
                    F = min(stiffness*ov, max_force) * d
                    forces[i] -= F; forces[j] += F

        # Walls
        for i in range(n):
            for d_ax in range(3):
                d_lo = np.zeros(3); d_lo[d_ax] = 1.0
                r_lo = _effective_radius_np(semi_axes[i]*scale, Rs[i], d_lo)
                if positions[i, d_ax] < r_lo:
                    forces[i, d_ax] += min(stiffness*(r_lo - positions[i, d_ax]), max_force)
                d_hi = np.zeros(3); d_hi[d_ax] = -1.0
                r_hi = _effective_radius_np(semi_axes[i]*scale, Rs[i], d_hi)
                if positions[i, d_ax] > domain[d_ax] - r_hi:
                    forces[i, d_ax] -= min(stiffness*(positions[i, d_ax] - (domain[d_ax]-r_hi)), max_force)
        return forces


# =============================================================================
# SIMULATION
# =============================================================================

class Simulation:
    def __init__(self, config, config_filepath):
        self.config = config
        self.config_name = Path(config_filepath).stem

        base_dir = Path(config["output"]["base_directory"])
        self.output_dir = base_dir / self.config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        with open(self.output_dir / f"{self.config_name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)

        self.domain_size = config["domain"]["side_length_um"]
        self.domain = np.array([self.domain_size]*3)
        self.n_functional, self.n_inert = calculate_granule_counts(config)
        self.n_total = self.n_functional + self.n_inert

        print(f"\nSimulation: {self.config_name}")
        print(f"  Domain: {self.domain_size:.0f}³ μm")
        print(f"  Granules: {self.n_functional} functional + {self.n_inert} inert = {self.n_total}")
        print(f"  Workers available: {N_WORKERS}")

        self.granules: Optional[GranuleSystem] = None
        self.time = 0.0
        self.step_count = 0
        self.dt = config["time"]["dt_initial_hours"]
        self.history = {
            'time_hours': [], 'n_contacts': [], 'n_bridges': [],
            'mean_coordination': [], 'max_velocity': [], 'packing_fraction': [],
            'max_overlap_um': []
        }

    def setup(self):
        config = self.config
        self.granules = GranuleSystem(self.n_total)

        all_sim, all_true, all_types, all_nc = [], [], [], []

        for _ in range(self.n_functional):
            r = np.clip(np.random.normal(config["functional_granules"]["radius_mean_um"],
                                         config["functional_granules"]["radius_std_um"]),
                        15.0, config["functional_granules"]["radius_mean_um"]*2)
            ss, ts = create_granule_shapes(r, config["functional_granules"])
            all_sim.append(ss); all_true.append(ts)
            all_types.append(0)
            all_nc.append(calculate_cells_on_granule(ss.surface_area, config))

        for _ in range(self.n_inert):
            r = np.clip(np.random.normal(config["inert_granules"]["radius_mean_um"],
                                         config["inert_granules"]["radius_std_um"]),
                        15.0, config["inert_granules"]["radius_mean_um"]*2)
            ss, ts = create_granule_shapes(r, config["inert_granules"])
            all_sim.append(ss); all_true.append(ts)
            all_types.append(1); all_nc.append(0)

        idx = np.random.permutation(self.n_total)
        self.granules.sim_shapes = [all_sim[i] for i in idx]
        self.granules.true_shapes = [all_true[i] for i in idx]
        self.granules.types = np.array([all_types[i] for i in idx], dtype=np.int32)
        self.granules.n_cells = np.array([all_nc[i] for i in idx], dtype=np.int32)
        self.granules.orientations = [Quaternion.random() for _ in range(self.n_total)]

        packer = PackingGenerator(self.domain_size,
                                  config["domain"]["target_packing_fraction"], verbose=True)
        pos, orient = packer.generate(self.n_total, self.granules.sim_shapes,
                                      self.granules.orientations)
        self.granules.positions = pos
        self.granules.orientations = orient

        # Sync true shapes with scaled sim shapes
        for i in range(self.n_total):
            ss = self.granules.sim_shapes[i]
            ts = self.granules.true_shapes[i]
            ts.a, ts.b, ts.c = ss.a, ss.b, ss.c

        for i in range(self.n_total):
            self.granules.drag_coeffs[i] = (config["mechanics"]["damping"] *
                                            self.granules.sim_shapes[i].equivalent_radius / 50.0)

        # Build contiguous arrays
        self.granules.sync_arrays()

        if config["output"]["save_true_shapes"]:
            print("Generating true shape meshes...")
            for ts in self.granules.true_shapes:
                ts.generate_surface_mesh(resolution=16)

        tc = np.sum(self.granules.n_cells)
        print(f"  Total cells: {tc}")
        print(f"  Mean cells per functional: {tc / max(1, self.n_functional):.1f}")
        self._report_overlap_stats("After packing")

    def _report_overlap_stats(self, label=""):
        self.granules.sync_arrays()
        semi = self.granules.semi_axes
        quats = self.granules.quats
        pos = self.granules.positions
        n = self.n_total
        overlaps = []

        if HAS_NUMBA:
            _, max_ov = _packing_forces_numba(pos, semi, quats, self.domain, 1.0, 1.0, 1e6)
            # For detailed stats, do a quick scan
            for i in range(n):
                for j in range(i+1, n):
                    r_ij = pos[j] - pos[i]
                    dist = np.linalg.norm(r_ij)
                    if dist < 1e-6: continue
                    if dist > semi[i].max() + semi[j].max(): continue
                    d = r_ij / dist
                    ri = _effective_radius_np(semi[i], _batch_rotation_matrices(quats[i:i+1])[0], d)
                    rj = _effective_radius_np(semi[j], _batch_rotation_matrices(quats[j:j+1])[0], -d)
                    ov = ri + rj - dist
                    if ov > 0: overlaps.append(ov)
        else:
            Rs = _batch_rotation_matrices(quats)
            for i in range(n):
                for j in range(i+1, n):
                    r_ij = pos[j] - pos[i]
                    dist = np.linalg.norm(r_ij)
                    if dist < 1e-6: continue
                    if dist > semi[i].max() + semi[j].max(): continue
                    d = r_ij / dist
                    ri = _effective_radius_np(semi[i], Rs[i], d)
                    rj = _effective_radius_np(semi[j], Rs[j], -d)
                    ov = ri + rj - dist
                    if ov > 0: overlaps.append(ov)

        if overlaps:
            print(f"  {label} overlaps: {len(overlaps)} pairs, "
                  f"max={max(overlaps):.2f} μm, mean={np.mean(overlaps):.2f} μm")
        else:
            print(f"  {label}: no overlaps detected")

    def _find_contacts_and_bridges(self):
        pos = self.granules.positions
        semi = self.granules.semi_axes
        quats = self.granules.quats
        types = self.granules.types
        n = self.n_total

        # NaN guard
        nan_mask = np.any(np.isnan(pos) | np.isinf(pos), axis=1)
        if np.any(nan_mask):
            for i in np.where(nan_mask)[0]:
                pos[i] = self.domain/2 + np.random.randn(3)*10
                self.granules.velocities[i] = 0

        max_r = np.max(semi)
        max_bridge = self.config["cell_properties"]["max_bridge_gap_um"]
        cutoff = 2*max_r + max_bridge

        # Candidate pairs via spatial index
        if HAS_SCIPY:
            try:
                tree = cKDTree(pos)
                pair_set = tree.query_pairs(cutoff)
                pair_i = np.array([p[0] for p in pair_set], dtype=np.int32)
                pair_j = np.array([p[1] for p in pair_set], dtype=np.int32)
            except:
                pair_i = np.array([i for i in range(n) for j in range(i+1, n)], dtype=np.int32)
                pair_j = np.array([j for i in range(n) for j in range(i+1, n)], dtype=np.int32)
        else:
            pair_i = np.array([i for i in range(n) for j in range(i+1, n)], dtype=np.int32)
            pair_j = np.array([j for i in range(n) for j in range(i+1, n)], dtype=np.int32)

        n_pairs = len(pair_i)
        if n_pairs == 0:
            return [], []

        if HAS_NUMBA:
            result = _find_contacts_numba(
                pos, semi, quats, types, self.granules.n_cells,
                pair_i, pair_j, n_pairs, max_bridge)

            c_i, c_j, c_ov, c_nx, c_ny, c_nz, c_ri, c_rj = result[:8]
            b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj = result[8:]

            contacts = (c_i, c_j, c_ov, c_nx, c_ny, c_nz, c_ri, c_rj)
            bridges = (b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj)
        else:
            contacts, bridges = self._find_contacts_numpy(pair_i, pair_j)

        return contacts, bridges

    def _find_contacts_numpy(self, pair_i, pair_j):
        """Numpy fallback for contact detection."""
        pos = self.granules.positions
        semi = self.granules.semi_axes
        Rs = _batch_rotation_matrices(self.granules.quats)
        types = self.granules.types
        nc = self.granules.n_cells
        mbg = self.config["cell_properties"]["max_bridge_gap_um"]

        contacts_list = []
        bridges_list = []

        for idx in range(len(pair_i)):
            i, j = pair_i[idx], pair_j[idx]
            r_ij = pos[j] - pos[i]
            dist = np.linalg.norm(r_ij)
            if dist < 1e-6: continue
            d = r_ij / dist
            ri = _effective_radius_np(semi[i], Rs[i], d)
            rj = _effective_radius_np(semi[j], Rs[j], -d)
            ov = ri + rj - dist

            if ov > 0:
                contacts_list.append((i, j, ov, d[0], d[1], d[2], ri, rj))

            if types[i] == 0 and types[j] == 0:
                gap = dist - ri - rj
                if gap < mbg:
                    prox = max(0, 1 - max(0, gap)/mbg)
                    n_br = max(1, int(np.sqrt(nc[i]*nc[j])*prox))
                    bridges_list.append((i, j, n_br, dist, d[0], d[1], d[2], ri, rj))

        if contacts_list:
            c = list(zip(*contacts_list))
            contacts = tuple(np.array(x) for x in c)
        else:
            contacts = tuple(np.array([], dtype=t) for t in
                             [np.int32]*2 + [np.float64]*6)

        if bridges_list:
            b = list(zip(*bridges_list))
            bridges = tuple(np.array(x) for x in b)
        else:
            bridges = tuple(np.array([], dtype=t) for t in
                            [np.int32]*2 + [np.int32] + [np.float64]*6)

        return contacts, bridges

    def _compute_forces(self, contacts, bridges):
        config = self.config
        n = self.n_total
        k_rep = config["mechanics"]["repulsion_stiffness"]
        k_wall = config["mechanics"]["wall_stiffness"]
        damping = config["mechanics"]["damping"]
        cell_force = config["cell_properties"]["force_per_cell_nN"]
        cell_diam = config["cell_properties"]["diameter_um"]
        max_force = 100.0
        k_cell = cell_force * 0.03

        if HAS_NUMBA:
            c_i, c_j, c_ov, c_nx, c_ny, c_nz, c_ri, c_rj = contacts
            b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj = bridges

            forces = _compute_contact_forces_numba(
                n, self.granules.positions, self.granules.velocities,
                self.granules.semi_axes, self.granules.quats,
                self.granules.types, self.granules.n_cells,
                self.domain,
                c_i, c_j, c_ov, c_nx, c_ny, c_nz, c_ri, c_rj, len(c_i),
                b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj, len(b_i),
                k_rep, k_wall, damping, k_cell, cell_diam, max_force, 0.03)
        else:
            forces = self._compute_forces_numpy(contacts, bridges,
                                                k_rep, k_wall, damping, k_cell,
                                                cell_diam, max_force)

        # Activity noise on functional (always in Python — cheap)
        func_mask = self.granules.types == 0
        noise = 0.03 * np.sqrt(self.granules.n_cells[func_mask] + 1)
        forces[func_mask] += noise[:, np.newaxis] * np.random.randn(np.sum(func_mask), 3)

        return forces

    def _compute_forces_numpy(self, contacts, bridges, k_rep, k_wall,
                              damping, k_cell, cell_diam, max_force):
        """Numpy fallback for force computation."""
        n = self.n_total
        forces = np.zeros((n, 3))
        pos = self.granules.positions
        vel = self.granules.velocities

        c_i, c_j, c_ov, c_nx, c_ny, c_nz, c_ri, c_rj = contacts
        for idx in range(len(c_i)):
            i, j = int(c_i[idx]), int(c_j[idx])
            ov = c_ov[idx]
            normal = np.array([c_nx[idx], c_ny[idx], c_nz[idx]])
            ri, rj = c_ri[idx], c_rj[idx]
            r_eff = np.sqrt(ri * rj)
            F_mag = min(k_rep * r_eff/50.0 * ov * (1.0 + 2.0*ov/max(r_eff, 1.0)), max_force)
            v_n = np.dot(vel[j] - vel[i], normal)
            F_total = max(0, F_mag - damping*0.5*v_n)
            forces[i] -= F_total * normal
            forces[j] += F_total * normal

        b_i, b_j, b_nc, b_dist, b_dx, b_dy, b_dz, b_ri, b_rj = bridges
        for idx in range(len(b_i)):
            i, j = int(b_i[idx]), int(b_j[idx])
            direction = np.array([b_dx[idx], b_dy[idx], b_dz[idx]])
            gap = b_dist[idx] - b_ri[idx] - b_rj[idx]
            ext = gap - cell_diam
            F = k_cell * b_nc[idx] * ext
            if ext < 0: F *= 0.1
            F = np.clip(F, -max_force/2, max_force/2)
            forces[i] += F * direction
            forces[j] -= F * direction

        # Walls
        semi = self.granules.semi_axes
        Rs = _batch_rotation_matrices(self.granules.quats)
        for i in range(n):
            for d in range(3):
                d_lo = np.zeros(3); d_lo[d] = 1.0
                r_lo = _effective_radius_np(semi[i], Rs[i], d_lo)
                if pos[i, d] < r_lo:
                    ov = r_lo - pos[i, d]
                    forces[i, d] += min(k_wall*ov*(1+ov/max(r_lo, 1)), max_force)
                d_hi = np.zeros(3); d_hi[d] = -1.0
                r_hi = _effective_radius_np(semi[i], Rs[i], d_hi)
                if pos[i, d] > self.domain[d] - r_hi:
                    ov = pos[i, d] - (self.domain[d] - r_hi)
                    forces[i, d] -= min(k_wall*ov*(1+ov/max(r_hi, 1)), max_force)

        # Cap
        f_mags = np.linalg.norm(forces, axis=1, keepdims=True)
        mask = (f_mags > max_force).flatten()
        if np.any(mask):
            forces[mask] *= max_force / f_mags[mask]
        return forces

    def step(self):
        self.granules.sync_arrays()

        contacts, bridges = self._find_contacts_and_bridges()
        forces = self._compute_forces(contacts, bridges)

        velocities = forces / self.granules.drag_coeffs[:, np.newaxis]

        # Cap velocities
        v_mags = np.linalg.norm(velocities, axis=1, keepdims=True)
        fast = (v_mags > 100).flatten()
        if np.any(fast):
            velocities[fast] *= 100.0 / v_mags[fast]

        # Adaptive timestep
        max_vel = np.max(v_mags)
        if max_vel > 1e-10:
            min_r = min(s.equivalent_radius for s in self.granules.sim_shapes)
            self.dt = np.clip(0.03*min_r/max_vel,
                              self.config["time"]["dt_min_hours"],
                              self.config["time"]["dt_max_hours"])
        else:
            self.dt = self.config["time"]["dt_max_hours"]

        self.granules.positions += velocities * self.dt
        self.granules.velocities = velocities

        # Boundary clamp (bounding radius — conservative)
        br = np.array([s.bounding_radius for s in self.granules.sim_shapes])
        for d in range(3):
            self.granules.positions[:, d] = np.clip(
                self.granules.positions[:, d], br + 1, self.domain[d] - br - 1)

        self.time += self.dt
        self.step_count += 1
        return contacts, bridges

    def _record_history(self, contacts, bridges):
        if HAS_NUMBA:
            c_i, c_j, c_ov = contacts[0], contacts[1], contacts[2]
        else:
            c_i, c_j, c_ov = contacts[0], contacts[1], contacts[2]

        nc = len(c_i)
        nb = len(bridges[0]) if bridges else 0

        coord = np.zeros(self.n_total)
        max_ov = 0.0
        for idx in range(nc):
            coord[int(c_i[idx])] += 1
            coord[int(c_j[idx])] += 1
            if c_ov[idx] > max_ov:
                max_ov = c_ov[idx]

        phi = sum(s.volume for s in self.granules.sim_shapes) / np.prod(self.domain)

        self.history['time_hours'].append(self.time)
        self.history['n_contacts'].append(nc)
        self.history['n_bridges'].append(nb)
        self.history['mean_coordination'].append(float(np.mean(coord)))
        self.history['max_velocity'].append(float(np.max(np.linalg.norm(
            self.granules.velocities, axis=1))))
        self.history['packing_fraction'].append(phi)
        self.history['max_overlap_um'].append(float(max_ov))

    def _save_history(self):
        """Write history to disk atomically (write tmp then rename)."""
        target = self.output_dir / f"{self.config_name}_history.json"
        tmp = target.with_suffix('.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(self.history, f)
        tmp.replace(target)  # atomic on same filesystem

    def save_frame(self, label=None):
        if label is None:
            label = f"t{self.time:.1f}h"
        fn = self.output_dir / f"{self.config_name}_frame_{label}.json"
        data = {
            'time_hours': self.time,
            'config_name': self.config_name,
            'n_granules': self.n_total,
            'domain': self.domain.tolist(),
            'positions': self.granules.positions.tolist(),
            'orientations': [q.q.tolist() for q in self.granules.orientations],
            'sim_shapes': [s.to_dict() for s in self.granules.sim_shapes],
            'true_shapes': [s.to_dict() for s in self.granules.true_shapes],
            'types': self.granules.types.tolist(),
            'n_cells': self.granules.n_cells.tolist()
        }
        with open(fn, 'w') as f:
            json.dump(data, f)

    def run(self):
        if self.granules is None:
            self.setup()

        total_h = self.config["time"]["total_hours"]
        save_int = self.config["time"]["save_interval_hours"]

        print(f"\nRunning simulation for {total_h:.0f} hours...")
        t0 = time_module.time()
        last_save = last_rec = 0.0

        self.save_frame("initial")
        self._save_history()  # write initial history so visualizer can start

        if HAS_TQDM:
            pbar = tqdm(total=total_h, desc="Simulating", unit="hr")
            pt = 0.0

        while self.time < total_h:
            contacts, bridges = self.step()

            if self.time - last_rec >= 0.1:
                self._record_history(contacts, bridges)
                last_rec = self.time

            if self.time - last_save >= save_int:
                self.save_frame()
                self._save_history()  # update history on every frame save
                last_save = self.time

            if HAS_TQDM:
                pbar.update(self.time - pt)
                pt = self.time

        if HAS_TQDM:
            pbar.close()

        self.save_frame("final")
        self._save_history()

        elapsed = time_module.time() - t0
        print(f"\nSimulation complete!")
        print(f"  Simulated: {self.time:.1f} hours ({self.time/24:.1f} days)")
        print(f"  Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Output: {self.output_dir}")
        self._report_overlap_stats("Final state")
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("DEM Simulation — Accelerated")
    print("=" * 60)

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        print("\nSelect configuration file...")
        config_path = select_config_file()

    if not config_path:
        print("No config file selected. Exiting.")
        return

    print(f"\nLoading config: {config_path}")
    config = load_config(config_path)
    sim = Simulation(config, config_path)
    sim.run()


if __name__ == "__main__":
    main()