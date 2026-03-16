"""
Overdamped Particle Dynamics Model for Cell-Driven Granular Rearrangement
==========================================================================
V1.6 — Individual Cell Tracking + Data Serialization (no plotting).

MODES:
    "2D"       — Pure 2D simulation with superellipses (V1.3 compatible)
    "2D-slice" — Generate 3D superellipsoid packing, slice at z-midplane, run 2D
    "3D"       — Full 3D volumetric simulation with superellipsoids

MATHEMATICAL MODEL
------------------
N rigid granules (functional or inert) in 2D or 3D, overdamped regime.
Granules maintain their shape; cells bridge nearby functional pairs and
pull them together.

EQUATIONS OF MOTION (overdamped Langevin):
    γ_i dx_i/dt = F_i^contact + F_i^cell + F_i^wall + F_i^noise

3D SHAPE: Superellipsoid
    (|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1
    Spheres: a=b=c, n1=n2=2.  Ellipsoids: n1=n2=2.  Blocky: n1,n2 > 2.

FORCE LAWS:
    (1) Hertzian contact: F = (4/3) E* √R* δ^{3/2}
    (2) Motor-clutch bridging (functional–functional, Chan & Odde 2008)
    (3) Wall repulsion (Hertz, rigid limit)
    (4) Activity noise (cell-driven fluctuations)
    (5) Area-dependent friction (Gong 2006, Pitenis 2014)
    (6) DMT adhesion (Derjaguin-Muller-Toporov)

3D ROTATIONAL DYNAMICS:
    Quaternion orientation (w,x,y,z), overdamped: I_eff dω/dt = τ

TRANSPORT METRICS (V1.4):
    Kozeny-Carman permeability, Darcy number, compaction ratio, porosity

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
from scipy.special import gamma as _gamma, beta as _beta
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum
import time as timer
import os
import json
import csv
import tarfile
import subprocess

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: identity decorator
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    # ── Simulation mode ──
    mode: str = "2D"                # "2D", "2D-slice", or "3D"

    # ── Domain (µm) ──
    Lx: float = 800.0
    Ly: float = 800.0
    Lz: float = 800.0              # depth (used in 3D and 2D-slice modes)
    boundary_mode: str = "walls"    # "walls" (default) or "periodic"

    # ── Granule physical properties ──
    R_func_mean: float = 40.0       # functional radius (µm)
    R_func_std: float = 5.0
    R_inert_mean: float = 60.0      # inert radius (µm)
    R_inert_std: float = 8.0

    # ── Composition ──
    # Option A: set phi_f_target + phi_i_target directly.
    # Option B: set phi_solid_target + func_ratio to derive them.
    #   phi_f = phi_solid * func_ratio, phi_i = phi_solid * (1 - func_ratio)
    # If phi_solid_target is set (> 0), it overrides phi_f/phi_i targets.
    phi_f_target: float = 0.25      # functional area fraction target
    phi_i_target: float = 0.20      # inert area fraction target
    phi_solid_target: float = 0.0   # total solid fraction (0 = use phi_f/phi_i directly)
    func_ratio: float = 0.5         # fraction of solids that are functional (0-1)

    # ── Cell geometry ──
    n_cells_per_granule: int = 8    # cells seeded per functional granule
    cell_diameter: float = 20.0     # µm, initial spherical cell diameter
    cell_height_spread: float = 5.0 # µm, height of spread ellipsoidal cell
    cell_coverage: float = 0.6      # max fraction of granule projected area covered

    # ── Cell attachment & spreading timeline ──
    t_attach_onset: float = 0.0     # hours, when cells begin attaching
    t_attach_half: float = 0.0      # hours after onset for 50% attachment (0 = instant)
    t_spread_duration: float = 3.0  # hours for attached cell to fully spread
    fa_maturation_rate: float = 0.3 # 1/h, focal adhesion maturation rate

    # ── Motor-clutch model (Chan & Odde 2008) ──
    n_motors: int = 50              # motor complexes (stress fibers) per cell
    F_motor_stall: float = 0.5     # nN per complex (500 pN / stress fiber)
    n_clutches: int = 75            # integrin clutch clusters per cell
    k_clutch: float = 5.0          # nN/µm, clutch cluster spring constant
    k_on_clutch: float = 1.0       # 1/s, clutch binding rate
    k_off_clutch: float = 0.1      # 1/s, baseline clutch unbinding rate
    F_bond: float = 0.002          # nN (2 pN), characteristic bond rupture force

    # ── Cell migration ──
    cell_migration_speed: float = 5.0  # µm/h, random walk speed on granule surface

    # ── Cell sensing & bridging ──
    cell_sense_distance: float = 40.0  # µm, filopodia sensing range
    F_max_per_cell: float = 50.0       # nN, absolute max force cap per cell
    L_rest: float = 5.0                # µm, rest length (~ spread cell thickness)

    # ── Bridge formation kinetics ──
    bridge_attempt_rate: float = 0.3   # 1/h, Poisson rate per eligible cell per timestep
    bridge_formation_time: float = 2.0 # h, ramp time for new bridge to reach full force
    bridge_senescence_time: float = 24.0  # h, sustained bridge load → senescence
    min_fa_for_bridge: float = 0.3     # min FA maturity to attempt bridging
    bridge_break_gap: float = 60.0     # µm, gap at which committed bridge ruptures
    bridge_lock_force_threshold: float = 20.0  # nN, force above which bridging cells lock in (won't go senescent)
    bridge_secondary_rate_mult: float = 3.0    # rate multiplier for new bridges when existing bridge present
    expected_bridge_force: float = 100.0       # nN, expected force per bridging cell (diagnostic reference)

    # ── Contact mechanics (Hertzian) ──
    E_modulus: float = 10.0         # kPa, Young's modulus of hydrogel
    poisson_ratio: float = 0.45     # Poisson's ratio (hydrogels ~0.4-0.5)

    # ── Hydrogel friction (area-dependent, Gong 2006 / Pitenis 2014) ──
    tau_0_ii: float = 50.0          # Pa, inert-inert shear stress (bare Gemini gel)
    tau_0_if: float = 500.0         # Pa, inert-functional (bare vs collagen-coated)
    tau_0_ff: float = 2000.0        # Pa, functional-functional (collagen-collagen)
    friction_v_ref: float = 1.0     # µm/h, regularisation velocity (tanh smoothing)

    # ── DMT adhesion (Derjaguin-Muller-Toporov) ──
    W_adh_ii: float = 0.0005       # J/m², inert-inert (bare hydrogel)
    W_adh_if: float = 0.001        # J/m², inert-functional
    W_adh_ff: float = 0.002        # J/m², functional-functional (collagen-collagen)

    # ── Drag ──
    eta: float = 1e-3               # Pa·s (water-like medium)
    drag_scale: float = 0.05        # nondim scaling for drag coefficient

    # ── Active noise ──
    T_active: float = 5.0           # active temperature (nN·µm)

    # ── Time integration ──
    dt: float = 0.5                 # hours
    t_total: float = 72.0           # hours
    save_every_h: float = 2.0       # save interval (hours)
    v_max: float = 20.0             # µm/hr, velocity cap

    # ── Granule shape (V1.3 — 2D superellipses) ──
    shape_enabled: bool = False                # False = circles/spheres (V1.2 compat)
    aspect_ratio_func_mean: float = 1.0        # a/b ratio for functional granules
    aspect_ratio_func_std: float = 0.0
    aspect_ratio_inert_mean: float = 1.0       # a/b ratio for inert granules
    aspect_ratio_inert_std: float = 0.0
    blockiness_func_mean: float = 2.0          # n1 exponent (2=ellipse, >2=blocky)
    blockiness_func_std: float = 0.0
    blockiness_inert_mean: float = 2.0
    blockiness_inert_std: float = 0.0
    drag_scale_rot: float = 0.05               # rotational drag scaling
    omega_max: float = 1.0                     # rad/h, angular velocity cap (2D)

    # ── Granule shape (V1.4 — 3D superellipsoids) ──
    aspect_ratio_c_func_mean: float = 1.0      # c/a ratio for functional (z-elongation)
    aspect_ratio_c_func_std: float = 0.0
    aspect_ratio_c_inert_mean: float = 1.0     # c/a ratio for inert
    aspect_ratio_c_inert_std: float = 0.0
    blockiness_n2_func_mean: float = 2.0       # n2 exponent (meridional blockiness)
    blockiness_n2_func_std: float = 0.0
    blockiness_n2_inert_mean: float = 2.0
    blockiness_n2_inert_std: float = 0.0
    omega_max_3d: float = 1.0                  # rad/h, angular velocity cap per axis (3D)

    # ── Packing ──
    packing_gap: float = 0.0        # µm, min gap between granule surfaces at placement
    boundary_exclusion: float = 0.2  # fraction of domain excluded from each edge for metrics
    packing_settle_steps: int = 200  # compression micro-steps after RSA to achieve contact

    # ── Rendering ──
    Ngrid: int = 200                # grid for 2D field rendering
    Ngrid_3d: int = 80              # grid for 3D field rendering (Ngrid^3 voxels)
    interface_width: float = 3.0    # µm, tanh smoothing

    # ── Performance ──
    use_numba: bool = True          # use Numba JIT if available

    # ── Data serialization (V1.5) ──
    save_data: bool = True              # save simulation data to disk
    save_fields: bool = True            # save phase field grids (needed for visualization)
    output_dir: str = "results/default" # output directory for serialized data
    compress_archive: bool = True       # create .tar.gz at end of simulation

    @property
    def L_max(self):
        """Max interaction distance: max of sensing and bridge break gap."""
        return max(self.cell_sense_distance, self.bridge_break_gap)

    @property
    def save_every(self):
        return max(1, int(self.save_every_h / self.dt))

    def __post_init__(self):
        """Derive phi_f/phi_i from phi_solid + func_ratio if phi_solid_target > 0."""
        if self.phi_solid_target > 0:
            self.phi_f_target = self.phi_solid_target * self.func_ratio
            self.phi_i_target = self.phi_solid_target * (1.0 - self.func_ratio)


# ══════════════════════════════════════════════════════════════════════
# Cell state enum (V1.5)
# ══════════════════════════════════════════════════════════════════════

class CellState(IntEnum):
    """Discrete states for individual cell tracking.  Extensible."""
    ATTACHED = 0        # attached but not yet spreading
    SPREADING = 1       # actively spreading on granule surface
    PROLIFERATING = 2   # fully spread, FA mature
    BRIDGING = 3        # generating traction force across a gap
    SENESCENT = 4       # overcrowded / inactive


# ══════════════════════════════════════════════════════════════════════
# Granule data structure
# ══════════════════════════════════════════════════════════════════════

class GranuleSystem:
    """Tracks all granule and per-granule cell state.  Mode-aware (2D/3D)."""
    def __init__(self, x, y, r, gtype, n_cells,
                 z=None,
                 a=None, b=None, c=None,
                 n_shape=None, n1=None, n2=None,
                 theta=None, quat=None,
                 mode="2D"):
        self.mode = mode
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.z = np.array(z, dtype=np.float64) if z is not None else np.zeros(len(x))
        self.r = np.array(r, dtype=np.float64)   # equivalent radius
        self.gtype = np.array(gtype, dtype=int)   # 0=func, 1=inert
        self.n_cells = np.array(n_cells, dtype=np.float64)
        self.N = len(x)
        self.func_mask = self.gtype == 0
        self.inert_mask = self.gtype == 1

        # ── Shape state ──
        if a is not None:
            self.a = np.array(a, dtype=np.float64)           # semi-axis x
            self.b = np.array(b, dtype=np.float64)           # semi-axis y
            self.c = np.array(c, dtype=np.float64) if c is not None else self.a.copy()
            # n1 = equatorial blockiness, n2 = meridional blockiness
            if n1 is not None:
                self.n1 = np.array(n1, dtype=np.float64)
                self.n2 = np.array(n2, dtype=np.float64)
            elif n_shape is not None:
                self.n1 = np.array(n_shape, dtype=np.float64)
                self.n2 = np.array(n_shape, dtype=np.float64)
            else:
                self.n1 = np.full(self.N, 2.0)
                self.n2 = np.full(self.N, 2.0)
            self.n_shape = self.n1   # backward compat alias for 2D code
            self.is_circle = False
        else:
            self.a = self.r.copy()
            self.b = self.r.copy()
            self.c = self.r.copy()
            self.n1 = np.full(self.N, 2.0)
            self.n2 = np.full(self.N, 2.0)
            self.n_shape = self.n1
            self.is_circle = True

        if mode == "3D":
            self.r_bound = np.maximum(np.maximum(self.a, self.b), self.c)
        else:
            self.r_bound = np.maximum(self.a, self.b)

        # ── Orientation ──
        if mode == "3D":
            if quat is not None:
                self.quat = np.array(quat, dtype=np.float64)
            else:
                self.quat = np.zeros((self.N, 4))
                self.quat[:, 0] = 1.0  # identity quaternion (w,x,y,z)
            self.omega_3d = np.zeros((self.N, 3))   # angular velocity (rad/h)
            self.theta = np.zeros(self.N)  # not used in 3D, but present for compat
            self.omega = np.zeros(self.N)
        else:
            self.theta = np.array(theta, dtype=np.float64) if theta is not None else np.zeros(self.N)
            self.omega = np.zeros(self.N)
            self.quat = None
            self.omega_3d = None

        # ── Cell state (per granule) ──
        self.n_attached = np.zeros(self.N)
        self.spread_fraction = np.zeros(self.N)
        self.fa_maturity = np.zeros(self.N)
        self.n_overcrowded = np.zeros(self.N)

        # ── Velocity state ──
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)
        self.vz = np.zeros(self.N)

        # ── Unwrapped positions (for displacement tracking with periodic BC) ──
        self.x_unwrap = self.x.copy()
        self.y_unwrap = self.y.copy()
        self.z_unwrap = self.z.copy()

        # ── Per-cell tracking (V1.5) ──
        total_cells = int(np.sum(n_cells))
        self.total_cells = total_cells
        # Cell-to-granule offset: cells for granule i live at
        #   cell_offset[i] : cell_offset[i+1]
        offsets = np.zeros(self.N + 1, dtype=int)
        for k in range(self.N):
            offsets[k + 1] = offsets[k] + int(n_cells[k])
        self.cell_offset = offsets
        # Per-cell arrays
        self.cell_granule_id = np.zeros(total_cells, dtype=int)
        self.cell_state = np.full(total_cells, int(CellState.ATTACHED), dtype=int)
        self.cell_theta_local = np.zeros(total_cells, dtype=np.float64)    # 2D surface angle
        self.cell_eta_local = np.zeros(total_cells, dtype=np.float64)      # 3D parametric eta
        self.cell_omega_local = np.zeros(total_cells, dtype=np.float64)    # 3D parametric omega
        self.cell_fx = np.zeros(total_cells, dtype=np.float64)             # force x (nN)
        self.cell_fy = np.zeros(total_cells, dtype=np.float64)             # force y (nN)
        self.cell_fz = np.zeros(total_cells, dtype=np.float64)             # force z (nN)
        self.cell_bridge_target = np.full(total_cells, -1, dtype=int)      # bridge target (-1=none)
        self.cell_contact_area = np.zeros(total_cells, dtype=np.float64)   # footprint area (µm²)
        self.cell_bridge_age = np.zeros(total_cells, dtype=np.float64)     # hours in bridge state

    def positions(self):
        if self.mode == "3D":
            return np.column_stack([self.x, self.y, self.z])
        return np.column_stack([self.x, self.y])

    def cells_on_granule(self, i):
        """Return slice for per-cell arrays corresponding to granule i."""
        return slice(self.cell_offset[i], self.cell_offset[i + 1])

    @property
    def is_3d(self):
        return self.mode == "3D"


# ══════════════════════════════════════════════════════════════════════
# Periodic boundary helpers
# ══════════════════════════════════════════════════════════════════════

def minimum_image_disp(dx, dy, Lx, Ly):
    """Minimum image displacement for 2D periodic boundaries."""
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)
    return dx, dy


def minimum_image_disp_3d(dx, dy, dz, Lx, Ly, Lz):
    """Minimum image displacement for 3D periodic boundaries."""
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)
    dz = dz - Lz * np.round(dz / Lz)
    return dx, dy, dz


def wrap_positions(gs, p):
    """Wrap particle positions into [0, L) using modulo."""
    gs.x[:gs.N] %= p.Lx
    gs.y[:gs.N] %= p.Ly
    if gs.mode == "3D":
        gs.z[:gs.N] %= p.Lz


# ══════════════════════════════════════════════════════════════════════
# Quaternion utilities (V1.4 — 3D rotations)
# ══════════════════════════════════════════════════════════════════════
# Quaternion convention: q = (w, x, y, z) where w is the scalar part.

@njit(cache=True)
def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


@njit(cache=True)
def quat_conjugate(q):
    """Conjugate of quaternion (w, x, y, z) -> (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@njit(cache=True)
def quat_normalize(q):
    """Normalize quaternion to unit length."""
    n = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if n < 1e-30:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


@njit(cache=True)
def quat_rotate(q, v):
    """Rotate vector v (3,) by quaternion q: q v q*."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    r = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return r[1:]


@njit(cache=True)
def quat_rotate_inv(q, v):
    """Inverse rotation: q* v q (world to body)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    r = quat_multiply(quat_multiply(quat_conjugate(q), qv), q)
    return r[1:]


@njit(cache=True)
def quat_to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def quat_from_axis_angle(axis, angle):
    """Quaternion from rotation axis (3,) and angle (radians)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if n < 1e-30:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    ha = angle / 2.0
    return np.array([np.cos(ha), axis[0]*np.sin(ha),
                     axis[1]*np.sin(ha), axis[2]*np.sin(ha)])


def quat_random(rng):
    """Uniformly random unit quaternion."""
    u = rng.random(3)
    q = np.array([
        np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
        np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
        np.sqrt(u[0]) * np.sin(2*np.pi*u[2]),
        np.sqrt(u[0]) * np.cos(2*np.pi*u[2])])
    return quat_normalize(q)


@njit(cache=True)
def quat_integrate(q, omega, dt):
    """
    Integrate quaternion by angular velocity omega (3-vector, rad/h) over dt (h).
    Uses first-order: q_new = q + 0.5 * dt * [0, omega] * q, then normalize.
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    dq = 0.5 * quat_multiply(omega_quat, q) * dt
    q_new = q + dq
    return quat_normalize(q_new)


# ══════════════════════════════════════════════════════════════════════
# Superellipsoid geometry (V1.4 — 3D shapes)
# ══════════════════════════════════════════════════════════════════════
#
# 3D superellipsoid: (|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1
# Parametric surface (eta in [-pi/2, pi/2], omega in [-pi, pi)):
#   x = a * sgnpow(cos(eta), 2/n2) * sgnpow(cos(omega), 2/n1)
#   y = b * sgnpow(cos(eta), 2/n2) * sgnpow(sin(omega), 2/n1)
#   z = c * sgnpow(sin(eta), 2/n2)
# where sgnpow(v, e) = sign(v) * |v|^e

@njit(cache=True)
def _sgnpow(v, e):
    """Signed power: sign(v) * |v|^e, safe for v=0."""
    return np.sign(v) * np.abs(v + 1e-30)**e


def superellipsoid_volume(a, b, c, n1, n2):
    """
    Exact volume of superellipsoid (Jaklic & Leonardis 2000).
    V = 2 * a * b * c * e1 * e2 * B(e1/2 + 1, e1) * B(e2/2, e2/2)
    where e1 = 2/n1, e2 = 2/n2.
    Simplifies for sphere (a=b=c, n1=n2=2): V = (4/3) pi r^3.
    """
    e1 = 2.0 / n1
    e2 = 2.0 / n2
    return (2.0 * a * b * c * e1 * e2 *
            _beta(e1/2.0 + 1.0, e1) * _beta(e2/2.0, e2/2.0))


@njit(cache=True)
def superellipsoid_point(eta, omega, a, b, c, n1, n2):
    """Parametric surface point in body frame."""
    e2 = 2.0 / n2
    e1 = 2.0 / n1
    ce = np.cos(eta)
    se = np.sin(eta)
    co = np.cos(omega)
    so = np.sin(omega)
    x = a * _sgnpow(ce, e2) * _sgnpow(co, e1)
    y = b * _sgnpow(ce, e2) * _sgnpow(so, e1)
    z = c * _sgnpow(se, e2)
    return np.array([x, y, z])


@njit(cache=True)
def superellipsoid_normal(eta, omega, a, b, c, n1, n2):
    """
    Outward unit normal at (eta, omega) in body frame.
    n = (1/a * sgnpow(cos(eta), 2-2/n2) * sgnpow(cos(omega), 2-2/n1),
         1/b * sgnpow(cos(eta), 2-2/n2) * sgnpow(sin(omega), 2-2/n1),
         1/c * sgnpow(sin(eta), 2-2/n2))  normalized.
    """
    e2c = 2.0 - 2.0/n2
    e1c = 2.0 - 2.0/n1
    ce = np.cos(eta)
    se = np.sin(eta)
    co = np.cos(omega)
    so = np.sin(omega)
    nx = (1.0/a) * _sgnpow(ce, e2c) * _sgnpow(co, e1c)
    ny = (1.0/b) * _sgnpow(ce, e2c) * _sgnpow(so, e1c)
    nz = (1.0/c) * _sgnpow(se, e2c)
    mag = np.sqrt(nx*nx + ny*ny + nz*nz)
    if mag < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return np.array([nx/mag, ny/mag, nz/mag])


@njit(cache=True)
def superellipsoid_curvature_radii(eta, omega, a, b, c, n1, n2):
    """
    Approximate principal radii of curvature at (eta, omega).
    Returns (R1, R2).  For Hertz: R_eff = sqrt(R1*R2) for each body,
    then combined R* = R_eff_i * R_eff_j / (R_eff_i + R_eff_j).
    """
    eps = 1e-8
    # Finite difference approach: sample surface nearby and fit curvature
    pt0 = superellipsoid_point(eta, omega, a, b, c, n1, n2)
    n0 = superellipsoid_normal(eta, omega, a, b, c, n1, n2)

    # Sample in eta and omega directions
    deta = 0.01
    domega = 0.01
    pt_de = superellipsoid_point(eta + deta, omega, a, b, c, n1, n2)
    pt_do = superellipsoid_point(eta, omega + domega, a, b, c, n1, n2)
    n_de = superellipsoid_normal(eta + deta, omega, a, b, c, n1, n2)
    n_do = superellipsoid_normal(eta, omega + domega, a, b, c, n1, n2)

    # Curvature ~ |dn/ds| where ds is arc length
    ds_eta = np.linalg.norm(pt_de - pt0)
    ds_omega = np.linalg.norm(pt_do - pt0)

    if ds_eta > eps:
        dn_eta = np.linalg.norm(n_de - n0)
        kappa1 = dn_eta / ds_eta
    else:
        kappa1 = 1.0 / max(a, b, c)

    if ds_omega > eps:
        dn_omega = np.linalg.norm(n_do - n0)
        kappa2 = dn_omega / ds_omega
    else:
        kappa2 = 1.0 / max(a, b, c)

    R1 = 1.0 / max(kappa1, 1e-6)
    R2 = 1.0 / max(kappa2, 1e-6)
    return R1, R2


@njit(cache=True)
def superellipsoid_implicit(bx, by, bz, a, b, c, n1, n2):
    """
    Implicit function value in body frame.
    Returns < 1 inside, = 1 on boundary, > 1 outside.
    """
    return ((np.abs(bx/a)**n1 + np.abs(by/b)**n1)**(n2/n1) +
            np.abs(bz/c)**n2)


def superellipsoid_implicit_world(px, py, pz, cx, cy, cz, a, b, c, n1, n2, quat):
    """Implicit function value at world point (px, py, pz)."""
    body = quat_rotate_inv(quat, np.array([px-cx, py-cy, pz-cz]))
    return superellipsoid_implicit(body[0], body[1], body[2], a, b, c, n1, n2)


def superellipsoid_mesh(a, b, c, n1, n2, n_pts=24):
    """
    Generate triangle mesh vertices for a superellipsoid (body frame).
    Returns vertices array of shape (n_pts*n_pts, 3).
    """
    eta = np.linspace(-np.pi/2, np.pi/2, n_pts)
    omega = np.linspace(-np.pi, np.pi, n_pts)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e2 = 2.0 / n2
    e1 = 2.0 / n1
    X = a * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    Y = b * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    Z = c * _sgnpow(np.sin(E), e2)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


# ══════════════════════════════════════════════════════════════════════
# Hertz contact mechanics
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def hertz_contact_force(E_star_Pa, R_eff_um, delta_um):
    """
    Hertzian normal contact force between two elastic spheres.

    F = (4/3) E* √R* δ^{3/2}   [SI: Pa, m, m → N]

    With simulation units (µm, nN):
        F_nN = (4/3) · E*_Pa · √(R*_µm) · δ_µm^{3/2} · 10⁻³
    """
    return (4.0 / 3.0) * E_star_Pa * np.sqrt(R_eff_um) * delta_um**1.5 * 1e-3


def get_pair_friction_params(gtype_i, gtype_j, p: Params):
    """
    Return (tau_0, W_adhesion) for a granule pair based on surface types.

    Functional granules (type 0) are collagen-I coated; inert (type 1) are
    bare hydrogel.  Three regimes:
        inert–inert:           low friction, low adhesion  (bare Gemini gel)
        inert–functional:      moderate  (asymmetric collagen adsorption)
        functional–functional: high  (mutual collagen H-bonding/entanglement)
    """
    if gtype_i == 1 and gtype_j == 1:       # inert–inert
        return p.tau_0_ii, p.W_adh_ii
    elif gtype_i == 0 and gtype_j == 0:     # functional–functional
        return p.tau_0_ff, p.W_adh_ff
    else:                                    # mixed
        return p.tau_0_if, p.W_adh_if


# ══════════════════════════════════════════════════════════════════════
# Superellipse geometry (V1.3)
# ══════════════════════════════════════════════════════════════════════
#
# A 2D superellipse: |x/a|^n + |y/b|^n = 1
# Parametric form:
#   x(t) = a |cos t|^(2/n) sign(cos t)
#   y(t) = b |sin t|^(2/n) sign(sin t)      t in [0, 2pi)

def superellipse_area(a, b, n):
    """Exact area of superellipse |x/a|^n + |y/b|^n = 1."""
    return 4.0 * a * b * _gamma(1.0 + 1.0/n)**2 / _gamma(1.0 + 2.0/n)


@njit(cache=True)
def superellipse_point(t, a, b, n):
    """Point (x, y) on superellipse boundary at parameter t (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    x = a * np.sign(ct) * np.abs(ct)**e
    y = b * np.sign(st) * np.abs(st)**e
    return x, y


@njit(cache=True)
def superellipse_tangent(t, a, b, n):
    """Unnormalised tangent vector dx/dt, dy/dt (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    # dx/dt = a * (2/n) * |cos t|^(2/n - 1) * sin t  (with signs handled)
    dxdt = -a * e * np.sign(ct) * np.abs(ct)**(e - 1.0) * st
    dydt =  b * e * np.sign(st) * np.abs(st)**(e - 1.0) * ct
    return dxdt, dydt


@njit(cache=True)
def superellipse_normal_vec(t, a, b, n):
    """Outward unit normal at parameter t (body frame)."""
    dxdt, dydt = superellipse_tangent(t, a, b, n)
    # outward normal = (dy/dt, -dx/dt) normalised
    nx, ny = dydt, -dxdt
    mag = np.sqrt(nx*nx + ny*ny)
    if mag < 1e-30:
        return 0.0, 0.0
    return nx / mag, ny / mag


@njit(cache=True)
def superellipse_curvature_radius(t, a, b, n):
    """
    Local radius of curvature R = 1/kappa at parameter t (body frame).

    kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    """
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    eps = 1e-30

    # First derivatives
    act = np.abs(ct) + eps
    ast = np.abs(st) + eps
    dxdt = -a * e * np.sign(ct) * act**(e - 1.0) * st
    dydt =  b * e * np.sign(st) * ast**(e - 1.0) * ct

    # Second derivatives
    d2xdt2 = -a * e * (
        np.sign(ct) * (e - 1.0) * act**(e - 2.0) * st * st
        + np.sign(ct) * act**(e - 1.0) * ct
    )
    d2ydt2 = b * e * (
        np.sign(st) * (e - 1.0) * ast**(e - 2.0) * ct * ct
        - np.sign(st) * ast**(e - 1.0) * st
    )

    num = abs(dxdt * d2ydt2 - dydt * d2xdt2)
    den = (dxdt**2 + dydt**2)**1.5
    if den < 1e-30 or num < 1e-30:
        return max(a, b) * 10.0  # fallback: large radius (nearly flat)
    return den / num


def superellipse_polygon_pts(cx, cy, a, b, n, theta, num_pts=64):
    """Polygon vertices for rendering a superellipse in world coords."""
    t = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
    e = 2.0 / n
    ct, st = np.cos(t), np.sin(t)
    bx = a * np.sign(ct) * np.abs(ct)**e
    by = b * np.sign(st) * np.abs(st)**e
    # Rotate to world frame
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    wx = cx + cos_th * bx - sin_th * by
    wy = cy + sin_th * bx + cos_th * by
    return np.column_stack([wx, wy])


def superellipse_perimeter(a, b, n, num_pts=256):
    """Numerical perimeter of a superellipse."""
    pts = superellipse_polygon_pts(0, 0, a, b, n, 0.0, num_pts)
    diffs = np.diff(pts, axis=0, append=pts[:1])
    return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))


@njit(cache=True)
def _world_to_body(px, py, cx, cy, theta):
    """Transform world point(s) to body frame of granule at (cx,cy,theta)."""
    dx, dy = px - cx, py - cy
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cos_th * dx + sin_th * dy, -sin_th * dx + cos_th * dy


@njit(cache=True)
def _body_to_world(bx, by, cx, cy, theta):
    """Transform body point(s) to world frame."""
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cx + cos_th * bx - sin_th * by, cy + sin_th * bx + cos_th * by


@njit(cache=True)
def superellipse_implicit(px, py, cx, cy, a, b, n, theta):
    """
    Evaluate the superellipse implicit function at world point (px, py).
    Returns value < 1 if inside, = 1 on boundary, > 1 outside.
    """
    bx, by = _world_to_body(px, py, cx, cy, theta)
    return (np.abs(bx) / a)**n + (np.abs(by) / b)**n


def _find_closest_param(bx_target, by_target, a, b, n, t_guess=None):
    """
    Find parameter t such that the superellipse point is closest to (bx_target, by_target).
    Used for contact detection. Minimises distance in body frame.
    """
    from scipy.optimize import minimize_scalar

    def dist2(t):
        px, py = superellipse_point(t, a, b, n)
        return (px - bx_target)**2 + (py - by_target)**2

    if t_guess is not None:
        # Local search near guess
        res = minimize_scalar(dist2, bounds=(t_guess - 0.5, t_guess + 0.5),
                              method='bounded')
        return res.x % (2 * np.pi)

    # Global search: sample then refine
    ts = np.linspace(0, 2*np.pi, 64, endpoint=False)
    d2 = np.array([dist2(t) for t in ts])
    best = ts[np.argmin(d2)]
    res = minimize_scalar(dist2, bounds=(best - 0.15, best + 0.15),
                          method='bounded')
    return res.x % (2 * np.pi)


# ── Contact detection: superellipse–superellipse (common normal) ──

@njit(cache=True)
def find_contact_superellipses(xi, yi, ai, bi, ni, thetai,
                                xj, yj, aj, bj, nj, thetaj):
    """
    Common normal contact detection between two superellipses.

    Returns: (in_contact, delta, nx, ny, cx, cy, R_loc_i, R_loc_j)
        in_contact: bool
        delta: penetration depth (µm), > 0 if overlapping
        nx, ny: unit contact normal (from i toward j)
        cx, cy: contact point (world coords)
        R_loc_i, R_loc_j: local curvature radii at contact (µm)

    Returns None if no contact (gap > 0).
    """
    # Direction between centres
    dx_c = xj - xi
    dy_c = yj - yi
    d_c = np.sqrt(dx_c**2 + dy_c**2)
    if d_c < 1e-12:
        return None

    # Unit vector from i to j
    ux, uy = dx_c / d_c, dy_c / d_c

    # Initial guess: parameter on each surface closest to the line of centres
    # Transform centre-to-centre direction into each body frame
    bx_i, by_i = (np.cos(thetai) * ux + np.sin(thetai) * uy,
                  -np.sin(thetai) * ux + np.cos(thetai) * uy)
    bx_j, by_j = (np.cos(thetaj) * (-ux) + np.sin(thetaj) * (-uy),
                  -np.sin(thetaj) * (-ux) + np.cos(thetaj) * (-uy))

    t_i = np.arctan2(by_i, bx_i)
    t_j = np.arctan2(by_j, bx_j)

    # Newton-Raphson iteration: refine contact parameters
    # Find points P_i on surface i and P_j on surface j such that the
    # normal at P_i points toward P_j and vice versa.
    for iteration in range(12):
        # Points on each surface (body frame)
        px_i, py_i = superellipse_point(t_i, ai, bi, ni)
        px_j, py_j = superellipse_point(t_j, aj, bj, nj)

        # Transform to world frame
        wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
        wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

        # Vector from P_i to P_j
        dpx = wx_j - wx_i
        dpy = wy_j - wy_i
        dp_mag = np.sqrt(dpx**2 + dpy**2)
        if dp_mag < 1e-12:
            break

        # Normal at P_i (body frame) then rotate to world
        nix, niy = superellipse_normal_vec(t_i, ai, bi, ni)
        cos_ti, sin_ti = np.cos(thetai), np.sin(thetai)
        nix_w = cos_ti * nix - sin_ti * niy
        niy_w = sin_ti * nix + cos_ti * niy

        # Normal at P_j (body frame) then rotate to world
        njx, njy = superellipse_normal_vec(t_j, aj, bj, nj)
        cos_tj, sin_tj = np.cos(thetaj), np.sin(thetaj)
        njx_w = cos_tj * njx - sin_tj * njy
        njy_w = sin_tj * njx + cos_tj * njy

        # Desired: n_i should point along P_i→P_j, n_j should point along P_j→P_i
        # Error: angle between n_i and (P_i→P_j)
        target_nx = dpx / dp_mag
        target_ny = dpy / dp_mag

        # Cross-product errors (should be zero when aligned)
        err_i = nix_w * target_ny - niy_w * target_nx
        err_j = njx_w * (-target_ny) - njy_w * (-target_nx)

        if abs(err_i) < 1e-8 and abs(err_j) < 1e-8:
            break

        # Simple gradient step (damped Newton)
        t_i -= 0.5 * err_i
        t_j -= 0.5 * err_j

    # Final contact geometry
    px_i, py_i = superellipse_point(t_i, ai, bi, ni)
    px_j, py_j = superellipse_point(t_j, aj, bj, nj)
    wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
    wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

    # Contact normal: from surface point i toward surface point j
    cpx = wx_j - wx_i
    cpy = wy_j - wy_i
    cp_mag = np.sqrt(cpx**2 + cpy**2)

    # Check if P_i is inside body j (overlap)
    val_j = superellipse_implicit(wx_i, wy_i, xj, yj, aj, bj, nj, thetaj)
    val_i = superellipse_implicit(wx_j, wy_j, xi, yi, ai, bi, ni, thetai)

    if val_j > 1.0 and val_i > 1.0:
        # No overlap
        return None

    # Penetration depth: distance between surface points (with sign)
    delta = cp_mag
    if cp_mag < 1e-12:
        # Coincident surface points — use implicit function for depth estimate
        delta = max(ai, bi) * (1.0 - val_j**(1.0/nj)) if val_j < 1.0 else 0.01

    # Contact normal (from i toward j, along line of centres as fallback)
    if cp_mag > 1e-12:
        nx, ny = cpx / cp_mag, cpy / cp_mag
    else:
        nx, ny = ux, uy

    # Contact point: midpoint of the two surface points
    contact_x = 0.5 * (wx_i + wx_j)
    contact_y = 0.5 * (wy_i + wy_j)

    # Local curvature radii
    R_loc_i = superellipse_curvature_radius(t_i, ai, bi, ni)
    R_loc_j = superellipse_curvature_radius(t_j, aj, bj, nj)

    return (True, delta, nx, ny, contact_x, contact_y, R_loc_i, R_loc_j)


# ── Contact detection: superellipse–wall ──

@njit(cache=True)
def find_contact_superellipse_wall(xi, yi, ai, bi, ni, thetai,
                                    wall_pos, wall_axis, wall_sign):
    """
    Contact detection between a superellipse and a flat wall.

    wall_axis: 0 = vertical wall (x = wall_pos), 1 = horizontal wall (y = wall_pos)
    wall_sign: +1 if granule should be to the right/above wall_pos,
               -1 if granule should be to the left/below wall_pos

    Returns: (penetration, R_local) or None if no contact.
        penetration: > 0 if overlapping
        R_local: curvature radius at contact point
    """
    # Find the extreme point of the superellipse in the wall-normal direction
    # Sample boundary and find the point closest to the wall
    n_sample = 64
    t_vals = np.linspace(0, 2*np.pi, n_sample, endpoint=False)
    e = 2.0 / ni
    ct, st = np.cos(t_vals), np.sin(t_vals)
    bx = ai * np.sign(ct) * np.abs(ct)**e
    by = bi * np.sign(st) * np.abs(st)**e
    cos_th, sin_th = np.cos(thetai), np.sin(thetai)
    wx = xi + cos_th * bx - sin_th * by
    wy = yi + sin_th * bx + cos_th * by

    if wall_axis == 0:
        coords = wx
    else:
        coords = wy

    if wall_sign > 0:
        # Wall at low side: penetration = wall_pos - min(coords)
        idx = np.argmin(coords)
        pen = wall_pos - coords[idx]
    else:
        # Wall at high side: penetration = max(coords) - wall_pos
        idx = np.argmax(coords)
        pen = coords[idx] - wall_pos

    if pen <= 0:
        return None

    # Refine: local curvature at the contact point
    t_contact = t_vals[idx]
    R_local = superellipse_curvature_radius(t_contact, ai, bi, ni)

    return (pen, R_local)


def overlap_lens_area(R1, R2, d):
    """
    Area of the lens-shaped intersection of two 2D circles.

    R1, R2 : radii (µm)
    d      : centre-to-centre distance (µm)
    Returns area in µm².
    """
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return np.pi * min(R1, R2)**2
    cos_a1 = np.clip((d*d + R1*R1 - R2*R2) / (2.0 * d * R1), -1, 1)
    cos_a2 = np.clip((d*d + R2*R2 - R1*R1) / (2.0 * d * R2), -1, 1)
    arg = (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)
    return R1*R1 * np.arccos(cos_a1) + R2*R2 * np.arccos(cos_a2) \
           - 0.5 * np.sqrt(max(0.0, arg))


def overlap_lens_volume(R1, R2, d):
    """
    Volume of the lens-shaped intersection of two 3D spheres.

    R1, R2 : radii (µm)
    d      : centre-to-centre distance (µm)
    Returns volume in µm³.
    """
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    if d < 1e-12:
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    # V = π(R1+R2-d)²(d² + 2d(R1+R2) - 3(R1-R2)²) / (12d)
    s = R1 + R2 - d
    numer = s * s * (d * d + 2.0 * d * (R1 + R2) - 3.0 * (R1 - R2)**2)
    return np.pi * numer / (12.0 * d)


def compute_effective_radii_3d(gs):
    """
    Volume-conserving effective radii for 3D spheres.

    When DEM spheres overlap, the lens-shaped intersection is material
    that is geometrically double-counted.  To conserve volume, each
    granule's display radius is inflated:

        4/3 π r_eff³ = 4/3 π r³ + ΔV_i

    where ΔV_i is granule i's share of its total overlap volume, split
    proportionally to r³.

    Returns (r_eff, overlap_volume_per_granule).
    """
    pos = gs.positions()
    max_r = np.max(gs.r)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_vol = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        d = np.sqrt(np.dot(dv, dv))
        if d < gs.r[i] + gs.r[j]:
            V_lens = overlap_lens_volume(gs.r[i], gs.r[j], d)
            frac_i = gs.r[i]**3 / (gs.r[i]**3 + gs.r[j]**3)
            overlap_vol[i] += frac_i * V_lens
            overlap_vol[j] += (1.0 - frac_i) * V_lens

    r_eff = (gs.r**3 + 3.0 * overlap_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return r_eff, overlap_vol


def compute_effective_radii(gs):
    """
    Volume-conserving effective radii (2D).

    When DEM circles overlap, the lens-shaped intersection is material
    that is geometrically double-counted.  To conserve 2D area (proxy
    for 3D volume), each granule's display radius is inflated:

        π r_eff² = π r² + ΔA_i

    where ΔA_i is granule i's share of its total overlap area, split
    proportionally to r².

    Returns (r_eff, overlap_area_per_granule).
    """
    pos = gs.positions()
    max_r = np.max(gs.r)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_area = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < gs.r[i] + gs.r[j]:
            A_lens = overlap_lens_area(gs.r[i], gs.r[j], d)
            frac_i = gs.r[i]**2 / (gs.r[i]**2 + gs.r[j]**2)
            overlap_area[i] += frac_i * A_lens
            overlap_area[j] += (1.0 - frac_i) * A_lens

    r_eff = np.sqrt(gs.r**2 + overlap_area / np.pi)
    return r_eff, overlap_area


def print_stiffness_info(p: Params):
    """Show expected overlap for the configured modulus."""
    E_star = (p.E_modulus * 1e3) / (2.0 * (1.0 - p.poisson_ratio**2))
    R_eff = p.R_func_mean / 2.0  # two same-size functional granules
    F_typ = p.F_max_per_cell

    coeff = (4.0 / 3.0) * E_star * np.sqrt(R_eff) * 1e-3
    delta_eq = (F_typ / coeff) ** (2.0 / 3.0)

    print(f"  Material: E = {p.E_modulus} kPa, ν = {p.poisson_ratio}")
    print(f"    E* = {E_star:.0f} Pa (effective modulus)")
    print(f"    For F = {F_typ:.0f} nN between R = {p.R_func_mean:.0f} µm granules:")
    print(f"    → Equilibrium overlap δ = {delta_eq:.2f} µm "
          f"({delta_eq / p.R_func_mean * 100:.1f}% of R)")


# ══════════════════════════════════════════════════════════════════════
# Cell geometry & motor-clutch model
# ══════════════════════════════════════════════════════════════════════

def cell_projected_area(spread_frac, cell_d, cell_h):
    """
    Projected area of a cell transitioning from sphere to spread ellipsoid.

    Sphere (spread_frac=0):
        A = π (d/2)²

    Fully spread oblate ellipsoid (spread_frac=1):
        Volume conserved: V = (4/3)π(d/2)³ = (4/3)π a² (h/2)
        → a² = d³ / (4h)
        → A_spread = π a² = π d³ / (4h)

    Interpolates linearly between the two based on spread_frac.
    """
    r = cell_d / 2.0
    A_sphere = np.pi * r ** 2

    V = (4.0 / 3.0) * np.pi * r ** 3
    c = cell_h / 2.0
    a2 = 3.0 * V / (4.0 * np.pi * c)  # semi-major axis squared
    A_spread = np.pi * a2

    return A_sphere + spread_frac * (A_spread - A_sphere)


def max_cells_on_granule(R, cell_proj_area, coverage):
    """Max cells that fit on a granule based on projected area ratio."""
    A_granule = np.pi * R ** 2
    return max(1, int(A_granule * coverage / cell_proj_area))


def motor_clutch_force(E_kPa, p: Params, fa_maturity_val):
    """
    Steady-state traction force per cell from the motor-clutch model.

    Based on Chan & Odde (2008).  For hydrogel substrates (1–100 kPa) we
    are in the rising portion of the stiffness–force curve:

        F = F_stall · k_sub / (k_sub + k_opt) · engagement · fa_maturity

    where
        k_sub = π E a / (1−ν²)        substrate stiffness at cell scale
        k_opt = n_clutches · k_clutch  optimal (clutch ensemble) stiffness
        engagement = k_on / (k_on + k_off)   steady-state clutch fraction

    Returns force per cell in nN.
    """
    # Total motor stall force
    F_stall = p.n_motors * p.F_motor_stall  # nN

    # Substrate stiffness at cell scale (nN/µm)
    # k = π E a / (1-ν²),  E [kPa] × a [µm] → kPa·µm = nN/µm
    a_cell = p.cell_diameter / 2.0
    k_sub = np.pi * E_kPa * a_cell / (1.0 - p.poisson_ratio ** 2)

    # Clutch ensemble stiffness
    k_opt = p.n_clutches * p.k_clutch

    # Steady-state engagement fraction
    engagement = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)

    # Motor-clutch force
    F_mc = F_stall * (k_sub / (k_sub + k_opt)) * engagement * fa_maturity_val

    return min(F_mc, p.F_max_per_cell)


def update_cell_state(gs: GranuleSystem, p: Params, t: float, rng=None):
    """
    Advance per-granule cell state for current simulation time.

    Timeline on each functional granule:
      t < t_attach_onset:          cells sit as spheres, no attachment
      t ≥ t_attach_onset:          cells attach (sigmoidal, half-time t_attach_half)
      after attachment:             cells spread (sphere → ellipsoid over t_spread_duration)
      during/after spreading:       focal adhesions mature at fa_maturation_rate
      if spread cells overcrowd:    excess cells crawl on top of others

    Stiffness-dependent spreading: stiffer substrates → faster spreading
    (motor-clutch effect on cell mechanotransduction).
    """
    # Stiffness modulates spreading speed
    a_cell = p.cell_diameter / 2.0
    k_sub = np.pi * p.E_modulus * a_cell / (1.0 - p.poisson_ratio ** 2)
    k_opt = p.n_clutches * p.k_clutch
    stiffness_factor = k_sub / (k_sub + k_opt)  # 0 on very soft, ~1 on stiff
    # Effective spread duration (faster on stiffer substrates)
    eff_spread_dur = p.t_spread_duration / max(0.2, stiffness_factor)

    for i in range(gs.N):
        if gs.gtype[i] != 0:
            continue

        # ── Attachment (sigmoidal kinetics or instant) ──
        if t >= p.t_attach_onset:
            if p.t_attach_half <= 0.01:
                # Instant full attachment
                frac = 1.0
            else:
                tau = t - p.t_attach_onset
                frac = 1.0 / (1.0 + np.exp(
                    -3.0 * (tau - p.t_attach_half) / max(0.1, p.t_attach_half)))
            gs.n_attached[i] = gs.n_cells[i] * frac
        else:
            gs.n_attached[i] = 0.0

        # ── Spreading (linear ramp after attachment) ──
        if gs.n_attached[i] > 0.5:
            t_since = max(0.0, t - p.t_attach_onset - p.t_attach_half)
            gs.spread_fraction[i] = min(1.0, t_since / eff_spread_dur)
        else:
            gs.spread_fraction[i] = 0.0

        # ── Focal adhesion maturation ──
        if gs.spread_fraction[i] > 0.1:
            t_since_spread = max(0.0, t - (p.t_attach_onset + p.t_attach_half))
            gs.fa_maturity[i] = min(1.0, t_since_spread * p.fa_maturation_rate)
        else:
            gs.fa_maturity[i] = 0.0

        # ── Overcrowding check ──
        A_cell = cell_projected_area(
            gs.spread_fraction[i], p.cell_diameter, p.cell_height_spread)
        cap = max_cells_on_granule(gs.r[i], A_cell, p.cell_coverage)
        gs.n_overcrowded[i] = max(0.0, gs.n_attached[i] - cap)

    # ── V1.5: Per-cell state tracking ──
    _update_individual_cells(gs, p, rng)


def _update_individual_cells(gs: GranuleSystem, p: Params, rng=None):
    """Map per-granule aggregate cell state to individual cell states.

    Committed BRIDGING cells are preserved — their bridge_age is incremented
    and they transition to SENESCENT after sustained load (bridge_senescence_time).
    New bridges are initiated probabilistically during force computation.

    Also applies random-walk migration for mobile cells (ATTACHED, SPREADING,
    PROLIFERATING) on the granule surface.  BRIDGING and SENESCENT cells
    do not migrate.
    """
    mobile_states = {int(CellState.ATTACHED), int(CellState.SPREADING),
                     int(CellState.PROLIFERATING)}

    for i in range(gs.N):
        if gs.gtype[i] != 0:
            continue
        n_total = gs.cell_offset[i + 1] - gs.cell_offset[i]
        if n_total == 0:
            continue

        n_att = int(round(gs.n_attached[i]))
        n_over = int(round(gs.n_overcrowded[i]))
        sf = gs.spread_fraction[i]
        fa = gs.fa_maturity[i]

        # Compute per-cell contact area from current spread state
        A_cell = cell_projected_area(sf, p.cell_diameter, p.cell_height_spread)

        for k in range(n_total):
            ci = gs.cell_offset[i] + k
            gs.cell_contact_area[ci] = A_cell if k < n_att else 0.0

            # ── Preserve committed bridges ──
            if gs.cell_state[ci] == int(CellState.BRIDGING):
                gs.cell_bridge_age[ci] += p.dt
                # Check force magnitude from previous timestep (still in cell_fx/fy/fz)
                F_mag = np.sqrt(gs.cell_fx[ci]**2 + gs.cell_fy[ci]**2
                                + gs.cell_fz[ci]**2)
                # High-force bridges lock in: cell prefers to stay bridging
                if F_mag >= p.bridge_lock_force_threshold:
                    continue  # locked in, skip senescence check
                # Sustained mechanical load without lock-in → senescence
                if gs.cell_bridge_age[ci] >= p.bridge_senescence_time:
                    gs.cell_state[ci] = int(CellState.SENESCENT)
                    gs.cell_bridge_target[ci] = -1
                    gs.cell_bridge_age[ci] = 0.0
                continue  # don't overwrite bridge state

            # ── Already senescent stays senescent ──
            if gs.cell_state[ci] == int(CellState.SENESCENT):
                continue

            # ── Normal state assignment ──
            if k >= n_att:
                gs.cell_state[ci] = int(CellState.SENESCENT)
            elif k >= (n_att - n_over) and n_over > 0:
                gs.cell_state[ci] = int(CellState.SENESCENT)
            elif sf >= 0.95 and fa >= 0.5:
                gs.cell_state[ci] = int(CellState.PROLIFERATING)
            elif sf > 0.0:
                gs.cell_state[ci] = int(CellState.SPREADING)
            else:
                gs.cell_state[ci] = int(CellState.ATTACHED)

        # ── Cell migration: random walk on granule surface ──
        if rng is not None and p.cell_migration_speed > 0:
            r_eff = max(gs.r[i], 1.0)
            sigma = p.cell_migration_speed * p.dt / r_eff
            for k in range(n_total):
                ci = gs.cell_offset[i] + k
                if gs.cell_state[ci] not in mobile_states:
                    continue
                if gs.mode == "3D":
                    gs.cell_eta_local[ci] += rng.normal(0, sigma)
                    gs.cell_omega_local[ci] += rng.normal(0, sigma)
                    # Clamp eta to avoid poles
                    gs.cell_eta_local[ci] = np.clip(
                        gs.cell_eta_local[ci], -0.85 * np.pi / 2, 0.85 * np.pi / 2)
                    gs.cell_omega_local[ci] %= (2 * np.pi)
                else:
                    gs.cell_theta_local[ci] += rng.normal(0, sigma)
                    gs.cell_theta_local[ci] %= (2 * np.pi)


# ══════════════════════════════════════════════════════════════════════
# Packing settle (compression to achieve granule contact)
# ══════════════════════════════════════════════════════════════════════

def _settle_packing_2d(gs, p):
    """Run short isotropic compression micro-steps to bring granules into contact.

    Uses a centroid-attraction + Hertz repulsion scheme:
    each granule is gently pushed toward the domain centre while
    overlapping neighbours are repelled. This produces a jammed
    packing where most granules are touching at least one neighbour.

    For periodic boundaries, centripetal attraction is replaced with
    random perturbation (no preferred centre), and positions are wrapped.
    """
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    cx_dom, cy_dom = p.Lx / 2, p.Ly / 2
    dt_settle = 0.02  # micro-step size (µm per step)
    repulsion_k = 2.0  # overlap repulsion strength

    print(f"  Settling packing ({p.packing_settle_steps} steps)...", end="", flush=True)

    rng_settle = np.random.default_rng(12345)

    for step_i in range(p.packing_settle_steps):
        pos = gs.positions()  # (N, 2)
        fx = np.zeros(N)
        fy = np.zeros(N)

        if periodic:
            # Small random jitter instead of centripetal attraction
            jitter = 0.3 * max(0.5 * (1.0 - step_i / p.packing_settle_steps), 0.05)
            fx += jitter * rng_settle.standard_normal(N)
            fy += jitter * rng_settle.standard_normal(N)
        else:
            # Gentle centripetal attraction (decays with step count)
            attract = max(0.5 * (1.0 - step_i / p.packing_settle_steps), 0.05)
            for i in range(N):
                dx = cx_dom - pos[i, 0]
                dy = cy_dom - pos[i, 1]
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                fx[i] += attract * dx / d * gs.r_bound[i]
                fy[i] += attract * dy / d * gs.r_bound[i]

        # Pairwise repulsion for overlapping bounding spheres
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
        else:
            tree = cKDTree(pos)
        max_rb = float(np.max(gs.r_bound))
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            if periodic:
                dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            overlap = gs.r_bound[i] + gs.r_bound[j] - d
            if overlap > 0:
                # Push apart proportional to overlap
                nx, ny = dx / d, dy / d
                f_rep = repulsion_k * overlap
                fx[i] -= f_rep * nx
                fy[i] -= f_rep * ny
                fx[j] += f_rep * nx
                fy[j] += f_rep * ny

        # Move
        gs.x += fx * dt_settle
        gs.y += fy * dt_settle

        # Boundary handling
        if periodic:
            gs.x[:N] %= p.Lx
            gs.y[:N] %= p.Ly
        else:
            for i in range(N):
                rb = gs.r_bound[i]
                gs.x[i] = np.clip(gs.x[i], rb, p.Lx - rb)
                gs.y[i] = np.clip(gs.y[i], rb, p.Ly - rb)

    # Count contacts after settling
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * float(np.max(gs.r_bound)) + 1.0, output_type='ndarray')
    n_contacts = 0
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            dv[0], dv[1] = minimum_image_disp(dv[0], dv[1], p.Lx, p.Ly)
        d = np.sqrt(np.dot(dv, dv))
        gap = d - gs.r_bound[i] - gs.r_bound[j]
        if gap < 1.0:  # within 1 µm = effectively touching
            n_contacts += 1
    print(f" done ({n_contacts} contacts, {n_contacts/max(N,1):.1f}/granule avg)")


def _settle_packing_3d(gs, p):
    """Run short isotropic compression micro-steps for 3D packing.

    Same algorithm as 2D but with z-coordinate.
    For periodic boundaries, uses random jitter instead of centripetal
    attraction and wraps positions.
    """
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    cx_dom = p.Lx / 2
    cy_dom = p.Ly / 2
    cz_dom = p.Lz / 2
    dt_settle = 0.02
    repulsion_k = 2.0

    print(f"  Settling 3D packing ({p.packing_settle_steps} steps)...", end="", flush=True)

    rng_settle = np.random.default_rng(12345)

    for step_i in range(p.packing_settle_steps):
        pos = gs.positions()  # (N, 3)
        fx = np.zeros(N)
        fy = np.zeros(N)
        fz = np.zeros(N)

        if periodic:
            jitter = 0.3 * max(0.5 * (1.0 - step_i / p.packing_settle_steps), 0.05)
            fx += jitter * rng_settle.standard_normal(N)
            fy += jitter * rng_settle.standard_normal(N)
            fz += jitter * rng_settle.standard_normal(N)
        else:
            # Centripetal attraction
            attract = max(0.5 * (1.0 - step_i / p.packing_settle_steps), 0.05)
            for i in range(N):
                dx = cx_dom - pos[i, 0]
                dy = cy_dom - pos[i, 1]
                dz = cz_dom - pos[i, 2]
                d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
                scale = attract * gs.r_bound[i] / d
                fx[i] += scale * dx
                fy[i] += scale * dy
                fz[i] += scale * dz

        # Pairwise repulsion
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(pos)
        max_rb = float(np.max(gs.r_bound))
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')
        for idx in range(len(pairs)):
            i, j = pairs[idx]
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            if periodic:
                dx, dy, dz = minimum_image_disp_3d(
                    dx, dy, dz, p.Lx, p.Ly, p.Lz)
            d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
            overlap = gs.r_bound[i] + gs.r_bound[j] - d
            if overlap > 0:
                nx, ny, nz = dx / d, dy / d, dz / d
                f_rep = repulsion_k * overlap
                fx[i] -= f_rep * nx; fy[i] -= f_rep * ny; fz[i] -= f_rep * nz
                fx[j] += f_rep * nx; fy[j] += f_rep * ny; fz[j] += f_rep * nz

        # Move
        gs.x += fx * dt_settle
        gs.y += fy * dt_settle
        gs.z += fz * dt_settle

        # Boundary handling
        if periodic:
            gs.x[:N] %= p.Lx
            gs.y[:N] %= p.Ly
            gs.z[:N] %= p.Lz
        else:
            for i in range(N):
                rb = gs.r_bound[i]
                gs.x[i] = np.clip(gs.x[i], rb, p.Lx - rb)
                gs.y[i] = np.clip(gs.y[i], rb, p.Ly - rb)
                gs.z[i] = np.clip(gs.z[i], rb, p.Lz - rb)

    # Count contacts
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * float(np.max(gs.r_bound)) + 1.0, output_type='ndarray')
    n_contacts = 0
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
        d = np.sqrt(np.dot(dv, dv))
        gap = d - gs.r_bound[i] - gs.r_bound[j]
        if gap < 1.0:
            n_contacts += 1
    print(f" done ({n_contacts} contacts, {n_contacts/max(N,1):.1f}/granule avg)")


# ══════════════════════════════════════════════════════════════════════
# Cell initialization (V1.5)
# ══════════════════════════════════════════════════════════════════════

def _initialize_cells(gs: GranuleSystem, rng):
    """Distribute cells uniformly on functional granule surfaces.

    Assigns cell_granule_id and initial surface positions.
    Cells start as ATTACHED (ready to spread and bridge immediately).
    """
    for i in range(gs.N):
        if gs.gtype[i] != 0:
            continue
        sl = gs.cells_on_granule(i)
        n = gs.cell_offset[i + 1] - gs.cell_offset[i]
        if n == 0:
            continue
        gs.cell_granule_id[sl] = i
        gs.cell_state[sl] = int(CellState.ATTACHED)
        gs.n_attached[i] = float(n)

        if gs.mode == "3D":
            # Distribute on superellipsoid surface (avoid poles)
            etas = np.linspace(-0.8 * np.pi / 2, 0.8 * np.pi / 2, n)
            omegas = rng.uniform(0, 2 * np.pi, n)
            gs.cell_eta_local[sl] = etas
            gs.cell_omega_local[sl] = omegas
        else:
            # Distribute uniformly around superellipse perimeter
            thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
            thetas += rng.uniform(0, 2 * np.pi / max(1, n))
            gs.cell_theta_local[sl] = thetas


# ══════════════════════════════════════════════════════════════════════
# Packing generator (random sequential addition)
# ══════════════════════════════════════════════════════════════════════

def generate_packing(p: Params, seed=42) -> GranuleSystem:
    rng = np.random.default_rng(seed)
    domain_area = p.Lx * p.Ly

    area_f = np.pi * p.R_func_mean**2
    area_i = np.pi * p.R_inert_mean**2
    n_func = int(round(p.phi_f_target * domain_area / area_f))
    n_inert = int(round(p.phi_i_target * domain_area / area_i))

    print(f"  Target: {n_func} functional (R~{p.R_func_mean:.0f}µm) + "
          f"{n_inert} inert (R~{p.R_inert_mean:.0f}µm)")
    if p.shape_enabled:
        print(f"  Shape: functional AR={p.aspect_ratio_func_mean:.2f}±"
              f"{p.aspect_ratio_func_std:.2f}, n={p.blockiness_func_mean:.1f}±"
              f"{p.blockiness_func_std:.1f}")
        print(f"         inert     AR={p.aspect_ratio_inert_mean:.2f}±"
              f"{p.aspect_ratio_inert_std:.2f}, n={p.blockiness_inert_mean:.1f}±"
              f"{p.blockiness_inert_std:.1f}")

    xs, ys, rs, types = [], [], [], []
    a_list, b_list, n_shape_list, theta_list = [], [], [], []
    gap = p.packing_gap  # minimum gap between granule surfaces (µm)

    # Interleave placement for good mixing
    order = []
    fi, ii = 0, 0
    while fi < n_func or ii < n_inert:
        if ii < n_inert: order.append(1); ii += 1
        if fi < n_func: order.append(0); fi += 1

    for gt in order:
        if gt == 0:
            r_eq = max(15, rng.normal(p.R_func_mean, p.R_func_std))
        else:
            r_eq = max(20, rng.normal(p.R_inert_mean, p.R_inert_std))

        # Sample shape parameters
        if p.shape_enabled:
            if gt == 0:
                ar = max(1.0, rng.normal(p.aspect_ratio_func_mean,
                                         p.aspect_ratio_func_std))
                n_s = max(1.5, rng.normal(p.blockiness_func_mean,
                                          p.blockiness_func_std))
            else:
                ar = max(1.0, rng.normal(p.aspect_ratio_inert_mean,
                                         p.aspect_ratio_inert_std))
                n_s = max(1.5, rng.normal(p.blockiness_inert_mean,
                                          p.blockiness_inert_std))
            # Compute semi-axes so superellipse has area = pi * r_eq^2
            # Initial guess: a = r_eq * sqrt(ar), b = r_eq / sqrt(ar)
            a0 = r_eq * np.sqrt(ar)
            b0 = r_eq / np.sqrt(ar)
            # Correct for exact area
            target_area = np.pi * r_eq**2
            actual_area = superellipse_area(a0, b0, n_s)
            if actual_area > 0:
                scale = np.sqrt(target_area / actual_area)
                a_val = a0 * scale
                b_val = b0 * scale
            else:
                a_val, b_val = r_eq, r_eq
            theta_val = rng.uniform(0, np.pi)
        else:
            a_val, b_val, n_s, theta_val = r_eq, r_eq, 2.0, 0.0

        # Bounding radius for overlap check
        r_bound_val = max(a_val, b_val)

        placed = False
        periodic = (p.boundary_mode == 'periodic')
        for _ in range(800):
            if periodic:
                cx = rng.uniform(0, p.Lx)
                cy = rng.uniform(0, p.Ly)
            else:
                cx = rng.uniform(r_bound_val + gap, p.Lx - r_bound_val - gap)
                cy = rng.uniform(r_bound_val + gap, p.Ly - r_bound_val - gap)
            ok = True
            for j in range(len(xs)):
                dx = cx - xs[j]; dy = cy - ys[j]
                if periodic:
                    dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
                rj_bound = max(a_list[j], b_list[j]) if p.shape_enabled else rs[j]
                if dx*dx + dy*dy < (r_bound_val + rj_bound + gap)**2:
                    ok = False; break
            if ok:
                xs.append(cx); ys.append(cy)
                rs.append(r_eq); types.append(gt)
                a_list.append(a_val); b_list.append(b_val)
                n_shape_list.append(n_s); theta_list.append(theta_val)
                placed = True; break
        if not placed:
            pass  # skip if can't place

    # Compute cells per granule (projected-area limited)
    n_cells = []
    A_cell_sphere = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
    for i in range(len(xs)):
        if types[i] == 0:
            cap = max_cells_on_granule(rs[i], A_cell_sphere, p.cell_coverage)
            nc = min(p.n_cells_per_granule, cap)
            n_cells.append(nc)
        else:
            n_cells.append(0)

    if p.shape_enabled:
        gs = GranuleSystem(xs, ys, rs, types, n_cells,
                           a=a_list, b=b_list, n_shape=n_shape_list,
                           theta=theta_list)
    else:
        gs = GranuleSystem(xs, ys, rs, types, n_cells)

    # ── Compression settle: push granules into contact ──
    if p.packing_settle_steps > 0 and gs.N > 1:
        _settle_packing_2d(gs, p)

    # Report packing fractions (use actual areas)
    if p.shape_enabled:
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                          for i in range(gs.N)])
        act_f = float(np.sum(areas[gs.func_mask])) / domain_area
        act_i = float(np.sum(areas[gs.inert_mask])) / domain_area
    else:
        act_f = sum(np.pi*gs.r[gs.func_mask]**2) / domain_area
        act_i = sum(np.pi*gs.r[gs.inert_mask]**2) / domain_area
    print(f"  Placed: {np.sum(gs.func_mask)} func (φ_f={act_f:.3f}) + "
          f"{np.sum(gs.inert_mask)} inert (φ_i={act_i:.3f})")
    print(f"  Void fraction: {1-act_f-act_i:.3f}")
    print(f"  Total cells: {int(sum(gs.n_cells))}")
    _initialize_cells(gs, rng)
    return gs


# ══════════════════════════════════════════════════════════════════════
# 3D packing generator (V1.4)
# ══════════════════════════════════════════════════════════════════════

def generate_packing_3d(p: Params, seed=42) -> GranuleSystem:
    """Generate a 3D random packing of superellipsoids."""
    rng = np.random.default_rng(seed)
    domain_vol = p.Lx * p.Ly * p.Lz

    vol_f = (4.0/3.0) * np.pi * p.R_func_mean**3
    vol_i = (4.0/3.0) * np.pi * p.R_inert_mean**3
    n_func = int(round(p.phi_f_target * domain_vol / vol_f))
    n_inert = int(round(p.phi_i_target * domain_vol / vol_i))

    print(f"  Target: {n_func} functional (R~{p.R_func_mean:.0f}µm) + "
          f"{n_inert} inert (R~{p.R_inert_mean:.0f}µm)")

    xs, ys, zs, rs, types = [], [], [], [], []
    a_list, b_list, c_list = [], [], []
    n1_list, n2_list = [], []
    quat_list = []
    gap = p.packing_gap

    # Interleave placement
    order = []
    fi, ii = 0, 0
    while fi < n_func or ii < n_inert:
        if ii < n_inert: order.append(1); ii += 1
        if fi < n_func: order.append(0); fi += 1

    for gt in order:
        if gt == 0:
            r_eq = max(15, rng.normal(p.R_func_mean, p.R_func_std))
        else:
            r_eq = max(20, rng.normal(p.R_inert_mean, p.R_inert_std))

        # Sample shape parameters
        if p.shape_enabled:
            if gt == 0:
                ar_ab = max(1.0, rng.normal(p.aspect_ratio_func_mean,
                                            p.aspect_ratio_func_std))
                ar_c = max(0.3, rng.normal(p.aspect_ratio_c_func_mean,
                                           p.aspect_ratio_c_func_std))
                n1_val = max(1.5, rng.normal(p.blockiness_func_mean,
                                             p.blockiness_func_std))
                n2_val = max(1.5, rng.normal(p.blockiness_n2_func_mean,
                                             p.blockiness_n2_func_std))
            else:
                ar_ab = max(1.0, rng.normal(p.aspect_ratio_inert_mean,
                                            p.aspect_ratio_inert_std))
                ar_c = max(0.3, rng.normal(p.aspect_ratio_c_inert_mean,
                                           p.aspect_ratio_c_inert_std))
                n1_val = max(1.5, rng.normal(p.blockiness_inert_mean,
                                             p.blockiness_inert_std))
                n2_val = max(1.5, rng.normal(p.blockiness_n2_inert_mean,
                                             p.blockiness_n2_inert_std))
            # Compute semi-axes: a, b from ar_ab; c from ar_c
            a0 = r_eq * np.sqrt(ar_ab)
            b0 = r_eq / np.sqrt(ar_ab)
            c0 = r_eq * ar_c
            # Scale to match target volume = (4/3) pi r_eq^3
            target_vol = (4.0/3.0) * np.pi * r_eq**3
            actual_vol = superellipsoid_volume(a0, b0, c0, n1_val, n2_val)
            if actual_vol > 0:
                scale = (target_vol / actual_vol)**(1.0/3.0)
                a_val, b_val, c_val = a0*scale, b0*scale, c0*scale
            else:
                a_val = b_val = c_val = r_eq
            q_val = quat_random(rng)
        else:
            a_val = b_val = c_val = r_eq
            n1_val = n2_val = 2.0
            q_val = np.array([1.0, 0.0, 0.0, 0.0])

        r_bound_val = max(a_val, b_val, c_val)

        placed = False
        periodic = (p.boundary_mode == 'periodic')
        for _ in range(800):
            if periodic:
                cx = rng.uniform(0, p.Lx)
                cy = rng.uniform(0, p.Ly)
                cz = rng.uniform(0, p.Lz)
            else:
                cx = rng.uniform(r_bound_val + gap, p.Lx - r_bound_val - gap)
                cy = rng.uniform(r_bound_val + gap, p.Ly - r_bound_val - gap)
                cz = rng.uniform(r_bound_val + gap, p.Lz - r_bound_val - gap)
            ok = True
            for j in range(len(xs)):
                dx = cx - xs[j]; dy = cy - ys[j]; dz = cz - zs[j]
                if periodic:
                    dx, dy, dz = minimum_image_disp_3d(
                        dx, dy, dz, p.Lx, p.Ly, p.Lz)
                rj_b = max(a_list[j], b_list[j], c_list[j])
                if dx*dx + dy*dy + dz*dz < (r_bound_val + rj_b + gap)**2:
                    ok = False; break
            if ok:
                xs.append(cx); ys.append(cy); zs.append(cz)
                rs.append(r_eq); types.append(gt)
                a_list.append(a_val); b_list.append(b_val); c_list.append(c_val)
                n1_list.append(n1_val); n2_list.append(n2_val)
                quat_list.append(q_val)
                placed = True; break
        if not placed:
            pass  # skip

    # Compute cells per granule
    n_cells = []
    A_cell_sphere = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
    for i in range(len(xs)):
        if types[i] == 0:
            cap = max_cells_on_granule(rs[i], A_cell_sphere, p.cell_coverage)
            nc = min(p.n_cells_per_granule, cap)
            n_cells.append(nc)
        else:
            n_cells.append(0)

    gs = GranuleSystem(
        xs, ys, rs, types, n_cells,
        z=zs,
        a=a_list if p.shape_enabled else None,
        b=b_list if p.shape_enabled else None,
        c=c_list if p.shape_enabled else None,
        n1=n1_list if p.shape_enabled else None,
        n2=n2_list if p.shape_enabled else None,
        quat=quat_list if p.shape_enabled else None,
        mode="3D")

    # ── Compression settle: push granules into contact ──
    if p.packing_settle_steps > 0 and gs.N > 1:
        _settle_packing_3d(gs, p)

    # Report packing fractions
    if p.shape_enabled:
        vols = np.array([superellipsoid_volume(gs.a[i], gs.b[i], gs.c[i],
                         gs.n1[i], gs.n2[i]) for i in range(gs.N)])
        act_f = float(np.sum(vols[gs.func_mask])) / domain_vol
        act_i = float(np.sum(vols[gs.inert_mask])) / domain_vol
    else:
        act_f = sum((4.0/3.0)*np.pi*gs.r[gs.func_mask]**3) / domain_vol
        act_i = sum((4.0/3.0)*np.pi*gs.r[gs.inert_mask]**3) / domain_vol
    print(f"  Placed: {np.sum(gs.func_mask)} func (φ_f={act_f:.3f}) + "
          f"{np.sum(gs.inert_mask)} inert (φ_i={act_i:.3f})")
    print(f"  Void fraction: {1-act_f-act_i:.3f}")
    print(f"  Total cells: {int(sum(gs.n_cells))}")
    _initialize_cells(gs, rng)
    return gs


# ══════════════════════════════════════════════════════════════════════
# 3D Contact detection (V1.4)
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def find_contact_spheres_3d(xi, yi, zi, ri, xj, yj, zj, rj):
    """
    Sphere-sphere contact in 3D.
    Returns (in_contact, delta, nx, ny, nz, cx, cy, cz, R_eff) or None.
    """
    dx = xj - xi; dy = yj - yi; dz = zj - zi
    d = np.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 1e-12:
        return None
    overlap = ri + rj - d
    if overlap <= 0:
        return None
    nx, ny, nz = dx/d, dy/d, dz/d
    R_eff = ri * rj / (ri + rj)
    cx = xi + (ri - overlap/2) * nx
    cy = yi + (ri - overlap/2) * ny
    cz = zi + (ri - overlap/2) * nz
    return (True, overlap, nx, ny, nz, cx, cy, cz, R_eff)


@njit(cache=True)
def find_contact_superellipsoids_3d(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
        xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """
    Common normal contact detection between two 3D superellipsoids.
    Returns (in_contact, delta, nx, ny, nz, cx, cy, cz, R_eff) or None.
    """
    dx_c = xj - xi; dy_c = yj - yi; dz_c = zj - zi
    d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
    if d_c < 1e-12:
        return None
    ux = dx_c/d_c; uy = dy_c/d_c; uz = dz_c/d_c

    # Initial guess: find parameter on each surface closest to line of centres
    dir_body_i = quat_rotate_inv(qi, np.array([ux, uy, uz]))
    dir_body_j = quat_rotate_inv(qj, np.array([-ux, -uy, -uz]))

    eta_i = np.arctan2(dir_body_i[2],
                       np.sqrt(dir_body_i[0]**2 + dir_body_i[1]**2) + 1e-30)
    omega_i = np.arctan2(dir_body_i[1], dir_body_i[0])
    eta_j = np.arctan2(dir_body_j[2],
                       np.sqrt(dir_body_j[0]**2 + dir_body_j[1]**2) + 1e-30)
    omega_j = np.arctan2(dir_body_j[1], dir_body_j[0])

    _eta_lo = -np.pi/2 + 0.01
    _eta_hi = np.pi/2 - 0.01

    # Newton-Raphson: find common normal
    for _ in range(15):
        # Surface points in body frame
        pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)

        # Transform to world
        pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
        pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

        # Vector from Pi to Pj
        dp = pj_world - pi_world
        dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
        if dp_mag < 1e-12:
            break
        target = dp / dp_mag

        # Normals in body frame, then to world
        ni_body = superellipsoid_normal(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        nj_body = superellipsoid_normal(eta_j, omega_j, aj, bj, cj, n1j, n2j)
        ni_world = quat_rotate(qi, ni_body)
        nj_world = quat_rotate(qj, nj_body)

        # Error: cross product (normal should align with target)
        err_i = np.cross(ni_world, target)
        err_j = np.cross(nj_world, -target)

        err_i_mag = np.sqrt(err_i[0]**2 + err_i[1]**2 + err_i[2]**2)
        err_j_mag = np.sqrt(err_j[0]**2 + err_j[1]**2 + err_j[2]**2)
        if err_i_mag < 1e-7 and err_j_mag < 1e-7:
            break

        # Damped update of surface parameters
        # Project error onto eta/omega directions (approximate Jacobian)
        eta_i -= 0.3 * (err_i[2] * np.cos(omega_i) - err_i[1] * np.sin(omega_i))
        omega_i -= 0.3 * err_i[0]
        eta_j -= 0.3 * (err_j[2] * np.cos(omega_j) - err_j[1] * np.sin(omega_j))
        omega_j -= 0.3 * err_j[0]

        # Clamp
        eta_i = min(max(eta_i, _eta_lo), _eta_hi)
        eta_j = min(max(eta_j, _eta_lo), _eta_hi)

    # Final geometry
    pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
    pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

    dp = pj_world - pi_world
    dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

    # Check overlap: is Pi inside body j?
    pi_in_j_body = quat_rotate_inv(qj, pi_world - np.array([xj, yj, zj]))
    val_j = superellipsoid_implicit(pi_in_j_body[0], pi_in_j_body[1],
                                    pi_in_j_body[2], aj, bj, cj, n1j, n2j)
    pj_in_i_body = quat_rotate_inv(qi, pj_world - np.array([xi, yi, zi]))
    val_i = superellipsoid_implicit(pj_in_i_body[0], pj_in_i_body[1],
                                    pj_in_i_body[2], ai, bi, ci, n1i, n2i)

    if val_j > 1.0 and val_i > 1.0:
        return None  # no overlap

    delta = dp_mag
    if dp_mag > 1e-12:
        nx, ny, nz = dp[0]/dp_mag, dp[1]/dp_mag, dp[2]/dp_mag
    else:
        nx, ny, nz = ux, uy, uz

    cx = 0.5 * (pi_world[0] + pj_world[0])
    cy = 0.5 * (pi_world[1] + pj_world[1])
    cz_pt = 0.5 * (pi_world[2] + pj_world[2])

    # Effective radius from local curvature
    R1i, R2i = superellipsoid_curvature_radii(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    R1j, R2j = superellipsoid_curvature_radii(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    R_eff_i = np.sqrt(R1i * R2i)
    R_eff_j = np.sqrt(R1j * R2j)
    R_eff = R_eff_i * R_eff_j / (R_eff_i + R_eff_j) if (R_eff_i + R_eff_j) > 0 else 1.0

    return (True, delta, nx, ny, nz, cx, cy, cz_pt, R_eff)


def find_contact_wall_3d(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound,
                         wall_pos, wall_axis, wall_sign):
    """
    Contact between a 3D granule and a flat wall.
    wall_axis: 0=x, 1=y, 2=z.  wall_sign: +1 if granule on positive side.
    Returns (penetration, R_local) or None.
    """
    # Sample the superellipsoid surface and find extreme point
    n_sample = 20
    eta = np.linspace(-np.pi/2, np.pi/2, n_sample)
    omega = np.linspace(-np.pi, np.pi, n_sample)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e2 = 2.0 / n2i
    e1 = 2.0 / n1i
    bx = ai * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    by = bi * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    bz = ci * _sgnpow(np.sin(E), e2)

    # Rotate to world frame
    R_mat = quat_to_rotation_matrix(qi)
    body_pts = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    world_pts = body_pts @ R_mat.T + np.array([xi, yi, zi])

    coords = world_pts[:, wall_axis]

    if wall_sign > 0:
        idx = np.argmin(coords)
        pen = wall_pos - coords[idx]
    else:
        idx = np.argmax(coords)
        pen = coords[idx] - wall_pos

    if pen <= 0:
        return None

    # Approximate local curvature radius from bounding sphere
    R_local = ri_bound * 0.5  # rough approximation
    return (pen, R_local)


# ══════════════════════════════════════════════════════════════════════
# 2D-slice mode (V1.4): generate 3D packing, slice at z-midplane
# ══════════════════════════════════════════════════════════════════════

def slice_superellipsoid_z(cx, cy, cz, a, b, c, n1, n2, quat, z_plane):
    """
    Slice a 3D superellipsoid at z = z_plane to produce a 2D cross-section.
    Returns (a_2d, b_2d, n_2d, theta_2d) or None if plane doesn't intersect.
    """
    # Transform z_plane to body frame
    R_mat = quat_to_rotation_matrix(quat)
    delta = np.array([0.0, 0.0, z_plane - cz])
    body_delta = R_mat.T @ delta

    # z coordinate in body frame
    z_body = body_delta[2]
    z_term = abs(z_body / c) ** n2
    if z_term >= 1.0:
        return None  # plane doesn't intersect

    # Cross-section: scale factor
    s = (1.0 - z_term) ** (1.0 / n2)
    a_2d = a * s
    b_2d = b * s
    n_2d = n1  # equatorial blockiness preserved in cross-section

    # Orientation from projected rotation matrix
    theta_2d = np.arctan2(R_mat[1, 0], R_mat[0, 0])

    return a_2d, b_2d, n_2d, theta_2d


def generate_packing_2d_slice(p: Params, seed=42) -> GranuleSystem:
    """
    Generate 3D packing, then slice at z-midplane to create 2D granule system.
    Mimics experimental confocal imaging of a 3D scaffold.
    """
    print("  2D-slice mode: generating 3D packing first...")
    gs_3d = generate_packing_3d(p, seed=seed)

    z_plane = p.Lz / 2.0
    xs, ys, rs, types, n_cells = [], [], [], [], []
    a_list, b_list, n_shape_list, theta_list = [], [], [], []

    for i in range(gs_3d.N):
        result = slice_superellipsoid_z(
            gs_3d.x[i], gs_3d.y[i], gs_3d.z[i],
            gs_3d.a[i], gs_3d.b[i], gs_3d.c[i],
            gs_3d.n1[i], gs_3d.n2[i],
            gs_3d.quat[i] if gs_3d.quat is not None else np.array([1,0,0,0]),
            z_plane)
        if result is None:
            continue
        a_2d, b_2d, n_2d, theta_2d = result
        r_eq = np.sqrt(a_2d * b_2d)  # equivalent radius from cross-section
        if r_eq < 5.0:
            continue  # too small slice

        xs.append(gs_3d.x[i])
        ys.append(gs_3d.y[i])
        rs.append(r_eq)
        types.append(gs_3d.gtype[i])
        n_cells.append(int(gs_3d.n_cells[i]))
        a_list.append(a_2d)
        b_list.append(b_2d)
        n_shape_list.append(n_2d)
        theta_list.append(theta_2d)

    print(f"  Sliced: {len(xs)} granules intersect z-midplane "
          f"(of {gs_3d.N} total)")

    if len(xs) == 0:
        raise ValueError("No granules intersect the slice plane")

    gs = GranuleSystem(
        xs, ys, rs, types, n_cells,
        a=a_list, b=b_list, n_shape=n_shape_list, theta=theta_list,
        mode="2D")

    domain_area = p.Lx * p.Ly
    if p.shape_enabled:
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n1[i])
                          for i in range(gs.N)])
        act_f = float(np.sum(areas[gs.func_mask])) / domain_area
        act_i = float(np.sum(areas[gs.inert_mask])) / domain_area
    else:
        act_f = sum(np.pi*gs.r[gs.func_mask]**2) / domain_area
        act_i = sum(np.pi*gs.r[gs.inert_mask]**2) / domain_area
    print(f"  2D slice packing: φ_f={act_f:.3f}, φ_i={act_i:.3f}, "
          f"φ_v={1-act_f-act_i:.3f}")
    _initialize_cells(gs, np.random.default_rng(seed))
    return gs


# ══════════════════════════════════════════════════════════════════════
# Per-cell bridge force recording (V1.5)
# ══════════════════════════════════════════════════════════════════════

def _service_committed_bridges(gs, gi, gj, gap, p, F_per_cell, nx, ny, nz, F):
    """Re-apply forces for cells already committed to a bridge between gi↔gj.

    For each committed bridging cell, apply force modulated by bridge maturity
    (ramp over bridge_formation_time).  If gap exceeds bridge_break_gap,
    the bridge ruptures and the cell goes senescent.

    Returns (n_committed_i, n_committed_j) — number of committed cells on each side.
    """
    n_ci = n_cj = 0

    for ci in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ci] == int(CellState.BRIDGING) and gs.cell_bridge_target[ci] == gj:
            if gap > p.bridge_break_gap:
                # Bridge rupture → senescent (cell was under excessive strain)
                gs.cell_state[ci] = int(CellState.SENESCENT)
                gs.cell_bridge_target[ci] = -1
                gs.cell_bridge_age[ci] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[ci] / max(0.1, p.bridge_formation_time))
                F_cell = min(F_per_cell * maturity, p.F_max_per_cell)
                gs.cell_fx[ci] += F_cell * nx
                gs.cell_fy[ci] += F_cell * ny
                gs.cell_fz[ci] += F_cell * nz
                F[gi, 0] += F_cell * nx if F.ndim == 2 else 0
                F[gi, 1] += F_cell * ny if F.ndim == 2 else 0
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gi, 2] += F_cell * nz
                elif F.ndim == 2 and F.shape[1] == 2:
                    pass  # 2D, no z
                n_ci += 1

    for cj in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[cj] == int(CellState.BRIDGING) and gs.cell_bridge_target[cj] == gi:
            if gap > p.bridge_break_gap:
                gs.cell_state[cj] = int(CellState.SENESCENT)
                gs.cell_bridge_target[cj] = -1
                gs.cell_bridge_age[cj] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[cj] / max(0.1, p.bridge_formation_time))
                F_cell = min(F_per_cell * maturity, p.F_max_per_cell)
                gs.cell_fx[cj] -= F_cell * nx
                gs.cell_fy[cj] -= F_cell * ny
                gs.cell_fz[cj] -= F_cell * nz
                F[gj, 0] -= F_cell * nx if F.ndim == 2 else 0
                F[gj, 1] -= F_cell * ny if F.ndim == 2 else 0
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gj, 2] -= F_cell * nz
                elif F.ndim == 2 and F.shape[1] == 2:
                    pass  # 2D, no z
                n_cj += 1

    return n_ci, n_cj


def _attempt_new_bridges(gs, gi, gj, gap, p, rng, F_per_cell, nx, ny, nz, F,
                         n_existing_bridges=0):
    """Probabilistically initiate new bridges between granules gi and gj.

    Only cells that are SPREADING or PROLIFERATING with sufficient FA maturity
    can attempt bridging.  Each eligible cell has a Poisson-distributed
    probability of finding and committing to a bridge target per timestep.

    When existing bridges are present (n_existing_bridges > 0), non-bridging
    cells can migrate along the established bridge and form additional
    connections, boosted by bridge_secondary_rate_mult.
    """
    bridgeable_states = (int(CellState.SPREADING), int(CellState.PROLIFERATING))
    min_fa = p.min_fa_for_bridge

    # Proximity modulates probability: closer gaps → easier to find target
    proximity = 1.0 - gap / p.cell_sense_distance
    rate = p.bridge_attempt_rate
    # Secondary migration: existing bridges act as highways for new cells
    if n_existing_bridges > 0:
        rate *= p.bridge_secondary_rate_mult
    p_bridge = (1.0 - np.exp(-rate * p.dt)) * proximity

    # Eligible cells on granule i (sufficient FA maturity, not already bridging)
    for ci in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ci] not in bridgeable_states:
            continue
        if gs.fa_maturity[gi] < min_fa:
            continue
        if rng.random() < p_bridge:
            gs.cell_state[ci] = int(CellState.BRIDGING)
            gs.cell_bridge_target[ci] = gj
            gs.cell_bridge_age[ci] = 0.0
            # Initial force is very weak (bridge just forming)
            maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
            F_cell = min(F_per_cell * maturity, p.F_max_per_cell)
            gs.cell_fx[ci] += F_cell * nx
            gs.cell_fy[ci] += F_cell * ny
            gs.cell_fz[ci] += F_cell * nz
            if F.ndim == 2 and F.shape[1] >= 2:
                F[gi, 0] += F_cell * nx
                F[gi, 1] += F_cell * ny
            if F.ndim == 2 and F.shape[1] == 3:
                F[gi, 2] += F_cell * nz

    # Eligible cells on granule j
    for cj in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[cj] not in bridgeable_states:
            continue
        if gs.fa_maturity[gj] < min_fa:
            continue
        if rng.random() < p_bridge:
            gs.cell_state[cj] = int(CellState.BRIDGING)
            gs.cell_bridge_target[cj] = gi
            gs.cell_bridge_age[cj] = 0.0
            maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
            F_cell = min(F_per_cell * maturity, p.F_max_per_cell)
            gs.cell_fx[cj] -= F_cell * nx
            gs.cell_fy[cj] -= F_cell * ny
            gs.cell_fz[cj] -= F_cell * nz
            if F.ndim == 2 and F.shape[1] >= 2:
                F[gj, 0] -= F_cell * nx
                F[gj, 1] -= F_cell * ny
            if F.ndim == 2 and F.shape[1] == 3:
                F[gj, 2] -= F_cell * nz


# ══════════════════════════════════════════════════════════════════════
# Force computation
# ══════════════════════════════════════════════════════════════════════

def compute_forces(gs: GranuleSystem, p: Params, rng):
    """
    Compute all forces and torques on each granule.
    Returns (F, torques) where F is (N, 2) force array in nN and
    torques is (N,) array in nN·µm.

    Contact forces:
      Hertz repulsion:  F = (4/3) E* √R* δ^{3/2}
      DMT adhesion:     F_adh = 2π W R*  (constant attractive during contact)
      Net normal:       F_n = F_Hertz − F_adh  (can be negative = attractive)

    Tangential friction (area-dependent, hydrogel tribology):
      F_fric = τ₀ · A_contact · tanh(|v_t|/v_ref)  opposing relative sliding
      where A_contact = π R* δ  (Hertzian contact area)

    For superellipses, R* uses local curvature at the contact point.

    Wall uses Hertz (rigid limit):
      E* = E / (1-ν²), R* = R_i (circle) or R_local (superellipse)
    """
    N = gs.N
    F = np.zeros((N, 2))
    torques = np.zeros(N)
    pos = gs.positions()
    contacts = []  # V1.5.2: per-contact data for stress visualization

    # V1.5: Reset per-cell force vectors (but preserve bridge state and targets)
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    # ── Precompute effective moduli (Pa) ──
    nu2 = p.poisson_ratio ** 2
    E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu2))   # granule-granule
    E_star_gw = (p.E_modulus * 1e3) / (1.0 - nu2)            # granule-wall

    # ── Friction/adhesion lookup by pair type ──
    friction_lut = {
        (0, 0): (p.tau_0_ff, p.W_adh_ff),  # functional–functional
        (0, 1): (p.tau_0_if, p.W_adh_if),  # mixed
        (1, 0): (p.tau_0_if, p.W_adh_if),  # mixed
        (1, 1): (p.tau_0_ii, p.W_adh_ii),  # inert–inert
    }

    # ── Neighbour search ──
    max_r = float(np.max(gs.r_bound))
    cutoff = 2*max_r + p.L_max
    periodic = (p.boundary_mode == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j,0] - pos[i,0]
        dy = pos[j,1] - pos[i,1]
        if periodic:
            dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
        d = np.sqrt(dx*dx + dy*dy)
        if d < 1e-6:
            continue
        nx, ny = dx/d, dy/d

        # ── Contact detection ──
        if gs.is_circle:
            # Fast circle path (V1.2 behavior)
            overlap = gs.r[i] + gs.r[j] - d
            if overlap > 0:
                R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                # Contact point at midpoint of overlap along line of centres
                contact_x = pos[i,0] + (gs.r[i] - overlap/2) * nx
                contact_y = pos[i,1] + (gs.r[i] - overlap/2) * ny
                in_contact = True
            else:
                in_contact = False
                overlap = 0.0
                R_eff = 0.0
                contact_x = contact_y = 0.0
        else:
            # Superellipse contact (common normal method)
            # For periodic BCs, use virtual position of j (nearest image)
            xj_v = pos[i,0] + dx  # pos_i + minimum-image displacement
            yj_v = pos[i,1] + dy
            result = find_contact_superellipses(
                pos[i,0], pos[i,1], gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                xj_v, yj_v, gs.a[j], gs.b[j], gs.n_shape[j], gs.theta[j])
            if result is not None:
                in_contact = True
                _, overlap, nx, ny, contact_x, contact_y, R_loc_i, R_loc_j = result
                R_eff = R_loc_i * R_loc_j / (R_loc_i + R_loc_j)
            else:
                in_contact = False
                overlap = 0.0
                R_eff = 0.0
                contact_x = contact_y = 0.0

        # ── Contact forces: Hertz + DMT + friction ──
        if in_contact and overlap > 0:
            Fc = hertz_contact_force(E_star_gg, R_eff, overlap)

            # Pair-type friction/adhesion parameters
            tau_0, W_adh = friction_lut[(gs.gtype[i], gs.gtype[j])]

            # DMT adhesion: F_adh = 2π W R* [nN]
            F_adh = 2.0 * np.pi * W_adh * R_eff * 1e3

            # Net normal force (positive = repulsive, negative = attractive)
            F_normal = Fc - F_adh
            Fnx = F_normal * nx
            Fny = F_normal * ny
            F[i,0] -= Fnx; F[i,1] -= Fny
            F[j,0] += Fnx; F[j,1] += Fny

            # Torque from normal force (off-centre contact)
            if not gs.is_circle:
                # τ = (contact - centre) × F
                rci_x = contact_x - pos[i,0]
                rci_y = contact_y - pos[i,1]
                rcj_x = contact_x - pos[j,0]
                rcj_y = contact_y - pos[j,1]
                torques[i] += rci_x * (-Fny) - rci_y * (-Fnx)
                torques[j] += rcj_x * Fny - rcj_y * Fnx

            # Tangential friction (area-dependent, opposes relative sliding)
            A_contact = np.pi * R_eff * overlap   # Hertzian contact area (µm²)
            dvx = gs.vx[j] - gs.vx[i]
            dvy = gs.vy[j] - gs.vy[i]
            v_dot_n = dvx * nx + dvy * ny
            vtx = dvx - v_dot_n * nx
            vty = dvy - v_dot_n * ny
            vt_mag = np.sqrt(vtx*vtx + vty*vty)

            if vt_mag > 1e-12:
                # F = τ₀ × A × tanh(|v_t|/v_ref) [Pa × µm² × 1e-3 → nN]
                F_fric = tau_0 * A_contact * 1e-3 * np.tanh(
                    vt_mag / p.friction_v_ref)
                tx, ty = vtx / vt_mag, vty / vt_mag
                Ftx, Fty = F_fric * tx, F_fric * ty
                F[i,0] += Ftx; F[i,1] += Fty
                F[j,0] -= Ftx; F[j,1] -= Fty

                # Torque from friction
                if not gs.is_circle:
                    torques[i] += rci_x * Fty - rci_y * Ftx
                    torques[j] += rcj_x * (-Fty) - rcj_y * (-Ftx)

            # V1.5.2: Record contact data for stress visualization
            contacts.append({
                'i': i, 'j': j,
                'cx': contact_x, 'cy': contact_y, 'cz': 0.0,
                'nx': nx, 'ny': ny, 'nz': 0.0,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
            })

        # ── Cell bridging (motor-clutch model, functional–functional) ──
        if gs.gtype[i] == 0 and gs.gtype[j] == 0:
            if gs.is_circle:
                gap = d - gs.r[i] - gs.r[j]
            else:
                if in_contact:
                    gap = -overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0:
                        gap = 0.0

            # Motor-clutch force per cell (stiffness + FA maturity)
            avg_maturity = 0.5 * (gs.fa_maturity[i] + gs.fa_maturity[j])
            F_per_cell = motor_clutch_force(p.E_modulus, p, avg_maturity)

            # 1) Service committed bridges (apply force with maturity ramp)
            n_existing = 0
            if gap > p.L_rest:
                n_ci, n_cj = _service_committed_bridges(
                    gs, i, j, gap, p, F_per_cell, nx, ny, 0.0, F)
                n_existing = n_ci + n_cj

            # 2) Probabilistic new bridge formation
            if 0 < gap < p.cell_sense_distance and gap > p.L_rest:
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_per_cell, nx, ny, 0.0, F,
                    n_existing_bridges=n_existing)

    # ── Wall repulsion (skip for periodic boundaries) ──
    if not periodic:
        for i in range(N):
            if gs.is_circle:
                r = gs.r[i]
                for pen, axis, sign in [
                    (r - gs.x[i],            0, +1),   # left
                    (gs.x[i] - (p.Lx - r),   0, -1),   # right
                    (r - gs.y[i],            1, +1),   # bottom
                    (gs.y[i] - (p.Ly - r),   1, -1),   # top
                ]:
                    if pen > 0:
                        Fw = hertz_contact_force(E_star_gw, r, pen)
                        F[i, axis] += sign * Fw
            else:
                # Superellipse wall contact
                walls = [
                    (0.0,    0, +1),   # left wall at x=0
                    (p.Lx,   0, -1),   # right wall at x=Lx
                    (0.0,    1, +1),   # bottom wall at y=0
                    (p.Ly,   1, -1),   # top wall at y=Ly
                ]
                for wall_pos, wall_axis, wall_sign in walls:
                    wresult = find_contact_superellipse_wall(
                        gs.x[i], gs.y[i], gs.a[i], gs.b[i],
                        gs.n_shape[i], gs.theta[i],
                        wall_pos, wall_axis, wall_sign)
                    if wresult is not None:
                        pen, R_local = wresult
                        Fw = hertz_contact_force(E_star_gw, R_local, pen)
                        F[i, wall_axis] += wall_sign * Fw

    # ── Active noise on functional granules (vectorized) ──
    if p.T_active > 0:
        func_mask = gs.gtype[:N] == 0
        n_func = int(np.sum(func_mask))
        if n_func > 0:
            gamma_func = p.drag_scale * gs.r[:N][func_mask]
            noise_amp = np.sqrt(2 * gamma_func * p.T_active / p.dt)
            F[:N, 0][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 1][func_mask] += noise_amp * rng.standard_normal(n_func)

    return F, torques, contacts


# ══════════════════════════════════════════════════════════════════════
# 3D Force computation (V1.4)
# ══════════════════════════════════════════════════════════════════════

def compute_forces_3d(gs: GranuleSystem, p: Params, rng):
    """
    Compute all forces and torques on each granule in 3D.
    Returns (F, torques) where F is (N, 3) and torques is (N, 3).
    """
    N = gs.N
    F = np.zeros((N, 3))
    torques = np.zeros((N, 3))
    pos = gs.positions()  # (N, 3)
    contacts = []  # V1.5.2: per-contact data for stress visualization

    # V1.5: Reset per-cell force vectors (but preserve bridge state and targets)
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    nu2 = p.poisson_ratio ** 2
    E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu2))
    E_star_gw = (p.E_modulus * 1e3) / (1.0 - nu2)

    friction_lut = {
        (0, 0): (p.tau_0_ff, p.W_adh_ff),
        (0, 1): (p.tau_0_if, p.W_adh_if),
        (1, 0): (p.tau_0_if, p.W_adh_if),
        (1, 1): (p.tau_0_ii, p.W_adh_ii),
    }

    # Neighbour search (3D)
    max_r = float(np.max(gs.r_bound))
    cutoff = 2*max_r + p.L_max
    periodic = (p.boundary_mode == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dp = pos[j] - pos[i]
        if periodic:
            dp[0], dp[1], dp[2] = minimum_image_disp_3d(
                dp[0], dp[1], dp[2], p.Lx, p.Ly, p.Lz)
        d = np.linalg.norm(dp)
        if d < 1e-6:
            continue
        nv = dp / d  # unit normal i->j

        # Contact detection — use virtual position of j for periodic BCs
        pj_v = pos[i] + dp  # nearest image of j relative to i
        if gs.is_circle:
            result = find_contact_spheres_3d(
                pos[i,0], pos[i,1], pos[i,2], gs.r[i],
                pj_v[0], pj_v[1], pj_v[2], gs.r[j])
        else:
            result = find_contact_superellipsoids_3d(
                pos[i,0], pos[i,1], pos[i,2],
                gs.a[i], gs.b[i], gs.c[i], gs.n1[i], gs.n2[i],
                gs.quat[i],
                pj_v[0], pj_v[1], pj_v[2],
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j],
                gs.quat[j])

        if result is not None:
            _, overlap, nx, ny, nz, cx, cy, cz_pt, R_eff = result
            n_vec = np.array([nx, ny, nz])

            if overlap > 0:
                Fc = hertz_contact_force(E_star_gg, R_eff, overlap)
                tau_0, W_adh = friction_lut[(gs.gtype[i], gs.gtype[j])]
                F_adh = 2.0 * np.pi * W_adh * R_eff * 1e3
                F_normal = Fc - F_adh
                Fn = F_normal * n_vec
                F[i] -= Fn
                F[j] += Fn

                # Torque from off-centre contact
                if not gs.is_circle:
                    rc_i = np.array([cx - pos[i,0], cy - pos[i,1], cz_pt - pos[i,2]])
                    rc_j = np.array([cx - pos[j,0], cy - pos[j,1], cz_pt - pos[j,2]])
                    torques[i] += np.cross(rc_i, -Fn)
                    torques[j] += np.cross(rc_j, Fn)

                # Tangential friction
                A_contact = np.pi * R_eff * overlap
                dv = np.array([gs.vx[j]-gs.vx[i], gs.vy[j]-gs.vy[i],
                               gs.vz[j]-gs.vz[i]])
                v_dot_n = np.dot(dv, n_vec)
                vt = dv - v_dot_n * n_vec
                vt_mag = np.linalg.norm(vt)

                if vt_mag > 1e-12:
                    F_fric = tau_0 * A_contact * 1e-3 * np.tanh(
                        vt_mag / p.friction_v_ref)
                    t_vec = vt / vt_mag
                    Ft = F_fric * t_vec
                    F[i] += Ft
                    F[j] -= Ft
                    if not gs.is_circle:
                        torques[i] += np.cross(rc_i, Ft)
                        torques[j] += np.cross(rc_j, -Ft)

            # V1.5.2: Record contact data for stress visualization
            contacts.append({
                'i': i, 'j': j,
                'cx': cx, 'cy': cy, 'cz': cz_pt,
                'nx': nx, 'ny': ny, 'nz': nz,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
            })

            in_contact = True
            contact_overlap = overlap
        else:
            in_contact = False
            contact_overlap = 0.0

        # Cell bridging (motor-clutch, functional-functional)
        if gs.gtype[i] == 0 and gs.gtype[j] == 0:
            if in_contact:
                gap = -contact_overlap
            else:
                gap = d - gs.r_bound[i] - gs.r_bound[j]
                if gap < 0:
                    gap = 0.0

            avg_maturity = 0.5 * (gs.fa_maturity[i] + gs.fa_maturity[j])
            F_per_cell = motor_clutch_force(p.E_modulus, p, avg_maturity)

            # 1) Service committed bridges (apply force with maturity ramp)
            n_existing = 0
            if gap > p.L_rest:
                n_ci, n_cj = _service_committed_bridges(
                    gs, i, j, gap, p, F_per_cell,
                    nv[0], nv[1], nv[2], F)
                n_existing = n_ci + n_cj

            # 2) Probabilistic new bridge formation
            if 0 < gap < p.cell_sense_distance and gap > p.L_rest:
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_per_cell,
                    nv[0], nv[1], nv[2], F,
                    n_existing_bridges=n_existing)

    # Wall repulsion (6 faces) — skip for periodic boundaries
    if not periodic:
        for i in range(N):
            walls = [
                (0.0, 0, +1), (p.Lx, 0, -1),  # x walls
                (0.0, 1, +1), (p.Ly, 1, -1),  # y walls
                (0.0, 2, +1), (p.Lz, 2, -1),  # z walls
            ]
            if gs.is_circle:
                r = gs.r[i]
                coords = [gs.x[i], gs.y[i], gs.z[i]]
                for wall_pos, axis, sign in walls:
                    if sign > 0:
                        pen = r - (coords[axis] - wall_pos)
                    else:
                        pen = (coords[axis] + r) - wall_pos
                    if pen > 0:
                        Fw = hertz_contact_force(E_star_gw, r, pen)
                        F[i, axis] += sign * Fw
            else:
                for wall_pos, axis, sign in walls:
                    wresult = find_contact_wall_3d(
                        gs.x[i], gs.y[i], gs.z[i],
                        gs.a[i], gs.b[i], gs.c[i],
                        gs.n1[i], gs.n2[i], gs.quat[i], gs.r_bound[i],
                        wall_pos, axis, sign)
                    if wresult is not None:
                        pen, R_local = wresult
                        Fw = hertz_contact_force(E_star_gw, R_local, pen)
                        F[i, axis] += sign * Fw

    # Active noise (vectorized)
    if p.T_active > 0:
        func_mask = gs.gtype[:N] == 0
        n_func = int(np.sum(func_mask))
        if n_func > 0:
            gamma_func = p.drag_scale * gs.r[:N][func_mask]
            noise_amp = np.sqrt(2 * gamma_func * p.T_active / p.dt)
            F[:N, 0][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 1][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 2][func_mask] += noise_amp * rng.standard_normal(n_func)

    return F, torques, contacts


# ══════════════════════════════════════════════════════════════════════
# Time integration (overdamped: γ dx/dt = F  →  dx = F/γ · dt)
# ══════════════════════════════════════════════════════════════════════

def step(gs: GranuleSystem, p: Params, rng, t: float):
    """One overdamped Euler step with cell state evolution and rotation."""
    update_cell_state(gs, p, t, rng)

    if gs.is_3d:
        F, torques, contacts = compute_forces_3d(gs, p, rng)
    else:
        F, torques, contacts = compute_forces(gs, p, rng)

    if gs.is_3d:
        # ── 3D integration (vectorized translational, per-granule quaternion) ──
        gamma = p.drag_scale * gs.r[:gs.N]         # (N,)
        vel = F[:gs.N] / gamma[:, None]             # (N, 3)
        speed = np.sqrt(np.sum(vel**2, axis=1))     # (N,)
        over = speed > p.v_max
        vel[over] *= (p.v_max / speed[over])[:, None]
        gs.vx[:gs.N] = vel[:, 0]
        gs.vy[:gs.N] = vel[:, 1]
        gs.vz[:gs.N] = vel[:, 2]
        dx_step = vel[:, 0] * p.dt
        dy_step = vel[:, 1] * p.dt
        dz_step = vel[:, 2] * p.dt
        gs.x[:gs.N] += dx_step
        gs.y[:gs.N] += dy_step
        gs.z[:gs.N] += dz_step
        # Track unwrapped positions for displacement calculation
        gs.x_unwrap[:gs.N] += dx_step
        gs.y_unwrap[:gs.N] += dy_step
        gs.z_unwrap[:gs.N] += dz_step

        # 3D rotational dynamics (quaternion — must be per-granule)
        if not gs.is_circle:
            for i in range(gs.N):
                I_eff = p.drag_scale_rot * (gs.a[i]**2 + gs.b[i]**2 + gs.c[i]**2) / 3.0
                if I_eff > 1e-20:
                    omega = torques[i] / I_eff
                    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
                    if omega_mag > p.omega_max_3d:
                        omega *= p.omega_max_3d / omega_mag
                    gs.omega_3d[i] = omega
                    gs.quat[i] = quat_integrate(gs.quat[i], omega, p.dt)

        # Boundary handling
        if p.boundary_mode == 'periodic':
            wrap_positions(gs, p)
        else:
            rb = gs.r_bound[:gs.N]
            gs.x[:gs.N] = np.clip(gs.x[:gs.N], rb + 0.5, p.Lx - rb - 0.5)
            gs.y[:gs.N] = np.clip(gs.y[:gs.N], rb + 0.5, p.Ly - rb - 0.5)
            gs.z[:gs.N] = np.clip(gs.z[:gs.N], rb + 0.5, p.Lz - rb - 0.5)
    else:
        # ── 2D integration (vectorized) ──
        gamma = p.drag_scale * gs.r[:gs.N]         # (N,)
        vel = F[:gs.N] / gamma[:, None]             # (N, 2)
        speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        over = speed > p.v_max
        vel[over] *= (p.v_max / speed[over])[:, None]
        gs.vx[:gs.N] = vel[:, 0]
        gs.vy[:gs.N] = vel[:, 1]
        dx_step = vel[:, 0] * p.dt
        dy_step = vel[:, 1] * p.dt
        gs.x[:gs.N] += dx_step
        gs.y[:gs.N] += dy_step
        gs.x_unwrap[:gs.N] += dx_step
        gs.y_unwrap[:gs.N] += dy_step

        # 2D rotational dynamics (superellipses only, vectorized)
        if not gs.is_circle:
            gamma_rot = p.drag_scale_rot * (gs.a[:gs.N]**2 + gs.b[:gs.N]**2) / 2.0
            valid = gamma_rot > 1e-20
            gs.omega[:gs.N] = np.where(valid, torques[:gs.N] / np.where(valid, gamma_rot, 1.0), 0.0)
            gs.omega[:gs.N] = np.clip(gs.omega[:gs.N], -p.omega_max, p.omega_max)
            gs.theta[:gs.N] += gs.omega[:gs.N] * p.dt

        # Boundary handling
        if p.boundary_mode == 'periodic':
            wrap_positions(gs, p)
        else:
            rb = gs.r_bound[:gs.N]
            gs.x[:gs.N] = np.clip(gs.x[:gs.N], rb + 0.5, p.Lx - rb - 0.5)
            gs.y[:gs.N] = np.clip(gs.y[:gs.N], rb + 0.5, p.Ly - rb - 0.5)

    return F, contacts


# ══════════════════════════════════════════════════════════════════════
# Phase field rendering (from particle positions)
# ══════════════════════════════════════════════════════════════════════

def _stamp_circle_2d(X, Y, cx, cy, r_eff_i, w, phi_f, phi_i, gtype):
    """Stamp a single circle profile onto the 2D field."""
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    profile = 0.5 * (1.0 - np.tanh((dist - r_eff_i) / w))
    if gtype == 0:
        np.maximum(phi_f, profile, out=phi_f)
    else:
        np.maximum(phi_i, profile, out=phi_i)


def _stamp_superellipse_2d(X, Y, cx, cy, a, b, n_s, theta, r, w, phi_f, phi_i, gtype):
    """Stamp a single superellipse profile onto the 2D field."""
    bx, by = _world_to_body(X, Y, cx, cy, theta)
    se_val = (np.abs(bx) / a)**n_s + (np.abs(by) / b)**n_s
    n_inv = 1.0 / n_s
    dist_approx = (se_val**n_inv - 1.0) * r
    profile = 0.5 * (1.0 - np.tanh(dist_approx / w))
    if gtype == 0:
        np.maximum(phi_f, profile, out=phi_f)
    else:
        np.maximum(phi_i, profile, out=phi_i)


def _periodic_image_offsets_2d(cx, cy, rb, Lx, Ly):
    """Return list of (dx, dy) offsets for ghost images near periodic boundaries."""
    offsets = [(0, 0)]
    for sx in (-Lx, 0, Lx):
        for sy in (-Ly, 0, Ly):
            if sx == 0 and sy == 0:
                continue
            # Only add if ghost image could overlap the domain
            gx = cx + sx
            gy = cy + sy
            if -rb < gx < Lx + rb and -rb < gy < Ly + rb:
                offsets.append((sx, sy))
    return offsets


def render_fields(gs: GranuleSystem, p: Params):
    """
    Stamp each granule as tanh-profile shape onto grid.
    For circles: radial profile with volume-conserving effective radii.
    For superellipses: implicit-function-based signed distance.
    For periodic boundaries, ghost images are stamped at boundary crossings.
    Returns φ_f, φ_i, φ_v arrays of shape (Ngrid, Ngrid).
    """
    Ng = p.Ngrid
    dx = p.Lx / Ng
    xg = np.linspace(dx/2, p.Lx - dx/2, Ng)
    yg = np.linspace(dx/2, p.Ly - dx/2, Ng)
    X, Y = np.meshgrid(xg, yg, indexing='ij')

    phi_f = np.zeros((Ng, Ng))
    phi_i = np.zeros((Ng, Ng))
    w = p.interface_width
    periodic = (p.boundary_mode == 'periodic')

    if gs.is_circle:
        r_eff, _ = compute_effective_radii(gs)
        for i in range(gs.N):
            if periodic:
                rb = r_eff[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_circle_2d(X, Y, gs.x[i]+sx, gs.y[i]+sy,
                                     r_eff[i], w, phi_f, phi_i, gs.gtype[i])
            else:
                _stamp_circle_2d(X, Y, gs.x[i], gs.y[i],
                                 r_eff[i], w, phi_f, phi_i, gs.gtype[i])
    else:
        for i in range(gs.N):
            if periodic:
                rb = gs.r_bound[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_superellipse_2d(
                        X, Y, gs.x[i]+sx, gs.y[i]+sy,
                        gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                        gs.r[i], w, phi_f, phi_i, gs.gtype[i])
            else:
                _stamp_superellipse_2d(
                    X, Y, gs.x[i], gs.y[i],
                    gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                    gs.r[i], w, phi_f, phi_i, gs.gtype[i])

    # Prevent total > 1
    total = phi_f + phi_i
    over = total > 0.99
    if np.any(over):
        phi_f[over] *= 0.99 / total[over]
        phi_i[over] *= 0.99 / total[over]

    return phi_f, phi_i, 1.0 - phi_f - phi_i


def _stamp_granule_3d(phi_f, phi_i, gs, i, cx, cy, cz, r_eff_3d_i,
                      xg, yg, zg, dx_g, dy_g, dz_g, Ng, w):
    """Stamp one 3D granule image at (cx, cy, cz) onto the field grids."""
    rb = gs.r_bound[i] + 3*w
    ix0 = max(0, int((cx - rb) / dx_g))
    ix1 = min(Ng, int((cx + rb) / dx_g) + 1)
    iy0 = max(0, int((cy - rb) / dy_g))
    iy1 = min(Ng, int((cy + rb) / dy_g) + 1)
    iz0 = max(0, int((cz - rb) / dz_g))
    iz1 = min(Ng, int((cz + rb) / dz_g) + 1)
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return

    X, Y, Z = np.meshgrid(xg[ix0:ix1], yg[iy0:iy1], zg[iz0:iz1],
                           indexing='ij')

    if gs.is_circle:
        dist = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2)
        profile = 0.5 * (1.0 - np.tanh((dist - r_eff_3d_i) / w))
    else:
        dx_l = X - cx
        dy_l = Y - cy
        dz_l = Z - cz
        R_mat = quat_to_rotation_matrix(gs.quat[i])
        bx = R_mat[0,0]*dx_l + R_mat[1,0]*dy_l + R_mat[2,0]*dz_l
        by = R_mat[0,1]*dx_l + R_mat[1,1]*dy_l + R_mat[2,1]*dz_l
        bz = R_mat[0,2]*dx_l + R_mat[1,2]*dy_l + R_mat[2,2]*dz_l

        se_val = ((np.abs(bx/gs.a[i])**gs.n1[i] +
                   np.abs(by/gs.b[i])**gs.n1[i])**(gs.n2[i]/gs.n1[i]) +
                  np.abs(bz/gs.c[i])**gs.n2[i])
        n_inv = 1.0 / gs.n2[i]
        dist_approx = (se_val**n_inv - 1.0) * gs.r[i]
        profile = 0.5 * (1.0 - np.tanh(dist_approx / w))

    if gs.gtype[i] == 0:
        phi_f[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(
            phi_f[ix0:ix1, iy0:iy1, iz0:iz1], profile)
    else:
        phi_i[ix0:ix1, iy0:iy1, iz0:iz1] = np.maximum(
            phi_i[ix0:ix1, iy0:iy1, iz0:iz1], profile)


def render_fields_3d(gs: GranuleSystem, p: Params):
    """
    Stamp each granule onto 3D grid using superellipsoid implicit function.
    For periodic boundaries, ghost images are stamped at boundary crossings.
    Returns φ_f, φ_i, φ_v arrays of shape (Ng, Ng, Ng).
    """
    Ng = p.Ngrid_3d
    dx_g = p.Lx / Ng
    dy_g = p.Ly / Ng
    dz_g = p.Lz / Ng
    xg = np.linspace(dx_g/2, p.Lx - dx_g/2, Ng)
    yg = np.linspace(dy_g/2, p.Ly - dy_g/2, Ng)
    zg = np.linspace(dz_g/2, p.Lz - dz_g/2, Ng)

    phi_f = np.zeros((Ng, Ng, Ng))
    phi_i = np.zeros((Ng, Ng, Ng))
    w = p.interface_width
    periodic = (p.boundary_mode == 'periodic')

    if gs.is_circle:
        r_eff_3d, _ = compute_effective_radii_3d(gs)
    else:
        r_eff_3d = gs.r

    for i in range(gs.N):
        r_eff_i = r_eff_3d[i]
        if periodic:
            rb = gs.r_bound[i] + 3*w
            # Generate ghost offsets for this granule
            for sx in (-p.Lx, 0, p.Lx):
                for sy in (-p.Ly, 0, p.Ly):
                    for sz in (-p.Lz, 0, p.Lz):
                        gx = gs.x[i] + sx
                        gy = gs.y[i] + sy
                        gz = gs.z[i] + sz
                        # Only stamp if image overlaps the domain
                        if (gx + rb > 0 and gx - rb < p.Lx and
                            gy + rb > 0 and gy - rb < p.Ly and
                            gz + rb > 0 and gz - rb < p.Lz):
                            _stamp_granule_3d(
                                phi_f, phi_i, gs, i, gx, gy, gz,
                                r_eff_i, xg, yg, zg,
                                dx_g, dy_g, dz_g, Ng, w)
        else:
            _stamp_granule_3d(
                phi_f, phi_i, gs, i, gs.x[i], gs.y[i], gs.z[i],
                r_eff_i, xg, yg, zg, dx_g, dy_g, dz_g, Ng, w)

    total = phi_f + phi_i
    over = total > 0.99
    if np.any(over):
        phi_f[over] *= 0.99 / total[over]
        phi_i[over] *= 0.99 / total[over]

    return phi_f, phi_i, 1.0 - phi_f - phi_i


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def kozeny_carman_permeability(porosity, d_grain_um):
    """
    Kozeny-Carman permeability estimate.
    K = eps^3 * d^2 / [180 * (1-eps)^2]   [µm²]
    """
    eps = np.clip(porosity, 0.01, 0.99)
    return eps**3 * d_grain_um**2 / (180.0 * (1.0 - eps)**2)


def rcp_fraction_superellipsoid(mean_aspect_ratio):
    """
    Estimate random close packing fraction for superellipsoids.
    For spheres: phi_RCP ~ 0.64.
    For ellipsoids: increases with aspect ratio (Donev et al. 2004).
    """
    # Empirical fit from Donev et al.: phi_RCP ~ 0.64 + 0.08*(AR - 1) up to AR ~ 2
    ar = max(1.0, mean_aspect_ratio)
    return min(0.74, 0.64 + 0.08 * (ar - 1.0))


def connectivity(field, thresh_frac=0.3):
    """Cluster analysis on thresholded field."""
    thr = np.mean(field) + thresh_frac * np.std(field)
    b = (field > thr).astype(int)
    lab, nc = label(b)
    if nc == 0 or b.sum() == 0:
        return 0, 0.0, 0.0
    sz = np.array([np.sum(lab == l) for l in range(1, nc+1)])
    return nc, float(sz.max()/b.sum()), float(b.sum()/b.size)


def compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces):
    m = dict(time=t)
    periodic = (p.boundary_mode == 'periodic')

    # ── Boundary exclusion: compute metrics on inner region only ──
    # For periodic boundaries, use full domain (no exclusion needed)
    bx = 0.0 if periodic else p.boundary_exclusion
    if bx > 0:
        shape = phi_f.shape
        # Index ranges for inner region (exclude bx fraction from each edge)
        lo = [int(bx * s) for s in shape]
        hi = [int((1.0 - bx) * s) for s in shape]
        if phi_f.ndim == 3:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        else:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        pf_inner = phi_f[inner]
        pi_inner = phi_i[inner]
        pv_inner = phi_v[inner]
        # Granule mask: which granules are inside the inner region
        x_lo = bx * p.Lx; x_hi = (1.0 - bx) * p.Lx
        y_lo = bx * p.Ly; y_hi = (1.0 - bx) * p.Ly
        inner_gran = (gs.x >= x_lo) & (gs.x <= x_hi) & (gs.y >= y_lo) & (gs.y <= y_hi)
        if gs.is_3d:
            z_lo = bx * p.Lz; z_hi = (1.0 - bx) * p.Lz
            inner_gran &= (gs.z >= z_lo) & (gs.z <= z_hi)
    else:
        pf_inner, pi_inner, pv_inner = phi_f, phi_i, phi_v
        inner_gran = np.ones(gs.N, dtype=bool)

    m['phi_f_mean'] = float(np.mean(pf_inner))
    m['phi_i_mean'] = float(np.mean(pi_inner))
    m['phi_v_mean'] = float(np.mean(pv_inner))
    m['boundary_exclusion'] = bx
    m['n_granules_inner'] = int(np.sum(inner_gran))

    # Functional connectivity (inner region)
    fn, fl, fc = connectivity(pf_inner, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)

    # Void connectivity (inner region)
    vn, vl, vc = connectivity(pv_inner, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)

    # Inert connectivity (inner region)
    inn, il, _ = connectivity(pi_inner, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    # Tissue: dense functional regions (inner region)
    m['tissue_frac'] = float(np.mean(pf_inner > 0.5))

    # Max cluster area (inner region)
    thr = np.mean(pf_inner) + 0.3*np.std(pf_inner)
    b = (pf_inner > thr).astype(int); lab, nc = label(b)
    dxg = p.Lx / p.Ngrid
    if nc > 0:
        sizes = np.array([np.sum(lab==l) for l in range(1, nc+1)])
        m['func_max_area'] = float(np.max(sizes) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    # Packing in functional-rich region (inner region)
    fr = pf_inner > thr
    pt = pf_inner + pi_inner
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    # Mean force magnitude
    if forces.shape[1] >= 3 and gs.is_3d:
        F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2 + forces[:,2]**2)
    else:
        F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[gs.func_mask])) if np.any(gs.func_mask) else 0.0

    # Overlap & contact diagnostics
    pos = gs.positions()
    max_rb = float(np.max(gs.r_bound))
    if periodic:
        if gs.is_3d:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

    n_contacts = 0
    n_contacts_ff = 0   # functional–functional
    n_contacts_if = 0   # inert–functional
    n_contacts_ii = 0   # inert–inert
    max_overlap_ratio = 0.0
    total_overlap_area = 0.0   # 2D: area; 3D: volume
    n_bridges = 0

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            if gs.is_3d:
                dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                    dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
            else:
                dv[0], dv[1] = minimum_image_disp(dv[0], dv[1], p.Lx, p.Ly)
        d = np.sqrt(np.dot(dv, dv))

        if gs.is_circle:
            overlap = gs.r[i] + gs.r[j] - d
        else:
            if gs.is_3d:
                # 3D superellipsoid: approximate overlap from bounding sphere
                overlap = gs.r_bound[i] + gs.r_bound[j] - d
            else:
                result = find_contact_superellipses(
                    pos[i,0], pos[i,1], gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                    pos[j,0], pos[j,1], gs.a[j], gs.b[j], gs.n_shape[j], gs.theta[j])
                overlap = result[1] if result is not None else 0.0

        if overlap > 0:
            n_contacts += 1
            ti, tj = gs.gtype[i], gs.gtype[j]
            if ti == 0 and tj == 0:
                n_contacts_ff += 1
            elif ti == 1 and tj == 1:
                n_contacts_ii += 1
            else:
                n_contacts_if += 1
            R_min = min(gs.r[i], gs.r[j])
            max_overlap_ratio = max(max_overlap_ratio, overlap / R_min)
            if gs.is_3d:
                if gs.is_circle:
                    total_overlap_area += overlap_lens_volume(gs.r[i], gs.r[j], d)
                else:
                    # Approximate overlap volume for superellipsoids
                    R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                    total_overlap_area += (4.0/3.0) * np.pi * R_eff * overlap**2
            else:
                if gs.is_circle:
                    total_overlap_area += overlap_lens_area(gs.r[i], gs.r[j], d)
                else:
                    # Approximate overlap area for superellipses
                    R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                    total_overlap_area += np.pi * R_eff * overlap

    # Bridge count (functional-functional pairs with attached cells in sensing range)
    cutoff_bridge = 2 * max_rb + p.cell_sense_distance
    pairs_b = tree.query_pairs(cutoff_bridge, output_type='ndarray')
    for idx in range(len(pairs_b)):
        i, j = pairs_b[idx]
        if gs.gtype[i] != 0 or gs.gtype[j] != 0:
            continue
        n_avail_i = max(0.0, gs.n_attached[i] - gs.n_overcrowded[i])
        n_avail_j = max(0.0, gs.n_attached[j] - gs.n_overcrowded[j])
        if n_avail_i < 0.1 or n_avail_j < 0.1:
            continue
        dv = pos[j] - pos[i]
        if periodic:
            if gs.is_3d:
                dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                    dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
            else:
                dv[0], dv[1] = minimum_image_disp(dv[0], dv[1], p.Lx, p.Ly)
        d = np.sqrt(np.dot(dv, dv))
        if gs.is_circle:
            gap = d - gs.r[i] - gs.r[j]
        else:
            gap = d - gs.r_bound[i] - gs.r_bound[j]
            if gap < 0:
                gap = 0.0
        if 0 < gap < p.cell_sense_distance:
            n_bridges += 1

    if gs.is_3d:
        if gs.is_circle:
            total_granule_vol = float(np.sum((4.0/3.0) * np.pi * gs.r**3))
        else:
            # Approximate superellipsoid volume using bounding sphere
            total_granule_vol = float(np.sum((4.0/3.0) * np.pi * gs.r**3))
        total_granule_area = total_granule_vol  # reuse variable name for conservation ratio
    else:
        if gs.is_circle:
            total_granule_area = float(np.sum(np.pi * gs.r**2))
        else:
            total_granule_area = float(sum(
                superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                for i in range(gs.N)))
    m['n_contacts'] = n_contacts
    m['n_contacts_ff'] = n_contacts_ff
    m['n_contacts_if'] = n_contacts_if
    m['n_contacts_ii'] = n_contacts_ii
    m['max_overlap_ratio'] = float(max_overlap_ratio)
    m['total_overlap_area'] = float(total_overlap_area)
    m['area_conservation'] = 1.0 - total_overlap_area / max(total_granule_area, 1e-30)
    if gs.is_3d:
        m['volume_conservation'] = m['area_conservation']  # same ratio, 3D volumes
    m['n_bridges'] = n_bridges

    # ── Shape descriptors (V1.3) ──
    if not gs.is_circle:
        ar_arr = np.maximum(gs.a, gs.b) / np.minimum(gs.a, gs.b)
        elong_arr = 1.0 - np.minimum(gs.a, gs.b) / np.maximum(gs.a, gs.b)
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                          for i in range(gs.N)])
        perims = np.array([superellipse_perimeter(gs.a[i], gs.b[i], gs.n_shape[i])
                           for i in range(gs.N)])
        circularity = 4.0 * np.pi * areas / (perims**2 + 1e-30)
        m['shape_aspect_ratio_mean'] = float(np.mean(ar_arr))
        m['shape_aspect_ratio_std'] = float(np.std(ar_arr))
        m['shape_elongation_mean'] = float(np.mean(elong_arr))
        m['shape_circularity_mean'] = float(np.mean(circularity))
        m['shape_blockiness_mean'] = float(np.mean(gs.n_shape))

    # ── Cell state metrics ──
    func = gs.func_mask
    m['n_attached_total'] = float(np.sum(gs.n_attached[func]))
    m['n_seeded_total'] = float(np.sum(gs.n_cells[func]))
    m['mean_spread_frac'] = float(np.mean(gs.spread_fraction[func])) if np.any(func) else 0.0
    m['mean_fa_maturity'] = float(np.mean(gs.fa_maturity[func])) if np.any(func) else 0.0
    m['n_overcrowded_total'] = float(np.sum(gs.n_overcrowded[func]))

    # ── Per-cell bridge force monitoring (V1.9) ──
    bridging_mask = gs.cell_state == int(CellState.BRIDGING)
    n_bridging_cells = int(np.sum(bridging_mask))
    m['n_bridging_cells'] = n_bridging_cells
    n_locked_in = 0
    if n_bridging_cells > 0:
        cell_F_mag = np.sqrt(gs.cell_fx**2 + gs.cell_fy**2 + gs.cell_fz**2)
        bridge_forces = cell_F_mag[bridging_mask]
        m['bridge_force_mean'] = float(np.mean(bridge_forces))
        m['bridge_force_max'] = float(np.max(bridge_forces))
        m['bridge_force_min'] = float(np.min(bridge_forces))
        m['bridge_force_std'] = float(np.std(bridge_forces))
        # Count locked-in cells (above force threshold)
        n_locked_in = int(np.sum(bridge_forces >= p.bridge_lock_force_threshold))
    else:
        m['bridge_force_mean'] = 0.0
        m['bridge_force_max'] = 0.0
        m['bridge_force_min'] = 0.0
        m['bridge_force_std'] = 0.0
    m['n_locked_in_cells'] = n_locked_in

    # ── Transport metrics (V1.4, inner region) ──
    porosity = float(np.mean(pv_inner))
    m['porosity'] = porosity
    inner_r = gs.r[inner_gran] if np.any(inner_gran) else gs.r
    d_grain = float(np.mean(2 * inner_r))
    m['d_grain_mean'] = d_grain
    m['K_kozeny_carman'] = float(kozeny_carman_permeability(porosity, d_grain))

    # Compaction ratio (inner region)
    phi_solid = float(np.mean(pf_inner) + np.mean(pi_inner))
    m['phi_solid'] = phi_solid
    ar_mean = float(np.mean(np.maximum(gs.a, gs.b) /
                            np.minimum(gs.a, gs.b))) if gs.N > 0 else 1.0
    phi_rcp = rcp_fraction_superellipsoid(ar_mean)
    m['phi_RCP'] = phi_rcp
    m['compaction_ratio'] = phi_solid / phi_rcp if phi_rcp > 0 else 0.0

    # Darcy number
    if gs.is_3d:
        L_char = (p.Lx * p.Ly * p.Lz)**(1.0/3.0)
    else:
        L_char = np.sqrt(p.Lx * p.Ly)
    m['Da_number'] = m['K_kozeny_carman'] / (L_char**2) if L_char > 0 else 0.0

    return m


def compute_displacement(gs, x0, y0, z0=None):
    """RMS displacement from initial positions.
    Uses unwrapped positions for correct displacement under periodic BCs.
    """
    # Use unwrapped coords if available (periodic BCs accumulate wraps)
    dx = gs.x_unwrap - x0; dy = gs.y_unwrap - y0
    if gs.is_3d and z0 is not None:
        dz = gs.z_unwrap - z0
        disp = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        disp = np.sqrt(dx**2 + dy**2)
    return float(np.mean(disp[gs.func_mask])), float(np.mean(disp[gs.inert_mask]))


# ══════════════════════════════════════════════════════════════════════
# Data serialization (V1.5)
# ══════════════════════════════════════════════════════════════════════

def save_snapshot_to_disk(snap_idx, gs, p, t, F, output_dir,
                           save_fields=False, phi_f=None, phi_i=None, phi_v=None,
                           contacts=None):
    """Save one timepoint of simulation data to disk as compressed .npz."""
    snap_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    # Granule data
    data = {
        'time': np.float64(t),
        'x': gs.x.copy(), 'y': gs.y.copy(), 'z': gs.z.copy(),
        'r': gs.r.copy(), 'gtype': gs.gtype.copy(),
        'a': gs.a.copy(), 'b': gs.b.copy(), 'c': gs.c.copy(),
        'n1': gs.n1.copy(), 'n2': gs.n2.copy(),
        'vx': gs.vx.copy(), 'vy': gs.vy.copy(), 'vz': gs.vz.copy(),
        'n_attached': gs.n_attached.copy(),
        'spread_fraction': gs.spread_fraction.copy(),
        'fa_maturity': gs.fa_maturity.copy(),
        'n_overcrowded': gs.n_overcrowded.copy(),
        'n_cells': gs.n_cells.copy(),
        'force_x': F[:, 0].copy(),
        'force_y': F[:, 1].copy(),
    }
    if F.shape[1] > 2:
        data['force_z'] = F[:, 2].copy()
    if gs.is_3d and gs.quat is not None:
        data['quat'] = gs.quat.copy()
    else:
        data['theta'] = gs.theta.copy()

    # Per-cell data (V1.5)
    data['cell_granule_id'] = gs.cell_granule_id.copy()
    data['cell_state'] = gs.cell_state.copy()
    data['cell_theta_local'] = gs.cell_theta_local.copy()
    data['cell_eta_local'] = gs.cell_eta_local.copy()
    data['cell_omega_local'] = gs.cell_omega_local.copy()
    data['cell_fx'] = gs.cell_fx.copy()
    data['cell_fy'] = gs.cell_fy.copy()
    data['cell_fz'] = gs.cell_fz.copy()
    data['cell_bridge_target'] = gs.cell_bridge_target.copy()
    data['cell_bridge_age'] = gs.cell_bridge_age.copy()
    data['cell_contact_area'] = gs.cell_contact_area.copy()
    data['cell_offset'] = gs.cell_offset.copy()

    # V1.5.2: Per-contact data for stress visualization
    if contacts:
        n_c = len(contacts)
        data['contact_i'] = np.array([c['i'] for c in contacts], dtype=np.int32)
        data['contact_j'] = np.array([c['j'] for c in contacts], dtype=np.int32)
        data['contact_cx'] = np.array([c['cx'] for c in contacts])
        data['contact_cy'] = np.array([c['cy'] for c in contacts])
        data['contact_cz'] = np.array([c.get('cz', 0.0) for c in contacts])
        data['contact_nx'] = np.array([c['nx'] for c in contacts])
        data['contact_ny'] = np.array([c['ny'] for c in contacts])
        data['contact_nz'] = np.array([c.get('nz', 0.0) for c in contacts])
        data['contact_overlap'] = np.array([c['overlap'] for c in contacts])
        data['contact_R_eff'] = np.array([c['R_eff'] for c in contacts])
        data['contact_F_normal'] = np.array([c['F_normal'] for c in contacts])
        data['contact_A_contact'] = np.array([c['A_contact'] for c in contacts])

    np.savez_compressed(
        os.path.join(snap_dir, f'snap_{snap_idx:04d}.npz'), **data)

    # Optional phase fields (large in 3D)
    if save_fields and phi_f is not None:
        fields_dir = os.path.join(output_dir, 'fields')
        os.makedirs(fields_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(fields_dir, f'fields_{snap_idx:04d}.npz'),
            phi_f=phi_f, phi_i=phi_i, phi_v=phi_v)


def save_history_to_disk(hist, output_dir):
    """Save scalar metrics history as CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    if not hist:
        return

    # CSV (pandas-friendly)
    keys = list(hist[0].keys())
    csv_path = os.path.join(output_dir, 'history.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(hist)

    # JSON (exact fidelity)
    json_path = os.path.join(output_dir, 'history.json')
    with open(json_path, 'w') as f:
        json.dump(hist, f, indent=2, default=float)


def save_params_metadata(p, gs, output_dir, seed):
    """Save Params and run metadata as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Params as dict
    from dataclasses import asdict
    params_dict = asdict(p)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f, indent=2, default=float)

    # Metadata
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    meta = {
        'version': 'V1.6',
        'git_hash': git_hash,
        'seed': seed,
        'mode': p.mode,
        'n_granules': int(gs.N),
        'n_functional': int(np.sum(gs.func_mask)),
        'n_inert': int(np.sum(gs.inert_mask)),
        'total_cells': int(gs.total_cells),
        'domain': [p.Lx, p.Ly, p.Lz],
        'cell_state_enum': {s.name: int(s.value) for s in CellState},
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def create_archive(output_dir):
    """Create a .tar.gz archive of the output directory for HPC transfer."""
    archive_name = output_dir.rstrip('/') + '.tar.gz'
    parent = os.path.dirname(output_dir) or '.'
    basename = os.path.basename(output_dir)
    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(output_dir, arcname=basename)
    print(f"  Archive: {archive_name}")
    return archive_name


# ══════════════════════════════════════════════════════════════════════
# Data loading (V1.5)
# ══════════════════════════════════════════════════════════════════════

def load_run(run_dir):
    """Load a saved simulation run from disk.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or a .tar.gz archive.

    Returns
    -------
    hist : list of dict
        Scalar metrics time series.
    snaps : list of dict
        Per-timepoint snapshot dicts (numpy arrays).
    p : Params
        Simulation parameters.
    metadata : dict
        Run metadata (version, git hash, seed, etc.).
    """
    # Handle .tar.gz input
    if run_dir.endswith('.tar.gz'):
        extract_dir = run_dir[:-7]  # strip .tar.gz
        if not os.path.isdir(extract_dir):
            with tarfile.open(run_dir, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(run_dir) or '.')
        run_dir = extract_dir

    # Load params
    params_path = os.path.join(run_dir, 'params.json')
    p = Params()
    if os.path.exists(params_path):
        with open(params_path) as f:
            params_dict = json.load(f)
        for k, v in params_dict.items():
            if hasattr(p, k):
                field_type = type(getattr(p, k))
                try:
                    setattr(p, k, field_type(v))
                except (ValueError, TypeError):
                    setattr(p, k, v)

    # Load metadata
    metadata = {}
    meta_path = os.path.join(run_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    # Load history
    hist = []
    hist_json = os.path.join(run_dir, 'history.json')
    hist_csv = os.path.join(run_dir, 'history.csv')
    if os.path.exists(hist_json):
        with open(hist_json) as f:
            hist = json.load(f)
    elif os.path.exists(hist_csv):
        with open(hist_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {}
                for k, v in row.items():
                    try:
                        entry[k] = int(v)
                    except ValueError:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            entry[k] = v
                hist.append(entry)

    # Load snapshots
    snaps = []
    snap_dir = os.path.join(run_dir, 'snapshots')
    if os.path.isdir(snap_dir):
        snap_files = sorted(
            f for f in os.listdir(snap_dir)
            if f.startswith('snap_') and f.endswith('.npz'))
        for sf in snap_files:
            data = dict(np.load(os.path.join(snap_dir, sf), allow_pickle=False))
            # Check for corresponding field file
            idx_str = sf.replace('snap_', '').replace('.npz', '')
            field_file = os.path.join(run_dir, 'fields', f'fields_{idx_str}.npz')
            if os.path.exists(field_file):
                fields = dict(np.load(field_file, allow_pickle=False))
                data.update(fields)
            snaps.append(data)

    return hist, snaps, p, metadata


def load_cells(run_dir, snap_index=None):
    """Load per-cell data from a specific snapshot (or all snapshots).

    Parameters
    ----------
    run_dir : str
        Path to the output directory.
    snap_index : int or None
        If given, load only that snapshot. Otherwise load all.

    Returns
    -------
    dict or list of dict
        Cell data arrays keyed by 'cell_*' names.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')

    if snap_index is not None:
        path = os.path.join(snap_dir, f'snap_{snap_index:04d}.npz')
        data = dict(np.load(path, allow_pickle=False))
        return {k: data[k] for k in data if k.startswith('cell_')}

    results = []
    snap_files = sorted(
        f for f in os.listdir(snap_dir)
        if f.startswith('snap_') and f.endswith('.npz'))
    for sf in snap_files:
        data = dict(np.load(os.path.join(snap_dir, sf), allow_pickle=False))
        results.append({k: data[k] for k in data if k.startswith('cell_')})
    return results


# ══════════════════════════════════════════════════════════════════════
# Main simulation loop
# ══════════════════════════════════════════════════════════════════════

def _warmup_jit(is_3d=False):
    """Trigger Numba JIT compilation of hot-path functions with dummy data."""
    a, b, n = 40.0, 35.0, 2.5
    # 2D geometry + NR solver
    superellipse_point(0.5, a, b, n)
    superellipse_normal_vec(0.5, a, b, n)
    superellipse_curvature_radius(0.5, a, b, n)
    find_contact_superellipses(0.0, 0.0, a, b, n, 0.0,
                                100.0, 0.0, a, b, n, 0.0)
    find_contact_superellipse_wall(50.0, 50.0, a, b, n, 0.1, 0.0, 0, 1)
    hertz_contact_force(5000.0, 20.0, 1.0)
    if is_3d:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        superellipsoid_point(0.3, 0.5, a, b, a, n, n)
        superellipsoid_normal(0.3, 0.5, a, b, a, n, n)
        quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        find_contact_spheres_3d(0.0, 0.0, 0.0, a, 100.0, 0.0, 0.0, a)
        find_contact_superellipsoids_3d(
            0.0, 0.0, 0.0, a, b, a, n, n, q,
            100.0, 0.0, 0.0, a, b, a, n, n, q)


def run(p=None, seed=None):
    if p is None: p = Params()
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"  Using random seed: {seed}")
    rng = np.random.default_rng(seed)

    # JIT warmup: trigger Numba compilation before timing begins
    if HAS_NUMBA and p.shape_enabled:
        _warmup_jit(p.mode == "3D")

    print(f"\n  Mode: {p.mode}")
    print("  Generating packing...")

    # Mode-aware packing generation
    if p.mode == "3D":
        gs = generate_packing_3d(p, seed=seed)
    elif p.mode == "2D-slice":
        gs = generate_packing_2d_slice(p, seed=seed)
    else:
        gs = generate_packing(p, seed=seed)

    # Store initial positions
    x0 = gs.x.copy(); y0 = gs.y.copy()
    z0 = gs.z.copy() if gs.is_3d else None

    n_steps = int(p.t_total / p.dt)
    hist, snaps = [], []

    # V1.5: Initialize output directory and save metadata
    output_dir = p.output_dir
    snap_counter = [0]  # mutable counter for closure
    if p.save_data:
        os.makedirs(output_dir, exist_ok=True)
        save_params_metadata(p, gs, output_dir, seed)

    def save(t, F, contacts=None):
        if gs.is_3d:
            pf, pi, pv = render_fields_3d(gs, p)
        else:
            pf, pi, pv = render_fields(gs, p)
        m = compute_metrics(gs, p, pf, pi, pv, t, F)
        df, di = compute_displacement(gs, x0, y0, z0)
        m['disp_func'] = df; m['disp_inert'] = di
        hist.append(m)
        # Snapshot: mode-aware
        snap = {
            'phi_f': pf.copy(), 'phi_i': pi.copy(), 'phi_v': pv.copy(),
            'x': gs.x.copy(), 'y': gs.y.copy(), 'z': gs.z.copy(),
            'r': gs.r.copy(), 'gtype': gs.gtype.copy(),
            'n_attached': gs.n_attached.copy(),
            'spread_fraction': gs.spread_fraction.copy(),
            'fa_maturity': gs.fa_maturity.copy(),
            'n_overcrowded': gs.n_overcrowded.copy(),
            'a': gs.a.copy(), 'b': gs.b.copy(), 'c': gs.c.copy(),
            'n1': gs.n1.copy(), 'n2': gs.n2.copy(),
            'mode': gs.mode,
            # V1.5.1: Force and cell data for viz scripts
            'force_x': F[:, 0].copy(),
            'force_y': F[:, 1].copy(),
            'cell_state': gs.cell_state.copy(),
            'cell_granule_id': gs.cell_granule_id.copy(),
            'cell_theta_local': gs.cell_theta_local.copy(),
            'cell_fx': gs.cell_fx.copy(),
            'cell_fy': gs.cell_fy.copy(),
            'cell_bridge_target': gs.cell_bridge_target.copy(),
            'cell_bridge_age': gs.cell_bridge_age.copy(),
            'cell_contact_area': gs.cell_contact_area.copy(),
            'cell_offset': gs.cell_offset.copy(),
            # V1.5.2: Per-contact data for stress visualization
            'contacts': contacts if contacts is not None else [],
        }
        if F.shape[1] > 2:
            snap['force_z'] = F[:, 2].copy()
            snap['cell_fz'] = gs.cell_fz.copy()
            snap['cell_eta_local'] = gs.cell_eta_local.copy()
            snap['cell_omega_local'] = gs.cell_omega_local.copy()
        if gs.is_3d and gs.quat is not None:
            snap['quat'] = gs.quat.copy()
        else:
            snap['theta'] = gs.theta.copy()
        # Store n_shape for 2D viz compat
        snap['n_shape'] = gs.n1.copy()
        snaps.append(snap)

        # V1.5: Save to disk
        if p.save_data:
            save_snapshot_to_disk(
                snap_counter[0], gs, p, t, F, output_dir,
                save_fields=p.save_fields,
                phi_f=pf, phi_i=pi, phi_v=pv,
                contacts=contacts)
            snap_counter[0] += 1

        return m

    # Initial save
    update_cell_state(gs, p, 0.0, rng)
    if gs.is_3d:
        F0, _, contacts0 = compute_forces_3d(gs, p, rng)
    else:
        F0, _, contacts0 = compute_forces(gs, p, rng)
    m = save(0.0, F0, contacts0)
    print(f"\n  {'t(h)':>6} {'f_cl':>5} {'f_lf':>6} {'v_cl':>5} "
          f"{'tissue':>7} {'bridges':>7} {'attach':>7} {'spread':>6} "
          f"{'FA_mat':>6} {'disp_f':>7} {'K_KC':>8} "
          f"{'bCells':>6} {'bF_avg':>6} {'lock':>4}")
    print(f"  {0:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} {m['void_nc']:5d} "
          f"{m['tissue_frac']:7.3f} {m['n_bridges']:7d} "
          f"{m['n_attached_total']:7.0f} {m['mean_spread_frac']:6.2f} "
          f"{m['mean_fa_maturity']:6.2f} {m['disp_func']:7.1f} "
          f"{m['K_kozeny_carman']:8.1f} "
          f"{m['n_bridging_cells']:6d} {m['bridge_force_mean']:6.1f} "
          f"{m['n_locked_in_cells']:4d}")

    wall_t0 = timer.time()
    t = 0.0
    for s in range(1, n_steps + 1):
        t += p.dt
        F, contacts = step(gs, p, rng, t)

        if s % p.save_every == 0:
            m = save(t, F, contacts)
            print(f"  {t:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} "
                  f"{m['void_nc']:5d} {m['tissue_frac']:7.3f} "
                  f"{m['n_bridges']:7d} {m['n_attached_total']:7.0f} "
                  f"{m['mean_spread_frac']:6.2f} {m['mean_fa_maturity']:6.2f} "
                  f"{m['disp_func']:7.1f} {m['K_kozeny_carman']:8.1f} "
                  f"{m['n_bridging_cells']:6d} {m['bridge_force_mean']:6.1f} "
                  f"{m['n_locked_in_cells']:4d}")

    elapsed = timer.time() - wall_t0
    print(f"\n  Done in {elapsed:.1f}s ({n_steps} steps, {gs.N} granules)")

    # ── Bridge force diagnostic (V1.9) ──
    if hist:
        final = hist[-1]
        n_bc = final.get('n_bridging_cells', 0)
        bf_mean = final.get('bridge_force_mean', 0.0)
        n_lock = final.get('n_locked_in_cells', 0)
        if n_bc > 0:
            print(f"\n  Bridge force diagnostic:")
            print(f"    Bridging cells:   {n_bc}")
            print(f"    Locked-in cells:  {n_lock}")
            print(f"    Mean bridge force: {bf_mean:.2f} nN")
            print(f"    Expected force:    {p.expected_bridge_force:.1f} nN")
            ratio = bf_mean / p.expected_bridge_force if p.expected_bridge_force > 0 else 0.0
            if ratio < 0.25:
                print(f"    WARNING: Bridge force is {ratio:.0%} of expected "
                      f"({p.expected_bridge_force:.0f} nN). Motor-clutch parameters "
                      f"may need tuning (n_motors={p.n_motors}, "
                      f"F_motor_stall={p.F_motor_stall}, F_max_per_cell={p.F_max_per_cell}).")
            elif ratio < 0.5:
                print(f"    NOTE: Bridge force is {ratio:.0%} of expected. "
                      f"Consider increasing n_motors or F_motor_stall.")

    # V1.5: Save history and create archive
    if p.save_data:
        save_history_to_disk(hist, output_dir)
        # Append wall-clock time to metadata
        meta_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as _mf:
                meta = json.load(_mf)
            meta['wall_time_s'] = round(elapsed, 1)
            meta['n_steps'] = n_steps
            with open(meta_path, 'w') as _mf:
                json.dump(meta, _mf, indent=2)
        n_saved = snap_counter[0]
        print(f"  Data: {n_saved} snapshots saved to {output_dir}/")
        if p.compress_archive:
            create_archive(output_dir)

    return hist, snaps, p, gs


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  GELLS-DEM V1.6: Cell-Driven Granular Rearrangement")
    print("="*65)

    p = Params()
    if p.mode == "3D":
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} × {p.Lz:.0f} µm (3D)")
    elif p.mode == "2D-slice":
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm (2D slice of "
              f"{p.Lx:.0f}×{p.Ly:.0f}×{p.Lz:.0f} 3D)")
    else:
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm (2D)")
    print(f"  R_func={p.R_func_mean:.0f}±{p.R_func_std:.0f} µm, "
          f"R_inert={p.R_inert_mean:.0f}±{p.R_inert_std:.0f} µm")
    print(f"  Cells/granule={p.n_cells_per_granule}, "
          f"d_cell={p.cell_diameter} µm, h_spread={p.cell_height_spread} µm")
    print(f"  Motor-clutch: {p.n_motors} motors × {p.F_motor_stall*1e3:.0f} pN, "
          f"{p.n_clutches} clutches, k_c={p.k_clutch} nN/µm")
    print(f"  Attach onset={p.t_attach_onset} h, "
          f"sense dist={p.cell_sense_distance} µm")
    print_stiffness_info(p)
    print(f"  Friction: τ₀_ii={p.tau_0_ii} Pa, τ₀_if={p.tau_0_if} Pa, "
          f"τ₀_ff={p.tau_0_ff} Pa")
    print(f"  Adhesion: W_ii={p.W_adh_ii*1e3:.1f}, W_if={p.W_adh_if*1e3:.1f}, "
          f"W_ff={p.W_adh_ff*1e3:.1f} mJ/m²")
    print(f"  Simulation: {p.t_total:.0f} h, dt={p.dt:.2f} h")

    hist, snaps, p, gs = run(p)

    h0, hf = hist[0], hist[-1]
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Functional: {h0['func_nc']} → {hf['func_nc']} clusters")
    print(f"    Largest frac: {h0['func_lf']:.1%} → {hf['func_lf']:.1%}")
    ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
          'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
    print(f"    Topology: {ft}")
    print(f"  Void: {h0['void_nc']} → {hf['void_nc']} clusters")
    vt = ('CONTINUOUS' if hf['void_lf'] > 0.8 else
          'FEW POCKETS' if hf['void_nc'] < 6 else 'MANY POCKETS')
    print(f"    Topology: {vt}")
    print(f"  Bridges: {h0['n_bridges']} → {hf['n_bridges']}")
    print(f"  Displacement: func={hf['disp_func']:.1f}µm, inert={hf['disp_inert']:.1f}µm")
    print(f"  Tissue: {h0['tissue_frac']:.1%} → {hf['tissue_frac']:.1%}")
    print(f"  Max cluster area: {h0['func_max_area']:.0f} → {hf['func_max_area']:.0f} µm²")
    print(f"  Contacts: {hf['n_contacts']} total "
          f"(ff={hf['n_contacts_ff']}, if={hf['n_contacts_if']}, "
          f"ii={hf['n_contacts_ii']})")
    print(f"    max δ/R = {hf['max_overlap_ratio']*100:.1f}%")
    print(f"  Area conservation: {hf['area_conservation']:.4f} "
          f"(overlap area = {hf['total_overlap_area']:.1f} µm²)")
    print(f"  Cell state:")
    print(f"    Attached: {hf['n_attached_total']:.0f} / {hf['n_seeded_total']:.0f}")
    print(f"    Spread: {hf['mean_spread_frac']:.0%}, FA maturity: {hf['mean_fa_maturity']:.0%}")
    print(f"    Overcrowded: {hf['n_overcrowded_total']:.0f}")
    print(f"  Transport (V1.4):")
    print(f"    Porosity: {h0['porosity']:.3f} → {hf['porosity']:.3f}")
    print(f"    K (Kozeny-Carman): {h0['K_kozeny_carman']:.1f} → "
          f"{hf['K_kozeny_carman']:.1f} µm²")
    print(f"    Compaction ratio: {h0['compaction_ratio']:.3f} → "
          f"{hf['compaction_ratio']:.3f} (φ_RCP={hf['phi_RCP']:.3f})")
    print(f"    Darcy number: {hf['Da_number']:.2e}")