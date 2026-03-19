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
    cell_surface_coverage: float = 0.0  # target surface coverage fraction (0=use n_cells_per_granule; 0.5–1.5 typical; >1 = stacking)

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
    bridge_contact_factor: float = 5.0         # path factor: boost at direct granule contact
    bridge_decay_length: float = 10.0          # µm, characteristic filopodia reach (void decay length)
    bridge_inert_factor: float = 0.1           # path factor: penalty when inert blocks line-of-sight
    bridge_commit_angle: float = 0.5           # rad (~29°), angular threshold for MIGRATING→BRIDGING
    bridge_directed_speed_mult: float = 2.0    # directed crawl speed = cell_migration_speed × this
    bridge_exclusion_angle: float = 0.8        # rad (~46°), min angular separation between bridges on same granule→target
    bridge_alignment_rate: float = 0.3         # 1/h, rate of stress fiber alignment along bridge axis
    bridge_alignment_min: float = 0.3          # initial alignment factor when bridge first forms
    cell_stacking_max: float = 3.0             # max cell layers before overcrowding (1.0 = monolayer)
    overcrowd_senescence_time: float = 12.0    # h, sustained overcrowding before senescence

    # ── Contact mechanics (Hertzian + MC-DEM) ──
    E_modulus: float = 10.0         # kPa, Young's modulus of hydrogel
    poisson_ratio: float = 0.45     # Poisson's ratio (hydrogels ~0.4-0.5)
    mc_dem_enabled: bool = True     # enable multi-contact DEM stiffening (Giannis 2021)
    mc_dem_kappa_max: float = 5.0   # max confinement correction factor (caps stiffening)

    # ── LS-DEM deformable granules (V2.3, Henzel & Karapiperis 2026) ──
    deformable_enabled: bool = False       # master switch (False = V2.2 rigid)
    n_def_modes: int = 2                   # deformation modes per particle (2D: 2, 3D: 3)
    sdf_resolution: int = 32              # SDF grid points per axis (body frame)
    n_surface_nodes: int = 128            # surface discretization nodes (2D)
    n_surface_nodes_3d: int = 512         # surface discretization nodes (3D)
    sdf_padding: float = 1.3             # SDF grid extent = padding * max_semi_axis
    def_drag_scale: float = 0.1          # deformation drag γ_def scaling
    def_eps_max: float = 0.3             # max deformation amplitude |ε_α| cap

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
    max_overlap_frac: float = 0.15   # max allowed overlap as fraction of min(ri,rj)

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
    packing_settle_steps: int = 400  # inflation steps to reach target radii (jammed packing)
    packing_relax_substeps: int = 15  # overlap relaxation sub-steps per inflation step
    packing_inflate_phi_safe: float = 0.20  # initial deflated packing fraction for RSA

    # ── Rendering ──
    Ngrid: int = 200                # grid for 2D field rendering
    Ngrid_3d: int = 80              # grid for 3D field rendering (Ngrid^3 voxels)
    interface_width: float = 3.0    # µm, tanh smoothing

    # ── Performance ──
    use_numba: bool = True          # use Numba JIT if available

    # ── Data serialization (V1.5, updated V2.6) ──
    save_data: bool = True              # save simulation data to disk
    save_fields: bool = False           # skip field grids during sim; reconstruct at plot time
    output_dir: str = "results/default" # output directory for serialized data
    compress_archive: bool = True       # create .tar.gz at end of simulation
    resume_from: str = ""               # path to run directory to resume from (empty = fresh)

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
    MIGRATING = 5       # directed crawl toward bridge target (V2.3)


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

        # ── LS-DEM deformation state (V2.3, allocated by lsdem.init_lsdem) ──
        self.epsilon = None              # (N, n_def_modes) deformation amplitudes
        self.d_epsilon = None            # (N, n_def_modes) time derivatives
        self.sdf_grids = None            # list of N ndarrays (body-frame SDF)
        self.sdf_extents = None          # (N, ndim, 2) grid bounding boxes
        self.surface_nodes_body = None   # (N, n_nodes, ndim) reference positions
        self.surface_normals_body = None # (N, n_nodes, ndim) outward normals
        self.surface_node_areas = None   # (N, n_nodes) area weights
        self.mode_shapes_at_nodes = None # (N, n_nodes, n_modes, ndim)
        self.K_def = None                # (N, n_modes, n_modes) stiffness matrices
        self.F_eps_accum = None          # (N, n_modes) accumulated deformation forces

        # ── JKR contact clip planes for rendering (V2.3) ──
        # Each element is a list of (nx, ny [,nz], clip_distance) tuples.
        # Populated during compute_forces(), consumed by render_fields().
        self.contact_clips = [[] for _ in range(self.N)]

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
        self.cell_bridge_locked = np.zeros(total_cells, dtype=np.bool_)    # persistent lock-in flag
        self.cell_overcrowd_age = np.zeros(total_cells, dtype=np.float64)  # hours in overcrowded state
        self.cell_alignment = np.zeros(total_cells, dtype=np.float64)      # stress fiber alignment (0–1)

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


# ── JKR (Johnson-Kendall-Roberts) adhesive contact model ──────────
# Correct model for soft hydrogels with high Tabor parameter μ_T >> 1.
# Reduces exactly to Hertz when W_adhesion = 0.

@njit(cache=True)
def jkr_force_from_overlap(delta_um, R_eff_um, E_star_Pa, W_Jm2):
    """
    JKR contact force and contact radius from overlap δ.

    Given overlap δ [µm], solve for contact radius a via Newton-Raphson on:
        δ = a²/R* − √(2πW a / E*)
    Then compute JKR force:
        F = (4/3)E* a³/R* − √(8πW E* a³)

    Units: δ [µm], R_eff [µm], E_star [Pa], W [J/m²]
    Returns: (F [nN], a [µm])

    When W=0, reduces to Hertz: F = (4/3)E*√R* δ^{3/2}, a = √(R*δ).
    """
    E_s = E_star_Pa * 1e-3   # nN/µm²
    W_s = W_Jm2              # nN/µm  (1 J/m² = 1 nN/µm)

    # Pure Hertz fast-path when no adhesion
    if W_s < 1e-15:
        if delta_um <= 0.0:
            return 0.0, 0.0
        a = np.sqrt(R_eff_um * delta_um)
        F = (4.0 / 3.0) * E_s * a**3 / R_eff_um
        return F, a

    # No contact and no adhesion pull
    if delta_um <= 0.0:
        # Check adhesive pull at zero overlap
        a = (6.0 * np.pi * W_s * R_eff_um**2 / E_s) ** (1.0 / 3.0)
        F = (4.0 / 3.0) * E_s * a**3 / R_eff_um - np.sqrt(
            8.0 * np.pi * W_s * E_s * a**3)
        if F >= 0.0:
            return 0.0, 0.0  # repulsive at zero gap — no contact
        # Adhesive pull — but only within a small range
        if delta_um < -0.1 * R_eff_um:
            return 0.0, 0.0

    # Initial guess for contact radius
    if delta_um > 0.0:
        a = np.sqrt(R_eff_um * delta_um)
    else:
        a = (6.0 * np.pi * W_s * R_eff_um**2 / E_s) ** (1.0 / 3.0)

    # Newton-Raphson: solve δ = a²/R* − √(2πW a / E*) for a
    for _ in range(30):
        if a < 1e-12:
            a = 1e-6
        sqrt_term = np.sqrt(2.0 * np.pi * W_s * a / E_s) if a > 0 else 0.0
        f = a**2 / R_eff_um - sqrt_term - delta_um
        if a > 1e-12:
            dfda = 2.0 * a / R_eff_um - np.sqrt(
                np.pi * W_s / (2.0 * E_s * a))
        else:
            dfda = 2.0 * a / R_eff_um
        if abs(dfda) < 1e-30:
            break
        a_new = a - f / dfda
        if a_new < 0.0:
            a_new = a * 0.5
        if abs(a_new - a) < 1e-8:
            a = a_new
            break
        a = a_new

    if a < 1e-10:
        return 0.0, 0.0

    # JKR force from contact radius
    F = (4.0 / 3.0) * E_s * a**3 / R_eff_um - np.sqrt(
        8.0 * np.pi * W_s * E_s * a**3)
    return F, a


@njit(cache=True)
def jkr_contact_radius(F_ext_nN, R_eff_um, E_star_Pa, W_Jm2):
    """
    JKR contact radius from applied external force.

    a³ = (R*/E*) [F + 3πWR* + √(6πWR*F + (3πWR*)²)]

    Units: F_ext [nN], R_eff [µm], E_star [Pa], W [J/m²]
    Returns: a [µm]
    """
    E_s = E_star_Pa * 1e-3
    W_s = W_Jm2
    term1 = 3.0 * np.pi * W_s * R_eff_um
    inner = 6.0 * np.pi * W_s * R_eff_um * F_ext_nN + term1**2
    if inner < 0.0:
        inner = 0.0
    a_cubed = (R_eff_um / E_s) * (F_ext_nN + term1 + np.sqrt(inner))
    if a_cubed < 0.0:
        return 0.0
    return a_cubed ** (1.0 / 3.0)


@njit(cache=True)
def jkr_pulloff_force(R_eff_um, W_Jm2):
    """Pull-off force: F_po = (3/2)πWR* [nN]."""
    return 1.5 * np.pi * W_Jm2 * R_eff_um


def mc_dem_correction(contacts, gs, p, E_star_gg, F):
    """Apply MC-DEM multi-contact stiffening correction (Giannis et al. 2021).

    When a soft particle has multiple contacts, each contact sees a stiffer
    response due to volumetric confinement.  The correction factor per particle:

        ε_V,i = Σ_c δ_c / (2 R_i)           volumetric overlap strain
        κ_i   = 1 + ν/(1-2ν) · ε_V,i        confinement multiplier

    For each contact between i,j the Hertz repulsion is scaled by
    κ_ij = (κ_i + κ_j)/2.  Only the repulsive (Hertz) component is scaled;
    adhesion and friction are left unchanged.

    Works for both 2D (F shape N×2) and 3D (F shape N×3).
    """
    N = gs.N
    if len(contacts) == 0:
        return
    ndim = F.shape[1]  # 2 or 3

    # Per-particle volumetric overlap strain
    eps_V = np.zeros(N)
    for c in contacts:
        delta = c['overlap']
        if delta > 0:
            eps_V[c['i']] += delta / (2.0 * gs.r[c['i']])
            eps_V[c['j']] += delta / (2.0 * gs.r[c['j']])

    # Confinement correction factor: κ = 1 + ν/(1−2ν) · ε_V
    nu = p.poisson_ratio
    c_mc = nu / max(1.0 - 2.0 * nu, 0.01)
    kappa = np.minimum(1.0 + c_mc * eps_V, p.mc_dem_kappa_max)

    # Apply correction to Hertzian component of each contact
    for c in contacts:
        delta = c['overlap']
        if delta <= 0:
            continue
        i, j = c['i'], c['j']
        kappa_ij = 0.5 * (kappa[i] + kappa[j])
        if kappa_ij < 1.001:
            continue
        Fc_hertz = hertz_contact_force(E_star_gg, c['R_eff'], delta)
        dF = (kappa_ij - 1.0) * Fc_hertz
        if ndim == 3:
            n = np.array([c['nx'], c['ny'], c['nz']])
        else:
            n = np.array([c['nx'], c['ny']])
        F[i] -= dF * n
        F[j] += dF * n
        # Update stored contact force for visualization
        c['F_normal'] += dF
        c['kappa'] = kappa_ij


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


def compute_effective_radii_3d(gs, p=None):
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
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_vol = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
        d = np.sqrt(np.dot(dv, dv))
        if d < gs.r[i] + gs.r[j]:
            V_lens = overlap_lens_volume(gs.r[i], gs.r[j], d)
            frac_i = gs.r[i]**3 / (gs.r[i]**3 + gs.r[j]**3)
            overlap_vol[i] += frac_i * V_lens
            overlap_vol[j] += (1.0 - frac_i) * V_lens

    r_eff = (gs.r**3 + 3.0 * overlap_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return r_eff, overlap_vol


def compute_effective_radii(gs, p=None):
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
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_area = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if periodic:
            dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
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


def cells_from_surface_coverage(R, cell_d, cell_h, coverage, mode="3D"):
    """Compute number of cells for a target surface coverage fraction.

    coverage = 1.0 is a full monolayer of spread cells; >1.0 means stacking.
    Uses the fully-spread cell footprint as the reference area.

    3D/2D-slice: uses sphere surface area 4πR².
    2D:          uses projected disk area πR².
    """
    A_cell_spread = cell_projected_area(1.0, cell_d, cell_h)
    if mode in ("3D", "2D-slice"):
        A_surface = 4.0 * np.pi * R ** 2
    else:
        A_surface = np.pi * R ** 2
    return max(1, int(round(A_surface * coverage / A_cell_spread)))


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
        if p.cell_surface_coverage > 0:
            cap = cells_from_surface_coverage(
                gs.r[i], p.cell_diameter, p.cell_height_spread,
                p.cell_surface_coverage, p.mode)
        else:
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
                # Stress fiber alignment: cells elongate along bridge axis over time
                gs.cell_alignment[ci] += p.bridge_alignment_rate * p.dt * (
                    1.0 - gs.cell_alignment[ci])
                gs.cell_alignment[ci] = min(gs.cell_alignment[ci], 1.0)
                # Check force magnitude from previous timestep (still in cell_fx/fy/fz)
                F_mag = np.sqrt(gs.cell_fx[ci]**2 + gs.cell_fy[ci]**2
                                + gs.cell_fz[ci]**2)
                # Persistent lock-in: once force exceeds threshold, bridge is
                # permanently locked and immune to senescence (even if force
                # drops later due to rearrangement).  Only gap rupture can
                # break a locked bridge.
                if F_mag >= p.bridge_lock_force_threshold:
                    gs.cell_bridge_locked[ci] = True
                if gs.cell_bridge_locked[ci]:
                    continue  # locked in permanently, skip senescence check
                # Sustained mechanical load without lock-in → senescence
                if gs.cell_bridge_age[ci] >= p.bridge_senescence_time:
                    gs.cell_state[ci] = int(CellState.SENESCENT)
                    gs.cell_bridge_target[ci] = -1
                    gs.cell_bridge_age[ci] = 0.0
                    gs.cell_bridge_locked[ci] = False
                    gs.cell_alignment[ci] = 0.0
                continue  # don't overwrite bridge state

            # ── Directed migration for MIGRATING cells (V2.3) ──
            if gs.cell_state[ci] == int(CellState.MIGRATING):
                tgt = int(gs.cell_bridge_target[ci])
                if tgt < 0 or tgt >= gs.N:
                    # Invalid target — revert
                    gs.cell_state[ci] = int(CellState.PROLIFERATING)
                    gs.cell_bridge_target[ci] = -1
                    continue

                # Abandonment: target out of sensing range?
                gap = _compute_gap(gs, i, tgt, p)
                if gap > p.cell_sense_distance:
                    gs.cell_state[ci] = int(CellState.PROLIFERATING)
                    gs.cell_bridge_target[ci] = -1
                    gs.cell_bridge_age[ci] = 0.0
                    continue

                # Angular distance to contact azimuth
                if gs.mode == "3D":
                    ang_d, tgt_eta, tgt_omega = _angular_distance_to_target_3d(
                        gs, ci, tgt, p)
                else:
                    ang_d, tgt_theta, signed_diff = _angular_distance_to_target_2d(
                        gs, ci, tgt, p)

                # Arrival: close enough to contact point → BRIDGING
                if ang_d < p.bridge_commit_angle:
                    gs.cell_state[ci] = int(CellState.BRIDGING)
                    gs.cell_bridge_age[ci] = 0.0  # reset for force ramp
                    gs.cell_alignment[ci] = p.bridge_alignment_min
                    continue

                # Directed migration step
                directed_speed = p.cell_migration_speed * p.bridge_directed_speed_mult
                r_eff = max(gs.r[i], 1.0)
                step = directed_speed * p.dt / r_eff  # radians per timestep

                if gs.mode == "3D":
                    # Great-circle slerp toward target
                    cell_eta = gs.cell_eta_local[ci]
                    cell_omega = gs.cell_omega_local[ci]
                    p_curr = np.array([np.cos(cell_eta) * np.cos(cell_omega),
                                       np.cos(cell_eta) * np.sin(cell_omega),
                                       np.sin(cell_eta)])
                    p_tgt = np.array([np.cos(tgt_eta) * np.cos(tgt_omega),
                                      np.cos(tgt_eta) * np.sin(tgt_omega),
                                      np.sin(tgt_eta)])
                    if ang_d > 1e-8:
                        frac = min(step, ang_d) / ang_d
                        sin_a = np.sin(ang_d)
                        if sin_a > 1e-12:
                            p_new = (np.sin((1 - frac) * ang_d) * p_curr
                                     + np.sin(frac * ang_d) * p_tgt) / sin_a
                        else:
                            p_new = p_curr
                        new_omega = np.arctan2(p_new[1], p_new[0]) % (2 * np.pi)
                        r_xy = np.sqrt(p_new[0]**2 + p_new[1]**2)
                        new_eta = np.arctan2(p_new[2], r_xy)
                        gs.cell_eta_local[ci] = np.clip(
                            new_eta, -0.85 * np.pi / 2, 0.85 * np.pi / 2)
                        gs.cell_omega_local[ci] = new_omega
                else:
                    # 2D: arc step toward target theta
                    actual_step = min(step, abs(signed_diff))
                    gs.cell_theta_local[ci] += np.sign(signed_diff) * actual_step
                    gs.cell_theta_local[ci] %= (2 * np.pi)

                gs.cell_bridge_age[ci] += p.dt  # track migration time
                continue  # don't overwrite migrating state

            # ── Already senescent stays senescent ──
            if gs.cell_state[ci] == int(CellState.SENESCENT):
                continue

            # ── Normal state assignment (with stacking tolerance) ──
            # Monolayer capacity from overcrowding check
            if p.cell_surface_coverage > 0:
                cap = cells_from_surface_coverage(
                    gs.r[i], p.cell_diameter, p.cell_height_spread,
                    p.cell_surface_coverage, p.mode)
            else:
                A_cell_cap = cell_projected_area(sf, p.cell_diameter, p.cell_height_spread)
                cap = max_cells_on_granule(gs.r[i], A_cell_cap, p.cell_coverage)
            effective_cap = int(round(cap * p.cell_stacking_max))

            if k >= effective_cap:
                # Beyond max stacking — immediate senescence
                gs.cell_state[ci] = int(CellState.SENESCENT)
                gs.cell_overcrowd_age[ci] = 0.0
            elif k >= cap and n_over > 0:
                # Overcrowded but within stacking tolerance — delayed senescence
                gs.cell_overcrowd_age[ci] += p.dt
                if gs.cell_overcrowd_age[ci] >= p.overcrowd_senescence_time:
                    gs.cell_state[ci] = int(CellState.SENESCENT)
                    gs.cell_overcrowd_age[ci] = 0.0
                # else: stay in current mobile state (crawling on other cells)
            elif sf >= 0.95 and fa >= 0.5:
                gs.cell_state[ci] = int(CellState.PROLIFERATING)
                gs.cell_overcrowd_age[ci] = 0.0
            elif sf > 0.0:
                gs.cell_state[ci] = int(CellState.SPREADING)
                gs.cell_overcrowd_age[ci] = 0.0
            else:
                gs.cell_state[ci] = int(CellState.ATTACHED)
                gs.cell_overcrowd_age[ci] = 0.0

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
    """Inflate granules from deflated RSA state to target radii (Lubachevsky-Stillinger).

    The system starts with all granules at reduced radii (set by generate_packing).
    Target radii are stored in gs._target_a/b/r.  This function:
      1. Linearly inflates radii from current to target over packing_settle_steps.
      2. At each inflation step, runs packing_relax_substeps of overlap relaxation
         using vectorised Hertz-like repulsion.
      3. For wall BCs, also applies centripetal attraction.
    Result: a jammed packing at the target solid fraction with Z ≈ 3–4.

    Falls back to simple centripetal settle if no deflation was applied.
    """
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    Lx, Ly = p.Lx, p.Ly
    n_inflate = p.packing_settle_steps
    n_relax = p.packing_relax_substeps
    dt_settle = 0.02
    k_rep = 5.0        # Hertzian repulsion stiffness
    v_cap_frac = 0.3   # max displacement per sub-step = v_cap_frac * mean_r

    # Retrieve target radii stored by generate_packing
    target_a = getattr(gs, '_target_a', gs.a.copy())
    target_b = getattr(gs, '_target_b', gs.b.copy())
    target_r = getattr(gs, '_target_r', gs.r.copy())

    # Compute initial deflation factor from current vs target
    ratio = gs.r[:N] / np.maximum(target_r[:N], 1e-6)
    alpha_start = float(np.median(ratio))
    alpha_start = max(alpha_start, 0.1)

    mean_r_target = float(np.mean(target_r[:N]))
    v_cap = v_cap_frac * mean_r_target

    print(f"  Settling 2D packing ({n_inflate} inflate × {n_relax} relax, "
          f"α={alpha_start:.2f}→1.00)...")

    cx_dom, cy_dom = Lx / 2, Ly / 2

    for step in range(n_inflate):
        # Quadratic ease-in: spend more time near α=1 where jamming is hard
        t = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t)**2)

        # Inflate radii
        gs.a[:N] = target_a[:N] * alpha
        gs.b[:N] = target_b[:N] * alpha
        gs.r[:N] = target_r[:N] * alpha
        gs.r_bound[:N] = np.maximum(gs.a[:N], gs.b[:N])

        max_rb = float(np.max(gs.r_bound[:N]))

        # Overlap relaxation sub-loop
        for sub in range(n_relax):
            pos = gs.positions()  # (N, 2)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall BCs: centripetal attraction (decays over inflation)
            if not periodic:
                attract = max(0.3 * (1.0 - t), 0.02)
                dx_c = cx_dom - pos[:, 0]
                dy_c = cy_dom - pos[:, 1]
                d_c = np.sqrt(dx_c**2 + dy_c**2) + 1e-12
                scale = attract * gs.r_bound[:N] / d_c
                fx += scale * dx_c
                fy += scale * dy_c

            # Neighbour search
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly])
            else:
                tree = cKDTree(pos)
            pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

            max_overlap = 0.0
            if len(pairs) > 0:
                pi = pairs[:, 0]
                pj = pairs[:, 1]
                dx = pos[pj, 0] - pos[pi, 0]
                dy = pos[pj, 1] - pos[pi, 1]
                if periodic:
                    dx -= Lx * np.round(dx / Lx)
                    dy -= Ly * np.round(dy / Ly)
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
                mask = overlap > 0
                if np.any(mask):
                    ov = overlap[mask]
                    max_overlap = float(np.max(ov))
                    inv_d = 1.0 / d[mask]
                    nx = dx[mask] * inv_d
                    ny = dy[mask] * inv_d
                    f = k_rep * ov ** 1.5
                    np.add.at(fx, pi[mask], -f * nx)
                    np.add.at(fy, pi[mask], -f * ny)
                    np.add.at(fx, pj[mask],  f * nx)
                    np.add.at(fy, pj[mask],  f * ny)

            # Overdamped move with velocity cap
            f_mag = np.sqrt(fx*fx + fy*fy) + 1e-12
            scale = np.minimum(dt_settle, v_cap / f_mag)
            gs.x[:N] += fx * scale
            gs.y[:N] += fy * scale

            # Boundary handling
            if periodic:
                gs.x[:N] %= Lx
                gs.y[:N] %= Ly
            else:
                rb = gs.r_bound[:N]
                gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
                gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)

            # Early exit if overlaps are small
            if max_overlap < 0.01 * mean_r_target:
                break

        # Progress
        if step % 100 == 0 or step == n_inflate - 1:
            pos = gs.positions()
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly])
            else:
                tree = cKDTree(pos)
            cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
            n_contacts = 0
            if len(cpairs) > 0:
                pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
                dx_c = pos[pj_c, 0] - pos[pi_c, 0]
                dy_c = pos[pj_c, 1] - pos[pi_c, 1]
                if periodic:
                    dx_c -= Lx * np.round(dx_c / Lx)
                    dy_c -= Ly * np.round(dy_c / Ly)
                d_c = np.sqrt(dx_c**2 + dy_c**2)
                gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
                n_contacts = int(np.sum(gaps < 1.0))
            Z = 2 * n_contacts / max(N, 1)
            print(f"    step {step}/{n_inflate}: α={alpha:.3f}, "
                  f"max_overlap={max_overlap:.1f} µm, Z={Z:.1f}")

    # ── Post-inflation relaxation: resolve remaining deep overlaps ──
    overlap_tol = 0.05 * mean_r_target
    max_post_relax = 1000
    max_rb = float(np.max(gs.r_bound[:N]))
    for extra in range(max_post_relax):
        pos = gs.positions()
        fx = np.zeros(N)
        fy = np.zeros(N)

        if periodic:
            tree = cKDTree(pos, boxsize=[Lx, Ly])
        else:
            tree = cKDTree(pos)
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

        max_overlap = 0.0
        if len(pairs) > 0:
            pi = pairs[:, 0]
            pj = pairs[:, 1]
            dx = pos[pj, 0] - pos[pi, 0]
            dy = pos[pj, 1] - pos[pi, 1]
            if periodic:
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
            mask = overlap > 0
            if np.any(mask):
                ov = overlap[mask]
                max_overlap = float(np.max(ov))
                inv_d = 1.0 / d[mask]
                nx = dx[mask] * inv_d
                ny = dy[mask] * inv_d
                f = k_rep * ov ** 1.5
                np.add.at(fx, pi[mask], -f * nx)
                np.add.at(fy, pi[mask], -f * ny)
                np.add.at(fx, pj[mask],  f * nx)
                np.add.at(fy, pj[mask],  f * ny)

        if max_overlap < overlap_tol:
            break

        f_mag = np.sqrt(fx*fx + fy*fy) + 1e-12
        scale = np.minimum(dt_settle, v_cap / f_mag)
        gs.x[:N] += fx * scale
        gs.y[:N] += fy * scale

        if periodic:
            gs.x[:N] %= Lx
            gs.y[:N] %= Ly
        else:
            rb = gs.r_bound[:N]
            gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
            gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)

    # Final contact count
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[Lx, Ly])
    else:
        tree = cKDTree(pos)
    cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
    n_contacts = 0
    if len(cpairs) > 0:
        pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
        dx_c = pos[pj_c, 0] - pos[pi_c, 0]
        dy_c = pos[pj_c, 1] - pos[pi_c, 1]
        if periodic:
            dx_c -= Lx * np.round(dx_c / Lx)
            dy_c -= Ly * np.round(dy_c / Ly)
        d_c = np.sqrt(dx_c**2 + dy_c**2)
        gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
        n_contacts = int(np.sum(gaps < 1.0))
    Z = 2 * n_contacts / max(N, 1)
    print(f"  Settle done: {n_contacts} contacts, Z={Z:.1f} "
          f"({extra+1 if max_overlap >= overlap_tol else extra} post-relax steps)")


def _settle_packing_3d(gs, p):
    """Inflate granules from deflated RSA state to target radii (Lubachevsky-Stillinger).

    The system starts with all granules at reduced radii (set by generate_packing_3d).
    Target radii are stored in gs._target_a/b/c/r.  This function:
      1. Linearly inflates radii from current to target over packing_settle_steps.
      2. At each inflation step, runs packing_relax_substeps of overlap relaxation
         using vectorised Hertz-like repulsion.
      3. For wall BCs, also applies centripetal attraction.
    Result: a jammed packing at the target solid fraction with Z ≈ 4–6.
    """
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    Lx, Ly, Lz = p.Lx, p.Ly, p.Lz
    n_inflate = p.packing_settle_steps
    n_relax = p.packing_relax_substeps
    dt_settle = 0.02
    k_rep = 5.0        # Hertzian repulsion stiffness
    v_cap_frac = 0.3   # max displacement per sub-step = v_cap_frac * mean_r

    # Retrieve target radii stored by generate_packing_3d
    target_a = getattr(gs, '_target_a', gs.a.copy())
    target_b = getattr(gs, '_target_b', gs.b.copy())
    target_c = getattr(gs, '_target_c', gs.c.copy())
    target_r = getattr(gs, '_target_r', gs.r.copy())

    # Compute initial deflation factor from current vs target
    ratio = gs.r[:N] / np.maximum(target_r[:N], 1e-6)
    alpha_start = float(np.median(ratio))
    alpha_start = max(alpha_start, 0.1)

    mean_r_target = float(np.mean(target_r[:N]))
    v_cap = v_cap_frac * mean_r_target

    print(f"  Settling 3D packing ({n_inflate} inflate × {n_relax} relax, "
          f"α={alpha_start:.2f}→1.00)...")

    cx_dom, cy_dom, cz_dom = Lx / 2, Ly / 2, Lz / 2

    for step in range(n_inflate):
        # Quadratic ease-in: spend more time near α=1 where jamming is hard
        t = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t)**2)

        # Inflate radii
        gs.a[:N] = target_a[:N] * alpha
        gs.b[:N] = target_b[:N] * alpha
        gs.c[:N] = target_c[:N] * alpha
        gs.r[:N] = target_r[:N] * alpha
        gs.r_bound[:N] = np.maximum(np.maximum(gs.a[:N], gs.b[:N]), gs.c[:N])

        max_rb = float(np.max(gs.r_bound[:N]))

        # Overlap relaxation sub-loop
        for sub in range(n_relax):
            pos = gs.positions()
            fx = np.zeros(N)
            fy = np.zeros(N)
            fz = np.zeros(N)

            # Wall BCs: centripetal attraction (decays over inflation)
            if not periodic:
                attract = max(0.3 * (1.0 - t), 0.02)
                dx_c = cx_dom - pos[:, 0]
                dy_c = cy_dom - pos[:, 1]
                dz_c = cz_dom - pos[:, 2]
                d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2) + 1e-12
                scale = attract * gs.r_bound[:N] / d_c
                fx += scale * dx_c
                fy += scale * dy_c
                fz += scale * dz_c

            # Neighbour search (use r_bound for cutoff — conservative)
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
            else:
                tree = cKDTree(pos)
            pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

            max_overlap = 0.0
            if len(pairs) > 0:
                pi = pairs[:, 0]
                pj = pairs[:, 1]
                dx = pos[pj, 0] - pos[pi, 0]
                dy = pos[pj, 1] - pos[pi, 1]
                dz = pos[pj, 2] - pos[pi, 2]
                if periodic:
                    dx -= Lx * np.round(dx / Lx)
                    dy -= Ly * np.round(dy / Ly)
                    dz -= Lz * np.round(dz / Lz)
                d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
                overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
                mask = overlap > 0
                if np.any(mask):
                    ov = overlap[mask]
                    max_overlap = float(np.max(ov))
                    inv_d = 1.0 / d[mask]
                    nx = dx[mask] * inv_d
                    ny = dy[mask] * inv_d
                    nz = dz[mask] * inv_d
                    f = k_rep * ov ** 1.5
                    np.add.at(fx, pi[mask], -f * nx)
                    np.add.at(fy, pi[mask], -f * ny)
                    np.add.at(fz, pi[mask], -f * nz)
                    np.add.at(fx, pj[mask],  f * nx)
                    np.add.at(fy, pj[mask],  f * ny)
                    np.add.at(fz, pj[mask],  f * nz)

            # Overdamped move with velocity cap
            f_mag = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-12
            scale = np.minimum(dt_settle, v_cap / f_mag)
            gs.x[:N] += fx * scale
            gs.y[:N] += fy * scale
            gs.z[:N] += fz * scale

            # Boundary handling
            if periodic:
                gs.x[:N] %= Lx
                gs.y[:N] %= Ly
                gs.z[:N] %= Lz
            else:
                rb = gs.r_bound[:N]
                gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
                gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)
                gs.z[:N] = np.clip(gs.z[:N], rb, Lz - rb)

            # Early exit if overlaps are small
            if max_overlap < 0.01 * mean_r_target:
                break

        # Progress
        if step % 100 == 0 or step == n_inflate - 1:
            # Count contacts
            pos = gs.positions()
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
            else:
                tree = cKDTree(pos)
            cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
            n_contacts = 0
            if len(cpairs) > 0:
                pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
                dx_c = pos[pj_c, 0] - pos[pi_c, 0]
                dy_c = pos[pj_c, 1] - pos[pi_c, 1]
                dz_c = pos[pj_c, 2] - pos[pi_c, 2]
                if periodic:
                    dx_c -= Lx * np.round(dx_c / Lx)
                    dy_c -= Ly * np.round(dy_c / Ly)
                    dz_c -= Lz * np.round(dz_c / Lz)
                d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
                gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
                n_contacts = int(np.sum(gaps < 1.0))
            Z = 2 * n_contacts / max(N, 1)
            print(f"    step {step}/{n_inflate}: α={alpha:.3f}, "
                  f"max_overlap={max_overlap:.1f} µm, Z={Z:.1f}")

    # ── Post-inflation relaxation: resolve remaining deep overlaps ──
    # After inflation reaches α=1.0, keep relaxing until max overlap is
    # below tolerance (5% of mean radius).  This prevents granules from
    # being trapped inside each other at high packing fractions.
    overlap_tol = 0.05 * mean_r_target
    max_post_relax = 1000  # safety cap on extra iterations
    max_rb = float(np.max(gs.r_bound[:N]))
    for extra in range(max_post_relax):
        pos = gs.positions()
        fx = np.zeros(N)
        fy = np.zeros(N)
        fz = np.zeros(N)

        if periodic:
            tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
        else:
            tree = cKDTree(pos)
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

        max_overlap = 0.0
        if len(pairs) > 0:
            pi = pairs[:, 0]
            pj = pairs[:, 1]
            dx = pos[pj, 0] - pos[pi, 0]
            dy = pos[pj, 1] - pos[pi, 1]
            dz = pos[pj, 2] - pos[pi, 2]
            if periodic:
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
                dz -= Lz * np.round(dz / Lz)
            d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
            overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
            mask = overlap > 0
            if np.any(mask):
                ov = overlap[mask]
                max_overlap = float(np.max(ov))
                inv_d = 1.0 / d[mask]
                nx = dx[mask] * inv_d
                ny = dy[mask] * inv_d
                nz = dz[mask] * inv_d
                f = k_rep * ov ** 1.5
                np.add.at(fx, pi[mask], -f * nx)
                np.add.at(fy, pi[mask], -f * ny)
                np.add.at(fz, pi[mask], -f * nz)
                np.add.at(fx, pj[mask],  f * nx)
                np.add.at(fy, pj[mask],  f * ny)
                np.add.at(fz, pj[mask],  f * nz)

        if max_overlap < overlap_tol:
            if extra > 0:
                print(f"    post-relax: {extra} extra steps, "
                      f"max_overlap={max_overlap:.1f} µm (tol={overlap_tol:.1f})")
            break

        f_mag = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-12
        scale = np.minimum(dt_settle, v_cap / f_mag)
        gs.x[:N] += fx * scale
        gs.y[:N] += fy * scale
        gs.z[:N] += fz * scale

        if periodic:
            gs.x[:N] %= Lx
            gs.y[:N] %= Ly
            gs.z[:N] %= Lz
        else:
            rb = gs.r_bound[:N]
            gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
            gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)
            gs.z[:N] = np.clip(gs.z[:N], rb, Lz - rb)
    else:
        print(f"    WARNING: post-relax hit {max_post_relax} steps, "
              f"max_overlap={max_overlap:.1f} µm (tol={overlap_tol:.1f})")

    # Final contact count
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
    else:
        tree = cKDTree(pos)
    cpairs = tree.query_pairs(2 * float(np.max(gs.r_bound[:N])) + 1.0, output_type='ndarray')
    n_contacts = 0
    if len(cpairs) > 0:
        pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
        dx_c = pos[pj_c, 0] - pos[pi_c, 0]
        dy_c = pos[pj_c, 1] - pos[pi_c, 1]
        dz_c = pos[pj_c, 2] - pos[pi_c, 2]
        if periodic:
            dx_c -= Lx * np.round(dx_c / Lx)
            dy_c -= Ly * np.round(dy_c / Ly)
            dz_c -= Lz * np.round(dz_c / Lz)
        d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
        gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
        n_contacts = int(np.sum(gaps < 1.0))
    Z = 2 * n_contacts / max(N, 1)
    print(f"  done ({n_contacts} contacts, Z={Z:.1f}/granule)")


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

    # Compute deflation factor for RSA placement (Lubachevsky-Stillinger).
    # Place granules at reduced bounding radii so RSA can achieve the target
    # count, then inflate-and-relax to the target packing fraction.
    phi_target = p.phi_f_target + p.phi_i_target
    if phi_target > 0.05 and p.packing_settle_steps > 0:
        alpha_deflate = (p.packing_inflate_phi_safe / max(phi_target, 0.01)) ** (1.0 / 2.0)
        alpha_deflate = float(np.clip(alpha_deflate, 0.3, 1.0))
    else:
        alpha_deflate = 1.0
    if alpha_deflate < 1.0:
        print(f"  RSA deflation: α={alpha_deflate:.3f} "
              f"(φ_safe={p.packing_inflate_phi_safe:.2f}, φ_target={phi_target:.3f})")

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
                if dx*dx + dy*dy < ((r_bound_val + rj_bound) * alpha_deflate + gap)**2:
                    ok = False; break
            if ok:
                xs.append(cx); ys.append(cy)
                rs.append(r_eq); types.append(gt)
                a_list.append(a_val); b_list.append(b_val)
                n_shape_list.append(n_s); theta_list.append(theta_val)
                placed = True; break
        if not placed:
            pass  # skip if can't place

    # Compute cells per granule
    n_cells = []
    A_cell_sphere = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
    for i in range(len(xs)):
        if types[i] == 0:
            if p.cell_surface_coverage > 0:
                nc = cells_from_surface_coverage(
                    rs[i], p.cell_diameter, p.cell_height_spread,
                    p.cell_surface_coverage, p.mode)
            else:
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

    # Store target radii and deflate for inflate-and-relax settling
    if alpha_deflate < 1.0 and p.packing_settle_steps > 0:
        gs._target_a = gs.a.copy()
        gs._target_b = gs.b.copy()
        gs._target_r = gs.r.copy()
        gs.a[:gs.N] *= alpha_deflate
        gs.b[:gs.N] *= alpha_deflate
        gs.r[:gs.N] *= alpha_deflate
        gs.r_bound[:gs.N] = np.maximum(gs.a[:gs.N], gs.b[:gs.N])

    # ── Inflate-and-relax settle: achieve jammed packing ──
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

    # Compute deflation factor for RSA placement (Lubachevsky-Stillinger).
    # Place granules at reduced bounding radii so RSA can achieve the target
    # count, then inflate-and-relax to the target packing fraction.
    phi_target = p.phi_f_target + p.phi_i_target
    if phi_target > 0.05 and p.packing_settle_steps > 0:
        alpha_deflate = (p.packing_inflate_phi_safe / max(phi_target, 0.01)) ** (1.0 / 3.0)
        alpha_deflate = float(np.clip(alpha_deflate, 0.3, 1.0))
    else:
        alpha_deflate = 1.0
    if alpha_deflate < 1.0:
        print(f"  RSA deflation: α={alpha_deflate:.3f} "
              f"(φ_safe={p.packing_inflate_phi_safe:.2f}, φ_target={phi_target:.3f})")

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
                if dx*dx + dy*dy + dz*dz < ((r_bound_val + rj_b) * alpha_deflate + gap)**2:
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
            if p.cell_surface_coverage > 0:
                nc = cells_from_surface_coverage(
                    rs[i], p.cell_diameter, p.cell_height_spread,
                    p.cell_surface_coverage, p.mode)
            else:
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

    # Store target radii and deflate for inflate-and-relax settling
    if alpha_deflate < 1.0 and p.packing_settle_steps > 0:
        gs._target_a = gs.a.copy()
        gs._target_b = gs.b.copy()
        gs._target_c = gs.c.copy()
        gs._target_r = gs.r.copy()
        gs.a[:gs.N] *= alpha_deflate
        gs.b[:gs.N] *= alpha_deflate
        gs.c[:gs.N] *= alpha_deflate
        gs.r[:gs.N] *= alpha_deflate
        gs.r_bound[:gs.N] = np.maximum(np.maximum(gs.a[:gs.N], gs.b[:gs.N]),
                                        gs.c[:gs.N])

    # ── Inflate-and-relax settle: achieve jammed packing ──
    if p.packing_settle_steps > 0 and gs.N > 1:
        _settle_packing_3d(gs, p)

    # Clean up temporary target attributes
    for attr in ('_target_a', '_target_b', '_target_c', '_target_r'):
        if hasattr(gs, attr):
            delattr(gs, attr)

    # Report packing fractions
    if p.shape_enabled:
        vols = np.array([superellipsoid_volume(gs.a[i], gs.b[i], gs.c[i],
                         gs.n1[i], gs.n2[i]) for i in range(gs.N)])
        act_f = float(np.sum(vols[gs.func_mask])) / domain_vol
        act_i = float(np.sum(vols[gs.inert_mask])) / domain_vol
    else:
        act_f = sum((4.0/3.0)*np.pi*gs.r[gs.func_mask]**3) / domain_vol
        act_i = sum((4.0/3.0)*np.pi*gs.r[gs.inert_mask]**3) / domain_vol
    phi_total = act_f + act_i
    print(f"  Placed: {np.sum(gs.func_mask)} func (φ_f={act_f:.3f}) + "
          f"{np.sum(gs.inert_mask)} inert (φ_i={act_i:.3f})")
    print(f"  Void fraction: {1-phi_total:.3f}")

    # Warn if packing fraction exceeds estimated RCP
    ar_mean = float(np.mean(np.maximum(gs.a[:gs.N], gs.b[:gs.N]) /
                            np.minimum(gs.a[:gs.N], gs.b[:gs.N])))
    phi_rcp = rcp_fraction_superellipsoid(ar_mean)
    if phi_total > phi_rcp:
        print(f"  WARNING: phi_solid={phi_total:.3f} exceeds estimated RCP={phi_rcp:.3f}.")
        print(f"    Granules will have unavoidable overlaps. Consider reducing phi_solid_target.")

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

def _cell_world_pos_2d(gs, ci):
    """Return world (x, y) for cell ci on its host granule (2D)."""
    gi = int(gs.cell_granule_id[ci])
    theta_cell = gs.cell_theta_local[ci]
    bx, by = superellipse_point(theta_cell, gs.a[gi], gs.b[gi], gs.n_shape[gi])
    ct = np.cos(gs.theta[gi])
    st = np.sin(gs.theta[gi])
    return gs.x[gi] + ct * bx - st * by, gs.y[gi] + st * bx + ct * by


def _cell_world_pos_3d(gs, ci):
    """Return world (x, y, z) for cell ci on its host granule (3D)."""
    gi = int(gs.cell_granule_id[ci])
    eta = gs.cell_eta_local[ci]
    omega = gs.cell_omega_local[ci]
    bp = superellipsoid_point(eta, omega, gs.a[gi], gs.b[gi], gs.c[gi],
                              gs.n_shape[gi], gs.n2[gi])
    bp_w = quat_rotate(gs.quat[gi], bp)
    return gs.x[gi] + bp_w[0], gs.y[gi] + bp_w[1], gs.z[gi] + bp_w[2]


def _compute_gap(gs, gi, tgt, p):
    """Surface-to-surface gap between granules gi and tgt (handles periodic BCs)."""
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    if p.boundary_mode == 'periodic':
        dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
    if hasattr(gs, 'is_3d') and gs.is_3d:
        dz = gs.z[tgt] - gs.z[gi]
        if p.boundary_mode == 'periodic':
            dz = dz - p.Lz * np.round(dz / p.Lz)
        d = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        d = np.sqrt(dx**2 + dy**2)
    return max(d - gs.r_bound[gi] - gs.r_bound[tgt], 0.0)


def _angular_distance_to_target_2d(gs, ci, tgt, p):
    """Angular distance from cell's surface position to contact azimuth (2D).

    Returns (angular_distance, target_theta_body) where target_theta_body is
    the body-frame angle on the host granule pointing toward granule tgt.
    """
    gi = int(gs.cell_granule_id[ci])
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    if p.boundary_mode == 'periodic':
        dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
    # Target direction in body frame of host granule
    target_theta_world = np.arctan2(dy, dx)
    target_theta_body = (target_theta_world - gs.theta[gi]) % (2.0 * np.pi)
    cell_theta = gs.cell_theta_local[ci] % (2.0 * np.pi)
    # Shortest arc
    diff = target_theta_body - cell_theta
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    return abs(diff), target_theta_body, diff


def _angular_distance_to_target_3d(gs, ci, tgt, p):
    """Angular distance from cell's surface position to contact azimuth (3D).

    Returns (angular_distance, target_eta, target_omega).
    Uses great-circle distance on the parametric unit sphere.
    """
    gi = int(gs.cell_granule_id[ci])
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    dz = gs.z[tgt] - gs.z[gi]
    if p.boundary_mode == 'periodic':
        dx = dx - p.Lx * np.round(dx / p.Lx)
        dy = dy - p.Ly * np.round(dy / p.Ly)
        dz = dz - p.Lz * np.round(dz / p.Lz)
    # Target direction in body frame
    d_world = np.array([dx, dy, dz])
    d_body = quat_rotate_inv(gs.quat[gi], d_world)
    # Convert to parametric angles
    r_xy = np.sqrt(d_body[0]**2 + d_body[1]**2)
    target_eta = np.arctan2(d_body[2], r_xy)
    target_omega = np.arctan2(d_body[1], d_body[0]) % (2.0 * np.pi)
    # Great-circle distance from cell position
    cell_eta = gs.cell_eta_local[ci]
    cell_omega = gs.cell_omega_local[ci]
    # Unit vectors on sphere
    p_curr = np.array([np.cos(cell_eta) * np.cos(cell_omega),
                       np.cos(cell_eta) * np.sin(cell_omega),
                       np.sin(cell_eta)])
    p_tgt = np.array([np.cos(target_eta) * np.cos(target_omega),
                      np.cos(target_eta) * np.sin(target_omega),
                      np.sin(target_eta)])
    dot = np.clip(np.dot(p_curr, p_tgt), -1.0, 1.0)
    ang_dist = np.arccos(dot)
    return ang_dist, target_eta, target_omega


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
                gs.cell_bridge_locked[ci] = False
                gs.cell_alignment[ci] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[ci] / max(0.1, p.bridge_formation_time))
                alignment = max(gs.cell_alignment[ci], p.bridge_alignment_min)
                F_cell = min(F_per_cell * maturity * alignment, p.F_max_per_cell)
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
                gs.cell_bridge_locked[cj] = False
                gs.cell_alignment[cj] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[cj] / max(0.1, p.bridge_formation_time))
                alignment = max(gs.cell_alignment[cj], p.bridge_alignment_min)
                F_cell = min(F_per_cell * maturity * alignment, p.F_max_per_cell)
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


def _classify_bridge_path(gs, p, gi, gj, pos_i, pos_j, gap):
    """Classify the path between functional granules for bridge probability.

    Returns a multiplicative path_factor:
      - Contact (gap ≤ 0): bridge_contact_factor (high)
      - Void (gap > 0, no obstruction): exp(-gap / bridge_decay_length)
      - Inert-blocked (gap > 0): bridge_inert_factor × exp(-gap / ...)
    """
    # Contact: cells crawl directly between touching surfaces
    if gap <= 0:
        return p.bridge_contact_factor

    # Exponential void decay (filopodia reach)
    void_decay = np.exp(-gap / max(p.bridge_decay_length, 0.1))

    # Ray-cast: check if any granule's bounding sphere intersects the
    # line segment between gi and gj centres.
    N = gs.N
    d_vec = pos_j - pos_i
    d_mag = np.linalg.norm(d_vec)
    if d_mag < 1e-12:
        return p.bridge_contact_factor * void_decay

    d_hat = d_vec / d_mag
    ri = gs.r_bound[gi]
    rj = gs.r_bound[gj]

    # Vectorised positions of all granules (minimum-image relative to pos_i)
    if gs.is_3d:
        all_pos = np.column_stack([gs.x[:N], gs.y[:N], gs.z[:N]])
    else:
        all_pos = np.column_stack([gs.x[:N], gs.y[:N]])
    dk = all_pos - pos_i  # vectors from gi to each gk
    if p.boundary_mode == 'periodic':
        Ls = np.array([p.Lx, p.Ly, p.Lz][:dk.shape[1]])
        dk -= Ls * np.round(dk / Ls)

    # Project onto line gi→gj
    t = dk @ d_hat                         # scalar projection for each gk

    # gk must project between the surfaces of gi and gj
    valid = (t > ri * 0.5) & (t < d_mag - rj * 0.5)
    valid[gi] = False
    valid[gj] = False

    if not np.any(valid):
        return void_decay  # void path — no obstruction

    # Perpendicular distance from each valid gk to the line
    proj = np.outer(t[valid], d_hat)       # (K, dim)
    perp = dk[valid] - proj                # (K, dim)
    d_perp = np.sqrt(np.sum(perp * perp, axis=1))

    # Check bounding-sphere intersection
    r_valid = gs.r_bound[:N][valid]
    intersects = d_perp < r_valid

    if not np.any(intersects):
        return void_decay  # void path

    # At least one granule blocks the line-of-sight
    gtype_valid = gs.gtype[:N][valid]
    if np.any(intersects & (gtype_valid == 1)):
        return p.bridge_inert_factor * void_decay  # inert blocks

    # Functional granule in path — cells would bridge to it instead,
    # so direct bridge is unlikely (treat same as void)
    return void_decay


def _attempt_new_bridges(gs, gi, gj, gap, p, rng, F_per_cell, nx, ny, nz, F,
                         n_existing_bridges=0, path_factor=1.0,
                         pos_i=None, pos_j=None):
    """Probabilistically initiate new bridges between granules gi and gj.

    Only cells that are SPREADING or PROLIFERATING with sufficient FA maturity
    can attempt bridging.  Each eligible cell has a Poisson-distributed
    probability of finding and committing to a bridge target per timestep.

    Cell spatial awareness (V2.3): each cell's position on the granule surface
    determines whether it can see the target. Cells on the far side of their
    host granule (hemisphere facing away from the target) cannot bridge.
    The bridge probability decays from the cell's centroid, not the granule
    centre, so cells closer to the target have higher probability.

    When existing bridges are present (n_existing_bridges > 0), non-bridging
    cells can migrate along the established bridge and form additional
    connections, boosted by bridge_secondary_rate_mult.
    """
    bridgeable_states = (int(CellState.SPREADING), int(CellState.PROLIFERATING))
    min_fa = p.min_fa_for_bridge
    is_3d = (nz != 0.0) or (gs.is_3d if hasattr(gs, 'is_3d') else False)

    # Base rate (before cell-specific decay)
    rate = p.bridge_attempt_rate
    if n_existing_bridges > 0:
        rate *= p.bridge_secondary_rate_mult
    p_bridge_base = 1.0 - np.exp(-rate * p.dt)

    # Path classification (contact/void/inert) from granule-level ray-cast
    # is used as a multiplicative modifier on top of cell-specific decay.
    # Extract the obstruction type from path_factor:
    # - path_factor already includes exp(-gap/λ) for void/inert paths
    # - We need to separate the obstruction modifier from the distance decay
    #   so we can replace distance decay with cell-specific distance.
    decay_len = max(p.bridge_decay_length, 0.1)
    if gap <= 0:
        # Contact: use path_factor directly (bridge_contact_factor), no distance decay
        obstruction_mod = path_factor
        use_cell_decay = False
    else:
        # Factor out the granule-level exponential decay to get pure obstruction modifier
        granule_decay = np.exp(-gap / decay_len)
        if granule_decay > 1e-15:
            obstruction_mod = path_factor / granule_decay
        else:
            obstruction_mod = path_factor
        use_cell_decay = True

    # Target granule centre (for cell-to-target distance)
    if pos_j is not None:
        tgt = pos_j
    elif is_3d:
        tgt = np.array([gs.x[gj], gs.y[gj], gs.z[gj]])
    else:
        tgt = np.array([gs.x[gj], gs.y[gj]])
    r_target = gs.r_bound[gj]

    # ── Collect angular positions of existing BRIDGING/MIGRATING cells for
    #    exclusion zone check (cells avoid bridging near existing bridges) ──
    _bridge_states = (int(CellState.BRIDGING), int(CellState.MIGRATING))
    existing_thetas_i = []  # angular positions of bridges on gi→gj
    existing_thetas_j = []  # angular positions of bridges on gj→gi
    for ck in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ck] in _bridge_states and gs.cell_bridge_target[ck] == gj:
            existing_thetas_i.append(gs.cell_theta_local[ck] if not is_3d else
                                     (gs.cell_eta_local[ck], gs.cell_omega_local[ck]))
    for ck in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[ck] in _bridge_states and gs.cell_bridge_target[ck] == gi:
            existing_thetas_j.append(gs.cell_theta_local[ck] if not is_3d else
                                     (gs.cell_eta_local[ck], gs.cell_omega_local[ck]))

    # ── Eligible cells on granule i ──
    for ci in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ci] not in bridgeable_states:
            continue
        if gs.fa_maturity[gi] < min_fa:
            continue

        # Cell world position
        if is_3d:
            cx, cy, cz = _cell_world_pos_3d(gs, ci)
            cell_pos = np.array([cx, cy, cz])
        else:
            cx, cy = _cell_world_pos_2d(gs, ci)
            cell_pos = np.array([cx, cy])

        # Hemisphere check: cell offset from granule centre dotted with
        # direction to target. Skip if cell faces away.
        d_to_tgt = np.array([nx, ny, nz][:len(cell_pos)])
        cell_offset = cell_pos - (pos_i if pos_i is not None else
                                  np.array([gs.x[gi], gs.y[gi]])
                                  if not is_3d else
                                  np.array([gs.x[gi], gs.y[gi], gs.z[gi]]))
        if np.dot(cell_offset, d_to_tgt) < 0:
            continue  # cell is on far side of granule

        # Bridge exclusion zone: skip if too close to existing bridge/migrating cell
        if p.bridge_exclusion_angle > 0 and existing_thetas_i:
            too_close = False
            if is_3d:
                eta_c, om_c = gs.cell_eta_local[ci], gs.cell_omega_local[ci]
                p_c = np.array([np.cos(eta_c) * np.cos(om_c),
                                np.cos(eta_c) * np.sin(om_c), np.sin(eta_c)])
                for (eta_k, om_k) in existing_thetas_i:
                    p_k = np.array([np.cos(eta_k) * np.cos(om_k),
                                    np.cos(eta_k) * np.sin(om_k), np.sin(eta_k)])
                    ang_sep = np.arccos(np.clip(np.dot(p_c, p_k), -1.0, 1.0))
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            else:
                theta_c = gs.cell_theta_local[ci]
                for theta_k in existing_thetas_i:
                    diff = abs(theta_c - theta_k)
                    ang_sep = min(diff, 2 * np.pi - diff)
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            if too_close:
                continue  # cell would rather crawl around than stack on existing bridge

        # Cell-to-target gap (cell surface → target surface)
        cell_to_tgt_dist = np.linalg.norm(cell_pos - tgt)
        cell_gap = max(cell_to_tgt_dist - r_target, 0.0)

        # Cell-specific probability
        if use_cell_decay:
            cell_decay = np.exp(-cell_gap / decay_len)
            p_bridge = p_bridge_base * obstruction_mod * cell_decay
        else:
            p_bridge = p_bridge_base * obstruction_mod

        if rng.random() < p_bridge:
            gs.cell_bridge_target[ci] = gj
            gs.cell_bridge_age[ci] = 0.0
            # Check if cell is already at the contact azimuth (fast-path)
            if is_3d:
                ang_d, _, _ = _angular_distance_to_target_3d(gs, ci, gj, p)
            else:
                ang_d, _, _ = _angular_distance_to_target_2d(gs, ci, gj, p)
            if ang_d < p.bridge_commit_angle:
                # Already at contact point → skip migration, bridge directly
                gs.cell_state[ci] = int(CellState.BRIDGING)
                gs.cell_alignment[ci] = p.bridge_alignment_min
                maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
                F_cell = min(F_per_cell * maturity * p.bridge_alignment_min,
                             p.F_max_per_cell)
                gs.cell_fx[ci] += F_cell * nx
                gs.cell_fy[ci] += F_cell * ny
                gs.cell_fz[ci] += F_cell * nz
                if F.ndim == 2 and F.shape[1] >= 2:
                    F[gi, 0] += F_cell * nx
                    F[gi, 1] += F_cell * ny
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gi, 2] += F_cell * nz
            else:
                # Cell needs to crawl to contact point first
                gs.cell_state[ci] = int(CellState.MIGRATING)
                gs.cell_alignment[ci] = 0.0

    # ── Eligible cells on granule j (target is gi) ──
    if pos_i is not None:
        tgt_j = pos_i
    elif is_3d:
        tgt_j = np.array([gs.x[gi], gs.y[gi], gs.z[gi]])
    else:
        tgt_j = np.array([gs.x[gi], gs.y[gi]])
    r_target_j = gs.r_bound[gi]

    for cj in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[cj] not in bridgeable_states:
            continue
        if gs.fa_maturity[gj] < min_fa:
            continue

        # Cell world position
        if is_3d:
            cx, cy, cz = _cell_world_pos_3d(gs, cj)
            cell_pos = np.array([cx, cy, cz])
        else:
            cx, cy = _cell_world_pos_2d(gs, cj)
            cell_pos = np.array([cx, cy])

        # Hemisphere check: cell should face toward gi (opposite direction)
        d_to_gi = -np.array([nx, ny, nz][:len(cell_pos)])
        cell_offset = cell_pos - (pos_j if pos_j is not None else
                                  np.array([gs.x[gj], gs.y[gj]])
                                  if not is_3d else
                                  np.array([gs.x[gj], gs.y[gj], gs.z[gj]]))
        if np.dot(cell_offset, d_to_gi) < 0:
            continue  # cell is on far side of granule

        # Bridge exclusion zone: skip if too close to existing bridge/migrating cell
        if p.bridge_exclusion_angle > 0 and existing_thetas_j:
            too_close = False
            if is_3d:
                eta_c, om_c = gs.cell_eta_local[cj], gs.cell_omega_local[cj]
                p_c = np.array([np.cos(eta_c) * np.cos(om_c),
                                np.cos(eta_c) * np.sin(om_c), np.sin(eta_c)])
                for (eta_k, om_k) in existing_thetas_j:
                    p_k = np.array([np.cos(eta_k) * np.cos(om_k),
                                    np.cos(eta_k) * np.sin(om_k), np.sin(eta_k)])
                    ang_sep = np.arccos(np.clip(np.dot(p_c, p_k), -1.0, 1.0))
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            else:
                theta_c = gs.cell_theta_local[cj]
                for theta_k in existing_thetas_j:
                    diff = abs(theta_c - theta_k)
                    ang_sep = min(diff, 2 * np.pi - diff)
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            if too_close:
                continue  # cell would rather crawl around than stack on existing bridge

        # Cell-to-target gap
        cell_to_tgt_dist = np.linalg.norm(cell_pos - tgt_j)
        cell_gap = max(cell_to_tgt_dist - r_target_j, 0.0)

        if use_cell_decay:
            cell_decay = np.exp(-cell_gap / decay_len)
            p_bridge = p_bridge_base * obstruction_mod * cell_decay
        else:
            p_bridge = p_bridge_base * obstruction_mod

        if rng.random() < p_bridge:
            gs.cell_bridge_target[cj] = gi
            gs.cell_bridge_age[cj] = 0.0
            if is_3d:
                ang_d, _, _ = _angular_distance_to_target_3d(gs, cj, gi, p)
            else:
                ang_d, _, _ = _angular_distance_to_target_2d(gs, cj, gi, p)
            if ang_d < p.bridge_commit_angle:
                gs.cell_state[cj] = int(CellState.BRIDGING)
                gs.cell_alignment[cj] = p.bridge_alignment_min
                maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
                F_cell = min(F_per_cell * maturity * p.bridge_alignment_min,
                             p.F_max_per_cell)
                gs.cell_fx[cj] -= F_cell * nx
                gs.cell_fy[cj] -= F_cell * ny
                gs.cell_fz[cj] -= F_cell * nz
                if F.ndim == 2 and F.shape[1] >= 2:
                    F[gj, 0] -= F_cell * nx
                    F[gj, 1] -= F_cell * ny
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gj, 2] -= F_cell * nz
            else:
                gs.cell_state[cj] = int(CellState.MIGRATING)
                gs.cell_alignment[cj] = 0.0


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

        # ── LS-DEM deformable contact detection (V2.3) ──
        if p.deformable_enabled:
            from lsdem import find_contacts_lsdem_2d
            xj_v = pos[i,0] + dx
            yj_v = pos[i,1] + dy
            lsdem_result = find_contacts_lsdem_2d(
                gs, i, j, pos[i,0], pos[i,1], xj_v, yj_v, p)
            if lsdem_result is not None:
                F[i] += lsdem_result['F_i']
                F[j] += lsdem_result['F_j']
                torques[i] += lsdem_result['tau_i']
                torques[j] += lsdem_result['tau_j']
                if gs.F_eps_accum is not None:
                    gs.F_eps_accum[i] += lsdem_result['F_eps_i']
                    gs.F_eps_accum[j] += lsdem_result['F_eps_j']
                overlap = lsdem_result['overlap_max']
                R_eff_approx = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])

                # JKR adhesion correction — translational only (V2.3)
                # LS-DEM nodes compute pure repulsion; adhesion added here
                # so it does NOT drive deformation DOFs
                _, W_adh_pair = friction_lut.get(
                    (int(gs.gtype[i]), int(gs.gtype[j])), (0.0, 0.0))
                if W_adh_pair > 0 and overlap > 0:
                    F_jkr, a_est = jkr_force_from_overlap(
                        overlap, R_eff_approx, E_star_gg, W_adh_pair)
                    F_hertz = hertz_contact_force(E_star_gg, R_eff_approx, overlap)
                    # Adhesion correction is JKR − Hertz (negative = attractive)
                    F_adh_corr = F_jkr - F_hertz
                    F[i] += F_adh_corr * np.array([nx, ny])
                    F[j] -= F_adh_corr * np.array([nx, ny])
                else:
                    a_est = np.sqrt(max(R_eff_approx * overlap, 0.0))

                # Estimate contact radius from overlap for clip rendering
                if a_est > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_est**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_est**2, 0.01))
                    gs.contact_clips[i].append((nx, ny, clip_d_i))
                    gs.contact_clips[j].append((-nx, -ny, clip_d_j))
                contacts.append({
                    'i': i, 'j': j,
                    'cx': (pos[i,0] + xj_v) / 2, 'cy': (pos[i,1] + yj_v) / 2, 'cz': 0.0,
                    'nx': nx, 'ny': ny, 'nz': 0.0,
                    'overlap': overlap, 'R_eff': R_eff_approx,
                    'F_normal': np.sqrt(lsdem_result['F_i'][0]**2 + lsdem_result['F_i'][1]**2),
                    'A_contact': np.pi * a_est**2,
                    'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                })
                in_contact = True
            else:
                in_contact = False
                overlap = 0.0

            # Cell bridging still uses gap-based logic
            if gs.gtype[i] == 0 and gs.gtype[j] == 0:
                if in_contact:
                    gap = -overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0:
                        gap = 0.0
                avg_maturity = 0.5 * (gs.fa_maturity[i] + gs.fa_maturity[j])
                F_per_cell = motor_clutch_force(p.E_modulus, p, avg_maturity)
                n_ci, n_cj = _service_committed_bridges(
                    gs, i, j, gap, p, F_per_cell, nx, ny, 0.0, F)
                n_existing = n_ci + n_cj
                if gap < p.cell_sense_distance:
                    pj_v_2d = pos[i] + np.array([dx, dy])
                    pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v_2d, gap)
                    _attempt_new_bridges(
                        gs, i, j, gap, p, rng, F_per_cell, nx, ny, 0.0, F,
                        n_existing_bridges=n_existing, path_factor=pf,
                        pos_i=pos[i], pos_j=pj_v_2d)
            continue  # skip rigid contact path

        # ── Contact detection (rigid, V2.2) ──
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

        # ── Contact forces: JKR adhesive contact + friction (V2.3) ──
        if in_contact and overlap > 0:
            # Pair-type friction/adhesion parameters
            tau_0, W_adh = friction_lut[(gs.gtype[i], gs.gtype[j])]

            # JKR force: replaces Hertz + DMT (reduces to Hertz when W=0)
            F_normal, a_contact = jkr_force_from_overlap(
                overlap, R_eff, E_star_gg, W_adh)

            Fnx = F_normal * nx
            Fny = F_normal * ny
            F[i,0] -= Fnx; F[i,1] -= Fny
            F[j,0] += Fnx; F[j,1] += Fny

            # Torque from normal force (off-centre contact)
            if not gs.is_circle:
                rci_x = contact_x - pos[i,0]
                rci_y = contact_y - pos[i,1]
                rcj_x = contact_x - pos[j,0]
                rcj_y = contact_y - pos[j,1]
                torques[i] += rci_x * (-Fny) - rci_y * (-Fnx)
                torques[j] += rcj_x * Fny - rcj_y * Fnx

            # Contact clip planes for rendering (half-plane at flat face)
            if a_contact > 1e-6:
                clip_d_i = np.sqrt(max(gs.r[i]**2 - a_contact**2, 0.01))
                clip_d_j = np.sqrt(max(gs.r[j]**2 - a_contact**2, 0.01))
                gs.contact_clips[i].append((nx, ny, clip_d_i))
                gs.contact_clips[j].append((-nx, -ny, clip_d_j))

            # Tangential friction (JKR contact area: π a²)
            A_contact = np.pi * a_contact**2 if a_contact > 0 else 0.0
            dvx = gs.vx[j] - gs.vx[i]
            dvy = gs.vy[j] - gs.vy[i]
            v_dot_n = dvx * nx + dvy * ny
            vtx = dvx - v_dot_n * nx
            vty = dvy - v_dot_n * ny
            vt_mag = np.sqrt(vtx*vtx + vty*vty)

            if vt_mag > 1e-12:
                F_fric = tau_0 * A_contact * 1e-3 * np.tanh(
                    vt_mag / p.friction_v_ref)
                tx, ty = vtx / vt_mag, vty / vt_mag
                Ftx, Fty = F_fric * tx, F_fric * ty
                F[i,0] += Ftx; F[i,1] += Fty
                F[j,0] -= Ftx; F[j,1] -= Fty

                if not gs.is_circle:
                    torques[i] += rci_x * Fty - rci_y * Ftx
                    torques[j] += rcj_x * (-Fty) - rcj_y * (-Ftx)

            contacts.append({
                'i': i, 'j': j,
                'cx': contact_x, 'cy': contact_y, 'cz': 0.0,
                'nx': nx, 'ny': ny, 'nz': 0.0,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'a_contact': a_contact,
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
            n_ci, n_cj = _service_committed_bridges(
                gs, i, j, gap, p, F_per_cell, nx, ny, 0.0, F)
            n_existing = n_ci + n_cj

            # 2) Path-dependent bridge formation (V2.1)
            if gap < p.cell_sense_distance:
                pj_v_2d = pos[i] + np.array([dx, dy])
                pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v_2d, gap)
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_per_cell, nx, ny, 0.0, F,
                    n_existing_bridges=n_existing, path_factor=pf,
                    pos_i=pos[i], pos_j=pj_v_2d)

    # ── Wall repulsion: JKR adhesive contact (skip for periodic boundaries) ──
    # Wall is rigid (R_eff = R_granule for flat surface), treated as inert surface.
    if not periodic:
        W_wall_lut = {0: p.W_adh_if, 1: p.W_adh_ii}  # func→mixed, inert→inert
        wall_normals_2d = [
            (0, +1, 0),  # left:   +x normal, axis=0
            (0, -1, 0),  # right:  -x normal, axis=0
            (1, +1, 1),  # bottom: +y normal, axis=1
            (1, -1, 1),  # top:    -y normal, axis=1
        ]
        for i in range(N):
            W_wall = W_wall_lut[int(gs.gtype[i])]
            if gs.is_circle:
                r = gs.r[i]
                for pen, axis, sign in [
                    (r - gs.x[i],            0, +1),   # left
                    (gs.x[i] - (p.Lx - r),   0, -1),   # right
                    (r - gs.y[i],            1, +1),   # bottom
                    (gs.y[i] - (p.Ly - r),   1, -1),   # top
                ]:
                    if pen > 0:
                        Fw, a_w = jkr_force_from_overlap(pen, r, E_star_gw, W_wall)
                        F[i, axis] += sign * Fw
                        # Wall clip: distance from centre to wall face
                        if axis == 0:
                            wall_d = abs(gs.x[i]) if sign > 0 else abs(p.Lx - gs.x[i])
                        else:
                            wall_d = abs(gs.y[i]) if sign > 0 else abs(p.Ly - gs.y[i])
                        # Clip normal points TOWARD wall (opposite of force sign)
                        nx_w = -float(sign) if axis == 0 else 0.0
                        ny_w = 0.0 if axis == 0 else -float(sign)
                        gs.contact_clips[i].append((nx_w, ny_w, wall_d))
            else:
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
                        Fw, a_w = jkr_force_from_overlap(
                            pen, R_local, E_star_gw, W_wall)
                        F[i, wall_axis] += wall_sign * Fw
                        # Wall clip
                        if wall_axis == 0:
                            wall_d = abs(gs.x[i] - wall_pos)
                        else:
                            wall_d = abs(gs.y[i] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        nx_w = -float(wall_sign) if wall_axis == 0 else 0.0
                        ny_w = 0.0 if wall_axis == 0 else -float(wall_sign)
                        gs.contact_clips[i].append((nx_w, ny_w, wall_d))

    # MC-DEM multi-contact stiffening correction (Giannis et al. 2021)
    if p.mc_dem_enabled and len(contacts) > 0:
        mc_dem_correction(contacts, gs, p, E_star_gg, F)

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

        # ── LS-DEM deformable contact detection 3D (V2.3) ──
        pj_v = pos[i] + dp
        if p.deformable_enabled:
            from lsdem import find_contacts_lsdem_3d
            lsdem_result = find_contacts_lsdem_3d(
                gs, i, j, pos[i,0], pos[i,1], pos[i,2],
                pj_v[0], pj_v[1], pj_v[2], p)
            if lsdem_result is not None:
                F[i] += lsdem_result['F_i']
                F[j] += lsdem_result['F_j']
                torques[i] += lsdem_result['tau_i']
                torques[j] += lsdem_result['tau_j']
                if gs.F_eps_accum is not None:
                    gs.F_eps_accum[i] += lsdem_result['F_eps_i']
                    gs.F_eps_accum[j] += lsdem_result['F_eps_j']
                overlap = lsdem_result['overlap_max']
                R_eff_approx = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])

                # JKR adhesion correction — translational only (V2.3)
                _, W_adh_pair = friction_lut.get(
                    (int(gs.gtype[i]), int(gs.gtype[j])), (0.0, 0.0))
                if W_adh_pair > 0 and overlap > 0:
                    F_jkr, a_est = jkr_force_from_overlap(
                        overlap, R_eff_approx, E_star_gg, W_adh_pair)
                    F_hertz = hertz_contact_force(E_star_gg, R_eff_approx, overlap)
                    F_adh_corr = F_jkr - F_hertz
                    F[i] += F_adh_corr * nv
                    F[j] -= F_adh_corr * nv
                else:
                    a_est = np.sqrt(max(R_eff_approx * overlap, 0.0))

                # Estimate contact radius from overlap for clip rendering
                if a_est > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_est**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_est**2, 0.01))
                    gs.contact_clips[i].append((nv[0], nv[1], nv[2], clip_d_i))
                    gs.contact_clips[j].append((-nv[0], -nv[1], -nv[2], clip_d_j))
                contacts.append({
                    'i': i, 'j': j,
                    'cx': (pos[i,0]+pj_v[0])/2, 'cy': (pos[i,1]+pj_v[1])/2,
                    'cz': (pos[i,2]+pj_v[2])/2,
                    'nx': nv[0], 'ny': nv[1], 'nz': nv[2],
                    'overlap': overlap, 'R_eff': R_eff_approx,
                    'F_normal': np.linalg.norm(lsdem_result['F_i']),
                    'A_contact': np.pi * a_est**2,
                    'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                })
                in_contact = True
                contact_overlap = overlap
            else:
                in_contact = False
                contact_overlap = 0.0

            # Cell bridging
            if gs.gtype[i] == 0 and gs.gtype[j] == 0:
                if in_contact:
                    gap = -contact_overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0: gap = 0.0
                avg_maturity = 0.5 * (gs.fa_maturity[i] + gs.fa_maturity[j])
                F_per_cell = motor_clutch_force(p.E_modulus, p, avg_maturity)
                n_ci, n_cj = _service_committed_bridges(
                    gs, i, j, gap, p, F_per_cell, nv[0], nv[1], nv[2], F)
                n_existing = n_ci + n_cj
                if gap < p.cell_sense_distance:
                    pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v, gap)
                    _attempt_new_bridges(
                        gs, i, j, gap, p, rng, F_per_cell,
                        nv[0], nv[1], nv[2], F,
                        n_existing_bridges=n_existing, path_factor=pf,
                        pos_i=pos[i], pos_j=pj_v)
            continue  # skip rigid contact path

        # Contact detection (rigid, V2.2) — use virtual position of j for periodic BCs
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
                # JKR adhesive contact (V2.3, replaces Hertz + DMT)
                tau_0, W_adh = friction_lut[(gs.gtype[i], gs.gtype[j])]
                F_normal, a_contact = jkr_force_from_overlap(
                    overlap, R_eff, E_star_gg, W_adh)

                Fn = F_normal * n_vec
                F[i] -= Fn
                F[j] += Fn

                # Torque from off-centre contact
                if not gs.is_circle:
                    rc_i = np.array([cx - pos[i,0], cy - pos[i,1], cz_pt - pos[i,2]])
                    rc_j = np.array([cx - pos[j,0], cy - pos[j,1], cz_pt - pos[j,2]])
                    torques[i] += np.cross(rc_i, -Fn)
                    torques[j] += np.cross(rc_j, Fn)

                # Contact clip planes for 3D rendering
                if a_contact > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_contact**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_contact**2, 0.01))
                    gs.contact_clips[i].append((nx, ny, nz, clip_d_i))
                    gs.contact_clips[j].append((-nx, -ny, -nz, clip_d_j))

                # Tangential friction (JKR contact area: π a²)
                A_contact = np.pi * a_contact**2 if a_contact > 0 else 0.0
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

            contacts.append({
                'i': i, 'j': j,
                'cx': cx, 'cy': cy, 'cz': cz_pt,
                'nx': nx, 'ny': ny, 'nz': nz,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'a_contact': a_contact,
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
            n_ci, n_cj = _service_committed_bridges(
                gs, i, j, gap, p, F_per_cell,
                nv[0], nv[1], nv[2], F)
            n_existing = n_ci + n_cj

            # 2) Path-dependent bridge formation (V2.1)
            if gap < p.cell_sense_distance:
                pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v, gap)
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_per_cell,
                    nv[0], nv[1], nv[2], F,
                    n_existing_bridges=n_existing, path_factor=pf,
                    pos_i=pos[i], pos_j=pj_v)

    # Wall repulsion: JKR adhesive contact (6 faces) — skip for periodic
    if not periodic:
        W_wall_lut_3d = {0: p.W_adh_if, 1: p.W_adh_ii}
        for i in range(N):
            W_wall = W_wall_lut_3d[int(gs.gtype[i])]
            walls = [
                (0.0, 0, +1), (p.Lx, 0, -1),
                (0.0, 1, +1), (p.Ly, 1, -1),
                (0.0, 2, +1), (p.Lz, 2, -1),
            ]
            coords = [gs.x[i], gs.y[i], gs.z[i]]
            if gs.is_circle:
                r = gs.r[i]
                for wall_pos, axis, sign in walls:
                    pen = (r - (coords[axis] - wall_pos)) if sign > 0 \
                        else ((coords[axis] + r) - wall_pos)
                    if pen > 0:
                        Fw, a_w = jkr_force_from_overlap(
                            pen, r, E_star_gw, W_wall)
                        F[i, axis] += sign * Fw
                        wall_d = abs(coords[axis] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        n3 = [0.0, 0.0, 0.0]; n3[axis] = -float(sign)
                        gs.contact_clips[i].append(
                            (n3[0], n3[1], n3[2], wall_d))
            else:
                for wall_pos, axis, sign in walls:
                    wresult = find_contact_wall_3d(
                        gs.x[i], gs.y[i], gs.z[i],
                        gs.a[i], gs.b[i], gs.c[i],
                        gs.n1[i], gs.n2[i], gs.quat[i], gs.r_bound[i],
                        wall_pos, axis, sign)
                    if wresult is not None:
                        pen, R_local = wresult
                        Fw, a_w = jkr_force_from_overlap(
                            pen, R_local, E_star_gw, W_wall)
                        F[i, axis] += sign * Fw
                        wall_d = abs(coords[axis] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        n3 = [0.0, 0.0, 0.0]; n3[axis] = -float(sign)
                        gs.contact_clips[i].append(
                            (n3[0], n3[1], n3[2], wall_d))

    # MC-DEM multi-contact stiffening correction (Giannis et al. 2021)
    if p.mc_dem_enabled and len(contacts) > 0:
        mc_dem_correction(contacts, gs, p, E_star_gg, F)

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
# Post-step overlap resolution (V2.1 — prevents granule pass-through)
# ══════════════════════════════════════════════════════════════════════

def _resolve_overlaps(gs, p):
    """Push apart deeply interpenetrating granules after position integration.

    Uses bounding-sphere overlap to detect pairs where overlap exceeds
    max_overlap_frac * min(r_bound_i, r_bound_j), then projects each
    granule apart by half the excess.  Handles both 2D/3D and periodic/wall BCs.
    """
    N = gs.N
    if N < 2 or p.max_overlap_frac <= 0:
        return

    r_bound = gs.r_bound[:N]
    max_r = float(np.max(r_bound))
    cutoff = 2.0 * max_r
    periodic = p.boundary_mode == 'periodic'

    if gs.is_3d:
        pos = np.column_stack([gs.x[:N], gs.y[:N], gs.z[:N]])
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(pos)
    else:
        pos = np.column_stack([gs.x[:N], gs.y[:N]])
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
        else:
            tree = cKDTree(pos)

    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if len(pairs) == 0:
        return

    ii = pairs[:, 0]
    jj = pairs[:, 1]

    dx = pos[jj, 0] - pos[ii, 0]
    dy = pos[jj, 1] - pos[ii, 1]
    if periodic:
        dx -= p.Lx * np.round(dx / p.Lx)
        dy -= p.Ly * np.round(dy / p.Ly)

    if gs.is_3d:
        dz = pos[jj, 2] - pos[ii, 2]
        if periodic:
            dz -= p.Lz * np.round(dz / p.Lz)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        dist = np.sqrt(dx * dx + dy * dy)

    dist = np.maximum(dist, 1e-12)
    overlap = r_bound[ii] + r_bound[jj] - dist

    # Only correct overlaps exceeding the allowed threshold
    r_min = np.minimum(r_bound[ii], r_bound[jj])
    threshold = p.max_overlap_frac * r_min
    deep = overlap > threshold
    if not np.any(deep):
        return

    # Push each granule apart by half the excess overlap
    excess = overlap[deep] - threshold[deep]
    inv_d = 1.0 / dist[deep]
    nx = dx[deep] * inv_d
    ny = dy[deep] * inv_d
    correction = 0.5 * excess

    i_deep = ii[deep]
    j_deep = jj[deep]

    corr_x = np.zeros(N)
    corr_y = np.zeros(N)
    np.subtract.at(corr_x, i_deep, correction * nx)
    np.subtract.at(corr_y, i_deep, correction * ny)
    np.add.at(corr_x, j_deep, correction * nx)
    np.add.at(corr_y, j_deep, correction * ny)
    gs.x[:N] += corr_x
    gs.y[:N] += corr_y

    if gs.is_3d:
        nz = dz[deep] * inv_d
        corr_z = np.zeros(N)
        np.subtract.at(corr_z, i_deep, correction * nz)
        np.add.at(corr_z, j_deep, correction * nz)
        gs.z[:N] += corr_z


# ══════════════════════════════════════════════════════════════════════
# Time integration (overdamped: γ dx/dt = F  →  dx = F/γ · dt)
# ══════════════════════════════════════════════════════════════════════

def step(gs: GranuleSystem, p: Params, rng, t: float):
    """One overdamped Euler step with cell state evolution and rotation."""
    update_cell_state(gs, p, t, rng)

    # Reset contact clip planes (populated by compute_forces for rendering)
    for k in range(gs.N):
        gs.contact_clips[k] = []

    # Reset deformation force accumulator before force computation
    if p.deformable_enabled and gs.F_eps_accum is not None:
        gs.F_eps_accum[:] = 0.0

    if gs.is_3d:
        F, torques, contacts = compute_forces_3d(gs, p, rng)
    else:
        F, torques, contacts = compute_forces(gs, p, rng)

    # Integrate deformation DOFs (implicit Euler, after force computation)
    if p.deformable_enabled and gs.epsilon is not None:
        from lsdem import integrate_deformation_implicit
        integrate_deformation_implicit(gs, p)

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

        # Post-step overlap resolution (V2.1 — prevents granule pass-through)
        _resolve_overlaps(gs, p)
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

        # Post-step overlap resolution (V2.1 — prevents granule pass-through)
        _resolve_overlaps(gs, p)
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

def _apply_clips_2d(sdf, X, Y, cx, cy, clips):
    """Apply JKR half-plane clips to a 2D SDF array (in-place max).

    Each clip is a tuple (nx, ny, clip_d) defining a half-plane constraint:
        sdf = max(sdf, dot(x - center, n) - clip_d)
    This creates flat faces at contact regions, matching JKR contact geometry.
    """
    if not clips:
        return sdf
    for clip in clips:
        nx_c, ny_c, clip_d = clip[0], clip[1], clip[2]
        plane = (X - cx) * nx_c + (Y - cy) * ny_c - clip_d
        np.maximum(sdf, plane, out=sdf)
    return sdf


def _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips):
    """Apply JKR half-plane clips to a 3D SDF array (in-place max).

    Each clip is a tuple (nx, ny, nz, clip_d) defining a half-space constraint.
    """
    if not clips:
        return sdf
    for clip in clips:
        nx_c, ny_c, nz_c, clip_d = clip[0], clip[1], clip[2], clip[3]
        plane = (X - cx) * nx_c + (Y - cy) * ny_c + (Z - cz) * nz_c - clip_d
        np.maximum(sdf, plane, out=sdf)
    return sdf


def _stamp_circle_2d(X, Y, cx, cy, r_eff_i, w, phi_f, phi_i, gtype, clips=None):
    """Stamp a single circle profile onto the 2D field (additive blending).

    When clips is provided, JKR half-plane clipping creates flat contact faces.
    """
    sdf = np.sqrt((X - cx)**2 + (Y - cy)**2) - r_eff_i
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)
    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    if gtype == 0:
        phi_f += profile
    else:
        phi_i += profile


def _stamp_superellipse_2d(X, Y, cx, cy, a, b, n_s, theta, r, w, phi_f, phi_i, gtype,
                            clips=None):
    """Stamp a single superellipse profile onto the 2D field (additive blending).

    When clips is provided, JKR half-plane clipping creates flat contact faces.
    """
    bx, by = _world_to_body(X, Y, cx, cy, theta)
    se_val = (np.abs(bx) / a)**n_s + (np.abs(by) / b)**n_s
    n_inv = 1.0 / n_s
    sdf = (se_val**n_inv - 1.0) * r
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)
    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    if gtype == 0:
        phi_f += profile
    else:
        phi_i += profile


def _stamp_deformed_2d(X, Y, cx, cy, theta, gs, i, p, w, phi_f, phi_i):
    """Stamp a deformed particle (semi-Lagrangian pull-back) onto 2D field.

    V2.3: Uses the mode-shape displacement to pull grid points back to the
    undeformed reference configuration, then evaluates the analytical implicit
    function.  JKR contact clips applied after pull-back SDF computation.
    """
    bx, by = _world_to_body(X, Y, cx, cy, theta)
    eps = gs.epsilon[i]          # (n_modes,)
    nu = p.poisson_ratio
    n_modes = len(eps)

    # Vectorised mode-shape displacement (mirrors lsdem._mode_displacement_2d)
    ux = eps[0] * (-nu * bx) + (eps[1] * bx if n_modes > 1 else 0.0)
    uy = eps[0] * by           + (eps[1] * (-by) if n_modes > 1 else 0.0)

    bx_ref = bx - ux
    by_ref = by - uy

    # Evaluate undeformed SDF at the pulled-back coordinate
    a_i, b_i, n_s = gs.a[i], gs.b[i], gs.n_shape[i]
    se_val = (np.abs(bx_ref) / a_i)**n_s + (np.abs(by_ref) / b_i)**n_s
    n_inv = 1.0 / n_s
    sdf = (se_val**n_inv - 1.0) * gs.r[i]

    # Apply JKR contact clips (flat faces at contact regions)
    clips = gs.contact_clips[i] if hasattr(gs, 'contact_clips') and gs.contact_clips else None
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    if gs.gtype[i] == 0:
        phi_f += profile
    else:
        phi_i += profile


def _stamp_deformed_3d(phi_f, phi_i, gs, i, cx, cy, cz, xg, yg, zg,
                       dx_g, dy_g, dz_g, Ng, w, p):
    """Stamp a deformed 3D particle via semi-Lagrangian SDF pull-back.

    V2.3: Transforms grid points to body frame, applies inverse mode-shape
    displacement, then evaluates the analytical superellipsoid implicit.
    """
    rb = gs.r_bound[i] + 3 * w
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
    dx_l, dy_l, dz_l = X - cx, Y - cy, Z - cz

    # Rotate to body frame
    R_mat = quat_to_rotation_matrix(gs.quat[i])
    bx = R_mat[0, 0] * dx_l + R_mat[1, 0] * dy_l + R_mat[2, 0] * dz_l
    by = R_mat[0, 1] * dx_l + R_mat[1, 1] * dy_l + R_mat[2, 1] * dz_l
    bz = R_mat[0, 2] * dx_l + R_mat[1, 2] * dy_l + R_mat[2, 2] * dz_l

    # Mode-shape displacement (mirrors lsdem._mode_displacement_3d)
    eps = gs.epsilon[i]
    nu = p.poisson_ratio
    n_modes = len(eps)
    ux = eps[0] * (-nu * bx)
    uy = eps[0] * (-nu * by)
    uz = eps[0] * bz
    if n_modes > 1:
        ux += eps[1] * bx
        uy += eps[1] * (-by)
        # uz += 0 for mode 1
    if n_modes > 2:
        ux += eps[2] * bx
        uy += eps[2] * by
        uz += eps[2] * bz

    bx_ref = bx - ux
    by_ref = by - uy
    bz_ref = bz - uz

    # Evaluate undeformed superellipsoid implicit at pulled-back point
    a_i, b_i, c_i = gs.a[i], gs.b[i], gs.c[i]
    n1_i, n2_i = gs.n1[i], gs.n2[i]
    se_val = ((np.abs(bx_ref / a_i)**n1_i +
               np.abs(by_ref / b_i)**n1_i)**(n2_i / n1_i) +
              np.abs(bz_ref / c_i)**n2_i)
    n_inv = 1.0 / n2_i
    sdf = (se_val**n_inv - 1.0) * gs.r[i]

    # Apply JKR contact clips (flat faces at contact regions)
    clips = gs.contact_clips[i] if hasattr(gs, 'contact_clips') and gs.contact_clips else None
    if clips:
        _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))

    if gs.gtype[i] == 0:
        phi_f[ix0:ix1, iy0:iy1, iz0:iz1] += profile
    else:
        phi_i[ix0:ix1, iy0:iy1, iz0:iz1] += profile


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
    Returns φ_f, φ_i, φ_v arrays of shape (Ng, Ng).
    """
    # Auto-scale grid to resolve interface (V2.4: prevents aliasing on large domains)
    # Ensure grid spacing ≤ 2× interface_width so tanh spans ≥ 2 pixels
    min_Ng = int(np.ceil(max(p.Lx, p.Ly) / (p.interface_width * 2.0)))
    Ng = max(p.Ngrid, min_Ng)
    dx = p.Lx / Ng
    dy = p.Ly / Ng
    xg = np.linspace(dx/2, p.Lx - dx/2, Ng)
    yg = np.linspace(dy/2, p.Ly - dy/2, Ng)
    X, Y = np.meshgrid(xg, yg, indexing='ij')

    phi_f = np.zeros((Ng, Ng))
    phi_i = np.zeros((Ng, Ng))
    w = p.interface_width
    periodic = (p.boundary_mode == 'periodic')

    use_deformed = (p.deformable_enabled and gs.epsilon is not None)

    if use_deformed:
        # V2.3: Semi-Lagrangian deformed rendering — all particles go through
        # the pull-back path regardless of circle/superellipse distinction.
        for i in range(gs.N):
            theta_i = gs.theta[i] if hasattr(gs, 'theta') and gs.theta is not None else 0.0
            if periodic:
                rb = gs.r_bound[i] + 3 * w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_deformed_2d(X, Y, gs.x[i] + sx, gs.y[i] + sy,
                                       theta_i, gs, i, p, w, phi_f, phi_i)
            else:
                _stamp_deformed_2d(X, Y, gs.x[i], gs.y[i],
                                   theta_i, gs, i, p, w, phi_f, phi_i)
    elif gs.is_circle:
        r_eff, _ = compute_effective_radii(gs, p)
        has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None
        for i in range(gs.N):
            clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
            if periodic:
                rb = r_eff[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_circle_2d(X, Y, gs.x[i]+sx, gs.y[i]+sy,
                                     r_eff[i], w, phi_f, phi_i, gs.gtype[i],
                                     clips=clips_i)
            else:
                _stamp_circle_2d(X, Y, gs.x[i], gs.y[i],
                                 r_eff[i], w, phi_f, phi_i, gs.gtype[i],
                                 clips=clips_i)
    else:
        has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None
        for i in range(gs.N):
            clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
            if periodic:
                rb = gs.r_bound[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_superellipse_2d(
                        X, Y, gs.x[i]+sx, gs.y[i]+sy,
                        gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                        gs.r[i], w, phi_f, phi_i, gs.gtype[i],
                        clips=clips_i)
            else:
                _stamp_superellipse_2d(
                    X, Y, gs.x[i], gs.y[i],
                    gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                    gs.r[i], w, phi_f, phi_i, gs.gtype[i],
                    clips=clips_i)

    # Volume-conserving normalization (V2.1): see render_fields_3d.
    A_pixel = (p.Lx / Ng) * (p.Ly / Ng)
    func_mask_g = gs.gtype == 0
    inert_mask_g = gs.gtype == 1
    if gs.is_circle:
        A_f_true = float(np.sum(np.pi * gs.r[func_mask_g] ** 2)) \
            if np.any(func_mask_g) else 0.0
        A_i_true = float(np.sum(np.pi * gs.r[inert_mask_g] ** 2)) \
            if np.any(inert_mask_g) else 0.0
    else:
        A_f_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
            for j in np.where(func_mask_g)[0])) if np.any(func_mask_g) else 0.0
        A_i_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
            for j in np.where(inert_mask_g)[0])) if np.any(inert_mask_g) else 0.0
    A_solid_true = A_f_true + A_i_true
    for _iter in range(20):
        A_solid_rendered = float(np.sum(phi_f + phi_i)) * A_pixel
        if A_solid_rendered < 1e-30:
            break
        if abs(A_solid_rendered - A_solid_true) / A_solid_true < 1e-3:
            break
        scale = A_solid_true / A_solid_rendered
        phi_f *= scale
        phi_i *= scale
        total = phi_f + phi_i
        over = total > 1.0
        if np.any(over):
            phi_f[over] /= total[over]
            phi_i[over] /= total[over]

    return phi_f, phi_i, 1.0 - phi_f - phi_i


def _stamp_granule_3d(phi_f, phi_i, gs, i, cx, cy, cz, r_eff_3d_i,
                      xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, clips=None):
    """Stamp one 3D granule image at (cx, cy, cz) onto the field grids.

    When clips is provided, JKR half-space clipping creates flat contact faces.
    """
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
        sdf = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - r_eff_3d_i
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
        sdf = (se_val**n_inv - 1.0) * gs.r[i]

    if clips:
        _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))

    if gs.gtype[i] == 0:
        phi_f[ix0:ix1, iy0:iy1, iz0:iz1] += profile
    else:
        phi_i[ix0:ix1, iy0:iy1, iz0:iz1] += profile


def render_fields_3d(gs: GranuleSystem, p: Params):
    """
    Stamp each granule onto 3D grid using superellipsoid implicit function.
    For periodic boundaries, ghost images are stamped at boundary crossings.
    Returns φ_f, φ_i, φ_v arrays of shape (Ng, Ng, Ng).
    """
    # Auto-scale grid to resolve interface (V2.4: prevents aliasing on large domains)
    min_Ng = int(np.ceil(max(p.Lx, p.Ly, p.Lz) / (p.interface_width * 2.0)))
    # Cap 3D auto-scale at 200 to avoid excessive memory (200^3 = 8M voxels)
    Ng = max(p.Ngrid_3d, min(min_Ng, 200))
    dx_g = p.Lx / Ng
    dy_g = p.Ly / Ng
    dz_g = p.Lz / Ng
    xg = np.linspace(dx_g/2, p.Lx - dx_g/2, Ng)
    yg = np.linspace(dy_g/2, p.Ly - dy_g/2, Ng)
    zg = np.linspace(dz_g/2, p.Lz - dz_g/2, Ng)

    # For 3D, also widen interface if grid is still too coarse (cheaper than more voxels)
    w_3d = max(p.interface_width, max(dx_g, dy_g, dz_g))

    phi_f = np.zeros((Ng, Ng, Ng))
    phi_i = np.zeros((Ng, Ng, Ng))
    w = w_3d
    periodic = (p.boundary_mode == 'periodic')

    use_deformed = (p.deformable_enabled and gs.epsilon is not None)

    if gs.is_circle:
        r_eff_3d, _ = compute_effective_radii_3d(gs, p)
    else:
        r_eff_3d = gs.r

    has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None

    for i in range(gs.N):
        r_eff_i = r_eff_3d[i]
        clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
        if periodic:
            rb = gs.r_bound[i] + 3*w
            for sx in (-p.Lx, 0, p.Lx):
                for sy in (-p.Ly, 0, p.Ly):
                    for sz in (-p.Lz, 0, p.Lz):
                        gx = gs.x[i] + sx
                        gy = gs.y[i] + sy
                        gz = gs.z[i] + sz
                        if (gx + rb > 0 and gx - rb < p.Lx and
                            gy + rb > 0 and gy - rb < p.Ly and
                            gz + rb > 0 and gz - rb < p.Lz):
                            if use_deformed:
                                _stamp_deformed_3d(
                                    phi_f, phi_i, gs, i, gx, gy, gz,
                                    xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, p)
                            else:
                                _stamp_granule_3d(
                                    phi_f, phi_i, gs, i, gx, gy, gz,
                                    r_eff_i, xg, yg, zg,
                                    dx_g, dy_g, dz_g, Ng, w,
                                    clips=clips_i)
        else:
            if use_deformed:
                _stamp_deformed_3d(
                    phi_f, phi_i, gs, i,
                    gs.x[i], gs.y[i], gs.z[i],
                    xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, p)
            else:
                _stamp_granule_3d(
                    phi_f, phi_i, gs, i, gs.x[i], gs.y[i], gs.z[i],
                    r_eff_i, xg, yg, zg, dx_g, dy_g, dz_g, Ng, w,
                    clips=clips_i)

    # Volume-conserving normalization (V2.1): additive blending over-counts
    # at overlap zones.  Iterative scale-and-cap: each iteration normalises
    # the integral to match true granule volume, then caps per-voxel at 1.0.
    # Excess from capped voxels is redistributed in the next iteration.
    # Converges in ~5-10 iterations to <0.1% volume error.
    V_voxel = dx_g * dy_g * dz_g
    func_mask_g = gs.gtype == 0
    inert_mask_g = gs.gtype == 1
    if gs.is_circle:
        V_f_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[func_mask_g] ** 3)) \
            if np.any(func_mask_g) else 0.0
        V_i_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[inert_mask_g] ** 3)) \
            if np.any(inert_mask_g) else 0.0
    else:
        V_f_true = float(sum(superellipsoid_volume(
            gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
            for j in np.where(func_mask_g)[0])) if np.any(func_mask_g) else 0.0
        V_i_true = float(sum(superellipsoid_volume(
            gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
            for j in np.where(inert_mask_g)[0])) if np.any(inert_mask_g) else 0.0
    V_solid_true = V_f_true + V_i_true
    for _iter in range(20):
        V_solid_rendered = float(np.sum(phi_f + phi_i)) * V_voxel
        if V_solid_rendered < 1e-30:
            break
        rel_err = abs(V_solid_rendered - V_solid_true) / V_solid_true
        if rel_err < 1e-3:
            break
        scale = V_solid_true / V_solid_rendered
        phi_f *= scale
        phi_i *= scale
        # Cap per-voxel total to 1.0 (preserving phi_f/phi_i ratio)
        total = phi_f + phi_i
        over = total > 1.0
        if np.any(over):
            phi_f[over] /= total[over]
            phi_i[over] /= total[over]

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
    dxg = p.Lx / phi_f.shape[0]
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
    migrating_mask = gs.cell_state == int(CellState.MIGRATING)
    m['n_migrating_cells'] = int(np.sum(migrating_mask))
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

    # True granule-based volume fractions (V2.1, invariant to rendering)
    if gs.is_3d:
        V_domain = p.Lx * p.Ly * p.Lz
        if gs.is_circle:
            V_f_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.func_mask] ** 3))
            V_i_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.inert_mask] ** 3))
        else:
            V_f_true = float(sum(superellipsoid_volume(
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                for j in np.where(gs.func_mask)[0]))
            V_i_true = float(sum(superellipsoid_volume(
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                for j in np.where(gs.inert_mask)[0]))
    else:
        V_domain = p.Lx * p.Ly
        if gs.is_circle:
            V_f_true = float(np.sum(np.pi * gs.r[gs.func_mask] ** 2))
            V_i_true = float(np.sum(np.pi * gs.r[gs.inert_mask] ** 2))
        else:
            V_f_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                for j in np.where(gs.func_mask)[0]))
            V_i_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                for j in np.where(gs.inert_mask)[0]))
    m['phi_f_true'] = V_f_true / V_domain
    m['phi_i_true'] = V_i_true / V_domain
    m['phi_solid_true'] = (V_f_true + V_i_true) / V_domain
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

    # V2.3: Deformation metrics (LS-DEM)
    if gs.epsilon is not None:
        eps_mag = np.sqrt(np.sum(gs.epsilon**2, axis=1))
        m['def_strain_mean'] = float(np.mean(eps_mag))
        m['def_strain_max'] = float(np.max(eps_mag))
        m['def_strain_std'] = float(np.std(eps_mag))
        for alpha in range(gs.epsilon.shape[1]):
            m[f'def_mode_{alpha}_mean'] = float(np.mean(gs.epsilon[:, alpha]))
            m[f'def_mode_{alpha}_std'] = float(np.std(gs.epsilon[:, alpha]))

    # V2.1: Two-compartment volume conservation (Voronoi shrink-wrap)
    if gs.N > 0 and np.any(gs.func_mask) and np.any(gs.inert_mask):
        pos_vc = gs.positions()
        if periodic:
            if gs.is_3d:
                tree_vc = cKDTree(pos_vc, boxsize=[p.Lx, p.Ly, p.Lz])
            else:
                tree_vc = cKDTree(pos_vc, boxsize=[p.Lx, p.Ly])
        else:
            tree_vc = tree  # reuse the tree built above
        grid_shape = phi_f.shape
        if gs.is_3d:
            nx, ny, nz = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
            cz = np.linspace(0.5 * p.Lz / nz, p.Lz - 0.5 * p.Lz / nz, nz)
            gx_v, gy_v, gz_v = np.meshgrid(cx, cy, cz, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel(), gz_v.ravel()])
        else:
            nx, ny = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
            gx_v, gy_v = np.meshgrid(cx, cy, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel()])
        _, nearest_idx = tree_vc.query(voxel_pts)
        nearest_type = gs.gtype[nearest_idx].reshape(grid_shape)
        fm = nearest_type == 0
        im = nearest_type == 1
        n_func_v = int(np.sum(fm))
        n_inert_v = int(np.sum(im))
        m['x_f'] = n_func_v / max(phi_f.size, 1)
        m['x_i'] = n_inert_v / max(phi_f.size, 1)
        m['phi_v_in_func'] = float(np.mean(phi_v[fm])) if n_func_v > 0 else 0.0
        m['phi_v_in_inert'] = float(np.mean(phi_v[im])) if n_inert_v > 0 else 0.0
        m['phi_solid_func'] = float(np.mean((phi_f + phi_i)[fm])) if n_func_v > 0 else 0.0
        m['phi_solid_inert'] = float(np.mean((phi_f + phi_i)[im])) if n_inert_v > 0 else 0.0
        m['phi_v_crosscheck'] = m['x_f'] * m['phi_v_in_func'] + m['x_i'] * m['phi_v_in_inert']

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
# Data serialization (V1.5, updated V2.6)
# ══════════════════════════════════════════════════════════════════════

def check_disk_space(output_dir):
    """Check available disk space for the output directory.

    Returns 'ok' (>= 5 GB), 'warning' (1-5 GB), or 'critical' (< 1 GB).
    """
    import shutil
    try:
        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            return 'critical'
        elif free_gb < 5.0:
            return 'warning'
        return 'ok'
    except OSError:
        return 'ok'  # can't check, assume ok



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
    data['cell_bridge_locked'] = gs.cell_bridge_locked.copy()
    data['cell_overcrowd_age'] = gs.cell_overcrowd_age.copy()
    data['cell_alignment'] = gs.cell_alignment.copy()
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

    # V2.6: LS-DEM deformation state (needed for resume)
    if gs.epsilon is not None:
        data['epsilon'] = gs.epsilon.copy()
        data['d_epsilon'] = gs.d_epsilon.copy()

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
# Resume from snapshot (V2.6)
# ══════════════════════════════════════════════════════════════════════

def find_last_snapshot(run_dir):
    """Find the last (highest-numbered) snapshot file in a run directory.

    Returns the absolute path to the last snap_XXXX.npz file.
    Raises FileNotFoundError if none found.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"No snapshots/ directory in {run_dir}")
    snap_files = sorted(
        f for f in os.listdir(snap_dir)
        if f.startswith('snap_') and f.endswith('.npz'))
    if not snap_files:
        raise FileNotFoundError(f"No snap_*.npz files in {snap_dir}")
    return os.path.join(snap_dir, snap_files[-1])


def restore_gs_from_snapshot(snap_path, p):
    """Restore a GranuleSystem from a saved snapshot .npz file.

    Parameters
    ----------
    snap_path : str
        Path to a snap_XXXX.npz file.
    p : Params
        Simulation parameters (needed for mode, LS-DEM config).

    Returns
    -------
    gs : GranuleSystem
        Fully restored granule system.
    t_resume : float
        Simulation time at the snapshot.
    snap_idx : int
        Snapshot index (parsed from filename).
    """
    data = dict(np.load(snap_path, allow_pickle=False))
    t_resume = float(data['time'])

    # Parse snap index from filename: snap_XXXX.npz -> XXXX
    basename = os.path.basename(snap_path)
    snap_idx = int(basename.replace('snap_', '').replace('.npz', ''))

    # Construct GranuleSystem
    gs = GranuleSystem(
        x=data['x'], y=data['y'], r=data['r'],
        gtype=data['gtype'], n_cells=data['n_cells'],
        z=data['z'] if 'z' in data else None,
        a=data['a'], b=data['b'],
        c=data.get('c', None),
        n1=data['n1'], n2=data['n2'],
        theta=data.get('theta', None),
        quat=data.get('quat', None),
        mode=p.mode
    )

    # Restore velocity state
    gs.vx[:] = data['vx']
    gs.vy[:] = data['vy']
    gs.vz[:] = data['vz']

    # Restore per-granule cell state
    gs.n_attached[:] = data['n_attached']
    gs.spread_fraction[:] = data['spread_fraction']
    gs.fa_maturity[:] = data['fa_maturity']
    gs.n_overcrowded[:] = data['n_overcrowded']

    # Restore per-cell arrays
    gs.cell_granule_id[:] = data['cell_granule_id']
    gs.cell_state[:] = data['cell_state']
    gs.cell_theta_local[:] = data['cell_theta_local']
    gs.cell_eta_local[:] = data.get('cell_eta_local', np.zeros(gs.total_cells))
    gs.cell_omega_local[:] = data.get('cell_omega_local', np.zeros(gs.total_cells))
    gs.cell_fx[:] = data['cell_fx']
    gs.cell_fy[:] = data['cell_fy']
    gs.cell_fz[:] = data.get('cell_fz', np.zeros(gs.total_cells))
    gs.cell_bridge_target[:] = data['cell_bridge_target']
    gs.cell_bridge_age[:] = data['cell_bridge_age']
    gs.cell_bridge_locked[:] = data['cell_bridge_locked']
    gs.cell_overcrowd_age[:] = data['cell_overcrowd_age']
    gs.cell_alignment[:] = data['cell_alignment']
    gs.cell_contact_area[:] = data['cell_contact_area']
    gs.cell_offset[:] = data['cell_offset']

    # Unwrapped positions: reset to current (displacement tracked from resume point)
    gs.x_unwrap[:] = gs.x
    gs.y_unwrap[:] = gs.y
    gs.z_unwrap[:] = gs.z

    # LS-DEM: re-init SDF grids from params, then restore deformation state
    if p.deformable_enabled:
        from lsdem import init_lsdem
        init_lsdem(gs, p)
        if 'epsilon' in data:
            gs.epsilon[:] = data['epsilon']
            gs.d_epsilon[:] = data['d_epsilon']
        else:
            print("  WARNING: Resuming deformable run but snapshot has no epsilon data.")
            print("           Starting deformation from zero.")

    return gs, t_resume, snap_idx


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
    # Re-derive phi targets in case overrides were applied after __post_init__
    if p.phi_solid_target > 0:
        p.phi_f_target = p.phi_solid_target * p.func_ratio
        p.phi_i_target = p.phi_solid_target * (1.0 - p.func_ratio)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"  Using random seed: {seed}")
    rng = np.random.default_rng(seed)

    # JIT warmup: trigger Numba compilation before timing begins
    if HAS_NUMBA and p.shape_enabled:
        _warmup_jit(p.mode == "3D")

    # ── V2.6: Resume or fresh start ──
    resume_t = 0.0
    resume_snap_idx = -1  # -1 means fresh start (snap_counter will be 0)
    hist = []

    if p.resume_from:
        print(f"\n  RESUMING from: {p.resume_from}")
        snap_path = find_last_snapshot(p.resume_from)
        gs, resume_t, resume_snap_idx = restore_gs_from_snapshot(snap_path, p)
        print(f"  Restored state at t={resume_t:.2f} h (snapshot {resume_snap_idx})")
        # Load existing history if available
        hist_json = os.path.join(p.resume_from, 'history.json')
        if os.path.exists(hist_json):
            with open(hist_json) as _hf:
                hist = json.load(_hf)
            print(f"  Loaded {len(hist)} history entries")
    else:
        print(f"\n  Mode: {p.mode}")
        print("  Generating packing...")
        # Mode-aware packing generation
        if p.mode == "3D":
            gs = generate_packing_3d(p, seed=seed)
        elif p.mode == "2D-slice":
            gs = generate_packing_2d_slice(p, seed=seed)
        else:
            gs = generate_packing(p, seed=seed)
        # V2.3: Initialize LS-DEM deformable particles if enabled
        if p.deformable_enabled:
            from lsdem import init_lsdem
            init_lsdem(gs, p)

    # Store initial positions (displacement tracked from this point)
    x0 = gs.x.copy(); y0 = gs.y.copy()
    z0 = gs.z.copy() if gs.is_3d else None

    n_steps = int(p.t_total / p.dt)
    snaps = []

    # V1.5: Initialize output directory and save metadata
    output_dir = p.output_dir
    snap_counter = [resume_snap_idx + 1]  # continues from resume point
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
            'cell_bridge_locked': gs.cell_bridge_locked.copy(),
            'cell_overcrowd_age': gs.cell_overcrowd_age.copy(),
            'cell_alignment': gs.cell_alignment.copy(),
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
        # V2.3: LS-DEM deformation state
        if gs.epsilon is not None:
            snap['epsilon'] = gs.epsilon.copy()
            snap['d_epsilon'] = gs.d_epsilon.copy()
        snaps.append(snap)

        # V1.5: Save to disk (V2.6: with disk space guard and rolling cleanup)
        if p.save_data:
            disk_status = check_disk_space(output_dir)
            if disk_status == 'critical':
                print(f"  CRITICAL: < 1 GB disk space — SKIPPING snapshot "
                      f"{snap_counter[0]} at t={t:.1f} h")
            else:
                use_fields = p.save_fields and disk_status != 'warning'
                if disk_status == 'warning':
                    print(f"  WARNING: < 5 GB disk space — saving WITHOUT fields")
                save_snapshot_to_disk(
                    snap_counter[0], gs, p, t, F, output_dir,
                    save_fields=use_fields,
                    phi_f=pf, phi_i=pi, phi_v=pv,
                    contacts=contacts)
                snap_counter[0] += 1

        return m

    # ── Print header ──
    _hdr = (f"\n  {'t(h)':>6} {'f_cl':>5} {'f_lf':>6} {'v_cl':>5} "
            f"{'tissue':>7} {'bridges':>7} {'attach':>7} {'spread':>6} "
            f"{'FA_mat':>6} {'disp_f':>7} {'K_KC':>8} "
            f"{'bCells':>6} {'bF_avg':>6} {'lock':>4}")

    def _print_metrics(t, m):
        print(f"  {t:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} "
              f"{m['void_nc']:5d} {m['tissue_frac']:7.3f} "
              f"{m['n_bridges']:7d} {m['n_attached_total']:7.0f} "
              f"{m['mean_spread_frac']:6.2f} {m['mean_fa_maturity']:6.2f} "
              f"{m['disp_func']:7.1f} {m['K_kozeny_carman']:8.1f} "
              f"{m['n_bridging_cells']:6d} {m['bridge_force_mean']:6.1f} "
              f"{m['n_locked_in_cells']:4d}")

    # ── Initial save (only for fresh starts) ──
    resume_step = int(round(resume_t / p.dt))
    if not p.resume_from:
        update_cell_state(gs, p, 0.0, rng)
        if gs.is_3d:
            F0, _, contacts0 = compute_forces_3d(gs, p, rng)
        else:
            F0, _, contacts0 = compute_forces(gs, p, rng)
        m = save(0.0, F0, contacts0)
        print(_hdr)
        _print_metrics(0.0, m)
    else:
        print(f"  Resuming from step {resume_step + 1} "
              f"(t={resume_t:.2f} h) → step {n_steps} (t={p.t_total:.2f} h)")
        print(_hdr)

    wall_t0 = timer.time()
    t = resume_t
    for s in range(resume_step + 1, n_steps + 1):
        t += p.dt
        F, contacts = step(gs, p, rng, t)

        if s % p.save_every == 0:
            m = save(t, F, contacts)
            _print_metrics(t, m)

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
    print("  GELS V1.6: Cell-Driven Granular Rearrangement")
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
    if p.cell_surface_coverage > 0:
        print(f"  Cell surface coverage={p.cell_surface_coverage:.2f} "
              f"({'stacked' if p.cell_surface_coverage > 1 else 'monolayer'}), "
              f"d_cell={p.cell_diameter} µm, h_spread={p.cell_height_spread} µm")
    else:
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