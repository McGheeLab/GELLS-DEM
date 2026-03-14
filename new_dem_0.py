"""
Overdamped Particle Dynamics Model for Cell-Driven Granular Rearrangement
==========================================================================

MATHEMATICAL MODEL
------------------
N rigid circular granules (functional or inert) in 2D, overdamped regime.
Granules maintain their shape; cells bridge nearby functional pairs and
pull them together.

EQUATIONS OF MOTION (overdamped Langevin):

    γ_i dx_i/dt = F_i^contact + F_i^cell + F_i^wall + F_i^noise

where γ_i = 6πη R_i is the Stokes drag on granule i.

FORCE LAWS:

(1) Contact repulsion (Hertzian, physically-based stiffness):
    R*_ij = R_i R_j / (R_i + R_j)              (reduced radius)
    E*    = E / [2(1 − ν²)]                    (effective modulus, identical materials)
    F_ij^contact = (4/3) E* √R* δ^{3/2} n̂_ij  if δ_ij = R_i+R_j - d_ij > 0
                 = 0                            otherwise

    Volume conservation: overlap lens area is tracked per granule and
    redistributed as an inflated effective radius for rendering:
        r_eff_i = √(r_i² + ΔA_i / π)

(2) Cell-mediated attraction (motor-clutch model, functional–functional only):
    Cells are initially spheres (d=20 µm).  After ~3 h they attach to the
    hydrogel via integrin clutches and spread into oblate ellipsoids (~5 µm
    tall, volume conserved).  If the spread footprint exceeds the granule
    area, excess cells crawl on top of neighbours.

    Attached cells sense nearby granules within a filopodia range
    (cell_sense_distance) and form bridges.  Bridge force per cell comes
    from the motor-clutch model (Chan & Odde 2008):

        k_sub  = π E a / (1−ν²)              (substrate stiffness at cell scale)
        k_opt  = n_clutches · k_clutch        (clutch ensemble stiffness)
        engagement = k_on / (k_on + k_off)    (steady-state clutch fraction)
        F_mc   = F_stall · k_sub/(k_sub+k_opt) · engagement · FA_maturity

    Bridge count and net force:
        gap_ij = d_ij − R_i − R_j             (surface separation)
        proximity = 1 − gap/cell_sense_distance
        n_bridges = √(n_avail_i · n_avail_j) · proximity
        F_ij^cell = F_mc · n_bridges · n̂_ij   (if gap > L_rest)
        |F_ij^cell| ≤ F_max · n_bridges       (force cap)

(3) Wall repulsion (Hertz, sphere against rigid flat):
    E*_wall = E / (1 − ν²)                         (rigid wall limit)
    F_wall  = (4/3) E*_wall √R_i · pen^{3/2} n̂     for each boundary

(4) Activity noise (small stochastic kicks on functional granules):
    F_noise ~ √(2 γ_i T_active) · ξ(t)            (cell-driven fluctuations)

(5) Tangential friction (area-dependent, hydrogel tribology):
    A_ij   = π R*_ij δ_ij                         (Hertzian contact area)
    v_t    = (v_j − v_i) − [(v_j − v_i)·n̂] n̂     (relative tangential velocity)
    F_fric = τ₀ · A_ij · tanh(|v_t|/v_ref) · (−v_t/|v_t|)

    τ₀ depends on pair surface chemistry (Gong 2006, Pitenis 2014):
        inert–inert (bare Gemini gel):           τ₀ ≈  50 Pa
        inert–functional (bare–collagen-I):      τ₀ ≈ 500 Pa
        functional–functional (col-I–col-I):     τ₀ ≈ 2000 Pa

(6) DMT adhesion (constant attractive force during contact):
    F_adh  = 2π W_adh R*_ij                       (Derjaguin-Muller-Toporov)
    F_norm = F_contact − F_adh                     (net; can be negative = attractive)

    W_adh depends on pair surface chemistry:
        inert–inert:             W ≈ 0.5 mJ/m²
        inert–functional:        W ≈ 1.0 mJ/m²
        functional–functional:   W ≈ 2.0 mJ/m²

CELL COUNT PER GRANULE (projected-area limited):
    A_cell  = π (d/2)²                                 (sphere projected area)
    n_cells = min(n_input, floor(π R² · coverage / A_cell))

OBSERVABLES (rendered from particle positions at each save step):
    φ_f(x), φ_i(x), φ_v(x) = 1 - φ_f - φ_i
    → void topology, functional topology, packing evolution, tissue metrics

PHYSICAL PARAMETER MAPPING:
    E      ~ 1-100 kPa               (hydrogel Young's modulus)
    ν      ~ 0.4-0.5                  (Poisson's ratio, nearly incompressible)
    η      ~ 1e-3 Pa·s               (culture medium viscosity)
    d_cell = 20 µm                    (initial cell diameter)
    h_cell = 5 µm                     (spread cell height)
    n_motors ~ 200, F_stall ~ 2 pN    (myosin motors)
    n_clutches ~ 75, k_clutch ~ 0.5 nN/µm  (integrin clutches)
    t_attach ~ 3 h                    (cell attachment onset)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
from scipy.special import gamma as _gamma
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time as timer


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    # ── Domain (µm) ──
    Lx: float = 800.0
    Ly: float = 800.0

    # ── Granule physical properties ──
    R_func_mean: float = 40.0       # functional radius (µm)
    R_func_std: float = 5.0
    R_inert_mean: float = 60.0      # inert radius (µm)
    R_inert_std: float = 8.0

    # ── Composition ──
    phi_f_target: float = 0.25      # functional area fraction target
    phi_i_target: float = 0.20      # inert area fraction target

    # ── Cell geometry ──
    n_cells_per_granule: int = 8    # cells seeded per functional granule
    cell_diameter: float = 20.0     # µm, initial spherical cell diameter
    cell_height_spread: float = 5.0 # µm, height of spread ellipsoidal cell
    cell_coverage: float = 0.6      # max fraction of granule projected area covered

    # ── Cell attachment & spreading timeline ──
    t_attach_onset: float = 3.0     # hours, when cells begin attaching
    t_attach_half: float = 1.5      # hours after onset for 50% attachment
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

    # ── Cell sensing & bridging ──
    cell_sense_distance: float = 40.0  # µm, filopodia sensing range
    F_max_per_cell: float = 50.0       # nN, absolute max force cap per cell
    L_rest: float = 5.0                # µm, rest length (~ spread cell thickness)

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

    # ── Granule shape (V1.3 — superellipses) ──
    shape_enabled: bool = False                # False = circles (V1.2 compat)
    aspect_ratio_func_mean: float = 1.0        # a/b ratio for functional granules
    aspect_ratio_func_std: float = 0.0
    aspect_ratio_inert_mean: float = 1.0       # a/b ratio for inert granules
    aspect_ratio_inert_std: float = 0.0
    blockiness_func_mean: float = 2.0          # n exponent (2=ellipse, >2=blocky)
    blockiness_func_std: float = 0.0
    blockiness_inert_mean: float = 2.0
    blockiness_inert_std: float = 0.0
    drag_scale_rot: float = 0.05               # rotational drag scaling
    omega_max: float = 1.0                     # rad/h, angular velocity cap

    # ── Rendering ──
    Ngrid: int = 200                # grid for field rendering
    interface_width: float = 3.0    # µm, tanh smoothing

    @property
    def L_max(self):
        """Max bridging distance equals cell sensing distance."""
        return self.cell_sense_distance

    @property
    def save_every(self):
        return max(1, int(self.save_every_h / self.dt))


# ══════════════════════════════════════════════════════════════════════
# Granule data structure
# ══════════════════════════════════════════════════════════════════════

class GranuleSystem:
    """Tracks all granule and per-granule cell state."""
    def __init__(self, x, y, r, gtype, n_cells,
                 a=None, b=None, n_shape=None, theta=None):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)   # equivalent radius (area = pi*r^2)
        self.gtype = np.array(gtype, dtype=int)   # 0=func, 1=inert
        self.n_cells = np.array(n_cells, dtype=np.float64)  # seeded cells
        self.N = len(x)
        self.func_mask = self.gtype == 0
        self.inert_mask = self.gtype == 1

        # ── Shape state (V1.3 — superellipses) ──
        if a is not None:
            self.a = np.array(a, dtype=np.float64)           # semi-axis, body x
            self.b = np.array(b, dtype=np.float64)           # semi-axis, body y
            self.n_shape = np.array(n_shape, dtype=np.float64)  # blockiness
            self.theta = np.array(theta, dtype=np.float64)   # orientation (rad)
            self.is_circle = False
        else:
            self.a = self.r.copy()
            self.b = self.r.copy()
            self.n_shape = np.full(self.N, 2.0)
            self.theta = np.zeros(self.N)
            self.is_circle = True
        self.r_bound = np.maximum(self.a, self.b)  # bounding circle radius
        self.omega = np.zeros(self.N)               # angular velocity (rad/h)

        # ── Cell state (per granule) ──
        self.n_attached = np.zeros(self.N)       # cells that have attached
        self.spread_fraction = np.zeros(self.N)  # 0 = sphere, 1 = fully spread
        self.fa_maturity = np.zeros(self.N)      # focal adhesion maturity [0, 1]
        self.n_overcrowded = np.zeros(self.N)    # cells crawling on others

        # ── Velocity state (for tangential friction calculation) ──
        self.vx = np.zeros(self.N)
        self.vy = np.zeros(self.N)

    def positions(self):
        return np.column_stack([self.x, self.y])


# ══════════════════════════════════════════════════════════════════════
# Hertz contact mechanics
# ══════════════════════════════════════════════════════════════════════

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


def superellipse_point(t, a, b, n):
    """Point (x, y) on superellipse boundary at parameter t (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    x = a * np.sign(ct) * np.abs(ct)**e
    y = b * np.sign(st) * np.abs(st)**e
    return x, y


def superellipse_tangent(t, a, b, n):
    """Unnormalised tangent vector dx/dt, dy/dt (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    # dx/dt = a * (2/n) * |cos t|^(2/n - 1) * sin t  (with signs handled)
    dxdt = -a * e * np.sign(ct) * np.abs(ct)**(e - 1.0) * st
    dydt =  b * e * np.sign(st) * np.abs(st)**(e - 1.0) * ct
    return dxdt, dydt


def superellipse_normal_vec(t, a, b, n):
    """Outward unit normal at parameter t (body frame)."""
    dxdt, dydt = superellipse_tangent(t, a, b, n)
    # outward normal = (dy/dt, -dx/dt) normalised
    nx, ny = dydt, -dxdt
    mag = np.sqrt(nx*nx + ny*ny)
    if mag < 1e-30:
        return 0.0, 0.0
    return nx / mag, ny / mag


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


def _world_to_body(px, py, cx, cy, theta):
    """Transform world point(s) to body frame of granule at (cx,cy,theta)."""
    dx, dy = px - cx, py - cy
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cos_th * dx + sin_th * dy, -sin_th * dx + cos_th * dy


def _body_to_world(bx, by, cx, cy, theta):
    """Transform body point(s) to world frame."""
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cx + cos_th * bx - sin_th * by, cy + sin_th * bx + cos_th * by


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


def compute_effective_radii(gs):
    """
    Volume-conserving effective radii.

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


def update_cell_state(gs: GranuleSystem, p: Params, t: float):
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

        # ── Attachment (sigmoidal kinetics) ──
        if t >= p.t_attach_onset:
            tau = t - p.t_attach_onset
            frac = 1.0 / (1.0 + np.exp(
                -3.0 * (tau - p.t_attach_half) / max(0.1, p.t_attach_half)))
            gs.n_attached[i] = gs.n_cells[i] * frac
        else:
            gs.n_attached[i] = 0.0

        # ── Spreading (linear ramp after half-attachment reached) ──
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
    gap = 2.0  # minimum gap between granule surfaces (µm)

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
        for _ in range(800):
            cx = rng.uniform(r_bound_val + gap, p.Lx - r_bound_val - gap)
            cy = rng.uniform(r_bound_val + gap, p.Ly - r_bound_val - gap)
            ok = True
            for j in range(len(xs)):
                dx = cx - xs[j]; dy = cy - ys[j]
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
    return gs


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
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j,0] - pos[i,0]
        dy = pos[j,1] - pos[i,1]
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
            result = find_contact_superellipses(
                pos[i,0], pos[i,1], gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                pos[j,0], pos[j,1], gs.a[j], gs.b[j], gs.n_shape[j], gs.theta[j])
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

        # ── Cell bridging (motor-clutch model, functional–functional) ──
        if gs.gtype[i] == 0 and gs.gtype[j] == 0:
            if gs.is_circle:
                gap = d - gs.r[i] - gs.r[j]
            else:
                # Surface-to-surface gap: negative of overlap (or use bounding)
                if in_contact:
                    gap = -overlap
                else:
                    # Approximate gap from bounding radii
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0:
                        gap = 0.0  # conservative: may be in contact

            if 0 < gap < p.cell_sense_distance:
                # Only attached, non-overcrowded cells can bridge
                n_avail_i = max(0.0, gs.n_attached[i] - gs.n_overcrowded[i])
                n_avail_j = max(0.0, gs.n_attached[j] - gs.n_overcrowded[j])

                if n_avail_i > 0.1 and n_avail_j > 0.1:
                    # Proximity factor: cells more likely to bridge when close
                    proximity = 1.0 - gap / p.cell_sense_distance

                    # Number of bridging cells (geometric mean × proximity)
                    n_br = np.sqrt(n_avail_i * n_avail_j) * proximity

                    # Motor-clutch force per cell (stiffness + FA maturity)
                    avg_maturity = 0.5 * (gs.fa_maturity[i] + gs.fa_maturity[j])
                    F_per_cell = motor_clutch_force(p.E_modulus, p, avg_maturity)

                    # Bridge engagement: tension when gap > rest length
                    if gap > p.L_rest:
                        F_mag = F_per_cell * n_br
                        F_cap = p.F_max_per_cell * n_br
                        F_mag = min(F_mag, F_cap)
                        # Attractive: pull i toward j
                        F[i,0] += F_mag * nx; F[i,1] += F_mag * ny
                        F[j,0] -= F_mag * nx; F[j,1] -= F_mag * ny

    # ── Wall repulsion ──
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

    # ── Active noise on functional granules ──
    if p.T_active > 0:
        for i in range(N):
            if gs.gtype[i] == 0:
                gamma_i = p.drag_scale * gs.r[i]
                noise_amp = np.sqrt(2 * gamma_i * p.T_active / p.dt)
                F[i,0] += noise_amp * rng.standard_normal()
                F[i,1] += noise_amp * rng.standard_normal()

    return F, torques


# ══════════════════════════════════════════════════════════════════════
# Time integration (overdamped: γ dx/dt = F  →  dx = F/γ · dt)
# ══════════════════════════════════════════════════════════════════════

def step(gs: GranuleSystem, p: Params, rng, t: float):
    """One overdamped Euler step with cell state evolution and rotation."""
    update_cell_state(gs, p, t)
    F, torques = compute_forces(gs, p, rng)

    for i in range(gs.N):
        gamma_i = p.drag_scale * gs.r[i]
        vx = F[i,0] / gamma_i
        vy = F[i,1] / gamma_i
        # Velocity cap
        v = np.sqrt(vx*vx + vy*vy)
        if v > p.v_max:
            vx *= p.v_max / v; vy *= p.v_max / v
        # Store velocities for next step's friction calculation
        gs.vx[i] = vx
        gs.vy[i] = vy
        gs.x[i] += vx * p.dt
        gs.y[i] += vy * p.dt

        # ── Rotational dynamics (V1.3, superellipses only) ──
        if not gs.is_circle:
            gamma_rot = p.drag_scale_rot * (gs.a[i]**2 + gs.b[i]**2) / 2.0
            if gamma_rot > 1e-20:
                gs.omega[i] = torques[i] / gamma_rot
                gs.omega[i] = np.clip(gs.omega[i], -p.omega_max, p.omega_max)
                gs.theta[i] += gs.omega[i] * p.dt

        # Hard wall clamp
        rb = gs.r_bound[i]
        gs.x[i] = np.clip(gs.x[i], rb+0.5, p.Lx-rb-0.5)
        gs.y[i] = np.clip(gs.y[i], rb+0.5, p.Ly-rb-0.5)

    return F


# ══════════════════════════════════════════════════════════════════════
# Phase field rendering (from particle positions)
# ══════════════════════════════════════════════════════════════════════

def render_fields(gs: GranuleSystem, p: Params):
    """
    Stamp each granule as tanh-profile shape onto grid.
    For circles: radial profile with volume-conserving effective radii.
    For superellipses: implicit-function-based signed distance.
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

    if gs.is_circle:
        # Volume-conserving radii (V1.2 path)
        r_eff, _ = compute_effective_radii(gs)
        for i in range(gs.N):
            dist = np.sqrt((X - gs.x[i])**2 + (Y - gs.y[i])**2)
            profile = 0.5 * (1.0 - np.tanh((dist - r_eff[i]) / w))
            if gs.gtype[i] == 0:
                phi_f = np.maximum(phi_f, profile)
            else:
                phi_i = np.maximum(phi_i, profile)
    else:
        # Superellipse implicit function
        for i in range(gs.N):
            # Transform grid to body frame
            bx, by = _world_to_body(X, Y, gs.x[i], gs.y[i], gs.theta[i])
            # Implicit function value: < 1 inside, > 1 outside
            se_val = (np.abs(bx) / gs.a[i])**gs.n_shape[i] + \
                     (np.abs(by) / gs.b[i])**gs.n_shape[i]
            # Approximate signed distance
            n_inv = 1.0 / gs.n_shape[i]
            dist_approx = (se_val**n_inv - 1.0) * gs.r[i]
            profile = 0.5 * (1.0 - np.tanh(dist_approx / w))
            if gs.gtype[i] == 0:
                phi_f = np.maximum(phi_f, profile)
            else:
                phi_i = np.maximum(phi_i, profile)

    # Prevent total > 1
    total = phi_f + phi_i
    over = total > 0.99
    if np.any(over):
        phi_f[over] *= 0.99 / total[over]
        phi_i[over] *= 0.99 / total[over]

    return phi_f, phi_i, 1.0 - phi_f - phi_i


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

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
    m['phi_f_mean'] = np.mean(phi_f)
    m['phi_i_mean'] = np.mean(phi_i)
    m['phi_v_mean'] = np.mean(phi_v)

    # Functional connectivity
    fn, fl, fc = connectivity(phi_f, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)

    # Void connectivity
    vn, vl, vc = connectivity(phi_v, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)

    # Inert connectivity
    inn, il, _ = connectivity(phi_i, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    # Tissue: dense functional regions
    m['tissue_frac'] = float(np.mean(phi_f > 0.5))

    # Max cluster area
    thr = np.mean(phi_f) + 0.3*np.std(phi_f)
    b = (phi_f > thr).astype(int); lab, nc = label(b)
    dxg = p.Lx / p.Ngrid
    if nc > 0:
        sizes = np.array([np.sum(lab==l) for l in range(1, nc+1)])
        m['func_max_area'] = float(np.max(sizes) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    # Mean displacement from initial (stored externally)
    # Packing in functional-rich region
    fr = phi_f > thr
    pt = phi_f + phi_i
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    # Mean force magnitude
    F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[gs.func_mask])) if np.any(gs.func_mask) else 0.0

    # Overlap & contact diagnostics
    pos = gs.positions()
    max_rb = float(np.max(gs.r_bound))
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

    n_contacts = 0
    n_contacts_ff = 0   # functional–functional
    n_contacts_if = 0   # inert–functional
    n_contacts_ii = 0   # inert–inert
    max_overlap_ratio = 0.0
    total_overlap_area = 0.0
    n_bridges = 0

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j,0] - pos[i,0]; dy = pos[j,1] - pos[i,1]
        d = np.sqrt(dx*dx + dy*dy)

        if gs.is_circle:
            overlap = gs.r[i] + gs.r[j] - d
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
        dx = pos[j,0] - pos[i,0]; dy = pos[j,1] - pos[i,1]
        d = np.sqrt(dx*dx + dy*dy)
        if gs.is_circle:
            gap = d - gs.r[i] - gs.r[j]
        else:
            gap = d - gs.r_bound[i] - gs.r_bound[j]
            if gap < 0:
                gap = 0.0
        if 0 < gap < p.cell_sense_distance:
            n_bridges += 1

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
    m['area_conservation'] = 1.0 - total_overlap_area / total_granule_area
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

    return m


def compute_displacement(gs, x0, y0):
    """RMS displacement from initial positions."""
    dx = gs.x - x0; dy = gs.y - y0
    disp = np.sqrt(dx**2 + dy**2)
    return float(np.mean(disp[gs.func_mask])), float(np.mean(disp[gs.inert_mask]))


# ══════════════════════════════════════════════════════════════════════
# Main simulation loop
# ══════════════════════════════════════════════════════════════════════

def run(p=None, seed=42):
    if p is None: p = Params()
    rng = np.random.default_rng(seed)

    print("\n  Generating packing...")
    gs = generate_packing(p, seed=seed)

    # Store initial positions for displacement tracking
    x0 = gs.x.copy(); y0 = gs.y.copy()

    n_steps = int(p.t_total / p.dt)
    hist, snaps, disp_hist = [], [], []

    def save(t, F):
        pf, pi, pv = render_fields(gs, p)
        m = compute_metrics(gs, p, pf, pi, pv, t, F)
        df, di = compute_displacement(gs, x0, y0)
        m['disp_func'] = df; m['disp_inert'] = di
        hist.append(m)
        snaps.append((pf.copy(), pi.copy(), pv.copy(),
                       gs.x.copy(), gs.y.copy(), gs.r.copy(), gs.gtype.copy(),
                       gs.n_attached.copy(), gs.spread_fraction.copy(),
                       gs.fa_maturity.copy(), gs.n_overcrowded.copy(),
                       gs.a.copy(), gs.b.copy(), gs.n_shape.copy(),
                       gs.theta.copy()))
        return m

    # Initial save (t=0, no cell attachment yet)
    update_cell_state(gs, p, 0.0)
    F0, _ = compute_forces(gs, p, rng)
    m = save(0.0, F0)
    print(f"\n  {'t(h)':>6} {'f_cl':>5} {'f_lf':>6} {'v_cl':>5} "
          f"{'tissue':>7} {'bridges':>7} {'attach':>7} {'spread':>6} "
          f"{'FA_mat':>6} {'disp_f':>7}")
    print(f"  {0:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} {m['void_nc']:5d} "
          f"{m['tissue_frac']:7.3f} {m['n_bridges']:7d} "
          f"{m['n_attached_total']:7.0f} {m['mean_spread_frac']:6.2f} "
          f"{m['mean_fa_maturity']:6.2f} {m['disp_func']:7.1f}")

    wall_t0 = timer.time()
    t = 0.0
    for s in range(1, n_steps + 1):
        t += p.dt
        F = step(gs, p, rng, t)

        if s % p.save_every == 0:
            m = save(t, F)
            print(f"  {t:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} "
                  f"{m['void_nc']:5d} {m['tissue_frac']:7.3f} "
                  f"{m['n_bridges']:7d} {m['n_attached_total']:7.0f} "
                  f"{m['mean_spread_frac']:6.2f} {m['mean_fa_maturity']:6.2f} "
                  f"{m['disp_func']:7.1f}")

    elapsed = timer.time() - wall_t0
    print(f"\n  Done in {elapsed:.1f}s ({n_steps} steps, {gs.N} granules)")
    return hist, snaps, p, gs


# ══════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════

def plot_granules(snaps, hist, p, indices=None):
    """Plot granule positions as circles or superellipses at selected times."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4*nc, 4))
    if nc == 1: axes = [axes]

    for c, si in enumerate(indices):
        snap = snaps[si]
        pf, pi, pv, xs, ys, rs, gt = snap[:7]
        # Shape arrays (V1.3) — default to circles if not present
        if len(snap) > 11:
            a_arr, b_arr, ns_arr, th_arr = snap[11], snap[12], snap[13], snap[14]
            has_shape = True
        else:
            has_shape = False

        ax = axes[c]; ax.set_xlim(0, p.Lx); ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

        # Draw granules
        for i in range(len(xs)):
            color = 'orangered' if gt[i] == 0 else 'steelblue'
            alpha = 0.75 if gt[i] == 0 else 0.55

            if has_shape and not (a_arr[i] == b_arr[i] and ns_arr[i] == 2.0):
                # Superellipse patch
                verts = superellipse_polygon_pts(
                    xs[i], ys[i], a_arr[i], b_arr[i], ns_arr[i], th_arr[i])
                patch = Polygon(verts, closed=True, fc=color, ec='k',
                                lw=0.3, alpha=alpha)
            else:
                patch = Circle((xs[i], ys[i]), rs[i], fc=color, ec='k',
                               lw=0.3, alpha=alpha)
            ax.add_patch(patch)

        ax.set_title(f"t = {hist[si]['time']:.1f} h", fontsize=10)
        if c == 0:
            ax.plot([], [], 'o', color='orangered', ms=8, label='Functional')
            ax.plot([], [], 'o', color='steelblue', ms=8, label='Inert')
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Granule Positions Over Time', fontsize=13, y=1.02)
    plt.tight_layout(); return fig


def plot_fields(snaps, hist, p, indices=None):
    """Phase fields rendered from granule positions."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    ext = [0, p.Lx, 0, p.Ly]
    fig, ax = plt.subplots(3, nc, figsize=(3.2*nc, 9))
    lbl = [r'$\phi_f$', r'$\phi_i$', r'$\phi_v$']
    cm = ['Oranges', 'Blues', 'Greens']

    for c, si in enumerate(indices):
        pf, pi, pv = snaps[si][0], snaps[si][1], snaps[si][2]
        flds = [pf, pi, pv]
        t = hist[si]['time']
        for r in range(3):
            a = ax[r, c]; vx = max(.01, flds[r].max()*1.05)
            im = a.imshow(flds[r].T, origin='lower', extent=ext,
                          cmap=cm[r], vmin=0, vmax=vx)
            a.set_title(f't={t:.1f}h', fontsize=8)
            if c == 0: a.set_ylabel(lbl[r], fontsize=11)
            plt.colorbar(im, ax=a, fraction=.046, pad=.04)
    fig.suptitle('Rendered Phase Fields', fontsize=13, y=1.01)
    plt.tight_layout(); return fig


def plot_timeseries(hist, p):
    t = [h['time'] for h in hist]
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))

    # (0,0) Cluster counts
    ax[0,0].plot(t, [h['func_nc'] for h in hist], 'C1-o', ms=3, label='Functional')
    ax[0,0].plot(t, [h['void_nc'] for h in hist], 'C2-s', ms=3, label='Void')
    ax[0,0].plot(t, [h['inert_nc'] for h in hist], 'C0-^', ms=3, label='Inert')
    ax[0,0].set(xlabel='time (h)', ylabel='# clusters',
                title='Cluster Count')
    ax[0,0].legend()

    # (0,1) Largest cluster fraction
    ax[0,1].plot(t, [h['func_lf'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[0,1].plot(t, [h['void_lf'] for h in hist], 'C2--', label='Void')
    ax[0,1].axhline(1, ls=':', c='gray', lw=0.8)
    ax[0,1].set(xlabel='time (h)', ylabel='largest / total',
                title='Connectivity (1 = percolated)')
    ax[0,1].legend()

    # (0,2) Bridge count
    ax[0,2].plot(t, [h['n_bridges'] for h in hist], 'C3-', lw=2)
    ax[0,2].set(xlabel='time (h)', ylabel='# bridges',
                title='Active Cell Bridges')

    # (1,0) Displacement
    ax[1,0].plot(t, [h['disp_func'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[1,0].plot(t, [h['disp_inert'] for h in hist], 'C0--', label='Inert')
    ax[1,0].set(xlabel='time (h)', ylabel='mean disp (µm)',
                title='Granule Displacement')
    ax[1,0].legend()

    # (1,1) Tissue & packing
    ax[1,1].plot(t, [h['tissue_frac'] for h in hist], 'C3-', lw=2,
                 label=r'Tissue ($\phi_f > 0.5$)')
    ax[1,1].plot(t, [h['packing_func_rich'] for h in hist], 'C1--',
                 label='Packing in func-rich')
    ax[1,1].set(xlabel='time (h)', ylabel='fraction', title='Tissue Remodeling')
    ax[1,1].legend()

    # (1,2) Max cluster area
    ax[1,2].plot(t, [h['func_max_area'] for h in hist], 'C1-', lw=2)
    ax[1,2].set(xlabel='time (h)', ylabel='area (µm²)',
                title='Largest Functional Cluster Area')

    # (2,0) Cell attachment
    ax[2,0].plot(t, [h['n_attached_total'] for h in hist], 'C4-', lw=2,
                 label='Attached')
    ax[2,0].plot(t, [h['n_seeded_total'] for h in hist], 'C7--', lw=1,
                 label='Seeded')
    ax[2,0].axvline(p.t_attach_onset, ls=':', c='gray', lw=0.8, label='Attach onset')
    ax[2,0].set(xlabel='time (h)', ylabel='# cells',
                title='Cell Attachment')
    ax[2,0].legend()

    # (2,1) Spread fraction & FA maturity
    ax[2,1].plot(t, [h['mean_spread_frac'] for h in hist], 'C5-', lw=2,
                 label='Spread fraction')
    ax[2,1].plot(t, [h['mean_fa_maturity'] for h in hist], 'C6--', lw=2,
                 label='FA maturity')
    ax[2,1].set(xlabel='time (h)', ylabel='fraction [0–1]',
                title='Cell Spreading & Focal Adhesion')
    ax[2,1].set_ylim(-0.05, 1.05)
    ax[2,1].legend()

    # (2,2) Overcrowding
    ax[2,2].plot(t, [h['n_overcrowded_total'] for h in hist], 'C3-', lw=2)
    ax[2,2].set(xlabel='time (h)', ylabel='# cells',
                title='Overcrowded Cells (crawling on others)')

    plt.tight_layout(); return fig


def plot_composite(snaps, hist, p, indices=None):
    """RGB composite from rendered fields."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(3.8*nc, 3.5))
    if nc == 1: axes = [axes]
    for c, si in enumerate(indices):
        pf, pi, pv = snaps[si][0], snaps[si][1], snaps[si][2]
        mx = max(pf.max(), pi.max(), pv.max(), 0.01)
        rgb = np.stack([pf/mx, pv/mx, pi/mx], axis=-1)
        axes[c].imshow(np.clip(np.transpose(rgb,(1,0,2)),0,1),
                       origin='lower', extent=[0,p.Lx,0,p.Ly])
        axes[c].set_title(f"t={hist[si]['time']:.1f}h", fontsize=9)
    fig.suptitle("R=functional  G=void  B=inert", fontsize=11, y=1.02)
    plt.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  Overdamped Particle Dynamics: Cell-Driven Granular Rearrangement")
    print("="*65)

    p = Params()
    print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm")
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

    fig1 = plot_granules(snaps, hist, p)
    fig2 = plot_fields(snaps, hist, p)
    fig3 = plot_timeseries(hist, p)
    fig4 = plot_composite(snaps, hist, p)
    plt.show()

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