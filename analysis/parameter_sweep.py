#!/usr/bin/env python3
"""
Mean-Field Parameter Sweep & Organ Target Prediction (V1.9)
============================================================
Generates a rich dataset by Latin Hypercube Sampling across 12 physical
parameters, solved via vectorised overdamped Euler integration of the
mean-field compaction ODE.  Predicts optimal scaffold parameters for
replicating 7 organ tissue architectures.

11 swept parameters
-------------------
    R_func, R_inert          -- granule radii (µm)
    phi_solid, func_ratio    -- total solid fraction + functional share
    E_func, E_inert          -- Young's moduli (kPa)
    n_cells_per_func         -- cells per functional granule
    aspect_ratio_func/inert  -- a/b semi-axis ratio
    blockiness_n2_func/inert -- superellipsoid exponent n2

Fixed parameters: cell_sense_distance = 50 µm

Physics extensions over single-E model
--------------------------------------
    - Motor-clutch traction uses E_func (substrate cells sit on).
    - Effective contact modulus E* from Hertzian mixing of f-f, f-i, i-i.
    - phi_RCP depends on aspect ratios, blockiness, and bidisperse size ratio.
    - Bridging rate modulated by cell_sense_distance vs mean packing gap.
    - Kozeny-Carman permeability uses Sauter mean grain diameter.

Usage:
    python analysis/parameter_sweep.py                       # 200K samples
    python analysis/parameter_sweep.py --quick               # 20K samples
    python analysis/parameter_sweep.py -n 500000 -o results/big_sweep

Units: micrometres, nanonewtons, hours, kPa.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import json, csv, os
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from analysis.organ_targets import (
    ORGAN_TARGETS, get_target, list_organs,
)
from analysis.arch_distance import (
    architectural_distance, LOG_TRANSFORM_KEYS,
)

# ======================================================================
# Parameter space definition
# ======================================================================

PARAM_DEFS = OrderedDict([
    ('R_func',              {'lo': 5,    'hi': 150,  'log': True,  'int': False, 'unit': 'µm'}),
    ('R_inert',             {'lo': 5,    'hi': 300,  'log': True,  'int': False, 'unit': 'µm'}),
    ('phi_solid',           {'lo': 0.75, 'hi': 0.95, 'log': False, 'int': False, 'unit': ''}),
    ('func_ratio',          {'lo': 0.05, 'hi': 0.95, 'log': False, 'int': False, 'unit': ''}),
    ('E_func',              {'lo': 0.5,  'hi': 200,  'log': True,  'int': False, 'unit': 'kPa'}),
    ('E_inert',             {'lo': 0.5,  'hi': 200,  'log': True,  'int': False, 'unit': 'kPa'}),
    ('n_cells_per_func',    {'lo': 1,    'hi': 30,   'log': False, 'int': True,  'unit': ''}),
    ('aspect_ratio_func',   {'lo': 1.0,  'hi': 3.0,  'log': False, 'int': False, 'unit': ''}),
    ('aspect_ratio_inert',  {'lo': 1.0,  'hi': 3.0,  'log': False, 'int': False, 'unit': ''}),
    ('blockiness_n2_func',  {'lo': 2.0,  'hi': 6.0,  'log': False, 'int': False, 'unit': ''}),
    ('blockiness_n2_inert', {'lo': 2.0,  'hi': 6.0,  'log': False, 'int': False, 'unit': ''}),
])

PARAM_NAMES = list(PARAM_DEFS.keys())
N_PARAMS = len(PARAM_NAMES)

# Fixed parameters (not swept — added to samples dict as constants)
FIXED_PARAMS = {'cell_sense_distance': 50.0}  # µm, filopodia sensing range

# 1D PDE spatial grid parameters
N_X = 20           # radial grid points (center=0, edge=1)
D_BASE = 0.01      # base diffusion coefficient for stress-driven mixing

# Tissue volume model parameters (V2.1)
# Cells on functional granules + bridges progressively fill inter-granular void
TISSUE_PARAMS = {
    'k_tissue': 0.05,            # h^-1, tissue formation rate
    'alpha_tissue_fill': 0.6,    # max fraction of void fillable by tissue
    'alpha_tissue_0': 0.1,       # min growth rate fraction (surface cells, no bridges)
    'n_cells_tissue_ref': 10.0,  # reference cell count for normalization
}

# Derived parameter names (not in PARAM_DEFS but added to samples dict)
DERIVED_NAMES = ['phi_f', 'phi_i']

N_SAMPLES_DEFAULT = 200_000
N_SAMPLES_QUICK   = 20_000

# Pretty labels for plots
PARAM_LABELS = {
    'R_func': '$R_{func}$ (µm)',
    'R_inert': '$R_{inert}$ (µm)',
    'phi_solid': r'$\phi_{solid}^0$',
    'func_ratio': r'$f_{func}$',
    'phi_f': r'$\phi_f^0$',
    'phi_i': r'$\phi_i^0$',
    'E_func': '$E_{func}$ (kPa)',
    'E_inert': '$E_{inert}$ (kPa)',
    'n_cells_per_func': 'Cells/granule',
    'aspect_ratio_func': 'AR$_{func}$',
    'aspect_ratio_inert': 'AR$_{inert}$',
    'blockiness_n2_func': '$n_2^{func}$',
    'blockiness_n2_inert': '$n_2^{inert}$',
    'cell_sense_distance': 'Sense dist (µm)',  # fixed at 50 µm
}


# ======================================================================
# Latin Hypercube Sampling
# ======================================================================

def generate_lhs(n_samples, seed=42):
    """Generate LHS samples in physical space, respecting log scales and types.

    phi_solid ∈ [0.75, 0.95] covers the jammed regime (phi_solid >= phi_RCP).
    Lower bound 0.75 is slightly below the minimum phi_RCP (~0.82 for
    spheres) to capture edge cases with favorable shape corrections.
    func_ratio ∈ [0.05, 0.95] ensures both granule types are present.
    Derived: phi_f = phi_solid * func_ratio,
    phi_i = phi_solid * (1 - func_ratio).  cell_sense_distance is fixed at 50 µm.

    Returns dict of 1-D arrays, one per parameter (plus fixed + derived).
    """
    rng = np.random.default_rng(seed)

    # Uniform samples in [0, 1]^d  (LHS)
    d = N_PARAMS
    # Each dimension gets a random permutation of n strata
    U = np.zeros((n_samples, d))
    for j in range(d):
        perm = rng.permutation(n_samples)
        U[:, j] = (perm + rng.uniform(size=n_samples)) / n_samples

    # Map to physical ranges
    samples = {}
    for j, name in enumerate(PARAM_NAMES):
        info = PARAM_DEFS[name]
        lo, hi = info['lo'], info['hi']
        if info['log']:
            log_lo, log_hi = np.log10(max(lo, 1e-30)), np.log10(max(hi, 1e-30))
            vals = 10.0 ** (log_lo + U[:, j] * (log_hi - log_lo))
        else:
            vals = lo + U[:, j] * (hi - lo)
        if info['int']:
            vals = np.round(vals).astype(int)
        samples[name] = vals

    # Fixed parameters (constant across all samples)
    for name, val in FIXED_PARAMS.items():
        samples[name] = np.full(n_samples, val)

    # Derive phi_f and phi_i from phi_solid and func_ratio
    samples['phi_f'] = samples['phi_solid'] * samples['func_ratio']
    samples['phi_i'] = samples['phi_solid'] * (1.0 - samples['func_ratio'])

    return samples


# ======================================================================
# Vectorised physics
# ======================================================================

def compute_phi_RCP(ar_func, ar_inert, n2_func, n2_inert,
                    phi_f, phi_i, R_func, R_inert):
    """Estimate 2-D random close packing fraction for bidisperse shaped granules.

    Base (2-D circles) ~ 0.82.
    Corrections:
        - Aspect ratio: +0.04*(AR-1) - 0.015*(AR-1)^2  (peaks ~AR 1.3)
        - Blockiness:   +0.015*(n2-2)                   (squarer = denser)
        - Bidisperse:   +0.18*x_s*(1-x_s)*(1-1/r)      (Farr & Groot 2009)
    """
    base = 0.82

    def _ar_eff(ar):
        d = ar - 1.0
        return 0.04 * d - 0.015 * d * d

    def _block_eff(n2):
        return 0.015 * (n2 - 2.0)

    phi_total = phi_f + phi_i
    w_f = np.where(phi_total > 0, phi_f / np.maximum(phi_total, 1e-30), 1.0)
    w_i = 1.0 - w_f

    shape = (w_f * (_ar_eff(ar_func) + _block_eff(n2_func)) +
             w_i * (_ar_eff(ar_inert) + _block_eff(n2_inert)))

    # Bidisperse size-ratio effect
    R_big   = np.maximum(R_func, R_inert)
    R_small = np.minimum(R_func, R_inert)
    r_ratio = R_big / np.maximum(R_small, 1.0)

    N_f = phi_f / np.maximum(np.pi * R_func ** 2, 1.0)
    N_i = phi_i / np.maximum(np.pi * R_inert ** 2, 1e-30)
    N_tot = N_f + N_i
    x_s = np.where(N_tot > 0,
                    np.minimum(N_f, N_i) / np.maximum(N_tot, 1e-30),
                    0.0)
    bidisperse = 0.18 * x_s * (1.0 - x_s) * (1.0 - 1.0 / np.maximum(r_ratio, 1.01))
    # Only when both phases present
    bidisperse = np.where(phi_i > 0.005, bidisperse, 0.0)

    return np.clip(base + shape + bidisperse, 0.60, 0.95)


def compute_phi_max_deformable(phi_RCP, E_func, n2_func):
    """Maximum local packing fraction for deformable superellipsoid granules.

    Rigid-particle RCP is the *onset* of contact resistance, not a hard wall.
    Deformable granules can pack beyond phi_RCP because:
      - Soft granules flatten at contacts → overlap δ reduces effective spacing.
      - Blockier shapes (n2 > 2) have flatter faces → tighter interlocking.

    The resistance function sigma_resist still grows continuously above phi_RCP,
    so this limit is rarely reached — it prevents unrealistic hard-clamping.

    Parameters
    ----------
    phi_RCP : array
        Rigid random close packing fraction.
    E_func : array
        Functional granule Young's modulus (kPa).
    n2_func : array
        Functional granule superellipsoid exponent n2 (2 = sphere, >2 = blockier).

    Returns
    -------
    array
        Maximum deformable packing fraction (phi_RCP < phi_max <= 0.99).
    """
    # Compliance: 0 for infinitely stiff, approaches 1 for infinitely soft
    E_ref = 10.0  # kPa — reference stiffness where compliance = 0.5
    compliance = E_ref / (E_func + E_ref)

    # Shape factor: blockier shapes allow tighter packing when deformed
    # Spheres (n2=2): shape_boost = 1.0; cubes (n2→∞): shape_boost → higher
    shape_boost = 1.0 + 0.15 * np.maximum(0.0, n2_func - 2.0)

    # Extra packing beyond RCP: up to ~50% of remaining space for very soft+blocky
    phi_deform_extra = (1.0 - phi_RCP) * 0.5 * compliance * shape_boost
    phi_max = phi_RCP + phi_deform_extra

    return np.clip(phi_max, phi_RCP, 0.99)


def motor_clutch_force_vec(E_kPa, n_motors=50, F_motor_stall=0.5,
                           F_max_per_cell=50.0):
    """Vectorised motor-clutch steady-state traction per cell (nN)."""
    a_cell   = 10.0     # cell_diameter / 2
    poisson  = 0.45
    F_stall  = n_motors * F_motor_stall
    k_opt    = 375.0    # n_clutches * k_clutch = 75 * 5
    engage   = 0.909    # k_on / (k_on + k_off) = 1 / 1.1

    k_sub = np.pi * E_kPa * a_cell / (1.0 - poisson ** 2)
    beta  = k_sub / (k_sub + k_opt)
    F_mc  = F_stall * beta * engage
    return np.minimum(F_mc, F_max_per_cell)


def effective_contact_modulus(E_func, E_inert, phi_f, phi_i):
    """Weighted-average Hertzian reduced modulus across f-f, f-i, i-i contacts."""
    nu = 0.45
    denom = 1.0 - nu * nu

    E_ff = E_func  / (2.0 * denom)
    E_ii = E_inert / (2.0 * denom)
    E_fi = E_func * E_inert / (denom * (E_func + E_inert + 1e-30))

    phi_tot = np.maximum(phi_f + phi_i, 1e-30)
    wf  = phi_f / phi_tot
    wi  = phi_i / phi_tot
    return wf * wf * E_ff + wi * wi * E_ii + 2.0 * wf * wi * E_fi


def sauter_mean_diameter(R_func, R_inert, phi_f, phi_i):
    """Volume-weighted mean grain diameter for Kozeny-Carman."""
    phi_tot = np.maximum(phi_f + phi_i, 1e-30)
    return 2.0 * (phi_f * R_func + phi_i * R_inert) / phi_tot


def bridge_rate_modifier(cell_sense_dist, R_func, phi_f, phi_i, phi_RCP):
    """Fraction of cells whose filopodia can reach a neighbour."""
    phi_solid = phi_f + phi_i
    ratio = phi_RCP / np.maximum(phi_solid, 0.01)
    mean_gap = R_func * (np.sqrt(np.maximum(ratio, 1.0)) - 1.0)
    return np.minimum(1.0, cell_sense_dist / np.maximum(mean_gap + 1.0, 1.0))


# ======================================================================
# Vectorised Euler ODE integration
# ======================================================================

def run_sweep_vectorised(samples, t_total=72.0, dt=0.5,
                         n_motors=50, F_motor_stall=0.5,
                         F_max_per_cell=50.0,
                         bridge_lock_force_threshold=None,
                         bridge_secondary_rate_mult=1.0,
                         bridge_senescence_time=24.0):
    """Solve 1D radial PDE compaction model for all samples (vectorised).

    Physics:
      - phi_f + phi_i = phi_solid = CONSTANT (granules are incompressible).
      - State: x_f(xi, t) on N_x radial grid points, xi in [0, 1] (center=0, edge=1).
      - Neighbor-count modifier: cells at center sense all neighbors,
        cells at edge sense fewer → bridge formation is spatially varying.
      - Stress-driven diffusion smooths gradients (prevents sharp interfaces).
      - BCs: zero-flux at xi=0 (symmetry) and xi=1 (edge).
      - Domain-averaged outputs are backward compatible with scalar ODE.

    Returns dict of 1-D result arrays (domain-averaged) plus spatial stats.
    """
    n = len(samples['phi_f'])
    n_steps = int(t_total / dt)
    N_x = N_X  # radial grid points

    # Radial grid: center=0, edge=1
    xi = np.linspace(0, 1, N_x)     # (N_x,)
    dxi = xi[1] - xi[0]             # grid spacing
    # Neighbor-count modifier: full neighbors at center, none at extreme edge
    neighbor_factor = 0.5 * (1.0 + np.cos(np.pi * xi))  # (N_x,)

    # Unpack
    phi_f    = samples['phi_f'].copy()
    phi_i    = samples['phi_i'].copy()
    phi_solid = phi_f + phi_i
    R_func   = samples['R_func']
    R_inert  = samples['R_inert']
    E_func   = samples['E_func']
    E_inert  = samples['E_inert']
    n_cells  = samples['n_cells_per_func'].astype(float)
    ar_f     = samples['aspect_ratio_func']
    ar_i     = samples['aspect_ratio_inert']
    n2_f     = samples['blockiness_n2_func']
    n2_i     = samples['blockiness_n2_inert']
    sense    = samples['cell_sense_distance']

    # phi_RCP for the functional zone (monodisperse functional granules)
    phi_RCP = compute_phi_RCP(ar_f, ar_i, n2_f, n2_i, phi_f, phi_i, R_func, R_inert)

    # Deformable packing limit: soft/blocky granules can pack beyond rigid RCP
    phi_max = compute_phi_max_deformable(phi_RCP, E_func, n2_f)

    # Motor-clutch force (cells sense E_func)
    F_cell = motor_clutch_force_vec(E_func, n_motors=n_motors,
                                     F_motor_stall=F_motor_stall,
                                     F_max_per_cell=F_max_per_cell)

    # Cell number density (2-D, 800x800 domain)
    domain_area = 800.0 * 800.0
    N_func = np.maximum(1.0, phi_f * domain_area / (np.pi * R_func ** 2))
    n_cell_density = N_func * n_cells / domain_area

    # Effective modulus for jamming resistance (functional zone contacts)
    E_eff = effective_contact_modulus(E_func, E_inert, phi_f, phi_i)

    # Fitting-parameter scaling
    eta_eff = 0.5 * (E_eff / 5.0) ** 0.3 * (R_func / 40.0) ** 1.5
    eta_eff = np.clip(eta_eff, 0.01, 100.0)
    sigma_0 = 0.005 * E_eff
    sigma_0 = np.maximum(sigma_0, 1e-6)

    # Bridge rate modifier from cell_sense_distance
    br_mod = bridge_rate_modifier(sense, R_func, phi_f, phi_i, phi_RCP)
    base_bridge_rate = 0.3

    # Bridge lock-in
    lock_in = np.zeros(n, dtype=bool)
    if bridge_lock_force_threshold is not None:
        lock_in = F_cell >= bridge_lock_force_threshold

    # Diffusion coefficient: D_eff = D_base * (R/40)^2 * (sigma_0/eta_eff)
    D_eff = D_BASE * (R_func / 40.0) ** 2 * sigma_0 / np.maximum(eta_eff, 1e-6)
    # CFL stability clamp: D_eff * dt / dxi^2 < 0.5
    D_max = 0.45 * dxi ** 2 / dt
    D_eff = np.minimum(D_eff, D_max)

    # ---- PDE integration (1D radial, Euler) ----
    t_spread = 3.0
    fa_rate  = 0.3
    bridge_formation_time = 2.0
    t_bridge_start = t_spread + 1.0

    # Initial condition: uniformly mixed across all radial positions
    x_f_0_scalar = np.where(phi_solid > 0, phi_f / phi_solid, 0.5)
    x_f_min = phi_f / np.maximum(phi_max, 0.01)
    x_f_0_scalar = np.maximum(x_f_0_scalar, x_f_min)

    # Expand to spatial grid: (n, N_x)
    x_f = np.broadcast_to(x_f_0_scalar[:, np.newaxis], (n, N_x)).copy()

    # Tissue volume fraction: (n, N_x), starts at zero (V2.1)
    phi_tissue = np.zeros((n, N_x))
    n_cells_tissue_norm = np.minimum(1.0, n_cells / TISSUE_PARAMS['n_cells_tissue_ref'])

    # Pre-broadcast per-sample arrays for spatial operations: (n, 1)
    phi_f_b   = phi_f[:, np.newaxis]
    phi_i_b   = phi_i[:, np.newaxis]
    phi_RCP_b = phi_RCP[:, np.newaxis]
    eta_eff_b = eta_eff[:, np.newaxis]
    sigma_0_b = sigma_0[:, np.newaxis]
    x_f_min_b = x_f_min[:, np.newaxis]
    D_eff_b   = D_eff[:, np.newaxis]
    n_cells_tissue_b = n_cells_tissue_norm[:, np.newaxis]
    # neighbor_factor: (N_x,) → broadcasts with (n, N_x)

    for step in range(n_steps):
        t = step * dt

        # FA maturity (scalar in time)
        t_eff = max(0.0, t - t_spread)
        maturity = 1.0 - np.exp(-fa_rate * t_eff)

        # Bridge fraction (per-sample, then modulated spatially)
        t_br_eff = max(0.0, t - t_spread - 1.0)
        if maturity < 0.1:
            f_bridge = np.zeros(n)
        else:
            eff_rate = base_bridge_rate * br_mod
            f_base = 1.0 - np.exp(-eff_rate * t_br_eff * maturity)

            if bridge_secondary_rate_mult > 1.0:
                eff_rate_boosted = eff_rate * bridge_secondary_rate_mult
                f_boosted = 1.0 - np.exp(-eff_rate_boosted * t_br_eff * maturity)
                f_bridge = np.where(f_base > 0.05,
                                    f_base + (1.0 - f_base) * (f_boosted - f_base),
                                    f_base)
            else:
                f_bridge = f_base

            if bridge_senescence_time > 0 and t_br_eff > 0:
                tau_s = bridge_senescence_time
                turnover = tau_s / (tau_s + t_br_eff)
                f_bridge = np.where(lock_in, f_bridge, f_bridge * turnover)

        # Bridge ramp
        t_since = max(0.0, t - t_bridge_start)
        ramp = min(1.0, t_since / bridge_formation_time) if bridge_formation_time > 0 else 1.0

        # Spatially-varying cell contractile stress: (n, N_x)
        # f_bridge is (n,), multiply by neighbor_factor (N_x,) → (n, N_x)
        s_cell_spatial = (n_cell_density * F_cell * f_bridge * maturity * ramp
                          )[:, np.newaxis] * neighbor_factor[np.newaxis, :]

        # Resistance: grows as local packing approaches RCP — (n, N_x)
        phi_f_local = phi_f_b / np.maximum(x_f, 1e-6)
        ratio = phi_f_local / phi_RCP_b
        s_resist = np.where(ratio > 1.0, sigma_0_b * (ratio - 1.0), 0.0)

        # Net compaction rate: (n, N_x)
        net = np.maximum(0.0, s_cell_spatial - s_resist)
        dx_f_compact = -x_f * net / eta_eff_b

        # Diffusion term: D * d²x_f/dxi² with zero-flux BCs — (n, N_x)
        # Laplacian via central differences
        laplacian = np.zeros_like(x_f)
        # Interior points
        laplacian[:, 1:-1] = (x_f[:, 2:] - 2.0 * x_f[:, 1:-1] + x_f[:, :-2]) / (dxi ** 2)
        # Zero-flux BCs: ghost points mirror interior
        # At xi=0: x_f[-1] = x_f[1] → laplacian[0] = 2*(x_f[1] - x_f[0])/dxi^2
        laplacian[:, 0] = 2.0 * (x_f[:, 1] - x_f[:, 0]) / (dxi ** 2)
        # At xi=1: x_f[N_x] = x_f[N_x-2] → laplacian[-1] = 2*(x_f[-2] - x_f[-1])/dxi^2
        laplacian[:, -1] = 2.0 * (x_f[:, -2] - x_f[:, -1]) / (dxi ** 2)

        dx_f_diffuse = D_eff_b * laplacian

        # Update
        x_f = x_f + dt * (dx_f_compact + dx_f_diffuse)

        # Clamp per grid point
        x_f = np.maximum(x_f, x_f_min_b)
        x_f = np.minimum(x_f, 1.0 - phi_i_b)

        # ---- Tissue volume evolution (V2.1) ----
        # Available void in functional zone at each grid point
        phi_void_func = np.maximum(0.0, 1.0 - phi_f_b / np.maximum(x_f, 1e-6))
        phi_tissue_max = TISSUE_PARAMS['alpha_tissue_fill'] * phi_void_func

        # Spatially-varying bridge effect: max(alpha_0, f_bridge * neighbor_factor)
        f_bridge_spatial = f_bridge[:, np.newaxis] * neighbor_factor[np.newaxis, :]
        growth_driver = np.maximum(TISSUE_PARAMS['alpha_tissue_0'], f_bridge_spatial)

        # Logistic growth ODE
        d_phi_tissue = (TISSUE_PARAMS['k_tissue']
                        * n_cells_tissue_b
                        * maturity
                        * growth_driver
                        * np.maximum(0.0, phi_tissue_max - phi_tissue))
        phi_tissue = phi_tissue + dt * d_phi_tissue
        phi_tissue = np.clip(phi_tissue, 0.0, phi_tissue_max)

    # ---- Domain-averaged outputs (backward compatible) ----
    x_f_avg = np.mean(x_f, axis=1)  # (n,)
    x_f_0_avg = x_f_0_scalar

    delta_x_f = x_f_avg - x_f_0_avg
    compaction_ratio = np.where(x_f_0_avg > 0, -delta_x_f / x_f_0_avg, 0.0)

    # Domain-averaged local void fractions
    phi_v_f_local = np.maximum(0.0, 1.0 - phi_f / np.maximum(x_f_avg, 1e-6))
    phi_v_i_local = np.maximum(0.0, 1.0 - phi_i / np.maximum(1.0 - x_f_avg, 1e-6))
    phi_v_global  = 1.0 - phi_solid

    # Effective permeability (parallel zones weighted by volume fraction)
    d_f = 2.0 * R_func
    d_i = 2.0 * R_inert
    eps_f = np.clip(phi_v_f_local, 0.01, 0.99)
    eps_i = np.clip(phi_v_i_local, 0.01, 0.99)
    K_f = eps_f ** 3 * d_f ** 2 / (180.0 * (1.0 - eps_f) ** 2)
    K_i = eps_i ** 3 * d_i ** 2 / (180.0 * (1.0 - eps_i) ** 2)
    K_eff = x_f_avg * K_f + (1.0 - x_f_avg) * K_i

    # Series (radial) permeability — harmonic mean across radial positions
    phi_v_f_spatial = np.maximum(0.0, 1.0 - phi_f_b / np.maximum(x_f, 1e-6))
    eps_f_spatial = np.clip(phi_v_f_spatial, 0.01, 0.99)
    K_f_spatial = eps_f_spatial ** 3 * d_f[:, np.newaxis] ** 2 / (
        180.0 * (1.0 - eps_f_spatial) ** 2)
    # Harmonic mean for series flow
    K_f_series = N_x / np.sum(1.0 / np.maximum(K_f_spatial, 1e-12), axis=1)

    # Effective pore sizes per zone
    r_pore_f = d_f * eps_f / (3.0 * np.maximum(1.0 - eps_f, 0.01))
    r_pore_i = d_i * eps_i / (3.0 * np.maximum(1.0 - eps_i, 0.01))
    mean_pore_radius = x_f_avg * r_pore_f + (1.0 - x_f_avg) * r_pore_i

    # Sauter mean diameter (unchanged — based on global composition)
    d_grain = sauter_mean_diameter(R_func, R_inert, phi_f, phi_i)

    # ---- Spatial heterogeneity stats ----
    x_f_final_std = np.std(x_f, axis=1)              # radial variation
    x_f_gradient = x_f[:, -1] - x_f[:, 0]            # edge - center (>0 = less compacted at edge)

    # Spatial void fraction variability
    phi_v_f_local_std = np.std(phi_v_f_spatial, axis=1)

    # ---- Tissue-corrected descriptors (V2.1) ----
    phi_tissue_avg = np.mean(phi_tissue, axis=1)           # (n,)
    phi_tissue_global = x_f_avg * phi_tissue_avg           # (n,)

    BV_TV_eff    = phi_f + phi_tissue_global               # granules + tissue
    porosity_eff = np.maximum(0.0, 1.0 - BV_TV_eff)

    # Tissue-corrected void and permeability in functional zone
    eps_f_tissue = np.clip(phi_v_f_local - phi_tissue_avg, 0.01, 0.99)
    K_f_tissue = eps_f_tissue ** 3 * d_f ** 2 / (180.0 * (1.0 - eps_f_tissue) ** 2)
    K_eff_tissue = x_f_avg * K_f_tissue + (1.0 - x_f_avg) * K_i

    # Tissue-corrected pore radius in functional zone
    r_pore_f_tissue = d_f * eps_f_tissue / (3.0 * np.maximum(1.0 - eps_f_tissue, 0.01))
    mean_pore_radius_tissue = x_f_avg * r_pore_f_tissue + (1.0 - x_f_avg) * r_pore_i

    # Spatial tissue stats
    phi_tissue_std = np.std(phi_tissue, axis=1)
    phi_tissue_gradient = phi_tissue[:, 0] - phi_tissue[:, -1]  # center − edge

    return {
        'x_f_final': x_f_avg,
        'x_f_0': x_f_0_avg,
        'phi_solid': phi_solid,
        'phi_v_global': phi_v_global,
        'phi_v_f_local': phi_v_f_local,
        'phi_v_i_local': phi_v_i_local,
        'delta_x_f': delta_x_f,
        'compaction_ratio': compaction_ratio,
        'K_f': K_f,
        'K_i': K_i,
        'K_permeability': K_eff,
        'K_f_series': K_f_series,
        'r_pore_f': r_pore_f,
        'r_pore_i': r_pore_i,
        'mean_pore_radius': mean_pore_radius,
        'F_cell_nN': F_cell,
        'phi_RCP': phi_RCP,
        'phi_max': phi_max,
        'E_eff': E_eff,
        'd_grain_eff': d_grain,
        'x_f_final_std': x_f_final_std,
        'x_f_gradient': x_f_gradient,
        'phi_v_f_local_std': phi_v_f_local_std,
        # Tissue volume outputs (V2.1)
        'phi_tissue_avg': phi_tissue_avg,
        'phi_tissue_global': phi_tissue_global,
        'BV_TV_eff': BV_TV_eff,
        'porosity_eff': porosity_eff,
        'K_f_tissue': K_f_tissue,
        'K_eff_tissue': K_eff_tissue,
        'r_pore_f_tissue': r_pore_f_tissue,
        'mean_pore_radius_tissue': mean_pore_radius_tissue,
        'phi_tissue_std': phi_tissue_std,
        'phi_tissue_gradient': phi_tissue_gradient,
    }


# ======================================================================
# Organ distance computation (vectorised where possible)
# ======================================================================

def compute_organ_distances(samples, outputs):
    """Compute architectural distance to each organ for every sample.

    Phase-to-tissue mapping is **organ-specific**:
      - phi_f (functional) → tissue parenchyma (cells, ECM) → BV/TV  [all organs]
      - phi_i (inert)      → structural void spaces  [organ-dependent fraction]
      - phi_v (liquid)     → perfusive channels       [organ-dependent fraction]

    Each organ defines ``perfusive_void_fraction`` (f_perf): the fraction of
    non-tissue space that is liquid-filled/perfusive vs structural void.
      - Lung  (f_perf=0.05): nearly all void is air       → phi_i dominant
      - Liver (f_perf=0.90): nearly all void is blood     → phi_v dominant
      - Bone  (f_perf=0.10): marrow cavities (structural) → phi_i dominant

    Universal descriptors (V2.1: tissue-corrected):
      BV/TV    = BV_TV_eff = phi_f + phi_tissue_global  (granules + tissue)
      Porosity = porosity_eff = 1 - BV_TV_eff
      Tb.Th    = 2*R_func
      Tb.Sp    = 2*R_inert

    Organ-specific descriptors:
      Permeability (V2.1: tissue-corrected for functional zone):
        - f_perf >= 0.5 → flow between tissue elements → K_f_tissue
        - f_perf <  0.5 → flow through structural voids → K_i (unchanged)
      Mean pore radius: same zone selection as permeability.

    Void-split penalty: additional z-score penalising scaffolds whose
    phi_i/(phi_i+phi_v) deviates from the organ's expected (1 - f_perf).

    Returns dict: 'd_<organ>' -> 1-D array.
    """
    n = len(outputs['phi_solid'])
    phi_i     = samples['phi_i']
    R_func    = samples['R_func']
    R_inert   = samples['R_inert']

    # Tissue-corrected descriptors (V2.1)
    BV_TV_eff      = outputs['BV_TV_eff']
    total_porosity = outputs['porosity_eff']

    # Per-zone permeability and pore radius (tissue-corrected for functional zone)
    K_f      = outputs['K_f_tissue']
    K_i      = outputs['K_i']
    r_pore_f = outputs['r_pore_f_tissue']
    r_pore_i = outputs['r_pore_i']

    results = {}
    closest = np.full(n, '', dtype='U30')
    closest_d = np.full(n, np.inf)

    for organ_name in ORGAN_TARGETS:
        target = ORGAN_TARGETS[organ_name]
        f_perf = target.get('perfusive_void_fraction', 0.5)

        # Select permeability & pore radius from the organ's dominant flow phase
        if f_perf >= 0.5:
            # Perfusive-dominated (liver, cardiac, kidney): flow through
            # channels between tissue elements → functional zone
            K_compare     = K_f
            r_pore_compare = r_pore_f
        else:
            # Structural-void-dominated (lung, bone, intestine, pancreas):
            # flow through the open void spaces → inert zone
            K_compare     = K_i
            r_pore_compare = r_pore_i

        # Organ's expected structural void fraction of total porosity
        target_structural_frac = 1.0 - f_perf  # expected phi_i / porosity
        # Scaffold's actual structural void fraction of total porosity
        scaffold_structural_frac = phi_i / np.maximum(total_porosity, 0.01)

        # Void-split z-score: how far off is the scaffold's phi_i/phi_v split?
        # Uncertainty ~ 0.15 (representative biological variability in void character)
        sigma_split = 0.15
        z_split = (scaffold_structural_frac - target_structural_frac) / sigma_split

        dists = np.empty(n)
        for i in range(n):
            desc = {
                'bv_tv':            float(BV_TV_eff[i]),
                'porosity':         float(total_porosity[i]),
                'permeability_KC':  float(K_compare[i]),
                'tb_th':            float(2.0 * R_func[i]),
                'tb_sp':            float(2.0 * R_inert[i]),
                'mean_pore_radius': float(r_pore_compare[i]),
            }
            d_arch = architectural_distance(desc, organ_name)
            # Combine architectural distance with void-split penalty
            dists[i] = np.sqrt(d_arch ** 2 + z_split[i] ** 2)

        results[f'd_{organ_name}'] = dists
        mask = dists < closest_d
        closest_d = np.where(mask, dists, closest_d)
        closest = np.where(mask, organ_name, closest)

    results['closest_organ'] = closest
    return results


# ======================================================================
# Full pipeline
# ======================================================================

def run_sweep(n_samples=N_SAMPLES_DEFAULT, t_total=72.0, seed=42, verbose=True,
              **sweep_kwargs):
    """Run LHS sampling + vectorised integration + organ distances.

    Extra keyword arguments are forwarded to run_sweep_vectorised
    (n_motors, F_motor_stall, F_max_per_cell, bridge_lock_force_threshold,
     bridge_secondary_rate_mult, bridge_senescence_time).

    Returns (samples, outputs, organ_dists) -- all dicts of 1-D arrays.
    """
    if verbose:
        print(f"  Generating {n_samples:,} LHS samples across {N_PARAMS} parameters ...")
    samples = generate_lhs(n_samples, seed=seed)

    # Compute phi_RCP for jamming constraint (before filtering)
    phi_RCP_pre = compute_phi_RCP(
        samples['aspect_ratio_func'], samples['aspect_ratio_inert'],
        samples['blockiness_n2_func'], samples['blockiness_n2_inert'],
        samples['phi_f'], samples['phi_i'],
        samples['R_func'], samples['R_inert'])

    # Feasibility filter:
    #   1. Need some functional phase for cells to act
    #   2. Jamming constraint: phi_solid >= phi_RCP (granules must form a
    #      jammed solid to resist gravity without external support)
    feasible = (samples['phi_f'] > 0.01) & (samples['phi_solid'] >= phi_RCP_pre)
    n_feas = int(np.sum(feasible))
    n_jammed = int(np.sum(samples['phi_solid'] >= phi_RCP_pre))
    if verbose:
        print(f"  {n_jammed:,} jammed ({100*n_jammed/n_samples:.0f}%), "
              f"{n_feas:,} feasible ({100*n_feas/n_samples:.0f}%)")
    for k in samples:
        samples[k] = samples[k][feasible]

    if verbose:
        print(f"  Integrating 1D radial PDE (t_total={t_total} h, dt=0.5 h) ...")
    outputs = run_sweep_vectorised(samples, t_total, **sweep_kwargs)

    if verbose:
        print(f"  Computing architectural distances to {len(ORGAN_TARGETS)} organs ...")
    organ_dists = compute_organ_distances(samples, outputs)

    return samples, outputs, organ_dists


def recommend_parameters(samples, outputs, organ_dists, top_n=5):
    """For each organ, return indices and info of the top_n closest samples."""
    recs = {}
    for organ in list_organs():
        dk = f'd_{organ}'
        d_arr = organ_dists[dk]
        idx = np.argsort(d_arr)[:top_n]
        entries = []
        for i in idx:
            params = {k: float(samples[k][i])
                      for k in list(PARAM_NAMES) + DERIVED_NAMES + list(FIXED_PARAMS.keys())}
            entries.append({
                'D_arch': float(d_arr[i]),
                'params': params,
                'x_f_final': float(outputs['x_f_final'][i]),
                'phi_solid': float(outputs['phi_solid'][i]),
                'phi_v_global': float(outputs['phi_v_global'][i]),
                'phi_v_f_local': float(outputs['phi_v_f_local'][i]),
                'phi_v_i_local': float(outputs['phi_v_i_local'][i]),
                'compaction_ratio': float(outputs['compaction_ratio'][i]),
                'K_permeability': float(outputs['K_permeability'][i]),
                # Tissue-corrected (V2.1)
                'phi_tissue_global': float(outputs['phi_tissue_global'][i]),
                'BV_TV_eff': float(outputs['BV_TV_eff'][i]),
                'porosity_eff': float(outputs['porosity_eff'][i]),
                'K_eff_tissue': float(outputs['K_eff_tissue'][i]),
            })
        recs[organ] = entries
    return recs


# ======================================================================
# Data saving
# ======================================================================

def _save_csv(samples, outputs, organ_dists, path, max_rows=None):
    """Save merged results to CSV."""
    keys_s = list(PARAM_NAMES) + list(FIXED_PARAMS.keys()) + DERIVED_NAMES
    keys_o = [k for k in outputs if k != 'phi_RCP' and k != 'E_eff' and k != 'd_grain_eff']
    keys_d = [k for k in organ_dists if k != 'closest_organ']
    all_keys = keys_s + keys_o + keys_d + ['closest_organ']

    n = len(samples[PARAM_NAMES[0]])
    if max_rows and n > max_rows:
        idx = np.linspace(0, n - 1, max_rows, dtype=int)
    else:
        idx = np.arange(n)

    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(all_keys)
        for i in idx:
            row = []
            for k in keys_s:
                row.append(f'{samples[k][i]:.6g}')
            for k in keys_o:
                row.append(f'{outputs[k][i]:.6g}')
            for k in keys_d:
                row.append(f'{organ_dists[k][i]:.4f}')
            row.append(organ_dists['closest_organ'][i])
            w.writerow(row)


def _save_json(obj, path):
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray):     return o.tolist()
            return super().default(o)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=_Enc)


# ======================================================================
# Plotting helpers
# ======================================================================

def _organ_colors():
    organs = list_organs()
    cmap = plt.colormaps.get_cmap('Set2').resampled(len(organs))
    return {o: cmap(i) for i, o in enumerate(organs)}


def _bin_median(x, y, n_bins=25):
    """Bin x and compute median of y in each bin.  Returns (bin_centres, medians)."""
    edges = np.linspace(np.nanmin(x), np.nanmax(x), n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    medians = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = (x >= edges[b]) & (x < edges[b + 1])
        if b == n_bins - 1:
            mask |= (x == edges[b + 1])
        if np.sum(mask) > 0:
            medians[b] = np.median(y[mask])
    return centres, medians


def _bin_2d(x, y, z, nx=30, ny=30):
    """2-D binning: returns (x_edges, y_edges, Z_median) for imshow."""
    xe = np.linspace(np.nanmin(x), np.nanmax(x), nx + 1)
    ye = np.linspace(np.nanmin(y), np.nanmax(y), ny + 1)
    Z = np.full((ny, nx), np.nan)
    for i in range(ny):
        for j in range(nx):
            mask = ((x >= xe[j]) & (x < xe[j + 1]) &
                    (y >= ye[i]) & (y < ye[i + 1]))
            if j == nx - 1: mask |= ((x == xe[j + 1]) & (y >= ye[i]) & (y < ye[i + 1]))
            if i == ny - 1: mask |= ((x >= xe[j]) & (x < xe[j + 1]) & (y == ye[i + 1]))
            if np.sum(mask) > 2:
                Z[i, j] = np.median(z[mask])
    return xe, ye, Z


# ======================================================================
# Plot 1: Marginal effects on phi_v
# ======================================================================

def plot_marginal_effects(samples, outputs, outdir):
    """One subplot per parameter: median compaction ratio vs that parameter."""
    comp = outputs['compaction_ratio']

    n_p = len(PARAM_NAMES)
    n_cols = 4
    n_rows = (n_p + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 3.5 * n_rows))
    fig.suptitle('Marginal Effect of Each Parameter on Functional Zone Compaction',
                 fontsize=14, fontweight='bold')

    for ax, pname in zip(axes.flat, PARAM_NAMES):
        x = samples[pname]
        cx, my = _bin_median(x, comp, n_bins=30)
        ax.plot(cx, my, 'C0o-', ms=4, lw=1.5)
        ax.set_xlabel(PARAM_LABELS.get(pname, pname), fontsize=9)
        ax.set_ylabel('Compaction ratio', fontsize=9)
        if PARAM_DEFS[pname]['log']:
            ax.set_xscale('log')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
    # Turn off unused axes
    for ax in axes.flat[n_p:]:
        ax.set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'marginal_effects_compaction.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 2: Compaction heatmaps (2-D binned)
# ======================================================================

def plot_compaction_heatmaps(samples, outputs, outdir):
    """2x2 heatmaps of median compaction ratio for key parameter pairs."""
    comp = outputs['compaction_ratio']
    pairs = [
        ('E_func', 'phi_solid'),
        ('phi_solid', 'func_ratio'),
        ('R_func', 'R_inert'),
        ('E_func', 'E_inert'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Functional Zone Compaction: 2-D Binned Heatmaps',
                 fontsize=14, fontweight='bold')

    phi_RCP_base = 0.82  # base RCP for monodisperse spheres

    for ax, (xk, yk) in zip(axes.flat, pairs):
        x, y = samples[xk], samples[yk]
        if PARAM_DEFS[xk]['log']:
            x = np.log10(x)
        if PARAM_DEFS[yk]['log']:
            y = np.log10(y)
        xe, ye, Z = _bin_2d(x, y, comp, nx=25, ny=25)
        im = ax.imshow(Z, origin='lower', aspect='auto', cmap='YlOrRd',
                        extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        interpolation='bilinear')
        xlab = PARAM_LABELS.get(xk, xk)
        ylab = PARAM_LABELS.get(yk, yk)
        if PARAM_DEFS[xk]['log']:
            xlab = f'log₁₀ {xlab}'
        if PARAM_DEFS[yk]['log']:
            ylab = f'log₁₀ {ylab}'
        ax.set_xlabel(xlab, fontsize=10)
        ax.set_ylabel(ylab, fontsize=10)
        plt.colorbar(im, ax=ax, label='median compaction ratio', shrink=0.85)

        # Jamming boundary annotation on phi_solid axes
        if yk == 'phi_solid':
            ax.axhline(phi_RCP_base, color='white', ls='--', lw=1.5, alpha=0.8)
            ax.text(xe[0] + 0.02 * (xe[-1] - xe[0]), phi_RCP_base + 0.005,
                    r'$\phi_{RCP}$ (spheres)', color='white', fontsize=7,
                    fontweight='bold', va='bottom')
        if xk == 'phi_solid':
            x_rcp = phi_RCP_base
            if PARAM_DEFS[xk]['log']:
                x_rcp = np.log10(phi_RCP_base)
            ax.axvline(x_rcp, color='white', ls='--', lw=1.5, alpha=0.8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'compaction_heatmaps.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 3: Solid phase balance
# ======================================================================

def plot_solid_phase_balance(samples, outputs, outdir):
    """Two-zone void redistribution: functional vs inert local void fractions."""
    phi_solid = samples['phi_solid']
    func_ratio = samples['func_ratio']
    phi_v_f = outputs['phi_v_f_local']
    phi_v_i = outputs['phi_v_i_local']
    comp = outputs['compaction_ratio']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle('Volume-Conserving Void Redistribution: Functional vs Inert Zones',
                 fontsize=13, fontweight='bold')

    slices = [(0.80, 0.84), (0.84, 0.88), (0.88, 0.92)]
    labels_s = [r'$\phi_{solid} \approx 0.82$',
                r'$\phi_{solid} \approx 0.86$',
                r'$\phi_{solid} \approx 0.90$']

    for ax, (lo, hi), lab in zip(axes, slices, labels_s):
        mask = (phi_solid >= lo) & (phi_solid < hi)
        if np.sum(mask) < 20:
            continue
        cx, mvf = _bin_median(func_ratio[mask], phi_v_f[mask], 20)
        _, mvi  = _bin_median(func_ratio[mask], phi_v_i[mask], 20)
        _, mc   = _bin_median(func_ratio[mask], comp[mask], 20)

        valid = np.isfinite(mvf)
        cx, mvf, mvi, mc = cx[valid], mvf[valid], mvi[valid], mc[valid]

        ax.plot(cx, mvf, 'C3o-', ms=4, lw=1.5, label='Void (func zone)')
        ax.plot(cx, mvi, 'C0s-', ms=4, lw=1.5, label='Void (inert zone)')
        ax2 = ax.twinx()
        ax2.plot(cx, mc, 'C2^--', ms=4, lw=1.2, alpha=0.7, label='Compaction')
        ax2.set_ylabel('Compaction ratio', fontsize=9, color='C2')
        ax2.tick_params(axis='y', labelcolor='C2')

        ax.set_title(lab, fontsize=11)
        ax.set_xlabel(r'Functional ratio $f_{func}$', fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.2)
        if ax == axes[0]:
            ax.set_ylabel('Local void fraction', fontsize=10)
            ax.legend(fontsize=7, loc='upper left')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'void_redistribution.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 4: Organ landscapes
# ======================================================================

def plot_organ_landscapes(samples, organ_dists, outdir):
    """For each organ: 2-D binned (E_func vs phi_solid) colored by median D_arch."""
    organs = list_organs()
    ncols = 4
    nrows = int(np.ceil(len(organs) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    fig.suptitle('Organ Distance Landscapes: $E_{func}$ vs $\\phi_{solid}^0$',
                 fontsize=13, fontweight='bold')

    x = np.log10(samples['E_func'])
    y = samples['phi_solid']

    phi_RCP_base = 0.82  # base RCP for monodisperse spheres

    for idx, organ in enumerate(organs):
        ax = axes.flat[idx]
        dk = f'd_{organ}'
        xe, ye, Z = _bin_2d(x, y, organ_dists[dk], nx=20, ny=20)
        im = ax.imshow(Z, origin='lower', aspect='auto', cmap='RdYlGn_r',
                        extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        interpolation='bilinear')
        ax.set_xlabel('log₁₀ $E_{func}$ (kPa)', fontsize=8)
        ax.set_ylabel(r'$\phi_{solid}^0$', fontsize=8)
        ax.set_title(organ.replace('_', ' ').title(), fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)
        plt.colorbar(im, ax=ax, label='$D_{arch}$', shrink=0.85)

        # Jamming boundary
        ax.axhline(phi_RCP_base, color='white', ls='--', lw=1.0, alpha=0.7)

    for idx in range(len(organs), len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(outdir, 'organ_landscapes.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 5: Recommendation table
# ======================================================================

def plot_recommendation_table(recs, outdir):
    """Table image of best parameters per organ."""
    organs = list_organs()
    all_param_keys = PARAM_NAMES + DERIVED_NAMES
    col_labels = ['Organ', '$D_{arch}$'] + \
                 [PARAM_LABELS.get(k, k) for k in all_param_keys] + \
                 ['Compact.', r'$\phi_v^{func}$', r'$\phi_v^{inert}$', '$K$ (µm²)',
                  r'$\phi_{tis}$', 'BV/TV$_{eff}$']
    rows = []
    for organ in organs:
        if organ not in recs or not recs[organ]:
            continue
        e = recs[organ][0]
        row = [organ.replace('_', ' ').title(), f"{e['D_arch']:.2f}"]
        for k in all_param_keys:
            v = e['params'][k]
            if k == 'n_cells_per_func':
                row.append(f'{int(round(v))}')
            elif abs(v) >= 100 or abs(v) < 0.1:
                row.append(f'{v:.2g}')
            else:
                row.append(f'{v:.2f}')
        row.append(f"{e['compaction_ratio']:.1%}")
        row.append(f"{e['phi_v_f_local']:.3f}")
        # Show inert zone void as "—" when inert fraction is negligible
        phi_i_val = e['params'].get('phi_i', 0.0)
        if phi_i_val < 0.02:
            row.append('—')
        else:
            row.append(f"{e['phi_v_i_local']:.3f}")
        row.append(f"{e['K_permeability']:.1e}")
        row.append(f"{e.get('phi_tissue_global', 0):.3f}")
        row.append(f"{e.get('BV_TV_eff', e['phi_solid']):.3f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(22, 0.5 * len(rows) + 2.0))
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=col_labels,
                      cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.5)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#d4e6f1')
        table[0, j].set_text_props(fontweight='bold', fontsize=6)
    for i in range(len(rows)):
        table[i + 1, 1].set_facecolor('#d5f5e3')
    ax.set_title('Recommended Scaffold Parameters per Organ Target',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()
    path = os.path.join(outdir, 'organ_recommendations_table.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 6: Parameter importance (Spearman |rho|)
# ======================================================================

def plot_parameter_importance(samples, organ_dists, outdir):
    """Heatmap of |Spearman rho| between each parameter and D_arch per organ."""
    from scipy.stats import spearmanr

    organs = list_organs()
    rho_matrix = np.zeros((N_PARAMS, len(organs)))

    for j, organ in enumerate(organs):
        d = organ_dists[f'd_{organ}']
        for i, pk in enumerate(PARAM_NAMES):
            x = samples[pk]
            r, _ = spearmanr(x, d)
            rho_matrix[i, j] = abs(r)

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(rho_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=0.6)
    ax.set_xticks(range(len(organs)))
    ax.set_xticklabels([o.replace('_', '\n') for o in organs], fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(N_PARAMS))
    ax.set_yticklabels([PARAM_LABELS.get(k, k) for k in PARAM_NAMES], fontsize=8)
    ax.set_title('Parameter Importance: |Spearman $\\rho$| with $D_{arch}$',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|$\\rho$|', shrink=0.8)

    for i in range(N_PARAMS):
        for j in range(len(organs)):
            ax.text(j, i, f'{rho_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=6, color='white' if rho_matrix[i,j] > 0.35 else 'black')

    plt.tight_layout()
    path = os.path.join(outdir, 'parameter_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 7: Closest organ map
# ======================================================================

def plot_closest_organ_map(outputs, organ_dists, outdir):
    """Scatter in (compaction, K_eff) coloured by closest organ."""
    fig, ax = plt.subplots(figsize=(10, 8))
    oc = _organ_colors()

    for organ in list_organs():
        mask = organ_dists['closest_organ'] == organ
        if np.sum(mask) == 0:
            continue
        n_pts = min(5000, int(np.sum(mask)))
        idx = np.where(mask)[0]
        np.random.default_rng(0).shuffle(idx)
        idx = idx[:n_pts]
        ax.scatter(outputs['compaction_ratio'][idx],
                   outputs['K_permeability'][idx],
                   c=[oc[organ]], s=5, alpha=0.3,
                   label=organ.replace('_', ' ').title())

    ax.set_xlabel('Compaction ratio', fontsize=11)
    ax.set_ylabel('$K_{eff}$ (µm²)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Compaction-Permeability Space by Closest Organ', fontsize=12)
    ax.legend(fontsize=7, loc='upper right', markerscale=4)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    path = os.path.join(outdir, 'closest_organ_map.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 8: Permeability-porosity
# ======================================================================

def plot_permeability_porosity(outputs, outdir):
    """K vs global porosity scatter with organ target bands."""
    fig, ax = plt.subplots(figsize=(10, 7))
    oc = _organ_colors()

    n_pts = min(30000, len(outputs['phi_v_global']))
    idx = np.random.default_rng(1).choice(len(outputs['phi_v_global']), n_pts, replace=False)
    ax.scatter(outputs['phi_v_global'][idx], outputs['K_permeability'][idx],
               s=3, c='grey', alpha=0.1, zorder=1)

    for organ in list_organs():
        t = get_target(organ)
        pm = t['mean'].get('porosity')
        ps = t['std'].get('porosity', 0)
        km = t['mean'].get('permeability_KC')
        ks = t['std'].get('permeability_KC', 0)
        if pm is None or km is None:
            continue
        rect = plt.Rectangle(
            (pm - ps, max(1e-2, km - ks)), 2 * ps, 2 * ks,
            facecolor=oc[organ], alpha=0.3, edgecolor=oc[organ], linewidth=2,
            label=organ.replace('_', ' ').title(), zorder=5)
        ax.add_patch(rect)
        ax.text(pm, km, organ.replace('_', '\n').title(),
                fontsize=7, fontweight='bold', ha='center', va='center', zorder=6)

    ax.set_xlabel('Porosity $\\phi_v$', fontsize=11)
    ax.set_ylabel('Kozeny-Carman $K$ (µm²)', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('Permeability-Porosity Space vs Organ Targets', fontsize=12)
    ax.legend(fontsize=7, loc='upper left')
    ax.grid(True, alpha=0.2)
    ax.set_xlim(0, 1.0)
    plt.tight_layout()
    path = os.path.join(outdir, 'permeability_porosity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 9: Radar comparison
# ======================================================================

def plot_radar_comparison(recs, outdir):
    """Spider chart: best scaffold vs organ target for each organ."""
    organs = list_organs()
    oc = _organ_colors()
    desc_keys = ['bv_tv', 'porosity', 'permeability_KC', 'tb_th', 'tb_sp', 'mean_pore_radius']
    n_desc = len(desc_keys)
    angles = np.linspace(0, 2 * np.pi, n_desc, endpoint=False).tolist()
    angles += angles[:1]
    labels = [k.replace('_', '\n') for k in desc_keys]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    all_vals = {k: [] for k in desc_keys}
    scaffold_descs = {}
    for organ in organs:
        t = get_target(organ)['mean']
        for k in desc_keys:
            v = t.get(k, 0)
            all_vals[k].append(np.log(max(v, 1e-30)) if k in LOG_TRANSFORM_KEYS and v > 0 else v)
        if organ in recs and recs[organ]:
            e = recs[organ][0]
            p = e['params']
            phi_s = e['phi_solid']
            phi_v = e['phi_v_global']
            dg = 2 * (p['phi_f'] * p['R_func'] + p['phi_i'] * p['R_inert']) / max(p['phi_f'] + p['phi_i'], 0.01)
            # Weighted mean pore radius from two-zone local voids
            d_f = 2 * p['R_func']
            d_i = 2 * p['R_inert']
            x_f = e['x_f_final']
            eps_f = max(e['phi_v_f_local'], 0.01)
            eps_i = max(e['phi_v_i_local'], 0.01)
            r_pore = x_f * d_f * eps_f / (3 * max(1 - eps_f, 0.01)) + \
                     (1 - x_f) * d_i * eps_i / (3 * max(1 - eps_i, 0.01))
            sd = {
                'bv_tv': e.get('BV_TV_eff', phi_s),
                'porosity': e.get('porosity_eff', phi_v),
                'permeability_KC': e.get('K_eff_tissue', e['K_permeability']), 'tb_th': dg,
                'tb_sp': dg * phi_v / max(phi_s, 0.01),
                'mean_pore_radius': r_pore,
            }
            scaffold_descs[organ] = sd
            for k in desc_keys:
                v = sd.get(k, 0)
                all_vals[k].append(np.log(max(v, 1e-30)) if k in LOG_TRANSFORM_KEYS and v > 0 else v)

    mins = {k: min(vs) if vs else 0 for k, vs in all_vals.items()}
    maxs = {k: max(vs) if vs else 1 for k, vs in all_vals.items()}

    def _n(k, raw):
        if k in LOG_TRANSFORM_KEYS and raw > 0:
            raw = np.log(max(raw, 1e-30))
        span = maxs[k] - mins[k]
        return (raw - mins[k]) / span if span > 0 else 0.5

    for organ in organs:
        t = get_target(organ)['mean']
        vals = [_n(k, t.get(k, 0)) for k in desc_keys] + [_n(desc_keys[0], t.get(desc_keys[0], 0))]
        ax.plot(angles, vals, '-', color=oc[organ], lw=1, alpha=0.5,
                label=organ.replace('_', ' ').title())
    for organ in organs:
        if organ not in scaffold_descs:
            continue
        sd = scaffold_descs[organ]
        vals = [_n(k, sd.get(k, 0)) for k in desc_keys] + [_n(desc_keys[0], sd.get(desc_keys[0], 0))]
        ax.plot(angles, vals, '--o', color=oc[organ], lw=2, ms=5, alpha=0.8, zorder=5)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title('Best Scaffold (dashed) vs Organ Target (solid)', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.05), fontsize=7)
    plt.tight_layout()
    path = os.path.join(outdir, 'radar_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 10: Best kinetics
# ======================================================================

def plot_best_kinetics(recs, outdir, t_total=72.0, n_motors=50,
                       F_motor_stall=0.5, F_max_per_cell=50.0):
    """Time-evolution for optimal scaffold per organ."""
    from analysis.mean_field_model import CompactionModel, MotorClutchParams
    organs = list_organs()
    oc = _organ_colors()

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Compaction & Tissue Kinetics for Optimal Scaffold per Organ',
                 fontsize=13, fontweight='bold')

    dt = 0.5
    n_steps = int(t_total / dt)
    t_arr = np.arange(n_steps + 1) * dt

    # Store spatial trajectories for radial profile plot
    spatial_data = {}

    for organ in organs:
        if organ not in recs or not recs[organ]:
            continue
        p = recs[organ][0]['params']
        (x_f_traj, phi_v_f_traj, phi_v_i_traj, x_f_spatial,
         tissue_traj, tissue_spatial) = _integrate_trajectory(
            p, t_total, dt, n_motors=n_motors,
            F_motor_stall=F_motor_stall, F_max_per_cell=F_max_per_cell)
        spatial_data[organ] = (p, x_f_spatial, tissue_spatial)

        c = oc[organ]
        lab = organ.replace('_', ' ').title()
        axes[0].plot(t_arr, x_f_traj, '-', color=c, lw=1.8, label=lab)
        axes[1].plot(t_arr, phi_v_f_traj, '-', color=c, lw=1.8, label=lab)
        axes[2].plot(t_arr, tissue_traj, '-', color=c, lw=1.8, label=lab)

    axes[0].set_xlabel('Time (h)'); axes[0].set_ylabel('$x_f$ (func zone fraction)')
    axes[0].set_title('(a) Functional Zone Compaction'); axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel('Time (h)'); axes[1].set_ylabel(r'$\phi_v^{func}$ (local void)')
    axes[1].set_title('(b) Functional Zone Local Void'); axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)
    axes[2].set_xlabel('Time (h)'); axes[2].set_ylabel(r'$\phi_{tissue}^{global}$')
    axes[2].set_title('(c) Tissue Volume Fraction'); axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'best_kinetics.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")

    # Plot radial profiles
    if spatial_data:
        plot_radial_profiles(spatial_data, outdir, oc)


def plot_radial_profiles(spatial_data, outdir, oc):
    """Plot final radial profiles of x_f(xi), phi_v(xi), and phi_tissue(xi)."""
    xi = np.linspace(0, 1, N_X)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle('Radial Profiles at t=72h (center=0, edge=1)',
                 fontsize=13, fontweight='bold')

    for organ, (p, x_f_spatial, tissue_spatial) in spatial_data.items():
        c = oc.get(organ, 'gray')
        lab = organ.replace('_', ' ').title()
        x_f_final = x_f_spatial[-1]
        phi_f = p['phi_f']
        phi_v_f = np.maximum(0.0, 1.0 - phi_f / np.maximum(x_f_final, 1e-6))
        tissue_final = tissue_spatial[-1]

        axes[0].plot(xi, x_f_final, '-', color=c, lw=1.8, label=lab)
        axes[1].plot(xi, phi_v_f, '-', color=c, lw=1.8, label=lab)
        axes[2].plot(xi, tissue_final, '-', color=c, lw=1.8, label=lab)

    axes[0].set_xlabel(r'Radial position $\xi$ (center → edge)')
    axes[0].set_ylabel('$x_f$ (func zone fraction)')
    axes[0].set_title('(a) Functional Zone Fraction')
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel(r'Radial position $\xi$ (center → edge)')
    axes[1].set_ylabel(r'$\phi_v^{func}$ (local void)')
    axes[1].set_title('(b) Functional Zone Local Void')
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel(r'Radial position $\xi$ (center → edge)')
    axes[2].set_ylabel(r'$\phi_{tissue}$ (local tissue)')
    axes[2].set_title('(c) Tissue Volume Fraction')
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'radial_profiles.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def _integrate_trajectory(p, t_total, dt, n_motors=50, F_motor_stall=0.5,
                          F_max_per_cell=50.0):
    """Single-sample 1D radial PDE integration returning trajectories.

    Returns (x_f_traj, phi_v_f_traj, phi_v_i_traj, x_f_spatial,
             tissue_traj, tissue_spatial)
    where first three are domain-averaged arrays of length n_steps+1,
    x_f_spatial is (n_steps+1, N_x), tissue_traj is (n_steps+1,),
    and tissue_spatial is (n_steps+1, N_x).
    """
    s = {k: np.array([p[k]]) for k in PARAM_NAMES + DERIVED_NAMES}
    for k, default in FIXED_PARAMS.items():
        s[k] = np.array([p.get(k, default) if isinstance(p, dict) else p[k]])

    phi_f  = s['phi_f'][0]
    phi_i  = s['phi_i'][0]
    phi_solid = phi_f + phi_i
    R_f    = s['R_func'][0]
    E_f    = s['E_func'][0]
    E_i    = s['E_inert'][0]
    n_c    = float(s['n_cells_per_func'][0])
    sense  = s['cell_sense_distance'][0]

    phi_RCP_val = float(compute_phi_RCP(
        s['aspect_ratio_func'], s['aspect_ratio_inert'],
        s['blockiness_n2_func'], s['blockiness_n2_inert'],
        s['phi_f'], s['phi_i'], s['R_func'], s['R_inert'])[0])
    phi_max_val = float(compute_phi_max_deformable(
        np.array([phi_RCP_val]), np.array([E_f]),
        s['blockiness_n2_func'])[0])

    F_cell = float(motor_clutch_force_vec(np.array([E_f]),
                                           n_motors=n_motors,
                                           F_motor_stall=F_motor_stall,
                                           F_max_per_cell=F_max_per_cell)[0])
    E_eff  = float(effective_contact_modulus(
        np.array([E_f]), np.array([E_i]), np.array([phi_f]), np.array([phi_i]))[0])

    domain_area = 640000.0
    N_func = max(1.0, phi_f * domain_area / (np.pi * R_f ** 2))
    ncd = N_func * n_c / domain_area

    eta = max(0.01, min(100.0, 0.5 * (E_eff / 5.0) ** 0.3 * (R_f / 40.0) ** 1.5))
    s0  = max(1e-6, 0.005 * E_eff)
    br_mod = float(bridge_rate_modifier(
        np.array([sense]), np.array([R_f]), np.array([phi_f]),
        np.array([phi_i]), np.array([phi_RCP_val]))[0])

    # Diffusion coefficient
    D_eff_val = D_BASE * (R_f / 40.0) ** 2 * s0 / max(eta, 1e-6)

    # 1D radial grid
    N_x = N_X
    xi = np.linspace(0, 1, N_x)
    dxi = xi[1] - xi[0]
    neighbor_factor = 0.5 * (1.0 + np.cos(np.pi * xi))

    # CFL clamp
    D_max = 0.45 * dxi ** 2 / dt
    D_eff_val = min(D_eff_val, D_max)

    n_steps = int(t_total / dt)
    x_f_min = phi_f / max(phi_max_val, 0.01)
    x_f_0_val = max(phi_f / max(phi_solid, 1e-6), x_f_min)

    # Spatial state: (N_x,)
    x_f = np.full(N_x, x_f_0_val)
    phi_tissue_sp = np.zeros(N_x)
    n_cells_tissue_n = min(1.0, n_c / TISSUE_PARAMS['n_cells_tissue_ref'])

    # Trajectory storage
    traj_xf = [x_f_0_val]
    traj_vf = [max(0.0, 1.0 - phi_f / max(x_f_0_val, 1e-6))]
    traj_vi = [max(0.0, 1.0 - phi_i / max(1.0 - x_f_0_val, 1e-6))]
    spatial_traj = [x_f.copy()]
    traj_tissue = [0.0]
    tissue_spatial_traj = [phi_tissue_sp.copy()]

    for step in range(n_steps):
        t = step * dt
        t_eff = max(0.0, t - 3.0)
        mat = 1.0 - np.exp(-0.3 * t_eff)
        t_br = max(0.0, t - 4.0)
        fb = (1.0 - np.exp(-0.3 * br_mod * t_br * mat)) if mat >= 0.1 else 0.0
        ramp = min(1.0, max(0.0, t - 4.0) / 2.0)

        # Spatially-varying cell stress
        sc = ncd * F_cell * fb * mat * ramp * neighbor_factor

        # Resistance
        pf_local = phi_f / np.maximum(x_f, 1e-6)
        sr = np.where(pf_local / phi_RCP_val > 1.0,
                       s0 * (pf_local / phi_RCP_val - 1.0), 0.0)

        # Compaction
        net = np.maximum(0.0, sc - sr)
        dx_compact = -x_f * net / eta

        # Diffusion: d²x_f/dxi² with zero-flux BCs
        laplacian = np.zeros(N_x)
        laplacian[1:-1] = (x_f[2:] - 2.0 * x_f[1:-1] + x_f[:-2]) / (dxi ** 2)
        laplacian[0] = 2.0 * (x_f[1] - x_f[0]) / (dxi ** 2)
        laplacian[-1] = 2.0 * (x_f[-2] - x_f[-1]) / (dxi ** 2)
        dx_diffuse = D_eff_val * laplacian

        x_f = x_f + dt * (dx_compact + dx_diffuse)
        x_f = np.maximum(x_f, x_f_min)
        x_f = np.minimum(x_f, 1.0 - phi_i)

        # Tissue volume evolution (V2.1)
        phi_void_f_local = np.maximum(0.0, 1.0 - phi_f / np.maximum(x_f, 1e-6))
        phi_tissue_max_loc = TISSUE_PARAMS['alpha_tissue_fill'] * phi_void_f_local
        fb_spatial = fb * neighbor_factor
        growth_drv = np.maximum(TISSUE_PARAMS['alpha_tissue_0'], fb_spatial)
        d_tissue = (TISSUE_PARAMS['k_tissue'] * n_cells_tissue_n * mat
                    * growth_drv * np.maximum(0.0, phi_tissue_max_loc - phi_tissue_sp))
        phi_tissue_sp = phi_tissue_sp + dt * d_tissue
        phi_tissue_sp = np.clip(phi_tissue_sp, 0.0, phi_tissue_max_loc)

        # Domain-averaged
        x_avg = float(np.mean(x_f))
        traj_xf.append(x_avg)
        traj_vf.append(max(0.0, 1.0 - phi_f / max(x_avg, 1e-6)))
        traj_vi.append(max(0.0, 1.0 - phi_i / max(1.0 - x_avg, 1e-6)))
        spatial_traj.append(x_f.copy())
        tissue_avg = float(np.mean(phi_tissue_sp))
        traj_tissue.append(x_avg * tissue_avg)
        tissue_spatial_traj.append(phi_tissue_sp.copy())

    return (np.array(traj_xf), np.array(traj_vf), np.array(traj_vi),
            np.array(spatial_traj), np.array(traj_tissue),
            np.array(tissue_spatial_traj))


# ======================================================================
# Plot 11: Compaction drivers (shape effects)
# ======================================================================

def plot_shape_effects(samples, outputs, outdir):
    """Marginal effect of shape parameters (AR, blockiness) on compaction."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle('Shape & Size Effects on Compaction', fontsize=13, fontweight='bold')

    comp = outputs['compaction_ratio']
    pairs = [
        ('aspect_ratio_func', 'Compaction ratio'),
        ('aspect_ratio_inert', 'Compaction ratio'),
        ('blockiness_n2_func', 'Compaction ratio'),
        ('R_func', '$\\phi_{RCP}$'),
    ]
    targets = [comp, comp, comp, outputs['phi_RCP']]

    for ax, (pk, ylab), tgt in zip(axes, pairs, targets):
        cx, my = _bin_median(samples[pk], tgt, 30)
        ax.plot(cx, my, 'C3o-', ms=4, lw=1.5)
        ax.set_xlabel(PARAM_LABELS.get(pk, pk), fontsize=9)
        ax.set_ylabel(ylab, fontsize=9)
        if PARAM_DEFS[pk]['log']:
            ax.set_xscale('log')
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'shape_effects.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 12: Jamming phase space
# ======================================================================

def plot_jamming_phase_space(samples, outputs, outdir):
    """Show jamming constraint boundary in (phi_solid, func_ratio) space.

    All samples satisfy phi_solid >= phi_RCP by construction (feasibility
    filter).  This plot shows:
    (a) phi_solid vs phi_RCP colored by compaction ratio
    (b) BV/TV (= phi_f) achievable within the jammed regime
    """
    phi_solid = samples['phi_solid']
    func_ratio = samples['func_ratio']
    phi_RCP = outputs['phi_RCP']
    phi_f = samples['phi_f']
    comp = outputs['compaction_ratio']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle('Jamming Constraint: $\\phi_{solid} \\geq \\phi_{RCP}$',
                 fontsize=14, fontweight='bold')

    # (a) phi_solid vs phi_RCP
    ax = axes[0]
    n_pts = min(20000, len(phi_solid))
    idx = np.random.default_rng(0).choice(len(phi_solid), n_pts, replace=False)
    sc = ax.scatter(phi_RCP[idx], phi_solid[idx], c=comp[idx], s=3, alpha=0.3,
                    cmap='YlOrRd', vmin=0, vmax=max(0.5, np.percentile(comp, 95)))
    # 1:1 line (jamming boundary)
    lims = [0.75, 0.96]
    ax.plot(lims, lims, 'k--', lw=2, label='$\\phi_{solid} = \\phi_{RCP}$')
    ax.fill_between(lims, lims, lims[0], color='lightcoral', alpha=0.15,
                    label='Unjammed (filtered)')
    ax.set_xlabel(r'$\phi_{RCP}$ (shape-dependent)', fontsize=10)
    ax.set_ylabel(r'$\phi_{solid}^0$', fontsize=10)
    ax.set_title('(a) Jammed Phase Space', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_aspect('equal')
    plt.colorbar(sc, ax=ax, label='Compaction ratio', shrink=0.8)

    # (b) Accessible BV/TV (= phi_f) vs func_ratio
    ax = axes[1]
    sc2 = ax.scatter(func_ratio[idx], phi_f[idx], c=phi_solid[idx], s=3, alpha=0.3,
                     cmap='viridis')
    ax.set_xlabel(r'Functional ratio $f_{func}$', fontsize=10)
    ax.set_ylabel(r'BV/TV = $\phi_f = \phi_{solid} \times f_{func}$', fontsize=10)
    ax.set_title('(b) Accessible BV/TV Range', fontsize=11)
    ax.grid(True, alpha=0.2)
    plt.colorbar(sc2, ax=ax, label=r'$\phi_{solid}^0$', shrink=0.8)

    # Add organ target BV/TV lines
    oc = _organ_colors()
    from analysis.organ_targets import ORGAN_TARGETS
    for organ_name, target in ORGAN_TARGETS.items():
        bv_tv = target['mean'].get('bv_tv')
        if bv_tv is not None:
            ax.axhline(bv_tv, color=oc.get(organ_name, 'gray'), ls='--',
                       lw=1.2, alpha=0.7)
            ax.text(0.97, bv_tv, organ_name.replace('_', ' ').title(),
                    fontsize=6, ha='right', va='bottom',
                    color=oc.get(organ_name, 'gray'), fontweight='bold')

    # (c) Jamming margin histogram
    ax = axes[2]
    margin = phi_solid - phi_RCP
    ax.hist(margin, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=2, label='Jamming boundary')
    ax.set_xlabel(r'$\phi_{solid} - \phi_{RCP}$ (jamming margin)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('(c) Jamming Margin Distribution', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(outdir, 'jamming_phase_space.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Plot 14: Tissue effects (V2.1)
# ======================================================================

def plot_tissue_effects(samples, outputs, outdir):
    """Impact of tissue volume on architecture descriptors (V2.1)."""
    phi_f = samples['phi_f']
    BV_TV_eff = outputs['BV_TV_eff']
    phi_tissue = outputs['phi_tissue_global']
    K_f = outputs['K_f']
    K_f_tissue = outputs['K_f_tissue']
    n_cells = samples['n_cells_per_func']
    func_ratio = samples['func_ratio']
    comp = outputs['compaction_ratio']

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Tissue Volume Effects on Architecture (V2.1)',
                 fontsize=14, fontweight='bold')

    n_pts = min(20000, len(phi_f))
    idx = np.random.default_rng(0).choice(len(phi_f), n_pts, replace=False)

    # (a) BV/TV_eff vs phi_f — tissue contribution
    ax = axes[0, 0]
    sc = ax.scatter(phi_f[idx], BV_TV_eff[idx], c=phi_tissue[idx], s=3, alpha=0.3,
                    cmap='YlOrRd', vmin=0)
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='No tissue ($BV/TV = \\phi_f$)')
    ax.set_xlabel(r'$\phi_f$ (granule fraction)', fontsize=10)
    ax.set_ylabel('$BV/TV_{eff}$ (granule + tissue)', fontsize=10)
    ax.set_title('(a) Tissue Adds to BV/TV', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.colorbar(sc, ax=ax, label=r'$\phi_{tissue}^{global}$', shrink=0.8)

    # (b) K_f_tissue vs K_f — permeability reduction
    ax = axes[0, 1]
    sc2 = ax.scatter(K_f[idx], K_f_tissue[idx], c=phi_tissue[idx], s=3, alpha=0.3,
                     cmap='YlOrRd', vmin=0)
    lims = [max(1e-3, K_f[idx].min()), K_f[idx].max()]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5, label='No tissue')
    ax.set_xlabel('$K_f$ without tissue (µm²)', fontsize=10)
    ax.set_ylabel('$K_f$ with tissue (µm²)', fontsize=10)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_title('(b) Tissue Reduces Permeability', fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.2)
    plt.colorbar(sc2, ax=ax, label=r'$\phi_{tissue}^{global}$', shrink=0.8)

    # (c) phi_tissue vs n_cells — cell count drives tissue
    ax = axes[1, 0]
    cx, my = _bin_median(n_cells, phi_tissue, 25)
    ax.plot(cx, my, 'C3o-', ms=5, lw=2)
    ax.set_xlabel('Cells per functional granule', fontsize=10)
    ax.set_ylabel(r'$\phi_{tissue}^{global}$ (median)', fontsize=10)
    ax.set_title('(c) Cell Count Drives Tissue Formation', fontsize=11)
    ax.grid(True, alpha=0.2)

    # (d) phi_tissue vs func_ratio — more functional granules → more tissue
    ax = axes[1, 1]
    cx, my = _bin_median(func_ratio, phi_tissue, 25)
    ax.plot(cx, my, 'C0o-', ms=5, lw=2)
    ax.set_xlabel(r'Functional ratio $f_{func}$', fontsize=10)
    ax.set_ylabel(r'$\phi_{tissue}^{global}$ (median)', fontsize=10)
    ax.set_title('(d) Functional Fraction Controls Tissue', fontsize=11)
    ax.grid(True, alpha=0.2)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, 'tissue_effects.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Unified runner
# ======================================================================

def run_all(outdir='results/parameter_sweep', n_samples=N_SAMPLES_DEFAULT,
            t_total=72.0, seed=42, verbose=True, **sweep_kwargs):
    """Run full pipeline: sample, integrate, analyse, plot.

    Extra keyword arguments are forwarded to run_sweep_vectorised
    (n_motors, F_motor_stall, F_max_per_cell, bridge_lock_force_threshold,
     bridge_secondary_rate_mult, bridge_senescence_time).

    Returns (samples, outputs, organ_dists, recs).
    """
    os.makedirs(outdir, exist_ok=True)

    print("=" * 70)
    print("  Mean-Field Parameter Sweep & Organ Target Prediction")
    print(f"  {N_PARAMS} parameters, {n_samples:,} LHS samples")
    if sweep_kwargs:
        print(f"  Overrides: {sweep_kwargs}")
    print("=" * 70)

    samples, outputs, organ_dists = run_sweep(n_samples, t_total, seed, verbose,
                                               **sweep_kwargs)
    n = len(outputs['phi_solid'])

    # Save data
    csv_path = os.path.join(outdir, 'sweep_data.csv')
    _save_csv(samples, outputs, organ_dists, csv_path)
    print(f"  Saved: {csv_path}  ({n:,} rows)")

    # Recommendations
    recs = recommend_parameters(samples, outputs, organ_dists)
    _save_json(recs, os.path.join(outdir, 'recommendations.json'))

    # Print summary
    print("\n" + "-" * 70)
    print("  OPTIMAL PARAMETERS PER ORGAN TARGET")
    print("-" * 70)
    for organ in list_organs():
        if organ not in recs or not recs[organ]:
            continue
        e = recs[organ][0]
        p = e['params']
        print(f"\n  {organ.replace('_', ' ').upper()}  (D_arch = {e['D_arch']:.2f})")
        print(f"    E_func={p['E_func']:.1f} kPa, E_inert={p['E_inert']:.1f} kPa")
        print(f"    R_func={p['R_func']:.0f} µm, R_inert={p['R_inert']:.0f} µm")
        print(f"    phi_solid={p['phi_solid']:.2f} (phi_f={p['phi_f']:.2f}, "
              f"phi_i={p['phi_i']:.2f}), n_cells={int(round(p['n_cells_per_func']))}")
        print(f"    AR_f={p['aspect_ratio_func']:.2f}, AR_i={p['aspect_ratio_inert']:.2f}, "
              f"n2_f={p['blockiness_n2_func']:.1f}, n2_i={p['blockiness_n2_inert']:.1f}")
        print(f"    sense={p['cell_sense_distance']:.0f} µm")
        print(f"    → compaction={e['compaction_ratio']:.1%}, "
              f"void_f={e['phi_v_f_local']:.3f}, void_i={e['phi_v_i_local']:.3f}, "
              f"K={e['K_permeability']:.2e} µm²")
        print(f"    → tissue={e.get('phi_tissue_global', 0):.3f}, "
              f"BV/TV_eff={e.get('BV_TV_eff', 0):.3f}, "
              f"K_tissue={e.get('K_eff_tissue', 0):.2e} µm²")

    # Plots
    print("\n" + "-" * 70)
    print("  GENERATING PLOTS")
    print("-" * 70)

    plot_marginal_effects(samples, outputs, outdir)
    plot_compaction_heatmaps(samples, outputs, outdir)
    plot_solid_phase_balance(samples, outputs, outdir)
    plot_organ_landscapes(samples, organ_dists, outdir)
    plot_recommendation_table(recs, outdir)
    plot_parameter_importance(samples, organ_dists, outdir)
    plot_closest_organ_map(outputs, organ_dists, outdir)
    plot_permeability_porosity(outputs, outdir)
    plot_radar_comparison(recs, outdir)
    plot_best_kinetics(recs, outdir, t_total,
                       n_motors=sweep_kwargs.get('n_motors', 50),
                       F_motor_stall=sweep_kwargs.get('F_motor_stall', 0.5),
                       F_max_per_cell=sweep_kwargs.get('F_max_per_cell', 50.0))
    plot_shape_effects(samples, outputs, outdir)
    plot_jamming_phase_space(samples, outputs, outdir)
    plot_tissue_effects(samples, outputs, outdir)

    # ── Scaffold evolution visualizations ──
    print("\n" + "-" * 70)
    print("  SCAFFOLD EVOLUTION VISUALIZATION")
    print("-" * 70)
    try:
        from viz.scaffold_evolution import make_evolution_figure, make_timelapse
        make_evolution_figure(recs, outdir, t_total=t_total, seed=42)
        try:
            make_timelapse(recs, outdir, t_total=t_total, seed=42,
                           fps=15, duration_sec=8.0, fmt='gif')
        except Exception as e:
            print(f"  Timelapse skipped: {e}")
    except ImportError as e:
        print(f"  scaffold_evolution skipped: {e}")
    except Exception as e:
        print(f"  scaffold_evolution failed: {e}")

    try:
        from viz.scaffold_evolution_3d import make_evolution_figure_3d
        make_evolution_figure_3d(recs, outdir, t_total=t_total, seed=42)
    except ImportError as e:
        print(f"  scaffold_evolution_3d skipped: {e}")
    except Exception as e:
        print(f"  scaffold_evolution_3d failed: {e}")

    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP COMPLETE")
    print("=" * 70)
    print(f"  {n:,} feasible samples evaluated")
    print(f"  Output: {outdir}/")
    print(f"  Data:  sweep_data.csv, recommendations.json")
    print(f"  Plots: marginal_effects_phiv, compaction_heatmaps,")
    print(f"         solid_phase_balance, organ_landscapes,")
    print(f"         organ_recommendations_table, parameter_importance,")
    print(f"         closest_organ_map, permeability_porosity,")
    print(f"         radar_comparison, best_kinetics, shape_effects,")
    print(f"         jamming_phase_space, tissue_effects,")
    print(f"         radial_profiles, scaffold_evolution, timelapse")

    return samples, outputs, organ_dists, recs


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Mean-field 12-parameter sweep for organ target prediction')
    parser.add_argument('-o', '--outdir', default='results/parameter_sweep',
                        help='Output directory')
    parser.add_argument('-n', '--n-samples', type=int, default=N_SAMPLES_DEFAULT,
                        help=f'Number of LHS samples (default: {N_SAMPLES_DEFAULT:,})')
    parser.add_argument('--t-total', type=float, default=72.0,
                        help='Simulation time in hours (default: 72)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for LHS (default: 42)')
    parser.add_argument('--quick', action='store_true',
                        help=f'Use {N_SAMPLES_QUICK:,} samples for fast testing')
    parser.add_argument('--n-motors', type=int, default=50,
                        help='Motor complexes per cell (default: 50)')
    parser.add_argument('--F-motor-stall', type=float, default=0.5,
                        help='Force per motor complex in nN (default: 0.5)')
    parser.add_argument('--F-max-per-cell', type=float, default=50.0,
                        help='Max force cap per cell in nN (default: 50)')
    parser.add_argument('--bridge-lock-threshold', type=float, default=None,
                        help='Bridge lock-in force threshold in nN (default: off)')
    parser.add_argument('--bridge-secondary-mult', type=float, default=1.0,
                        help='Secondary bridge rate multiplier (default: 1.0)')
    parser.add_argument('--bridge-senescence-time', type=float, default=24.0,
                        help='Bridge senescence time in hours (default: 24)')
    args = parser.parse_args()

    ns = N_SAMPLES_QUICK if args.quick else args.n_samples
    run_all(outdir=args.outdir, n_samples=ns, t_total=args.t_total, seed=args.seed,
            n_motors=args.n_motors,
            F_motor_stall=args.F_motor_stall,
            F_max_per_cell=args.F_max_per_cell,
            bridge_lock_force_threshold=args.bridge_lock_threshold,
            bridge_secondary_rate_mult=args.bridge_secondary_mult,
            bridge_senescence_time=args.bridge_senescence_time)
