#!/usr/bin/env python3
"""
Dimensionless Analysis for GELS DOE Results
==================================================
Collapses DOE simulation results by key dimensionless groups to reveal
universal scaling relationships governing cell-driven granular scaffold
rearrangement.

Dimensionless groups:
    beta  -- Motor-clutch engagement ratio (master variable, 0-1)
    Ca    -- Cellular capillary number (cell stress / granule stiffness)
    phi_J -- Jamming proximity (solid fraction / RCP fraction)
    Da    -- Darcy number (permeability / domain scale)
    T_r   -- Timescale ratio (biological / mechanical relaxation)
    phi_r -- Composition ratio (functional / inert solid fraction)
    R_r   -- Size ratio (functional / inert granule radius)

Usage:
    python viz/dimensionless.py                                # default scan
    python viz/dimensionless.py --scan-dir Trials/trials/
    python viz/dimensionless.py --scan-dir results/trials/ -o results/dimensionless
    python viz/dimensionless.py --skip phase_space jamming

Or import programmatically:
    from viz.dimensionless import load_doe_data, run_all
    data = load_doe_data('Trials/trials/')
    run_all(data, outdir='results/dimensionless')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import argparse
import csv
import json
import glob as globmod
import os
import sys
import warnings
from collections import OrderedDict


# ======================================================================
# Configuration
# ======================================================================

# Default motor-clutch parameters (from Params dataclass defaults / DOE base)
_MC_DEFAULTS = dict(
    n_motors=50,
    F_motor_stall=0.5,       # nN
    n_clutches=75,
    k_clutch=5.0,            # nN/um
    k_on_clutch=1.0,         # 1/s
    k_off_clutch=0.1,        # 1/s
    cell_diameter=20.0,      # um
    poisson_ratio=0.49,
    bridge_attempt_rate=0.3, # 1/h
    t_spread_duration=3.0,   # h
    drag_scale=0.05,
)

# RCP fraction for spheres
PHI_J_SPHERES = 0.64


# ======================================================================
# Data loading
# ======================================================================

def _load_history_csv(path):
    """Load history.csv as list of dicts with numeric values."""
    rows = []
    with open(path) as f:
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
            rows.append(entry)
    return rows


def _load_history_json(path):
    """Load history.json as list of dicts."""
    with open(path) as f:
        return json.load(f)


def load_doe_data(scan_dir, trials_dir=None):
    """Scan for DOE_* directories and load params + history from each.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE_01/, DOE_02/, ... subdirectories
        (or DOE_01.tar.gz, etc.).
    trials_dir : str or None
        Directory containing DOE_01.json, ... config files.
        Defaults to Trials/ relative to this script.

    Returns
    -------
    data : dict with keys:
        'runs' : list of dicts, each with:
            'run_id': int, 'name': str, 'params': dict, 'hist': list[dict],
            'factors': dict, 'block': int, 'description': str
        'factor_names': list of str
        'factor_levels': dict mapping factor -> set of levels
    """
    if trials_dir is None:
        trials_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'Trials')

    runs = []
    factor_levels = {}

    for i in range(1, 25):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(scan_dir, name)
        tar_path = run_dir + ".tar.gz"

        # Try extracting tar.gz if directory not present
        if not os.path.isdir(run_dir) and os.path.isfile(tar_path):
            try:
                import tarfile
                with tarfile.open(tar_path, 'r:gz') as tf:
                    tf.extractall(path=scan_dir)
            except Exception as e:
                print(f"  WARNING: {name} -- could not extract tar.gz: {e}")

        # Load history
        hist = None
        hist_csv = os.path.join(run_dir, 'history.csv')
        hist_json = os.path.join(run_dir, 'history.json')
        if os.path.isfile(hist_csv):
            hist = _load_history_csv(hist_csv)
        elif os.path.isfile(hist_json):
            hist = _load_history_json(hist_json)

        if hist is None or len(hist) < 2:
            print(f"  WARNING: {name} -- no usable history, skipping")
            continue

        # Load params (from results or trial config)
        params = {}
        params_path = os.path.join(run_dir, 'params.json')
        config_path = os.path.join(trials_dir, f"{name}.json")

        if os.path.isfile(params_path):
            with open(params_path) as f:
                params = json.load(f)
        elif os.path.isfile(config_path):
            with open(config_path) as f:
                params = json.load(f)
        else:
            # Last resort: use trial config alone
            print(f"  WARNING: {name} -- no params.json, using trial config")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    params = json.load(f)
            else:
                print(f"  WARNING: {name} -- no config found, skipping")
                continue

        # Also merge trial config for DOE metadata if params lacks it
        if '_doe_factors' not in params and os.path.isfile(config_path):
            with open(config_path) as f:
                trial_cfg = json.load(f)
            for key in ('_doe_factors', '_doe_block', '_doe_description', '_doe_run'):
                if key in trial_cfg:
                    params[key] = trial_cfg[key]

        factors = params.get('_doe_factors', {})
        block = params.get('_doe_block', 1 if i <= 16 else 2)
        description = params.get('_doe_description', '')

        # Track factor levels
        for fk, fv in factors.items():
            factor_levels.setdefault(fk, set()).add(fv)

        runs.append({
            'run_id': i,
            'name': name,
            'params': params,
            'hist': hist,
            'factors': factors,
            'block': block,
            'description': description,
        })

    factor_names = sorted(factor_levels.keys())
    print(f"\n  Loaded {len(runs)} / 24 DOE runs from {scan_dir}")

    return {
        'runs': runs,
        'factor_names': factor_names,
        'factor_levels': {k: sorted(v) for k, v in factor_levels.items()},
    }


# ======================================================================
# Dimensionless group computation
# ======================================================================

def _get_param(params, key, default=None):
    """Safely get a parameter, falling back to default."""
    val = params.get(key, default)
    if val is None:
        val = _MC_DEFAULTS.get(key)
    return val


def compute_dimensionless_groups(params):
    """Compute all dimensionless groups from a params dict.

    Parameters
    ----------
    params : dict
        Simulation parameters (flat format).

    Returns
    -------
    groups : dict with keys:
        'beta', 'Ca', 'phi_ratio', 'R_ratio', 'jamming_proximity',
        'timescale_ratio', 'phi_solid', 'phi_J'
    """
    # Extract parameters with defaults
    E_modulus = _get_param(params, 'E_modulus', 10.0)         # kPa
    poisson = _get_param(params, 'poisson_ratio', 0.49)
    cell_diam = _get_param(params, 'cell_diameter', 20.0)     # um
    n_motors = _get_param(params, 'n_motors', 50)
    F_stall_per = _get_param(params, 'F_motor_stall', 0.5)    # nN
    n_clutches = _get_param(params, 'n_clutches', 75)
    k_clutch = _get_param(params, 'k_clutch', 5.0)            # nN/um
    k_on = _get_param(params, 'k_on_clutch', 1.0)             # 1/s
    k_off = _get_param(params, 'k_off_clutch', 0.1)           # 1/s
    n_cells = _get_param(params, 'n_cells_per_granule', 8)
    R_func = _get_param(params, 'R_func_mean', 40.0)          # um
    R_inert = _get_param(params, 'R_inert_mean', 60.0)        # um
    phi_f = _get_param(params, 'phi_f_target', 0.25)
    phi_i = _get_param(params, 'phi_i_target', 0.20)
    bridge_rate = _get_param(params, 'bridge_attempt_rate', 0.3)  # 1/h
    t_spread = _get_param(params, 't_spread_duration', 3.0)      # h
    drag_scale = _get_param(params, 'drag_scale', 0.05)
    aspect_ratio = _get_param(params, 'aspect_ratio_func_mean', 1.0)

    # -- beta: Motor-clutch engagement ratio --
    a_cell = cell_diam / 2.0
    # k_sub = pi * E * a / (1 - nu^2),  units: kPa * um = nN/um
    k_sub = np.pi * E_modulus * a_cell / (1.0 - poisson**2)
    k_opt = n_clutches * k_clutch  # nN/um
    beta = k_sub / (k_sub + k_opt)

    # -- Ca: Cellular capillary number --
    F_stall = n_motors * F_stall_per  # nN, total stall force
    engagement = k_on / (k_on + k_off)
    F_cell = F_stall * beta * engagement  # force per cell at full maturity (nN)
    sigma_cell = n_cells * F_cell / (np.pi * R_func**2)  # nN/um^2
    # Effective Hertzian modulus: E* = E / (2*(1-nu^2)), units kPa
    E_star = E_modulus / (2.0 * (1.0 - poisson**2))
    # Convert E_star from kPa to nN/um^2: 1 kPa = 1e-3 nN/um^2
    E_star_nN = E_star * 1e-3
    Ca = sigma_cell / E_star_nN if E_star_nN > 0 else 0.0

    # -- Jamming proximity --
    phi_solid = phi_f + phi_i
    phi_J = _rcp_estimate(aspect_ratio)
    jamming_proximity = phi_solid / phi_J if phi_J > 0 else 0.0

    # -- Composition ratio --
    phi_ratio = phi_f / phi_i if phi_i > 0 else float('inf')

    # -- Size ratio --
    R_ratio = R_func / R_inert if R_inert > 0 else 1.0

    # -- Timescale ratio (biological / mechanical) --
    tau_bridge = 1.0 / bridge_rate if bridge_rate > 0 else float('inf')  # h
    tau_spread = t_spread / max(0.2, beta)  # h
    # Mechanical relaxation: gamma * R / (E_star_Pa * R * 1e-6)
    # gamma = drag_scale * R  (nN*h/um)
    # tau_damp ~ gamma / (E_star * 1e-3)  (h)  [E_star in kPa, convert to nN/um^2]
    gamma = drag_scale * R_func  # nN*h/um
    tau_damp = gamma / max(1e-12, E_star * 1e-3)  # h
    timescale_ratio = tau_bridge / tau_damp if tau_damp > 0 else float('inf')

    return {
        'beta': beta,
        'Ca': Ca,
        'phi_ratio': phi_ratio,
        'R_ratio': R_ratio,
        'jamming_proximity': jamming_proximity,
        'phi_solid': phi_solid,
        'phi_J': phi_J,
        'timescale_ratio': timescale_ratio,
        'tau_spread': tau_spread,
        'tau_bridge': tau_bridge,
        'tau_damp': tau_damp,
        'F_cell': F_cell,
        'sigma_cell': sigma_cell,
        'E_star': E_star,
        'k_sub': k_sub,
        'k_opt': k_opt,
    }


def _rcp_estimate(mean_aspect_ratio):
    """Estimate random close packing fraction (Donev et al. 2004)."""
    ar = max(1.0, mean_aspect_ratio)
    return min(0.74, PHI_J_SPHERES + 0.08 * (ar - 1.0))


# ======================================================================
# Dimensionless response computation
# ======================================================================

def compute_dimensionless_responses(hist, params=None):
    """Compute dimensionless response variables from a history time series.

    Parameters
    ----------
    hist : list of dict
        History entries with keys like phi_f_mean, porosity, K_kozeny_carman, etc.
    params : dict or None
        Params dict (used to estimate bridge capacity).

    Returns
    -------
    responses : dict
    """
    if not hist or len(hist) < 2:
        return {}

    h0 = hist[0]
    hf = hist[-1]

    responses = {}

    # Compaction efficiency
    phi_f_i = h0.get('phi_f_mean', h0.get('phi_solid', 0))
    phi_f_f = hf.get('phi_f_mean', hf.get('phi_solid', 0))
    if phi_f_i > 0:
        responses['compaction_efficiency'] = (phi_f_f - phi_f_i) / phi_f_i
    else:
        responses['compaction_efficiency'] = 0.0

    # Connectivity gain
    lf_i = h0.get('func_lf', 0.0)
    lf_f = hf.get('func_lf', 0.0)
    responses['connectivity_gain'] = lf_f - lf_i

    # Permeability ratio
    K_i = h0.get('K_kozeny_carman', 0.0)
    K_f = hf.get('K_kozeny_carman', 0.0)
    if K_i > 0:
        responses['permeability_ratio'] = K_f / K_i
    else:
        responses['permeability_ratio'] = 1.0

    # Bridge saturation (estimated)
    n_bridges_f = hf.get('n_bridges', 0)
    # Estimate max bridges: n_func_granules * n_cells * 0.5 (each bridge connects 2)
    if params is not None:
        n_cells = params.get('n_cells_per_granule', 8)
        phi_f = params.get('phi_f_target', 0.25)
        R_func = params.get('R_func_mean', 40.0)
        Lx = params.get('Lx', 800.0)
        Ly = params.get('Ly', 800.0)
        Lz = params.get('Lz', 800.0)
        V_gran = (4.0 / 3.0) * np.pi * R_func**3
        n_func = max(1, phi_f * Lx * Ly * Lz / V_gran)
        n_bridges_max = n_func * n_cells * 0.5  # rough estimate
        responses['bridge_saturation'] = n_bridges_f / max(1, n_bridges_max)
    else:
        responses['bridge_saturation'] = 0.0

    # Final porosity
    responses['porosity_final'] = hf.get('porosity', 0.0)

    # Final compaction ratio
    responses['compaction_ratio_final'] = hf.get('compaction_ratio', 0.0)

    # Tissue fraction final
    responses['tissue_frac_final'] = hf.get('tissue_frac', 0.0)

    # Functional displacement
    responses['disp_func_final'] = hf.get('disp_func', 0.0)

    # Final bridges
    responses['n_bridges_final'] = n_bridges_f

    # Da number final
    responses['Da_final'] = hf.get('Da_number', 0.0)

    return responses


# ======================================================================
# Helper: enrich run data with dimensionless groups and responses
# ======================================================================

def _enrich_runs(data):
    """Add dimensionless groups and responses to each run in data."""
    for run in data['runs']:
        if 'groups' not in run:
            run['groups'] = compute_dimensionless_groups(run['params'])
        if 'responses' not in run:
            run['responses'] = compute_dimensionless_responses(
                run['hist'], run['params'])
    return data


# ======================================================================
# Helper: linear / power-law fits with R^2
# ======================================================================

def _fit_line(x, y):
    """Ordinary least squares line fit. Returns (slope, intercept, R2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    coeffs = np.polyfit(x, y, 1)
    y_pred = np.polyval(coeffs, x)
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return coeffs[0], coeffs[1], r2


def _fit_power(x, y):
    """Power-law fit y = a * x^b via log-log OLS. Returns (a, b, R2)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return None
    lx, ly = np.log10(x), np.log10(y)
    coeffs = np.polyfit(lx, ly, 1)
    b = coeffs[0]
    a = 10**coeffs[1]
    y_pred = a * x**b
    ss_res = np.sum((y - y_pred)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return a, b, r2


def _annotate_fit(ax, fit_result, fit_type='linear', color='black', loc='upper right'):
    """Add a fit annotation to an axes."""
    if fit_result is None:
        return
    if fit_type == 'linear':
        slope, intercept, r2 = fit_result
        sign = '+' if intercept >= 0 else '-'
        text = f'y = {slope:.3g}x {sign} {abs(intercept):.3g}\n$R^2$ = {r2:.3f}'
    elif fit_type == 'power':
        a, b, r2 = fit_result
        text = f'y = {a:.3g} $x^{{{b:.2f}}}$\n$R^2$ = {r2:.3f}'
    else:
        return

    props = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                 edgecolor='gray', linewidth=0.5)
    # Map string loc to axes coordinates
    locs = {
        'upper right': (0.97, 0.97, 'right', 'top'),
        'upper left': (0.03, 0.97, 'left', 'top'),
        'lower right': (0.97, 0.03, 'right', 'bottom'),
        'lower left': (0.03, 0.03, 'left', 'bottom'),
    }
    x, y, ha, va = locs.get(loc, locs['upper right'])
    ax.text(x, y, text, transform=ax.transAxes, fontsize=8, color=color,
            ha=ha, va=va, bbox=props)


def _draw_fit_line(ax, x, y, fit_type='linear', color='gray', alpha=0.6):
    """Draw a best-fit line on the axes, return the fit result."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if fit_type == 'linear':
        result = _fit_line(x, y)
        if result is None:
            return None
        slope, intercept, r2 = result
        xf = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yf = slope * xf + intercept
        ax.plot(xf, yf, '--', color=color, alpha=alpha, linewidth=1.5, zorder=1)
        return result
    elif fit_type == 'power':
        result = _fit_power(x, y)
        if result is None:
            return None
        a, b, r2 = result
        mask = x > 0
        if mask.sum() < 2:
            return result
        xf = np.linspace(np.nanmin(x[mask]), np.nanmax(x[mask]), 100)
        yf = a * xf**b
        ax.plot(xf, yf, '--', color=color, alpha=alpha, linewidth=1.5, zorder=1)
        return result
    return None


# ======================================================================
# Plot 1: Beta collapse
# ======================================================================

def plot_beta_collapse(data, outdir):
    """Master plot showing how beta (motor-clutch engagement) collapses behaviour.

    Panel 1: phi_f(t) curves colored by beta
    Panel 2: phi_f(t/tau_spread) rescaled
    Panel 3: Final compaction vs beta
    Panel 4: Final permeability ratio vs beta
    """
    data = _enrich_runs(data)
    runs = data['runs']

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Collect beta values for color normalization
    betas = [r['groups']['beta'] for r in runs]
    norm = Normalize(vmin=min(betas), vmax=max(betas))
    cmap = plt.cm.viridis

    # ---- Panel 1: phi_f(t) colored by beta ----
    ax = axes[0, 0]
    for run in runs:
        beta = run['groups']['beta']
        times = [h.get('time', 0) for h in run['hist']]
        # Use phi_f_mean if available, else phi_solid
        phi_f_vals = [h.get('phi_f_mean', h.get('phi_solid', 0)) for h in run['hist']]
        color = cmap(norm(beta))
        ax.plot(times, phi_f_vals, color=color, alpha=0.7, linewidth=1.2)
    ax.set_xlabel('Time (h)', fontsize=10)
    ax.set_ylabel(r'$\phi_f$ (mean)', fontsize=10)
    ax.set_title(r'Functional fraction vs time, colored by $\beta$', fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label=r'$\beta$', fraction=0.046, pad=0.04)

    # ---- Panel 2: phi_f(t/tau_spread) rescaled ----
    ax = axes[0, 1]
    for run in runs:
        beta = run['groups']['beta']
        tau_s = run['groups']['tau_spread']
        if tau_s <= 0 or not np.isfinite(tau_s):
            continue
        times = [h.get('time', 0) for h in run['hist']]
        phi_f_vals = [h.get('phi_f_mean', h.get('phi_solid', 0)) for h in run['hist']]
        t_scaled = [t / tau_s for t in times]
        color = cmap(norm(beta))
        ax.plot(t_scaled, phi_f_vals, color=color, alpha=0.7, linewidth=1.2)
    ax.set_xlabel(r'$t / \tau_{spread}$', fontsize=10)
    ax.set_ylabel(r'$\phi_f$ (mean)', fontsize=10)
    ax.set_title(r'Rescaled by $\tau_{spread}$ -- should collapse if $\beta$ controls timescale',
                 fontsize=10)
    plt.colorbar(sm, ax=ax, label=r'$\beta$', fraction=0.046, pad=0.04)

    # ---- Panel 3: Final compaction vs beta ----
    ax = axes[1, 0]
    x_vals, y_vals = [], []
    for run in runs:
        beta = run['groups']['beta']
        comp = run['responses'].get('compaction_efficiency', None)
        if comp is not None and np.isfinite(comp):
            ax.scatter(beta, comp, c=[cmap(norm(beta))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(beta)
            y_vals.append(comp)
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
    _annotate_fit(ax, fit, fit_type='linear')
    ax.set_xlabel(r'$\beta$ (engagement ratio)', fontsize=10)
    ax.set_ylabel('Compaction efficiency', fontsize=10)
    ax.set_title(r'Final compaction efficiency vs $\beta$', fontsize=11)

    # ---- Panel 4: Permeability ratio vs beta ----
    ax = axes[1, 1]
    x_vals, y_vals = [], []
    for run in runs:
        beta = run['groups']['beta']
        perm = run['responses'].get('permeability_ratio', None)
        if perm is not None and np.isfinite(perm) and perm > 0:
            ax.scatter(beta, perm, c=[cmap(norm(beta))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(beta)
            y_vals.append(perm)
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
    _annotate_fit(ax, fit, fit_type='linear')
    ax.set_xlabel(r'$\beta$ (engagement ratio)', fontsize=10)
    ax.set_ylabel(r'$K_{final} / K_{initial}$', fontsize=10)
    ax.set_title(r'Permeability retention vs $\beta$', fontsize=11)
    ax.axhline(1.0, ls=':', color='gray', lw=0.8, alpha=0.5)

    fig.suptitle(r'$\beta$-Collapse Analysis (Motor-Clutch Engagement)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'beta_collapse.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 2: Ca scaling
# ======================================================================

def plot_Ca_scaling(data, outdir):
    """Compaction efficiency and displacement vs cellular capillary number Ca."""
    data = _enrich_runs(data)
    runs = data['runs']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    cas = [r['groups']['Ca'] for r in runs if np.isfinite(r['groups']['Ca'])]
    if not cas:
        plt.close(fig)
        return
    norm = Normalize(vmin=min(cas), vmax=max(cas))
    cmap = plt.cm.plasma

    # ---- Panel 1: Compaction efficiency vs Ca ----
    ax = axes[0]
    x_vals, y_vals = [], []
    for run in runs:
        ca = run['groups']['Ca']
        comp = run['responses'].get('compaction_efficiency', None)
        if comp is not None and np.isfinite(ca) and np.isfinite(comp) and ca > 0:
            ax.scatter(ca, comp, c=[cmap(norm(ca))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(ca)
            y_vals.append(comp)
    # Try power-law first, fall back to linear
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='power', color='gray')
    if fit is not None:
        _annotate_fit(ax, fit, fit_type='power')
    else:
        fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
        _annotate_fit(ax, fit, fit_type='linear')
    ax.set_xlabel('Ca (cellular capillary number)', fontsize=10)
    ax.set_ylabel('Compaction efficiency', fontsize=10)
    ax.set_title('Compaction efficiency vs Ca', fontsize=11)
    if x_vals and min(x_vals) > 0 and max(x_vals) / min(x_vals) > 10:
        ax.set_xscale('log')

    # ---- Panel 2: Functional displacement vs Ca ----
    ax = axes[1]
    x_vals, y_vals = [], []
    for run in runs:
        ca = run['groups']['Ca']
        disp = run['responses'].get('disp_func_final', None)
        if disp is not None and np.isfinite(ca) and np.isfinite(disp) and ca > 0:
            ax.scatter(ca, disp, c=[cmap(norm(ca))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(ca)
            y_vals.append(disp)
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='power', color='gray')
    if fit is not None:
        _annotate_fit(ax, fit, fit_type='power')
    else:
        fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
        _annotate_fit(ax, fit, fit_type='linear')
    ax.set_xlabel('Ca (cellular capillary number)', fontsize=10)
    ax.set_ylabel('Functional displacement (um)', fontsize=10)
    ax.set_title('Functional displacement vs Ca', fontsize=11)
    if x_vals and min(x_vals) > 0 and max(x_vals) / min(x_vals) > 10:
        ax.set_xscale('log')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label='Ca', fraction=0.02, pad=0.04)

    fig.suptitle('Cellular Capillary Number Scaling', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'Ca_scaling.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 3: Jamming diagram
# ======================================================================

def plot_jamming_diagram(data, outdir):
    """Phase diagram: phi_solid(final) vs phi_J, colored by beta, sized by Ca."""
    data = _enrich_runs(data)
    runs = data['runs']

    fig, ax = plt.subplots(figsize=(9, 7))

    betas = [r['groups']['beta'] for r in runs]
    norm_beta = Normalize(vmin=min(betas), vmax=max(betas))
    cmap = plt.cm.viridis

    cas = [r['groups']['Ca'] for r in runs if np.isfinite(r['groups']['Ca'])]
    ca_min = min(cas) if cas else 0
    ca_max = max(cas) if cas else 1

    for run in runs:
        beta = run['groups']['beta']
        ca = run['groups']['Ca']
        phi_J = run['groups']['phi_J']

        # Final solid fraction from history
        hf = run['hist'][-1]
        phi_solid_final = hf.get('phi_solid', hf.get('compaction_ratio', 0))
        if phi_solid_final == 0:
            phi_solid_final = hf.get('phi_f_mean', 0) + hf.get('phi_i_mean', 0)

        # Size proportional to Ca
        if np.isfinite(ca) and ca_max > ca_min:
            size = 40 + 200 * (ca - ca_min) / (ca_max - ca_min)
        else:
            size = 80

        ax.scatter(phi_J, phi_solid_final,
                   c=[cmap(norm_beta(beta))],
                   s=size, edgecolors='black', linewidths=0.5, zorder=3, alpha=0.85)

    # Reference line: jamming boundary
    ax.plot([0.5, 0.8], [0.5, 0.8], 'k--', lw=1.5, alpha=0.4,
            label=r'$\phi_{solid} = \phi_J$ (jamming)')

    # Shade jammed region
    ax.fill_between([0.5, 0.8], [0.5, 0.8], [0.85, 0.85],
                     color='red', alpha=0.06, label='Jammed region')
    ax.fill_between([0.5, 0.8], [0, 0], [0.5, 0.8],
                     color='blue', alpha=0.06, label='Unjammed region')

    ax.set_xlabel(r'$\phi_J$ (RCP estimate)', fontsize=11)
    ax.set_ylabel(r'$\phi_{solid}$ (final)', fontsize=11)
    ax.set_title('Jamming Proximity Diagram', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_beta)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label=r'$\beta$', fraction=0.046, pad=0.04)

    # Size legend for Ca
    for ca_val, label in [(ca_min, f'Ca={ca_min:.1f}'),
                          ((ca_min + ca_max) / 2, ''),
                          (ca_max, f'Ca={ca_max:.1f}')]:
        if ca_max > ca_min:
            s = 40 + 200 * (ca_val - ca_min) / (ca_max - ca_min)
        else:
            s = 80
        ax.scatter([], [], s=s, c='gray', edgecolors='black',
                   linewidths=0.5, label=label)
    ax.legend(fontsize=8, loc='upper left', ncol=1)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'jamming_diagram.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 4: Composition effects
# ======================================================================

def plot_composition_effects(data, outdir):
    """Compaction and permeability vs phi_f/phi_i ratio, colored by beta."""
    data = _enrich_runs(data)
    runs = data['runs']

    # Filter out inf phi_ratio (func-only runs with phi_i=0)
    runs_finite = [r for r in runs
                   if np.isfinite(r['groups']['phi_ratio'])]

    if len(runs_finite) < 2:
        print("    Skipping composition_effects: too few runs with finite phi_ratio")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    betas = [r['groups']['beta'] for r in runs_finite]
    norm = Normalize(vmin=min(betas), vmax=max(betas))
    cmap = plt.cm.viridis

    # ---- Panel 1: Compaction vs phi_ratio ----
    ax = axes[0]
    x_vals, y_vals = [], []
    for run in runs_finite:
        phi_r = run['groups']['phi_ratio']
        comp = run['responses'].get('compaction_efficiency', None)
        beta = run['groups']['beta']
        if comp is not None and np.isfinite(comp):
            ax.scatter(phi_r, comp, c=[cmap(norm(beta))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(phi_r)
            y_vals.append(comp)
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
    _annotate_fit(ax, fit, fit_type='linear')
    ax.set_xlabel(r'$\phi_f / \phi_i$ (composition ratio)', fontsize=10)
    ax.set_ylabel('Compaction efficiency', fontsize=10)
    ax.set_title(r'Compaction vs composition, colored by $\beta$', fontsize=11)

    # ---- Panel 2: Permeability vs phi_ratio ----
    ax = axes[1]
    x_vals, y_vals = [], []
    for run in runs_finite:
        phi_r = run['groups']['phi_ratio']
        perm = run['responses'].get('permeability_ratio', None)
        beta = run['groups']['beta']
        if perm is not None and np.isfinite(perm):
            ax.scatter(phi_r, perm, c=[cmap(norm(beta))], s=60,
                       edgecolors='white', linewidths=0.5, zorder=3)
            x_vals.append(phi_r)
            y_vals.append(perm)
    fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
    _annotate_fit(ax, fit, fit_type='linear')
    ax.axhline(1.0, ls=':', color='gray', lw=0.8, alpha=0.5)
    ax.set_xlabel(r'$\phi_f / \phi_i$ (composition ratio)', fontsize=10)
    ax.set_ylabel(r'$K_{final} / K_{initial}$', fontsize=10)
    ax.set_title(r'Permeability vs composition, colored by $\beta$', fontsize=11)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=axes, label=r'$\beta$', fraction=0.02, pad=0.04)

    fig.suptitle('Composition Ratio Effects', fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'composition_effects.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 5: Factor effects (classical DOE main + interaction)
# ======================================================================

def plot_factor_effects(data, outdir):
    """Classical DOE analysis: main effects and 2-factor interactions.

    Uses the _doe_factors dict from each run's params.
    """
    data = _enrich_runs(data)
    runs = data['runs']

    # Only use block 1 (full factorial) for this analysis
    block1 = [r for r in runs if r['block'] == 1]
    if len(block1) < 4:
        print("    Skipping factor_effects: too few Block 1 runs")
        return

    factors = sorted(data['factor_names'])
    response_keys = ['compaction_efficiency', 'permeability_ratio',
                     'connectivity_gain', 'bridge_saturation']
    response_labels = {
        'compaction_efficiency': 'Compaction efficiency',
        'permeability_ratio': 'Permeability ratio',
        'connectivity_gain': 'Connectivity gain',
        'bridge_saturation': 'Bridge saturation',
    }

    available = [k for k in response_keys
                 if any(k in r['responses'] for r in block1)]
    if not available:
        return

    # ---- Main effects plot ----
    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for ri, resp_key in enumerate(available):
        ax = axes[ri]
        effects = {}
        for f in factors:
            hi = [r['responses'].get(resp_key, np.nan) for r in block1
                  if r['factors'].get(f) == 1]
            lo = [r['responses'].get(resp_key, np.nan) for r in block1
                  if r['factors'].get(f) == -1]
            hi = [v for v in hi if np.isfinite(v)]
            lo = [v for v in lo if np.isfinite(v)]
            if hi and lo:
                effects[f] = np.mean(hi) - np.mean(lo)
            else:
                effects[f] = 0.0

        sorted_f = sorted(effects.items(), key=lambda x: abs(x[1]), reverse=True)
        labels = [f for f, _ in sorted_f]
        values = [v for _, v in sorted_f]
        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]

        ax.barh(range(len(labels)), values, color=colors, edgecolor='white',
                linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel('Effect magnitude', fontsize=9)
        ax.set_title(response_labels.get(resp_key, resp_key), fontsize=11,
                     fontweight='bold')

    fig.suptitle('Main Effects on Dimensionless Responses (Block 1)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'factor_main_effects.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # ---- Interaction plot (for top response: compaction_efficiency) ----
    top_resp = available[0]
    n_f = len(factors)
    fig, axes = plt.subplots(n_f - 1, n_f - 1, figsize=(3.5 * (n_f - 1),
                                                          3.5 * (n_f - 1)))
    if n_f - 1 == 1:
        axes = np.array([[axes]])

    plot_idx = 0
    for i in range(n_f):
        for j in range(i + 1, n_f):
            fi, fj = factors[i], factors[j]
            r_idx = j - 1
            c_idx = i
            if r_idx >= axes.shape[0] or c_idx >= axes.shape[1]:
                continue
            ax = axes[r_idx, c_idx]

            for fi_level, color, marker in [(-1, '#3498db', 'o'), (1, '#e74c3c', 's')]:
                means = []
                for fj_level in (-1, 1):
                    vals = [r['responses'].get(top_resp, np.nan) for r in block1
                            if r['factors'].get(fi) == fi_level
                            and r['factors'].get(fj) == fj_level]
                    vals = [v for v in vals if np.isfinite(v)]
                    means.append(np.mean(vals) if vals else np.nan)
                label = f'{fi}={"+" if fi_level == 1 else "-"}'
                ax.plot([-1, 1], means, f'{marker}-', color=color,
                        label=label, markersize=6, linewidth=1.5)

            ax.set_xticks([-1, 1])
            ax.set_xticklabels([f'{fj}-', f'{fj}+'], fontsize=8)
            ax.set_title(f'{fi} x {fj}', fontsize=9)
            if plot_idx == 0:
                ax.legend(fontsize=7)
            plot_idx += 1

    # Hide unused axes
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            if j > i:
                axes[i, j].set_visible(False)

    fig.suptitle(f'2-Factor Interactions -- {response_labels.get(top_resp, top_resp)} (Block 1)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'factor_interactions.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 6: Dimensionless dashboard (3x2 summary)
# ======================================================================

def plot_dimensionless_dashboard(data, outdir):
    """3x2 summary dashboard of dimensionless groups vs responses."""
    data = _enrich_runs(data)
    runs = data['runs']

    fig, axes = plt.subplots(3, 2, figsize=(14, 17))

    panels = [
        ('beta', 'compaction_efficiency',
         r'$\beta$', 'Compaction efficiency', 'viridis'),
        ('Ca', 'compaction_efficiency',
         'Ca', 'Compaction efficiency', 'plasma'),
        ('beta', 'permeability_ratio',
         r'$\beta$', r'$K_{final}/K_{initial}$', 'viridis'),
        ('jamming_proximity', 'tissue_frac_final',
         r'$\phi_{solid}/\phi_J$', 'Tissue fraction (final)', 'magma'),
        ('timescale_ratio', 'bridge_saturation',
         r'$\tau_{bridge}/\tau_{damp}$', 'Bridge saturation', 'cividis'),
        ('phi_ratio', 'connectivity_gain',
         r'$\phi_f/\phi_i$', 'Connectivity gain', 'viridis'),
    ]

    for idx, (x_key, y_key, xlabel, ylabel, cmap_name) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        cmap = plt.cm.get_cmap(cmap_name)

        x_vals, y_vals, c_vals = [], [], []
        for run in runs:
            xv = run['groups'].get(x_key, None)
            yv = run['responses'].get(y_key, None)
            if xv is None or yv is None:
                continue
            if not np.isfinite(xv) or not np.isfinite(yv):
                continue
            x_vals.append(xv)
            y_vals.append(yv)
            c_vals.append(run['groups']['beta'])

        if not x_vals:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='gray')
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            continue

        norm = Normalize(vmin=min(c_vals), vmax=max(c_vals))
        sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap=cmap, norm=norm,
                        s=60, edgecolors='white', linewidths=0.5, zorder=3)

        # Best-fit line
        fit = _draw_fit_line(ax, x_vals, y_vals, fit_type='linear')
        _annotate_fit(ax, fit, fit_type='linear', loc='upper left')

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)

        # Add horizontal reference lines where appropriate
        if y_key == 'permeability_ratio':
            ax.axhline(1.0, ls=':', color='gray', lw=0.8, alpha=0.5)
        if y_key == 'connectivity_gain':
            ax.axhline(0.0, ls=':', color='gray', lw=0.8, alpha=0.5)

    fig.suptitle('Dimensionless Groups Dashboard', fontsize=15, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'dimensionless_dashboard.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Plot 7: Phase space (beta vs Ca, colored by compaction)
# ======================================================================

def plot_phase_space(data, outdir):
    """2D phase space: beta (x) vs Ca (y), colored by compaction efficiency.

    Includes contour overlay from interpolation and regime annotations.
    """
    data = _enrich_runs(data)
    runs = data['runs']

    fig, ax = plt.subplots(figsize=(10, 8))

    x_vals, y_vals, c_vals = [], [], []
    for run in runs:
        beta = run['groups']['beta']
        ca = run['groups']['Ca']
        comp = run['responses'].get('compaction_efficiency', None)
        if comp is not None and np.isfinite(beta) and np.isfinite(ca) and np.isfinite(comp):
            x_vals.append(beta)
            y_vals.append(ca)
            c_vals.append(comp)

    if len(x_vals) < 3:
        print("    Skipping phase_space: too few data points")
        plt.close(fig)
        return

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    c_vals = np.array(c_vals)

    # Scatter plot
    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap='RdYlGn', s=100,
                    edgecolors='black', linewidths=0.8, zorder=5)
    plt.colorbar(sc, ax=ax, label='Compaction efficiency', fraction=0.046, pad=0.04)

    # Contour interpolation (if scipy available)
    try:
        from scipy.interpolate import griddata
        xi = np.linspace(x_vals.min() * 0.95, x_vals.max() * 1.05, 80)
        yi = np.linspace(y_vals.min() * 0.95, y_vals.max() * 1.05, 80)
        xg, yg = np.meshgrid(xi, yi)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zg = griddata((x_vals, y_vals), c_vals, (xg, yg),
                          method='linear', fill_value=np.nan)
        # Only draw contours if we have enough non-NaN values
        if np.count_nonzero(~np.isnan(zg)) > 10:
            cs = ax.contourf(xg, yg, zg, levels=15, cmap='RdYlGn', alpha=0.25,
                             zorder=1)
            ax.contour(xg, yg, zg, levels=8, colors='gray', linewidths=0.3,
                       alpha=0.5, zorder=2)
    except ImportError:
        pass

    # Annotate run IDs next to each point
    for run in runs:
        beta = run['groups']['beta']
        ca = run['groups']['Ca']
        comp = run['responses'].get('compaction_efficiency', None)
        if comp is not None and np.isfinite(beta) and np.isfinite(ca):
            ax.annotate(str(run['run_id']), (beta, ca),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=7, color='#333333', zorder=6)

    # Regime labels
    x_mid = (x_vals.min() + x_vals.max()) / 2
    y_mid = (y_vals.min() + y_vals.max()) / 2
    ax.text(x_vals.min() * 0.97, y_vals.max() * 0.97,
            'High stress\nLow engagement',
            fontsize=8, ha='left', va='top', color='#666666', style='italic')
    ax.text(x_vals.max() * 1.02, y_vals.max() * 0.97,
            'High stress\nHigh engagement',
            fontsize=8, ha='right', va='top', color='#666666', style='italic')
    ax.text(x_vals.min() * 0.97, y_vals.min() * 1.03,
            'Low stress\nLow engagement',
            fontsize=8, ha='left', va='bottom', color='#666666', style='italic')
    ax.text(x_vals.max() * 1.02, y_vals.min() * 1.03,
            'Low stress\nHigh engagement',
            fontsize=8, ha='right', va='bottom', color='#666666', style='italic')

    ax.set_xlabel(r'$\beta$ (motor-clutch engagement)', fontsize=11)
    ax.set_ylabel('Ca (cellular capillary number)', fontsize=11)
    ax.set_title(r'Phase Space: $\beta$ vs Ca, colored by compaction efficiency',
                 fontsize=13)

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase_space.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Summary table (text output)
# ======================================================================

def _write_summary_table(data, outdir):
    """Write a text summary of dimensionless groups and responses."""
    data = _enrich_runs(data)
    runs = data['runs']

    path = os.path.join(outdir, 'dimensionless_summary.txt')
    with open(path, 'w') as f:
        f.write("GELS Dimensionless Analysis Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Runs analysed: {len(runs)} / 24\n\n")

        # Header
        hdr = (f"{'Run':>6} {'beta':>7} {'Ca':>10} {'phi_r':>7} {'R_r':>6} "
               f"{'phi/phiJ':>8} {'T_ratio':>8}  "
               f"{'Comp_eff':>9} {'K_ratio':>8} {'Conn_g':>7} {'Brdg_s':>7}")
        f.write(hdr + "\n")
        f.write("-" * len(hdr) + "\n")

        for run in runs:
            g = run['groups']
            r = run['responses']
            phi_r_str = f"{g['phi_ratio']:.2f}" if np.isfinite(g['phi_ratio']) else "inf"
            t_r_str = f"{g['timescale_ratio']:.1f}" if np.isfinite(g['timescale_ratio']) else "inf"
            f.write(
                f"{run['name']:>6} "
                f"{g['beta']:7.4f} "
                f"{g['Ca']:10.4f} "
                f"{phi_r_str:>7} "
                f"{g['R_ratio']:6.2f} "
                f"{g['jamming_proximity']:8.4f} "
                f"{t_r_str:>8}  "
                f"{r.get('compaction_efficiency', 0):9.4f} "
                f"{r.get('permeability_ratio', 0):8.4f} "
                f"{r.get('connectivity_gain', 0):7.4f} "
                f"{r.get('bridge_saturation', 0):7.4f}\n"
            )

        f.write("\n\nDIMENSIONLESS GROUP DEFINITIONS\n")
        f.write("-" * 70 + "\n")
        f.write("  beta  = k_sub / (k_sub + k_opt)         Motor-clutch engagement (0-1)\n")
        f.write("  Ca    = sigma_cell / E*                  Cellular capillary number\n")
        f.write("  phi_r = phi_f / phi_i                    Composition ratio\n")
        f.write("  R_r   = R_func / R_inert                 Size ratio\n")
        f.write("  phi/phiJ = (phi_f+phi_i) / phi_RCP       Jamming proximity\n")
        f.write("  T_ratio = tau_bridge / tau_damp           Timescale ratio (bio/mech)\n")
        f.write("\n  where:\n")
        f.write("    k_sub = pi * E * (d_cell/2) / (1-nu^2)     substrate stiffness\n")
        f.write("    k_opt = n_clutches * k_clutch               clutch ensemble stiffness\n")
        f.write("    sigma_cell = n_cells * F_cell / (pi*R^2)    cell stress\n")
        f.write("    E* = E / (2*(1-nu^2))                       effective Hertzian modulus\n")

    print(f"  Summary table: {path}")


# ======================================================================
# Orchestrator
# ======================================================================

MODULES = OrderedDict([
    ('beta_collapse',       ('Beta collapse',           plot_beta_collapse)),
    ('Ca_scaling',          ('Ca scaling',              plot_Ca_scaling)),
    ('jamming',             ('Jamming diagram',         plot_jamming_diagram)),
    ('composition',         ('Composition effects',     plot_composition_effects)),
    ('factor_effects',      ('Factor effects (DOE)',    plot_factor_effects)),
    ('dashboard',           ('Dimensionless dashboard', plot_dimensionless_dashboard)),
    ('phase_space',         ('Phase space',             plot_phase_space)),
])


def run_all(data=None, scan_dir=None, outdir=None, skip=None):
    """Load data if not provided and generate all dimensionless analysis plots.

    Parameters
    ----------
    data : dict or None
        Output from load_doe_data(). If None, loads from scan_dir.
    scan_dir : str or None
        Directory to scan for DOE results. Used if data is None.
    outdir : str or None
        Output directory for plots. Defaults to <scan_dir>/dimensionless/.
    skip : set of str or None
        Module names to skip.
    """
    skip = set(skip or [])

    if data is None:
        if scan_dir is None:
            # Try common locations
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates = [
                os.path.join(base, 'Trials', 'trials'),
                os.path.join(base, 'results', 'trials'),
            ]
            for c in candidates:
                if os.path.isdir(c):
                    scan_dir = c
                    break
            if scan_dir is None:
                print("ERROR: No scan directory found. Use --scan-dir")
                return
        data = load_doe_data(scan_dir)

    if not data['runs']:
        print("ERROR: No DOE runs loaded. Check paths.")
        return

    if outdir is None:
        if scan_dir is not None:
            outdir = os.path.join(scan_dir, 'dimensionless')
        else:
            outdir = 'results/dimensionless'
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  GELS Dimensionless Analysis")
    print("=" * 65)
    print(f"  Runs: {len(data['runs'])}")
    print(f"  Output: {outdir}/")

    # Enrich all runs with dimensionless data
    data = _enrich_runs(data)

    # Print a quick summary of dimensionless group ranges
    betas = [r['groups']['beta'] for r in data['runs']]
    cas = [r['groups']['Ca'] for r in data['runs']]
    print(f"\n  beta range:  [{min(betas):.4f}, {max(betas):.4f}]")
    print(f"  Ca range:    [{min(cas):.4f}, {max(cas):.4f}]")

    # Write text summary
    _write_summary_table(data, outdir)

    # Run all plotting modules
    for i, (name, (label, func)) in enumerate(MODULES.items(), 1):
        if name in skip:
            print(f"  [{i}/{len(MODULES)}] {label}... SKIPPED")
            continue
        print(f"  [{i}/{len(MODULES)}] {label}...")
        try:
            func(data, outdir)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  All plots saved to: {outdir}/")
    print("=" * 65)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GELS dimensionless analysis: collapse DOE results by '
                    'key dimensionless groups (beta, Ca, jamming proximity, etc.).')
    parser.add_argument('--scan-dir', default=None,
                        help='Directory containing DOE_01/...DOE_24/ results '
                             '(default: Trials/trials/ or results/trials/)')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for plots (default: <scan-dir>/dimensionless/)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Analysis modules to skip (e.g., phase_space jamming)')
    parser.add_argument('--list-modules', action='store_true',
                        help='List available analysis modules')
    args = parser.parse_args()

    if args.list_modules:
        print("Available dimensionless analysis modules:")
        for name, (label, _) in MODULES.items():
            print(f"  {name:20s} -- {label}")
        sys.exit(0)

    run_all(scan_dir=args.scan_dir, outdir=args.output, skip=set(args.skip))
