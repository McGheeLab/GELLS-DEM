#!/usr/bin/env python3
"""
2D vs 3D DOE Comparison & Dimensional Scaling Analysis
=======================================================
Paired comparison of matched DOE runs (same LHS design, seed=2025)
in 2D and 3D modes.  Looks for dimensional scaling relationships.

Analysis modules:
    1. Paired scatter plots       — 2D vs 3D final metrics with 1:1 line + regression
    2. Dimensionless collapse     — normalize by Z_iso(d), φ_RCP(d), geometry
    3. Effect ranking comparison  — Pareto heatmap of factor importance in 2D vs 3D
    4. Time-evolution overlay     — paired trajectories for matched runs
    5. Scaling regression table   — fit 3D = α + β·2D for each metric
    6. Summary dashboard          — radar chart + key findings

Usage:
    python viz/doe_2d_vs_3d.py
    python viz/doe_2d_vs_3d.py --results-dir results/LHC --n-runs 20
    python viz/doe_2d_vs_3d.py --skip time_evolution

Or import:
    from viz.doe_2d_vs_3d import load_paired_data, run_all
    pairs = load_paired_data('results/LHC', n_runs=20)
    run_all(pairs, outdir='results/doe_2d_vs_3d')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import argparse
import csv
import json
import os
import sys
from collections import OrderedDict
from scipy import stats as sp_stats


# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

# Metrics to compare (key, display name, unit, higher-is-better)
METRICS = OrderedDict([
    ('func_lf',           ('Functional connectivity', '', True)),
    ('func_nc',           ('Functional clusters', '', False)),
    ('porosity',          ('Porosity', '', None)),
    ('K_kozeny_carman',   ('Permeability (K)', 'µm²', True)),
    ('n_bridges',         ('Active bridges', '', True)),
    ('compaction_ratio',  ('Compaction ratio', '', None)),
    ('F_mean',            ('Mean force', 'nN', None)),
    ('n_contacts',        ('Contact count', '', None)),
    ('disp_func',         ('Func displacement', 'µm', None)),
    ('mean_spread_frac',  ('Cell spread fraction', '', True)),
    ('phi_solid',         ('Solid fraction', '', None)),
    ('n_contacts_ff',     ('F-F contacts', '', None)),
])

# Dimensionless normalization factors
# Z_iso = 2d for frictionless spheres (Maxwell criterion)
Z_ISO = {2: 4.0, 3: 6.0}
# RCP (random close packing)
PHI_RCP = {2: 0.84, 3: 0.64}
# Percolation threshold (site, regular lattice)
P_PERC = {2: 0.593, 3: 0.312}

# Factor metadata (matches generate_doe_v2.py)
FACTOR_KEYS = ['E_modulus', 'func_ratio', 'cell_surface_coverage',
               'cell_sense_distance', 'R_func_mean', 'R_inert_mean']
FACTOR_LABELS = {
    'E_modulus': 'Stiffness (kPa)',
    'func_ratio': 'Functional ratio',
    'cell_surface_coverage': 'Cell coverage',
    'cell_sense_distance': 'Sensing dist (µm)',
    'R_func_mean': 'R_func (µm)',
    'R_inert_mean': 'R_inert (µm)',
}


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def _load_history_csv(path):
    """Load history.csv as list of dicts with float values."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_paired_data(results_dir, n_runs=20, trials_dir=None):
    """Load matched 2D/3D DOE pairs.

    Parameters
    ----------
    results_dir : str
        Directory containing DOE_2D_NNNN/ and DOE_3D_NNNN/ subdirectories.
    n_runs : int
        Number of runs per set (default 20).
    trials_dir : str or None
        Directory containing DOE_2D_NNNN.json / DOE_3D_NNNN.json configs.

    Returns
    -------
    pairs : list of dict
        Each has 'run_id', 'name_2d', 'name_3d', 'factors',
        'hist_2d', 'hist_3d', 'final_2d', 'final_3d',
        'complete_2d', 'complete_3d', 'params_2d', 'params_3d'
    """
    if trials_dir is None:
        trials_dir = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'Trials')

    pairs = []
    for i in range(1, n_runs + 1):
        name_2d = f"DOE_2D_{i:04d}"
        name_3d = f"DOE_3D_{i:04d}"

        # Load histories
        hist_2d = hist_3d = None
        for name, prefix in [(name_2d, '2D'), (name_3d, '3D')]:
            csv_path = os.path.join(results_dir, name, 'history.csv')
            if os.path.isfile(csv_path):
                h = _load_history_csv(csv_path)
                if prefix == '2D':
                    hist_2d = h
                else:
                    hist_3d = h

        if hist_2d is None or hist_3d is None:
            print(f"  WARNING: pair {i} — missing {'2D' if hist_2d is None else '3D'} data")
            continue

        # Load factor values from trial config (try 2D, then 3D, then params)
        factors = {}
        for try_name in [name_2d, name_3d]:
            config_path = os.path.join(trials_dir, f"{try_name}.json")
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    cfg = json.load(f)
                factors = cfg.get('_doe_factor_values', {})
                if factors:
                    break
                for k in FACTOR_KEYS:
                    if k not in factors and k in cfg:
                        factors[k] = cfg[k]
                if factors:
                    break
        # Fallback: read factor keys from params.json
        if not factors:
            for name in [name_2d, name_3d]:
                pp = os.path.join(results_dir, name, 'params.json')
                if os.path.isfile(pp):
                    with open(pp) as f:
                        pj = json.load(f)
                    for k in FACTOR_KEYS:
                        if k in pj:
                            factors[k] = pj[k]
                    if factors:
                        break

        # Load params
        params_2d = params_3d = {}
        for name, tag in [(name_2d, '2d'), (name_3d, '3d')]:
            pp = os.path.join(results_dir, name, 'params.json')
            if os.path.isfile(pp):
                with open(pp) as f:
                    if tag == '2d':
                        params_2d = json.load(f)
                    else:
                        params_3d = json.load(f)

        t_total = 48.0
        complete_2d = len(hist_2d) > 1 and hist_2d[-1]['time'] >= t_total - 1.0
        complete_3d = len(hist_3d) > 1 and hist_3d[-1]['time'] >= t_total - 1.0

        pairs.append({
            'run_id': i,
            'name_2d': name_2d,
            'name_3d': name_3d,
            'factors': factors,
            'hist_2d': hist_2d,
            'hist_3d': hist_3d,
            'final_2d': hist_2d[-1],
            'final_3d': hist_3d[-1],
            'initial_2d': hist_2d[0],
            'initial_3d': hist_3d[0],
            'complete_2d': complete_2d,
            'complete_3d': complete_3d,
            'params_2d': params_2d,
            'params_3d': params_3d,
        })

    n_complete = sum(1 for p in pairs if p['complete_2d'] and p['complete_3d'])
    print(f"\n  Loaded {len(pairs)} pairs ({n_complete} fully complete)")
    return pairs


# ══════════════════════════════════════════════════════════════════════
# 1. Paired scatter plots (2D vs 3D)
# ══════════════════════════════════════════════════════════════════════

def plot_paired_scatter(pairs, outdir):
    """Scatter plot: 2D value vs 3D value for each metric, with regression."""
    available = []
    for key in METRICS:
        v2 = [p['final_2d'].get(key) for p in pairs]
        v3 = [p['final_3d'].get(key) for p in pairs]
        if any(v is not None for v in v2) and any(v is not None for v in v3):
            available.append(key)

    n = len(available)
    if n == 0:
        return
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, key in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        name, unit, _ = METRICS[key]

        x_vals, y_vals = [], []
        colors = []
        for p in pairs:
            v2 = p['final_2d'].get(key)
            v3 = p['final_3d'].get(key)
            if v2 is not None and v3 is not None:
                x_vals.append(v2)
                y_vals.append(v3)
                # Color by completeness
                both_complete = p['complete_2d'] and p['complete_3d']
                colors.append('#2c3e50' if both_complete else '#e74c3c')

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        # Scatter
        for xi, yi, ci in zip(x_vals, y_vals, colors):
            ax.scatter(xi, yi, c=ci, s=40, edgecolors='white', linewidth=0.5,
                       zorder=3, alpha=0.85)

        # 1:1 line
        lims = [min(x_vals.min(), y_vals.min()),
                max(x_vals.max(), y_vals.max())]
        margin = 0.05 * (lims[1] - lims[0] + 1e-10)
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.5,
                label='1:1')

        # Linear regression
        if len(x_vals) >= 3 and np.std(x_vals) > 1e-10 and np.std(y_vals) > 1e-10:
            slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_vals, y_vals)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, '-', color='#e67e22',
                    linewidth=1.5, alpha=0.8)
            ax.text(0.05, 0.92, f'y = {slope:.2f}x + {intercept:.2g}\n'
                    f'R² = {r_val**2:.3f}, p = {p_val:.1e}',
                    transform=ax.transAxes, fontsize=7,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat',
                              alpha=0.5))

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel(f'2D {name}', fontsize=8)
        ax.set_ylabel(f'3D {name}', fontsize=8)
        ax.set_title(f'{name}' + (f' ({unit})' if unit else ''), fontsize=9,
                     fontweight='bold')
        ax.tick_params(labelsize=7)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('2D vs 3D — Paired Comparison (each point = matched LHS design)',
                 fontsize=13, y=1.02)
    # Legend for completeness
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2c3e50',
               markersize=8, label='Both complete'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e74c3c',
               markersize=8, label='3D partial'),
        Line2D([0], [0], linestyle='--', color='gray', label='1:1 line'),
        Line2D([0], [0], linestyle='-', color='#e67e22', label='OLS fit'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'paired_scatter.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 2. Dimensionless collapse
# ══════════════════════════════════════════════════════════════════════

def plot_dimensionless_collapse(pairs, outdir):
    """Normalize metrics by known dimensional factors and check collapse."""
    # Dimensionless transformations:
    # φ/φ_RCP(d), Z/Z_iso(d), p_c/p_perc(d), K/d_grain²
    transforms = OrderedDict([
        ('compaction_ratio_norm', {
            'source': 'phi_solid',
            'norm_2d': PHI_RCP[2], 'norm_3d': PHI_RCP[3],
            'label': 'φ_solid / φ_RCP(d)',
        }),
        ('contacts_per_Ziso', {
            'source': 'n_contacts',
            'compute': lambda p, dim: (
                p[f'final_{dim}'].get('n_contacts', 0) /
                max(p[f'final_{dim}'].get('n_granules_inner', 1), 1) /
                Z_ISO[int(dim[0])]
            ),
            'label': 'Z / Z_iso(d)',
        }),
        ('porosity_norm', {
            'source': 'porosity',
            'norm_2d': 1.0 - PHI_RCP[2], 'norm_3d': 1.0 - PHI_RCP[3],
            'label': 'ε / ε_RCP(d) = ε / (1 − φ_RCP)',
        }),
        ('K_normalized', {
            'source': 'K_kozeny_carman',
            'compute': lambda p, dim: (
                p[f'final_{dim}'].get('K_kozeny_carman', 0) /
                max(p[f'final_{dim}'].get('d_grain_mean', 1)**2, 1e-6)
            ),
            'label': 'K / d²_grain (Kozeny-Carman dimensionless)',
        }),
    ])

    n = len(transforms)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    for idx, (tkey, tdef) in enumerate(transforms.items()):
        ax = axes[idx]
        x_vals, y_vals = [], []
        colors = []

        for p in pairs:
            if 'compute' in tdef:
                try:
                    v2 = tdef['compute'](p, '2d')
                    v3 = tdef['compute'](p, '3d')
                except (KeyError, ZeroDivisionError):
                    continue
            else:
                src = tdef['source']
                v2_raw = p['final_2d'].get(src)
                v3_raw = p['final_3d'].get(src)
                if v2_raw is None or v3_raw is None:
                    continue
                v2 = v2_raw / tdef['norm_2d']
                v3 = v3_raw / tdef['norm_3d']

            if np.isfinite(v2) and np.isfinite(v3):
                x_vals.append(v2)
                y_vals.append(v3)
                both_complete = p['complete_2d'] and p['complete_3d']
                colors.append('#2c3e50' if both_complete else '#e74c3c')

        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)

        if len(x_vals) == 0:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                    ha='center', fontsize=10)
            ax.set_title(tdef['label'], fontsize=9, fontweight='bold')
            continue

        for xi, yi, ci in zip(x_vals, y_vals, colors):
            ax.scatter(xi, yi, c=ci, s=45, edgecolors='white', linewidth=0.5,
                       zorder=3, alpha=0.85)

        # 1:1 line
        lims = [min(x_vals.min(), y_vals.min()),
                max(x_vals.max(), y_vals.max())]
        margin = 0.05 * (lims[1] - lims[0] + 1e-10)
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.5)

        # Regression
        if len(x_vals) >= 3:
            slope, intercept, r_val, _, _ = sp_stats.linregress(x_vals, y_vals)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, '-', color='#e67e22',
                    linewidth=1.5, alpha=0.8)
            ax.text(0.05, 0.92,
                    f'slope = {slope:.3f}\nR² = {r_val**2:.3f}',
                    transform=ax.transAxes, fontsize=8,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.6))
            # Annotate if near 1:1
            if abs(slope - 1.0) < 0.15 and r_val**2 > 0.7:
                ax.text(0.95, 0.05, 'COLLAPSE', transform=ax.transAxes,
                        fontsize=10, fontweight='bold', color='green',
                        ha='right', va='bottom', alpha=0.7)

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')
        ax.set_xlabel('2D (normalized)', fontsize=9)
        ax.set_ylabel('3D (normalized)', fontsize=9)
        ax.set_title(tdef['label'], fontsize=9, fontweight='bold')
        ax.tick_params(labelsize=7)

    fig.suptitle('Dimensionless Collapse — 2D vs 3D\n'
                 '(slope ≈ 1.0 + high R² = predictive 2D model)',
                 fontsize=12, y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'dimensionless_collapse.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 3. Effect ranking comparison (factor importance heatmap)
# ══════════════════════════════════════════════════════════════════════

def _compute_factor_effects(pairs, metric_key, dim='2d'):
    """Compute linear effect of each factor on a metric using correlation."""
    values = []
    factor_matrix = []
    for p in pairs:
        val = p[f'final_{dim}'].get(metric_key)
        if val is None:
            continue
        fv = [p['factors'].get(k, np.nan) for k in FACTOR_KEYS]
        if any(np.isnan(f) for f in fv):
            continue
        values.append(val)
        factor_matrix.append(fv)

    if len(values) < 5:
        return {k: 0.0 for k in FACTOR_KEYS}

    values = np.array(values)
    factor_matrix = np.array(factor_matrix)

    # Standardize factors to [-1,1] for comparable effect sizes
    effects = {}
    for j, k in enumerate(FACTOR_KEYS):
        fj = factor_matrix[:, j]
        if np.std(fj) > 0:
            fj_std = (fj - np.mean(fj)) / np.std(fj)
            if np.std(values) > 0:
                r, _ = sp_stats.pearsonr(fj_std, values)
                effects[k] = r
            else:
                effects[k] = 0.0
        else:
            effects[k] = 0.0

    return effects


def plot_effect_comparison(pairs, outdir):
    """Side-by-side heatmap of factor importance in 2D vs 3D."""
    key_metrics = ['func_lf', 'porosity', 'K_kozeny_carman', 'n_bridges',
                   'compaction_ratio', 'F_mean', 'n_contacts', 'mean_spread_frac']
    available = [k for k in key_metrics
                 if any(k in p['final_2d'] for p in pairs)]

    n_metrics = len(available)
    n_factors = len(FACTOR_KEYS)

    # Build effect matrices
    eff_2d = np.zeros((n_metrics, n_factors))
    eff_3d = np.zeros((n_metrics, n_factors))

    for mi, mkey in enumerate(available):
        e2 = _compute_factor_effects(pairs, mkey, '2d')
        e3 = _compute_factor_effects(pairs, mkey, '3d')
        for fi, fkey in enumerate(FACTOR_KEYS):
            eff_2d[mi, fi] = e2.get(fkey, 0)
            eff_3d[mi, fi] = e3.get(fkey, 0)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
        figsize=(18, max(4, n_metrics * 0.6 + 1)),
        gridspec_kw={'width_ratios': [1, 1, 0.8]})

    vmax = max(np.abs(eff_2d).max(), np.abs(eff_3d).max(), 0.1)
    metric_labels = [METRICS[k][0] for k in available]
    factor_labels = [FACTOR_LABELS[k] for k in FACTOR_KEYS]

    for ax, eff, title in [(ax1, eff_2d, '2D Effects (Pearson r)'),
                           (ax2, eff_3d, '3D Effects (Pearson r)')]:
        im = ax.imshow(eff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        ax.set_xticks(range(n_factors))
        ax.set_xticklabels(factor_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(n_metrics))
        ax.set_yticklabels(metric_labels, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')

        for mi in range(n_metrics):
            for fi in range(n_factors):
                val = eff[mi, fi]
                color = 'white' if abs(val) > 0.5 * vmax else 'black'
                ax.text(fi, mi, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    # Difference panel
    diff = eff_3d - eff_2d
    vdiff = max(np.abs(diff).max(), 0.1)
    im3 = ax3.imshow(diff, cmap='PiYG', vmin=-vdiff, vmax=vdiff, aspect='auto')
    ax3.set_xticks(range(n_factors))
    ax3.set_xticklabels(factor_labels, rotation=45, ha='right', fontsize=8)
    ax3.set_yticks(range(n_metrics))
    ax3.set_yticklabels([], fontsize=8)
    ax3.set_title('Δ (3D − 2D)', fontsize=11, fontweight='bold')
    for mi in range(n_metrics):
        for fi in range(n_factors):
            val = diff[mi, fi]
            color = 'white' if abs(val) > 0.5 * vdiff else 'black'
            ax3.text(fi, mi, f'{val:+.2f}', ha='center', va='center',
                     fontsize=7, color=color)

    plt.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, label='Pearson r')
    plt.colorbar(im3, ax=ax3, fraction=0.04, pad=0.02, label='Δr')

    fig.suptitle('Factor Importance — 2D vs 3D\n'
                 '(Pearson correlation between factor and final metric value)',
                 fontsize=13, y=1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'effect_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 4. Time-evolution overlay
# ══════════════════════════════════════════════════════════════════════

def plot_time_evolution(pairs, outdir):
    """Overlay 2D and 3D time traces for key metrics."""
    metrics = ['func_lf', 'porosity', 'n_bridges', 'compaction_ratio',
               'mean_spread_frac', 'F_mean']
    metric_names = {
        'func_lf': 'Functional connectivity',
        'porosity': 'Porosity',
        'n_bridges': 'Active bridges',
        'compaction_ratio': 'Compaction ratio',
        'mean_spread_frac': 'Cell spread fraction',
        'F_mean': 'Mean force (nN)',
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for mi, mkey in enumerate(metrics):
        ax = axes_flat[mi]

        # Individual traces (light)
        for p in pairs:
            t2 = [h['time'] for h in p['hist_2d']]
            v2 = [h.get(mkey, np.nan) for h in p['hist_2d']]
            t3 = [h['time'] for h in p['hist_3d']]
            v3 = [h.get(mkey, np.nan) for h in p['hist_3d']]

            ax.plot(t2, v2, color='#3498db', alpha=0.15, linewidth=0.6)
            ax.plot(t3, v3, color='#e74c3c', alpha=0.15, linewidth=0.6)

        # Mean traces
        for dim, color, label_str in [('2d', '#3498db', '2D mean'),
                                       ('3d', '#e74c3c', '3D mean')]:
            # Interpolate all traces to common time grid
            t_common = np.arange(0, 49, 1.0)
            interp_vals = []
            for p in pairs:
                hist = p[f'hist_{dim}']
                times = np.array([h['time'] for h in hist])
                vals = np.array([h.get(mkey, np.nan) for h in hist])
                valid = ~np.isnan(vals)
                if valid.sum() < 2:
                    continue
                interp = np.interp(t_common, times[valid], vals[valid],
                                   left=np.nan, right=np.nan)
                interp_vals.append(interp)

            if interp_vals:
                interp_vals = np.array(interp_vals)
                mean_v = np.nanmean(interp_vals, axis=0)
                std_v = np.nanstd(interp_vals, axis=0)
                valid_t = ~np.isnan(mean_v)
                ax.plot(t_common[valid_t], mean_v[valid_t],
                        color=color, linewidth=2.5, label=label_str, zorder=5)
                ax.fill_between(t_common[valid_t],
                                mean_v[valid_t] - std_v[valid_t],
                                mean_v[valid_t] + std_v[valid_t],
                                color=color, alpha=0.15)

        ax.set_xlabel('Time (h)', fontsize=9)
        ax.set_ylabel(metric_names.get(mkey, mkey), fontsize=9)
        ax.set_title(metric_names.get(mkey, mkey), fontsize=10, fontweight='bold')
        if mi == 0:
            ax.legend(fontsize=9)
        ax.set_xlim(0, 48)

    fig.suptitle('Time Evolution — 2D (blue) vs 3D (red)\n'
                 '(individual traces + mean ± σ)',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'time_evolution_overlay.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 5. Scaling regression table
# ══════════════════════════════════════════════════════════════════════

def compute_scaling_table(pairs, outdir):
    """Fit linear regression 3D = α + β·2D for each metric. Save as CSV + plot."""
    rows = []
    for key, (name, unit, _) in METRICS.items():
        x_vals, y_vals = [], []
        for p in pairs:
            v2 = p['final_2d'].get(key)
            v3 = p['final_3d'].get(key)
            if v2 is not None and v3 is not None and np.isfinite(v2) and np.isfinite(v3):
                x_vals.append(v2)
                y_vals.append(v3)

        if len(x_vals) < 5:
            continue

        x = np.array(x_vals)
        y = np.array(y_vals)
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            continue
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x, y)

        # Ratio statistics (3D / 2D)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = y / np.where(np.abs(x) > 1e-10, x, np.nan)
            ratios = ratios[np.isfinite(ratios)]

        ratio_mean = np.mean(ratios) if len(ratios) > 0 else np.nan
        ratio_std = np.std(ratios) if len(ratios) > 1 else np.nan

        rows.append({
            'metric': key,
            'name': name,
            'n_pairs': len(x_vals),
            'slope': slope,
            'intercept': intercept,
            'R2': r_val**2,
            'p_value': p_val,
            'std_err': std_err,
            'ratio_3d_2d_mean': ratio_mean,
            'ratio_3d_2d_std': ratio_std,
            'mean_2d': np.mean(x),
            'mean_3d': np.mean(y),
        })

    # Save CSV
    csv_path = os.path.join(outdir, 'scaling_table.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"  Scaling table: {csv_path}")

    # Print text summary
    txt_path = os.path.join(outdir, 'scaling_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("2D vs 3D Scaling Relationships\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"{'Metric':<25} {'slope':>7} {'intcpt':>8} {'R²':>6} "
                f"{'p-val':>9} {'3D/2D':>8} {'n':>4}\n")
        f.write("-" * 75 + "\n")
        for r in rows:
            f.write(f"{r['name']:<25} {r['slope']:>7.3f} {r['intercept']:>8.2g} "
                    f"{r['R2']:>6.3f} {r['p_value']:>9.1e} "
                    f"{r['ratio_3d_2d_mean']:>7.2f}± {r['ratio_3d_2d_std']:.2f}"
                    f" {r['n_pairs']:>4d}\n")

        f.write("\n\nINTERPRETATION\n")
        f.write("-" * 75 + "\n")
        good_collapse = [r for r in rows if r['R2'] > 0.7 and abs(r['slope'] - 1.0) < 0.3]
        scale_shift = [r for r in rows if r['R2'] > 0.7 and abs(r['slope'] - 1.0) >= 0.3]
        poor = [r for r in rows if r['R2'] <= 0.7]

        if good_collapse:
            f.write("\nGOOD COLLAPSE (slope ≈ 1, R² > 0.7):\n")
            for r in good_collapse:
                f.write(f"  {r['name']}: 3D ≈ {r['slope']:.2f}×2D + {r['intercept']:.2g}"
                        f" (R²={r['R2']:.3f})\n")
            f.write("  → 2D model is directly predictive for these metrics.\n")

        if scale_shift:
            f.write("\nSYSTEMATIC SCALING (slope ≠ 1, R² > 0.7):\n")
            for r in scale_shift:
                f.write(f"  {r['name']}: 3D ≈ {r['slope']:.2f}×2D + {r['intercept']:.2g}"
                        f" (R²={r['R2']:.3f}, 3D/2D={r['ratio_3d_2d_mean']:.2f})\n")
            f.write("  → Predictable 2D→3D correction exists.\n")

        if poor:
            f.write("\nPOOR CORRELATION (R² ≤ 0.7):\n")
            for r in poor:
                f.write(f"  {r['name']}: R²={r['R2']:.3f}\n")
            f.write("  → 3D physics differs qualitatively; 2D not predictive.\n")

        # Known theoretical predictions
        f.write("\n\nTHEORETICAL REFERENCE\n")
        f.write("-" * 75 + "\n")
        f.write(f"  Maxwell isostatic Z:  2D = {Z_ISO[2]},  3D = {Z_ISO[3]}  "
                f"(ratio = {Z_ISO[3]/Z_ISO[2]:.2f})\n")
        f.write(f"  RCP packing:          2D ≈ {PHI_RCP[2]},  3D ≈ {PHI_RCP[3]}  "
                f"(ratio = {PHI_RCP[3]/PHI_RCP[2]:.2f})\n")
        f.write(f"  Site percolation:     2D ≈ {P_PERC[2]},  3D ≈ {P_PERC[3]}  "
                f"(ratio = {P_PERC[3]/P_PERC[2]:.2f})\n")
        f.write(f"  Delesse principle:    area fraction = volume fraction (exact)\n")

    print(f"  Scaling summary: {txt_path}")

    # Bar chart of R² values
    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r['name'] for r in rows]
        r2s = [r['R2'] for r in rows]
        colors = ['#2ecc71' if r2 > 0.7 else '#f39c12' if r2 > 0.4 else '#e74c3c'
                  for r2 in r2s]
        bars = ax.barh(range(len(names)), r2s, color=colors, edgecolor='white',
                       linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('R² (2D→3D linear regression)', fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.text(0.71, len(names) - 0.5, 'R²=0.7', fontsize=8, color='gray')

        # Annotate with slope
        for i, r in enumerate(rows):
            ax.text(r['R2'] + 0.02, i, f'β={r["slope"]:.2f}', fontsize=7,
                    va='center')

        ax.invert_yaxis()
        ax.set_title('2D→3D Predictability — Linear Regression R²\n'
                     '(green: good collapse, orange: some signal, red: poor)',
                     fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, 'scaling_r2.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    return rows


# ══════════════════════════════════════════════════════════════════════
# 6. Summary dashboard
# ══════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(pairs, outdir):
    """Radar chart of mean 2D vs 3D values across key metrics."""
    key_metrics = ['func_lf', 'porosity', 'n_bridges', 'compaction_ratio',
                   'mean_spread_frac']
    available = [k for k in key_metrics
                 if any(k in p['final_2d'] for p in pairs)]

    if len(available) < 3:
        return

    # Compute means and normalize to [0, 1] for radar
    means_2d = []
    means_3d = []
    labels = []
    for k in available:
        vals_2d = [p['final_2d'].get(k, np.nan) for p in pairs]
        vals_3d = [p['final_3d'].get(k, np.nan) for p in pairs]
        m2 = np.nanmean(vals_2d)
        m3 = np.nanmean(vals_3d)
        means_2d.append(m2)
        means_3d.append(m3)
        labels.append(METRICS[k][0])

    # Normalize to max of 2D and 3D for each metric
    means_2d = np.array(means_2d)
    means_3d = np.array(means_3d)
    max_vals = np.maximum(np.abs(means_2d), np.abs(means_3d))
    max_vals[max_vals == 0] = 1
    norm_2d = means_2d / max_vals
    norm_3d = means_3d / max_vals

    # Radar chart
    N = len(available)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    vals_2d_r = norm_2d.tolist() + [norm_2d[0]]
    vals_3d_r = norm_3d.tolist() + [norm_3d[0]]

    ax.plot(angles, vals_2d_r, 'o-', color='#3498db', linewidth=2, label='2D')
    ax.fill(angles, vals_2d_r, color='#3498db', alpha=0.15)
    ax.plot(angles, vals_3d_r, 's-', color='#e74c3c', linewidth=2, label='3D')
    ax.fill(angles, vals_3d_r, color='#e74c3c', alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    # Annotate with raw values
    for i, (l, m2, m3) in enumerate(zip(labels, means_2d, means_3d)):
        angle = angles[i]
        ax.text(angle, 1.15, f'2D:{m2:.2g}\n3D:{m3:.2g}', fontsize=7,
                ha='center', va='bottom')

    ax.set_title('2D vs 3D — Mean Response Summary\n(normalized to max)',
                 fontsize=12, y=1.15)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary_radar.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 7. 3D/2D ratio vs factor values
# ══════════════════════════════════════════════════════════════════════

def plot_ratio_vs_factors(pairs, outdir):
    """For key metrics, plot the 3D/2D ratio against each factor to find
    parameter-dependent scaling."""
    key_metrics = ['func_lf', 'n_bridges', 'n_contacts', 'porosity',
                   'compaction_ratio', 'K_kozeny_carman']
    available = [k for k in key_metrics
                 if any(k in p['final_2d'] for p in pairs)]

    for mkey in available:
        name = METRICS[mkey][0]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes_flat = axes.flatten()

        for fi, fkey in enumerate(FACTOR_KEYS):
            ax = axes_flat[fi]
            x_vals, ratios = [], []
            for p in pairs:
                v2 = p['final_2d'].get(mkey)
                v3 = p['final_3d'].get(mkey)
                fval = p['factors'].get(fkey)
                if v2 is not None and v3 is not None and fval is not None:
                    if abs(v2) > 1e-10:
                        x_vals.append(fval)
                        ratios.append(v3 / v2)

            if len(x_vals) < 3:
                ax.text(0.5, 0.5, 'Insufficient data',
                        transform=ax.transAxes, ha='center')
                ax.set_title(FACTOR_LABELS[fkey], fontsize=9)
                continue

            x_vals = np.array(x_vals)
            ratios = np.array(ratios)

            ax.scatter(x_vals, ratios, c='#2c3e50', s=40, edgecolors='white',
                       linewidth=0.5, alpha=0.85)
            ax.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

            # Regression
            slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_vals, ratios)
            x_fit = np.linspace(x_vals.min(), x_vals.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, '-', color='#e67e22',
                    linewidth=1.5, alpha=0.8)
            sig = '*' if p_val < 0.05 else ''
            ax.text(0.05, 0.92,
                    f'r = {r_val:.2f}{sig}\nslope = {slope:.3g}',
                    transform=ax.transAxes, fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.6))

            ax.set_xlabel(FACTOR_LABELS[fkey], fontsize=9)
            ax.set_ylabel(f'3D/2D ratio', fontsize=9)
            ax.set_title(FACTOR_LABELS[fkey], fontsize=9, fontweight='bold')

        fig.suptitle(f'3D/2D Ratio vs Factor Values — {name}\n'
                     f'(ratio = 1 → same in both dims; '
                     f'slope ≠ 0 → factor-dependent scaling)',
                     fontsize=12, y=1.04)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f'ratio_vs_factors_{mkey}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 8. Physics-informed transfer function
# ══════════════════════════════════════════════════════════════════════

def _compute_correction_features(p):
    """Compute a priori dimensional correction features for a single pair.

    These are quantities we can compute from the params alone (no simulation
    needed) that capture the known physics of 2D→3D differences.

    Returns dict of named features.
    """
    import math
    p2 = p['params_2d']
    p3 = p['params_3d']
    fv = p['factors']

    R_f = fv.get('R_func_mean', 60.0)
    R_i = fv.get('R_inert_mean', 60.0)
    func_ratio = fv.get('func_ratio', 0.5)
    phi_solid = 0.70  # fixed across DOE

    # --- Granule count ratio (N_3D / N_2D) ---
    # 2D: N ~ L² × φ / (π R²)
    # 3D: N ~ Lxy² × Lz × φ / (4/3 π R³)
    R_avg = func_ratio * R_f + (1 - func_ratio) * R_i
    Lx_2d = p2.get('Lx', 800)
    Ly_2d = p2.get('Ly', 800)
    Lx_3d = p3.get('Lx', 800)
    Ly_3d = p3.get('Ly', 800)
    Lz_3d = p3.get('Lz', 800)

    A_2d = Lx_2d * Ly_2d
    V_3d = Lx_3d * Ly_3d * Lz_3d
    N_2d_est = A_2d * phi_solid / (math.pi * R_avg**2)
    N_3d_est = V_3d * phi_solid / (4.0/3.0 * math.pi * R_avg**3)
    N_ratio = N_3d_est / max(N_2d_est, 1)

    # --- Coordination number ratio Z_3D/Z_2D ---
    # Isostatic: Z_iso = 2d. For frictional: somewhat lower.
    Z_ratio = Z_ISO[3] / Z_ISO[2]  # 1.5

    # --- Contact count ratio (N × Z / 2) ---
    contact_ratio = N_ratio * Z_ratio

    # --- Cell count ratio ---
    # 3D: cells per granule ~ 4πR² × coverage / A_cell
    # 2D: cells per granule ~ πR² × coverage / A_cell (actually 2πR × coverage)
    # But cell_surface_coverage uses actual geometry in the engine
    # Ratio of surface area: 4πR² / (2πR × cell_diameter) ≈ 2R/cell_diam
    cell_diam = 20.0  # fixed
    cell_h = 5.0
    A_cell_spread = math.pi * (cell_diam/2)**2  # spread cell footprint
    # 3D: n_cells_3d ~ 4πR² × cov / A_cell
    # 2D: n_cells_2d ~ πR² × cov / A_cell (uses area, not perimeter)
    # Ratio = 4
    cell_ratio_per_gran = 4.0  # 3D sphere surface / 2D disk area
    cell_ratio_total = cell_ratio_per_gran * N_ratio

    # --- Surface area per granule ratio ---
    # 3D: 4πR² (sphere), 2D: πR² (disk cross-section / perimeter approximation)
    surface_ratio = 4.0

    # --- Bridge probability correction ---
    # In 3D, bridges are 3D cones; sensing volume scales differently
    # Sensing volume: 3D ~ (4/3)π sense³, 2D ~ π sense²
    sense = fv.get('cell_sense_distance', 40.0)
    sense_vol_ratio = (4.0/3.0 * sense) / sense  # simplifies to 4/3

    # --- Percolation threshold correction ---
    # p_c(2D)/p_c(3D) ≈ 0.593/0.312 ≈ 1.9
    # Systems closer to p_c in 2D are further above it in 3D
    perc_advantage = P_PERC[2] / P_PERC[3]  # ~1.9

    # --- Slab aspect ratio (Lz / Lxy) ---
    slab_ratio = Lz_3d / max(Lx_3d, 1)

    return {
        'N_ratio': N_ratio,
        'Z_ratio': Z_ratio,
        'contact_ratio': contact_ratio,
        'cell_ratio_total': cell_ratio_total,
        'surface_ratio': surface_ratio,
        'perc_advantage': perc_advantage,
        'slab_ratio': slab_ratio,
        'log_N_ratio': math.log(max(N_ratio, 0.01)),
        'R_func': R_f,
        'R_inert': R_i,
        'func_ratio': func_ratio,
        'E_modulus': fv.get('E_modulus', 10.0),
        'cell_coverage': fv.get('cell_surface_coverage', 1.0),
        'sense_dist': fv.get('cell_sense_distance', 40.0),
    }


# Per-metric transfer function specifications:
# Each specifies which correction features to use as regressors
TRANSFER_SPECS = OrderedDict([
    ('func_lf', {
        'name': 'Functional connectivity',
        'features': ['log_N_ratio', 'func_ratio', 'sense_dist'],
        'log_transform': False,
        'description': 'Connectivity depends on extra neighbors in 3D + percolation threshold shift',
    }),
    ('func_nc', {
        'name': 'Functional clusters',
        'features': ['log_N_ratio', 'func_ratio'],
        'log_transform': True,
        'description': 'Cluster count scales with system size and percolation',
    }),
    ('n_bridges', {
        'name': 'Active bridges',
        'features': ['log_N_ratio', 'func_ratio', 'cell_coverage'],
        'log_transform': True,
        'description': 'Bridges scale with N × Z × surface area × cells',
    }),
    ('n_contacts', {
        'name': 'Contact count',
        'features': ['log_N_ratio', 'Z_ratio'],
        'log_transform': True,
        'description': 'Contacts = N × Z / 2, both scale with dimension',
    }),
    ('n_contacts_ff', {
        'name': 'F-F contacts',
        'features': ['log_N_ratio', 'func_ratio'],
        'log_transform': True,
        'description': 'Func-func contacts scale with N × Z × func_ratio²',
    }),
    ('F_mean', {
        'name': 'Mean force',
        'features': ['log_N_ratio', 'E_modulus', 'R_func'],
        'log_transform': True,
        'description': 'Force depends on confinement (more neighbors in 3D)',
    }),
    ('K_kozeny_carman', {
        'name': 'Permeability',
        'features': [],  # Already collapses — include as baseline
        'log_transform': False,
        'description': 'Kozeny-Carman is dimension-independent (already R²=0.94)',
    }),
    ('porosity', {
        'name': 'Porosity',
        'features': ['slab_ratio'],
        'log_transform': False,
        'description': 'Porosity set by φ_solid, minor slab geometry effects',
    }),
    ('compaction_ratio', {
        'name': 'Compaction ratio',
        'features': ['log_N_ratio', 'E_modulus'],
        'log_transform': False,
        'description': 'Compaction depends on force balance (more contacts in 3D)',
    }),
    ('disp_func', {
        'name': 'Functional displacement',
        'features': ['log_N_ratio', 'E_modulus'],
        'log_transform': False,
        'description': 'Displacement driven by force imbalance and mobility',
    }),
    ('mean_spread_frac', {
        'name': 'Cell spread fraction',
        'features': [],  # Intrinsic cell property, should be identical
        'log_transform': False,
        'description': 'Cell spreading is dimension-independent',
    }),
])


def _loo_r2(X, y):
    """Leave-one-out cross-validated R² for OLS.

    With n=20, LOO-CV gives an honest estimate of predictive power.
    """
    n = len(y)
    if n < 4:
        return np.nan, np.full(n, np.nan), np.full(X.shape[1], np.nan)

    y_pred_loo = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train, y_train = X[mask], y[mask]
        X_test = X[i:i+1]

        # OLS: β = (X'X)^-1 X'y
        try:
            beta = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
            y_pred_loo[i] = float(X[i] @ beta)
        except np.linalg.LinAlgError:
            y_pred_loo[i] = np.nan

    valid = ~np.isnan(y_pred_loo)
    if valid.sum() < 3:
        return np.nan, y_pred_loo, np.full(X.shape[1], np.nan)

    ss_res = np.sum((y[valid] - y_pred_loo[valid])**2)
    ss_tot = np.sum((y[valid] - np.mean(y[valid]))**2)
    r2_cv = 1.0 - ss_res / max(ss_tot, 1e-20)

    # Full-data coefficients for reporting
    beta_full = np.linalg.lstsq(X, y, rcond=None)[0]

    return r2_cv, y_pred_loo, beta_full


def fit_transfer_functions(pairs, outdir):
    """Fit physics-informed transfer functions: 3D = f(2D, correction_features).

    For each metric, fits:
        Model 0 (baseline):  3D = α + β·(2D)
        Model 1 (corrected): 3D = α + β·(2D) + Σ γ_k·feature_k

    Reports LOO-CV R² for both, and the improvement from the correction.
    """
    # Compute correction features for all pairs
    all_features = []
    for p in pairs:
        try:
            feats = _compute_correction_features(p)
            all_features.append(feats)
        except Exception:
            all_features.append(None)

    results = []

    for mkey, spec in TRANSFER_SPECS.items():
        name = spec['name']
        feat_keys = spec['features']
        use_log = spec['log_transform']

        # Collect data
        y_2d, y_3d, feat_matrix = [], [], []
        valid_idx = []
        for i, p in enumerate(pairs):
            v2 = p['final_2d'].get(mkey)
            v3 = p['final_3d'].get(mkey)
            if v2 is None or v3 is None or all_features[i] is None:
                continue
            if not (np.isfinite(v2) and np.isfinite(v3)):
                continue
            if use_log and (v2 <= 0 or v3 <= 0):
                continue
            y_2d.append(v2)
            y_3d.append(v3)
            feat_matrix.append([all_features[i].get(k, 0) for k in feat_keys])
            valid_idx.append(i)

        if len(y_2d) < 6:
            continue

        y_2d = np.array(y_2d)
        y_3d = np.array(y_3d)
        feat_matrix = np.array(feat_matrix) if feat_keys else np.empty((len(y_2d), 0))

        # Apply log transform if specified
        if use_log:
            y_2d_t = np.log(y_2d)
            y_3d_t = np.log(y_3d)
            if feat_matrix.shape[1] > 0:
                # Don't log-transform features that are already logs or ratios
                pass
        else:
            y_2d_t = y_2d
            y_3d_t = y_3d

        n = len(y_2d_t)

        # Model 0: 3D = α + β·(2D)
        X0 = np.column_stack([np.ones(n), y_2d_t])
        if np.std(y_2d_t) < 1e-10:
            r2_baseline = 0.0
            y_pred_0 = np.full(n, np.nan)
            beta_0 = np.array([np.mean(y_3d_t), 0.0])
        else:
            r2_baseline, y_pred_0, beta_0 = _loo_r2(X0, y_3d_t)

        # Model 1: 3D = α + β·(2D) + Σ γ_k·feature_k
        if feat_keys:
            # Standardize features for stable regression
            feat_means = np.mean(feat_matrix, axis=0)
            feat_stds = np.std(feat_matrix, axis=0)
            feat_stds[feat_stds < 1e-10] = 1.0
            feat_std = (feat_matrix - feat_means) / feat_stds

            X1 = np.column_stack([np.ones(n), y_2d_t, feat_std])
            r2_corrected, y_pred_1, beta_1 = _loo_r2(X1, y_3d_t)

            # Unstandardize coefficients for interpretability
            # β_feat_orig = β_feat_std / σ_feat
            # intercept absorbs the mean shifts
            beta_report = np.zeros(2 + len(feat_keys))
            beta_report[0] = beta_1[0]  # intercept (includes mean shifts)
            beta_report[1] = beta_1[1]  # 2D coefficient
            for j in range(len(feat_keys)):
                beta_report[2 + j] = beta_1[2 + j] / feat_stds[j]
                beta_report[0] -= beta_1[2 + j] * feat_means[j] / feat_stds[j]
        else:
            r2_corrected = r2_baseline
            y_pred_1 = y_pred_0
            beta_report = beta_0
            beta_1 = beta_0

        improvement = r2_corrected - r2_baseline if np.isfinite(r2_corrected) and np.isfinite(r2_baseline) else 0

        results.append({
            'metric': mkey,
            'name': name,
            'n': n,
            'features': feat_keys,
            'log_transform': use_log,
            'R2_baseline': r2_baseline,
            'R2_corrected': r2_corrected,
            'improvement': improvement,
            'beta': beta_report,
            'y_2d': y_2d,
            'y_3d': y_3d,
            'y_pred': y_pred_1 if feat_keys else y_pred_0,
            'use_log': use_log,
            'description': spec['description'],
        })

    # --- Plot: predicted vs actual ---
    n_res = len(results)
    ncols = min(4, n_res)
    nrows = (n_res + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    axes = np.atleast_2d(axes)

    for idx, res in enumerate(results):
        ax = axes[idx // ncols, idx % ncols]
        y_actual = np.log(res['y_3d']) if res['use_log'] else res['y_3d']
        y_pred = res['y_pred']

        valid = np.isfinite(y_pred)
        if valid.sum() < 2:
            ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                    ha='center')
            ax.set_title(res['name'], fontsize=9)
            continue

        ax.scatter(y_actual[valid], y_pred[valid], c='#2c3e50', s=45,
                   edgecolors='white', linewidth=0.5, zorder=3, alpha=0.85)
        lims = [min(y_actual[valid].min(), y_pred[valid].min()),
                max(y_actual[valid].max(), y_pred[valid].max())]
        margin = 0.05 * (lims[1] - lims[0] + 1e-10)
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.8)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal')

        label_y = f'{"log " if res["use_log"] else ""}3D (predicted, LOO-CV)'
        label_x = f'{"log " if res["use_log"] else ""}3D (actual)'
        ax.set_xlabel(label_x, fontsize=8)
        ax.set_ylabel(label_y, fontsize=8)

        r2b = res['R2_baseline']
        r2c = res['R2_corrected']
        imp = res['improvement']
        color = '#2ecc71' if r2c > 0.7 else '#f39c12' if r2c > 0.4 else '#e74c3c'
        ax.set_title(f'{res["name"]}\n'
                     f'R²_cv: {r2b:.2f} → {r2c:.2f} (+{imp:.2f})',
                     fontsize=8, fontweight='bold', color=color)
        ax.tick_params(labelsize=7)

    for idx in range(n_res, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Transfer Function: Predicted vs Actual 3D (LOO Cross-Validated)\n'
                 'Model: 3D = α + β·(2D) + Σ γ_k · correction_feature_k',
                 fontsize=12, y=1.04)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'transfer_function_predicted_vs_actual.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Plot: R² improvement bar chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in results]
    r2_base = [r['R2_baseline'] for r in results]
    r2_corr = [r['R2_corrected'] for r in results]

    y_pos = np.arange(len(names))
    bars1 = ax.barh(y_pos + 0.2, r2_base, 0.35, color='#bdc3c7', label='Baseline (2D only)')
    bars2 = ax.barh(y_pos - 0.2, r2_corr, 0.35, color='#2ecc71', label='Corrected (2D + physics)')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('LOO Cross-Validated R²', fontsize=10)
    ax.set_xlim(-0.5, 1.0)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()
    ax.set_title('Transfer Function: Baseline vs Physics-Corrected Prediction\n'
                 '(LOO-CV R², green bars should exceed gray dashed line)',
                 fontsize=11)

    # Annotate improvement
    for i, r in enumerate(results):
        imp = r['improvement']
        if abs(imp) > 0.01:
            ax.text(max(r['R2_corrected'], r['R2_baseline']) + 0.02, i,
                    f'Δ = {imp:+.2f}', fontsize=7, va='center',
                    color='#27ae60' if imp > 0.05 else '#7f8c8d')

    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'transfer_function_r2_improvement.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Text: transfer function equations ---
    eq_path = os.path.join(outdir, 'transfer_equations.txt')
    with open(eq_path, 'w') as f:
        f.write("Physics-Informed 2D→3D Transfer Functions\n")
        f.write("=" * 75 + "\n")
        f.write("\nModel: metric_3D = α + β · metric_2D + Σ γ_k · feature_k\n")
        f.write("(LOO cross-validated R² reported for honest predictive accuracy)\n\n")

        for r in results:
            f.write("-" * 75 + "\n")
            f.write(f"{r['name']} (n={r['n']})\n")
            f.write(f"  {r['description']}\n")
            log_note = "  [log-transformed: log(3D) = f(log(2D), features)]\n" if r['use_log'] else ""
            f.write(log_note)
            f.write(f"  R²_cv baseline:  {r['R2_baseline']:.3f}\n")
            f.write(f"  R²_cv corrected: {r['R2_corrected']:.3f}  "
                    f"(Δ = {r['improvement']:+.3f})\n")

            # Equation
            terms = [f"{r['beta'][0]:.4g}"]
            if len(r['beta']) > 1:
                v = r['beta'][1]
                var = f"log({r['metric']})" if r['use_log'] else r['metric']
                terms.append(f"{v:+.4g} · {var}_2D")
            for j, fk in enumerate(r['features']):
                v = r['beta'][2 + j]
                terms.append(f"{v:+.4g} · {fk}")

            lhs = f"log({r['metric']})_3D" if r['use_log'] else f"{r['metric']}_3D"
            f.write(f"\n  {lhs} = {' '.join(terms)}\n\n")

        # Summary
        f.write("\n" + "=" * 75 + "\n")
        f.write("SUMMARY\n")
        f.write("-" * 75 + "\n")
        good = [r for r in results if r['R2_corrected'] > 0.7]
        moderate = [r for r in results if 0.4 < r['R2_corrected'] <= 0.7]
        poor = [r for r in results if r['R2_corrected'] <= 0.4]
        improved = [r for r in results if r['improvement'] > 0.1]

        f.write(f"\n  Strong prediction (R²_cv > 0.7): "
                f"{', '.join(r['name'] for r in good) or 'none'}\n")
        f.write(f"  Moderate prediction (0.4 < R²_cv ≤ 0.7): "
                f"{', '.join(r['name'] for r in moderate) or 'none'}\n")
        f.write(f"  Poor prediction (R²_cv ≤ 0.4): "
                f"{', '.join(r['name'] for r in poor) or 'none'}\n")
        f.write(f"\n  Substantially improved by correction (ΔR² > 0.1): "
                f"{', '.join(r['name'] for r in improved) or 'none'}\n")

        f.write("\nKEY CORRECTION FEATURES:\n")
        f.write("  log_N_ratio  — log(N_granules_3D / N_granules_2D), "
                "computed from domain sizes and radii\n")
        f.write("  func_ratio   — functional granule fraction (design variable)\n")
        f.write("  Z_ratio      — Z_iso(3D)/Z_iso(2D) = 6/4 = 1.5 (constant)\n")
        f.write("  cell_coverage — cell surface coverage (design variable)\n")
        f.write("  sense_dist   — cell sensing distance (design variable)\n")
        f.write("  E_modulus    — hydrogel stiffness (design variable)\n")
        f.write("  slab_ratio   — Lz/Lxy (3D slab geometry)\n")
        f.write("  R_func       — functional granule radius (design variable)\n")

    print(f"  Transfer equations: {eq_path}")

    return results


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

MODULES = OrderedDict([
    ('paired_scatter',    ('Paired scatter plots',       plot_paired_scatter)),
    ('dimensionless',     ('Dimensionless collapse',     plot_dimensionless_collapse)),
    ('effects',           ('Effect ranking comparison',  plot_effect_comparison)),
    ('time_evolution',    ('Time evolution overlay',      plot_time_evolution)),
    ('scaling',           ('Scaling regression table',    compute_scaling_table)),
    ('summary',           ('Summary dashboard',           plot_summary_dashboard)),
    ('ratio_factors',     ('3D/2D ratio vs factors',      plot_ratio_vs_factors)),
    ('transfer',          ('Physics-informed transfer fn', fit_transfer_functions)),
])


def run_all(pairs, outdir='results/doe_2d_vs_3d', skip=None):
    """Run all 2D vs 3D comparison modules."""
    skip = set(skip or [])
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  GELS 2D vs 3D DOE Comparison")
    print("=" * 65)

    for i, (name, (label, func)) in enumerate(MODULES.items(), 1):
        if name in skip:
            print(f"  [{i}/{len(MODULES)}] {label}... SKIPPED")
            continue
        print(f"  [{i}/{len(MODULES)}] {label}...")
        try:
            func(pairs, outdir)
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n  All plots saved to: {outdir}/")
    print("=" * 65)


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GELS 2D vs 3D DOE comparison and dimensional scaling.')
    parser.add_argument('--results-dir', default=None,
                        help='Directory with DOE_2D_NNNN/ and DOE_3D_NNNN/')
    parser.add_argument('--trials-dir', default=None,
                        help='Directory with DOE trial JSON configs')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Number of runs per set (default: 20)')
    parser.add_argument('-o', '--output', default='results/doe_2d_vs_3d',
                        help='Output directory for plots')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Analysis modules to skip')
    parser.add_argument('--list-modules', action='store_true',
                        help='List available modules')
    args = parser.parse_args()

    if args.list_modules:
        print("Available 2D vs 3D comparison modules:")
        for name, (label, _) in MODULES.items():
            print(f"  {name:20s} — {label}")
        sys.exit(0)

    results_dir = args.results_dir
    if results_dir is None:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(_root, 'results', 'LHC')

    pairs = load_paired_data(results_dir, n_runs=args.n_runs,
                             trials_dir=args.trials_dir)
    if not pairs:
        print("ERROR: No paired runs loaded.")
        sys.exit(1)

    run_all(pairs, outdir=args.output, skip=set(args.skip))
