#!/usr/bin/env python3
"""
DEM vs Contact Network Model Comparison
=========================================
Validates whether the stochastic contact-network model (~2.5s per realization)
can substitute for expensive 3D DEM (~hours) by running both on identical
initial packings from the 20-point DOE.

Analysis modules:
    1. Paired scatter         — DEM final vs Network final for each metric
    2. Time evolution overlay — DEM trajectory vs Network ensemble mean ± σ
    3. Predictability table   — R², slope, intercept for Network→DEM regression
    4. Network-only metrics   — Percolation, cluster sizes, bridge fraction
    5. Summary dashboard      — Key findings, R² heatmap

Usage:
    python viz/doe_dem_vs_network.py
    python viz/doe_dem_vs_network.py --results-dir results/LHC --n-runs 20
    python viz/doe_dem_vs_network.py --n-ensemble 50 --skip network_only

Or import:
    from viz.doe_dem_vs_network import load_comparison_data, run_all
    data = load_comparison_data('results/LHC', n_runs=20, n_ensemble=50)
    run_all(data, outdir='results/doe_dem_vs_network')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv
import json
import os
import sys
import time as _time
from collections import OrderedDict
from scipy import stats as sp_stats

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ══════════════════════════════════════════════════════════════════════
# Configuration — metric mapping between DEM and network models
# ══════════════════════════════════════════════════════════════════════

# (display_name, dem_key, stochastic_key, deterministic_key, unit)
COMPARABLE_METRICS = OrderedDict([
    ('n_contacts',    ('Contact count',      'n_contacts',       'n_contacts',      'n_contacts',        '')),
    ('n_bridges',     ('Active bridges',     'n_bridges',        'n_bridges',       'n_bridges',         '')),
    ('Z_contact',     ('Coordination Z',     None,               'Z_contact',       'Z',                 '')),
    ('K_perm',        ('Permeability (K)',    'K_kozeny_carman',  'K_perm',          None,                'µm²')),
    ('porosity',      ('Porosity',           'porosity',         'phi_v',           None,                '')),
    ('F_bridge_mean', ('Bridge force (mean)', 'bridge_force_mean','F_bridge_mean',  'mean_force_bridge', 'nN')),
    ('n_locked',      ('Locked bridges',     'n_locked_in_cells','n_locked',        'n_bridges_locked',  '')),
])

NETWORK_ONLY = OrderedDict([
    ('spanning',       ('Percolation (spanning)', 'spanning',       'percolation')),
    ('largest_cluster',('Largest cluster',        'largest_cluster','largest_cluster_size')),
    ('Z_bridge',       ('Bridge coordination',    'Z_bridge',       None)),
    ('bridge_fraction',('Bridge fraction',        None,             'bridge_fraction')),
])


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def _extract_packing(snap):
    """Extract sphere arrays from a DEM snapshot for network model init."""
    x, y = snap['x'], snap['y']
    z = snap['z'] if 'z' in snap else None
    r = snap['r']
    gtype = snap['gtype'].astype(int)

    if z is not None and len(z) > 0:
        pos = np.column_stack([x, y, z])
    else:
        pos = np.column_stack([x, y])

    # Cell counts per granule
    if 'cell_offset' in snap:
        offsets = snap['cell_offset']
        n_cells = (offsets[1:] - offsets[:-1]).astype(int)
    elif 'n_cells' in snap:
        n_cells = snap['n_cells'].astype(int)
    else:
        n_cells = np.zeros(len(r), dtype=int)

    return pos, r, gtype, n_cells


def _dem_to_network_params(p):
    """Convert DEM Params to stochastic NetworkParams by field-name matching."""
    from analysis.contact_network_model import NetworkParams
    import dataclasses
    fields = {f.name for f in dataclasses.fields(NetworkParams)}
    d = {}
    for k in fields:
        if hasattr(p, k):
            d[k] = getattr(p, k)
    return NetworkParams(**d)


def _get_dem_Z(hist_entry, n_granules):
    """Compute coordination number from DEM history entry."""
    nc = hist_entry.get('n_contacts', 0)
    return 2.0 * nc / max(n_granules, 1)


def _get_dem_metric(hist_entry, dem_key, n_granules=None):
    """Extract a metric from DEM history, computing Z if needed."""
    if dem_key is None:
        return None
    if dem_key == '_Z_computed':
        return _get_dem_Z(hist_entry, n_granules) if n_granules else None
    return hist_entry.get(dem_key)


# ══════════════════════════════════════════════════════════════════════
# Data loading + network model execution
# ══════════════════════════════════════════════════════════════════════

def load_comparison_data(results_dir, n_runs=20, n_ensemble=50, verbose=True):
    """Load DEM results and run network models on same initial packings.

    Returns list of dicts, one per DOE point.
    """
    from new_dem_0 import load_run
    from analysis.contact_network import ContactNetwork
    from analysis.contact_network_model import (
        from_packing as cnm_from_packing, solve as cnm_solve)

    data = []
    t_total_wall = _time.time()

    for i in range(1, n_runs + 1):
        name = f"DOE_3D_{i:04d}"
        run_dir = os.path.join(results_dir, name)

        if not os.path.isdir(run_dir):
            if verbose:
                print(f"  SKIP {name}: directory not found")
            continue

        # --- Load DEM data ---
        try:
            hist, snaps, p, metadata = load_run(run_dir)
        except Exception as e:
            if verbose:
                print(f"  SKIP {name}: load failed ({e})")
            continue

        if not snaps or not hist:
            if verbose:
                print(f"  SKIP {name}: no snapshots or history")
            continue

        snap_0 = snaps[0]
        t_max = hist[-1].get('time', 48.0)
        n_gran = len(snap_0['x'])
        complete = t_max >= 47.0

        # --- Extract packing ---
        pos, r, gtype, n_cells = _extract_packing(snap_0)

        # --- Deterministic ContactNetwork ---
        t0 = _time.time()
        try:
            net = ContactNetwork.from_snapshot(snap_0, p)
            det_result = net.solve(
                t_span=(0, t_max), dt=0.5, seed=42, record_every=1.0)
            det_time = _time.time() - t0
        except Exception as e:
            if verbose:
                print(f"  {name}: deterministic FAILED ({e})")
            det_result = None
            det_time = 0

        # --- Stochastic Gillespie ensemble ---
        np_params = _dem_to_network_params(p)
        np_params.t_total = t_max
        np_params.save_every_h = 1.0

        t0 = _time.time()
        try:
            model = cnm_from_packing(pos, r, gtype, n_cells, np_params)
        except Exception as e:
            if verbose:
                print(f"  {name}: stochastic model build FAILED ({e})")
            model = None

        ens_runs = []
        if model is not None:
            for j in range(n_ensemble):
                try:
                    result = cnm_solve(model, seed=100 + j, verbose=False)
                    ens_runs.append(result)
                except Exception:
                    pass
        ens_time = _time.time() - t0

        # Compute ensemble summary (interpolate to common time grid)
        ens_summary = None
        if ens_runs:
            t_grid = np.arange(0, t_max + 0.5, 1.0)
            # Collect all stochastic metric keys
            stoch_keys = [k for k in ens_runs[0]['history'][0].keys()
                          if k != 'time' and not k.startswith('_')]
            summary = {k: [] for k in stoch_keys}
            summary['time'] = t_grid

            for run in ens_runs:
                times = np.array([h['time'] for h in run['history']])
                for k in stoch_keys:
                    vals = np.array([h.get(k, np.nan) for h in run['history']],
                                   dtype=float)
                    interp = np.interp(t_grid, times, vals,
                                       left=np.nan, right=np.nan)
                    summary[k].append(interp)

            ens_summary = {'time': t_grid}
            for k in stoch_keys:
                arr = np.array(summary[k])
                ens_summary[f'{k}_mean'] = np.nanmean(arr, axis=0)
                ens_summary[f'{k}_std'] = np.nanstd(arr, axis=0)
                ens_summary[f'{k}_q25'] = np.nanpercentile(arr, 25, axis=0)
                ens_summary[f'{k}_q75'] = np.nanpercentile(arr, 75, axis=0)

        entry = {
            'run_id': i,
            'name': name,
            'n_granules': n_gran,
            'dem_hist': hist,
            't_max': t_max,
            'complete': complete,
            'params': p,
            'det_result': det_result,
            'det_time': det_time,
            'ens_runs': ens_runs,
            'ens_summary': ens_summary,
            'ens_time': ens_time,
            'n_ensemble': len(ens_runs),
        }
        data.append(entry)

        if verbose:
            n_ens = len(ens_runs)
            det_ok = 'OK' if det_result else 'FAIL'
            print(f"  {name}: N={n_gran}, t={t_max:.0f}h, "
                  f"det={det_ok}({det_time:.1f}s), "
                  f"ens={n_ens}/{n_ensemble}({ens_time:.1f}s)"
                  f"{'' if complete else ' [PARTIAL]'}")

    wall = _time.time() - t_total_wall
    if verbose:
        print(f"\n  Loaded {len(data)} DOE points in {wall:.0f}s "
              f"({sum(d['n_ensemble'] for d in data)} total Gillespie runs)")

    return data


# ══════════════════════════════════════════════════════════════════════
# 1. Paired scatter (DEM final vs Network final)
# ══════════════════════════════════════════════════════════════════════

def plot_paired_scatter(data, outdir):
    """Scatter: DEM final value vs Network final value for each metric."""
    metrics_to_plot = [
        ('n_contacts',    'Contact count',      'n_contacts',       'n_contacts',     'n_contacts'),
        ('n_bridges',     'Active bridges',     'n_bridges',        'n_bridges',      'n_bridges'),
        ('K_perm',        'Permeability (K)',    'K_kozeny_carman',  'K_perm',         None),
        ('F_bridge_mean', 'Bridge force (nN)',   'bridge_force_mean','F_bridge_mean',  'mean_force_bridge'),
        ('n_locked',      'Locked bridges',     'n_locked_in_cells','n_locked',       'n_bridges_locked'),
        ('Z_contact',     'Coordination Z',     '_Z_computed',      'Z_contact',      'Z'),
    ]

    n = len(metrics_to_plot)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (mkey, label, dem_key, stoch_key, det_key) in enumerate(metrics_to_plot):
        ax = axes[idx // ncols, idx % ncols]

        # Collect DEM vs stochastic ensemble mean (final values)
        x_dem, y_stoch, y_stoch_std = [], [], []
        y_det = []
        markers_complete = []

        for d in data:
            # DEM final value
            dem_final = d['dem_hist'][-1]
            n_gran = d['n_granules']
            v_dem = _get_dem_metric(dem_final, dem_key, n_gran)
            if v_dem is None:
                continue

            # Stochastic final (mean of ensemble finals)
            v_stoch = None
            v_std = 0
            if d['ens_summary'] is not None and f'{stoch_key}_mean' in d['ens_summary']:
                arr = d['ens_summary'][f'{stoch_key}_mean']
                std_arr = d['ens_summary'][f'{stoch_key}_std']
                v_stoch = arr[-1]
                v_std = std_arr[-1]

            # Deterministic final
            v_det = None
            if det_key and d['det_result'] is not None:
                det_metrics = d['det_result']['metrics']
                if det_metrics:
                    v_det = det_metrics[-1].get(det_key)

            if v_stoch is not None:
                x_dem.append(float(v_dem))
                y_stoch.append(float(v_stoch))
                y_stoch_std.append(float(v_std))
                y_det.append(float(v_det) if v_det is not None else np.nan)
                markers_complete.append(d['complete'])

        if not x_dem:
            ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
            ax.set_title(label, fontsize=10)
            continue

        x_dem = np.array(x_dem)
        y_stoch = np.array(y_stoch)
        y_stoch_std = np.array(y_stoch_std)
        y_det = np.array(y_det)

        # Plot stochastic ensemble mean with error bars
        for xi, yi, si, comp in zip(x_dem, y_stoch, y_stoch_std, markers_complete):
            color = '#e74c3c'
            marker = 'o' if comp else '^'
            ax.errorbar(xi, yi, yerr=si, fmt=marker, color=color,
                        markersize=6, capsize=3, alpha=0.8, linewidth=0.8)

        # Plot deterministic
        valid_det = np.isfinite(y_det)
        if valid_det.any():
            ax.scatter(x_dem[valid_det], y_det[valid_det], c='#3498db',
                       marker='s', s=30, alpha=0.7, label='Deterministic',
                       edgecolors='white', linewidth=0.5, zorder=4)

        # 1:1 line
        all_vals = np.concatenate([x_dem, y_stoch[np.isfinite(y_stoch)]])
        if valid_det.any():
            all_vals = np.concatenate([all_vals, y_det[valid_det]])
        lims = [all_vals.min(), all_vals.max()]
        margin = 0.05 * (lims[1] - lims[0] + 1e-10)
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, '--', color='gray', linewidth=0.8, alpha=0.5)

        # Regression (stochastic)
        if len(x_dem) >= 3 and np.std(x_dem) > 1e-10 and np.std(y_stoch) > 1e-10:
            slope, intercept, r_val, p_val, _ = sp_stats.linregress(x_dem, y_stoch)
            x_fit = np.linspace(x_dem.min(), x_dem.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, '-', color='#e74c3c',
                    linewidth=1.5, alpha=0.7)
            ax.text(0.05, 0.92, f'Stoch: R²={r_val**2:.3f}\nβ={slope:.2f}',
                    transform=ax.transAxes, fontsize=7, va='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                              alpha=0.6))

        ax.set_xlabel(f'DEM {label}', fontsize=9)
        ax.set_ylabel(f'Network {label}', fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)
        if idx == 0:
            from matplotlib.lines import Line2D
            legend = [
                Line2D([0], [0], marker='o', color='#e74c3c', linestyle='',
                       markersize=6, label='Stochastic mean±σ'),
                Line2D([0], [0], marker='s', color='#3498db', linestyle='',
                       markersize=6, label='Deterministic'),
                Line2D([0], [0], linestyle='--', color='gray', label='1:1'),
            ]
            ax.legend(handles=legend, fontsize=7, loc='lower right')

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('DEM vs Contact Network — Paired Final Values\n'
                 '(same initial packing, stochastic ensemble with error bars)',
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'paired_scatter.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 2. Time evolution overlay
# ══════════════════════════════════════════════════════════════════════

def plot_time_evolution(data, outdir):
    """Overlay DEM trajectories vs stochastic ensemble mean ± σ."""
    metrics_to_plot = [
        ('n_contacts',    'Contact count',      'n_contacts',       'n_contacts'),
        ('n_bridges',     'Active bridges',     'n_bridges',        'n_bridges'),
        ('F_bridge_mean', 'Bridge force (nN)',   'bridge_force_mean','F_bridge_mean'),
        ('K_perm',        'Permeability (K)',    'K_kozeny_carman',  'K_perm'),
        ('n_locked',      'Locked bridges',     'n_locked_in_cells','n_locked'),
        ('Z_contact',     'Coordination Z',     '_Z_computed',      'Z_contact'),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes_flat = axes.flatten()

    for mi, (mkey, label, dem_key, stoch_key) in enumerate(metrics_to_plot):
        ax = axes_flat[mi]

        # Individual DEM traces
        for d in data:
            times = [h['time'] for h in d['dem_hist']]
            if dem_key == '_Z_computed':
                vals = [_get_dem_Z(h, d['n_granules']) for h in d['dem_hist']]
            else:
                vals = [h.get(dem_key, np.nan) for h in d['dem_hist']]
            ax.plot(times, vals, color='#3498db', alpha=0.2, linewidth=0.6)

        # Individual stochastic ensemble traces (grand mean ± σ across DOE points)
        # Interpolate all ensemble summaries to common grid
        t_grid = np.arange(0, 49, 1.0)
        dem_interps = []
        stoch_interps = []

        for d in data:
            # DEM interpolation
            times = np.array([h['time'] for h in d['dem_hist']])
            if dem_key == '_Z_computed':
                vals = np.array([_get_dem_Z(h, d['n_granules'])
                                 for h in d['dem_hist']])
            else:
                vals = np.array([h.get(dem_key, np.nan) for h in d['dem_hist']],
                                dtype=float)
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                dem_interps.append(np.interp(t_grid, times[valid], vals[valid],
                                             left=np.nan, right=np.nan))

            # Stochastic mean interpolation
            if d['ens_summary'] is not None and f'{stoch_key}_mean' in d['ens_summary']:
                t_ens = d['ens_summary']['time']
                v_mean = d['ens_summary'][f'{stoch_key}_mean']
                stoch_interps.append(np.interp(t_grid, t_ens, v_mean,
                                               left=np.nan, right=np.nan))

        # Grand mean traces
        if dem_interps:
            dem_arr = np.array(dem_interps)
            dem_mean = np.nanmean(dem_arr, axis=0)
            dem_std = np.nanstd(dem_arr, axis=0)
            valid_t = ~np.isnan(dem_mean)
            ax.plot(t_grid[valid_t], dem_mean[valid_t], color='#2980b9',
                    linewidth=2.5, label='DEM mean', zorder=5)
            ax.fill_between(t_grid[valid_t],
                            dem_mean[valid_t] - dem_std[valid_t],
                            dem_mean[valid_t] + dem_std[valid_t],
                            color='#3498db', alpha=0.15)

        if stoch_interps:
            stoch_arr = np.array(stoch_interps)
            stoch_mean = np.nanmean(stoch_arr, axis=0)
            stoch_std = np.nanstd(stoch_arr, axis=0)
            valid_t = ~np.isnan(stoch_mean)
            ax.plot(t_grid[valid_t], stoch_mean[valid_t], color='#c0392b',
                    linewidth=2.5, label='Network mean', zorder=5)
            ax.fill_between(t_grid[valid_t],
                            stoch_mean[valid_t] - stoch_std[valid_t],
                            stoch_mean[valid_t] + stoch_std[valid_t],
                            color='#e74c3c', alpha=0.15)

        ax.set_xlabel('Time (h)', fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_xlim(0, 48)
        if mi == 0:
            ax.legend(fontsize=9)

    fig.suptitle('Time Evolution — DEM (blue) vs Stochastic Network (red)\n'
                 '(individual DEM traces + grand mean ± σ across DOE points)',
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'time_evolution.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 3. Predictability table (R² regression)
# ══════════════════════════════════════════════════════════════════════

def compute_predictability(data, outdir):
    """Fit DEM_final = α + β × Network_final for each metric."""
    metrics_to_fit = [
        ('n_contacts',    'Contact count',      'n_contacts',       'n_contacts'),
        ('n_bridges',     'Active bridges',     'n_bridges',        'n_bridges'),
        ('K_perm',        'Permeability (K)',    'K_kozeny_carman',  'K_perm'),
        ('F_bridge_mean', 'Bridge force (nN)',   'bridge_force_mean','F_bridge_mean'),
        ('n_locked',      'Locked bridges',     'n_locked_in_cells','n_locked'),
        ('Z_contact',     'Coordination Z',     '_Z_computed',      'Z_contact'),
    ]

    rows = []
    for mkey, label, dem_key, stoch_key in metrics_to_fit:
        x_net, y_dem = [], []
        for d in data:
            dem_final = d['dem_hist'][-1]
            v_dem = _get_dem_metric(dem_final, dem_key, d['n_granules'])
            if v_dem is None:
                continue
            if d['ens_summary'] is None or f'{stoch_key}_mean' not in d['ens_summary']:
                continue
            v_net = d['ens_summary'][f'{stoch_key}_mean'][-1]
            if np.isfinite(v_dem) and np.isfinite(v_net):
                x_net.append(float(v_net))
                y_dem.append(float(v_dem))

        if len(x_net) < 5:
            continue
        x_net = np.array(x_net)
        y_dem = np.array(y_dem)
        if np.std(x_net) < 1e-10 or np.std(y_dem) < 1e-10:
            continue

        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_net, y_dem)

        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = y_dem / np.where(np.abs(x_net) > 1e-10, x_net, np.nan)
            ratios = ratios[np.isfinite(ratios)]
        ratio_mean = np.mean(ratios) if len(ratios) > 0 else np.nan

        rows.append({
            'metric': mkey,
            'name': label,
            'n': len(x_net),
            'slope': slope,
            'intercept': intercept,
            'R2': r_val**2,
            'p_value': p_val,
            'ratio_dem_net': ratio_mean,
            'mean_dem': np.mean(y_dem),
            'mean_net': np.mean(x_net),
        })

    # Save CSV
    if rows:
        csv_path = os.path.join(outdir, 'predictability_table.csv')
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Save text summary
    txt_path = os.path.join(outdir, 'predictability_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("DEM vs Stochastic Contact Network — Predictability\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Metric':<25} {'slope':>7} {'intcpt':>8} {'R²':>6} "
                f"{'p-val':>9} {'DEM/Net':>8} {'n':>4}\n")
        f.write("-" * 70 + "\n")
        for r in rows:
            f.write(f"{r['name']:<25} {r['slope']:>7.3f} {r['intercept']:>8.2g} "
                    f"{r['R2']:>6.3f} {r['p_value']:>9.1e} "
                    f"{r['ratio_dem_net']:>8.2f} {r['n']:>4d}\n")

        f.write("\n\nINTERPRETATION\n" + "-" * 70 + "\n")
        good = [r for r in rows if r['R2'] > 0.7]
        moderate = [r for r in rows if 0.4 < r['R2'] <= 0.7]
        poor = [r for r in rows if r['R2'] <= 0.4]

        if good:
            f.write("\nGOOD SURROGATE (R² > 0.7):\n")
            for r in good:
                f.write(f"  {r['name']}: R²={r['R2']:.3f}, "
                        f"slope={r['slope']:.2f}, DEM/Net={r['ratio_dem_net']:.2f}\n")
            f.write("  → Network model is a reliable surrogate for these metrics.\n")
        if moderate:
            f.write("\nMODERATE (0.4 < R² ≤ 0.7):\n")
            for r in moderate:
                f.write(f"  {r['name']}: R²={r['R2']:.3f}\n")
        if poor:
            f.write("\nPOOR PREDICTOR (R² ≤ 0.4):\n")
            for r in poor:
                f.write(f"  {r['name']}: R²={r['R2']:.3f}\n")
            f.write("  → Network model insufficient for these metrics.\n")

        f.write("\nKNOWN SYSTEMATIC DIFFERENCES:\n")
        f.write("  - Network uses spheres; DEM uses superellipsoids\n")
        f.write("  - Network has wall BCs; DEM uses periodic BCs\n")
        f.write("  - Network tracks aggregate cells; DEM tracks individual cells\n")
        f.write("  - Network phi_f from sphere volumes; DEM from grid rendering\n")

    print(f"  Predictability: {txt_path}")

    # Bar chart of R²
    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        names = [r['name'] for r in rows]
        r2s = [r['R2'] for r in rows]
        colors = ['#2ecc71' if r2 > 0.7 else '#f39c12' if r2 > 0.4 else '#e74c3c'
                  for r2 in r2s]
        ax.barh(range(len(names)), r2s, color=colors, edgecolor='white')
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('R² (Network → DEM regression)', fontsize=10)
        ax.set_xlim(0, 1)
        ax.axvline(0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        for i, r in enumerate(rows):
            ax.text(r['R2'] + 0.02, i, f'β={r["slope"]:.2f}', fontsize=7,
                    va='center')
        ax.invert_yaxis()
        ax.set_title('Contact Network as DEM Surrogate — Predictability (R²)\n'
                     '(green: good, orange: moderate, red: poor)',
                     fontsize=11)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, 'predictability_r2.png'),
                    dpi=200, bbox_inches='tight')
        plt.close(fig)

    return rows


# ══════════════════════════════════════════════════════════════════════
# 4. Network-only metrics (percolation, clusters)
# ══════════════════════════════════════════════════════════════════════

def plot_network_only(data, outdir):
    """Plot metrics that only the network model provides."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Percolation onset time ---
    ax = axes[0, 0]
    onset_times = []
    bridge_counts = []
    for d in data:
        if d['ens_summary'] is None:
            continue
        if 'spanning_mean' not in d['ens_summary']:
            continue
        t_grid = d['ens_summary']['time']
        span_mean = d['ens_summary']['spanning_mean']
        # Onset = first time >50% of ensemble has spanning cluster
        idx_onset = np.where(span_mean > 0.5)[0]
        t_onset = float(t_grid[idx_onset[0]]) if len(idx_onset) > 0 else np.nan
        onset_times.append(t_onset)
        # Final bridge count from DEM
        bc = d['dem_hist'][-1].get('n_bridges', 0)
        bridge_counts.append(bc)

    if onset_times:
        onset_arr = np.array(onset_times)
        bc_arr = np.array(bridge_counts)
        valid = np.isfinite(onset_arr)
        ax.scatter(bc_arr[valid], onset_arr[valid], c='#2c3e50', s=50,
                   edgecolors='white', zorder=3)
        ax.scatter(bc_arr[~valid], [48]*int((~valid).sum()), c='#e74c3c',
                   marker='x', s=50, label='No percolation', zorder=3)
        ax.set_xlabel('DEM final bridge count', fontsize=9)
        ax.set_ylabel('Percolation onset time (h)', fontsize=9)
        ax.set_title('Percolation Onset vs DEM Bridges', fontsize=10,
                     fontweight='bold')
        ax.legend(fontsize=8)

    # --- Largest cluster evolution ---
    ax = axes[0, 1]
    t_grid = np.arange(0, 49, 1.0)
    for d in data:
        if d['ens_summary'] is None:
            continue
        if 'largest_cluster_mean' not in d['ens_summary']:
            continue
        t_ens = d['ens_summary']['time']
        lc_mean = d['ens_summary']['largest_cluster_mean']
        interp = np.interp(t_grid, t_ens, lc_mean, left=np.nan, right=np.nan)
        ax.plot(t_grid, interp, alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Time (h)', fontsize=9)
    ax.set_ylabel('Largest cluster size', fontsize=9)
    ax.set_title('Largest Bridge Cluster (all DOE points)', fontsize=10,
                 fontweight='bold')
    ax.set_xlim(0, 48)

    # --- Bridge fraction evolution ---
    ax = axes[1, 0]
    for d in data:
        if d['ens_summary'] is None:
            continue
        if 'n_bridges_mean' not in d['ens_summary']:
            continue
        t_ens = d['ens_summary']['time']
        nb_mean = d['ens_summary']['n_bridges_mean']
        interp = np.interp(t_grid, t_ens, nb_mean, left=np.nan, right=np.nan)
        ax.plot(t_grid, interp, alpha=0.4, linewidth=0.8, color='#e74c3c')

    # Overlay DEM bridge counts
    for d in data:
        times = [h['time'] for h in d['dem_hist']]
        nb = [h.get('n_bridges', 0) for h in d['dem_hist']]
        ax.plot(times, nb, alpha=0.3, linewidth=0.8, color='#3498db')

    ax.set_xlabel('Time (h)', fontsize=9)
    ax.set_ylabel('Active bridges', fontsize=9)
    ax.set_title('Bridge Count — DEM (blue) vs Network (red)', fontsize=10,
                 fontweight='bold')
    ax.set_xlim(0, 48)

    # --- Spanning probability at t=48h ---
    ax = axes[1, 1]
    span_probs = []
    run_labels = []
    for d in data:
        if d['ens_summary'] is None:
            continue
        if 'spanning_mean' not in d['ens_summary']:
            continue
        span_final = d['ens_summary']['spanning_mean'][-1]
        span_probs.append(span_final)
        run_labels.append(d['name'].replace('DOE_3D_', ''))

    if span_probs:
        colors = ['#2ecc71' if s > 0.5 else '#f39c12' if s > 0.1 else '#e74c3c'
                  for s in span_probs]
        ax.bar(range(len(span_probs)), span_probs, color=colors, edgecolor='white')
        ax.set_xticks(range(len(span_probs)))
        ax.set_xticklabels(run_labels, fontsize=7, rotation=45)
        ax.set_ylabel('P(spanning) at final time', fontsize=9)
        ax.set_title('Percolation Probability by DOE Point', fontsize=10,
                     fontweight='bold')
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    fig.suptitle('Network-Only Metrics — Percolation & Cluster Analysis\n'
                 '(metrics DEM cannot compute)',
                 fontsize=12, y=1.03)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'network_only_metrics.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 5. Summary dashboard
# ══════════════════════════════════════════════════════════════════════

def plot_summary(data, outdir):
    """Summary: cost comparison + key findings."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Cost comparison ---
    det_times = [d['det_time'] for d in data if d['det_result']]
    ens_times = [d['ens_time'] for d in data if d['ens_runs']]
    n_ens = [d['n_ensemble'] for d in data if d['ens_runs']]

    if det_times and ens_times:
        # Assume DEM walltime ~2-6 hours per run (from SLURM estimates)
        # Use metadata if available
        dem_walltimes = []
        for d in data:
            # Estimate from history timesteps × dt
            n_steps = len(d['dem_hist']) * (1.0 / 0.1)  # save_every_h / dt
            # Rough: 3D DEM ~0.5-2s per step for 250 granules
            dem_walltimes.append(n_steps * 1.0)  # ~1s per step estimate

        labels = [d['name'].replace('DOE_3D_', '') for d in data]
        x_pos = np.arange(len(data))

        ax1.bar(x_pos - 0.2, det_times, 0.35, color='#3498db',
                label='Deterministic', alpha=0.8)
        ax1.bar(x_pos + 0.2, ens_times, 0.35, color='#e74c3c',
                label=f'Ensemble ({n_ens[0] if n_ens else "?"}×)', alpha=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(labels, fontsize=7, rotation=45)
        ax1.set_ylabel('Walltime (seconds)', fontsize=9)
        ax1.set_title('Network Model Compute Time', fontsize=10,
                       fontweight='bold')
        ax1.legend(fontsize=8)

        # Speedup annotation
        mean_det = np.mean(det_times)
        mean_ens = np.mean(ens_times)
        ax1.text(0.95, 0.92,
                 f'Mean det: {mean_det:.1f}s\nMean ens: {mean_ens:.0f}s\n'
                 f'DEM est: ~1-3 hrs\nSpeedup: ~{3600/max(mean_ens,1):.0f}×',
                 transform=ax1.transAxes, fontsize=8, va='top', ha='right',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                           alpha=0.6))

    # --- R² radar ---
    metrics_to_fit = [
        ('n_contacts',    'Contacts',     'n_contacts',       'n_contacts'),
        ('n_bridges',     'Bridges',      'n_bridges',        'n_bridges'),
        ('K_perm',        'Permeability', 'K_kozeny_carman',  'K_perm'),
        ('F_bridge_mean', 'Bridge F',     'bridge_force_mean','F_bridge_mean'),
        ('n_locked',      'Locked',       'n_locked_in_cells','n_locked'),
    ]

    r2_vals = []
    r2_labels = []
    for mkey, label, dem_key, stoch_key in metrics_to_fit:
        x_net, y_dem = [], []
        for d in data:
            dem_final = d['dem_hist'][-1]
            v_dem = _get_dem_metric(dem_final, dem_key, d['n_granules'])
            if v_dem is None:
                continue
            if d['ens_summary'] is None or f'{stoch_key}_mean' not in d['ens_summary']:
                continue
            v_net = d['ens_summary'][f'{stoch_key}_mean'][-1]
            if np.isfinite(v_dem) and np.isfinite(v_net):
                x_net.append(float(v_net))
                y_dem.append(float(v_dem))

        if len(x_net) >= 3 and np.std(x_net) > 1e-10 and np.std(y_dem) > 1e-10:
            _, _, r_val, _, _ = sp_stats.linregress(x_net, y_dem)
            r2_vals.append(r_val**2)
        else:
            r2_vals.append(0)
        r2_labels.append(label)

    if r2_vals:
        colors = ['#2ecc71' if r > 0.7 else '#f39c12' if r > 0.4 else '#e74c3c'
                  for r in r2_vals]
        ax2.barh(range(len(r2_labels)), r2_vals, color=colors, edgecolor='white')
        ax2.set_yticks(range(len(r2_labels)))
        ax2.set_yticklabels(r2_labels, fontsize=9)
        ax2.set_xlabel('R² (Network → DEM)', fontsize=9)
        ax2.set_xlim(0, 1)
        ax2.axvline(0.7, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax2.invert_yaxis()
        ax2.set_title('Surrogate Accuracy Summary', fontsize=10,
                       fontweight='bold')

    fig.suptitle('DEM vs Contact Network — Summary Dashboard', fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary_dashboard.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

MODULES = OrderedDict([
    ('scatter',       ('Paired scatter',          plot_paired_scatter)),
    ('time_evo',      ('Time evolution overlay',   plot_time_evolution)),
    ('predictability',('Predictability table',     compute_predictability)),
    ('network_only',  ('Network-only metrics',     plot_network_only)),
    ('summary',       ('Summary dashboard',        plot_summary)),
])


def run_all(data, outdir='results/doe_dem_vs_network', skip=None):
    """Run all comparison modules."""
    skip = set(skip or [])
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  DEM vs Contact Network Model Comparison")
    print("=" * 65)

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


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DEM vs Contact Network comparison on matched DOE packings.')
    parser.add_argument('--results-dir', default=None,
                        help='Directory with DOE_3D_NNNN/ subdirectories')
    parser.add_argument('--n-runs', type=int, default=20,
                        help='Number of DOE runs (default: 20)')
    parser.add_argument('--n-ensemble', type=int, default=50,
                        help='Stochastic ensemble size (default: 50)')
    parser.add_argument('-o', '--output', default='results/doe_dem_vs_network',
                        help='Output directory for plots')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Modules to skip')
    parser.add_argument('--list-modules', action='store_true')
    args = parser.parse_args()

    if args.list_modules:
        print("Available modules:")
        for name, (label, _) in MODULES.items():
            print(f"  {name:20s} — {label}")
        sys.exit(0)

    results_dir = args.results_dir
    if results_dir is None:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(_root, 'results', 'LHC')

    print(f"\n  Loading DEM data + running network models...")
    print(f"  Ensemble size: {args.n_ensemble} per DOE point")
    data = load_comparison_data(results_dir, n_runs=args.n_runs,
                                n_ensemble=args.n_ensemble)
    if not data:
        print("ERROR: No DOE runs loaded.")
        sys.exit(1)

    run_all(data, outdir=args.output, skip=set(args.skip))
