#!/usr/bin/env python3
"""
DOE Analysis & Visualization for GELS Fractional Factorial
================================================================
Loads all DOE trial results and performs full statistical analysis of
the 2^(5-1) + augmented func-only design.

Analysis modules:
    1. Main effects & Pareto charts      — which factors dominate each response?
    2. Two-factor interaction plots       — synergistic/antagonistic pairs
    3. Normal probability (Daniel) plots  — statistical significance screening
    4. Time-evolution by factor level     — dynamic response trajectories
    5. Func-only comparison               — Block 2 vs Block 1 (A=0 vs A=±1)
    6. Response correlation heatmap       — which outputs co-vary?
    7. Response surface contours          — top 2 factors for key responses
    8. Summary dashboard                  — radar chart + best/worst runs

Usage:
    python viz/doe.py                             # scan results/trials/DOE_*
    python viz/doe.py --results-dir results/trials
    python viz/doe.py --skip interactions surfaces

Or import programmatically:
    from viz.doe import load_doe_data, run_all
    data = load_doe_data('results/trials')
    run_all(data, outdir='results/doe_analysis')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import argparse
import csv
import json
import glob as globmod
import os
import sys
from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

# Factor metadata for labelling
FACTOR_LABELS = OrderedDict([
    ('A', 'Composition\n(F:I ratio)'),
    ('B', 'Granule size\n(R_func)'),
    ('C', 'Cells / granule'),
    ('D', 'Stiffness\n(E_modulus)'),
    ('E', 'Sensing dist\n(cell_sense)'),
])

FACTOR_UNITS = {
    'A': '',
    'B': 'µm',
    'C': '',
    'D': 'kPa',
    'E': 'µm',
}

FACTOR_LEVEL_LABELS = {
    'A': {-1: 'Inert-heavy\n(0.20/0.40)', 0: 'Func-only\n(0.60/0.00)',
           1: 'Func-heavy\n(0.40/0.20)'},
    'B': {-1: 'Small (30)', 1: 'Large (60)'},
    'C': {-1: 'Low (4)', 1: 'High (16)'},
    'D': {-1: 'Soft (2)', 1: 'Stiff (50)'},
    'E': {-1: 'Short (20)', 1: 'Long (80)'},
}

# Responses to analyse (key in history dict, display name, higher-is-better)
RESPONSES = OrderedDict([
    ('func_lf',           ('Functional connectivity', True)),
    ('porosity',          ('Porosity', None)),  # None = neutral
    ('K_kozeny_carman',   ('Permeability (K)', True)),
    ('n_bridges',         ('Active bridges', True)),
    ('compaction_ratio',  ('Compaction ratio', None)),
    ('F_mean',            ('Mean force (nN)', None)),
    ('n_contacts',        ('Contact count', None)),
    ('disp_func',         ('Functional displacement (µm)', None)),
    ('mean_spread_frac',  ('Cell spread fraction', True)),
    ('n_overcrowded_total', ('Overcrowded cells', False)),
])

# Derived responses (computed from initial + final history)
DERIVED_RESPONSES = OrderedDict([
    ('delta_porosity',     ('Porosity change', None)),
    ('delta_compaction',   ('Compaction change', None)),
    ('bridge_rate',        ('Bridge formation rate (/h)', True)),
    ('permeability_retention', ('Permeability retention', True)),
    ('connectivity_gain',  ('Connectivity gain', True)),
])


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


def load_doe_data(results_dir, trials_dir=None):
    """Load all DOE trial results and factor assignments.

    Parameters
    ----------
    results_dir : str
        Directory containing DOE_01/, DOE_02/, ... subdirectories
        (or .tar.gz archives).
    trials_dir : str or None
        Directory containing DOE_01.json, ... config files.
        Defaults to Trials/ relative to this script.

    Returns
    -------
    data : dict with keys:
        'runs' : list of dicts, each with:
            'run_id': int, 'name': str, 'factors': dict,
            'block': int, 'params': dict, 'hist': list[dict],
            'initial': dict, 'final': dict, 'responses': dict
        'factor_keys': list of str ('A','B','C','D','E')
        'response_keys': list of str
    """
    if trials_dir is None:
        trials_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'Trials')

    runs = []
    for i in range(1, 25):
        name = f"DOE_{i:02d}"
        run_dir = os.path.join(results_dir, name)
        tar_path = run_dir + ".tar.gz"

        # Find history data
        hist_csv = os.path.join(run_dir, 'history.csv')
        if not os.path.isfile(hist_csv):
            # Try extracting from tar.gz
            if os.path.isfile(tar_path):
                import tarfile
                with tarfile.open(tar_path, 'r:gz') as tf:
                    tf.extractall(path=results_dir)
                if not os.path.isfile(hist_csv):
                    print(f"  WARNING: {name} — no history.csv after extraction")
                    continue
            else:
                print(f"  WARNING: {name} — not found, skipping")
                continue

        # Load trial config for factor assignments
        config_path = os.path.join(trials_dir, f"{name}.json")
        if os.path.isfile(config_path):
            with open(config_path) as f:
                config = json.load(f)
            factors = config.get('_doe_factors', {})
            block = config.get('_doe_block', 1 if i <= 16 else 2)
        else:
            # Fallback: try reading from params.json in results
            params_path = os.path.join(run_dir, 'params.json')
            if os.path.isfile(params_path):
                with open(params_path) as f:
                    config = json.load(f)
                factors = config.get('_doe_factors', {})
                block = config.get('_doe_block', 1 if i <= 16 else 2)
            else:
                print(f"  WARNING: {name} — no config found, skipping")
                continue

        # Load history
        hist = _load_history_csv(hist_csv)
        if len(hist) < 2:
            print(f"  WARNING: {name} — history too short ({len(hist)} rows)")
            continue

        initial = hist[0]
        final = hist[-1]

        # Extract direct responses (final values)
        responses = {}
        for key in RESPONSES:
            if key in final:
                responses[key] = final[key]

        # Compute derived responses
        if 'porosity' in initial and 'porosity' in final:
            responses['delta_porosity'] = final['porosity'] - initial['porosity']
        if 'compaction_ratio' in initial and 'compaction_ratio' in final:
            responses['delta_compaction'] = (final['compaction_ratio'] -
                                              initial['compaction_ratio'])
        t_final = final.get('time', 1.0)
        if 'n_bridges' in final and t_final > 0:
            responses['bridge_rate'] = final['n_bridges'] / t_final
        if ('K_kozeny_carman' in initial and 'K_kozeny_carman' in final and
                initial['K_kozeny_carman'] > 0):
            responses['permeability_retention'] = (final['K_kozeny_carman'] /
                                                    initial['K_kozeny_carman'])
        if 'func_lf' in initial and 'func_lf' in final:
            responses['connectivity_gain'] = final['func_lf'] - initial['func_lf']

        runs.append({
            'run_id': i,
            'name': name,
            'factors': factors,
            'block': block,
            'config': config,
            'hist': hist,
            'initial': initial,
            'final': final,
            'responses': responses,
        })

    all_response_keys = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())

    print(f"\n  Loaded {len(runs)} / 24 DOE runs")
    print(f"  Block 1: {sum(1 for r in runs if r['block']==1)} runs (A=±1)")
    print(f"  Block 2: {sum(1 for r in runs if r['block']==2)} runs (A=func-only)")

    return {
        'runs': runs,
        'factor_keys': list(FACTOR_LABELS.keys()),
        'response_keys': all_response_keys,
    }


# ══════════════════════════════════════════════════════════════════════
# Effect computation
# ══════════════════════════════════════════════════════════════════════

def compute_effects(data, response_key, block=1):
    """Compute main effects and 2-factor interactions for a response.

    Uses only runs from the specified block. Block 1 has full 2^(5-1)
    design; Block 2 has 2^(4-1) with A fixed.

    Returns dict: {'main': {factor: effect}, 'interactions': {(fi,fj): effect}}
    """
    runs = [r for r in data['runs'] if r['block'] == block]
    factors = ['A', 'B', 'C', 'D', 'E'] if block == 1 else ['B', 'C', 'D', 'E']

    main_effects = {}
    for f in factors:
        hi = [r['responses'].get(response_key, np.nan)
              for r in runs if r['factors'].get(f) == 1]
        lo = [r['responses'].get(response_key, np.nan)
              for r in runs if r['factors'].get(f) == -1]
        hi = [v for v in hi if not np.isnan(v)]
        lo = [v for v in lo if not np.isnan(v)]
        if hi and lo:
            main_effects[f] = np.mean(hi) - np.mean(lo)
        else:
            main_effects[f] = 0.0

    interactions = {}
    for i, fi in enumerate(factors):
        for fj in factors[i+1:]:
            # Interaction = 0.5 * [(mean when fi*fj=+1) - (mean when fi*fj=-1)]
            same = [r['responses'].get(response_key, np.nan)
                    for r in runs
                    if r['factors'].get(fi, 0) * r['factors'].get(fj, 0) == 1]
            diff = [r['responses'].get(response_key, np.nan)
                    for r in runs
                    if r['factors'].get(fi, 0) * r['factors'].get(fj, 0) == -1]
            same = [v for v in same if not np.isnan(v)]
            diff = [v for v in diff if not np.isnan(v)]
            if same and diff:
                interactions[(fi, fj)] = 0.5 * (np.mean(same) - np.mean(diff))
            else:
                interactions[(fi, fj)] = 0.0

    return {'main': main_effects, 'interactions': interactions}


# ══════════════════════════════════════════════════════════════════════
# 1. Pareto charts of effects
# ══════════════════════════════════════════════════════════════════════

def plot_pareto(data, outdir):
    """Pareto chart of main effects + 2FI for each response."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]

    n = len(available)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes)

    for idx, resp_key in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        effects = compute_effects(data, resp_key, block=1)

        # Combine main + interactions
        labels = []
        values = []
        for f, e in effects['main'].items():
            labels.append(f)
            values.append(e)
        for (fi, fj), e in effects['interactions'].items():
            labels.append(f"{fi}×{fj}")
            values.append(e)

        # Sort by absolute value
        order = np.argsort(np.abs(values))[::-1]
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]

        colors = ['#e74c3c' if v > 0 else '#3498db' for v in values]
        ax.barh(range(len(labels)), values, color=colors, edgecolor='white',
                linewidth=0.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color='k', linewidth=0.5)

        resp_name = RESPONSES.get(resp_key, DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
        ax.set_title(resp_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Effect magnitude', fontsize=8)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Pareto Charts — Main Effects & 2-Factor Interactions (Block 1)',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'pareto_effects.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 2. Main effects plots
# ══════════════════════════════════════════════════════════════════════

def plot_main_effects(data, outdir):
    """Main effect plots: response mean at each factor level."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]
    block1 = [r for r in data['runs'] if r['block'] == 1]
    factors = ['A', 'B', 'C', 'D', 'E']

    for resp_key in available:
        fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
        resp_name = RESPONSES.get(resp_key,
                                   DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]

        for fi, fkey in enumerate(factors):
            ax = axes[fi]
            for level in (-1, 1):
                vals = [r['responses'][resp_key] for r in block1
                        if r['factors'].get(fkey) == level
                        and resp_key in r['responses']]
                if vals:
                    mean = np.mean(vals)
                    std = np.std(vals) if len(vals) > 1 else 0
                    label = FACTOR_LEVEL_LABELS[fkey][level]
                    x = 0 if level == -1 else 1
                    ax.errorbar(x, mean, yerr=std, fmt='o-', capsize=5,
                                markersize=8, linewidth=2,
                                color='#e74c3c' if level == 1 else '#3498db')
                    ax.text(x, mean + std + 0.02 * abs(mean + 0.01),
                            f'{mean:.3g}', ha='center', fontsize=7)

            ax.set_xticks([0, 1])
            ax.set_xticklabels([FACTOR_LEVEL_LABELS[fkey][-1],
                                FACTOR_LEVEL_LABELS[fkey][1]],
                               fontsize=7)
            ax.set_xlabel(FACTOR_LABELS[fkey], fontsize=9)
            if fi == 0:
                ax.set_ylabel(resp_name, fontsize=9)

        fig.suptitle(f'Main Effects — {resp_name}', fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f'main_effects_{resp_key}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 3. Interaction plots
# ══════════════════════════════════════════════════════════════════════

def plot_interactions(data, outdir):
    """Interaction plots for the top responses."""
    top_responses = ['func_lf', 'n_bridges', 'K_kozeny_carman',
                     'compaction_ratio', 'mean_spread_frac']
    available = [k for k in top_responses
                 if any(k in r['responses'] for r in data['runs'])]
    block1 = [r for r in data['runs'] if r['block'] == 1]
    factors = ['A', 'B', 'C', 'D', 'E']

    for resp_key in available:
        nf = len(factors)
        fig, axes = plt.subplots(nf, nf, figsize=(16, 16))
        resp_name = RESPONSES.get(resp_key,
                                   DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]

        for i, fi in enumerate(factors):
            for j, fj in enumerate(factors):
                ax = axes[i, j]
                if i == j:
                    # Diagonal: main effect plot
                    for level in (-1, 1):
                        vals = [r['responses'][resp_key] for r in block1
                                if r['factors'].get(fi) == level
                                and resp_key in r['responses']]
                        if vals:
                            ax.bar(level, np.mean(vals), width=0.8,
                                   color='#e74c3c' if level == 1 else '#3498db',
                                   alpha=0.7)
                    ax.set_title(FACTOR_LABELS[fi].replace('\n', ' '),
                                 fontsize=8, fontweight='bold')
                elif j > i:
                    # Upper triangle: interaction plot
                    for fi_level in (-1, 1):
                        means = []
                        for fj_level in (-1, 1):
                            vals = [r['responses'][resp_key] for r in block1
                                    if r['factors'].get(fi) == fi_level
                                    and r['factors'].get(fj) == fj_level
                                    and resp_key in r['responses']]
                            means.append(np.mean(vals) if vals else np.nan)
                        label = f'{fi}={"+" if fi_level==1 else "−"}'
                        color = '#e74c3c' if fi_level == 1 else '#3498db'
                        ax.plot([-1, 1], means, 'o-', color=color,
                                label=label, markersize=5, linewidth=1.5)
                    ax.set_xticks([-1, 1])
                    ax.set_xticklabels(['-', '+'], fontsize=8)
                    if i == 0:
                        ax.legend(fontsize=6, loc='best')
                else:
                    # Lower triangle: empty
                    ax.set_visible(False)

                ax.tick_params(labelsize=6)

        # Column labels at bottom
        for j, fj in enumerate(factors):
            axes[-1, j].set_xlabel(FACTOR_LABELS[fj].replace('\n', ' '),
                                    fontsize=8)

        fig.suptitle(f'Interaction Matrix — {resp_name}', fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f'interactions_{resp_key}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 4. Normal probability (Daniel) plots
# ══════════════════════════════════════════════════════════════════════

def plot_normal_probability(data, outdir):
    """Half-normal probability plots to screen for significant effects."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]

    n = len(available)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, resp_key in enumerate(available):
        ax = axes[idx // ncols, idx % ncols]
        effects = compute_effects(data, resp_key, block=1)

        # Collect all effects with labels
        labels = []
        abs_effects = []
        for f, e in effects['main'].items():
            labels.append(f)
            abs_effects.append(abs(e))
        for (fi, fj), e in effects['interactions'].items():
            labels.append(f"{fi}×{fj}")
            abs_effects.append(abs(e))

        # Sort by absolute effect
        order = np.argsort(abs_effects)
        labels = [labels[i] for i in order]
        abs_effects = [abs_effects[i] for i in order]

        # Expected quantiles (half-normal)
        n_eff = len(abs_effects)
        probs = [(i + 0.5) / n_eff for i in range(n_eff)]
        from scipy.stats import halfnorm
        quantiles = halfnorm.ppf(probs)

        ax.scatter(abs_effects, quantiles, c='#2c3e50', s=30, zorder=3)

        # Label the top 3 effects (likely significant)
        for k in range(max(0, n_eff - 3), n_eff):
            ax.annotate(labels[k], (abs_effects[k], quantiles[k]),
                        xytext=(5, 3), textcoords='offset points',
                        fontsize=8, fontweight='bold', color='#e74c3c')

        # Reference line through the lower effects (noise estimate)
        if n_eff > 3:
            noise_effects = abs_effects[:n_eff - 3]
            noise_quantiles = quantiles[:n_eff - 3]
            if len(noise_effects) > 1 and np.std(noise_effects) > 0:
                m, b = np.polyfit(noise_effects, noise_quantiles, 1)
                x_line = np.linspace(0, max(abs_effects) * 1.1, 50)
                ax.plot(x_line, m * x_line + b, '--', color='gray',
                        linewidth=0.8, alpha=0.6)

        resp_name = RESPONSES.get(resp_key,
                                   DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
        ax.set_title(resp_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('|Effect|', fontsize=8)
        ax.set_ylabel('Half-normal quantile', fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Half-Normal Probability Plots — Significance Screening',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'normal_probability.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 5. Time evolution by factor level
# ══════════════════════════════════════════════════════════════════════

def plot_time_evolution(data, outdir):
    """Time traces of key metrics, colored by factor level."""
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

    block1 = [r for r in data['runs'] if r['block'] == 1]
    factors = ['A', 'B', 'C', 'D', 'E']

    for fkey in factors:
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = axes.flatten()

        for mi, mkey in enumerate(metrics):
            ax = axes[mi]
            for run in block1:
                level = run['factors'].get(fkey, 0)
                times = [h['time'] for h in run['hist']]
                values = [h.get(mkey, np.nan) for h in run['hist']]
                color = '#e74c3c' if level == 1 else '#3498db'
                alpha = 0.5
                ax.plot(times, values, color=color, alpha=alpha, linewidth=0.8)

            # Add mean traces
            for level, color, ls in [(-1, '#3498db', '-'), (1, '#e74c3c', '-')]:
                level_runs = [r for r in block1
                              if r['factors'].get(fkey) == level]
                if not level_runs:
                    continue
                # Align to common time grid
                max_len = max(len(r['hist']) for r in level_runs)
                for t_idx in range(max_len):
                    vals = [r['hist'][t_idx].get(mkey, np.nan)
                            for r in level_runs
                            if t_idx < len(r['hist'])]
                    vals = [v for v in vals if not np.isnan(v)]
                # Just plot the mean of aligned histories
                all_times = level_runs[0]['hist']
                mean_vals = []
                for t_idx in range(len(all_times)):
                    vals = [r['hist'][t_idx].get(mkey, np.nan)
                            for r in level_runs if t_idx < len(r['hist'])]
                    vals = [v for v in vals if not np.isnan(v)]
                    mean_vals.append(np.mean(vals) if vals else np.nan)
                times = [h['time'] for h in all_times]
                label_str = FACTOR_LEVEL_LABELS[fkey][level].split('\n')[0]
                ax.plot(times, mean_vals, color=color, linewidth=2.5,
                        label=f'{label_str} (mean)', zorder=5)

            ax.set_xlabel('Time (h)', fontsize=9)
            ax.set_ylabel(metric_names.get(mkey, mkey), fontsize=9)
            ax.set_title(metric_names.get(mkey, mkey), fontsize=10)
            if mi == 0:
                ax.legend(fontsize=8)

        fig.suptitle(f'Time Evolution by Factor {fkey} — '
                     f'{FACTOR_LABELS[fkey].replace(chr(10), " ")}',
                     fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f'time_evolution_{fkey}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 6. Func-only comparison (Block 2 vs Block 1)
# ══════════════════════════════════════════════════════════════════════

def plot_func_only_comparison(data, outdir):
    """Compare 3 levels of factor A: inert-heavy, func-heavy, func-only."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]

    a_groups = {
        -1: ('Inert-heavy\n(F:I = 1:2)', '#3498db'),
         1: ('Func-heavy\n(F:I = 2:1)', '#e74c3c'),
         0: ('Func-only\n(F = 0.60)', '#2ecc71'),
    }

    n = len(available)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.atleast_2d(axes).flatten()

    for idx, resp_key in enumerate(available):
        ax = axes[idx]
        positions = []
        for gi, (a_level, (label, color)) in enumerate(a_groups.items()):
            vals = [r['responses'][resp_key] for r in data['runs']
                    if r['factors'].get('A') == a_level
                    and resp_key in r['responses']]
            if vals:
                bp = ax.boxplot([vals], positions=[gi], widths=0.6,
                                patch_artist=True,
                                boxprops=dict(facecolor=color, alpha=0.6),
                                medianprops=dict(color='black', linewidth=2),
                                showmeans=True,
                                meanprops=dict(marker='D', markerfacecolor='white',
                                               markeredgecolor='black', markersize=6))
                # Scatter individual points
                ax.scatter([gi] * len(vals), vals, color=color,
                           edgecolors='white', s=30, zorder=3, alpha=0.8)
            positions.append(gi)

        ax.set_xticks(list(range(len(a_groups))))
        ax.set_xticklabels([v[0] for v in a_groups.values()], fontsize=8)
        resp_name = RESPONSES.get(resp_key,
                                   DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
        ax.set_title(resp_name, fontsize=10, fontweight='bold')

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle('Factor A Comparison — Inert-Heavy vs Func-Heavy vs Func-Only',
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'func_only_comparison.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 7. Response correlation heatmap
# ══════════════════════════════════════════════════════════════════════

def plot_correlation_matrix(data, outdir):
    """Heatmap of pairwise correlations between response variables."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]

    # Build response matrix
    n_runs = len(data['runs'])
    n_resp = len(available)
    matrix = np.full((n_runs, n_resp), np.nan)
    for i, run in enumerate(data['runs']):
        for j, key in enumerate(available):
            matrix[i, j] = run['responses'].get(key, np.nan)

    # Compute correlation (handle NaN)
    valid_mask = ~np.isnan(matrix)
    corr = np.corrcoef(matrix[:, valid_mask.all(axis=0)].T)

    valid_keys = [available[j] for j in range(n_resp) if valid_mask[:, j].all()]
    n_valid = len(valid_keys)

    if n_valid < 2:
        return

    labels = []
    for k in valid_keys:
        name = RESPONSES.get(k, DERIVED_RESPONSES.get(k, (k,)))[0]
        labels.append(name[:25])  # truncate long names

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    ax.set_xticks(range(n_valid))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(n_valid))
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells
    for i in range(n_valid):
        for j in range(n_valid):
            val = corr[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    ax.set_title('Response Correlation Matrix (all 24 runs)', fontsize=13)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'correlation_matrix.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 8. Response surface contours (top 2 factors)
# ══════════════════════════════════════════════════════════════════════

def plot_response_surfaces(data, outdir):
    """Contour-style response surfaces for the top 2 most influential factors."""
    key_responses = ['func_lf', 'n_bridges', 'K_kozeny_carman', 'compaction_ratio']
    available = [k for k in key_responses
                 if any(k in r['responses'] for r in data['runs'])]
    block1 = [r for r in data['runs'] if r['block'] == 1]

    for resp_key in available:
        effects = compute_effects(data, resp_key, block=1)
        # Find top 2 factors by |main effect|
        sorted_factors = sorted(effects['main'].items(),
                                 key=lambda x: abs(x[1]), reverse=True)
        if len(sorted_factors) < 2:
            continue
        f1, _ = sorted_factors[0]
        f2, _ = sorted_factors[1]

        # Compute means for all 4 corners
        grid = np.zeros((2, 2))
        counts = np.zeros((2, 2))
        for run in block1:
            lev1 = run['factors'].get(f1, 0)
            lev2 = run['factors'].get(f2, 0)
            val = run['responses'].get(resp_key, np.nan)
            if np.isnan(val):
                continue
            i = 0 if lev1 == -1 else 1
            j = 0 if lev2 == -1 else 1
            grid[i, j] += val
            counts[i, j] += 1
        mask = counts > 0
        grid[mask] /= counts[mask]

        fig, ax = plt.subplots(figsize=(7, 5.5))
        # Interpolate to smooth surface
        from scipy.interpolate import RegularGridInterpolator
        x = np.array([-1, 1])
        y = np.array([-1, 1])
        interp = RegularGridInterpolator((x, y), grid, method='linear',
                                          bounds_error=False, fill_value=None)
        xf = np.linspace(-1, 1, 50)
        yf = np.linspace(-1, 1, 50)
        xg, yg = np.meshgrid(xf, yf)
        zg = interp(np.stack([xg.ravel(), yg.ravel()], axis=1)).reshape(xg.shape)

        cs = ax.contourf(xg, yg, zg, levels=20, cmap='viridis', alpha=0.85)
        plt.colorbar(cs, ax=ax, label=RESPONSES.get(
            resp_key, DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0])
        ax.contour(xg, yg, zg, levels=10, colors='white', linewidths=0.3,
                   alpha=0.5)

        # Plot actual data points
        for run in block1:
            lev1 = run['factors'].get(f1, 0)
            lev2 = run['factors'].get(f2, 0)
            val = run['responses'].get(resp_key, np.nan)
            if not np.isnan(val):
                ax.scatter(lev1, lev2, c='white', edgecolors='black',
                           s=60, zorder=5)
                ax.annotate(f'{val:.2g}', (lev1, lev2),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=7, fontweight='bold')

        f1_label = FACTOR_LABELS[f1].replace('\n', ' ')
        f2_label = FACTOR_LABELS[f2].replace('\n', ' ')
        ax.set_xlabel(f'{f1_label}  (−1 = Low, +1 = High)', fontsize=10)
        ax.set_ylabel(f'{f2_label}  (−1 = Low, +1 = High)', fontsize=10)
        ax.set_xticks([-1, 1])
        ax.set_xticklabels([FACTOR_LEVEL_LABELS[f1][-1].split('\n')[0],
                             FACTOR_LEVEL_LABELS[f1][1].split('\n')[0]])
        ax.set_yticks([-1, 1])
        ax.set_yticklabels([FACTOR_LEVEL_LABELS[f2][-1].split('\n')[0],
                             FACTOR_LEVEL_LABELS[f2][1].split('\n')[0]])

        resp_name = RESPONSES.get(resp_key,
                                   DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
        ax.set_title(f'Response Surface — {resp_name}\n'
                     f'(top factors: {f1} × {f2})', fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(outdir, f'surface_{resp_key}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════
# 9. Summary dashboard
# ══════════════════════════════════════════════════════════════════════

def plot_summary_dashboard(data, outdir):
    """Summary table of all effects + heatmap of factor importance."""
    all_responses = list(RESPONSES.keys()) + list(DERIVED_RESPONSES.keys())
    available = [k for k in all_responses
                 if any(k in r['responses'] for r in data['runs'])]
    factors = ['A', 'B', 'C', 'D', 'E']

    # Build effect matrix (normalised)
    effect_matrix = np.zeros((len(available), len(factors)))
    for ri, resp_key in enumerate(available):
        effects = compute_effects(data, resp_key, block=1)
        for fi, fkey in enumerate(factors):
            effect_matrix[ri, fi] = effects['main'].get(fkey, 0)

    # Normalise each row to unit max for comparison
    row_max = np.max(np.abs(effect_matrix), axis=1, keepdims=True)
    row_max[row_max == 0] = 1
    norm_matrix = effect_matrix / row_max

    fig, ax = plt.subplots(figsize=(10, max(6, len(available) * 0.5 + 1)))
    im = ax.imshow(norm_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(factors)))
    ax.set_xticklabels([FACTOR_LABELS[f].replace('\n', ' ') for f in factors],
                        fontsize=9)
    ax.set_yticks(range(len(available)))
    resp_labels = [RESPONSES.get(k, DERIVED_RESPONSES.get(k, (k,)))[0][:30]
                   for k in available]
    ax.set_yticklabels(resp_labels, fontsize=8)

    # Annotate with raw effect values
    for ri in range(len(available)):
        for fi in range(len(factors)):
            val = effect_matrix[ri, fi]
            nval = norm_matrix[ri, fi]
            color = 'white' if abs(nval) > 0.5 else 'black'
            ax.text(fi, ri, f'{val:.2g}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label='Normalised effect (−1 to +1)')
    ax.set_title('Factor Importance Heatmap — Main Effects (Block 1)',
                 fontsize=13, pad=15)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'summary_heatmap.png'),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- Text summary ---
    summary_path = os.path.join(outdir, 'doe_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("GELS DOE Analysis Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Runs loaded: {len(data['runs'])} / 24\n")
        f.write(f"Block 1 (A=±1): {sum(1 for r in data['runs'] if r['block']==1)}\n")
        f.write(f"Block 2 (A=0):  {sum(1 for r in data['runs'] if r['block']==2)}\n\n")

        f.write("FACTOR IMPORTANCE RANKING (by |main effect|)\n")
        f.write("-" * 60 + "\n")
        for ri, resp_key in enumerate(available):
            resp_name = RESPONSES.get(resp_key,
                                       DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
            effects = compute_effects(data, resp_key, block=1)
            ranked = sorted(effects['main'].items(),
                             key=lambda x: abs(x[1]), reverse=True)
            f.write(f"\n{resp_name}:\n")
            for rank, (fkey, eff) in enumerate(ranked, 1):
                fname = FACTOR_LABELS[fkey].replace('\n', ' ')
                direction = "↑" if eff > 0 else "↓"
                f.write(f"  {rank}. {fkey} ({fname}): {eff:+.4g} {direction}\n")

        f.write("\n\nTOP INTERACTIONS (Block 1)\n")
        f.write("-" * 60 + "\n")
        for ri, resp_key in enumerate(available):
            resp_name = RESPONSES.get(resp_key,
                                       DERIVED_RESPONSES.get(resp_key, (resp_key,)))[0]
            effects = compute_effects(data, resp_key, block=1)
            ranked = sorted(effects['interactions'].items(),
                             key=lambda x: abs(x[1]), reverse=True)[:3]
            f.write(f"\n{resp_name}:\n")
            for (fi, fj), eff in ranked:
                f.write(f"  {fi}×{fj}: {eff:+.4g}\n")

    print(f"  Text summary: {summary_path}")


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

MODULES = OrderedDict([
    ('pareto',        ('Pareto charts',              plot_pareto)),
    ('main_effects',  ('Main effects',               plot_main_effects)),
    ('interactions',  ('Interaction plots',           plot_interactions)),
    ('normal_prob',   ('Normal probability plots',    plot_normal_probability)),
    ('time_evolution',('Time evolution',              plot_time_evolution)),
    ('func_only',     ('Func-only comparison',        plot_func_only_comparison)),
    ('correlation',   ('Correlation matrix',          plot_correlation_matrix)),
    ('surfaces',      ('Response surfaces',           plot_response_surfaces)),
    ('summary',       ('Summary dashboard',           plot_summary_dashboard)),
])


def run_all(data, outdir='results/doe_analysis', skip=None):
    """Run all DOE analysis modules.

    Parameters
    ----------
    data : dict
        Output from load_doe_data().
    outdir : str
        Output directory for plots and text summaries.
    skip : set of str or None
        Module names to skip.
    """
    skip = set(skip or [])
    os.makedirs(outdir, exist_ok=True)

    print("\n" + "=" * 65)
    print("  GELS DOE Analysis")
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
        description='GELS DOE analysis: load all DOE trial results and '
                    'produce statistical analysis plots.')
    parser.add_argument('--results-dir', default=None,
                        help='Directory containing DOE_01/...DOE_24/ '
                             '(default: results/trials)')
    parser.add_argument('--trials-dir', default=None,
                        help='Directory containing DOE_01.json...DOE_24.json '
                             '(default: Trials/)')
    parser.add_argument('-o', '--output', default='results/doe_analysis',
                        help='Output directory for plots')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Analysis modules to skip')
    parser.add_argument('--list-modules', action='store_true',
                        help='List available analysis modules')
    args = parser.parse_args()

    if args.list_modules:
        print("Available DOE analysis modules:")
        for name, (label, _) in MODULES.items():
            print(f"  {name:20s} — {label}")
        sys.exit(0)

    results_dir = args.results_dir
    if results_dir is None:
        # Default: look for DOE results
        _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default = os.path.join(_project_root, 'results', 'trials')
        if os.path.isdir(default):
            results_dir = default
        else:
            # Try Trials/trials/ (where tar.gz might be extracted)
            alt = os.path.join(_project_root, 'Trials', 'trials')
            if os.path.isdir(alt):
                results_dir = alt
            else:
                print("ERROR: No results directory found. Use --results-dir")
                sys.exit(1)

    data = load_doe_data(results_dir, trials_dir=args.trials_dir)
    if not data['runs']:
        print("ERROR: No DOE runs loaded. Check paths.")
        sys.exit(1)

    run_all(data, outdir=args.output, skip=set(args.skip))
