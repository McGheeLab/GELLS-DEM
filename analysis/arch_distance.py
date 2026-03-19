#!/usr/bin/env python3
"""
Architectural Distance: Scaffold vs. Native Organ Targets
==========================================================
Computes the weighted Mahalanobis-like distance between GELS scaffold
descriptors and native organ architecture targets defined in organ_targets.py.

The distance metric uses log-transform for scale-dependent quantities so that
ratios (not absolute differences) drive the comparison:

    D_arch = sqrt( sum_k  w_k * ((log(d_k^scaffold) - log(d_k^organ)) / sigma_k)^2 )

where sigma_k is the standard deviation of the log-transformed organ descriptor.

Usage:
    # Single run vs target organ:
    python analysis/arch_distance.py -i results/default --target trabecular_bone

    # DOE scan against all organs:
    python analysis/arch_distance.py --scan-dir results/trials --target lung_alveoli

    # All analyses:
    python analysis/arch_distance.py -i results/default --scan-dir results/trials -o results/arch_analysis

Or import programmatically:
    from analysis.arch_distance import architectural_distance, distance_to_all_organs
    d = architectural_distance(scaffold_desc, 'trabecular_bone')

NOTE: Scaffold descriptors should come from tissue_descriptors.compute_descriptor_vector().
Organ targets should be validated against the user's own histological measurements.

Units: micrometres, um^2, kPa, dimensionless — matching organ_targets.py and tissue_descriptors.py.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import os
import json
import csv
import argparse

from analysis.organ_targets import (
    ORGAN_TARGETS, DESCRIPTOR_KEYS, get_target, list_organs,
    target_vector, descriptor_keys,
)


# ======================================================================
# Configuration: which descriptors get log-transformed
# ======================================================================

# Scale-dependent quantities where ratios matter more than absolute diffs.
# These are log-transformed before computing the distance.
LOG_TRANSFORM_KEYS = {
    'bv_tv', 'surface_density', 'specific_surface',
    'tb_th', 'tb_sp', 'tb_n',
    'correlation_length',
    'mean_chord_tissue', 'mean_chord_pore',
    'mean_pore_radius',
    'porosity',
    'permeability_KC',
    'connectivity_density',
}


# ======================================================================
# Core distance computation
# ======================================================================

def _safe_log(x):
    """Log-transform with protection against zero/negative values."""
    return np.log(np.clip(np.abs(x), 1e-30, None))


def architectural_distance(scaffold_descriptors, organ_name, weights=None):
    """Compute the weighted Mahalanobis-like distance between scaffold and organ target.

    D = sqrt( sum_k  w_k * ((d_k^scaffold - d_k^organ) / sigma_k)^2 )

    For log-transformed descriptors, the comparison is done in log-space:
        d_k = log(scaffold_k),  mu_k = log(organ_mean_k),  sigma_k = organ_std_k / organ_mean_k

    Parameters
    ----------
    scaffold_descriptors : dict
        Descriptor vector (key -> float). Keys should match DESCRIPTOR_KEYS.
    organ_name : str
        Target organ name (key in ORGAN_TARGETS).
    weights : dict or None
        Per-descriptor weights. Missing keys default to 1.0. If None, uniform.

    Returns
    -------
    float
        Architectural distance (lower = closer match to the target organ).
    """
    target = get_target(organ_name)
    target_mean = target['mean']
    target_std = target['std']

    sum_sq = 0.0
    n_used = 0
    for k in DESCRIPTOR_KEYS:
        # Skip descriptors not present in scaffold
        if k not in scaffold_descriptors or k not in target_mean:
            continue
        s_val = scaffold_descriptors[k]
        t_mean = target_mean[k]
        t_std = target_std.get(k, None)
        if s_val is None or t_mean is None or t_std is None:
            continue
        if t_std == 0:
            continue

        w = 1.0
        if weights is not None:
            w = weights.get(k, 1.0)

        if k in LOG_TRANSFORM_KEYS:
            # Log-space comparison: use coefficient of variation as sigma
            if t_mean == 0 or s_val == 0:
                continue
            s_log = _safe_log(s_val)
            t_log = _safe_log(t_mean)
            # sigma in log-space approx = CV = std/mean (for small CV)
            # For larger CV, use log(1 + CV) which is more accurate
            cv = abs(t_std / t_mean)
            sigma_log = np.log(1.0 + cv) if cv < 1.0 else cv
            if sigma_log < 1e-10:
                continue
            z = (s_log - t_log) / sigma_log
        else:
            # Linear-space comparison
            z = (s_val - t_mean) / t_std

        sum_sq += w * z * z
        n_used += 1

    if n_used == 0:
        return float('inf')
    return float(np.sqrt(sum_sq))


def distance_to_all_organs(scaffold_descriptors, weights=None):
    """Compute architectural distance to every organ target.

    Parameters
    ----------
    scaffold_descriptors : dict
        Descriptor vector from tissue_descriptors.
    weights : dict or None
        Per-descriptor weights.

    Returns
    -------
    list of (str, float)
        Sorted list of (organ_name, distance), closest first.
    """
    results = []
    for organ_name in ORGAN_TARGETS:
        d = architectural_distance(scaffold_descriptors, organ_name, weights)
        results.append((organ_name, d))
    results.sort(key=lambda x: x[1])
    return results


def closest_organ(scaffold_descriptors, weights=None):
    """Return the name of the closest matching organ.

    Parameters
    ----------
    scaffold_descriptors : dict
        Descriptor vector.
    weights : dict or None
        Per-descriptor weights.

    Returns
    -------
    str
        Name of the closest organ target.
    """
    ranked = distance_to_all_organs(scaffold_descriptors, weights)
    return ranked[0][0] if ranked else None


# ======================================================================
# Trajectory analysis (time series)
# ======================================================================

def _load_run_lazy(run_dir):
    """Lazy import and call of new_dem_0.load_run."""
    from new_dem_0 import load_run
    return load_run(run_dir)


def _descriptors_from_history_entry(h):
    """Extract a scaffold descriptor dict from a history entry.

    Maps history keys (from new_dem_0 metrics) to DESCRIPTOR_KEYS where
    possible. Many descriptors require field-level analysis and will not
    be present in the scalar history -- those are simply omitted.
    """
    desc = {}
    # Direct mappings from history dict keys to descriptor keys
    mapping = {
        'porosity':         'porosity',
        'K_kozeny_carman':  'permeability_KC',
        'phi_f_mean':       'bv_tv',  # approximate: tissue fraction ~ BV/TV
    }
    for hist_key, desc_key in mapping.items():
        if hist_key in h:
            desc[desc_key] = h[hist_key]
    # Derived descriptors
    if 'porosity' in h:
        desc['porosity'] = h['porosity']
    if 'd_grain_mean' in h:
        # Approximate Tb.Th from mean grain diameter
        desc['tb_th'] = h['d_grain_mean']
    return desc


def distance_trajectory(run_dir, weights=None):
    """Compute architectural distance to all organs at each saved timepoint.

    Parameters
    ----------
    run_dir : str
        Path to simulation output directory.
    weights : dict or None
        Per-descriptor weights.

    Returns
    -------
    dict
        organ_name -> np.ndarray of distances over time.
    times : np.ndarray
        Time values corresponding to each distance entry.
    """
    hist, snaps, p, metadata = _load_run_lazy(run_dir)
    if not hist:
        return {}, np.array([])

    times = np.array([h.get('time', i) for i, h in enumerate(hist)])
    trajectories = {organ: [] for organ in ORGAN_TARGETS}

    for h in hist:
        desc = _descriptors_from_history_entry(h)
        for organ_name in ORGAN_TARGETS:
            d = architectural_distance(desc, organ_name, weights)
            trajectories[organ_name].append(d)

    trajectories = {k: np.array(v) for k, v in trajectories.items()}
    return trajectories, times


# ======================================================================
# DOE analysis
# ======================================================================

def _load_doe_results(scan_dir):
    """Load all DOE run results from a scan directory.

    Returns list of dicts with keys: 'run_id', 'run_dir', 'hist', 'params'.
    """
    runs = []
    # Look for DOE_XX or any subdirectories with history files
    entries = sorted(os.listdir(scan_dir))
    for entry in entries:
        run_dir = os.path.join(scan_dir, entry)
        # Try .tar.gz
        tar_path = run_dir + '.tar.gz' if not entry.endswith('.tar.gz') else run_dir
        if entry.endswith('.tar.gz'):
            run_dir = run_dir[:-7]
            entry = entry[:-7]

        hist_csv = os.path.join(run_dir, 'history.csv')
        hist_json = os.path.join(run_dir, 'history.json')
        params_json = os.path.join(run_dir, 'params.json')

        # Extract if needed
        if not os.path.isdir(run_dir) and os.path.isfile(tar_path):
            import tarfile
            try:
                with tarfile.open(tar_path, 'r:gz') as tf:
                    tf.extractall(path=scan_dir)
            except Exception:
                continue

        hist = None
        if os.path.isfile(hist_csv):
            with open(hist_csv) as f:
                reader = csv.DictReader(f)
                hist = []
                for row in reader:
                    hist.append({k: _try_float(v) for k, v in row.items()})
        elif os.path.isfile(hist_json):
            with open(hist_json) as f:
                hist = json.load(f)

        if hist is None or len(hist) < 1:
            continue

        params = {}
        if os.path.isfile(params_json):
            with open(params_json) as f:
                params = json.load(f)

        runs.append({
            'run_id': entry,
            'run_dir': run_dir,
            'hist': hist,
            'params': params,
        })
    return runs


def _try_float(v):
    """Try to convert a string to float, return as-is on failure."""
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


def optimal_parameters(scan_dir, target_organ, weights=None):
    """Find which DOE parameter set minimizes D_arch to the target organ.

    Across all DOE runs in scan_dir, compute the final-timepoint architectural
    distance to the target organ.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE run subdirectories.
    target_organ : str
        Target organ name.
    weights : dict or None
        Per-descriptor weights.

    Returns
    -------
    list of (str, float, dict)
        Sorted list of (run_id, D_arch, params_summary), closest first.
    """
    runs = _load_doe_results(scan_dir)
    results = []
    for run in runs:
        # Use final history entry
        h_final = run['hist'][-1]
        desc = _descriptors_from_history_entry(h_final)
        d = architectural_distance(desc, target_organ, weights)
        # Summarize key params
        p = run['params']
        summary = {k: p.get(k) for k in [
            'E_modulus', 'R_func', 'R_inert', 'n_cells', 'cell_sense_distance',
            'func_fraction', 'inert_fraction', 'mode',
        ] if p.get(k) is not None}
        results.append((run['run_id'], d, summary))
    results.sort(key=lambda x: x[1])
    return results


def sensitivity_analysis(scan_dir, target_organ, weights=None):
    """Compute how each DOE factor affects D_arch to the target organ.

    Uses the factor assignments from DOE config files (keys '_doe_factors')
    to compute main effects via difference of means at high vs low levels.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE run subdirectories.
    target_organ : str
        Target organ name.
    weights : dict or None
        Per-descriptor weights.

    Returns
    -------
    dict
        factor_name -> (main_effect, p_value).
        main_effect is the difference in mean D_arch between high and low levels.
        p_value is from a two-sample t-test (None if insufficient data).
    """
    from scipy import stats

    runs = _load_doe_results(scan_dir)

    # Collect (factor_levels, D_arch) for each run
    factor_distances = {}  # factor_key -> {level: [distances]}
    trials_dir = os.path.join(os.path.dirname(scan_dir), 'Trials')

    for run in runs:
        h_final = run['hist'][-1]
        desc = _descriptors_from_history_entry(h_final)
        d = architectural_distance(desc, target_organ, weights)

        # Get factor assignments from params or config
        factors = run['params'].get('_doe_factors', {})
        if not factors:
            # Try loading from Trials/ config
            config_path = os.path.join(trials_dir, run['run_id'] + '.json')
            if os.path.isfile(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                factors = config.get('_doe_factors', {})

        for fkey, level in factors.items():
            if fkey not in factor_distances:
                factor_distances[fkey] = {}
            level_int = int(level) if level != 0 else 0
            if level_int not in factor_distances[fkey]:
                factor_distances[fkey][level_int] = []
            factor_distances[fkey][level_int].append(d)

    # Compute main effects and significance
    results = {}
    for fkey, levels in factor_distances.items():
        high = np.array(levels.get(1, []))
        low = np.array(levels.get(-1, []))
        if len(high) < 1 or len(low) < 1:
            results[fkey] = (0.0, None)
            continue
        main_effect = float(np.mean(high) - np.mean(low))
        if len(high) >= 2 and len(low) >= 2:
            _, p_value = stats.ttest_ind(high, low, equal_var=False)
            p_value = float(p_value)
        else:
            p_value = None
        results[fkey] = (main_effect, p_value)

    return results


# ======================================================================
# Plotting functions
# ======================================================================

def _setup_matplotlib():
    """Configure matplotlib for headless operation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_distance_radar(scaffold_descriptors, organs=None, ax=None):
    """Radar/spider chart showing scaffold position relative to multiple organs.

    Each spoke is a descriptor (log-normalized where appropriate). The scaffold
    and each organ target are overlaid for visual comparison.

    Parameters
    ----------
    scaffold_descriptors : dict
        Scaffold descriptor vector.
    organs : list of str or None
        Organ targets to include. Defaults to all.
    ax : matplotlib Axes or None
        If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure (or None if ax was provided)
    """
    plt = _setup_matplotlib()

    if organs is None:
        organs = list_organs()

    # Find descriptors present in scaffold
    keys = [k for k in DESCRIPTOR_KEYS if k in scaffold_descriptors]
    if len(keys) < 3:
        print("  WARNING: fewer than 3 descriptors available for radar chart")
        return None

    # Collect all values for normalization (organs + scaffold)
    all_vals = {k: [] for k in keys}
    for organ in organs:
        target = get_target(organ)
        for k in keys:
            v = target['mean'].get(k, None)
            if v is not None:
                if k in LOG_TRANSFORM_KEYS and v > 0:
                    all_vals[k].append(_safe_log(v))
                else:
                    all_vals[k].append(v)
    for k in keys:
        v = scaffold_descriptors.get(k, None)
        if v is not None:
            if k in LOG_TRANSFORM_KEYS and v > 0:
                all_vals[k].append(_safe_log(v))
            else:
                all_vals[k].append(v)

    # Normalization range
    mins = {k: min(all_vals[k]) if all_vals[k] else 0 for k in keys}
    maxs = {k: max(all_vals[k]) if all_vals[k] else 1 for k in keys}

    def _normalize(k, raw_val):
        if k in LOG_TRANSFORM_KEYS and raw_val > 0:
            raw_val = _safe_log(raw_val)
        span = maxs[k] - mins[k]
        if span == 0:
            return 0.5
        return (raw_val - mins[k]) / span

    n_keys = len(keys)
    angles = np.linspace(0, 2 * np.pi, n_keys, endpoint=False).tolist()
    angles += angles[:1]
    labels = [k.replace('_', '\n') for k in keys]

    created_fig = (ax is None)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    else:
        fig = None

    # Plot organs
    colors = plt.cm.Set2(np.linspace(0, 1, len(organs)))
    for organ, color in zip(organs, colors):
        target = get_target(organ)
        values = [_normalize(k, target['mean'].get(k, 0)) for k in keys]
        values += [values[0]]
        ax.plot(angles, values, '-', label=organ.replace('_', ' ').title(),
                color=color, linewidth=1.0, alpha=0.6)

    # Plot scaffold (bold)
    scaffold_vals = [_normalize(k, scaffold_descriptors.get(k, 0)) for k in keys]
    scaffold_vals += [scaffold_vals[0]]
    ax.plot(angles, scaffold_vals, 'k-o', label='Scaffold', linewidth=2.5,
            markersize=5, zorder=10)
    ax.fill(angles, scaffold_vals, alpha=0.1, color='black')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_title('Scaffold vs Organ Targets', fontsize=12, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.05), fontsize=7)

    return fig


def plot_distance_trajectory(run_dir, organs=None, outdir=None):
    """Plot D_arch vs time for each organ, showing convergence.

    Parameters
    ----------
    run_dir : str
        Simulation output directory.
    organs : list of str or None
        Organs to plot. Defaults to all.
    outdir : str or None
        Output directory. Defaults to run_dir.
    """
    plt = _setup_matplotlib()

    if organs is None:
        organs = list_organs()
    if outdir is None:
        outdir = run_dir

    trajectories, times = distance_trajectory(run_dir)
    if len(times) == 0:
        print("  WARNING: no time data for distance trajectory plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(organs)))
    for organ, color in zip(organs, colors):
        if organ in trajectories:
            ax.plot(times, trajectories[organ], '-', color=color,
                    label=organ.replace('_', ' ').title(), linewidth=1.5)

    ax.set_xlabel('Time (h)', fontsize=11)
    ax.set_ylabel('Architectural Distance D_arch', fontsize=11)
    ax.set_title('Scaffold Convergence to Organ Targets', fontsize=13)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'distance_trajectory.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_distance_heatmap(scan_dir, outdir=None):
    """Heatmap of DOE runs (rows) x organs (columns), colored by D_arch.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE run subdirectories.
    outdir : str or None
        Output directory. Defaults to scan_dir.
    """
    plt = _setup_matplotlib()

    if outdir is None:
        outdir = scan_dir

    runs = _load_doe_results(scan_dir)
    if not runs:
        print("  WARNING: no DOE runs found for heatmap")
        return

    organs = list_organs()
    run_ids = [r['run_id'] for r in runs]

    # Build distance matrix
    dist_matrix = np.full((len(runs), len(organs)), np.nan)
    for i, run in enumerate(runs):
        h_final = run['hist'][-1]
        desc = _descriptors_from_history_entry(h_final)
        for j, organ in enumerate(organs):
            dist_matrix[i, j] = architectural_distance(desc, organ)

    fig, ax = plt.subplots(figsize=(max(8, len(organs) * 1.2),
                                     max(6, len(runs) * 0.35)))
    im = ax.imshow(dist_matrix, aspect='auto', cmap='RdYlGn_r')

    # Annotate cells
    for i in range(len(runs)):
        for j in range(len(organs)):
            val = dist_matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=6, color='white' if val > np.nanmedian(dist_matrix) else 'black')

    # Annotate closest organ per run
    for i in range(len(runs)):
        row = dist_matrix[i]
        if np.any(np.isfinite(row)):
            best_j = int(np.nanargmin(row))
            ax.add_patch(plt.Rectangle(
                (best_j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='gold', linewidth=2))

    ax.set_xticks(range(len(organs)))
    ax.set_xticklabels([o.replace('_', '\n') for o in organs], fontsize=7, rotation=45, ha='right')
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels(run_ids, fontsize=7)
    ax.set_xlabel('Organ Target')
    ax.set_ylabel('DOE Run')
    ax.set_title('Architectural Distance: DOE Runs vs Organ Targets', fontsize=12)
    plt.colorbar(im, ax=ax, label='D_arch', shrink=0.8)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'distance_heatmap.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_parameter_sensitivity(scan_dir, target_organ, outdir=None):
    """Bar chart of main effects of each DOE factor on D_arch.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE run subdirectories.
    target_organ : str
        Target organ name.
    outdir : str or None
        Output directory.
    """
    plt = _setup_matplotlib()

    if outdir is None:
        outdir = scan_dir

    effects = sensitivity_analysis(scan_dir, target_organ)
    if not effects:
        print("  WARNING: no factor data for sensitivity plot")
        return

    # Try to get nice labels from viz_doe
    try:
        from viz.doe import FACTOR_LABELS
        labels = {k: FACTOR_LABELS.get(k, k) for k in effects}
    except ImportError:
        labels = {k: k for k in effects}

    factor_keys = sorted(effects.keys())
    main_effects = [effects[k][0] for k in factor_keys]
    p_values = [effects[k][1] for k in factor_keys]
    xlabels = [labels.get(k, k).replace('\n', ' ') for k in factor_keys]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#e74c3c' if me > 0 else '#2ecc71' for me in main_effects]
    bars = ax.barh(range(len(factor_keys)), main_effects, color=colors, alpha=0.8)

    # Annotate significance
    for i, (me, pv) in enumerate(zip(main_effects, p_values)):
        sig = ''
        if pv is not None:
            if pv < 0.01:
                sig = ' **'
            elif pv < 0.05:
                sig = ' *'
        ax.text(me + (0.01 if me >= 0 else -0.01), i,
                f'{me:+.2f}{sig}', va='center',
                ha='left' if me >= 0 else 'right', fontsize=8)

    ax.set_yticks(range(len(factor_keys)))
    ax.set_yticklabels(xlabels, fontsize=9)
    ax.set_xlabel(f'Main Effect on D_arch to {target_organ.replace("_", " ").title()}',
                  fontsize=10)
    ax.set_title(f'Parameter Sensitivity: D_arch to {target_organ.replace("_", " ").title()}',
                 fontsize=12)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, axis='x', alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'sensitivity_{target_organ}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_optimization_landscape(scan_dir, target_organ, outdir=None):
    """2D scatter plot colored by D_arch to target organ.

    X-axis: a stiffness-related parameter (E_modulus or first DOE factor).
    Y-axis: composition ratio (func_fraction / (func_fraction + inert_fraction)).
    Color: D_arch to target organ.

    Parameters
    ----------
    scan_dir : str
        Directory containing DOE run subdirectories.
    target_organ : str
        Target organ name.
    outdir : str or None
        Output directory.
    """
    plt = _setup_matplotlib()

    if outdir is None:
        outdir = scan_dir

    runs = _load_doe_results(scan_dir)
    if not runs:
        print("  WARNING: no DOE runs found for optimization landscape")
        return

    xs, ys, ds, run_labels = [], [], [], []
    for run in runs:
        h_final = run['hist'][-1]
        desc = _descriptors_from_history_entry(h_final)
        d = architectural_distance(desc, target_organ)
        p = run['params']

        # X: stiffness
        x = p.get('E_modulus', None)
        # Y: composition ratio
        ff = p.get('func_fraction', None)
        fi = p.get('inert_fraction', None)
        if x is not None and ff is not None and fi is not None:
            total = ff + fi
            y = ff / total if total > 0 else 0.5
        elif x is not None:
            y = 0.5  # default if composition not known
        else:
            continue

        xs.append(x)
        ys.append(y)
        ds.append(d)
        run_labels.append(run['run_id'])

    if len(xs) < 2:
        print("  WARNING: insufficient data for optimization landscape")
        return

    xs = np.array(xs)
    ys = np.array(ys)
    ds = np.array(ds)

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(xs, ys, c=ds, cmap='RdYlGn_r', s=80, edgecolor='black',
                    linewidth=0.5, zorder=5)
    plt.colorbar(sc, ax=ax, label='D_arch')

    # Label best point
    best_idx = np.argmin(ds)
    ax.annotate(
        f'{run_labels[best_idx]}\nD={ds[best_idx]:.1f}',
        (xs[best_idx], ys[best_idx]),
        textcoords='offset points', xytext=(10, 10),
        fontsize=7, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('E_modulus (kPa)', fontsize=11)
    ax.set_ylabel('Functional Fraction (composition ratio)', fontsize=11)
    ax.set_title(f'Optimization Landscape: D_arch to {target_organ.replace("_", " ").title()}',
                 fontsize=12)
    ax.grid(True, alpha=0.3)

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f'landscape_{target_organ}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ======================================================================
# Unified runner
# ======================================================================

def run_all(run_dir=None, scan_dir=None, target_organ=None, outdir=None):
    """Run all architectural distance analyses and generate plots.

    Parameters
    ----------
    run_dir : str or None
        Single simulation run directory (for trajectory analysis).
    scan_dir : str or None
        DOE scan directory (for multi-run analysis).
    target_organ : str or None
        Target organ for optimization analysis. Defaults to 'trabecular_bone'.
    outdir : str or None
        Output directory. Defaults to run_dir or scan_dir.
    """
    if target_organ is None:
        target_organ = 'trabecular_bone'

    if outdir is None:
        outdir = run_dir or scan_dir or '.'
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("Architectural Distance Analysis")
    print("=" * 60)
    print(f"  Target organ: {target_organ}")
    print(f"  Output: {outdir}")

    # Single-run analyses
    if run_dir is not None:
        print(f"\n--- Single-run analysis: {run_dir} ---")
        try:
            trajectories, times = distance_trajectory(run_dir)
            if times.size > 0:
                # Print final distances
                print("  Final D_arch to each organ:")
                final_dists = [(org, traj[-1]) for org, traj in trajectories.items()
                               if len(traj) > 0]
                final_dists.sort(key=lambda x: x[1])
                for org, d in final_dists:
                    marker = ' <-- closest' if org == final_dists[0][0] else ''
                    print(f"    {org:25s}: {d:8.2f}{marker}")

                # Trajectory plot
                plot_distance_trajectory(run_dir, outdir=outdir)

                # Radar chart of final state
                hist, _, _, _ = _load_run_lazy(run_dir)
                if hist:
                    desc = _descriptors_from_history_entry(hist[-1])
                    fig = plot_distance_radar(desc)
                    if fig is not None:
                        path = os.path.join(outdir, 'distance_radar.png')
                        fig.savefig(path, dpi=150, bbox_inches='tight')
                        import matplotlib.pyplot as plt
                        plt.close(fig)
                        print(f"  Saved: {path}")
        except Exception as e:
            print(f"  ERROR in single-run analysis: {e}")

    # DOE scan analyses
    if scan_dir is not None:
        print(f"\n--- DOE scan analysis: {scan_dir} ---")

        # Heatmap
        try:
            plot_distance_heatmap(scan_dir, outdir=outdir)
        except Exception as e:
            print(f"  ERROR in heatmap: {e}")

        # Optimal parameters
        try:
            ranked = optimal_parameters(scan_dir, target_organ)
            if ranked:
                print(f"\n  Top 5 runs closest to {target_organ}:")
                for run_id, d, params in ranked[:5]:
                    print(f"    {run_id:15s}: D_arch={d:8.2f}  {params}")
        except Exception as e:
            print(f"  ERROR in optimal parameters: {e}")

        # Sensitivity
        try:
            plot_parameter_sensitivity(scan_dir, target_organ, outdir=outdir)
        except Exception as e:
            print(f"  ERROR in sensitivity plot: {e}")

        # Optimization landscape
        try:
            plot_optimization_landscape(scan_dir, target_organ, outdir=outdir)
        except Exception as e:
            print(f"  ERROR in optimization landscape: {e}")

    print("\n  Architectural distance analysis complete.")


# ======================================================================
# CLI
# ======================================================================

def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description='Architectural distance analysis: scaffold vs organ targets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python arch_distance.py -i results/default --target trabecular_bone\n'
            '  python arch_distance.py --scan-dir results/trials\n'
            '  python arch_distance.py -i results/default --scan-dir results/trials -o results/arch\n'
        ))
    parser.add_argument('-i', '--run-dir', type=str, default=None,
                        help='Single simulation run directory')
    parser.add_argument('--scan-dir', type=str, default=None,
                        help='DOE scan directory with multiple run subdirectories')
    parser.add_argument('--target', type=str, default='trabecular_bone',
                        help='Target organ name (default: trabecular_bone)')
    parser.add_argument('-o', '--outdir', type=str, default=None,
                        help='Output directory for plots and results')
    parser.add_argument('--list-organs', action='store_true',
                        help='List available organ targets and exit')

    args = parser.parse_args()

    if args.list_organs:
        print("Available organ targets:")
        for name in list_organs():
            target = get_target(name)
            print(f"  {name:25s} -- {target['description'][:60]}...")
        return

    if args.run_dir is None and args.scan_dir is None:
        parser.print_help()
        print("\nError: specify at least one of -i/--run-dir or --scan-dir")
        return

    run_all(
        run_dir=args.run_dir,
        scan_dir=args.scan_dir,
        target_organ=args.target,
        outdir=args.outdir,
    )


if __name__ == '__main__':
    main()
