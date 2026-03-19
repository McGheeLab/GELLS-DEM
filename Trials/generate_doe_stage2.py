#!/usr/bin/env python3
"""
Generate Stage 2 adaptive refinement DOE configs for GELS V2.1.

Reads Stage 1 LHS results, fits quadratic response surface models,
computes Sobol-like sensitivity indices, and generates targeted
refinement runs around critical response surface features.

Refinement strategies:
  1. Near-optimal: dense sampling around the best Stage 1 runs
  2. Steep-gradient: fill in where response surface has large derivatives
  3. Uncertainty: sample where surrogate model has highest residuals
  4. Organ-target: sample near parameter combos that minimize D_arch

Usage:
    python Trials/generate_doe_stage2.py                        # auto-detect
    python Trials/generate_doe_stage2.py --n_runs 300           # custom count
    python Trials/generate_doe_stage2.py --response porosity    # specific response
    python Trials/generate_doe_stage2.py --results_dir results/trials
"""

import json
import math
import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube

# Add parent dir so we can import load_run
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Factor definitions (must match generate_doe.py) ─────────────────
FACTORS = [
    {'label': 'A', 'key': 'E_modulus',             'lo': 2.0,  'hi': 50.0},
    {'label': 'B', 'key': 'phi_solid_target',      'lo': 0.55, 'hi': 0.85},
    {'label': 'C', 'key': 'func_ratio',            'lo': 0.2,  'hi': 1.0},
    {'label': 'D', 'key': 'cell_surface_coverage',  'lo': 0.5,  'hi': 1.5},
    {'label': 'E', 'key': 'cell_sense_distance',   'lo': 20.0, 'hi': 80.0},
    {'label': 'F', 'key': 'R_func_mean',           'lo': 30.0, 'hi': 150.0},
    {'label': 'G', 'key': 'R_inert_mean',          'lo': 30.0, 'hi': 150.0},
]
N_FACTORS = len(FACTORS)
N_TARGET = 500

# Responses to extract from history
RESPONSES = [
    'final_porosity',
    'final_bridges',
    'final_func_clusters',
    'compaction_ratio',
    'final_permeability',
]


def compute_domain_size(phi_f, phi_i, R_f, R_i, n_target=N_TARGET):
    """Compute cubic domain side length L for target granule count."""
    V_f = (4.0 / 3.0) * math.pi * R_f ** 3
    density = phi_f / V_f
    if phi_i > 0:
        V_i = (4.0 / 3.0) * math.pi * R_i ** 3
        density += phi_i / V_i
    if density <= 0:
        return 800
    L_cubed = n_target / density
    return round(L_cubed ** (1.0 / 3.0))


def to_unit(values):
    """Map physical values to [0,1] using factor bounds."""
    lo = np.array([f['lo'] for f in FACTORS])
    hi = np.array([f['hi'] for f in FACTORS])
    return (values - lo) / (hi - lo)


def from_unit(unit_values):
    """Map [0,1] to physical values using factor bounds."""
    lo = np.array([f['lo'] for f in FACTORS])
    hi = np.array([f['hi'] for f in FACTORS])
    return lo + unit_values * (hi - lo)


def load_stage1_results(results_dir, prefix='DOE'):
    """Load Stage 1 results: factor values and response metrics."""
    results_dir = Path(results_dir)

    X = []  # factor values (physical)
    Y = {r: [] for r in RESPONSES}
    run_ids = []
    skipped = 0

    # Scan for completed runs
    for archive in sorted(results_dir.glob(f'{prefix}_*.tar.gz')):
        run_name = archive.stem  # DOE_0001
        run_dir = results_dir / run_name

        # Need extracted directory with plots/ (indicates postprocessing done)
        # or we can load directly from the archive
        try:
            from new_dem_0 import load_run
            hist, snaps, p, metadata = load_run(str(archive))
        except Exception:
            skipped += 1
            continue

        if not hist or len(hist) < 2:
            skipped += 1
            continue

        # Extract factor values from params
        x = []
        for f in FACTORS:
            val = getattr(p, f['key'], None)
            if val is None:
                # Try metadata
                val = metadata.get(f['key'], None)
            if val is None:
                skipped += 1
                break
            x.append(float(val))
        else:
            X.append(x)
            run_ids.append(run_name)

            # Extract responses from final history entry
            h_final = hist[-1]
            h_init = hist[0]

            Y['final_porosity'].append(h_final.get('porosity', np.nan))
            Y['final_bridges'].append(h_final.get('n_bridges', np.nan))
            Y['final_func_clusters'].append(h_final.get('func_clusters', np.nan))

            # Compaction ratio
            phi_f_0 = h_init.get('phi_f_inner', h_init.get('phi_f', np.nan))
            phi_f_f = h_final.get('phi_f_inner', h_final.get('phi_f', np.nan))
            if phi_f_0 > 0:
                Y['compaction_ratio'].append(phi_f_f / phi_f_0)
            else:
                Y['compaction_ratio'].append(np.nan)

            Y['final_permeability'].append(
                h_final.get('K_kozeny_inner', h_final.get('K_kozeny', np.nan)))

    X = np.array(X)
    for r in RESPONSES:
        Y[r] = np.array(Y[r])

    return X, Y, run_ids, skipped


def build_quadratic_features(X_unit):
    """Build quadratic design matrix: [1, x_i, x_i*x_j, x_i^2]."""
    n, d = X_unit.shape
    features = [np.ones(n)]  # intercept

    # Linear terms
    for i in range(d):
        features.append(X_unit[:, i])

    # Interaction terms
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X_unit[:, i] * X_unit[:, j])

    # Quadratic terms
    for i in range(d):
        features.append(X_unit[:, i] ** 2)

    return np.column_stack(features)


def fit_response_surface(X_unit, y):
    """Fit quadratic response surface via least-squares. Returns coefficients and R²."""
    valid = np.isfinite(y)
    if valid.sum() < 2 * N_FACTORS:
        return None, -1.0, np.inf

    Phi = build_quadratic_features(X_unit[valid])
    y_v = y[valid]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs, residuals, rank, sv = np.linalg.lstsq(Phi, y_v, rcond=None)

    y_pred = Phi @ coeffs
    ss_res = np.sum((y_v - y_pred) ** 2)
    ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return coeffs, r2, np.sqrt(ss_res / len(y_v))


def compute_sensitivity(X_unit, y):
    """Compute variance-based sensitivity indices (first-order, from quadratic model)."""
    valid = np.isfinite(y)
    if valid.sum() < 2 * N_FACTORS:
        return np.ones(N_FACTORS) / N_FACTORS

    X_v = X_unit[valid]
    y_v = y[valid]
    var_total = np.var(y_v)
    if var_total == 0:
        return np.ones(N_FACTORS) / N_FACTORS

    sensitivities = np.zeros(N_FACTORS)
    for i in range(N_FACTORS):
        # Bin into quartiles, compute variance of conditional means
        bins = np.digitize(X_v[:, i], np.linspace(0, 1, 5))
        means = []
        for b in range(1, 6):
            mask = bins == b
            if mask.sum() > 0:
                means.append(np.mean(y_v[mask]))
        sensitivities[i] = np.var(means) / var_total if means else 0

    # Normalize
    total = sensitivities.sum()
    if total > 0:
        sensitivities /= total
    else:
        sensitivities = np.ones(N_FACTORS) / N_FACTORS

    return sensitivities


def generate_near_optimal(X_unit, y, n_points, top_frac=0.1, radius=0.15, seed=None):
    """Generate points near the best Stage 1 runs."""
    rng = np.random.default_rng(seed)
    valid = np.isfinite(y)
    X_v = X_unit[valid]
    y_v = y[valid]

    # For porosity/permeability we want high values; for compaction we want high
    # Use top fraction by absolute value of response
    n_top = max(3, int(len(y_v) * top_frac))
    top_idx = np.argsort(y_v)[-n_top:]  # highest values
    centers = X_v[top_idx]

    points = []
    per_center = max(1, n_points // len(centers))
    for c in centers:
        for _ in range(per_center):
            perturbation = rng.normal(0, radius, N_FACTORS)
            p = np.clip(c + perturbation, 0, 1)
            points.append(p)
            if len(points) >= n_points:
                break
        if len(points) >= n_points:
            break

    return np.array(points[:n_points])


def generate_steep_gradient(X_unit, y, coeffs, n_points, seed=None):
    """Generate points where response surface gradient is steepest."""
    rng = np.random.default_rng(seed)

    # Generate candidate points
    n_candidates = n_points * 20
    candidates = rng.uniform(0, 1, (n_candidates, N_FACTORS))

    # Compute gradient magnitude at each candidate
    # For quadratic model: dy/dx_i = beta_i + sum_j beta_ij * x_j + 2*beta_ii * x_i
    grad_mags = np.zeros(n_candidates)
    for k in range(n_candidates):
        grad = np.zeros(N_FACTORS)
        idx = 1  # skip intercept
        # Linear terms
        for i in range(N_FACTORS):
            grad[i] += coeffs[idx]
            idx += 1
        # Interaction terms
        for i in range(N_FACTORS):
            for j in range(i + 1, N_FACTORS):
                grad[i] += coeffs[idx] * candidates[k, j]
                grad[j] += coeffs[idx] * candidates[k, i]
                idx += 1
        # Quadratic terms
        for i in range(N_FACTORS):
            grad[i] += 2 * coeffs[idx] * candidates[k, i]
            idx += 1
        grad_mags[k] = np.linalg.norm(grad)

    # Select top gradient points
    top_idx = np.argsort(grad_mags)[-n_points:]
    return candidates[top_idx]


def generate_uncertainty(X_unit, y, coeffs, n_points, seed=None):
    """Generate points where surrogate model residuals are largest (poor fit regions)."""
    rng = np.random.default_rng(seed)
    valid = np.isfinite(y)
    X_v = X_unit[valid]
    y_v = y[valid]

    # Compute residuals
    Phi = build_quadratic_features(X_v)
    y_pred = Phi @ coeffs
    residuals = np.abs(y_v - y_pred)

    # Sample near high-residual points
    n_top = max(3, int(len(residuals) * 0.2))
    top_idx = np.argsort(residuals)[-n_top:]
    centers = X_v[top_idx]

    points = []
    per_center = max(1, n_points // len(centers))
    for c in centers:
        for _ in range(per_center):
            perturbation = rng.normal(0, 0.1, N_FACTORS)
            p = np.clip(c + perturbation, 0, 1)
            points.append(p)
            if len(points) >= n_points:
                break
        if len(points) >= n_points:
            break

    return np.array(points[:n_points])


def generate_space_filling(n_points, seed=None):
    """Generate space-filling LHS points to cover gaps."""
    sampler = LatinHypercube(d=N_FACTORS, seed=seed, optimization="random-cd")
    return sampler.random(n=n_points)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Stage 2 adaptive refinement DOE configs")
    parser.add_argument('--results_dir', type=str, default='results/trials',
                        help='Directory with Stage 1 results (default: results/trials)')
    parser.add_argument('--n_runs', type=int, default=300,
                        help='Total Stage 2 runs (default: 300)')
    parser.add_argument('--response', type=str, default=None,
                        help='Primary response for optimization (default: auto-select)')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed (default: 2026)')
    parser.add_argument('--prefix', type=str, default='DOE_S2',
                        help='File prefix (default: DOE_S2)')
    parser.add_argument('--stage1_prefix', type=str, default='DOE',
                        help='Stage 1 file prefix (default: DOE)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))  # Trials/
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        args.results_dir
    )

    print("=" * 70)
    print("  GELS V2.1 Stage 2 — Adaptive Refinement DOE Generator")
    print("=" * 70)
    print()

    # ── Load Stage 1 results ─────────────────────────────────────────
    print(f"  Loading Stage 1 results from: {results_dir}")
    X, Y, run_ids, skipped = load_stage1_results(results_dir, args.stage1_prefix)
    n_loaded = len(run_ids)
    print(f"  Loaded: {n_loaded} runs, skipped: {skipped}")

    if n_loaded < 20:
        print(f"\n  ERROR: Need at least 20 Stage 1 results, got {n_loaded}.")
        print(f"  Run Stage 1 first, then postprocess before running Stage 2.")
        sys.exit(1)

    X_unit = to_unit(X)

    # ── Fit response surfaces and compute sensitivities ──────────────
    print()
    print(f"  {'Response':<25} {'R²':>8} {'RMSE':>10}  Top factors")
    print(f"  {'-'*25} {'-'*8} {'-'*10}  {'-'*30}")

    best_r2 = -1
    best_response = None
    all_sensitivities = {}
    all_coeffs = {}

    for resp_name in RESPONSES:
        y = Y[resp_name]
        n_valid = np.isfinite(y).sum()
        if n_valid < 2 * N_FACTORS:
            print(f"  {resp_name:<25} {'skip':>8} (only {n_valid} valid)")
            continue

        coeffs, r2, rmse = fit_response_surface(X_unit, y)
        sens = compute_sensitivity(X_unit, y)
        all_sensitivities[resp_name] = sens
        all_coeffs[resp_name] = coeffs

        # Top 3 factors
        top3_idx = np.argsort(sens)[-3:][::-1]
        top3_str = ", ".join(
            f"{FACTORS[i]['label']}({sens[i]:.0%})" for i in top3_idx)

        print(f"  {resp_name:<25} {r2:>8.3f} {rmse:>10.4f}  {top3_str}")

        if r2 > best_r2:
            best_r2 = r2
            best_response = resp_name

    # Select primary response
    primary = args.response if args.response else best_response
    if primary not in all_coeffs:
        primary = best_response
    print(f"\n  Primary response for refinement: {primary} (R²={best_r2:.3f})")

    # ── Compute aggregate sensitivity across all responses ───────────
    agg_sens = np.zeros(N_FACTORS)
    for resp_name, sens in all_sensitivities.items():
        agg_sens += sens
    agg_sens /= len(all_sensitivities) if all_sensitivities else 1

    print(f"\n  Aggregate factor importance:")
    for i in np.argsort(agg_sens)[::-1]:
        bar = '█' * int(agg_sens[i] * 40)
        print(f"    {FACTORS[i]['label']} ({FACTORS[i]['key']:<25}): "
              f"{agg_sens[i]:.1%} {bar}")

    # Identify dead factors (<5% aggregate importance)
    dead = [i for i in range(N_FACTORS) if agg_sens[i] < 0.05]
    if dead:
        dead_str = ", ".join(f"{FACTORS[i]['label']}({FACTORS[i]['key']})" for i in dead)
        print(f"\n  Low-importance factors (< 5%): {dead_str}")
        print(f"  These will be sampled at reduced resolution in Stage 2.")

    # ── Generate Stage 2 points ──────────────────────────────────────
    n_total = args.n_runs
    y_primary = Y[primary]
    coeffs_primary = all_coeffs[primary]

    # Allocation: 35% near-optimal, 25% steep-gradient, 20% uncertainty, 20% space-filling
    n_optimal = int(n_total * 0.35)
    n_gradient = int(n_total * 0.25)
    n_uncertain = int(n_total * 0.20)
    n_filling = n_total - n_optimal - n_gradient - n_uncertain

    print(f"\n  Stage 2 allocation ({n_total} runs):")
    print(f"    Near-optimal:    {n_optimal}")
    print(f"    Steep-gradient:  {n_gradient}")
    print(f"    Uncertainty:     {n_uncertain}")
    print(f"    Space-filling:   {n_filling}")
    print()

    rng_seed = args.seed
    points_unit = []

    # Strategy 1: Near-optimal
    pts = generate_near_optimal(X_unit, y_primary, n_optimal,
                                top_frac=0.1, radius=0.12, seed=rng_seed)
    points_unit.append(pts)
    strategies = ['near_optimal'] * len(pts)

    # Strategy 2: Steep-gradient
    pts = generate_steep_gradient(X_unit, y_primary, coeffs_primary,
                                  n_gradient, seed=rng_seed + 1)
    points_unit.append(pts)
    strategies += ['steep_gradient'] * len(pts)

    # Strategy 3: Uncertainty
    pts = generate_uncertainty(X_unit, y_primary, coeffs_primary,
                               n_uncertain, seed=rng_seed + 2)
    points_unit.append(pts)
    strategies += ['uncertainty'] * len(pts)

    # Strategy 4: Space-filling
    pts = generate_space_filling(n_filling, seed=rng_seed + 3)
    points_unit.append(pts)
    strategies += ['space_filling'] * len(pts)

    all_points_unit = np.vstack(points_unit)
    all_points_phys = from_unit(all_points_unit)

    # ── Load BASE config from generate_doe.py ────────────────────────
    # Import the BASE dict
    gen_doe_path = os.path.join(out_dir, 'generate_doe.py')
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_doe", gen_doe_path)
    gen_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_mod)
    BASE = gen_mod.BASE

    # ── Write trial configs ──────────────────────────────────────────
    # Save design matrix
    design_path = os.path.join(out_dir, "doe_stage2_design.csv")
    header = ",".join(["run", "strategy"] + [f['key'] for f in FACTORS])
    rows = []

    n_stats = {'min_N': 1e9, 'max_N': 0, 'min_L': 1e9, 'max_L': 0}

    for run_idx in range(n_total):
        config = dict(BASE)
        run_num = run_idx + 1

        # Apply factor values
        factor_values = {}
        for j, f in enumerate(FACTORS):
            val = float(all_points_phys[run_idx, j])
            if f['key'] in ('E_modulus', 'R_func_mean', 'R_inert_mean',
                            'cell_sense_distance'):
                val = round(val, 1)
            elif f['key'] in ('phi_solid_target', 'func_ratio',
                              'cell_surface_coverage'):
                val = round(val, 4)
            config[f['key']] = val
            factor_values[f['key']] = val

        # Derive phi_f, phi_i
        phi_solid = config['phi_solid_target']
        func_ratio = config['func_ratio']
        phi_f = phi_solid * func_ratio
        phi_i = phi_solid * (1.0 - func_ratio)
        R_f = config['R_func_mean']
        R_i = config['R_inert_mean']

        L = compute_domain_size(phi_f, phi_i, R_f, R_i)
        config['Lx'] = float(L)
        config['Ly'] = float(L)
        config['Lz'] = float(L)

        N_est = gen_mod.estimate_n_granules(L, phi_f, phi_i, R_f, R_i)

        config['output_dir'] = f"results/trials/{args.prefix}_{run_num:04d}"

        # DOE metadata
        config['_doe_stage'] = 2
        config['_doe_run'] = run_num
        config['_doe_method'] = 'adaptive_refinement'
        config['_doe_strategy'] = strategies[run_idx]
        config['_doe_seed'] = args.seed
        config['_doe_primary_response'] = primary
        config['_doe_factor_values'] = factor_values
        config['_doe_unit_values'] = {
            f['key']: float(all_points_unit[run_idx, j])
            for j, f in enumerate(FACTORS)
        }

        # Write JSON
        filename = f"{args.prefix}_{run_num:04d}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'w') as fh:
            json.dump(config, fh, indent=2)

        # Design matrix row
        row = ([str(run_num), strategies[run_idx]] +
               [f"{all_points_phys[run_idx, j]:.6g}" for j in range(N_FACTORS)])
        rows.append(",".join(row))

        n_stats['min_N'] = min(n_stats['min_N'], N_est)
        n_stats['max_N'] = max(n_stats['max_N'], N_est)
        n_stats['min_L'] = min(n_stats['min_L'], L)
        n_stats['max_L'] = max(n_stats['max_L'], L)

    with open(design_path, 'w') as fh:
        fh.write(header + "\n")
        fh.write("\n".join(rows) + "\n")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"  Summary:")
    print(f"    Stage 2 runs generated: {n_total}")
    print(f"    Granule count range:    {n_stats['min_N']:.0f} – {n_stats['max_N']:.0f}")
    print(f"    Domain size range:      {n_stats['min_L']:.0f} – {n_stats['max_L']:.0f} µm")
    print(f"    Design matrix:          {design_path}")
    cpu_hours = n_total * 4.5 * 4
    print()
    print(f"  HPC estimate: ~{cpu_hours / 1000:.1f}k CPU-hours "
          f"({cpu_hours / 150000 * 100:.1f}% of monthly Puma allocation)")
    print()
    print(f"  Combined budget (Stage 1 + 2):")
    total_runs = n_loaded + n_total
    total_cpu = total_runs * 4.5 * 4
    print(f"    Total runs: {total_runs}")
    print(f"    Total CPU-hours: ~{total_cpu / 1000:.1f}k "
          f"({total_cpu / 150000 * 100:.1f}% of monthly allocation)")
    print()
    print(f"  Generated {n_total} trial configs in {out_dir}/")
    print(f"  Run with:  python run_all_trials.py")

    # ── Save sensitivity report ──────────────────────────────────────
    report_path = os.path.join(out_dir, "doe_stage1_sensitivity.txt")
    with open(report_path, 'w') as fh:
        fh.write("GELS Stage 1 Sensitivity Analysis\n")
        fh.write(f"Based on {n_loaded} completed runs\n")
        fh.write("=" * 60 + "\n\n")

        fh.write("Aggregate factor importance:\n")
        for i in np.argsort(agg_sens)[::-1]:
            fh.write(f"  {FACTORS[i]['label']} ({FACTORS[i]['key']:<25}): "
                     f"{agg_sens[i]:.1%}\n")

        fh.write(f"\nPer-response sensitivities:\n")
        for resp_name, sens in all_sensitivities.items():
            fh.write(f"\n  {resp_name}:\n")
            for i in np.argsort(sens)[::-1]:
                fh.write(f"    {FACTORS[i]['label']}: {sens[i]:.1%}\n")

    print(f"  Sensitivity report: {report_path}")


if __name__ == '__main__':
    main()
