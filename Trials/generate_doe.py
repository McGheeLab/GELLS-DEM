#!/usr/bin/env python3
"""
Generate Stage 1 Latin Hypercube DOE trial configs for GELLS-DEM V2.1.

Two-stage adaptive DOE strategy:
  Stage 1 (this script): 500-run maximin LHS across 7 factors.
  Stage 2 (generate_doe_stage2.py): Adaptive refinement around critical
           response surface features, informed by Stage 1 results.

Design:
    7 continuous factors sampled via maximin Latin Hypercube.
    Default: 500 runs (~4,500 CPU-hours on Puma, 3% monthly allocation).
    Target N ≈ 250 granules per run, 48 h simulation time, periodic BCs.

    Domain size L is computed as max of:
      - L_n:     cube root of (N_target / number_density)
      - L_image: 2 * (2*r_bound_max + L_max) + margin  [periodic minimum image]
      - L_rve:   4 * 2 * r_bound_max                    [RVE for bulk statistics]

    If the resulting N exceeds N_MAX (600), L is clamped and a warning is
    printed.  Extreme R_f/R_i ratios (>3:1) produce many small granules
    relative to the large ones and may still yield high N.

Factors:
    A — E_modulus (kPa)          : hydrogel stiffness       [2, 50]
    B — phi_solid_target         : total packing fraction    [0.55, 0.85]
    C — func_ratio               : functional fraction       [0.2, 1.0]
    D — cell_surface_coverage    : cell surface coverage     [0.5, 1.5]
    E — cell_sense_distance (µm) : filopodia sensing range   [20, 80]
    F — R_func_mean (µm)         : functional granule radius [40, 120]
    G — R_inert_mean (µm)        : inert granule radius      [40, 120]

Fixed:
    mode = 3D, boundary_mode = periodic, bridge lock-in enabled,
    t_total = 48 h, monodisperse (std=0).

Run:
    python Trials/generate_doe.py              # 500 runs (default)
    python Trials/generate_doe.py --n_runs 300 # custom count
    python Trials/generate_doe.py --seed 42    # reproducible design
"""

import json
import math
import os
import sys
import argparse

import numpy as np
from scipy.stats.qmc import LatinHypercube


# ── Factor definitions (continuous bounds) ───────────────────────────
FACTORS = [
    {
        'label': 'A',
        'key': 'E_modulus',
        'description': 'Hydrogel stiffness (kPa)',
        'lo': 2.0,
        'hi': 50.0,
    },
    {
        'label': 'B',
        'key': 'phi_solid_target',
        'description': 'Total solid packing fraction',
        'lo': 0.55,
        'hi': 0.85,
    },
    {
        'label': 'C',
        'key': 'func_ratio',
        'description': 'Functional granule fraction',
        'lo': 0.2,
        'hi': 1.0,
    },
    {
        'label': 'D',
        'key': 'cell_surface_coverage',
        'description': 'Cell surface coverage (>1 = stacking)',
        'lo': 0.5,
        'hi': 1.5,
    },
    {
        'label': 'E',
        'key': 'cell_sense_distance',
        'description': 'Cell sensing distance (µm)',
        'lo': 20.0,
        'hi': 80.0,
    },
    {
        'label': 'F',
        'key': 'R_func_mean',
        'description': 'Functional granule radius (µm)',
        'lo': 40.0,
        'hi': 120.0,
    },
    {
        'label': 'G',
        'key': 'R_inert_mean',
        'description': 'Inert granule radius (µm)',
        'lo': 40.0,
        'hi': 120.0,
    },
]

N_FACTORS = len(FACTORS)
N_TARGET = 250   # target granule count (periodic BCs — no boundary exclusion)
N_MAX = 600      # cap to avoid blowup from extreme size ratios
AR_MAX = 1.35    # worst-case bounding radius = R * AR_MAX (AR 1.2 + blockiness)


def compute_domain_size(phi_f, phi_i, R_f, R_i, cell_sense_distance,
                        bridge_break_gap, n_target=N_TARGET):
    """Compute cubic domain side length L satisfying three constraints.

    1. N_target:       enough granules for statistical convergence.
    2. Minimum image:  L > 2 * cutoff so cKDTree periodic BCs are correct.
    3. RVE size:       L > 4 * d_max for representative bulk packing.

    Returns (L, constraint_name) where constraint_name indicates which
    constraint was binding.
    """
    # --- 1. Target granule count ---
    V_f = (4.0 / 3.0) * math.pi * R_f ** 3
    density = phi_f / V_f
    if phi_i > 0:
        V_i = (4.0 / 3.0) * math.pi * R_i ** 3
        density += phi_i / V_i
    if density <= 0:
        return 800, 'fallback'
    L_n = (n_target / density) ** (1.0 / 3.0)

    # --- 2. Periodic minimum image: L > 2 * cutoff ---
    r_bound_max = max(R_f, R_i) * AR_MAX
    L_max = max(cell_sense_distance, bridge_break_gap)
    cutoff = 2.0 * r_bound_max + L_max
    L_image = 2.0 * cutoff + 2.0 * max(R_f, R_i)   # safety margin

    # --- 3. RVE: at least 4 large-granule diameters ---
    L_rve = 4.0 * 2.0 * r_bound_max

    # Pick the binding constraint
    candidates = [('N_target', L_n), ('min_image', L_image), ('RVE', L_rve)]
    constraint, L = max(candidates, key=lambda x: x[1])
    return round(L), constraint


def estimate_n_granules(L, phi_f, phi_i, R_f, R_i):
    """Estimate actual granule count for a given domain size."""
    V_f = (4.0 / 3.0) * math.pi * R_f ** 3
    n_est = int(round(L**3 * phi_f / V_f))
    if phi_i > 0:
        V_i = (4.0 / 3.0) * math.pi * R_i ** 3
        n_est += int(round(L**3 * phi_i / V_i))
    return n_est


# ── Base config (constants across all runs) ──────────────────────────
BASE = {
    "_format": "flat",
    "mode": "3D",

    # Radii std = 0 (monodisperse, mean set by factors F, G)
    "R_func_std": 0.0,
    "R_inert_std": 0.0,

    # Cell geometry
    "cell_diameter": 20.0,
    "cell_height_spread": 5.0,
    "cell_coverage": 0.6,

    # Cell timeline
    "t_attach_onset": 0.0,
    "t_attach_half": 0.0,
    "t_spread_duration": 3.0,
    "fa_maturation_rate": 0.3,

    # Motor-clutch (Chan & Odde 2008)
    "n_motors": 200,
    "F_motor_stall": 0.5,
    "n_clutches": 75,
    "k_clutch": 5.0,
    "k_on_clutch": 1.0,
    "k_off_clutch": 0.1,
    "F_bond": 0.002,

    # Cell migration
    "cell_migration_speed": 5.0,

    # Cell bridging
    "F_max_per_cell": 100.0,
    "L_rest": 5.0,
    "bridge_attempt_rate": 0.3,
    "bridge_formation_time": 2.0,
    "bridge_senescence_time": 24.0,
    "min_fa_for_bridge": 0.3,
    "bridge_break_gap": 60.0,
    "bridge_lock_force_threshold": 5.0,
    "bridge_secondary_rate_mult": 3.0,
    "expected_bridge_force": 100.0,

    # Contact mechanics
    "poisson_ratio": 0.49,

    # Friction
    "tau_0_ii": 50.0,
    "tau_0_if": 500.0,
    "tau_0_ff": 2000.0,
    "friction_v_ref": 1.0,

    # Adhesion
    "W_adh_ii": 0.0005,
    "W_adh_if": 0.001,
    "W_adh_ff": 0.002,

    # Drag
    "eta": 0.001,
    "drag_scale": 0.05,
    "T_active": 5.0,

    # Time
    "dt": 0.1,
    "t_total": 48.0,
    "save_every_h": 1.0,
    "v_max": 20.0,

    # Shape (mild polydispersity)
    "shape_enabled": True,
    "aspect_ratio_func_mean": 1.2,
    "aspect_ratio_func_std": 0.15,
    "aspect_ratio_inert_mean": 1.2,
    "aspect_ratio_inert_std": 0.15,
    "blockiness_func_mean": 2.5,
    "blockiness_func_std": 0.3,
    "blockiness_inert_mean": 2.5,
    "blockiness_inert_std": 0.3,
    "drag_scale_rot": 0.05,
    "omega_max": 1.0,
    "aspect_ratio_c_func_mean": 1.0,
    "aspect_ratio_c_func_std": 0.1,
    "aspect_ratio_c_inert_mean": 1.0,
    "aspect_ratio_c_inert_std": 0.1,
    "blockiness_n2_func_mean": 2.5,
    "blockiness_n2_func_std": 0.3,
    "blockiness_n2_inert_mean": 2.5,
    "blockiness_n2_inert_std": 0.3,
    "omega_max_3d": 1.0,

    # Boundary conditions (periodic — no wall artifacts, no exclusion zone)
    "boundary_mode": "periodic",
    "packing_gap": 0.0,
    "boundary_exclusion": 0.0,
    "packing_settle_steps": 200,

    # Rendering
    "Ngrid": 200,
    "Ngrid_3d": 60,
    "interface_width": 3.0,

    # Performance & IO
    "use_numba": True,
    "save_data": True,
    "save_fields": True,
    "compress_archive": True,
}


def generate_lhs(n_runs, seed=None):
    """Generate maximin Latin Hypercube samples in [0,1]^d, then scale to factor bounds."""
    sampler = LatinHypercube(d=N_FACTORS, seed=seed, optimization="random-cd")
    samples_unit = sampler.random(n=n_runs)  # shape (n_runs, N_FACTORS)

    # Scale to physical bounds
    lo = np.array([f['lo'] for f in FACTORS])
    hi = np.array([f['hi'] for f in FACTORS])
    samples = lo + samples_unit * (hi - lo)

    return samples, samples_unit


def main():
    parser = argparse.ArgumentParser(description="Generate Stage 1 LHS DOE configs")
    parser.add_argument('--n_runs', type=int, default=500,
                        help='Number of LHS samples (default: 500)')
    parser.add_argument('--seed', type=int, default=2025,
                        help='Random seed for reproducibility (default: 2025)')
    parser.add_argument('--prefix', type=str, default='DOE',
                        help='File prefix (default: DOE)')
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(__file__))  # Trials/
    n_runs = args.n_runs

    print("=" * 70)
    print("  GELLS-DEM V2.1 Stage 1 — Latin Hypercube DOE Generator")
    print(f"  {N_FACTORS} factors, {n_runs} runs (maximin LHS)")
    print("=" * 70)
    print()

    for f in FACTORS:
        print(f"  Factor {f['label']}: {f['description']}")
        print(f"           {f['key']} ∈ [{f['lo']}, {f['hi']}]")
    print()
    print(f"  Fixed: mode=3D, boundary_mode=periodic, t_total=48 h, monodisperse (std=0)")
    print(f"  Target granules: {N_TARGET} (cap: {N_MAX})")
    print(f"  Seed: {args.seed}")
    print()

    # Generate LHS design
    samples, samples_unit = generate_lhs(n_runs, seed=args.seed)

    # Save the raw design matrix for analysis
    design_path = os.path.join(out_dir, "doe_stage1_design.csv")
    header = ",".join(["run"] + [f['key'] for f in FACTORS])
    rows = []
    for i in range(n_runs):
        row = [str(i + 1)] + [f"{samples[i, j]:.6g}" for j in range(N_FACTORS)]
        rows.append(",".join(row))
    with open(design_path, 'w') as fh:
        fh.write(header + "\n")
        fh.write("\n".join(rows) + "\n")

    # Print header
    hdr = (f"{'Run':>5}  {'E_mod':>6} {'phi_s':>6} {'f_rat':>5} "
           f"{'cov':>5} {'sense':>5} {'R_f':>5} {'R_i':>5}  "
           f"{'phi_f':>5} {'phi_i':>5} {'L':>5} {'bind':>9}  {'N_est':>5}")
    print(hdr)
    print("-" * len(hdr))

    n_stats = {'min_N': 1e9, 'max_N': 0, 'min_L': 1e9, 'max_L': 0}
    n_capped = 0
    n_warnings = 0

    for run_idx in range(n_runs):
        config = dict(BASE)

        # Apply factor values from LHS
        factor_values = {}
        for j, f in enumerate(FACTORS):
            val = float(samples[run_idx, j])
            # Round to sensible precision
            if f['key'] in ('E_modulus', 'R_func_mean', 'R_inert_mean',
                            'cell_sense_distance'):
                val = round(val, 1)
            elif f['key'] in ('phi_solid_target', 'func_ratio',
                              'cell_surface_coverage'):
                val = round(val, 4)
            config[f['key']] = val
            factor_values[f['key']] = val

        # Derive phi_f, phi_i from phi_solid_target + func_ratio
        phi_solid = config['phi_solid_target']
        func_ratio = config['func_ratio']
        phi_f = phi_solid * func_ratio
        phi_i = phi_solid * (1.0 - func_ratio)

        R_f = config['R_func_mean']
        R_i = config['R_inert_mean']
        sense = config['cell_sense_distance']

        # Compute domain size with physics constraints
        L, constraint = compute_domain_size(
            phi_f, phi_i, R_f, R_i, sense, BASE['bridge_break_gap'])
        N_est = estimate_n_granules(L, phi_f, phi_i, R_f, R_i)

        # Cap N to avoid blowup from extreme size ratios
        capped = False
        if N_est > N_MAX:
            # Shrink L to bring N closer to N_MAX
            V_f_vol = (4.0 / 3.0) * math.pi * R_f ** 3
            density = phi_f / V_f_vol
            if phi_i > 0:
                V_i_vol = (4.0 / 3.0) * math.pi * R_i ** 3
                density += phi_i / V_i_vol
            if density > 0:
                L_capped = round((N_MAX / density) ** (1.0 / 3.0))
                # But don't go below minimum image constraint
                r_bound_max = max(R_f, R_i) * AR_MAX
                cutoff = 2.0 * r_bound_max + max(sense, BASE['bridge_break_gap'])
                L_min_image = round(2.0 * cutoff + 2.0 * max(R_f, R_i))
                if L_capped >= L_min_image:
                    # Cap is achievable within physics constraints
                    L = L_capped
                    constraint = 'capped'
                else:
                    # Min-image prevents cap — N will exceed N_MAX
                    L = L_min_image
                    constraint = 'uncappable'
                N_est = estimate_n_granules(L, phi_f, phi_i, R_f, R_i)
                capped = True
                n_capped += 1

        # Warn on extreme size ratios
        ratio = max(R_f, R_i) / max(min(R_f, R_i), 1.0)
        if ratio > 3.0:
            n_warnings += 1

        config['Lx'] = float(L)
        config['Ly'] = float(L)
        config['Lz'] = float(L)

        # Output directory
        run_num = run_idx + 1
        config['output_dir'] = f"results/trials/{args.prefix}_{run_num:04d}"

        # DOE metadata
        config['_doe_stage'] = 1
        config['_doe_run'] = run_num
        config['_doe_method'] = 'LHS'
        config['_doe_seed'] = args.seed
        config['_doe_n_target'] = N_TARGET
        config['_doe_constraint'] = constraint
        config['_doe_factor_values'] = factor_values
        config['_doe_unit_values'] = {
            f['key']: float(samples_unit[run_idx, j])
            for j, f in enumerate(FACTORS)
        }

        # Write JSON
        filename = f"{args.prefix}_{run_num:04d}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'w') as fh:
            json.dump(config, fh, indent=2)

        # Track stats
        n_stats['min_N'] = min(n_stats['min_N'], N_est)
        n_stats['max_N'] = max(n_stats['max_N'], N_est)
        n_stats['min_L'] = min(n_stats['min_L'], L)
        n_stats['max_L'] = max(n_stats['max_L'], L)

        # Print sampled rows
        flag = ' *' if capped else ''
        if run_num <= 5 or run_num > n_runs - 3 or run_num % 100 == 0:
            print(f"  {run_num:5d}  "
                  f"{config['E_modulus']:6.1f} {phi_solid:.4f} {func_ratio:.3f} "
                  f"{config['cell_surface_coverage']:.3f} {config['cell_sense_distance']:5.1f} "
                  f"{R_f:5.1f} {R_i:5.1f}  "
                  f"{phi_f:.3f} {phi_i:.3f} "
                  f"{L:5d} {constraint:>9}  {N_est:5d}{flag}")
        elif run_num == 6:
            print(f"  {'...':>5}")

    print()
    print(f"  Summary:")
    print(f"    Runs generated:      {n_runs}")
    print(f"    Granule count range: {n_stats['min_N']:.0f} – {n_stats['max_N']:.0f}")
    print(f"    Domain size range:   {n_stats['min_L']:.0f} – {n_stats['max_L']:.0f} µm")
    print(f"    Boundary mode:       periodic (no wall artifacts)")
    if n_capped > 0:
        print(f"    N-capped runs:       {n_capped} (L shrunk to keep N <= {N_MAX})")
    if n_warnings > 0:
        print(f"    Size ratio > 3:1:    {n_warnings} runs (may have under-resolved large species)")
    cpu_hours = n_runs * 2.5 * 4  # ~2.5 hrs/run with N≈250
    print(f"    Design matrix:       {design_path}")
    print()
    print(f"  HPC estimate: ~{cpu_hours / 1000:.1f}k CPU-hours "
          f"({cpu_hours / 150000 * 100:.1f}% of monthly Puma allocation)")
    print()
    print(f"  Generated {n_runs} trial configs in {out_dir}/")
    print(f"  Run with:  python run_all_trials.py")


if __name__ == '__main__':
    main()
