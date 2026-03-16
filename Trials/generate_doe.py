#!/usr/bin/env python3
"""
Generate fractional factorial DOE trial configs for GELLS-DEM.

Design:
    Block 1 (runs 1–16):  2^(5-1) Resolution V, generator E = ABCD
        A at two levels: inert-heavy vs functional-heavy
    Block 2 (runs 17–24): 2^(4-1) Resolution IV, generator E = BCD
        A fixed at "functional only" (phi_f=0.60, phi_i=0.00)

    Total: 24 runs.  Target N ≈ 300 granules, 45 h simulation time each.

Factors:
    A — Composition              (3 levels: inert-heavy / func-heavy / func-only)
    B — Functional granule size  (R_func: 30 vs 60 µm)
    C — Cells per granule        (4 vs 16)
    D — Hydrogel stiffness       (E_modulus: 2 vs 50 kPa)
    E — Cell sensing distance    (20 vs 80 µm)

Run:
    python Trials/generate_doe.py
    # Creates Trials/DOE_01.json ... Trials/DOE_24.json
"""

import json
import math
import os

# ── Factor levels ─────────────────────────────────────────────────────
#
# Factor A has 3 levels (categorical):
#   -1  = inert-heavy:    phi_f=0.20, phi_i=0.40, R_inert per factor B
#   +1  = func-heavy:     phi_f=0.40, phi_i=0.20, R_inert per factor B
#    0  = func-only:      phi_f=0.60, phi_i=0.00, R_inert irrelevant
#
# Factor B: when inert granules exist, controls BOTH R_func and R_inert.
#   When A=0 (func-only), only R_func matters.
#
FACTORS_BCE = {
    'B': ('Functional granule size',   {'R_func_mean': 30.0, 'R_inert_mean': 60.0},
                                        {'R_func_mean': 60.0, 'R_inert_mean': 30.0}),
    'C': ('Cells per granule',         {'n_cells_per_granule': 4},
                                        {'n_cells_per_granule': 16}),
    'D': ('E_modulus (kPa)',           {'E_modulus': 2.0},
                                        {'E_modulus': 50.0}),
    'E': ('Cell sensing distance (µm)',{'cell_sense_distance': 20.0},
                                        {'cell_sense_distance': 80.0}),
}

A_LEVELS = {
    -1: ('Inert-heavy (1:2)',  {'phi_f_target': 0.20, 'phi_i_target': 0.40}),
    +1: ('Func-heavy (2:1)',   {'phi_f_target': 0.40, 'phi_i_target': 0.20}),
     0: ('Func-only',          {'phi_f_target': 0.60, 'phi_i_target': 0.00}),
}

N_TARGET = 300  # target granule count

# ── Block 1: 2^(5-1) Resolution V, generator E = ABCD ────────────────
BLOCK1 = []
for a in (-1, +1):
    for b in (-1, +1):
        for c in (-1, +1):
            for d in (-1, +1):
                e = a * b * c * d  # generator E = ABCD
                BLOCK1.append({'A': a, 'B': b, 'C': c, 'D': d, 'E': e})

# ── Block 2: 2^(4-1) Resolution IV, generator E = BCD, A fixed at 0 ─
BLOCK2 = []
for b in (-1, +1):
    for c in (-1, +1):
        for d in (-1, +1):
            e = b * c * d  # generator E = BCD
            BLOCK2.append({'A': 0, 'B': b, 'C': c, 'D': d, 'E': e})

DESIGN = BLOCK1 + BLOCK2  # 16 + 8 = 24 runs


def compute_domain_size(phi_f, phi_i, R_f, R_i, n_target=N_TARGET):
    """Compute cubic domain side length L for target granule count."""
    V_f = (4.0 / 3.0) * math.pi * R_f ** 3
    density = phi_f / V_f
    if phi_i > 0:
        V_i = (4.0 / 3.0) * math.pi * R_i ** 3
        density += phi_i / V_i
    L_cubed = n_target / density
    return round(L_cubed ** (1.0 / 3.0))


# ── Base config (constants across all runs) ──────────────────────────
BASE = {
    "_format": "flat",
    "mode": "3D",
    # Domain — computed per run
    # Radii — set per run by factor B
    "R_func_std": 0.0,
    "R_inert_std": 0.0,
    # Composition — set per run by factor A
    # Cell geometry
    "cell_diameter": 20.0,
    "cell_height_spread": 5.0,
    "cell_coverage": 0.4,
    # Cell timeline
    "t_attach_onset": 0.0,
    "t_attach_half": 0.0,
    "t_spread_duration": 3.0,
    "fa_maturation_rate": 0.3,
    # Motor-clutch (Chan & Odde 2008)
    "n_motors": 50,
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
    "t_total": 45.0,
    "save_every_h": 1.0,
    "v_max": 20.0,
    # Shape (mild polydispersity, same as scaling trials)
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
    # Packing
    "packing_gap": 0.0,
    "boundary_exclusion": 0.2,
    "packing_settle_steps": 200,
    # Rendering
    "Ngrid": 200,
    "Ngrid_3d": 80,
    "interface_width": 3.0,
    # Performance & IO
    "use_numba": True,
    "save_data": True,
    "save_fields": True,
    "compress_archive": True,
}


def _a_label(a):
    if a == -1:
        return 'lo'
    if a == +1:
        return 'hi'
    return 'func_only'


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))  # Trials/

    print("=" * 75)
    print("  GELLS-DEM Fractional Factorial DOE Generator")
    print("  Block 1: 2^(5-1) Res V (A=±1, E=ABCD)     → 16 runs")
    print("  Block 2: 2^(4-1) Res IV (A=func-only, E=BCD) →  8 runs")
    print("  Total: 24 runs")
    print("=" * 75)
    print()
    print("  Factor A — Composition (3 levels):")
    for level, (name, vals) in A_LEVELS.items():
        print(f"    {level:+2d}: {name:20s}  {vals}")
    print()
    print("  Factors B–E (2 levels each):")
    for key, (name, low, high) in FACTORS_BCE.items():
        print(f"    {key}: {name}")
        print(f"       Low:  {low}")
        print(f"       High: {high}")
    print()
    print(f"  Target granules: {N_TARGET}")
    print(f"  Simulation time: {BASE['t_total']} h  (dt={BASE['dt']} h → "
          f"{int(BASE['t_total']/BASE['dt'])} steps)")
    print()

    header = (f"{'Run':>4}  {'Blk':>3}  {'A':>9} {'B':>3} {'C':>3} {'D':>3} "
              f"{'E':>3}  {'phi_f':>5} {'phi_i':>5} {'R_f':>4} {'R_i':>4} "
              f"{'cells':>5} {'E_mod':>5} {'sense':>5}  {'L':>5}  {'N_est':>5}")
    print(header)
    print("-" * len(header))

    for i, row in enumerate(DESIGN, 1):
        config = dict(BASE)
        block = 1 if i <= 16 else 2

        # Apply factor A
        a_level = row['A']
        _, a_vals = A_LEVELS[a_level]
        config.update(a_vals)

        # Apply factors B, C, D, E
        for factor_key in ('B', 'C', 'D', 'E'):
            _, low_vals, high_vals = FACTORS_BCE[factor_key]
            vals = high_vals if row[factor_key] == +1 else low_vals
            config.update(vals)

        # For func-only runs, R_inert doesn't matter but set R_func from B
        if a_level == 0:
            # B=-1 → R_func=30 (small), B=+1 → R_func=60 (large)
            # R_inert is irrelevant (phi_i=0), set to R_func for consistency
            config['R_inert_mean'] = config['R_func_mean']

        # Compute domain size for N≈300
        L = compute_domain_size(
            config['phi_f_target'], config['phi_i_target'],
            config['R_func_mean'], config['R_inert_mean'])
        config['Lx'] = float(L)
        config['Ly'] = float(L)
        config['Lz'] = float(L)

        # Estimate actual granule count
        V_f = (4/3) * math.pi * config['R_func_mean']**3
        n_est = int(round(L**3 * config['phi_f_target'] / V_f))
        if config['phi_i_target'] > 0:
            V_i = (4/3) * math.pi * config['R_inert_mean']**3
            n_est += int(round(L**3 * config['phi_i_target'] / V_i))

        # Output directory
        config['output_dir'] = f"results/trials/DOE_{i:02d}"

        # Encode DOE metadata
        config['_doe_run'] = i
        config['_doe_block'] = block
        config['_doe_factors'] = row
        config['_doe_description'] = (
            f"A={_a_label(row['A'])}_"
            f"B={'hi' if row['B']==1 else 'lo'}_"
            f"C={'hi' if row['C']==1 else 'lo'}_"
            f"D={'hi' if row['D']==1 else 'lo'}_"
            f"E={'hi' if row['E']==1 else 'lo'}"
        )

        # Write JSON
        filename = f"DOE_{i:02d}.json"
        filepath = os.path.join(out_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        a_str = _a_label(a_level)
        print(f"  {i:2d}    {block}   {a_str:>9s} {row['B']:+2d} {row['C']:+2d} "
              f"{row['D']:+2d} {row['E']:+2d}  "
              f"{config['phi_f_target']:.2f}  {config['phi_i_target']:.2f}  "
              f"{config['R_func_mean']:3.0f}  {config['R_inert_mean']:3.0f}  "
              f"{config['n_cells_per_granule']:5d} {config['E_modulus']:5.1f} "
              f"{config['cell_sense_distance']:5.0f}  "
              f"{L:5d}  {n_est:5d}")

    print()
    print(f"  Generated {len(DESIGN)} trial configs in {out_dir}/")
    print(f"  Run with:  python run_all_trials.py")
    print(f"  (Set TRIALS_DIR='Trials' and it will find DOE_*.json)")


if __name__ == '__main__':
    main()
