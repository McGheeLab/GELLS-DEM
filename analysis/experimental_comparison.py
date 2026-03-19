#!/usr/bin/env python3
"""
Experimental vs Simulation Comparison — Day 1 Initial Packing
==============================================================
V2.4 — Compares hand-segmented experimental area fractions (Day 1)
with simulation packing predictions and mean-field model.

Experimental data: 36 samples across 9 size combinations × 4 functional ratios.
Size codes: LF/MF/SF (Large/Medium/Small Functional) × LI/MI/SI (inert).
Ratios: 0.2, 0.3, 0.5, 0.7 (functional fraction by volume).

Measurements per sample:
  - Tissue%, Inert%, Void% (area fractions from 2D cross-section)
  - F over/under, I over/under (area fraction / expected from volume ratio)
  - T-cont%, I-cont%, V-cont% (% of phase area in largest connected component)

Usage:
    python analysis/experimental_comparison.py
    python analysis/experimental_comparison.py --run-sims   # also run matching DEM packings
    python analysis/experimental_comparison.py -o plots/exp  # custom output dir

Or import programmatically:
    from analysis.experimental_comparison import load_experimental_data, run_comparison
    df = load_experimental_data()
    run_comparison(df, outdir='results/experimental_comparison')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ---------------------------------------------------------------------------
# Experimental granule radii (µm) — best estimates from sieve fractions
# Update these if you have exact measured distributions.
# ---------------------------------------------------------------------------
SIZE_RADII = {
    'S': 50.0,    # Small: ~50 µm radius
    'M': 75.0,    # Medium: ~75 µm radius
    'L': 100.0,   # Large: ~100 µm radius
}

# Colour palette for size combos
SIZE_COMBO_COLORS = {
    'LF_LI': '#1f77b4', 'LF_MI': '#2ca02c', 'LF_SI': '#d62728',
    'MF_LI': '#9467bd', 'MF_MI': '#8c564b', 'MF_SI': '#e377c2',
    'SF_LI': '#7f7f7f', 'SF_MI': '#bcbd22', 'SF_SI': '#17becf',
}

SIZE_COMBO_MARKERS = {
    'LF_LI': 'o', 'LF_MI': 's', 'LF_SI': 'D',
    'MF_LI': '^', 'MF_MI': 'v', 'MF_SI': '<',
    'SF_LI': '>', 'SF_MI': 'p', 'SF_SI': '*',
}

RATIO_COLORS = {0.2: '#1f77b4', 0.3: '#ff7f0e', 0.5: '#2ca02c', 0.7: '#d62728'}


# ======================================================================
# Data Loading
# ======================================================================

def load_experimental_data(xlsx_path=None):
    """Load Day 1 experimental data from Excel into a list of dicts.

    Each dict has keys:
        file, size_combo, func_ratio, phi_f, phi_i, phi_v,
        f_over_under, i_over_under, cont_t, cont_i, cont_v,
        R_func, R_inert, size_ratio, phi_solid,
        target_phi_f, target_phi_i, enrichment_f, enrichment_i
    """
    if xlsx_path is None:
        xlsx_path = Path(__file__).parent.parent / 'Experimental Results' / 'Day1 analysis results.xlsx'
    xlsx_path = Path(xlsx_path)

    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required: pip install openpyxl")

    wb = openpyxl.load_workbook(str(xlsx_path), data_only=True)
    ws = wb['Sheet1']

    records = []
    for row in ws.iter_rows(min_row=23, max_row=58):
        vals = [c.value for c in row[:11]]  # columns A-K
        if vals[0] is None:
            continue

        filename = str(vals[0])
        size_combo = str(vals[1])  # e.g. 'LF_LI'
        func_ratio = float(vals[2])
        phi_f = float(vals[3])
        phi_i = float(vals[4])
        phi_v = float(vals[5])
        f_over_under = float(vals[6])
        i_over_under = float(vals[7])
        cont_t = float(vals[8])
        cont_i = float(vals[9])
        cont_v = float(vals[10])

        # Parse size codes
        parts = size_combo.split('_')
        func_size_code = parts[0][0]   # L, M, or S
        inert_size_code = parts[1][0]  # L, M, or S
        R_func = SIZE_RADII[func_size_code]
        R_inert = SIZE_RADII[inert_size_code]

        phi_solid = phi_f + phi_i
        # What fractions would you expect from a perfectly mixed packing?
        target_phi_f = phi_solid * func_ratio
        target_phi_i = phi_solid * (1.0 - func_ratio)
        enrichment_f = phi_f / target_phi_f if target_phi_f > 0 else 0.0
        enrichment_i = phi_i / target_phi_i if target_phi_i > 0 else 0.0

        records.append({
            'file': filename,
            'size_combo': size_combo,
            'func_size': func_size_code,
            'inert_size': inert_size_code,
            'func_ratio': func_ratio,
            'phi_f': phi_f,
            'phi_i': phi_i,
            'phi_v': phi_v,
            'f_over_under': f_over_under,
            'i_over_under': i_over_under,
            'cont_t': cont_t,
            'cont_i': cont_i,
            'cont_v': cont_v,
            'R_func': R_func,
            'R_inert': R_inert,
            'size_ratio': R_func / R_inert,
            'phi_solid': phi_solid,
            'target_phi_f': target_phi_f,
            'target_phi_i': target_phi_i,
            'enrichment_f': enrichment_f,
            'enrichment_i': enrichment_i,
        })

    wb.close()
    return records


# ======================================================================
# Analytical Packing Predictions
# ======================================================================

def predict_bidisperse_packing_2d(R_large, R_small, phi_solid_target, func_ratio,
                                  blockiness=2.5, aspect_ratio=1.2):
    """Predict 2D area fractions for bidisperse packing.

    For a random 2D packing of bidisperse superellipses:
    - RCP depends on size ratio (Desmond & Weeks 2014, Farr & Groot 2009)
    - 2D cross-section of 3D packing maps volume → area with stereological bias

    Returns dict with predicted phi_f, phi_i, phi_v and analytical corrections.
    """
    gamma = min(R_large, R_small) / max(R_large, R_small)  # size ratio ≤ 1

    # Bidisperse RCP enhancement (Farr & Groot 2009 approximation)
    # Monodisperse RCP ≈ 0.64 (3D spheres), 0.82-0.84 (2D discs)
    # Bidisperse can reach higher packing when size ratio deviates from 1
    phi_rcp_mono = 0.64   # 3D monodisperse RCP
    delta_phi = 0.05 * (1.0 - gamma)**2  # enhancement from size disparity
    phi_rcp_bi = phi_rcp_mono + delta_phi

    # Effective solid fraction (capped at bidisperse RCP)
    phi_solid = min(phi_solid_target, phi_rcp_bi * 1.1)  # soft granules can exceed

    phi_f = phi_solid * func_ratio
    phi_i = phi_solid * (1.0 - func_ratio)
    phi_v = 1.0 - phi_solid

    # Stereological correction: 2D cross-section area fraction vs 3D volume fraction
    # For spheres: <A_2D> / A_domain = (2/3) * phi_3D (Delesse principle: they're equal
    # for random cross-sections of a statistically homogeneous packing!)
    # So for well-mixed packings, 2D area fraction ≈ 3D volume fraction.
    # But larger granules are more likely to be intersected by a cross-section.
    # Stereological bias: P(intersect) ∝ 2R, so larger granules are over-represented.
    # Correction factor for each phase:
    R_f = SIZE_RADII.get('L', R_large) if R_large > R_small else R_small
    R_i = R_small if R_large > R_small else R_large

    # Actually use the passed radii directly
    # Weight by intersection probability ∝ R for each phase
    # Number density: n_f ∝ phi_f / R_f^3 (3D), n_i ∝ phi_i / R_i^3
    # Intersection weight: w_f = n_f * 2*R_f ∝ phi_f / R_f^2
    # Area per intersection: A_f ∝ R_f^2
    # So area fraction ∝ phi_f / R_f^2 * R_f^2 = phi_f (Delesse's principle)
    # The correction only appears for non-random cross-sections or near boundaries

    return {
        'phi_f': phi_f,
        'phi_i': phi_i,
        'phi_v': phi_v,
        'phi_solid': phi_solid,
        'phi_rcp_bi': phi_rcp_bi,
        'gamma': gamma,
        'stereological_bias': 1.0,  # Delesse: no bias for random sections
    }


def compute_simulation_predictions(records, phi_solid_target=0.70):
    """For each experimental condition, compute what the simulation would predict.

    The simulation targets phi_f = phi_solid * func_ratio via RSA packing.
    At Day 1 (t=0), no cell-driven compaction has occurred, so the comparison
    is purely about packing geometry.
    """
    predictions = []
    for rec in records:
        pred = predict_bidisperse_packing_2d(
            R_large=max(rec['R_func'], rec['R_inert']),
            R_small=min(rec['R_func'], rec['R_inert']),
            phi_solid_target=phi_solid_target,
            func_ratio=rec['func_ratio'],
        )
        pred['size_combo'] = rec['size_combo']
        pred['func_ratio'] = rec['func_ratio']
        predictions.append(pred)
    return predictions


# ======================================================================
# Statistical Summary
# ======================================================================

def print_summary(records):
    """Print statistical summary of experimental data."""
    print("=" * 75)
    print("  EXPERIMENTAL DATA SUMMARY — Day 1 Initial Packing")
    print("=" * 75)

    # Overall statistics
    phi_f_all = [r['phi_f'] for r in records]
    phi_i_all = [r['phi_i'] for r in records]
    phi_v_all = [r['phi_v'] for r in records]
    phi_s_all = [r['phi_solid'] for r in records]

    print(f"\n  N samples: {len(records)}")
    print(f"\n  Phase fractions (mean ± std):")
    print(f"    phi_f (tissue):  {np.mean(phi_f_all):.3f} ± {np.std(phi_f_all):.3f}  "
          f"[{np.min(phi_f_all):.3f}, {np.max(phi_f_all):.3f}]")
    print(f"    phi_i (inert):   {np.mean(phi_i_all):.3f} ± {np.std(phi_i_all):.3f}  "
          f"[{np.min(phi_i_all):.3f}, {np.max(phi_i_all):.3f}]")
    print(f"    phi_v (void):    {np.mean(phi_v_all):.3f} ± {np.std(phi_v_all):.3f}  "
          f"[{np.min(phi_v_all):.3f}, {np.max(phi_v_all):.3f}]")
    print(f"    phi_solid:       {np.mean(phi_s_all):.3f} ± {np.std(phi_s_all):.3f}  "
          f"[{np.min(phi_s_all):.3f}, {np.max(phi_s_all):.3f}]")

    # By functional ratio
    print(f"\n  By functional ratio:")
    print(f"  {'Ratio':>6s}  {'phi_f':>8s}  {'phi_i':>8s}  {'phi_v':>8s}  {'phi_solid':>10s}  {'enrich_f':>9s}  {'enrich_i':>9s}")
    print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*9}  {'-'*9}")
    for ratio in [0.2, 0.3, 0.5, 0.7]:
        sub = [r for r in records if r['func_ratio'] == ratio]
        pf = [r['phi_f'] for r in sub]
        pi = [r['phi_i'] for r in sub]
        pv = [r['phi_v'] for r in sub]
        ps = [r['phi_solid'] for r in sub]
        ef = [r['enrichment_f'] for r in sub]
        ei = [r['enrichment_i'] for r in sub]
        print(f"  {ratio:6.1f}  {np.mean(pf):8.3f}  {np.mean(pi):8.3f}  "
              f"{np.mean(pv):8.3f}  {np.mean(ps):10.3f}  {np.mean(ef):9.3f}  {np.mean(ei):9.3f}")

    # By size combo
    print(f"\n  By size combination:")
    print(f"  {'Combo':>8s}  {'phi_f':>8s}  {'phi_i':>8s}  {'phi_v':>8s}  {'phi_solid':>10s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")
    combos = sorted(set(r['size_combo'] for r in records))
    for combo in combos:
        sub = [r for r in records if r['size_combo'] == combo]
        pf = [r['phi_f'] for r in sub]
        pi = [r['phi_i'] for r in sub]
        pv = [r['phi_v'] for r in sub]
        ps = [r['phi_solid'] for r in sub]
        print(f"  {combo:>8s}  {np.mean(pf):8.3f}  {np.mean(pi):8.3f}  "
              f"{np.mean(pv):8.3f}  {np.mean(ps):10.3f}")

    print()


# ======================================================================
# Plotting
# ======================================================================

def plot_phase_fractions_vs_ratio(records, outdir):
    """Phase fractions vs functional ratio, colored by size combo."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    phase_keys = [('phi_f', 'Tissue (functional)'), ('phi_i', 'Inert'), ('phi_v', 'Void')]

    for ax, (key, label) in zip(axes, phase_keys):
        for combo in sorted(SIZE_COMBO_COLORS.keys()):
            sub = [r for r in records if r['size_combo'] == combo]
            if not sub:
                continue
            x = [r['func_ratio'] for r in sub]
            y = [r[key] for r in sub]
            ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                       marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                       label=combo, edgecolors='k', linewidths=0.5)

        # Add ideal mixing line
        ratios = np.linspace(0.15, 0.75, 50)
        phi_s_mean = np.mean([r['phi_solid'] for r in records])
        if key == 'phi_f':
            ideal = phi_s_mean * ratios
            ax.plot(ratios, ideal, 'k--', alpha=0.4, lw=1.5, label=f'Ideal (φ_s={phi_s_mean:.2f})')
        elif key == 'phi_i':
            ideal = phi_s_mean * (1.0 - ratios)
            ax.plot(ratios, ideal, 'k--', alpha=0.4, lw=1.5, label=f'Ideal (φ_s={phi_s_mean:.2f})')
        elif key == 'phi_v':
            ax.axhline(1.0 - phi_s_mean, color='k', ls='--', alpha=0.4, lw=1.5,
                        label=f'Ideal void (1−{phi_s_mean:.2f})')

        ax.set_xlabel('Functional Ratio (vol/vol)', fontsize=11)
        ax.set_ylabel(f'{label} Area Fraction', fontsize=11)
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(0, max(0.75, max(r[key] for r in records) * 1.1))
        ax.grid(True, alpha=0.3)

    # Single legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Day 1 Experimental Area Fractions vs Functional Ratio', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'phase_fractions_vs_ratio.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_packing_efficiency(records, outdir):
    """Total solid fraction (packing efficiency) by size combo and ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: phi_solid vs func_ratio by size combo
    ax = axes[0]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in records if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['phi_solid'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=70, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)

    ax.axhline(0.64, color='grey', ls=':', lw=1, label='3D RCP (spheres)')
    ax.axhline(0.84, color='grey', ls='--', lw=1, label='2D RCP (discs)')
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Total Solid Fraction (φ_solid)', fontsize=11)
    ax.set_title('Packing Efficiency', fontsize=12, fontweight='bold')
    ax.set_xlim(0.1, 0.8)
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: phi_solid by size combo (box-like summary)
    ax = axes[1]
    combos = sorted(set(r['size_combo'] for r in records))
    positions = range(len(combos))
    for i, combo in enumerate(combos):
        sub = [r for r in records if r['size_combo'] == combo]
        vals = [r['phi_solid'] for r in sub]
        ax.scatter([i]*len(vals), vals, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=70, alpha=0.8,
                   edgecolors='k', linewidths=0.5)
        ax.plot([i-0.2, i+0.2], [np.mean(vals)]*2, 'k-', lw=2)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(combos, rotation=45, ha='right', fontsize=9)
    ax.axhline(0.64, color='grey', ls=':', lw=1, label='3D RCP')
    ax.set_ylabel('Total Solid Fraction (φ_solid)', fontsize=11)
    ax.set_title('Packing Efficiency by Size Combination', fontsize=12, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Day 1 Packing Efficiency', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'packing_efficiency.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_enrichment(records, outdir):
    """Over/under-representation of phases (stereological bias analysis)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: Functional enrichment
    ax = axes[0]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in records if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['enrichment_f'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)

    ax.axhline(1.0, color='k', ls='-', lw=1, alpha=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Enrichment (actual / expected)', fontsize=11)
    ax.set_title('Functional Phase Enrichment', fontsize=12, fontweight='bold')
    ax.set_xlim(0.1, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    # Right: Inert enrichment
    ax = axes[1]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in records if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['enrichment_i'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)

    ax.axhline(1.0, color='k', ls='-', lw=1, alpha=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Enrichment (actual / expected)', fontsize=11)
    ax.set_title('Inert Phase Enrichment', fontsize=12, fontweight='bold')
    ax.set_xlim(0.1, 0.8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2)

    fig.suptitle('Day 1 Phase Enrichment (area fraction / expected from volume ratio)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'phase_enrichment.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_size_ratio_effects(records, outdir):
    """How granule size ratio affects packing and phase distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    size_ratios = [r['size_ratio'] for r in records]

    # Left: phi_solid vs size ratio
    ax = axes[0]
    for ratio in [0.2, 0.3, 0.5, 0.7]:
        sub = [r for r in records if r['func_ratio'] == ratio]
        x = [r['size_ratio'] for r in sub]
        y = [r['phi_solid'] for r in sub]
        ax.scatter(x, y, c=RATIO_COLORS[ratio], s=60, alpha=0.8,
                   label=f'ratio={ratio}', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Size Ratio (R_func / R_inert)', fontsize=11)
    ax.set_ylabel('φ_solid', fontsize=11)
    ax.set_title('Packing Efficiency vs Size Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: enrichment_f vs size ratio
    ax = axes[1]
    for ratio in [0.2, 0.3, 0.5, 0.7]:
        sub = [r for r in records if r['func_ratio'] == ratio]
        x = [r['size_ratio'] for r in sub]
        y = [r['enrichment_f'] for r in sub]
        ax.scatter(x, y, c=RATIO_COLORS[ratio], s=60, alpha=0.8,
                   label=f'ratio={ratio}', edgecolors='k', linewidths=0.5)
    ax.axhline(1.0, color='k', ls='-', lw=1, alpha=0.5)
    ax.set_xlabel('Size Ratio (R_func / R_inert)', fontsize=11)
    ax.set_ylabel('Functional Enrichment', fontsize=11)
    ax.set_title('Functional Enrichment vs Size Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: void vs size ratio
    ax = axes[2]
    for ratio in [0.2, 0.3, 0.5, 0.7]:
        sub = [r for r in records if r['func_ratio'] == ratio]
        x = [r['size_ratio'] for r in sub]
        y = [r['phi_v'] for r in sub]
        ax.scatter(x, y, c=RATIO_COLORS[ratio], s=60, alpha=0.8,
                   label=f'ratio={ratio}', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Size Ratio (R_func / R_inert)', fontsize=11)
    ax.set_ylabel('Void Fraction', fontsize=11)
    ax.set_title('Void Fraction vs Size Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Size Ratio Effects on Day 1 Packing', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'size_ratio_effects.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_continuity(records, outdir):
    """Phase continuity (largest connected component fraction) analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    cont_keys = [('cont_t', 'Tissue Continuity'), ('cont_i', 'Inert Continuity'),
                 ('cont_v', 'Void Continuity')]

    for ax, (key, label) in zip(axes, cont_keys):
        for combo in sorted(SIZE_COMBO_COLORS.keys()):
            sub = [r for r in records if r['size_combo'] == combo]
            if not sub:
                continue
            x = [r['func_ratio'] for r in sub]
            y = [r[key] for r in sub]
            ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                       marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                       label=combo, edgecolors='k', linewidths=0.5)

        ax.set_xlabel('Functional Ratio', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Day 1 Phase Continuity (fraction of phase in largest connected component)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'phase_continuity.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_continuity_vs_fraction(records, outdir):
    """Continuity vs actual phase fraction (percolation-like analysis)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pairs = [('phi_f', 'cont_t', 'Tissue'), ('phi_i', 'cont_i', 'Inert'),
             ('phi_v', 'cont_v', 'Void')]

    for ax, (phi_key, cont_key, label) in zip(axes, pairs):
        for combo in sorted(SIZE_COMBO_COLORS.keys()):
            sub = [r for r in records if r['size_combo'] == combo]
            if not sub:
                continue
            x = [r[phi_key] for r in sub]
            y = [r[cont_key] for r in sub]
            ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                       marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                       label=combo, edgecolors='k', linewidths=0.5)

        # Percolation threshold reference
        ax.axvline(0.45, color='grey', ls=':', lw=1, alpha=0.5, label='~Percolation (0.45)')
        ax.set_xlabel(f'{label} Area Fraction', fontsize=11)
        ax.set_ylabel(f'{label} Continuity', fontsize=11)
        ax.set_title(f'{label}: Continuity vs Fraction', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 0.75)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.08))

    fig.suptitle('Continuity vs Area Fraction (percolation analysis)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'continuity_vs_fraction.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_simulation_comparison(records, outdir, phi_solid_target=0.70):
    """Compare experimental data with simulation (mean-field) packing predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    ratios_dense = np.linspace(0.15, 0.75, 50)

    # Simulation prediction: ideal packing at target phi_solid
    sim_phi_f = phi_solid_target * ratios_dense
    sim_phi_i = phi_solid_target * (1.0 - ratios_dense)
    sim_phi_v = np.full_like(ratios_dense, 1.0 - phi_solid_target)

    phase_data = [
        ('phi_f', sim_phi_f, 'Tissue (functional)'),
        ('phi_i', sim_phi_i, 'Inert'),
        ('phi_v', sim_phi_v, 'Void'),
    ]

    for ax, (key, sim_pred, label) in zip(axes, phase_data):
        # Simulation prediction
        ax.plot(ratios_dense, sim_pred, 'r-', lw=2.5, alpha=0.7,
                label=f'Sim target (φ_s={phi_solid_target:.2f})', zorder=5)

        # Also show simulation at experimentally-observed mean phi_solid
        phi_s_exp_mean = np.mean([r['phi_solid'] for r in records])
        if key == 'phi_f':
            ax.plot(ratios_dense, phi_s_exp_mean * ratios_dense, 'b--', lw=1.5,
                    alpha=0.6, label=f'Ideal at exp φ_s={phi_s_exp_mean:.2f}')
        elif key == 'phi_i':
            ax.plot(ratios_dense, phi_s_exp_mean * (1.0 - ratios_dense), 'b--', lw=1.5,
                    alpha=0.6, label=f'Ideal at exp φ_s={phi_s_exp_mean:.2f}')
        elif key == 'phi_v':
            ax.axhline(1.0 - phi_s_exp_mean, color='b', ls='--', lw=1.5, alpha=0.6,
                        label=f'Ideal void at exp φ_s={phi_s_exp_mean:.2f}')

        # Experimental data with error-bar-like grouping
        for ratio in [0.2, 0.3, 0.5, 0.7]:
            sub = [r for r in records if r['func_ratio'] == ratio]
            vals = [r[key] for r in sub]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            ax.errorbar(ratio, mean_val, yerr=std_val, fmt='ko', ms=8, capsize=5,
                         capthick=1.5, elinewidth=1.5, zorder=10)
            # Individual points with jitter
            jitter = np.random.RandomState(42).uniform(-0.015, 0.015, len(vals))
            ax.scatter([ratio + j for j in jitter], vals, c='gray', s=25, alpha=0.5, zorder=8)

        ax.set_xlabel('Functional Ratio', fontsize=11)
        ax.set_ylabel(f'{label} Area Fraction', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlim(0.1, 0.8)
        ax.set_ylim(0, max(0.75, max(r[key] for r in records) * 1.1))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Experiment vs Simulation Packing Prediction (Day 1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'experiment_vs_simulation.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_ternary_composition(records, outdir):
    """Ternary composition diagram (phi_f, phi_i, phi_v)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Transform to Cartesian coordinates for ternary plot
    # Using: x = phi_i + phi_v/2, y = phi_v * sqrt(3)/2
    def ternary_to_cart(phi_f, phi_i, phi_v):
        x = phi_i + phi_v / 2.0
        y = phi_v * np.sqrt(3) / 2.0
        return x, y

    # Draw triangle
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(corners[:, 0], corners[:, 1], 'k-', lw=1.5)

    # Grid lines
    for f in np.arange(0.1, 1.0, 0.1):
        # Lines of constant phi_f
        p1 = ternary_to_cart(f, 1-f, 0)
        p2 = ternary_to_cart(f, 0, 1-f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', lw=0.3, alpha=0.5)
        # Lines of constant phi_i
        p1 = ternary_to_cart(0, f, 1-f)
        p2 = ternary_to_cart(1-f, f, 0)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', lw=0.3, alpha=0.5)
        # Lines of constant phi_v
        p1 = ternary_to_cart(1-f, 0, f)
        p2 = ternary_to_cart(0, 1-f, f)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'grey', lw=0.3, alpha=0.5)

    # Corner labels
    ax.text(-0.05, -0.03, 'φ_f = 1\n(all tissue)', ha='center', fontsize=9)
    ax.text(1.05, -0.03, 'φ_i = 1\n(all inert)', ha='center', fontsize=9)
    ax.text(0.5, np.sqrt(3)/2 + 0.03, 'φ_v = 1\n(all void)', ha='center', fontsize=9)

    # Plot experimental data
    for ratio in [0.2, 0.3, 0.5, 0.7]:
        sub = [r for r in records if r['func_ratio'] == ratio]
        for r in sub:
            x, y = ternary_to_cart(r['phi_f'], r['phi_i'], r['phi_v'])
            ax.scatter(x, y, c=RATIO_COLORS[ratio], s=60, alpha=0.8,
                       edgecolors='k', linewidths=0.5, zorder=5)

    # Legend
    for ratio, color in RATIO_COLORS.items():
        ax.scatter([], [], c=color, s=60, label=f'ratio={ratio}', edgecolors='k', linewidths=0.5)
    ax.legend(fontsize=9, title='Func. Ratio')

    # Ideal mixing line for phi_solid=0.7
    phi_s = 0.70
    for ratio in np.linspace(0.0, 1.0, 100):
        pf = phi_s * ratio
        pi = phi_s * (1 - ratio)
        pv = 1 - phi_s
        x, y = ternary_to_cart(pf, pi, pv)
        ax.plot(x, y, 'r.', ms=1, alpha=0.5)
    # Label the ideal line
    x_mid, y_mid = ternary_to_cart(phi_s*0.5, phi_s*0.5, 1-phi_s)
    ax.annotate(f'Ideal φ_s={phi_s}', xy=(x_mid, y_mid),
                xytext=(x_mid+0.1, y_mid+0.08),
                arrowprops=dict(arrowstyle='->', color='red', lw=1),
                fontsize=9, color='red')

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Ternary Composition Diagram (Day 1)', fontsize=13, fontweight='bold')

    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'ternary_composition.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_functional_size_heatmap(records, outdir):
    """Heatmap of packing metrics by functional size × inert size."""
    func_sizes = ['S', 'M', 'L']
    inert_sizes = ['S', 'M', 'L']

    metrics = [
        ('phi_solid', 'φ_solid (mean)', 'Blues'),
        ('phi_v', 'φ_void (mean)', 'Reds'),
        ('enrichment_f', 'Functional Enrichment (mean)', 'RdYlGn'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (key, label, cmap) in zip(axes, metrics):
        data = np.zeros((3, 3))
        for i, fs in enumerate(func_sizes):
            for j, isz in enumerate(inert_sizes):
                sub = [r for r in records if r['func_size'] == fs and r['inert_size'] == isz]
                if sub:
                    data[i, j] = np.mean([r[key] for r in sub])

        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_xticks(range(3))
        ax.set_xticklabels([f'{s}\n(R≈{SIZE_RADII[s]:.0f}µm)' for s in inert_sizes])
        ax.set_yticks(range(3))
        ax.set_yticklabels([f'{s}\n(R≈{SIZE_RADII[s]:.0f}µm)' for s in func_sizes])
        ax.set_xlabel('Inert Size', fontsize=11)
        ax.set_ylabel('Functional Size', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')

        # Annotate cells
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{data[i,j]:.3f}', ha='center', va='center',
                        fontsize=10, fontweight='bold',
                        color='white' if data[i,j] > np.mean(data) else 'black')

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('Packing Metrics by Granule Size Combination (averaged over all ratios)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'size_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Mean-Field Model Comparison
# ======================================================================

def run_mean_field_comparison(records, outdir, E_modulus=10.0, t_total=48.0):
    """Run mean-field compaction model for each experimental condition.

    Uses the experimental Day 1 phase fractions as initial conditions,
    then predicts 48h compaction trajectory. Also computes parity between
    the model's expected initial fractions and what was actually measured.

    Parameters
    ----------
    records : list of dict
        Experimental data records.
    outdir : str
        Output directory.
    E_modulus : float
        Hydrogel Young's modulus in kPa (default 10).
    t_total : float
        Total simulation time in hours (default 48).

    Returns
    -------
    list of dict
        Mean-field results per condition.
    """
    from analysis.mean_field_model import (CompactionModel, MotorClutchParams,
                                           motor_clutch_force)

    mc = MotorClutchParams()

    # Group by unique conditions
    conditions = {}
    for r in records:
        key = (r['size_combo'], r['func_ratio'])
        conditions.setdefault(key, []).append(r)

    mf_results = []
    print(f"\n  Running mean-field model for {len(conditions)} conditions...")

    for (combo, fratio), recs in sorted(conditions.items()):
        rec = recs[0]
        # Use experimental mean phi_solid as initial condition
        exp_phi_s = np.mean([r['phi_solid'] for r in recs])
        phi_f0 = exp_phi_s * fratio
        phi_i0 = exp_phi_s * (1.0 - fratio)

        # Compute domain and granule counts (2D)
        R_func = rec['R_func']
        R_inert = rec['R_inert']
        Lx = 800.0
        domain_area = Lx * Lx
        gran_area_f = np.pi * R_func**2
        N_func = max(1, int(round(phi_f0 * domain_area / gran_area_f)))
        n_cells = 8  # default

        model = CompactionModel(
            E_modulus=E_modulus,
            phi_f0=phi_f0,
            phi_i0=phi_i0,
            R_func=R_func,
            n_cells_per_granule=n_cells,
            N_func=N_func,
            domain_volume=domain_area,
            mc_params=mc,
            phi_RCP=0.64,
        )

        # Solve compaction ODE for 48 hours
        eta_eff = 1.0     # reasonable default viscosity
        sigma_0 = 0.01    # jamming stress prefactor
        alpha = 1.0        # Hertzian exponent
        sol = model.solve((0, t_total), eta_eff, sigma_0, alpha, n_points=200)

        x_f_init = sol.y[0][0]
        x_f_final = sol.y[0][-1]
        phi_v_f_init, phi_v_i_init = model.local_voids(x_f_init)
        phi_v_f_final, phi_v_i_final = model.local_voids(x_f_final)

        # Model's expected initial fractions (ideal mixing at exp phi_solid)
        model_phi_f = phi_f0
        model_phi_i = phi_i0
        model_phi_v = 1.0 - exp_phi_s

        # Compare with experimental data (average over replicates)
        exp_phi_f = np.mean([r['phi_f'] for r in recs])
        exp_phi_i = np.mean([r['phi_i'] for r in recs])
        exp_phi_v = np.mean([r['phi_v'] for r in recs])

        mf_results.append({
            'size_combo': combo,
            'func_ratio': fratio,
            'R_func': R_func,
            'R_inert': R_inert,
            # Model initial conditions (ideal mixing at exp phi_solid)
            'model_phi_f': model_phi_f,
            'model_phi_i': model_phi_i,
            'model_phi_v': model_phi_v,
            'model_phi_solid': exp_phi_s,
            # Experimental data
            'exp_phi_f': exp_phi_f,
            'exp_phi_i': exp_phi_i,
            'exp_phi_v': exp_phi_v,
            'exp_phi_solid': exp_phi_s,
            # Compaction predictions
            'x_f_init': x_f_init,
            'x_f_final': x_f_final,
            'compaction_ratio': x_f_final / max(x_f_init, 1e-6),
            'phi_v_f_final': phi_v_f_final,
            'phi_v_i_final': phi_v_i_final,
            'F_cell': model.F_cell,
            'sigma_cell_24h': model.sigma_cell(24.0),
            'f_bridge_24h': model._bridge_fraction(24.0),
            # Full trajectory
            'sol_t': sol.t,
            'sol_xf': sol.y[0],
        })

    return mf_results


def plot_mean_field_parity(mf_results, outdir):
    """Parity plot: mean-field initial fractions vs experimental."""
    if not mf_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    pairs = [
        ('model_phi_f', 'exp_phi_f', 'Tissue (φ_f)'),
        ('model_phi_i', 'exp_phi_i', 'Inert (φ_i)'),
        ('model_phi_v', 'exp_phi_v', 'Void (φ_v)'),
    ]

    for ax, (mod_key, exp_key, label) in zip(axes, pairs):
        mod_vals = [r[mod_key] for r in mf_results]
        exp_vals = [r[exp_key] for r in mf_results]
        combos = [r['size_combo'] for r in mf_results]

        for m, e, combo in zip(mod_vals, exp_vals, combos):
            ax.scatter(e, m, c=SIZE_COMBO_COLORS.get(combo, 'gray'),
                       marker=SIZE_COMBO_MARKERS.get(combo, 'o'),
                       s=80, alpha=0.8, edgecolors='k', linewidths=0.5)

        all_vals = mod_vals + exp_vals
        lo = min(all_vals) * 0.85
        hi = max(all_vals) * 1.1
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='1:1')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f'Experimental {label}', fontsize=11)
        ax.set_ylabel(f'Mean-Field {label}', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # R² and RMSE
        exp_arr = np.array(exp_vals)
        mod_arr = np.array(mod_vals)
        ss_res = np.sum((exp_arr - mod_arr)**2)
        ss_tot = np.sum((exp_arr - np.mean(exp_arr))**2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
        rmse = np.sqrt(np.mean((exp_arr - mod_arr)**2))
        ax.text(0.05, 0.95, f'R²={r2:.3f}\nRMSE={rmse:.3f}',
                transform=ax.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', fc='wheat', alpha=0.7))

    # Legend from first axis
    handles, labels = [], []
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        if any(r['size_combo'] == combo for r in mf_results):
            handles.append(plt.Line2D([0], [0], marker=SIZE_COMBO_MARKERS[combo],
                           color='w', markerfacecolor=SIZE_COMBO_COLORS[combo],
                           markersize=8, markeredgecolor='k', markeredgewidth=0.5))
            labels.append(combo)
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Mean-Field Model vs Experiment — Initial Packing (Day 1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'mean_field_parity.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_field_compaction_predictions(mf_results, outdir):
    """Predicted compaction trajectories grouped by size combo."""
    if not mf_results:
        return

    # Group by size combo
    combos = sorted(set(r['size_combo'] for r in mf_results))
    n_combos = len(combos)
    ncols = min(3, n_combos)
    nrows = (n_combos + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows), squeeze=False)

    for idx, combo in enumerate(combos):
        ax = axes[idx // ncols][idx % ncols]
        sub = sorted([r for r in mf_results if r['size_combo'] == combo],
                     key=lambda r: r['func_ratio'])

        for r in sub:
            color = RATIO_COLORS[r['func_ratio']]
            ax.plot(r['sol_t'], r['sol_xf'], '-', color=color, lw=2,
                    label=f"ratio={r['func_ratio']:.1f}")
            # Mark initial and final
            ax.plot(0, r['x_f_init'], 'o', color=color, ms=6)
            ax.plot(r['sol_t'][-1], r['x_f_final'], 's', color=color, ms=6)

        ax.set_xlabel('Time (h)', fontsize=10)
        ax.set_ylabel('$x_f$ (functional zone fraction)', fontsize=10)
        ax.set_title(f'{combo} (R_f={sub[0]["R_func"]:.0f}, R_i={sub[0]["R_inert"]:.0f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    # Hide unused axes
    for idx in range(n_combos, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle('Mean-Field Compaction Predictions (48h) — per Size Combination',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'mean_field_compaction_trajectories.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_field_summary(mf_results, outdir):
    """Summary: predicted compaction ratio and bridge fraction by condition."""
    if not mf_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: compaction ratio vs func_ratio
    ax = axes[0]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in mf_results if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['compaction_ratio'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=70, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)
    ax.axhline(1.0, color='grey', ls=':', lw=1)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Compaction Ratio ($x_f^{final}/x_f^{init}$)', fontsize=11)
    ax.set_title('Predicted Compaction (48h)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Middle: bridge fraction at 24h
    ax = axes[1]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in mf_results if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['f_bridge_24h'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=70, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Bridge Fraction at 24h', fontsize=11)
    ax.set_title('Bridge Formation (24h)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: cell stress at 24h
    ax = axes[2]
    for combo in sorted(SIZE_COMBO_COLORS.keys()):
        sub = [r for r in mf_results if r['size_combo'] == combo]
        if not sub:
            continue
        x = [r['func_ratio'] for r in sub]
        y = [r['sigma_cell_24h'] for r in sub]
        ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                   marker=SIZE_COMBO_MARKERS[combo], s=70, alpha=0.8,
                   label=combo, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Cell Stress at 24h (nN/µm²)', fontsize=11)
    ax.set_title('Cell Traction Stress (24h)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Mean-Field Model Predictions by Condition',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'mean_field_summary.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_mean_field_residuals(mf_results, records, outdir):
    """Residual analysis: deviation of experimental from ideal mixing."""
    if not mf_results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # For each record, compute residual = exp - model (ideal mixing)
    residuals_f = []
    residuals_i = []
    residuals_v = []
    size_ratios = []
    func_ratios = []
    combos_list = []

    for r in records:
        phi_s = r['phi_solid']
        ideal_f = phi_s * r['func_ratio']
        ideal_i = phi_s * (1.0 - r['func_ratio'])
        ideal_v = 1.0 - phi_s
        residuals_f.append(r['phi_f'] - ideal_f)
        residuals_i.append(r['phi_i'] - ideal_i)
        residuals_v.append(r['phi_v'] - ideal_v)
        size_ratios.append(r['size_ratio'])
        func_ratios.append(r['func_ratio'])
        combos_list.append(r['size_combo'])

    phase_data = [
        (residuals_f, 'Tissue Residual (exp − ideal)'),
        (residuals_i, 'Inert Residual (exp − ideal)'),
        (residuals_v, 'Void Residual (exp − ideal)'),
    ]

    for ax, (resids, label) in zip(axes, phase_data):
        for combo in sorted(SIZE_COMBO_COLORS.keys()):
            idx_c = [i for i, c in enumerate(combos_list) if c == combo]
            if not idx_c:
                continue
            x = [func_ratios[i] for i in idx_c]
            y = [resids[i] for i in idx_c]
            ax.scatter(x, y, c=SIZE_COMBO_COLORS[combo],
                       marker=SIZE_COMBO_MARKERS[combo], s=60, alpha=0.8,
                       label=combo, edgecolors='k', linewidths=0.5)

        ax.axhline(0, color='k', ls='-', lw=1, alpha=0.5)
        ax.set_xlabel('Functional Ratio', fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label.split('(')[0].strip(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Model Residuals: How Experimental Fractions Deviate from Ideal Mixing',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(Path(outdir) / 'mean_field_residuals.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Main Entry Point
# ======================================================================

def run_comparison(records=None, outdir=None, run_mean_field=True,
                   phi_solid_target=0.70, E_modulus=10.0):
    """Run the full experimental comparison analysis.

    Parameters
    ----------
    records : list of dict or None
        Experimental data. Loaded from Excel if None.
    outdir : str or None
        Output directory. Defaults to results/experimental_comparison.
    run_mean_field : bool
        If True, run mean-field compaction model for each condition.
    phi_solid_target : float
        Simulation target solid fraction for comparison.
    E_modulus : float
        Hydrogel modulus (kPa) for mean-field model.
    """
    if records is None:
        records = load_experimental_data()

    if outdir is None:
        outdir = str(Path(__file__).parent.parent / 'results' / 'experimental_comparison')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    print_summary(records)

    print("  Generating plots...")
    print("  [1/8] Phase fractions vs ratio...")
    plot_phase_fractions_vs_ratio(records, outdir)

    print("  [2/8] Packing efficiency...")
    plot_packing_efficiency(records, outdir)

    print("  [3/8] Phase enrichment...")
    plot_enrichment(records, outdir)

    print("  [4/8] Size ratio effects...")
    plot_size_ratio_effects(records, outdir)

    print("  [5/8] Phase continuity...")
    plot_continuity(records, outdir)
    plot_continuity_vs_fraction(records, outdir)

    print("  [6/8] Simulation comparison (analytical)...")
    plot_simulation_comparison(records, outdir, phi_solid_target=phi_solid_target)

    print("  [7/8] Ternary composition & size heatmap...")
    plot_ternary_composition(records, outdir)
    plot_functional_size_heatmap(records, outdir)

    # Mean-field model comparison
    mf_results = []
    if run_mean_field:
        print("  [8/8] Mean-field model comparison...")
        mf_results = run_mean_field_comparison(records, outdir, E_modulus=E_modulus)
        plot_mean_field_parity(mf_results, outdir)
        plot_mean_field_compaction_predictions(mf_results, outdir)
        plot_mean_field_summary(mf_results, outdir)
        plot_mean_field_residuals(mf_results, records, outdir)
    else:
        print("  [8/8] Skipping mean-field model (--no-mean-field)")

    print("\n" + "=" * 75)
    print("  EXPERIMENTAL COMPARISON COMPLETE")
    print("=" * 75)
    print(f"  Plots saved to: {outdir}/")
    print(f"  Files:")
    print(f"    phase_fractions_vs_ratio.png")
    print(f"    packing_efficiency.png")
    print(f"    phase_enrichment.png")
    print(f"    size_ratio_effects.png")
    print(f"    phase_continuity.png")
    print(f"    continuity_vs_fraction.png")
    print(f"    experiment_vs_simulation.png")
    print(f"    ternary_composition.png")
    print(f"    size_heatmap.png")
    if mf_results:
        print(f"    mean_field_parity.png")
        print(f"    mean_field_compaction_trajectories.png")
        print(f"    mean_field_summary.png")
        print(f"    mean_field_residuals.png")

    return records, mf_results


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compare Day 1 experimental packing data with simulation predictions.')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: results/experimental_comparison)')
    parser.add_argument('--no-mean-field', action='store_true',
                        help='Skip mean-field model comparison')
    parser.add_argument('--phi-solid', type=float, default=0.70,
                        help='Target solid fraction for simulation comparison (default: 0.70)')
    parser.add_argument('--E-modulus', type=float, default=10.0,
                        help='Hydrogel modulus in kPa for mean-field model (default: 10)')
    parser.add_argument('--xlsx', default=None,
                        help='Path to experimental Excel file')
    args = parser.parse_args()

    records = load_experimental_data(args.xlsx)
    run_comparison(records, outdir=args.outdir,
                   run_mean_field=not args.no_mean_field,
                   phi_solid_target=args.phi_solid,
                   E_modulus=args.E_modulus)
