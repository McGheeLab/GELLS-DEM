#!/usr/bin/env python3
"""
Energy Landscape Visualization
===============================

Generates figures showing the free energy landscape for cell-driven
granular scaffold compaction, decomposed into six physical mechanisms.

Figures produced:
  1. energy_landscape_organs.png     — G(ξ) per organ with all 6 terms
  2. energy_landscape_evolution.png  — Time-evolving landscape (one organ)
  3. energy_decomposition.png        — Bar chart of term magnitudes at ξ*
  4. energy_design_space.png         — Dimensionless group map across organs

Usage:
    python viz/energy_landscape.py -i results/parameter_sweep
"""

import sys, os, json, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.energy_landscape import EnergyLandscape

# ======================================================================
# Configuration
# ======================================================================

ORGANS_TO_SHOW = [
    'trabecular_bone',
    'intestinal_mucosa',
    'kidney_cortex',
    'cardiac_muscle',
]
ORGAN_LABELS = {
    'trabecular_bone':   'Trabecular Bone',
    'intestinal_mucosa': 'Intestinal Mucosa',
    'kidney_cortex':     'Kidney Cortex',
    'cardiac_muscle':    'Cardiac Muscle',
    'liver':             'Liver',
    'lung_alveoli':      'Lung Alveoli',
    'pancreatic_islet':  'Pancreatic Islet',
}

# Colours for energy terms
C_TOTAL   = '#212121'   # near-black
C_CELL    = '#2E7D32'   # green
C_ELASTIC = '#1565C0'   # blue
C_YIELD   = '#C62828'   # red
C_VOID    = '#6A1B9A'   # purple
C_INERT   = '#795548'   # brown
C_SURFACE = '#E65100'   # orange

TERM_STYLE = {
    'G_cell':    {'color': C_CELL,    'ls': '-',  'lw': 1.8,
                  'label': '$G_{\\mathrm{cell}}$ (traction)'},
    'G_elastic': {'color': C_ELASTIC, 'ls': '-',  'lw': 1.8,
                  'label': '$G_{\\mathrm{elastic}}$ (Hertz)'},
    'G_yield':   {'color': C_YIELD,   'ls': '--', 'lw': 1.5,
                  'label': '$G_{\\mathrm{yield}}$ (H-B)'},
    'G_void':    {'color': C_VOID,    'ls': '-.', 'lw': 1.5,
                  'label': '$G_{\\mathrm{void}}$ (osmotic)'},
    'G_inert':   {'color': C_INERT,   'ls': ':',  'lw': 1.8,
                  'label': '$G_{\\mathrm{inert}}$ (frustration)'},
    'G_surface': {'color': C_SURFACE, 'ls': '-',  'lw': 1.5,
                  'label': '$G_{\\gamma}$ (interfacial)'},
}

TERMS = ['G_cell', 'G_elastic', 'G_yield', 'G_void', 'G_inert', 'G_surface']


# ======================================================================
# Figure 1: Energy landscape per organ
# ======================================================================

def _make_landscape_figure(landscapes, outdir):
    """2x2 grid: G(ξ) with 6 decomposed terms per organ."""

    n = len(landscapes)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows),
                             squeeze=False)

    for idx, (organ, (el, D_arch)) in enumerate(landscapes.items()):
        ax = axes[idx // ncols, idx % ncols]
        lbl = ORGAN_LABELS.get(organ, organ)

        data = el.compute_landscape(n_pts=400, f_bridge=1.0, maturity=1.0)
        xi = data['xi']

        # Individual terms
        for term in TERMS:
            s = TERM_STYLE[term]
            ax.plot(xi, data[term], color=s['color'], ls=s['ls'],
                    lw=s['lw'], label=s['label'], alpha=0.85)

        # Total (thick)
        ax.plot(xi, data['G_total'], color=C_TOTAL, lw=2.5,
                label='$G_{\\mathrm{total}}$', zorder=5)

        # Mark equilibrium
        xi_star, G_star = el.find_equilibrium(f_bridge=1.0, maturity=1.0)
        ax.plot(xi_star, G_star, 'o', color=C_TOTAL, ms=8, zorder=6,
                markeredgecolor='white', markeredgewidth=1.5)
        # Adaptive annotation offset
        y_range = max(data['G_total']) - min(data['G_total'])
        y_off = 0.08 * max(y_range, 0.1)
        x_off = 0.06 * el.xi_max
        ax.annotate(f'$\\xi^* = {xi_star:.2f}$',
                    xy=(xi_star, G_star),
                    xytext=(xi_star + x_off, G_star + y_off),
                    fontsize=9, fontweight='bold', color=C_TOTAL,
                    arrowprops=dict(arrowstyle='->', color=C_TOTAL, lw=1.2))

        # Mark barrier if present
        xi_b, DG = el.find_barrier(f_bridge=1.0, maturity=1.0)
        if DG > 0.01:
            s = max(el.sigma_cell, 1e-12)
            G_b_norm = el.G_total(xi_b) / s
            ax.annotate(f'$\\Delta \\tilde{{G}}^* = {DG:.2f}$',
                        xy=(xi_b, G_b_norm),
                        xytext=(xi_b + x_off, G_b_norm + y_off * 0.5),
                        fontsize=8, color=C_YIELD,
                        arrowprops=dict(arrowstyle='->', color=C_YIELD,
                                        lw=1.0, ls='--'))

        ax.axhline(0, color='#9E9E9E', lw=0.5, zorder=0)
        ax.set_xlabel('Compaction coordinate  $\\xi$', fontsize=10)
        ax.set_ylabel('$\\tilde{G} = G / \\sigma_{cell}$', fontsize=10)
        ax.set_title(f'{lbl}  ($D_{{arch}} = {D_arch:.1f}$)',
                     fontsize=12, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(True, alpha=0.2)

        if idx == 0:
            ax.legend(fontsize=7.5, loc='best', ncol=2, framealpha=0.9,
                      edgecolor='#BDBDBD')

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle('Free Energy Landscape for Cell-Driven Scaffold Compaction',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(outdir, 'energy_landscape_organs.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Figure 2: Time-evolving landscape
# ======================================================================

def _make_evolution_figure(landscapes, outdir):
    """Show how the landscape evolves as cells mature and bridges form."""

    # Pick one representative organ (most compaction)
    best_organ = None
    best_xi = 0
    for organ, (el, D_arch) in landscapes.items():
        xi_s, _ = el.find_equilibrium(1.0, 1.0)
        if xi_s > best_xi:
            best_xi = xi_s
            best_organ = organ

    if best_organ is None:
        return None

    el, D_arch = landscapes[best_organ]
    lbl = ORGAN_LABELS.get(best_organ, best_organ)

    # Solve kinetics to get f_bridge(t) and maturity(t)
    t_arr, xi_arr, fb_arr, mat_arr = el.solve_kinetics(t_total=72.0, dt=0.1)

    # Snapshots at different times
    snap_times = [0, 3, 8, 18, 36, 72]
    cmap = plt.colormaps.get_cmap('viridis').resampled(len(snap_times))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={'width_ratios': [1.4, 1]})

    # Left: evolving G(ξ)
    for si, st in enumerate(snap_times):
        tidx = min(np.searchsorted(t_arr, st), len(t_arr) - 1)
        fb = fb_arr[tidx]
        mat = mat_arr[tidx]

        data = el.compute_landscape(n_pts=300, f_bridge=fb, maturity=mat)
        xi = data['xi']
        c = cmap(si / max(len(snap_times) - 1, 1))

        ax1.plot(xi, data['G_total'], color=c, lw=2.0,
                 label=f't = {st} h  ($f_b$={fb:.2f})')

        # Mark equilibrium for this time
        xi_s, G_s = el.find_equilibrium(fb, mat)
        ax1.plot(xi_s, G_s, 'o', color=c, ms=7,
                 markeredgecolor='white', markeredgewidth=1.0, zorder=5)

    ax1.axhline(0, color='#9E9E9E', lw=0.5, zorder=0)
    ax1.set_xlabel('Compaction coordinate  $\\xi$', fontsize=11)
    ax1.set_ylabel('$\\tilde{G}_{total} = G / \\sigma_{cell}$',
                   fontsize=11)
    ax1.set_title(f'Evolving Energy Landscape — {lbl}', fontsize=13,
                  fontweight='bold')
    ax1.legend(fontsize=8.5, loc='upper right', framealpha=0.9)
    ax1.grid(True, alpha=0.2)
    ax1.tick_params(labelsize=9)

    # Right: trajectory ξ(t) on the landscape
    ax2.plot(t_arr, xi_arr, '-', color=C_CELL, lw=2.5,
             label='$\\xi(t)$')
    ax2.fill_between(t_arr, 0, xi_arr, color=C_CELL, alpha=0.08)

    ax2_r = ax2.twinx()
    ax2_r.plot(t_arr, fb_arr, '--', color=C_YIELD, lw=1.5, alpha=0.7,
               label='$f_{bridge}$')
    ax2_r.plot(t_arr, mat_arr, ':', color=C_VOID, lw=1.5, alpha=0.7,
               label='maturity')
    ax2_r.set_ylabel('Bridge fraction / Maturity', fontsize=10,
                     color='#616161')
    ax2_r.set_ylim(0, 1.1)
    ax2_r.tick_params(labelsize=8, colors='#616161')

    ax2.set_xlabel('Time (h)', fontsize=11)
    ax2.set_ylabel('Compaction  $\\xi(t)$', fontsize=11)
    ax2.set_title('Kinetics on the Landscape', fontsize=13,
                  fontweight='bold')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.2)
    ax2.tick_params(labelsize=9)

    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8.5,
               loc='center right', framealpha=0.9)

    fig.suptitle('Time Evolution of the Compaction Energy Landscape',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(outdir, 'energy_landscape_evolution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Figure 3: Energy decomposition bar chart
# ======================================================================

def _make_decomposition_figure(landscapes, outdir):
    """Stacked bar chart: magnitude of each energy term at ξ* per organ."""

    organs = list(landscapes.keys())
    labels = [ORGAN_LABELS.get(o, o) for o in organs]
    n = len(organs)

    # Collect decompositions
    decomp = {}
    for organ, (el, D_arch) in landscapes.items():
        decomp[organ] = el.energy_decomposition_at_eq(f_bridge=1.0,
                                                       maturity=1.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # --- Left: stacked bars (absolute magnitudes) ---
    x_pos = np.arange(n)
    bar_w = 0.55

    # Separate driving (negative) and resisting (positive) terms
    driving_terms = ['G_cell', 'G_surface']  # can be negative
    resist_terms = ['G_elastic', 'G_yield', 'G_void', 'G_inert']

    # Positive (resisting) stacked bars
    bottoms_pos = np.zeros(n)
    for term in resist_terms:
        vals = np.array([max(0, decomp[o][term]) for o in organs])
        s = TERM_STYLE[term]
        ax1.bar(x_pos, vals, bar_w, bottom=bottoms_pos,
                color=s['color'], alpha=0.85, label=s['label'],
                edgecolor='white', linewidth=0.5)
        bottoms_pos += vals

    # Negative (driving) stacked bars
    bottoms_neg = np.zeros(n)
    for term in driving_terms:
        vals = np.array([min(0, decomp[o][term]) for o in organs])
        s = TERM_STYLE[term]
        ax1.bar(x_pos, vals, bar_w, bottom=bottoms_neg,
                color=s['color'], alpha=0.85, label=s['label'],
                edgecolor='white', linewidth=0.5)
        bottoms_neg += vals

    # ξ* labels on bars
    for i, organ in enumerate(organs):
        xi_s = decomp[organ]['xi_star']
        y_top = bottoms_pos[i]
        ax1.text(i, y_top + 0.001, f'$\\xi^*$={xi_s:.2f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.axhline(0, color='#424242', lw=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(labels, fontsize=9, fontweight='bold')
    ax1.set_ylabel('$\\tilde{G}$ at $\\xi^*$  ($G / \\sigma_{cell}$)',
                   fontsize=10)
    ax1.set_title('Energy Decomposition at Equilibrium', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=7.5, loc='upper left', framealpha=0.9, ncol=2)
    ax1.grid(True, axis='y', alpha=0.2)
    ax1.tick_params(labelsize=9)

    # --- Right: dimensionless numbers radar/table ---
    dim_data = {}
    for organ, (el, D_arch) in landscapes.items():
        dim_data[organ] = el.dimensionless_groups(f_bridge=1.0, maturity=1.0)

    dim_names = ['$\\beta$\n(cell/elastic)', '$Ca$\n(cell/yield)',
                 '$\\Phi_r$\n(inert ratio)', '$\\Psi$\n(packing)',
                 '$\\Gamma$\n(interfacial)']
    dim_keys = ['beta', 'Ca', 'Phi_r', 'Psi', 'Gamma']

    cmap_organ = plt.colormaps.get_cmap('Set1').resampled(max(n, 3))
    x_dim = np.arange(len(dim_keys))
    bar_w2 = 0.8 / n

    for i, organ in enumerate(organs):
        vals = [dim_data[organ][k] for k in dim_keys]
        # Log scale for better visibility
        vals_log = [np.log10(max(v, 1e-8)) for v in vals]
        ax2.bar(x_dim + i * bar_w2, vals_log, bar_w2,
                color=cmap_organ(i), alpha=0.85,
                label=ORGAN_LABELS.get(organ, organ),
                edgecolor='white', linewidth=0.5)

    ax2.set_xticks(x_dim + bar_w2 * (n - 1) / 2)
    ax2.set_xticklabels(dim_names, fontsize=8)
    ax2.set_ylabel('$\\log_{10}$ (dimensionless number)', fontsize=10)
    ax2.set_title('Governing Dimensionless Groups', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=8, loc='best', framealpha=0.9)
    ax2.grid(True, axis='y', alpha=0.2)
    ax2.tick_params(labelsize=9)
    ax2.axhline(0, color='#9E9E9E', lw=0.5)

    fig.suptitle('Energy Balance and Design Parameters per Organ Target',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    path = os.path.join(outdir, 'energy_decomposition.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Figure 4: Design space — ξ* sensitivity to key parameters
# ======================================================================

def _make_design_figure(landscapes, outdir):
    """Heat maps showing how ξ* varies with key parameter pairs."""

    # Pick one representative organ
    ref_organ = list(landscapes.keys())[2]  # kidney cortex
    el, D_arch = landscapes[ref_organ]
    lbl = ORGAN_LABELS.get(ref_organ, ref_organ)
    base_params = dict(el.params)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    def _sweep_2d(param_x, x_range, param_y, y_range, n=30):
        """Sweep two parameters and compute ξ* on a grid."""
        xs = np.linspace(*x_range, n)
        ys = np.linspace(*y_range, n)
        Z = np.zeros((n, n))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                p = dict(base_params)
                p[param_x] = float(xv)
                p[param_y] = float(yv)
                try:
                    el_tmp = EnergyLandscape(p)
                    xi_s, _ = el_tmp.find_equilibrium(f_bridge=1.0,
                                                       maturity=1.0)
                    Z[i, j] = xi_s
                except Exception:
                    Z[i, j] = np.nan
        return xs, ys, Z

    # Panel 1: E_func vs phi_f
    print("    Sweeping E_func vs phi_f...", end='', flush=True)
    xs, ys, Z = _sweep_2d('E_func', (1.0, 20.0),
                           'phi_f', (0.08, 0.45), n=35)
    im = axes[0].pcolormesh(xs, ys, Z, cmap='YlGn', shading='auto')
    axes[0].set_xlabel('$E_{func}$ (kPa)', fontsize=11)
    axes[0].set_ylabel('$\\phi_f$', fontsize=11)
    axes[0].set_title('$\\xi^*$ vs Stiffness & Composition', fontsize=11,
                      fontweight='bold')
    # Mark base point
    axes[0].plot(base_params['E_func'], base_params['phi_f'], '*',
                 color='red', ms=14, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=5)
    plt.colorbar(im, ax=axes[0], label='$\\xi^*$', shrink=0.85)
    print(' done')

    # Panel 2: phi_i vs R_func
    print("    Sweeping phi_i vs R_func...", end='', flush=True)
    xs, ys, Z = _sweep_2d('phi_i', (0.02, 0.35),
                           'R_func', (25.0, 70.0), n=35)
    im = axes[1].pcolormesh(xs, ys, Z, cmap='YlGn', shading='auto')
    axes[1].set_xlabel('$\\phi_i$', fontsize=11)
    axes[1].set_ylabel('$R_{func}$ ($\\mu$m)', fontsize=11)
    axes[1].set_title('$\\xi^*$ vs Inert Fraction & Size', fontsize=11,
                      fontweight='bold')
    axes[1].plot(base_params['phi_i'], base_params['R_func'], '*',
                 color='red', ms=14, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=5)
    plt.colorbar(im, ax=axes[1], label='$\\xi^*$', shrink=0.85)
    print(' done')

    # Panel 3: E_func vs E_inert
    print("    Sweeping E_func vs E_inert...", end='', flush=True)
    xs, ys, Z = _sweep_2d('E_func', (1.0, 20.0),
                           'E_inert', (1.0, 25.0), n=35)
    im = axes[2].pcolormesh(xs, ys, Z, cmap='YlGn', shading='auto')
    axes[2].set_xlabel('$E_{func}$ (kPa)', fontsize=11)
    axes[2].set_ylabel('$E_{inert}$ (kPa)', fontsize=11)
    axes[2].set_title('$\\xi^*$ vs Modulus Mismatch', fontsize=11,
                      fontweight='bold')
    axes[2].plot(base_params['E_func'], base_params['E_inert'], '*',
                 color='red', ms=14, markeredgecolor='white',
                 markeredgewidth=1.5, zorder=5)
    # Equal-modulus line
    lim = [1, min(20, 25)]
    axes[2].plot(lim, lim, '--', color='white', lw=1.5, alpha=0.7)
    plt.colorbar(im, ax=axes[2], label='$\\xi^*$', shrink=0.85)
    print(' done')

    for ax in axes:
        ax.tick_params(labelsize=9)

    fig.suptitle(f'Design Space for Equilibrium Compaction — {lbl}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    path = os.path.join(outdir, 'energy_design_space.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Public API
# ======================================================================

def make_energy_figures(recommendations, outdir, t_total=72.0):
    """Generate all energy landscape figures.

    Parameters
    ----------
    recommendations : dict
        organ -> list of recommendation dicts (from parameter_sweep).
    outdir : str
        Output directory.

    Returns
    -------
    list of str
        Paths to saved figures.
    """
    organs = [o for o in ORGANS_TO_SHOW if o in recommendations]
    if not organs:
        print("  Warning: no matching organs in recommendations")
        return []

    os.makedirs(outdir, exist_ok=True)
    saved = []

    # Build landscape for each organ
    landscapes = {}
    for organ in organs:
        rec = recommendations[organ][0]
        p = rec['params']
        D_arch = rec['D_arch']
        el = EnergyLandscape(p)
        landscapes[organ] = (el, D_arch)
        lbl = ORGAN_LABELS.get(organ, organ)
        xi_s, G_s = el.find_equilibrium(1.0, 1.0)
        print(f"  {lbl}: xi_max={el.xi_max:.3f}, "
              f"xi*={xi_s:.3f}, G*={G_s:.4f} kPa")

    # Generate figures
    print("\n  [1/4] Energy landscape per organ...")
    p1 = _make_landscape_figure(landscapes, outdir)
    if p1:
        saved.append(p1)

    print("\n  [2/4] Time-evolving landscape...")
    p2 = _make_evolution_figure(landscapes, outdir)
    if p2:
        saved.append(p2)

    print("\n  [3/4] Energy decomposition...")
    p3 = _make_decomposition_figure(landscapes, outdir)
    if p3:
        saved.append(p3)

    print("\n  [4/4] Design space maps...")
    p4 = _make_design_figure(landscapes, outdir)
    if p4:
        saved.append(p4)

    return saved


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Energy landscape visualization')
    parser.add_argument('-i', '--input', default='results/parameter_sweep',
                        help='Directory containing recommendations.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: same as input)')
    args = parser.parse_args()

    indir = args.input
    outdir = args.output or indir

    rec_path = os.path.join(indir, 'recommendations.json')
    if not os.path.isfile(rec_path):
        print(f"Error: {rec_path} not found. Run parameter_sweep.py first.")
        sys.exit(1)

    with open(rec_path) as f:
        recommendations = json.load(f)

    print("=" * 60)
    print("  Energy Landscape Analysis")
    print("=" * 60)

    paths = make_energy_figures(recommendations, outdir)

    print("\n" + "=" * 60)
    print(f"  Done. {len(paths)} figures saved.")
    print("=" * 60)


if __name__ == '__main__':
    main()
