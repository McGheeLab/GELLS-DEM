"""
Phase Fraction Visualization (Module 3)
========================================
Global phase fractions (conservation check) and local phase fractions
within Voronoi cells showing compaction and relaxation of functional/inert spaces.

Usage:
    python viz2/phase_fractions.py -i results/default
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
from pathlib import Path
import argparse

from viz2.common import (
    FUNC_COLOR, INERT_COLOR,
    select_indices, get_snap_time, setup_scaffold_axes,
    ensure_phase_fields, save_fig,
    is_3d, slice_snap_z_midplane, slice_field_z_midplane,
)
from viz2.parallel import pmap
from viz2.voronoi import (
    voronoi_from_centers, compute_local_phase_fractions, polygon_centroid,
    polygon_area,
)


# ======================================================================
# Global phase fractions (conservation check)
# ======================================================================

def plot_global_phases(hist, p, outdir=None):
    """Global phi_f, phi_i, phi_v vs time.

    Two panels:
    (a) Stacked area chart showing conservation (should sum to 1.0)
    (b) Line plots with sum-check as dashed line
    """
    times = [h['time'] for h in hist]
    phi_f = [h.get('phi_f_mean', 0) for h in hist]
    phi_i = [h.get('phi_i_mean', 0) for h in hist]
    phi_v = [h.get('phi_v_mean', 0) for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Stacked area
    ax = axes[0]
    t = np.array(times)
    pf = np.array(phi_f)
    pi = np.array(phi_i)
    pv = np.array(phi_v)

    ax.fill_between(t, 0, pf, color=FUNC_COLOR, alpha=0.7, label=r'$\phi_f$ (functional)')
    ax.fill_between(t, pf, pf + pi, color=INERT_COLOR, alpha=0.7, label=r'$\phi_i$ (inert)')
    ax.fill_between(t, pf + pi, pf + pi + pv, color='#BBBBBB', alpha=0.5, label=r'$\phi_v$ (void)')
    ax.axhline(1.0, ls='--', color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Phase fraction')
    ax.set_title('Stacked Phase Fractions (Conservation)', fontweight='bold')
    ax.legend(fontsize=9, loc='center right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2)

    # (b) Line plots
    ax = axes[1]
    ax.plot(t, pf, '-', color=FUNC_COLOR, lw=2, label=r'$\phi_f$')
    ax.plot(t, pi, '-', color=INERT_COLOR, lw=2, label=r'$\phi_i$')
    ax.plot(t, pv, '-', color='#666666', lw=2, label=r'$\phi_v$')
    total = pf + pi + pv
    ax.plot(t, total, '--', color='k', lw=1.5, alpha=0.6, label=r'$\Sigma$ (should = 1)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Phase fraction')
    ax.set_title('Phase Fraction Evolution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2)

    # Annotate max deviation from 1.0
    dev = np.max(np.abs(total - 1.0))
    ax.annotate(f'max |sum - 1| = {dev:.4f}', xy=(0.02, 0.98),
                xycoords='axes fraction', fontsize=8, va='top',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    fig.suptitle("Global Phase Fractions", fontsize=14)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'global_phases.png')
    return fig


def _local_fractions(args):
    """(polygons, local phase fractions) for one snapshot — the unit of work."""
    snap, p, _3d = args
    phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)
    work_snap = snap
    if _3d:
        phi_f = slice_field_z_midplane(phi_f)
        phi_i = slice_field_z_midplane(phi_i)
        phi_v = slice_field_z_midplane(phi_v)
        work_snap = slice_snap_z_midplane(snap, p)
    polygons, _ = voronoi_from_centers(work_snap, p)
    local_pf = compute_local_phase_fractions(work_snap, p, polygons,
                                             phi_f=phi_f, phi_i=phi_i, phi_v=phi_v)
    return polygons, local_pf


def _inner_outer_means(args):
    """Mean local phi_f/phi_i/phi_v inside and outside the domain core."""
    snap, p, _3d, cx_domain, cy_domain, r_inner = args
    _polygons, local_pf = _local_fractions((snap, p, _3d))
    inner = {'phi_f': [], 'phi_i': [], 'phi_v': []}
    outer = {'phi_f': [], 'phi_i': [], 'phi_v': []}
    for _gi, data in local_pf.items():
        cent = data['centroid']
        dist = np.sqrt((cent[0] - cx_domain) ** 2 + (cent[1] - cy_domain) ** 2)
        target = inner if dist < r_inner else outer
        for k in ('phi_f', 'phi_i', 'phi_v'):
            target[k].append(data[k])
    return ({k: (float(np.mean(v)) if v else 0.0) for k, v in inner.items()},
            {k: (float(np.mean(v)) if v else 0.0) for k, v in outer.items()})


def local_fractions_for(snaps, p, indices, workers=None, label=None):
    """Polygons + local fractions for `indices`, one process per snapshot."""
    _3d = is_3d(snaps[0], p) if snaps else False
    return pmap(_local_fractions, [(snaps[si], p, _3d) for si in indices],
                workers, label=label)


def plot_species_fractions(hist, p, outdir=None):
    """Per-species solid fractions vs time (V3.0), plus the collagen-weighted solid.

    Uses the phi_sp_{k}_mean history keys written by compute_metrics; returns
    None for histories without them (pre-V3.0 runs).
    """
    from viz2.common import species_table
    if not hist or 'phi_sp_0_mean' not in hist[0]:
        return None
    species = species_table(p)
    K = 0
    while f'phi_sp_{K}_mean' in hist[0]:
        K += 1
    t = np.array([h['time'] for h in hist])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bottom = np.zeros_like(t)
    for k in range(K):
        vals = np.array([h.get(f'phi_sp_{k}_mean', 0.0) for h in hist])
        sp = species[k] if k < len(species) else {'name': f'species_{k}', 'color': '#888888', 'f': float('nan')}
        ax.fill_between(t, bottom, bottom + vals, color=sp['color'], alpha=0.75,
                        label=f"{sp['name']} (f={sp['f']:.2f})")
        bottom = bottom + vals
    pv = np.array([h.get('phi_v_mean', 0.0) for h in hist])
    ax.fill_between(t, bottom, bottom + pv, color='#BBBBBB', alpha=0.5, label='void')
    ax.axhline(1.0, ls='--', color='k', lw=0.8, alpha=0.5)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Fraction of domain')
    ax.set_title('Species Fractions (stacked)', fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc='center right')
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for k in range(K):
        vals = np.array([h.get(f'phi_sp_{k}_mean', 0.0) for h in hist])
        sp = species[k] if k < len(species) else {'name': f'species_{k}', 'color': '#888888'}
        ax.plot(t, vals, '-', color=sp['color'], lw=2, label=sp['name'])
    if 'phi_c_mean' in hist[0]:
        pc = np.array([h.get('phi_c_mean', 0.0) for h in hist])
        ax.plot(t, pc, '--', color='k', lw=1.5, label='collagen-weighted solid')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Fraction of domain')
    ax.set_title('Species Fraction Evolution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle('Granule Species (V3.0)', fontsize=14)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'species_fractions.png')
    return fig


# ======================================================================
# Local phase fraction heatmaps
# ======================================================================

def plot_local_phase_heatmaps(snaps, hist, p, indices=None, outdir=None,
                              workers=None, precomputed=None):
    """Voronoi cells colored by local phi_f, phi_i, phi_v at selected times.

    3 columns (phi_f, phi_i, phi_v) x N timepoint rows.
    """
    indices = select_indices(len(snaps), indices)
    n_rows = len(indices)
    fields = ['phi_f', 'phi_i', 'phi_v']
    field_labels = [r'Local $\phi_f$ (functional)', r'Local $\phi_i$ (inert)',
                    r'Local $\phi_v$ (void)']
    cmaps = ['Reds', 'Greens', 'Greys']

    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    tess = precomputed if precomputed is not None else local_fractions_for(
        snaps, p, indices, workers=workers, label='local phase heatmaps')

    for row, si in enumerate(indices):
        t = get_snap_time(hist, si)
        polygons, local_pf = tess[row]

        for col, (field, label, cmap_name) in enumerate(
                zip(fields, field_labels, cmaps)):
            ax = axes[row, col]
            setup_scaffold_axes(ax, p, dark=False)
            ax.set_facecolor('#F5F5F5')

            values = [local_pf[k][field] for k in polygons if k in local_pf]
            if not values:
                continue

            vmin = 0.0
            vmax = max(max(values), 0.01)

            cmap = plt.get_cmap(cmap_name)
            patches = []
            colors = []
            for gi, verts in polygons.items():
                if gi not in local_pf:
                    continue
                patches.append(MplPolygon(verts, closed=True))
                val = local_pf[gi][field]
                colors.append(cmap(val / vmax))

            pc = PatchCollection(patches, facecolors=colors,
                                 edgecolors='#333333', linewidths=0.3)
            ax.add_collection(pc)

            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"t = {t:.1f} h", fontsize=9)

            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin, vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Local Phase Fractions (Voronoi)", fontsize=13, y=1.02)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'local_phase_heatmaps.png')
    return fig


# ======================================================================
# Local phase fraction timeseries (inner vs outer)
# ======================================================================

def plot_local_phase_timeseries(snaps, hist, p, outdir=None, workers=None):
    """Mean local phi_f/phi_i/phi_v in inner vs outer Voronoi cells over time.

    Inner cells: centroid within 40% of domain center.
    Outer cells: centroid outside 40% of domain center.
    """
    cx_domain = p.Lx / 2.0
    cy_domain = p.Ly / 2.0
    r_inner = 0.4 * min(p.Lx, p.Ly) / 2.0
    _3d = is_3d(snaps[0], p) if snaps else False

    times = [get_snap_time(hist, si) for si in range(len(snaps))]
    inner_stats = {'phi_f': [], 'phi_i': [], 'phi_v': []}
    outer_stats = {'phi_f': [], 'phi_i': [], 'phi_v': []}

    per_snap = pmap(_inner_outer_means,
                    [(snaps[si], p, _3d, cx_domain, cy_domain, r_inner)
                     for si in range(len(snaps))],
                    workers, label='local phase timeseries')
    for inner, outer in per_snap:
        for k in ('phi_f', 'phi_i', 'phi_v'):
            inner_stats[k].append(inner[k])
            outer_stats[k].append(outer[k])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t = np.array(times)

    for col, (field, label, c_in, c_out) in enumerate([
        ('phi_f', r'$\phi_f$ (functional)', FUNC_COLOR, '#FF8888'),
        ('phi_i', r'$\phi_i$ (inert)', INERT_COLOR, '#88DD88'),
        ('phi_v', r'$\phi_v$ (void)', '#666666', '#BBBBBB'),
    ]):
        ax = axes[col]
        ax.plot(t, inner_stats[field], '-', color=c_in, lw=2, label='Inner')
        ax.plot(t, outer_stats[field], '--', color=c_out, lw=2, label='Outer')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Local Phase Fractions: Inner vs Outer Voronoi Cells",
                 fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'local_phase_timeseries.png')
    return fig


# ======================================================================
# Local vs global scatter
# ======================================================================

def plot_local_vs_global(snaps, hist, p, indices=None, outdir=None,
                         workers=None, precomputed=None):
    """Scatter of local phi_f vs global phi_f for each Voronoi cell.

    Shows heterogeneity of compaction at selected timepoints.
    """
    indices = select_indices(len(snaps), indices)
    n = len(indices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    tess = precomputed if precomputed is not None else local_fractions_for(
        snaps, p, indices, workers=workers, label='local vs global')

    for c, si in enumerate(indices):
        ax = axes[c]
        t = get_snap_time(hist, si)

        phi_f_global = hist[si].get('phi_f_mean', 0) if si < len(hist) else 0
        _polygons, local_pf = tess[c]

        local_vals = [local_pf[k]['phi_f'] for k in local_pf]
        areas = [local_pf[k]['area'] for k in local_pf]

        if local_vals:
            ax.scatter(local_vals, areas, c=local_vals, cmap='Reds',
                       s=30, alpha=0.7, edgecolors='#333333', linewidths=0.3)
            ax.axvline(phi_f_global, ls='--', color=FUNC_COLOR, lw=1.5,
                       label=f'Global $\\phi_f$ = {phi_f_global:.3f}')
            ax.legend(fontsize=8)

        ax.set_xlabel(r'Local $\phi_f$')
        ax.set_ylabel(r'Voronoi cell area ($\mu m^2$)')
        ax.set_title(f"t = {t:.1f} h", fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Local vs Global Phase Fraction Heterogeneity", fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'local_vs_global.png')
    return fig


# ======================================================================
# run_all
# ======================================================================

# Above this many granules the Voronoi-based local plots (one tessellation per
# snapshot, clipped granule by granule in Python) take longer than the simulation
# itself; the global and per-species panels are always drawn.
LOCAL_VORONOI_MAX_GRANULES = 20000


def run_all(snaps, hist, p, outdir=None, workers=None):
    """Generate all phase fraction visualizations."""
    print("  [3/6] Phase fractions...")

    fig1 = plot_global_phases(hist, p, outdir=outdir)
    plt.close(fig1)

    fig1b = plot_species_fractions(hist, p, outdir=outdir)
    if fig1b is not None:
        plt.close(fig1b)

    n_gran = len(snaps[0]['x']) if snaps else 0
    if n_gran > LOCAL_VORONOI_MAX_GRANULES:
        print(f"    Skipping local Voronoi phase plots: {n_gran} granules > "
              f"{LOCAL_VORONOI_MAX_GRANULES} (LOCAL_VORONOI_MAX_GRANULES)")
        return

    # the heatmaps and the local-vs-global scatter key on the same tessellation
    # of the same snapshots: compute it once, in parallel, and share it
    indices = select_indices(len(snaps), None)
    tess = local_fractions_for(snaps, p, indices, workers=workers,
                               label='local phase fractions')

    fig2 = plot_local_phase_heatmaps(snaps, hist, p, indices=indices, outdir=outdir,
                                     workers=workers, precomputed=tess)
    plt.close(fig2)

    fig3 = plot_local_phase_timeseries(snaps, hist, p, outdir=outdir, workers=workers)
    plt.close(fig3)

    fig4 = plot_local_vs_global(snaps, hist, p, indices=indices, outdir=outdir,
                                workers=workers, precomputed=tess)
    plt.close(fig4)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase fraction viz')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='Per-snapshot worker processes (default: auto, 1 = serial)')
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir, workers=args.workers)
