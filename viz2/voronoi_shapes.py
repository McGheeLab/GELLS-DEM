"""
Voronoi Shape Factor Visualization (Module 2)
===============================================
Voronoi tessellation overlays on scaffold maps, plus shape factor analysis
(circularity, elongation, area) for functional granule territory.

Usage:
    python viz2/voronoi_shapes.py -i results/default
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
    select_indices, get_snap_time, setup_scaffold_axes,
    draw_granule_patches, draw_cells_on_ax, save_fig,
    is_3d, slice_snap_z_midplane,
)
from viz2.voronoi import (
    voronoi_from_centers, voronoi_from_boundaries,
    compute_shape_factors,
)


# ======================================================================
# Voronoi overlay on scaffold map
# ======================================================================

def plot_voronoi_overlay(snaps, hist, p, indices=None, mode='centers',
                         outdir=None):
    """Scaffold map with Voronoi tessellation edges overlaid in white.

    Parameters
    ----------
    mode : str
        'centers' — standard Voronoi from granule centers.
        'boundary' — shrink-wrap Voronoi from granule surface points.
        'both' — side-by-side comparison (doubles columns).
    """
    indices = select_indices(len(snaps), indices)

    if mode == 'both':
        modes = ['centers', 'boundary']
        nc = len(indices) * 2
        fig, axes = plt.subplots(2, len(indices),
                                 figsize=(4.5 * len(indices), 9))
        if len(indices) == 1:
            axes = axes.reshape(2, 1)
    else:
        modes = [mode]
        nc = len(indices)
        fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
        if nc == 1:
            axes = np.array([axes])
        axes = axes.reshape(1, -1)

    _3d = is_3d(snaps[0], p) if snaps else False

    for row, m in enumerate(modes):
        for col, si in enumerate(indices):
            ax = axes[row, col]
            snap = snaps[si]
            t = get_snap_time(hist, si)

            # 3D: slice to z-midplane for 2D Voronoi
            draw_snap = slice_snap_z_midplane(snap, p) if _3d else snap

            # Draw scaffold background
            setup_scaffold_axes(ax, p, dark=True)
            draw_granule_patches(ax, draw_snap, p)
            draw_cells_on_ax(ax, draw_snap, p)

            # Compute Voronoi
            if m == 'boundary':
                polygons, _ = voronoi_from_boundaries(draw_snap, p)
            else:
                polygons, _ = voronoi_from_centers(draw_snap, p)

            # Draw Voronoi edges
            for gi, verts in polygons.items():
                closed = np.vstack([verts, verts[:1]])
                ax.plot(closed[:, 0], closed[:, 1], '-', color='white',
                        lw=0.8, alpha=0.6)

            title = f"t = {t:.1f} h"
            if mode == 'both':
                title = f"{m.title()} | {title}"
            ax.set_title(title, fontsize=9, color='white')

    fig.patch.set_facecolor('black')
    fig.suptitle("Voronoi Tessellation Overlay", fontsize=12, y=1.02,
                 color='white')
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'voronoi_overlay.png')
    return fig


# ======================================================================
# Shape factor spatial maps
# ======================================================================

def plot_shape_factor_maps(snaps, hist, p, indices=None, outdir=None):
    """Voronoi cells colored by shape factor: circularity, elongation, area.

    3 columns x N timepoint rows.
    """
    indices = select_indices(len(snaps), indices)
    n_rows = len(indices)
    metrics = ['circularity', 'elongation', 'area']
    cmaps = ['RdYlGn', 'RdYlBu_r', 'viridis']
    labels = ['Circularity (4piA/P^2)', 'Elongation (1 - b/a)', 'Area (um^2)']

    fig, axes = plt.subplots(n_rows, 3, figsize=(13.5, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    _3d = is_3d(snaps[0], p) if snaps else False

    for row, si in enumerate(indices):
        snap = snaps[si]
        t = get_snap_time(hist, si)
        work_snap = slice_snap_z_midplane(snap, p) if _3d else snap
        polygons, _ = voronoi_from_centers(work_snap, p)
        sf = compute_shape_factors(polygons)

        for col, (metric, cmap_name, label) in enumerate(
                zip(metrics, cmaps, labels)):
            ax = axes[row, col]
            setup_scaffold_axes(ax, p, dark=False)
            ax.set_facecolor('#F5F5F5')

            values = [sf[k][metric] for k in polygons]
            if not values:
                continue

            vmin = min(values)
            vmax = max(values)
            if vmin == vmax:
                vmax = vmin + 1e-6

            cmap = plt.get_cmap(cmap_name)
            patches = []
            colors = []
            for gi, verts in polygons.items():
                patches.append(MplPolygon(verts, closed=True))
                val = sf[gi][metric]
                colors.append(cmap((val - vmin) / (vmax - vmin)))

            pc = PatchCollection(patches, facecolors=colors,
                                 edgecolors='#333333', linewidths=0.3)
            ax.add_collection(pc)

            if row == 0:
                ax.set_title(label, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"t = {t:.1f} h", fontsize=9)

            # Colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap,
                                       norm=plt.Normalize(vmin, vmax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Voronoi Shape Factors", fontsize=13, y=1.02)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'shape_factors.png')
    return fig


# ======================================================================
# Shape factor distributions
# ======================================================================

def plot_shape_distributions(snaps, hist, p, indices=None, outdir=None):
    """Histograms of shape factors at selected timepoints."""
    indices = select_indices(len(snaps), indices)
    metrics = ['circularity', 'elongation', 'area']
    labels = ['Circularity', 'Elongation', 'Area (um^2)']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    cmap = plt.get_cmap('viridis')
    n_t = len(indices)

    _3d = is_3d(snaps[0], p) if snaps else False

    for col, (metric, label) in enumerate(zip(metrics, labels)):
        ax = axes[col]
        for ti, si in enumerate(indices):
            snap = snaps[si]
            t = get_snap_time(hist, si)
            work_snap = slice_snap_z_midplane(snap, p) if _3d else snap
            polygons, _ = voronoi_from_centers(work_snap, p)
            sf = compute_shape_factors(polygons)

            vals = [sf[k][metric] for k in sf]
            if not vals:
                continue

            color = cmap(ti / max(n_t - 1, 1))
            ax.hist(vals, bins=15, alpha=0.5, color=color,
                    label=f"t={t:.0f}h", density=True, histtype='stepfilled')

        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Shape Factor Distributions Over Time", fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'shape_distributions.png')
    return fig


# ======================================================================
# Shape factor timeseries
# ======================================================================

def plot_shape_timeseries(snaps, hist, p, outdir=None):
    """Mean +/- std of each shape factor vs time."""
    metrics = ['circularity', 'elongation', 'area']
    labels = ['Circularity', 'Elongation', 'Area (um^2)']
    colors = ['#2E7D32', '#1565C0', '#C62828']

    times = []
    stats = {m: {'mean': [], 'std': []} for m in metrics}

    _3d = is_3d(snaps[0], p) if snaps else False

    for si in range(len(snaps)):
        t = get_snap_time(hist, si)
        times.append(t)
        work_snap = slice_snap_z_midplane(snaps[si], p) if _3d else snaps[si]
        polygons, _ = voronoi_from_centers(work_snap, p)
        sf = compute_shape_factors(polygons)
        for m in metrics:
            vals = [sf[k][m] for k in sf]
            if vals:
                stats[m]['mean'].append(np.mean(vals))
                stats[m]['std'].append(np.std(vals))
            else:
                stats[m]['mean'].append(0)
                stats[m]['std'].append(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for col, (m, label, c) in enumerate(zip(metrics, labels, colors)):
        ax = axes[col]
        mu = np.array(stats[m]['mean'])
        sigma = np.array(stats[m]['std'])
        t = np.array(times)

        ax.plot(t, mu, '-', color=c, lw=2)
        ax.fill_between(t, mu - sigma, mu + sigma, color=c, alpha=0.2)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Voronoi Shape Factor Evolution", fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'shape_timeseries.png')
    return fig


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate all Voronoi shape factor visualizations."""
    print("  [2/6] Voronoi shape factors...")
    fig1 = plot_voronoi_overlay(snaps, hist, p, outdir=outdir)
    plt.close(fig1)

    fig2 = plot_shape_factor_maps(snaps, hist, p, outdir=outdir)
    plt.close(fig2)

    fig3 = plot_shape_distributions(snaps, hist, p, outdir=outdir)
    plt.close(fig3)

    fig4 = plot_shape_timeseries(snaps, hist, p, outdir=outdir)
    plt.close(fig4)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Voronoi shape factor viz')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
