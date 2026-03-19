"""
Scaffold Map Visualization (Module 1)
======================================
Vector-drawn scaffold map: red=functional, green=inert, black=void.
Cell bridges drawn in crimson but count as functional space in data analysis.

This is the foundation visual that all other viz2 modules build on.

Usage:
    python viz2/scaffold_map.py -i results/default
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from viz2.common import (
    FUNC_COLOR, INERT_COLOR, VOID_COLOR, CELL_COLORS, CELL_LABELS,
    CellState, select_indices, get_snap_time, setup_scaffold_axes,
    draw_granule_patches, draw_cells_on_ax, make_scaffold_legend,
    save_fig, is_3d, slice_snap_z_midplane,
)


# ======================================================================
# Single-frame drawing (used by movies.py and composites)
# ======================================================================

def plot_scaffold_map_single(snap, p, t=0.0, ax=None, show_cells=True,
                             show_title=True):
    """Draw scaffold map for a single snapshot on an existing or new axes.

    Parameters
    ----------
    snap : dict
        Snapshot data.
    p : Params
        Simulation parameters.
    t : float
        Time label (hours).
    ax : matplotlib Axes or None
        If None, creates a new figure and axes.
    show_cells : bool
        Whether to overlay cell morphology.
    show_title : bool
        Whether to add time title.

    Returns
    -------
    ax : matplotlib Axes
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        created_fig = True

    # 3D data: slice to z-midplane for 2D rendering
    draw_snap = snap
    z_label = ""
    if is_3d(snap, p):
        draw_snap = slice_snap_z_midplane(snap, p)
        z_mid = draw_snap.get('_z_slice', 0)
        z_label = f"  (z = {z_mid:.0f} \u00b5m)"

    setup_scaffold_axes(ax, p, dark=True)
    draw_granule_patches(ax, draw_snap, p)

    if show_cells:
        draw_cells_on_ax(ax, draw_snap, p)

    if show_title:
        ax.set_title(f"t = {t:.1f} h{z_label}", fontsize=10, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    return ax


# ======================================================================
# Multi-panel scaffold map
# ======================================================================

def plot_scaffold_map(snaps, hist, p, indices=None, outdir=None):
    """Vector-drawn scaffold map at selected timepoints.

    Black background = void, red = functional, green = inert.
    Cell bridges in crimson. Cell states colored by CellState enum.

    Parameters
    ----------
    snaps : list of dict
        Snapshot data at each timepoint.
    hist : list of dict
        History metrics at each timepoint.
    p : Params
        Simulation parameters.
    indices : list of int or None
        Snapshot indices to plot. None = auto-select 5.
    outdir : str or None
        Output directory. If None, figure is not saved.

    Returns
    -------
    fig : matplotlib Figure
    """
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        t = get_snap_time(hist, si)
        plot_scaffold_map_single(snaps[si], p, t=t, ax=axes[c])

    # Collect seen cell states for legend
    seen_states = set()
    for si in indices:
        cs = snaps[si].get('cell_state', np.array([], dtype=int))
        seen_states.update(int(s) for s in cs)

    legend_handles = make_scaffold_legend(seen_states)
    axes[0].legend(handles=legend_handles, fontsize=7, loc='upper right',
                   framealpha=0.7, facecolor='#333333', labelcolor='white')

    fig.patch.set_facecolor('black')
    fig.suptitle("Scaffold Map", fontsize=12, y=1.02, color='white')
    plt.tight_layout()

    if outdir:
        save_fig(fig, outdir, 'scaffold_map.png')

    return fig


# ======================================================================
# run_all API
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate scaffold map visualization."""
    print("  [1/6] Scaffold map...")
    fig = plot_scaffold_map(snaps, hist, p, outdir=outdir)
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Scaffold map visualization')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, meta = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
