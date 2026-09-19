"""
Shared Utilities for V2 Visualization
======================================
Centralises color schemes, snapshot helpers, granule/cell drawing, and data loading
used across all viz2 modules.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection, EllipseCollection
from matplotlib.lines import Line2D
from pathlib import Path

from gels.engine import (
    superellipse_polygon_pts, superellipse_point, CellState,
    load_run,
    superellipsoid_point, quat_rotate, quat_to_rotation_matrix,
)

# ======================================================================
# Colours, species tables and matplotlib-free snapshot helpers (V3.0)
# ======================================================================
# Moved to viz2/palette.py and viz2/snapshot_ops.py so the live viewer can
# use them without importing this module (which pins the Agg backend).
# Re-exported here so every existing `from viz2.common import ...` still works.
from viz2.palette import (  # noqa: F401
    FUNC_COLOR, INERT_COLOR, VOID_COLOR, CELL_COLORS, CELL_LABELS, ENERGY_COLORS,
    LOCKED_BRIDGE_COLOR, species_table, species_ids, species_colors, granule_rgba,
    cell_rgba, hex_to_rgba, legend_handles_species,
)
from viz2.snapshot_ops import (  # noqa: F401
    is_3d, slice_field_z_midplane, slice_snap_z_midplane, cell_world_positions,
    cell_world_positions_3d, ensure_phase_fields, _periodic_offsets, _min_image,
)


# ======================================================================
# Data loading
# ======================================================================

def load_data(run_dir):
    """Load simulation data from disk.

    Returns (hist, snaps, p, metadata).
    """
    return load_run(run_dir)


# ======================================================================
# Snapshot helpers
# ======================================================================

def select_indices(n, indices=None, n_panels=5):
    """Pick evenly-spaced snapshot indices for multi-panel figures."""
    if indices is not None:
        return [i for i in indices if 0 <= i < n]
    if n <= n_panels:
        return list(range(n))
    return sorted(set([
        0,
        *[int(round(i * (n - 1) / (n_panels - 1))) for i in range(1, n_panels - 1)],
        n - 1,
    ]))


def get_snap_time(hist, si):
    """Get time for snapshot index si from history."""
    if si < len(hist):
        return hist[si].get('time', 0.0)
    return 0.0


# ======================================================================
# Drawing helpers
# ======================================================================

def setup_scaffold_axes(ax, p, dark=True):
    """Configure axes for scaffold map: limits, aspect, background."""
    if dark:
        ax.set_facecolor(VOID_COLOR)
    ax.set_xlim(0, p.Lx)
    ax.set_ylim(0, p.Ly)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])


def draw_granule_patches(ax, snap, p, colors=None, alpha=0.95,
                         edgecolors='none', linewidths=0):
    """Draw superellipse/circle patches for all granules.

    When ``p.boundary_mode == 'periodic'``, ghost images are drawn for
    granules whose bounding extent crosses the domain boundary.

    Parameters
    ----------
    colors : list or None
        Per-granule face colors. If None, granules are coloured by species
        (viz2.palette.species_colors; legacy runs: red / green by gtype).

    Returns
    -------
    PatchCollection added to the axes.
    """
    xs, ys, rs, gt = snap['x'], snap['y'], snap['r'], snap['gtype']
    a_arr, b_arr = snap.get('a', rs), snap.get('b', rs)
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
    theta_arr = snap.get('theta', np.zeros(len(xs)))
    N = len(xs)
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'
    Lx, Ly = float(p.Lx), float(p.Ly)
    if colors is None:
        colors = species_colors(snap, p)

    patches = []
    fc_list = []

    for i in range(N):
        color_i = colors[i]
        is_circle = (a_arr[i] == b_arr[i] and n1_arr[i] == 2.0)
        r_bound = max(float(a_arr[i]), float(b_arr[i]))

        # Determine image offsets
        if periodic:
            x_offsets = _periodic_offsets(xs[i], r_bound, Lx)
            y_offsets = _periodic_offsets(ys[i], r_bound, Ly)
        else:
            x_offsets = [0.0]
            y_offsets = [0.0]

        for dx in x_offsets:
            for dy in y_offsets:
                xi = xs[i] + dx
                yi = ys[i] + dy
                if is_circle:
                    patches.append(Circle((xi, yi), rs[i]))
                else:
                    verts = superellipse_polygon_pts(
                        xi, yi, a_arr[i], b_arr[i], n1_arr[i], theta_arr[i])
                    patches.append(Polygon(verts, closed=True))
                fc_list.append(color_i)

    pc = PatchCollection(patches, facecolors=fc_list, edgecolors=edgecolors,
                         linewidths=linewidths, alpha=alpha)
    ax.add_collection(pc)
    return pc


def draw_cells_on_ax(ax, snap, p):
    """Overlay cell morphology on existing axes (V3.0: collections, not one patch per cell).

    Bridges are elongated crimson ellipses from the cell to the target surface,
    migrating cells small oriented ellipses, all others small circles coloured
    by state. With ``p.boundary_mode == 'periodic'`` bridges use the minimum
    image and cells near an edge get ghost images. Returns the collections.
    Visual output matches the V2.x per-patch version; drawing 10⁴ cells takes
    milliseconds instead of seconds.
    """
    cell_states = np.asarray(snap.get('cell_state', np.array([], dtype=int)), dtype=int)
    n_cells = len(cell_states)
    if n_cells == 0:
        return []

    xs, ys, rs = snap['x'], snap['y'], snap['r']
    N = len(xs)
    Lx, Ly = float(p.Lx), float(p.Ly)
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'
    cell_theta = np.asarray(snap.get('cell_theta_local', np.zeros(n_cells)), dtype=float)
    wx, wy = cell_world_positions(snap)
    if periodic:
        wx = wx % Lx
        wy = wy % Ly

    # ghost-image offsets per cell (r_ghost = 15 µm tolerance, as before)
    r_ghost = 15.0
    x_opts = [np.ones(n_cells, dtype=bool)]
    x_offs = [0.0]
    y_opts = [np.ones(n_cells, dtype=bool)]
    y_offs = [0.0]
    if periodic:
        x_opts += [wx - r_ghost < 0, wx + r_ghost > Lx]; x_offs += [Lx, -Lx]
        y_opts += [wy - r_ghost < 0, wy + r_ghost > Ly]; y_offs += [Ly, -Ly]

    rgba = cell_rgba(cell_states, alpha=1.0)
    is_bridge = cell_states == int(CellState.BRIDGING)
    is_migr = cell_states == int(CellState.MIGRATING)
    is_prolif = cell_states == int(CellState.PROLIFERATING)

    # bridge geometry (vectorised, minimum image)
    bt = np.asarray(snap.get('cell_bridge_target', np.full(n_cells, -1)), dtype=int)
    valid_bt = is_bridge & (bt >= 0) & (bt < N)
    bdx = np.zeros(n_cells); bdy = np.zeros(n_cells)
    if np.any(valid_bt):
        tb = bt[valid_bt]
        dx_ct = np.asarray(xs)[tb] - wx[valid_bt]
        dy_ct = np.asarray(ys)[tb] - wy[valid_bt]
        if periodic:
            dx_ct = _min_image(dx_ct, Lx)
            dy_ct = _min_image(dy_ct, Ly)
        dist = np.sqrt(dx_ct**2 + dy_ct**2)
        safe = np.where(dist > 1e-6, dist, 1.0)
        tx = wx[valid_bt] + dx_ct - np.asarray(rs)[tb] * dx_ct / safe
        ty = wy[valid_bt] + dy_ct - np.asarray(rs)[tb] * dy_ct / safe
        tx = np.where(dist > 1e-6, tx, wx[valid_bt]); ty = np.where(dist > 1e-6, ty, wy[valid_bt])
        bdx[valid_bt] = tx - wx[valid_bt]
        bdy[valid_bt] = ty - wy[valid_bt]
    blen = np.sqrt(bdx**2 + bdy**2)
    bridge_ok = is_bridge & (blen > 1e-6)

    circles = ~(bridge_ok | is_migr)              # includes zero-length bridges
    r_cell = np.where(is_prolif, 6.0, 5.0)

    out = []
    for mx, dx in zip(x_opts, x_offs):
        for my, dy in zip(y_opts, y_offs):
            img = mx & my
            # circles
            sel = np.where(img & circles)[0]
            if sel.size:
                col = EllipseCollection(2 * r_cell[sel], 2 * r_cell[sel], np.zeros(sel.size), units='xy',
                                        offsets=np.column_stack([wx[sel] + dx, wy[sel] + dy]),
                                        offset_transform=ax.transData,
                                        facecolors=rgba[sel], edgecolors='white', linewidths=0.2, alpha=0.8)
                ax.add_collection(col); out.append(col)
            # migrating: oriented 14 × 8 ellipses
            sel = np.where(img & is_migr)[0]
            if sel.size:
                col = EllipseCollection(np.full(sel.size, 14.0), np.full(sel.size, 8.0),
                                        np.degrees(cell_theta[sel]), units='xy',
                                        offsets=np.column_stack([wx[sel] + dx, wy[sel] + dy]),
                                        offset_transform=ax.transData,
                                        facecolors=rgba[sel], edgecolors='white', linewidths=0.2, alpha=0.8)
                ax.add_collection(col); out.append(col)
            # bridges: volume-conserving elongated ellipses cell → target surface
            sel = np.where(img & bridge_ok)[0]
            if sel.size:
                a_long = np.maximum(blen[sel] / 2.0, 5.0)
                a_perp = np.maximum(100.0 / a_long, 2.0)
                ang = np.degrees(np.arctan2(bdy[sel], bdx[sel]))
                col = EllipseCollection(2 * a_long, 2 * a_perp, ang, units='xy',
                                        offsets=np.column_stack([wx[sel] + dx + bdx[sel] / 2.0,
                                                                 wy[sel] + dy + bdy[sel] / 2.0]),
                                        offset_transform=ax.transData,
                                        facecolors=rgba[sel], edgecolors='white', linewidths=0.3, alpha=0.85)
                ax.add_collection(col); out.append(col)
    return out


def make_scaffold_legend(seen_states=None, species=None, p=None):
    """Legend handles: one square per granule species (V3.0), void, and cell states.

    ``species`` is a species table (see viz2.palette.species_table); when
    omitted it is taken from ``p`` (legacy runs: Functional / Inert).
    """
    if species is None:
        species = species_table(p)
    handles = legend_handles_species(species)
    handles.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor=VOID_COLOR,
               markeredgecolor='white', markersize=10, linestyle='None', label='Void'))
    if seen_states:
        for s_val in sorted(seen_states):
            if s_val in CELL_COLORS:
                handles.append(
                    Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=CELL_COLORS[s_val], markersize=7,
                           linestyle='None', label=CELL_LABELS.get(s_val, f'State {s_val}')))
    return handles


# ======================================================================
# Figure utilities
# ======================================================================

def save_fig(fig, outdir, filename, dpi=200):
    """Save figure to outdir/filename, creating directory if needed."""
    if outdir is None:
        return None
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"    Saved: {path}")
    return path


def render_frame_to_array(fig):
    """Render a matplotlib figure to an RGB numpy array for GIF assembly."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()
