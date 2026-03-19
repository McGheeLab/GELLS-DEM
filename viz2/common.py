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
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from pathlib import Path

from new_dem_0 import (
    superellipse_polygon_pts, superellipse_point, CellState,
    load_run,
    superellipsoid_point, quat_rotate, quat_to_rotation_matrix,
)

# ======================================================================
# Color constants
# ======================================================================

FUNC_COLOR = '#CC2222'       # red for functional granules
INERT_COLOR = '#22AA22'      # green for inert granules
VOID_COLOR = '#000000'       # black background (void space)

CELL_COLORS = {
    int(CellState.ATTACHED):      '#DAA520',   # goldenrod
    int(CellState.SPREADING):     '#FF8C00',   # dark orange
    int(CellState.PROLIFERATING): '#228B22',   # forest green
    int(CellState.BRIDGING):      '#DC143C',   # crimson
    int(CellState.SENESCENT):     '#696969',   # dim gray
    int(CellState.MIGRATING):     '#4169E1',   # royal blue
}

CELL_LABELS = {
    int(CellState.ATTACHED):      'Attached',
    int(CellState.SPREADING):     'Spreading',
    int(CellState.PROLIFERATING): 'Proliferating',
    int(CellState.BRIDGING):      'Bridging',
    int(CellState.SENESCENT):     'Senescent',
    int(CellState.MIGRATING):     'Migrating',
}

# Energy mode color scheme
ENERGY_COLORS = {
    'traction':      '#2E7D32',   # green
    'contact':       '#1565C0',   # blue
    'friction':      '#C62828',   # red
    'osmotic':       '#6A1B9A',   # purple
    'frustration':   '#795548',   # brown
    'interfacial':   '#E65100',   # orange
}


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


def is_3d(snap, p):
    """Detect whether the data is from a 3D simulation."""
    mode = getattr(p, 'mode', '2D')
    if mode == '3D':
        return True
    if 'z' in snap and len(snap['z']) > 0 and np.any(snap['z'] != 0):
        return True
    return False


def slice_field_z_midplane(field_3d, z_frac=0.5):
    """Extract a 2D slice from a 3D phase field at z = z_frac * Nz.

    Parameters
    ----------
    field_3d : ndarray (Nx, Ny, Nz)
    z_frac : float
        Fractional z-position (0=bottom, 1=top).

    Returns
    -------
    ndarray (Nx, Ny)
    """
    if field_3d.ndim == 2:
        return field_3d
    Nz = field_3d.shape[2]
    iz = min(int(z_frac * Nz), Nz - 1)
    return field_3d[:, :, iz]


def slice_snap_z_midplane(snap, p, z_frac=0.5):
    """Create a 2D-compatible snapshot by slicing 3D data at z-midplane.

    Selects granules whose center is within one bounding radius of the
    slice plane. Projects to 2D: keeps (x, y), computes cross-section
    radii, approximates theta from quaternion yaw. Carries cells on
    selected granules.

    Parameters
    ----------
    snap : dict
        3D snapshot data.
    p : Params
    z_frac : float
        Fractional z-position for the slice.

    Returns
    -------
    snap_2d : dict
        2D-compatible snapshot.
    """
    Lz = float(getattr(p, 'Lz', 0))
    z_mid = Lz * z_frac
    zs = snap['z']
    N = len(zs)

    a_arr = snap.get('a', snap['r']).astype(np.float64)
    b_arr = snap.get('b', snap['r']).astype(np.float64)
    c_arr = snap.get('c', snap['r']).astype(np.float64)
    r_bound = np.maximum(np.maximum(a_arr, b_arr), c_arr)

    # Select granules intersecting the slice plane
    dz = np.abs(zs - z_mid)
    mask = dz < r_bound
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(N)  # fallback: use all

    # Cross-section semi-axes: for a superellipsoid, the z-slice at dz
    # gives an approximate ellipse with a_eff = a * (1 - (dz/c)^n2)^(1/n2)
    n2_arr = snap.get('n2', np.full(N, 2.0)).astype(np.float64)
    frac_z = np.clip(dz[idx] / np.maximum(c_arr[idx], 1e-6), 0.0, 0.999)
    # Generalized superellipsoid slice factor
    shrink = np.maximum(1.0 - frac_z ** n2_arr[idx], 0.0) ** (1.0 / n2_arr[idx])

    a_eff = a_arr[idx] * shrink
    b_eff = b_arr[idx] * shrink
    r_eff = snap['r'][idx] * shrink

    # Approximate theta from quaternion yaw (rotation about z-axis)
    if 'quat' in snap:
        quats = snap['quat'][idx]
        # Yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
        w, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        theta_eff = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    else:
        theta_eff = snap.get('theta', np.zeros(N))[idx]

    # Build index mapping from old granule index to new
    old_to_new = np.full(N, -1, dtype=int)
    old_to_new[idx] = np.arange(len(idx))

    # Build 2D snapshot
    snap_2d = {
        'x': snap['x'][idx].copy(),
        'y': snap['y'][idx].copy(),
        'r': r_eff,
        'a': a_eff,
        'b': b_eff,
        'n1': snap.get('n1', np.full(N, 2.0))[idx].copy(),
        'theta': theta_eff,
        'gtype': snap['gtype'][idx].copy(),
    }

    # Carry cell arrays for selected granules
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    if len(cell_gi) > 0:
        cell_mask = np.isin(cell_gi, idx)
        cell_keys = [k for k in snap if k.startswith('cell_') and k != 'cell_offset']
        for k in cell_keys:
            arr = snap[k]
            if len(arr) == len(cell_gi):
                snap_2d[k] = arr[cell_mask].copy()
        # Remap granule IDs to new indices
        if 'cell_granule_id' in snap_2d:
            snap_2d['cell_granule_id'] = old_to_new[snap_2d['cell_granule_id']]
        # Remap bridge targets
        if 'cell_bridge_target' in snap_2d:
            bt = snap_2d['cell_bridge_target'].copy()
            valid = (bt >= 0) & (bt < N)
            bt[valid] = old_to_new[bt[valid]]
            bt[bt < 0] = -1
            snap_2d['cell_bridge_target'] = bt

    # Carry contact arrays, filtering to granule pairs in the slice
    ci = snap.get('contact_i', np.array([], dtype=int))
    if len(ci) > 0:
        cj = snap.get('contact_j', np.array([], dtype=int))
        contact_mask = np.isin(ci, idx) & np.isin(cj, idx)
        contact_keys = [k for k in snap if k.startswith('contact_')]
        for k in contact_keys:
            arr = snap[k]
            if len(arr) == len(ci):
                snap_2d[k] = arr[contact_mask].copy()
        if 'contact_i' in snap_2d:
            snap_2d['contact_i'] = old_to_new[snap_2d['contact_i']]
            snap_2d['contact_j'] = old_to_new[snap_2d['contact_j']]

    # Store slice metadata
    snap_2d['_z_slice'] = z_mid
    snap_2d['_3d_idx'] = idx

    return snap_2d


def cell_world_positions_3d(snap):
    """Compute world (x, y, z) for all cells in a 3D snapshot.

    Uses superellipsoid parametric point + quaternion rotation.
    Returns (wx, wy, wz) arrays of shape (n_cells,).
    """
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    n_cells = len(cell_gi)
    if n_cells == 0:
        return np.array([]), np.array([]), np.array([])

    xs, ys, zs = snap['x'], snap['y'], snap['z']
    a_arr = snap.get('a', snap['r'])
    b_arr = snap.get('b', snap['r'])
    c_arr = snap.get('c', snap['r'])
    n1_arr = snap.get('n1', np.full(len(xs), 2.0))
    n2_arr = snap.get('n2', np.full(len(xs), 2.0))
    quats = snap.get('quat', np.tile([1, 0, 0, 0], (len(xs), 1)))
    cell_eta = snap.get('cell_eta_local', np.zeros(n_cells))
    cell_omega = snap.get('cell_omega_local', np.zeros(n_cells))

    wx = np.zeros(n_cells)
    wy = np.zeros(n_cells)
    wz = np.zeros(n_cells)

    for ci in range(n_cells):
        gi = int(cell_gi[ci])
        if gi < 0 or gi >= len(xs):
            continue
        # Body-frame surface point
        bp = superellipsoid_point(
            cell_eta[ci], cell_omega[ci],
            a_arr[gi], b_arr[gi], c_arr[gi], n1_arr[gi], n2_arr[gi])
        # Rotate to world frame
        wp = quat_rotate(quats[gi], bp)
        wx[ci] = xs[gi] + wp[0]
        wy[ci] = ys[gi] + wp[1]
        wz[ci] = zs[gi] + wp[2]

    return wx, wy, wz


def ensure_phase_fields(snap, p):
    """Ensure phi_f, phi_i, phi_v are available in the snapshot.

    If not present, re-renders from particle positions using render_fields()
    (2D) or render_fields_3d() (3D). Returns (phi_f, phi_i, phi_v).
    """
    if 'phi_f' in snap and snap['phi_f'] is not None:
        return snap['phi_f'], snap['phi_i'], snap['phi_v']

    from new_dem_0 import GranuleSystem, Params, render_fields

    pp = p if p is not None else Params()
    N = len(snap['x'])
    gs = GranuleSystem.__new__(GranuleSystem)
    gs.N = N
    gs.x = snap['x'].astype(np.float64)
    gs.y = snap['y'].astype(np.float64)
    gs.r = snap['r'].astype(np.float64)
    gs.gtype = snap['gtype'].astype(np.int32)
    gs.a = snap.get('a', snap['r']).astype(np.float64)
    gs.b = snap.get('b', snap['r']).astype(np.float64)
    gs.func_mask = gs.gtype == 0
    gs.contact_clips = [None] * N
    gs.epsilon = None

    _is_3d = is_3d(snap, p)

    if _is_3d:
        from new_dem_0 import render_fields_3d
        gs.z = snap['z'].astype(np.float64)
        gs.c = snap.get('c', snap['r']).astype(np.float64)
        gs.n1 = snap.get('n1', np.full(N, 2.0)).astype(np.float64)
        gs.n2 = snap.get('n2', np.full(N, 2.0)).astype(np.float64)
        gs.n_shape = gs.n1
        gs.quat = snap.get('quat', np.tile([1, 0, 0, 0], (N, 1))).astype(np.float64)
        gs.r_bound = np.maximum(np.maximum(gs.a, gs.b), gs.c)
        gs.is_circle = False
        gs.mode = "3D"
        gs.theta = np.zeros(N)
        phi_f, phi_i, phi_v = render_fields_3d(gs, pp)
    else:
        n1 = snap.get('n1', snap.get('n_shape', np.full(N, 2.0)))
        gs.n_shape = n1.astype(np.float64)
        gs.theta = snap.get('theta', np.zeros(N)).astype(np.float64)
        gs.r_bound = np.maximum(gs.a, gs.b)
        gs.is_circle = np.all(gs.a == gs.b) and np.all(n1 == 2.0)
        gs.mode = "2D"
        phi_f, phi_i, phi_v = render_fields(gs, pp)

    snap['phi_f'] = phi_f
    snap['phi_i'] = phi_i
    snap['phi_v'] = phi_v
    return phi_f, phi_i, phi_v


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


def _periodic_offsets(x, r_bound, L):
    """Return list of periodic offsets for a granule near a boundary.

    If the granule's bounding extent crosses x=0 or x=L, return the shift(s)
    needed to draw ghost images. Always includes 0 (primary image).
    """
    offsets = [0.0]
    if x - r_bound < 0:
        offsets.append(L)
    if x + r_bound > L:
        offsets.append(-L)
    return offsets


def draw_granule_patches(ax, snap, p, colors=None, alpha=0.95,
                         edgecolors='none', linewidths=0):
    """Draw superellipse/circle patches for all granules.

    When ``p.boundary_mode == 'periodic'``, ghost images are drawn for
    granules whose bounding extent crosses the domain boundary.

    Parameters
    ----------
    colors : list or None
        Per-granule face colors. If None, uses FUNC_COLOR/INERT_COLOR by gtype.

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

    patches = []
    fc_list = []

    for i in range(N):
        color_i = (colors[i] if colors is not None
                   else (FUNC_COLOR if gt[i] == 0 else INERT_COLOR))
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


def cell_world_positions(snap):
    """Compute world (x, y) for all cells in a 2D snapshot.

    Returns (wx, wy) arrays of shape (n_cells,).
    """
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    n_cells = len(cell_gi)
    if n_cells == 0:
        return np.array([]), np.array([])

    xs, ys = snap['x'], snap['y']
    a_arr = snap.get('a', snap['r'])
    b_arr = snap.get('b', snap['r'])
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
    theta_arr = snap.get('theta', np.zeros(len(xs)))
    cell_theta = snap.get('cell_theta_local', np.zeros(n_cells))

    wx = np.zeros(n_cells)
    wy = np.zeros(n_cells)

    for ci in range(n_cells):
        gi = int(cell_gi[ci])
        if gi < 0 or gi >= len(xs):
            continue
        bx, by = superellipse_point(cell_theta[ci], a_arr[gi], b_arr[gi], n1_arr[gi])
        ct = np.cos(theta_arr[gi])
        st = np.sin(theta_arr[gi])
        wx[ci] = xs[gi] + ct * bx - st * by
        wy[ci] = ys[gi] + st * bx + ct * by

    return wx, wy


def _min_image(dx, L):
    """Apply minimum image convention: shift dx to [-L/2, L/2)."""
    return dx - L * np.round(dx / L)


def draw_cells_on_ax(ax, snap, p):
    """Overlay cell morphology on existing axes.

    Bridges drawn as elongated crimson ellipses.
    Migrating cells as small oriented ellipses.
    Others as small circles colored by state.

    When ``p.boundary_mode == 'periodic'``, bridges use minimum-image
    convention and cells near edges are drawn as ghost images.
    """
    cell_states = snap.get('cell_state', np.array([], dtype=int))
    n_cells = len(cell_states)
    if n_cells == 0:
        return

    xs, ys, rs = snap['x'], snap['y'], snap['r']
    N = len(xs)
    Lx, Ly = float(p.Lx), float(p.Ly)
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'
    cell_theta = snap.get('cell_theta_local', np.zeros(n_cells))
    wx, wy = cell_world_positions(snap)

    # Wrap cell positions into [0, L) for periodic
    if periodic:
        wx = wx % Lx
        wy = wy % Ly

    # Tolerance for ghost images (cell radius ~ 5-10 µm)
    r_ghost = 15.0

    for ci in range(n_cells):
        state = int(cell_states[ci])
        cc = CELL_COLORS.get(state, '#AAAAAA')

        # Determine image positions for this cell
        if periodic:
            x_offsets = _periodic_offsets(wx[ci], r_ghost, Lx)
            y_offsets = _periodic_offsets(wy[ci], r_ghost, Ly)
        else:
            x_offsets = [0.0]
            y_offsets = [0.0]

        if state == int(CellState.BRIDGING):
            bt = int(snap['cell_bridge_target'][ci])
            if 0 <= bt < N:
                # Vector from cell to target center, with minimum image
                dx_ct = xs[bt] - wx[ci]
                dy_ct = ys[bt] - wy[ci]
                if periodic:
                    dx_ct = _min_image(dx_ct, Lx)
                    dy_ct = _min_image(dy_ct, Ly)

                # Target surface point: project from target center toward cell
                dist_ct = np.sqrt(dx_ct**2 + dy_ct**2)
                if dist_ct > 1e-6:
                    # Point on target surface closest to cell
                    tx = wx[ci] + dx_ct - rs[bt] * dx_ct / dist_ct
                    ty = wy[ci] + dy_ct - rs[bt] * dy_ct / dist_ct
                else:
                    tx, ty = wx[ci], wy[ci]

                # Bridge ellipse (volume-conserving)
                bdx = tx - wx[ci]
                bdy = ty - wy[ci]
                blen = np.sqrt(bdx**2 + bdy**2)
                if blen > 1e-6:
                    a_long = max(blen / 2.0, 5.0)
                    a_perp = max(np.pi * 100.0 / (np.pi * a_long), 2.0)
                    ang_b = np.degrees(np.arctan2(bdy, bdx))

                    for dx_off in x_offsets:
                        for dy_off in y_offsets:
                            cx_b = wx[ci] + dx_off + bdx / 2.0
                            cy_b = wy[ci] + dy_off + bdy / 2.0
                            patch = Ellipse((cx_b, cy_b), width=2 * a_long,
                                            height=2 * a_perp, angle=ang_b,
                                            fc=cc, ec='white', lw=0.3, alpha=0.85)
                            ax.add_patch(patch)
                else:
                    for dx_off in x_offsets:
                        for dy_off in y_offsets:
                            ax.add_patch(Circle((wx[ci] + dx_off, wy[ci] + dy_off), 5,
                                                fc=cc, ec='white', lw=0.2, alpha=0.8))

        elif state == int(CellState.MIGRATING):
            for dx_off in x_offsets:
                for dy_off in y_offsets:
                    ax.add_patch(Ellipse((wx[ci] + dx_off, wy[ci] + dy_off),
                                         width=14, height=8,
                                         angle=np.degrees(cell_theta[ci]),
                                         fc=cc, ec='white', lw=0.2, alpha=0.8))
        else:
            r_cell = 6 if state == int(CellState.PROLIFERATING) else 5
            for dx_off in x_offsets:
                for dy_off in y_offsets:
                    ax.add_patch(Circle((wx[ci] + dx_off, wy[ci] + dy_off), r_cell,
                                        fc=cc, ec='white', lw=0.2, alpha=0.8))


def make_scaffold_legend(seen_states=None):
    """Build legend handles for scaffold map + cell states.

    Returns list of Line2D / Patch handles.
    """
    handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=FUNC_COLOR,
               markersize=10, linestyle='None', label='Functional'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=INERT_COLOR,
               markersize=10, linestyle='None', label='Inert'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=VOID_COLOR,
               markeredgecolor='white', markersize=10, linestyle='None', label='Void'),
    ]
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
