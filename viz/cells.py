"""
Cell & Stress Visualization
============================
Visualizes individual cell states and mechanical stress on granules.

Plots produced:
  1. Stress map — granules colored by net force magnitude with direction arrows
  2. Cell states — fibroblast-like cell morphology colored by CellState
  3. Cell timelapse — animated GIF of cell activity over time

Usage:
    python viz/cells.py -i ./results/run1
    python viz/cells.py -i ./results/run1 -o ./plots

Or import programmatically:
    from viz.cells import run_all
    run_all(snaps, hist, p, outdir='plots/')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

from new_dem_0 import (
    superellipse_point, superellipse_polygon_pts, CellState,
    superellipsoid_point, quat_rotate
)

# ── Color scheme per CellState ──────────────────────────────────────
CELL_COLORS = {
    int(CellState.ATTACHED):     '#DAA520',  # goldenrod
    int(CellState.SPREADING):    '#FF8C00',  # dark orange
    int(CellState.PROLIFERATING):'#228B22',  # forest green
    int(CellState.BRIDGING):     '#DC143C',  # crimson
    int(CellState.SENESCENT):    '#696969',  # dim gray
    int(CellState.MIGRATING):    '#4169E1',  # royal blue
}

CELL_LABELS = {
    int(CellState.ATTACHED):     'Attached',
    int(CellState.SPREADING):    'Spreading',
    int(CellState.PROLIFERATING):'Proliferating',
    int(CellState.BRIDGING):     'Bridging',
    int(CellState.SENESCENT):    'Senescent',
    int(CellState.MIGRATING):    'Migrating',
}


# ── Helpers ─────────────────────────────────────────────────────────

def _is_3d(snap, p=None):
    """Check if snapshot is from a 3D simulation.

    Checks p.mode first, then falls back to detecting 'quat' key in snapshot
    (reliable for disk-loaded data where 'mode' isn't stored in NPZ).
    """
    if p is not None:
        mode = getattr(p, 'mode', '2D')
        return mode == '3D'
    # Fallback: quat presence is a reliable 3D indicator
    if 'quat' in snap:
        return True
    mode = snap.get('mode', '2D')
    if isinstance(mode, np.ndarray):
        mode = str(mode)
    return mode == '3D'


def _cell_world_positions(snap, p=None):
    """Compute world (x, y) for all cells. Handles 2D and 3D (XY projection).

    Returns arrays (wx, wy) of shape (n_cells,).
    """
    cell_gi = snap['cell_granule_id']
    n_cells = len(cell_gi)
    if n_cells == 0:
        return np.array([]), np.array([])

    xs, ys = snap['x'], snap['y']
    a_arr, b_arr = snap['a'], snap['b']
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))

    wx = np.zeros(n_cells)
    wy = np.zeros(n_cells)

    if _is_3d(snap, p):
        # 3D: use superellipsoid_point + quaternion rotation, project to XY
        c_arr = snap.get('c', a_arr.copy())
        n2_arr = snap.get('n2', n1_arr.copy())
        quats = snap.get('quat', None)
        cell_eta = snap.get('cell_eta_local', np.zeros(n_cells))
        cell_omega = snap.get('cell_omega_local', np.zeros(n_cells))

        for ci in range(n_cells):
            gi = int(cell_gi[ci])
            if gi < 0 or gi >= len(xs):
                continue
            # Body-frame surface point
            bp = superellipsoid_point(
                cell_eta[ci], cell_omega[ci],
                a_arr[gi], b_arr[gi], c_arr[gi], n1_arr[gi], n2_arr[gi])
            # Rotate to world frame
            if quats is not None:
                bp = quat_rotate(quats[gi], bp)
            wx[ci] = xs[gi] + bp[0]
            wy[ci] = ys[gi] + bp[1]
    else:
        # 2D: use superellipse_point + 2D rotation
        theta_arr = snap.get('theta', np.zeros(len(xs)))
        cell_theta = snap['cell_theta_local']

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


def _select_indices(snaps, indices=None):
    """Select timepoint indices for multi-panel plots."""
    n = len(snaps)
    if indices is not None:
        return indices
    return sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))


def _draw_granules(ax, snap, p, fill_color='#E8E8E8', edge_color='k', lw=0.3, alpha=0.6):
    """Draw granule outlines as superellipse polygons or circles.

    For 3D mode, draws circles at XY positions (top-down projection).
    """
    xs, ys, rs = snap['x'], snap['y'], snap['r']
    a_arr, b_arr = snap['a'], snap['b']
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
    is_3d = _is_3d(snap, p)

    patches = []
    if is_3d:
        # 3D: draw circles using equatorial radius (max of a, b)
        for i in range(len(xs)):
            r_proj = max(a_arr[i], b_arr[i])
            patches.append(Circle((xs[i], ys[i]), r_proj))
    else:
        # 2D: superellipse outlines
        theta_arr = snap.get('theta', np.zeros(len(xs)))
        for i in range(len(xs)):
            has_shape = not (a_arr[i] == b_arr[i] and n1_arr[i] == 2.0)
            if has_shape:
                verts = superellipse_polygon_pts(
                    xs[i], ys[i], a_arr[i], b_arr[i], n1_arr[i], theta_arr[i])
                patch = Polygon(verts, closed=True)
            else:
                patch = Circle((xs[i], ys[i]), rs[i])
            patches.append(patch)

    pc = PatchCollection(patches, facecolors=fill_color, edgecolors=edge_color,
                         linewidths=lw, alpha=alpha)
    ax.add_collection(pc)
    ax.set_xlim(0, p.Lx)
    ax.set_ylim(0, p.Ly)
    ax.set_aspect('equal')
    return patches


def _star_vertices(cx, cy, n_arms, r_outer, r_inner, angle_offset=0.0):
    """Generate vertices for a star/stellate polygon."""
    verts = []
    for k in range(n_arms):
        theta_out = angle_offset + 2 * np.pi * k / n_arms
        theta_in = angle_offset + 2 * np.pi * (k + 0.5) / n_arms
        verts.append((cx + r_outer * np.cos(theta_out),
                      cy + r_outer * np.sin(theta_out)))
        verts.append((cx + r_inner * np.cos(theta_in),
                      cy + r_inner * np.sin(theta_in)))
    return verts


def _nearest_surface_point(snap, gi, ext_x, ext_y, p=None):
    """Approximate nearest surface point on granule gi from (ext_x, ext_y).

    For 2D: finds surface point on superellipse closest to external point.
    For 3D: projects circle at equatorial radius toward external point.
    Uses minimum image convention for periodic boundaries so bridges don't
    span the entire domain.
    """
    cx, cy = float(snap['x'][gi]), float(snap['y'][gi])
    dx = ext_x - cx
    dy = ext_y - cy
    # Apply minimum image for periodic boundaries
    if p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic':
        Lx, Ly = float(p.Lx), float(p.Ly)
        if Lx > 0:
            dx -= Lx * round(dx / Lx)
        if Ly > 0:
            dy -= Ly * round(dy / Ly)
        # Use nearest image of granule centre for surface point computation
        cx = ext_x - dx
        cy = ext_y - dy
    dist = np.sqrt(dx**2 + dy**2)
    if dist < 1e-10:
        return cx + float(snap['a'][gi]), cy

    if _is_3d(snap, p):
        r_proj = max(float(snap['a'][gi]), float(snap['b'][gi]))
        return cx + r_proj * dx / dist, cy + r_proj * dy / dist

    # 2D: transform to body frame, find superellipse surface point
    a = float(snap['a'][gi])
    b = float(snap['b'][gi])
    n1 = float(snap.get('n1', snap.get('n_shape'))[gi])
    theta = float(snap.get('theta', np.zeros(len(snap['x'])))[gi])
    ct, st = np.cos(theta), np.sin(theta)
    dx_body = ct * dx + st * dy
    dy_body = -st * dx + ct * dy
    t_body = np.arctan2(dy_body, dx_body)
    bx, by = superellipse_point(t_body, a, b, n1)
    wx = cx + ct * bx - st * by
    wy = cy + st * bx + ct * by
    return wx, wy


def _draw_fibroblast(ax, wx, wy, state, angle=0.0, scale=1.0):
    """Draw a non-bridging cell as a fibroblast-like patch at (wx, wy)."""
    color = CELL_COLORS.get(state, '#AAAAAA')
    s = state

    if s == int(CellState.ATTACHED):
        from matplotlib.patches import Ellipse
        patch = Ellipse((wx, wy), width=18 * scale, height=14 * scale,
                        angle=np.degrees(angle),
                        fc=color, ec='k', lw=0.2, alpha=0.75)
        ax.add_patch(patch)

    elif s == int(CellState.SPREADING):
        verts = _star_vertices(wx, wy, 4, 12 * scale, 7 * scale,
                               angle_offset=angle)
        patch = Polygon(verts, closed=True, fc=color, ec='k', lw=0.2, alpha=0.7)
        ax.add_patch(patch)

    elif s == int(CellState.PROLIFERATING):
        verts = _star_vertices(wx, wy, 6, 15 * scale, 5 * scale,
                               angle_offset=angle)
        patch = Polygon(verts, closed=True, fc=color, ec='k', lw=0.2, alpha=0.8)
        ax.add_patch(patch)

    elif s == int(CellState.SENESCENT):
        r = 11 * scale
        patch = Circle((wx, wy), r, fc=color, ec='k', lw=0.2, alpha=0.45)
        ax.add_patch(patch)


def _draw_bridge_cell(ax, host_x, host_y, target_x, target_y,
                      contact_area=0.0):
    """Draw a bridging cell as a volume-conserving elongated ellipse.

    The cell is rendered as a smooth elliptical shape spanning from host
    surface to target surface, reflecting realistic fibroblast morphology
    during bridging (prolate elongation with volume conservation).
    """
    color = CELL_COLORS[int(CellState.BRIDGING)]
    dx = target_x - host_x
    dy = target_y - host_y
    L = np.sqrt(dx**2 + dy**2)
    if L < 1e-6:
        ax.add_patch(Circle((host_x, host_y), 5, fc=color, ec='k', lw=0.2, alpha=0.8))
        return

    # Volume-conserving ellipse: V_cell = pi * a_long * a_perp (2D cross-section)
    # Cell diameter 20 µm → 2D area ~ pi * 10^2 = 314 µm²
    A_cell = np.pi * 10.0**2
    a_long = L / 2.0  # semi-major = half bridge length
    a_long = max(a_long, 5.0)
    a_perp = A_cell / (np.pi * a_long)  # semi-minor from area conservation
    a_perp = max(a_perp, 2.0)  # minimum visibility

    # Center of ellipse
    cx = (host_x + target_x) / 2.0
    cy = (host_y + target_y) / 2.0
    angle = np.degrees(np.arctan2(dy, dx))

    from matplotlib.patches import Ellipse
    patch = Ellipse((cx, cy), width=2*a_long, height=2*a_perp,
                    angle=angle, fc=color, ec='k', lw=0.3, alpha=0.8)
    ax.add_patch(patch)


# ── Plot 1: Stress Map ─────────────────────────────────────────────

def plot_stress_map(snaps, hist, p, indices=None, outdir=None):
    """Granules colored by net force magnitude with direction arrows."""
    indices = _select_indices(snaps, indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        snap = snaps[si]
        ax = axes[c]

        xs, ys, rs = snap['x'], snap['y'], snap['r']
        a_arr, b_arr = snap['a'], snap['b']
        n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
        theta_arr = snap.get('theta', np.zeros(len(xs)))

        fx = snap.get('force_x', np.zeros(len(xs)))
        fy = snap.get('force_y', np.zeros(len(xs)))
        fz = snap.get('force_z', np.zeros(len(xs)))
        fmag = np.sqrt(fx**2 + fy**2 + fz**2)

        is_3d = _is_3d(snap, p)

        # Build patches
        patches = []
        if is_3d:
            for i in range(len(xs)):
                r_proj = max(a_arr[i], b_arr[i])
                patches.append(Circle((xs[i], ys[i]), r_proj))
        else:
            for i in range(len(xs)):
                has_shape = not (a_arr[i] == b_arr[i] and n1_arr[i] == 2.0)
                if has_shape:
                    verts = superellipse_polygon_pts(
                        xs[i], ys[i], a_arr[i], b_arr[i], n1_arr[i], theta_arr[i])
                    patch = Polygon(verts, closed=True)
                else:
                    patch = Circle((xs[i], ys[i]), rs[i])
                patches.append(patch)

        pc = PatchCollection(patches, cmap='YlOrRd', edgecolors='k', linewidths=0.3)
        pc.set_array(fmag)
        pc.set_clim(0, max(fmag.max(), 1e-6))
        ax.add_collection(pc)

        # Force direction arrows (quiver)
        # Normalize for visibility; skip near-zero forces
        mask = fmag > fmag.max() * 0.05
        if mask.any():
            arrow_scale = rs[mask].mean() * 0.8
            fn = fmag[mask]
            fn_norm = fn / fn.max()
            ax.quiver(xs[mask], ys[mask],
                      fx[mask] / fmag[mask], fy[mask] / fmag[mask],
                      scale=6, scale_units='width', width=0.003,
                      color='k', alpha=0.5, headwidth=3)

        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')
        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f't = {t:.1f} h', fontsize=10)
        if c == nc - 1:
            plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04,
                         label='Force magnitude (nN)')

    fig.suptitle('Stress Map: Force on Granules', fontsize=13, y=1.02)
    plt.tight_layout()

    if outdir:
        fig.savefig(str(Path(outdir) / 'stress_map.png'),
                    dpi=150, bbox_inches='tight')
    return fig


# ── Plot 2: Cell States ────────────────────────────────────────────

def plot_cell_states(snaps, hist, p, indices=None, outdir=None):
    """Cells rendered as fibroblast-like shapes colored by CellState."""
    indices = _select_indices(snaps, indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        snap = snaps[si]
        ax = axes[c]

        # Draw granule outlines
        _draw_granules(ax, snap, p)

        # Get cell data
        cell_states = snap.get('cell_state', np.array([], dtype=int))
        n_cells = len(cell_states)
        if n_cells == 0:
            t = hist[si]['time'] if si < len(hist) else 0
            ax.set_title(f't = {t:.1f} h', fontsize=10)
            continue

        wx, wy = _cell_world_positions(snap, p)

        # Draw each cell
        for ci in range(n_cells):
            state = int(cell_states[ci])
            if state == int(CellState.BRIDGING):
                bt = int(snap['cell_bridge_target'][ci])
                if 0 <= bt < len(snap['x']):
                    # Nearest surface point on target granule from cell position
                    tx, ty = _nearest_surface_point(snap, bt, wx[ci], wy[ci], p)
                    ca = snap.get('cell_contact_area', np.zeros(n_cells))
                    _draw_bridge_cell(ax, wx[ci], wy[ci], tx, ty,
                                      contact_area=float(ca[ci]))
            else:
                angle = snap['cell_theta_local'][ci] if ci < len(snap.get('cell_theta_local', [])) else 0.0
                _draw_fibroblast(ax, wx[ci], wy[ci], state, angle=angle)

        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f't = {t:.1f} h', fontsize=10)

    # Legend (first axes)
    legend_handles = []
    for s_val in sorted(CELL_COLORS.keys()):
        legend_handles.append(
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=CELL_COLORS[s_val], markersize=8,
                   label=CELL_LABELS[s_val]))
    axes[0].legend(handles=legend_handles, fontsize=7, loc='upper right',
                   framealpha=0.8)

    fig.suptitle('Cell States: Fibroblast Morphology', fontsize=13, y=1.02)
    plt.tight_layout()

    if outdir:
        fig.savefig(str(Path(outdir) / 'cell_states.png'),
                    dpi=150, bbox_inches='tight')
    return fig


# ── Plot 3: Cell Timelapse GIF ─────────────────────────────────────

def cell_timelapse_gif(snaps, hist, p, outdir=None):
    """Animated GIF of cell activity over time."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 6))

    def _draw_frame(si):
        ax.clear()
        snap = snaps[si]

        # Granule outlines
        _draw_granules(ax, snap, p)

        # Cells
        cell_states = snap.get('cell_state', np.array([], dtype=int))
        n_cells = len(cell_states)
        if n_cells > 0:
            wx, wy = _cell_world_positions(snap, p)
            for ci in range(n_cells):
                state = int(cell_states[ci])
                if state == int(CellState.BRIDGING):
                    bt = int(snap['cell_bridge_target'][ci])
                    if 0 <= bt < len(snap['x']):
                        tx, ty = _nearest_surface_point(snap, bt, wx[ci], wy[ci], p)
                        ca = snap.get('cell_contact_area', np.zeros(n_cells))
                        _draw_bridge_cell(ax, wx[ci], wy[ci], tx, ty,
                                          contact_area=float(ca[ci]))
                else:
                    angle = snap['cell_theta_local'][ci] if ci < len(snap.get('cell_theta_local', [])) else 0.0
                    _draw_fibroblast(ax, wx[ci], wy[ci], state, angle=angle)

        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f't = {t:.1f} h', fontsize=11)
        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

    anim = FuncAnimation(fig, _draw_frame, frames=len(snaps),
                         interval=500, repeat=True)

    gif_path = None
    if outdir:
        gif_path = str(Path(outdir) / 'cell_timelapse.gif')
        try:
            anim.save(gif_path, writer=PillowWriter(fps=2), dpi=100)
        except Exception:
            # Fallback: try imageio
            try:
                import imageio
                frames = []
                for si in range(len(snaps)):
                    _draw_frame(si)
                    fig.canvas.draw()
                    w, h = fig.canvas.get_width_height()
                    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                    frames.append(buf.reshape(h, w, 3))
                imageio.mimsave(gif_path, frames, duration=0.5, loop=0)
            except ImportError:
                print("  Warning: Could not save GIF (install Pillow or imageio)")
                gif_path = None

    plt.close(fig)
    return gif_path


# ── Public API ──────────────────────────────────────────────────────

def run_all(snaps, hist, p, outdir=None):
    """Generate all cell/stress visualizations."""
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    print("  viz_cells: stress map...")
    fig1 = plot_stress_map(snaps, hist, p, outdir=outdir)
    plt.close(fig1)

    print("  viz_cells: cell states...")
    fig2 = plot_cell_states(snaps, hist, p, outdir=outdir)
    plt.close(fig2)

    print("  viz_cells: cell timelapse GIF...")
    gif = cell_timelapse_gif(snaps, hist, p, outdir=outdir)
    if gif:
        print(f"  viz_cells: saved {gif}")

    # V1.5.2: Delegate 3D stress rendering to viz_stress if available
    is_3d = _is_3d(snaps[0], p) if snaps else False
    if is_3d:
        try:
            from viz import stress as viz_stress
            if viz_stress.HAS_PYVISTA:
                print("  viz_cells: delegating 3D stress rendering to viz_stress...")
                viz_stress.run_all(snaps, hist, p, outdir=outdir)
        except ImportError:
            print("  viz_cells: viz_stress not available, skipping 3D stress")


# ── CLI ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cell & stress visualization')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    try:
        from new_dem_0 import load_run
        hist, snaps, p, meta = load_run(str(indir))
        run_all(snaps, hist, p, outdir=outdir)
    except Exception as e:
        print(f"Could not load run data: {e}")
        print(f"Use programmatically: from viz.cells import run_all")
        print(f"  run_all(snaps, hist, p, outdir='{outdir}')")
