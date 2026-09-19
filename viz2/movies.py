"""
GIF Movie Generation (Module 4)
=================================
Animated GIFs of all temporal evolutions:
  - Scaffold map
  - Voronoi tessellation
  - Local phase fractions
  - Stress maps
  - Energy mode maps

Frames are independent, so they are rendered in a process pool
(`viz2.parallel.pmap`); pass `workers=1` for the serial path.

Usage:
    python viz2/movies.py -i results/default
    python viz2/movies.py -i results/default --workers 16
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from viz2.common import (
    select_indices, get_snap_time, setup_scaffold_axes,
    draw_granule_patches, draw_cells_on_ax, ensure_phase_fields,
    render_frame_to_array, save_fig, ENERGY_COLORS,
    is_3d, slice_snap_z_midplane, slice_field_z_midplane,
)
from viz2.parallel import pmap


def _frame_indices(n_snaps, max_frames=40):
    """Select up to max_frames evenly-spaced snapshot indices."""
    if n_snaps <= max_frames:
        return list(range(n_snaps))
    return [int(round(i * (n_snaps - 1) / (max_frames - 1)))
            for i in range(max_frames)]


def _save_gif(frames, path, fps=4):
    """Save list of RGB arrays as GIF."""
    frames = [f for f in frames if f is not None]
    if not frames:
        print(f"    Warning: no frames rendered, skipping {Path(path).name}")
        return False
    try:
        import imageio
        duration = 1.0 / fps
        imageio.mimsave(path, frames, duration=duration, loop=0)
        print(f"    Saved GIF: {path}")
        return True
    except ImportError:
        print("    Warning: imageio not available, skipping GIF")
        return False


# ======================================================================
# Scaffold map GIF
# ======================================================================

def _frame_scaffold(args):
    """One scaffold-map frame (module level so a pool worker can import it)."""
    snap, p, t = args
    from viz2.scaffold_map import plot_scaffold_map_single

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    plot_scaffold_map_single(snap, p, t=t, ax=ax)
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    arr = render_frame_to_array(fig)
    plt.close(fig)
    return arr


def gif_scaffold_map(snaps, hist, p, outdir, fps=4, max_frames=40, workers=None):
    """Scaffold map evolution over time."""
    indices = _frame_indices(len(snaps), max_frames)
    frames = pmap(_frame_scaffold,
                  [(snaps[si], p, get_snap_time(hist, si)) for si in indices],
                  workers, label='scaffold frames')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'scaffold_evolution.gif'), fps)


# ======================================================================
# Voronoi GIF
# ======================================================================

def _frame_voronoi(args):
    """One Voronoi-overlay frame."""
    snap, p, t, _3d = args
    from viz2.voronoi import voronoi_from_centers

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    draw_snap = slice_snap_z_midplane(snap, p) if _3d else snap

    setup_scaffold_axes(ax, p, dark=True)
    draw_granule_patches(ax, draw_snap, p)
    draw_cells_on_ax(ax, draw_snap, p)

    polygons, _ = voronoi_from_centers(draw_snap, p)
    for gi, verts in polygons.items():
        closed = np.vstack([verts, verts[:1]])
        ax.plot(closed[:, 0], closed[:, 1], '-', color='white',
                lw=0.8, alpha=0.6)

    ax.set_title(f"Voronoi | t = {t:.1f} h", fontsize=10, color='white')
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    arr = render_frame_to_array(fig)
    plt.close(fig)
    return arr


def gif_voronoi(snaps, hist, p, outdir, fps=4, max_frames=40, workers=None):
    """Voronoi tessellation evolution."""
    indices = _frame_indices(len(snaps), max_frames)
    _3d = is_3d(snaps[0], p) if snaps else False
    frames = pmap(_frame_voronoi,
                  [(snaps[si], p, get_snap_time(hist, si), _3d) for si in indices],
                  workers, label='voronoi frames')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'voronoi_evolution.gif'), fps)


# ======================================================================
# Local phase fraction GIF
# ======================================================================

def _frame_local_phases(args):
    """One local-phase-fraction frame."""
    snap, p, t, _3d, field, cmap_name = args
    from viz2.voronoi import voronoi_from_centers, compute_local_phase_fractions
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    setup_scaffold_axes(ax, p, dark=False)
    ax.set_facecolor('#F5F5F5')

    phi_f_arr, phi_i_arr, phi_v_arr = ensure_phase_fields(snap, p)

    # 3D: slice fields and snap to z-midplane
    work_snap = snap
    if _3d:
        phi_f_arr = slice_field_z_midplane(phi_f_arr)
        phi_i_arr = slice_field_z_midplane(phi_i_arr)
        phi_v_arr = slice_field_z_midplane(phi_v_arr)
        work_snap = slice_snap_z_midplane(snap, p)

    polygons, _ = voronoi_from_centers(work_snap, p)
    local_pf = compute_local_phase_fractions(work_snap, p, polygons,
                                             phi_f=phi_f_arr,
                                             phi_i=phi_i_arr,
                                             phi_v=phi_v_arr)

    values = [local_pf[k][field] for k in polygons if k in local_pf]
    vmax = max(max(values), 0.01) if values else 1.0

    cmap = plt.get_cmap(cmap_name)
    patches = []
    colors = []
    for gi, verts in polygons.items():
        if gi not in local_pf:
            continue
        patches.append(MplPolygon(verts, closed=True))
        val = local_pf[gi][field]
        colors.append(cmap(val / vmax))

    if patches:
        pc = PatchCollection(patches, facecolors=colors,
                             edgecolors='#333333', linewidths=0.3)
        ax.add_collection(pc)

    ax.set_title(f"Local {field} | t = {t:.1f} h", fontsize=10)
    plt.tight_layout()
    arr = render_frame_to_array(fig)
    plt.close(fig)
    return arr


def gif_local_phases(snaps, hist, p, outdir, fps=4, max_frames=40,
                     field='phi_f', workers=None):
    """Local phase fraction heatmap evolution."""
    indices = _frame_indices(len(snaps), max_frames)
    cmap_name = {'phi_f': 'Reds', 'phi_i': 'Greens', 'phi_v': 'Greys'}[field]
    _3d = is_3d(snaps[0], p) if snaps else False
    frames = pmap(_frame_local_phases,
                  [(snaps[si], p, get_snap_time(hist, si), _3d, field, cmap_name)
                   for si in indices],
                  workers, label=f'local {field} frames')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / f'local_{field}_evolution.gif'), fps)


# ======================================================================
# Stress GIF
# ======================================================================

def _frame_stress(args):
    """One von-Mises stress frame."""
    snap, p, t = args

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    try:
        from analysis.coarse_grain import coarse_grain_field
        result = coarse_grain_field(snap, p, quantity='stress', Ngrid=20)
        extent = [0, p.Lx, 0, p.Ly]
        im = ax.imshow(result['von_mises'].T, origin='lower',
                       extent=extent, cmap='hot', aspect='equal')
        ax.set_title(f"von Mises | t = {t:.1f} h", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    except Exception:
        ax.text(0.5, 0.5, 'No contact data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12)
        ax.set_title(f"t = {t:.1f} h", fontsize=10)

    ax.set_xlim(0, p.Lx)
    ax.set_ylim(0, p.Ly)
    ax.set_aspect('equal')
    plt.tight_layout()
    arr = render_frame_to_array(fig)
    plt.close(fig)
    return arr


def gif_stress(snaps, hist, p, outdir, fps=4, max_frames=40, workers=None):
    """Von Mises stress spatial map evolution."""
    indices = _frame_indices(len(snaps), max_frames)
    frames = pmap(_frame_stress,
                  [(snaps[si], p, get_snap_time(hist, si)) for si in indices],
                  workers, label='stress frames')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'stress_evolution.gif'), fps)


# ======================================================================
# Energy modes GIF
# ======================================================================

_ENERGY_MODES = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
                 'interfacial']
_ENERGY_LABELS = ['Traction', 'Contact', 'Friction',
                  'Osmotic', 'Frustration', 'Interfacial']
_ENERGY_CMAPS = ['Greens', 'Blues', 'Reds', 'Purples', 'YlOrBr', 'Oranges']


def _frame_energy_modes(args):
    """One six-panel energy-mode frame."""
    snap, p, t = args
    from viz2.energy_stress import compute_energy_field

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    extent = [0, p.Lx, 0, p.Ly]

    for idx, (mode, label, cmap_name) in enumerate(
            zip(_ENERGY_MODES, _ENERGY_LABELS, _ENERGY_CMAPS)):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        field, gx, gy = compute_energy_field(snap, p, mode, Ngrid=25)
        vmax = np.percentile(field, 99) if field.max() > 0 else 1.0

        ax.imshow(field.T, origin='lower', extent=extent,
                  cmap=cmap_name, vmin=0, vmax=max(vmax, 1e-10),
                  aspect='equal')
        ax.set_title(label, fontsize=9,
                     color=ENERGY_COLORS.get(mode, 'k'))
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(f"Energy Modes | t = {t:.1f} h", fontsize=12)
    plt.tight_layout()
    arr = render_frame_to_array(fig)
    plt.close(fig)
    return arr


def gif_energy_modes(snaps, hist, p, outdir, fps=4, max_frames=40, workers=None):
    """6-panel energy mode map evolution."""
    indices = _frame_indices(len(snaps), max_frames)
    frames = pmap(_frame_energy_modes,
                  [(snaps[si], p, get_snap_time(hist, si)) for si in indices],
                  workers, label='energy-mode frames')

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'energy_modes_evolution.gif'), fps)


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None, workers=None):
    """Generate all GIF animations."""
    print("  [4/6] Movies (GIFs)...")
    if outdir is None:
        return

    gif_scaffold_map(snaps, hist, p, outdir, workers=workers)
    gif_voronoi(snaps, hist, p, outdir, workers=workers)
    gif_local_phases(snaps, hist, p, outdir, field='phi_f', workers=workers)
    gif_stress(snaps, hist, p, outdir, workers=workers)
    gif_energy_modes(snaps, hist, p, outdir, workers=workers)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIF animation generation')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=40)
    parser.add_argument('--workers', type=int, default=None,
                        help='Frame-rendering processes (default: auto, 1 = serial)')
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir, workers=args.workers)
