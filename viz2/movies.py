"""
GIF Movie Generation (Module 4)
=================================
Animated GIFs of all temporal evolutions:
  - Scaffold map
  - Voronoi tessellation
  - Local phase fractions
  - Stress maps
  - Energy mode maps

Usage:
    python viz2/movies.py -i results/default
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


def _frame_indices(n_snaps, max_frames=40):
    """Select up to max_frames evenly-spaced snapshot indices."""
    if n_snaps <= max_frames:
        return list(range(n_snaps))
    return [int(round(i * (n_snaps - 1) / (max_frames - 1)))
            for i in range(max_frames)]


def _save_gif(frames, path, fps=4):
    """Save list of RGB arrays as GIF."""
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

def gif_scaffold_map(snaps, hist, p, outdir, fps=4, max_frames=40):
    """Scaffold map evolution over time."""
    from viz2.scaffold_map import plot_scaffold_map_single

    indices = _frame_indices(len(snaps), max_frames)
    frames = []

    for si in indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        t = get_snap_time(hist, si)
        plot_scaffold_map_single(snaps[si], p, t=t, ax=ax)
        fig.patch.set_facecolor('black')
        plt.tight_layout()
        frames.append(render_frame_to_array(fig))
        plt.close(fig)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'scaffold_evolution.gif'), fps)


# ======================================================================
# Voronoi GIF
# ======================================================================

def gif_voronoi(snaps, hist, p, outdir, fps=4, max_frames=40):
    """Voronoi tessellation evolution."""
    from viz2.voronoi import voronoi_from_centers

    indices = _frame_indices(len(snaps), max_frames)
    frames = []
    _3d = is_3d(snaps[0], p) if snaps else False

    for si in indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        snap = snaps[si]
        t = get_snap_time(hist, si)

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
        frames.append(render_frame_to_array(fig))
        plt.close(fig)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'voronoi_evolution.gif'), fps)


# ======================================================================
# Local phase fraction GIF
# ======================================================================

def gif_local_phases(snaps, hist, p, outdir, fps=4, max_frames=40,
                     field='phi_f'):
    """Local phase fraction heatmap evolution."""
    from viz2.voronoi import voronoi_from_centers, compute_local_phase_fractions
    from matplotlib.patches import Polygon as MplPolygon
    from matplotlib.collections import PatchCollection

    indices = _frame_indices(len(snaps), max_frames)
    frames = []
    cmap_name = {'phi_f': 'Reds', 'phi_i': 'Greens', 'phi_v': 'Greys'}[field]
    _3d = is_3d(snaps[0], p) if snaps else False

    for si in indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        snap = snaps[si]
        t = get_snap_time(hist, si)

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
        frames.append(render_frame_to_array(fig))
        plt.close(fig)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / f'local_{field}_evolution.gif'), fps)


# ======================================================================
# Stress GIF
# ======================================================================

def gif_stress(snaps, hist, p, outdir, fps=4, max_frames=40):
    """Von Mises stress spatial map evolution."""
    indices = _frame_indices(len(snaps), max_frames)
    frames = []
    _3d = is_3d(snaps[0], p) if snaps else False

    for si in indices:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        snap = snaps[si]
        t = get_snap_time(hist, si)

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
        frames.append(render_frame_to_array(fig))
        plt.close(fig)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'stress_evolution.gif'), fps)


# ======================================================================
# Energy modes GIF
# ======================================================================

def gif_energy_modes(snaps, hist, p, outdir, fps=4, max_frames=40):
    """6-panel energy mode map evolution."""
    from viz2.energy_stress import compute_energy_field

    indices = _frame_indices(len(snaps), max_frames)
    frames = []

    modes = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
             'interfacial']
    mode_labels = ['Traction', 'Contact', 'Friction',
                   'Osmotic', 'Frustration', 'Interfacial']
    cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'YlOrBr', 'Oranges']

    for si in indices:
        snap = snaps[si]
        t = get_snap_time(hist, si)

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        extent = [0, p.Lx, 0, p.Ly]

        for idx, (mode, label, cmap_name) in enumerate(
                zip(modes, mode_labels, cmaps)):
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
        frames.append(render_frame_to_array(fig))
        plt.close(fig)

    Path(outdir).mkdir(parents=True, exist_ok=True)
    _save_gif(frames, str(Path(outdir) / 'energy_modes_evolution.gif'), fps)


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate all GIF animations."""
    print("  [4/6] Movies (GIFs)...")
    if outdir is None:
        return

    gif_scaffold_map(snaps, hist, p, outdir)
    gif_voronoi(snaps, hist, p, outdir)
    gif_local_phases(snaps, hist, p, outdir, field='phi_f')
    gif_stress(snaps, hist, p, outdir)
    gif_energy_modes(snaps, hist, p, outdir)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GIF animation generation')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--max-frames', type=int, default=40)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
