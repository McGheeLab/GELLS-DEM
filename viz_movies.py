"""
3D Volumetric Compaction Movies
================================
Generates animated visualizations of granular scaffold compaction.

Animations produced:
  1. Rotating 3D isosurface at fixed time
  2. Time-lapse compaction movie (fixed camera)
  3. Cross-section sweep through z
  4. Composite 2x2: 3D isosurface + XY/XZ/YZ midplanes

Usage:
    python viz_movies.py -i ./simulations/run1
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse, json

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

try:
    from skimage.measure import marching_cubes
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAS_ANIM = True
except ImportError:
    HAS_ANIM = False

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Rendering helpers ──

def isosurface_from_field(phi, level=0.5, spacing=(1, 1, 1)):
    """Extract isosurface triangles from 3D field using marching cubes."""
    if not HAS_SKIMAGE:
        return None, None
    try:
        verts, faces, _, _ = marching_cubes(phi, level=level, spacing=spacing)
        return verts, faces
    except (ValueError, RuntimeError):
        return None, None


def plot_isosurface_matplotlib(ax, verts, faces, color='orangered', alpha=0.5):
    """Render isosurface on matplotlib 3D axis."""
    if verts is None or len(faces) == 0:
        return
    mesh = Poly3DCollection(verts[faces], alpha=alpha,
                            facecolor=color, edgecolor='none')
    ax.add_collection3d(mesh)


def plot_isosurface_pyvista(phi, level=0.5, spacing=(1, 1, 1),
                             color='orangered', opacity=0.5, plotter=None):
    """Render isosurface using PyVista."""
    if not HAS_PYVISTA:
        return None
    grid = pv.ImageData(dimensions=phi.shape, spacing=spacing)
    grid.point_data['values'] = phi.ravel(order='F')
    contour = grid.contour([level], scalars='values')
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(contour, color=color, opacity=opacity)
    return plotter


# ── Animation 1: Rotating 3D view ──

def rotating_gif(snap, p, outpath='rotating.gif', n_frames=36):
    """Create rotating 3D view GIF at a single time step."""
    phi_f = snap['phi_f'] if isinstance(snap, dict) else snap[0]
    phi_i = snap['phi_i'] if isinstance(snap, dict) else snap[1]

    if phi_f.ndim != 3:
        print("  Skipping rotating GIF: field is not 3D")
        return

    if HAS_PYVISTA:
        Ng = phi_f.shape[0]
        sp = (p.Lx/Ng, p.Ly/Ng, p.Lz/Ng) if hasattr(p, 'Lz') else (1,1,1)
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
        plot_isosurface_pyvista(phi_f, 0.3, sp, 'orangered', 0.6, plotter)
        plot_isosurface_pyvista(phi_i, 0.3, sp, 'steelblue', 0.4, plotter)
        plotter.camera_position = 'iso'
        plotter.open_gif(str(outpath))
        for angle in np.linspace(0, 360, n_frames, endpoint=False):
            plotter.camera.azimuth = angle
            plotter.write_frame()
        plotter.close()
        print(f"  Rotating GIF saved: {outpath}")
        return

    if not HAS_ANIM or not HAS_SKIMAGE:
        print("  Skipping rotating GIF: requires matplotlib animation + skimage")
        return

    Ng = phi_f.shape[0]
    sp = (p.Lx/Ng, p.Ly/Ng, p.Lz/Ng)
    vf, ff = isosurface_from_field(phi_f, 0.3, sp)
    vi, fi = isosurface_from_field(phi_i, 0.3, sp)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        ax.view_init(elev=25, azim=frame * 360 / n_frames)
        if vf is not None:
            plot_isosurface_matplotlib(ax, vf, ff, 'orangered', 0.5)
        if vi is not None:
            plot_isosurface_matplotlib(ax, vi, fi, 'steelblue', 0.3)
        ax.set_xlim(0, p.Lx); ax.set_ylim(0, p.Ly)
        if hasattr(p, 'Lz'):
            ax.set_zlim(0, p.Lz)
        ax.set_xlabel('X (µm)'); ax.set_ylabel('Y (µm)'); ax.set_zlabel('Z (µm)')
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"  Rotating GIF saved: {outpath}")


# ── Animation 2: Time-lapse compaction ──

def timelapse_gif(snaps, hist, p, outpath='timelapse.gif'):
    """Create time-lapse GIF of compaction evolution."""
    if not HAS_ANIM:
        print("  Skipping time-lapse: requires matplotlib animation")
        return

    # Select subset of frames
    n_frames = min(len(snaps), 30)
    indices = np.linspace(0, len(snaps)-1, n_frames, dtype=int)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ['Functional', 'Inert', 'Void']
    cmaps = ['Oranges', 'Blues', 'Greens']

    def update(frame_idx):
        si = indices[frame_idx]
        snap = snaps[si]
        phi_f = snap['phi_f'] if isinstance(snap, dict) else snap[0]
        phi_i = snap['phi_i'] if isinstance(snap, dict) else snap[1]
        phi_v = snap['phi_v'] if isinstance(snap, dict) else snap[2]
        fields = [phi_f, phi_i, phi_v]
        t = hist[si]['time']

        for c, (ax, fld, lbl, cm) in enumerate(zip(axes, fields, labels, cmaps)):
            ax.clear()
            if fld.ndim == 3:
                # Show midplane XY slice
                mid_z = fld.shape[2] // 2
                ax.imshow(fld[:, :, mid_z].T, origin='lower', cmap=cm,
                         vmin=0, vmax=max(0.01, fld.max()))
            else:
                ax.imshow(fld.T, origin='lower', cmap=cm,
                         vmin=0, vmax=max(0.01, fld.max()))
            ax.set_title(f'{lbl} (t={t:.1f}h)')
            ax.set_xticks([]); ax.set_yticks([])
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=4))
    plt.close(fig)
    print(f"  Time-lapse GIF saved: {outpath}")


# ── Animation 3: Cross-section sweep ──

def zsweep_gif(snap, p, outpath='zsweep.gif', n_slices=40):
    """Animate a cutting plane sweeping through z."""
    phi_f = snap['phi_f'] if isinstance(snap, dict) else snap[0]
    if phi_f.ndim != 3:
        print("  Skipping z-sweep: field is not 3D")
        return

    if not HAS_ANIM:
        return

    Ng = phi_f.shape[2]
    phi_i = snap['phi_i'] if isinstance(snap, dict) else snap[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    z_indices = np.linspace(0, Ng-1, n_slices, dtype=int)

    def update(frame):
        zi = z_indices[frame]
        z_um = zi * p.Lz / Ng
        ax1.clear(); ax2.clear()
        ax1.imshow(phi_f[:, :, zi].T, origin='lower', cmap='Oranges',
                   vmin=0, vmax=phi_f.max(), extent=[0, p.Lx, 0, p.Ly])
        ax1.set_title(f'Functional (z={z_um:.0f} µm)')
        ax2.imshow(phi_i[:, :, zi].T, origin='lower', cmap='Blues',
                   vmin=0, vmax=phi_i.max(), extent=[0, p.Lx, 0, p.Ly])
        ax2.set_title(f'Inert (z={z_um:.0f} µm)')
        return []

    anim = FuncAnimation(fig, update, frames=n_slices, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"  Z-sweep GIF saved: {outpath}")


# ── Animation 4: Composite 2x2 ──

def composite_gif(snaps, hist, p, outpath='composite.gif'):
    """Composite 2x2: 3D midplane XY + XZ + YZ + RGB composite."""
    if not HAS_ANIM:
        return

    n_frames = min(len(snaps), 20)
    indices = np.linspace(0, len(snaps)-1, n_frames, dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    def update(frame_idx):
        si = indices[frame_idx]
        snap = snaps[si]
        phi_f = snap['phi_f'] if isinstance(snap, dict) else snap[0]
        phi_i = snap['phi_i'] if isinstance(snap, dict) else snap[1]
        phi_v = snap['phi_v'] if isinstance(snap, dict) else snap[2]
        t = hist[si]['time']

        for ax in axes.flat:
            ax.clear()

        if phi_f.ndim == 3:
            Nx, Ny, Nz = phi_f.shape
            # XY midplane
            axes[0,0].imshow(phi_f[:,:,Nz//2].T + 0.5*phi_i[:,:,Nz//2].T,
                            origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[0,0].set_title(f'XY midplane (t={t:.1f}h)')

            # XZ midplane
            axes[0,1].imshow(phi_f[:,Ny//2,:].T + 0.5*phi_i[:,Ny//2,:].T,
                            origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[0,1].set_title('XZ midplane')

            # YZ midplane
            axes[1,0].imshow(phi_f[Nx//2,:,:].T + 0.5*phi_i[Nx//2,:,:].T,
                            origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[1,0].set_title('YZ midplane')

            # RGB composite (XY midplane)
            mx = max(phi_f[:,:,Nz//2].max(), phi_i[:,:,Nz//2].max(),
                    phi_v[:,:,Nz//2].max(), 0.01)
            rgb = np.stack([phi_f[:,:,Nz//2]/mx, phi_v[:,:,Nz//2]/mx,
                           phi_i[:,:,Nz//2]/mx], axis=-1)
            axes[1,1].imshow(np.clip(rgb.transpose(1,0,2), 0, 1), origin='lower')
            axes[1,1].set_title('RGB (R=func, G=void, B=inert)')
        else:
            mx = max(phi_f.max(), phi_i.max(), phi_v.max(), 0.01)
            axes[0,0].imshow(phi_f.T, origin='lower', cmap='Oranges')
            axes[0,0].set_title(f'Functional (t={t:.1f}h)')
            axes[0,1].imshow(phi_i.T, origin='lower', cmap='Blues')
            axes[0,1].set_title('Inert')
            axes[1,0].imshow(phi_v.T, origin='lower', cmap='Greens')
            axes[1,0].set_title('Void')
            rgb = np.stack([phi_f/mx, phi_v/mx, phi_i/mx], axis=-1)
            axes[1,1].imshow(np.clip(rgb.transpose(1,0,2), 0, 1), origin='lower')
            axes[1,1].set_title('RGB composite')

        for ax in axes.flat:
            ax.set_xticks([]); ax.set_yticks([])
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    anim.save(str(outpath), writer=PillowWriter(fps=3))
    plt.close(fig)
    print(f"  Composite GIF saved: {outpath}")


def run_all(snaps, hist, p, outdir=None):
    """Generate all movie visualizations."""
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        od = Path(outdir)
    else:
        od = Path('.')

    if len(snaps) > 0:
        # Last frame for rotating and z-sweep
        rotating_gif(snaps[-1], p, od / 'rotating.gif')
        zsweep_gif(snaps[-1], p, od / 'zsweep.gif')
        timelapse_gif(snaps, hist, p, od / 'timelapse.gif')
        composite_gif(snaps, hist, p, od / 'composite.gif')

    print(f"  Movie files saved to {outdir or '.'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie visualization')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    # V1.5: Load from disk via load_run()
    try:
        from new_dem_0 import load_run
        hist, snaps, p, meta = load_run(str(indir))
        run_all(snaps, hist, p, outdir=outdir)
    except Exception as e:
        print(f"Could not load run data: {e}")
        print(f"Use programmatically: from viz_movies import run_all")
        print(f"  run_all(snaps, hist, p, outdir='{outdir}')")
