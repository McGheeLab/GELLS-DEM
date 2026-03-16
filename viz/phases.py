"""
Individual Phase Volume Visualization
======================================
Renders each solid phase (functional, inert) and the liquid (void)
as separate 3D volumes evolving over time.

Plots produced:
  1. Three-panel 3D isosurface strip at selected times
  2. Phase volume fractions vs time
  3. Tri-plane evolution: XY, XZ, YZ midplane slices per phase
  4. Phase interface area vs time (isosurface area at phi=0.5)

Usage:
    python viz/phases.py -i ./simulations/run1
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

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

from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# ── Helpers ──

def _extract_fields(snap):
    """Return (phi_f, phi_i, phi_v) from a snapshot dict or tuple."""
    if isinstance(snap, dict):
        return snap['phi_f'], snap['phi_i'], snap['phi_v']
    return snap[0], snap[1], snap[2]


def isosurface_area(phi, level=0.5, spacing=(1, 1, 1)):
    """Compute the area of an isosurface via marching cubes."""
    if not HAS_SKIMAGE:
        return 0.0
    try:
        verts, faces, _, _ = marching_cubes(phi, level=level, spacing=spacing)
    except (ValueError, RuntimeError):
        return 0.0
    # Triangle areas via cross product
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.sum(np.linalg.norm(cross, axis=1))


# ── Plot 1: Three-panel isosurface strip ──

def plot_isosurface_strip(snaps, hist, p, indices=None, outdir=None):
    """Three-panel 3D isosurface strip at selected times.

    Each column is a time step; rows are functional, inert, void.
    """
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))

    n_times = len(indices)
    phi_f0, _, _ = _extract_fields(snaps[0])
    is_3d = phi_f0.ndim == 3

    phases = ['Functional', 'Inert', 'Void']
    cmaps_iso = ['orangered', 'steelblue', 'limegreen']

    if is_3d and HAS_PYVISTA:
        # PyVista rendering: one image per (phase, time)
        fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 12))
        if n_times == 1:
            axes = axes.reshape(3, 1)

        Ng = phi_f0.shape[0]
        sp = (p.Lx / Ng, p.Ly / Ng, p.Lz / Ng) if hasattr(p, 'Lz') else (1, 1, 1)

        for col, si in enumerate(indices):
            fields = list(_extract_fields(snaps[si]))
            t = hist[si]['time']
            for row, (fld, name, color) in enumerate(zip(fields, phases, cmaps_iso)):
                plotter = pv.Plotter(off_screen=True, window_size=(400, 400))
                grid = pv.ImageData(dimensions=fld.shape, spacing=sp)
                grid.point_data['values'] = fld.ravel(order='F')
                try:
                    contour = grid.contour([0.3], scalars='values')
                    plotter.add_mesh(contour, color=color, opacity=0.6)
                except Exception:
                    pass
                plotter.camera_position = 'iso'
                img = plotter.screenshot(return_img=True)
                plotter.close()
                axes[row, col].imshow(img)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if col == 0:
                    axes[row, col].set_ylabel(name, fontsize=12)
                if row == 0:
                    axes[row, col].set_title(f't = {t:.1f} h', fontsize=11)

        fig.suptitle('Phase Isosurfaces Over Time', fontsize=14)
        plt.tight_layout()
        if outdir:
            fig.savefig(Path(outdir) / 'phase_isosurfaces.png', dpi=150)
        return fig

    # Matplotlib fallback: midplane slices
    fig, axes = plt.subplots(3, n_times, figsize=(4 * n_times, 12))
    if n_times == 1:
        axes = axes.reshape(3, 1)
    cmaps_2d = ['Oranges', 'Blues', 'Greens']

    for col, si in enumerate(indices):
        fields = list(_extract_fields(snaps[si]))
        t = hist[si]['time']
        for row, (fld, name, cm) in enumerate(zip(fields, phases, cmaps_2d)):
            if fld.ndim == 3:
                mid_z = fld.shape[2] // 2
                img = fld[:, :, mid_z].T
            else:
                img = fld.T
            axes[row, col].imshow(img, origin='lower', cmap=cm,
                                  vmin=0, vmax=max(0.01, fld.max()))
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if col == 0:
                axes[row, col].set_ylabel(name, fontsize=12)
            if row == 0:
                axes[row, col].set_title(f't = {t:.1f} h', fontsize=11)

    fig.suptitle('Phase Fields Over Time (midplane)', fontsize=14)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'phase_isosurfaces.png', dpi=150)
    return fig


# ── Plot 2: Phase volume fractions vs time ──

def plot_phase_fractions(hist, outdir=None):
    """Phase volume fractions (functional, inert, void) vs time."""
    t = [h['time'] for h in hist]
    phi_f = [h.get('phi_f_mean', 0) for h in hist]
    phi_i = [h.get('phi_i_mean', 0) for h in hist]
    phi_v = [h.get('porosity', h.get('phi_v_mean', 0)) for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Individual lines
    ax1.plot(t, phi_f, 'C1-', lw=2, label='Functional')
    ax1.plot(t, phi_i, 'C0-', lw=2, label='Inert')
    ax1.plot(t, phi_v, 'C2-', lw=2, label='Void (liquid)')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Volume fraction')
    ax1.set_title('Phase Fractions Over Time')
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Normalised (should sum to ~1)
    total = [f + i + v for f, i, v in zip(phi_f, phi_i, phi_v)]
    phi_f_n = [f / max(s, 1e-12) for f, s in zip(phi_f, total)]
    phi_i_n = [i / max(s, 1e-12) for i, s in zip(phi_i, total)]
    phi_v_n = [v / max(s, 1e-12) for v, s in zip(phi_v, total)]

    ax2.stackplot(t, phi_f_n, phi_i_n, phi_v_n,
                  labels=['Functional', 'Inert', 'Void'],
                  colors=['orangered', 'steelblue', 'lightgreen'], alpha=0.8)
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Normalised fraction')
    ax2.set_title('Normalised Phase Fractions (Stacked)')
    ax2.legend(loc='center right')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'phase_fractions.png', dpi=150)
    return fig


# ── Plot 3: Tri-plane evolution ──

def plot_triplane(snaps, hist, p, indices=None, outdir=None):
    """XY, XZ, YZ midplane slices for each phase at selected times."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n // 2, n - 1]))

    phi_f0, _, _ = _extract_fields(snaps[0])
    if phi_f0.ndim != 3:
        print("  Skipping tri-plane: field is not 3D")
        return None

    n_times = len(indices)
    # 3 rows (XY, XZ, YZ) x n_times columns; each cell is a composite image
    fig, axes = plt.subplots(3, n_times, figsize=(5 * n_times, 14))
    if n_times == 1:
        axes = axes.reshape(3, 1)

    plane_labels = ['XY midplane', 'XZ midplane', 'YZ midplane']

    for col, si in enumerate(indices):
        phi_f, phi_i, phi_v = _extract_fields(snaps[si])
        t = hist[si]['time']
        Nx, Ny, Nz = phi_f.shape

        # Composite RGB: R=functional, G=void, B=inert
        mx = max(phi_f.max(), phi_i.max(), phi_v.max(), 0.01)

        # XY midplane (z = Nz//2)
        rgb_xy = np.stack([phi_f[:, :, Nz // 2] / mx,
                           phi_v[:, :, Nz // 2] / mx,
                           phi_i[:, :, Nz // 2] / mx], axis=-1)
        axes[0, col].imshow(np.clip(rgb_xy.transpose(1, 0, 2), 0, 1),
                            origin='lower',
                            extent=[0, p.Lx, 0, p.Ly])
        axes[0, col].set_xlabel('X (µm)')
        axes[0, col].set_ylabel('Y (µm)')

        # XZ midplane (y = Ny//2)
        rgb_xz = np.stack([phi_f[:, Ny // 2, :] / mx,
                           phi_v[:, Ny // 2, :] / mx,
                           phi_i[:, Ny // 2, :] / mx], axis=-1)
        axes[1, col].imshow(np.clip(rgb_xz.transpose(1, 0, 2), 0, 1),
                            origin='lower',
                            extent=[0, p.Lx, 0, p.Lz])
        axes[1, col].set_xlabel('X (µm)')
        axes[1, col].set_ylabel('Z (µm)')

        # YZ midplane (x = Nx//2)
        rgb_yz = np.stack([phi_f[Nx // 2, :, :] / mx,
                           phi_v[Nx // 2, :, :] / mx,
                           phi_i[Nx // 2, :, :] / mx], axis=-1)
        axes[2, col].imshow(np.clip(rgb_yz.transpose(1, 0, 2), 0, 1),
                            origin='lower',
                            extent=[0, p.Ly, 0, p.Lz])
        axes[2, col].set_xlabel('Y (µm)')
        axes[2, col].set_ylabel('Z (µm)')

        for row in range(3):
            if col == 0:
                axes[row, col].set_ylabel(
                    f'{plane_labels[row]}\n{axes[row, col].get_ylabel()}',
                    fontsize=10)
            axes[row, col].set_title(f't = {t:.1f} h' if row == 0 else '')

    fig.suptitle('Tri-Plane Phase Evolution (R=func, G=void, B=inert)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'triplane_evolution.png', dpi=150)
    return fig


# ── Plot 4: Phase interface area vs time ──

def plot_interface_area(snaps, hist, p, outdir=None):
    """Phase interface area (isosurface area at phi=0.5) vs time."""
    phi_f0, _, _ = _extract_fields(snaps[0])
    if phi_f0.ndim != 3:
        print("  Skipping interface area: field is not 3D or skimage unavailable")
        return None

    Ng = phi_f0.shape[0]
    sp = (p.Lx / Ng, p.Ly / Ng, p.Lz / Ng) if hasattr(p, 'Lz') else (1, 1, 1)

    t_arr = []
    area_f, area_i, area_v = [], [], []

    for si, snap in enumerate(snaps):
        phi_f, phi_i, phi_v = _extract_fields(snap)
        t_arr.append(hist[si]['time'])
        area_f.append(isosurface_area(phi_f, 0.5, sp))
        area_i.append(isosurface_area(phi_i, 0.5, sp))
        area_v.append(isosurface_area(phi_v, 0.5, sp))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_arr, area_f, 'C1-', lw=2, label='Functional')
    ax.plot(t_arr, area_i, 'C0-', lw=2, label='Inert')
    ax.plot(t_arr, area_v, 'C2-', lw=2, label='Void')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Interface area (µm²)')
    ax.set_title('Phase Interface Area Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'interface_area.png', dpi=150)
    return fig


# ── Run all ──

def run_all(hist, snaps=None, p=None, outdir=None):
    """Generate all phase visualizations."""
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    figs = {}
    figs['phase_fractions'] = plot_phase_fractions(hist, outdir)

    if snaps is not None and len(snaps) > 0:
        figs['isosurface_strip'] = plot_isosurface_strip(snaps, hist, p, outdir=outdir)
        figs['triplane'] = plot_triplane(snaps, hist, p, outdir=outdir)
        if HAS_SKIMAGE:
            figs['interface_area'] = plot_interface_area(snaps, hist, p, outdir=outdir)

    print(f"  Phase plots saved to {outdir or 'memory'}")
    return figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Phase volume visualization')
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--outdir', default=None, help='Output directory')
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    # V1.5: Load from disk via load_run()
    try:
        from new_dem_0 import load_run
        hist, snaps, p, meta = load_run(str(indir))
        run_all(hist, snaps=snaps, p=p, outdir=outdir)
    except Exception as e:
        print(f"Could not load run data: {e}")
        hist_file = indir / 'history.json'
        if hist_file.exists():
            with open(hist_file) as f:
                hist = json.load(f)
            run_all(hist, outdir=outdir)
        else:
            print(f"No history.json found in {indir}")
