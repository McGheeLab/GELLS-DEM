"""
3D Scaffold Map Visualization (Module 7 — PyVista)
====================================================
Off-screen PyVista renders of the 3D granular scaffold:
  - Superellipsoid granule meshes colored by type (functional/inert)
  - Cell spheres colored by state
  - Bridge cylinders between bridging cells and targets
  - Z-midplane clip plane for cross-section view

Multi-panel composites assembled in matplotlib.
Guarded by HAS_PYVISTA; silently skipped when unavailable or data is 2D.

Usage:
    python viz2/scaffold_map_3d.py -i results/DOE_3D_0001
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

try:
    import pyvista as pv
    pv.OFF_SCREEN = True
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from viz2.common import (
    select_indices, get_snap_time, save_fig,
    is_3d, cell_world_positions_3d, CellState,
    FUNC_COLOR, INERT_COLOR, CELL_COLORS,
)


# PyVista color constants (RGB 0-1)
C_FUNC_PV   = (0.80, 0.13, 0.13)   # matches FUNC_COLOR #CC2222
C_INERT_PV  = (0.13, 0.67, 0.13)   # matches INERT_COLOR #22AA22
C_BRIDGE_PV = (0.86, 0.08, 0.24)   # crimson
C_BG        = (0.04, 0.04, 0.04)   # near-black (matches void)

# Map CellState int -> PyVista RGB
_CELL_PV_COLORS = {
    int(CellState.ATTACHED):      (0.85, 0.65, 0.13),
    int(CellState.SPREADING):     (1.00, 0.55, 0.00),
    int(CellState.PROLIFERATING): (0.13, 0.55, 0.13),
    int(CellState.BRIDGING):      (0.86, 0.08, 0.24),
    int(CellState.SENESCENT):     (0.41, 0.41, 0.41),
    int(CellState.MIGRATING):     (0.25, 0.41, 0.88),
}


# ======================================================================
# Mesh generation (reuse viz/stress.py pattern)
# ======================================================================

def _granule_mesh(snap, gi, n_pts=24):
    """Generate a watertight PyVista PolyData mesh for 3D granule gi.

    Returns mesh or None.
    """
    if not HAS_PYVISTA:
        return None

    from new_dem_0 import _sgnpow, quat_rotate

    a = float(snap['a'][gi])
    b = float(snap['b'][gi])
    c = float(snap.get('c', snap['a'])[gi])
    n1 = float(snap['n1'][gi])
    n2 = float(snap.get('n2', snap['n1'])[gi])
    cx = float(snap['x'][gi])
    cy = float(snap['y'][gi])
    cz = float(snap.get('z', np.zeros(len(snap['x'])))[gi])

    n_lat = n_pts
    n_lon = n_pts

    eta = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]
    omega = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    E, O = np.meshgrid(eta, omega, indexing='ij')

    e2 = 2.0 / n2
    e1 = 2.0 / n1
    X = a * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    Y = b * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    Z = c * _sgnpow(np.sin(E), e2)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Pole vertices
    south = np.array([[0.0, 0.0, -c]])
    north = np.array([[0.0, 0.0,  c]])
    verts = np.vstack([verts, south, north])
    south_idx = n_lat * n_lon
    north_idx = south_idx + 1

    # Faces
    faces = []
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([3, south_idx, j, jn])
    for i in range(n_lat - 1):
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            v00 = i * n_lon + j
            v01 = i * n_lon + jn
            v10 = (i + 1) * n_lon + j
            v11 = (i + 1) * n_lon + jn
            faces.append([3, v00, v10, v11])
            faces.append([3, v00, v11, v01])
    base = (n_lat - 1) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([3, north_idx, base + jn, base + j])

    faces_arr = np.array(faces, dtype=np.int32).ravel()

    # Rotate to world frame
    if 'quat' in snap:
        q = snap['quat'][gi]
        verts = np.array([quat_rotate(q, v) for v in verts])

    verts[:, 0] += cx
    verts[:, 1] += cy
    verts[:, 2] += cz

    try:
        mesh = pv.PolyData(verts, faces_arr)
        mesh.compute_normals(consistent_normals=True, inplace=True)
        return mesh
    except Exception:
        return None


# ======================================================================
# Single snapshot render
# ======================================================================

def render_snapshot_3d(snap, p, clip_z=True, window_size=(800, 800)):
    """Off-screen PyVista render of a 3D snapshot.

    Parameters
    ----------
    snap : dict
        3D snapshot data.
    p : Params
    clip_z : bool
        If True, clip at z = Lz*0.55 for cross-section view.
    window_size : tuple

    Returns
    -------
    img : ndarray (H, W, 3) RGB image, or None if PyVista unavailable.
    """
    if not HAS_PYVISTA:
        return None

    from new_dem_0 import quat_rotate

    Lx, Ly = float(p.Lx), float(p.Ly)
    Lz = float(getattr(p, 'Lz', Ly))
    z_cut = Lz * 0.55 if clip_z else Lz * 1.1
    N = len(snap['x'])

    pl = pv.Plotter(off_screen=True, window_size=window_size,
                     lighting='three lights')
    pl.set_background(C_BG)

    # Wireframe bounding box
    kept_box = pv.Box(bounds=[0, Lx, 0, Ly, 0, min(z_cut, Lz)])
    pl.add_mesh(kept_box, style='wireframe', color='#555555',
                line_width=0.8, opacity=0.3)

    # Granules
    for gi in range(N):
        z_gi = float(snap.get('z', np.zeros(N))[gi])
        r_bound = float(np.max([
            snap.get('a', snap['r'])[gi],
            snap.get('b', snap['r'])[gi],
            snap.get('c', snap['r'])[gi],
        ]))
        if z_gi - r_bound > z_cut:
            continue

        mesh = _granule_mesh(snap, gi, n_pts=20)
        if mesh is None:
            continue

        if z_gi + r_bound > z_cut and clip_z:
            try:
                mesh = mesh.clip(normal=(0, 0, -1), origin=(0, 0, z_cut))
            except Exception:
                continue
        if mesh.n_points < 3:
            continue

        gt = int(snap['gtype'][gi])
        color = C_FUNC_PV if gt == 0 else C_INERT_PV

        pl.add_mesh(mesh, color=color, opacity=1.0,
                    smooth_shading=True, specular=0.25,
                    diffuse=0.85, ambient=0.15)

    # Cells
    cell_states = snap.get('cell_state', np.array([], dtype=int))
    n_cells = len(cell_states)
    if n_cells > 0 and 'z' in snap:
        wx, wy, wz = cell_world_positions_3d(snap)
        cell_r = max(3.0, min(Lx, Ly) * 0.008)

        for ci in range(n_cells):
            if wz[ci] > z_cut:
                continue
            state = int(cell_states[ci])
            c_rgb = _CELL_PV_COLORS.get(state, (0.6, 0.6, 0.6))

            # Bridge tubes
            if state == int(CellState.BRIDGING):
                bt = int(snap.get('cell_bridge_target', np.full(n_cells, -1))[ci])
                if 0 <= bt < N:
                    tx = float(snap['x'][bt])
                    ty = float(snap['y'][bt])
                    tz = float(snap['z'][bt])
                    if tz <= z_cut:
                        tube_r = max(1.5, cell_r * 0.4)
                        try:
                            tube = pv.Line([wx[ci], wy[ci], wz[ci]],
                                          [tx, ty, tz]).tube(radius=tube_r)
                            if clip_z:
                                tube = tube.clip(normal=(0, 0, -1),
                                                origin=(0, 0, z_cut))
                            if tube.n_points > 0:
                                pl.add_mesh(tube, color=C_BRIDGE_PV, opacity=0.7)
                        except Exception:
                            pass

            pl.add_mesh(pv.Sphere(radius=cell_r, center=[wx[ci], wy[ci], wz[ci]]),
                        color=c_rgb, opacity=1.0)

    # Camera
    cam_dist = max(Lx, Ly, Lz) * 2.0
    ctr = [Lx / 2, Ly / 2, min(z_cut, Lz) * 0.4]
    cam_pos = [Lx / 2 + cam_dist * 0.45,
               Ly / 2 - cam_dist * 0.45,
               ctr[2] + cam_dist * 0.65]
    pl.camera_position = [cam_pos, ctr, (0, 0, 1)]
    pl.enable_anti_aliasing('ssaa')

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ======================================================================
# Multi-panel figure
# ======================================================================

def plot_scaffold_3d(snaps, hist, p, indices=None, outdir=None):
    """Multi-panel 3D scaffold renders at selected timepoints."""
    if not HAS_PYVISTA:
        print("    PyVista not available, skipping 3D scaffold map")
        return None

    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        ax = axes[c]
        t = get_snap_time(hist, si)
        img = render_snapshot_3d(snaps[si], p, clip_z=True)
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"t = {t:.1f} h", fontsize=10, color='white')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.patch.set_facecolor('black')
    fig.suptitle("3D Scaffold Map (z-clip)", fontsize=12, y=1.02,
                 color='white')
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'scaffold_3d.png')
    return fig


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate 3D scaffold visualization. Skips if 2D or no PyVista."""
    if not snaps:
        return
    if not is_3d(snaps[0], p):
        return
    if not HAS_PYVISTA:
        print("  [7/7] 3D scaffold map... SKIPPED (no PyVista)")
        return

    print("  [7/7] 3D scaffold map (PyVista)...")
    fig = plot_scaffold_3d(snaps, hist, p, outdir=outdir)
    if fig is not None:
        plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D scaffold map viz (PyVista)')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
