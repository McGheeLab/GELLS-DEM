"""
3D Surface Stress & Granule Isosurface Visualization
=====================================================
PyVista-based rendering of Hertzian contact pressure mapped onto
superellipsoid granule surfaces, with cross-sectional views,
isolated granule evolution GIFs, and 3D granules.png.

Plots produced:
  1. Stress cross-section — z-slab of thickness ~1 granule, surface stress colormap
  2. Granule evolution GIFs — isolated granule groups with stress evolution
  3. granules_3d.png — isosurface rendering with z-slice opacity
  4. Cell ellipsoid meshes for 3D cell rendering

Usage:
    python viz/stress.py -i ./results/run1
    python viz/stress.py -i ./results/run1 -o ./plots

Or import programmatically:
    from viz.stress import run_all
    run_all(snaps, hist, p, outdir='plots/')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
import argparse

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from new_dem_0 import (
    superellipsoid_point, superellipsoid_normal, superellipsoid_mesh,
    quat_rotate, quat_rotate_inv, CellState,
    find_contact_superellipsoids_3d, find_contact_spheres_3d,
)


# ── Mesh generation ──────────────────────────────────────────────────

def granule_surface_mesh(snap, gi, n_pts=32):
    """Generate a watertight PyVista PolyData mesh for granule gi.

    Creates a closed surface with proper omega wrapping and pole caps
    so granules render as solids, not hollow shells.

    Returns (mesh, vertices, faces) or (None, None, None) if pyvista missing.
    """
    if not HAS_PYVISTA:
        return None, None, None

    a = float(snap['a'][gi])
    b = float(snap['b'][gi])
    c = float(snap.get('c', snap['a'])[gi])
    n1 = float(snap['n1'][gi])
    n2 = float(snap.get('n2', snap['n1'])[gi])
    cx = float(snap['x'][gi])
    cy = float(snap['y'][gi])
    cz = float(snap.get('z', np.zeros(len(snap['x'])))[gi])

    n_lat = n_pts   # latitude rings (excluding poles)
    n_lon = n_pts   # longitude points per ring

    # Parametric grid: skip exact poles (degenerate) — added as single vertices
    eta = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]
    omega = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)  # wraps seamlessly
    E, O = np.meshgrid(eta, omega, indexing='ij')

    # Body-frame vertices via superellipsoid parametric equations
    from new_dem_0 import _sgnpow
    e2 = 2.0 / n2
    e1 = 2.0 / n1
    X = a * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    Y = b * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    Z = c * _sgnpow(np.sin(E), e2)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Pole vertices (single point each, not degenerate rings)
    south = np.array([[0.0, 0.0, -c]])
    north = np.array([[0.0, 0.0,  c]])
    verts = np.vstack([verts, south, north])
    south_idx = n_lat * n_lon
    north_idx = south_idx + 1

    # Triangle faces with proper topology
    faces = []

    # South pole fan → first latitude ring (row 0)
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([3, south_idx, j, jn])

    # Quad strips between adjacent latitude rings (omega wraps via modulo)
    for i in range(n_lat - 1):
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            v00 = i * n_lon + j
            v01 = i * n_lon + jn
            v10 = (i + 1) * n_lon + j
            v11 = (i + 1) * n_lon + jn
            faces.append([3, v00, v10, v11])
            faces.append([3, v00, v11, v01])

    # North pole fan → last latitude ring
    base = (n_lat - 1) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([3, north_idx, base + jn, base + j])

    faces_arr = np.array(faces, dtype=np.int32).ravel()

    # Rotate to world frame
    if 'quat' in snap:
        q = snap['quat'][gi]
        verts = np.array([quat_rotate(q, v) for v in verts])

    # Translate
    verts[:, 0] += cx
    verts[:, 1] += cy
    verts[:, 2] += cz

    mesh = pv.PolyData(verts, faces_arr)
    mesh.compute_normals(consistent_normals=True, inplace=True)
    return mesh, verts, faces_arr


# ── Surface stress computation ───────────────────────────────────────

def compute_surface_stress(mesh, contacts, gi):
    """Map Hertzian contact pressure to mesh vertices for granule gi.

    Returns stress array of shape (n_vertices,) in kPa.
    """
    n_verts = mesh.n_points
    stress = np.zeros(n_verts)
    verts = np.array(mesh.points)

    # Filter contacts involving this granule
    gi_contacts = [c for c in contacts
                   if c['i'] == gi or c['j'] == gi]
    if not gi_contacts:
        return stress

    for c in gi_contacts:
        overlap = c['overlap']
        R_eff = c['R_eff']
        F_normal = abs(c['F_normal'])

        if overlap <= 0 or R_eff <= 0:
            continue

        # Hertzian contact radius
        a_c = np.sqrt(R_eff * overlap)
        if a_c < 1e-6:
            continue

        # Peak Hertzian pressure: p0 = 3F / (2 pi a_c^2)  [nN / µm² = kPa * 1e-3]
        # Convert to kPa: F is in nN, a_c in µm → p0 in nN/µm² = kPa * 1e-3
        # Actually nN/µm² = 1e-9 N / (1e-6 m)² = 1e-9 / 1e-12 = 1e3 Pa = 1 kPa
        p0 = (3.0 * F_normal) / (2.0 * np.pi * a_c**2)  # in kPa

        # Contact point
        cp = np.array([c['cx'], c['cy'], c.get('cz', 0.0)])

        # Distance from each vertex to contact point
        dv = verts - cp[np.newaxis, :]
        r = np.linalg.norm(dv, axis=1)

        # Hertzian pressure distribution: p(r) = p0 * sqrt(1 - r²/a_c²)
        inside = r < a_c
        stress[inside] += p0 * np.sqrt(1.0 - (r[inside] / a_c)**2)

        # Gaussian smoothing tail for a_c < r < 2*a_c
        sigma = a_c * 0.3
        tail = (r >= a_c) & (r < 2 * a_c)
        stress[tail] += p0 * np.exp(-((r[tail] - a_c) / sigma)**2)

    return stress


def _contacts_from_snap(snap):
    """Extract contacts list from snapshot dict."""
    contacts = snap.get('contacts', [])
    if contacts:
        return contacts
    # Try to reconstruct from contact arrays (disk-loaded data)
    if 'contact_i' in snap:
        n_c = len(snap['contact_i'])
        contacts = []
        for k in range(n_c):
            contacts.append({
                'i': int(snap['contact_i'][k]),
                'j': int(snap['contact_j'][k]),
                'cx': float(snap['contact_cx'][k]),
                'cy': float(snap['contact_cy'][k]),
                'cz': float(snap.get('contact_cz', np.zeros(n_c))[k]),
                'nx': float(snap['contact_nx'][k]),
                'ny': float(snap['contact_ny'][k]),
                'nz': float(snap.get('contact_nz', np.zeros(n_c))[k]),
                'overlap': float(snap['contact_overlap'][k]),
                'R_eff': float(snap['contact_R_eff'][k]),
                'F_normal': float(snap['contact_F_normal'][k]),
                'A_contact': float(snap['contact_A_contact'][k]),
            })
        return contacts
    return []


def reconstruct_contacts(snap, p):
    """Re-run contact detection to recover contacts for older snapshots.

    Fallback when snapshot has no stored contact data.
    """
    from scipy.spatial import cKDTree

    xs, ys = snap['x'], snap['y']
    zs = snap.get('z', np.zeros(len(xs)))
    N = len(xs)
    rs = snap['r']
    a_arr, b_arr = snap['a'], snap['b']
    c_arr = snap.get('c', a_arr.copy())
    n1_arr = snap['n1']
    n2_arr = snap.get('n2', n1_arr.copy())
    quats = snap.get('quat', None)

    is_sphere = np.all(a_arr == b_arr) and np.all(a_arr == c_arr) and np.all(n1_arr == 2.0)

    r_bound = np.maximum(a_arr, np.maximum(b_arr, c_arr))
    max_r = float(np.max(r_bound))
    cutoff = 2 * max_r + 50.0  # generous cutoff

    pos = np.column_stack([xs, ys, zs])
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    contacts = []
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        if is_sphere or (a_arr[i] == b_arr[i] == c_arr[i] and n1_arr[i] == 2.0 and
                         a_arr[j] == b_arr[j] == c_arr[j] and n1_arr[j] == 2.0):
            result = find_contact_spheres_3d(
                xs[i], ys[i], zs[i], rs[i],
                xs[j], ys[j], zs[j], rs[j])
        else:
            qi = quats[i] if quats is not None else np.array([1, 0, 0, 0.])
            qj = quats[j] if quats is not None else np.array([1, 0, 0, 0.])
            result = find_contact_superellipsoids_3d(
                xs[i], ys[i], zs[i],
                a_arr[i], b_arr[i], c_arr[i], n1_arr[i], n2_arr[i], qi,
                xs[j], ys[j], zs[j],
                a_arr[j], b_arr[j], c_arr[j], n1_arr[j], n2_arr[j], qj)

        if result is not None:
            _, overlap, nx, ny, nz, cx, cy, cz, R_eff = result
            if overlap > 0:
                from new_dem_0 import hertz_contact_force
                nu = getattr(p, 'poisson_ratio', 0.49)
                E_star = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu**2))
                Fc = hertz_contact_force(E_star, R_eff, overlap)
                A_contact = np.pi * R_eff * overlap
                contacts.append({
                    'i': int(i), 'j': int(j),
                    'cx': cx, 'cy': cy, 'cz': cz,
                    'nx': nx, 'ny': ny, 'nz': nz,
                    'overlap': overlap, 'R_eff': R_eff,
                    'F_normal': float(Fc), 'A_contact': A_contact,
                })

    return contacts


# ── Cross-section stress map ─────────────────────────────────────────

def _get_contacts(snap, p):
    """Get contacts from snapshot, reconstructing if needed."""
    contacts = _contacts_from_snap(snap)
    if not contacts:
        contacts = reconstruct_contacts(snap, p)
    return contacts


def _func_r_mean(snap):
    """Mean radius of functional granules."""
    gtype = snap['gtype']
    rs = snap['r']
    func_mask = gtype == 0
    if func_mask.any():
        return float(np.mean(rs[func_mask]))
    return float(np.mean(rs))


def plot_stress_cross_section(snaps, hist, p, indices=None, outdir=None):
    """Cross-sectional z-projection stress map for several timepoints."""
    if not HAS_PYVISTA:
        print("  viz_stress: skipping cross-section (pyvista not available)")
        return None

    n = len(snaps)
    if indices is None:
        indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    nc = len(indices)

    # Composite figure via individual PyVista screenshots
    fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 5))
    if nc == 1:
        axes = [axes]

    # Global stress range for consistent coloring
    global_max = 0.0
    for si in indices:
        snap = snaps[si]
        contacts = _get_contacts(snap, p)
        for c in contacts:
            if c['overlap'] > 0 and c['R_eff'] > 0:
                a_c = np.sqrt(c['R_eff'] * c['overlap'])
                if a_c > 1e-6:
                    p0 = (3.0 * abs(c['F_normal'])) / (2.0 * np.pi * a_c**2)
                    global_max = max(global_max, p0)
    if global_max < 1e-6:
        global_max = 1.0

    for c_idx, si in enumerate(indices):
        snap = snaps[si]
        contacts = _get_contacts(snap, p)

        R_mean = _func_r_mean(snap)
        z_mid = p.Lz / 2.0
        z_lo = z_mid - R_mean
        z_hi = z_mid + R_mean

        zs = snap.get('z', np.zeros(len(snap['x'])))

        # Render with PyVista
        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
        plotter.set_background('white')

        for gi in range(len(snap['x'])):
            gz = float(zs[gi])
            in_slab = z_lo <= gz <= z_hi

            mesh, _, _ = granule_surface_mesh(snap, gi, n_pts=24)
            if mesh is None:
                continue

            if in_slab:
                # Compute and apply stress
                stress = compute_surface_stress(mesh, contacts, gi)
                mesh.point_data['stress'] = stress

                # Clip to z-slab
                clipped = mesh.clip(normal='z', origin=(0, 0, z_hi), invert=True)
                if clipped.n_points > 0:
                    clipped = clipped.clip(normal='-z', origin=(0, 0, z_lo), invert=True)
                if clipped.n_points > 0:
                    plotter.add_mesh(clipped, scalars='stress', cmap='YlOrRd',
                                     clim=[0, global_max], show_scalar_bar=False)
            else:
                # Ghost granules outside slab
                plotter.add_mesh(mesh, color='lightgray', opacity=0.08)

        plotter.camera_position = 'xy'
        plotter.camera.zoom(1.2)

        # Screenshot to numpy array
        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax = axes[c_idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f't = {t:.1f} h', fontsize=10)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                norm=plt.Normalize(0, global_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label('Contact Pressure (kPa)', fontsize=10)

    fig.suptitle('Stress Cross-Section (z-slab)', fontsize=13, y=1.02)
    plt.tight_layout()

    if outdir:
        path = str(Path(outdir) / 'stress_cross_section.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  viz_stress: saved {path}")
    return fig


# ── Granule selection ────────────────────────────────────────────────

def select_interesting_granules(snaps, hist, p, n=3):
    """Auto-select granules for evolution GIFs by contacts, stress, bridges."""
    snap = snaps[-1]
    contacts = _get_contacts(snap, p)
    N = len(snap['x'])
    gtype = snap['gtype']
    func_mask = gtype == 0

    # Score 1: contact count
    contact_count = np.zeros(N)
    for c in contacts:
        contact_count[c['i']] += 1
        contact_count[c['j']] += 1

    # Score 2: total force magnitude
    fx = snap.get('force_x', np.zeros(N))
    fy = snap.get('force_y', np.zeros(N))
    fz = snap.get('force_z', np.zeros(N))
    fmag = np.sqrt(fx**2 + fy**2 + fz**2)

    # Score 3: bridge count (cells bridging TO this granule)
    bridge_count = np.zeros(N)
    bt = snap.get('cell_bridge_target', np.array([], dtype=int))
    for target in bt:
        if 0 <= target < N:
            bridge_count[target] += 1

    # Exclude boundary granules (within 20% of domain edge)
    xs, ys = snap['x'], snap['y']
    zs = snap.get('z', np.zeros(N))
    excl = 0.2
    interior = ((xs > excl * p.Lx) & (xs < (1 - excl) * p.Lx) &
                (ys > excl * p.Ly) & (ys < (1 - excl) * p.Ly))
    if hasattr(p, 'Lz') and p.Lz > 0:
        interior &= (zs > excl * p.Lz) & (zs < (1 - excl) * p.Lz)

    # Only functional granules in interior
    eligible = func_mask & interior

    if not eligible.any():
        eligible = func_mask  # fallback
    if not eligible.any():
        return list(range(min(n, N)))

    # Composite score (normalized)
    def _norm(arr):
        mx = arr.max()
        return arr / mx if mx > 0 else arr

    score = (_norm(contact_count) + _norm(fmag) + _norm(bridge_count))
    score[~eligible] = -1

    # Top n unique
    ranked = np.argsort(score)[::-1]
    selected = []
    for gi in ranked:
        if len(selected) >= n:
            break
        if score[gi] > 0:
            selected.append(int(gi))
    return selected


# ── Granule evolution GIF ────────────────────────────────────────────

def _get_connected_granules(snap, gi, contacts):
    """Find all granules directly in contact with gi."""
    connected = set()
    for c in contacts:
        if c['i'] == gi:
            connected.add(c['j'])
        elif c['j'] == gi:
            connected.add(c['i'])
    # Also check bridging targets
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    cell_bt = snap.get('cell_bridge_target', np.array([], dtype=int))
    for ci_idx in range(len(cell_gi)):
        if int(cell_gi[ci_idx]) == gi and 0 <= int(cell_bt[ci_idx]) < len(snap['x']):
            connected.add(int(cell_bt[ci_idx]))
        if int(cell_bt[ci_idx]) == gi and 0 <= int(cell_gi[ci_idx]) < len(snap['x']):
            connected.add(int(cell_gi[ci_idx]))
    return connected


def granule_evolution_gif(snaps, hist, p, gi, outdir=None):
    """Isolated granule group time evolution in isometric view."""
    if not HAS_PYVISTA:
        print(f"  viz_stress: skipping evolution GIF for granule {gi} (no pyvista)")
        return None

    # Collect all connected granules across all snapshots
    all_connected = set()
    for snap in snaps:
        contacts = _get_contacts(snap, p)
        all_connected.update(_get_connected_granules(snap, gi, contacts))
    all_connected.add(gi)
    group = sorted(all_connected)

    # Compute global stress range
    global_max = 0.0
    for snap in snaps:
        contacts = _get_contacts(snap, p)
        for g in group:
            if g >= len(snap['x']):
                continue
            for c in contacts:
                if (c['i'] == g or c['j'] == g) and c['overlap'] > 0 and c['R_eff'] > 0:
                    a_c = np.sqrt(c['R_eff'] * c['overlap'])
                    if a_c > 1e-6:
                        p0 = (3.0 * abs(c['F_normal'])) / (2.0 * np.pi * a_c**2)
                        global_max = max(global_max, p0)
    if global_max < 1e-6:
        global_max = 1.0

    gif_name = f'granule_evolution_{gi}.gif'
    gif_path = str(Path(outdir) / gif_name) if outdir else gif_name

    plotter = pv.Plotter(off_screen=True, window_size=(600, 600))
    plotter.set_background('white')
    plotter.open_gif(gif_path)

    for si, snap in enumerate(snaps):
        plotter.clear()
        contacts = _get_contacts(snap, p)

        # Center camera on target granule
        cx = float(snap['x'][gi])
        cy = float(snap['y'][gi])
        cz = float(snap.get('z', np.zeros(len(snap['x'])))[gi])

        for g in group:
            if g >= len(snap['x']):
                continue
            mesh, _, _ = granule_surface_mesh(snap, g, n_pts=24)
            if mesh is None:
                continue

            stress = compute_surface_stress(mesh, contacts, g)
            mesh.point_data['stress'] = stress

            if g == gi:
                plotter.add_mesh(mesh, scalars='stress', cmap='YlOrRd',
                                 clim=[0, global_max], show_scalar_bar=(si == 0),
                                 opacity=1.0)
            else:
                plotter.add_mesh(mesh, scalars='stress', cmap='YlOrRd',
                                 clim=[0, global_max], show_scalar_bar=False,
                                 opacity=0.7)

        # Add cell ellipsoids for cells on this group
        _add_cell_meshes(plotter, snap, group, p)

        # Camera
        r_view = float(snap['r'][gi]) * 6
        plotter.camera_position = [
            (cx + r_view * 0.7, cy + r_view * 0.7, cz + r_view * 0.7),
            (cx, cy, cz),
            (0, 0, 1)]

        t = hist[si]['time'] if si < len(hist) else 0
        plotter.add_text(f't = {t:.1f} h', position='upper_left',
                         font_size=12, color='black')
        plotter.write_frame()

    plotter.close()
    print(f"  viz_stress: saved {gif_path}")
    return gif_path


# ── Cell ellipsoid mesh ──────────────────────────────────────────────

def cell_ellipsoid_mesh(snap, ci, p=None, n_pts=16):
    """Volume-conserving ellipsoid mesh for cell ci.

    Returns pv.PolyData mesh colored by cell state, or None.
    """
    if not HAS_PYVISTA:
        return None

    cell_states = snap.get('cell_state', np.array([], dtype=int))
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    if ci >= len(cell_states) or ci >= len(cell_gi):
        return None

    state = int(cell_states[ci])
    gi = int(cell_gi[ci])
    if gi < 0 or gi >= len(snap['x']):
        return None

    # Cell volume: 20 µm sphere → V = 4/3 π (10)³ ≈ 4189 µm³
    d_cell = 20.0
    r_cell = d_cell / 2.0
    V_cell = (4.0 / 3.0) * np.pi * r_cell**3

    # Get spread fraction for this granule
    sf = float(snap.get('spread_fraction', np.zeros(len(snap['x'])))[gi])

    # Shape depends on state
    if state == int(CellState.ATTACHED):
        ax_c = r_cell * max(0.3, 1.0 - 0.5 * sf)
        ax_a = ax_b = np.sqrt(V_cell / ((4.0/3.0) * np.pi * ax_c))
    elif state == int(CellState.SPREADING):
        ax_c = r_cell * max(0.2, 1.0 - 0.7 * sf)
        ax_a = ax_b = np.sqrt(V_cell / ((4.0/3.0) * np.pi * ax_c))
    elif state == int(CellState.PROLIFERATING):
        ax_c = 7.0
        ax_a = ax_b = np.sqrt(V_cell / ((4.0/3.0) * np.pi * ax_c))
    elif state == int(CellState.BRIDGING):
        # Prolate: elongated toward bridge target
        bt = int(snap.get('cell_bridge_target', np.full(len(cell_states), -1))[ci])
        if 0 <= bt < len(snap['x']):
            dx = snap['x'][bt] - snap['x'][gi]
            dy = snap['y'][bt] - snap['y'][gi]
            dz = snap.get('z', np.zeros(len(snap['x'])))[bt] - \
                 snap.get('z', np.zeros(len(snap['x'])))[gi]
            gap = np.sqrt(dx**2 + dy**2 + dz**2) - snap['r'][gi] - snap['r'][bt]
            gap = max(gap, 5.0)
            ax_a = gap / 2.0  # long axis = half bridge
            ax_a = max(ax_a, 8.0)  # minimum 8 µm
        else:
            ax_a = 15.0
        ax_b = ax_c = np.sqrt(V_cell / ((4.0/3.0) * np.pi * ax_a))
        ax_b = max(ax_b, 3.0)
        ax_c = ax_b
    elif state == int(CellState.SENESCENT):
        ax_c = 6.0
        ax_a = ax_b = np.sqrt(V_cell / ((4.0/3.0) * np.pi * ax_c))
    else:
        ax_a = ax_b = ax_c = r_cell

    # Generate watertight ellipsoid mesh from PyVista sphere primitive
    sphere = pv.Sphere(radius=1.0, theta_resolution=n_pts, phi_resolution=n_pts)
    verts = np.array(sphere.points, copy=True)
    verts[:, 0] *= ax_a
    verts[:, 1] *= ax_b
    verts[:, 2] *= ax_c
    faces_arr = sphere.faces

    # Compute world position
    wx, wy, wz = _cell_world_position_3d(snap, ci)

    # Orient: for bridging cells, align long axis toward target
    if state == int(CellState.BRIDGING):
        bt = int(snap.get('cell_bridge_target', np.full(len(cell_states), -1))[ci])
        if 0 <= bt < len(snap['x']):
            target_pos = np.array([
                snap['x'][bt], snap['y'][bt],
                snap.get('z', np.zeros(len(snap['x'])))[bt]])
            cell_pos = np.array([wx, wy, wz])
            direction = target_pos - cell_pos
            d_mag = np.linalg.norm(direction)
            if d_mag > 1e-6:
                direction /= d_mag
                # Rotation matrix aligning x-axis to direction
                R = _rotation_align_x(direction)
                verts = (R @ verts.T).T
    else:
        # Align with granule surface normal
        nx, ny, nz = _surface_normal_at_cell(snap, ci)
        R = _rotation_align_z(np.array([nx, ny, nz]))
        verts = (R @ verts.T).T

    verts[:, 0] += wx
    verts[:, 1] += wy
    verts[:, 2] += wz

    mesh = pv.PolyData(verts, faces_arr)
    mesh.compute_normals(consistent_normals=True, inplace=True)
    return mesh


def _cell_world_position_3d(snap, ci):
    """Compute world (x, y, z) for cell ci in 3D."""
    cell_gi = snap['cell_granule_id']
    gi = int(cell_gi[ci])
    if gi < 0 or gi >= len(snap['x']):
        return 0.0, 0.0, 0.0

    xs, ys = snap['x'], snap['y']
    zs = snap.get('z', np.zeros(len(xs)))
    a_arr, b_arr = snap['a'], snap['b']
    c_arr = snap.get('c', a_arr.copy())
    n1_arr = snap['n1']
    n2_arr = snap.get('n2', n1_arr.copy())

    cell_eta = snap.get('cell_eta_local', np.zeros(len(cell_gi)))
    cell_omega = snap.get('cell_omega_local', np.zeros(len(cell_gi)))

    bp = superellipsoid_point(
        cell_eta[ci], cell_omega[ci],
        a_arr[gi], b_arr[gi], c_arr[gi], n1_arr[gi], n2_arr[gi])

    if 'quat' in snap:
        bp = quat_rotate(snap['quat'][gi], bp)

    return float(xs[gi]) + bp[0], float(ys[gi]) + bp[1], float(zs[gi]) + bp[2]


def _surface_normal_at_cell(snap, ci):
    """Get granule surface normal at cell position."""
    cell_gi = snap['cell_granule_id']
    gi = int(cell_gi[ci])
    if gi < 0 or gi >= len(snap['x']):
        return 0.0, 0.0, 1.0

    a_arr, b_arr = snap['a'], snap['b']
    c_arr = snap.get('c', a_arr.copy())
    n1_arr = snap['n1']
    n2_arr = snap.get('n2', n1_arr.copy())

    cell_eta = snap.get('cell_eta_local', np.zeros(len(cell_gi)))
    cell_omega = snap.get('cell_omega_local', np.zeros(len(cell_gi)))

    n_body = superellipsoid_normal(
        cell_eta[ci], cell_omega[ci],
        a_arr[gi], b_arr[gi], c_arr[gi], n1_arr[gi], n2_arr[gi])

    if 'quat' in snap:
        n_world = quat_rotate(snap['quat'][gi], n_body)
    else:
        n_world = n_body

    return float(n_world[0]), float(n_world[1]), float(n_world[2])


def _rotation_align_x(direction):
    """3x3 rotation matrix that aligns the x-axis to direction."""
    d = direction / np.linalg.norm(direction)
    # Find orthonormal basis
    if abs(d[0]) < 0.9:
        up = np.array([1, 0, 0.])
    else:
        up = np.array([0, 1, 0.])
    v = np.cross(d, up)
    v /= np.linalg.norm(v)
    w = np.cross(d, v)
    return np.column_stack([d, v, w])


def _rotation_align_z(normal):
    """3x3 rotation matrix that aligns the z-axis to normal."""
    n = normal / (np.linalg.norm(normal) + 1e-30)
    if abs(n[2]) < 0.9:
        up = np.array([0, 0, 1.])
    else:
        up = np.array([1, 0, 0.])
    u = np.cross(up, n)
    u_mag = np.linalg.norm(u)
    if u_mag < 1e-12:
        return np.eye(3)
    u /= u_mag
    v = np.cross(n, u)
    return np.column_stack([u, v, n])


def _add_cell_meshes(plotter, snap, group, p):
    """Add cell ellipsoid meshes to plotter for cells on given granule group."""
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    cell_states = snap.get('cell_state', np.array([], dtype=int))
    group_set = set(group)

    COLORS = {
        int(CellState.ATTACHED):     '#DAA520',
        int(CellState.SPREADING):    '#FF8C00',
        int(CellState.PROLIFERATING):'#228B22',
        int(CellState.BRIDGING):     '#DC143C',
        int(CellState.SENESCENT):    '#696969',
    }

    for ci in range(len(cell_gi)):
        if int(cell_gi[ci]) not in group_set:
            continue
        mesh = cell_ellipsoid_mesh(snap, ci, p)
        if mesh is not None and mesh.n_points > 0:
            color = COLORS.get(int(cell_states[ci]), '#AAAAAA')
            plotter.add_mesh(mesh, color=color, opacity=0.85)


# ── 3D Granule Rendering (granules_3d.png) ───────────────────────────

def plot_granules_3d(snaps, hist, p, indices=None, outdir=None):
    """Phase field isosurface rendering — continuous solid with z-slice opacity.

    Uses phi_f and phi_i phase fields to create merged isosurfaces where
    overlapping granules form a continuous solid. Mid-slice granules are
    opaque; the rest are rendered as ghosts (alpha=0.1).
    """
    if not HAS_PYVISTA:
        print("  viz_stress: skipping 3D granules (pyvista not available)")
        return None

    n = len(snaps)
    if indices is None:
        indices = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
    nc = len(indices)

    fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 5))
    if nc == 1:
        axes = [axes]

    for c_idx, si in enumerate(indices):
        snap = snaps[si]
        phi_f = snap.get('phi_f', None)
        phi_i = snap.get('phi_i', None)

        if phi_f is None or phi_f.ndim != 3:
            # No 3D fields — fall back to individual meshes
            _plot_granules_3d_meshes(snap, hist, p, si, axes[c_idx])
            continue

        R_mean = _func_r_mean(snap)
        Ng = phi_f.shape[0]
        sp = (p.Lx / Ng, p.Ly / Ng, p.Lz / Ng)
        z_mid = p.Lz / 2.0
        z_lo = z_mid - R_mean
        z_hi = z_mid + R_mean

        plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
        plotter.set_background('white')

        iso_level = 0.3  # isosurface threshold

        # Functional phase — continuous solid
        grid_f = pv.ImageData(dimensions=phi_f.shape, spacing=sp)
        grid_f.point_data['phi'] = phi_f.ravel(order='F')
        contour_f = grid_f.contour([iso_level], scalars='phi')
        if contour_f.n_points > 0:
            # Clip to z-slab for opaque mid-slice
            slab_f = contour_f.clip(normal='z', origin=(0, 0, z_hi), invert=True)
            if slab_f.n_points > 0:
                slab_f = slab_f.clip(normal='-z', origin=(0, 0, z_lo), invert=True)
            if slab_f.n_points > 0:
                plotter.add_mesh(slab_f, color='orangered', opacity=1.0)
            # Rest as ghost
            ghost_f_lo = contour_f.clip(normal='z', origin=(0, 0, z_lo))
            ghost_f_hi = contour_f.clip(normal='-z', origin=(0, 0, z_hi))
            if ghost_f_lo.n_points > 0:
                plotter.add_mesh(ghost_f_lo, color='orangered', opacity=0.1)
            if ghost_f_hi.n_points > 0:
                plotter.add_mesh(ghost_f_hi, color='orangered', opacity=0.1)

        # Inert phase — continuous solid
        if phi_i is not None:
            grid_i = pv.ImageData(dimensions=phi_i.shape, spacing=sp)
            grid_i.point_data['phi'] = phi_i.ravel(order='F')
            contour_i = grid_i.contour([iso_level], scalars='phi')
            if contour_i.n_points > 0:
                slab_i = contour_i.clip(normal='z', origin=(0, 0, z_hi), invert=True)
                if slab_i.n_points > 0:
                    slab_i = slab_i.clip(normal='-z', origin=(0, 0, z_lo), invert=True)
                if slab_i.n_points > 0:
                    plotter.add_mesh(slab_i, color='steelblue', opacity=1.0)
                ghost_i_lo = contour_i.clip(normal='z', origin=(0, 0, z_lo))
                ghost_i_hi = contour_i.clip(normal='-z', origin=(0, 0, z_hi))
                if ghost_i_lo.n_points > 0:
                    plotter.add_mesh(ghost_i_lo, color='steelblue', opacity=0.1)
                if ghost_i_hi.n_points > 0:
                    plotter.add_mesh(ghost_i_hi, color='steelblue', opacity=0.1)

        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.2)

        img = plotter.screenshot(return_img=True)
        plotter.close()

        ax = axes[c_idx]
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f't = {t:.1f} h', fontsize=10)

    fig.suptitle('3D Granule Scaffold (mid-slice opaque)', fontsize=13, y=1.02)
    plt.tight_layout()

    if outdir:
        path = str(Path(outdir) / 'granules_3d.png')
        fig.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  viz_stress: saved {path}")
    return fig


def _plot_granules_3d_meshes(snap, hist, p, si, ax):
    """Fallback: render individual parametric meshes when phase fields unavailable."""
    R_mean = _func_r_mean(snap)
    z_mid = p.Lz / 2.0
    z_lo = z_mid - R_mean
    z_hi = z_mid + R_mean
    zs = snap.get('z', np.zeros(len(snap['x'])))
    gt = snap['gtype']

    plotter = pv.Plotter(off_screen=True, window_size=(800, 800))
    plotter.set_background('white')

    for gi in range(len(snap['x'])):
        mesh, _, _ = granule_surface_mesh(snap, gi, n_pts=24)
        if mesh is None:
            continue
        gz = float(zs[gi])
        in_slab = z_lo <= gz <= z_hi
        color = 'orangered' if gt[gi] == 0 else 'steelblue'
        plotter.add_mesh(mesh, color=color, opacity=1.0 if in_slab else 0.1)

    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    img = plotter.screenshot(return_img=True)
    plotter.close()

    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    t = hist[si]['time'] if si < len(hist) else 0
    ax.set_title(f't = {t:.1f} h', fontsize=10)


# ── Public API ───────────────────────────────────────────────────────

def run_all(snaps, hist, p, outdir=None):
    """Generate all 3D stress and granule visualizations."""
    if not HAS_PYVISTA:
        print("  viz_stress: pyvista not available, skipping all")
        return

    # Check if 3D mode
    mode = getattr(p, 'mode', '2D')
    if mode != '3D':
        snap0 = snaps[0] if snaps else {}
        if 'quat' not in snap0:
            print("  viz_stress: 2D mode detected, skipping 3D-only visualizations")
            return

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    print("  viz_stress: stress cross-section...")
    fig1 = plot_stress_cross_section(snaps, hist, p, outdir=outdir)
    if fig1:
        plt.close(fig1)

    print("  viz_stress: selecting interesting granules...")
    selected = select_interesting_granules(snaps, hist, p, n=3)
    print(f"  viz_stress: selected granules {selected}")

    for gi in selected:
        print(f"  viz_stress: granule evolution GIF for granule {gi}...")
        granule_evolution_gif(snaps, hist, p, gi, outdir=outdir)

    print("  viz_stress: 3D granule rendering...")
    fig2 = plot_granules_3d(snaps, hist, p, outdir=outdir)
    if fig2:
        plt.close(fig2)


# ── CLI ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='3D surface stress and granule visualization')
    parser.add_argument('-i', '--input', required=True,
                        help='Simulation output directory')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: <input>/visualizations)')
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    try:
        from new_dem_0 import load_run, Params
        hist, snaps, p, _ = load_run(str(indir))
        run_all(snaps, hist, p, outdir=outdir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
