#!/usr/bin/env python3
"""
3-D Scaffold Evolution Visualization (V1.9)
=============================================
Volumetric PyVista renderings showing time evolution of granular scaffold
microstructure under cell-driven compaction predicted by the mean-field model.

For 4 representative organ targets, renders superellipsoid granules in a 3-D
domain, clipped at the midplane so internal structure is visible.  Functional
granules (green) cluster over time while inert granules (tan) redistribute.

Usage:
    python viz/scaffold_evolution_3d.py -i results/parameter_sweep
    python viz/scaffold_evolution_3d.py -i results/parameter_sweep -o results/viz
"""

import sys, os, json, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyvista as pv
pv.OFF_SCREEN = True

from viz.scaffold_evolution import _solve_trajectory   # reuse ODE solver

# ======================================================================
# Configuration
# ======================================================================

ORGANS_TO_SHOW = [
    'trabecular_bone',
    'intestinal_mucosa',
    'kidney_cortex',
    'cardiac_muscle',
]
ORGAN_LABELS = {
    'trabecular_bone':   'Trabecular\nBone',
    'intestinal_mucosa': 'Intestinal\nMucosa',
    'kidney_cortex':     'Kidney\nCortex',
    'cardiac_muscle':    'Cardiac\nMuscle',
    'liver':             'Liver',
    'lung_alveoli':      'Lung\nAlveoli',
    'pancreatic_islet':  'Pancreatic\nIslet',
}
TIME_SNAPSHOTS = [0, 6, 18, 36, 72]  # hours

# Colours (RGB 0-1 for PyVista)
C_FUNC_PV   = (0.40, 0.73, 0.42)   # #66BB6A
C_INERT_PV  = (0.84, 0.80, 0.78)   # #D7CCC8
C_CELL_PV   = (0.78, 0.16, 0.16)   # #C62828
C_BRIDGE_PV = (0.90, 0.56, 0.56)   # #E57373
C_BG        = (0.98, 0.98, 0.98)   # near-white

# matplotlib equivalents
C_FUNC_MPL  = '#66BB6A'
C_INERT_MPL = '#D7CCC8'
C_CELL_MPL  = '#C62828'

TARGET_N_GRANULES = 50   # 3-D needs fewer for clarity
RENDER_SIZE = (600, 600) # pixels per panel


# ======================================================================
# 3-D packing generation (RSA)
# ======================================================================

def _generate_packing_3d(R_func, R_inert, phi_f, phi_i,
                         target_N=50, seed=42):
    """RSA in a cubic domain.  Returns (positions[N,3], radii[N], types[N], L)."""
    rng = np.random.default_rng(seed)
    V_f = (4.0 / 3.0) * np.pi * R_func ** 3
    V_i = (4.0 / 3.0) * np.pi * R_inert ** 3

    if phi_i < 0.005:
        N_func, N_inert = target_N, 0
    elif phi_f < 0.005:
        N_func, N_inert = 0, target_N
    else:
        N_func = max(1, int(round(
            target_N * V_i * phi_f / (V_f * phi_i + V_i * phi_f))))
        N_inert = max(1, target_N - N_func)

    total_vol = N_func * V_f + N_inert * V_i
    phi_solid = max(phi_f + phi_i, 0.05)
    L = (total_vol / phi_solid) ** (1.0 / 3.0)

    to_place = [(R_func, 0)] * N_func + [(R_inert, 1)] * N_inert
    rng.shuffle(to_place)

    positions, radii, types = [], [], []
    for r, gtype in to_place:
        placed = False
        for _ in range(8000):
            xyz = rng.uniform(r + 0.5, L - r - 0.5, size=3)
            ok = True
            for pos_i, r_i in zip(positions, radii):
                if np.linalg.norm(xyz - pos_i) < r + r_i + 0.5:
                    ok = False
                    break
            if ok:
                positions.append(xyz)
                radii.append(r)
                types.append(gtype)
                placed = True
                break
        if not placed:
            positions.append(rng.uniform(r + 0.5, L - r - 0.5, size=3))
            radii.append(r)
            types.append(gtype)

    return (np.array(positions), np.array(radii),
            np.array(types, dtype=int), float(L))


def _compact_positions_3d(pos, radii, types, x_f_0, x_f_t, L):
    """Move functional granules toward COM; resolve overlaps in 3-D."""
    new = pos.copy()
    fmask = types == 0

    if not np.any(fmask) or x_f_t >= x_f_0 * 0.999:
        return new

    # Volume scale (cube-root for 3-D)
    scale = max(x_f_t / max(x_f_0, 1e-6), 0.02) ** (1.0 / 3.0)
    com = np.mean(pos[fmask], axis=0)

    for i in np.where(fmask)[0]:
        new[i] = com + scale * (pos[i] - com)

    # Overlap resolution
    for _ in range(300):
        max_ov = 0.0
        for i in range(len(new)):
            for j in range(i + 1, len(new)):
                dx = new[j] - new[i]
                d = np.linalg.norm(dx)
                mind = radii[i] + radii[j]
                if d < mind and d > 0.01:
                    ov = mind - d
                    max_ov = max(max_ov, ov)
                    n_hat = dx / d
                    if types[i] == 0 and types[j] == 0:
                        push = 0.25 * ov
                    else:
                        push = 0.55 * ov
                    new[i] -= push * n_hat
                    new[j] += push * n_hat

        for i in range(len(new)):
            r = radii[i]
            new[i] = np.clip(new[i], r, L - r)

        if max_ov < 0.5:
            break

    return new


# ======================================================================
# Superellipsoid mesh creation
# ======================================================================

def _make_superellipsoid(cx, cy, cz, R, ar, n2, theta_z=0.0):
    """Create a positioned + rotated superellipsoid mesh.

    Parameters
    ----------
    cx, cy, cz : float   Centre.
    R : float             Bounding-sphere radius.
    ar : float            Aspect ratio (a/b), >= 1.
    n2 : float            Blockiness exponent (our convention: 2 = sphere).
    theta_z : float       Rotation about z-axis (radians).
    """
    # Semi-axes (conserve volume: a*b*c ~ R^3)
    a = R * ar ** (1.0 / 3.0)
    b = R / ar ** (1.0 / 6.0)
    c = R / ar ** (1.0 / 6.0)

    # Map our blockiness to VTK convention:
    # VTK: n1=n2=1 → sphere;  smaller → blockier
    vtk_n = 2.0 / max(n2, 0.5)
    vtk_n = np.clip(vtk_n, 0.05, 2.0)

    mesh = pv.ParametricSuperEllipsoid(
        xradius=a, yradius=b, zradius=c,
        n1=vtk_n, n2=vtk_n)

    # Rotate about z
    mesh.rotate_z(np.degrees(theta_z), inplace=True)
    # Also a slight random tilt for visual variety
    mesh.rotate_x(np.degrees(theta_z * 0.3), inplace=True)

    mesh.translate([cx, cy, cz], inplace=True)
    return mesh


# ======================================================================
# Off-screen rendering of one snapshot
# ======================================================================

def _render_snapshot(positions, radii, types, params,
                     x_f, x_f_0, L, f_bridge=0.0,
                     window_size=RENDER_SIZE):
    """Render a cut-away view: remove top half, camera from upper-right.

    Single clip at z = L*0.55 keeps the bottom half plus a thin slice above
    the midplane.  Fully opaque granules with bold colours so the cross-
    section reads clearly: green = functional, tan = inert, white = void.
    """
    ar_f = params['aspect_ratio_func']
    ar_i = params['aspect_ratio_inert']
    n2_f = params['blockiness_n2_func']
    n2_i = params['blockiness_n2_inert']

    orient_rng = np.random.default_rng(314)
    thetas = orient_rng.uniform(0, 2 * np.pi, size=len(positions))

    z_cut = L * 0.55                       # clip plane

    pl = pv.Plotter(off_screen=True, window_size=window_size,
                     lighting='three lights')
    pl.set_background('white')

    # Floor plane at z = 0 for grounding
    floor = pv.Plane(center=(L / 2, L / 2, -0.1),
                     direction=(0, 0, 1),
                     i_size=L * 1.05, j_size=L * 1.05)
    pl.add_mesh(floor, color=(0.95, 0.95, 0.95), opacity=0.4)

    # Wireframe box of the kept volume
    kept_box = pv.Box(bounds=[0, L, 0, L, 0, z_cut])
    pl.add_mesh(kept_box, style='wireframe', color='#9E9E9E',
                line_width=0.8, opacity=0.35)

    func_idx = np.where(types == 0)[0]

    # --- Granules (fully opaque, clipped at z_cut) ---
    for i in range(len(positions)):
        cx, cy, cz = positions[i]
        r = radii[i]
        if cz - r > z_cut:                # entirely above cut
            continue

        if types[i] == 0:
            ar, n2 = ar_f, n2_f
            face_color = (0.26, 0.63, 0.28)    # saturated green
            edge_color = (0.15, 0.40, 0.16)
        else:
            ar, n2 = ar_i, n2_i
            face_color = (0.76, 0.60, 0.50)    # warm tan
            edge_color = (0.50, 0.38, 0.30)

        mesh = _make_superellipsoid(cx, cy, cz, r, ar, n2, thetas[i])
        if cz + r > z_cut:                # crosses the cut plane
            try:
                mesh = mesh.clip(normal=(0, 0, -1),
                                 origin=(0, 0, z_cut))
            except Exception:
                continue
        if mesh.n_points < 3:
            continue

        pl.add_mesh(mesh, color=face_color, opacity=1.0,
                    smooth_shading=True, specular=0.25,
                    diffuse=0.85, ambient=0.15)
        # Bold edge outline on the cut face
        try:
            edges = mesh.extract_feature_edges(
                boundary_edges=True, feature_edges=False,
                manifold_edges=False)
            if edges.n_points > 0:
                pl.add_mesh(edges, color=edge_color,
                            line_width=1.5, opacity=0.8)
        except Exception:
            pass

    # --- Cell bridges ---
    sense = params.get('cell_sense_distance', 50.0)
    if f_bridge > 0.05 and len(func_idx) > 1:
        br_rng = np.random.default_rng(789)
        tube_r = max(0.6, L * 0.004)
        for ii in range(len(func_idx)):
            i = func_idx[ii]
            if positions[i, 2] - radii[i] > z_cut:
                continue
            for jj in range(ii + 1, len(func_idx)):
                j = func_idx[jj]
                if positions[j, 2] - radii[j] > z_cut:
                    continue
                d = np.linalg.norm(positions[i] - positions[j])
                gap = d - radii[i] - radii[j]
                if gap < sense and br_rng.random() < f_bridge:
                    tube = pv.Line(positions[i], positions[j]).tube(
                        radius=tube_r)
                    try:
                        tube = tube.clip(normal=(0, 0, -1),
                                         origin=(0, 0, z_cut))
                    except Exception:
                        continue
                    if tube.n_points > 0:
                        pl.add_mesh(tube, color=C_BRIDGE_PV, opacity=0.7)

    # --- Cells (small red spheres, only below cut) ---
    n_cells_show = min(int(params['n_cells_per_func']), 3)
    if n_cells_show > 0:
        cell_rng = np.random.default_rng(456)
        cell_r = max(1.8, L * 0.014)
        for i in func_idx:
            cx, cy, cz = positions[i]
            r = radii[i]
            if cz - r > z_cut:
                cell_rng.random(n_cells_show * 3)
                continue
            for _ in range(n_cells_show):
                phi_a = cell_rng.uniform(0, 2 * np.pi)
                cos_t = cell_rng.uniform(-1, 1)
                sin_t = np.sqrt(1 - cos_t ** 2)
                rr = r * 0.85
                cp = np.array([cx + rr * sin_t * np.cos(phi_a),
                               cy + rr * sin_t * np.sin(phi_a),
                               cz + rr * cos_t])
                if cp[2] > z_cut:
                    continue
                pl.add_mesh(pv.Sphere(radius=cell_r, center=cp),
                            color=C_CELL_PV, opacity=1.0)

    # Camera: 55° elevation from upper-right, looking at domain centre
    cam_dist = L * 2.2
    ctr = [L / 2, L / 2, z_cut * 0.4]
    cam_pos = [L / 2 + cam_dist * 0.45,
               L / 2 - cam_dist * 0.45,
               ctr[2] + cam_dist * 0.72]
    pl.camera_position = [cam_pos, ctr, (0, 0, 1)]

    pl.enable_anti_aliasing('ssaa')

    img = pl.screenshot(return_img=True)
    pl.close()
    return img


# ======================================================================
# Main figure: composite 3-D renders into matplotlib
# ======================================================================

def make_evolution_figure_3d(recommendations, outdir,
                             t_total=72.0, seed=42):
    """Create multi-panel 3-D scaffold evolution figure.

    Returns list of saved file paths.
    """
    organs = [o for o in ORGANS_TO_SHOW if o in recommendations]
    if not organs:
        print("  Warning: no matching organs in recommendations")
        return []

    n_rows = len(organs)
    n_cols = len(TIME_SNAPSHOTS)

    fig = plt.figure(figsize=(n_cols * 3.2 + 1.0,
                              n_rows * 3.2 + 1.6))
    gs = GridSpec(n_rows + 1, n_cols,
                  height_ratios=[1] * n_rows + [0.12],
                  hspace=0.15, wspace=0.05,
                  left=0.07, right=0.96, top=0.93, bottom=0.04)

    saved = []

    for row, organ in enumerate(organs):
        rec = recommendations[organ][0]
        p = rec['params']
        label = ORGAN_LABELS.get(organ, organ.replace('_', '\n').title())
        D_arch = rec['D_arch']

        print(f"  {label.replace(chr(10), ' ')} (D={D_arch:.2f})...")

        # Solve trajectory (reuse 2-D ODE — physics is dim-agnostic)
        t_arr, x_f_arr, fb_arr = _solve_trajectory(p, t_total=t_total)
        x_f_0 = x_f_arr[0]

        # Generate initial 3-D packing
        pos0, radii, gtypes, L = _generate_packing_3d(
            p['R_func'], p['R_inert'], p['phi_f'], p['phi_i'],
            target_N=TARGET_N_GRANULES, seed=seed + row * 7)

        for col, t_snap in enumerate(TIME_SNAPSHOTS):
            ax = fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])

            # Interpolate x_f and f_bridge
            idx = min(np.searchsorted(t_arr, t_snap), len(x_f_arr) - 1)
            x_f = x_f_arr[idx]
            fb = fb_arr[idx]

            # Compact positions in 3-D
            pos = _compact_positions_3d(pos0, radii, gtypes, x_f_0, x_f, L)

            # Render
            img = _render_snapshot(pos, radii, gtypes, p,
                                   x_f, x_f_0, L, f_bridge=fb)
            ax.imshow(img)

            # Column header
            if row == 0:
                ax.set_title(f't = {t_snap} h', fontsize=10,
                             fontweight='bold', pad=4)

            # Row label
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight='bold',
                              labelpad=8, linespacing=1.3)

            # Compaction metric
            compact_pct = max(0, (x_f_0 - x_f) / x_f_0 * 100)
            phi_f = p['phi_f']
            phi_i = p['phi_i']
            phi_v_f = max(0, 1 - phi_f / max(x_f, 1e-6))
            phi_v_i = max(0, 1 - phi_i / max(1 - x_f, 1e-6))
            info = (f"$x_f$={x_f:.2f}  "
                    f"$\\phi_{{v,f}}$={phi_v_f:.2f}  "
                    f"$\\phi_{{v,i}}$={phi_v_i:.2f}")
            ax.text(0.5, -0.02, info, transform=ax.transAxes, fontsize=5,
                    ha='center', va='top', color='#616161',
                    family='monospace')

            # D_arch on rightmost column
            if col == n_cols - 1:
                ax.text(1.03, 0.5, f'$D_{{arch}}$\n{D_arch:.1f}',
                        transform=ax.transAxes, fontsize=7,
                        ha='left', va='center', color='#757575',
                        linespacing=1.4)

            for spine in ax.spines.values():
                spine.set_visible(False)

    # Legend
    ax_leg = fig.add_subplot(gs[n_rows, :])
    ax_leg.axis('off')
    legend_elements = [
        Patch(facecolor=C_FUNC_MPL, edgecolor='#388E3C', lw=0.8,
              label='Functional granule'),
        Patch(facecolor=C_INERT_MPL, edgecolor='#8D6E63', lw=0.8,
              label='Inert granule'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C_CELL_MPL, markersize=5,
               markeredgewidth=0, label='Cell'),
        Patch(facecolor='white', edgecolor='gray', lw=0.5,
              label='Void space'),
    ]
    ax_leg.legend(handles=legend_elements, loc='center', ncol=4,
                  fontsize=8.5, frameon=False, handlelength=1.8,
                  columnspacing=2.5)

    fig.suptitle(
        '3-D Scaffold Microstructure Evolution Under Cell-Driven Compaction\n'
        '(z-midplane cross-section)',
        fontsize=13, fontweight='bold')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'scaffold_evolution_3d.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    saved.append(path)
    return saved


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='3-D scaffold evolution visualization')
    parser.add_argument('-i', '--input', default='results/parameter_sweep',
                        help='Directory containing recommendations.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t-total', type=float, default=72.0)
    args = parser.parse_args()

    indir = args.input
    outdir = args.output or indir

    rec_path = os.path.join(indir, 'recommendations.json')
    if not os.path.isfile(rec_path):
        print(f"Error: {rec_path} not found. Run parameter_sweep.py first.")
        sys.exit(1)

    with open(rec_path) as f:
        recommendations = json.load(f)

    print("=" * 60)
    print("  3-D Scaffold Evolution Visualization")
    print("=" * 60)

    make_evolution_figure_3d(recommendations, outdir,
                             t_total=args.t_total, seed=args.seed)

    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
