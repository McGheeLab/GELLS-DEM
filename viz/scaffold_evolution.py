#!/usr/bin/env python3
"""
Scaffold Evolution Visualization (V1.9)
========================================
2D spatial maps showing time evolution of granular scaffold microstructure
under cell-driven compaction predicted by the mean-field model.

For 4 representative organ targets, generates a panel figure showing:
- Functional granules (green) clustering over time as cells compact the scaffold
- Inert granules (tan) redistributing as functional zone shrinks
- Cell bodies (pink dots) on functional granule surfaces
- Cell bridges (pink lines) forming between nearby functional granules
- Void space (white) expelled from functional zone to inert zone

Usage:
    python viz/scaffold_evolution.py -i results/parameter_sweep
    python viz/scaffold_evolution.py -i results/parameter_sweep -o results/viz
"""

import sys, os, json, argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch, FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.parameter_sweep import (
    compute_phi_RCP, compute_phi_max_deformable,
    motor_clutch_force_vec, effective_contact_modulus,
    bridge_rate_modifier,
)

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
    'trabecular_bone':    'Trabecular\nBone',
    'intestinal_mucosa':  'Intestinal\nMucosa',
    'kidney_cortex':      'Kidney\nCortex',
    'cardiac_muscle':     'Cardiac\nMuscle',
    'liver':              'Liver',
    'lung_alveoli':       'Lung\nAlveoli',
    'pancreatic_islet':   'Pancreatic\nIslet',
}
TIME_SNAPSHOTS = [0, 6, 18, 36, 72]  # hours

# Colours
C_FUNC       = '#388E3C'   # dark green
C_FUNC_FILL  = '#66BB6A'   # lighter green fill
C_INERT      = '#8D6E63'   # brown edge
C_INERT_FILL = '#D7CCC8'   # light tan fill
C_CELL       = '#C62828'   # deep red
C_BRIDGE     = '#E57373'   # soft red
C_DOMAIN_BG  = '#FAFAFA'   # very light gray

TARGET_N_GRANULES = 70      # total granules per panel


# ======================================================================
# Superellipse geometry
# ======================================================================

def superellipse_vertices(cx, cy, a, b, n, theta=0.0, n_pts=48):
    """Parametric superellipse |x/a|^n + |y/b|^n = 1, rotated by theta."""
    t = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    e = 2.0 / max(n, 0.5)
    x = a * np.sign(np.cos(t)) * np.abs(np.cos(t)) ** e
    y = b * np.sign(np.sin(t)) * np.abs(np.sin(t)) ** e
    ct, st = np.cos(theta), np.sin(theta)
    xr = ct * x - st * y + cx
    yr = st * x + ct * y + cy
    return np.column_stack([xr, yr])


# ======================================================================
# 2-D packing generation (RSA)
# ======================================================================

def _generate_packing(R_func, R_inert, phi_f, phi_i,
                      target_N=70, seed=42):
    """Random sequential addition of functional + inert granules.

    Returns (positions [N,2], radii [N], types [N], domain_size).
    """
    rng = np.random.default_rng(seed)
    A_f = np.pi * R_func ** 2
    A_i = np.pi * R_inert ** 2

    # Decide how many of each type
    if phi_i < 0.005:
        N_func, N_inert = target_N, 0
    elif phi_f < 0.005:
        N_func, N_inert = 0, target_N
    else:
        N_func = max(1, int(round(
            target_N * A_i * phi_f / (A_f * phi_i + A_i * phi_f))))
        N_inert = max(1, target_N - N_func)

    # Domain size so that total packing fraction matches
    total_solid_area = N_func * A_f + N_inert * A_i
    phi_solid = phi_f + phi_i
    L = np.sqrt(total_solid_area / max(phi_solid, 0.05))

    # Interleave placement order
    to_place = [(R_func, 0)] * N_func + [(R_inert, 1)] * N_inert
    rng.shuffle(to_place)

    positions, radii, types = [], [], []
    for r, gtype in to_place:
        placed = False
        for _ in range(6000):
            x = rng.uniform(r + 0.5, L - r - 0.5)
            y = rng.uniform(r + 0.5, L - r - 0.5)
            ok = True
            for (px, py), pr in zip(positions, radii):
                if np.hypot(x - px, y - py) < r + pr + 0.8:
                    ok = False
                    break
            if ok:
                positions.append((x, y))
                radii.append(r)
                types.append(gtype)
                placed = True
                break
        if not placed:
            # Force-place (will be resolved later)
            x = rng.uniform(r + 0.5, L - r - 0.5)
            y = rng.uniform(r + 0.5, L - r - 0.5)
            positions.append((x, y))
            radii.append(r)
            types.append(gtype)

    return (np.array(positions), np.array(radii),
            np.array(types, dtype=int), float(L))


def _compact_positions(pos, radii, types, x_f_0, x_f_t, L,
                       max_iters=200):
    """Displace functional granules toward their COM to represent compaction.

    Uses scipy.spatial.cKDTree for O(N log N) neighbor search so that
    large packings (300+ granules) remain tractable for animation.
    """
    from scipy.spatial import cKDTree

    new = pos.copy()
    fmask = types == 0

    if not np.any(fmask) or x_f_t >= x_f_0 * 0.999:
        return new

    # Area scale factor (2-D)
    scale = np.sqrt(max(x_f_t / max(x_f_0, 1e-6), 0.02))
    com = np.mean(pos[fmask], axis=0)

    fi = np.where(fmask)[0]
    new[fi] = com + scale * (pos[fi] - com)

    # Overlap resolution with spatial index
    r_max = float(np.max(radii))
    cutoff = 2.0 * r_max + 1.0
    is_func = (types == 0)

    for it in range(max_iters):
        # Rebuild tree every few iterations
        if it % 4 == 0:
            tree = cKDTree(new)
            pairs = list(tree.query_pairs(cutoff))

        max_ov = 0.0
        for i, j in pairs:
            dx = new[j] - new[i]
            d = np.linalg.norm(dx)
            mind = radii[i] + radii[j]
            if d < mind and d > 0.01:
                ov = mind - d
                if ov > max_ov:
                    max_ov = ov
                n_hat = dx / d
                push = 0.25 * ov if (is_func[i] and is_func[j]) else 0.55 * ov
                new[i] -= push * n_hat
                new[j] += push * n_hat

        # Soft wall clamp
        for i in range(len(new)):
            r = radii[i]
            new[i, 0] = np.clip(new[i, 0], r, L - r)
            new[i, 1] = np.clip(new[i, 1], r, L - r)

        if max_ov < 0.5:
            break

    return new


# ======================================================================
# Scalar mean-field ODE solver
# ======================================================================

def _solve_trajectory(params, t_total=72.0, dt=0.1):
    """Integrate the two-zone compaction ODE for one parameter set.

    Returns (t_array, x_f_array, f_bridge_array).
    """
    phi_f = params['phi_f']
    phi_i = params['phi_i']
    phi_solid = phi_f + phi_i
    if phi_solid < 0.01:
        t_a = np.arange(0, t_total + dt, dt)
        return t_a, np.full_like(t_a, 0.5), np.zeros_like(t_a)

    _a = lambda v: np.array([v])
    R_f = params['R_func']

    phi_RCP = compute_phi_RCP(
        _a(params['aspect_ratio_func']), _a(params['aspect_ratio_inert']),
        _a(params['blockiness_n2_func']), _a(params['blockiness_n2_inert']),
        _a(phi_f), _a(phi_i), _a(R_f), _a(params['R_inert']),
    )[0]
    phi_max = compute_phi_max_deformable(
        _a(phi_RCP), _a(params['E_func']), _a(params['blockiness_n2_func'])
    )[0]
    F_cell = motor_clutch_force_vec(_a(params['E_func']))[0]
    E_eff = effective_contact_modulus(
        _a(params['E_func']), _a(params['E_inert']),
        _a(phi_f), _a(phi_i),
    )[0]
    sense = params.get('cell_sense_distance', 50.0)
    br_mod = bridge_rate_modifier(
        _a(sense), _a(R_f), _a(phi_f), _a(phi_i), _a(phi_RCP)
    )[0]

    # Derived constants
    domain_area = 800.0 * 800.0
    N_func = max(1.0, phi_f * domain_area / (np.pi * R_f ** 2))
    n_cells = float(params['n_cells_per_func'])
    n_cell_density = N_func * n_cells / domain_area

    eta_eff = np.clip(0.5 * (E_eff / 5.0) ** 0.3 * (R_f / 40.0) ** 1.5,
                      0.01, 100.0)
    sigma_0 = max(0.005 * E_eff, 1e-6)

    t_spread = 3.0
    fa_rate = 0.3
    base_br_rate = 0.3
    bridge_form_t = 2.0
    t_br_start = t_spread + 1.0

    x_f = max(phi_f / phi_solid, phi_f / max(phi_max, 0.01))
    x_f_min = phi_f / max(phi_max, 0.01)

    n_steps = int(t_total / dt)
    t_arr = np.zeros(n_steps + 1)
    x_f_arr = np.zeros(n_steps + 1)
    fb_arr = np.zeros(n_steps + 1)
    x_f_arr[0] = x_f

    for step in range(1, n_steps + 1):
        t = (step - 1) * dt
        t_eff = max(0.0, t - t_spread)
        maturity = 1.0 - np.exp(-fa_rate * t_eff)
        t_br_eff = max(0.0, t - t_spread - 1.0)

        if maturity < 0.1:
            f_bridge = 0.0
        else:
            f_bridge = 1.0 - np.exp(-base_br_rate * br_mod * t_br_eff * maturity)

        t_since = max(0.0, t - t_br_start)
        ramp = min(1.0, t_since / bridge_form_t) if bridge_form_t > 0 else 1.0

        s_cell = n_cell_density * F_cell * f_bridge * maturity * ramp
        phi_f_local = phi_f / max(x_f, 1e-6)
        s_resist = sigma_0 * max(0.0, phi_f_local / phi_RCP - 1.0)

        net = max(0.0, s_cell - s_resist)
        x_f += dt * (-x_f * net / eta_eff)
        x_f = np.clip(x_f, x_f_min, 1.0 - phi_i)

        t_arr[step] = step * dt
        x_f_arr[step] = x_f
        fb_arr[step] = f_bridge

    return t_arr, x_f_arr, fb_arr


# ======================================================================
# Drawing helpers
# ======================================================================

def _draw_snapshot(ax, positions, radii, types, params,
                   x_f, x_f_0, t_hr, L, f_bridge=0.0,
                   viewport=None):
    """Render one 2-D microstructure snapshot on the given axes.

    Parameters
    ----------
    viewport : tuple or None
        (x0, y0, x1, y1) sub-region to render.  If None, uses full domain.
    """
    phi_f = params['phi_f']
    phi_i = params['phi_i']

    if viewport is None:
        vx0, vy0, vx1, vy1 = 0.0, 0.0, float(L), float(L)
    else:
        vx0, vy0, vx1, vy1 = viewport

    ax.set_facecolor(C_DOMAIN_BG)
    ax.set_xlim(vx0, vx1)
    ax.set_ylim(vy0, vy1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDBDBD')
        spine.set_linewidth(0.8)

    ar_f = params['aspect_ratio_func']
    ar_i = params['aspect_ratio_inert']
    n2_f = params['blockiness_n2_func']
    n2_i = params['blockiness_n2_inert']

    # Deterministic orientations (same across time snapshots for continuity)
    orient_rng = np.random.default_rng(314)
    thetas = orient_rng.uniform(0, 2 * np.pi, size=len(positions))

    # --- Draw inert granules first (behind functional) ---
    for i in np.where(types == 1)[0]:
        cx, cy = positions[i]
        r = radii[i]
        a = r * np.sqrt(ar_i)
        b = r / np.sqrt(ar_i)
        verts = superellipse_vertices(cx, cy, a, b, n2_i, thetas[i])
        poly = MplPolygon(verts, closed=True, zorder=2)
        poly.set_facecolor(C_INERT_FILL)
        poly.set_edgecolor(C_INERT)
        poly.set_linewidth(0.6)
        ax.add_patch(poly)

    # --- Draw cell bridges between nearby functional granules ---
    func_idx = np.where(types == 0)[0]
    if f_bridge > 0.05 and len(func_idx) > 1:
        sense = params.get('cell_sense_distance', 50.0)
        br_rng = np.random.default_rng(789)
        for ii in range(len(func_idx)):
            for jj in range(ii + 1, len(func_idx)):
                i, j = func_idx[ii], func_idx[jj]
                d = np.linalg.norm(positions[i] - positions[j])
                gap = d - radii[i] - radii[j]
                if gap < sense and br_rng.random() < f_bridge:
                    ax.plot([positions[i, 0], positions[j, 0]],
                            [positions[i, 1], positions[j, 1]],
                            '-', color=C_BRIDGE, linewidth=0.7,
                            alpha=0.5, zorder=3)

    # --- Draw functional granules ---
    for i in func_idx:
        cx, cy = positions[i]
        r = radii[i]
        a = r * np.sqrt(ar_f)
        b = r / np.sqrt(ar_f)
        verts = superellipse_vertices(cx, cy, a, b, n2_f, thetas[i])
        poly = MplPolygon(verts, closed=True, zorder=4)
        poly.set_facecolor(C_FUNC_FILL)
        poly.set_edgecolor(C_FUNC)
        poly.set_linewidth(0.6)
        ax.add_patch(poly)

    # --- Draw cells as small dots on functional granule surfaces ---
    n_cells_show = min(int(params['n_cells_per_func']), 6)
    if n_cells_show > 0:
        cell_rng = np.random.default_rng(456)
        for i in func_idx:
            cx, cy = positions[i]
            r = radii[i] * 0.85
            for _ in range(n_cells_show):
                ang = cell_rng.uniform(0, 2 * np.pi)
                rr = r * (0.3 + 0.7 * cell_rng.random())
                ax.plot(cx + rr * np.cos(ang), cy + rr * np.sin(ang),
                        'o', color=C_CELL, markersize=1.2, alpha=0.85,
                        zorder=5, markeredgewidth=0)

    # --- Time and compaction annotation ---
    compact_pct = max(0, (x_f_0 - x_f) / x_f_0 * 100)
    phi_v_f = max(0, 1 - phi_f / max(x_f, 1e-6))
    phi_v_i = max(0, 1 - phi_i / max(1 - x_f, 1e-6))

    info = (f"$x_f$={x_f:.2f}  "
            f"$\\phi_{{v,f}}$={phi_v_f:.2f}  "
            f"$\\phi_{{v,i}}$={phi_v_i:.2f}")
    ax.text(0.5, -0.04, info, transform=ax.transAxes, fontsize=5.5,
            ha='center', va='top', color='#616161', family='monospace')


# ======================================================================
# Main figure: spatial evolution panels
# ======================================================================

def make_evolution_figure(recommendations, outdir,
                          t_total=72.0, seed=42):
    """Create multi-panel scaffold evolution figure.

    Parameters
    ----------
    recommendations : dict
        organ_name -> list of recommendation dicts (from parameter_sweep).
    outdir : str
        Output directory for saved figures.
    t_total : float
        Total integration time in hours.
    seed : int
        RNG seed for packing generation.

    Returns
    -------
    list of str
        Paths to saved figures.
    """
    organs = [o for o in ORGANS_TO_SHOW if o in recommendations]
    if not organs:
        print("  Warning: no matching organs in recommendations")
        return []

    n_rows = len(organs)
    n_cols = len(TIME_SNAPSHOTS)
    panel_w, panel_h = 2.8, 2.8

    fig = plt.figure(figsize=(n_cols * panel_w + 1.2,
                              n_rows * (panel_h + 0.35) + 1.8))
    gs = GridSpec(n_rows + 1, n_cols,
                  height_ratios=[1] * n_rows + [0.35],
                  hspace=0.35, wspace=0.12,
                  left=0.08, right=0.97, top=0.93, bottom=0.06)

    saved = []
    all_trajectories = {}

    for row, organ in enumerate(organs):
        rec = recommendations[organ][0]  # best match
        p = rec['params']
        label = ORGAN_LABELS.get(organ, organ.replace('_', '\n').title())
        D_arch = rec['D_arch']

        print(f"  {label.replace(chr(10), ' ')} (D={D_arch:.2f})...")

        # Solve trajectory
        t_arr, x_f_arr, fb_arr = _solve_trajectory(p, t_total=t_total)
        x_f_0 = x_f_arr[0]
        all_trajectories[organ] = (t_arr, x_f_arr, fb_arr, p)

        # Generate initial packing
        pos0, radii, gtypes, L = _generate_packing(
            p['R_func'], p['R_inert'], p['phi_f'], p['phi_i'],
            target_N=TARGET_N_GRANULES, seed=seed + row * 7)

        for col, t_snap in enumerate(TIME_SNAPSHOTS):
            ax = fig.add_subplot(gs[row, col])

            # Interpolate x_f and f_bridge at this time
            idx = min(np.searchsorted(t_arr, t_snap), len(x_f_arr) - 1)
            x_f = x_f_arr[idx]
            fb = fb_arr[idx]

            # Compact positions
            pos = _compact_positions(pos0, radii, gtypes, x_f_0, x_f, L)

            _draw_snapshot(ax, pos, radii, gtypes, p,
                           x_f, x_f_0, t_snap, L, f_bridge=fb)

            # Column header (time)
            if row == 0:
                ax.set_title(f't = {t_snap} h', fontsize=10,
                             fontweight='bold', pad=6)

            # Row label
            if col == 0:
                ax.text(-0.22, 0.5, label,
                        transform=ax.transAxes, fontsize=9,
                        fontweight='bold', ha='right', va='center',
                        color='#212121', linespacing=1.3)

            # D_arch badge on rightmost column
            if col == n_cols - 1:
                ax.text(1.02, 0.5, f'$D_{{arch}}$\n{D_arch:.1f}',
                        transform=ax.transAxes, fontsize=7,
                        ha='left', va='center', color='#757575',
                        linespacing=1.4)

    # --- Legend strip ---
    ax_leg = fig.add_subplot(gs[n_rows, :])
    ax_leg.axis('off')
    legend_elements = [
        Patch(facecolor=C_FUNC_FILL, edgecolor=C_FUNC, lw=0.8,
              label='Functional granule'),
        Patch(facecolor=C_INERT_FILL, edgecolor=C_INERT, lw=0.8,
              label='Inert granule'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C_CELL, markersize=5,
               markeredgewidth=0, label='Cell'),
        Line2D([0], [0], color=C_BRIDGE, lw=1.5, alpha=0.6,
               label='Cell bridge'),
        Patch(facecolor=C_DOMAIN_BG, edgecolor='#BDBDBD', lw=0.8,
              label='Void space'),
    ]
    ax_leg.legend(handles=legend_elements, loc='center', ncol=5,
                  fontsize=8.5, frameon=False, handlelength=1.8,
                  columnspacing=2.0)

    fig.suptitle(
        'Scaffold Microstructure Evolution Under Cell-Driven Compaction',
        fontsize=14, fontweight='bold')

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'scaffold_evolution.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    saved.append(path)

    # ---- Kinetics companion plot ----
    path2 = _make_kinetics_figure(all_trajectories, outdir)
    if path2:
        saved.append(path2)

    return saved


# ======================================================================
# Companion: kinetics trajectories
# ======================================================================

def _make_kinetics_figure(trajectories, outdir):
    """Plot x_f(t), local void fractions, and bridge fraction for each organ."""
    if not trajectories:
        return None

    organs = list(trajectories.keys())
    n = len(organs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    cmap = plt.colormaps.get_cmap('Set1').resampled(max(n, 3))
    colors = {org: cmap(i) for i, org in enumerate(organs)}

    for organ in organs:
        t_arr, x_f_arr, fb_arr, p = trajectories[organ]
        phi_f = p['phi_f']
        phi_i = p['phi_i']
        c = colors[organ]
        lbl = ORGAN_LABELS.get(organ, organ).replace('\n', ' ')

        # Panel 1: x_f(t) — functional zone fraction
        axes[0].plot(t_arr, x_f_arr, '-', color=c, lw=2, label=lbl)

        # Panel 2: local void fractions
        phi_v_f = np.maximum(0, 1 - phi_f / np.maximum(x_f_arr, 1e-6))
        phi_v_i = np.maximum(0, 1 - phi_i / np.maximum(1 - x_f_arr, 1e-6))
        axes[1].plot(t_arr, phi_v_f, '-', color=c, lw=2, label=f'{lbl} (func)')
        axes[1].plot(t_arr, phi_v_i, '--', color=c, lw=1.5, alpha=0.7)

        # Panel 3: bridge fraction
        axes[2].plot(t_arr, fb_arr, '-', color=c, lw=2, label=lbl)

    axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel('Functional zone fraction $x_f$')
    axes[0].set_title('Zone Compaction', fontweight='bold')
    axes[0].legend(fontsize=7, loc='best')
    axes[0].set_ylim(bottom=0)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel('Local void fraction')
    axes[1].set_title('Void Redistribution', fontweight='bold')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    # Manual legend for solid/dashed
    axes[1].plot([], [], 'k-', lw=1.5, label='Functional zone')
    axes[1].plot([], [], 'k--', lw=1.5, alpha=0.7, label='Inert zone')
    axes[1].legend(fontsize=7, loc='best')

    axes[2].set_xlabel('Time (h)')
    axes[2].set_ylabel('Bridge fraction')
    axes[2].set_title('Cell Bridging Kinetics', fontweight='bold')
    axes[2].legend(fontsize=7, loc='best')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle('Mean-Field Compaction Kinetics per Organ Target',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    path = os.path.join(outdir, 'scaffold_kinetics.png')
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")
    return path


# ======================================================================
# Timelapse animation
# ======================================================================

def _render_frame_from_pos(pos_current, radii, gtypes, L, params,
                           x_f, x_f_0, fb, frame_t,
                           t_arr, x_f_arr,
                           organ_label, D_arch,
                           viewport=None,
                           fig_size=(8, 8)):
    """Render one animation frame from pre-computed positions.

    Returns RGB uint8 array.
    """
    phi_f = params['phi_f']
    phi_i = params['phi_i']

    fig = plt.figure(figsize=fig_size, facecolor='white')
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.88])
    _draw_snapshot(ax, pos_current, radii, gtypes, params,
                   x_f, x_f_0, frame_t, L, f_bridge=fb,
                   viewport=viewport)
    for txt in list(ax.texts):
        txt.remove()

    # HUD
    compact_pct = max(0, (x_f_0 - x_f) / x_f_0 * 100)
    phi_v_f = max(0, 1 - phi_f / max(x_f, 1e-6))
    phi_v_i = max(0, 1 - phi_i / max(1 - x_f, 1e-6))
    phi_v_global = 1 - phi_f - phi_i

    title = (f"{organ_label.replace(chr(10), ' ')}  |  "
             f"$D_{{arch}}$ = {D_arch:.1f}")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    hud = (f"t = {frame_t:.1f} h    "
           f"compaction = {compact_pct:.1f}%    "
           f"$\\phi_{{v}}^{{global}}$ = {phi_v_global:.2f}    "
           f"$\\phi_{{v,f}}^{{local}}$ = {phi_v_f:.2f}    "
           f"$\\phi_{{v,i}}^{{local}}$ = {phi_v_i:.2f}")
    ax.text(0.5, 1.005, hud, transform=ax.transAxes, fontsize=8,
            ha='center', va='bottom', color='#424242', family='monospace')

    # Progress bar
    t_total = t_arr[-1]
    prog = frame_t / t_total if t_total > 0 else 0
    bar_ax = fig.add_axes([0.15, 0.005, 0.70, 0.012])
    bar_ax.set_xlim(0, 1)
    bar_ax.set_ylim(0, 1)
    bar_ax.barh(0.5, prog, height=1.0, color='#388E3C', alpha=0.7)
    bar_ax.barh(0.5, 1.0, height=1.0, color='#E0E0E0', alpha=0.3)
    bar_ax.set_xticks([])
    bar_ax.set_yticks([])
    for spine in bar_ax.spines.values():
        spine.set_visible(False)
    bar_ax.text(-0.01, 0.5, '0 h', fontsize=6, ha='right', va='center',
                transform=bar_ax.transAxes, color='#757575')
    bar_ax.text(1.01, 0.5, f'{t_total:.0f} h', fontsize=6, ha='left',
                va='center', transform=bar_ax.transAxes, color='#757575')

    # Kinetics inset
    inset = fig.add_axes([0.68, 0.06, 0.28, 0.22])
    inset.plot(t_arr, x_f_arr, '-', color='#388E3C', lw=1.5)
    inset.axvline(frame_t, color='#C62828', lw=1.0, ls='--', alpha=0.8)
    inset.set_xlim(0, t_arr[-1])
    inset.set_ylim(0, max(x_f_arr) * 1.1)
    inset.set_xlabel('t (h)', fontsize=7)
    inset.set_ylabel('$x_f$', fontsize=7)
    inset.tick_params(labelsize=6)
    inset.set_facecolor('white')
    inset.patch.set_alpha(0.85)
    for spine in inset.spines.values():
        spine.set_edgecolor('#BDBDBD')
    inset.grid(True, alpha=0.2)

    # Legend
    legend_elements = [
        Patch(facecolor=C_FUNC_FILL, edgecolor=C_FUNC, lw=0.6,
              label='Functional'),
        Patch(facecolor=C_INERT_FILL, edgecolor=C_INERT, lw=0.6,
              label='Inert'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=C_CELL, markersize=4,
               markeredgewidth=0, label='Cell'),
        Line2D([0], [0], color=C_BRIDGE, lw=1, alpha=0.6,
               label='Bridge'),
    ]
    ax.legend(handles=legend_elements, loc='lower left',
              fontsize=7, frameon=True, framealpha=0.85,
              edgecolor='#BDBDBD', ncol=4, handlelength=1.2,
              columnspacing=1.0, borderpad=0.4)

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(h, w, 4)[:, :, :3]
    plt.close(fig)
    return img


def _advance_positions(pos, radii, types, target, L, damping=0.15,
                       overlap_iters=8):
    """Smoothly advance positions toward target with overlap resolution.

    This is the core of temporal coherence: each frame inherits positions
    from the previous frame and moves a fraction of the way toward the
    target (overdamped dynamics).  A few overlap resolution iterations
    prevent interpenetration without causing jitter.
    """
    from scipy.spatial import cKDTree

    new = pos.copy()

    # Damped step toward target
    new += damping * (target - new)

    # Gentle overlap resolution
    r_max = float(np.max(radii))
    cutoff = 2.0 * r_max + 1.0
    is_func = (types == 0)

    for _ in range(overlap_iters):
        tree = cKDTree(new)
        pairs = tree.query_pairs(cutoff)
        max_ov = 0.0
        for i, j in pairs:
            dx = new[j] - new[i]
            d = np.linalg.norm(dx)
            mind = radii[i] + radii[j]
            if d < mind and d > 0.01:
                ov = mind - d
                if ov > max_ov:
                    max_ov = ov
                n_hat = dx / d
                push = 0.2 * ov if (is_func[i] and is_func[j]) else 0.45 * ov
                new[i] -= push * n_hat
                new[j] += push * n_hat

        for i in range(len(new)):
            r = radii[i]
            new[i, 0] = np.clip(new[i, 0], r, L - r)
            new[i, 1] = np.clip(new[i, 1], r, L - r)

        if max_ov < 0.3:
            break

    return new


DOMAIN_BUFFER = 2.2   # large domain = BUFFER × viewport in each dimension

def make_timelapse(recommendations, outdir,
                   t_total=72.0, fps=15, duration_sec=8,
                   seed=42, fmt='mp4'):
    """Create a smooth timelapse MP4 (or GIF) for each organ.

    The packing is generated in a domain DOMAIN_BUFFER× larger than the
    viewport.  As cells compact the functional zone, granules stream inward
    from outside the visible window — modelling an effectively infinite
    scaffold surrounding the field of view.

    Parameters
    ----------
    recommendations : dict
        organ -> list of recommendation dicts.
    outdir : str
        Output directory.
    t_total : float
        Integration time (hours).
    fps : int
        Frames per second.
    duration_sec : float
        Total video duration in seconds.
    seed : int
        RNG seed for packing.
    fmt : str
        'mp4' or 'gif'.

    Returns
    -------
    list of str
        Paths to saved videos.
    """
    import imageio

    organs = [o for o in ORGANS_TO_SHOW if o in recommendations]
    if not organs:
        print("  Warning: no matching organs in recommendations")
        return []

    n_frames = int(fps * duration_sec)
    frame_times = np.linspace(0, t_total, n_frames)
    buf = DOMAIN_BUFFER

    os.makedirs(outdir, exist_ok=True)
    saved = []

    for row, organ in enumerate(organs):
        rec = recommendations[organ][0]
        p = rec['params']
        label = ORGAN_LABELS.get(organ, organ.replace('_', '\n').title())
        D_arch = rec['D_arch']

        # Large packing: buf^2 × more granules to fill the extended domain
        n_large = int(TARGET_N_GRANULES * buf * buf)
        print(f"  {label.replace(chr(10), ' ')} "
              f"({n_large} granules, {n_frames} frames)...",
              end='', flush=True)

        t_arr, x_f_arr, fb_arr = _solve_trajectory(p, t_total=t_total)

        pos0, radii, gtypes, L_big = _generate_packing(
            p['R_func'], p['R_inert'], p['phi_f'], p['phi_i'],
            target_N=n_large, seed=seed + row * 7)

        # Viewport: central window of side L_big / buf
        L_view = L_big / buf
        margin = (L_big - L_view) / 2.0
        viewport = (margin, margin, margin + L_view, margin + L_view)

        ext = 'gif' if fmt == 'gif' else 'mp4'
        safe_name = organ.replace(' ', '_')
        path = os.path.join(outdir, f'scaffold_timelapse_{safe_name}.{ext}')

        if fmt == 'mp4':
            writer_kwargs = {'fps': fps, 'format': 'FFMPEG',
                             'codec': 'libx264', 'quality': 8}
        else:
            writer_kwargs = {'fps': fps, 'format': 'GIF', 'mode': 'I'}

        # Temporal coherence: maintain state across frames
        x_f_0 = x_f_arr[0]
        pos_current = pos0.copy()

        with imageio.get_writer(path, **writer_kwargs) as writer:
            for fi, ft in enumerate(frame_times):
                # Interpolate trajectory at this time
                idx = min(np.searchsorted(t_arr, ft), len(x_f_arr) - 1)
                x_f = x_f_arr[idx]
                fb = fb_arr[idx]

                # Compute target positions for this x_f
                target = _compact_positions(pos0, radii, gtypes,
                                            x_f_0, x_f, L_big)

                # Smoothly advance toward target (damped + overlap resolve)
                pos_current = _advance_positions(
                    pos_current, radii, gtypes, target, L_big,
                    damping=0.18, overlap_iters=10)

                img = _render_frame_from_pos(
                    pos_current, radii, gtypes, L_big, p,
                    x_f, x_f_0, fb, ft,
                    t_arr, x_f_arr,
                    label, D_arch,
                    viewport=viewport)
                writer.append_data(img)
                if (fi + 1) % 20 == 0:
                    print('.', end='', flush=True)

        print(f' saved: {path}')
        saved.append(path)

    return saved


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scaffold evolution visualization from mean-field sweep')
    parser.add_argument('-i', '--input', default='results/parameter_sweep',
                        help='Directory containing recommendations.json')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory (default: same as input)')
    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed for packing')
    parser.add_argument('--t-total', type=float, default=72.0,
                        help='Total integration time in hours')
    parser.add_argument('--timelapse', action='store_true',
                        help='Generate timelapse animations (MP4)')
    parser.add_argument('--gif', action='store_true',
                        help='Use GIF format instead of MP4')
    parser.add_argument('--fps', type=int, default=15,
                        help='Frames per second for timelapse')
    parser.add_argument('--duration', type=float, default=8.0,
                        help='Timelapse duration in seconds')
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
    print("  Scaffold Evolution Visualization")
    print("=" * 60)

    if args.timelapse or args.gif:
        fmt = 'gif' if args.gif else 'mp4'
        make_timelapse(recommendations, outdir,
                       t_total=args.t_total, fps=args.fps,
                       duration_sec=args.duration, seed=args.seed,
                       fmt=fmt)
    else:
        make_evolution_figure(recommendations, outdir,
                              t_total=args.t_total, seed=args.seed)

    print("=" * 60)
    print("  Done.")
    print("=" * 60)


if __name__ == '__main__':
    main()
