"""
Unified Post-Processing & Visualization for GELS
=====================================================
V1.6 — Loads saved simulation data and produces all visualizations.

Usage:
    # Process a single run:
    python viz/postprocess.py -i results/default
    python viz/postprocess.py -i results/default.tar.gz
    python viz/postprocess.py -i results/default --skip movies stress

    # Auto-process ALL unprocessed .tar.gz files in a directory:
    python viz/postprocess.py                          # scans results/trials/
    python viz/postprocess.py --scan-dir path/to/runs  # custom scan directory
    python viz/postprocess.py --skip movies stress      # auto-scan with skips

Or import programmatically:
    from viz.postprocess import run_all, run_all_unprocessed
    run_all('results/default')
    run_all_unprocessed('results/trials/')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import argparse
import glob as globmod
import os
import sys


# ══════════════════════════════════════════════════════════════════════
# Built-in plots (moved from new_dem_0.py)
# ══════════════════════════════════════════════════════════════════════

def _select_indices(n, indices=None):
    """Pick ~5 evenly-spaced snapshot indices."""
    if indices is not None:
        return indices
    return sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))


def plot_granules(snaps, hist, p, indices=None):
    """Plot granule positions as circles or superellipses at selected times.

    For 3D mode, delegates to viz_stress.plot_granules_3d() if available.
    """
    is_3d = (getattr(p, 'mode', '2D') == '3D' or
             (snaps and isinstance(snaps[0], dict) and 'quat' in snaps[0]))
    if is_3d:
        try:
            from viz import stress as viz_stress
            if viz_stress.HAS_PYVISTA:
                return viz_stress.plot_granules_3d(snaps, hist, p, indices=indices)
        except ImportError:
            pass

    from new_dem_0 import superellipse_polygon_pts

    indices = _select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4 * nc, 4))
    if nc == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        snap = snaps[si]
        if isinstance(snap, dict):
            xs, ys, rs, gt = snap['x'], snap['y'], snap['r'], snap['gtype']
            has_shape = 'a' in snap and 'b' in snap and 'n_shape' in snap
            if has_shape:
                a_arr, b_arr = snap['a'], snap['b']
                ns_arr = snap.get('n1', snap['n_shape'])
                th_arr = snap.get('theta', np.zeros(len(xs)))
        else:
            xs, ys, rs, gt = snap[3], snap[4], snap[5], snap[6]
            if len(snap) > 11:
                a_arr, b_arr, ns_arr, th_arr = snap[11], snap[12], snap[13], snap[14]
                has_shape = True
            else:
                has_shape = False

        ax = axes[c]
        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

        for i in range(len(xs)):
            color = 'orangered' if gt[i] == 0 else 'steelblue'
            alpha = 0.75 if gt[i] == 0 else 0.55

            if has_shape and not (a_arr[i] == b_arr[i] and ns_arr[i] == 2.0):
                verts = superellipse_polygon_pts(
                    xs[i], ys[i], a_arr[i], b_arr[i], ns_arr[i], th_arr[i])
                patch = Polygon(verts, closed=True, fc=color, ec='k',
                                lw=0.3, alpha=alpha)
            else:
                patch = Circle((xs[i], ys[i]), rs[i], fc=color, ec='k',
                               lw=0.3, alpha=alpha)
            ax.add_patch(patch)

        ax.set_title(f"t = {hist[si]['time']:.1f} h", fontsize=10)
        if c == 0:
            ax.plot([], [], 'o', color='orangered', ms=8, label='Functional')
            ax.plot([], [], 'o', color='steelblue', ms=8, label='Inert')
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Granule Positions Over Time', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_fields(snaps, hist, p, indices=None):
    """Phase fields rendered from granule positions."""
    indices = _select_indices(len(snaps), indices)
    nc = len(indices)
    ext = [0, p.Lx, 0, p.Ly]
    fig, ax = plt.subplots(3, nc, figsize=(3.2 * nc, 9))
    lbl = [r'$\phi_f$', r'$\phi_i$', r'$\phi_v$']
    cm = ['Oranges', 'Blues', 'Greens']

    for c, si in enumerate(indices):
        snap = snaps[si]
        if isinstance(snap, dict):
            pf, pi, pv = snap['phi_f'], snap['phi_i'], snap['phi_v']
        else:
            pf, pi, pv = snap[0], snap[1], snap[2]
        if pf.ndim == 3:
            mid_z = pf.shape[2] // 2
            pf, pi, pv = pf[:, :, mid_z], pi[:, :, mid_z], pv[:, :, mid_z]
        flds = [pf, pi, pv]
        t = hist[si]['time']
        for r in range(3):
            a = ax[r, c]
            vx = max(.01, flds[r].max() * 1.05)
            im = a.imshow(flds[r].T, origin='lower', extent=ext,
                          cmap=cm[r], vmin=0, vmax=vx)
            a.set_title(f't={t:.1f}h', fontsize=8)
            if c == 0:
                a.set_ylabel(lbl[r], fontsize=11)
            plt.colorbar(im, ax=a, fraction=.046, pad=.04)
    fig.suptitle('Rendered Phase Fields', fontsize=13, y=1.01)
    plt.tight_layout()
    return fig


def plot_timeseries(hist, p):
    """3x3 grid of scalar metric time series."""
    t = [h['time'] for h in hist]
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))

    ax[0, 0].plot(t, [h['func_nc'] for h in hist], 'C1-o', ms=3, label='Functional')
    ax[0, 0].plot(t, [h['void_nc'] for h in hist], 'C2-s', ms=3, label='Void')
    ax[0, 0].plot(t, [h['inert_nc'] for h in hist], 'C0-^', ms=3, label='Inert')
    ax[0, 0].set(xlabel='time (h)', ylabel='# clusters', title='Cluster Count')
    ax[0, 0].legend()

    ax[0, 1].plot(t, [h['func_lf'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[0, 1].plot(t, [h['void_lf'] for h in hist], 'C2--', label='Void')
    ax[0, 1].axhline(1, ls=':', c='gray', lw=0.8)
    ax[0, 1].set(xlabel='time (h)', ylabel='largest / total',
                 title='Connectivity (1 = percolated)')
    ax[0, 1].legend()

    ax[0, 2].plot(t, [h['n_bridges'] for h in hist], 'C3-', lw=2)
    ax[0, 2].set(xlabel='time (h)', ylabel='# bridges', title='Active Cell Bridges')

    ax[1, 0].plot(t, [h['disp_func'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[1, 0].plot(t, [h['disp_inert'] for h in hist], 'C0--', label='Inert')
    ax[1, 0].set(xlabel='time (h)', ylabel='mean disp (µm)',
                 title='Granule Displacement')
    ax[1, 0].legend()

    ax[1, 1].plot(t, [h['tissue_frac'] for h in hist], 'C3-', lw=2,
                  label=r'Tissue ($\phi_f > 0.5$)')
    ax[1, 1].plot(t, [h['packing_func_rich'] for h in hist], 'C1--',
                  label='Packing in func-rich')
    ax[1, 1].set(xlabel='time (h)', ylabel='fraction', title='Tissue Remodeling')
    ax[1, 1].legend()

    ax[1, 2].plot(t, [h['func_max_area'] for h in hist], 'C1-', lw=2)
    ax[1, 2].set(xlabel='time (h)', ylabel='area (µm²)',
                 title='Largest Functional Cluster Area')

    ax[2, 0].plot(t, [h['n_attached_total'] for h in hist], 'C4-', lw=2,
                  label='Attached')
    ax[2, 0].plot(t, [h['n_seeded_total'] for h in hist], 'C7--', lw=1,
                  label='Seeded')
    ax[2, 0].axvline(p.t_attach_onset, ls=':', c='gray', lw=0.8, label='Attach onset')
    ax[2, 0].set(xlabel='time (h)', ylabel='# cells', title='Cell Attachment')
    ax[2, 0].legend()

    ax[2, 1].plot(t, [h['mean_spread_frac'] for h in hist], 'C5-', lw=2,
                  label='Spread fraction')
    ax[2, 1].plot(t, [h['mean_fa_maturity'] for h in hist], 'C6--', lw=2,
                  label='FA maturity')
    ax[2, 1].set(xlabel='time (h)', ylabel='fraction [0–1]',
                 title='Cell Spreading & Focal Adhesion')
    ax[2, 1].set_ylim(-0.05, 1.05)
    ax[2, 1].legend()

    ax[2, 2].plot(t, [h['n_overcrowded_total'] for h in hist], 'C3-', lw=2)
    ax[2, 2].set(xlabel='time (h)', ylabel='# cells',
                 title='Overcrowded Cells (crawling on others)')

    plt.tight_layout()
    return fig


def plot_composite(snaps, hist, p, indices=None):
    """RGB composite: R=functional, G=void, B=inert."""
    indices = _select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(3.8 * nc, 3.5))
    if nc == 1:
        axes = [axes]
    for c, si in enumerate(indices):
        snap = snaps[si]
        if isinstance(snap, dict):
            pf, pi, pv = snap['phi_f'], snap['phi_i'], snap['phi_v']
        else:
            pf, pi, pv = snap[0], snap[1], snap[2]
        if pf.ndim == 3:
            mid_z = pf.shape[2] // 2
            pf, pi, pv = pf[:, :, mid_z], pi[:, :, mid_z], pv[:, :, mid_z]
        mx = max(pf.max(), pi.max(), pv.max(), 0.01)
        rgb = np.stack([pf / mx, pv / mx, pi / mx], axis=-1)
        axes[c].imshow(np.clip(np.transpose(rgb, (1, 0, 2)), 0, 1),
                       origin='lower', extent=[0, p.Lx, 0, p.Ly])
        axes[c].set_title(f"t={hist[si]['time']:.1f}h", fontsize=9)
    fig.suptitle("R=functional  G=void  B=inert", fontsize=11, y=1.02)
    plt.tight_layout()
    return fig


def plot_scaffold_map(snaps, hist, p, indices=None):
    """Vector-drawn scaffold map: red=functional, green=inert, black=void, cells overlaid.

    Uses matplotlib patches (circles/superellipses) instead of grid-based phase fields,
    avoiding aliasing artifacts entirely. Cell morphology drawn on top using cell-state
    colors from viz.cells.
    """
    from new_dem_0 import superellipse_polygon_pts, superellipse_point, CellState

    indices = _select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4.5 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    # Cell-state color map (matches viz/cells.py)
    cell_colors = {
        int(CellState.ATTACHED):      '#DAA520',
        int(CellState.SPREADING):     '#FF8C00',
        int(CellState.PROLIFERATING): '#228B22',
        int(CellState.BRIDGING):      '#DC143C',
        int(CellState.SENESCENT):     '#696969',
        int(CellState.MIGRATING):     '#4169E1',
    }
    cell_labels = {
        int(CellState.ATTACHED):      'Attached',
        int(CellState.SPREADING):     'Spreading',
        int(CellState.PROLIFERATING): 'Proliferating',
        int(CellState.BRIDGING):      'Bridging',
        int(CellState.SENESCENT):     'Senescent',
        int(CellState.MIGRATING):     'Migrating',
    }

    is_3d = (getattr(p, 'mode', '2D') == '3D' or
             (snaps and isinstance(snaps[0], dict) and 'quat' in snaps[0]))
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'

    for c, si in enumerate(indices):
        snap = snaps[si]
        ax = axes[c]
        ax.set_facecolor('black')
        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

        xs, ys, rs, gt = snap['x'], snap['y'], snap['r'], snap['gtype']
        a_arr, b_arr = snap['a'], snap['b']
        n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
        theta_arr = snap.get('theta', np.zeros(len(xs)))
        N = len(xs)

        # -- Build granule patches with per-granule colors --
        patches = []
        colors = []
        for i in range(N):
            fc = '#CC2222' if gt[i] == 0 else '#22AA22'  # red / green
            # Collect positions to draw (original + periodic ghosts)
            positions = [(xs[i], ys[i])]
            if periodic:
                for dx_off in [-p.Lx, 0, p.Lx]:
                    for dy_off in [-p.Ly, 0, p.Ly]:
                        if dx_off == 0 and dy_off == 0:
                            continue
                        gx = xs[i] + dx_off
                        gy = ys[i] + dy_off
                        rbound = rs[i] * 1.1
                        if (gx + rbound > 0 and gx - rbound < p.Lx and
                                gy + rbound > 0 and gy - rbound < p.Ly):
                            positions.append((gx, gy))

            for (px, py) in positions:
                if is_3d:
                    r_proj = max(a_arr[i], b_arr[i])
                    patches.append(Circle((px, py), r_proj))
                elif not (a_arr[i] == b_arr[i] and n1_arr[i] == 2.0):
                    verts = superellipse_polygon_pts(
                        px, py, a_arr[i], b_arr[i], n1_arr[i], theta_arr[i])
                    patches.append(Polygon(verts, closed=True))
                else:
                    patches.append(Circle((px, py), rs[i]))
                colors.append(fc)

        pc = PatchCollection(patches, facecolors=colors, edgecolors='none',
                             linewidths=0, alpha=0.95)
        ax.add_collection(pc)

        # -- Draw cells on top --
        cell_states = snap.get('cell_state', np.array([], dtype=int))
        n_cells = len(cell_states)
        if n_cells > 0:
            cell_gi = snap['cell_granule_id']
            cell_theta = snap.get('cell_theta_local', np.zeros(n_cells))

            # Compute cell world positions (inline, 2D only for now)
            wx = np.zeros(n_cells)
            wy = np.zeros(n_cells)
            for ci in range(n_cells):
                gi = int(cell_gi[ci])
                if gi < 0 or gi >= N:
                    continue
                if is_3d:
                    r_proj = max(a_arr[gi], b_arr[gi])
                    ang = cell_theta[ci]
                    wx[ci] = xs[gi] + r_proj * np.cos(ang)
                    wy[ci] = ys[gi] + r_proj * np.sin(ang)
                else:
                    bx, by = superellipse_point(
                        cell_theta[ci], a_arr[gi], b_arr[gi], n1_arr[gi])
                    ct = np.cos(theta_arr[gi])
                    st = np.sin(theta_arr[gi])
                    wx[ci] = xs[gi] + ct * bx - st * by
                    wy[ci] = ys[gi] + st * bx + ct * by

            # Wrap cell positions for periodic
            if periodic:
                wx = wx % p.Lx
                wy = wy % p.Ly

            for ci in range(n_cells):
                state = int(cell_states[ci])
                cc = cell_colors.get(state, '#AAAAAA')

                if state == int(CellState.BRIDGING):
                    bt = int(snap['cell_bridge_target'][ci])
                    if 0 <= bt < N:
                        # Target surface point
                        dx_t = wx[ci] - xs[bt]
                        dy_t = wy[ci] - ys[bt]
                        if periodic:
                            dx_t -= p.Lx * round(dx_t / p.Lx)
                            dy_t -= p.Ly * round(dy_t / p.Ly)
                        dist = np.sqrt(dx_t**2 + dy_t**2)
                        if dist > 1e-6:
                            tx = xs[bt] + rs[bt] * dx_t / dist
                            ty = ys[bt] + rs[bt] * dy_t / dist
                            if periodic:
                                tx = tx % p.Lx
                                ty = ty % p.Ly
                        else:
                            tx, ty = wx[ci], wy[ci]
                        # Bridge ellipse
                        bdx = tx - wx[ci]
                        bdy = ty - wy[ci]
                        if periodic:
                            bdx -= p.Lx * round(bdx / p.Lx)
                            bdy -= p.Ly * round(bdy / p.Ly)
                        blen = np.sqrt(bdx**2 + bdy**2)
                        if blen > 1e-6:
                            a_long = max(blen / 2.0, 5.0)
                            a_perp = max(np.pi * 100.0 / (np.pi * a_long), 2.0)
                            cx_b = wx[ci] + bdx / 2.0
                            cy_b = wy[ci] + bdy / 2.0
                            ang_b = np.degrees(np.arctan2(bdy, bdx))
                            patch = Ellipse((cx_b, cy_b), width=2*a_long,
                                            height=2*a_perp, angle=ang_b,
                                            fc=cc, ec='white', lw=0.3, alpha=0.85)
                            ax.add_patch(patch)
                        else:
                            ax.add_patch(Circle((wx[ci], wy[ci]), 5,
                                                fc=cc, ec='white', lw=0.2, alpha=0.8))
                elif state == int(CellState.MIGRATING):
                    # Migrating: small arrow-like marker
                    ax.add_patch(Ellipse((wx[ci], wy[ci]), width=14, height=8,
                                         angle=np.degrees(cell_theta[ci]),
                                         fc=cc, ec='white', lw=0.2, alpha=0.8))
                else:
                    # Attached/spreading/proliferating/senescent: small circle
                    r_cell = 6 if state == int(CellState.PROLIFERATING) else 5
                    ax.add_patch(Circle((wx[ci], wy[ci]), r_cell,
                                        fc=cc, ec='white', lw=0.2, alpha=0.8))

        t = hist[si]['time'] if si < len(hist) else 0
        ax.set_title(f"t = {t:.1f} h", fontsize=10, color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')

    # Legend
    legend_handles = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#CC2222',
               markersize=10, linestyle='None', label='Functional'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#22AA22',
               markersize=10, linestyle='None', label='Inert'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
               markeredgecolor='white', markersize=10, linestyle='None', label='Void'),
    ]
    # Add cell state entries if cells present
    seen_states = set()
    for si in indices:
        cs = snaps[si].get('cell_state', np.array([], dtype=int))
        seen_states.update(int(s) for s in cs)
    for s_val in sorted(seen_states):
        if s_val in cell_colors:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=cell_colors[s_val], markersize=7,
                       linestyle='None', label=cell_labels.get(s_val, f'State {s_val}')))

    axes[0].legend(handles=legend_handles, fontsize=7, loc='upper right',
                   framealpha=0.7, facecolor='#333333', labelcolor='white')

    fig.patch.set_facecolor('black')
    fig.suptitle("Scaffold Map: Red=Functional  Green=Inert  Black=Void",
                 fontsize=11, y=1.02, color='white')
    plt.tight_layout()
    return fig


def plot_deformation(snaps, hist, p, indices=None):
    """Plot granules colored by deformation magnitude (LS-DEM V2.3).

    Each particle is drawn as a circle/superellipse and colored by its total
    deformation strain |epsilon| = sqrt(sum epsilon_alpha^2).  Only produces
    output if snapshot data contains 'epsilon'.
    """
    # Check if any snapshot has deformation data
    if not snaps or not isinstance(snaps[0], dict) or 'epsilon' not in snaps[0]:
        return None

    from new_dem_0 import superellipse_polygon_pts

    indices = _select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4 * nc, 4.5))
    if nc == 1:
        axes = [axes]

    # Global colour scale across all snapshots shown
    vmax = 0.0
    for si in indices:
        eps = snaps[si].get('epsilon')
        if eps is not None:
            vmax = max(vmax, np.sqrt(np.sum(eps**2, axis=1)).max())
    if vmax < 1e-10:
        vmax = 0.1  # avoid zero-range colourbar

    cmap = plt.cm.inferno

    for c, si in enumerate(indices):
        snap = snaps[si]
        xs, ys, rs, gt = snap['x'], snap['y'], snap['r'], snap['gtype']
        eps_arr = snap.get('epsilon')
        has_shape = 'a' in snap and 'b' in snap and 'n_shape' in snap
        if has_shape:
            a_arr, b_arr = snap['a'], snap['b']
            ns_arr = snap.get('n1', snap['n_shape'])
            th_arr = snap.get('theta', np.zeros(len(xs)))

        ax = axes[c]
        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

        for i in range(len(xs)):
            if eps_arr is not None:
                strain = np.sqrt(np.sum(eps_arr[i]**2))
                color = cmap(strain / vmax)
            else:
                color = (0.5, 0.5, 0.5)

            if has_shape and not (a_arr[i] == b_arr[i] and ns_arr[i] == 2.0):
                verts = superellipse_polygon_pts(
                    xs[i], ys[i], a_arr[i], b_arr[i], ns_arr[i], th_arr[i])
                patch = Polygon(verts, closed=True, fc=color, ec='k',
                                lw=0.3, alpha=0.85)
            else:
                patch = Circle((xs[i], ys[i]), rs[i], fc=color, ec='k',
                               lw=0.3, alpha=0.85)
            ax.add_patch(patch)

        ax.set_title(f"t = {hist[si]['time']:.1f} h", fontsize=10)

    # Add colourbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04)
    cbar.set_label(r'Deformation strain $|\epsilon|$', fontsize=10)

    fig.suptitle('LS-DEM Particle Deformation', fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_deformation_timeseries(hist, p):
    """Time series of deformation metrics (LS-DEM V2.3).

    Only produces output when deformation metrics are present in history.
    """
    if not hist or 'def_strain_mean' not in hist[0]:
        return None

    t = [h['time'] for h in hist]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel 1: Mean and max deformation strain
    ax = axes[0]
    ax.plot(t, [h['def_strain_mean'] for h in hist], 'C1-', lw=2,
            label='Mean $|\\epsilon|$')
    ax.plot(t, [h['def_strain_max'] for h in hist], 'C3--', lw=2,
            label='Max $|\\epsilon|$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Deformation strain')
    ax.set_title('LS-DEM Deformation Over Time')
    ax.legend(fontsize=9)

    # Panel 2: Per-mode breakdown
    ax = axes[1]
    mode_keys = sorted([k for k in hist[0] if k.startswith('def_mode_') and k.endswith('_mean')])
    colours = ['C0', 'C1', 'C2', 'C4', 'C5']
    mode_names = ['Axial compress.', 'Prolate-oblate', 'Volumetric']
    for mi, key in enumerate(mode_keys):
        label = mode_names[mi] if mi < len(mode_names) else f'Mode {mi}'
        colour = colours[mi % len(colours)]
        means = [h[key] for h in hist]
        ax.plot(t, means, f'{colour}-', lw=2, label=label)
        # Shade mean +/- std
        std_key = key.replace('_mean', '_std')
        if std_key in hist[0]:
            stds = [h[std_key] for h in hist]
            ax.fill_between(t,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            alpha=0.2, color=colour)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Mode amplitude $\\epsilon_\\alpha$')
    ax.set_title('Per-Mode Deformation Amplitudes')
    ax.legend(fontsize=9)
    ax.axhline(0, ls=':', c='gray', lw=0.6)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

def _rerender_snapshot(snap, p):
    """Re-render phi fields from saved particle positions using current render code.

    This allows post-hoc correction of rendering artifacts (e.g. grid resolution,
    interface width) without re-running the simulation.
    """
    from new_dem_0 import GranuleSystem, render_fields, render_fields_3d

    mode = snap.get('mode', getattr(p, 'mode', '2D'))
    if isinstance(mode, np.ndarray):
        mode = str(mode)
    N = len(snap['x'])

    # Reconstruct minimal GranuleSystem
    gs = GranuleSystem.__new__(GranuleSystem)
    gs.N = N
    gs.x = snap['x'].copy()
    gs.y = snap['y'].copy()
    gs.z = snap.get('z', np.zeros(N))
    gs.r = snap['r'].copy()
    gs.gtype = snap['gtype'].copy()
    gs.a = snap.get('a', snap['r'].copy())
    gs.b = snap.get('b', snap['r'].copy())
    gs.c = snap.get('c', snap['r'].copy())
    gs.n1 = snap.get('n1', np.full(N, 2.0))
    gs.n2 = snap.get('n2', np.full(N, 2.0))
    gs.n_shape = snap.get('n_shape', gs.n1)
    gs.theta = snap.get('theta', np.zeros(N))
    gs.quat = snap.get('quat', None)
    gs.mode = mode if isinstance(mode, str) else '2D'
    gs.is_circle = bool(np.all(gs.a == gs.b) and np.all(gs.n1 == 2.0))
    gs.func_mask = (gs.gtype == 0)
    gs.contact_clips = [[] for _ in range(N)]  # no clips on re-render
    gs.epsilon = snap.get('epsilon', None)
    gs.d_epsilon = snap.get('d_epsilon', None)
    if gs.is_3d:
        gs.r_bound = np.maximum(np.maximum(gs.a, gs.b), gs.c)
    else:
        gs.r_bound = np.maximum(gs.a, gs.b)

    if gs.is_3d:
        pf, pi, pv = render_fields_3d(gs, p)
    else:
        pf, pi, pv = render_fields(gs, p)

    return pf, pi, pv


def run_all(run_dir, outdir=None, skip=None, rerender=False):
    """Load saved simulation data and produce all visualizations.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or .tar.gz archive.
    outdir : str or None
        Directory for output plots. Defaults to ``run_dir/plots``.
    skip : set of str or None
        Names of visualization modules to skip. Valid names:
        'granules', 'fields', 'timeseries', 'composite', 'deformation',
        'compaction', 'percolation', 'movies', 'phases', 'cells', 'stress',
        'mean_field', 'coarse_grain', 'tissue', 'arch_distance',
        'conservation'.
    rerender : bool
        If True, re-render phi fields from saved particle positions using
        current render code (fixes resolution/aliasing issues without
        re-running the simulation).
    """
    from new_dem_0 import load_run

    skip = set(skip or [])

    print("=" * 65)
    print("  GELS Post-Processing")
    print("=" * 65)
    print(f"\n  Loading data from: {run_dir}")

    hist, snaps, p, metadata = load_run(run_dir)

    if not hist:
        print("  ERROR: No history data found. Aborting.")
        return
    if not snaps:
        print("  WARNING: No snapshot data found. Field-based plots will be skipped.")

    # Re-render phi fields from particle positions if requested
    if rerender and snaps and 'x' in snaps[0]:
        print("  Re-rendering phi fields from particle positions...")
        for si, snap in enumerate(snaps):
            pf, pi, pv = _rerender_snapshot(snap, p)
            snap['phi_f'] = pf
            snap['phi_i'] = pi
            snap['phi_v'] = pv
        print(f"    Re-rendered {len(snaps)} snapshots"
              f" at {snaps[0]['phi_f'].shape[0]}x{snaps[0]['phi_f'].shape[1]} grid")

    # Resolve the actual directory (in case of .tar.gz)
    actual_dir = run_dir[:-7] if run_dir.endswith('.tar.gz') else run_dir
    if outdir is None:
        outdir = os.path.join(actual_dir, 'plots')
    os.makedirs(outdir, exist_ok=True)

    n_snaps = len(snaps)
    n_hist = len(hist)
    mode = getattr(p, 'mode', metadata.get('mode', '2D'))
    print(f"  Mode: {mode}")
    print(f"  Snapshots: {n_snaps}, History entries: {n_hist}")
    print(f"  Output: {outdir}/")

    has_fields = snaps and 'phi_f' in snaps[0]

    # --- Built-in plots ---
    if snaps and has_fields and 'granules' not in skip:
        print("\n  [1/17] Granule positions...")
        fig = plot_granules(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'granules.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if snaps and has_fields and 'fields' not in skip:
        print("  [2/17] Phase fields...")
        fig = plot_fields(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'fields.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if 'timeseries' not in skip:
        print("  [3/17] Timeseries...")
        fig = plot_timeseries(hist, p)
        fig.savefig(os.path.join(outdir, 'timeseries.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if snaps and 'composite' not in skip:
        print("  [4/17] Scaffold map (vector)...")
        fig = plot_scaffold_map(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'scaffold_map.png'), dpi=200,
                    bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
        # Also save legacy composite if phi fields available
        if has_fields:
            fig = plot_composite(snaps, hist, p)
            fig.savefig(os.path.join(outdir, 'composite.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

    # --- LS-DEM deformation plots (V2.3) ---
    if snaps and 'deformation' not in skip:
        print("  [5/17] Deformation map...")
        fig = plot_deformation(snaps, hist, p)
        if fig is not None:
            fig.savefig(os.path.join(outdir, 'deformation.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)
        else:
            print("    SKIPPED (no deformation data)")

    if 'deformation' not in skip:
        print("  [6/17] Deformation timeseries...")
        fig = plot_deformation_timeseries(hist, p)
        if fig is not None:
            fig.savefig(os.path.join(outdir, 'deformation_timeseries.png'),
                        dpi=150, bbox_inches='tight')
            plt.close(fig)
        else:
            print("    SKIPPED (no deformation metrics)")

    # --- External visualization scripts ---
    if 'compaction' not in skip:
        print("  [7/17] Compaction...")
        try:
            from viz import compaction as viz_compaction
            viz_compaction.run_all(hist, snaps=snaps if snaps else None, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if 'percolation' not in skip:
        print("  [8/17] Percolation...")
        try:
            from viz import percolation as viz_percolation
            viz_percolation.run_all(hist, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'movies' not in skip:
        print("  [9/17] Movies...")
        try:
            from viz import movies as viz_movies
            viz_movies.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if 'phases' not in skip:
        print("  [10/17] Phases...")
        try:
            from viz import phases as viz_phases
            viz_phases.run_all(hist, snaps=snaps if snaps else None, p=p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'cells' not in skip:
        print("  [11/17] Cells...")
        try:
            from viz import cells as viz_cells
            viz_cells.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'stress' not in skip:
        print("  [12/17] Stress...")
        try:
            from viz import stress as viz_stress
            viz_stress.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    # --- Mathematical analysis modules (V1.7) ---
    if 'mean_field' not in skip:
        print("  [13/17] Mean-field model...")
        try:
            from analysis import mean_field_model
            mean_field_model.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'coarse_grain' not in skip:
        print("  [14/17] Coarse-graining...")
        try:
            from analysis import coarse_grain
            coarse_grain.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and has_fields and 'tissue' not in skip:
        print("  [15/17] Tissue descriptors...")
        try:
            from analysis import tissue_descriptors
            tissue_descriptors.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and has_fields and 'arch_distance' not in skip:
        print("  [16/17] Architectural distance...")
        try:
            from analysis import arch_distance
            arch_distance.run_all(run_dir=actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and has_fields and 'conservation' not in skip:
        print("  [17/17] Volume conservation...")
        try:
            from analysis import volume_conservation
            volume_conservation.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    # --- Summary ---
    h0, hf = hist[0], hist[-1]
    print("\n" + "=" * 65)
    print("  POST-PROCESSING COMPLETE")
    print("=" * 65)
    print(f"  Plots saved to: {outdir}/")
    print(f"  Time range: {h0['time']:.1f} → {hf['time']:.1f} h")
    print(f"  Functional clusters: {h0['func_nc']} → {hf['func_nc']}")
    print(f"  Bridges: {h0['n_bridges']} → {hf['n_bridges']}")
    print(f"  Porosity: {h0['porosity']:.3f} → {hf['porosity']:.3f}")


# ══════════════════════════════════════════════════════════════════════
# Batch auto-processing
# ══════════════════════════════════════════════════════════════════════

def _is_processed(tar_path):
    """Check if a .tar.gz run already has a plots/ directory with timeseries.png."""
    run_dir = tar_path[:-7]  # strip .tar.gz
    plots_dir = os.path.join(run_dir, 'plots')
    return os.path.isfile(os.path.join(plots_dir, 'timeseries.png'))


def find_unprocessed(scan_dir):
    """Find all .tar.gz files in scan_dir (recursively) that lack a plots/ directory.

    Returns a sorted list of paths.
    """
    pattern = os.path.join(scan_dir, '**', '*.tar.gz')
    all_archives = sorted(set(globmod.glob(pattern, recursive=True)))
    return [p for p in all_archives if not _is_processed(p)]


def run_all_unprocessed(scan_dir, skip=None):
    """Find and process all unprocessed .tar.gz archives in scan_dir.

    Parameters
    ----------
    scan_dir : str
        Directory to scan recursively for .tar.gz files.
    skip : set of str or None
        Visualization modules to skip.
    """
    pattern = os.path.join(scan_dir, '**', '*.tar.gz')
    all_archives = sorted(set(globmod.glob(pattern, recursive=True)))
    unprocessed = [p for p in all_archives if not _is_processed(p)]
    already_done = len(all_archives) - len(unprocessed)

    print("=" * 65)
    print("  GELS Batch Post-Processing")
    print("=" * 65)
    print(f"\n  Scan directory: {scan_dir}")
    print(f"  Total .tar.gz archives found: {len(all_archives)}")
    print(f"  Already processed (have plots/): {already_done}")
    print(f"  To process: {len(unprocessed)}")

    if not unprocessed:
        print("\n  Nothing to do — all runs already have plots.")
        return

    print()
    for i, tar_path in enumerate(unprocessed, 1):
        name = os.path.basename(tar_path).replace('.tar.gz', '')
        print(f"\n{'─' * 65}")
        print(f"  [{i}/{len(unprocessed)}] {name}")
        print(f"{'─' * 65}")
        try:
            run_all(tar_path, skip=skip)
        except Exception as e:
            print(f"  ERROR processing {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'═' * 65}")
    print(f"  BATCH COMPLETE — processed {len(unprocessed)} runs")
    print(f"{'═' * 65}")

    # Auto-run DOE analysis if we processed DOE runs
    doe_names = [os.path.basename(p).replace('.tar.gz', '')
                 for p in all_archives if 'DOE_' in os.path.basename(p)]
    if len(doe_names) >= 16:
        print(f"\n  Detected {len(doe_names)} DOE runs — running DOE analysis...")
        try:
            from viz import doe as viz_doe
            data = viz_doe.load_doe_data(scan_dir)
            if data['runs']:
                doe_outdir = os.path.join(scan_dir, 'doe_analysis')
                viz_doe.run_all(data, outdir=doe_outdir, skip=skip)
        except Exception as e:
            print(f"  DOE analysis error: {e}")

        # Dimensionless analysis across DOE runs (V1.7)
        if 'dimensionless' not in (skip or set()):
            print(f"\n  Running dimensionless analysis across DOE runs...")
            try:
                from viz import dimensionless as viz_dimensionless
                doe_outdir = os.path.join(scan_dir, 'doe_analysis')
                viz_dimensionless.run_all(scan_dir=scan_dir, outdir=doe_outdir)
            except Exception as e:
                print(f"  Dimensionless analysis error: {e}")

        # Architectural distance across DOE runs (V1.7)
        if 'arch_distance' not in (skip or set()):
            print(f"\n  Running architectural distance analysis...")
            try:
                from analysis import arch_distance
                doe_outdir = os.path.join(scan_dir, 'doe_analysis')
                arch_distance.run_all(scan_dir=scan_dir, outdir=doe_outdir)
            except Exception as e:
                print(f"  Architectural distance error: {e}")


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════

_DEFAULT_SCAN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'results', 'trials')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GELS post-processing: load saved data and produce all visualizations.'
                    ' With no -i, auto-discovers and processes all unprocessed .tar.gz files.')
    parser.add_argument('-i', '--input', default=None,
                        help='Path to a single simulation output directory or .tar.gz archive')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for plots (default: <input>/plots)')
    parser.add_argument('--scan-dir', default=None,
                        help=f'Directory to scan for .tar.gz files (default: results/trials/)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Visualization modules to skip (e.g., movies stress)')
    parser.add_argument('--list', action='store_true',
                        help='List unprocessed archives without running visualizations')
    parser.add_argument('--rerender', action='store_true',
                        help='Re-render phi fields from saved particle positions'
                             ' (fixes resolution/aliasing without re-running simulation)')
    args = parser.parse_args()

    skip = set(args.skip)

    if args.input:
        # Single-run mode
        run_all(args.input, outdir=args.output, skip=skip, rerender=args.rerender)
    else:
        # Auto-discovery mode
        scan_dir = args.scan_dir or _DEFAULT_SCAN_DIR
        if not os.path.isdir(scan_dir):
            print(f"ERROR: Scan directory not found: {scan_dir}")
            sys.exit(1)

        if args.list:
            unprocessed = find_unprocessed(scan_dir)
            print(f"Unprocessed archives in {scan_dir}:")
            for p in unprocessed:
                print(f"  {os.path.relpath(p)}")
            print(f"\nTotal: {len(unprocessed)}")
        else:
            run_all_unprocessed(scan_dir, skip=skip)
