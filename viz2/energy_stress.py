"""
Energy Landscape & Stress/Strain Maps (Module 5)
==================================================
Spatial maps of stress, strain, and all 6 energy modes:
  - Cell traction
  - Hertz/JKR contact
  - Granular friction
  - Osmotic pressure
  - Inert frustration
  - Interfacial tension

Usage:
    python viz2/energy_stress.py -i results/default
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from viz2.common import (
    ENERGY_COLORS, select_indices, get_snap_time, setup_scaffold_axes,
    ensure_phase_fields, save_fig, cell_world_positions,
    is_3d, slice_snap_z_midplane, slice_field_z_midplane,
    cell_world_positions_3d,
)


# ======================================================================
# Gaussian coarse-graining kernel
# ======================================================================

def _gaussian_smear(field, gx, gy, source_x, source_y, values, w):
    """Add Gaussian-smeared point sources onto a 2D grid.

    Parameters
    ----------
    field : ndarray (Ng, Ng) — accumulated field (modified in-place)
    gx, gy : 1D arrays — grid coordinates
    source_x, source_y : 1D arrays — source positions
    values : 1D array — source magnitudes
    w : float — Gaussian width
    """
    Ng = len(gx)
    dx = gx[1] - gx[0] if Ng > 1 else 1.0
    hw = int(3 * w / dx) + 1  # half-width in grid cells

    for k in range(len(source_x)):
        if values[k] == 0:
            continue
        ix0 = max(0, int((source_x[k] - 3 * w) / dx))
        ix1 = min(Ng, int((source_x[k] + 3 * w) / dx) + 1)
        iy0 = max(0, int((source_y[k] - 3 * w) / dx))
        iy1 = min(Ng, int((source_y[k] + 3 * w) / dx) + 1)

        for ix in range(ix0, ix1):
            for iy in range(iy0, iy1):
                d2 = (gx[ix] - source_x[k])**2 + (gy[iy] - source_y[k])**2
                wt = np.exp(-d2 / (2 * w**2))
                field[ix, iy] += values[k] * wt


def _make_grid(p, Ngrid):
    """Create grid coordinates and Gaussian width."""
    dx = p.Lx / Ngrid
    dy = p.Ly / Ngrid
    gx = np.linspace(dx / 2, p.Lx - dx / 2, Ngrid)
    gy = np.linspace(dy / 2, p.Ly - dy / 2, Ngrid)
    return gx, gy


# ======================================================================
# Stress/strain maps (reuse analysis.coarse_grain)
# ======================================================================

def _try_coarse_grain(snap, p, quantity, Ngrid):
    """Try to use analysis.coarse_grain.coarse_grain_field. Returns dict or None.

    For 3D data, scalar fields in the result are sliced at z-midplane so they
    can be used directly with imshow.
    """
    try:
        from analysis.coarse_grain import coarse_grain_field
        result = coarse_grain_field(snap, p, quantity=quantity, Ngrid=Ngrid)
        # Slice 3D result fields to z-midplane for 2D display
        if result is not None:
            for key in list(result.keys()):
                val = result[key]
                if isinstance(val, np.ndarray) and val.ndim == 3:
                    iz = val.shape[2] // 2
                    result[key] = val[:, :, iz]
        return result
    except (ImportError, Exception):
        return None


def plot_stress_maps(snaps, hist, p, indices=None, Ngrid=20, outdir=None):
    """Spatial pressure and von Mises stress maps at selected timepoints."""
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(2, nc, figsize=(4.5 * nc, 9))
    if nc == 1:
        axes = axes.reshape(2, 1)

    for col, si in enumerate(indices):
        snap = snaps[si]
        t = get_snap_time(hist, si)

        result = _try_coarse_grain(snap, p, 'stress', Ngrid)
        if result is None:
            for row in range(2):
                axes[row, col].set_title(f"t={t:.1f}h (no contact data)")
            continue

        gx = result['grid_x']
        gy = result['grid_y']
        extent = [0, p.Lx, 0, p.Ly]

        # Pressure
        ax = axes[0, col]
        im = ax.imshow(result['pressure'].T, origin='lower', extent=extent,
                       cmap='RdBu_r', aspect='equal')
        ax.set_title(f"Pressure (kPa) | t={t:.1f}h", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Von Mises
        ax = axes[1, col]
        im = ax.imshow(result['von_mises'].T, origin='lower', extent=extent,
                       cmap='hot', aspect='equal')
        ax.set_title(f"von Mises (kPa) | t={t:.1f}h", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Stress Maps (Coarse-Grained)", fontsize=13, y=1.02)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'stress_maps.png')
    return fig


def plot_strain_maps(snaps, hist, p, indices=None, Ngrid=20, outdir=None):
    """Spatial strain rate maps at selected timepoints."""
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(2, nc, figsize=(4.5 * nc, 9))
    if nc == 1:
        axes = axes.reshape(2, 1)

    for col, si in enumerate(indices):
        snap = snaps[si]
        t = get_snap_time(hist, si)

        result = _try_coarse_grain(snap, p, 'strain_rate', Ngrid)
        if result is None:
            for row in range(2):
                axes[row, col].set_title(f"t={t:.1f}h (no velocity data)")
            continue

        extent = [0, p.Lx, 0, p.Ly]

        ax = axes[0, col]
        vsr = result.get('volumetric_strain_rate')
        if vsr is not None:
            im = ax.imshow(vsr.T, origin='lower', extent=extent,
                           cmap='PiYG', aspect='equal')
            ax.set_title(f"Vol. strain rate (1/h) | t={t:.1f}h", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[1, col]
        ssr = result.get('shear_strain_rate')
        if ssr is not None:
            im = ax.imshow(ssr.T, origin='lower', extent=extent,
                           cmap='magma', aspect='equal')
            ax.set_title(f"Shear strain rate (1/h) | t={t:.1f}h", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Strain Rate Maps (Coarse-Grained)", fontsize=13, y=1.02)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'strain_maps.png')
    return fig


# ======================================================================
# Energy mode field computations
# ======================================================================

def compute_energy_field(snap, p, mode, Ngrid=30):
    """Compute spatial energy density field for a given mode.

    For 3D data, point-source modes (traction, contact, friction, frustration)
    are computed from the z-midplane-sliced snapshot. Grid-based modes (osmotic,
    interfacial) are computed on the full field then z-sliced.

    Parameters
    ----------
    mode : str
        One of: 'traction', 'contact', 'friction', 'osmotic',
        'frustration', 'interfacial'.

    Returns
    -------
    field : ndarray (Ngrid, Ngrid)
    gx, gy : 1D arrays of grid coordinates
    """
    # For 3D data, use z-sliced snapshot for point-source modes
    _3d = is_3d(snap, p)
    work_snap = slice_snap_z_midplane(snap, p) if _3d else snap

    gx, gy = _make_grid(p, Ngrid)
    field = np.zeros((Ngrid, Ngrid))
    r_mean = float(np.mean(snap['r']))
    w = 1.5 * r_mean

    if mode == 'traction':
        field = _compute_traction_field(work_snap, p, gx, gy, w)
    elif mode == 'contact':
        field = _compute_contact_field(work_snap, p, gx, gy, w)
    elif mode == 'friction':
        field = _compute_friction_field(work_snap, p, gx, gy, w)
    elif mode == 'osmotic':
        field = _compute_osmotic_field(snap, p, Ngrid, slice_3d=_3d)
    elif mode == 'frustration':
        field = _compute_frustration_field(work_snap, p, gx, gy, w)
    elif mode == 'interfacial':
        field = _compute_interfacial_field(snap, p, Ngrid, slice_3d=_3d)

    return field, gx, gy


def _compute_traction_field(snap, p, gx, gy, w):
    """Cell traction energy: E = 0.5 * |F_cell| * bridge_length."""
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    cell_states = snap.get('cell_state', np.array([], dtype=int))
    n_cells = len(cell_states)
    if n_cells == 0:
        return field

    cell_fx = snap.get('cell_fx', np.zeros(n_cells))
    cell_fy = snap.get('cell_fy', np.zeros(n_cells))
    cell_bt = snap.get('cell_bridge_target', -np.ones(n_cells, dtype=int))

    wx, wy = cell_world_positions(snap)
    xs, ys, rs = snap['x'], snap['y'], snap['r']

    src_x, src_y, vals = [], [], []
    for ci in range(n_cells):
        f_mag = np.sqrt(cell_fx[ci]**2 + cell_fy[ci]**2)
        if f_mag < 1e-10:
            continue
        bt = int(cell_bt[ci])
        if bt >= 0 and bt < len(xs):
            bridge_len = np.sqrt((wx[ci] - xs[bt])**2 + (wy[ci] - ys[bt])**2)
        else:
            bridge_len = rs[int(snap['cell_granule_id'][ci])] if ci < len(snap.get('cell_granule_id', [])) else r_mean
        energy = 0.5 * f_mag * bridge_len
        src_x.append(wx[ci])
        src_y.append(wy[ci])
        vals.append(energy)

    if src_x:
        _gaussian_smear(field, gx, gy, np.array(src_x), np.array(src_y),
                        np.array(vals), w)
    return field


def _compute_contact_field(snap, p, gx, gy, w):
    """Hertz/JKR contact energy: E = (2/5)*E*sqrt(R_eff)*delta^(5/2)."""
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    ci_arr = snap.get('contact_i', np.array([], dtype=int))
    if len(ci_arr) == 0:
        return field

    cx = snap.get('contact_cx', np.zeros(len(ci_arr)))
    cy = snap.get('contact_cy', np.zeros(len(ci_arr)))
    overlap = snap.get('contact_overlap', np.zeros(len(ci_arr)))
    R_eff = snap.get('contact_R_eff', np.ones(len(ci_arr)) * 20.0)
    A_contact = snap.get('contact_A_contact', np.zeros(len(ci_arr)))

    E_mod = getattr(p, 'E_modulus', 10.0) * 1e3  # kPa -> Pa, but we're in nN/um^2 = kPa
    nu = getattr(p, 'poisson_ratio', 0.45)
    E_star = E_mod / (2.0 * (1.0 - nu**2))  # kPa

    src_x, src_y, vals = [], [], []
    for k in range(len(ci_arr)):
        if overlap[k] <= 0:
            continue
        # Hertz elastic energy: (2/5) * E* * sqrt(R_eff) * delta^(5/2) in nN*um
        E_hertz = (2.0 / 5.0) * E_star * np.sqrt(R_eff[k]) * overlap[k]**2.5
        src_x.append(cx[k])
        src_y.append(cy[k])
        vals.append(E_hertz)

    if src_x:
        _gaussian_smear(field, gx, gy, np.array(src_x), np.array(src_y),
                        np.array(vals), w)
    return field


def _compute_friction_field(snap, p, gx, gy, w):
    """Friction dissipation: P = tau_0 * A_contact * |v_relative|."""
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    ci_arr = snap.get('contact_i', np.array([], dtype=int))
    cj_arr = snap.get('contact_j', np.array([], dtype=int))
    if len(ci_arr) == 0:
        return field

    cx = snap.get('contact_cx', np.zeros(len(ci_arr)))
    cy = snap.get('contact_cy', np.zeros(len(ci_arr)))
    A_contact = snap.get('contact_A_contact', np.zeros(len(ci_arr)))
    F_normal = snap.get('contact_F_normal', np.zeros(len(ci_arr)))

    vx = snap.get('vx', np.zeros(len(snap['x'])))
    vy = snap.get('vy', np.zeros(len(snap['x'])))
    gt = snap['gtype']

    # tau_0 lookup by granule type pair
    tau_ff = getattr(p, 'tau_0_ff', 2000.0)
    tau_if = getattr(p, 'tau_0_if', 500.0)
    tau_ii = getattr(p, 'tau_0_ii', 50.0)

    src_x, src_y, vals = [], [], []
    for k in range(len(ci_arr)):
        i, j = int(ci_arr[k]), int(cj_arr[k])
        if i >= len(vx) or j >= len(vx):
            continue
        # Relative velocity magnitude
        dvx = vx[j] - vx[i]
        dvy = vy[j] - vy[i]
        v_rel = np.sqrt(dvx**2 + dvy**2)

        # tau_0 by type
        ti, tj = gt[i], gt[j]
        if ti == 0 and tj == 0:
            tau = tau_ff
        elif ti == 1 and tj == 1:
            tau = tau_ii
        else:
            tau = tau_if

        # Friction dissipation rate (Pa -> kPa: tau is in Pa, A is um^2)
        # tau [Pa] * A [um^2] * v [um/h] -> [Pa*um^3/h] = [nN*um/h]
        P_fric = tau * 1e-3 * A_contact[k] * v_rel  # nN*um/h
        src_x.append(cx[k])
        src_y.append(cy[k])
        vals.append(P_fric)

    if src_x:
        _gaussian_smear(field, gx, gy, np.array(src_x), np.array(src_y),
                        np.array(vals), w)
    return field


def _compute_osmotic_field(snap, p, Ngrid, slice_3d=False):
    """Osmotic pressure from void fraction gradients: Pi ~ |grad(phi_v)|^2 / phi_v."""
    phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)

    # 3D: slice to z-midplane first
    if slice_3d and phi_v.ndim == 3:
        phi_v = slice_field_z_midplane(phi_v)

    # Resample to Ngrid if different
    from scipy.ndimage import zoom
    if phi_v.shape[0] != Ngrid:
        zf = Ngrid / phi_v.shape[0]
        phi_v = zoom(phi_v, zf, order=1)

    dx = p.Lx / Ngrid
    grad_y, grad_x = np.gradient(phi_v, dx)
    grad_mag2 = grad_x**2 + grad_y**2

    Pi = grad_mag2 / np.maximum(phi_v, 0.01)
    return Pi


def _compute_frustration_field(snap, p, gx, gy, w):
    """Inert frustration: F_normal * overlap at inert-functional contacts."""
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    ci_arr = snap.get('contact_i', np.array([], dtype=int))
    cj_arr = snap.get('contact_j', np.array([], dtype=int))
    if len(ci_arr) == 0:
        return field

    cx = snap.get('contact_cx', np.zeros(len(ci_arr)))
    cy = snap.get('contact_cy', np.zeros(len(ci_arr)))
    overlap = snap.get('contact_overlap', np.zeros(len(ci_arr)))
    F_normal = snap.get('contact_F_normal', np.zeros(len(ci_arr)))
    gt = snap['gtype']

    src_x, src_y, vals = [], [], []
    for k in range(len(ci_arr)):
        i, j = int(ci_arr[k]), int(cj_arr[k])
        if i >= len(gt) or j >= len(gt):
            continue
        # Frustration = energy at inert-functional or inert-inert contacts
        # near functional zones
        ti, tj = gt[i], gt[j]
        is_mixed = (ti == 0 and tj == 1) or (ti == 1 and tj == 0)
        is_ii = (ti == 1 and tj == 1)

        if is_mixed or is_ii:
            E_frust = abs(F_normal[k]) * max(overlap[k], 0)
            src_x.append(cx[k])
            src_y.append(cy[k])
            vals.append(E_frust)

    if src_x:
        _gaussian_smear(field, gx, gy, np.array(src_x), np.array(src_y),
                        np.array(vals), w)
    return field


def _compute_interfacial_field(snap, p, Ngrid, slice_3d=False):
    """Interfacial tension: gamma * (|grad(phi_f)|^2 + |grad(phi_i)|^2)."""
    phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)

    # 3D: slice to z-midplane first
    if slice_3d and phi_f.ndim == 3:
        phi_f = slice_field_z_midplane(phi_f)
        phi_i = slice_field_z_midplane(phi_i)

    from scipy.ndimage import zoom
    if phi_f.shape[0] != Ngrid:
        zf = Ngrid / phi_f.shape[0]
        phi_f = zoom(phi_f, zf, order=1)
        phi_i = zoom(phi_i, zf, order=1)

    dx = p.Lx / Ngrid

    gfy, gfx = np.gradient(phi_f, dx)
    giy, gix = np.gradient(phi_i, dx)

    iw = getattr(p, 'interface_width', 3.0)
    E_mod = getattr(p, 'E_modulus', 10.0)
    gamma = E_mod * iw * 0.01

    E_interface = gamma * (gfx**2 + gfy**2 + gix**2 + giy**2)
    return E_interface


# ======================================================================
# 6-panel energy mode figure
# ======================================================================

def plot_energy_modes(snaps, hist, p, indices=None, Ngrid=30, outdir=None):
    """2x3 panel figure of all energy mode spatial maps.

    Layout:
      [Cell traction | JKR contact | Friction]
      [Osmotic       | Inert frust | Interfacial]

    Produces one figure per selected timepoint.
    """
    indices = select_indices(len(snaps), indices, n_panels=3)
    modes = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
             'interfacial']
    mode_labels = ['Cell Traction', 'Hertz/JKR Contact', 'Granular Friction',
                   'Osmotic Pressure', 'Inert Frustration', 'Interfacial Tension']
    cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'YlOrBr', 'Oranges']

    figs = []
    for si in indices:
        snap = snaps[si]
        t = get_snap_time(hist, si)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        extent = [0, p.Lx, 0, p.Ly]

        for idx, (mode, label, cmap_name) in enumerate(
                zip(modes, mode_labels, cmaps)):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            field, gx, gy = compute_energy_field(snap, p, mode, Ngrid)

            vmax = np.percentile(field, 99) if field.max() > 0 else 1.0
            im = ax.imshow(field.T, origin='lower', extent=extent,
                           cmap=cmap_name, vmin=0, vmax=max(vmax, 1e-10),
                           aspect='equal')
            ax.set_title(label, fontsize=10, color=ENERGY_COLORS.get(mode, 'k'))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlim(0, p.Lx)
            ax.set_ylim(0, p.Ly)

        fig.suptitle(f"Energy Mode Spatial Maps | t = {t:.1f} h",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        figs.append(fig)

        if outdir:
            save_fig(fig, outdir, f'energy_modes_t{t:.0f}.png')

    return figs


# ======================================================================
# Energy timeseries
# ======================================================================

def plot_energy_timeseries(snaps, hist, p, Ngrid=30, outdir=None):
    """Total energy per mode vs time."""
    modes = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
             'interfacial']
    mode_labels = ['Cell Traction', 'Hertz/JKR', 'Friction',
                   'Osmotic', 'Inert Frust.', 'Interfacial']

    times = []
    totals = {m: [] for m in modes}

    for si in range(len(snaps)):
        t = get_snap_time(hist, si)
        times.append(t)
        snap = snaps[si]

        for mode in modes:
            field, _, _ = compute_energy_field(snap, p, mode, Ngrid)
            pixel_area = (p.Lx / Ngrid) * (p.Ly / Ngrid)
            totals[mode].append(float(np.sum(field)) * pixel_area)

    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.array(times)

    for mode, label in zip(modes, mode_labels):
        c = ENERGY_COLORS.get(mode, 'k')
        ax.plot(t, totals[mode], '-', color=c, lw=2, label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Total Energy (nN*um)')
    ax.set_title('Energy Mode Evolution', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'energy_timeseries.png')
    return fig


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate all energy/stress/strain visualizations."""
    print("  [5/6] Energy & stress maps...")

    fig1 = plot_stress_maps(snaps, hist, p, outdir=outdir)
    plt.close(fig1)

    fig2 = plot_strain_maps(snaps, hist, p, outdir=outdir)
    plt.close(fig2)

    figs = plot_energy_modes(snaps, hist, p, outdir=outdir)
    for f in figs:
        plt.close(f)

    fig4 = plot_energy_timeseries(snaps, hist, p, outdir=outdir)
    plt.close(fig4)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energy & stress viz')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
