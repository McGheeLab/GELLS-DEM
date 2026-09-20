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

from viz2.parallel import pmap
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


def plot_stress_maps(snaps, hist, p, indices=None, Ngrid=20, outdir=None,
                     workers=None):
    """Spatial pressure and von Mises stress maps at selected timepoints."""
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(2, nc, figsize=(4.5 * nc, 9))
    if nc == 1:
        axes = axes.reshape(2, 1)

    per_snap = pmap(_coarse_grained,
                    [(snaps[si], p, 'stress', Ngrid) for si in indices],
                    workers, label='stress maps')

    for col, si in enumerate(indices):
        t = get_snap_time(hist, si)

        result = per_snap[col]
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


def plot_strain_maps(snaps, hist, p, indices=None, Ngrid=20, outdir=None,
                     workers=None):
    """Spatial strain rate maps at selected timepoints."""
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(2, nc, figsize=(4.5 * nc, 9))
    if nc == 1:
        axes = axes.reshape(2, 1)

    per_snap = pmap(_coarse_grained,
                    [(snaps[si], p, 'strain_rate', Ngrid) for si in indices],
                    workers, label='strain maps')

    for col, si in enumerate(indices):
        t = get_snap_time(hist, si)

        result = per_snap[col]
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
    """Cell strain energy, nN*um:  U = F^2 / 2k,  1/k = 1/k_cell + 1/k_sub.

    V3.7. This used to be ``0.5 * |F_cell| * bridge_length`` -- the work of
    dragging a granule the whole length of a bridge, which scales with how far
    apart the granules happen to be rather than with how hard the cell pulls.
    The engine has had the real quantity since V3.6, and it is the one traction
    force microscopy reports (1 pJ = 1000 nN*um), so it is now shared rather
    than re-invented here.
    """
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    cell_states = snap.get('cell_state', np.array([], dtype=int))
    n_cells = len(cell_states)
    if n_cells == 0:
        return field

    from gels.stress import snapshot_cell_strain_energy
    U = snapshot_cell_strain_energy(snap, p)
    if U.size != n_cells:
        return field

    wx, wy = cell_world_positions(snap)
    loaded = U > 0
    if not np.any(loaded):
        return field
    _gaussian_smear(field, gx, gy, np.asarray(wx)[loaded], np.asarray(wy)[loaded],
                    U[loaded], w)
    return field


def _compute_contact_field(snap, p, gx, gy, w):
    """JKR contact energy, nN*um -- the engine's own, per pair.

    V3.7. What stood here was ``(2/5) * E* * sqrt(R) * d^2.5`` with E* from the
    GLOBAL ``p.E_modulus``, and it was wrong three ways that compounded to a
    factor of 790 on a hydrogel bed:

      * the engine's ``E_star`` is in Pa and every force expression that uses it
        carries an explicit ``1e-3`` to reach nN/um^2. Omitting it is x1000.
      * integrating ``F = (4/3) E* sqrt(R) d^{3/2}`` gives ``(8/15)``, not
        ``(2/5)``. That is x0.75.
      * the contact is JKR, not Hertz, and the adhesive terms matter most at
        shallow overlap, where most contacts live.

    and a fourth that is worse on a mixed bed: a global modulus ignores the
    per-species value and ``contact.E_cap``, which on the PMMA preset is a
    factor of 3e4. `gels.stress.snapshot_contact_energy` uses the per-pair
    columns the snapshot has carried since V3.7, and falls back -- visibly --
    for older runs.
    """
    Ngrid = len(gx)
    field = np.zeros((Ngrid, Ngrid))

    ci_arr = snap.get('contact_i', np.array([], dtype=int))
    if len(ci_arr) == 0:
        return field

    from gels.stress import snapshot_contact_energy
    E_c, _exact = snapshot_contact_energy(snap, p)
    cx = np.asarray(snap.get('contact_cx', np.zeros(len(ci_arr))))
    cy = np.asarray(snap.get('contact_cy', np.zeros(len(ci_arr))))
    overlap = np.asarray(snap.get('contact_overlap', np.zeros(len(ci_arr))))
    keep = (overlap > 0) & np.isfinite(E_c)
    if not np.any(keep):
        return field
    _gaussian_smear(field, gx, gy, cx[keep], cy[keep], E_c[keep], w)
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

    # V3.0: pair friction from the coverage mixing rule (equals the old
    # ff / if / ii lookup for f in {0, 1}, i.e. for legacy snapshots)
    from viz2.snapshot_ops import contact_friction
    tau_arr = contact_friction(snap, p, ci_arr, cj_arr)

    src_x, src_y, vals = [], [], []
    for k in range(len(ci_arr)):
        i, j = int(ci_arr[k]), int(cj_arr[k])
        if i >= len(vx) or j >= len(vx):
            continue
        # Relative velocity magnitude
        dvx = vx[j] - vx[i]
        dvy = vy[j] - vy[i]
        v_rel = np.sqrt(dvx**2 + dvy**2)

        tau = float(tau_arr[k])

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
    """The share of the contact energy carried at inert-involving contacts.

    V3.7. This was ``|F_normal| * overlap``, which is neither the stored energy
    (for Hertz that is ``(2/5) F d``) nor independent of the contact panel: it
    is the SAME contacts, scored differently. It is now literally the contact
    energy restricted to inert-functional and inert-inert pairs, so the panel
    is an honest subset of "JKR contact" rather than a sixth mode that
    double-counts it. Read the two together, not as a sum.
    """
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
    from gels.stress import snapshot_contact_energy
    E_contact, _exact = snapshot_contact_energy(snap, p)

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
            E_frust = E_contact[k]
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

def plot_energy_modes(snaps, hist, p, indices=None, Ngrid=30, outdir=None,
                      workers=None):
    """2x3 panel figure of all energy mode spatial maps.

    Layout:
      [Cell traction | JKR contact | Friction]
      [Osmotic       | Inert frust | Interfacial]

    Produces one figure per selected timepoint.
    """
    indices = select_indices(len(snaps), indices, n_panels=3)
    modes = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
             'interfacial']
    # V3.7: each panel carries its own units. Three of these six are not
    # energies at all -- friction is a dissipation RATE, and osmotic and
    # interfacial have no energy scale (their coefficient is
    # `E_modulus * interface_width * 0.01`, which has no source). They are
    # informative maps and stay, but the figure no longer implies that the
    # reader may compare their magnitudes.
    mode_labels = ['Cell strain energy\n(nN*um)',
                   'JKR contact energy\n(nN*um)',
                   'Friction dissipation RATE\n(nN*um/h)',
                   'Osmotic indicator\n(|grad phi_v|^2/phi_v, um^-2)',
                   'Contact energy at inert pairs\n(nN*um, subset of panel 2)',
                   'Interfacial indicator\n(arbitrary scale)']
    cmaps = ['Greens', 'Blues', 'Reds', 'Purples', 'YlOrBr', 'Oranges']

    per_snap = pmap(_energy_fields, [(snaps[si], p, Ngrid) for si in indices],
                    workers, label='energy mode maps')

    figs = []
    for row_i, si in enumerate(indices):
        t = get_snap_time(hist, si)

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        extent = [0, p.Lx, 0, p.Ly]

        for idx, (mode, label, cmap_name) in enumerate(
                zip(modes, mode_labels, cmaps)):
            row, col = divmod(idx, 3)
            ax = axes[row, col]

            field = per_snap[row_i][mode]

            vmax = np.percentile(field, 99) if field.max() > 0 else 1.0
            im = ax.imshow(field.T, origin='lower', extent=extent,
                           cmap=cmap_name, vmin=0, vmax=max(vmax, 1e-10),
                           aspect='equal')
            ax.set_title(label, fontsize=10, color=ENERGY_COLORS.get(mode, 'k'))
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlim(0, p.Lx)
            ax.set_ylim(0, p.Ly)

        fig.suptitle(f"Energy Mode Spatial Maps | t = {t:.1f} h"
                     "   (panels are in DIFFERENT units -- see each title)",
                     fontsize=13, y=1.02)
        plt.tight_layout()
        figs.append(fig)

        if outdir:
            save_fig(fig, outdir, f'energy_modes_t{t:.0f}.png')

    return figs


# ======================================================================
# Per-snapshot workers (module level: a pool worker imports them by name)
# ======================================================================

ENERGY_MODES = ['traction', 'contact', 'friction', 'osmotic', 'frustration',
                'interfacial']


def _energy_fields(args):
    """All six energy-mode fields for one snapshot (small grids, cheap to ship)."""
    snap, p, Ngrid = args
    return {mode: compute_energy_field(snap, p, mode, Ngrid)[0]
            for mode in ENERGY_MODES}


def _coarse_grained(args):
    """Coarse-grained stress or strain-rate result for one snapshot (or None)."""
    snap, p, quantity, Ngrid = args
    return _try_coarse_grain(snap, p, quantity, Ngrid)


# ======================================================================
# Energy timeseries
# ======================================================================

def plot_energy_timeseries(snaps, hist, p, Ngrid=30, outdir=None, workers=None):
    """Total energy per mode vs time."""
    # V3.7: only the two genuine, independent energies go on the energy axis.
    # Before this, a power (friction), a um^-2 indicator (osmotic), an
    # unscaled one (interfacial) and a double count of the contact energy
    # (frustration) were plotted on a single axis labelled 'Total Energy
    # (nN*um)', which cannot answer the question the figure exists to ask.
    modes = ['traction', 'contact']
    mode_labels = ['Cell strain energy', 'JKR contact energy']

    times = [get_snap_time(hist, si) for si in range(len(snaps))]
    totals = {m: [] for m in modes}

    per_snap = pmap(_energy_fields,
                    [(snaps[si], p, Ngrid) for si in range(len(snaps))],
                    workers, label='energy timeseries')
    pixel_area = (p.Lx / Ngrid) * (p.Ly / Ngrid)
    for fields in per_snap:
        for mode in modes:
            totals[mode].append(float(np.sum(fields[mode])) * pixel_area)

    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.array(times)

    for mode, label in zip(modes, mode_labels):
        c = ENERGY_COLORS.get(mode, 'k')
        ax.plot(t, totals[mode], '-', color=c, lw=2, label=label)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Total energy (nN*um)')
    ax.set_title('Stored elastic energy: cells vs granule contacts',
                 fontweight='bold')
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

def run_all(snaps, hist, p, outdir=None, workers=None):
    """Generate all energy/stress/strain visualizations."""
    print("  [5/6] Energy & stress maps...")

    fig1 = plot_stress_maps(snaps, hist, p, outdir=outdir, workers=workers)
    plt.close(fig1)

    fig2 = plot_strain_maps(snaps, hist, p, outdir=outdir, workers=workers)
    plt.close(fig2)

    figs = plot_energy_modes(snaps, hist, p, outdir=outdir, workers=workers)
    for f in figs:
        plt.close(f)

    fig4 = plot_energy_timeseries(snaps, hist, p, outdir=outdir, workers=workers)
    plt.close(fig4)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Energy & stress viz')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='Per-snapshot worker processes (default: auto, 1 = serial)')
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
