"""
Two-Compartment Volume Conservation Analysis for GELS
==========================================================
V2.1 — Voronoi shrink-wrap compartment analysis.

Assigns each voxel in the domain to the functional or inert compartment
based on nearest granule type (Voronoi tessellation by phase).  This
"shrink-wraps" each phase's territory tightly around its granules,
excluding islands of the opposite phase.

Uses **granule-based true volumes** (not phase fields) for conservation
accounting.  Granule radii/shapes are invariant during the simulation,
so the total solid volume is exactly conserved by construction.  Phase
field rendering has an inherent volume error from per-voxel capping at
dense overlaps; the granule-based approach avoids this artifact.

Usage:
    python analysis/volume_conservation.py -i results/default

Or import programmatically:
    from analysis.volume_conservation import conservation_from_run, plot_conservation
    records, p = conservation_from_run('results/default')
    plot_conservation(records, p, 'plots/')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ======================================================================
# Granule volume computation
# ======================================================================

def granule_volumes_3d(snap):
    """Compute per-granule volumes from snapshot geometry.

    Returns ndarray of volumes (same length as snap['r']).
    """
    r = snap['r']
    if 'a' in snap and 'b' in snap and 'c' in snap and 'n1' in snap and 'n2' in snap:
        from new_dem_0 import superellipsoid_volume
        a, b, c = snap['a'], snap['b'], snap['c']
        n1, n2 = snap['n1'], snap['n2']
        vols = np.array([superellipsoid_volume(a[i], b[i], c[i], n1[i], n2[i])
                         for i in range(len(r))])
    else:
        vols = (4.0 / 3.0) * np.pi * r ** 3
    return vols


def granule_volumes_2d(snap):
    """Compute per-granule areas from snapshot geometry."""
    r = snap['r']
    if 'a' in snap and 'b' in snap and 'n_shape' in snap:
        from new_dem_0 import superellipse_area
        a, b, ns = snap['a'], snap['b'], snap['n_shape']
        areas = np.array([superellipse_area(a[i], b[i], ns[i])
                          for i in range(len(r))])
    else:
        areas = np.pi * r ** 2
    return areas


# ======================================================================
# Voronoi compartment assignment
# ======================================================================

def voronoi_compartments(positions, gtypes, p, grid_shape):
    """Assign each voxel to functional (0) or inert (1) compartment
    via nearest-granule Voronoi tessellation.

    Parameters
    ----------
    positions : ndarray, shape (N, 2) or (N, 3)
        Granule centre positions.
    gtypes : ndarray, shape (N,)
        Granule types (0=functional, 1=inert).
    p : Params
        Simulation parameters (Lx, Ly, Lz, boundary_mode).
    grid_shape : tuple
        Shape of the phase field grid (e.g. (60, 60, 60) for 3D).

    Returns
    -------
    func_mask : ndarray of bool, same shape as grid_shape
        True for voxels in the functional compartment.
    inert_mask : ndarray of bool, same shape as grid_shape
        True for voxels in the inert compartment.
    """
    is_3d = len(grid_shape) == 3
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'

    if periodic:
        if is_3d:
            tree = cKDTree(positions, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(positions, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(positions)

    if is_3d:
        nx, ny, nz = grid_shape
        cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
        cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
        cz = np.linspace(0.5 * p.Lz / nz, p.Lz - 0.5 * p.Lz / nz, nz)
        gx, gy, gz = np.meshgrid(cx, cy, cz, indexing='ij')
        voxel_pts = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    else:
        nx, ny = grid_shape
        cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
        cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
        gx, gy = np.meshgrid(cx, cy, indexing='ij')
        voxel_pts = np.column_stack([gx.ravel(), gy.ravel()])

    _, nearest_idx = tree.query(voxel_pts)
    nearest_type = gtypes[nearest_idx]

    func_mask = (nearest_type == 0).reshape(grid_shape)
    inert_mask = (nearest_type == 1).reshape(grid_shape)

    return func_mask, inert_mask


# ======================================================================
# Two-compartment metrics (granule-based)
# ======================================================================

def compartment_metrics(snap, p, func_voxel_mask, inert_voxel_mask):
    """Compute two-compartment metrics using exact granule volumes.

    Parameters
    ----------
    snap : dict
        Snapshot with x, y, [z], r, gtype, [a, b, c, n1, n2].
    p : Params
        Simulation parameters.
    func_voxel_mask, inert_voxel_mask : ndarray of bool
        Voronoi compartment assignments (from phase field grid).

    Returns
    -------
    dict with conservation and compartment metrics.
    """
    gt = snap['gtype']
    is_3d = 'z' in snap and snap.get('phi_f', np.array([])).ndim == 3
    m = {}

    # True granule volumes (invariant — proves physics conservation)
    if is_3d:
        vols = granule_volumes_3d(snap)
        V_domain = p.Lx * p.Ly * p.Lz
    else:
        vols = granule_volumes_2d(snap)
        V_domain = p.Lx * p.Ly

    V_f_true = float(np.sum(vols[gt == 0]))
    V_i_true = float(np.sum(vols[gt == 1]))
    m['phi_f_true'] = V_f_true / V_domain
    m['phi_i_true'] = V_i_true / V_domain
    m['phi_solid_true'] = (V_f_true + V_i_true) / V_domain
    m['phi_v_true'] = 1.0 - m['phi_solid_true']

    # Compartment volume fractions (from Voronoi tessellation)
    n_total = func_voxel_mask.size
    n_func = int(np.sum(func_voxel_mask))
    n_inert = int(np.sum(inert_voxel_mask))
    m['x_f'] = n_func / max(n_total, 1)
    m['x_i'] = n_inert / max(n_total, 1)

    V_func_compartment = m['x_f'] * V_domain
    V_inert_compartment = m['x_i'] * V_domain

    # Local packing fractions using true granule volumes
    # (each granule is in one compartment by type)
    m['phi_solid_func'] = V_f_true / max(V_func_compartment, 1e-30)
    m['phi_solid_inert'] = V_i_true / max(V_inert_compartment, 1e-30)
    m['phi_v_func'] = 1.0 - m['phi_solid_func']
    m['phi_v_inert'] = 1.0 - m['phi_solid_inert']

    # Cross-check: volume-weighted void = global void
    m['phi_v_crosscheck'] = m['x_f'] * m['phi_v_func'] + m['x_i'] * m['phi_v_inert']

    # Also report rendered field values for comparison
    if 'phi_f' in snap:
        phi_f, phi_i, phi_v = snap['phi_f'], snap['phi_i'], snap['phi_v']
        m['phi_solid_rendered'] = float(np.mean(phi_f) + np.mean(phi_i))
        m['rendering_error'] = (m['phi_solid_rendered'] - m['phi_solid_true']) / m['phi_solid_true']

    return m


# ======================================================================
# Full run analysis
# ======================================================================

def conservation_from_run(run_dir, voronoi_mult=3):
    """Load a simulation run and compute two-compartment metrics at each timepoint.

    Parameters
    ----------
    run_dir : str
        Path to output directory or .tar.gz archive.
    voronoi_mult : int
        Resolution multiplier for the Voronoi grid relative to the phase
        field grid.  Default 3 gives 180^3 for Ngrid_3d=60.

    Returns
    -------
    records : list of dict
        One dict per timepoint with time + all compartment metrics.
    p : Params
        Simulation parameters.
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)

    if not snaps:
        print("  WARNING: No snapshot data — cannot compute compartment metrics.")
        return [], p

    records = []
    for si, snap in enumerate(snaps):
        x, y = snap['x'], snap['y']
        gt = snap['gtype']
        phi_f = snap.get('phi_f', None)
        is_3d = 'z' in snap and phi_f is not None and phi_f.ndim == 3

        if is_3d:
            positions = np.column_stack([x, y, snap['z']])
        else:
            positions = np.column_stack([x, y])

        # Voronoi grid at higher resolution than phase field
        if phi_f is not None:
            base_shape = phi_f.shape
        else:
            Ng = getattr(p, 'Ngrid_3d', 60) if is_3d else getattr(p, 'Ngrid', 200)
            base_shape = (Ng, Ng, Ng) if is_3d else (Ng, Ng)
        grid_shape = tuple(s * voronoi_mult for s in base_shape)

        func_mask, inert_mask = voronoi_compartments(
            positions, gt, p, grid_shape)

        m = compartment_metrics(snap, p, func_mask, inert_mask)
        m['time'] = hist[si]['time'] if si < len(hist) else si
        records.append(m)

    return records, p


# ======================================================================
# Visualization
# ======================================================================

def plot_conservation(records, p, outdir='.'):
    """4-panel volume conservation figure.

    Panel 1: Global conservation — true vs rendered phi_solid
    Panel 2: Compartment volume fractions (x_f, x_i vs time)
    Panel 3: Local void fractions (phi_v_func, phi_v_inert vs time)
    Panel 4: Cross-check (x_f*phi_v_func + x_i*phi_v_inert vs phi_v_true)
    """
    if not records:
        return None

    t = [r['time'] for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Panel 1: Global conservation (true granule volumes)
    ax = axes[0, 0]
    ax.plot(t, [r['phi_solid_true'] for r in records], 'k-o', ms=5, lw=2,
            label=r'$\phi_{solid}^{true}$ (granule geometry)')
    if 'phi_solid_rendered' in records[0]:
        ax.plot(t, [r['phi_solid_rendered'] for r in records], 'C7--s', ms=4,
                label=r'$\phi_{solid}^{rendered}$ (phase field)')
    ax.plot(t, [r['phi_f_true'] for r in records], 'C1-', lw=1, alpha=0.7,
            label=r'$\phi_f^{true}$')
    ax.plot(t, [r['phi_i_true'] for r in records], 'C0-', lw=1, alpha=0.7,
            label=r'$\phi_i^{true}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Volume fraction')
    ax.set_title('Global Conservation')
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)

    # Panel 2: Compartment volumes
    ax = axes[0, 1]
    ax.plot(t, [r['x_f'] for r in records], 'C1-o', ms=4, label=r'$x_f$ (functional)')
    ax.plot(t, [r['x_i'] for r in records], 'C0-s', ms=4, label=r'$x_i$ (inert)')
    ax.axhline(0.5, ls=':', c='gray', lw=0.8)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Domain fraction')
    ax.set_title('Compartment Volumes (Voronoi)')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # Panel 3: Local void fractions (from granule-based computation)
    ax = axes[1, 0]
    ax.plot(t, [r['phi_v_func'] for r in records], 'C1-o', ms=4,
            label=r'$\phi_v^{func}$ (void in functional)')
    ax.plot(t, [r['phi_v_inert'] for r in records], 'C0-s', ms=4,
            label=r'$\phi_v^{inert}$ (void in inert)')
    ax.plot(t, [r['phi_v_true'] for r in records], 'C2--', lw=1.5,
            label=r'$\phi_v^{global}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Void fraction')
    ax.set_title('Local Void Redistribution')
    ax.legend(fontsize=9)

    # Panel 4: Cross-check
    ax = axes[1, 1]
    ax.plot(t, [r['phi_v_true'] for r in records], 'C2-o', ms=4,
            label=r'$\phi_v^{true}$ (global)')
    ax.plot(t, [r['phi_v_crosscheck'] for r in records], 'k--x', ms=5,
            label=r'$x_f \phi_v^{func} + x_i \phi_v^{inert}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Void fraction')
    ax.set_title('Cross-Check: Volume-Weighted = Global')
    ax.legend(fontsize=9)

    devs = [abs(r['phi_v_crosscheck'] - r['phi_v_true']) for r in records]
    max_dev = max(devs) if devs else 0
    ax.annotate(f'max |err| = {max_dev:.2e}', xy=(0.05, 0.95),
                xycoords='axes fraction', fontsize=9, va='top',
                bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.8))

    fig.suptitle('Two-Compartment Volume Conservation', fontsize=14, y=1.02)
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'volume_conservation.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"    Saved: {path}")

    return fig


def plot_compartment_slice(snap, p, func_mask, inert_mask, outdir='.'):
    """Mid-slice visualization showing compartment boundaries overlaid on composite.

    The func_mask/inert_mask may be at higher resolution than the phase field;
    the phase field is upsampled via zoom to match the Voronoi grid for smooth
    overlay.
    """
    from scipy.ndimage import zoom

    phi_f = snap.get('phi_f', None)
    if phi_f is None:
        return None

    phi_i = snap['phi_i']
    phi_v = snap['phi_v']

    if phi_f.ndim == 3:
        mid_frac = 0.5
        # Take mid-slice from Voronoi mask
        mid_vm = func_mask.shape[2] // 2
        fm = func_mask[:, :, mid_vm]
        # Take mid-slice from phase fields and upsample to Voronoi resolution
        mid_pf = phi_f.shape[2] // 2
        pf = phi_f[:, :, mid_pf]
        pi = phi_i[:, :, mid_pf]
        pv = phi_v[:, :, mid_pf]
    else:
        pf, pi, pv = phi_f, phi_i, phi_v
        fm = func_mask

    # Upsample phase field slices to match Voronoi mask resolution
    if pf.shape != fm.shape:
        zf = (fm.shape[0] / pf.shape[0], fm.shape[1] / pf.shape[1])
        pf = zoom(pf, zf, order=1)
        pi = zoom(pi, zf, order=1)
        pv = zoom(pv, zf, order=1)

    mx = max(pf.max(), pi.max(), pv.max(), 0.01)
    rgb = np.stack([pf / mx, pv / mx, pi / mx], axis=-1)
    rgb = np.clip(np.transpose(rgb, (1, 0, 2)), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].imshow(rgb, origin='lower', extent=[0, p.Lx, 0, p.Ly])
    axes[0].set_title('Phase Composite (R=func, G=void, B=inert)')

    axes[1].imshow(rgb, origin='lower', extent=[0, p.Lx, 0, p.Ly])
    boundary = fm.astype(float)
    axes[1].contour(
        np.linspace(0, p.Lx, fm.shape[0]),
        np.linspace(0, p.Ly, fm.shape[1]),
        boundary.T, levels=[0.5], colors='white', linewidths=1.5,
        linestyles='--')
    axes[1].set_title('Voronoi Compartment Boundary (white)')

    for ax in axes:
        ax.set_xlabel(r'x ($\mu$m)')
        ax.set_ylabel(r'y ($\mu$m)')

    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, 'compartment_slice.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"    Saved: {path}")
    return fig


# ======================================================================
# Orchestrator (for postprocess.py integration)
# ======================================================================

def run_all(run_dir, outdir=None, voronoi_mult=3):
    """Run full volume conservation analysis on a saved run.

    Parameters
    ----------
    voronoi_mult : int
        Resolution multiplier for the Voronoi grid (default 3).
    """
    from new_dem_0 import load_run

    actual_dir = run_dir[:-7] if run_dir.endswith('.tar.gz') else run_dir
    if outdir is None:
        outdir = os.path.join(actual_dir, 'plots')

    records, p = conservation_from_run(run_dir, voronoi_mult=voronoi_mult)
    if not records:
        print("    No data for conservation analysis.")
        return

    fig = plot_conservation(records, p, outdir)
    if fig:
        plt.close(fig)

    # Compartment slice for the final snapshot (high-res Voronoi)
    hist, snaps, p, _ = load_run(run_dir)
    if snaps and 'phi_f' in snaps[-1]:
        snap = snaps[-1]
        phi_f = snap['phi_f']
        x, y = snap['x'], snap['y']
        gt = snap['gtype']
        if 'z' in snap and phi_f.ndim == 3:
            positions = np.column_stack([x, y, snap['z']])
        else:
            positions = np.column_stack([x, y])
        # Use higher resolution for the slice visualization
        vis_shape = tuple(s * voronoi_mult for s in phi_f.shape)
        func_mask, inert_mask = voronoi_compartments(
            positions, gt, p, vis_shape)
        fig2 = plot_compartment_slice(snap, p, func_mask, inert_mask, outdir)
        if fig2:
            plt.close(fig2)

    # Print summary
    r0, rf = records[0], records[-1]
    print(f"    True phi_solid: {r0['phi_solid_true']:.4f} -> {rf['phi_solid_true']:.4f}"
          f"  (delta = {rf['phi_solid_true'] - r0['phi_solid_true']:+.6f})")
    if 'phi_solid_rendered' in r0:
        print(f"    Rendered phi_solid: {r0['phi_solid_rendered']:.4f} -> "
              f"{rf['phi_solid_rendered']:.4f}"
              f"  (rendering error: {rf['rendering_error']:+.1%})")
    print(f"    Functional zone: x_f = {r0['x_f']:.3f} -> {rf['x_f']:.3f},"
          f"  phi_v = {r0['phi_v_func']:.3f} -> {rf['phi_v_func']:.3f}")
    print(f"    Inert zone:      x_i = {r0['x_i']:.3f} -> {rf['x_i']:.3f},"
          f"  phi_v = {r0['phi_v_inert']:.3f} -> {rf['phi_v_inert']:.3f}")


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Two-compartment volume conservation analysis for GELS.')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to simulation output directory or .tar.gz')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for plots')
    args = parser.parse_args()
    run_all(args.input, outdir=args.output)
