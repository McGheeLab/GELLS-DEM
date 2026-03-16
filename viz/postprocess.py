"""
Unified Post-Processing & Visualization for GELLS-DEM
=====================================================
V1.6 — Loads saved simulation data and produces all visualizations.

Usage:
    # Process a single run:
    python viz/postprocess.py -i results/default
    python viz/postprocess.py -i results/default.tar.gz
    python viz/postprocess.py -i results/default --skip movies stress

    # Auto-process ALL unprocessed .tar.gz files in a directory:
    python viz/postprocess.py                          # scans Trials/trials/
    python viz/postprocess.py --scan-dir path/to/runs  # custom scan directory
    python viz/postprocess.py --skip movies stress      # auto-scan with skips

Or import programmatically:
    from viz.postprocess import run_all, run_all_unprocessed
    run_all('results/default')
    run_all_unprocessed('Trials/trials/')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
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


# ══════════════════════════════════════════════════════════════════════
# Orchestrator
# ══════════════════════════════════════════════════════════════════════

def run_all(run_dir, outdir=None, skip=None):
    """Load saved simulation data and produce all visualizations.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or .tar.gz archive.
    outdir : str or None
        Directory for output plots. Defaults to ``run_dir/plots``.
    skip : set of str or None
        Names of visualization modules to skip. Valid names:
        'granules', 'fields', 'timeseries', 'composite',
        'compaction', 'percolation', 'movies', 'phases', 'cells', 'stress',
        'mean_field', 'coarse_grain', 'tissue', 'arch_distance'.
    """
    from new_dem_0 import load_run

    skip = set(skip or [])

    print("=" * 65)
    print("  GELLS-DEM Post-Processing")
    print("=" * 65)
    print(f"\n  Loading data from: {run_dir}")

    hist, snaps, p, metadata = load_run(run_dir)

    if not hist:
        print("  ERROR: No history data found. Aborting.")
        return
    if not snaps:
        print("  WARNING: No snapshot data found. Field-based plots will be skipped.")

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
        print("\n  [1/14] Granule positions...")
        fig = plot_granules(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'granules.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if snaps and has_fields and 'fields' not in skip:
        print("  [2/14] Phase fields...")
        fig = plot_fields(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'fields.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if 'timeseries' not in skip:
        print("  [3/14] Timeseries...")
        fig = plot_timeseries(hist, p)
        fig.savefig(os.path.join(outdir, 'timeseries.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    if snaps and has_fields and 'composite' not in skip:
        print("  [4/14] Composite...")
        fig = plot_composite(snaps, hist, p)
        fig.savefig(os.path.join(outdir, 'composite.png'), dpi=150, bbox_inches='tight')
        plt.close(fig)

    # --- External visualization scripts ---
    if 'compaction' not in skip:
        print("  [5/14] Compaction...")
        try:
            from viz import compaction as viz_compaction
            viz_compaction.run_all(hist, snaps=snaps if snaps else None, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if 'percolation' not in skip:
        print("  [6/14] Percolation...")
        try:
            from viz import percolation as viz_percolation
            viz_percolation.run_all(hist, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'movies' not in skip:
        print("  [7/14] Movies...")
        try:
            from viz import movies as viz_movies
            viz_movies.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if 'phases' not in skip:
        print("  [8/14] Phases...")
        try:
            from viz import phases as viz_phases
            viz_phases.run_all(hist, snaps=snaps if snaps else None, p=p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'cells' not in skip:
        print("  [9/14] Cells...")
        try:
            from viz import cells as viz_cells
            viz_cells.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'stress' not in skip:
        print("  [10/14] Stress...")
        try:
            from viz import stress as viz_stress
            viz_stress.run_all(snaps, hist, p, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    # --- Mathematical analysis modules (V1.7) ---
    if 'mean_field' not in skip:
        print("  [11/14] Mean-field model...")
        try:
            from analysis import mean_field_model
            mean_field_model.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and 'coarse_grain' not in skip:
        print("  [12/14] Coarse-graining...")
        try:
            from analysis import coarse_grain
            coarse_grain.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and has_fields and 'tissue' not in skip:
        print("  [13/14] Tissue descriptors...")
        try:
            from analysis import tissue_descriptors
            tissue_descriptors.run_all(actual_dir, outdir=outdir)
        except Exception as e:
            print(f"    SKIPPED (error: {e})")

    if snaps and has_fields and 'arch_distance' not in skip:
        print("  [14/14] Architectural distance...")
        try:
            from analysis import arch_distance
            arch_distance.run_all(run_dir=actual_dir, outdir=outdir)
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
    print("  GELLS-DEM Batch Post-Processing")
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
                                 'Trials', 'trials')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GELLS-DEM post-processing: load saved data and produce all visualizations.'
                    ' With no -i, auto-discovers and processes all unprocessed .tar.gz files.')
    parser.add_argument('-i', '--input', default=None,
                        help='Path to a single simulation output directory or .tar.gz archive')
    parser.add_argument('-o', '--output', default=None,
                        help='Output directory for plots (default: <input>/plots)')
    parser.add_argument('--scan-dir', default=None,
                        help=f'Directory to scan for .tar.gz files (default: Trials/trials/)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Visualization modules to skip (e.g., movies stress)')
    parser.add_argument('--list', action='store_true',
                        help='List unprocessed archives without running visualizations')
    args = parser.parse_args()

    skip = set(args.skip)

    if args.input:
        # Single-run mode
        run_all(args.input, outdir=args.output, skip=skip)
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
