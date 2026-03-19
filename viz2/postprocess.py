"""
V2 Visualization Postprocessor for GELS
=============================================
Unified orchestrator that runs all viz2 modules on simulation output.

Usage:
    python viz2/postprocess.py -i results/default
    python viz2/postprocess.py -i results/default --skip movies energy
    python viz2/postprocess.py -i results/default --only scaffold voronoi

Or import programmatically:
    from viz2 import run_all
    run_all(run_dir='results/default')

    # Or with live data:
    from viz2.postprocess import run_all
    run_all(snaps=snaps, hist=hist, p=p)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')

from pathlib import Path


# Module registry: (name, import_path, function_name)
MODULES = [
    ('scaffold',    'viz2.scaffold_map',     'run_all'),
    ('voronoi',     'viz2.voronoi_shapes',   'run_all'),
    ('phases',      'viz2.phase_fractions',  'run_all'),
    ('movies',      'viz2.movies',           'run_all'),
    ('energy',      'viz2.energy_stress',    'run_all'),
    ('percolation', 'viz2.void_percolation', 'run_all'),
    ('scaffold_3d', 'viz2.scaffold_map_3d',  'run_all'),
]


def run_all(run_dir=None, snaps=None, hist=None, p=None, metadata=None,
            outdir=None, skip=None, only=None):
    """Run all viz2 modules.

    Can accept either run_dir (loads from disk) or live data (snaps, hist, p).

    Parameters
    ----------
    run_dir : str or None
        Path to saved simulation output directory or .tar.gz.
    snaps : list of dict or None
        Snapshot data (alternative to run_dir).
    hist : list of dict or None
        History metrics (alternative to run_dir).
    p : Params or None
        Simulation parameters (alternative to run_dir).
    metadata : dict or None
        Simulation metadata.
    outdir : str or None
        Output directory. Default: run_dir/viz2_plots.
    skip : set of str or None
        Module names to skip.
    only : set of str or None
        If set, only run these modules.
    """
    # Load data if needed
    if snaps is None or hist is None or p is None:
        if run_dir is None:
            raise ValueError("Must provide either run_dir or (snaps, hist, p)")
        from viz2.common import load_data
        hist, snaps, p, metadata = load_data(run_dir)

    if not snaps:
        print("  WARNING: No snapshot data available.")
        return

    # Output directory
    if outdir is None:
        if run_dir is not None:
            actual = run_dir[:-7] if run_dir.endswith('.tar.gz') else run_dir
            outdir = os.path.join(actual, 'viz2_plots')
        else:
            outdir = 'viz2_plots'

    Path(outdir).mkdir(parents=True, exist_ok=True)

    skip = set(skip) if skip else set()
    only = set(only) if only else None

    print("=" * 60)
    print("  GELS V2 Visualization")
    print(f"  {len(snaps)} snapshots, output -> {outdir}")
    print("=" * 60)

    t0 = time.time()
    n_run = 0
    n_fail = 0

    for name, mod_path, func_name in MODULES:
        # Skip / only filtering
        if only is not None and name not in only:
            continue
        if name in skip:
            print(f"  [{name}] SKIPPED")
            continue

        try:
            import importlib
            mod = importlib.import_module(mod_path)
            func = getattr(mod, func_name)
            func(snaps, hist, p, outdir=outdir)
            n_run += 1
        except Exception as e:
            print(f"  [{name}] FAILED: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    elapsed = time.time() - t0
    print("=" * 60)
    print(f"  Done. {n_run} modules completed, {n_fail} failed. "
          f"({elapsed:.1f}s)")
    print(f"  Output: {outdir}")
    print("=" * 60)


def run_all_from_dir(run_dir, **kwargs):
    """Convenience wrapper: load data and run all viz2 modules."""
    return run_all(run_dir=run_dir, **kwargs)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='V2 Visualization Postprocessor for GELS')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to simulation output directory or .tar.gz')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: <input>/viz2_plots)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Module names to skip: scaffold voronoi phases '
                             'movies energy percolation scaffold_3d')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Only run these modules')
    parser.add_argument('--list', action='store_true',
                        help='List available modules and exit')
    args = parser.parse_args()

    if args.list:
        print("Available viz2 modules:")
        for name, mod_path, _ in MODULES:
            print(f"  {name:15s} -> {mod_path}")
        sys.exit(0)

    run_all(
        run_dir=args.input,
        outdir=args.outdir,
        skip=set(args.skip),
        only=set(args.only) if args.only else None,
    )
