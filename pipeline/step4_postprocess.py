"""
STEP 4 of 5 - Post-process into figures and movies.
===================================================

Loads the saved run and renders it. Two visualisation suites are available:

    viz2  (default)  scaffold maps, Voronoi, phase fractions, movies,
                     energy/stress, void percolation, 3D scaffold maps
    viz              the older V1.x suite: granule/field panels, timeseries,
                     compaction, percolation, cells, surface stress,
                     deformation, shape galleries

Since V2.6 the engine does not save phase-field grids by default
(``save_fields=False``), because they can be rebuilt from particle positions.
Both suites are driven here in a way that rebuilds them on demand - viz2 via
``ensure_phase_fields()``, viz via its ``rerender`` flag - so figures come out
correct whether or not ``fields/`` exists.

Examples
--------
    python pipeline/step4_postprocess.py -i results/my_run
    python pipeline/step4_postprocess.py -i results/my_run --suite both
    python pipeline/step4_postprocess.py -i results/my_run --skip movies
    python pipeline/step4_postprocess.py -i results/my_run --only scaffold phases
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    banner, done, guard, mark_done, print_progress, require,
)


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 4: render figures and movies.")
    parser.add_argument('-i', '--run-dir', required=True,
                        help='Run directory containing the simulated run')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Figure output directory '
                             '(default: <run_dir>/viz2_plots or /plots)')
    parser.add_argument('--suite', choices=['viz2', 'viz', 'both'],
                        default='viz2',
                        help='Which visualisation suite to run (default: viz2)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Module names to skip')
    parser.add_argument('--only', nargs='*', default=None,
                        help='Run only these modules (viz2 suite)')
    parser.add_argument('--workers', type=int, default=None, metavar='N',
                        help='Processes for the per-snapshot plotting work '
                             '(default: auto = physical cores, capped at 8; 1 = serial)')
    parser.add_argument('--force', action='store_true',
                        help='Re-render even if this step already completed')
    args = parser.parse_args()

    run_dir = args.run_dir
    banner('postprocess', run_dir)
    require(run_dir, 'config')
    require(run_dir, 'pack')
    require(run_dir, 'simulate')
    guard(run_dir, 'postprocess', args.force)

    skip = set(args.skip) if args.skip else None
    only = set(args.only) if args.only else None
    produced = {}
    t0 = time.time()

    if args.suite in ('viz2', 'both'):
        outdir = args.outdir or os.path.join(run_dir, 'viz2_plots')
        print(f"\n  --- viz2 suite -> {outdir} ---")
        try:
            from viz2 import run_all as viz2_run_all
            viz2_run_all(run_dir=run_dir, outdir=outdir, skip=skip, only=only,
                         workers=args.workers)
            produced['viz2'] = _count_figures(outdir)
        except Exception as exc:  # a broken module must not lose the others
            import traceback
            traceback.print_exc()
            print(f"  viz2 suite FAILED: {type(exc).__name__}: {exc}")
            produced['viz2'] = f'failed: {type(exc).__name__}'

    if args.suite in ('viz', 'both'):
        outdir = (os.path.join(run_dir, 'plots') if args.suite == 'both'
                  else (args.outdir or os.path.join(run_dir, 'plots')))
        print(f"\n  --- viz suite -> {outdir} ---")
        try:
            from viz.postprocess import run_all as viz_run_all
            # rerender=True rebuilds phase fields from particle positions,
            # which V2.6 runs no longer save.
            viz_run_all(run_dir, outdir=outdir, skip=skip, rerender=True)
            produced['viz'] = _count_figures(outdir)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  viz suite FAILED: {type(exc).__name__}: {exc}")
            produced['viz'] = f'failed: {type(exc).__name__}'

    elapsed = time.time() - t0

    print()
    for suite, count in produced.items():
        print(f"  {suite}: {count} file(s)")

    mark_done(run_dir, 'postprocess', {
        'suite': args.suite,
        'skipped': sorted(skip) if skip else [],
        'only': sorted(only) if only else [],
        'workers': args.workers,
        'output': produced,
        'wall_seconds': round(elapsed, 1),
    })

    print_progress(run_dir)
    done('postprocess', run_dir, elapsed)


def _count_figures(outdir):
    """Count rendered files below `outdir`."""
    exts = ('.png', '.pdf', '.svg', '.gif', '.mp4')
    if not os.path.isdir(outdir):
        return 0
    return sum(1 for root, _dirs, files in os.walk(outdir)
               for f in files if f.lower().endswith(exts))


if __name__ == '__main__':
    main()
