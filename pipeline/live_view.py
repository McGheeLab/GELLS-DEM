"""
Watch, replay or dump a GELS run without touching the simulation.
================================================================

  python pipeline/live_view.py -i results/NAME              # attach to a running step 3 (new snapshots only)
  python pipeline/live_view.py -i results/NAME --from-start # ...replaying what is already on disk first
  python pipeline/live_view.py -i results/NAME --replay     # play back a finished run
  python pipeline/live_view.py -i results/NAME --replay --loop --fps 15
  python pipeline/live_view.py -i results/NAME --headless   # one PNG per snapshot -> results/NAME/live/

The run directory is read only; the viewer never blocks step 3. Half-written
snapshots (the engine writes atomically) are retried on the next poll.
Keys inside the window: space pause, n step, x/q quit, c colour mode,
b/l/m toggles, [ ] 0 z-slice (3D), s save PNG, h help.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass


def resolve_run_dir(arg):
    """Accept a path or a run name under results/."""
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    cand = os.path.join(REPO, 'results', arg)
    if os.path.isdir(cand):
        return cand
    raise SystemExit(f"run directory not found: {arg}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-i', '--input', required=True, help='run directory or run name under results/')
    ap.add_argument('--replay', action='store_true', help='play back saved snapshots instead of tailing')
    ap.add_argument('--loop', action='store_true', help='replay: start over at the end')
    ap.add_argument('--from-start', action='store_true', help='tail: show existing snapshots before following')
    ap.add_argument('--fps', type=float, default=10.0, help='replay cadence (frames/s); viewer refresh is >= 20')
    ap.add_argument('--headless', action='store_true', help='no window: write a PNG per snapshot (implies --replay)')
    ap.add_argument('--out', default=None, help='headless output directory (default RUN/live)')
    ap.add_argument('--idle-finish', type=float, default=None,
                    help='tail: finish after this many seconds without a new snapshot')
    ap.add_argument('--no-metrics', action='store_true', help='hide the metrics column')
    ap.add_argument('--color-by', choices=['species', 'f', 'speed'], default='species')
    ap.add_argument('--z-frac', type=float, default=0.5, help='3D: initial z-slice as a fraction of Lz')
    a = ap.parse_args(argv)

    run_dir = resolve_run_dir(a.input)
    if a.headless:
        os.environ['MPLBACKEND'] = 'Agg'
    else:
        os.environ.setdefault('MPLBACKEND', 'TkAgg')
    import matplotlib
    matplotlib.use(os.environ['MPLBACKEND'])

    from gels.live.tail import ReplaySource, SnapshotTailSource
    from gels.live.viewer import LocalControl, run_viewer

    if a.replay or a.headless:
        source = ReplaySource(run_dir, fps=1e9 if a.headless else a.fps, loop=a.loop and not a.headless)
        print(f"replaying {len(source.files)} snapshots from {run_dir}")
    else:
        source = SnapshotTailSource(run_dir, start='first' if a.from_start else 'latest',
                                    idle_finish_s=a.idle_finish)
        print(f"tailing {run_dir} (Ctrl+C or q to leave; the simulation is unaffected)")
    out_dir = a.out or os.path.join(run_dir, 'live')
    opts = dict(headless=a.headless, out=out_dir, fps=max(20.0, a.fps), hold=True,
                metrics=not a.no_metrics, color_by=a.color_by, z_frac=a.z_frac,
                backend=os.environ['MPLBACKEND'])
    try:
        state = run_viewer(source, LocalControl(source), opts)
    except KeyboardInterrupt:
        print("\nviewer closed")
        return 0
    if a.headless:
        print(f"wrote {state['n_saved']} PNG(s) to {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
