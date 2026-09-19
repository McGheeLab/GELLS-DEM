"""
STEP 3 of 5 - Run the simulation.
=================================

Restores the packing written by step 2 and advances the overdamped dynamics
to ``t_total``, appending a snapshot every ``save_every_h`` hours. This is
the expensive step.

It drives the engine's V2.6 resume machinery: ``Params.resume_from`` is
pointed at the run directory, so ``run()`` skips packing generation, restores
the last snapshot on disk, and continues the main loop from there. That has a
useful consequence - if a long run is interrupted, re-running this step with
``--continue`` picks up from the last saved snapshot instead of starting over.

Examples
--------
    python pipeline/step3_simulate.py -i results/my_run
    python pipeline/step3_simulate.py -i results/my_run --continue
    python pipeline/step3_simulate.py -i results/my_run --t_total 120 --continue
    python pipeline/step3_simulate.py -i results/my_run --force --archive
    python pipeline/step3_simulate.py -i results/my_run --live            # watch it run
    python pipeline/step3_simulate.py -i results/my_run --live --live-every 5 --threads 20

Live view (V3.0)
----------------
``--live`` opens a matplotlib window in a *separate process* that is fed
frames through a bounded queue; the simulation never waits for it. Keys in
the window: space pause/resume, n single step, x stop the run gracefully
(state is saved, ``--continue`` picks it up), q detach (the run carries on),
c colour mode (species / f / speed), b l m toggles, [ ] z-slice in 3D,
s screenshot, h help. A run can also be watched from another terminal
without ``--live``: ``python pipeline/live_view.py -i results/my_run``.
"""

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    banner, done, guard, mark_done, params_from_run_dir, print_progress,
    require, save_params, seed_for,
)


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 3: run the simulation.")
    parser.add_argument('-i', '--run-dir', required=True,
                        help='Run directory prepared by steps 1 and 2')
    parser.add_argument('--t_total', type=float, default=None,
                        help='Override the total simulated time (hours)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the seed recorded at step 1')
    parser.add_argument('--continue', dest='cont', action='store_true',
                        help='Resume an interrupted run from its last snapshot')
    parser.add_argument('--archive', action='store_true',
                        help='Also write a .tar.gz of the run directory')
    parser.add_argument('--force', action='store_true',
                        help='Discard simulated snapshots and restart from '
                             'the packing (snapshot 0000)')
    parser.add_argument('--live', action='store_true',
                        help='Open a real-time viewer in a separate window '
                             'while the run advances')
    parser.add_argument('--live-every', type=int, default=0, metavar='N',
                        help='Send a frame to the viewer every N steps '
                             '(0 = adapt to the step rate)')
    parser.add_argument('--live-fps', type=float, default=20.0, metavar='F',
                        help='Viewer refresh ceiling (frames per second)')
    parser.add_argument('--no-live-hold', action='store_true',
                        help='Close the viewer automatically when the run finishes')
    parser.add_argument('--threads', type=int, default=None, metavar='N',
                        help='Compute threads for the compiled kernels '
                             '(0 = physical cores, -1 = all logical CPUs)')
    args = parser.parse_args()

    run_dir = args.run_dir
    banner('simulate', run_dir)
    require(run_dir, 'config')
    require(run_dir, 'pack')
    if not args.cont:
        guard(run_dir, 'simulate', args.force)

    from gels.engine import find_last_snapshot, run

    p = params_from_run_dir(run_dir)
    p.output_dir = run_dir
    p.save_data = True
    p.compress_archive = bool(args.archive)
    p.perf_keep_snaps_in_memory = False   # step 3 never uses the returned snaps
    if args.threads is not None:
        p.perf_threads = args.threads
    if args.t_total is not None:
        p.t_total = args.t_total
    seed = args.seed if args.seed is not None else seed_for(run_dir, 42)

    snap_dir = os.path.join(run_dir, 'snapshots')

    # --force restarts the dynamics from the packing: drop every snapshot
    # after 0000 and truncate the history back to its t = 0 entry.
    if args.force and not args.cont:
        removed = 0
        for path in sorted(glob.glob(os.path.join(snap_dir, 'snap_*.npz'))):
            if os.path.basename(path) != 'snap_0000.npz':
                os.remove(path)
                removed += 1
        fields_dir = os.path.join(run_dir, 'fields')
        if os.path.isdir(fields_dir):
            for path in sorted(glob.glob(os.path.join(fields_dir, 'fields_*.npz'))):
                if os.path.basename(path) != 'fields_0000.npz':
                    os.remove(path)
        _truncate_history(run_dir)
        if removed:
            print(f"  Discarded {removed} simulated snapshot(s); "
                  f"restarting from the packing")

    # Resume from whatever is on disk: snapshot 0000 on a fresh run, or the
    # last snapshot written when continuing an interrupted one.
    p.resume_from = run_dir
    last_snap = find_last_snapshot(run_dir)
    print(f"  Resuming from: {os.path.basename(last_snap)}")
    print(f"  Target time:   {p.t_total:.1f} h "
          f"(dt={p.dt:.2f} h, save every {p.save_every_h:.1f} h)")
    print(f"  Seed:          {seed}")
    if p.perf_threads:
        print(f"  Threads:       {p.perf_threads}")
    observer, viewer = (None, None)
    if args.live:
        observer, viewer = _start_live_view(run_dir, args)
    print()

    t0 = time.time()
    hist, _snaps, p, gs = run(p, seed=seed, observer=observer)
    elapsed = time.time() - t0

    # run() rewrites params.json with resume_from set; clear it so the file
    # describes the run rather than the mechanics of how it was restarted.
    p.resume_from = ""
    save_params(p, run_dir)

    # ── Report ──
    n_snaps = len(glob.glob(os.path.join(snap_dir, 'snap_*.npz')))
    print()
    if hist:
        h0, hf = hist[0], hist[-1]
        print(f"  Functional clusters: {h0['func_nc']} -> {hf['func_nc']}"
              f"  (largest {h0['func_lf']:.1%} -> {hf['func_lf']:.1%})")
        print(f"  Void clusters:       {h0['void_nc']} -> {hf['void_nc']}")
        print(f"  Bridges:             {h0['n_bridges']} -> {hf['n_bridges']}")
        print(f"  Porosity:            {h0['porosity']:.3f} -> {hf['porosity']:.3f}")
        print(f"  Permeability:        {h0['K_kozeny_carman']:.1f} -> "
              f"{hf['K_kozeny_carman']:.1f} um^2")
        print(f"  Tissue fraction:     {h0['tissue_frac']:.1%} -> "
              f"{hf['tissue_frac']:.1%}")
        print(f"  Displacement:        func={hf['disp_func']:.1f} um, "
              f"inert={hf['disp_inert']:.1f} um")
    print(f"  Snapshots on disk:   {n_snaps}")

    mark_done(run_dir, 'simulate', {
        'seed': seed,
        't_total': p.t_total,
        'n_snapshots': n_snaps,
        'n_history_entries': len(hist),
        'wall_seconds': round(elapsed, 1),
        'archived': bool(args.archive),
    })

    print_progress(run_dir)
    done('simulate', run_dir, elapsed)
    _finish_live_view(viewer, args)


def _start_live_view(run_dir, args):
    """Spawn the viewer process; return (observer, process).

    The viewer runs under a ``spawn`` context (the only one on Windows) and
    sets its own matplotlib backend (TkAgg) before importing matplotlib -
    this process stays on Agg. Frames travel through a bounded queue and
    are dropped, never awaited, when the window falls behind.
    """
    import multiprocessing as mp

    from gels.live.observer import LiveViewObserver
    from gels.live.viewer import viewer_main

    ctx = mp.get_context('spawn')
    frame_q = ctx.Queue(maxsize=4)
    ctrl_q = ctx.Queue()
    ctrl_evt = ctx.Event()
    stop_evt = ctx.Event()
    opts = dict(fps=args.live_fps, hold=not args.no_live_hold, backend='TkAgg', metrics=True)
    proc = ctx.Process(target=viewer_main, args=(frame_q, ctrl_q, ctrl_evt, stop_evt, opts),
                       name='gels-live-viewer', daemon=True)
    proc.start()
    observer = LiveViewObserver(frame_q, ctrl_q, ctrl_evt, stop_evt,
                                every_n_steps=args.live_every, max_fps=args.live_fps,
                                run_dir=run_dir)
    every = args.live_every if args.live_every > 0 else 'auto'
    print(f"  Live view:     every {every} step(s), <= {args.live_fps:.0f} fps "
          f"(space pause, n step, x stop, q detach, h help)")
    return observer, proc


def _finish_live_view(viewer, args):
    """Keep the window up until the user closes it (unless --no-live-hold)."""
    if viewer is None:
        return
    if viewer.is_alive() and not args.no_live_hold:
        print("  Viewer window is still open - close it (or press q) to finish this step.")
        try:
            viewer.join()
        except KeyboardInterrupt:
            pass
    if viewer.is_alive():
        viewer.join(timeout=3.0)          # let it draw the final state and exit on its own
    if viewer.is_alive():
        viewer.terminate()
        viewer.join(timeout=2.0)


def _truncate_history(run_dir):
    """Drop every history entry after t = 0 (used by --force)."""
    import json

    hist_json = os.path.join(run_dir, 'history.json')
    if not os.path.exists(hist_json):
        return
    with open(hist_json) as f:
        hist = json.load(f)
    if len(hist) <= 1:
        return
    with open(hist_json, 'w') as f:
        json.dump(hist[:1], f, indent=2, default=float)
    # history.csv is rewritten wholesale at the end of run(); removing the
    # stale copy avoids a half-updated file if the run is interrupted.
    hist_csv = os.path.join(run_dir, 'history.csv')
    if os.path.exists(hist_csv):
        os.remove(hist_csv)


if __name__ == '__main__':
    main()
