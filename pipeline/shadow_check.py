#!/usr/bin/env python3
"""
Calibrate the convergence detector against runs you have already paid for.

    python pipeline/shadow_check.py results/
    python pipeline/shadow_check.py results/my_run --sweep eps_gel=0.005,0.01,0.02

An operator tool, not a test: it reads finished run directories and replays each
one through `gels.convergence.ConvergenceMonitor` frame by frame, at the SAME
save resolution the live detector sees, so a tolerance calibrated here is valid
live. That identity is the reason the path-length signal is accumulated from
saved frames rather than from every step.

For each run it reports

    t_stop        when the detector would have stopped it
    saved         the fraction of the run that would not have been computed
    damage        how much the answer would have changed, as the relative move
                  in gran_lf_func / phi_loc_func / n_bridges between t_stop and
                  the end -- i.e. what stopping early would have cost you
    blocking      the LAST signal to fail before the stop, or the one still
                  blocking at the end

`blocking` is the most useful column and is not in robotsim's version: it says
which tolerance to loosen, rather than leaving you to bisect twelve of them.

Runs that predate the Laguerre metrics cannot be calibrated -- the compaction
guard has no input -- and are reported as such rather than being silently
scored with one guard missing.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gels.convergence import COLS, ConvergenceMonitor, TOLERANCES  # noqa: E402

DAMAGE_KEYS = ('gran_lf_func', 'phi_loc_func', 'n_bridges')


def _snapshot_paths(run_dir):
    d = os.path.join(run_dir, 'snapshots')
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npz'))


def replay(run_dir, overrides=None, t_min_h=0.0):
    """Replay one finished run through the monitor. Returns a report dict."""
    hist_path = os.path.join(run_dir, 'history.json')
    if not os.path.exists(hist_path):
        return None
    with open(hist_path, encoding='utf-8') as fh:
        hist = json.load(fh)
    if len(hist) < 4:
        return None
    snaps = _snapshot_paths(run_dir)

    mean_r = 20.0
    if snaps:
        try:
            with np.load(snaps[0], allow_pickle=False) as z:
                mean_r = float(np.mean(z['r'])) if 'r' in z.files else 20.0
        except (OSError, ValueError, KeyError):
            pass

    mon = ConvergenceMonitor(enable=True, t_min_h=t_min_h, mean_r=mean_r,
                             overrides=overrides)
    t_stop = None
    blocking = ''
    n = min(len(hist), len(snaps)) if snaps else len(hist)
    for k in range(n):
        m = hist[k]
        if snaps:
            try:
                with np.load(snaps[k], allow_pickle=False) as z:
                    cols = [z['x'], z['y']] + ([z['z']] if 'z' in z.files else [])
                    pos = np.stack([np.asarray(c, float) for c in cols], axis=1)
                    func = np.asarray(z['gtype']) == 0 if 'gtype' in z.files \
                        else np.ones(len(pos), bool)
                mon.observe_positions(pos, func)
            except (OSError, ValueError, KeyError):
                pass
        rec = mon.update(mon.row_from_metrics(float(m.get('t', k)), m))
        blocking = rec.get('conv_blocking', '') or blocking
        if mon.converged and t_stop is None:
            t_stop = float(m.get('t', k))
            stop_k = k
    t_end = float(hist[-1].get('t', len(hist) - 1))
    out = dict(run=os.path.basename(run_dir.rstrip('/')), t_end=t_end,
               t_stop=t_stop, n_frames=n,
               has_laguerre=bool(mon.rows and mon.rows[-1][COLS.index('has_laguerre')] > 0.5),
               blocking=blocking, reason=mon.reason)
    out['saved'] = (1.0 - t_stop / t_end) if (t_stop and t_end > 0) else 0.0
    if t_stop is not None:
        dmg = {}
        for key in DAMAGE_KEYS:
            a, b = float(hist[stop_k].get(key, 0.0)), float(hist[-1].get(key, 0.0))
            dmg[key] = abs(b - a) / max(abs(a), abs(b), 1e-9)
        out['damage'] = dmg
        out['damage_max'] = max(dmg.values())
    else:
        out['damage'], out['damage_max'] = {}, 0.0
    return out


def find_runs(root):
    if os.path.exists(os.path.join(root, 'history.json')):
        return [root]
    out = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, 'history.json')):
            out.append(d)
    return out


def _print(rows):
    print(f"\n  {'run':<26} {'frames':>6} {'t_end':>7} {'t_stop':>7} "
          f"{'saved':>7} {'damage':>7}  blocking")
    print('  ' + '-' * 88)
    for r in rows:
        if not r['has_laguerre']:
            print(f"  {r['run']:<26} {r['n_frames']:>6} {r['t_end']:>7.1f} "
                  f"{'--':>7} {'--':>7} {'--':>7}  NO LAGUERRE METRICS "
                  f"(cannot calibrate this run)")
            continue
        ts = f"{r['t_stop']:.1f}" if r['t_stop'] else '--'
        sv = f"{100 * r['saved']:.0f}%" if r['t_stop'] else '--'
        dm = f"{100 * r['damage_max']:.1f}%" if r['t_stop'] else '--'
        print(f"  {r['run']:<26} {r['n_frames']:>6} {r['t_end']:>7.1f} {ts:>7} "
              f"{sv:>7} {dm:>7}  {r['blocking']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='a run directory, or a directory of them')
    ap.add_argument('--t-min', type=float, default=0.0,
                    help='conv_t_min_h to replay with (default 0: let it stop as early '
                         'as the signals allow, which is what you want when calibrating)')
    ap.add_argument('--sweep', default='',
                    help='NAME=v1,v2,... -- replay every run at each value and print the '
                         'saved-vs-damage frontier')
    ap.add_argument('--json', default='', help='also write the report here')
    args = ap.parse_args(argv)

    runs = find_runs(args.root)
    if not runs:
        print(f'  no runs with a history.json under {args.root}')
        return 1

    if args.sweep:
        name, _, vals = args.sweep.partition('=')
        name = name.strip()
        if name not in TOLERANCES:
            print(f'  unknown tolerance {name!r}; known: {sorted(TOLERANCES)}')
            return 1
        print(f'\n  sweep {name} over {vals}   ({len(runs)} run(s))')
        print(f"\n  {name:>12} {'stopped':>9} {'mean saved':>11} {'max damage':>11}")
        print('  ' + '-' * 48)
        for raw in vals.split(','):
            v = float(raw)
            reps = [replay(d, overrides={name: v}, t_min_h=args.t_min) for d in runs]
            reps = [r for r in reps if r and r['has_laguerre']]
            if not reps:
                print(f'  {v:>12g}   no calibratable runs')
                continue
            stopped = [r for r in reps if r['t_stop']]
            print(f'  {v:>12g} {len(stopped):>4}/{len(reps):<4} '
                  f"{100 * np.mean([r['saved'] for r in reps]):>10.0f}% "
                  f"{100 * max([r['damage_max'] for r in reps] or [0]):>10.1f}%")
        print('\n  Pick the largest value whose damage you can live with.\n')
        return 0

    rows = [r for r in (replay(d, t_min_h=args.t_min) for d in runs) if r]
    _print(rows)
    usable = [r for r in rows if r['has_laguerre']]
    if not usable:
        print('\n  Nothing calibratable: every run predates output.metrics_laguerre.'
              '\n  Re-run one with it on before trusting a tolerance.\n')
    else:
        stopped = [r for r in usable if r['t_stop']]
        print(f'\n  {len(stopped)}/{len(usable)} would have stopped early; '
              f"mean saving {100 * np.mean([r['saved'] for r in usable]):.0f} %, "
              f"worst damage {100 * max([r['damage_max'] for r in usable] or [0]):.1f} %.")
        print('  Calibrate with --sweep before setting convergence.enable.\n')
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump(rows, fh, indent=2, default=float)
        print(f'  wrote {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
