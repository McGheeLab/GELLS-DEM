"""
Engine observer hook, graceful stop and hardened I/O (V3.0 Phase 5a).

  * an Observer sees on_start once, on_step every step, on_save once per
    saved snapshot, on_end('complete'); a plain callable works too,
  * returning True from on_step stops the run: a final snapshot is written at
    the stop step, history.json holds every entry, metadata records the stop,
    and the run can be continued,
  * snapshots and history are written atomically (no .tmp files left behind)
    and history.json exists as soon as the first save happened,
  * load_params() reproduces load_run()'s Params.

Run:  python -m unittest tests.test_observer_hook -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import Params, load_params, load_run, run  # noqa: E402
from gels.live.observer import Observer, RecordingObserver  # noqa: E402


def _params(**kw):
    base = dict(mode='2D', Lx=400.0, Ly=400.0, t_total=1.5, dt=0.5, save_every_h=0.5,
                Ngrid=48, packing_settle_steps=40, save_data=False, compress_archive=False,
                perf_keep_snaps_in_memory=False)
    base.update(kw)
    return Params(**base)


class TestObserverCallbacks(unittest.TestCase):

    def test_callback_counts_and_order(self):
        obs = RecordingObserver()
        hist, snaps, p, gs = run(_params(), seed=3, observer=obs)
        self.assertEqual(obs.starts, 1)
        self.assertEqual([s for s, _ in obs.steps], [1, 2, 3])
        self.assertEqual(len(obs.saves), len(hist))          # t = 0 plus one per step here
        self.assertEqual([idx for idx, *_ in obs.saves], [0, 1, 2, 3])
        self.assertEqual(obs.end, ('complete', len(hist)))
        self.assertEqual(snaps, [])

    def test_callable_observer(self):
        seen = []
        hist, *_ = run(_params(), seed=3, observer=lambda s, t, gs, F: seen.append(s) is not None and False)
        self.assertEqual(seen, [1, 2, 3])

    def test_no_observer_is_default(self):
        hist_a, *_ = run(_params(), seed=5)
        hist_b, *_ = run(_params(), seed=5, observer=None)
        self.assertEqual(json.dumps(hist_a, default=float), json.dumps(hist_b, default=float))


class TestGracefulStop(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_obs_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_stop_writes_final_state_and_can_continue(self):
        # save every 2 steps; stop at step 3 (not a save step) -> a final snapshot is forced
        p = _params(t_total=3.0, dt=0.5, save_every_h=1.0, save_data=True, output_dir=self.tmp)
        obs = RecordingObserver(stop_at_step=3)
        hist, snaps, p, gs = run(p, seed=3, observer=obs)
        self.assertEqual(obs.end[0], 'stopped')
        snaps_on_disk = sorted(f for f in os.listdir(os.path.join(self.tmp, 'snapshots')) if f.endswith('.npz'))
        self.assertEqual(snaps_on_disk, ['snap_0000.npz', 'snap_0001.npz', 'snap_0002.npz'])
        self.assertFalse(any(f.endswith('.tmp') for f in os.listdir(os.path.join(self.tmp, 'snapshots'))))
        with open(os.path.join(self.tmp, 'history.json')) as fh:
            h = json.load(fh)
        self.assertEqual([round(e['time'], 3) for e in h], [0.0, 1.0, 1.5])
        with open(os.path.join(self.tmp, 'metadata.json')) as fh:
            meta = json.load(fh)
        self.assertTrue(meta['stopped_early'])
        self.assertEqual(meta['stop_reason'], 'stopped')
        self.assertAlmostEqual(meta['t_reached'], 1.5)
        self.assertEqual(meta['steps_completed'], 3)
        # continue to the end from the stop snapshot
        p2 = load_params(self.tmp)
        p2.resume_from = self.tmp
        p2.output_dir = self.tmp
        p2.perf_keep_snaps_in_memory = False
        hist2, *_ = run(p2, seed=3)
        self.assertAlmostEqual(hist2[-1]['time'], 3.0)
        with open(os.path.join(self.tmp, 'metadata.json')) as fh:
            meta2 = json.load(fh)
        self.assertFalse(meta2['stopped_early'])

    def test_history_is_written_incrementally(self):
        p = _params(t_total=1.0, dt=0.5, save_every_h=0.5, save_data=True, output_dir=self.tmp)

        class Probe(Observer):
            def __init__(self):
                self.sizes = []

            def on_save(self, snap_idx, s, t, snap, m):
                path = os.path.join(p.output_dir, 'history.json')
                with open(path) as fh:
                    self.sizes.append(len(json.load(fh)))

        probe = Probe()
        run(p, seed=3, observer=probe)
        self.assertEqual(probe.sizes, [1, 2, 3])     # history.json grows at every save

    def test_load_params_matches_load_run(self):
        p = _params(t_total=0.5, save_data=True, output_dir=self.tmp, E_modulus=7.5)
        run(p, seed=3)
        pa = load_params(self.tmp)
        _, _, pb, _ = load_run(self.tmp)
        from dataclasses import asdict
        self.assertEqual(asdict(pa), asdict(pb))
        self.assertEqual(pa.E_modulus, 7.5)
        self.assertEqual(len(pa.species), 2)


if __name__ == '__main__':
    unittest.main()
