"""
gels/live viewer, tail/replay sources and the never-blocking observer (V3.0 Phase 5).

All headless (Agg). Uses the 6-snapshot fixture runs under tests/fixtures/.

  * run_viewer(headless) over a ReplaySource writes one PNG per snapshot, draws
    every granule in its species colour and every cell as a marker,
  * a 3-species run renders three distinct granule colours; a 3D run renders
    the z-slice with a subset of granules,
  * SnapshotTailSource ignores *.tmp files, retries a truncated .npz until it is
    complete, and start='latest' skips what is already on disk,
  * LiveViewObserver never blocks when nobody drains the frame queue,
  * LocalControl maps pause / step / every / stop onto a replay source.

Run:  python -m unittest tests.test_live_viewer -v
"""

import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ['MPLBACKEND'] = 'Agg'
FIX = os.path.join(HERE, 'fixtures')

from gels.engine import Params, generate_packing, run  # noqa: E402
from gels.live.observer import LiveViewObserver  # noqa: E402
from gels.live.tail import ReplaySource, SnapshotTailSource  # noqa: E402
from gels.live.viewer import LocalControl, run_viewer  # noqa: E402

THREE = [
    dict(name='full', f=1.0, volume_fraction=0.3, color='#CC2222', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
    dict(name='half', f=0.5, volume_fraction=0.4, color='#2255CC', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
    dict(name='bare', f=0.0, volume_fraction=0.3, color='#22AA22', radius_mean=55.0, radius_std=5.0, radius_min=20.0),
]


def _distinct_colors(collection):
    return {tuple(np.round(c[:3], 3)) for c in collection.get_facecolors()}


def _headless_opts(out):
    return dict(headless=True, out=out, fps=20.0, hold=True, metrics=True, color_by='species', z_frac=0.5,
                backend='Agg')


class TestHeadlessReplay(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_live_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_replay_2d_writes_png_per_snapshot(self):
        src = ReplaySource(os.path.join(FIX, 'run2d_walls'), fps=1e9)
        state = run_viewer(src, LocalControl(src), _headless_opts(self.tmp))
        pngs = sorted(f for f in os.listdir(self.tmp) if f.endswith('.png'))
        self.assertEqual(len(pngs), len(src.files))
        self.assertEqual(state['n_saved'], len(src.files))
        self.assertTrue(state['done'])
        r = state['renderer']
        self.assertEqual(len(r.granules.get_offsets()), r.N)
        # legacy run: collagen + bare -> two colours
        self.assertEqual(len(_distinct_colors(r.granules)), 2)
        n_cells_drawn = sum(len(ln.get_xdata()) for ln in r.cell_artists.values())
        self.assertEqual(n_cells_drawn, r.start['n_cells'])
        self.assertGreater(len(src.hist), 0)

    def test_replay_3d_draws_slice(self):
        src = ReplaySource(os.path.join(FIX, 'run3d_spheres'), fps=1e9)
        state = run_viewer(src, LocalControl(src), _headless_opts(self.tmp))
        r = state['renderer']
        self.assertTrue(r.is3d)
        n_shown = len(r.granules.get_offsets())
        self.assertGreater(n_shown, 0)
        self.assertLess(n_shown, r.N)                     # only granules cutting the mid-plane
        self.assertEqual(state['n_saved'], len(src.files))

    def test_three_species_three_colours(self):
        p = Params(mode='2D', Lx=400.0, Ly=400.0, t_total=0.5, dt=0.5, save_every_h=0.5, save_data=True,
                   compress_archive=False, output_dir=self.tmp, phi_solid_target=0.55,
                   species=[dict(s) for s in THREE], cell_surface_coverage=1.0, packing_settle_steps=30,
                   Ngrid=40)
        run(p, seed=3)
        out = os.path.join(self.tmp, 'live')
        src = ReplaySource(self.tmp, fps=1e9)
        state = run_viewer(src, LocalControl(src), _headless_opts(out))
        r = state['renderer']
        self.assertEqual(len(r.species), 3)
        got = _distinct_colors(r.granules)
        from viz2.palette import hex_to_rgba
        want = {tuple(np.round(hex_to_rgba(s['color'])[:3], 3)) for s in THREE}
        self.assertEqual(got, want)
        self.assertEqual(state['n_saved'], len(src.files))
        # colour-by-f mode gives a different (viridis) palette
        r.color_mode = 'f'
        r.update(r.frame)
        self.assertNotEqual(_distinct_colors(r.granules), want)


class TestTailSource(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_tail_')
        self.run = os.path.join(self.tmp, 'run')
        shutil.copytree(os.path.join(FIX, 'run2d_walls'), self.run)
        self.snaps = os.path.join(self.run, 'snapshots')
        self.held = {}
        for k in (3, 4, 5):
            name = f'snap_{k:04d}.npz'
            with open(os.path.join(self.snaps, name), 'rb') as fh:
                self.held[name] = fh.read()
            os.remove(os.path.join(self.snaps, name))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_from_start_then_follow_ignoring_tmp_and_truncated(self):
        src = SnapshotTailSource(self.run, poll_s=0.0, start='first')
        msgs = src.poll()
        self.assertEqual([m['kind'] for m in msgs], ['start', 'frame', 'frame', 'frame'])
        self.assertEqual(msgs[0]['N'], len(msgs[1]['x']))
        self.assertEqual(src.poll(), [])
        # a .tmp file (atomic write in progress) is invisible
        with open(os.path.join(self.snaps, 'snap_0003.npz.tmp'), 'wb') as fh:
            fh.write(b'garbage')
        self.assertEqual(src.poll(), [])
        # a truncated .npz is retried, never raises
        with open(os.path.join(self.snaps, 'snap_0003.npz'), 'wb') as fh:
            fh.write(self.held['snap_0003.npz'][:200])
        self.assertEqual(src.poll(), [])
        with open(os.path.join(self.snaps, 'snap_0003.npz'), 'wb') as fh:
            fh.write(self.held['snap_0003.npz'])
        msgs = src.poll()
        self.assertEqual([m['kind'] for m in msgs], ['frame'])
        self.assertEqual(msgs[0]['step'], int(round(float(msgs[0]['t']) / src.p.dt)))
        # two more arrive together
        for name in ('snap_0004.npz', 'snap_0005.npz'):
            with open(os.path.join(self.snaps, name), 'wb') as fh:
                fh.write(self.held[name])
        self.assertEqual(len(src.poll()), 2)
        self.assertEqual(src.poll(), [])

    def test_latest_skips_existing_and_idle_finish(self):
        src = SnapshotTailSource(self.run, poll_s=0.0, start='latest', idle_finish_s=0.05)
        msgs = src.poll()
        self.assertEqual([m['kind'] for m in msgs], ['start', 'frame'])
        self.assertEqual(msgs[1]['step'], int(round(float(msgs[1]['t']) / src.p.dt)))
        time.sleep(0.1)
        msgs = src.poll()
        self.assertEqual([m['kind'] for m in msgs], ['end'])
        self.assertTrue(src.finished)

    def test_metrics_joined_from_history(self):
        src = SnapshotTailSource(self.run, poll_s=0.0, start='first')
        msgs = [m for m in src.poll() if m['kind'] == 'frame']
        self.assertTrue(all(m.get('metrics') for m in msgs))
        self.assertIn('n_bridges', msgs[0]['metrics'])
        self.assertIn('time', msgs[0]['metrics'])


class TestObserverNeverBlocks(unittest.TestCase):

    def test_full_queue_drops_frames_without_waiting(self):
        p = Params(mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.5, packing_settle_steps=20, save_data=False,
                   cell_surface_coverage=1.0)
        gs = generate_packing(p, seed=1)
        frame_q = queue.Queue(maxsize=1)
        ctrl_q = queue.Queue()
        ctrl_evt, stop_evt = threading.Event(), threading.Event()
        obs = LiveViewObserver(frame_q, ctrl_q, ctrl_evt, stop_evt, every_n_steps=1, max_fps=1e6)
        obs.on_start(gs, p, seed=1, n_steps=100, resume_step=0)
        self.assertEqual(frame_q.qsize(), 1)             # the start message fills the only slot
        t0 = time.perf_counter()
        for s in range(1, 51):
            self.assertFalse(obs.on_step(s, s * p.dt, gs, None, None))
        self.assertLess(time.perf_counter() - t0, 2.0)
        self.assertGreaterEqual(obs.n_dropped, 40)
        self.assertEqual(obs.n_sent, 0)
        # control channel: stop ends the run at the next step
        ctrl_q.put(('stop',))
        ctrl_evt.set()
        self.assertTrue(obs.on_step(51, 51 * p.dt, gs, None, None))
        obs.on_end([], gs, 'stopped')                    # must not raise on a full queue

    def test_pause_step_resume_and_detach(self):
        p = Params(mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.5, packing_settle_steps=20, save_data=False)
        gs = generate_packing(p, seed=1)
        frame_q = queue.Queue(maxsize=8)
        ctrl_q = queue.Queue()
        ctrl_evt, stop_evt = threading.Event(), threading.Event()
        obs = LiveViewObserver(frame_q, ctrl_q, ctrl_evt, stop_evt, every_n_steps=1)
        obs.on_start(gs, p, seed=1, n_steps=10, resume_step=0)
        # pause, then release it from another thread with a single step credit
        ctrl_q.put(('pause',))
        ctrl_evt.set()

        def release():
            time.sleep(0.2)
            ctrl_q.put(('step', 1))
            ctrl_evt.set()
        threading.Thread(target=release).start()
        t0 = time.perf_counter()
        self.assertFalse(obs.on_step(1, p.dt, gs, None, None))
        self.assertGreaterEqual(time.perf_counter() - t0, 0.15)      # held while paused
        self.assertTrue(obs.paused)
        ctrl_q.put(('resume',))
        ctrl_evt.set()
        self.assertFalse(obs.on_step(2, 2 * p.dt, gs, None, None))
        self.assertFalse(obs.paused)
        ctrl_q.put(('detach',))
        ctrl_evt.set()
        self.assertFalse(obs.on_step(3, 3 * p.dt, gs, None, None))
        self.assertTrue(obs.detached)
        kinds = []
        while not frame_q.empty():
            kinds.append(frame_q.get_nowait()['kind'])
        self.assertEqual(kinds[0], 'start')
        self.assertIn('frame', kinds)


class TestLocalControl(unittest.TestCase):

    def test_replay_control(self):
        src = ReplaySource(os.path.join(FIX, 'run2d_periodic'), fps=1e9)
        ctrl = LocalControl(src)
        msgs = src.poll()
        self.assertEqual([m['kind'] for m in msgs], ['start', 'frame'])
        ctrl.send('pause')
        self.assertTrue(src.paused)
        self.assertEqual(src.poll(), [])
        ctrl.send('step', 1)
        self.assertEqual(len(src.poll()), 1)
        self.assertEqual(src.poll(), [])
        ctrl.send('every', 2)
        self.assertEqual(src.stride, 2)
        ctrl.send('resume')
        self.assertFalse(src.paused)
        ctrl.stop()
        self.assertTrue(src.finished)
        self.assertEqual(src.poll(), [])


if __name__ == '__main__':
    unittest.main()
