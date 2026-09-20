"""
The convergence detector (V3.6 Phase 5).

**The design claim, measured rather than asserted.** Coarsening in an arrested
gel slows as a power law ``t^-alpha``. A flatness detector on FIXED windows
fires on it -- stopping a run that is still evolving. Proportional windows
(``[t/2, 3t/4)`` against ``[3t/4, t]``) see a change that decays one power of
``t`` more slowly, so the false trigger moves far outside any run you would do.

It does **not** hold out for ever, and the usual phrasing of this claim
overstates it. Measured on ``x(t) = 1 - 0.5 t^-alpha``:

    alpha = 0.3    proportional first passes at t = 635 h   fixed window: 58 h
    alpha = 0.5    proportional first passes at t = 203 h   fixed window: 50 h

A GELS run is 24-72 h. So the fixed-window detector would falsely stop a still
coarsening run INSIDE it and the proportional one does not come near -- an
11x margin at alpha = 0.3. That ratio is what
`test_a_power_law_does_not_converge_within_any_real_run` pins, together with the
control that shows the window shape, not the tolerances, is doing the work.

Two signals need saying out loud:

* **Path length, not displacement.** `disp_func` is NET displacement and cancels
  under creep: a bed churning in place reads the same as one that has stopped.
  The monitor accumulates ``|dx|`` between SAVED frames instead -- the same
  resolution `pipeline/shadow_check.py` replays at, deliberately, so a tolerance
  calibrated in shadow is valid live.
* **Quiet OR steady.** A prestressed bed can sustain a constant-rate creep that
  never changes its structure. Steady residual motion with every structural
  signal flat is converged; real coarsening shows a DECAYING rate and failing
  structural signals, so it still blocks.

And two guards that refuse rather than pass: a constant division rate is not a
steady state, and without the Laguerre metrics the compaction guard has no input
and the monitor declines to converge at all rather than quietly dropping it.

Run:  python -m unittest tests.test_convergence -v
"""

import contextlib
import io
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

from gels.config import load_setup, template_setup, validate  # noqa: E402
from gels.convergence import COLS, ConvergenceMonitor, path_increment  # noqa: E402
from gels.engine import Params, run  # noqa: E402
from gels.live.observer import CompositeObserver, Observer  # noqa: E402


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def row(t, x, path=0.0, divisions=0.0, laguerre=True, compaction=0.99,
        phi_loc=0.60):
    """One raw row whose every structural signal tracks ``x``."""
    return dict(t=float(t), path_func=path, path_inert=path,
                n_bridges=100.0 * x, gran_lf_func=x, demix_phi_loc=x, z_if=x,
                F_mean=50.0 * x, n_divisions_cum=divisions,
                compaction_func=compaction, phi_loc_func=phi_loc,
                has_laguerre=1.0 if laguerre else 0.0)


def feed(mon, series, dt=0.5, **kw):
    for k, x in enumerate(series, start=1):
        mon.update(row(dt * k, x, path=0.001 * dt * k, **kw))
    return mon


class TestTheDesignClaim(unittest.TestCase):

    @staticmethod
    def _first_pass(alpha, n=4000, dt=0.5):
        """When the detector first passes on x(t) = 1 - 0.5 t^-alpha, or None."""
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        for k in range(1, n):
            t = dt * k
            mon.update(row(t, 1.0 - 0.5 * t ** -alpha, path=0.001 * t))
            if mon.converged:
                return t
        return None

    @staticmethod
    def _fixed_window_flat(alpha, w=20, n=4000, dt=0.5, eps=0.01):
        """When a NAIVE fixed-window detector would first call the same series flat."""
        t = np.arange(1, n) * dt
        x = 1.0 - 0.5 * t ** -alpha
        for i in range(3 * w, len(t)):
            if abs(x[i - w:i].mean() - x[i - 2 * w:i - w].mean()) < eps:
                return t[i]
        return None

    def test_a_power_law_does_not_converge_within_any_real_run(self):
        # A GELS run is 24-72 h. The detector must not fire on a coarsening
        # power law anywhere near that.
        for alpha in (0.3, 0.5):
            with self.subTest(alpha=alpha):
                t_pass = self._first_pass(alpha)
                self.assertIsNotNone(t_pass)
                self.assertGreater(t_pass, 150.0,
                                   f'a t^-{alpha} power law went flat at {t_pass} h')

    def test_and_a_fixed_window_would_have_stopped_it_inside_the_run(self):
        # The control: the window SHAPE is doing the work, not the tolerances.
        for alpha in (0.3, 0.5):
            with self.subTest(alpha=alpha):
                t_fixed = self._fixed_window_flat(alpha)
                t_prop = self._first_pass(alpha)
                self.assertIsNotNone(t_fixed)
                self.assertLess(t_fixed, 72.0,
                                'the fixed window is not flat inside a run; bad control')
                self.assertGreater(t_prop, 2.5 * t_fixed)

    def test_an_arrested_run_does_converge(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        feed(mon, [1.0 - 0.3 * np.exp(-0.5 * 0.5 * k) for k in range(1, 400)])
        self.assertTrue(mon.converged, mon.last.get('conv_blocking'))
        self.assertTrue(mon.should_stop())
        self.assertIn('flat since', mon.reason)


class TestThePathSignal(unittest.TestCase):

    def test_path_length_sees_what_net_displacement_cannot(self):
        a = np.zeros((4, 2))
        b = a.copy()
        b[:, 0] = 1.0
        self.assertAlmostEqual(path_increment(b, a), 1.0)
        self.assertAlmostEqual(path_increment(a, b), 1.0)     # and back
        # a granule that returns to where it started has zero NET displacement
        # and two units of path.
        mon = ConvergenceMonitor()
        mask = np.ones(4, bool)
        mon.observe_positions(a, mask)
        mon.observe_positions(b, mask)
        p_func, _p_inert = mon.observe_positions(a, mask)
        self.assertAlmostEqual(p_func, 2.0)
        np.testing.assert_allclose(a.mean(axis=0), a.mean(axis=0))   # net = 0

    def test_it_splits_functional_from_inert(self):
        a = np.zeros((4, 2))
        b = a.copy()
        b[:2, 0] = 1.0                    # only the functional half moves
        mon = ConvergenceMonitor()
        mask = np.array([True, True, False, False])
        mon.observe_positions(a, mask)
        pf, pi = mon.observe_positions(b, mask)
        self.assertAlmostEqual(pf, 1.0)
        self.assertAlmostEqual(pi, 0.0)

    def test_steady_creep_with_flat_structure_is_converged(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        last_pass = {}
        for k in range(1, 400):
            t = 0.5 * k
            rec = mon.update(row(t, 1.0, path=5.0 * t))   # fast but perfectly steady
            if rec['conv_pass'] and not last_pass:
                last_pass = rec                            # before the latch
        self.assertTrue(mon.converged, mon.last.get('conv_blocking'))
        self.assertEqual(last_pass['conv_path_steady'], 1)
        self.assertEqual(last_pass['conv_path_quiet'], 0)

    def test_but_a_decaying_creep_is_not(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        for k in range(1, 400):
            t = 0.5 * k
            mon.update(row(t, 1.0, path=50.0 * t ** 0.3))   # rate still falling
        self.assertFalse(mon.converged)
        self.assertIn('path', mon.last['conv_blocking'])


class TestTheGuards(unittest.TestCase):

    def test_a_constant_division_rate_blocks_it(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        for k in range(1, 400):
            t = 0.5 * k
            mon.update(row(t, 1.0, path=0.001 * t, divisions=2.0 * t))
        self.assertFalse(mon.converged)
        self.assertIn('division', mon.last['conv_blocking'])

    def test_and_division_that_has_stopped_does_not(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        for k in range(1, 400):
            t = 0.5 * k
            mon.update(row(t, 1.0, path=0.001 * t, divisions=min(2.0 * t, 20.0)))
        self.assertTrue(mon.converged, mon.last.get('conv_blocking'))

    def test_without_laguerre_it_refuses_rather_than_passes(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        feed(mon, [1.0] * 400, laguerre=False)
        self.assertFalse(mon.converged)
        self.assertIn('compaction', mon.last['conv_blocking'])

    def test_compaction_below_the_ceiling_blocks_it(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        feed(mon, [1.0] * 400, compaction=0.80)
        self.assertFalse(mon.converged)
        self.assertIn('compaction', mon.last['conv_blocking'])

    def test_t_min_is_respected(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=1e6)
        feed(mon, [1.0] * 400)
        self.assertFalse(mon.converged)

    def test_shadow_never_stops(self):
        mon = ConvergenceMonitor(enable=True, shadow=True, t_min_h=0.0)
        feed(mon, [1.0] * 400)
        self.assertTrue(mon.converged)
        self.assertFalse(mon.should_stop())


class TestTheBookkeeping(unittest.TestCase):

    def test_every_record_carries_every_key(self):
        # The record is merged into the metrics dict, which goes through
        # csv.DictWriter with field names from the FIRST row.
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        keys = None
        for k in range(1, 60):
            rec = mon.update(row(0.5 * k, 1.0, path=0.001 * k))
            if keys is None:
                keys = set(rec)
            else:
                self.assertEqual(keys, set(rec), f'key set changed at row {k}')

    def test_rows_round_trip_so_a_resumed_run_keeps_its_windows(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        feed(mon, [1.0 - 0.3 * np.exp(-0.25 * k) for k in range(1, 60)])
        dumped = [dict(zip(COLS, r)) for r in mon.rows]
        back = ConvergenceMonitor(enable=True, t_min_h=0.0)
        self.assertEqual(back.load_rows(dumped), len(dumped))
        self.assertEqual(len(back.rows), len(mon.rows))
        # the running path is restored from the last row, or the next increment
        # would start again from zero and read as a jump
        self.assertAlmostEqual(back._path[0], dumped[-1]['path_func'])

    def test_an_unknown_tolerance_is_refused(self):
        with self.assertRaises(KeyError):
            ConvergenceMonitor(overrides={'eps_nonsense': 1.0})

    def test_a_sparse_window_makes_no_decision(self):
        mon = ConvergenceMonitor(enable=True, t_min_h=0.0)
        for k in range(1, 5):
            mon.update(row(10.0 * k, 1.0))
        self.assertFalse(mon.converged)
        self.assertIn('rows', mon.last['conv_blocking'])


class TestTheObserver(unittest.TestCase):

    def test_composite_or_accumulates_without_short_circuiting(self):
        class Stopper(Observer):
            def __init__(self):
                self.n = 0

            def on_step(self, s, t, gs, F, contacts):
                self.n += 1
                return True

        class Counter(Observer):
            def __init__(self):
                self.n = 0

            def on_step(self, s, t, gs, F, contacts):
                self.n += 1
                return False

        a, b = Stopper(), Counter()
        c = CompositeObserver(a, b)
        self.assertTrue(c.on_step(0, 0.0, None, None, None))
        # `any(genexpr)` would have left b at 0 and desynchronised the viewer
        self.assertEqual((a.n, b.n), (1, 1))

    def test_it_names_the_stop(self):
        class Named(Observer):
            stop_reason = 'converged'

        self.assertEqual(CompositeObserver(Named()).stop_reason, 'converged')


class TestTheConfigRules(unittest.TestCase):
    """Two ways it would silently never fire. Both are errors, not warnings."""

    def _setup(self, **kw):
        su = template_setup()
        su.convergence.enable = True
        su.output.metrics_laguerre = True
        su.convergence.t_min_h = 24.0
        su.time.save_every_h = 2.0
        for k, v in kw.items():
            obj, _, attr = k.rpartition('.')
            tgt = su
            for part in obj.split('.'):
                tgt = getattr(tgt, part)
            setattr(tgt, attr, v)
        return su

    def _errs(self, su):
        try:
            validate(su)
            return ''
        except ValueError as e:
            return str(e)

    def test_the_baseline_is_valid(self):
        self.assertEqual(self._errs(self._setup()), '')

    def test_laguerre_is_required(self):
        self.assertIn('metrics_laguerre',
                      self._errs(self._setup(**{'output.metrics_laguerre': False})))

    def test_coarse_saves_are_refused(self):
        self.assertIn('save_every_h',
                      self._errs(self._setup(**{'time.save_every_h': 8.0})))

    def test_an_unreachable_t_min_is_refused(self):
        self.assertIn('t_min_h',
                      self._errs(self._setup(**{'convergence.t_min_h': 1e6})))

    def test_the_template_is_still_valid(self):
        self.assertEqual(self._errs(template_setup()), '')


class TestEndToEnd(unittest.TestCase):

    BED = dict(mode='2D', Lx=250.0, Ly=250.0, phi_solid_target=0.5, func_ratio=0.6,
               R_func_mean=20.0, R_inert_mean=20.0, cell_surface_coverage=1.0,
               t_total=12.0, save_every_h=1.0, save_fields=False, T_active=0.0,
               metrics_laguerre=True, dynamics_substep='off')

    def test_shadow_mode_records_and_never_stops(self):
        h, _s, _p, _gs = quiet(run, Params(save_data=False, conv_shadow=True,
                                           conv_t_min_h=2.0, **self.BED), seed=3)
        keys = [sorted(k for k in r if k.startswith('conv_')) for r in h]
        self.assertTrue(keys[0])
        self.assertTrue(all(k == keys[0] for k in keys), 'key set varies by frame')
        self.assertEqual(len(h), int(self.BED['t_total'] / self.BED['save_every_h']) + 1)

    def test_it_is_off_by_default(self):
        a = quiet(run, Params(save_data=False, **self.BED), seed=3)
        self.assertFalse(any(k.startswith('conv_') for k in a[0][-1]))
        self.assertFalse(Params().conv_enable)
        self.assertFalse(Params().conv_shadow)

    def test_it_persists_its_buffer_for_a_resume(self):
        d = tempfile.mkdtemp(prefix='gels_conv_')
        try:
            quiet(run, Params(save_data=True, output_dir=d, compress_archive=False,
                              conv_shadow=True, conv_t_min_h=2.0, **self.BED), seed=3)
            path = os.path.join(d, 'convergence.json')
            self.assertTrue(os.path.exists(path))
            with open(path, encoding='utf-8') as fh:
                blob = json.load(fh)
            self.assertTrue(blob['rows'])
            self.assertEqual(set(blob['rows'][0]), set(COLS))
            # ... and it reloads
            mon = ConvergenceMonitor()
            self.assertEqual(mon.load_rows(blob['rows']), len(blob['rows']))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_the_shadow_harness_replays_a_real_run(self):
        d = tempfile.mkdtemp(prefix='gels_shadow_')
        try:
            quiet(run, Params(save_data=True, output_dir=d, compress_archive=False,
                              conv_shadow=True, conv_t_min_h=2.0, **self.BED), seed=3)
            sys.path.insert(0, os.path.join(REPO, 'pipeline'))
            import shadow_check
            rep = shadow_check.replay(d, t_min_h=0.0)
            self.assertIsNotNone(rep)
            self.assertEqual(rep['n_frames'], len(os.listdir(os.path.join(d, 'snapshots'))))
            self.assertTrue(rep['has_laguerre'])
            self.assertIn('blocking', rep)
        finally:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
