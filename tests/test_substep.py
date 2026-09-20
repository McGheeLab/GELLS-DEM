"""
dt is a coupling interval, not an integration step (V3.6 Phase 2).

`contact_semi_implicit` takes vel = F/(gamma + dt k). At the fixed point F = 0,
so the equilibrium is exact for ANY dt -- that is what makes it unconditionally
stable and why V3.2 adopted it. But the rate it gives is F/(gamma + dt k) where
the overdamped rate is F/gamma, so with

    S = dt k / gamma

every granule moves (1 + S) times too slowly.

A packing, a sedimentation or a relaxation ends AT a fixed point and does not
care. A cell-driven compaction run's entire answer is a rate, and it cares:

    2D bed, 24 h, 3 seeds, velocity cap lifted so it cannot confound

      dt (h)   F_mean (nN)    disp_func (um)   overlap_clip_fraction
      0.5      136 +- 21      11.3 +- 1.3      0.062
      0.1      85.7 +- 3.9    25.2 +- 2.0      0.038
      0.02     40.1 +- 5.1    36.8 +- 0.2      0.000
      0.005    29.7 +- 2.3    48.4 +- 3.7      0.000

    and S at the shipped dt = 0.5 h averages 41 and reaches 88, with 90 % of
    granules above 1.

`dynamics.substep = auto` fixes it by making dt the interval at which the run is
SAMPLED and advancing the mechanics inside it in n_sub substeps sized so S stays
near `dynamics.substep_target`. Everything inside `step` is already a rate times
dt -- bridge formation is 1 - exp(-rate dt), the cell clocks are += dt, the
active noise is sqrt(2 gamma T / dt) -- so a substep is an ordinary `step` with a
smaller dt, not a special mechanics-only path.

Two things are load-bearing:

* **`auto` declines a passive run.** Measured: a gravity bed at dt = 0.5 h has a
  bed height 0.22 um out of 444 from the same bed at dt = 0.005 h, i.e. 0.05 %,
  pre-loaded or not. Substepping it would be 256x the cost for a fifth of a
  micron.
* **`v_max` is scaled with the subdivision**, so it caps displacement per
  coupling interval rather than speed per substep. Without that, subdividing
  tightens the absolute rail silently -- fixed dt = 0.005 h clips 49 % of
  granules where dt = 0.5 h clips none, purely because the semi-implicit damping
  is no longer suppressing the velocity -- and the run trades the overlap rail
  for the velocity rail with nobody told.

Run:  python -m unittest tests.test_substep -v
"""

import contextlib
import io
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gels.engine as E  # noqa: E402
from gels.engine import (  # noqa: E402
    Params, advance, compute_forces, coordination_class, generate_packing,
    population_speed_cap, run, step, stiffness_rate, substep_count, substep_mode,
    substep_metrics,
)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


CELLS = dict(
    mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.55, func_ratio=0.6,
    save_data=False, save_fields=False, t_total=2.0, save_every_h=1.0,
    R_func_mean=20.0, R_func_std=3.0, R_inert_mean=20.0, R_inert_std=3.0,
    cell_surface_coverage=1.0, T_active=0.0,
)
PASSIVE = dict(
    mode='2D', Lx=300.0, Ly=600.0, phi_solid_target=0.30,
    save_data=False, save_fields=False, t_total=2.0, save_every_h=1.0,
    R_func_mean=20.0, R_func_std=0.0, R_inert_mean=20.0, R_inert_std=0.0,
    n_cells_per_granule=0, cell_coverage=0.0, cell_surface_coverage=0.0,
    gravity_enabled=True, granule_density=1180.0, boundary_top='free',
    packing_consolidation='gravity', packing_settle_steps=120, T_active=0.0,
)


def bed(cells=True, seed=3, **over):
    p = Params(**dict(CELLS if cells else PASSIVE, **over))
    gs = quiet(generate_packing, p, seed=seed)
    F, _tq, c = quiet(compute_forces, gs, p, np.random.default_rng(0))
    return p, gs, F, c


class TestTheNumber(unittest.TestCase):
    """S = dt k/gamma, and that it really is as large as claimed."""

    def test_the_rate_is_free_of_dt(self):
        p_a, gs, _F, c = bed()
        p_b = Params(**dict(CELLS, dt=0.01))
        np.testing.assert_allclose(stiffness_rate(gs, p_a, c),
                                   stiffness_rate(gs, p_b, c), rtol=0, atol=0)

    def test_the_shipped_dt_runs_at_S_far_above_one(self):
        p, gs, _F, c = bed()
        S = stiffness_rate(gs, p, c) * p.dt
        self.assertGreater(float(np.percentile(S, 95)), 10.0)
        self.assertGreater(float(np.mean(S > 1.0)), 0.5)

    def test_the_wall_stiffness_is_in_it(self):
        # V3.6 Phase 1 put wall contacts into `contact_stiffness_per_granule`,
        # so a bed resting on a wall raises its own substep count.
        p, gs, _F, c = bed(cells=False)
        onw = np.asarray(gs.wall_stiffness[:gs.N]) > 0
        self.assertGreater(int(onw.sum()), 0)
        self.assertTrue(np.all(stiffness_rate(gs, p, c)[onw] > 0.0))


class TestTheController(unittest.TestCase):

    def _gs(self, rate, **ov):
        p = Params(**dict(CELLS, **ov))
        gs = type('G', (), {})()
        gs.stiffness_rate_p95 = rate
        gs.n_substeps = 1
        gs.cell_offset = np.array([0, 1])
        gs.N = 0
        return p, gs

    def test_it_solves_dt_rate_over_target(self):
        p, gs = self._gs(100.0, dt=0.5, dynamics_substep='always',
                         dynamics_substep_target=0.2, dynamics_substep_max=10000)
        self.assertEqual(substep_count(gs, p), int(np.ceil(0.5 * 100.0 / 0.2)))

    def test_the_budget_binds_and_says_so(self):
        p, gs = self._gs(100.0, dt=0.5, dynamics_substep='always',
                         dynamics_substep_target=0.2, dynamics_substep_max=8)
        self.assertEqual(substep_count(gs, p), 8)
        self.assertTrue(gs.substep_budget_bound)

    def test_it_is_one_when_the_bed_is_soft_enough(self):
        p, gs = self._gs(0.1, dt=0.5, dynamics_substep='always',
                         dynamics_substep_target=0.2)
        self.assertEqual(substep_count(gs, p), 1)
        self.assertFalse(gs.substep_budget_bound)

    def test_off_never_substeps(self):
        p, gs = self._gs(1e6, dynamics_substep='off')
        self.assertEqual(substep_count(gs, p), 1)

    def test_auto_takes_the_load_into_account(self):
        _p, gs_cells, _F, _c = bed(cells=True)
        _p2, gs_passive, _F2, _c2 = bed(cells=False)
        for mode, want_cells, want_passive in (('auto', 'always', 'off'),
                                               ('always', 'always', 'always'),
                                               ('off', 'off', 'off')):
            with self.subTest(mode=mode):
                p = Params(**dict(CELLS, dynamics_substep=mode))
                self.assertEqual(substep_mode(gs_cells, p), want_cells)
                self.assertEqual(substep_mode(gs_passive, p), want_passive)


class TestAdvance(unittest.TestCase):

    def test_one_substep_is_exactly_step(self):
        kw = dict(dynamics_substep='off', dynamics_outlier_speed=0.0)
        p_a, gs_a, _F, _c = bed(**kw)
        p_b, gs_b, _F, _c = bed(**kw)
        rng_a, rng_b = np.random.default_rng(1), np.random.default_rng(1)
        for s in range(3):
            t = (s + 1) * p_a.dt
            quiet(step, gs_a, p_a, rng_a, t)
            quiet(advance, gs_b, p_b, rng_b, t)
        np.testing.assert_array_equal(gs_a.pos[:gs_a.N], gs_b.pos[:gs_b.N])

    def test_it_restores_dt_and_v_max(self):
        p, gs, _F, _c = bed(dynamics_substep='always')
        gs.stiffness_rate_p95 = 100.0
        dt0, v0 = p.dt, p.v_max
        quiet(advance, gs, p, np.random.default_rng(0), p.dt)
        self.assertEqual((p.dt, p.v_max), (dt0, v0))
        self.assertGreater(gs.n_substeps, 1)

    def test_it_restores_them_after_a_failure(self):
        p, gs, _F, _c = bed(dynamics_substep='always')
        gs.stiffness_rate_p95 = 100.0
        dt0, v0 = p.dt, p.v_max
        orig = E.step

        def boom(*a, **kw):
            raise RuntimeError('force evaluation failed')

        E.step = boom
        try:
            with self.assertRaises(RuntimeError):
                advance(gs, p, np.random.default_rng(0), p.dt)
        finally:
            E.step = orig
        self.assertEqual((p.dt, p.v_max), (dt0, v0))

    def test_the_velocity_cap_is_scaled_so_subdividing_does_not_tighten_it(self):
        # The measurement: fixed small dt clips ~half the bed; the same physical
        # resolution reached by substepping clips none.
        p_fix = Params(**dict(CELLS, dt=0.5 / 64, dynamics_substep='off',
                              dynamics_outlier_speed=0.0, t_total=2.0))
        h_fix, _s, _p, _gs = quiet(run, p_fix, seed=3)
        p_sub = Params(**dict(CELLS, dt=0.5, dynamics_substep='always',
                              dynamics_substep_max=64, dynamics_substep_target=1e-9,
                              dynamics_outlier_speed=0.0))
        h_sub, _s, _p, gs_sub = quiet(run, p_sub, seed=3)
        self.assertEqual(gs_sub.n_substeps, 64)
        self.assertGreater(float(h_fix[-1]['frac_velocity_clipped']), 0.05)
        self.assertEqual(float(h_sub[-1]['frac_velocity_clipped']), 0.0)


class TestItChangesTheAnswerInTheRightDirection(unittest.TestCase):
    """The convergence claim, at a size that runs in seconds."""

    @classmethod
    def setUpClass(cls):
        base = dict(CELLS, t_total=6.0, save_every_h=6.0)
        cls.off = quiet(run, Params(**dict(base, dynamics_substep='off',
                                           dynamics_outlier_speed=0.0)), seed=3)[0][-1]
        cls.auto = quiet(run, Params(**dict(base, dynamics_substep='auto',
                                            dynamics_outlier_speed=0.0)), seed=3)[0][-1]

    def test_the_stiffness_number_comes_down_to_target(self):
        self.assertGreater(float(self.off['stiffness_number']), 10.0)
        self.assertLess(float(self.auto['stiffness_number']), 1.0)

    def test_the_bed_comes_off_the_overlap_rail(self):
        self.assertLessEqual(float(self.auto['overlap_clip_fraction']),
                             float(self.off['overlap_clip_fraction']))
        self.assertEqual(float(self.auto['overlap_clip_fraction']), 0.0)

    def test_the_granules_move_further_and_push_less_hard(self):
        # Both follow from the rate no longer being (1+S)x too slow.
        self.assertGreater(float(self.auto['disp_func']), float(self.off['disp_func']))
        self.assertLess(float(self.auto['F_mean']), float(self.off['F_mean']))

    def test_a_passive_run_is_untouched_by_auto(self):
        base = dict(PASSIVE, t_total=2.0)
        a = quiet(run, Params(**dict(base, dynamics_substep='off')), seed=5)
        b = quiet(run, Params(**dict(base, dynamics_substep='auto')), seed=5)
        np.testing.assert_array_equal(a[3].pos[:a[3].N], b[3].pos[:b[3].N])
        self.assertEqual(b[0][-1]['n_substeps'], 1)


class TestThePopulationRail(unittest.TestCase):
    """A granule should not differ much from granules in its own condition."""

    def test_classes_are_contact_counts(self):
        _p, gs, _F, c = bed()
        z = coordination_class(gs, c)
        self.assertEqual(z.shape, (gs.N,))
        self.assertTrue(np.all(z >= 0) and np.all(z <= 6))

    def test_a_planted_outlier_is_capped_and_its_peers_are_not(self):
        p, gs, _F, c = bed(dynamics_outlier_speed=8.0)
        speed = np.full(gs.N, 1.0)
        speed[0] = 500.0
        cap = population_speed_cap(gs, p, speed, c)
        self.assertIsNotNone(cap)
        # every class large enough to have a median caps the outlier
        if np.isfinite(cap[0]):
            self.assertLess(cap[0], speed[0])
        finite = np.isfinite(cap)
        self.assertTrue(np.all(speed[finite & (np.arange(gs.N) != 0)]
                               <= cap[finite & (np.arange(gs.N) != 0)]))

    def test_a_uniform_population_is_never_clipped(self):
        p, gs, _F, c = bed(dynamics_outlier_speed=8.0)
        speed = np.full(gs.N, 3.0)
        cap = population_speed_cap(gs, p, speed, c)
        self.assertTrue(np.all(speed <= cap))

    def test_a_population_at_rest_is_never_clipped(self):
        # mad = 0 AND med = 0 would give a cap of 0; it must not.
        p, gs, _F, c = bed(dynamics_outlier_speed=8.0)
        cap = population_speed_cap(gs, p, np.zeros(gs.N), c)
        self.assertTrue(np.all(np.asarray(cap) >= 0.0))

    def test_a_class_too_small_to_be_a_population_is_left_alone(self):
        p, gs, _F, c = bed(dynamics_outlier_speed=8.0)
        z = coordination_class(gs, c)
        counts = np.bincount(z, minlength=7)
        speed = np.full(gs.N, 1.0)
        cap = population_speed_cap(gs, p, speed, c)
        for k in np.nonzero(counts)[0]:
            if counts[k] < E.OUTLIER_MIN_CLASS:
                self.assertTrue(np.all(~np.isfinite(cap[z == k])))

    def test_off_is_bit_identical(self):
        kw = dict(dynamics_substep='off')
        a = quiet(run, Params(**dict(CELLS, dynamics_outlier_speed=0.0, **kw)), seed=3)
        b = quiet(run, Params(**dict(CELLS, dynamics_outlier_speed=0.0, **kw)), seed=3)
        np.testing.assert_array_equal(a[3].pos[:a[3].N], b[3].pos[:b[3].N])
        self.assertEqual(a[0][-1]['frac_outlier_clipped'], 0.0)

    def test_the_two_rails_never_double_count(self):
        p, gs, _F, c = bed(dynamics_outlier_speed=8.0, v_max=0.5)
        speed = np.full(gs.N, 1.0)
        speed[0] = 500.0
        over = speed > p.v_max                       # everything, here
        E._speed_rails(gs, p, speed, c, over)
        self.assertEqual(gs.frac_velocity_clipped, 1.0)
        self.assertEqual(gs.frac_outlier_clipped, 0.0)


class TestTheMetrics(unittest.TestCase):

    def test_the_keys_are_plain_finite_scalars(self):
        _p, gs, _F, _c = bed()
        m = substep_metrics(gs)
        for k, v in m.items():
            self.assertIsInstance(v, (int, float, bool), k)
            self.assertTrue(np.isfinite(float(v)), k)

    def test_both_twins_report_them(self):
        for numba in (True, False):
            with self.subTest(use_numba=numba):
                h, _s, _p, _gs = quiet(run, Params(**dict(CELLS, use_numba=numba)), seed=3)
                for k in substep_metrics(_gs):
                    self.assertIn(k, h[-1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
