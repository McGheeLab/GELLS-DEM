"""V3.8: the cell's surface search is a persistent random walk.

Until V3.8 the walk drew ``N(0, v*dt/r)`` -- a BALLISTIC displacement used as
the width of a DIFFUSIVE increment. Variance adds, so the search over a window
``T`` in steps of ``dt`` went as ``(v/r) sqrt(T dt)``: it depended on the
timestep. V3.6 made that bite, because ``dt`` became a coupling interval and
``advance()`` subdivides it 129-330x on the flagship preset, suppressing the
search by ``sqrt(n_sub)`` -- measured 11.75x.

Two traps cost a wrong measurement each while this was being found, and both are
pinned below so they cannot cost another:

  * ``fibroblast_realistic`` has division ON, and ``add_cells`` rebuilds the CSR,
    so an index-wise comparison of ``cell_theta_local`` across it silently
    compares DIFFERENT cells;
  * ``pi/sqrt(3) = 1.8138 rad`` is the RMS of a uniform distribution on a
    circle, so any sufficiently long window reads that value whatever the
    walk does, and two runs look identical when they are 11.75x apart.
"""

import contextlib
import io
import math
import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.config import template_setup, setup_to_params          # noqa: E402
from gels.presets import apply_presets                            # noqa: E402
from gels.engine import (crowd_mobility, cell_capacity,           # noqa: E402
                         generate_packing, run)
from gels.celltypes import available, get                         # noqa: E402

SATURATION = math.pi / math.sqrt(3.0)      # RMS of a uniform circle, 1.8138 rad


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def search_run(substep='off', T=2.0, numba=False, seed=11, **over):
    """RMS surface arc travelled in T hours. Division OFF: see the module note."""
    s = template_setup()
    apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
    p = setup_to_params(s)
    p.save_data = False
    p.t_total = T
    p.save_every_h = T
    p.dynamics_substep = substep
    p.cell_division_enabled = False          # keep the CSR stable
    p.perf_keep_snaps_in_memory = True
    p.use_numba = numba
    for k, v in over.items():
        setattr(p, k, v)
    hist, snaps, p2, gs = quiet(run, p, seed=seed)
    th0 = np.asarray(snaps[0]['cell_theta_local'], float)
    th1 = np.asarray(snaps[-1]['cell_theta_local'], float)
    assert th0.size == th1.size, "CSR moved: division must be off"
    d = np.abs((th1 - th0 + np.pi) % (2 * np.pi) - np.pi)
    rms = float(np.sqrt(np.mean(d ** 2)))
    return hist[-1], rms, rms * float(np.mean(gs.r[:gs.N]))


class TestDtIndependence(unittest.TestCase):
    """The bug. The search must not care how the interval was subdivided."""

    def test_substepping_does_not_suppress_the_search(self):
        _, rms_off, arc_off = search_run('off', T=2.0)
        row, rms_auto, arc_auto = search_run('auto', T=2.0)
        self.assertGreater(int(row.get('n_substeps', 1)), 20,
                           "this test is vacuous unless auto actually subdivides")
        self.assertLess(rms_off, SATURATION * 0.95,
                        "measurement is saturated; shorten T")
        ratio = arc_off / max(arc_auto, 1e-12)
        # pre-V3.8 this was 11.75x. What remains is the coarse resolution of the
        # heading process at dt/tau = 0.5, not a scaling error.
        self.assertLess(ratio, 1.5, f"substepping still suppresses the walk: {ratio:.2f}x")
        self.assertGreater(ratio, 0.67, f"substepping now inflates the walk: {ratio:.2f}x")


class TestLimits(unittest.TestCase):
    """MSD(t) = 2 d D [t - tau(1 - e^{-t/tau})] must be right at both ends."""

    def test_ballistic_for_t_much_less_than_tau(self):
        ct = get('fibroblast')
        v = ct.migration_speed_um_per_h.value
        t = 0.004                                  # a substep
        self.assertAlmostEqual(ct.search_distance_um(t) / (v * t), 1.0, places=2)

    def test_diffusive_for_t_much_greater_than_tau(self):
        ct = get('fibroblast')
        tau = ct.persistence_time_h.value
        D = ct.surface_diffusivity_um2_per_h()
        t = 200.0 * tau
        self.assertAlmostEqual(
            ct.search_distance_um(t) / math.sqrt(2 * 2 * D * t), 1.0, places=2)

    def test_the_diffusive_form_overestimates_at_short_times(self):
        """Why the walk carries a heading instead of just using sqrt(2 D dt)."""
        ct = get('fibroblast')
        D = ct.surface_diffusivity_um2_per_h()
        t = 0.5
        pure = math.sqrt(2 * 2 * D * t)
        self.assertGreater(pure / ct.search_distance_um(t), 2.0)

    def test_D_is_v_squared_tau_over_two(self):
        ct = get('fibroblast')
        v = ct.migration_speed_um_per_h.value
        tau = ct.persistence_time_h.value
        self.assertAlmostEqual(ct.surface_diffusivity_um2_per_h(), v * v * tau / 2.0)


class TestCrowding(unittest.TestCase):
    """mobility = (1 - theta) + theta * p_climb, p_climb = f_cell_cell."""

    def setUp(self):
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        self.p = setup_to_params(s)
        self.p.save_data = False
        self.gs = quiet(generate_packing, self.p, seed=11)
        self.occupied = [i for i in range(self.gs.N)
                         if cell_capacity(self.gs, i, self.p) > 0]
        self.assertTrue(self.occupied, "no granule has capacity to crowd")

    def test_parity_at_f_cell_cell_one(self):
        """No preference for the granule means no crowding: the pre-V3.8 walk."""
        self.p.f_cell_cell = 1.0
        for i in self.occupied:
            self.assertAlmostEqual(crowd_mobility(self.gs, i, self.p), 1.0)

    def test_disabled_is_exactly_one(self):
        self.p.cell_crowding_enabled = False
        for i in self.occupied:
            self.assertAlmostEqual(crowd_mobility(self.gs, i, self.p), 1.0)

    def test_pure_exclusion_at_f_cell_cell_zero(self):
        """A cell that will not climb has mobility exactly 1 - theta."""
        self.p.f_cell_cell = 0.0
        for i in self.occupied:
            cap = cell_capacity(self.gs, i, self.p)
            th = min(float(self.gs.n_attached[i]) / cap, 1.0)
            self.assertAlmostEqual(crowd_mobility(self.gs, i, self.p), 1.0 - th)

    def test_it_is_bounded_and_monotone_in_f_cell_cell(self):
        vals = []
        for fcc in (0.0, 0.15, 0.5, 1.0):
            self.p.f_cell_cell = fcc
            m = crowd_mobility(self.gs, self.occupied[0], self.p)
            self.assertGreaterEqual(m, 0.0)
            self.assertLessEqual(m, 1.0)
            vals.append(m)
        self.assertEqual(vals, sorted(vals))

    def test_an_empty_granule_is_unhindered(self):
        self.p.f_cell_cell = 0.0
        for i in range(self.gs.N):
            if cell_capacity(self.gs, i, self.p) > 0 and self.gs.n_attached[i] == 0:
                self.assertAlmostEqual(crowd_mobility(self.gs, i, self.p), 1.0)
                return


class TestCrowdingIsObservable(unittest.TestCase):
    """A mobility factor that silently multiplies the search must be reported."""

    @staticmethod
    def _row(f_cell_cell, crowding):
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        p.save_data = False
        p.t_total = 2.0
        p.save_every_h = 2.0
        p.f_cell_cell = f_cell_cell
        p.cell_crowding_enabled = crowding
        return quiet(run, p, seed=11)[0][-1]

    def test_the_metrics_report_what_crowding_did(self):
        r = self._row(0.15, True)
        self.assertGreater(r['cell_occupancy_mean'], 0.0)
        self.assertLess(r['cell_crowd_mobility'], 1.0)
        # mobility = (1 - th) + th * f_cell_cell, on the reported mean occupancy
        th = r['cell_occupancy_mean']
        self.assertAlmostEqual(r['cell_crowd_mobility'], (1 - th) + th * 0.15,
                               delta=0.02)

    def test_parity_is_exact_end_to_end(self):
        """f_cell_cell = 1 must be indistinguishable from crowding OFF.

        Not a statistical claim: the mobility is identically 1.0 at every
        occupancy, so the two runs consume the same RNG and take the same path.
        This is the falsifiable statement of "crowding is off by construction at
        parity" -- the same discipline V3.6's stacking parity test follows.
        """
        a = self._row(1.0, True)
        b = self._row(0.15, False)
        self.assertAlmostEqual(a['cell_crowd_mobility'], 1.0)
        self.assertAlmostEqual(b['cell_crowd_mobility'], 1.0)
        self.assertEqual(a['n_bridges'], b['n_bridges'])
        self.assertAlmostEqual(a['F_mean'], b['F_mean'], places=9)


class TestSeedingStream(unittest.TestCase):
    """The heading draw must not touch the packing stream."""

    def test_headings_are_randomised_but_cost_the_packing_nothing(self):
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        p.save_data = False
        gs = quiet(generate_packing, p, seed=11)
        C = gs.total_cells
        self.assertGreater(C, 50)
        h = np.asarray(gs.cell_heading[:C])
        # 2D: a sign. Both signs must be present, and balanced to ~1/sqrt(C).
        self.assertEqual(set(np.unique(h).tolist()), {-1.0, 1.0})
        self.assertLess(abs(float(h.mean())), 5.0 / math.sqrt(C))

    def test_the_draw_comes_from_a_spawned_child_not_the_parent(self):
        """`spawn` derives from the seed sequence without advancing the parent.

        Taking the headings from `rng` shifted every later draw and moved the
        seeded cell-age CV from 0.577 to 0.706 -- caught by
        `test_seeding_draws_nothing_when_division_is_off`, which contracts
        seeding to consume exactly the surface-angle draws when division is off.
        """
        a = np.random.default_rng(4)
        b = np.random.default_rng(4)
        _ = a.spawn(1)[0].random(64)
        np.testing.assert_array_equal(a.random(5), b.random(5))


class TestCellTypesMustStateIt(unittest.TestCase):

    def test_every_type_states_a_persistence_time(self):
        """It is a REQUIRED field, so a type cannot be defined without one."""
        for name in available():
            ct = get(name)
            self.assertTrue(hasattr(ct, 'persistence_time_h'), name)
            self.assertGreater(ct.persistence_time_h.value, 0.0, name)
            self.assertTrue(ct.persistence_time_h.source.strip(), name)

    def test_the_fibroblast_value_is_sourced_not_assumed(self):
        ct = get('fibroblast')
        self.assertIn('Gail', ct.persistence_time_h.source)
        self.assertEqual(ct.persistence_time_h.value, 1.0)

    def test_msc_declares_its_value_an_assumption(self):
        """msc has not been through a literature review and must say so."""
        from gels.celltypes.base import ASSUMPTION
        self.assertIn(ASSUMPTION, get('msc').persistence_time_h.source)

    def test_it_reaches_params_through_the_overrides(self):
        ov = '\n'.join(get('fibroblast').to_overrides())
        self.assertIn('cells.migration.persistence_time_h=1', ov)


class TestTheTrapsThatCostMeasurements(unittest.TestCase):
    """Both of these made a real measurement read the wrong answer."""

    def test_pi_over_sqrt3_is_the_saturation_value(self):
        rng = np.random.default_rng(0)
        u = rng.uniform(-np.pi, np.pi, 400000)
        self.assertAlmostEqual(float(np.sqrt(np.mean(u ** 2))), SATURATION, places=2)

    def test_division_moves_the_csr_and_invalidates_cell_indices(self):
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        p.save_data = False
        p.t_total = 24.0
        p.save_every_h = 24.0
        p.perf_keep_snaps_in_memory = True
        hist, snaps, p2, gs = quiet(run, p, seed=11)
        self.assertNotEqual(len(snaps[0]['cell_theta_local']),
                            len(snaps[-1]['cell_theta_local']),
                            "division did not add cells, so the trap is not demonstrated")


class TestTwins(unittest.TestCase):

    def test_the_twins_agree_on_the_search(self):
        _, _, arc_ref = search_run('off', T=2.0, numba=False)
        _, _, arc_ker = search_run('off', T=2.0, numba=True)
        # bridging is statistically equivalent, not bit-identical (V3.0), so the
        # cell population diverges slightly; the walk itself must still match.
        self.assertLess(abs(arc_ref - arc_ker) / max(arc_ref, 1e-12), 0.05)


if __name__ == '__main__':
    unittest.main()
