"""
FIRE relaxation: the packer hands the dynamics a bed in force balance (V3.5)
===========================================================================

V3.4 measured the disease and shipped the detector: the packer hands the
dynamics a bed pre-loaded far above the driving load, because the settle stops
on a LENGTH tolerance measured with a proxy that knows nothing about the
contact law the dynamics will apply. The first hours of such a run are that
unwinding, not the physics asked for.

``packing.relax = 'fire'`` is the fix. It runs AFTER the settle, in
``generate_packing``, where ``compute_forces`` is an ordinary local call, and
relaxes the bed under the **exact** force law the dynamics will use -- stopping
on a force tolerance in multiples of ``dynamics_load_scale``, the same scale the
detector reports.

The V3.3 plan specified FIRE reusing the settle's own ``k_rep ov^1.5`` law to
dodge a circular import. ``test_fire_stops_in_the_dynamics_own_units`` is the
test that would have caught why that could not have worked: the handoff error is
a mismatch BETWEEN two force laws, so balance under the settle's says nothing
about ``max|F|`` under JKR.

Run:  python -m unittest tests.test_fire_relax -v
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

from gels.engine import (  # noqa: E402
    Params, apply_position_bounds, compute_forces, compute_forces_3d,
    constraint_clamped, dynamics_load_scale, free_force_residual,
    generate_packing, generate_packing_3d, handoff_force_balance,
    free_force_percentile, penetration_stats, relax_mode, settle_overlap_tolerance,
    wall_clamp_margin,
)

# A shaped, gravity-consolidated bed: the configuration the V3.4 finding was
# measured on, shrunk until it runs in a few seconds.
SHAPED_BED = dict(
    mode='2D', Lx=600.0, Ly=900.0, phi_solid_target=0.45,
    save_data=False, save_fields=False,
    shape_enabled=True, aspect_ratio_func_mean=1.8, blockiness_func_mean=3.5,
    aspect_ratio_inert_mean=1.8, blockiness_inert_mean=3.5,
    R_func_mean=18.0, R_func_std=2.0, R_inert_mean=22.0, R_inert_std=2.0,
    n_cells_per_granule=0, cell_coverage=0.0, cell_surface_coverage=0.0,
    gravity_enabled=True, granule_density=1180.0, boundary_top='free',
    packing_consolidation='gravity', packing_settle_steps=150, T_active=0.0,
)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def bed(relax='none', three_d=False, seed=7, **over):
    base = dict(SHAPED_BED)
    base.update(over)
    p = Params(packing_relax=relax, **base)
    gen = generate_packing_3d if three_d else generate_packing
    gs = quiet(gen, p, seed=seed)
    F, _tq, c = quiet(compute_forces_3d if three_d else compute_forces,
                      gs, p, np.random.default_rng(0))
    return p, gs, F, c


class TestTheHandoff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p0, cls.gs0, cls.F0, cls.c0 = bed('none')
        cls.p1, cls.gs1, cls.F1, cls.c1 = bed('fire')

    def _ratio(self, gs, p, F):
        scale, _kind = dynamics_load_scale(gs, p)
        self.assertIsNotNone(scale, "this bed is supposed to have gravity")
        free, _held = free_force_residual(gs, p, F)
        return free / scale

    def test_the_settle_alone_hands_over_a_badly_loaded_bed(self):
        """The disease, restated as a test so it cannot quietly go away."""
        self.assertGreater(self._ratio(self.gs0, self.p0, self.F0), 50.0)

    def test_fire_brings_it_into_balance(self):
        """Judged on the p95, not the max.

        `max|F|` over a loose bed is a max over a heavy tail: one wedged
        granule read 132x the gravity load on this bed while the second-worst
        read 0.97x. The p95 goes 181x -> 0.78x and is the number that describes
        the bed. The max is kept as a much looser guard against divergence.
        """
        scale, _kind = dynamics_load_scale(self.gs0, self.p0)
        p95_before = free_force_percentile(self.gs0, self.p0, self.F0) / scale
        p95_after = free_force_percentile(self.gs1, self.p1, self.F1) / scale
        self.assertGreater(p95_before, 20.0, "the settle alone should be far out of balance")
        self.assertLess(p95_after, 2.0, f"p95 handoff still {p95_after:.3g}x the load")
        self.assertLess(p95_after, p95_before / 20.0,
                        f"p95 {p95_before:.4g}x -> {p95_after:.4g}x")
        self.assertLess(self._ratio(self.gs1, self.p1, self.F1),
                        self._ratio(self.gs0, self.p0, self.F0),
                        "even the tail must not get worse")

    def test_it_also_removes_the_penetration_the_settle_left(self):
        """The length symptom of the same disease: 1.4 um -> 0.03 um."""
        p0 = penetration_stats(self.c0)['penetration_max']
        p1 = penetration_stats(self.c1)['penetration_max']
        self.assertGreater(p0, 0.5)
        self.assertLess(p1, p0 / 10.0, f"{p0:.4g} -> {p1:.4g} um")

    def test_fire_stops_in_the_dynamics_own_units(self):
        """The correction to the V3.3 plan, made checkable.

        The tolerance must be a multiple of `dynamics_load_scale`, which is a
        force in the units of the law that runs NEXT. Relaxing to balance under
        the settle's own `k_rep ov^1.5` -- which is what the V3.3 plan
        specified, to avoid a circular import -- would leave this unconstrained.
        """
        r = self.gs1.relax_report
        scale, kind = dynamics_load_scale(self.gs1, self.p1)
        self.assertEqual(r['fire_load_kind'], kind)
        self.assertAlmostEqual(r['fire_f_tol'],
                               self.p1.packing_relax_force_tol * scale, places=12)

    def test_the_energy_never_rises_across_the_relaxation(self):
        """FIRE is a minimiser; the Phase 1 energy is what it minimises."""
        r = self.gs1.relax_report
        self.assertLess(r['energy_after'], r['energy_before'])

    def test_the_report_says_why_it_stopped(self):
        r = self.gs1.relax_report
        self.assertIn(r['fire_stop_reason'], ('converged', 'stalled', 'out of steps'))
        for k in ('fire_steps', 'n_rattlers', 'n_unbalanced', 'n_wall_clamped',
                  'ratio_before', 'ratio_after'):
            self.assertIn(k, r)

    def test_the_detector_and_the_fix_agree(self):
        """`handoff_force_balance` and `relax_packing` read the same scale, so
        a bed the fix calls relaxed cannot be one the detector warns about."""
        h = quiet(handoff_force_balance, self.gs1, self.p1, self.F1)
        self.assertAlmostEqual(h['handoff_ratio'],
                               self._ratio(self.gs1, self.p1, self.F1), places=9)


class TestTheWallClamp(unittest.TestCase):
    """Where the position clip sits relative to the wall (`boundary.wall_clamp`).

    Until V3.5 it was a hardcoded +0.5 um: every granule held clear of every
    wall, so a granule resting on the floor was NEVER in wall contact -- the
    JKR wall force saw a positive gap -- and its whole weight was carried by
    the clip. The bed rested on a numerical shelf, every wall contact force
    read zero, and every residual-force measure had an irreducible floor of one
    granule weight. V3.5 moves the clip INSIDE the wall by the same allowance
    the engine already permits between two granules, so the contact carries the
    load; `legacy` restores the old standoff and `force` clips only at the wall
    plane.
    """

    def test_legacy_reproduces_the_v27_standoff(self):
        # `relax='none'` never calls `apply_position_bounds`, so the settle's own
        # wall handling is what is left; `fire` applies the dynamics' clip on entry
        p, gs, _F, _c = bed('fire', boundary_wall_clamp='legacy')
        floor = gs.y[:gs.N] - gs.r_bound[:gs.N]
        self.assertAlmostEqual(float(np.min(floor)), 0.5, places=6)

    def test_legacy_leaves_a_floor_granule_out_of_contact(self):
        """The defect, stated as a test: the wall force on a resting granule is
        identically zero and the clip carries its whole weight."""
        p, gs, F, _c = bed('fire', boundary_wall_clamp='legacy')
        held = constraint_clamped(gs, p, F)
        self.assertGreater(int(np.sum(held)), 0, "a gravity bed must rest on something")
        mag = np.linalg.norm(F[:gs.N], axis=1)
        self.assertGreater(float(mag[held].max()), float(mag[~held].max()),
                           "the clamped granules are the badly balanced ones")

    def test_contact_lets_the_bed_sit_on_the_wall_instead_of_above_it(self):
        """The default. The granule rests on its own wall contact rather than
        0.5 um above it on a numerical shelf."""
        p0, gs0, _F0, _c0 = bed('fire', boundary_wall_clamp='legacy', shape_enabled=False)
        p1, gs1, _F1, _c1 = bed('fire', boundary_wall_clamp='contact', shape_enabled=False)
        gap0 = float(np.min(gs0.y[:gs0.N] - gs0.r_bound[:gs0.N]))
        gap1 = float(np.min(gs1.y[:gs1.N] - gs1.r_bound[:gs1.N]))
        self.assertLess(gap1, gap0, f"legacy {gap0:.4g} vs contact {gap1:.4g} um")

    def test_the_clip_sits_inside_the_wall_by_the_overlap_allowance(self):
        p = Params(**dict(SHAPED_BED, boundary_wall_clamp='contact'))
        gs = quiet(generate_packing, p, seed=7)
        reach = gs.r_bound[:gs.N]
        np.testing.assert_allclose(wall_clamp_margin(gs, p, reach),
                                   -p.max_overlap_frac * reach, rtol=1e-12)

    def test_force_mode_clips_only_at_the_wall_plane(self):
        p = Params(**dict(SHAPED_BED, boundary_wall_clamp='force'))
        gs = quiet(generate_packing, p, seed=7)
        reach = gs.r_bound[:gs.N]
        np.testing.assert_allclose(wall_clamp_margin(gs, p, reach), -reach, rtol=1e-12)

    def test_every_mode_keeps_centres_inside_the_box(self):
        """The one guarantee the clip must not lose: the neighbour and render
        grids need positions inside the box. Surfaces may cross a wall now."""
        for mode in ('legacy', 'contact', 'force'):
            p, gs, _F, _c = bed('none', boundary_wall_clamp=mode)
            with self.subTest(clamp=mode):
                self.assertGreaterEqual(float(np.min(gs.x[:gs.N])), 0.0)
                self.assertLessEqual(float(np.max(gs.x[:gs.N])), p.Lx)
                self.assertGreaterEqual(float(np.min(gs.y[:gs.N])), 0.0)
                self.assertLessEqual(float(np.max(gs.y[:gs.N])), p.Ly)

    def test_excluding_clamped_granules_changes_the_verdict(self):
        """On a sphere bed under the legacy clip, the artefact was over half the
        ratio V3.4 reported."""
        p, gs, F, _c = bed('none', shape_enabled=False, boundary_wall_clamp='legacy')
        scale, _k = dynamics_load_scale(gs, p)
        raw = float(np.linalg.norm(F[:gs.N], axis=1).max()) / scale
        free, held = free_force_residual(gs, p, F)
        self.assertGreater(held, 0)
        self.assertLess(free / scale, 0.75 * raw, f"raw {raw:.3g}x vs free {free / scale:.3g}x")

    def test_the_detector_restores_the_configuration_exactly(self):
        """`constraint_clamped` nudges positions to probe the bounds."""
        p, gs, F, _c = bed('none')
        before = gs.pos.copy()
        top = int(getattr(gs, 'n_top_clamped', 0) or 0)
        constraint_clamped(gs, p, F)
        np.testing.assert_array_equal(gs.pos, before)
        self.assertEqual(int(getattr(gs, 'n_top_clamped', 0) or 0), top)

    def test_periodic_has_no_clamp(self):
        p, gs, F, _c = bed('none', boundary_mode='periodic', boundary_top='wall',
                           gravity_enabled=False, packing_consolidation='none')
        self.assertFalse(np.any(constraint_clamped(gs, p, F)))


class TestTheSettleToleranceIsNotTheProblem(unittest.TestCase):
    """Two negative results, pinned because they are what justify FIRE.

    The V3.3 plan's first item was "stop the settle on the TRUE penetration
    instead of the directional-radius proxy". These tests are why that would
    not have fixed anything, and why the fix had to be a force tolerance
    applied after the settle instead of a better length applied inside it.
    """

    def test_the_proxy_is_barely_wrong(self):
        """0.98 um reported vs 1.40 um true. Measuring it perfectly buys 1.4x,
        against a handoff measured at 346x."""
        p, gs, _F, c = bed('none')
        reported = 0.05 * float(np.mean(gs.r[:gs.N]))     # the V2.7 settle rule
        true_pen = penetration_stats(c)['penetration_max']
        self.assertLess(true_pen / reported, 2.0,
                        f"proxy {reported:.4g} vs true {true_pen:.4g} um")

    def test_the_criterion_is_the_wrong_dimension(self):
        """Force balance needs 0.035 um where the rule asks for 0.98 um: 28x in
        LENGTH, and since F ~ delta^1.5 that is ~150x in FORCE."""
        p, gs, _F, c = bed('none')
        rule = 0.05 * float(np.mean(gs.r[:gs.N]))
        balance = settle_overlap_tolerance(
            gs, Params(**dict(SHAPED_BED, packing_overlap_tol_model='elastic')))
        self.assertIsNotNone(balance)
        self.assertGreater(rule / balance, 10.0,
                           f"rule {rule:.4g} vs force balance {balance:.4g} um")

    def test_the_settle_never_stops_on_its_tolerance_anyway(self):
        """The second negative result, and the more surprising one.

        Tightening the tolerance to the force-balance value changes the 2D bed
        **not at all** -- bit-identical penetration -- because the post-relax
        loop already runs to its step cap and exits on the budget. In 3D, where
        the loop does sometimes stop on the tolerance, it helps (131x -> 108x
        alone, 3.6x -> 1.1x combined with FIRE). V3.5 makes the settle say
        which of the two stopped it.
        """
        _p0, _gs0, _F0, c0 = bed('none', packing_overlap_tol_model='fixed')
        _p1, _gs1, _F1, c1 = bed('none', packing_overlap_tol_model='elastic')
        self.assertAlmostEqual(penetration_stats(c0)['penetration_max'],
                               penetration_stats(c1)['penetration_max'], places=9,
                               msg="if these now differ, the 2D settle has started "
                                   "stopping on its tolerance -- update the changelog")


class TestItIsOffByDefault(unittest.TestCase):

    def test_relax_none_writes_no_report(self):
        p, gs, _F, _c = bed('none')
        self.assertFalse(hasattr(gs, 'relax_report'))

    def test_auto_is_the_default_and_keys_off_the_driving_load(self):
        """`auto` is `fire` exactly when something drives the run. A bed with no
        gravity and no cells has a genuinely loose equilibrium under JKR, and no
        force scale to stop on, so relaxing it just lets it expand."""
        self.assertEqual(Params().packing_relax, 'auto')
        p_load, gs_load, _F, _c = bed('none', contact_shape_dynamics=True)
        self.assertEqual(relax_mode(p_load, gs_load), 'none')      # explicit wins
        p2 = Params(**dict(SHAPED_BED, packing_relax='auto', contact_shape_dynamics=True))
        self.assertEqual(relax_mode(p2, gs_load), 'fire')
        p3 = Params(**dict(SHAPED_BED, packing_relax='auto', contact_shape_dynamics=True,
                           gravity_enabled=False))
        self.assertEqual(relax_mode(p3, gs_load), 'none')

    def test_auto_declines_a_shaped_bed_without_shape_dynamics(self):
        """The bounding-sphere overlap projection would undo the relaxation:
        15-17 % of pairs clipped, and the gradient-flow monitor reports energy
        ascents of 4e3-9e3 times the work against 7-27 with it on."""
        p, gs, _F, _c = bed('none')
        self.assertTrue(p.shape_enabled)
        self.assertFalse(p.contact_shape_dynamics)
        p_auto = Params(**dict(SHAPED_BED, packing_relax='auto'))
        self.assertEqual(quiet(relax_mode, p_auto, gs), 'none')

    def test_the_tolerance_is_below_one_on_purpose(self):
        """An unsupported granule has |F| = exactly its weight, so a tolerance
        of 1.0 or more is satisfied by a bed in free fall."""
        self.assertLess(Params().packing_relax_force_tol, 1.0)


class TestOtherGeometries(unittest.TestCase):

    def test_3d_shaped_gravity_bed(self):
        p0, gs0, F0, _c0 = bed('none', three_d=True, mode='3D', Lx=300.0, Ly=300.0, Lz=400.0,
                               phi_solid_target=0.35, aspect_ratio_c_func_mean=0.8,
                               blockiness_n2_func_mean=3.0, R_func_mean=22.0,
                               R_inert_mean=26.0, packing_settle_steps=100)
        p1, gs1, F1, _c1 = bed('fire', three_d=True, mode='3D', Lx=300.0, Ly=300.0, Lz=400.0,
                               phi_solid_target=0.35, aspect_ratio_c_func_mean=0.8,
                               blockiness_n2_func_mean=3.0, R_func_mean=22.0,
                               R_inert_mean=26.0, packing_settle_steps=100)
        s0, _ = dynamics_load_scale(gs0, p0)
        before = free_force_residual(gs0, p0, F0)[0] / s0
        after = free_force_residual(gs1, p1, F1)[0] / s0
        self.assertGreater(before, 20.0)
        self.assertLess(after, before / 10.0, f"3D: {before:.4g}x -> {after:.4g}x")

    def test_no_load_scale_falls_back_to_a_relative_target(self):
        """Nothing drives the run, so there is no absolute force to aim at."""
        p, gs, F, _c = bed('fire', shape_enabled=False, gravity_enabled=False,
                           boundary_top='wall', packing_consolidation='centre',
                           mode='2D', Lx=500.0, Ly=500.0, phi_solid_target=0.55,
                           packing_settle_steps=100)
        self.assertIsNone(dynamics_load_scale(gs, p)[0])
        r = gs.relax_report
        self.assertEqual(r['fire_load_kind'], 'relative')
        self.assertTrue(r['fire_converged'])
        self.assertLess(r['f_max_after'], r['f_max_before'])


if __name__ == '__main__':
    unittest.main()
