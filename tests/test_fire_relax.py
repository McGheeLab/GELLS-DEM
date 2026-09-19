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
    penetration_stats,
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
        """346x -> 1.0x of a granule's weight, measured. 10x is the guard."""
        before = self._ratio(self.gs0, self.p0, self.F0)
        after = self._ratio(self.gs1, self.p1, self.F1)
        self.assertLess(after, 10.0, f"handoff still {after:.3g}x the load")
        self.assertLess(after, before / 20.0, f"{before:.4g}x -> {after:.4g}x")

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


class TestTheWallClampFloor(unittest.TestCase):
    """`apply_position_bounds` holds every granule 0.5 um clear of every wall.

    A granule resting on the floor is therefore NEVER in wall contact -- the
    JKR wall force sees a positive gap -- and its net force stays exactly its
    own weight, carried by the clamp. Any residual measure that ignores this
    has an irreducible floor of one granule weight per gravity bed.

    Pre-existing, not introduced by V3.5. These tests pin it so that when the
    clamp and the wall law are reconciled, the change is visible.
    """

    def test_a_floor_granule_sits_half_a_micron_clear(self):
        p, gs, F, _c = bed('fire')
        # the clamp is written in terms of r_bound, so that is what stands off
        floor = gs.y[:gs.N] - gs.r_bound[:gs.N]
        self.assertAlmostEqual(float(np.min(floor)), 0.5, places=6,
                               msg="the clamp standoff moved; see constraint_clamped")

    def test_the_clamp_is_what_holds_them(self):
        p, gs, F, _c = bed('fire')
        held = constraint_clamped(gs, p, F)
        self.assertGreater(int(np.sum(held)), 0, "a gravity bed must rest on something")
        mag = np.linalg.norm(F[:gs.N], axis=1)
        self.assertGreater(float(mag[held].max()), float(mag[~held].max()),
                           "the clamped granules are the badly balanced ones")

    def test_excluding_them_changes_the_verdict(self):
        """On a sphere bed the clamp artefact was over half the reported ratio."""
        p, gs, F, _c = bed('none', shape_enabled=False)
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


class TestItIsOffByDefault(unittest.TestCase):

    def test_default_is_none_and_writes_no_report(self):
        p, gs, _F, _c = bed('none')
        self.assertEqual(Params().packing_relax, 'none')
        self.assertFalse(hasattr(gs, 'relax_report'))

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
