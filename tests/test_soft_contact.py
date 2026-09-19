"""
Soft-granule contact mechanics (V3.2).
======================================

The lab's granules are hydrogel at 10-50 kPa, not the 3 GPa PMMA V3.1 was built
around, so contacts genuinely flatten under cell traction. This checks that the
overlap rail is sized from the contact law rather than by hand, that the
interpenetration a bed reports is not double-counted, and that the numerical
rails (overlap projection, velocity cap) are observable rather than silent.

Run:  python -m unittest tests.test_soft_contact -v
"""

import copy
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
    CellState, GranuleSystem, Params, bridge_weight_map,
    contact_stiffness_per_granule, elastic_overlap_frac,
    jkr_force_from_overlap, overlap_lens_area, overlap_lens_volume,
    overlap_solid_volume, pair_bridge_weight, projection_metrics,
)


def _delta_eq(E_kPa, nu, r, F):
    """Overlap at which a like-pair Hertz contact carries F (the plan's formula)."""
    E_s = E_kPa / (2.0 * (1.0 - nu * nu))          # nN/um^2
    return (3.0 * F / (4.0 * E_s * np.sqrt(0.5 * r))) ** (2.0 / 3.0)


def _p(E, r, F, **kw):
    sp = [dict(name='c', f=1.0, volume_fraction=1.0, color='#CC2222',
               radius_mean=r, radius_std=0.0, radius_min=r)]
    base = dict(E_modulus=E, poisson_ratio=0.45, expected_bridge_force=F,
                species=copy.deepcopy(sp), save_data=False)
    base.update(kw)
    return Params(**base)


class TestElasticRail(unittest.TestCase):

    def test_rail_matches_the_equilibrium_overlap(self):
        """The derived fraction is safety * delta_eq / r, from the contact law."""
        for E, r, F in ((10.0, 20.0, 180.0), (10.0, 40.0, 180.0), (50.0, 20.0, 100.0)):
            p = _p(E, r, F, contact_overlap_model='elastic', contact_overlap_safety=1.5)
            want = min(0.30, max(0.05, 1.5 * _delta_eq(E, 0.45, r, F) / r))
            self.assertAlmostEqual(p.max_overlap_frac, want, places=9, msg=f"E={E} r={r} F={F}")

    def test_soft_gets_a_larger_rail_than_rigid(self):
        soft = _p(10.0, 20.0, 180.0, contact_overlap_model='elastic').max_overlap_frac
        stiff = _p(50.0, 20.0, 180.0, contact_overlap_model='elastic').max_overlap_frac
        rigid = _p(3.0e6, 20.0, 180.0, contact_overlap_model='elastic').max_overlap_frac
        self.assertGreater(soft, stiff)
        self.assertGreater(stiff, rigid)
        self.assertAlmostEqual(rigid, 0.05, places=9)      # clamped at the floor

    def test_fixed_model_is_untouched(self):
        self.assertEqual(Params().max_overlap_frac, 0.15)
        self.assertEqual(_p(10.0, 20.0, 180.0).max_overlap_frac, 0.15)

    def test_the_rail_is_a_compaction_ceiling(self):
        """(2/(2-f))^3 is why V3.1's 0.03 left only 4.6 % of headroom."""
        self.assertAlmostEqual((2.0 / (2.0 - 0.03)) ** 3, 1.0464, places=3)
        self.assertAlmostEqual((2.0 / (2.0 - 0.15)) ** 3, 1.2633, places=3)


class TestElasticRailThroughTheConfig(unittest.TestCase):
    """Regression: setup_to_params builds a default Params then setattrs every
    field, so a value derived in __post_init__ is silently overwritten. The
    pipeline uses that path, so the rail was inert exactly where it matters."""

    def test_config_path_derives_the_rail(self):
        from gels.config import Setup, setup_to_params
        from gels.presets import apply_presets
        s = Setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        self.assertEqual(p.contact_overlap_model, 'elastic')
        self.assertNotAlmostEqual(p.max_overlap_frac, 0.20, places=6,
                                  msg="max_overlap_frac is still the preset's literal value")
        self.assertAlmostEqual(p.max_overlap_frac, elastic_overlap_frac(p), places=12)


class TestOverlapVolume(unittest.TestCase):

    def test_matches_the_scalar_lens_formulas(self):
        r1, r2, d = 40.0, 30.0, 55.0
        gs3 = GranuleSystem([0.0, d], [0.0, 0.0], [r1, r2], [0, 0], [0, 0],
                            z=[0.0, 0.0], mode='3D')
        self.assertAlmostEqual(overlap_solid_volume(gs3), overlap_lens_volume(r1, r2, d), places=6)
        gs2 = GranuleSystem([0.0, d], [0.0, 0.0], [r1, r2], [0, 0], [0, 0], mode='2D')
        self.assertAlmostEqual(overlap_solid_volume(gs2), overlap_lens_area(r1, r2, d), places=6)

    def test_zero_when_separated_and_full_when_contained(self):
        far = GranuleSystem([0.0, 500.0], [0.0, 0.0], [20.0, 20.0], [0, 0], [0, 0], mode='2D')
        self.assertEqual(overlap_solid_volume(far), 0.0)
        inside = GranuleSystem([0.0, 1.0], [0.0, 0.0], [40.0, 5.0], [0, 0], [0, 0], mode='2D')
        self.assertAlmostEqual(overlap_solid_volume(inside), np.pi * 25.0, places=6)


class TestContactStiffness(unittest.TestCase):

    def test_stiffness_is_two_E_star_a(self):
        """dF/d(delta) = 2 E_s a, accumulated on both partners."""
        class _C(list):
            def column(self, name):
                return np.array([d[name] for d in self])
        E_star_Pa, a = 6.27e3, 2.5
        gs = GranuleSystem([0.0, 30.0], [0.0, 0.0], [20.0, 20.0], [0, 0], [0, 0], mode='2D')
        k = contact_stiffness_per_granule(
            gs, _C([dict(i=0, j=1, a_contact=a, E_star=E_star_Pa)]))
        want = 2.0 * (E_star_Pa * 1e-3) * a
        np.testing.assert_allclose(k, [want, want], rtol=1e-12)

    def test_no_contacts_gives_zero(self):
        gs = GranuleSystem([0.0, 500.0], [0.0, 0.0], [20.0, 20.0], [0, 0], [0, 0], mode='2D')
        np.testing.assert_allclose(contact_stiffness_per_granule(gs, []), [0.0, 0.0])

    def test_stiffness_crosses_the_explicit_stability_bound(self):
        """Every load-bearing soft contact is outside k < gamma/dt -- which is why
        the semi-implicit step exists."""
        p = Params(drag_scale=0.05, dt=0.5)
        gamma = p.drag_scale * 20.0                       # r = 20 um
        bound = gamma / p.dt                              # 2 nN/um
        d = _delta_eq(10.0, 0.45, 20.0, 180.0)
        E_s = 10.0 / (2.0 * (1.0 - 0.45 ** 2))
        k = 2.0 * E_s * np.sqrt(0.5 * 20.0 * d)           # 2 E_s a, a = sqrt(R* delta)
        self.assertGreater(k, 10.0 * bound)


class TestSoftDeformsMoreThanRigid(unittest.TestCase):
    """The claim Part B rests on, as a direct force balance rather than a run."""

    def test_equilibrium_overlap_scales_with_compliance(self):
        r, F = 20.0, 180.0
        for E, nu in ((10.0, 0.45), (50.0, 0.45), (3.0e6, 0.37)):
            delta = _delta_eq(E, nu, r, F)
            E_star_Pa = (E * 1e3) / (2.0 * (1.0 - nu * nu))
            got, _a = jkr_force_from_overlap(delta, 0.5 * r, E_star_Pa, 0.0)
            self.assertAlmostEqual(got, F, delta=1e-6 * F, msg=f"E={E}")
        soft = _delta_eq(10.0, 0.45, r, F)
        stiff = _delta_eq(50.0, 0.45, r, F)
        rigid = _delta_eq(3.0e6, 0.37, r, F)
        self.assertGreater(soft / stiff, 2.5)          # (50/10)^(2/3) = 2.92
        self.assertGreater(soft / rigid, 1000.0)       # 3.59 um vs 0.001 um


class TestCellContactAdhesion(unittest.TestCase):
    """Cells bridging ACROSS a contact stop the granules sliding at it.

    The normal direction needs nothing new -- the bridge already pulls the pair
    together with up to the stall force, and the contact it deepens grows its own
    shear cap through the contact radius. What was missing is the collar's
    resistance to shear, which is what locks a compacted structure in place.
    """

    def _two(self, target=(1, 0), age=10.0):
        p = Params(mode='2D', cell_contact_adhesion=20.0, bridge_formation_time=2.0,
                   save_data=False)
        gs = GranuleSystem([0.0, 30.0], [0.0, 0.0], [20.0, 20.0], [0, 0], [1, 1], mode='2D')
        gs.cell_granule_id[:] = [0, 1]
        gs.cell_state[:] = int(CellState.BRIDGING)
        gs.cell_bridge_target[:] = list(target)
        gs.cell_bridge_age[:] = age
        return gs, p

    def test_weight_counts_mature_bridges_in_either_sense(self):
        gs, p = self._two()
        w = pair_bridge_weight(gs, p, np.array([0]), np.array([1]))
        self.assertAlmostEqual(float(w[0]), 2.0)          # one cell each way, both mature
        self.assertEqual(bridge_weight_map(gs, p), {(0, 1): 2.0})

    def test_weight_ramps_with_bridge_maturity(self):
        gs, p = self._two(age=1.0)                        # half of formation_time
        w = pair_bridge_weight(gs, p, np.array([0]), np.array([1]))
        self.assertAlmostEqual(float(w[0]), 1.0)          # 2 cells x 0.5 maturity

    def test_no_weight_without_a_committed_bridge(self):
        gs, p = self._two(target=(-1, -1))
        w = pair_bridge_weight(gs, p, np.array([0]), np.array([1]))
        self.assertEqual(float(w[0]), 0.0)
        self.assertEqual(bridge_weight_map(gs, p), {})

    def test_disabled_costs_nothing(self):
        gs, p = self._two()
        p.cell_contact_adhesion = 0.0
        self.assertEqual(bridge_weight_map(gs, p), {})    # the map is the expensive half

    def test_weight_is_zero_for_unbridged_pairs(self):
        gs, p = self._two()
        w = pair_bridge_weight(gs, p, np.array([0, 0]), np.array([1, 1]))
        np.testing.assert_allclose(w, [2.0, 2.0])
        far = pair_bridge_weight(gs, p, np.array([0]), np.array([0]))
        self.assertEqual(float(far[0]), 0.0)


class TestRailDiagnostics(unittest.TestCase):

    def test_projection_metrics_report_both_rails(self):
        gs = GranuleSystem([0.0, 30.0], [0.0, 0.0], [20.0, 20.0], [0, 0], [0, 0], mode='2D')
        m = projection_metrics(gs)
        self.assertEqual(m['n_overlap_clipped'], 0)
        self.assertEqual(m['overlap_clip_fraction'], 0.0)
        self.assertEqual(m['frac_velocity_clipped'], 0.0)
        gs.n_overlap_clipped, gs.n_overlap_pairs = 3, 12
        gs.frac_velocity_clipped = 0.25
        m = projection_metrics(gs)
        self.assertAlmostEqual(m['overlap_clip_fraction'], 0.25)
        self.assertAlmostEqual(m['frac_velocity_clipped'], 0.25)


if __name__ == '__main__':
    unittest.main()
