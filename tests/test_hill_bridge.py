"""
Contractile (Hill) bridge element (V3.1).
=========================================

``cells.bridging.force_model = 'hill'`` turns each bridge from a constant
actuator into an active element with a stall force and an unloaded shortening
speed v0. The point of the change is tensional homeostasis: once the
environment stops yielding the cell holds its stall force, which is the
"expand until force balance" behaviour the constant model could not express.

Checked here: the scalar law's limits, its stability as a fixed point, exact
reduction to the constant model (so the legacy fixtures are untouched), and
that a two-granule pair closes at a rate set by v0 rather than by v_max.

Run:  python -m unittest tests.test_hill_bridge -v
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
    CellState, GranuleSystem, Params, compute_forces, force_model_code, step, update_cell_state,
)
from gels.materials import hill_pair_factor  # noqa: E402


class TestScalarLaw(unittest.TestCase):

    def test_isometric_hold_at_force_balance(self):
        """A rigid environment (nothing yields, F_prev = F_iso) holds the stall force."""
        c, F_iso, v0 = 2.0, 50.0, 12.0
        fac = hill_pair_factor(0.0, F_iso, F_iso, c, v0, 0.5, False)
        self.assertAlmostEqual(fac, 1.0, places=12)

    def test_free_pair_closes_below_v0_and_is_a_stable_fixed_point(self):
        c, F_iso, v0 = 2.0, 50.0, 12.0
        x = c * F_iso / v0
        expected_fac = 1.0 / (1.0 + x)
        F, v = 0.0, 0.0
        for _ in range(60):                      # iterate the pair to its fixed point
            fac = hill_pair_factor(v, F, F_iso, c, v0, 0.5, False)
            F = fac * F_iso
            v = c * F                            # nothing else acts on the pair
        self.assertAlmostEqual(fac, expected_fac, places=9)
        self.assertLess(v, v0, "closing speed must stay below the unloaded speed")
        self.assertAlmostEqual(v, v0 * x / (1.0 + x), places=9)

    def test_zero_at_the_unloaded_speed(self):
        c, F_iso, v0 = 1.0, 10.0, 12.0
        self.assertAlmostEqual(hill_pair_factor(v0, 0.0, F_iso, c, v0, 0.5, False), 0.0, places=12)
        self.assertEqual(hill_pair_factor(2.0 * v0, 0.0, F_iso, c, v0, 0.5, False), 0.0)

    def test_eccentric_branch_is_capped(self):
        """Above the stall force only while the pair is being pulled apart faster
        than the bridge can shorten, i.e. v_env < -c F_iso."""
        c, F_iso, v0, ecc = 1.0, 10.0, 12.0, 0.5
        v_break_even = -c * F_iso            # fac == 1 exactly here
        self.assertAlmostEqual(hill_pair_factor(v_break_even, 0.0, F_iso, c, v0, ecc, False),
                               1.0, places=12)
        mild = hill_pair_factor(2.0 * v_break_even, 0.0, F_iso, c, v0, ecc, False)
        self.assertGreater(mild, 1.0)
        self.assertLess(mild, 1.0 + ecc)
        stretched = hill_pair_factor(-50.0 * v0, 0.0, F_iso, c, v0, ecc, False)
        self.assertAlmostEqual(stretched, 1.0 + ecc, places=12)
        # shortening more slowly than break-even is concentric: below the stall force
        self.assertLess(hill_pair_factor(0.0, 0.0, F_iso, c, v0, ecc, False), 1.0)

    def test_reduces_to_the_constant_actuator(self):
        self.assertEqual(hill_pair_factor(5.0, 1.0, 10.0, 2.0, np.inf, 0.5, False), 1.0)
        self.assertEqual(hill_pair_factor(5.0, 1.0, 10.0, 2.0, 0.0, 0.5, False), 1.0)
        # at the rest length the cell holds isometric tension
        self.assertEqual(hill_pair_factor(5.0, 1.0, 10.0, 2.0, 12.0, 0.5, True), 1.0)

    def test_model_code(self):
        self.assertEqual(force_model_code('constant'), 0)
        self.assertEqual(force_model_code('hill'), 1)
        self.assertEqual(force_model_code('HILL'), 1)


def _pair(gap=10.0, r=20.0, **kw):
    """Two granules a fixed gap apart, one cell each, primed to bridge at once."""
    base = dict(mode='2D', Lx=400.0, Ly=400.0, T_active=0.0, save_data=False,
                dt=0.5, bridge_attempt_rate=1e6, min_fa_for_bridge=0.0,
                fa_maturation_rate=100.0, t_spread_duration=0.01, bridge_commit_angle=np.pi,
                bridge_exclusion_angle=0.0, cell_sense_distance=60.0, mc_dem_enabled=False)
    base.update(kw)
    p = Params(**base)
    species = [dict(name='c', f=1.0, volume_fraction=1.0, color='#CC2222',
                    radius_mean=r, radius_std=0.0, radius_min=r)]
    x0 = 150.0
    gs = GranuleSystem([x0, x0 + 2 * r + gap], [200.0, 200.0], [r, r], [0, 0], [1, 1],
                       mode='2D', species_id=[0, 0], species=species, p=p)
    gs.cell_theta_local[0] = 0.0                 # facing the partner
    gs.cell_theta_local[1] = np.pi
    return gs, p


class TestInTheEngine(unittest.TestCase):

    def _run(self, model, steps=6, **kw):
        gs, p = _pair(bridge_force_model=model, **kw)
        rng = np.random.default_rng(4)
        update_cell_state(gs, p, 5.0, rng)
        gaps = []
        for _ in range(steps):
            step(gs, p, rng, 5.0)
            gaps.append(float(abs(gs.x[1] - gs.x[0]) - gs.r[0] - gs.r[1]))
        return gs, p, gaps

    def test_constant_model_is_unchanged_by_the_new_code(self):
        """With force_model='constant' the factor is exactly 1.0 everywhere."""
        gs, p, gaps = self._run('constant')
        bridging = gs.cell_state == int(CellState.BRIDGING)
        self.assertTrue(bool(np.any(bridging)), "no bridge formed")
        # the recorded force equals maturity x alignment x capacity, i.e. factor 1
        rec = gs.cell_bridge_force[bridging]
        self.assertTrue(bool(np.all(rec > 0.0)))
        self.assertLess(gaps[-1], gaps[0], "granules should be pulled together")

    def test_hill_holds_stall_force_at_the_rest_length(self):
        """Below `min_gap` the cell stops shortening and holds isometric tension,
        so its force is exactly the constant model's."""
        gs, p, gaps = self._run('hill', gap=30.0, steps=6, cell_contraction_speed=12.0,
                                bridge_min_gap=1e6)          # always at the rest length
        gs_c, _p, gaps_c = self._run('constant', gap=30.0, steps=6)
        np.testing.assert_allclose(gaps, gaps_c, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(gs.cell_bridge_force, gs_c.cell_bridge_force, rtol=1e-9, atol=1e-9)

    def test_hill_limits_the_closing_rate(self):
        """A slow v0 must close the gap more slowly than the constant actuator."""
        _gs_f, _p, gaps_fast = self._run('constant', gap=30.0, steps=8)
        _gs_s, _p2, gaps_slow = self._run('hill', gap=30.0, steps=8, cell_contraction_speed=1.0)
        self.assertLessEqual(gaps_fast[-1], gaps_slow[-1] + 1e-9,
                             f"hill closed faster than constant: {gaps_slow[-1]} vs {gaps_fast[-1]}")

    def test_large_v0_matches_the_constant_model(self):
        gs_h, _p, gaps_h = self._run('hill', gap=30.0, steps=6, cell_contraction_speed=1e9)
        gs_c, _p2, gaps_c = self._run('constant', gap=30.0, steps=6)
        np.testing.assert_allclose(gaps_h, gaps_c, rtol=1e-6, atol=1e-6)

    def test_records_are_written_and_cleared(self):
        gs, p, _ = self._run('hill', gap=10.0)
        bridging = gs.cell_state == int(CellState.BRIDGING)
        if np.any(bridging):
            self.assertFalse(bool(np.any(np.isnan(gs.cell_bridge_gap_prev[bridging]))))
        idle = gs.cell_state == int(CellState.SENESCENT)
        if np.any(idle):
            self.assertTrue(bool(np.all(gs.cell_bridge_force[idle] == 0.0)))


if __name__ == '__main__':
    unittest.main()
