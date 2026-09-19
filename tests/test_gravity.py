"""
Gravity / buoyancy (V3.1).
==========================

Every mobile granule carries a buoyant weight W = (rho_gran - rho_medium) g V
in nN, added to the force array along -z (3D) or -y (2D) by
``gels.engine.apply_gravity`` on both the compiled and the reference path.
Immobile boundary granules and runs with ``gravity_enabled = False`` get
nothing, so the legacy fixtures are untouched.

Run:  python -m unittest tests.test_gravity -v
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
    GranuleSystem, Params, apply_gravity, compute_forces, compute_forces_3d, granule_weights,
)

PMMA = 1180.0          # kg/m^3
WATER = 1000.0


def _one_granule(mode, r=20.0, density=PMMA, **kw):
    """A single granule far from every wall, with no cells and no noise."""
    p = Params(mode=mode, Lx=400.0, Ly=400.0, Lz=400.0, T_active=0.0, save_data=False,
               gravity_enabled=True, granule_density=density, medium_density=WATER, **kw)
    species = [dict(name='g', f=1.0, volume_fraction=1.0, color='#CC2222',
                    radius_mean=r, radius_std=0.0, radius_min=r,
                    density_kg_m3=density)]
    gs = GranuleSystem([200.0], [200.0], [r], [0], [0],
                       z=[200.0] if mode == '3D' else None,
                       mode=mode, species_id=[0], species=species, p=p)
    return gs, p


class TestWeight(unittest.TestCase):

    def test_matches_archimedes(self):
        for r in (20.0, 40.0, 75.0):
            gs, p = _one_granule('3D', r=r)
            expected = (PMMA - WATER) * 9.81 * (4.0 / 3.0) * np.pi * r ** 3 * 1e-9
            self.assertAlmostEqual(float(granule_weights(gs, p)[0]), expected, delta=1e-12 * max(expected, 1.0))
        # the numbers quoted in the plan / CHANGELOG
        gs, p = _one_granule('3D', r=20.0)
        self.assertAlmostEqual(float(granule_weights(gs, p)[0]), 0.0592, places=4)

    def test_scale_and_disable(self):
        gs, p = _one_granule('3D')
        w1 = float(granule_weights(gs, p)[0])
        p2 = copy.deepcopy(p)
        p2.gravity_scale = 10.0
        self.assertAlmostEqual(float(granule_weights(gs, p2)[0]), 10.0 * w1, places=12)
        p3 = copy.deepcopy(p)
        p3.gravity_enabled = False
        self.assertEqual(float(granule_weights(gs, p3)[0]), 0.0)

    def test_neutrally_buoyant_and_fixed_granules_weigh_nothing(self):
        gs, p = _one_granule('3D', density=WATER)
        self.assertEqual(float(granule_weights(gs, p)[0]), 0.0)
        gs, p = _one_granule('3D')
        gs.fixed[:] = True
        self.assertEqual(float(granule_weights(gs, p)[0]), 0.0)

    def test_apply_gravity_direction(self):
        gs, p = _one_granule('3D')
        F = np.zeros((1, 3))
        apply_gravity(gs, p, F)
        w = float(granule_weights(gs, p)[0])
        np.testing.assert_allclose(F[0], [0.0, 0.0, -w], atol=0, rtol=0)
        gs, p = _one_granule('2D')
        F = np.zeros((1, 2))
        apply_gravity(gs, p, F)
        np.testing.assert_allclose(F[0], [0.0, -float(granule_weights(gs, p)[0])], atol=0, rtol=0)


class TestInForceAssembly(unittest.TestCase):
    """The lone granule has no neighbours and no cells, so F is the weight alone."""

    def _force(self, mode, use_numba):
        gs, p = _one_granule(mode)
        p.use_numba = use_numba
        gs.n_top_clamped = 0
        rng = np.random.default_rng(3)
        fn = compute_forces_3d if mode == '3D' else compute_forces
        F, _t, _c = fn(gs, p, rng)
        return F[0], float(granule_weights(gs, p)[0])

    def test_3d_both_paths(self):
        for use_numba in (False, True):
            F, w = self._force('3D', use_numba)
            np.testing.assert_allclose(F, [0.0, 0.0, -w], atol=1e-12)

    def test_2d_both_paths(self):
        for use_numba in (False, True):
            F, w = self._force('2D', use_numba)
            np.testing.assert_allclose(F[:2], [0.0, -w], atol=1e-12)

    def test_disabled_leaves_forces_untouched(self):
        for mode in ('2D', '3D'):
            gs, p = _one_granule(mode)
            p.gravity_enabled = False
            rng = np.random.default_rng(3)
            fn = compute_forces_3d if mode == '3D' else compute_forces
            F, _t, _c = fn(gs, p, rng)
            np.testing.assert_allclose(F[0], np.zeros(F.shape[1]), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
