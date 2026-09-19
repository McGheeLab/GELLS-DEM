"""
Wall functionalization and the immobile boundary lining (V3.1).
===============================================================

The lab sees two regimes: against an INERT boundary the bed detaches and
compacts, while against a FUNCTIONALIZED boundary it stays pinned and the
functional phase coarsens locally instead. Two settings express that:

  * ``boundary.functionalization`` (f_wall) mixes the smooth wall's JKR
    adhesion with ``mix_pair(f_i, f_wall, ...)`` — f_wall = 0 is the V2.7
    bare wall exactly;
  * ``boundary.layer.enabled`` lines the floor and side wall with immobile
    granules of that coverage, so cells bridge to the container through the
    ordinary bridging machinery.

Immobile granules must never move: not by integration, not by the overlap
projection, not by the position clamp, and not during the packing settle.

Run:  python -m unittest tests.test_boundary_layer -v
"""

import os
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels import materials as M  # noqa: E402
from gels.engine import (  # noqa: E402
    BOUNDARY_SPECIES_NAME, Params, boundary_geometry, boundary_layer_radius, boundary_layer_sites,
    generate_packing, generate_packing_3d, resolve_species, step, update_cell_state,
)


class TestWallFunctionalization(unittest.TestCase):

    def test_f_wall_zero_is_the_bare_wall(self):
        f = [1.0, 0.5, 0.0]
        E = [10.0, 10.0, 10.0]
        nu = [0.45, 0.45, 0.45]
        args = (2e-3, 1e-3, 5e-4, 2000.0, 500.0, 50.0)
        base = M.build_pair_tables(f, E, nu, *args)
        same = M.build_pair_tables(f, E, nu, *args, f_wall=0.0)
        np.testing.assert_array_equal(same['wall_W'], base['wall_W'])
        for k, fi in enumerate(f):
            self.assertAlmostEqual(base['wall_W'][k], M.mix_wall(fi, 1e-3, 5e-4), places=15)

    def test_f_wall_one_uses_the_collagen_pair_rule(self):
        f = [1.0, 0.5, 0.0]
        tables = M.build_pair_tables(f, [10.0] * 3, [0.45] * 3,
                                     2e-3, 1e-3, 5e-4, 2000.0, 500.0, 50.0, f_wall=1.0)
        for k, fi in enumerate(f):
            self.assertAlmostEqual(tables['wall_W'][k], M.mix_pair(fi, 1.0, 2e-3, 1e-3, 5e-4), places=15)
        # a coated wall is stickier to a coated granule than a bare wall is
        bare = M.build_pair_tables(f, [10.0] * 3, [0.45] * 3, 2e-3, 1e-3, 5e-4, 2000.0, 500.0, 50.0)
        self.assertGreater(tables['wall_W'][0], bare['wall_W'][0])

    def test_stiffness_cap_touches_contacts_only(self):
        f, nu = [1.0], [0.35]
        E = [3.0e6]                      # PMMA
        args = (2e-3, 1e-3, 5e-4, 2000.0, 500.0, 50.0)
        uncapped = M.build_pair_tables(f, E, nu, *args)
        capped = M.build_pair_tables(f, E, nu, *args, E_cap_kPa=100.0)
        self.assertLess(capped['pair_Estar'][0, 0], uncapped['pair_Estar'][0, 0] / 1000.0)
        self.assertAlmostEqual(capped['pair_Estar'][0, 0], 100.0 * 1e3 / (2.0 * (1.0 - 0.35 ** 2)), places=6)
        self.assertAlmostEqual(capped['wall_Estar'][0], 100.0 * 1e3 / (1.0 - 0.35 ** 2), places=6)
        # a cap above the modulus changes nothing
        high = M.build_pair_tables(f, E, nu, *args, E_cap_kPa=1.0e9)
        np.testing.assert_array_equal(high['pair_Estar'], uncapped['pair_Estar'])
        # ... and the granule modulus the cells feel is untouched (Params, not the tables)
        p = Params(contact_E_cap=100.0,
                   species=[dict(name='pmma', f=1.0, volume_fraction=1.0, color='#CCC',
                                 radius_mean=20.0, radius_std=0.0, radius_min=20.0, E_kPa=3.0e6)])
        gs = generate_packing(Params(**{**p.__dict__, 'Lx': 200.0, 'Ly': 200.0,
                                        'phi_solid_target': 0.3, 'packing_settle_steps': 10,
                                        'save_data': False}), seed=1)
        self.assertAlmostEqual(float(gs.E_gran[0]), 3.0e6, places=3)
        self.assertLess(float(gs.pair_Estar[0, 0]), 1e6)


def _well(**kw):
    base = dict(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, phi_solid_target=0.15, func_ratio=1.0,
                R_func_mean=20.0, R_func_std=2.0, cell_surface_coverage=0.5,
                packing_settle_steps=60, save_data=False, T_active=0.0,
                boundary_shape='cylinder', boundary_top='free', gravity_enabled=True,
                granule_density=1180.0, packing_consolidation='gravity',
                boundary_layer_enabled=True, boundary_functionalization=1.0)
    base.update(kw)
    return Params(**base)


class TestBoundaryLayer(unittest.TestCase):

    def test_species_appended_once_and_marked_fixed(self):
        p = _well()
        sp1 = resolve_species(p)
        sp2 = resolve_species(p)
        names = [s['name'] for s in sp2]
        self.assertEqual(names.count(BOUNDARY_SPECIES_NAME), 1, names)
        self.assertEqual(len(sp1), len(sp2))
        b = sp2[names.index(BOUNDARY_SPECIES_NAME)]
        self.assertTrue(b['fixed'])
        self.assertEqual(b['volume_fraction'], 0.0)     # the lining is container, not sample
        self.assertEqual(b['f'], 1.0)

    def test_disabled_by_default(self):
        p = Params(save_data=False)
        self.assertNotIn(BOUNDARY_SPECIES_NAME, [s['name'] for s in resolve_species(p)])

    def test_sites_lie_on_the_container_surface(self):
        p = _well()
        r_w = boundary_layer_radius(p, resolve_species(p))
        geom = boundary_geometry(p, '3D')
        sites = boundary_layer_sites(p, r_w, '3D')
        self.assertGreater(len(sites), 20)
        inset = r_w * (1.0 - 2.0 * p.boundary_layer_embed_frac)
        floor = [s for s in sites if abs(s[2] - inset) < 1e-9]
        wall = [s for s in sites if abs(s[2] - inset) >= 1e-9]
        self.assertGreater(len(floor), 5)
        self.assertGreater(len(wall), 5)
        for (x, y, _z) in floor:
            self.assertLessEqual(np.hypot(x - geom.cx, y - geom.cy), geom.R_cyl - inset + 1e-9)
        for (x, y, z) in wall:
            self.assertAlmostEqual(float(np.hypot(x - geom.cx, y - geom.cy)), geom.R_cyl - inset, places=6)
            self.assertLessEqual(z, p.Lz)

    def test_packing_marks_them_fixed_and_seeds_cells(self):
        p = _well()
        gs = generate_packing_3d(p, seed=2)
        self.assertGreater(int(gs.fixed.sum()), 20)
        self.assertGreater(int((~gs.fixed).sum()), 20)
        np.testing.assert_array_equal(gs.f[gs.fixed], 1.0)
        self.assertGreater(int(gs.n_cells[gs.fixed].sum()), 0)
        # seeding can be switched off for the lining
        p2 = _well(boundary_layer_seed_cells=False)
        gs2 = generate_packing_3d(p2, seed=2)
        self.assertEqual(int(gs2.n_cells[gs2.fixed].sum()), 0)
        self.assertGreater(int(gs2.n_cells[~gs2.fixed].sum()), 0)

    def test_lining_never_moves_during_a_step(self):
        p = _well(dt=0.5, bridge_attempt_rate=1e3, min_fa_for_bridge=0.0,
                  fa_maturation_rate=100.0, t_spread_duration=0.1, max_overlap_frac=0.02)
        for use_numba in (False, True):
            p.use_numba = use_numba
            gs = generate_packing_3d(p, seed=3)
            before = gs.pos[gs.fixed].copy()
            mobile_before = gs.pos[~gs.fixed].copy()
            rng = np.random.default_rng(5)
            update_cell_state(gs, p, 5.0, rng)
            for _ in range(3):
                step(gs, p, rng, 5.0)
            np.testing.assert_array_equal(gs.pos[gs.fixed], before)
            self.assertGreater(float(np.abs(gs.pos[~gs.fixed] - mobile_before).max()), 0.0,
                               "mobile granules did not move at all")

    def test_2d_dish_lining(self):
        p = Params(mode='2D', Lx=400.0, Ly=500.0, phi_solid_target=0.2, func_ratio=1.0,
                   R_func_mean=15.0, R_func_std=1.0, cell_surface_coverage=0.5,
                   packing_settle_steps=60, save_data=False, boundary_top='free',
                   gravity_enabled=True, packing_consolidation='gravity', bed_phi_assumed=0.8,
                   boundary_layer_enabled=True, boundary_functionalization=1.0)
        gs = generate_packing(p, seed=4)
        self.assertGreater(int(gs.fixed.sum()), 10)
        r_w = boundary_layer_radius(p, resolve_species(p))
        fixed_pos = gs.pos[gs.fixed]
        on_floor = np.abs(fixed_pos[:, 1] - 0.0) < r_w
        on_side = (np.abs(fixed_pos[:, 0] - 0.0) < r_w) | (np.abs(fixed_pos[:, 0] - p.Lx) < r_w)
        self.assertTrue(bool(np.all(on_floor | on_side)))


if __name__ == '__main__':
    unittest.main()
