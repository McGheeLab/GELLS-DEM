"""
Packing on the compiled path is bit-identical to the reference path (V3.0 Phase 6f).

The RSA overlap test moves from an all-granule Python scan to a neighbour
grid (same inequality, same decision, same random draws) and the settle to a
compiled substep that reproduces np.add.at's accumulation order, so with the
tree pair order (perf_neighbor_backend='ckdtree') the generated GranuleSystem
— positions, radii, shapes, orientations, species, cell counts and initial
cell placement — must match exactly with use_numba=False and use_numba=True.
With the default cell list the RSA stage is still identical (same radii,
species, counts and pre-settle positions) and the settled packing is valid,
deterministic and thread-independent.

Run:  python -m unittest tests.test_packing_vs_reference -v
"""

import copy
import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import Params, generate_packing, generate_packing_3d  # noqa: E402
from gels.kernels import HAS_NUMBA  # noqa: E402

ARRAYS = ('x', 'y', 'z', 'r', 'a', 'b', 'c', 'n1', 'n2', 'theta', 'r_bound', 'species_id', 'n_cells',
          'cell_granule_id', 'cell_theta_local', 'cell_eta_local', 'cell_omega_local')


def _params(mode='2D', shape=False, periodic=False, **kw):
    base = dict(mode=mode, Lx=500.0, Ly=500.0, Lz=800.0, phi_solid_target=0.6, func_ratio=0.6,
                cell_surface_coverage=1.0, packing_settle_steps=60, save_data=False,
                boundary_mode='periodic' if periodic else 'walls')
    if mode == '3D':
        base.update(Lx=300.0, Ly=300.0, Lz=300.0, phi_solid_target=0.5)
    if shape:
        base.update(shape_enabled=True, aspect_ratio_func_mean=1.3, blockiness_func_mean=2.6,
                    aspect_ratio_inert_mean=1.2, blockiness_inert_mean=2.2)
        if mode == '3D':
            base.update(aspect_ratio_c_func_mean=0.9, blockiness_n2_func_mean=2.2)
    base.update(kw)
    return Params(**base)


def _both(p, seed):
    p_ref = copy.deepcopy(p)
    p_ref.use_numba = False
    p_k = copy.deepcopy(p)
    p_k.use_numba = True
    p_k.perf_neighbor_backend = 'ckdtree'        # reference pair order -> bit-identical settle
    gen = generate_packing_3d if p.mode == '3D' else generate_packing
    t0 = time.perf_counter()
    gs_r = gen(p_ref, seed=seed)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    gs_k = gen(p_k, seed=seed)
    t_k = time.perf_counter() - t0
    return gs_r, gs_k, t_ref, t_k


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestPackingBitIdentical(unittest.TestCase):

    def _check(self, p, seeds=(1, 2)):
        for seed in seeds:
            gs_r, gs_k, t_ref, t_k = _both(p, seed)
            self.assertEqual(gs_k.N, gs_r.N, f"seed {seed}: granule count")
            self.assertGreater(gs_r.N, 5)
            for name in ARRAYS:
                a = getattr(gs_r, name)
                b = getattr(gs_k, name)
                np.testing.assert_array_equal(b, a, err_msg=f"seed {seed}: {name}")
            if gs_r.quat is not None:
                np.testing.assert_array_equal(gs_k.quat, gs_r.quat, err_msg=f"seed {seed}: quat")
            self.assertEqual(gs_k.total_cells, gs_r.total_cells)

    def test_2d_circles_walls(self):
        self._check(_params('2D'))

    def test_2d_circles_periodic(self):
        self._check(_params('2D', periodic=True))

    def test_2d_shapes_walls(self):
        self._check(_params('2D', shape=True))

    def test_2d_shapes_periodic(self):
        self._check(_params('2D', shape=True, periodic=True), seeds=(3,))

    def test_3d_spheres_walls(self):
        self._check(_params('3D'), seeds=(1,))

    def test_3d_spheres_periodic(self):
        self._check(_params('3D', periodic=True), seeds=(2,))

    def test_3d_shapes_walls(self):
        self._check(_params('3D', shape=True), seeds=(1,))

    def test_no_settle_and_no_deflation(self):
        # settle disabled: only the RSA path is exercised (alpha = 1)
        self._check(_params('2D', packing_settle_steps=0), seeds=(4,))

    # ── V3.1 consolidation modes, open containers, cylinder ──
    def test_3d_consolidation_none(self):
        self._check(_params('3D', packing_consolidation='none'), seeds=(4,))

    def test_3d_gravity_free_top_box(self):
        self._check(_params('3D', phi_solid_target=0.3, boundary_top='free', gravity_enabled=True,
                            granule_density=1180.0, packing_consolidation='gravity'), seeds=(1,))

    def test_3d_gravity_cylinder(self):
        self._check(_params('3D', phi_solid_target=0.3, boundary_shape='cylinder', boundary_top='free',
                            gravity_enabled=True, granule_density=1180.0,
                            packing_consolidation='gravity'), seeds=(2,))

    def test_2d_gravity_dish(self):
        self._check(_params('2D', phi_solid_target=0.35, Ly=800.0, boundary_top='free', gravity_enabled=True,
                            granule_density=1180.0, packing_consolidation='gravity',
                            bed_phi_assumed=0.8), seeds=(3,))

    def test_three_species(self):
        species = [
            dict(name='full', f=1.0, volume_fraction=0.3, color='#CC2222', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
            dict(name='half', f=0.5, volume_fraction=0.4, color='#2255CC', radius_mean=30.0, radius_std=3.0, radius_min=15.0),
            dict(name='bare', f=0.0, volume_fraction=0.3, color='#22AA22', radius_mean=55.0, radius_std=5.0, radius_min=20.0),
        ]
        self._check(_params('2D', species=species), seeds=(5,))


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestCellListSettle(unittest.TestCase):

    def test_default_backend_packing_valid_and_thread_independent(self):
        import numba
        p = _params('2D')
        p.use_numba = True
        p.perf_neighbor_backend = 'cells'
        p_tree = copy.deepcopy(p)
        p_tree.perf_neighbor_backend = 'ckdtree'
        gs_tree = generate_packing(p_tree, seed=7)
        old = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            gs1 = generate_packing(p, seed=7)
            numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
            gs8 = generate_packing(p, seed=7)
        finally:
            numba.set_num_threads(old)
        # RSA stage identical (radii, species, counts); settle deterministic across threads
        for name in ('r', 'a', 'b', 'species_id', 'n_cells'):
            np.testing.assert_array_equal(getattr(gs1, name), getattr(gs_tree, name), err_msg=name)
        np.testing.assert_array_equal(gs1.pos, gs8.pos)
        # same jamming quality as the tree settle: no deep overlaps, similar coordination
        from gels.kernels.metrics import contact_stats_k
        from gels.kernels.neighbors import half_pairs_tree

        def stats(gs):
            pos = np.ascontiguousarray(gs.pos[:, :2])
            pi, pj = half_pairs_tree(pos, 2 * float(gs.r_bound.max()), False, (p.Lx, p.Ly))
            out = contact_stats_k(pos, gs.r, gs.r_bound, gs.a, gs.b, gs.n_shape, gs.theta,
                                  gs.gtype.astype(np.int64), gs.species_id, gs.species_f, gs.K,
                                  pi, pj, False, True, False, p.Lx, p.Ly, p.Lz)
            return out[0], out[6]     # n_contacts, max overlap ratio
        nc1, ov1 = stats(gs1)
        nct, ovt = stats(gs_tree)
        self.assertLess(ov1, 0.25)
        self.assertLess(abs(nc1 - nct), max(6, 0.25 * nct))
        self.assertTrue(np.all(np.isfinite(gs1.pos)))


if __name__ == '__main__':
    unittest.main()
