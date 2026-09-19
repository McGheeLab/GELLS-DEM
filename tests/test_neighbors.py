"""
Linked-cell neighbour list vs cKDTree (V3.0 Phase 6g).

  * the same set of half pairs (i < j, distance <= cutoff) for walls and
    periodic boxes in 2D and 3D, sorted lexicographically,
  * the fallback to the tree for periodic boxes narrower than three cells,
  * the CSR built on it, and thread-count independence,
  * the compiled overlap resolution agrees with the reference to rounding.

Run:  python -m unittest tests.test_neighbors -v
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

from gels.engine import Params, generate_packing, generate_packing_3d  # noqa: E402
from gels.kernels import HAS_NUMBA  # noqa: E402
from gels.kernels import reference as ref  # noqa: E402
from gels.kernels.neighbors import csr_from_pairs, half_pairs_cells, half_pairs_tree  # noqa: E402

if HAS_NUMBA:
    from gels.kernels import integrate  # noqa: E402


def _sorted_set(pi, pj):
    return set(zip(pi.tolist(), pj.tolist()))


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestCellList(unittest.TestCase):

    def _check(self, pos, cutoff, periodic, box):
        ti, tj = half_pairs_tree(pos, cutoff, periodic, box)
        ci, cj = half_pairs_cells(pos, cutoff, periodic, box)
        self.assertEqual(_sorted_set(ci, cj), _sorted_set(ti, tj))
        self.assertTrue(np.all(ci < cj))
        order = np.lexsort((cj, ci))
        np.testing.assert_array_equal(order, np.arange(ci.shape[0]))     # lexicographically sorted
        self.assertGreater(ci.shape[0], 0)
        return ci, cj

    def test_2d_walls_and_periodic(self):
        rng = np.random.default_rng(1)
        box = (900.0, 700.0)
        pos = np.ascontiguousarray(rng.uniform(0, 1, size=(2000, 2)) * np.array(box))
        self._check(pos, 60.0, False, box)
        self._check(pos, 60.0, True, box)

    def test_3d_walls_and_periodic(self):
        rng = np.random.default_rng(2)
        box = (400.0, 300.0, 350.0)
        pos = np.ascontiguousarray(rng.uniform(0, 1, size=(1500, 3)) * np.array(box))
        self._check(pos, 45.0, False, box)
        self._check(pos, 45.0, True, box)

    def test_periodic_narrow_box_falls_back_to_tree(self):
        rng = np.random.default_rng(3)
        box = (100.0, 100.0)
        pos = np.ascontiguousarray(rng.uniform(0, 1, size=(200, 2)) * np.array(box))
        ci, cj = half_pairs_cells(pos, 45.0, True, box)          # 100 // 45 = 2 cells < 3
        ti, tj = half_pairs_tree(pos, 45.0, True, box)
        self.assertEqual(_sorted_set(ci, cj), _sorted_set(ti, tj))

    def test_csr_and_threads(self):
        import numba
        rng = np.random.default_rng(4)
        box = (600.0, 600.0)
        pos = np.ascontiguousarray(rng.uniform(0, 1, size=(3000, 2)) * np.array(box))
        old = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            a = half_pairs_cells(pos, 50.0, False, box)
            numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
            b = half_pairs_cells(pos, 50.0, False, box)
        finally:
            numba.set_num_threads(old)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])
        off, nbr_pair, nbr_side = csr_from_pairs(a[0], a[1], pos.shape[0])
        self.assertEqual(off[-1], 2 * a[0].shape[0])
        # every incidence points back at a pair containing the particle
        for i in range(0, pos.shape[0], 97):
            for m in range(off[i], off[i + 1]):
                k = nbr_pair[m]
                self.assertEqual(i, a[0][k] if nbr_side[m] == 0 else a[1][k])


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestResolveOverlaps(unittest.TestCase):

    def _check(self, p, seed=1):
        gen = generate_packing_3d if p.mode == '3D' else generate_packing
        gs = gen(p, seed=seed)
        # push granules together so several pairs exceed the allowed overlap
        c = np.array([p.Lx, p.Ly, p.Lz]) / 2
        gs.pos[:, :3] = c + 0.85 * (gs.pos[:, :3] - c)
        gs_r = copy.deepcopy(gs)
        gs_k = copy.deepcopy(gs)
        p_ref = copy.deepcopy(p)
        p_ref.use_numba = False
        ref._resolve_overlaps(gs_r, p_ref)
        integrate.resolve_overlaps(gs_k, p)
        self.assertGreater(float(np.abs(gs_r.pos - gs.pos).max()), 0.0, "no correction happened")
        np.testing.assert_allclose(gs_k.pos, gs_r.pos, rtol=1e-12, atol=1e-9)

    def test_2d(self):
        self._check(Params(Lx=400.0, Ly=400.0, phi_solid_target=0.6, packing_settle_steps=20, save_data=False))

    def test_2d_periodic(self):
        self._check(Params(Lx=400.0, Ly=400.0, phi_solid_target=0.6, packing_settle_steps=20, save_data=False,
                           boundary_mode='periodic'))

    def test_3d(self):
        self._check(Params(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.5,
                           packing_settle_steps=20, save_data=False), seed=2)

    def test_tree_backend_bit_identical(self):
        p = Params(Lx=400.0, Ly=400.0, phi_solid_target=0.6, packing_settle_steps=20, save_data=False,
                   perf_neighbor_backend='ckdtree')
        gs = generate_packing(p, seed=3)
        c = np.array([p.Lx, p.Ly, p.Lz]) / 2
        gs.pos[:, :3] = c + 0.85 * (gs.pos[:, :3] - c)
        gs_r = copy.deepcopy(gs)
        gs_k = copy.deepcopy(gs)
        p_ref = copy.deepcopy(p)
        p_ref.use_numba = False
        ref._resolve_overlaps(gs_r, p_ref)
        integrate.resolve_overlaps(gs_k, p)
        np.testing.assert_array_equal(gs_k.pos, gs_r.pos)


if __name__ == '__main__':
    unittest.main()
