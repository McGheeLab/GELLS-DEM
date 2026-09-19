"""
Laguerre (radical) tessellation for local packing fraction (V3.3).
==================================================================

The tests that matter here are the ones that would catch a sign error or a
wrong bisector, because either produces plausible-looking numbers:

* **volume closure** -- the cells must tile the container exactly. This is the
  strongest single check; any sign error in the half-space construction breaks it.
* **monodisperse reduction** -- with equal radii the radical diagram IS the
  ordinary Voronoi diagram. This proves the RADICAL planes are right, not
  merely that the cells are bounded.
* **polydisperse correctness** -- the case the feature exists for. An ordinary
  Voronoi puts the plane at the midpoint and hands a small granule a cell
  smaller than itself, i.e. phi_loc > 1.

Run:  python -m unittest tests.test_laguerre -v
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

from gels.laguerre import radical_volumes  # noqa: E402


class TestVolumeClosure(unittest.TestCase):
    """Cells must tile the container. Catches every sign error."""

    def test_closure_2d_and_3d(self):
        rng = np.random.default_rng(0)
        for dim, box in ((2, (100.0, 100.0)), (3, (100.0, 100.0, 100.0))):
            for label, poly in (('monodisperse', False), ('polydisperse', True)):
                # a jittered lattice, like a real packing: uniformly random
                # points put some centres almost coincident, and a power cell
                # is genuinely EMPTY when a larger neighbour dominates a
                # smaller one at short range (see TestEmptyCells)
                k = 4 if dim == 3 else 6
                g = (np.arange(k) + 0.5) * (70.0 / k) + 15.0
                pos = np.stack(np.meshgrid(*([g] * dim), indexing='ij'), -1).reshape(-1, dim)
                pos = pos + rng.uniform(-2.0, 2.0, pos.shape)
                nn = len(pos)
                r = rng.uniform(3.0, 5.0, nn) if poly else np.full(nn, 4.0)
                vol, _ = radical_volumes(pos, r, box, dim)
                with self.subTest(dim=dim, kind=label):
                    self.assertEqual(np.isnan(vol).sum(), 0)
                    self.assertAlmostEqual(
                        float(np.nansum(vol)) / float(np.prod(box[:dim])), 1.0, places=9)

    def test_closure_on_a_real_packing(self):
        from gels.engine import Params, generate_packing, generate_packing_3d
        for mode, kw, dim in (('2D', dict(Lx=600.0, Ly=600.0), 2),
                              ('3D', dict(Lx=400.0, Ly=400.0, Lz=400.0), 3)):
            p = Params(mode=mode, phi_solid_target=0.55, func_ratio=0.6,
                       packing_settle_steps=40, save_data=False, **kw)
            gen = generate_packing_3d if mode == '3D' else generate_packing
            with contextlib.redirect_stdout(io.StringIO()):
                gs = gen(p, seed=5)
            box = (p.Lx, p.Ly, p.Lz)
            vol, _ = radical_volumes(gs.positions(), gs.r[:gs.N], box, dim)
            with self.subTest(mode=mode):
                self.assertEqual(np.isnan(vol).sum(), 0)
                self.assertAlmostEqual(
                    float(np.nansum(vol)) / float(np.prod(box[:dim])), 1.0, places=9)


class TestReducesToVoronoi(unittest.TestCase):

    def test_equal_radii_give_the_ordinary_voronoi_diagram(self):
        from scipy.spatial import ConvexHull, Voronoi
        rng = np.random.default_rng(1)
        L = 100.0
        pos = rng.uniform(20, 80, (25, 3))
        vlag, _ = radical_volumes(pos, np.full(25, 4.0), (L, L, L), 3)
        pts = [pos]
        for k in range(3):
            for bound in (0.0, L):
                q = pos.copy()
                q[:, k] = 2 * bound - q[:, k]
                pts.append(q)
        vor = Voronoi(np.vstack(pts))
        for i in range(25):
            reg = vor.regions[vor.point_region[i]]
            if not reg or -1 in reg:
                continue
            want = ConvexHull(vor.vertices[reg]).volume
            self.assertAlmostEqual(vlag[i] / want, 1.0, places=6)


class TestPolydisperseIsCorrect(unittest.TestCase):
    """The reason the feature exists."""

    def test_radical_plane_sits_at_the_tangency_point(self):
        # two touching spheres, r2 = 2*r1, centres 3r apart on x
        r1, r2 = 5.0, 10.0
        d = r1 + r2
        pos = np.array([[40.0, 50.0, 50.0], [40.0 + d, 50.0, 50.0]])
        r = np.array([r1, r2])
        vol, _ = radical_volumes(pos, r, (200.0, 100.0, 100.0), 3)
        # the radical plane of two TOUCHING spheres passes through the contact
        # point, at r1 from centre 1. Split a 200-long box at x = 40 + r1 = 45.
        self.assertAlmostEqual(vol[0] / (45.0 * 100.0 * 100.0), 1.0, places=6)
        self.assertAlmostEqual(vol[1] / (155.0 * 100.0 * 100.0), 1.0, places=6)

    def test_local_fraction_never_exceeds_one_on_a_real_packing(self):
        """An unweighted Voronoi drives phi_loc > 1 for the small granules at
        large size disparity; the radical diagram must not."""
        from gels.engine import Params, generate_packing_3d
        p = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, phi_solid_target=0.55,
                   func_ratio=0.6, packing_settle_steps=40, save_data=False)
        with contextlib.redirect_stdout(io.StringIO()):
            gs = generate_packing_3d(p, seed=5)
        vol, _ = radical_volumes(gs.positions(), gs.r[:gs.N], (p.Lx, p.Ly, p.Lz), 3)
        phi = (4.0 / 3.0) * np.pi * gs.r[:gs.N] ** 3 / vol
        self.assertTrue(np.all(phi <= 1.0), f"max phi_loc = {phi.max():.3f}")
        self.assertGreater(phi.mean(), 0.3)


class TestFreeTop(unittest.TestCase):

    def test_lid_cells_are_flagged_and_a_closed_box_flags_none(self):
        rng = np.random.default_rng(3)
        pos = rng.uniform(15, 60, (25, 3))
        r = np.full(25, 4.0)
        # closed box: nothing touches a lid, because there is no lid
        _, touch_closed = radical_volumes(pos, r, (100.0, 100.0, 100.0), 3)
        self.assertEqual(int(touch_closed.sum()), 0)
        # free top: the upper cells reach the loose lid and are flagged
        _, touch_free = radical_volumes(pos, r, (100.0, 100.0, 100.0), 3,
                                        top_free=True, lid=90.0)
        self.assertGreater(int(touch_free.sum()), 0)

    def test_a_valid_cell_does_not_depend_on_the_lid_height(self):
        """A cell that does not reach the lid is bounded by its neighbours, so
        its volume must not move when the lid does. (Raising the lid does
        legitimately ADD cells -- a tall but bounded cell stops touching it --
        so the aggregate is not invariant, only the per-cell volumes are.)"""
        rng = np.random.default_rng(4)
        pos = rng.uniform(15, 60, (25, 3))
        r = np.full(25, 4.0)
        vol_lo, touch_lo = radical_volumes(pos, r, (100.0, 100.0, 300.0), 3,
                                           top_free=True, lid=80.0)
        vol_hi, touch_hi = radical_volumes(pos, r, (100.0, 100.0, 300.0), 3,
                                           top_free=True, lid=160.0)
        ok = np.isfinite(vol_lo) & (~touch_lo)
        self.assertGreater(int(ok.sum()), 0)
        np.testing.assert_allclose(vol_lo[ok], vol_hi[ok], rtol=1e-9)
        # a cell clear of the low lid is clear of the high one too
        self.assertTrue(np.all(~touch_hi[ok]))


class TestCentreOutsideItsOwnCell(unittest.TestCase):
    """A granule's centre is not always inside its own power cell.

    When a much larger neighbour overlaps it deeply (|d|^2 < rj^2 - ri^2) the
    centre falls outside, but the CELL IS STILL NON-EMPTY. Seeding
    HalfspaceIntersection with the centre fails there, and returning NaN would
    silently discard a real region and break the tiling -- so the solver falls
    back to the Chebyshev centre.
    """

    def test_tiling_survives_a_centre_outside_its_cell(self):
        pos = np.array([[50.0, 50.0, 50.0], [51.0, 50.0, 50.0]])
        r = np.array([4.0, 6.0])
        # the precondition: |d|^2 = 1 < r1^2 - r0^2 = 20
        self.assertLess(1.0, r[1] ** 2 - r[0] ** 2)
        vol, _ = radical_volumes(pos, r, (100.0, 100.0, 100.0), 3)
        self.assertTrue(np.all(np.isfinite(vol)), "a non-empty cell was dropped")
        self.assertAlmostEqual(float(vol.sum()) / 1e6, 1.0, places=9)
        # the radical plane sits 10.5 um from c1 toward c0, i.e. at x = 40.5
        self.assertAlmostEqual(vol[0] / (40.5 * 100.0 * 100.0), 1.0, places=6)


class TestCylinder(unittest.TestCase):

    def test_cells_tile_the_cylinder(self):
        rng = np.random.default_rng(5)
        R, L, H = 40.0, 100.0, 100.0
        ang = rng.uniform(0, 2 * np.pi, 30)
        rad = R * 0.7 * np.sqrt(rng.uniform(0, 1, 30))
        pos = np.column_stack([L / 2 + rad * np.cos(ang),
                               L / 2 + rad * np.sin(ang),
                               rng.uniform(15, 85, 30)])
        vol, _ = radical_volumes(pos, np.full(30, 4.0), (L, L, H), 3,
                                 shape_code=1, R_cyl=R, cx=L / 2, cy=L / 2)
        want = np.pi * R ** 2 * H
        # 64 tangent planes circumscribe the circle, so the polygon is slightly
        # LARGER than the cylinder: 1/cos(pi/64)^2 - 1 = 0.24 %
        self.assertLess(abs(float(np.nansum(vol)) / want - 1.0), 0.005)


if __name__ == '__main__':
    unittest.main()
