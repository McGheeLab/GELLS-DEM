"""
Wall contact from the support function (V3.4 Phase 3.5).

A wall is a half-space, so it is the easy case of the MTD machinery: the
maximising direction is handed to us by the wall, and

    penetration = h(-w) - (c - q).w        contact point = c + grad h(-w)

is exact in one support evaluation. This replaces six brute-force surface
samplers (a 64-point ring in 2D; four 20 x 20 (eta, omega) grids in 3D) and the
hardcoded ``R_local = 0.5 * r_bound``.

**No stored fixture exercises the shaped wall path** -- checked: `run2d_shapes`
has zero wall contacts at every frame, because `packing_consolidation='centre'`
pulls the bed off the walls. So the suite passing is not evidence for this
rewrite; this file is.

The oracle is a boundary mesh 400x finer than the grid being replaced, which
knows nothing about support functions.

Run:  python -m unittest tests.test_wall_contact -v
"""

import os
import sys
import unittest
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import (  # noqa: E402
    find_contact_superellipse_wall, find_contact_superellipse_wall_cn,
    find_contact_wall_3d, find_contact_wall_3d_cn,
    find_contact_wall_plane_3d, find_contact_wall_plane_3d_cn,
    quat_random, se2d_wall_core, se3d_wall_core, se3d_wall_plane_core,
    superellipse_implicit, superellipsoid_implicit,
)
from gels.kernels.geometry2d import se2d_wall_k, se2d_wall_point_k  # noqa: E402
from gels.kernels.geometry3d import (  # noqa: E402
    wall3d_k, wall3d_plane_k, wall3d_plane_point_k, wall3d_point_k,
)

RNG = np.random.default_rng(20260919)
IDQ = np.array([1.0, 0.0, 0.0, 0.0])


class SeededCase(unittest.TestCase):
    def setUp(self):
        global RNG
        key = f"{type(self).__name__}.{self._testMethodName}"
        RNG = np.random.default_rng(zlib.crc32(key.encode()))


# ── oracles ──────────────────────────────────────────────────────────

def rot2(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def rot3(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def mesh2(a, b, n, m=200000):
    """Body-frame boundary of a superellipse, 3000x the retired 64-point ring."""
    t = np.arange(m) * (2 * np.pi / m)
    e = 2.0 / n
    ct, st = np.cos(t), np.sin(t)
    return np.stack([a * np.sign(ct) * np.abs(ct) ** e,
                     b * np.sign(st) * np.abs(st) ** e], 1)


def mesh3(a, b, c, n1, n2, ne=400, no=800):
    """Body-frame boundary of a superellipsoid, 800x the retired 20x20 grid."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, ne)
    om = np.linspace(-np.pi, np.pi, no, endpoint=False)
    E, O = np.meshgrid(eta, om, indexing='ij')
    e2, e1 = 2.0 / n2, 2.0 / n1

    def sp(v, e):
        return np.sign(v) * np.abs(v) ** e

    return np.stack([(a * sp(np.cos(E), e2) * sp(np.cos(O), e1)).ravel(),
                     (b * sp(np.cos(E), e2) * sp(np.sin(O), e1)).ravel(),
                     (c * sp(np.sin(E), e2)).ravel()], 1)


def brute_pen_2d(cx, cy, a, b, n, th, wall_pos, axis, sign):
    w = mesh2(a, b, n) @ rot2(th).T + np.array([cx, cy])
    coord = w[:, axis]
    return (wall_pos - coord.min()) if sign > 0 else (coord.max() - wall_pos)


def brute_pen_plane_3d(c, q, nrm, a, b, cc, n1, n2, quat):
    w = mesh3(a, b, cc, n1, n2) @ rot3(quat).T + c
    return float(-((w - q) @ nrm).min())


def rand_unit(rng, dim=3):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


# ── anchors ──────────────────────────────────────────────────────────

class TestAnchors(SeededCase):

    def test_circle_penetration_and_radius_are_exact(self):
        R = 40.0
        for d in (39.0, 35.0, 20.0, 0.0, -5.0):
            out = se2d_wall_core(d, 0.0, R, R, 2.0, 0.0, 0.0, 0, +1)
            self.assertTrue(out[0])
            self.assertAlmostEqual(out[1], R - d, places=9)
            self.assertAlmostEqual(out[2], R, places=4, msg="R_local of a circle is R")
            np.testing.assert_allclose(out[3:5], (d - R, 0.0), atol=1e-7)

    def test_sphere_penetration_and_radius_are_exact(self):
        R = 40.0
        for axis in (0, 1, 2):
            for sign in (+1, -1):
                c = np.zeros(3)
                c[axis] = sign * 35.0
                out = se3d_wall_core(c[0], c[1], c[2], R, R, R, 2.0, 2.0, IDQ, 0.0,
                                     axis, sign)
                self.assertTrue(out[0])
                self.assertAlmostEqual(out[1], 5.0, places=9)
                self.assertAlmostEqual(out[2], R, places=4)

    def test_no_contact_when_clear_of_the_wall(self):
        self.assertIsNone(find_contact_superellipse_wall(200.0, 0.0, 40.0, 40.0, 2.0,
                                                         0.0, 0.0, 0, +1))
        self.assertIsNone(find_contact_wall_3d(0.0, 0.0, 200.0, 40.0, 40.0, 40.0,
                                               2.0, 2.0, IDQ, 40.0, 0.0, 2, +1))
        self.assertFalse(se2d_wall_k(200.0, 0.0, 40.0, 40.0, 2.0, 0.0, 0.0, 0, +1)[0])

    def test_axis_aligned_box_touches_at_its_half_width(self):
        """n -> large is a box; its extent along a body axis is exactly that axis."""
        a, b, c = 40.0, 25.0, 33.0
        for axis, half in ((0, a), (1, b), (2, c)):
            out = se3d_wall_core(0.0, 0.0, 0.0, a, b, c, 30.0, 30.0, IDQ,
                                 -half + 3.0, axis, +1)
            self.assertTrue(out[0])
            self.assertAlmostEqual(out[1], 3.0, places=6, msg=f"axis {axis}")


# ── against a 400x finer oracle ──────────────────────────────────────

class TestAgainstABruteForceMesh(SeededCase):

    def test_2d_penetration(self):
        errs = []
        for _ in range(60):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            n = RNG.uniform(2.0, 8.0)
            th = RNG.uniform(0, 2 * np.pi)
            axis = int(RNG.integers(0, 2))
            sign = int(RNG.choice([-1, 1]))
            c = RNG.uniform(-10.0, 10.0, size=2)
            wall = c[axis] - sign * RNG.uniform(0.5, 0.95) * max(a, b)
            out = se2d_wall_core(c[0], c[1], a, b, n, th, wall, axis, sign)
            truth = brute_pen_2d(c[0], c[1], a, b, n, th, wall, axis, sign)
            if truth <= 0:
                self.assertFalse(out[0])
                continue
            self.assertTrue(out[0])
            errs.append(abs(out[1] - truth))
            # the exact answer must DOMINATE any sampled one
            self.assertGreaterEqual(out[1], truth - 1e-9)
        self.assertGreaterEqual(len(errs), 30)
        self.assertLess(max(errs), 1e-5, "um")

    def test_3d_penetration_on_a_tilted_plane(self):
        errs = []
        for _ in range(25):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            c = RNG.uniform(15.0, 45.0)
            n1 = RNG.uniform(2.0, 8.0)
            n2 = RNG.uniform(2.0, 8.0)
            q = quat_random(RNG)
            centre = RNG.uniform(-10.0, 10.0, size=3)
            nrm = rand_unit(RNG)
            pt = centre - nrm * RNG.uniform(0.5, 0.95) * max(a, b, c)
            out = se3d_wall_plane_core(centre[0], centre[1], centre[2], a, b, c,
                                       n1, n2, q, pt[0], pt[1], pt[2],
                                       nrm[0], nrm[1], nrm[2])
            truth = brute_pen_plane_3d(centre, pt, nrm, a, b, c, n1, n2, q)
            if truth <= 0:
                self.assertFalse(out[0])
                continue
            self.assertTrue(out[0])
            errs.append(abs(out[1] - truth))
            self.assertGreaterEqual(out[1], truth - 1e-9)
        self.assertGreaterEqual(len(errs), 12)
        # The residual here is the ORACLE's: a 400x800 mesh leaves ~0.45 deg
        # between samples, so it under-reports by O(R theta^2/2) ~ 1e-3 um at
        # R = 40. The one-sided check above is the real statement -- the exact
        # value dominates every sampled one, as a maximum over a superset must.
        self.assertLess(max(errs), 5e-3, "um")

    def test_contact_point_is_the_deepest_surface_point(self):
        for _ in range(25):
            a, b, c = 40.0, 28.0, 33.0
            n1 = RNG.uniform(2.0, 6.0)
            n2 = RNG.uniform(2.0, 6.0)
            q = quat_random(RNG)
            centre = RNG.uniform(-10.0, 10.0, size=3)
            nrm = rand_unit(RNG)
            pt = centre - nrm * RNG.uniform(0.5, 0.9) * max(a, b, c)
            out = se3d_wall_plane_core(centre[0], centre[1], centre[2], a, b, c,
                                       n1, n2, q, pt[0], pt[1], pt[2],
                                       nrm[0], nrm[1], nrm[2])
            if not out[0]:
                continue
            cp = np.array(out[3:6])
            # on the surface
            v = rot3(q).T @ (cp - centre)
            F = superellipsoid_implicit(v[0], v[1], v[2], a, b, c, n1, n2)
            self.assertAlmostEqual(F, 1.0, places=6)
            # and its depth past the plane IS the reported penetration
            self.assertAlmostEqual(float(-(cp - pt) @ nrm), out[1], places=6)

    def test_2d_contact_point_is_on_the_surface(self):
        for _ in range(40):
            a, b, n = 40.0, 26.0, RNG.uniform(2.0, 6.0)
            th = RNG.uniform(0, 2 * np.pi)
            axis = int(RNG.integers(0, 2))
            sign = int(RNG.choice([-1, 1]))
            c = RNG.uniform(-10.0, 10.0, size=2)
            wall = c[axis] - sign * RNG.uniform(0.5, 0.9) * max(a, b)
            hit, pen, R, cx, cy = se2d_wall_core(c[0], c[1], a, b, n, th, wall, axis, sign)
            if not hit:
                continue
            self.assertAlmostEqual(
                superellipse_implicit(cx, cy, c[0], c[1], a, b, n, th), 1.0, places=6)
            depth = (wall - (cx, cy)[axis]) if sign > 0 else ((cx, cy)[axis] - wall)
            self.assertAlmostEqual(depth, pen, places=6)


# ── exactness properties ─────────────────────────────────────────────

class TestExactness(SeededCase):

    def test_penetration_is_linear_in_the_approach(self):
        """A wall is a half-space, so d(pen)/d(c) = -w EXACTLY -- not to a
        tolerance. Moving the granule 1 um toward the wall must add exactly
        1 um of penetration, whatever the shape or orientation."""
        for _ in range(30):
            a, b, c = 40.0, 28.0, 33.0
            n1 = RNG.uniform(2.0, 8.0)
            n2 = RNG.uniform(2.0, 8.0)
            q = quat_random(RNG)
            z0 = RNG.uniform(-20.0, 0.0)
            base = se3d_wall_core(0.0, 0.0, z0, a, b, c, n1, n2, q, -45.0, 2, +1)
            if not base[0]:
                continue
            for d in (0.5, 1.0, 3.7, 10.0):
                out = se3d_wall_core(0.0, 0.0, z0 - d, a, b, c, n1, n2, q, -45.0, 2, +1)
                self.assertTrue(out[0])
                self.assertAlmostEqual(out[1] - base[1], d, places=9)
                # and the contact geometry does not move with a pure translation
                self.assertAlmostEqual(out[2], base[2], places=9, msg="R_local")

    def test_rotating_the_granule_does_not_move_the_wall_normal(self):
        """R_local and penetration depend on orientation; the wall does not."""
        a, b, c, n1, n2 = 40.0, 22.0, 31.0, 3.5, 2.8
        pens = []
        for _ in range(20):
            q = quat_random(RNG)
            out = se3d_wall_core(0.0, 0.0, 0.0, a, b, c, n1, n2, q, -30.0, 2, +1)
            if out[0]:
                pens.append(out[1])
        self.assertGreater(len(pens), 5)
        # the extent along z varies between the short and long semi-axis
        self.assertGreater(max(pens) - min(pens), 1.0)

    def test_axis_wall_equals_the_equivalent_plane(self):
        for _ in range(30):
            a, b, c = 40.0, 28.0, 33.0
            n1, n2 = RNG.uniform(2.0, 6.0), RNG.uniform(2.0, 6.0)
            q = quat_random(RNG)
            centre = RNG.uniform(-15.0, 15.0, size=3)
            axis = int(RNG.integers(0, 3))
            sign = int(RNG.choice([-1, 1]))
            wall = centre[axis] - sign * RNG.uniform(0.4, 0.95) * max(a, b, c)
            ax = se3d_wall_core(centre[0], centre[1], centre[2], a, b, c, n1, n2, q,
                                wall, axis, sign)
            q_pt = centre.copy()
            q_pt[axis] = wall
            nrm = np.zeros(3)
            nrm[axis] = sign
            pl = se3d_wall_plane_core(centre[0], centre[1], centre[2], a, b, c, n1, n2, q,
                                      q_pt[0], q_pt[1], q_pt[2], nrm[0], nrm[1], nrm[2])
            self.assertEqual(ax[0], pl[0])
            if ax[0]:
                np.testing.assert_allclose(np.array(ax[1:]), np.array(pl[1:]), atol=1e-9)


# ── twin parity ──────────────────────────────────────────────────────

class TestTwinParity(SeededCase):

    def test_python_and_compiled_wall_solvers_agree_exactly(self):
        for _ in range(40):
            a, b, c = 40.0, 28.0, 33.0
            n1, n2 = RNG.uniform(2.0, 6.0), RNG.uniform(2.0, 6.0)
            q = quat_random(RNG)
            centre = RNG.uniform(-15.0, 15.0, size=3)
            axis = int(RNG.integers(0, 3))
            sign = int(RNG.choice([-1, 1]))
            wall = centre[axis] - sign * RNG.uniform(0.4, 0.95) * max(a, b, c)
            py = find_contact_wall_3d(centre[0], centre[1], centre[2], a, b, c,
                                      n1, n2, q, max(a, b, c), wall, axis, sign)
            kn = wall3d_k(centre[0], centre[1], centre[2], a, b, c, n1, n2, q,
                          max(a, b, c), wall, axis, sign)
            self.assertEqual(py is not None, bool(kn[0]))
            if py is not None:
                self.assertEqual(py[0], kn[1])
                self.assertEqual(py[1], kn[2])

    def test_2d_python_and_compiled_agree_exactly(self):
        for _ in range(40):
            a, b, n = 40.0, 26.0, RNG.uniform(2.0, 6.0)
            th = RNG.uniform(0, 2 * np.pi)
            axis = int(RNG.integers(0, 2))
            sign = int(RNG.choice([-1, 1]))
            cx, cy = RNG.uniform(-10.0, 10.0, size=2)
            wall = (cx, cy)[axis] - sign * RNG.uniform(0.4, 0.95) * max(a, b)
            py = find_contact_superellipse_wall(cx, cy, a, b, n, th, wall, axis, sign)
            kn = se2d_wall_k(cx, cy, a, b, n, th, wall, axis, sign)
            self.assertEqual(py is not None, bool(kn[0]))
            if py is not None:
                self.assertEqual(py[0], kn[1])
                self.assertEqual(py[1], kn[2])

    def test_point_variants_agree_with_the_plain_ones(self):
        a, b, c, n1, n2 = 40.0, 28.0, 33.0, 3.0, 4.0
        q = quat_random(RNG)
        plain = wall3d_k(0.0, 0.0, -20.0, a, b, c, n1, n2, q, 40.0, -45.0, 2, +1)
        pt = wall3d_point_k(0.0, 0.0, -20.0, a, b, c, n1, n2, q, -45.0, 2, +1)
        self.assertEqual(plain, pt[:3])
        p2 = se2d_wall_point_k(0.0, 0.0, a, b, n1, 0.3, -35.0, 1, +1)
        self.assertEqual(se2d_wall_k(0.0, 0.0, a, b, n1, 0.3, -35.0, 1, +1), p2[:3])


# ── what it replaces ─────────────────────────────────────────────────

class TestAgainstTheRetiredSamplers(SeededCase):
    """Reproducible evidence for the V3.4 changelog."""

    def test_the_hardcoded_wall_radius_was_a_factor_of_two_low_for_a_sphere(self):
        """`R_local = 0.5 * r_bound` had no geometric content. For a sphere of
        radius R against a flat wall the true radius of curvature is R, so the
        old value was exactly 2x low -- and `F ~ sqrt(R_local)`, so every wall
        contact was 1.41x too soft before any shape effect."""
        R = 40.0
        old = find_contact_wall_3d_cn(0.0, 0.0, 35.0, R, R, R, 2.0, 2.0, IDQ, R,
                                      0.0, 2, +1)
        new = find_contact_wall_3d(0.0, 0.0, 35.0, R, R, R, 2.0, 2.0, IDQ, R, 0.0, 2, +1)
        self.assertAlmostEqual(old[1], 0.5 * R, places=9)
        self.assertAlmostEqual(new[1], R, places=4)
        self.assertAlmostEqual(np.sqrt(new[1] / old[1]), np.sqrt(2.0), places=4)

    def test_the_sampled_penetration_was_grid_limited(self):
        """20 x 20 on a superellipsoid is ~9 degrees between samples, so the
        deepest point is missed by O(R theta^2 / 2). Measured against the exact
        value over random tumbled granules."""
        errs = []
        for _ in range(25):
            a, b, c = 40.0, 28.0, 33.0
            n1, n2 = RNG.uniform(2.0, 4.0), RNG.uniform(2.0, 4.0)
            q = quat_random(RNG)
            z0 = RNG.uniform(-15.0, 15.0)
            wall = z0 - RNG.uniform(0.5, 0.9) * max(a, b, c)
            new = find_contact_wall_3d(0.0, 0.0, z0, a, b, c, n1, n2, q, 40.0, wall, 2, +1)
            old = find_contact_wall_3d_cn(0.0, 0.0, z0, a, b, c, n1, n2, q, 40.0,
                                          wall, 2, +1)
            if new is None or old is None:
                continue
            # the sampler can only ever UNDER-report: it maximises over a subset
            self.assertLessEqual(old[0], new[0] + 1e-9)
            errs.append(new[0] - old[0])
        self.assertGreaterEqual(len(errs), 12)
        self.assertGreater(np.median(errs), 1e-3,
                           "the 20x20 grid should be visibly coarse")

    def test_the_2d_ring_sampler_was_also_coarse(self):
        errs = []
        for _ in range(40):
            a, b = 40.0, 26.0
            n = RNG.uniform(2.0, 4.0)
            th = RNG.uniform(0, 2 * np.pi)
            cx = RNG.uniform(-10.0, 10.0)
            wall = cx - RNG.uniform(0.5, 0.9) * max(a, b)
            new = find_contact_superellipse_wall(cx, 0.0, a, b, n, th, wall, 0, +1)
            old = find_contact_superellipse_wall_cn(cx, 0.0, a, b, n, th, wall, 0, +1)
            if new is None or old is None:
                continue
            self.assertLessEqual(old[0], new[0] + 1e-9)
            errs.append(new[0] - old[0])
        self.assertGreaterEqual(len(errs), 20)
        self.assertGreater(np.median(errs), 1e-4)


# ── wall torque (contact.wall_torque) ────────────────────────────────

class TestWallTorque(SeededCase):
    """`contact.wall_torque` (V3.4, default ON for non-spherical granules).

    The lever arm is free once the wall solver returns its contact point.
    Discarding it was an omission, not a modelling choice: a shaped granule that
    cannot tip flat against a wall is wrong. Both twins AND the flag with
    ``not is_circle``, and a sphere's wall contact is on its own centre line, so
    its wall torque is identically zero -- which is why turning this on by
    default cannot move a sphere run.
    """

    @staticmethod
    def _wall_energy_torque_2d(cx, cy, a, b, n, th, wall_pos, axis, sign, Fw):
        """-dE/dtheta for a wall contact, from the support function alone.

        `pen = h(w; theta) - sign*(c - wall)`, and the repulsive force magnitude
        is `E'(pen) = Fw`, so the generalised force conjugate to theta is
        `-Fw * dh/dtheta`. Finite-differenced here, so it shares nothing with the
        `r x F` the solver computes.
        """
        eps = 1e-6

        def h(t):
            w = (1.0, 0.0) if axis == 0 else (0.0, 1.0)
            from gels.engine import se2d_support
            ct, st = np.cos(t), np.sin(t)
            bx = ct * w[0] + st * w[1]
            by = -st * w[0] + ct * w[1]
            return se2d_support(bx, by, a, b, n)[0]

        return -Fw * (h(th + eps) - h(th - eps)) / (2 * eps)

    def test_torque_is_minus_the_energy_gradient_in_orientation(self):
        """The conservativity statement: r x F at the support point equals
        -Fw * dh/dtheta, finite-differenced independently."""
        from gels.kernels.geometry2d import se2d_wall_point_k
        errs = []
        for _ in range(60):
            a, b = 40.0, 24.0
            n = RNG.uniform(2.0, 6.0)
            th = RNG.uniform(0, 2 * np.pi)
            axis = int(RNG.integers(0, 2))
            sign = int(RNG.choice([-1, 1]))
            c = RNG.uniform(-10.0, 10.0, size=2)
            wall = c[axis] - sign * RNG.uniform(0.5, 0.9) * max(a, b)
            hit, pen, R, wcx, wcy = se2d_wall_point_k(c[0], c[1], a, b, n, th,
                                                      wall, axis, sign)
            if not hit:
                continue
            Fw = 13.7      # any magnitude; both sides are linear in it
            if axis == 0:
                tau = -(wcy - c[1]) * sign * Fw
            else:
                tau = (wcx - c[0]) * sign * Fw
            ref = self._wall_energy_torque_2d(c[0], c[1], a, b, n, th, wall, axis,
                                              sign, Fw)
            errs.append(abs(tau - ref) / max(abs(ref), 1.0))
        self.assertGreaterEqual(len(errs), 30)
        self.assertLess(float(np.median(errs)), 1e-5)
        self.assertLess(float(np.percentile(errs, 90)), 1e-3)

    def test_a_circle_has_no_wall_torque(self):
        """Its contact point is on the centre line, so r is parallel to F. This
        is why both kernels skip the branch entirely when `is_circle`."""
        from gels.kernels.geometry2d import se2d_wall_point_k
        for _ in range(20):
            R = 40.0
            cy = RNG.uniform(-50.0, 50.0)
            hit, pen, Rl, wcx, wcy = se2d_wall_point_k(20.0, cy, R, R, 2.0,
                                                       RNG.uniform(0, 6.28), 0.0, 0, +1)
            self.assertTrue(hit)
            self.assertAlmostEqual(wcy, cy, places=7)      # zero lever arm

    def test_torque_rotates_a_blocky_granule_flat_against_the_wall(self):
        """The falsifiable physical claim. A square-ish granule tilted against a
        flat wall must feel a torque toward face-on, i.e. toward the orientation
        that MINIMISES its extent along the wall normal."""
        from gels.kernels.geometry2d import se2d_wall_point_k
        from gels.engine import se2d_support
        a = b = 35.0
        n = 8.0                      # nearly a square
        Fw = 10.0
        for th in (0.2, 0.5, 0.9, 1.1, -0.3, -0.7):
            wall = -30.0
            hit, pen, R, wcx, wcy = se2d_wall_point_k(0.0, 0.0, a, b, n, th, wall, 0, +1)
            self.assertTrue(hit, f"th={th}")
            tau = -(wcy - 0.0) * Fw
            # dh/dtheta at this orientation
            eps = 1e-6
            hp = se2d_support(np.cos(th + eps), -np.sin(th + eps), a, b, n)[0]
            hm = se2d_support(np.cos(th - eps), -np.sin(th - eps), a, b, n)[0]
            dh = (hp - hm) / (2 * eps)
            # torque must oppose the growth of the extent -> drive toward face-on
            self.assertLess(tau * dh, 0.0, f"th={th}: torque increases the extent")

    @staticmethod
    def _bed_pressed_against_a_wall(seed=5):
        """A packed shaped bed, translated until granules really overlap x = 0.

        Necessary, and it is the reason these tests exist: a bed packed with the
        default `packing_consolidation = 'centre'` sits OFF the walls, so a run
        can look healthy while the whole wall path is dead code.
        """
        import copy
        import contextlib
        import io as _io
        from gels.engine import Params, generate_packing
        base = dict(mode='2D', Lx=300.0, Ly=300.0, shape_enabled=True,
                    aspect_ratio_func_mean=1.8, blockiness_func_mean=3.5,
                    aspect_ratio_inert_mean=1.6, blockiness_inert_mean=3.2,
                    packing_shape_contact=True, contact_shape_dynamics=True,
                    packing_consolidation='none', phi_solid_target=0.55,
                    cell_surface_coverage=0.0, n_cells_per_granule=0, save_data=False)
        p = Params(**base, contact_wall_torque=False)
        with contextlib.redirect_stdout(_io.StringIO()):
            gs = generate_packing(copy.deepcopy(p), seed=seed)
        # put the leftmost centre half a bounding radius from x = 0, so its true
        # extent certainly crosses whatever its orientation happens to be
        shift = gs.x.min() - 0.5 * float(gs.r_bound.min())
        gs.pos[:, 0] -= shift
        gs.x[:] = gs.pos[:, 0]
        return gs, base

    def test_the_flag_changes_the_torque_and_only_the_torque(self):
        from gels.engine import Params
        from gels.kernels.contact2d import compute_forces_2d
        gs, base = self._bed_pressed_against_a_wall()
        self.assertTrue(_any_wall_contact(gs, Params(**base)),
                        "the fixture must actually touch a wall")
        off = Params(**base, contact_wall_torque=False)
        on = Params(**base, contact_wall_torque=True)
        F0, t0, _ = compute_forces_2d(gs, off, np.random.default_rng(0))
        F1, t1, _ = compute_forces_2d(gs, on, np.random.default_rng(0))
        np.testing.assert_array_equal(F0, F1, "the flag must not move any force")
        self.assertGreater(int(np.count_nonzero(np.asarray(t0) != np.asarray(t1))), 0,
                           "the flag did nothing on a bed that touches walls")

    def test_a_sphere_bed_is_bit_identical_with_the_flag_on(self):
        """The safety property the ON-by-default rests on.

        Both twins AND the flag with ``not is_circle``, and a sphere's wall
        contact is on its own centre line, so `r x F` is identically zero. The
        default is therefore exactly "on for non-spherical granules" and no
        sphere run can move. Gate B is the end-to-end form of this claim --
        `run2d_walls` and `run3d_spheres` are unchanged at atol = 0 across the
        default flip -- and this is the direct one.
        """
        import contextlib
        import copy
        import io as _io
        from gels.engine import Params, generate_packing_3d
        from gels.kernels.contact3d import compute_forces_3d
        base = dict(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.5,
                    cell_surface_coverage=1.0, packing_settle_steps=40, save_data=False)
        off = Params(**base, contact_wall_torque=False)
        with contextlib.redirect_stdout(_io.StringIO()):
            gs = generate_packing_3d(copy.deepcopy(off), seed=4)
        self.assertTrue(gs.is_circle, "this fixture must be spheres")
        on = Params(**base, contact_wall_torque=True)
        # press the bed into the x = 0 wall so the wall branch is live
        gs.pos[:, 0] -= gs.x.min() - 0.5 * float(gs.r.min())
        gs.x[:] = gs.pos[:, 0]
        F0, t0, _ = compute_forces_3d(gs, off, np.random.default_rng(0))
        F1, t1, _ = compute_forces_3d(gs, on, np.random.default_rng(0))
        np.testing.assert_array_equal(F0, F1)
        np.testing.assert_array_equal(np.asarray(t0), np.asarray(t1))

    def test_both_twins_agree_with_the_flag_on(self):
        from gels.engine import Params
        import gels.kernels.reference as reference
        from gels.kernels.contact2d import compute_forces_2d
        gs, base = self._bed_pressed_against_a_wall()
        on = Params(**base, contact_wall_torque=True)
        Fk, tk, _ = compute_forces_2d(gs, on, np.random.default_rng(0))
        Fr, tr, _ = reference.compute_forces(gs, on, np.random.default_rng(0))
        np.testing.assert_allclose(np.asarray(Fr), Fk, rtol=0, atol=1e-12)
        np.testing.assert_allclose(np.asarray(tr), np.asarray(tk), rtol=0, atol=1e-12)


def _any_wall_contact(gs, p):
    from gels.engine import se2d_wall_core
    walls = [(0.0, 0, +1), (p.Lx, 0, -1), (0.0, 1, +1), (p.Ly, 1, -1)]
    for i in range(gs.N):
        for wp, ax, sg in walls:
            if se2d_wall_core(gs.x[i], gs.y[i], gs.a[i], gs.b[i], gs.n_shape[i],
                              gs.theta[i], wp, ax, sg)[0]:
                return True
    return False


if __name__ == '__main__':
    unittest.main()
