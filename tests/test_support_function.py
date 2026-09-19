"""
Support function of the superellipse / superellipsoid (V3.4 Phase 3.1).

``h(n) = max_{x in K} x.n`` is the object the minimum-translation-distance solver
is built on, and these tests pin it against brute force rather than against the
derivation that produced it:

* ``h`` vs an explicit maximisation over a dense surface mesh;
* ``grad h`` vs the support POINT found by that same maximisation (the envelope
  theorem, which is what makes the contact point free);
* ``(eta, omega)`` vs ``superellipsoid_point``, which is what keeps every existing
  parametric consumer working;
* ``support_R_eff_*`` vs a circle fitted to the surface, and against the analytic
  ellipse/ellipsoid radii where those exist;
* the numerical hazards the implementation exists to avoid: q = 60 at a = 40 um
  would overflow an unfactored norm, and n <= 1 has no Holder conjugate.

Run:  python -m unittest tests.test_support_function -v
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

from gels.engine import (  # noqa: E402
    Q_MAX, _dual_exp, se2d_support, se3d_support,
    support_R_eff_2d, support_R_eff_3d,
    superellipse_point, superellipsoid_point, superellipsoid_normal,
)

RNG = np.random.default_rng(20260919)


# ── brute-force oracles ──────────────────────────────────────────────

def brute_support_2d(nx, ny, a, b, n, m=200000):
    t = np.arange(m) * (2.0 * np.pi / m)
    pts = np.array([superellipse_point(tt, a, b, n) for tt in t[::max(1, m // 4000)]])
    d = pts[:, 0] * nx + pts[:, 1] * ny
    k = int(np.argmax(d))
    return d[k], pts[k]


def brute_support_3d(nx, ny, nz, a, b, c, n1, n2, ne=600, no=1200):
    """max over a dense (eta, omega) mesh. Mesh-limited to ~1e-5 relative."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, ne)
    om = np.linspace(-np.pi, np.pi, no, endpoint=False)
    E, O = np.meshgrid(eta, om, indexing='ij')
    e2, e1 = 2.0 / n2, 2.0 / n1
    ce, se = np.cos(E), np.sin(E)
    co, so = np.cos(O), np.sin(O)

    def sp(v, e):
        return np.sign(v) * np.abs(v) ** e

    X = a * sp(ce, e2) * sp(co, e1)
    Y = b * sp(ce, e2) * sp(so, e1)
    Z = c * sp(se, e2)
    D = X * nx + Y * ny + Z * nz
    k = np.unravel_index(np.argmax(D), D.shape)
    return D[k], np.array([X[k], Y[k], Z[k]])


def rand_unit(rng, dim):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


# ── the conjugate exponent ───────────────────────────────────────────

class TestDualExponent(unittest.TestCase):

    def test_ellipse_is_self_dual(self):
        self.assertAlmostEqual(_dual_exp(2.0), 2.0, places=12)

    def test_box_limit_is_the_one_norm(self):
        # n -> inf is the box, whose support function is the 1-norm
        self.assertLess(_dual_exp(1e6), 1.0 + 1e-5)
        self.assertGreaterEqual(_dual_exp(1e6), 1.0)

    def test_nonconvex_exponents_map_to_the_max_norm(self):
        # n <= 1 has no Holder conjugate (n/(n-1) is negative or infinite).
        for n in (0.2, 0.9, 1.0, 1.0 + 1e-12):
            self.assertEqual(_dual_exp(n), Q_MAX, msg=f"n={n}")

    def test_clamped_from_above_never_below_one(self):
        for n in np.linspace(1.001, 50.0, 200):
            q = _dual_exp(n)
            self.assertGreaterEqual(q, 1.0)
            self.assertLessEqual(q, Q_MAX)


# ── 2D ───────────────────────────────────────────────────────────────

class TestSupport2D(unittest.TestCase):

    def test_circle_is_exact(self):
        R = 37.0
        for _ in range(50):
            nx, ny = rand_unit(RNG, 2)
            h, gx, gy = se2d_support(nx, ny, R, R, 2.0)
            self.assertAlmostEqual(h, R, places=10)
            np.testing.assert_allclose([gx, gy], [R * nx, R * ny], atol=1e-9)

    def test_ellipse_is_exact(self):
        a, b = 40.0, 25.0
        for _ in range(50):
            nx, ny = rand_unit(RNG, 2)
            h, gx, gy = se2d_support(nx, ny, a, b, 2.0)
            self.assertAlmostEqual(h, np.hypot(a * nx, b * ny), places=9)

    def test_matches_brute_force_over_the_surface(self):
        for _ in range(40):
            a = RNG.uniform(10.0, 50.0)
            b = RNG.uniform(10.0, 50.0)
            n = RNG.uniform(2.0, 6.0)
            nx, ny = rand_unit(RNG, 2)
            h, gx, gy = se2d_support(nx, ny, a, b, n)
            hb, pb = brute_support_2d(nx, ny, a, b, n)
            self.assertGreaterEqual(h + 1e-9, hb, "support must dominate every surface point")
            self.assertLess(abs(h - hb) / h, 2e-5, f"a={a} b={b} n={n}")
            self.assertLess(np.linalg.norm(np.array([gx, gy]) - pb), 5e-3 * max(a, b))

    def test_gradient_is_the_support_point_on_the_boundary(self):
        for _ in range(40):
            a = RNG.uniform(10.0, 50.0)
            b = RNG.uniform(10.0, 50.0)
            n = RNG.uniform(2.0, 8.0)
            nx, ny = rand_unit(RNG, 2)
            h, gx, gy = se2d_support(nx, ny, a, b, n)
            # the implicit function must be 1 at the support point
            F = abs(gx / a) ** n + abs(gy / b) ** n
            self.assertAlmostEqual(F, 1.0, places=8, msg=f"a={a} b={b} n={n}")
            # and Euler's identity h = x.n
            self.assertAlmostEqual(gx * nx + gy * ny, h, places=8)

    def test_gradient_matches_finite_difference_of_h(self):
        a, b, n = 40.0, 22.0, 3.7
        eps = 1e-6
        for _ in range(20):
            nx, ny = rand_unit(RNG, 2)
            h, gx, gy = se2d_support(nx, ny, a, b, n)
            hp, _, _ = se2d_support(nx + eps, ny, a, b, n)
            hm, _, _ = se2d_support(nx - eps, ny, a, b, n)
            self.assertAlmostEqual((hp - hm) / (2 * eps), gx, places=4)
            hp, _, _ = se2d_support(nx, ny + eps, a, b, n)
            hm, _, _ = se2d_support(nx, ny - eps, a, b, n)
            self.assertAlmostEqual((hp - hm) / (2 * eps), gy, places=4)

    def test_positively_homogeneous_of_degree_one(self):
        a, b, n = 33.0, 18.0, 4.2
        nx, ny = rand_unit(RNG, 2)
        h1, gx1, gy1 = se2d_support(nx, ny, a, b, n)
        h2, gx2, gy2 = se2d_support(7.3 * nx, 7.3 * ny, a, b, n)
        self.assertAlmostEqual(h2, 7.3 * h1, places=8)
        # grad h is degree 0 -- this is what lets the solver skip normalising
        self.assertAlmostEqual(gx2, gx1, places=10)
        self.assertAlmostEqual(gy2, gy1, places=10)


# ── 3D ───────────────────────────────────────────────────────────────

class TestSupport3D(unittest.TestCase):

    def test_sphere_is_exact(self):
        R = 40.0
        for _ in range(50):
            nx, ny, nz = rand_unit(RNG, 3)
            h, gx, gy, gz, eta, om = se3d_support(nx, ny, nz, R, R, R, 2.0, 2.0)
            self.assertAlmostEqual(h, R, places=10)
            np.testing.assert_allclose([gx, gy, gz], [R * nx, R * ny, R * nz], atol=1e-9)

    def test_ellipsoid_is_exact(self):
        a, b, c = 40.0, 30.0, 20.0
        for _ in range(50):
            nx, ny, nz = rand_unit(RNG, 3)
            h, gx, gy, gz, _, _ = se3d_support(nx, ny, nz, a, b, c, 2.0, 2.0)
            self.assertAlmostEqual(h, np.sqrt((a * nx) ** 2 + (b * ny) ** 2 + (c * nz) ** 2),
                                   places=9)

    def test_matches_brute_force_with_unequal_exponents(self):
        """The whole point: this is NOT restricted to n1 == n2 or a == b == c."""
        worst = 0.0
        for _ in range(12):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            c = RNG.uniform(15.0, 45.0)
            n1 = RNG.uniform(2.0, 6.0)
            n2 = RNG.uniform(2.0, 6.0)
            nx, ny, nz = rand_unit(RNG, 3)
            h, gx, gy, gz, _, _ = se3d_support(nx, ny, nz, a, b, c, n1, n2)
            hb, pb = brute_support_3d(nx, ny, nz, a, b, c, n1, n2)
            self.assertGreaterEqual(h + 1e-9, hb, "support must dominate every surface point")
            rel = abs(h - hb) / h
            worst = max(worst, rel)
            self.assertLess(rel, 1e-4, f"a={a} b={b} c={c} n1={n1} n2={n2}")
        self.assertLess(worst, 1e-4)

    def test_gradient_lands_on_the_surface(self):
        for _ in range(60):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            c = RNG.uniform(15.0, 45.0)
            n1 = RNG.uniform(2.0, 8.0)
            n2 = RNG.uniform(2.0, 8.0)
            nx, ny, nz = rand_unit(RNG, 3)
            h, gx, gy, gz, _, _ = se3d_support(nx, ny, nz, a, b, c, n1, n2)
            F = (abs(gx / a) ** n1 + abs(gy / b) ** n1) ** (n2 / n1) + abs(gz / c) ** n2
            self.assertAlmostEqual(F, 1.0, places=7, msg=f"n1={n1} n2={n2}")
            self.assertAlmostEqual(gx * nx + gy * ny + gz * nz, h, places=7)

    def test_parametric_inversion_reproduces_the_gradient(self):
        """(eta, omega) must feed superellipsoid_point back to grad h."""
        worst = 0.0
        for _ in range(80):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            c = RNG.uniform(15.0, 45.0)
            n1 = RNG.uniform(2.0, 8.0)
            n2 = RNG.uniform(2.0, 8.0)
            nx, ny, nz = rand_unit(RNG, 3)
            h, gx, gy, gz, eta, om = se3d_support(nx, ny, nz, a, b, c, n1, n2)
            p = superellipsoid_point(eta, om, a, b, c, n1, n2)
            err = np.linalg.norm(p - np.array([gx, gy, gz])) / max(a, b, c)
            worst = max(worst, err)
            self.assertLess(err, 1e-9, f"n1={n1} n2={n2}")
        self.assertLess(worst, 1e-9)

    def test_normal_at_the_inverted_parameters_is_the_query_direction(self):
        for _ in range(40):
            a, b, c = 40.0, 28.0, 33.0
            n1 = RNG.uniform(2.0, 6.0)
            n2 = RNG.uniform(2.0, 6.0)
            u = rand_unit(RNG, 3)
            _, _, _, _, eta, om = se3d_support(u[0], u[1], u[2], a, b, c, n1, n2)
            nn = superellipsoid_normal(eta, om, a, b, c, n1, n2)
            self.assertGreater(float(np.dot(nn, u)), 1.0 - 1e-7, f"n1={n1} n2={n2}")

    def test_gradient_matches_finite_difference_of_h(self):
        a, b, c, n1, n2 = 40.0, 22.0, 31.0, 3.7, 2.6
        eps = 1e-6
        for _ in range(20):
            u = rand_unit(RNG, 3)
            h, gx, gy, gz, _, _ = se3d_support(u[0], u[1], u[2], a, b, c, n1, n2)
            for k, g in enumerate((gx, gy, gz)):
                up = u.copy(); up[k] += eps
                um = u.copy(); um[k] -= eps
                hp = se3d_support(up[0], up[1], up[2], a, b, c, n1, n2)[0]
                hm = se3d_support(um[0], um[1], um[2], a, b, c, n1, n2)[0]
                self.assertAlmostEqual((hp - hm) / (2 * eps), g, places=4)

    def test_polar_axis_returns_the_pole(self):
        a, b, c, n1, n2 = 40.0, 25.0, 18.0, 3.5, 4.5
        for sgn in (+1.0, -1.0):
            h, gx, gy, gz, eta, om = se3d_support(0.0, 0.0, sgn, a, b, c, n1, n2)
            self.assertAlmostEqual(h, c, places=12)
            self.assertAlmostEqual(gx, 0.0, places=12)
            self.assertAlmostEqual(gy, 0.0, places=12)
            self.assertAlmostEqual(gz, sgn * c, places=12)
            self.assertAlmostEqual(eta, sgn * np.pi / 2, places=12)

    def test_equatorial_normal_has_no_z_component(self):
        a, b, c, n1, n2 = 40.0, 25.0, 18.0, 3.5, 4.5
        h, gx, gy, gz, eta, om = se3d_support(1.0, 0.0, 0.0, a, b, c, n1, n2)
        self.assertAlmostEqual(h, a, places=12)
        self.assertAlmostEqual(gz, 0.0, places=12)
        self.assertAlmostEqual(eta, 0.0, places=12)

    def test_convexity_the_property_the_solver_relies_on(self):
        """h must be convex: h(u+v) <= h(u) + h(v) for every pair of directions.

        `sep(n) = (c2-c1).n - h1(n) - h2(n)` is concave *because* of this, which is
        what makes a positive sep at any n a certificate of separation.
        """
        a, b, c, n1, n2 = 40.0, 25.0, 33.0, 4.0, 3.0
        for _ in range(200):
            u = rand_unit(RNG, 3) * RNG.uniform(0.2, 3.0)
            v = rand_unit(RNG, 3) * RNG.uniform(0.2, 3.0)
            hu = se3d_support(u[0], u[1], u[2], a, b, c, n1, n2)[0]
            hv = se3d_support(v[0], v[1], v[2], a, b, c, n1, n2)[0]
            w = u + v
            hw = se3d_support(w[0], w[1], w[2], a, b, c, n1, n2)[0]
            self.assertLessEqual(hw, hu + hv + 1e-9)


# ── the numerical hazards the implementation exists to avoid ─────────

class TestNumericalHazards(unittest.TestCase):

    def test_no_overflow_at_the_clamped_exponent(self):
        """|a n_x|^60 at a = 40 um is ~1e96; an unfactored norm overflows here."""
        a, b, c = 40.0, 40.0, 40.0
        for n in (1.0000001, 1.001, 1.02):     # q pinned at Q_MAX = 60
            for _ in range(20):
                u = rand_unit(RNG, 3)
                h, gx, gy, gz, _, _ = se3d_support(u[0], u[1], u[2], a, b, c, n, n)
                self.assertTrue(np.isfinite(h) and h > 0.0, f"n={n}")
                self.assertTrue(np.all(np.isfinite([gx, gy, gz])), f"n={n}")

    def test_octahedral_limit_is_the_max_norm(self):
        """n -> 1 is the octahedron |x/a|+|y/b|+|z/c| = 1, support = max |s_k n_k|."""
        a, b, c = 40.0, 25.0, 33.0
        for _ in range(30):
            u = rand_unit(RNG, 3)
            h = se3d_support(u[0], u[1], u[2], a, b, c, 1.0, 1.0)[0]
            expect = max(abs(a * u[0]), abs(b * u[1]), abs(c * u[2]))
            self.assertLess(abs(h - expect) / expect, 0.25)   # q = 60, not infinity

    def test_zero_direction_returns_zero_not_nan(self):
        self.assertEqual(se2d_support(0.0, 0.0, 40.0, 25.0, 3.0), (0.0, 0.0, 0.0))
        self.assertEqual(se3d_support(0.0, 0.0, 0.0, 40.0, 25.0, 33.0, 3.0, 4.0),
                         (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    def test_axis_aligned_directions_are_finite_for_blocky_shapes(self):
        """A flat face is where the weight is exactly 1 and its partner exactly 0."""
        a, b, c = 40.0, 25.0, 33.0
        for n in (2.0, 4.0, 10.0, 40.0):
            for d in ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.),
                      (-1., 0., 0.), (0., -1., 0.), (0., 0., -1.)):
                out = se3d_support(d[0], d[1], d[2], a, b, c, n, n)
                self.assertTrue(np.all(np.isfinite(out)), f"n={n} d={d}")
            expect = (a, b, c)
            for k, d in enumerate(((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))):
                self.assertAlmostEqual(se3d_support(d[0], d[1], d[2], a, b, c, n, n)[0],
                                       expect[k], places=9)


# ── curvature ────────────────────────────────────────────────────────

class TestSupportCurvature(unittest.TestCase):

    def test_sphere_radius_is_the_radius(self):
        R = 40.0
        for _ in range(30):
            u = rand_unit(RNG, 3)
            self.assertAlmostEqual(support_R_eff_3d(u[0], u[1], u[2], R, R, R, 2.0, 2.0),
                                   R, places=6)

    def test_circle_radius_is_the_radius(self):
        R = 37.0
        for _ in range(30):
            u = rand_unit(RNG, 2)
            self.assertAlmostEqual(support_R_eff_2d(u[0], u[1], R, R, 2.0), R, places=6)

    def test_ellipse_matches_the_analytic_radius(self):
        """R(theta) = (a^2 sin^2 + b^2 cos^2)^{3/2} / (a b) at the parametric angle."""
        a, b = 40.0, 22.0
        for t in np.linspace(0.0, 2 * np.pi, 17)[:-1]:
            # outward normal direction at parametric angle t
            nx, ny = np.cos(t) / a, np.sin(t) / b
            m = np.hypot(nx, ny)
            R = support_R_eff_2d(nx / m, ny / m, a, b, 2.0)
            expect = (a ** 2 * np.sin(t) ** 2 + b ** 2 * np.cos(t) ** 2) ** 1.5 / (a * b)
            self.assertLess(abs(R - expect) / expect, 1e-5, f"t={t}")

    def test_ellipsoid_matches_the_analytic_gaussian_radius(self):
        """sqrt(1/K) with K = 1/(a^2 b^2 c^2 * (x^2/a^4 + y^2/b^4 + z^2/c^4)^2)."""
        a, b, c = 40.0, 30.0, 22.0
        for _ in range(30):
            u = rand_unit(RNG, 3)
            h, gx, gy, gz, _, _ = se3d_support(u[0], u[1], u[2], a, b, c, 2.0, 2.0)
            R = support_R_eff_3d(u[0], u[1], u[2], a, b, c, 2.0, 2.0)
            s = gx ** 2 / a ** 4 + gy ** 2 / b ** 4 + gz ** 2 / c ** 4
            K = 1.0 / (a ** 2 * b ** 2 * c ** 2 * s ** 2)
            self.assertLess(abs(R - 1.0 / np.sqrt(K)) / R, 1e-5)

    def test_blocky_flat_face_has_a_huge_radius(self):
        """This is exactly why contact.curvature_R_cap stops being optional."""
        a = b = c = 40.0
        R = support_R_eff_3d(1.0, 0.0, 0.0, a, b, c, 8.0, 8.0)
        self.assertGreater(R, 1e3 * a)

    def test_radius_is_positive_everywhere_on_a_convex_body(self):
        for _ in range(60):
            a = RNG.uniform(15.0, 45.0)
            b = RNG.uniform(15.0, 45.0)
            c = RNG.uniform(15.0, 45.0)
            n1 = RNG.uniform(2.0, 6.0)
            n2 = RNG.uniform(2.0, 6.0)
            u = rand_unit(RNG, 3)
            self.assertGreater(support_R_eff_3d(u[0], u[1], u[2], a, b, c, n1, n2), 0.0)

    def test_beats_the_parametric_finite_difference_it_replaces(self):
        """The 0.01-radian scheme in superellipsoid_curvature_radii is ~1e-2; this
        must be orders of magnitude better on the case where truth is known."""
        from gels.engine import superellipsoid_curvature_radii
        a, b, c = 40.0, 30.0, 22.0
        errs_new, errs_old = [], []
        for _ in range(25):
            u = rand_unit(RNG, 3)
            h, gx, gy, gz, eta, om = se3d_support(u[0], u[1], u[2], a, b, c, 2.0, 2.0)
            s = gx ** 2 / a ** 4 + gy ** 2 / b ** 4 + gz ** 2 / c ** 4
            truth = a * b * c * s               # sqrt(1/K), K the Gaussian curvature
            errs_new.append(abs(support_R_eff_3d(u[0], u[1], u[2], a, b, c, 2.0, 2.0)
                                - truth) / truth)
            R1, R2 = superellipsoid_curvature_radii(eta, om, a, b, c, 2.0, 2.0)
            errs_old.append(abs(np.sqrt(R1 * R2) - truth) / truth)
        self.assertLess(np.median(errs_new), 1e-6)
        self.assertGreater(np.median(errs_old) / max(np.median(errs_new), 1e-15), 100.0)


if __name__ == '__main__':
    unittest.main()
