"""
Shape-aware packing (V3.2).
===========================

The lab's granules are irregular, roughly cuboidal fragments. Until V3.2 the
packer was entirely bounding-sphere, so turning shapes on made packing strictly
WORSE: an equal-volume AR-1.8 / n-3.5 fragment's bounding sphere claims 2.4x its
volume, and the measured 2D bed came out at phi 0.52 against 0.75 for spheres.

This checks the directional radius and its gradient against finite differences
(the settle is gradient descent on a potential only if they agree), that the
corrected bounding radius actually bounds, and that a shaped bed packs denser
than a sphere bed.

Run:  python -m unittest tests.test_shape_packing -v
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
    Params, bed_surface, generate_packing, granule_bound_radius, lambda_world,
    se2d_lambda_grad, se3d_lambda_grad, shape_bound_factor, shape_bound_radius,
    superellipse_area, superellipsoid_volume,
)


def _species(ar, n, r=20.0):
    return [dict(name='c', f=1.0, volume_fraction=1.0, color='#CC2222',
                 radius_mean=r, radius_std=0.1 * r, radius_min=0.6 * r,
                 density_kg_m3=1050.0,
                 aspect_ratio_mean=ar, aspect_ratio_std=0.4 if ar > 1 else 0.0,
                 blockiness_mean=n, blockiness_std=1.0 if n > 2 else 0.0)]


def _rot2(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def _body2(u, t):
    """World direction -> body frame (2D)."""
    return np.array([u[0] * np.cos(t) + u[1] * np.sin(t),
                     -u[0] * np.sin(t) + u[1] * np.cos(t)])


class TestDirectionalRadius(unittest.TestCase):

    def test_2d_point_lies_on_the_surface(self):
        rng = np.random.default_rng(0)
        a, b = 1.3, 0.7
        for n in (2.0, 3.5, 6.0):
            for _ in range(50):
                u = rng.normal(size=2)
                u /= np.linalg.norm(u)
                lam = se2d_lambda_grad(u[0], u[1], a, b, n)[0]
                q = lam * u
                self.assertAlmostEqual(abs(q[0] / a) ** n + abs(q[1] / b) ** n, 1.0, places=12)

    def test_3d_point_lies_on_the_surface_for_any_n1_n2(self):
        """Homogeneous of degree n2 even when n1 differs from n2, so the closed
        form needs no Newton fallback."""
        rng = np.random.default_rng(1)
        a, b, c = 1.3, 0.8, 1.05
        for n1, n2 in ((2.0, 2.0), (3.5, 3.5), (1.6, 5.0), (5.0, 1.6), (8.0, 2.0)):
            for _ in range(40):
                u = rng.normal(size=3)
                u /= np.linalg.norm(u)
                lam = se3d_lambda_grad(u[0], u[1], u[2], a, b, c, n1, n2)[0]
                q = lam * u
                val = ((abs(q[0] / a) ** n1 + abs(q[1] / b) ** n1) ** (n2 / n1)
                       + abs(q[2] / c) ** n2)
                self.assertAlmostEqual(val, 1.0, places=11, msg=f"n1={n1} n2={n2}")

    def test_gradients_match_finite_differences(self):
        rng = np.random.default_rng(2)
        h = 1e-6
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        a, b, c, n1, n2 = 1.3, 0.8, 1.05, 3.5, 2.2
        _lam, gx, gy, gz = se3d_lambda_grad(u[0], u[1], u[2], a, b, c, n1, n2)
        fd = []
        for k in range(3):
            up, um = u.copy(), u.copy()
            up[k] += h
            um[k] -= h
            fd.append((se3d_lambda_grad(up[0], up[1], up[2], a, b, c, n1, n2)[0]
                       - se3d_lambda_grad(um[0], um[1], um[2], a, b, c, n1, n2)[0]) / (2 * h))
        np.testing.assert_allclose([gx, gy, gz], fd, rtol=1e-6, atol=1e-8)

    def test_gradient_is_degree_minus_one_not_zero(self):
        """grad(lambda).u = -lambda, so the RADIAL part must be projected out.

        Keeping it gives a force about 2.5x too large that is not the gradient
        of any energy -- the easiest thing to get wrong in this formulation.
        """
        u = np.array([0.6, 0.8])
        a, b, n = 1.3, 0.7, 3.5
        lam, gx, gy = se2d_lambda_grad(u[0], u[1], a, b, n)
        self.assertAlmostEqual(gx * u[0] + gy * u[1], -lam, places=10)
        proj = (np.eye(2) - np.outer(u, u)) @ np.array([gx, gy])
        self.assertAlmostEqual(float(proj @ u), 0.0, places=12)

    def test_gap_gradient_matches_finite_differences(self):
        """The projected gradient IS d(gap)/dx, so the settle is gradient descent
        on an energy and cannot pump energy or limit-cycle."""
        a1, b1, n1 = 1.3, 0.7, 3.5
        a2, b2, n2 = 1.1, 0.9, 4.0
        t1, t2 = 0.3, -0.8
        xi = np.array([0.0, 0.0])
        xj = np.array([1.6, 0.4])

        def gap(xj_):
            dv = xj_ - xi
            d = float(np.linalg.norm(dv))
            u = dv / d
            bi = _body2(u, t1)
            bj = _body2(-u, t2)
            return (se2d_lambda_grad(bi[0], bi[1], a1, b1, n1)[0]
                    + se2d_lambda_grad(bj[0], bj[1], a2, b2, n2)[0] - d)

        h = 1e-6
        fd = np.array([(gap(xj + np.array([h, 0.0])) - gap(xj - np.array([h, 0.0]))) / (2 * h),
                       (gap(xj + np.array([0.0, h])) - gap(xj - np.array([0.0, h]))) / (2 * h)])
        dv = xj - xi
        d = float(np.linalg.norm(dv))
        u = dv / d
        bi = _body2(u, t1)
        bj = _body2(-u, t2)
        _l1, g1x, g1y = se2d_lambda_grad(bi[0], bi[1], a1, b1, n1)
        _l2, g2x, g2y = se2d_lambda_grad(bj[0], bj[1], a2, b2, n2)
        Gi = _rot2(t1) @ np.array([g1x, g1y])
        Gj = _rot2(t2) @ np.array([g2x, g2y])
        analytic = (np.eye(2) - np.outer(u, u)) @ (Gi - Gj) / d - u
        np.testing.assert_allclose(analytic, fd, rtol=1e-6, atol=1e-8)

    def test_torque_matches_finite_differences(self):
        """d(gap)/dtheta_i = -(u x G_i)_z, which is where the nesting comes from."""
        a1, b1, n1 = 1.3, 0.7, 3.5
        t1 = 0.3
        u = np.array([0.8, 0.6])

        def gap_of_theta(t):
            bi = _body2(u, t)
            return se2d_lambda_grad(bi[0], bi[1], a1, b1, n1)[0]

        h = 1e-6
        fd = (gap_of_theta(t1 + h) - gap_of_theta(t1 - h)) / (2 * h)
        bi = _body2(u, t1)
        _l, gx, gy = se2d_lambda_grad(bi[0], bi[1], a1, b1, n1)
        G = _rot2(t1) @ np.array([gx, gy])
        self.assertAlmostEqual(-(u[0] * G[1] - u[1] * G[0]), fd, places=7)

    def test_lambda_world_matches_the_leaf(self):
        a, b, n, th = 1.3, 0.8, 3.5, 0.7
        u = np.array([0.6, -0.8])
        got = lambda_world(np.array([a]), np.array([b]), None, np.array([n]), None,
                           np.array([th]), None, u[:1], u[1:], None, False)[0]
        bx, by = _body2(u, th)
        self.assertAlmostEqual(got, se2d_lambda_grad(bx, by, a, b, n)[0], places=14)


def _support_mtd_2d(a, b, n, th_i, th_j, dx, dy, m=4000):
    """Exact minimum translation distance for two identical superellipses.

    The body is diag(a, b) applied to the unit L^n ball, whose support function
    is the Hoelder dual norm, so for a centrally symmetric pair
        overlap  <=>  min over |nu| = 1 of [h_i(nu) + h_j(nu) - nu . r]  >  0
    and that minimum IS the penetration depth. Used as the reference the settle
    proxy is checked against.
    """
    mm = n / (n - 1.0)
    ang = np.linspace(0.0, 2.0 * np.pi, m, endpoint=False)
    nux, nuy = np.cos(ang), np.sin(ang)

    def h(th):
        c, s = np.cos(-th), np.sin(-th)
        bx = nux * c - nuy * s
        by = nux * s + nuy * c
        return (np.abs(a * bx) ** mm + np.abs(b * by) ** mm) ** (1.0 / mm)

    return float((h(th_i) + h(th_j) - (nux * dx + nuy * dy)).min())


class TestProxyTracksTheTruePenetration(unittest.TestCase):
    """The settle's directional-radius gap must be a faithful overlap.

    This is the guarantee the shape-aware packer rests on. Until V3.4 it was
    also what distinguished the settle from the DYNAMICS, which ran the
    common-normal solver -- that missed blocky contacts entirely and, when it
    did fire, returned an "overlap" larger than the granule (the KNOWN BLOCKER
    of the V3.2 changelog). The dynamics now use the support-function MTD
    solver, so the two agree on what an overlap is; this test still pins the
    settle's own proxy, which remains a separate, cheaper law.
    """

    def _case(self, n, frac):
        ar = 1.8
        sc = 20.0 * np.sqrt(np.pi / superellipse_area(ar, 1.0, n))
        a, b = ar * sc, sc
        th_i, th_j = 0.35, -0.9
        ux, uy = np.cos(0.6), np.sin(0.6)
        ci, si = np.cos(th_i), np.sin(th_i)
        cj, sj = np.cos(th_j), np.sin(th_j)
        li = se2d_lambda_grad(ux * ci + uy * si, -ux * si + uy * ci, a, b, n)[0]
        lj = se2d_lambda_grad(-ux * cj - uy * sj, ux * sj - uy * cj, a, b, n)[0]
        d = (li + lj) * frac
        proxy = (li + lj) - d
        true = _support_mtd_2d(a, b, n, th_i, th_j, d * ux, d * uy)
        return proxy, true

    def test_proxy_is_accurate_for_blocky_granules(self):
        for frac in (0.98, 0.95, 0.90):
            proxy, true = self._case(3.5, frac)
            self.assertGreater(true, 0.0)
            self.assertAlmostEqual(proxy / true, 1.0, delta=0.05,
                                   msg=f"proxy {proxy:.3f} vs true {true:.3f} at frac {frac}")

    def test_proxy_never_over_reports(self):
        """It is provably a lower bound on the true touching distance, so it can
        under-report an overlap but must never invent one."""
        for n in (2.0, 2.5, 3.5, 5.0):
            for frac in (1.02, 0.98, 0.90):
                proxy, true = self._case(n, frac)
                if proxy > 0:
                    self.assertGreater(true, -1e-9,
                                       f"proxy claims overlap where there is none (n={n})")

    def test_proxy_sign_is_right_at_contact(self):
        for n in (2.0, 3.5):
            self.assertLess(self._case(n, 1.02)[0], 0.0)     # separated
            self.assertGreater(self._case(n, 0.95)[0], 0.0)  # overlapping


class TestBoundingRadius(unittest.TestCase):

    def _extent(self, a, b, c, n, seed=3, m=3000):
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(m, 3))
        U /= np.linalg.norm(U, axis=1)[:, None]
        return max(se3d_lambda_grad(u[0], u[1], u[2], a, b, c, n, n)[0] for u in U)

    def _equal_volume_axes(self, ar, n):
        sc = ((4.0 / 3.0) * np.pi / superellipsoid_volume(ar, 1.0, 1.0, n, n)) ** (1.0 / 3.0)
        return ar * sc, sc, sc

    def test_it_bounds_and_is_tight(self):
        for ar in (1.0, 1.4, 1.8, 2.2):
            for n in (2.0, 3.5):
                a, b, c = self._equal_volume_axes(ar, n)
                true = self._extent(a, b, c, n)
                bnd = float(shape_bound_radius(a, b, c, n, n))
                self.assertGreaterEqual(bnd, true - 1e-9, f"does not bound (AR {ar}, n {n})")
                # `true` is a Monte-Carlo maximum over 3000 directions, so it
                # sits a hair below the real one; 0.5 % is the sampling error.
                self.assertLess(bnd / true, 1.005, f"not tight (AR {ar}, n {n})")

    def test_the_v31_bound_does_not_bound_blocky_granules(self):
        """Documents the bug: max(a,b,c) under-bounds by ~21 % at AR 1, n 3.5."""
        a, b, c = self._equal_volume_axes(1.0, 3.5)
        self.assertLess(max(a, b, c), 0.85 * self._extent(a, b, c, 3.5))

    def test_legacy_branch_is_exactly_max_abc(self):
        a, b, c, n = np.array([1.3]), np.array([0.8]), np.array([1.05]), np.array([3.5])
        np.testing.assert_array_equal(granule_bound_radius(a, b, c, n, n, True, False),
                                      np.maximum(np.maximum(a, b), c))
        np.testing.assert_array_equal(granule_bound_radius(a, b, None, n, None, False, False),
                                      np.maximum(a, b))

    def test_sphere_is_unchanged(self):
        one, two = np.array([1.0]), np.array([2.0])
        got = np.asarray(shape_bound_radius(one, one, one, two, two)).ravel()[0]
        self.assertAlmostEqual(float(got), 1.0, places=12)


class TestRSADeflation(unittest.TestCase):

    def test_bound_factor_is_one_when_the_shape_path_is_off(self):
        p = Params(species=copy.deepcopy(_species(1.8, 3.5)))
        self.assertEqual(shape_bound_factor(p, '2D'), 1.0)

    def test_bound_factor_reflects_the_reserved_volume(self):
        p = Params(species=copy.deepcopy(_species(1.8, 3.5)),
                   shape_enabled=True, packing_shape_contact=True)
        self.assertGreater(shape_bound_factor(p, '3D'), 2.0)   # measured 2.39
        self.assertGreater(shape_bound_factor(p, '2D'), 1.5)


class TestPackedDensity(unittest.TestCase):
    """The goal: a shaped bed must pack DENSER than a sphere bed, not looser."""

    def _bed(self, shape, flag, seed=5):
        p = Params(mode='2D', Lx=700.0, Ly=900.0, boundary_shape='box', boundary_top='free',
                   bed_height=450.0, bed_phi_assumed=0.75, gravity_enabled=True,
                   granule_density=1050.0, packing_consolidation='auto',
                   packing_settle_steps=200, save_data=False, Ngrid=100,
                   species=copy.deepcopy(_species(1.8 if shape else 1.0,
                                                  3.5 if shape else 2.0)),
                   shape_enabled=shape, packing_shape_contact=flag,
                   contact_shape_dynamics=flag, curvature_R_cap=2.0 if flag else 0.0,
                   cell_surface_coverage=0.0, n_cells_per_granule=0)
        gs = generate_packing(p, seed=seed)
        area = float(np.sum([superellipse_area(gs.a[i], gs.b[i], gs.n1[i])
                             for i in range(gs.N)]))
        env = bed_surface(gs, p)['bed_envelope_volume']
        return gs.N, (area / env if env > 0 else 0.0)

    def test_shape_aware_packer_beats_spheres(self):
        _ns, phi_s = self._bed(False, False)
        _nk, phi_k = self._bed(True, True)
        self.assertGreater(phi_k, phi_s,
                           "the shape-aware packer must beat the sphere solid fraction")

    def test_blind_packer_is_much_worse_than_spheres(self):
        """Why the two halves ship behind ONE flag: the corrected bounding radius
        without the directional overlap lowers the achievable density."""
        _ns, phi_s = self._bed(False, False)
        _nb, phi_blind = self._bed(True, False)
        self.assertLess(phi_blind, 0.85 * phi_s)

    def test_shaped_bed_places_every_granule(self):
        """The RSA deflation must account for the bounding sphere a blocky
        granule reserves, or placement runs past saturation and granules vanish."""
        nk, _phi = self._bed(True, True)
        ns, _phi_s = self._bed(False, False)
        self.assertEqual(nk, ns)


if __name__ == '__main__':
    unittest.main()
