"""
Support-function MTD contact solver (V3.4 Phase 3.2).

The solver under test is `se3d_mtd_core` / `se2d_mtd_core` in `gels.engine`.
It is checked against three *independent* oracles rather than against the
derivation that produced it:

1. **A surface-sampling overlap oracle.** Densely mesh one body's boundary,
   transform into the other's body frame and evaluate the implicit function.
   This knows nothing about support functions. Bisecting the translation along
   the reported normal gives a ground-truth minimum translation distance.
2. **Exhaustive maximisation of `sep`** over a 20k-direction Fibonacci sphere.
   This tests the *ascent* -- the part that can stall on a blocky ridge --
   rather than the support function, which `test_support_function` already pins.
3. **The conservativity identity** `d(delta)/d(c2) = -n*`, which follows from the
   envelope theorem and is precisely the property the common-normal solver never
   had. It is finite-differenced here against the solver's own output.

The last class of tests re-measures the common-normal solver the MTD path
replaces, so the 12 %/9.8x claim in the changelog is reproducible from the suite.

Run:  python -m unittest tests.test_mtd_contact -v
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
    MTD_GPERP_RTOL, mtd_contact_2d, mtd_contact_3d,
    find_contact_spheres_3d, find_contact_superellipsoids_3d,
    quat_random, quat_rotate, quat_rotate_inv,
    se2d_mtd_core, se3d_mtd_core, se2d_support, se3d_support,
    superellipse_implicit, superellipsoid_implicit,
    _qrot_s, _qrot_inv_s,
)

RNG = np.random.default_rng(20260919)
IDQ = np.array([1.0, 0.0, 0.0, 0.0])


class SeededCase(unittest.TestCase):
    """Every case reseeds the module RNG from its own name.

    Without this the draws -- and therefore the shapes a tolerance was measured
    against -- depend on which subset of the file is being run, so a bound tuned
    on a full run can fail when the class is run alone.
    """

    def setUp(self):
        global RNG
        # zlib.crc32, not hash(): str hashing is salted per process unless
        # PYTHONHASHSEED is pinned, which would make the draws vary run to run
        # class+method, NOT self.id(): the module is `tests.test_mtd_contact`
        # under `-m unittest` and `test_mtd_contact` under `discover`, so the
        # full id would give a different stream depending on how it is run
        key = f"{type(self).__name__}.{self._testMethodName}"
        RNG = np.random.default_rng(zlib.crc32(key.encode()))


# ── independent oracles ──────────────────────────────────────────────

def surface_mesh_3d(a, b, c, n1, n2, ne=200, no=400):
    """Dense body-frame boundary mesh of a superellipsoid."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, ne)
    om = np.linspace(-np.pi, np.pi, no, endpoint=False)
    E, O = np.meshgrid(eta, om, indexing='ij')
    e2, e1 = 2.0 / n2, 2.0 / n1

    def sp(v, e):
        return np.sign(v) * np.abs(v) ** e

    X = a * sp(np.cos(E), e2) * sp(np.cos(O), e1)
    Y = b * sp(np.cos(E), e2) * sp(np.sin(O), e1)
    Z = c * sp(np.sin(E), e2)
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)


def min_implicit_of_i_in_j(ci, qi, mesh_i, cj, qj, aj, bj, cj_, n1j, n2j):
    """min of body j's implicit function over body i's sampled boundary.

    < 1 means a point of ``partial B_i`` is strictly inside ``B_j``, i.e. the two
    convex bodies overlap. Uses nothing from the support machinery.
    """
    w = np.einsum('ij,kj->ki', rot_matrix(qi), mesh_i) + ci
    v = np.einsum('ij,kj->ki', rot_matrix(qj).T, w - cj)
    F = (np.abs(v[:, 0] / aj) ** n1j + np.abs(v[:, 1] / bj) ** n1j) ** (n2j / n1j) \
        + np.abs(v[:, 2] / cj_) ** n2j
    return float(F.min())


def rot_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def overlaps_3d(ci, qi, shp_i, cj, qj, shp_j, mesh_i=None):
    ai, bi, ci_, n1i, n2i = shp_i
    aj, bj, cj_, n1j, n2j = shp_j
    if mesh_i is None:
        mesh_i = surface_mesh_3d(ai, bi, ci_, n1i, n2i)
    return min_implicit_of_i_in_j(ci, qi, mesh_i, cj, qj, aj, bj, cj_, n1j, n2j) < 1.0


def mtd_by_bisection(ci, qi, shp_i, cj, qj, shp_j, nrm, r_scale):
    """Ground-truth MTD: the smallest t with B_i and B_j + t*nrm disjoint.

    Bisection on the surface-sampling overlap oracle. Independent of the support
    function, of `sep`, and of the ascent.
    """
    mesh_i = surface_mesh_3d(*shp_i)
    lo, hi = 0.0, 2.0 * r_scale
    for _ in range(46):
        mid = 0.5 * (lo + hi)
        if overlaps_3d(ci, qi, shp_i, cj + mid * nrm, qj, shp_j, mesh_i):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fib_sphere(m):
    k = np.arange(m) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / m)
    th = np.pi * (1.0 + 5.0 ** 0.5) * k
    return np.stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)], 1)


def sep_grid_max(ci, qi, shp_i, cj, qj, shp_j, m=20000):
    """Exhaustive maximisation of sep over a Fibonacci sphere."""
    ai, bi, ci_, n1i, n2i = shp_i
    aj, bj, cj_, n1j, n2j = shp_j
    dc = cj - ci
    N = fib_sphere(m)
    Ri, Rj = rot_matrix(qi), rot_matrix(qj)
    Bi = N @ Ri
    Bj = N @ Rj

    def hh(B, a, b, c, n1, n2):
        q1 = n1 / (n1 - 1.0)
        q2 = n2 / (n2 - 1.0)
        u = np.abs(a * B[:, 0]); v = np.abs(b * B[:, 1]); w = np.abs(c * B[:, 2])
        mm = np.maximum(u, v)
        P = np.where(mm > 0, mm * ((u / np.maximum(mm, 1e-300)) ** q1
                                   + (v / np.maximum(mm, 1e-300)) ** q1) ** (1 / q1), 0.0)
        m2 = np.maximum(P, w)
        return np.where(m2 > 0, m2 * ((P / np.maximum(m2, 1e-300)) ** q2
                                      + (w / np.maximum(m2, 1e-300)) ** q2) ** (1 / q2), 0.0)

    s = N @ dc - hh(Bi, ai, bi, ci_, n1i, n2i) - hh(Bj, aj, bj, cj_, n1j, n2j)
    k = int(np.argmax(s))
    return float(s[k]), N[k]


def sep_max_polished(ci, qi, shp_i, cj, qj, shp_j, m=20000):
    """Reference max of `sep`: a Fibonacci grid, then Nelder-Mead in spherical
    coordinates. The bare grid has ~1e-3 relative discretisation error near a
    quadratic maximum, which is coarser than the solver being tested, so the
    polish is what makes this a reference rather than a competitor."""
    from scipy.optimize import minimize
    best, n0 = sep_grid_max(ci, qi, shp_i, cj, qj, shp_j, m)
    dc = cj - ci
    Ri, Rj = rot_matrix(qi), rot_matrix(qj)

    def f(pq):
        th, ph = pq
        n = np.array([np.cos(ph) * np.sin(th), np.sin(ph) * np.sin(th), np.cos(th)])
        hi = se3d_support(*(Ri.T @ n), *shp_i)[0]
        hj = se3d_support(*(Rj.T @ n), *shp_j)[0]
        return -(dc @ n - hi - hj)

    r = minimize(f, [np.arccos(np.clip(n0[2], -1, 1)), np.arctan2(n0[1], n0[0])],
                 method='Nelder-Mead',
                 options=dict(xatol=1e-13, fatol=1e-15, maxiter=4000))
    return max(best, float(-r.fun))


def rand_unit(rng, dim=3):
    v = rng.normal(size=dim)
    return v / np.linalg.norm(v)


def touch_distance(ci, qi, shp_i, qj, shp_j, u, r_scale):
    """Centre distance at which the pair, offset along u, exactly touches."""
    mesh_i = surface_mesh_3d(*shp_i)
    lo, hi = 0.0, 4.0 * r_scale
    for _ in range(46):
        mid = 0.5 * (lo + hi)
        if overlaps_3d(ci, qi, shp_i, ci + mid * u, qj, shp_j, mesh_i):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ── anchors ──────────────────────────────────────────────────────────

class TestSphereAnchor(SeededCase):

    def test_matches_the_analytic_sphere_solver(self):
        for _ in range(60):
            ri = RNG.uniform(15.0, 45.0)
            rj = RNG.uniform(15.0, 45.0)
            u = rand_unit(RNG)
            d = (ri + rj) * RNG.uniform(0.75, 0.999)
            cj = d * u
            ref = find_contact_spheres_3d(0.0, 0.0, 0.0, ri, cj[0], cj[1], cj[2], rj)
            out = se3d_mtd_core(0.0, 0.0, 0.0, ri, ri, ri, 2.0, 2.0, IDQ,
                                cj[0], cj[1], cj[2], rj, rj, rj, 2.0, 2.0, IDQ)
            self.assertTrue(out[0])
            self.assertAlmostEqual(out[1], ref[1], places=9, msg="overlap")
            np.testing.assert_allclose(out[2:5], ref[2:5], atol=1e-10)
            np.testing.assert_allclose(out[5:8], ref[5:8], atol=1e-8)
            self.assertLess(abs(out[8] - ref[8]) / ref[8], 1e-7, "R_eff")

    def test_separated_spheres_are_rejected_by_the_certificate(self):
        for _ in range(60):
            ri, rj = 30.0, 25.0
            u = rand_unit(RNG)
            cj = (ri + rj) * RNG.uniform(1.001, 3.0) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, ri, ri, ri, 2.0, 2.0, IDQ,
                                cj[0], cj[1], cj[2], rj, rj, rj, 2.0, 2.0, IDQ)
            self.assertFalse(out[0])
            self.assertIsNone(mtd_contact_3d(0.0, 0.0, 0.0, ri, ri, ri, 2.0, 2.0, IDQ,
                                             cj[0], cj[1], cj[2], rj, rj, rj, 2.0, 2.0, IDQ))

    def test_2d_circles_are_exact(self):
        for _ in range(60):
            ri = RNG.uniform(15.0, 45.0)
            rj = RNG.uniform(15.0, 45.0)
            u = rand_unit(RNG, 2)
            d = (ri + rj) * RNG.uniform(0.75, 0.999)
            cj = d * u
            out = se2d_mtd_core(0.0, 0.0, ri, ri, 2.0, 0.0,
                                cj[0], cj[1], rj, rj, 2.0, 0.0)
            self.assertTrue(out[0])
            self.assertAlmostEqual(out[1], ri + rj - d, places=9)
            np.testing.assert_allclose(out[2:4], u, atol=1e-10)
            np.testing.assert_allclose(out[4:6], (ri - out[1] / 2) * u, atol=1e-8)
            self.assertAlmostEqual(out[6], ri, places=4)
            self.assertAlmostEqual(out[7], rj, places=4)


# ── the ascent, against exhaustive search ────────────────────────────

class TestAscentFindsTheGlobalMaximum(SeededCase):

    def _case(self, shp_i, shp_j, med_tol, p90_tol, n_trials=40):
        """Gates are the MEDIAN and the p90, not the max.

        The two measure different things. The median pins the ascent's
        convergence and is at rounding for smooth bodies. The p90 pins the
        seeding tail: `sep` is concave on R^3, but the unit sphere is not a
        convex constraint set and in penetration `sep` is negative everywhere on
        it, so the sphere-restricted problem genuinely admits local maxima and a
        minority of tumbled near-polyhedral pairs ascend into one. That tail is
        heavy, so a max-gate would be a coin flip; it is bounded instead by
        `test_no_contact_is_ever_lost`, which is the property that matters.
        All four numbers are measured over 120 pairs per shape class.
        """
        rels = []
        for _ in range(n_trials):
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            r = max(max(shp_i[:3]), max(shp_j[:3]))
            cj = 2.0 * r * RNG.uniform(0.70, 0.95) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            s_ref = sep_max_polished(np.zeros(3), qi, shp_i, cj, qj, shp_j, m=4000)
            if not out[0]:
                self.assertGreater(s_ref, -1e-6, "solver rejected an overlapping pair")
                continue
            rels.append(abs(-out[1] - s_ref) / max(abs(s_ref), 1e-12))
        self.assertGreaterEqual(len(rels), 10, "too few overlaps to judge")
        self.assertLess(float(np.median(rels)), med_tol)
        self.assertLess(float(np.percentile(rels, 90)), p90_tol)

    def test_ellipsoids(self):
        """Smooth and strictly convex: the ascent is exact to rounding."""
        self._case((40.0, 30.0, 22.0, 2.0, 2.0), (35.0, 35.0, 25.0, 2.0, 2.0),
                   med_tol=1e-9, p90_tol=1e-7)

    def test_realistic_blockiness(self):
        """n ~ 2.6-3.5 is what the presets actually run."""
        self._case((40.0, 30.0, 25.0, 3.0, 3.0), (30.0, 34.0, 30.0, 2.6, 3.4),
                   med_tol=1e-8, p90_tol=1e-5)

    def test_blocky_equal_exponents(self):
        self._case((40.0, 30.0, 25.0, 4.0, 4.0), (30.0, 30.0, 30.0, 4.0, 4.0),
                   med_tol=1e-6, p90_tol=3e-3)

    def test_blocky_unequal_exponents(self):
        self._case((40.0, 26.0, 33.0, 5.0, 2.6), (32.0, 38.0, 24.0, 2.4, 6.0),
                   med_tol=1e-6, p90_tol=1e-2)

    def test_near_cube_degrades_gracefully(self):
        """Documented limitation, not a regression: at n = 10 the support
        function is near-piecewise-linear and the maximum sits at a kink."""
        self._case((35.0, 35.0, 35.0, 10.0, 10.0), (35.0, 35.0, 35.0, 10.0, 10.0),
                   med_tol=1e-3, p90_tol=5e-2)

    def test_no_contact_is_ever_lost(self):
        """The bound that matters operationally. Truncating the ascent can only
        UNDER-estimate sep -- `sep(n) <= sep(n*)` for every n -- so delta comes
        out too large, never too small, and an overlapping pair can never be
        reported as apart. A stiff contact is recoverable; a missed one is not."""
        missed = 0
        total = 0
        for shp in ((40.0, 30.0, 25.0, 3.0, 3.0), (35.0, 35.0, 35.0, 4.0, 4.0),
                    (35.0, 35.0, 35.0, 10.0, 10.0)):
            for _ in range(40):
                qi = quat_random(RNG)
                qj = quat_random(RNG)
                u = rand_unit(RNG)
                cj = 2.0 * max(shp[:3]) * RNG.uniform(0.70, 0.99) * u
                out = se3d_mtd_core(0.0, 0.0, 0.0, *shp, qi, cj[0], cj[1], cj[2], *shp, qj)
                s_ref = sep_max_polished(np.zeros(3), qi, shp, cj, qj, shp, m=4000)
                if s_ref >= -1e-9:
                    continue
                total += 1
                if not out[0]:
                    missed += 1
                else:
                    self.assertGreaterEqual(out[1], -s_ref - 1e-6,
                                            "delta under-estimated; truncation must be stiff")
        self.assertGreater(total, 30)
        self.assertEqual(missed, 0, f"{missed}/{total} true contacts dropped")


# ── delta means what it says ─────────────────────────────────────────

class TestPenetrationIsAMinimumTranslation(SeededCase):

    def test_translating_by_delta_along_n_separates_the_pair(self):
        """The defining property, checked with a surface-sampling oracle that
        knows nothing about support functions."""
        errs = []
        for _ in range(30):
            shp_i = (40.0, 30.0, 25.0, RNG.uniform(2.0, 5.0), RNG.uniform(2.0, 5.0))
            shp_j = (33.0, 36.0, 28.0, RNG.uniform(2.0, 5.0), RNG.uniform(2.0, 5.0))
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.72, 0.88) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            if not out[0]:
                continue
            nrm = np.array(out[2:5])
            truth = mtd_by_bisection(np.zeros(3), qi, shp_i, cj, qj, shp_j, nrm, 45.0)
            errs.append(abs(out[1] - truth) / max(truth, 1e-9))
        self.assertGreaterEqual(len(errs), 15, "too few overlapping samples to judge")
        # the oracle's own resolution is ~1e-4 relative (its overlap test is a
        # 200x400 surface mesh), so the median here measures the ORACLE, not the
        # solver; the p90 is the honest bound on the pair
        self.assertLess(float(np.median(errs)), 1e-3)
        self.assertLess(float(np.percentile(errs, 90)), 1e-2)

    def test_contact_point_lies_inside_both_bodies(self):
        for _ in range(40):
            shp_i = (40.0, 30.0, 25.0, 3.0, 3.5)
            shp_j = (33.0, 36.0, 28.0, 2.5, 4.0)
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.70, 0.90) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            if not out[0]:
                continue
            cp = np.array(out[5:8])
            vi = quat_rotate_inv(qi, cp)
            vj = quat_rotate_inv(qj, cp - cj)
            Fi = superellipsoid_implicit(vi[0], vi[1], vi[2], *shp_i)
            Fj = superellipsoid_implicit(vj[0], vj[1], vj[2], *shp_j)
            # inside both, up to the truncation bias (delta is over-estimated,
            # so the midpoint sits at worst marginally outside)
            self.assertLess(Fi, 1.05)
            self.assertLess(Fj, 1.05)

    def test_normal_points_from_i_toward_j(self):
        for _ in range(40):
            shp = (40.0, 30.0, 25.0, 3.0, 3.0)
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.70, 0.92) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp, qi, cj[0], cj[1], cj[2], *shp, qj)
            if not out[0]:
                continue
            self.assertGreater(float(np.dot(np.array(out[2:5]), u)), 0.0)

    def _residuals(self, n_lo, n_hi, n_trials=60):
        out_r = []
        for _ in range(n_trials):
            shp_i = (40.0, 30.0, 25.0, RNG.uniform(n_lo, n_hi), RNG.uniform(n_lo, n_hi))
            shp_j = (33.0, 36.0, 28.0, RNG.uniform(n_lo, n_hi), RNG.uniform(n_lo, n_hi))
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.70, 0.92) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            if out[0]:
                out_r.append(out[9] / (max(shp_i[:3]) + max(shp_j[:3])))
        self.assertGreaterEqual(len(out_r), 20)
        return np.array(out_r)

    def test_residual_certifies_convergence_in_the_realistic_range(self):
        """|grad sep perp n| -> 0 at a smooth maximum, so it is the ascent's own
        convergence certificate over the blockiness GELS actually runs."""
        r = self._residuals(2.0, 4.0, n_trials=120)
        self.assertLess(float(np.median(r)), 1e-4)
        self.assertLess(float(np.percentile(r, 90)), 5e-3)

    def test_residual_stops_being_a_certificate_at_high_blockiness(self):
        """Recorded so the diagnostic is not misread. Above n ~ 4 the support
        function is near-piecewise-linear and the maximum of `sep` sits at a KINK,
        where grad sep jumps rather than vanishing. The ascent then terminates on
        the step floor with a finite tangent gradient -- which is correct
        behaviour, not a failure, and is why the accuracy gate is the polished
        reference in TestAscentFindsTheGlobalMaximum rather than this number."""
        lo = self._residuals(2.0, 3.0)
        hi = self._residuals(6.0, 8.0)
        self.assertGreater(float(np.median(hi)), 100.0 * float(np.median(lo)))


# ── the property the old solver never had ────────────────────────────

class TestConservativeByConstruction(SeededCase):

    def test_d_delta_d_c2_is_minus_the_normal(self):
        """The envelope theorem: `sep` depends on c2 only through the linear term,
        so d(delta)/d(c2) = -n* exactly. This is what makes F = k delta^(3/2) n
        the gradient of an energy -- the common-normal solver satisfies nothing
        like it, which is why it generates spurious torque."""
        eps = 1e-5
        errs = []
        for _ in range(40):
            shp_i = (40.0, 30.0, 25.0, RNG.uniform(2.0, 4.0), RNG.uniform(2.0, 4.0))
            shp_j = (33.0, 36.0, 28.0, RNG.uniform(2.0, 4.0), RNG.uniform(2.0, 4.0))
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.72, 0.90) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            if not out[0]:
                continue
            nrm = np.array(out[2:5])
            grad = np.zeros(3)
            for k in range(3):
                for s in (+1, -1):
                    c = cj.copy()
                    c[k] += s * eps
                    o = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, c[0], c[1], c[2], *shp_j, qj)
                    grad[k] += s * o[1]
                grad[k] /= (2 * eps)
            errs.append(float(np.linalg.norm(grad + nrm)))
        self.assertGreaterEqual(len(errs), 20)
        # |n*| = 1, so this IS the relative error of the identity
        self.assertLess(float(np.median(errs)), 1e-4, "d(delta)/d(c2) != -n*")
        self.assertLess(float(np.percentile(errs, 90)), 1e-2)

    def test_overlap_is_continuous_through_first_touch(self):
        """delta must go to zero smoothly as the pair separates. The
        common-normal solver jumps, because its `delta` is a distance between
        surface points that stays finite at first touch."""
        shp = (40.0, 28.0, 33.0, 3.5, 2.8)
        qi = quat_random(RNG)
        qj = quat_random(RNG)
        u = rand_unit(RNG)
        d_touch = touch_distance(np.zeros(3), qi, shp, qj, shp, u, 45.0)
        prev = None
        for f in (0.999, 0.9995, 0.9999, 0.99999):
            cj = d_touch * f * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp, qi, cj[0], cj[1], cj[2], *shp, qj)
            if not out[0]:
                continue
            self.assertLess(out[1], 0.02 * d_touch,
                            "delta must vanish at first touch, not jump")
            if prev is not None:
                self.assertLessEqual(out[1], prev + 1e-9, "delta must be monotone")
            prev = out[1]

    def test_monotone_in_approach(self):
        shp_i = (40.0, 30.0, 25.0, 3.0, 4.0)
        shp_j = (32.0, 36.0, 28.0, 4.0, 3.0)
        qi = quat_random(RNG)
        qj = quat_random(RNG)
        u = rand_unit(RNG)
        last = -1.0
        for d in np.linspace(0.95, 0.65, 25) * 2.0 * 40.0:
            cj = d * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            cur = out[1] if out[0] else 0.0
            self.assertGreaterEqual(cur, last - 1e-9, f"delta decreased at d={d:.2f}")
            last = cur


# ── 2D ───────────────────────────────────────────────────────────────

class TestMTD2D(SeededCase):

    @staticmethod
    def _sep_grid_2d(ci, ti, shp_i, cj, tj, shp_j, m=20000):
        th = np.arange(m) * (2 * np.pi / m)
        N = np.stack([np.cos(th), np.sin(th)], 1)

        def h(N, t, a, b, n):
            q = n / (n - 1.0)
            R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            B = N @ R.T
            u = np.abs(a * B[:, 0]); v = np.abs(b * B[:, 1])
            mm = np.maximum(np.maximum(u, v), 1e-300)
            return mm * ((u / mm) ** q + (v / mm) ** q) ** (1 / q)

        s = N @ (cj - ci) - h(N, ti, *shp_i) - h(N, tj, *shp_j)
        k = int(np.argmax(s))
        return float(s[k]), N[k]

    def test_ascent_matches_exhaustive_search(self):
        worst = 0.0
        for _ in range(40):
            shp_i = (40.0, 26.0, RNG.uniform(2.0, 8.0))
            shp_j = (33.0, 35.0, RNG.uniform(2.0, 8.0))
            ti = RNG.uniform(0, 2 * np.pi)
            tj = RNG.uniform(0, 2 * np.pi)
            u = rand_unit(RNG, 2)
            cj = 2.0 * 40.0 * RNG.uniform(0.65, 0.95) * u
            out = se2d_mtd_core(0.0, 0.0, *shp_i, ti, cj[0], cj[1], *shp_j, tj)
            s_grid, _ = self._sep_grid_2d(np.zeros(2), ti, shp_i, cj, tj, shp_j)
            if not out[0]:
                self.assertGreater(s_grid, -1e-6)
                continue
            self.assertGreaterEqual(-out[1], s_grid - 1e-9)
            worst = max(worst, abs(-out[1] - s_grid) / max(abs(s_grid), 1e-12))
        self.assertLess(worst, 1e-3)

    def test_contact_point_is_inside_both(self):
        for _ in range(40):
            shp_i = (40.0, 26.0, 3.5)
            shp_j = (33.0, 35.0, 2.8)
            ti = RNG.uniform(0, 2 * np.pi)
            tj = RNG.uniform(0, 2 * np.pi)
            u = rand_unit(RNG, 2)
            cj = 2.0 * 40.0 * RNG.uniform(0.65, 0.92) * u
            out = se2d_mtd_core(0.0, 0.0, *shp_i, ti, cj[0], cj[1], *shp_j, tj)
            if not out[0]:
                continue
            cx, cy = out[4], out[5]
            self.assertLess(superellipse_implicit(cx, cy, 0.0, 0.0, *shp_i, ti), 1.05)
            self.assertLess(superellipse_implicit(cx, cy, cj[0], cj[1], *shp_j, tj), 1.05)

    def test_d_delta_d_c2_is_minus_the_normal(self):
        eps = 1e-5
        worst = 0.0
        for _ in range(30):
            shp_i = (40.0, 26.0, RNG.uniform(2.0, 6.0))
            shp_j = (33.0, 35.0, RNG.uniform(2.0, 6.0))
            ti = RNG.uniform(0, 2 * np.pi)
            tj = RNG.uniform(0, 2 * np.pi)
            u = rand_unit(RNG, 2)
            cj = 2.0 * 40.0 * RNG.uniform(0.68, 0.90) * u
            out = se2d_mtd_core(0.0, 0.0, *shp_i, ti, cj[0], cj[1], *shp_j, tj)
            if not out[0]:
                continue
            nrm = np.array(out[2:4])
            grad = np.zeros(2)
            for k in range(2):
                for s in (+1, -1):
                    c = cj.copy()
                    c[k] += s * eps
                    o = se2d_mtd_core(0.0, 0.0, *shp_i, ti, c[0], c[1], *shp_j, tj)
                    grad[k] += s * o[1]
                grad[k] /= (2 * eps)
            worst = max(worst, float(np.linalg.norm(grad + nrm)))
        self.assertLess(worst, 2e-3)

    def test_optional_wrapper_contract(self):
        apart = mtd_contact_2d(0.0, 0.0, 40.0, 40.0, 2.0, 0.0, 200.0, 0.0, 40.0, 40.0, 2.0, 0.0)
        self.assertIsNone(apart)
        hit = mtd_contact_2d(0.0, 0.0, 40.0, 40.0, 2.0, 0.0, 70.0, 0.0, 40.0, 40.0, 2.0, 0.0)
        self.assertEqual(len(hit), 8)
        self.assertTrue(hit[0])
        self.assertAlmostEqual(hit[1], 10.0, places=9)


# ── invariances ──────────────────────────────────────────────────────

class TestInvariance(SeededCase):

    def test_scalar_quaternion_rotation_matches_the_array_form(self):
        worst = 0.0
        for _ in range(200):
            q = quat_random(RNG)
            v = RNG.normal(size=3)
            a = np.array(_qrot_s(q[0], q[1], q[2], q[3], v[0], v[1], v[2]))
            worst = max(worst, float(np.abs(a - quat_rotate(q, v)).max()))
            a = np.array(_qrot_inv_s(q[0], q[1], q[2], q[3], v[0], v[1], v[2]))
            worst = max(worst, float(np.abs(a - quat_rotate_inv(q, v)).max()))
        self.assertLess(worst, 1e-12)

    def test_swapping_i_and_j_flips_the_normal_and_keeps_delta(self):
        """delta is a property of the PAIR, so it must not depend on which body
        is called i. The seeds are tried in i-then-j order, so at high blockiness
        a swap can select a different one; the agreement is therefore the
        solver's accuracy, and is gated on the distribution, not per sample."""
        dd, dn = [], []
        for _ in range(120):
            shp_i = (40.0, 30.0, 25.0, 3.0, 4.0)
            shp_j = (32.0, 36.0, 28.0, 4.0, 3.0)
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            u = rand_unit(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.70, 0.90) * u
            f = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            b = se3d_mtd_core(cj[0], cj[1], cj[2], *shp_j, qj, 0.0, 0.0, 0.0, *shp_i, qi)
            self.assertEqual(f[0], b[0], "the two orders must agree a pair is in contact")
            if not f[0]:
                continue
            dd.append(abs(f[1] - b[1]) / f[1])
            dn.append(float(np.abs(np.array(f[2:5]) + np.array(b[2:5])).max()))
            # the contact POINT is a pair property too, and is well conditioned
            np.testing.assert_allclose(np.array(f[5:8]), np.array(b[5:8]),
                                       atol=5e-2 * max(shp_i[:3]))
        self.assertGreaterEqual(len(dd), 60)
        self.assertLess(float(np.median(dd)), 1e-8)
        self.assertLess(float(np.percentile(dd, 90)), 1e-3)
        # the NORMAL is the ill-conditioned output near a flat face -- many
        # directions there give almost the same sep -- so it is held looser than
        # delta, which is what the force law actually uses
        self.assertLess(float(np.median(dn)), 1e-5)
        self.assertLess(float(np.percentile(dn, 90)), 1e-2)

    def test_rigid_rotation_of_the_whole_pair(self):
        """delta is invariant and the normal corotates."""
        from gels.engine import quat_multiply
        shp_i = (40.0, 30.0, 25.0, 3.5, 2.5)
        shp_j = (32.0, 36.0, 28.0, 2.5, 4.5)
        for _ in range(20):
            qi = quat_random(RNG)
            qj = quat_random(RNG)
            cj = 2.0 * 40.0 * RNG.uniform(0.70, 0.90) * rand_unit(RNG)
            g = quat_random(RNG)
            Rg = rot_matrix(g)
            a = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            cj2 = Rg @ cj
            b = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, quat_multiply(g, qi),
                              cj2[0], cj2[1], cj2[2], *shp_j, quat_multiply(g, qj))
            self.assertEqual(a[0], b[0])
            if not a[0]:
                continue
            self.assertLess(abs(a[1] - b[1]) / a[1], 1e-4)
            np.testing.assert_allclose(Rg @ np.array(a[2:5]), np.array(b[2:5]), atol=1e-3)


# ── what it replaces ─────────────────────────────────────────────────

class TestAgainstTheCommonNormalSolver(SeededCase):
    """Re-measures the solver the MTD path replaces, so the changelog's
    12 %/9.8x numbers are reproducible from the suite rather than asserted."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(4242)
        shp = (40.0, 28.0, 33.0, 3.5, 3.0)
        cls.n_true = 0
        cls.n_old = 0
        cls.ratios = []
        for _ in range(60):
            qi = quat_random(rng)
            qj = quat_random(rng)
            u = rand_unit(rng)
            d_touch = touch_distance(np.zeros(3), qi, shp, qj, shp, u, 45.0)
            cj = 0.98 * d_touch * u           # 2 % past first touch
            mtd = se3d_mtd_core(0.0, 0.0, 0.0, *shp, qi, cj[0], cj[1], cj[2], *shp, qj)
            if not mtd[0]:
                continue
            cls.n_true += 1
            old = find_contact_superellipsoids_3d(0.0, 0.0, 0.0, *shp, qi,
                                                  cj[0], cj[1], cj[2], *shp, qj)
            if old is not None:
                cls.n_old += 1
                cls.ratios.append(old[1] / mtd[1])

    def test_mtd_finds_every_true_contact(self):
        self.assertGreaterEqual(self.n_true, 55, "ground truth is degenerate")

    def test_common_normal_misses_most_of_them(self):
        rate = self.n_old / self.n_true
        self.assertLess(rate, 0.5,
                        f"common-normal detection {rate:.0%}; the KNOWN BLOCKER should "
                        f"still be present on the legacy path")

    def test_common_normal_over_reports_penetration_by_an_order_of_magnitude(self):
        if not self.ratios:
            self.skipTest("no shared detections to compare")
        self.assertGreater(np.median(self.ratios), 3.0)

    def test_common_normal_reports_the_normal_backwards_about_half_the_time(self):
        """The most consequential of the legacy solver's defects, and the one
        that was not previously on record.

        It reports `(p_j - p_i)/|.|`, where p_i is body i's common-normal surface
        point. Once the bodies overlap enough for those points to cross, that
        vector points from j back toward i. The force law applies
        `-F_normal * n` to body i, so a flipped normal turns repulsion into
        ATTRACTION. Measured over 400 random tumbled pairs at preset blockiness
        (n = 2.2-3.5): the sign is essentially a coin flip, median n.u = -0.03.

        This is why `contact.solver = 'common_normal'` should be understood as a
        REPRODUCIBILITY path for the stored V2.7 fixtures, not as physics.
        """
        rng = np.random.default_rng(5)
        flipped = 0
        total = 0
        dots = []
        for _ in range(400):
            shp_i = (40.0, 30.0, 25.0, rng.uniform(2.2, 3.5), rng.uniform(2.2, 3.5))
            shp_j = (33.0, 36.0, 28.0, rng.uniform(2.2, 3.5), rng.uniform(2.2, 3.5))
            qi = quat_random(rng)
            qj = quat_random(rng)
            u = rand_unit(rng)
            cj = 2.0 * 40.0 * rng.uniform(0.70, 0.98) * u
            old = find_contact_superellipsoids_3d(0.0, 0.0, 0.0, *shp_i, qi,
                                                  cj[0], cj[1], cj[2], *shp_j, qj)
            if old is None:
                continue
            total += 1
            d = float(np.dot(np.array(old[2:5]), u))
            dots.append(d)
            if d < 0.0:
                flipped += 1
        self.assertGreater(total, 30)
        self.assertGreater(flipped / total, 0.3,
                           "the legacy normal-sign defect should still be present")
        self.assertLess(abs(float(np.median(dots))), 0.3, "median alignment ~ 0")

    def test_mtd_never_reports_the_normal_backwards(self):
        """The same 400 pairs through the MTD solver."""
        rng = np.random.default_rng(5)
        total = 0
        for _ in range(400):
            shp_i = (40.0, 30.0, 25.0, rng.uniform(2.2, 3.5), rng.uniform(2.2, 3.5))
            shp_j = (33.0, 36.0, 28.0, rng.uniform(2.2, 3.5), rng.uniform(2.2, 3.5))
            qi = quat_random(rng)
            qj = quat_random(rng)
            u = rand_unit(rng)
            cj = 2.0 * 40.0 * rng.uniform(0.70, 0.98) * u
            out = se3d_mtd_core(0.0, 0.0, 0.0, *shp_i, qi, cj[0], cj[1], cj[2], *shp_j, qj)
            if not out[0]:
                continue
            total += 1
            self.assertGreater(float(np.dot(np.array(out[2:5]), u)), 0.0)
        self.assertGreater(total, 100)


class TestLegacyCurvatureDegeneracy(SeededCase):

    def test_superellipse_curvature_falls_into_its_own_fallback_on_an_axis(self):
        """`superellipse_curvature_radius` computes `sign(sin t)` in dy/dt, which
        is exactly 0 at t = 0, so BOTH derivatives vanish, the `num < 1e-30`
        guard fires and it returns the `max(a,b)*10` "nearly flat" fallback --
        400 um for a circle of radius 40. An aligned circular pair converges to
        exactly t = 0, so this is not a measure-zero curiosity in practice.
        The support-function radius has no such branch."""
        from gels.engine import superellipse_curvature_radius, support_R_eff_2d
        self.assertAlmostEqual(superellipse_curvature_radius(0.0, 40.0, 40.0, 2.0), 400.0)
        self.assertAlmostEqual(superellipse_curvature_radius(0.3, 40.0, 40.0, 2.0),
                               40.0, places=6)
        for t in (0.0, np.pi / 2, np.pi, 0.3, 1.1):
            R = support_R_eff_2d(np.cos(t), np.sin(t), 40.0, 40.0, 2.0)
            self.assertAlmostEqual(R, 40.0, places=5, msg=f"t={t}")


if __name__ == '__main__':
    unittest.main()
