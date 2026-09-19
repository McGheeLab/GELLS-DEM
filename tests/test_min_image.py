"""
Minimum-image guard for periodic neighbour search (V3.3).
=========================================================

``cKDTree(pos, boxsize=L).query_pairs(r)`` returns each unordered pair AT MOST
ONCE, at its minimum image, and raises nothing when ``r > L/2``. Above that a
granule has more than one image of a neighbour inside the cutoff and only one
is reported, so an interaction is silently **dropped** -- a missing contact,
not a double-counted one, which is why it never presented as an obviously
wrong force.

``gels.kernels.neighbors.check_min_image`` refuses that configuration. These
tests pin the failure mode it prevents, the exact threshold, and the fact that
walled geometry (which has no images) is exempt.

Run:  python -m unittest tests.test_min_image -v
"""

import itertools
import os
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.kernels.neighbors import check_min_image, half_pairs  # noqa: E402


def _brute_image_count(pos, box, cutoff, dim):
    """Interactions between distinct points over all 3^dim periodic images."""
    n = len(pos)
    total = 0
    for a, b in itertools.combinations(range(n), 2):
        for shift in itertools.product((-1, 0, 1), repeat=dim):
            d = np.linalg.norm(pos[b] + np.array(shift) * box - pos[a])
            if d <= cutoff:
                total += 1
    return total


class TestDroppedPairRegression(unittest.TestCase):
    """The concrete bug the guard exists to prevent."""

    def test_query_pairs_drops_the_second_image(self):
        L = 10.0
        box = np.array([L, L])
        pos = np.array([[1.0, 0.5], [9.0, 0.5]])

        # below L/2: one image in range, and query_pairs agrees with brute force
        got = cKDTree(pos, boxsize=box).query_pairs(2.5, output_type='ndarray')
        self.assertEqual(len(got), 1)
        self.assertEqual(_brute_image_count(pos, box, 2.5, 2), 1)

        # above L/2: TWO images are within the cutoff (gaps of 2 and 8) but
        # query_pairs still reports the pair once. This is the silent drop.
        got = cKDTree(pos, boxsize=box).query_pairs(8.0, output_type='ndarray')
        self.assertEqual(len(got), 1)
        self.assertEqual(_brute_image_count(pos, box, 8.0, 2), 2)


class TestCheckMinImage(unittest.TestCase):

    def test_threshold_is_strict_at_half_the_shortest_edge(self):
        box = [10.0, 10.0]
        check_min_image(4.0, box)          # comfortably inside
        check_min_image(4.999, box)        # just inside
        # query_pairs is inclusive (d <= r), so equality is already unsafe
        with self.assertRaises(ValueError):
            check_min_image(5.0, box)
        with self.assertRaises(ValueError):
            check_min_image(8.0, box)

    def test_uses_the_shortest_edge_and_names_it(self):
        with self.assertRaises(ValueError) as cm:
            check_min_image(30.0, [400.0, 50.0, 400.0])
        msg = str(cm.exception)
        self.assertIn('y = 50.000', msg)           # the binding axis
        self.assertIn('60.000', msg)               # the required edge, 2*cutoff
        # the long axes on their own would have been fine
        check_min_image(30.0, [400.0, 400.0, 400.0])

    def test_respects_dim_for_a_2d_run_in_a_3d_box(self):
        # a 2D run must not be judged against Lz
        check_min_image(100.0, [400.0, 400.0, 1.0], dim=2)
        with self.assertRaises(ValueError):
            check_min_image(100.0, [400.0, 400.0, 1.0], dim=3)


class TestHalfPairsGuard(unittest.TestCase):

    def setUp(self):
        self.box = np.array([10.0, 10.0])
        self.pos = np.array([[1.0, 0.5], [9.0, 0.5], [5.0, 5.0]])

    def test_both_backends_raise(self):
        for backend in ('cells', 'ckdtree'):
            with self.subTest(backend=backend):
                with self.assertRaises(ValueError):
                    half_pairs(self.pos, 8.0, True, self.box, backend)

    def test_both_backends_accept_a_valid_cutoff(self):
        for backend in ('cells', 'ckdtree'):
            with self.subTest(backend=backend):
                half_pairs(self.pos, 4.0, True, self.box, backend)

    def test_walls_are_exempt(self):
        # no periodic images exist, so no cutoff is too large
        for backend in ('cells', 'ckdtree'):
            with self.subTest(backend=backend):
                half_pairs(self.pos, 99.0, False, self.box, backend)


class TestRealConfigurationsAreUnaffected(unittest.TestCase):
    """The guard must not fire on configurations the project actually ships."""

    def test_default_params_are_clear_of_the_limit(self):
        from gels.engine import Params
        p = Params()
        # generous r_bound estimate: the larger species at mean + 3 sd
        r_max = max(p.R_func_mean + 3 * p.R_func_std,
                    p.R_inert_mean + 3 * p.R_inert_std)
        cutoff = 2 * r_max + p.L_max
        check_min_image(cutoff, [p.Lx, p.Ly, p.Lz])


if __name__ == '__main__':
    unittest.main()
