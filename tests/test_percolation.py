"""
Union-find percolation on the granule contact graph (V3.3).
===========================================================

Pins the algorithm in gels/kernels/percolation.py: cluster labelling, the
Newman-Ziff periodic wrapping test, the walled spanning test, and the
degenerate cases that would otherwise put NaN into history.json.

Run:  python -m unittest tests.test_percolation -v
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

from gels.kernels.percolation import clusters, graph_metrics  # noqa: E402


def _edges(pairs):
    pi = np.array([a for a, _ in pairs], dtype=np.int64)
    pj = np.array([b for _, b in pairs], dtype=np.int64)
    return pi, pj, np.ones(len(pairs), dtype=bool)


class TestClusters(unittest.TestCase):

    def _run(self, pairs, n, pos, box=(1000.0, 1000.0), periodic=False, keep=None):
        pi, pj, ic = _edges(pairs)
        if keep is None:
            keep = np.ones(n, dtype=bool)
        return clusters(pi, pj, ic, n, keep, np.asarray(pos, float), box, periodic, 2)

    def test_chain_is_one_cluster(self):
        pos = [[i * 10.0, 0.0] for i in range(5)]
        sizes, _, n_keep = self._run([(0, 1), (1, 2), (2, 3), (3, 4)], 5, pos)
        self.assertEqual(list(sizes), [5])
        self.assertEqual(sizes.max() / n_keep, 1.0)

    def test_two_chains(self):
        pos = [[i * 10.0, 0.0] for i in range(6)]
        sizes, _, n_keep = self._run([(0, 1), (1, 2), (3, 4), (4, 5)], 6, pos)
        self.assertEqual(sorted(sizes), [3, 3])
        self.assertAlmostEqual(sizes.max() / n_keep, 0.5)

    def test_isolated_granule_is_a_singleton(self):
        """Sizes must account for every kept granule, or lf is computed on a
        denominator that does not match the numerator."""
        pos = [[0.0, 0.0], [10.0, 0.0], [500.0, 500.0]]
        sizes, _, n_keep = self._run([(0, 1)], 3, pos)
        self.assertEqual(sorted(sizes), [1, 2])
        self.assertEqual(sizes.sum(), n_keep)

    def test_only_contacting_edges_participate(self):
        pos = [[i * 10.0, 0.0] for i in range(4)]
        pi, pj, _ = _edges([(0, 1), (1, 2), (2, 3)])
        ic = np.array([True, False, True])          # middle edge is not a contact
        sizes, _, n_keep = clusters(pi, pj, ic, 4, np.ones(4, bool),
                                    np.asarray(pos, float), (1000.0, 1000.0), False, 2)
        self.assertEqual(sorted(sizes), [2, 2])

    def test_mask_restricts_the_subgraph(self):
        pos = [[i * 10.0, 0.0] for i in range(4)]
        keep = np.array([True, False, True, True])
        sizes, _, n_keep = self._run([(0, 1), (1, 2), (2, 3)], 4, pos, keep=keep)
        self.assertEqual(n_keep, 3)
        # 0 is isolated once 1 is excluded; 2-3 stay joined
        self.assertEqual(sorted(sizes), [1, 2])


class TestPeriodicWrapping(unittest.TestCase):
    """The Newman-Ziff image-offset test -- the easiest thing here to get wrong."""

    L = 100.0

    def _wrap(self, pairs):
        pos = np.array([[5.0, 50.0], [30.0, 50.0], [55.0, 50.0], [80.0, 50.0]])
        pi, pj, ic = _edges(pairs)
        _, wrap, _ = clusters(pi, pj, ic, 4, np.ones(4, bool), pos,
                              (self.L, self.L), True, 2)
        return list(bool(w) for w in wrap)

    def test_ring_closing_through_the_boundary_wraps(self):
        # 3 -> 0 is only a contact through the x boundary
        self.assertEqual(self._wrap([(0, 1), (1, 2), (2, 3), (3, 0)]), [True, False])

    def test_broken_ring_does_not_wrap(self):
        self.assertEqual(self._wrap([(0, 1), (1, 2), (2, 3)]), [False, False])

    def test_a_cluster_touching_both_faces_without_closing_does_not_wrap(self):
        """Reaching both faces is not the same as being connected through them."""
        self.assertEqual(self._wrap([(0, 1), (2, 3)]), [False, False])


class TestGraphMetrics(unittest.TestCase):

    def _metrics(self, pairs, n, pos, groups, periodic=False, **kw):
        pi, pj, ic = _edges(pairs)
        return graph_metrics(pi, pj, ic, n, np.asarray(pos, float),
                             (1000.0, 1000.0, 1000.0), periodic, 2, groups, **kw)

    def test_coordination_and_rattlers(self):
        pos = [[i * 10.0, 0.0] for i in range(4)]
        m = self._metrics([(0, 1), (1, 2), (2, 3)], 4, pos,
                          {'solid': np.ones(4, bool)})
        # degrees are 1,2,2,1 -> mean 1.5; all below the 2D rattler cut of 3
        self.assertAlmostEqual(m['gran_z_mean'], 1.5)
        self.assertAlmostEqual(m['gran_rattler_frac'], 1.0)

    def test_empty_and_single_produce_no_nan(self):
        """Every value reaches history.json through csv/json, which cannot
        carry NaN, so the degenerate cases must be finite."""
        for n in (0, 1):
            m = self._metrics([], n, np.zeros((n, 2)), {'solid': np.ones(n, bool)})
            for k, v in m.items():
                if isinstance(v, float):
                    self.assertFalse(np.isnan(v), f"{k} is NaN at n={n}")

    def test_groups_are_reported_independently(self):
        pos = [[i * 10.0, 0.0] for i in range(4)]
        func = np.array([True, True, False, False])
        m = self._metrics([(0, 1), (1, 2), (2, 3)], 4, pos,
                          {'func': func, 'solid': np.ones(4, bool)})
        self.assertEqual(m['gran_nc_func'], 1)
        self.assertAlmostEqual(m['gran_lf_func'], 1.0)
        self.assertEqual(m['gran_nc_solid'], 1)


class TestAgainstTheFieldMeasure(unittest.TestCase):

    def test_graph_never_exceeds_the_field_measure_on_a_settled_packing(self):
        """The tanh field joins granules separated by ~2*interface_width of
        nothing, so field continuity should over-report relative to real
        contacts. If this ever inverts, one of the two is wrong."""
        import contextlib
        import io
        from gels.engine import (Params, generate_packing, render_fields_species,
                                 group_species_fields, compute_metrics)
        p = Params(mode='2D', Lx=600.0, Ly=600.0, phi_solid_target=0.55,
                   func_ratio=0.6, packing_settle_steps=30, save_data=False,
                   cell_surface_coverage=0.8)
        with contextlib.redirect_stdout(io.StringIO()):
            gs = generate_packing(p, seed=5)
        phi_s = render_fields_species(gs, p)
        pf, pi_, pv = group_species_fields(gs, phi_s)
        m = compute_metrics(gs, p, pf, pi_, pv, 0.0, np.zeros((gs.N, 2)), phi_s=phi_s)
        self.assertGreaterEqual(m['func_lf'] + 1e-9, m['gran_lf_func'])
        self.assertGreaterEqual(m['gran_lf_func'], 0.0)
        self.assertLessEqual(m['gran_lf_func'], 1.0)


if __name__ == '__main__':
    unittest.main()
