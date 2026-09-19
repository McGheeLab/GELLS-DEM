"""
Numba environment checks the V3.0 kernels depend on.
=====================================================

The compiled kernels are all ``@njit(parallel=True, nogil=True, cache=True)``
with ``prange`` loops. This test proves that combination compiles, caches and
produces correct results on the installed numba / Python pair (numba 0.67 on
Python 3.14 was new when V3.0 started), and that the engine's thread
configuration behaves as documented. It runs in a few seconds.

Run:  python -m unittest tests.test_numba_env -v
"""

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gels.kernels import HAS_NUMBA, configure_threads, physical_cores  # noqa: E402


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestNumbaEnvironment(unittest.TestCase):

    def test_parallel_nogil_cached_kernel(self):
        """prange + nogil + cache compile together and give the numpy answer."""
        from numba import njit, prange

        @njit(parallel=True, nogil=True, cache=True)
        def rowsum(a, out):
            for i in prange(a.shape[0]):
                s = 0.0
                for j in range(a.shape[1]):
                    s += a[i, j]
                out[i] = s

        rng = np.random.default_rng(0)
        a = rng.random((5000, 64))
        out = np.empty(a.shape[0])
        rowsum(a, out)
        np.testing.assert_allclose(out, a.sum(axis=1), rtol=1e-12)
        # A second call must not recompile (dispatcher has the signature).
        self.assertEqual(len(rowsum.signatures), 1)
        rowsum(a, out)
        self.assertEqual(len(rowsum.signatures), 1)

    def test_configure_threads_auto_and_clamp(self):
        import numba
        layer, n = configure_threads(0, 'omp')
        self.assertEqual(n, min(physical_cores(), numba.config.NUMBA_NUM_THREADS))
        self.assertIn(layer, ('omp', 'tbb', 'workqueue', 'default'))
        # explicit request is clamped to what numba allows
        _, n2 = configure_threads(10**6, 'omp')
        self.assertEqual(n2, numba.config.NUMBA_NUM_THREADS)
        # restore the auto default for later tests
        configure_threads(0, 'omp')

    def test_thread_count_does_not_change_results(self):
        """Owner-writes-only kernels must be bit-identical for 1 vs many threads."""
        import numba
        from numba import njit, prange

        @njit(parallel=True, nogil=True, cache=True)
        def scale(a, out):
            for i in prange(a.shape[0]):
                out[i] = np.sqrt(a[i]) * 3.0 + 1.0

        a = np.random.default_rng(1).random(200_000)
        out1 = np.empty_like(a)
        outn = np.empty_like(a)
        numba.set_num_threads(1)
        scale(a, out1)
        numba.set_num_threads(min(8, numba.config.NUMBA_NUM_THREADS))
        scale(a, outn)
        configure_threads(0, 'omp')
        self.assertTrue(np.array_equal(out1, outn))


if __name__ == '__main__':
    unittest.main()
