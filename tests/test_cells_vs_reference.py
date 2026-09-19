"""
Compiled cell state machine and bridging vs the reference (V3.0 Phase 6c).

  * deterministic parts of the state machine (attachment, spreading, FA
    maturity, overcrowding, bridge ageing / lock-in / senescence, directed
    migration, state assignment) match the reference exactly when the random
    walk is off (rng=None),
  * servicing and rupture of committed bridges match exactly when no new
    attempts can happen (bridge_attempt_rate = 0),
  * with attempts on, bridging statistics over seeds are equivalent (the
    compiled path uses a hashed RNG, so trajectories are not comparable),
  * results are bit-identical for 1 vs 8 threads,
  * the hashed uniform is uniform and the perf_cells_backend switch routes.

Run:  python -m unittest tests.test_cells_vs_reference -v
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
    CellState, Params, cells_kernels_enabled, generate_packing, generate_packing_3d, step,
)
from gels.kernels import HAS_NUMBA  # noqa: E402
from gels.kernels import reference as ref  # noqa: E402

if HAS_NUMBA:
    from gels.kernels import cells, contact2d, contact3d  # noqa: E402

BRIDGING = int(CellState.BRIDGING)
MIGRATING = int(CellState.MIGRATING)


def _params(mode='2D', periodic=False, backend='kernels', **kw):
    base = dict(mode=mode, Lx=600.0, Ly=600.0, Lz=800.0,
                phi_solid_target=0.6, func_ratio=0.6, cell_surface_coverage=1.0,
                packing_settle_steps=30, save_data=False, compress_archive=False,
                bridge_attempt_rate=2.0, min_fa_for_bridge=0.0, fa_maturation_rate=100.0,
                t_spread_duration=0.1, T_active=0.0, cell_migration_speed=30.0,
                boundary_mode='periodic' if periodic else 'walls', perf_cells_backend=backend)
    if mode == '3D':
        base.update(Lx=300.0, Ly=300.0, Lz=300.0, phi_solid_target=0.5)
    base.update(kw)
    return Params(**base)


def _prepared(p_python, seed, n_steps=3):
    """Packing advanced n_steps on the exact reference cell path (creates bridging / migrating cells)."""
    gen = generate_packing_3d if p_python.mode == '3D' else generate_packing
    gs = gen(p_python, seed=seed)
    rng = np.random.default_rng(seed)
    t = 0.0
    for _ in range(n_steps):
        t += p_python.dt
        step(gs, p_python, rng, t)
    return gs, t


def _cell_arrays(gs):
    return {name: np.array(getattr(gs, name)) for name in (
        'n_attached', 'spread_fraction', 'fa_maturity', 'n_overcrowded',
        'cell_state', 'cell_bridge_target', 'cell_bridge_age', 'cell_alignment', 'cell_bridge_locked',
        'cell_contact_area', 'cell_overcrowd_age', 'cell_theta_local', 'cell_eta_local', 'cell_omega_local',
        'cell_fx', 'cell_fy', 'cell_fz')}


def _assert_cells_equal(tc, ak, ar, exact=('cell_state', 'cell_bridge_target', 'cell_bridge_locked'),
                        rtol=1e-9, atol=1e-9):
    for name in ar:
        if name in exact:
            np.testing.assert_array_equal(ak[name], ar[name], err_msg=name)
        else:
            np.testing.assert_allclose(ak[name], ar[name], rtol=rtol, atol=atol, err_msg=name)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestDeterministicParts(unittest.TestCase):

    def _check_state_machine(self, mode, periodic=False, seed=1, foothold=None):
        kw = {} if foothold is None else dict(cell_capacity_foothold=foothold)
        p_py = _params(mode, periodic, backend='python', **kw)
        gs, t = _prepared(p_py, seed)
        self.assertGreater(int(np.sum(gs.cell_state == BRIDGING)) + int(np.sum(gs.cell_state == MIGRATING)), 0,
                           "prepared system has no bridging/migrating cells")
        gs_r = copy.deepcopy(gs)
        gs_k = copy.deepcopy(gs)
        p_k = copy.deepcopy(p_py)
        p_k.perf_cells_backend = 'kernels'
        # rng=None: the reference skips the random walk; the kernel does too
        ref.update_cell_state(gs_r, p_py, t + p_py.dt, None)
        cells.update_cell_state_k(gs_k, p_k, t + p_k.dt, None)
        _assert_cells_equal(self, _cell_arrays(gs_k), _cell_arrays(gs_r))

    def test_state_machine_2d_walls(self):
        self._check_state_machine('2D')

    def test_state_machine_2d_periodic(self):
        self._check_state_machine('2D', periodic=True)

    def test_state_machine_3d(self):
        self._check_state_machine('3D', seed=2)

    # ---- V3.3: cell_capacity_foothold parity (regression) ----------------
    # The compiled _capacity had no foothold argument, so with
    # cell_capacity_foothold < 1 the two backends computed different
    # capacities (64 vs 16 at r=40 um, 3D, foothold=0.25 -- the
    # fibroblast_realistic default). Capacity sets the overcrowding and
    # division thresholds, so the backends disagreed on cell fate.

    def test_capacity_leaf_matches_reference_under_foothold(self):
        from gels.engine import cells_from_surface_coverage
        for foothold in (1.0, 0.5, 0.25, 0.1):
            for R in (10.0, 20.0, 40.0):
                for is_3d_like, mode in ((True, '3D'), (False, '2D')):
                    want = cells_from_surface_coverage(
                        R, 20.0, 5.0, 1.0, mode, foothold)
                    got = cells._capacity(R, 1.0, 1.0, 20.0, 5.0, 1.0, 0.6,
                                          is_3d_like, foothold)
                    self.assertEqual(got, want,
                                     f"foothold={foothold} R={R} mode={mode}")

    def test_state_machine_3d_with_foothold(self):
        self._check_state_machine('3D', seed=2, foothold=0.25)

    def _check_service(self, mode, periodic=False, seed=1):
        p_py = _params(mode, periodic, backend='python')
        gs, t = _prepared(p_py, seed)
        self.assertGreater(int(np.sum(gs.cell_state == BRIDGING)), 0)
        p_py2 = copy.deepcopy(p_py)
        p_py2.bridge_attempt_rate = 0.0            # no new bridges: service + rupture only
        p_k = copy.deepcopy(p_py2)
        p_k.perf_cells_backend = 'kernels'
        gs_r = copy.deepcopy(gs)
        gs_k = copy.deepcopy(gs)
        fn = contact3d.compute_forces_3d if mode == '3D' else contact2d.compute_forces_2d
        Fr, Tr, Cr = fn(gs_r, p_py2, np.random.default_rng(3))
        Fk, Tk, Ck = fn(gs_k, p_k, np.random.default_rng(3))
        np.testing.assert_allclose(Fk, Fr, rtol=1e-9, atol=1e-9)
        _assert_cells_equal(self, _cell_arrays(gs_k), _cell_arrays(gs_r))
        self.assertGreater(float(np.abs(gs_r.cell_fx).sum()), 0.0, "no bridge forces were applied")

    def test_service_2d_walls(self):
        self._check_service('2D')

    def test_service_2d_periodic(self):
        self._check_service('2D', periodic=True)

    def test_service_3d(self):
        self._check_service('3D', seed=2)


def _run_stats(p, seed, n_steps):
    gen = generate_packing_3d if p.mode == '3D' else generate_packing
    gs = gen(p, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    t = 0.0
    for _ in range(n_steps):
        t += p.dt
        step(gs, p, rng, t)
    bridging = gs.cell_state == BRIDGING
    n_b = int(np.sum(bridging))
    f_mag = np.sqrt(gs.cell_fx**2 + gs.cell_fy**2 + gs.cell_fz**2)
    return dict(n_bridging=n_b, n_migrating=int(np.sum(gs.cell_state == MIGRATING)),
                f_bridge=float(f_mag[bridging].mean()) if n_b else 0.0,
                n_senescent=int(np.sum(gs.cell_state == int(CellState.SENESCENT))))


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestStatisticalEquivalence(unittest.TestCase):

    def _compare(self, mode, seeds, n_steps, periodic=False):
        p_py = _params(mode, periodic, backend='python')
        p_k = _params(mode, periodic, backend='kernels')
        rows_r = [_run_stats(p_py, s, n_steps) for s in seeds]
        rows_k = [_run_stats(p_k, s, n_steps) for s in seeds]
        for key in ('n_bridging', 'n_migrating', 'f_bridge'):
            a = np.array([r[key] for r in rows_r], dtype=float)
            b = np.array([r[key] for r in rows_k], dtype=float)
            se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) + 1e-12
            self.assertLess(abs(a.mean() - b.mean()), 3.0 * se + 0.02 * max(a.mean(), b.mean()),
                            f"{key}: reference {a.mean():.3g} ± {a.std(ddof=1):.3g} vs kernels "
                            f"{b.mean():.3g} ± {b.std(ddof=1):.3g}")
        self.assertGreater(np.mean([r['n_bridging'] for r in rows_k]), 0.0)
        self.assertGreater(np.mean([r['n_bridging'] for r in rows_r]), 0.0)

    def test_2d_walls(self):
        self._compare('2D', seeds=range(1, 13), n_steps=20)

    def test_2d_periodic(self):
        self._compare('2D', seeds=range(1, 9), n_steps=20, periodic=True)

    def test_3d_spheres(self):
        self._compare('3D', seeds=range(1, 7), n_steps=12)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestKernelProperties(unittest.TestCase):

    def test_threads_bit_identical(self):
        import numba
        p = _params('2D')
        gs0 = generate_packing(p, seed=4)
        n_max = numba.config.NUMBA_NUM_THREADS
        old = numba.get_num_threads()
        results = []
        try:
            for nt in (1, min(8, n_max)):
                numba.set_num_threads(nt)
                gs = copy.deepcopy(gs0)
                rng = np.random.default_rng(9)
                t = 0.0
                for _ in range(4):
                    t += p.dt
                    step(gs, p, rng, t)
                results.append((np.array(gs.pos), _cell_arrays(gs)))
        finally:
            numba.set_num_threads(old)
        np.testing.assert_array_equal(results[0][0], results[1][0])
        for name in results[0][1]:
            np.testing.assert_array_equal(results[0][1][name], results[1][1][name], err_msg=name)

    def test_hash_uniform_and_pure(self):
        u = np.array([cells.hash_u01(12345, i, 7, 3) for i in range(20000)])
        self.assertTrue(np.all((u >= 0.0) & (u < 1.0)))
        self.assertAlmostEqual(float(u.mean()), 0.5, delta=0.01)
        self.assertAlmostEqual(float(u.var()), 1.0 / 12.0, delta=0.005)
        self.assertEqual(cells.hash_u01(1, 2, 3, 4), cells.hash_u01(1, 2, 3, 4))
        self.assertNotEqual(cells.hash_u01(1, 2, 3, 4), cells.hash_u01(1, 2, 3, 5))
        z = np.array([cells.hash_normal2(99, i, 0, 7)[0] for i in range(20000)])
        self.assertAlmostEqual(float(z.mean()), 0.0, delta=0.03)
        self.assertAlmostEqual(float(z.std()), 1.0, delta=0.03)

    def test_periodic_bridges_use_minimum_image(self):
        p = _params('2D', periodic=True, Lx=400.0, Ly=400.0)
        gs = generate_packing(p, seed=5)
        rng = np.random.default_rng(5)
        t = 0.0
        for _ in range(8):
            t += p.dt
            step(gs, p, rng, t)
        bridging = np.nonzero(gs.cell_state == BRIDGING)[0]
        self.assertGreater(bridging.size, 0)
        for ci in bridging:
            i = int(gs.cell_granule_id[ci])
            j = int(gs.cell_bridge_target[ci])
            dx = gs.x[j] - gs.x[i]
            dy = gs.y[j] - gs.y[i]
            dx -= p.Lx * np.round(dx / p.Lx)
            dy -= p.Ly * np.round(dy / p.Ly)
            gap = np.hypot(dx, dy) - gs.r_bound[i] - gs.r_bound[j]
            self.assertLess(gap, p.bridge_break_gap)

    def test_backend_switch(self):
        self.assertTrue(cells_kernels_enabled(Params()))
        self.assertFalse(cells_kernels_enabled(Params(perf_cells_backend='python')))
        self.assertFalse(cells_kernels_enabled(Params(use_numba=False)))


if __name__ == '__main__':
    unittest.main()
