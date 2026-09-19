"""
Cell division and the cell cycle (V3.1, extended V3.2).
======================================================

Fibroblasts double every 20-30 h, so a 72 h run should end with roughly three
times the seeded population. V3.0 fixed the cell count at seeding; this checks
the growth law, contact inhibition, the CSR surgery that grows the per-cell
arrays, and that snapshots round-trip a changed cell count.

Run:  python -m unittest tests.test_cell_division -v
"""

import copy
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.division import age_cells, capacity_vector, divide_cells  # noqa: E402
from gels.engine import (  # noqa: E402
    CellState, GranuleSystem, Params, _initialize_cells, capacity_coverage, generate_packing,
    restore_gs_from_snapshot, save_snapshot_to_disk, update_cell_state,
)

SP = [dict(name='c', f=1.0, volume_fraction=1.0, color='#CC2222',
           radius_mean=40.0, radius_std=0.0, radius_min=40.0)]


def _sys(n_gran=4, n_cells=2, **kw):
    base = dict(mode='2D', Lx=2000.0, Ly=2000.0, save_data=False, dt=0.5,
                cell_surface_coverage=0.2, cell_capacity_coverage=1.0,
                cell_division_enabled=True, cell_doubling_time=24.0,
                cell_division_min_age=0.0, cell_division_max_layers=1.0,
                species=copy.deepcopy(SP))
    base.update(kw)
    p = Params(**base)
    xs = [200.0 + 400.0 * i for i in range(n_gran)]
    gs = GranuleSystem(xs, [200.0] * n_gran, [40.0] * n_gran, [0] * n_gran,
                       [n_cells] * n_gran, mode='2D',
                       species_id=[0] * n_gran, species=copy.deepcopy(SP), p=p)
    for i in range(n_gran):
        sl = gs.cells_on_granule(i)
        gs.cell_granule_id[sl] = i
    return gs, p


class TestCapacity(unittest.TestCase):

    def test_capacity_coverage_separates_seeding_from_capacity(self):
        p = Params(cell_surface_coverage=0.25, cell_capacity_coverage=1.0)
        self.assertEqual(capacity_coverage(p), 1.0)
        p2 = Params(cell_surface_coverage=0.25)          # 0 -> the V3.0 behaviour
        self.assertEqual(capacity_coverage(p2), 0.25)

    def test_capacity_vector_matches_the_monolayer_rule(self):
        gs, p = _sys()
        cap = capacity_vector(gs, p)
        # 2D: pi R^2 * coverage / A_cell with A_cell = pi d^3 / (4 h) = 1256.6 um^2,
        # so a 40 um granule holds 4 cells at full coverage (16 in 3D).
        A_cell = np.pi * p.cell_diameter ** 3 / (4.0 * p.cell_height_spread)
        expected = max(1.0, round(np.pi * 40.0 ** 2 * 1.0 / A_cell))
        self.assertEqual(expected, 4.0)
        self.assertTrue(bool(np.all(cap == expected)), cap)


class TestGrowthLaw(unittest.TestCase):

    def test_population_doubles_on_schedule(self):
        """With capacity far above the count, N(t) = N0 2^(t/T_d)."""
        T_d, hours = 24.0, 48.0
        counts = []
        for seed in range(12):
            gs, p = _sys(n_gran=6, n_cells=3, cell_doubling_time=T_d,
                         cell_division_max_layers=1e6)
            rng = np.random.default_rng(seed)
            n0 = gs.total_cells
            for k in range(int(hours / p.dt)):
                divide_cells(gs, p, rng, k * p.dt)
            counts.append(gs.total_cells / n0)
        got = float(np.mean(counts))
        expected = 2.0 ** (hours / T_d)
        se = float(np.std(counts, ddof=1)) / np.sqrt(len(counts))
        self.assertAlmostEqual(got, expected, delta=max(3.0 * se, 0.12 * expected),
                               msg=f"growth {got:.2f}x vs {expected:.2f}x expected")

    def test_no_division_at_capacity(self):
        gs, p = _sys(n_gran=4, n_cells=2, cell_capacity_coverage=0.0,
                     cell_surface_coverage=0.2)
        cap = capacity_vector(gs, p)
        gs.n_cells[:] = cap                       # exactly confluent
        rng = np.random.default_rng(1)
        born = sum(divide_cells(gs, p, rng, k * p.dt) for k in range(100))
        self.assertEqual(born, 0)

    def test_disabled_makes_no_draws(self):
        gs, p = _sys(cell_division_enabled=False)
        rng_a = np.random.default_rng(7)
        rng_b = np.random.default_rng(7)
        for k in range(20):
            self.assertEqual(divide_cells(gs, p, rng_a, k * p.dt), 0)
        np.testing.assert_array_equal(rng_a.random(5), rng_b.random(5))
        self.assertEqual(gs.total_cells, 8)

    def test_min_age_delays_division(self):
        """Blocked before the refractory period, dividing after it.

        V3.1 asserted only the first half, and that half was vacuously true:
        nothing advanced ``cell_age``, so no cell ever cleared ANY min_age.
        """
        gs, p = _sys(cell_division_min_age=12.0, cell_division_max_layers=1e6)
        rng = np.random.default_rng(2)
        early = 0
        for k in range(20):                       # cell_age reaches 10 h < 12 h
            age_cells(gs, p)
            early += divide_cells(gs, p, rng, k * p.dt)
        self.assertEqual(early, 0)
        self.assertAlmostEqual(float(gs.cell_age.max()), 10.0, places=9)
        late = 0
        for k in range(20, 100):                  # on past the refractory period
            age_cells(gs, p)
            late += divide_cells(gs, p, rng, k * p.dt)
        self.assertGreater(late, 0)

    def test_default_min_age_still_divides(self):
        """Regression for the V3.1 bug that made division a no-op in real runs.

        ``gs.cell_age`` was allocated, read and reset but never incremented --
        the compiled kernel's parameter of that name is bound to
        ``cell_bridge_age``. With the shipped default ``min_age_h = 8.0``,
        which the ``fibroblast_realistic`` preset uses, the eligibility test was
        false for ever. Every division test set ``min_age = 0``, so the suite
        never noticed.
        """
        gs, p = _sys(cell_division_min_age=8.0, cell_division_max_layers=1e6)
        rng = np.random.default_rng(3)
        born = 0
        for k in range(int(48 / p.dt)):
            age_cells(gs, p)
            born += divide_cells(gs, p, rng, k * p.dt)
        self.assertGreater(born, 0,
                           "no divisions at the shipped default min_age -- cell_age is not advancing")

    def test_age_cells_advances_live_cells_only(self):
        gs, p = _sys()
        gs.cell_state[0] = int(CellState.SENESCENT)
        age_cells(gs, p)
        age_cells(gs, p)
        self.assertEqual(float(gs.cell_age[0]), 0.0)
        self.assertAlmostEqual(float(gs.cell_age[1]), 2.0 * p.dt, places=9)

    def test_parent_reset_survives_the_csr_rebuild(self):
        """Regression: the parent's age reset landed on the wrong cell (V3.1).

        ``add_cells`` shifts every absolute cell index above a birth, so the
        ``parents`` index array is stale once the CSR layout is rebuilt. V3.1
        reset afterwards, zeroing whichever cell had moved into that slot. It
        was invisible under the memoryless rule but makes the cycle model
        diverge -- a parent that never resets divides again on every step.
        """
        gs, p = _sys(n_gran=4, n_cells=2, cell_division_max_layers=1e6,
                     cell_division_min_age=0.0, cell_division_model='cycle')
        n0 = gs.total_cells
        gs.cell_cycle_time[:] = 24.0
        gs.cell_age[:] = 25.0                       # every cell is due to divide
        born = divide_cells(gs, p, np.random.default_rng(0), 0.0)
        self.assertEqual(born, n0)
        self.assertEqual(gs.total_cells, 2 * n0)
        # parents reset, daughters born fresh: nobody is left mid-cycle
        np.testing.assert_allclose(gs.cell_age, 0.0)

    def test_seeded_population_is_not_synchronised(self):
        """Cells arrive spread through the cycle, not all at phase zero."""
        gs, p = _sys(n_gran=20, n_cells=4, cell_division_model='cycle',
                     cell_division_cv=0.15)
        _initialize_cells(gs, np.random.default_rng(5), p)
        age = gs.cell_age[:gs.total_cells]
        T = gs.cell_cycle_time[:gs.total_cells]
        self.assertTrue(np.all(age >= 0.0) and np.all(age < T))
        # U(0, T) has CV = 1/sqrt(3); allow for the sample size and the T spread
        self.assertAlmostEqual(age.std() / age.mean(), 1.0 / np.sqrt(3.0), delta=0.12)
        # and the cycle lengths themselves are spread at roughly the stated CV
        self.assertAlmostEqual(T.std() / T.mean(), 0.15, delta=0.06)

    def test_seeding_draws_nothing_when_division_is_off(self):
        """The desynchronisation draws must not perturb the packing stream."""
        gs, p = _sys(n_gran=6, n_cells=3, cell_division_enabled=False)
        rng_a = np.random.default_rng(4)
        rng_b = np.random.default_rng(4)
        _initialize_cells(gs, rng_a, p)
        # replay the surface-angle draws the loop itself consumes
        for i in range(gs.N):
            n = gs.cell_offset[i + 1] - gs.cell_offset[i]
            if n:
                rng_b.uniform(0, 2 * np.pi / max(1, n))
        np.testing.assert_array_equal(rng_a.random(5), rng_b.random(5))
        np.testing.assert_allclose(gs.cell_age, 0.0)

    def test_cycle_model_doubles_on_schedule_despite_the_refractory(self):
        """The cycle model realises its stated doubling time; poisson does not.

        With a memoryless rule the refractory period adds to the mean cycle, so
        ``doubling_time_h`` is a rate constant rather than the realised doubling
        time. Under the cycle model the cell divides at its own T_i, so an 8 h
        refractory below T_i costs nothing.
        """
        def grow(**kw):
            gs, p = _sys(n_gran=40, n_cells=4, cell_division_max_layers=1e6, **kw)
            _initialize_cells(gs, np.random.default_rng(5), p)
            rng = np.random.default_rng(11)
            n0 = gs.total_cells
            for k in range(int(48 / p.dt)):
                age_cells(gs, p)
                divide_cells(gs, p, rng, k * p.dt)
            return gs.total_cells / n0

        cycle = grow(cell_division_model='cycle', cell_division_cv=0.15,
                     cell_division_min_age=8.0)
        self.assertAlmostEqual(cycle, 4.0, delta=0.6)          # 2^(48/24)
        poisson = grow(cell_division_model='poisson', cell_division_min_age=8.0)
        self.assertLess(poisson, cycle)

    def test_no_division_burst_at_the_refractory_period(self):
        """A synchronised population divides in a spike; a seeded one does not."""
        def per_step(sync):
            gs, p = _sys(n_gran=40, n_cells=4, cell_division_max_layers=1e6,
                         cell_division_min_age=8.0, cell_division_model='cycle',
                         cell_division_cv=0.15)
            _initialize_cells(gs, np.random.default_rng(5), p)
            if sync:
                gs.cell_age[:] = 0.0                # the V3.1 state: everyone at phase 0
                gs.cell_cycle_time[:] = p.cell_doubling_time
            rng = np.random.default_rng(11)
            out = []
            for k in range(int(30 / p.dt)):
                age_cells(gs, p)
                out.append(divide_cells(gs, p, rng, k * p.dt))
            return np.array(out, dtype=float)

        spread, sync = per_step(False), per_step(True)
        # a synchronised population divides in one step and is silent otherwise
        self.assertLessEqual(int((sync > 0).sum()), 2)
        # a seeded one divides throughout, with no spike
        self.assertGreater(int((spread > 0).sum()), 20)
        self.assertLess(spread.max(), 0.2 * sync.max())

    def test_senescent_cells_do_not_divide(self):
        gs, p = _sys(cell_division_max_layers=1e6)
        gs.cell_state[:] = int(CellState.SENESCENT)
        rng = np.random.default_rng(3)
        born = sum(divide_cells(gs, p, rng, k * p.dt) for k in range(40))
        self.assertEqual(born, 0)


class TestCSRIntegrity(unittest.TestCase):

    def test_invariants_and_attributes_survive(self):
        gs, p = _sys(n_gran=5, n_cells=3, cell_division_max_layers=1e6)
        gs.cell_theta_local[:] = np.arange(gs.total_cells)
        gs.cell_bridge_target[:] = -1
        gs.cell_bridge_target[1] = 4
        gs.cell_state[2] = int(CellState.BRIDGING)
        rng = np.random.default_rng(11)
        for k in range(20):
            divide_cells(gs, p, rng, k * p.dt)
        self.assertGreater(gs.total_cells, 15)
        np.testing.assert_array_equal(np.diff(gs.cell_offset), gs.n_cells.astype(int))
        self.assertEqual(int(gs.cell_offset[-1]), gs.total_cells)
        for i in range(gs.N):
            sl = gs.cells_on_granule(i)
            self.assertTrue(bool(np.all(gs.cell_granule_id[sl] == i)), i)
        for name, _dt, _f in gs.CELL_ARRAYS:
            self.assertEqual(len(getattr(gs, name)), gs.total_cells, name)
        # the preserved cell keeps its bridge target and state
        sl0 = gs.cells_on_granule(0)
        self.assertEqual(int(gs.cell_bridge_target[sl0][1]), 4)
        self.assertEqual(int(gs.cell_state[sl0][2]), int(CellState.BRIDGING))
        self.assertEqual(int(gs.cell_theta_local[sl0][0]), 0)
        self.assertGreater(int(gs.n_divisions_cum), 0)

    def test_daughters_start_fresh(self):
        gs, p = _sys(n_gran=2, n_cells=1, cell_division_max_layers=1e6)
        rng = np.random.default_rng(5)
        before = gs.total_cells
        while gs.total_cells == before:
            divide_cells(gs, p, rng, 1.0)
        new = gs.cell_generation > 0
        self.assertTrue(bool(np.any(new)))
        self.assertTrue(bool(np.all(gs.cell_age[new] == 0.0)))
        self.assertTrue(bool(np.all(gs.cell_bridge_target[new] == -1)))
        self.assertTrue(bool(np.all(gs.cell_state[new] == int(CellState.ATTACHED))))


class TestSnapshotRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_div_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_grown_population_round_trips(self):
        p = Params(mode='2D', Lx=600.0, Ly=600.0, phi_solid_target=0.4, func_ratio=1.0,
                   R_func_mean=40.0, R_func_std=2.0, cell_surface_coverage=0.2,
                   cell_capacity_coverage=1.0, cell_division_enabled=True,
                   cell_division_min_age=0.0, cell_division_max_layers=1e6,
                   packing_settle_steps=20, save_data=False, output_dir=self.tmp)
        gs = generate_packing(p, seed=1)
        rng = np.random.default_rng(2)
        n0 = gs.total_cells
        for k in range(10):
            update_cell_state(gs, p, k * p.dt, rng)
        self.assertGreater(gs.total_cells, n0)
        F = np.zeros((gs.N, 2))
        save_snapshot_to_disk(0, gs, p, 5.0, F, self.tmp)
        path = os.path.join(self.tmp, 'snapshots', 'snap_0000.npz')
        gs2, t2, idx = restore_gs_from_snapshot(path, p)
        self.assertEqual(gs2.total_cells, gs.total_cells)
        np.testing.assert_array_equal(gs2.cell_offset, gs.cell_offset)
        np.testing.assert_array_equal(gs2.cell_granule_id, gs.cell_granule_id)
        np.testing.assert_array_equal(gs2.cell_generation, gs.cell_generation)
        np.testing.assert_allclose(gs2.cell_age, gs.cell_age)
        self.assertEqual(gs2.n_divisions_cum, gs.n_divisions_cum)

    def test_legacy_snapshot_without_the_new_keys(self):
        p = Params(mode='2D', Lx=400.0, Ly=400.0, phi_solid_target=0.3, func_ratio=1.0,
                   R_func_mean=40.0, cell_surface_coverage=0.2, packing_settle_steps=10,
                   save_data=False, output_dir=self.tmp)
        gs = generate_packing(p, seed=3)
        F = np.zeros((gs.N, 2))
        save_snapshot_to_disk(0, gs, p, 0.0, F, self.tmp)
        path = os.path.join(self.tmp, 'snapshots', 'snap_0000.npz')
        data = dict(np.load(path, allow_pickle=False))
        for k in ('cell_age', 'cell_generation', 'cell_bridge_gap_prev',
                  'cell_bridge_force', 'n_divisions_cum', 'fixed'):
            data.pop(k, None)
        np.savez_compressed(path, **data)
        gs2, _t, _i = restore_gs_from_snapshot(path, p)
        self.assertEqual(gs2.total_cells, gs.total_cells)
        self.assertTrue(bool(np.all(gs2.cell_age == 0.0)))
        self.assertTrue(bool(np.all(np.isnan(gs2.cell_bridge_gap_prev))))
        self.assertEqual(gs2.n_divisions_cum, 0)


class TestBackendsAgree(unittest.TestCase):

    def test_python_and_kernel_paths_divide_equivalently(self):
        """Both update_cell_state twins run the same division pass.

        They are statistically equivalent rather than bit-identical: the
        compiled cell machinery consumes the run's Generator differently (one
        draw per pass plus a counter hash, against per-cell draws in the
        reference), which is the documented V3.0 behaviour for every
        stochastic cell rule. Growth rate and CSR integrity must still match.
        """
        out = []
        for use_numba in (False, True):
            p = Params(mode='2D', Lx=600.0, Ly=600.0, phi_solid_target=0.4, func_ratio=1.0,
                       R_func_mean=40.0, R_func_std=2.0, cell_surface_coverage=0.2,
                       cell_capacity_coverage=1.0, cell_division_enabled=True,
                       cell_division_min_age=0.0, packing_settle_steps=20,
                       save_data=False, use_numba=use_numba)
            gs = generate_packing(p, seed=4)
            n0 = gs.total_cells
            rng = np.random.default_rng(9)
            for k in range(12):
                update_cell_state(gs, p, k * p.dt, rng)
            np.testing.assert_array_equal(np.diff(gs.cell_offset), gs.n_cells.astype(int))
            self.assertEqual(gs.total_cells, n0 + gs.n_divisions_cum)
            out.append(gs.total_cells / n0)
        self.assertGreater(out[0], 1.0)
        self.assertAlmostEqual(out[0], out[1], delta=0.15 * out[0],
                               msg=f"growth {out[0]:.3f}x (python) vs {out[1]:.3f}x (kernels)")


class TestClockJitter(unittest.TestCase):
    """V3.2: attachment, spreading and FA maturation are functions of global
    simulation time, so without a per-granule offset the whole bed matures on
    exactly the same step."""

    def _mature_at(self, jitter, t=6.0):
        gs, p = _sys(n_gran=30, n_cells=2, cell_clock_jitter=jitter,
                     t_attach_onset=3.0, t_attach_half=2.0)
        _initialize_cells(gs, np.random.default_rng(5), p)
        update_cell_state(gs, p, t, np.random.default_rng(1))
        return gs.spread_fraction[:gs.N].copy(), gs.fa_maturity[:gs.N].copy()

    def test_no_jitter_matures_the_whole_bed_in_lockstep(self):
        spread, fa = self._mature_at(0.0)
        self.assertLess(float(spread.std()), 1e-12)     # identical bar float noise
        self.assertLess(float(fa.std()), 1e-12)

    def test_jitter_spreads_maturation_across_granules(self):
        spread, fa = self._mature_at(3.0)
        self.assertGreater(float(spread.std()), 0.05)
        self.assertGreater(float(fa.std()), 0.05)
        self.assertTrue(np.all((spread >= 0.0) & (spread <= 1.0)))


if __name__ == '__main__':
    unittest.main()
