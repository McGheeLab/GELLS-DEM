"""
Physics mapping of functionalization f into the force / cell loops (V3.0 Phase 2).

  * motor-clutch force is non-decreasing in the ligand gain g and equals the
    V2.7 value at g = 1 (guards the n_clutches sign trap),
  * mixed-species contacts carry the coverage-mixed W / τ₀ and the per-pair E*,
  * bridging never targets a bare (f = 0) granule but does target a partially
    coated one,
  * a partially coated blocker applies the interpolated line-of-sight penalty,
  * seeding and capacity scale with the species coverage.

Run:  python -m unittest tests.test_physics_mapping -v
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

from gels import materials as M  # noqa: E402
from gels.engine import (  # noqa: E402
    CellState, GranuleSystem, Params, cell_capacity, cells_from_surface_coverage,
    compute_forces, motor_clutch_force, seeded_cells, update_cell_state,
)
from gels.kernels import reference  # noqa: E402

SPECIES = [
    dict(name='coated', f=1.0, volume_fraction=0.4, color='#CC2222', radius_mean=40.0, radius_std=0.0,
         E_kPa=None, poisson_ratio=None),
    dict(name='half', f=0.5, volume_fraction=0.3, color='#2255CC', radius_mean=40.0, radius_std=0.0,
         E_kPa=None, poisson_ratio=None),
    dict(name='bare', f=0.0, volume_fraction=0.3, color='#22AA22', radius_mean=40.0, radius_std=0.0,
         E_kPa=20.0, poisson_ratio=None),
]


def _row_system(species_id, p, overlap=2.0, n_cells=8):
    """Three 40-µm circles in a row along x, each pair overlapping by `overlap` µm."""
    r = 40.0
    x = np.array([100.0, 100.0 + 2 * r - overlap, 100.0 + 2 * (2 * r - overlap)])
    y = np.full(3, 100.0)
    rr = np.full(3, r)
    f = np.array([SPECIES[k]['f'] for k in species_id])
    nc = np.array([n_cells if fk >= p.f_min_adhesion else 0 for fk in f])
    gs = GranuleSystem(x, y, rr, None, nc, species_id=np.array(species_id),
                       species=[dict(s) for s in SPECIES], p=p)
    from gels.engine import _initialize_cells
    _initialize_cells(gs, np.random.default_rng(0))
    return gs


def _quiet_params(**kw):
    base = dict(mode='2D', Lx=400.0, Ly=200.0, T_active=0.0, bridge_attempt_rate=0.0,
                cell_migration_speed=0.0, species=[dict(s) for s in SPECIES])
    base.update(kw)
    return Params(**base)


class TestMotorClutchGain(unittest.TestCase):

    def test_g_one_reproduces_v27_and_monotone(self):
        p = Params()
        for E in (1.0, 10.0, 100.0):
            legacy = motor_clutch_force(E, p, 1.0)
            self.assertEqual(motor_clutch_force(E, p, 1.0, nu=p.poisson_ratio, g=1.0), legacy)
            gs = np.linspace(0.0, 1.0, 51)
            F = np.array([motor_clutch_force(E, p, 1.0, g=g) for g in gs])
            self.assertEqual(F[0], 0.0)
            self.assertTrue(np.all(np.diff(F) >= 0.0), f"traction not monotone in g at E={E}")
            self.assertLess(F[25], F[-1])

    def test_host_stiffness_matters(self):
        p = Params()
        soft = motor_clutch_force(2.0, p, 1.0)
        stiff = motor_clutch_force(50.0, p, 1.0)
        self.assertLess(soft, stiff)


class TestMixedSpeciesContacts(unittest.TestCase):

    def test_contact_records_use_species_tables(self):
        p = _quiet_params()
        gs = _row_system([0, 1, 2], p)
        F, torques, contacts = compute_forces(gs, p, np.random.default_rng(1))
        self.assertEqual(len(contacts), 2)
        by_pair = {(c['i'], c['j']): c for c in contacts}
        c01 = by_pair[(0, 1)]
        c12 = by_pair[(1, 2)]
        # coated–half: mixing of the pair values
        self.assertAlmostEqual(c01['W'], M.mix_pair(1.0, 0.5, p.W_adh_cc, p.W_adh_cb, p.W_adh_bb), places=15)
        self.assertAlmostEqual(c01['tau_0'], M.mix_pair(1.0, 0.5, p.tau_0_cc, p.tau_0_cb, p.tau_0_bb), places=12)
        self.assertEqual(c01['E_star'], gs.pair_Estar[0, 1])
        # half–bare (E = 20 kPa on the bare species): harmonic-mean E*
        self.assertAlmostEqual(c12['E_star'], M.pair_E_star_Pa(p.E_modulus, p.poisson_ratio, 20.0, p.poisson_ratio),
                               places=6)
        self.assertGreater(c12['E_star'], c01['E_star'])
        for c in contacts:
            self.assertIn('species_i', c)
            self.assertIn('f_j', c)
        # forces balanced on the isolated row (no walls touched, no noise)
        self.assertLess(abs(F[:, 0].sum()), 1e-9)

    def test_legacy_species_reproduce_global_constants(self):
        p = _quiet_params(species=[])
        gs = GranuleSystem([100.0, 178.0], [100.0, 100.0], [40.0, 40.0], [0, 1], [8, 0], p=p)
        F, _, contacts = compute_forces(gs, p, np.random.default_rng(1))
        c = contacts[0]
        E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - p.poisson_ratio ** 2))
        self.assertEqual(c['E_star'], E_star_gg)
        self.assertEqual(c['W'], p.W_adh_if)
        self.assertEqual(c['tau_0'], p.tau_0_if)


class TestBridgingEligibility(unittest.TestCase):

    def _run_bridging(self, species_id, seed=3):
        p = _quiet_params(bridge_attempt_rate=1e6, min_fa_for_bridge=0.0, t_spread_duration=0.1,
                          fa_maturation_rate=100.0, cell_sense_distance=60.0)
        gs = _row_system(species_id, p, overlap=-5.0)      # 5 µm gaps, within sensing range
        rng = np.random.default_rng(seed)
        update_cell_state(gs, p, 5.0, rng)                  # spread + mature FAs
        compute_forces(gs, p, rng)
        return gs

    def test_no_bridges_to_bare_target(self):
        gs = self._run_bridging([0, 2, 0])   # coated | bare | coated
        host = gs.cells_on_granule(0)
        states = set(int(s) for s in gs.cell_state[host])
        self.assertFalse(states & {int(CellState.BRIDGING), int(CellState.MIGRATING)},
                         "cells on a coated granule must not bridge toward a bare one")
        self.assertTrue(np.all(gs.cell_bridge_target[host] == -1))

    def test_bridges_to_partially_coated_target(self):
        gs = self._run_bridging([0, 1, 0])   # coated | half | coated
        host = gs.cells_on_granule(0)
        states = set(int(s) for s in gs.cell_state[host])
        self.assertTrue(states & {int(CellState.BRIDGING), int(CellState.MIGRATING)},
                        "cells should attempt bridges toward a half-coated granule")
        targets = set(int(t) for t in gs.cell_bridge_target[host] if t >= 0)
        self.assertEqual(targets, {1})

    def test_bridge_force_scales_with_target_coverage(self):
        p = Params()
        g_half = M.traction_gain(1.0, 0.5, M.LAW_LANGMUIR, M.RULE_TARGET, 1.0,
                                 p.K_sigma_traction / p.sigma_ligand_max)
        F_full = motor_clutch_force(p.E_modulus, p, 1.0, g=1.0)
        F_half = motor_clutch_force(p.E_modulus, p, 1.0, g=g_half)
        self.assertLess(F_half, F_full)
        self.assertGreater(F_half, 0.5 * F_full)   # Langmuir: 50% coating gives > 50% force


class TestBlockerPenalty(unittest.TestCase):

    def _factor(self, blocker_species):
        p = _quiet_params(Lx=600.0, Ly=300.0)
        x = np.array([100.0, 400.0, 250.0])
        y = np.array([100.0, 100.0, 120.0])   # blocker 20 µm off the i→j line, r = 40 → intersects
        r = np.full(3, 40.0)
        sid = np.array([0, 0, blocker_species])
        nc = np.array([8, 8, 8 if SPECIES[blocker_species]['f'] > 0 else 0])
        gs = GranuleSystem(x, y, r, None, nc, species_id=sid, species=[dict(s) for s in SPECIES], p=p)
        pos_i = np.array([x[0], y[0]])
        pos_j = np.array([x[1], y[1]])
        gap = 300.0 - 80.0
        pf = reference._classify_bridge_path(gs, p, 0, 1, pos_i, pos_j, gap)
        void = np.exp(-gap / p.bridge_decay_length)
        return pf / void, p

    def test_bare_half_and_coated_blockers(self):
        f_bare, p = self._factor(2)
        f_half, _ = self._factor(1)
        f_coated, _ = self._factor(0)
        self.assertAlmostEqual(f_bare, p.bridge_inert_factor, places=12)
        self.assertAlmostEqual(f_coated, 1.0, places=12)
        self.assertAlmostEqual(f_half, p.bridge_inert_factor + (1 - p.bridge_inert_factor) * 0.5, places=9)


class TestSeedingAndCapacity(unittest.TestCase):

    def test_seeded_cells_and_capacity_scale_with_f(self):
        p = Params(cell_surface_coverage=1.0, species=[dict(s) for s in SPECIES])
        n_full = cells_from_surface_coverage(40.0, p.cell_diameter, p.cell_height_spread, 1.0, '2D')
        self.assertEqual(seeded_cells(n_full, 1.0, p), n_full)
        self.assertEqual(seeded_cells(n_full, 0.0, p), 0)
        # n_full is only 4 on a 40 um 2D granule, so the strict drop shows at
        # f = 0.2 (gain 0.80 -> 3 cells); at f = 0.5 (gain 0.94) rounding keeps 4
        n_20 = seeded_cells(n_full, 0.2, p)
        self.assertTrue(0 < n_20 < n_full, (n_20, n_full))
        n_half = seeded_cells(n_full, 0.5, p)
        self.assertTrue(0 < n_half <= n_full)
        gs = _row_system([0, 1, 2], p, n_cells=n_full)
        self.assertEqual(cell_capacity(gs, 0, p), n_full)
        self.assertEqual(cell_capacity(gs, 1, p), n_half)
        self.assertEqual(cell_capacity(gs, 2, p), 0)


if __name__ == '__main__':
    unittest.main()
