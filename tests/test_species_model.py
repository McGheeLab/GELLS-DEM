"""
Species / functionalization data model (V3.0 Phase 1).

Checks that
  * legacy construction (gtype only) yields the two-species table and an
    identical derived gtype,
  * an explicit multi-species list maps f / E / ν per granule and derives
    gtype from f_min_adhesion,
  * snapshots round-trip species arrays through save/restore, and a V2.7
    snapshot (no species_id) restores through its gtype,
  * the renamed adhesion/friction fields are reachable under their old names
    everywhere the pipeline reads them (params.json, trial JSON, CLI flags).

Run:  python -m unittest tests.test_species_model -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import (  # noqa: E402
    GranuleSystem, Params, legacy_species_from_params, resolve_species,
    restore_gs_from_snapshot, save_snapshot_to_disk, seeding_gain,
)

THREE_SPECIES = [
    dict(name='Granule_1', f=1.0, volume_fraction=0.2, color='#CC2222',
         radius_mean=40.0, radius_std=5.0, E_kPa=None, poisson_ratio=None),
    dict(name='Granule_2', f=0.5, volume_fraction=0.4, color='#2255CC',
         radius_mean=40.0, radius_std=5.0, E_kPa=5.0, poisson_ratio=0.40),
    dict(name='Granule_3', f=0.0, volume_fraction=0.4, color='#22AA22',
         radius_mean=60.0, radius_std=8.0, E_kPa=20.0, poisson_ratio=None),
]


def _tiny_gs(species_id=None, species=None, p=None, gtype=(0, 0, 1, 0)):
    x = np.array([50.0, 150.0, 250.0, 350.0])
    y = np.array([60.0, 60.0, 60.0, 60.0])
    r = np.array([40.0, 40.0, 60.0, 40.0])
    n_cells = np.array([4, 4, 0, 4])
    return GranuleSystem(x, y, r, np.array(gtype), n_cells,
                         species_id=species_id, species=species, p=p)


class TestLegacyConstruction(unittest.TestCase):

    def test_gtype_only_gives_two_species_and_identical_gtype(self):
        gs = _tiny_gs()
        self.assertEqual(gs.K, 2)
        self.assertEqual(gs.species_names, ['collagen', 'bare'])
        np.testing.assert_array_equal(gs.species_id, [0, 0, 1, 0])
        np.testing.assert_array_equal(gs.gtype, [0, 0, 1, 0])
        np.testing.assert_array_equal(gs.f, [1.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(gs.func_mask, [True, True, False, True])
        self.assertEqual(gs.gtype.dtype, np.dtype(int))
        # tables reproduce the V2.7 LUT
        p = Params()
        self.assertEqual(gs.pair_W[0, 0], p.W_adh_ff)
        self.assertEqual(gs.pair_W[0, 1], p.W_adh_if)
        self.assertEqual(gs.pair_W[1, 1], p.W_adh_ii)
        self.assertEqual(gs.pair_tau[0, 1], p.tau_0_if)
        self.assertEqual(gs.wall_W[0], p.W_adh_if)   # wall LUT {0: W_if, 1: W_ii}
        self.assertEqual(gs.wall_W[1], p.W_adh_ii)
        E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - p.poisson_ratio ** 2))
        self.assertAlmostEqual(gs.pair_Estar[0, 1], E_star_gg, places=9)

    def test_legacy_species_from_params_follows_composition(self):
        p = Params(phi_solid_target=0.7, func_ratio=0.6, R_func_mean=45.0, R_inert_std=3.0)
        sp = legacy_species_from_params(p)
        self.assertEqual([s['f'] for s in sp], [1.0, 0.0])
        self.assertAlmostEqual(sp[0]['volume_fraction'], 0.6)
        self.assertAlmostEqual(sp[1]['volume_fraction'], 0.4)
        self.assertEqual(sp[0]['radius_mean'], 45.0)
        self.assertEqual(sp[1]['radius_std'], 3.0)
        self.assertEqual((sp[0]['radius_min'], sp[1]['radius_min']), (15.0, 20.0))
        # plain scalars only → JSON-safe
        json.dumps(sp)
        # resolve caches on p
        self.assertEqual(resolve_species(p), sp)
        self.assertIs(resolve_species(p), p.species)


class TestMultiSpecies(unittest.TestCase):

    def test_explicit_species_map_per_granule(self):
        p = Params()
        gs = _tiny_gs(species_id=[0, 1, 2, 1], species=THREE_SPECIES, p=p)
        self.assertEqual(gs.K, 3)
        np.testing.assert_array_equal(gs.f, [1.0, 0.5, 0.0, 0.5])
        np.testing.assert_array_equal(gs.E_gran, [p.E_modulus, 5.0, 20.0, 5.0])
        np.testing.assert_array_equal(gs.nu_gran, [p.poisson_ratio, 0.40, p.poisson_ratio, 0.40])
        # derived binary view: adhesive (f >= f_min) → 0, bare → 1
        np.testing.assert_array_equal(gs.gtype, [0, 0, 1, 0])
        np.testing.assert_array_equal(gs.adhesive_mask, [True, True, False, True])
        # mixed-E pair modulus
        from gels.materials import pair_E_star_Pa
        self.assertAlmostEqual(gs.pair_Estar[1, 2], pair_E_star_Pa(5.0, 0.40, 20.0, p.poisson_ratio), places=6)
        # f_min threshold moves the derived gtype
        p2 = Params(f_min_adhesion=0.6)
        gs2 = _tiny_gs(species_id=[0, 1, 2, 1], species=THREE_SPECIES, p=p2)
        np.testing.assert_array_equal(gs2.gtype, [0, 1, 1, 1])

    def test_species_id_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            _tiny_gs(species_id=[0, 3, 2, 1], species=THREE_SPECIES)

    def test_seeding_gain_limits(self):
        p = Params()
        g = seeding_gain(np.array([0.0, 0.2, 1.0]), p)
        self.assertEqual(g[0], 0.0)
        self.assertEqual(g[2], 1.0)
        # attachment saturates faster than traction: with K_attach = 50 and
        # sigma_max = 750 (kappa = 1/15), 20% coating keeps 80% of the cells
        self.assertAlmostEqual(g[1], 0.80, places=2)
        pp = Params(cell_seeding_law='power', cell_seeding_exponent=1.0)
        np.testing.assert_allclose(seeding_gain(np.array([0.0, 0.2, 1.0]), pp), [0.0, 0.2, 1.0])
        # activity on the system equals the seeding gain of f
        gs = _tiny_gs(species_id=[0, 1, 2, 1], species=THREE_SPECIES, p=p)
        np.testing.assert_allclose(gs.activity, seeding_gain(gs.f, p))


class TestSnapshotRoundTrip(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_species_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_three_species_survive_save_and_restore(self):
        p = Params(species=[dict(s) for s in THREE_SPECIES])
        gs = _tiny_gs(species_id=[0, 1, 2, 1], species=p.species, p=p)
        F = np.zeros((gs.N, 2))
        save_snapshot_to_disk(0, gs, p, 0.0, F, self.tmp)
        path = os.path.join(self.tmp, 'snapshots', 'snap_0000.npz')
        data = dict(np.load(path, allow_pickle=False))
        for key in ('species_id', 'f', 'E_gran', 'nu_gran', 'gtype'):
            self.assertIn(key, data)
        gs2, t, idx = restore_gs_from_snapshot(path, p)
        np.testing.assert_array_equal(gs2.species_id, gs.species_id)
        np.testing.assert_array_equal(gs2.f, gs.f)
        np.testing.assert_array_equal(gs2.E_gran, gs.E_gran)
        np.testing.assert_array_equal(gs2.gtype, gs.gtype)
        self.assertEqual(gs2.species_names, gs.species_names)

    def test_v27_snapshot_restores_via_gtype(self):
        fixture = os.path.join(HERE, 'fixtures', 'run2d_walls', 'snapshots', 'snap_0001.npz')
        if not os.path.exists(fixture):
            self.skipTest("fixtures not generated")
        # The fixtures are produced by the current engine (regenerated 2026-09-17 for the
        # displacement fix), so they carry the V3.0 keys; a genuinely pre-V3.0 snapshot is
        # the same file with every key that did not exist in V2.7 removed.
        data = dict(np.load(fixture, allow_pickle=False))
        for key in ('species_id', 'f', 'E_gran', 'nu_gran'):
            data.pop(key, None)
        legacy = os.path.join(self.tmp, 'snap_0001.npz')
        np.savez(legacy, **data)
        self.assertNotIn('species_id', dict(np.load(legacy, allow_pickle=False)))
        p = Params(mode='2D', Lx=400.0, Ly=400.0)
        gs, t, idx = restore_gs_from_snapshot(legacy, p)
        np.testing.assert_array_equal(gs.species_id, data['gtype'])
        np.testing.assert_array_equal(gs.gtype, data['gtype'])
        np.testing.assert_array_equal(gs.f, 1.0 - data['gtype'])
        self.assertEqual(gs.species_names, ['collagen', 'bare'])

    def test_from_snapshot_dict(self):
        p = Params(species=[dict(s) for s in THREE_SPECIES])
        gs = _tiny_gs(species_id=[0, 1, 2, 1], species=p.species, p=p)
        snap = {'x': gs.x.copy(), 'y': gs.y.copy(), 'r': gs.r.copy(), 'gtype': gs.gtype.copy(),
                'species_id': gs.species_id.copy(), 'f': gs.f.copy(), 'cell_offset': gs.cell_offset.copy()}
        gs2 = GranuleSystem.from_snapshot_dict(snap, p)
        np.testing.assert_array_equal(gs2.species_id, gs.species_id)
        np.testing.assert_array_equal(gs2.f, gs.f)
        self.assertTrue(gs2.is_circle)
        self.assertEqual(gs2.total_cells, gs.total_cells)


class TestLegacyAliases(unittest.TestCase):

    def test_alias_properties_read_and_write(self):
        p = Params()
        self.assertEqual(p.W_adh_ff, p.W_adh_cc)
        p.W_adh_ff = 0.01
        self.assertEqual(p.W_adh_cc, 0.01)
        p.tau_0_ii = 75.0
        self.assertEqual(p.tau_0_bb, 75.0)
        d = asdict(p)
        self.assertIn('W_adh_cc', d)
        self.assertNotIn('W_adh_ff', d)             # aliases are not fields
        self.assertEqual(d['species'], [])

    def test_trial_json_keeps_legacy_pair_values(self):
        sys.path.insert(0, os.path.join(REPO, 'pipeline'))
        from _common import load_trial_json
        overrides = load_trial_json(os.path.join(REPO, 'Trials', 'default_trial.json'))
        # the old names are passed through instead of being dropped with a warning
        self.assertIn('W_adh_ff', overrides)
        self.assertIn('tau_0_ii', overrides)
        p = Params()
        for k, v in overrides.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        self.assertEqual(p.W_adh_cc, 0.002)
        self.assertEqual(p.tau_0_bb, 50.0)

    def test_cli_alias_flag(self):
        import argparse
        sys.path.insert(0, os.path.join(REPO, 'pipeline'))
        from _common import add_params_args, apply_params_args
        parser = add_params_args(argparse.ArgumentParser())
        args = parser.parse_args(['--W_adh_ff', '0.003', '--tau_0_cc', '1500'])
        p = Params()
        changed = apply_params_args(p, args)
        self.assertEqual(p.W_adh_cc, 0.003)
        self.assertEqual(p.tau_0_cc, 1500.0)
        self.assertIn('W_adh_cc', changed)


if __name__ == '__main__':
    unittest.main()
