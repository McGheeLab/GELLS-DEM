"""
Sectioned setup (gels/config.py) — coverage, round trips, template, overrides.

  * every Params field is reachable from exactly one place (FLAT_MAP, the
    species list, or is explicitly deprecated / pipeline-managed),
  * Setup() -> to_params() reproduces Params() defaults; Params -> Setup ->
    Params is idempotent; a legacy trial converts to two species with the
    original scalar values intact,
  * the commented template parses, validates and carries the recommended
    physical defaults; YAML and JSON forms load identically,
  * dotted --set overrides are coerced to the field type and rejected when
    the path is unknown.

Run:  python -m unittest tests.test_config_roundtrip -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from dataclasses import fields

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import Params  # noqa: E402
from gels import config as C  # noqa: E402

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class TestCoverage(unittest.TestCase):

    def test_every_params_field_mapped_exactly_once(self):
        mapped = [f for _, f in C.FLAT_MAP]
        self.assertEqual(len(mapped), len(set(mapped)), "duplicate FLAT_MAP targets")
        allf = {f.name for f in fields(Params)}
        covered = set(mapped) | C.SPECIES_DERIVED | C.DEPRECATED | C.PIPELINE_MANAGED
        self.assertEqual(allf - covered, set(), "Params fields not reachable from a setup")
        self.assertEqual(covered - allf, set(), "FLAT_MAP refers to unknown Params fields")
        self.assertEqual(set(mapped) & C.SPECIES_DERIVED, set())

    def test_every_flat_path_resolves(self):
        s = C.Setup()
        for path, _ in C.FLAT_MAP:
            C.get_path(s, path)


class TestRoundTrips(unittest.TestCase):

    def test_default_setup_reproduces_default_params(self):
        p0 = Params()
        p1 = C.Setup().to_params()
        for _, f in C.FLAT_MAP:
            if f == 'phi_solid_target':
                continue      # a Setup always states the solid fraction explicitly
            self.assertEqual(getattr(p0, f), getattr(p1, f), f)
        # default Params give phi_f/phi_i directly (phi_solid_target = 0); the
        # sectioned form expresses the same composition as solid × fractions
        self.assertAlmostEqual(p1.phi_solid_target, p0.phi_f_target + p0.phi_i_target, places=12)
        self.assertAlmostEqual(p1.phi_f_target, p0.phi_f_target, places=12)
        self.assertAlmostEqual(p1.phi_i_target, p0.phi_i_target, places=12)
        self.assertAlmostEqual(p1.func_ratio, p0.phi_f_target / (p0.phi_f_target + p0.phi_i_target), places=12)
        for f in sorted(C.SPECIES_DERIVED - {'species', 'phi_f_target', 'phi_i_target', 'func_ratio'}):
            self.assertEqual(getattr(p0, f), getattr(p1, f), f)
        self.assertEqual(len(p1.species), 2)
        self.assertEqual([s['f'] for s in p1.species], [1.0, 0.0])

    def test_params_setup_params_idempotent(self):
        p = Params(mode='3D', Lx=500.0, E_modulus=12.0, phi_solid_target=0.7, func_ratio=0.6,
                   R_func_mean=45.0, cell_migration_speed=25.0, tau_0_cc=1500.0, perf_threads=8)
        s1 = C.Setup.from_params(p)
        p2 = s1.to_params()
        s2 = C.Setup.from_params(p2)
        self.assertEqual(C.setup_to_dict(s1), C.setup_to_dict(s2))
        for _, f in C.FLAT_MAP:
            self.assertEqual(getattr(p, f), getattr(p2, f), f)
        # phi_solid_target > 0: the composition survives exactly (same products as __post_init__)
        self.assertEqual(p2.func_ratio, 0.6)
        self.assertEqual(p2.phi_f_target, 0.7 * 0.6)
        self.assertEqual(p2.R_func_mean, 45.0)
        # legacy species keep the V2.7 count rule at the species level; the
        # granules-level rule (a Params field) is untouched
        self.assertEqual([sp.count_rule for sp in s1.granules.species], ['mean_radius', 'mean_radius'])
        self.assertEqual([d['count_rule'] for d in p2.species], ['mean_radius', 'mean_radius'])
        self.assertEqual(s1.granules.count_rule, p.packing_count_rule)
        self.assertEqual([d['phi_target'] for d in p2.species], [0.7 * 0.6, 0.7 * (1.0 - 0.6)])

    def test_legacy_trial_converts_to_two_species(self):
        sys.path.insert(0, os.path.join(REPO, 'pipeline'))
        from _common import load_trial_json
        p = Params()
        for k, v in load_trial_json(os.path.join(REPO, 'Trials', 'default_trial.json')).items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))
        s = C.Setup.from_legacy_params(p)
        self.assertEqual(len(s.granules.species), 2)
        self.assertEqual(s.granules.species[0].functionalization, 1.0)
        self.assertEqual(s.granules.species[1].functionalization, 0.0)
        self.assertAlmostEqual(s.granules.species[0].volume_fraction, 0.6)
        self.assertEqual(s.granules.species[1].radius.mean_um, 60.0)
        self.assertEqual(s.contact.adhesion_energy_J_per_m2.cc, 0.002)
        p2 = s.to_params()
        for f in ('E_modulus', 'poisson_ratio', 'n_motors', 'F_max_per_cell', 'cell_surface_coverage',
                  'dt', 't_total', 'shape_enabled', 'aspect_ratio_func_mean', 'blockiness_n2_inert_std',
                  'W_adh_cc', 'tau_0_bb', 'R_func_mean', 'R_inert_mean', 'func_ratio', 'phi_solid_target'):
            self.assertEqual(getattr(p, f), getattr(p2, f), f)

    def test_legacy_conversion_reproduces_v27_params_fixtures(self):
        """Setup.from_legacy_params(x).to_params() keeps every scalar of the V2.7 params fixtures."""
        import glob
        fixtures = sorted(glob.glob(os.path.join(HERE, 'fixtures', 'params_v27_*.json')))
        if not fixtures:
            self.skipTest("no params fixtures")
        aliases = getattr(Params, 'LEGACY_ALIASES', {})
        skip = {'output_dir', 'resume_from', 'species'}
        for path in fixtures:
            with open(path) as fh:
                ref = json.load(fh)
            p = Params()
            for k, v in ref.items():
                if hasattr(p, k):
                    try:
                        setattr(p, k, type(getattr(p, k))(v))
                    except (ValueError, TypeError):
                        setattr(p, k, v)
            p2 = C.Setup.from_legacy_params(p).to_params()
            for k, v in ref.items():
                if k in skip:
                    continue
                key = k if hasattr(Params, k) and not isinstance(getattr(Params, k, None), property) else aliases.get(k, k)
                got = getattr(p2, key)
                if isinstance(v, float):
                    self.assertAlmostEqual(v, got, places=12, msg=f"{os.path.basename(path)}: {k}")
                else:
                    self.assertEqual(v, got, f"{os.path.basename(path)}: {k}")

    def test_species_dicts_are_json_safe(self):
        p = C.template_setup().to_params() if HAS_YAML else C.Setup().to_params()
        json.dumps(p.species)
        for d in p.species:
            self.assertIn('phi_target', d)
            self.assertIn('radius_min', d)


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestTemplateAndFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_cfg_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_template_parses_and_carries_physical_defaults(self):
        t = C.template_setup()
        t.validate()
        p = t.to_params()
        self.assertEqual(len(t.granules.species), 3)
        self.assertEqual(p.cell_migration_speed, 30.0)
        self.assertEqual(p.n_motors, 200)
        self.assertEqual(p.F_max_per_cell, 150.0)
        self.assertEqual(p.traction_f_law, 'langmuir')
        self.assertEqual(p.sigma_ligand_max, 750.0)
        self.assertEqual(p.K_sigma_traction, 160.0)
        self.assertAlmostEqual(sum(s['volume_fraction'] for s in p.species), 1.0)
        self.assertAlmostEqual(p.species[1]['phi_target'], 0.65 * 0.4)

    def test_yaml_and_json_load_identically(self):
        t = C.template_setup()
        ypath = os.path.join(self.tmp, 'setup.yaml')
        jpath = os.path.join(self.tmp, 'setup.json')
        C.save_setup(t, ypath)
        C.save_setup(t, jpath)
        a = C.load_setup(ypath)
        b = C.load_setup(jpath)
        self.assertEqual(C.setup_to_dict(a), C.setup_to_dict(b))
        self.assertEqual(C.setup_to_dict(a), C.setup_to_dict(t))

    def test_unknown_key_is_rejected(self):
        path = os.path.join(self.tmp, 'bad.yaml')
        with open(path, 'w') as fh:
            fh.write("domain: {mode: 2D, size_um: [400, 400, 400], sizes: 3}\n")
        with self.assertRaises(ValueError) as cm:
            C.load_setup(path)
        self.assertIn('sizes', str(cm.exception))

    def test_validation_catches_bad_volume_fractions(self):
        t = C.template_setup()
        t.granules.species[0].volume_fraction = 0.5
        with self.assertRaises(ValueError):
            t.validate()


class TestOverrides(unittest.TestCase):

    def test_dotted_overrides_and_coercion(self):
        s = C.Setup()
        C.apply_overrides(s, ['time.t_total_h=12', 'performance.threads=8',
                              'granules.species[1].functionalization=0.3',
                              'domain.size_um[0]=1000', 'contact.mc_dem.enabled=false',
                              'boundary.mode=periodic'])
        self.assertEqual(s.time.t_total_h, 12.0)
        self.assertIsInstance(s.time.t_total_h, float)
        self.assertEqual(s.performance.threads, 8)
        self.assertIsInstance(s.performance.threads, int)
        self.assertEqual(s.granules.species[1].functionalization, 0.3)
        self.assertEqual(s.domain.size_um[0], 1000.0)
        self.assertIs(s.contact.mc_dem.enabled, False)
        self.assertEqual(s.boundary.mode, 'periodic')

    def test_bad_paths_are_rejected(self):
        s = C.Setup()
        with self.assertRaises(ValueError):
            C.apply_overrides(s, ['cells.nope=1'])
        with self.assertRaises(ValueError):
            C.apply_overrides(s, ['contact.mc_dem=1'])       # a section, not a value
        with self.assertRaises(ValueError):
            C.apply_overrides(s, ['time.t_total_h'])          # no '='


if __name__ == '__main__':
    unittest.main()
