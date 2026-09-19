"""
Setup presets and the well calibration sweep (V3.1).
====================================================

A preset is a bundle of correlated overrides (container + material + gravity +
packing, or the whole cell model). The risk with a bundle is that one of its
dotted paths silently stops resolving, so the central test is that every
preset still applies, validates and produces a usable Params.

Run:  python -m unittest tests.test_presets -v
"""

import copy
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE, os.path.join(REPO, 'pipeline')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels import config as C  # noqa: E402
from gels.presets import (  # noqa: E402
    PRESET_DESCRIPTIONS, PRESETS, apply_presets, preset_deltas, preset_names, resolve_presets,
)

try:
    import yaml  # noqa: F401
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestPresets(unittest.TestCase):

    def test_every_preset_applies_and_validates(self):
        for name in PRESETS:
            with self.subTest(preset=name):
                s = C.template_setup()
                apply_presets(s, name)
                C.validate(s)
                s.to_params()          # every path must reach a real Params field

    def test_every_preset_is_described_and_listed_in_the_template(self):
        for name in PRESETS:
            self.assertIn(name, PRESET_DESCRIPTIONS, name)
            self.assertIn(name, C.TEMPLATE_YAML, f"{name} missing from the template's preset block")

    def test_idempotent(self):
        once = C.template_setup()
        apply_presets(once, 'pmma_well')
        twice = C.template_setup()
        apply_presets(twice, 'pmma_well')
        apply_presets(twice, 'pmma_well')
        self.assertEqual(C.setup_to_dict(once), C.setup_to_dict(twice))
        self.assertEqual(twice.meta.presets, ['pmma_well'])

    def test_comma_and_list_forms_agree(self):
        a = C.template_setup(); apply_presets(a, 'pmma_well,fibroblast_realistic')
        b = C.template_setup(); apply_presets(b, ['pmma_well', 'fibroblast_realistic'])
        self.assertEqual(C.setup_to_dict(a), C.setup_to_dict(b))
        self.assertEqual(a.meta.presets, ['pmma_well', 'fibroblast_realistic'])

    def test_unknown_preset_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            resolve_presets('not_a_preset')
        self.assertIn('not_a_preset', str(cm.exception))
        s = C.template_setup()
        s.meta.presets = ['not_a_preset']
        with self.assertRaises(ValueError):
            C.validate(s)

    def test_legacy_preset_restores_the_v30_defaults(self):
        """hydrogel_box_legacy must undo pmma_well + fibroblast_realistic."""
        s = C.template_setup()
        apply_presets(s, 'pmma_well,fibroblast_realistic')
        apply_presets(s, 'hydrogel_box_legacy')
        p = s.to_params()
        self.assertEqual(p.boundary_shape, 'box')
        self.assertEqual(p.boundary_top, 'wall')
        self.assertFalse(p.gravity_enabled)
        self.assertEqual(p.packing_consolidation, 'centre')
        self.assertEqual(p.bridge_force_model, 'constant')
        self.assertFalse(p.cell_division_enabled)
        self.assertEqual(p.E_modulus, 10.0)
        self.assertEqual(p.contact_E_cap, 0.0)
        self.assertEqual(p.friction_mu, 0.0)

    def test_pmma_well_geometry_and_material(self):
        s = C.template_setup()
        apply_presets(s, 'pmma_well')
        p = s.to_params()
        self.assertEqual(p.boundary_shape, 'cylinder')
        self.assertEqual(p.boundary_top, 'free')
        self.assertTrue(p.gravity_enabled)
        self.assertEqual(p.granule_density, 1180.0)
        self.assertGreater(p.contact_E_cap, 0.0)
        self.assertFalse(p.mc_dem_enabled)
        self.assertGreater(p.friction_mu, 0.0)
        # the bed height sets the amount: V_solid = phi_bed * A_base * H_bed
        self.assertAlmostEqual(p.phi_solid_target, 0.60 * 1500.0 / 2100.0, places=12)

    def test_set_beats_a_preset(self):
        s = C.template_setup()
        apply_presets(s, 'pmma_well')
        C.apply_overrides(s, ['contact.friction_mu=0.9'])
        self.assertEqual(s.to_params().friction_mu, 0.9)

    def test_deltas_describe_real_changes(self):
        base = C.template_setup()
        deltas = preset_deltas(copy.deepcopy(base), 'pmma_well')
        self.assertGreater(len(deltas), 10)
        for name, path, old, new in deltas:
            self.assertEqual(name, 'pmma_well')
            self.assertNotEqual(old, new, path)
            C.get_path(base, path)         # the path must resolve on a stock Setup

    def test_preset_names_helper(self):
        self.assertEqual(preset_names('a,b'), ['a', 'b'])
        self.assertEqual(preset_names(['a', 'b,c']), ['a', 'b', 'c'])
        self.assertEqual(preset_names(None), [])


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestSavedSetupCarriesProvenance(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_preset_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_header_and_round_trip(self):
        s = C.template_setup()
        apply_presets(s, 'pmma_dish_slice,fibroblast_realistic')
        path = os.path.join(self.tmp, 'setup.yaml')
        C.save_setup(s, path, header=['written by a test', 'second line'])
        with open(path, encoding='utf-8') as fh:
            text = fh.read()
        self.assertTrue(text.startswith('# written by a test'))
        self.assertIn('# second line', text)
        back = C.load_setup(path)
        self.assertEqual(C.setup_to_dict(back), C.setup_to_dict(s))
        self.assertEqual(back.meta.presets, ['pmma_dish_slice', 'fibroblast_realistic'])


@unittest.skipUnless(HAS_YAML, "PyYAML not installed")
class TestWellSweep(unittest.TestCase):

    def setUp(self):
        from run_showcase import build_well
        self.conds = build_well()

    def test_keys_unique_and_setups_validate(self):
        keys = [c.key for c in self.conds]
        self.assertEqual(len(keys), len(set(keys)), keys)
        self.assertGreaterEqual(len(keys), 8)
        from run_showcase import build_setup
        for c in self.conds:
            with self.subTest(cond=c.key):
                s = build_setup(c, 4.0, 1)
                C.validate(s)
                p = s.to_params()
                self.assertEqual(p.boundary_shape, c.boundary_shape)
                self.assertEqual(p.boundary_top, c.boundary_top)
                self.assertEqual(list(s.meta.presets), list(c.presets))

    def test_counts_match_the_bed_volume(self):
        import math
        for c in self.conds:
            if not c.bed_height_um:
                continue
            with self.subTest(cond=c.key):
                phi_bed = 0.60 if c.mode == '3D' else 0.80
                V = phi_bed * c.base_area() * c.bed_height_um
                mu, sd, _ = c.species[0]['radius']
                if c.mode == '3D':
                    mean_vol = (4.0 / 3.0) * math.pi * (mu ** 3 + 3 * mu * sd ** 2)
                else:
                    mean_vol = math.pi * (mu ** 2 + sd ** 2)
                expected = round(V * c.species[0]['vf'] / mean_vol)
                self.assertAlmostEqual(c.estimated_counts()[0], expected, delta=2)
                self.assertEqual(c.factors['n_estimated'], sum(c.estimated_counts()))

    def test_the_set_covers_the_intended_contrasts(self):
        """The point of the set is that each factor varies alone."""
        by = {c.key: c.factors for c in self.conds}
        self.assertEqual(by['dish_const_nodiv']['force_model'], 'constant')
        self.assertFalse(by['dish_const_nodiv']['division'])
        self.assertEqual(by['dish_hill_nodiv']['force_model'], 'hill')
        self.assertFalse(by['dish_hill_nodiv']['division'])
        self.assertTrue(by['dish_const_div']['division'])
        self.assertEqual(by['dish_hill_div']['wall'], 'inert')
        self.assertEqual(by['dish_hill_div_fwall']['wall'], 'functionalized')
        self.assertEqual(by['box_legacy_hill_div']['geometry'], 'closed box')
        self.assertEqual(by['well3d_hill_div']['geometry'], 'well')

    def test_cylinder_only_in_3d(self):
        for c in self.conds:
            if c.boundary_shape == 'cylinder':
                self.assertEqual(c.mode, '3D', c.key)

    def test_registered_as_a_sweep(self):
        from run_showcase import SWEEPS, SWEEP_TITLES
        self.assertIn('well', SWEEPS)
        self.assertIn('well', SWEEP_TITLES)
        builder, out, setups = SWEEPS['well']
        self.assertTrue(out.endswith('well_calibration'))
        self.assertTrue(setups.endswith('well_calibration'))
        self.assertEqual(len(builder()), len(self.conds))


if __name__ == '__main__':
    unittest.main()
