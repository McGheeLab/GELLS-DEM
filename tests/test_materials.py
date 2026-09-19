"""
Unit tests for gels/materials.py — the functionalization rules.

Every rule must reduce exactly to the V2.7 binary model at f ∈ {0, 1}; the
Langmuir gain must have the documented limits; and the pair tables for the
legacy two species must reproduce the old friction/adhesion LUT bit for bit.

Run:  python -m unittest tests.test_materials -v
"""

import os
import sys
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from gels import materials as M  # noqa: E402

# V2.7 defaults (engine.Params): W_adh_ii/if/ff, tau_0_ii/if/ff
W_BB, W_CB, W_CC = 0.0005, 0.001, 0.002
T_BB, T_CB, T_CC = 50.0, 500.0, 2000.0


class TestPairMixing(unittest.TestCase):

    def test_endpoints_reproduce_lut(self):
        self.assertEqual(M.mix_pair(1.0, 1.0, W_CC, W_CB, W_BB), W_CC)
        self.assertEqual(M.mix_pair(0.0, 0.0, W_CC, W_CB, W_BB), W_BB)
        self.assertEqual(M.mix_pair(1.0, 0.0, W_CC, W_CB, W_BB), W_CB)
        self.assertEqual(M.mix_pair(0.0, 1.0, W_CC, W_CB, W_BB), W_CB)
        self.assertEqual(M.mix_wall(1.0, W_CB, W_BB), W_CB)   # wall = bare partner
        self.assertEqual(M.mix_wall(0.0, W_CB, W_BB), W_BB)

    def test_symmetric_and_bounded(self):
        for fi, fj in [(0.2, 0.7), (0.5, 0.5), (0.9, 0.1)]:
            a = M.mix_pair(fi, fj, W_CC, W_CB, W_BB)
            b = M.mix_pair(fj, fi, W_CC, W_CB, W_BB)
            self.assertAlmostEqual(a, b, places=15)
            self.assertGreaterEqual(a, W_BB)
            self.assertLessEqual(a, W_CC)

    def test_effective_modulus_matches_v27(self):
        E, nu = 10.0, 0.45
        self.assertAlmostEqual(M.pair_E_star_Pa(E, nu, E, nu),
                               (E * 1e3) / (2.0 * (1.0 - nu ** 2)), places=9)
        self.assertAlmostEqual(M.wall_E_star_Pa(E, nu), (E * 1e3) / (1.0 - nu ** 2), places=9)
        # mixed stiffness: harmonic-type combination lies between the two
        e_soft = M.pair_E_star_Pa(5.0, nu, 5.0, nu)
        e_stiff = M.pair_E_star_Pa(20.0, nu, 20.0, nu)
        e_mix = M.pair_E_star_Pa(5.0, nu, 20.0, nu)
        self.assertGreater(e_mix, e_soft)
        self.assertLess(e_mix, e_stiff)


class TestLigandFactor(unittest.TestCase):
    KAPPA = 160.0 / 750.0   # K_sigma_traction / sigma_ligand_max defaults

    def test_limits(self):
        self.assertEqual(M.ligand_factor(0.0, self.KAPPA), 0.0)
        self.assertEqual(M.ligand_factor(1.0, self.KAPPA), 1.0)
        # kappa -> inf recovers the linear law
        self.assertAlmostEqual(M.ligand_factor(0.3, 1e9), 0.3, places=6)
        # kappa -> 0 is a step for any positive coverage
        self.assertEqual(M.ligand_factor(0.01, 0.0), 1.0)
        self.assertEqual(M.ligand_factor(0.0, 0.0), 0.0)

    def test_review_prediction_20_percent_coating(self):
        # From the ligand-density review: sigma_max = 750, K = 160 -> g(0.2) ≈ 0.59
        self.assertAlmostEqual(M.ligand_factor(0.2, self.KAPPA), 0.587, places=2)
        self.assertAlmostEqual(M.ligand_factor(0.5, self.KAPPA), 0.855, places=2)
        # A dense RGD monolayer (sigma_max ~ 4e4) makes 20% dilution nearly invisible
        self.assertGreater(M.ligand_factor(0.2, 160.0 / 4e4), 0.97)

    def test_strictly_monotone(self):
        f = np.linspace(0.0, 1.0, 201)
        g = np.array([M.ligand_factor(x, self.KAPPA) for x in f])
        self.assertTrue(np.all(np.diff(g) > 0))


class TestTractionGain(unittest.TestCase):
    KAPPA = 160.0 / 750.0

    def test_binary_endpoints_for_all_laws_and_rules(self):
        for law in (M.LAW_LANGMUIR, M.LAW_POWER):
            for rule in (M.RULE_TARGET, M.RULE_MIN, M.RULE_PRODUCT):
                self.assertEqual(M.traction_gain(1.0, 1.0, law, rule, 1.0, self.KAPPA), 1.0)
                self.assertEqual(M.traction_gain(0.0, 0.0, law, rule, 1.0, self.KAPPA), 0.0)
                # bare target: no grip whatever the host
                self.assertEqual(M.traction_gain(1.0, 0.0, law, rule, 1.0, self.KAPPA), 0.0)

    def test_rules_pick_f_eff(self):
        fh, ft = 0.3, 0.8
        g_t = M.traction_gain(fh, ft, M.LAW_POWER, M.RULE_TARGET, 1.0, 0.0)
        g_m = M.traction_gain(fh, ft, M.LAW_POWER, M.RULE_MIN, 1.0, 0.0)
        g_p = M.traction_gain(fh, ft, M.LAW_POWER, M.RULE_PRODUCT, 1.0, 0.0)
        self.assertAlmostEqual(g_t, ft)
        self.assertAlmostEqual(g_m, fh)
        self.assertAlmostEqual(g_p, fh * ft)

    def test_power_law_exponent(self):
        self.assertAlmostEqual(M.traction_gain(1.0, 0.25, M.LAW_POWER, M.RULE_TARGET, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(M.traction_gain(1.0, 0.25, M.LAW_POWER, M.RULE_TARGET, 2.0, 0.0), 0.0625)

    def test_law_and_rule_codes(self):
        self.assertEqual(M.law_code('Langmuir'), M.LAW_LANGMUIR)
        self.assertEqual(M.rule_code('product'), M.RULE_PRODUCT)
        with self.assertRaises(ValueError):
            M.law_code('linear')


class TestBlockerFactor(unittest.TestCase):

    def test_endpoints(self):
        self.assertEqual(M.blocker_factor(0.0, 0.1), 0.1)   # bare blocker: V2.7 inert branch
        self.assertEqual(M.blocker_factor(1.0, 0.1), 1.0)   # coated blocker: V2.7 void branch
        self.assertAlmostEqual(M.blocker_factor(0.5, 0.1), 0.55)


class TestPairTables(unittest.TestCase):

    def test_legacy_two_species_reproduce_lut(self):
        t = M.build_pair_tables([1.0, 0.0], [10.0, 10.0], [0.45, 0.45],
                                W_CC, W_CB, W_BB, T_CC, T_CB, T_BB)
        # index 0 = collagen (old gtype 0), 1 = bare (old gtype 1)
        self.assertEqual(t['pair_W'][0, 0], W_CC)
        self.assertEqual(t['pair_W'][0, 1], W_CB)
        self.assertEqual(t['pair_W'][1, 0], W_CB)
        self.assertEqual(t['pair_W'][1, 1], W_BB)
        self.assertEqual(t['pair_tau'][0, 0], T_CC)
        self.assertEqual(t['pair_tau'][1, 1], T_BB)
        # V2.7 wall LUT: {0: W_adh_if, 1: W_adh_ii}
        self.assertEqual(t['wall_W'][0], W_CB)
        self.assertEqual(t['wall_W'][1], W_BB)
        E_star_gg = (10.0 * 1e3) / (2.0 * (1.0 - 0.45 ** 2))
        self.assertAlmostEqual(t['pair_Estar'][0, 1], E_star_gg, places=9)
        self.assertAlmostEqual(t['wall_Estar'][0], (10.0 * 1e3) / (1.0 - 0.45 ** 2), places=9)

    def test_tables_are_symmetric_and_contiguous(self):
        t = M.build_pair_tables([1.0, 0.5, 0.0], [10.0, 5.0, 20.0], [0.45, 0.4, 0.49],
                                W_CC, W_CB, W_BB, T_CC, T_CB, T_BB)
        for key in ('pair_W', 'pair_tau', 'pair_Estar'):
            self.assertTrue(np.allclose(t[key], t[key].T))
            self.assertTrue(t[key].flags['C_CONTIGUOUS'])
        # the leaf function and the table agree
        self.assertAlmostEqual(t['pair_W'][0, 1], M.mix_pair(1.0, 0.5, W_CC, W_CB, W_BB), places=15)
        self.assertAlmostEqual(t['pair_Estar'][1, 2], M.pair_E_star_Pa(5.0, 0.4, 20.0, 0.49), places=6)


if __name__ == '__main__':
    unittest.main()
