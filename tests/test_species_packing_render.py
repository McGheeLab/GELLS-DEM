"""
Species-aware packing, per-species rendering and metrics (V3.0 Phase 3).

  * species_counts reproduces the V2.7 count formula for legacy species and
    honours volume fractions / mean-volume rule for new ones,
  * a three-species 2D packing carries every species, seeds cells only on
    adhesive granules, renders one grid per species whose grouping gives the
    legacy phi_f / phi_i, and emits per-species metrics,
  * the 2D-slice packer carries species_id through from the 3D packing,
  * a short run() produces the new history keys.

Run:  python -m unittest tests.test_species_packing_render -v
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

from gels.engine import (  # noqa: E402
    Params, collagen_field, compute_metrics, generate_packing, generate_packing_2d_slice,
    group_species_fields, legacy_species_from_params, render_fields, render_fields_species,
    resolve_species, run, species_counts,
)

THREE = [
    dict(name='Granule_1', f=1.0, volume_fraction=0.2, color='#CC2222',
         radius_mean=40.0, radius_std=5.0, radius_min=15.0),
    dict(name='Granule_2', f=0.5, volume_fraction=0.4, color='#2255CC',
         radius_mean=40.0, radius_std=5.0, radius_min=15.0),
    dict(name='Granule_3', f=0.0, volume_fraction=0.4, color='#22AA22',
         radius_mean=60.0, radius_std=8.0, radius_min=20.0),
]


class TestSpeciesCounts(unittest.TestCase):

    def test_legacy_counts_match_v27_formula(self):
        p = Params(Lx=400.0, Ly=400.0, phi_f_target=0.25, phi_i_target=0.20)
        sp = legacy_species_from_params(p)
        n = species_counts(sp, p, mode='2D')
        A = p.Lx * p.Ly
        self.assertEqual(n[0], int(round(p.phi_f_target * A / (np.pi * p.R_func_mean ** 2))))
        self.assertEqual(n[1], int(round(p.phi_i_target * A / (np.pi * p.R_inert_mean ** 2))))
        p3 = Params(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.6, func_ratio=0.7)
        sp3 = legacy_species_from_params(p3)
        n3 = species_counts(sp3, p3, mode='3D')
        V = p3.Lx * p3.Ly * p3.Lz
        self.assertEqual(n3[0], int(round(p3.phi_f_target * V / ((4.0 / 3.0) * np.pi * p3.R_func_mean ** 3))))
        self.assertEqual(n3[1], int(round(p3.phi_i_target * V / ((4.0 / 3.0) * np.pi * p3.R_inert_mean ** 3))))

    def test_volume_fraction_species(self):
        p = Params(Lx=1000.0, Ly=1000.0, phi_solid_target=0.6, func_ratio=0.5,
                   species=[dict(s) for s in THREE], packing_count_rule='mean_volume')
        n = species_counts(p.species, p, mode='2D')
        A = 1e6
        # mean-volume rule uses E[R^2] = mu^2 + sigma^2
        expect0 = int(round(0.6 * 0.2 * A / (np.pi * (40.0 ** 2 + 5.0 ** 2))))
        expect2 = int(round(0.6 * 0.4 * A / (np.pi * (60.0 ** 2 + 8.0 ** 2))))
        self.assertEqual(n[0], expect0)
        self.assertEqual(n[2], expect2)
        # mean-radius rule differs when sigma > 0
        p.packing_count_rule = 'mean_radius'
        n_r = species_counts(p.species, p, mode='2D')
        self.assertEqual(n_r[0], int(round(0.6 * 0.2 * A / (np.pi * 40.0 ** 2))))
        self.assertGreaterEqual(n_r[0], n[0])


class TestThreeSpeciesPacking(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = Params(mode='2D', Lx=600.0, Ly=600.0, phi_solid_target=0.55, func_ratio=0.5,
                       species=[dict(s) for s in THREE], cell_surface_coverage=1.0,
                       Ngrid=64, packing_settle_steps=60, save_data=False)
        cls.gs = generate_packing(cls.p, seed=5)

    def test_species_present_and_cells_only_on_adhesive(self):
        gs = self.gs
        self.assertEqual(gs.K, 3)
        self.assertEqual(set(np.unique(gs.species_id).tolist()), {0, 1, 2})
        np.testing.assert_array_equal(gs.gtype, np.where(gs.species_id == 2, 1, 0))
        bare = gs.species_id == 2
        self.assertTrue(np.all(gs.n_cells[bare] == 0))
        self.assertTrue(np.all(gs.n_cells[~bare] > 0))
        # every granule's cell count follows the seeding law for its own coverage
        from gels.engine import cells_from_surface_coverage, seeded_cells
        p = self.p
        for i in range(gs.N):
            f_i = float(gs.f[i])
            n_full_i = cells_from_surface_coverage(gs.r[i], p.cell_diameter, p.cell_height_spread,
                                                   p.cell_surface_coverage, p.mode)
            expected = seeded_cells(n_full_i, f_i, p) if f_i >= p.f_min_adhesion else 0
            # radii are re-inflated during settling; allow one cell of rounding slack
            self.assertLessEqual(abs(int(gs.n_cells[i]) - expected), 1, (i, f_i, gs.n_cells[i], expected))
        half = gs.species_id == 1
        n_full_half = sum(cells_from_surface_coverage(r, p.cell_diameter, p.cell_height_spread,
                                                      p.cell_surface_coverage, p.mode) for r in gs.r[half])
        self.assertLessEqual(gs.n_cells[half].sum(), n_full_half + np.sum(half))
        # radii follow their species distributions
        self.assertGreater(np.mean(gs.r[bare]), np.mean(gs.r[~bare]))

    def test_render_species_and_grouping(self):
        gs, p = self.gs, self.p
        phi_s = render_fields_species(gs, p)
        self.assertEqual(phi_s.shape[0], 3)
        total = phi_s.sum(axis=0)
        self.assertLessEqual(float(total.max()), 1.0 + 1e-12)
        pf, pi, pv = group_species_fields(gs, phi_s)
        np.testing.assert_allclose(pf, phi_s[0] + phi_s[1])
        np.testing.assert_allclose(pi, phi_s[2])
        np.testing.assert_allclose(pv, 1.0 - pf - pi)
        pf2, pi2, pv2 = render_fields(gs, p)
        np.testing.assert_array_equal(pf, pf2)
        phi_c = collagen_field(gs, phi_s)
        np.testing.assert_allclose(phi_c, phi_s[0] + 0.5 * phi_s[1])
        # rendered solid matches true area within the normalisation tolerance
        A_true = float(np.sum(np.pi * gs.r ** 2))
        A_pix = (p.Lx / phi_s.shape[1]) * (p.Ly / phi_s.shape[2])
        self.assertLess(abs(float(total.sum()) * A_pix - A_true) / A_true, 2e-3)

    def test_species_metrics_keys(self):
        gs, p = self.gs, self.p
        phi_s = render_fields_species(gs, p)
        pf, pi, pv = group_species_fields(gs, phi_s)
        F = np.zeros((gs.N, 2))
        m = compute_metrics(gs, p, pf, pi, pv, 0.0, F, phi_s=phi_s)
        for k in range(3):
            for key in (f'n_gran_sp_{k}', f'n_cells_sp_{k}', f'n_bridging_sp_{k}',
                        f'phi_sp_{k}_true', f'phi_sp_{k}_mean', f'Z_sp_{k}'):
                self.assertIn(key, m)
        self.assertEqual(sum(m[f'n_gran_sp_{k}'] for k in range(3)), gs.N)
        self.assertAlmostEqual(sum(m[f'phi_sp_{k}_true'] for k in range(3)), m['phi_solid_true'], places=12)
        self.assertAlmostEqual(m['phi_c_true'], m['phi_sp_0_true'] + 0.5 * m['phi_sp_1_true'], places=12)
        n_pairs = sum(m[f'n_contacts_sp_{a}_{b}'] for a in range(3) for b in range(a, 3))
        self.assertEqual(n_pairs, m['n_contacts'])
        self.assertTrue(0.0 <= m['contact_ff_weight'] <= 1.0)
        self.assertTrue(0.0 <= m['f_solid_mean'] <= 1.0)
        # legacy 2-class keys still consistent with the species view
        self.assertEqual(m['n_contacts_ff'] + m['n_contacts_if'] + m['n_contacts_ii'], m['n_contacts'])


class TestSliceAndRun(unittest.TestCase):

    def test_2d_slice_carries_species(self):
        p = Params(mode='2D-slice', Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.5,
                   func_ratio=0.5, species=[dict(s) for s in THREE], cell_surface_coverage=1.0,
                   packing_settle_steps=40, save_data=False)
        gs = generate_packing_2d_slice(p, seed=2)
        self.assertEqual(gs.K, 3)
        self.assertTrue(np.all(gs.n_cells[gs.species_id == 2] == 0))
        self.assertEqual(gs.mode, '2D')

    def test_run_emits_species_history_keys(self):
        p = Params(mode='2D', Lx=400.0, Ly=400.0, phi_solid_target=0.5, func_ratio=0.5,
                   species=[dict(s) for s in THREE], cell_surface_coverage=1.0,
                   t_total=1.0, dt=0.5, save_every_h=0.5, Ngrid=48,
                   packing_settle_steps=40, save_data=False, perf_keep_snaps_in_memory=False)
        hist, snaps, p2, gs = run(p, seed=4)
        self.assertEqual(snaps, [])
        last = hist[-1]
        for key in ('phi_sp_0_mean', 'phi_sp_2_true', 'n_contacts_sp_0_1', 'disp_sp_1',
                    'phi_c_mean', 'f_solid_mean', 'contact_ff_weight'):
            self.assertIn(key, last)
        self.assertEqual(len(resolve_species(p2)), 3)


if __name__ == '__main__':
    unittest.main()
