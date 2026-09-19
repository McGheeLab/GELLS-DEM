"""
viz2 after the V3.0 split: species colours, collection-based cells, palette/snapshot_ops.

  * viz2.common re-exports everything the modules import,
  * granules are coloured by species (three species → three colours; legacy → red/green),
  * draw_cells_on_ax draws one shape per cell (per periodic image) with a handful
    of collections and renders in well under a second for hundreds of cells,
  * the legend lists every species,
  * ensure_phase_fields rebuilds fields via GranuleSystem.from_snapshot_dict.

Run:  python -m unittest tests.test_viz2_species -v
"""

import os
import sys
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('MPLBACKEND', 'Agg')

from gels.engine import CellState, Params, compute_forces, generate_packing, update_cell_state  # noqa: E402

THREE = [
    dict(name='full', f=1.0, volume_fraction=0.3, color='#CC2222', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
    dict(name='half', f=0.5, volume_fraction=0.4, color='#2255CC', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
    dict(name='bare', f=0.0, volume_fraction=0.3, color='#22AA22', radius_mean=55.0, radius_std=5.0, radius_min=20.0),
]


def _snap_from(gs):
    keys = ['x', 'y', 'r', 'gtype', 'species_id', 'f', 'a', 'b', 'n1', 'theta', 'cell_granule_id',
            'cell_state', 'cell_theta_local', 'cell_bridge_target', 'cell_bridge_locked']
    return {k: np.asarray(getattr(gs, k)).copy() for k in keys}


class TestViz2Species(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.p = Params(Lx=600.0, Ly=600.0, phi_solid_target=0.6, species=[dict(s) for s in THREE],
                       cell_surface_coverage=1.0, packing_settle_steps=40, save_data=False,
                       bridge_attempt_rate=1e3, min_fa_for_bridge=0.0, fa_maturation_rate=100.0,
                       t_spread_duration=0.1, Ngrid=48)
        cls.gs = generate_packing(cls.p, seed=7)
        rng = np.random.default_rng(7)
        update_cell_state(cls.gs, cls.p, 5.0, rng)
        compute_forces(cls.gs, cls.p, rng)
        cls.snap = _snap_from(cls.gs)

    def test_reexports(self):
        from viz2 import common
        for name in ('FUNC_COLOR', 'INERT_COLOR', 'VOID_COLOR', 'CELL_COLORS', 'CELL_LABELS', 'ENERGY_COLORS',
                     'is_3d', 'slice_snap_z_midplane', 'cell_world_positions', 'cell_world_positions_3d',
                     'ensure_phase_fields', '_periodic_offsets', '_min_image', 'draw_granule_patches',
                     'draw_cells_on_ax', 'make_scaffold_legend', 'species_table', 'render_frame_to_array'):
            self.assertTrue(hasattr(common, name), name)

    def test_granules_coloured_by_species(self):
        import matplotlib.pyplot as plt
        from viz2 import common, palette
        fig, ax = plt.subplots()
        common.setup_scaffold_axes(ax, self.p)
        pc = common.draw_granule_patches(ax, self.snap, self.p)
        fc = pc.get_facecolors()
        distinct = {tuple(np.round(c[:3], 3)) for c in fc}
        self.assertEqual(len(distinct), 3)
        expected = {tuple(np.round(palette.hex_to_rgba(s['color'])[:3], 3)) for s in THREE}
        self.assertEqual(distinct, expected)
        plt.close(fig)
        # legacy snapshot (no species_id): red / green by gtype
        legacy = dict(self.snap)
        del legacy['species_id']
        fig, ax = plt.subplots()
        pc = common.draw_granule_patches(ax, legacy, Params())
        distinct = {tuple(np.round(c[:3], 3)) for c in pc.get_facecolors()}
        self.assertEqual(distinct, {tuple(np.round(palette.hex_to_rgba(palette.FUNC_COLOR)[:3], 3)),
                                    tuple(np.round(palette.hex_to_rgba(palette.INERT_COLOR)[:3], 3))})
        plt.close(fig)

    def test_cells_drawn_with_collections(self):
        import matplotlib.pyplot as plt
        from viz2 import common
        fig, ax = plt.subplots(figsize=(6, 6))
        common.setup_scaffold_axes(ax, self.p)
        t0 = time.perf_counter()
        cols = common.draw_cells_on_ax(ax, self.snap, self.p)
        fig.canvas.draw()
        dt = time.perf_counter() - t0
        n_drawn = sum(len(c.get_offsets()) for c in cols)
        self.assertEqual(n_drawn, self.gs.total_cells)
        self.assertLessEqual(len(cols), 3)             # circles / migrating / bridges, one image each (walls)
        self.assertLess(dt, 2.0)
        self.assertGreater(int(np.sum(self.snap['cell_state'] == int(CellState.BRIDGING))), 0)
        plt.close(fig)

    def test_periodic_ghost_cells(self):
        import matplotlib.pyplot as plt
        from viz2 import common
        p = Params(Lx=600.0, Ly=600.0, boundary_mode='periodic')
        snap = dict(self.snap)
        snap['x'] = snap['x'] % p.Lx
        fig, ax = plt.subplots()
        cols = common.draw_cells_on_ax(ax, snap, p)
        n_drawn = sum(len(c.get_offsets()) for c in cols)
        self.assertGreaterEqual(n_drawn, self.gs.total_cells)   # ghosts add images, never remove cells
        plt.close(fig)

    def test_legend_lists_species(self):
        from viz2 import common
        labels = [h.get_label() for h in common.make_scaffold_legend(seen_states={0, 3}, p=self.p)]
        self.assertTrue(any(lab.startswith('full') for lab in labels))
        self.assertTrue(any(lab.startswith('bare') for lab in labels))
        self.assertIn('Void', labels)
        self.assertIn('Bridging', labels)
        # legacy runs: one entry per legacy species (collagen f=1 / bare f=0), in table order
        from viz2 import palette
        legacy = [h.get_label() for h in common.make_scaffold_legend(p=Params())]
        for s in palette.species_table(Params()):
            self.assertTrue(any(lab.startswith(s['name']) for lab in legacy), (s['name'], legacy))

    def test_ensure_phase_fields_via_from_snapshot_dict(self):
        from viz2.snapshot_ops import ensure_phase_fields
        snap = dict(self.snap)
        pf, pi, pv = ensure_phase_fields(snap, self.p)
        self.assertEqual(pf.shape, pi.shape)
        self.assertIn('phi_f', snap)
        self.assertLessEqual(float((pf + pi).max()), 1.0 + 1e-9)
        self.assertGreater(float(pf.mean()), 0.0)


class TestViz2SpeciesHelpers(unittest.TestCase):

    def test_contact_friction_reduces_to_legacy_lookup(self):
        from viz2.snapshot_ops import contact_friction
        p = Params()
        snap = {'gtype': np.array([0, 0, 1, 1]), 'x': np.zeros(4)}
        ci = np.array([0, 0, 2])
        cj = np.array([1, 2, 3])
        tau = contact_friction(snap, p, ci, cj)
        np.testing.assert_allclose(tau, [p.tau_0_cc, p.tau_0_cb, p.tau_0_bb])
        snap_f = dict(snap, f=np.array([1.0, 0.5, 0.0, 0.0]))
        tau_f = contact_friction(snap_f, p, np.array([0]), np.array([1]))
        self.assertLess(tau_f[0], p.tau_0_cc)
        self.assertGreater(tau_f[0], p.tau_0_cb)

    def test_species_fraction_panel(self):
        import matplotlib.pyplot as plt
        from viz2.phase_fractions import plot_species_fractions
        p = Params(species=[dict(s) for s in THREE])
        hist = [dict(time=float(t), phi_v_mean=0.4, phi_c_mean=0.3,
                     phi_sp_0_mean=0.2, phi_sp_1_mean=0.25, phi_sp_2_mean=0.15) for t in range(4)]
        fig = plot_species_fractions(hist, p)
        self.assertIsNotNone(fig)
        labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
        self.assertTrue(any(lab.startswith('full') for lab in labels))
        plt.close(fig)
        self.assertIsNone(plot_species_fractions([dict(time=0.0, phi_f_mean=0.3)], p))


if __name__ == '__main__':
    unittest.main()
