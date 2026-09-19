"""
gels/live/frames.py — vectorised geometry and message builders (V3.0 Phase 5).

  * cell_world_xy / cell_world_xyz agree with the engine's scalar helpers,
  * bridge segments end on the target surface (minimum image when periodic),
  * a frame built from a GranuleSystem equals one built from its snapshot dict,
  * the start message carries the species table and static arrays.

Run:  python -m unittest tests.test_frames -v
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
    CellState, GranuleSystem, Params, _cell_world_pos_2d, _cell_world_pos_3d, generate_packing,
    generate_packing_3d, update_cell_state, compute_forces, compute_forces_3d,
)
from gels.live import frames  # noqa: E402


def _system_2d(shape=False, periodic=False, seed=1):
    p = Params(Lx=500.0, Ly=500.0, phi_solid_target=0.55, func_ratio=0.6, cell_surface_coverage=1.0,
               packing_settle_steps=30, save_data=False, shape_enabled=shape,
               aspect_ratio_func_mean=1.3, blockiness_func_mean=2.6, aspect_ratio_inert_mean=1.2,
               boundary_mode='periodic' if periodic else 'walls', bridge_attempt_rate=1e3,
               min_fa_for_bridge=0.0, fa_maturation_rate=100.0, t_spread_duration=0.1)
    gs = generate_packing(p, seed=seed)
    rng = np.random.default_rng(seed)
    update_cell_state(gs, p, 5.0, rng)
    compute_forces(gs, p, rng)
    return gs, p


class TestVectorisedGeometry(unittest.TestCase):

    def test_cell_world_xy_matches_engine(self):
        for shape in (False, True):
            gs, p = _system_2d(shape=shape)
            wx, wy = frames.cell_world_xy(gs.x, gs.y, gs.theta, gs.a, gs.b, gs.n_shape,
                                          gs.cell_granule_id, gs.cell_theta_local)
            ref = np.array([_cell_world_pos_2d(gs, ci) for ci in range(gs.total_cells)])
            np.testing.assert_allclose(wx, ref[:, 0], atol=1e-9)
            np.testing.assert_allclose(wy, ref[:, 1], atol=1e-9)

    def test_cell_world_xyz_matches_engine(self):
        p = Params(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.5, func_ratio=0.6,
                   cell_surface_coverage=1.0, packing_settle_steps=30, save_data=False,
                   shape_enabled=True, aspect_ratio_func_mean=1.2, blockiness_func_mean=2.5,
                   aspect_ratio_c_func_mean=0.9, blockiness_n2_func_mean=2.2)
        gs = generate_packing_3d(p, seed=2)
        rng = np.random.default_rng(2)
        update_cell_state(gs, p, 1.0, rng)
        wx, wy, wz = frames.cell_world_xyz(gs.x, gs.y, gs.z, gs.quat, gs.a, gs.b, gs.c, gs.n1, gs.n2,
                                           gs.cell_granule_id, gs.cell_eta_local, gs.cell_omega_local)
        ref = np.array([_cell_world_pos_3d(gs, ci) for ci in range(gs.total_cells)])
        np.testing.assert_allclose(np.column_stack([wx, wy, wz]), ref, atol=1e-9)

    def test_quat_rotate_many_matches_scalar(self):
        from gels.engine import quat_rotate
        rng = np.random.default_rng(0)
        q = rng.normal(size=(50, 4))
        q /= np.linalg.norm(q, axis=1)[:, None]
        v = rng.normal(size=(50, 3))
        got = frames.quat_rotate_many(q, v)
        ref = np.array([quat_rotate(q[i], v[i]) for i in range(50)])
        np.testing.assert_allclose(got, ref, atol=1e-12)

    def test_invalid_host_index_gives_origin(self):
        wx, wy = frames.cell_world_xy(np.array([10.0]), np.array([20.0]), np.zeros(1), np.array([5.0]),
                                      np.array([5.0]), np.array([2.0]), np.array([0, 7, -1]), np.zeros(3))
        self.assertEqual(wx[0], 15.0)
        self.assertEqual((wx[1], wy[1], wx[2], wy[2]), (0.0, 0.0, 0.0, 0.0))


class TestBridgeSegments(unittest.TestCase):

    def test_segment_ends_on_target_surface(self):
        pos = np.array([[0.0, 0.0], [100.0, 0.0]])
        r = np.array([40.0, 40.0])
        cw = np.array([[40.0, 0.0]])                 # a cell on granule 0 facing granule 1
        state = np.array([int(CellState.BRIDGING)])
        bt = np.array([1])
        seg, sel = frames.bridge_segments(cw, state, bt, pos, r, np.array([1000.0, 1000.0]), False)
        self.assertEqual(sel.tolist(), [0])
        np.testing.assert_allclose(seg[0, 0], [40.0, 0.0])
        np.testing.assert_allclose(seg[0, 1], [60.0, 0.0])   # target surface: 100 - 40

    def test_periodic_minimum_image(self):
        pos = np.array([[10.0, 50.0], [190.0, 50.0]])
        r = np.array([5.0, 5.0])
        cw = np.array([[5.0, 50.0]])
        seg, _ = frames.bridge_segments(cw, np.array([int(CellState.BRIDGING)]), np.array([1]),
                                        pos, r, np.array([200.0, 100.0]), True)
        # the target is 15 µm away across the boundary, not 185 µm the long way
        self.assertLess(seg[0, 1, 0], 5.0)
        np.testing.assert_allclose(seg[0, 1], [-10.0 + 5.0, 50.0])

    def test_non_bridging_cells_ignored(self):
        pos = np.zeros((2, 2)); r = np.ones(2)
        seg, sel = frames.bridge_segments(np.zeros((3, 2)), np.array([0, 1, 4]), np.array([1, 1, 1]),
                                          pos, r, np.array([10.0, 10.0]), False)
        self.assertEqual(seg.shape, (0, 2, 2))
        self.assertEqual(sel.size, 0)


class TestMessages(unittest.TestCase):

    def test_frame_from_gs_equals_frame_from_snapshot(self):
        gs, p = _system_2d(periodic=True)
        msg, static = frames.start_message(gs, p, seed=1, n_steps=10, run_dir='x')
        self.assertEqual(msg['kind'], 'start')
        self.assertEqual(msg['N'], gs.N)
        self.assertEqual(len(msg['species']), gs.K)
        self.assertEqual(msg['species'][0]['color'], gs.species_colors[0])
        f_gs = frames.frame_message(gs, p, 3, 1.5, static, metrics={'n_bridges': 2.0})
        snap = {'x': gs.x.copy(), 'y': gs.y.copy(), 'theta': gs.theta.copy(), 'r': gs.r.copy(),
                'cell_state': gs.cell_state.copy(), 'cell_bridge_target': gs.cell_bridge_target.copy(),
                'cell_bridge_locked': gs.cell_bridge_locked.copy(), 'cell_theta_local': gs.cell_theta_local.copy(),
                'time': 1.5}
        f_snap = frames.frame_from_snap(snap, p, static, step=3)
        for key in ('x', 'y', 'theta', 'cell_x', 'cell_y', 'cell_state', 'bridge_seg', 'bridge_locked'):
            np.testing.assert_array_equal(f_gs[key], f_snap[key], err_msg=key)
        self.assertEqual(f_gs['metrics'], {'n_bridges': 2.0})
        self.assertEqual(f_gs['x'].dtype, np.float32)
        self.assertEqual(f_gs['cell_state'].dtype, np.int8)
        n_bridging = int(np.sum(gs.cell_state == int(CellState.BRIDGING)))
        self.assertEqual(f_gs['bridge_seg'].shape[0], n_bridging)
        # periodic: cell positions wrapped into the box
        self.assertTrue(np.all(f_gs['cell_x'] >= 0) and np.all(f_gs['cell_x'] < p.Lx))

    def test_speed_and_no_cells_options(self):
        gs, p = _system_2d()
        msg, static = frames.start_message(gs, p)
        f = frames.frame_message(gs, p, 1, 0.5, static, send_cells=False, send_speed=True)
        self.assertNotIn('cell_x', f)
        self.assertEqual(f['speed'].shape, (gs.N,))


if __name__ == '__main__':
    unittest.main()
