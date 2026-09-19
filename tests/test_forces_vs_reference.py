"""
Compiled contact kernels vs the reference Python loops (V3.0 Phase 6a).

Same GranuleSystem (deep copy), same Params, same RNG seed. The kernel path
(gels.kernels.contact2d / contact3d + the Python bridging pass) must give

  * forces and torques equal to rounding (only the summation order differs),
  * the same contact set with the same geometry, forces and material data,
  * the same clip planes, in the same order,
  * identical cell state, bridge targets, lock flags and per-cell forces,
  * an identical random-generator state afterwards (same number of draws),

for 2D circles (walls, periodic), 2D superellipses (walls, periodic), 3D spheres
(walls, periodic) and 3D superellipsoids (walls). Also: 1 vs many threads are
bit-identical, a short trajectory through step() stays close, and the
dispatch flags route as documented.

Run:  python -m unittest tests.test_forces_vs_reference -v
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
    CellState, Params, generate_packing, generate_packing_3d, kernels_enabled, step,
    update_cell_state,
)
from gels.kernels import HAS_NUMBA  # noqa: E402
from gels.kernels import reference as ref  # noqa: E402

if HAS_NUMBA:
    from gels.kernels import contact2d, contact3d  # noqa: E402
    from gels.kernels.contacts import clips_to_lists  # noqa: E402


def _params(mode='2D', shape=False, periodic=False, **kw):
    base = dict(mode=mode, Lx=600.0, Ly=600.0, Lz=800.0,
                phi_solid_target=0.6, func_ratio=0.6, cell_surface_coverage=1.0,
                packing_settle_steps=30, save_data=False, compress_archive=False,
                bridge_attempt_rate=1e3, min_fa_for_bridge=0.0, fa_maturation_rate=100.0,
                t_spread_duration=0.1, T_active=5.0,
                boundary_mode='periodic' if periodic else 'walls',
                perf_cells_backend='python')     # exact reference cell machinery on top of the kernels
    if mode == '3D':
        base.update(Lx=400.0, Ly=400.0, Lz=400.0, phi_solid_target=0.5)
    if shape:
        base.update(shape_enabled=True, aspect_ratio_func_mean=1.3, blockiness_func_mean=2.6,
                    aspect_ratio_inert_mean=1.2, blockiness_inert_mean=2.2)
        if mode == '3D':
            base.update(Lx=250.0, Ly=250.0, Lz=250.0, aspect_ratio_c_func_mean=0.9,
                        blockiness_n2_func_mean=2.2)
    base.update(kw)
    return Params(**base)


def _prepare(p, seed):
    gen = generate_packing_3d if p.mode == '3D' else generate_packing
    gs = gen(p, seed=seed)
    rng = np.random.default_rng(seed)
    update_cell_state(gs, p, 5.0, rng)            # spread cells, mature focal adhesions
    dim = 3 if p.mode == '3D' else 2
    vrng = np.random.default_rng(seed + 100)
    gs.vel[:, :dim] = vrng.normal(scale=3.0, size=(gs.N, dim))   # exercise friction
    if p.shape_enabled:
        # the settle separates BOUNDING spheres, so the bodies inside are often not
        # touching; inflate so the shape solvers see real overlaps (both paths get
        # the same gs). Superellipsoids sit deeper inside their bounding spheres.
        factor = 1.15 if dim == 3 else 1.06
        for name in ('a', 'b', 'c', 'r', 'r_bound'):
            getattr(gs, name)[:] *= factor
    return gs


def _evaluate(gs, p, seed, kernel):
    rng = np.random.default_rng(seed + 7)
    for k in range(gs.N):
        gs.contact_clips[k] = []
    if p.mode == '3D':
        fn = contact3d.compute_forces_3d if kernel else ref.compute_forces_3d
    else:
        fn = contact2d.compute_forces_2d if kernel else ref.compute_forces
    F, T, C = fn(gs, p, rng)
    return F, T, C, rng


def _contacts_table(C):
    rows = [(int(c['i']), int(c['j']), c['overlap'], c['F_normal'], c['A_contact'], c['R_eff'],
             c['cx'], c['cy'], c.get('cz', 0.0), c['nx'], c['ny'], c.get('nz', 0.0),
             c['E_star'], c['W'], c['tau_0'], c['species_i'], c['species_j'])
            for c in C]
    rows.sort(key=lambda r: (r[0], r[1]))
    return np.array(rows, dtype=float) if rows else np.zeros((0, 17))


def _clips_array(clips):
    return np.array(clips, dtype=float).reshape(len(clips), -1) if clips else np.zeros((0, 0))


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestKernelsMatchReference(unittest.TestCase):

    def _check(self, p, seed=1):
        gs_r = _prepare(p, seed)
        gs_k = copy.deepcopy(gs_r)
        Fr, Tr, Cr, rr = _evaluate(gs_r, p, seed, kernel=False)
        Fk, Tk, Ck, rk = _evaluate(gs_k, p, seed, kernel=True)

        self.assertGreater(len(Cr), 0, "test system has no contacts")
        np.testing.assert_allclose(Fk, Fr, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(Tk, Tr, rtol=1e-9, atol=1e-9)

        self.assertEqual(len(Ck), len(Cr))
        np.testing.assert_allclose(_contacts_table(Ck), _contacts_table(Cr), rtol=1e-9, atol=1e-12)

        clips_to_lists(gs_k, *gs_k.clip_arrays)        # kernel path keeps clips as arrays
        for i in range(gs_r.N):
            cr = _clips_array(gs_r.contact_clips[i])
            ck = _clips_array(gs_k.contact_clips[i])
            self.assertEqual(cr.shape, ck.shape, f"clip count differs on granule {i}")
            if cr.size:
                np.testing.assert_allclose(ck, cr, rtol=1e-9, atol=1e-12)

        np.testing.assert_array_equal(gs_k.cell_state, gs_r.cell_state)
        np.testing.assert_array_equal(gs_k.cell_bridge_target, gs_r.cell_bridge_target)
        np.testing.assert_array_equal(gs_k.cell_bridge_locked, gs_r.cell_bridge_locked)
        np.testing.assert_array_equal(gs_k.cell_bridge_age, gs_r.cell_bridge_age)
        for name in ('cell_fx', 'cell_fy', 'cell_fz', 'cell_alignment', 'cell_theta_local'):
            np.testing.assert_allclose(getattr(gs_k, name), getattr(gs_r, name), rtol=1e-9, atol=1e-9,
                                       err_msg=name)
        self.assertEqual(rk.bit_generator.state, rr.bit_generator.state, "random draws differ")
        self.assertGreater(int(np.sum(gs_r.cell_state == int(CellState.BRIDGING))), 0,
                           "test system formed no bridges")
        return gs_r, gs_k, Fr, Fk, Cr, Ck

    def test_2d_circles_walls(self):
        self._check(_params('2D'))

    def test_2d_circles_periodic(self):
        self._check(_params('2D', periodic=True))

    def test_2d_shapes_walls(self):
        gs_r, gs_k, Fr, Fk, Cr, Ck = self._check(_params('2D', shape=True))
        self.assertFalse(gs_r.is_circle)

    def test_2d_shapes_periodic(self):
        self._check(_params('2D', shape=True, periodic=True))

    def test_3d_spheres_walls(self):
        self._check(_params('3D'))

    def test_3d_spheres_periodic(self):
        self._check(_params('3D', periodic=True))

    def test_3d_shapes_walls(self):
        gs_r, gs_k, Fr, Fk, Cr, Ck = self._check(_params('3D', shape=True), seed=2)
        self.assertFalse(gs_r.is_circle)

    # ── V3.1 container geometry and gravity ──
    def test_3d_spheres_cylinder_free_top(self):
        self._check(_params('3D', boundary_shape='cylinder', boundary_top='free',
                            gravity_enabled=True, granule_density=1180.0,
                            packing_consolidation='gravity'), seed=4)

    def test_3d_shapes_cylinder_free_top(self):
        self._check(_params('3D', shape=True, boundary_shape='cylinder', boundary_top='free',
                            gravity_enabled=True, granule_density=1180.0,
                            packing_consolidation='gravity'), seed=5)

    def test_2d_dish_free_top(self):
        self._check(_params('2D', boundary_top='free', gravity_enabled=True,
                            granule_density=1180.0, packing_consolidation='gravity'), seed=6)

    def test_2d_shapes_dish_free_top(self):
        self._check(_params('2D', shape=True, boundary_top='free', gravity_enabled=True,
                            granule_density=1180.0, packing_consolidation='gravity'), seed=7)

    # ── V3.2: cells bridging across a contact resist shear at it ──
    def test_2d_cell_contact_adhesion(self):
        self._check(_params('2D', friction_mu=0.3, cell_contact_adhesion=20.0))

    def test_3d_cell_contact_adhesion(self):
        self._check(_params('3D', friction_mu=0.3, cell_contact_adhesion=20.0))

    def test_mixed_stiffness_species_use_pair_E_star(self):
        species = [
            dict(name='soft', f=1.0, volume_fraction=0.5, color='#CC2222', radius_mean=40.0,
                 radius_std=4.0, radius_min=15.0, E_kPa=5.0),
            dict(name='stiff', f=0.3, volume_fraction=0.5, color='#2255CC', radius_mean=40.0,
                 radius_std=4.0, radius_min=15.0, E_kPa=20.0),
        ]
        p = _params('2D', species=species)
        gs_r, gs_k, Fr, Fk, Cr, Ck = self._check(p, seed=3)
        from gels.materials import pair_E_star_Pa
        for c in Ck:
            si, sj = int(c['species_i']), int(c['species_j'])
            want = pair_E_star_Pa(gs_k.species_E[si], gs_k.species_nu[si], gs_k.species_E[sj], gs_k.species_nu[sj])
            self.assertAlmostEqual(c['E_star'], want, places=6)

    def test_threads_bit_identical(self):
        import numba
        p = _params('2D', shape=True)
        gs = _prepare(p, 4)
        n_max = numba.config.NUMBA_NUM_THREADS
        old = numba.get_num_threads()
        try:
            numba.set_num_threads(1)
            F1, T1, C1, _ = _evaluate(copy.deepcopy(gs), p, 4, kernel=True)
            numba.set_num_threads(min(8, n_max))
            F8, T8, C8, _ = _evaluate(copy.deepcopy(gs), p, 4, kernel=True)
        finally:
            numba.set_num_threads(old)
        np.testing.assert_array_equal(F1, F8)
        np.testing.assert_array_equal(T1, T8)
        np.testing.assert_array_equal(_contacts_table(C1), _contacts_table(C8))

    def test_step_trajectory_close(self):
        p_k = _params('2D')
        p_r = copy.deepcopy(p_k)
        p_r.use_numba = False
        self.assertTrue(kernels_enabled(p_k))
        self.assertFalse(kernels_enabled(p_r))
        gs_k = generate_packing(p_k, seed=5)
        gs_r = copy.deepcopy(gs_k)
        rng_k = np.random.default_rng(11)
        rng_r = np.random.default_rng(11)
        t = 0.0
        for _ in range(3):
            t += p_k.dt
            Fk, Ck = step(gs_k, p_k, rng_k, t)
            Fr, Cr = step(gs_r, p_r, rng_r, t)
        np.testing.assert_allclose(gs_k.pos, gs_r.pos, rtol=1e-7, atol=1e-7)
        np.testing.assert_array_equal(gs_k.cell_state, gs_r.cell_state)
        self.assertEqual(rng_k.bit_generator.state, rng_r.bit_generator.state)
        self.assertTrue(hasattr(Ck, 'column'))          # ContactSoA on the kernel path
        self.assertIsInstance(Cr, list)                 # dict list on the reference path

    def test_contact_soa_behaves_like_dict_list(self):
        p = _params('2D')
        gs = _prepare(p, 1)
        F, T, C, _ = _evaluate(gs, p, 1, kernel=True)
        self.assertTrue(bool(C))
        self.assertEqual(len(list(C)), len(C))
        first = C[0]
        self.assertIsInstance(first['i'], int)
        self.assertIsInstance(first['overlap'], float)
        self.assertEqual(set(first), set(C.FIELDS))
        self.assertEqual(len(C[-1:]), 1)
        self.assertEqual(C.column('overlap').shape, (len(C),))
        self.assertTrue(np.all(C.column('overlap') > 0))


class TestDispatchFlags(unittest.TestCase):

    def test_flags(self):
        self.assertEqual(kernels_enabled(Params()), HAS_NUMBA)
        self.assertFalse(kernels_enabled(Params(use_numba=False)))
        self.assertFalse(kernels_enabled(Params(deformable_enabled=True)))
        self.assertFalse(kernels_enabled(Params(perf_neighbor_backend='reference')))


if __name__ == '__main__':
    unittest.main()
