"""
Compiled rendering and metrics vs the reference (V3.0 Phase 6d).

  * per-species fields from gels.kernels.render agree with the reference
    renderer to 2e-5 (the 2D kernel truncates each tanh profile at 8 interface
    widths; 3D uses the reference's 3w bounding box) for circles / superellipses
    (walls, periodic) and spheres / superellipsoids,
  * effective radii are bit-identical,
  * gels.kernels.metrics.compute_metrics returns the same keys and values
    (ints exact, floats to 1e-9) as the reference on the same fields,
  * the engine dispatchers route to the kernels when enabled.

Run:  python -m unittest tests.test_render_metrics_vs_reference -v
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

from gels import engine  # noqa: E402
from gels.engine import (  # noqa: E402
    Params, compute_effective_radii, compute_effective_radii_3d, generate_packing,
    generate_packing_3d, group_species_fields, kernels_enabled, step, update_cell_state,
)
from gels.kernels import HAS_NUMBA  # noqa: E402
from gels.kernels import reference as ref  # noqa: E402

if HAS_NUMBA:
    from gels.kernels import metrics as kmetrics  # noqa: E402
    from gels.kernels import render as krender  # noqa: E402
    from gels.kernels.contacts import clips_to_lists  # noqa: E402


def _params(mode='2D', shape=False, periodic=False, **kw):
    base = dict(mode=mode, Lx=400.0, Ly=400.0, Lz=800.0, Ngrid=64, Ngrid_3d=24,
                phi_solid_target=0.6, func_ratio=0.6, cell_surface_coverage=1.0,
                packing_settle_steps=30, save_data=False, compress_archive=False,
                bridge_attempt_rate=1e3, min_fa_for_bridge=0.0, fa_maturation_rate=100.0,
                t_spread_duration=0.1, boundary_mode='periodic' if periodic else 'walls',
                perf_metrics_voronoi_stride=1)      # exact comparison with the full-grid reference
    if mode == '3D':
        base.update(Lx=250.0, Ly=250.0, Lz=250.0, phi_solid_target=0.5)
    if periodic:
        # V3.3: halve the granules under periodic BCs. The defaults (R 40/60,
        # inflated further by r_bound for superellipsoids) put max r_bound at
        # 82 um in a 400 um box -- about four granule diameters across, which
        # makes the interaction cutoff 2*max_r_bound + L_max exceed L/2. Past
        # that, cKDTree(boxsize=).query_pairs silently drops the second image
        # of a neighbour, so the configuration was measuring the wrong physics;
        # gels.kernels.neighbors.check_min_image now refuses it. Smaller
        # granules keep the granule COUNT (and the runtime) rather than growing
        # the box, which would be cubic in 3D.
        base.update(R_func_mean=20.0, R_func_std=2.5, R_inert_mean=30.0, R_inert_std=4.0)
        if mode == '3D':
            # Even halved, 3D lands at cutoff 125.015 against a 125.0 limit:
            # L_max (= max(cell_sense_distance, bridge_break_gap) = 60) is fixed
            # and dominates 2*r_bound here, so the box has to give. 300 leaves a
            # comfortable margin at ~1.7x the granule count.
            base.update(Lx=300.0, Ly=300.0, Lz=300.0)
    if shape:
        base.update(shape_enabled=True, aspect_ratio_func_mean=1.3, blockiness_func_mean=2.6,
                    aspect_ratio_inert_mean=1.2, blockiness_inert_mean=2.2)
        if mode == '3D':
            base.update(aspect_ratio_c_func_mean=0.9, blockiness_n2_func_mean=2.2)
    base.update(kw)
    return Params(**base)


def _system(p, seed=1):
    """Packing + spread cells + one dynamics step (so contacts, clips and forces exist)."""
    gen = generate_packing_3d if p.mode == '3D' else generate_packing
    gs = gen(p, seed=seed)
    if p.shape_enabled:
        factor = 1.15 if p.mode == '3D' else 1.06
        for name in ('a', 'b', 'c', 'r', 'r_bound'):
            getattr(gs, name)[:] *= factor
    rng = np.random.default_rng(seed)
    update_cell_state(gs, p, 5.0, rng)
    F, _ = step(gs, p, rng, p.dt)
    if getattr(gs, 'clip_arrays', None) is not None:
        clips_to_lists(gs, *gs.clip_arrays)          # the reference renderer reads the lists
    return gs, F


def _compare_metrics(tc, m_k, m_r):
    tc.assertEqual(set(m_k), set(m_r), "metric keys differ")
    for key in m_r:
        vr, vk = m_r[key], m_k[key]
        if isinstance(vr, (bool, np.bool_)) or isinstance(vr, (int, np.integer)):
            tc.assertEqual(vk, vr, key)
        else:
            np.testing.assert_allclose(vk, vr, rtol=1e-9, atol=1e-14, err_msg=key)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestRenderMatchesReference(unittest.TestCase):

    def _check_fields(self, p, seed=1, atol=2e-5):
        gs, F = _system(p, seed)
        phi_r = ref.render_fields_species(gs, p)
        phi_k = krender.render_fields_species(gs, p)
        self.assertEqual(phi_k.shape, phi_r.shape)
        self.assertGreater(float(phi_r.max()), 0.5)
        np.testing.assert_allclose(phi_k, phi_r, rtol=0.0, atol=atol)
        # some granules carry clip planes, so the flat-face path was exercised
        self.assertTrue(any(len(c) > 0 for c in gs.contact_clips))
        return gs, F, phi_r, phi_k

    def test_2d_circles_walls(self):
        self._check_fields(_params('2D'))

    def test_2d_circles_periodic(self):
        self._check_fields(_params('2D', periodic=True))

    def test_2d_shapes_walls(self):
        self._check_fields(_params('2D', shape=True))

    def test_2d_shapes_periodic(self):
        self._check_fields(_params('2D', shape=True, periodic=True))

    def test_3d_spheres_walls(self):
        self._check_fields(_params('3D'))

    def test_3d_spheres_periodic(self):
        self._check_fields(_params('3D', periodic=True))

    def test_3d_shapes_walls(self):
        self._check_fields(_params('3D', shape=True), seed=2)

    def test_effective_radii_bit_identical(self):
        p = _params('2D', periodic=True)
        gs, F = _system(p)
        r_ref, ov_ref = compute_effective_radii(gs, p)
        r_k, ov_k = krender.effective_radii_2d(gs, p)
        np.testing.assert_array_equal(r_k, r_ref)
        np.testing.assert_array_equal(ov_k, ov_ref)
        p3 = _params('3D')
        gs3, F3 = _system(p3)
        r_ref3, ov_ref3 = compute_effective_radii_3d(gs3, p3)
        r_k3, ov_k3 = krender.effective_radii_3d(gs3, p3)
        # the reference takes d from np.dot (BLAS) - agree to rounding, not bit for bit
        np.testing.assert_allclose(r_k3, r_ref3, rtol=1e-13, atol=0)
        np.testing.assert_allclose(ov_k3, ov_ref3, rtol=1e-12, atol=1e-9)

    def test_dispatch(self):
        p = _params('2D')
        gs, F = _system(p)
        self.assertTrue(kernels_enabled(p))
        phi_k = krender.render_fields_species(gs, p)
        phi_e = engine.render_fields_species(gs, p)
        np.testing.assert_array_equal(phi_e, phi_k)
        pf, pi, pv = engine.render_fields(gs, p)
        pf2, pi2, pv2 = group_species_fields(gs, phi_k)
        np.testing.assert_array_equal(pf, pf2)
        p_ref = copy.deepcopy(p)
        p_ref.use_numba = False
        phi_r = engine.render_fields_species(gs, p_ref)
        np.testing.assert_allclose(phi_e, phi_r, atol=2e-5)


@unittest.skipUnless(HAS_NUMBA, "numba not installed")
class TestMetricsMatchReference(unittest.TestCase):

    def _check(self, p, seed=1):
        gs, F = _system(p, seed)
        phi_s = ref.render_fields_species(gs, p)
        pf, pi, pv = group_species_fields(gs, phi_s)
        m_r = ref.compute_metrics(gs, p, pf, pi, pv, 0.5, F, phi_s=phi_s)
        m_k = kmetrics.compute_metrics(gs, p, pf, pi, pv, 0.5, F, phi_s=phi_s)
        _compare_metrics(self, m_k, m_r)
        self.assertGreater(m_r['n_contacts'], 0)
        return gs, m_r, m_k

    def test_2d_circles_walls(self):
        gs, m_r, m_k = self._check(_params('2D'))
        self.assertGreater(m_r['n_bridges'], 0)

    def test_2d_circles_periodic(self):
        self._check(_params('2D', periodic=True))

    def test_2d_shapes_walls(self):
        gs, m_r, m_k = self._check(_params('2D', shape=True))
        self.assertIn('shape_circularity_mean', m_k)

    def test_2d_shapes_periodic(self):
        # larger box: the 400 um periodic system happens to have no interior contacts
        self._check(_params('2D', shape=True, periodic=True, Lx=600.0, Ly=600.0))

    def test_3d_spheres_walls(self):
        self._check(_params('3D'))

    def test_3d_spheres_periodic(self):
        self._check(_params('3D', periodic=True))

    def test_3d_shapes_walls(self):
        self._check(_params('3D', shape=True), seed=2)

    def test_three_species(self):
        species = [
            dict(name='full', f=1.0, volume_fraction=0.3, color='#CC2222', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
            dict(name='half', f=0.5, volume_fraction=0.4, color='#2255CC', radius_mean=40.0, radius_std=4.0, radius_min=15.0),
            dict(name='bare', f=0.0, volume_fraction=0.3, color='#22AA22', radius_mean=55.0, radius_std=5.0, radius_min=20.0),
        ]
        gs, m_r, m_k = self._check(_params('2D', species=species), seed=3)
        self.assertIn('phi_sp_2_mean', m_k)
        self.assertIn('n_contacts_sp_0_1', m_k)

    def test_voronoi_stride_close_to_full_grid(self):
        p = _params('2D', Ngrid=128)
        gs, F = _system(p)
        phi_s = ref.render_fields_species(gs, p)
        pf, pi, pv = group_species_fields(gs, phi_s)
        m1 = kmetrics.compute_metrics(gs, p, pf, pi, pv, 0.5, F, phi_s=phi_s)
        p4 = copy.deepcopy(p)
        p4.perf_metrics_voronoi_stride = 4
        m4 = kmetrics.compute_metrics(gs, p4, pf, pi, pv, 0.5, F, phi_s=phi_s)
        for key in ('x_f', 'x_i', 'phi_v_in_func', 'phi_v_in_inert', 'phi_solid_func', 'phi_solid_inert'):
            self.assertLess(abs(m4[key] - m1[key]), 2e-2, key)
        for key in m1:
            if key not in ('x_f', 'x_i', 'phi_v_in_func', 'phi_v_in_inert', 'phi_solid_func',
                           'phi_solid_inert', 'phi_v_crosscheck'):
                np.testing.assert_allclose(m4[key], m1[key], rtol=1e-12, atol=1e-14, err_msg=key)

    def test_engine_dispatch_and_shape_cache(self):
        p = _params('2D', shape=True)
        gs, F = _system(p)
        phi_s = engine.render_fields_species(gs, p)
        pf, pi, pv = group_species_fields(gs, phi_s)
        m1 = engine.compute_metrics(gs, p, pf, pi, pv, 0.5, F, phi_s=phi_s)
        self.assertTrue(hasattr(gs, '_shape_cache'))
        m2 = engine.compute_metrics(gs, p, pf, pi, pv, 0.5, F, phi_s=phi_s)   # cached path
        _compare_metrics(self, m2, m1)


if __name__ == '__main__':
    unittest.main()
