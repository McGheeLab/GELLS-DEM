"""
Well geometry (V3.1): cylindrical side wall, free top, shape-aware volumes.
==========================================================================

``boundary.shape = cylinder`` (3D: axis z, R = min(Lx,Ly)/2 about the centre)
and ``boundary.top = free`` (no lid; z is up in 3D, y is up in a 2D dish
slice) replace faces of the V2.7 box. Covered here:

  * the cylinder wall force is radial, inward and independent of azimuth,
    and the compiled kernel agrees with the reference;
  * a free top exerts no force and does not clip until the container height;
  * granule counts and metric volumes follow the container shape;
  * RSA places every granule inside the cylinder and below the bed height;
  * validation rejects the combinations that have no meaning.

Run:  python -m unittest tests.test_well_geometry -v
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

from gels import config as C  # noqa: E402
from gels.engine import (  # noqa: E402
    GranuleSystem, Params, apply_position_bounds, boundary_geometry, compute_forces_3d,
    domain_base_area, domain_volume, find_contact_wall_3d, find_contact_wall_plane_3d,
    generate_packing_3d, species_counts, resolve_species,
)

R_GRAN = 20.0


def _well(**kw):
    base = dict(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, T_active=0.0, save_data=False,
                boundary_shape='cylinder', boundary_top='free')
    base.update(kw)
    return Params(**base)


def _single(p, x, y, z, r=R_GRAN, shape=False):
    """One granule at (x, y, z), no cells."""
    species = [dict(name='g', f=1.0, volume_fraction=1.0, color='#CC2222',
                    radius_mean=r, radius_std=0.0, radius_min=r)]
    kw = {}
    if shape:
        kw = dict(a=[r * 1.3], b=[r / 1.3], c=[r], n1=[2.4], n2=[2.2],
                  quat=[np.array([1.0, 0.0, 0.0, 0.0])])
    return GranuleSystem([x], [y], [r], [0], [0], z=[z], mode='3D',
                         species_id=[0], species=species, p=p, **kw)


class TestCylinderWall(unittest.TestCase):

    def _force(self, p, gs, use_numba):
        p.use_numba = use_numba
        for k in range(gs.N):
            gs.contact_clips[k] = []
        F, _t, _c = compute_forces_3d(gs, p, np.random.default_rng(1))
        return F[0].copy()

    def test_radial_inward_and_azimuth_independent(self):
        p = _well()
        geom = boundary_geometry(p)
        pen = 0.5                                  # penetrate the wall by half a micron
        rho = geom.R_cyl - R_GRAN + pen
        mags_ref, mags_ker = [], []
        for k in range(8):
            ang = 2.0 * np.pi * k / 8.0
            x = geom.cx + rho * np.cos(ang)
            y = geom.cy + rho * np.sin(ang)
            u = np.array([np.cos(ang), np.sin(ang), 0.0])
            for use_numba, mags in ((False, mags_ref), (True, mags_ker)):
                gs = _single(p, x, y, 200.0)
                F = self._force(p, gs, use_numba)
                self.assertLess(float(F[2]), 1e-12)        # no z component
                self.assertLess(float(np.dot(F, u)), 0.0, f"not inward at {ang:.2f} rad")
                # purely radial: no tangential component
                t = np.array([-np.sin(ang), np.cos(ang), 0.0])
                self.assertAlmostEqual(float(np.dot(F, t)), 0.0, places=9)
                mags.append(float(np.linalg.norm(F)))
        self.assertAlmostEqual(max(mags_ref), min(mags_ref), places=9)   # azimuth-independent
        np.testing.assert_allclose(mags_ker, mags_ref, rtol=1e-9, atol=1e-9)
        self.assertGreater(mags_ref[0], 0.0)

    def test_no_force_away_from_the_wall(self):
        p = _well()
        geom = boundary_geometry(p)
        for use_numba in (False, True):
            gs = _single(p, geom.cx, geom.cy, 200.0)
            F = self._force(p, gs, use_numba)
            np.testing.assert_allclose(F, np.zeros(3), atol=1e-12)

    def test_superellipsoid_uses_the_tangent_plane(self):
        p = _well(shape_enabled=True)
        geom = boundary_geometry(p)
        rho = geom.R_cyl - R_GRAN * 1.3 + 0.5
        for use_numba in (False, True):
            gs = _single(p, geom.cx + rho, geom.cy, 200.0, shape=True)
            F = self._force(p, gs, use_numba)
            self.assertLess(float(F[0]), 0.0)
            self.assertAlmostEqual(float(F[1]), 0.0, places=9)

    def test_plane_wall_reduces_to_the_axis_wall(self):
        """find_contact_wall_plane_3d on an axis plane == find_contact_wall_3d."""
        q = np.array([0.9, 0.2, 0.3, 0.1])
        q = q / np.linalg.norm(q)
        a, b, c = 26.0, 15.0, 20.0
        x, y, z = 12.0, 200.0, 200.0               # overlapping the x = 0 wall
        axis_res = find_contact_wall_3d(x, y, z, a, b, c, 2.4, 2.2, q, max(a, b, c), 0.0, 0, +1)
        plane_res = find_contact_wall_plane_3d(x, y, z, a, b, c, 2.4, 2.2, q, max(a, b, c),
                                               0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        self.assertIsNotNone(axis_res)
        self.assertIsNotNone(plane_res)
        self.assertAlmostEqual(axis_res[0], plane_res[0], places=9)
        self.assertAlmostEqual(axis_res[1], plane_res[1], places=12)


class TestFreeTop(unittest.TestCase):

    def test_no_lid_force_and_no_clip_below_the_container_height(self):
        p = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, T_active=0.0, save_data=False,
                   boundary_top='free')
        z = p.Lz - R_GRAN + 2.0                    # would overlap a lid by 2 um
        for use_numba in (False, True):
            p.use_numba = use_numba
            gs = _single(p, 200.0, 200.0, z)
            for k in range(gs.N):
                gs.contact_clips[k] = []
            F, _t, _c = compute_forces_3d(gs, p, np.random.default_rng(1))
            np.testing.assert_allclose(F[0], np.zeros(3), atol=1e-12)
        # a lidded box does push back
        p_lid = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, T_active=0.0, save_data=False)
        gs = _single(p_lid, 200.0, 200.0, z)
        F, _t, _c = compute_forces_3d(gs, p_lid, np.random.default_rng(1))
        self.assertLess(float(F[0, 2]), 0.0)

    def test_position_bounds_clamp_at_the_container_height_and_count(self):
        # V3.5: the clip sits INSIDE the wall by `max_overlap_frac * reach` so the
        # wall contact can engage and carry the load -- see `wall_clamp_margin`.
        p = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, save_data=False, boundary_top='free')
        gs = _single(p, 200.0, 200.0, 300.0)
        margin = -p.max_overlap_frac * R_GRAN
        gs.n_top_clamped = 0
        gs.z[0] = 350.0                            # still inside
        apply_position_bounds(gs, p)
        self.assertAlmostEqual(float(gs.z[0]), 350.0, places=12)
        self.assertEqual(gs.n_top_clamped, 0)
        gs.z[0] = 399.5                            # past Lz - r - margin
        apply_position_bounds(gs, p)
        self.assertAlmostEqual(float(gs.z[0]), p.Lz - R_GRAN - margin, places=12)
        self.assertEqual(gs.n_top_clamped, 1)

    def test_the_legacy_clamp_still_holds_granules_clear_of_the_wall(self):
        """`boundary.wall_clamp = legacy` reproduces the V2.7 +0.5 um standoff."""
        p = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, save_data=False,
                   boundary_top='free', boundary_wall_clamp='legacy')
        gs = _single(p, 200.0, 200.0, 300.0)
        gs.z[0] = 395.0
        apply_position_bounds(gs, p)
        self.assertAlmostEqual(float(gs.z[0]), p.Lz - R_GRAN - 0.5, places=12)

    def test_the_centre_stays_inside_the_box_in_every_mode(self):
        """The one thing the clip must guarantee: the neighbour and render grids
        need positions inside the box. Surfaces may now cross a wall; centres
        may not."""
        for mode in ('contact', 'force', 'legacy'):
            p = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, save_data=False,
                       boundary_top='free', boundary_wall_clamp=mode)
            gs = _single(p, 200.0, 200.0, 300.0)
            for probe in (-500.0, -1.0, 0.0, 399.0, 900.0):
                gs.z[0] = probe
                gs.x[0] = probe
                apply_position_bounds(gs, p)
                with self.subTest(clamp=mode, probe=probe):
                    self.assertGreaterEqual(float(gs.z[0]), 0.0)
                    self.assertLessEqual(float(gs.z[0]), p.Lz)
                    self.assertGreaterEqual(float(gs.x[0]), 0.0)
                    self.assertLessEqual(float(gs.x[0]), p.Lx)

    def test_cylinder_bounds_project_radially(self):
        p = _well()
        geom = boundary_geometry(p)
        gs = _single(p, geom.cx + geom.R_cyl - 5.0, geom.cy, 200.0)
        apply_position_bounds(gs, p)
        rho = float(np.hypot(gs.x[0] - geom.cx, gs.y[0] - geom.cy))
        margin = -p.max_overlap_frac * R_GRAN          # V3.5: the clip sits inside the wall
        self.assertAlmostEqual(rho, geom.R_cyl - R_GRAN - margin, places=9)
        self.assertAlmostEqual(float(gs.y[0]), geom.cy, places=12)


class TestVolumesAndCounts(unittest.TestCase):

    def test_domain_volume_and_base_area(self):
        p_box = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=600.0)
        self.assertEqual(domain_volume(p_box), 400.0 * 400.0 * 600.0)
        self.assertEqual(domain_base_area(p_box), 400.0 * 400.0)
        p_cyl = _well(Lz=600.0)
        self.assertAlmostEqual(domain_volume(p_cyl), np.pi * 200.0 ** 2 * 600.0, places=6)
        self.assertAlmostEqual(domain_base_area(p_cyl), np.pi * 200.0 ** 2, places=6)
        p2 = Params(mode='2D', Lx=400.0, Ly=600.0)
        self.assertEqual(domain_volume(p2), 400.0 * 600.0)
        # a cylinder is 3D-only: a 2D setup keeps the box
        self.assertEqual(boundary_geometry(Params(mode='2D', boundary_shape='cylinder')).shape_code, 0)

    def test_counts_follow_the_container(self):
        kw = dict(phi_solid_target=0.5, func_ratio=0.5, R_func_mean=20.0, R_func_std=0.0,
                  R_inert_mean=20.0, R_inert_std=0.0)
        p_box = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, **kw)
        p_cyl = Params(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, boundary_shape='cylinder',
                       boundary_top='free', **kw)
        n_box = sum(species_counts(resolve_species(p_box), p_box, mode='3D'))
        n_cyl = sum(species_counts(resolve_species(p_cyl), p_cyl, mode='3D'))
        self.assertAlmostEqual(n_cyl / n_box, np.pi / 4.0, places=2)

    def test_bed_height_sets_the_solid_fraction(self):
        s = C.Setup()
        s.domain.mode = '3D'
        s.domain.size_um = [1000.0, 1000.0, 1000.0]
        s.boundary.shape = 'cylinder'
        s.boundary.top = 'free'
        s.granules.bed_height_um = 400.0
        s.granules.bed_solid_fraction = 0.6
        p = s.to_params()
        self.assertAlmostEqual(p.phi_solid_target, 0.6 * 400.0 / 1000.0, places=12)
        # V_solid = phi_bed * A_base * H_bed
        self.assertAlmostEqual(p.phi_solid_target * domain_volume(p),
                               0.6 * domain_base_area(p) * 400.0, places=3)


class TestRSAPlacement(unittest.TestCase):

    def test_every_granule_inside_the_cylinder_and_the_bed(self):
        p = _well(Lz=600.0, phi_solid_target=0.2, func_ratio=1.0, R_func_mean=R_GRAN,
                  R_func_std=2.0, cell_surface_coverage=0.0, n_cells_per_granule=0,
                  packing_settle_steps=60, gravity_enabled=True, granule_density=1180.0,
                  packing_consolidation='gravity')
        gs = generate_packing_3d(p, seed=3)
        geom = boundary_geometry(p)
        rho = np.hypot(gs.x - geom.cx, gs.y - geom.cy)
        # V3.5: a granule's SURFACE may cross a wall by up to the clip allowance
        # (`max_overlap_frac * reach`), which is what lets the wall contact carry
        # its load instead of the clip. Its CENTRE may not -- the neighbour and
        # render grids need positions inside the box.
        slack = p.max_overlap_frac * float(np.max(gs.r_bound[:gs.N])) + 1e-9
        self.assertLessEqual(float(np.max(rho + gs.r_bound)), geom.R_cyl + slack)
        self.assertLessEqual(float(np.max(rho)), geom.R_cyl + 1e-9)
        self.assertGreaterEqual(float(np.min(gs.z - gs.r_bound)), -slack)
        self.assertGreaterEqual(float(np.min(gs.z)), -1e-9)
        self.assertLess(float(np.max(gs.z + gs.r_bound)), p.Lz + slack)
        self.assertGreater(gs.N, 50)


class TestValidation(unittest.TestCase):

    def _bad(self, **kw):
        s = C.Setup()
        s.domain.mode = kw.pop('mode', '3D')
        for k, v in kw.items():
            C.set_path(s, k, v)
        with self.assertRaises(ValueError):
            C.validate(s)

    def test_rejections(self):
        self._bad(mode='2D', **{'boundary.shape': 'cylinder'})
        self._bad(**{'boundary.shape': 'cylinder', 'boundary.mode': 'periodic'})
        self._bad(**{'boundary.top': 'free', 'boundary.mode': 'periodic'})
        self._bad(**{'boundary.shape': 'sphere'})
        self._bad(**{'boundary.top': 'open'})
        self._bad(**{'granules.bed_height_um': 400.0})              # needs a free top
        self._bad(**{'boundary.top': 'free', 'granules.bed_height_um': 1e4})   # taller than the container
        self._bad(**{'packing.consolidation': 'sink'})
        self._bad(**{'packing.consolidation': 'gravity', 'boundary.mode': 'periodic'})

    def test_accepts_the_well(self):
        s = C.Setup()
        s.domain.mode = '3D'
        s.domain.size_um = [2000.0, 2000.0, 2100.0]
        s.boundary.shape = 'cylinder'
        s.boundary.top = 'free'
        s.gravity.enabled = True
        s.packing.consolidation = 'auto'
        s.granules.bed_height_um = 1500.0
        C.validate(s)


if __name__ == '__main__':
    unittest.main()
