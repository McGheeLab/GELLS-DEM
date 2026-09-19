"""
Packing consolidation modes (V3.1).
===================================

The V2.7 settle pulled every granule toward the box centre for the whole
inflation (``packing_consolidation = 'centre'``), which in a lidded box below
random close packing consolidates the bed into a ball and leaves the corners
empty (the sweep3d artefact). Two modes replace it for new setups:

  * ``none``    — no consolidation: a closed box is filled uniformly to the
                  corners (the RSA distribution is preserved by the inflation);
  * ``gravity`` — a downward body force sediments the granules into a bed with
                  a free surface at the bottom of the container (box or
                  cylinder, 3D or the 2D dish slice).

Run:  python -m unittest tests.test_packing_consolidation -v
"""

import os
import sys
import unittest

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gels.engine import (  # noqa: E402
    Params, bed_solid_volume, bed_surface, boundary_geometry, domain_base_area,
    generate_packing, generate_packing_3d,
)


def _pack3d(**kw):
    base = dict(mode='3D', Lx=400.0, Ly=400.0, Lz=400.0, phi_solid_target=0.5, func_ratio=0.5,
                R_func_mean=20.0, R_func_std=2.0, R_inert_mean=20.0, R_inert_std=2.0,
                cell_surface_coverage=0.5, packing_settle_steps=300, save_data=False)
    base.update(kw)
    return Params(**base)


def _contacts(gs):
    """(contacts per granule, deepest overlap) from bounding radii."""
    dim = 3 if gs.is_3d else 2
    pos = np.ascontiguousarray(gs.pos[:, :dim])
    pairs = cKDTree(pos).query_pairs(2.0 * float(gs.r_bound.max()) + 1.0, output_type='ndarray')
    if len(pairs) == 0:
        return np.zeros(gs.N, dtype=int), 0.0
    d = np.linalg.norm(pos[pairs[:, 1]] - pos[pairs[:, 0]], axis=1)
    gap = d - gs.r_bound[pairs[:, 0]] - gs.r_bound[pairs[:, 1]]
    touching = gap < 1.0
    nc = np.bincount(pairs[touching].ravel(), minlength=gs.N)
    return nc, float(max(0.0, -float(gap.min())))


class TestNoneFillsTheBox(unittest.TestCase):

    def test_corners_are_occupied(self):
        p = _pack3d(packing_consolidation='none')
        gs = generate_packing_3d(p, seed=11)
        L = p.Lx
        q = L / 4.0
        pos = gs.pos[:, :3]
        expected_corner = gs.N / 64.0
        n_corners = 0
        for sx in (0, 1):
            for sy in (0, 1):
                for sz in (0, 1):
                    lo = np.array([sx, sy, sz], dtype=float) * (L - q)
                    inside = np.all((pos >= lo) & (pos <= lo + q), axis=1)
                    n = int(inside.sum())
                    self.assertGreater(n, 0.4 * expected_corner, f"corner {(sx, sy, sz)} nearly empty: {n}")
                    n_corners += n
        self.assertLess(abs(n_corners - 8 * expected_corner), 0.35 * 8 * expected_corner,
                        f"corner occupancy {n_corners} vs uniform {8 * expected_corner:.0f}")
        inner = np.all((pos >= q) & (pos <= L - q), axis=1)
        n_inner = int(inner.sum())
        self.assertLess(abs(n_inner - gs.N / 8.0), 0.35 * gs.N / 8.0,
                        f"inner occupancy {n_inner} vs uniform {gs.N / 8.0:.0f}")
        _, max_ov = _contacts(gs)
        self.assertLess(max_ov, 0.06 * float(np.mean(gs.r)))


class TestGravityBed(unittest.TestCase):

    def _check_bed(self, gs, p, phi_lo, phi_hi):
        bed = bed_surface(gs, p)
        V = bed_solid_volume(gs)
        self.assertGreater(bed['bed_envelope_volume'], 0.0)
        phi_bed = V / bed['bed_envelope_volume']
        H_est = V / (p.bed_phi_assumed * domain_base_area(p))
        mean_r = float(np.mean(gs.r))
        up = gs.z if gs.is_3d else gs.y
        L_up = p.Lz if gs.is_3d else p.Ly
        msg = (f"phi_bed={phi_bed:.3f}, height={bed['bed_height_mean']:.0f} "
               f"(p95 {bed['bed_height_p95']:.0f}) vs estimate {H_est:.0f}, N={gs.N}")
        self.assertGreater(phi_bed, phi_lo, msg)
        self.assertLess(phi_bed, phi_hi, msg)
        self.assertGreater(bed['bed_height_mean'], 0.75 * H_est, msg)
        self.assertLess(bed['bed_height_mean'], 1.3 * H_est, msg)
        # Inside the container. V3.5: a surface may cross a wall by up to the
        # clip allowance -- that is what lets the wall contact carry the load
        # instead of the clip (`wall_clamp_margin`) -- but a CENTRE may not.
        slack = p.max_overlap_frac * float(np.max(gs.r_bound[:gs.N])) + 1e-9
        self.assertGreaterEqual(float(np.min(up - gs.r_bound)), -slack)
        self.assertLessEqual(float(np.max(up + gs.r_bound)), L_up + slack)
        self.assertGreaterEqual(float(np.min(up[:gs.N])), -1e-9)
        self.assertLessEqual(float(np.max(up[:gs.N])), L_up + 1e-9)
        # converged: no deep overlaps, (almost) no floating granules
        nc, max_ov = _contacts(gs)
        self.assertLess(max_ov, 0.06 * mean_r, msg)
        floating = (nc == 0) & ((up - gs.r_bound) > 1.0)
        self.assertLessEqual(int(floating.sum()), max(2, int(0.01 * gs.N)), msg)
        return phi_bed, bed

    def test_3d_box_free_top(self):
        p = _pack3d(Lz=600.0, phi_solid_target=0.25, boundary_top='free', gravity_enabled=True,
                    granule_density=1180.0, packing_consolidation='gravity')
        gs = generate_packing_3d(p, seed=5)
        self._check_bed(gs, p, 0.45, 0.68)

    def test_3d_cylinder_free_top(self):
        p = _pack3d(Lz=600.0, phi_solid_target=0.25, boundary_shape='cylinder', boundary_top='free',
                    gravity_enabled=True, granule_density=1180.0, packing_consolidation='gravity')
        gs = generate_packing_3d(p, seed=6)
        geom = boundary_geometry(p)
        rho = np.hypot(gs.x - geom.cx, gs.y - geom.cy)
        # V3.5: surfaces may cross the wall by the clip allowance; centres may not
        slack = p.max_overlap_frac * float(np.max(gs.r_bound[:gs.N])) + 1e-9
        self.assertLessEqual(float(np.max(rho + gs.r_bound)), geom.R_cyl + slack)
        self.assertLessEqual(float(np.max(rho[:gs.N])), geom.R_cyl + 1e-9)
        self._check_bed(gs, p, 0.45, 0.68)

    def test_2d_dish(self):
        # explicit species: the legacy pair floors radii at 15 / 20 um
        species = [
            dict(name='coated', f=1.0, volume_fraction=0.5, color='#CC2222',
                 radius_mean=10.0, radius_std=1.0, radius_min=5.0),
            dict(name='bare', f=0.0, volume_fraction=0.5, color='#22AA22',
                 radius_mean=10.0, radius_std=1.0, radius_min=5.0),
        ]
        p = Params(mode='2D', Lx=500.0, Ly=800.0, phi_solid_target=0.3, species=species,
                   cell_surface_coverage=0.5, packing_settle_steps=300, save_data=False,
                   boundary_top='free', gravity_enabled=True, packing_consolidation='gravity',
                   bed_phi_assumed=0.8)
        gs = generate_packing(p, seed=7)
        self.assertLess(float(np.max(gs.r)), 14.0)
        self._check_bed(gs, p, 0.65, 0.95)


if __name__ == '__main__':
    unittest.main()
