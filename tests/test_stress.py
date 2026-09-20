"""V3.7: the Love-Weber virial stress.

The load-bearing cases are the ones that pin a CONVENTION, because a stress
tensor is only useful if its sign and its trace mean what the docstring says
they mean:

  * a compressed pair gives P > 0 (compression positive);
  * a contracting cell bridge gives P < 0 (tension) -- the sign
    `analysis/coarse_grain.py` had backwards;
  * the trace is divided by 2 in 2D, not by 3.
"""

import io
import contextlib
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gels.config import template_setup, setup_to_params          # noqa: E402
from gels.presets import apply_presets                            # noqa: E402
from gels.engine import (Params, generate_packing, compute_forces,  # noqa: E402
                         compute_forces_3d, record_stress, run)
from gels.stress import (STRESS_KEYS, stress_metrics, virial_stress,  # noqa: E402
                         stress_volume, fabric_stress, _pressure,
                         _dev_magnitude, _von_mises, _deviator)


def quiet(fn, *a, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*a, **kw)


def build(presets=(), seed=3, **over):
    s = template_setup()
    if presets:
        apply_presets(s, list(presets))
    p = setup_to_params(s)
    p.save_data = False
    for k, v in over.items():
        setattr(p, k, v)
    gs = quiet(generate_packing, p, seed=seed)
    cf = compute_forces_3d if gs.is_3d else compute_forces
    F, tq, contacts = quiet(cf, gs, p, np.random.default_rng(0))
    return p, gs, contacts


class TestConvention(unittest.TestCase):
    """The three facts the docstring asserts, each pinned by construction."""

    def _two_spheres(self, gap, boundary_mode='walls', straddle=False):
        """One overlapping pair and nothing else in contact.

        Truncating with ``gs.N = 2`` is not enough -- the neighbour search
        still sees the remaining rows -- so every other granule is parked on a
        coarse grid instead, far enough apart that the pair under test is the
        only contact in the system.
        """
        L = 400.0
        p = Params(mode='2D', Lx=L, Ly=L, save_data=False,
                   use_numba=False, boundary_mode=boundary_mode)
        gs = quiet(generate_packing, p, seed=1)
        N = gs.N
        r = 20.0
        # Park every other granule on a coarse grid AND shrink it: a 0.5 um
        # granule 90 um from its neighbour cannot contact anything, whatever
        # the packing happened to produce, so the pair under test is provably
        # the only contact in the system.
        k = np.arange(N)
        gs.r[:N] = 0.5
        gs.r_bound[:N] = 0.5
        gs.x[:N] = 40.0 + 90.0 * (k % 4)
        gs.y[:N] = 40.0 + 90.0 * (k // 4)
        gs.r[:2] = r
        gs.r_bound[:2] = r
        if straddle:                       # across the x = 0 periodic face
            gs.x[0], gs.y[0] = 5.0, 200.0
            gs.x[1], gs.y[1] = L - (2 * r - gap - 5.0), 200.0
        else:
            gs.x[0], gs.y[0] = 200.0, 200.0
            gs.x[1], gs.y[1] = 200.0 + 2 * r - gap, 200.0
        gs.vx[:N] = 0.0
        gs.vy[:N] = 0.0
        return p, gs

    @staticmethod
    def _only(contacts):
        """The single contact, as (i, j, F_normal)."""
        if hasattr(contacts, 'column'):
            return (int(contacts.column('i')[0]), int(contacts.column('j')[0]),
                    float(contacts.column('F_normal')[0]))
        c = contacts[0]
        return int(c['i']), int(c['j']), float(c['F_normal'])

    def test_a_compressed_pair_reads_positive_pressure(self):
        p, gs = self._two_spheres(gap=0.4)
        F, tq, contacts = quiet(compute_forces, gs, p, np.random.default_rng(0))
        self.assertGreater(len(contacts), 0, "the pair must actually be in contact")
        sig, parts = virial_stress(gs, p, contacts)
        P = _pressure(sig, 2)
        self.assertGreater(P, 0.0,
                           "compression positive: a squeezed pair must give P > 0")

    def test_the_analytic_value_of_a_single_contact(self):
        """sigma_xx = F_n * |l| / V exactly, for one contact along x."""
        p, gs = self._two_spheres(gap=0.4)
        F, tq, contacts = quiet(compute_forces, gs, p, np.random.default_rng(0))
        self.assertEqual(len(contacts), 1, "the pair must be the ONLY contact")
        i, j, Fn = self._only(contacts)
        l = abs(gs.x[j] - gs.x[i])
        V = stress_volume(gs, p)
        sig, _ = virial_stress(gs, p, contacts)
        self.assertAlmostEqual(sig[0, 0], Fn * l / V, places=12)
        # the contact is along x, so there is no yy or xy component
        self.assertAlmostEqual(sig[1, 1], 0.0, places=14)
        self.assertAlmostEqual(sig[0, 1], 0.0, places=14)

    def test_the_2d_trace_is_divided_by_two_not_three(self):
        sig = np.diag([3.0, 5.0, 0.0])
        self.assertAlmostEqual(_pressure(sig, 2), 4.0)
        self.assertAlmostEqual(_pressure(sig, 3), 8.0 / 3.0)
        # the bug this replaces: coarse_grain divided by 3 unconditionally, so a
        # 2D pressure came out 2/3 of its true value.
        self.assertNotAlmostEqual(_pressure(sig, 2), np.trace(sig) / 3.0)

    def test_a_contracting_bridge_is_tension(self):
        """The sign `analysis/coarse_grain.py` had backwards."""
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p2 = setup_to_params(s)
        p2.save_data = False
        p2.t_total = 4.0
        p2.save_every_h = 2.0
        h, _, _, gs2 = quiet(run, p2, seed=3)
        row = h[-1]
        self.assertGreater(row['stress_n_bridges'], 0, "no bridges formed to test")
        self.assertLess(row['P_active'], 0.0,
                        "a contracting bridge pulls its granules together: tension")
        self.assertGreater(row['P_contact'], 0.0,
                           "the granular skeleton is still in compression")


def setup_p():
    s = template_setup()
    p = setup_to_params(s)
    p.save_data = False
    return p


class TestKeys(unittest.TestCase):

    def test_every_key_is_present_and_plain(self):
        p, gs, contacts = build()
        record_stress(gs, p, contacts)
        m = stress_metrics(gs)
        self.assertEqual(set(m), set(STRESS_KEYS))
        for k, v in m.items():
            self.assertIsInstance(v, (int, float), k)
            self.assertTrue(np.isfinite(v), f"{k} = {v} is not finite")

    def test_a_never_stepped_system_still_reports_the_full_set(self):
        """`gs.stress_total` is None, not a zero tensor, until a step runs."""
        p = setup_p()
        gs = quiet(generate_packing, p, seed=1)
        m = stress_metrics(gs)
        self.assertEqual(set(m), set(STRESS_KEYS))
        self.assertEqual(m['P_vir'], 0.0)

    def test_an_empty_system_does_not_divide_by_zero(self):
        p = setup_p()
        gs = quiet(generate_packing, p, seed=1)
        gs.N = 0
        sig, parts = virial_stress(gs, p, [])
        self.assertTrue(np.all(sig == 0.0))
        self.assertEqual(parts['n_contacts'], 0)

    def test_active_frac_is_bounded(self):
        p = setup_p()
        gs = quiet(generate_packing, p, seed=1)
        gs.stress_contact = np.diag([1.0, -1.0, 0.0])   # P_contact == 0 exactly
        gs.stress_active = np.diag([2.0, 2.0, 0.0])
        gs.stress_total = gs.stress_contact + gs.stress_active
        m = stress_metrics(gs)
        self.assertGreaterEqual(m['stress_active_frac'], 0.0)
        self.assertLessEqual(m['stress_active_frac'], 1.0)


class TestTwins(unittest.TestCase):

    def test_the_twins_agree(self):
        """Kernel and reference must produce the same stress on ONE bed.

        Evaluated on a single packing, because the two force paths agree to
        rounding and not bit-for-bit, and two independently packed beds would
        not be comparable at all (the V3.5 note on FIRE).
        """
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        p.save_data = False
        gs = quiet(generate_packing, p, seed=5)

        p.use_numba = True
        Fk, _, ck = quiet(compute_forces, gs, p, np.random.default_rng(0))
        sig_k, _ = virial_stress(gs, p, ck)

        p.use_numba = False
        Fr, _, cr = quiet(compute_forces, gs, p, np.random.default_rng(0))
        sig_r, _ = virial_stress(gs, p, cr)

        np.testing.assert_allclose(sig_k, sig_r, rtol=1e-9, atol=1e-14)


class TestVolume(unittest.TestCase):

    def test_a_free_top_uses_the_bed_envelope_not_the_container(self):
        """Headroom is not sample: dividing by the container dilutes P."""
        from gels.engine import domain_volume, boundary_geometry
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice'])
        p = setup_to_params(s)
        p.save_data = False
        gs = quiet(generate_packing, p, seed=3)
        self.assertTrue(boundary_geometry(p, gs.mode).top_free)
        V = stress_volume(gs, p)
        self.assertLess(V, domain_volume(p),
                        "a bed with a free surface occupies less than its container")
        self.assertGreater(V, 0.0)

    def test_a_closed_box_uses_the_container(self):
        from gels.engine import domain_volume
        p = setup_p()
        gs = quiet(generate_packing, p, seed=3)
        self.assertAlmostEqual(stress_volume(gs, p), domain_volume(p))


class TestPeriodic(unittest.TestCase):

    def test_a_contact_across_a_periodic_face_is_minimum_imaged(self):
        """Without the minimum image the branch vector is nearly a box long."""
        conv = TestConvention()
        gap = 0.4
        p, gs = conv._two_spheres(gap, boundary_mode='periodic', straddle=True)
        F, tq, contacts = quiet(compute_forces, gs, p, np.random.default_rng(0))
        self.assertEqual(len(contacts), 1,
                         "the straddling pair must be the only contact")
        i, j, Fn = conv._only(contacts)
        sig, _ = virial_stress(gs, p, contacts)
        V = stress_volume(gs, p)
        # |l| must be the SHORT way round -- 2r - gap - the naive |x_j - x_i|
        # would be L - that, which is an order of magnitude larger and of the
        # opposite sign.
        expect = 2 * 20.0 - gap
        self.assertAlmostEqual(abs(sig[0, 0]) * V / abs(Fn), expect, places=6)
        self.assertGreater(sig[0, 0], 0.0, "still a compressed pair")


if __name__ == '__main__':
    unittest.main()


class TestDeviator(unittest.TestCase):
    """The deviator must be taken in d dimensions, not in the 3x3 embedding.

    A 2D stress is stored zero-padded as diag(P, P, 0). Subtracting tr/3 from
    that leaves diag(P/3, P/3, -2P/3), whose magnitude is P/sqrt(3) = 0.577 P
    -- a shear invented entirely by the padding. Three 2D beds reported
    q/p = 0.579, 0.581 and 0.652 before this was caught, the first two being
    the artefact almost neat.
    """

    def test_an_isotropic_2d_state_has_no_shear(self):
        iso = np.diag([2.0, 2.0, 0.0])
        self.assertAlmostEqual(_dev_magnitude(iso, 2), 0.0, places=15)
        self.assertAlmostEqual(_von_mises(iso, 2), 0.0, places=15)
        # and the artefact it replaces, for the record
        s3 = iso - (np.trace(iso) / 3.0) * np.eye(3)
        self.assertAlmostEqual(np.sqrt(0.5 * np.sum(s3 * s3)) / 2.0,
                               1.0 / np.sqrt(3.0), places=12)

    def test_an_isotropic_3d_state_has_no_shear(self):
        self.assertAlmostEqual(_dev_magnitude(np.diag([2.0, 2.0, 2.0]), 3), 0.0,
                               places=15)

    def test_3d_von_mises_is_the_standard_one(self):
        """Uniaxial diag(3,1,1) has von Mises 2."""
        self.assertAlmostEqual(_von_mises(np.diag([3.0, 1.0, 1.0]), 3), 2.0, places=12)

    def test_the_2d_deviator_is_traceless_in_2d(self):
        d = _deviator(np.diag([3.0, 1.0, 0.0]), 2)
        self.assertEqual(d.shape, (2, 2))
        self.assertAlmostEqual(np.trace(d), 0.0, places=15)


class TestFabric(unittest.TestCase):
    """Stress-force-fabric (Rothenburg & Bathurst 1989)."""

    def test_the_isotropic_identity_for_the_pressure(self):
        """P must equal (Nc/V) <f_n |l|> / d exactly.

        An independent route to the same number: it goes through the mean of a
        scalar product rather than through the tensor, so it checks the trace
        convention and the volume at once.
        """
        p, gs, contacts = build(['soft_gel_dish_slice'])
        sig, parts = virial_stress(gs, p, contacts)
        dim = 3 if gs.is_3d else 2
        col = contacts.column
        i = np.asarray(col('i'), int)
        j = np.asarray(col('j'), int)
        fn = np.asarray(col('F_normal'), float)
        pos = np.column_stack((gs.x[:gs.N], gs.y[:gs.N],
                               gs.z[:gs.N] if gs.is_3d else np.zeros(gs.N)))
        lmag = np.linalg.norm(pos[j] - pos[i], axis=1)
        P_direct = (len(fn) / parts['volume']) * np.mean(fn * lmag) / dim
        self.assertAlmostEqual(_pressure(sig, dim) / P_direct, 1.0, places=9)

    def test_sff_closes_on_the_contact_stress(self):
        """The prediction must land within ~20 % of the measured contact q/p.

        Measured 0.82-1.04 across four reference beds, with and without cells.
        It is NOT exact, and is not meant to be: the tangential force is not on
        the contact record, so `sff_closure` is reported rather than absorbed,
        and its distance from 1 is the tangential plus higher-order share.
        """
        for presets in (['soft_gel_dish_slice'],
                        ['soft_gel_dish_slice', 'fibroblast_realistic']):
            with self.subTest(presets=presets):
                p, gs, contacts = build(presets)
                fab = fabric_stress(gs, p, contacts)
                self.assertGreater(fab['sff_closure'], 0.6, presets)
                self.assertLess(fab['sff_closure'], 1.4, presets)

    def test_sff_is_compared_against_contact_not_total(self):
        """Pairing SFF with the TOTAL stress is wrong once cells pull.

        a_c and a_n are built from contact normals and contact forces, so the
        only shear they can explain is the contact network's own.
        """
        s = template_setup()
        apply_presets(s, ['soft_gel_dish_slice', 'fibroblast_realistic'])
        p = setup_to_params(s)
        p.save_data = False
        p.t_total = 4.0
        p.save_every_h = 2.0
        h, _, _, _ = quiet(run, p, seed=3)
        row = h[-1]
        self.assertGreater(row['stress_n_bridges'], 0)
        # the cells dominate the total shear, so the two ratios must differ
        self.assertGreater(row['stress_q_over_p'], 3.0 * row['sff_q_over_p'])
        # yet the contact-paired closure still holds
        self.assertGreater(row['sff_closure'], 0.6)
        self.assertLess(row['sff_closure'], 1.4)

    def test_a_perfectly_isotropic_fabric_has_zero_anisotropy(self):
        """Contact normals on a uniform ring give a_c = 0."""
        from gels.stress import _anisotropy
        th = np.arange(2048) * (2 * np.pi / 2048)
        n = np.column_stack((np.cos(th), np.sin(th), np.zeros_like(th)))
        fabric = (n.T @ n) / len(th)
        self.assertAlmostEqual(_anisotropy(fabric, 2), 0.0, places=10)

    def test_the_patch_diagnostic_is_reported(self):
        """a_contact / min(r) -- the flag on the small-strain contact law."""
        p, gs, contacts = build(['soft_gel_dish_slice'])
        fab = fabric_stress(gs, p, contacts)
        self.assertGreater(fab['stress_patch_p95'], 0.0)
        self.assertGreaterEqual(fab['stress_patch_max'], fab['stress_patch_p95'])
        # 2D hydrogel beds sit comfortably inside the small-strain regime
        self.assertLess(fab['stress_patch_p95'], 0.3)
