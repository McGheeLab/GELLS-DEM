"""
Katz-Thompson permeability and geodesic tortuosity (V3.3).
==========================================================

Run:  python -m unittest tests.test_permeability -v
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

from gels.pore import (critical_pore, katz_thompson, kozeny_carman_Sv,  # noqa: E402
                       signed_field, specific_surface, tortuosity_geo)

N = 40


class TestCriticalPore(unittest.TestCase):

    def test_open_box(self):
        sd = np.full((N, N, N), 5.0)
        eps, lc = critical_pore(sd)
        self.assertAlmostEqual(eps, 1.0)
        self.assertAlmostEqual(lc, 5.0, places=3)

    def test_plugged_box_does_not_span(self):
        sd = np.full((N, N, N), 5.0)
        sd[:, :, N // 2 - 2:N // 2 + 2] = -1.0
        _eps, lc = critical_pore(sd)
        self.assertAlmostEqual(lc, 0.0, places=6)

    def test_recovers_a_known_channel_width(self):
        """l_c is the widest threshold that still spans, so it recovers the
        sd value inside a uniform channel."""
        sd = np.full((N, N, N), -1.0)
        sd[15:25, 15:25, :] = 7.0
        _eps, lc = critical_pore(sd)
        self.assertAlmostEqual(lc, 7.0, places=2)


class TestTortuosity(unittest.TestCase):

    def test_straight_channel_is_one(self):
        sd = np.full((N, N, N), -1.0)
        sd[15:25, 15:25, :] = 3.0
        tau, spans = tortuosity_geo(sd)
        self.assertTrue(spans)
        self.assertAlmostEqual(tau, 1.0, places=6)

    def test_blocked_returns_zero_not_nan(self):
        """history.json is written with json.dump/csv, which cannot carry NaN."""
        sd = np.full((N, N, N), -1.0)
        tau, spans = tortuosity_geo(sd)
        self.assertFalse(spans)
        self.assertEqual(tau, 0.0)
        self.assertFalse(np.isnan(tau))

    def test_bend_exceeds_one(self):
        sd = np.full((N, N, N), -1.0)
        sd[5:15, :, 0:N // 2 + 3] = 3.0
        sd[5:35, :, N // 2 - 3:N // 2 + 3] = 3.0
        sd[25:35, :, N // 2 - 3:N] = 3.0
        tau, spans = tortuosity_geo(sd)
        self.assertTrue(spans)
        self.assertGreater(tau, 1.0)

    def test_lateral_wrap_only_under_periodic(self):
        """A wall is not a mirror. A channel that only connects around x must
        span under periodic BCs and not under walls."""
        sd = np.full((N, N, N), -1.0)
        sd[0:3, 18:22, 0:N // 2 + 2] = 3.0          # enters at x ~ 0
        sd[N - 3:N, 18:22, N // 2 - 2:N] = 3.0      # leaves at x ~ L
        sd[0:3, 18:22, N // 2 - 2:N // 2 + 2] = 3.0
        sd[N - 3:N, 18:22, N // 2 - 2:N // 2 + 2] = 3.0
        _, spans_walls = tortuosity_geo(sd, periodic=False)
        _, spans_per = tortuosity_geo(sd, periodic=True)
        self.assertFalse(spans_walls)
        self.assertTrue(spans_per)


class TestAnalytic(unittest.TestCase):

    def test_specific_surface(self):
        r = np.array([10.0, 20.0, 30.0])
        self.assertAlmostEqual(specific_surface(r),
                               3 * np.sum(r ** 2) / np.sum(r ** 3), places=12)

    def test_signed_field_recovers_a_sphere_surface(self):
        pos = np.array([[50.0, 50.0, 50.0]])
        r = np.array([20.0])
        sd = signed_field(pos, r, (50, 50, 50), (100.0, 100.0, 100.0), 3)
        # the zero level set should sit at radius 20 from the centre
        voxel = 2.0
        solid = (sd <= 0).sum() * voxel ** 3
        self.assertAlmostEqual(solid / ((4 / 3) * np.pi * 20.0 ** 3), 1.0, delta=0.02)

    def test_degenerate_inputs_are_finite(self):
        self.assertEqual(kozeny_carman_Sv(0.0, 1.0), 0.0)
        self.assertEqual(kozeny_carman_Sv(1.0, 1.0), 0.0)
        self.assertEqual(kozeny_carman_Sv(0.4, 0.0), 0.0)
        self.assertEqual(specific_surface(np.zeros(3)), 0.0)


class TestChannelizationSensitivity(unittest.TestCase):
    """The reason Katz-Thompson is worth having at all."""

    def test_kt_sees_channelization_and_kc_does_not(self):
        M = 60
        r = np.full(50, 10.0)
        dispersed = np.full((M, M, M), -1.0)
        for i in range(0, M, 6):
            for j in range(0, M, 6):
                dispersed[i:i + 2, j:j + 2, :] = 1.0
        w = int(round(np.sqrt((dispersed > 0).sum() / M) / 2))
        coarsened = np.full((M, M, M), -1.0)
        coarsened[M // 2 - w:M // 2 + w, M // 2 - w:M // 2 + w, :] = float(w)

        out = []
        for sd in (dispersed, coarsened):
            eps, lc = critical_pore(sd)
            tau, spans = tortuosity_geo(sd)
            out.append((eps,
                        katz_thompson(lc, eps, tau if spans else 1.0),
                        kozeny_carman_Sv(eps, specific_surface(r))))

        # same porosity by construction
        self.assertAlmostEqual(out[0][0], out[1][0], places=6)
        # Katz-Thompson moves by orders of magnitude, Kozeny-Carman not at all
        self.assertGreater(out[1][1] / out[0][1], 10.0)
        self.assertAlmostEqual(out[1][2] / out[0][2], 1.0, places=9)


class TestMetricsIntegration(unittest.TestCase):

    def test_block_present_only_when_enabled(self):
        import contextlib
        import io
        from gels.engine import (Params, generate_packing, render_fields_species,
                                 group_species_fields, compute_metrics)
        keys = ('K_katz_thompson', 'pore_lc_radius', 'tortuosity_geo', 'K_kc_analytic')
        for enabled in (False, True):
            p = Params(mode='2D', Lx=400.0, Ly=400.0, Ngrid=48, phi_solid_target=0.55,
                       func_ratio=0.6, packing_settle_steps=30, save_data=False,
                       metrics_pore_field=enabled)
            with contextlib.redirect_stdout(io.StringIO()):
                gs = generate_packing(p, seed=3)
            phi_s = render_fields_species(gs, p)
            pf, pi_, pv = group_species_fields(gs, phi_s)
            m = compute_metrics(gs, p, pf, pi_, pv, 0.0, np.zeros((gs.N, 2)), phi_s=phi_s)
            for k in keys:
                self.assertEqual(k in m, enabled, f"{k} enabled={enabled}")
            if enabled:
                for k in keys:
                    if isinstance(m[k], float):
                        self.assertFalse(np.isnan(m[k]), k)


if __name__ == '__main__':
    unittest.main()
