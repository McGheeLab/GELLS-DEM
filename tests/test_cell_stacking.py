"""
Cells crawl on cells (V3.6 Phase 4).

**The physics, stated so it is falsifiable.** A cell's anchorage is the
*existing* motor-clutch expression evaluated on whatever it stands on:

* layer 0 -- ``E = gs.E_gran[i]``, ``nu = gs.nu_gran[i]``, ``g = traction_gain(...)``;
* layer >= 1 -- ``E = cells.stacking.substrate_E_kPa``,
  ``nu = substrate_poisson``, ``g = traction_gain(...) * cadherin_gain(f_cell_cell)``.

That single substitution IS the feature. The preference for the granule is a
consequence, not a rule, and it reproduces the V3.3 plan's prediction exactly:

    traction ratio cell/granule    0.178 at 10 kPa, 0.109 at 50, 0.091 on PMMA

falling as granules stiffen, which is the motor-clutch story. **Stiffness does
~91 % of the work**: `f_cell_cell = 1.0` alone moves the ratio only
0.178 -> 0.196, so a parity test that sets the cadherin coverage and not the
modulus proves nothing.

**The trap.** ``g`` enters `motor_clutch_force` TWICE -- as a prefactor and
inside ``k_sub/(k_sub + g k_opt)`` -- so ``F(g.h) != F.h``. The cadherin factor
is folded into ``g`` before the call and never multiplied onto the result;
`test_the_cadherin_factor_is_folded_into_g` is that constraint.

**State: one int8 array, `cell_layer`, and no host-cell pointer.** A pointer
would hold an absolute cell index, which `add_cells` invalidates -- and no
physics needs one: the substrate is a MATERIAL, not a particular neighbour, and
bridge force already accumulates onto the host granule's row, so a stacked cell
correctly pulls its granule THROUGH the cell beneath. The layer is DERIVED from
CSR rank every step (`k // cap`), which is the same rank the overcrowding rule
already uses, so the two cannot disagree and there is no stale-index trap.

**What had to change besides the substitution.** Before V3.6 a tolerated
overcrowded cell was frozen in whatever state it was seeded in -- ATTACHED -- so
it never spread and never bridged: `cell_stacking_max` was a senescence delay,
not a second storey. Freezing it was honest while there was nothing for it to
stand on. With stacking enabled it now follows the normal progression, and the
overcrowd clock keeps running.

Run:  python -m unittest tests.test_cell_stacking -v
"""

import contextlib
import io
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
    CellState, Params, cadherin_gain, motor_clutch_force, run,
    stacked_traction_force, stacking_enabled, traction_metrics,
)
from gels.presets import PRESETS  # noqa: E402


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# A bed seeded at 1.8 storeys, one species, uniform modulus -- so "parity" is a
# single pair of numbers rather than a per-granule distribution.
STACKED = dict(
    mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.5, func_ratio=1.0,
    save_data=False, save_fields=False, t_total=6.0, save_every_h=3.0,
    R_func_mean=20.0, R_func_std=0.0, R_inert_mean=20.0, R_inert_std=0.0,
    cell_surface_coverage=1.8, cell_capacity_coverage=1.0,
    cell_stacking_max=3.0, T_active=0.0, E_modulus=10.0, poisson_ratio=0.45,
)
FLAT = dict(STACKED, cell_surface_coverage=0.5)      # nobody is above the monolayer
PARITY = dict(cell_substrate_E=10.0, cell_substrate_poisson=0.45, f_cell_cell=1.0)


class TestTheSubstitution(unittest.TestCase):

    def test_the_ratio_and_that_stiffness_does_most_of_the_work(self):
        want = {10.0: 0.178, 50.0: 0.109, 3e6: 0.091}
        for E, expect in want.items():
            with self.subTest(E_kPa=E):
                p = Params(f_cell_cell=0.15)
                ratio = stacked_traction_force(p, 1.0, 1.0) / motor_clutch_force(E, p, 1.0, g=1.0)
                self.assertAlmostEqual(ratio, expect, places=3)
        # cadherin coverage alone barely moves it: 0.178 -> 0.196 at 10 kPa
        p1 = Params(f_cell_cell=1.0)
        r1 = stacked_traction_force(p1, 1.0, 1.0) / motor_clutch_force(10.0, p1, 1.0, g=1.0)
        self.assertAlmostEqual(r1, 0.196, places=3)

    def test_the_preference_strengthens_as_granules_stiffen(self):
        p = Params(f_cell_cell=0.15)
        ratios = [stacked_traction_force(p, 1.0, 1.0) / motor_clutch_force(E, p, 1.0, g=1.0)
                  for E in (10.0, 50.0, 200.0, 3e6)]
        self.assertTrue(all(a > b for a, b in zip(ratios, ratios[1:])), ratios)

    def test_the_cadherin_factor_is_folded_into_g(self):
        # If it were multiplied onto the RESULT the two would agree; they must not,
        # because g also sits inside k_sub/(k_sub + g k_opt).
        p = Params(f_cell_cell=0.15)
        cad = cadherin_gain(p)
        folded = stacked_traction_force(p, 1.0, 1.0)
        multiplied = motor_clutch_force(p.cell_substrate_E, p, 1.0,
                                        nu=p.cell_substrate_poisson, g=1.0) * cad
        self.assertNotAlmostEqual(folded, multiplied, places=6)

    def test_cadherin_gain_is_exactly_one_at_full_coverage(self):
        # This is what makes the parity run bit-identical rather than close.
        self.assertEqual(cadherin_gain(Params(f_cell_cell=1.0)), 1.0)

    def test_at_parity_the_substitution_is_a_no_op(self):
        p = Params(**dict(STACKED, cell_stacking_enabled=True, **PARITY))
        for fa in (0.2, 1.0):
            for g in (0.3, 1.0):
                self.assertAlmostEqual(
                    stacked_traction_force(p, fa, g),
                    motor_clutch_force(p.E_modulus, p, fa, nu=p.poisson_ratio, g=g),
                    places=12)


class TestTheLayer(unittest.TestCase):

    def test_it_is_csr_rank_over_capacity(self):
        _h, _s, _p, gs = quiet(run, Params(**STACKED), seed=3)
        C = int(gs.cell_offset[gs.N])
        self.assertGreater(C, 0)
        lay = np.asarray(gs.cell_layer[:C])
        self.assertGreater(int(np.sum(lay > 0)), 0, 'nothing is stacked')
        for i in range(gs.N):
            a, b = int(gs.cell_offset[i]), int(gs.cell_offset[i + 1])
            if b > a:
                # monotone non-decreasing in rank, and starts at 0
                self.assertEqual(int(lay[a]), 0)
                self.assertTrue(np.all(np.diff(lay[a:b]) >= 0))

    def test_a_bed_below_capacity_is_all_layer_zero(self):
        _h, _s, _p, gs = quiet(run, Params(**FLAT), seed=3)
        C = int(gs.cell_offset[gs.N])
        np.testing.assert_array_equal(gs.cell_layer[:C], 0)

    def test_it_is_written_whether_or_not_stacking_is_on(self):
        for on in (False, True):
            _h, _s, _p, gs = quiet(run, Params(**dict(STACKED,
                                                      cell_stacking_enabled=on)), seed=3)
            C = int(gs.cell_offset[gs.N])
            self.assertGreater(int(np.max(gs.cell_layer[:C])), 0)

    def test_it_reaches_the_snapshots(self):
        # The array has to survive a save/restore round trip, or a resumed run
        # would silently start every cell back on the granule.
        import shutil
        import tempfile
        from gels.engine import find_last_snapshot, load_run
        d = tempfile.mkdtemp(prefix='gels_stack_')
        try:
            p = Params(**dict(STACKED, save_data=True, output_dir=d,
                              save_fields=False, compress_archive=False))
            _h, _s, _p, gs = quiet(run, p, seed=3)
            snap = np.load(find_last_snapshot(d), allow_pickle=False)
            self.assertIn('cell_layer', snap.files)
            np.testing.assert_array_equal(snap['cell_layer'],
                                          gs.cell_layer[:int(gs.cell_offset[gs.N])])
            self.assertGreater(int(np.max(snap['cell_layer'])), 0)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestWhatItChanges(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.off = quiet(run, Params(**STACKED), seed=3)
        cls.on = quiet(run, Params(**dict(STACKED, cell_stacking_enabled=True)), seed=3)
        cls.par = quiet(run, Params(**dict(STACKED, cell_stacking_enabled=True,
                                           **PARITY)), seed=3)

    def test_stacked_cells_used_to_be_frozen_and_now_are_not(self):
        # The pre-existing behaviour being corrected: with stacking off, every
        # cell above the monolayer is still ATTACHED and has never bridged.
        gs = self.off[3]
        C = int(gs.cell_offset[gs.N])
        up = np.asarray(gs.cell_layer[:C]) > 0
        st = np.asarray(gs.cell_state[:C])
        self.assertTrue(np.all(st[up] == int(CellState.ATTACHED)))
        gs2 = self.on[3]
        C2 = int(gs2.cell_offset[gs2.N])
        up2 = np.asarray(gs2.cell_layer[:C2]) > 0
        st2 = np.asarray(gs2.cell_state[:C2])
        self.assertFalse(np.all(st2[up2] == int(CellState.ATTACHED)))

    def test_the_run_changes(self):
        self.assertNotAlmostEqual(float(self.off[0][-1]['F_mean']),
                                  float(self.on[0][-1]['F_mean']))

    def test_parity_isolates_the_substitution_from_the_unfreezing(self):
        # Turning stacking on does two things: it unfreezes the stacked cells
        # AND it gives them a cell substrate. At parity only the first happens,
        # so `par` is the control that separates them -- and `on` must differ
        # from it, or the substitution is doing nothing.
        self.assertNotAlmostEqual(float(self.par[0][-1]['F_mean']),
                                  float(self.on[0][-1]['F_mean']))

    def test_a_bed_with_nothing_stacked_is_bit_identical(self):
        a = quiet(run, Params(**FLAT), seed=3)
        b = quiet(run, Params(**dict(FLAT, cell_stacking_enabled=True)), seed=3)
        np.testing.assert_array_equal(a[3].pos[:a[3].N], b[3].pos[:b[3].N])

    def test_the_stacked_cells_pull_less_hard(self):
        gs = self.on[3]
        C = int(gs.cell_offset[gs.N])
        lay = np.asarray(gs.cell_layer[:C])
        F = np.sqrt(np.asarray(gs.cell_fx[:C]) ** 2 + np.asarray(gs.cell_fy[:C]) ** 2)
        up, down = (lay > 0) & (F > 0), (lay == 0) & (F > 0)
        if up.any() and down.any():
            self.assertLess(float(np.mean(F[up])), float(np.mean(F[down])))


class TestTheMetricsAndThePreset(unittest.TestCase):

    def test_the_keys_are_plain_finite_scalars(self):
        _h, _s, _p, gs = quiet(run, Params(**dict(STACKED,
                                                  cell_stacking_enabled=True)), seed=3)
        m = traction_metrics(gs, Params(**dict(STACKED, cell_stacking_enabled=True)))
        for k in ('n_cells_stacked', 'cell_layer_mean', 'cell_layer_max',
                  'stacked_traction_ratio'):
            self.assertIn(k, m)
            self.assertTrue(np.isfinite(float(m[k])), k)
        self.assertGreater(m['n_cells_stacked'], 0)
        self.assertLess(m['stacked_traction_ratio'], 1.0)

    def test_the_ratio_is_one_when_the_feature_is_off(self):
        p = Params(**STACKED)
        _h, _s, _p, gs = quiet(run, p, seed=3)
        self.assertEqual(traction_metrics(gs, p)['stacked_traction_ratio'], 1.0)

    def test_the_preset_actually_stacks(self):
        self.assertIn('stacked_monolayer', PRESETS)
        self.assertIn('cells.stacking.enabled=true', PRESETS['stacked_monolayer'])

    def test_both_twins_report_the_same_keys(self):
        keys = None
        for numba in (True, False):
            h, _s, _p, _gs = quiet(run, Params(**dict(STACKED, use_numba=numba,
                                                      cell_stacking_enabled=True)), seed=3)
            got = {k for k in h[-1] if k.startswith(('cell_layer', 'n_cells_stacked',
                                                     'stacked_'))}
            if keys is None:
                keys = got
            else:
                self.assertEqual(keys, got)
        self.assertTrue(keys)


if __name__ == '__main__':
    unittest.main(verbosity=2)
