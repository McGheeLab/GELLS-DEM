"""
Cell types, traction stress and cell strain energy (V3.6 Phase 3).

Three things, one subject: what the cell IS.

**The type** (`gels/celltypes/`). A cell type is the set of measured properties
that distinguish one cell from another, each carrying its source. Only
`fibroblast` has been through the literature review in
``CodeLog/References/fibroblast_parameters.md``; `msc` exists because an
interface with one implementation is a guess, and says so in its own report.

**The traction as a stress.** The cell can only pull as hard as its adhesion can
hold, and the adhesion is a patch of the cell's footprint, so the ceiling is
``sigma x A`` -- which GROWS as the cell spreads instead of being a number
chosen per run. ``F_max_per_cell`` stays as an absolute backstop.

The stress is NOT a constant, and that is the one thing the reference file
explicitly forbids (section 2, item 3): ``sigma_FA = rho_bond F_b plog(gamma/e)``
scales with engaged-bond density. So `cells.traction.stress_Pa` is the value at
saturating ligand and the engine multiplies it by the Langmuir gain and the
clutch engagement fraction.

**The strain energy.** A loaded cell is a spring in series with what it grips:
``1/k = 1/k_cell + 1/k_sub``, and ``U = F^2/2k``. This is the quantity traction
force microscopy reports, so it is a direct comparison with experiment the model
did not previously offer. On a granule stiffer than ~50 kPa the CELL is the soft
element, so U barely moves with granule modulus -- itself a testable claim, and
`test_the_cell_is_the_soft_element` is the test of it.

What the instrumentation says about the model, recorded rather than tuned away:
on a 24 h 2D bed the model's cells reach **0.044 pJ per loaded cell** and a
**21.8 Pa** footprint traction, against 0.1-10 pJ and ~300 Pa (Gaudet 2003) for
a fibroblast on flat TFM. About 10x low. That is a calibration target, and the
point of computing it is that it is now visible.

Run:  python -m unittest tests.test_celltypes -v
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

import gels.celltypes as CT  # noqa: E402
from gels.celltypes import CellType, Measured, apply_cell_type, available, get  # noqa: E402
from gels.config import FLAT_MAP, setup_to_params, template_setup  # noqa: E402
from gels.engine import (  # noqa: E402
    Params, adhesion_force_ceiling, cell_projected_area, cell_strain_energy,
    cell_substrate_stiffness, compute_forces, generate_packing, motor_clutch_force,
    run, system_energy, traction_metrics,
)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


BED = dict(
    mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.5, func_ratio=0.6,
    save_data=False, save_fields=False, t_total=4.0, save_every_h=2.0,
    R_func_mean=20.0, R_func_std=0.0, R_inert_mean=20.0, R_inert_std=0.0,
    cell_surface_coverage=1.0, T_active=0.0,
)


def bed(**over):
    p = Params(**dict(BED, **over))
    gs = quiet(generate_packing, p, seed=3)
    F, _tq, c = quiet(compute_forces, gs, p, np.random.default_rng(0))
    return p, gs, F, c


class TestTheRegistry(unittest.TestCase):

    def test_the_folder_is_the_registry(self):
        av = available()
        self.assertIn('fibroblast', av)
        self.assertIn('msc', av)
        self.assertEqual(CT.DEFAULT_TYPE, 'fibroblast')

    def test_an_unknown_type_says_what_there_is(self):
        with self.assertRaises(KeyError) as cm:
            get('t_cell')
        self.assertIn('fibroblast', str(cm.exception))

    def test_names_are_case_insensitive(self):
        self.assertIs(get('Fibroblast'), get('fibroblast'))

    def test_every_number_carries_a_source(self):
        for name in available():
            ct = get(name)
            for f in ct.__dataclass_fields__:
                m = getattr(ct, f)
                if isinstance(m, Measured):
                    self.assertTrue(m.source.strip(), f'{name}.{f} has no source')

    def test_the_report_flags_assumptions_and_out_of_range_values(self):
        r = get('msc').report()
        self.assertIn('ASSUMPTION', r)
        self.assertIn('NOT YET REVIEWED', r)
        self.assertNotIn('OUTSIDE the reported range', get('fibroblast').report())


class TestTheTypeIsSelfConsistent(unittest.TestCase):
    """The checks a plausible-looking type fails and a real one passes."""

    def test_sigma_times_area_reproduces_the_measured_whole_cell_traction(self):
        for name in available():
            with self.subTest(name=name):
                ct = get(name)
                F_derived = ct.adhesion_force_nN(1.0, 1.0, ct.engagement())
                F_meas = float(ct.total_traction_nN)
                self.assertLess(abs(F_derived / F_meas - 1.0), 0.5,
                                f'{name}: sigma x A = {F_derived:.0f} nN but the measured '
                                f'whole-cell traction is {F_meas:.0f} nN')

    def test_its_geometry_agrees_with_the_engine(self):
        # The type duplicates the footprint expression so it can be inspected
        # without importing the engine; the engine stays the authority.
        for name in available():
            ct = get(name)
            for sf in (0.0, 0.37, 1.0):
                self.assertAlmostEqual(
                    ct.spread_area_um2(sf),
                    float(cell_projected_area(sf, float(ct.diameter_um),
                                              float(ct.spread_height_um))),
                    places=9, msg=f'{name} at spread {sf}')

    def test_the_strain_energy_lands_in_the_measured_band(self):
        # 0.1-10 pJ per cell by TFM, at the type's own whole-cell traction.
        for name in available():
            ct = get(name)
            U = ct.strain_energy_nN_um(float(ct.total_traction_nN), 394.0) / 1000.0
            self.assertTrue(0.05 <= U <= 20.0, f'{name}: {U:.3f} pJ')

    def test_every_override_is_a_real_config_key(self):
        keys = {k for k, _ in FLAT_MAP}
        for name in available():
            for ov in get(name).to_overrides():
                self.assertIn(ov.split('=')[0], keys, f'{name}: {ov}')

    def test_applying_one_reaches_params_and_records_itself(self):
        su = template_setup()
        apply_cell_type(su, 'fibroblast')
        self.assertEqual(su.meta.cell_type, 'fibroblast')
        p = setup_to_params(su)
        ct = get('fibroblast')
        self.assertEqual(p.cell_type, 'fibroblast')
        self.assertAlmostEqual(p.cell_traction_stress_Pa, float(ct.traction_stress_Pa))
        self.assertAlmostEqual(p.cell_migration_speed, float(ct.migration_speed_um_per_h))
        self.assertAlmostEqual(p.cell_series_stiffness, float(ct.k_cell_nN_per_um))

    def test_two_types_differ_in_what_they_do(self):
        su_f, su_m = template_setup(), template_setup()
        apply_cell_type(su_f, 'fibroblast')
        apply_cell_type(su_m, 'msc')
        pf, pm = setup_to_params(su_f), setup_to_params(su_m)
        self.assertNotEqual(pf.cell_traction_stress_Pa, pm.cell_traction_stress_Pa)
        self.assertNotEqual(pf.cell_doubling_time, pm.cell_doubling_time)
        self.assertNotEqual(pf.cell_migration_speed, pm.cell_migration_speed)


class TestTheTractionCeiling(unittest.TestCase):

    def test_off_by_default(self):
        _p, gs, _F, _c = bed()
        self.assertIsNone(adhesion_force_ceiling(gs, Params(**BED)))

    def test_it_grows_as_the_cell_spreads(self):
        p, gs, _F, _c = bed(cell_traction_stress_Pa=4000.0)
        gs.spread_fraction[:gs.N] = 0.0
        rounded = adhesion_force_ceiling(gs, p).copy()
        gs.spread_fraction[:gs.N] = 1.0
        spread = adhesion_force_ceiling(gs, p)
        self.assertTrue(np.all(spread > rounded))
        # volume conservation: A_spread/A_sphere = d/h = 4 at the defaults
        np.testing.assert_allclose(spread / rounded,
                                   p.cell_diameter / p.cell_height_spread, rtol=1e-12)

    def test_it_is_sigma_times_area_in_the_right_units(self):
        p, gs, _F, _c = bed(cell_traction_stress_Pa=4000.0, cell_adhesion_area_frac=0.08)
        gs.spread_fraction[:gs.N] = 1.0
        A = float(cell_projected_area(1.0, p.cell_diameter, p.cell_height_spread))
        eng = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)
        want = 4000.0 * eng * 0.08 * A * 1e-3     # Pa * um^2 = 1e-3 nN
        np.testing.assert_allclose(adhesion_force_ceiling(gs, p), want, rtol=1e-12)

    def test_it_scales_with_the_ligand_gain_not_with_nothing(self):
        # The reference file forbids a density-independent adhesion stress.
        p = Params(**dict(BED, cell_traction_stress_Pa=4000.0))
        args = (10.0, p, 1.0)
        full = motor_clutch_force(*args, g=1.0, F_adh=50.0)
        half = motor_clutch_force(*args, g=0.5, F_adh=50.0)
        self.assertLess(half, full)

    def test_it_binds_below_F_max_and_is_never_above_it(self):
        p = Params(**dict(BED, F_max_per_cell=1e6))
        free = motor_clutch_force(1e6, p, 1.0, g=1.0)
        self.assertGreater(free, 5.0)
        self.assertAlmostEqual(motor_clutch_force(1e6, p, 1.0, g=1.0, F_adh=5.0), 5.0)
        # whichever ceiling is lower wins, and neither can raise the force
        p2 = Params(**dict(BED, F_max_per_cell=2.0))
        self.assertAlmostEqual(motor_clutch_force(1e6, p2, 1.0, g=1.0, F_adh=5.0), 2.0)
        self.assertAlmostEqual(motor_clutch_force(1e6, p, 1.0, g=1.0, F_adh=1e9), free)

    def test_it_does_not_bind_at_the_bare_default_motor_count(self):
        # Worth pinning: at the shipped n_motors = 50 the motor-clutch force is
        # ~23 nN and a fully spread fibroblast adhesion can hold 366 nN, so the
        # ceiling is inert. It only becomes the binding constraint at a realistic
        # motor count -- which is what `--cell-type fibroblast` sets.
        p = Params(**dict(BED, cell_traction_stress_Pa=4000.0))
        self.assertLess(motor_clutch_force(1e6, p, 1.0, g=1.0), 30.0)
        _p, gs, _F, _c = bed(cell_traction_stress_Pa=4000.0)
        gs.spread_fraction[:gs.N] = 1.0
        self.assertGreater(float(np.min(adhesion_force_ceiling(gs, p))), 300.0)

    def test_the_twins_agree_on_the_capped_force(self):
        # NOT by comparing two runs: bridging is statistically equivalent between
        # the twins, not identical, so per-cell forces legitimately differ. The
        # sharp test is the leaf itself, on identical arguments.
        from gels.kernels.cells import mc_force_k
        p = Params(**dict(BED, F_max_per_cell=1e6))
        F_stall = p.n_motors * p.F_motor_stall
        k_opt = p.n_clutches * p.k_clutch
        eng = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)
        a_cell = p.cell_diameter / 2.0
        for E in (1.0, 10.0, 3e6):
            for g in (0.2, 1.0):
                for F_adh in (0.0, 5.0, 1e9):
                    with self.subTest(E=E, g=g, F_adh=F_adh):
                        ref = motor_clutch_force(
                            E, p, 1.0, nu=p.poisson_ratio, g=g,
                            F_adh=(F_adh if F_adh > 0 else None))
                        ker = mc_force_k(E, p.poisson_ratio, 1.0, g, F_stall, a_cell,
                                         k_opt, eng, p.F_max_per_cell, F_adh)
                        self.assertAlmostEqual(ref, ker, places=12)


class TestTheStrainEnergy(unittest.TestCase):

    def test_zero_without_cells_and_without_load(self):
        p, gs, _F, _c = bed(n_cells_per_granule=0, cell_coverage=0.0,
                            cell_surface_coverage=0.0)
        self.assertEqual(cell_strain_energy(gs, p), 0.0)
        p2, gs2, _F2, _c2 = bed()
        gs2.cell_fx[:] = 0.0
        gs2.cell_fy[:] = 0.0
        gs2.cell_fz[:] = 0.0
        self.assertEqual(cell_strain_energy(gs2, p2), 0.0)

    def test_it_is_F_squared_over_two_k_in_series(self):
        p, gs, _F, _c = bed()
        C = int(gs.cell_offset[gs.N])
        self.assertGreater(C, 0)
        gs.cell_fx[:] = 0.0
        gs.cell_fy[:] = 0.0
        gs.cell_fz[:] = 0.0
        gs.cell_bridge_target[:C] = -1
        gs.cell_fx[0] = 30.0
        k_sub = cell_substrate_stiffness(gs, p)[int(gs.cell_granule_id[0])]
        k = 1.0 / (1.0 / p.cell_series_stiffness + 1.0 / k_sub)
        self.assertAlmostEqual(cell_strain_energy(gs, p), 0.5 * 30.0 ** 2 / k, places=9)

    def test_a_bridging_cell_is_in_series_with_two_granules(self):
        p, gs, _F, _c = bed()
        C = int(gs.cell_offset[gs.N])
        gs.cell_fx[:] = 0.0
        gs.cell_fy[:] = 0.0
        gs.cell_fz[:] = 0.0
        gs.cell_bridge_target[:C] = -1
        gs.cell_fx[0] = 30.0
        u_one = cell_strain_energy(gs, p)
        gs.cell_bridge_target[0] = (int(gs.cell_granule_id[0]) + 1) % gs.N
        u_two = cell_strain_energy(gs, p)
        self.assertGreater(u_two, u_one)      # softer series -> more stored

    def test_the_cell_is_the_soft_element(self):
        # The claim: on a stiff granule U is set by k_cell and barely moves with
        # the granule modulus. Stated as a prediction, so it can be wrong.
        ct = get('fibroblast')
        F = float(ct.total_traction_nN)
        soft = ct.strain_energy_nN_um(F, 394.0)      # 10 kPa granule
        rigid = ct.strain_energy_nN_um(F, 1e5)       # PMMA
        self.assertLess(abs(soft / rigid - 1.0), 0.05)

    def test_it_reaches_the_energy_audit(self):
        p, gs, _F, c = bed()
        _tot, parts = system_energy(gs, p, c)
        self.assertIn('cell', parts)

    def test_and_is_zero_during_packing_so_FIRE_is_untouched(self):
        # relax_packing runs AFTER _initialize_cells, so this has to hold or the
        # V3.5 trust region would have silently changed.
        p, gs, _F, c = bed(packing_relax='fire')
        _tot, parts = system_energy(gs, p, c)
        self.assertGreaterEqual(parts['cell'], 0.0)

    def test_the_metrics_are_plain_finite_scalars(self):
        p, gs, _F, _c = bed(cell_traction_stress_Pa=4000.0)
        for k, v in traction_metrics(gs, p).items():
            self.assertIsInstance(v, (int, float), k)
            self.assertTrue(np.isfinite(float(v)), k)

    def test_the_two_stresses_differ_by_the_area_fraction(self):
        p, gs, _F, _c = bed(cell_traction_stress_Pa=4000.0, cell_adhesion_area_frac=0.08)
        C = int(gs.cell_offset[gs.N])
        gs.cell_fx[:] = 0.0
        gs.cell_fy[:] = 0.0
        gs.cell_fz[:] = 0.0
        gs.cell_fx[:C] = 5.0
        m = traction_metrics(gs, p)
        self.assertAlmostEqual(m['traction_stress_mean_Pa'] * 0.08,
                               m['traction_stress_footprint_Pa'], places=6)


class TestItIsOffByDefault(unittest.TestCase):

    def test_the_default_run_is_unchanged(self):
        a = quiet(run, Params(**BED), seed=3)
        b = quiet(run, Params(**BED), seed=3)
        np.testing.assert_array_equal(a[3].pos[:a[3].N], b[3].pos[:b[3].N])
        self.assertEqual(Params().cell_traction_stress_Pa, 0.0)

    def test_the_ceiling_binds_on_a_rigid_granule_and_not_on_a_soft_one(self):
        # A finding worth pinning, not just a switch test. The motor-clutch
        # force carries a factor beta = k_sub/(k_sub + g k_opt) that is 0.165 on
        # a 10 kPa granule and ~1 on PMMA, so at a realistic motor count:
        #
        #   10 kPa    motor-clutch  30 nN  <  91 nN rounded-cell adhesion ceiling
        #   50 kPa                  90 nN  ~  91
        #   PMMA                   182 nN  >  91   <- the adhesion is the limit
        #
        # So this is not a cap that fires everywhere: it is the statement that on
        # a stiff substrate an UNSPREAD cell cannot pull as hard as its motors
        # could, because it has not built the adhesion area yet.
        kw = dict(n_motors=400, n_clutches=400, F_max_per_cell=400.0)
        for E in (10.0, 3e6):
            free = motor_clutch_force(E, Params(**dict(BED, **kw)), 1.0, g=1.0)
            rounded_ceiling = 4000.0 * 0.909090909 * 0.08 * float(
                cell_projected_area(0.0, 20.0, 5.0)) * 1e-3
            binds = motor_clutch_force(E, Params(**dict(BED, **kw)), 1.0, g=1.0,
                                       F_adh=rounded_ceiling) < free - 1e-12
            self.assertEqual(binds, E > 1e3,
                             f'at E = {E} kPa the adhesion ceiling should '
                             f'{"bind" if E > 1e3 else "not bind"}')

    def test_NEGATIVE_the_ceiling_never_binds_at_the_fibroblast_numbers(self):
        """A negative result, pinned so it is not rediscovered.

        With the fibroblast's measured geometry and stress the adhesion ceiling
        is never the binding constraint in a run -- on 10 kPa, on 50 kPa and on
        PMMA, at 1, 2, 4 and 8 h, ``F_mean`` is bit-identical with it on and off.

        The reason is that the two ramp TOGETHER: the motor-clutch force carries
        `fa_maturity`, and the ceiling carries `spread_fraction`, and both start
        near zero and reach one over a few hours. At full spread the adhesion can
        hold 366 nN while the motors deliver at most 182 nN even on a rigid
        granule.

        That is not a wasted model -- it is the statement that
        ``F_max_per_cell = 150-200 nN`` is a PHYSICALLY REASONABLE cap rather
        than an arbitrary one, since the cell turns out to pull at about half its
        adhesion's capacity. The value of sigma x A here is the two reported
        stresses and the strain energy, not a cap that fires.
        """
        kw = dict(n_motors=400, n_clutches=400, F_max_per_cell=400.0,
                  E_modulus=3e6, contact_E_cap=50.0, mc_dem_enabled=False,
                  save_every_h=0.5)
        for t in (1.0, 4.0):
            with self.subTest(t_total=t):
                off = quiet(run, Params(**dict(BED, t_total=t, **kw)), seed=3)[0][-1]
                on = quiet(run, Params(**dict(BED, t_total=t,
                                              cell_traction_stress_Pa=4000.0, **kw)),
                           seed=3)[0][-1]
                self.assertEqual(float(off['F_mean']), float(on['F_mean']))

    def test_but_the_mechanism_works_when_the_adhesion_is_the_limit(self):
        # Shrink the adhesion area and it binds, which is what makes
        # `adhesion_area_frac` the calibration knob the negative result points at.
        kw = dict(n_motors=400, n_clutches=400, F_max_per_cell=400.0,
                  E_modulus=3e6, contact_E_cap=50.0, mc_dem_enabled=False,
                  t_total=8.0, save_every_h=4.0)
        off = quiet(run, Params(**dict(BED, **kw)), seed=3)[0][-1]
        on = quiet(run, Params(**dict(BED, cell_traction_stress_Pa=4000.0,
                                      cell_adhesion_area_frac=0.01, **kw)), seed=3)[0][-1]
        self.assertLess(float(on['F_mean']), float(off['F_mean']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
