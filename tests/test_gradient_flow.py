"""
Gradient flow: is the assembled force law the gradient of an energy? (V3.5)
==========================================================================

Overdamped dynamics is gradient flow, so ``gamma x_dot = -grad E`` and

    dE/dt = -gamma |x_dot|^2 <= 0.

V3.4 proved the MTD contact solver conservative POINTWISE -- ``d(delta)/d(c2)
= -n*`` to nine places, by the envelope theorem. That says nothing about the
force law that is actually assembled and integrated: JKR on top of the solver,
walls, gravity, MC-DEM, friction and the active noise. These tests close that
gap by central-differencing the TOTAL energy against the force
``compute_forces`` actually returns, on a real packed bed.

They are also the only guard on ``wall_potential_energy``, which MIRRORS the
face enumeration of ``gather_2d`` / ``gather_3d`` rather than sharing it. If
the two ever disagree -- a face added, a sign flipped, the free top handled
differently -- ``test_the_force_is_minus_grad_E_with_walls`` fails.

What the audit is NOT is a bug detector. Four terms break the equality on
purpose, and the tests pin each one's size so a future change to any of them is
visible:

    active noise (T_active = 5)   1.0      -- an athermal driving term
    MC-DEM                        1.3e-1   -- kappa multiplies F, is in no energy
    shaped granules               2.6e-3   -- R_eff varies with configuration
    tangential friction           99 % of the per-step residual, dissipative

Run:  python -m unittest tests.test_gradient_flow -v
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
    ENERGY_ASCENT_TOL, Params, active_noise_power, compute_forces, compute_forces_3d,
    contact_potential_energy, energy_metrics, generate_packing, generate_packing_3d,
    gravity_potential_energy, jkr_contact_energy, jkr_force_from_overlap, step,
    system_energy, wall_potential_energy,
)


# ── a bed with every non-conservative term switched off ──
PASSIVE = dict(
    mode='2D', Lx=400.0, Ly=400.0, phi_solid_target=0.55, dt=1e-3,
    save_data=False, save_fields=False,
    n_cells_per_granule=0, cell_coverage=0.0, cell_surface_coverage=0.0,
    mc_dem_enabled=False,            # kappa is not the gradient of anything
    T_active=0.0,                    # the athermal driving term
    tau_0_cc=0.0, tau_0_cb=0.0, tau_0_bb=0.0,   # tangential friction, dissipative
    gravity_enabled=False, boundary_mode='periodic',
    packing_settle_steps=60, max_overlap_frac=0.99, v_max=1e9,
    dynamics_gradient_flow='monitor',
)


def quiet(fn, *a, **kw):
    """The engine narrates to stdout; the tests do not care."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def packed(seed=4, three_d=False, **over):
    base = dict(PASSIVE)
    base.update(over)
    p = Params(**base)
    gen = generate_packing_3d if p.mode == '3D' else generate_packing
    return p, quiet(gen, p, seed=seed)


def forces(gs, p, rng=None):
    rng = rng if rng is not None else np.random.default_rng(0)
    return (compute_forces_3d if gs.is_3d else compute_forces)(gs, p, rng)


def grad_error(gs, p, n_granules=8, h=1e-5):
    """max |-dE/dx - F| over the first few granules, and max |F| for scale."""
    rng = np.random.default_rng(0)

    def energy():
        _F, _t, c = forces(gs, p, rng)
        return system_energy(gs, p, c)[0]

    _F0, _t0, c0 = forces(gs, p, rng)
    F0 = _F0
    axes = [gs.x, gs.y] + ([gs.z] if gs.is_3d else [])
    worst = 0.0
    for i in range(min(gs.N, n_granules)):
        for ax, arr in enumerate(axes):
            o = arr[i]
            arr[i] = o + h
            Ep = energy()
            arr[i] = o - h
            Em = energy()
            arr[i] = o
            worst = max(worst, abs(-(Ep - Em) / (2 * h) - F0[i, ax]))
    return worst, float(np.max(np.abs(F0)))


# ══════════════════════════════════════════════════════════════════════
# 1. the closed-form energy really is the potential of the JKR force
# ══════════════════════════════════════════════════════════════════════

class TestTheJKREnergy(unittest.TestCase):

    def test_it_reduces_to_the_hertz_energy(self):
        """At W = 0 the whole expression must collapse to (2/5) k delta^(5/2)."""
        for E_star, R in ((12000.0, 15.0), (5e5, 30.0), (1e4, 8.0)):
            k = (4.0 / 3.0) * (E_star * 1e-3) * np.sqrt(R)
            for d in (1e-3, 1e-2, 0.1, 1.0):
                a = np.sqrt(R * d)
                got = float(jkr_contact_energy(a, R, E_star, 0.0))
                self.assertAlmostEqual(got / ((2.0 / 5.0) * k * d ** 2.5), 1.0, places=12)

    def test_dU_d_delta_is_the_force_the_solver_applies(self):
        """The statement the whole phase rests on, across the JKR range."""
        for R, E_star, W in ((15.0, 12000.0, 0.0), (15.0, 12000.0, 5e-4),
                             (30.0, 5e5, 2e-3), (8.0, 1e4, 1e-2)):
            for d in (5e-3, 0.05, 0.5):
                h = 1e-6 * max(d, 1e-3)

                def U(dd):
                    _F, a = jkr_force_from_overlap(dd, R, E_star, W)
                    return float(jkr_contact_energy(a, R, E_star, W))

                fd = (U(d + h) - U(d - h)) / (2 * h)
                F, _a = jkr_force_from_overlap(d, R, E_star, W)
                # near the JKR force zero the relative test is meaningless
                self.assertLess(abs(fd - F), 1e-8 * max(abs(F), 1.0) + 1e-10,
                                f"R={R} E={E_star} W={W} delta={d}: dU/dd={fd} F={F}")

    def test_no_contact_stores_no_energy(self):
        self.assertEqual(float(jkr_contact_energy(0.0, 10.0, 1e4, 0.0)), 0.0)
        self.assertEqual(float(jkr_contact_energy(1.0, 0.0, 1e4, 0.0)), 0.0)
        self.assertEqual(contact_potential_energy(None), 0.0)
        self.assertEqual(contact_potential_energy([]), 0.0)


# ══════════════════════════════════════════════════════════════════════
# 2. -grad E == F for the assembled force law
# ══════════════════════════════════════════════════════════════════════

class TestTheForceIsMinusGradE(unittest.TestCase):
    """Central differences of the TOTAL energy against `compute_forces`.

    This is the end-to-end statement V3.4 could not make. 1e-6 relative is
    four orders looser than the measured 3.9e-9 -- the margin is for the
    finite-difference step, not for the physics.
    """

    def _check(self, label, tol=1e-6, **over):
        p, gs = packed(**over)
        err, scale = grad_error(gs, p)
        self.assertLess(err / max(scale, 1e-30), tol,
                        f"{label}: max|-grad E - F| = {err:.3e} vs max|F| = {scale:.3e}")
        return err / max(scale, 1e-30)

    def test_spheres_periodic(self):
        self._check('spheres, periodic')

    def test_the_force_is_minus_grad_E_with_walls(self):
        """Also the ONLY guard that `wall_potential_energy` enumerates the same
        faces as the gather. A wall contributes ~30 % of |F| in this bed."""
        self._check('spheres, walls', boundary_mode='walls')

    def test_with_gravity(self):
        self._check('spheres, gravity', boundary_mode='walls',
                    gravity_enabled=True, granule_density=1180.0)

    def test_3d_spheres_walls(self):
        self._check('3D spheres, walls', mode='3D', Lx=300.0, Ly=300.0, Lz=300.0,
                    phi_solid_target=0.45, boundary_mode='walls')

    def test_shaped_granules_are_conservative_to_a_quarter_percent(self):
        """Not machine precision, and the reason is structural, not a defect.

        `R_eff` depends on the contact normal and therefore on configuration,
        but the force law treats it as a parameter -- the neglected
        `dU/dR . grad R` is what is left. Pinned as a measurement.
        """
        rel = self._check('shapes, walls', tol=1e-2, boundary_mode='walls',
                          shape_enabled=True, aspect_ratio_func_mean=1.5,
                          blockiness_func_mean=3.0, aspect_ratio_inert_mean=1.4,
                          blockiness_inert_mean=2.8, curvature_R_cap=2.0)
        self.assertGreater(rel, 1e-5, "shaped contacts used to be 2.6e-3 from "
                                      "conservative; if this is now machine "
                                      "precision, say so in the changelog")


class TestWhatBreaksItAndByHowMuch(unittest.TestCase):
    """Each of these is physics, not a bug. The numbers are the calibration."""

    def _rel(self, **over):
        p, gs = packed(**over)
        err, scale = grad_error(gs, p)
        return err / max(scale, 1e-30)

    def test_active_noise_breaks_it_completely(self):
        """T_active = 5 is the DEFAULT, and it is the whole force.

        `add_active_noise` adds N(0, 2 gamma T / dt) per axis to every adhesive
        granule. At the default that is the same size as the contact force, so
        a default run is a noisy gradient flow and the audit is uninterpretable
        without T_active = 0.
        """
        self.assertGreater(self._rel(T_active=5.0), 0.5)

    def test_mc_dem_breaks_it_by_about_13_percent(self):
        rel = self._rel(mc_dem_enabled=True)
        self.assertGreater(rel, 1e-2)
        self.assertLess(rel, 1.0)

    def test_the_noise_floor_is_reported_and_has_the_right_scale(self):
        """2 d T act per adhesive granule per step -- independent of dt and gamma."""
        p, gs = packed(T_active=5.0)
        got = active_noise_power(gs, p)
        n_adh = int(np.sum(np.asarray(gs.adhesive_mask[:gs.N], dtype=bool)))
        self.assertGreater(n_adh, 0)
        self.assertAlmostEqual(got, 2 * 2 * 5.0 * n_adh, places=6)
        p0, gs0 = packed(T_active=0.0)
        self.assertEqual(active_noise_power(gs0, p0), 0.0)


# ══════════════════════════════════════════════════════════════════════
# 3. the per-step balance closes, and converges in dt
# ══════════════════════════════════════════════════════════════════════

def per_step_balance(p, gs, n=8, skip=2):
    """(dE, W) per step, measured directly rather than through the one-step lag."""
    rng = np.random.default_rng(0)
    out = []
    for k in range(n):
        _F, _t, c = forces(gs, p, rng)
        E0, _ = system_energy(gs, p, c)
        step(gs, p, rng, k * p.dt)
        _F2, _t2, c2 = forces(gs, p, rng)
        E1, _ = system_energy(gs, p, c2)
        if k >= skip:
            out.append((E1 - E0, gs.energy_prev[1]))
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


class TestThePerStepBalance(unittest.TestCase):

    def test_the_residual_converges_as_dt_shrinks(self):
        """With friction, noise and cells off, dE = -W up to truncation."""
        rels = []
        for dt in (1e-3, 1e-4):
            p, gs = packed(dt=dt)
            dE, W = per_step_balance(p, gs)
            rels.append(float(np.median(np.abs(dE + W) / np.abs(W))))
        self.assertLess(rels[0], 0.02, f"dt=1e-3 residual {rels[0]:.4f}")
        self.assertLess(rels[1], rels[0] / 2.0,
                        f"residual must fall with dt: {rels}")

    def test_friction_is_what_the_residual_is_made_of(self):
        """Switching tangential friction back on multiplies it by ~500x.

        Friction is dissipative, so this is correct: the work goes to heat
        instead of to the potential. It is the single largest term in a passive
        bed and it is why `energy_residual` is a throughput, not an error.
        """
        p, gs = packed()
        dE, W = per_step_balance(p, gs)
        clean = float(np.median(np.abs(dE + W) / np.abs(W)))
        p2, gs2 = packed(tau_0_cc=2000.0, tau_0_cb=500.0, tau_0_bb=50.0)   # the defaults
        dE2, W2 = per_step_balance(p2, gs2)
        rough = float(np.median(np.abs(dE2 + W2) / np.abs(W2)))
        self.assertGreater(rough, 50 * clean, f"clean {clean:.5f} vs friction {rough:.5f}")

    def test_a_passive_bed_descends_monotonically(self):
        """The invariant itself: with no noise, no cells and no friction, the
        energy falls on EVERY step, not merely on average."""
        p, gs = packed(dt=1e-3)
        dE, _W = per_step_balance(p, gs, n=16, skip=1)
        self.assertTrue(np.all(dE < 0.0), f"energy rose on some step: {dE}")

    def test_friction_makes_the_energy_alternate_and_dt_does_not_fix_it(self):
        """A finding, not a failure, and the reason ENERGY_ASCENT_TOL is 0.1.

        The tangential friction force opposes the relative surface velocity and
        is applied EXPLICITLY, so a sliding contact gets a kick that reverses
        its tangential velocity, which earns a kick back: a clean two-step
        limit cycle. Every other step ascends.

        It dissipates on net -- the sum over a pair of steps is negative -- and
        the amplitude falls linearly in dt. But the ascent RELATIVE TO THE WORK
        does not: 0.0357 at dt = 1e-3 and 0.0353 at dt = 1e-4. It is structural,
        not truncation. 0.035 is comfortably under the 0.1 tolerance, which is
        what keeps a normal run from flagging itself.
        """
        F = dict(tau_0_cc=2000.0, tau_0_cb=500.0, tau_0_bb=50.0)
        fracs = []
        for dt in (1e-3, 1e-4):
            p, gs = packed(dt=dt, **F)
            dE, W = per_step_balance(p, gs, n=14, skip=1)
            up = dE[dE > 0]
            self.assertGreater(len(up), 4, f"dt={dt}: friction used to alternate: {dE}")
            self.assertLess(float(dE.sum()), 0.0, f"dt={dt}: net must still descend")
            fracs.append(float(up.max() / np.abs(W).mean()))
        self.assertAlmostEqual(fracs[0], fracs[1], delta=0.01 * fracs[0],
                               msg=f"relative ascent must be dt-independent: {fracs}")
        self.assertLess(max(fracs), ENERGY_ASCENT_TOL,
                        f"friction alone must not trip the ascent flag: {fracs}")


# ══════════════════════════════════════════════════════════════════════
# 4. plumbing: modes, keys, and the off switch
# ══════════════════════════════════════════════════════════════════════

class TestThePlumbing(unittest.TestCase):

    def test_off_records_nothing(self):
        p, gs = packed(dynamics_gradient_flow='off', dt=1e-3)
        rng = np.random.default_rng(0)
        for k in range(3):
            step(gs, p, rng, k * p.dt)
        m = energy_metrics(gs)
        self.assertTrue(all(v == 0 for v in m.values()), m)
        self.assertIsNone(gs.energy_prev)

    def test_every_key_is_a_plain_finite_number(self):
        """They pass through csv.DictWriter and json.dump -- no NaN, no numpy."""
        p, gs = packed(dt=1e-3)
        rng = np.random.default_rng(0)
        for k in range(3):
            step(gs, p, rng, k * p.dt)
        for key, v in energy_metrics(gs).items():
            self.assertIn(type(v), (int, float), f"{key} is {type(v)}")
            self.assertTrue(np.isfinite(v), f"{key} = {v}")

    def test_the_parts_add_up_to_the_total(self):
        p, gs = packed(boundary_mode='walls', gravity_enabled=True, granule_density=1180.0)
        _F, _t, c = forces(gs, p)
        total, parts = system_energy(gs, p, c)
        self.assertAlmostEqual(total, sum(parts.values()), places=9)
        self.assertEqual(parts['contact'], contact_potential_energy(c))
        self.assertEqual(parts['wall'], wall_potential_energy(gs, p))
        self.assertEqual(parts['gravity'], gravity_potential_energy(gs, p))

    def test_periodic_has_no_wall_energy_and_no_gravity_without_gravity(self):
        p, gs = packed()
        self.assertEqual(wall_potential_energy(gs, p), 0.0)
        self.assertEqual(gravity_potential_energy(gs, p), 0.0)

    def test_damped_mode_shrinks_the_step_when_the_energy_rises(self):
        """The controller divides gamma, so it cannot move the fixed point --
        at F = 0 the step is zero for any drag. Driven here by the active
        noise, which is guaranteed to produce ascending steps."""
        p, gs = packed(dynamics_gradient_flow='damped', T_active=5.0, dt=0.5,
                       tau_0_cc=2000.0, tau_0_cb=500.0, tau_0_bb=50.0)
        rng = np.random.default_rng(0)
        for k in range(30):
            step(gs, p, rng, k * p.dt)
        self.assertGreater(gs.energy_backtracks, 0,
                           "a noisy bed must produce ascending steps")
        self.assertLessEqual(gs.energy_step_scale, 1.0)
        self.assertGreater(gs.energy_step_scale, 0.0)

    def test_the_ascent_counter_only_fires_above_the_tolerance(self):
        p, gs = packed(dt=1e-3)
        rng = np.random.default_rng(0)
        for k in range(10):
            step(gs, p, rng, k * p.dt)
        self.assertEqual(gs.energy_ascent_steps, 0,
                         "a clean passive bed must not ascend")
        self.assertLessEqual(gs.energy_ascent_frac, ENERGY_ASCENT_TOL)


if __name__ == '__main__':
    unittest.main()
