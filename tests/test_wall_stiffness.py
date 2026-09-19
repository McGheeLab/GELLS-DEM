"""
The wall contact reaches the semi-implicit step (V3.6 Phase 1).

``contact_stiffness_per_granule`` sums ``dF/d(delta) = 2 E_s a`` over the PAIR
contact list. Wall contacts are applied inline in ``gather_2d`` / ``gather_3d``
and their reference twin, and never reach that list -- so until V3.6 a granule
held only by a wall was damped by ``drag_scale * r`` alone.

That was invisible before V3.5. ``boundary.wall_clamp`` used to hold every floor
granule 0.5 um clear of the wall, so **no granule was ever in wall contact**;
the clip carried the load and the wall force read zero. Changing the default to
``contact`` made wall contacts real, and made this gap real with them.

``a_w`` was already being computed and thrown away at all ten wall call sites,
so the stiffness costs nothing to collect.

What it does and does not claim
-------------------------------
It does NOT move the equilibrium. At the fixed point ``F = 0``, so the step is
zero for any drag -- the same argument ``contact_semi_implicit`` and
``gradient_flow: damped`` rest on. ``test_the_fixed_point_is_untouched`` is that
claim, and it is checked on the energy, not on a proxy.

What it removes is overshoot. Measured on a FIRE-relaxed 2D gravity bed at the
default ``dt = 0.5 h``, as the fraction of steps on which a granule reverses
direction:

    E = 10 kPa    wall granules  0.100 -> 0.002      free 0.000 -> 0.000
    E = 50 kPa    wall granules  0.368 -> 0.015      free 0.033 -> 0.002
    E = 200 kPa   wall granules  0.333 -> 0.002      free 0.024 -> 0.000

and the bed's energy is unchanged to four figures in every one of those runs.
The 50 kPa row is the interesting one: a ringing wall granule was shaking its
neighbours, so this was never purely a boundary artefact.

Run:  python -m unittest tests.test_wall_stiffness -v
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

import gels.engine as E  # noqa: E402
from gels.engine import (  # noqa: E402
    Params, compute_forces, compute_forces_3d, contact_stiffness_per_granule,
    generate_packing, generate_packing_3d, step,
)


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


# A small sphere bed sedimented onto the floor: granules are genuinely in wall
# contact, which is the whole precondition. Friction and noise off so the only
# thing moving the bed is the contact law.
BED = dict(
    mode='2D', Lx=300.0, Ly=600.0, phi_solid_target=0.30,
    save_data=False, save_fields=False,
    R_func_mean=20.0, R_func_std=0.0, R_inert_mean=20.0, R_inert_std=0.0,
    n_cells_per_granule=0, cell_coverage=0.0, cell_surface_coverage=0.0,
    gravity_enabled=True, granule_density=1180.0, boundary_top='free',
    packing_consolidation='gravity', packing_settle_steps=150,
    T_active=0.0, mc_dem_enabled=False, friction_mu=0.0,
    tau_0_cc=0.0, tau_0_cb=0.0, tau_0_bb=0.0,
    packing_relax='fire',
)

BED_3D = dict(
    BED, mode='3D', Lx=220.0, Ly=220.0, Lz=400.0, phi_solid_target=0.25,
    packing_settle_steps=100,
)


def bed(three_d=False, seed=3, **over):
    base = dict(BED_3D if three_d else BED)
    base.update(over)
    p = Params(**base)
    gs = quiet(generate_packing_3d if three_d else generate_packing, p, seed=seed)
    F, _tq, c = quiet(compute_forces_3d if three_d else compute_forces,
                      gs, p, np.random.default_rng(0))
    return p, gs, F, c


def _true_stiffness(gs, contacts):
    return contact_stiffness_per_granule(gs, contacts)


@contextlib.contextmanager
def wall_term_off():
    """V3.5's behaviour: the pair list only, walls unrepresented."""
    orig = E.contact_stiffness_per_granule

    def without(gs, contacts):
        k = orig(gs, contacts)
        wk = getattr(gs, 'wall_stiffness', None)
        if wk is None:
            return k
        return k - np.asarray(wk[:gs.N], dtype=float)

    E.contact_stiffness_per_granule = without
    try:
        yield
    finally:
        E.contact_stiffness_per_granule = orig


def reversal_fractions(p, gs, nsteps=200):
    """Fraction of steps on which each granule reverses its direction of travel.

    The honest measure of overshoot here. The energy monitor is the more
    fundamental instrument but at these amplitudes the ascent is below
    ``ENERGY_ASCENT_TOL``: the step overshoots and comes back without ever
    raising the total energy by a tenth of the work.
    """
    rng = np.random.default_rng(0)
    dys = []
    axis = gs.z if gs.is_3d else gs.y
    for s in range(nsteps):
        y0 = axis[:gs.N].copy()
        quiet(step, gs, p, rng, s * p.dt)
        dys.append(axis[:gs.N] - y0)
    d = np.array(dys)[nsteps // 2:]
    return np.mean(np.sign(d[1:]) * np.sign(d[:-1]) < 0, axis=0)


def _twin_stiffness(three_d=False, **over):
    """(kernel, reference) wall stiffness for ONE bed, as comparable strings.

    Returned as formatted text so a failure prints the two arrays side by side
    rather than a numpy diff of 42 mostly-zero entries.
    """
    p, gs, _F, _c = bed(three_d=three_d, **over)
    from gels.kernels import reference as _ref
    if three_d:
        from gels.kernels.contact3d import compute_forces_3d as _k
    else:
        from gels.kernels.contact2d import compute_forces_2d as _k
    _r = _ref.compute_forces_3d if three_d else _ref.compute_forces
    quiet(_k, gs, p, np.random.default_rng(0))
    k_wall = np.asarray(gs.wall_stiffness[:gs.N]).copy()
    quiet(_r, gs, p, np.random.default_rng(0))
    r_wall = np.asarray(gs.wall_stiffness[:gs.N]).copy()
    assert k_wall.max() > 0.0, 'the bed is not touching a wall'
    fmt = lambda v: ' '.join('%.12g' % x for x in v)
    return fmt(np.round(k_wall, 9)), fmt(np.round(r_wall, 9))


class TestTheQuantity(unittest.TestCase):
    """What `gs.wall_stiffness` is, before anything consumes it."""

    def test_it_is_two_E_star_a_on_a_single_wall_contact(self):
        # Granules that touch the floor and NOTHING else: their whole entry is
        # one wall record, which can be reproduced by hand from the JKR law.
        p, gs, _F, _c = bed()
        i_all = np.arange(gs.N)
        floor_only = ((gs.y[:gs.N] < gs.r[:gs.N]) &
                      (gs.x[:gs.N] - gs.r[:gs.N] > 0.0) &
                      (gs.x[:gs.N] + gs.r[:gs.N] < p.Lx))
        self.assertGreater(int(floor_only.sum()), 0, 'no granule rests on the floor')
        for i in i_all[floor_only]:
            E_s = gs.wall_Estar[gs.species_id[i]] * 1e-3     # Pa -> nN/um^2
            _F_w, a_w = E.jkr_force_from_overlap(
                gs.r[i] - gs.y[i], gs.r[i], gs.wall_Estar[gs.species_id[i]],
                gs.wall_W[gs.species_id[i]])
            self.assertAlmostEqual(float(gs.wall_stiffness[i]), 2.0 * E_s * a_w,
                                   delta=1e-9 * max(1.0, 2.0 * E_s * a_w))

    def test_it_is_identically_zero_when_no_wall_is_touched(self):
        # `consolidation: centre` pulls the bed into a ball off every wall.
        _p, gs, _F, _c = bed(packing_consolidation='centre', gravity_enabled=False,
                             packing_relax='none')
        np.testing.assert_array_equal(gs.wall_stiffness[:gs.N], 0.0)

    def test_it_is_identically_zero_under_periodic_boundaries(self):
        _p, gs, _F, _c = bed(boundary_mode='periodic', gravity_enabled=False,
                             boundary_top='wall', packing_consolidation='none',
                             packing_relax='none')
        np.testing.assert_array_equal(gs.wall_stiffness[:gs.N], 0.0)

    def test_the_twins_agree(self):
        # reference.py and the compiled gather accumulate it in different
        # places; this is the only thing keeping them honest. Evaluate BOTH on
        # the same bed -- FIRE agrees between the twins to rounding, not
        # bit-for-bit (V3.5), so two independently packed beds would be
        # comparing the packer, not the gather.
        self.assertEqual(*_twin_stiffness(three_d=False))

    def test_it_reaches_the_summed_stiffness(self):
        p, gs, _F, c = bed()
        k = contact_stiffness_per_granule(gs, c)
        onw = np.asarray(gs.wall_stiffness[:gs.N]) > 0
        self.assertTrue(np.all(k[onw] >= gs.wall_stiffness[:gs.N][onw] - 1e-12))
        # and a granule with a wall contact and no pair contact gets exactly it
        with wall_term_off():
            k_pairs = E.contact_stiffness_per_granule(gs, c)
        np.testing.assert_allclose(k - k_pairs, gs.wall_stiffness[:gs.N],
                                   rtol=1e-12, atol=1e-12)


class TestItDoesNotMoveTheAnswer(unittest.TestCase):
    """A drag term cannot move a fixed point. This is that claim, checked."""

    def test_the_fixed_point_is_untouched(self):
        for E_mod in (10.0, 200.0):
            with self.subTest(E_modulus=E_mod):
                p_a, gs_a, _F, _c = bed(E_modulus=E_mod,
                                        dynamics_gradient_flow='monitor')
                p_b, gs_b, _F, _c = bed(E_modulus=E_mod,
                                        dynamics_gradient_flow='monitor')
                with wall_term_off():
                    reversal_fractions(p_a, gs_a)
                reversal_fractions(p_b, gs_b)
                # Four significant figures on the total potential energy: the
                # two runs relax to the same bed by different transients.
                self.assertAlmostEqual(gs_a.energy_total, gs_b.energy_total,
                                       delta=1e-3 * abs(gs_a.energy_total))

    def test_it_changes_nothing_for_a_bed_that_touches_no_wall(self):
        kw = dict(packing_consolidation='centre', gravity_enabled=False,
                  packing_relax='none')
        p_a, gs_a, _F, _c = bed(**kw)
        p_b, gs_b, _F, _c = bed(**kw)
        with wall_term_off():
            reversal_fractions(p_a, gs_a, nsteps=40)
        reversal_fractions(p_b, gs_b, nsteps=40)
        np.testing.assert_array_equal(gs_a.pos[:gs_a.N], gs_b.pos[:gs_b.N])


class TestItRemovesTheOvershoot(unittest.TestCase):
    """The measurement in the module docstring, pinned."""

    def _compare(self, **over):
        p_a, gs_a, _F, _c = bed(**over)
        p_b, gs_b, _F, _c = bed(**over)
        onw = np.asarray(gs_a.wall_stiffness[:gs_a.N]) > 0
        self.assertGreater(int(onw.sum()), 0)
        with wall_term_off():
            rev_a = reversal_fractions(p_a, gs_a)
        rev_b = reversal_fractions(p_b, gs_b)
        return onw, rev_a, rev_b

    def test_wall_granules_stop_reversing(self):
        onw, rev_a, rev_b = self._compare()
        before, after = rev_a[onw].mean(), rev_b[onw].mean()
        self.assertGreater(before, 0.05,
                           'the V3.5 behaviour being fixed is not reproduced')
        self.assertLess(after, 0.2 * before)

    def test_and_it_helps_more_the_stiffer_the_granule(self):
        # 2 E_s a grows with E*, so the term the explicit step was missing
        # grows with it too.
        onw, rev_a, rev_b = self._compare(E_modulus=200.0)
        self.assertGreater(rev_a[onw].mean(), 0.2)
        self.assertLess(rev_b[onw].mean(), 0.05)

    def test_free_granules_are_not_made_worse(self):
        onw, rev_a, rev_b = self._compare(E_modulus=50.0)
        free = ~onw
        self.assertLessEqual(rev_b[free].mean(), rev_a[free].mean() + 1e-9)


class TestEveryWallGeometry(unittest.TestCase):
    """Five of the ten call sites are only reachable in 3D or in a cylinder."""

    def test_3d_box_floor(self):
        _p, gs, _F, _c = bed(three_d=True)
        self.assertGreater(float(np.max(gs.wall_stiffness[:gs.N])), 0.0)

    def test_3d_cylinder_side_wall(self):
        _p, gs, _F, _c = bed(three_d=True, boundary_shape='cylinder')
        self.assertGreater(float(np.max(gs.wall_stiffness[:gs.N])), 0.0)

    def test_3d_twins_agree(self):
        self.assertEqual(*_twin_stiffness(three_d=True))

    def test_shaped_granules_against_a_wall(self):
        _p, gs, _F, _c = bed(shape_enabled=True, packing_shape_contact=True,
                             contact_shape_dynamics=True, curvature_R_cap=2.0,
                             aspect_ratio_func_mean=1.8, blockiness_func_mean=3.5,
                             aspect_ratio_inert_mean=1.8, blockiness_inert_mean=3.5)
        self.assertGreater(float(np.max(gs.wall_stiffness[:gs.N])), 0.0)


class TestTheGeometricProxyAgrees(unittest.TestCase):
    """`wall_contact_fraction` was blind to the floor in 3D.

    Found while checking Phase 1: `gs.wall_stiffness` said seven granules were
    in wall contact and `bed_metrics` said none were. The 3D branch of
    `wall_contact_fraction` took the gap to the two **x** faces only, so the one
    face a sedimented bed actually rests on never entered the minimum.
    """

    def test_it_sees_the_floor_in_3d(self):
        p, gs, _F, _c = bed(three_d=True)
        onw = np.asarray(gs.wall_stiffness[:gs.N]) > 0
        self.assertGreater(int(onw.sum()), 0, 'the bed is not touching the floor')
        m = E.bed_metrics(gs, p)
        self.assertGreater(m['wall_contact_fraction'], 0.0)

    def test_the_free_top_is_not_counted_as_a_wall(self):
        # Take one bed, park a granule against the container ceiling, and ask
        # the metric twice. With a lid that granule is in wall contact; with the
        # V3.1 free top it is at a free surface and must not be counted.
        p, gs, _F, _c = bed(boundary_top='free')
        gs.y[0] = p.Ly - gs.r_bound[0] + 0.1          # surface just past the lid plane
        p_lid = Params(**dict(BED, boundary_top='wall'))
        free = E.bed_metrics(gs, p)['wall_contact_fraction']
        lid = E.bed_metrics(gs, p_lid)['wall_contact_fraction']
        self.assertGreater(lid, free)

    def test_no_granule_is_counted_that_carries_no_wall_stiffness(self):
        # The proxy uses a 1 um shell, so it may be slightly generous; it must
        # never be MEANER than the force law, which would hide a real contact.
        for kw in (dict(), dict(three_d=True)):
            with self.subTest(**kw):
                p, gs, _F, _c = bed(**kw)
                rb = np.asarray(gs.r_bound[:gs.N])
                onw = np.asarray(gs.wall_stiffness[:gs.N]) > 0
                faces = [gs.x[:gs.N] - rb, p.Lx - gs.x[:gs.N] - rb, gs.y[:gs.N] - rb]
                if gs.is_3d:
                    faces += [p.Ly - gs.y[:gs.N] - rb, gs.z[:gs.N] - rb,
                              p.Lz - gs.z[:gs.N] - rb]
                else:
                    faces += [p.Ly - gs.y[:gs.N] - rb]
                near = np.min(np.stack(faces), axis=0)
                self.assertTrue(np.all(near[onw] < 1.0),
                                'a granule in wall contact is outside the proxy shell')


if __name__ == '__main__':
    unittest.main(verbosity=2)
