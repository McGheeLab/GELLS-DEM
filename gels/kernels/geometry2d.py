"""
Status-tuple 2D superellipse contact solvers for compiled callers (V3.0).
=======================================================================

Compiled callers cannot use the Optional-returning wrappers in ``gels.engine``
(numba cannot call an Optional-returning ``@njit`` function from another
compiled function), so the contact solvers appear here returning
``(hit, ...)`` tuples instead.

V3.4: ``se2d_contact_k`` is now the support-function MTD solver, wrapping the
single core ``gels.engine.se2d_mtd_core`` that the Python path also calls --
so the two paths cannot run different physics. ``se2d_contact_cn_k`` is the
retired common-normal copy, kept verbatim as evidence for the defect tests and
called by nothing; see ``gels.engine.find_contact_superellipses_cn``.
"""

import numpy as np

from gels.engine import (
    _body_to_world, se2d_mtd_core, superellipse_curvature_radius,
    superellipse_implicit, superellipse_normal_vec, superellipse_point,
)
from gels.kernels import njit


@njit(cache=True)
def se2d_contact_cn_k(xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj):
    """RETIRED (V3.4): the common-normal contact of two superellipses.

    Evidence only -- nothing calls this. See
    ``gels.engine.find_contact_superellipses_cn``.

    Returns (hit, delta, nx, ny, cx, cy, R_loc_i, R_loc_j); all zeros with
    hit=False when the bodies do not overlap.
    """
    dx_c = xj - xi
    dy_c = yj - yi
    d_c = np.sqrt(dx_c**2 + dy_c**2)
    if d_c < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    ux, uy = dx_c / d_c, dy_c / d_c

    bx_i, by_i = (np.cos(thetai) * ux + np.sin(thetai) * uy,
                  -np.sin(thetai) * ux + np.cos(thetai) * uy)
    bx_j, by_j = (np.cos(thetaj) * (-ux) + np.sin(thetaj) * (-uy),
                  -np.sin(thetaj) * (-ux) + np.cos(thetaj) * (-uy))

    t_i = np.arctan2(by_i, bx_i)
    t_j = np.arctan2(by_j, bx_j)

    for iteration in range(12):
        px_i, py_i = superellipse_point(t_i, ai, bi, ni)
        px_j, py_j = superellipse_point(t_j, aj, bj, nj)

        wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
        wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

        dpx = wx_j - wx_i
        dpy = wy_j - wy_i
        dp_mag = np.sqrt(dpx**2 + dpy**2)
        if dp_mag < 1e-12:
            break

        nix, niy = superellipse_normal_vec(t_i, ai, bi, ni)
        cos_ti, sin_ti = np.cos(thetai), np.sin(thetai)
        nix_w = cos_ti * nix - sin_ti * niy
        niy_w = sin_ti * nix + cos_ti * niy

        njx, njy = superellipse_normal_vec(t_j, aj, bj, nj)
        cos_tj, sin_tj = np.cos(thetaj), np.sin(thetaj)
        njx_w = cos_tj * njx - sin_tj * njy
        njy_w = sin_tj * njx + cos_tj * njy

        target_nx = dpx / dp_mag
        target_ny = dpy / dp_mag

        err_i = nix_w * target_ny - niy_w * target_nx
        err_j = njx_w * (-target_ny) - njy_w * (-target_nx)

        if abs(err_i) < 1e-8 and abs(err_j) < 1e-8:
            break

        t_i -= 0.5 * err_i
        t_j -= 0.5 * err_j

    px_i, py_i = superellipse_point(t_i, ai, bi, ni)
    px_j, py_j = superellipse_point(t_j, aj, bj, nj)
    wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
    wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

    cpx = wx_j - wx_i
    cpy = wy_j - wy_i
    cp_mag = np.sqrt(cpx**2 + cpy**2)

    val_j = superellipse_implicit(wx_i, wy_i, xj, yj, aj, bj, nj, thetaj)
    val_i = superellipse_implicit(wx_j, wy_j, xi, yi, ai, bi, ni, thetai)

    if val_j > 1.0 and val_i > 1.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    delta = cp_mag
    if cp_mag < 1e-12:
        delta = max(ai, bi) * (1.0 - val_j**(1.0 / nj)) if val_j < 1.0 else 0.01

    if cp_mag > 1e-12:
        nx, ny = cpx / cp_mag, cpy / cp_mag
    else:
        nx, ny = ux, uy

    contact_x = 0.5 * (wx_i + wx_j)
    contact_y = 0.5 * (wy_i + wy_j)

    R_loc_i = superellipse_curvature_radius(t_i, ai, bi, ni)
    R_loc_j = superellipse_curvature_radius(t_j, aj, bj, nj)

    return True, delta, nx, ny, contact_x, contact_y, R_loc_i, R_loc_j


@njit(cache=True)
def se2d_contact_k(xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj):
    """Contact of two superellipses by support-function MTD (V3.4).

    Returns ``(hit, delta, nx, ny, cx, cy, R_loc_i, R_loc_j)``; all zeros with
    ``hit=False`` when the bodies do not overlap.

    A thin wrapper over the one solver in ``gels.engine``, not a second copy --
    all it does is drop the convergence residual, which only the tests consume.
    Both twins going through the same core is what stops them drifting the way
    ``_capacity`` did in V3.3.
    """
    hit, delta, nx, ny, cx, cy, R_loc_i, R_loc_j, _res = se2d_mtd_core(
        xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj)
    return hit, delta, nx, ny, cx, cy, R_loc_i, R_loc_j


@njit(cache=True)
def se2d_wall_k(xi, yi, ai, bi, ni, thetai, wall_pos, wall_axis, wall_sign):
    """Superellipse–wall contact: (hit, penetration, R_local)."""
    n_sample = 64
    t_vals = np.arange(n_sample) * (2.0 * np.pi / n_sample)
    e = 2.0 / ni
    ct, st = np.cos(t_vals), np.sin(t_vals)
    bx = ai * np.sign(ct) * np.abs(ct)**e
    by = bi * np.sign(st) * np.abs(st)**e
    cos_th, sin_th = np.cos(thetai), np.sin(thetai)
    wx = xi + cos_th * bx - sin_th * by
    wy = yi + sin_th * bx + cos_th * by

    if wall_axis == 0:
        coords = wx
    else:
        coords = wy

    if wall_sign > 0:
        idx = np.argmin(coords)
        pen = wall_pos - coords[idx]
    else:
        idx = np.argmax(coords)
        pen = coords[idx] - wall_pos

    if pen <= 0:
        return False, 0.0, 0.0

    t_contact = t_vals[idx]
    R_local = superellipse_curvature_radius(t_contact, ai, bi, ni)
    return True, pen, R_local


__all__ = ['se2d_contact_k', 'se2d_contact_cn_k', 'se2d_wall_k']
