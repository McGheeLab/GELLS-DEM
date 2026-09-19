"""
Status-tuple 3D contact solvers for compiled callers (V3.0).
==========================================================

Copies of ``find_contact_spheres_3d``, ``find_contact_superellipsoids_3d``
and ``find_contact_wall_3d`` from ``gels.engine`` returning ``(hit, ...)``
tuples instead of ``None`` (numba cannot call Optional-returning functions
from compiled code). The sphere and wall routines were plain Python and are
re-expressed with scalar loops, which agree with the originals to rounding.

V3.4: ``se3d_contact_k`` is now the support-function MTD solver, wrapping the
single core ``gels.engine.se3d_mtd_core`` that the Python path also calls, and
it is allocation-free. ``se3d_contact_cn_k`` is the retired common-normal copy,
kept verbatim as evidence for the defect tests and called by nothing; see
``gels.engine.find_contact_superellipsoids_3d_cn``.
"""

import numpy as np

from gels.engine import (
    quat_rotate, quat_rotate_inv, quat_to_rotation_matrix, se3d_mtd_core,
    se3d_wall_core, se3d_wall_plane_core, superellipsoid_curvature_radii,
    superellipsoid_implicit, superellipsoid_normal, superellipsoid_point,
)
from gels.kernels import njit


@njit(cache=True)
def sph3d_contact_k(xi, yi, zi, ri, xj, yj, zj, rj):
    """Sphere–sphere contact: (hit, overlap, nx, ny, nz, cx, cy, cz, R_eff)."""
    dx = xj - xi
    dy = yj - yi
    dz = zj - zi
    d = np.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    overlap = ri + rj - d
    if overlap <= 0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    nx, ny, nz = dx/d, dy/d, dz/d
    R_eff = ri * rj / (ri + rj)
    cx = xi + (ri - overlap/2) * nx
    cy = yi + (ri - overlap/2) * ny
    cz = zi + (ri - overlap/2) * nz
    return True, overlap, nx, ny, nz, cx, cy, cz, R_eff


@njit(cache=True)
def se3d_contact_cn_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                   xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """Common-normal contact of two superellipsoids.

    Returns (hit, delta, nx, ny, nz, cx, cy, cz, R_eff).
    """
    dx_c = xj - xi
    dy_c = yj - yi
    dz_c = zj - zi
    d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
    if d_c < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ux = dx_c/d_c
    uy = dy_c/d_c
    uz = dz_c/d_c

    dir_body_i = quat_rotate_inv(qi, np.array([ux, uy, uz]))
    dir_body_j = quat_rotate_inv(qj, np.array([-ux, -uy, -uz]))

    eta_i = np.arctan2(dir_body_i[2],
                       np.sqrt(dir_body_i[0]**2 + dir_body_i[1]**2) + 1e-30)
    omega_i = np.arctan2(dir_body_i[1], dir_body_i[0])
    eta_j = np.arctan2(dir_body_j[2],
                       np.sqrt(dir_body_j[0]**2 + dir_body_j[1]**2) + 1e-30)
    omega_j = np.arctan2(dir_body_j[1], dir_body_j[0])

    _eta_lo = -np.pi/2 + 0.01
    _eta_hi = np.pi/2 - 0.01

    for _ in range(15):
        pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)

        pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
        pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

        dp = pj_world - pi_world
        dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
        if dp_mag < 1e-12:
            break
        target = dp / dp_mag

        ni_body = superellipsoid_normal(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        nj_body = superellipsoid_normal(eta_j, omega_j, aj, bj, cj, n1j, n2j)
        ni_world = quat_rotate(qi, ni_body)
        nj_world = quat_rotate(qj, nj_body)

        err_i = np.cross(ni_world, target)
        err_j = np.cross(nj_world, -target)

        err_i_mag = np.sqrt(err_i[0]**2 + err_i[1]**2 + err_i[2]**2)
        err_j_mag = np.sqrt(err_j[0]**2 + err_j[1]**2 + err_j[2]**2)
        if err_i_mag < 1e-7 and err_j_mag < 1e-7:
            break

        eta_i -= 0.3 * (err_i[2] * np.cos(omega_i) - err_i[1] * np.sin(omega_i))
        omega_i -= 0.3 * err_i[0]
        eta_j -= 0.3 * (err_j[2] * np.cos(omega_j) - err_j[1] * np.sin(omega_j))
        omega_j -= 0.3 * err_j[0]

        eta_i = min(max(eta_i, _eta_lo), _eta_hi)
        eta_j = min(max(eta_j, _eta_lo), _eta_hi)

    pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
    pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

    dp = pj_world - pi_world
    dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

    pi_in_j_body = quat_rotate_inv(qj, pi_world - np.array([xj, yj, zj]))
    val_j = superellipsoid_implicit(pi_in_j_body[0], pi_in_j_body[1],
                                    pi_in_j_body[2], aj, bj, cj, n1j, n2j)
    pj_in_i_body = quat_rotate_inv(qi, pj_world - np.array([xi, yi, zi]))
    val_i = superellipsoid_implicit(pj_in_i_body[0], pj_in_i_body[1],
                                    pj_in_i_body[2], ai, bi, ci, n1i, n2i)

    if val_j > 1.0 and val_i > 1.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    delta = dp_mag
    if dp_mag > 1e-12:
        nx, ny, nz = dp[0]/dp_mag, dp[1]/dp_mag, dp[2]/dp_mag
    else:
        nx, ny, nz = ux, uy, uz

    cx = 0.5 * (pi_world[0] + pj_world[0])
    cy = 0.5 * (pi_world[1] + pj_world[1])
    cz_pt = 0.5 * (pi_world[2] + pj_world[2])

    R1i, R2i = superellipsoid_curvature_radii(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    R1j, R2j = superellipsoid_curvature_radii(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    R_eff_i = np.sqrt(R1i * R2i)
    R_eff_j = np.sqrt(R1j * R2j)
    R_eff = R_eff_i * R_eff_j / (R_eff_i + R_eff_j) if (R_eff_i + R_eff_j) > 0 else 1.0

    return True, delta, nx, ny, nz, cx, cy, cz_pt, R_eff


@njit(cache=True)
def se3d_contact_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                   xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """Contact of two superellipsoids by support-function MTD (V3.4).

    Returns ``(hit, delta, nx, ny, nz, cx, cy, cz, R_eff)``.

    A thin wrapper, not a second implementation: there is exactly one MTD solver
    (``gels.engine.se3d_mtd_core``) and both twins call it, so they cannot drift
    the way ``_capacity`` did in V3.3. All this does is drop the convergence
    residual, which only the tests consume.
    """
    hit, delta, nx, ny, nz, cx, cy, cz, R_eff, _res = se3d_mtd_core(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
        xj, yj, zj, aj, bj, cj, n1j, n2j, qj)
    return hit, delta, nx, ny, nz, cx, cy, cz, R_eff


@njit(cache=True)
def wall3d_plane_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound, px, py, pz, nx, ny, nz):
    """Superellipsoid vs an arbitrary plane: (hit, penetration, R_local) (V3.4).

    One support evaluation, wrapping ``gels.engine.se3d_wall_plane_core``, in
    place of the 20 x 20 (eta, omega) surface grid this used to sample -- and in
    place of ``R_local = 0.5 * ri_bound``, which had no geometric content at all
    (for a sphere against a flat wall it is a factor of 2 low, and `F ~ sqrt(R)`).
    ``ri_bound`` is retained in the signature only because the call sites have it.
    """
    hit, pen, R_local, _cx, _cy, _cz = se3d_wall_plane_core(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi, px, py, pz, nx, ny, nz)
    return hit, pen, R_local


@njit(cache=True)
def wall3d_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound, wall_pos, wall_axis, wall_sign):
    """Superellipsoid-wall contact: (hit, penetration, R_local) (V3.4)."""
    hit, pen, R_local, _cx, _cy, _cz = se3d_wall_core(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi, wall_pos, wall_axis, wall_sign)
    return hit, pen, R_local


@njit(cache=True)
def wall3d_point_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, wall_pos, wall_axis, wall_sign):
    """As ``wall3d_k`` but also the contact point: (hit, pen, R_local, cx, cy, cz)."""
    return se3d_wall_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                          wall_pos, wall_axis, wall_sign)


@njit(cache=True)
def wall3d_plane_point_k(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, px, py, pz, nx, ny, nz):
    """As ``wall3d_plane_k`` but also the contact point."""
    return se3d_wall_plane_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                                px, py, pz, nx, ny, nz)


__all__ = ['sph3d_contact_k', 'se3d_contact_k', 'se3d_contact_cn_k',
           'wall3d_k', 'wall3d_plane_k', 'wall3d_point_k', 'wall3d_plane_point_k']
