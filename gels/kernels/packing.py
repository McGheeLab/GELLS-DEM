"""
Packing kernels (V3.0 Phase 6f): RSA neighbour grid and compiled settle.
=======================================================================

Both stages can keep the packing **bit-identical** to the reference path
(``perf_neighbor_backend = 'ckdtree'``, or the exact cell mode), so a run's
initial condition then does not depend on the compute backend. With the
default linked-cell list the settle is an order of magnitude faster at large
N (no tree per substep) and still deterministic and thread-independent, but
its pair order — hence the rounding of the jammed configuration — differs
from the reference.

* ``RSAChecker`` answers the random-sequential-addition overlap question
  ("does this candidate centre clash with any placed granule?") either with
  the reference's Python loop or with a linked-cell grid searched over an
  adaptive reach. The inequality and its operands are the same, only the set
  of granules examined shrinks from all N to the neighbouring cells, so the
  decision — and therefore the random draws consumed — are unchanged.
* ``settle_packing_2d/3d`` run the Lubachevsky–Stillinger inflate-and-relax
  of the reference with a compiled substep. Every particle sums its i-side
  contributions in pair order followed by its j-side ones — the order
  ``np.add.at`` used — so with the tree's pair order forces, moves and final
  positions match the reference to the bit; with the sorted cell-list order
  they agree to rounding.
"""

import math

import numpy as np
from scipy.spatial import cKDTree

from gels.engine import (GravityConsolidation, bed_solid_volume, bed_surface, boundary_geometry,
                         consolidation_mode, consolidation_weights, granule_bound_radius, minimum_image_disp,
    se2d_lambda_grad, se3d_lambda_grad,
                         minimum_image_disp_3d)
from gels.kernels import HAS_NUMBA, njit
from gels.kernels.neighbors import csr_from_pairs, half_pairs, neighbor_backend


# ──────────────────────────────────────────────────────────────────────
# RSA overlap check
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _rsa_ok_grid(cx, cy, cz, rb_val, alpha, gap, dim, periodic, Lx, Ly, Lz,
                 pos, rb, head, nxt, nc, sx, sy, sz, reach):
    ix = int(cx // sx)
    iy = int(cy // sy)
    iz = int(cz // sz) if dim == 3 else 0
    if ix < 0:
        ix = 0
    if ix >= nc[0]:
        ix = nc[0] - 1
    if iy < 0:
        iy = 0
    if iy >= nc[1]:
        iy = nc[1] - 1
    if iz < 0:
        iz = 0
    if iz >= nc[2]:
        iz = nc[2] - 1
    rz = reach if dim == 3 else 0
    for ox in range(-reach, reach + 1):
        jx = ix + ox
        if periodic:
            jx = jx % nc[0]
        elif jx < 0 or jx >= nc[0]:
            continue
        for oy in range(-reach, reach + 1):
            jy = iy + oy
            if periodic:
                jy = jy % nc[1]
            elif jy < 0 or jy >= nc[1]:
                continue
            for oz in range(-rz, rz + 1):
                jz = iz + oz
                if periodic:
                    jz = jz % nc[2]
                elif jz < 0 or jz >= nc[2]:
                    continue
                j = head[jx, jy, jz]
                while j >= 0:
                    dx = cx - pos[j, 0]
                    dy = cy - pos[j, 1]
                    if periodic:
                        dx = dx - Lx * np.rint(dx / Lx)
                        dy = dy - Ly * np.rint(dy / Ly)
                    thr = (rb_val + rb[j]) * alpha + gap
                    if dim == 3:
                        dz = cz - pos[j, 2]
                        if periodic:
                            dz = dz - Lz * np.rint(dz / Lz)
                        if dx*dx + dy*dy + dz*dz < thr*thr:
                            return False
                    else:
                        if dx*dx + dy*dy < thr*thr:
                            return False
                    j = nxt[j]
    return True


def reference_radius(species):
    """Largest plausible bounding radius of the species mix (mean + 3σ), for the grid cell size."""
    best = 1.0
    for sp in species:
        mean = float(sp.get('radius_mean', 40.0) or 40.0)
        std = float(sp.get('radius_std', 0.0) or 0.0)
        best = max(best, mean + 3.0 * std)
    return best


class RSAChecker:
    """Overlap test for random sequential addition; exact loop or compiled grid (same decision)."""

    def __init__(self, dim, p, alpha, gap, n_max, r_ref, use_kernel):
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.gap = float(gap)
        self.periodic = (p.boundary_mode == 'periodic')
        self.Lx, self.Ly, self.Lz = float(p.Lx), float(p.Ly), float(p.Lz)
        self.shape_enabled = bool(p.shape_enabled)
        n_max = max(int(n_max), 1)
        self.n = 0
        self.pos = np.zeros((n_max, 3))
        self.rb = np.zeros(n_max)
        self.rb_max = 0.0
        self.use_kernel = bool(use_kernel and HAS_NUMBA)
        if self.use_kernel:
            s_target = max(2.0 * float(r_ref) * self.alpha + self.gap, 1e-6)
            Ls = (self.Lx, self.Ly, self.Lz if dim == 3 else self.Lz)
            nc = [max(1, min(512, int(L / s_target))) for L in Ls[:dim]]
            if dim == 2:
                nc.append(1)
            self.nc = np.array(nc, dtype=np.int64)
            self.sx = self.Lx / nc[0]
            self.sy = self.Ly / nc[1]
            self.sz = self.Lz / nc[2] if dim == 3 else self.Lz
            self.head = np.full((nc[0], nc[1], nc[2]), -1, dtype=np.int64)
            self.nxt = np.full(n_max, -1, dtype=np.int64)

    def ok(self, cx, cy, cz, rb_val):
        if self.n == 0:
            return True
        if self.use_kernel:
            reach_len = (rb_val + self.rb_max) * self.alpha + self.gap
            reach = int(math.ceil(reach_len / min(self.sx, self.sy, self.sz if self.dim == 3 else self.sx)))
            return bool(_rsa_ok_grid(float(cx), float(cy), float(cz), float(rb_val), self.alpha, self.gap,
                                     self.dim, self.periodic, self.Lx, self.Ly, self.Lz,
                                     self.pos, self.rb, self.head, self.nxt, self.nc,
                                     self.sx, self.sy, self.sz, reach))
        # reference loop, statement for statement
        n = self.n
        pos = self.pos
        rb = self.rb
        for j in range(n):
            dx = cx - pos[j, 0]
            dy = cy - pos[j, 1]
            if self.dim == 3:
                dz = cz - pos[j, 2]
                if self.periodic:
                    dx, dy, dz = minimum_image_disp_3d(dx, dy, dz, self.Lx, self.Ly, self.Lz)
                if dx*dx + dy*dy + dz*dz < ((rb_val + rb[j]) * self.alpha + self.gap)**2:
                    return False
            else:
                if self.periodic:
                    dx, dy = minimum_image_disp(dx, dy, self.Lx, self.Ly)
                if dx*dx + dy*dy < ((rb_val + rb[j]) * self.alpha + self.gap)**2:
                    return False
        return True

    def add(self, cx, cy, cz, rb_val):
        j = self.n
        if j >= self.pos.shape[0]:
            grow = max(16, self.pos.shape[0])
            self.pos = np.vstack([self.pos, np.zeros((grow, 3))])
            self.rb = np.concatenate([self.rb, np.zeros(grow)])
            if self.use_kernel:
                self.nxt = np.concatenate([self.nxt, np.full(grow, -1, dtype=np.int64)])
        self.pos[j, 0] = cx
        self.pos[j, 1] = cy
        self.pos[j, 2] = cz
        self.rb[j] = rb_val
        self.rb_max = max(self.rb_max, float(rb_val))
        self.n = j + 1
        if self.use_kernel:
            ix = min(max(int(cx // self.sx), 0), int(self.nc[0]) - 1)
            iy = min(max(int(cy // self.sy), 0), int(self.nc[1]) - 1)
            iz = min(max(int(cz // self.sz), 0), int(self.nc[2]) - 1) if self.dim == 3 else 0
            self.nxt[j] = self.head[ix, iy, iz]
            self.head[ix, iy, iz] = j


# ──────────────────────────────────────────────────────────────────────
# settle substep
# ──────────────────────────────────────────────────────────────────────

# Rotation caps in SETTLE units, alongside dt_settle and k_rep. Not Params
# fields: they are solver constants, and every Params field costs a FLAT_MAP
# entry, a template line and a validate rule. Above ~0.3 rad the first-order
# quaternion update loses accuracy quickly.
THETA_CAP_2D = 0.15
THETA_CAP_3D = 0.12


@njit(cache=True)
def settle_sub_shape_k(pos, rb, dim, pair_i, pair_j, off, nbr_pair, nbr_side, periodic, Lx, Ly, Lz,
                       k_rep, dt_settle, v_cap,
                       g_mag, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile,
                       aa, bb, cc, n1, n2, theta, quat, eta):
    """One shape-aware relaxation substep: directional overlap, force AND torque (V3.2).

    The legacy substep works on bounding spheres, so a blocky granule can never
    rotate to nest and the packing jams far below what its shape allows. Here the
    pair gap is

        g = eta * (lambda_i(u) + lambda_j(-u)) - d

    with ``lambda`` the exact distance from the centre to the surface along the
    centre line, and the generalised forces are -dE/dq of the SAME scalar with
    ``E = (2/5) k_rep g^{5/2}`` (so dE/dg = k_rep g^{3/2}, the legacy magnitude).
    That makes the substep gradient descent on a potential: it cannot pump energy
    and cannot limit-cycle.

    Two details that are easy to get wrong and are verified against finite
    differences in tests/test_shape_packing.py:

    * ``lambda`` is homogeneous of degree -1, so ``grad(lambda) . u = -lambda``,
      NOT zero. The gap depends on u only through its direction, so the RADIAL
      part of the gradient must be projected out. Keeping it gives a force ~2.5x
      too large that is not the gradient of anything.
    * A purely radial force produces NO torque at all -- the arm is parallel to
      the force -- so without the tangential tilt from that gradient nothing
      rotates and the density gain disappears entirely.

    Returns (max overlap, largest move, largest rotation x radius of gyration).
    """
    N = rb.shape[0]
    M = pair_i.shape[0]
    hit = np.zeros(M, dtype=np.bool_)
    fpx = np.zeros(M)
    fpy = np.zeros(M)
    fpz = np.zeros(M)
    tix = np.zeros(M)
    tiy = np.zeros(M)
    tiz = np.zeros(M)
    tjx = np.zeros(M)
    tjy = np.zeros(M)
    tjz = np.zeros(M)
    max_ov = 0.0

    for k in range(M):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = 0.0
        if dim == 3:
            dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if dim == 3:
                dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
        # bounding-sphere pre-reject: cheap, and rb is a true bound under this flag
        if d >= rb[i] + rb[j]:
            continue
        inv_d = 1.0 / d
        ux = dx * inv_d
        uy = dy * inv_d
        uz = dz * inv_d if dim == 3 else 0.0

        if dim == 2:
            ci = np.cos(theta[i]); si = np.sin(theta[i])
            cj = np.cos(theta[j]); sj = np.sin(theta[j])
            bix = ux * ci + uy * si
            biy = -ux * si + uy * ci
            bjx = -ux * cj - uy * sj
            bjy = ux * sj - uy * cj
            lam_i, gbix, gbiy = se2d_lambda_grad(bix, biy, aa[i], bb[i], n1[i])
            lam_j, gbjx, gbjy = se2d_lambda_grad(bjx, bjy, aa[j], bb[j], n1[j])
            # body -> world
            Gix = gbix * ci - gbiy * si
            Giy = gbix * si + gbiy * ci
            Gjx = gbjx * cj - gbjy * sj
            Gjy = gbjx * sj + gbjy * cj
            Giz = 0.0; Gjz = 0.0
        else:
            biz = 0.0; bjz = 0.0
            w = quat[i, 0]; qx = quat[i, 1]; qy = quat[i, 2]; qz = quat[i, 3]
            # b = R^T u  (conjugate rotation)
            tx = 2.0 * (qy * uz - qz * uy)
            ty = 2.0 * (qz * ux - qx * uz)
            tz = 2.0 * (qx * uy - qy * ux)
            bix = ux - w * tx + (qy * tz - qz * ty)
            biy = uy - w * ty + (qz * tx - qx * tz)
            biz = uz - w * tz + (qx * ty - qy * tx)
            vx = -ux; vy = -uy; vz = -uz
            w2 = quat[j, 0]; q2x = quat[j, 1]; q2y = quat[j, 2]; q2z = quat[j, 3]
            sx = 2.0 * (q2y * vz - q2z * vy)
            sy = 2.0 * (q2z * vx - q2x * vz)
            sz = 2.0 * (q2x * vy - q2y * vx)
            bjx = vx - w2 * sx + (q2y * sz - q2z * sy)
            bjy = vy - w2 * sy + (q2z * sx - q2x * sz)
            bjz = vz - w2 * sz + (q2x * sy - q2y * sx)
            lam_i, gbix, gbiy, gbiz = se3d_lambda_grad(bix, biy, biz, aa[i], bb[i], cc[i], n1[i], n2[i])
            lam_j, gbjx, gbjy, gbjz = se3d_lambda_grad(bjx, bjy, bjz, aa[j], bb[j], cc[j], n1[j], n2[j])
            # body -> world (forward rotation by q)
            tx = 2.0 * (qy * gbiz - qz * gbiy)
            ty = 2.0 * (qz * gbix - qx * gbiz)
            tz = 2.0 * (qx * gbiy - qy * gbix)
            Gix = gbix + w * tx + (qy * tz - qz * ty)
            Giy = gbiy + w * ty + (qz * tx - qx * tz)
            Giz = gbiz + w * tz + (qx * ty - qy * tx)
            sx = 2.0 * (q2y * gbjz - q2z * gbjy)
            sy = 2.0 * (q2z * gbjx - q2x * gbjz)
            sz = 2.0 * (q2x * gbjy - q2y * gbjx)
            Gjx = gbjx + w2 * sx + (q2y * sz - q2z * sy)
            Gjy = gbjy + w2 * sy + (q2z * sx - q2x * sz)
            Gjz = gbjz + w2 * sz + (q2x * sy - q2y * sx)

        g = eta * (lam_i + lam_j) - d
        if g <= 0.0:
            continue
        hit[k] = True
        if g > max_ov:
            max_ov = g
        kappa = k_rep * g ** 1.5

        Dx = eta * (Gix - Gjx)
        Dy = eta * (Giy - Gjy)
        Dz = eta * (Giz - Gjz)
        dot = Dx * ux + Dy * uy + Dz * uz
        Px = Dx - dot * ux
        Py = Dy - dot * uy
        Pz = Dz - dot * uz
        fpx[k] = kappa * (ux - Px * inv_d)
        fpy[k] = kappa * (uy - Py * inv_d)
        if dim == 3:
            fpz[k] = kappa * (uz - Pz * inv_d)

        ke = kappa * eta
        if dim == 2:
            tiz[k] = ke * (ux * Giy - uy * Gix)
            tjz[k] = -ke * (ux * Gjy - uy * Gjx)
        else:
            tix[k] = ke * (uy * Giz - uz * Giy)
            tiy[k] = ke * (uz * Gix - ux * Giz)
            tiz[k] = ke * (ux * Giy - uy * Gix)
            tjx[k] = -ke * (uy * Gjz - uz * Gjy)
            tjy[k] = -ke * (uz * Gjx - ux * Gjz)
            tjz[k] = -ke * (ux * Gjy - uy * Gjx)

    max_disp = 0.0
    max_rot = 0.0
    for i in range(N):
        fx = 0.0; fy = 0.0; fz = 0.0
        tx = 0.0; ty = 0.0; tz = 0.0
        if g_mag > 0.0:
            gwi = g_mag * gw[i]
            fx += gwi * gx
            fy += gwi * gy
            if dim == 3:
                fz += gwi * gz
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 0 and hit[k]:
                fx -= fpx[k]; fy -= fpy[k]; fz -= fpz[k]
                tx += tix[k]; ty += tiy[k]; tz += tiz[k]
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 1 and hit[k]:
                fx += fpx[k]; fy += fpy[k]; fz += fpz[k]
                tx += tjx[k]; ty += tjy[k]; tz += tjz[k]
        if not mobile[i]:
            continue

        fmag = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-12
        sc = min(dt_settle, v_cap / fmag)
        mv = fmag * sc
        if mv > max_disp:
            max_disp = mv

        # orientation: overdamped, with the second moment of the shape standing
        # in for inertia, exactly as the dynamics does
        if dim == 2:
            I_s = 0.5 * (aa[i] * aa[i] + bb[i] * bb[i])
            r_gyr = np.sqrt(I_s)
            if I_s > 1e-20:
                om = tz / I_s
                om_abs = abs(om)
                sc_r = min(dt_settle, THETA_CAP_2D / om_abs) if om_abs > 1e-20 else dt_settle
                dth = om * sc_r
                theta[i] += dth
                if abs(dth) * r_gyr > max_rot:
                    max_rot = abs(dth) * r_gyr
        else:
            I_s = (aa[i]*aa[i] + bb[i]*bb[i] + cc[i]*cc[i]) / 3.0
            r_gyr = np.sqrt(I_s)
            if I_s > 1e-20:
                ox = tx / I_s; oy = ty / I_s; oz = tz / I_s
                om_abs = np.sqrt(ox*ox + oy*oy + oz*oz)
                sc_r = min(dt_settle, THETA_CAP_3D / om_abs) if om_abs > 1e-20 else dt_settle
                hx = 0.5 * ox * sc_r; hy = 0.5 * oy * sc_r; hz = 0.5 * oz * sc_r
                w0 = quat[i, 0]; x0 = quat[i, 1]; y0 = quat[i, 2]; z0 = quat[i, 3]
                nw = w0 + (-hx * x0 - hy * y0 - hz * z0)
                nx_ = x0 + (hx * w0 + hy * z0 - hz * y0)
                ny_ = y0 + (hy * w0 + hz * x0 - hx * z0)
                nz_ = z0 + (hz * w0 + hx * y0 - hy * x0)
                nrm = np.sqrt(nw*nw + nx_*nx_ + ny_*ny_ + nz_*nz_)
                if nrm > 1e-20:
                    quat[i, 0] = nw / nrm
                    quat[i, 1] = nx_ / nrm
                    quat[i, 2] = ny_ / nrm
                    quat[i, 3] = nz_ / nrm
                if om_abs * sc_r * r_gyr > max_rot:
                    max_rot = om_abs * sc_r * r_gyr

        x = pos[i, 0] + fx * sc
        y = pos[i, 1] + fy * sc
        if periodic:
            x = x % Lx
            y = y % Ly
        else:
            if shape_code == 1:
                dxc = x - cxc
                dyc = y - cyc
                rho = np.sqrt(dxc * dxc + dyc * dyc)
                rmax = R_cyl - rb[i]
                if rho > rmax:
                    s_r = rmax / rho
                    x = cxc + dxc * s_r
                    y = cyc + dyc * s_r
            else:
                x = min(max(x, rb[i]), Lx - rb[i])
                y = min(max(y, rb[i]), Ly - rb[i])
        pos[i, 0] = x
        pos[i, 1] = y
        if dim == 3:
            z = pos[i, 2] + fz * sc
            if periodic:
                z = z % Lz
            else:
                z = min(max(z, rb[i]), Lz - rb[i])
            pos[i, 2] = z
    return max_ov, max_disp, max_rot


@njit(cache=True)
def settle_sub_k(pos, rb, dim, pair_i, pair_j, off, nbr_pair, nbr_side, periodic, Lx, Ly, Lz,
                 attract, cx_dom, cy_dom, cz_dom, k_rep, dt_settle, v_cap,
                 g_mag, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile):
    """One overlap-relaxation substep of the reference settle (bit-identical accumulation).

    V3.1: optional body force ``g_mag * gw[i] * (gx, gy, gz)`` (gravity
    consolidation, accumulated where the legacy centripetal term is), radial
    clamp for a cylindrical container (``shape_code == 1``) and immobile
    granules (``mobile[i] == False`` are never moved). Returns
    (max overlap, largest move of a mobile granule).
    """
    N = rb.shape[0]
    M = pair_i.shape[0]
    hit = np.zeros(M, dtype=np.bool_)
    fn = np.zeros(M)
    nx = np.zeros(M)
    ny = np.zeros(M)
    nz = np.zeros(M)
    max_ov = 0.0
    for k in range(M):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = 0.0
        if dim == 3:
            dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if dim == 3:
                dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
        ov = rb[i] + rb[j] - d
        if ov > 0:
            hit[k] = True
            if ov > max_ov:
                max_ov = ov
            inv_d = 1.0 / d
            nx[k] = dx * inv_d
            ny[k] = dy * inv_d
            nz[k] = dz * inv_d
            fn[k] = k_rep * ov ** 1.5
    max_disp = 0.0
    for i in range(N):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        if attract > 0.0:
            dxc = cx_dom - pos[i, 0]
            dyc = cy_dom - pos[i, 1]
            dzc = cz_dom - pos[i, 2] if dim == 3 else 0.0
            dc = np.sqrt(dxc*dxc + dyc*dyc + dzc*dzc) + 1e-12
            sc = attract * rb[i] / dc
            fx += sc * dxc
            fy += sc * dyc
            if dim == 3:
                fz += sc * dzc
        if g_mag > 0.0:
            gwi = g_mag * gw[i]
            fx += gwi * gx
            fy += gwi * gy
            if dim == 3:
                fz += gwi * gz
        # np.add.at order: every i-side contribution (pair order), then every j-side one
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 0 and hit[k]:
                fx += (-fn[k]) * nx[k]
                fy += (-fn[k]) * ny[k]
                if dim == 3:
                    fz += (-fn[k]) * nz[k]
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 1 and hit[k]:
                fx += fn[k] * nx[k]
                fy += fn[k] * ny[k]
                if dim == 3:
                    fz += fn[k] * nz[k]
        fmag = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-12
        sc = min(dt_settle, v_cap / fmag)
        if not mobile[i]:
            continue
        mv = fmag * sc
        if mv > max_disp:
            max_disp = mv
        x = pos[i, 0] + fx * sc
        y = pos[i, 1] + fy * sc
        if periodic:
            x = x % Lx
            y = y % Ly
        else:
            if shape_code == 1:
                dxc = x - cxc
                dyc = y - cyc
                rho = np.sqrt(dxc * dxc + dyc * dyc)
                rmax = R_cyl - rb[i]
                if rho > rmax:
                    s_r = rmax / rho
                    x = cxc + dxc * s_r
                    y = cyc + dyc * s_r
            else:
                x = min(max(x, rb[i]), Lx - rb[i])
                y = min(max(y, rb[i]), Ly - rb[i])
        pos[i, 0] = x
        pos[i, 1] = y
        if dim == 3:
            z = pos[i, 2] + fz * sc
            if periodic:
                z = z % Lz
            else:
                z = min(max(z, rb[i]), Lz - rb[i])
            pos[i, 2] = z
    return max_ov, max_disp


def _pairs_and_csr(gs, p, cutoff, dim):
    N = gs.N
    pos = np.ascontiguousarray(gs.pos[:N, :dim])
    periodic = (p.boundary_mode == 'periodic')
    pair_i, pair_j = half_pairs(pos, cutoff, periodic, (p.Lx, p.Ly, p.Lz), neighbor_backend(p))
    off, nbr_pair, nbr_side = csr_from_pairs(pair_i, pair_j, N)
    return pair_i, pair_j, off, nbr_pair, nbr_side


def _count_contacts(gs, p, dim, cutoff):
    N = gs.N
    pos = np.ascontiguousarray(gs.pos[:N, :dim])
    periodic = (p.boundary_mode == 'periodic')
    pair_i, pair_j = half_pairs(pos, cutoff, periodic, (p.Lx, p.Ly, p.Lz), neighbor_backend(p))
    if pair_i.shape[0] == 0:
        return 0
    d = pos[pair_j] - pos[pair_i]
    if periodic:
        L = np.array([p.Lx, p.Ly, p.Lz][:dim])
        d -= L * np.round(d / L)
    dist = np.sqrt(np.sum(d * d, axis=1))
    gaps = dist - gs.r_bound[pair_i] - gs.r_bound[pair_j]
    return int(np.sum(gaps < 1.0))


def _settle(gs, p, dim):
    _shape_rb = bool(getattr(p, 'packing_shape_contact', False))   # V3.2
    _eta = 1.0 + float(getattr(p, 'packing_shape_margin', 0.0))
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    _c_arr = gs.c if (dim == 3 and gs.c is not None) else np.zeros(N)
    _n2_arr = gs.n2 if (dim == 3 and gs.n2 is not None) else np.full(N, 2.0)
    _quat_arr = gs.quat if (dim == 3 and gs.quat is not None) else np.zeros((N, 4))
    Lx, Ly, Lz = float(p.Lx), float(p.Ly), float(p.Lz)
    n_inflate = p.packing_settle_steps
    n_relax = p.packing_relax_substeps
    dt_settle = 0.02
    k_rep = 5.0
    v_cap_frac = 0.3

    target_a = getattr(gs, '_target_a', gs.a.copy())
    target_b = getattr(gs, '_target_b', gs.b.copy())
    target_c = getattr(gs, '_target_c', gs.c.copy())
    target_r = getattr(gs, '_target_r', gs.r.copy())

    ratio = gs.r[:N] / np.maximum(target_r[:N], 1e-6)
    alpha_start = float(np.median(ratio))
    alpha_start = max(alpha_start, 0.1)
    mean_r_target = float(np.mean(target_r[:N]))
    v_cap = v_cap_frac * mean_r_target

    # V3.1 consolidation: centre (V2.7 pull toward the box centre) | gravity | none
    mode = consolidation_mode(p) if not periodic else 'none'
    use_centre = (not periodic) and mode == 'centre'
    use_gravity = (not periodic) and mode == 'gravity'
    geom = boundary_geometry(p, '3D' if dim == 3 else '2D')
    shape_code = int(geom.shape_code) if not periodic else 0
    R_cyl, cxc, cyc = float(geom.R_cyl), float(geom.cx), float(geom.cy)
    fixed = getattr(gs, 'fixed', None)
    if fixed is None:
        mobile = np.ones(N, dtype=np.bool_)
    else:
        mobile = np.ascontiguousarray(~np.asarray(fixed[:N], dtype=bool))
    any_fixed = not bool(np.all(mobile))
    if use_gravity:
        H_place = float(getattr(gs, '_H_place', Lz if dim == 3 else Ly))
        sched = GravityConsolidation(mean_r_target, H_place, k_rep)
        gw = np.ascontiguousarray(consolidation_weights(target_r[:N], mean_r_target, dim))
    else:
        sched = None
        gw = np.ones(N)
    if dim == 3:
        gx, gy, gz = 0.0, 0.0, -1.0
    else:
        gx, gy, gz = 0.0, -1.0, 0.0

    print(f"  Settling {dim}D packing ({n_inflate} inflate × {n_relax} relax, "
          f"α={alpha_start:.2f}→1.00) [compiled, {neighbor_backend(p)} pairs, consolidation {mode}]...")
    cx_dom, cy_dom, cz_dom = Lx / 2, Ly / 2, Lz / 2
    pos = gs.pos                 # (N, 3) C-contiguous; the kernel updates it in place

    max_overlap = 0.0
    max_disp = np.inf
    for step in range(n_inflate):
        t = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t)**2)
        alpha_v = np.where(mobile, alpha, 1.0) if any_fixed else alpha
        gs.a[:N] = target_a[:N] * alpha_v
        gs.b[:N] = target_b[:N] * alpha_v
        if dim == 3:
            gs.c[:N] = target_c[:N] * alpha_v
        gs.r[:N] = target_r[:N] * alpha_v
        gs.r_bound[:N] = granule_bound_radius(
            gs.a[:N], gs.b[:N], gs.c[:N] if dim == 3 else None,
            gs.n1[:N], gs.n2[:N] if dim == 3 else None, dim == 3, _shape_rb)
        max_rb = float(np.max(gs.r_bound[:N]))
        attract = max(0.3 * (1.0 - t), 0.02) if use_centre else 0.0
        for sub in range(n_relax):
            pair_i, pair_j, off, nbr_pair, nbr_side = _pairs_and_csr(gs, p, 2 * max_rb, dim)
            if _shape_rb:
                max_overlap, max_disp, _mr = settle_sub_shape_k(
                    pos, gs.r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                    periodic, Lx, Ly, Lz, k_rep, dt_settle, v_cap,
                    0.0, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile,
                    gs.a, gs.b, _c_arr, gs.n1, _n2_arr, gs.theta, _quat_arr, _eta)
            else:
                max_overlap, max_disp = settle_sub_k(pos, gs.r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                                                     periodic, Lx, Ly, Lz, attract, cx_dom, cy_dom, cz_dom,
                                                     k_rep, dt_settle, v_cap,
                                                     0.0, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile)
            if max_overlap < 0.01 * mean_r_target:
                break
        if step % 100 == 0 or step == n_inflate - 1:
            n_contacts = _count_contacts(gs, p, dim, 2 * max_rb + 1.0)
            Z = 2 * n_contacts / max(N, 1)
            print(f"    step {step}/{n_inflate}: α={alpha:.3f}, "
                  f"max_overlap={max_overlap:.1f} µm, Z={Z:.1f}")

    overlap_tol = 0.05 * mean_r_target
    max_post_relax = 3000 if use_gravity else 1000
    max_rb = float(np.max(gs.r_bound[:N]))
    extra = 0
    max_disp = np.inf
    for extra in range(max_post_relax):
        pair_i, pair_j, off, nbr_pair, nbr_side = _pairs_and_csr(gs, p, 2 * max_rb, dim)
        # the reference evaluates the overlap before deciding whether to move: same here,
        # by moving with a zero step when the tolerance is already met
        max_overlap = _max_overlap_only(pos, gs.r_bound, dim, pair_i, pair_j, periodic, Lx, Ly, Lz)
        if use_gravity:
            g_k = sched.g(extra, max_disp)
            if sched.converged(g_k, max_overlap, overlap_tol, max_disp):
                break
        else:
            g_k = 0.0
            if max_overlap < overlap_tol:
                break
        if _shape_rb:
            _, max_disp, _mr = settle_sub_shape_k(
                pos, gs.r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                periodic, Lx, Ly, Lz, k_rep, dt_settle, v_cap,
                g_k, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile,
                gs.a, gs.b, _c_arr, gs.n1, _n2_arr, gs.theta, _quat_arr, _eta)
            max_disp = max(max_disp, _mr)        # the settle must not stop while the bed re-orients
        else:
            _, max_disp = settle_sub_k(pos, gs.r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                                       periodic, Lx, Ly, Lz, 0.0, cx_dom, cy_dom, cz_dom, k_rep, dt_settle, v_cap,
                                       g_k, gx, gy, gz, gw, shape_code, R_cyl, cxc, cyc, mobile)
    else:
        if dim == 3 or use_gravity:
            print(f"    WARNING: post-relax hit {max_post_relax} steps, "
                  f"max_overlap={max_overlap:.1f} µm (tol={overlap_tol:.1f})")
    if dim == 3 and extra > 0 and max_overlap < overlap_tol:
        print(f"    post-relax: {extra} extra steps, "
              f"max_overlap={max_overlap:.1f} µm (tol={overlap_tol:.1f})")

    n_contacts = _count_contacts(gs, p, dim, 2 * float(np.max(gs.r_bound[:N])) + 1.0)
    Z = 2 * n_contacts / max(N, 1)
    if dim == 3:
        print(f"  done ({n_contacts} contacts, Z={Z:.1f}/granule)")
    else:
        print(f"  Settle done: {n_contacts} contacts, Z={Z:.1f} "
              f"({extra + 1 if max_overlap >= overlap_tol else extra} post-relax steps)")
    if use_gravity:
        bed = bed_surface(gs, p)
        V = bed_solid_volume(gs)
        env = bed['bed_envelope_volume']
        phi_bed = V / env if env > 0 else float('nan')
        print(f"  Sedimented bed: height {bed['bed_height_mean']:.0f} um "
              f"(p95 {bed['bed_height_p95']:.0f}), envelope solid fraction {phi_bed:.3f}, "
              f"{extra} post-relax steps")


@njit(cache=True)
def _max_overlap_only(pos, rb, dim, pair_i, pair_j, periodic, Lx, Ly, Lz):
    max_ov = 0.0
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = 0.0
        if dim == 3:
            dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if dim == 3:
                dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
        ov = rb[i] + rb[j] - d
        if ov > max_ov:
            max_ov = ov
    return max_ov


def settle_packing_2d(gs, p):
    """Compiled twin of ``reference._settle_packing_2d`` (bit-identical with the tree backend)."""
    _settle(gs, p, 2)


def settle_packing_3d(gs, p):
    """Compiled twin of ``reference._settle_packing_3d`` (bit-identical with the tree backend)."""
    _settle(gs, p, 3)


__all__ = ['RSAChecker', 'reference_radius', 'settle_packing_2d', 'settle_packing_3d', 'settle_sub_k']
