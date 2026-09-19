"""
Compiled post-step overlap resolution (V3.0 Phase 6g).
=====================================================

``resolve_overlaps`` is the reference's ``_resolve_overlaps`` with the pair
list from the configured neighbour backend and a compiled correction pass.
Per particle, the i-side corrections are summed in pair order and then the
j-side ones — the order ``np.subtract.at`` / ``np.add.at`` used — so with the
tree backend the positions match the reference to the bit; with the cell
list only the pair order (hence rounding) differs.
"""

import numpy as np

from gels.kernels import njit, prange
from gels.engine import se2d_lambda_grad, se3d_lambda_grad, shape_dynamics_on
from gels.kernels.neighbors import csr_from_pairs, half_pairs


@njit(parallel=True, cache=True, nogil=True)
def resolve_overlaps_k(pos, r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                       periodic, Lx, Ly, Lz, max_overlap_frac, mobile,
                       shape_mode, aa, bb, cc, n1, n2, theta, quat, eta):
    M = pair_i.shape[0]
    N = r_bound.shape[0]
    deep = np.zeros(M, dtype=np.bool_)
    ccx = np.zeros(M)          # unit normal components (V3.1)
    ccy = np.zeros(M)
    ccz = np.zeros(M)
    cci = np.zeros(M)          # correction magnitude for the i side
    ccj = np.zeros(M)
    for k in prange(M):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2] if dim == 3 else 0.0
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if dim == 3:
                dz = dz - Lz * np.rint(dz / Lz)
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        if dist < 1e-12:
            dist = 1e-12
        if shape_mode == 0:
            overlap = r_bound[i] + r_bound[j] - dist
            r_min = min(r_bound[i], r_bound[j])
        else:
            # V3.2: the directional radius, not the bounding sphere. At a true
            # phi 0.68 with AR 1.8 / n 3.5 the BOUNDING-SPHERE fraction exceeds 1,
            # so every contact reads as a deep overlap and this projection pulls
            # the packing apart -- measured 7.25 % bed expansion in 6 h, no cells.
            inv = 1.0 / dist
            ux = dx * inv
            uy = dy * inv
            uz = dz * inv if dim == 3 else 0.0
            if dim == 2:
                ci = np.cos(theta[i]); si = np.sin(theta[i])
                cj = np.cos(theta[j]); sj = np.sin(theta[j])
                lam_i, _a1, _a2 = se2d_lambda_grad(ux * ci + uy * si,
                                                   -ux * si + uy * ci, aa[i], bb[i], n1[i])
                lam_j, _b1, _b2 = se2d_lambda_grad(-ux * cj - uy * sj,
                                                   ux * sj - uy * cj, aa[j], bb[j], n1[j])
            else:
                w = quat[i, 0]; qx = quat[i, 1]; qy = quat[i, 2]; qz = quat[i, 3]
                tx = 2.0 * (qy * uz - qz * uy)
                ty = 2.0 * (qz * ux - qx * uz)
                tz = 2.0 * (qx * uy - qy * ux)
                lam_i, _a1, _a2, _a3 = se3d_lambda_grad(
                    ux - w * tx + (qy * tz - qz * ty),
                    uy - w * ty + (qz * tx - qx * tz),
                    uz - w * tz + (qx * ty - qy * tx),
                    aa[i], bb[i], cc[i], n1[i], n2[i])
                vx = -ux; vy = -uy; vz = -uz
                w2 = quat[j, 0]; q2x = quat[j, 1]; q2y = quat[j, 2]; q2z = quat[j, 3]
                sx = 2.0 * (q2y * vz - q2z * vy)
                sy = 2.0 * (q2z * vx - q2x * vz)
                sz = 2.0 * (q2x * vy - q2y * vx)
                lam_j, _b1, _b2, _b3 = se3d_lambda_grad(
                    vx - w2 * sx + (q2y * sz - q2z * sy),
                    vy - w2 * sy + (q2z * sx - q2x * sz),
                    vz - w2 * sz + (q2x * sy - q2y * sx),
                    aa[j], bb[j], cc[j], n1[j], n2[j])
            overlap = eta * (lam_i + lam_j) - dist
            r_min = min(lam_i, lam_j)
        threshold = max_overlap_frac * r_min
        if overlap > threshold:
            deep[k] = True
            excess = overlap - threshold
            inv_d = 1.0 / dist
            # V3.1: an immobile granule never moves, so its mobile partner
            # takes the whole correction; all-mobile is the V2.7 half split.
            if mobile[i] and mobile[j]:
                ci = 0.5 * excess
                cj = ci
            elif mobile[j]:
                ci = 0.0
                cj = excess
            elif mobile[i]:
                ci = excess
                cj = 0.0
            else:
                ci = 0.0
                cj = 0.0
            cci[k] = ci
            ccj[k] = cj
            ccx[k] = dx * inv_d
            ccy[k] = dy * inv_d
            if dim == 3:
                ccz[k] = dz * inv_d
    n_deep = 0
    for k in range(M):
        if deep[k]:
            n_deep += 1
    if n_deep == 0:
        return 0
    for i in prange(N):
        cx = 0.0
        cy = 0.0
        cz = 0.0
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 0 and deep[k]:
                cx -= cci[k] * ccx[k]
                cy -= cci[k] * ccy[k]
                cz -= cci[k] * ccz[k]
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if nbr_side[m] == 1 and deep[k]:
                cx += ccj[k] * ccx[k]
                cy += ccj[k] * ccy[k]
                cz += ccj[k] * ccz[k]
        pos[i, 0] += cx
        pos[i, 1] += cy
        if dim == 3:
            pos[i, 2] += cz
    return n_deep


def resolve_overlaps(gs, p):
    """Drop-in for ``reference._resolve_overlaps`` on the kernel path."""
    N = gs.N
    if N < 2 or p.max_overlap_frac <= 0:
        return
    dim = 3 if gs.is_3d else 2
    periodic = (p.boundary_mode == 'periodic')
    backend = getattr(p, 'perf_neighbor_backend', 'cells')
    if getattr(p, 'perf_cells_backend', 'kernels') != 'kernels':
        backend = 'ckdtree'          # exact-path runs keep the reference pair order
    pos = np.ascontiguousarray(gs.pos[:N, :dim])
    cutoff = 2.0 * float(np.max(gs.r_bound[:N]))
    pair_i, pair_j = half_pairs(pos, cutoff, periodic, (p.Lx, p.Ly, p.Lz), backend)
    if pair_i.shape[0] == 0:
        return
    off, nbr_pair, nbr_side = csr_from_pairs(pair_i, pair_j, N)
    fixed = getattr(gs, 'fixed', None)
    if fixed is None:
        mobile = np.ones(N, dtype=np.bool_)
    else:
        mobile = np.ascontiguousarray(~np.asarray(fixed[:N], dtype=bool))
    _shape_mode = 1 if shape_dynamics_on(p, gs) else 0
    _c = gs.c if (dim == 3 and gs.c is not None) else np.zeros(N)
    _n2 = gs.n2 if (dim == 3 and gs.n2 is not None) else np.full(N, 2.0)
    _q = gs.quat if (dim == 3 and gs.quat is not None) else np.zeros((N, 4))
    n_deep = resolve_overlaps_k(gs.pos, gs.r_bound, dim, pair_i, pair_j, off, nbr_pair, nbr_side,
                                periodic, float(p.Lx), float(p.Ly), float(p.Lz),
                                float(p.max_overlap_frac), mobile,
                                _shape_mode, gs.a, gs.b, _c, gs.n1, _n2, gs.theta, _q,
                                1.0 + float(getattr(p, 'packing_shape_margin', 0.0)))
    # V3.2 diagnostic: how often the geometric rail fires. A run where it fires
    # on many contacts is reporting the rail, not the contact law.
    gs.n_overlap_clipped = int(n_deep)
    gs.n_overlap_pairs = int(pair_i.shape[0])


__all__ = ['resolve_overlaps', 'resolve_overlaps_k']
