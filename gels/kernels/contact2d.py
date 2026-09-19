"""
Compiled 2D contact forces (V3.0 Phase 6a).
==========================================

Three passes replace the per-pair Python loop of ``reference.compute_forces``:

  pair_pass_2d   prange over half pairs → per-pair contact records
                 (geometry, JKR normal force, friction)
  mc_dem_pairs   per-particle volumetric strain → per-pair Hertz correction
  gather_2d      prange over particles → F, torque, clip planes, walls
                 (owner writes only: no atomics, thread-count independent)

Cell bridging stays in Python for now (``gels.kernels.bridging``) over the
same candidate pairs in the same order as the reference, so cell state and
the random stream are unchanged. Active noise is the reference's numpy
code. Forces agree with the reference to rounding (summation order
differs); contact records, clips and cell state match.
"""

import time

import numpy as np

from gels.engine import (apply_gravity, boundary_geometry, hertz_contact_force,
                         jkr_force_from_overlap, pair_bridge_weight)
from gels.kernels import njit, prange
from gels.kernels.bridging import bridging_pass
from gels.kernels.contacts import ContactSoA, add_active_noise, alloc_records, clips_to_lists
from gels.kernels.geometry2d import se2d_contact_k, se2d_wall_k, se2d_wall_point_k
from gels.kernels.neighbors import csr_from_pairs, half_pairs, neighbor_backend


@njit(parallel=True, cache=True, nogil=True)
def pair_pass_2d(pos, vel, r, a, b, n_shape, theta, sid, is_circle, periodic, Lx, Ly,
                 pair_i, pair_j, pair_W, pair_tau, pair_Estar, v_ref, mu, cell_adh, bridge_w, R_cap,
                 c_hit, c_overlap, c_nx, c_ny, c_cx, c_cy, c_Reff, c_Fn, c_a, c_A,
                 c_Ftx, c_Fty, c_d, c_dx, c_dy):
    M = pair_i.shape[0]
    for k in prange(M):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
        d = np.sqrt(dx*dx + dy*dy)
        c_d[k] = d
        c_dx[k] = dx
        c_dy[k] = dy
        c_overlap[k] = 0.0
        c_cx[k] = 0.0
        c_cy[k] = 0.0
        c_Reff[k] = 0.0
        c_Fn[k] = 0.0
        c_a[k] = 0.0
        c_A[k] = 0.0
        c_Ftx[k] = 0.0
        c_Fty[k] = 0.0
        if d < 1e-6:
            c_hit[k] = -1
            c_nx[k] = 0.0
            c_ny[k] = 0.0
            continue
        nx = dx / d
        ny = dy / d

        hit = 0
        overlap = 0.0
        R_eff = 0.0
        cx = 0.0
        cy = 0.0
        if is_circle:
            overlap = r[i] + r[j] - d
            if overlap > 0:
                R_eff = r[i] * r[j] / (r[i] + r[j])
                cx = pos[i, 0] + (r[i] - overlap/2) * nx
                cy = pos[i, 1] + (r[i] - overlap/2) * ny
                hit = 1
            else:
                overlap = 0.0
        else:
            xj_v = pos[i, 0] + dx
            yj_v = pos[i, 1] + dy
            ok, delta, snx, sny, scx, scy, R_loc_i, R_loc_j = se2d_contact_k(
                pos[i, 0], pos[i, 1], a[i], b[i], n_shape[i], theta[i],
                xj_v, yj_v, a[j], b[j], n_shape[j], theta[j])
            if ok:
                hit = 1
                overlap = delta
                nx = snx
                ny = sny
                cx = scx
                cy = scy
                R_eff = R_loc_i * R_loc_j / (R_loc_i + R_loc_j)
                if R_cap > 0.0:
                    # V3.2: the local curvature radius of a superellipse runs away
                    # at a flat face -- measured 3.5e15 um at n = 3.5 against 48 for
                    # a circle, and F ~ sqrt(R_eff), so the contact is ~1e6 times too
                    # stiff and the bed explodes. Cap it at a multiple of the granule.
                    lim = R_cap * min(r[i], r[j])
                    if R_eff > lim:
                        R_eff = lim
        c_hit[k] = hit
        c_nx[k] = nx
        c_ny[k] = ny
        c_overlap[k] = overlap
        c_Reff[k] = R_eff
        c_cx[k] = cx
        c_cy[k] = cy

        if hit == 1 and overlap > 0:
            si = sid[i]
            sj = sid[j]
            tau_0 = pair_tau[si, sj]
            W_adh = pair_W[si, sj]
            E_star_pair = pair_Estar[si, sj]
            F_normal, a_contact = jkr_force_from_overlap(overlap, R_eff, E_star_pair, W_adh)
            c_Fn[k] = F_normal
            c_a[k] = a_contact
            A_contact = np.pi * a_contact**2 if a_contact > 0 else 0.0
            c_A[k] = A_contact
            dvx = vel[j, 0] - vel[i, 0]
            dvy = vel[j, 1] - vel[i, 1]
            v_dot_n = dvx * nx + dvy * ny
            vtx = dvx - v_dot_n * nx
            vty = dvy - v_dot_n * ny
            vt_mag = np.sqrt(vtx*vtx + vty*vty)
            if vt_mag > 1e-12:
                F_t_cap = tau_0 * A_contact * 1e-3
                if mu > 0.0 and F_normal > 0.0:
                    F_t_cap += mu * F_normal
                if cell_adh > 0.0:
                    F_t_cap += cell_adh * bridge_w[k]   # V3.2: the cells' own grip
                F_fric = F_t_cap * np.tanh(vt_mag / v_ref)
                tx = vtx / vt_mag
                ty = vty / vt_mag
                c_Ftx[k] = F_fric * tx
                c_Fty[k] = F_fric * ty


@njit(cache=True)
def mc_dem_pairs(pair_i, pair_j, r, nu, kappa_max, c_hit, c_overlap, c_Reff, c_Estar, dF, c_kappa):
    """MC-DEM stiffening (Giannis 2021): per-pair extra normal force dF and κ_ij.

    eps_V accumulates in pair order exactly like the reference's contact
    loop, so κ matches bit for bit.
    """
    N = r.shape[0]
    M = pair_i.shape[0]
    eps_V = np.zeros(N)
    for k in range(M):
        if c_hit[k] == 1 and c_overlap[k] > 0:
            delta = c_overlap[k]
            eps_V[pair_i[k]] += delta / (2.0 * r[pair_i[k]])
            eps_V[pair_j[k]] += delta / (2.0 * r[pair_j[k]])
    kappa = np.empty(N)
    for i in range(N):
        c_mc = nu[i] / max(1.0 - 2.0 * nu[i], 0.01)
        kappa[i] = min(1.0 + c_mc * eps_V[i], kappa_max)
    for k in range(M):
        dF[k] = 0.0
        c_kappa[k] = 1.0
        if c_hit[k] == 1 and c_overlap[k] > 0:
            kappa_ij = 0.5 * (kappa[pair_i[k]] + kappa[pair_j[k]])
            if kappa_ij < 1.001:
                continue
            dF[k] = (kappa_ij - 1.0) * hertz_contact_force(c_Estar[k], c_Reff[k], c_overlap[k])
            c_kappa[k] = kappa_ij


@njit(parallel=True, cache=True, nogil=True)
def gather_2d(pos, r, a, b, n_shape, theta, sid, is_circle, periodic, Lx, Ly, top_free,
              off, nbr_pair, nbr_side,
              c_hit, c_overlap, c_nx, c_ny, c_cx, c_cy, c_Fn, c_a, c_Ftx, c_Fty, dF,
              wall_W, wall_Estar, F, torques, clip_n, clip_d, clip_cnt, clip_over,
              wall_torque):
    N = pos.shape[0]
    max_clips = clip_d.shape[1]
    for i in prange(N):
        fx = 0.0
        fy = 0.0
        tq = 0.0
        nclip = 0
        nover = 0
        # ── pair contacts, in pair order ──
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if c_hit[k] != 1 or c_overlap[k] <= 0.0:
                continue
            side = nbr_side[m]
            nx = c_nx[k]
            ny = c_ny[k]
            Fnx = c_Fn[k] * nx
            Fny = c_Fn[k] * ny
            Ftx = c_Ftx[k]
            Fty = c_Fty[k]
            if side == 0:
                fx -= Fnx
                fy -= Fny
                if not is_circle:
                    rcx = c_cx[k] - pos[i, 0]
                    rcy = c_cy[k] - pos[i, 1]
                    tq += rcx * (-Fny) - rcy * (-Fnx)
                fx += Ftx
                fy += Fty
                if not is_circle:
                    tq += rcx * Fty - rcy * Ftx
            else:
                fx += Fnx
                fy += Fny
                if not is_circle:
                    rcx = c_cx[k] - pos[i, 0]
                    rcy = c_cy[k] - pos[i, 1]
                    tq += rcx * Fny - rcy * Fnx
                fx -= Ftx
                fy -= Fty
                if not is_circle:
                    tq += rcx * (-Fty) - rcy * (-Ftx)
            if c_a[k] > 1e-6:
                if nclip < max_clips:
                    if side == 0:
                        clip_n[i, nclip, 0] = nx
                        clip_n[i, nclip, 1] = ny
                    else:
                        clip_n[i, nclip, 0] = -nx
                        clip_n[i, nclip, 1] = -ny
                    clip_d[i, nclip] = np.sqrt(max(r[i]**2 - c_a[k]**2, 0.01))
                    nclip += 1
                else:
                    nover += 1
        # ── walls (rigid, bare surface) ──
        if not periodic:
            si = sid[i]
            W_wall = wall_W[si]
            E_star_gw = wall_Estar[si]
            xi = pos[i, 0]
            yi = pos[i, 1]
            if is_circle:
                ri = r[i]
                for w in range(4):
                    if top_free and w == 3:
                        continue                   # V3.1 open top: y = Ly is a free surface
                    if w == 0:
                        pen = ri - xi
                    elif w == 1:
                        pen = xi - (Lx - ri)
                    elif w == 2:
                        pen = ri - yi
                    else:
                        pen = yi - (Ly - ri)
                    if pen > 0:
                        Fw, a_w = jkr_force_from_overlap(pen, ri, E_star_gw, W_wall)
                        if w == 0:
                            fx += Fw
                            wall_d = abs(xi)
                            nxw = -1.0
                            nyw = 0.0
                        elif w == 1:
                            fx -= Fw
                            wall_d = abs(Lx - xi)
                            nxw = 1.0
                            nyw = 0.0
                        elif w == 2:
                            fy += Fw
                            wall_d = abs(yi)
                            nxw = 0.0
                            nyw = -1.0
                        else:
                            fy -= Fw
                            wall_d = abs(Ly - yi)
                            nxw = 0.0
                            nyw = 1.0
                        if nclip < max_clips:
                            clip_n[i, nclip, 0] = nxw
                            clip_n[i, nclip, 1] = nyw
                            clip_d[i, nclip] = wall_d
                            nclip += 1
                        else:
                            nover += 1
            else:
                for w in range(4):
                    if top_free and w == 3:
                        continue                   # V3.1 open top: y = Ly is a free surface
                    if w == 0:
                        wall_pos = 0.0
                        axis = 0
                        sign = 1
                    elif w == 1:
                        wall_pos = Lx
                        axis = 0
                        sign = -1
                    elif w == 2:
                        wall_pos = 0.0
                        axis = 1
                        sign = 1
                    else:
                        wall_pos = Ly
                        axis = 1
                        sign = -1
                    # V3.4: the support-function wall solver returns the contact
                    # point, so the lever arm is free. A circle's wall contact is
                    # on the centre line and its torque is identically zero, so
                    # the point variant is only worth calling for shapes.
                    wcx = 0.0
                    wcy = 0.0
                    if wall_torque:
                        ok, pen, R_local, wcx, wcy = se2d_wall_point_k(
                            xi, yi, a[i], b[i], n_shape[i], theta[i], wall_pos, axis, sign)
                    else:
                        ok, pen, R_local = se2d_wall_k(xi, yi, a[i], b[i], n_shape[i],
                                                       theta[i], wall_pos, axis, sign)
                    if ok:
                        Fw, a_w = jkr_force_from_overlap(pen, R_local, E_star_gw, W_wall)
                        if axis == 0:
                            fx += sign * Fw
                            wall_d = abs(xi - wall_pos)
                            nxw = -float(sign)
                            nyw = 0.0
                            if wall_torque:
                                tq += -(wcy - yi) * sign * Fw
                        else:
                            fy += sign * Fw
                            wall_d = abs(yi - wall_pos)
                            nxw = 0.0
                            nyw = -float(sign)
                            if wall_torque:
                                tq += (wcx - xi) * sign * Fw
                        if nclip < max_clips:
                            clip_n[i, nclip, 0] = nxw
                            clip_n[i, nclip, 1] = nyw
                            clip_d[i, nclip] = wall_d
                            nclip += 1
                        else:
                            nover += 1
        # ── MC-DEM correction, in pair order ──
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if dF[k] != 0.0:
                if nbr_side[m] == 0:
                    fx -= dF[k] * c_nx[k]
                    fy -= dF[k] * c_ny[k]
                else:
                    fx += dF[k] * c_nx[k]
                    fy += dF[k] * c_ny[k]
        F[i, 0] = fx
        F[i, 1] = fy
        torques[i] = tq
        clip_cnt[i] = nclip
        clip_over[i] = nover


def compute_forces_2d(gs, p, rng):
    """Drop-in for ``reference.compute_forces``: returns (F (N,2), torques (N,), ContactSoA)."""
    N = gs.N
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    periodic = (p.boundary_mode == 'periodic')
    t0 = time.perf_counter()
    pos = np.ascontiguousarray(gs.pos[:N, :2])
    vel = np.ascontiguousarray(gs.vel[:N, :2])
    cutoff = 2 * float(np.max(gs.r_bound)) + p.L_max
    pair_i, pair_j = half_pairs(pos, cutoff, periodic, (p.Lx, p.Ly), neighbor_backend(p))
    M = pair_i.shape[0]
    t1 = time.perf_counter()

    # V3.2: cells bridging across a contact resist shear at it
    _cell_adh = float(getattr(p, 'cell_contact_adhesion', 0.0))
    _bridge_w = (pair_bridge_weight(gs, p, pair_i, pair_j) if _cell_adh > 0.0
                 else np.zeros(M))
    rec = alloc_records(M, 2)
    pair_pass_2d(pos, vel, gs.r, gs.a, gs.b, gs.n_shape, gs.theta, gs.species_id,
                 bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly),
                 pair_i, pair_j, gs.pair_W, gs.pair_tau, gs.pair_Estar, float(p.friction_v_ref),
                 float(getattr(p, 'friction_mu', 0.0)),
                 float(_cell_adh), _bridge_w, float(getattr(p, 'curvature_R_cap', 0.0)),
                 rec['hit'], rec['overlap'], rec['nx'], rec['ny'], rec['cx'], rec['cy'], rec['Reff'],
                 rec['Fn'], rec['a'], rec['A'], rec['Ftx'], rec['Fty'], rec['d'], rec['dx'], rec['dy'])

    si = gs.species_id[pair_i]
    sj = gs.species_id[pair_j]
    c_Estar = np.ascontiguousarray(gs.pair_Estar[si, sj])
    c_W = np.ascontiguousarray(gs.pair_W[si, sj])
    c_tau = np.ascontiguousarray(gs.pair_tau[si, sj])
    dF = np.zeros(M)
    c_kappa = np.ones(M)
    if p.mc_dem_enabled and M > 0:
        mc_dem_pairs(pair_i, pair_j, gs.r, gs.nu_gran, float(p.mc_dem_kappa_max),
                     rec['hit'], rec['overlap'], rec['Reff'], c_Estar, dF, c_kappa)

    off, nbr_pair, nbr_side = csr_from_pairs(pair_i, pair_j, N)
    F = np.zeros((N, 2))
    torques = np.zeros(N)
    max_clips = int(p.perf_max_clips)
    clip_n = np.zeros((N, max_clips, 2))
    clip_d = np.zeros((N, max_clips))
    clip_cnt = np.zeros(N, dtype=np.int32)
    clip_over = np.zeros(N, dtype=np.int32)
    gather_2d(pos, gs.r, gs.a, gs.b, gs.n_shape, gs.theta, gs.species_id,
              bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly),
              bool(boundary_geometry(p, '2D').top_free),
              off, nbr_pair, nbr_side,
              rec['hit'], rec['overlap'], rec['nx'], rec['ny'], rec['cx'], rec['cy'], rec['Fn'],
              rec['a'], rec['Ftx'], rec['Fty'], dF,
              gs.wall_W, gs.wall_Estar, F, torques, clip_n, clip_d, clip_cnt, clip_over,
              bool(getattr(p, 'contact_wall_torque', False)) and not bool(gs.is_circle))
    gs.clip_arrays = (clip_n, clip_d, clip_cnt)     # consumed by the compiled renderer
    _warn_clip_overflow(clip_over, max_clips)

    contacts = ContactSoA.from_records(pair_i, pair_j, rec, gs, c_Estar, c_W, c_tau, dF, c_kappa, 2)
    t2 = time.perf_counter()

    if getattr(p, 'perf_cells_backend', 'kernels') == 'kernels':
        from gels.kernels.cells import bridging_k
        n_cand = bridging_k(gs, p, rng, F, pair_i, pair_j, rec, pos, 2, (off, nbr_pair, nbr_side))
    else:
        n_cand = bridging_pass(gs, p, rng, F, pair_i, pair_j, rec, pos, 2, csr=(off, nbr_pair, nbr_side))
    t3 = time.perf_counter()
    apply_gravity(gs, p, F)               # V3.1 buoyant weight along -y
    add_active_noise(gs, p, rng, F)
    TIMINGS.update(neighbours=t1 - t0, contacts=t2 - t1, bridging=t3 - t2,
                   noise=time.perf_counter() - t3, n_pairs=M, n_candidates=n_cand,
                   n_contacts=len(contacts))
    return F, torques, contacts


# Wall-clock split of the last compute_forces_2d call (seconds), for gels.bench.
TIMINGS = {}


_overflow_warned = [False]


def _warn_clip_overflow(clip_over, max_clips):
    if not _overflow_warned[0] and clip_over.size and int(clip_over.max()) > 0:
        _overflow_warned[0] = True
        print(f"  note: a granule had more than perf_max_clips={max_clips} contact clip planes; "
              f"extra planes are dropped (rendering only)")


__all__ = ['pair_pass_2d', 'mc_dem_pairs', 'gather_2d', 'compute_forces_2d']
