"""
Compiled 3D contact forces (V3.0 Phase 6a).
==========================================

Same three-pass structure as ``contact2d``: ``pair_pass_3d`` (prange over
half pairs → records), ``mc_dem_pairs`` (shared), ``gather_3d`` (prange over
particles → F, torque vectors, clips, six walls). Bridging runs in Python
over the same candidate pairs in pair order (``gels.kernels.bridging``);
noise is the reference's numpy code.
"""

import time

import numpy as np

from gels.engine import (apply_gravity, boundary_geometry, jkr_force_from_overlap,
                         pair_bridge_weight)
from gels.kernels import njit, prange
from gels.kernels.bridging import bridging_pass
from gels.kernels.contact2d import _warn_clip_overflow, mc_dem_pairs
from gels.kernels.contacts import ContactSoA, add_active_noise, alloc_records, clips_to_lists
from gels.kernels.geometry3d import se3d_contact_k, sph3d_contact_k, wall3d_k, wall3d_plane_k
from gels.kernels.neighbors import csr_from_pairs, half_pairs, neighbor_backend


@njit(parallel=True, cache=True, nogil=True)
def pair_pass_3d(pos, vel, r, a, b, c, n1, n2, quat, sid, is_sphere, periodic, Lx, Ly, Lz,
                 pair_i, pair_j, pair_W, pair_tau, pair_Estar, v_ref, mu, cell_adh, bridge_w, R_cap,
                 c_hit, c_overlap, c_nx, c_ny, c_nz, c_cx, c_cy, c_cz, c_Reff, c_Fn, c_a, c_A,
                 c_Ftx, c_Fty, c_Ftz, c_d, c_dx, c_dy, c_dz):
    M = pair_i.shape[0]
    for k in prange(M):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        c_d[k] = d
        c_dx[k] = dx
        c_dy[k] = dy
        c_dz[k] = dz
        c_overlap[k] = 0.0
        c_cx[k] = 0.0
        c_cy[k] = 0.0
        c_cz[k] = 0.0
        c_Reff[k] = 0.0
        c_Fn[k] = 0.0
        c_a[k] = 0.0
        c_A[k] = 0.0
        c_Ftx[k] = 0.0
        c_Fty[k] = 0.0
        c_Ftz[k] = 0.0
        if d < 1e-6:
            c_hit[k] = -1
            c_nx[k] = 0.0
            c_ny[k] = 0.0
            c_nz[k] = 0.0
            continue
        nx = dx / d
        ny = dy / d
        nz = dz / d
        xj_v = pos[i, 0] + dx
        yj_v = pos[i, 1] + dy
        zj_v = pos[i, 2] + dz

        if is_sphere:
            ok, overlap, snx, sny, snz, cx, cy, cz, R_eff = sph3d_contact_k(
                pos[i, 0], pos[i, 1], pos[i, 2], r[i], xj_v, yj_v, zj_v, r[j])
        else:
            ok, overlap, snx, sny, snz, cx, cy, cz, R_eff = se3d_contact_k(
                pos[i, 0], pos[i, 1], pos[i, 2], a[i], b[i], c[i], n1[i], n2[i], quat[i],
                xj_v, yj_v, zj_v, a[j], b[j], c[j], n1[j], n2[j], quat[j])
        if ok:
            c_hit[k] = 1
            nx = snx
            ny = sny
            nz = snz
        else:
            c_hit[k] = 0
            overlap = 0.0
            R_eff = 0.0
            cx = 0.0
            cy = 0.0
            cz = 0.0
        if R_cap > 0.0 and R_eff > 0.0:
            # V3.2: superellipsoid curvature runs away at a flat face (see the 2D twin)
            lim = R_cap * min(r[i], r[j])
            if R_eff > lim:
                R_eff = lim
        c_nx[k] = nx
        c_ny[k] = ny
        c_nz[k] = nz
        c_overlap[k] = overlap
        c_Reff[k] = R_eff
        c_cx[k] = cx
        c_cy[k] = cy
        c_cz[k] = cz

        if ok and overlap > 0:
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
            dvz = vel[j, 2] - vel[i, 2]
            v_dot_n = dvx * nx + dvy * ny + dvz * nz
            vtx = dvx - v_dot_n * nx
            vty = dvy - v_dot_n * ny
            vtz = dvz - v_dot_n * nz
            vt_mag = np.sqrt(vtx*vtx + vty*vty + vtz*vtz)
            if vt_mag > 1e-12:
                F_t_cap = tau_0 * A_contact * 1e-3
                if mu > 0.0 and F_normal > 0.0:
                    F_t_cap += mu * F_normal
                if cell_adh > 0.0:
                    F_t_cap += cell_adh * bridge_w[k]   # V3.2: the cells' own grip
                F_fric = F_t_cap * np.tanh(vt_mag / v_ref)
                c_Ftx[k] = F_fric * (vtx / vt_mag)
                c_Fty[k] = F_fric * (vty / vt_mag)
                c_Ftz[k] = F_fric * (vtz / vt_mag)


@njit(parallel=True, cache=True, nogil=True)
def gather_3d(pos, r, a, b, c, n1, n2, quat, r_bound, sid, is_sphere, periodic, Lx, Ly, Lz,
              shape_code, R_cyl, cxc, cyc, top_free,
              off, nbr_pair, nbr_side,
              c_hit, c_overlap, c_nx, c_ny, c_nz, c_cx, c_cy, c_cz, c_Fn, c_a, c_Ftx, c_Fty, c_Ftz, dF,
              wall_W, wall_Estar, F, torques, clip_n, clip_d, clip_cnt, clip_over):
    N = pos.shape[0]
    max_clips = clip_d.shape[1]
    for i in prange(N):
        fx = 0.0
        fy = 0.0
        fz = 0.0
        tx = 0.0
        ty = 0.0
        tz = 0.0
        nclip = 0
        nover = 0
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if c_hit[k] != 1 or c_overlap[k] <= 0.0:
                continue
            side = nbr_side[m]
            nx = c_nx[k]
            ny = c_ny[k]
            nz = c_nz[k]
            Fnx = c_Fn[k] * nx
            Fny = c_Fn[k] * ny
            Fnz = c_Fn[k] * nz
            Ftx = c_Ftx[k]
            Fty = c_Fty[k]
            Ftz = c_Ftz[k]
            if side == 0:
                fx -= Fnx
                fy -= Fny
                fz -= Fnz
                if not is_sphere:
                    rcx = c_cx[k] - pos[i, 0]
                    rcy = c_cy[k] - pos[i, 1]
                    rcz = c_cz[k] - pos[i, 2]
                    # torque += rc × (−Fn)
                    tx += rcy * (-Fnz) - rcz * (-Fny)
                    ty += rcz * (-Fnx) - rcx * (-Fnz)
                    tz += rcx * (-Fny) - rcy * (-Fnx)
                fx += Ftx
                fy += Fty
                fz += Ftz
                if not is_sphere:
                    tx += rcy * Ftz - rcz * Fty
                    ty += rcz * Ftx - rcx * Ftz
                    tz += rcx * Fty - rcy * Ftx
            else:
                fx += Fnx
                fy += Fny
                fz += Fnz
                if not is_sphere:
                    rcx = c_cx[k] - pos[i, 0]
                    rcy = c_cy[k] - pos[i, 1]
                    rcz = c_cz[k] - pos[i, 2]
                    tx += rcy * Fnz - rcz * Fny
                    ty += rcz * Fnx - rcx * Fnz
                    tz += rcx * Fny - rcy * Fnx
                fx -= Ftx
                fy -= Fty
                fz -= Ftz
                if not is_sphere:
                    tx += rcy * (-Ftz) - rcz * (-Fty)
                    ty += rcz * (-Ftx) - rcx * (-Ftz)
                    tz += rcx * (-Fty) - rcy * (-Ftx)
            if c_a[k] > 1e-6:
                if nclip < max_clips:
                    if side == 0:
                        clip_n[i, nclip, 0] = nx
                        clip_n[i, nclip, 1] = ny
                        clip_n[i, nclip, 2] = nz
                    else:
                        clip_n[i, nclip, 0] = -nx
                        clip_n[i, nclip, 1] = -ny
                        clip_n[i, nclip, 2] = -nz
                    clip_d[i, nclip] = np.sqrt(max(r[i]**2 - c_a[k]**2, 0.01))
                    nclip += 1
                else:
                    nover += 1
        # ── container walls: six box faces, or floor (+ lid) and the cylinder ──
        if not periodic:
            si = sid[i]
            W_wall = wall_W[si]
            E_star_gw = wall_Estar[si]
            for w in range(6):
                axis = w // 2
                if top_free and w == 5:
                    continue                       # V3.1 open top
                if shape_code == 1 and axis < 2:
                    continue                       # V3.1 cylinder replaces the x/y faces
                if w % 2 == 0:
                    wall_pos = 0.0
                    sign = 1
                else:
                    if axis == 0:
                        wall_pos = Lx
                    elif axis == 1:
                        wall_pos = Ly
                    else:
                        wall_pos = Lz
                    sign = -1
                coord = pos[i, axis]
                if is_sphere:
                    ri = r[i]
                    if sign > 0:
                        pen = ri - (coord - wall_pos)
                    else:
                        pen = (coord + ri) - wall_pos
                    ok = pen > 0
                    R_local = ri
                else:
                    ok, pen, R_local = wall3d_k(pos[i, 0], pos[i, 1], pos[i, 2], a[i], b[i], c[i],
                                                n1[i], n2[i], quat[i], r_bound[i], wall_pos, axis, sign)
                if ok:
                    Fw, a_w = jkr_force_from_overlap(pen, R_local, E_star_gw, W_wall)
                    if axis == 0:
                        fx += sign * Fw
                    elif axis == 1:
                        fy += sign * Fw
                    else:
                        fz += sign * Fw
                    if nclip < max_clips:
                        clip_n[i, nclip, 0] = 0.0
                        clip_n[i, nclip, 1] = 0.0
                        clip_n[i, nclip, 2] = 0.0
                        clip_n[i, nclip, axis] = -float(sign)
                        clip_d[i, nclip] = abs(coord - wall_pos)
                        nclip += 1
                    else:
                        nover += 1
            # ── V3.1 cylindrical side wall: axis z, radius R_cyl about (cxc, cyc) ──
            if shape_code == 1:
                dxc = pos[i, 0] - cxc
                dyc = pos[i, 1] - cyc
                rho = np.sqrt(dxc * dxc + dyc * dyc)
                if rho > 1e-12:
                    ux = dxc / rho
                    uy = dyc / rho
                    if is_sphere:
                        pen_c = r[i] + rho - R_cyl
                        ok_c = pen_c > 0
                        R_loc_c = r[i]
                    else:
                        ok_c, pen_c, R_loc_c = wall3d_plane_k(
                            pos[i, 0], pos[i, 1], pos[i, 2], a[i], b[i], c[i], n1[i], n2[i], quat[i],
                            r_bound[i], cxc + R_cyl * ux, cyc + R_cyl * uy, pos[i, 2], -ux, -uy, 0.0)
                    if ok_c:
                        Fw, a_w = jkr_force_from_overlap(pen_c, R_loc_c, E_star_gw, W_wall)
                        fx -= Fw * ux
                        fy -= Fw * uy
                        if nclip < max_clips:
                            clip_n[i, nclip, 0] = ux
                            clip_n[i, nclip, 1] = uy
                            clip_n[i, nclip, 2] = 0.0
                            clip_d[i, nclip] = R_cyl - rho
                            nclip += 1
                        else:
                            nover += 1
        # ── MC-DEM correction ──
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            if dF[k] != 0.0:
                if nbr_side[m] == 0:
                    fx -= dF[k] * c_nx[k]
                    fy -= dF[k] * c_ny[k]
                    fz -= dF[k] * c_nz[k]
                else:
                    fx += dF[k] * c_nx[k]
                    fy += dF[k] * c_ny[k]
                    fz += dF[k] * c_nz[k]
        F[i, 0] = fx
        F[i, 1] = fy
        F[i, 2] = fz
        torques[i, 0] = tx
        torques[i, 1] = ty
        torques[i, 2] = tz
        clip_cnt[i] = nclip
        clip_over[i] = nover


def compute_forces_3d(gs, p, rng):
    """Drop-in for ``reference.compute_forces_3d``: (F (N,3), torques (N,3), ContactSoA)."""
    N = gs.N
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    periodic = (p.boundary_mode == 'periodic')
    t0 = time.perf_counter()
    pos = np.ascontiguousarray(gs.pos[:N, :3])
    vel = np.ascontiguousarray(gs.vel[:N, :3])
    quat = np.ascontiguousarray(gs.quat) if gs.quat is not None else np.zeros((N, 4))
    if gs.quat is None:
        quat[:, 0] = 1.0
    cutoff = 2 * float(np.max(gs.r_bound)) + p.L_max
    pair_i, pair_j = half_pairs(pos, cutoff, periodic, (p.Lx, p.Ly, p.Lz), neighbor_backend(p))
    M = pair_i.shape[0]
    t1 = time.perf_counter()

    # V3.2: cells bridging across a contact resist shear at it
    _cell_adh = float(getattr(p, 'cell_contact_adhesion', 0.0))
    _bridge_w = (pair_bridge_weight(gs, p, pair_i, pair_j) if _cell_adh > 0.0
                 else np.zeros(M))
    rec = alloc_records(M, 3)
    pair_pass_3d(pos, vel, gs.r, gs.a, gs.b, gs.c, gs.n1, gs.n2, quat, gs.species_id,
                 bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly), float(p.Lz),
                 pair_i, pair_j, gs.pair_W, gs.pair_tau, gs.pair_Estar, float(p.friction_v_ref),
                 float(getattr(p, 'friction_mu', 0.0)),
                 float(_cell_adh), _bridge_w, float(getattr(p, 'curvature_R_cap', 0.0)),
                 rec['hit'], rec['overlap'], rec['nx'], rec['ny'], rec['nz'], rec['cx'], rec['cy'], rec['cz'],
                 rec['Reff'], rec['Fn'], rec['a'], rec['A'], rec['Ftx'], rec['Fty'], rec['Ftz'],
                 rec['d'], rec['dx'], rec['dy'], rec['dz'])

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

    geom = boundary_geometry(p, '3D')
    off, nbr_pair, nbr_side = csr_from_pairs(pair_i, pair_j, N)
    F = np.zeros((N, 3))
    torques = np.zeros((N, 3))
    max_clips = max(int(p.perf_max_clips), 24)
    clip_n = np.zeros((N, max_clips, 3))
    clip_d = np.zeros((N, max_clips))
    clip_cnt = np.zeros(N, dtype=np.int32)
    clip_over = np.zeros(N, dtype=np.int32)
    gather_3d(pos, gs.r, gs.a, gs.b, gs.c, gs.n1, gs.n2, quat, gs.r_bound, gs.species_id,
              bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly), float(p.Lz),
              int(geom.shape_code), float(geom.R_cyl), float(geom.cx), float(geom.cy), bool(geom.top_free),
              off, nbr_pair, nbr_side,
              rec['hit'], rec['overlap'], rec['nx'], rec['ny'], rec['nz'], rec['cx'], rec['cy'], rec['cz'],
              rec['Fn'], rec['a'], rec['Ftx'], rec['Fty'], rec['Ftz'], dF,
              gs.wall_W, gs.wall_Estar, F, torques, clip_n, clip_d, clip_cnt, clip_over)
    gs.clip_arrays = (clip_n, clip_d, clip_cnt)     # consumed by the compiled renderer
    _warn_clip_overflow(clip_over, max_clips)

    contacts = ContactSoA.from_records(pair_i, pair_j, rec, gs, c_Estar, c_W, c_tau, dF, c_kappa, 3)
    t2 = time.perf_counter()

    if getattr(p, 'perf_cells_backend', 'kernels') == 'kernels':
        from gels.kernels.cells import bridging_k
        n_cand = bridging_k(gs, p, rng, F, pair_i, pair_j, rec, pos, 3, (off, nbr_pair, nbr_side))
    else:
        n_cand = bridging_pass(gs, p, rng, F, pair_i, pair_j, rec, pos, 3, csr=(off, nbr_pair, nbr_side))
    t3 = time.perf_counter()
    apply_gravity(gs, p, F)               # V3.1 buoyant weight along -z
    add_active_noise(gs, p, rng, F)
    TIMINGS.update(neighbours=t1 - t0, contacts=t2 - t1, bridging=t3 - t2,
                   noise=time.perf_counter() - t3, n_pairs=M, n_candidates=n_cand,
                   n_contacts=len(contacts))
    return F, torques, contacts


# Wall-clock split of the last compute_forces_3d call (seconds), for gels.bench.
TIMINGS = {}


__all__ = ['pair_pass_3d', 'gather_3d', 'compute_forces_3d']
