"""
Compiled cell state machine and bridging (V3.0 Phase 6c).
========================================================

Three parallel passes replace ``reference.update_cell_state`` /
``_update_individual_cells`` and the bridging half of the reference force
loops:

  aggregates_k     prange over granules → attachment, spreading, FA maturity,
                   overcrowding (deterministic; matches the reference to rounding)
  cells_update_k   prange over granules, sequential over each granule's cells →
                   bridge ageing / lock-in / senescence, directed migration,
                   state assignment, random-walk migration
  bridge_pairs_k   prange over candidate pairs → gap, per-side traction, path
                   factor (restricted ray-cast over N(i) ∪ N(j)), committed count
  bridge_cells_k   prange over granules → service committed bridges (rupture or
                   force), attempt new bridges for the granule's own cells; each
                   granule writes only its own cells and its own force row.

Randomness comes from a counter-based hash (splitmix64) keyed by one 62-bit
seed drawn from the run's ``Generator`` per call plus the cell / partner
indices, so results are deterministic for a seed and independent of the
thread count — but not the reference's random stream: bridging statistics
are equivalent, trajectories are not comparable step by step
(``Params.perf_cells_backend = 'python'`` keeps the exact reference cell
machinery on top of the compiled contacts).

Semantic differences from the reference, on purpose:
  * a cell committed to a bridge in pair (i, j) stops being eligible for later
    pairs of the same step, as before; but the "existing bridges" count that
    boosts the attempt rate is taken at the start of the step (the reference
    saw bridges formed earlier in the same sequential sweep);
  * with periodic boundaries the partner granule is always the minimum image
    of the owner (the reference mixed a virtual image with real positions on
    the second side of a pair straddling the boundary).
"""

import numpy as np

from gels.engine import (CellState, adhesion_force_ceiling, capacity_coverage,
                         quat_rotate, quat_rotate_inv,
                         superellipse_point, superellipsoid_point)
from gels.kernels import njit, prange
from gels.engine import PACKING_EFFICIENCY
from gels.materials import blocker_factor, hill_pair_factor, law_code, rule_code, traction_gain

ATTACHED = int(CellState.ATTACHED)
SPREADING = int(CellState.SPREADING)
PROLIFERATING = int(CellState.PROLIFERATING)
MIGRATING = int(CellState.MIGRATING)
BRIDGING = int(CellState.BRIDGING)
SENESCENT = int(CellState.SENESCENT)

_TWO_PI = 2.0 * np.pi
_ETA_CLAMP = 0.85 * np.pi / 2

_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xBF58476D1CE4E5B9)
_K3 = np.uint64(0x94D049BB133111EB)
_S30 = np.uint64(30)
_S27 = np.uint64(27)
_S31 = np.uint64(31)
_S11 = np.uint64(11)
_INV53 = 1.0 / 9007199254740992.0


# ──────────────────────────────────────────────────────────────────────
# counter-based random numbers
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _mix64(x):
    x = x ^ (x >> _S30)
    x = x * _K2
    x = x ^ (x >> _S27)
    x = x * _K3
    x = x ^ (x >> _S31)
    return x


@njit(cache=True)
def hash_u01(seed, a, b, salt):
    """Uniform in [0, 1) from (seed, a, b, salt); 53 random bits."""
    x = (np.uint64(seed) + _K1 * (np.uint64(a) + np.uint64(1))
         + _K2 * (np.uint64(b) + np.uint64(2)) + _K3 * (np.uint64(salt) + np.uint64(3)))
    z = _mix64(x)
    return (z >> _S11) * _INV53


@njit(cache=True)
def hash_normal2(seed, a, b, salt):
    """Two standard normals (Box–Muller) from (seed, a, b, salt)."""
    u1 = hash_u01(seed, a, b, salt)
    u2 = hash_u01(seed, a, b, salt + 1)
    if u1 < 1e-300:
        u1 = 1e-300
    rad = np.sqrt(-2.0 * np.log(u1))
    th = _TWO_PI * u2
    return rad * np.cos(th), rad * np.sin(th)


def _adh_or_zeros(gs, p):
    """V3.6 adhesion ceiling as an (N,) float array, zeros when the feature is off.

    The kernel cannot take None, and a zero entry is the documented "off" value
    in `mc_force_k`, so this is the whole adaptation.
    """
    a = adhesion_force_ceiling(gs, p)
    return np.ascontiguousarray(a, dtype=np.float64) if a is not None \
        else np.zeros(gs.N, dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────
# compiled leaves
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def mc_force_k(E_kPa, nu, fa, g, F_stall, a_cell, k_opt, engagement, F_max, F_adh):
    """``engine.motor_clutch_force`` with the Params scalars pre-extracted.

    ``F_adh`` is the V3.6 adhesion ceiling for this granule AT g = 1, from
    `engine.adhesion_force_ceiling`; <= 0 means the feature is off and the cap is
    ``F_max`` alone. It is scaled by g HERE because the ligand gain is a property
    of the pair, and the adhesion stress scales with engaged-bond density.
    """
    k_sub = np.pi * E_kPa * a_cell / (1.0 - nu**2)
    F_mc = F_stall * (k_sub / (k_sub + g * k_opt)) * engagement * fa * g
    cap = F_max
    if F_adh > 0.0:
        cap_adh = F_adh * g
        if cap_adh < cap:
            cap = cap_adh
    return min(F_mc, cap)


@njit(cache=True)
def _proj_area(spread_frac, cell_d, cell_h):
    r = cell_d / 2.0
    A_sphere = np.pi * r ** 2
    V = (4.0 / 3.0) * np.pi * r ** 3
    c = cell_h / 2.0
    a2 = 3.0 * V / (4.0 * np.pi * c)
    A_spread = np.pi * a2
    return A_sphere + spread_frac * (A_spread - A_sphere)


@njit(cache=True)
def _capacity(R, activity, spread_frac, cell_d, cell_h, cov_surface, cov, is_3d_like,
              foothold):
    """``engine.cell_capacity``: monolayer capacity scaled by the coverage gain.

    ``foothold`` mirrors ``engine.cells_from_surface_coverage``: the fraction of
    the fully-spread footprint a cell needs to hold a place, floored at the
    ROUNDED cell's cross-section. V3.3 bug fix -- this argument was missing, so
    with ``cell_capacity_foothold < 1`` the compiled path computed a different
    capacity from the reference (4x at the ``fibroblast_realistic`` default of
    0.25), and capacity sets the overcrowding and division thresholds.
    """
    if cov_surface > 0:
        A_spread = _proj_area(1.0, cell_d, cell_h)
        if foothold < 1.0:
            A_spread = max(_proj_area(0.0, cell_d, cell_h), foothold * A_spread)
        if is_3d_like:
            A_surface = 4.0 * np.pi * R ** 2
        else:
            A_surface = np.pi * R ** 2
        n_full = max(1, int(round(A_surface * cov_surface * PACKING_EFFICIENCY / A_spread)))
    else:
        A_cell = _proj_area(spread_frac, cell_d, cell_h)
        n_full = max(1, int(np.pi * R ** 2 * cov / A_cell))
    return int(round(activity * n_full))


@njit(cache=True)
def _cell_world_2d(gx, gy, th_g, a, b, n, th_cell):
    bx, by = superellipse_point(th_cell, a, b, n)
    ct = np.cos(th_g)
    st = np.sin(th_g)
    return gx + ct * bx - st * by, gy + st * bx + ct * by


@njit(cache=True)
def _cell_world_3d(gx, gy, gz, q, a, b, c, n1, n2, eta, omega):
    bp = superellipsoid_point(eta, omega, a, b, c, n1, n2)
    bw = quat_rotate(q, bp)
    return gx + bw[0], gy + bw[1], gz + bw[2]


@njit(cache=True)
def _ang_dist_2d(dx, dy, theta_g, cell_theta):
    """(angular distance, target body angle, signed shortest arc) — reference formula."""
    target_theta_world = np.arctan2(dy, dx)
    target_theta_body = (target_theta_world - theta_g) % _TWO_PI
    ct = cell_theta % _TWO_PI
    diff = target_theta_body - ct
    diff = (diff + np.pi) % _TWO_PI - np.pi
    return abs(diff), target_theta_body, diff


@njit(cache=True)
def _ang_dist_3d(dx, dy, dz, q, cell_eta, cell_omega):
    d_body = quat_rotate_inv(q, np.array([dx, dy, dz]))
    r_xy = np.sqrt(d_body[0]**2 + d_body[1]**2)
    target_eta = np.arctan2(d_body[2], r_xy)
    target_omega = np.arctan2(d_body[1], d_body[0]) % _TWO_PI
    pcx = np.cos(cell_eta) * np.cos(cell_omega)
    pcy = np.cos(cell_eta) * np.sin(cell_omega)
    pcz = np.sin(cell_eta)
    ptx = np.cos(target_eta) * np.cos(target_omega)
    pty = np.cos(target_eta) * np.sin(target_omega)
    ptz = np.sin(target_eta)
    dot = pcx * ptx + pcy * pty + pcz * ptz
    if dot > 1.0:
        dot = 1.0
    if dot < -1.0:
        dot = -1.0
    return np.arccos(dot), target_eta, target_omega


@njit(cache=True)
def _min_image(d, L):
    return d - L * np.rint(d / L)


# ──────────────────────────────────────────────────────────────────────
# Part A: per-granule aggregates
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, nogil=True)
def aggregates_k(t, adhesive, n_cells, E_gran, nu_gran, r, activity,
                 n_attached, spread, fa, n_over,
                 a_cell, k_opt, t_spread, t_onset, t_half, fa_rate, cell_d, cell_h,
                 cov_surface, cov, is_3d_like, clock_offset, foothold):
    N = adhesive.shape[0]
    for i in prange(N):
        if not adhesive[i]:
            continue
        k_sub = np.pi * E_gran[i] * a_cell / (1.0 - nu_gran[i] ** 2)
        stiffness_factor = k_sub / (k_sub + k_opt)
        eff_spread_dur = t_spread / max(0.2, stiffness_factor)

        # V3.2 per-granule clock offset (zero unless clock_jitter_h is set)
        t_i = t + clock_offset[i]

        if t_i >= t_onset:
            if t_half <= 0.01:
                frac = 1.0
            else:
                tau = t_i - t_onset
                frac = 1.0 / (1.0 + np.exp(-3.0 * (tau - t_half) / max(0.1, t_half)))
            n_attached[i] = n_cells[i] * frac
        else:
            n_attached[i] = 0.0

        if n_attached[i] > 0.5:
            t_since = max(0.0, t_i - t_onset - t_half)
            spread[i] = min(1.0, t_since / eff_spread_dur)
        else:
            spread[i] = 0.0

        if spread[i] > 0.1:
            t_since_spread = max(0.0, t_i - (t_onset + t_half))
            fa[i] = min(1.0, t_since_spread * fa_rate)
        else:
            fa[i] = 0.0

        cap = _capacity(r[i], activity[i], spread[i], cell_d, cell_h, cov_surface, cov, is_3d_like,
                        foothold)
        n_over[i] = max(0.0, n_attached[i] - cap)


# ──────────────────────────────────────────────────────────────────────
# Part B: per-cell state machine
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, nogil=True)
def cells_update_k(seed, do_walk, dt, is_3d, periodic, Lx, Ly, Lz,
                   adhesive, cell_offset, n_attached, n_over, spread, fa, r, r_bound, f, pos, theta, quat,
                   activity,
                   cell_state, cell_target, cell_bridge_age, cell_align, cell_locked, cell_fx, cell_fy, cell_fz,
                   cell_area, cell_over_age, cell_theta, cell_eta, cell_omega,
                   cell_d, cell_h, cov_surface, cov, is_3d_like, foothold, stacking_max, over_sen_time,
                   align_rate, align_min, lock_thr0, lock_scales, law, rule, gamma, kappa,
                   sen_time, sense, commit_angle, mig_speed, directed_mult):
    N = adhesive.shape[0]
    for i in prange(N):
        if not adhesive[i]:
            continue
        c0 = cell_offset[i]
        c1 = cell_offset[i + 1]
        n_total = c1 - c0
        if n_total == 0:
            continue

        n_att = int(round(n_attached[i]))
        n_ov = int(round(n_over[i]))
        sf = spread[i]
        fa_i = fa[i]
        A_cell = _proj_area(sf, cell_d, cell_h)
        cap = _capacity(r[i], activity[i], sf, cell_d, cell_h, cov_surface, cov, is_3d_like,
                        foothold)
        effective_cap = int(round(cap * stacking_max))

        for k in range(n_total):
            ci = c0 + k
            cell_area[ci] = A_cell if k < n_att else 0.0
            st = cell_state[ci]

            if st == BRIDGING:
                cell_bridge_age[ci] += dt
                cell_align[ci] += align_rate * dt * (1.0 - cell_align[ci])
                cell_align[ci] = min(cell_align[ci], 1.0)
                F_mag = np.sqrt(cell_fx[ci]**2 + cell_fy[ci]**2 + cell_fz[ci]**2)
                lock_thr = lock_thr0
                if lock_scales:
                    tgt = cell_target[ci]
                    if tgt >= 0:
                        lock_thr = lock_thr * traction_gain(f[i], f[tgt], law, rule, gamma, kappa)
                if F_mag >= lock_thr:
                    cell_locked[ci] = True
                if cell_locked[ci]:
                    continue
                if cell_bridge_age[ci] >= sen_time:
                    cell_state[ci] = SENESCENT
                    cell_target[ci] = -1
                    cell_bridge_age[ci] = 0.0
                    cell_locked[ci] = False
                    cell_align[ci] = 0.0
                continue

            if st == MIGRATING:
                tgt = cell_target[ci]
                if tgt < 0 or tgt >= N:
                    cell_state[ci] = PROLIFERATING
                    cell_target[ci] = -1
                    continue
                dx = pos[tgt, 0] - pos[i, 0]
                dy = pos[tgt, 1] - pos[i, 1]
                dz = 0.0
                if is_3d:
                    dz = pos[tgt, 2] - pos[i, 2]
                if periodic:
                    dx = _min_image(dx, Lx)
                    dy = _min_image(dy, Ly)
                    if is_3d:
                        dz = _min_image(dz, Lz)
                d = np.sqrt(dx*dx + dy*dy + dz*dz)
                gap = max(d - r_bound[i] - r_bound[tgt], 0.0)
                if gap > sense:
                    cell_state[ci] = PROLIFERATING
                    cell_target[ci] = -1
                    cell_bridge_age[ci] = 0.0
                    continue
                if is_3d:
                    ang_d, tgt_eta, tgt_omega = _ang_dist_3d(dx, dy, dz, quat[i], cell_eta[ci], cell_omega[ci])
                    signed_diff = 0.0
                else:
                    ang_d, tgt_theta, signed_diff = _ang_dist_2d(dx, dy, theta[i], cell_theta[ci])
                    tgt_eta = 0.0
                    tgt_omega = 0.0
                if ang_d < commit_angle:
                    cell_state[ci] = BRIDGING
                    cell_bridge_age[ci] = 0.0
                    cell_align[ci] = align_min
                    continue
                directed_speed = mig_speed * directed_mult
                r_eff = max(r[i], 1.0)
                step = directed_speed * dt / r_eff
                if is_3d:
                    ce = cell_eta[ci]
                    co = cell_omega[ci]
                    pcx = np.cos(ce) * np.cos(co)
                    pcy = np.cos(ce) * np.sin(co)
                    pcz = np.sin(ce)
                    ptx = np.cos(tgt_eta) * np.cos(tgt_omega)
                    pty = np.cos(tgt_eta) * np.sin(tgt_omega)
                    ptz = np.sin(tgt_eta)
                    if ang_d > 1e-8:
                        frac = min(step, ang_d) / ang_d
                        sin_a = np.sin(ang_d)
                        if sin_a > 1e-12:
                            wa = np.sin((1.0 - frac) * ang_d) / sin_a
                            wb = np.sin(frac * ang_d) / sin_a
                            pnx = wa * pcx + wb * ptx
                            pny = wa * pcy + wb * pty
                            pnz = wa * pcz + wb * ptz
                        else:
                            pnx = pcx
                            pny = pcy
                            pnz = pcz
                        new_omega = np.arctan2(pny, pnx) % _TWO_PI
                        r_xy = np.sqrt(pnx**2 + pny**2)
                        new_eta = np.arctan2(pnz, r_xy)
                        cell_eta[ci] = min(max(new_eta, -_ETA_CLAMP), _ETA_CLAMP)
                        cell_omega[ci] = new_omega
                else:
                    actual_step = min(step, abs(signed_diff))
                    sgn = 1.0 if signed_diff > 0 else (-1.0 if signed_diff < 0 else 0.0)
                    cell_theta[ci] += sgn * actual_step
                    cell_theta[ci] = cell_theta[ci] % _TWO_PI
                cell_bridge_age[ci] += dt
                continue

            if st == SENESCENT:
                continue

            if k >= effective_cap:
                cell_state[ci] = SENESCENT
                cell_over_age[ci] = 0.0
            elif k >= cap and n_ov > 0:
                cell_over_age[ci] += dt
                if cell_over_age[ci] >= over_sen_time:
                    cell_state[ci] = SENESCENT
                    cell_over_age[ci] = 0.0
            elif sf >= 0.95 and fa_i >= 0.5:
                cell_state[ci] = PROLIFERATING
                cell_over_age[ci] = 0.0
            elif sf > 0.0:
                cell_state[ci] = SPREADING
                cell_over_age[ci] = 0.0
            else:
                cell_state[ci] = ATTACHED
                cell_over_age[ci] = 0.0

        # ── random-walk migration of mobile cells ──
        if do_walk and mig_speed > 0:
            r_eff = max(r[i], 1.0)
            sigma = mig_speed * dt / r_eff
            for k in range(n_total):
                ci = c0 + k
                st = cell_state[ci]
                if st != ATTACHED and st != SPREADING and st != PROLIFERATING:
                    continue
                z0, z1 = hash_normal2(seed, ci, 0, 7)
                if is_3d:
                    cell_eta[ci] += sigma * z0
                    cell_omega[ci] += sigma * z1
                    cell_eta[ci] = min(max(cell_eta[ci], -_ETA_CLAMP), _ETA_CLAMP)
                    cell_omega[ci] = cell_omega[ci] % _TWO_PI
                else:
                    cell_theta[ci] += sigma * z0
                    cell_theta[ci] = cell_theta[ci] % _TWO_PI


# ──────────────────────────────────────────────────────────────────────
# Part C: bridging
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _path_factor_k(i, j, gap, dpx, dpy, dpz, d_mag, is_3d, r_bound, f, pos,
                   off, nbr_pair, nbr_side, pair_i, pair_j, periodic, Lx, Ly, Lz,
                   contact_factor, decay_len_raw, inert_factor):
    """``reference._classify_bridge_path`` over the candidate blockers N(i) ∪ N(j)."""
    if gap <= 0:
        return contact_factor
    void_decay = np.exp(-gap / max(decay_len_raw, 0.1))
    if d_mag < 1e-12:
        return contact_factor * void_decay
    hx = dpx / d_mag
    hy = dpy / d_mag
    hz = dpz / d_mag
    ri = r_bound[i]
    rj = r_bound[j]
    best = 2.0
    found = False
    for owner in (i, j):
        for m in range(off[owner], off[owner + 1]):
            kk = nbr_pair[m]
            g = pair_j[kk] if nbr_side[m] == 0 else pair_i[kk]
            if g == i or g == j:
                continue
            gx = pos[g, 0] - pos[i, 0]
            gy = pos[g, 1] - pos[i, 1]
            gz = 0.0
            if is_3d:
                gz = pos[g, 2] - pos[i, 2]
            if periodic:
                gx = _min_image(gx, Lx)
                gy = _min_image(gy, Ly)
                if is_3d:
                    gz = _min_image(gz, Lz)
            t = gx * hx + gy * hy + gz * hz
            if t > ri * 0.5 and t < d_mag - rj * 0.5:
                px = gx - t * hx
                py = gy - t * hy
                pz = gz - t * hz
                d_perp = np.sqrt(px*px + py*py + pz*pz)
                if d_perp < r_bound[g]:
                    found = True
                    fac = blocker_factor(f[g], inert_factor)
                    if fac < best:
                        best = fac
    if not found:
        return void_decay
    return best * void_decay


@njit(parallel=True, cache=True, nogil=True)
def bridge_pairs_k(cidx, pair_i, pair_j, hit, overlap, d_rec, dx, dy, dz, nx_rec, ny_rec,
                   is_3d, is_circle, periodic, Lx, Ly, Lz,
                   r, r_bound, f, E_gran, nu_gran, fa, pos,
                   cell_offset, cell_state, cell_target, off, nbr_pair, nbr_side,
                   law, rule, gamma, kappa, F_stall, a_cell, k_opt, engagement, F_max,
                   F_adh_g1,
                   break_gap, sense, contact_factor, decay_len_raw, inert_factor,
                   cell_bridge_age, cell_align, cell_force_prev, cell_gap_prev,
                   model, dt, drag_scale, v0, f_ecc, gap_min, formation_time, align_min,
                   out_gap, out_nx, out_ny, out_nz, out_Fi, out_Fj, out_pf, out_nexist, out_fac):
    M = cidx.shape[0]
    for q in prange(M):
        k = cidx[q]
        i = pair_i[k]
        j = pair_j[k]
        dpx = dx[k]
        dpy = dy[k]
        dpz = 0.0
        if is_3d:
            dpz = dz[k]
            d = np.sqrt(dpx*dpx + dpy*dpy + dpz*dpz)
            nx = dpx / d
            ny = dpy / d
            nz = dpz / d
            if hit[k] == 1:
                gap = -overlap[k]
            else:
                gap = d - r_bound[i] - r_bound[j]
                if gap < 0:
                    gap = 0.0
        else:
            nx = nx_rec[k]
            ny = ny_rec[k]
            nz = 0.0
            d = d_rec[k]
            if is_circle:
                gap = d - r[i] - r[j]
            else:
                if hit[k] == 1:
                    gap = -overlap[k]
                else:
                    gap = d - r_bound[i] - r_bound[j]
                    if gap < 0:
                        gap = 0.0
        g_i = traction_gain(f[i], f[j], law, rule, gamma, kappa)
        g_j = traction_gain(f[j], f[i], law, rule, gamma, kappa)
        out_Fi[q] = mc_force_k(E_gran[i], nu_gran[i], fa[i], g_i, F_stall, a_cell, k_opt,
                               engagement, F_max, F_adh_g1[i])
        out_Fj[q] = mc_force_k(E_gran[j], nu_gran[j], fa[j], g_j, F_stall, a_cell, k_opt,
                               engagement, F_max, F_adh_g1[j])
        n_ex = 0
        # V3.1: the pair's isometric capacity, last step's force and last
        # step's gap drive one force-velocity factor for all of its bridges.
        F_iso = 0.0
        F_prev = 0.0
        gap_prev = np.nan
        if gap <= break_gap:
            for ci in range(cell_offset[i], cell_offset[i + 1]):
                if cell_state[ci] == BRIDGING and cell_target[ci] == j:
                    n_ex += 1
                    if model == 1:
                        mat = min(1.0, cell_bridge_age[ci] / max(0.1, formation_time))
                        al = max(cell_align[ci], align_min)
                        F_iso += min(out_Fi[q] * mat * al, F_max)
                        F_prev += cell_force_prev[ci]
                        if np.isnan(gap_prev):
                            gap_prev = cell_gap_prev[ci]
            for cj in range(cell_offset[j], cell_offset[j + 1]):
                if cell_state[cj] == BRIDGING and cell_target[cj] == i:
                    n_ex += 1
                    if model == 1:
                        mat = min(1.0, cell_bridge_age[cj] / max(0.1, formation_time))
                        al = max(cell_align[cj], align_min)
                        F_iso += min(out_Fj[q] * mat * al, F_max)
                        F_prev += cell_force_prev[cj]
                        if np.isnan(gap_prev):
                            gap_prev = cell_gap_prev[cj]
        fac = 1.0
        if model == 1 and F_iso > 0.0:
            v_rel = 0.0 if np.isnan(gap_prev) else (gap_prev - gap) / dt
            c_pair = 1.0 / (drag_scale * r[i]) + 1.0 / (drag_scale * r[j])
            fac = hill_pair_factor(v_rel, F_prev, F_iso, c_pair, v0, f_ecc, gap <= gap_min)
        out_fac[q] = fac
        pf = 0.0
        if gap < sense:
            pf = _path_factor_k(i, j, gap, dpx, dpy, dpz, d, is_3d, r_bound, f, pos,
                                off, nbr_pair, nbr_side, pair_i, pair_j, periodic, Lx, Ly, Lz,
                                contact_factor, decay_len_raw, inert_factor)
        out_gap[q] = gap
        out_nx[q] = nx
        out_ny[q] = ny
        out_nz[q] = nz
        out_pf[q] = pf
        out_nexist[q] = n_ex


@njit(parallel=True, cache=True, nogil=True)
def bridge_cells_k(seed, dt, is_3d, periodic, Lx, Ly, Lz,
                   cidx_of_pair, pair_i, pair_j, off, nbr_pair, nbr_side, dx, dy, dz,
                   out_gap, out_nx, out_ny, out_nz, out_Fi, out_Fj, out_pf, out_nexist, out_fac,
                   cell_force_prev, cell_gap_prev,
                   pos, r, r_bound, a, b, c, n1, n2, theta, quat, fa, adhesive,
                   cell_offset, cell_state, cell_target, cell_bridge_age, cell_align, cell_locked,
                   cell_theta, cell_eta, cell_omega, cell_fx, cell_fy, cell_fz, F,
                   break_gap, formation_time, align_min, F_max, sense, attempt_rate, secondary_mult,
                   decay_len_raw, min_fa, exclusion_angle, commit_angle):
    N = adhesive.shape[0]
    decay_len = max(decay_len_raw, 0.1)
    for i in prange(N):
        if not adhesive[i]:
            continue
        c0 = cell_offset[i]
        c1 = cell_offset[i + 1]
        if c1 == c0:
            continue
        fx_acc = 0.0
        fy_acc = 0.0
        fz_acc = 0.0
        for m in range(off[i], off[i + 1]):
            k = nbr_pair[m]
            q = cidx_of_pair[k]
            if q < 0:
                continue
            side = nbr_side[m]
            if side == 0:
                j = pair_j[k]
                nx = out_nx[q]
                ny = out_ny[q]
                nz = out_nz[q]
                Fc = out_Fi[q]
                dpx = dx[k]
                dpy = dy[k]
                dpz = dz[k] if is_3d else 0.0
            else:
                j = pair_i[k]
                nx = -out_nx[q]
                ny = -out_ny[q]
                nz = -out_nz[q]
                Fc = out_Fj[q]
                dpx = -dx[k]
                dpy = -dy[k]
                dpz = -dz[k] if is_3d else 0.0
            gap = out_gap[q]
            fac = out_fac[q]
            tx = pos[i, 0] + dpx
            ty = pos[i, 1] + dpy
            tz = pos[i, 2] + dpz if is_3d else 0.0

            # ── service committed bridges of i toward j ──
            for ci in range(c0, c1):
                if cell_state[ci] == BRIDGING and cell_target[ci] == j:
                    if gap > break_gap:
                        cell_state[ci] = SENESCENT
                        cell_target[ci] = -1
                        cell_bridge_age[ci] = 0.0
                        cell_locked[ci] = False
                        cell_align[ci] = 0.0
                        cell_gap_prev[ci] = np.nan
                        cell_force_prev[ci] = 0.0
                    else:
                        maturity = min(1.0, cell_bridge_age[ci] / max(0.1, formation_time))
                        alignment = max(cell_align[ci], align_min)
                        F_cell = min(Fc * maturity * alignment, F_max) * fac
                        cell_gap_prev[ci] = gap
                        cell_force_prev[ci] = F_cell
                        cell_fx[ci] += F_cell * nx
                        cell_fy[ci] += F_cell * ny
                        cell_fz[ci] += F_cell * nz
                        fx_acc += F_cell * nx
                        fy_acc += F_cell * ny
                        fz_acc += F_cell * nz

            # ── attempts by the granule's own eligible cells ──
            if gap < sense and fa[i] >= min_fa:
                rate = attempt_rate
                if out_nexist[q] > 0:
                    rate = rate * secondary_mult
                p_base = 1.0 - np.exp(-rate * dt)
                pf = out_pf[q]
                if gap <= 0:
                    obstruction = pf
                    use_decay = False
                else:
                    granule_decay = np.exp(-gap / decay_len)
                    if granule_decay > 1e-15:
                        obstruction = pf / granule_decay
                    else:
                        obstruction = pf
                    use_decay = True
                r_target = r_bound[j]

                # exclusion zone: bridging / migrating cells of i already headed to j
                n_ex = 0
                for ck in range(c0, c1):
                    stk = cell_state[ck]
                    if (stk == BRIDGING or stk == MIGRATING) and cell_target[ck] == j:
                        n_ex += 1
                ex_a = np.empty(n_ex)
                ex_b = np.empty(n_ex)
                n_ex = 0
                for ck in range(c0, c1):
                    stk = cell_state[ck]
                    if (stk == BRIDGING or stk == MIGRATING) and cell_target[ck] == j:
                        if is_3d:
                            ex_a[n_ex] = cell_eta[ck]
                            ex_b[n_ex] = cell_omega[ck]
                        else:
                            ex_a[n_ex] = cell_theta[ck]
                        n_ex += 1

                for ci in range(c0, c1):
                    st = cell_state[ci]
                    if st != SPREADING and st != PROLIFERATING:
                        continue
                    if is_3d:
                        cx, cy, cz = _cell_world_3d(pos[i, 0], pos[i, 1], pos[i, 2], quat[i],
                                                    a[i], b[i], c[i], n1[i], n2[i],
                                                    cell_eta[ci], cell_omega[ci])
                    else:
                        cx, cy = _cell_world_2d(pos[i, 0], pos[i, 1], theta[i], a[i], b[i], n1[i],
                                                cell_theta[ci])
                        cz = 0.0
                    ox = cx - pos[i, 0]
                    oy = cy - pos[i, 1]
                    oz = cz - pos[i, 2] if is_3d else 0.0
                    if ox * nx + oy * ny + oz * nz < 0:
                        continue
                    if exclusion_angle > 0 and n_ex > 0:
                        too_close = False
                        if is_3d:
                            ec = cell_eta[ci]
                            oc = cell_omega[ci]
                            pcx = np.cos(ec) * np.cos(oc)
                            pcy = np.cos(ec) * np.sin(oc)
                            pcz = np.sin(ec)
                            for e in range(n_ex):
                                pkx = np.cos(ex_a[e]) * np.cos(ex_b[e])
                                pky = np.cos(ex_a[e]) * np.sin(ex_b[e])
                                pkz = np.sin(ex_a[e])
                                dot = pcx * pkx + pcy * pky + pcz * pkz
                                if dot > 1.0:
                                    dot = 1.0
                                if dot < -1.0:
                                    dot = -1.0
                                if np.arccos(dot) < exclusion_angle:
                                    too_close = True
                                    break
                        else:
                            tc = cell_theta[ci]
                            for e in range(n_ex):
                                diff = abs(tc - ex_a[e])
                                ang_sep = min(diff, _TWO_PI - diff)
                                if ang_sep < exclusion_angle:
                                    too_close = True
                                    break
                        if too_close:
                            continue
                    ddx = cx - tx
                    ddy = cy - ty
                    ddz = cz - tz if is_3d else 0.0
                    cell_gap = max(np.sqrt(ddx*ddx + ddy*ddy + ddz*ddz) - r_target, 0.0)
                    if use_decay:
                        p_bridge = p_base * obstruction * np.exp(-cell_gap / decay_len)
                    else:
                        p_bridge = p_base * obstruction
                    if hash_u01(seed, ci, j, 3) < p_bridge:
                        cell_target[ci] = j
                        cell_bridge_age[ci] = 0.0
                        if is_3d:
                            ang_d, _te, _to = _ang_dist_3d(dpx, dpy, dpz, quat[i], cell_eta[ci], cell_omega[ci])
                        else:
                            ang_d, _tt, _sd = _ang_dist_2d(dpx, dpy, theta[i], cell_theta[ci])
                        if ang_d < commit_angle:
                            cell_state[ci] = BRIDGING
                            cell_align[ci] = align_min
                            maturity = min(1.0, dt / max(0.1, formation_time))
                            F_cell = min(Fc * maturity * align_min, F_max) * fac
                            cell_gap_prev[ci] = gap
                            cell_force_prev[ci] = F_cell
                            cell_fx[ci] += F_cell * nx
                            cell_fy[ci] += F_cell * ny
                            cell_fz[ci] += F_cell * nz
                            fx_acc += F_cell * nx
                            fy_acc += F_cell * ny
                            fz_acc += F_cell * nz
                        else:
                            cell_state[ci] = MIGRATING
                            cell_align[ci] = 0.0
        F[i, 0] += fx_acc
        F[i, 1] += fy_acc
        if is_3d:
            F[i, 2] += fz_acc


# ──────────────────────────────────────────────────────────────────────
# drivers
# ──────────────────────────────────────────────────────────────────────

def step_seed(rng):
    """One 62-bit draw from the run's Generator (0 when no Generator is supplied)."""
    if rng is None:
        return 0
    return int(rng.integers(0, 2**62))


def _is_3d_like(p):
    return p.mode in ("3D", "2D-slice")


def force_model_code(name):
    """'constant' -> 0, 'hill' -> 1 (V3.1 contractile bridge)."""
    return 1 if str(name).lower() == 'hill' else 0


def _law_rule_kappa(p):
    law = law_code(p.traction_f_law)
    rule = rule_code(p.traction_f_rule)
    kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
    return law, rule, kappa


def update_cell_state_k(gs, p, t, rng=None):
    """Drop-in for ``reference.update_cell_state`` on the compiled path."""
    N = gs.N
    a_cell = p.cell_diameter / 2.0
    k_opt = p.n_clutches * p.k_clutch
    is_3d_like = _is_3d_like(p)
    foothold = float(getattr(p, 'cell_capacity_foothold', 1.0))
    aggregates_k(float(t), gs.adhesive_mask, gs.n_cells, gs.E_gran, gs.nu_gran, gs.r, gs.activity,
                 gs.n_attached, gs.spread_fraction, gs.fa_maturity, gs.n_overcrowded,
                 float(a_cell), float(k_opt), float(p.t_spread_duration), float(p.t_attach_onset),
                 float(p.t_attach_half), float(p.fa_maturation_rate), float(p.cell_diameter),
                 float(p.cell_height_spread), float(capacity_coverage(p)), float(p.cell_coverage),
                 bool(is_3d_like), gs.cell_clock_offset, foothold)
    # V3.1: cell division (before the per-cell pass; the cell arrays may grow)
    # V3.2: the cycle clock has to be advanced first -- nothing did that before.
    from gels.division import age_cells, divide_cells
    age_cells(gs, p)
    divide_cells(gs, p, rng, t)
    if gs.total_cells == 0:
        return
    law, rule, kappa = _law_rule_kappa(p)
    is_3d = bool(gs.is_3d)
    periodic = (p.boundary_mode == 'periodic')
    quat = gs.quat if (is_3d and gs.quat is not None) else np.zeros((N, 4))
    seed = step_seed(rng)
    cells_update_k(seed, rng is not None, float(p.dt), is_3d, periodic,
                   float(p.Lx), float(p.Ly), float(p.Lz),
                   gs.adhesive_mask, gs.cell_offset, gs.n_attached, gs.n_overcrowded, gs.spread_fraction,
                   gs.fa_maturity, gs.r, gs.r_bound, gs.f, gs.pos, gs.theta, quat, gs.activity,
                   gs.cell_state, gs.cell_bridge_target, gs.cell_bridge_age, gs.cell_alignment,
                   gs.cell_bridge_locked, gs.cell_fx, gs.cell_fy, gs.cell_fz, gs.cell_contact_area,
                   gs.cell_overcrowd_age, gs.cell_theta_local, gs.cell_eta_local, gs.cell_omega_local,
                   float(p.cell_diameter), float(p.cell_height_spread), float(capacity_coverage(p)),
                   float(p.cell_coverage), bool(is_3d_like), foothold, float(p.cell_stacking_max),
                   float(p.overcrowd_senescence_time), float(p.bridge_alignment_rate),
                   float(p.bridge_alignment_min), float(p.bridge_lock_force_threshold),
                   bool(p.bridge_lock_scales_with_ligand), law, rule, float(p.traction_exponent), float(kappa),
                   float(p.bridge_senescence_time), float(p.cell_sense_distance), float(p.bridge_commit_angle),
                   float(p.cell_migration_speed), float(p.bridge_directed_speed_mult))


def bridging_k(gs, p, rng, F, pair_i, pair_j, rec, pos, dim, csr):
    """Compiled replacement for ``gels.kernels.bridging.bridging_pass`` (same signature + csr)."""
    off, nbr_pair, nbr_side = csr
    hit = rec['hit']
    adh = gs.adhesive_mask
    n_cells = gs.n_cells
    cand_mask = (hit != -1) & adh[pair_i] & adh[pair_j] & ((n_cells[pair_i] + n_cells[pair_j]) > 0)
    cidx = np.nonzero(cand_mask)[0].astype(np.int64)
    Mc = cidx.shape[0]
    if Mc == 0 or gs.total_cells == 0:
        return 0
    N = gs.N
    is_3d = (dim == 3)
    periodic = (p.boundary_mode == 'periodic')
    law, rule, kappa = _law_rule_kappa(p)
    F_stall = p.n_motors * p.F_motor_stall
    a_cell = p.cell_diameter / 2.0
    k_opt = p.n_clutches * p.k_clutch
    engagement = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)
    dz = rec['dz'] if is_3d else rec['dx']       # unused in 2D (never read)
    quat = gs.quat if (is_3d and gs.quat is not None) else np.zeros((N, 4))
    pos3 = np.ascontiguousarray(gs.pos[:N, :3])

    out_gap = np.empty(Mc)
    out_nx = np.empty(Mc)
    out_ny = np.empty(Mc)
    out_nz = np.empty(Mc)
    out_Fi = np.empty(Mc)
    out_Fj = np.empty(Mc)
    out_pf = np.empty(Mc)
    out_nexist = np.empty(Mc, dtype=np.int64)
    out_fac = np.empty(Mc)
    model = force_model_code(getattr(p, 'bridge_force_model', 'constant'))
    bridge_pairs_k(cidx, pair_i, pair_j, hit, rec['overlap'], rec['d'], rec['dx'], rec['dy'], dz,
                   rec['nx'], rec['ny'], is_3d, bool(gs.is_circle), periodic,
                   float(p.Lx), float(p.Ly), float(p.Lz),
                   gs.r, gs.r_bound, gs.f, gs.E_gran, gs.nu_gran, gs.fa_maturity, pos3,
                   gs.cell_offset, gs.cell_state, gs.cell_bridge_target, off, nbr_pair, nbr_side,
                   law, rule, float(p.traction_exponent), float(kappa), float(F_stall), float(a_cell),
                   float(k_opt), float(engagement), float(p.F_max_per_cell),
                   _adh_or_zeros(gs, p),                      # V3.6
                   float(p.bridge_break_gap), float(p.cell_sense_distance), float(p.bridge_contact_factor),
                   float(p.bridge_decay_length), float(p.bridge_inert_factor),
                   gs.cell_bridge_age, gs.cell_alignment, gs.cell_bridge_force, gs.cell_bridge_gap_prev,
                   int(model), float(p.dt), float(p.drag_scale),
                   float(p.cell_contraction_speed), float(p.bridge_eccentric_gain),
                   float(p.bridge_min_gap), float(p.bridge_formation_time), float(p.bridge_alignment_min),
                   out_gap, out_nx, out_ny, out_nz, out_Fi, out_Fj, out_pf, out_nexist, out_fac)

    cidx_of_pair = np.full(pair_i.shape[0], -1, dtype=np.int64)
    cidx_of_pair[cidx] = np.arange(Mc, dtype=np.int64)
    seed = step_seed(rng)
    bridge_cells_k(seed, float(p.dt), is_3d, periodic, float(p.Lx), float(p.Ly), float(p.Lz),
                   cidx_of_pair, pair_i, pair_j, off, nbr_pair, nbr_side, rec['dx'], rec['dy'], dz,
                   out_gap, out_nx, out_ny, out_nz, out_Fi, out_Fj, out_pf, out_nexist, out_fac,
                   gs.cell_bridge_force, gs.cell_bridge_gap_prev,
                   pos3, gs.r, gs.r_bound, gs.a, gs.b, gs.c, gs.n1, gs.n2, gs.theta, quat,
                   gs.fa_maturity, gs.adhesive_mask,
                   gs.cell_offset, gs.cell_state, gs.cell_bridge_target, gs.cell_bridge_age,
                   gs.cell_alignment, gs.cell_bridge_locked, gs.cell_theta_local, gs.cell_eta_local,
                   gs.cell_omega_local, gs.cell_fx, gs.cell_fy, gs.cell_fz, F,
                   float(p.bridge_break_gap), float(p.bridge_formation_time), float(p.bridge_alignment_min),
                   float(p.F_max_per_cell), float(p.cell_sense_distance), float(p.bridge_attempt_rate),
                   float(p.bridge_secondary_rate_mult), float(p.bridge_decay_length),
                   float(p.min_fa_for_bridge), float(p.bridge_exclusion_angle), float(p.bridge_commit_angle))
    return int(Mc)


__all__ = ['update_cell_state_k', 'bridging_k', 'hash_u01', 'hash_normal2', 'mc_force_k', 'step_seed']
