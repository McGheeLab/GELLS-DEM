"""
Cell bridging over the kernel's candidate pairs (Python; V3.0 Phase 6a/6c-1).
============================================================================

The reference force loops interleave cell bridging with contact mechanics
inside one Python loop over neighbour pairs. With the contact mechanics
compiled, bridging runs here afterwards over the *same* candidate pairs
(adhesive on both sides, cells present, centres not coincident) in the
*same* pair order, calling the reference's ``_service_committed_bridges``
and ``_attempt_new_bridges`` with identical arguments. The cell state
machine and the random stream are therefore unchanged; only the order in
which forces are summed into ``F`` differs.

The bridge-path ray-cast is the restricted, exact form: a granule can only
block the line of sight between two bridging candidates if it lies within
its bounding radius of the segment joining them, and every such granule is
within ``2·max(r_bound) + L_max`` of one of the two — i.e. inside the
neighbour lists the contact kernel already built (``L_max ≥
cell_sense_distance``). Testing N(i) ∪ N(j) instead of all N granules gives
the same blocker set, hence the same factor, at O(Z) instead of O(N) per pair.

This module is the interim until the compiled bridging passes (B1 pair
pass, B2 per-granule pass, hashed RNG) land.
"""

import numpy as np

from gels.engine import (adhesion_force_ceiling, motor_clutch_force,
                         stacked_traction_force, stacking_enabled)
from gels.materials import blocker_factor, law_code, rule_code, traction_gain


def _partners_csr(pair_i, pair_j, off, nbr_pair, nbr_side):
    """Neighbour index of every incidence in the particle CSR."""
    if nbr_pair.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    return np.where(nbr_side == 0, pair_j[nbr_pair], pair_i[nbr_pair]).astype(np.int64)


def classify_path_restricted(gs, p, gi, gj, pos_i, pos_j, gap, cand, all_pos, Ls):
    """``reference._classify_bridge_path`` evaluated over the candidate blockers ``cand`` only.

    ``cand`` must contain every granule within its bounding radius of the
    segment gi→gj (N(i) ∪ N(j) does); gi and gj themselves are ignored.
    """
    if gap <= 0:
        return p.bridge_contact_factor

    void_decay = np.exp(-gap / max(p.bridge_decay_length, 0.1))

    d_vec = pos_j - pos_i
    d_mag = np.linalg.norm(d_vec)
    if d_mag < 1e-12:
        return p.bridge_contact_factor * void_decay

    if cand.shape[0] == 0:
        return void_decay

    d_hat = d_vec / d_mag
    ri = gs.r_bound[gi]
    rj = gs.r_bound[gj]

    dk = all_pos[cand] - pos_i
    if Ls is not None:
        dk -= Ls * np.round(dk / Ls)

    t = dk @ d_hat
    valid = (t > ri * 0.5) & (t < d_mag - rj * 0.5)
    valid &= (cand != gi) & (cand != gj)
    if not np.any(valid):
        return void_decay

    proj = np.outer(t[valid], d_hat)
    perp = dk[valid] - proj
    d_perp = np.sqrt(np.sum(perp * perp, axis=1))

    r_valid = gs.r_bound[cand][valid]
    intersects = d_perp < r_valid
    if not np.any(intersects):
        return void_decay

    f_block = gs.f[cand][valid][intersects]
    factor = min(blocker_factor(float(fk), p.bridge_inert_factor) for fk in f_block)
    return factor * void_decay


def bridging_pass(gs, p, rng, F, pair_i, pair_j, rec, pos, dim, csr=None):
    """Service and attempt bridges for every candidate pair, in pair order.

    ``csr = (off, nbr_pair, nbr_side)`` from ``neighbors.csr_from_pairs`` enables
    the restricted ray-cast; without it the reference's all-granule scan is used.
    """
    from gels.kernels import reference as _ref

    hit = rec['hit']
    adh = gs.adhesive_mask
    n_cells = gs.n_cells
    cand_mask = (hit != -1) & adh[pair_i] & adh[pair_j] & ((n_cells[pair_i] + n_cells[pair_j]) > 0)
    idx = np.nonzero(cand_mask)[0]
    if idx.size == 0:
        return 0

    law = law_code(p.traction_f_law)
    rule = rule_code(p.traction_f_rule)
    kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
    # V3.6: this hybrid path has to apply the adhesion ceiling and the cell-on-cell
    # substrate too, or `perf_cells_backend='python'` would silently be a third
    # force law rather than the exact V2.7 cell machinery on compiled contacts.
    _adh = adhesion_force_ceiling(gs, p)
    _stack = stacking_enabled(p)
    gamma = p.traction_exponent
    sense = p.cell_sense_distance
    periodic = (p.boundary_mode == 'periodic')
    Ls = np.array([p.Lx, p.Ly, p.Lz][:dim]) if periodic else None

    r = gs.r
    rb = gs.r_bound
    overlap = rec['overlap']
    d_rec = rec['d']
    dx = rec['dx']
    dy = rec['dy']
    dz = rec['dz'] if dim == 3 else None
    nx_rec = rec['nx']
    ny_rec = rec['ny']
    is_circle = gs.is_circle

    if csr is not None:
        off, nbr_pair, nbr_side = csr
        partner = _partners_csr(pair_i, pair_j, off, nbr_pair, nbr_side)
    else:
        off = partner = None
    all_pos = pos          # (N, dim) contiguous copy made by the caller

    for k in idx:
        i = int(pair_i[k])
        j = int(pair_j[k])
        in_contact = hit[k] == 1
        if dim == 2:
            # 2D: bridging uses the contact normal (the solver's when it hit)
            nx = nx_rec[k]
            ny = ny_rec[k]
            nz = 0.0
            d = d_rec[k]
            if is_circle:
                gap = d - r[i] - r[j]
            else:
                if in_contact:
                    gap = -overlap[k]
                else:
                    gap = d - rb[i] - rb[j]
                    if gap < 0:
                        gap = 0.0
            pos_i = pos[i]
            pos_j = pos_i + np.array([dx[k], dy[k]])
        else:
            # 3D: bridging uses the centre-line normal nv = dp / |dp| (reference)
            dp = np.array([dx[k], dy[k], dz[k]])
            d = np.linalg.norm(dp)
            nv = dp / d
            nx, ny, nz = nv[0], nv[1], nv[2]
            if in_contact:
                gap = -overlap[k]
            else:
                gap = d - rb[i] - rb[j]
                if gap < 0:
                    gap = 0.0
            pos_i = pos[i]
            pos_j = pos_i + dp

        _g_i = traction_gain(gs.f[i], gs.f[j], law, rule, gamma, kappa)
        _g_j = traction_gain(gs.f[j], gs.f[i], law, rule, gamma, kappa)
        F_cell_i = motor_clutch_force(
            gs.E_gran[i], p, gs.fa_maturity[i], nu=gs.nu_gran[i], g=_g_i,
            F_adh=(_adh[i] if _adh is not None else None))
        F_cell_j = motor_clutch_force(
            gs.E_gran[j], p, gs.fa_maturity[j], nu=gs.nu_gran[j], g=_g_j,
            F_adh=(_adh[j] if _adh is not None else None))
        F_up_i = F_up_j = None
        if _stack:
            F_up_i = stacked_traction_force(
                p, gs.fa_maturity[i], _g_i,
                F_adh=(_adh[i] if _adh is not None else None))
            F_up_j = stacked_traction_force(
                p, gs.fa_maturity[j], _g_j,
                F_adh=(_adh[j] if _adh is not None else None))

        n_ci, n_cj, _fac = _ref._service_committed_bridges(
            gs, i, j, gap, p, F_cell_i, F_cell_j, nx, ny, nz, F,
            F_up_i=F_up_i, F_up_j=F_up_j)
        n_existing = n_ci + n_cj

        if gap < sense:
            if off is not None:
                cand = np.concatenate([partner[off[i]:off[i + 1]], partner[off[j]:off[j + 1]]])
                pf = classify_path_restricted(gs, p, i, j, pos_i, pos_j, gap, cand, all_pos, Ls)
            else:
                pf = _ref._classify_bridge_path(gs, p, i, j, pos_i, pos_j, gap)
            _ref._attempt_new_bridges(
                gs, i, j, gap, p, rng, F_cell_i, F_cell_j, nx, ny, nz, F,
                F_up_i=F_up_i, F_up_j=F_up_j, pair_fac=_fac,
                n_existing_bridges=n_existing, path_factor=pf,
                pos_i=pos_i, pos_j=pos_j)
    return int(idx.size)


__all__ = ['bridging_pass', 'classify_path_restricted']
