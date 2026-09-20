"""
Compiled metrics (V3.0 Phase 6d).
================================

Same keys, same values as ``reference.compute_metrics``. The per-pair Python
loops (contact statistics, bridge count) become sequential compiled loops
over the same ``cKDTree`` pairs in the same order, so every sum accumulates
in the reference order; cluster sizes use ``np.bincount`` instead of one
full-grid comparison per label; the Voronoi compartment query runs on all
cores; superellipse areas / perimeters / superellipsoid volumes are computed
once per ``GranuleSystem`` (rigid granules) and reused at every save.
"""

import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree

from gels.engine import (
    CellState, bed_metrics, boundary_geometry, collagen_field, domain_volume,
    energy_metrics, kozeny_carman_permeability, overlap_solid_volume, projection_metrics,
    substep_metrics, traction_metrics,
    rcp_fraction_superellipsoid,
    species_view, superellipse_area, superellipse_perimeter, superellipsoid_volume,
)
from gels.kernels import njit
from gels.kernels.geometry2d import se2d_contact_k
from gels.kernels.percolation import graph_percolation_metrics
from gels.laguerre import laguerre_metrics
from gels.pore import pore_metrics


# ──────────────────────────────────────────────────────────────────────
# compiled leaves
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def lens_area_k(R1, R2, d):
    """Area of the lens-shaped intersection of two circles (reference formula)."""
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return np.pi * min(R1, R2)**2
    cos_a1 = min(max((d*d + R1*R1 - R2*R2) / (2.0 * d * R1), -1.0), 1.0)
    cos_a2 = min(max((d*d + R2*R2 - R1*R1) / (2.0 * d * R2), -1.0), 1.0)
    arg = (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)
    return R1*R1 * np.arccos(cos_a1) + R2*R2 * np.arccos(cos_a2) - 0.5 * np.sqrt(max(0.0, arg))


@njit(cache=True)
def lens_volume_k(R1, R2, d):
    """Volume of the lens-shaped intersection of two spheres (reference formula)."""
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    if d < 1e-12:
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    s = R1 + R2 - d
    numer = s * s * (d * d + 2.0 * d * (R1 + R2) - 3.0 * (R1 - R2)**2)
    return np.pi * numer / (12.0 * d)


@njit(cache=True)
def contact_stats_k(pos, r, r_bound, a, b, n_shape, theta, gtype, sid, f_sp, K,
                    pair_i, pair_j, is_3d, is_circle, periodic, Lx, Ly, Lz,
                    out_is_contact):
    """Contact counts, species pair matrix, overlap statistics — in pair order.

    ``out_is_contact`` (V3.3) receives the per-pair contact flag, so the graph
    percolation in ``gels.kernels.percolation`` uses the engine's OWN contact
    predicate rather than a second definition of "touching".
    """
    n_contacts = 0
    n_ff = 0
    n_if = 0
    n_ii = 0
    max_ratio = 0.0
    total_overlap = 0.0
    f_prod_sum = 0.0
    sp_pair = np.zeros((K, K), dtype=np.int64)
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = 0.0
        if is_3d:
            dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if is_3d:
                dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz)

        if is_circle:
            overlap = r[i] + r[j] - d
        elif is_3d:
            overlap = r_bound[i] + r_bound[j] - d
        else:
            # reference quirk kept: the solver sees the real position of j, not its image
            ok, delta, snx, sny, scx, scy, R_li, R_lj = se2d_contact_k(
                pos[i, 0], pos[i, 1], a[i], b[i], n_shape[i], theta[i],
                pos[j, 0], pos[j, 1], a[j], b[j], n_shape[j], theta[j])
            overlap = delta if ok else 0.0

        if overlap > 0:
            out_is_contact[k] = 1
            n_contacts += 1
            ti = gtype[i]
            tj = gtype[j]
            if ti == 0 and tj == 0:
                n_ff += 1
            elif ti == 1 and tj == 1:
                n_ii += 1
            else:
                n_if += 1
            si = sid[i]
            sj = sid[j]
            if si <= sj:
                sp_pair[si, sj] += 1
            else:
                sp_pair[sj, si] += 1
            f_prod_sum += f_sp[si] * f_sp[sj]
            R_min = min(r[i], r[j])
            ratio = overlap / R_min
            if ratio > max_ratio:
                max_ratio = ratio
            if is_3d:
                if is_circle:
                    total_overlap += lens_volume_k(r[i], r[j], d)
                else:
                    R_eff = r[i] * r[j] / (r[i] + r[j])
                    total_overlap += (4.0/3.0) * np.pi * R_eff * overlap**2
            else:
                if is_circle:
                    total_overlap += lens_area_k(r[i], r[j], d)
                else:
                    R_eff = r[i] * r[j] / (r[i] + r[j])
                    total_overlap += np.pi * R_eff * overlap
    return n_contacts, n_ff, n_if, n_ii, sp_pair, f_prod_sum, max_ratio, total_overlap


@njit(cache=True)
def bridge_count_k(pos, r, r_bound, gtype, n_attached, n_overcrowded, pair_i, pair_j,
                   is_3d, is_circle, periodic, Lx, Ly, Lz, sense):
    """Functional pairs with available cells whose gap lies in (0, sense)."""
    n = 0
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        if gtype[i] != 0 or gtype[j] != 0:
            continue
        n_avail_i = max(0.0, n_attached[i] - n_overcrowded[i])
        n_avail_j = max(0.0, n_attached[j] - n_overcrowded[j])
        if n_avail_i < 0.1 or n_avail_j < 0.1:
            continue
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        dz = 0.0
        if is_3d:
            dz = pos[j, 2] - pos[i, 2]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
            if is_3d:
                dz = dz - Lz * np.rint(dz / Lz)
        d = np.sqrt(dx*dx + dy*dy + dz*dz)
        if is_circle:
            gap = d - r[i] - r[j]
        else:
            gap = d - r_bound[i] - r_bound[j]
            if gap < 0:
                gap = 0.0
        if gap > 0 and gap < sense:
            n += 1
    return n


# ──────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────

def _connectivity_full(field, thresh_frac=0.3):
    thr = np.mean(field) + thresh_frac * np.std(field)
    b = (field > thr).astype(int)
    lab, nc = label(b)
    if nc == 0 or b.sum() == 0:
        return 0, 0.0, 0.0, nc, None
    sz = np.bincount(lab.ravel(), minlength=nc + 1)[1:]
    return nc, float(sz.max() / b.sum()), float(b.sum() / b.size), nc, sz


def connectivity(field, thresh_frac=0.3):
    """Cluster analysis on a thresholded field (bincount cluster sizes; same values)."""
    nc, lf, cov, _nc, _sz = _connectivity_full(field, thresh_frac)
    return nc, lf, cov


def shape_cache(gs):
    """Per-granule areas / perimeters (2D) or volumes (3D) of the rigid shapes, computed once."""
    cache = getattr(gs, '_shape_cache', None)
    if cache is not None and cache['N'] == gs.N and cache['id'] == (id(gs.a), id(gs.b)):
        return cache
    N = gs.N
    if gs.is_3d:
        vols = np.array([superellipsoid_volume(gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                         for j in range(N)], dtype=np.float64)
        cache = {'N': N, 'id': (id(gs.a), id(gs.b)), 'vols': vols}
    else:
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i]) for i in range(N)],
                         dtype=np.float64)
        perims = np.array([superellipse_perimeter(gs.a[i], gs.b[i], gs.n_shape[i]) for i in range(N)],
                          dtype=np.float64)
        cache = {'N': N, 'id': (id(gs.a), id(gs.b)), 'areas': areas, 'perims': perims}
    gs._shape_cache = cache
    return cache


def _seq_sum(values):
    """Python-order sequential sum (matches the reference's sum(generator))."""
    return float(sum(values))


def _tree(gs, p, pos):
    periodic = (p.boundary_mode == 'periodic')
    if periodic:
        if gs.is_3d:
            return cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        return cKDTree(pos, boxsize=[p.Lx, p.Ly])
    return cKDTree(pos)


def _pairs(tree, cutoff):
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return (np.ascontiguousarray(pairs[:, 0], dtype=np.int32),
            np.ascontiguousarray(pairs[:, 1], dtype=np.int32))


# ──────────────────────────────────────────────────────────────────────
# metrics
# ──────────────────────────────────────────────────────────────────────

def compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces, phi_s=None):
    """Drop-in for ``reference.compute_metrics`` (same keys and values)."""
    m = dict(time=t)
    periodic = (p.boundary_mode == 'periodic')
    N = gs.N

    # ── Boundary exclusion ──
    bx = 0.0 if periodic else p.boundary_exclusion
    if bx > 0:
        shape = phi_f.shape
        lo = [int(bx * s) for s in shape]
        hi = [int((1.0 - bx) * s) for s in shape]
        if phi_f.ndim == 3:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        else:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        pf_inner = phi_f[inner]
        pi_inner = phi_i[inner]
        pv_inner = phi_v[inner]
        inner_slice = inner
        x_lo = bx * p.Lx; x_hi = (1.0 - bx) * p.Lx
        y_lo = bx * p.Ly; y_hi = (1.0 - bx) * p.Ly
        inner_gran = (gs.x >= x_lo) & (gs.x <= x_hi) & (gs.y >= y_lo) & (gs.y <= y_hi)
        if gs.is_3d:
            z_lo = bx * p.Lz; z_hi = (1.0 - bx) * p.Lz
            inner_gran &= (gs.z >= z_lo) & (gs.z <= z_hi)
    else:
        pf_inner, pi_inner, pv_inner = phi_f, phi_i, phi_v
        inner_gran = np.ones(N, dtype=bool)
        inner_slice = None

    m['phi_f_mean'] = float(np.mean(pf_inner))
    m['phi_i_mean'] = float(np.mean(pi_inner))
    m['phi_v_mean'] = float(np.mean(pv_inner))
    m['boundary_exclusion'] = bx
    m['n_granules_inner'] = int(np.sum(inner_gran))
    # V3.1 cell population
    m['n_cells_total'] = int(gs.total_cells)
    m['n_live_cells'] = int(np.sum(gs.cell_state != int(CellState.SENESCENT)))
    m['n_divisions_cum'] = int(getattr(gs, 'n_divisions_cum', 0))
    from gels.division import cycle_metrics as _cycle_metrics   # local: avoids a circular import
    m.update(_cycle_metrics(gs))                 # V3.2 cell-cycle diagnostics
    m['n_boundary'] = int(np.sum(gs.fixed)) if hasattr(gs, 'fixed') else 0
    # V3.1 bed observables (positional: free of the rendering halo). Only for
    # open / cylindrical containers, where a bed surface exists at all.
    _geom = boundary_geometry(p)
    if _geom.top_free or _geom.shape_code == 1:
        m.update(bed_metrics(gs, p))

    fn, fl, fc, nc_f, sizes_f = _connectivity_full(pf_inner, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)
    vn, vl, vc = connectivity(pv_inner, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)
    inn, il, _ = connectivity(pi_inner, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    m['tissue_frac'] = float(np.mean(pf_inner > 0.5))

    # same threshold and labelling as the functional connectivity above: reuse it
    thr = np.mean(pf_inner) + 0.3*np.std(pf_inner)
    dxg = p.Lx / phi_f.shape[0]
    if nc_f > 0 and sizes_f is not None:
        m['func_max_area'] = float(np.max(sizes_f) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    fr = pf_inner > thr
    pt = pf_inner + pi_inner
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    if forces.shape[1] >= 3 and gs.is_3d:
        F_mag = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2 + forces[:, 2]**2)
    else:
        F_mag = np.sqrt(forces[:, 0]**2 + forces[:, 1]**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[gs.func_mask])) if np.any(gs.func_mask) else 0.0

    # ── Overlap & contact diagnostics (compiled, pair order preserved) ──
    pos = gs.positions()
    pos_c = np.ascontiguousarray(pos)
    max_rb = float(np.max(gs.r_bound))
    tree = _tree(gs, p, pos)
    pair_i, pair_j = _pairs(tree, 2 * max_rb)
    K_sp, sid_sp, adh_sp, f_sp = species_view(gs)
    sid_c = np.ascontiguousarray(sid_sp, dtype=np.int32)
    f_sp_c = np.ascontiguousarray(f_sp, dtype=np.float64)
    gtype_c = np.ascontiguousarray(gs.gtype, dtype=np.int64)
    Lz = float(getattr(p, 'Lz', 0.0))
    is_contact = np.zeros(pair_i.shape[0], dtype=np.uint8)
    (n_contacts, n_contacts_ff, n_contacts_if, n_contacts_ii, sp_pair, f_prod_sum,
     max_overlap_ratio, total_overlap_area) = contact_stats_k(
        pos_c, gs.r, gs.r_bound, gs.a, gs.b, gs.n_shape, gs.theta, gtype_c, sid_c, f_sp_c, int(K_sp),
        pair_i, pair_j, bool(gs.is_3d), bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly), Lz,
        is_contact)
    m.update(graph_percolation_metrics(gs, p, pos, pair_i, pair_j, is_contact, periodic))

    cutoff_bridge = 2 * max_rb + p.cell_sense_distance
    pair_bi, pair_bj = _pairs(tree, cutoff_bridge)
    n_bridges = int(bridge_count_k(
        pos_c, gs.r, gs.r_bound, gtype_c, gs.n_attached, gs.n_overcrowded, pair_bi, pair_bj,
        bool(gs.is_3d), bool(gs.is_circle), periodic, float(p.Lx), float(p.Ly), Lz,
        float(p.cell_sense_distance)))

    cache = shape_cache(gs) if not gs.is_circle else None
    if gs.is_3d:
        total_granule_vol = float(np.sum((4.0/3.0) * np.pi * gs.r**3))
        total_granule_area = total_granule_vol
    else:
        if gs.is_circle:
            total_granule_area = float(np.sum(np.pi * gs.r**2))
        else:
            total_granule_area = _seq_sum(cache['areas'])
    m['n_contacts'] = int(n_contacts)
    m['n_contacts_ff'] = int(n_contacts_ff)
    m['n_contacts_if'] = int(n_contacts_if)
    m['n_contacts_ii'] = int(n_contacts_ii)
    m['max_overlap_ratio'] = float(max_overlap_ratio)
    m['total_overlap_area'] = float(total_overlap_area)
    m['area_conservation'] = 1.0 - total_overlap_area / max(total_granule_area, 1e-30)
    if gs.is_3d:
        m['volume_conservation'] = m['area_conservation']
    m['n_bridges'] = n_bridges

    # ── Shape descriptors ──
    if not gs.is_circle:
        ar_arr = np.maximum(gs.a, gs.b) / np.minimum(gs.a, gs.b)
        elong_arr = 1.0 - np.minimum(gs.a, gs.b) / np.maximum(gs.a, gs.b)
        if gs.is_3d:
            areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i]) for i in range(N)])
            perims = np.array([superellipse_perimeter(gs.a[i], gs.b[i], gs.n_shape[i]) for i in range(N)])
        else:
            areas = cache['areas']
            perims = cache['perims']
        circularity = 4.0 * np.pi * areas / (perims**2 + 1e-30)
        m['shape_aspect_ratio_mean'] = float(np.mean(ar_arr))
        m['shape_aspect_ratio_std'] = float(np.std(ar_arr))
        m['shape_elongation_mean'] = float(np.mean(elong_arr))
        m['shape_circularity_mean'] = float(np.mean(circularity))
        m['shape_blockiness_mean'] = float(np.mean(gs.n_shape))

    # ── Cell state metrics ──
    func = gs.func_mask
    m['n_attached_total'] = float(np.sum(gs.n_attached[func]))
    m['n_seeded_total'] = float(np.sum(gs.n_cells[func]))
    m['mean_spread_frac'] = float(np.mean(gs.spread_fraction[func])) if np.any(func) else 0.0
    m['mean_fa_maturity'] = float(np.mean(gs.fa_maturity[func])) if np.any(func) else 0.0
    m['n_overcrowded_total'] = float(np.sum(gs.n_overcrowded[func]))

    migrating_mask = gs.cell_state == int(CellState.MIGRATING)
    m['n_migrating_cells'] = int(np.sum(migrating_mask))
    bridging_mask = gs.cell_state == int(CellState.BRIDGING)
    n_bridging_cells = int(np.sum(bridging_mask))
    m['n_bridging_cells'] = n_bridging_cells
    n_locked_in = 0
    if n_bridging_cells > 0:
        cell_F_mag = np.sqrt(gs.cell_fx**2 + gs.cell_fy**2 + gs.cell_fz**2)
        bridge_forces = cell_F_mag[bridging_mask]
        m['bridge_force_mean'] = float(np.mean(bridge_forces))
        m['bridge_force_max'] = float(np.max(bridge_forces))
        m['bridge_force_min'] = float(np.min(bridge_forces))
        m['bridge_force_std'] = float(np.std(bridge_forces))
        n_locked_in = int(np.sum(bridge_forces >= p.bridge_lock_force_threshold))
    else:
        m['bridge_force_mean'] = 0.0
        m['bridge_force_max'] = 0.0
        m['bridge_force_min'] = 0.0
        m['bridge_force_std'] = 0.0
    m['n_locked_in_cells'] = n_locked_in

    # ── Transport metrics ──
    porosity = float(np.mean(pv_inner))
    m['porosity'] = porosity
    inner_r = gs.r[inner_gran] if np.any(inner_gran) else gs.r
    d_grain = float(np.mean(2 * inner_r))
    m['d_grain_mean'] = d_grain
    m['K_kozeny_carman'] = float(kozeny_carman_permeability(porosity, d_grain))

    phi_solid = float(np.mean(pf_inner) + np.mean(pi_inner))
    m['phi_solid'] = phi_solid

    if gs.is_3d:
        V_domain = domain_volume(p, '3D')      # V3.1: cylinder-aware
        if gs.is_circle:
            V_f_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.func_mask] ** 3))
            V_i_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.inert_mask] ** 3))
        else:
            V_f_true = _seq_sum(cache['vols'][gs.func_mask])
            V_i_true = _seq_sum(cache['vols'][gs.inert_mask])
    else:
        V_domain = domain_volume(p, '2D')
        if gs.is_circle:
            V_f_true = float(np.sum(np.pi * gs.r[gs.func_mask] ** 2))
            V_i_true = float(np.sum(np.pi * gs.r[gs.inert_mask] ** 2))
        else:
            V_f_true = _seq_sum(cache['areas'][gs.func_mask])
            V_i_true = _seq_sum(cache['areas'][gs.inert_mask])
    m['phi_f_true'] = V_f_true / V_domain
    m['phi_i_true'] = V_i_true / V_domain
    m['phi_solid_true'] = (V_f_true + V_i_true) / V_domain
    # V3.2: overlap is counted twice in the sums above (1-4 % once a soft bed
    # compacts through interpenetration), so report the net value too.
    _V_lens = overlap_solid_volume(gs, mobile_only=False)
    m['overlap_volume'] = _V_lens
    m['overlap_volume_fraction'] = (_V_lens / (V_f_true + V_i_true)) if (V_f_true + V_i_true) > 0 else 0.0
    m['phi_solid_true_net'] = max(0.0, (V_f_true + V_i_true) - _V_lens) / V_domain
    m.update(projection_metrics(gs))            # V3.2: is the rail or the contact law in charge?
    m.update(energy_metrics(gs))                # V3.5: is the force law a gradient?
    m.update(substep_metrics(gs))               # V3.6: what the substep controller did
    m.update(traction_metrics(gs, p))           # V3.6: what the cells pull with, and store
    ar_mean = float(np.mean(np.maximum(gs.a, gs.b) /
                            np.minimum(gs.a, gs.b))) if N > 0 else 1.0
    phi_rcp = rcp_fraction_superellipsoid(ar_mean)
    m['phi_RCP'] = phi_rcp
    m['compaction_ratio'] = phi_solid / phi_rcp if phi_rcp > 0 else 0.0
    m.update(laguerre_metrics(gs, p, pos, phi_rcp))
    m.update(pore_metrics(gs, p, phi_f.shape, periodic))

    if gs.is_3d:
        L_char = domain_volume(p, '3D') ** (1.0/3.0)
    else:
        L_char = np.sqrt(p.Lx * p.Ly)
    m['Da_number'] = m['K_kozeny_carman'] / (L_char**2) if L_char > 0 else 0.0

    if gs.epsilon is not None:
        eps_mag = np.sqrt(np.sum(gs.epsilon**2, axis=1))
        m['def_strain_mean'] = float(np.mean(eps_mag))
        m['def_strain_max'] = float(np.max(eps_mag))
        m['def_strain_std'] = float(np.std(eps_mag))
        for alpha in range(gs.epsilon.shape[1]):
            m[f'def_mode_{alpha}_mean'] = float(np.mean(gs.epsilon[:, alpha]))
            m[f'def_mode_{alpha}_std'] = float(np.std(gs.epsilon[:, alpha]))

    # ── Two-compartment volume conservation (Voronoi shrink-wrap), all cores ──
    # perf_metrics_voronoi_stride > 1 samples every s-th grid point per axis
    # (the compartment fractions are estimates; stride 4 changes them by < 2e-3).
    if N > 0 and np.any(gs.func_mask) and np.any(gs.inert_mask):
        tree_vc = tree
        stride = max(1, int(getattr(p, 'perf_metrics_voronoi_stride', 1)))
        grid_shape = phi_f.shape
        sub = tuple(slice(None, None, stride) for _ in grid_shape)
        if gs.is_3d:
            nx, ny, nz = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)[::stride]
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)[::stride]
            cz = np.linspace(0.5 * p.Lz / nz, p.Lz - 0.5 * p.Lz / nz, nz)[::stride]
            gx_v, gy_v, gz_v = np.meshgrid(cx, cy, cz, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel(), gz_v.ravel()])
        else:
            nx, ny = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)[::stride]
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)[::stride]
            gx_v, gy_v = np.meshgrid(cx, cy, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel()])
        sub_shape = gx_v.shape
        _, nearest_idx = tree_vc.query(voxel_pts, workers=-1)
        nearest_type = gs.gtype[nearest_idx].reshape(sub_shape)
        fm = nearest_type == 0
        im = nearest_type == 1
        n_func_v = int(np.sum(fm))
        n_inert_v = int(np.sum(im))
        n_sub = max(int(np.prod(sub_shape)), 1)
        pv_sub = phi_v[sub]
        ps_sub = (phi_f + phi_i)[sub]
        m['x_f'] = n_func_v / n_sub
        m['x_i'] = n_inert_v / n_sub
        m['phi_v_in_func'] = float(np.mean(pv_sub[fm])) if n_func_v > 0 else 0.0
        m['phi_v_in_inert'] = float(np.mean(pv_sub[im])) if n_inert_v > 0 else 0.0
        m['phi_solid_func'] = float(np.mean(ps_sub[fm])) if n_func_v > 0 else 0.0
        m['phi_solid_inert'] = float(np.mean(ps_sub[im])) if n_inert_v > 0 else 0.0
        m['phi_v_crosscheck'] = m['x_f'] * m['phi_v_in_func'] + m['x_i'] * m['phi_v_in_inert']

    # ── per-species metrics ──
    counts_sp = np.bincount(sid_sp, minlength=K_sp)
    if gs.total_cells > 0:
        bridging = np.asarray(gs.cell_state) == int(CellState.BRIDGING)
        bridging_sp = np.bincount(sid_sp[np.asarray(gs.cell_granule_id)[bridging]], minlength=K_sp)
    else:
        bridging_sp = np.zeros(K_sp, dtype=np.int64)
    phi_c_true = 0.0
    for k in range(K_sp):
        mk = sid_sp == k
        m[f'n_gran_sp_{k}'] = int(counts_sp[k])
        m[f'n_cells_sp_{k}'] = int(np.sum(gs.n_cells[mk])) if np.any(mk) else 0
        m[f'n_bridging_sp_{k}'] = int(bridging_sp[k])
        if not np.any(mk):
            V_k = 0.0
        elif gs.is_3d:
            V_k = (float(np.sum((4.0 / 3.0) * np.pi * gs.r[mk] ** 3)) if gs.is_circle else
                   _seq_sum(cache['vols'][mk]))
        else:
            V_k = (float(np.sum(np.pi * gs.r[mk] ** 2)) if gs.is_circle else
                   _seq_sum(cache['areas'][mk]))
        m[f'phi_sp_{k}_true'] = V_k / V_domain
        phi_c_true += f_sp[k] * m[f'phi_sp_{k}_true']
        endpoints = int(np.sum(sp_pair[k, :]) + np.sum(sp_pair[:, k]))
        m[f'Z_sp_{k}'] = endpoints / counts_sp[k] if counts_sp[k] > 0 else 0.0
        if phi_s is not None:
            pk = phi_s[k][inner_slice] if inner_slice is not None else phi_s[k]
            m[f'phi_sp_{k}_mean'] = float(np.mean(pk))
    for a_ in range(K_sp):
        for b_ in range(a_, K_sp):
            m[f'n_contacts_sp_{a_}_{b_}'] = int(sp_pair[a_, b_])
    m['phi_c_true'] = float(phi_c_true)
    m['f_solid_mean'] = float(phi_c_true / m['phi_solid_true']) if m['phi_solid_true'] > 0 else 0.0
    m['contact_ff_weight'] = float(f_prod_sum / n_contacts) if n_contacts > 0 else 0.0
    if phi_s is not None:
        phi_c = collagen_field(gs, phi_s)
        pc_inner = phi_c[inner_slice] if inner_slice is not None else phi_c
        m['phi_c_mean'] = float(np.mean(pc_inner))

    return m


__all__ = ['compute_metrics', 'connectivity', 'shape_cache', 'contact_stats_k', 'bridge_count_k',
           'lens_area_k', 'lens_volume_k']
