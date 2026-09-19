"""
Compiled phase-field rendering (V3.0 Phase 6d).
==============================================

Bounding-box stamps, parallel over grid rows (2D) or x-slabs (3D) with a
per-slab list of granule images, so every pixel accumulates its granules in
a fixed order (thread-count independent). Effective radii come from a
compiled pass over the same ``cKDTree`` pairs as the reference, in the same
order. The volume-conserving normalisation is the reference's numpy code.

2D: the reference stamps every granule over the whole grid; here each tanh
profile is truncated at 8 interface widths, where it is below 1e-7 — fields
agree to ~1e-6. 3D: same 3w bounding box and image criterion as the
reference. LS-DEM (deformed) rendering stays on the reference path.
"""

import numpy as np
from scipy.spatial import cKDTree

from gels.engine import quat_to_rotation_matrix, species_view
from gels.kernels import njit, prange
from gels.kernels.metrics import lens_area_k, lens_volume_k, shape_cache
from gels.kernels.reference import _normalize_species_fields


# ──────────────────────────────────────────────────────────────────────
# effective radii (volume-conserving display radii)
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _overlap_area_2d(pos, r, pair_i, pair_j, periodic, Lx, Ly):
    N = r.shape[0]
    out = np.zeros(N)
    for k in range(pair_i.shape[0]):
        i = pair_i[k]
        j = pair_j[k]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if periodic:
            dx = dx - Lx * np.rint(dx / Lx)
            dy = dy - Ly * np.rint(dy / Ly)
        d = np.sqrt(dx*dx + dy*dy)
        if d < r[i] + r[j]:
            A_lens = lens_area_k(r[i], r[j], d)
            frac_i = r[i]**2 / (r[i]**2 + r[j]**2)
            out[i] += frac_i * A_lens
            out[j] += (1.0 - frac_i) * A_lens
    return out


@njit(cache=True)
def _overlap_volume_3d(pos, r, pair_i, pair_j, periodic, Lx, Ly, Lz):
    N = r.shape[0]
    out = np.zeros(N)
    for k in range(pair_i.shape[0]):
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
        if d < r[i] + r[j]:
            V_lens = lens_volume_k(r[i], r[j], d)
            frac_i = r[i]**3 / (r[i]**3 + r[j]**3)
            out[i] += frac_i * V_lens
            out[j] += (1.0 - frac_i) * V_lens
    return out


def _tree_pairs(pos, cutoff, periodic, box):
    d = pos.shape[1]
    tree = cKDTree(pos, boxsize=[float(b) for b in box[:d]]) if periodic else cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return (np.ascontiguousarray(pairs[:, 0], dtype=np.int32),
            np.ascontiguousarray(pairs[:, 1], dtype=np.int32))


def effective_radii_2d(gs, p=None):
    """Compiled twin of ``engine.compute_effective_radii`` → (r_eff, overlap_area)."""
    N = gs.N
    pos = np.ascontiguousarray(gs.pos[:N, :2])
    max_r = float(np.max(gs.r))
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    box = (p.Lx, p.Ly) if p is not None else (0.0, 0.0)
    pair_i, pair_j = _tree_pairs(pos, 2 * max_r, periodic, box)
    overlap_area = _overlap_area_2d(pos, gs.r, pair_i, pair_j, periodic, float(box[0]), float(box[1]))
    r_eff = np.sqrt(gs.r**2 + overlap_area / np.pi)
    return r_eff, overlap_area


def effective_radii_3d(gs, p=None):
    """Compiled twin of ``engine.compute_effective_radii_3d`` → (r_eff, overlap_volume)."""
    N = gs.N
    pos = np.ascontiguousarray(gs.pos[:N, :3])
    max_r = float(np.max(gs.r))
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    box = (p.Lx, p.Ly, p.Lz) if p is not None else (0.0, 0.0, 0.0)
    pair_i, pair_j = _tree_pairs(pos, 2 * max_r, periodic, box)
    overlap_vol = _overlap_volume_3d(pos, gs.r, pair_i, pair_j, periodic,
                                     float(box[0]), float(box[1]), float(box[2]))
    r_eff = (gs.r**3 + 3.0 * overlap_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return r_eff, overlap_vol


# ──────────────────────────────────────────────────────────────────────
# clip planes and periodic images
# ──────────────────────────────────────────────────────────────────────

def clips_from_lists(gs, dim):
    """Fixed-width arrays (clip_n (N, C, dim), clip_d (N, C), clip_cnt (N,)) from gs.contact_clips."""
    N = gs.N
    arrays = getattr(gs, 'clip_arrays', None)
    if arrays is not None and arrays[0].shape[0] == N and arrays[0].shape[2] == dim:
        return arrays
    lists = getattr(gs, 'contact_clips', None)
    if not lists:
        return np.zeros((N, 1, dim)), np.zeros((N, 1)), np.zeros(N, dtype=np.int32)
    counts = np.fromiter((len(c) for c in lists), dtype=np.int32, count=N)
    maxc = max(1, int(counts.max()))
    clip_n = np.zeros((N, maxc, dim))
    clip_d = np.zeros((N, maxc))
    for i in np.nonzero(counts)[0]:
        arr = np.asarray(lists[i], dtype=np.float64).reshape(int(counts[i]), dim + 1)
        clip_n[i, :counts[i], :] = arr[:, :dim]
        clip_d[i, :counts[i]] = arr[:, dim]
    return clip_n, clip_d, counts


def _images_2d(x, y, rb, Lx, Ly, periodic):
    """Granule images to stamp: every granule once, plus ghosts whose box touches the domain."""
    N = x.shape[0]
    idx = [np.arange(N, dtype=np.int32)]
    cxs = [x]
    cys = [y]
    if periodic:
        for sx in (-Lx, 0.0, Lx):
            for sy in (-Ly, 0.0, Ly):
                if sx == 0.0 and sy == 0.0:
                    continue
                gx = x + sx
                gy = y + sy
                m = (-rb < gx) & (gx < Lx + rb) & (-rb < gy) & (gy < Ly + rb)
                if np.any(m):
                    idx.append(np.nonzero(m)[0].astype(np.int32))
                    cxs.append(gx[m])
                    cys.append(gy[m])
    return (np.ascontiguousarray(np.concatenate(idx)), np.ascontiguousarray(np.concatenate(cxs)),
            np.ascontiguousarray(np.concatenate(cys)))


def _images_3d(x, y, z, rb, Lx, Ly, Lz, periodic):
    """Reference criterion: gx + rb > 0 and gx - rb < L on every axis (base image included)."""
    N = x.shape[0]
    if not periodic:
        return (np.arange(N, dtype=np.int32), np.ascontiguousarray(x), np.ascontiguousarray(y),
                np.ascontiguousarray(z))
    idx, cxs, cys, czs = [], [], [], []
    for sx in (-Lx, 0.0, Lx):
        for sy in (-Ly, 0.0, Ly):
            for sz in (-Lz, 0.0, Lz):
                gx = x + sx
                gy = y + sy
                gz = z + sz
                m = ((gx + rb > 0) & (gx - rb < Lx) & (gy + rb > 0) & (gy - rb < Ly)
                     & (gz + rb > 0) & (gz - rb < Lz))
                if np.any(m):
                    idx.append(np.nonzero(m)[0].astype(np.int32))
                    cxs.append(gx[m])
                    cys.append(gy[m])
                    czs.append(gz[m])
    return (np.ascontiguousarray(np.concatenate(idx)), np.ascontiguousarray(np.concatenate(cxs)),
            np.ascontiguousarray(np.concatenate(cys)), np.ascontiguousarray(np.concatenate(czs)))


@njit(cache=True)
def _slab_index_2d(img_cx, img_rb, dx, Ng):
    """Rows touched by each image (generous box) → CSR row → images."""
    M = img_cx.shape[0]
    ix0 = np.empty(M, dtype=np.int64)
    ix1 = np.empty(M, dtype=np.int64)
    count = np.zeros(Ng, dtype=np.int64)
    for m in range(M):
        lo = int(np.floor((img_cx[m] - img_rb[m]) / dx - 0.5)) - 1
        hi = int(np.ceil((img_cx[m] + img_rb[m]) / dx - 0.5)) + 2
        if lo < 0:
            lo = 0
        if hi > Ng:
            hi = Ng
        if hi < lo:
            hi = lo
        ix0[m] = lo
        ix1[m] = hi
        for ix in range(lo, hi):
            count[ix] += 1
    off = np.zeros(Ng + 1, dtype=np.int64)
    for ix in range(Ng):
        off[ix + 1] = off[ix] + count[ix]
    fill = off[:Ng].copy()
    slab = np.empty(off[Ng], dtype=np.int32)
    for m in range(M):
        for ix in range(ix0[m], ix1[m]):
            slab[fill[ix]] = m
            fill[ix] += 1
    return off, slab


@njit(cache=True)
def _slab_index_3d(img_cx, img_rb, dx, Ng):
    """x-slabs of each image with the reference's box: [int((cx-rb)/dx), int((cx+rb)/dx)+1)."""
    M = img_cx.shape[0]
    ix0 = np.empty(M, dtype=np.int64)
    ix1 = np.empty(M, dtype=np.int64)
    count = np.zeros(Ng, dtype=np.int64)
    for m in range(M):
        lo = int((img_cx[m] - img_rb[m]) / dx)
        hi = int((img_cx[m] + img_rb[m]) / dx) + 1
        if lo < 0:
            lo = 0
        if hi > Ng:
            hi = Ng
        if hi < lo:
            hi = lo
        ix0[m] = lo
        ix1[m] = hi
        for ix in range(lo, hi):
            count[ix] += 1
    off = np.zeros(Ng + 1, dtype=np.int64)
    for ix in range(Ng):
        off[ix + 1] = off[ix] + count[ix]
    fill = off[:Ng].copy()
    slab = np.empty(off[Ng], dtype=np.int32)
    for m in range(M):
        for ix in range(ix0[m], ix1[m]):
            slab[fill[ix]] = m
            fill[ix] += 1
    return off, slab


# ──────────────────────────────────────────────────────────────────────
# stamping kernels
# ──────────────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True, nogil=True)
def _render_rows_2d(phi_s, xg, yg, dy, off, slab, img_gran, img_cx, img_cy, img_rb, sid, is_circle,
                    r_eff, a, b, n_shape, theta, r, w, clip_n, clip_d, clip_cnt):
    Ng = xg.shape[0]
    for ix in prange(Ng):
        X = xg[ix]
        for q in range(off[ix], off[ix + 1]):
            m = slab[q]
            g = img_gran[m]
            cx = img_cx[m]
            cy = img_cy[m]
            rb = img_rb[m]
            lo = int(np.floor((cy - rb) / dy - 0.5)) - 1
            hi = int(np.ceil((cy + rb) / dy - 0.5)) + 2
            if lo < 0:
                lo = 0
            if hi > Ng:
                hi = Ng
            k = sid[g]
            nclip = clip_cnt[g]
            dxp = X - cx
            if is_circle:
                re = r_eff[g]
                for iy in range(lo, hi):
                    dyp = yg[iy] - cy
                    sdf = np.sqrt(dxp*dxp + dyp*dyp) - re
                    for c in range(nclip):
                        plane = dxp * clip_n[g, c, 0] + dyp * clip_n[g, c, 1] - clip_d[g, c]
                        if plane > sdf:
                            sdf = plane
                    phi_s[k, ix, iy] += 0.5 * (1.0 - np.tanh(sdf / w))
            else:
                ct = np.cos(theta[g])
                st = np.sin(theta[g])
                ag = a[g]
                bg = b[g]
                ng = n_shape[g]
                rg = r[g]
                n_inv = 1.0 / ng
                for iy in range(lo, hi):
                    dyp = yg[iy] - cy
                    bx = ct * dxp + st * dyp
                    by = -st * dxp + ct * dyp
                    se_val = (abs(bx) / ag)**ng + (abs(by) / bg)**ng
                    sdf = (se_val**n_inv - 1.0) * rg
                    for c in range(nclip):
                        plane = dxp * clip_n[g, c, 0] + dyp * clip_n[g, c, 1] - clip_d[g, c]
                        if plane > sdf:
                            sdf = plane
                    phi_s[k, ix, iy] += 0.5 * (1.0 - np.tanh(sdf / w))


@njit(parallel=True, cache=True, nogil=True)
def _render_slabs_3d(phi_s, xg, yg, zg, dy, dz, off, slab, img_gran, img_cx, img_cy, img_cz, img_rb,
                     sid, is_sphere, r_eff, a, b, c, n1, n2, quat, r, w, clip_n, clip_d, clip_cnt):
    Ng = xg.shape[0]
    for ix in prange(Ng):
        X = xg[ix]
        for q in range(off[ix], off[ix + 1]):
            m = slab[q]
            g = img_gran[m]
            cx = img_cx[m]
            cy = img_cy[m]
            cz = img_cz[m]
            rb = img_rb[m]
            iy0 = int((cy - rb) / dy)
            iy1 = int((cy + rb) / dy) + 1
            iz0 = int((cz - rb) / dz)
            iz1 = int((cz + rb) / dz) + 1
            if iy0 < 0:
                iy0 = 0
            if iz0 < 0:
                iz0 = 0
            if iy1 > Ng:
                iy1 = Ng
            if iz1 > Ng:
                iz1 = Ng
            if iy0 >= iy1 or iz0 >= iz1:
                continue
            k = sid[g]
            nclip = clip_cnt[g]
            dxl = X - cx
            if is_sphere:
                re = r_eff[g]
                for iy in range(iy0, iy1):
                    dyl = yg[iy] - cy
                    for iz in range(iz0, iz1):
                        dzl = zg[iz] - cz
                        sdf = np.sqrt(dxl*dxl + dyl*dyl + dzl*dzl) - re
                        for cc in range(nclip):
                            plane = (dxl * clip_n[g, cc, 0] + dyl * clip_n[g, cc, 1]
                                     + dzl * clip_n[g, cc, 2] - clip_d[g, cc])
                            if plane > sdf:
                                sdf = plane
                        phi_s[k, ix, iy, iz] += 0.5 * (1.0 - np.tanh(sdf / w))
            else:
                R = quat_to_rotation_matrix(quat[g])
                ag = a[g]
                bg = b[g]
                cg = c[g]
                n1g = n1[g]
                n2g = n2[g]
                rg = r[g]
                e_ratio = n2g / n1g
                n_inv = 1.0 / n2g
                for iy in range(iy0, iy1):
                    dyl = yg[iy] - cy
                    for iz in range(iz0, iz1):
                        dzl = zg[iz] - cz
                        bxb = R[0, 0]*dxl + R[1, 0]*dyl + R[2, 0]*dzl
                        byb = R[0, 1]*dxl + R[1, 1]*dyl + R[2, 1]*dzl
                        bzb = R[0, 2]*dxl + R[1, 2]*dyl + R[2, 2]*dzl
                        se_val = ((abs(bxb / ag)**n1g + abs(byb / bg)**n1g)**e_ratio
                                  + abs(bzb / cg)**n2g)
                        sdf = (se_val**n_inv - 1.0) * rg
                        for cc in range(nclip):
                            plane = (dxl * clip_n[g, cc, 0] + dyl * clip_n[g, cc, 1]
                                     + dzl * clip_n[g, cc, 2] - clip_d[g, cc])
                            if plane > sdf:
                                sdf = plane
                        phi_s[k, ix, iy, iz] += 0.5 * (1.0 - np.tanh(sdf / w))


# ──────────────────────────────────────────────────────────────────────
# drivers
# ──────────────────────────────────────────────────────────────────────

def render_species_2d(gs, p):
    """phi_s (K, Ng, Ng) — compiled twin of ``reference._render_species_2d``."""
    min_Ng = int(np.ceil(max(p.Lx, p.Ly) / (p.interface_width * 2.0)))
    Ng = max(p.Ngrid, min_Ng)
    dx = p.Lx / Ng
    dy = p.Ly / Ng
    xg = np.linspace(dx/2, p.Lx - dx/2, Ng)
    yg = np.linspace(dy/2, p.Ly - dy/2, Ng)
    K, sid, _adh, _fsp = species_view(gs)
    sid = np.ascontiguousarray(sid, dtype=np.int32)
    w = float(p.interface_width)
    periodic = (p.boundary_mode == 'periodic')
    N = gs.N
    x = np.ascontiguousarray(gs.x[:N])
    y = np.ascontiguousarray(gs.y[:N])

    # Ghost images are selected with the reference's criterion (3w beyond the
    # edge); each stamped image is truncated at 8w, where the profile is < 1e-7.
    if gs.is_circle:
        r_eff, _ = effective_radii_2d(gs, p)
        rb_img = r_eff + 3.0 * w
        rb_box = r_eff + 8.0 * w
    else:
        r_eff = gs.r
        rb_img = gs.r_bound + 3.0 * w
        rb_box = np.sqrt(gs.a**2 + gs.b**2) * (1.0 + 8.0 * w / gs.r)

    img_gran, img_cx, img_cy = _images_2d(x, y, rb_img, float(p.Lx), float(p.Ly), periodic)
    img_rb = np.ascontiguousarray(rb_box[img_gran])
    clip_n, clip_d, clip_cnt = clips_from_lists(gs, 2)
    off, slab = _slab_index_2d(img_cx, img_rb, dx, Ng)

    phi_s = np.zeros((K, Ng, Ng))
    _render_rows_2d(phi_s, xg, yg, dy, off, slab, img_gran, img_cx, img_cy, img_rb, sid,
                    bool(gs.is_circle), np.ascontiguousarray(r_eff), gs.a, gs.b, gs.n_shape, gs.theta,
                    gs.r, w, clip_n, clip_d, clip_cnt)

    A_pixel = (p.Lx / Ng) * (p.Ly / Ng)
    A_true = []
    areas = None if gs.is_circle else shape_cache(gs)['areas']
    for k in range(K):
        mk = sid == k
        if not np.any(mk):
            A_true.append(0.0)
        elif gs.is_circle:
            A_true.append(float(np.sum(np.pi * gs.r[mk] ** 2)))
        else:
            A_true.append(float(sum(areas[mk])))
    _normalize_species_fields(phi_s, A_true, A_pixel)
    return phi_s


def render_species_3d(gs, p):
    """phi_s (K, Ng, Ng, Ng) — compiled twin of ``reference._render_species_3d``."""
    min_Ng = int(np.ceil(max(p.Lx, p.Ly, p.Lz) / (p.interface_width * 2.0)))
    Ng = max(p.Ngrid_3d, min(min_Ng, 200))
    dx_g = p.Lx / Ng
    dy_g = p.Ly / Ng
    dz_g = p.Lz / Ng
    xg = np.linspace(dx_g/2, p.Lx - dx_g/2, Ng)
    yg = np.linspace(dy_g/2, p.Ly - dy_g/2, Ng)
    zg = np.linspace(dz_g/2, p.Lz - dz_g/2, Ng)
    w = max(p.interface_width, max(dx_g, dy_g, dz_g))
    K, sid, _adh, _fsp = species_view(gs)
    sid = np.ascontiguousarray(sid, dtype=np.int32)
    periodic = (p.boundary_mode == 'periodic')
    N = gs.N
    x = np.ascontiguousarray(gs.x[:N])
    y = np.ascontiguousarray(gs.y[:N])
    z = np.ascontiguousarray(gs.z[:N])

    if gs.is_circle:
        r_eff_3d, _ = effective_radii_3d(gs, p)
    else:
        r_eff_3d = gs.r
    rb = gs.r_bound + 3 * w

    img_gran, img_cx, img_cy, img_cz = _images_3d(x, y, z, rb, float(p.Lx), float(p.Ly), float(p.Lz),
                                                  periodic)
    img_rb = np.ascontiguousarray(rb[img_gran])
    clip_n, clip_d, clip_cnt = clips_from_lists(gs, 3)
    off, slab = _slab_index_3d(img_cx, img_rb, dx_g, Ng)
    quat = np.ascontiguousarray(gs.quat) if gs.quat is not None else np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))

    phi_s = np.zeros((K, Ng, Ng, Ng))
    _render_slabs_3d(phi_s, xg, yg, zg, dy_g, dz_g, off, slab, img_gran, img_cx, img_cy, img_cz, img_rb,
                     sid, bool(gs.is_circle), np.ascontiguousarray(r_eff_3d), gs.a, gs.b, gs.c, gs.n1, gs.n2,
                     quat, gs.r, float(w), clip_n, clip_d, clip_cnt)

    V_voxel = dx_g * dy_g * dz_g
    V_true = []
    vols = None if gs.is_circle else shape_cache(gs)['vols']
    for k in range(K):
        mk = sid == k
        if not np.any(mk):
            V_true.append(0.0)
        elif gs.is_circle:
            V_true.append(float(np.sum((4.0 / 3.0) * np.pi * gs.r[mk] ** 3)))
        else:
            V_true.append(float(sum(vols[mk])))
    _normalize_species_fields(phi_s, V_true, V_voxel)
    return phi_s


def render_fields_species(gs, p):
    """Per-species phase fields; dispatches on the system's dimensionality."""
    if gs.is_3d:
        return render_species_3d(gs, p)
    return render_species_2d(gs, p)


__all__ = ['render_fields_species', 'render_species_2d', 'render_species_3d',
           'effective_radii_2d', 'effective_radii_3d', 'clips_from_lists']
