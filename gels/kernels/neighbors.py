"""
Neighbour pair lists for the compiled kernels (V3.0 Phase 6a / 6g).
===================================================================

Two backends produce the same *set* of half pairs (i < j, distance ≤ cutoff):

* ``half_pairs_tree`` — ``cKDTree.query_pairs`` in the reference's order. Used
  whenever results must match the reference step for step: the exact Python
  cell machinery (``perf_cells_backend = 'python'``), the packing settle, and
  ``perf_neighbor_backend = 'ckdtree'``.
* ``half_pairs_cells`` — a compiled linked-cell grid, parallel over particles,
  with each particle's partners sorted, so the pair order is lexicographic
  (deterministic and independent of the thread count). Default on the
  kernel path (``perf_neighbor_backend = 'cells'``). Falls back to the tree
  for periodic boxes narrower than three cells.

``csr_from_pairs`` turns a half-pair list into the per-particle CSR
(``off``, ``nbr_pair``, ``nbr_side``) used by the owner-writes gather passes.
"""

import numpy as np
from scipy.spatial import cKDTree

from gels.kernels import njit, prange


def half_pairs_tree(pos, cutoff, periodic, box):
    """Half pair list (i < j) within ``cutoff`` from cKDTree — int32 arrays, tree order."""
    d = pos.shape[1]
    if periodic:
        tree = cKDTree(pos, boxsize=[float(b) for b in box[:d]])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if pairs.shape[0] == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    return (np.ascontiguousarray(pairs[:, 0], dtype=np.int32),
            np.ascontiguousarray(pairs[:, 1], dtype=np.int32))


@njit(cache=True)
def _cell_bins(pos, dim, ncx, ncy, ncz, sx, sy, sz):
    N = pos.shape[0]
    cell = np.empty(N, dtype=np.int64)
    for i in range(N):
        ix = int(pos[i, 0] // sx)
        iy = int(pos[i, 1] // sy)
        iz = int(pos[i, 2] // sz) if dim == 3 else 0
        if ix < 0:
            ix = 0
        if ix >= ncx:
            ix = ncx - 1
        if iy < 0:
            iy = 0
        if iy >= ncy:
            iy = ncy - 1
        if iz < 0:
            iz = 0
        if iz >= ncz:
            iz = ncz - 1
        cell[i] = (ix * ncy + iy) * ncz + iz
    ncell = ncx * ncy * ncz
    start = np.zeros(ncell + 1, dtype=np.int64)
    for i in range(N):
        start[cell[i] + 1] += 1
    for c in range(ncell):
        start[c + 1] += start[c]
    fill = start[:ncell].copy()
    items = np.empty(N, dtype=np.int64)
    for i in range(N):
        items[fill[cell[i]]] = i
        fill[cell[i]] += 1
    return cell, start, items


@njit(parallel=True, cache=True, nogil=True)
def _count_pairs(pos, dim, cell, start, items, ncx, ncy, ncz, periodic, Lx, Ly, Lz, cutoff2, cnt):
    N = pos.shape[0]
    rz = 1 if dim == 3 else 0
    for i in prange(N):
        c = cell[i]
        iz = c % ncz
        iy = (c // ncz) % ncy
        ix = c // (ncz * ncy)
        n = 0
        for ox in range(-1, 2):
            jx = ix + ox
            if periodic:
                jx = jx % ncx
            elif jx < 0 or jx >= ncx:
                continue
            for oy in range(-1, 2):
                jy = iy + oy
                if periodic:
                    jy = jy % ncy
                elif jy < 0 or jy >= ncy:
                    continue
                for oz in range(-rz, rz + 1):
                    jz = iz + oz
                    if periodic:
                        jz = jz % ncz
                    elif jz < 0 or jz >= ncz:
                        continue
                    cc = (jx * ncy + jy) * ncz + jz
                    for q in range(start[cc], start[cc + 1]):
                        j = items[q]
                        if j <= i:
                            continue
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        dz = pos[j, 2] - pos[i, 2] if dim == 3 else 0.0
                        if periodic:
                            dx = dx - Lx * np.rint(dx / Lx)
                            dy = dy - Ly * np.rint(dy / Ly)
                            if dim == 3:
                                dz = dz - Lz * np.rint(dz / Lz)
                        if dx*dx + dy*dy + dz*dz <= cutoff2:
                            n += 1
        cnt[i] = n


@njit(parallel=True, cache=True, nogil=True)
def _fill_pairs(pos, dim, cell, start, items, ncx, ncy, ncz, periodic, Lx, Ly, Lz, cutoff2,
                off, pair_i, pair_j):
    N = pos.shape[0]
    rz = 1 if dim == 3 else 0
    for i in prange(N):
        c = cell[i]
        iz = c % ncz
        iy = (c // ncz) % ncy
        ix = c // (ncz * ncy)
        w = off[i]
        for ox in range(-1, 2):
            jx = ix + ox
            if periodic:
                jx = jx % ncx
            elif jx < 0 or jx >= ncx:
                continue
            for oy in range(-1, 2):
                jy = iy + oy
                if periodic:
                    jy = jy % ncy
                elif jy < 0 or jy >= ncy:
                    continue
                for oz in range(-rz, rz + 1):
                    jz = iz + oz
                    if periodic:
                        jz = jz % ncz
                    elif jz < 0 or jz >= ncz:
                        continue
                    cc = (jx * ncy + jy) * ncz + jz
                    for q in range(start[cc], start[cc + 1]):
                        j = items[q]
                        if j <= i:
                            continue
                        dx = pos[j, 0] - pos[i, 0]
                        dy = pos[j, 1] - pos[i, 1]
                        dz = pos[j, 2] - pos[i, 2] if dim == 3 else 0.0
                        if periodic:
                            dx = dx - Lx * np.rint(dx / Lx)
                            dy = dy - Ly * np.rint(dy / Ly)
                            if dim == 3:
                                dz = dz - Lz * np.rint(dz / Lz)
                        if dx*dx + dy*dy + dz*dz <= cutoff2:
                            pair_i[w] = i
                            pair_j[w] = j
                            w += 1
        # sort this particle's partners (insertion sort; Z is small)
        lo = off[i]
        for a in range(lo + 1, w):
            v = pair_j[a]
            b = a - 1
            while b >= lo and pair_j[b] > v:
                pair_j[b + 1] = pair_j[b]
                b -= 1
            pair_j[b + 1] = v


def half_pairs_cells(pos, cutoff, periodic, box):
    """Half pair list within ``cutoff`` from a linked-cell grid; lexicographically sorted."""
    N, dim = pos.shape
    if N < 2:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int32)
    ncs = [max(1, int(float(box[k]) // cutoff)) for k in range(dim)]
    if periodic and min(ncs) < 3:
        return half_pairs_tree(pos, cutoff, periodic, box)
    if dim == 2:
        ncs.append(1)
    ncx, ncy, ncz = ncs
    sx = float(box[0]) / ncx
    sy = float(box[1]) / ncy
    sz = float(box[2]) / ncz if dim == 3 else 1.0
    Lx, Ly = float(box[0]), float(box[1])
    Lz = float(box[2]) if dim == 3 else 1.0
    cell, start, items = _cell_bins(pos, dim, ncx, ncy, ncz, sx, sy, sz)
    cnt = np.zeros(N, dtype=np.int64)
    cutoff2 = float(cutoff) * float(cutoff)
    _count_pairs(pos, dim, cell, start, items, ncx, ncy, ncz, periodic, Lx, Ly, Lz, cutoff2, cnt)
    off = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(cnt, out=off[1:])
    M = int(off[N])
    pair_i = np.empty(M, dtype=np.int32)
    pair_j = np.empty(M, dtype=np.int32)
    if M:
        _fill_pairs(pos, dim, cell, start, items, ncx, ncy, ncz, periodic, Lx, Ly, Lz, cutoff2,
                    off, pair_i, pair_j)
    return pair_i, pair_j


def half_pairs(pos, cutoff, periodic, box, backend='cells'):
    """Dispatch on ``perf_neighbor_backend``: 'cells' (default) or 'ckdtree'."""
    if backend == 'cells':
        return half_pairs_cells(pos, cutoff, periodic, box)
    return half_pairs_tree(pos, cutoff, periodic, box)


def neighbor_backend(p):
    """Backend for this Params: the tree whenever the exact Python cell machinery runs
    (its random stream depends on the reference pair order), else ``perf_neighbor_backend``."""
    if getattr(p, 'perf_cells_backend', 'kernels') != 'kernels':
        return 'ckdtree'
    return getattr(p, 'perf_neighbor_backend', 'cells')


@njit(cache=True)
def csr_from_pairs(pair_i, pair_j, N):
    """Per-particle CSR over a half pair list.

    Returns ``off`` (N+1,) int64, ``nbr_pair`` (2M,) int32 — the pair index
    of each incidence, in pair order — and ``nbr_side`` (2M,) int8 — 0 when
    the particle is ``pair_i`` of that pair, 1 when it is ``pair_j``.
    """
    M = pair_i.shape[0]
    count = np.zeros(N, dtype=np.int64)
    for k in range(M):
        count[pair_i[k]] += 1
        count[pair_j[k]] += 1
    off = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        off[i + 1] = off[i] + count[i]
    fill = off[:N].copy()
    nbr_pair = np.empty(off[N], dtype=np.int32)
    nbr_side = np.empty(off[N], dtype=np.int8)
    for k in range(M):
        i = pair_i[k]
        j = pair_j[k]
        nbr_pair[fill[i]] = k
        nbr_side[fill[i]] = 0
        fill[i] += 1
        nbr_pair[fill[j]] = k
        nbr_side[fill[j]] = 1
        fill[j] += 1
    return off, nbr_pair, nbr_side


__all__ = ['half_pairs_tree', 'half_pairs_cells', 'half_pairs', 'neighbor_backend', 'csr_from_pairs']
