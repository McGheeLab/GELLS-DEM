"""
Laguerre (radical / power) tessellation for local packing fraction (V3.3).
==========================================================================

``phi_solid`` / ``porosity`` / ``compaction_ratio`` are field measures: they are
inflated ~1.5x at R = 20 um by the tanh interface halo and again by pairwise
overlap double counting (CLAUDE.md). This module measures the same quantity
without a grid -- each granule's own share of space, from an exact partition.

A **radical** cell weights each bisector by radius: the plane between i and j
sits where ``|x-ci|^2 - ri^2 = |x-cj|^2 - rj^2``, i.e. at the radical plane
rather than the midpoint. That is the correct construction for a polydisperse
pack; an ordinary (unweighted) Voronoi mis-assigns space at large size
disparity and drives the small granules' local fraction above 1.

The cell is already a half-space intersection, so **a container wall is just
one more row**. That makes the walled case SIMPLER than robotsim's periodic
original, which needs image points purely to bound the cells: with six box
planes every cell is bounded even at N = 5.

Three geometries:

* **box** -- six axis-aligned planes.
* **cylinder** -- a curved wall is not a half-space, so it is polygonalised
  into ``CYL_WALL_PLANES`` tangent planes (0.12 % radius error at 64).
* **free top** -- there is no lid, so surface cells would be unbounded. A loose
  lid is added well above the bed and any cell touching it is marked INVALID
  and excluded from the aggregates, rather than being given a meaningless
  volume. ``laguerre_valid_frac`` is reported so a mean is never quoted without
  its denominator. In a closed box the exclusion count must be zero, which is a
  built-in consistency check.

Qhull cannot be compiled, so there is no kernel twin: both metrics twins call
this one module, which is also why they agree to rounding.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, cKDTree

# Solver constants, not Params fields -- every Params field costs a FLAT_MAP
# entry, a template line and a validate rule (gels/kernels/packing.py:196-198).
NBR_CUTOFF_FACTOR = 6.0    # neighbour search radius, in max radii (robotsim uses 6)
CYL_WALL_PLANES = 64       # tangent planes for a cylindrical wall (0.12 % radius error)
LID_CLEARANCE = 2.0        # loose lid height above the bed, in max radii


def container_planes(box, dim, shape_code=0, R_cyl=None, cx=None, cy=None,
                     top_free=False, lid=None):
    """Half-spaces ``A x <= b`` bounding the container, as ``(A, b)``.

    With ``top_free`` the top plane is the loose lid, which exists only to
    bound the tessellation; cells that touch it are invalid.
    """
    A = []
    b = []
    up = dim - 1                      # z in 3D, y in 2D
    for k in range(dim):
        hi = float(box[k])
        if k == up and top_free:
            hi = float(lid)
        n_hi = np.zeros(dim)
        n_hi[k] = 1.0
        A.append(n_hi)
        b.append(hi)
        n_lo = np.zeros(dim)
        n_lo[k] = -1.0
        A.append(n_lo)
        b.append(0.0)
    if shape_code == 1 and dim == 3:
        # replace the x/y box planes with tangent planes to the cylinder
        A = A[4:]
        b = b[4:]
        for t in np.arange(CYL_WALL_PLANES) * (2.0 * np.pi / CYL_WALL_PLANES):
            nx, ny = np.cos(t), np.sin(t)
            A.append(np.array([nx, ny, 0.0]))
            b.append(float(R_cyl) + nx * float(cx) + ny * float(cy))
    return np.asarray(A, dtype=float), np.asarray(b, dtype=float)


def _safe_cutoff(dv, r_i, r_max):
    """Radius beyond which no granule can still cut a cell whose farthest
    vertex is at ``dv``.

    The radical plane between i and j meets the line of centres at
    ``t = (|d|^2 + ri^2 - rj^2) / (2|d|)``. It can only clip a vertex at
    distance ``dv`` if ``t < dv``, i.e. ``|d|^2 - 2 dv |d| + (ri^2 - rj^2) < 0``.
    Taking the worst case ``rj = r_max`` and solving for ``|d|`` gives the
    bound below. For equal radii this reduces to the familiar ``2 dv`` of an
    unweighted Voronoi diagram.
    """
    disc = dv * dv + max(0.0, r_max * r_max - r_i * r_i)
    return dv + np.sqrt(disc)


def _interior_point(A, b, guess):
    """A point strictly inside ``A x <= b``, or None if the region is empty.

    ``guess`` (the granule centre) is used when it works, which is almost
    always. It does NOT work when the granule's own centre lies outside its
    power cell -- possible when a much larger neighbour overlaps it deeply
    (``|d|^2 < rj^2 - ri^2``, which needs roughly a 6:1 radius ratio at the
    overlaps this engine allows). The cell is still non-empty there, so falling
    back to NaN would silently drop a real region; instead solve for the
    Chebyshev centre, the deepest interior point.
    """
    slack = b - A @ guess
    if np.all(slack > 1e-9):
        return guess
    # maximise the inscribed radius t subject to  a.x + |a| t <= b
    norms = np.linalg.norm(A, axis=1)
    n = A.shape[1]
    res = linprog(
        c=np.concatenate([np.zeros(n), [-1.0]]),
        A_ub=np.hstack([A, norms[:, None]]), b_ub=b,
        bounds=[(None, None)] * n + [(0, None)], method='highs')
    if not res.success or res.x[-1] <= 1e-9:
        return None                                  # genuinely empty
    return res.x[:n]


def _cell(ci, r_i, cj, w_j, w_i, Aw, bw, up, top_free, lid):
    """Build one radical cell; returns (volume, max vertex distance, touches lid)."""
    if len(cj):
        A = 2.0 * (cj - ci)
        b = (np.sum(cj * cj, axis=1) - w_j) - (float(ci @ ci) - w_i)
        A = np.vstack([A, Aw])
        b = np.concatenate([b, bw])
    else:
        A, b = Aw, bw
    interior = _interior_point(A, b, ci)
    if interior is None:
        raise ValueError('empty radical cell')
    hs = np.hstack([A, -b[:, None]])
    pts = HalfspaceIntersection(hs, interior).intersections
    vol = ConvexHull(pts).volume
    dv = float(np.max(np.linalg.norm(pts - ci, axis=1)))
    touch = bool(np.max(pts[:, up]) >= float(lid) - 1e-6) if top_free else False
    return vol, dv, touch


def radical_volumes(pos, r, box, dim, shape_code=0, R_cyl=None, cx=None, cy=None,
                    top_free=False, lid=None, cutoff=None, max_rounds=6):
    """Radical cell volume per granule, and which cells touch the lid.

    Returns ``(vol, touches_lid)``. ``vol`` is NaN where the cell could not be
    built -- a granule whose power cell is empty, which happens when a much
    larger overlapping neighbour's weight swallows it.

    The neighbour cutoff is **verified, not guessed**: after building a cell we
    check that no granule outside the cutoff could have clipped it
    (``_safe_cutoff``), and grow the cutoff and rebuild if one could. A fixed
    radius silently produces cells that are too large -- at 6*r_max on a
    40-granule box the total came to 4.8x the box volume.
    """
    pos = np.asarray(pos, dtype=float)[:, :dim]
    r = np.asarray(r, dtype=float)
    n = len(pos)
    vol = np.full(n, np.nan)
    touches = np.zeros(n, dtype=bool)
    if n == 0:
        return vol, touches

    Aw, bw = container_planes(box, dim, shape_code, R_cyl, cx, cy, top_free, lid)
    r_max = float(r.max())
    w = r ** 2
    up = dim - 1
    tree = cKDTree(pos)
    # start from the local spacing, not from a radius multiple: the two are
    # unrelated in a dilute pack
    span = float(np.prod([float(box[k]) for k in range(dim)])) ** (1.0 / dim)
    cut0 = max(NBR_CUTOFF_FACTOR * r_max, 2.5 * span / max(n, 1) ** (1.0 / dim))
    if cutoff is not None:
        cut0 = float(cutoff)

    for i in range(n):
        ci = pos[i]
        cut = cut0
        for _ in range(max_rounds):
            idx = tree.query_ball_point(ci, cut)
            idx = [j for j in idx if j != i]
            try:
                v, dv, touch = _cell(ci, r[i], pos[idx], w[idx], w[i],
                                     Aw, bw, up, top_free, lid)
            except Exception:
                break                               # empty / degenerate -> NaN
            if cut >= _safe_cutoff(dv, float(r[i]), r_max) or len(idx) >= n - 1:
                vol[i] = v
                touches[i] = touch
                break
            cut *= 2.0
        else:
            vol[i] = v
            touches[i] = touch
    return vol, touches


def laguerre_metrics(gs, p, pos, phi_rcp=None):
    """The ``phi_loc_*`` metric block. Called by BOTH metrics twins.

    Returns ``{}`` when ``metrics_laguerre`` is off, so the keys simply do not
    appear -- ``save_history_to_disk`` takes the union of keys in first-seen
    order and ``FRAME_METRIC_KEYS`` filters with ``if k in m``, so conditional
    keys are safe.
    """
    if not bool(getattr(p, 'metrics_laguerre', False)):
        return {}
    from gels.engine import boundary_geometry, domain_volume

    n = int(gs.N)
    if n == 0:
        return {}
    dim = 3 if gs.is_3d else 2
    geom = boundary_geometry(p)
    box = (float(p.Lx), float(p.Ly), float(getattr(p, 'Lz', 0.0)))
    r = np.asarray(gs.r[:n], dtype=float)
    up_idx = dim - 1
    lid = None
    if geom.top_free:
        lid = float(np.max(pos[:n, up_idx] + gs.r_bound[:n])
                    + LID_CLEARANCE * float(gs.r_bound[:n].max()))

    vol, touches = radical_volumes(
        pos[:n], r, box, dim,
        shape_code=geom.shape_code, R_cyl=geom.R_cyl, cx=geom.cx, cy=geom.cy,
        top_free=geom.top_free, lid=lid)

    # granule volume: the rigid shape's own measure, not the bounding sphere
    if gs.is_circle:
        gvol = (4.0 / 3.0) * np.pi * r ** 3 if dim == 3 else np.pi * r ** 2
    else:
        from gels.kernels.metrics import shape_cache
        cache = shape_cache(gs)
        gvol = np.asarray(cache['vols'] if dim == 3 else cache['areas'], dtype=float)[:n]

    valid = np.isfinite(vol) & (vol > 0) & (~touches)
    fixed = np.asarray(getattr(gs, 'fixed', np.zeros(n, dtype=bool))[:n], dtype=bool)
    valid &= ~fixed            # boundary-layer granules bound their neighbours
    out = {
        'laguerre_n_valid': int(valid.sum()),
        'laguerre_valid_frac': float(valid.sum() / n),
        # tiling check: 1.0 in a closed box, < 1 when lid cells are excluded
        'laguerre_vol_closure': float(np.nansum(vol) / max(domain_volume(p), 1e-30)),
    }
    if not valid.any():
        return out

    phi = np.where(valid, gvol / np.where(vol > 0, vol, 1.0), np.nan)
    pv = phi[valid]
    out['phi_loc_mean'] = float(pv.mean())
    out['phi_loc_std'] = float(pv.std())
    out['phi_loc_p10'] = float(np.percentile(pv, 10))
    out['phi_loc_p90'] = float(np.percentile(pv, 90))

    gtype = np.asarray(gs.gtype[:n])
    for key, mask in (('func', gtype == 0), ('inert', gtype == 1)):
        sel = mask & valid
        # aggregate as a RATIO OF SUMS, not a mean of ratios: this is a phase
        # packing fraction, so big granules must count for more
        out[f'phi_loc_{key}'] = (float(gvol[sel].sum() / vol[sel].sum())
                                 if sel.any() else 0.0)
    out['demix_phi_loc'] = out['phi_loc_func'] - out['phi_loc_inert']

    sid = np.asarray(gs.species_id[:n])
    for k in range(int(getattr(gs, 'K', 0)) or (sid.max() + 1 if n else 0)):
        sel = (sid == k) & valid
        out[f'phi_loc_sp_{k}'] = (float(gvol[sel].sum() / vol[sel].sum())
                                  if sel.any() else 0.0)

    if phi_rcp:
        # the halo-free twin of compaction_ratio, sharing its phi_RCP so the
        # DIFFERENCE between the two is the halo inflation
        out['compaction_func'] = out['phi_loc_func'] / phi_rcp
        out['compaction_inert'] = out['phi_loc_inert'] / phi_rcp
    return out
