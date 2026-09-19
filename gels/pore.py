"""
Katz-Thompson permeability and geodesic tortuosity (V3.3).
==========================================================

``K_kozeny_carman`` depends only on porosity and specific surface, so it is
**blind to channelization**: in a fixed volume the bulk porosity barely moves
while the pore geometry changes completely, as coarsening merges small pores
into wide channels. Katz-Thompson instead keys on the *critical pore* -- the
widest aperture at which the void still spans the sample -- which is exactly
the quantity that controls flow.

The signed-distance field this needs is already half-built in both metrics
twins: ``tree.query(voxel_pts)`` returns ``(distance, index)`` and they keep
only the index. ``sd = distance - r[nearest]`` is the distance to the nearest
granule SURFACE, positive in the void, and its value there is the local pore
radius.

Two approximations, inherited from robotsim and stated rather than hidden:

* nearest-CENTRE minus that centre's radius is not the distance to the nearest
  SURFACE for a polydisperse pack; the error is bounded by ``r_max - r_min``
  and only arises in the shadow of a large granule. ``k=1`` is used in-metrics;
  the post-hoc module can afford ``k=4``.
* ``gs.r`` is the equal-volume radius, so for superellipsoids the field is
  wrong by up to ``r_bound - r``. Using ``r`` matches the ``phi_solid_true``
  convention; ``r_bound`` would over-fill the pore space.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

# Solver constants (see gels/kernels/packing.py:196-198 for why these are not
# Params fields).
N_BISECT = 16              # bisection steps for the critical pore radius
KT_CONSTANT = 226.0        # Katz-Thompson  k = d_c^2 / 226 * (phi/tau)
KC_CONSTANT = 5.0          # Kozeny-Carman in the specific-surface form


def signed_field(pos, r, shape, box, dim, periodic=False, k=1):
    """Distance to the nearest granule surface on a regular grid (>0 = void)."""
    axes = [(np.arange(shape[i]) + 0.5) * (float(box[i]) / shape[i]) for i in range(dim)]
    grids = np.meshgrid(*axes, indexing='ij')
    pts = np.column_stack([g.ravel() for g in grids])
    if periodic:
        tree = cKDTree(np.mod(pos[:, :dim], [float(b) for b in box[:dim]]),
                       boxsize=[float(b) for b in box[:dim]])
    else:
        tree = cKDTree(pos[:, :dim])
    dist, idx = tree.query(pts, k=k, workers=-1)
    if k == 1:
        sd = dist - r[idx]
    else:
        sd = np.min(dist - r[idx], axis=1)
    return sd.reshape(tuple(shape))


def critical_pore(sd, mask=None, axis=-1):
    """(porosity, critical pore RADIUS): the widest threshold that still spans.

    Invasion percolation of the pore space -- bisect on the threshold and ask
    whether ``sd >= t`` still connects the inlet face to the outlet face.
    """
    inside = np.ones_like(sd, dtype=bool) if mask is None else mask
    void = (sd > 0) & inside
    eps = float(void.sum() / max(inside.sum(), 1))
    lo, hi, span = 0.0, float(sd.max()) if sd.size else 0.0, 0.0
    for _ in range(N_BISECT):
        mid = 0.5 * (lo + hi)
        lab, _n = ndimage.label((sd >= mid) & inside)
        first = set(np.unique(np.take(lab, 0, axis=axis))) - {0}
        last = set(np.unique(np.take(lab, -1, axis=axis))) - {0}
        if first & last:
            span = mid
            lo = mid
        else:
            hi = mid
    return eps, span


def tortuosity_geo(sd, mask=None, axis=-1, periodic=False):
    """(tortuosity, spans): geodesic path length along ``axis`` / straight distance.

    Grassfire BFS from the inlet face. Lateral wrap-around is applied only
    under periodic boundaries -- a wall is not a mirror, and rolling across one
    invents paths and deflates the tortuosity.
    """
    inside = np.ones_like(sd, dtype=bool) if mask is None else mask
    void = (sd > 0) & inside
    n_along = void.shape[axis]
    inlet = np.take(void, 0, axis=axis)
    outlet = np.take(void, -1, axis=axis)
    if not inlet.any() or not outlet.any():
        return 0.0, False
    ndim = void.ndim
    lateral = [a for a in range(ndim) if a != (axis % ndim)]
    cur = np.zeros_like(void)
    idx = [slice(None)] * ndim
    idx[axis] = 0
    cur[tuple(idx)] = inlet
    visited = cur.copy()
    for d in range(n_along * 4):
        if not cur.any():
            break
        tail = [slice(None)] * ndim
        tail[axis] = -1
        if cur[tuple(tail)].any():
            return float(d / max(n_along - 1, 1)), True
        nb = np.zeros_like(void)
        if periodic:
            for a in lateral:
                nb |= np.roll(cur, 1, a) | np.roll(cur, -1, a)
        else:
            for a in lateral:
                sl_hi = [slice(None)] * ndim
                sl_lo = [slice(None)] * ndim
                sl_hi[a] = slice(1, None)
                sl_lo[a] = slice(None, -1)
                up = np.zeros_like(void)
                up[tuple(sl_hi)] = cur[tuple(sl_lo)]
                dn = np.zeros_like(void)
                dn[tuple(sl_lo)] = cur[tuple(sl_hi)]
                nb |= up | dn
        sl_hi = [slice(None)] * ndim
        sl_lo = [slice(None)] * ndim
        sl_hi[axis] = slice(1, None)
        sl_lo[axis] = slice(None, -1)
        up = np.zeros_like(void)
        up[tuple(sl_hi)] = cur[tuple(sl_lo)]
        dn = np.zeros_like(void)
        dn[tuple(sl_lo)] = cur[tuple(sl_hi)]
        nb |= up | dn
        cur = nb & void & (~visited)
        visited |= cur
    return 0.0, False


def specific_surface(r):
    """Surface area per unit SOLID volume of a sphere pack: 3 sum r^2 / sum r^3."""
    r = np.asarray(r, dtype=float)
    denom = float(np.sum(r ** 3))
    return float(3.0 * np.sum(r ** 2) / denom) if denom > 0 else 0.0


def katz_thompson(lc_radius, eps, tau):
    """k = d_c^2 / 226 * (phi / tau)   [length^2], d_c = 2 * critical radius."""
    if tau <= 0:
        tau = 1.0
    return (2.0 * lc_radius) ** 2 / KT_CONSTANT * (eps / tau)


def kozeny_carman_Sv(eps, Sv):
    """Kozeny-Carman in the specific-surface form: eps^3 / (c0 Sv^2 (1-eps)^2).

    Stated on the SAME eps and Sv as Katz-Thompson so the only difference
    between the two is the channelization term -- which is the point.
    """
    if eps <= 0 or eps >= 1 or Sv <= 0:
        return 0.0
    return eps ** 3 / (KC_CONSTANT * Sv ** 2 * (1.0 - eps) ** 2)


def pore_metrics(gs, p, grid_shape, periodic):
    """The pore-field metric block. Called by BOTH metrics twins.

    Returns ``{}`` when ``metrics_pore_field`` is off. Computed on the FULL
    grid, deliberately ignoring ``perf_metrics_voronoi_stride``: striding
    changes ndimage connectivity and would move the critical pore radius by far
    more than the stride whitelist tolerates, so these keys stay
    stride-independent instead.
    """
    if not bool(getattr(p, 'metrics_pore_field', False)):
        return {}
    from gels.engine import container_voxel_mask

    n = int(gs.N)
    if n == 0:
        return {}
    dim = 3 if gs.is_3d else 2
    box = (float(p.Lx), float(p.Ly), float(getattr(p, 'Lz', 0.0)))
    pos = gs.positions()[:n]
    r = np.asarray(gs.r[:n], dtype=float)
    shape = tuple(grid_shape)

    sd = signed_field(pos, r, shape, box, dim, periodic=periodic)
    snap = {'r': r, 'z' if dim == 3 else 'y': pos[:, dim - 1]}
    mask = container_voxel_mask(p, shape, snap=snap)
    if mask is not None:
        sd = np.where(mask, sd, -np.inf)

    axis = dim - 1                       # z is up in 3D, y in 2D
    eps, lc = critical_pore(sd, mask, axis)
    tau, spans = tortuosity_geo(sd, mask, axis, periodic)
    Sv = specific_surface(r)
    v = sd[(sd > 0) & (mask if mask is not None else True)]
    return {
        'porosity_pore_field': float(eps),
        'pore_r_mean': float(v.mean()) if v.size else 0.0,
        'pore_r_median': float(np.median(v)) if v.size else 0.0,
        'pore_r_p90': float(np.percentile(v, 90)) if v.size else 0.0,
        'pore_lc_radius': float(lc),
        'tortuosity_geo': float(tau),          # 0.0, never NaN: history is JSON/CSV
        'tortuosity_spans': bool(spans),
        'specific_surface': Sv,
        'K_katz_thompson': float(katz_thompson(lc, eps, tau if spans else 1.0)),
        'K_kc_analytic': float(kozeny_carman_Sv(eps, Sv)),
    }
