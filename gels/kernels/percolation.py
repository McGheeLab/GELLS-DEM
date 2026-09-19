"""
Union-find percolation on the REAL granule contact graph (V3.3).
================================================================

``func_lf`` and friends (``metrics._connectivity_full``) threshold the rendered
tanh phase field at ``mean + 0.3*std`` and run ``scipy.ndimage.label``. That
measures *field* continuity at the interface-width scale: it joins granules
separated by up to ~2*``interface_width`` of nothing, and its value moves with
``Ngrid``. This module measures *mechanical* continuity instead -- clusters of
granules that actually touch -- so the two disagree by construction, and their
ratio is a direct readout of the halo inflation CLAUDE.md warns about.

Shared by both metrics twins so they cannot drift apart, the same way
``gels.division.cycle_metrics`` is. Edges come from the caller's own contact
predicate (``overlap > 0``), so there is no second definition of "touching"
and no tolerance to tune -- unlike a post-hoc analysis, which has to guess one.

Periodic boxes use the Newman-Ziff wrapping test: each edge carries the
periodic image it was resolved through, union-find accumulates the image
offset, and a cycle that returns to the same root with a nonzero offset means
the cluster wraps the box. Walled and cylindrical containers have no images,
so they get a floor-to-ceiling spanning test instead.
"""

import numpy as np

from gels.kernels import njit

# A granule with fewer than d+1 contacts cannot be mechanically stable in d
# dimensions, so it carries no load however the graph connects it. Reported
# separately rather than removed from the clusters, because for a phase
# CONTINUITY question a rattler bridged by cells still counts as present.
RATTLER_Z = {2: 3, 3: 4}


@njit(cache=True)
def _label_and_wrap(pair_i, pair_j, img, keep, n, dim):
    """Weighted union-find with image-offset tracking.

    ``img`` is the integer periodic image each edge was resolved through
    (zeros for a non-periodic box). ``keep`` selects the sub-graph. Returns
    ``(root, wrap)``: the root of every node, and per-axis wrapping flags.
    """
    parent = np.arange(n)
    disp = np.zeros((n, dim), dtype=np.int64)
    wrap = np.zeros(dim, dtype=np.bool_)

    for k in range(pair_i.shape[0]):
        a = pair_i[k]
        b = pair_j[k]
        if not (keep[a] and keep[b]):
            continue
        # walk a to its root, accumulating the offset
        ra = a
        da = np.zeros(dim, dtype=np.int64)
        while parent[ra] != ra:
            da = da + disp[ra]
            ra = parent[ra]
        rb = b
        db = np.zeros(dim, dtype=np.int64)
        while parent[rb] != rb:
            db = db + disp[rb]
            rb = parent[rb]
        off = da - db - img[k]
        if ra != rb:
            parent[rb] = ra
            disp[rb] = off
        else:
            # a cycle back to the same root: a nonzero offset means the cluster
            # closes on itself through the boundary, i.e. it spans the box
            for c in range(dim):
                if off[c] != 0:
                    wrap[c] = True

    root = np.empty(n, dtype=np.int64)
    for i in range(n):
        r = i
        while parent[r] != r:
            r = parent[r]
        root[i] = r
    return root, wrap


def _minimum_image(pos, pair_i, pair_j, box, periodic, dim):
    """Integer periodic image each edge is resolved through (zeros if walled)."""
    if not periodic or pair_i.size == 0:
        return np.zeros((pair_i.shape[0], dim), dtype=np.int64)
    d = pos[pair_j, :dim] - pos[pair_i, :dim]
    L = np.asarray(box[:dim], dtype=float)
    return np.rint(d / L).astype(np.int64)


def clusters(pair_i, pair_j, is_contact, n, keep, pos, box, periodic, dim):
    """Cluster labels, sizes and wrapping flags for the sub-graph on ``keep``.

    Only edges with ``is_contact`` participate. Isolated members of ``keep``
    are singleton clusters, so the sizes always sum to ``keep.sum()``.
    """
    keep = np.ascontiguousarray(keep, dtype=np.bool_)
    n_keep = int(keep.sum())
    if n_keep == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(dim, dtype=bool), 0
    sel = np.asarray(is_contact, dtype=bool)
    pi = np.ascontiguousarray(pair_i[sel], dtype=np.int64)
    pj = np.ascontiguousarray(pair_j[sel], dtype=np.int64)
    img = np.ascontiguousarray(_minimum_image(pos, pi, pj, box, periodic, dim))
    root, wrap = _label_and_wrap(pi, pj, img, keep, int(n), int(dim))
    sizes = np.bincount(root[keep])
    return sizes[sizes > 0], np.asarray(wrap), n_keep


def graph_metrics(pair_i, pair_j, is_contact, n, pos, box, periodic, dim,
                  groups, up=None, floor_tol=1.0, ceiling=None):
    """Percolation metrics for each named sub-graph of the contact network.

    ``groups`` maps a key suffix to a boolean granule mask. ``up`` (the height
    coordinate) and ``ceiling`` enable the walled floor-to-ceiling spanning
    test; under periodic boundaries the Newman-Ziff wrapping flags are
    reported instead and spanning is left False.
    """
    out = {}
    sel = np.asarray(is_contact, dtype=bool)
    pi = np.asarray(pair_i)[sel]
    pj = np.asarray(pair_j)[sel]

    # coordination over the whole graph, and the rattler fraction
    z = np.zeros(int(n), dtype=np.int64)
    if pi.size:
        z += np.bincount(pi, minlength=int(n))
        z += np.bincount(pj, minlength=int(n))
    linked = z > 0
    out['gran_z_mean'] = float(z[linked].mean()) if linked.any() else 0.0
    out['gran_rattler_frac'] = (float(np.mean(z < RATTLER_Z[int(dim)]))
                                if n else 0.0)

    for key, mask in groups.items():
        sizes, wrap, n_keep = clusters(pair_i, pair_j, is_contact, n, mask,
                                       pos, box, periodic, dim)
        out[f'gran_nc_{key}'] = int(sizes.size)
        out[f'gran_lf_{key}'] = float(sizes.max() / n_keep) if sizes.size else 0.0

    if periodic:
        axes = ('x', 'y', 'z')[:dim]
        # wrapping of the whole solid graph is the useful global statement
        solid = groups.get('solid')
        if solid is None:
            solid = np.ones(int(n), dtype=bool)
        _, wrap, _ = clusters(pair_i, pair_j, is_contact, n, solid,
                              pos, box, periodic, dim)
        for c, ax in enumerate(axes):
            out[f'gran_wrap_{ax}'] = bool(wrap[c])
        for key in groups:
            out[f'gran_span_{key}'] = False
    else:
        for c, ax in enumerate(('x', 'y', 'z')[:dim]):
            out[f'gran_wrap_{ax}'] = False
        for key, mask in groups.items():
            out[f'gran_span_{key}'] = _spans(pair_i, pair_j, is_contact, n, mask,
                                             pos, box, dim, up, floor_tol, ceiling)
    return out


def _spans(pair_i, pair_j, is_contact, n, keep, pos, box, dim, up, floor_tol, ceiling):
    """True if the largest cluster of ``keep`` reaches both floor and ceiling.

    The ceiling is the BED SURFACE for a free top, not the container height --
    nothing reaches a lid that is not there.
    """
    if up is None or ceiling is None:
        return False
    keep = np.asarray(keep, dtype=bool)
    if not keep.any():
        return False
    sel = np.asarray(is_contact, dtype=bool)
    pi = np.ascontiguousarray(np.asarray(pair_i)[sel], dtype=np.int64)
    pj = np.ascontiguousarray(np.asarray(pair_j)[sel], dtype=np.int64)
    img = np.zeros((pi.shape[0], dim), dtype=np.int64)
    root, _ = _label_and_wrap(pi, pj, img, np.ascontiguousarray(keep), int(n), int(dim))
    at_floor = keep & (up <= floor_tol)
    at_top = keep & (up >= ceiling)
    if not at_floor.any() or not at_top.any():
        return False
    return bool(np.intersect1d(root[at_floor], root[at_top]).size > 0)


def graph_percolation_metrics(gs, p, pos, pair_i, pair_j, is_contact, periodic):
    """The ``gran_*`` metric block, from a GranuleSystem and its pair list.

    Called by BOTH metrics twins so the sub-graph definitions cannot drift.
    ``gels.engine`` is imported inside the function, the same way
    ``gels.kernels.metrics`` imports ``gels.division``, to keep the module
    importable from either twin without a cycle.
    """
    from gels.engine import boundary_geometry

    dim = 3 if gs.is_3d else 2
    n = int(gs.N)
    box = (float(p.Lx), float(p.Ly), float(getattr(p, 'Lz', 0.0)))
    gtype = np.asarray(gs.gtype[:n])
    groups = {
        'func': gtype == 0,
        'inert': gtype == 1,
        'solid': np.ones(n, dtype=bool),
    }

    up = None
    ceiling = None
    if not periodic and n:
        geom = boundary_geometry(p)
        # z is up in 3D, y is up in 2D (V3.1 free-surface convention)
        up_idx = 2 if dim == 3 else 1
        up = np.asarray(pos[:n, up_idx], dtype=float)
        rb = np.asarray(gs.r_bound[:n], dtype=float)
        if geom.top_free:
            # the ceiling is the BED SURFACE -- nothing reaches a lid that is
            # not there. Matches bed_height_p95's definition.
            ceiling = float(np.percentile(up + rb, 95))
        else:
            ceiling = float(box[up_idx] if dim == 3 else box[1])
        # "touching" the boundary means within one radius of it
        up = up.copy()
        ceiling = ceiling - float(np.mean(rb))

    return graph_metrics(pair_i, pair_j, is_contact, n, pos, box, periodic, dim,
                         groups, up=up,
                         floor_tol=float(np.mean(gs.r_bound[:n])) if n else 1.0,
                         ceiling=ceiling)
