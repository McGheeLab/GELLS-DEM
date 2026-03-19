"""
Voronoi Tessellation Engine
============================
Provides two tessellation modes for spatial analysis of functional granule territory:

1. **Center-based**: Standard Voronoi from functional granule centers.
2. **Boundary-based (shrink-wrap)**: Voronoi from points sampled on granule surfaces,
   merged per-granule to tightly wrap the functional phase.

Both modes produce clipped polygons within the simulation domain, enabling
local phase fraction computation and shape factor analysis.
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull, cKDTree
from matplotlib.path import Path as MplPath

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from new_dem_0 import superellipse_polygon_pts


# ======================================================================
# Geometry utilities
# ======================================================================

def polygon_area(verts):
    """Shoelace formula for polygon area. verts is (N, 2) array."""
    if len(verts) < 3:
        return 0.0
    x, y = verts[:, 0], verts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_perimeter(verts):
    """Perimeter of a polygon. verts is (N, 2) array."""
    if len(verts) < 2:
        return 0.0
    diffs = np.diff(np.vstack([verts, verts[:1]]), axis=0)
    return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))


def polygon_centroid(verts):
    """Centroid of a polygon via the signed-area formula. verts is (N, 2)."""
    if len(verts) < 3:
        return np.mean(verts, axis=0) if len(verts) > 0 else np.array([0.0, 0.0])
    x, y = verts[:, 0], verts[:, 1]
    x1 = np.roll(x, -1)
    y1 = np.roll(y, -1)
    cross = x * y1 - x1 * y
    A = 0.5 * np.sum(cross)
    if abs(A) < 1e-30:
        return np.mean(verts, axis=0)
    cx = np.sum((x + x1) * cross) / (6.0 * A)
    cy = np.sum((y + y1) * cross) / (6.0 * A)
    return np.array([cx, cy])


def _clip_edge(poly, edge_start, edge_end):
    """Clip polygon against one infinite edge (left side is inside)."""
    if len(poly) == 0:
        return poly
    clipped = []
    ex = edge_end[0] - edge_start[0]
    ey = edge_end[1] - edge_start[1]

    def _inside(pt):
        return (ex * (pt[1] - edge_start[1]) - ey * (pt[0] - edge_start[0])) >= 0

    def _intersect(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        denom = ex * dy - ey * dx
        if abs(denom) < 1e-15:
            return p1  # parallel, return p1
        t = (ex * (p1[1] - edge_start[1]) - ey * (p1[0] - edge_start[0])) / (-denom)
        return np.array([p1[0] + t * dx, p1[1] + t * dy])

    prev = poly[-1]
    prev_in = _inside(prev)
    for curr in poly:
        curr_in = _inside(curr)
        if curr_in:
            if not prev_in:
                clipped.append(_intersect(prev, curr))
            clipped.append(curr)
        elif prev_in:
            clipped.append(_intersect(prev, curr))
        prev = curr
        prev_in = curr_in
    return clipped


def clip_polygon_to_rect(verts, xmin, ymin, xmax, ymax):
    """Sutherland-Hodgman algorithm to clip a polygon to a rectangle.

    Parameters
    ----------
    verts : ndarray (N, 2)
        Polygon vertices in order.
    xmin, ymin, xmax, ymax : float
        Rectangle bounds.

    Returns
    -------
    ndarray (M, 2) — clipped polygon vertices, or empty (0, 2) if fully outside.
    """
    poly = list(verts)

    # Clip against all 4 edges (defined as directed edges, inside = left)
    # Bottom: left to right
    poly = _clip_edge(poly, np.array([xmin, ymin]), np.array([xmax, ymin]))
    # Right: bottom to top
    poly = _clip_edge(poly, np.array([xmax, ymin]), np.array([xmax, ymax]))
    # Top: right to left
    poly = _clip_edge(poly, np.array([xmax, ymax]), np.array([xmin, ymax]))
    # Left: top to bottom
    poly = _clip_edge(poly, np.array([xmin, ymax]), np.array([xmin, ymin]))

    if len(poly) < 3:
        return np.zeros((0, 2))
    return np.array(poly)


def clip_polygon_to_convex(subject_verts, clip_verts):
    """Sutherland-Hodgman: clip subject polygon by a convex clip polygon.

    Parameters
    ----------
    subject_verts : ndarray (N, 2)
    clip_verts : ndarray (M, 2) — must be convex, ordered CCW.

    Returns
    -------
    ndarray (K, 2) — clipped polygon, or empty (0, 2) if no overlap.
    """
    poly = list(subject_verts)
    n = len(clip_verts)
    for i in range(n):
        edge_start = clip_verts[i]
        edge_end = clip_verts[(i + 1) % n]
        poly = _clip_edge(poly, edge_start, edge_end)
        if len(poly) == 0:
            return np.zeros((0, 2))
    if len(poly) < 3:
        return np.zeros((0, 2))
    return np.array(poly)


# ======================================================================
# Granule polygon helper
# ======================================================================

def _granule_polygon(snap, gi, n_pts=64):
    """Return CCW-ordered boundary polygon for granule gi."""
    xs, ys = snap['x'], snap['y']
    a_arr = snap.get('a', snap['r'])
    b_arr = snap.get('b', snap['r'])
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
    theta_arr = snap.get('theta', np.zeros(len(xs)))
    pts = superellipse_polygon_pts(
        xs[gi], ys[gi], a_arr[gi], b_arr[gi], n1_arr[gi], theta_arr[gi],
        num_pts=n_pts)
    # Ensure CCW ordering (positive signed area)
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    if signed_area < 0:
        pts = pts[::-1]
    return pts


# ======================================================================
# Center-based Voronoi (all granules, functional clipped to body)
# ======================================================================

def voronoi_from_centers(snap, p, func_only=False):
    """Voronoi tessellation from ALL granule centers.

    Functional cells are clipped to their granule boundary so their domain
    equals the granule body. Inert cells keep their full Voronoi region,
    expanding to fill remaining space.

    Parameters
    ----------
    snap : dict
        Snapshot data.
    p : Params
        Simulation parameters.
    func_only : bool
        Legacy flag; default False (all granules).

    Returns
    -------
    polygons : dict
        Maps granule index -> clipped polygon ndarray (N, 2).
    granule_indices : ndarray
        Indices of granules used as seeds.
    """
    xs, ys, gt = snap['x'], snap['y'], snap['gtype']
    indices = np.arange(len(xs))
    if len(indices) < 2:
        return {}, indices

    seeds = np.column_stack([xs, ys])
    Lx, Ly = float(p.Lx), float(p.Ly)

    # Mirror guard points for bounded Voronoi
    guards = []
    for s in seeds:
        guards.append([2 * 0 - s[0], s[1]])       # mirror across x=0
        guards.append([2 * Lx - s[0], s[1]])       # mirror across x=Lx
        guards.append([s[0], 2 * 0 - s[1]])        # mirror across y=0
        guards.append([s[0], 2 * Ly - s[1]])       # mirror across y=Ly
    guards = np.array(guards)

    all_pts = np.vstack([seeds, guards])
    n_seeds = len(seeds)

    vor = Voronoi(all_pts)

    polygons = {}
    for si in range(n_seeds):
        gi = int(indices[si])
        region_idx = vor.point_region[si]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        verts = vor.vertices[region]
        clipped = clip_polygon_to_rect(verts, 0, 0, Lx, Ly)
        if len(clipped) < 3:
            continue

        # Functional granules: clip Voronoi cell to granule boundary
        if gt[gi] == 0:
            gpoly = _granule_polygon(snap, gi, n_pts=64)
            clipped = clip_polygon_to_convex(clipped, gpoly)
            if len(clipped) < 3:
                continue

        polygons[gi] = clipped

    return polygons, indices


# ======================================================================
# Boundary-based Voronoi (shrink-wrap)
# ======================================================================

def voronoi_from_boundaries(snap, p, n_pts=32, func_only=False):
    """Voronoi tessellation from points sampled on ALL granule surfaces.

    Seeds are boundary points on every granule. Sub-cells per granule are
    merged via ConvexHull. Functional cells are then clipped to their granule
    boundary (domain = granule body). Inert cells keep the full merged hull,
    expanding to fill remaining space.

    Guard point spacing: 0.33 × smallest granule radius.

    Parameters
    ----------
    snap : dict
        Snapshot data.
    p : Params
        Simulation parameters.
    n_pts : int
        Number of boundary points sampled per granule.
    func_only : bool
        Legacy flag; default False (all granules).

    Returns
    -------
    polygons : dict
        Maps granule index -> clipped polygon ndarray (N, 2).
    granule_indices : ndarray
        Indices of granules used.
    """
    xs, ys, gt = snap['x'], snap['y'], snap['gtype']
    a_arr = snap.get('a', snap['r'])
    b_arr = snap.get('b', snap['r'])
    n1_arr = snap.get('n1', snap.get('n_shape', np.full(len(xs), 2.0)))
    theta_arr = snap.get('theta', np.zeros(len(xs)))

    indices = np.arange(len(xs))

    if len(indices) < 2:
        return {}, indices

    Lx, Ly = float(p.Lx), float(p.Ly)

    # Step 1: Sample boundary points on each granule
    all_seeds = []
    granule_of_seed = []
    for gi in indices:
        pts = superellipse_polygon_pts(
            xs[gi], ys[gi], a_arr[gi], b_arr[gi], n1_arr[gi], theta_arr[gi],
            num_pts=n_pts)
        all_seeds.append(pts)
        granule_of_seed.extend([int(gi)] * len(pts))

    all_seeds = np.vstack(all_seeds)
    granule_of_seed = np.array(granule_of_seed, dtype=int)

    # Step 2: Add domain boundary guard points (0.33 × smallest granule)
    r_min = float(np.min(snap['r']))
    spacing = 0.33 * r_min
    guards = []
    for x in np.arange(0, Lx + spacing, spacing):
        guards.append([x, -spacing])
        guards.append([x, Ly + spacing])
    for y in np.arange(0, Ly + spacing, spacing):
        guards.append([-spacing, y])
        guards.append([Lx + spacing, y])
    guards = np.array(guards)

    all_pts = np.vstack([all_seeds, guards])
    n_seeds = len(all_seeds)

    # Step 3: Compute Voronoi
    vor = Voronoi(all_pts)

    # Step 4: Collect Voronoi vertices per granule
    granule_verts = {int(gi): [] for gi in indices}
    for si in range(n_seeds):
        gi = granule_of_seed[si]
        region_idx = vor.point_region[si]
        region = vor.regions[region_idx]
        if -1 in region or len(region) < 3:
            continue
        verts = vor.vertices[region]
        granule_verts[gi].append(verts)

    # Step 5: Merge sub-cells per granule via ConvexHull, then clip
    polygons = {}
    for gi in indices:
        sub_verts = granule_verts.get(int(gi), [])
        if not sub_verts:
            continue
        all_v = np.vstack(sub_verts)
        if len(all_v) < 3:
            continue
        try:
            hull = ConvexHull(all_v)
            hull_pts = all_v[hull.vertices]
        except Exception:
            continue

        clipped = clip_polygon_to_rect(hull_pts, 0, 0, Lx, Ly)
        if len(clipped) < 3:
            continue

        # Functional granules: clip to granule body
        if gt[gi] == 0:
            gpoly = _granule_polygon(snap, gi, n_pts=64)
            clipped = clip_polygon_to_convex(clipped, gpoly)
            if len(clipped) < 3:
                continue

        polygons[int(gi)] = clipped

    return polygons, indices


# ======================================================================
# Shape factor computation
# ======================================================================

def compute_shape_factors(polygons):
    """Compute shape descriptors for each Voronoi cell polygon.

    Parameters
    ----------
    polygons : dict
        Maps key -> ndarray (N, 2) polygon vertices.

    Returns
    -------
    dict mapping key -> {area, perimeter, circularity, aspect_ratio, elongation, centroid}
    """
    results = {}
    for key, verts in polygons.items():
        area = polygon_area(verts)
        perim = polygon_perimeter(verts)
        circ = 4.0 * np.pi * area / (perim**2) if perim > 0 else 0.0
        cent = polygon_centroid(verts)

        # PCA for aspect ratio
        centered = verts - cent
        if len(centered) >= 3:
            cov = np.cov(centered.T)
            evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
            evals = np.maximum(evals, 1e-30)
            ar = np.sqrt(evals[0] / evals[1])
        else:
            ar = 1.0

        results[key] = {
            'area': area,
            'perimeter': perim,
            'circularity': circ,
            'aspect_ratio': ar,
            'elongation': 1.0 - 1.0 / max(ar, 1e-10),
            'centroid': cent,
        }
    return results


# ======================================================================
# Local phase fraction computation
# ======================================================================

def compute_local_phase_fractions(snap, p, polygons, phi_f=None, phi_i=None,
                                  phi_v=None):
    """Compute local phase fractions within each Voronoi cell.

    Uses point-in-polygon test on the phase field grid.

    Parameters
    ----------
    snap : dict
        Snapshot data.
    p : Params
        Simulation parameters.
    polygons : dict
        Maps key -> ndarray (N, 2) polygon vertices.
    phi_f, phi_i, phi_v : ndarray or None
        Phase fields. If None, extracted from snap or re-rendered.

    Returns
    -------
    dict mapping key -> {phi_f, phi_i, phi_v, area, centroid}
    """
    if phi_f is None:
        from viz2.common import ensure_phase_fields
        phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)

    Ng = phi_f.shape[0]
    dx = p.Lx / Ng
    dy = p.Ly / Ng
    gx = np.linspace(dx / 2, p.Lx - dx / 2, Ng)
    gy = np.linspace(dy / 2, p.Ly - dy / 2, Ng)
    GX, GY = np.meshgrid(gx, gy, indexing='ij')
    grid_pts = np.column_stack([GX.ravel(), GY.ravel()])

    results = {}
    for key, verts in polygons.items():
        if len(verts) < 3:
            continue

        path = MplPath(verts)
        inside = path.contains_points(grid_pts)
        n_inside = np.sum(inside)

        if n_inside == 0:
            results[key] = {
                'phi_f': 0.0, 'phi_i': 0.0, 'phi_v': 1.0,
                'area': polygon_area(verts),
                'centroid': polygon_centroid(verts),
            }
            continue

        inside_2d = inside.reshape(Ng, Ng)
        pf_local = float(np.mean(phi_f[inside_2d]))
        pi_local = float(np.mean(phi_i[inside_2d]))
        pv_local = float(np.mean(phi_v[inside_2d]))

        results[key] = {
            'phi_f': pf_local,
            'phi_i': pi_local,
            'phi_v': pv_local,
            'area': polygon_area(verts),
            'centroid': polygon_centroid(verts),
        }

    return results
