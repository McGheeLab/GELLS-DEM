#!/usr/bin/env python3
"""
Deep Image Analysis of Segmented Experimental Cross-Sections
=============================================================
V2.4 — Extracts granule-level morphology, contact network, interface lengths,
void geometry, and spatial statistics from segmented confocal images.

Input: RGB masks where red=functional, green=inert, black=void.

Extracted metrics per image:
  1. Granule morphology: count, area, equivalent radius, aspect ratio,
     circularity, orientation — per phase and overall.
  2. Contact network: coordination number, phase-pair contact counts
     (FF, FI, II), contact angles.
  3. Interface lengths: total boundary length between each phase pair.
  4. Void geometry: channel width distribution (distance transform),
     tortuosity estimate, percolation (does void span the domain?).
  5. Spatial statistics: nearest-neighbor distances, pair correlation.

Usage:
    python analysis/image_analysis.py
    python analysis/image_analysis.py -o results/image_analysis
    python analysis/image_analysis.py --image "Experimental Results/firstz/cropped_day1_LFLI02.ome_Z00_mask.png"

Or import:
    from analysis.image_analysis import analyze_image, analyze_all, load_experimental_data
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree


# ======================================================================
# Image Classification
# ======================================================================

def classify_pixels(img_rgb):
    """Classify RGB image into functional (red), inert (green), void (black).

    Returns
    -------
    labels : ndarray of uint8
        0=void, 1=functional (red), 2=inert (green), 3=ambiguous
    """
    r, g, b = img_rgb[:,:,0], img_rgb[:,:,1], img_rgb[:,:,2]
    labels = np.full(r.shape, 3, dtype=np.uint8)  # default ambiguous
    labels[(r > 128) & (g < 128) & (b < 128)] = 1  # red = functional
    labels[(g > 128) & (r < 128) & (b < 128)] = 2  # green = inert
    labels[(r < 64) & (g < 64) & (b < 64)] = 0     # black = void
    return labels


# ======================================================================
# Granule Morphology
# ======================================================================

def extract_granules(labels, min_area_px=50):
    """Extract individual granule properties from labeled image.

    Uses connected-component analysis on each phase separately.
    Optimized: pre-filters by area and uses find_objects for local crops.

    Parameters
    ----------
    labels : ndarray
        Pixel labels (1=functional, 2=inert).
    min_area_px : int
        Minimum granule area in pixels (filters noise).

    Returns
    -------
    list of dict
        Per-granule properties: phase, area_px, centroid, equiv_radius_px,
        aspect_ratio, circularity, orientation_deg, bbox, solidity.
    """
    from scipy.ndimage import label as ndlabel, find_objects, sum as ndsum

    granules = []

    for phase_val, phase_name in [(1, 'functional'), (2, 'inert')]:
        mask = (labels == phase_val)
        labeled, n_components = ndlabel(mask)
        if n_components == 0:
            continue

        # Pre-compute areas for ALL components at once (fast)
        comp_ids = np.arange(1, n_components + 1)
        areas = ndsum(mask, labeled, comp_ids)
        areas = np.asarray(areas)

        # Get bounding box slices for all components
        slices = find_objects(labeled)

        # Only process components above min area
        big_mask = areas >= min_area_px
        big_ids = comp_ids[big_mask]

        for comp_id in big_ids:
            sl = slices[comp_id - 1]
            if sl is None:
                continue

            # Work on local crop (much faster than full image)
            local_labeled = labeled[sl]
            local_mask = (local_labeled == comp_id)
            area_px = int(areas[comp_id - 1])

            # Bounding box in global coords
            y0, x0 = sl[0].start, sl[1].start
            bbox_h_local = sl[0].stop - sl[0].start
            bbox_w_local = sl[1].stop - sl[1].start

            # Local pixel coordinates
            local_ys, local_xs = np.where(local_mask)

            # Global centroid
            cx = float(np.mean(local_xs)) + x0
            cy = float(np.mean(local_ys)) + y0

            equiv_radius = np.sqrt(area_px / np.pi)

            bbox = (x0, y0, x0 + bbox_w_local - 1, y0 + bbox_h_local - 1)
            aspect_ratio = max(bbox_w_local, bbox_h_local) / max(min(bbox_w_local, bbox_h_local), 1)

            # Perimeter (count boundary pixels on local crop)
            eroded = ndimage.binary_erosion(local_mask)
            perimeter_px = int(np.sum(local_mask & ~eroded))
            circularity = 4.0 * np.pi * area_px / max(perimeter_px**2, 1)
            circularity = min(circularity, 1.0)

            # Orientation from image moments (local coords, centroid-relative)
            yc = local_ys - np.mean(local_ys)
            xc = local_xs - np.mean(local_xs)
            mu20 = np.sum(xc**2) / area_px
            mu02 = np.sum(yc**2) / area_px
            mu11 = np.sum(xc * yc) / area_px
            theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
            orientation_deg = np.degrees(theta)

            solidity = area_px / max(bbox_w_local * bbox_h_local, 1)

            lam1 = 0.5 * ((mu20 + mu02) + np.sqrt(4*mu11**2 + (mu20 - mu02)**2))
            lam2 = 0.5 * ((mu20 + mu02) - np.sqrt(4*mu11**2 + (mu20 - mu02)**2))
            moment_ar = np.sqrt(max(lam1, 1e-6) / max(lam2, 1e-6))

            granules.append({
                'phase': phase_name,
                'phase_val': phase_val,
                'area_px': area_px,
                'centroid_x': float(cx),
                'centroid_y': float(cy),
                'equiv_radius_px': float(equiv_radius),
                'aspect_ratio_bbox': float(aspect_ratio),
                'aspect_ratio_moment': float(moment_ar),
                'circularity': float(circularity),
                'orientation_deg': float(orientation_deg),
                'solidity': float(solidity),
                'perimeter_px': perimeter_px,
                'bbox': bbox,
                'comp_id': int(comp_id),
            })

    return granules


# ======================================================================
# Contact Network
# ======================================================================

def analyze_contacts(labels, granules, contact_distance_px=3):
    """Analyze granule-granule contacts from the segmented image.

    Two granules are "in contact" if their boundaries are within
    contact_distance_px pixels of each other.

    Returns
    -------
    dict with keys:
        contacts: list of (i, j, phase_i, phase_j) tuples
        coordination: dict mapping granule index to number of contacts
        contact_counts: dict with FF, FI, II counts
        mean_coordination: float
    """
    if len(granules) < 2:
        return {'contacts': [], 'coordination': {}, 'contact_counts': {'FF': 0, 'FI': 0, 'II': 0},
                'mean_coordination': 0.0}

    # Build centroid tree for quick neighbor lookup
    centroids = np.array([[g['centroid_x'], g['centroid_y']] for g in granules])
    radii = np.array([g['equiv_radius_px'] for g in granules])

    # Maximum possible contact distance
    max_r = 2 * np.max(radii) + contact_distance_px
    tree = cKDTree(centroids)

    contacts = []
    coordination = defaultdict(int)

    for i in range(len(granules)):
        # Find nearby granules
        neighbors = tree.query_ball_point(centroids[i], max_r)
        for j in neighbors:
            if j <= i:
                continue
            # Check if granule boundaries are close enough
            dist = np.linalg.norm(centroids[i] - centroids[j])
            gap = dist - radii[i] - radii[j]
            if gap < contact_distance_px:
                pi = granules[i]['phase']
                pj = granules[j]['phase']
                contacts.append((i, j, pi, pj))
                coordination[i] += 1
                coordination[j] += 1

    # Count contact types
    cc = {'FF': 0, 'FI': 0, 'II': 0}
    for _, _, pi, pj in contacts:
        key = ''.join(sorted([pi[0].upper(), pj[0].upper()]))
        cc[key] = cc.get(key, 0) + 1

    coord_vals = list(coordination.values()) if coordination else [0]
    mean_coord = np.mean(coord_vals) if coord_vals else 0.0

    return {
        'contacts': contacts,
        'coordination': dict(coordination),
        'contact_counts': cc,
        'mean_coordination': float(mean_coord),
        'coordination_values': coord_vals,
    }


# ======================================================================
# Interface Lengths
# ======================================================================

def compute_interface_lengths(labels):
    """Compute total boundary length between each phase pair.

    Uses pixel neighbor counting (4-connectivity).

    Returns
    -------
    dict with keys FV, IV, FI (boundary pixel counts) and total.
    """
    h, w = labels.shape

    # Horizontal boundaries
    h_diff_r = labels[:, :-1] != labels[:, 1:]
    # Vertical boundaries
    v_diff_d = labels[:-1, :] != labels[1:, :]

    interfaces = {'FV': 0, 'IV': 0, 'FI': 0}

    # Horizontal
    for y in range(h):
        for x in range(w - 1):
            if h_diff_r[y, x]:
                a, b = int(labels[y, x]), int(labels[y, x+1])
                key = _interface_key(a, b)
                if key:
                    interfaces[key] += 1

    # Vertical
    for y in range(h - 1):
        for x in range(w):
            if v_diff_d[y, x]:
                a, b = int(labels[y, x]), int(labels[y+1, x])
                key = _interface_key(a, b)
                if key:
                    interfaces[key] += 1

    interfaces['total'] = sum(interfaces.values())
    return interfaces


def compute_interface_lengths_fast(labels):
    """Fast vectorized interface length computation."""
    # Horizontal neighbors
    h_left = labels[:, :-1]
    h_right = labels[:, 1:]
    # Vertical neighbors
    v_top = labels[:-1, :]
    v_bot = labels[1:, :]

    interfaces = {}

    for key, a, b in [('FV', 1, 0), ('IV', 2, 0), ('FI', 1, 2)]:
        h_count = np.sum(((h_left == a) & (h_right == b)) |
                         ((h_left == b) & (h_right == a)))
        v_count = np.sum(((v_top == a) & (v_bot == b)) |
                         ((v_top == b) & (v_bot == a)))
        interfaces[key] = int(h_count + v_count)

    interfaces['total'] = interfaces['FV'] + interfaces['IV'] + interfaces['FI']

    # Normalize by image perimeter for scale-free metric
    h, w = labels.shape
    domain_perimeter = 2 * (h + w)
    interfaces['FV_norm'] = interfaces['FV'] / domain_perimeter
    interfaces['IV_norm'] = interfaces['IV'] / domain_perimeter
    interfaces['FI_norm'] = interfaces['FI'] / domain_perimeter
    interfaces['total_norm'] = interfaces['total'] / domain_perimeter

    return interfaces


def _interface_key(a, b):
    """Map label pair to interface key."""
    pair = tuple(sorted([a, b]))
    mapping = {(0, 1): 'FV', (0, 2): 'IV', (1, 2): 'FI'}
    return mapping.get(pair)


# ======================================================================
# Void Geometry
# ======================================================================

def analyze_void_geometry(labels):
    """Analyze void phase geometry: channel widths, tortuosity, percolation.

    Returns
    -------
    dict with:
        distance_transform: array (for visualization)
        channel_widths: array of local channel half-widths (distance to nearest solid)
        mean_channel_width: float
        median_channel_width: float
        max_channel_width: float
        percolates_x: bool (void spans left to right)
        percolates_y: bool (void spans top to bottom)
        void_euler_number: int (connected components - holes)
        n_void_components: int
    """
    void_mask = (labels == 0)
    solid_mask = (labels == 1) | (labels == 2)

    # Distance transform: distance from each void pixel to nearest solid
    if np.any(solid_mask):
        dist = ndimage.distance_transform_edt(void_mask)
    else:
        dist = np.zeros_like(void_mask, dtype=float)

    # Channel widths = distance values at void pixels
    void_dists = dist[void_mask]

    # Connected components
    void_labeled, n_void = ndimage.label(void_mask)

    # Percolation: does void span the domain?
    h, w = labels.shape
    labels_left = set(void_labeled[:, 0].ravel()) - {0}
    labels_right = set(void_labeled[:, -1].ravel()) - {0}
    percolates_x = bool(labels_left & labels_right)

    labels_top = set(void_labeled[0, :].ravel()) - {0}
    labels_bot = set(void_labeled[-1, :].ravel()) - {0}
    percolates_y = bool(labels_top & labels_bot)

    # Euler number (topology)
    try:
        from skimage.measure import euler_number
        euler = int(euler_number(void_mask))
    except ImportError:
        euler = n_void  # approximate

    result = {
        'distance_transform': dist,
        'mean_channel_width': float(np.mean(void_dists)) if len(void_dists) > 0 else 0.0,
        'median_channel_width': float(np.median(void_dists)) if len(void_dists) > 0 else 0.0,
        'max_channel_width': float(np.max(void_dists)) if len(void_dists) > 0 else 0.0,
        'std_channel_width': float(np.std(void_dists)) if len(void_dists) > 0 else 0.0,
        'channel_width_distribution': void_dists,
        'percolates_x': percolates_x,
        'percolates_y': percolates_y,
        'n_void_components': n_void,
        'void_euler_number': euler,
    }

    return result


# ======================================================================
# Spatial Statistics
# ======================================================================

def compute_spatial_statistics(granules):
    """Nearest-neighbor distances and pair correlation by phase.

    Returns
    -------
    dict with nn_distances (per phase), pair_correlation data.
    """
    if len(granules) < 2:
        return {'nn_all': [], 'nn_func': [], 'nn_inert': [],
                'nn_cross': [], 'mean_nn_all': 0.0}

    centroids = np.array([[g['centroid_x'], g['centroid_y']] for g in granules])
    phases = np.array([g['phase_val'] for g in granules])
    tree = cKDTree(centroids)

    # All-to-all nearest neighbor
    dists_all, _ = tree.query(centroids, k=2)  # k=2: self + nearest
    nn_all = dists_all[:, 1]  # exclude self

    # Phase-specific nearest neighbors
    func_idx = np.where(phases == 1)[0]
    inert_idx = np.where(phases == 2)[0]

    nn_func = []
    nn_inert = []
    nn_cross = []  # nearest neighbor of different phase

    if len(func_idx) > 1:
        func_tree = cKDTree(centroids[func_idx])
        d, _ = func_tree.query(centroids[func_idx], k=2)
        nn_func = d[:, 1].tolist()

    if len(inert_idx) > 1:
        inert_tree = cKDTree(centroids[inert_idx])
        d, _ = inert_tree.query(centroids[inert_idx], k=2)
        nn_inert = d[:, 1].tolist()

    # Cross-phase: nearest inert for each functional (and vice versa)
    if len(func_idx) > 0 and len(inert_idx) > 0:
        inert_tree = cKDTree(centroids[inert_idx])
        d_fi, _ = inert_tree.query(centroids[func_idx], k=1)
        func_tree = cKDTree(centroids[func_idx])
        d_if, _ = func_tree.query(centroids[inert_idx], k=1)
        nn_cross = np.concatenate([d_fi.ravel(), d_if.ravel()]).tolist()

    return {
        'nn_all': nn_all.tolist(),
        'nn_func': nn_func,
        'nn_inert': nn_inert,
        'nn_cross': nn_cross,
        'mean_nn_all': float(np.mean(nn_all)),
        'mean_nn_func': float(np.mean(nn_func)) if nn_func else 0.0,
        'mean_nn_inert': float(np.mean(nn_inert)) if nn_inert else 0.0,
        'mean_nn_cross': float(np.mean(nn_cross)) if nn_cross else 0.0,
    }


# ======================================================================
# Master Analysis Function
# ======================================================================

def analyze_image(img_path, min_area_px=50, contact_distance_px=5):
    """Run full analysis pipeline on a single segmented image.

    Parameters
    ----------
    img_path : str or Path
        Path to RGB segmented image.
    min_area_px : int
        Minimum granule area in pixels.
    contact_distance_px : int
        Maximum gap (px) to count as contact.

    Returns
    -------
    dict with all extracted metrics.
    """
    img_path = Path(img_path)
    img = np.array(Image.open(img_path).convert('RGB'))
    h, w, _ = img.shape

    # 1. Classify
    labels = classify_pixels(img)

    # 2. Area fractions
    total_px = h * w
    phi_f = np.sum(labels == 1) / total_px
    phi_i = np.sum(labels == 2) / total_px
    phi_v = np.sum(labels == 0) / total_px

    # 3. Granule morphology
    granules = extract_granules(labels, min_area_px=min_area_px)
    func_granules = [g for g in granules if g['phase'] == 'functional']
    inert_granules = [g for g in granules if g['phase'] == 'inert']

    # 4. Contacts
    contact_info = analyze_contacts(labels, granules, contact_distance_px)

    # 5. Interfaces
    interfaces = compute_interface_lengths_fast(labels)

    # 6. Void geometry
    void_info = analyze_void_geometry(labels)

    # 7. Spatial statistics
    spatial = compute_spatial_statistics(granules)

    # Compile summary
    def _phase_stats(glist, name):
        if not glist:
            return {f'n_{name}': 0}
        areas = [g['area_px'] for g in glist]
        radii = [g['equiv_radius_px'] for g in glist]
        ars = [g['aspect_ratio_moment'] for g in glist]
        circs = [g['circularity'] for g in glist]
        return {
            f'n_{name}': len(glist),
            f'area_mean_{name}': float(np.mean(areas)),
            f'area_std_{name}': float(np.std(areas)),
            f'radius_mean_{name}': float(np.mean(radii)),
            f'radius_std_{name}': float(np.std(radii)),
            f'radius_min_{name}': float(np.min(radii)),
            f'radius_max_{name}': float(np.max(radii)),
            f'ar_mean_{name}': float(np.mean(ars)),
            f'ar_std_{name}': float(np.std(ars)),
            f'circularity_mean_{name}': float(np.mean(circs)),
            f'circularity_std_{name}': float(np.std(circs)),
        }

    result = {
        'filename': img_path.name,
        'image_size': (w, h),
        'phi_f': float(phi_f),
        'phi_i': float(phi_i),
        'phi_v': float(phi_v),
        'n_granules': len(granules),
    }
    result.update(_phase_stats(func_granules, 'func'))
    result.update(_phase_stats(inert_granules, 'inert'))
    result.update({
        'mean_coordination': contact_info['mean_coordination'],
        'contact_FF': contact_info['contact_counts'].get('FF', 0),
        'contact_FI': contact_info['contact_counts'].get('FI', 0),
        'contact_II': contact_info['contact_counts'].get('II', 0),
        'coordination_values': contact_info.get('coordination_values', []),
    })
    result.update({
        'interface_FV': interfaces['FV'],
        'interface_IV': interfaces['IV'],
        'interface_FI': interfaces['FI'],
        'interface_total': interfaces['total'],
        'interface_FV_norm': interfaces['FV_norm'],
        'interface_IV_norm': interfaces['IV_norm'],
        'interface_FI_norm': interfaces['FI_norm'],
    })
    result.update({
        'void_mean_width': void_info['mean_channel_width'],
        'void_median_width': void_info['median_channel_width'],
        'void_max_width': void_info['max_channel_width'],
        'void_std_width': void_info['std_channel_width'],
        'void_percolates_x': void_info['percolates_x'],
        'void_percolates_y': void_info['percolates_y'],
        'n_void_components': void_info['n_void_components'],
    })
    result.update({
        'nn_mean_all': spatial['mean_nn_all'],
        'nn_mean_func': spatial['mean_nn_func'],
        'nn_mean_inert': spatial['mean_nn_inert'],
        'nn_mean_cross': spatial['mean_nn_cross'],
    })

    # Keep raw data for plotting
    result['_granules'] = granules
    result['_void_widths'] = void_info['channel_width_distribution']
    result['_distance_transform'] = void_info['distance_transform']
    result['_labels'] = labels
    result['_nn'] = spatial

    return result


# ======================================================================
# Batch Analysis
# ======================================================================

def analyze_all(img_dir=None, outdir=None, min_area_px=50):
    """Analyze all images in the firstz folder.

    Returns list of result dicts and saves summary CSV + plots.
    """
    if img_dir is None:
        img_dir = Path(__file__).parent.parent / 'Experimental Results' / 'firstz'
    img_dir = Path(img_dir)

    if outdir is None:
        outdir = Path(__file__).parent.parent / 'results' / 'image_analysis'
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(img_dir.glob('*.png'))
    print(f"Found {len(img_files)} images in {img_dir}")

    # Also load Excel data for cross-referencing
    excel_data = _load_excel_lookup()

    results = []
    for i, img_path in enumerate(img_files):
        print(f"  [{i+1}/{len(img_files)}] {img_path.name}...", end=' ')
        r = analyze_image(img_path, min_area_px=min_area_px)

        # Cross-reference with Excel
        fname = img_path.name
        if fname in excel_data:
            xls = excel_data[fname]
            r['size_combo'] = xls['size_combo']
            r['func_ratio'] = xls['ratio']
            r['func_size'] = xls['size_combo'].split('_')[0][0]
            r['inert_size'] = xls['size_combo'].split('_')[1][0]

        perc_str = ('X' if r['void_percolates_x'] else '-') + ('Y' if r['void_percolates_y'] else '-')
        print(f"n_gran={r['n_granules']} "
              f"(F:{r.get('n_func',0)} I:{r.get('n_inert',0)}) "
              f"coord={r['mean_coordination']:.1f} "
              f"void_w={r['void_mean_width']:.1f}px "
              f"perc={perc_str}")
        results.append(r)

    # Save scalar summary as JSON (exclude large arrays)
    _save_summary(results, outdir)

    # Generate plots
    print("\n  Generating plots...")
    _plot_granule_sizes(results, outdir)
    _plot_coordination(results, outdir)
    _plot_interfaces(results, outdir)
    _plot_void_geometry(results, outdir)
    _plot_spatial(results, outdir)
    _plot_overview_grid(results, outdir)

    print(f"\n  Results saved to {outdir}/")
    return results


def _load_excel_lookup():
    """Load Excel data as filename-keyed dict."""
    try:
        import openpyxl
        xlsx = Path(__file__).parent.parent / 'Experimental Results' / 'Day1 analysis results.xlsx'
        wb = openpyxl.load_workbook(str(xlsx), data_only=True)
        ws = wb['Sheet1']
        data = {}
        for row in ws.iter_rows(min_row=23, max_row=58):
            vals = [c.value for c in row[:11]]
            if vals[0] is None:
                continue
            data[str(vals[0])] = {
                'size_combo': str(vals[1]),
                'ratio': float(vals[2]),
                'phi_f': float(vals[3]),
                'phi_i': float(vals[4]),
                'phi_v': float(vals[5]),
            }
        wb.close()
        return data
    except Exception:
        return {}


def _save_summary(results, outdir):
    """Save scalar metrics as JSON."""
    summary = []
    for r in results:
        s = {k: v for k, v in r.items() if not k.startswith('_')}
        # Convert numpy types
        for k, v in s.items():
            if isinstance(v, (np.integer,)):
                s[k] = int(v)
            elif isinstance(v, (np.floating,)):
                s[k] = float(v)
            elif isinstance(v, np.ndarray):
                s[k] = v.tolist()
            elif isinstance(v, (np.bool_,)):
                s[k] = bool(v)
        summary.append(s)

    with open(outdir / 'image_analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


# ======================================================================
# Plotting
# ======================================================================

SIZE_COMBO_COLORS = {
    'LF_LI': '#1f77b4', 'LF_MI': '#2ca02c', 'LF_SI': '#d62728',
    'MF_LI': '#9467bd', 'MF_MI': '#8c564b', 'MF_SI': '#e377c2',
    'SF_LI': '#7f7f7f', 'SF_MI': '#bcbd22', 'SF_SI': '#17becf',
}
RATIO_COLORS = {0.2: '#1f77b4', 0.3: '#ff7f0e', 0.5: '#2ca02c', 0.7: '#d62728'}


def _plot_granule_sizes(results, outdir):
    """Granule size distributions by phase and condition."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # Collect all granule radii by phase and func_size/inert_size
    func_radii_by_size = defaultdict(list)
    inert_radii_by_size = defaultdict(list)
    all_func_radii = []
    all_inert_radii = []

    for r in results:
        fs = r.get('func_size', '?')
        isz = r.get('inert_size', '?')
        for g in r['_granules']:
            if g['phase'] == 'functional':
                func_radii_by_size[fs].append(g['equiv_radius_px'])
                all_func_radii.append(g['equiv_radius_px'])
            else:
                inert_radii_by_size[isz].append(g['equiv_radius_px'])
                all_inert_radii.append(g['equiv_radius_px'])

    # Top-left: overall size distributions
    ax = axes[0, 0]
    if all_func_radii:
        ax.hist(all_func_radii, bins=30, alpha=0.6, color='red', label='Functional', density=True)
    if all_inert_radii:
        ax.hist(all_inert_radii, bins=30, alpha=0.6, color='green', label='Inert', density=True)
    ax.set_xlabel('Equivalent Radius (px)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Overall Granule Size Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: functional by size code
    ax = axes[0, 1]
    for sz in ['S', 'M', 'L']:
        if func_radii_by_size[sz]:
            ax.hist(func_radii_by_size[sz], bins=20, alpha=0.5,
                    label=f'{sz} (n={len(func_radii_by_size[sz])}, '
                          f'mean={np.mean(func_radii_by_size[sz]):.1f})',
                    density=True)
    ax.set_xlabel('Functional Granule Radius (px)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Functional Granule Size by Size Code', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-left: inert by size code
    ax = axes[1, 0]
    for sz in ['S', 'M', 'L']:
        if inert_radii_by_size[sz]:
            ax.hist(inert_radii_by_size[sz], bins=20, alpha=0.5,
                    label=f'{sz} (n={len(inert_radii_by_size[sz])}, '
                          f'mean={np.mean(inert_radii_by_size[sz]):.1f})',
                    density=True)
    ax.set_xlabel('Inert Granule Radius (px)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Inert Granule Size by Size Code', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: aspect ratio distributions
    ax = axes[1, 1]
    func_ar = [g['aspect_ratio_moment'] for r in results for g in r['_granules'] if g['phase'] == 'functional']
    inert_ar = [g['aspect_ratio_moment'] for r in results for g in r['_granules'] if g['phase'] == 'inert']
    if func_ar:
        ax.hist(func_ar, bins=30, alpha=0.6, color='red', label=f'Functional (mean={np.mean(func_ar):.2f})', density=True)
    if inert_ar:
        ax.hist(inert_ar, bins=30, alpha=0.6, color='green', label=f'Inert (mean={np.mean(inert_ar):.2f})', density=True)
    ax.set_xlabel('Aspect Ratio (from moments)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Granule Aspect Ratio Distributions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Granule Morphology from Segmented Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'granule_sizes.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_coordination(results, outdir):
    """Coordination number analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: mean coordination vs func_ratio
    ax = axes[0]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['func_ratio'] for r in sub]
            y = [r['mean_coordination'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Mean Coordination Number', fontsize=11)
    ax.set_title('Coordination vs Ratio', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Middle: contact type distribution
    ax = axes[1]
    ff = [r['contact_FF'] for r in results]
    fi = [r['contact_FI'] for r in results]
    ii = [r['contact_II'] for r in results]
    ratios = [r.get('func_ratio', 0.5) for r in results]
    total_c = [f + fi_ + i for f, fi_, i in zip(ff, fi, ii)]
    ff_frac = [f/max(t, 1) for f, t in zip(ff, total_c)]
    fi_frac = [f/max(t, 1) for f, t in zip(fi, total_c)]
    ii_frac = [f/max(t, 1) for f, t in zip(ii, total_c)]

    ax.scatter(ratios, ff_frac, c='red', s=50, alpha=0.7, label='F-F')
    ax.scatter(ratios, fi_frac, c='orange', s=50, alpha=0.7, label='F-I')
    ax.scatter(ratios, ii_frac, c='green', s=50, alpha=0.7, label='I-I')
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Fraction of Contacts', fontsize=11)
    ax.set_title('Contact Type Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: coordination histogram
    ax = axes[2]
    all_coord = []
    for r in results:
        all_coord.extend(r.get('coordination_values', []))
    if all_coord:
        ax.hist(all_coord, bins=range(0, max(all_coord)+2), alpha=0.7,
                color='steelblue', edgecolor='k', density=True)
        ax.axvline(np.mean(all_coord), color='red', ls='--', lw=2,
                    label=f'mean={np.mean(all_coord):.1f}')
    ax.set_xlabel('Coordination Number', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Coordination Number Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Contact Network Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'coordination.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_interfaces(results, outdir):
    """Interface length analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (key, label, color) in zip(axes, [
        ('interface_FV_norm', 'Functional-Void', 'red'),
        ('interface_IV_norm', 'Inert-Void', 'green'),
        ('interface_FI_norm', 'Functional-Inert', 'orange'),
    ]):
        for combo, cc in SIZE_COMBO_COLORS.items():
            sub = [r for r in results if r.get('size_combo') == combo]
            if sub:
                x = [r['func_ratio'] for r in sub]
                y = [r[key] for r in sub]
                ax.scatter(x, y, c=cc, s=70, alpha=0.8, label=combo,
                           edgecolors='k', linewidths=0.5)
        ax.set_xlabel('Functional Ratio', fontsize=11)
        ax.set_ylabel(f'{label} Interface (norm)', fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Normalized Interface Lengths Between Phases', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'interfaces.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_void_geometry(results, outdir):
    """Void channel width and percolation analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: mean void channel width vs ratio
    ax = axes[0]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['func_ratio'] for r in sub]
            y = [r['void_mean_width'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Mean Void Channel Width (px)', fontsize=11)
    ax.set_title('Void Channel Width', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Middle: channel width vs void fraction
    ax = axes[1]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['phi_v'] for r in sub]
            y = [r['void_mean_width'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Void Fraction', fontsize=11)
    ax.set_ylabel('Mean Void Channel Width (px)', fontsize=11)
    ax.set_title('Channel Width vs Void Fraction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: void channel width distribution (aggregated)
    ax = axes[2]
    all_widths = np.concatenate([r['_void_widths'] for r in results if len(r['_void_widths']) > 0])
    if len(all_widths) > 0:
        ax.hist(all_widths, bins=50, alpha=0.7, color='steelblue', edgecolor='k', density=True)
        ax.axvline(np.mean(all_widths), color='red', ls='--', lw=2,
                    label=f'mean={np.mean(all_widths):.1f}px')
        ax.axvline(np.median(all_widths), color='orange', ls='--', lw=2,
                    label=f'median={np.median(all_widths):.1f}px')
    ax.set_xlabel('Void Channel Half-Width (px)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Void Channel Width Distribution (all images)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle('Void Geometry Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'void_geometry.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_spatial(results, outdir):
    """Spatial statistics plots."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: mean NN distance vs ratio
    ax = axes[0]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['func_ratio'] for r in sub]
            y = [r['nn_mean_all'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Mean NN Distance (px)', fontsize=11)
    ax.set_title('Nearest-Neighbor Distance', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Middle: cross-phase NN (how far apart are F and I)
    ax = axes[1]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['func_ratio'] for r in sub]
            y = [r['nn_mean_cross'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Mean Cross-Phase NN Distance (px)', fontsize=11)
    ax.set_title('Cross-Phase Nearest Neighbor', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Right: granule count vs func_ratio
    ax = axes[2]
    for combo, color in SIZE_COMBO_COLORS.items():
        sub = [r for r in results if r.get('size_combo') == combo]
        if sub:
            x = [r['func_ratio'] for r in sub]
            y = [r['n_granules'] for r in sub]
            ax.scatter(x, y, c=color, s=70, alpha=0.8, label=combo,
                       edgecolors='k', linewidths=0.5)
    ax.set_xlabel('Functional Ratio', fontsize=11)
    ax.set_ylabel('Total Granule Count', fontsize=11)
    ax.set_title('Granule Count', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Spatial Statistics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'spatial_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_overview_grid(results, outdir):
    """4-image overview showing analysis overlays on sample images."""
    # Pick 4 representative images spanning conditions
    targets = ['cropped_day1_LFLI05.ome_Z00_mask.png',
               'cropped_day1_SFSI05.ome_Z00_mask.png',
               'cropped_day1_LFSI07.ome_Z00_mask.png',
               'cropped_day1_SFLI02.ome_Z00_mask.png']

    selected = []
    for t in targets:
        for r in results:
            if r['filename'] == t:
                selected.append(r)
                break

    if len(selected) < 2:
        return

    fig, axes = plt.subplots(2, len(selected), figsize=(5*len(selected), 10))

    for col, r in enumerate(selected):
        # Top row: distance transform overlay
        ax = axes[0, col]
        dt = r['_distance_transform']
        labels = r['_labels']
        # Show void distance transform, mask solid as NaN
        dt_vis = dt.copy().astype(float)
        dt_vis[labels != 0] = np.nan
        ax.imshow(labels, cmap='RdYlGn', alpha=0.3)
        im = ax.imshow(dt_vis, cmap='hot', alpha=0.7)
        ax.set_title(f"{r.get('size_combo','?')} r={r.get('func_ratio','?')}\n"
                     f"void_w={r['void_mean_width']:.1f}px", fontsize=9)
        ax.axis('off')

        # Bottom row: granule centroids + contacts
        ax = axes[1, col]
        # Show labels as background
        rgb = np.zeros((*labels.shape, 3), dtype=np.uint8)
        rgb[labels == 1] = [200, 50, 50]
        rgb[labels == 2] = [50, 200, 50]
        rgb[labels == 0] = [30, 30, 30]
        ax.imshow(rgb)

        # Plot centroids
        for g in r['_granules']:
            color = 'yellow' if g['phase'] == 'functional' else 'cyan'
            ax.plot(g['centroid_x'], g['centroid_y'], 'o', color=color,
                    ms=3, mec='white', mew=0.3)

        ax.set_title(f"n={r['n_granules']} coord={r['mean_coordination']:.1f}\n"
                     f"FF:{r['contact_FF']} FI:{r['contact_FI']} II:{r['contact_II']}",
                     fontsize=9)
        ax.axis('off')

    axes[0, 0].set_ylabel('Void Channel Width', fontsize=11)
    axes[1, 0].set_ylabel('Granule Centroids', fontsize=11)

    fig.suptitle('Image Analysis Overview', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(str(outdir / 'overview_grid.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Deep image analysis of segmented experimental cross-sections.')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: results/image_analysis)')
    parser.add_argument('--image', default=None,
                        help='Analyze a single image (prints results)')
    parser.add_argument('--min-area', type=int, default=50,
                        help='Minimum granule area in pixels (default: 50)')
    args = parser.parse_args()

    if args.image:
        r = analyze_image(args.image, min_area_px=args.min_area)
        # Print summary
        print(f"\n  Image: {r['filename']}")
        print(f"  Size: {r['image_size']}")
        print(f"  Phases: φ_f={r['phi_f']:.3f} φ_i={r['phi_i']:.3f} φ_v={r['phi_v']:.3f}")
        print(f"  Granules: {r['n_granules']} (F:{r.get('n_func',0)} I:{r.get('n_inert',0)})")
        if r.get('n_func', 0) > 0:
            print(f"  Functional: R={r['radius_mean_func']:.1f}±{r['radius_std_func']:.1f}px, "
                  f"AR={r['ar_mean_func']:.2f}±{r['ar_std_func']:.2f}, "
                  f"circ={r['circularity_mean_func']:.3f}")
        if r.get('n_inert', 0) > 0:
            print(f"  Inert:      R={r['radius_mean_inert']:.1f}±{r['radius_std_inert']:.1f}px, "
                  f"AR={r['ar_mean_inert']:.2f}±{r['ar_std_inert']:.2f}, "
                  f"circ={r['circularity_mean_inert']:.3f}")
        print(f"  Coordination: {r['mean_coordination']:.1f} "
              f"(FF:{r['contact_FF']} FI:{r['contact_FI']} II:{r['contact_II']})")
        print(f"  Interfaces: FV={r['interface_FV']} IV={r['interface_IV']} FI={r['interface_FI']}")
        print(f"  Void: mean_w={r['void_mean_width']:.1f}px, "
              f"perc_x={r['void_percolates_x']}, perc_y={r['void_percolates_y']}")
        print(f"  NN distances: all={r['nn_mean_all']:.1f}px, cross={r['nn_mean_cross']:.1f}px")
    else:
        analyze_all(outdir=args.outdir, min_area_px=args.min_area)
