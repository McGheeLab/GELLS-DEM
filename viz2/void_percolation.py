"""
Void Percolation Theory (Module 6)
====================================
Void space connectivity analysis: cluster labeling, percolation detection,
cluster size distributions P(s) with power-law fits, Kozeny-Carman permeability.

Usage:
    python viz2/void_percolation.py -i results/default
"""

import numpy as np
from scipy.ndimage import label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from viz2.common import (
    select_indices, get_snap_time, setup_scaffold_axes,
    ensure_phase_fields, draw_granule_patches, save_fig,
    is_3d, slice_snap_z_midplane, slice_field_z_midplane,
)


# ======================================================================
# Cluster analysis
# ======================================================================

def compute_void_clusters(phi_v, threshold=0.3):
    """Label connected void regions above threshold.

    Returns
    -------
    labels : ndarray — integer label array (0=solid, 1..n=clusters)
    n_clusters : int
    sizes : ndarray — cluster sizes in pixels (sorted descending)
    """
    binary = (phi_v > threshold).astype(int)
    labels, n_clusters = label(binary)
    if n_clusters == 0:
        return labels, 0, np.array([])

    sizes = np.array([np.sum(labels == k) for k in range(1, n_clusters + 1)])
    return labels, n_clusters, np.sort(sizes)[::-1]


def detect_percolation(labels, axis=0):
    """Check if any cluster spans the domain along the given axis.

    Works for both 2D and 3D label arrays.
    axis=0: x-direction, axis=1: y-direction, axis=2: z-direction (3D only)

    Returns
    -------
    percolated : bool
    spanning_cluster_ids : list of int
    """
    if labels.max() == 0:
        return False, []

    ndim = labels.ndim
    # Build slice tuples for first and last hyperplane along axis
    first_slice = tuple(slice(None) if d != axis else 0 for d in range(ndim))
    last_slice = tuple(slice(None) if d != axis else -1 for d in range(ndim))

    first_labels = set(np.unique(labels[first_slice])) - {0}
    last_labels = set(np.unique(labels[last_slice])) - {0}

    spanning = first_labels & last_labels
    return len(spanning) > 0, sorted(spanning)


def cluster_size_distribution(labels, n_clusters):
    """Compute P(s) = number of clusters of size s.

    Returns
    -------
    sizes : ndarray — unique cluster sizes
    counts : ndarray — count of clusters with each size
    """
    if n_clusters == 0:
        return np.array([]), np.array([])

    all_sizes = np.array([np.sum(labels == k) for k in range(1, n_clusters + 1)])
    unique_sizes, counts = np.unique(all_sizes, return_counts=True)
    return unique_sizes, counts


def fit_power_law(sizes, counts, s_min=5):
    """Fit P(s) ~ s^(-tau) above s_min via log-log linear regression.

    Returns
    -------
    tau : float — power-law exponent
    r_squared : float — coefficient of determination
    """
    mask = sizes >= s_min
    if np.sum(mask) < 3:
        return 0.0, 0.0

    log_s = np.log10(sizes[mask].astype(float))
    log_c = np.log10(counts[mask].astype(float))

    # Linear regression in log-log space
    A = np.vstack([log_s, np.ones(len(log_s))]).T
    result = np.linalg.lstsq(A, log_c, rcond=None)
    slope, intercept = result[0]

    # R^2
    residuals = log_c - (slope * log_s + intercept)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_c - np.mean(log_c))**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return -slope, r2  # tau = -slope (P(s) ~ s^(-tau))


# ======================================================================
# Plotting: void clusters
# ======================================================================

def plot_void_clusters(snaps, hist, p, indices=None, outdir=None):
    """Labeled void clusters colored by ID overlaid on scaffold map."""
    indices = select_indices(len(snaps), indices)
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(5 * nc, 5))
    if nc == 1:
        axes = [axes]

    _3d = is_3d(snaps[0], p) if snaps else False

    for c, si in enumerate(indices):
        ax = axes[c]
        snap = snaps[si]
        t = get_snap_time(hist, si)

        phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)

        # For display, use z-midplane slice of 3D fields
        draw_snap = snap
        phi_v_2d = phi_v
        if _3d:
            phi_v_2d = slice_field_z_midplane(phi_v)
            draw_snap = slice_snap_z_midplane(snap, p)

        labels, n_clusters, sizes = compute_void_clusters(phi_v_2d)

        # Draw scaffold as background
        setup_scaffold_axes(ax, p, dark=True)
        draw_granule_patches(ax, draw_snap, p)

        # Overlay void clusters with qualitative colormap
        if n_clusters > 0:
            cmap = plt.get_cmap('tab20', max(n_clusters, 1))
            Ng = labels.shape[0]
            rgba = np.zeros((Ng, Ng, 4))
            for k in range(1, n_clusters + 1):
                mask = labels == k
                color = cmap(k % cmap.N)
                rgba[mask, :3] = color[:3]
                rgba[mask, 3] = 0.6

            extent = [0, p.Lx, 0, p.Ly]
            ax.imshow(rgba.transpose(1, 0, 2), origin='lower', extent=extent,
                      interpolation='nearest')

        # Full 3D percolation check for stats
        if _3d:
            labels_3d, nc_3d, _ = compute_void_clusters(phi_v)
            perc_axes = [detect_percolation(labels_3d, axis=a)[0] for a in range(3)]
            perc_str = "PERC " + "/".join(
                d for d, p_ax in zip("xyz", perc_axes) if p_ax) or "disconnected"
            if not any(perc_axes):
                perc_str = "disconnected"
        else:
            perc_x, _ = detect_percolation(labels, axis=0)
            perc_y, _ = detect_percolation(labels, axis=1)
            perc_str = "PERCOLATING" if (perc_x or perc_y) else "disconnected"

        ax.set_title(f"t={t:.1f}h | {n_clusters} clusters | {perc_str}",
                     fontsize=9, color='white')

    fig.patch.set_facecolor('black')
    fig.suptitle("Void Cluster Map", fontsize=12, color='white', y=1.02)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'void_clusters.png')
    return fig


# ======================================================================
# Plotting: percolation evolution
# ======================================================================

def plot_percolation_evolution(snaps, hist, p, outdir=None):
    """Time series: percolation status, largest cluster fraction,
    cluster count, mean cluster size."""
    _3d = is_3d(snaps[0], p) if snaps else False

    times = []
    n_clusters_arr = []
    largest_frac_arr = []
    percolated_arr = []
    mean_size_arr = []

    for si in range(len(snaps)):
        t = get_snap_time(hist, si)
        times.append(t)

        phi_f, phi_i, phi_v = ensure_phase_fields(snaps[si], p)
        labels, nc, sizes = compute_void_clusters(phi_v)

        n_clusters_arr.append(nc)
        total_void = np.sum(phi_v > 0.3)
        largest_frac_arr.append(sizes[0] / max(total_void, 1) if len(sizes) > 0 else 0)
        mean_size_arr.append(np.mean(sizes) if len(sizes) > 0 else 0)

        n_axes = 3 if _3d else 2
        perc_any = any(detect_percolation(labels, axis=a)[0] for a in range(n_axes))
        percolated_arr.append(perc_any)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    t = np.array(times)

    # (a) Percolation status
    ax = axes[0, 0]
    colors = ['#2E7D32' if p else '#C62828' for p in percolated_arr]
    ax.scatter(t, percolated_arr, c=colors, s=40, zorder=3)
    ax.set_ylabel('Percolating?')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])
    ax.set_title('Void Percolation Status')
    ax.grid(True, alpha=0.2)

    # (b) Largest cluster fraction
    ax = axes[0, 1]
    ax.plot(t, largest_frac_arr, '-o', color='#1565C0', lw=2, ms=4)
    ax.set_ylabel('Largest cluster / total void')
    ax.set_title('Largest Void Cluster Fraction')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    # (c) Cluster count
    ax = axes[1, 0]
    ax.plot(t, n_clusters_arr, '-s', color='#C62828', lw=2, ms=4)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Number of clusters')
    ax.set_title('Void Cluster Count')
    ax.grid(True, alpha=0.2)

    # (d) Mean cluster size
    ax = axes[1, 1]
    ax.plot(t, mean_size_arr, '-^', color='#6A1B9A', lw=2, ms=4)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Mean cluster size (pixels)')
    ax.set_title('Mean Void Cluster Size')
    ax.grid(True, alpha=0.2)

    fig.suptitle("Void Percolation Evolution", fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'percolation_evolution.png')
    return fig


# ======================================================================
# Plotting: cluster size distribution
# ======================================================================

def plot_cluster_size_dist(snaps, hist, p, indices=None, outdir=None):
    """Log-log P(s) at selected timepoints with power-law fits."""
    indices = select_indices(len(snaps), indices)
    cmap = plt.get_cmap('viridis')
    n_t = len(indices)

    fig, ax = plt.subplots(figsize=(8, 6))

    for ti, si in enumerate(indices):
        t = get_snap_time(hist, si)
        phi_f, phi_i, phi_v = ensure_phase_fields(snaps[si], p)
        labels, nc, _ = compute_void_clusters(phi_v)
        sizes, counts = cluster_size_distribution(labels, nc)

        if len(sizes) == 0:
            continue

        color = cmap(ti / max(n_t - 1, 1))
        ax.scatter(sizes, counts, c=[color], s=20, alpha=0.7,
                   label=f"t={t:.0f}h")

        # Power-law fit
        tau, r2 = fit_power_law(sizes, counts)
        if r2 > 0.5 and tau > 0:
            s_fit = np.linspace(5, sizes.max(), 100)
            c_fit = 10**(np.log10(counts.max()) - tau * np.log10(s_fit / sizes.min()))
            ax.plot(s_fit, c_fit, '--', color=color, alpha=0.5, lw=1)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Cluster size s (pixels)')
    ax.set_ylabel('P(s)')
    ax.set_title('Void Cluster Size Distribution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, which='both')

    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'cluster_size_distribution.png')
    return fig


# ======================================================================
# Plotting: Kozeny-Carman permeability
# ======================================================================

def plot_kozeny_carman(hist, p, outdir=None):
    """K(t) Kozeny-Carman permeability with percolation annotations."""
    times = [h['time'] for h in hist]
    K_vals = [h.get('K_kozeny_carman', 0) for h in hist]
    porosity = [h.get('porosity', h.get('phi_v_mean', 0)) for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (a) Permeability
    ax = axes[0]
    ax.plot(times, K_vals, '-o', color='#1565C0', lw=2, ms=4)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$K$ ($\mu m^2$)')
    ax.set_title('Kozeny-Carman Permeability')
    ax.grid(True, alpha=0.2)
    if any(k > 0 for k in K_vals):
        ax.set_yscale('log')

    # (b) Porosity
    ax = axes[1]
    ax.plot(times, porosity, '-s', color='#2E7D32', lw=2, ms=4)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Porosity')
    ax.set_title('Porosity Evolution')
    ax.grid(True, alpha=0.2)

    fig.suptitle("Void Transport Properties", fontsize=13)
    plt.tight_layout()
    if outdir:
        save_fig(fig, outdir, 'kozeny_carman.png')
    return fig


# ======================================================================
# run_all
# ======================================================================

def run_all(snaps, hist, p, outdir=None):
    """Generate all void percolation visualizations."""
    print("  [6/6] Void percolation...")

    fig1 = plot_void_clusters(snaps, hist, p, outdir=outdir)
    plt.close(fig1)

    fig2 = plot_percolation_evolution(snaps, hist, p, outdir=outdir)
    plt.close(fig2)

    fig3 = plot_cluster_size_dist(snaps, hist, p, outdir=outdir)
    plt.close(fig3)

    fig4 = plot_kozeny_carman(hist, p, outdir=outdir)
    plt.close(fig4)


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Void percolation viz')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from viz2.common import load_data
    hist, snaps, p, _ = load_data(args.input)
    outdir = args.outdir or str(Path(args.input) / 'viz2_plots')
    run_all(snaps, hist, p, outdir=outdir)
