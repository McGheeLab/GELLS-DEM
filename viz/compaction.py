"""
Compaction & Void-Space Evolution Visualization
================================================
Tracks void space between granules of each solid phase over time.

Plots produced:
  1. Void fraction vs time (total, functional-adjacent, inert-adjacent)
  2. Solid packing fraction vs time (functional, inert, total)
  3. Compaction ratio vs time
  4. Void size distribution at selected times
  5. Phase fraction evolution (stacked area)

Usage:
    python viz/compaction.py -i ./simulations/run1
    python viz/compaction.py --hist history.json --outdir ./plots
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import argparse, json

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from scipy.ndimage import label, binary_dilation


# ── Utility: load history from run() output or JSON ──

def load_history(hist_path):
    """Load history list of dicts from JSON file."""
    with open(hist_path) as f:
        return json.load(f)


def void_near_phase(phi_v, phi_phase, v_thresh=0.3, p_thresh=0.3):
    """Fraction of void voxels/pixels adjacent to a given solid phase."""
    void_mask = phi_v > v_thresh
    phase_mask = phi_phase > p_thresh
    # Dilate phase mask by 1 pixel to find adjacency
    dilated = binary_dilation(phase_mask, iterations=1)
    adjacent_void = void_mask & dilated & ~phase_mask
    return float(np.sum(adjacent_void)) / max(1, np.sum(void_mask))


def void_cluster_sizes(phi_v, thresh=0.3):
    """Return array of connected void cluster sizes (in voxels/pixels)."""
    binary = (phi_v > thresh).astype(int)
    lab, nc = label(binary)
    if nc == 0:
        return np.array([0])
    return np.array([np.sum(lab == l) for l in range(1, nc + 1)])


# ── Plotting functions ──

def plot_void_fraction(hist, outdir=None):
    """Plot 1: Void fraction (porosity) vs time."""
    t = [h['time'] for h in hist]
    porosity = [h.get('porosity', h.get('phi_v_mean', 0)) for h in hist]
    phi_f = [h.get('phi_f_mean', 0) for h in hist]
    phi_i = [h.get('phi_i_mean', 0) for h in hist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, porosity, 'C2-', lw=2, label='Void (porosity)')
    ax.plot(t, phi_f, 'C1--', lw=1.5, label='Functional solid')
    ax.plot(t, phi_i, 'C0--', lw=1.5, label='Inert solid')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Volume/Area fraction')
    ax.set_title('Phase Fractions Over Time')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'void_fraction.png', dpi=150)
    return fig


def plot_packing_fraction(hist, outdir=None):
    """Plot 2: Solid packing fraction vs time."""
    t = [h['time'] for h in hist]
    phi_solid = [h.get('phi_solid', h.get('phi_f_mean', 0) + h.get('phi_i_mean', 0))
                 for h in hist]
    phi_rcp = [h.get('phi_RCP', 0.64) for h in hist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, phi_solid, 'C3-', lw=2, label='Solid packing fraction')
    ax.plot(t, phi_rcp, 'k--', lw=1, alpha=0.5, label='RCP limit')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Packing fraction')
    ax.set_title('Solid Packing vs Random Close Packing')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'packing_fraction.png', dpi=150)
    return fig


def plot_compaction_ratio(hist, outdir=None):
    """Plot 3: Compaction ratio vs time."""
    t = [h['time'] for h in hist]
    cr = [h.get('compaction_ratio', 0) for h in hist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, cr, 'C4-', lw=2)
    ax.axhline(1.0, ls=':', c='gray', lw=0.8, label='RCP')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\phi_{solid} / \phi_{RCP}$')
    ax.set_title('Compaction Ratio Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'compaction_ratio.png', dpi=150)
    return fig


def plot_void_distribution(snaps, hist, indices=None, outdir=None):
    """Plot 4: Void cluster size distribution at selected times."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))

    fig, axes = plt.subplots(1, len(indices), figsize=(4*len(indices), 4))
    if len(indices) == 1:
        axes = [axes]

    for c, si in enumerate(indices):
        snap = snaps[si]
        phi_v = snap['phi_v'] if isinstance(snap, dict) else snap[2]
        sizes = void_cluster_sizes(phi_v)
        axes[c].hist(sizes, bins=20, color='C2', alpha=0.7, edgecolor='k')
        t = hist[si]['time']
        axes[c].set_title(f't = {t:.1f} h')
        axes[c].set_xlabel('Cluster size (voxels)')
        axes[c].set_ylabel('Count')

    fig.suptitle('Void Cluster Size Distribution', fontsize=13)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'void_distribution.png', dpi=150)
    return fig


def plot_stacked_phases(hist, outdir=None):
    """Plot 5: Stacked area chart of phase fractions."""
    t = [h['time'] for h in hist]
    phi_f = [h.get('phi_f_mean', 0) for h in hist]
    phi_i = [h.get('phi_i_mean', 0) for h in hist]
    phi_v = [h.get('porosity', h.get('phi_v_mean', 0)) for h in hist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(t, phi_f, phi_i, phi_v,
                 labels=['Functional', 'Inert', 'Void'],
                 colors=['orangered', 'steelblue', 'lightgreen'], alpha=0.8)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Volume/Area fraction')
    ax.set_title('Phase Evolution (Stacked)')
    ax.legend(loc='center right')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'stacked_phases.png', dpi=150)
    return fig


def run_all(hist, snaps=None, outdir=None):
    """Generate all compaction plots."""
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    figs = {}
    figs['void_fraction'] = plot_void_fraction(hist, outdir)
    figs['packing_fraction'] = plot_packing_fraction(hist, outdir)
    figs['compaction_ratio'] = plot_compaction_ratio(hist, outdir)
    figs['stacked_phases'] = plot_stacked_phases(hist, outdir)
    if snaps is not None and len(snaps) > 0:
        figs['void_distribution'] = plot_void_distribution(snaps, hist, outdir=outdir)

    print(f"  Compaction plots saved to {outdir or 'memory'}")
    return figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compaction visualization')
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--outdir', default=None, help='Output directory')
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    # V1.5: Try load_run() first, fall back to history.json
    try:
        from new_dem_0 import load_run
        hist, snaps, p, meta = load_run(str(indir))
        run_all(hist, snaps=snaps, outdir=outdir)
    except Exception:
        hist_file = indir / 'history.json'
        if hist_file.exists():
            hist = load_history(hist_file)
            run_all(hist, outdir=outdir)
        else:
            print(f"No history.json found in {indir}")
