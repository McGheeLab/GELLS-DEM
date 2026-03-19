"""
Mesoscale Model Visualization
==============================
V2.5 — Plots for the SpatialPDE and ContactNetwork models.

Provides kymographs, radial profiles, network visualizations,
coordination evolution, cluster analysis, and combined panels.

Usage:
    python viz/mesoscale.py -i results/default
    python viz/mesoscale.py --standalone

Or import programmatically:
    from viz.mesoscale import plot_pde_kymograph, plot_network_evolution
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm


# ======================================================================
# SpatialPDE plots
# ======================================================================

def plot_pde_kymograph(result, model, outdir=None):
    """Kymograph: x_f(ξ, t) as a heatmap.

    Parameters
    ----------
    result : dict
        Output from SpatialPDE.solve().
    model : SpatialPDE
        The model instance.
    outdir : str or None
        Output directory for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    time = result['time']
    xi = model.xi

    # x_f kymograph
    ax = axes[0]
    X, T = np.meshgrid(xi, time)
    im = ax.pcolormesh(T, X, result['x_f'], cmap='RdYlBu_r', shading='auto')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Radial position $\xi$ (center=0, edge=1)')
    ax.set_title(r'$x_f(\xi, t)$ — Functional zone fraction')
    fig.colorbar(im, ax=ax, label=r'$x_f$')

    # Void fraction kymograph
    ax = axes[1]
    im = ax.pcolormesh(T, X, result['phi_v_f'], cmap='Blues', shading='auto')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\xi$')
    ax.set_title(r'$\phi_v^{func}(\xi, t)$ — Local void')
    fig.colorbar(im, ax=ax, label=r'$\phi_v$')

    # Permeability kymograph (log scale)
    ax = axes[2]
    K_f = result['K_f']
    K_f_log = np.log10(np.maximum(K_f, 1e-6))
    im = ax.pcolormesh(T, X, K_f_log, cmap='viridis', shading='auto')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\xi$')
    ax.set_title(r'$\log_{10} K_f(\xi, t)$ — Permeability')
    fig.colorbar(im, ax=ax, label=r'$\log_{10} K$ (µm²)')

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'pde_kymograph.png'), dpi=150,
                    bbox_inches='tight')
    return fig


def plot_pde_profiles(result, model, t_targets=None, outdir=None):
    """Radial profiles of x_f, void, tissue at selected times.

    Parameters
    ----------
    result : dict
        Output from SpatialPDE.solve().
    model : SpatialPDE
        The model instance.
    t_targets : list of float or None
        Times to plot. Defaults to 5 evenly spaced.
    outdir : str or None
        Output directory.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if t_targets is None:
        t_min, t_max = result['time'][0], result['time'][-1]
        t_targets = np.linspace(t_min, t_max, 5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    cmap = cm.get_cmap('viridis', len(t_targets))

    for i, t_tgt in enumerate(t_targets):
        prof = model.spatial_profile(result, t_tgt)
        color = cmap(i / max(len(t_targets) - 1, 1))
        label = f't = {prof["time"]:.0f} h'

        axes[0].plot(prof['xi'], prof['x_f'], color=color, lw=2, label=label)
        axes[1].plot(prof['xi'], prof['phi_v_f'], color=color, lw=2, label=label)
        axes[2].plot(prof['xi'], prof['K_f'], color=color, lw=2, label=label)

    axes[0].set_ylabel(r'$x_f$')
    axes[0].set_title('Functional zone fraction')
    axes[1].set_ylabel(r'$\phi_v^{func}$')
    axes[1].set_title('Local void fraction')
    axes[2].set_ylabel(r'$K_f$ (µm²)')
    axes[2].set_title('Permeability')
    axes[2].set_yscale('log')

    for ax in axes:
        ax.set_xlabel(r'$\xi$ (center $\rightarrow$ edge)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'pde_profiles.png'), dpi=150,
                    bbox_inches='tight')
    return fig


def plot_pde_timeseries(result, model, outdir=None):
    """Domain-averaged timeseries: x_f, compaction, stress balance.

    Parameters
    ----------
    result : dict
        Output from SpatialPDE.solve().
    model : SpatialPDE
        The model instance.
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    t = result['time']

    # x_f evolution
    ax = axes[0]
    ax.plot(t, result['x_f_avg'], 'C1-', lw=2, label=r'$\langle x_f \rangle$')
    ax.axhline(model.x_f0, color='grey', ls=':', lw=1, label=r'$x_f^0$')
    ax.axhline(model.x_f_min, color='grey', ls='--', lw=1,
               label=r'$x_f^{min}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$x_f$')
    ax.set_title('Domain-averaged compaction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Compaction ratio
    ax = axes[1]
    ax.plot(t, result['compaction_ratio'], 'C3-', lw=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Compaction ratio')
    ax.set_title(r'$-\Delta x_f / x_f^0$')
    ax.grid(True, alpha=0.3)

    # Stress balance (domain average)
    ax = axes[2]
    ax.plot(t, result['sigma_cell'], 'C1-', lw=2, label=r'$\sigma_{cell}$')
    # Domain-averaged resistance
    sigma_resist_avg = [float(np.mean(s)) for s in result['sigma_resist']]
    ax.plot(t, sigma_resist_avg, 'C0-', lw=2, label=r'$\langle\sigma_{resist}\rangle$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Stress (nN/µm²)')
    ax.set_title('Stress balance')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'pde_timeseries.png'), dpi=150,
                    bbox_inches='tight')
    return fig


# ======================================================================
# ContactNetwork plots
# ======================================================================

def plot_network_snapshot(net, outdir=None, title_suffix=''):
    """Visualize current network state: granules + contacts + bridges.

    Parameters
    ----------
    net : ContactNetwork
        Network instance.
    outdir : str or None
    title_suffix : str

    Returns
    -------
    matplotlib.figure.Figure
    """
    from analysis.contact_network import BridgeState

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw granules
    for i in range(net.N):
        color = 'C1' if net.gtype[i] == 0 else 'C0'
        alpha = 0.4
        circle = plt.Circle(net.pos[i, :2], net.radii[i],
                             color=color, alpha=alpha)
        ax.add_patch(circle)

    # Draw contact edges (grey)
    ei = net._edge_i
    ej = net._edge_j
    overlap = net._edge_overlap
    for k in range(len(ei)):
        if overlap[k] > 0:
            p1 = net.pos[ei[k], :2]
            p2 = net.pos[ej[k], :2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    'k-', lw=0.5, alpha=0.3)

    # Draw bridges (colored by state)
    bridge_colors = {
        BridgeState.FORMING: 'gold',
        BridgeState.ACTIVE: 'C3',
        BridgeState.LOCKED: 'C2',
    }
    for (i, j), br in net._bridges.items():
        if br['state'] not in bridge_colors:
            continue
        p1 = net.pos[i, :2]
        p2 = net.pos[j, :2]
        color = bridge_colors[br['state']]
        lw = 1.0 + 2.0 * br['ramp']
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                color=color, lw=lw, alpha=0.8)

    Lx = float(net._get('Lx', 800.0))
    Ly = float(net._get('Ly', 800.0))
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    ax.set_title(f'Contact Network{title_suffix}')

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        plt.Circle((0, 0), 1, color='C1', alpha=0.4, label='Functional'),
        plt.Circle((0, 0), 1, color='C0', alpha=0.4, label='Inert'),
        Line2D([0], [0], color='k', lw=0.5, alpha=0.3, label='Contact'),
        Line2D([0], [0], color='gold', lw=2, label='Bridge (forming)'),
        Line2D([0], [0], color='C3', lw=2, label='Bridge (active)'),
        Line2D([0], [0], color='C2', lw=2, label='Bridge (locked)'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=7)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'network_snapshot.png'), dpi=150,
                    bbox_inches='tight')
    return fig


def plot_network_evolution(result, outdir=None):
    """Plot coordination, bridges, percolation, forces over time.

    Parameters
    ----------
    result : dict
        Output from ContactNetwork.solve().
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    t = result['time']
    metrics = result['metrics']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Coordination number
    ax = axes[0, 0]
    Z = [m['Z'] for m in metrics]
    Z_ff = [m['Z_ff'] for m in metrics]
    Z_fi = [m['Z_fi'] for m in metrics]
    ax.plot(t, Z, 'k-', lw=2, label='Z (total)')
    ax.plot(t, Z_ff, 'C1--', lw=1.5, label='Z_ff')
    ax.plot(t, Z_fi, 'C2-.', lw=1.5, label='Z_fi')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Coordination number')
    ax.set_title('Coordination Evolution')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bridge counts
    ax = axes[0, 1]
    n_forming = [m['n_bridges_forming'] for m in metrics]
    n_active = [m['n_bridges_active'] for m in metrics]
    n_locked = [m['n_bridges_locked'] for m in metrics]
    ax.fill_between(t, 0, n_locked, color='C2', alpha=0.4, label='Locked')
    ax.fill_between(t, n_locked, np.array(n_locked) + np.array(n_active) - np.array(n_locked),
                    color='C3', alpha=0.4, label='Active')
    ax.fill_between(t,
                    np.array(n_active),
                    np.array(n_active) + np.array(n_forming),
                    color='gold', alpha=0.4, label='Forming')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Count')
    ax.set_title('Bridge States')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bridge fraction + percolation
    ax = axes[1, 0]
    bf = [m['bridge_fraction'] for m in metrics]
    perc = [m['percolation'] for m in metrics]
    ax.plot(t, bf, 'C3-', lw=2, label='Bridge fraction')
    # Mark percolation events
    perc_times = [t[i] for i in range(len(t)) if perc[i]]
    if perc_times:
        ax.axvspan(perc_times[0], perc_times[-1], color='C2', alpha=0.1,
                   label='Percolated')
        for pt in perc_times:
            ax.axvline(pt, color='C2', lw=0.5, alpha=0.3)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Bridge fraction')
    ax.set_title('Bridge Percolation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Cluster sizes
    ax = axes[1, 1]
    largest = [m['largest_cluster_size'] for m in metrics]
    ax.plot(t, largest, 'C4-', lw=2, label='Largest cluster')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Cluster size (granules)')
    ax.set_title('Largest Bridge-Connected Cluster')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'network_evolution.png'), dpi=150,
                    bbox_inches='tight')
    return fig


# ======================================================================
# Combined panels
# ======================================================================

def plot_combined_summary(pde_result, pde_model, net_result, outdir=None):
    """Combined summary: PDE spatial profiles + network topology.

    Parameters
    ----------
    pde_result : dict
        Output from SpatialPDE.solve().
    pde_model : SpatialPDE
        SpatialPDE instance.
    net_result : dict
        Output from ContactNetwork.solve().
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    t_pde = pde_result['time']
    t_net = net_result['time']
    net_metrics = net_result['metrics']

    # Top-left: PDE kymograph
    ax = axes[0, 0]
    X, T = np.meshgrid(pde_model.xi, t_pde)
    im = ax.pcolormesh(T, X, pde_result['x_f'], cmap='RdYlBu_r',
                        shading='auto')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\xi$')
    ax.set_title(r'Spatial PDE: $x_f(\xi, t)$')
    fig.colorbar(im, ax=ax, label=r'$x_f$')

    # Top-right: Network Z(t)
    ax = axes[0, 1]
    Z = [m['Z'] for m in net_metrics]
    Z_ff = [m['Z_ff'] for m in net_metrics]
    ax.plot(t_net, Z, 'k-', lw=2, label='Z')
    ax.plot(t_net, Z_ff, 'C1--', lw=1.5, label='Z_ff')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Coordination')
    ax.set_title('Contact Network: Z(t)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: PDE compaction + domain average
    ax = axes[1, 0]
    ax.plot(t_pde, pde_result['x_f_avg'], 'C1-', lw=2,
            label=r'PDE $\langle x_f \rangle$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$x_f$')
    ax.set_title('Domain-averaged compaction')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Bridge fraction + percolation
    ax = axes[1, 1]
    bf = [m['bridge_fraction'] for m in net_metrics]
    perc = [m['percolation'] for m in net_metrics]
    ax.plot(t_net, bf, 'C3-', lw=2, label='Bridge fraction')
    perc_times = [t_net[i] for i in range(len(t_net)) if perc[i]]
    if perc_times:
        ax.axvspan(min(perc_times), max(perc_times), color='C2', alpha=0.1,
                   label='Percolated')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Fraction')
    ax.set_title('Bridge percolation')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Mesoscale Model Summary: Spatial PDE + Contact Network',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'mesoscale_summary.png'), dpi=150,
                    bbox_inches='tight')
    return fig


# ======================================================================
# run_all convenience
# ======================================================================

def run_all(run_dir=None, outdir=None, p=None, seed=42,
            t_total=72.0, skip=None):
    """Run both mesoscale models and generate all plots.

    Parameters
    ----------
    run_dir : str or None
        DEM output directory. If None, runs standalone.
    outdir : str or None
        Output directory.
    p : Params or None
        Parameters (used if run_dir is None).
    seed : int
        RNG seed.
    t_total : float
        Total simulation time.
    skip : set or None
        Skip keys: 'pde', 'network', 'combined'.

    Returns
    -------
    dict with 'pde_model', 'pde_result', 'net', 'net_result'.
    """
    if skip is None:
        skip = set()

    from analysis.spatial_pde import SpatialPDE
    from analysis.contact_network import ContactNetwork

    if p is None:
        from new_dem_0 import Params
        p = Params()

    if outdir is None:
        if run_dir:
            outdir = str(Path(run_dir) / 'plots_mesoscale')
        else:
            outdir = 'results/mesoscale_standalone'

    Path(outdir).mkdir(parents=True, exist_ok=True)
    results = {}

    # --- SpatialPDE ---
    if 'pde' not in skip:
        print("  [1] Spatial PDE model...")
        if run_dir:
            pde_model, fit, data = SpatialPDE.from_run(run_dir)
            pde_result = fit['solution']
            print(f"      Fit: R²={fit['R2']:.4f}, η_eff={fit['eta_eff']:.3g}")
        else:
            pde_model = SpatialPDE.from_params(p)
            pde_result = pde_model.solve(t_span=(0, t_total), record_every=2.0)

        results['pde_model'] = pde_model
        results['pde_result'] = pde_result

        plot_pde_kymograph(pde_result, pde_model, outdir)
        plot_pde_profiles(pde_result, pde_model, outdir=outdir)
        plot_pde_timeseries(pde_result, pde_model, outdir)
        print("      PDE plots saved.")

    # --- ContactNetwork ---
    if 'network' not in skip:
        print("  [2] Contact network model...")
        if run_dir:
            net, hist, p_loaded = ContactNetwork.from_run(run_dir)
        else:
            net = ContactNetwork.from_packing(p, seed=seed)

        net_result = net.solve(t_span=(0, t_total), seed=seed,
                               record_every=2.0)
        results['net'] = net
        results['net_result'] = net_result

        plot_network_snapshot(net, outdir,
                              title_suffix=f' (t={t_total:.0f}h)')
        plot_network_evolution(net_result, outdir)

        final = net_result['metrics'][-1]
        print(f"      Z={final['Z']:.2f}, bridges={final['n_bridges_active']}, "
              f"perc={final['percolation']}")
        print("      Network plots saved.")

    # --- Combined ---
    if 'combined' not in skip and 'pde' not in skip and 'network' not in skip:
        print("  [3] Combined summary...")
        plot_combined_summary(pde_result, pde_model, net_result, outdir)
        print("      Combined plot saved.")

    print(f"\n  All mesoscale plots saved to: {outdir}/")
    return results
