"""
Percolation & Transport Property Visualization
===============================================
Analyzes void network transport properties via Darcy's law,
Kozeny-Carman equation, random close packing, and dimensionless groups.

Plots produced:
  1. Kozeny-Carman permeability K(t) vs time
  2. Porosity evolution with RCP reference
  3. Dimensionless groups dashboard (2x3 subplots)
  4. Darcy flow rate for given pressure gradient
  5. Void connectivity / percolation indicator

Usage:
    python viz/percolation.py -i ./simulations/run1
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


# ── Transport physics ──

def kozeny_carman_K(porosity, d_grain):
    """K = eps^3 * d^2 / [180*(1-eps)^2]  [µm²]"""
    eps = np.clip(porosity, 0.01, 0.99)
    return eps**3 * d_grain**2 / (180.0 * (1.0 - eps)**2)


def darcy_flow_rate(K_um2, mu_Pa_s, dP_Pa, L_um):
    """
    Darcy flow rate q = K/mu * dP/L  [µm/h]
    K in µm², mu in Pa·s, dP in Pa, L in µm.
    Returns q in µm/h (after unit conversion).
    """
    # K [µm²] = K * 1e-12 [m²]
    # q [m/s] = K[m²] * dP[Pa] / (mu[Pa·s] * L[m])
    K_m2 = K_um2 * 1e-12
    L_m = L_um * 1e-6
    q_m_s = K_m2 * dP_Pa / (mu_Pa_s * L_m)
    # Convert m/s to µm/h
    return q_m_s * 1e6 * 3600.0


def reynolds_pore(q_um_h, d_pore_um, rho_kg_m3=1000.0, mu_Pa_s=1e-3):
    """Pore-scale Reynolds number Re = rho * v * d / mu."""
    v_m_s = q_um_h * 1e-6 / 3600.0
    d_m = d_pore_um * 1e-6
    return rho_kg_m3 * v_m_s * d_m / mu_Pa_s


def peclet_number(v_um_h, L_um, D_um2_h):
    """Peclet number Pe = v*L/D."""
    if D_um2_h <= 0:
        return 0
    return v_um_h * L_um / D_um2_h


# ── Plotting functions ──

def plot_permeability(hist, outdir=None):
    """Plot 1: Kozeny-Carman permeability vs time."""
    t = [h['time'] for h in hist]
    K = [h.get('K_kozeny_carman', 0) for h in hist]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, K, 'C0-', lw=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Permeability K (µm²)')
    ax.set_title('Kozeny-Carman Permeability Over Time')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'permeability.png', dpi=150)
    return fig


def plot_porosity_rcp(hist, outdir=None):
    """Plot 2: Porosity evolution with RCP line."""
    t = [h['time'] for h in hist]
    porosity = [h.get('porosity', h.get('phi_v_mean', 0)) for h in hist]
    phi_solid = [1.0 - p for p in porosity]
    phi_rcp = [h.get('phi_RCP', 0.64) for h in hist]
    void_rcp = [1.0 - r for r in phi_rcp]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t, porosity, 'C2-', lw=2, label='Porosity ε(t)')
    ax1.plot(t, void_rcp, 'k--', lw=1, alpha=0.5, label='1 - φ_RCP')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Porosity ε')
    ax1.set_title('Porosity Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Porosity ratio
    eps_0 = porosity[0] if porosity[0] > 0 else 1
    eps_ratio = [p / eps_0 for p in porosity]
    ax2.plot(t, eps_ratio, 'C5-', lw=2)
    ax2.axhline(1.0, ls=':', c='gray', lw=0.8)
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('ε(t) / ε(0)')
    ax2.set_title('Porosity Ratio')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'porosity_rcp.png', dpi=150)
    return fig


def plot_dimensionless_groups(hist, mu=1e-3, dP=100.0, D_um2_h=360.0,
                               outdir=None):
    """
    Plot 3: Dimensionless groups dashboard.
    mu: viscosity (Pa·s), dP: pressure gradient (Pa),
    D_um2_h: diffusion coefficient (µm²/h, ~100 µm²/s for small molecule).
    """
    t = [h['time'] for h in hist]
    K = [h.get('K_kozeny_carman', 0) for h in hist]
    porosity = [h.get('porosity', 0.5) for h in hist]
    d_grain = [h.get('d_grain_mean', 80) for h in hist]
    cr = [h.get('compaction_ratio', 0) for h in hist]
    Da = [h.get('Da_number', 0) for h in hist]

    # Compute derived quantities
    L = d_grain[0] * 10  # characteristic length ~ 10 grain diameters
    q = [darcy_flow_rate(k, mu, dP, L) for k in K]
    Re = [reynolds_pore(qi, di * porosity[idx], mu_Pa_s=mu)
          for idx, (qi, di) in enumerate(zip(q, d_grain))]
    Pe = [peclet_number(qi, L, D_um2_h) for qi in q]

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) Permeability K(t)
    ax[0,0].plot(t, K, 'C0-', lw=2)
    ax[0,0].set_xlabel('Time (h)')
    ax[0,0].set_ylabel('K (µm²)')
    ax[0,0].set_title('Kozeny-Carman Permeability')
    ax[0,0].grid(True, alpha=0.3)

    # (0,1) Compaction ratio
    ax[0,1].plot(t, cr, 'C4-', lw=2)
    ax[0,1].axhline(1.0, ls=':', c='gray', lw=0.8, label='RCP')
    ax[0,1].set_xlabel('Time (h)')
    ax[0,1].set_ylabel(r'$\phi / \phi_{RCP}$')
    ax[0,1].set_title('Compaction Ratio')
    ax[0,1].legend()
    ax[0,1].grid(True, alpha=0.3)

    # (0,2) Darcy number
    ax[0,2].semilogy(t, [max(d, 1e-20) for d in Da], 'C3-', lw=2)
    ax[0,2].set_xlabel('Time (h)')
    ax[0,2].set_ylabel(r'$Da = K/L^2$')
    ax[0,2].set_title('Darcy Number')
    ax[0,2].grid(True, alpha=0.3)

    # (1,0) Peclet number
    ax[1,0].plot(t, Pe, 'C1-', lw=2)
    ax[1,0].set_xlabel('Time (h)')
    ax[1,0].set_ylabel('Pe = vL/D')
    ax[1,0].set_title(f'Peclet Number (dP={dP} Pa)')
    ax[1,0].grid(True, alpha=0.3)

    # (1,1) Reynolds number
    ax[1,1].semilogy(t, [max(r, 1e-20) for r in Re], 'C5-', lw=2)
    ax[1,1].set_xlabel('Time (h)')
    ax[1,1].set_ylabel('Re')
    ax[1,1].set_title('Pore Reynolds Number')
    ax[1,1].grid(True, alpha=0.3)

    # (1,2) Porosity ratio
    eps_0 = porosity[0] if porosity[0] > 0 else 1
    eps_ratio = [p / eps_0 for p in porosity]
    ax[1,2].plot(t, eps_ratio, 'C2-', lw=2)
    ax[1,2].axhline(1.0, ls=':', c='gray', lw=0.8)
    ax[1,2].set_xlabel('Time (h)')
    ax[1,2].set_ylabel('ε(t) / ε(0)')
    ax[1,2].set_title('Porosity Ratio')
    ax[1,2].grid(True, alpha=0.3)

    fig.suptitle('Dimensionless Groups Dashboard', fontsize=14, y=1.01)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'dimensionless_groups.png', dpi=150)
    return fig


def plot_darcy_flow(hist, mu=1e-3, pressures=[10, 50, 100, 500], outdir=None):
    """Plot 4: Darcy flow rate for different pressure gradients."""
    t = [h['time'] for h in hist]
    K = [h.get('K_kozeny_carman', 0) for h in hist]
    d_grain = [h.get('d_grain_mean', 80) for h in hist]
    L = d_grain[0] * 10

    fig, ax = plt.subplots(figsize=(8, 5))
    for dP in pressures:
        q = [darcy_flow_rate(k, mu, dP, L) for k in K]
        ax.plot(t, q, lw=2, label=f'dP = {dP} Pa')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Darcy flow rate q (µm/h)')
    ax.set_title('Darcy Flow Rate vs Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'darcy_flow.png', dpi=150)
    return fig


def plot_void_connectivity(hist, outdir=None):
    """Plot 5: Void connectivity / percolation indicator."""
    t = [h['time'] for h in hist]
    void_lf = [h.get('void_lf', 0) for h in hist]
    void_nc = [h.get('void_nc', 0) for h in hist]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t, void_lf, 'C2-', lw=2)
    ax1.axhline(1.0, ls=':', c='gray', lw=0.8, label='Percolated')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel('Largest void cluster / total')
    ax1.set_title('Void Connectivity (1 = spanning)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, void_nc, 'C2-', lw=2)
    ax2.set_xlabel('Time (h)')
    ax2.set_ylabel('Number of void clusters')
    ax2.set_title('Void Fragmentation')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        fig.savefig(Path(outdir) / 'void_connectivity.png', dpi=150)
    return fig


def run_all(hist, outdir=None):
    """Generate all percolation/transport plots."""
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)

    figs = {}
    figs['permeability'] = plot_permeability(hist, outdir)
    figs['porosity_rcp'] = plot_porosity_rcp(hist, outdir)
    figs['dimensionless'] = plot_dimensionless_groups(hist, outdir=outdir)
    figs['darcy_flow'] = plot_darcy_flow(hist, outdir=outdir)
    figs['void_connectivity'] = plot_void_connectivity(hist, outdir)

    print(f"  Percolation plots saved to {outdir or 'memory'}")
    return figs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Percolation visualization')
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    indir = Path(args.input)
    outdir = args.outdir or str(indir / 'visualizations')

    # V1.5: Try load_run() first, fall back to history.json
    try:
        from new_dem_0 import load_run
        hist, snaps, p, meta = load_run(str(indir))
        run_all(hist, outdir=outdir)
    except Exception:
        hist_file = indir / 'history.json'
        if hist_file.exists():
            with open(hist_file) as f:
                hist = json.load(f)
            run_all(hist, outdir=outdir)
        else:
            print(f"No history.json found in {indir}")
