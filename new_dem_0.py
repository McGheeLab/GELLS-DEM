"""
Overdamped Particle Dynamics Model for Cell-Driven Granular Rearrangement
==========================================================================

MATHEMATICAL MODEL
------------------
N rigid circular granules (functional or inert) in 2D, overdamped regime.
Granules maintain their shape; cells bridge nearby functional pairs and
pull them together.

EQUATIONS OF MOTION (overdamped Langevin):

    γ_i dx_i/dt = F_i^contact + F_i^cell + F_i^wall + F_i^noise

where γ_i = 6πη R_i is the Stokes drag on granule i.

FORCE LAWS:

(1) Contact repulsion (soft sphere, prevents overlap):
    F_ij^contact = k_n δ_ij n̂_ij      if δ_ij = R_i+R_j - d_ij > 0
                 = 0                    otherwise

(2) Cell-mediated attraction (functional–functional pairs only):
    gap_ij = d_ij - R_i - R_j                     (surface separation)
    proximity = max(0, 1 - gap_ij / L_max)         (0 at L_max, 1 at contact)
    n_bridges = √(n_cells_i · n_cells_j) · proximity
    F_ij^cell = -k_cell · n_bridges · max(0, gap_ij - L_rest) · n̂_ij
    |F_ij^cell| ≤ F_max · n_bridges                (force cap per bridge)

(3) Wall repulsion (confining boundary):
    F_wall = k_wall · penetration · n̂_wall        for each boundary

(4) Activity noise (small stochastic kicks on functional granules):
    F_noise ~ √(2 γ_i T_active) · ξ(t)            (cell-driven fluctuations)

CELL COUNT PER GRANULE (surface-area limited):
    n_cells_i = min(n_cells_input, floor(4π R_i² · coverage / A_cell))

OBSERVABLES (rendered from particle positions at each save step):
    φ_f(x), φ_i(x), φ_v(x) = 1 - φ_f - φ_i
    → void topology, functional topology, packing evolution, tissue metrics

PHYSICAL PARAMETER MAPPING:
    k_n    ~ E_hydrogel · R_eff       (contact stiffness from modulus)
    k_cell ~ F_cell / L_cell          (cell spring constant, ~nN/µm)
    L_max  ~ 2-5 cell diameters       (max bridging distance)
    η      ~ 1e-3 Pa·s               (culture medium viscosity)
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time as timer


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    # ── Domain (µm) ──
    Lx: float = 800.0
    Ly: float = 800.0

    # ── Granule physical properties ──
    R_func_mean: float = 40.0       # functional radius (µm)
    R_func_std: float = 5.0
    R_inert_mean: float = 60.0      # inert radius (µm)
    R_inert_std: float = 8.0

    # ── Composition ──
    phi_f_target: float = 0.25      # functional area fraction target
    phi_i_target: float = 0.20      # inert area fraction target

    # ── Cell properties ──
    n_cells_per_granule: int = 8    # cells seeded per functional granule
    cell_diameter: float = 15.0     # µm
    cell_coverage: float = 0.6      # max fraction of surface covered
    F_max_per_cell: float = 50.0    # nN, max force per cell

    # ── Cell bridging ──
    k_cell: float = 0.8            # nN/µm, cell spring constant
    L_max: float = 60.0             # µm, max bridging gap
    L_rest: float = 12.0            # µm, rest length (~ cell diameter)

    # ── Contact mechanics ──
    k_contact: float = 50.0         # nN/µm, repulsive stiffness
    k_wall: float = 80.0            # nN/µm, wall stiffness

    # ── Drag ──
    eta: float = 1e-3               # Pa·s (water-like medium)
    drag_scale: float = 0.05        # nondim scaling for drag coefficient

    # ── Active noise ──
    T_active: float = 5.0           # active temperature (nN·µm)

    # ── Time integration ──
    dt: float = 0.5                 # hours
    t_total: float = 72.0           # hours
    save_every_h: float = 2.0       # save interval (hours)
    v_max: float = 20.0             # µm/hr, velocity cap

    # ── Rendering ──
    Ngrid: int = 200                # grid for field rendering
    interface_width: float = 3.0    # µm, tanh smoothing

    @property
    def save_every(self):
        return max(1, int(self.save_every_h / self.dt))


# ══════════════════════════════════════════════════════════════════════
# Granule data structure
# ══════════════════════════════════════════════════════════════════════

class GranuleSystem:
    """Tracks all granule state."""
    def __init__(self, x, y, r, gtype, n_cells):
        self.x = np.array(x, dtype=np.float64)
        self.y = np.array(y, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)
        self.gtype = np.array(gtype, dtype=int)  # 0=func, 1=inert
        self.n_cells = np.array(n_cells, dtype=np.float64)
        self.N = len(x)
        self.func_mask = self.gtype == 0
        self.inert_mask = self.gtype == 1

    def positions(self):
        return np.column_stack([self.x, self.y])


# ══════════════════════════════════════════════════════════════════════
# Packing generator (random sequential addition)
# ══════════════════════════════════════════════════════════════════════

def generate_packing(p: Params, seed=42) -> GranuleSystem:
    rng = np.random.default_rng(seed)
    domain_area = p.Lx * p.Ly

    area_f = np.pi * p.R_func_mean**2
    area_i = np.pi * p.R_inert_mean**2
    n_func = int(round(p.phi_f_target * domain_area / area_f))
    n_inert = int(round(p.phi_i_target * domain_area / area_i))

    print(f"  Target: {n_func} functional (R~{p.R_func_mean:.0f}µm) + "
          f"{n_inert} inert (R~{p.R_inert_mean:.0f}µm)")

    xs, ys, rs, types = [], [], [], []
    gap = 2.0  # minimum gap between granule surfaces (µm)

    # Interleave placement for good mixing
    order = []
    fi, ii = 0, 0
    while fi < n_func or ii < n_inert:
        if ii < n_inert: order.append(1); ii += 1
        if fi < n_func: order.append(0); fi += 1

    for gt in order:
        if gt == 0:
            r = max(15, rng.normal(p.R_func_mean, p.R_func_std))
        else:
            r = max(20, rng.normal(p.R_inert_mean, p.R_inert_std))

        placed = False
        for _ in range(800):
            cx = rng.uniform(r + gap, p.Lx - r - gap)
            cy = rng.uniform(r + gap, p.Ly - r - gap)
            ok = True
            for j in range(len(xs)):
                dx = cx - xs[j]; dy = cy - ys[j]
                if dx*dx + dy*dy < (r + rs[j] + gap)**2:
                    ok = False; break
            if ok:
                xs.append(cx); ys.append(cy)
                rs.append(r); types.append(gt)
                placed = True; break
        if not placed:
            pass  # skip if can't place

    # Compute cells per granule
    n_cells = []
    cell_area = np.pi * (p.cell_diameter/2)**2
    for i in range(len(xs)):
        if types[i] == 0:
            surface = 2 * np.pi * rs[i]  # circumference in 2D
            max_from_area = int(surface * p.cell_coverage / p.cell_diameter)
            nc = min(p.n_cells_per_granule, max(1, max_from_area))
            n_cells.append(nc)
        else:
            n_cells.append(0)

    gs = GranuleSystem(xs, ys, rs, types, n_cells)
    act_f = sum(np.pi*gs.r[gs.func_mask]**2) / domain_area
    act_i = sum(np.pi*gs.r[gs.inert_mask]**2) / domain_area
    print(f"  Placed: {np.sum(gs.func_mask)} func (φ_f={act_f:.3f}) + "
          f"{np.sum(gs.inert_mask)} inert (φ_i={act_i:.3f})")
    print(f"  Void fraction: {1-act_f-act_i:.3f}")
    print(f"  Total cells: {int(sum(gs.n_cells))}")
    return gs


# ══════════════════════════════════════════════════════════════════════
# Force computation
# ══════════════════════════════════════════════════════════════════════

def compute_forces(gs: GranuleSystem, p: Params, rng) -> np.ndarray:
    """
    Compute all forces on each granule.
    Returns (N, 2) force array in nN.
    """
    N = gs.N
    F = np.zeros((N, 2))
    pos = gs.positions()

    # ── Neighbour search ──
    max_r = np.max(gs.r)
    cutoff = 2*max_r + p.L_max
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j,0] - pos[i,0]
        dy = pos[j,1] - pos[i,1]
        d = np.sqrt(dx*dx + dy*dy)
        if d < 1e-6:
            continue
        nx, ny = dx/d, dy/d

        # ── Contact repulsion ──
        overlap = gs.r[i] + gs.r[j] - d
        if overlap > 0:
            Fc = p.k_contact * overlap
            F[i,0] -= Fc * nx; F[i,1] -= Fc * ny
            F[j,0] += Fc * nx; F[j,1] += Fc * ny

        # ── Cell bridging (functional-functional only) ──
        if gs.gtype[i] == 0 and gs.gtype[j] == 0:
            gap = d - gs.r[i] - gs.r[j]
            if 0 < gap < p.L_max:
                proximity = 1.0 - gap / p.L_max
                n_br = np.sqrt(gs.n_cells[i] * gs.n_cells[j]) * proximity
                extension = gap - p.L_rest
                if extension > 0:
                    F_mag = p.k_cell * n_br * extension
                    F_cap = p.F_max_per_cell * n_br
                    F_mag = min(F_mag, F_cap)
                    # Attractive: pull i toward j
                    F[i,0] += F_mag * nx; F[i,1] += F_mag * ny
                    F[j,0] -= F_mag * nx; F[j,1] -= F_mag * ny

    # ── Wall repulsion ──
    for i in range(N):
        r = gs.r[i]
        if gs.x[i] < r:
            F[i,0] += p.k_wall * (r - gs.x[i])
        if gs.x[i] > p.Lx - r:
            F[i,0] -= p.k_wall * (gs.x[i] - (p.Lx - r))
        if gs.y[i] < r:
            F[i,1] += p.k_wall * (r - gs.y[i])
        if gs.y[i] > p.Ly - r:
            F[i,1] -= p.k_wall * (gs.y[i] - (p.Ly - r))

    # ── Active noise on functional granules ──
    if p.T_active > 0:
        for i in range(N):
            if gs.gtype[i] == 0:
                gamma_i = p.drag_scale * gs.r[i]
                noise_amp = np.sqrt(2 * gamma_i * p.T_active / p.dt)
                F[i,0] += noise_amp * rng.standard_normal()
                F[i,1] += noise_amp * rng.standard_normal()

    return F


# ══════════════════════════════════════════════════════════════════════
# Time integration (overdamped: γ dx/dt = F  →  dx = F/γ · dt)
# ══════════════════════════════════════════════════════════════════════

def step(gs: GranuleSystem, p: Params, rng):
    """One overdamped Euler step."""
    F = compute_forces(gs, p, rng)

    for i in range(gs.N):
        gamma_i = p.drag_scale * gs.r[i]
        vx = F[i,0] / gamma_i
        vy = F[i,1] / gamma_i
        # Velocity cap
        v = np.sqrt(vx*vx + vy*vy)
        if v > p.v_max:
            vx *= p.v_max / v; vy *= p.v_max / v
        gs.x[i] += vx * p.dt
        gs.y[i] += vy * p.dt
        # Hard wall clamp
        r = gs.r[i]
        gs.x[i] = np.clip(gs.x[i], r+0.5, p.Lx-r-0.5)
        gs.y[i] = np.clip(gs.y[i], r+0.5, p.Ly-r-0.5)

    return F


# ══════════════════════════════════════════════════════════════════════
# Phase field rendering (from particle positions)
# ══════════════════════════════════════════════════════════════════════

def render_fields(gs: GranuleSystem, p: Params):
    """
    Stamp each granule as tanh-profile disk onto grid.
    Returns φ_f, φ_i, φ_v arrays of shape (Ngrid, Ngrid).
    """
    Ng = p.Ngrid
    dx = p.Lx / Ng
    xg = np.linspace(dx/2, p.Lx - dx/2, Ng)
    yg = np.linspace(dx/2, p.Ly - dx/2, Ng)
    X, Y = np.meshgrid(xg, yg, indexing='ij')

    phi_f = np.zeros((Ng, Ng))
    phi_i = np.zeros((Ng, Ng))
    w = p.interface_width

    for i in range(gs.N):
        dist = np.sqrt((X - gs.x[i])**2 + (Y - gs.y[i])**2)
        profile = 0.5 * (1.0 - np.tanh((dist - gs.r[i]) / w))
        if gs.gtype[i] == 0:
            phi_f = np.maximum(phi_f, profile)
        else:
            phi_i = np.maximum(phi_i, profile)

    # Prevent total > 1
    total = phi_f + phi_i
    over = total > 0.99
    if np.any(over):
        phi_f[over] *= 0.99 / total[over]
        phi_i[over] *= 0.99 / total[over]

    return phi_f, phi_i, 1.0 - phi_f - phi_i


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def connectivity(field, thresh_frac=0.3):
    """Cluster analysis on thresholded field."""
    thr = np.mean(field) + thresh_frac * np.std(field)
    b = (field > thr).astype(int)
    lab, nc = label(b)
    if nc == 0 or b.sum() == 0:
        return 0, 0.0, 0.0
    sz = np.array([np.sum(lab == l) for l in range(1, nc+1)])
    return nc, float(sz.max()/b.sum()), float(b.sum()/b.size)


def compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces):
    m = dict(time=t)
    m['phi_f_mean'] = np.mean(phi_f)
    m['phi_i_mean'] = np.mean(phi_i)
    m['phi_v_mean'] = np.mean(phi_v)

    # Functional connectivity
    fn, fl, fc = connectivity(phi_f, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)

    # Void connectivity
    vn, vl, vc = connectivity(phi_v, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)

    # Inert connectivity
    inn, il, _ = connectivity(phi_i, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    # Tissue: dense functional regions
    m['tissue_frac'] = float(np.mean(phi_f > 0.5))

    # Max cluster area
    thr = np.mean(phi_f) + 0.3*np.std(phi_f)
    b = (phi_f > thr).astype(int); lab, nc = label(b)
    dxg = p.Lx / p.Ngrid
    if nc > 0:
        sizes = np.array([np.sum(lab==l) for l in range(1, nc+1)])
        m['func_max_area'] = float(np.max(sizes) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    # Mean displacement from initial (stored externally)
    # Packing in functional-rich region
    fr = phi_f > thr
    pt = phi_f + phi_i
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    # Mean force magnitude
    F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[gs.func_mask])) if np.any(gs.func_mask) else 0.0

    # Bridge count
    pos = gs.positions()
    n_bridges = 0
    for i in range(gs.N):
        if gs.gtype[i] != 0: continue
        for j in range(i+1, gs.N):
            if gs.gtype[j] != 0: continue
            d = np.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
            gap = d - gs.r[i] - gs.r[j]
            if 0 < gap < p.L_max:
                n_bridges += 1
    m['n_bridges'] = n_bridges

    # Mean functional granule displacement
    # (stored as running metric)
    return m


def compute_displacement(gs, x0, y0):
    """RMS displacement from initial positions."""
    dx = gs.x - x0; dy = gs.y - y0
    disp = np.sqrt(dx**2 + dy**2)
    return float(np.mean(disp[gs.func_mask])), float(np.mean(disp[gs.inert_mask]))


# ══════════════════════════════════════════════════════════════════════
# Main simulation loop
# ══════════════════════════════════════════════════════════════════════

def run(p=None, seed=42):
    if p is None: p = Params()
    rng = np.random.default_rng(seed)

    print("\n  Generating packing...")
    gs = generate_packing(p, seed=seed)

    # Store initial positions for displacement tracking
    x0 = gs.x.copy(); y0 = gs.y.copy()

    n_steps = int(p.t_total / p.dt)
    hist, snaps, disp_hist = [], [], []

    def save(t, F):
        pf, pi, pv = render_fields(gs, p)
        m = compute_metrics(gs, p, pf, pi, pv, t, F)
        df, di = compute_displacement(gs, x0, y0)
        m['disp_func'] = df; m['disp_inert'] = di
        hist.append(m)
        snaps.append((pf.copy(), pi.copy(), pv.copy(),
                       gs.x.copy(), gs.y.copy(), gs.r.copy(), gs.gtype.copy()))
        return m

    # Initial save
    F0 = compute_forces(gs, p, rng)
    m = save(0.0, F0)
    print(f"\n  {'t(h)':>6} {'f_cl':>5} {'f_lf':>6} {'v_cl':>5} "
          f"{'tissue':>7} {'bridges':>7} {'disp_f':>7}")
    print(f"  {0:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} {m['void_nc']:5d} "
          f"{m['tissue_frac']:7.3f} {m['n_bridges']:7d} {m['disp_func']:7.1f}")

    wall_t0 = timer.time()
    t = 0.0
    for s in range(1, n_steps + 1):
        F = step(gs, p, rng)
        t += p.dt

        if s % p.save_every == 0:
            m = save(t, F)
            print(f"  {t:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} "
                  f"{m['void_nc']:5d} {m['tissue_frac']:7.3f} "
                  f"{m['n_bridges']:7d} {m['disp_func']:7.1f}")

    elapsed = timer.time() - wall_t0
    print(f"\n  Done in {elapsed:.1f}s ({n_steps} steps, {gs.N} granules)")
    return hist, snaps, p, gs


# ══════════════════════════════════════════════════════════════════════
# Visualisation
# ══════════════════════════════════════════════════════════════════════

def plot_granules(snaps, hist, p, indices=None):
    """Plot granule positions as circles at selected times."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(4*nc, 4))
    if nc == 1: axes = [axes]

    for c, si in enumerate(indices):
        pf, pi, pv, xs, ys, rs, gt = snaps[si]
        ax = axes[c]; ax.set_xlim(0, p.Lx); ax.set_ylim(0, p.Ly)
        ax.set_aspect('equal')

        # Draw granules
        for i in range(len(xs)):
            if gt[i] == 0:
                color = 'orangered'; alpha = 0.75
            else:
                color = 'steelblue'; alpha = 0.55
            circ = Circle((xs[i], ys[i]), rs[i], fc=color, ec='k',
                          lw=0.3, alpha=alpha)
            ax.add_patch(circ)

        ax.set_title(f"t = {hist[si]['time']:.1f} h", fontsize=10)
        if c == 0:
            ax.plot([], [], 'o', color='orangered', ms=8, label='Functional')
            ax.plot([], [], 'o', color='steelblue', ms=8, label='Inert')
            ax.legend(fontsize=8, loc='upper right')

    fig.suptitle('Granule Positions Over Time', fontsize=13, y=1.02)
    plt.tight_layout(); return fig


def plot_fields(snaps, hist, p, indices=None):
    """Phase fields rendered from granule positions."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    ext = [0, p.Lx, 0, p.Ly]
    fig, ax = plt.subplots(3, nc, figsize=(3.2*nc, 9))
    lbl = [r'$\phi_f$', r'$\phi_i$', r'$\phi_v$']
    cm = ['Oranges', 'Blues', 'Greens']

    for c, si in enumerate(indices):
        pf, pi, pv = snaps[si][0], snaps[si][1], snaps[si][2]
        flds = [pf, pi, pv]
        t = hist[si]['time']
        for r in range(3):
            a = ax[r, c]; vx = max(.01, flds[r].max()*1.05)
            im = a.imshow(flds[r].T, origin='lower', extent=ext,
                          cmap=cm[r], vmin=0, vmax=vx)
            a.set_title(f't={t:.1f}h', fontsize=8)
            if c == 0: a.set_ylabel(lbl[r], fontsize=11)
            plt.colorbar(im, ax=a, fraction=.046, pad=.04)
    fig.suptitle('Rendered Phase Fields', fontsize=13, y=1.01)
    plt.tight_layout(); return fig


def plot_timeseries(hist, p):
    t = [h['time'] for h in hist]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) Cluster counts
    ax[0,0].plot(t, [h['func_nc'] for h in hist], 'C1-o', ms=3, label='Functional')
    ax[0,0].plot(t, [h['void_nc'] for h in hist], 'C2-s', ms=3, label='Void')
    ax[0,0].plot(t, [h['inert_nc'] for h in hist], 'C0-^', ms=3, label='Inert')
    ax[0,0].set(xlabel='time (h)', ylabel='# clusters',
                title='Cluster Count (↓ = coalescence)')
    ax[0,0].legend()

    # (0,1) Largest cluster fraction
    ax[0,1].plot(t, [h['func_lf'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[0,1].plot(t, [h['void_lf'] for h in hist], 'C2--', label='Void')
    ax[0,1].axhline(1, ls=':', c='gray', lw=0.8)
    ax[0,1].set(xlabel='time (h)', ylabel='largest / total',
                title='Connectivity (1 = percolated)')
    ax[0,1].legend()

    # (0,2) Bridge count
    ax[0,2].plot(t, [h['n_bridges'] for h in hist], 'C3-', lw=2)
    ax[0,2].set(xlabel='time (h)', ylabel='# bridges',
                title='Active Cell Bridges')

    # (1,0) Displacement
    ax[1,0].plot(t, [h['disp_func'] for h in hist], 'C1-', lw=2, label='Functional')
    ax[1,0].plot(t, [h['disp_inert'] for h in hist], 'C0--', label='Inert')
    ax[1,0].set(xlabel='time (h)', ylabel='mean disp (µm)',
                title='Granule Displacement')
    ax[1,0].legend()

    # (1,1) Tissue & packing
    ax[1,1].plot(t, [h['tissue_frac'] for h in hist], 'C3-', lw=2,
                 label=r'Tissue ($\phi_f > 0.5$)')
    ax[1,1].plot(t, [h['packing_func_rich'] for h in hist], 'C1--',
                 label='Packing in func-rich')
    ax[1,1].set(xlabel='time (h)', ylabel='fraction', title='Tissue Remodeling')
    ax[1,1].legend()

    # (1,2) Max cluster area
    ax[1,2].plot(t, [h['func_max_area'] for h in hist], 'C1-', lw=2)
    ax[1,2].set(xlabel='time (h)', ylabel='area (µm²)',
                title='Largest Functional Cluster Area')

    plt.tight_layout(); return fig


def plot_composite(snaps, hist, p, indices=None):
    """RGB composite from rendered fields."""
    if indices is None:
        n = len(snaps)
        indices = sorted(set([0, n//4, n//2, 3*n//4, n-1]))
    nc = len(indices)
    fig, axes = plt.subplots(1, nc, figsize=(3.8*nc, 3.5))
    if nc == 1: axes = [axes]
    for c, si in enumerate(indices):
        pf, pi, pv = snaps[si][0], snaps[si][1], snaps[si][2]
        mx = max(pf.max(), pi.max(), pv.max(), 0.01)
        rgb = np.stack([pf/mx, pv/mx, pi/mx], axis=-1)
        axes[c].imshow(np.clip(np.transpose(rgb,(1,0,2)),0,1),
                       origin='lower', extent=[0,p.Lx,0,p.Ly])
        axes[c].set_title(f"t={hist[si]['time']:.1f}h", fontsize=9)
    fig.suptitle("R=functional  G=void  B=inert", fontsize=11, y=1.02)
    plt.tight_layout(); return fig


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  Overdamped Particle Dynamics: Cell-Driven Granular Rearrangement")
    print("="*65)

    p = Params()
    print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm")
    print(f"  R_func={p.R_func_mean:.0f}±{p.R_func_std:.0f} µm, "
          f"R_inert={p.R_inert_mean:.0f}±{p.R_inert_std:.0f} µm")
    print(f"  Cells/granule={p.n_cells_per_granule}, "
          f"k_cell={p.k_cell}, L_max={p.L_max} µm")
    print(f"  Simulation: {p.t_total:.0f} h, dt={p.dt:.2f} h")

    hist, snaps, p, gs = run(p)

    fig1 = plot_granules(snaps, hist, p)
    fig2 = plot_fields(snaps, hist, p)
    fig3 = plot_timeseries(hist, p)
    fig4 = plot_composite(snaps, hist, p)
    plt.show()

    h0, hf = hist[0], hist[-1]
    print("\n" + "="*65)
    print("  SUMMARY")
    print("="*65)
    print(f"  Functional: {h0['func_nc']} → {hf['func_nc']} clusters")
    print(f"    Largest frac: {h0['func_lf']:.1%} → {hf['func_lf']:.1%}")
    ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
          'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
    print(f"    Topology: {ft}")
    print(f"  Void: {h0['void_nc']} → {hf['void_nc']} clusters")
    vt = ('CONTINUOUS' if hf['void_lf'] > 0.8 else
          'FEW POCKETS' if hf['void_nc'] < 6 else 'MANY POCKETS')
    print(f"    Topology: {vt}")
    print(f"  Bridges: {h0['n_bridges']} → {hf['n_bridges']}")
    print(f"  Displacement: func={hf['disp_func']:.1f}µm, inert={hf['disp_inert']:.1f}µm")
    print(f"  Tissue: {h0['tissue_frac']:.1%} → {hf['tissue_frac']:.1%}")
    print(f"  Max cluster area: {h0['func_max_area']:.0f} → {hf['func_max_area']:.0f} µm²")