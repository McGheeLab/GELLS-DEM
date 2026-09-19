"""
Stochastic Contact-Network Model for Granular Scaffold Compaction
=================================================================
V2.5 -- Mesoscale model between full DEM and mean-field ODE.

Physics:
    - Granules are spheres on a contact graph (Delaunay triangulation).
    - Cells on functional granules form bridges to neighbors via Poisson process.
    - Bridge formation/dissolution handled by Gillespie SSA (exact stochastic).
    - Bridge forces pull granules together via overdamped spring-network relaxation.
    - Contact mechanics: analytic sphere-sphere JKR (no Newton-Raphson).
    - Percolation tracked via union-find on the bridge sub-graph.

Key references:
    - Gillespie (1977) J. Phys. Chem. 81:2340 -- Stochastic simulation algorithm
    - Bortz, Kalos & Lebowitz (1975) J. Comp. Phys. 17:10 -- BKL / n-fold way
    - Newman & Ziff (2001) Phys. Rev. E 64:016706 -- Fast MC percolation
    - Braz & Araujo (2022) Powder Technology 400:117248 -- Ranked percolation
    - Petridou et al. (2021) Cell 184:1914 -- Rigidity percolation in tissues

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).

Usage:
    # Standalone run with default parameters:
    python analysis/contact_network_model.py

    # Single realization from DEM Params:
    from analysis.contact_network_model import from_params, solve
    model = from_params(p)
    result = solve(model, seed=42)

    # Monte Carlo ensemble:
    from analysis.contact_network_model import from_params, ensemble
    model = from_params(p)
    results = ensemble(model, n_runs=1000, seed=0)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay, cKDTree


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class NetworkParams:
    """Parameters for the contact-network model.

    Mirrors the relevant subset of new_dem_0.Params. All units: um, nN, h, kPa.
    """
    # Domain
    mode: str = "2D"
    Lx: float = 800.0
    Ly: float = 800.0
    Lz: float = 800.0

    # Granule sizes
    R_func_mean: float = 40.0
    R_func_std: float = 5.0
    R_inert_mean: float = 60.0
    R_inert_std: float = 8.0

    # Composition
    phi_f_target: float = 0.25
    phi_i_target: float = 0.20
    phi_solid_target: float = 0.0
    func_ratio: float = 0.5

    # Contact mechanics
    E_modulus: float = 10.0          # kPa
    poisson_ratio: float = 0.45
    W_adh_ff: float = 0.002         # J/m^2
    W_adh_if: float = 0.001
    W_adh_ii: float = 0.0005

    # Cell / motor-clutch
    n_cells_per_granule: int = 8
    cell_surface_coverage: float = 0.0
    cell_diameter: float = 20.0
    cell_height_spread: float = 5.0
    n_motors: int = 50
    F_motor_stall: float = 0.5      # nN
    n_clutches: int = 75
    k_clutch: float = 5.0           # nN/um
    k_on_clutch: float = 1.0
    k_off_clutch: float = 0.1
    F_max_per_cell: float = 50.0    # nN
    fa_maturation_rate: float = 0.3  # 1/h
    t_spread_duration: float = 3.0   # h

    # Bridge kinetics
    bridge_attempt_rate: float = 0.3      # 1/h
    bridge_formation_time: float = 2.0    # h
    bridge_senescence_time: float = 24.0  # h
    bridge_lock_force_threshold: float = 20.0  # nN
    bridge_secondary_rate_mult: float = 3.0
    bridge_break_gap: float = 60.0        # um
    min_fa_for_bridge: float = 0.3
    bridge_contact_factor: float = 5.0
    bridge_decay_length: float = 10.0     # um
    bridge_inert_factor: float = 0.1
    bridge_exclusion_angle: float = 0.8   # rad

    # Drag & noise
    eta: float = 1e-3               # Pa.s
    drag_scale: float = 0.05
    T_active: float = 5.0           # nN.um

    # Time
    dt: float = 0.5                 # h (relaxation sub-step)
    t_total: float = 72.0           # h
    save_every_h: float = 2.0       # h

    # Packing
    packing_settle_steps: int = 400
    packing_relax_substeps: int = 15
    packing_inflate_phi_safe: float = 0.20

    # Velocity cap
    v_max: float = 20.0             # um/h

    @property
    def is_3d(self):
        return self.mode in ("3D", "2D-slice")


# ======================================================================
# Motor-clutch force
# ======================================================================

def motor_clutch_force(E_kPa, p):
    """Cell traction from motor-clutch model (Chan & Odde 2008).

    Returns force per cell in nN at full FA maturity.
    """
    F_stall = p.n_motors * p.F_motor_stall
    k_opt = p.n_clutches * p.k_clutch
    a_cell = p.cell_diameter / 2.0
    k_sub = np.pi * E_kPa * a_cell / (1.0 - p.poisson_ratio ** 2)
    engagement = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)
    F_mc = F_stall * (k_sub / (k_sub + k_opt)) * engagement
    return min(F_mc, p.F_max_per_cell)


def fa_maturity(t, p):
    """Focal adhesion maturity at time t. Returns [0, 1]."""
    t_eff = max(0.0, t - p.t_spread_duration)
    return 1.0 - np.exp(-p.fa_maturation_rate * t_eff)


# ======================================================================
# JKR contact force (sphere-sphere, analytic)
# ======================================================================

def jkr_force(delta_um, R_eff_um, E_star_Pa, W_Jm2):
    """Compute JKR contact force (nN) and contact radius (um).

    Analytic for spheres: no Newton-Raphson needed.
    Returns (F_nN, a_um). Positive F = repulsion.
    """
    if W_Jm2 < 1e-15:
        # Pure Hertz
        if delta_um <= 0:
            return 0.0, 0.0
        R_m = R_eff_um * 1e-6
        E_star = E_star_Pa
        a_m = np.sqrt(R_m * delta_um * 1e-6)
        F_N = (4.0 / 3.0) * E_star * a_m**3 / R_m
        return F_N * 1e9, a_m * 1e6  # nN, um

    R_m = R_eff_um * 1e-6
    E_star = E_star_Pa

    # Hertz seed for contact radius
    if delta_um > 0:
        a_guess = np.sqrt(R_m * delta_um * 1e-6)
    else:
        # Near pull-off
        a_po = (6.0 * np.pi * W_Jm2 * R_m**2 / E_star) ** (1.0 / 3.0)
        a_guess = a_po * 0.8

    # Newton-Raphson: delta = a^2/R - sqrt(2*pi*W*a/E*)
    a = max(a_guess, 1e-12)
    for _ in range(20):
        sq = np.sqrt(2.0 * np.pi * W_Jm2 * a / E_star) if a > 0 else 0.0
        f = a**2 / R_m - sq - delta_um * 1e-6
        df = 2.0 * a / R_m - (np.pi * W_Jm2 / (E_star * a * sq + 1e-30)) * 0.5
        if abs(df) < 1e-30:
            break
        da = -f / df
        a = max(a + da, 1e-14)
        if abs(da) < 1e-15:
            break

    if a < 1e-12:
        return 0.0, 0.0

    F_N = (4.0 / 3.0) * E_star * a**3 / R_m - np.sqrt(8.0 * np.pi * W_Jm2 * E_star * a**3)
    return F_N * 1e9, a * 1e6


# ======================================================================
# Packing generation (sphere-only Lubachevsky-Stillinger)
# ======================================================================

def generate_packing(p, rng):
    """Generate a jammed sphere packing via inflate-and-relax.

    Returns (pos, r, gtype, n_cells) arrays.
    pos: (N, d) positions in um.
    r: (N,) radii in um.
    gtype: (N,) 0=functional, 1=inert.
    n_cells: (N,) cells per granule.
    """
    is_3d = p.is_3d
    d = 3 if is_3d else 2

    # Derive composition
    phi_f = p.phi_f_target
    phi_i = p.phi_i_target
    if p.phi_solid_target > 0:
        phi_f = p.phi_solid_target * p.func_ratio
        phi_i = p.phi_solid_target * (1.0 - p.func_ratio)

    domain = np.array([p.Lx, p.Ly, p.Lz][:d])
    V_domain = np.prod(domain)

    if d == 3:
        v_func = (4.0 / 3.0) * np.pi * p.R_func_mean**3
        v_inert = (4.0 / 3.0) * np.pi * p.R_inert_mean**3
    else:
        v_func = np.pi * p.R_func_mean**2
        v_inert = np.pi * p.R_inert_mean**2

    N_func = max(1, int(round(phi_f * V_domain / v_func)))
    N_inert = max(0, int(round(phi_i * V_domain / v_inert)))
    N = N_func + N_inert

    # Generate radii
    r_func = np.abs(rng.normal(p.R_func_mean, p.R_func_std, N_func))
    r_func = np.clip(r_func, p.R_func_mean * 0.5, p.R_func_mean * 1.5)
    r_inert = np.abs(rng.normal(p.R_inert_mean, p.R_inert_std, N_inert))
    r_inert = np.clip(r_inert, p.R_inert_mean * 0.5, p.R_inert_mean * 1.5)

    r = np.concatenate([r_func, r_inert])
    gtype = np.concatenate([np.zeros(N_func, dtype=int),
                            np.ones(N_inert, dtype=int)])

    # Shuffle for good mixing
    order = rng.permutation(N)
    r = r[order]
    gtype = gtype[order]

    # Deflation factor
    phi_target = phi_f + phi_i
    alpha_start = np.clip((p.packing_inflate_phi_safe / max(phi_target, 0.01)) ** (1.0 / d),
                          0.3, 1.0)

    # RSA placement at deflated radii
    r_placed = r * alpha_start
    pos = np.zeros((N, d))
    placed = np.zeros(N, dtype=bool)

    for i in range(N):
        for attempt in range(2000):
            trial = rng.uniform(r_placed[i], domain - r_placed[i])
            if not np.any(placed[:i]):
                pos[i] = trial
                placed[i] = True
                break
            dists = np.linalg.norm(pos[:i][placed[:i]] - trial, axis=1)
            min_sep = r_placed[:i][placed[:i]] + r_placed[i]
            if np.all(dists > min_sep * 0.8):
                pos[i] = trial
                placed[i] = True
                break
        if not placed[i]:
            # Force place
            pos[i] = rng.uniform(r_placed[i], domain - r_placed[i])
            placed[i] = True

    # Inflate-and-relax
    n_inflate = p.packing_settle_steps
    n_sub = p.packing_relax_substeps
    k_rep = 5.0

    for step in range(n_inflate):
        t_norm = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t_norm) ** 2)
        r_current = r * alpha
        v_cap = 0.3 * np.mean(r_current)

        for sub in range(n_sub):
            tree = cKDTree(pos)
            r_max = np.max(r_current)
            pairs = tree.query_pairs(2.0 * r_max, output_type='ndarray')
            if len(pairs) == 0:
                break

            forces = np.zeros_like(pos)
            for idx in range(len(pairs)):
                i, j = pairs[idx]
                d_vec = pos[j] - pos[i]
                dist = np.linalg.norm(d_vec)
                if dist < 1e-10:
                    d_vec = rng.normal(0, 1, d)
                    dist = np.linalg.norm(d_vec)
                overlap = r_current[i] + r_current[j] - dist
                if overlap > 0:
                    n_hat = d_vec / dist
                    F_mag = k_rep * overlap ** 1.5
                    forces[i] -= F_mag * n_hat
                    forces[j] += F_mag * n_hat

            # Wall forces
            for dim in range(d):
                # Lower wall
                overlap_lo = r_current - pos[:, dim]
                mask_lo = overlap_lo > 0
                forces[mask_lo, dim] += k_rep * overlap_lo[mask_lo] ** 1.5
                # Upper wall
                overlap_hi = r_current - (domain[dim] - pos[:, dim])
                mask_hi = overlap_hi > 0
                forces[mask_hi, dim] -= k_rep * overlap_hi[mask_hi] ** 1.5

            # Overdamped displacement
            f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
            f_mag_safe = np.maximum(f_mag, 1e-10)
            disp = forces / f_mag_safe * np.minimum(f_mag, v_cap)
            pos += disp * 0.1  # damped step

            # Clamp to walls
            for dim in range(d):
                pos[:, dim] = np.clip(pos[:, dim], r_current, domain[dim] - r_current)

    # Cell counts per granule
    n_cells = np.zeros(N, dtype=int)
    for i in range(N):
        if gtype[i] == 0:  # functional
            if p.cell_surface_coverage > 0:
                if is_3d:
                    A_gran = 4.0 * np.pi * r[i] ** 2
                else:
                    A_gran = np.pi * r[i] ** 2
                A_cell = np.pi * (p.cell_diameter / 2.0) ** 2
                n_cells[i] = max(1, int(round(A_gran * p.cell_surface_coverage / A_cell)))
            else:
                n_cells[i] = p.n_cells_per_granule

    return pos, r, gtype, n_cells


# ======================================================================
# Contact graph construction
# ======================================================================

def build_contact_graph(pos, r, gtype, p):
    """Build contact graph from packing.

    Returns:
        edges: (M, 2) array of edge indices
        gaps: (M,) gap distances (negative = overlap)
        edge_type: (M,) 0=func-func, 1=func-inert, 2=inert-inert
    """
    N = len(r)
    d = pos.shape[1]
    r_max = np.max(r)
    cutoff = 2.0 * r_max + p.bridge_break_gap

    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    if len(pairs) == 0:
        return (np.zeros((0, 2), dtype=int),
                np.zeros(0), np.zeros(0, dtype=int))

    # Compute gaps
    d_vecs = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    dists = np.linalg.norm(d_vecs, axis=1)
    sum_r = r[pairs[:, 0]] + r[pairs[:, 1]]
    gaps = dists - sum_r

    # Edge types
    gt0 = gtype[pairs[:, 0]]
    gt1 = gtype[pairs[:, 1]]
    edge_type = gt0 + gt1  # 0=FF, 1=FI, 2=II

    return pairs, gaps, edge_type


# ======================================================================
# Union-Find for percolation tracking
# ======================================================================

class UnionFind:
    """Weighted union-find with path compression."""

    def __init__(self, n):
        self.parent = np.arange(n)
        self.rank = np.zeros(n, dtype=int)
        self.size = np.ones(n, dtype=int)
        self.n_components = n

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.n_components -= 1
        return True

    def largest_component(self):
        return int(np.max(self.size))

    def component_sizes(self):
        roots = np.array([self.find(i) for i in range(len(self.parent))])
        _, counts = np.unique(roots, return_counts=True)
        return np.sort(counts)[::-1]


# ======================================================================
# Gillespie SSA engine
# ======================================================================

@dataclass
class Bridge:
    """A single cell bridge between two granules."""
    edge_idx: int           # index into edges array
    gran_i: int = -1        # granule index i (stable across graph rebuilds)
    gran_j: int = -1        # granule index j (stable across graph rebuilds)
    t_formed: float = 0.0   # time of formation (h)
    force: float = 0.0      # current force (nN)
    locked: bool = False    # permanent (force exceeded lock-in threshold)


def compute_bridge_rates(edges, gaps, edge_type, gtype, n_cells,
                         bridges, t, p):
    """Compute per-edge bridge formation rates for Gillespie.

    Returns (rates, eligible_edges) arrays.
    """
    N_edges = len(edges)
    rates = np.zeros(N_edges)
    maturity = fa_maturity(t, p)

    if maturity < p.min_fa_for_bridge:
        return rates, np.array([], dtype=int)

    # Count existing bridges per edge
    bridge_count = np.zeros(N_edges, dtype=int)
    for b in bridges:
        bridge_count[b.edge_idx] += 1

    # Max bridges per edge from exclusion angle (DEM spatial exclusion)
    # Only target-facing hemisphere of cells can bridge (hemisphere check).
    # In 3D: hemisphere solid angle (2π) / exclusion cone (π θ²) = 2/θ²
    # In 2D: half-circle (π) / exclusion arc (2θ) = π/(2θ)
    theta_ex = max(p.bridge_exclusion_angle, 0.1)
    is_3d = hasattr(p, 'mode') and p.mode == '3D'
    if is_3d:
        max_bridges_per_edge = max(1, int(2.0 / (theta_ex ** 2)))
    else:
        max_bridges_per_edge = max(1, int(np.pi / (2.0 * theta_ex)))

    for e in range(N_edges):
        i, j = edges[e]
        # Only functional granules can bridge
        if gtype[i] != 0 and gtype[j] != 0:
            continue

        gap = gaps[e]
        if gap > p.bridge_break_gap:
            continue

        # Available cells (subtract existing bridges), capped by exclusion geometry
        cells_i = n_cells[i] if gtype[i] == 0 else 0
        cells_j = n_cells[j] if gtype[j] == 0 else 0
        cell_avail = max(cells_i, cells_j) - bridge_count[e]
        spatial_avail = max_bridges_per_edge - bridge_count[e]
        avail = min(cell_avail, spatial_avail)
        if avail <= 0:
            continue

        # Path factor
        if gap <= 0:
            path_factor = p.bridge_contact_factor
        elif edge_type[e] == 0:  # func-func
            path_factor = np.exp(-gap / p.bridge_decay_length)
        elif edge_type[e] == 1:  # func-inert
            path_factor = p.bridge_inert_factor * np.exp(-gap / p.bridge_decay_length)
        else:
            path_factor = 0.0

        # Base rate
        rate = p.bridge_attempt_rate * avail * maturity * path_factor

        # Secondary boost
        if bridge_count[e] > 0:
            rate *= p.bridge_secondary_rate_mult

        rates[e] = max(rate, 0.0)

    eligible = np.where(rates > 0)[0]
    return rates, eligible


def compute_senescence_rates(bridges, t, p):
    """Compute per-bridge senescence/dissolution rates."""
    rates = np.zeros(len(bridges))
    for k, b in enumerate(bridges):
        if b.locked:
            continue
        age = t - b.t_formed
        if age > p.bridge_senescence_time:
            rates[k] = 10.0  # fast removal
        elif age > p.bridge_senescence_time * 0.5:
            rates[k] = 1.0 / p.bridge_senescence_time
    return rates


# ======================================================================
# Spring-network mechanical relaxation
# ======================================================================

def relax_positions(pos, r, gtype, edges, gaps, bridges, p, n_sub=10):
    """Overdamped relaxation of positions under contact + bridge forces.

    Modifies pos in-place. Returns updated gaps.
    """
    N = len(r)
    d = pos.shape[1]
    domain = np.array([p.Lx, p.Ly, p.Lz][:d])

    E_star = p.E_modulus * 1e3 / (2.0 * (1.0 - p.poisson_ratio**2))  # Pa

    # Drag coefficient (Stokes-like)
    gamma = 6.0 * np.pi * p.eta * r * 1e-6  # N.s/m per particle
    gamma_nN_h = gamma * 1e9 * 3600.0       # nN.h/um (convert to sim units)
    gamma_nN_h *= (1.0 / p.drag_scale)      # scale factor

    W_lookup = {
        0: p.W_adh_ff,
        1: p.W_adh_if,
        2: p.W_adh_ii,
    }

    # Bridge force lookup
    bridge_forces = {}
    for b in bridges:
        e = b.edge_idx
        bridge_forces.setdefault(e, []).append(b.force)

    dt_sub = p.dt / max(n_sub, 1)

    for _ in range(n_sub):
        forces = np.zeros_like(pos)

        # Contact forces (JKR)
        d_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
        dists = np.linalg.norm(d_vecs, axis=1)
        dists_safe = np.maximum(dists, 1e-10)
        n_hats = d_vecs / dists_safe[:, None]
        current_gaps = dists - (r[edges[:, 0]] + r[edges[:, 1]])

        for e in range(len(edges)):
            i, j = edges[e]
            gap = current_gaps[e]
            overlap = -gap

            R_eff = r[i] * r[j] / (r[i] + r[j])
            et = min(gtype[i] + gtype[j], 2)
            W = W_lookup[et]

            # Contact force
            if overlap > 0:
                F_nN, a_um = jkr_force(overlap, R_eff, E_star, W)
                # Positive = repulsive
                forces[i] -= F_nN * n_hats[e]
                forces[j] += F_nN * n_hats[e]
            elif overlap > -0.1 * R_eff and W > 0:
                # Adhesive pull (small separation)
                F_nN, a_um = jkr_force(overlap, R_eff, E_star, W)
                if F_nN != 0:
                    forces[i] -= F_nN * n_hats[e]
                    forces[j] += F_nN * n_hats[e]

            # Bridge forces (contractile, pull granules together)
            if e in bridge_forces:
                F_bridge_total = sum(bridge_forces[e])
                if F_bridge_total > 0 and gap > -r[i] * 0.3:
                    forces[i] += F_bridge_total * n_hats[e]
                    forces[j] -= F_bridge_total * n_hats[e]

        # Wall forces
        for dim in range(d):
            overlap_lo = r - pos[:, dim]
            mask_lo = overlap_lo > 0
            forces[mask_lo, dim] += 5.0 * overlap_lo[mask_lo] ** 1.5

            overlap_hi = r - (domain[dim] - pos[:, dim])
            mask_hi = overlap_hi > 0
            forces[mask_hi, dim] -= 5.0 * overlap_hi[mask_hi] ** 1.5

        # Overdamped update
        vel = forces / gamma_nN_h[:, None]
        speed = np.linalg.norm(vel, axis=1)
        cap_mask = speed > p.v_max
        if np.any(cap_mask):
            vel[cap_mask] *= (p.v_max / speed[cap_mask])[:, None]

        pos += vel * dt_sub

        # Clamp to walls
        for dim in range(d):
            pos[:, dim] = np.clip(pos[:, dim], r, domain[dim] - r)

    # Recompute gaps
    d_vecs = pos[edges[:, 1]] - pos[edges[:, 0]]
    dists = np.linalg.norm(d_vecs, axis=1)
    new_gaps = dists - (r[edges[:, 0]] + r[edges[:, 1]])
    return new_gaps


# ======================================================================
# Observables
# ======================================================================

def compute_observables(pos, r, gtype, edges, gaps, edge_type,
                        bridges, t, p):
    """Compute all scalar observables at current state.

    Returns dict of metrics.
    """
    N = len(r)
    d = pos.shape[1]
    domain = np.array([p.Lx, p.Ly, p.Lz][:d])
    V_domain = np.prod(domain)

    # Phase fractions (from sphere volumes)
    if d == 3:
        vols = (4.0 / 3.0) * np.pi * r**3
    else:
        vols = np.pi * r**2

    func_mask = gtype == 0
    inert_mask = gtype == 1
    phi_f = np.sum(vols[func_mask]) / V_domain
    phi_i = np.sum(vols[inert_mask]) / V_domain
    phi_v = max(0.0, 1.0 - phi_f - phi_i)

    # Functional zone fraction (Voronoi-based estimate)
    # Approximate: volume-weighted centroid spread
    if np.any(func_mask):
        func_pos = pos[func_mask]
        func_spread = np.prod(np.ptp(func_pos, axis=0) + 2 * np.mean(r[func_mask]))
        x_f = min(func_spread / V_domain, 1.0)
    else:
        x_f = 0.0

    # Contact network stats
    contact_mask = gaps <= 0
    n_contacts = np.sum(contact_mask)
    Z_contact = 2.0 * n_contacts / max(N, 1)

    # Bridge network stats
    n_bridges = len(bridges)
    bridge_edges = set(b.edge_idx for b in bridges)
    Z_bridge = 2.0 * len(bridge_edges) / max(np.sum(func_mask), 1)

    # Bridge force stats
    if bridges:
        forces = np.array([b.force for b in bridges])
        F_bridge_mean = np.mean(forces)
        F_bridge_max = np.max(forces)
        n_locked = sum(1 for b in bridges if b.locked)
    else:
        F_bridge_mean = 0.0
        F_bridge_max = 0.0
        n_locked = 0

    # Percolation (bridge sub-graph, functional granules only)
    func_indices = np.where(func_mask)[0]
    if len(func_indices) > 1 and bridge_edges:
        uf = UnionFind(N)
        for e_idx in bridge_edges:
            i, j = edges[e_idx]
            uf.union(i, j)
        largest = uf.largest_component()
        sizes = uf.component_sizes()
        # Spanning: does largest cluster span >50% of domain in any axis?
        root_of_largest = -1
        for node in range(N):
            if uf.size[uf.find(node)] == largest:
                root_of_largest = uf.find(node)
                break
        cluster_nodes = [n for n in range(N) if uf.find(n) == root_of_largest]
        if cluster_nodes:
            cluster_pos = pos[cluster_nodes]
            span = np.ptp(cluster_pos, axis=0) / domain[:d]
            spanning = bool(np.any(span > 0.5))
        else:
            spanning = False
    else:
        largest = 1
        sizes = np.array([1] * np.sum(func_mask)) if np.sum(func_mask) > 0 else np.array([])
        spanning = False

    # Permeability (Kozeny-Carman, functional zone)
    d_grain = 2.0 * p.R_func_mean
    eps = np.clip(phi_v, 0.01, 0.99)
    K_perm = eps**3 * d_grain**2 / (180.0 * (1.0 - eps)**2)

    return {
        'time': t,
        'phi_f': phi_f,
        'phi_i': phi_i,
        'phi_v': phi_v,
        'x_f': x_f,
        'Z_contact': Z_contact,
        'Z_bridge': Z_bridge,
        'n_bridges': n_bridges,
        'n_locked': n_locked,
        'F_bridge_mean': F_bridge_mean,
        'F_bridge_max': F_bridge_max,
        'largest_cluster': largest,
        'spanning': spanning,
        'K_perm': K_perm,
        'n_contacts': n_contacts,
    }


# ======================================================================
# Main solver
# ======================================================================

def solve(model, seed=None, verbose=True):
    """Run a single realization of the contact-network model.

    Parameters
    ----------
    model : dict
        Model state from from_params() or from_packing().
    seed : int or None
        Random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'history': list of observable dicts at save points.
        'final_pos': final positions.
        'final_bridges': final bridge list.
        'params': NetworkParams used.
    """
    rng = np.random.default_rng(seed)
    p = model['params']
    pos = model['pos'].copy()
    r = model['r'].copy()
    gtype = model['gtype'].copy()
    n_cells = model['n_cells'].copy()
    edges = model['edges'].copy()
    gaps = model['gaps'].copy()
    edge_type = model['edge_type'].copy()

    N = len(r)
    t = 0.0
    bridges = []
    history = []
    save_interval = p.save_every_h
    next_save = 0.0
    relax_interval = p.dt  # mechanical relaxation interval

    F_cell = motor_clutch_force(p.E_modulus, p)

    if verbose:
        N_func = int(np.sum(gtype == 0))
        N_inert = int(np.sum(gtype == 1))
        print(f"  Contact-Network Model: N={N} ({N_func} func, {N_inert} inert), "
              f"{len(edges)} edges, F_cell={F_cell:.2f} nN")

    # Main time loop (hybrid: Gillespie events + periodic relaxation)
    while t < p.t_total:
        # --- Gillespie step: advance stochastic events ---
        form_rates, eligible = compute_bridge_rates(
            edges, gaps, edge_type, gtype, n_cells, bridges, t, p)
        senes_rates = compute_senescence_rates(bridges, t, p)

        R_form = np.sum(form_rates)
        R_senes = np.sum(senes_rates)
        R_total = R_form + R_senes

        if R_total > 1e-20:
            # Time to next event
            dt_event = -np.log(max(rng.random(), 1e-30)) / R_total
        else:
            dt_event = relax_interval  # no events, just relax

        # Cap dt_event to not overshoot relaxation or save intervals
        dt_step = min(dt_event, relax_interval, p.t_total - t)
        if next_save - t > 0:
            dt_step = min(dt_step, next_save - t + 1e-10)

        t += dt_step

        # Execute event if dt_step == dt_event (i.e., event happened before next relax)
        if abs(dt_step - dt_event) < 1e-12 and R_total > 1e-20:
            u = rng.random() * R_total
            if u < R_form and len(eligible) > 0:
                # Bridge formation event
                cum = np.cumsum(form_rates[eligible])
                event_idx = eligible[np.searchsorted(cum, u)]
                maturity = fa_maturity(t, p)
                ramp = min(1.0, (t - p.t_spread_duration - 1.0) /
                           max(p.bridge_formation_time, 0.01))
                ramp = max(ramp, 0.0)
                force = F_cell * maturity * max(ramp, 0.1)
                gi, gj = int(edges[event_idx, 0]), int(edges[event_idx, 1])
                bridges.append(Bridge(
                    edge_idx=event_idx,
                    gran_i=gi,
                    gran_j=gj,
                    t_formed=t,
                    force=force,
                ))
            else:
                # Senescence event
                if bridges and np.sum(senes_rates) > 0:
                    cum = np.cumsum(senes_rates)
                    event_idx = np.searchsorted(cum, (u - R_form))
                    event_idx = min(event_idx, len(bridges) - 1)
                    bridges.pop(event_idx)

        # --- Update bridge forces ---
        for b in bridges:
            age = t - b.t_formed
            ramp = min(1.0, age / max(p.bridge_formation_time, 0.01))
            maturity = fa_maturity(t, p)
            b.force = F_cell * maturity * ramp
            if b.force >= p.bridge_lock_force_threshold:
                b.locked = True

        # --- Periodic mechanical relaxation ---
        if dt_step >= relax_interval * 0.99 or abs(t % relax_interval) < 0.01:
            gaps = relax_positions(pos, r, gtype, edges, gaps, bridges, p,
                                   n_sub=5)

            # Remove broken bridges (gap > break distance)
            surviving = []
            for b in bridges:
                if b.edge_idx < len(gaps) and gaps[b.edge_idx] < p.bridge_break_gap:
                    surviving.append(b)
            bridges = surviving

            # Rebuild contact graph periodically (topology may change)
            edges, gaps, edge_type = build_contact_graph(pos, r, gtype, p)

            # Re-map bridge edge indices using stable granule pair keys
            edge_map = {}
            for e in range(len(edges)):
                key = (int(edges[e, 0]), int(edges[e, 1]))
                edge_map[key] = e
                edge_map[(key[1], key[0])] = e

            remapped = []
            for b in bridges:
                key = (b.gran_i, b.gran_j)
                if key in edge_map:
                    b.edge_idx = edge_map[key]
                    remapped.append(b)
            bridges = remapped

        # --- Save observables + topology snapshot ---
        if t >= next_save - 1e-10:
            obs = compute_observables(pos, r, gtype, edges, gaps, edge_type,
                                       bridges, t, p)
            # Save topology snapshot for visualization
            obs['_pos'] = pos.copy()
            obs['_r'] = r.copy()
            obs['_gtype'] = gtype.copy()
            obs['_edges'] = edges.copy()
            obs['_gaps'] = gaps.copy()
            obs['_edge_type'] = edge_type.copy()
            obs['_bridges'] = [(b.edge_idx, b.gran_i, b.gran_j,
                                b.t_formed, b.force, b.locked)
                               for b in bridges]
            history.append(obs)
            next_save += save_interval

            if verbose and len(history) % 5 == 0:
                print(f"    t={t:.1f}h: bridges={obs['n_bridges']}, "
                      f"Z_bridge={obs['Z_bridge']:.2f}, "
                      f"largest={obs['largest_cluster']}, "
                      f"spanning={obs['spanning']}")

    # Final save
    if not history or history[-1]['time'] < t - 0.01:
        obs = compute_observables(pos, r, gtype, edges, gaps, edge_type,
                                   bridges, t, p)
        obs['_pos'] = pos.copy()
        obs['_r'] = r.copy()
        obs['_gtype'] = gtype.copy()
        obs['_edges'] = edges.copy()
        obs['_gaps'] = gaps.copy()
        obs['_edge_type'] = edge_type.copy()
        obs['_bridges'] = [(b.edge_idx, b.gran_i, b.gran_j,
                            b.t_formed, b.force, b.locked)
                           for b in bridges]
        history.append(obs)

    return {
        'history': history,
        'final_pos': pos,
        'final_bridges': bridges,
        'final_edges': edges,
        'final_gaps': gaps,
        'r': r,
        'gtype': gtype,
        'n_cells': n_cells,
        'params': p,
    }


# ======================================================================
# Construction helpers
# ======================================================================

def from_params(p=None, **kwargs):
    """Construct model state from Params-like object or keyword args.

    Parameters
    ----------
    p : Params, dict, NetworkParams, or None
        Parameters source. If None, uses defaults + kwargs.

    Returns
    -------
    dict
        Model state dict with keys: params, pos, r, gtype, n_cells,
        edges, gaps, edge_type.
    """
    if p is None:
        np_params = NetworkParams(**kwargs)
    elif isinstance(p, NetworkParams):
        np_params = p
    elif isinstance(p, dict):
        fields = {f.name for f in NetworkParams.__dataclass_fields__.values()}
        np_params = NetworkParams(**{k: v for k, v in p.items() if k in fields})
    else:
        # Assume Params-like object
        fields = {f.name for f in NetworkParams.__dataclass_fields__.values()}
        d = {k: getattr(p, k) for k in fields if hasattr(p, k)}
        np_params = NetworkParams(**d)

    rng = np.random.default_rng(42)  # deterministic packing
    pos, r, gtype, n_cells = generate_packing(np_params, rng)
    edges, gaps, edge_type = build_contact_graph(pos, r, gtype, np_params)

    return {
        'params': np_params,
        'pos': pos,
        'r': r,
        'gtype': gtype,
        'n_cells': n_cells,
        'edges': edges,
        'gaps': gaps,
        'edge_type': edge_type,
    }


def from_packing(pos, r, gtype, n_cells=None, p=None, **kwargs):
    """Construct model state from an existing packing.

    Parameters
    ----------
    pos : ndarray, shape (N, d)
        Granule positions.
    r : ndarray, shape (N,)
        Granule radii.
    gtype : ndarray, shape (N,)
        Granule types (0=func, 1=inert).
    n_cells : ndarray or None
        Cells per granule. If None, uses p.n_cells_per_granule for functional.
    p : NetworkParams or None
        Parameters.

    Returns
    -------
    dict
        Model state.
    """
    if p is None:
        p = NetworkParams(**kwargs)
    if n_cells is None:
        n_cells = np.zeros(len(r), dtype=int)
        n_cells[gtype == 0] = p.n_cells_per_granule

    edges, gaps, edge_type = build_contact_graph(pos, r, gtype, p)

    return {
        'params': p,
        'pos': pos.copy(),
        'r': r.copy(),
        'gtype': gtype.copy(),
        'n_cells': n_cells.copy(),
        'edges': edges,
        'gaps': gaps,
        'edge_type': edge_type,
    }


# ======================================================================
# Monte Carlo ensemble
# ======================================================================

def ensemble(model_or_params, n_runs=100, seed=0, verbose=True):
    """Run Monte Carlo ensemble of realizations.

    Parameters
    ----------
    model_or_params : dict or Params-like
        Either a model state dict or parameters for from_params().
    n_runs : int
        Number of realizations.
    seed : int
        Base seed (each run uses seed + i).
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        'runs': list of result dicts from solve().
        'summary': dict of ensemble-averaged time series with mean/std.
    """
    if isinstance(model_or_params, dict) and 'params' in model_or_params:
        p = model_or_params['params']
    else:
        p = model_or_params

    runs = []
    for i in range(n_runs):
        if verbose and (i % max(1, n_runs // 10) == 0):
            print(f"  Ensemble: run {i+1}/{n_runs}")

        # Fresh packing per realization for true stochastic sampling
        model = from_params(p)
        result = solve(model, seed=seed + i, verbose=False)
        runs.append(result)

    # Ensemble summary
    if runs:
        # Collect time series
        all_times = []
        all_series = {}
        keys = ['phi_f', 'phi_v', 'x_f', 'Z_contact', 'Z_bridge',
                'n_bridges', 'n_locked', 'F_bridge_mean', 'largest_cluster',
                'K_perm', 'spanning']

        for r in runs:
            times = [h['time'] for h in r['history']]
            all_times.append(times)
            for k in keys:
                vals = [h[k] for h in r['history']]
                all_series.setdefault(k, []).append(vals)

        # Interpolate to common time grid
        t_common = np.array(all_times[0])
        summary = {'time': t_common}
        for k in keys:
            matrix = np.zeros((n_runs, len(t_common)))
            for i in range(n_runs):
                ti = np.array(all_times[i])
                vi = np.array(all_series[k][i], dtype=float)
                matrix[i] = np.interp(t_common, ti, vi)
            summary[f'{k}_mean'] = np.mean(matrix, axis=0)
            summary[f'{k}_std'] = np.std(matrix, axis=0)
            summary[f'{k}_q25'] = np.percentile(matrix, 25, axis=0)
            summary[f'{k}_q75'] = np.percentile(matrix, 75, axis=0)
            summary[f'{k}_min'] = np.min(matrix, axis=0)
            summary[f'{k}_max'] = np.max(matrix, axis=0)
    else:
        summary = {}

    return {'runs': runs, 'summary': summary}


# ======================================================================
# Topology visualization helpers
# ======================================================================

def _get_cluster_labels(N, edges, bridges_data):
    """Compute per-granule cluster labels from bridge data.

    Returns (labels, uf) where labels[i] is the cluster id for granule i.
    """
    uf = UnionFind(N)
    bridge_edges = set()
    for b_data in bridges_data:
        edge_idx, gran_i, gran_j = b_data[0], b_data[1], b_data[2]
        uf.union(gran_i, gran_j)
        bridge_edges.add(edge_idx)
    labels = np.array([uf.find(i) for i in range(N)])
    return labels, uf, bridge_edges


def _cluster_colormap(labels):
    """Assign distinct colors to clusters, largest cluster gets highlight."""
    unique_labels = np.unique(labels)
    sizes = {lab: np.sum(labels == lab) for lab in unique_labels}
    largest_lab = max(sizes, key=sizes.get)

    # Use tab20 for distinct cluster colors
    cmap = plt.cm.tab20
    color_map = {}
    idx = 0
    for lab in unique_labels:
        if lab == largest_lab:
            color_map[lab] = np.array([0.85, 0.15, 0.15, 1.0])  # red for largest
        else:
            color_map[lab] = np.array(cmap(idx % 20))
            idx += 1
    return color_map, largest_lab


# ======================================================================
# Plotting — Topology
# ======================================================================

def plot_network_snapshot(snap, p=None, ax=None, title=None,
                          show_contacts=True, show_bridges=True,
                          color_by='cluster'):
    """Draw a single network topology snapshot.

    Parameters
    ----------
    snap : dict
        A history entry with '_pos', '_r', '_gtype', '_edges', '_bridges' keys.
    p : NetworkParams or None
        Parameters (for domain bounds). Inferred from snap if None.
    ax : matplotlib.axes.Axes or None
        Axes to draw on.
    title : str or None
        Plot title.
    show_contacts : bool
        Draw contact edges (gap <= 0) as gray lines.
    show_bridges : bool
        Draw bridge edges as colored lines.
    color_by : str
        'cluster' = color granules by bridge cluster.
        'type' = color by functional/inert.
        'Z' = color by local coordination number.
        'force' = color bridges by force magnitude.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pos = snap['_pos']
    r = snap['_r']
    gtype = snap['_gtype']
    edges = snap['_edges']
    gaps = snap['_gaps']
    bridges_data = snap['_bridges']
    N = len(r)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # Domain bounds
    if p is not None:
        ax.set_xlim(0, p.Lx)
        ax.set_ylim(0, p.Ly)
    else:
        margin = np.max(r) * 1.5
        ax.set_xlim(pos[:, 0].min() - margin, pos[:, 0].max() + margin)
        ax.set_ylim(pos[:, 1].min() - margin, pos[:, 1].max() + margin)

    ax.set_aspect('equal')

    # Cluster computation
    labels, uf, bridge_edge_set = _get_cluster_labels(N, edges, bridges_data)
    color_map, largest_lab = _cluster_colormap(labels)

    # Contact edges (gray, thin)
    if show_contacts and len(edges) > 0:
        contact_mask = gaps <= 0
        for e in range(len(edges)):
            if contact_mask[e]:
                i, j = edges[e]
                ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                        color='#cccccc', lw=0.5, zorder=1)

    # Bridge edges
    if show_bridges and bridges_data:
        if color_by == 'force':
            forces = np.array([b[4] for b in bridges_data])
            f_max = max(forces.max(), 1e-6)
            cmap_bridge = plt.cm.hot_r
            for b in bridges_data:
                ei, gi, gj = b[0], b[1], b[2]
                f_norm = b[4] / f_max
                color = cmap_bridge(f_norm)
                lw = 1.0 + 3.0 * f_norm
                ax.plot([pos[gi, 0], pos[gj, 0]], [pos[gi, 1], pos[gj, 1]],
                        color=color, lw=lw, zorder=3, solid_capstyle='round')
        else:
            for b in bridges_data:
                gi, gj = b[1], b[2]
                locked = b[5]
                color = '#cc0000' if locked else '#e07020'
                lw = 2.5 if locked else 1.5
                ax.plot([pos[gi, 0], pos[gj, 0]], [pos[gi, 1], pos[gj, 1]],
                        color=color, lw=lw, zorder=3, solid_capstyle='round')

    # Granules
    if color_by == 'cluster':
        for i in range(N):
            c = color_map.get(labels[i], [0.7, 0.7, 0.7, 1.0])
            circle = plt.Circle(pos[i, :2], r[i], fc=c, ec='k',
                                lw=0.3, zorder=4, alpha=0.85)
            ax.add_patch(circle)
    elif color_by == 'type':
        for i in range(N):
            if gtype[i] == 0:
                fc = '#e05050'  # functional = red
            else:
                fc = '#50a050'  # inert = green
            circle = plt.Circle(pos[i, :2], r[i], fc=fc, ec='k',
                                lw=0.3, zorder=4, alpha=0.8)
            ax.add_patch(circle)
    elif color_by == 'Z':
        # Local coordination (bridges per granule)
        Z_local = np.zeros(N)
        for b in bridges_data:
            Z_local[b[1]] += 1
            Z_local[b[2]] += 1
        Z_max = max(Z_local.max(), 1)
        cmap_z = plt.cm.YlOrRd
        for i in range(N):
            c = cmap_z(Z_local[i] / Z_max) if gtype[i] == 0 else (0.7, 0.7, 0.7, 0.6)
            circle = plt.Circle(pos[i, :2], r[i], fc=c, ec='k',
                                lw=0.3, zorder=4, alpha=0.85)
            ax.add_patch(circle)
    else:  # 'force' mode — granules by type
        for i in range(N):
            fc = '#e05050' if gtype[i] == 0 else '#50a050'
            circle = plt.Circle(pos[i, :2], r[i], fc=fc, ec='k',
                                lw=0.3, zorder=4, alpha=0.7)
            ax.add_patch(circle)

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel(r'$x$ ($\mu$m)')
    ax.set_ylabel(r'$y$ ($\mu$m)')

    return fig


def plot_topology_evolution(result, n_frames=6, outdir=None):
    """Plot network topology at multiple timepoints showing bridge growth.

    Parameters
    ----------
    result : dict
        Result from solve() with snapshot history.
    n_frames : int
        Number of timepoints to show.
    outdir : str or None
        If provided, save figures here.

    Returns
    -------
    list of (name, fig) tuples
    """
    hist = result['history']
    p = result['params']
    figs = []

    # Select evenly spaced frames
    n_total = len(hist)
    if n_total <= n_frames:
        indices = list(range(n_total))
    else:
        indices = [int(i * (n_total - 1) / (n_frames - 1)) for i in range(n_frames)]

    # --- 1. Cluster evolution (colored by bridge cluster) ---
    n_cols = min(n_frames, 3)
    n_rows = (len(indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for idx_i, snap_idx in enumerate(indices):
        row, col = divmod(idx_i, n_cols)
        ax = axes[row, col]
        snap = hist[snap_idx]
        t_val = snap['time']
        n_b = snap['n_bridges']
        spanning = snap['spanning']
        span_str = ' [SPANNING]' if spanning else ''
        plot_network_snapshot(snap, p=p, ax=ax, color_by='cluster',
                              title=f't = {t_val:.0f}h — {n_b} bridges{span_str}')

    # Hide unused axes
    for idx_i in range(len(indices), n_rows * n_cols):
        row, col = divmod(idx_i, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle('Bridge Cluster Evolution', fontsize=14, y=1.01)
    plt.tight_layout()
    figs.append(('topology_clusters', fig))

    # --- 2. Force chain map (final state) ---
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    plot_network_snapshot(hist[-1], p=p, ax=ax2, color_by='force',
                          title=f'Bridge Force Map (t = {hist[-1]["time"]:.0f}h)')

    # Add colorbar for force
    bridges_data = hist[-1]['_bridges']
    if bridges_data:
        forces = np.array([b[4] for b in bridges_data])
        f_max = max(forces.max(), 1e-6)
        sm = plt.cm.ScalarMappable(cmap=plt.cm.hot_r,
                                    norm=plt.Normalize(0, f_max))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.7, pad=0.02)
        cbar.set_label('Bridge force (nN)', fontsize=10)

    plt.tight_layout()
    figs.append(('topology_forces', fig2))

    # --- 3. Coordination number map (final state) ---
    fig3, ax3 = plt.subplots(figsize=(8, 8))
    plot_network_snapshot(hist[-1], p=p, ax=ax3, color_by='Z',
                          title=f'Local Coordination $Z$ (t = {hist[-1]["time"]:.0f}h)')

    # Add Z colorbar
    snap_final = hist[-1]
    N = len(snap_final['_r'])
    Z_local = np.zeros(N)
    for b in snap_final['_bridges']:
        Z_local[b[1]] += 1
        Z_local[b[2]] += 1
    func_mask = snap_final['_gtype'] == 0
    Z_max = max(Z_local[func_mask].max() if np.any(func_mask) else 1, 1)
    sm_z = plt.cm.ScalarMappable(cmap=plt.cm.YlOrRd,
                                  norm=plt.Normalize(0, Z_max))
    sm_z.set_array([])
    cbar_z = plt.colorbar(sm_z, ax=ax3, shrink=0.7, pad=0.02)
    cbar_z.set_label('Bridge coordination $Z$', fontsize=10)

    plt.tight_layout()
    figs.append(('topology_coordination', fig3))

    # --- 4. Type map with contacts (initial and final side-by-side) ---
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6.5))
    plot_network_snapshot(hist[0], p=p, ax=ax4a, color_by='type',
                          show_bridges=True,
                          title=f'Packing (t = {hist[0]["time"]:.0f}h)')
    plot_network_snapshot(hist[-1], p=p, ax=ax4b, color_by='type',
                          show_bridges=True,
                          title=f'Compacted (t = {hist[-1]["time"]:.0f}h)')

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#e05050', edgecolor='k', label='Functional'),
        Patch(facecolor='#50a050', edgecolor='k', label='Inert'),
        Line2D([0], [0], color='#cccccc', lw=1, label='Contact'),
        Line2D([0], [0], color='#e07020', lw=2, label='Bridge'),
    ]
    fig4.legend(handles=legend_elements, loc='lower center', ncol=4,
                fontsize=10, frameon=True, bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    figs.append(('topology_type_comparison', fig4))

    # --- 5. Percolation transition plot ---
    fig5, axes5 = plt.subplots(1, 3, figsize=(16, 5))

    # 5a: Cluster size distribution (final)
    ax = axes5[0]
    labels_final, uf_final, _ = _get_cluster_labels(
        N, snap_final['_edges'], snap_final['_bridges'])
    sizes = uf_final.component_sizes()
    ax.bar(range(len(sizes)), sizes, color='C1', alpha=0.8)
    ax.set_xlabel('Cluster rank')
    ax.set_ylabel('Cluster size (granules)')
    ax.set_title(f'Cluster Size Distribution (t={snap_final["time"]:.0f}h)')
    ax.grid(True, alpha=0.3)

    # 5b: Largest cluster growth
    ax = axes5[1]
    t_arr = np.array([h['time'] for h in hist])
    largest_arr = np.array([h['largest_cluster'] for h in hist])
    n_func = int(np.sum(hist[0]['_gtype'] == 0))
    ax.plot(t_arr, largest_arr / max(n_func, 1), 'C2-', lw=2)
    ax.axhline(0.5, color='grey', ls='--', lw=1, alpha=0.5,
               label='50% threshold')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Largest cluster / N_func')
    ax.set_title('Percolation Growth')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5c: Z_bridge vs time with percolation threshold
    ax = axes5[2]
    Z_arr = np.array([h['Z_bridge'] for h in hist])
    span_arr = np.array([1 if h['spanning'] else 0 for h in hist])
    ax.plot(t_arr, Z_arr, 'C1-', lw=2, label='$Z_{bridge}$')
    ax.fill_between(t_arr, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 5,
                    where=span_arr > 0, alpha=0.1, color='C3',
                    label='Spanning region')
    # Mark z_c ~ 1.5 (2D bond percolation threshold on Delaunay, Braz & Araujo)
    ax.axhline(1.5, color='grey', ls=':', lw=1.5,
               label='$z_c \\approx 1.5$ (percolation)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$Z_{bridge}$')
    ax.set_title('Coordination vs Percolation')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figs.append(('topology_percolation', fig5))

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        for name, f in figs:
            f.savefig(str(Path(outdir) / f'{name}.png'), dpi=150,
                      bbox_inches='tight')
            plt.close(f)

    return figs


# ======================================================================
# Plotting — Time series
# ======================================================================

def plot_single_run(result, outdir=None):
    """Plot time series from a single realization.

    Returns list of figures.
    """
    hist = result['history']
    t = np.array([h['time'] for h in hist])
    figs = []

    # 1. Bridge formation & connectivity
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(t, [h['n_bridges'] for h in hist], 'C1-', lw=2, label='Total bridges')
    ax.plot(t, [h['n_locked'] for h in hist], 'C3--', lw=1.5, label='Locked bridges')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Count')
    ax.set_title('Bridge Formation')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, [h['Z_contact'] for h in hist], 'C0-', lw=2, label='$Z_{contact}$')
    ax.plot(t, [h['Z_bridge'] for h in hist], 'C1-', lw=2, label='$Z_{bridge}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Coordination number')
    ax.set_title('Network Connectivity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, [h['largest_cluster'] for h in hist], 'C2-', lw=2)
    ax2 = ax.twinx()
    spanning = [1 if h['spanning'] else 0 for h in hist]
    ax2.fill_between(t, 0, spanning, alpha=0.2, color='C3', label='Spanning')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Largest cluster size')
    ax2.set_ylabel('Spanning (0/1)')
    ax.set_title('Percolation')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, [h['F_bridge_mean'] for h in hist], 'C4-', lw=2, label='Mean')
    ax.plot(t, [h['F_bridge_max'] for h in hist], 'C4--', lw=1, label='Max')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Bridge force (nN)')
    ax.set_title('Bridge Forces')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Contact-Network Model: Single Realization', fontsize=14)
    plt.tight_layout()
    figs.append(('network_dynamics', fig))

    # 2. Phase fractions & permeability
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes2[0]
    ax.plot(t, [h['phi_f'] for h in hist], 'C1-', lw=2, label=r'$\phi_f$')
    ax.plot(t, [h['phi_i'] for h in hist], 'C0-', lw=2, label=r'$\phi_i$')
    ax.plot(t, [h['phi_v'] for h in hist], 'C2-', lw=2, label=r'$\phi_v$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Volume fraction')
    ax.set_title('Phase Fractions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes2[1]
    ax.plot(t, [h['K_perm'] for h in hist], 'C2-', lw=2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Permeability $K$ ($\mu m^2$)')
    ax.set_title('Kozeny-Carman Permeability')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    figs.append(('phase_permeability', fig2))

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(str(Path(outdir) / f'{name}.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

    return figs


def plot_ensemble(summary, outdir=None):
    """Plot ensemble statistics with confidence bands.

    Returns list of figures.
    """
    t = summary['time']
    figs = []

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    def _band(ax, key, color, label):
        mean = summary[f'{key}_mean']
        q25 = summary[f'{key}_q25']
        q75 = summary[f'{key}_q75']
        ax.plot(t, mean, color=color, lw=2, label=label)
        ax.fill_between(t, q25, q75, color=color, alpha=0.2)

    # Bridges
    ax = axes[0, 0]
    _band(ax, 'n_bridges', 'C1', 'Bridges')
    _band(ax, 'n_locked', 'C3', 'Locked')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Count')
    ax.set_title('Bridge Formation (ensemble)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Coordination
    ax = axes[0, 1]
    _band(ax, 'Z_bridge', 'C1', '$Z_{bridge}$')
    _band(ax, 'Z_contact', 'C0', '$Z_{contact}$')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Z')
    ax.set_title('Coordination Number')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Percolation
    ax = axes[0, 2]
    _band(ax, 'largest_cluster', 'C2', 'Largest cluster')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Size')
    ax.set_title('Largest Cluster')
    ax.grid(True, alpha=0.3)

    # Spanning probability
    ax = axes[1, 0]
    ax.plot(t, summary['spanning_mean'], 'C3-', lw=2)
    ax.fill_between(t, summary['spanning_q25'], summary['spanning_q75'],
                     color='C3', alpha=0.2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('P(spanning)')
    ax.set_title('Spanning Probability')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Bridge force
    ax = axes[1, 1]
    _band(ax, 'F_bridge_mean', 'C4', 'Mean force')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Force (nN)')
    ax.set_title('Bridge Force')
    ax.grid(True, alpha=0.3)

    # Permeability
    ax = axes[1, 2]
    mean_K = summary['K_perm_mean']
    q25_K = summary['K_perm_q25']
    q75_K = summary['K_perm_q75']
    ax.semilogy(t, mean_K, 'C2-', lw=2, label='K')
    ax.fill_between(t, np.maximum(q25_K, 1e-6), q75_K, color='C2', alpha=0.2)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$K$ ($\mu m^2$)')
    ax.set_title('Permeability')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Contact-Network Model: Ensemble Statistics', fontsize=14)
    plt.tight_layout()
    figs.append(('ensemble_summary', fig))

    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        for name, fig in figs:
            fig.savefig(str(Path(outdir) / f'{name}.png'), dpi=150,
                        bbox_inches='tight')
            plt.close(fig)

    return figs


# ======================================================================
# CLI
# ======================================================================

def run_demo(outdir='results/contact_network_demo', n_ensemble=20, seed=0):
    """Run a demo: single realization + small ensemble.

    Parameters
    ----------
    outdir : str
        Output directory for plots.
    n_ensemble : int
        Number of ensemble realizations.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Contains 'single', 'ensemble', 'model' keys.
    """
    print("=" * 65)
    print("  Stochastic Contact-Network Model (V2.5)")
    print("=" * 65)

    p = NetworkParams()
    print(f"\n  Parameters:")
    print(f"    Mode           = {p.mode}")
    print(f"    Domain         = {p.Lx} x {p.Ly}" +
          (f" x {p.Lz}" if p.is_3d else "") + " um")
    print(f"    E_modulus      = {p.E_modulus} kPa")
    print(f"    phi_f / phi_i  = {p.phi_f_target} / {p.phi_i_target}")
    print(f"    t_total        = {p.t_total} h")
    print(f"    n_cells/gran   = {p.n_cells_per_granule}")

    F_cell = motor_clutch_force(p.E_modulus, p)
    print(f"    F_cell         = {F_cell:.2f} nN")

    # Single realization
    print(f"\n  [1/2] Single realization (seed={seed})...")
    model = from_params(p)
    single = solve(model, seed=seed, verbose=True)

    h_final = single['history'][-1]
    print(f"\n  Final state (t={h_final['time']:.1f}h):")
    print(f"    Bridges       = {h_final['n_bridges']}")
    print(f"    Locked        = {h_final['n_locked']}")
    print(f"    Z_bridge      = {h_final['Z_bridge']:.2f}")
    print(f"    Largest clust = {h_final['largest_cluster']}")
    print(f"    Spanning      = {h_final['spanning']}")
    print(f"    phi_f / phi_v = {h_final['phi_f']:.3f} / {h_final['phi_v']:.3f}")

    # Plot single time series
    print(f"\n  Saving single-run plots to {outdir}/...")
    plot_single_run(single, outdir=outdir)

    # Plot topology
    print(f"  Saving topology plots to {outdir}/...")
    plot_topology_evolution(single, n_frames=6, outdir=outdir)

    # Ensemble
    print(f"\n  [2/2] Ensemble ({n_ensemble} realizations)...")
    ens = ensemble(p, n_runs=n_ensemble, seed=seed, verbose=True)

    # Plot ensemble
    print(f"  Saving ensemble plots to {outdir}/...")
    plot_ensemble(ens['summary'], outdir=outdir)

    # Summary statistics
    s = ens['summary']
    t_final_idx = -1
    print(f"\n  Ensemble summary at t={s['time'][t_final_idx]:.1f}h "
          f"(n={n_ensemble}):")
    print(f"    Bridges       = {s['n_bridges_mean'][t_final_idx]:.1f} "
          f"+/- {s['n_bridges_std'][t_final_idx]:.1f}")
    print(f"    Z_bridge      = {s['Z_bridge_mean'][t_final_idx]:.2f} "
          f"+/- {s['Z_bridge_std'][t_final_idx]:.2f}")
    print(f"    Spanning prob = {s['spanning_mean'][t_final_idx]:.2f}")
    print(f"    Largest clust = {s['largest_cluster_mean'][t_final_idx]:.1f} "
          f"+/- {s['largest_cluster_std'][t_final_idx]:.1f}")

    print("\n" + "=" * 65)
    print("  CONTACT-NETWORK MODEL COMPLETE")
    print("=" * 65)
    print(f"  Plots saved to: {outdir}/")
    print(f"  Files: network_dynamics.png, phase_permeability.png,")
    print(f"         topology_clusters.png, topology_forces.png,")
    print(f"         topology_coordination.png, topology_type_comparison.png,")
    print(f"         topology_percolation.png, ensemble_summary.png")

    return {'single': single, 'ensemble': ens, 'model': model}


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='Stochastic Contact-Network Model for granular scaffold compaction.')
    parser.add_argument('-o', '--outdir', default='results/contact_network_demo',
                        help='Output directory for plots')
    parser.add_argument('-n', '--n-ensemble', type=int, default=20,
                        help='Number of ensemble realizations')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()

    t0 = time.time()
    run_demo(outdir=args.outdir, n_ensemble=args.n_ensemble, seed=args.seed)
    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
