"""
Contact Network Model for Granular Scaffold Compaction
=======================================================
V2.5 — Graph-based mesoscale model that tracks packing topology,
bridge percolation, force chains, and coordination evolution.

Sits between the full DEM (expensive, per-particle) and the mean-field
ODE (cheap, no spatial or topological info).  Captures discrete topology
that continuous models miss: bridge percolation thresholds, force chain
statistics, cluster size distributions, coordination number evolution.

Physics:
    - Spheres only (no superellipsoid solver — the key simplification)
    - Hertz contact repulsion + JKR adhesion (optional)
    - Motor-clutch cell traction via bridges
    - Overdamped position update with velocity cap
    - Bridge state machine: NONE → FORMING → ACTIVE → LOCKED/SENESCENT
    - Aggregate cell count per granule (no individual cell tracking)
    - cKDTree neighbor search with periodic/wall BCs

Can be initialised from:
    (A) DEM snapshot  — ContactNetwork.from_snapshot(snap, p)
    (B) Random packing — ContactNetwork.from_packing(p, seed)
    (C) Params only    — ContactNetwork.from_params(p, seed)

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).

Usage:
    python analysis/contact_network.py -i results/default
    python analysis/contact_network.py --standalone --seed 42

Or import programmatically:
    from analysis.contact_network import ContactNetwork
    net = ContactNetwork.from_params(Params(), seed=42)
    result = net.solve(t_span=(0, 72), dt=0.5)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Dict, Tuple, List
from pathlib import Path


# ======================================================================
# Bridge state machine
# ======================================================================

class BridgeState(IntEnum):
    NONE = 0
    FORMING = 1
    ACTIVE = 2
    LOCKED = 3
    SENESCENT = 4
    BROKEN = 5


# ======================================================================
# Contact Network
# ======================================================================

class ContactNetwork:
    """Graph-based contact network model for granular scaffold.

    Nodes = granules, edges = contacts and near-neighbor pairs.
    Evolves bridge states, contact forces, and granule positions
    via overdamped dynamics on the graph.

    Attributes
    ----------
    N : int
        Number of granules.
    pos : ndarray (N, D)
        Granule positions (D=2 or 3).
    radii : ndarray (N,)
        Granule radii.
    gtype : ndarray (N,) int
        Granule type (0=functional, 1=inert).
    n_cells : ndarray (N,) int
        Number of cells per granule.
    fa_maturity : ndarray (N,)
        Focal adhesion maturity per granule [0, 1].
    """

    def __init__(self, positions, radii, gtypes, n_cells_per_node, p,
                 is_3d=None):
        """Initialise from arrays.

        Parameters
        ----------
        positions : ndarray (N, D)
            Granule centre positions (D=2 or 3).
        radii : ndarray (N,)
            Granule radii in µm.
        gtypes : ndarray (N,) int
            0 = functional, 1 = inert.
        n_cells_per_node : ndarray (N,) int
            Number of cells on each granule.
        p : Params-like
            Simulation parameters (or dict with matching keys).
        is_3d : bool or None
            If None, inferred from positions shape.
        """
        self.pos = np.asarray(positions, dtype=np.float64)
        self.N = len(self.pos)
        self.D = self.pos.shape[1] if self.pos.ndim == 2 else 2
        if is_3d is None:
            is_3d = (self.D == 3)
        self.is_3d = is_3d
        self.radii = np.asarray(radii, dtype=np.float64)
        self.gtype = np.asarray(gtypes, dtype=np.int32)
        self.n_cells = np.asarray(n_cells_per_node, dtype=np.int32)
        self.p = p

        # Per-node state
        self.fa_maturity = np.zeros(self.N)
        self.vel = np.zeros_like(self.pos)

        # Precompute parameters
        self._E_star = self._compute_E_star()
        self._F_cell_max = self._compute_F_cell()

        # Edge storage (sparse — rebuilt each step)
        self._edge_i = np.array([], dtype=np.int32)
        self._edge_j = np.array([], dtype=np.int32)
        self._edge_dist = np.array([], dtype=np.float64)
        self._edge_overlap = np.array([], dtype=np.float64)

        # Bridge storage: dict keyed by (min(i,j), max(i,j))
        # Values: {'state': BridgeState, 'age': float, 'force': float,
        #          'ramp': float, 'alignment': float}
        self._bridges: Dict[Tuple[int, int], dict] = {}

        # History
        self._history: List[dict] = []

        # Build initial edges
        self._rebuild_edges()

    # ------------------------------------------------------------------
    # Precomputed physics
    # ------------------------------------------------------------------

    def _get(self, key, default=None):
        """Get parameter value from Params object or dict."""
        if isinstance(self.p, dict):
            return self.p.get(key, default)
        return getattr(self.p, key, default)

    def _compute_E_star(self):
        """Hertzian reduced modulus (nN/µm²)."""
        E = float(self._get('E_modulus', 10.0))
        nu = float(self._get('poisson_ratio', 0.45))
        # E* = E / (2(1-ν²))  [kPa → Pa for Hertz, then ×1e-3 for nN/µm²]
        return E * 1e3 / (2.0 * (1.0 - nu ** 2))

    def _compute_F_cell(self):
        """Motor-clutch steady-state traction per cell (nN) at full maturity."""
        E = float(self._get('E_modulus', 10.0))
        n_motors = int(self._get('n_motors', 50))
        F_stall_per = float(self._get('F_motor_stall', 0.5))
        n_clutches = int(self._get('n_clutches', 75))
        k_clutch = float(self._get('k_clutch', 5.0))
        k_on = float(self._get('k_on_clutch', 1.0))
        k_off = float(self._get('k_off_clutch', 0.1))
        nu = float(self._get('poisson_ratio', 0.45))
        cell_d = float(self._get('cell_diameter', 20.0))
        F_max = float(self._get('F_max_per_cell', 50.0))

        F_stall = n_motors * F_stall_per
        k_opt = n_clutches * k_clutch
        engagement = k_on / (k_on + k_off)
        a_cell = cell_d / 2.0
        k_sub = np.pi * E * a_cell / (1.0 - nu ** 2)
        beta = k_sub / (k_sub + k_opt)
        return min(F_stall * beta * engagement, F_max)

    # ------------------------------------------------------------------
    # Edge / neighbor computation
    # ------------------------------------------------------------------

    def _rebuild_edges(self):
        """Rebuild edge list from cKDTree neighbor search."""
        sense_dist = float(self._get('cell_sense_distance', 40.0))
        break_gap = float(self._get('bridge_break_gap', 60.0))
        max_r = float(np.max(self.radii)) if self.N > 0 else 0.0
        cutoff = 2.0 * max_r + max(sense_dist, break_gap)

        boundary = self._get('boundary_mode', 'walls')
        if boundary == 'periodic':
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            # Wrap positions into [0, L) for periodic cKDTree
            self.pos[:, 0] = self.pos[:, 0] % Lx
            self.pos[:, 1] = self.pos[:, 1] % Ly
            if self.is_3d:
                Lz = float(self._get('Lz', 800.0))
                self.pos[:, 2] = self.pos[:, 2] % Lz
                boxsize = [Lx, Ly, Lz]
            else:
                boxsize = [Lx, Ly]
            tree = cKDTree(self.pos, boxsize=boxsize)
        else:
            tree = cKDTree(self.pos)

        if self.N == 0:
            self._edge_i = np.array([], dtype=np.int32)
            self._edge_j = np.array([], dtype=np.int32)
            self._edge_dist = np.array([], dtype=np.float64)
            self._edge_overlap = np.array([], dtype=np.float64)
            return

        pairs = tree.query_pairs(cutoff, output_type='ndarray')
        if len(pairs) == 0:
            self._edge_i = np.array([], dtype=np.int32)
            self._edge_j = np.array([], dtype=np.int32)
            self._edge_dist = np.array([], dtype=np.float64)
            self._edge_overlap = np.array([], dtype=np.float64)
            return

        ei = pairs[:, 0].astype(np.int32)
        ej = pairs[:, 1].astype(np.int32)

        # Compute distances (minimum image for periodic)
        dp = self.pos[ej] - self.pos[ei]
        if boundary == 'periodic':
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            dp[:, 0] -= Lx * np.round(dp[:, 0] / Lx)
            dp[:, 1] -= Ly * np.round(dp[:, 1] / Ly)
            if self.is_3d:
                Lz = float(self._get('Lz', 800.0))
                dp[:, 2] -= Lz * np.round(dp[:, 2] / Lz)

        dist = np.sqrt(np.sum(dp ** 2, axis=1))
        overlap = self.radii[ei] + self.radii[ej] - dist

        self._edge_i = ei
        self._edge_j = ej
        self._edge_dist = dist
        self._edge_overlap = overlap
        self._edge_dp = dp  # displacement vectors (j - i)

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def _hertz_force(self, overlap, R_eff):
        """Hertzian contact force (nN). Vectorised."""
        delta = np.maximum(overlap, 0.0)
        return (4.0 / 3.0) * self._E_star * np.sqrt(R_eff) * delta ** 1.5 * 1e-3

    @staticmethod
    def _max_overlap(ri, rj, frac=0.3):
        """Maximum allowed overlap = frac × min(Ri, Rj)."""
        return frac * np.minimum(ri, rj)

    def _compute_forces(self, t):
        """Compute net force on each granule from contacts + bridges.

        Returns (N, D) force array in nN.
        """
        F = np.zeros_like(self.pos)
        n_edges = len(self._edge_i)
        if n_edges == 0:
            return F

        ei = self._edge_i
        ej = self._edge_j
        dist = self._edge_dist
        overlap = self._edge_overlap
        dp = self._edge_dp

        # Unit normals (i → j)
        safe_dist = np.maximum(dist, 1e-6)
        normals = dp / safe_dist[:, None]

        # Effective radii
        R_eff = self.radii[ei] * self.radii[ej] / (self.radii[ei] + self.radii[ej])

        # --- Contact forces (Hertz repulsion + anti-encapsulation) ---
        contact_mask = overlap > 0
        F_contact = np.zeros(n_edges)
        if np.any(contact_mask):
            F_contact[contact_mask] = self._hertz_force(
                overlap[contact_mask], R_eff[contact_mask])

        # Anti-encapsulation penalty: stiff linear barrier when overlap
        # exceeds 30% of the smaller radius.  Penalty stiffness is 10×
        # the Hertz force at the threshold, making deep penetration
        # energetically prohibitive.
        delta_max = self._max_overlap(self.radii[ei], self.radii[ej], frac=0.3)
        over_limit = contact_mask & (overlap > delta_max)
        if np.any(over_limit):
            excess = overlap[over_limit] - delta_max[over_limit]
            # Penalty stiffness: scale of Hertz at the cap
            F_hertz_cap = self._hertz_force(delta_max[over_limit],
                                            R_eff[over_limit])
            k_penalty = 10.0 * F_hertz_cap / np.maximum(delta_max[over_limit], 1.0)
            F_contact[over_limit] += k_penalty * excess

        # JKR adhesion (optional, simplified)
        W_ff = float(self._get('W_adh_ff', 0.002))
        W_fi = float(self._get('W_adh_if', 0.001))
        W_ii = float(self._get('W_adh_ii', 0.0005))

        gi = self.gtype[ei]
        gj = self.gtype[ej]
        W = np.where((gi == 0) & (gj == 0), W_ff,
                np.where((gi == 1) & (gj == 1), W_ii, W_fi))
        F_adh = 2.0 * np.pi * W * R_eff * 1e3  # DMT adhesion (nN)

        # Suppress adhesion when overlap already exceeds the cap
        F_adh_safe = np.where(overlap > delta_max, 0.0, F_adh)
        F_normal = F_contact - np.where(contact_mask, F_adh_safe, 0.0)

        # --- Bridge forces (suppressed when deeply overlapping) ---
        F_bridge = np.zeros(n_edges)
        for k in range(n_edges):
            key = (min(int(ei[k]), int(ej[k])), max(int(ei[k]), int(ej[k])))
            br = self._bridges.get(key)
            if br is None or br['state'] in (BridgeState.NONE,
                                              BridgeState.BROKEN,
                                              BridgeState.SENESCENT):
                continue

            # Bridge attractive force
            ramp = br['ramp']
            alignment = br['alignment']
            nc_i = self.n_cells[ei[k]] if self.gtype[ei[k]] == 0 else 0
            nc_j = self.n_cells[ej[k]] if self.gtype[ej[k]] == 0 else 0
            host_cells = max(nc_i, nc_j)
            mat_i = self.fa_maturity[ei[k]]
            mat_j = self.fa_maturity[ej[k]]
            maturity = max(mat_i, mat_j)
            F_br = self._F_cell_max * ramp * alignment * maturity

            # Suppress bridge attraction when overlap exceeds limit
            if overlap[k] > delta_max[k]:
                F_br = 0.0

            F_bridge[k] = F_br
            br['force'] = F_br

        # Accumulate: contact pushes apart (along normal), bridge pulls together
        F_edge = (F_normal - F_bridge)  # positive = repulsive along i→j

        # Scatter to nodes
        F_vec = F_edge[:, None] * normals
        for k in range(n_edges):
            F[ei[k]] += F_vec[k]
            F[ej[k]] -= F_vec[k]

        # --- Wall forces (if not periodic) ---
        if self._get('boundary_mode', 'walls') == 'walls':
            F = self._add_wall_forces(F)

        return F

    def _add_wall_forces(self, F):
        """Add Hertz wall repulsion."""
        Lx = float(self._get('Lx', 800.0))
        Ly = float(self._get('Ly', 800.0))
        r = self.radii

        # x-walls
        overlap_lo_x = r - self.pos[:, 0]
        overlap_hi_x = r - (Lx - self.pos[:, 0])
        # y-walls
        overlap_lo_y = r - self.pos[:, 1]
        overlap_hi_y = r - (Ly - self.pos[:, 1])

        wall_R_eff = r  # wall = infinite radius → R_eff = R
        for overlap, axis, sign in [
            (overlap_lo_x, 0, +1), (overlap_hi_x, 0, -1),
            (overlap_lo_y, 1, +1), (overlap_hi_y, 1, -1),
        ]:
            mask = overlap > 0
            if np.any(mask):
                Fw = self._hertz_force(overlap[mask], wall_R_eff[mask])
                F[mask, axis] += sign * Fw

        if self.is_3d:
            Lz = float(self._get('Lz', 800.0))
            overlap_lo_z = r - self.pos[:, 2]
            overlap_hi_z = r - (Lz - self.pos[:, 2])
            for overlap, sign in [(overlap_lo_z, +1), (overlap_hi_z, -1)]:
                mask = overlap > 0
                if np.any(mask):
                    Fw = self._hertz_force(overlap[mask], wall_R_eff[mask])
                    F[mask, 2] += sign * Fw

        return F

    # ------------------------------------------------------------------
    # Bridge state evolution
    # ------------------------------------------------------------------

    def _update_bridges(self, dt, t, rng):
        """Evolve bridge states and attempt new bridge formation.

        Bridge state machine per FF edge:
            NONE → FORMING  (Poisson roll)
            FORMING → ACTIVE  (age > bridge_formation_time)
            ACTIVE → LOCKED  (force > lock threshold)
            ACTIVE → BROKEN  (gap > break gap)
            ACTIVE → SENESCENT  (age > senescence time, unless locked)
        """
        bridge_rate = float(self._get('bridge_attempt_rate', 0.3))
        form_time = float(self._get('bridge_formation_time', 2.0))
        senes_time = float(self._get('bridge_senescence_time', 24.0))
        min_fa = float(self._get('min_fa_for_bridge', 0.3))
        break_gap = float(self._get('bridge_break_gap', 60.0))
        lock_thresh = float(self._get('bridge_lock_force_threshold', 20.0))
        sense_dist = float(self._get('cell_sense_distance', 40.0))
        contact_factor = float(self._get('bridge_contact_factor', 5.0))
        decay_length = float(self._get('bridge_decay_length', 10.0))
        inert_factor = float(self._get('bridge_inert_factor', 0.1))
        exclusion_angle = float(self._get('bridge_exclusion_angle', 0.8))
        alignment_rate = float(self._get('bridge_alignment_rate', 0.3))
        alignment_min = float(self._get('bridge_alignment_min', 0.3))
        secondary_mult = float(self._get('bridge_secondary_rate_mult', 3.0))

        n_edges = len(self._edge_i)
        ei = self._edge_i
        ej = self._edge_j
        overlap = self._edge_overlap
        dist = self._edge_dist

        # --- Update existing bridges ---
        keys_to_remove = []
        for key, br in self._bridges.items():
            if br['state'] in (BridgeState.BROKEN, BridgeState.SENESCENT):
                keys_to_remove.append(key)
                continue

            br['age'] += dt

            # Check if edge still exists (gap not too large)
            i, j = key
            gap = dist_between(self, i, j)
            if gap is None or gap > break_gap:
                br['state'] = BridgeState.BROKEN
                continue

            # Ramp update
            if form_time > 0:
                br['ramp'] = min(1.0, br['age'] / form_time)
            else:
                br['ramp'] = 1.0

            # Alignment growth
            br['alignment'] = min(1.0, br['alignment'] +
                                  alignment_rate * dt)

            # State transitions
            if br['state'] == BridgeState.FORMING:
                if br['age'] >= form_time:
                    br['state'] = BridgeState.ACTIVE

            elif br['state'] == BridgeState.ACTIVE:
                if br['force'] >= lock_thresh:
                    br['state'] = BridgeState.LOCKED
                elif br['age'] >= senes_time:
                    br['state'] = BridgeState.SENESCENT

            # LOCKED bridges persist indefinitely

        for key in keys_to_remove:
            del self._bridges[key]

        # --- Attempt new bridges on FF edges ---
        for k in range(n_edges):
            i_idx, j_idx = int(ei[k]), int(ej[k])
            # Only func-func pairs
            if self.gtype[i_idx] != 0 or self.gtype[j_idx] != 0:
                continue

            key = (min(i_idx, j_idx), max(i_idx, j_idx))
            if key in self._bridges:
                continue

            gap = max(0.0, -overlap[k])  # gap > 0 means separated
            if gap > sense_dist:
                continue

            # FA maturity check
            mat = max(self.fa_maturity[i_idx], self.fa_maturity[j_idx])
            if mat < min_fa:
                continue

            # Path factor
            if overlap[k] > 0:
                path_factor = contact_factor
            else:
                path_factor = np.exp(-gap / max(decay_length, 1.0))

            # Boost rate if existing bridges on either granule
            has_existing = any(
                (key2[0] == i_idx or key2[1] == i_idx or
                 key2[0] == j_idx or key2[1] == j_idx)
                and br2['state'] in (BridgeState.FORMING,
                                     BridgeState.ACTIVE,
                                     BridgeState.LOCKED)
                for key2, br2 in self._bridges.items()
            )
            rate = bridge_rate * (secondary_mult if has_existing else 1.0)

            # Poisson probability
            prob = 1.0 - np.exp(-rate * dt * mat * path_factor)
            if rng.random() < prob:
                self._bridges[key] = {
                    'state': BridgeState.FORMING,
                    'age': 0.0,
                    'force': 0.0,
                    'ramp': 0.0,
                    'alignment': alignment_min,
                }

    # ------------------------------------------------------------------
    # FA maturity
    # ------------------------------------------------------------------

    def _update_fa_maturity(self, t):
        """Update focal adhesion maturity for functional granules."""
        t_spread = float(self._get('t_spread_duration', 3.0))
        fa_rate = float(self._get('fa_maturation_rate', 0.3))
        t_eff = max(0.0, t - t_spread)
        mat = 1.0 - np.exp(-fa_rate * t_eff)
        # Only functional granules have cells
        self.fa_maturity = np.where(self.gtype == 0, mat, 0.0)

    # ------------------------------------------------------------------
    # Integration step
    # ------------------------------------------------------------------

    def step(self, dt, t, rng):
        """Advance one timestep.

        1. Update FA maturity
        2. Update bridge states and attempt new bridges
        3. Compute forces (contact + bridge + wall)
        4. Overdamped position update
        5. Boundary handling
        6. Rebuild neighbor list
        """
        self._update_fa_maturity(t)
        self._update_bridges(dt, t, rng)
        F = self._compute_forces(t)

        # Overdamped: v = F / γ, where γ = drag_scale * R
        drag_scale = float(self._get('drag_scale', 0.05))
        v_max = float(self._get('v_max', 20.0))
        gamma = drag_scale * self.radii
        vel = F / gamma[:, None]

        # Velocity cap
        speed = np.sqrt(np.sum(vel ** 2, axis=1))
        over = speed > v_max
        if np.any(over):
            vel[over] *= (v_max / speed[over])[:, None]

        self.vel = vel
        self.pos += vel * dt

        # Boundary handling
        if self._get('boundary_mode', 'walls') == 'periodic':
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            self.pos[:, 0] %= Lx
            self.pos[:, 1] %= Ly
            if self.is_3d:
                Lz = float(self._get('Lz', 800.0))
                self.pos[:, 2] %= Lz
        else:
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            r = self.radii
            self.pos[:, 0] = np.clip(self.pos[:, 0], r + 0.5, Lx - r - 0.5)
            self.pos[:, 1] = np.clip(self.pos[:, 1], r + 0.5, Ly - r - 0.5)
            if self.is_3d:
                Lz = float(self._get('Lz', 800.0))
                self.pos[:, 2] = np.clip(self.pos[:, 2], r + 0.5, Lz - r - 0.5)

        # Resolve encapsulation: push apart any pair whose overlap exceeds
        # the max allowed (30% of smaller radius).  Two passes to resolve
        # chains of overlapping granules.
        self._resolve_overlaps()

        # Rebuild edges
        self._rebuild_edges()

    def _resolve_overlaps(self, max_passes=3):
        """Push apart granule pairs whose overlap exceeds the allowed cap.

        After the velocity update, some pairs may still have excessive
        overlap (especially under large bridge attraction).  This
        projection step enforces the hard overlap limit by shifting both
        particles symmetrically along the contact normal.
        """
        for _ in range(max_passes):
            tree = cKDTree(self.pos)
            max_r = float(np.max(self.radii))
            pairs = tree.query_pairs(2.0 * max_r, output_type='ndarray')
            if len(pairs) == 0:
                break

            ei = pairs[:, 0]
            ej = pairs[:, 1]
            dp = self.pos[ej] - self.pos[ei]
            dist = np.sqrt(np.sum(dp ** 2, axis=1))
            overlap = self.radii[ei] + self.radii[ej] - dist
            delta_max = self._max_overlap(self.radii[ei], self.radii[ej],
                                           frac=0.3)
            excess = overlap - delta_max

            violated = excess > 0.1  # µm tolerance
            if not np.any(violated):
                break

            idx = np.where(violated)[0]
            for k in idx:
                d = max(dist[k], 1e-6)
                n = dp[k] / d
                shift = 0.5 * excess[k] * n
                self.pos[ei[k]] -= shift
                self.pos[ej[k]] += shift

            # Re-clip to domain
            if self._get('boundary_mode', 'walls') != 'periodic':
                Lx = float(self._get('Lx', 800.0))
                Ly = float(self._get('Ly', 800.0))
                r = self.radii
                self.pos[:, 0] = np.clip(self.pos[:, 0], r + 0.5,
                                         Lx - r - 0.5)
                self.pos[:, 1] = np.clip(self.pos[:, 1], r + 0.5,
                                         Ly - r - 0.5)
                if self.is_3d:
                    Lz = float(self._get('Lz', 800.0))
                    self.pos[:, 2] = np.clip(self.pos[:, 2], r + 0.5,
                                             Lz - r - 0.5)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def metrics(self) -> dict:
        """Compute current snapshot metrics.

        Returns
        -------
        dict with keys:
            Z, Z_ff, Z_fi, Z_ii : coordination numbers
            n_contacts, n_contacts_ff, n_contacts_fi, n_contacts_ii : counts
            n_bridges, n_bridges_active, n_bridges_locked : bridge counts
            bridge_fraction : fraction of FF edges with active/locked bridges
            percolation : bool, whether bridge-connected cluster spans domain
            largest_cluster_size : int
            cluster_sizes : list of int
            mean_force_contact, mean_force_bridge : average forces (nN)
            force_chain_lengths : list of chain lengths
        """
        ei = self._edge_i
        ej = self._edge_j
        overlap = self._edge_overlap

        # Contact counts by type
        contact_mask = overlap > 0
        ci_c = ei[contact_mask]
        cj_c = ej[contact_mask]
        gi_c = self.gtype[ci_c]
        gj_c = self.gtype[cj_c]

        n_contacts = int(np.sum(contact_mask))
        n_ff = int(np.sum((gi_c == 0) & (gj_c == 0)))
        n_fi = int(np.sum(gi_c != gj_c))
        n_ii = int(np.sum((gi_c == 1) & (gj_c == 1)))

        N_func = int(np.sum(self.gtype == 0))
        N_inert = int(np.sum(self.gtype == 1))
        N_total = max(1, self.N)

        Z = 2.0 * n_contacts / N_total
        Z_ff = 2.0 * n_ff / max(1, N_func)
        Z_fi = n_fi / max(1, N_func + N_inert)
        Z_ii = 2.0 * n_ii / max(1, N_inert)

        # Bridge counts
        n_bridges = len(self._bridges)
        n_active = sum(1 for b in self._bridges.values()
                       if b['state'] in (BridgeState.ACTIVE, BridgeState.LOCKED))
        n_locked = sum(1 for b in self._bridges.values()
                       if b['state'] == BridgeState.LOCKED)
        n_forming = sum(1 for b in self._bridges.values()
                        if b['state'] == BridgeState.FORMING)

        # Bridge fraction: active+locked bridges / total FF edges within sensing
        sense_dist = float(self._get('cell_sense_distance', 40.0))
        ff_edges = 0
        for k in range(len(ei)):
            if self.gtype[ei[k]] == 0 and self.gtype[ej[k]] == 0:
                gap = max(0.0, -overlap[k])
                if gap <= sense_dist:
                    ff_edges += 1
        bridge_fraction = n_active / max(1, ff_edges)

        # Bridge forces
        bridge_forces = [b['force'] for b in self._bridges.values()
                         if b['state'] in (BridgeState.ACTIVE, BridgeState.LOCKED)
                         and b['force'] > 0]
        mean_force_bridge = float(np.mean(bridge_forces)) if bridge_forces else 0.0

        # Contact forces
        contact_forces = self._hertz_force(
            np.maximum(overlap[contact_mask], 0.0),
            self.radii[ci_c] * self.radii[cj_c] /
            (self.radii[ci_c] + self.radii[cj_c])
        ) if n_contacts > 0 else np.array([])
        mean_force_contact = float(np.mean(contact_forces)) if len(contact_forces) > 0 else 0.0

        # Cluster analysis (connected components of bridge-connected functional granules)
        cluster_sizes, largest, percolation = self._cluster_analysis()

        return {
            'Z': Z, 'Z_ff': Z_ff, 'Z_fi': Z_fi, 'Z_ii': Z_ii,
            'n_contacts': n_contacts,
            'n_contacts_ff': n_ff, 'n_contacts_fi': n_fi, 'n_contacts_ii': n_ii,
            'n_bridges': n_bridges,
            'n_bridges_forming': n_forming,
            'n_bridges_active': n_active,
            'n_bridges_locked': n_locked,
            'bridge_fraction': bridge_fraction,
            'percolation': percolation,
            'largest_cluster_size': largest,
            'cluster_sizes': cluster_sizes,
            'mean_force_contact': mean_force_contact,
            'mean_force_bridge': mean_force_bridge,
        }

    def _cluster_analysis(self):
        """Connected components of bridge-connected functional granules.

        Returns (cluster_sizes, largest_cluster_size, percolation_bool).
        Percolation = largest bridge-cluster spans > 50% of domain in any axis.
        """
        func_idx = np.where(self.gtype == 0)[0]
        N_func = len(func_idx)
        if N_func == 0:
            return [], 0, False

        # Build sparse adjacency for active/locked bridges between functional granules
        # Map global indices to local (functional-only) indices
        global_to_local = -np.ones(self.N, dtype=np.int32)
        global_to_local[func_idx] = np.arange(N_func)

        rows, cols = [], []
        for (i, j), br in self._bridges.items():
            if br['state'] not in (BridgeState.ACTIVE, BridgeState.LOCKED):
                continue
            li = global_to_local[i]
            lj = global_to_local[j]
            if li >= 0 and lj >= 0:
                rows.extend([li, lj])
                cols.extend([lj, li])

        if not rows:
            return [1] * N_func, 1, False

        data = np.ones(len(rows), dtype=np.int8)
        adj = sparse.csr_matrix((data, (rows, cols)),
                                shape=(N_func, N_func))
        n_components, labels = sparse.csgraph.connected_components(
            adj, directed=False)

        cluster_sizes = []
        for c in range(n_components):
            cluster_sizes.append(int(np.sum(labels == c)))
        cluster_sizes.sort(reverse=True)
        largest = cluster_sizes[0] if cluster_sizes else 0

        # Percolation check: does largest cluster span > 50% of domain?
        percolation = False
        if largest >= 2:
            largest_mask = labels == np.argmax(np.bincount(labels))
            cl_pos = self.pos[func_idx[largest_mask]]
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            span_x = (np.max(cl_pos[:, 0]) - np.min(cl_pos[:, 0])) / Lx
            span_y = (np.max(cl_pos[:, 1]) - np.min(cl_pos[:, 1])) / Ly
            if span_x > 0.5 or span_y > 0.5:
                percolation = True
            if self.is_3d:
                Lz = float(self._get('Lz', 800.0))
                span_z = (np.max(cl_pos[:, 2]) - np.min(cl_pos[:, 2])) / Lz
                if span_z > 0.5:
                    percolation = True

        return cluster_sizes, largest, percolation

    def adjacency_matrix(self, edge_type='contact'):
        """Sparse adjacency matrix.

        Parameters
        ----------
        edge_type : str
            'contact' (overlap > 0), 'bridge' (active/locked bridges),
            'all' (any edge within cutoff).

        Returns
        -------
        scipy.sparse.csr_matrix (N, N)
        """
        ei = self._edge_i
        ej = self._edge_j
        overlap = self._edge_overlap

        if edge_type == 'contact':
            mask = overlap > 0
            rows = np.concatenate([ei[mask], ej[mask]])
            cols = np.concatenate([ej[mask], ei[mask]])
        elif edge_type == 'bridge':
            rows, cols = [], []
            for (i, j), br in self._bridges.items():
                if br['state'] in (BridgeState.ACTIVE, BridgeState.LOCKED):
                    rows.extend([i, j])
                    cols.extend([j, i])
            rows = np.array(rows, dtype=np.int32)
            cols = np.array(cols, dtype=np.int32)
        else:  # 'all'
            rows = np.concatenate([ei, ej])
            cols = np.concatenate([ej, ei])

        if len(rows) == 0:
            return sparse.csr_matrix((self.N, self.N))

        data = np.ones(len(rows), dtype=np.float64)
        return sparse.csr_matrix((data, (rows, cols)),
                                 shape=(self.N, self.N))

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------

    def solve(self, t_span=(0, 72), dt=None, seed=None,
              record_every=None) -> dict:
        """Integrate forward in time.

        Parameters
        ----------
        t_span : tuple (t_start, t_end)
            Time range in hours.
        dt : float or None
            Timestep (hours). Defaults to p.dt.
        seed : int or None
            RNG seed for bridge stochasticity.
        record_every : float or None
            Record metrics every N hours. Defaults to p.save_every_h.

        Returns
        -------
        dict
            'time': array of recorded times
            'metrics': list of metrics dicts (one per recorded time)
            'positions_final': (N, D) final positions
            'bridges_final': dict of final bridge states
        """
        if dt is None:
            dt = float(self._get('dt', 0.5))
        if record_every is None:
            record_every = float(self._get('save_every_h', 2.0))

        rng = np.random.default_rng(seed)
        t_start, t_end = t_span
        t = t_start
        next_record = t_start

        times = []
        metrics_list = []

        while t <= t_end + 1e-10:
            if t >= next_record - 1e-10:
                m = self.metrics()
                m['time'] = t
                metrics_list.append(m)
                times.append(t)
                next_record += record_every

            if t >= t_end:
                break
            self.step(dt, t, rng)
            t += dt

        return {
            'time': np.array(times),
            'metrics': metrics_list,
            'positions_final': self.pos.copy(),
            'bridges_final': dict(self._bridges),
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_snapshot(cls, snap, p):
        """Build network from a DEM snapshot.

        Parameters
        ----------
        snap : dict
            Snapshot dict from load_run() with x, y, [z], r, gtype, etc.
        p : Params
            Simulation parameters.

        Returns
        -------
        ContactNetwork
        """
        x = np.asarray(snap['x'])
        y = np.asarray(snap['y'])
        r = np.asarray(snap['r'])
        gtype = np.asarray(snap['gtype'], dtype=np.int32)
        N = len(x)

        mode = getattr(p, 'mode', '2D')
        is_3d = mode == '3D'

        if is_3d and 'z' in snap:
            z = np.asarray(snap['z'])
            pos = np.column_stack([x, y, z])
        else:
            pos = np.column_stack([x, y])
            is_3d = False

        # Wrap positions into domain for periodic BCs (cKDTree requires [0, L])
        boundary = getattr(p, 'boundary_mode', 'walls')
        if boundary == 'periodic':
            Lx = float(getattr(p, 'Lx', 800.0))
            Ly = float(getattr(p, 'Ly', 800.0))
            pos[:, 0] = pos[:, 0] % Lx
            pos[:, 1] = pos[:, 1] % Ly
            if is_3d:
                Lz = float(getattr(p, 'Lz', 800.0))
                pos[:, 2] = pos[:, 2] % Lz

        # Cell count per granule
        n_cells_per = int(getattr(p, 'n_cells_per_granule', 8))
        n_cells = np.where(gtype == 0, n_cells_per, 0).astype(np.int32)

        # If individual cell data is available, use actual counts
        if 'cell_granule_id' in snap and 'cell_offset' in snap:
            offsets = np.asarray(snap['cell_offset'])
            if len(offsets) == N + 1:
                n_cells = (offsets[1:] - offsets[:-1]).astype(np.int32)

        return cls(pos, r, gtype, n_cells, p, is_3d=is_3d)

    @classmethod
    def from_packing(cls, p, seed=None):
        """Generate jammed sphere packing via the DEM's Lubachevsky-Stillinger
        packer and build network from it.

        Delegates to ``generate_packing()`` / ``generate_packing_3d()`` from
        ``new_dem_0.py`` which performs proper RSA + inflate-and-relax with
        Hertz repulsion (400 steps × 15 substeps) to achieve a jammed state.

        Parameters
        ----------
        p : Params
            Simulation parameters (passed directly to the DEM packer).
        seed : int or None
            RNG seed.

        Returns
        -------
        ContactNetwork
        """
        from new_dem_0 import generate_packing, generate_packing_3d

        mode = getattr(p, 'mode', '2D')
        is_3d = (mode == '3D')
        _seed = seed if seed is not None else 42

        # Use the DEM's full Lubachevsky-Stillinger packer
        if is_3d:
            gs = generate_packing_3d(p, seed=_seed)
        else:
            gs = generate_packing(p, seed=_seed)

        N = gs.N
        x = gs.x[:N].copy()
        y = gs.y[:N].copy()
        radii = gs.r[:N].copy()
        gtype = gs.gtype[:N].copy().astype(np.int32)
        n_cells = gs.n_cells[:N].copy().astype(np.int32)

        if is_3d:
            z = gs.z[:N].copy()
            pos = np.column_stack([x, y, z])
        else:
            pos = np.column_stack([x, y])

        return cls(pos, radii, gtype, n_cells, p, is_3d=is_3d)

    @classmethod
    def from_params(cls, p, seed=None):
        """Convenience alias for from_packing."""
        return cls.from_packing(p, seed=seed)

    @classmethod
    def from_run(cls, run_dir, snap_index=0):
        """Load DEM output and build network from a specific snapshot.

        Parameters
        ----------
        run_dir : str
            Path to simulation output directory or .tar.gz archive.
        snap_index : int
            Which snapshot to use (0 = initial packing).

        Returns
        -------
        tuple of (ContactNetwork, dict, Params)
            (network, hist, params)
        """
        from new_dem_0 import load_run
        hist, snaps, p, metadata = load_run(run_dir)
        if not snaps:
            raise ValueError(f"No snapshots found in {run_dir}")
        snap = snaps[min(snap_index, len(snaps) - 1)]
        net = cls.from_snapshot(snap, p)
        return net, hist, p

    # ------------------------------------------------------------------
    # Descriptors (tissue architecture)
    # ------------------------------------------------------------------

    def descriptors(self) -> dict:
        """Compute tissue architecture descriptors from current graph state.

        Returns dict compatible with analysis/tissue_descriptors.py format.
        These are topology-derived descriptors that complement the
        field-based descriptors from the SpatialPDE model.
        """
        m = self.metrics()
        N_func = int(np.sum(self.gtype == 0))
        N_inert = int(np.sum(self.gtype == 1))

        # Packing fractions from sphere volumes
        if self.is_3d:
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            Lz = float(self._get('Lz', 800.0))
            domain_vol = Lx * Ly * Lz
            gran_vols = (4.0 / 3.0) * np.pi * self.radii ** 3
        else:
            Lx = float(self._get('Lx', 800.0))
            Ly = float(self._get('Ly', 800.0))
            domain_vol = Lx * Ly
            gran_vols = np.pi * self.radii ** 2

        phi_f = np.sum(gran_vols[self.gtype == 0]) / domain_vol
        phi_i = np.sum(gran_vols[self.gtype == 1]) / domain_vol
        phi_v = 1.0 - phi_f - phi_i

        # Permeability (Kozeny-Carman)
        d_f = 2.0 * float(np.mean(self.radii[self.gtype == 0])) if N_func > 0 else 80.0
        d_i = 2.0 * float(np.mean(self.radii[self.gtype == 1])) if N_inert > 0 else 120.0
        eps = max(0.01, min(0.99, phi_v))
        d_grain = (phi_f * d_f + phi_i * d_i) / max(phi_f + phi_i, 1e-6)
        K_kc = eps ** 3 * d_grain ** 2 / (180.0 * (1.0 - eps) ** 2)

        return {
            'BV_TV': phi_f + phi_i,
            'porosity': phi_v,
            'phi_f': phi_f,
            'phi_i': phi_i,
            'Z': m['Z'],
            'Z_ff': m['Z_ff'],
            'Z_fi': m['Z_fi'],
            'n_bridges_active': m['n_bridges_active'],
            'bridge_fraction': m['bridge_fraction'],
            'percolation': m['percolation'],
            'largest_cluster_size': m['largest_cluster_size'],
            'K_kozeny_carman': K_kc,
            'd_grain_mean': d_grain,
            'mean_force_contact': m['mean_force_contact'],
            'mean_force_bridge': m['mean_force_bridge'],
        }


# ======================================================================
# Utility
# ======================================================================

def dist_between(net, i, j):
    """Distance between granules i and j in the network.

    Returns surface-to-surface gap (negative = overlap).
    Returns None if indices out of range.
    """
    if i >= net.N or j >= net.N:
        return None
    dp = net.pos[j] - net.pos[i]
    if net._get('boundary_mode', 'walls') == 'periodic':
        Lx = float(net._get('Lx', 800.0))
        Ly = float(net._get('Ly', 800.0))
        dp[0] -= Lx * round(dp[0] / Lx)
        dp[1] -= Ly * round(dp[1] / Ly)
        if net.is_3d:
            Lz = float(net._get('Lz', 800.0))
            dp[2] -= Lz * round(dp[2] / Lz)
    d = np.sqrt(np.sum(dp ** 2))
    return d - net.radii[i] - net.radii[j]


def run_ensemble(p, n_realizations=10, t_span=(0, 72), seed=None, **kwargs):
    """Run multiple stochastic realizations and return aggregated metrics.

    Parameters
    ----------
    p : Params
        Simulation parameters.
    n_realizations : int
        Number of Monte Carlo realizations.
    t_span : tuple
        Time range.
    seed : int or None
        Base seed (each realization uses seed + i).

    Returns
    -------
    dict
        'time': shared time array
        'metrics_mean': dict of arrays (mean across realizations)
        'metrics_std': dict of arrays (std across realizations)
        'all_metrics': list of per-realization metric lists
    """
    base_seed = seed if seed is not None else 0
    all_results = []

    for i in range(n_realizations):
        net = ContactNetwork.from_packing(p, seed=base_seed + i)
        result = net.solve(t_span=t_span, seed=base_seed + 1000 + i, **kwargs)
        all_results.append(result)

    # Aggregate scalar metrics
    times = all_results[0]['time']
    n_times = len(times)
    scalar_keys = ['Z', 'Z_ff', 'Z_fi', 'Z_ii',
                   'n_contacts', 'n_bridges_active', 'n_bridges_locked',
                   'bridge_fraction', 'mean_force_contact', 'mean_force_bridge',
                   'largest_cluster_size']

    means = {}
    stds = {}
    for key in scalar_keys:
        vals = np.zeros((n_realizations, n_times))
        for r_i, res in enumerate(all_results):
            for t_i, m in enumerate(res['metrics']):
                if t_i < n_times:
                    vals[r_i, t_i] = m.get(key, 0.0)
        means[key] = np.mean(vals, axis=0)
        stds[key] = np.std(vals, axis=0)

    # Percolation time: first time percolation == True (per realization)
    perc_times = []
    for res in all_results:
        perc_t = None
        for m in res['metrics']:
            if m.get('percolation', False):
                perc_t = m['time']
                break
        perc_times.append(perc_t)

    return {
        'time': times,
        'metrics_mean': means,
        'metrics_std': stds,
        'percolation_times': perc_times,
        'all_results': all_results,
    }


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Contact network mesoscale model for granular scaffold compaction.')
    parser.add_argument('-i', '--input', default=None,
                        help='Path to DEM output directory (optional)')
    parser.add_argument('--standalone', action='store_true',
                        help='Run standalone (generate packing, no DEM input)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t-total', type=float, default=72.0)
    parser.add_argument('--ensemble', type=int, default=0,
                        help='Number of Monte Carlo realizations (0=single run)')
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from new_dem_0 import Params

    if args.input and not args.standalone:
        print(f"Loading DEM output from {args.input}...")
        net, hist, p = ContactNetwork.from_run(args.input)
        outdir = args.outdir or str(Path(args.input) / 'plots_network')
    else:
        print("Running standalone (generating packing)...")
        p = Params()
        net = ContactNetwork.from_packing(p, seed=args.seed)
        outdir = args.outdir or 'results/network_standalone'

    Path(outdir).mkdir(parents=True, exist_ok=True)

    if args.ensemble > 0:
        print(f"Running {args.ensemble}-realization ensemble...")
        ens = run_ensemble(p, n_realizations=args.ensemble,
                           t_span=(0, args.t_total), seed=args.seed)
        # Save summary
        import json
        summary = {
            'n_realizations': args.ensemble,
            't_total': args.t_total,
            'percolation_times': [float(t) if t is not None else None
                                  for t in ens['percolation_times']],
        }
        for key in ens['metrics_mean']:
            summary[f'{key}_final_mean'] = float(ens['metrics_mean'][key][-1])
            summary[f'{key}_final_std'] = float(ens['metrics_std'][key][-1])

        with open(str(Path(outdir) / 'ensemble_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Ensemble summary saved to {outdir}/ensemble_summary.json")
    else:
        print(f"Running single simulation (seed={args.seed})...")
        result = net.solve(t_span=(0, args.t_total), seed=args.seed)

        # Print final metrics
        final = result['metrics'][-1]
        print(f"\n{'='*60}")
        print(f"  Contact Network Model — t = {args.t_total} h")
        print(f"{'='*60}")
        print(f"  Granules: {net.N} ({np.sum(net.gtype==0)} func, {np.sum(net.gtype==1)} inert)")
        print(f"  Z = {final['Z']:.2f}  (Z_ff={final['Z_ff']:.2f}, Z_fi={final['Z_fi']:.2f}, Z_ii={final['Z_ii']:.2f})")
        print(f"  Contacts: {final['n_contacts']}")
        print(f"  Bridges: {final['n_bridges_active']} active, {final['n_bridges_locked']} locked")
        print(f"  Bridge fraction: {final['bridge_fraction']:.3f}")
        print(f"  Percolation: {final['percolation']}")
        print(f"  Largest cluster: {final['largest_cluster_size']}")
        print(f"  Mean contact force: {final['mean_force_contact']:.2f} nN")
        print(f"  Mean bridge force: {final['mean_force_bridge']:.2f} nN")

        # Save metrics timeseries
        import json
        ts_out = {key: [float(m.get(key, 0)) for m in result['metrics']]
                  for key in ['Z', 'Z_ff', 'Z_fi', 'Z_ii', 'n_contacts',
                              'n_bridges_active', 'n_bridges_locked',
                              'bridge_fraction', 'largest_cluster_size',
                              'mean_force_contact', 'mean_force_bridge']}
        ts_out['time'] = result['time'].tolist()
        ts_out['percolation'] = [bool(m.get('percolation', False))
                                 for m in result['metrics']]

        with open(str(Path(outdir) / 'network_timeseries.json'), 'w') as f:
            json.dump(ts_out, f, indent=2)
        print(f"\n  Timeseries saved to {outdir}/network_timeseries.json")
