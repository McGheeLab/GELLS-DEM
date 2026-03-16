#!/usr/bin/env python3
"""
Energy Landscape Analysis for Cell-Driven Granular Scaffold Compaction
======================================================================

Models the free energy of the granular scaffold as a function of the
compaction coordinate  ξ = 1 − x_f / x_{f,0}  ∈ [0, ξ_max],
decomposed into six physically motivated terms:

1. G_cell    — Cell traction + bridge adhesion      (negative → drives compaction)
2. G_elastic — Hertzian elastic contact energy       (resists at high packing)
3. G_yield   — Herschel-Bulkley yield barrier        (threshold for rearrangement)
4. G_void    — Void redistribution osmotic energy    (entropic resistance)
5. G_inert   — Inert granule frustration energy      (geometric barriers)
6. G_surface — Interfacial tension at f/i boundary   (favours compact, round zones)

The equilibrium compaction  ξ*  satisfies  dG/dξ = 0,  and kinetics
follow overdamped dynamics:  dξ/dt = −(1/η_eff) dG/dξ.

Energy terms are derived to be consistent with the mean-field ODE model
in ``_solve_trajectory``.  The cell-elastic balance sets the primary
equilibrium; yield, void, inert, and surface terms are perturbative
corrections that shift the minimum and shape the landscape.

Energies are reported normalised by the cell stress scale σ_cell so that
the landscape has O(1) structure on the y-axis.

Units (internal): kPa for energy density, µm for length, nN for force,
hours for time.  Plots use dimensionless energy G̃ = G / σ_cell.
"""

import numpy as np
from scipy.optimize import minimize_scalar

from analysis.parameter_sweep import (
    compute_phi_RCP, compute_phi_max_deformable,
    motor_clutch_force_vec, effective_contact_modulus,
    bridge_rate_modifier,
)


# ======================================================================
# Energy landscape class
# ======================================================================

class EnergyLandscape:
    """Free energy landscape for cell-driven granular scaffold compaction.

    Parameters
    ----------
    params : dict
        Parameter dict (same format as recommendation params from
        parameter_sweep.py).  Required keys: phi_f, phi_i, R_func,
        R_inert, E_func, E_inert, aspect_ratio_func, aspect_ratio_inert,
        blockiness_n2_func, blockiness_n2_inert, n_cells_per_func.
    """

    def __init__(self, params):
        self.params = dict(params)  # own copy
        self._compute_derived()

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def _compute_derived(self):
        p = self.params
        _a = lambda v: np.array([v])

        self.phi_f = p['phi_f']
        self.phi_i = p['phi_i']
        self.phi_solid = self.phi_f + self.phi_i
        self.R_func = p['R_func']
        self.R_inert = p['R_inert']
        self.E_func = p['E_func']
        self.E_inert = p['E_inert']

        # Packing limits
        self.phi_RCP = compute_phi_RCP(
            _a(p['aspect_ratio_func']), _a(p['aspect_ratio_inert']),
            _a(p['blockiness_n2_func']), _a(p['blockiness_n2_inert']),
            _a(self.phi_f), _a(self.phi_i),
            _a(self.R_func), _a(self.R_inert),
        )[0]
        self.phi_max = compute_phi_max_deformable(
            _a(self.phi_RCP), _a(self.E_func), _a(p['blockiness_n2_func'])
        )[0]

        # Cell mechanics
        self.F_cell = motor_clutch_force_vec(_a(self.E_func))[0]
        self.E_eff = effective_contact_modulus(
            _a(self.E_func), _a(self.E_inert),
            _a(self.phi_f), _a(self.phi_i),
        )[0]
        sense = p.get('cell_sense_distance', 50.0)
        self.br_mod = bridge_rate_modifier(
            _a(sense), _a(self.R_func),
            _a(self.phi_f), _a(self.phi_i), _a(self.phi_RCP)
        )[0]

        # Zone fractions
        self.x_f0 = max(self.phi_f / self.phi_solid,
                        self.phi_f / max(self.phi_max, 0.01))
        self.x_f_min = self.phi_f / max(self.phi_max, 0.01)
        self.xi_max = max(1e-3, 1.0 - self.x_f_min / self.x_f0)

        # Cell stress scale  (the natural energy unit)
        domain_area = 800.0 * 800.0
        N_func = max(1.0, self.phi_f * domain_area / (np.pi * self.R_func**2))
        n_cells = float(p['n_cells_per_func'])
        self.rho_cell = N_func * n_cells / domain_area
        self.sigma_cell = self.rho_cell * self.F_cell   # kPa

        # Effective viscosity (consistent with mean-field model)
        self.eta_eff = np.clip(
            0.5 * (self.E_eff / 5.0)**0.3 * (self.R_func / 40.0)**1.5,
            0.01, 100.0
        )

        # --- ODE-consistent elastic resistance ---
        # From mean-field: σ_resist = σ_0 max(0, φ_local/φ_RCP − 1)
        self.sigma_0 = max(0.005 * self.E_eff, 1e-6)

        # Onset of elastic resistance: ξ_c where φ_local = φ_RCP
        u = self.phi_f / (self.x_f0 * self.phi_RCP)
        self.xi_onset = max(0.0, 1.0 - u)  # φ_local = φ_RCP at this ξ

        # Jamming onset: ξ_J where φ_local = φ_J < φ_RCP
        self.phi_J = self.phi_RCP * 0.92
        u_J = self.phi_f / (self.x_f0 * self.phi_J)
        self.xi_jam = max(0.0, 1.0 - u_J)  # φ_local = φ_J at this ξ

        # --- Perturbative term scales (relative to sigma_cell) ---

        # Herschel-Bulkley: yield barrier near jamming, fraction of σ_0
        # but capped so it doesn't overwhelm cell driving
        self.c_yield = 0.25   # σ_y scale = c_yield * σ_0
        self.HB_delta = 1.5

        # Void osmotic: quadratic cost of void redistribution
        self.Pi_osm = 3.0 * self.sigma_cell

        # Inert frustration: geometric obstacle cost
        self.k_frust = 1.5 * self.sigma_cell

        # Interfacial tension
        E_mismatch = abs(self.E_func - self.E_inert) / max(
            self.E_func + self.E_inert, 0.1)
        self.gamma_fi = 2.5 * self.sigma_cell * (0.3 + E_mismatch)

        # Initial local void fractions
        self.phi_v_f0 = max(0, 1.0 - self.phi_f / max(self.x_f0, 1e-8))
        x_i0 = 1.0 - self.x_f0
        self.phi_v_i0 = max(0, 1.0 - self.phi_i / max(x_i0, 1e-8))

    # ------------------------------------------------------------------
    # Local packing fractions
    # ------------------------------------------------------------------

    def _phi_f_local(self, xi):
        """Functional granule local packing fraction at compaction ξ."""
        x_f = self.x_f0 * (1.0 - xi)
        return self.phi_f / max(x_f, 1e-8)

    def _phi_i_local(self, xi):
        """Inert granule local packing fraction at compaction ξ."""
        x_i = 1.0 - self.x_f0 * (1.0 - xi)
        return self.phi_i / max(x_i, 1e-8)

    # ------------------------------------------------------------------
    # Individual energy terms  (all return energy density in kPa)
    # ------------------------------------------------------------------

    def G_cell(self, xi, f_bridge=1.0, maturity=1.0):
        """Cell traction driving energy (negative).

        Derived from ODE:  dG_cell/dξ = −(1−ξ) σ_cell
        With density enhancement (cell concentration rises as zone
        shrinks), the integrated form becomes:

            G_cell(ξ) = σ_cell · ln(1 − ξ)     (< 0 for ξ > 0)
        """
        sigma = self.sigma_cell * f_bridge * maturity
        xi_s = np.clip(xi, 0, 1.0 - 1e-6)
        if np.ndim(xi_s) == 0:
            return sigma * np.log(1.0 - float(xi_s)) if xi_s > 0 else 0.0
        return np.where(xi_s > 0, sigma * np.log(1.0 - xi_s), 0.0)

    def G_elastic(self, xi):
        """Hertzian elastic contact energy (ODE-consistent).

        Derived by integrating the mean-field resistance stress:
            σ_resist = σ_0 max(0, φ_local/φ_RCP − 1)

        Yields a parabola above onset ξ_c:
            G_elastic(ξ) = (σ_0 / 2) · max(0, ξ − ξ_c)²

        This is exact for the linearised ODE near ξ_c and gives the
        correct equilibrium σ_cell = σ_resist.
        """
        delta_xi = np.maximum(0.0, xi - self.xi_onset)
        return self.sigma_0 / 2.0 * delta_xi**2

    def G_yield(self, xi):
        """Herschel-Bulkley yield barrier energy.

        The granular scaffold has a yield stress near jamming:
            σ_y = c_yield · σ_0 · max(0, φ_local/φ_J − 1)^Δ

        The yield barrier acts between ξ_J (jamming onset) and ξ_c
        (elastic onset), creating a threshold that cells must overcome
        before elastic compression begins.

            G_yield = σ_y(ξ) · max(0, ξ − ξ_J)
        """
        phi_local = self._phi_f_local(xi)
        excess = np.maximum(0.0, phi_local / self.phi_J - 1.0)
        sigma_y = self.c_yield * self.sigma_0 * excess**self.HB_delta
        delta = np.maximum(0.0, xi - self.xi_jam)
        return sigma_y * delta

    def G_void(self, xi):
        """Void redistribution osmotic energy.

        As voids are expelled from the functional zone, local void
        fractions depart from their initial equilibrium values.
        The osmotic cost is quadratic in the deviation, weighted by
        zone volume fraction:

            G_void = Π · [Δφ²_{v,f} · x_f  +  Δφ²_{v,i} · x_i]
        """
        phi_f_loc = self._phi_f_local(xi)
        phi_i_loc = self._phi_i_local(xi)
        phi_v_f = max(0.0, 1.0 - phi_f_loc)
        phi_v_i = max(0.0, 1.0 - phi_i_loc)

        x_f = self.x_f0 * (1.0 - xi)
        x_i = 1.0 - x_f

        cost_f = (phi_v_f - self.phi_v_f0)**2 * x_f
        cost_i = (phi_v_i - self.phi_v_i0)**2 * x_i
        return self.Pi_osm * (cost_f + cost_i)

    def G_inert(self, xi):
        """Inert granule frustration energy.

        Inert granules act as geometric obstacles.  The displacement
        cost grows quadratically with compaction, scaled by the
        inert-to-functional volume ratio:

            G_inert = k · (φ_i / φ_f) · ξ²
        """
        ratio = self.phi_i / max(self.phi_f, 0.01)
        return self.k_frust * ratio * xi**2

    def G_surface(self, xi):
        """Interfacial energy at functional/inert boundary.

        The functional zone boundary has interfacial tension γ_{fi}
        arising from modulus mismatch and differential adhesion.

        As the zone compacts to a smaller, rounder region, its
        perimeter decreases (2D: P ∝ √x_f → P/P₀ = √(1−ξ)):

            G_compact = γ · [√(1−ξ) − 1]      (< 0, favours compaction)

        But trapped inert granules increase the total interface:

            G_mixing  = γ · φ_i · ξ / 2        (> 0, resists)
        """
        xi_s = np.clip(xi, 0, 1.0 - 1e-6)
        G_compact = self.gamma_fi * (np.sqrt(1.0 - xi_s) - 1.0)
        G_mixing = self.gamma_fi * self.phi_i * xi_s * 0.5
        return G_compact + G_mixing

    # ------------------------------------------------------------------
    # Total energy and gradient
    # ------------------------------------------------------------------

    def G_total(self, xi, f_bridge=1.0, maturity=1.0):
        """Total free energy density at compaction ξ (kPa)."""
        return (self.G_cell(xi, f_bridge, maturity)
                + self.G_elastic(xi)
                + self.G_yield(xi)
                + self.G_void(xi)
                + self.G_inert(xi)
                + self.G_surface(xi))

    def dG_dxi(self, xi, f_bridge=1.0, maturity=1.0, dxi=1e-4):
        """Numerical gradient dG/dξ (generalised thermodynamic force)."""
        lo = max(0, xi - dxi)
        hi = min(self.xi_max * 0.999, xi + dxi)
        return ((self.G_total(hi, f_bridge, maturity)
                 - self.G_total(lo, f_bridge, maturity)) / (hi - lo))

    # ------------------------------------------------------------------
    # Landscape evaluation
    # ------------------------------------------------------------------

    def compute_landscape(self, n_pts=300, f_bridge=1.0, maturity=1.0):
        """Evaluate all energy terms on a uniform grid of ξ.

        Returns dict with keys: 'xi', 'G_cell', 'G_elastic', 'G_yield',
            'G_void', 'G_inert', 'G_surface', 'G_total'.
        All values are *normalised* by σ_cell (dimensionless energy).
        """
        xi = np.linspace(0, self.xi_max * 0.98, n_pts)
        s = max(self.sigma_cell, 1e-12)  # normalisation scale
        out = {'xi': xi, 'sigma_cell': s}

        out['G_cell'] = np.array(
            [self.G_cell(x, f_bridge, maturity) / s for x in xi])
        out['G_elastic'] = np.array([self.G_elastic(x) / s for x in xi])
        out['G_yield'] = np.array([self.G_yield(x) / s for x in xi])
        out['G_void'] = np.array([self.G_void(x) / s for x in xi])
        out['G_inert'] = np.array([self.G_inert(x) / s for x in xi])
        out['G_surface'] = np.array([self.G_surface(x) / s for x in xi])
        out['G_total'] = (out['G_cell'] + out['G_elastic'] + out['G_yield']
                          + out['G_void'] + out['G_inert'] + out['G_surface'])
        return out

    # ------------------------------------------------------------------
    # Equilibrium and barrier analysis
    # ------------------------------------------------------------------

    def find_equilibrium(self, f_bridge=1.0, maturity=1.0):
        """Find equilibrium compaction ξ* (energy minimum).

        Returns (xi_star, G_star_normalised).
        """
        s = max(self.sigma_cell, 1e-12)
        res = minimize_scalar(
            lambda x: float(self.G_total(x, f_bridge, maturity)),
            bounds=(1e-6, self.xi_max * 0.98),
            method='bounded',
        )
        return float(res.x), float(res.fun / s)

    def find_barrier(self, f_bridge=1.0, maturity=1.0, n_pts=500):
        """Find activation barrier ΔG* (first local max before minimum).

        Returns (xi_barrier, DeltaG_normalised).
        """
        s = max(self.sigma_cell, 1e-12)
        xi = np.linspace(1e-6, self.xi_max * 0.98, n_pts)
        G = np.array([float(self.G_total(x, f_bridge, maturity)) for x in xi])
        G0 = float(self.G_total(0, f_bridge, maturity))

        for i in range(1, len(G) - 1):
            if G[i] > G[i - 1] and G[i] > G[i + 1]:
                return float(xi[i]), float((G[i] - G0) / s)
        return 0.0, 0.0

    # ------------------------------------------------------------------
    # Kinetics: overdamped dynamics on the landscape
    # ------------------------------------------------------------------

    def solve_kinetics(self, t_total=72.0, dt=0.1):
        """Integrate overdamped dynamics  dξ/dt = −(1/η) dG/dξ.

        Cell maturation and bridge formation evolve in time (same
        kinetics as the mean-field ODE model).

        Returns (t_arr, xi_arr, fb_arr, maturity_arr).
        """
        t_spread = 3.0
        fa_rate = 0.3
        base_br_rate = 0.3
        bridge_form_t = 2.0
        t_br_start = t_spread + 1.0

        n_steps = int(t_total / dt)
        t_arr = np.zeros(n_steps + 1)
        xi_arr = np.zeros(n_steps + 1)
        fb_arr = np.zeros(n_steps + 1)
        mat_arr = np.zeros(n_steps + 1)

        xi = 0.0

        for step in range(1, n_steps + 1):
            t = (step - 1) * dt
            t_eff = max(0.0, t - t_spread)
            maturity = 1.0 - np.exp(-fa_rate * t_eff)
            t_br_eff = max(0.0, t - t_spread - 1.0)

            if maturity < 0.1:
                f_bridge = 0.0
            else:
                f_bridge = 1.0 - np.exp(
                    -base_br_rate * self.br_mod * t_br_eff * maturity)

            t_since = max(0.0, t - t_br_start)
            ramp = min(1.0, t_since / bridge_form_t) if bridge_form_t > 0 \
                else 1.0
            fb_eff = f_bridge * ramp

            force = -self.dG_dxi(xi, fb_eff, maturity)
            dxi = dt * force / self.eta_eff
            xi = np.clip(xi + max(dxi, 0), 0, self.xi_max * 0.98)

            t_arr[step] = step * dt
            xi_arr[step] = xi
            fb_arr[step] = fb_eff
            mat_arr[step] = maturity

        return t_arr, xi_arr, fb_arr, mat_arr

    # ------------------------------------------------------------------
    # Design-rule helpers
    # ------------------------------------------------------------------

    def dimensionless_groups(self, f_bridge=1.0, maturity=1.0):
        """Compute key dimensionless numbers governing the landscape.

        Returns dict with:
            beta  — cell-to-elastic ratio  σ_cell / σ_0
            Ca    — cell-to-yield ratio    σ_cell / σ_y₀
            Phi_r — inert obstruction      φ_i / φ_f
            Psi   — packing proximity      φ_local₀ / φ_RCP
            Gamma — interfacial number     γ / σ_cell
        """
        sigma_cell = self.sigma_cell * f_bridge * maturity
        sigma_y0_est = self.c_yield * self.sigma_0

        return {
            'beta': sigma_cell / max(self.sigma_0, 1e-8),
            'Ca': sigma_cell / max(sigma_y0_est, 1e-8),
            'Phi_r': self.phi_i / max(self.phi_f, 1e-4),
            'Psi': self._phi_f_local(0) / max(self.phi_RCP, 0.1),
            'Gamma': self.gamma_fi / max(sigma_cell, 1e-8),
        }

    def energy_decomposition_at_eq(self, f_bridge=1.0, maturity=1.0):
        """Evaluate each energy term at equilibrium ξ*.

        Returns dict mapping term name → normalised energy at ξ*.
        """
        xi_star, _ = self.find_equilibrium(f_bridge, maturity)
        s = max(self.sigma_cell, 1e-12)
        return {
            'xi_star': xi_star,
            'G_cell': self.G_cell(xi_star, f_bridge, maturity) / s,
            'G_elastic': self.G_elastic(xi_star) / s,
            'G_yield': self.G_yield(xi_star) / s,
            'G_void': self.G_void(xi_star) / s,
            'G_inert': self.G_inert(xi_star) / s,
            'G_surface': self.G_surface(xi_star) / s,
        }
