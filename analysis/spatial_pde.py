"""
Spatially-Resolved PDE Model for Granular Scaffold Compaction
==============================================================
V2.5 — 1D radial PDE extending the mean-field ODE with spatial resolution.

Sits between the full DEM (expensive, per-particle) and the mean-field
ODE (cheap, no spatial info).  Captures compaction waves, local jamming
fronts, heterogeneous porosity and permeability profiles.

Physics:
    - 1D grid ξ ∈ [0, 1] (center=0, edge=1), N_x points
    - State: x_f(ξ, t) = functional zone volume fraction per grid point
    - φ_f + φ_i = φ_solid = CONSTANT (volume conserved)
    - Cell traction stress: σ_cell(ξ,t) = n_cell · F_mc · f_bridge · maturity · neighbor_factor(ξ)
    - Contact resistance:   σ_resist(ξ,t) = σ_0 · (φ_f/x_f(ξ)/φ_RCP − 1)^α
    - ∂x_f/∂t = −x_f·(σ_cell − σ_resist)/η_eff + D·∂²x_f/∂ξ²
    - Tissue volume: φ_tissue(ξ,t) logistic growth (V2.1)
    - BCs: zero-flux at ξ=0 (symmetry) and ξ=1 (edge)

Can be initialised from:
    (A) Params only        — SpatialPDE.from_params(p)
    (B) DEM snapshot       — SpatialPDE.from_snapshot(snap, p)
    (C) DEM run (fit)      — SpatialPDE.from_run(run_dir)

Extracted and refactored from analysis/parameter_sweep.py (V1.9) into
a standalone model class with fitting, plotting, and DEM comparison.

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).

Usage:
    python analysis/spatial_pde.py -i results/default
    python analysis/spatial_pde.py --standalone

Or import programmatically:
    from analysis.spatial_pde import SpatialPDE
    model = SpatialPDE.from_params(Params())
    result = model.solve(t_span=(0, 72))
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


# ======================================================================
# Tissue volume model parameters (matches parameter_sweep.py)
# ======================================================================

TISSUE_PARAMS = {
    'k_tissue': 0.05,            # h^-1, tissue formation rate
    'alpha_tissue_fill': 0.6,    # max fraction of void fillable by tissue
    'alpha_tissue_0': 0.1,       # min growth rate fraction (surface cells, no bridges)
    'n_cells_tissue_ref': 10.0,  # reference cell count for normalization
}


# ======================================================================
# Spatial PDE Model
# ======================================================================

class SpatialPDE:
    """1D spatially-resolved compaction model.

    Extends the mean-field CompactionModel with spatial resolution on a
    radial grid.  The state variable x_f(ξ, t) tracks the functional zone
    fraction at each grid point, capturing compaction waves and local
    jamming fronts that the scalar ODE misses.

    Parameters
    ----------
    phi_f : float
        Functional solid fraction (constant, volume conserved).
    phi_i : float
        Inert solid fraction (constant, volume conserved).
    R_func : float
        Mean functional granule radius (µm).
    R_inert : float
        Mean inert granule radius (µm).
    E_func : float
        Functional granule Young's modulus (kPa).
    E_inert : float
        Inert granule Young's modulus (kPa).
    n_cells_per_granule : int
        Cells per functional granule.
    domain_volume : float
        Domain volume (µm³) or area (µm²).
    phi_RCP : float
        Random close packing fraction.
    N_x : int
        Number of radial grid points.
    """

    def __init__(self, phi_f, phi_i, R_func, R_inert, E_func, E_inert,
                 n_cells_per_granule, domain_volume, phi_RCP=0.82,
                 N_x=20, is_3d=False,
                 # Bridge kinetics
                 bridge_attempt_rate=0.3,
                 bridge_formation_time=2.0,
                 bridge_senescence_time=24.0,
                 bridge_lock_force_threshold=20.0,
                 bridge_secondary_rate_mult=3.0,
                 # Cell parameters
                 t_spread_duration=3.0,
                 fa_maturation_rate=0.3,
                 cell_sense_distance=40.0,
                 # Motor-clutch
                 n_motors=50,
                 F_motor_stall=0.5,
                 F_max_per_cell=50.0,
                 n_clutches=75,
                 k_clutch=5.0,
                 k_on_clutch=1.0,
                 k_off_clutch=0.1,
                 poisson_ratio=0.45,
                 cell_diameter=20.0,
                 # Shape
                 blockiness_n2_func=2.5,
                 aspect_ratio_func=1.0,
                 aspect_ratio_inert=1.0,
                 blockiness_n2_inert=2.5,
                 ):
        self.phi_f = phi_f
        self.phi_i = phi_i
        self.phi_solid = phi_f + phi_i
        self.R_func = R_func
        self.R_inert = R_inert
        self.E_func = E_func
        self.E_inert = E_inert
        self.n_cells_per_granule = n_cells_per_granule
        self.domain_volume = domain_volume
        self.is_3d = is_3d
        self.N_x = N_x

        # Shape / packing
        self.aspect_ratio_func = aspect_ratio_func
        self.aspect_ratio_inert = aspect_ratio_inert
        self.blockiness_n2_func = blockiness_n2_func
        self.blockiness_n2_inert = blockiness_n2_inert

        # Compute phi_RCP with shape corrections
        self.phi_RCP = self._compute_phi_RCP() if phi_RCP is None else phi_RCP
        self.phi_max = self._compute_phi_max()

        # Bridge kinetics
        self.bridge_attempt_rate = bridge_attempt_rate
        self.bridge_formation_time = bridge_formation_time
        self.bridge_senescence_time = bridge_senescence_time
        self.bridge_lock_force_threshold = bridge_lock_force_threshold
        self.bridge_secondary_rate_mult = bridge_secondary_rate_mult

        # Cell timing
        self.t_spread = t_spread_duration
        self.fa_rate = fa_maturation_rate
        self.cell_sense_distance = cell_sense_distance

        # Motor-clutch
        self.n_motors = n_motors
        self.F_motor_stall = F_motor_stall
        self.F_max_per_cell = F_max_per_cell
        self.n_clutches = n_clutches
        self.k_clutch = k_clutch
        self.k_on_clutch = k_on_clutch
        self.k_off_clutch = k_off_clutch
        self.poisson_ratio = poisson_ratio
        self.cell_diameter = cell_diameter

        # Derived: motor-clutch force
        self.F_cell = self._motor_clutch_force()

        # Derived: number of functional granules and cell density
        if is_3d:
            gran_vol = (4.0 / 3.0) * np.pi * R_func ** 3
        else:
            gran_vol = np.pi * R_func ** 2
        self.N_func = max(1, int(round(phi_f * domain_volume / gran_vol)))
        self.n_total_cells = self.N_func * n_cells_per_granule
        self.n_cell_density = self.n_total_cells / domain_volume

        # Initial condition
        self.x_f_min = phi_f / max(self.phi_max, 0.01)
        self.x_f0 = max(phi_f / max(self.phi_solid, 1e-6), self.x_f_min)

        # Lock-in eligibility
        self.lock_in_eligible = self.F_cell >= bridge_lock_force_threshold

        # Grid
        self.xi = np.linspace(0, 1, N_x)
        self.dxi = self.xi[1] - self.xi[0] if N_x > 1 else 1.0
        # Neighbor-count modifier: full neighbors at center, fewer at edge
        self.neighbor_factor = 0.5 * (1.0 + np.cos(np.pi * self.xi))

    # ------------------------------------------------------------------
    # Physics helpers
    # ------------------------------------------------------------------

    def _motor_clutch_force(self):
        """Steady-state traction per cell (nN) from motor-clutch model."""
        F_stall = self.n_motors * self.F_motor_stall
        k_opt = self.n_clutches * self.k_clutch
        engagement = self.k_on_clutch / (self.k_on_clutch + self.k_off_clutch)
        a_cell = self.cell_diameter / 2.0
        k_sub = np.pi * self.E_func * a_cell / (1.0 - self.poisson_ratio ** 2)
        beta = k_sub / (k_sub + k_opt)
        return min(F_stall * beta * engagement, self.F_max_per_cell)

    def _compute_phi_RCP(self):
        """Estimate random close packing fraction with shape corrections."""
        base = 0.82 if not self.is_3d else 0.64
        ar = self.aspect_ratio_func
        n2 = self.blockiness_n2_func
        d = ar - 1.0
        shape = 0.04 * d - 0.015 * d * d + 0.015 * (n2 - 2.0)
        return min(0.95, base + shape)

    def _compute_phi_max(self):
        """Deformable packing limit (soft granules pack beyond rigid RCP)."""
        E_ref = 10.0
        compliance = E_ref / (self.E_func + E_ref)
        shape_boost = 1.0 + 0.15 * max(0.0, self.blockiness_n2_func - 2.0)
        extra = (1.0 - self.phi_RCP) * 0.5 * compliance * shape_boost
        return min(self.phi_RCP + extra, 0.99)

    def _effective_contact_modulus(self):
        """Weighted-average Hertzian reduced modulus across contact types."""
        nu = self.poisson_ratio
        denom = 1.0 - nu ** 2
        E_ff = self.E_func / (2.0 * denom)
        E_ii = self.E_inert / (2.0 * denom)
        E_fi = (self.E_func * self.E_inert /
                (denom * (self.E_func + self.E_inert + 1e-30)))

        phi_tot = max(self.phi_f + self.phi_i, 1e-30)
        wf = self.phi_f / phi_tot
        wi = self.phi_i / phi_tot
        return wf ** 2 * E_ff + wi ** 2 * E_ii + 2.0 * wf * wi * E_fi

    def _bridge_rate_modifier(self):
        """Fraction of cells whose filopodia can reach a neighbour."""
        ratio = self.phi_RCP / max(self.phi_solid, 0.01)
        mean_gap = self.R_func * (np.sqrt(max(ratio, 1.0)) - 1.0)
        return min(1.0, self.cell_sense_distance / max(mean_gap + 1.0, 1.0))

    def _maturity(self, t):
        """FA maturity at time t."""
        t_eff = max(0.0, t - self.t_spread)
        return 1.0 - np.exp(-self.fa_rate * t_eff)

    def _bridge_fraction(self, t):
        """Fraction of cells actively bridging at time t."""
        maturity = self._maturity(t)
        if maturity < 0.1:
            return 0.0

        t_br_eff = max(0.0, t - self.t_spread - 1.0)
        br_mod = self._bridge_rate_modifier()
        rate = self.bridge_attempt_rate * br_mod
        f_base = 1.0 - np.exp(-rate * t_br_eff * maturity)

        if self.bridge_secondary_rate_mult > 1.0 and f_base > 0.05:
            rate_boosted = rate * self.bridge_secondary_rate_mult
            f_boosted = 1.0 - np.exp(-rate_boosted * t_br_eff * maturity)
            f_bridge = f_base + (1.0 - f_base) * (f_boosted - f_base)
        else:
            f_bridge = f_base

        if not self.lock_in_eligible and self.bridge_senescence_time > 0:
            if t_br_eff > 0:
                tau_s = self.bridge_senescence_time
                turnover = tau_s / (tau_s + t_br_eff)
                f_bridge *= turnover

        return min(1.0, f_bridge)

    def _bridge_ramp(self, t):
        """Bridge force ramp factor."""
        if self.bridge_formation_time <= 0:
            return 1.0
        t_bridge_start = self.t_spread + 1.0
        t_since = max(0.0, t - t_bridge_start)
        return min(1.0, t_since / self.bridge_formation_time)

    # ------------------------------------------------------------------
    # PDE solver
    # ------------------------------------------------------------------

    def solve(self, t_span=(0, 72), dt=0.5, eta_eff=None, sigma_0=None,
              alpha=1.0, D_eff=None, x_f_init=None,
              record_every=None) -> dict:
        """Integrate the 1D radial PDE.

        Parameters
        ----------
        t_span : tuple (t_start, t_end)
            Time range in hours.
        dt : float
            Timestep (hours).
        eta_eff : float or None
            Effective viscosity. If None, estimated from material properties.
        sigma_0 : float or None
            Jamming stress prefactor. If None, estimated from E_eff.
        alpha : float
            Jamming exponent (1.0 = Hertzian).
        D_eff : float or None
            Diffusion coefficient. If None, computed from material properties.
        x_f_init : ndarray (N_x,) or None
            Initial x_f profile. If None, uniform at x_f0.
        record_every : float or None
            Record interval (hours). If None, every dt.

        Returns
        -------
        dict
            'time': (n_records,) array
            'x_f': (n_records, N_x) array — functional zone fraction profiles
            'phi_v_f': (n_records, N_x) — local void in functional zone
            'phi_v_i': (n_records, N_x) — local void in inert zone
            'phi_tissue': (n_records, N_x) — tissue volume fraction
            'K_f': (n_records, N_x) — functional zone permeability
            'sigma_cell': (n_records,) — cell stress (domain averaged)
            'sigma_resist': (n_records, N_x) — resistance stress
            'x_f_avg': (n_records,) — domain-averaged x_f
            'compaction_ratio': (n_records,) — domain-averaged compaction
        """
        N_x = self.N_x
        xi = self.xi
        dxi = self.dxi

        # Material-property estimates for free parameters
        E_eff = self._effective_contact_modulus()
        if eta_eff is None:
            eta_eff = 0.5 * (E_eff / 5.0) ** 0.3 * (self.R_func / 40.0) ** 1.5
            eta_eff = max(0.01, min(100.0, eta_eff))
        if sigma_0 is None:
            sigma_0 = max(1e-6, 0.005 * E_eff)
        if D_eff is None:
            D_base = 0.01
            D_eff = D_base * (self.R_func / 40.0) ** 2 * sigma_0 / max(eta_eff, 1e-6)
            # CFL stability
            D_max = 0.45 * dxi ** 2 / dt if dxi > 0 else 1.0
            D_eff = min(D_eff, D_max)

        # Initial condition
        if x_f_init is not None:
            x_f = np.asarray(x_f_init, dtype=np.float64).copy()
        else:
            x_f = np.full(N_x, self.x_f0)

        phi_tissue = np.zeros(N_x)
        n_cells_tissue_norm = min(1.0, self.n_cells_per_granule /
                                  TISSUE_PARAMS['n_cells_tissue_ref'])

        # Integration
        t_start, t_end = t_span
        n_steps = int((t_end - t_start) / dt)
        if record_every is None:
            record_every = dt

        records = {
            'time': [], 'x_f': [], 'phi_v_f': [], 'phi_v_i': [],
            'phi_tissue': [], 'K_f': [],
            'sigma_cell': [], 'sigma_resist': [],
            'x_f_avg': [], 'compaction_ratio': [],
        }
        next_record = t_start

        for step_i in range(n_steps + 1):
            t = t_start + step_i * dt

            # Record
            if t >= next_record - 1e-10:
                phi_v_f = np.maximum(0.0, 1.0 - self.phi_f / np.maximum(x_f, 1e-6))
                phi_v_i = np.maximum(0.0, 1.0 - self.phi_i /
                                     np.maximum(1.0 - x_f, 1e-6))
                d_f = 2.0 * self.R_func
                eps_f = np.clip(phi_v_f, 0.01, 0.99)
                K_f = eps_f ** 3 * d_f ** 2 / (180.0 * (1.0 - eps_f) ** 2)

                records['time'].append(t)
                records['x_f'].append(x_f.copy())
                records['phi_v_f'].append(phi_v_f.copy())
                records['phi_v_i'].append(phi_v_i.copy())
                records['phi_tissue'].append(phi_tissue.copy())
                records['K_f'].append(K_f.copy())
                records['x_f_avg'].append(float(np.mean(x_f)))
                records['compaction_ratio'].append(
                    float(-np.mean(x_f - self.x_f0) / max(self.x_f0, 1e-6)))

                # Stresses
                maturity = self._maturity(t)
                f_bridge = self._bridge_fraction(t)
                ramp = self._bridge_ramp(t)
                s_cell_scalar = (self.n_cell_density * self.F_cell *
                                 f_bridge * maturity * ramp)
                records['sigma_cell'].append(s_cell_scalar)

                phi_f_local = self.phi_f / np.maximum(x_f, 1e-6)
                ratio = phi_f_local / self.phi_RCP
                s_resist = np.where(ratio > 1.0, sigma_0 * (ratio - 1.0) ** alpha, 0.0)
                records['sigma_resist'].append(s_resist.copy())

                next_record += record_every

            if step_i >= n_steps:
                break

            # --- PDE step ---
            maturity = self._maturity(t)
            f_bridge = self._bridge_fraction(t)
            ramp = self._bridge_ramp(t)

            # Spatially-varying cell stress
            s_cell = (self.n_cell_density * self.F_cell * f_bridge *
                      maturity * ramp * self.neighbor_factor)

            # Resistance stress per grid point
            phi_f_local = self.phi_f / np.maximum(x_f, 1e-6)
            ratio = phi_f_local / self.phi_RCP
            s_resist = np.where(ratio > 1.0, sigma_0 * (ratio - 1.0) ** alpha, 0.0)

            # Net compaction
            net = np.maximum(0.0, s_cell - s_resist)
            dx_f_compact = -x_f * net / eta_eff

            # Diffusion (Laplacian with zero-flux BCs)
            laplacian = np.zeros(N_x)
            if N_x > 2:
                laplacian[1:-1] = (x_f[2:] - 2.0 * x_f[1:-1] + x_f[:-2]) / (dxi ** 2)
                laplacian[0] = 2.0 * (x_f[1] - x_f[0]) / (dxi ** 2)
                laplacian[-1] = 2.0 * (x_f[-2] - x_f[-1]) / (dxi ** 2)

            dx_f_diffuse = D_eff * laplacian

            # Update x_f
            x_f += dt * (dx_f_compact + dx_f_diffuse)
            x_f = np.maximum(x_f, self.x_f_min)
            x_f = np.minimum(x_f, 1.0 - self.phi_i)

            # --- Tissue volume evolution ---
            phi_void_func = np.maximum(0.0, 1.0 - self.phi_f / np.maximum(x_f, 1e-6))
            phi_tissue_max = TISSUE_PARAMS['alpha_tissue_fill'] * phi_void_func

            f_bridge_spatial = f_bridge * self.neighbor_factor
            growth_driver = np.maximum(TISSUE_PARAMS['alpha_tissue_0'],
                                       f_bridge_spatial)

            d_tissue = (TISSUE_PARAMS['k_tissue'] *
                        n_cells_tissue_norm * maturity * growth_driver *
                        np.maximum(0.0, phi_tissue_max - phi_tissue))
            phi_tissue += dt * d_tissue
            phi_tissue = np.clip(phi_tissue, 0.0, phi_tissue_max)

        # Convert lists to arrays
        for key in records:
            records[key] = np.array(records[key])

        return records

    # ------------------------------------------------------------------
    # Spatial profile accessors
    # ------------------------------------------------------------------

    def spatial_profile(self, result, t_target) -> dict:
        """Extract spatial profiles at a specific time from solve() output.

        Parameters
        ----------
        result : dict
            Output from solve().
        t_target : float
            Target time (nearest recorded time is used).

        Returns
        -------
        dict with xi, x_f, phi_v_f, phi_v_i, phi_tissue, K_f at that time.
        """
        idx = int(np.argmin(np.abs(result['time'] - t_target)))
        return {
            'xi': self.xi,
            'time': float(result['time'][idx]),
            'x_f': result['x_f'][idx],
            'phi_v_f': result['phi_v_f'][idx],
            'phi_v_i': result['phi_v_i'][idx],
            'phi_tissue': result['phi_tissue'][idx],
            'K_f': result['K_f'][idx],
        }

    def descriptors(self, result, t_target=None) -> dict:
        """Compute tissue architecture descriptors from spatial state.

        Parameters
        ----------
        result : dict
            Output from solve().
        t_target : float or None
            Time to evaluate. If None, uses final time.

        Returns
        -------
        dict of descriptor values.
        """
        if t_target is None:
            idx = -1
        else:
            idx = int(np.argmin(np.abs(result['time'] - t_target)))

        x_f = result['x_f'][idx]
        phi_tissue = result['phi_tissue'][idx]
        x_f_avg = float(np.mean(x_f))

        phi_v_f = np.maximum(0.0, 1.0 - self.phi_f / np.maximum(x_f, 1e-6))
        phi_v_i = np.maximum(0.0, 1.0 - self.phi_i /
                             np.maximum(1.0 - x_f, 1e-6))

        phi_tissue_avg = float(np.mean(phi_tissue))
        phi_tissue_global = x_f_avg * phi_tissue_avg

        BV_TV_eff = self.phi_f + phi_tissue_global
        porosity_eff = max(0.0, 1.0 - BV_TV_eff)

        # Permeability
        d_f = 2.0 * self.R_func
        d_i = 2.0 * self.R_inert
        eps_f = max(0.01, min(0.99, float(np.mean(phi_v_f))))
        eps_i = max(0.01, min(0.99, float(np.mean(phi_v_i))))
        K_f = eps_f ** 3 * d_f ** 2 / (180.0 * (1.0 - eps_f) ** 2)
        K_i = eps_i ** 3 * d_i ** 2 / (180.0 * (1.0 - eps_i) ** 2)
        K_eff = x_f_avg * K_f + (1.0 - x_f_avg) * K_i

        # Series permeability (radial flow)
        eps_f_spatial = np.clip(phi_v_f, 0.01, 0.99)
        K_f_spatial = eps_f_spatial ** 3 * d_f ** 2 / (
            180.0 * (1.0 - eps_f_spatial) ** 2)
        K_f_series = float(self.N_x / np.sum(1.0 / np.maximum(K_f_spatial, 1e-12)))

        # Pore sizes
        r_pore_f = d_f * eps_f / (3.0 * max(1.0 - eps_f, 0.01))
        r_pore_i = d_i * eps_i / (3.0 * max(1.0 - eps_i, 0.01))

        # Spatial heterogeneity
        x_f_std = float(np.std(x_f))
        x_f_gradient = float(x_f[-1] - x_f[0])  # edge − center

        return {
            'BV_TV': self.phi_f + self.phi_i,
            'BV_TV_eff': BV_TV_eff,
            'porosity': 1.0 - self.phi_f - self.phi_i,
            'porosity_eff': porosity_eff,
            'phi_f': self.phi_f,
            'phi_i': self.phi_i,
            'x_f_avg': x_f_avg,
            'compaction_ratio': float(result['compaction_ratio'][idx]),
            'K_f': K_f,
            'K_i': K_i,
            'K_eff': K_eff,
            'K_f_series': K_f_series,
            'r_pore_f': r_pore_f,
            'r_pore_i': r_pore_i,
            'phi_tissue_global': phi_tissue_global,
            'x_f_std': x_f_std,
            'x_f_gradient': x_f_gradient,
        }

    # ------------------------------------------------------------------
    # Fitting to DEM data
    # ------------------------------------------------------------------

    def fit_to_data(self, t_data, x_f_profiles, bounds=None):
        """Fit η_eff, σ_0, α to spatially-resolved DEM data.

        Parameters
        ----------
        t_data : array (n_times,)
            Time values in hours.
        x_f_profiles : array (n_times, N_x) or (n_times,) for domain-averaged
            Functional zone fraction data (from DEM binning or scalar).
        bounds : dict or None
            Bounds for {'eta_eff': (lo, hi), 'sigma_0': (lo, hi), 'alpha': (lo, hi)}.

        Returns
        -------
        dict with eta_eff, sigma_0, alpha, R2, residual, solution.
        """
        t_data = np.asarray(t_data, dtype=float)
        x_f_profiles = np.asarray(x_f_profiles, dtype=float)
        is_spatial = x_f_profiles.ndim == 2

        if bounds is None:
            bounds = {
                'eta_eff': (1e-6, 1e6),
                'sigma_0': (1e-10, 1e4),
                'alpha': (0.5, 3.0),
            }

        t_span = (float(t_data[0]), float(t_data[-1]))
        record_every = float(np.median(np.diff(t_data))) if len(t_data) > 1 else 1.0

        def objective(params_log):
            log_eta, log_s0, alpha_val = params_log
            eta = 10.0 ** log_eta
            s0 = 10.0 ** log_s0
            alpha_val = np.clip(alpha_val, bounds['alpha'][0], bounds['alpha'][1])

            try:
                result = self.solve(t_span, eta_eff=eta, sigma_0=s0,
                                    alpha=alpha_val, record_every=record_every)
                if is_spatial:
                    # Interpolate model to data times, compare spatial profiles
                    residual = 0.0
                    for i, td in enumerate(t_data):
                        idx = np.argmin(np.abs(result['time'] - td))
                        residual += np.sum((x_f_profiles[i] - result['x_f'][idx]) ** 2)
                else:
                    # Compare domain-averaged x_f
                    x_f_model = np.interp(t_data, result['time'], result['x_f_avg'])
                    residual = np.sum((x_f_profiles - x_f_model) ** 2)
                return residual
            except Exception:
                return 1e20

        x0 = [0.0, -2.0, 1.0]
        opt_bounds = [
            (np.log10(bounds['eta_eff'][0]), np.log10(bounds['eta_eff'][1])),
            (np.log10(bounds['sigma_0'][0]), np.log10(bounds['sigma_0'][1])),
            bounds['alpha'],
        ]

        result_opt = minimize(objective, x0, method='L-BFGS-B',
                              bounds=opt_bounds,
                              options={'maxiter': 2000, 'ftol': 1e-12})

        eta_eff = 10.0 ** result_opt.x[0]
        sigma_0 = 10.0 ** result_opt.x[1]
        alpha = result_opt.x[2]

        # Final solution
        sol = self.solve(t_span, eta_eff=eta_eff, sigma_0=sigma_0,
                         alpha=alpha, record_every=record_every)

        # R²
        if is_spatial:
            ss_res = 0.0
            ss_tot = 0.0
            mean_val = np.mean(x_f_profiles)
            for i, td in enumerate(t_data):
                idx = np.argmin(np.abs(sol['time'] - td))
                ss_res += np.sum((x_f_profiles[i] - sol['x_f'][idx]) ** 2)
                ss_tot += np.sum((x_f_profiles[i] - mean_val) ** 2)
        else:
            x_f_model = np.interp(t_data, sol['time'], sol['x_f_avg'])
            ss_res = np.sum((x_f_profiles - x_f_model) ** 2)
            ss_tot = np.sum((x_f_profiles - np.mean(x_f_profiles)) ** 2)

        R2 = 1.0 - ss_res / max(ss_tot, 1e-30)

        return {
            'eta_eff': eta_eff,
            'sigma_0': sigma_0,
            'alpha': alpha,
            'R2': R2,
            'residual': ss_res,
            'solution': sol,
        }

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_params(cls, p):
        """Construct from simulation Params.

        Parameters
        ----------
        p : Params or dict
            Simulation parameters.

        Returns
        -------
        SpatialPDE
        """
        def _get(key, default=None):
            if isinstance(p, dict):
                return p.get(key, default)
            return getattr(p, key, default)

        mode = _get('mode', '2D')
        is_3d = mode == '3D'
        Lx = float(_get('Lx', 800.0))
        Ly = float(_get('Ly', 800.0))
        R_func = float(_get('R_func_mean', 40.0))

        if is_3d:
            Lz = float(_get('Lz', 800.0))
            domain_vol = Lx * Ly * Lz
        else:
            domain_vol = Lx * Ly

        phi_f = float(_get('phi_f_target', 0.25))
        phi_i = float(_get('phi_i_target', 0.20))
        if float(_get('phi_solid_target', 0.0)) > 0:
            phi_s = float(_get('phi_solid_target'))
            fr = float(_get('func_ratio', 0.5))
            phi_f = phi_s * fr
            phi_i = phi_s * (1.0 - fr)

        return cls(
            phi_f=phi_f,
            phi_i=phi_i,
            R_func=R_func,
            R_inert=float(_get('R_inert_mean', 60.0)),
            E_func=float(_get('E_modulus', 10.0)),
            E_inert=float(_get('E_modulus', 10.0)),
            n_cells_per_granule=int(_get('n_cells_per_granule', 8)),
            domain_volume=domain_vol,
            is_3d=is_3d,
            N_x=20,
            bridge_attempt_rate=float(_get('bridge_attempt_rate', 0.3)),
            bridge_formation_time=float(_get('bridge_formation_time', 2.0)),
            bridge_senescence_time=float(_get('bridge_senescence_time', 24.0)),
            bridge_lock_force_threshold=float(_get('bridge_lock_force_threshold', 20.0)),
            bridge_secondary_rate_mult=float(_get('bridge_secondary_rate_mult', 3.0)),
            t_spread_duration=float(_get('t_spread_duration', 3.0)),
            fa_maturation_rate=float(_get('fa_maturation_rate', 0.3)),
            cell_sense_distance=float(_get('cell_sense_distance', 40.0)),
            n_motors=int(_get('n_motors', 50)),
            F_motor_stall=float(_get('F_motor_stall', 0.5)),
            F_max_per_cell=float(_get('F_max_per_cell', 50.0)),
            n_clutches=int(_get('n_clutches', 75)),
            k_clutch=float(_get('k_clutch', 5.0)),
            k_on_clutch=float(_get('k_on_clutch', 1.0)),
            k_off_clutch=float(_get('k_off_clutch', 0.1)),
            poisson_ratio=float(_get('poisson_ratio', 0.45)),
            cell_diameter=float(_get('cell_diameter', 20.0)),
            blockiness_n2_func=float(_get('blockiness_n2_func_mean', 2.5)),
            aspect_ratio_func=float(_get('aspect_ratio_func_mean', 1.0)),
            aspect_ratio_inert=float(_get('aspect_ratio_inert_mean', 1.0)),
            blockiness_n2_inert=float(_get('blockiness_n2_inert_mean', 2.5)),
        )

    @classmethod
    def from_snapshot(cls, snap, p):
        """Initialize spatial profile from DEM snapshot.

        Bins granule positions into radial shells to get x_f(ξ, t=0)
        from the actual packing geometry, not uniform.

        Parameters
        ----------
        snap : dict
            DEM snapshot.
        p : Params
            Simulation parameters.

        Returns
        -------
        tuple of (SpatialPDE, ndarray)
            (model, x_f_init) where x_f_init is the binned profile.
        """
        model = cls.from_params(p)

        x = np.asarray(snap['x'])
        y = np.asarray(snap['y'])
        gtype = np.asarray(snap['gtype'])
        r = np.asarray(snap['r'])
        N = len(x)

        Lx = float(getattr(p, 'Lx', 800.0))
        Ly = float(getattr(p, 'Ly', 800.0))
        cx, cy = Lx / 2.0, Ly / 2.0

        # Radial distance from center, normalized to [0, 1]
        dx = x - cx
        dy = y - cy
        r_dist = np.sqrt(dx ** 2 + dy ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        xi_gran = np.clip(r_dist / r_max, 0.0, 1.0)

        # Bin into radial shells
        N_x = model.N_x
        is_3d = getattr(p, 'mode', '2D') == '3D'
        if is_3d:
            gran_vol = (4.0 / 3.0) * np.pi * r ** 3
        else:
            gran_vol = np.pi * r ** 2

        edges = np.linspace(0, 1, N_x + 1)
        x_f_init = np.full(N_x, model.x_f0)

        for b in range(N_x):
            mask = (xi_gran >= edges[b]) & (xi_gran < edges[b + 1])
            if not np.any(mask):
                continue
            func_vol = np.sum(gran_vol[mask & (gtype == 0)])
            total_vol = np.sum(gran_vol[mask])
            if total_vol > 0:
                # x_f = fraction of this shell occupied by functional granules
                phi_local = total_vol / (model.domain_volume / N_x)
                func_frac = func_vol / max(total_vol, 1e-6)
                x_f_init[b] = max(model.x_f_min,
                                  min(func_frac, 1.0 - model.phi_i))

        return model, x_f_init

    @classmethod
    def from_run(cls, run_dir):
        """Load DEM run, construct model, fit to data, return results.

        Parameters
        ----------
        run_dir : str
            Path to simulation output directory or .tar.gz archive.

        Returns
        -------
        tuple of (SpatialPDE, dict, dict)
            (model, fit_result, data)
        """
        from new_dem_0 import load_run

        hist, snaps, p, metadata = load_run(run_dir)
        if not hist:
            raise ValueError(f"No history data in {run_dir}")

        model = cls.from_params(p)

        # Extract time series
        t_data = np.array([h['time'] for h in hist])
        phi_f_data = np.array([h.get('phi_f_mean', 0.0) for h in hist])

        # Map to x_f (domain-averaged proxy)
        if len(phi_f_data) > 0 and phi_f_data[0] > 0:
            model.phi_f = phi_f_data[0]
            model.phi_solid = model.phi_f + model.phi_i
            model.x_f0 = model.phi_f / max(model.phi_solid, 1e-6)
            model.x_f_min = model.phi_f / max(model.phi_max, 0.01)

        phi_f0_val = phi_f_data[0] if phi_f_data[0] > 0 else 1.0
        x_f_data = model.x_f0 * phi_f_data / phi_f0_val

        # Fit (domain-averaged)
        fit_result = model.fit_to_data(t_data, x_f_data)

        data = {
            'time': t_data,
            'x_f': x_f_data,
            'phi_f': phi_f_data,
            'hist': hist,
            'params': p,
            'metadata': metadata,
        }

        return model, fit_result, data


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Spatial PDE mesoscale model for granular scaffold compaction.')
    parser.add_argument('-i', '--input', default=None,
                        help='Path to DEM output directory (optional)')
    parser.add_argument('--standalone', action='store_true',
                        help='Run standalone from default Params')
    parser.add_argument('--t-total', type=float, default=72.0)
    parser.add_argument('-o', '--outdir', default=None)
    args = parser.parse_args()

    from new_dem_0 import Params

    if args.input and not args.standalone:
        print(f"Loading DEM output from {args.input}...")
        model, fit_result, data = SpatialPDE.from_run(args.input)
        outdir = args.outdir or str(Path(args.input) / 'plots_spatial_pde')

        print(f"\n{'='*60}")
        print(f"  Spatial PDE Model — Fit to DEM")
        print(f"{'='*60}")
        print(f"  η_eff  = {fit_result['eta_eff']:.4g}")
        print(f"  σ_0    = {fit_result['sigma_0']:.4g}")
        print(f"  α      = {fit_result['alpha']:.3f}")
        print(f"  R²     = {fit_result['R2']:.6f}")

        sol = fit_result['solution']
    else:
        print("Running standalone (default Params)...")
        p = Params()
        model = SpatialPDE.from_params(p)
        outdir = args.outdir or 'results/spatial_pde_standalone'

        sol = model.solve(t_span=(0, args.t_total), record_every=2.0)

    Path(outdir).mkdir(parents=True, exist_ok=True)

    # Print final profile
    print(f"\n  Grid: N_x = {model.N_x}")
    print(f"  φ_f = {model.phi_f:.3f}, φ_i = {model.phi_i:.3f}")
    print(f"  F_cell = {model.F_cell:.2f} nN")
    print(f"  φ_RCP = {model.phi_RCP:.3f}, φ_max = {model.phi_max:.3f}")

    desc = model.descriptors(sol)
    print(f"\n  Final descriptors:")
    print(f"    x_f_avg = {desc['x_f_avg']:.4f}")
    print(f"    compaction = {desc['compaction_ratio']:.4f}")
    print(f"    K_eff = {desc['K_eff']:.2f} µm²")
    print(f"    BV/TV_eff = {desc['BV_TV_eff']:.4f}")
    print(f"    x_f gradient = {desc['x_f_gradient']:.4f} (edge−center)")

    # Save profiles
    import json
    profile_data = {
        'time': sol['time'].tolist(),
        'xi': model.xi.tolist(),
        'x_f_avg': sol['x_f_avg'].tolist(),
        'compaction_ratio': sol['compaction_ratio'].tolist(),
    }
    with open(str(Path(outdir) / 'spatial_profiles.json'), 'w') as f:
        json.dump(profile_data, f, indent=2)
    print(f"\n  Profiles saved to {outdir}/spatial_profiles.json")
