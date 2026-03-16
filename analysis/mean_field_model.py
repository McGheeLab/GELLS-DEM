"""
Volume-Conserving Two-Zone Mean-Field Model for Granular Scaffold Compaction
============================================================================
V1.9 -- Two-zone model with volume conservation.

Physics:
    - phi_f + phi_i = phi_solid = CONSTANT (granules are incompressible)
    - State variable: x_f = volume fraction of domain occupied by functional zone
    - Compaction shrinks x_f: functional granules pack tighter, void expelled
    - Local void in functional zone: phi_v_f = 1 - phi_f / x_f
    - Local void in inert zone: phi_v_i = 1 - phi_i / (1 - x_f)
    - Global void = 1 - phi_solid = always constant
    - Motor-clutch cell traction (Chan & Odde 2008) drives compaction
    - Hertzian contact resistance as local packing approaches RCP
    - Bridge formation kinetics (Poisson process) ramp cell connectivity
    - Kozeny-Carman permeability per zone, volume-weighted for effective K

Units throughout: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).

Usage:
    # Fit to a saved simulation run:
    python analysis/mean_field_model.py -i results/default

    # Specify output directory:
    python analysis/mean_field_model.py -i results/default -o plots/mean_field

Or import programmatically:
    from analysis.mean_field_model import from_run, CompactionModel, plot_fit
    model, fit_result, data = from_run('results/default')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.optimize import minimize


# ======================================================================
# Motor-Clutch Parameters
# ======================================================================

@dataclass
class MotorClutchParams:
    """Motor-clutch model parameters extracted from simulation Params.

    All values in simulation units: nN, um, kPa, hours.
    """
    n_motors: int = 50                  # motor complexes per cell
    F_motor_stall: float = 0.5          # nN per motor complex
    n_clutches: int = 75                # integrin clutch clusters per cell
    k_clutch: float = 5.0              # nN/um, clutch spring constant
    k_on_clutch: float = 1.0           # 1/s, clutch binding rate
    k_off_clutch: float = 0.1          # 1/s, baseline clutch unbinding rate
    cell_diameter: float = 20.0         # um
    poisson_ratio: float = 0.45         # Poisson's ratio
    F_max_per_cell: float = 50.0        # nN, absolute cap
    fa_maturation_rate: float = 0.3     # 1/h
    t_spread_duration: float = 3.0      # h

    @property
    def F_stall(self):
        """Total motor stall force (nN)."""
        return self.n_motors * self.F_motor_stall

    @property
    def k_opt(self):
        """Optimal clutch ensemble stiffness (nN/um)."""
        return self.n_clutches * self.k_clutch

    @property
    def engagement(self):
        """Steady-state clutch engagement fraction."""
        return self.k_on_clutch / (self.k_on_clutch + self.k_off_clutch)


def motor_clutch_force(E_kPa, mc_params):
    """Steady-state traction force per cell from the motor-clutch model.

    Based on Chan & Odde (2008). For hydrogel substrates (1-100 kPa) we
    are in the rising portion of the stiffness-force curve:

        F = F_stall * beta * engagement

    where beta = k_sub / (k_sub + k_opt) is the motor-clutch engagement ratio.

    Parameters
    ----------
    E_kPa : float
        Young's modulus of substrate in kPa.
    mc_params : MotorClutchParams
        Motor-clutch model parameters.

    Returns
    -------
    float
        Traction force per cell in nN (at full FA maturity).
    """
    a_cell = mc_params.cell_diameter / 2.0
    k_sub = np.pi * E_kPa * a_cell / (1.0 - mc_params.poisson_ratio ** 2)
    beta = k_sub / (k_sub + mc_params.k_opt)
    F_mc = mc_params.F_stall * beta * mc_params.engagement
    return min(F_mc, mc_params.F_max_per_cell)


# ======================================================================
# Compaction Model
# ======================================================================

class CompactionModel:
    """Volume-conserving two-zone mean-field model for granular scaffold compaction.

    Physics:
      - phi_f + phi_i = phi_solid = CONSTANT (granules are incompressible).
      - State variable: x_f = fraction of domain occupied by functional zone
        (functional granules + their local void).
      - Initially uniformly mixed: x_f_0 = phi_f / phi_solid.
      - Compaction shrinks x_f → functional granules pack tighter locally.
      - Void expelled from functional zone → inert zone void increases.
      - Limit: x_f >= phi_f / phi_max (deformable limit > RCP for soft granules).
      - Local void in functional zone: phi_v_f = 1 - phi_f / x_f.
      - Local void in inert zone: phi_v_i = 1 - phi_i / (1 - x_f).
      - Global void = 1 - phi_solid = always constant.

    Parameters
    ----------
    E_modulus : float
        Young's modulus of hydrogel in kPa.
    phi_f0 : float
        Functional phase fraction (constant — volume conserved).
    phi_i0 : float
        Inert phase fraction (constant — volume conserved).
    R_func : float
        Mean functional granule radius in um.
    n_cells_per_granule : int
        Number of cells per functional granule.
    N_func : int
        Number of functional granules.
    domain_volume : float
        Domain volume in um^3 (or area in um^2 for 2D).
    mc_params : MotorClutchParams
        Motor-clutch model parameters.
    bridge_attempt_rate : float
        Bridge formation rate (1/h), Poisson process.
    bridge_formation_time : float
        Time for bridge force to ramp to full (hours).
    phi_RCP : float
        Random close packing fraction for functional zone (default 0.64).
    blockiness_n2 : float
        Superellipsoid exponent n2 for functional granules (2 = sphere).
    """

    def __init__(self, E_modulus, phi_f0, phi_i0, R_func, n_cells_per_granule,
                 N_func, domain_volume, mc_params, bridge_attempt_rate=0.3,
                 bridge_formation_time=2.0, phi_RCP=0.64,
                 bridge_senescence_time=24.0,
                 bridge_lock_force_threshold=20.0,
                 bridge_secondary_rate_mult=3.0,
                 expected_bridge_force=100.0,
                 blockiness_n2=2.5):
        self.E_modulus = E_modulus
        self.phi_f0 = phi_f0       # constant (volume conserved)
        self.phi_i0 = phi_i0       # constant (volume conserved)
        self.phi_solid = phi_f0 + phi_i0  # constant total solid
        self.R_func = R_func
        self.n_cells_per_granule = n_cells_per_granule
        self.N_func = N_func
        self.domain_volume = domain_volume
        self.mc_params = mc_params
        self.bridge_attempt_rate = bridge_attempt_rate
        self.bridge_formation_time = bridge_formation_time
        self.phi_RCP = phi_RCP
        self.blockiness_n2 = blockiness_n2
        self.bridge_senescence_time = bridge_senescence_time
        self.bridge_lock_force_threshold = bridge_lock_force_threshold
        self.bridge_secondary_rate_mult = bridge_secondary_rate_mult
        self.expected_bridge_force = expected_bridge_force

        # Derived quantities
        self.F_cell = motor_clutch_force(E_modulus, mc_params)
        self.n_total_cells = N_func * n_cells_per_granule
        self.n_cell_density = self.n_total_cells / domain_volume

        # Deformable packing limit: soft/blocky granules can pack beyond rigid RCP
        # compliance: 0 for infinitely stiff, approaches 1 for infinitely soft
        E_ref = 10.0  # kPa
        compliance = E_ref / (E_modulus + E_ref)
        shape_boost = 1.0 + 0.15 * max(0.0, blockiness_n2 - 2.0)
        phi_deform_extra = (1.0 - phi_RCP) * 0.5 * compliance * shape_boost
        self.phi_max = min(phi_RCP + phi_deform_extra, 0.99)

        # Minimum x_f: deformable limit (resistance still grows above phi_RCP)
        self.x_f_min = phi_f0 / max(self.phi_max, 0.01)

        # Initial functional zone fraction (uniformly mixed, clamped to feasible)
        self.x_f0 = max(phi_f0 / max(self.phi_solid, 1e-6), self.x_f_min)

        # Lock-in eligibility
        self.lock_in_eligible = self.F_cell >= self.bridge_lock_force_threshold

    def _maturity(self, t):
        """Focal adhesion maturity fraction at time t.

        Exponential approach: maturity(t) = 1 - exp(-fa_maturation_rate * t),
        delayed by spread duration.

        Parameters
        ----------
        t : float
            Time in hours.

        Returns
        -------
        float
            FA maturity fraction in [0, 1].
        """
        t_eff = max(0.0, t - self.mc_params.t_spread_duration)
        return 1.0 - np.exp(-self.mc_params.fa_maturation_rate * t_eff)

    def _bridge_fraction(self, t):
        """Fraction of cells actively bridging at time t.

        Bridge formation is a Poisson process with rate bridge_attempt_rate.
        The fraction saturates as: f_bridge(t) = 1 - exp(-rate * t_effective).

        The effective time accounts for the FA maturity threshold.

        When lock-in is active (F_cell >= bridge_lock_force_threshold), bridges
        persist indefinitely rather than decaying via senescence.  Additionally,
        secondary migration along existing bridges boosts the formation rate
        once initial bridges are established.

        Parameters
        ----------
        t : float
            Time in hours.

        Returns
        -------
        float
            Bridge fraction in [0, 1].
        """
        maturity = self._maturity(t)
        # Bridges can only form after cells have sufficient FA maturity
        # (min_fa_for_bridge ~ 0.3). Approximate effective bridging time:
        if maturity < 0.1:
            return 0.0
        t_bridge_eff = max(0.0, t - self.mc_params.t_spread_duration - 1.0)

        # Base bridge fraction (Poisson formation)
        rate = self.bridge_attempt_rate
        f_base = 1.0 - np.exp(-rate * t_bridge_eff * maturity)

        # Secondary migration boost: once some bridges exist, new cells form
        # bridges faster by migrating along established bridges
        if f_base > 0.05:
            rate_boosted = rate * self.bridge_secondary_rate_mult
            f_boosted = 1.0 - np.exp(-rate_boosted * t_bridge_eff * maturity)
            # Weighted blend: fraction of non-bridging cells that benefit
            f_bridge = f_base + (1.0 - f_base) * (f_boosted - f_base)
        else:
            f_bridge = f_base

        # Lock-in: high-force bridges don't undergo senescence turnover.
        # Without lock-in, effective bridge fraction decays as bridges senesce.
        # With lock-in, bridges accumulate without turnover.
        if not self.lock_in_eligible and self.bridge_senescence_time > 0:
            # Apply senescence turnover: steady-state fraction is reduced
            tau_s = self.bridge_senescence_time
            turnover = tau_s / (tau_s + t_bridge_eff) if t_bridge_eff > 0 else 1.0
            f_bridge *= turnover

        return min(1.0, f_bridge)

    def _bridge_ramp(self, t):
        """Bridge force ramp factor.

        New bridges ramp force linearly over bridge_formation_time.
        For the mean field, this smooths out the force onset.

        Parameters
        ----------
        t : float
            Time in hours.

        Returns
        -------
        float
            Ramp factor in [0, 1].
        """
        if self.bridge_formation_time <= 0:
            return 1.0
        t_bridge_start = self.mc_params.t_spread_duration + 1.0
        t_since = max(0.0, t - t_bridge_start)
        return min(1.0, t_since / self.bridge_formation_time)

    def sigma_cell(self, t):
        """Cell-generated compaction stress at time t.

        sigma_cell = n_cell_density * F_cell * f_bridge(t) * maturity(t) * ramp(t)

        Parameters
        ----------
        t : float
            Time in hours.

        Returns
        -------
        float
            Cell stress in nN/um^2 (= kPa * 1e-3... but we keep in nN/um^2
            since forces are in nN and lengths in um).
        """
        maturity = self._maturity(t)
        f_bridge = self._bridge_fraction(t)
        ramp = self._bridge_ramp(t)
        return self.n_cell_density * self.F_cell * f_bridge * maturity * ramp

    def sigma_resist(self, x_f, sigma_0, alpha):
        """Contact resistance stress in functional zone near local jamming.

        Resistance grows as local packing phi_f/x_f approaches phi_RCP.

        Parameters
        ----------
        x_f : float
            Functional zone volume fraction (state variable).
        sigma_0 : float
            Jamming stress prefactor (nN/um^2).
        alpha : float
            Jamming exponent (1 for Hertzian contacts).

        Returns
        -------
        float
            Resistance stress in nN/um^2.
        """
        phi_f_local = self.phi_f0 / max(x_f, 1e-6)
        ratio = phi_f_local / self.phi_RCP
        if ratio <= 1.0:
            return 0.0
        return sigma_0 * (ratio - 1.0) ** alpha

    def compaction_rate(self, x_f, t, eta_eff, sigma_0, alpha):
        """Compute dx_f/dt (two-zone volume-conserving model).

        dx_f/dt = -x_f * (sigma_cell - sigma_resist) / eta_eff

        Compaction only occurs when cell stress exceeds resistance.

        Parameters
        ----------
        x_f : float
            Current functional zone volume fraction.
        t : float
            Current time in hours.
        eta_eff : float
            Effective granular viscosity (nN*h/um^2).
        sigma_0 : float
            Jamming stress prefactor.
        alpha : float
            Jamming exponent.

        Returns
        -------
        float
            Rate of change of x_f (1/h).
        """
        s_cell = self.sigma_cell(t)
        s_resist = self.sigma_resist(x_f, sigma_0, alpha)
        net_stress = s_cell - s_resist
        if net_stress <= 0:
            return 0.0
        eps_dot = net_stress / eta_eff
        return -x_f * eps_dot

    def solve(self, t_span, eta_eff, sigma_0, alpha, n_points=500):
        """Integrate the two-zone compaction ODE over t_span.

        State: x_f (functional zone volume fraction).
        phi_f and phi_i are constant (volume conserved).

        Parameters
        ----------
        t_span : tuple of (float, float)
            (t_start, t_end) in hours.
        eta_eff : float
            Effective granular viscosity (nN*h/um^2).
        sigma_0 : float
            Jamming stress prefactor.
        alpha : float
            Jamming exponent.
        n_points : int
            Number of evaluation points.

        Returns
        -------
        scipy.integrate.OdeSolution
            Solution object with .t and .y attributes.
            y[0] = x_f trajectory. Derive local voids as:
            phi_v_f = 1 - phi_f / x_f, phi_v_i = 1 - phi_i / (1 - x_f).
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t, y):
            x_f = y[0]
            x_f = np.clip(x_f, self.x_f_min, 1.0 - self.phi_i0)
            return [self.compaction_rate(x_f, t, eta_eff, sigma_0, alpha)]

        sol = solve_ivp(rhs, t_span, [self.x_f0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10,
                        max_step=0.5)
        return sol

    def local_voids(self, x_f):
        """Compute local void fractions from x_f.

        Returns (phi_v_f_local, phi_v_i_local).
        """
        phi_v_f = max(0.0, 1.0 - self.phi_f0 / max(x_f, 1e-6))
        phi_v_i = max(0.0, 1.0 - self.phi_i0 / max(1.0 - x_f, 1e-6))
        return phi_v_f, phi_v_i

    def permeability(self, phi_v, d_grain=None):
        """Kozeny-Carman permeability.

        K = phi_v^3 * d^2 / [180 * (1 - phi_v)^2]

        Parameters
        ----------
        phi_v : float or array
            Void fraction.
        d_grain : float or None
            Grain diameter in um. Defaults to 2 * R_func.

        Returns
        -------
        float or array
            Permeability K in um^2.
        """
        if d_grain is None:
            d_grain = 2.0 * self.R_func
        eps = np.clip(phi_v, 0.01, 0.99)
        return eps**3 * d_grain**2 / (180.0 * (1.0 - eps)**2)

    def darcy_flow(self, phi_v, d_grain=None, delta_P=1.0, mu=1e-3,
                   L=100.0, A=1.0):
        """Darcy volumetric flow rate.

        Q = K * A * delta_P / (mu * L)

        Parameters
        ----------
        phi_v : float or array
            Void fraction.
        d_grain : float or None
            Grain diameter in um. Defaults to 2 * R_func.
        delta_P : float
            Pressure drop in Pa (default 1 Pa).
        mu : float
            Dynamic viscosity in Pa*s (default 1e-3 for water).
        L : float
            Flow path length in um.
        A : float
            Cross-sectional area in um^2.

        Returns
        -------
        float or array
            Flow rate Q in um^3/s.
        """
        K = self.permeability(phi_v, d_grain)
        # K is in um^2. Convert delta_P to nN/um^2: 1 Pa = 1e-3 kPa = 1e-3 nN/um^2
        # Actually: 1 Pa = 1 N/m^2 = 1e-9 nN / (1e-6 m)^2 = 1e-9/1e-12 = 1e3 nN/m^2
        # = 1e3 * 1e-12 nN/um^2 = 1e-9 nN/um^2
        # Simpler: work in SI micron units. K [um^2], dP [Pa], mu [Pa*s], L [um], A [um^2]
        # Q [um^3/s] = K [um^2] * A [um^2] * dP [Pa] / (mu [Pa*s] * L [um])
        #            = K * A * dP / (mu * L)  with K in um^2, L in um
        # But um^2 * um^2 * Pa / (Pa*s * um) = um^3/s. Correct.
        return K * A * delta_P / (mu * L)


# ======================================================================
# Fitting
# ======================================================================

def fit_to_data(t_data, x_f_data, model, bounds=None):
    """Fit compaction model parameters to simulation/experimental data.

    Fits three parameters: eta_eff, sigma_0, alpha.

    Parameters
    ----------
    t_data : array-like
        Time values in hours.
    x_f_data : array-like
        Functional zone fraction trajectory (from DEM clustering analysis
        or experimental measurement). x_f = V_func_zone / V_domain.
    model : CompactionModel
        Compaction model instance (with fixed physical parameters).
    bounds : dict or None
        Optional bounds dict with keys 'eta_eff', 'sigma_0', 'alpha',
        each a (min, max) tuple. Defaults to reasonable ranges.

    Returns
    -------
    dict
        Fitted parameters and diagnostics:
        - 'eta_eff': fitted effective viscosity
        - 'sigma_0': fitted jamming prefactor
        - 'alpha': fitted jamming exponent
        - 'residual': sum of squared residuals
        - 'R2': coefficient of determination
        - 'sol': ODE solution at fitted parameters
    """
    t_data = np.asarray(t_data, dtype=float)
    x_f_data = np.asarray(x_f_data, dtype=float)

    if bounds is None:
        bounds = {
            'eta_eff': (1e-6, 1e6),
            'sigma_0': (1e-10, 1e4),
            'alpha': (0.5, 3.0),
        }

    t_span = (float(t_data[0]), float(t_data[-1]))

    def objective(params_log):
        """Objective in log-space for eta_eff and sigma_0, linear for alpha."""
        log_eta, log_s0, alpha = params_log
        eta_eff = 10.0 ** log_eta
        sigma_0 = 10.0 ** log_s0
        alpha = np.clip(alpha, bounds['alpha'][0], bounds['alpha'][1])

        try:
            sol = model.solve(t_span, eta_eff, sigma_0, alpha,
                              n_points=max(200, len(t_data)))
            if sol.status != 0:
                return 1e20
            # Interpolate to data times
            x_f_model = np.interp(t_data, sol.t, sol.y[0])
            residual = np.sum((x_f_data - x_f_model) ** 2)
            return residual
        except Exception:
            return 1e20

    # Initial guesses in log space
    log_eta_0 = 0.0    # eta_eff ~ 1
    log_s0_0 = -2.0    # sigma_0 ~ 0.01
    alpha_0 = 1.0       # Hertzian

    x0 = [log_eta_0, log_s0_0, alpha_0]
    opt_bounds = [
        (np.log10(bounds['eta_eff'][0]), np.log10(bounds['eta_eff'][1])),
        (np.log10(bounds['sigma_0'][0]), np.log10(bounds['sigma_0'][1])),
        bounds['alpha'],
    ]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=opt_bounds,
                      options={'maxiter': 2000, 'ftol': 1e-12})

    eta_eff = 10.0 ** result.x[0]
    sigma_0 = 10.0 ** result.x[1]
    alpha = result.x[2]

    # Compute final solution
    sol = model.solve(t_span, eta_eff, sigma_0, alpha,
                      n_points=max(500, 2 * len(t_data)))
    x_f_model = np.interp(t_data, sol.t, sol.y[0])

    # R-squared
    ss_res = np.sum((x_f_data - x_f_model) ** 2)
    ss_tot = np.sum((x_f_data - np.mean(x_f_data)) ** 2)
    R2 = 1.0 - ss_res / max(ss_tot, 1e-30)

    return {
        'eta_eff': eta_eff,
        'sigma_0': sigma_0,
        'alpha': alpha,
        'residual': ss_res,
        'R2': R2,
        'sol': sol,
        'optimizer_result': result,
    }


# ======================================================================
# Construction helpers
# ======================================================================

def from_params(p):
    """Construct CompactionModel from a Params-like object.

    Parameters
    ----------
    p : Params or dict
        Simulation parameters. If dict, keys should match Params field names.

    Returns
    -------
    CompactionModel
        Configured compaction model instance.
    """
    # Handle dict or Params object
    def _get(key, default=None):
        if isinstance(p, dict):
            return p.get(key, default)
        return getattr(p, key, default)

    mc = MotorClutchParams(
        n_motors=int(_get('n_motors', 50)),
        F_motor_stall=float(_get('F_motor_stall', 0.5)),
        n_clutches=int(_get('n_clutches', 75)),
        k_clutch=float(_get('k_clutch', 5.0)),
        k_on_clutch=float(_get('k_on_clutch', 1.0)),
        k_off_clutch=float(_get('k_off_clutch', 0.1)),
        cell_diameter=float(_get('cell_diameter', 20.0)),
        poisson_ratio=float(_get('poisson_ratio', 0.45)),
        F_max_per_cell=float(_get('F_max_per_cell', 50.0)),
        fa_maturation_rate=float(_get('fa_maturation_rate', 0.3)),
        t_spread_duration=float(_get('t_spread_duration', 3.0)),
    )

    E_modulus = float(_get('E_modulus', 10.0))
    phi_f_target = float(_get('phi_f_target', 0.25))
    phi_i_target = float(_get('phi_i_target', 0.20))
    R_func = float(_get('R_func_mean', 40.0))
    n_cells = int(_get('n_cells_per_granule', 8))
    Lx = float(_get('Lx', 800.0))
    Ly = float(_get('Ly', 800.0))
    mode = _get('mode', '2D')

    if mode == '3D' or mode == '2D-slice':
        Lz = float(_get('Lz', 800.0))
        domain_volume = Lx * Ly * Lz
        gran_volume = (4.0 / 3.0) * np.pi * R_func ** 3
    else:
        Lz = 0.0
        domain_volume = Lx * Ly
        gran_volume = np.pi * R_func ** 2

    N_func = max(1, int(round(phi_f_target * domain_volume / gran_volume)))

    bridge_attempt_rate = float(_get('bridge_attempt_rate', 0.3))
    bridge_formation_time = float(_get('bridge_formation_time', 2.0))
    bridge_senescence_time = float(_get('bridge_senescence_time', 24.0))
    bridge_lock_force_threshold = float(_get('bridge_lock_force_threshold', 20.0))
    bridge_secondary_rate_mult = float(_get('bridge_secondary_rate_mult', 3.0))
    expected_bridge_force = float(_get('expected_bridge_force', 100.0))

    # Estimate phi_RCP from aspect ratio
    ar_mean = float(_get('aspect_ratio_func_mean', 1.0))
    phi_RCP = min(0.74, 0.64 + 0.08 * (max(1.0, ar_mean) - 1.0))

    blockiness_n2 = float(_get('blockiness_n2_func_mean',
                                _get('blockiness_n2_func', 2.5)))

    return CompactionModel(
        E_modulus=E_modulus,
        phi_f0=phi_f_target,
        phi_i0=phi_i_target,
        R_func=R_func,
        n_cells_per_granule=n_cells,
        N_func=N_func,
        domain_volume=domain_volume,
        mc_params=mc,
        bridge_attempt_rate=bridge_attempt_rate,
        bridge_formation_time=bridge_formation_time,
        phi_RCP=phi_RCP,
        bridge_senescence_time=bridge_senescence_time,
        bridge_lock_force_threshold=bridge_lock_force_threshold,
        bridge_secondary_rate_mult=bridge_secondary_rate_mult,
        expected_bridge_force=expected_bridge_force,
        blockiness_n2=blockiness_n2,
    )


def from_run(run_dir):
    """Load data from a simulation run, construct model, fit, and return results.

    In the two-zone model, phi_f is constant (volume conserved). The DEM
    simulation tracks actual granule positions; x_f (functional zone fraction)
    is estimated from the ratio of functional granule Voronoi volume to
    total domain volume. As a proxy, we use normalized phi_f variations:
    x_f(t) ≈ x_f0 * phi_f(t) / phi_f(0), where phi_f(t) is the coarse-grained
    functional phase fraction (which varies slightly due to field discretisation).

    Parameters
    ----------
    run_dir : str
        Path to simulation output directory or .tar.gz archive.

    Returns
    -------
    tuple of (CompactionModel, dict, dict)
        (model, fit_result, data) where data contains the raw time series.
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)

    if not hist:
        raise ValueError(f"No history data found in {run_dir}")

    # Extract time series
    t_data = np.array([h['time'] for h in hist])
    phi_f_data = np.array([h.get('phi_f_mean', 0.0) for h in hist])
    phi_i_data = np.array([h.get('phi_i_mean', 0.0) for h in hist])
    phi_v_data = np.array([h.get('porosity', h.get('phi_v_mean', 0.0))
                           for h in hist])

    # Use actual initial conditions from data
    model = from_params(p)
    if len(phi_f_data) > 0 and phi_f_data[0] > 0:
        model.phi_f0 = phi_f_data[0]
    if len(phi_i_data) > 0 and phi_i_data[0] > 0:
        model.phi_i0 = phi_i_data[0]
    model.phi_solid = model.phi_f0 + model.phi_i0
    model.x_f0 = model.phi_f0 / max(model.phi_solid, 1e-6)
    model.x_f_min = model.phi_f0 / max(model.phi_max, 0.01)

    # Update N_func from snapshot if available
    if snaps:
        snap0 = snaps[0]
        gtype = snap0.get('gtype', np.array([]))
        if len(gtype) > 0:
            model.N_func = int(np.sum(gtype == 0))
            model.n_total_cells = model.N_func * model.n_cells_per_granule
            model.n_cell_density = model.n_total_cells / model.domain_volume

    # Estimate x_f trajectory from coarse-grained phi_f variations
    # phi_f(t) varies slightly in DEM due to granule rearrangement;
    # use it as proxy for x_f change
    phi_f0_val = phi_f_data[0] if phi_f_data[0] > 0 else 1.0
    x_f_data = model.x_f0 * phi_f_data / phi_f0_val

    # Fit
    fit_result = fit_to_data(t_data, x_f_data, model)

    data = {
        'time': t_data,
        'x_f': x_f_data,
        'phi_f': phi_f_data,
        'phi_i': phi_i_data,
        'phi_v': phi_v_data,
        'hist': hist,
        'params': p,
        'metadata': metadata,
    }

    return model, fit_result, data


# ======================================================================
# Plotting functions
# ======================================================================

def plot_fit(t_data, x_f_data, model, fit_params, ax=None):
    """Overlay DEM data and mean-field model fit.

    Parameters
    ----------
    t_data : array-like
        Time values from data.
    x_f_data : array-like
        Functional zone fraction from data.
    model : CompactionModel
        Compaction model.
    fit_params : dict
        Fitted parameters from fit_to_data().
    ax : matplotlib.axes.Axes or None
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    t_data = np.asarray(t_data)
    x_f_data = np.asarray(x_f_data)

    sol = fit_params['sol']

    ax.plot(t_data, x_f_data, 'ko', ms=4, alpha=0.6, label='DEM data')
    ax.plot(sol.t, sol.y[0], 'C1-', lw=2, label='Mean-field model')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('$x_f$ (functional zone fraction)')
    ax.set_title(r'Compaction Fit: $R^2$ = {:.4f}'.format(fit_params['R2']))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Annotation with fitted parameters
    txt = (f"$\\eta_{{eff}}$ = {fit_params['eta_eff']:.3g}\n"
           f"$\\sigma_0$ = {fit_params['sigma_0']:.3g}\n"
           f"$\\alpha$ = {fit_params['alpha']:.2f}")
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, fontsize=9,
            va='top', ha='right', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    plt.tight_layout()
    return fig


def plot_phase_evolution(model, fit_params, t_span, ax=None):
    """Plot two-zone compaction: x_f, local voids, and permeability vs time.

    Parameters
    ----------
    model : CompactionModel
        Compaction model.
    fit_params : dict
        Fitted parameters from fit_to_data().
    t_span : tuple of (float, float)
        Time range in hours.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    else:
        fig = ax.get_figure()
        axes = [ax, ax.twinx()]

    sol = model.solve(t_span, fit_params['eta_eff'],
                      fit_params['sigma_0'], fit_params['alpha'])

    x_f = sol.y[0]
    phi_v_f = np.maximum(0.0, 1.0 - model.phi_f0 / np.maximum(x_f, 1e-6))
    phi_v_i = np.maximum(0.0, 1.0 - model.phi_i0 / np.maximum(1.0 - x_f, 1e-6))

    # Left: zone fractions and local voids
    axes[0].plot(sol.t, x_f, 'C1-', lw=2, label='$x_f$ (func zone)')
    axes[0].plot(sol.t, 1.0 - x_f, 'C0-', lw=2, label='$x_i$ (inert zone)')
    axes[0].plot(sol.t, phi_v_f, 'C3--', lw=1.5, label=r'$\phi_v^{func}$ (local void)')
    axes[0].plot(sol.t, phi_v_i, 'C9--', lw=1.5, label=r'$\phi_v^{inert}$ (local void)')
    axes[0].axhline(model.phi_f0 / model.phi_max, color='grey', ls=':', lw=1,
                     label=f'$x_f^{{min}}$ (deformable limit)')
    axes[0].axhline(model.phi_f0 / model.phi_RCP, color='grey', ls='--', lw=0.8,
                     alpha=0.5, label=f'$x_f^{{RCP}}$ (rigid limit)')
    axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel('Volume fraction')
    axes[0].set_title('Two-Zone Compaction (volume conserved)')
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc='center right')
    axes[0].grid(True, alpha=0.3)

    # Right: permeability evolution
    d_f = 2.0 * model.R_func
    eps_f = np.clip(phi_v_f, 0.01, 0.99)
    K_f = eps_f ** 3 * d_f ** 2 / (180.0 * (1.0 - eps_f) ** 2)
    axes[1].plot(sol.t, K_f, 'C1-', lw=2, label='$K_{func}$')
    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel('Permeability $K$ (µm²)')
    axes[1].set_yscale('log')
    axes[1].set_title('Functional Zone Permeability')
    axes[1].legend(fontsize=7)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_permeability_evolution(model, fit_params, t_span, d_grain=None,
                                ax=None):
    """Plot Kozeny-Carman permeability and Darcy flow vs time.

    Parameters
    ----------
    model : CompactionModel
        Compaction model.
    fit_params : dict
        Fitted parameters from fit_to_data().
    t_span : tuple of (float, float)
        Time range in hours.
    d_grain : float or None
        Grain diameter in um. Defaults to 2 * R_func.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig = ax.get_figure()
        axes = [ax, ax.twinx()]

    sol = model.solve(t_span, fit_params['eta_eff'],
                      fit_params['sigma_0'], fit_params['alpha'])

    x_f = sol.y[0]
    phi_v_f = np.maximum(0.0, 1.0 - model.phi_f0 / np.maximum(x_f, 1e-6))

    K = model.permeability(phi_v_f, d_grain)
    Q = model.darcy_flow(phi_v_f, d_grain, delta_P=1.0, mu=1e-3,
                         L=2.0 * model.R_func, A=(2.0 * model.R_func) ** 2)

    axes[0].plot(sol.t, K, 'C2-', lw=2)
    axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel(r'Permeability $K$ ($\mu m^2$)')
    axes[0].set_title('Kozeny-Carman Permeability')
    axes[0].set_yscale('log')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sol.t, Q, 'C4-', lw=2)
    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel(r'Darcy flow $Q$ ($\mu m^3$/s)')
    axes[1].set_title('Darcy Flow Rate')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_stress_balance(model, fit_params, t_span, ax=None):
    """Plot cell stress and resistance stress vs time.

    Parameters
    ----------
    model : CompactionModel
        Compaction model.
    fit_params : dict
        Fitted parameters from fit_to_data().
    t_span : tuple of (float, float)
        Time range in hours.
    ax : matplotlib.axes.Axes or None
        Axes to plot on. Created if None.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    sol = model.solve(t_span, fit_params['eta_eff'],
                      fit_params['sigma_0'], fit_params['alpha'])

    x_f = sol.y[0]
    sigma_0 = fit_params['sigma_0']
    alpha = fit_params['alpha']

    s_cell = np.array([model.sigma_cell(t) for t in sol.t])
    s_resist = np.array([model.sigma_resist(x_f[i], sigma_0, alpha)
                          for i, t in enumerate(sol.t)])
    s_net = s_cell - s_resist

    ax.plot(sol.t, s_cell, 'C1-', lw=2, label=r'$\sigma_{cell}$ (traction)')
    ax.plot(sol.t, s_resist, 'C0-', lw=2, label=r'$\sigma_{resist}$ (contact)')
    ax.fill_between(sol.t, s_resist, s_cell,
                    where=s_net > 0, color='C1', alpha=0.15,
                    label='Net compaction stress')
    ax.fill_between(sol.t, s_cell, s_resist,
                    where=s_net <= 0, color='C0', alpha=0.15,
                    label='Arrested (jammed)')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'Stress (nN/$\mu m^2$)')
    ax.set_title('Stress Balance: Cell Traction vs Contact Resistance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def run_all(run_dir, outdir=None):
    """Load simulation data, fit model, produce all plots.

    Parameters
    ----------
    run_dir : str
        Path to simulation output directory or .tar.gz archive.
    outdir : str or None
        Output directory for plots. Defaults to <run_dir>/plots_mean_field.

    Returns
    -------
    tuple of (CompactionModel, dict, dict)
        (model, fit_result, data).
    """
    model, fit_result, data = from_run(run_dir)

    # Resolve output directory
    actual_dir = run_dir[:-7] if run_dir.endswith('.tar.gz') else run_dir
    if outdir is None:
        outdir = str(Path(actual_dir) / 'plots_mean_field')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    t_data = data['time']
    x_f_data = data['x_f']
    t_span = (float(t_data[0]), float(t_data[-1]))

    print("=" * 65)
    print("  Mean-Field Compaction Model (Two-Zone, Volume-Conserving)")
    print("=" * 65)
    print(f"\n  Run directory: {run_dir}")
    print(f"  Output: {outdir}/")
    print(f"\n  Model parameters:")
    print(f"    E_modulus     = {model.E_modulus:.1f} kPa")
    print(f"    phi_f (const) = {model.phi_f0:.3f}")
    print(f"    phi_i (const) = {model.phi_i0:.3f}")
    print(f"    phi_solid     = {model.phi_solid:.3f}")
    print(f"    x_f0          = {model.x_f0:.3f}")
    print(f"    R_func        = {model.R_func:.1f} um")
    print(f"    N_func        = {model.N_func}")
    print(f"    n_cells/gran  = {model.n_cells_per_granule}")
    print(f"    F_cell        = {model.F_cell:.2f} nN")
    print(f"    phi_RCP       = {model.phi_RCP:.3f}")
    print(f"    phi_max       = {model.phi_max:.3f} (deformable limit)")
    print(f"\n  Fitted parameters:")
    print(f"    eta_eff       = {fit_result['eta_eff']:.4g} nN*h/um^2")
    print(f"    sigma_0       = {fit_result['sigma_0']:.4g} nN/um^2")
    print(f"    alpha         = {fit_result['alpha']:.3f}")
    print(f"    R^2           = {fit_result['R2']:.6f}")
    print(f"    Residual      = {fit_result['residual']:.4g}")

    # Bridge force diagnostic
    print(f"\n  Bridge force diagnostic:")
    print(f"    F_cell (motor-clutch) = {model.F_cell:.2f} nN")
    print(f"    Expected per cell     = {model.expected_bridge_force:.1f} nN")
    print(f"    Lock-in threshold     = {model.bridge_lock_force_threshold:.1f} nN")
    print(f"    Lock-in eligible      = {model.lock_in_eligible}")
    ratio = model.F_cell / model.expected_bridge_force if model.expected_bridge_force > 0 else 0.0
    if ratio < 0.25:
        print(f"    WARNING: Motor-clutch force ({model.F_cell:.1f} nN) is only "
              f"{ratio:.0%} of expected ({model.expected_bridge_force:.0f} nN).")
        print(f"    Consider: n_motors={model.mc_params.n_motors}, "
              f"F_motor_stall={model.mc_params.F_motor_stall}, "
              f"F_max_per_cell={model.mc_params.F_max_per_cell}")
    elif ratio < 0.5:
        print(f"    NOTE: Motor-clutch force is {ratio:.0%} of expected. "
              f"May need tuning.")

    # 1. Compaction fit
    print("\n  [1/4] Compaction fit...")
    fig1 = plot_fit(t_data, x_f_data, model, fit_result)
    fig1.savefig(str(Path(outdir) / 'compaction_fit.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. Phase evolution (two-zone)
    print("  [2/4] Two-zone evolution...")
    fig2 = plot_phase_evolution(model, fit_result, t_span)
    # Overlay x_f data on left panel
    ax = fig2.axes[0]
    ax.plot(t_data, x_f_data, 'C1o', ms=3, alpha=0.5, zorder=5, label='$x_f$ data')
    fig2.savefig(str(Path(outdir) / 'phase_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close(fig2)

    # 3. Permeability
    print("  [3/4] Permeability evolution...")
    fig3 = plot_permeability_evolution(model, fit_result, t_span)
    fig3.savefig(str(Path(outdir) / 'permeability_evolution.png'), dpi=150,
                 bbox_inches='tight')
    plt.close(fig3)

    # 4. Stress balance
    print("  [4/4] Stress balance...")
    fig4 = plot_stress_balance(model, fit_result, t_span)
    fig4.savefig(str(Path(outdir) / 'stress_balance.png'), dpi=150, bbox_inches='tight')
    plt.close(fig4)

    # Summary
    print("\n" + "=" * 65)
    print("  MEAN-FIELD MODEL COMPLETE")
    print("=" * 65)
    print(f"  Plots saved to: {outdir}/")
    print(f"  Files: compaction_fit.png, phase_evolution.png,")
    print(f"         permeability_evolution.png, stress_balance.png")

    return model, fit_result, data


# ======================================================================
# CLI
# ======================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Mean-field ODE model for cell-driven granular scaffold compaction. '
                    'Fits to DEM simulation data and produces diagnostic plots.')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to simulation output directory or .tar.gz archive')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory for plots '
                             '(default: <input>/plots_mean_field)')
    args = parser.parse_args()

    try:
        run_all(args.input, outdir=args.outdir)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
