"""
Mean-Field ODE Model for Cell-Driven Granular Scaffold Compaction
=================================================================
V1.6 -- Fits to DEM simulation data or experimental phase evolution data.

Physics:
    - Motor-clutch cell traction (Chan & Odde 2008) drives compaction
    - Hertzian contact resistance near jamming opposes compaction
    - Bridge formation kinetics (Poisson process) ramp cell connectivity
    - Overdamped: compaction rate = (cell stress - resist stress) / viscosity
    - Kozeny-Carman permeability tracks transport evolution

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
    """Mean-field ODE model for cell-driven granular scaffold compaction.

    The model tracks the functional phase fraction phi_f over time.
    The inert phase fraction phi_i is assumed constant (passive granules).
    Void fraction phi_v = 1 - phi_f - phi_i.

    The compaction rate is governed by the balance between cell-generated
    traction stress and contact resistance near the jamming transition.

    Parameters
    ----------
    E_modulus : float
        Young's modulus of hydrogel in kPa.
    phi_f0 : float
        Initial functional phase fraction.
    phi_i0 : float
        Initial (constant) inert phase fraction.
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
        Random close packing fraction (default 0.64 for spheres).
    """

    def __init__(self, E_modulus, phi_f0, phi_i0, R_func, n_cells_per_granule,
                 N_func, domain_volume, mc_params, bridge_attempt_rate=0.3,
                 bridge_formation_time=2.0, phi_RCP=0.64):
        self.E_modulus = E_modulus
        self.phi_f0 = phi_f0
        self.phi_i0 = phi_i0  # constant
        self.R_func = R_func
        self.n_cells_per_granule = n_cells_per_granule
        self.N_func = N_func
        self.domain_volume = domain_volume
        self.mc_params = mc_params
        self.bridge_attempt_rate = bridge_attempt_rate
        self.bridge_formation_time = bridge_formation_time
        self.phi_RCP = phi_RCP

        # Derived quantities
        self.F_cell = motor_clutch_force(E_modulus, mc_params)
        self.n_total_cells = N_func * n_cells_per_granule

        # Cell number density (cells per um^3 or um^2)
        self.n_cell_density = self.n_total_cells / domain_volume

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
        return 1.0 - np.exp(-self.bridge_attempt_rate * t_bridge_eff * maturity)

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

    def sigma_resist(self, phi_solid, sigma_0, alpha):
        """Contact resistance stress near jamming.

        For phi_solid > phi_J (jamming point = phi_RCP):
            sigma_resist = sigma_0 * (phi_solid / phi_J - 1)^alpha

        For phi_solid <= phi_J:
            sigma_resist = 0

        Parameters
        ----------
        phi_solid : float
            Total solid fraction (phi_f + phi_i).
        sigma_0 : float
            Jamming stress prefactor (nN/um^2).
        alpha : float
            Jamming exponent (1 for Hertzian contacts).

        Returns
        -------
        float
            Resistance stress in nN/um^2.
        """
        ratio = phi_solid / self.phi_RCP
        if ratio <= 1.0:
            return 0.0
        return sigma_0 * (ratio - 1.0) ** alpha

    def compaction_rate(self, phi_f, t, eta_eff, sigma_0, alpha):
        """Compute dphi_f/dt.

        dphi_f/dt = -phi_f * epsilon_dot_comp

        where epsilon_dot_comp = (sigma_cell - sigma_resist) / eta_eff

        Compaction only occurs when cell stress exceeds resistance.

        Parameters
        ----------
        phi_f : float
            Current functional phase fraction.
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
            Rate of change of phi_f (1/h).
        """
        phi_solid = phi_f + self.phi_i0
        s_cell = self.sigma_cell(t)
        s_resist = self.sigma_resist(phi_solid, sigma_0, alpha)
        net_stress = s_cell - s_resist
        if net_stress <= 0:
            return 0.0
        eps_dot = net_stress / eta_eff
        return -phi_f * eps_dot

    def solve(self, t_span, eta_eff, sigma_0, alpha, n_points=500):
        """Integrate the compaction ODE over t_span.

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
        """
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        def rhs(t, y):
            phi_f = y[0]
            # Clamp phi_f to physical range
            phi_f = np.clip(phi_f, 0.01, 0.99 - self.phi_i0)
            return [self.compaction_rate(phi_f, t, eta_eff, sigma_0, alpha)]

        sol = solve_ivp(rhs, t_span, [self.phi_f0], t_eval=t_eval,
                        method='RK45', rtol=1e-8, atol=1e-10,
                        max_step=0.5)
        return sol

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

def fit_to_data(t_data, phi_f_data, model, bounds=None):
    """Fit compaction model parameters to simulation/experimental data.

    Fits three parameters: eta_eff, sigma_0, alpha.

    Parameters
    ----------
    t_data : array-like
        Time values in hours.
    phi_f_data : array-like
        Functional phase fraction values.
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
    phi_f_data = np.asarray(phi_f_data, dtype=float)

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
            phi_f_model = np.interp(t_data, sol.t, sol.y[0])
            residual = np.sum((phi_f_data - phi_f_model) ** 2)
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
    phi_f_model = np.interp(t_data, sol.t, sol.y[0])

    # R-squared
    ss_res = np.sum((phi_f_data - phi_f_model) ** 2)
    ss_tot = np.sum((phi_f_data - np.mean(phi_f_data)) ** 2)
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

    # Estimate phi_RCP from aspect ratio
    ar_mean = float(_get('aspect_ratio_func_mean', 1.0))
    phi_RCP = min(0.74, 0.64 + 0.08 * (max(1.0, ar_mean) - 1.0))

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
    )


def from_run(run_dir):
    """Load data from a simulation run, construct model, fit, and return results.

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

    # Update N_func from snapshot if available
    if snaps:
        snap0 = snaps[0]
        gtype = snap0.get('gtype', np.array([]))
        if len(gtype) > 0:
            model.N_func = int(np.sum(gtype == 0))
            model.n_total_cells = model.N_func * model.n_cells_per_granule
            model.n_cell_density = model.n_total_cells / model.domain_volume

    # Fit
    fit_result = fit_to_data(t_data, phi_f_data, model)

    data = {
        'time': t_data,
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

def plot_fit(t_data, phi_f_data, model, fit_params, ax=None):
    """Overlay DEM data and mean-field model fit.

    Parameters
    ----------
    t_data : array-like
        Time values from data.
    phi_f_data : array-like
        Functional phase fraction from data.
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
    phi_f_data = np.asarray(phi_f_data)

    sol = fit_params['sol']

    ax.plot(t_data, phi_f_data, 'ko', ms=4, alpha=0.6, label='DEM data')
    ax.plot(sol.t, sol.y[0], 'C1-', lw=2, label='Mean-field model')

    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$\phi_f$ (functional phase fraction)')
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
    """Plot phi_f, phi_i, phi_v vs time from the mean-field model.

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

    phi_f = sol.y[0]
    phi_i = np.full_like(sol.t, model.phi_i0)
    phi_v = 1.0 - phi_f - phi_i

    ax.fill_between(sol.t, 0, phi_f, color='orangered', alpha=0.4,
                    label=r'$\phi_f$ (functional)')
    ax.fill_between(sol.t, phi_f, phi_f + phi_i, color='steelblue', alpha=0.4,
                    label=r'$\phi_i$ (inert)')
    ax.fill_between(sol.t, phi_f + phi_i, 1.0, color='lightgreen', alpha=0.4,
                    label=r'$\phi_v$ (void)')

    ax.plot(sol.t, phi_f, 'C1-', lw=2)
    ax.plot(sol.t, phi_f + phi_i, 'C0-', lw=1.5)
    ax.plot(sol.t, np.ones_like(sol.t), 'k-', lw=0.5)

    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Phase fraction')
    ax.set_title('Mean-Field Phase Evolution')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)

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

    phi_f = sol.y[0]
    phi_v = 1.0 - phi_f - model.phi_i0

    K = model.permeability(phi_v, d_grain)
    Q = model.darcy_flow(phi_v, d_grain, delta_P=1.0, mu=1e-3,
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

    phi_f = sol.y[0]
    sigma_0 = fit_params['sigma_0']
    alpha = fit_params['alpha']

    s_cell = np.array([model.sigma_cell(t) for t in sol.t])
    s_resist = np.array([model.sigma_resist(phi_f[i] + model.phi_i0,
                                             sigma_0, alpha)
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
    phi_f_data = data['phi_f']
    t_span = (float(t_data[0]), float(t_data[-1]))

    print("=" * 65)
    print("  Mean-Field Compaction Model")
    print("=" * 65)
    print(f"\n  Run directory: {run_dir}")
    print(f"  Output: {outdir}/")
    print(f"\n  Model parameters:")
    print(f"    E_modulus     = {model.E_modulus:.1f} kPa")
    print(f"    phi_f0        = {model.phi_f0:.3f}")
    print(f"    phi_i0        = {model.phi_i0:.3f}")
    print(f"    R_func        = {model.R_func:.1f} um")
    print(f"    N_func        = {model.N_func}")
    print(f"    n_cells/gran  = {model.n_cells_per_granule}")
    print(f"    F_cell        = {model.F_cell:.2f} nN")
    print(f"    phi_RCP       = {model.phi_RCP:.3f}")
    print(f"\n  Fitted parameters:")
    print(f"    eta_eff       = {fit_result['eta_eff']:.4g} nN*h/um^2")
    print(f"    sigma_0       = {fit_result['sigma_0']:.4g} nN/um^2")
    print(f"    alpha         = {fit_result['alpha']:.3f}")
    print(f"    R^2           = {fit_result['R2']:.6f}")
    print(f"    Residual      = {fit_result['residual']:.4g}")

    # 1. Compaction fit
    print("\n  [1/4] Compaction fit...")
    fig1 = plot_fit(t_data, phi_f_data, model, fit_result)
    # Overlay phi_i and phi_v data for reference
    ax = fig1.axes[0]
    ax.plot(t_data, data['phi_i'], 'C0s', ms=3, alpha=0.4, label=r'$\phi_i$ data')
    ax.plot(t_data, data['phi_v'], 'C2^', ms=3, alpha=0.4, label=r'$\phi_v$ data')
    ax.legend(fontsize=8)
    fig1.savefig(str(Path(outdir) / 'compaction_fit.png'), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # 2. Phase evolution
    print("  [2/4] Phase evolution...")
    fig2 = plot_phase_evolution(model, fit_result, t_span)
    # Overlay data points
    ax = fig2.axes[0]
    ax.plot(t_data, phi_f_data, 'C1o', ms=3, alpha=0.5, zorder=5)
    ax.plot(t_data, phi_f_data + data['phi_i'], 'C0o', ms=3, alpha=0.5, zorder=5)
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
