#!/usr/bin/env python3
"""
Generate all example figures for the Mean Field Modeling Guide.
=============================================================

Run from the repo root:
    python CodeLog/MeanFieldModeling/generate_examples.py

Produces PNG figures in CodeLog/MeanFieldModeling/figures/.

Examples 1-10: Synthetic pedagogical examples (always work).
Examples 11-14: Real DEM data examples (require results/LHC/ directory).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUTDIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'results', 'LHC')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'figure.facecolor': 'white',
})


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# =====================================================================
# Example 1: Coffee Cooling
# =====================================================================
def example_01():
    print("\n[1/14] Coffee Cooling...")
    T_0, T_room, k = 90.0, 22.0, 0.1
    t = np.linspace(0, 60, 200)
    T = T_room + (T_0 - T_room) * np.exp(-k * t)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, T, 'C1-', lw=2.5,
            label=r'$T(t) = T_{room} + (T_0 - T_{room})\,e^{-kt}$')
    ax.axhline(T_room, color='grey', ls='--', label=f'Room temp ({T_room}°C)')
    ax.scatter([0], [T_0], color='C1', s=60, zorder=5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('Example 1: Coffee Cooling — Your First Mean Field Model')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(15, 95)
    save(fig, 'example_01_coffee_cooling.png')


# =====================================================================
# Example 2: Logistic Growth
# =====================================================================
def example_02():
    print("[2/14] Logistic Growth...")
    N_0, r, K = 100, 0.5, 1_000_000
    t_eval = np.linspace(0, 40, 300)

    def logistic(t, y):
        return [r * y[0] * (1.0 - y[0] / K)]

    sol = solve_ivp(logistic, (0, 40), [N_0], t_eval=t_eval, method='RK45')
    N_exp = N_0 * np.exp(r * t_eval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(t_eval, sol.y[0], 'C0-', lw=2.5, label='Logistic (mean field)')
    ax1.plot(t_eval, np.clip(N_exp, 0, 5e6), 'C3--', lw=1.5, label='Uncapped exponential')
    ax1.axhline(K, color='grey', ls=':', label=f'Carrying capacity K = {K:,.0f}')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Population N')
    ax1.set_title('Linear Scale')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 2e6)
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(t_eval, sol.y[0], 'C0-', lw=2.5, label='Logistic')
    ax2.semilogy(t_eval, N_exp, 'C3--', lw=1.5, label='Exponential')
    ax2.axhline(K, color='grey', ls=':')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Population N (log scale)')
    ax2.set_title('Log Scale')
    ax2.legend(fontsize=9)
    ax2.set_ylim(1, 1e8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Example 2: Population Growth — Adding Nonlinearity', fontsize=14, y=1.02)
    save(fig, 'example_02_logistic_growth.png')


# =====================================================================
# Example 3: Two Competing Forces (Thermostat)
# =====================================================================
def example_03():
    print("[3/14] Competing Forces (Thermostat)...")
    C, T_set, T_0 = 10.0, 25.0, 18.0
    Q_max, tau = 8.0, 3.0
    s0_heat, alpha_heat = 2.0, 1.5

    def Q_heater(t):
        return Q_max * (1.0 - np.exp(-t / tau))

    def Q_cooling(T):
        if T <= T_set:
            return 0.0
        return s0_heat * ((T - T_set) / T_set) ** alpha_heat

    def rhs(t, y):
        net = Q_heater(t) - Q_cooling(y[0])
        return [max(0.0, net) / C]

    t_eval = np.linspace(0, 30, 500)
    sol = solve_ivp(rhs, (0, 30), [T_0], t_eval=t_eval, method='RK45')
    T = sol.y[0]
    q_h = np.array([Q_heater(t) for t in sol.t])
    q_c = np.array([Q_cooling(T[i]) for i in range(len(T))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(sol.t, T, 'C1-', lw=2.5, label='Temperature T(t)')
    ax1.axhline(T_set, color='grey', ls='--', label=f'Threshold = {T_set}°C')
    ax1.set_xlabel('Time (hours)')
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('System Response')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(sol.t, q_h, 'C1-', lw=2.5, label='Driving (heater)')
    ax2.plot(sol.t, q_c, 'C0-', lw=2.5, label='Resistance (cooling)')
    ax2.fill_between(sol.t, q_c, q_h, where=q_h > q_c,
                     color='C1', alpha=0.15, label='Net driving')
    ax2.fill_between(sol.t, q_h, q_c, where=q_c >= q_h,
                     color='C0', alpha=0.15, label='Arrested')
    ax2.set_xlabel('Time (hours)')
    ax2.set_ylabel('Heat flux')
    ax2.set_title('Stress Balance')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Example 3: Two Competing Forces — Driving vs Resistance', fontsize=14, y=1.02)
    save(fig, 'example_03_competing_forces.png')


# =====================================================================
# Example 4: Kozeny-Carman Permeability
# =====================================================================
def example_04():
    print("[4/14] Kozeny-Carman Permeability...")
    d_grain = 80.0
    phi_v = np.linspace(0.05, 0.60, 200)
    eps = np.clip(phi_v, 0.01, 0.99)
    K = eps**3 * d_grain**2 / (180.0 * (1.0 - eps)**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(phi_v, K, 'C2-', lw=2.5)
    ax1.axvspan(0.15, 0.25, alpha=0.12, color='C3', label='Typical scaffold range')
    ax1.set_xlabel(r'Void fraction $\phi_v$')
    ax1.set_ylabel(r'Permeability $K$ (µm²)')
    ax1.set_title(f'Linear Scale (d = {d_grain:.0f} µm)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(phi_v, K, 'C2-', lw=2.5)
    ax2.axvspan(0.15, 0.25, alpha=0.12, color='C3', label='Typical scaffold range')
    ax2.set_xlabel(r'Void fraction $\phi_v$')
    ax2.set_ylabel(r'Permeability $K$ (µm²) — log scale')
    ax2.set_title('Log Scale — Cubic Sensitivity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Example 4: Kozeny-Carman — Structure Determines Transport', fontsize=14, y=1.02)
    save(fig, 'example_04_kozeny_carman.png')


# =====================================================================
# Example 5: Volume Conservation
# =====================================================================
def example_05():
    print("[5/14] Volume Conservation...")
    phi_f, phi_i = 0.40, 0.25
    phi_solid = phi_f + phi_i

    x_f = np.linspace(phi_f / 0.95, phi_f / phi_solid + 0.15, 200)
    phi_v_f = 1.0 - phi_f / x_f
    phi_v_i = 1.0 - phi_i / (1.0 - x_f)
    phi_v_g = x_f * phi_v_f + (1.0 - x_f) * phi_v_i

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x_f, phi_v_f, 'C1-', lw=2.5, label=r'$\phi_v^{func}$ (functional zone)')
    ax1.plot(x_f, phi_v_i, 'C0-', lw=2.5, label=r'$\phi_v^{inert}$ (inert zone)')
    ax1.plot(x_f, phi_v_g, 'k--', lw=1.5, label=r'$\phi_v^{global}$ (total)')
    ax1.axvline(phi_f / phi_solid, color='grey', ls=':', lw=1.5,
                label=r'$x_f^0$ (uniformly mixed)')
    ax1.set_xlabel(r'$x_f$ (functional zone fraction)')
    ax1.set_ylabel('Void fraction')
    ax1.set_title('Local vs Global Void Fractions')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 0.8)
    ax1.annotate('Compaction\ndirection', fontsize=10, ha='center',
                 xy=(phi_f/phi_solid - 0.05, 0.5),
                 xytext=(phi_f/phi_solid + 0.06, 0.5),
                 arrowprops=dict(arrowstyle='->', lw=1.5))

    # Stacked bars at three time points
    xf_vals = [phi_f / phi_solid, 0.55, 0.48]
    labels = ['t = 0 h\n(uniform)', 't = 12 h\n(mid)', 't = 48 h\n(final)']
    for idx, (xf_val, label) in enumerate(zip(xf_vals, labels)):
        ps_f = phi_f / xf_val
        pv_f = 1.0 - ps_f
        ps_i = phi_i / (1.0 - xf_val)
        pv_i = 1.0 - ps_i
        ax2.bar(idx - 0.15, ps_f * xf_val, 0.25, color='C1', alpha=0.8,
                label='Func solid' if idx == 0 else '')
        ax2.bar(idx - 0.15, pv_f * xf_val, 0.25, bottom=ps_f * xf_val,
                color='C1', alpha=0.3, label='Func void' if idx == 0 else '')
        ax2.bar(idx + 0.15, ps_i * (1-xf_val), 0.25, color='C0', alpha=0.8,
                label='Inert solid' if idx == 0 else '')
        ax2.bar(idx + 0.15, pv_i * (1-xf_val), 0.25, bottom=ps_i * (1-xf_val),
                color='C0', alpha=0.3, label='Inert void' if idx == 0 else '')
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Volume fraction of total domain')
    ax2.set_title('Void Redistribution During Compaction')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Example 5: Volume Conservation — The Key Constraint', fontsize=14, y=1.02)
    save(fig, 'example_05_volume_conservation.png')


# =====================================================================
# Example 6: Motor-Clutch Force
# =====================================================================
def example_06():
    print("[6/14] Motor-Clutch Model...")
    F_stall = 25.0
    k_opt = 375.0
    engagement = 1.0 / 1.1
    cell_r = 10.0
    nu = 0.45

    def mc_force(E):
        k_sub = np.pi * E * cell_r / (1.0 - nu**2)
        beta = k_sub / (k_sub + k_opt)
        return F_stall * beta * engagement

    E_range = np.logspace(-1, 3, 200)
    F = np.array([mc_force(E) for E in E_range])
    F_max = F_stall * engagement

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogx(E_range, F, 'C1-', lw=2.5)
    ax.axvspan(1, 50, alpha=0.1, color='C2', label='Hydrogel range (1-50 kPa)')
    ax.axhline(F_max, color='grey', ls='--', alpha=0.5,
               label=f'Saturation: {F_max:.1f} nN')
    ax.set_xlabel('Substrate Stiffness $E$ (kPa)')
    ax.set_ylabel('Traction Force per Cell (nN)')
    ax.set_title('Example 6: Motor-Clutch — Cell Force vs Substrate Stiffness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, F_max * 1.2)
    ax.annotate('Soft: clutches slip,\nlow traction', fontsize=9,
                xy=(0.3, mc_force(0.3)),
                xytext=(0.12, 10),
                arrowprops=dict(arrowstyle='->', color='C3'))
    ax.annotate('Stiff: motors\nsaturated', fontsize=9,
                xy=(500, mc_force(500)),
                xytext=(80, 15),
                arrowprops=dict(arrowstyle='->', color='C3'))
    save(fig, 'example_06_motor_clutch.png')


# =====================================================================
# Example 7: Jamming Resistance
# =====================================================================
def example_07():
    print("[7/14] Jamming Resistance...")
    phi_f, phi_RCP = 0.40, 0.64
    x_f = np.linspace(phi_f / 0.95, 0.80, 300)
    phi_local = phi_f / x_f
    sigma_0 = 0.01

    fig, ax = plt.subplots(figsize=(8, 5))
    for alpha, color, ls in [(0.5, 'C0', '-'), (1.0, 'C1', '--'),
                              (1.5, 'C3', ':'), (2.0, 'C4', '-.')]:
        ratio = phi_local / phi_RCP
        sigma = np.where(ratio > 1.0, sigma_0 * (ratio - 1.0)**alpha, 0.0)
        ax.plot(x_f, sigma, color=color, ls=ls, lw=2.5, label=f'$\\alpha$ = {alpha}')

    ax.axvline(phi_f / phi_RCP, color='grey', ls='--', lw=1.5,
               label=f'Jamming ($\\phi_{{RCP}}$ = {phi_RCP})')
    ax.set_xlabel('$x_f$ (functional zone fraction)')
    ax.set_ylabel(r'$\sigma_{resist}$')
    ax.set_title('Example 7: Jamming Resistance — Stress Diverges Near $\\phi_{RCP}$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.42, 0.80)
    save(fig, 'example_07_jamming_resistance.png')


# =====================================================================
# Example 8: Full Model (4-panel)
# =====================================================================
def example_08():
    print("[8/14] Full GELS Model...")
    E_mod = 10.0
    phi_f, phi_i = 0.40, 0.25
    phi_solid = phi_f + phi_i
    phi_RCP, phi_max = 0.64, 0.72
    R_func = 40.0
    N_func, n_cells = 160, 8
    domain_area = 800**2
    t_spread, fa_rate = 3.0, 0.3
    b_rate, b_form_time = 0.3, 2.0

    k_sub = np.pi * E_mod * 10.0 / (1.0 - 0.45**2)
    F_cell = 25.0 * (k_sub / (k_sub + 375.0)) * (1.0 / 1.1)
    n_dens = N_func * n_cells / domain_area
    x_f0 = phi_f / phi_solid
    x_f_min = phi_f / phi_max

    def maturity(t):
        return 1.0 - np.exp(-fa_rate * max(0.0, t - t_spread))

    def bridge_frac(t):
        m = maturity(t)
        if m < 0.1: return 0.0
        tb = max(0.0, t - t_spread - 1.0)
        return 1.0 - np.exp(-b_rate * tb * m)

    def bridge_ramp(t):
        return min(1.0, max(0.0, t - t_spread - 1.0) / b_form_time)

    def sigma_cell(t):
        return n_dens * F_cell * bridge_frac(t) * maturity(t) * bridge_ramp(t)

    eta_eff, sigma_0, alpha = 0.5, 0.001, 1.2

    def sigma_resist(xf):
        ratio = (phi_f / max(xf, 1e-6)) / phi_RCP
        return sigma_0 * (ratio - 1.0)**alpha if ratio > 1.0 else 0.0

    def rhs(t, y):
        xf = np.clip(y[0], x_f_min, 1.0 - phi_i)
        net = sigma_cell(t) - sigma_resist(xf)
        return [-xf * max(0.0, net) / eta_eff]

    t_eval = np.linspace(0, 72, 500)
    sol = solve_ivp(rhs, (0, 72), [x_f0], t_eval=t_eval, method='RK45',
                    rtol=1e-8, atol=1e-10)

    xf = sol.y[0]
    pv_f = 1.0 - phi_f / np.maximum(xf, 1e-6)
    pv_i = 1.0 - phi_i / np.maximum(1.0 - xf, 1e-6)
    sc = np.array([sigma_cell(t) for t in sol.t])
    sr = np.array([sigma_resist(xf[i]) for i in range(len(sol.t))])
    d_f = 2.0 * R_func
    eps = np.clip(pv_f, 0.01, 0.99)
    K_f = eps**3 * d_f**2 / (180.0 * (1.0 - eps)**2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0,0].plot(sol.t, xf, 'C1-', lw=2.5, label='$x_f(t)$')
    axes[0,0].axhline(phi_f/phi_RCP, color='grey', ls='--', alpha=0.5, label='Jamming limit')
    axes[0,0].axhline(x_f_min, color='grey', ls=':', alpha=0.5, label='Deformable limit')
    axes[0,0].set_xlabel('Time (h)')
    axes[0,0].set_ylabel('$x_f$')
    axes[0,0].set_title('(a) Functional Zone Compaction')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    axes[0,1].plot(sol.t, pv_f, 'C1-', lw=2.5, label=r'$\phi_v^{func}$')
    axes[0,1].plot(sol.t, pv_i, 'C0-', lw=2.5, label=r'$\phi_v^{inert}$')
    axes[0,1].set_xlabel('Time (h)')
    axes[0,1].set_ylabel('Local void fraction')
    axes[0,1].set_title('(b) Void Redistribution')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    axes[1,0].plot(sol.t, sc, 'C1-', lw=2.5, label=r'$\sigma_{cell}$ (traction)')
    axes[1,0].plot(sol.t, sr, 'C0-', lw=2.5, label=r'$\sigma_{resist}$ (jamming)')
    axes[1,0].fill_between(sol.t, sr, sc, where=sc > sr,
                           color='C1', alpha=0.15, label='Net compaction')
    axes[1,0].set_xlabel('Time (h)')
    axes[1,0].set_ylabel(r'Stress (nN/µm²)')
    axes[1,0].set_title('(c) Stress Balance')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    axes[1,1].semilogy(sol.t, K_f, 'C2-', lw=2.5)
    axes[1,1].set_xlabel('Time (h)')
    axes[1,1].set_ylabel('Permeability $K$ (µm²)')
    axes[1,1].set_title('(d) Functional Zone Permeability')
    axes[1,1].grid(True, alpha=0.3)

    fig.suptitle('Example 8: Complete GELS Mean Field Model', fontsize=14, y=1.01)
    plt.tight_layout()
    save(fig, 'example_08_full_model.png')

    return maturity, bridge_frac, bridge_ramp


# =====================================================================
# Example 9: Biological Timeline
# =====================================================================
def example_09(maturity, bridge_frac, bridge_ramp):
    print("[9/14] Biological Timeline...")
    t = np.linspace(0, 48, 300)
    mat = np.array([maturity(ti) for ti in t])
    fb = np.array([bridge_frac(ti) for ti in t])
    rp = np.array([bridge_ramp(ti) for ti in t])
    combined = mat * fb * rp

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, mat, 'C0-', lw=2, label='FA Maturity')
    ax.plot(t, fb, 'C1-', lw=2, label='Bridge Fraction')
    ax.plot(t, rp, 'C2--', lw=1.5, label='Force Ramp')
    ax.plot(t, combined, 'k-', lw=2.5, label='Combined (mat $\\times$ bridge $\\times$ ramp)')
    ax.axvline(3.0, color='grey', ls=':', alpha=0.5)
    ax.text(3.3, 0.95, 'Spreading\ncomplete', fontsize=9, color='grey')
    ax.axvline(4.0, color='grey', ls=':', alpha=0.5)
    ax.text(4.3, 0.85, 'Bridges\nstart', fontsize=9, color='grey')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Fraction')
    ax.set_title('Example 9: Biological Timeline — How Cell Force Ramps Up')
    ax.legend(loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 1.05)
    save(fig, 'example_09_biological_timeline.png')


# =====================================================================
# Example 10: Parameter Sensitivity
# =====================================================================
def example_10():
    print("[10/14] Parameter Sensitivity...")
    phi_f, phi_i = 0.40, 0.25
    phi_solid = phi_f + phi_i
    phi_RCP, phi_max = 0.64, 0.72
    x_f0 = phi_f / phi_solid
    x_f_min = phi_f / phi_max

    def solve_model(eta_eff, sigma_0, alpha, E_mod, t_total=72):
        k_sub = np.pi * E_mod * 10.0 / (1.0 - 0.45**2)
        F_cell = 25.0 * (k_sub / (k_sub + 375.0)) * (1.0 / 1.1)
        n_dens = 1280 / 640000.0

        def rhs(t, y):
            xf = np.clip(y[0], x_f_min, 1.0 - phi_i)
            t_eff = max(0.0, t - 3.0)
            mat = 1.0 - np.exp(-0.3 * t_eff)
            if mat < 0.1: return [0.0]
            tb = max(0.0, t - 4.0)
            fb = 1.0 - np.exp(-0.3 * tb * mat)
            rp = min(1.0, max(0.0, t - 4.0) / 2.0)
            sc = n_dens * F_cell * fb * mat * rp
            ratio = (phi_f / max(xf, 1e-6)) / phi_RCP
            sr = sigma_0 * (ratio - 1.0)**alpha if ratio > 1.0 else 0.0
            net = sc - sr
            return [-xf * max(0.0, net) / eta_eff]

        te = np.linspace(0, t_total, 300)
        sol = solve_ivp(rhs, (0, t_total), [x_f0], t_eval=te, method='RK45',
                        rtol=1e-8, atol=1e-10)
        return sol.t, sol.y[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Viscosity
    for eta, ls in [(0.1, '-'), (0.5, '--'), (2.0, ':'), (10.0, '-.')]:
        t_a, xf_a = solve_model(eta, 0.001, 1.2, 10.0)
        axes[0,0].plot(t_a, xf_a, ls, lw=2, label=f'$\\eta$ = {eta}')
    axes[0,0].set_title('(a) Effect of Viscosity $\\eta_{eff}$')
    axes[0,0].set_xlabel('Time (h)')
    axes[0,0].set_ylabel('$x_f$')
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)

    # (b) Jamming prefactor
    for s0, ls in [(0.0001, '-'), (0.001, '--'), (0.01, ':'), (0.1, '-.')]:
        t_a, xf_a = solve_model(0.5, s0, 1.2, 10.0)
        axes[0,1].plot(t_a, xf_a, ls, lw=2, label=f'$\\sigma_0$ = {s0}')
    axes[0,1].set_title('(b) Effect of Jamming Prefactor $\\sigma_0$')
    axes[0,1].set_xlabel('Time (h)')
    axes[0,1].set_ylabel('$x_f$')
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)

    # (c) Jamming exponent
    for al, ls in [(0.5, '-'), (1.0, '--'), (1.5, ':'), (2.5, '-.')]:
        t_a, xf_a = solve_model(0.5, 0.001, al, 10.0)
        axes[1,0].plot(t_a, xf_a, ls, lw=2, label=f'$\\alpha$ = {al}')
    axes[1,0].set_title('(c) Effect of Jamming Exponent $\\alpha$')
    axes[1,0].set_xlabel('Time (h)')
    axes[1,0].set_ylabel('$x_f$')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # (d) Substrate stiffness
    for E, ls in [(1.0, '-'), (5.0, '--'), (20.0, ':'), (100.0, '-.')]:
        t_a, xf_a = solve_model(0.5, 0.001, 1.2, E)
        axes[1,1].plot(t_a, xf_a, ls, lw=2, label=f'E = {E} kPa')
    axes[1,1].set_title('(d) Effect of Substrate Stiffness $E$')
    axes[1,1].set_xlabel('Time (h)')
    axes[1,1].set_ylabel('$x_f$')
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)

    fig.suptitle('Example 10: Parameter Sensitivity — How Each Parameter Affects Compaction',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save(fig, 'example_10_parameter_sensitivity.png')


# =====================================================================
# Helper: load DEM history from a trial directory
# =====================================================================
def _load_dem_history(trial_dir):
    """Load time, x_f, porosity, n_bridges from a DOE trial."""
    import csv, json
    params_path = os.path.join(trial_dir, 'params.json')
    history_path = os.path.join(trial_dir, 'history.csv')
    if not os.path.exists(params_path) or not os.path.exists(history_path):
        return None
    with open(params_path) as f:
        params = json.load(f)
    with open(history_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    t = np.array([float(r['time']) for r in rows])
    xf = np.array([float(r.get('x_f', 0.5)) for r in rows])
    porosity = np.array([float(r.get('porosity', 0.3)) for r in rows])
    n_bridges = np.array([float(r.get('n_bridges', 0)) for r in rows])
    phi_f = np.array([float(r.get('phi_f_mean', 0)) for r in rows])
    K_kc = np.array([float(r.get('K_kozeny_carman', 0)) for r in rows])
    return {
        'params': params, 'time': t, 'x_f': xf, 'porosity': porosity,
        'n_bridges': n_bridges, 'phi_f': phi_f, 'K_kc': K_kc, 'name': os.path.basename(trial_dir)
    }


# =====================================================================
# Example 11: Single DEM fit (real data)
# =====================================================================
def example_11():
    """Fit mean field model to a single DEM run and show the 4-panel diagnostic."""
    # Pick a trial that exists
    trial_candidates = ['DOE_2D_0010', 'DOE_2D_0006', 'DOE_2D_0001']
    trial_dir = None
    for name in trial_candidates:
        path = os.path.join(RESULTS_DIR, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'history.csv')):
            trial_dir = path
            trial_name = name
            break
    if trial_dir is None:
        print("[11/14] Single DEM Fit... SKIPPED (no results/LHC/ directory)")
        return
    print(f"[11/14] Single DEM Fit ({trial_name})...")

    data = _load_dem_history(trial_dir)
    p = data['params']
    E_mod = p.get('E_modulus', 10.0)
    phi_f_t = p.get('phi_f_target', 0.4)
    phi_i_t = p.get('phi_i_target', 0.3)
    R_f = p.get('R_func_mean', 40.0)

    # Build a simple mean field model from these parameters
    phi_solid = phi_f_t + phi_i_t
    phi_RCP = 0.82  # 2D
    E_ref = 10.0
    compliance = E_ref / (E_mod + E_ref)
    phi_max = min(phi_RCP + (1.0 - phi_RCP) * 0.5 * compliance, 0.99)
    x_f0 = data['x_f'][0] if data['x_f'][0] > 0.01 else phi_f_t / max(phi_solid, 0.01)
    x_f_min = phi_f_t / max(phi_max, 0.01)

    # Motor-clutch
    k_sub = np.pi * E_mod * 10.0 / (1.0 - 0.45**2)
    F_cell = 25.0 * (k_sub / (k_sub + 375.0)) * (1.0 / 1.1)
    N_func = max(1, int(round(phi_f_t * 800**2 / (np.pi * R_f**2))))
    n_cells = p.get('n_cells_per_granule', 8)
    n_dens = N_func * n_cells / 800**2

    # Solve with several eta/sigma_0/alpha to find best fit
    best_res, best_params = 1e20, (1.0, 0.001, 1.0)

    for log_eta in np.linspace(-2, 2, 15):
        for log_s0 in np.linspace(-5, -1, 10):
            for alpha in [0.8, 1.0, 1.2, 1.5, 2.0]:
                eta = 10**log_eta
                s0 = 10**log_s0

                def _rhs(t, y, _eta=eta, _s0=s0, _al=alpha):
                    xf = np.clip(y[0], x_f_min, 1.0 - phi_i_t)
                    te = max(0.0, t - 3.0)
                    mat = 1.0 - np.exp(-0.3 * te)
                    if mat < 0.1: return [0.0]
                    tb = max(0.0, t - 4.0)
                    fb = 1.0 - np.exp(-0.3 * tb * mat)
                    rp = min(1.0, max(0.0, t - 4.0) / 2.0)
                    sc = n_dens * F_cell * fb * mat * rp
                    ratio = (phi_f_t / max(xf, 1e-6)) / phi_RCP
                    sr = _s0 * (ratio - 1.0)**_al if ratio > 1.0 else 0.0
                    net = sc - sr
                    return [-xf * max(0.0, net) / _eta]

                try:
                    sol = solve_ivp(_rhs, (0, data['time'][-1]), [x_f0],
                                    t_eval=data['time'], method='RK45',
                                    rtol=1e-7, atol=1e-9)
                    if sol.status == 0 and len(sol.y[0]) == len(data['x_f']):
                        res = np.sum((data['x_f'] - sol.y[0])**2)
                        if res < best_res:
                            best_res = res
                            best_params = (eta, s0, alpha)
                except Exception:
                    pass

    eta, s0, al = best_params
    ss_tot = np.sum((data['x_f'] - np.mean(data['x_f']))**2)
    R2 = 1.0 - best_res / max(ss_tot, 1e-30)

    # Solve with best params for smooth curve
    def rhs_best(t, y):
        xf = np.clip(y[0], x_f_min, 1.0 - phi_i_t)
        te = max(0.0, t - 3.0)
        mat = 1.0 - np.exp(-0.3 * te)
        if mat < 0.1: return [0.0]
        tb = max(0.0, t - 4.0)
        fb = 1.0 - np.exp(-0.3 * tb * mat)
        rp = min(1.0, max(0.0, t - 4.0) / 2.0)
        sc = n_dens * F_cell * fb * mat * rp
        ratio = (phi_f_t / max(xf, 1e-6)) / phi_RCP
        sr = s0 * (ratio - 1.0)**al if ratio > 1.0 else 0.0
        net = sc - sr
        return [-xf * max(0.0, net) / eta]

    t_smooth = np.linspace(0, data['time'][-1], 300)
    sol = solve_ivp(rhs_best, (0, data['time'][-1]), [x_f0],
                    t_eval=t_smooth, method='RK45', rtol=1e-8, atol=1e-10)

    xf_model = sol.y[0]
    pv_f_model = 1.0 - phi_f_t / np.maximum(xf_model, 1e-6)

    # Stresses along model trajectory
    def _sc(t):
        te = max(0.0, t - 3.0)
        mat = 1.0 - np.exp(-0.3 * te)
        if mat < 0.1: return 0.0
        tb = max(0.0, t - 4.0)
        fb = 1.0 - np.exp(-0.3 * tb * mat)
        rp = min(1.0, max(0.0, t - 4.0) / 2.0)
        return n_dens * F_cell * fb * mat * rp

    def _sr(xf):
        ratio = (phi_f_t / max(xf, 1e-6)) / phi_RCP
        return s0 * (ratio - 1.0)**al if ratio > 1.0 else 0.0

    sc_arr = np.array([_sc(t) for t in sol.t])
    sr_arr = np.array([_sr(xf_model[i]) for i in range(len(sol.t))])

    d_f = 2.0 * R_f
    eps_f = np.clip(pv_f_model, 0.01, 0.99)
    K_model = eps_f**3 * d_f**2 / (180.0 * (1.0 - eps_f)**2)

    # 4-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Compaction fit
    axes[0,0].plot(data['time'], data['x_f'], 'ko', ms=5, alpha=0.7, label='DEM data')
    axes[0,0].plot(sol.t, xf_model, 'C1-', lw=2.5, label='Mean field fit')
    axes[0,0].set_xlabel('Time (h)')
    axes[0,0].set_ylabel('$x_f$')
    axes[0,0].set_title(f'(a) Compaction Fit — $R^2$ = {R2:.3f}')
    axes[0,0].legend(fontsize=9)
    axes[0,0].grid(True, alpha=0.3)
    txt = (f"$\\eta_{{eff}}$ = {eta:.3g}\n"
           f"$\\sigma_0$ = {s0:.3g}\n"
           f"$\\alpha$ = {al:.2f}")
    axes[0,0].text(0.97, 0.97, txt, transform=axes[0,0].transAxes, fontsize=9,
                   va='top', ha='right', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    # (b) Porosity comparison
    axes[0,1].plot(data['time'], data['porosity'], 'C2o', ms=5, alpha=0.7, label='DEM porosity')
    pv_global = 1.0 - phi_f_t - phi_i_t
    axes[0,1].axhline(pv_global, color='grey', ls='--', alpha=0.5,
                       label=f'Expected (1 - $\\phi_{{solid}}$) = {pv_global:.3f}')
    axes[0,1].set_xlabel('Time (h)')
    axes[0,1].set_ylabel('Porosity')
    axes[0,1].set_title('(b) Global Porosity (should be ~constant)')
    axes[0,1].legend(fontsize=9)
    axes[0,1].grid(True, alpha=0.3)

    # (c) Stress balance
    axes[1,0].plot(sol.t, sc_arr, 'C1-', lw=2.5, label=r'$\sigma_{cell}$')
    axes[1,0].plot(sol.t, sr_arr, 'C0-', lw=2.5, label=r'$\sigma_{resist}$')
    axes[1,0].fill_between(sol.t, sr_arr, sc_arr, where=sc_arr > sr_arr,
                           color='C1', alpha=0.15, label='Net compaction')
    axes[1,0].set_xlabel('Time (h)')
    axes[1,0].set_ylabel(r'Stress (nN/µm²)')
    axes[1,0].set_title('(c) Stress Balance')
    axes[1,0].legend(fontsize=9)
    axes[1,0].grid(True, alpha=0.3)

    # (d) Permeability
    axes[1,1].semilogy(sol.t, K_model, 'C2-', lw=2.5, label='Model $K$')
    if np.any(data['K_kc'] > 0):
        axes[1,1].semilogy(data['time'], np.maximum(data['K_kc'], 1e-3),
                           'ko', ms=4, alpha=0.5, label='DEM $K$')
    axes[1,1].set_xlabel('Time (h)')
    axes[1,1].set_ylabel('Permeability $K$ (µm²)')
    axes[1,1].set_title('(d) Kozeny-Carman Permeability')
    axes[1,1].legend(fontsize=9)
    axes[1,1].grid(True, alpha=0.3)

    fig.suptitle(f'Example 11: Mean Field Fit to DEM — {trial_name}\n'
                 f'E = {E_mod} kPa, $\\phi_f$ = {phi_f_t:.3f}, '
                 f'$\\phi_i$ = {phi_i_t:.3f}, R = {R_f} µm',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    save(fig, 'example_11_dem_single_fit.png')


# =====================================================================
# Example 12: DOE comparison (fitted params vs physical params)
# =====================================================================
def example_12():
    """Show how fitted parameters vary across DOE trials."""
    if not os.path.isdir(RESULTS_DIR):
        print("[12/14] DOE Comparison... SKIPPED (no results/LHC/ directory)")
        return
    print("[12/14] DOE Comparison...")

    # Collect data from all available trials
    trials = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(path) or not d.startswith('DOE_2D_'):
            continue
        data = _load_dem_history(path)
        if data is None:
            continue
        p = data['params']
        xf0 = data['x_f'][0] if data['x_f'][0] > 0 else 0.5
        xf_final = data['x_f'][-1]
        compaction = (xf0 - xf_final) / max(xf0, 1e-6)
        trials.append({
            'name': d,
            'E': p.get('E_modulus', 10),
            'func_ratio': p.get('func_ratio', 0.5),
            'R_func': p.get('R_func_mean', 40),
            'phi_solid': p.get('phi_solid_target', 0.7),
            'x_f_0': xf0,
            'x_f_final': xf_final,
            'compaction': compaction,
            'porosity_final': data['porosity'][-1],
            'K_final': data['K_kc'][-1] if data['K_kc'][-1] > 0 else np.nan,
            'max_bridges': data['n_bridges'].max(),
        })

    if len(trials) < 3:
        print("  Not enough trials for comparison, skipping.")
        return

    E_arr = np.array([t['E'] for t in trials])
    fr_arr = np.array([t['func_ratio'] for t in trials])
    R_arr = np.array([t['R_func'] for t in trials])
    comp_arr = np.array([t['compaction'] for t in trials])
    por_arr = np.array([t['porosity_final'] for t in trials])
    K_arr = np.array([t.get('K_final', np.nan) for t in trials])
    bridge_arr = np.array([t['max_bridges'] for t in trials])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Compaction vs modulus
    sc = axes[0,0].scatter(E_arr, comp_arr * 100, c=fr_arr, cmap='RdYlBu',
                           s=80, edgecolors='k', linewidths=0.5, vmin=0.3, vmax=0.95)
    axes[0,0].set_xlabel('Young\'s Modulus $E$ (kPa)')
    axes[0,0].set_ylabel('Compaction (%)')
    axes[0,0].set_title('(a) Compaction vs Stiffness')
    axes[0,0].grid(True, alpha=0.3)
    cb = plt.colorbar(sc, ax=axes[0,0])
    cb.set_label('Functional ratio')

    # (b) Compaction vs functional ratio
    sc2 = axes[0,1].scatter(fr_arr, comp_arr * 100, c=E_arr, cmap='viridis',
                            s=80, edgecolors='k', linewidths=0.5)
    axes[0,1].set_xlabel('Functional Ratio')
    axes[0,1].set_ylabel('Compaction (%)')
    axes[0,1].set_title('(b) Compaction vs Composition')
    axes[0,1].grid(True, alpha=0.3)
    cb2 = plt.colorbar(sc2, ax=axes[0,1])
    cb2.set_label('$E$ (kPa)')

    # (c) Permeability vs porosity
    valid_K = ~np.isnan(K_arr) & (K_arr > 0)
    if np.any(valid_K):
        sc3 = axes[1,0].scatter(por_arr[valid_K], K_arr[valid_K],
                                c=E_arr[valid_K], cmap='viridis',
                                s=80, edgecolors='k', linewidths=0.5)
        axes[1,0].set_yscale('log')
        cb3 = plt.colorbar(sc3, ax=axes[1,0])
        cb3.set_label('$E$ (kPa)')
    # Overlay K-C curve
    pv_range = np.linspace(0.15, 0.65, 100)
    d_mean = 2.0 * np.mean(R_arr)
    K_kc = pv_range**3 * d_mean**2 / (180.0 * (1.0 - pv_range)**2)
    axes[1,0].plot(pv_range, K_kc, 'k--', lw=1.5, alpha=0.5, label=f'K-C (d={d_mean:.0f} µm)')
    axes[1,0].set_xlabel('Final Porosity')
    axes[1,0].set_ylabel('Permeability $K$ (µm²)')
    axes[1,0].set_title('(c) Structure-Transport Relationship')
    axes[1,0].legend(fontsize=8)
    axes[1,0].grid(True, alpha=0.3)

    # (d) Max bridges vs granule size
    sc4 = axes[1,1].scatter(R_arr, bridge_arr, c=fr_arr, cmap='RdYlBu',
                            s=80, edgecolors='k', linewidths=0.5, vmin=0.3, vmax=0.95)
    axes[1,1].set_xlabel('Functional Granule Radius (µm)')
    axes[1,1].set_ylabel('Peak Bridge Count')
    axes[1,1].set_title('(d) Bridging vs Granule Size')
    axes[1,1].grid(True, alpha=0.3)
    cb4 = plt.colorbar(sc4, ax=axes[1,1])
    cb4.set_label('Functional ratio')

    fig.suptitle(f'Example 12: DOE Comparison — {len(trials)} DEM Trials',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    save(fig, 'example_12_doe_comparison.png')


# =====================================================================
# Example 13: Multi-run overlay
# =====================================================================
def example_13():
    """Overlay x_f trajectories from multiple DEM runs."""
    if not os.path.isdir(RESULTS_DIR):
        print("[13/14] Multi-Run Overlay... SKIPPED (no results/LHC/ directory)")
        return
    print("[13/14] Multi-Run Overlay...")

    # Collect trajectories
    trajectories = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, d)
        if not os.path.isdir(path) or not d.startswith('DOE_2D_'):
            continue
        data = _load_dem_history(path)
        if data is None:
            continue
        trajectories.append(data)

    if len(trajectories) < 3:
        print("  Not enough trials for overlay, skipping.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sort by E_modulus for coloring
    trajectories.sort(key=lambda d: d['params'].get('E_modulus', 10))
    E_values = [d['params'].get('E_modulus', 10) for d in trajectories]
    E_min, E_max = min(E_values), max(E_values)
    cmap = plt.cm.viridis

    for data in trajectories:
        E = data['params'].get('E_modulus', 10)
        fr = data['params'].get('func_ratio', 0.5)
        frac = (E - E_min) / max(E_max - E_min, 1)
        color = cmap(frac)

        # (a) x_f trajectories
        axes[0].plot(data['time'], data['x_f'], '-', color=color, lw=1.5, alpha=0.7)

        # (b) Porosity trajectories
        axes[1].plot(data['time'], data['porosity'], '-', color=color, lw=1.5, alpha=0.7)

        # (c) Bridge count
        axes[2].plot(data['time'], data['n_bridges'], '-', color=color, lw=1.5, alpha=0.7)

    axes[0].set_xlabel('Time (h)')
    axes[0].set_ylabel('$x_f$ (functional zone fraction)')
    axes[0].set_title('(a) Compaction Trajectories')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time (h)')
    axes[1].set_ylabel('Porosity')
    axes[1].set_title('(b) Porosity Evolution')
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel('Time (h)')
    axes[2].set_ylabel('Bridge Count')
    axes[2].set_title('(c) Bridge Formation')
    axes[2].grid(True, alpha=0.3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(E_min, E_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('$E$ (kPa)')

    fig.suptitle(f'Example 13: Multi-Run Overlay — {len(trajectories)} DOE Trials (colored by stiffness)',
                 fontsize=13, y=1.03)
    plt.tight_layout()
    save(fig, 'example_13_multi_run_overlay.png')


# =====================================================================
# Example 14: Energy Landscape
# =====================================================================
def example_14():
    """Show the energy landscape decomposition for a typical scaffold."""
    print("[14/14] Energy Landscape...")

    # Representative parameters
    phi_f, phi_i = 0.40, 0.25
    phi_solid = phi_f + phi_i
    phi_RCP = 0.64
    sigma_0 = 0.01

    # Cell stress (at full maturity/bridging)
    n_dens = 1280 / 640000.0
    k_sub = np.pi * 10.0 * 10.0 / (1.0 - 0.45**2)
    F_cell = 25.0 * (k_sub / (k_sub + 375.0)) * (1.0 / 1.1)
    sigma_cell = n_dens * F_cell

    x_f0 = phi_f / phi_solid

    # Compaction coordinate xi in [0, xi_max]
    xi_max = 1.0 - phi_f / (x_f0 * 0.99)  # avoid singularity
    xi = np.linspace(0, min(xi_max, 0.6), 300)
    xi_pos = xi[xi > 0]

    # Energy terms (normalized by sigma_cell)
    # G_cell: logarithmic attraction
    G_cell = np.where(xi > 0, np.log(1.0 - xi), 0.0)

    # G_elastic: parabolic barrier above jamming onset
    xi_c = 1.0 - phi_f / (x_f0 * phi_RCP)
    G_elastic = np.where(xi > xi_c, (sigma_0 / sigma_cell) * 0.5 * (xi - xi_c)**2, 0.0)

    # G_yield: yield barrier near jamming
    phi_J = 0.92 * phi_RCP
    xi_J = max(0.0, 1.0 - phi_f / (x_f0 * phi_J))
    c_yield = 0.25
    G_yield = np.where(xi > xi_J, c_yield * (sigma_0 / sigma_cell) * (xi - xi_J), 0.0)

    # G_void: osmotic penalty
    Pi_osm = 3.0
    x_f_xi = x_f0 * (1.0 - xi)
    pv_f = 1.0 - phi_f / np.maximum(x_f_xi, 1e-6)
    pv_f0 = 1.0 - phi_f / x_f0
    G_void = Pi_osm * (pv_f - pv_f0)**2 * np.maximum(x_f_xi, 0)

    # G_inert: frustration
    k_frust = 1.5
    G_inert = k_frust * (phi_i / phi_f) * xi**2

    # G_surface: interfacial
    E_mismatch = 0.3
    gamma_fi = 2.5 * (0.3 + E_mismatch)
    G_surface = gamma_fi * (np.sqrt(np.maximum(1.0 - xi, 0)) - 1.0 + phi_i * xi / 2.0)

    G_total = G_cell + G_elastic + G_yield + G_void + G_inert + G_surface

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Individual terms
    ax1.plot(xi, G_cell, 'C1-', lw=2, label='$G_{cell}$ (traction)')
    ax1.plot(xi, G_elastic, 'C0-', lw=2, label='$G_{elastic}$ (Hertz)')
    ax1.plot(xi, G_yield, 'C3--', lw=1.5, label='$G_{yield}$ (friction)')
    ax1.plot(xi, G_void, 'C4:', lw=1.5, label='$G_{void}$ (osmotic)')
    ax1.plot(xi, G_inert, 'C5-.', lw=1.5, label='$G_{inert}$ (frustration)')
    ax1.plot(xi, G_surface, 'C6:', lw=1.5, label='$G_{surface}$ (interface)')
    ax1.axhline(0, color='k', lw=0.5)
    ax1.set_xlabel(r'Compaction coordinate $\xi$')
    ax1.set_ylabel(r'$\tilde{G}$ (normalized energy)')
    ax1.set_title('(a) Energy Decomposition')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # (b) Total energy landscape
    ax2.plot(xi, G_total, 'k-', lw=3, label='$G_{total}$')
    ax2.fill_between(xi, G_total, alpha=0.1, color='k')
    # Mark equilibrium
    if len(G_total) > 2:
        eq_idx = np.argmin(G_total[1:]) + 1  # skip xi=0
        if eq_idx > 0 and eq_idx < len(xi) - 1:
            ax2.plot(xi[eq_idx], G_total[eq_idx], 'r*', ms=15, zorder=5,
                     label=f'Equilibrium $\\xi^*$ = {xi[eq_idx]:.3f}')
    ax2.axhline(0, color='grey', lw=0.5)
    if xi_c > 0:
        ax2.axvline(xi_c, color='C0', ls=':', alpha=0.5, label=f'Jamming onset $\\xi_c$ = {xi_c:.3f}')
    ax2.set_xlabel(r'Compaction coordinate $\xi$')
    ax2.set_ylabel(r'$\tilde{G}_{total}$')
    ax2.set_title('(b) Total Free Energy Landscape')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Example 14: Energy Landscape — Thermodynamic View of Compaction',
                 fontsize=14, y=1.02)
    save(fig, 'example_14_energy_landscape.png')


# =====================================================================
# Main
# =====================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  Mean Field Modeling Guide — Generating Example Figures")
    print("=" * 60)

    # Synthetic examples (always work)
    example_01()
    example_02()
    example_03()
    example_04()
    example_05()
    example_06()
    example_07()
    maturity, bridge_frac, bridge_ramp = example_08()
    example_09(maturity, bridge_frac, bridge_ramp)
    example_10()

    # Real DEM data examples (require results/LHC/)
    print("\n--- Real DEM Data Examples ---")
    example_11()
    example_12()
    example_13()
    example_14()

    print("\n" + "=" * 60)
    print(f"  All figures saved to: {OUTDIR}/")
    print("=" * 60)
