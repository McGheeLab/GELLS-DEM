#!/usr/bin/env python3
"""
JKR contact-clipping test: physically accurate adhesive elastic contact.

JKR (Johnson-Kendall-Roberts) contact gives a larger contact patch than Hertz
because adhesion pulls the surfaces together. The contact radius 'a' determines
the flat face size for rendering via SDF half-plane clipping.

For soft hydrogels (E ~ 1 kPa, W ~ 0.001-0.01 J/m²), the Tabor parameter
μ_T = (R W² / E*² z0³)^(1/3) >> 1, so JKR is the correct model (not DMT).

No visual amplification — shows true-scale deformation.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import os

os.makedirs('results', exist_ok=True)

# ---------------------------------------------------------------------------
# JKR contact model
# ---------------------------------------------------------------------------

def jkr_contact_radius(F_ext, R_eff, E_star, W):
    """
    JKR contact radius for applied force F_ext (nN).

    a³ = (R* / E*) [F + 3πWR* + √(6πWR*F + (3πWR*)²)]

    Units: F_ext [nN], R_eff [µm], E_star [Pa], W [J/m²]
    Returns: a [µm]
    """
    # Convert to consistent µm-nN system:
    #   E_star [Pa] = [N/m²] = [nN/µm²] × 1e-3  →  E_star_sim = E_star * 1e-3 [nN/µm²]
    #   W [J/m²] = [N/m] = [nN/µm] × 1e-3  →  W_sim = W * 1e3 [nN/µm]  NO...
    # Let's be careful:
    #   1 J/m² = 1 N/m = 10⁶ nN / 10⁶ µm = 1 nN/µm
    # So W [J/m²] = W [nN/µm] numerically.
    #   1 Pa = 1 N/m² = 1 nN / (10⁶ µm)² × 10⁹ = 10⁻³ nN/µm²
    # So E*_sim [nN/µm²] = E_star [Pa] × 1e-3

    E_s = E_star * 1e-3   # nN/µm²
    W_s = W               # nN/µm  (1 J/m² = 1 nN/µm)

    term1 = 3.0 * np.pi * W_s * R_eff                      # nN
    inner = 6.0 * np.pi * W_s * R_eff * F_ext + term1**2   # nN²
    if inner < 0:
        inner = 0.0
    a_cubed = (R_eff / E_s) * (F_ext + term1 + np.sqrt(inner))  # µm³
    if a_cubed < 0:
        return 0.0
    return a_cubed ** (1.0 / 3.0)  # µm


def jkr_overlap(a, R_eff, E_star, W):
    """
    JKR overlap (indentation) given contact radius a.

    δ = a²/R* - √(2πW a / E*)

    Returns δ [µm]. Can be negative (adhesive neck stretching).
    """
    E_s = E_star * 1e-3   # nN/µm²
    W_s = W               # nN/µm
    if a < 1e-12:
        return 0.0
    return a**2 / R_eff - np.sqrt(2.0 * np.pi * W_s * a / E_s)


def jkr_force_from_overlap(delta, R_eff, E_star, W):
    """
    Given overlap δ [µm], find the JKR force F [nN].

    Solve iteratively: given δ, find a from δ = a²/R* - √(2πWa/E*),
    then F = (4E*a³)/(3R*) - √(8πWE*a³).

    For efficiency, use Newton-Raphson on a.
    """
    E_s = E_star * 1e-3   # nN/µm²
    W_s = W               # nN/µm

    if delta <= 0 and W_s < 1e-15:
        return 0.0, 0.0  # no contact, no adhesion

    # Initial guess: Hertz contact radius
    if delta > 0:
        a = np.sqrt(R_eff * delta)
    else:
        # Adhesive: start from pull-off contact radius
        a = (6.0 * np.pi * W_s * R_eff**2 / E_s) ** (1.0 / 3.0)

    # Newton-Raphson to find a from δ equation
    for _ in range(30):
        if a < 1e-12:
            a = 1e-6
        sqrt_term = np.sqrt(2.0 * np.pi * W_s * a / E_s) if a > 0 else 0.0
        f = a**2 / R_eff - sqrt_term - delta
        # df/da = 2a/R* - √(πW/(2E*a))
        if a > 1e-12:
            dfda = 2.0 * a / R_eff - np.sqrt(np.pi * W_s / (2.0 * E_s * a))
        else:
            dfda = 2.0 * a / R_eff
        if abs(dfda) < 1e-30:
            break
        a_new = a - f / dfda
        if a_new < 0:
            a_new = a * 0.5
        if abs(a_new - a) < 1e-8:
            a = a_new
            break
        a = a_new

    if a < 1e-10:
        return 0.0, 0.0

    # JKR force from contact radius
    F = (4.0 / 3.0) * E_s * a**3 / R_eff - np.sqrt(8.0 * np.pi * W_s * E_s * a**3)
    return F, a   # nN, µm


def jkr_pulloff_force(R_eff, W):
    """Pull-off force magnitude: F_po = (3/2)πWR*  [nN]."""
    W_s = W   # nN/µm
    return 1.5 * np.pi * W_s * R_eff


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_compression_test():
    # --- Material / geometry ---
    N = 9
    R = 30.0          # µm
    E = 0.2           # kPa  (soft hydrogel, e.g. Matrigel-class)
    nu = 0.49
    E_star = E * 1e3 / (2.0 * (1.0 - nu**2))   # Pa, reduced modulus for equal spheres
    W = 0.02          # J/m²  (adhesive hydrogel, collagen-coated)

    print(f"E* = {E_star:.1f} Pa,  W = {W} J/m²")
    print(f"Tabor parameter estimate: μ_T ~ {(R*1e-6 * W**2 / (E_star**2 * (0.4e-9)**3))**(1/3):.1f}")
    print(f"Pull-off force: {jkr_pulloff_force(R/2, W):.2f} nN")

    # Verify JKR at a test overlap
    F_test, a_test = jkr_force_from_overlap(2.0, R/2, E_star, W)
    print(f"JKR at δ=2µm: F={F_test:.2f} nN, a={a_test:.2f} µm (a/R={a_test/R:.3f})")
    F_test, a_test = jkr_force_from_overlap(5.0, R/2, E_star, W)
    print(f"JKR at δ=5µm: F={F_test:.2f} nN, a={a_test:.2f} µm (a/R={a_test/R:.3f})")

    # --- Dynamics ---
    gamma = 3.0        # translational drag  (nN·h/µm)
    dt = 0.25          # h  (smaller for stability)
    n_steps = 200
    wall_speed = 0.3   # µm/h per wall (all 4 closing)
    v_cap = 20.0       # µm/h velocity cap

    # --- Initial box (3×3 grid, particles nearly touching) ---
    L0 = 195.0
    spacing = 65.0    # center-to-center (2R=60, so 5µm gap)
    offset = (L0 - 2.0 * spacing) / 2.0
    x = np.array([offset + c * spacing for _ in range(3) for c in range(3)])
    y = np.array([offset + r * spacing for r in range(3) for _ in range(3)])

    # Wall positions
    wL, wR, wB, wT = 0.0, L0, 0.0, L0

    # History
    hist_t, hist_phi, hist_nc, hist_max_a = [], [], [], []
    frames = []
    render_res = 600

    for step in range(n_steps):
        t = step * dt

        # --- Move walls ---
        wL = wall_speed * t
        wR = L0 - wall_speed * t
        wB = wall_speed * t
        wT = L0 - wall_speed * t
        Lx = wR - wL
        Ly = wT - wB

        # --- Contact detection + forces ---
        fx = np.zeros(N)
        fy = np.zeros(N)
        # Store contacts for rendering: (nx, ny, clip_dist_from_center)
        contacts = [[] for _ in range(N)]
        max_a = 0.0
        nc = 0

        # Particle–particle
        for i in range(N):
            for j in range(i + 1, N):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.sqrt(dx**2 + dy**2)
                overlap = 2.0 * R - dist

                # JKR has adhesive range beyond contact; check within R
                if overlap > -R * 0.1:   # allow small adhesive gap
                    R_eff = R / 2.0
                    if overlap > 0:
                        F, a = jkr_force_from_overlap(overlap, R_eff, E_star, W)
                    else:
                        # Small gap — check if adhesion pulls them in
                        F, a = jkr_force_from_overlap(0.0, R_eff, E_star, W)
                        if F >= 0:  # no adhesive pull
                            continue

                    if a < 1e-6:
                        continue

                    nx_c = dx / dist
                    ny_c = dy / dist

                    # Force: positive = repulsive (pushes apart)
                    fx[i] -= F * nx_c
                    fy[i] -= F * ny_c
                    fx[j] += F * nx_c
                    fy[j] += F * ny_c

                    # Contact clipping: flat face at distance (R - a²/(2R)) from center
                    # For JKR, the half-width of the contact circle projected onto
                    # the centre-centre axis gives the clip distance.
                    # Geometric: clip_d = R - δ/2 for equal radii
                    # But JKR contact radius a gives a better measure:
                    # The contact patch half-chord = a
                    # Clip distance from center = sqrt(R² - a²) ≈ R - a²/(2R)
                    clip_d = np.sqrt(max(R**2 - a**2, 0.01))
                    contacts[i].append((nx_c, ny_c, clip_d))
                    contacts[j].append((-nx_c, -ny_c, clip_d))
                    max_a = max(max_a, a)
                    nc += 1

        # Particle–wall  (rigid wall: particle takes all deformation)
        for i in range(N):
            for d_val, nx_c, ny_c, f_sign_x, f_sign_y in [
                (x[i] - wL,  -1, 0,  1, 0),   # left wall
                (wR - x[i],   1, 0, -1, 0),   # right wall
                (y[i] - wB,   0,-1,  0, 1),   # bottom wall
                (wT - y[i],   0, 1,  0,-1),   # top wall
            ]:
                overlap = R - d_val
                if overlap > 0:
                    # Wall is rigid, so R_eff = R (flat surface)
                    F, a = jkr_force_from_overlap(overlap, R, E_star, W)
                    fx[i] += f_sign_x * F
                    fy[i] += f_sign_y * F
                    # Clip at wall position (rigid wall doesn't deform)
                    contacts[i].append((nx_c, ny_c, d_val))
                    max_a = max(max_a, a)
                    nc += 1

        # --- Overdamped integration (no position clamp — walls enforce via force) ---
        vx = fx / gamma
        vy = fy / gamma
        for i in range(N):
            v = np.sqrt(vx[i]**2 + vy[i]**2)
            if v > v_cap:
                vx[i] *= v_cap / v
                vy[i] *= v_cap / v
        x += vx * dt
        y += vy * dt

        # --- Metrics ---
        phi = N * np.pi * R**2 / (Lx * Ly)
        hist_t.append(t)
        hist_phi.append(phi)
        hist_nc.append(nc)
        hist_max_a.append(max_a)

        # --- Render (every 2nd step) ---
        if step % 2 != 0 and step != n_steps - 1:
            continue
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                        gridspec_kw={'width_ratios': [1.2, 1]})
        margin = 10.0
        # Fixed rendering window (prevents zoom changes between frames)
        gx = np.linspace(-margin, L0 + margin, render_res)
        gy = np.linspace(-margin, L0 + margin, render_res)
        GX, GY = np.meshgrid(gx, gy)

        # Render each particle individually with its own clipped SDF
        # Wall SDF for masking
        wall_sdf = np.maximum(
            np.maximum(wL - GX, GX - wR),
            np.maximum(wB - GY, GY - wT))

        for i in range(N):
            dxg = GX - x[i]
            dyg = GY - y[i]
            sdf = np.sqrt(dxg**2 + dyg**2) - R

            for (nxc, nyc, cd) in contacts[i]:
                plane = dxg * nxc + dyg * nyc - cd
                sdf = np.maximum(sdf, plane)

            # Mask outside walls
            sdf = np.maximum(sdf, wall_sdf)

            ax1.contourf(GX, GY, sdf, levels=[-1e6, 0], colors=['#4878a8'])
            ax1.contour(GX, GY, sdf, levels=[0], colors=['#1a2744'], linewidths=1.0)
        ax1.plot([wL, wR, wR, wL, wL], [wB, wB, wT, wT, wB], 'k-', lw=2.5)
        ax1.set_xlim(-margin, L0 + margin)
        ax1.set_ylim(-margin, L0 + margin)
        ax1.set_aspect('equal')
        ax1.set_xlabel('x (µm)')
        ax1.set_ylabel('y (µm)')
        ax1.set_title(f't={t:.1f}h   φ={phi:.2f}   {nc} contacts   '
                       f'max a={max_a:.1f} µm  (a/R={max_a/R:.2f})',
                       fontsize=11)

        # --- History panel (fixed axis ranges to prevent layout shifts) ---
        t_total = n_steps * dt
        ax2.plot(hist_t, hist_max_a, 'b-', lw=2, label='max a (µm)')
        ax2.set_xlabel('Time (h)')
        ax2.set_ylabel('max contact radius a (µm)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2.set_xlim(0, t_total)
        ax2.set_ylim(0, R * 0.6)
        ax2b = ax2.twinx()
        ax2b.plot(hist_t, hist_phi, 'r--', lw=2, label='φ')
        ax2b.set_ylabel('φ solid', color='r')
        ax2b.tick_params(axis='y', labelcolor='r')
        ax2b.set_ylim(0.6, 1.0)
        ax2.set_title('JKR Contact-Clipping Compression', fontsize=11)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3].copy()
        frames.append(img)
        plt.close(fig)

        if step % 10 == 0:
            print(f"  step {step:3d}  t={t:5.1f}h  φ={phi:.3f}  "
                  f"nc={nc}  max_a={max_a:.2f}µm  (a/R={max_a/R:.3f})")

    imageio.mimsave('results/lsdem_jkr_clipping.gif', frames, fps=8, loop=0)
    imageio.imwrite('results/lsdem_jkr_final.png', frames[-1])
    print(f"\nSaved {len(frames)} frames → results/lsdem_jkr_clipping.gif")
    print(f"Final frame → results/lsdem_jkr_final.png")


if __name__ == '__main__':
    run_compression_test()
