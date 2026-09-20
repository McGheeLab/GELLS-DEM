"""
Reference (pure-Python) implementations of the GELS loops — V2.7 code, verbatim.
=================================================================================

This module holds the per-pair / per-cell / per-voxel Python loops that made up
the V2.7 engine: force computation, the cell state machine and bridge
formation, overlap resolution, phase-field rendering, metrics and packing
relaxation. They were moved here unchanged in V3.0 Phase 0 so that

  * they remain the **oracle** every compiled kernel in ``gels.kernels`` is
    regression-tested against (``tests/test_*_vs_reference.py``), and
  * LS-DEM deformable granules (``Params.deformable_enabled``), which are
    deferred from the kernel work, keep running exactly as before.

``gels.engine`` keeps same-named dispatcher wrappers that forward here (or to
the kernels once they exist), so the public API is unchanged. Physics changes
in the V3.0 plan (species / functionalization rules) are applied to these
loops first, then mirrored in the kernels.

Do not import this module directly for normal use; call the ``gels.engine``
entry points.
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label

from gels.engine import *  # noqa: F401,F403  (public helpers, leaf njit functions, Params, GranuleSystem, CellState, ...)
from gels.materials import traction_gain, blocker_factor, law_code, rule_code  # noqa: E402
from gels.kernels.percolation import graph_percolation_metrics  # noqa: E402
from gels.laguerre import laguerre_metrics  # noqa: E402
from gels.stress import stress_metrics  # noqa: E402  # V3.7
from gels.pore import pore_metrics  # noqa: E402
from gels.engine import (  # noqa: E402  private helpers used by the moved loops
    _angular_distance_to_target_2d,
    _angular_distance_to_target_3d,
    _apply_clips_2d,
    _apply_clips_3d,
    _cell_world_pos_2d,
    _cell_world_pos_3d,
    _compute_gap,
    _world_to_body,
)


def mc_dem_correction(contacts, gs, p, E_star_gg, F):
    """Apply MC-DEM multi-contact stiffening correction (Giannis et al. 2021).

    When a soft particle has multiple contacts, each contact sees a stiffer
    response due to volumetric confinement.  The correction factor per particle:

        ε_V,i = Σ_c δ_c / (2 R_i)           volumetric overlap strain
        κ_i   = 1 + ν/(1-2ν) · ε_V,i        confinement multiplier

    For each contact between i,j the Hertz repulsion is scaled by
    κ_ij = (κ_i + κ_j)/2.  Only the repulsive (Hertz) component is scaled;
    adhesion and friction are left unchanged.

    Works for both 2D (F shape N×2) and 3D (F shape N×3).
    """
    N = gs.N
    if len(contacts) == 0:
        return
    ndim = F.shape[1]  # 2 or 3

    # Per-particle volumetric overlap strain
    eps_V = np.zeros(N)
    for c in contacts:
        delta = c['overlap']
        if delta > 0:
            eps_V[c['i']] += delta / (2.0 * gs.r[c['i']])
            eps_V[c['j']] += delta / (2.0 * gs.r[c['j']])

    # Confinement correction factor: κ_i = 1 + ν_i/(1−2ν_i) · ε_V,i
    # (per-granule Poisson ratio from the species table, V3.0)
    nu = gs.nu_gran[:N]
    c_mc = nu / np.maximum(1.0 - 2.0 * nu, 0.01)
    kappa = np.minimum(1.0 + c_mc * eps_V, p.mc_dem_kappa_max)

    # Apply correction to Hertzian component of each contact
    for c in contacts:
        delta = c['overlap']
        if delta <= 0:
            continue
        i, j = c['i'], c['j']
        kappa_ij = 0.5 * (kappa[i] + kappa[j])
        if kappa_ij < 1.001:
            continue
        # Per-pair E* recorded by the force loop (V3.0); E_star_gg is the
        # fallback for records without it (deferred LS-DEM branch).
        Fc_hertz = hertz_contact_force(c.get('E_star', E_star_gg), c['R_eff'], delta)
        dF = (kappa_ij - 1.0) * Fc_hertz
        if ndim == 3:
            n = np.array([c['nx'], c['ny'], c['nz']])
        else:
            n = np.array([c['nx'], c['ny']])
        F[i] -= dF * n
        F[j] += dF * n
        # Update stored contact force for visualization
        c['F_normal'] += dF
        c['kappa'] = kappa_ij


def update_cell_state(gs: GranuleSystem, p: Params, t: float, rng=None):
    """
    Advance per-granule cell state for current simulation time.

    Timeline on each functional granule:
      t < t_attach_onset:          cells sit as spheres, no attachment
      t ≥ t_attach_onset:          cells attach (sigmoidal, half-time t_attach_half)
      after attachment:             cells spread (sphere → ellipsoid over t_spread_duration)
      during/after spreading:       focal adhesions mature at fa_maturation_rate
      if spread cells overcrowd:    excess cells crawl on top of others

    Stiffness-dependent spreading: stiffer substrates → faster spreading
    (motor-clutch effect on cell mechanotransduction).
    """
    # Stiffness modulates spreading speed — per host granule E, ν (V3.0)
    a_cell = p.cell_diameter / 2.0
    k_opt = p.n_clutches * p.k_clutch

    for i in range(gs.N):
        if not gs.adhesive_mask[i]:
            continue
        E_i = float(gs.E_gran[i])
        nu_i = float(gs.nu_gran[i])
        k_sub = np.pi * E_i * a_cell / (1.0 - nu_i ** 2)
        stiffness_factor = k_sub / (k_sub + k_opt)  # 0 on very soft, ~1 on stiff
        # Effective spread duration (faster on stiffer substrates)
        eff_spread_dur = p.t_spread_duration / max(0.2, stiffness_factor)

        # V3.2: per-granule clock offset, so the bed does not mature in lockstep.
        # Zero unless cells.seeding.clock_jitter_h is set, and t + 0.0 == t.
        t_i = t + gs.cell_clock_offset[i]

        # ── Attachment (sigmoidal kinetics or instant) ──
        if t_i >= p.t_attach_onset:
            if p.t_attach_half <= 0.01:
                # Instant full attachment
                frac = 1.0
            else:
                tau = t_i - p.t_attach_onset
                frac = 1.0 / (1.0 + np.exp(
                    -3.0 * (tau - p.t_attach_half) / max(0.1, p.t_attach_half)))
            gs.n_attached[i] = gs.n_cells[i] * frac
        else:
            gs.n_attached[i] = 0.0

        # ── Spreading (linear ramp after attachment) ──
        if gs.n_attached[i] > 0.5:
            t_since = max(0.0, t_i - p.t_attach_onset - p.t_attach_half)
            gs.spread_fraction[i] = min(1.0, t_since / eff_spread_dur)
        else:
            gs.spread_fraction[i] = 0.0

        # ── Focal adhesion maturation ──
        if gs.spread_fraction[i] > 0.1:
            t_since_spread = max(0.0, t_i - (p.t_attach_onset + p.t_attach_half))
            gs.fa_maturity[i] = min(1.0, t_since_spread * p.fa_maturation_rate)
        else:
            gs.fa_maturity[i] = 0.0

        # ── Overcrowding check (capacity scales with coverage f, V3.0) ──
        A_cell = cell_projected_area(
            gs.spread_fraction[i], p.cell_diameter, p.cell_height_spread)
        cap = cell_capacity(gs, i, p, A_cell)
        gs.n_overcrowded[i] = max(0.0, gs.n_attached[i] - cap)

    # ── V3.1: cell division (before the per-cell pass; the cell arrays may grow) ──
    # V3.2: the cycle clock has to be advanced first -- nothing did that before.
    from gels.division import age_cells, divide_cells
    age_cells(gs, p)
    divide_cells(gs, p, rng, t)

    # ── V1.5: Per-cell state tracking ──
    _update_individual_cells(gs, p, rng)


def _update_individual_cells(gs: GranuleSystem, p: Params, rng=None):
    """Map per-granule aggregate cell state to individual cell states.

    Committed BRIDGING cells are preserved — their bridge_age is incremented
    and they transition to SENESCENT after sustained load (bridge_senescence_time).
    New bridges are initiated probabilistically during force computation.

    Also applies random-walk migration for mobile cells (ATTACHED, SPREADING,
    PROLIFERATING) on the granule surface.  BRIDGING and SENESCENT cells
    do not migrate.
    """
    mobile_states = {int(CellState.ATTACHED), int(CellState.SPREADING),
                     int(CellState.PROLIFERATING)}
    # Traction gain constants (V3.0) for the ligand-scaled lock-in threshold
    _law = law_code(p.traction_f_law)
    _rule = rule_code(p.traction_f_rule)
    _kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0

    _stack_cells = stacking_enabled(p)      # V3.6
    for i in range(gs.N):
        if not gs.adhesive_mask[i]:
            continue
        n_total = gs.cell_offset[i + 1] - gs.cell_offset[i]
        if n_total == 0:
            continue

        n_att = int(round(gs.n_attached[i]))
        n_over = int(round(gs.n_overcrowded[i]))
        sf = gs.spread_fraction[i]
        fa = gs.fa_maturity[i]

        # Compute per-cell contact area from current spread state
        A_cell = cell_projected_area(sf, p.cell_diameter, p.cell_height_spread)
        # V3.6: which storey each cell stands on. CSR rank is seniority, so this
        # reduces EXACTLY to the k >= cap rule the overcrowding test already
        # uses -- layer 0 is the monolayer, layer 1 is the first storey above it.
        # Written whether or not stacking is enabled: nothing reads it when it is
        # off, and the renderer and the snapshots want it either way.
        _cap_g = max(1, int(round(cell_capacity(gs, i, p, A_cell))))

        for k in range(n_total):
            ci = gs.cell_offset[i] + k
            gs.cell_contact_area[ci] = A_cell if k < n_att else 0.0
            gs.cell_layer[ci] = min(k // _cap_g, 127)

            # ── Preserve committed bridges ──
            if gs.cell_state[ci] == int(CellState.BRIDGING):
                gs.cell_bridge_age[ci] += p.dt
                # Stress fiber alignment: cells elongate along bridge axis over time
                gs.cell_alignment[ci] += p.bridge_alignment_rate * p.dt * (
                    1.0 - gs.cell_alignment[ci])
                gs.cell_alignment[ci] = min(gs.cell_alignment[ci], 1.0)
                # Check force magnitude from previous timestep (still in cell_fx/fy/fz)
                F_mag = np.sqrt(gs.cell_fx[ci]**2 + gs.cell_fy[ci]**2
                                + gs.cell_fz[ci]**2)
                # Persistent lock-in: once force exceeds threshold, bridge is
                # permanently locked and immune to senescence (even if force
                # drops later due to rearrangement).  Only gap rupture can
                # break a locked bridge.
                # The sustainable cluster force is linear in bond number
                # (Erdmann & Schwarz 2004), so the lock-in threshold scales
                # with the same ligand gain as the traction (V3.0).
                lock_thr = p.bridge_lock_force_threshold
                if p.bridge_lock_scales_with_ligand:
                    tgt = int(gs.cell_bridge_target[ci])
                    if tgt >= 0:
                        lock_thr = lock_thr * traction_gain(
                            float(gs.f[i]), float(gs.f[tgt]),
                            _law, _rule, p.traction_exponent, _kappa)
                if F_mag >= lock_thr:
                    gs.cell_bridge_locked[ci] = True
                if gs.cell_bridge_locked[ci]:
                    continue  # locked in permanently, skip senescence check
                # Sustained mechanical load without lock-in → senescence
                if gs.cell_bridge_age[ci] >= p.bridge_senescence_time:
                    gs.cell_state[ci] = int(CellState.SENESCENT)
                    gs.cell_bridge_target[ci] = -1
                    gs.cell_bridge_age[ci] = 0.0
                    gs.cell_bridge_locked[ci] = False
                    gs.cell_alignment[ci] = 0.0
                continue  # don't overwrite bridge state

            # ── Directed migration for MIGRATING cells (V2.3) ──
            if gs.cell_state[ci] == int(CellState.MIGRATING):
                tgt = int(gs.cell_bridge_target[ci])
                if tgt < 0 or tgt >= gs.N:
                    # Invalid target — revert
                    gs.cell_state[ci] = int(CellState.PROLIFERATING)
                    gs.cell_bridge_target[ci] = -1
                    continue

                # Abandonment: target out of sensing range?
                gap = _compute_gap(gs, i, tgt, p)
                if gap > p.cell_sense_distance:
                    gs.cell_state[ci] = int(CellState.PROLIFERATING)
                    gs.cell_bridge_target[ci] = -1
                    gs.cell_bridge_age[ci] = 0.0
                    continue

                # Angular distance to contact azimuth
                if gs.mode == "3D":
                    ang_d, tgt_eta, tgt_omega = _angular_distance_to_target_3d(
                        gs, ci, tgt, p)
                else:
                    ang_d, tgt_theta, signed_diff = _angular_distance_to_target_2d(
                        gs, ci, tgt, p)

                # Arrival: close enough to contact point → BRIDGING
                if ang_d < p.bridge_commit_angle:
                    gs.cell_state[ci] = int(CellState.BRIDGING)
                    gs.cell_bridge_age[ci] = 0.0  # reset for force ramp
                    gs.cell_alignment[ci] = p.bridge_alignment_min
                    continue

                # Directed migration step
                directed_speed = p.cell_migration_speed * p.bridge_directed_speed_mult
                r_eff = max(gs.r[i], 1.0)
                step = directed_speed * p.dt / r_eff  # radians per timestep

                if gs.mode == "3D":
                    # Great-circle slerp toward target
                    cell_eta = gs.cell_eta_local[ci]
                    cell_omega = gs.cell_omega_local[ci]
                    p_curr = np.array([np.cos(cell_eta) * np.cos(cell_omega),
                                       np.cos(cell_eta) * np.sin(cell_omega),
                                       np.sin(cell_eta)])
                    p_tgt = np.array([np.cos(tgt_eta) * np.cos(tgt_omega),
                                      np.cos(tgt_eta) * np.sin(tgt_omega),
                                      np.sin(tgt_eta)])
                    if ang_d > 1e-8:
                        frac = min(step, ang_d) / ang_d
                        sin_a = np.sin(ang_d)
                        if sin_a > 1e-12:
                            p_new = (np.sin((1 - frac) * ang_d) * p_curr
                                     + np.sin(frac * ang_d) * p_tgt) / sin_a
                        else:
                            p_new = p_curr
                        new_omega = np.arctan2(p_new[1], p_new[0]) % (2 * np.pi)
                        r_xy = np.sqrt(p_new[0]**2 + p_new[1]**2)
                        new_eta = np.arctan2(p_new[2], r_xy)
                        gs.cell_eta_local[ci] = np.clip(
                            new_eta, -0.85 * np.pi / 2, 0.85 * np.pi / 2)
                        gs.cell_omega_local[ci] = new_omega
                else:
                    # 2D: arc step toward target theta
                    actual_step = min(step, abs(signed_diff))
                    gs.cell_theta_local[ci] += np.sign(signed_diff) * actual_step
                    gs.cell_theta_local[ci] %= (2 * np.pi)

                gs.cell_bridge_age[ci] += p.dt  # track migration time
                continue  # don't overwrite migrating state

            # ── Already senescent stays senescent ──
            if gs.cell_state[ci] == int(CellState.SENESCENT):
                continue

            # ── Normal state assignment (with stacking tolerance) ──
            # Monolayer capacity from overcrowding check
            A_cell_cap = cell_projected_area(sf, p.cell_diameter, p.cell_height_spread)
            cap = cell_capacity(gs, i, p, A_cell_cap)
            effective_cap = int(round(cap * p.cell_stacking_max))

            if k >= effective_cap:
                # Beyond max stacking — immediate senescence
                gs.cell_state[ci] = int(CellState.SENESCENT)
                gs.cell_overcrowd_age[ci] = 0.0
            elif k >= cap and n_over > 0:
                # Overcrowded but within stacking tolerance — delayed senescence
                gs.cell_overcrowd_age[ci] += p.dt
                if gs.cell_overcrowd_age[ci] >= p.overcrowd_senescence_time:
                    gs.cell_state[ci] = int(CellState.SENESCENT)
                    gs.cell_overcrowd_age[ci] = 0.0
                elif _stack_cells:
                    # V3.6: it has a substrate now. Before V3.6 a tolerated
                    # overcrowded cell was FROZEN in whatever state it was
                    # seeded in -- ATTACHED -- so it never spread and never
                    # bridged, and `cell_stacking_max` was a senescence delay
                    # rather than a second storey. Freezing it was the honest
                    # thing while there was nothing for it to stand on. The
                    # overcrowd clock keeps running either way.
                    if sf >= 0.95 and fa >= 0.5:
                        gs.cell_state[ci] = int(CellState.PROLIFERATING)
                    elif sf > 0.0:
                        gs.cell_state[ci] = int(CellState.SPREADING)
                    else:
                        gs.cell_state[ci] = int(CellState.ATTACHED)
                # else: stay in current mobile state (V3.1)
            elif sf >= 0.95 and fa >= 0.5:
                gs.cell_state[ci] = int(CellState.PROLIFERATING)
                gs.cell_overcrowd_age[ci] = 0.0
            elif sf > 0.0:
                gs.cell_state[ci] = int(CellState.SPREADING)
                gs.cell_overcrowd_age[ci] = 0.0
            else:
                gs.cell_state[ci] = int(CellState.ATTACHED)
                gs.cell_overcrowd_age[ci] = 0.0

        # ── Cell migration: PERSISTENT random walk on the granule surface ──
        # V3.8. Until V3.8 this drew N(0, v*dt/r) directly -- a BALLISTIC
        # displacement used as the width of a DIFFUSIVE increment. Variance
        # adds, so the search over a window T went as (v/r)*sqrt(T*dt): it
        # depended on the timestep, and V3.6's substepping (n_sub 129-330 on the
        # flagship preset) suppressed it by sqrt(n_sub) -- measured 11.75x.
        #
        # A random walk composes under subdivision only when sigma ~ sqrt(dt).
        # Here the only random term is the HEADING, which has exactly that
        # scaling, and the displacement is v*dt along it. That gives
        # MSD(t) = 2 d D [t - tau(1 - e^{-t/tau})] with D = v^2 tau / d:
        # ballistic for t << tau, diffusive for t >> tau, and dt-independent.
        if rng is not None and p.cell_migration_speed > 0:
            r_eff = max(gs.r[i], 1.0)
            tau = max(float(getattr(p, 'cell_persistence_time', 0.0)), 0.0)
            step = p.cell_migration_speed * p.dt * crowd_mobility(gs, i, p) / r_eff
            # tau <= 0 means "no persistence": fall back to the uncorrelated
            # walk, which is the tau -> 0 limit of the same process.
            turn = np.sqrt(2.0 * p.dt / tau) if tau > 0 else 0.0
            p_flip = -np.expm1(-p.dt / tau) if tau > 0 else 0.5
            for k in range(n_total):
                ci = gs.cell_offset[i] + k
                if gs.cell_state[ci] not in mobile_states:
                    continue
                if gs.mode == "3D":
                    # the surface is 2D: a heading angle in its tangent plane,
                    # diffusing at 1/tau.
                    if tau > 0:
                        gs.cell_heading[ci] += rng.normal(0, turn)
                    else:
                        gs.cell_heading[ci] = rng.uniform(0.0, 2 * np.pi)
                    h = gs.cell_heading[ci]
                    gs.cell_eta_local[ci] += step * np.cos(h)
                    gs.cell_omega_local[ci] += step * np.sin(h)
                    # Clamp eta to avoid poles
                    gs.cell_eta_local[ci] = np.clip(
                        gs.cell_eta_local[ci], -0.85 * np.pi / 2, 0.85 * np.pi / 2)
                    gs.cell_omega_local[ci] %= (2 * np.pi)
                else:
                    # the surface is the 1D circumference: a telegraph process,
                    # the heading sign flipping at rate 1/tau.
                    if rng.random() < p_flip:
                        gs.cell_heading[ci] = -gs.cell_heading[ci]
                    gs.cell_theta_local[ci] += step * gs.cell_heading[ci]
                    gs.cell_theta_local[ci] %= (2 * np.pi)


def _print_settled_bed(gs, p, extra):
    bed = bed_surface(gs, p)
    V = bed_solid_volume(gs)
    env = bed['bed_envelope_volume']
    phi_bed = V / env if env > 0 else float('nan')
    print(f"  Sedimented bed: height {bed['bed_height_mean']:.0f} um "
          f"(p95 {bed['bed_height_p95']:.0f}), envelope solid fraction {phi_bed:.3f}, "
          f"{extra} post-relax steps")


def _settle_move_2d(gs, N, fx, fy, dt_settle, v_cap, periodic, Lx, Ly, mobile=None, any_fixed=False):
    """Overdamped move with velocity cap + boundary handling of the 2D settle.

    The V2.7 statements verbatim; V3.1 adds immobile granules (restored after
    the move) and returns the largest move for the gravity stop rule.
    """
    f_mag = np.sqrt(fx*fx + fy*fy) + 1e-12
    scale = np.minimum(dt_settle, v_cap / f_mag)
    saved = gs.pos[:N][~mobile].copy() if any_fixed else None
    gs.x[:N] += fx * scale
    gs.y[:N] += fy * scale

    if periodic:
        gs.x[:N] %= Lx
        gs.y[:N] %= Ly
    else:
        rb = gs.r_bound[:N]
        gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
        gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)
    if any_fixed:
        gs.pos[:N][~mobile] = saved
    mv = f_mag * scale
    if any_fixed:
        mv = np.where(mobile, mv, 0.0)
    return float(np.max(mv)) if N > 0 else 0.0


def _settle_move_3d(gs, N, fx, fy, fz, dt_settle, v_cap, periodic, Lx, Ly, Lz,
                    shape_code=0, R_cyl=0.0, cxc=0.0, cyc=0.0, mobile=None, any_fixed=False):
    """Overdamped move with velocity cap + boundary handling of the 3D settle.

    The V2.7 statements verbatim; V3.1 adds the radial clamp of a cylindrical
    container, immobile granules (restored after the move) and returns the
    largest move for the gravity stop rule.
    """
    f_mag = np.sqrt(fx*fx + fy*fy + fz*fz) + 1e-12
    scale = np.minimum(dt_settle, v_cap / f_mag)
    saved = gs.pos[:N][~mobile].copy() if any_fixed else None
    gs.x[:N] += fx * scale
    gs.y[:N] += fy * scale
    gs.z[:N] += fz * scale

    if periodic:
        gs.x[:N] %= Lx
        gs.y[:N] %= Ly
        gs.z[:N] %= Lz
    else:
        rb = gs.r_bound[:N]
        if shape_code == 1:
            dxc = gs.x[:N] - cxc
            dyc = gs.y[:N] - cyc
            rho = np.sqrt(dxc * dxc + dyc * dyc)
            rmax = R_cyl - rb
            over = rho > rmax
            if np.any(over):
                s_r = rmax[over] / rho[over]
                gs.x[:N][over] = cxc + dxc[over] * s_r
                gs.y[:N][over] = cyc + dyc[over] * s_r
        else:
            gs.x[:N] = np.clip(gs.x[:N], rb, Lx - rb)
            gs.y[:N] = np.clip(gs.y[:N], rb, Ly - rb)
        gs.z[:N] = np.clip(gs.z[:N], rb, Lz - rb)
    if any_fixed:
        gs.pos[:N][~mobile] = saved
    mv = f_mag * scale
    if any_fixed:
        mv = np.where(mobile, mv, 0.0)
    return float(np.max(mv)) if N > 0 else 0.0


def _settle_consolidation(gs, p, N, dim, target_r, mean_r_target, k_rep, periodic, L_up):
    """Consolidation setup shared by the 2D/3D reference settles (mirrors kernels/packing._settle)."""
    mode = consolidation_mode(p) if not periodic else 'none'
    use_centre = (not periodic) and mode == 'centre'
    use_gravity = (not periodic) and mode == 'gravity'
    geom = boundary_geometry(p, '3D' if dim == 3 else '2D')
    shape_code = int(geom.shape_code) if not periodic else 0
    fixed = getattr(gs, 'fixed', None)
    if fixed is None:
        mobile = np.ones(N, dtype=bool)
    else:
        mobile = ~np.asarray(fixed[:N], dtype=bool)
    any_fixed = not bool(np.all(mobile))
    if use_gravity:
        H_place = float(getattr(gs, '_H_place', L_up))
        sched = GravityConsolidation(mean_r_target, H_place, k_rep)
        gw = consolidation_weights(target_r[:N], mean_r_target, dim)
    else:
        sched = None
        gw = np.ones(N)
    return dict(mode=mode, use_centre=use_centre, use_gravity=use_gravity, shape_code=shape_code,
                R_cyl=float(geom.R_cyl), cxc=float(geom.cx), cyc=float(geom.cy),
                mobile=mobile, any_fixed=any_fixed, sched=sched, gw=gw)


def _settle_packing_2d(gs, p):
    """Inflate granules from deflated RSA state to target radii (Lubachevsky-Stillinger).

    The system starts with all granules at reduced radii (set by generate_packing).
    Target radii are stored in gs._target_a/b/r.  This function:
      1. Linearly inflates radii from current to target over packing_settle_steps.
      2. At each inflation step, runs packing_relax_substeps of overlap relaxation
         using vectorised Hertz-like repulsion.
      3. For wall BCs, also applies centripetal attraction.
    Result: a jammed packing at the target solid fraction with Z ≈ 3–4.

    Falls back to simple centripetal settle if no deflation was applied.
    """
    _shape_rb = bool(getattr(p, 'packing_shape_contact', False))   # V3.2
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    Lx, Ly = p.Lx, p.Ly
    n_inflate = p.packing_settle_steps
    n_relax = p.packing_relax_substeps
    dt_settle = 0.02
    k_rep = 5.0        # Hertzian repulsion stiffness
    v_cap_frac = 0.3   # max displacement per sub-step = v_cap_frac * mean_r

    # Retrieve target radii stored by generate_packing
    target_a = getattr(gs, '_target_a', gs.a.copy())
    target_b = getattr(gs, '_target_b', gs.b.copy())
    target_r = getattr(gs, '_target_r', gs.r.copy())

    # Compute initial deflation factor from current vs target
    ratio = gs.r[:N] / np.maximum(target_r[:N], 1e-6)
    alpha_start = float(np.median(ratio))
    alpha_start = max(alpha_start, 0.1)

    mean_r_target = float(np.mean(target_r[:N]))
    v_cap = v_cap_frac * mean_r_target

    cs = _settle_consolidation(gs, p, N, 2, target_r, mean_r_target, k_rep, periodic, Ly)
    use_centre, use_gravity = cs['use_centre'], cs['use_gravity']
    mobile, any_fixed, gw, sched = cs['mobile'], cs['any_fixed'], cs['gw'], cs['sched']
    max_disp = np.inf

    print(f"  Settling 2D packing ({n_inflate} inflate × {n_relax} relax, "
          f"α={alpha_start:.2f}→1.00, consolidation {cs['mode']})...")

    cx_dom, cy_dom = Lx / 2, Ly / 2

    for step in range(n_inflate):
        # Quadratic ease-in: spend more time near α=1 where jamming is hard
        t = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t)**2)

        # Inflate radii
        alpha_v = np.where(mobile, alpha, 1.0) if any_fixed else alpha
        gs.a[:N] = target_a[:N] * alpha_v
        gs.b[:N] = target_b[:N] * alpha_v
        gs.r[:N] = target_r[:N] * alpha_v
        gs.r_bound[:N] = granule_bound_radius(gs.a[:N], gs.b[:N], None,
                                              gs.n1[:N], None, False, _shape_rb)

        max_rb = float(np.max(gs.r_bound[:N]))

        # Overlap relaxation sub-loop
        for sub in range(n_relax):
            pos = gs.positions()  # (N, 2)
            fx = np.zeros(N)
            fy = np.zeros(N)

            # Wall BCs: centripetal attraction (decays over inflation) — V2.7 'centre' mode
            if use_centre:
                attract = max(0.3 * (1.0 - t), 0.02)
                dx_c = cx_dom - pos[:, 0]
                dy_c = cy_dom - pos[:, 1]
                d_c = np.sqrt(dx_c**2 + dy_c**2) + 1e-12
                scale = attract * gs.r_bound[:N] / d_c
                fx += scale * dx_c
                fy += scale * dy_c

            # Neighbour search
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly])
            else:
                tree = cKDTree(pos)
            pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

            max_overlap = 0.0
            if len(pairs) > 0:
                pi = pairs[:, 0]
                pj = pairs[:, 1]
                dx = pos[pj, 0] - pos[pi, 0]
                dy = pos[pj, 1] - pos[pi, 1]
                if periodic:
                    dx -= Lx * np.round(dx / Lx)
                    dy -= Ly * np.round(dy / Ly)
                d = np.sqrt(dx*dx + dy*dy) + 1e-12
                overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
                mask = overlap > 0
                if np.any(mask):
                    ov = overlap[mask]
                    max_overlap = float(np.max(ov))
                    inv_d = 1.0 / d[mask]
                    nx = dx[mask] * inv_d
                    ny = dy[mask] * inv_d
                    f = k_rep * ov ** 1.5
                    np.add.at(fx, pi[mask], -f * nx)
                    np.add.at(fy, pi[mask], -f * ny)
                    np.add.at(fx, pj[mask],  f * nx)
                    np.add.at(fy, pj[mask],  f * ny)

            # Overdamped move with velocity cap + boundary handling
            max_disp = _settle_move_2d(gs, N, fx, fy, dt_settle, v_cap, periodic, Lx, Ly,
                                       mobile, any_fixed)

            # Early exit if overlaps are small
            if max_overlap < 0.01 * mean_r_target:
                break

        # Progress
        if step % 100 == 0 or step == n_inflate - 1:
            pos = gs.positions()
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly])
            else:
                tree = cKDTree(pos)
            cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
            n_contacts = 0
            if len(cpairs) > 0:
                pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
                dx_c = pos[pj_c, 0] - pos[pi_c, 0]
                dy_c = pos[pj_c, 1] - pos[pi_c, 1]
                if periodic:
                    dx_c -= Lx * np.round(dx_c / Lx)
                    dy_c -= Ly * np.round(dy_c / Ly)
                d_c = np.sqrt(dx_c**2 + dy_c**2)
                gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
                n_contacts = int(np.sum(gaps < 1.0))
            Z = 2 * n_contacts / max(N, 1)
            print(f"    step {step}/{n_inflate}: α={alpha:.3f}, "
                  f"max_overlap={max_overlap:.1f} µm, Z={Z:.1f}")

    # ── Post-inflation relaxation: resolve remaining deep overlaps ──
    # V3.5: `packing.overlap_tol_model: elastic` replaces this with the penetration at
    # which the DYNAMICS' contact law carries the driving load. The 0.05*r rule is not a
    # mismeasurement, it is the wrong dimension -- 28x too loose in length, 145x in force
    # on a shaped gravity bed. Derived once in `settle_overlap_tolerance` and handed over
    # on `gs`, so the two twins cannot disagree about it.
    overlap_tol = float(getattr(gs, '_overlap_tol', 0.0)) or 0.05 * mean_r_target
    max_post_relax = 3000 if use_gravity else 1000
    max_rb = float(np.max(gs.r_bound[:N]))
    max_disp = np.inf
    for extra in range(max_post_relax):
        pos = gs.positions()
        fx = np.zeros(N)
        fy = np.zeros(N)
        g_k = sched.g(extra, max_disp) if use_gravity else 0.0
        if use_gravity:
            fy += (g_k * gw) * (-1.0)        # V3.1 gravity consolidation (y is up)

        if periodic:
            tree = cKDTree(pos, boxsize=[Lx, Ly])
        else:
            tree = cKDTree(pos)
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

        max_overlap = 0.0
        if len(pairs) > 0:
            pi = pairs[:, 0]
            pj = pairs[:, 1]
            dx = pos[pj, 0] - pos[pi, 0]
            dy = pos[pj, 1] - pos[pi, 1]
            if periodic:
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
            d = np.sqrt(dx*dx + dy*dy) + 1e-12
            overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
            mask = overlap > 0
            if np.any(mask):
                ov = overlap[mask]
                max_overlap = float(np.max(ov))
                inv_d = 1.0 / d[mask]
                nx = dx[mask] * inv_d
                ny = dy[mask] * inv_d
                f = k_rep * ov ** 1.5
                np.add.at(fx, pi[mask], -f * nx)
                np.add.at(fy, pi[mask], -f * ny)
                np.add.at(fx, pj[mask],  f * nx)
                np.add.at(fy, pj[mask],  f * ny)

        if use_gravity:
            settled = sched.converged(g_k, max_overlap, overlap_tol, max_disp)
        else:
            settled = max_overlap < overlap_tol
        if settled:
            break

        max_disp = _settle_move_2d(gs, N, fx, fy, dt_settle, v_cap, periodic, Lx, Ly,
                                   mobile, any_fixed)
    else:
        # V3.5: a 2D packing runs this loop to its cap and never meets its own
        # tolerance -- and said nothing, so the tolerance looked like the thing
        # that stopped the settle when the step budget was. Diagnostic only.
        print(f"    WARNING: post-relax hit {max_post_relax} steps without reaching "
              f"the tolerance: max_overlap={max_overlap:.3g} um (tol={overlap_tol:.3g}). "
              f"The step budget stopped the settle, not force balance.")

    # Final contact count
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[Lx, Ly])
    else:
        tree = cKDTree(pos)
    cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
    n_contacts = 0
    if len(cpairs) > 0:
        pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
        dx_c = pos[pj_c, 0] - pos[pi_c, 0]
        dy_c = pos[pj_c, 1] - pos[pi_c, 1]
        if periodic:
            dx_c -= Lx * np.round(dx_c / Lx)
            dy_c -= Ly * np.round(dy_c / Ly)
        d_c = np.sqrt(dx_c**2 + dy_c**2)
        gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
        n_contacts = int(np.sum(gaps < 1.0))
    Z = 2 * n_contacts / max(N, 1)
    print(f"  Settle done: {n_contacts} contacts, Z={Z:.1f} "
          f"({extra+1 if max_overlap >= overlap_tol else extra} post-relax steps)")
    if use_gravity:
        _print_settled_bed(gs, p, extra)


def _settle_packing_3d(gs, p):
    """Inflate granules from deflated RSA state to target radii (Lubachevsky-Stillinger).

    The system starts with all granules at reduced radii (set by generate_packing_3d).
    Target radii are stored in gs._target_a/b/c/r.  This function:
      1. Linearly inflates radii from current to target over packing_settle_steps.
      2. At each inflation step, runs packing_relax_substeps of overlap relaxation
         using vectorised Hertz-like repulsion.
      3. For wall BCs, also applies centripetal attraction.
    Result: a jammed packing at the target solid fraction with Z ≈ 4–6.
    """
    _shape_rb = bool(getattr(p, 'packing_shape_contact', False))   # V3.2
    N = gs.N
    periodic = (p.boundary_mode == 'periodic')
    Lx, Ly, Lz = p.Lx, p.Ly, p.Lz
    n_inflate = p.packing_settle_steps
    n_relax = p.packing_relax_substeps
    dt_settle = 0.02
    k_rep = 5.0        # Hertzian repulsion stiffness
    v_cap_frac = 0.3   # max displacement per sub-step = v_cap_frac * mean_r

    # Retrieve target radii stored by generate_packing_3d
    target_a = getattr(gs, '_target_a', gs.a.copy())
    target_b = getattr(gs, '_target_b', gs.b.copy())
    target_c = getattr(gs, '_target_c', gs.c.copy())
    target_r = getattr(gs, '_target_r', gs.r.copy())

    # Compute initial deflation factor from current vs target
    ratio = gs.r[:N] / np.maximum(target_r[:N], 1e-6)
    alpha_start = float(np.median(ratio))
    alpha_start = max(alpha_start, 0.1)

    mean_r_target = float(np.mean(target_r[:N]))
    v_cap = v_cap_frac * mean_r_target

    cs = _settle_consolidation(gs, p, N, 3, target_r, mean_r_target, k_rep, periodic, Lz)
    use_centre, use_gravity = cs['use_centre'], cs['use_gravity']
    mobile, any_fixed, gw, sched = cs['mobile'], cs['any_fixed'], cs['gw'], cs['sched']
    shape_code, R_cyl, cxc, cyc = cs['shape_code'], cs['R_cyl'], cs['cxc'], cs['cyc']
    max_disp = np.inf

    print(f"  Settling 3D packing ({n_inflate} inflate × {n_relax} relax, "
          f"α={alpha_start:.2f}→1.00, consolidation {cs['mode']})...")

    cx_dom, cy_dom, cz_dom = Lx / 2, Ly / 2, Lz / 2

    for step in range(n_inflate):
        # Quadratic ease-in: spend more time near α=1 where jamming is hard
        t = (step + 1) / n_inflate
        alpha = alpha_start + (1.0 - alpha_start) * (1.0 - (1.0 - t)**2)

        # Inflate radii
        alpha_v = np.where(mobile, alpha, 1.0) if any_fixed else alpha
        gs.a[:N] = target_a[:N] * alpha_v
        gs.b[:N] = target_b[:N] * alpha_v
        gs.c[:N] = target_c[:N] * alpha_v
        gs.r[:N] = target_r[:N] * alpha_v
        gs.r_bound[:N] = granule_bound_radius(gs.a[:N], gs.b[:N], gs.c[:N],
                                              gs.n1[:N], gs.n2[:N], True, _shape_rb)

        max_rb = float(np.max(gs.r_bound[:N]))

        # Overlap relaxation sub-loop
        for sub in range(n_relax):
            pos = gs.positions()
            fx = np.zeros(N)
            fy = np.zeros(N)
            fz = np.zeros(N)

            # Wall BCs: centripetal attraction (decays over inflation) — V2.7 'centre' mode
            if use_centre:
                attract = max(0.3 * (1.0 - t), 0.02)
                dx_c = cx_dom - pos[:, 0]
                dy_c = cy_dom - pos[:, 1]
                dz_c = cz_dom - pos[:, 2]
                d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2) + 1e-12
                scale = attract * gs.r_bound[:N] / d_c
                fx += scale * dx_c
                fy += scale * dy_c
                fz += scale * dz_c

            # Neighbour search (use r_bound for cutoff — conservative)
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
            else:
                tree = cKDTree(pos)
            pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

            max_overlap = 0.0
            if len(pairs) > 0:
                pi = pairs[:, 0]
                pj = pairs[:, 1]
                dx = pos[pj, 0] - pos[pi, 0]
                dy = pos[pj, 1] - pos[pi, 1]
                dz = pos[pj, 2] - pos[pi, 2]
                if periodic:
                    dx -= Lx * np.round(dx / Lx)
                    dy -= Ly * np.round(dy / Ly)
                    dz -= Lz * np.round(dz / Lz)
                d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
                overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
                mask = overlap > 0
                if np.any(mask):
                    ov = overlap[mask]
                    max_overlap = float(np.max(ov))
                    inv_d = 1.0 / d[mask]
                    nx = dx[mask] * inv_d
                    ny = dy[mask] * inv_d
                    nz = dz[mask] * inv_d
                    f = k_rep * ov ** 1.5
                    np.add.at(fx, pi[mask], -f * nx)
                    np.add.at(fy, pi[mask], -f * ny)
                    np.add.at(fz, pi[mask], -f * nz)
                    np.add.at(fx, pj[mask],  f * nx)
                    np.add.at(fy, pj[mask],  f * ny)
                    np.add.at(fz, pj[mask],  f * nz)

            # Overdamped move with velocity cap + boundary handling
            max_disp = _settle_move_3d(gs, N, fx, fy, fz, dt_settle, v_cap, periodic, Lx, Ly, Lz,
                                       shape_code, R_cyl, cxc, cyc, mobile, any_fixed)

            # Early exit if overlaps are small
            if max_overlap < 0.01 * mean_r_target:
                break

        # Progress
        if step % 100 == 0 or step == n_inflate - 1:
            # Count contacts
            pos = gs.positions()
            if periodic:
                tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
            else:
                tree = cKDTree(pos)
            cpairs = tree.query_pairs(2 * max_rb + 1.0, output_type='ndarray')
            n_contacts = 0
            if len(cpairs) > 0:
                pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
                dx_c = pos[pj_c, 0] - pos[pi_c, 0]
                dy_c = pos[pj_c, 1] - pos[pi_c, 1]
                dz_c = pos[pj_c, 2] - pos[pi_c, 2]
                if periodic:
                    dx_c -= Lx * np.round(dx_c / Lx)
                    dy_c -= Ly * np.round(dy_c / Ly)
                    dz_c -= Lz * np.round(dz_c / Lz)
                d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
                gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
                n_contacts = int(np.sum(gaps < 1.0))
            Z = 2 * n_contacts / max(N, 1)
            print(f"    step {step}/{n_inflate}: α={alpha:.3f}, "
                  f"max_overlap={max_overlap:.1f} µm, Z={Z:.1f}")

    # ── Post-inflation relaxation: resolve remaining deep overlaps ──
    # After inflation reaches α=1.0, keep relaxing until max overlap is
    # below tolerance (5% of mean radius).  This prevents granules from
    # being trapped inside each other at high packing fractions.
    # V3.5: `packing.overlap_tol_model: elastic` replaces this with the penetration at
    # which the DYNAMICS' contact law carries the driving load. The 0.05*r rule is not a
    # mismeasurement, it is the wrong dimension -- 28x too loose in length, 145x in force
    # on a shaped gravity bed. Derived once in `settle_overlap_tolerance` and handed over
    # on `gs`, so the two twins cannot disagree about it.
    overlap_tol = float(getattr(gs, '_overlap_tol', 0.0)) or 0.05 * mean_r_target
    max_post_relax = 3000 if use_gravity else 1000  # safety cap on extra iterations
    max_rb = float(np.max(gs.r_bound[:N]))
    max_disp = np.inf
    for extra in range(max_post_relax):
        pos = gs.positions()
        fx = np.zeros(N)
        fy = np.zeros(N)
        fz = np.zeros(N)
        g_k = sched.g(extra, max_disp) if use_gravity else 0.0
        if use_gravity:
            fz += (g_k * gw) * (-1.0)        # V3.1 gravity consolidation (z is up)

        if periodic:
            tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
        else:
            tree = cKDTree(pos)
        pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

        max_overlap = 0.0
        if len(pairs) > 0:
            pi = pairs[:, 0]
            pj = pairs[:, 1]
            dx = pos[pj, 0] - pos[pi, 0]
            dy = pos[pj, 1] - pos[pi, 1]
            dz = pos[pj, 2] - pos[pi, 2]
            if periodic:
                dx -= Lx * np.round(dx / Lx)
                dy -= Ly * np.round(dy / Ly)
                dz -= Lz * np.round(dz / Lz)
            d = np.sqrt(dx*dx + dy*dy + dz*dz) + 1e-12
            overlap = gs.r_bound[pi] + gs.r_bound[pj] - d
            mask = overlap > 0
            if np.any(mask):
                ov = overlap[mask]
                max_overlap = float(np.max(ov))
                inv_d = 1.0 / d[mask]
                nx = dx[mask] * inv_d
                ny = dy[mask] * inv_d
                nz = dz[mask] * inv_d
                f = k_rep * ov ** 1.5
                np.add.at(fx, pi[mask], -f * nx)
                np.add.at(fy, pi[mask], -f * ny)
                np.add.at(fz, pi[mask], -f * nz)
                np.add.at(fx, pj[mask],  f * nx)
                np.add.at(fy, pj[mask],  f * ny)
                np.add.at(fz, pj[mask],  f * nz)

        if use_gravity:
            settled = sched.converged(g_k, max_overlap, overlap_tol, max_disp)
        else:
            settled = max_overlap < overlap_tol
        if settled:
            if extra > 0:
                print(f"    post-relax: {extra} extra steps, "
                      f"max_overlap={max_overlap:.1f} µm (tol={overlap_tol:.1f})")
            break

        max_disp = _settle_move_3d(gs, N, fx, fy, fz, dt_settle, v_cap, periodic, Lx, Ly, Lz,
                                   shape_code, R_cyl, cxc, cyc, mobile, any_fixed)
    else:
        print(f"    WARNING: post-relax hit {max_post_relax} steps without reaching "
              f"the tolerance: max_overlap={max_overlap:.3g} µm (tol={overlap_tol:.3g}). "
              f"The step budget stopped the settle, not force balance.")
    if use_gravity:
        _print_settled_bed(gs, p, extra)

    # Final contact count
    pos = gs.positions()
    if periodic:
        tree = cKDTree(pos, boxsize=[Lx, Ly, Lz])
    else:
        tree = cKDTree(pos)
    cpairs = tree.query_pairs(2 * float(np.max(gs.r_bound[:N])) + 1.0, output_type='ndarray')
    n_contacts = 0
    if len(cpairs) > 0:
        pi_c, pj_c = cpairs[:, 0], cpairs[:, 1]
        dx_c = pos[pj_c, 0] - pos[pi_c, 0]
        dy_c = pos[pj_c, 1] - pos[pi_c, 1]
        dz_c = pos[pj_c, 2] - pos[pi_c, 2]
        if periodic:
            dx_c -= Lx * np.round(dx_c / Lx)
            dy_c -= Ly * np.round(dy_c / Ly)
            dz_c -= Lz * np.round(dz_c / Lz)
        d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
        gaps = d_c - gs.r_bound[pi_c] - gs.r_bound[pj_c]
        n_contacts = int(np.sum(gaps < 1.0))
    Z = 2 * n_contacts / max(N, 1)
    print(f"  done ({n_contacts} contacts, Z={Z:.1f}/granule)")


def _service_committed_bridges(gs, gi, gj, gap, p, F_cell_i, F_cell_j, nx, ny, nz, F,
                               pair_fac=None, F_up_i=None, F_up_j=None):
    """Re-apply forces for cells already committed to a bridge between gi↔gj.

    F_cell_i / F_cell_j are the per-cell traction capacities of cells hosted
    on gi / gj (each from its host's E, ν, FA maturity and the ligand gain of
    the granule it grips — V3.0).

    For each committed bridging cell, apply force modulated by bridge maturity
    (ramp over bridge_formation_time).  If gap exceeds bridge_break_gap,
    the bridge ruptures and the cell goes senescent.

    V3.1: with ``bridge_force_model = 'hill'`` every bridge of the pair is
    scaled by one force-velocity factor (``pair_fac``, computed once per pair
    by ``bridge_pair_factor``); the constant model passes 1.0 and reproduces
    the V2.7 arithmetic exactly. Each serviced cell records the pair gap and
    the force it applied, which the next step's factor reads.

    Returns (n_committed_i, n_committed_j, pair_factor).
    """
    n_ci = n_cj = 0
    fac = bridge_pair_factor(gs, gi, gj, gap, p) if pair_fac is None else pair_fac

    for ci in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ci] == int(CellState.BRIDGING) and gs.cell_bridge_target[ci] == gj:
            if gap > p.bridge_break_gap:
                # Bridge rupture → senescent (cell was under excessive strain)
                gs.cell_state[ci] = int(CellState.SENESCENT)
                gs.cell_bridge_target[ci] = -1
                gs.cell_bridge_age[ci] = 0.0
                gs.cell_bridge_locked[ci] = False
                gs.cell_alignment[ci] = 0.0
                gs.cell_bridge_gap_prev[ci] = np.nan
                gs.cell_bridge_force[ci] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[ci] / max(0.1, p.bridge_formation_time))
                alignment = max(gs.cell_alignment[ci], p.bridge_alignment_min)
                _src = (F_cell_i if (F_up_i is None or gs.cell_layer[ci] == 0)
                        else F_up_i)      # V3.6: what this cell is standing on
                F_cell = min(_src * maturity * alignment, p.F_max_per_cell) * fac
                gs.cell_bridge_gap_prev[ci] = gap
                gs.cell_bridge_force[ci] = F_cell
                gs.cell_fx[ci] += F_cell * nx
                gs.cell_fy[ci] += F_cell * ny
                gs.cell_fz[ci] += F_cell * nz
                F[gi, 0] += F_cell * nx if F.ndim == 2 else 0
                F[gi, 1] += F_cell * ny if F.ndim == 2 else 0
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gi, 2] += F_cell * nz
                elif F.ndim == 2 and F.shape[1] == 2:
                    pass  # 2D, no z
                n_ci += 1

    for cj in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[cj] == int(CellState.BRIDGING) and gs.cell_bridge_target[cj] == gi:
            if gap > p.bridge_break_gap:
                gs.cell_state[cj] = int(CellState.SENESCENT)
                gs.cell_bridge_target[cj] = -1
                gs.cell_bridge_age[cj] = 0.0
                gs.cell_bridge_locked[cj] = False
                gs.cell_alignment[cj] = 0.0
                gs.cell_bridge_gap_prev[cj] = np.nan
                gs.cell_bridge_force[cj] = 0.0
            else:
                maturity = min(1.0, gs.cell_bridge_age[cj] / max(0.1, p.bridge_formation_time))
                alignment = max(gs.cell_alignment[cj], p.bridge_alignment_min)
                _src = (F_cell_j if (F_up_j is None or gs.cell_layer[ci] == 0)
                        else F_up_j)      # V3.6: what this cell is standing on
                F_cell = min(_src * maturity * alignment, p.F_max_per_cell) * fac
                gs.cell_bridge_gap_prev[cj] = gap
                gs.cell_bridge_force[cj] = F_cell
                gs.cell_fx[cj] -= F_cell * nx
                gs.cell_fy[cj] -= F_cell * ny
                gs.cell_fz[cj] -= F_cell * nz
                F[gj, 0] -= F_cell * nx if F.ndim == 2 else 0
                F[gj, 1] -= F_cell * ny if F.ndim == 2 else 0
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gj, 2] -= F_cell * nz
                elif F.ndim == 2 and F.shape[1] == 2:
                    pass  # 2D, no z
                n_cj += 1

    return n_ci, n_cj, fac


def _classify_bridge_path(gs, p, gi, gj, pos_i, pos_j, gap):
    """Classify the path between functional granules for bridge probability.

    Returns a multiplicative path_factor:
      - Contact (gap ≤ 0): bridge_contact_factor (high)
      - Void (gap > 0, no obstruction): exp(-gap / bridge_decay_length)
      - Inert-blocked (gap > 0): bridge_inert_factor × exp(-gap / ...)
    """
    # Contact: cells crawl directly between touching surfaces
    if gap <= 0:
        return p.bridge_contact_factor

    # Exponential void decay (filopodia reach)
    void_decay = np.exp(-gap / max(p.bridge_decay_length, 0.1))

    # Ray-cast: check if any granule's bounding sphere intersects the
    # line segment between gi and gj centres.
    N = gs.N
    d_vec = pos_j - pos_i
    d_mag = np.linalg.norm(d_vec)
    if d_mag < 1e-12:
        return p.bridge_contact_factor * void_decay

    d_hat = d_vec / d_mag
    ri = gs.r_bound[gi]
    rj = gs.r_bound[gj]

    # Vectorised positions of all granules (minimum-image relative to pos_i)
    if gs.is_3d:
        all_pos = np.column_stack([gs.x[:N], gs.y[:N], gs.z[:N]])
    else:
        all_pos = np.column_stack([gs.x[:N], gs.y[:N]])
    dk = all_pos - pos_i  # vectors from gi to each gk
    if p.boundary_mode == 'periodic':
        Ls = np.array([p.Lx, p.Ly, p.Lz][:dk.shape[1]])
        dk -= Ls * np.round(dk / Ls)

    # Project onto line gi→gj
    t = dk @ d_hat                         # scalar projection for each gk

    # gk must project between the surfaces of gi and gj
    valid = (t > ri * 0.5) & (t < d_mag - rj * 0.5)
    valid[gi] = False
    valid[gj] = False

    if not np.any(valid):
        return void_decay  # void path — no obstruction

    # Perpendicular distance from each valid gk to the line
    proj = np.outer(t[valid], d_hat)       # (K, dim)
    perp = dk[valid] - proj                # (K, dim)
    d_perp = np.sqrt(np.sum(perp * perp, axis=1))

    # Check bounding-sphere intersection
    r_valid = gs.r_bound[:N][valid]
    intersects = d_perp < r_valid

    if not np.any(intersects):
        return void_decay  # void path

    # At least one granule blocks the line-of-sight. The penalty follows the
    # blocker's coverage (V3.0): bare (f=0) → bridge_inert_factor, fully
    # coated (f=1) → none (cells would bridge to it instead, same as void),
    # continuous in between; the worst blocker wins.
    f_block = gs.f[:N][valid][intersects]
    factor = min(blocker_factor(float(fk), p.bridge_inert_factor) for fk in f_block)
    return factor * void_decay


def _attempt_new_bridges(gs, gi, gj, gap, p, rng, F_cell_i, F_cell_j, nx, ny, nz, F,
                         F_up_i=None, F_up_j=None,
                         pair_fac=1.0,
                         n_existing_bridges=0, path_factor=1.0,
                         pos_i=None, pos_j=None):
    """Probabilistically initiate new bridges between granules gi and gj.

    Only cells that are SPREADING or PROLIFERATING with sufficient FA maturity
    can attempt bridging.  Each eligible cell has a Poisson-distributed
    probability of finding and committing to a bridge target per timestep.

    Cell spatial awareness (V2.3): each cell's position on the granule surface
    determines whether it can see the target. Cells on the far side of their
    host granule (hemisphere facing away from the target) cannot bridge.
    The bridge probability decays from the cell's centroid, not the granule
    centre, so cells closer to the target have higher probability.

    When existing bridges are present (n_existing_bridges > 0), non-bridging
    cells can migrate along the established bridge and form additional
    connections, boosted by bridge_secondary_rate_mult.
    """
    bridgeable_states = (int(CellState.SPREADING), int(CellState.PROLIFERATING))
    min_fa = p.min_fa_for_bridge
    is_3d = (nz != 0.0) or (gs.is_3d if hasattr(gs, 'is_3d') else False)

    # Base rate (before cell-specific decay)
    rate = p.bridge_attempt_rate
    if n_existing_bridges > 0:
        rate *= p.bridge_secondary_rate_mult
    p_bridge_base = 1.0 - np.exp(-rate * p.dt)

    # Path classification (contact/void/inert) from granule-level ray-cast
    # is used as a multiplicative modifier on top of cell-specific decay.
    # Extract the obstruction type from path_factor:
    # - path_factor already includes exp(-gap/λ) for void/inert paths
    # - We need to separate the obstruction modifier from the distance decay
    #   so we can replace distance decay with cell-specific distance.
    decay_len = max(p.bridge_decay_length, 0.1)
    if gap <= 0:
        # Contact: use path_factor directly (bridge_contact_factor), no distance decay
        obstruction_mod = path_factor
        use_cell_decay = False
    else:
        # Factor out the granule-level exponential decay to get pure obstruction modifier
        granule_decay = np.exp(-gap / decay_len)
        if granule_decay > 1e-15:
            obstruction_mod = path_factor / granule_decay
        else:
            obstruction_mod = path_factor
        use_cell_decay = True

    # Target granule centre (for cell-to-target distance)
    if pos_j is not None:
        tgt = pos_j
    elif is_3d:
        tgt = np.array([gs.x[gj], gs.y[gj], gs.z[gj]])
    else:
        tgt = np.array([gs.x[gj], gs.y[gj]])
    r_target = gs.r_bound[gj]

    # ── Collect angular positions of existing BRIDGING/MIGRATING cells for
    #    exclusion zone check (cells avoid bridging near existing bridges) ──
    _bridge_states = (int(CellState.BRIDGING), int(CellState.MIGRATING))
    existing_thetas_i = []  # angular positions of bridges on gi→gj
    existing_thetas_j = []  # angular positions of bridges on gj→gi
    for ck in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ck] in _bridge_states and gs.cell_bridge_target[ck] == gj:
            existing_thetas_i.append(gs.cell_theta_local[ck] if not is_3d else
                                     (gs.cell_eta_local[ck], gs.cell_omega_local[ck]))
    for ck in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[ck] in _bridge_states and gs.cell_bridge_target[ck] == gi:
            existing_thetas_j.append(gs.cell_theta_local[ck] if not is_3d else
                                     (gs.cell_eta_local[ck], gs.cell_omega_local[ck]))

    # ── Eligible cells on granule i ──
    for ci in range(gs.cell_offset[gi], gs.cell_offset[gi + 1]):
        if gs.cell_state[ci] not in bridgeable_states:
            continue
        if gs.fa_maturity[gi] < min_fa:
            continue

        # Cell world position
        if is_3d:
            cx, cy, cz = _cell_world_pos_3d(gs, ci)
            cell_pos = np.array([cx, cy, cz])
        else:
            cx, cy = _cell_world_pos_2d(gs, ci)
            cell_pos = np.array([cx, cy])

        # Hemisphere check: cell offset from granule centre dotted with
        # direction to target. Skip if cell faces away.
        d_to_tgt = np.array([nx, ny, nz][:len(cell_pos)])
        cell_offset = cell_pos - (pos_i if pos_i is not None else
                                  np.array([gs.x[gi], gs.y[gi]])
                                  if not is_3d else
                                  np.array([gs.x[gi], gs.y[gi], gs.z[gi]]))
        if np.dot(cell_offset, d_to_tgt) < 0:
            continue  # cell is on far side of granule

        # Bridge exclusion zone: skip if too close to existing bridge/migrating cell
        if p.bridge_exclusion_angle > 0 and existing_thetas_i:
            too_close = False
            if is_3d:
                eta_c, om_c = gs.cell_eta_local[ci], gs.cell_omega_local[ci]
                p_c = np.array([np.cos(eta_c) * np.cos(om_c),
                                np.cos(eta_c) * np.sin(om_c), np.sin(eta_c)])
                for (eta_k, om_k) in existing_thetas_i:
                    p_k = np.array([np.cos(eta_k) * np.cos(om_k),
                                    np.cos(eta_k) * np.sin(om_k), np.sin(eta_k)])
                    ang_sep = np.arccos(np.clip(np.dot(p_c, p_k), -1.0, 1.0))
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            else:
                theta_c = gs.cell_theta_local[ci]
                for theta_k in existing_thetas_i:
                    diff = abs(theta_c - theta_k)
                    ang_sep = min(diff, 2 * np.pi - diff)
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            if too_close:
                continue  # cell would rather crawl around than stack on existing bridge

        # Cell-to-target gap (cell surface → target surface)
        cell_to_tgt_dist = np.linalg.norm(cell_pos - tgt)
        cell_gap = max(cell_to_tgt_dist - r_target, 0.0)

        # Cell-specific probability
        if use_cell_decay:
            cell_decay = np.exp(-cell_gap / decay_len)
            p_bridge = p_bridge_base * obstruction_mod * cell_decay
        else:
            p_bridge = p_bridge_base * obstruction_mod

        if rng.random() < p_bridge:
            gs.cell_bridge_target[ci] = gj
            gs.cell_bridge_age[ci] = 0.0
            # Check if cell is already at the contact azimuth (fast-path)
            if is_3d:
                ang_d, _, _ = _angular_distance_to_target_3d(gs, ci, gj, p)
            else:
                ang_d, _, _ = _angular_distance_to_target_2d(gs, ci, gj, p)
            if ang_d < p.bridge_commit_angle:
                # Already at contact point → skip migration, bridge directly
                gs.cell_state[ci] = int(CellState.BRIDGING)
                gs.cell_alignment[ci] = p.bridge_alignment_min
                maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
                _src = (F_cell_i if (F_up_i is None or gs.cell_layer[ci] == 0)
                        else F_up_i)      # V3.6: what this cell is standing on
                F_cell = min(_src * maturity * p.bridge_alignment_min,
                             p.F_max_per_cell) * pair_fac
                gs.cell_bridge_gap_prev[ci] = gap
                gs.cell_bridge_force[ci] = F_cell
                gs.cell_fx[ci] += F_cell * nx
                gs.cell_fy[ci] += F_cell * ny
                gs.cell_fz[ci] += F_cell * nz
                if F.ndim == 2 and F.shape[1] >= 2:
                    F[gi, 0] += F_cell * nx
                    F[gi, 1] += F_cell * ny
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gi, 2] += F_cell * nz
            else:
                # Cell needs to crawl to contact point first
                gs.cell_state[ci] = int(CellState.MIGRATING)
                gs.cell_alignment[ci] = 0.0

    # ── Eligible cells on granule j (target is gi) ──
    if pos_i is not None:
        tgt_j = pos_i
    elif is_3d:
        tgt_j = np.array([gs.x[gi], gs.y[gi], gs.z[gi]])
    else:
        tgt_j = np.array([gs.x[gi], gs.y[gi]])
    r_target_j = gs.r_bound[gi]

    for cj in range(gs.cell_offset[gj], gs.cell_offset[gj + 1]):
        if gs.cell_state[cj] not in bridgeable_states:
            continue
        if gs.fa_maturity[gj] < min_fa:
            continue

        # Cell world position
        if is_3d:
            cx, cy, cz = _cell_world_pos_3d(gs, cj)
            cell_pos = np.array([cx, cy, cz])
        else:
            cx, cy = _cell_world_pos_2d(gs, cj)
            cell_pos = np.array([cx, cy])

        # Hemisphere check: cell should face toward gi (opposite direction)
        d_to_gi = -np.array([nx, ny, nz][:len(cell_pos)])
        cell_offset = cell_pos - (pos_j if pos_j is not None else
                                  np.array([gs.x[gj], gs.y[gj]])
                                  if not is_3d else
                                  np.array([gs.x[gj], gs.y[gj], gs.z[gj]]))
        if np.dot(cell_offset, d_to_gi) < 0:
            continue  # cell is on far side of granule

        # Bridge exclusion zone: skip if too close to existing bridge/migrating cell
        if p.bridge_exclusion_angle > 0 and existing_thetas_j:
            too_close = False
            if is_3d:
                eta_c, om_c = gs.cell_eta_local[cj], gs.cell_omega_local[cj]
                p_c = np.array([np.cos(eta_c) * np.cos(om_c),
                                np.cos(eta_c) * np.sin(om_c), np.sin(eta_c)])
                for (eta_k, om_k) in existing_thetas_j:
                    p_k = np.array([np.cos(eta_k) * np.cos(om_k),
                                    np.cos(eta_k) * np.sin(om_k), np.sin(eta_k)])
                    ang_sep = np.arccos(np.clip(np.dot(p_c, p_k), -1.0, 1.0))
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            else:
                theta_c = gs.cell_theta_local[cj]
                for theta_k in existing_thetas_j:
                    diff = abs(theta_c - theta_k)
                    ang_sep = min(diff, 2 * np.pi - diff)
                    if ang_sep < p.bridge_exclusion_angle:
                        too_close = True
                        break
            if too_close:
                continue  # cell would rather crawl around than stack on existing bridge

        # Cell-to-target gap
        cell_to_tgt_dist = np.linalg.norm(cell_pos - tgt_j)
        cell_gap = max(cell_to_tgt_dist - r_target_j, 0.0)

        if use_cell_decay:
            cell_decay = np.exp(-cell_gap / decay_len)
            p_bridge = p_bridge_base * obstruction_mod * cell_decay
        else:
            p_bridge = p_bridge_base * obstruction_mod

        if rng.random() < p_bridge:
            gs.cell_bridge_target[cj] = gi
            gs.cell_bridge_age[cj] = 0.0
            if is_3d:
                ang_d, _, _ = _angular_distance_to_target_3d(gs, cj, gi, p)
            else:
                ang_d, _, _ = _angular_distance_to_target_2d(gs, cj, gi, p)
            if ang_d < p.bridge_commit_angle:
                gs.cell_state[cj] = int(CellState.BRIDGING)
                gs.cell_alignment[cj] = p.bridge_alignment_min
                maturity = min(1.0, p.dt / max(0.1, p.bridge_formation_time))
                _src = (F_cell_j if (F_up_j is None or gs.cell_layer[ci] == 0)
                        else F_up_j)      # V3.6: what this cell is standing on
                F_cell = min(_src * maturity * p.bridge_alignment_min,
                             p.F_max_per_cell) * pair_fac
                gs.cell_bridge_gap_prev[cj] = gap
                gs.cell_bridge_force[cj] = F_cell
                gs.cell_fx[cj] -= F_cell * nx
                gs.cell_fy[cj] -= F_cell * ny
                gs.cell_fz[cj] -= F_cell * nz
                if F.ndim == 2 and F.shape[1] >= 2:
                    F[gj, 0] -= F_cell * nx
                    F[gj, 1] -= F_cell * ny
                if F.ndim == 2 and F.shape[1] == 3:
                    F[gj, 2] -= F_cell * nz
            else:
                gs.cell_state[cj] = int(CellState.MIGRATING)
                gs.cell_alignment[cj] = 0.0


def compute_forces(gs: GranuleSystem, p: Params, rng):
    """
    Compute all forces and torques on each granule.
    Returns (F, torques) where F is (N, 2) force array in nN and
    torques is (N,) array in nN·µm.

    Contact forces:
      Hertz repulsion:  F = (4/3) E* √R* δ^{3/2}
      DMT adhesion:     F_adh = 2π W R*  (constant attractive during contact)
      Net normal:       F_n = F_Hertz − F_adh  (can be negative = attractive)

    Tangential friction (area-dependent, hydrogel tribology):
      F_fric = τ₀ · A_contact · tanh(|v_t|/v_ref)  opposing relative sliding
      where A_contact = π R* δ  (Hertzian contact area)

    For superellipses, R* uses local curvature at the contact point.

    Wall uses Hertz (rigid limit):
      E* = E / (1-ν²), R* = R_i (circle) or R_local (superellipse)
    """
    N = gs.N
    F = np.zeros((N, 2))
    torques = np.zeros(N)
    # V3.6: per-granule wall contact stiffness, zeroed every evaluation and
    # read by `contact_stiffness_per_granule`. Stays 0 under periodic
    # boundaries and for a bed that never touches a wall.
    gs.wall_stiffness = np.zeros(N)
    pos = gs.positions()
    contacts = []  # V1.5.2: per-contact data for stress visualization

    # V1.5: Reset per-cell force vectors (but preserve bridge state and targets)
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    # ── Precompute effective moduli (Pa) ──
    nu2 = p.poisson_ratio ** 2
    E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu2))   # granule-granule
    E_star_gw = (p.E_modulus * 1e3) / (1.0 - nu2)            # granule-wall

    # ── V3.0: pair adhesion / friction / E* come from the species tables on
    # gs (pair_W, pair_tau, pair_Estar; wall_W, wall_Estar). E_star_gg above
    # is kept only for the deferred LS-DEM branch.
    _law = law_code(p.traction_f_law)
    _rule = rule_code(p.traction_f_rule)
    _kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
    _adh = adhesion_force_ceiling(gs, p)   # V3.6: sigma x A ceiling, at g = 1
    _stack = stacking_enabled(p)           # V3.6: cells standing on cells
    _mu = float(getattr(p, 'friction_mu', 0.0))     # V3.1 Coulomb friction term
    _cell_adh = float(getattr(p, 'cell_contact_adhesion', 0.0))   # V3.2 cell grip at the contact
    _R_cap = float(getattr(p, 'curvature_R_cap', 0.0))            # V3.2 curvature cap
    _bridge_map = bridge_weight_map(gs, p) if _cell_adh > 0.0 else {}

    # ── Neighbour search ──
    max_r = float(np.max(gs.r_bound))
    cutoff = 2*max_r + p.L_max
    periodic = (p.boundary_mode == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j,0] - pos[i,0]
        dy = pos[j,1] - pos[i,1]
        if periodic:
            dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
        d = np.sqrt(dx*dx + dy*dy)
        if d < 1e-6:
            continue
        nx, ny = dx/d, dy/d

        # ── LS-DEM deformable contact detection (V2.3) ──
        if p.deformable_enabled:
            from gels.lsdem import find_contacts_lsdem_2d
            xj_v = pos[i,0] + dx
            yj_v = pos[i,1] + dy
            lsdem_result = find_contacts_lsdem_2d(
                gs, i, j, pos[i,0], pos[i,1], xj_v, yj_v, p)
            if lsdem_result is not None:
                F[i] += lsdem_result['F_i']
                F[j] += lsdem_result['F_j']
                torques[i] += lsdem_result['tau_i']
                torques[j] += lsdem_result['tau_j']
                if gs.F_eps_accum is not None:
                    gs.F_eps_accum[i] += lsdem_result['F_eps_i']
                    gs.F_eps_accum[j] += lsdem_result['F_eps_j']
                overlap = lsdem_result['overlap_max']
                R_eff_approx = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])

                # JKR adhesion correction — translational only (V2.3)
                # LS-DEM nodes compute pure repulsion; adhesion added here
                # so it does NOT drive deformation DOFs
                W_adh_pair = float(gs.pair_W[gs.species_id[i], gs.species_id[j]])
                if W_adh_pair > 0 and overlap > 0:
                    F_jkr, a_est = jkr_force_from_overlap(
                        overlap, R_eff_approx, E_star_gg, W_adh_pair)
                    F_hertz = hertz_contact_force(E_star_gg, R_eff_approx, overlap)
                    # Adhesion correction is JKR − Hertz (negative = attractive)
                    F_adh_corr = F_jkr - F_hertz
                    F[i] += F_adh_corr * np.array([nx, ny])
                    F[j] -= F_adh_corr * np.array([nx, ny])
                else:
                    a_est = np.sqrt(max(R_eff_approx * overlap, 0.0))

                # Estimate contact radius from overlap for clip rendering
                if a_est > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_est**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_est**2, 0.01))
                    gs.contact_clips[i].append((nx, ny, clip_d_i))
                    gs.contact_clips[j].append((-nx, -ny, clip_d_j))
                contacts.append({
                    'i': i, 'j': j,
                    'cx': (pos[i,0] + xj_v) / 2, 'cy': (pos[i,1] + yj_v) / 2, 'cz': 0.0,
                    'nx': nx, 'ny': ny, 'nz': 0.0,
                    'overlap': overlap, 'R_eff': R_eff_approx,
                    'F_normal': np.sqrt(lsdem_result['F_i'][0]**2 + lsdem_result['F_i'][1]**2),
                    'A_contact': np.pi * a_est**2,
                    'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                })
                in_contact = True
            else:
                in_contact = False
                overlap = 0.0

            # Cell bridging still uses gap-based logic
            if gs.adhesive_mask[i] and gs.adhesive_mask[j] and (gs.n_cells[i] + gs.n_cells[j] > 0):
                if in_contact:
                    gap = -overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0:
                        gap = 0.0
                F_cell_i = motor_clutch_force(
                    gs.E_gran[i], p, gs.fa_maturity[i], nu=gs.nu_gran[i],
                    g=traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[i] if _adh is not None else None))
                F_cell_j = motor_clutch_force(
                    gs.E_gran[j], p, gs.fa_maturity[j], nu=gs.nu_gran[j],
                    g=traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[j] if _adh is not None else None))
                # V3.6: the same cell's traction if it is standing on CELLS rather
                # than on the granule. Both computed properly; the service loop picks
                # per cell by `cell_layer`, because a ratio applied afterwards would
                # be the F(g.h) != F.h trap.
                F_up_i = F_up_j = None
                if _stack:
                    F_up_i = stacked_traction_force(
                        p, gs.fa_maturity[i], traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                        F_adh=(_adh[i] if _adh is not None else None))
                    F_up_j = stacked_traction_force(
                        p, gs.fa_maturity[j], traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                        F_adh=(_adh[j] if _adh is not None else None))
                n_ci, n_cj, _fac = _service_committed_bridges(
                    gs, i, j, gap, p, F_cell_i, F_cell_j, nx, ny, 0.0, F,
                F_up_i=F_up_i, F_up_j=F_up_j)
                n_existing = n_ci + n_cj
                if gap < p.cell_sense_distance:
                    pj_v_2d = pos[i] + np.array([dx, dy])
                    pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v_2d, gap)
                    _attempt_new_bridges(
                        gs, i, j, gap, p, rng, F_cell_i, F_cell_j, nx, ny, 0.0, F,
                    F_up_i=F_up_i, F_up_j=F_up_j,
                        pair_fac=_fac,
                        n_existing_bridges=n_existing, path_factor=pf,
                        pos_i=pos[i], pos_j=pj_v_2d)
            continue  # skip rigid contact path

        # ── Contact detection (rigid, V2.2) ──
        if gs.is_circle:
            # Fast circle path (V1.2 behavior)
            overlap = gs.r[i] + gs.r[j] - d
            if overlap > 0:
                R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                # Contact point at midpoint of overlap along line of centres
                contact_x = pos[i,0] + (gs.r[i] - overlap/2) * nx
                contact_y = pos[i,1] + (gs.r[i] - overlap/2) * ny
                in_contact = True
            else:
                in_contact = False
                overlap = 0.0
                R_eff = 0.0
                contact_x = contact_y = 0.0
        else:
            # Superellipse contact (common normal method)
            # For periodic BCs, use virtual position of j (nearest image)
            xj_v = pos[i,0] + dx  # pos_i + minimum-image displacement
            yj_v = pos[i,1] + dy
            result = find_contact_superellipses(
                pos[i,0], pos[i,1], gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                xj_v, yj_v, gs.a[j], gs.b[j], gs.n_shape[j], gs.theta[j])
            if result is not None:
                in_contact = True
                _, overlap, nx, ny, contact_x, contact_y, R_loc_i, R_loc_j = result
                R_eff = R_loc_i * R_loc_j / (R_loc_i + R_loc_j)
                if _R_cap > 0.0 and R_eff > 0.0:
                    # V3.2: superellipse curvature runs away at a flat face --
                    # 3.5e15 um at n = 3.5 vs 48 for a circle, and F ~ sqrt(R_eff)
                    _lim = _R_cap * min(gs.r[i], gs.r[j])
                    if R_eff > _lim:
                        R_eff = _lim
            else:
                in_contact = False
                overlap = 0.0
                R_eff = 0.0
                contact_x = contact_y = 0.0

        # ── Contact forces: JKR adhesive contact + friction (V2.3) ──
        if in_contact and overlap > 0:
            # Pair-type friction/adhesion parameters
            si = gs.species_id[i]
            sj = gs.species_id[j]
            tau_0 = gs.pair_tau[si, sj]
            W_adh = gs.pair_W[si, sj]
            E_star_pair = gs.pair_Estar[si, sj]

            # JKR force: replaces Hertz + DMT (reduces to Hertz when W=0)
            F_normal, a_contact = jkr_force_from_overlap(
                overlap, R_eff, E_star_pair, W_adh)

            Fnx = F_normal * nx
            Fny = F_normal * ny
            F[i,0] -= Fnx; F[i,1] -= Fny
            F[j,0] += Fnx; F[j,1] += Fny

            # Torque from normal force (off-centre contact)
            if not gs.is_circle:
                rci_x = contact_x - pos[i,0]
                rci_y = contact_y - pos[i,1]
                rcj_x = contact_x - pos[j,0]
                rcj_y = contact_y - pos[j,1]
                torques[i] += rci_x * (-Fny) - rci_y * (-Fnx)
                torques[j] += rcj_x * Fny - rcj_y * Fnx

            # Contact clip planes for rendering (half-plane at flat face)
            if a_contact > 1e-6:
                clip_d_i = np.sqrt(max(gs.r[i]**2 - a_contact**2, 0.01))
                clip_d_j = np.sqrt(max(gs.r[j]**2 - a_contact**2, 0.01))
                gs.contact_clips[i].append((nx, ny, clip_d_i))
                gs.contact_clips[j].append((-nx, -ny, clip_d_j))

            # Tangential friction (JKR contact area: π a²)
            A_contact = np.pi * a_contact**2 if a_contact > 0 else 0.0
            dvx = gs.vx[j] - gs.vx[i]
            dvy = gs.vy[j] - gs.vy[i]
            v_dot_n = dvx * nx + dvy * ny
            vtx = dvx - v_dot_n * nx
            vty = dvy - v_dot_n * ny
            vt_mag = np.sqrt(vtx*vtx + vty*vty)

            if vt_mag > 1e-12:
                F_t_cap = tau_0 * A_contact * 1e-3
                if _mu > 0.0 and F_normal > 0.0:
                    F_t_cap += _mu * F_normal
                if _cell_adh > 0.0:
                    F_t_cap += _cell_adh * _bridge_map.get(
                        (i, j) if i < j else (j, i), 0.0)     # V3.2: the cells' own grip
                F_fric = F_t_cap * np.tanh(vt_mag / p.friction_v_ref)
                tx, ty = vtx / vt_mag, vty / vt_mag
                Ftx, Fty = F_fric * tx, F_fric * ty
                F[i,0] += Ftx; F[i,1] += Fty
                F[j,0] -= Ftx; F[j,1] -= Fty

                if not gs.is_circle:
                    torques[i] += rci_x * Fty - rci_y * Ftx
                    torques[j] += rcj_x * (-Fty) - rcj_y * (-Ftx)

            contacts.append({
                'i': i, 'j': j,
                'cx': contact_x, 'cy': contact_y, 'cz': 0.0,
                'nx': nx, 'ny': ny, 'nz': 0.0,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'a_contact': a_contact,
                'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                'species_i': int(si), 'species_j': int(sj),
                'f_i': float(gs.f[i]), 'f_j': float(gs.f[j]),
                'E_star': float(E_star_pair), 'W': float(W_adh), 'tau_0': float(tau_0),
            })

        # ── Cell bridging (motor-clutch model, functional–functional) ──
        if gs.adhesive_mask[i] and gs.adhesive_mask[j] and (gs.n_cells[i] + gs.n_cells[j] > 0):
            if gs.is_circle:
                gap = d - gs.r[i] - gs.r[j]
            else:
                if in_contact:
                    gap = -overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0:
                        gap = 0.0

            # Motor-clutch force per cell (stiffness + FA maturity)
            F_cell_i = motor_clutch_force(
                gs.E_gran[i], p, gs.fa_maturity[i], nu=gs.nu_gran[i],
                g=traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                F_adh=(_adh[i] if _adh is not None else None))
            F_cell_j = motor_clutch_force(
                gs.E_gran[j], p, gs.fa_maturity[j], nu=gs.nu_gran[j],
                g=traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                F_adh=(_adh[j] if _adh is not None else None))
            # V3.6: the same cell's traction if it is standing on CELLS rather
            # than on the granule. Both computed properly; the service loop picks
            # per cell by `cell_layer`, because a ratio applied afterwards would
            # be the F(g.h) != F.h trap.
            F_up_i = F_up_j = None
            if _stack:
                F_up_i = stacked_traction_force(
                    p, gs.fa_maturity[i], traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[i] if _adh is not None else None))
                F_up_j = stacked_traction_force(
                    p, gs.fa_maturity[j], traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[j] if _adh is not None else None))

            # 1) Service committed bridges (apply force with maturity ramp)
            n_ci, n_cj, _fac = _service_committed_bridges(
                gs, i, j, gap, p, F_cell_i, F_cell_j, nx, ny, 0.0, F,
                F_up_i=F_up_i, F_up_j=F_up_j)
            n_existing = n_ci + n_cj

            # 2) Path-dependent bridge formation (V2.1)
            if gap < p.cell_sense_distance:
                pj_v_2d = pos[i] + np.array([dx, dy])
                pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v_2d, gap)
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_cell_i, F_cell_j, nx, ny, 0.0, F,
                    F_up_i=F_up_i, F_up_j=F_up_j,
                    pair_fac=_fac,
                    n_existing_bridges=n_existing, path_factor=pf,
                    pos_i=pos[i], pos_j=pj_v_2d)

    # ── Wall repulsion: JKR adhesive contact (skip for periodic boundaries) ──
    # Wall is rigid (R_eff = R_granule for flat surface), treated as inert surface.
    if not periodic:
        # Wall = bare surface: per-species W and E* from gs.wall_W / wall_Estar (V3.0)
        _geom = boundary_geometry(p, '2D')
        wall_normals_2d = [
            (0, +1, 0),  # left:   +x normal, axis=0
            (0, -1, 0),  # right:  -x normal, axis=0
            (1, +1, 1),  # bottom: +y normal, axis=1
            (1, -1, 1),  # top:    -y normal, axis=1
        ]
        for i in range(N):
            W_wall = gs.wall_W[gs.species_id[i]]
            E_star_gw = gs.wall_Estar[gs.species_id[i]]
            if gs.is_circle:
                r = gs.r[i]
                _pens = [
                    (r - gs.x[i],            0, +1),   # left
                    (gs.x[i] - (p.Lx - r),   0, -1),   # right
                    (r - gs.y[i],            1, +1),   # bottom
                    (gs.y[i] - (p.Ly - r),   1, -1),   # top
                ]
                if _geom.top_free:
                    _pens = _pens[:3]                  # V3.1 open top
                for pen, axis, sign in _pens:
                    if pen > 0:
                        Fw, a_w = jkr_force_from_overlap(pen, r, E_star_gw, W_wall)
                        # V3.6: wall contact stiffness, dF/d(delta) = 2 E_s a  (nN/um)
                        gs.wall_stiffness[i] += 2.0 * E_star_gw * 1e-3 * a_w
                        F[i, axis] += sign * Fw
                        # Wall clip: distance from centre to wall face
                        if axis == 0:
                            wall_d = abs(gs.x[i]) if sign > 0 else abs(p.Lx - gs.x[i])
                        else:
                            wall_d = abs(gs.y[i]) if sign > 0 else abs(p.Ly - gs.y[i])
                        # Clip normal points TOWARD wall (opposite of force sign)
                        nx_w = -float(sign) if axis == 0 else 0.0
                        ny_w = 0.0 if axis == 0 else -float(sign)
                        gs.contact_clips[i].append((nx_w, ny_w, wall_d))
            else:
                walls = [
                    (0.0,    0, +1),   # left wall at x=0
                    (p.Lx,   0, -1),   # right wall at x=Lx
                    (0.0,    1, +1),   # bottom wall at y=0
                    (p.Ly,   1, -1),   # top wall at y=Ly
                ]
                if _geom.top_free:
                    walls = walls[:3]                  # V3.1 open top
                for wall_pos, wall_axis, wall_sign in walls:
                    wresult = find_contact_superellipse_wall(
                        gs.x[i], gs.y[i], gs.a[i], gs.b[i],
                        gs.n_shape[i], gs.theta[i],
                        wall_pos, wall_axis, wall_sign)
                    if wresult is not None:
                        pen, R_local = wresult
                        Fw, a_w = jkr_force_from_overlap(
                            pen, R_local, E_star_gw, W_wall)
                        # V3.6: wall contact stiffness, dF/d(delta) = 2 E_s a  (nN/um)
                        gs.wall_stiffness[i] += 2.0 * E_star_gw * 1e-3 * a_w
                        F[i, wall_axis] += wall_sign * Fw
                        if getattr(p, 'contact_wall_torque', False) and not gs.is_circle:
                            # V3.4: r x F at the wall contact point, which the
                            # support-function solver hands back for free. A
                            # circle's contact is on the centre line, so its wall
                            # torque is identically zero and the branch is skipped.
                            _, _, _, wcx, wcy = se2d_wall_core(
                                gs.x[i], gs.y[i], gs.a[i], gs.b[i], gs.n_shape[i],
                                gs.theta[i], wall_pos, wall_axis, wall_sign)
                            if wall_axis == 0:
                                torques[i] += -(wcy - gs.y[i]) * wall_sign * Fw
                            else:
                                torques[i] += (wcx - gs.x[i]) * wall_sign * Fw
                        # Wall clip
                        if wall_axis == 0:
                            wall_d = abs(gs.x[i] - wall_pos)
                        else:
                            wall_d = abs(gs.y[i] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        nx_w = -float(wall_sign) if wall_axis == 0 else 0.0
                        ny_w = 0.0 if wall_axis == 0 else -float(wall_sign)
                        gs.contact_clips[i].append((nx_w, ny_w, wall_d))

    # MC-DEM multi-contact stiffening correction (Giannis et al. 2021)
    if p.mc_dem_enabled and len(contacts) > 0:
        mc_dem_correction(contacts, gs, p, E_star_gg, F)

    # V3.1 buoyant weight along -y
    apply_gravity(gs, p, F)

    # ── Active noise on functional granules (vectorized) ──
    if p.T_active > 0:
        func_mask = gs.adhesive_mask[:N]
        n_func = int(np.sum(func_mask))
        if n_func > 0:
            gamma_func = p.drag_scale * gs.r[:N][func_mask]
            # Cells generate the noise: amplitude scales with sqrt(activity) (V3.0)
            noise_amp = (np.sqrt(2 * gamma_func * p.T_active / p.dt)
                         * np.sqrt(gs.activity[:N][func_mask]))
            F[:N, 0][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 1][func_mask] += noise_amp * rng.standard_normal(n_func)

    return F, torques, contacts


def compute_forces_3d(gs: GranuleSystem, p: Params, rng):
    """
    Compute all forces and torques on each granule in 3D.
    Returns (F, torques) where F is (N, 3) and torques is (N, 3).
    """
    N = gs.N
    F = np.zeros((N, 3))
    torques = np.zeros((N, 3))
    # V3.6: per-granule wall contact stiffness, zeroed every evaluation and
    # read by `contact_stiffness_per_granule`. Stays 0 under periodic
    # boundaries and for a bed that never touches a wall.
    gs.wall_stiffness = np.zeros(N)
    pos = gs.positions()  # (N, 3)
    contacts = []  # V1.5.2: per-contact data for stress visualization

    # V1.5: Reset per-cell force vectors (but preserve bridge state and targets)
    gs.cell_fx[:] = 0.0
    gs.cell_fy[:] = 0.0
    gs.cell_fz[:] = 0.0

    nu2 = p.poisson_ratio ** 2
    E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu2))
    E_star_gw = (p.E_modulus * 1e3) / (1.0 - nu2)

    # ── V3.0: pair adhesion / friction / E* come from the species tables on
    # gs (pair_W, pair_tau, pair_Estar; wall_W, wall_Estar). E_star_gg above
    # is kept only for the deferred LS-DEM branch.
    _law = law_code(p.traction_f_law)
    _rule = rule_code(p.traction_f_rule)
    _kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
    _adh = adhesion_force_ceiling(gs, p)   # V3.6: sigma x A ceiling, at g = 1
    _stack = stacking_enabled(p)           # V3.6: cells standing on cells
    _mu = float(getattr(p, 'friction_mu', 0.0))     # V3.1 Coulomb friction term
    _cell_adh = float(getattr(p, 'cell_contact_adhesion', 0.0))   # V3.2 cell grip at the contact
    _R_cap = float(getattr(p, 'curvature_R_cap', 0.0))            # V3.2 curvature cap
    _bridge_map = bridge_weight_map(gs, p) if _cell_adh > 0.0 else {}

    # Neighbour search (3D)
    max_r = float(np.max(gs.r_bound))
    cutoff = 2*max_r + p.L_max
    periodic = (p.boundary_mode == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dp = pos[j] - pos[i]
        if periodic:
            dp[0], dp[1], dp[2] = minimum_image_disp_3d(
                dp[0], dp[1], dp[2], p.Lx, p.Ly, p.Lz)
        d = np.linalg.norm(dp)
        if d < 1e-6:
            continue
        nv = dp / d  # unit normal i->j

        # ── LS-DEM deformable contact detection 3D (V2.3) ──
        pj_v = pos[i] + dp
        if p.deformable_enabled:
            from gels.lsdem import find_contacts_lsdem_3d
            lsdem_result = find_contacts_lsdem_3d(
                gs, i, j, pos[i,0], pos[i,1], pos[i,2],
                pj_v[0], pj_v[1], pj_v[2], p)
            if lsdem_result is not None:
                F[i] += lsdem_result['F_i']
                F[j] += lsdem_result['F_j']
                torques[i] += lsdem_result['tau_i']
                torques[j] += lsdem_result['tau_j']
                if gs.F_eps_accum is not None:
                    gs.F_eps_accum[i] += lsdem_result['F_eps_i']
                    gs.F_eps_accum[j] += lsdem_result['F_eps_j']
                overlap = lsdem_result['overlap_max']
                R_eff_approx = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])

                # JKR adhesion correction — translational only (V2.3)
                W_adh_pair = float(gs.pair_W[gs.species_id[i], gs.species_id[j]])
                if W_adh_pair > 0 and overlap > 0:
                    F_jkr, a_est = jkr_force_from_overlap(
                        overlap, R_eff_approx, E_star_gg, W_adh_pair)
                    F_hertz = hertz_contact_force(E_star_gg, R_eff_approx, overlap)
                    F_adh_corr = F_jkr - F_hertz
                    F[i] += F_adh_corr * nv
                    F[j] -= F_adh_corr * nv
                else:
                    a_est = np.sqrt(max(R_eff_approx * overlap, 0.0))

                # Estimate contact radius from overlap for clip rendering
                if a_est > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_est**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_est**2, 0.01))
                    gs.contact_clips[i].append((nv[0], nv[1], nv[2], clip_d_i))
                    gs.contact_clips[j].append((-nv[0], -nv[1], -nv[2], clip_d_j))
                contacts.append({
                    'i': i, 'j': j,
                    'cx': (pos[i,0]+pj_v[0])/2, 'cy': (pos[i,1]+pj_v[1])/2,
                    'cz': (pos[i,2]+pj_v[2])/2,
                    'nx': nv[0], 'ny': nv[1], 'nz': nv[2],
                    'overlap': overlap, 'R_eff': R_eff_approx,
                    'F_normal': np.linalg.norm(lsdem_result['F_i']),
                    'A_contact': np.pi * a_est**2,
                    'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                })
                in_contact = True
                contact_overlap = overlap
            else:
                in_contact = False
                contact_overlap = 0.0

            # Cell bridging
            if gs.adhesive_mask[i] and gs.adhesive_mask[j] and (gs.n_cells[i] + gs.n_cells[j] > 0):
                if in_contact:
                    gap = -contact_overlap
                else:
                    gap = d - gs.r_bound[i] - gs.r_bound[j]
                    if gap < 0: gap = 0.0
                F_cell_i = motor_clutch_force(
                    gs.E_gran[i], p, gs.fa_maturity[i], nu=gs.nu_gran[i],
                    g=traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[i] if _adh is not None else None))
                F_cell_j = motor_clutch_force(
                    gs.E_gran[j], p, gs.fa_maturity[j], nu=gs.nu_gran[j],
                    g=traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[j] if _adh is not None else None))
                # V3.6: the same cell's traction if it is standing on CELLS rather
                # than on the granule. Both computed properly; the service loop picks
                # per cell by `cell_layer`, because a ratio applied afterwards would
                # be the F(g.h) != F.h trap.
                F_up_i = F_up_j = None
                if _stack:
                    F_up_i = stacked_traction_force(
                        p, gs.fa_maturity[i], traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                        F_adh=(_adh[i] if _adh is not None else None))
                    F_up_j = stacked_traction_force(
                        p, gs.fa_maturity[j], traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                        F_adh=(_adh[j] if _adh is not None else None))
                n_ci, n_cj, _fac = _service_committed_bridges(
                    gs, i, j, gap, p, F_cell_i, F_cell_j, nv[0], nv[1], nv[2], F,
                    F_up_i=F_up_i, F_up_j=F_up_j)
                n_existing = n_ci + n_cj
                if gap < p.cell_sense_distance:
                    pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v, gap)
                    _attempt_new_bridges(
                        gs, i, j, gap, p, rng, F_cell_i, F_cell_j,
                        nv[0], nv[1], nv[2], F,
                        F_up_i=F_up_i, F_up_j=F_up_j, pair_fac=_fac,
                        n_existing_bridges=n_existing, path_factor=pf,
                        pos_i=pos[i], pos_j=pj_v)
            continue  # skip rigid contact path

        # Contact detection (rigid, V2.2) — use virtual position of j for periodic BCs
        if gs.is_circle:
            result = find_contact_spheres_3d(
                pos[i,0], pos[i,1], pos[i,2], gs.r[i],
                pj_v[0], pj_v[1], pj_v[2], gs.r[j])
        else:
            result = find_contact_superellipsoids_3d(
                pos[i,0], pos[i,1], pos[i,2],
                gs.a[i], gs.b[i], gs.c[i], gs.n1[i], gs.n2[i],
                gs.quat[i],
                pj_v[0], pj_v[1], pj_v[2],
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j],
                gs.quat[j])

        if result is not None:
            _, overlap, nx, ny, nz, cx, cy, cz_pt, R_eff = result
            if _R_cap > 0.0 and R_eff > 0.0:
                _lim = _R_cap * min(gs.r[i], gs.r[j])      # V3.2, see the 2D twin
                if R_eff > _lim:
                    R_eff = _lim
            n_vec = np.array([nx, ny, nz])

            if overlap > 0:
                # JKR adhesive contact (V2.3, replaces Hertz + DMT)
                si = gs.species_id[i]
                sj = gs.species_id[j]
                tau_0 = gs.pair_tau[si, sj]
                W_adh = gs.pair_W[si, sj]
                E_star_pair = gs.pair_Estar[si, sj]
                F_normal, a_contact = jkr_force_from_overlap(
                    overlap, R_eff, E_star_pair, W_adh)

                Fn = F_normal * n_vec
                F[i] -= Fn
                F[j] += Fn

                # Torque from off-centre contact
                if not gs.is_circle:
                    rc_i = np.array([cx - pos[i,0], cy - pos[i,1], cz_pt - pos[i,2]])
                    rc_j = np.array([cx - pos[j,0], cy - pos[j,1], cz_pt - pos[j,2]])
                    torques[i] += np.cross(rc_i, -Fn)
                    torques[j] += np.cross(rc_j, Fn)

                # Contact clip planes for 3D rendering
                if a_contact > 1e-6:
                    clip_d_i = np.sqrt(max(gs.r[i]**2 - a_contact**2, 0.01))
                    clip_d_j = np.sqrt(max(gs.r[j]**2 - a_contact**2, 0.01))
                    gs.contact_clips[i].append((nx, ny, nz, clip_d_i))
                    gs.contact_clips[j].append((-nx, -ny, -nz, clip_d_j))

                # Tangential friction (JKR contact area: π a²)
                A_contact = np.pi * a_contact**2 if a_contact > 0 else 0.0
                dv = np.array([gs.vx[j]-gs.vx[i], gs.vy[j]-gs.vy[i],
                               gs.vz[j]-gs.vz[i]])
                v_dot_n = np.dot(dv, n_vec)
                vt = dv - v_dot_n * n_vec
                vt_mag = np.linalg.norm(vt)

                if vt_mag > 1e-12:
                    F_t_cap = tau_0 * A_contact * 1e-3
                    if _mu > 0.0 and F_normal > 0.0:
                        F_t_cap += _mu * F_normal
                    if _cell_adh > 0.0:
                        F_t_cap += _cell_adh * _bridge_map.get(
                            (i, j) if i < j else (j, i), 0.0)   # V3.2: the cells' own grip
                    F_fric = F_t_cap * np.tanh(vt_mag / p.friction_v_ref)
                    t_vec = vt / vt_mag
                    Ft = F_fric * t_vec
                    F[i] += Ft
                    F[j] -= Ft
                    if not gs.is_circle:
                        torques[i] += np.cross(rc_i, Ft)
                        torques[j] += np.cross(rc_j, -Ft)

            contacts.append({
                'i': i, 'j': j,
                'cx': cx, 'cy': cy, 'cz': cz_pt,
                'nx': nx, 'ny': ny, 'nz': nz,
                'overlap': overlap, 'R_eff': R_eff,
                'F_normal': F_normal, 'A_contact': A_contact,
                'a_contact': a_contact,
                'gtype_i': int(gs.gtype[i]), 'gtype_j': int(gs.gtype[j]),
                'species_i': int(si), 'species_j': int(sj),
                'f_i': float(gs.f[i]), 'f_j': float(gs.f[j]),
                'E_star': float(E_star_pair), 'W': float(W_adh), 'tau_0': float(tau_0),
            })

            in_contact = True
            contact_overlap = overlap
        else:
            in_contact = False
            contact_overlap = 0.0

        # Cell bridging (motor-clutch, functional-functional)
        if gs.adhesive_mask[i] and gs.adhesive_mask[j] and (gs.n_cells[i] + gs.n_cells[j] > 0):
            if in_contact:
                gap = -contact_overlap
            else:
                gap = d - gs.r_bound[i] - gs.r_bound[j]
                if gap < 0:
                    gap = 0.0

            F_cell_i = motor_clutch_force(
                gs.E_gran[i], p, gs.fa_maturity[i], nu=gs.nu_gran[i],
                g=traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                F_adh=(_adh[i] if _adh is not None else None))
            F_cell_j = motor_clutch_force(
                gs.E_gran[j], p, gs.fa_maturity[j], nu=gs.nu_gran[j],
                g=traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                F_adh=(_adh[j] if _adh is not None else None))
            # V3.6: the same cell's traction if it is standing on CELLS rather
            # than on the granule. Both computed properly; the service loop picks
            # per cell by `cell_layer`, because a ratio applied afterwards would
            # be the F(g.h) != F.h trap.
            F_up_i = F_up_j = None
            if _stack:
                F_up_i = stacked_traction_force(
                    p, gs.fa_maturity[i], traction_gain(gs.f[i], gs.f[j], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[i] if _adh is not None else None))
                F_up_j = stacked_traction_force(
                    p, gs.fa_maturity[j], traction_gain(gs.f[j], gs.f[i], _law, _rule, p.traction_exponent, _kappa),
                    F_adh=(_adh[j] if _adh is not None else None))

            # 1) Service committed bridges (apply force with maturity ramp)
            n_ci, n_cj, _fac = _service_committed_bridges(
                gs, i, j, gap, p, F_cell_i, F_cell_j,
                nv[0], nv[1], nv[2], F, F_up_i=F_up_i, F_up_j=F_up_j)
            n_existing = n_ci + n_cj

            # 2) Path-dependent bridge formation (V2.1)
            if gap < p.cell_sense_distance:
                pf = _classify_bridge_path(gs, p, i, j, pos[i], pj_v, gap)
                _attempt_new_bridges(
                    gs, i, j, gap, p, rng, F_cell_i, F_cell_j,
                    nv[0], nv[1], nv[2], F,
                    F_up_i=F_up_i, F_up_j=F_up_j, pair_fac=_fac,
                    n_existing_bridges=n_existing, path_factor=pf,
                    pos_i=pos[i], pos_j=pj_v)

    # Wall repulsion: JKR adhesive contact (6 faces) — skip for periodic
    if not periodic:
        # Wall = per-species W and E* (V3.0); container shape / open top (V3.1)
        _geom = boundary_geometry(p, '3D')
        _walls_all = [
            (0.0, 0, +1), (p.Lx, 0, -1),
            (0.0, 1, +1), (p.Ly, 1, -1),
            (0.0, 2, +1), (p.Lz, 2, -1),
        ]
        if _geom.shape_code == 1:
            _walls_all = [w for w in _walls_all if w[1] == 2]    # cylinder replaces the x/y faces
        if _geom.top_free:
            _walls_all = [w for w in _walls_all if not (w[1] == 2 and w[2] < 0)]   # no lid
        for i in range(N):
            W_wall = gs.wall_W[gs.species_id[i]]
            E_star_gw = gs.wall_Estar[gs.species_id[i]]
            walls = _walls_all
            coords = [gs.x[i], gs.y[i], gs.z[i]]
            if gs.is_circle:
                r = gs.r[i]
                for wall_pos, axis, sign in walls:
                    pen = (r - (coords[axis] - wall_pos)) if sign > 0 \
                        else ((coords[axis] + r) - wall_pos)
                    if pen > 0:
                        Fw, a_w = jkr_force_from_overlap(
                            pen, r, E_star_gw, W_wall)
                        # V3.6: wall contact stiffness, dF/d(delta) = 2 E_s a  (nN/um)
                        gs.wall_stiffness[i] += 2.0 * E_star_gw * 1e-3 * a_w
                        F[i, axis] += sign * Fw
                        wall_d = abs(coords[axis] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        n3 = [0.0, 0.0, 0.0]; n3[axis] = -float(sign)
                        gs.contact_clips[i].append(
                            (n3[0], n3[1], n3[2], wall_d))
            else:
                for wall_pos, axis, sign in walls:
                    wresult = find_contact_wall_3d(
                        gs.x[i], gs.y[i], gs.z[i],
                        gs.a[i], gs.b[i], gs.c[i],
                        gs.n1[i], gs.n2[i], gs.quat[i], gs.r_bound[i],
                        wall_pos, axis, sign)
                    if wresult is not None:
                        pen, R_local = wresult
                        Fw, a_w = jkr_force_from_overlap(
                            pen, R_local, E_star_gw, W_wall)
                        # V3.6: wall contact stiffness, dF/d(delta) = 2 E_s a  (nN/um)
                        gs.wall_stiffness[i] += 2.0 * E_star_gw * 1e-3 * a_w
                        F[i, axis] += sign * Fw
                        if getattr(p, 'contact_wall_torque', False) and not gs.is_circle:
                            # V3.4: see the 2D twin above.
                            _, _, _, wcx, wcy, wcz = se3d_wall_core(
                                gs.x[i], gs.y[i], gs.z[i], gs.a[i], gs.b[i], gs.c[i],
                                gs.n1[i], gs.n2[i], gs.quat[i], wall_pos, axis, sign)
                            rw = np.array([wcx - gs.x[i], wcy - gs.y[i], wcz - gs.z[i]])
                            Fw_vec = np.zeros(3)
                            Fw_vec[axis] = sign * Fw
                            torques[i] += np.cross(rw, Fw_vec)
                        wall_d = abs(coords[axis] - wall_pos)
                        # Clip normal points TOWARD wall (opposite of force sign)
                        n3 = [0.0, 0.0, 0.0]; n3[axis] = -float(sign)
                        gs.contact_clips[i].append(
                            (n3[0], n3[1], n3[2], wall_d))
            # ── V3.1 cylindrical side wall: axis z, radius R_cyl about (cx, cy) ──
            if _geom.shape_code == 1:
                dxc = gs.x[i] - _geom.cx
                dyc = gs.y[i] - _geom.cy
                rho = np.sqrt(dxc * dxc + dyc * dyc)
                if rho > 1e-12:
                    ux = dxc / rho
                    uy = dyc / rho
                    if gs.is_circle:
                        pen_c = gs.r[i] + rho - _geom.R_cyl
                        wres_c = (pen_c, gs.r[i]) if pen_c > 0 else None
                    else:
                        wres_c = find_contact_wall_plane_3d(
                            gs.x[i], gs.y[i], gs.z[i], gs.a[i], gs.b[i], gs.c[i],
                            gs.n1[i], gs.n2[i], gs.quat[i], gs.r_bound[i],
                            _geom.cx + _geom.R_cyl * ux, _geom.cy + _geom.R_cyl * uy, gs.z[i],
                            -ux, -uy, 0.0)
                    if wres_c is not None:
                        pen_c, R_loc_c = wres_c
                        Fw, a_w = jkr_force_from_overlap(pen_c, R_loc_c, E_star_gw, W_wall)
                        # V3.6: wall contact stiffness, dF/d(delta) = 2 E_s a  (nN/um)
                        gs.wall_stiffness[i] += 2.0 * E_star_gw * 1e-3 * a_w
                        F[i, 0] -= Fw * ux
                        F[i, 1] -= Fw * uy
                        if getattr(p, 'contact_wall_torque', False) and not gs.is_circle:
                            # V3.4: see the axis-wall branch above.
                            _, _, _, wcx, wcy, wcz = se3d_wall_plane_core(
                                gs.x[i], gs.y[i], gs.z[i], gs.a[i], gs.b[i], gs.c[i],
                                gs.n1[i], gs.n2[i], gs.quat[i],
                                _geom.cx + _geom.R_cyl * ux, _geom.cy + _geom.R_cyl * uy,
                                gs.z[i], -ux, -uy, 0.0)
                            rw = np.array([wcx - gs.x[i], wcy - gs.y[i], wcz - gs.z[i]])
                            torques[i] += np.cross(rw, np.array([-Fw * ux, -Fw * uy, 0.0]))
                        gs.contact_clips[i].append((ux, uy, 0.0, _geom.R_cyl - rho))

    # MC-DEM multi-contact stiffening correction (Giannis et al. 2021)
    if p.mc_dem_enabled and len(contacts) > 0:
        mc_dem_correction(contacts, gs, p, E_star_gg, F)

    # V3.1 buoyant weight along -z
    apply_gravity(gs, p, F)

    # Active noise (vectorized)
    if p.T_active > 0:
        func_mask = gs.adhesive_mask[:N]
        n_func = int(np.sum(func_mask))
        if n_func > 0:
            gamma_func = p.drag_scale * gs.r[:N][func_mask]
            # Cells generate the noise: amplitude scales with sqrt(activity) (V3.0)
            noise_amp = (np.sqrt(2 * gamma_func * p.T_active / p.dt)
                         * np.sqrt(gs.activity[:N][func_mask]))
            F[:N, 0][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 1][func_mask] += noise_amp * rng.standard_normal(n_func)
            F[:N, 2][func_mask] += noise_amp * rng.standard_normal(n_func)

    return F, torques, contacts


def _resolve_overlaps(gs, p):
    """Push apart deeply interpenetrating granules after position integration.

    Uses bounding-sphere overlap to detect pairs where overlap exceeds
    max_overlap_frac * min(r_bound_i, r_bound_j), then projects each
    granule apart by half the excess.  Handles both 2D/3D and periodic/wall BCs.
    """
    N = gs.N
    if N < 2 or p.max_overlap_frac <= 0:
        return

    r_bound = gs.r_bound[:N]
    max_r = float(np.max(r_bound))
    cutoff = 2.0 * max_r
    periodic = p.boundary_mode == 'periodic'

    if gs.is_3d:
        pos = np.column_stack([gs.x[:N], gs.y[:N], gs.z[:N]])
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(pos)
    else:
        pos = np.column_stack([gs.x[:N], gs.y[:N]])
        if periodic:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
        else:
            tree = cKDTree(pos)

    pairs = tree.query_pairs(cutoff, output_type='ndarray')
    if len(pairs) == 0:
        return

    ii = pairs[:, 0]
    jj = pairs[:, 1]

    dx = pos[jj, 0] - pos[ii, 0]
    dy = pos[jj, 1] - pos[ii, 1]
    if periodic:
        dx -= p.Lx * np.round(dx / p.Lx)
        dy -= p.Ly * np.round(dy / p.Ly)

    if gs.is_3d:
        dz = pos[jj, 2] - pos[ii, 2]
        if periodic:
            dz -= p.Lz * np.round(dz / p.Lz)
        dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        dist = np.sqrt(dx * dx + dy * dy)

    dist = np.maximum(dist, 1e-12)
    if shape_dynamics_on(p, gs):
        # V3.2: bounding spheres here would destroy a shaped bed. At a true
        # phi 0.68 with AR 1.8 / n 3.5 the BOUNDING-SPHERE fraction is above 1,
        # so every contact reads as a deep overlap and the projection blows the
        # packing apart -- measured 7.25 % bed expansion in 6 h with no cells and
        # 37 % of pairs clipped. The directional radius is the right scale.
        _eta = 1.0 + float(getattr(p, 'packing_shape_margin', 0.0))
        _ux, _uy = dx / dist, dy / dist
        _uz = (dz / dist) if gs.is_3d else None
        _c = gs.c if gs.is_3d else None
        _n2 = gs.n2 if gs.is_3d else None
        lam_i = lambda_world(gs.a[ii], gs.b[ii], None if _c is None else _c[ii],
                             gs.n1[ii], None if _n2 is None else _n2[ii],
                             gs.theta[ii], None if not gs.is_3d else gs.quat[ii],
                             _ux, _uy, _uz, gs.is_3d)
        lam_j = lambda_world(gs.a[jj], gs.b[jj], None if _c is None else _c[jj],
                             gs.n1[jj], None if _n2 is None else _n2[jj],
                             gs.theta[jj], None if not gs.is_3d else gs.quat[jj],
                             -_ux, -_uy, None if _uz is None else -_uz, gs.is_3d)
        overlap = _eta * (lam_i + lam_j) - dist
        r_min = np.minimum(lam_i, lam_j)
    else:
        overlap = r_bound[ii] + r_bound[jj] - dist
        r_min = np.minimum(r_bound[ii], r_bound[jj])

    # Only correct overlaps exceeding the allowed threshold
    threshold = p.max_overlap_frac * r_min
    _fixed = getattr(gs, 'fixed', None)      # V3.1: immobile granules keep their position
    deep = overlap > threshold
    if not np.any(deep):
        return

    # Push each granule apart by half the excess overlap
    excess = overlap[deep] - threshold[deep]
    inv_d = 1.0 / dist[deep]
    nx = dx[deep] * inv_d
    ny = dy[deep] * inv_d
    correction = 0.5 * excess

    i_deep = ii[deep]
    j_deep = jj[deep]

    # V3.1: an immobile granule never moves, so its mobile partner takes the
    # whole correction (0.5 -> 1.0); with no immobile granules this is the
    # V2.7 half-and-half projection unchanged.
    fixed = getattr(gs, 'fixed', None)
    if fixed is not None and np.any(fixed[:N]):
        fi = fixed[i_deep]
        fj = fixed[j_deep]
        corr_i = np.where(fi, 0.0, np.where(fj, excess, correction))
        corr_j = np.where(fj, 0.0, np.where(fi, excess, correction))
    else:
        corr_i = corr_j = correction

    gs.n_overlap_clipped = int(np.count_nonzero(deep))      # V3.2 diagnostic
    gs.n_overlap_pairs = int(deep.size)

    corr_x = np.zeros(N)
    corr_y = np.zeros(N)
    np.subtract.at(corr_x, i_deep, corr_i * nx)
    np.subtract.at(corr_y, i_deep, corr_i * ny)
    np.add.at(corr_x, j_deep, corr_j * nx)
    np.add.at(corr_y, j_deep, corr_j * ny)
    gs.x[:N] += corr_x
    gs.y[:N] += corr_y

    if gs.is_3d:
        nz = dz[deep] * inv_d
        corr_z = np.zeros(N)
        np.subtract.at(corr_z, i_deep, corr_i * nz)
        np.add.at(corr_z, j_deep, corr_j * nz)
        gs.z[:N] += corr_z


def _stamp_circle_2d(X, Y, cx, cy, r_eff_i, w, phi_out, clips=None):
    """Stamp a single circle profile onto the 2D field (additive blending).

    When clips is provided, JKR half-plane clipping creates flat contact faces.
    """
    sdf = np.sqrt((X - cx)**2 + (Y - cy)**2) - r_eff_i
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)
    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    phi_out += profile


def _stamp_superellipse_2d(X, Y, cx, cy, a, b, n_s, theta, r, w, phi_out,
                            clips=None):
    """Stamp a single superellipse profile onto the 2D field (additive blending).

    When clips is provided, JKR half-plane clipping creates flat contact faces.
    """
    bx, by = _world_to_body(X, Y, cx, cy, theta)
    se_val = (np.abs(bx) / a)**n_s + (np.abs(by) / b)**n_s
    n_inv = 1.0 / n_s
    sdf = (se_val**n_inv - 1.0) * r
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)
    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    phi_out += profile


def _stamp_deformed_2d(X, Y, cx, cy, theta, gs, i, p, w, phi_out):
    """Stamp a deformed particle (semi-Lagrangian pull-back) onto 2D field.

    V2.3: Uses the mode-shape displacement to pull grid points back to the
    undeformed reference configuration, then evaluates the analytical implicit
    function.  JKR contact clips applied after pull-back SDF computation.
    """
    bx, by = _world_to_body(X, Y, cx, cy, theta)
    eps = gs.epsilon[i]          # (n_modes,)
    nu = p.poisson_ratio
    n_modes = len(eps)

    # Vectorised mode-shape displacement (mirrors lsdem._mode_displacement_2d)
    ux = eps[0] * (-nu * bx) + (eps[1] * bx if n_modes > 1 else 0.0)
    uy = eps[0] * by           + (eps[1] * (-by) if n_modes > 1 else 0.0)

    bx_ref = bx - ux
    by_ref = by - uy

    # Evaluate undeformed SDF at the pulled-back coordinate
    a_i, b_i, n_s = gs.a[i], gs.b[i], gs.n_shape[i]
    se_val = (np.abs(bx_ref) / a_i)**n_s + (np.abs(by_ref) / b_i)**n_s
    n_inv = 1.0 / n_s
    sdf = (se_val**n_inv - 1.0) * gs.r[i]

    # Apply JKR contact clips (flat faces at contact regions)
    clips = gs.contact_clips[i] if hasattr(gs, 'contact_clips') and gs.contact_clips else None
    if clips:
        _apply_clips_2d(sdf, X, Y, cx, cy, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))
    phi_out += profile


def _stamp_deformed_3d(phi_out, gs, i, cx, cy, cz, xg, yg, zg,
                       dx_g, dy_g, dz_g, Ng, w, p):
    """Stamp a deformed 3D particle via semi-Lagrangian SDF pull-back.

    V2.3: Transforms grid points to body frame, applies inverse mode-shape
    displacement, then evaluates the analytical superellipsoid implicit.
    """
    rb = gs.r_bound[i] + 3 * w
    ix0 = max(0, int((cx - rb) / dx_g))
    ix1 = min(Ng, int((cx + rb) / dx_g) + 1)
    iy0 = max(0, int((cy - rb) / dy_g))
    iy1 = min(Ng, int((cy + rb) / dy_g) + 1)
    iz0 = max(0, int((cz - rb) / dz_g))
    iz1 = min(Ng, int((cz + rb) / dz_g) + 1)
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return

    X, Y, Z = np.meshgrid(xg[ix0:ix1], yg[iy0:iy1], zg[iz0:iz1],
                           indexing='ij')
    dx_l, dy_l, dz_l = X - cx, Y - cy, Z - cz

    # Rotate to body frame
    R_mat = quat_to_rotation_matrix(gs.quat[i])
    bx = R_mat[0, 0] * dx_l + R_mat[1, 0] * dy_l + R_mat[2, 0] * dz_l
    by = R_mat[0, 1] * dx_l + R_mat[1, 1] * dy_l + R_mat[2, 1] * dz_l
    bz = R_mat[0, 2] * dx_l + R_mat[1, 2] * dy_l + R_mat[2, 2] * dz_l

    # Mode-shape displacement (mirrors lsdem._mode_displacement_3d)
    eps = gs.epsilon[i]
    nu = p.poisson_ratio
    n_modes = len(eps)
    ux = eps[0] * (-nu * bx)
    uy = eps[0] * (-nu * by)
    uz = eps[0] * bz
    if n_modes > 1:
        ux += eps[1] * bx
        uy += eps[1] * (-by)
        # uz += 0 for mode 1
    if n_modes > 2:
        ux += eps[2] * bx
        uy += eps[2] * by
        uz += eps[2] * bz

    bx_ref = bx - ux
    by_ref = by - uy
    bz_ref = bz - uz

    # Evaluate undeformed superellipsoid implicit at pulled-back point
    a_i, b_i, c_i = gs.a[i], gs.b[i], gs.c[i]
    n1_i, n2_i = gs.n1[i], gs.n2[i]
    se_val = ((np.abs(bx_ref / a_i)**n1_i +
               np.abs(by_ref / b_i)**n1_i)**(n2_i / n1_i) +
              np.abs(bz_ref / c_i)**n2_i)
    n_inv = 1.0 / n2_i
    sdf = (se_val**n_inv - 1.0) * gs.r[i]

    # Apply JKR contact clips (flat faces at contact regions)
    clips = gs.contact_clips[i] if hasattr(gs, 'contact_clips') and gs.contact_clips else None
    if clips:
        _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))

    phi_out[ix0:ix1, iy0:iy1, iz0:iz1] += profile


def _periodic_image_offsets_2d(cx, cy, rb, Lx, Ly):
    """Return list of (dx, dy) offsets for ghost images near periodic boundaries."""
    offsets = [(0, 0)]
    for sx in (-Lx, 0, Lx):
        for sy in (-Ly, 0, Ly):
            if sx == 0 and sy == 0:
                continue
            # Only add if ghost image could overlap the domain
            gx = cx + sx
            gy = cy + sy
            if -rb < gx < Lx + rb and -rb < gy < Ly + rb:
                offsets.append((sx, sy))
    return offsets


def _sum_species(phi_s):
    """Σ_k phi_s[k] accumulated in index order (matches the legacy phi_f + phi_i)."""
    total = phi_s[0].copy()
    for k in range(1, phi_s.shape[0]):
        total += phi_s[k]
    return total


def _normalize_species_fields(phi_s, V_true, V_cell, max_iter=20, tol=1e-3):
    """Volume-conserving normalisation (V2.1) generalised to K species grids.

    Additive stamping over-counts overlap zones. Iteratively scale every grid
    by the same factor so the rendered solid measure matches the true granule
    measure, then cap the per-cell total at 1.0 preserving the species ratios.
    Same operations, same order as the V2.7 two-grid version.
    """
    K = phi_s.shape[0]
    V_solid_true = V_true[0]
    for k in range(1, K):
        V_solid_true = V_solid_true + V_true[k]
    for _iter in range(max_iter):
        total = _sum_species(phi_s)
        V_rendered = float(np.sum(total)) * V_cell
        if V_rendered < 1e-30:
            break
        if abs(V_rendered - V_solid_true) / V_solid_true < tol:
            break
        scale = V_solid_true / V_rendered
        for k in range(K):
            phi_s[k] *= scale
        total = _sum_species(phi_s)
        over = total > 1.0
        if np.any(over):
            for k in range(K):
                phi_s[k][over] /= total[over]


def render_fields(gs: GranuleSystem, p: Params):
    """φ_f, φ_i, φ_v of shape (Ng, Ng): adhesive species, non-adhesive species, void.

    Thin wrapper over the per-species renderer (V3.0); see render_fields_species.
    """
    return group_species_fields(gs, _render_species_2d(gs, p))


def render_fields_3d(gs: GranuleSystem, p: Params):
    """φ_f, φ_i, φ_v of shape (Ng, Ng, Ng) — wrapper over the per-species renderer."""
    return group_species_fields(gs, _render_species_3d(gs, p))


def render_fields_species(gs, p):
    """Per-species phase fields phi_s, shape (K, Ng, Ng) in 2D or (K, Ng, Ng, Ng) in 3D.

    phi_s[k] is the rendered occupancy of species k; group_species_fields()
    collapses it to the legacy φ_f/φ_i/φ_v and collagen_field() to the
    collagen-weighted solid Σ f_k φ_k.
    """
    if gs.is_3d:
        return _render_species_3d(gs, p)
    return _render_species_2d(gs, p)


def _render_species_2d(gs, p):
    """
    Stamp each granule as tanh-profile shape onto its species grid.
    For circles: radial profile with volume-conserving effective radii.
    For superellipses: implicit-function-based signed distance.
    For periodic boundaries, ghost images are stamped at boundary crossings.
    Returns phi_s of shape (K, Ng, Ng).
    """
    # Auto-scale grid to resolve interface (V2.4: prevents aliasing on large domains)
    # Ensure grid spacing ≤ 2× interface_width so tanh spans ≥ 2 pixels
    min_Ng = int(np.ceil(max(p.Lx, p.Ly) / (p.interface_width * 2.0)))
    Ng = max(p.Ngrid, min_Ng)
    dx = p.Lx / Ng
    dy = p.Ly / Ng
    xg = np.linspace(dx/2, p.Lx - dx/2, Ng)
    yg = np.linspace(dy/2, p.Ly - dy/2, Ng)
    X, Y = np.meshgrid(xg, yg, indexing='ij')

    K, sid, _adh, _fsp = species_view(gs)
    phi_s = np.zeros((K, Ng, Ng))
    w = p.interface_width
    periodic = (p.boundary_mode == 'periodic')

    use_deformed = (p.deformable_enabled and gs.epsilon is not None)

    if use_deformed:
        # V2.3: Semi-Lagrangian deformed rendering — all particles go through
        # the pull-back path regardless of circle/superellipse distinction.
        for i in range(gs.N):
            theta_i = gs.theta[i] if hasattr(gs, 'theta') and gs.theta is not None else 0.0
            if periodic:
                rb = gs.r_bound[i] + 3 * w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_deformed_2d(X, Y, gs.x[i] + sx, gs.y[i] + sy,
                                       theta_i, gs, i, p, w, phi_s[sid[i]])
            else:
                _stamp_deformed_2d(X, Y, gs.x[i], gs.y[i],
                                   theta_i, gs, i, p, w, phi_s[sid[i]])
    elif gs.is_circle:
        r_eff, _ = compute_effective_radii(gs, p)
        has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None
        for i in range(gs.N):
            clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
            if periodic:
                rb = r_eff[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_circle_2d(X, Y, gs.x[i]+sx, gs.y[i]+sy,
                                     r_eff[i], w, phi_s[sid[i]],
                                     clips=clips_i)
            else:
                _stamp_circle_2d(X, Y, gs.x[i], gs.y[i],
                                 r_eff[i], w, phi_s[sid[i]],
                                 clips=clips_i)
    else:
        has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None
        for i in range(gs.N):
            clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
            if periodic:
                rb = gs.r_bound[i] + 3*w
                for sx, sy in _periodic_image_offsets_2d(
                        gs.x[i], gs.y[i], rb, p.Lx, p.Ly):
                    _stamp_superellipse_2d(
                        X, Y, gs.x[i]+sx, gs.y[i]+sy,
                        gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                        gs.r[i], w, phi_s[sid[i]],
                        clips=clips_i)
            else:
                _stamp_superellipse_2d(
                    X, Y, gs.x[i], gs.y[i],
                    gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                    gs.r[i], w, phi_s[sid[i]],
                    clips=clips_i)

    # Volume-conserving normalization (V2.1) over the K species grids.
    A_pixel = (p.Lx / Ng) * (p.Ly / Ng)
    A_true = []
    for k in range(K):
        mk = sid == k
        if not np.any(mk):
            A_true.append(0.0)
        elif gs.is_circle:
            A_true.append(float(np.sum(np.pi * gs.r[mk] ** 2)))
        else:
            A_true.append(float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                                    for j in np.where(mk)[0])))
    _normalize_species_fields(phi_s, A_true, A_pixel)
    return phi_s


def _stamp_granule_3d(phi_out, gs, i, cx, cy, cz, r_eff_3d_i,
                      xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, clips=None):
    """Stamp one 3D granule image at (cx, cy, cz) onto the field grids.

    When clips is provided, JKR half-space clipping creates flat contact faces.
    """
    rb = gs.r_bound[i] + 3*w
    ix0 = max(0, int((cx - rb) / dx_g))
    ix1 = min(Ng, int((cx + rb) / dx_g) + 1)
    iy0 = max(0, int((cy - rb) / dy_g))
    iy1 = min(Ng, int((cy + rb) / dy_g) + 1)
    iz0 = max(0, int((cz - rb) / dz_g))
    iz1 = min(Ng, int((cz + rb) / dz_g) + 1)
    if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1:
        return

    X, Y, Z = np.meshgrid(xg[ix0:ix1], yg[iy0:iy1], zg[iz0:iz1],
                           indexing='ij')

    if gs.is_circle:
        sdf = np.sqrt((X - cx)**2 + (Y - cy)**2 + (Z - cz)**2) - r_eff_3d_i
    else:
        dx_l = X - cx
        dy_l = Y - cy
        dz_l = Z - cz
        R_mat = quat_to_rotation_matrix(gs.quat[i])
        bx = R_mat[0,0]*dx_l + R_mat[1,0]*dy_l + R_mat[2,0]*dz_l
        by = R_mat[0,1]*dx_l + R_mat[1,1]*dy_l + R_mat[2,1]*dz_l
        bz = R_mat[0,2]*dx_l + R_mat[1,2]*dy_l + R_mat[2,2]*dz_l

        se_val = ((np.abs(bx/gs.a[i])**gs.n1[i] +
                   np.abs(by/gs.b[i])**gs.n1[i])**(gs.n2[i]/gs.n1[i]) +
                  np.abs(bz/gs.c[i])**gs.n2[i])
        n_inv = 1.0 / gs.n2[i]
        sdf = (se_val**n_inv - 1.0) * gs.r[i]

    if clips:
        _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips)

    profile = 0.5 * (1.0 - np.tanh(sdf / w))

    phi_out[ix0:ix1, iy0:iy1, iz0:iz1] += profile


def _render_species_3d(gs, p):
    """
    Stamp each granule onto its species 3D grid using the superellipsoid
    implicit function. For periodic boundaries, ghost images are stamped at
    boundary crossings. Returns phi_s of shape (K, Ng, Ng, Ng).
    """
    # Auto-scale grid to resolve interface (V2.4: prevents aliasing on large domains)
    min_Ng = int(np.ceil(max(p.Lx, p.Ly, p.Lz) / (p.interface_width * 2.0)))
    # Cap 3D auto-scale at 200 to avoid excessive memory (200^3 = 8M voxels)
    Ng = max(p.Ngrid_3d, min(min_Ng, 200))
    dx_g = p.Lx / Ng
    dy_g = p.Ly / Ng
    dz_g = p.Lz / Ng
    xg = np.linspace(dx_g/2, p.Lx - dx_g/2, Ng)
    yg = np.linspace(dy_g/2, p.Ly - dy_g/2, Ng)
    zg = np.linspace(dz_g/2, p.Lz - dz_g/2, Ng)

    # For 3D, also widen interface if grid is still too coarse (cheaper than more voxels)
    w_3d = max(p.interface_width, max(dx_g, dy_g, dz_g))

    K, sid, _adh, _fsp = species_view(gs)
    phi_s = np.zeros((K, Ng, Ng, Ng))
    w = w_3d
    periodic = (p.boundary_mode == 'periodic')

    use_deformed = (p.deformable_enabled and gs.epsilon is not None)

    if gs.is_circle:
        r_eff_3d, _ = compute_effective_radii_3d(gs, p)
    else:
        r_eff_3d = gs.r

    has_clips = hasattr(gs, 'contact_clips') and gs.contact_clips is not None

    for i in range(gs.N):
        r_eff_i = r_eff_3d[i]
        clips_i = gs.contact_clips[i] if has_clips and gs.contact_clips[i] else None
        if periodic:
            rb = gs.r_bound[i] + 3*w
            for sx in (-p.Lx, 0, p.Lx):
                for sy in (-p.Ly, 0, p.Ly):
                    for sz in (-p.Lz, 0, p.Lz):
                        gx = gs.x[i] + sx
                        gy = gs.y[i] + sy
                        gz = gs.z[i] + sz
                        if (gx + rb > 0 and gx - rb < p.Lx and
                            gy + rb > 0 and gy - rb < p.Ly and
                            gz + rb > 0 and gz - rb < p.Lz):
                            if use_deformed:
                                _stamp_deformed_3d(
                                    phi_s[sid[i]], gs, i, gx, gy, gz,
                                    xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, p)
                            else:
                                _stamp_granule_3d(
                                    phi_s[sid[i]], gs, i, gx, gy, gz,
                                    r_eff_i, xg, yg, zg,
                                    dx_g, dy_g, dz_g, Ng, w,
                                    clips=clips_i)
        else:
            if use_deformed:
                _stamp_deformed_3d(
                    phi_s[sid[i]], gs, i,
                    gs.x[i], gs.y[i], gs.z[i],
                    xg, yg, zg, dx_g, dy_g, dz_g, Ng, w, p)
            else:
                _stamp_granule_3d(
                    phi_s[sid[i]], gs, i, gs.x[i], gs.y[i], gs.z[i],
                    r_eff_i, xg, yg, zg, dx_g, dy_g, dz_g, Ng, w,
                    clips=clips_i)

    # Volume-conserving normalization (V2.1): additive blending over-counts
    # at overlap zones.  Iterative scale-and-cap: each iteration normalises
    # the integral to match true granule volume, then caps per-voxel at 1.0.
    # Excess from capped voxels is redistributed in the next iteration.
    # Converges in ~5-10 iterations to <0.1% volume error.
    V_voxel = dx_g * dy_g * dz_g
    V_true = []
    for k in range(K):
        mk = sid == k
        if not np.any(mk):
            V_true.append(0.0)
        elif gs.is_circle:
            V_true.append(float(np.sum((4.0 / 3.0) * np.pi * gs.r[mk] ** 3)))
        else:
            V_true.append(float(sum(superellipsoid_volume(
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                for j in np.where(mk)[0])))
    _normalize_species_fields(phi_s, V_true, V_voxel)
    return phi_s


def connectivity(field, thresh_frac=0.3):
    """Cluster analysis on thresholded field."""
    thr = np.mean(field) + thresh_frac * np.std(field)
    b = (field > thr).astype(int)
    lab, nc = label(b)
    if nc == 0 or b.sum() == 0:
        return 0, 0.0, 0.0
    sz = np.array([np.sum(lab == l) for l in range(1, nc+1)])
    return nc, float(sz.max()/b.sum()), float(b.sum()/b.size)


def compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces, phi_s=None):
    m = dict(time=t)
    periodic = (p.boundary_mode == 'periodic')

    # ── Boundary exclusion: compute metrics on inner region only ──
    # For periodic boundaries, use full domain (no exclusion needed)
    bx = 0.0 if periodic else p.boundary_exclusion
    if bx > 0:
        shape = phi_f.shape
        # Index ranges for inner region (exclude bx fraction from each edge)
        lo = [int(bx * s) for s in shape]
        hi = [int((1.0 - bx) * s) for s in shape]
        if phi_f.ndim == 3:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        else:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        pf_inner = phi_f[inner]
        pi_inner = phi_i[inner]
        pv_inner = phi_v[inner]
        inner_slice = inner
        # Granule mask: which granules are inside the inner region
        x_lo = bx * p.Lx; x_hi = (1.0 - bx) * p.Lx
        y_lo = bx * p.Ly; y_hi = (1.0 - bx) * p.Ly
        inner_gran = (gs.x >= x_lo) & (gs.x <= x_hi) & (gs.y >= y_lo) & (gs.y <= y_hi)
        if gs.is_3d:
            z_lo = bx * p.Lz; z_hi = (1.0 - bx) * p.Lz
            inner_gran &= (gs.z >= z_lo) & (gs.z <= z_hi)
    else:
        pf_inner, pi_inner, pv_inner = phi_f, phi_i, phi_v
        inner_gran = np.ones(gs.N, dtype=bool)
        inner_slice = None

    m['phi_f_mean'] = float(np.mean(pf_inner))
    m['phi_i_mean'] = float(np.mean(pi_inner))
    m['phi_v_mean'] = float(np.mean(pv_inner))
    m['boundary_exclusion'] = bx
    m['n_granules_inner'] = int(np.sum(inner_gran))
    # V3.1 cell population
    m['n_cells_total'] = int(gs.total_cells)
    m['n_live_cells'] = int(np.sum(gs.cell_state != int(CellState.SENESCENT)))
    m['n_divisions_cum'] = int(getattr(gs, 'n_divisions_cum', 0))
    from gels.division import cycle_metrics as _cycle_metrics   # local: avoids a circular import
    m.update(_cycle_metrics(gs))                 # V3.2 cell-cycle diagnostics
    m['n_boundary'] = int(np.sum(gs.fixed)) if hasattr(gs, 'fixed') else 0
    # V3.1 bed observables (positional: free of the rendering halo). Only for
    # open / cylindrical containers, where a bed surface exists at all.
    _geom = boundary_geometry(p)
    if _geom.top_free or _geom.shape_code == 1:
        m.update(bed_metrics(gs, p))

    # Functional connectivity (inner region)
    fn, fl, fc = connectivity(pf_inner, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)

    # Void connectivity (inner region)
    vn, vl, vc = connectivity(pv_inner, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)

    # Inert connectivity (inner region)
    inn, il, _ = connectivity(pi_inner, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    # Tissue: dense functional regions (inner region)
    m['tissue_frac'] = float(np.mean(pf_inner > 0.5))

    # Max cluster area (inner region)
    thr = np.mean(pf_inner) + 0.3*np.std(pf_inner)
    b = (pf_inner > thr).astype(int); lab, nc = label(b)
    dxg = p.Lx / phi_f.shape[0]
    if nc > 0:
        sizes = np.array([np.sum(lab==l) for l in range(1, nc+1)])
        m['func_max_area'] = float(np.max(sizes) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    # Packing in functional-rich region (inner region)
    fr = pf_inner > thr
    pt = pf_inner + pi_inner
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    # Mean force magnitude
    if forces.shape[1] >= 3 and gs.is_3d:
        F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2 + forces[:,2]**2)
    else:
        F_mag = np.sqrt(forces[:,0]**2 + forces[:,1]**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[gs.func_mask])) if np.any(gs.func_mask) else 0.0

    # Overlap & contact diagnostics
    pos = gs.positions()
    max_rb = float(np.max(gs.r_bound))
    if periodic:
        if gs.is_3d:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
        else:
            tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_rb, output_type='ndarray')

    n_contacts = 0
    n_contacts_ff = 0   # functional–functional
    n_contacts_if = 0   # inert–functional
    n_contacts_ii = 0   # inert–inert
    max_overlap_ratio = 0.0
    total_overlap_area = 0.0   # 2D: area; 3D: volume
    n_bridges = 0
    # V3.0 species-resolved contact statistics
    K_sp, sid_sp, adh_sp, f_sp = species_view(gs)
    sp_pair = np.zeros((K_sp, K_sp), dtype=np.int64)
    f_prod_sum = 0.0
    # V3.3: per-pair contact flag for the graph percolation, recorded from this
    # loop so the twin uses the engine's own predicate, not a second one.
    is_contact_ref = np.zeros(len(pairs), dtype=np.uint8)

    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            if gs.is_3d:
                dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                    dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
            else:
                dv[0], dv[1] = minimum_image_disp(dv[0], dv[1], p.Lx, p.Ly)
        d = np.sqrt(np.dot(dv, dv))

        if gs.is_circle:
            overlap = gs.r[i] + gs.r[j] - d
        else:
            if gs.is_3d:
                # 3D superellipsoid: approximate overlap from bounding sphere
                overlap = gs.r_bound[i] + gs.r_bound[j] - d
            else:
                result = find_contact_superellipses(
                    pos[i,0], pos[i,1], gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                    pos[j,0], pos[j,1], gs.a[j], gs.b[j], gs.n_shape[j], gs.theta[j])
                overlap = result[1] if result is not None else 0.0

        if overlap > 0:
            is_contact_ref[idx] = 1
            n_contacts += 1
            ti, tj = gs.gtype[i], gs.gtype[j]
            if ti == 0 and tj == 0:
                n_contacts_ff += 1
            elif ti == 1 and tj == 1:
                n_contacts_ii += 1
            else:
                n_contacts_if += 1
            si_c = int(sid_sp[i]); sj_c = int(sid_sp[j])
            if si_c <= sj_c:
                sp_pair[si_c, sj_c] += 1
            else:
                sp_pair[sj_c, si_c] += 1
            f_prod_sum += float(f_sp[si_c] * f_sp[sj_c])
            R_min = min(gs.r[i], gs.r[j])
            max_overlap_ratio = max(max_overlap_ratio, overlap / R_min)
            if gs.is_3d:
                if gs.is_circle:
                    total_overlap_area += overlap_lens_volume(gs.r[i], gs.r[j], d)
                else:
                    # Approximate overlap volume for superellipsoids
                    R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                    total_overlap_area += (4.0/3.0) * np.pi * R_eff * overlap**2
            else:
                if gs.is_circle:
                    total_overlap_area += overlap_lens_area(gs.r[i], gs.r[j], d)
                else:
                    # Approximate overlap area for superellipses
                    R_eff = gs.r[i] * gs.r[j] / (gs.r[i] + gs.r[j])
                    total_overlap_area += np.pi * R_eff * overlap

    m.update(graph_percolation_metrics(gs, p, pos, pairs[:, 0], pairs[:, 1],
                                       is_contact_ref, periodic))

    # Bridge count (functional-functional pairs with attached cells in sensing range)
    cutoff_bridge = 2 * max_rb + p.cell_sense_distance
    pairs_b = tree.query_pairs(cutoff_bridge, output_type='ndarray')
    for idx in range(len(pairs_b)):
        i, j = pairs_b[idx]
        if gs.gtype[i] != 0 or gs.gtype[j] != 0:
            continue
        n_avail_i = max(0.0, gs.n_attached[i] - gs.n_overcrowded[i])
        n_avail_j = max(0.0, gs.n_attached[j] - gs.n_overcrowded[j])
        if n_avail_i < 0.1 or n_avail_j < 0.1:
            continue
        dv = pos[j] - pos[i]
        if periodic:
            if gs.is_3d:
                dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                    dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
            else:
                dv[0], dv[1] = minimum_image_disp(dv[0], dv[1], p.Lx, p.Ly)
        d = np.sqrt(np.dot(dv, dv))
        if gs.is_circle:
            gap = d - gs.r[i] - gs.r[j]
        else:
            gap = d - gs.r_bound[i] - gs.r_bound[j]
            if gap < 0:
                gap = 0.0
        if 0 < gap < p.cell_sense_distance:
            n_bridges += 1

    if gs.is_3d:
        if gs.is_circle:
            total_granule_vol = float(np.sum((4.0/3.0) * np.pi * gs.r**3))
        else:
            # Approximate superellipsoid volume using bounding sphere
            total_granule_vol = float(np.sum((4.0/3.0) * np.pi * gs.r**3))
        total_granule_area = total_granule_vol  # reuse variable name for conservation ratio
    else:
        if gs.is_circle:
            total_granule_area = float(np.sum(np.pi * gs.r**2))
        else:
            total_granule_area = float(sum(
                superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                for i in range(gs.N)))
    m['n_contacts'] = n_contacts
    m['n_contacts_ff'] = n_contacts_ff
    m['n_contacts_if'] = n_contacts_if
    m['n_contacts_ii'] = n_contacts_ii
    m['max_overlap_ratio'] = float(max_overlap_ratio)
    m['total_overlap_area'] = float(total_overlap_area)
    m['area_conservation'] = 1.0 - total_overlap_area / max(total_granule_area, 1e-30)
    if gs.is_3d:
        m['volume_conservation'] = m['area_conservation']  # same ratio, 3D volumes
    m['n_bridges'] = n_bridges

    # ── Shape descriptors (V1.3) ──
    if not gs.is_circle:
        ar_arr = np.maximum(gs.a, gs.b) / np.minimum(gs.a, gs.b)
        elong_arr = 1.0 - np.minimum(gs.a, gs.b) / np.maximum(gs.a, gs.b)
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                          for i in range(gs.N)])
        perims = np.array([superellipse_perimeter(gs.a[i], gs.b[i], gs.n_shape[i])
                           for i in range(gs.N)])
        circularity = 4.0 * np.pi * areas / (perims**2 + 1e-30)
        m['shape_aspect_ratio_mean'] = float(np.mean(ar_arr))
        m['shape_aspect_ratio_std'] = float(np.std(ar_arr))
        m['shape_elongation_mean'] = float(np.mean(elong_arr))
        m['shape_circularity_mean'] = float(np.mean(circularity))
        m['shape_blockiness_mean'] = float(np.mean(gs.n_shape))

    # ── Cell state metrics ──
    func = gs.func_mask
    m['n_attached_total'] = float(np.sum(gs.n_attached[func]))
    m['n_seeded_total'] = float(np.sum(gs.n_cells[func]))
    m['mean_spread_frac'] = float(np.mean(gs.spread_fraction[func])) if np.any(func) else 0.0
    m['mean_fa_maturity'] = float(np.mean(gs.fa_maturity[func])) if np.any(func) else 0.0
    m['n_overcrowded_total'] = float(np.sum(gs.n_overcrowded[func]))

    # ── Per-cell bridge force monitoring (V1.9) ──
    migrating_mask = gs.cell_state == int(CellState.MIGRATING)
    m['n_migrating_cells'] = int(np.sum(migrating_mask))
    bridging_mask = gs.cell_state == int(CellState.BRIDGING)
    n_bridging_cells = int(np.sum(bridging_mask))
    m['n_bridging_cells'] = n_bridging_cells
    n_locked_in = 0
    if n_bridging_cells > 0:
        cell_F_mag = np.sqrt(gs.cell_fx**2 + gs.cell_fy**2 + gs.cell_fz**2)
        bridge_forces = cell_F_mag[bridging_mask]
        m['bridge_force_mean'] = float(np.mean(bridge_forces))
        m['bridge_force_max'] = float(np.max(bridge_forces))
        m['bridge_force_min'] = float(np.min(bridge_forces))
        m['bridge_force_std'] = float(np.std(bridge_forces))
        # Count locked-in cells (above force threshold)
        n_locked_in = int(np.sum(bridge_forces >= p.bridge_lock_force_threshold))
    else:
        m['bridge_force_mean'] = 0.0
        m['bridge_force_max'] = 0.0
        m['bridge_force_min'] = 0.0
        m['bridge_force_std'] = 0.0
    m['n_locked_in_cells'] = n_locked_in

    # ── Transport metrics (V1.4, inner region) ──
    porosity = float(np.mean(pv_inner))
    m['porosity'] = porosity
    inner_r = gs.r[inner_gran] if np.any(inner_gran) else gs.r
    d_grain = float(np.mean(2 * inner_r))
    m['d_grain_mean'] = d_grain
    m['K_kozeny_carman'] = float(kozeny_carman_permeability(porosity, d_grain))

    # Compaction ratio (inner region)
    phi_solid = float(np.mean(pf_inner) + np.mean(pi_inner))
    m['phi_solid'] = phi_solid

    # True granule-based volume fractions (V2.1, invariant to rendering)
    if gs.is_3d:
        V_domain = domain_volume(p, '3D')      # V3.1: cylinder-aware
        if gs.is_circle:
            V_f_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.func_mask] ** 3))
            V_i_true = float(np.sum((4.0 / 3.0) * np.pi * gs.r[gs.inert_mask] ** 3))
        else:
            V_f_true = float(sum(superellipsoid_volume(
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                for j in np.where(gs.func_mask)[0]))
            V_i_true = float(sum(superellipsoid_volume(
                gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                for j in np.where(gs.inert_mask)[0]))
    else:
        V_domain = domain_volume(p, '2D')
        if gs.is_circle:
            V_f_true = float(np.sum(np.pi * gs.r[gs.func_mask] ** 2))
            V_i_true = float(np.sum(np.pi * gs.r[gs.inert_mask] ** 2))
        else:
            V_f_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                for j in np.where(gs.func_mask)[0]))
            V_i_true = float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                for j in np.where(gs.inert_mask)[0]))
    m['phi_f_true'] = V_f_true / V_domain
    m['phi_i_true'] = V_i_true / V_domain
    m['phi_solid_true'] = (V_f_true + V_i_true) / V_domain
    # V3.2: overlap is counted twice in the sums above (1-4 % once a soft bed
    # compacts through interpenetration), so report the net value too.
    _V_lens = overlap_solid_volume(gs, mobile_only=False)
    m['overlap_volume'] = _V_lens
    m['overlap_volume_fraction'] = (_V_lens / (V_f_true + V_i_true)) if (V_f_true + V_i_true) > 0 else 0.0
    m['phi_solid_true_net'] = max(0.0, (V_f_true + V_i_true) - _V_lens) / V_domain
    m.update(projection_metrics(gs))            # V3.2: is the rail or the contact law in charge?
    m.update(energy_metrics(gs))                # V3.5: is the force law a gradient?
    m.update(substep_metrics(gs))               # V3.6: what the substep controller did
    m.update(stress_metrics(gs))                # V3.7: the virial stress, in kPa
    m.update(traction_metrics(gs, p))           # V3.6: what the cells pull with, and store
    ar_mean = float(np.mean(np.maximum(gs.a, gs.b) /
                            np.minimum(gs.a, gs.b))) if gs.N > 0 else 1.0
    phi_rcp = rcp_fraction_superellipsoid(ar_mean)
    m['phi_RCP'] = phi_rcp
    m['compaction_ratio'] = phi_solid / phi_rcp if phi_rcp > 0 else 0.0
    m.update(laguerre_metrics(gs, p, pos, phi_rcp))
    m.update(pore_metrics(gs, p, phi_f.shape, periodic))

    # Darcy number
    if gs.is_3d:
        L_char = domain_volume(p, '3D') ** (1.0/3.0)
    else:
        L_char = np.sqrt(p.Lx * p.Ly)
    m['Da_number'] = m['K_kozeny_carman'] / (L_char**2) if L_char > 0 else 0.0

    # V2.3: Deformation metrics (LS-DEM)
    if gs.epsilon is not None:
        eps_mag = np.sqrt(np.sum(gs.epsilon**2, axis=1))
        m['def_strain_mean'] = float(np.mean(eps_mag))
        m['def_strain_max'] = float(np.max(eps_mag))
        m['def_strain_std'] = float(np.std(eps_mag))
        for alpha in range(gs.epsilon.shape[1]):
            m[f'def_mode_{alpha}_mean'] = float(np.mean(gs.epsilon[:, alpha]))
            m[f'def_mode_{alpha}_std'] = float(np.std(gs.epsilon[:, alpha]))

    # V2.1: Two-compartment volume conservation (Voronoi shrink-wrap)
    if gs.N > 0 and np.any(gs.func_mask) and np.any(gs.inert_mask):
        pos_vc = gs.positions()
        if periodic:
            if gs.is_3d:
                tree_vc = cKDTree(pos_vc, boxsize=[p.Lx, p.Ly, p.Lz])
            else:
                tree_vc = cKDTree(pos_vc, boxsize=[p.Lx, p.Ly])
        else:
            tree_vc = tree  # reuse the tree built above
        grid_shape = phi_f.shape
        if gs.is_3d:
            nx, ny, nz = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
            cz = np.linspace(0.5 * p.Lz / nz, p.Lz - 0.5 * p.Lz / nz, nz)
            gx_v, gy_v, gz_v = np.meshgrid(cx, cy, cz, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel(), gz_v.ravel()])
        else:
            nx, ny = grid_shape
            cx = np.linspace(0.5 * p.Lx / nx, p.Lx - 0.5 * p.Lx / nx, nx)
            cy = np.linspace(0.5 * p.Ly / ny, p.Ly - 0.5 * p.Ly / ny, ny)
            gx_v, gy_v = np.meshgrid(cx, cy, indexing='ij')
            voxel_pts = np.column_stack([gx_v.ravel(), gy_v.ravel()])
        _, nearest_idx = tree_vc.query(voxel_pts)
        nearest_type = gs.gtype[nearest_idx].reshape(grid_shape)
        fm = nearest_type == 0
        im = nearest_type == 1
        n_func_v = int(np.sum(fm))
        n_inert_v = int(np.sum(im))
        m['x_f'] = n_func_v / max(phi_f.size, 1)
        m['x_i'] = n_inert_v / max(phi_f.size, 1)
        m['phi_v_in_func'] = float(np.mean(phi_v[fm])) if n_func_v > 0 else 0.0
        m['phi_v_in_inert'] = float(np.mean(phi_v[im])) if n_inert_v > 0 else 0.0
        m['phi_solid_func'] = float(np.mean((phi_f + phi_i)[fm])) if n_func_v > 0 else 0.0
        m['phi_solid_inert'] = float(np.mean((phi_f + phi_i)[im])) if n_inert_v > 0 else 0.0
        m['phi_v_crosscheck'] = m['x_f'] * m['phi_v_in_func'] + m['x_i'] * m['phi_v_in_inert']

    # ── V3.0: per-species metrics (indices; names live in metadata.json) ──
    counts_sp = np.bincount(sid_sp, minlength=K_sp)
    if gs.total_cells > 0:
        bridging = np.asarray(gs.cell_state) == int(CellState.BRIDGING)
        bridging_sp = np.bincount(sid_sp[np.asarray(gs.cell_granule_id)[bridging]], minlength=K_sp)
    else:
        bridging_sp = np.zeros(K_sp, dtype=np.int64)
    phi_c_true = 0.0
    for k in range(K_sp):
        mk = sid_sp == k
        m[f'n_gran_sp_{k}'] = int(counts_sp[k])
        m[f'n_cells_sp_{k}'] = int(np.sum(gs.n_cells[mk])) if np.any(mk) else 0
        m[f'n_bridging_sp_{k}'] = int(bridging_sp[k])
        if not np.any(mk):
            V_k = 0.0
        elif gs.is_3d:
            V_k = (float(np.sum((4.0 / 3.0) * np.pi * gs.r[mk] ** 3)) if gs.is_circle else
                   float(sum(superellipsoid_volume(gs.a[j], gs.b[j], gs.c[j], gs.n1[j], gs.n2[j])
                             for j in np.where(mk)[0])))
        else:
            V_k = (float(np.sum(np.pi * gs.r[mk] ** 2)) if gs.is_circle else
                   float(sum(superellipse_area(gs.a[j], gs.b[j], gs.n_shape[j])
                             for j in np.where(mk)[0])))
        m[f'phi_sp_{k}_true'] = V_k / V_domain
        phi_c_true += f_sp[k] * m[f'phi_sp_{k}_true']
        # coordination per species: every contact touches two granules
        endpoints = int(np.sum(sp_pair[k, :]) + np.sum(sp_pair[:, k]))
        m[f'Z_sp_{k}'] = endpoints / counts_sp[k] if counts_sp[k] > 0 else 0.0
        if phi_s is not None:
            pk = phi_s[k][inner_slice] if inner_slice is not None else phi_s[k]
            m[f'phi_sp_{k}_mean'] = float(np.mean(pk))
    for a in range(K_sp):
        for b in range(a, K_sp):
            m[f'n_contacts_sp_{a}_{b}'] = int(sp_pair[a, b])
    m['phi_c_true'] = float(phi_c_true)
    m['f_solid_mean'] = float(phi_c_true / m['phi_solid_true']) if m['phi_solid_true'] > 0 else 0.0
    m['contact_ff_weight'] = float(f_prod_sum / n_contacts) if n_contacts > 0 else 0.0
    if phi_s is not None:
        phi_c = collagen_field(gs, phi_s)
        pc_inner = phi_c[inner_slice] if inner_slice is not None else phi_c
        m['phi_c_mean'] = float(np.mean(pc_inner))

    return m


