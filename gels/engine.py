"""
Overdamped Particle Dynamics Model for Cell-Driven Granular Rearrangement
==========================================================================
V1.6 — Individual Cell Tracking + Data Serialization (no plotting).

MODES:
    "2D"       — Pure 2D simulation with superellipses (V1.3 compatible)
    "2D-slice" — Generate 3D superellipsoid packing, slice at z-midplane, run 2D
    "3D"       — Full 3D volumetric simulation with superellipsoids

MATHEMATICAL MODEL
------------------
N rigid granules (functional or inert) in 2D or 3D, overdamped regime.
Granules maintain their shape; cells bridge nearby functional pairs and
pull them together.

EQUATIONS OF MOTION (overdamped Langevin):
    γ_i dx_i/dt = F_i^contact + F_i^cell + F_i^wall + F_i^noise

3D SHAPE: Superellipsoid
    (|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1
    Spheres: a=b=c, n1=n2=2.  Ellipsoids: n1=n2=2.  Blocky: n1,n2 > 2.

FORCE LAWS:
    (1) Hertzian contact: F = (4/3) E* √R* δ^{3/2}
    (2) Motor-clutch bridging (functional–functional, Chan & Odde 2008)
    (3) Wall repulsion (Hertz, rigid limit)
    (4) Activity noise (cell-driven fluctuations)
    (5) Area-dependent friction (Gong 2006, Pitenis 2014)
    (6) DMT adhesion (Derjaguin-Muller-Toporov)

3D ROTATIONAL DYNAMICS:
    Quaternion orientation (w,x,y,z), overdamped: I_eff dω/dt = τ

TRANSPORT METRICS (V1.4):
    Kozeny-Carman permeability, Darcy number, compaction ratio, porosity

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import label
from scipy.special import gamma as _gamma, beta as _beta
from scipy.optimize import brentq, minimize_scalar
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from enum import IntEnum
import time as timer
import os
import json
import csv
import tarfile
import subprocess
import sys

from gels.kernels import HAS_NUMBA as _HAS_NUMBA

# The engine prints Greek letters and micro signs in its progress output. On a
# Windows console that is still cp1252 this used to raise UnicodeEncodeError
# inside print(); degrade unencodable glyphs to '?' instead (encoding untouched).
for _stream in (sys.stdout, sys.stderr):
    try:
        if _stream is not None and getattr(_stream, 'errors', None) not in ('replace', 'backslashreplace'):
            _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback: identity decorator
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# ══════════════════════════════════════════════════════════════════════
# Parameters
# ══════════════════════════════════════════════════════════════════════

@dataclass
class Params:
    # ── Simulation mode ──
    mode: str = "2D"                # "2D", "2D-slice", or "3D"

    # ── Domain (µm) ──
    Lx: float = 800.0
    Ly: float = 800.0
    Lz: float = 800.0              # depth (used in 3D and 2D-slice modes)
    boundary_mode: str = "walls"    # "walls" (default) or "periodic"
    # -- Container geometry (V3.1). shape/top need boundary_mode == walls. --
    boundary_shape: str = "box"     # box | cylinder (3D only: axis z, R = min(Lx,Ly)/2, centre (Lx/2, Ly/2))
    boundary_top: str = "wall"      # wall | free (open top; z is up in 3D, y is up in 2D "dish slice")
    boundary_functionalization: float = 0.0   # collagen coverage f of the container wall (0 = inert, V2.7)
    boundary_layer_enabled: bool = False      # immobile granule lattice lining floor + wall (cells bridge to it)
    boundary_layer_radius: float = 0.0        # um; 0 -> mean radius of the smallest mobile species
    boundary_layer_embed_frac: float = 0.5    # 0.5 = hemispheres protrude from the wall
    boundary_layer_seed_cells: bool = True    # seed cells on the layer granules (per the seeding law)

    # ── Granule physical properties ──
    R_func_mean: float = 40.0       # functional radius (µm)
    R_func_std: float = 5.0
    R_inert_mean: float = 60.0      # inert radius (µm)
    R_inert_std: float = 8.0

    # ── Composition ──
    # Option A: set phi_f_target + phi_i_target directly.
    # Option B: set phi_solid_target + func_ratio to derive them.
    #   phi_f = phi_solid * func_ratio, phi_i = phi_solid * (1 - func_ratio)
    # If phi_solid_target is set (> 0), it overrides phi_f/phi_i targets.
    phi_f_target: float = 0.25      # functional area fraction target
    phi_i_target: float = 0.20      # inert area fraction target
    phi_solid_target: float = 0.0   # total solid fraction (0 = use phi_f/phi_i directly)
    func_ratio: float = 0.5         # fraction of solids that are functional (0-1)

    # ── Granule species & functionalization (V3.0) ──
    # Every granule belongs to a species carrying a degree of collagen-I
    # functionalization f ∈ [0,1] (1 = fully coated, 0 = bare). `species` is a
    # list of plain dicts (name, f, volume_fraction, color, radius_*, shape
    # moments, E_kPa, poisson_ratio) — see resolve_species(). When empty, the
    # legacy two-species model is derived from R_func_*/R_inert_*/func_ratio
    # (species 0 = collagen f=1, species 1 = bare f=0), reproducing V2.7.
    species: list = field(default_factory=list)
    f_min_adhesion: float = 0.05        # f below which cells cannot attach to or grip a granule
    packing_count_rule: str = "mean_volume"  # mean_volume | mean_radius (V2.7 parity)

    # ── Cell geometry ──
    n_cells_per_granule: int = 8    # cells seeded per functional granule
    cell_diameter: float = 20.0     # µm, initial spherical cell diameter
    cell_height_spread: float = 5.0 # µm, height of spread ellipsoidal cell
    cell_coverage: float = 0.6      # max fraction of granule projected area covered
    cell_surface_coverage: float = 0.0  # target surface coverage fraction (0=use n_cells_per_granule; 0.5–1.5 typical; >1 = stacking)

    # ── Functionalization → cell response (V3.0; rules in gels/materials.py) ──
    # Traction saturates in ABSOLUTE ligand density σ = f·σ_max (Langmuir),
    # not linearly in the coating fraction: g(f) = f(1+κ)/(f+κ), κ = K_σ/σ_max.
    # With the defaults g(0.2) ≈ 0.59. See CodeLog/References/fibroblast_parameters.md.
    sigma_ligand_max: float = 750.0     # collagen-I molecules/µm² at f = 1 (Gaudet 2003 upper point)
    K_sigma_traction: float = 160.0     # molecules/µm², half-saturation for traction (Gaudet σ*)
    K_sigma_attach: float = 50.0        # molecules/µm², half-saturation for cell attachment
    traction_f_law: str = "langmuir"    # langmuir | power
    traction_f_rule: str = "target"     # target | min | product — which f a bridging cell grips with
    traction_exponent: float = 1.0      # γ in g = f^γ (power law only)
    cell_seeding_law: str = "langmuir"  # langmuir | power — cells per granule vs f
    cell_seeding_exponent: float = 1.0  # power law only
    bridge_lock_scales_with_ligand: bool = True  # lock-in force threshold × g (f_c ∝ bond number)

    # ── Cell attachment & spreading timeline ──
    t_attach_onset: float = 0.0     # hours, when cells begin attaching
    t_attach_half: float = 0.0      # hours after onset for 50% attachment (0 = instant)
    t_spread_duration: float = 3.0  # hours for attached cell to fully spread
    fa_maturation_rate: float = 0.3 # 1/h, focal adhesion maturation rate

    # ── Motor-clutch model (Chan & Odde 2008) ──
    n_motors: int = 50              # motor complexes (stress fibers) per cell
    F_motor_stall: float = 0.5     # nN per complex (500 pN / stress fiber)
    n_clutches: int = 75            # integrin clutch clusters per cell
    k_clutch: float = 5.0          # nN/µm, clutch cluster spring constant
    k_on_clutch: float = 1.0       # 1/s, clutch binding rate
    k_off_clutch: float = 0.1      # 1/s, baseline clutch unbinding rate
    F_bond: float = 0.002          # nN (2 pN), characteristic bond rupture force

    # ── Cell migration ──
    cell_migration_speed: float = 5.0  # µm/h, random walk speed on granule surface

    # ── Cell sensing & bridging ──
    cell_sense_distance: float = 40.0  # µm, filopodia sensing range
    F_max_per_cell: float = 50.0       # nN, absolute max force cap per cell
    L_rest: float = 5.0                # µm, rest length (~ spread cell thickness)

    # ── Bridge formation kinetics ──
    bridge_attempt_rate: float = 0.3   # 1/h, Poisson rate per eligible cell per timestep
    bridge_formation_time: float = 2.0 # h, ramp time for new bridge to reach full force
    bridge_senescence_time: float = 24.0  # h, sustained bridge load → senescence
    min_fa_for_bridge: float = 0.3     # min FA maturity to attempt bridging
    bridge_break_gap: float = 60.0     # µm, gap at which committed bridge ruptures
    bridge_lock_force_threshold: float = 20.0  # nN, force above which bridging cells lock in (won't go senescent)
    bridge_secondary_rate_mult: float = 3.0    # rate multiplier for new bridges when existing bridge present
    expected_bridge_force: float = 100.0       # nN, expected force per bridging cell (diagnostic reference)
    bridge_contact_factor: float = 5.0         # path factor: boost at direct granule contact
    bridge_decay_length: float = 10.0          # µm, characteristic filopodia reach (void decay length)
    bridge_inert_factor: float = 0.1           # path factor: penalty when inert blocks line-of-sight
    bridge_commit_angle: float = 0.5           # rad (~29°), angular threshold for MIGRATING→BRIDGING
    bridge_directed_speed_mult: float = 2.0    # directed crawl speed = cell_migration_speed × this
    bridge_exclusion_angle: float = 0.8        # rad (~46°), min angular separation between bridges on same granule→target
    bridge_alignment_rate: float = 0.3         # 1/h, rate of stress fiber alignment along bridge axis
    bridge_alignment_min: float = 0.3          # initial alignment factor when bridge first forms
    cell_stacking_max: float = 3.0             # max cell layers before overcrowding (1.0 = monolayer)
    overcrowd_senescence_time: float = 12.0    # h, sustained overcrowding before senescence
    cell_capacity_coverage: float = 0.0        # capacity coverage for overcrowding/division; 0 -> cell_surface_coverage

    # -- Contractile bridge element (V3.1) --
    # constant: V2.7 actuator (force independent of gap, strain and velocity).
    # hill: pair-implicit force-velocity law - isometric stall force once the
    # environment stops yielding, closing speed bounded by v0, eccentric branch
    # when the bridge is stretched. Reduces exactly to constant as v0 -> inf.
    bridge_force_model: str = "constant"       # constant | hill
    cell_contraction_speed: float = 12.0       # um/h, unloaded shortening speed v0 (0.1-0.5 um/min)
    bridge_eccentric_gain: float = 0.5         # extra force fraction when stretched (F up to (1+gain) F_s)
    bridge_min_gap: float = 0.0                # um, gap below which active shortening stops (isometric hold)

    # -- Cell division (V3.1) --
    cell_division_enabled: bool = False
    cell_doubling_time: float = 24.0           # h, population doubling time (human dermal fibroblasts 20-30 h)
    cell_division_min_age: float = 8.0         # h, refractory period after birth / last division
    cell_division_max_layers: float = 1.0      # contact inhibition: divide only while n < capacity * max_layers
    cell_divide_while_bridging: bool = True    # bridging cells may divide (they round up transiently and re-spread)

    # -- Cell cycle (V3.2): a seeded population is not synchronised --
    cell_division_model: str = 'poisson'       # poisson (memoryless, V3.1) | cycle (per-cell cycle length)
    cell_division_cv: float = 0.0              # lognormal CV of the per-cell cycle length (cycle model; 0 = uniform T_d)
    cell_clock_jitter: float = 0.0             # h, per-granule offset on the attachment / spreading / FA clock

    # ── Contact mechanics (Hertzian + MC-DEM) ──
    E_modulus: float = 10.0         # kPa, Young's modulus of hydrogel
    poisson_ratio: float = 0.45     # Poisson's ratio (hydrogels ~0.4-0.5)
    mc_dem_enabled: bool = True     # enable multi-contact DEM stiffening (Giannis 2021)
    mc_dem_kappa_max: float = 5.0   # max confinement correction factor (caps stiffening)
    contact_E_cap: float = 0.0      # kPa, numerical cap on the CONTACT modulus only (0 = none); cells see the true E
    friction_mu: float = 0.0        # Coulomb coefficient added to the shear-stress friction (0 = hydrogel law only)
    contact_overlap_model: str = 'fixed'   # fixed (max_overlap_frac as given) | elastic (derived from the contact law, V3.2)
    contact_overlap_safety: float = 1.5    # elastic only: headroom over the equilibrium overlap
    contact_semi_implicit: bool = True     # V3.4 (was False in V3.2-3.3): damp the step by the
                                           # local contact stiffness. Measured across the four
                                           # reference configs, the fraction of granules pinned
                                           # at the velocity cap falls 0.18-0.35 -> 0.000, so a
                                           # run reports the contact law instead of the rail.
    cell_contact_adhesion: float = 0.0     # nN per mature bridging cell added to that contact's shear cap
    packing_shape_contact: bool = False    # V3.2: shape-aware settle (true r_bound + directional overlap + rotation)
    packing_shape_margin: float = 0.0      # inflate every directional radius by (1 + margin); blunt, see the plan
    contact_shape_dynamics: bool = False   # V3.2: shape-aware overlap projection, wall clamp and bed surface
    contact_wall_torque: bool = True       # V3.4: apply r x F at the wall contact point.
                                           # Free now that the support function returns the
                                           # contact point. ON by default: discarding the lever
                                           # arm was an omission, not a modelling choice, and a
                                           # shaped granule that cannot tip flat against a wall
                                           # is wrong. Both twins AND this with `not is_circle`,
                                           # and a sphere touches a wall on its own centre line,
                                           # so its wall torque is identically zero -- the
                                           # default is therefore exactly "on for non-spherical
                                           # granules" and no sphere run can move.
    curvature_R_cap: float = 2.0           # V3.4: cap R_eff at cap*min(r_i,r_j) (0 = off). Was 0.0
                                           # in V3.2-3.3. Cannot bind for spheres --
                                           # R_eff = r_i r_j/(r_i+r_j) <= min(r_i,r_j) < 2 min --
                                           # so this only affects shapes, where it is REQUIRED:
                                           # the true curvature radius at a flat face is ~1e15 um.
    cell_capacity_foothold: float = 1.0    # V3.2: fraction of the spread footprint a cell needs to HOLD a place

    # ── LS-DEM deformable granules (V2.3, Henzel & Karapiperis 2026) ──
    deformable_enabled: bool = False       # master switch (False = V2.2 rigid)
    n_def_modes: int = 2                   # deformation modes per particle (2D: 2, 3D: 3)
    sdf_resolution: int = 32              # SDF grid points per axis (body frame)
    n_surface_nodes: int = 128            # surface discretization nodes (2D)
    n_surface_nodes_3d: int = 512         # surface discretization nodes (3D)
    sdf_padding: float = 1.3             # SDF grid extent = padding * max_semi_axis
    def_drag_scale: float = 0.1          # deformation drag γ_def scaling
    def_eps_max: float = 0.3             # max deformation amplitude |ε_α| cap

    # ── Hydrogel friction (area-dependent, Gong 2006 / Pitenis 2014) ──
    # Pair values by coating: bb = bare–bare, cb = collagen–bare, cc =
    # collagen–collagen; mixed by coverage in gels.materials.mix_pair. The
    # legacy names tau_0_ii / tau_0_if / tau_0_ff remain as read/write aliases.
    tau_0_bb: float = 50.0          # Pa, bare–bare shear stress (bare Gemini gel)
    tau_0_cb: float = 500.0         # Pa, collagen–bare
    tau_0_cc: float = 2000.0        # Pa, collagen–collagen
    friction_v_ref: float = 1.0     # µm/h, regularisation velocity (tanh smoothing)

    # ── Adhesion energy (JKR; legacy aliases W_adh_ii / W_adh_if / W_adh_ff) ──
    W_adh_bb: float = 0.0005       # J/m², bare–bare (bare hydrogel)
    W_adh_cb: float = 0.001        # J/m², collagen–bare
    W_adh_cc: float = 0.002        # J/m², collagen–collagen

    # ── Drag ──
    eta: float = 1e-3               # Pa·s (water-like medium)
    drag_scale: float = 0.05        # nondim scaling for drag coefficient

    # -- Gravity / buoyancy (V3.1) --
    # Body force W = (rho_granule - rho_medium) g V on every mobile granule (nN),
    # along -z (3D) or -y (2D). The initial bed is sedimented by the packer
    # (packing_consolidation); in the dynamics gravity only biases the bed down.
    gravity_enabled: bool = False
    g_accel: float = 9.81           # m/s^2
    medium_density: float = 1000.0  # kg/m^3 (culture medium ~1000-1007)
    granule_density: float = 1000.0 # kg/m^3 default for species without density_kg_m3 (PMMA 1180, hydrogel ~1050)
    gravity_scale: float = 1.0      # multiplier for numerical experiments (1 = physical)

    # ── Active noise ──
    T_active: float = 5.0           # active temperature (nN·µm)

    # ── Time integration ──
    dt: float = 0.5                 # hours
    t_total: float = 72.0           # hours
    save_every_h: float = 2.0       # save interval (hours)
    v_max: float = 20.0             # µm/hr, velocity cap
    max_overlap_frac: float = 0.15   # max allowed overlap as fraction of min(ri,rj)

    # ── Granule shape (V1.3 — 2D superellipses) ──
    shape_enabled: bool = False                # False = circles/spheres (V1.2 compat)
    aspect_ratio_func_mean: float = 1.0        # a/b ratio for functional granules
    aspect_ratio_func_std: float = 0.0
    aspect_ratio_inert_mean: float = 1.0       # a/b ratio for inert granules
    aspect_ratio_inert_std: float = 0.0
    blockiness_func_mean: float = 2.0          # n1 exponent (2=ellipse, >2=blocky)
    blockiness_func_std: float = 0.0
    blockiness_inert_mean: float = 2.0
    blockiness_inert_std: float = 0.0
    drag_scale_rot: float = 0.05               # rotational drag scaling
    omega_max: float = 1.0                     # rad/h, angular velocity cap (2D)

    # ── Granule shape (V1.4 — 3D superellipsoids) ──
    aspect_ratio_c_func_mean: float = 1.0      # c/a ratio for functional (z-elongation)
    aspect_ratio_c_func_std: float = 0.0
    aspect_ratio_c_inert_mean: float = 1.0     # c/a ratio for inert
    aspect_ratio_c_inert_std: float = 0.0
    blockiness_n2_func_mean: float = 2.0       # n2 exponent (meridional blockiness)
    blockiness_n2_func_std: float = 0.0
    blockiness_n2_inert_mean: float = 2.0
    blockiness_n2_inert_std: float = 0.0
    omega_max_3d: float = 1.0                  # rad/h, angular velocity cap per axis (3D)
    dynamics_gradient_flow: str = 'off'        # V3.5: off | monitor | damped. Overdamped
                                               # dynamics is gradient flow, so the energy must
                                               # fall every step and the work must account for
                                               # the fall. `monitor` records both (and the
                                               # residual, which is every non-conservative
                                               # force: MC-DEM, cell bridges, the overlap
                                               # projection). `damped` additionally divides
                                               # gamma by a controller that halves on an
                                               # ascending step -- the fixed point is untouched.
    dynamics_substep: str = 'auto'             # V3.6: off | auto. `dt` is then the COUPLING
                                               # interval (saving, history, the cell clock the
                                               # user asked for) and the mechanics is advanced
                                               # in `n_sub` substeps chosen so the semi-implicit
                                               # damping number S = dt k/gamma stays near
                                               # `dynamics_substep_target`. At S >> 1 the step
                                               # is stable and its FIXED POINT is exact, but the
                                               # RATE is (1+S)x too slow -- and a compaction
                                               # run's whole answer is a rate. See `substep_count`.
    dynamics_substep_target: float = 0.2       # V3.6: the S the controller aims for.
    dynamics_substep_max: int = 512            # V3.6: ceiling on n_sub, so a pathological
                                               # configuration costs time, not unbounded time.
    dynamics_outlier_speed: float = 8.0        # V3.6: cap a granule's speed at this many times
                                               # the median of granules in ITS OWN coordination
                                               # class (0 = off). A population rail, not an
                                               # absolute one: we do not expect a granule to
                                               # differ much from granules in its own condition,
                                               # and V3.5 measured one wedged granule reading
                                               # 132x the load while the second-worst read 0.97x.

    # ── Packing ──
    boundary_wall_clamp: str = 'contact'   # V3.5: contact | force | legacy. Where the position
                                    # clip sits relative to the wall. 'legacy' is the V2.7
                                    # +0.5 um standoff, which held every granule clear of
                                    # every wall so the wall contact NEVER engaged and the
                                    # clip carried the bed. See `wall_clamp_margin`.
    packing_overlap_tol_model: str = 'fixed'   # V3.5: fixed (0.05 * mean_r, the V2.7 rule) |
                                    # elastic (the penetration at which the DYNAMICS' contact
                                    # law carries the driving load). The V2.7 rule is 28x too
                                    # loose in length and 145x in force on a shaped gravity
                                    # bed. See `settle_overlap_tolerance`.
    packing_relax: str = 'auto'     # V3.5: auto | none | fire. FIRE-relax the packed bed under
                                    # the DYNAMICS' force law before handing it over, so the run
                                    # starts in force balance instead of spending its first
                                    # hours unwinding the packer. `auto` = on whenever the run
                                    # HAS a driving load (gravity or cells) and off otherwise,
                                    # because a bed with no load has a genuinely loose
                                    # equilibrium and relaxing it just lets it expand.
                                    # See `relax_packing` and `relax_mode`.
    packing_relax_force_tol: float = 0.5   # V3.5: stop when max|F| < tol * dynamics_load_scale.
                                           # (a granule's weight, or one cell's traction).
                                           # BELOW 1 on purpose: an unsupported granule has
                                           # |F| = exactly its weight, so tol >= 1 is satisfied
                                           # by a bed in free fall.
    packing_gap: float = 0.0        # µm, min gap between granule surfaces at placement
    boundary_exclusion: float = 0.2  # fraction of domain excluded from each edge for metrics
    # V3.3: Laguerre (radical) local packing fraction in the metrics. Off by
    # default because Qhull cannot be compiled and it costs ~0.3-1 s/frame at
    # N = 1000; the same numbers are available post-hoc from viz2/step5.
    metrics_laguerre: bool = False
    # V3.3: Katz-Thompson critical-pore permeability + geodesic tortuosity.
    # Off by default: 16 ndimage labellings of the FULL grid plus a BFS.
    metrics_pore_field: bool = False
    packing_settle_steps: int = 400  # inflation steps to reach target radii (jammed packing)
    packing_relax_substeps: int = 15  # overlap relaxation sub-steps per inflation step
    packing_inflate_phi_safe: float = 0.20  # initial deflated packing fraction for RSA
    packing_consolidation: str = "centre"   # centre (V2.7 pull toward the box centre) | gravity | none | auto
    bed_height: float = 0.0         # um; > 0 (needs boundary_top free): granule amount as a settled bed height
    bed_phi_assumed: float = 0.60   # packing fraction assumed for the settled bed when converting bed_height to a count

    # ── Rendering ──
    Ngrid: int = 200                # grid for 2D field rendering
    Ngrid_3d: int = 80              # grid for 3D field rendering (Ngrid^3 voxels)
    interface_width: float = 3.0    # µm, tanh smoothing

    # ── Performance (V3.0: compiled parallel kernels, see gels/kernels) ──
    use_numba: bool = True              # False → pure-Python reference path everywhere
    perf_threads: int = 0               # numba threads: 0 = auto (physical cores), -1 = all logical CPUs
    perf_threading_layer: str = "omp"   # omp | tbb | workqueue | default
    perf_neighbor_backend: str = "cells"  # cells | ckdtree | reference
    perf_cells_backend: str = "kernels"   # kernels (compiled state machine + bridging, hashed RNG) | python (exact V2.7 cell code)
    # DEPRECATED (V3.3): there is no Verlet list and there will not be one.
    # half_pairs() is rebuilt every force evaluation. Measured at N=3841 in 3D:
    # a 20 µm skin grows the pair list 1.46× and a step 19.1 → 33.3 ms, while
    # the rebuild it would eliminate is only ~12 % of a step -- so even free
    # rebuilds would be ~53 % slower, and no positive skin breaks even. The
    # cutoff is dominated by L_max (cell sensing), not contact range, so the
    # candidate list is already ~50 neighbours per granule. The field is kept
    # ONLY so stored fixtures' params.json still loads; it is in
    # config.DEPRECATED and is not reachable from a setup file.
    perf_neighbor_skin: float = 20.0
    perf_max_clips: int = 16            # contact clip planes stored per granule (2D; 3D uses 24)
    perf_field_dtype: str = "float32"   # phase-field grid dtype for compiled rendering
    perf_field_res_um: float = 0.0      # grid spacing (µm); 0 → 2·interface_width
    perf_max_grid_2d: int = 4096        # cap on Ng per axis (2D)
    perf_max_grid_3d: int = 256         # cap on Ng per axis (3D)
    perf_metrics_voronoi_stride: int = 4  # sub-grid stride for the Voronoi compartment metric
    perf_keep_snaps_in_memory: bool = False  # keep every snapshot dict in RAM (run() returns them); off: run() returns []
    perf_async_io: bool = True          # write snapshots from a background thread
    perf_io_queue_depth: int = 2
    perf_io_compresslevel: int = 1      # zlib level for snapshot .npz (1 = fast)
    perf_fastmath: bool = True
    perf_warmup: bool = True            # compile every kernel on a tiny system before timing starts

    # ── Data serialization (V1.5, updated V2.6) ──
    save_data: bool = True              # save simulation data to disk
    save_fields: bool = False           # skip field grids during sim; reconstruct at plot time
    output_dir: str = "results/default" # output directory for serialized data
    compress_archive: bool = True       # create .tar.gz at end of simulation
    resume_from: str = ""               # path to run directory to resume from (empty = fresh)

    @property
    def L_max(self):
        """Max interaction distance: max of sensing and bridge break gap."""
        return max(self.cell_sense_distance, self.bridge_break_gap)

    @property
    def save_every(self):
        return max(1, int(self.save_every_h / self.dt))

    def __post_init__(self):
        """Derive phi_f/phi_i from phi_solid + func_ratio if phi_solid_target > 0."""
        if self.phi_solid_target > 0:
            self.phi_f_target = self.phi_solid_target * self.func_ratio
            self.phi_i_target = self.phi_solid_target * (1.0 - self.func_ratio)
        # V3.2: size the overlap rail from the contact law instead of by hand.
        # Baked in here so it is recorded in params.json and applies to a direct
        # Params(...) as well as to a setup file.
        if str(getattr(self, 'contact_overlap_model', 'fixed')) == 'elastic':
            self.max_overlap_frac = elastic_overlap_frac(self)


def _alias_property(target):
    """Read/write alias of another Params field (legacy parameter names)."""
    def fget(self):
        return getattr(self, target)

    def fset(self, value):
        setattr(self, target, value)

    return property(fget, fset, doc=f"legacy alias of {target}")


# V2.7 → V3.0 renames. Aliases are properties (not dataclass fields), so
# params.json and asdict() carry the new names while old trial files, old
# params.json and `p.W_adh_ff`-style code keep working via hasattr/getattr/setattr.
Params.LEGACY_ALIASES = {
    'W_adh_ff': 'W_adh_cc', 'W_adh_if': 'W_adh_cb', 'W_adh_ii': 'W_adh_bb',
    'tau_0_ff': 'tau_0_cc', 'tau_0_if': 'tau_0_cb', 'tau_0_ii': 'tau_0_bb',
}
for _old, _new in Params.LEGACY_ALIASES.items():
    setattr(Params, _old, _alias_property(_new))


# ══════════════════════════════════════════════════════════════════════
# Cell state enum (V1.5)
# ══════════════════════════════════════════════════════════════════════

class CellState(IntEnum):
    """Discrete states for individual cell tracking.  Extensible."""
    ATTACHED = 0        # attached but not yet spreading
    SPREADING = 1       # actively spreading on granule surface
    PROLIFERATING = 2   # fully spread, FA mature
    BRIDGING = 3        # generating traction force across a gap
    SENESCENT = 4       # overcrowded / inactive
    MIGRATING = 5       # directed crawl toward bridge target (V2.3)


# ══════════════════════════════════════════════════════════════════════
# Granule data structure
# ══════════════════════════════════════════════════════════════════════

class GranuleSystem:
    """Tracks all granule and per-granule cell state.  Mode-aware (2D/3D)."""

    # Per-cell arrays: (attribute, dtype, fill value). One registry drives the
    # allocation in __init__, GranuleSystem.add_cells (V3.1 division) and the
    # snapshot round trip, so a new per-cell quantity is added in one place.
    CELL_ARRAYS = (
        ('cell_granule_id', int, 0),
        ('cell_state', int, int(CellState.ATTACHED)),
        ('cell_theta_local', np.float64, 0.0),      # 2D surface angle
        ('cell_eta_local', np.float64, 0.0),        # 3D parametric eta
        ('cell_omega_local', np.float64, 0.0),      # 3D parametric omega
        ('cell_fx', np.float64, 0.0),               # force x (nN)
        ('cell_fy', np.float64, 0.0),               # force y (nN)
        ('cell_fz', np.float64, 0.0),               # force z (nN)
        ('cell_bridge_target', int, -1),            # bridge target granule (-1 = none)
        ('cell_contact_area', np.float64, 0.0),     # footprint area (um^2)
        ('cell_bridge_age', np.float64, 0.0),       # hours in bridge state
        ('cell_bridge_locked', np.bool_, False),    # persistent lock-in flag
        ('cell_overcrowd_age', np.float64, 0.0),    # hours in overcrowded state
        ('cell_alignment', np.float64, 0.0),        # stress fibre alignment (0-1)
        # V3.1
        ('cell_bridge_gap_prev', np.float64, np.nan),  # pair gap at the last bridge service (NaN = unknown)
        ('cell_bridge_force', np.float64, 0.0),     # |F| applied at the last bridge service (nN)
        ('cell_age', np.float64, 0.0),              # hours since birth / last division
        ('cell_generation', np.int8, 0),            # division generation (0 = seeded)
        # V3.2
        ('cell_cycle_time', np.float64, 0.0),       # h, this cell's own cycle length (0 = use cell_doubling_time)
    )

    def __init__(self, x, y, r, gtype, n_cells,
                 z=None,
                 a=None, b=None, c=None,
                 n_shape=None, n1=None, n2=None,
                 theta=None, quat=None,
                 mode="2D",
                 species_id=None, species=None, p=None):
        self.mode = mode
        # V3.0: positions live in one C-contiguous (N,3) array — the layout the
        # compiled kernels read (one cache line per neighbour). x/y/z are
        # write-through column properties (defined after the class body), so
        # every existing `gs.x` read, `gs.x[:] = ...` write and `gs.x = arr`
        # rebind keeps working unchanged.
        self.pos = np.zeros((len(x), 3), dtype=np.float64)
        self.pos[:, 0] = np.asarray(x, dtype=np.float64)
        self.pos[:, 1] = np.asarray(y, dtype=np.float64)
        if z is not None:
            self.pos[:, 2] = np.asarray(z, dtype=np.float64)
        self.r = np.array(r, dtype=np.float64)   # equivalent radius
        self.n_cells = np.array(n_cells, dtype=np.float64)
        self.N = len(x)

        # ── Species / functionalization (V3.0) ──
        # `gtype`, `func_mask` and `inert_mask` are DERIVED from f in
        # set_species_table() (gtype = 0 where f ≥ f_min_adhesion else 1) and
        # kept for every downstream consumer; they are no longer authoritative.
        # Legacy callers pass gtype only: species 0 = collagen (f=1), 1 = bare.
        if species_id is None:
            if gtype is None:
                raise ValueError("GranuleSystem needs species_id or gtype")
            species_id = np.asarray(gtype)
        self.species_id = np.ascontiguousarray(np.asarray(species_id, dtype=np.int32))
        if self.species_id.shape != (self.N,):
            raise ValueError(f"species_id has shape {self.species_id.shape}, expected ({self.N},)")
        self.set_species_table(species, p)

        # ── Shape state ──
        if a is not None:
            self.a = np.array(a, dtype=np.float64)           # semi-axis x
            self.b = np.array(b, dtype=np.float64)           # semi-axis y
            self.c = np.array(c, dtype=np.float64) if c is not None else self.a.copy()
            # n1 = equatorial blockiness, n2 = meridional blockiness
            if n1 is not None:
                self.n1 = np.array(n1, dtype=np.float64)
                self.n2 = np.array(n2, dtype=np.float64)
            elif n_shape is not None:
                self.n1 = np.array(n_shape, dtype=np.float64)
                self.n2 = np.array(n_shape, dtype=np.float64)
            else:
                self.n1 = np.full(self.N, 2.0)
                self.n2 = np.full(self.N, 2.0)
            self.n_shape = self.n1   # backward compat alias for 2D code
            self.is_circle = False
        else:
            self.a = self.r.copy()
            self.b = self.r.copy()
            self.c = self.r.copy()
            self.n1 = np.full(self.N, 2.0)
            self.n2 = np.full(self.N, 2.0)
            self.n_shape = self.n1
            self.is_circle = True

        if mode == "3D":
            self.r_bound = np.maximum(np.maximum(self.a, self.b), self.c)
        else:
            self.r_bound = np.maximum(self.a, self.b)

        # ── Orientation ──
        if mode == "3D":
            if quat is not None:
                self.quat = np.array(quat, dtype=np.float64)
            else:
                self.quat = np.zeros((self.N, 4))
                self.quat[:, 0] = 1.0  # identity quaternion (w,x,y,z)
            self.omega_3d = np.zeros((self.N, 3))   # angular velocity (rad/h)
            self.theta = np.zeros(self.N)  # not used in 3D, but present for compat
            self.omega = np.zeros(self.N)
        else:
            self.theta = np.array(theta, dtype=np.float64) if theta is not None else np.zeros(self.N)
            self.omega = np.zeros(self.N)
            self.quat = None
            self.omega_3d = None

        # ── Cell state (per granule) ──
        self.n_attached = np.zeros(self.N)
        self.spread_fraction = np.zeros(self.N)
        self.fa_maturity = np.zeros(self.N)
        self.n_overcrowded = np.zeros(self.N)
        # V3.2: per-granule offset on the attachment / spreading / FA clock, so the
        # bed does not mature in lockstep. Zero unless cell_clock_jitter > 0.
        self.cell_clock_offset = np.zeros(self.N)

        # V3.2 numerical-rail diagnostics (set by _resolve_overlaps and step)
        self.n_overlap_clipped = 0
        self.n_overlap_pairs = 0
        self.frac_velocity_clipped = 0.0

        # V3.6: per-granule wall contact stiffness (nN/um), written by whichever
        # force evaluation ran last. Walls are applied inline in the gathers and
        # never reach the pair contact list, so this is the only route by which
        # the semi-implicit step can learn that a granule is held by a wall.
        self.wall_stiffness = np.zeros(self.N)

        # V3.6 substep controller. `stiffness_rate_*` is k/gamma in 1/h, recorded
        # by the step just taken and read by `substep_count` to size the next --
        # free of dt, so measuring it on one step and using it on another is
        # sound. `frac_outlier_clipped` is the population speed rail's own
        # diagnostic, kept separate from `frac_velocity_clipped` so the absolute
        # rail and the relative one are never confused for each other.
        self.stiffness_rate_p95 = 0.0
        self.stiffness_rate_max = 0.0
        self.n_substeps = 1
        self.dt_substep = 0.0
        self.substep_budget_bound = False
        self.substep_rate_used = 0.0
        self.frac_outlier_clipped = 0.0

        # V3.5 gradient-flow audit (set by `step` when dynamics.gradient_flow
        # is on; `energy_metrics` reads them and they stay zero when it is off)
        self.energy_total = 0.0
        self.energy_contact = 0.0
        self.energy_wall = 0.0
        self.energy_gravity = 0.0
        self.energy_noise_expected = 0.0
        self.energy_delta = 0.0
        self.energy_work = 0.0
        self.energy_residual = 0.0
        self.energy_ascent_frac = 0.0
        self.energy_ascent_steps = 0
        self.energy_max_ascent_frac = 0.0
        self.energy_backtracks = 0
        self.energy_prev = None          # None until the first audited step
        self.energy_step_scale = 1.0     # 'damped' mode: gamma is divided by this

        # ── Velocity state (N,3); vx/vy/vz are column properties ──
        self.vel = np.zeros((self.N, 3), dtype=np.float64)

        # ── Unwrapped positions (for displacement tracking with periodic BC) ──
        self.pos_unwrap = self.pos.copy()   # x_unwrap/y_unwrap/z_unwrap are column properties

        # ── LS-DEM deformation state (V2.3, allocated by lsdem.init_lsdem) ──
        self.epsilon = None              # (N, n_def_modes) deformation amplitudes
        self.d_epsilon = None            # (N, n_def_modes) time derivatives
        self.sdf_grids = None            # list of N ndarrays (body-frame SDF)
        self.sdf_extents = None          # (N, ndim, 2) grid bounding boxes
        self.surface_nodes_body = None   # (N, n_nodes, ndim) reference positions
        self.surface_normals_body = None # (N, n_nodes, ndim) outward normals
        self.surface_node_areas = None   # (N, n_nodes) area weights
        self.mode_shapes_at_nodes = None # (N, n_nodes, n_modes, ndim)
        self.K_def = None                # (N, n_modes, n_modes) stiffness matrices
        self.F_eps_accum = None          # (N, n_modes) accumulated deformation forces

        # ── JKR contact clip planes for rendering (V2.3) ──
        # Each element is a list of (nx, ny [,nz], clip_distance) tuples.
        # Populated during compute_forces(), consumed by render_fields().
        self.contact_clips = [[] for _ in range(self.N)]

        # ── Per-cell tracking (V1.5) ──
        total_cells = int(np.sum(n_cells))
        self.total_cells = total_cells
        # Cell-to-granule offset: cells for granule i live at
        #   cell_offset[i] : cell_offset[i+1]
        offsets = np.zeros(self.N + 1, dtype=int)
        for k in range(self.N):
            offsets[k + 1] = offsets[k] + int(n_cells[k])
        self.cell_offset = offsets
        # Per-cell arrays (see CELL_ARRAYS)
        for name, dtype, fill in self.CELL_ARRAYS:
            setattr(self, name, np.full(total_cells, fill, dtype=dtype))
        # V3.1 counters
        self.n_divisions_cum = 0         # cell divisions since t = 0
        self.n_top_clamped = 0           # granules clamped at the container top since the last save

    # ── V3.0 species table ──────────────────────────────────────────────
    def set_species_table(self, species=None, p=None):
        """(Re)build all per-granule functionalization state from `species_id`.

        Parameters
        ----------
        species : list of dict or None
            Species records (see resolve_species). None → resolve from `p`.
        p : Params or None
            Source of global E/ν, pair adhesion/friction, f_min_adhesion and
            the seeding law. None → Params() defaults.
        """
        from gels import materials as _mat

        if p is None:
            p = Params()
        if species is None:
            species = resolve_species(p)
        K = len(species)
        if K == 0:
            raise ValueError("species list is empty")
        if self.N and int(self.species_id.max()) >= K:
            raise ValueError(f"species_id up to {int(self.species_id.max())} but only "
                             f"{K} species defined")
        self.species = species
        self.K = K
        self.species_names = [str(sp.get('name', f'species_{k}')) for k, sp in enumerate(species)]
        self.species_colors = [str(sp.get('color', '#888888')) for sp in species]
        self.species_f = np.array([float(sp['f']) for sp in species], dtype=np.float64)
        self.species_E = np.array([float(sp['E_kPa']) if sp.get('E_kPa') is not None
                                   else float(p.E_modulus) for sp in species], dtype=np.float64)
        self.species_nu = np.array([float(sp['poisson_ratio']) if sp.get('poisson_ratio') is not None
                                    else float(p.poisson_ratio) for sp in species], dtype=np.float64)
        # V3.1: density (gravity) and immobile boundary-layer species
        self.species_rho = np.array([float(sp['density_kg_m3']) if sp.get('density_kg_m3') is not None
                                     else float(getattr(p, 'granule_density', 1000.0)) for sp in species],
                                    dtype=np.float64)
        self.species_fixed = np.array([bool(sp.get('fixed', False)) for sp in species], dtype=np.bool_)

        sid = self.species_id
        self.f = np.ascontiguousarray(self.species_f[sid])
        self.E_gran = np.ascontiguousarray(self.species_E[sid])
        self.nu_gran = np.ascontiguousarray(self.species_nu[sid])
        self.rho_gran = np.ascontiguousarray(self.species_rho[sid])
        self.fixed = np.ascontiguousarray(self.species_fixed[sid])
        self.adhesive_mask = self.f >= p.f_min_adhesion
        self.species_adhesive = self.species_f >= p.f_min_adhesion
        self.activity = seeding_gain(self.f, p)
        # Derived binary view kept for viz/analysis/metrics (int64 as in V2.7)
        self.gtype = np.where(self.adhesive_mask, 0, 1).astype(int)
        self.func_mask = self.gtype == 0
        self.inert_mask = self.gtype == 1

        tables = _mat.build_pair_tables(
            self.species_f, self.species_E, self.species_nu,
            p.W_adh_cc, p.W_adh_cb, p.W_adh_bb, p.tau_0_cc, p.tau_0_cb, p.tau_0_bb,
            E_cap_kPa=float(getattr(p, 'contact_E_cap', 0.0)),
            f_wall=float(getattr(p, 'boundary_functionalization', 0.0)))
        self.pair_W = tables['pair_W']
        self.pair_tau = tables['pair_tau']
        self.pair_Estar = tables['pair_Estar']
        self.wall_W = tables['wall_W']
        self.wall_Estar = tables['wall_Estar']

    @classmethod
    def from_snapshot_dict(cls, snap, p=None, mode=None):
        """Build a GranuleSystem from a snapshot dict (load_run / run().save()).

        Intended for rendering and analysis after the fact. Reproduces the
        conventions of the former viz shims: `is_circle` uses an exact
        a == b and n == 2 test in 2D and is False in 3D; no contact clips.
        """
        pp = p if p is not None else Params()
        N = len(snap['x'])
        if mode is None:
            mode = snap.get('mode', pp.mode)
        if isinstance(mode, np.ndarray):
            mode = str(mode)
        r = np.asarray(snap['r'], dtype=np.float64)
        a = np.asarray(snap.get('a', r), dtype=np.float64)
        b = np.asarray(snap.get('b', r), dtype=np.float64)
        c = np.asarray(snap.get('c', r), dtype=np.float64)
        n1 = np.asarray(snap.get('n1', snap.get('n_shape', np.full(N, 2.0))), dtype=np.float64)
        n2 = np.asarray(snap.get('n2', n1), dtype=np.float64)
        if 'n_cells' in snap:
            n_cells = snap['n_cells']
        elif 'cell_offset' in snap:
            n_cells = np.diff(np.asarray(snap['cell_offset']))
        else:
            n_cells = np.zeros(N)
        gs = cls(snap['x'], snap['y'], r, snap.get('gtype'), n_cells,
                 z=snap.get('z'), a=a, b=b, c=c, n1=n1, n2=n2,
                 theta=snap.get('theta'), quat=snap.get('quat'), mode=mode,
                 species_id=snap.get('species_id'), p=pp)
        gs.is_circle = bool(mode != "3D" and np.all(a == b) and np.all(n1 == 2.0))
        gs.contact_clips = [[] for _ in range(N)]
        gs.epsilon = snap.get('epsilon', None)
        gs.d_epsilon = snap.get('d_epsilon', None)
        return gs

    def positions(self):
        if self.mode == "3D":
            return np.column_stack([self.x, self.y, self.z])
        return np.column_stack([self.x, self.y])

    def cells_on_granule(self, i):
        """Return slice for per-cell arrays corresponding to granule i."""
        return slice(self.cell_offset[i], self.cell_offset[i + 1])

    def add_cells(self, dest_granule, theta=None, eta=None, omega=None, generation=None):
        """Append cells to the given granules, rebuilding the CSR layout (V3.1).

        The per-cell arrays are one CSR block per granule, so a birth on
        granule i shifts every slot above it. Existing cells keep their
        position within their granule's range (slot k stays slot k) and the
        daughters are appended at the end of it, so nothing that indexes a
        cell by its rank within a granule changes meaning.
        ``cell_bridge_target`` holds GRANULE indices, so it is invariant.

        Returns the indices of the new cells in the rebuilt arrays.
        """
        dest = np.asarray(dest_granule, dtype=int).ravel()
        if dest.size == 0:
            return np.zeros(0, dtype=int)
        N, C_old = self.N, self.total_cells
        births = np.bincount(dest, minlength=N)
        n_old = np.diff(self.cell_offset).astype(int)
        off_new = np.zeros(N + 1, dtype=self.cell_offset.dtype)
        np.cumsum(n_old + births, out=off_new[1:])
        C_new = int(off_new[-1])

        # old slot k of granule g -> new slot k of granule g. The CSR offsets
        # are authoritative here, not cell_granule_id, which a caller may not
        # have filled in yet (_initialize_cells does it at packing time).
        gid_old = np.repeat(np.arange(N), n_old)
        rank_old = np.arange(C_old) - self.cell_offset[gid_old]
        map_old = off_new[gid_old] + rank_old

        # daughters: appended after the granule's existing cells, in call order
        order = np.argsort(dest, kind='stable')
        dest_sorted = dest[order]
        rank_new = np.arange(dest.size) - np.searchsorted(dest_sorted, dest_sorted)
        new_idx_sorted = off_new[dest_sorted] + n_old[dest_sorted] + rank_new
        new_idx = np.empty(dest.size, dtype=int)
        new_idx[order] = new_idx_sorted

        for name, dtype, fill in self.CELL_ARRAYS:
            old = getattr(self, name)
            arr = np.full(C_new, fill, dtype=dtype)
            if C_old:
                arr[map_old] = old[:C_old]
            setattr(self, name, arr)

        self.cell_offset = off_new
        self.total_cells = C_new
        self.n_cells = (n_old + births).astype(np.float64)
        self.cell_granule_id[new_idx] = dest
        self.cell_state[new_idx] = int(CellState.ATTACHED)
        self.cell_bridge_target[new_idx] = -1
        if theta is not None:
            self.cell_theta_local[new_idx] = np.asarray(theta, dtype=np.float64)
        if eta is not None:
            self.cell_eta_local[new_idx] = np.asarray(eta, dtype=np.float64)
        if omega is not None:
            self.cell_omega_local[new_idx] = np.asarray(omega, dtype=np.float64)
        if generation is not None:
            self.cell_generation[new_idx] = np.asarray(generation, dtype=np.int8)
        return new_idx

    @property
    def is_3d(self):
        return self.mode == "3D"


def _column_property(store, col, doc):
    """Write-through view of column `col` of the (N,3) array `self.<store>`.

    Reading returns a strided view (so in-place writes like ``gs.x[i] += dx``
    hit the shared storage). Assigning a whole array (``gs.x = arr``) copies
    into the column instead of rebinding, allocating the storage first when
    the object was built without ``__init__`` (the viz shims do this).
    """
    def fget(self):
        return getattr(self, store)[:, col]

    def fset(self, value):
        value = np.asarray(value, dtype=np.float64)
        arr = self.__dict__.get(store)
        if value.ndim == 1 and (arr is None or arr.shape[0] != value.shape[0]):
            new = np.zeros((value.shape[0], 3), dtype=np.float64)
            if arr is not None:
                m = min(arr.shape[0], new.shape[0])
                new[:m] = arr[:m]
            self.__dict__[store] = new
            arr = new
        arr[:, col] = value

    return property(fget, fset, doc=doc)


GranuleSystem.x = _column_property('pos', 0, "x positions (µm) — view of pos[:, 0]")
GranuleSystem.y = _column_property('pos', 1, "y positions (µm) — view of pos[:, 1]")
GranuleSystem.z = _column_property('pos', 2, "z positions (µm) — view of pos[:, 2]")
GranuleSystem.vx = _column_property('vel', 0, "x velocities (µm/h) — view of vel[:, 0]")
GranuleSystem.vy = _column_property('vel', 1, "y velocities (µm/h) — view of vel[:, 1]")
GranuleSystem.vz = _column_property('vel', 2, "z velocities (µm/h) — view of vel[:, 2]")
GranuleSystem.x_unwrap = _column_property('pos_unwrap', 0, "unwrapped x — view of pos_unwrap[:, 0]")
GranuleSystem.y_unwrap = _column_property('pos_unwrap', 1, "unwrapped y — view of pos_unwrap[:, 1]")
GranuleSystem.z_unwrap = _column_property('pos_unwrap', 2, "unwrapped z — view of pos_unwrap[:, 2]")


# ══════════════════════════════════════════════════════════════════════
# Species helpers (V3.0)
# ══════════════════════════════════════════════════════════════════════

LEGACY_SPECIES_COLORS = ('#CC2222', '#22AA22')   # collagen (red), bare (green) — V2.7 viz colours


def legacy_species_from_params(p):
    """Two species reproducing the V2.7 functional/inert model from a Params.

    Species 0 = 'collagen' (f = 1) from the R_func_* / *_func_* fields,
    species 1 = 'bare' (f = 0) from the R_inert_* / *_inert_* fields. Volume
    fractions follow func_ratio (or phi_f/(phi_f+phi_i)). Values are plain
    Python scalars so the list survives asdict() → JSON → list().
    """
    if p.phi_solid_target > 0:
        vf_c = float(p.func_ratio)
        # Same expression as Params.__post_init__, so these equal the derived
        # phi_f_target / phi_i_target bit for bit.
        phi_c = p.phi_solid_target * p.func_ratio
        phi_b = p.phi_solid_target * (1.0 - p.func_ratio)
    else:
        tot = p.phi_f_target + p.phi_i_target
        vf_c = float(p.phi_f_target / tot) if tot > 0 else 0.5
        phi_c = float(p.phi_f_target)
        phi_b = float(p.phi_i_target)
    return [
        dict(name='collagen', f=1.0, volume_fraction=vf_c, phi_target=phi_c,
             count_rule='mean_radius', color=LEGACY_SPECIES_COLORS[0],
             radius_mean=float(p.R_func_mean), radius_std=float(p.R_func_std),
             radius_min=15.0, radius_distribution='normal',
             aspect_ratio_mean=float(p.aspect_ratio_func_mean),
             aspect_ratio_std=float(p.aspect_ratio_func_std),
             aspect_ratio_c_mean=float(p.aspect_ratio_c_func_mean),
             aspect_ratio_c_std=float(p.aspect_ratio_c_func_std),
             blockiness_mean=float(p.blockiness_func_mean),
             blockiness_std=float(p.blockiness_func_std),
             blockiness_n2_mean=float(p.blockiness_n2_func_mean),
             blockiness_n2_std=float(p.blockiness_n2_func_std),
             E_kPa=None, poisson_ratio=None),
        dict(name='bare', f=0.0, volume_fraction=1.0 - vf_c, phi_target=phi_b,
             count_rule='mean_radius', color=LEGACY_SPECIES_COLORS[1],
             radius_mean=float(p.R_inert_mean), radius_std=float(p.R_inert_std),
             radius_min=20.0, radius_distribution='normal',
             aspect_ratio_mean=float(p.aspect_ratio_inert_mean),
             aspect_ratio_std=float(p.aspect_ratio_inert_std),
             aspect_ratio_c_mean=float(p.aspect_ratio_c_inert_mean),
             aspect_ratio_c_std=float(p.aspect_ratio_c_inert_std),
             blockiness_mean=float(p.blockiness_inert_mean),
             blockiness_std=float(p.blockiness_inert_std),
             blockiness_n2_mean=float(p.blockiness_n2_inert_mean),
             blockiness_n2_std=float(p.blockiness_n2_inert_std),
             E_kPa=None, poisson_ratio=None),
    ]


def resolve_species(p):
    """The species list of a run: `p.species` if set, else the legacy pair.

    The legacy pair is cached on `p.species` so params.json, metadata and viz
    all see the same table.
    """
    if p.species:
        for k, sp in enumerate(p.species):
            if 'f' not in sp:
                raise ValueError(f"species[{k}] has no 'f' (functionalization)")
        return _with_boundary_species(p, p.species)
    p.species = legacy_species_from_params(p)
    return _with_boundary_species(p, p.species)


def _with_boundary_species(p, species):
    """Append the immobile boundary species when the wall lining is enabled (V3.1)."""
    if not getattr(p, 'boundary_layer_enabled', False):
        return species
    if any(str(sp.get('name')) == BOUNDARY_SPECIES_NAME for sp in species):
        return species
    species.append(boundary_layer_species(p, species))
    return species


BOUNDARY_SPECIES_NAME = 'boundary'
BOUNDARY_SPECIES_COLOR = '#8C8C8C'


def boundary_layer_radius(p, species):
    """Radius of a boundary-layer granule: the explicit value, else the smallest mobile mean."""
    r = float(getattr(p, 'boundary_layer_radius', 0.0) or 0.0)
    if r > 0:
        return r
    means = [float(sp['radius_mean']) for sp in species
             if str(sp.get('name')) != BOUNDARY_SPECIES_NAME]
    return min(means) if means else 40.0


def boundary_layer_species(p, species):
    """Species record of the immobile wall lining (f = boundary functionalization)."""
    r = boundary_layer_radius(p, species)
    return dict(name=BOUNDARY_SPECIES_NAME, f=float(getattr(p, 'boundary_functionalization', 0.0)),
                volume_fraction=0.0, phi_target=0.0, color=BOUNDARY_SPECIES_COLOR,
                radius_mean=r, radius_std=0.0, radius_min=r, radius_distribution='normal',
                aspect_ratio_mean=1.0, aspect_ratio_std=0.0,
                aspect_ratio_c_mean=1.0, aspect_ratio_c_std=0.0,
                blockiness_mean=2.0, blockiness_std=0.0,
                blockiness_n2_mean=2.0, blockiness_n2_std=0.0,
                E_kPa=None, poisson_ratio=None, count_rule='mean_radius',
                density_kg_m3=None, fixed=True)


def _hex_rows(length, pitch):
    """Row offsets and per-row counts of a hexagonal lattice spanning `length`."""
    n = max(1, int(np.floor(length / pitch)))
    return n, (length - (n - 1) * pitch) / 2.0 if n > 1 else length / 2.0


def boundary_layer_sites(p, r_w, mode):
    """Centres of the immobile wall lining: floor first, then the side wall(s).

    Granules are embedded so that ``embed_frac`` of each one sits inside the
    wall (0.5 = hemispheres protruding into the container); the lattice pitch
    is one diameter, so the lining is a closed, rough surface rather than a
    smooth plane. 2D lines the floor (y = 0) and the two side walls; 3D lines
    the floor (z = 0) and either the four box faces or the cylinder.
    """
    geom = boundary_geometry(p, mode)
    inset = r_w * (1.0 - 2.0 * float(p.boundary_layer_embed_frac))
    pitch = 2.0 * r_w
    sites = []
    if geom.dim == 2:
        nx, x0 = _hex_rows(p.Lx, pitch)
        for i in range(nx):
            sites.append((x0 + i * pitch, inset, 0.0))
        ny, y0 = _hex_rows(p.Ly, pitch)
        for j in range(ny):
            y = y0 + j * pitch
            if y <= inset + 1e-9:
                continue                   # already covered by the floor row
            sites.append((inset, y, 0.0))
            sites.append((p.Lx - inset, y, 0.0))
        return sites
    # ── 3D floor ──
    if geom.shape_code == 1:
        R_in = geom.R_cyl - inset
        nx, x0 = _hex_rows(2.0 * R_in, pitch)
        ny, y0 = _hex_rows(2.0 * R_in, pitch)
        for i in range(nx):
            for j in range(ny):
                x = geom.cx - R_in + x0 + i * pitch
                y = geom.cy - R_in + y0 + j * pitch
                if (x - geom.cx) ** 2 + (y - geom.cy) ** 2 <= R_in ** 2:
                    sites.append((x, y, inset))
    else:
        nx, x0 = _hex_rows(p.Lx, pitch)
        ny, y0 = _hex_rows(p.Ly, pitch)
        for i in range(nx):
            for j in range(ny):
                sites.append((x0 + i * pitch, y0 + j * pitch, inset))
    # ── 3D side wall, in rings / rows above the floor layer ──
    nz, z0 = _hex_rows(p.Lz, pitch)
    for k in range(nz):
        z = z0 + k * pitch
        if z <= inset + pitch * 0.5:
            continue                       # the floor layer already occupies this height
        if geom.shape_code == 1:
            rho = geom.R_cyl - inset
            n_ring = max(3, int(np.floor(2.0 * np.pi * rho / pitch)))
            for m in range(n_ring):
                ang = 2.0 * np.pi * m / n_ring
                sites.append((geom.cx + rho * np.cos(ang), geom.cy + rho * np.sin(ang), z))
        else:
            nx, x0 = _hex_rows(p.Lx, pitch)
            ny, y0 = _hex_rows(p.Ly, pitch)
            for i in range(nx):
                sites.append((x0 + i * pitch, inset, z))
                sites.append((x0 + i * pitch, p.Ly - inset, z))
            for j in range(ny):
                y = y0 + j * pitch
                if y <= inset + 1e-9 or y >= p.Ly - inset - 1e-9:
                    continue
                sites.append((inset, y, z))
                sites.append((p.Lx - inset, y, z))
    return sites


FORCE_MODEL_CONSTANT = 0
FORCE_MODEL_HILL = 1


def force_model_code(name):
    """'constant' -> 0, 'hill' -> 1 (V3.1 contractile bridge)."""
    return FORCE_MODEL_HILL if str(name).lower() == 'hill' else FORCE_MODEL_CONSTANT


def bridge_pair_factor(gs, gi, gj, gap, p):
    """Hill force-velocity multiplier for every bridge of the pair (gi, gj).

    Returns 1.0 for the constant model, so the V2.7 arithmetic is reproduced
    exactly. See gels.materials.hill_pair_factor for the law.
    """
    if force_model_code(getattr(p, 'bridge_force_model', 'constant')) == FORCE_MODEL_CONSTANT:
        return 1.0
    from gels.materials import hill_pair_factor
    F_iso = 0.0
    F_prev = 0.0
    gap_prev = np.nan
    for gsrc, gdst in ((gi, gj), (gj, gi)):
        for ci in range(gs.cell_offset[gsrc], gs.cell_offset[gsrc + 1]):
            if gs.cell_state[ci] == int(CellState.BRIDGING) and gs.cell_bridge_target[ci] == gdst:
                mat = min(1.0, gs.cell_bridge_age[ci] / max(0.1, p.bridge_formation_time))
                al = max(gs.cell_alignment[ci], p.bridge_alignment_min)
                F_src = _cell_capacity_force(gs, gsrc, gdst, p)
                F_iso += min(F_src * mat * al, p.F_max_per_cell)
                F_prev += gs.cell_bridge_force[ci]
                if gap_prev != gap_prev:                      # first cell with a record
                    gap_prev = gs.cell_bridge_gap_prev[ci]
    if F_iso <= 0.0:
        return 1.0
    v_rel = 0.0 if gap_prev != gap_prev else (gap_prev - gap) / p.dt
    c_pair = 1.0 / (p.drag_scale * gs.r[gi]) + 1.0 / (p.drag_scale * gs.r[gj])
    return float(hill_pair_factor(v_rel, F_prev, F_iso, c_pair,
                                  float(p.cell_contraction_speed), float(p.bridge_eccentric_gain),
                                  bool(gap <= p.bridge_min_gap)))


def _cell_capacity_force(gs, host, target, p):
    """Per-cell traction capacity of a cell on `host` gripping `target` (V3.0 rule)."""
    from gels.materials import law_code, rule_code, traction_gain
    kappa = p.K_sigma_traction / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
    g = traction_gain(gs.f[host], gs.f[target], law_code(p.traction_f_law),
                      rule_code(p.traction_f_rule), p.traction_exponent, kappa)
    return motor_clutch_force(gs.E_gran[host], p, gs.fa_maturity[host],
                              nu=gs.nu_gran[host], g=g)


def capacity_coverage(p):
    """Surface coverage defining the cell CAPACITY of a granule (V3.1).

    Seeding coverage and capacity were the same number in V3.0, so a granule
    seeded at its monolayer density was already confluent and could never
    gain a cell. ``cells.seeding.capacity_coverage`` separates them; 0 keeps
    the V3.0 behaviour (capacity = seeding coverage).
    """
    cap_cov = float(getattr(p, 'cell_capacity_coverage', 0.0) or 0.0)
    return cap_cov if cap_cov > 0 else float(p.cell_surface_coverage)


def seeding_gain(f, p):
    """Fraction of the f = 1 cell capacity that coverage f supports (vectorised).

    langmuir: f(1+κ)/(f+κ) with κ = K_sigma_attach/sigma_ligand_max;
    power:    f ** cell_seeding_exponent.  Both give 0 at f = 0 and 1 at f = 1.
    """
    f = np.asarray(f, dtype=np.float64)
    if str(p.cell_seeding_law).lower() == 'power':
        g = np.where(f > 0.0, np.power(np.clip(f, 0.0, None), p.cell_seeding_exponent), 0.0)
    else:
        kappa = p.K_sigma_attach / p.sigma_ligand_max if p.sigma_ligand_max > 0 else 0.0
        if kappa > 0:
            g = np.where(f > 0.0, f * (1.0 + kappa) / (f + kappa), 0.0)
        else:
            g = np.where(f > 0.0, 1.0, 0.0)
    return np.where(f >= 1.0, 1.0, g)


def species_view(gs):
    """(K, species_id, species_adhesive, species_f) for any granule-like object.

    Full GranuleSystems carry the species table; the lightweight viz shims
    built with ``__new__`` only have ``gtype`` and get the legacy two-species
    view (0 = collagen f=1, 1 = bare f=0).
    """
    if hasattr(gs, 'species_id') and hasattr(gs, 'K'):
        return (int(gs.K), np.asarray(gs.species_id),
                np.asarray(gs.species_adhesive, dtype=bool),
                np.asarray(gs.species_f, dtype=np.float64))
    gt = np.asarray(gs.gtype)
    sid = np.where(gt == 0, 0, 1).astype(np.int32)
    return 2, sid, np.array([True, False]), np.array([1.0, 0.0])


def group_species_fields(gs, phi_s):
    """Collapse per-species grids to (phi_f, phi_i, phi_v).

    phi_f sums the adhesive species (f ≥ f_min_adhesion), phi_i the rest,
    phi_v = 1 − phi_f − phi_i. Summation is in species-index order so a
    two-species run reproduces the V2.7 fields exactly.
    """
    K, _sid, adhesive, _f = species_view(gs)
    phi_f = np.zeros_like(phi_s[0])
    phi_i = np.zeros_like(phi_s[0])
    for k in range(K):
        if adhesive[k]:
            phi_f += phi_s[k]
        else:
            phi_i += phi_s[k]
    return phi_f, phi_i, 1.0 - phi_f - phi_i


def collagen_field(gs, phi_s):
    """Collagen-weighted solid field phi_c = Σ_k f_k · phi_s[k]."""
    K, _sid, _adh, f_sp = species_view(gs)
    phi_c = np.zeros_like(phi_s[0])
    for k in range(K):
        if f_sp[k] != 0.0:
            phi_c += f_sp[k] * phi_s[k]
    return phi_c


def compute_displacement_species(gs, x0, y0, z0=None):
    """Mean displacement per species → {'disp_sp_k': ...} (unwrapped coordinates)."""
    dx = gs.x_unwrap - x0
    dy = gs.y_unwrap - y0
    if gs.is_3d and z0 is not None:
        dz = gs.z_unwrap - z0
        disp = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        disp = np.sqrt(dx**2 + dy**2)
    K, sid, _adh, _f = species_view(gs)
    out = {}
    for k in range(K):
        mask = sid == k
        out[f'disp_sp_{k}'] = float(np.mean(disp[mask])) if np.any(mask) else 0.0
    return out


# ══════════════════════════════════════════════════════════════════════
# Periodic boundary helpers
# ══════════════════════════════════════════════════════════════════════

def minimum_image_disp(dx, dy, Lx, Ly):
    """Minimum image displacement for 2D periodic boundaries."""
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)
    return dx, dy


def minimum_image_disp_3d(dx, dy, dz, Lx, Ly, Lz):
    """Minimum image displacement for 3D periodic boundaries."""
    dx = dx - Lx * np.round(dx / Lx)
    dy = dy - Ly * np.round(dy / Ly)
    dz = dz - Lz * np.round(dz / Lz)
    return dx, dy, dz


def wrap_positions(gs, p):
    """Wrap particle positions into [0, L) using modulo."""
    gs.x[:gs.N] %= p.Lx
    gs.y[:gs.N] %= p.Ly
    if gs.mode == "3D":
        gs.z[:gs.N] %= p.Lz


# ══════════════════════════════════════════════════════════════════════
# Quaternion utilities (V1.4 — 3D rotations)
# ══════════════════════════════════════════════════════════════════════
# Quaternion convention: q = (w, x, y, z) where w is the scalar part.

@njit(cache=True)
def quat_multiply(q1, q2):
    """Hamilton product of two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


@njit(cache=True)
def quat_conjugate(q):
    """Conjugate of quaternion (w, x, y, z) -> (w, -x, -y, -z)."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


@njit(cache=True)
def quat_normalize(q):
    """Normalize quaternion to unit length."""
    n = np.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
    if n < 1e-30:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


@njit(cache=True)
def quat_rotate(q, v):
    """Rotate vector v (3,) by quaternion q: q v q*."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    r = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return r[1:]


@njit(cache=True)
def quat_rotate_inv(q, v):
    """Inverse rotation: q* v q (world to body)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    r = quat_multiply(quat_multiply(quat_conjugate(q), qv), q)
    return r[1:]


@njit(cache=True)
def quat_to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
        [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)]])


def quat_from_axis_angle(axis, angle):
    """Quaternion from rotation axis (3,) and angle (radians)."""
    axis = np.asarray(axis, dtype=np.float64)
    n = np.sqrt(axis[0]**2 + axis[1]**2 + axis[2]**2)
    if n < 1e-30:
        return np.array([1.0, 0.0, 0.0, 0.0])
    axis = axis / n
    ha = angle / 2.0
    return np.array([np.cos(ha), axis[0]*np.sin(ha),
                     axis[1]*np.sin(ha), axis[2]*np.sin(ha)])


def quat_random(rng):
    """Uniformly random unit quaternion."""
    u = rng.random(3)
    q = np.array([
        np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
        np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
        np.sqrt(u[0]) * np.sin(2*np.pi*u[2]),
        np.sqrt(u[0]) * np.cos(2*np.pi*u[2])])
    return quat_normalize(q)


@njit(cache=True)
def quat_integrate(q, omega, dt):
    """
    Integrate quaternion by angular velocity omega (3-vector, rad/h) over dt (h).
    Uses first-order: q_new = q + 0.5 * dt * [0, omega] * q, then normalize.
    """
    omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
    dq = 0.5 * quat_multiply(omega_quat, q) * dt
    q_new = q + dq
    return quat_normalize(q_new)


# ══════════════════════════════════════════════════════════════════════
# Superellipsoid geometry (V1.4 — 3D shapes)
# ══════════════════════════════════════════════════════════════════════
#
# 3D superellipsoid: (|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1
# Parametric surface (eta in [-pi/2, pi/2], omega in [-pi, pi)):
#   x = a * sgnpow(cos(eta), 2/n2) * sgnpow(cos(omega), 2/n1)
#   y = b * sgnpow(cos(eta), 2/n2) * sgnpow(sin(omega), 2/n1)
#   z = c * sgnpow(sin(eta), 2/n2)
# where sgnpow(v, e) = sign(v) * |v|^e

@njit(cache=True)
def _sgnpow(v, e):
    """Signed power: sign(v) * |v|^e, safe for v=0."""
    return np.sign(v) * np.abs(v + 1e-30)**e


def superellipsoid_volume(a, b, c, n1, n2):
    """
    Exact volume of superellipsoid (Jaklic & Leonardis 2000).
    V = 2 * a * b * c * e1 * e2 * B(e1/2 + 1, e1) * B(e2/2, e2/2)
    where e1 = 2/n1, e2 = 2/n2.
    Simplifies for sphere (a=b=c, n1=n2=2): V = (4/3) pi r^3.
    """
    e1 = 2.0 / n1
    e2 = 2.0 / n2
    return (2.0 * a * b * c * e1 * e2 *
            _beta(e1/2.0 + 1.0, e1) * _beta(e2/2.0, e2/2.0))


@njit(cache=True)
def superellipsoid_point(eta, omega, a, b, c, n1, n2):
    """Parametric surface point in body frame."""
    e2 = 2.0 / n2
    e1 = 2.0 / n1
    ce = np.cos(eta)
    se = np.sin(eta)
    co = np.cos(omega)
    so = np.sin(omega)
    x = a * _sgnpow(ce, e2) * _sgnpow(co, e1)
    y = b * _sgnpow(ce, e2) * _sgnpow(so, e1)
    z = c * _sgnpow(se, e2)
    return np.array([x, y, z])


@njit(cache=True)
def superellipsoid_normal(eta, omega, a, b, c, n1, n2):
    """
    Outward unit normal at (eta, omega) in body frame.
    n = (1/a * sgnpow(cos(eta), 2-2/n2) * sgnpow(cos(omega), 2-2/n1),
         1/b * sgnpow(cos(eta), 2-2/n2) * sgnpow(sin(omega), 2-2/n1),
         1/c * sgnpow(sin(eta), 2-2/n2))  normalized.
    """
    e2c = 2.0 - 2.0/n2
    e1c = 2.0 - 2.0/n1
    ce = np.cos(eta)
    se = np.sin(eta)
    co = np.cos(omega)
    so = np.sin(omega)
    nx = (1.0/a) * _sgnpow(ce, e2c) * _sgnpow(co, e1c)
    ny = (1.0/b) * _sgnpow(ce, e2c) * _sgnpow(so, e1c)
    nz = (1.0/c) * _sgnpow(se, e2c)
    mag = np.sqrt(nx*nx + ny*ny + nz*nz)
    if mag < 1e-30:
        return np.array([0.0, 0.0, 1.0])
    return np.array([nx/mag, ny/mag, nz/mag])


@njit(cache=True)
def superellipsoid_curvature_radii(eta, omega, a, b, c, n1, n2):
    """
    Approximate principal radii of curvature at (eta, omega).
    Returns (R1, R2).  For Hertz: R_eff = sqrt(R1*R2) for each body,
    then combined R* = R_eff_i * R_eff_j / (R_eff_i + R_eff_j).
    """
    eps = 1e-8
    # Finite difference approach: sample surface nearby and fit curvature
    pt0 = superellipsoid_point(eta, omega, a, b, c, n1, n2)
    n0 = superellipsoid_normal(eta, omega, a, b, c, n1, n2)

    # Sample in eta and omega directions
    deta = 0.01
    domega = 0.01
    pt_de = superellipsoid_point(eta + deta, omega, a, b, c, n1, n2)
    pt_do = superellipsoid_point(eta, omega + domega, a, b, c, n1, n2)
    n_de = superellipsoid_normal(eta + deta, omega, a, b, c, n1, n2)
    n_do = superellipsoid_normal(eta, omega + domega, a, b, c, n1, n2)

    # Curvature ~ |dn/ds| where ds is arc length
    ds_eta = np.linalg.norm(pt_de - pt0)
    ds_omega = np.linalg.norm(pt_do - pt0)

    if ds_eta > eps:
        dn_eta = np.linalg.norm(n_de - n0)
        kappa1 = dn_eta / ds_eta
    else:
        kappa1 = 1.0 / max(a, b, c)

    if ds_omega > eps:
        dn_omega = np.linalg.norm(n_do - n0)
        kappa2 = dn_omega / ds_omega
    else:
        kappa2 = 1.0 / max(a, b, c)

    R1 = 1.0 / max(kappa1, 1e-6)
    R2 = 1.0 / max(kappa2, 1e-6)
    return R1, R2


@njit(cache=True)
def superellipsoid_implicit(bx, by, bz, a, b, c, n1, n2):
    """
    Implicit function value in body frame.
    Returns < 1 inside, = 1 on boundary, > 1 outside.
    """
    return ((np.abs(bx/a)**n1 + np.abs(by/b)**n1)**(n2/n1) +
            np.abs(bz/c)**n2)


def superellipsoid_implicit_world(px, py, pz, cx, cy, cz, a, b, c, n1, n2, quat):
    """Implicit function value at world point (px, py, pz)."""
    body = quat_rotate_inv(quat, np.array([px-cx, py-cy, pz-cz]))
    return superellipsoid_implicit(body[0], body[1], body[2], a, b, c, n1, n2)


def superellipsoid_mesh(a, b, c, n1, n2, n_pts=24):
    """
    Generate triangle mesh vertices for a superellipsoid (body frame).
    Returns vertices array of shape (n_pts*n_pts, 3).
    """
    eta = np.linspace(-np.pi/2, np.pi/2, n_pts)
    omega = np.linspace(-np.pi, np.pi, n_pts)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e2 = 2.0 / n2
    e1 = 2.0 / n1
    X = a * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    Y = b * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    Z = c * _sgnpow(np.sin(E), e2)
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


# ══════════════════════════════════════════════════════════════════════
# Hertz contact mechanics
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def hertz_contact_force(E_star_Pa, R_eff_um, delta_um):
    """
    Hertzian normal contact force between two elastic spheres.

    F = (4/3) E* √R* δ^{3/2}   [SI: Pa, m, m → N]

    With simulation units (µm, nN):
        F_nN = (4/3) · E*_Pa · √(R*_µm) · δ_µm^{3/2} · 10⁻³
    """
    return (4.0 / 3.0) * E_star_Pa * np.sqrt(R_eff_um) * delta_um**1.5 * 1e-3


# ── JKR (Johnson-Kendall-Roberts) adhesive contact model ──────────
# Correct model for soft hydrogels with high Tabor parameter μ_T >> 1.
# Reduces exactly to Hertz when W_adhesion = 0.

@njit(cache=True)
def jkr_force_from_overlap(delta_um, R_eff_um, E_star_Pa, W_Jm2):
    """
    JKR contact force and contact radius from overlap δ.

    Given overlap δ [µm], solve for contact radius a via Newton-Raphson on:
        δ = a²/R* − √(2πW a / E*)
    Then compute JKR force:
        F = (4/3)E* a³/R* − √(8πW E* a³)

    Units: δ [µm], R_eff [µm], E_star [Pa], W [J/m²]
    Returns: (F [nN], a [µm])

    When W=0, reduces to Hertz: F = (4/3)E*√R* δ^{3/2}, a = √(R*δ).
    """
    E_s = E_star_Pa * 1e-3   # nN/µm²
    W_s = W_Jm2              # nN/µm  (1 J/m² = 1 nN/µm)

    # Pure Hertz fast-path when no adhesion
    if W_s < 1e-15:
        if delta_um <= 0.0:
            return 0.0, 0.0
        a = np.sqrt(R_eff_um * delta_um)
        F = (4.0 / 3.0) * E_s * a**3 / R_eff_um
        return F, a

    # No contact and no adhesion pull
    if delta_um <= 0.0:
        # Check adhesive pull at zero overlap
        a = (6.0 * np.pi * W_s * R_eff_um**2 / E_s) ** (1.0 / 3.0)
        F = (4.0 / 3.0) * E_s * a**3 / R_eff_um - np.sqrt(
            8.0 * np.pi * W_s * E_s * a**3)
        if F >= 0.0:
            return 0.0, 0.0  # repulsive at zero gap — no contact
        # Adhesive pull — but only within a small range
        if delta_um < -0.1 * R_eff_um:
            return 0.0, 0.0

    # Initial guess for contact radius
    if delta_um > 0.0:
        a = np.sqrt(R_eff_um * delta_um)
    else:
        a = (6.0 * np.pi * W_s * R_eff_um**2 / E_s) ** (1.0 / 3.0)

    # Newton-Raphson: solve δ = a²/R* − √(2πW a / E*) for a
    for _ in range(30):
        if a < 1e-12:
            a = 1e-6
        sqrt_term = np.sqrt(2.0 * np.pi * W_s * a / E_s) if a > 0 else 0.0
        f = a**2 / R_eff_um - sqrt_term - delta_um
        if a > 1e-12:
            dfda = 2.0 * a / R_eff_um - np.sqrt(
                np.pi * W_s / (2.0 * E_s * a))
        else:
            dfda = 2.0 * a / R_eff_um
        if abs(dfda) < 1e-30:
            break
        a_new = a - f / dfda
        if a_new < 0.0:
            a_new = a * 0.5
        if abs(a_new - a) < 1e-8:
            a = a_new
            break
        a = a_new

    if a < 1e-10:
        return 0.0, 0.0

    # JKR force from contact radius
    F = (4.0 / 3.0) * E_s * a**3 / R_eff_um - np.sqrt(
        8.0 * np.pi * W_s * E_s * a**3)
    return F, a


@njit(cache=True)
def jkr_contact_radius(F_ext_nN, R_eff_um, E_star_Pa, W_Jm2):
    """
    JKR contact radius from applied external force.

    a³ = (R*/E*) [F + 3πWR* + √(6πWR*F + (3πWR*)²)]

    Units: F_ext [nN], R_eff [µm], E_star [Pa], W [J/m²]
    Returns: a [µm]
    """
    E_s = E_star_Pa * 1e-3
    W_s = W_Jm2
    term1 = 3.0 * np.pi * W_s * R_eff_um
    inner = 6.0 * np.pi * W_s * R_eff_um * F_ext_nN + term1**2
    if inner < 0.0:
        inner = 0.0
    a_cubed = (R_eff_um / E_s) * (F_ext_nN + term1 + np.sqrt(inner))
    if a_cubed < 0.0:
        return 0.0
    return a_cubed ** (1.0 / 3.0)


@njit(cache=True)
def jkr_pulloff_force(R_eff_um, W_Jm2):
    """Pull-off force: F_po = (3/2)πWR* [nN]."""
    return 1.5 * np.pi * W_Jm2 * R_eff_um


# ══════════════════════════════════════════════════════════════════════
# V3.2: directional radius of a superellipse / superellipsoid
# ══════════════════════════════════════════════════════════════════════
# The packer works on bounding spheres, which is why turning shapes on today
# makes packing WORSE: for an equal-volume AR-1.8 / n-3.5 fragment the bounding
# sphere claims 1.97x the granule's volume, so RSA and the settle jam at a true
# solid fraction near 0.33. What the settle needs is a cheap anisotropic overlap.
#
# The GELS implicit function is positively homogeneous of degree n2 for ANY
# (n1, n2) -- the inner ^n1 is immediately raised to n2/n1 -- so the distance
# from the centre to the surface along a body-frame direction u is closed form:
#
#     lambda(u) = F(u) ** (-1/n2)          (2D: F = |ux/a|^n + |uy/b|^n, degree n)
#
# The pair gap is then g = lambda_i(u) + lambda_j(-u) - d, and BOTH the force and
# the torque follow from its gradient. A purely radial force cannot nest blocky
# granules at all: the torque arm would be parallel to the force, so the torque
# would be identically zero and nothing could rotate to interlock.
#
# NOTE, and it matters: lambda is homogeneous of degree -1, so Euler gives
# grad(lambda).u = -lambda, NOT zero. The gap depends on u only through the unit
# direction, so the RADIAL part of the gradient must be projected out. Verified
# against finite differences: with the projection the analytic gradient matches
# d(gap)/dx to 2e-10; without it the error is 1.37, i.e. a factor ~2.5 and a
# force that is not the gradient of any energy.


@njit(cache=True)
def se2d_lambda_grad(ux, uy, a, b, n):
    """(lambda, dlambda/dux, dlambda/duy) for a superellipse, body frame (V3.2)."""
    qa = abs(ux / a) ** n
    qb = abs(uy / b) ** n
    F = qa + qb
    if F < 1e-300:
        return 0.0, 0.0, 0.0
    lam = F ** (-1.0 / n)
    c = -(F ** (-1.0 / n - 1.0))
    gx = c * (abs(ux / a) ** (n - 1.0)) * (1.0 if ux >= 0.0 else -1.0) / a
    gy = c * (abs(uy / b) ** (n - 1.0)) * (1.0 if uy >= 0.0 else -1.0) / b
    return lam, gx, gy


@njit(cache=True)
def se3d_lambda_grad(ux, uy, uz, a, b, c, n1, n2):
    """(lambda, grad) for a superellipsoid, body frame (V3.2). Any (n1, n2)."""
    S = abs(ux / a) ** n1 + abs(uy / b) ** n1
    qz = abs(uz / c) ** n2
    Spow = S ** (n2 / n1) if S > 1e-300 else 0.0
    F = Spow + qz
    if F < 1e-300:
        return 0.0, 0.0, 0.0, 0.0
    lam = F ** (-1.0 / n2)
    pre = -(F ** (-1.0 / n2 - 1.0))
    Sfac = (S ** (n2 / n1 - 1.0)) if S > 1e-300 else 0.0
    gx = pre * Sfac * (abs(ux / a) ** (n1 - 1.0)) * (1.0 if ux >= 0.0 else -1.0) / a
    gy = pre * Sfac * (abs(uy / b) ** (n1 - 1.0)) * (1.0 if uy >= 0.0 else -1.0) / b
    gz = pre * (abs(uz / c) ** (n2 - 1.0)) * (1.0 if uz >= 0.0 else -1.0) / c
    return lam, gx, gy, gz


# ══════════════════════════════════════════════════════════════════════
# V3.4: support function of a superellipse / superellipsoid
# ══════════════════════════════════════════════════════════════════════
# The support function h(n) = max_{x in K} x.n is the object the whole
# minimum-translation-distance formulation is built on, and for the GELS
# two-exponent superellipsoid it is CLOSED FORM -- a nested dual norm:
#
#     h(n) = || ( ||(a n_x, b n_y)||_q1 , c n_z ) ||_q2
#     q1 = n1/(n1-1),   q2 = n2/(n2-1)
#
# Derivation (it is worth recording, because three other quantities fall out of
# the same intermediates). At the surface point p whose outward normal is
# parallel to n, the parametric point and the UNNORMALISED normal N satisfy
#
#     p.N = cos^2(eta) cos^2(omega) + cos^2(eta) sin^2(omega) + sin^2(eta) = 1
#
# identically -- every semi-axis cancels. So h(n_hat) = 1/|N|. Writing N = t n_hat
# and substituting the parametrisation turns cos^2(omega)+sin^2(omega) = 1 into
# A^q1 + B^q1 = cos(eta)^(beta q1) with A = a n_x t, B = b n_y t, beta = 2-2/n2,
# and then cos^2(eta)+sin^2(eta) = 1 into P^q2 + C^q2 = 1. Solving for t gives the
# norm above. Verified against a 600x1200 brute-force surface maximisation:
# 4.9e-6 relative (mesh-limited) for random n1 != n2, 3.6e-15 at the sphere anchor.
#
# What comes free from the same intermediates:
#   grad h(n) is the support POINT itself (envelope theorem), so the contact
#     point costs nothing extra;
#   (eta, omega) invert in closed form, keeping the new solver interoperable
#     with every existing parametric consumer;
#   the reverse Gauss map x(n_hat) = grad h(n_hat) makes the principal radii the
#     eigenvalues of grad^2 h, so R_eff becomes exact instead of a 1e-2 finite
#     difference (see support_R_eff_3d).
#
# NUMERICS, and it is not optional: never evaluate |a n_x|**q directly. At
# a ~ 40 um and q = 60 that is ~1e96 and the outer level overflows to inf. Every
# norm here is MAX-FACTORED, so each base sits in [0, 1] and the gradient weights
# come out in [0, 1] for free.
#
# Note also that _sgnpow's +1e-30 bias is deliberately NOT replicated: it would
# break the p.N = 1 identity the derivation rests on.

Q_MAX = 60.0          # numerically the max-norm; the correct octahedral limit
Q_MIN = 1.0 + 1e-9


@njit(cache=True)
def _dual_exp(n):
    """Holder conjugate q = n/(n-1) of the blockiness exponent, clamped (V3.4).

    n = 2 -> 2 (the ellipsoid, self-dual). n -> inf -> 1 (the box, dual to the
    1-norm). n <= 1 is non-convex and has no support function, so it is mapped to
    the max-norm rather than to a negative exponent.
    """
    if n <= 1.0 + 1e-9:
        return Q_MAX
    q = n / (n - 1.0)
    if q > Q_MAX:
        return Q_MAX
    if q < Q_MIN:
        return Q_MIN
    return q


@njit(cache=True)
def _norm2_q(u, v, q):
    """Max-factored ||(u, v)||_q and the two weights (|u|/N)^q, (|v|/N)^q."""
    au = abs(u)
    av = abs(v)
    m = au if au > av else av
    if m < 1e-300:
        return 0.0, 0.0, 0.0
    tu = (au / m) ** q
    tv = (av / m) ** q
    s = tu + tv
    nrm = m * s ** (1.0 / q)
    return nrm, tu / s, tv / s


@njit(cache=True)
def se2d_support(nx, ny, a, b, n):
    """Support function of a superellipse in its BODY frame (V3.4).

    Returns ``(h, gx, gy)`` where ``h = max_{x in K} x.n`` and ``(gx, gy) = grad h``
    is the support point -- the point of the boundary whose outward normal is
    parallel to ``n``. ``n`` need not be a unit vector: ``h`` is positively
    homogeneous of degree 1 and ``grad h`` of degree 0, so the point is exact for
    any positive scaling.
    """
    q = _dual_exp(n)
    u = a * nx
    v = b * ny
    h, wu, wv = _norm2_q(u, v, q)
    if h < 1e-300:
        return 0.0, 0.0, 0.0
    # dh/dnx = a * sgn(nx) * (|a nx| / h)^(q-1); the weight w = (|u|/h)^q, so
    # (|u|/h)^(q-1) = w * h / |u| -- but forming it that way divides by zero at a
    # flat face. Use the weight directly: dh/du = sgn(u) * w^(1-1/q).
    gu = 0.0 if wu <= 0.0 else wu ** (1.0 - 1.0 / q)
    gv = 0.0 if wv <= 0.0 else wv ** (1.0 - 1.0 / q)
    if u < 0.0:
        gu = -gu
    if v < 0.0:
        gv = -gv
    return h, a * gu, b * gv


@njit(cache=True)
def se3d_support(nx, ny, nz, a, b, c, n1, n2):
    """Support function of a superellipsoid in its BODY frame (V3.4).

    Returns ``(h, gx, gy, gz, eta, omega)``. ``(gx, gy, gz) = grad h`` is the
    support point; ``(eta, omega)`` are its parametric coordinates, so
    ``superellipsoid_point(eta, omega, a, b, c, n1, n2)`` reproduces the gradient
    (verified to 4e-16) and every existing parametric consumer keeps working.
    """
    q1 = _dual_exp(n1)
    q2 = _dual_exp(n2)
    u = a * nx
    v = b * ny
    w = c * nz
    P, wu, wv = _norm2_q(u, v, q1)
    h, wP, ww = _norm2_q(P, w, q2)
    if h < 1e-300:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # z component: dh/dw = sgn(w) * ww^(1-1/q2)
    gw = 0.0 if ww <= 0.0 else ww ** (1.0 - 1.0 / q2)
    if w < 0.0:
        gw = -gw

    # equatorial: dh/du = (dh/dP)(dP/du) = wP^(1-1/q2) * sgn(u) * wu^(1-1/q1)
    if wP <= 0.0 or P < 1e-300:
        # n lies on the polar axis: the support point is the pole (0, 0, +-c)
        return h, 0.0, 0.0, c * gw, (0.5 * np.pi if w >= 0.0 else -0.5 * np.pi), 0.0
    dP = wP ** (1.0 - 1.0 / q2)
    gu = 0.0 if wu <= 0.0 else wu ** (1.0 - 1.0 / q1)
    gv = 0.0 if wv <= 0.0 else wv ** (1.0 - 1.0 / q1)
    if u < 0.0:
        gu = -gu
    if v < 0.0:
        gv = -gv

    # parametric inversion: cos(eta)^(2/q2) = P/h, sin(eta)^(2/q2) = w/h
    ce = wP ** (0.5)          # (P/h)^(q2/2) == wP^(1/2), since wP = (P/h)^q2
    se = ww ** (0.5)
    if w < 0.0:
        se = -se
    eta = np.arctan2(se, ce)
    co = wu ** 0.5            # wu = (|u|/P)^q1, so (|u|/P)^(q1/2) = wu^(1/2)
    so = wv ** 0.5
    if u < 0.0:
        co = -co
    if v < 0.0:
        so = -so
    omega = np.arctan2(so, co)
    return h, a * dP * gu, b * dP * gv, c * gw, eta, omega


@njit(cache=True)
def support_R_eff_3d(nx, ny, nz, a, b, c, n1, n2, eps=1e-5):
    """Exact-to-1e-8 effective radius of curvature at the support point (V3.4).

    For a convex body the reverse Gauss map is ``x(n_hat) = grad h(n_hat)``, so the
    principal radii of curvature are the nonzero eigenvalues of the Hessian
    ``grad^2 h`` restricted to ``n_hat^perp`` (``grad^2 h . n_hat = 0``, because
    ``grad h`` is homogeneous of degree 0). Hence

        R_eff = sqrt(det grad^2 h |_{n_hat perp})

    which is exactly the ``sqrt(R1 R2)`` that Hertz wants. This is obtained by
    central-differencing **grad h** along two tangents, not by differencing the
    surface point twice, so the accuracy is ~1e-8 against the ~1e-2 of the fixed
    0.01-radian parametric scheme in ``superellipsoid_curvature_radii``.

    Returns ``R_eff`` in um, or 0.0 on a degenerate normal.
    """
    m = np.sqrt(nx*nx + ny*ny + nz*nz)
    if m < 1e-300:
        return 0.0
    ux = nx / m
    uy = ny / m
    uz = nz / m
    # any two unit tangents
    if abs(uz) < 0.9:
        tx, ty, tz = -uy, ux, 0.0
    else:
        tx, ty, tz = 0.0, -uz, uy
    tm = np.sqrt(tx*tx + ty*ty + tz*tz)
    tx /= tm; ty /= tm; tz /= tm
    sx = uy*tz - uz*ty
    sy = uz*tx - ux*tz
    sz = ux*ty - uy*tx

    # central differences of grad h along t and s
    _, gxp, gyp, gzp, _, _ = se3d_support(ux + eps*tx, uy + eps*ty, uz + eps*tz,
                                          a, b, c, n1, n2)
    _, gxm, gym, gzm, _, _ = se3d_support(ux - eps*tx, uy - eps*ty, uz - eps*tz,
                                          a, b, c, n1, n2)
    dt_x = (gxp - gxm) / (2.0 * eps)
    dt_y = (gyp - gym) / (2.0 * eps)
    dt_z = (gzp - gzm) / (2.0 * eps)

    _, gxp, gyp, gzp, _, _ = se3d_support(ux + eps*sx, uy + eps*sy, uz + eps*sz,
                                          a, b, c, n1, n2)
    _, gxm, gym, gzm, _, _ = se3d_support(ux - eps*sx, uy - eps*sy, uz - eps*sz,
                                          a, b, c, n1, n2)
    ds_x = (gxp - gxm) / (2.0 * eps)
    ds_y = (gyp - gym) / (2.0 * eps)
    ds_z = (gzp - gzm) / (2.0 * eps)

    # 2x2 Hessian in the (t, s) basis; symmetric up to FD error, so symmetrise
    Htt = dt_x*tx + dt_y*ty + dt_z*tz
    Hts = dt_x*sx + dt_y*sy + dt_z*sz
    Hst = ds_x*tx + ds_y*ty + ds_z*tz
    Hss = ds_x*sx + ds_y*sy + ds_z*sz
    off = 0.5 * (Hts + Hst)
    det = Htt * Hss - off * off
    if det <= 0.0:
        return 0.0
    return np.sqrt(det)


@njit(cache=True)
def support_R_eff_2d(nx, ny, a, b, n, eps=1e-5):
    """Radius of curvature of a superellipse at the support point (V3.4).

    The 1D analogue of :func:`support_R_eff_3d`: ``R = dh/dtheta`` of the support
    point along the single tangent, i.e. the lone nonzero eigenvalue of
    ``grad^2 h``. Returns 0.0 on a degenerate normal.
    """
    m = np.sqrt(nx*nx + ny*ny)
    if m < 1e-300:
        return 0.0
    ux = nx / m
    uy = ny / m
    tx, ty = -uy, ux
    _, gxp, gyp = se2d_support(ux + eps*tx, uy + eps*ty, a, b, n)
    _, gxm, gym = se2d_support(ux - eps*tx, uy - eps*ty, a, b, n)
    R = ((gxp - gxm) * tx + (gyp - gym) * ty) / (2.0 * eps)
    return R if R > 0.0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# V3.4: minimum-translation-distance contact from the support function
# ══════════════════════════════════════════════════════════════════════
# For two convex bodies the separation along a unit direction n is
#
#     sep(n) = (c2 - c1).n - h1(n) - h2(-n)
#
# and because the superellipsoid is centrally symmetric, h2(-n) = h2(n). The
# function is CONCAVE (linear minus two convex supports), so
#
#     n* = argmax sep,    delta = -sep(n*)  (> 0 in contact)
#
# is a single well-posed maximisation with no parallel-branch ambiguity, and
# sep(n) > 0 at ANY n is a certificate of separation. That certificate is what
# pays for the ascent: neighbour lists are generous, so most candidate pairs are
# rejected by one evaluation, where the common-normal solver runs 15
# unconditional Newton iterations on every candidate.
#
# Two identities make the iteration cheap. With x1 = grad h1(n) and
# x2 = grad h2(n) (the support POINTS, free by the envelope theorem),
#
#     grad sep = (c2 - c1) - x1 - x2 = p2 - p1     (witness-point difference)
#     grad sep . n = sep                            (Euler, since grad h . n = h)
#
# so the ascent direction costs nothing beyond the two support evaluations that
# sep already needed.
#
# WHY THIS REPLACES THE COMMON-NORMAL SOLVER. `find_contact_superellipsoids_3d`
# sets `delta = dp_mag`, the distance between two common-normal SURFACE POINTS,
# which is not a penetration depth, and reports `(p_j - p_i)/|.|` as the normal,
# which is not a surface normal. Measured at 2 % past first touch: it detects
# 25/200 = 12 % of true contacts and over-reports penetration by median 9.8x,
# p90 43x. Here, `sep` depends on c2 only through the linear term, so the
# envelope theorem gives d(delta)/d(c2) = -n* EXACTLY -- the penalty force is
# the gradient of an energy, which is the property the old solver never had.
#
# Truncating the ascent early is safe in a specific and useful way: sep(n) <=
# sep(n*) for every n, so a short ascent under-estimates sep and therefore
# OVER-estimates delta. The error is a slightly stiff contact, bounded and
# continuous -- never a missed one.

# ITERATION BUDGET, measured rather than guessed. Against a Nelder-Mead-polished
# reference over 30 random tumbled pairs per shape class, the MEDIAN relative
# error in delta falls 2e-9 -> 4e-12 -> 2e-14 as the budget goes 24 -> 32 -> 64,
# while the p90 and the MAX are FLAT across that whole range:
#
#   budget   n=4 med / p90 / max        n=10 med / p90 / max     support evals
#     24     3.0e-07 1.5e-04 1.1e-03    9.6e-05 7.4e-03 5.1e-02       152
#     32     1.5e-08 1.1e-04 6.1e-04    2.2e-05 4.9e-03 5.1e-02       200
#     64     2.5e-13 5.9e-05 3.9e-04    1.2e-06 4.9e-03 5.1e-02       388
#
# The tail is SEEDING-limited, not iteration-limited. `sep` is concave on R^3,
# but the unit sphere is not a convex constraint set, and in penetration `sep` is
# negative everywhere on it -- so the sphere-restricted problem genuinely admits
# local maxima, and a minority of tumbled near-polyhedral pairs ascend into one.
# Doubling the budget cannot fix that; only more seeds can. 32 is therefore the
# knee: it halves the cost of 64 and gives up nothing that a delta^(3/2) force
# law can feel. The residual tail at n >= 8 is documented, not hidden -- the
# realistic GELS range is n in [2, 4] (presets run 2.2-3.5), where the p90 is
# 1e-4 relative, i.e. a 1.5e-4 relative force error.
MTD_MAX_ITERS = 32
MTD_STEP0 = 0.3           # initial angular step (rad-ish; |n| is renormalised)
MTD_STEP_GROW = 1.3
MTD_STEP_SHRINK = 0.5
MTD_STEP_MAX = 1.0
MTD_STEP_FLOOR = 1e-7     # give up when the bracketing step is this small
MTD_GPERP_RTOL = 1e-7     # break when |grad sep perp n| < rtol * (r_i + r_j)
MTD_MULTISEED_N = 3.0     # below this blockiness one seed is provably enough


@njit(cache=True)
def _qrot_s(w, x, y, z, vx, vy, vz):
    """Scalar quaternion rotation, body -> world. Allocation-free twin of
    ``quat_rotate``; agrees with it to rounding (asserted in the tests)."""
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


@njit(cache=True)
def _qrot_inv_s(w, x, y, z, vx, vy, vz):
    """Scalar quaternion rotation, world -> body."""
    return _qrot_s(w, -x, -y, -z, vx, vy, vz)


# ── 3D ───────────────────────────────────────────────────────────────

@njit(cache=True)
def _mtd_sep_3d(nx, ny, nz, dcx, dcy, dcz,
                qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
                qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j):
    """``sep(n)`` and the two support-point offsets, world frame."""
    bx, by, bz = _qrot_inv_s(qiw, qix, qiy, qiz, nx, ny, nz)
    hi, gx, gy, gz, _, _ = se3d_support(bx, by, bz, ai, bi, ci, n1i, n2i)
    x1x, x1y, x1z = _qrot_s(qiw, qix, qiy, qiz, gx, gy, gz)
    bx, by, bz = _qrot_inv_s(qjw, qjx, qjy, qjz, nx, ny, nz)
    hj, gx, gy, gz, _, _ = se3d_support(bx, by, bz, aj, bj, cj, n1j, n2j)
    x2x, x2y, x2z = _qrot_s(qjw, qjx, qjy, qjz, gx, gy, gz)
    sep = dcx * nx + dcy * ny + dcz * nz - hi - hj
    return sep, x1x, x1y, x1z, x2x, x2y, x2z


@njit(cache=True)
def _mtd_ascend_3d(n0x, n0y, n0z, dcx, dcy, dcz,
                   qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
                   qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j, tol):
    """Projected-gradient ascent of the concave ``sep`` on the unit sphere.

    The step is taken along the UNIT tangent gradient, so it is an angular step
    decoupled from the gradient magnitude; it grows on success and halves on
    failure, which is what keeps it robust across the ~1e3 range of curvature
    scales a blocky pair spans.
    """
    m = np.sqrt(n0x * n0x + n0y * n0y + n0z * n0z)
    if m < 1e-12:
        return -1e30, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e30
    nx = n0x / m
    ny = n0y / m
    nz = n0z / m
    sep, x1x, x1y, x1z, x2x, x2y, x2z = _mtd_sep_3d(
        nx, ny, nz, dcx, dcy, dcz,
        qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
        qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j)
    step = MTD_STEP0
    gpm = 0.0
    for _ in range(MTD_MAX_ITERS):
        gx = dcx - x1x - x2x
        gy = dcy - x1y - x2y
        gz = dcz - x1z - x2z
        gn = gx * nx + gy * ny + gz * nz
        px = gx - gn * nx
        py = gy - gn * ny
        pz = gz - gn * nz
        gpm = np.sqrt(px * px + py * py + pz * pz)
        if gpm < tol or step < MTD_STEP_FLOOR:
            break
        px /= gpm
        py /= gpm
        pz /= gpm
        tx = nx + step * px
        ty = ny + step * py
        tz = nz + step * pz
        tm = np.sqrt(tx * tx + ty * ty + tz * tz)
        tx /= tm
        ty /= tm
        tz /= tm
        s2, y1x, y1y, y1z, y2x, y2y, y2z = _mtd_sep_3d(
            tx, ty, tz, dcx, dcy, dcz,
            qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
            qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j)
        if s2 >= sep:
            nx = tx; ny = ty; nz = tz
            sep = s2
            x1x = y1x; x1y = y1y; x1z = y1z
            x2x = y2x; x2y = y2y; x2z = y2z
            step = step * MTD_STEP_GROW
            if step > MTD_STEP_MAX:
                step = MTD_STEP_MAX
        else:
            step *= MTD_STEP_SHRINK
    else:
        gx = dcx - x1x - x2x
        gy = dcy - x1y - x2y
        gz = dcz - x1z - x2z
        gn = gx * nx + gy * ny + gz * nz
        px = gx - gn * nx
        py = gy - gn * ny
        pz = gz - gn * nz
        gpm = np.sqrt(px * px + py * py + pz * pz)
    return sep, nx, ny, nz, x1x, x1y, x1z, x2x, x2y, x2z, gpm


@njit(cache=True)
def _best_axis_3d(qw, qx, qy, qz, tx, ty, tz):
    """The body axis (a face normal for a blocky shape) best aligned with t."""
    bx, by, bz = _qrot_inv_s(qw, qx, qy, qz, tx, ty, tz)
    ax = abs(bx); ay = abs(by); az = abs(bz)
    if ax >= ay and ax >= az:
        s = 1.0 if bx >= 0.0 else -1.0
        return _qrot_s(qw, qx, qy, qz, s, 0.0, 0.0)
    if ay >= az:
        s = 1.0 if by >= 0.0 else -1.0
        return _qrot_s(qw, qx, qy, qz, 0.0, s, 0.0)
    s = 1.0 if bz >= 0.0 else -1.0
    return _qrot_s(qw, qx, qy, qz, 0.0, 0.0, s)


@njit(cache=True)
def se3d_mtd_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                  xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """Support-function MTD contact of two superellipsoids (V3.4).

    Returns ``(hit, delta, nx, ny, nz, cx, cy, cz, R_eff, residual)`` -- the
    first nine fields are exactly the tuple ``se3d_contact_k`` returns, so call
    sites switch solvers with one ``if``. ``residual`` is ``|grad sep perp n|``
    in um, the ascent's own convergence certificate; it is dropped by the
    status-tuple wrapper and consumed only by the tests.

    ``delta`` is a true minimum translation distance and ``n`` a true contact
    normal, so ``F = k delta^(3/2) n`` is conservative.
    """
    dcx = xj - xi
    dcy = yj - yi
    dcz = zj - zi
    d = np.sqrt(dcx * dcx + dcy * dcy + dcz * dcz)
    if d < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n0x = dcx / d
    n0y = dcy / d
    n0z = dcz / d

    qiw = qi[0]; qix = qi[1]; qiy = qi[2]; qiz = qi[3]
    qjw = qj[0]; qjx = qj[1]; qjy = qj[2]; qjz = qj[3]

    # Separation certificate: sep(n) > 0 at ANY n proves the bodies are apart,
    # because sep(n*) >= sep(n). One evaluation rejects most candidate pairs.
    sep0, x1x, x1y, x1z, x2x, x2y, x2z = _mtd_sep_3d(
        n0x, n0y, n0z, dcx, dcy, dcz,
        qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
        qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j)
    if sep0 > 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    scale = max(ai, bi, ci) + max(aj, bj, cj)
    tol = MTD_GPERP_RTOL * scale

    sep, nx, ny, nz, x1x, x1y, x1z, x2x, x2y, x2z, res = _mtd_ascend_3d(
        n0x, n0y, n0z, dcx, dcy, dcz,
        qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
        qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j, tol)

    # A flat face makes the centre-line ascent stall on a ridge, so restart from
    # each body's best-aligned face normal and keep the highest sep. Smooth
    # shapes are strictly convex with a unique maximum and do not need it.
    if (n1i > MTD_MULTISEED_N or n2i > MTD_MULTISEED_N or
            n1j > MTD_MULTISEED_N or n2j > MTD_MULTISEED_N):
        for k in range(2):
            if k == 0:
                sx, sy, sz = _best_axis_3d(qiw, qix, qiy, qiz, n0x, n0y, n0z)
            else:
                sx, sy, sz = _best_axis_3d(qjw, qjx, qjy, qjz, -n0x, -n0y, -n0z)
            s2, m2x, m2y, m2z, a1x, a1y, a1z, a2x, a2y, a2z, r2 = _mtd_ascend_3d(
                sx, sy, sz, dcx, dcy, dcz,
                qiw, qix, qiy, qiz, ai, bi, ci, n1i, n2i,
                qjw, qjx, qjy, qjz, aj, bj, cj, n1j, n2j, tol)
            if s2 > sep:
                sep = s2
                nx = m2x; ny = m2y; nz = m2z
                x1x = a1x; x1y = a1y; x1z = a1z
                x2x = a2x; x2y = a2y; x2z = a2z
                res = r2

    if sep >= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, res
    delta = -sep

    # Witness points: body i's extreme along +n, body j's along -n (central
    # symmetry makes grad h_j(-n) = -grad h_j(n)). Their midpoint is the contact
    # point, and its offset from each centre is the torque lever arm.
    p1x = xi + x1x; p1y = yi + x1y; p1z = zi + x1z
    p2x = xj - x2x; p2y = yj - x2y; p2z = zj - x2z
    cx = 0.5 * (p1x + p2x)
    cy = 0.5 * (p1y + p2y)
    cz = 0.5 * (p1z + p2z)

    bx, by, bz = _qrot_inv_s(qiw, qix, qiy, qiz, nx, ny, nz)
    Ri = support_R_eff_3d(bx, by, bz, ai, bi, ci, n1i, n2i)
    bx, by, bz = _qrot_inv_s(qjw, qjx, qjy, qjz, nx, ny, nz)
    Rj = support_R_eff_3d(bx, by, bz, aj, bj, cj, n1j, n2j)
    if Ri + Rj > 0.0:
        R_eff = Ri * Rj / (Ri + Rj)
    else:
        R_eff = 1.0
    return True, delta, nx, ny, nz, cx, cy, cz, R_eff, res


# ── 2D ───────────────────────────────────────────────────────────────

@njit(cache=True)
def _mtd_sep_2d(nx, ny, dcx, dcy, cti, sti, ai, bi, ni, ctj, stj, aj, bj, nj):
    """``sep(n)`` and the two support-point offsets, world frame (2D)."""
    bxi = cti * nx + sti * ny
    byi = -sti * nx + cti * ny
    hi, gx, gy = se2d_support(bxi, byi, ai, bi, ni)
    x1x = cti * gx - sti * gy
    x1y = sti * gx + cti * gy
    bxj = ctj * nx + stj * ny
    byj = -stj * nx + ctj * ny
    hj, gx, gy = se2d_support(bxj, byj, aj, bj, nj)
    x2x = ctj * gx - stj * gy
    x2y = stj * gx + ctj * gy
    return dcx * nx + dcy * ny - hi - hj, x1x, x1y, x2x, x2y


@njit(cache=True)
def _mtd_ascend_2d(n0x, n0y, dcx, dcy, cti, sti, ai, bi, ni,
                   ctj, stj, aj, bj, nj, tol):
    """Ascent of the concave ``sep`` on the unit circle.

    In 2D the tangent is one-dimensional, so the "unit tangent gradient" is just
    a sign -- the step is a signed rotation and the scheme reduces to a bisection
    that cannot wander.
    """
    m = np.sqrt(n0x * n0x + n0y * n0y)
    if m < 1e-12:
        return -1e30, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e30
    nx = n0x / m
    ny = n0y / m
    sep, x1x, x1y, x2x, x2y = _mtd_sep_2d(nx, ny, dcx, dcy, cti, sti, ai, bi, ni,
                                          ctj, stj, aj, bj, nj)
    step = MTD_STEP0
    gpm = 0.0
    for _ in range(MTD_MAX_ITERS):
        gx = dcx - x1x - x2x
        gy = dcy - x1y - x2y
        gn = gx * nx + gy * ny
        px = gx - gn * nx
        py = gy - gn * ny
        gpm = np.sqrt(px * px + py * py)
        if gpm < tol or step < MTD_STEP_FLOOR:
            break
        px /= gpm
        py /= gpm
        tx = nx + step * px
        ty = ny + step * py
        tm = np.sqrt(tx * tx + ty * ty)
        tx /= tm
        ty /= tm
        s2, y1x, y1y, y2x, y2y = _mtd_sep_2d(tx, ty, dcx, dcy, cti, sti, ai, bi, ni,
                                             ctj, stj, aj, bj, nj)
        if s2 >= sep:
            nx = tx; ny = ty
            sep = s2
            x1x = y1x; x1y = y1y; x2x = y2x; x2y = y2y
            step = step * MTD_STEP_GROW
            if step > MTD_STEP_MAX:
                step = MTD_STEP_MAX
        else:
            step *= MTD_STEP_SHRINK
    else:
        gx = dcx - x1x - x2x
        gy = dcy - x1y - x2y
        gn = gx * nx + gy * ny
        px = gx - gn * nx
        py = gy - gn * ny
        gpm = np.sqrt(px * px + py * py)
    return sep, nx, ny, x1x, x1y, x2x, x2y, gpm


@njit(cache=True)
def se2d_mtd_core(xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj):
    """Support-function MTD contact of two superellipses (V3.4).

    Returns ``(hit, delta, nx, ny, cx, cy, R_loc_i, R_loc_j, residual)`` -- the
    first eight fields are exactly the tuple ``se2d_contact_k`` returns.
    """
    dcx = xj - xi
    dcy = yj - yi
    d = np.sqrt(dcx * dcx + dcy * dcy)
    if d < 1e-12:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    n0x = dcx / d
    n0y = dcy / d
    cti = np.cos(thetai); sti = np.sin(thetai)
    ctj = np.cos(thetaj); stj = np.sin(thetaj)

    sep0, x1x, x1y, x2x, x2y = _mtd_sep_2d(n0x, n0y, dcx, dcy, cti, sti, ai, bi, ni,
                                           ctj, stj, aj, bj, nj)
    if sep0 > 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    scale = max(ai, bi) + max(aj, bj)
    tol = MTD_GPERP_RTOL * scale
    sep, nx, ny, x1x, x1y, x2x, x2y, res = _mtd_ascend_2d(
        n0x, n0y, dcx, dcy, cti, sti, ai, bi, ni, ctj, stj, aj, bj, nj, tol)

    if ni > MTD_MULTISEED_N or nj > MTD_MULTISEED_N:
        for k in range(4):
            if k == 0:
                sx, sy = cti, sti
            elif k == 1:
                sx, sy = -sti, cti
            elif k == 2:
                sx, sy = ctj, stj
            else:
                sx, sy = -stj, ctj
            if sx * n0x + sy * n0y < 0.0:
                sx = -sx; sy = -sy
            s2, m2x, m2y, a1x, a1y, a2x, a2y, r2 = _mtd_ascend_2d(
                sx, sy, dcx, dcy, cti, sti, ai, bi, ni, ctj, stj, aj, bj, nj, tol)
            if s2 > sep:
                sep = s2
                nx = m2x; ny = m2y
                x1x = a1x; x1y = a1y; x2x = a2x; x2y = a2y
                res = r2

    if sep >= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, res
    delta = -sep

    p1x = xi + x1x; p1y = yi + x1y
    p2x = xj - x2x; p2y = yj - x2y
    cx = 0.5 * (p1x + p2x)
    cy = 0.5 * (p1y + p2y)

    bxi = cti * nx + sti * ny
    byi = -sti * nx + cti * ny
    R_loc_i = support_R_eff_2d(bxi, byi, ai, bi, ni)
    bxj = ctj * nx + stj * ny
    byj = -stj * nx + ctj * ny
    R_loc_j = support_R_eff_2d(bxj, byj, aj, bj, nj)
    return True, delta, nx, ny, cx, cy, R_loc_i, R_loc_j, res


# THE SHAPED CONTACT SOLVER. These two names are what the engine, the oracle in
# `kernels/reference.py` and `viz/stress.py` all call, so pointing them at the
# MTD core switches every path at once -- there is no `contact.solver` flag, no
# per-call-site branch, and therefore no way for one path to be running
# different physics from another. The common-normal bodies are retained under
# `*_cn` names as evidence for the defect tests and are called by nothing.
#
# Returning Optional from Python (rather than a status tuple from @njit) is
# free here: every caller of these two is pure Python. Compiled callers use
# `kernels/geometry2d.se2d_contact_k` / `geometry3d.se3d_contact_k`, which wrap
# the same core.

def find_contact_superellipses(xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj):
    """Contact of two superellipses by support-function MTD (V3.4).

    ``None`` when apart, else ``(True, delta, nx, ny, cx, cy, R_loc_i, R_loc_j)``
    -- the same contract the common-normal solver had, with `delta` now a true
    minimum translation distance, `n` a true contact normal pointing from i to
    j, and the local radii exact rather than a 0.01-radian finite difference.
    """
    out = se2d_mtd_core(xi, yi, ai, bi, ni, thetai, xj, yj, aj, bj, nj, thetaj)
    if not out[0]:
        return None
    return out[:8]


def find_contact_superellipsoids_3d(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                                    xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """Contact of two superellipsoids by support-function MTD (V3.4).

    ``None`` when apart, else ``(True, delta, nx, ny, nz, cx, cy, cz, R_eff)``.
    """
    out = se3d_mtd_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                        xj, yj, zj, aj, bj, cj, n1j, n2j, qj)
    if not out[0]:
        return None
    return out[:9]


# Back-compat aliases for anything that imported the V3.4-Phase-3.2 names.
mtd_contact_2d = find_contact_superellipses
mtd_contact_3d = find_contact_superellipsoids_3d


# ══════════════════════════════════════════════════════════════════════
# V3.4: wall contact from the support function
# ══════════════════════════════════════════════════════════════════════
# A wall is a half-space, so it is the EASY case of the same machinery: for a
# plane through q with unit normal w pointing into the container,
#
#     penetration = h_i(-w) - (c_i - q).w        contact point = c_i + grad h_i(-w)
#
# exactly, in one support evaluation. No iteration -- the maximising direction is
# handed to us by the wall.
#
# This replaces SIX brute-force surface samplers (a 64-point ring in 2D, four
# 20 x 20 (eta, omega) grids in 3D). Those were wrong in two ways at once. The
# grid is coarse: 20 x 20 on a superellipsoid puts ~9 degrees between samples, so
# the deepest point is missed by O(R theta^2/2) ~ 0.15 um at R = 40 um, which is
# the same size as the overlaps being resolved. And the reported curvature was
# `ri_bound * 0.5` -- a hardcoded constant with no geometric content, feeding
# `F ~ sqrt(R_local)`.
#
# Because the superellipsoid is centrally symmetric, h(-w) = h(w) and
# grad h(-w) = -grad h(w), so one evaluation along +w serves both.

@njit(cache=True)
def _support_world_2d(nx, ny, a, b, n, theta):
    """``(h, gx_world, gy_world, nx_body, ny_body)`` for a world direction."""
    ct = np.cos(theta)
    st = np.sin(theta)
    bx = ct * nx + st * ny
    by = -st * nx + ct * ny
    h, gx, gy = se2d_support(bx, by, a, b, n)
    return h, ct * gx - st * gy, st * gx + ct * gy, bx, by


@njit(cache=True)
def _support_world_3d(nx, ny, nz, a, b, c, n1, n2, qw, qx, qy, qz):
    """``(h, grad_world (3), n_body (3))`` for a world direction."""
    bx, by, bz = _qrot_inv_s(qw, qx, qy, qz, nx, ny, nz)
    h, gx, gy, gz, _, _ = se3d_support(bx, by, bz, a, b, c, n1, n2)
    wx, wy, wz = _qrot_s(qw, qx, qy, qz, gx, gy, gz)
    return h, wx, wy, wz, bx, by, bz


@njit(cache=True)
def se2d_wall_core(xi, yi, ai, bi, ni, thetai, wall_pos, wall_axis, wall_sign):
    """Superellipse vs an axis-aligned wall (V3.4).

    Returns ``(hit, penetration, R_local, cx, cy)``. ``wall_axis`` 0 = x, 1 = y;
    ``wall_sign`` +1 when the granule belongs on the high side of ``wall_pos``.
    The contact point is the support point itself, so the lever arm for wall
    torque costs nothing.
    """
    wx = 1.0 if wall_axis == 0 else 0.0
    wy = 0.0 if wall_axis == 0 else 1.0
    h, gx, gy, bnx, bny = _support_world_2d(wx, wy, ai, bi, ni, thetai)
    centre = xi if wall_axis == 0 else yi
    pen = h - wall_sign * (centre - wall_pos)
    if pen <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0
    # support point along -sign*w, i.e. the surface point nearest the wall
    cx = xi - wall_sign * gx
    cy = yi - wall_sign * gy
    R_local = support_R_eff_2d(bnx, bny, ai, bi, ni)
    return True, pen, R_local, cx, cy


@njit(cache=True)
def se3d_wall_plane_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                         px, py, pz, nx, ny, nz):
    """Superellipsoid vs an arbitrary plane (V3.4).

    The plane passes through ``(px, py, pz)`` with unit normal ``(nx, ny, nz)``
    pointing INTO the container. Returns ``(hit, penetration, R_local, cx, cy, cz)``.
    An axis wall is the special case ``n = +-e_axis``; the cylinder side wall
    uses the tangent plane at the granule's azimuth.
    """
    h, gx, gy, gz, bnx, bny, bnz = _support_world_3d(
        nx, ny, nz, ai, bi, ci, n1i, n2i, qi[0], qi[1], qi[2], qi[3])
    d = (xi - px) * nx + (yi - py) * ny + (zi - pz) * nz
    pen = h - d
    if pen <= 0.0:
        return False, 0.0, 0.0, 0.0, 0.0, 0.0
    R_local = support_R_eff_3d(bnx, bny, bnz, ai, bi, ci, n1i, n2i)
    return True, pen, R_local, xi - gx, yi - gy, zi - gz


@njit(cache=True)
def se3d_wall_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, wall_pos, wall_axis, wall_sign):
    """Superellipsoid vs an axis-aligned wall (V3.4); see :func:`se3d_wall_plane_core`."""
    nx = wall_sign if wall_axis == 0 else 0.0
    ny = wall_sign if wall_axis == 1 else 0.0
    nz = wall_sign if wall_axis == 2 else 0.0
    px = wall_pos if wall_axis == 0 else xi
    py = wall_pos if wall_axis == 1 else yi
    pz = wall_pos if wall_axis == 2 else zi
    return se3d_wall_plane_core(xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
                                px, py, pz, float(nx), float(ny), float(nz))


def find_contact_superellipse_wall(xi, yi, ai, bi, ni, thetai,
                                   wall_pos, wall_axis, wall_sign):
    """Superellipse-wall contact: ``(penetration, R_local)`` or ``None`` (V3.4)."""
    hit, pen, R_local, _cx, _cy = se2d_wall_core(
        xi, yi, ai, bi, ni, thetai, wall_pos, wall_axis, wall_sign)
    if not hit:
        return None
    return pen, R_local





def shape_bound_radius(a, b, c=None, n1=2.0, n2=None):
    """True circumscribed radius of a superellipse / superellipsoid (V3.2).

    ``r_bound = max(a, b, c)`` is only correct at n = 2. For n > 2 the farthest
    surface point is toward a corner, so the current bound does not bound the
    body at all: an equal-volume AR-1.0 / n-3.5 granule has max(a,b,c) = 0.880
    against a true 1.114, a 27 % under-estimate, while for AR 1.8 it
    over-estimates the SHORT axes. Both are wrong and in opposite directions.

    For n1 == n2 == n > 2 the interior stationary point of
    ``r^2 = sum s_k^2 u_k^(2/n)`` on the simplex is the maximum, giving

        r_bound = s_max * (sum_k (s_k/s_max)^q)^(1/2 - 1/n),   q = 2n/(n-2)

    (the ratio form avoids overflow as n -> 2+). For n <= 2, and for n < 2 where
    the objective is convex and the maximum sits at a vertex, it is s_max. For
    n1 != n2 there is no closed form, so a golden section over the xy/z split is
    used -- it runs once per granule per inflation step in numpy, never in a
    kernel, so the cost is irrelevant.

    Accepts scalars or arrays; returns the same shape.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1 = np.asarray(n1, dtype=float)
    if c is None:
        axes = [a, b]
    else:
        axes = [a, b, np.asarray(c, dtype=float)]
    n2a = n1 if n2 is None else np.asarray(n2, dtype=float)

    def _iso(s_list, n):
        s = np.stack(np.broadcast_arrays(*s_list))
        s_max = np.max(s, axis=0)
        out = np.array(s_max, dtype=float, copy=True)
        sharp = n > 2.0 + 1e-6
        if np.any(sharp):
            q = np.where(sharp, 2.0 * n / np.maximum(n - 2.0, 1e-12), 1.0)
            ratio = np.divide(s, np.maximum(s_max, 1e-300))
            tot = np.sum(ratio ** q, axis=0)
            out = np.where(sharp, s_max * tot ** (0.5 - 1.0 / n), out)
        return out

    if c is None or np.all(np.abs(n1 - n2a) < 1e-9):
        return _iso(axes, n1 if c is None else n1)
    # n1 != n2 (3D): golden section over the share of the implicit budget in xy
    r_xy = _iso([axes[0], axes[1]], n1)
    cc = axes[2]
    lo, hi = np.zeros_like(r_xy), np.ones_like(r_xy)
    gr = (np.sqrt(5.0) - 1.0) / 2.0

    def _r2(sigma):
        sig = np.clip(sigma, 1e-12, 1.0 - 1e-12)
        return (sig ** (2.0 / n1)) * r_xy ** 2 + (cc ** 2) * (1.0 - sig ** (n2a / n1)) ** (2.0 / n2a)

    for _ in range(40):
        x1 = hi - gr * (hi - lo)
        x2 = lo + gr * (hi - lo)
        take = _r2(x1) > _r2(x2)
        hi = np.where(take, x2, hi)
        lo = np.where(take, lo, x1)
    return np.sqrt(np.maximum(_r2(0.5 * (lo + hi)), np.maximum(r_xy, cc) ** 2))


def lambda_world(a, b, c, n1, n2, theta, quat, ux, uy, uz, is_3d):
    """Directional radius along WORLD directions, vectorised (V3.2).

    Every argument is an array of the same length: one entry per granule (walls,
    bed surface) or per pair (overlap resolution). The direction is rotated into
    each granule's body frame and the closed-form lambda evaluated there.

    numpy twin of ``se2d_lambda_grad`` / ``se3d_lambda_grad``; the compiled paths
    call the njit leaves directly. Only the radius is returned -- the gradient is
    needed for forces, and the callers here (the overlap projection, the wall
    clamp, the bed surface) act along a fixed direction.
    """
    ux = np.asarray(ux, dtype=float)
    uy = np.asarray(uy, dtype=float)
    if not is_3d:
        ct, st = np.cos(theta), np.sin(theta)
        bx = ux * ct + uy * st
        by = -ux * st + uy * ct
        F = np.abs(bx / a) ** n1 + np.abs(by / b) ** n1
        return np.where(F > 1e-300, F ** (-1.0 / n1), 0.0)
    uz = np.asarray(uz, dtype=float)
    w, qx, qy, qz = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    tx = 2.0 * (qy * uz - qz * uy)
    ty = 2.0 * (qz * ux - qx * uz)
    tz = 2.0 * (qx * uy - qy * ux)
    bx = ux - w * tx + (qy * tz - qz * ty)          # b = R^T u
    by = uy - w * ty + (qz * tx - qx * tz)
    bz = uz - w * tz + (qx * ty - qy * tx)
    S = np.abs(bx / a) ** n1 + np.abs(by / b) ** n1
    F = np.where(S > 1e-300, S ** (n2 / n1), 0.0) + np.abs(bz / c) ** n2
    return np.where(F > 1e-300, F ** (-1.0 / n2), 0.0)


def shape_dynamics_on(p, gs=None):
    """Is the shape-aware dynamics path active? (V3.2)"""
    if not getattr(p, 'contact_shape_dynamics', False):
        return False
    if gs is not None and getattr(gs, 'is_circle', False):
        return False
    return bool(getattr(p, 'shape_enabled', False))


def granule_bound_radius(a, b, c, n1, n2, is_3d, shape_aware):
    """Bounding radius for the current semi-axes, V3.1 or V3.2 rule (V3.2).

    ``shape_aware`` False reproduces ``max(a, b, c)`` exactly, bit for bit, which
    is what every legacy path needs. True uses :func:`shape_bound_radius`, which
    actually bounds a blocky granule. The two MUST NOT be mixed within a run: the
    correct bound is LARGER, so switching it on alone lowers the bounding-sphere
    jamming ceiling and makes packing worse -- it only pays off together with the
    directional overlap in the settle, which is why both sit behind one flag.
    """
    if not shape_aware:
        if is_3d:
            return np.maximum(np.maximum(a, b), c)
        return np.maximum(a, b)
    if is_3d:
        return shape_bound_radius(a, b, c, n1, n2)
    return shape_bound_radius(a, b, None, n1, None)


def elastic_overlap_frac(p, lo=0.05, hi=0.30):
    """Size ``max_overlap_frac`` from the contact law rather than by hand (V3.2).

    The overlap projection in ``_resolve_overlaps`` permits a standing overlap of
    ``max_overlap_frac * r`` and pushes away only the excess, so that number is a
    hard ceiling on how far a bed can compact: the volumetric multiplier is
    ``(2 / (2 - frac))**3`` -- 1.046 at 0.03, 1.263 at 0.15.

    It is meant to be a NUMERICAL rail against explicit-step overshoot, not the
    thing that decides the physics. It becomes the latter whenever it falls below
    the overlap at which the contact carries the cell force,

        F_JKR(delta_eq) = F_ref   =>   delta_eq = (3 F_ref / (4 E_s sqrt(R*)))**(2/3)

    with ``E_s = E / (2 (1 - nu^2))`` in nN/um^2 and ``R* = r/2`` for a like pair.
    V3.1's PMMA preset pinned it at 0.03, which at 10-50 kPa is far below
    delta_eq, so the rail rather than the contact law set the compaction. This
    sizes it from the softest, smallest species -- the binding case -- with
    ``contact_overlap_safety`` of headroom, clamped to a sane band.
    """
    E = float(getattr(p, 'E_modulus', 10.0))
    nu = float(getattr(p, 'poisson_ratio', 0.45))
    radii = []
    moduli = []
    for sp in (getattr(p, 'species', None) or []):
        rm = sp.get('radius_mean')
        if rm:
            radii.append(float(rm))
        spE = sp.get('E_kPa')
        if spE:
            moduli.append(float(spE))
    if not radii:
        radii = [float(getattr(p, 'R_func_mean', 40.0)), float(getattr(p, 'R_inert_mean', 40.0))]
    if moduli:
        E = min(moduli)
    cap = float(getattr(p, 'contact_E_cap', 0.0))
    if cap > 0:
        E = min(E, cap)                       # the CONTACT modulus is what resists
    r = max(1e-9, min(rr for rr in radii if rr > 0))
    E_s = max(1e-12, E / (2.0 * (1.0 - nu * nu)))      # kPa -> nN/um^2 is 1e-3 * 1e3
    R_star = 0.5 * r
    F_ref = max(1e-12, float(getattr(p, 'expected_bridge_force', 100.0)))
    delta_eq = (3.0 * F_ref / (4.0 * E_s * np.sqrt(R_star))) ** (2.0 / 3.0)
    frac = float(getattr(p, 'contact_overlap_safety', 1.5)) * delta_eq / r
    return float(min(hi, max(lo, frac)))


# ══════════════════════════════════════════════════════════════════════
# V3.0 dispatch: pure-Python reference loops live in gels.kernels.reference
# ══════════════════════════════════════════════════════════════════════
# The per-pair / per-cell / per-voxel loops were moved verbatim to
# gels/kernels/reference.py (Phase 0). These wrappers keep every existing
# call site and import working; the compiled kernels (Phase 6) plug in here
# behind `p.use_numba` / `p.deformable_enabled`.

_reference_module = None

def _reference():
    """Lazily import the reference implementation (avoids a circular import)."""
    global _reference_module
    if _reference_module is None:
        from gels.kernels import reference as _ref
        _reference_module = _ref
    return _reference_module


def cells_kernels_enabled(p):
    """Compiled cell state machine + bridging (statistically equivalent to the reference)."""
    return kernels_enabled(p) and getattr(p, 'perf_cells_backend', 'kernels') == 'kernels'


def update_cell_state(gs: GranuleSystem, p: Params, t: float, rng=None):
    if cells_kernels_enabled(p):
        from gels.kernels.cells import update_cell_state_k
        return update_cell_state_k(gs, p, t, rng)
    return _reference().update_cell_state(gs, p, t, rng)


def mc_dem_correction(contacts, gs, p, E_star_gg, F):
    return _reference().mc_dem_correction(contacts, gs, p, E_star_gg, F)


def kernels_enabled(p):
    """True when the compiled kernels (gels.kernels.contact2d / contact3d) serve this Params.

    Requires numba, ``use_numba``, rigid granules (LS-DEM runs on the
    reference path) and ``perf_neighbor_backend != 'reference'``.
    """
    return bool(_HAS_NUMBA and getattr(p, 'use_numba', True)
                and not getattr(p, 'deformable_enabled', False)
                and getattr(p, 'perf_neighbor_backend', 'cells') != 'reference')


def compute_forces(gs: GranuleSystem, p: Params, rng):
    if kernels_enabled(p):
        from gels.kernels.contact2d import compute_forces_2d
        return compute_forces_2d(gs, p, rng)
    return _reference().compute_forces(gs, p, rng)


def compute_forces_3d(gs: GranuleSystem, p: Params, rng):
    if kernels_enabled(p):
        from gels.kernels.contact3d import compute_forces_3d as _compute_forces_3d_k
        return _compute_forces_3d_k(gs, p, rng)
    return _reference().compute_forces_3d(gs, p, rng)


def _resolve_overlaps(gs, p):
    if kernels_enabled(p):
        from gels.kernels.integrate import resolve_overlaps
        return resolve_overlaps(gs, p)
    return _reference()._resolve_overlaps(gs, p)


def _render_kernels_ok(gs, p):
    """Compiled rendering serves rigid granules only (LS-DEM keeps the pull-back path)."""
    return kernels_enabled(p) and getattr(gs, 'epsilon', None) is None


def render_fields(gs: GranuleSystem, p: Params):
    if _render_kernels_ok(gs, p):
        from gels.kernels.render import render_fields_species as _rfs
        return group_species_fields(gs, _rfs(gs, p))
    return _reference().render_fields(gs, p)


def render_fields_species(gs, p):
    """Per-species phase-field grids phi_s of shape (K, Ng, Ng[, Ng]) (V3.0)."""
    if _render_kernels_ok(gs, p):
        from gels.kernels.render import render_fields_species as _rfs
        return _rfs(gs, p)
    return _reference().render_fields_species(gs, p)


def render_fields_3d(gs: GranuleSystem, p: Params):
    if _render_kernels_ok(gs, p):
        from gels.kernels.render import render_fields_species as _rfs
        return group_species_fields(gs, _rfs(gs, p))
    return _reference().render_fields_3d(gs, p)


def connectivity(field, thresh_frac=0.3):
    return _reference().connectivity(field, thresh_frac)


def compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces, phi_s=None):
    if kernels_enabled(p):
        from gels.kernels.metrics import compute_metrics as _cm
        return _cm(gs, p, phi_f, phi_i, phi_v, t, forces, phi_s=phi_s)
    return _reference().compute_metrics(gs, p, phi_f, phi_i, phi_v, t, forces, phi_s=phi_s)


def _settle_packing_2d(gs, p):
    if kernels_enabled(p):
        from gels.kernels.packing import settle_packing_2d
        return settle_packing_2d(gs, p)
    return _reference()._settle_packing_2d(gs, p)


def _settle_packing_3d(gs, p):
    if kernels_enabled(p):
        from gels.kernels.packing import settle_packing_3d
        return settle_packing_3d(gs, p)
    return _reference()._settle_packing_3d(gs, p)


def _sp_get(sp, key, default):
    """Float species attribute with a default for missing/None keys."""
    v = sp.get(key)
    return float(default) if v is None else float(v)


def species_phi_targets(species, p):
    """Absolute solid fraction of the domain each species should occupy.

    A species may state `phi_target` directly (legacy conversion does, using
    the exact V2.7 values); otherwise it is `solid_fraction × volume_fraction`
    with the solid fraction from `phi_solid_target` (or phi_f + phi_i).
    """
    solid = p.phi_solid_target if p.phi_solid_target > 0 else (p.phi_f_target + p.phi_i_target)
    out = []
    for sp in species:
        if sp.get('phi_target') is not None:
            out.append(float(sp['phi_target']))
        else:
            out.append(solid * _sp_get(sp, 'volume_fraction', 0.0))
    return out


def total_phi_target(species, p):
    """Σ species phi targets, summed in index order (legacy: phi_f + phi_i)."""
    phis = species_phi_targets(species, p)
    tot = phis[0] if phis else 0.0
    for v in phis[1:]:
        tot = tot + v
    return tot


def _mean_particle_measure(sp, dim, rule):
    """Mean granule area (2D) or volume (3D) for the species' radius distribution.

    rule 'mean_radius': V(μ) — the V2.7 rule. 'mean_volume': E[V] from the
    distribution moments (normal: E[R²]=μ²+σ², E[R³]=μ³+3μσ²; lognormal via
    s² = ln(1+σ²/μ²)).
    """
    mu = float(sp['radius_mean'])
    sd = _sp_get(sp, 'radius_std', 0.0)
    dist = str(sp.get('radius_distribution') or 'normal').lower()
    if rule == 'mean_radius' or sd <= 0.0:
        return np.pi * mu**2 if dim == 2 else (4.0/3.0) * np.pi * mu**3
    if dist == 'lognormal':
        s2 = np.log(1.0 + (sd / mu) ** 2)
        ER2 = mu**2 * np.exp(s2)
        ER3 = mu**3 * np.exp(3.0 * s2)
    else:
        ER2 = mu**2 + sd**2
        ER3 = mu**3 + 3.0 * mu * sd**2
    return np.pi * ER2 if dim == 2 else (4.0/3.0) * np.pi * ER3


def species_counts(species, p, mode=None):
    """Number of granules to place per species for the domain in `p`.

    n_k = round(phi_k · V_domain / V̄_k). A species-level `count_rule`
    overrides `p.packing_count_rule`. 2D-slice counts the 3D packing.
    """
    mode = mode or p.mode
    dim = 3 if mode in ('3D', '2D-slice') else 2
    V_dom = domain_volume(p, mode)          # box: Lx*Ly*Lz | Lx*Ly verbatim; cylinder: pi R^2 Lz (V3.1)
    counts = []
    for sp, phi_k in zip(species, species_phi_targets(species, p)):
        rule = str(sp.get('count_rule') or p.packing_count_rule).lower()
        Vbar = _mean_particle_measure(sp, dim, rule)
        counts.append(int(round(phi_k * V_dom / Vbar)))
    return counts


def _placement_order(species, counts):
    """Round-robin over species in ascending f (bare first) for good mixing.

    For the legacy pair this is exactly the V2.7 inert/functional interleave.
    """
    ks = sorted(range(len(species)), key=lambda k: (float(species[k]['f']), k))
    remaining = list(counts)
    order = []
    while any(remaining[k] > 0 for k in ks):
        for k in ks:
            if remaining[k] > 0:
                order.append(k)
                remaining[k] -= 1
    return order


def _sample_radius(sp, rng):
    """One radius draw for a species (normal or lognormal), floored at radius_min."""
    mu = float(sp['radius_mean'])
    sd = _sp_get(sp, 'radius_std', 0.0)
    rmin = _sp_get(sp, 'radius_min', 1.0)
    dist = str(sp.get('radius_distribution') or 'normal').lower()
    if dist == 'lognormal' and sd > 0.0 and mu > 0.0:
        s2 = np.log(1.0 + (sd / mu) ** 2)
        draw = rng.lognormal(np.log(mu) - 0.5 * s2, np.sqrt(s2))
    else:
        draw = rng.normal(mu, sd)
    return max(rmin, draw)


def _print_species_targets(species, counts, p):
    for k, (sp, n) in enumerate(zip(species, counts)):
        line = (f"  Target: {n} × {sp.get('name', k)} (f={float(sp['f']):.2f}, "
                f"R~{float(sp['radius_mean']):.0f}µm")
        if p.shape_enabled:
            line += (f", AR={_sp_get(sp, 'aspect_ratio_mean', 1.0):.2f}±"
                     f"{_sp_get(sp, 'aspect_ratio_std', 0.0):.2f}, "
                     f"n={_sp_get(sp, 'blockiness_mean', 2.0):.1f}±"
                     f"{_sp_get(sp, 'blockiness_std', 0.0):.1f}")
        print(line + ")")


def _print_species_placed(gs, frac_per_granule):
    parts = []
    for k in range(gs.K):
        mk = gs.species_id == k
        parts.append(f"{int(np.sum(mk))} {gs.species_names[k]} "
                     f"(f={gs.species_f[k]:.2f}, φ={float(np.sum(frac_per_granule[mk])):.3f})")
    print("  Placed: " + " + ".join(parts))
    print(f"  Void fraction: {1.0 - float(np.sum(frac_per_granule)):.3f}")


def get_pair_friction_params(f_i, f_j, p: Params):
    """Return (tau_0 [Pa], W_adhesion [J/m²]) for a pair of coverages f_i, f_j.

    Coverage mixing (gels.materials.mix_pair): collagen–collagen with
    probability f_i·f_j, collagen–bare with f_i(1−f_j)+f_j(1−f_i), bare–bare
    otherwise. f = 1/0 reproduces the V2.7 functional/inert LUT exactly.
    Vectorises over numpy arrays.
    """
    from gels import materials as _mat
    tau = _mat.mix_pair(f_i, f_j, p.tau_0_cc, p.tau_0_cb, p.tau_0_bb)
    W = _mat.mix_pair(f_i, f_j, p.W_adh_cc, p.W_adh_cb, p.W_adh_bb)
    return tau, W


# ══════════════════════════════════════════════════════════════════════
# Superellipse geometry (V1.3)
# ══════════════════════════════════════════════════════════════════════
#
# A 2D superellipse: |x/a|^n + |y/b|^n = 1
# Parametric form:
#   x(t) = a |cos t|^(2/n) sign(cos t)
#   y(t) = b |sin t|^(2/n) sign(sin t)      t in [0, 2pi)

def superellipse_area(a, b, n):
    """Exact area of superellipse |x/a|^n + |y/b|^n = 1."""
    return 4.0 * a * b * _gamma(1.0 + 1.0/n)**2 / _gamma(1.0 + 2.0/n)


@njit(cache=True)
def superellipse_point(t, a, b, n):
    """Point (x, y) on superellipse boundary at parameter t (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    x = a * np.sign(ct) * np.abs(ct)**e
    y = b * np.sign(st) * np.abs(st)**e
    return x, y


@njit(cache=True)
def superellipse_tangent(t, a, b, n):
    """Unnormalised tangent vector dx/dt, dy/dt (body frame)."""
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    # dx/dt = a * (2/n) * |cos t|^(2/n - 1) * sin t  (with signs handled)
    dxdt = -a * e * np.sign(ct) * np.abs(ct)**(e - 1.0) * st
    dydt =  b * e * np.sign(st) * np.abs(st)**(e - 1.0) * ct
    return dxdt, dydt


@njit(cache=True)
def superellipse_normal_vec(t, a, b, n):
    """Outward unit normal at parameter t (body frame)."""
    dxdt, dydt = superellipse_tangent(t, a, b, n)
    # outward normal = (dy/dt, -dx/dt) normalised
    nx, ny = dydt, -dxdt
    mag = np.sqrt(nx*nx + ny*ny)
    if mag < 1e-30:
        return 0.0, 0.0
    return nx / mag, ny / mag


@njit(cache=True)
def superellipse_curvature_radius(t, a, b, n):
    """
    Local radius of curvature R = 1/kappa at parameter t (body frame).

    kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)
    """
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / n
    eps = 1e-30

    # First derivatives
    act = np.abs(ct) + eps
    ast = np.abs(st) + eps
    dxdt = -a * e * np.sign(ct) * act**(e - 1.0) * st
    dydt =  b * e * np.sign(st) * ast**(e - 1.0) * ct

    # Second derivatives
    d2xdt2 = -a * e * (
        np.sign(ct) * (e - 1.0) * act**(e - 2.0) * st * st
        + np.sign(ct) * act**(e - 1.0) * ct
    )
    d2ydt2 = b * e * (
        np.sign(st) * (e - 1.0) * ast**(e - 2.0) * ct * ct
        - np.sign(st) * ast**(e - 1.0) * st
    )

    num = abs(dxdt * d2ydt2 - dydt * d2xdt2)
    den = (dxdt**2 + dydt**2)**1.5
    if den < 1e-30 or num < 1e-30:
        return max(a, b) * 10.0  # fallback: large radius (nearly flat)
    return den / num


def superellipse_polygon_pts(cx, cy, a, b, n, theta, num_pts=64):
    """Polygon vertices for rendering a superellipse in world coords."""
    t = np.linspace(0, 2*np.pi, num_pts, endpoint=False)
    e = 2.0 / n
    ct, st = np.cos(t), np.sin(t)
    bx = a * np.sign(ct) * np.abs(ct)**e
    by = b * np.sign(st) * np.abs(st)**e
    # Rotate to world frame
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    wx = cx + cos_th * bx - sin_th * by
    wy = cy + sin_th * bx + cos_th * by
    return np.column_stack([wx, wy])


def superellipse_perimeter(a, b, n, num_pts=256):
    """Numerical perimeter of a superellipse."""
    pts = superellipse_polygon_pts(0, 0, a, b, n, 0.0, num_pts)
    diffs = np.diff(pts, axis=0, append=pts[:1])
    return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))


@njit(cache=True)
def _world_to_body(px, py, cx, cy, theta):
    """Transform world point(s) to body frame of granule at (cx,cy,theta)."""
    dx, dy = px - cx, py - cy
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cos_th * dx + sin_th * dy, -sin_th * dx + cos_th * dy


@njit(cache=True)
def _body_to_world(bx, by, cx, cy, theta):
    """Transform body point(s) to world frame."""
    cos_th, sin_th = np.cos(theta), np.sin(theta)
    return cx + cos_th * bx - sin_th * by, cy + sin_th * bx + cos_th * by


@njit(cache=True)
def superellipse_implicit(px, py, cx, cy, a, b, n, theta):
    """
    Evaluate the superellipse implicit function at world point (px, py).
    Returns value < 1 if inside, = 1 on boundary, > 1 outside.
    """
    bx, by = _world_to_body(px, py, cx, cy, theta)
    return (np.abs(bx) / a)**n + (np.abs(by) / b)**n


def _find_closest_param(bx_target, by_target, a, b, n, t_guess=None):
    """
    Find parameter t such that the superellipse point is closest to (bx_target, by_target).
    Used for contact detection. Minimises distance in body frame.
    """
    from scipy.optimize import minimize_scalar

    def dist2(t):
        px, py = superellipse_point(t, a, b, n)
        return (px - bx_target)**2 + (py - by_target)**2

    if t_guess is not None:
        # Local search near guess
        res = minimize_scalar(dist2, bounds=(t_guess - 0.5, t_guess + 0.5),
                              method='bounded')
        return res.x % (2 * np.pi)

    # Global search: sample then refine
    ts = np.linspace(0, 2*np.pi, 64, endpoint=False)
    d2 = np.array([dist2(t) for t in ts])
    best = ts[np.argmin(d2)]
    res = minimize_scalar(dist2, bounds=(best - 0.15, best + 0.15),
                          method='bounded')
    return res.x % (2 * np.pi)


# ── Contact detection: superellipse–superellipse (common normal) ──

@njit(cache=True)
def find_contact_superellipses_cn(xi, yi, ai, bi, ni, thetai,
                                   xj, yj, aj, bj, nj, thetaj):
    """
    RETIRED (V3.4). The common-normal solver, kept verbatim as EVIDENCE only.

    Nothing in the engine calls this. It is retained so the two defects recorded
    in the V3.4 changelog stay checkable from the suite
    (``tests/test_mtd_contact.py::TestAgainstTheCommonNormalSolver``):

      * `delta` is `dp_mag`, the distance between two common-normal SURFACE
        POINTS, not a penetration depth -- over-reported by median 9.8x, p90 43x;
      * the reported normal is `(p_j - p_i)/|.|`, which points from j back toward
        i once those points cross, i.e. for **29 % of 2D and 50 % of 3D** shaped
        contacts at preset blockiness. Since the force law applies `-F_normal * n`
        to body i, a flipped normal turns repulsion into attraction.

    `find_contact_superellipses` now dispatches to the support-function MTD
    solver. Delete this once the V2.7 `run2d_shapes` fixture is retired.

    Common normal contact detection between two superellipses.

    Returns: (in_contact, delta, nx, ny, cx, cy, R_loc_i, R_loc_j)
        in_contact: bool
        delta: penetration depth (µm), > 0 if overlapping
        nx, ny: unit contact normal (from i toward j)
        cx, cy: contact point (world coords)
        R_loc_i, R_loc_j: local curvature radii at contact (µm)

    Returns None if no contact (gap > 0).
    """
    # Direction between centres
    dx_c = xj - xi
    dy_c = yj - yi
    d_c = np.sqrt(dx_c**2 + dy_c**2)
    if d_c < 1e-12:
        return None

    # Unit vector from i to j
    ux, uy = dx_c / d_c, dy_c / d_c

    # Initial guess: parameter on each surface closest to the line of centres
    # Transform centre-to-centre direction into each body frame
    bx_i, by_i = (np.cos(thetai) * ux + np.sin(thetai) * uy,
                  -np.sin(thetai) * ux + np.cos(thetai) * uy)
    bx_j, by_j = (np.cos(thetaj) * (-ux) + np.sin(thetaj) * (-uy),
                  -np.sin(thetaj) * (-ux) + np.cos(thetaj) * (-uy))

    t_i = np.arctan2(by_i, bx_i)
    t_j = np.arctan2(by_j, bx_j)

    # Newton-Raphson iteration: refine contact parameters
    # Find points P_i on surface i and P_j on surface j such that the
    # normal at P_i points toward P_j and vice versa.
    for iteration in range(12):
        # Points on each surface (body frame)
        px_i, py_i = superellipse_point(t_i, ai, bi, ni)
        px_j, py_j = superellipse_point(t_j, aj, bj, nj)

        # Transform to world frame
        wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
        wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

        # Vector from P_i to P_j
        dpx = wx_j - wx_i
        dpy = wy_j - wy_i
        dp_mag = np.sqrt(dpx**2 + dpy**2)
        if dp_mag < 1e-12:
            break

        # Normal at P_i (body frame) then rotate to world
        nix, niy = superellipse_normal_vec(t_i, ai, bi, ni)
        cos_ti, sin_ti = np.cos(thetai), np.sin(thetai)
        nix_w = cos_ti * nix - sin_ti * niy
        niy_w = sin_ti * nix + cos_ti * niy

        # Normal at P_j (body frame) then rotate to world
        njx, njy = superellipse_normal_vec(t_j, aj, bj, nj)
        cos_tj, sin_tj = np.cos(thetaj), np.sin(thetaj)
        njx_w = cos_tj * njx - sin_tj * njy
        njy_w = sin_tj * njx + cos_tj * njy

        # Desired: n_i should point along P_i→P_j, n_j should point along P_j→P_i
        # Error: angle between n_i and (P_i→P_j)
        target_nx = dpx / dp_mag
        target_ny = dpy / dp_mag

        # Cross-product errors (should be zero when aligned)
        err_i = nix_w * target_ny - niy_w * target_nx
        err_j = njx_w * (-target_ny) - njy_w * (-target_nx)

        if abs(err_i) < 1e-8 and abs(err_j) < 1e-8:
            break

        # Simple gradient step (damped Newton)
        t_i -= 0.5 * err_i
        t_j -= 0.5 * err_j

    # Final contact geometry
    px_i, py_i = superellipse_point(t_i, ai, bi, ni)
    px_j, py_j = superellipse_point(t_j, aj, bj, nj)
    wx_i, wy_i = _body_to_world(px_i, py_i, xi, yi, thetai)
    wx_j, wy_j = _body_to_world(px_j, py_j, xj, yj, thetaj)

    # Contact normal: from surface point i toward surface point j
    cpx = wx_j - wx_i
    cpy = wy_j - wy_i
    cp_mag = np.sqrt(cpx**2 + cpy**2)

    # Check if P_i is inside body j (overlap)
    val_j = superellipse_implicit(wx_i, wy_i, xj, yj, aj, bj, nj, thetaj)
    val_i = superellipse_implicit(wx_j, wy_j, xi, yi, ai, bi, ni, thetai)

    if val_j > 1.0 and val_i > 1.0:
        # No overlap
        return None

    # Penetration depth: distance between surface points (with sign)
    delta = cp_mag
    if cp_mag < 1e-12:
        # Coincident surface points — use implicit function for depth estimate
        delta = max(ai, bi) * (1.0 - val_j**(1.0/nj)) if val_j < 1.0 else 0.01

    # Contact normal (from i toward j, along line of centres as fallback)
    if cp_mag > 1e-12:
        nx, ny = cpx / cp_mag, cpy / cp_mag
    else:
        nx, ny = ux, uy

    # Contact point: midpoint of the two surface points
    contact_x = 0.5 * (wx_i + wx_j)
    contact_y = 0.5 * (wy_i + wy_j)

    # Local curvature radii
    R_loc_i = superellipse_curvature_radius(t_i, ai, bi, ni)
    R_loc_j = superellipse_curvature_radius(t_j, aj, bj, nj)

    return (True, delta, nx, ny, contact_x, contact_y, R_loc_i, R_loc_j)


# ── Contact detection: superellipse–wall ──

@njit(cache=True)
def find_contact_superellipse_wall_cn(xi, yi, ai, bi, ni, thetai,
                                      wall_pos, wall_axis, wall_sign):
    """
    RETIRED (V3.4): the 64-point ring sampler. Evidence only -- nothing calls
    this. ``find_contact_superellipse_wall`` is now one support evaluation, and
    `tests/test_wall_contact.py` measures the two against each other.

    Contact detection between a superellipse and a flat wall.

    wall_axis: 0 = vertical wall (x = wall_pos), 1 = horizontal wall (y = wall_pos)
    wall_sign: +1 if granule should be to the right/above wall_pos,
               -1 if granule should be to the left/below wall_pos

    Returns: (penetration, R_local) or None if no contact.
        penetration: > 0 if overlapping
        R_local: curvature radius at contact point
    """
    # Find the extreme point of the superellipse in the wall-normal direction
    # Sample boundary and find the point closest to the wall
    n_sample = 64
    # np.linspace(..., endpoint=False) is unsupported in numba nopython mode;
    # this arange form is numerically identical and compiles.
    t_vals = np.arange(n_sample) * (2.0 * np.pi / n_sample)
    e = 2.0 / ni
    ct, st = np.cos(t_vals), np.sin(t_vals)
    bx = ai * np.sign(ct) * np.abs(ct)**e
    by = bi * np.sign(st) * np.abs(st)**e
    cos_th, sin_th = np.cos(thetai), np.sin(thetai)
    wx = xi + cos_th * bx - sin_th * by
    wy = yi + sin_th * bx + cos_th * by

    if wall_axis == 0:
        coords = wx
    else:
        coords = wy

    if wall_sign > 0:
        # Wall at low side: penetration = wall_pos - min(coords)
        idx = np.argmin(coords)
        pen = wall_pos - coords[idx]
    else:
        # Wall at high side: penetration = max(coords) - wall_pos
        idx = np.argmax(coords)
        pen = coords[idx] - wall_pos

    if pen <= 0:
        return None

    # Refine: local curvature at the contact point
    t_contact = t_vals[idx]
    R_local = superellipse_curvature_radius(t_contact, ai, bi, ni)

    return (pen, R_local)


def overlap_lens_area(R1, R2, d):
    """
    Area of the lens-shaped intersection of two 2D circles.

    R1, R2 : radii (µm)
    d      : centre-to-centre distance (µm)
    Returns area in µm².
    """
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return np.pi * min(R1, R2)**2
    cos_a1 = np.clip((d*d + R1*R1 - R2*R2) / (2.0 * d * R1), -1, 1)
    cos_a2 = np.clip((d*d + R2*R2 - R1*R1) / (2.0 * d * R2), -1, 1)
    arg = (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)
    return R1*R1 * np.arccos(cos_a1) + R2*R2 * np.arccos(cos_a2) \
           - 0.5 * np.sqrt(max(0.0, arg))


def overlap_lens_volume(R1, R2, d):
    """
    Volume of the lens-shaped intersection of two 3D spheres.

    R1, R2 : radii (µm)
    d      : centre-to-centre distance (µm)
    Returns volume in µm³.
    """
    if d >= R1 + R2:
        return 0.0
    if d <= abs(R1 - R2):
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    if d < 1e-12:
        return (4.0 / 3.0) * np.pi * min(R1, R2)**3
    # V = π(R1+R2-d)²(d² + 2d(R1+R2) - 3(R1-R2)²) / (12d)
    s = R1 + R2 - d
    numer = s * s * (d * d + 2.0 * d * (R1 + R2) - 3.0 * (R1 - R2)**2)
    return np.pi * numer / (12.0 * d)


def compute_effective_radii_3d(gs, p=None):
    """
    Volume-conserving effective radii for 3D spheres.

    When DEM spheres overlap, the lens-shaped intersection is material
    that is geometrically double-counted.  To conserve volume, each
    granule's display radius is inflated:

        4/3 π r_eff³ = 4/3 π r³ + ΔV_i

    where ΔV_i is granule i's share of its total overlap volume, split
    proportionally to r³.

    Returns (r_eff, overlap_volume_per_granule).
    """
    pos = gs.positions()
    max_r = np.max(gs.r)
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly, p.Lz])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_vol = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dv = pos[j] - pos[i]
        if periodic:
            dv[0], dv[1], dv[2] = minimum_image_disp_3d(
                dv[0], dv[1], dv[2], p.Lx, p.Ly, p.Lz)
        d = np.sqrt(np.dot(dv, dv))
        if d < gs.r[i] + gs.r[j]:
            V_lens = overlap_lens_volume(gs.r[i], gs.r[j], d)
            frac_i = gs.r[i]**3 / (gs.r[i]**3 + gs.r[j]**3)
            overlap_vol[i] += frac_i * V_lens
            overlap_vol[j] += (1.0 - frac_i) * V_lens

    r_eff = (gs.r**3 + 3.0 * overlap_vol / (4.0 * np.pi)) ** (1.0 / 3.0)
    return r_eff, overlap_vol


def compute_effective_radii(gs, p=None):
    """
    Volume-conserving effective radii (2D).

    When DEM circles overlap, the lens-shaped intersection is material
    that is geometrically double-counted.  To conserve 2D area (proxy
    for 3D volume), each granule's display radius is inflated:

        π r_eff² = π r² + ΔA_i

    where ΔA_i is granule i's share of its total overlap area, split
    proportionally to r².

    Returns (r_eff, overlap_area_per_granule).
    """
    pos = gs.positions()
    max_r = np.max(gs.r)
    periodic = (p is not None and getattr(p, 'boundary_mode', 'walls') == 'periodic')
    if periodic:
        tree = cKDTree(pos, boxsize=[p.Lx, p.Ly])
    else:
        tree = cKDTree(pos)
    pairs = tree.query_pairs(2 * max_r, output_type='ndarray')

    overlap_area = np.zeros(gs.N)
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dx = pos[j, 0] - pos[i, 0]
        dy = pos[j, 1] - pos[i, 1]
        if periodic:
            dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
        d = np.sqrt(dx*dx + dy*dy)
        if d < gs.r[i] + gs.r[j]:
            A_lens = overlap_lens_area(gs.r[i], gs.r[j], d)
            frac_i = gs.r[i]**2 / (gs.r[i]**2 + gs.r[j]**2)
            overlap_area[i] += frac_i * A_lens
            overlap_area[j] += (1.0 - frac_i) * A_lens

    r_eff = np.sqrt(gs.r**2 + overlap_area / np.pi)
    return r_eff, overlap_area


def print_stiffness_info(p: Params):
    """Show expected overlap for the configured modulus."""
    E_star = (p.E_modulus * 1e3) / (2.0 * (1.0 - p.poisson_ratio**2))
    R_eff = p.R_func_mean / 2.0  # two same-size functional granules
    F_typ = p.F_max_per_cell

    coeff = (4.0 / 3.0) * E_star * np.sqrt(R_eff) * 1e-3
    delta_eq = (F_typ / coeff) ** (2.0 / 3.0)

    print(f"  Material: E = {p.E_modulus} kPa, ν = {p.poisson_ratio}")
    print(f"    E* = {E_star:.0f} Pa (effective modulus)")
    print(f"    For F = {F_typ:.0f} nN between R = {p.R_func_mean:.0f} µm granules:")
    print(f"    → Equilibrium overlap δ = {delta_eq:.2f} µm "
          f"({delta_eq / p.R_func_mean * 100:.1f}% of R)")


# ══════════════════════════════════════════════════════════════════════
# Cell geometry & motor-clutch model
# ══════════════════════════════════════════════════════════════════════

def cell_projected_area(spread_frac, cell_d, cell_h):
    """
    Projected area of a cell transitioning from sphere to spread ellipsoid.

    Sphere (spread_frac=0):
        A = π (d/2)²

    Fully spread oblate ellipsoid (spread_frac=1):
        Volume conserved: V = (4/3)π(d/2)³ = (4/3)π a² (h/2)
        → a² = d³ / (4h)
        → A_spread = π a² = π d³ / (4h)

    Interpolates linearly between the two based on spread_frac.
    """
    r = cell_d / 2.0
    A_sphere = np.pi * r ** 2

    V = (4.0 / 3.0) * np.pi * r ** 3
    c = cell_h / 2.0
    a2 = 3.0 * V / (4.0 * np.pi * c)  # semi-major axis squared
    A_spread = np.pi * a2

    return A_sphere + spread_frac * (A_spread - A_sphere)


def max_cells_on_granule(R, cell_proj_area, coverage):
    """Max cells that fit on a granule based on projected area ratio."""
    A_granule = np.pi * R ** 2
    return max(1, int(A_granule * coverage / cell_proj_area))


# V3.3: areal efficiency of packing discs on a surface. A_surface / A_cell
# tiles at 100 %, which no packing achieves: the planar maximum is the
# hexagonal 0.9069 (random close packing of discs is ~0.83). Without this the
# capacity rule returned geometrically IMPOSSIBLE numbers -- at R = 40 um with
# cell_capacity_foothold = 0.25 it gave 64 cells where 58 rigid 20 um discs is
# the hard ceiling. Curvature makes it slightly worse still (a 20 um disc
# subtends 319 um^2 of an R = 40 sphere, not 314), which this does not model.
# A geometric constant, not a knob: module-level rather than a Params field,
# following gels/kernels/packing.py:196-198.
PACKING_EFFICIENCY = 0.9069


def cells_from_surface_coverage(R, cell_d, cell_h, coverage, mode="3D", foothold=1.0):
    """Compute number of cells for a target surface coverage fraction.

    coverage = 1.0 is a full monolayer of spread cells; >1.0 means stacking.

    ``foothold`` (V3.2) is the fraction of the fully-spread footprint a cell
    actually needs to hold a place, and is used for CAPACITY rather than for
    seeding. Cells do not insist on spreading flat: on a small or crowded
    granule they stay rounder and grip whatever contact area they can get, and
    a cell spanning two granules needs only a foothold on each. With the V3.1
    rule (foothold = 1) a 40 um granule in 2D has a projected area of 1257 um^2
    against a spread cell's 1257 um^2, so its capacity is exactly ONE cell and
    the bed is confluent at seeding -- which the lab does not see.

    The reference area is floored at the ROUNDED cell's cross-section, because a
    cell cannot occupy less than that; for d = 20, h = 5 that floor is 0.25 of
    the spread footprint, so foothold <= 0.25 gives a rounded-cell reference and
    a 40 um granule holds 4.

    The result is scaled by ``PACKING_EFFICIENCY`` (V3.3), because cells cannot
    tile a surface at 100 %.

    3D/2D-slice: uses sphere surface area 4πR².
    2D:          uses projected disk area πR².
    """
    A_cell_spread = cell_projected_area(1.0, cell_d, cell_h)
    if foothold < 1.0:
        A_cell_spread = max(cell_projected_area(0.0, cell_d, cell_h),
                            float(foothold) * A_cell_spread)
    if mode in ("3D", "2D-slice"):
        A_surface = 4.0 * np.pi * R ** 2
    else:
        A_surface = np.pi * R ** 2
    return max(1, int(round(A_surface * coverage * PACKING_EFFICIENCY / A_cell_spread)))


def seeded_cells(n_full, f, p):
    """Cells seeded on a granule of coverage f: seeding_gain(f) × the f = 1 count.

    f = 1 keeps the V2.7 count exactly; f = 0 seeds none; partial coatings
    follow the attachment Langmuir (K_sigma_attach) or power law. May be 0.
    """
    g = float(seeding_gain(f, p))
    return int(round(g * n_full))


def cell_capacity(gs, i, p, A_cell=None):
    """Monolayer capacity of granule i, scaled by its coverage (V3.0).

    Uses cell_surface_coverage when set, otherwise the projected-area rule
    with `A_cell` (the current per-cell footprint; defaults to the spread
    state of granule i). Equals the V2.7 capacity at f = 1.
    """
    if capacity_coverage(p) > 0:
        n_full = cells_from_surface_coverage(
            gs.r[i], p.cell_diameter, p.cell_height_spread,
            capacity_coverage(p), p.mode,
            float(getattr(p, 'cell_capacity_foothold', 1.0)))
    else:
        if A_cell is None:
            A_cell = cell_projected_area(
                gs.spread_fraction[i], p.cell_diameter, p.cell_height_spread)
        n_full = max_cells_on_granule(gs.r[i], A_cell, p.cell_coverage)
    return int(round(float(gs.activity[i]) * n_full))


def motor_clutch_force(E_kPa, p: Params, fa_maturity_val, nu=None, g=1.0):
    """
    Steady-state traction force per cell from the motor-clutch model.

    Based on Chan & Odde (2008).  For hydrogel substrates (1–100 kPa) we
    are in the rising portion of the stiffness–force curve:

        F = F_stall · k_sub / (k_sub + g·k_opt) · engagement · fa_maturity · g

    where
        k_sub = π E a / (1−ν²)        substrate stiffness at cell scale
        k_opt = n_clutches · k_clutch  optimal (clutch ensemble) stiffness
        engagement = k_on / (k_on + k_off)   steady-state clutch fraction
        g          ligand gain from the coating (gels.materials.traction_gain);
                   enters as the engaged-bond ceiling AND the reduced clutch
                   ensemble stiffness (Erdmann & Schwarz 2004; Bangasser 2017).
                   g = 1 is the V2.7 model. Never emulate g by lowering
                   n_clutches — in this formula that *raises* the force.

    E_kPa, nu are the HOST granule's modulus and Poisson ratio (V3.0);
    nu=None uses p.poisson_ratio. Returns force per cell in nN.
    """
    # Total motor stall force
    F_stall = p.n_motors * p.F_motor_stall  # nN

    # Substrate stiffness at cell scale (nN/µm)
    # k = π E a / (1-ν²),  E [kPa] × a [µm] → kPa·µm = nN/µm
    a_cell = p.cell_diameter / 2.0
    nu_val = float(p.poisson_ratio if nu is None else nu)
    k_sub = np.pi * float(E_kPa) * a_cell / (1.0 - nu_val ** 2)

    # Clutch ensemble stiffness
    k_opt = p.n_clutches * p.k_clutch

    # Steady-state engagement fraction
    engagement = p.k_on_clutch / (p.k_on_clutch + p.k_off_clutch)

    # Motor-clutch force with ligand gain
    F_mc = F_stall * (k_sub / (k_sub + g * k_opt)) * engagement * fa_maturity_val * g

    return min(F_mc, p.F_max_per_cell)


# ══════════════════════════════════════════════════════════════════════
# Packing settle (compression to achieve granule contact)
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# Cell initialization (V1.5)
# ══════════════════════════════════════════════════════════════════════

def _initialize_cells(gs: GranuleSystem, rng, p=None):
    """Distribute cells uniformly on functional granule surfaces.

    Assigns cell_granule_id and initial surface positions.
    Cells start as ATTACHED (ready to spread and bridge immediately).

    V3.2 also desynchronises the population (see the block at the end): a real
    seeded population is spread through the cell cycle, not all at phase zero.
    """
    for i in range(gs.N):
        if not gs.adhesive_mask[i]:
            continue
        sl = gs.cells_on_granule(i)
        n = gs.cell_offset[i + 1] - gs.cell_offset[i]
        if n == 0:
            continue
        gs.cell_granule_id[sl] = i
        gs.cell_state[sl] = int(CellState.ATTACHED)
        gs.n_attached[i] = float(n)

        if gs.mode == "3D":
            # Distribute on superellipsoid surface (avoid poles)
            etas = np.linspace(-0.8 * np.pi / 2, 0.8 * np.pi / 2, n)
            omegas = rng.uniform(0, 2 * np.pi, n)
            gs.cell_eta_local[sl] = etas
            gs.cell_omega_local[sl] = omegas
        else:
            # Distribute uniformly around superellipse perimeter
            thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
            thetas += rng.uniform(0, 2 * np.pi / max(1, n))
            gs.cell_theta_local[sl] = thetas

    # ── V3.2: desynchronise the population ────────────────────
    # Without this every cell sits at phase 0, clears the division refractory
    # period on the same step, and the whole bed divides in a burst.
    #
    # These draws are deliberately AFTER the per-granule loop: that loop is the
    # last consumer of the packing RNG stream (generate_packing* return straight
    # after this call), so an extra draw here cannot shift the surface-angle
    # draws above and the V2.7 fixtures stay bit-identical. A draw inside the
    # loop would break them.
    if p is None:
        return
    C = gs.total_cells
    if C > 0 and getattr(p, 'cell_division_enabled', False):
        T_d = max(1e-9, float(p.cell_doubling_time))
        cv = max(0.0, float(getattr(p, 'cell_division_cv', 0.0)))
        if getattr(p, 'cell_division_model', 'poisson') == 'cycle' and cv > 0:
            # lognormal with MEDIAN T_d, so the typical cell doubles on schedule
            sigma = np.sqrt(np.log1p(cv * cv))
            gs.cell_cycle_time[:C] = T_d * np.exp(sigma * rng.standard_normal(C))
        else:
            gs.cell_cycle_time[:C] = T_d
        gs.cell_age[:C] = rng.uniform(0.0, 1.0, C) * gs.cell_cycle_time[:C]
    jitter = float(getattr(p, 'cell_clock_jitter', 0.0))
    if jitter > 0.0:
        gs.cell_clock_offset[:gs.N] = rng.uniform(-jitter, jitter, gs.N)


# ══════════════════════════════════════════════════════════════════════
# Packing generator (random sequential addition)
# ══════════════════════════════════════════════════════════════════════

def generate_packing(p: Params, seed=42) -> GranuleSystem:
    rng = np.random.default_rng(seed)
    domain_area = domain_volume(p, '2D')

    # V3.0: one count per species from its solid-fraction target and mean
    # particle area (legacy species use the mean-radius rule → V2.7 counts)
    species = resolve_species(p)
    counts = species_counts(species, p, mode='2D')
    _print_species_targets(species, counts, p)

    xs, ys, rs, types = [], [], [], []
    a_list, b_list, n_shape_list, theta_list = [], [], [], []
    gap = p.packing_gap  # minimum gap between granule surfaces (µm)

    # Compute deflation factor for RSA placement (Lubachevsky-Stillinger).
    # Place granules at reduced bounding radii so RSA can achieve the target
    # count, then inflate-and-relax to the target packing fraction.
    phi_target = total_phi_target(species, p)
    r_ref_max = max([_sp_get(sp, 'radius_mean', 40.0) + 3.0 * _sp_get(sp, 'radius_std', 0.0)
                     for sp in species], default=40.0)
    H_place, phi_place, alpha_deflate, L_up = packing_placement(p, '2D', phi_target, r_ref_max)
    if alpha_deflate < 1.0:
        print(f"  RSA deflation: α={alpha_deflate:.3f} "
              f"(φ_safe={p.packing_inflate_phi_safe:.2f}, φ_target={phi_place:.3f})")
    if H_place < L_up:
        print(f"  Placement height: {H_place:.0f} of {L_up:.0f} um (gravity consolidation)")
        check_bed_fraction(p, p.mode, p.bed_phi_assumed)     # V3.2

    # Round-robin placement over species (ascending f, bare first) for good
    # mixing — reproduces the V2.7 inert/functional interleave for two species.
    order = _placement_order(species, counts)

    # V3.0: overlap test — the reference loop, or the compiled neighbour grid
    # on the kernel path (same inequality, same decision, same random draws).
    from gels.kernels.packing import RSAChecker, reference_radius
    checker = RSAChecker(2, p, alpha_deflate, gap, len(order) + 4096, reference_radius(species),
                         kernels_enabled(p))

    # ── V3.1: immobile boundary lining, placed before RSA so mobile granules pack against it ──
    n_boundary = 0
    if p.boundary_layer_enabled:
        gt_b = next(k for k, sp in enumerate(species) if str(sp.get('name')) == BOUNDARY_SPECIES_NAME)
        r_w = boundary_layer_radius(p, species)
        for (bx_, by_, _bz) in boundary_layer_sites(p, r_w, '2D'):
            xs.append(bx_); ys.append(by_)
            rs.append(r_w); types.append(gt_b)
            a_list.append(r_w); b_list.append(r_w)
            n_shape_list.append(2.0); theta_list.append(0.0)
            checker.add(bx_, by_, 0.0, r_w)
            n_boundary += 1
        print(f"  Boundary lining: {n_boundary} immobile granules (R={r_w:.0f} um, "
              f"f={p.boundary_functionalization:g})")

    n_dropped = 0        # V3.2
    for gt in order:
        sp = species[gt]
        r_eq = _sample_radius(sp, rng)

        # Sample shape parameters (per species)
        if p.shape_enabled:
            ar = max(1.0, rng.normal(_sp_get(sp, 'aspect_ratio_mean', 1.0),
                                     _sp_get(sp, 'aspect_ratio_std', 0.0)))
            n_s = max(1.5, rng.normal(_sp_get(sp, 'blockiness_mean', 2.0),
                                      _sp_get(sp, 'blockiness_std', 0.0)))
            # Compute semi-axes so superellipse has area = pi * r_eq^2
            # Initial guess: a = r_eq * sqrt(ar), b = r_eq / sqrt(ar)
            a0 = r_eq * np.sqrt(ar)
            b0 = r_eq / np.sqrt(ar)
            # Correct for exact area
            target_area = np.pi * r_eq**2
            actual_area = superellipse_area(a0, b0, n_s)
            if actual_area > 0:
                scale = np.sqrt(target_area / actual_area)
                a_val = a0 * scale
                b_val = b0 * scale
            else:
                a_val, b_val = r_eq, r_eq
            theta_val = rng.uniform(0, np.pi)
        else:
            a_val, b_val, n_s, theta_val = r_eq, r_eq, 2.0, 0.0

        # Bounding radius for overlap check
        r_bound_val = max(a_val, b_val)

        placed = False
        periodic = (p.boundary_mode == 'periodic')
        for _ in range(800):
            if periodic:
                cx = rng.uniform(0, p.Lx)
                cy = rng.uniform(0, p.Ly)
            else:
                cx = rng.uniform(r_bound_val + gap, p.Lx - r_bound_val - gap)
                cy = rng.uniform(r_bound_val + gap, H_place - r_bound_val - gap)
            if checker.ok(cx, cy, 0.0, r_bound_val):
                xs.append(cx); ys.append(cy)
                rs.append(r_eq); types.append(gt)
                a_list.append(a_val); b_list.append(b_val)
                n_shape_list.append(n_s); theta_list.append(theta_val)
                checker.add(cx, cy, 0.0, r_bound_val)
                placed = True; break
        if not placed:
            n_dropped += 1      # V3.2: counted and reported, not silently skipped

    if n_dropped:
        # V3.2: this used to be a silent `pass`, which is how a 20 % shortfall in
        # a shaped 3D packing went unnoticed. It is a warning, not an error:
        # small dense test systems legitimately lose a few granules to RSA.
        frac = n_dropped / max(1, n_dropped + len(xs))
        print(f"  WARNING: RSA could not place {n_dropped} granule(s) "
              f"({100 * frac:.1f}% of the target); the packing is short by that much.")
        if frac > 0.05:
            print("    The placement is near RSA saturation. Lower the solid fraction, "
                  "or reduce the aspect ratio / blockiness -- a blocky granule reserves "
                  "a much larger bounding sphere than its volume.")

    # Compute cells per granule
    n_cells = []
    A_cell_sphere = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
    # V3.0: cells per granule = coverage-dependent fraction of the f = 1 capacity
    species_list = resolve_species(p)
    for i in range(len(xs)):
        f_i = float(species_list[int(types[i])]['f'])
        is_boundary = str(species_list[int(types[i])].get('name')) == BOUNDARY_SPECIES_NAME
        if is_boundary and not p.boundary_layer_seed_cells:
            n_cells.append(0)
            continue
        if f_i >= p.f_min_adhesion:
            if p.cell_surface_coverage > 0:
                n_full = cells_from_surface_coverage(
                    rs[i], p.cell_diameter, p.cell_height_spread,
                    p.cell_surface_coverage, p.mode)
            else:
                cap = max_cells_on_granule(rs[i], A_cell_sphere, p.cell_coverage)
                n_full = min(p.n_cells_per_granule, cap)
            n_cells.append(seeded_cells(n_full, f_i, p))
        else:
            n_cells.append(0)

    if p.shape_enabled:
        gs = GranuleSystem(xs, ys, rs, types, n_cells,
                           a=a_list, b=b_list, n_shape=n_shape_list,
                           theta=theta_list,
                           species_id=types, species=species, p=p)
    else:
        gs = GranuleSystem(xs, ys, rs, types, n_cells,
                           species_id=types, species=species, p=p)

    # Store target radii and deflate for inflate-and-relax settling
    if alpha_deflate < 1.0 and p.packing_settle_steps > 0:
        gs._target_a = gs.a.copy()
        gs._target_b = gs.b.copy()
        gs._target_r = gs.r.copy()
        gs.a[:gs.N] *= alpha_deflate
        gs.b[:gs.N] *= alpha_deflate
        gs.r[:gs.N] *= alpha_deflate
        gs.r_bound[:gs.N] = np.maximum(gs.a[:gs.N], gs.b[:gs.N])

    # ── Inflate-and-relax settle: achieve jammed packing ──
    gs._H_place = H_place                 # placement height (gravity consolidation schedule)
    _ov_tol = settle_overlap_tolerance(gs, p)          # V3.5; None keeps the V2.7 rule
    if _ov_tol is not None:
        gs._overlap_tol = _ov_tol
    if p.packing_settle_steps > 0 and gs.N > 1:
        _settle_packing_2d(gs, p)
    for attr in ('_H_place', '_overlap_tol'):
        if hasattr(gs, attr):
            delattr(gs, attr)

    # Report packing fractions per species (actual areas)
    if p.shape_enabled:
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n_shape[i])
                          for i in range(gs.N)])
    else:
        areas = np.pi * gs.r ** 2
    _print_species_placed(gs, areas / domain_area)
    if consolidation_mode(p) == 'gravity' or boundary_geometry(p, '2D').top_free:
        _print_bed_report(gs, p)
    print(f"  Total cells: {int(sum(gs.n_cells))}")
    _initialize_cells(gs, rng, p)
    relax_packing(gs, p, rng)      # V3.5: no-op unless packing.relax == 'fire'
    gs.pos_unwrap[:] = gs.pos      # the settle is not part of the dynamics: displacement starts at 0
    return gs


# ══════════════════════════════════════════════════════════════════════
# 3D packing generator (V1.4)
# ══════════════════════════════════════════════════════════════════════

def generate_packing_3d(p: Params, seed=42) -> GranuleSystem:
    """Generate a 3D random packing of superellipsoids."""
    rng = np.random.default_rng(seed)
    domain_vol = domain_volume(p, '3D')

    # V3.0: one count per species (legacy species: mean-radius rule → V2.7 counts)
    species = resolve_species(p)
    counts = species_counts(species, p, mode='3D')
    _print_species_targets(species, counts, p)

    xs, ys, zs, rs, types = [], [], [], [], []
    a_list, b_list, c_list = [], [], []
    n1_list, n2_list = [], []
    quat_list = []
    gap = p.packing_gap

    # Compute deflation factor for RSA placement (Lubachevsky-Stillinger).
    # Place granules at reduced bounding radii so RSA can achieve the target
    # count, then inflate-and-relax to the target packing fraction.
    phi_target = total_phi_target(species, p)
    r_ref_max = max([_sp_get(sp, 'radius_mean', 40.0) + 3.0 * _sp_get(sp, 'radius_std', 0.0)
                     for sp in species], default=40.0)
    H_place, phi_place, alpha_deflate, L_up = packing_placement(p, '3D', phi_target, r_ref_max)
    geom = boundary_geometry(p, '3D')
    if alpha_deflate < 1.0:
        print(f"  RSA deflation: α={alpha_deflate:.3f} "
              f"(φ_safe={p.packing_inflate_phi_safe:.2f}, φ_target={phi_place:.3f})")
    if H_place < L_up:
        print(f"  Placement height: {H_place:.0f} of {L_up:.0f} um (gravity consolidation)")
        check_bed_fraction(p, p.mode, p.bed_phi_assumed)     # V3.2
    if geom.shape_code == 1:
        print(f"  Container: cylinder R={geom.R_cyl:.0f} um"
              + (", free top" if geom.top_free else ""))
    elif geom.top_free:
        print("  Container: box, free top")

    # Round-robin placement over species (ascending f, bare first)
    order = _placement_order(species, counts)

    from gels.kernels.packing import RSAChecker, reference_radius
    checker = RSAChecker(3, p, alpha_deflate, gap, len(order) + 4096, reference_radius(species),
                         kernels_enabled(p))

    # ── V3.1: immobile boundary lining, placed before RSA so mobile granules pack against it ──
    n_boundary = 0
    if p.boundary_layer_enabled:
        gt_b = next(k for k, sp in enumerate(species) if str(sp.get('name')) == BOUNDARY_SPECIES_NAME)
        r_w = boundary_layer_radius(p, species)
        for (bx_, by_, bz_) in boundary_layer_sites(p, r_w, '3D'):
            xs.append(bx_); ys.append(by_); zs.append(bz_)
            rs.append(r_w); types.append(gt_b)
            a_list.append(r_w); b_list.append(r_w); c_list.append(r_w)
            n1_list.append(2.0); n2_list.append(2.0)
            quat_list.append(np.array([1.0, 0.0, 0.0, 0.0]))
            checker.add(bx_, by_, bz_, r_w)
            n_boundary += 1
        print(f"  Boundary lining: {n_boundary} immobile granules (R={r_w:.0f} um, "
              f"f={p.boundary_functionalization:g})")

    n_dropped = 0        # V3.2
    for gt in order:
        sp = species[gt]
        r_eq = _sample_radius(sp, rng)

        # Sample shape parameters (per species; same draw order as V2.7)
        if p.shape_enabled:
            ar_ab = max(1.0, rng.normal(_sp_get(sp, 'aspect_ratio_mean', 1.0),
                                        _sp_get(sp, 'aspect_ratio_std', 0.0)))
            ar_c = max(0.3, rng.normal(_sp_get(sp, 'aspect_ratio_c_mean', 1.0),
                                       _sp_get(sp, 'aspect_ratio_c_std', 0.0)))
            n1_val = max(1.5, rng.normal(_sp_get(sp, 'blockiness_mean', 2.0),
                                         _sp_get(sp, 'blockiness_std', 0.0)))
            n2_val = max(1.5, rng.normal(_sp_get(sp, 'blockiness_n2_mean', 2.0),
                                         _sp_get(sp, 'blockiness_n2_std', 0.0)))
            # Compute semi-axes: a, b from ar_ab; c from ar_c
            a0 = r_eq * np.sqrt(ar_ab)
            b0 = r_eq / np.sqrt(ar_ab)
            c0 = r_eq * ar_c
            # Scale to match target volume = (4/3) pi r_eq^3
            target_vol = (4.0/3.0) * np.pi * r_eq**3
            actual_vol = superellipsoid_volume(a0, b0, c0, n1_val, n2_val)
            if actual_vol > 0:
                scale = (target_vol / actual_vol)**(1.0/3.0)
                a_val, b_val, c_val = a0*scale, b0*scale, c0*scale
            else:
                a_val = b_val = c_val = r_eq
            q_val = quat_random(rng)
        else:
            a_val = b_val = c_val = r_eq
            n1_val = n2_val = 2.0
            q_val = np.array([1.0, 0.0, 0.0, 0.0])

        r_bound_val = max(a_val, b_val, c_val)

        placed = False
        periodic = (p.boundary_mode == 'periodic')
        for _ in range(800):
            if periodic:
                cx = rng.uniform(0, p.Lx)
                cy = rng.uniform(0, p.Ly)
                cz = rng.uniform(0, p.Lz)
            else:
                cx = rng.uniform(r_bound_val + gap, p.Lx - r_bound_val - gap)
                cy = rng.uniform(r_bound_val + gap, p.Ly - r_bound_val - gap)
                cz = rng.uniform(r_bound_val + gap, H_place - r_bound_val - gap)
                if geom.shape_code == 1:
                    ddx = cx - geom.cx
                    ddy = cy - geom.cy
                    r_in = geom.R_cyl - r_bound_val - gap
                    if ddx * ddx + ddy * ddy > r_in * r_in:
                        continue
            if checker.ok(cx, cy, cz, r_bound_val):
                xs.append(cx); ys.append(cy); zs.append(cz)
                rs.append(r_eq); types.append(gt)
                a_list.append(a_val); b_list.append(b_val); c_list.append(c_val)
                n1_list.append(n1_val); n2_list.append(n2_val)
                quat_list.append(q_val)
                checker.add(cx, cy, cz, r_bound_val)
                placed = True; break
        if not placed:
            n_dropped += 1      # V3.2: counted and reported, not silently skipped

    if n_dropped:
        # V3.2: this used to be a silent `pass`, which is how a 20 % shortfall in
        # a shaped 3D packing went unnoticed. It is a warning, not an error:
        # small dense test systems legitimately lose a few granules to RSA.
        frac = n_dropped / max(1, n_dropped + len(xs))
        print(f"  WARNING: RSA could not place {n_dropped} granule(s) "
              f"({100 * frac:.1f}% of the target); the packing is short by that much.")
        if frac > 0.05:
            print("    The placement is near RSA saturation. Lower the solid fraction, "
                  "or reduce the aspect ratio / blockiness -- a blocky granule reserves "
                  "a much larger bounding sphere than its volume.")

    # Compute cells per granule
    n_cells = []
    A_cell_sphere = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
    # V3.0: cells per granule = coverage-dependent fraction of the f = 1 capacity
    species_list = resolve_species(p)
    for i in range(len(xs)):
        f_i = float(species_list[int(types[i])]['f'])
        is_boundary = str(species_list[int(types[i])].get('name')) == BOUNDARY_SPECIES_NAME
        if is_boundary and not p.boundary_layer_seed_cells:
            n_cells.append(0)
            continue
        if f_i >= p.f_min_adhesion:
            if p.cell_surface_coverage > 0:
                n_full = cells_from_surface_coverage(
                    rs[i], p.cell_diameter, p.cell_height_spread,
                    p.cell_surface_coverage, p.mode)
            else:
                cap = max_cells_on_granule(rs[i], A_cell_sphere, p.cell_coverage)
                n_full = min(p.n_cells_per_granule, cap)
            n_cells.append(seeded_cells(n_full, f_i, p))
        else:
            n_cells.append(0)

    gs = GranuleSystem(
        xs, ys, rs, types, n_cells,
        z=zs,
        species_id=types, species=species, p=p,
        a=a_list if p.shape_enabled else None,
        b=b_list if p.shape_enabled else None,
        c=c_list if p.shape_enabled else None,
        n1=n1_list if p.shape_enabled else None,
        n2=n2_list if p.shape_enabled else None,
        quat=quat_list if p.shape_enabled else None,
        mode="3D")

    # Store target radii and deflate for inflate-and-relax settling
    if alpha_deflate < 1.0 and p.packing_settle_steps > 0:
        gs._target_a = gs.a.copy()
        gs._target_b = gs.b.copy()
        gs._target_c = gs.c.copy()
        gs._target_r = gs.r.copy()
        gs.a[:gs.N] *= alpha_deflate
        gs.b[:gs.N] *= alpha_deflate
        gs.c[:gs.N] *= alpha_deflate
        gs.r[:gs.N] *= alpha_deflate
        gs.r_bound[:gs.N] = np.maximum(np.maximum(gs.a[:gs.N], gs.b[:gs.N]),
                                        gs.c[:gs.N])

    # ── Inflate-and-relax settle: achieve jammed packing ──
    gs._H_place = H_place                 # placement height (gravity consolidation schedule)
    _ov_tol = settle_overlap_tolerance(gs, p)          # V3.5; None keeps the V2.7 rule
    if _ov_tol is not None:
        gs._overlap_tol = _ov_tol
    if p.packing_settle_steps > 0 and gs.N > 1:
        _settle_packing_3d(gs, p)

    # Clean up temporary target attributes
    for attr in ('_target_a', '_target_b', '_target_c', '_target_r', '_H_place', '_overlap_tol'):
        if hasattr(gs, attr):
            delattr(gs, attr)

    # Report packing fractions per species
    if p.shape_enabled:
        vols = np.array([superellipsoid_volume(gs.a[i], gs.b[i], gs.c[i],
                         gs.n1[i], gs.n2[i]) for i in range(gs.N)])
    else:
        vols = (4.0/3.0) * np.pi * gs.r**3
    frac = vols / domain_vol
    _print_species_placed(gs, frac)
    if consolidation_mode(p) == 'gravity' or geom.top_free:
        _print_bed_report(gs, p)
    phi_total = float(np.sum(frac))

    # Warn if packing fraction exceeds estimated RCP
    ar_mean = float(np.mean(np.maximum(gs.a[:gs.N], gs.b[:gs.N]) /
                            np.minimum(gs.a[:gs.N], gs.b[:gs.N])))
    phi_rcp = rcp_fraction_superellipsoid(ar_mean)
    if phi_total > phi_rcp:
        print(f"  WARNING: phi_solid={phi_total:.3f} exceeds estimated RCP={phi_rcp:.3f}.")
        print(f"    Granules will have unavoidable overlaps. Consider reducing phi_solid_target.")

    print(f"  Total cells: {int(sum(gs.n_cells))}")
    _initialize_cells(gs, rng, p)
    relax_packing(gs, p, rng)      # V3.5: no-op unless packing.relax == 'fire'
    gs.pos_unwrap[:] = gs.pos      # the settle is not part of the dynamics: displacement starts at 0
    return gs


# ══════════════════════════════════════════════════════════════════════
# 3D Contact detection (V1.4)
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def find_contact_spheres_3d(xi, yi, zi, ri, xj, yj, zj, rj):
    """
    Sphere-sphere contact in 3D.
    Returns (in_contact, delta, nx, ny, nz, cx, cy, cz, R_eff) or None.
    """
    dx = xj - xi; dy = yj - yi; dz = zj - zi
    d = np.sqrt(dx*dx + dy*dy + dz*dz)
    if d < 1e-12:
        return None
    overlap = ri + rj - d
    if overlap <= 0:
        return None
    nx, ny, nz = dx/d, dy/d, dz/d
    R_eff = ri * rj / (ri + rj)
    cx = xi + (ri - overlap/2) * nx
    cy = yi + (ri - overlap/2) * ny
    cz = zi + (ri - overlap/2) * nz
    return (True, overlap, nx, ny, nz, cx, cy, cz, R_eff)


@njit(cache=True)
def find_contact_superellipsoids_3d_cn(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi,
        xj, yj, zj, aj, bj, cj, n1j, n2j, qj):
    """
    RETIRED (V3.4) -- see :func:`find_contact_superellipses_cn`. Evidence only;
    nothing in the engine calls this. ``find_contact_superellipsoids_3d`` now
    dispatches to the support-function MTD solver.

    Common normal contact detection between two 3D superellipsoids.
    Returns (in_contact, delta, nx, ny, nz, cx, cy, cz, R_eff) or None.
    """
    dx_c = xj - xi; dy_c = yj - yi; dz_c = zj - zi
    d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
    if d_c < 1e-12:
        return None
    ux = dx_c/d_c; uy = dy_c/d_c; uz = dz_c/d_c

    # Initial guess: find parameter on each surface closest to line of centres
    dir_body_i = quat_rotate_inv(qi, np.array([ux, uy, uz]))
    dir_body_j = quat_rotate_inv(qj, np.array([-ux, -uy, -uz]))

    eta_i = np.arctan2(dir_body_i[2],
                       np.sqrt(dir_body_i[0]**2 + dir_body_i[1]**2) + 1e-30)
    omega_i = np.arctan2(dir_body_i[1], dir_body_i[0])
    eta_j = np.arctan2(dir_body_j[2],
                       np.sqrt(dir_body_j[0]**2 + dir_body_j[1]**2) + 1e-30)
    omega_j = np.arctan2(dir_body_j[1], dir_body_j[0])

    _eta_lo = -np.pi/2 + 0.01
    _eta_hi = np.pi/2 - 0.01

    # Newton-Raphson: find common normal
    for _ in range(15):
        # Surface points in body frame
        pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)

        # Transform to world
        pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
        pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

        # Vector from Pi to Pj
        dp = pj_world - pi_world
        dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)
        if dp_mag < 1e-12:
            break
        target = dp / dp_mag

        # Normals in body frame, then to world
        ni_body = superellipsoid_normal(eta_i, omega_i, ai, bi, ci, n1i, n2i)
        nj_body = superellipsoid_normal(eta_j, omega_j, aj, bj, cj, n1j, n2j)
        ni_world = quat_rotate(qi, ni_body)
        nj_world = quat_rotate(qj, nj_body)

        # Error: cross product (normal should align with target)
        err_i = np.cross(ni_world, target)
        err_j = np.cross(nj_world, -target)

        err_i_mag = np.sqrt(err_i[0]**2 + err_i[1]**2 + err_i[2]**2)
        err_j_mag = np.sqrt(err_j[0]**2 + err_j[1]**2 + err_j[2]**2)
        if err_i_mag < 1e-7 and err_j_mag < 1e-7:
            break

        # Damped update of surface parameters
        # Project error onto eta/omega directions (approximate Jacobian)
        eta_i -= 0.3 * (err_i[2] * np.cos(omega_i) - err_i[1] * np.sin(omega_i))
        omega_i -= 0.3 * err_i[0]
        eta_j -= 0.3 * (err_j[2] * np.cos(omega_j) - err_j[1] * np.sin(omega_j))
        omega_j -= 0.3 * err_j[0]

        # Clamp
        eta_i = min(max(eta_i, _eta_lo), _eta_hi)
        eta_j = min(max(eta_j, _eta_lo), _eta_hi)

    # Final geometry
    pi_body = superellipsoid_point(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    pj_body = superellipsoid_point(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    pi_world = quat_rotate(qi, pi_body) + np.array([xi, yi, zi])
    pj_world = quat_rotate(qj, pj_body) + np.array([xj, yj, zj])

    dp = pj_world - pi_world
    dp_mag = np.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2)

    # Check overlap: is Pi inside body j?
    pi_in_j_body = quat_rotate_inv(qj, pi_world - np.array([xj, yj, zj]))
    val_j = superellipsoid_implicit(pi_in_j_body[0], pi_in_j_body[1],
                                    pi_in_j_body[2], aj, bj, cj, n1j, n2j)
    pj_in_i_body = quat_rotate_inv(qi, pj_world - np.array([xi, yi, zi]))
    val_i = superellipsoid_implicit(pj_in_i_body[0], pj_in_i_body[1],
                                    pj_in_i_body[2], ai, bi, ci, n1i, n2i)

    if val_j > 1.0 and val_i > 1.0:
        return None  # no overlap

    delta = dp_mag
    if dp_mag > 1e-12:
        nx, ny, nz = dp[0]/dp_mag, dp[1]/dp_mag, dp[2]/dp_mag
    else:
        nx, ny, nz = ux, uy, uz

    cx = 0.5 * (pi_world[0] + pj_world[0])
    cy = 0.5 * (pi_world[1] + pj_world[1])
    cz_pt = 0.5 * (pi_world[2] + pj_world[2])

    # Effective radius from local curvature
    R1i, R2i = superellipsoid_curvature_radii(eta_i, omega_i, ai, bi, ci, n1i, n2i)
    R1j, R2j = superellipsoid_curvature_radii(eta_j, omega_j, aj, bj, cj, n1j, n2j)
    R_eff_i = np.sqrt(R1i * R2i)
    R_eff_j = np.sqrt(R1j * R2j)
    R_eff = R_eff_i * R_eff_j / (R_eff_i + R_eff_j) if (R_eff_i + R_eff_j) > 0 else 1.0

    return (True, delta, nx, ny, nz, cx, cy, cz_pt, R_eff)


def find_contact_wall_plane_3d(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound,
                               px, py, pz, nx, ny, nz):
    """Contact between a 3D granule and an arbitrary plane (V3.1; V3.4 exact).

    The plane passes through (px, py, pz) with unit normal (nx, ny, nz)
    pointing INTO the container; used for the tangent plane of a cylindrical
    side wall at the granule's azimuth. Returns (penetration, R_local) or None.

    ``ri_bound`` is now unused -- it was the argument of the hardcoded
    ``R_local = 0.5 * ri_bound``. It is kept in the signature because five call
    sites pass it and the value is free at each of them.
    """
    hit, pen, R_local, _cx, _cy, _cz = se3d_wall_plane_core(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi, px, py, pz, nx, ny, nz)
    if not hit:
        return None
    return pen, R_local


def find_contact_wall_plane_3d_cn(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound,
                                  px, py, pz, nx, ny, nz):
    """RETIRED (V3.4): the 20 x 20 plane sampler. Evidence only."""
    n_sample = 20
    eta = np.linspace(-np.pi/2, np.pi/2, n_sample)
    omega = np.linspace(-np.pi, np.pi, n_sample)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e2 = 2.0 / n2i
    e1 = 2.0 / n1i
    bx = ai * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    by = bi * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    bz = ci * _sgnpow(np.sin(E), e2)
    R_mat = quat_to_rotation_matrix(qi)
    body_pts = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    world_pts = body_pts @ R_mat.T + np.array([xi, yi, zi])
    sd = ((world_pts - np.array([px, py, pz])) * np.array([nx, ny, nz])).sum(axis=1)
    worst = float(np.min(sd))
    if worst >= 0.0:
        return None
    return (-worst, ri_bound * 0.5)


def find_contact_wall_3d(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound,
                         wall_pos, wall_axis, wall_sign):
    """
    Contact between a 3D granule and a flat wall (V1.4; V3.4 exact).
    wall_axis: 0=x, 1=y, 2=z.  wall_sign: +1 if granule on positive side.
    Returns (penetration, R_local) or None.

    ``ri_bound`` is unused since V3.4 -- see :func:`find_contact_wall_plane_3d`.
    """
    hit, pen, R_local, _cx, _cy, _cz = se3d_wall_core(
        xi, yi, zi, ai, bi, ci, n1i, n2i, qi, wall_pos, wall_axis, wall_sign)
    if not hit:
        return None
    return pen, R_local


def find_contact_wall_3d_cn(xi, yi, zi, ai, bi, ci, n1i, n2i, qi, ri_bound,
                            wall_pos, wall_axis, wall_sign):
    """RETIRED (V3.4): the 20 x 20 axis-wall sampler. Evidence only."""
    # Sample the superellipsoid surface and find extreme point
    n_sample = 20
    eta = np.linspace(-np.pi/2, np.pi/2, n_sample)
    omega = np.linspace(-np.pi, np.pi, n_sample)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e2 = 2.0 / n2i
    e1 = 2.0 / n1i
    bx = ai * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    by = bi * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    bz = ci * _sgnpow(np.sin(E), e2)

    # Rotate to world frame
    R_mat = quat_to_rotation_matrix(qi)
    body_pts = np.column_stack([bx.ravel(), by.ravel(), bz.ravel()])
    world_pts = body_pts @ R_mat.T + np.array([xi, yi, zi])

    coords = world_pts[:, wall_axis]

    if wall_sign > 0:
        idx = np.argmin(coords)
        pen = wall_pos - coords[idx]
    else:
        idx = np.argmax(coords)
        pen = coords[idx] - wall_pos

    if pen <= 0:
        return None

    # Approximate local curvature radius from bounding sphere
    R_local = ri_bound * 0.5  # rough approximation
    return (pen, R_local)


# ══════════════════════════════════════════════════════════════════════
# 2D-slice mode (V1.4): generate 3D packing, slice at z-midplane
# ══════════════════════════════════════════════════════════════════════

def slice_superellipsoid_z(cx, cy, cz, a, b, c, n1, n2, quat, z_plane):
    """
    Slice a 3D superellipsoid at z = z_plane to produce a 2D cross-section.
    Returns (a_2d, b_2d, n_2d, theta_2d) or None if plane doesn't intersect.
    """
    # Transform z_plane to body frame
    R_mat = quat_to_rotation_matrix(quat)
    delta = np.array([0.0, 0.0, z_plane - cz])
    body_delta = R_mat.T @ delta

    # z coordinate in body frame
    z_body = body_delta[2]
    z_term = abs(z_body / c) ** n2
    if z_term >= 1.0:
        return None  # plane doesn't intersect

    # Cross-section: scale factor
    s = (1.0 - z_term) ** (1.0 / n2)
    a_2d = a * s
    b_2d = b * s
    n_2d = n1  # equatorial blockiness preserved in cross-section

    # Orientation from projected rotation matrix
    theta_2d = np.arctan2(R_mat[1, 0], R_mat[0, 0])

    return a_2d, b_2d, n_2d, theta_2d


def generate_packing_2d_slice(p: Params, seed=42) -> GranuleSystem:
    """
    Generate 3D packing, then slice at z-midplane to create 2D granule system.
    Mimics experimental confocal imaging of a 3D scaffold.
    """
    print("  2D-slice mode: generating 3D packing first...")
    gs_3d = generate_packing_3d(p, seed=seed)

    z_plane = p.Lz / 2.0
    if boundary_geometry(p, '3D').top_free:
        z_plane = 0.5 * bed_surface(gs_3d, p)['bed_height_mean']    # mid-height of the settled bed
        print(f"  Free top: slicing at half the bed height, z = {z_plane:.0f} um")
    xs, ys, rs, types, n_cells = [], [], [], [], []
    a_list, b_list, n_shape_list, theta_list = [], [], [], []

    for i in range(gs_3d.N):
        result = slice_superellipsoid_z(
            gs_3d.x[i], gs_3d.y[i], gs_3d.z[i],
            gs_3d.a[i], gs_3d.b[i], gs_3d.c[i],
            gs_3d.n1[i], gs_3d.n2[i],
            gs_3d.quat[i] if gs_3d.quat is not None else np.array([1,0,0,0]),
            z_plane)
        if result is None:
            continue
        a_2d, b_2d, n_2d, theta_2d = result
        r_eq = np.sqrt(a_2d * b_2d)  # equivalent radius from cross-section
        if r_eq < 5.0:
            continue  # too small slice

        xs.append(gs_3d.x[i])
        ys.append(gs_3d.y[i])
        rs.append(r_eq)
        types.append(int(gs_3d.species_id[i]))
        n_cells.append(int(gs_3d.n_cells[i]))
        a_list.append(a_2d)
        b_list.append(b_2d)
        n_shape_list.append(n_2d)
        theta_list.append(theta_2d)

    print(f"  Sliced: {len(xs)} granules intersect z-midplane "
          f"(of {gs_3d.N} total)")

    if len(xs) == 0:
        raise ValueError("No granules intersect the slice plane")

    gs = GranuleSystem(
        xs, ys, rs, types, n_cells,
        a=a_list, b=b_list, n_shape=n_shape_list, theta=theta_list,
        mode="2D",
        species_id=types, species=gs_3d.species, p=p)

    domain_area = p.Lx * p.Ly
    if p.shape_enabled:
        areas = np.array([superellipse_area(gs.a[i], gs.b[i], gs.n1[i])
                          for i in range(gs.N)])
    else:
        areas = np.pi * gs.r**2
    print("  2D slice packing:")
    _print_species_placed(gs, areas / domain_area)
    _initialize_cells(gs, np.random.default_rng(seed), p)
    gs.pos_unwrap[:] = gs.pos
    return gs


# ══════════════════════════════════════════════════════════════════════
# Per-cell bridge force recording (V1.5)
# ══════════════════════════════════════════════════════════════════════

def _cell_world_pos_2d(gs, ci):
    """Return world (x, y) for cell ci on its host granule (2D)."""
    gi = int(gs.cell_granule_id[ci])
    theta_cell = gs.cell_theta_local[ci]
    bx, by = superellipse_point(theta_cell, gs.a[gi], gs.b[gi], gs.n_shape[gi])
    ct = np.cos(gs.theta[gi])
    st = np.sin(gs.theta[gi])
    return gs.x[gi] + ct * bx - st * by, gs.y[gi] + st * bx + ct * by


def _cell_world_pos_3d(gs, ci):
    """Return world (x, y, z) for cell ci on its host granule (3D)."""
    gi = int(gs.cell_granule_id[ci])
    eta = gs.cell_eta_local[ci]
    omega = gs.cell_omega_local[ci]
    bp = superellipsoid_point(eta, omega, gs.a[gi], gs.b[gi], gs.c[gi],
                              gs.n_shape[gi], gs.n2[gi])
    bp_w = quat_rotate(gs.quat[gi], bp)
    return gs.x[gi] + bp_w[0], gs.y[gi] + bp_w[1], gs.z[gi] + bp_w[2]


def _compute_gap(gs, gi, tgt, p):
    """Surface-to-surface gap between granules gi and tgt (handles periodic BCs)."""
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    if p.boundary_mode == 'periodic':
        dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
    if hasattr(gs, 'is_3d') and gs.is_3d:
        dz = gs.z[tgt] - gs.z[gi]
        if p.boundary_mode == 'periodic':
            dz = dz - p.Lz * np.round(dz / p.Lz)
        d = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        d = np.sqrt(dx**2 + dy**2)
    return max(d - gs.r_bound[gi] - gs.r_bound[tgt], 0.0)


def _angular_distance_to_target_2d(gs, ci, tgt, p):
    """Angular distance from cell's surface position to contact azimuth (2D).

    Returns (angular_distance, target_theta_body) where target_theta_body is
    the body-frame angle on the host granule pointing toward granule tgt.
    """
    gi = int(gs.cell_granule_id[ci])
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    if p.boundary_mode == 'periodic':
        dx, dy = minimum_image_disp(dx, dy, p.Lx, p.Ly)
    # Target direction in body frame of host granule
    target_theta_world = np.arctan2(dy, dx)
    target_theta_body = (target_theta_world - gs.theta[gi]) % (2.0 * np.pi)
    cell_theta = gs.cell_theta_local[ci] % (2.0 * np.pi)
    # Shortest arc
    diff = target_theta_body - cell_theta
    diff = (diff + np.pi) % (2.0 * np.pi) - np.pi  # wrap to [-pi, pi]
    return abs(diff), target_theta_body, diff


def _angular_distance_to_target_3d(gs, ci, tgt, p):
    """Angular distance from cell's surface position to contact azimuth (3D).

    Returns (angular_distance, target_eta, target_omega).
    Uses great-circle distance on the parametric unit sphere.
    """
    gi = int(gs.cell_granule_id[ci])
    dx = gs.x[tgt] - gs.x[gi]
    dy = gs.y[tgt] - gs.y[gi]
    dz = gs.z[tgt] - gs.z[gi]
    if p.boundary_mode == 'periodic':
        dx = dx - p.Lx * np.round(dx / p.Lx)
        dy = dy - p.Ly * np.round(dy / p.Ly)
        dz = dz - p.Lz * np.round(dz / p.Lz)
    # Target direction in body frame
    d_world = np.array([dx, dy, dz])
    d_body = quat_rotate_inv(gs.quat[gi], d_world)
    # Convert to parametric angles
    r_xy = np.sqrt(d_body[0]**2 + d_body[1]**2)
    target_eta = np.arctan2(d_body[2], r_xy)
    target_omega = np.arctan2(d_body[1], d_body[0]) % (2.0 * np.pi)
    # Great-circle distance from cell position
    cell_eta = gs.cell_eta_local[ci]
    cell_omega = gs.cell_omega_local[ci]
    # Unit vectors on sphere
    p_curr = np.array([np.cos(cell_eta) * np.cos(cell_omega),
                       np.cos(cell_eta) * np.sin(cell_omega),
                       np.sin(cell_eta)])
    p_tgt = np.array([np.cos(target_eta) * np.cos(target_omega),
                      np.cos(target_eta) * np.sin(target_omega),
                      np.sin(target_eta)])
    dot = np.clip(np.dot(p_curr, p_tgt), -1.0, 1.0)
    ang_dist = np.arccos(dot)
    return ang_dist, target_eta, target_omega


# ══════════════════════════════════════════════════════════════════════
# Force computation
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# 3D Force computation (V1.4)
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# Post-step overlap resolution (V2.1 — prevents granule pass-through)
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
# Time integration (overdamped: γ dx/dt = F  →  dx = F/γ · dt)
# ══════════════════════════════════════════════════════════════════════

def _sync_unwrapped(gs, p, pos_before):
    """Advance the unwrapped positions by this step's true displacement.

    ``pos_unwrap`` must equal ``pos`` plus an integer number of periodic images,
    so it has to follow every position change of the step — integration, wall
    clipping and overlap resolution — not just the integrated velocity (which
    drifted away from the real positions for granules pressed against a wall
    or pushed apart by the overlap resolution). Periodic wraps are removed
    with the minimum image of the per-step displacement; granules move far
    less than half a box per step.
    """
    N = gs.N
    d = gs.pos[:N] - pos_before
    if p.boundary_mode == 'periodic':
        L = np.array([p.Lx, p.Ly, p.Lz], dtype=float)
        d -= L * np.rint(d / L)
    gs.pos_unwrap[:N] += d


# ══════════════════════════════════════════════════════════════════════
# Container geometry and gravity (V3.1)
# ══════════════════════════════════════════════════════════════════════

from collections import namedtuple

BoundaryGeom = namedtuple('BoundaryGeom', 'dim shape_code R_cyl cx cy top_free f_wall')


def boundary_geometry(p, mode=None):
    """Scalar description of the container (kernel-friendly).

    shape_code 0 = box, 1 = cylinder (axis z, radius min(Lx,Ly)/2, centre
    (Lx/2, Ly/2); 3D only). top_free: no lid (z is up in 3D, y is up in 2D).
    Legacy Params (box / wall) give shape_code 0, top_free False.
    """
    m = mode if mode is not None else p.mode
    dim = 3 if m in ('3D', '2D-slice') else 2
    shape = getattr(p, 'boundary_shape', 'box')
    top = getattr(p, 'boundary_top', 'wall')
    shape_code = 1 if (shape == 'cylinder' and dim == 3) else 0
    return BoundaryGeom(dim, shape_code, 0.5 * float(min(p.Lx, p.Ly)), 0.5 * float(p.Lx),
                        0.5 * float(p.Ly), bool(top == 'free'),
                        float(getattr(p, 'boundary_functionalization', 0.0)))


def container_voxel_mask(p, shape, snap=None, mode=None):
    """True where a voxel is inside the container and below the free surface (V3.3).

    Promoted from ``viz2.void_percolation.container_mask`` so the compiled
    metrics path can reach it -- ``gels.kernels`` cannot import from ``viz2``.
    Behaviour is unchanged.

    Without this, everything outside a cylindrical wall or above an open top
    reads as void: one cluster touching every face, so percolation is trivially
    true and the cluster-size distribution is meaningless. Returns ``None`` for
    a closed box, where the whole grid is the container and the V3.0 behaviour
    is exactly right -- callers must treat ``None`` as "no mask", not as "empty".

    ``shape`` is the grid shape (axis 0 = x, the engine convention). ``snap``,
    when given, supplies ``r`` and ``z``/``y`` so a free surface is cut at the
    99th percentile of ``z + r`` rather than at the container height.
    """
    geom = boundary_geometry(p, mode)
    if geom.shape_code == 0 and not geom.top_free:
        return None
    ndim = len(shape)
    axes = [(np.arange(shape[k]) + 0.5) * (L / shape[k])
            for k, L in enumerate((p.Lx, p.Ly, p.Lz)[:ndim])]
    grids = np.meshgrid(*axes, indexing='ij')
    mask = np.ones(tuple(shape), dtype=bool)
    if geom.shape_code == 1 and ndim == 3:
        mask &= ((grids[0] - geom.cx) ** 2 + (grids[1] - geom.cy) ** 2
                 <= geom.R_cyl ** 2)
    if geom.top_free:
        up = grids[2] if ndim == 3 else grids[1]
        h = None
        if snap is not None:
            r = np.asarray(snap.get('r', []), dtype=float)
            if r.size:
                z = np.asarray(snap.get('z' if ndim == 3 else 'y'), dtype=float)
                h = float(np.percentile(z + r, 99))
        if h is None:
            h = float(p.Lz if ndim == 3 else p.Ly)
        mask &= up <= h
    return mask


def domain_volume(p, mode=None):
    """Container volume (um^3; area in 2D). Box: the V2.7 expression verbatim."""
    geom = boundary_geometry(p, mode)
    if geom.shape_code == 1:
        return np.pi * geom.R_cyl ** 2 * p.Lz
    return p.Lx * p.Ly * p.Lz if geom.dim == 3 else p.Lx * p.Ly


def domain_base_area(p, mode=None):
    """Footprint of the container: pi R^2 (cylinder), Lx*Ly (box), Lx (2D dish)."""
    geom = boundary_geometry(p, mode)
    if geom.shape_code == 1:
        return np.pi * geom.R_cyl ** 2
    return p.Lx * p.Ly if geom.dim == 3 else p.Lx


def consolidation_mode(p):
    """Packer consolidation: 'auto' -> gravity when gravity is on, else none."""
    mode = getattr(p, 'packing_consolidation', 'centre')
    if mode == 'auto':
        return 'gravity' if getattr(p, 'gravity_enabled', False) else 'none'
    return mode


# Consolidation body force as a fraction of the settle's overlap tolerance:
# the deepest (bottom-layer) contact carries LOAD_FRAC x tol at full load.
CONSOLIDATION_LOAD_FRAC = 0.5


def consolidation_gravity_schedule(mean_r_target, H_place, k_rep=5.0):
    """Body-force magnitudes of the packer's gravity consolidation (settle units).

    The force is bounded by what the settle can resolve, not by the granule
    size: with ``n_layers`` granule layers the bottom contact of a column
    carries n_layers*g, so an equilibrium overlap of ``frac * tol``
    (tol = 0.05 mean_r) needs g = k_rep (frac tol)^1.5 / n_layers. ``g_hi``
    uses frac = CONSOLIDATION_LOAD_FRAC, ``g_lo`` a quarter of that force.

    Measured on a 400x400x600 um free-top box and cylinder (r = 20 um,
    N = 558-716): between frac 0.3 and 1.0 the settled bed is the same to
    within 1 um of height and 0.001 of phi_bed — the inflation does the
    compaction and the body force only reseats rattlers and flattens the
    surface. The V3.1 predecessor of this schedule used ~0.5 mean_r instead
    of ~0.5 tol (a 67x larger force) and left residual overlaps at 1.5-2 um.
    """
    n_layers = max(1.0, float(H_place) / (2.0 * float(mean_r_target)))
    tol = 0.05 * float(mean_r_target)
    g_hi = k_rep * (CONSOLIDATION_LOAD_FRAC * tol) ** 1.5 / n_layers
    g_lo = 0.25 * g_hi
    return g_hi, g_lo


class GravityConsolidation:
    """Post-relax body-force schedule of the packer's gravity consolidation.

    Shared by the compiled and the reference settle so both make identical
    decisions. The granules are inflated WITHOUT gravity (a body force on the
    deflated granules drops them into a dense coplanar layer on the floor
    that stays laterally jammed once inflated); the sedimentation happens
    here in two phases:

      A  settle:  g = g_hi while the bed still moves (largest move >= 0.05 mean_r)
                  and fewer than ``hold_max`` steps have elapsed;
      B  unload:  g = max(g_hi 0.7^((k - k0)//20), g_lo).

    ``converged`` once the bed is still (largest move < 0.02 mean_r) at g_lo
    and either the overlap tolerance is met or the bed has stopped moving
    altogether for ``stall_steps`` steps (a single granule wedged slightly
    above tolerance must not spin the loop to its cap).
    """

    def __init__(self, mean_r_target, H_place, k_rep=5.0, hold_max=600, stall_steps=100):
        self.g_hi, self.g_lo = consolidation_gravity_schedule(mean_r_target, H_place, k_rep)
        self.mean_r = float(mean_r_target)
        self.hold_max = int(hold_max)
        self.stall_steps = int(stall_steps)
        self.k0 = None
        self.n_stalled = 0

    def g(self, k, max_disp):
        if self.k0 is None:
            if k >= self.hold_max or max_disp < 0.05 * self.mean_r:
                self.k0 = k
            else:
                return self.g_hi
        return max(self.g_hi * 0.7 ** ((k - self.k0) // 20), self.g_lo)

    def converged(self, g_k, max_overlap, overlap_tol, max_disp):
        if self.k0 is None or g_k != self.g_lo:
            return False
        still = max_disp < 0.02 * self.mean_r
        self.n_stalled = self.n_stalled + 1 if max_disp < 0.002 * self.mean_r else 0
        return still and (max_overlap < overlap_tol or self.n_stalled >= self.stall_steps)


def consolidation_weights(target_r, mean_r_target, dim):
    """Per-granule body-force weights: volume (3D) / area (2D) relative to the mean granule."""
    return (np.asarray(target_r, dtype=np.float64) / float(mean_r_target)) ** (3 if dim == 3 else 2)


def shape_bound_factor(p, mode):
    """E[(r_bound / r_eq)^dim] for the species' MEAN shape (V3.2).

    RSA tests BOUNDING SPHERES while the deflation is computed from the
    equal-volume radius, so with shapes on the effective RSA fraction is higher
    than intended by this factor -- 2.39x in 3D at AR 1.8 / n 3.5, which pushes
    RSA past its ~0.38 saturation and makes it silently drop granules. Computed
    from the species moments only, so it consumes no random draws and the packing
    stream is untouched. Returns 1.0 whenever the shape path is off.
    """
    if not (getattr(p, 'packing_shape_contact', False) and getattr(p, 'shape_enabled', False)):
        return 1.0
    dim = 3 if mode in ('3D', '2D-slice') else 2
    vols, facs = [], []
    for sp in (getattr(p, 'species', None) or []):
        vf = float(sp.get('volume_fraction', 0.0) or 0.0)
        ar = max(1.0, float(sp.get('aspect_ratio_mean', 1.0) or 1.0))
        nb = max(1.5, float(sp.get('blockiness_mean', 2.0) or 2.0))
        n2 = max(1.5, float(sp.get('blockiness_n2_mean', nb) or nb))
        arc = max(1e-9, float(sp.get('aspect_ratio_c_mean', 1.0) or 1.0))
        if dim == 3:
            a0, b0, c0 = ar, 1.0, arc
            sc = ((4.0 / 3.0) * np.pi / superellipsoid_volume(a0, b0, c0, nb, n2)) ** (1.0 / 3.0)
            rb = float(shape_bound_radius(a0 * sc, b0 * sc, c0 * sc, nb, n2))
        else:
            a0, b0 = ar, 1.0
            sc = np.sqrt(np.pi / superellipse_area(a0, b0, nb))
            rb = float(shape_bound_radius(a0 * sc, b0 * sc, None, nb, None))
        vols.append(vf)
        facs.append(rb ** dim)
    if not vols or sum(vols) <= 0:
        return 1.0
    return float(sum(v * f for v, f in zip(vols, facs)) / sum(vols))


def check_bed_fraction(p, mode, phi_bed_request):
    """Warn when the requested bed fraction is not something sedimentation gives (V3.2).

    The packer does NOT discover a bed density: it places granules inside a
    height ``V_solid / (request * A_base)`` and inflates them there, so gravity
    only reseats rattlers. Measured with a FIXED granule count in a fixed
    container, the achieved fraction tracks the request at 0.97-1.00x all the
    way from 0.60 to 0.85, and the coordination number rises with it from 2.5 to
    4.1. So ``granules.bed_solid_fraction`` is the ANSWER, not a guess at one,
    and it has to be chosen physically:

      * below random loose packing the bed is under-coordinated -- 2D discs are
        isostatic at Z = 3, so a request of 0.60 (Z = 2.5) is a loose
        arrangement held up by the placement, not a mechanically jammed bed;
      * above random close packing the settle cannot relieve the overlaps it is
        asked to create and the post-relax runs out of steps.

    Reference values: 3D spheres RLP 0.55-0.60, RCP 0.64; 2D discs RLP ~0.78,
    RCP ~0.84, hexagonal 0.9069. Blocky superellipsoids raise the upper end.
    Leave headroom: a bed that STARTS near close packing has nowhere to compact
    to, which is the whole point of the simulation.
    """
    dim = 3 if mode in ('3D', '2D-slice') else 2
    lo, hi = (0.55, 0.64) if dim == 3 else (0.78, 0.84)
    if getattr(p, 'shape_enabled', False) and getattr(p, 'packing_shape_contact', False):
        hi *= 1.08          # blocky shapes jam a little higher
    req = float(phi_bed_request)
    if req > hi:
        print(f"  WARNING: bed_solid_fraction {req:.2f} is above random CLOSE packing "
              f"(~{hi:.2f} in {dim}D). The packer will try to place the granules at that "
              f"density anyway; expect the post-relax to run out of steps, and note that a "
              f"bed starting this dense has almost no room left to compact.")
    elif req < lo:
        print(f"  NOTE: bed_solid_fraction {req:.2f} is below random LOOSE packing "
              f"(~{lo:.2f} in {dim}D), so the bed will be under-coordinated rather than "
              f"jammed -- deliberate if you want maximum compaction headroom.")


def packing_placement(p, mode, phi_target, r_max):
    """RSA placement height and deflation factor (V3.1).

    Legacy consolidation (centre / none) or periodic boxes: the whole
    container, alpha = (phi_safe / phi_target)^(1/dim) as in V2.7. Gravity
    consolidation: granules are placed inside the expected settled bed height
    H_place = min(L_up, max(H_bed, 4 r_max)), H_bed = V_solid / (phi_bed A_base),
    with the deflation computed from the placement volume, so the inflation
    does the compaction and the body force only reseats rattlers and flattens
    the surface. Returns (H_place, phi_place, alpha_deflate, L_up).
    """
    dim = 3 if mode in ('3D', '2D-slice') else 2
    L_up = float(p.Lz if dim == 3 else p.Ly)
    phi_place = phi_target
    H_place = L_up
    if (consolidation_mode(p) == 'gravity' and p.boundary_mode != 'periodic'
            and phi_target > 0.0):
        A_base = domain_base_area(p, mode)
        V_solid = phi_target * domain_volume(p, mode)
        H_bed = V_solid / (float(p.bed_phi_assumed) * A_base)
        H_place = min(L_up, max(H_bed, 4.0 * float(r_max)))
        phi_place = V_solid / (A_base * H_place)
    if phi_place > 0.05 and p.packing_settle_steps > 0:
        # V3.2: RSA tests bounding spheres, so a shaped granule reserves
        # (r_bound/r_eq)^dim more room than its volume. Without this the
        # placement is past RSA saturation and granules are silently dropped.
        phi_rsa = phi_place * shape_bound_factor(p, mode)
        alpha = (p.packing_inflate_phi_safe / max(phi_rsa, 0.01)) ** (1.0 / dim)
        alpha = float(np.clip(alpha, 0.3, 1.0))
    else:
        alpha = 1.0
    return H_place, phi_place, alpha, L_up


def bed_solid_volume(gs):
    """Solid volume (um^3; area in 2D) of the mobile granules from their equivalent radii."""
    N = gs.N
    fixed = getattr(gs, 'fixed', None)
    mob = np.ones(N, dtype=bool) if fixed is None else ~np.asarray(fixed[:N], dtype=bool)
    r = np.asarray(gs.r[:N])[mob]
    if gs.is_3d:
        return float(np.sum((4.0 / 3.0) * np.pi * r ** 3))
    return float(np.sum(np.pi * r ** 2))


def pair_bridge_weight(gs, p, pair_i, pair_j):
    """Mature bridging cells spanning each candidate pair (V3.2).

    Cells that bridge ACROSS a contact bind the two granules together and stop
    them sliding past each other -- the "cells binding and tightening the
    contact" the model was missing. The normal direction needs nothing new: the
    bridge already pulls the pair together with up to the stall force, and the
    contact it deepens already grows its own contact radius and hence its shear
    cap. What is absent is the collar's resistance to SHEAR, which is what locks
    a compacted structure in place.

    Read from the persistent per-cell arrays (``cell_granule_id`` and
    ``cell_bridge_target``), not from this step's bridging pass, because the
    contact pass runs first -- so these are the bridges committed before the
    step, which is the physically right set. Weighted by bridge maturity, the
    same ramp the bridge force uses.
    """
    M = int(pair_i.shape[0])
    out = np.zeros(M)
    C = int(gs.total_cells)
    if C == 0 or M == 0:
        return out
    tgt = np.asarray(gs.cell_bridge_target[:C])
    live = tgt >= 0
    if not np.any(live):
        return out
    g = np.asarray(gs.cell_granule_id[:C])[live].astype(np.int64)
    t = tgt[live].astype(np.int64)
    w = np.minimum(1.0, np.asarray(gs.cell_bridge_age[:C])[live]
                   / max(0.1, float(p.bridge_formation_time)))
    N = max(int(gs.N), 1)
    key = np.minimum(g, t) * N + np.maximum(g, t)      # undirected: either sense spans the pair
    uk, inv = np.unique(key, return_inverse=True)
    tot = np.zeros(uk.size)
    np.add.at(tot, inv, w)
    pi = np.asarray(pair_i, dtype=np.int64)
    pj = np.asarray(pair_j, dtype=np.int64)
    pk = np.minimum(pi, pj) * N + np.maximum(pi, pj)
    idx = np.clip(np.searchsorted(uk, pk), 0, uk.size - 1)
    hit = uk[idx] == pk
    out[hit] = tot[idx[hit]]
    return out


def bridge_weight_map(gs, p):
    """``{(lo, hi): mature bridging-cell weight}`` for the reference pair loops (V3.2).

    Dict form of :func:`pair_bridge_weight`, because the reference iterates pairs
    rather than indexing a precomputed per-pair array. Empty (and free) when the
    feature is off.
    """
    out = {}
    C = int(gs.total_cells)
    if C == 0 or float(getattr(p, 'cell_contact_adhesion', 0.0)) <= 0.0:
        return out
    tgt = np.asarray(gs.cell_bridge_target[:C])
    live = np.nonzero(tgt >= 0)[0]
    if live.size == 0:
        return out
    gid = np.asarray(gs.cell_granule_id[:C])
    age = np.asarray(gs.cell_bridge_age[:C])
    ramp = max(0.1, float(p.bridge_formation_time))
    for c in live:
        a, b = int(gid[c]), int(tgt[c])
        key = (a, b) if a < b else (b, a)
        out[key] = out.get(key, 0.0) + min(1.0, float(age[c]) / ramp)
    return out


def contact_stiffness_per_granule(gs, contacts):
    """Sum of dF/d(delta) over each granule's contacts, in nN/um (V3.2).

    For a Hertz/JKR contact ``F = (4/3) E_s sqrt(R*) delta^{3/2}``, so
    ``dF/d(delta) = 2 E_s a`` with ``a`` the contact radius -- both already on
    the contact record, so this costs one bincount and no kernel change.

    Used by the semi-implicit step: explicit overdamped Euler is stable only for
    ``k < gamma/dt`` (2 nN/um at r = 20 um), and at 10 kPa that bound is crossed
    at delta ~ 0.01 um, i.e. at every load-bearing contact. Damping the step by
    ``dt * k_i`` is unconditionally stable in the diagonal part and leaves the
    FIXED POINT untouched (at equilibrium F = 0, so the step is zero for any
    damping) -- it changes the transient, not the physics.

    V3.6: the WALL contacts are added too. They are applied inline in both
    gathers and never reach the contact list, so before V3.6 a granule held only
    by a wall was damped by ``drag_scale * r`` alone and rang about its contact
    equilibrium. That was invisible until V3.5 changed ``boundary.wall_clamp``
    to ``contact``, because until then the clip held every floor granule 0.5 um
    clear of the wall and no granule was ever in wall contact at all.
    """
    N = gs.N
    k = np.zeros(N)
    wk = getattr(gs, 'wall_stiffness', None)    # V3.6; None on a restored gs
    if wk is not None and len(wk) >= N:
        k += np.asarray(wk[:N], dtype=float)
    if contacts is None or len(contacts) == 0:
        return k
    col = getattr(contacts, 'column', None)
    if col is not None:
        i = np.asarray(col('i'), dtype=np.int64)
        j = np.asarray(col('j'), dtype=np.int64)
        a = np.asarray(col('a_contact'), dtype=float)
        E_s = np.asarray(col('E_star'), dtype=float) * 1e-3     # Pa -> nN/um^2
    else:                                   # reference list-of-dicts path
        i = np.array([c['i'] for c in contacts], dtype=np.int64)
        j = np.array([c['j'] for c in contacts], dtype=np.int64)
        a = np.array([c.get('a_contact', 0.0) for c in contacts], dtype=float)
        E_s = np.array([c.get('E_star', 0.0) for c in contacts], dtype=float) * 1e-3
    kp = 2.0 * E_s * a
    k += np.bincount(i, kp, minlength=N)[:N]
    k += np.bincount(j, kp, minlength=N)[:N]
    return k


# ── V3.6: the substep controller ──────────────────────────────────────────
#
# `contact_semi_implicit` takes the step vel = F/(gamma + dt k). At the fixed
# point F = 0, so the equilibrium is exact for any dt -- which is what makes the
# scheme unconditionally stable and why V3.2 adopted it. But the RATE it gives
# is F/(gamma + dt k) where the overdamped rate is F/gamma, so with
#
#     S = dt k / gamma
#
# every granule moves (1 + S)x too slowly. A packing or a sedimentation run ends
# at a fixed point and does not care. A cell-driven compaction run's entire
# answer is a rate, and it cares a great deal: measured on a 2D bed at the
# shipped dt = 0.5 h, S averages 41 and reaches 88, and `disp_func` over 24 h is
# a factor of 4 below its dt -> 0 value (and still moving at dt = 0.005 h).
#
# S is free: `k` is already computed every step for the semi-implicit
# denominator. `k/gamma` has units of 1/h and does not depend on dt, so the
# controller reads it from the step just taken and sizes the next one.
#
# NOTE this is deliberately NOT the energy. Where dt matters is exactly where
# the V3.5 energy audit is invalid -- with cells seeded, monotone descent is
# false because bridges are actuators, so `energy_delta` cannot separate "dt too
# large" from "the cells did work". In the passive case, where the audit IS
# exact, dt = 0.5 h is already accurate to 0.05 % on the bulk observables. The
# energy monitor stays the verifier; S is the controller.
OUTLIER_MIN_CLASS = 8       # a coordination class smaller than this is not a population
OUTLIER_MAD_SCALE = 1.4826  # MAD -> sigma for a normal


def stiffness_rate(gs, p, contacts):
    """k_i / gamma_i in 1/h -- the inverse of each granule's stiff timescale.

    Free of dt by construction, so it can be measured on one step and used to
    size the next.
    """
    N = gs.N
    if N == 0:
        return np.zeros(0)
    gamma = p.drag_scale * gs.r[:N]
    return contact_stiffness_per_granule(gs, contacts) / np.maximum(gamma, 1e-12)


def substep_mode(gs, p):
    """Resolve `dynamics.substep` for this system: 'off' or 'always' (V3.6).

    `auto` substeps when the run is DRIVEN by something that is not the contact
    law -- in practice, when cells are seeded, because bridges are actuators and
    the bed therefore never reaches a fixed point, so the answer the run reports
    is a rate.

    It declines otherwise, and that is a measurement rather than a guess. A
    packing, a sedimentation or a relaxation ends AT a fixed point, and the
    semi-implicit step puts the fixed point in exactly the right place for any
    dt -- the rate it takes to get there is wrong but nobody reads it. Measured:
    a 2D gravity bed run to 6 h at dt = 0.5 h has a bed height 0.22 um out of
    444 from the same bed at dt = 0.005 h (0.05 %) and a phi_bed 0.0003 from it,
    whether or not the packer handed it over pre-loaded. Substepping that would
    be 256x the cost for a fifth of a micron.

    `always` is the unconditional form, for a passive run whose PATH is the
    subject rather than its endpoint.
    """
    mode = str(getattr(p, 'dynamics_substep', 'off'))
    if mode in ('off', 'always'):
        return mode
    if mode != 'auto':
        return 'off'
    off = getattr(gs, 'cell_offset', None)
    n_cells = int(off[gs.N]) if off is not None and len(off) > gs.N else 0
    return 'always' if n_cells > 0 else 'off'


def substep_count(gs, p):
    """How many mechanical substeps this outer step should be advanced in (V3.6).

    Read from the stiffness rate the LAST step recorded, so it costs no force
    evaluation, and seeded from the t = 0 evaluation so the first interval is
    already sized. There is deliberately no ramp: the rate is a **p95**, not a
    max, so one transient contact cannot move it and there is nothing for a
    growth clamp to protect against. `dynamics_substep_max` is the only ceiling.

    When the budget `dynamics_substep_max` binds before the target is met, the
    run is NOT converged in dt and `substep_budget_bound` says so -- the same
    contract as V3.5's warning that the settle ended on its step cap rather than
    on its tolerance.
    """
    if substep_mode(gs, p) != 'always':
        gs.substep_budget_bound = False
        return 1
    rate = float(getattr(gs, 'stiffness_rate_p95', 0.0))
    if not np.isfinite(rate) or rate <= 0.0:
        gs.substep_budget_bound = False
        return 1
    target = max(float(getattr(p, 'dynamics_substep_target', 0.2)), 1e-6)
    n_max = max(int(getattr(p, 'dynamics_substep_max', 64)), 1)
    want = int(np.ceil(p.dt * rate / target))
    gs.substep_rate_used = rate     # what the decision was made on, for the console
    gs.substep_budget_bound = bool(want > n_max)
    return int(min(max(1, want), n_max))


def coordination_class(gs, contacts):
    """Per-granule contact count, bucketed -- "granules in the same condition".

    A rattler really does move faster than a jammed granule, and comparing the
    two would be comparing different physics. Buckets are the raw count capped
    at 6, which separates free / singly-held / weakly-held / jammed without
    inventing thresholds.
    """
    N = gs.N
    z = np.zeros(N, dtype=np.int64)
    if contacts is not None and len(contacts):
        col = getattr(contacts, 'column', None)
        if col is not None:
            i = np.asarray(col('i'), dtype=np.int64)
            j = np.asarray(col('j'), dtype=np.int64)
        else:
            i = np.array([c['i'] for c in contacts], dtype=np.int64)
            j = np.array([c['j'] for c in contacts], dtype=np.int64)
        z += np.bincount(i, minlength=N)[:N]
        z += np.bincount(j, minlength=N)[:N]
    return np.minimum(z, 6)


def population_speed_cap(gs, p, speed, contacts):
    """A speed ceiling from the population a granule belongs to (V3.6).

    ``v_max`` is an absolute ceiling and therefore has to be set for the fastest
    thing the model can produce; at the shipped 20 um/h against 180-440 nN
    bridge forces on a drag of ~1 nN.h/um it is a FORCE ceiling for most of the
    bed, discarding magnitude information wholesale (V3.2's
    ``frac_velocity_clipped`` is what makes that visible).

    This is the relative rail instead: we do not expect a granule to differ much
    from granules in its own condition, so a granule moving many times the
    median of its own coordination class is a numerical outlier -- the single
    wedged granule V3.5 measured at 132x the load while the second-worst read
    0.97x -- not a physical one.

    Returns the per-granule ceiling, or None when the rail is off or the
    population is too small to have a median worth trusting.
    """
    K = float(getattr(p, 'dynamics_outlier_speed', 0.0))
    if K <= 0.0 or gs.N == 0:
        return None
    klass = coordination_class(gs, contacts)
    fixed = getattr(gs, 'fixed', None)
    mob = (np.ones(gs.N, dtype=bool) if fixed is None
           else ~np.asarray(fixed[:gs.N], dtype=bool))
    cap = np.full(gs.N, np.inf)
    for c in np.unique(klass[mob]) if mob.any() else ():
        m = mob & (klass == c)
        n = int(m.sum())
        if n < OUTLIER_MIN_CLASS:
            continue                      # not a population; leave it to v_max
        v = speed[m]
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * OUTLIER_MAD_SCALE
        # `med` floors the spread so a class that is uniformly slow (mad ~ 0)
        # does not get a cap of zero, and a class at rest is never clipped.
        cap[m] = med + K * max(mad, med)
    return cap


def _speed_rails(gs, p, speed, contacts, over):
    """The per-granule velocity scale factor, and the two diagnostics (V3.6).

    Two ceilings, reported separately because they mean different things:

    * ``v_max`` -- the absolute rail. Unchanged, and still what catches a
      genuine blowup. `frac_velocity_clipped` counts it.
    * the population rail -- `population_speed_cap`. `frac_outlier_clipped`
      counts granules the population rail caught that ``v_max`` did not, so the
      two never double-count.

    With the population rail off this is exactly the V3.5 expression: the scale
    is 1.0 where the granule is under the cap and ``v_max / speed`` where it is
    not.
    """
    gs.frac_velocity_clipped = float(np.mean(over)) if over.size else 0.0
    ceil_v = np.full(speed.shape, float(p.v_max))
    ocap = population_speed_cap(gs, p, speed, contacts)
    if ocap is not None:
        ceil_v = np.minimum(ceil_v, ocap)
    hit = speed > ceil_v
    gs.frac_outlier_clipped = float(np.mean(hit & ~over)) if hit.size else 0.0
    return np.where(hit, ceil_v / np.maximum(speed, 1e-300), 1.0)


def projection_metrics(gs):
    """How much of the dynamics is the numerical rail rather than the contact law (V3.2).

    Two things stand in for a contact solver in GELS: the overlap projection and
    the velocity cap. Neither was observable, so a run in which they dominated
    looked exactly like one in which the contact law did. These make it visible.

    ``overlap_clip_fraction`` well above zero means the geometric rail, not
    Hertz/JKR, is setting how far the bed compacts -- see ``elastic_overlap_frac``.
    ``frac_velocity_clipped`` above zero means granule speeds are saturated, so
    force MAGNITUDE information is being discarded (the cap is an effective force
    ceiling of v_max * drag_scale * r, 20-60 nN against 180 nN bridges).
    """
    n = int(getattr(gs, 'n_overlap_clipped', 0))
    tot = int(getattr(gs, 'n_overlap_pairs', 0))
    return {
        'n_overlap_clipped': n,
        'overlap_clip_fraction': float(n) / tot if tot else 0.0,
        'frac_velocity_clipped': float(getattr(gs, 'frac_velocity_clipped', 0.0)),
    }


def overlap_solid_volume(gs, mobile_only=True):
    """Total double-counted volume (area in 2D) where granules interpenetrate (V3.2).

    ``phi_solid_true`` and ``phi_bed`` sum (4/3)pi r^3 over NOMINAL radii, so
    every contact lens is counted twice. At the ~3 % overlaps of a rigid bed
    that is 0.1-0.2 % of the solid and can be ignored; at the 10-15 % overlaps a
    soft (10-50 kPa) bed reaches under cell traction it is 1.1-4.1 % depending
    on coordination number -- and that is precisely the regime V3.2 is about,
    so the net value is reported alongside the raw one.

    Vectorised twin of ``overlap_lens_area`` / ``overlap_lens_volume``. Like
    ``_resolve_overlaps``, it uses a plain (non-periodic) neighbour query, so
    under periodic boundaries it misses pairs across the wrap.
    """
    N = gs.N
    if N < 2:
        return 0.0
    fixed = getattr(gs, 'fixed', None)
    keep = (np.ones(N, dtype=bool) if (fixed is None or not mobile_only)
            else ~np.asarray(fixed[:N], dtype=bool))
    idx = np.nonzero(keep)[0]
    if idx.size < 2:
        return 0.0
    r = np.asarray(gs.r[:N], dtype=float)[idx]
    dim = 3 if gs.is_3d else 2
    pos = np.asarray(gs.pos[:N, :dim], dtype=float)[idx]
    tree = cKDTree(pos)
    pairs = tree.query_pairs(2.0 * float(r.max()), output_type='ndarray')
    if pairs.size == 0:
        return 0.0
    i, j = pairs[:, 0], pairs[:, 1]
    d = np.linalg.norm(pos[i] - pos[j], axis=1)
    R1, R2 = r[i], r[j]
    hit = d < (R1 + R2)
    if not np.any(hit):
        return 0.0
    d, R1, R2 = d[hit], R1[hit], R2[hit]
    d = np.maximum(d, 1e-12)
    Rmin = np.minimum(R1, R2)
    inside = d <= np.abs(R1 - R2)          # one granule fully inside the other
    if dim == 3:
        sgap = R1 + R2 - d
        lens = np.pi * sgap ** 2 * (d * d + 2.0 * d * (R1 + R2)
                                    - 3.0 * (R1 - R2) ** 2) / (12.0 * d)
        full = (4.0 / 3.0) * np.pi * Rmin ** 3
    else:
        c1 = np.clip((d * d + R1 * R1 - R2 * R2) / (2.0 * d * R1), -1.0, 1.0)
        c2 = np.clip((d * d + R2 * R2 - R1 * R1) / (2.0 * d * R2), -1.0, 1.0)
        arg = (-d + R1 + R2) * (d + R1 - R2) * (d - R1 + R2) * (d + R1 + R2)
        lens = (R1 * R1 * np.arccos(c1) + R2 * R2 * np.arccos(c2)
                - 0.5 * np.sqrt(np.maximum(0.0, arg)))
        full = np.pi * Rmin ** 2
    return float(np.sum(np.where(inside, full, np.maximum(lens, 0.0))))


def granule_reach(gs, p, N, ux, uy, uz=None):
    """How far each granule reaches along a world direction (V3.2).

    Falls back to ``r_bound`` unless the shape-aware dynamics is on. Used
    wherever the code needs "how much room does this granule take toward that
    wall" -- the floor and wall clamps and the bed-surface column tops. With a
    blocky granule ``r_bound`` is the CORNER radius, so using it holds a
    flat-lying fragment well off the floor and inflates the measured bed height.
    """
    rb = np.asarray(gs.r_bound[:N])
    if not shape_dynamics_on(p, gs):
        return rb
    ux = np.broadcast_to(np.asarray(ux, dtype=float), (N,))
    uy = np.broadcast_to(np.asarray(uy, dtype=float), (N,))
    if gs.is_3d:
        uz = np.broadcast_to(np.asarray(uz if uz is not None else 0.0, dtype=float), (N,))
        return lambda_world(gs.a[:N], gs.b[:N], gs.c[:N], gs.n1[:N], gs.n2[:N],
                            gs.theta[:N], gs.quat[:N], ux, uy, uz, True)
    return lambda_world(gs.a[:N], gs.b[:N], None, gs.n1[:N], None,
                        gs.theta[:N], None, ux, uy, None, False)


def _reach_up(gs, p, N):
    """Upward reach (the bed-surface column top)."""
    if gs.is_3d:
        return granule_reach(gs, p, N, 0.0, 0.0, 1.0)
    return granule_reach(gs, p, N, 0.0, 1.0)


def bed_surface(gs, p, col_width=None):
    """Column-top description of a granule bed (free-surface containers).

    The base is divided into columns about one granule diameter wide; each
    column's top is max(z_i + r_i) over the mobile granules in it (y in 2D).
    Returns dict(bed_height_mean, bed_height_p95, bed_envelope_volume,
    n_columns, col_tops); unoccupied columns (and, for a cylinder, columns
    whose centre lies outside the wall) are excluded.
    """
    N = gs.N
    geom = boundary_geometry(p, gs.mode if gs.mode in ('2D', '3D') else None)
    fixed = getattr(gs, 'fixed', None)
    mob = np.ones(N, dtype=bool) if fixed is None else ~np.asarray(fixed[:N], dtype=bool)
    empty = dict(bed_height_mean=0.0, bed_height_p95=0.0, bed_envelope_volume=0.0,
                 n_columns=0, col_tops=np.zeros(0))
    if N == 0 or not np.any(mob):
        return empty
    rb = np.asarray(gs.r_bound[:N])
    # V3.2: a column top is how far the granule actually reaches UPWARD. With a
    # blocky granule r_bound is the CORNER radius, so using it inflates the bed
    # height (and deflates phi_bed) by the corner-to-face difference.
    rb_up = _reach_up(gs, p, N)
    w = float(col_width) if col_width else 2.0 * float(np.max(rb[mob]))
    if gs.is_3d:
        nx = max(1, int(p.Lx / w))
        ny = max(1, int(p.Ly / w))
        ix = np.clip((gs.x[:N] / p.Lx * nx).astype(int), 0, nx - 1)
        iy = np.clip((gs.y[:N] / p.Ly * ny).astype(int), 0, ny - 1)
        idx = ix * ny + iy
        tops = np.full(nx * ny, -1.0)
        np.maximum.at(tops, idx[mob], (gs.z[:N] + rb_up)[mob])
        A_col = (p.Lx / nx) * (p.Ly / ny)
        if geom.shape_code == 1:
            cxs = (np.arange(nx) + 0.5) * (p.Lx / nx)
            cys = (np.arange(ny) + 0.5) * (p.Ly / ny)
            CX, CY = np.meshgrid(cxs, cys, indexing='ij')
            inside = ((CX - geom.cx) ** 2 + (CY - geom.cy) ** 2).ravel() <= geom.R_cyl ** 2
            tops = np.where(inside, tops, -1.0)
    else:
        nx = max(1, int(p.Lx / w))
        ix = np.clip((gs.x[:N] / p.Lx * nx).astype(int), 0, nx - 1)
        tops = np.full(nx, -1.0)
        np.maximum.at(tops, ix[mob], (gs.y[:N] + rb_up)[mob])
        A_col = p.Lx / nx
    occ = tops >= 0.0
    if not np.any(occ):
        return empty
    t = tops[occ]
    return dict(bed_height_mean=float(np.mean(t)), bed_height_p95=float(np.percentile(t, 95)),
                bed_envelope_volume=float(A_col * np.sum(t)), n_columns=int(np.sum(occ)),
                col_tops=tops)


def _print_bed_report(gs, p):
    bed = bed_surface(gs, p)
    V = bed_solid_volume(gs)
    env = bed['bed_envelope_volume']
    phi_bed = V / env if env > 0 else float('nan')
    unit = 'um^3' if gs.is_3d else 'um^2'
    print(f"  Bed:           height {bed['bed_height_mean']:.0f} um (p95 {bed['bed_height_p95']:.0f}), "
          f"envelope solid fraction {phi_bed:.3f} (solid {V:.3g} {unit} in {env:.3g} {unit})")


def bed_metrics(gs, p):
    """Positional observables of a granule bed (V3.1).

    Everything here is computed from centres and radii, so unlike the
    field-based ``phi_solid`` it is free of the tanh interface halo and of
    pairwise-overlap double counting — these are the numbers to compare with
    a measured bed. Immobile boundary granules are excluded throughout: the
    lining is container, not sample.

    Keys: bed_height_mean / bed_height_p95 / bed_envelope_volume / phi_bed /
    bed_radius_p95 / wall_contact_fraction / n_floating / n_top_clamped.
    """
    out = {}
    N = gs.N
    if N == 0:
        return out
    geom = boundary_geometry(p, gs.mode if gs.mode in ('2D', '3D') else None)
    fixed = getattr(gs, 'fixed', None)
    mob = np.ones(N, dtype=bool) if fixed is None else ~np.asarray(fixed[:N], dtype=bool)
    bed = bed_surface(gs, p)
    V_solid = bed_solid_volume(gs)
    env = bed['bed_envelope_volume']
    out['bed_height_mean'] = bed['bed_height_mean']
    out['bed_height_p95'] = bed['bed_height_p95']
    out['bed_envelope_volume'] = env
    out['phi_bed'] = float(V_solid / env) if env > 0 else 0.0
    # V3.2: the same fractions with the interpenetration lens removed. phi_bed
    # sums nominal radii, so a bed compacting THROUGH overlap -- which is what a
    # soft granule does under cell traction -- reads high by 1-4 %.
    V_lens = overlap_solid_volume(gs)
    out['bed_solid_volume_net'] = float(max(0.0, V_solid - V_lens))
    out['phi_bed_net'] = float(out['bed_solid_volume_net'] / env) if env > 0 else 0.0
    out['overlap_volume_fraction'] = float(V_lens / V_solid) if V_solid > 0 else 0.0

    rb = np.asarray(gs.r_bound[:N])
    if geom.shape_code == 1:
        rho = np.hypot(gs.x[:N] - geom.cx, gs.y[:N] - geom.cy)
        reach = rho + rb
        out['bed_radius_p95'] = float(np.percentile(reach[mob], 95)) if np.any(mob) else 0.0
        shell = mob & (reach > geom.R_cyl - 2.0 * float(np.mean(rb[mob]) if np.any(mob) else 1.0))
        touching = mob & (geom.R_cyl - reach < 1.0)
        out['wall_contact_fraction'] = (float(np.sum(touching)) / float(np.sum(shell))
                                        if np.any(shell) else 0.0)
    else:
        cx, cy = 0.5 * p.Lx, 0.5 * p.Ly
        rho = np.hypot(gs.x[:N] - cx, gs.y[:N] - cy)
        out['bed_radius_p95'] = float(np.percentile((rho + rb)[mob], 95)) if np.any(mob) else 0.0
        # V3.6: the surface-to-nearest-wall gap, over the faces that ARE walls.
        # Until V3.6 the 3D branch looked at the two x faces only, so the floor
        # -- the one face a sedimented bed actually rests on -- was invisible
        # and this read 0.0 on a bed that was demonstrably in wall contact. The
        # V3.1 free top is not a wall and is excluded in both dimensions.
        _faces = [gs.x[:N] - rb, p.Lx - gs.x[:N] - rb, gs.y[:N] - rb]
        if gs.is_3d:
            _faces.append(p.Ly - gs.y[:N] - rb)
            _faces.append(gs.z[:N] - rb)
            if not geom.top_free:
                _faces.append(p.Lz - gs.z[:N] - rb)
        elif not geom.top_free:
            _faces.append(p.Ly - gs.y[:N] - rb)
        near = _faces[0]
        for _f in _faces[1:]:
            near = np.minimum(near, _f)
        shell = mob & (near < 2.0 * float(np.mean(rb[mob]) if np.any(mob) else 1.0))
        out['wall_contact_fraction'] = (float(np.sum(mob & (near < 1.0))) / float(np.sum(shell))
                                        if np.any(shell) else 0.0)

    # floating: no bounding-sphere contact, off the floor, not bridged
    up = gs.z[:N] if gs.is_3d else gs.y[:N]
    try:
        from scipy.spatial import cKDTree
        dim = 3 if gs.is_3d else 2
        pos = np.ascontiguousarray(gs.pos[:N, :dim])
        pairs = cKDTree(pos).query_pairs(2.0 * float(np.max(rb)) + 1.0, output_type='ndarray')
        nc = np.zeros(N, dtype=int)
        if len(pairs):
            d = np.linalg.norm(pos[pairs[:, 1]] - pos[pairs[:, 0]], axis=1)
            touch = d - rb[pairs[:, 0]] - rb[pairs[:, 1]] < 1.0
            if np.any(touch):
                nc = np.bincount(pairs[touch].ravel(), minlength=N)
        bridged = np.zeros(N, dtype=bool)
        if gs.total_cells:
            host = gs.cell_granule_id[gs.cell_bridge_target >= 0]
            tgt = gs.cell_bridge_target[gs.cell_bridge_target >= 0]
            bridged[np.asarray(host, dtype=int)] = True
            bridged[np.asarray(tgt, dtype=int)] = True
        out['n_floating'] = int(np.sum(mob & (nc == 0) & ((up - rb) > 1.0) & ~bridged))
    except Exception:
        out['n_floating'] = 0
    out['n_top_clamped'] = int(getattr(gs, 'n_top_clamped', 0))
    return out


def phi_z_profile(gs, p, n_bins=0):
    """Solid fraction against height, from exact sphere-slab volumes (V3.1).

    Returns (edges, phi). Each granule contributes its analytic cap volume to
    every bin it spans, so the profile is free of the rendering halo.
    """
    N = gs.N
    fixed = getattr(gs, 'fixed', None)
    mob = np.ones(N, dtype=bool) if fixed is None else ~np.asarray(fixed[:N], dtype=bool)
    up = (gs.z[:N] if gs.is_3d else gs.y[:N])[mob]
    r = np.asarray(gs.r[:N])[mob]
    L_up = float(p.Lz if gs.is_3d else p.Ly)
    if n_bins <= 0:
        r_max = float(np.max(r)) if r.size else 1.0
        n_bins = int(np.clip(L_up / max(2.0 * r_max, 1e-9), 10, 200))
    edges = np.linspace(0.0, L_up, n_bins + 1)
    phi = np.zeros(n_bins)
    if r.size == 0:
        return edges, phi
    A_base = domain_base_area(p, gs.mode if gs.mode in ('2D', '3D') else None)
    dz = edges[1] - edges[0]
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        # spherical-cap volume of each granule between lo and hi
        h_hi = np.clip(hi - (up - r), 0.0, 2.0 * r)
        h_lo = np.clip(lo - (up - r), 0.0, 2.0 * r)
        if gs.is_3d:
            def cap(h):
                return np.pi * (r * h ** 2 - h ** 3 / 3.0)
        else:
            def cap(h):
                d = np.clip(h - r, -r, r)
                return r ** 2 * (np.arcsin(d / r) + 0.5 * np.sin(2.0 * np.arcsin(d / r))) + \
                    0.5 * np.pi * r ** 2
        phi[b] = float(np.sum(cap(h_hi) - cap(h_lo))) / (A_base * dz)
    return edges, phi


# V3.4: warn when the packed bed hands the dynamics more than this multiple of
# the driving load. 100x is generous -- it is a smell test, not a tolerance.
HANDOFF_WARN_RATIO = 100.0


def granule_weights(gs, p):
    """Buoyant weight per granule in nN: (rho_gran - rho_medium) g (4/3) pi r^3 * 1e-9.

    Zero for every granule when gravity is off, and for immobile boundary
    granules. (rho in kg/m^3, g in m/s^2, r in um.)
    """
    N = gs.N
    if not getattr(p, 'gravity_enabled', False):
        return np.zeros(N)
    rho = getattr(gs, 'rho_gran', None)
    if rho is None:
        rho = np.full(N, float(getattr(p, 'granule_density', 1000.0)))
    d_rho = rho[:N] - float(p.medium_density)
    vol = (4.0 / 3.0) * np.pi * gs.r[:N] ** 3
    w = d_rho * float(p.g_accel) * vol * 1e-9 * float(p.gravity_scale)
    fixed = getattr(gs, 'fixed', None)
    if fixed is not None:
        w = np.where(fixed[:N], 0.0, w)
    return w


def settle_overlap_tolerance(gs, p):
    """The penetration the settle should stop at, in um, or ``None`` (V3.5).

    ``None`` unless ``packing.overlap_tol_model == 'elastic'``, in which case
    the settle keeps its V2.7 rule of ``0.05 * mean_r``.

    **That rule is not a mismeasurement, it is the wrong dimension.** Measured
    on a 199-granule shaped gravity bed: the settle's proxy reported 0.975 um
    where the true MTD penetration was 1.40 um -- only 1.4x out. But force
    balance for that bed needs **0.035 um**, so the criterion is 28x too loose
    in LENGTH, and because ``F ~ delta^1.5`` that is 145x in FORCE. The handoff
    was measured at 346x. Stopping on a better-measured length, which is what
    the V3.3 plan proposed, would have bought the factor of 1.4.

    So derive the length from the force instead, by inverting Hertz at the load
    the run is actually driven by::

        (4/3) E* sqrt(R*) delta^1.5 * 1e-3 = relax_force_tol * load_scale

    This is the same move ``contact.overlap_model: elastic`` makes for
    ``max_overlap_frac`` (V3.2): size the length from the contact law rather
    than by hand. It costs no force evaluation, so unlike `relax_packing` it is
    free -- but it is also only as good as the settle's own soft repulsion,
    which equilibrates at 0.091 um on that bed. The two compose: this gets the
    bed close, FIRE finishes it.

    Returns ``None`` when nothing drives the run, since then there is no load
    to size against.
    """
    if str(getattr(p, 'packing_overlap_tol_model', 'fixed')) != 'elastic':
        return None
    scale, _kind = dynamics_load_scale(gs, p)
    if not scale:
        return None
    E_star = (float(p.E_modulus) * 1e3) / (2.0 * (1.0 - float(p.poisson_ratio) ** 2))
    cap = float(getattr(p, 'contact_E_cap', 0.0) or 0.0)
    if cap > 0.0:
        E_star = min(E_star, (cap * 1e3) / (2.0 * (1.0 - float(p.poisson_ratio) ** 2)))
    R_star = 0.5 * float(np.mean(gs.r[:gs.N])) if gs.N else 1.0
    coeff = (4.0 / 3.0) * E_star * np.sqrt(max(R_star, 1e-9)) * 1e-3
    if coeff <= 0.0:
        return None
    f_tol = float(getattr(p, 'packing_relax_force_tol', 0.5)) * scale
    return float((f_tol / coeff) ** (2.0 / 3.0))


def constraint_clamped(gs, p, F, eps=1e-6):
    """Which granules are held by the POSITION CLAMP rather than by a force (V3.5).

    ``apply_position_bounds`` keeps every granule **0.5 um clear of every
    wall** (`np.clip(x, rb + 0.5, L - rb - 0.5)`), so a granule resting on the
    floor is never in wall contact: the JKR wall force sees a positive gap and
    does nothing, and the granule's net force stays exactly its own weight,
    carried by the clamp. The clamp is a rigid constraint and that granule IS
    in equilibrium -- but the reaction is not in `F`.

    So any residual-force measure that ignores this has an **irreducible floor
    of one granule weight** for every gravity bed, and no relaxation can get
    under it. Measured on a 199-granule sedimented bed, every one of the five
    worst-balanced granules sat at exactly `y - r = 0.500` with its full weight
    unbalanced.

    This is a pre-existing inconsistency between the position clamp and the
    wall contact law, not something V3.5 introduced. V3.5 only stops
    misreporting it.

    Detected by nudging along `F` and asking who cannot move, which is
    geometry-agnostic -- box, cylinder, free top and the legacy lid all work
    without duplicating `apply_position_bounds`'s branches.
    """
    N = gs.N
    if N == 0 or p.boundary_mode == 'periodic':
        return np.zeros(max(N, 0), dtype=bool)
    dim = 3 if gs.is_3d else 2
    F = np.asarray(F)[:N, :dim]
    n = np.linalg.norm(F, axis=1)
    live = n > 0.0
    if not np.any(live):
        return np.zeros(N, dtype=bool)
    save = gs.pos[:N].copy()
    save_top = int(getattr(gs, 'n_top_clamped', 0) or 0)
    nudge = np.zeros((N, dim))
    nudge[live] = (F[live] / n[live, None]) * eps
    gs.pos[:N, :dim] += nudge
    intended = gs.pos[:N, :dim].copy()
    apply_position_bounds(gs, p)
    blocked = np.any(np.abs(gs.pos[:N, :dim] - intended) > 0.1 * eps, axis=1)
    gs.pos[:N] = save
    gs.n_top_clamped = save_top
    return blocked & live


def free_force_residual(gs, p, F):
    """``(max|F| over granules the clamp is not holding, n_clamped)`` (V3.5).

    The honest residual: a granule pressed into a rigid floor is in
    equilibrium even though its `F` is its whole weight. See
    :func:`constraint_clamped` for why that case exists at all.
    """
    N = gs.N
    if N == 0:
        return 0.0, 0
    held = constraint_clamped(gs, p, F)
    fixed = getattr(gs, 'fixed', None)
    if fixed is not None:
        held = held | np.asarray(fixed[:N], dtype=bool)
    mag = np.linalg.norm(np.asarray(F)[:N], axis=1)
    free = ~held
    return (float(mag[free].max()) if np.any(free) else 0.0), int(np.sum(held))


def free_force_percentile(gs, p, F, q=95.0):
    """The q-th percentile of the clamp-free residual, nN (V3.5).

    The max is a max over a heavy tail: on a loose shaped bed a SINGLE wedged
    granule read 132x the gravity load while the second-worst read 0.97x. Quote
    both, or the verdict is decided by one granule.
    """
    N = gs.N
    if N == 0:
        return 0.0
    held = constraint_clamped(gs, p, F)
    fixed = getattr(gs, 'fixed', None)
    if fixed is not None:
        held = held | np.asarray(fixed[:N], dtype=bool)
    mag = np.linalg.norm(np.asarray(F)[:N], axis=1)[~held]
    return float(np.percentile(mag, q)) if mag.size else 0.0


def dynamics_load_scale(gs, p):
    """The largest force the run is SUPPOSED to be driven by, in nN (V3.4).

    Returns ``(scale, name)``, or ``(None, None)`` when nothing drives the run.
    Two loads apply: the buoyant weight of a granule when ``gravity_enabled``,
    and the force one cell can exert when cells are seeded. The comparison is
    against the largest, because a residual below the biggest driver cannot be
    what the run is about.

    Shared by :func:`handoff_force_balance`, which reports the ratio, and by
    :func:`relax_packing`, which relaxes until the ratio is small -- so the
    detector and the fix cannot disagree about what "small" means.
    """
    scales = {}
    w = granule_weights(gs, p)
    if w.size and float(np.max(w)) > 0.0:
        scales['gravity'] = float(np.max(w))
    if int(getattr(gs, 'total_cells', 0) or 0) > 0:
        traction = float(getattr(p, 'F_max_per_cell', 0.0) or 0.0)
        if traction <= 0.0:
            traction = float(getattr(p, 'expected_bridge_force', 0.0) or 0.0)
        if traction > 0.0:
            scales['traction'] = traction
    if not scales:
        return None, None
    name = max(scales, key=scales.get)
    return scales[name], name


def handoff_force_balance(gs, p, F0):
    """Is the packed bed in force balance under the DYNAMICS' force law? (V3.4)

    The packer and the dynamics are two different force models, and the settle
    stops on a LENGTH tolerance (``max_overlap < packing_overlap_tol``) that
    makes no reference to the law the dynamics will then apply. Nothing checked
    the handoff, so a packing could be delivered enormously pre-loaded and the
    first hours of the run would be it exploding -- physical descent from an
    unphysical initial condition, which no integrator fix can help.

    The invariant is a scale comparison, not a conservation law: at t = 0 the
    residual contact force should be at most comparable to whatever is supposed
    to drive the run. Two load scales apply:

      * gravity -- the buoyant weight of a granule, when ``gravity_enabled``;
      * traction -- the force one cell can exert, when cells are seeded.

    Measured on a gravity-consolidated 2D bed of AR-1.8 / n-3.5 granules, the
    ratio is **5e4**: median true penetration 4.66 um where force balance needs
    4.8 nm. See the V3.4 changelog.

    Returns a dict (recorded in ``metadata['packing']``); prints a warning when
    the ratio exceeds ``HANDOFF_WARN_RATIO``.
    """
    F0 = np.asarray(F0)
    if F0.size == 0:
        return {}
    f_max = float(np.linalg.norm(F0, axis=1).max())
    f_mean = float(np.linalg.norm(F0, axis=1).mean())
    # V3.5: judge on the residual the clamp is NOT holding. A granule resting on
    # the floor keeps its full weight in `F` forever (see `constraint_clamped`),
    # so the raw max has an irreducible floor of ~1x the gravity load.
    f_free, n_held = free_force_residual(gs, p, F0)

    scale, name = dynamics_load_scale(gs, p)
    if scale is None:
        gs.handoff_report = {'handoff_F_max': f_max, 'handoff_F_mean': f_mean,
                             'handoff_F_max_free': f_free, 'handoff_n_clamped': n_held}
        return gs.handoff_report
    ratio = f_free / scale
    out = {'handoff_F_max': f_max, 'handoff_F_mean': f_mean,
           'handoff_F_max_free': f_free, 'handoff_n_clamped': n_held,
           'handoff_load_scale': scale, 'handoff_load_kind': name,
           'handoff_ratio': ratio}
    gs.handoff_report = out
    if ratio > HANDOFF_WARN_RATIO:
        print(f"  WARNING: the packed bed is not in force balance for the dynamics.\n"
              f"    max |F| at t=0 is {f_free:.3g} nN (over the {gs.N - n_held} granules the "
              f"wall clamp is not holding)\n"
              f"    against a {name} load scale of {scale:.3g} nN ({ratio:.3g}x).\n"
              f"    The run will begin by relaxing that, not by doing the physics you "
              f"asked for.\n"
              f"    The settle stops on a LENGTH tolerance (packing.overlap_tol) that "
              f"knows nothing about\n"
              f"    the contact law; for a gravity bed, force balance needs overlaps "
              f"~1e-3 of the tolerance.")
    return out


# ══════════════════════════════════════════════════════════════════════
# V3.5 — gradient flow: the energy whose gradient the dynamics follow
# ══════════════════════════════════════════════════════════════════════
#
# Overdamped dynamics is gradient flow. If every force is -grad E then
# gamma x_dot = -grad E, so
#
#     dE/dt = grad E . x_dot = -gamma |x_dot|^2 <= 0
#
# and the energy must fall every step, with the work the forces do on the step
# accounting for the whole of the fall. Two things are readable from that:
#
#   * an ASCENT (`energy_delta > 0`) is an integrator failure -- and it
#     survives friction, which only ever removes energy;
#   * the RESIDUAL `energy_delta + energy_work` is the non-conservative
#     throughput: every watt that went somewhere other than the potential.
#
# The second is why this exists. V3.4 proved the MTD contact solver
# conservative POINTWISE -- d(delta)/d(c2) = -n* to nine places, by the
# envelope theorem -- but never for the ASSEMBLED force law on a real run.
# Checking `-grad E == F` by central differences on a packed 13-granule bed
# (tests/test_gradient_flow.py) turns that into an end-to-end statement, and
# the measured decomposition of what is left is:
#
#   | term                          | relative size of `-grad E - F`      |
#   |-------------------------------|-------------------------------------|
#   | active noise, T_active = 5    | 1.0        (the whole force)        |
#   | active noise, T_active = 0    | 3.9e-9     (machine, spheres)       |
#   | MC-DEM on                     | 1.3e-1                              |
#   | shaped granules (MTD)         | 2.6e-3                              |
#   | tangential friction           | 99 % of the per-step residual       |
#
# None of those is a bug. Friction is dissipative and SHOULD break the
# equality; the noise is an athermal driving term; MC-DEM's kappa is a
# state-dependent multiplier on the force rather than a term in any energy;
# and for shaped granules `R_eff` depends on configuration while the force law
# treats it as a parameter, so the neglected `dU/dR . grad R` costs 0.26 %.
# Cell bridges are actuators and inject work by design. What the audit buys is
# telling "this force law is WRONG" apart from "this force law is ACTIVE" --
# and the numbers above are the calibration for doing so.
#
# With friction, noise and cells all off, the residual converges as O(dt):
# 0.2 % at dt = 1e-3 h, 0.02 % at dt = 1e-4 h.
#
# The system is NOT a gradient flow when cells are seeded. That is physics, not
# a defect, and V3.5 does not try to remove it. The invariant is exact for the
# PASSIVE problem -- packing, sedimentation, settling, relaxation -- which is
# exactly where the V3.4 packer-handoff finding lives.

ENERGY_ASCENT_TOL = 0.1        # a step is flagged when dE > tol * |work done on it|
# ...but only once dE is bigger than the float resolution of E itself. A bed that FIRE has
# already relaxed does almost no work per step, so `dE / |W|` is 0/0 and explodes: measured
# at 9.3e3 on a relaxed bed whose dE and W were both at rounding. Below this the SIGN of dE
# is not information.
ENERGY_NOISE_REL = 1e-9
ENERGY_MAX_BACKTRACK = 6       # 'damped': halvings of the step scale before it floors
ENERGY_SCALE_RELAX = 1.1       # 'damped': per-descending-step relaxation back toward 1


def jkr_contact_energy(a_um, R_eff_um, E_star_Pa, W_Jm2):
    """Energy stored in a JKR contact of radius ``a``, in nN um (V3.5).

    The JKR equilibrium branch is single-valued in the contact radius a::

        delta(a) = a^2/R* - sqrt(2 pi W a / E*)
        F(a)     = (4/3) E* a^3/R* - sqrt(8 pi W E* a^3)

    so ``U = int_0^delta F d(delta')  =  int_0^a F(a') delta''(a') da'`` is
    closed form::

        U(a) = (8/15) E* a^5 / R*^2
             - (4/3) sqrt(2 pi W E*) a^(7/2) / R*
             + pi W a^2

    At ``W = 0``, ``a = sqrt(R* delta)`` this reduces to the Hertz energy
    ``(2/5) k delta^(5/2)`` with ``k = (4/3) E* sqrt(R*)`` -- the identity that
    pins the algebra, and the first test in ``tests/test_gradient_flow.py``.

    Vectorised: a, R_eff, E_star and W may be arrays. Units are the engine's
    (um, Pa, J/m^2 = nN/um) and the result is in nN um.
    """
    a = np.asarray(a_um, dtype=float)
    R = np.asarray(R_eff_um, dtype=float)
    E_s = np.asarray(E_star_Pa, dtype=float) * 1e-3      # Pa -> nN/um^2
    W_s = np.maximum(np.asarray(W_Jm2, dtype=float), 0.0)  # J/m^2 == nN/um
    ok = (a > 0.0) & (R > 0.0)
    a = np.where(ok, a, 0.0)
    Rs = np.where(ok, R, 1.0)
    U = (8.0 / 15.0) * E_s * a ** 5 / Rs ** 2
    U = U - (4.0 / 3.0) * np.sqrt(2.0 * np.pi * W_s * E_s) * a ** 3.5 / Rs
    U = U + np.pi * W_s * a ** 2
    return np.where(ok, U, 0.0)


def contact_potential_energy(contacts):
    """Total JKR energy on the granule-granule contact list, nN um (V3.5).

    One vectorised pass over columns the contact record already carries
    (``a_contact``, ``R_eff``, ``E_star``, ``W``) -- the same trick
    :func:`contact_stiffness_per_granule` uses, so no kernel changes and the
    two twins cannot disagree about it.

    **Not** included: the MC-DEM factor ``kappa``. It multiplies the force
    without being the gradient of anything, so there is no energy to add for
    it; its effect is precisely the residual this audit reports.
    """
    if contacts is None or len(contacts) == 0:
        return 0.0
    col = getattr(contacts, 'column', None)
    if col is not None:
        a = np.asarray(col('a_contact'), dtype=float)
        R = np.asarray(col('R_eff'), dtype=float)
        E = np.asarray(col('E_star'), dtype=float)
        W = np.asarray(col('W'), dtype=float)
    else:                                   # reference list-of-dicts path
        a = np.array([c.get('a_contact', 0.0) for c in contacts], dtype=float)
        R = np.array([c.get('R_eff', 0.0) for c in contacts], dtype=float)
        E = np.array([c.get('E_star', 0.0) for c in contacts], dtype=float)
        W = np.array([c.get('W', 0.0) for c in contacts], dtype=float)
    return float(np.sum(jkr_contact_energy(a, R, E, W)))


def gravity_potential_energy(gs, p):
    """Total gravitational potential of the mobile granules, nN um (V3.5).

    ``sum_i w_i h_i`` with ``w_i`` the identical :func:`granule_weights` array
    the force path uses, so the two cannot disagree about the sign or about
    which granules are immobile. Up is +z in 3D and +y in 2D, matching
    :func:`apply_gravity`.
    """
    N = gs.N
    w = granule_weights(gs, p)
    if not np.any(w):
        return 0.0
    h = gs.z[:N] if gs.is_3d else gs.y[:N]
    return float(np.dot(w, h))


def wall_potential_energy(gs, p):
    """Total JKR energy in the granule-wall contacts, nN um (V3.5).

    Wall contacts are not on the contact list -- both gathers apply them inline
    -- so this recomputes them from positions. V3.4 made that exact and cheap:
    a wall is a half-space, so ``penetration = h(-w_hat) - (c - q).w_hat`` is a
    single support evaluation.

    This MIRRORS the face enumeration of ``gather_2d`` / ``gather_3d`` rather
    than sharing it, and that duplication is the risk. It is converted into a
    tested invariant by ``tests/test_gradient_flow.py``, which central-
    differences the TOTAL energy against the force the gather actually returns
    -- if the two enumerations ever disagree, that test fails.

    Cost is one njit call per granule per face, from Python. At N = 1000 in 3D
    that is ~6000 calls per step, so ``gradient_flow`` is opt-in.
    """
    if p.boundary_mode == 'periodic':
        return 0.0
    N = gs.N
    if N == 0:
        return 0.0
    geom = boundary_geometry(p)
    sid = gs.species_id
    Wt = gs.wall_W
    Et = gs.wall_Estar
    U = 0.0
    if gs.is_3d:
        L = (float(p.Lx), float(p.Ly), float(p.Lz))
        for i in range(N):
            Ww = float(Wt[sid[i]])
            Ew = float(Et[sid[i]])
            xi, yi, zi = float(gs.x[i]), float(gs.y[i]), float(gs.z[i])
            crd = (xi, yi, zi)
            for w in range(6):
                axis = w // 2
                if geom.top_free and w == 5:
                    continue                       # V3.1 open top
                if geom.shape_code == 1 and axis < 2:
                    continue                       # V3.1 cylinder replaces the x/y faces
                if w % 2 == 0:
                    wall_pos, sign = 0.0, 1
                else:
                    wall_pos, sign = L[axis], -1
                if gs.is_circle:
                    ri = float(gs.r[i])
                    pen = (ri - (crd[axis] - wall_pos)) if sign > 0 else ((crd[axis] + ri) - wall_pos)
                    ok, R_local = pen > 0.0, ri
                else:
                    ok, pen, R_local, _a, _b, _c = se3d_wall_core(
                        xi, yi, zi, gs.a[i], gs.b[i], gs.c[i], gs.n1[i], gs.n2[i],
                        gs.quat[i], wall_pos, axis, sign)
                if ok:
                    _F, a_w = jkr_force_from_overlap(pen, R_local, Ew, Ww)
                    U += float(jkr_contact_energy(a_w, R_local, Ew, Ww))
            if geom.shape_code == 1:               # V3.1 cylindrical side wall
                dxc, dyc = xi - geom.cx, yi - geom.cy
                rho = float(np.hypot(dxc, dyc))
                if rho > 1e-12:
                    ux, uy = dxc / rho, dyc / rho
                    if gs.is_circle:
                        pen = float(gs.r[i]) + rho - geom.R_cyl
                        ok, R_local = pen > 0.0, float(gs.r[i])
                    else:
                        ok, pen, R_local, _a, _b, _c = se3d_wall_plane_core(
                            xi, yi, zi, gs.a[i], gs.b[i], gs.c[i], gs.n1[i], gs.n2[i], gs.quat[i],
                            geom.cx + geom.R_cyl * ux, geom.cy + geom.R_cyl * uy, zi,
                            -ux, -uy, 0.0)
                    if ok:
                        _F, a_w = jkr_force_from_overlap(pen, R_local, Ew, Ww)
                        U += float(jkr_contact_energy(a_w, R_local, Ew, Ww))
    else:
        Lx, Ly = float(p.Lx), float(p.Ly)
        for i in range(N):
            Ww = float(Wt[sid[i]])
            Ew = float(Et[sid[i]])
            xi, yi = float(gs.x[i]), float(gs.y[i])
            for w in range(4):
                if geom.top_free and w == 3:
                    continue                       # V3.1 open top: y = Ly is a free surface
                if gs.is_circle:
                    ri = float(gs.r[i])
                    pen = (ri - xi, xi - (Lx - ri), ri - yi, yi - (Ly - ri))[w]
                    ok, R_local = pen > 0.0, ri
                else:
                    wall_pos, axis, sign = ((0.0, 0, 1), (Lx, 0, -1), (0.0, 1, 1), (Ly, 1, -1))[w]
                    ok, pen, R_local, _a, _b = se2d_wall_core(
                        xi, yi, gs.a[i], gs.b[i], gs.n_shape[i], gs.theta[i],
                        wall_pos, axis, sign)
                if ok:
                    _F, a_w = jkr_force_from_overlap(pen, R_local, Ew, Ww)
                    U += float(jkr_contact_energy(a_w, R_local, Ew, Ww))
    return U


def active_noise_power(gs, p):
    """Expected work the ACTIVE NOISE injects per step, nN um (V3.5).

    :func:`gels.kernels.contacts.add_active_noise` adds an independent
    ``F ~ N(0, 2 gamma T act / dt)`` per axis to every adhesive granule, so the
    noise's contribution to its own work ``F . v dt = F^2/gamma dt`` has
    expectation ``2 d T act`` per granule per step -- independent of both dt and
    gamma.

    This matters more than it looks. ``T_active`` defaults to 5 nN um, which in
    2D is **20 nN um per functional granule per step**, the same size as the
    work the contact law does. A run at the default is therefore a NOISY
    gradient flow, not a gradient flow, and ``energy_residual`` is dominated by
    the noise rather than by anything wrong with the contact law. Measured
    against a 13-granule bed the force check goes from 100 % error at
    ``T_active = 5`` to 3.9e-9 at ``T_active = 0``.

    Reported so the residual can be read against its own noise floor. Set
    ``T_active = 0`` to audit the contact law alone.
    """
    T = float(getattr(p, 'T_active', 0.0))
    if T <= 0.0:
        return 0.0
    N = gs.N
    m = np.asarray(gs.adhesive_mask[:N], dtype=bool)
    fixed = getattr(gs, 'fixed', None)
    if fixed is not None:
        m = m & ~np.asarray(fixed[:N], dtype=bool)
    if not np.any(m):
        return 0.0
    dim = 3 if gs.is_3d else 2
    return float(2.0 * dim * T * np.sum(gs.activity[:N][m]))


def system_energy(gs, p, contacts):
    """The potential the dynamics is supposed to be descending, nN um (V3.5).

    Returns ``(total, parts)``. ``contacts`` is the list from the force
    evaluation at the CURRENT configuration -- pass the one you already have,
    never recompute it for this.
    """
    e_c = contact_potential_energy(contacts)
    e_w = wall_potential_energy(gs, p)
    e_g = gravity_potential_energy(gs, p)
    return e_c + e_w + e_g, {'contact': e_c, 'wall': e_w, 'gravity': e_g}


def _energy_audit(gs, p, mode, E0, parts, work):
    """Close out one step of the gradient-flow audit (V3.5).

    The audit is uniformly ONE STEP BEHIND, and deliberately so: closing it
    would need the energy at the post-step configuration, which costs a second
    force evaluation per step. ``energy_total`` is therefore E at the
    configuration this step started from, and ``energy_delta`` / ``energy_work``
    describe the step BEFORE it. The lag costs one row and no accuracy.

    ``damped`` mode is a feedback controller on the effective drag, not a
    guarantee. It does not stop the first ascending step; it stops a run from
    SUSTAINING ascent. Three reasons it is built this way rather than as a
    backtracking line search:

      * a line search needs a second force evaluation every step (2x the run);
      * dividing gamma exactly preserves the fixed point -- at F = 0 the step
        is zero for any drag -- so it changes the transient, never the physics.
        That is the identical argument `contact_semi_implicit` rests on, and
        the two compose;
      * with cells seeded, monotone descent is FALSE (bridges are actuators),
        so a mode that promised to enforce it would be promising something the
        model does not claim.
    """
    gs.energy_total = E0
    gs.energy_contact = parts['contact']
    gs.energy_wall = parts['wall']
    gs.energy_gravity = parts['gravity']
    gs.energy_noise_expected = active_noise_power(gs, p)
    prev = gs.energy_prev
    if prev is None:
        gs.energy_delta = 0.0
        gs.energy_work = 0.0
        gs.energy_residual = 0.0
        gs.energy_ascent_frac = 0.0
    else:
        E_prev, w_prev = prev
        dE = E0 - E_prev
        gs.energy_delta = dE
        gs.energy_work = w_prev
        gs.energy_residual = dE + w_prev
        if abs(dE) <= ENERGY_NOISE_REL * max(abs(E0), 1.0):
            frac = 0.0            # both dE and the work are at rounding; 0/0 is not an ascent
        else:
            frac = dE / max(abs(w_prev), 1e-30)
        gs.energy_ascent_frac = frac
        ascending = frac > ENERGY_ASCENT_TOL
        if ascending:
            gs.energy_ascent_steps += 1
            if frac > gs.energy_max_ascent_frac:
                gs.energy_max_ascent_frac = frac
        if mode == 'damped':
            if ascending:
                gs.energy_step_scale = max(gs.energy_step_scale * 0.5,
                                           2.0 ** -ENERGY_MAX_BACKTRACK)
                gs.energy_backtracks += 1
            else:
                gs.energy_step_scale = min(1.0, gs.energy_step_scale * ENERGY_SCALE_RELAX)
    gs.energy_prev = (E0, work)


def energy_metrics(gs):
    """The gradient-flow audit as flat metric keys (V3.5).

    Shared by both metric twins, like :func:`projection_metrics`, so the key
    sets cannot diverge. Every value is a plain float and never NaN; all zero
    when ``dynamics.gradient_flow`` is ``off``, so Gate B only ever sees added
    keys holding a constant.

    ``energy_work`` is ``sum F . dx + sum tau . dtheta`` over the step, i.e.
    the work the force field did. For a conservative force law
    ``energy_delta = -energy_work`` to O(dt), so ``energy_residual`` is the
    non-conservative throughput -- the work that went somewhere other than the
    potential. In a passive bed that is almost entirely FRICTION HEAT (99 % of
    it at the default ``tau_0``); the rest is the active noise, MC-DEM, cell
    bridges, the overlap-resolution projection and the velocity cap.

    Read it against ``energy_noise_expected``, the floor the active noise alone
    puts under it. At the default ``T_active = 5`` that floor is the same size
    as the work the contact law does, so a residual below it says nothing at
    all -- set ``T_active = 0`` to audit the contact law.

    ``energy_delta <= 0`` is the sharper statement and the one to watch: it
    survives friction, which only ever removes energy. It does NOT survive
    active noise or cell traction, both of which inject.
    """
    return {
        'energy_total': float(getattr(gs, 'energy_total', 0.0)),
        'energy_contact': float(getattr(gs, 'energy_contact', 0.0)),
        'energy_wall': float(getattr(gs, 'energy_wall', 0.0)),
        'energy_gravity': float(getattr(gs, 'energy_gravity', 0.0)),
        'energy_delta': float(getattr(gs, 'energy_delta', 0.0)),
        'energy_work': float(getattr(gs, 'energy_work', 0.0)),
        'energy_residual': float(getattr(gs, 'energy_residual', 0.0)),
        'energy_noise_expected': float(getattr(gs, 'energy_noise_expected', 0.0)),
        'energy_ascent_frac': float(getattr(gs, 'energy_ascent_frac', 0.0)),
        'energy_ascent_steps': int(getattr(gs, 'energy_ascent_steps', 0)),
        'energy_max_ascent_frac': float(getattr(gs, 'energy_max_ascent_frac', 0.0)),
        'energy_backtracks': int(getattr(gs, 'energy_backtracks', 0)),
    }


# ══════════════════════════════════════════════════════════════════════
# V3.5 Phase 2 — FIRE relaxation of the packed bed (the handoff fix)
# ══════════════════════════════════════════════════════════════════════
#
# V3.4 measured the disease: the packer hands the dynamics a bed pre-loaded by
# ~2.7e3x the gravitational load, so the first hours of a run are that
# unwinding rather than the physics asked for. `handoff_force_balance` was the
# detector. This is the fix.
#
# The V3.3 plan specified FIRE reusing the SETTLE's own `k_rep ov^1.5` law,
# stopping on `max|F| < f_tol` in that law's units, to avoid a circular import
# (`gels/kernels/packing.py` cannot reach `compute_forces`). That would NOT have
# fixed this: the handoff error is a mismatch BETWEEN two force laws, and force
# balance under the settle's law says nothing about `max|F|` under the
# dynamics' JKR. The tolerance has to be expressed in the units of the law that
# runs next.
#
# So the relaxation moved up a level instead of changing its force law. It runs
# in `generate_packing`, AFTER the settle returns, where `compute_forces` is an
# ordinary local call and no import is circular. It relaxes the bed under the
# EXACT force law the dynamics will apply -- JKR, walls, gravity, MC-DEM and
# all -- and stops on a force tolerance measured in multiples of
# `dynamics_load_scale`, the same scale the detector reports. Detector and fix
# therefore cannot disagree about what "in balance" means.
#
# FIRE (Bitzek 2006) rather than more overdamped steps because it is a
# minimiser, not a dynamics: the bed's trajectory during relaxation is not
# physical and need not be, only its endpoint. The velocity projection at wall
# clamps is not optional -- without it `P = F.v` counts motion the clamp
# deleted, FIRE's sign test misfires, and the bed pumps against the wall.
#
# Plain FIRE is NOT monotone, and that is not academic: on a 3D shaped bed with
# `boundary.wall_clamp = force` it drove the energy from 2378 to 4753 nN um and
# the handoff from 264x to 5178x -- it returned a WORSE bed than the settle gave
# it. Stiff wall contacts are what does it; the MD step goes unstable long
# before the uphill test notices. So every step is checked against the energy
# and rejected if it rises (positions restored, dt halved, velocities zeroed),
# and the best configuration seen is restored at the end. `relax_packing` can
# then only improve the bed it was handed -- which is the one property a
# packer-handoff fix must not lack.
#
# Adhesion: JKR pulls, so relaxing under it can draw the bed together rather
# than only apart. That is correct and is the point. If the dynamics' own law
# wants the bed closer, the run would have done it in the first hour anyway;
# doing it here means the run starts from that state instead of spending its
# first hours getting there.

FIRE_DT0 = 0.01            # h, initial MD step
FIRE_DT_MAX = 0.5          # h, ceiling (the dynamics' own default dt)
FIRE_F_INC = 1.1
FIRE_F_DEC = 0.5
FIRE_ALPHA0 = 0.1
FIRE_F_ALPHA = 0.99
FIRE_N_MIN = 5             # uphill-free steps before the step may grow
FIRE_MAX_STEPS = 20000     # a ceiling, not the terminator -- FIRE_STALL_* ends it (~3.4e3 measured)
FIRE_FALLBACK_REDUCTION = 1e-3   # with no load scale, relax this far below the handoff force
# The last granules approach balance asymptotically, so a tight tolerance can burn the whole
# budget for nothing: stop when progress stops and say so -- 'stalled at 1.3x the load' is a
# useful report, 'ran out of steps' is not.
#
# The stall test is on the ENERGY, not on max|F|. max|F| is NOT monotone under FIRE (the
# minimiser is free to load one granule while unloading ten), and keying on it stopped a
# relaxation at 400 steps that was still descending. The energy is the objective and it is
# what Phase 1 made available here. A window whose energy drop is a small fraction of the
# total drop so far has reached the minimum; further steps cannot help.
FIRE_STALL_WINDOW = 200
FIRE_STALL_GAIN = 0.01
# How far uphill a single step may go before it is rejected, as a fraction of the energy
# FIRE started from. FIRE is inertial and SHOULD be allowed small uphill moves -- rejecting
# every one of them stops it dead on the legacy wall clamp, where the clip fights the first
# step. What must be caught is divergence, which is orders of magnitude, not percent.
FIRE_UPHILL_TOL = 0.01


def _fire_masses(gs, p):
    """(m, I) for the FIRE minimiser: the drag coefficients of `step`.

    Any positive mass minimises the same energy -- FIRE is not integrating
    physics. Using the drag coefficients keeps `dt` in the same units as the
    dynamics' own, so `FIRE_DT_MAX` is a number with a meaning.
    """
    N = gs.N
    m = p.drag_scale * gs.r[:N]
    if gs.is_3d:
        I = p.drag_scale_rot * (gs.a[:N] ** 2 + gs.b[:N] ** 2 + gs.c[:N] ** 2) / 3.0
    else:
        I = p.drag_scale_rot * (gs.a[:N] ** 2 + gs.b[:N] ** 2) / 2.0
    return np.maximum(m, 1e-12), np.maximum(I, 1e-20)


def penetration_stats(contacts):
    """mean / p95 / max true penetration on the contact list, um (V3.5).

    Only computable since V3.4: ``overlap`` on a shaped contact record is now
    the MTD penetration depth rather than the distance between two
    common-normal surface points.
    """
    if contacts is None or len(contacts) == 0:
        return {'penetration_mean': 0.0, 'penetration_p95': 0.0,
                'penetration_max': 0.0, 'n_contacts': 0}
    col = getattr(contacts, 'column', None)
    ov = (np.asarray(col('overlap'), dtype=float) if col is not None
          else np.array([c['overlap'] for c in contacts], dtype=float))
    return {'penetration_mean': float(np.mean(ov)),
            'penetration_p95': float(np.percentile(ov, 95)),
            'penetration_max': float(np.max(ov)),
            'n_contacts': int(len(contacts))}


def relax_mode(p, gs=None):
    """Resolve ``packing.relax``: ``'fire'`` or ``'none'`` (V3.5).

    ``auto`` -- the default -- is ``fire`` exactly when
    :func:`dynamics_load_scale` finds something driving the run, and ``none``
    otherwise. That is the right discriminator rather than a guess: a bed with
    no gravity and no cells has a genuinely LOOSE equilibrium under JKR (a
    `consolidation: centre` packing drops from 202 contacts to 19 when
    relaxed), because nothing is holding it together. There is also no force
    scale to stop on, so FIRE would be aiming at a relative target with no
    physical meaning. Beds that do have a load get the fix; beds that do not
    are left alone.
    """
    mode = str(getattr(p, 'packing_relax', 'auto'))
    if mode != 'auto':
        return mode
    if gs is None:
        return 'none'
    # V3.5: do not relax a shaped bed into a projection that will undo it.
    # With `contact.shape_dynamics` off the overlap projection measures overlap
    # between BOUNDING SPHERES, which touch long before the shapes do. On a
    # FIRE-relaxed shaped bed it then fires on 15-17 % of pairs and does work
    # no energy accounts for -- the gradient-flow monitor reports ascents of
    # 4e3-9e3 times the work, against 7-27 with `shape_dynamics` on. `auto`
    # declines; an explicit `relax: fire` is still honoured, with this warning.
    if bool(getattr(p, 'shape_enabled', False)) and not bool(
            getattr(p, 'contact_shape_dynamics', False)):
        print("  packing.relax=auto declined: shaped granules need "
              "contact.shape_dynamics=true, or the bounding-sphere overlap "
              "projection undoes the relaxation (15-17 % of pairs clipped).")
        return 'none'
    scale, _kind = dynamics_load_scale(gs, p)
    return 'fire' if scale else 'none'


def relax_packing(gs, p, rng=None):
    """FIRE-relax the packed bed under the DYNAMICS' force law (V3.5).

    No-op unless ``packing.relax == 'fire'``. Returns the report it also stores
    on ``gs.relax_report`` (and which `save_params_metadata` records), or ``{}``.

    The stopping rule is a FORCE tolerance in the units of the law that runs
    next: ``max|F| <= relax_force_tol * dynamics_load_scale``. With nothing
    driving the run there is no load scale, so it falls back to reducing
    ``max|F|`` by ``FIRE_FALLBACK_REDUCTION`` from where the settle left it --
    relative, because in that case there is no absolute meaning to aim at.

    Translations and rotations are relaxed together as one generalised
    coordinate: the FIRE power test is ``P = sum F.v + sum tau.omega``, so a bed
    that is in force balance but not torque balance does not report success.
    """
    if relax_mode(p, gs) != 'fire' or gs.N < 2:
        return {}
    rng = rng if rng is not None else np.random.default_rng(0)
    N = gs.N
    dim = 3 if gs.is_3d else 2
    periodic = (p.boundary_mode == 'periodic')
    m, I = _fire_masses(gs, p)
    fixed = getattr(gs, 'fixed', None)
    mobile = (~np.asarray(fixed[:N], dtype=bool)) if fixed is not None else np.ones(N, dtype=bool)
    spin = not bool(gs.is_circle)

    def evaluate():
        F, tq, c = (compute_forces_3d if gs.is_3d else compute_forces)(gs, p, rng)
        F = np.array(F[:N], dtype=float)
        F[~mobile] = 0.0
        tq = np.array(tq[:N], dtype=float) if spin else None
        if tq is not None:
            tq[~mobile] = 0.0
        return F, tq, c

    # V3.5: the active noise would make the "force" a random variable and FIRE
    # would chase it forever. Relax the deterministic bed; the noise is part of
    # the dynamics, not of the initial condition.
    T_save = float(getattr(p, 'T_active', 0.0))
    p.T_active = 0.0
    try:
        # The settle has its own wall handling, so the bed it hands over need not
        # be inside the DYNAMICS' clip. Under the legacy +0.5 um standoff the
        # first step teleports every wall granule by the full 0.5 um and injects
        # 65 nN um -- measured. Pay that projection up front, so it is attributed
        # to the clip rather than to FIRE, and so FIRE starts from a state it can
        # descend from. Under `wall_clamp: contact` the displacement is exactly
        # zero, which is one more argument for it.
        _pos_in = gs.pos[:N].copy()
        apply_position_bounds(gs, p)
        entry_shift = float(np.abs(gs.pos[:N] - _pos_in).max())
        F, tq, contacts = evaluate()
        f_max0, n_held0 = free_force_residual(gs, p, F)
        f_p95_0 = free_force_percentile(gs, p, F)
        scale, kind = dynamics_load_scale(gs, p)
        tol_mult = float(getattr(p, 'packing_relax_force_tol', 1.0))
        if scale is None:
            kind = 'relative'
            f_tol = FIRE_FALLBACK_REDUCTION * f_max0
        else:
            f_tol = tol_mult * scale
        pen0 = penetration_stats(contacts)

        v = np.zeros_like(F)
        w = np.zeros_like(tq) if tq is not None else None
        dt = FIRE_DT0
        alpha = FIRE_ALPHA0
        n_pos = 0
        steps = 0
        rejected = 0
        converged = False
        stalled = False
        E0, _parts = system_energy(gs, p, contacts)
        E_window = E0
        E_cur = E0
        E_best = E0
        best = (gs.pos[:N].copy(), gs.theta[:N].copy(),
                gs.quat[:N].copy() if gs.quat is not None else None)
        for _ in range(FIRE_MAX_STEPS):
            f_max, _n_held = free_force_residual(gs, p, F)
            if f_max < f_tol:
                converged = True
                break
            if steps and steps % FIRE_STALL_WINDOW == 0:
                drop = E_window - E_cur
                total = E0 - E_cur
                if drop <= FIRE_STALL_GAIN * max(total, 1e-30):
                    stalled = True
                    break
                E_window = E_cur
            P = float(np.sum(F * v))
            if w is not None:
                P += float(np.sum(tq * w))
            if P > 0.0:
                fnorm = np.sqrt(float(np.sum(F * F)) + (float(np.sum(tq * tq)) if w is not None else 0.0))
                vnorm = np.sqrt(float(np.sum(v * v)) + (float(np.sum(w * w)) if w is not None else 0.0))
                if fnorm > 1e-30:
                    v = (1.0 - alpha) * v + alpha * (F / fnorm) * vnorm
                    if w is not None:
                        w = (1.0 - alpha) * w + alpha * (tq / fnorm) * vnorm
                n_pos += 1
                if n_pos > FIRE_N_MIN:
                    dt = min(dt * FIRE_F_INC, FIRE_DT_MAX)
                    alpha *= FIRE_F_ALPHA
            else:
                v[:] = 0.0
                if w is not None:
                    w[:] = 0.0
                dt *= FIRE_F_DEC
                alpha = FIRE_ALPHA0
                n_pos = 0
                if dt < 1e-9 * FIRE_DT0:
                    break

            prev = (gs.pos[:N].copy(), gs.theta[:N].copy(),
                    gs.quat[:N].copy() if gs.quat is not None else None)
            v += (F / m[:, None]) * dt
            intended = gs.pos[:N, :dim] + v * dt
            gs.pos[:N, :dim] = intended
            steps += 1
            if w is not None:
                w += (tq / (I[:, None] if tq.ndim == 2 else I)) * dt
                if gs.is_3d:
                    for i in range(N):
                        if mobile[i]:
                            gs.quat[i] = quat_integrate(gs.quat[i], w[i], dt)
                else:
                    gs.theta[:N] += np.asarray(w).reshape(-1) * dt
            apply_position_bounds(gs, p)
            # The clamp DELETED motion, so the velocity that produced it is not
            # real. Without this projection `P = F.v` counts it, FIRE's sign
            # test misfires and the bed pumps against the wall. Skipped when
            # periodic: a wrap is a relabelling, not a clamp.
            if not periodic:
                clamped = np.any(np.abs(gs.pos[:N, :dim] - intended) > 1e-12, axis=1)
                if np.any(clamped):
                    v[clamped] = 0.0
            F, tq, contacts = evaluate()
            E_new, _parts = system_energy(gs, p, contacts)
            if E_new > E_cur + FIRE_UPHILL_TOL * abs(E0):
                # Too far uphill for an inertial minimiser: restore, shorten the
                # step and drop the velocity -- the same response FIRE already
                # makes to P < 0, applied to the quantity that actually matters.
                gs.pos[:N] = prev[0]
                gs.theta[:N] = prev[1]
                if prev[2] is not None:
                    gs.quat[:N] = prev[2]
                v[:] = 0.0
                if w is not None:
                    w[:] = 0.0
                dt *= FIRE_F_DEC
                alpha = FIRE_ALPHA0
                n_pos = 0
                rejected += 1
                F, tq, contacts = evaluate()
                if dt < 1e-9 * FIRE_DT0:
                    break
                continue
            E_cur = E_new
            if E_new < E_best:
                E_best = E_new
                best = (gs.pos[:N].copy(), gs.theta[:N].copy(),
                        gs.quat[:N].copy() if gs.quat is not None else None)

        # Hand back the best configuration seen, never the last one.
        if E_best < E_cur:
            gs.pos[:N] = best[0]
            gs.theta[:N] = best[1]
            if best[2] is not None:
                gs.quat[:N] = best[2]
            F, tq, contacts = evaluate()
        f_max, n_held = free_force_residual(gs, p, F)
        f_p95 = free_force_percentile(gs, p, F)
        E_end, _parts = system_energy(gs, p, contacts)
        pen = penetration_stats(contacts)
        # Why it stalled, if it did. A RATTLER has no contacts at all, so its
        # |F| is exactly its own weight and no relaxation can reduce it -- a
        # bed that stalls at ~1x the gravity load with a few rattlers is as
        # relaxed as it can be, and that is a different report from one that
        # stalls with every granule overloaded.
        touched = np.zeros(N, dtype=bool)
        if contacts is not None and len(contacts):
            col = getattr(contacts, 'column', None)
            ii = (np.asarray(col('i'), dtype=np.int64) if col is not None
                  else np.array([c['i'] for c in contacts], dtype=np.int64))
            jj = (np.asarray(col('j'), dtype=np.int64) if col is not None
                  else np.array([c['j'] for c in contacts], dtype=np.int64))
            touched[ii] = True
            touched[jj] = True
        n_rattlers = int(np.sum(mobile & ~touched))
        held = constraint_clamped(gs, p, F)
        n_unbalanced = int(np.sum((np.linalg.norm(F, axis=1) >= f_tol) & ~held))
    finally:
        p.T_active = T_save

    why = 'converged' if converged else ('stalled' if stalled else 'out of steps')
    report = {'relax': 'fire', 'fire_steps': steps, 'fire_converged': bool(converged),
              'fire_stalled': bool(stalled), 'fire_stop_reason': why,
              'fire_f_tol': f_tol, 'fire_load_kind': kind or 'relative',
              'f_max_before': f_max0, 'f_max_after': f_max,
              'energy_before': E0, 'energy_after': E_end,
              'fire_rejected': rejected, 'entry_clip_shift': entry_shift,
              'n_rattlers': n_rattlers, 'n_unbalanced': n_unbalanced,
              'n_wall_clamped': n_held, 'n_wall_clamped_before': n_held0,
              'f_p95_before': f_p95_0, 'f_p95_after': f_p95,
              'ratio_before': (f_max0 / scale) if scale else float('nan'),
              'ratio_after': (f_max / scale) if scale else float('nan'),
              'ratio_p95_before': (f_p95_0 / scale) if scale else float('nan'),
              'ratio_p95_after': (f_p95 / scale) if scale else float('nan'),
              'penetration_before': pen0, 'penetration_after': pen}
    gs.relax_report = report
    print(f"  FIRE relax ({why}, {steps} steps): max|F| {f_max0:.3g} -> {f_max:.3g} nN "
          f"(tol {f_tol:.3g})")
    if scale:
        print(f"    handoff {f_max0 / scale:.3g}x -> {f_max / scale:.3g}x the {kind} load "
              f"(p95 {f_p95_0 / scale:.3g}x -> {f_p95 / scale:.3g}x)")
    print(f"    max penetration {pen0['penetration_max']:.3g} -> "
          f"{pen['penetration_max']:.3g} um, contacts "
          f"{pen0['n_contacts']} -> {pen['n_contacts']}")
    if not converged:
        print(f"    {n_unbalanced} granule(s) still above the tolerance, "
              f"{n_rattlers} of {N} are rattlers (no contacts, so |F| is their own weight); "
              f"{n_held} held by the wall clamp and excluded")
    return report


def apply_gravity(gs, p, F):
    """Add the buoyant weight to the force array (-z in 3D, -y in 2D). No-op unless enabled."""
    if not getattr(p, 'gravity_enabled', False):
        return
    N = gs.N
    axis = 2 if (gs.is_3d and F.shape[1] > 2) else 1
    F[:N, axis] -= granule_weights(gs, p)


WALL_CLAMP_LEGACY_MARGIN = 0.5      # um, the V2.7 standoff


def wall_clamp_margin(gs, p, reach):
    """Signed margin added to a granule's reach when clipping it inside a wall (V3.5).

    Until V3.5 this was a hardcoded ``+0.5`` um: every granule was held half a
    micron clear of every wall, so a granule resting on the floor was **never
    in wall contact** -- the JKR wall force saw a positive gap and did nothing,
    and the granule's whole weight was carried by the clip. The bed rested on a
    numerical shelf, every wall contact force read zero, and every
    residual-force measure had an irreducible floor of one granule weight. See
    :func:`constraint_clamped`, which exists only to report around it.

    ``boundary.wall_clamp`` picks what the clip is for:

    ``contact`` (default)
        Clip at ``reach - max_overlap_frac * reach``, so a granule may sink
        into the wall by the same fraction the engine already allows between
        two granules. The wall contact engages, carries the load, and the clip
        goes back to being a backstop against escape rather than the thing
        holding the bed up.
    ``force``
        Clip only at the wall plane itself (centre inside the box), so the
        contact force does all the work. The neighbour and render grids still
        need positions inside the box, which is the one thing this guarantees.
    ``legacy``
        The V2.7 ``+0.5`` um standoff, bit-for-bit. Pinned in the fixtures.

    Note the SETTLE still uses its own wall handling and its own force law, so
    this changes the dynamics and `relax_packing`, not the packer.
    """
    reach = np.asarray(reach, dtype=float)
    mode = str(getattr(p, 'boundary_wall_clamp', 'contact'))
    if mode == 'legacy':
        return np.full(reach.shape, WALL_CLAMP_LEGACY_MARGIN)
    if mode == 'force':
        return -reach
    frac = float(getattr(p, 'max_overlap_frac', 0.0) or 0.0)
    return -np.minimum(np.maximum(frac, 0.0) * reach, reach)


def apply_position_bounds(gs, p):
    """Keep every granule inside the container after a position update.

    Periodic: wrap. Legacy box with a lid: the V2.7 clip statements verbatim.
    Otherwise: radial projection to the cylinder wall, floor at 0, and for a
    free top a clamp at the container height that is counted in
    ``gs.n_top_clamped`` (the neighbour and render grids need positions inside
    the box). Immobile boundary granules are never moved.
    """
    N = gs.N
    if p.boundary_mode == 'periodic':
        wrap_positions(gs, p)
        return
    rb = gs.r_bound[:N]
    m_rb = wall_clamp_margin(gs, p, rb)          # V3.5, was a hardcoded +0.5
    geom = boundary_geometry(p)
    fixed = getattr(gs, 'fixed', None)
    has_fixed = fixed is not None and bool(np.any(fixed[:N]))
    if geom.shape_code == 0 and not geom.top_free and not has_fixed:
        gs.x[:N] = np.clip(gs.x[:N], rb + m_rb, p.Lx - rb - m_rb)
        gs.y[:N] = np.clip(gs.y[:N], rb + m_rb, p.Ly - rb - m_rb)
        if gs.is_3d:
            gs.z[:N] = np.clip(gs.z[:N], rb + m_rb, p.Lz - rb - m_rb)
        return
    saved = gs.pos[:N][fixed[:N]].copy() if has_fixed else None
    if geom.shape_code == 1:
        dx = gs.x[:N] - geom.cx
        dy = gs.y[:N] - geom.cy
        rho = np.sqrt(dx * dx + dy * dy)
        rho_max = geom.R_cyl - rb - m_rb
        over = rho > rho_max
        if np.any(over):
            scale = rho_max[over] / rho[over]
            gs.x[:N][over] = geom.cx + dx[over] * scale
            gs.y[:N][over] = geom.cy + dy[over] * scale
    else:
        gs.x[:N] = np.clip(gs.x[:N], rb + m_rb, p.Lx - rb - m_rb)
        if gs.is_3d or not geom.top_free:
            gs.y[:N] = np.clip(gs.y[:N], rb + m_rb, p.Ly - rb - m_rb)
    if gs.is_3d:
        up, L_up = gs.z, p.Lz
    else:
        up, L_up = gs.y, p.Ly
    # V3.2: the reach toward the floor and toward the free surface, not the
    # corner radius -- otherwise a flat-lying fragment is held off the floor.
    if gs.is_3d:
        _down = granule_reach(gs, p, N, 0.0, 0.0, -1.0)
        _up = granule_reach(gs, p, N, 0.0, 0.0, 1.0)
    else:
        _down = granule_reach(gs, p, N, 0.0, -1.0)
        _up = granule_reach(gs, p, N, 0.0, 1.0)
    lo = _down + wall_clamp_margin(gs, p, _down)
    hi = L_up - _up - wall_clamp_margin(gs, p, _up)
    if geom.top_free:
        n_over = int(np.count_nonzero(up[:N] > hi))
        if n_over:
            gs.n_top_clamped = int(getattr(gs, 'n_top_clamped', 0)) + n_over
    if gs.is_3d:
        gs.z[:N] = np.clip(gs.z[:N], lo, hi)
    elif geom.top_free:
        gs.y[:N] = np.clip(gs.y[:N], lo, hi)
    if has_fixed:
        gs.pos[:N][fixed[:N]] = saved


def step(gs: GranuleSystem, p: Params, rng, t: float):
    """One overdamped Euler step with cell state evolution and rotation."""
    update_cell_state(gs, p, t, rng)

    # Reset contact clip planes (populated by compute_forces for rendering);
    # the kernel path keeps them as arrays (gs.clip_arrays) instead of lists.
    if kernels_enabled(p):
        gs.clip_arrays = None
    else:
        for k in range(gs.N):
            gs.contact_clips[k] = []

    # Reset deformation force accumulator before force computation
    if p.deformable_enabled and gs.F_eps_accum is not None:
        gs.F_eps_accum[:] = 0.0

    if gs.is_3d:
        F, torques, contacts = compute_forces_3d(gs, p, rng)
    else:
        F, torques, contacts = compute_forces(gs, p, rng)

    # V3.5 gradient flow: the energy of the configuration this step starts from.
    # `contacts` is the one the force evaluation just produced -- never recompute
    # it for this. Off by default and then costs nothing.
    _gf = getattr(p, 'dynamics_gradient_flow', 'off')
    _substep_on = str(getattr(p, 'dynamics_substep', 'off')) != 'off'    # V3.6
    gs.dt_substep = float(p.dt)      # `advance` has already narrowed p.dt if substepping
    _E0 = 0.0
    _parts = None
    _work = 0.0
    if _gf != 'off':
        _E0, _parts = system_energy(gs, p, contacts)

    # Integrate deformation DOFs (implicit Euler, after force computation)
    if p.deformable_enabled and gs.epsilon is not None:
        from gels.lsdem import integrate_deformation_implicit
        integrate_deformation_implicit(gs, p)

    pos_before = gs.pos[:gs.N].copy()             # for the unwrapped-position update below
    _fixed = getattr(gs, 'fixed', None)           # V3.1 immobile boundary lining
    if _fixed is not None and np.any(_fixed[:gs.N]):
        F[:gs.N][_fixed[:gs.N]] = 0.0

    if gs.is_3d:
        # ── 3D integration (vectorized translational, per-granule quaternion) ──
        gamma = p.drag_scale * gs.r[:gs.N]         # (N,)
        if p.contact_semi_implicit or _substep_on:  # V3.2; V3.6 also feeds the controller
            _k = contact_stiffness_per_granule(gs, contacts)
            _rate = _k / np.maximum(gamma, 1e-12)   # 1/h, free of dt
            gs.stiffness_rate_p95 = float(np.percentile(_rate, 95)) if _rate.size else 0.0
            gs.stiffness_rate_max = float(np.max(_rate)) if _rate.size else 0.0
            if p.contact_semi_implicit:
                gamma = gamma + p.dt * _k
        if _gf == 'damped':                         # V3.5: see `_energy_audit`
            gamma = gamma / gs.energy_step_scale
        vel = F[:gs.N] / gamma[:, None]             # (N, 3)
        speed = np.sqrt(np.sum(vel**2, axis=1))     # (N,)
        over = speed > p.v_max
        vel *= _speed_rails(gs, p, speed, contacts, over)[:, None]
        # V3.2 diagnostic: the cap is an effective per-granule force ceiling
        # F_cap = v_max * drag_scale * r (20-60 nN against 180 nN bridges), so a
        # run with many clipped granules has lost force-magnitude information.
        gs.frac_velocity_clipped = float(np.mean(over)) if over.size else 0.0
        gs.vx[:gs.N] = vel[:, 0]
        gs.vy[:gs.N] = vel[:, 1]
        gs.vz[:gs.N] = vel[:, 2]
        dx_step = vel[:, 0] * p.dt
        dy_step = vel[:, 1] * p.dt
        dz_step = vel[:, 2] * p.dt
        gs.x[:gs.N] += dx_step
        gs.y[:gs.N] += dy_step
        gs.z[:gs.N] += dz_step

        # 3D rotational dynamics (quaternion — must be per-granule)
        if not gs.is_circle:
            for i in range(gs.N):
                I_eff = p.drag_scale_rot * (gs.a[i]**2 + gs.b[i]**2 + gs.c[i]**2) / 3.0
                if I_eff > 1e-20:
                    omega = torques[i] / I_eff
                    omega_mag = np.sqrt(omega[0]**2 + omega[1]**2 + omega[2]**2)
                    if omega_mag > p.omega_max_3d:
                        omega *= p.omega_max_3d / omega_mag
                    gs.omega_3d[i] = omega
                    gs.quat[i] = quat_integrate(gs.quat[i], omega, p.dt)

        if _gf != 'off':
            _work = float(np.sum(F[:gs.N, :3] * vel)) * p.dt
            if not gs.is_circle:
                _work += float(np.sum(torques[:gs.N] * gs.omega_3d[:gs.N])) * p.dt

        # Boundary handling
        apply_position_bounds(gs, p)

        # Post-step overlap resolution (V2.1 — prevents granule pass-through)
        _resolve_overlaps(gs, p)
        apply_position_bounds(gs, p)
    else:
        # ── 2D integration (vectorized) ──
        gamma = p.drag_scale * gs.r[:gs.N]         # (N,)
        if p.contact_semi_implicit or _substep_on:  # V3.2; V3.6 also feeds the controller
            _k = contact_stiffness_per_granule(gs, contacts)
            _rate = _k / np.maximum(gamma, 1e-12)   # 1/h, free of dt
            gs.stiffness_rate_p95 = float(np.percentile(_rate, 95)) if _rate.size else 0.0
            gs.stiffness_rate_max = float(np.max(_rate)) if _rate.size else 0.0
            if p.contact_semi_implicit:
                gamma = gamma + p.dt * _k
        if _gf == 'damped':                         # V3.5: see `_energy_audit`
            gamma = gamma / gs.energy_step_scale
        vel = F[:gs.N] / gamma[:, None]             # (N, 2)
        speed = np.sqrt(vel[:, 0]**2 + vel[:, 1]**2)
        over = speed > p.v_max
        vel *= _speed_rails(gs, p, speed, contacts, over)[:, None]
        # V3.2 diagnostic: the cap is an effective per-granule force ceiling
        # F_cap = v_max * drag_scale * r (20-60 nN against 180 nN bridges), so a
        # run with many clipped granules has lost force-magnitude information.
        gs.frac_velocity_clipped = float(np.mean(over)) if over.size else 0.0
        gs.vx[:gs.N] = vel[:, 0]
        gs.vy[:gs.N] = vel[:, 1]
        dx_step = vel[:, 0] * p.dt
        dy_step = vel[:, 1] * p.dt
        gs.x[:gs.N] += dx_step
        gs.y[:gs.N] += dy_step

        # 2D rotational dynamics (superellipses only, vectorized)
        if not gs.is_circle:
            gamma_rot = p.drag_scale_rot * (gs.a[:gs.N]**2 + gs.b[:gs.N]**2) / 2.0
            valid = gamma_rot > 1e-20
            gs.omega[:gs.N] = np.where(valid, torques[:gs.N] / np.where(valid, gamma_rot, 1.0), 0.0)
            gs.omega[:gs.N] = np.clip(gs.omega[:gs.N], -p.omega_max, p.omega_max)
            gs.theta[:gs.N] += gs.omega[:gs.N] * p.dt

        if _gf != 'off':
            _work = float(np.sum(F[:gs.N, :2] * vel)) * p.dt
            if not gs.is_circle:
                _work += float(np.sum(torques[:gs.N] * gs.omega[:gs.N])) * p.dt

        # Boundary handling
        apply_position_bounds(gs, p)

        # Post-step overlap resolution (V2.1 — prevents granule pass-through)
        _resolve_overlaps(gs, p)
        apply_position_bounds(gs, p)

    if _gf != 'off':
        _energy_audit(gs, p, _gf, _E0, _parts, _work)

    _sync_unwrapped(gs, p, pos_before)
    return F, contacts


def advance(gs: GranuleSystem, p: Params, rng, t: float):
    """One COUPLING interval, advanced in as many mechanical substeps as it needs.

    V3.6. With ``dynamics.substep = 'auto'``, ``p.dt`` stops being the
    integration step and becomes the interval at which the run is sampled --
    history rows, snapshots, the console. The mechanics inside it is advanced in
    ``n_sub`` equal substeps chosen by `substep_count` from the stiffness number
    ``S = dt k / gamma``.

    **Everything inside `step` is already a rate times dt** -- the bridge
    formation probability is ``1 - exp(-rate dt)``, the cell clocks are
    ``+= dt``, migration is ``speed dt``, the active noise is
    ``sqrt(2 gamma T / dt)``, a new bridge's maturity is ``dt / t_form``. Every
    one of those composes correctly under subdivision, which is why a substep is
    an ordinary `step` with a smaller ``dt`` rather than a special mechanics-only
    path. The substeps are therefore not an approximation of the outer step;
    the outer step was the approximation.

    ``v_max`` is scaled with the subdivision, so it limits DISPLACEMENT PER
    COUPLING INTERVAL rather than speed per substep. Without that, subdividing
    would silently tighten the absolute rail -- the cap clips 36 % of granules at
    dt = 0.02 h and 49 % at dt = 0.005 h where it clips none at 0.5 h, purely
    because the semi-implicit damping is no longer suppressing the velocity --
    and the run would trade the overlap rail for the velocity rail without
    anyone being told. Scaled, subdividing cannot change how much force
    magnitude the cap discards; it can only improve the accuracy of the path.
    """
    n_sub = substep_count(gs, p)
    gs.n_substeps = n_sub
    if n_sub <= 1:
        return step(gs, p, rng, t)
    dt0, v0 = p.dt, p.v_max
    t0 = t - dt0
    try:
        p.dt = dt0 / n_sub
        p.v_max = v0 * n_sub          # a displacement cap, not a speed cap
        F = contacts = None
        for k in range(n_sub):
            F, contacts = step(gs, p, rng, t0 + (k + 1) * p.dt)
    finally:
        p.dt, p.v_max = dt0, v0
    return F, contacts


def substep_metrics(gs):
    """What the V3.6 controller did, for both twins (flat, finite floats)."""
    n = max(int(getattr(gs, 'n_substeps', 1)), 1)
    dt_sub = float(getattr(gs, 'dt_substep', 0.0))
    return {
        'n_substeps': n,
        'dt_substep': dt_sub,
        # S = dt_sub * k/gamma: the factor by which the semi-implicit step is
        # slowing the granule down. Below ~0.2 the rate is the contact law's;
        # at 41 (the V3.5 default) it is the integrator's.
        'stiffness_number': float(getattr(gs, 'stiffness_rate_p95', 0.0)) * dt_sub,
        'stiffness_number_max': float(getattr(gs, 'stiffness_rate_max', 0.0)) * dt_sub,
        'stiffness_rate_p95': float(getattr(gs, 'stiffness_rate_p95', 0.0)),
        'substep_budget_bound': bool(getattr(gs, 'substep_budget_bound', False)),
        'frac_outlier_clipped': float(getattr(gs, 'frac_outlier_clipped', 0.0)),
    }


# ══════════════════════════════════════════════════════════════════════
# Phase field rendering (from particle positions)
# ══════════════════════════════════════════════════════════════════════

def _apply_clips_2d(sdf, X, Y, cx, cy, clips):
    """Apply JKR half-plane clips to a 2D SDF array (in-place max).

    Each clip is a tuple (nx, ny, clip_d) defining a half-plane constraint:
        sdf = max(sdf, dot(x - center, n) - clip_d)
    This creates flat faces at contact regions, matching JKR contact geometry.
    """
    if not clips:
        return sdf
    for clip in clips:
        nx_c, ny_c, clip_d = clip[0], clip[1], clip[2]
        plane = (X - cx) * nx_c + (Y - cy) * ny_c - clip_d
        np.maximum(sdf, plane, out=sdf)
    return sdf


def _apply_clips_3d(sdf, X, Y, Z, cx, cy, cz, clips):
    """Apply JKR half-plane clips to a 3D SDF array (in-place max).

    Each clip is a tuple (nx, ny, nz, clip_d) defining a half-space constraint.
    """
    if not clips:
        return sdf
    for clip in clips:
        nx_c, ny_c, nz_c, clip_d = clip[0], clip[1], clip[2], clip[3]
        plane = (X - cx) * nx_c + (Y - cy) * ny_c + (Z - cz) * nz_c - clip_d
        np.maximum(sdf, plane, out=sdf)
    return sdf


# ══════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════

def kozeny_carman_permeability(porosity, d_grain_um):
    """
    Kozeny-Carman permeability estimate.
    K = eps^3 * d^2 / [180 * (1-eps)^2]   [µm²]
    """
    eps = np.clip(porosity, 0.01, 0.99)
    return eps**3 * d_grain_um**2 / (180.0 * (1.0 - eps)**2)


def rcp_fraction_superellipsoid(mean_aspect_ratio):
    """
    Estimate random close packing fraction for superellipsoids.
    For spheres: phi_RCP ~ 0.64.
    For ellipsoids: increases with aspect ratio (Donev et al. 2004).
    """
    # Empirical fit from Donev et al.: phi_RCP ~ 0.64 + 0.08*(AR - 1) up to AR ~ 2
    ar = max(1.0, mean_aspect_ratio)
    return min(0.74, 0.64 + 0.08 * (ar - 1.0))


def compute_displacement(gs, x0, y0, z0=None):
    """RMS displacement from initial positions.
    Uses unwrapped positions for correct displacement under periodic BCs.
    """
    # Use unwrapped coords if available (periodic BCs accumulate wraps)
    dx = gs.x_unwrap - x0; dy = gs.y_unwrap - y0
    if gs.is_3d and z0 is not None:
        dz = gs.z_unwrap - z0
        disp = np.sqrt(dx**2 + dy**2 + dz**2)
    else:
        disp = np.sqrt(dx**2 + dy**2)
    return float(np.mean(disp[gs.func_mask])), float(np.mean(disp[gs.inert_mask]))


# ══════════════════════════════════════════════════════════════════════
# Data serialization (V1.5, updated V2.6)
# ══════════════════════════════════════════════════════════════════════

def check_disk_space(output_dir):
    """Check available disk space for the output directory.

    Returns 'ok' (>= 5 GB), 'warning' (1-5 GB), or 'critical' (< 1 GB).
    """
    import shutil
    try:
        usage = shutil.disk_usage(output_dir)
        free_gb = usage.free / (1024 ** 3)
        if free_gb < 1.0:
            return 'critical'
        elif free_gb < 5.0:
            return 'warning'
        return 'ok'
    except OSError:
        return 'ok'  # can't check, assume ok



def _atomic_savez(path, **arrays):
    """np.savez_compressed to `path` via a temporary file + os.replace.

    A reader tailing the directory (the live viewer, --continue after a crash)
    never sees a half-written .npz. The temporary name has no .npz suffix, so
    it is written through an open handle (np.savez would append '.npz').
    """
    from gels.io.writer import write_npz_atomic
    write_npz_atomic(path, arrays, compresslevel=6)


def save_snapshot_to_disk(snap_idx, gs, p, t, F, output_dir,
                           save_fields=False, phi_f=None, phi_i=None, phi_v=None,
                           contacts=None, writer=None):
    """Save one timepoint of simulation data to disk as compressed .npz.

    ``writer`` (gels.io.SnapshotWriter, V3.0) hands the copied arrays to a
    background thread; without it the file is written synchronously. Either
    way the file appears atomically (.tmp + rename) at deflate level
    ``perf_io_compresslevel``.
    """
    from gels.io.writer import write_npz_atomic
    snap_dir = os.path.join(output_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)
    level = int(getattr(p, 'perf_io_compresslevel', 1))

    def _emit(path, arrays):
        if writer is not None:
            writer.submit(path, arrays)
        else:
            write_npz_atomic(path, arrays, compresslevel=level)

    # Granule data
    data = {
        'time': np.float64(t),
        'x': gs.x.copy(), 'y': gs.y.copy(), 'z': gs.z.copy(),
        'r': gs.r.copy(), 'gtype': gs.gtype.copy(),
        'species_id': gs.species_id.copy(), 'f': gs.f.copy(),
        'a': gs.a.copy(), 'b': gs.b.copy(), 'c': gs.c.copy(),
        'n1': gs.n1.copy(), 'n2': gs.n2.copy(),
        'vx': gs.vx.copy(), 'vy': gs.vy.copy(), 'vz': gs.vz.copy(),
        'n_attached': gs.n_attached.copy(),
        'spread_fraction': gs.spread_fraction.copy(),
        'fa_maturity': gs.fa_maturity.copy(),
        'n_overcrowded': gs.n_overcrowded.copy(),
        'n_cells': gs.n_cells.copy(),
        'E_gran': gs.E_gran.copy(), 'nu_gran': gs.nu_gran.copy(),
        'force_x': F[:, 0].copy(),
        'force_y': F[:, 1].copy(),
    }
    if F.shape[1] > 2:
        data['force_z'] = F[:, 2].copy()
    if gs.is_3d and gs.quat is not None:
        data['quat'] = gs.quat.copy()
    else:
        data['theta'] = gs.theta.copy()

    # Per-cell data (V1.5)
    data['cell_granule_id'] = gs.cell_granule_id.copy()
    data['cell_state'] = gs.cell_state.copy()
    data['cell_theta_local'] = gs.cell_theta_local.copy()
    data['cell_eta_local'] = gs.cell_eta_local.copy()
    data['cell_omega_local'] = gs.cell_omega_local.copy()
    data['cell_fx'] = gs.cell_fx.copy()
    data['cell_fy'] = gs.cell_fy.copy()
    data['cell_fz'] = gs.cell_fz.copy()
    data['cell_bridge_target'] = gs.cell_bridge_target.copy()
    data['cell_bridge_age'] = gs.cell_bridge_age.copy()
    data['cell_bridge_locked'] = gs.cell_bridge_locked.copy()
    data['cell_overcrowd_age'] = gs.cell_overcrowd_age.copy()
    data['cell_alignment'] = gs.cell_alignment.copy()
    data['cell_contact_area'] = gs.cell_contact_area.copy()
    data['cell_offset'] = gs.cell_offset.copy()
    # V3.1
    data['cell_bridge_gap_prev'] = gs.cell_bridge_gap_prev.copy()
    data['cell_bridge_force'] = gs.cell_bridge_force.copy()
    data['cell_age'] = gs.cell_age.copy()
    data['cell_generation'] = gs.cell_generation.copy()
    data['cell_cycle_time'] = gs.cell_cycle_time.copy()          # V3.2
    data['fixed'] = gs.fixed.copy()
    data['n_divisions_cum'] = np.int64(getattr(gs, 'n_divisions_cum', 0))
    _geom = boundary_geometry(p)
    if _geom.top_free or _geom.shape_code == 1:
        _edges, _phi = phi_z_profile(gs, p)
        data['phi_z_edges'] = _edges
        data['phi_z_profile'] = _phi

    # V1.5.2: Per-contact data for stress visualization (V3.0: ContactSoA fast path)
    if contacts is not None and len(contacts) > 0 and hasattr(contacts, 'column'):
        data['contact_i'] = np.asarray(contacts.column('i'), dtype=np.int32)
        data['contact_j'] = np.asarray(contacts.column('j'), dtype=np.int32)
        for name in ('cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'overlap', 'R_eff', 'F_normal', 'A_contact'):
            data['contact_' + name] = np.asarray(contacts.column(name), dtype=np.float64)
    elif contacts:
        n_c = len(contacts)
        data['contact_i'] = np.array([c['i'] for c in contacts], dtype=np.int32)
        data['contact_j'] = np.array([c['j'] for c in contacts], dtype=np.int32)
        data['contact_cx'] = np.array([c['cx'] for c in contacts])
        data['contact_cy'] = np.array([c['cy'] for c in contacts])
        data['contact_cz'] = np.array([c.get('cz', 0.0) for c in contacts])
        data['contact_nx'] = np.array([c['nx'] for c in contacts])
        data['contact_ny'] = np.array([c['ny'] for c in contacts])
        data['contact_nz'] = np.array([c.get('nz', 0.0) for c in contacts])
        data['contact_overlap'] = np.array([c['overlap'] for c in contacts])
        data['contact_R_eff'] = np.array([c['R_eff'] for c in contacts])
        data['contact_F_normal'] = np.array([c['F_normal'] for c in contacts])
        data['contact_A_contact'] = np.array([c['A_contact'] for c in contacts])

    # V2.6: LS-DEM deformation state (needed for resume)
    if gs.epsilon is not None:
        data['epsilon'] = gs.epsilon.copy()
        data['d_epsilon'] = gs.d_epsilon.copy()

    _emit(os.path.join(snap_dir, f'snap_{snap_idx:04d}.npz'), data)

    # Optional phase fields (large in 3D)
    if save_fields and phi_f is not None:
        fields_dir = os.path.join(output_dir, 'fields')
        os.makedirs(fields_dir, exist_ok=True)
        _emit(os.path.join(fields_dir, f'fields_{snap_idx:04d}.npz'),
              {'phi_f': phi_f.copy(), 'phi_i': phi_i.copy(), 'phi_v': phi_v.copy()})


def save_history_to_disk(hist, output_dir):
    """Save scalar metrics history as CSV and JSON."""
    os.makedirs(output_dir, exist_ok=True)
    if not hist:
        return

    # CSV (pandas-friendly). Use the union of keys across all entries, in
    # first-seen order: entries can differ (e.g. a resumed run whose earlier
    # history was written by a different code path), and keying off hist[0]
    # alone made DictWriter raise at the very end of a completed run, losing
    # both the CSV and the history.json written after it.
    keys = list(dict.fromkeys(k for entry in hist for k in entry))
    csv_path = os.path.join(output_dir, 'history.csv')
    with open(csv_path + '.tmp', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(hist)
    os.replace(csv_path + '.tmp', csv_path)

    # JSON (exact fidelity); written atomically — it is rewritten at every save
    json_path = os.path.join(output_dir, 'history.json')
    with open(json_path + '.tmp', 'w') as f:
        json.dump(hist, f, indent=2, default=float)
    os.replace(json_path + '.tmp', json_path)


def save_params_metadata(p, gs, output_dir, seed):
    """Save Params and run metadata as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Params as dict
    from dataclasses import asdict
    params_dict = asdict(p)
    with open(os.path.join(output_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f, indent=2, default=float)

    # Metadata
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = 'unknown'

    meta = {
        'version': 'V1.6',
        'git_hash': git_hash,
        'seed': seed,
        'mode': p.mode,
        'n_granules': int(gs.N),
        'n_functional': int(np.sum(gs.func_mask)),
        'n_inert': int(np.sum(gs.inert_mask)),
        'total_cells': int(gs.total_cells),
        'species': [
            {'index': k, 'name': gs.species_names[k], 'f': float(gs.species_f[k]),
             'color': gs.species_colors[k], 'E_kPa': float(gs.species_E[k]),
             'poisson_ratio': float(gs.species_nu[k]),
             'n_granules': int(np.sum(gs.species_id == k)),
             'n_cells': int(np.sum(gs.n_cells[gs.species_id == k]))}
            for k in range(gs.K)],
        'domain': [p.Lx, p.Ly, p.Lz],
        'boundary': {'mode': p.boundary_mode, 'shape': getattr(p, 'boundary_shape', 'box'),
                     'top': getattr(p, 'boundary_top', 'wall'),
                     'functionalization': float(getattr(p, 'boundary_functionalization', 0.0)),
                     'R_cyl_um': float(boundary_geometry(p).R_cyl),
                     'bed_height_um': (float(bed_surface(gs, p)['bed_height_mean'])
                                       if getattr(p, 'boundary_top', 'wall') == 'free' else None)},
        'gravity': {'enabled': bool(getattr(p, 'gravity_enabled', False)),
                    'medium_density_kg_m3': float(getattr(p, 'medium_density', 1000.0)),
                    'species_density_kg_m3': [float(v) for v in gs.species_rho]},
        'n_boundary': int(np.sum(gs.fixed)),
        'cell_state_enum': {s.name: int(s.value) for s in CellState},
        # V3.4: is the packed bed in force balance for the DYNAMICS? Present
        # only once forces have been evaluated at t = 0.
        'handoff': getattr(gs, 'handoff_report', None),
        'relax': getattr(gs, 'relax_report', None),      # V3.5 FIRE
    }
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def create_archive(output_dir):
    """Create a .tar.gz archive of the output directory for HPC transfer."""
    archive_name = output_dir.rstrip('/') + '.tar.gz'
    parent = os.path.dirname(output_dir) or '.'
    basename = os.path.basename(output_dir)
    with tarfile.open(archive_name, 'w:gz') as tar:
        tar.add(output_dir, arcname=basename)
    print(f"  Archive: {archive_name}")
    return archive_name


# ══════════════════════════════════════════════════════════════════════
# Data loading (V1.5)
# ══════════════════════════════════════════════════════════════════════

def load_params(run_dir):
    """Params of a saved run from <run_dir>/params.json (defaults where absent).

    Values are coerced to the field types; legacy alias names are accepted
    through the alias properties; species are resolved (old runs get the
    legacy two-species table).
    """
    params_path = os.path.join(run_dir, 'params.json')
    p = Params()
    if os.path.exists(params_path):
        with open(params_path) as f:
            params_dict = json.load(f)
        for k, v in params_dict.items():
            if hasattr(p, k):
                field_type = type(getattr(p, k))
                try:
                    setattr(p, k, field_type(v))
                except (ValueError, TypeError):
                    setattr(p, k, v)
    resolve_species(p)   # V3.0: old runs get the legacy two-species table
    return p


def load_run(run_dir):
    """Load a saved simulation run from disk.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or a .tar.gz archive.

    Returns
    -------
    hist : list of dict
        Scalar metrics time series.
    snaps : list of dict
        Per-timepoint snapshot dicts (numpy arrays).
    p : Params
        Simulation parameters.
    metadata : dict
        Run metadata (version, git hash, seed, etc.).
    """
    # Handle .tar.gz input
    if run_dir.endswith('.tar.gz'):
        extract_dir = run_dir[:-7]  # strip .tar.gz
        if not os.path.isdir(extract_dir):
            with tarfile.open(run_dir, 'r:gz') as tar:
                tar.extractall(path=os.path.dirname(run_dir) or '.')
        run_dir = extract_dir

    p = load_params(run_dir)

    # Load metadata
    metadata = {}
    meta_path = os.path.join(run_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    # Load history
    hist = []
    hist_json = os.path.join(run_dir, 'history.json')
    hist_csv = os.path.join(run_dir, 'history.csv')
    if os.path.exists(hist_json):
        with open(hist_json) as f:
            hist = json.load(f)
    elif os.path.exists(hist_csv):
        with open(hist_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {}
                for k, v in row.items():
                    try:
                        entry[k] = int(v)
                    except ValueError:
                        try:
                            entry[k] = float(v)
                        except ValueError:
                            entry[k] = v
                hist.append(entry)

    # Load snapshots
    snaps = []
    snap_dir = os.path.join(run_dir, 'snapshots')
    if os.path.isdir(snap_dir):
        snap_files = sorted(
            f for f in os.listdir(snap_dir)
            if f.startswith('snap_') and f.endswith('.npz'))
        for sf in snap_files:
            data = dict(np.load(os.path.join(snap_dir, sf), allow_pickle=False))
            # Check for corresponding field file
            idx_str = sf.replace('snap_', '').replace('.npz', '')
            field_file = os.path.join(run_dir, 'fields', f'fields_{idx_str}.npz')
            if os.path.exists(field_file):
                fields = dict(np.load(field_file, allow_pickle=False))
                data.update(fields)
            snaps.append(data)

    return hist, snaps, p, metadata


def load_cells(run_dir, snap_index=None):
    """Load per-cell data from a specific snapshot (or all snapshots).

    Parameters
    ----------
    run_dir : str
        Path to the output directory.
    snap_index : int or None
        If given, load only that snapshot. Otherwise load all.

    Returns
    -------
    dict or list of dict
        Cell data arrays keyed by 'cell_*' names.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')

    if snap_index is not None:
        path = os.path.join(snap_dir, f'snap_{snap_index:04d}.npz')
        data = dict(np.load(path, allow_pickle=False))
        return {k: data[k] for k in data if k.startswith('cell_')}

    results = []
    snap_files = sorted(
        f for f in os.listdir(snap_dir)
        if f.startswith('snap_') and f.endswith('.npz'))
    for sf in snap_files:
        data = dict(np.load(os.path.join(snap_dir, sf), allow_pickle=False))
        results.append({k: data[k] for k in data if k.startswith('cell_')})
    return results


# ══════════════════════════════════════════════════════════════════════
# Resume from snapshot (V2.6)
# ══════════════════════════════════════════════════════════════════════

def find_last_snapshot(run_dir):
    """Find the last (highest-numbered) snapshot file in a run directory.

    Returns the absolute path to the last snap_XXXX.npz file.
    Raises FileNotFoundError if none found.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')
    if not os.path.isdir(snap_dir):
        raise FileNotFoundError(f"No snapshots/ directory in {run_dir}")
    snap_files = sorted(
        f for f in os.listdir(snap_dir)
        if f.startswith('snap_') and f.endswith('.npz'))
    if not snap_files:
        raise FileNotFoundError(f"No snap_*.npz files in {snap_dir}")
    return os.path.join(snap_dir, snap_files[-1])


def restore_gs_from_snapshot(snap_path, p):
    """Restore a GranuleSystem from a saved snapshot .npz file.

    Parameters
    ----------
    snap_path : str
        Path to a snap_XXXX.npz file.
    p : Params
        Simulation parameters (needed for mode, LS-DEM config).

    Returns
    -------
    gs : GranuleSystem
        Fully restored granule system.
    t_resume : float
        Simulation time at the snapshot.
    snap_idx : int
        Snapshot index (parsed from filename).
    """
    data = dict(np.load(snap_path, allow_pickle=False))
    t_resume = float(data['time'])

    # Parse snap index from filename: snap_XXXX.npz -> XXXX
    basename = os.path.basename(snap_path)
    snap_idx = int(basename.replace('snap_', '').replace('.npz', ''))

    # Construct GranuleSystem
    gs = GranuleSystem(
        x=data['x'], y=data['y'], r=data['r'],
        gtype=data['gtype'], n_cells=data['n_cells'],
        z=data['z'] if 'z' in data else None,
        a=data['a'], b=data['b'],
        c=data.get('c', None),
        n1=data['n1'], n2=data['n2'],
        theta=data.get('theta', None),
        quat=data.get('quat', None),
        mode=p.mode,
        species_id=data.get('species_id', None),   # V2.7 snapshots: derived from gtype
        p=p,
    )

    # GranuleSystem flags any system built with explicit a/b as non-circular,
    # but a restored packing of circles/spheres must keep the circle fast-path
    # it had before the run was interrupted. is_circle gates contact detection
    # and the shape descriptor metrics, so getting it wrong on resume both
    # slows the run down and changes which metrics land in history.
    gs.is_circle = bool(
        np.allclose(gs.a, gs.r) and np.allclose(gs.b, gs.r)
        and np.allclose(gs.n1, 2.0) and np.allclose(gs.n2, 2.0)
        and (not gs.is_3d or np.allclose(gs.c, gs.r))
    )

    # Restore velocity state
    gs.vx[:] = data['vx']
    gs.vy[:] = data['vy']
    gs.vz[:] = data['vz']

    # Restore per-granule cell state
    gs.n_attached[:] = data['n_attached']
    gs.spread_fraction[:] = data['spread_fraction']
    gs.fa_maturity[:] = data['fa_maturity']
    gs.n_overcrowded[:] = data['n_overcrowded']

    # Restore per-cell arrays
    gs.cell_granule_id[:] = data['cell_granule_id']
    gs.cell_state[:] = data['cell_state']
    gs.cell_theta_local[:] = data['cell_theta_local']
    gs.cell_eta_local[:] = data.get('cell_eta_local', np.zeros(gs.total_cells))
    gs.cell_omega_local[:] = data.get('cell_omega_local', np.zeros(gs.total_cells))
    gs.cell_fx[:] = data['cell_fx']
    gs.cell_fy[:] = data['cell_fy']
    gs.cell_fz[:] = data.get('cell_fz', np.zeros(gs.total_cells))
    gs.cell_bridge_target[:] = data['cell_bridge_target']
    gs.cell_bridge_age[:] = data['cell_bridge_age']
    gs.cell_bridge_locked[:] = data['cell_bridge_locked']
    gs.cell_overcrowd_age[:] = data['cell_overcrowd_age']
    gs.cell_alignment[:] = data['cell_alignment']
    gs.cell_contact_area[:] = data['cell_contact_area']
    gs.cell_offset[:] = data['cell_offset']
    # V3.1 arrays (absent in older snapshots -> registry fill values)
    n_c = gs.total_cells
    gs.cell_bridge_gap_prev[:] = data.get('cell_bridge_gap_prev', np.full(n_c, np.nan))
    gs.cell_bridge_force[:] = data.get('cell_bridge_force', np.zeros(n_c))
    gs.cell_age[:] = data.get('cell_age', np.zeros(n_c))
    gs.cell_generation[:] = data.get('cell_generation', np.zeros(n_c, dtype=np.int8))
    gs.cell_cycle_time[:] = data.get('cell_cycle_time', np.zeros(n_c))       # V3.2
    if 'fixed' in data:
        gs.fixed[:] = np.asarray(data['fixed'], dtype=bool)
    gs.n_divisions_cum = int(data.get('n_divisions_cum', 0))

    # Unwrapped positions: reset to current (displacement tracked from resume point)
    gs.x_unwrap[:] = gs.x
    gs.y_unwrap[:] = gs.y
    gs.z_unwrap[:] = gs.z

    # LS-DEM: re-init SDF grids from params, then restore deformation state
    if p.deformable_enabled:
        from gels.lsdem import init_lsdem
        init_lsdem(gs, p)
        if 'epsilon' in data:
            gs.epsilon[:] = data['epsilon']
            gs.d_epsilon[:] = data['d_epsilon']
        else:
            print("  WARNING: Resuming deformable run but snapshot has no epsilon data.")
            print("           Starting deformation from zero.")

    return gs, t_resume, snap_idx


# ══════════════════════════════════════════════════════════════════════
# Main simulation loop
# ══════════════════════════════════════════════════════════════════════

def _warmup_jit(is_3d=False):
    """Trigger Numba JIT compilation of hot-path functions with dummy data."""
    a, b, n = 40.0, 35.0, 2.5
    # 2D geometry + NR solver
    superellipse_point(0.5, a, b, n)
    superellipse_normal_vec(0.5, a, b, n)
    superellipse_curvature_radius(0.5, a, b, n)
    se2d_mtd_core(0.0, 0.0, a, b, n, 0.0, 100.0, 0.0, a, b, n, 0.0)
    se2d_wall_core(50.0, 50.0, a, b, n, 0.1, 0.0, 0, 1)
    hertz_contact_force(5000.0, 20.0, 1.0)
    se2d_support(0.6, 0.8, a, b, n)          # V3.4 support leaves
    support_R_eff_2d(0.6, 0.8, a, b, n)
    if is_3d:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        superellipsoid_point(0.3, 0.5, a, b, a, n, n)
        superellipsoid_normal(0.3, 0.5, a, b, a, n, n)
        quat_rotate(q, np.array([1.0, 0.0, 0.0]))
        find_contact_spheres_3d(0.0, 0.0, 0.0, a, 100.0, 0.0, 0.0, a)
        se3d_mtd_core(0.0, 0.0, 0.0, a, b, a, n, n, q,
                      100.0, 0.0, 0.0, a, b, a, n, n, q)
        se3d_support(0.6, 0.0, 0.8, a, b, a, n, n)   # V3.4 support leaves
        support_R_eff_3d(0.6, 0.0, 0.8, a, b, a, n, n)
        se3d_wall_core(0.0, 0.0, 0.0, a, b, a, n, n, q, -100.0, 2, 1)


def _snapshot_dict(gs, F, pf, pi, pv, contacts):
    """In-memory snapshot dict (the run() return value / observer payload); copies every array."""
    snap = {
        'phi_f': pf.copy(), 'phi_i': pi.copy(), 'phi_v': pv.copy(),
        'x': gs.x.copy(), 'y': gs.y.copy(), 'z': gs.z.copy(),
        'r': gs.r.copy(), 'gtype': gs.gtype.copy(),
    'species_id': gs.species_id.copy(), 'f': gs.f.copy(),
        'n_attached': gs.n_attached.copy(),
        'spread_fraction': gs.spread_fraction.copy(),
        'fa_maturity': gs.fa_maturity.copy(),
        'n_overcrowded': gs.n_overcrowded.copy(),
        'a': gs.a.copy(), 'b': gs.b.copy(), 'c': gs.c.copy(),
        'n1': gs.n1.copy(), 'n2': gs.n2.copy(),
        'mode': gs.mode,
        # V1.5.1: Force and cell data for viz scripts
        'force_x': F[:, 0].copy(),
        'force_y': F[:, 1].copy(),
        'cell_state': gs.cell_state.copy(),
        'cell_granule_id': gs.cell_granule_id.copy(),
        'cell_theta_local': gs.cell_theta_local.copy(),
        'cell_fx': gs.cell_fx.copy(),
        'cell_fy': gs.cell_fy.copy(),
        'cell_bridge_target': gs.cell_bridge_target.copy(),
        'cell_bridge_age': gs.cell_bridge_age.copy(),
        'cell_bridge_locked': gs.cell_bridge_locked.copy(),
        'cell_overcrowd_age': gs.cell_overcrowd_age.copy(),
        'cell_alignment': gs.cell_alignment.copy(),
        'cell_contact_area': gs.cell_contact_area.copy(),
        'cell_offset': gs.cell_offset.copy(),
        # V3.1
        'cell_bridge_gap_prev': gs.cell_bridge_gap_prev.copy(),
        'cell_bridge_force': gs.cell_bridge_force.copy(),
        'cell_age': gs.cell_age.copy(),
        'cell_generation': gs.cell_generation.copy(),
        'cell_cycle_time': gs.cell_cycle_time.copy(),   # V3.2
        'fixed': gs.fixed.copy(),
        'n_divisions_cum': int(getattr(gs, 'n_divisions_cum', 0)),
        # V1.5.2: Per-contact data for stress visualization
        'contacts': contacts if contacts is not None else [],
    }
    if F.shape[1] > 2:
        snap['force_z'] = F[:, 2].copy()
        snap['cell_fz'] = gs.cell_fz.copy()
        snap['cell_eta_local'] = gs.cell_eta_local.copy()
        snap['cell_omega_local'] = gs.cell_omega_local.copy()
    if gs.is_3d and gs.quat is not None:
        snap['quat'] = gs.quat.copy()
    else:
        snap['theta'] = gs.theta.copy()
    # Store n_shape for 2D viz compat
    snap['n_shape'] = gs.n1.copy()
    # V2.3: LS-DEM deformation state
    if gs.epsilon is not None:
        snap['epsilon'] = gs.epsilon.copy()
        snap['d_epsilon'] = gs.d_epsilon.copy()
    return snap


def run(p=None, seed=None, observer=None):
    """Run a simulation.

    observer : gels.live.Observer, callable or None (V3.0)
        Called at start, after every step, after every saved snapshot and at
        the end. ``on_step`` may return True to stop the run gracefully (a
        final snapshot is written). Zero overhead when None.
    """
    if p is None: p = Params()
    from gels.live.observer import as_observer
    observer = as_observer(observer)
    # V3.0: numba threading layer for the compiled kernels (the thread count is
    # chosen once the granule count is known — see below). No effect on results.
    from gels.kernels import configure_threads
    configure_threads(p.perf_threads if p.perf_threads != 0 else 1, p.perf_threading_layer)
    # Re-derive phi targets in case overrides were applied after __post_init__
    if p.phi_solid_target > 0:
        p.phi_f_target = p.phi_solid_target * p.func_ratio
        p.phi_i_target = p.phi_solid_target * (1.0 - p.func_ratio)
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"  Using random seed: {seed}")
    rng = np.random.default_rng(seed)

    # JIT warmup: trigger Numba compilation before timing begins
    if HAS_NUMBA and p.shape_enabled:
        _warmup_jit(p.mode == "3D")

    # ── V2.6: Resume or fresh start ──
    resume_t = 0.0
    resume_snap_idx = -1  # -1 means fresh start (snap_counter will be 0)
    hist = []

    if p.resume_from:
        print(f"\n  RESUMING from: {p.resume_from}")
        snap_path = find_last_snapshot(p.resume_from)
        gs, resume_t, resume_snap_idx = restore_gs_from_snapshot(snap_path, p)
        print(f"  Restored state at t={resume_t:.2f} h (snapshot {resume_snap_idx})")
        # Load existing history if available
        hist_json = os.path.join(p.resume_from, 'history.json')
        if os.path.exists(hist_json):
            with open(hist_json) as _hf:
                hist = json.load(_hf)
            print(f"  Loaded {len(hist)} history entries")
    else:
        print(f"\n  Mode: {p.mode}")
        print("  Generating packing...")
        # Mode-aware packing generation
        if p.mode == "3D":
            gs = generate_packing_3d(p, seed=seed)
        elif p.mode == "2D-slice":
            gs = generate_packing_2d_slice(p, seed=seed)
        else:
            gs = generate_packing(p, seed=seed)
        # V2.3: Initialize LS-DEM deformable particles if enabled
        if p.deformable_enabled:
            from gels.lsdem import init_lsdem
            init_lsdem(gs, p)

    # Thread count for this system size (perf_threads = 0: auto; more threads
    # than ~N/1000 only add fork/join overhead on the many small kernels).
    configure_threads(p.perf_threads, p.perf_threading_layer, n_granules=gs.N)

    # Store initial positions (displacement tracked from this point)
    x0 = gs.x.copy(); y0 = gs.y.copy()
    z0 = gs.z.copy() if gs.is_3d else None

    n_steps = int(p.t_total / p.dt)
    snaps = []

    # V1.5: Initialize output directory and save metadata
    output_dir = p.output_dir
    snap_counter = [resume_snap_idx + 1]  # continues from resume point
    if p.save_data:
        os.makedirs(output_dir, exist_ok=True)
        save_params_metadata(p, gs, output_dir, seed)

    s_cur = [0]   # current step, for observer.on_save

    # V3.0: snapshots go to disk from a background thread (bounded queue: the
    # run only waits when the disk falls perf_io_queue_depth files behind).
    writer = None
    if p.save_data and getattr(p, 'perf_async_io', True):
        from gels.io.writer import SnapshotWriter
        writer = SnapshotWriter(depth=getattr(p, 'perf_io_queue_depth', 2),
                                compresslevel=getattr(p, 'perf_io_compresslevel', 1))
    keep_snap_dicts = bool(p.perf_keep_snaps_in_memory) or observer is not None

    def save(t, F, contacts=None):
        # V3.0: render one grid per species, collapse to the legacy
        # phi_f / phi_i / phi_v for metrics and snapshots.
        phi_s = render_fields_species(gs, p)
        pf, pi, pv = group_species_fields(gs, phi_s)
        m = compute_metrics(gs, p, pf, pi, pv, t, F, phi_s=phi_s)
        del phi_s
        df, di = compute_displacement(gs, x0, y0, z0)
        m['disp_func'] = df; m['disp_inert'] = di
        m.update(compute_displacement_species(gs, x0, y0, z0))
        hist.append(m)
        if p.save_data:
            # V3.0: rewrite history at every save so an interrupted or crashed
            # run keeps its metrics (it used to be written only at the end).
            save_history_to_disk(hist, output_dir)
        # Snapshot dict (RAM): only when kept for the caller or handed to an observer
        snap = _snapshot_dict(gs, F, pf, pi, pv, contacts) if keep_snap_dicts else None
        if snap is not None and p.perf_keep_snaps_in_memory:
            snaps.append(snap)

        snap_idx_this = len(hist) - 1   # snapshot number (== snap_NNNN when saving; counts resumed saves)
        # V1.5: Save to disk (V2.6: with disk space guard and rolling cleanup)
        if p.save_data:
            disk_status = check_disk_space(output_dir)
            if disk_status == 'critical':
                print(f"  CRITICAL: < 1 GB disk space — SKIPPING snapshot "
                      f"{snap_counter[0]} at t={t:.1f} h")
            else:
                use_fields = p.save_fields and disk_status != 'warning'
                if disk_status == 'warning':
                    print(f"  WARNING: < 5 GB disk space — saving WITHOUT fields")
                save_snapshot_to_disk(
                    snap_counter[0], gs, p, t, F, output_dir,
                    save_fields=use_fields,
                    phi_f=pf, phi_i=pi, phi_v=pv,
                    contacts=contacts, writer=writer)
                snap_counter[0] += 1

        if observer is not None:
            observer.on_save(snap_idx_this, s_cur[0], t, snap, m)
        return m

    # ── Print header ──
    _hdr = (f"\n  {'t(h)':>6} {'f_cl':>5} {'f_lf':>6} {'v_cl':>5} "
            f"{'tissue':>7} {'bridges':>7} {'attach':>7} {'spread':>6} "
            f"{'FA_mat':>6} {'disp_f':>7} {'K_KC':>8} "
            f"{'bCells':>6} {'bF_avg':>6} {'lock':>4}")

    def _print_metrics(t, m):
        print(f"  {t:6.1f} {m['func_nc']:5d} {m['func_lf']:6.2f} "
              f"{m['void_nc']:5d} {m['tissue_frac']:7.3f} "
              f"{m['n_bridges']:7d} {m['n_attached_total']:7.0f} "
              f"{m['mean_spread_frac']:6.2f} {m['mean_fa_maturity']:6.2f} "
              f"{m['disp_func']:7.1f} {m['K_kozeny_carman']:8.1f} "
              f"{m['n_bridging_cells']:6d} {m['bridge_force_mean']:6.1f} "
              f"{m['n_locked_in_cells']:4d}")

    # ── Initial save (only for fresh starts) ──
    resume_step = int(round(resume_t / p.dt))
    if observer is not None:
        observer.on_start(gs, p, seed, n_steps, resume_step)
    if not p.resume_from:
        update_cell_state(gs, p, 0.0, rng)
        if gs.is_3d:
            F0, _, contacts0 = compute_forces_3d(gs, p, rng)
        else:
            F0, _, contacts0 = compute_forces(gs, p, rng)
        handoff_force_balance(gs, p, F0)                 # V3.4
        # V3.6: seed the substep controller from the bed it is about to be
        # handed, so step 1 is already sized. Without this it ramps from 1 under
        # SUBSTEP_GROWTH and a short run never reaches its target at all.
        _rate0 = stiffness_rate(gs, p, contacts0)
        gs.stiffness_rate_p95 = float(np.percentile(_rate0, 95)) if _rate0.size else 0.0
        gs.stiffness_rate_max = float(np.max(_rate0)) if _rate0.size else 0.0
        gs.n_substeps = substep_count(gs, p)
        if getattr(p, 'dynamics_gradient_flow', 'off') != 'off':   # V3.5
            _energy_audit(gs, p, p.dynamics_gradient_flow,
                          *system_energy(gs, p, contacts0), 0.0)
        if p.save_data:
            save_params_metadata(p, gs, output_dir, seed)   # now carries the handoff
        m = save(0.0, F0, contacts0)
        print(_hdr)
        _print_metrics(0.0, m)
    else:
        print(f"  Resuming from step {resume_step + 1} "
              f"(t={resume_t:.2f} h) → step {n_steps} (t={p.t_total:.2f} h)")
        print(_hdr)

    wall_t0 = timer.time()
    _sub_said = [False]          # V3.6 substep reporting
    _sub_bound = [False]
    _sub_n = []
    t = resume_t
    stop_reason = None
    s_last = resume_step
    F = None
    contacts = None
    try:
        for s in range(resume_step + 1, n_steps + 1):
            t += p.dt
            s_cur[0] = s
            F, contacts = advance(gs, p, rng, t)      # V3.6: substeps if asked
            s_last = s

            if gs.n_substeps > 1 and not _sub_said[0]:
                _sub_said[0] = True                       # V3.6: say it once, not 144 times
                _r = float(getattr(gs, 'substep_rate_used', 0.0))
                print(f"  Substepping the mechanics: {gs.n_substeps} x "
                      f"{p.dt / gs.n_substeps:.4g} h per {p.dt:g} h interval "
                      f"(S = {_r * p.dt / gs.n_substeps:.2f}, "
                      f"target {p.dynamics_substep_target:g}); at dt alone S would be "
                      f"{_r * p.dt:.0f}, and the contact rate wrong by that factor")
            _sub_bound[0] = _sub_bound[0] or bool(getattr(gs, 'substep_budget_bound', False))
            _sub_n.append(gs.n_substeps)

            if s % p.save_every == 0:
                m = save(t, F, contacts)
                _print_metrics(t, m)

            # V3.0: observer may request a graceful stop
            if observer is not None and observer.on_step(s, t, gs, F, contacts):
                stop_reason = 'stopped'
                break
    except KeyboardInterrupt:
        # V3.0: Ctrl+C keeps what was computed instead of losing the run
        stop_reason = 'interrupted'
        print("\n  Interrupted — writing the current state so the run can be resumed...")

    if stop_reason is not None and F is not None and s_last % p.save_every != 0:
        # final snapshot at the stop step so --continue resumes exactly here
        m = save(t, F, contacts)
        _print_metrics(t, m)

    if writer is not None:
        io_errors = writer.close()
        for path, err in io_errors:
            print(f"  WARNING: snapshot write failed for {path}: {err}")

    if _sub_n and max(_sub_n) > 1:
        print(f"  Substeps: mean {float(np.mean(_sub_n)):.0f}, max {max(_sub_n)} "
              f"per {p.dt:g} h interval")
        if _sub_bound[0]:
            print(f"  WARNING: the substep budget (dynamics.substep_max="
                  f"{p.dynamics_substep_max}) bound before the target S="
                  f"{p.dynamics_substep_target:g} was reached, so this run is NOT converged "
                  f"in dt. Raise substep_max, or accept a rate error of about the reported "
                  f"stiffness_number.")

    elapsed = timer.time() - wall_t0
    if stop_reason is None:
        print(f"\n  Done in {elapsed:.1f}s ({n_steps} steps, {gs.N} granules)")
    else:
        print(f"\n  Run {stop_reason} at step {s_last}/{n_steps} (t={t:.2f} h) after "
              f"{elapsed:.1f}s; state saved ({gs.N} granules)")

    # ── Bridge force diagnostic (V1.9) ──
    if hist:
        final = hist[-1]
        n_bc = final.get('n_bridging_cells', 0)
        bf_mean = final.get('bridge_force_mean', 0.0)
        n_lock = final.get('n_locked_in_cells', 0)
        if n_bc > 0:
            print(f"\n  Bridge force diagnostic:")
            print(f"    Bridging cells:   {n_bc}")
            print(f"    Locked-in cells:  {n_lock}")
            print(f"    Mean bridge force: {bf_mean:.2f} nN")
            print(f"    Expected force:    {p.expected_bridge_force:.1f} nN")
            ratio = bf_mean / p.expected_bridge_force if p.expected_bridge_force > 0 else 0.0
            if ratio < 0.25:
                print(f"    WARNING: Bridge force is {ratio:.0%} of expected "
                      f"({p.expected_bridge_force:.0f} nN). Motor-clutch parameters "
                      f"may need tuning (n_motors={p.n_motors}, "
                      f"F_motor_stall={p.F_motor_stall}, F_max_per_cell={p.F_max_per_cell}).")
            elif ratio < 0.5:
                print(f"    NOTE: Bridge force is {ratio:.0%} of expected. "
                      f"Consider increasing n_motors or F_motor_stall.")

    # V1.5: Save history and create archive
    if p.save_data:
        save_history_to_disk(hist, output_dir)
        # Append wall-clock time to metadata
        meta_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(meta_path):
            with open(meta_path) as _mf:
                meta = json.load(_mf)
            meta['wall_time_s'] = round(elapsed, 1)
            meta['n_steps'] = n_steps
            meta['steps_completed'] = int(s_last)
            meta['t_reached'] = float(t)
            meta['stopped_early'] = bool(stop_reason)
            meta['stop_reason'] = stop_reason or 'complete'
            with open(meta_path, 'w') as _mf:
                json.dump(meta, _mf, indent=2)
        n_saved = snap_counter[0]
        print(f"  Data: {n_saved} snapshots saved to {output_dir}/")
        if p.compress_archive:
            create_archive(output_dir)

    if observer is not None:
        observer.on_end(hist, gs, stop_reason or 'complete')
    return hist, snaps, p, gs


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*65)
    print("  GELS V1.6: Cell-Driven Granular Rearrangement")
    print("="*65)

    p = Params()
    if p.mode == "3D":
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} × {p.Lz:.0f} µm (3D)")
    elif p.mode == "2D-slice":
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm (2D slice of "
              f"{p.Lx:.0f}×{p.Ly:.0f}×{p.Lz:.0f} 3D)")
    else:
        print(f"\n  Domain: {p.Lx:.0f} × {p.Ly:.0f} µm (2D)")
    print(f"  R_func={p.R_func_mean:.0f}±{p.R_func_std:.0f} µm, "
          f"R_inert={p.R_inert_mean:.0f}±{p.R_inert_std:.0f} µm")
    if p.cell_surface_coverage > 0:
        print(f"  Cell surface coverage={p.cell_surface_coverage:.2f} "
              f"({'stacked' if p.cell_surface_coverage > 1 else 'monolayer'}), "
              f"d_cell={p.cell_diameter} µm, h_spread={p.cell_height_spread} µm")
    else:
        print(f"  Cells/granule={p.n_cells_per_granule}, "
              f"d_cell={p.cell_diameter} µm, h_spread={p.cell_height_spread} µm")
    print(f"  Motor-clutch: {p.n_motors} motors × {p.F_motor_stall*1e3:.0f} pN, "
          f"{p.n_clutches} clutches, k_c={p.k_clutch} nN/µm")
    print(f"  Attach onset={p.t_attach_onset} h, "
          f"sense dist={p.cell_sense_distance} µm")
    print_stiffness_info(p)
    print(f"  Friction: τ₀_ii={p.tau_0_ii} Pa, τ₀_if={p.tau_0_if} Pa, "
          f"τ₀_ff={p.tau_0_ff} Pa")
    print(f"  Adhesion: W_ii={p.W_adh_ii*1e3:.1f}, W_if={p.W_adh_if*1e3:.1f}, "
          f"W_ff={p.W_adh_ff*1e3:.1f} mJ/m²")
    print(f"  Simulation: {p.t_total:.0f} h, dt={p.dt:.2f} h")

    hist, snaps, p, gs = run(p)

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
    print(f"  Contacts: {hf['n_contacts']} total "
          f"(ff={hf['n_contacts_ff']}, if={hf['n_contacts_if']}, "
          f"ii={hf['n_contacts_ii']})")
    print(f"    max δ/R = {hf['max_overlap_ratio']*100:.1f}%")
    print(f"  Area conservation: {hf['area_conservation']:.4f} "
          f"(overlap area = {hf['total_overlap_area']:.1f} µm²)")
    print(f"  Cell state:")
    print(f"    Attached: {hf['n_attached_total']:.0f} / {hf['n_seeded_total']:.0f}")
    print(f"    Spread: {hf['mean_spread_frac']:.0%}, FA maturity: {hf['mean_fa_maturity']:.0%}")
    print(f"    Overcrowded: {hf['n_overcrowded_total']:.0f}")
    print(f"  Transport (V1.4):")
    print(f"    Porosity: {h0['porosity']:.3f} → {hf['porosity']:.3f}")
    print(f"    K (Kozeny-Carman): {h0['K_kozeny_carman']:.1f} → "
          f"{hf['K_kozeny_carman']:.1f} µm²")
    print(f"    Compaction ratio: {h0['compaction_ratio']:.3f} → "
          f"{hf['compaction_ratio']:.3f} (φ_RCP={hf['phi_RCP']:.3f})")
    print(f"    Darcy number: {hf['Da_number']:.2e}")