# GELS Architecture Document

**Version:** V3.6 (in progress)
**Last updated:** 2026-09-19
**Primary source files:** `gels/engine.py`, `gels/kernels/reference.py`,
`gels/convergence.py`, `gels/celltypes/`, `gels/laguerre.py`, `gels/pore.py`

> **Reading order.** Sections 1–4 describe the shape of the system and are
> broadly stable. Sections 2.5, 2.8 and 2.10 were rewritten for V3.4–V3.6 and
> are the ones where the physics actually changed: the contact law is **JKR**,
> not Hertz; shaped contact detection is the **support-function MTD** solver,
> not a common normal; and `dt` is a **coupling interval** above which
> `advance()` substeps the mechanics, not the integration step. "Modules added
> V3.2–V3.6" and "The four numerical rails" in section 5 are the map of
> everything after V3.0. `CLAUDE.md` is the authority on defaults and their
> reasons; `CodeLog/Updates/CHANGELOG.md` on what changed when.

---

## 1. System Overview

GELS simulates the rearrangement of hydrogel granular scaffolds driven by
cell-mediated forces. It models two populations of granules --- functional (cell-laden)
and inert (passive) --- within a confined domain, using overdamped Langevin dynamics.

The simulator operates in three modes:
- **2D**: Pure 2D simulation with superellipse granules (V1.3 behavior)
- **2D-slice**: Generate 3D superellipsoid packing, slice at z-midplane, run 2D
- **3D**: Full volumetric simulation with superellipsoid granules and quaternion orientation

The simulator predicts scaffold microstructure evolution (void network topology,
functional connectivity, tissue formation, transport properties) over experimentally
relevant timescales (24--72 hours).

```
┌──────────────────────────────────────────────────────────────────────┐
│                        gels/engine.py                                │
│                                                                      │
│  ┌──────────┐   ┌────────────────────┐   ┌────────────────────────┐ │
│  │  Params   │──▶│ generate_packing*()│──▶│    GranuleSystem       │ │
│  │ (config)  │   │  2D / 3D / slice   │   │  (mode-aware state)    │ │
│  └──────────┘   └────────────────────┘   └──────────┬─────────────┘ │
│                                                      │               │
│  ┌───────────────────────────────────────────────────▼────────────┐  │
│  │                        run() loop                              │  │
│  │                                                                │  │
│  │   ┌───────────────────────┐    ┌────────────────────────────┐  │  │
│  │   │ compute_forces*()     │───▶│     step()                 │  │  │
│  │   │  • JKR contact (V2.3) │    │  • update_cell_state       │  │  │
│  │   │  • LS-DEM (gels/lsdem.py)│    │  • deformation integration │  │  │
│  │   │  • motor-clutch       │    │  • overdamped Euler (2D/3D)│  │  │
│  │   │  • wall (4 or 6 face) │    │  • velocity cap            │  │  │
│  │   │  • friction + torques │    │  • wall clamp              │  │  │
│  │   │  • contact clips      │    │  • quaternion integration  │  │  │
│  │   └───────────────────────┘    └────────────────────────────┘  │  │
│  │                                                                │  │
│  │   ┌───────────────────────┐    ┌────────────────────────────┐  │  │
│  │   │ render_fields*()      │───▶│ compute_metrics()          │  │  │
│  │   │  • 2D: tanh profiles  │    │  • connectivity            │  │  │
│  │   │  • 3D: volumetric     │    │  • overlap stats           │  │  │
│  │   │  • eff. radii         │    │  • Kozeny-Carman, Darcy    │  │  │
│  │   │  • bbox clipping (3D) │    │  • RCP, compaction ratio   │  │  │
│  │   │  • deformed SDF (V2.3)│    │  • deformation strain      │  │  │
│  │   └───────────────────────┘    └────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
            │ saves data to disk (snapshots, fields, history, params)
            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  viz/postprocess.py (V2.3)                             │
│                  Unified post-processing orchestrator                 │
│                                                                      │
│  Delegates to viz/ package:                                          │
│  viz/compaction.py     Void-space evolution, packing, compaction     │
│  viz/percolation.py    Darcy, Kozeny-Carman, dimensionless groups    │
│  viz/movies.py         Rotating GIF, timelapse, z-sweep, composite   │
│  viz/phases.py         Phase isosurfaces, fractions, tri-plane       │
│  viz/cells.py          Cell morphology, stress maps, GIFs (V1.5.1)  │
│  viz/stress.py         3D surface stress, isosurfaces, GIFs (V1.5.2)│
│  viz/shapes.py         Granule shape gallery (superellipses/oids)    │
│  viz/doe.py            DOE statistical analysis & visualization      │
│  viz/dimensionless.py  Dimensionless analysis, data collapse (V1.7)  │
│  viz/scaffold_evolution.py    2D microstructure evolution (V1.9)      │
│  viz/scaffold_evolution_3d.py 3D volumetric evolution, PyVista (V1.9) │
│  viz/energy_landscape.py      Energy landscape visualization (V1.11)  │
│                                                                      │
│  Delegates to analysis/ package:                                     │
│  analysis/mean_field_model.py   Two-zone compaction ODE (V1.9)       │
│  analysis/parameter_sweep.py    LHS + 1D PDE sweep (V1.10)           │
│  analysis/energy_landscape.py   Free energy decomposition (V1.11)    │
│  analysis/coarse_grain.py       Stress tensor, viscosity (V1.7)      │
│  analysis/tissue_descriptors.py Tissue architecture vector (V1.7)    │
│  analysis/organ_targets.py      Organ targets + phase mapping (V1.9) │
│  analysis/arch_distance.py      Distance to organ targets (V1.7)     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│              viz2/postprocess.py (V2.4) — NEW                         │
│              Rebuilt 2D/3D visualization system                       │
│                                                                      │
│  viz2/common.py           Shared colors, drawing, 3D slicing         │
│  viz2/voronoi.py          Voronoi engine (center + boundary modes)   │
│  viz2/scaffold_map.py     Module 1: Vector scaffold map              │
│  viz2/voronoi_shapes.py   Module 2: Voronoi shape factors + overlay  │
│  viz2/phase_fractions.py  Module 3: Global + local phase fractions   │
│  viz2/movies.py           Module 4: GIF animations (all evolutions)  │
│  viz2/energy_stress.py    Module 5: 6 energy modes + stress/strain   │
│  viz2/void_percolation.py Module 6: Void percolation theory          │
│  viz2/scaffold_map_3d.py  Module 7: PyVista 3D rendering (optional)  │
│  viz2/postprocess.py      Orchestrator (CLI + programmatic)          │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│          analysis/ — Mathematical Analysis Framework (V1.7+)          │
│                                                                      │
│  mean_field_model.py    Two-zone ODE: dx_f/dt = -x_f*(σ_cell-σ_r)/η │
│  parameter_sweep.py     11-D LHS + 1D PDE, organ prediction (V1.10) │
│  energy_landscape.py    6-term free energy decomposition (V1.11)     │
│  coarse_grain.py        Love-Weber stress, strain rate, η_eff        │
│  tissue_descriptors.py  BV/TV, Tb.Th, Tb.Sp, SMI, MIL, tortuosity  │
│  organ_targets.py       7 organ targets + phase mapping (V1.9)       │
│  arch_distance.py       Weighted Mahalanobis distance to organs      │
│  spatial_pde.py         1D radial PDE compaction model (V2.5)        │
│  contact_network.py     Graph-based topology model (V2.5)            │
│  contact_network_model.py Gillespie SSA stochastic model (V2.5)     │
│  MATHEMATICAL_MODEL.md  Formal model document for publications       │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│       Mesoscale Modelling Hierarchy (V2.5)                            │
│                                                                      │
│  Level 1: Mean-field ODE (mean_field_model.py)                       │
│    • 1 DOF (x_f scalar), milliseconds, no spatial info               │
│                                                                      │
│  Level 2a: Spatial PDE (spatial_pde.py)  ◄──► Level 2b               │
│    • N_x=20 radial grid points           │                           │
│    • Compaction waves, local jamming      │                           │
│    • Porosity/permeability profiles       │                           │
│                                           │                           │
│  Level 2b: Contact Network (contact_network.py) ◄──► Level 2a       │
│    • Graph G=(V,E), N nodes              │                           │
│    • Bridge percolation, Z(t), clusters  │                           │
│    • Force chain statistics              │                           │
│    • Deterministic per-timestep stepping │                           │
│                                                                      │
│  Level 2c: Stochastic Network (contact_network_model.py)            │
│    • Gillespie SSA for exact bridge kinetics (BKL 1975)             │
│    • Union-find percolation tracking (Newman-Ziff 2001)             │
│    • Monte Carlo ensemble: 1000 runs → mean, IQR, spanning prob    │
│    • ~2.5s per 72h realization, seconds for full ensemble           │
│                                                                      │
│  Level 3: Full DEM (gels/engine.py)                                  │
│    • N × (pos + orient + cells + deform), hours                      │
│    • Full particle resolution, superellipsoids, LS-DEM               │
│                                                                      │
│  viz/mesoscale.py: Kymographs, profiles, network plots, combined     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Breakdown

### 2.1 Configuration — `Params` (dataclass)

| Group | Parameters | Units | Purpose |
|-------|-----------|-------|---------|
| Mode | `mode` | -- | `"2D"`, `"2D-slice"`, or `"3D"` |
| Boundary | `boundary_mode` | -- | `"walls"` (default) or `"periodic"` (V1.10) |
| Container shape | `boundary_shape` | -- | `"box"` (default) or `"cylinder"` (3D only; axis z, R = min(Lx,Ly)/2) (V3.1) |
| Container top | `boundary_top` | -- | `"wall"` (default) or `"free"` (open top: z up in 3D, y up in 2D) (V3.1) |
| Wall chemistry | `boundary_functionalization`, `boundary_layer_*` | -- | coated wall; optional immobile granule lining cells can bridge to (V3.1) |
| Gravity | `gravity_enabled`, `granule_density`, `medium_density` | kg/m^3 | buoyant weight per mobile granule, -z (3D) / -y (2D) (V3.1) |
| Consolidation | `packing_consolidation` | -- | `centre` (V2.7 pull to the box centre) / `gravity` / `none` / `auto` (V3.1) |
| Granule amount | `bed_height_um`, `bed_phi_assumed` | um | state the amount as a settled bed height (V3.1) |
| Contact cap | `contact_E_cap`, `friction_mu` | kPa, -- | numerical cap on the CONTACT modulus + Coulomb friction, for rigid materials (V3.1) |
| Bridge force | `bridge_force_model`, `cell_contraction_speed` | -- , um/h | `constant` (V2.7 actuator) or `hill` (force-velocity) (V3.1) |
| Cell division | `cell_division_enabled`, `cell_doubling_time` | -- , h | population growth with contact inhibition (V3.1) |
| Domain | `Lx`, `Ly`, `Lz` | um | Simulation box size (Lz used in 3D/2D-slice) |
| Granule sizes | `R_func_mean/std`, `R_inert_mean/std` | um | Gaussian size distributions |
| Composition | `phi_f_target`, `phi_i_target` | -- | Target area/volume fractions |
| Composition (alt) | `phi_solid_target`, `func_ratio` | -- | Total solid + functional ratio (V1.9) |
| Cell geometry | `n_cells_per_granule`, `cell_diameter`, `cell_height_spread`, `cell_coverage` | --, um, um, -- | Cell seeding (20 um spheres, 5 um spread height) |
| Cell timeline | `t_attach_onset`, `t_attach_half`, `t_spread_duration`, `fa_maturation_rate` | h, h, h, 1/h | Attachment/spreading/FA kinetics |
| Motor-clutch | `n_motors`, `F_motor_stall`, `n_clutches`, `k_clutch`, `k_on_clutch`, `k_off_clutch` | --, nN, --, nN/um, 1/s, 1/s | Chan & Odde 2008 force model |
| Cell bridging | `cell_sense_distance`, `F_max_per_cell`, `L_rest` | um, nN, um | Filopodia range, force cap, rest length |
| Bridge kinetics | `bridge_attempt_rate`, `bridge_formation_time`, `bridge_senescence_time`, `min_fa_for_bridge`, `bridge_break_gap`, `bridge_lock_force_threshold`, `bridge_secondary_rate_mult`, `expected_bridge_force` | 1/h, h, h, --, um, nN, --, nN | Stochastic bridge initiation, maturity ramp, senescence, lock-in, secondary migration (V1.5.2, V1.9) |
| Contact mechanics | `E_modulus`, `poisson_ratio` | kPa, -- | Hertzian contact (V1.1) |
| Friction | `tau_0_ii/if/ff`, `friction_v_ref` | Pa, µm/h | Area-dependent hydrogel friction (V1.2) |
| Adhesion | `W_adh_ii/if/ff` | J/m² | DMT adhesion by pair type (V1.2) |
| Shape (2D) | `shape_enabled`, `aspect_ratio_func/inert_mean/std`, `blockiness_func/inert_mean/std` | --, --, -- | Superellipse shape (V1.3) |
| Shape (3D) | `aspect_ratio_c_func/inert_mean/std`, `blockiness_n2_func/inert_mean/std` | --, -- | Superellipsoid c-axis ratio and meridional blockiness (V1.4) |
| Drag | `eta`, `drag_scale`, `drag_scale_rot` | Pa-s, --, -- | Overdamped dynamics |
| Noise | `T_active` | nN-um | Active temperature (functional only) |
| Time | `dt`, `t_total`, `save_every_h`, `v_max` | h, h, h, um/h | Integration control |
| Rendering | `Ngrid`, `Ngrid_3d`, `interface_width` | --, --, um | Phase field grid (2D and 3D) |
| Performance | `use_numba` | -- | Optional Numba JIT acceleration |

### 2.2 State Container — `GranuleSystem`

Mode-aware container. Stores all per-granule arrays:

**Common (all modes):**
- **Position**: `x`, `y` (float64); `z` (float64, 3D only)
- **Radius**: `r` (float64, constant throughout simulation)
- **Type**: `gtype` (int, 0=functional, 1=inert)
- **Cell count**: `n_cells` (float64)
- **Cell state** (V1.2): `n_attached`, `spread_fraction`, `fa_maturity`, `n_overcrowded`
- **Per-cell tracking** (V1.5): flat arrays indexed by `cell_offset[i]:cell_offset[i+1]`
  - `cell_granule_id`, `cell_state` (CellState enum), `cell_theta_local`/`cell_eta_local`/`cell_omega_local`
  - `cell_fx`/`cell_fy`/`cell_fz` (force vector), `cell_bridge_target`, `cell_bridge_age`, `cell_contact_area`
- **Velocity state**: `vx`, `vy` (float64); `vz` (float64, 3D only)
- **Shape semi-axes**: `a`, `b` (float64); `c` (float64, 3D only)
- **Blockiness**: `n_shape` (equatorial n1); `n1`, `n2` (3D mode)
- **Bounding radius**: `r_bound = max(a, b)` in 2D, `max(a, b, c)` in 3D
- **Fast-path flag**: `is_circle` (bool)

**2D only:**
- `theta` (float64) — orientation angle (rad)
- `omega` (float64) — angular velocity (rad/h)

**3D only:**
- `quat` (N,4 float64) — unit quaternion (w,x,y,z)
- `omega_3d` (N,3 float64) — angular velocity vector (rad/h)

**Properties:**
- `is_3d` — True if mode is "3D"
- `positions()` — returns (N,2) in 2D or (N,3) in 3D
- `func_mask`, `inert_mask` — boolean arrays

### 2.3 Quaternion Utilities (V1.4)

Pure functions for quaternion math:

| Function | Purpose |
|----------|---------|
| `quat_multiply(q1, q2)` | Hamilton product |
| `quat_conjugate(q)` | Inverse rotation |
| `quat_normalize(q)` | Enforce unit length |
| `quat_rotate(q, v)` | Rotate vector by quaternion |
| `quat_rotate_inv(q, v)` | Inverse rotation (world → body) |
| `quat_to_rotation_matrix(q)` | 3×3 rotation matrix |
| `quat_from_axis_angle(axis, angle)` | Create from axis-angle |
| `quat_random(rng)` | Uniform random orientation |
| `quat_integrate(q, omega, dt)` | Integrate angular velocity |

### 2.4 Superellipsoid Geometry (V1.4)

3D shape functions for `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1`:

| Function | Purpose |
|----------|---------|
| `superellipsoid_volume(a,b,c,n1,n2)` | Exact volume (Jaklic formula) |
| `superellipsoid_point(eta,omega,...)` | Parametric surface point |
| `superellipsoid_normal(eta,omega,...)` | Outward unit normal |
| `superellipsoid_curvature_radii(...)` | Principal curvature radii (for Hertz R_eff) |
| `superellipsoid_implicit(...)` | Inside/outside test in body frame |
| `superellipsoid_implicit_world(...)` | Inside/outside test in world frame (uses quaternion) |
| `superellipsoid_mesh(...)` | Triangle mesh for rendering |

### 2.5 Contact Mechanics — JKR since V2.3

**`hertz_contact_force` is no longer the force law**; it is retained for
reference and for the V1.x comparison. The live law is JKR adhesive contact:

#### `jkr_force_from_overlap(delta_um, R_eff_um, E_star_Pa, W_Jm2) -> (F, a)`
`F = (4/3) E* a^3/R* - sqrt(8 pi W E* a^3)`, with the contact radius `a` solved
from the overlap by Newton-Raphson. Reduces exactly to Hertz at `W = 0`. Returns
the contact radius as well, which several later features consume without a
second solve: the semi-implicit stiffness (`2 E_s a`), the wall stiffness (V3.6)
and the JKR contact ENERGY (V3.5).

#### `jkr_contact_energy(a, R*, E*, W)` (V3.5)
`U(a) = (8/15) E* a^5/R*^2 - (4/3) sqrt(2 pi W E*) a^(7/2)/R* + pi W a^2`, the
potential whose gradient is the force above. `dU/d(delta) = F` to 1e-10 across
the JKR range.

#### `hertz_contact_force(E_star_Pa, R_eff_um, delta_um) -> float`
Hertzian normal force: `F = (4/3) E* sqrt(R*) delta^(3/2)`.
Unit conversion factor of 1e-3 maps (Pa, um, um) → nN.

#### `overlap_lens_area(R1, R2, d) -> float`
Exact 2D lens-shaped intersection area between two circles.

#### `compute_effective_radii(gs) -> (r_eff, overlap_area)`
Volume-conserving display radii: `r_eff = sqrt(r² + ΔA/π)`.

### 2.6 Cell Geometry & Motor-Clutch Model (V1.2)

#### `motor_clutch_force(E_kPa, p, fa_maturity)`
Steady-state traction per cell (Chan & Odde 2008):
`F_mc = F_stall * k_sub/(k_sub+k_opt) * engagement * fa_maturity`

#### `update_cell_state(gs, p, t)`
Per-step lifecycle: attachment → spreading → FA maturation → overcrowding

### 2.7 Packing Generators

| Generator | Mode | Description |
|-----------|------|-------------|
| `generate_packing()` | 2D | 2D RSA with superellipse shapes |
| `generate_packing_3d()` | 3D | 3D RSA with superellipsoid shapes, random quaternion orientation |
| `generate_packing_2d_slice()` | 2D-slice | Calls `generate_packing_3d()`, slices at z=Lz/2 via `slice_superellipsoid_z()` |

### 2.8 Contact Detection — support-function MTD since V3.4

**The common-normal solver was replaced, not tuned.** It detected **12 %** of
true contacts at 2 % past first touch, over-reported penetration by a median
**9.8x**, and reported the normal BACKWARDS for ~50 % of 3D shaped contacts --
and since force is applied as `-F n`, that was attraction. The retired bodies
survive under `*_cn` names, called by nothing, as evidence for the defect tests.

The support function of the two-exponent superellipsoid is closed form -- a
nested dual norm `h(n) = ||(||(a nx, b ny)||_q1, c nz)||_q2` with `q = n/(n-1)` --
so `sep(n) = (c2-c1).n - h1(n) - h2(n)` is concave, `delta = -max sep` is a true
penetration depth, and the force is the gradient of an energy.

- **Circle / sphere fast path** — unchanged analytical overlap.
- `find_contact_superellipses` / `find_contact_superellipsoids_3d` — **are** the
  MTD solver; there is no solver flag. `grad h` IS the support point, so the
  contact point is free, `(eta, omega)` invert in closed form, and
  `R_eff = sqrt(det grad^2 h)` on the tangent plane is exact to ~1e-11 against
  the old scheme's ~1e-3.
- **`sep(n0) > 0` PROVES separation** and returns immediately, which is why the
  new solver is 4.5x faster at realistic neighbour-list occupancy while finding
  3.3x more contacts. The old one ran 15 unconditional iterations per candidate.
- **Walls are the easy case**: a wall is a half-space, so
  `penetration = h(-w) - (c-q).w` and the contact point `= c + grad h(-w)` are
  exact in ONE support evaluation. This replaced six brute-force samplers that
  under-reported penetration by a median 0.118 um -- the size of the overlaps
  being resolved.

`contact.curvature_R_cap` defaults to 2.0 because the new `R_eff` at a flat face
is the true ~1e15 um; it cannot bind for spheres.

### 2.9 Force Computation

| Force | Scope | Law | Key parameters |
|-------|-------|-----|----------------|
| **Contact (normal)** | All overlapping pairs | **JKR** (V2.3; Hertz+DMT before) | `E_modulus`, `poisson_ratio`, `W_adh_*`, `contact_E_cap` |
| **Contact (tangential)** | All overlapping pairs | Area-dependent friction | `tau_0_*`, `friction_v_ref` |
| **Contact torque** | Non-spherical granules | τ = (contact_pt − centre) × F | Off-centre contacts |
| **Cell bridging** | Functional pairs with spreading/proliferating cells | Motor-clutch (probabilistic, maturity-ramped, lock-in) | `bridge_attempt_rate`, `bridge_formation_time`, `min_fa_for_bridge`, `bridge_lock_force_threshold` (V1.9) |
| **Wall** | Granules penetrating boundary (walls mode) | JKR on a half-space, from the support function (V3.4) | 4 walls (2D), 6 faces or floor+cylinder (3D); `contact.wall_torque`; disabled in periodic mode |
| **MC-DEM** | Soft granules with many contacts | Confinement factor `kappa` on the repulsion (V2.2) | `mc_dem_enabled`, `mc_dem_kappa_max`. **Not a gradient of any energy** — it is most of the V3.5 audit's residual for shapes |
| **Gravity** | All granules (V3.1) | Buoyant weight `(rho_g - rho_m) g V` | `gravity_enabled`, per-species `density_kg_m3` |
| **Active noise** | Functional granules only | Gaussian white noise | `T_active` |

Dispatch: `compute_forces()` → 2D path, `compute_forces_3d()` → 3D path.

### 2.10 Time Integration — `step()`, and `advance()` above it

```
update_cell_state(gs, p, t)                  # cell state machine, division, layers
F, torques, contacts = compute_forces*(gs, p, rng)
gamma = drag_scale * r
gamma += dt * contact_stiffness_per_granule(gs, contacts)   # semi-implicit (V3.2)
v = F / gamma
v *= speed_rails(v_max, population_cap)      # two rails, reported separately (V3.6)
x += v * dt;   theta / quat integrate
apply_position_bounds(gs, p);  _resolve_overlaps(gs, p);  apply_position_bounds
```

Three things about this loop are easy to get wrong and are documented where they
are decided:

* **`contact_semi_implicit` (default true since V3.4)** takes
  `v = F/(gamma + dt k)`. At the fixed point `F = 0`, so the equilibrium is
  exact for any `dt` — that is the unconditional stability. But the RATE is
  wrong by `(1 + S)` with `S = dt k/gamma`, and at the shipped `dt = 0.5 h` on a
  cell-seeded bed **S averages 41**. `k` includes WALL contacts since V3.6.
* **`advance()` (V3.6)** therefore sits above `step()`: `dt` is the interval at
  which the run is SAMPLED, and the mechanics inside it is advanced in `n_sub`
  substeps sized so `S ~ 0.2`. A substep is an ordinary `step` with a smaller
  `dt`, because everything inside is already a rate times `dt`. `v_max` is
  scaled with the subdivision so it caps displacement per coupling interval.
* **Two velocity rails, reported separately.** `v_max` is absolute;
  `dynamics.outlier_speed` caps a granule at 8x the median of its own
  coordination class. `frac_velocity_clipped` / `frac_outlier_clipped` never
  double-count.

### 2.11 Phase Field Rendering

**2D:** `render_fields()` — tanh-profile stamping with max-blending on Ngrid² grid.
**3D:** `render_fields_3d()` — superellipsoid implicit function with tanh profile on Ngrid_3d³ grid. Bounding-box clipping per granule for performance.

### 2.12 Transport Metrics (V1.4)

| Metric | Formula | Added to `compute_metrics()` |
|--------|---------|------------------------------|
| Porosity | `ε = phi_v_mean` | `porosity` |
| Kozeny-Carman permeability | `K = ε³d²/[180(1-ε)²]` | `K_kozeny_carman` |
| RCP fraction | `φ_RCP ≈ 0.64 + 0.08*(AR-1)` | `phi_RCP` |
| Compaction ratio | `φ_solid / φ_RCP` | `compaction_ratio` |
| Darcy number | `Da = K / L²` | `Da_number` |
| Grain diameter | Mean `2*r` | `d_grain_mean` |

### 2.13 Built-in Visualisation Functions

| Function | Output |
|----------|--------|
| `plot_granules()` | Circle/polygon patches at 5 time snapshots |
| `plot_fields()` | 3-row (phi_f, phi_i, phi_v) phase field heatmaps |
| `plot_timeseries()` | 3×3 grid: topology, dynamics, cell state |
| `plot_composite()` | RGB composite (R=functional, G=void, B=inert) |

---

## 3. Data Flow

```
Params ──▶ generate_packing*() ──▶ GranuleSystem
                                        │
                         ┌──────────────┘
                         ▼
                   ┌───────────┐
                   │  run()    │◀── rng (seeded)
                   │  loop     │
                   └─────┬─────┘
                         │
               ┌─────────┴─────────┐
               ▼                   ▼
      hist (list of dicts)   snaps (list of dicts)
               │                   │
    ┌──────────┼───────────────────┼──────────────┐
    ▼          ▼                   ▼              ▼
viz_compaction  viz_percolation  viz_movies   viz_phases
plot_timeseries plot_granules    plot_fields  plot_composite
```

**Snapshot dict format (V1.4):**
```python
{
    'phi_f': ndarray,          # Phase field (Ngrid² or Ngrid_3d³)
    'phi_i': ndarray,
    'phi_v': ndarray,
    'x': ndarray,              # Granule positions
    'y': ndarray,
    'z': ndarray,              # (3D only)
    'r': ndarray,
    'gtype': ndarray,
    'n_attached': ndarray,     # Cell state
    'spread_fraction': ndarray,
    'fa_maturity': ndarray,
    'n_overcrowded': ndarray,
    'a': ndarray, 'b': ndarray,  # Shape
    'n_shape': ndarray,
    'theta': ndarray,          # (2D only)
    'c': ndarray,              # (3D only)
    'n1': ndarray, 'n2': ndarray,  # (3D only)
    'quat': ndarray,           # (3D only, N×4)
}
```

---

## 4. Visualization & Analysis Scripts

### 4.1 Visualization Scripts — `viz/` package (V1.12)

| Script | Plots | Input |
|--------|-------|-------|
| `viz/compaction.py` | Void fraction, packing fraction, compaction ratio, void cluster size distribution, stacked phase areas | `hist`, `snaps` |
| `viz/percolation.py` | Kozeny-Carman K(t), porosity + RCP, dimensionless groups dashboard (2×3), Darcy flow rate, void connectivity | `hist` |
| `viz/movies.py` | Rotating 3D isosurface GIF, time-lapse compaction, z-sweep cross-section, composite 2×2 | `snaps`, `hist`, `p` |
| `viz/phases.py` | Phase isosurface strip, phase fractions vs time, tri-plane evolution (XY/XZ/YZ), interface area vs time | `hist`, `snaps`, `p` |
| `viz/cells.py` | Cell morphology patches, stress map, cell timelapse GIF (V1.5.1) | `snaps`, `hist`, `p` |
| `viz/stress.py` | 3D surface stress, granule isosurfaces, cell ellipsoids, evolution GIF (V1.5.2) | `snaps`, `hist`, `p` |
| `viz/shapes.py` | Granule shape gallery (superellipses, superellipsoids) | `snaps`, `p` |
| `viz/doe.py` | DOE statistical analysis & visualization | DOE `scan_dir` |
| `viz/dimensionless.py` | β-collapse, Ca-scaling, jamming diagram, factor effects, phase space (V1.7) | DOE `scan_dir` |
| `viz/scaffold_evolution.py` | 2D microstructure evolution timelapse for organ targets (V1.9) | `sweep_dir` |
| `viz/scaffold_evolution_3d.py` | 3D volumetric evolution with PyVista per organ (V1.9) | `sweep_dir` |
| `viz/energy_landscape.py` | Energy landscape per organ, time evolution, decomposition, design space (V1.11) | `sweep_dir` |

### 4.2 Rebuilt 2D Visualization — `viz2/` package (V2.4)

Unified 2D visualization system built on the scaffold map foundation, with Voronoi-based
spatial analysis, full energy mode decomposition, and void percolation theory.

**Architecture**: All modules share `viz2/common.py` for drawing primitives and data loading.
`viz2/voronoi.py` is the shared computational engine used by modules 2, 3, and 4.

| File | Module | Outputs |
|------|--------|---------|
| `viz2/common.py` | Shared utilities | Colors, `load_data()`, `ensure_phase_fields()`, `draw_granule_patches()`, `draw_cells_on_ax()`, `render_frame_to_array()`, 3D: `is_3d()`, `slice_snap_z_midplane()`, `slice_field_z_midplane()`, `cell_world_positions_3d()` |
| `viz2/voronoi.py` | Voronoi engine | `voronoi_from_centers()` (all granules, functional clipped to body), `voronoi_from_boundaries()` (shrink-wrap), `compute_shape_factors()`, `compute_local_phase_fractions()`, Sutherland-Hodgman clipping |
| `viz2/scaffold_map.py` | 1: Scaffold map | Multi-panel scaffold maps (void=black, functional=red, inert=green, bridges=crimson). 3D: z-midplane slice |
| `viz2/voronoi_shapes.py` | 2: Shape factors | Voronoi overlay, shape factor maps (circularity/elongation/area), distributions, timeseries. 3D: z-slice |
| `viz2/phase_fractions.py` | 3: Phase fractions | Global stacked area (conservation check), local heatmaps, inner vs outer timeseries, heterogeneity scatter. 3D: field + snap slicing |
| `viz2/movies.py` | 4: GIF movies | scaffold_evolution.gif, voronoi_evolution.gif, local_phi_f_evolution.gif, stress_evolution.gif, energy_modes_evolution.gif. 3D: z-slice per frame |
| `viz2/energy_stress.py` | 5: Energy + stress | Stress/strain maps via coarse-graining, 6 energy mode spatial maps (traction, contact, friction, osmotic, frustration, interfacial), timeseries. 3D: midplane slice of coarse-grained fields |
| `viz2/void_percolation.py` | 6: Percolation | Void cluster maps, percolation status evolution, P(s) power-law fits, Kozeny-Carman K(t). 3D: all 3 axes checked, z-slice for display |
| `viz2/scaffold_map_3d.py` | 7: 3D rendering | PyVista off-screen: superellipsoid meshes, cell spheres, bridge tubes, z-clip. Skips if 2D or no PyVista |
| `viz2/postprocess.py` | Orchestrator | CLI: `python viz2/postprocess.py -i results/default [--skip movies] [--only scaffold voronoi]` |
| `viz2/parallel.py` | Per-snapshot process pool (V3.0) | `pmap(fn, items, workers)` over snapshots: GIF frames, Voronoi tessellations, energy fields, cluster analysis. Spawn pool, order preserving, single-threaded workers, automatic serial fallback. Worker count: argument → `$VIZ2_WORKERS` → physical cores capped at 8 |
| `viz2/compare_runs.py` | Multi-run comparison (V3.0) | Any set of finished runs on the same figures: `compare_scaffolds.png` (one panel per run, same colour per f, scale bars, shared legend), `compare_timeseries.png` (one line per run), `compare_species.png`, `compare_dose_response.png` (per-species final outcome vs that species' f), `compare_timelapse.gif` (common time axis), `compare_summary.md/.csv`. Circles via one `EllipseCollection` per panel (10⁵ granules in seconds); 3D as z-midplane slice |

**Key design decisions:**
- Cell bridges count toward functional space in data analysis but retain crimson color for visibility
- Voronoi tessellation supports two modes: center-based (standard) and boundary-based shrink-wrap (superellipse surface sampling + ConvexHull merge)
- `ensure_phase_fields()` reconstructs phase fields from particle data when `save_fields=False`
- Pure numpy polygon clipping (Sutherland-Hodgman) avoids shapely dependency
- Each module is fault-isolated: one failure doesn't abort the rest

### 4.3 Mathematical Analysis Scripts — `analysis/` package (V1.12)

| Script | Plots | Input |
|--------|-------|-------|
| `analysis/mean_field_model.py` | Two-zone compaction fit, phase evolution, permeability, stress balance (V1.9) | `run_dir` |
| `analysis/parameter_sweep.py` | 11-D LHS sweep, compaction heatmaps, organ landscapes, radial profiles (V1.10) | standalone |
| `analysis/energy_landscape.py` | Free energy decomposition, kinetics, design space heatmaps (V1.11) | `sweep_dir` |
| `analysis/coarse_grain.py` | Stress timeseries, strain rate, viscosity evolution, coordination number | `run_dir` |
| `analysis/tissue_descriptors.py` | Descriptor summary, thickness distribution, S₂(r), pore size distribution | `run_dir` |
| `analysis/arch_distance.py` | Distance radar, trajectory, heatmap, sensitivity, optimization landscape | `run_dir` or `scan_dir` |
| `analysis/organ_targets.py` | Organ profile radar chart, organ comparison table, organ-specific phase mapping (V1.9) | — |

All scripts: CLI (`-i input_dir`), importable (`run_all()`), matplotlib Agg backend.

---

## 5. External Files

### Trial Configuration JSONs (`Trials/`)

Two formats supported (V1.4.1):

**Flat format** (recommended) — keys map directly to `Params` field names:
```json
{
    "_format": "flat",
    "mode": "3D",
    "Lx": 800, "Ly": 800, "Lz": 500,
    "E_modulus": 10.0,
    "t_total": 48.0,
    "shape_enabled": true
}
```

**Legacy format** — nested sections with translated key names:
```json
{
    "mode": "3D",
    "domain": { "side_length_um": 400, "Lz_um": 400 },
    "shape": { "aspect_ratio_c_func_mean": 1.0 },
    "output": { "Ngrid_3d": 80 }
}
```

Missing `"mode"` defaults to `"2D"`. See `Trial15_3D.json` for a flat format example.

### Compute layout — `gels/kernels/` (V3.0)

| File | Purpose |
|------|---------|
| `gels/kernels/__init__.py` | `HAS_NUMBA` / `njit` / `prange` fallbacks; `configure_threads(n, layer)` — numba threading layer (`omp` default) and thread count (0 = auto physical cores) |
| `gels/kernels/reference.py` | The V2.7 per-pair / per-cell / per-voxel Python loops, moved verbatim (forces, cell state machine and bridging, overlap resolution, rendering, metrics, packing relaxation). Oracle for the compiled kernels; the path LS-DEM runs on |
| `gels/kernels/neighbors.py` | Half pair lists: `half_pairs_tree` (cKDTree, reference order) and `half_pairs_cells` (compiled linked-cell grid, sorted pairs, thread-independent); `csr_from_pairs` (particle → its pairs); `neighbor_backend(p)` |
| `gels/kernels/geometry2d.py`, `geometry3d.py` | Status-tuple copies of the superellipse / superellipsoid / wall contact solvers (numba cannot call the `None`-returning originals) |
| `gels/kernels/contact2d.py`, `contact3d.py` | Contact forces: `pair_pass_*` (prange over pairs → records), `mc_dem_pairs`, `gather_*` (prange over particles → F, torques, clip arrays, walls; owner writes only). `compute_forces_2d/3d` drivers; `TIMINGS` for the bench |
| `gels/kernels/contacts.py` | `ContactSoA` (structure-of-arrays contact records that still iterate as dicts), record allocation, active noise |
| `gels/kernels/cells.py` | Cell state machine (`aggregates_k`, `cells_update_k`) and bridging (`bridge_pairs_k`, `bridge_cells_k`) with a splitmix64 counter RNG; `update_cell_state_k`, `bridging_k` |
| `gels/kernels/bridging.py` | Interim Python bridging over the kernel's candidate pairs with the restricted ray-cast (`perf_cells_backend='python'`: exact V2.7 cell machinery on compiled contacts) |
| `gels/kernels/integrate.py` | `resolve_overlaps` — post-step overlap correction on the neighbour list |
| `gels/kernels/render.py` | Bounding-box phase-field stamps parallel over grid rows / slabs with per-slab image lists; compiled effective radii; `render_fields_species` |
| `gels/kernels/metrics.py` | `compute_metrics` twin: compiled contact / bridge-count loops (reference order), bincount cluster sizes, strided Voronoi query, cached shape descriptors |
| `gels/kernels/packing.py` | `RSAChecker` (exact loop or neighbour grid — same decision, same draws) and the compiled settle substep (bit-identical positions) |
| `gels/io/writer.py` | `write_npz_atomic` (np.load-compatible zip, deflate level, atomic rename) and `SnapshotWriter` (bounded-queue background thread) |

**Exactness policy.** `Params.use_numba=False` (or `perf_neighbor_backend='reference'`) runs the pure-Python
reference everywhere; the bit-identical fixture gate (`tests/make_fixtures.py`) pins it. On the kernel path,
packing is bit-identical to the reference; contact forces, rendering and metrics agree to rounding
(`tests/test_*_vs_reference.py`); the cell state machine's deterministic parts are exact, while bridging draws
its randomness from a counter hash keyed by one `Generator` draw per pass — statistically equivalent, not
step-for-step comparable. `perf_cells_backend='python'` restores the exact cell machinery on top of the
compiled contacts. Every kernel is thread-count independent (owner-writes gathers, sorted pair lists).

**Compute layout of one step (kernel path).** `update_cell_state_k` (2 prange passes) → neighbour list
(`half_pairs_cells`) → `pair_pass` → `mc_dem_pairs` → `gather` → `bridge_pairs_k` → `bridge_cells_k` → noise
(numpy) → integration (numpy) → `resolve_overlaps` → at save steps `render_fields_species` (rows / slabs) →
`compute_metrics` → `SnapshotWriter.submit`. The thread count defaults to `auto_threads(N)` =
clamp(N / 1000, 4, physical cores); more threads only add fork/join overhead below N ≈ 10⁴.

| `gels/engine.py` | `Params`, `GranuleSystem` (`pos`/`vel`/`pos_unwrap` as `(N,3)` arrays with `x/y/z…` column properties), packing, `run()`/`step()` orchestration, serialization, and same-named **dispatcher wrappers** that forward to `reference` (kernels plug in behind `p.use_numba` / `p.deformable_enabled`) |
| `gels/materials.py` | Functionalization rules shared by loops and kernels: coverage-mixing of adhesion/friction, per-pair E*, Langmuir/power traction gain, blocker penalty, species pair tables |
| `gels/bench.py` | `python -m gels.bench` — per-component timings vs N and thread count |
| `tests/` | `unittest` suite; `tests/fixtures/` holds the V2.7 bit-identical oracle runs (`make_fixtures.py`) |

### Live view — `gels/live/` and the viz2 split (V3.0)

| Module | Role |
|---|---|
| `gels/live/observer.py` | `Observer` protocol (`on_start` / `on_step → stop?` / `on_save` / `on_end`) called by `run(p, seed, observer=)`; `LiveViewObserver` ships frames through a bounded queue with `put_nowait` (drop when full, adaptive cadence) and services pause / step / stop / cadence / detach commands |
| `gels/live/frames.py` | Vectorised cell world positions and bridge segments; compact `start` (static arrays, species table) and `frame` (float32 positions, int8 states, bridge segments, save-step metrics) messages built from a `GranuleSystem` or a snapshot dict |
| `gels/live/viewer.py` | TkAgg viewer process: `EllipseCollection` granules, per-state cell markers, `LineCollection` bridges, HUD, legend, metrics panel, blitting via Tk `after()`; z-slice for 3D; headless PNG mode. Never imports `viz2.common` |
| `gels/live/tail.py` | `SnapshotTailSource` (follows a run directory; `.tmp` files invisible, truncated files retried) and `ReplaySource` |
| `pipeline/live_view.py` | CLI: tail a running step 3, replay a finished run, or dump PNGs |
| `viz2/palette.py`, `viz2/snapshot_ops.py` | matplotlib-free colours / species tables and snapshot helpers shared by viz2 and the viewer; `viz2/common.py` re-exports them |

Process model: `step3 --live` spawns the viewer (`multiprocessing` `spawn`, daemon) with a
`Queue(maxsize=4)` for frames and a control queue + events back to the engine. The engine
never blocks on the viewer; a closed window detaches and the run continues. Snapshot and
history writes are atomic (`.tmp` + `os.replace`), so a tailing viewer never sees a
half-written file. The parent process stays on Agg; the child sets `MPLBACKEND=TkAgg`
before importing matplotlib.

### Modules added V3.2–V3.6

The engine grew six top-level modules after V3.0. Each is deliberately a LEAF —
it imports from `gels.engine` or from nothing, never the reverse — so the twins,
the pipeline and the tests can all use it and the kernels stay importable.

| Module | Added | What it is | Imports from `gels.engine`? |
|---|---|---|---|
| `gels/division.py` | V3.1 | Cell division pass: doubling time, contact inhibition, the `cycle` model that makes 24 h *mean* 24 h. `age_cells` is called from **both** `update_cell_state` twins — until V3.2 nothing advanced `gs.cell_age`, so with the default 8 h refractory no cell ever divided | yes |
| `gels/laguerre.py` | V3.3 | Radical (Laguerre) Voronoi: the **halo-free** local packing fraction. `compaction_func = phi_loc_func / phi_RCP` is the honest twin of `compaction_ratio`, which is ~1.5x inflated by the tanh interface. Qhull cannot be compiled, so this is top level, not a kernel; `metrics_laguerre` is off by default at 0.3–1 s/frame | yes |
| `gels/pore.py` | V3.3 | Katz-Thompson permeability and geodesic tortuosity on the signed-distance field both twins were already building and half-discarding | yes |
| `gels/kernels/percolation.py` | V3.3 | Union-find percolation on the **real contact graph** (`gran_lf_*`, `gran_span_*`, `gran_z_mean`, `gran_rattler_frac`), against `func_lf`'s thresholded tanh field. Pure NumPy, called by both twins so they cannot disagree | no |
| `gels/celltypes/` | V3.6 | Instantiable cell types, every value carrying its source. `to_overrides()` emits the same dotted currency as a preset, so a preset states the scaffold and a cell type states the cell | only for the geometry cross-check, in tests |
| `gels/convergence.py` | V3.6 | Proportional-window arrest detector. Consumes the flat metrics dict and a position array and imports **nothing** from the engine, which is what lets `pipeline/shadow_check.py` replay a finished run through the identical code | **no** |

### The four numerical rails, and how to tell which one a run is on

A recurring theme from V3.2 onward: the engine has several mechanisms that stand
in for the contact law, and a run dominated by one of them looks exactly like a
run doing physics. Each is now observable.

| Rail | What it does instead of the contact law | Diagnostic | Added |
|---|---|---|---|
| Overlap projection | Pushes granules apart geometrically when `max_overlap_frac` is exceeded | `n_overlap_clipped`, `overlap_clip_fraction`, `max_overlap_ratio` (equal to `max_overlap_frac` means pinned) | V3.2 |
| Velocity cap | Discards force MAGNITUDE above `v_max * drag_scale * r` | `frac_velocity_clipped` | V3.2 |
| Population speed cap | Limits a granule against its own coordination class | `frac_outlier_clipped` | V3.6 |
| Semi-implicit damping | Correct fixed point, RATE too slow by `1 + dt k/gamma` | `stiffness_number`, `n_substeps`, `substep_budget_bound` | V3.6 |

And two audits that say whether the force law itself is sound:

* **`dynamics.gradient_flow`** (V3.5) — overdamped dynamics is gradient flow, so
  the energy must fall every step and the work must account for the fall.
  Central-differencing the total energy against `compute_forces` gives **3.9e-9**
  for spheres with walls and gravity. Four things break it and all are
  intentional: active noise, explicit friction, MC-DEM and shaped granules.
  `energy_residual` is then the non-conservative throughput — and with cells
  seeded, after V3.6 subtracts `energy_cell`, it is the **myosin work**.
* **`handoff_force_balance`** (V3.4) + **`packing.relax`** (V3.5) — the packer
  used to hand over a bed pre-loaded far above the driving load, because the
  settle stops on a LENGTH tolerance that knows nothing about the contact law
  the dynamics will apply. FIRE-relaxing under the *exact* dynamics force law
  takes a shaped gravity bed from 346x the granule weight to 1.0x.

### Execution — `pipeline/` package (V2.7+)

Runs are local and proceed as five numbered steps, each a standalone script
operating on one run directory. Progress is checkpointed in
`<run_dir>/pipeline_state.json`, so each step verifies its prerequisite,
refuses to clobber completed work without `--force`, and can be re-run alone.

| File | Purpose |
|------|---------|
| `pipeline/step1_config.py` | Resolve Params (defaults → trial JSON → CLI flags) → `params.json` |
| `pipeline/step2_pack.py` | Packing, cell seeding, `t = 0` evaluation → `snap_0000.npz` |
| `pipeline/step3_simulate.py` | Advance to `t_total`; `--continue` resumes or extends a run |
| `pipeline/step4_postprocess.py` | Drive the `viz2` and/or `viz` suites |
| `pipeline/step5_analysis.py` | Run the `analysis/` package → `analysis/summary.json` |
| `pipeline/run_showcase.py` | V3.0 showcase: a table of conditions written as setup files (`Trials/showcase_v3/`), each run as its own steps 1–5 process under a weighted thread budget, then `viz2/compare_runs.py` on all of them |
| `pipeline/_common.py` | Step state and guards, Params/trial-JSON loading, formatting |

Steps 2 and 3 are joined by the engine's V2.6 resume mechanism: step 2 writes
snapshot 0000, step 3 restores it and continues the main loop. Since resuming
restarts the RNG stream, a packing + simulate pair is deterministic for a given
seed but not bit-for-bit identical to a monolithic `run()` call.

**HPC support was removed in V2.7.** The SLURM templates, cluster config and
sync scripts are archived unmodified under `old code/hpc/`, alongside the
retired `run_hpc_headless.py`, `run_all_trials.py` and
`run_analysis_pipeline.py`. See `old code/README.md`.

### Mathematical Analysis Framework — `analysis/` package (V1.7+)

| File | Purpose |
|------|---------|
| `analysis/mean_field_model.py` | Volume-conserving two-zone ODE: dx_f/dt = -x_f(σ_cell - σ_resist)/η_eff. Motor-clutch cell stress, jamming resistance, bridge lock-in/secondary migration. Fits η_eff, σ_0, α to DEM data. (V1.9) |
| `analysis/parameter_sweep.py` | 11-D Latin hypercube sweep with 1D radial PDE model. Organ distance prediction, design recommendations, jamming constraint. (V1.10) |
| `analysis/energy_landscape.py` | Free energy landscape: 6 terms (cell, elastic, yield, void, inert, surface). Equilibrium, barrier, overdamped kinetics, dimensionless groups. (V1.11) |
| `analysis/coarse_grain.py` | Coarse-graining: Love-Weber stress tensor from per-contact data, strain rate from velocity field, effective viscosity η_eff = σ_dev/(2ε̇_dev). Spatial fields via Gaussian weighting. |
| `analysis/tissue_descriptors.py` | Tissue architecture descriptor vector from 3D phase fields: BV/TV, Tb.Th, Tb.Sp, SMI, Euler characteristic, MIL tensor, tortuosity, S₂(r), pore size distribution. |
| `analysis/organ_targets.py` | Target descriptor vectors for 7 organ systems with organ-specific phase mapping (perfusive_void_fraction). (V1.9) |
| `analysis/arch_distance.py` | Weighted Mahalanobis-like architectural distance with log-transform on scale-dependent descriptors. Distance trajectories, DOE optimization, sensitivity. |
| `viz/dimensionless.py` | Dimensionless analysis across DOE runs. Computes β, Ca, jamming proximity. Data collapse, factor effects, phase space. |
| `CodeLog/Architecture/MATHEMATICAL_MODEL.md` | Formal mathematical model document: microscale DEM, coarse-graining, two-zone ODE, free energy landscape, dimensionless groups, tissue characterization. |

---

## 6. Performance (V1.6)

### 6.1 Numba JIT Compilation

~20 hot-path functions are decorated with `@njit(cache=True)`. When Numba is not installed,
a no-op identity decorator is used as fallback. JIT'd function groups:

| Group | Functions | Count |
|-------|-----------|-------|
| Quaternion | `quat_multiply`, `quat_conjugate`, `quat_normalize`, `quat_rotate`, `quat_rotate_inv`, `quat_to_rotation_matrix`, `quat_integrate` | 7 |
| Superellipsoid | `_sgnpow`, `superellipsoid_point`, `superellipsoid_normal`, `superellipsoid_curvature_radii`, `superellipsoid_implicit` | 5 |
| Superellipse | `superellipse_point`, `superellipse_tangent`, `superellipse_normal_vec`, `superellipse_curvature_radius`, `_world_to_body`, `_body_to_world`, `superellipse_implicit` | 7 |
| Contact solvers | `find_contact_superellipses`, `find_contact_superellipse_wall`, `find_contact_spheres_3d`, `find_contact_superellipsoids_3d` | 4 |
| Physics | `hertz_contact_force` | 1 |

**JIT warmup**: `_warmup_jit(is_3d)` is called before the first timestep to trigger
compilation with dummy data, avoiding a compilation delay on step 1.

**Numba compatibility notes**: Scalar `np.clip` replaced with `min(max(x, lo), hi)`.
`np.linalg.norm` on small arrays replaced with manual `np.sqrt(x**2+y**2+z**2)`.

### 6.2 Vectorized Integration

Both 2D and 3D integration paths in `step()` use array-wide NumPy operations instead of
per-granule Python loops:

- **Force → velocity**: `vel = F[:N] / gamma[:, None]`
- **Velocity cap**: boolean mask on `speed > v_max`, rescale in-place
- **Position update**: `gs.x[:N] += vel[:, 0] * dt` (vectorized)
- **Boundary clamping**: `np.clip` on all coordinates simultaneously
- **Rotational dynamics (2D)**: vectorized `omega = torque / gamma_rot`, clip, integrate
- **Active noise**: vectorized per functional-granule subset

### 6.3 Scaling

Runtime bottleneck: Newton-Raphson contact solver (~80% of wall time).
Empirical scaling: `T ≈ 1.106 * N^1.092` seconds per 20 timesteps (benchmarked on
Puma HPC, single core). Near-linear in granule count due to cKDTree neighbour pruning.

### 6.4 Walltime Estimation (run_all_trials.py)

Power-law model predicts SLURM walltime from trial parameters:
- Estimates granule count from domain volume, target phi, and mean radius
- Applies RSA efficiency factor (0.55) for realistic packing
- Scales by timestep count relative to benchmark reference
- Adds overhead (120 s) and safety factor (2×)
- Maximum across all trials sets SLURM `--time`

## 7. Design Constraints and Assumptions

1. **Three modes**: 2D, 2D-slice, and 3D. Default is 2D for backward compatibility.
2. **Two boundary modes** (V1.10): `"walls"` (rigid Hertzian walls, default) or `"periodic"`
   (minimum image convention, cKDTree boxsize, position wrapping, no walls).
3. **Overdamped regime**: No inertial terms. Valid for cell-culture timescales
   (hours) in viscous medium.
4. **Rigid or deformable granules**: Rigid shapes use volume-conserving effective radii;
   deformable LS-DEM shapes use modal deformation DOFs (V2.3). Both rendered with JKR
   contact-face clipping for physically realistic flat faces at contacts.
5. **JKR contact model** (V2.3): Replaces Hertz+DMT. JKR is correct for soft hydrogels
   (Tabor parameter μ_T >> 1). Reduces exactly to Hertz when W=0.
6. **Random by default**: Seed is `None` (random) unless explicitly set. All randomness
   flows through `numpy.random.Generator`.
7. **Performance target**: 500-1000 granules in 3D with optional Numba JIT.
8. **Rendering backends**: PyVista primary for 3D (high quality, off-screen capable),
   matplotlib fallback when PyVista unavailable.
9. **Aggregate-authoritative cell state**: Per-granule aggregates (n_attached, etc.) are the
   source of truth for force computation. Individual cell states are derived from aggregates.
   This ensures physics identical to V1.4.1.
10. **Bridge lock-in** (V1.9): High-force bridges (F ≥ threshold) bypass senescence and
    persist indefinitely. Secondary migration boosts bridge formation along existing bridges.

---

## 8. Data Serialization (V1.5)

### 8.1 CellState Enum

`CellState(IntEnum)` with values: ATTACHED(0), SPREADING(1),
PROLIFERATING(2), BRIDGING(3), SENESCENT(4). Stored as int arrays, extensible.

### 8.2 Output Format

```
results/<run_name>/
  params.json           — Params dataclass as JSON
  metadata.json         — version, git hash, seed, cell state enum map
  history.csv           — scalar metrics (pandas-loadable)
  history.json          — scalar metrics (exact fidelity)
  snapshots/
    snap_NNNN.npz       — granule + cell arrays per timepoint
  fields/               — optional (save_fields=True)
    fields_NNNN.npz     — phi_f, phi_i, phi_v grids
  <run_name>.tar.gz     — archive of everything
```

### 8.3 Per-Snapshot NPZ Contents

**Granule arrays (N):** x, y, z, r, gtype, a, b, c, n1, n2, vx, vy, vz,
n_attached, spread_fraction, fa_maturity, n_overcrowded, n_cells,
force_x, force_y, force_z, theta/quat.

**Cell arrays (N_cells):** cell_granule_id, cell_state, cell_theta_local,
cell_eta_local, cell_omega_local, cell_fx, cell_fy, cell_fz,
cell_bridge_target, cell_bridge_age, cell_contact_area, cell_offset.

**Contact arrays (N_contacts, V1.5.2):** contact_i, contact_j (granule pair),
contact_cx, contact_cy, contact_cz (contact point), contact_nx, contact_ny,
contact_nz (contact normal), contact_overlap, contact_R_eff, contact_F_normal,
contact_A_contact.

### 8.4 Loading API

- `load_run(run_dir)` → `(hist, snaps, p, metadata)` — from directory or .tar.gz
- `load_cells(run_dir, snap_index=None)` — targeted cell data loading
