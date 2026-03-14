# GELLS-DEM Changelog

All notable changes to the GELLS-DEM simulation engine are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering: `MAJOR.MINOR` where MAJOR tracks breaking API changes and
MINOR tracks feature additions and improvements.

---

## [V1.3] - 2026-03-14

### Added
- **Superellipse granule shapes**: Granules are now parameterised as
  superellipses `|x/a|^n + |y/b|^n = 1` with per-granule semi-axes (a, b),
  blockiness exponent (n), and orientation angle (θ). Controlled by
  `shape_enabled` flag in `Params` (default: `False` for V1.2 backward
  compatibility). Per-type distributions for aspect ratio and blockiness.
- **Common normal contact detection**: Newton-Raphson iterative solver for
  superellipse–superellipse contact. Returns penetration depth, contact normal,
  contact point, and local curvature radii. Circle fast-path preserves V1.2
  performance when `shape_enabled=False`.
- **Superellipse–wall contact**: Boundary sampling method to detect penetration
  of non-circular granules against domain walls.
- **Rotational dynamics**: Overdamped angular integration `γ_rot dθ/dt = Σ τ`
  from off-centre contact forces and friction. Angular velocity cap `omega_max`.
  Only active for non-circular granules.
- **Hertz with local curvature**: Contact force uses local radius of curvature
  at the contact point rather than the global granule radius. Generalises the
  Hertz, DMT adhesion, and area-dependent friction models to non-circular shapes.
- **Shape descriptors** (following Liu et al. 2025, Table 1): Per-granule
  circularity (`4πA/P²`), aspect ratio, elongation, and blockiness. Population
  statistics (mean, std) added to metrics output.
- **Superellipse geometry utilities**: `superellipse_area()`, `superellipse_perimeter()`,
  `superellipse_point()`, `superellipse_normal_vec()`, `superellipse_curvature_radius()`,
  `superellipse_polygon_pts()`, `superellipse_implicit()`.
- **New `Params` fields**: `shape_enabled`, `aspect_ratio_func/inert_mean/std`,
  `blockiness_func/inert_mean/std`, `drag_scale_rot`, `omega_max`.
- **New `GranuleSystem` arrays**: `a`, `b`, `n_shape`, `theta`, `omega`, `r_bound`,
  `is_circle` flag.

### Changed
- **`compute_forces()`** now returns `(F, torques)` tuple instead of just `F`.
  Torque array is `(N,)` in units of nN·µm.
- **`step()`** integrates both translational and rotational dynamics.
- **Wall clamp** uses `r_bound` (bounding circle radius) instead of `r`.
- **Neighbour search** uses `r_bound` for cutoff distance.
- **`render_fields()`** uses superellipse implicit function for field profiles
  when `shape_enabled=True`.
- **`plot_granules()`** renders `Polygon` patches for superellipses, `Circle`
  patches for circles.
- **`compute_metrics()`** uses shape-aware contact detection and area
  computation for superellipses.
- **Snapshot tuple** extended with shape arrays (a, b, n_shape, theta) at
  indices 11–14.

---

## [V1.2] - 2026-03-14

### Added
- **HPC support files** for UArizona Puma cluster:
  - `run_hpc.slurm` — SLURM batch job script for headless simulation runs.
  - `setup_hpc_env.sh` — One-time environment setup (Python 3.11 venv).
  - `run_hpc_headless.py` — CLI-driven headless runner that saves figures to
    disk (no `plt.show()`), with argparse overrides for all `Params` fields.
  - `CodeLog/Readme/HPC_SETUP.md` — Step-by-step guide for VSCode Remote SSH
    to UArizona Puma, environment setup, and job submission.
  - `hpc/generate_hpc_scripts.py` — Generates per-user `setup_env.sh` and
    `run.slurm` from a JSON config file (e.g. `hpc/Alex.json`).
  - `hpc/Alex.json` — Example user config for HPC script generation.
- **Trial batch runner** (`run_all_trials.py`): Runs all Trial JSON configs
  in `Trials/` either locally (sequential) or on HPC (SLURM array job, one
  task per trial running in parallel).
- **Trial JSON loading** in `run_hpc_headless.py`: `--trial` flag maps
  the legacy nested Trial JSON format to current `Params` fields.

### Added
- **Area-dependent hydrogel friction** (Gong 2006, Pitenis et al. 2014,
  Uruena et al. 2018): Tangential friction at granule contacts uses the
  hydrogel-tribology model `F_fric = τ₀ × A_contact × tanh(|v_t|/v_ref)`,
  where `A_contact = π R* δ` is the Hertzian contact area. Three pair types
  with distinct interfacial shear stresses: inert–inert (τ₀=50 Pa, bare
  Gemini gel), inert–functional (500 Pa, bare vs collagen), functional–
  functional (2000 Pa, collagen–collagen H-bonding/entanglement).
- **DMT adhesion** (Derjaguin, Muller, Toporov 1975): Constant attractive
  normal force during contact `F_adh = 2π W R*`. Work of adhesion varies by
  pair type: W_ii=0.5, W_if=1.0, W_ff=2.0 mJ/m². Net normal force
  `F = F_Hertz − F_adh` can be negative (attractive).
- **`get_pair_friction_params()`**: Helper to look up τ₀ and W by granule
  pair surface chemistry (bare gel vs collagen-I coated).
- **Velocity state in `GranuleSystem`**: `vx`, `vy` arrays store per-granule
  velocities from the previous timestep for friction force calculation.
- **New parameters in `Params`**: `tau_0_ii/if/ff`, `W_adh_ii/if/ff`,
  `friction_v_ref`.
- **Contact-type metrics**: `n_contacts_ff`, `n_contacts_if`, `n_contacts_ii`
  tracked per save step.
- **Literature references document**: `CodeLog/References/REFERENCES.md`
  catalogues all papers, equations, assumptions, and parameter derivations.
- **Cell lifecycle model**: Cells now follow a biologically motivated lifecycle:
  seeded (spheres, d=20 um) -> attached (~3 h onset, sigmoidal kinetics) ->
  spread (volume-conserving ellipsoid, ~5 um tall) -> bridging (motor-clutch).
- **Motor-clutch force model** (Chan & Odde 2008): Replaces the simple linear
  spring (`k_cell * extension`) with a substrate-stiffness-dependent force model.
  Force per cell depends on `E_modulus` through `F_mc = F_stall * k_sub/(k_sub + k_opt)
  * engagement * FA_maturity`, giving ~2 nN on 1 kPa and ~21 nN on 100 kPa substrates.
- **Projected-area cell placement**: Cell capacity per granule is now computed from
  the cell's projected area (`pi*(d/2)^2` for spheres) divided into the granule's
  projected area (`pi*R^2 * coverage`), not circumference.
- **Cell state arrays in `GranuleSystem`**: `n_attached`, `spread_fraction`,
  `fa_maturity`, `n_overcrowded` track per-granule cell state each timestep.
- **Cell geometry functions**: `cell_projected_area()` computes the footprint as
  cells transition from sphere to oblate ellipsoid (volume conserved).
  `max_cells_on_granule()` derives capacity from projected area ratio.
- **`update_cell_state()`**: Per-step cell state evolution with sigmoidal
  attachment kinetics, stiffness-dependent spreading rate, FA maturation,
  and overcrowding detection (excess cells crawl on each other).
- **`motor_clutch_force()`**: Steady-state motor-clutch force per cell with
  correct unit handling (kPa * um = nN/um for substrate stiffness).
- **Cell state metrics**: `n_attached_total`, `n_seeded_total`, `mean_spread_frac`,
  `mean_fa_maturity`, `n_overcrowded_total` tracked at each save step.
- **Cell state in snapshots**: Snapshots now include `n_attached`, `spread_fraction`,
  `fa_maturity`, `n_overcrowded` arrays for post-processing.
- **Cell state timeseries plots**: Third row in `plot_timeseries()` shows
  attachment, spreading/FA maturity, and overcrowding evolution.
- **New parameters in `Params`**: `cell_height_spread`, `t_attach_onset`,
  `t_attach_half`, `t_spread_duration`, `fa_maturation_rate`, `n_motors`,
  `F_motor_stall`, `n_clutches`, `k_clutch`, `k_on_clutch`, `k_off_clutch`,
  `F_bond`, `cell_sense_distance`.

### Changed
- **`cell_diameter`**: Default changed from 15.0 to 20.0 um (measured cell diameter).
- **`L_rest`**: Default changed from 12.0 to 5.0 um (spread cell thickness).
- **Cell bridging force**: Only attached, non-overcrowded cells can form bridges.
  Bridge force is `F_mc * n_bridges` (motor-clutch, not spring-extension).
- **Bridge metric**: `n_bridges` now only counts pairs where both granules have
  attached cells, matching the force computation.
- **`L_max`**: Now a derived property (`= cell_sense_distance`) instead of an
  independent parameter.
- **Console output**: Status table now shows `attach`, `spread`, `FA_mat`
  columns instead of `delta/R%` and `AreaCon`.
- **`step()` signature**: Now takes `t` (simulation time) to drive cell state.
- **`plot_timeseries()`**: Expanded from 2x3 to 3x3 grid with cell state row.

### Removed
- `k_cell` parameter (replaced by motor-clutch model).
- `L_max` as independent parameter (now derived from `cell_sense_distance`).

---

## [V1.1] - 2026-03-14

### Added
- **Hertzian contact mechanics**: Replaced arbitrary linear spring constant
  (`k_contact = 50 nN/um`) with physically meaningful Hertz contact theory.
  Contact force now follows `F = (4/3) E* sqrt(R*) delta^(3/2)`, derived from
  the hydrogel Young's modulus `E_modulus` (kPa) and Poisson's ratio
  `poisson_ratio`.
- **New parameters in `Params`**: `E_modulus` (default 10.0 kPa) and
  `poisson_ratio` (default 0.45) replace the old `k_contact` and `k_wall`.
- **`hertz_contact_force()`**: Unit-converting Hertz force function
  (Pa, um, um -> nN).
- **`overlap_lens_area()`**: Exact 2D lens intersection area between two circles
  for volume conservation tracking.
- **`compute_effective_radii()`**: Computes volume-conserving display radii by
  inflating each granule proportionally to its accumulated overlap area:
  `r_eff = sqrt(r^2 + delta_A / pi)`.
- **`print_stiffness_info()`**: Diagnostic that prints expected equilibrium
  overlap for the configured modulus, helping users verify physical meaning.
- **Volume-conserving phase field rendering**: `render_fields()` now uses
  effective radii instead of original radii, so overlapping material is
  redistributed rather than lost.
- **Hertzian wall contact**: Wall repulsion uses the rigid-wall Hertz limit
  `E* = E / (1 - nu^2)` with `R* = R_i` (sphere-flat geometry), giving
  physically consistent and stiffer boundary response.
- **Overlap diagnostics in `compute_metrics()`**: New tracked metrics per
  save step: `n_contacts`, `max_overlap_ratio` (delta/R), `total_overlap_area`
  (um^2), `area_conservation` (should stay near 1.0).
- **Console output columns**: Added `delta/R%` and `AreaCon` columns to the
  per-step status table.
- **Summary output**: Final summary now reports contact count (by pair type),
  max overlap ratio, area conservation, and friction/adhesion parameters.

### Changed
- **Contact force law**: From linear `F = k_contact * delta` to nonlinear
  Hertzian `F ~ delta^(3/2)`. The nonlinearity means softer response at small
  overlaps and stiffer response at large overlaps.
- **Wall force law**: From linear `F = k_wall * penetration` to Hertzian
  `F ~ pen^(3/2)` with the rigid-wall effective modulus.
- **Default stiffness calibration**: `E_modulus = 10 kPa` was chosen to produce
  similar contact forces to the old `k_contact = 50 nN/um` at typical overlaps,
  preserving existing simulation dynamics while grounding them in material
  properties.

### Removed
- `k_contact` parameter (replaced by `E_modulus` + `poisson_ratio`).
- `k_wall` parameter (now derived from Hertz theory with rigid-wall limit).

---

## [V1.0] - 2026-02-26

### Added
- Initial 2D overdamped particle dynamics engine (`new_dem_0.py`).
- `Params` dataclass for all simulation parameters.
- `GranuleSystem` state container for granule positions, radii, types, cell counts.
- Random sequential addition packing generator with interleaved placement.
- Linear elastic soft-sphere contact repulsion (`k_contact * delta`).
- Cell-mediated bridging forces between functional granule pairs with
  proximity-weighted bridge count, spring extension, and force cap.
- Linear wall repulsion (`k_wall * penetration`).
- Active noise on functional granules (overdamped Langevin).
- Overdamped Euler integrator with velocity cap and hard wall clamping.
- Phase field rendering via tanh-profile stamping with max-blending.
- Cluster analysis metrics (functional, inert, void connectivity).
- Tissue fraction, bridge count, displacement, and force statistics.
- Four built-in visualisation functions: `plot_granules`, `plot_fields`,
  `plot_timeseries`, `plot_composite`.
- Unified post-processing script (`new_dem_visualization.py`) for z-stacks,
  3D isosurfaces, metric dashboards, and animated GIFs.
- Legacy-compatible post-processor (`new_dem_postprocess.py`).
- Trial configuration JSONs (Trial1--Trial12) for parameter sweeps.
- Legacy 3D superellipsoid code archived in `old/`.

---

## Bug Fixes

*No bug fixes have been recorded yet. Bug fixes will be listed here with their
associated version, date, and description of the issue and resolution.*

<!-- Template for future bug fix entries:

### [V1.x] - YYYY-MM-DD
- **Fixed**: [Description of the bug].
  **Root cause**: [What caused it].
  **Resolution**: [How it was fixed].
  **Files changed**: [List of modified files].

-->
