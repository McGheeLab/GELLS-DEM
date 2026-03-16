# GELLS-DEM Changelog

All notable changes to the GELLS-DEM simulation engine are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering: `MAJOR.MINOR` where MAJOR tracks breaking API changes and
MINOR tracks feature additions and improvements.

---

## [V1.8] - 2026-03-15

### Changed
- **Project restructured**: Organized ~20 root-level Python files into `viz/` and `analysis/`
  packages. Root now contains only the simulation engine (`new_dem_0.py`) and 4 runner scripts.
- **Visualization files** moved to `viz/` package (10 files): `viz_postprocess.py` → `viz/postprocess.py`,
  `viz_cells.py` → `viz/cells.py`, `viz_stress.py` → `viz/stress.py`, etc. Import as
  `from viz.postprocess import run_all`.
- **Analysis files** moved to `analysis/` package (5 files): `mean_field_model.py` →
  `analysis/mean_field_model.py`, `coarse_grain.py` → `analysis/coarse_grain.py`, etc.
  Import as `from analysis.coarse_grain import compute_stress_tensor`.
- **HPC files consolidated**: `run_hpc.slurm` and `setup_hpc_env.sh` moved into `hpc/`.

### Removed
- `dem_config.json` (legacy, unused)
- `old/` directory (empty)

---

## [V1.7] - 2026-03-15

### Added
- **Mathematical analysis framework**: Complete continuum-scale mathematical model connecting
  DEM particle simulations to tissue-level predictions and organ comparisons.
- **`mean_field_model.py`**: Mean-field ODE model for cell-driven scaffold compaction.
  `dφ_f/dt = -φ_f(σ_cell - σ_resist)/η_eff` with motor-clutch cell stress, jamming resistance,
  and Poisson bridge formation kinetics. `CompactionModel` class with `solve()`, `fit_to_data()`,
  and `from_run()` for automatic fitting to simulation data. Extracts η_eff (effective viscosity),
  σ_0 (jamming stress), α (jamming exponent). Predicts permeability evolution via Kozeny-Carman
  and Darcy flow rate. Plotting: fit overlay, phase evolution, permeability, stress balance.
- **`coarse_grain.py`**: Coarse-graining routines extracting continuum quantities from DEM
  snapshots. `compute_stress_tensor()` via Love-Weber formula (uses per-contact data from V1.5.2).
  `compute_strain_rate()` from velocity field. `compute_effective_viscosity()` = σ_dev/(2ε̇_dev).
  `compute_coordination()` decomposed by pair type. `extract_continuum_timeseries()` for full
  time evolution. `coarse_grain_field()` for spatially-resolved stress/strain fields via
  Gaussian weighting. Handles both 2D and 3D modes.
- **`viz_dimensionless.py`**: Dimensionless analysis and data collapse across DOE runs.
  Computes β (motor-clutch engagement), Ca (cellular capillary number), jamming proximity,
  timescale ratio, composition ratio, size ratio. Seven plot types: β-collapse (rescaled
  compaction trajectories), Ca-scaling (power-law fits), jamming diagram, composition effects,
  DOE factor effects (main effects + interactions), dimensionless dashboard (3×2), phase space
  (β vs Ca contour). Includes summary statistics table output.
- **`tissue_descriptors.py`**: Comprehensive tissue architecture descriptor vector from 3D
  phase fields (inert phase treated as pore space). 18 descriptors in 8 categories: volume
  fractions (BV/TV, porosity, S/V), morphometry (Tb.Th, Tb.Sp, Tb.N via distance transform),
  topology (Euler characteristic, connectivity density, SMI), spatial statistics (two-point
  correlation S₂(r) via FFT, correlation length, chord length distribution), anisotropy (MIL
  tensor, degree of anisotropy DA, fractional anisotropy FA), transport (tortuosity via Laplace
  solver, Kozeny-Carman permeability), pore size distribution. Handles 2D gracefully.
- **`organ_targets.py`**: Literature-based target descriptor vectors for 7 native organ systems:
  trabecular bone, lung alveoli, liver, kidney cortex, cardiac muscle, pancreatic islet,
  intestinal mucosa. Each with mean, standard deviation (natural variability), description,
  and key references. Radar/spider chart for organ profile comparison.
- **`arch_distance.py`**: Weighted Mahalanobis-like architectural distance between GELLS
  scaffolds and organ targets. Log-transform for scale-dependent descriptors (BV/TV, Tb.Th,
  etc.) so ratios drive comparison. `distance_trajectory()` tracks which organ the scaffold
  converges toward over time. `optimal_parameters()` finds DOE conditions minimizing distance
  to a target organ. `sensitivity_analysis()` quantifies DOE factor effects on organ matching.
  Heatmap, radar, and optimization landscape plots.
- **`CodeLog/Architecture/MATHEMATICAL_MODEL.md`**: Formal mathematical model document for
  publications. Covers: microscale DEM equations (Hertz, motor-clutch, friction, adhesion,
  bridge kinetics), mesoscale coarse-graining (Love-Weber, strain rate), macroscale continuum
  model (phase conservation, Darcy flow), mean-field ODE, 6 dimensionless groups with scaling
  laws, tissue architecture characterization framework, and experimental validation strategy.

### Changed
- **`viz/postprocess.py`** (was `viz_postprocess.py`): Pipeline expanded from 10 to 14 modules. New modules 11-14:
  mean_field_model, coarse_grain, tissue_descriptors, arch_distance. Batch DOE processing
  now auto-triggers dimensionless analysis and architectural distance. New skip names:
  'mean_field', 'coarse_grain', 'tissue', 'arch_distance', 'dimensionless'.
- **`CodeLog/Architecture/ARCHITECTURE.md`**: Updated to V1.7 with mathematical analysis
  framework section. New architecture diagram shows analysis pipeline.

---

## [V1.6] - 2026-03-15

### Changed
- **Visualization decoupled from simulation engine** (`new_dem_0.py`): All matplotlib imports,
  built-in plotting functions (`plot_granules`, `plot_fields`, `plot_timeseries`,
  `plot_composite`), and visualization script calls removed from `new_dem_0.py`. The simulation
  engine now only runs physics and saves data — no plotting dependencies.
- **`save_fields` default changed to `True`**: Phase field grids (`phi_f`, `phi_i`, `phi_v`)
  are now saved to disk by default so that postprocessing can produce all visualizations
  without re-running the simulation.
- **Removed UNATTACHED cell state**: Cells are now always attached from the start.
  `CellState` enum: ATTACHED=0, SPREADING=1, PROLIFERATING=2, BRIDGING=3, SENESCENT=4.
  Excess cells during packing go directly to SENESCENT instead of UNATTACHED.

### Added
- **`viz_postprocess.py`**: Unified post-processing script that loads saved simulation data
  and produces all visualizations. Runs the 4 built-in plots (granules, fields, timeseries,
  composite) plus all `viz_*.py` scripts (compaction, percolation, movies, phases, cells,
  stress). CLI: `python viz_postprocess.py -i results/default`. Supports `--skip` to exclude
  specific visualization modules. Programmatic API: `viz_postprocess.run_all(run_dir)`.
- **Numba JIT acceleration** (`new_dem_0.py`): `@njit(cache=True)` applied to ~20 hot-path
  functions: quaternion utilities (7), superellipsoid geometry (5), superellipse geometry (8),
  contact solvers (4: `find_contact_superellipses`, `find_contact_superellipse_wall`,
  `find_contact_spheres_3d`, `find_contact_superellipsoids_3d`), and `hertz_contact_force`.
  Identity fallback when Numba is not installed. JIT warmup (`_warmup_jit()`) runs before
  first timestep to avoid first-step compilation delay.
- **Vectorized integration** (`new_dem_0.py`): Both 2D and 3D integration paths in `step()`
  replaced per-granule Python for-loops with vectorized NumPy operations: force→velocity,
  velocity cap, position update, boundary clamping, rotational dynamics. Active noise in
  `compute_forces()` and `compute_forces_3d()` also vectorized.
- **Walltime estimation** (`run_all_trials.py`): Power-law scaling model
  `T = α * N^β * (steps/ref_steps)` with RSA efficiency correction and safety factor.
  `estimate_walltime()` reads trial JSON, predicts wall time, and `run_hpc()` uses the
  maximum estimate across all trials for SLURM `--time`.
- **Scaling benchmark trials** (`Trials/Trial22-29`): 8 trials varying domain size from
  200 µm to 2000 µm cube (6 to ~7300 granules) for compute-time benchmarking.
- **Wall time recording**: `metadata.json` now includes `wall_time_s` and `n_steps` after
  simulation completes.

---

## [V1.5.2] - 2026-03-14

### Added
- **Per-contact data storage** (`new_dem_0.py`): `compute_forces_3d()` and `compute_forces()`
  now return a third element — a list of contact dicts with per-contact fields: `i`, `j`
  (granule pair), `cx/cy/cz` (contact point), `nx/ny/nz` (contact normal), `overlap`,
  `R_eff`, `F_normal`, `A_contact`. Contacts propagated through `step()`, `save()`, and
  serialized to disk in `.npz` snapshots as structured arrays (`contact_i`, `contact_j`,
  `contact_cx`, etc.).
- **`viz_stress.py`**: New PyVista-based 3D visualization script for surface stress fields.
  - `granule_surface_mesh()`: Generates PyVista PolyData mesh from superellipsoid parametric
    surface with quaternion rotation and world-frame translation.
  - `compute_surface_stress()`: Maps Hertzian contact pressure distribution
    `p(r) = p0 * sqrt(1 - r²/a_c²)` to mesh vertices, with Gaussian smoothing tail.
  - `reconstruct_contacts()`: Fallback contact detection for older snapshots without
    stored contact data.
  - `plot_stress_cross_section()`: Z-slab cross-section (thickness = 1 functional granule
    diameter) showing continuous surface stress colormap. Top-down camera, multi-timepoint.
  - `select_interesting_granules()`: Auto-selects granules by composite score (contact count,
    force magnitude, bridge count), excluding boundary granules.
  - `granule_evolution_gif()`: Isolated granule group time evolution in isometric view with
    surface stress colormap and cell ellipsoid rendering. Fixed color scale across frames.
  - `cell_ellipsoid_mesh()`: Volume-conserving ellipsoid meshes for cells — sphere for
    unattached, oblate for spreading, prolate for bridging (elongated toward target).
  - `plot_granules_3d()`: Isosurface rendering with mid-slice opaque (`opacity=1.0`) and
    remaining granules as ghosts (`opacity=0.1`).
  - `run_all()`: Public API + CLI with `-i`/`-o`, uses `load_run()`.

### Changed
- **Instant cell attachment**: Cells now start as ATTACHED (not UNATTACHED) with
  `t_attach_onset=0.0` and `t_attach_half=0.0` (instant full attachment). When
  `t_attach_half <= 0.01`, the sigmoidal kinetics are bypassed and all cells attach
  immediately.
- **Bridge formation kinetics** (`new_dem_0.py`): Bridge formation is now a gradual,
  stochastic process instead of deterministic and instantaneous:
  - **Probabilistic initiation**: Each eligible cell has a Poisson-distributed probability
    of finding a bridge target per timestep (`bridge_attempt_rate`, default 0.3/h), modulated
    by gap proximity. Only SPREADING or PROLIFERATING cells with sufficient FA maturity
    (`min_fa_for_bridge`, default 0.3) can attempt bridges.
  - **Maturity ramp**: New bridges start weak and ramp to full motor-clutch force over
    `bridge_formation_time` (default 2 h), modeling filopodia extension, migration to gap
    edge, and adhesion formation on the target granule.
  - **Persistent bridges**: Committed BRIDGING cells retain their state across timesteps
    (no longer reset to ATTACHED each step). Bridge age tracked in `cell_bridge_age` array.
  - **Bridge senescence**: After sustained mechanical load (`bridge_senescence_time`,
    default 24 h), bridging cells transition to SENESCENT — they do not detach.
  - **Bridge rupture**: If the gap exceeds `bridge_break_gap` (default 60 µm), the bridge
    ruptures and the cell goes senescent (mechanically damaged).
  - New parameters: `bridge_attempt_rate`, `bridge_formation_time`,
    `bridge_senescence_time`, `min_fa_for_bridge`, `bridge_break_gap`.
- **Phase field isosurface rendering** (`viz_stress.plot_granules_3d()`): `granules_3d.png`
  now uses phase field isosurfaces (`phi_f`, `phi_i`) so overlapping/touching granules
  merge into a continuous solid, matching the physical scaffold. Mid z-slice is opaque,
  rest rendered as ghosts. Falls back to individual parametric meshes when phase fields
  are unavailable.
- **Bridge cell rendering** (`viz_cells.py`): Replaced dumbbell shape with volume-conserving
  elongated ellipse spanning the bridge gap. Cell width derived from area conservation
  (`A_cell / (π * a_long)`), providing realistic fibroblast morphology.
- **`plot_granules()`** (`new_dem_0.py`): 3D mode now delegates to
  `viz_stress.plot_granules_3d()` for isosurface rendering with z-slice opacity, falling
  back to 2D circle projection if PyVista is unavailable.
- **`run_all_trials.py`**: Added `viz_stress.run_all()` to local trial pipeline after
  `viz_cells`.
- **`viz_cells.run_all()`**: In 3D mode, automatically delegates stress rendering to
  `viz_stress` if PyVista is available.

### Bug Fixes
- **Watertight granule meshes** (`viz_stress.granule_surface_mesh()`): Fixed hollow shell
  rendering caused by open parametric surface seams. Omega now uses `endpoint=False` with
  modulo-wrapped face connectivity, and single-vertex pole caps replace degenerate pole rings.
  Meshes are verified manifold via `mesh.is_manifold`. `compute_normals(consistent_normals=True)`
  ensures correct outward face normals for solid rendering and clipping.
- **Watertight cell ellipsoids** (`viz_stress.cell_ellipsoid_mesh()`): Replaced open parametric
  mesh with `pv.Sphere()` primitive scaled by axis radii — guaranteed watertight with proper
  normals.

---

## [V1.5.1] - 2026-03-14

### Added
- **`viz_cells.py`**: New visualization script for cell and stress analysis (2D + 3D).
  - `plot_stress_map()`: Granules colored by net force magnitude (`YlOrRd` colormap)
    with directional arrows (quiver). Multi-panel layout at selected timepoints.
  - `plot_cell_states()`: Fibroblast-like cell morphology rendered as matplotlib patches.
    Stellate/star shapes for proliferating cells, dumbbell-shaped bridging cells that span
    surface-to-surface with contact patches at both ends, round blobs for unattached.
  - `cell_timelapse_gif()`: Animated GIF of cell migration, bridge formation, and pulling
    across all snapshot timepoints. Uses PillowWriter with imageio fallback.
  - `run_all()`: Public API matching other viz scripts, plus CLI with `-i`/`-o`.
  - 3D support via XY projection (circles for granules, quaternion-rotated cell positions).
- **Cell migration** (`new_dem_0.py`): Non-bridging attached cells (ATTACHED, SPREADING,
  PROLIFERATING) now perform a random walk on the granule surface each timestep.
  Controlled by `Params.cell_migration_speed` (default 5.0 µm/h). Applies in 2D
  (`cell_theta_local`) and 3D (`cell_eta_local`, `cell_omega_local`).

### Changed
- **`new_dem_0.py` `save()` closure**: In-memory snapshot dicts now include `force_x`,
  `force_y`, `cell_state`, `cell_granule_id`, `cell_theta_local`, `cell_fx`, `cell_fy`,
  `cell_bridge_target`, `cell_contact_area`, `cell_offset`. 3D mode also adds `force_z`,
  `cell_fz`, `cell_eta_local`, `cell_omega_local`. Previously these were only in on-disk
  `.npz` files — now available to viz scripts called in-process.
- **`update_cell_state()`**: Now accepts optional `rng` parameter, passed through to
  `_update_individual_cells()` for cell migration random walk.
- **Bridge cell rendering**: Bridging cells now render as dumbbell shapes spanning from
  the host granule surface to the nearest point on the target granule surface, with
  contact patches at both attachment points (radius proportional to contact area) and
  a thin process in the middle. Previously rendered as small fixed-size shapes with
  dashed lines to the target granule center.
- **`run_all_trials.py`**: Added `viz_cells.run_all()` to local trial pipeline.

---

## [V1.5] - 2026-03-14

### Added
- **Individual cell tracking (V1.5)**: Each cell is tracked individually with its own
  state (`CellState` enum: UNATTACHED, ATTACHED, SPREADING, PROLIFERATING, BRIDGING,
  SENESCENT), surface position on its host granule, force vector, contact area with
  host granule, and bridge target granule. Cell data stored as flat numpy arrays
  indexed by `cell_offset` per granule.
- **CellState enum**: `IntEnum` with 6 discrete states, extensible for future cell types.
  Stored as integers in numpy arrays for efficient serialization.
- **Data serialization**: Full simulation data now saved to disk during `run()`.
  Per-timepoint `.npz` snapshots include granule positions, velocities, forces, shapes,
  orientations, and all per-cell tracking arrays. Scalar metrics saved as `history.csv`
  (pandas-loadable) and `history.json` (exact fidelity). Phase fields optionally saved
  separately. All output archived as `.tar.gz` for HPC transfer.
- **`load_run()` function**: Load a complete simulation from disk (directory or `.tar.gz`).
  Returns `(hist, snaps, p, metadata)` matching the `run()` return signature.
- **`load_cells()` function**: Targeted loading of per-cell data for specific snapshots.
- **New Params fields**: `save_data` (bool, default True), `save_fields` (bool, default
  False), `output_dir` (str), `compress_archive` (bool, default True).
- **Granule velocity and forces in snapshots**: `vx`, `vy`, `vz`, `force_x`, `force_y`,
  `force_z` now included in serialized data.
- **`_record_bridge_forces()` helper**: Distributes bridging forces to individual cells
  during force computation.
- **`_initialize_cells()` helper**: Distributes cells uniformly on functional granule
  surfaces during packing.

### Changed
- **`update_cell_state()`**: Now also updates individual cell states via
  `_update_individual_cells()`. Per-granule aggregates remain authoritative for force
  computation (backward compatible, identical physics to V1.4.1).
- **`compute_forces()` / `compute_forces_3d()`**: Now record per-cell bridging forces
  and bridge targets. Resets cell force arrays at start of each call.
- **`save()` inner function in `run()`**: Now writes to disk when `p.save_data=True`.
- **Viz scripts**: All 4 viz scripts (`viz_compaction`, `viz_percolation`, `viz_movies`,
  `viz_phases`) can now load full simulation data from disk via `load_run()` in their
  `__main__` blocks.
- **`run_hpc_headless.py`**: `--output-dir` now sets both figure and data serialization paths.
- **`run_all_trials.py`**: Sets `p.output_dir` per trial for data serialization.

### Output Format
```
results/<run_name>/
  params.json          — Params dataclass as JSON dict
  metadata.json        — Version, git hash, seed, domain, cell state enum mapping
  history.csv          — Scalar metrics (one row per save point, pandas-loadable)
  history.json         — Scalar metrics (exact numeric fidelity)
  snapshots/
    snap_0000.npz      — Granule + cell arrays at t=0
    snap_0001.npz      — etc.
  fields/              — Optional (save_fields=True)
    fields_0000.npz    — phi_f, phi_i, phi_v grids
  <run_name>.tar.gz    — Archive of everything
```

---

## [V1.4.1] - 2026-03-14

### Added
- **Packing settle phase**: After RSA placement, `_settle_packing_2d()` and
  `_settle_packing_3d()` run isotropic compression micro-steps (centripetal
  attraction + Hertz repulsion) to bring granules into contact. Controlled by
  `Params.packing_settle_steps` (default 200).
- **Boundary exclusion zone**: `Params.boundary_exclusion` (default 0.2) specifies
  the fraction of the domain excluded from each edge when computing metrics. Only
  granules and field voxels in the inner region are used for connectivity, porosity,
  permeability, and compaction metrics.
- **Flat Trial JSON format (V1.4+)**: Trial configs can now use a flat key-value
  format where keys map directly to `Params` field names. Detected by `"_format": "flat"`
  or presence of Params field names at top level. Legacy nested format still supported.
- **New Params fields**: `packing_gap` (default 0.0 µm), `boundary_exclusion`
  (default 0.2), `packing_settle_steps` (default 200).
- **Example Trial configs**: `Trial15_3D.json`, `Trial16_3D.json` (flat 3D),
  `Trial17_2D.json` (flat 2D).
- **HPC result sync script**: `hpc/sync_results.py` — standalone script to check
  job status and rsync results from the cluster. Supports `--list-only` flag.

### Changed
- **Packing gap**: Default changed from 2.0 µm to 0.0 µm (`Params.packing_gap`).
  Granules now start in contact rather than with artificial spacing.
- **Seed handling**: `run()` default seed changed from 42 to `None` (random).
  `run_all_trials.py` default `SEED = None`. Each run produces a unique packing.
  For HPC, a random seed is generated once and embedded in the SLURM script.
- **`load_trial_json()`**: Now auto-detects flat vs legacy JSON format. Flat format
  maps keys directly to Params fields; unrecognized keys produce a warning.
- **`compute_metrics()`**: All field-based metrics (phi means, connectivity, porosity,
  compaction ratio, Kozeny-Carman) now computed on the inner region after boundary
  exclusion. Reports `boundary_exclusion` and `n_granules_inner` in metrics dict.
- **Legacy Trial configs removed**: Deleted Trial1–Trial12 (legacy nested format).
  Replaced with Trial15–17 using flat format.

### Fixed (HPC Best Practices Audit)
- **`squeue -r` for array jobs**: All `squeue` calls now use `-r` flag to expand
  array sub-tasks into individual rows. Without `-r`, `squeue` shows parent array
  job as single "R" line while any sub-task runs, preventing completion detection.
- **Removed `--mem` from SLURM directives**: Puma allocates 5 GB/CPU automatically
  via `--cpus-per-task`. Specifying both `--mem` and `--cpus-per-task` can cause
  invalid resource requests or redirect to high-memory queues.
- **`slurm_logs/` directory**: SLURM opens output files before script body executes.
  Directory is now created during repo sync (`_sync_repo_to_cluster`) instead of
  inside the script body.
- **`$HOME` in SLURM scripts**: `venv_path` and `repo_path` now use `$HOME` instead
  of `~` inside generated SLURM script content. `~` is kept for SSH/rsync commands
  where the remote shell expands it.
- **Trial validation**: Array job scripts now validate that `$TRIAL` is non-empty
  and exit with error if no trial found for the array task ID.
- **`seff` reminder**: All job completion messages now remind to run `seff JOBID`
  for CPU/memory efficiency checking.
- **Per-task progress tracking**: `_wait_and_sync()` rewritten to show
  `[N done, M running, K pending]` with incremental result sync.
- **Dead code cleanup**: Removed unused `mem_gb` default from `generate_hpc_scripts.py`.

---

## [V1.4] - 2026-03-14

### Added
- **Full 3D volumetric simulation**: Three simulation modes controlled by `Params.mode`:
  `"2D"` (pure V1.3 behavior), `"2D-slice"` (generate 3D packing, slice at z-midplane,
  run 2D simulation), `"3D"` (full 3D overdamped particle dynamics with volumetric
  phase fields).
- **Superellipsoid granule shapes**: 3D analog of superellipses —
  `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1` with equatorial blockiness (n1),
  meridional blockiness (n2), and three semi-axes (a, b, c). Spheres are the special
  case a=b=c, n1=n2=2.
- **Superellipsoid geometry utilities**: `superellipsoid_volume()` (Jaklic & Leonardis 2000),
  `superellipsoid_point()`, `superellipsoid_normal()`, `superellipsoid_curvature_radii()`,
  `superellipsoid_implicit()`, `superellipsoid_implicit_world()`, `superellipsoid_mesh()`.
- **Quaternion orientation system**: Unit quaternion (w,x,y,z) representation for 3D
  rotational state. Utilities: `quat_multiply()`, `quat_conjugate()`, `quat_normalize()`,
  `quat_rotate()`, `quat_rotate_inv()`, `quat_to_rotation_matrix()`, `quat_from_axis_angle()`,
  `quat_random()`, `quat_integrate()`.
- **3D packing generator**: `generate_packing_3d()` — random sequential addition in
  Lx × Ly × Lz box with volume-preserving semi-axis scaling, random quaternion
  orientation, and bounding-sphere overlap checks.
- **3D contact detection**: `find_contact_spheres_3d()` for sphere fast-path,
  `find_contact_superellipsoids_3d()` for general superellipsoid–superellipsoid contact
  via 4-unknown Newton-Raphson (eta_i, omega_i, eta_j, omega_j).
- **3D wall contact**: `find_contact_wall_3d()` for 6 wall faces with Hertzian response.
- **3D force computation**: `compute_forces_3d()` with (N,3) force array, (N,3) torque
  array, 3D tangential friction, 3D cell bridging, 3D active noise.
- **3D integration**: Quaternion rotational integration via `quat_integrate()`,
  3D translational overdamped Euler, 6-face wall clamp.
- **3D volumetric rendering**: `render_fields_3d()` on Ngrid_3d³ grid with
  bounding-box clipping per granule for performance.
- **2D-slice mode**: `slice_superellipsoid_z()` slices a 3D superellipsoid at z-midplane
  to produce a 2D superellipse cross-section. `generate_packing_2d_slice()` generates
  3D packing then slices for 2D simulation.
- **Transport metrics**: `kozeny_carman_permeability()` (K = ε³d²/[180(1−ε)²]),
  `rcp_fraction_superellipsoid()` (φ_RCP ~ 0.64 + 0.08*(AR−1)), porosity, d_grain_mean,
  phi_solid, phi_RCP, compaction_ratio, Da_number added to `compute_metrics()`.
- **Visualization script: `viz_compaction.py`**: Void fraction, packing fraction,
  compaction ratio, void size distribution, stacked phase evolution.
- **Visualization script: `viz_percolation.py`**: Kozeny-Carman permeability, Darcy flow
  rate, porosity with RCP reference, dimensionless groups dashboard (2×3: K, compaction
  ratio, Darcy number, Peclet, Reynolds, porosity ratio), void connectivity.
- **Visualization script: `viz_movies.py`**: Rotating 3D isosurface GIF, time-lapse
  compaction, z-sweep cross-section, composite 2×2. PyVista primary, matplotlib fallback.
- **Visualization script: `viz_phases.py`**: Three-panel isosurface strip at selected
  times, phase volume fractions vs time, tri-plane evolution (XY/XZ/YZ midplanes),
  phase interface area vs time.
- **Example 3D trial config**: `Trials/Trial13_3D.json` with `"mode": "3D"` and
  3D-specific fields (Lz_um, aspect_ratio_c_range, blockiness_n2_range, Ngrid_3d).
- **Optional Numba JIT**: `@njit` decorator for Newton-Raphson solver and geometry
  functions. Falls back to pure Python/NumPy when Numba unavailable. Controlled by
  `Params.use_numba`.
- **New `Params` fields**: `mode`, `Lz`, `aspect_ratio_c_func/inert_mean/std`,
  `blockiness_n2_func/inert_mean/std`, `omega_max_3d`, `Ngrid_3d`, `use_numba`.
- **New `GranuleSystem` arrays**: `z`, `vz`, `c`, `n1`, `n2`, `quat` (N,4),
  `omega_3d` (N,3), mode-aware `positions()`, `is_3d` property.

### Changed
- **`GranuleSystem`** is now mode-aware: stores `z`, `vz`, `quat`, `omega_3d` in 3D
  mode; `theta`, `omega` in 2D mode. `r_bound = max(a, b, c)` in 3D.
- **`compute_forces()`** dispatches to `compute_forces_3d()` in 3D mode. Returns
  `(F, torques)` where F is (N,3) and torques is (N,3) in 3D.
- **`step()`** dispatches to 3D integration (quaternion rotation, 6-face clamp) in 3D mode.
- **`render_fields()`** dispatches to `render_fields_3d()` in 3D mode.
- **`compute_metrics()`** adds transport metrics (porosity, K, compaction_ratio, etc.)
  in all modes.
- **`run()`** dispatches to mode-appropriate packing generator. Snapshots are now dicts
  (not tuples) with named keys.
- **`run_hpc_headless.py`** supports `mode`, `Lz`, `Ngrid_3d` from Trial JSON configs.
- **Console output** shows transport metrics (porosity, K, compaction ratio) per save step.
- **`compute_displacement()`** handles 3D positions (z0 parameter).

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
