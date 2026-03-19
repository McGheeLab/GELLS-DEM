# GELS Changelog

All notable changes to the GELS simulation engine are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering: `MAJOR.MINOR` where MAJOR tracks breaking API changes and
MINOR tracks feature additions and improvements.

---

## [V2.6] - 2026-03-18

### Added (2D vs 3D DOE Comparison)
- **`viz/doe_2d_vs_3d.py`**: New comparison analysis script for matched 2D/3D DOE runs
  (same LHS design matrix, seed=2025). 7 analysis modules: paired scatter plots with OLS
  regression, dimensionless collapse (φ/φ_RCP, Z/Z_iso, K/d²), factor importance heatmaps
  (Pearson r, 2D vs 3D vs Δ), time evolution overlay (2D blue vs 3D red, mean ± σ),
  scaling regression table (3D = α + β·2D with R² bar chart), summary radar chart, and
  3D/2D ratio vs factor values (parameter-dependent scaling detection).
- **Key finding**: Permeability (K) is the only metric with strong 2D→3D collapse
  (R²=0.94, slope=0.83). Connectivity, bridges, and contacts show dimension-dependent
  scaling that cannot be reduced to a simple power law.

### Added (HPC Storage Protection & Resume)
- **Results to /groups**: New `results_path` field in `hpc/Alex.json` directs simulation
  output to `/groups/mcgheealex/results` (500 GB) instead of `/home` (50 GB). SLURM
  templates use absolute paths. All sync functions (`_do_sync`, `_wait_and_sync`,
  `_wait_and_sync_multi`, `sync_results.py`) updated to read from `results_path`.
- **Disk space guard**: `check_disk_space()` function checks available space before each
  snapshot save. < 5 GB → skip fields (save particle data only). < 1 GB → skip snapshot
  entirely. SLURM pre-flight bash check aborts job if < 2 GB free.
- **Fields off by default**: `save_fields` default changed from `True` to `False`. Phase
  field grids (phi_f/phi_i/phi_v) are no longer saved during simulation — they're
  reconstructable from particle positions at plot time. Roughly halves storage per run.
- **Staggered starts**: `max_concurrent` in Alex.json (default 10, was 200). Plus 0-30s
  random sleep per task to reduce I/O storms.
- **Resume from snapshot**: `resume_from` parameter in Params. `find_last_snapshot()` and
  `restore_gs_from_snapshot()` reconstruct full GranuleSystem from saved .npz. `run()`
  supports resume branch: skips packing, loads state, continues from last saved timestep.
  `--resume-from` CLI arg in `run_hpc_headless.py`.
- **Batch resume (RUN_MODE=4)**: `run_all_trials.py` mode 4 scans local results for
  incomplete trials, uploads last snapshot to HPC via `upload_for_resume()`, generates
  resume SLURM scripts, and submits.

### Added (DEM vs Contact Network Comparison)
- **`viz/doe_dem_vs_network.py`**: Validates stochastic contact-network model as a surrogate
  for expensive 3D DEM by running both on identical initial packings from the 20-point DOE.
  Custom ensemble: bypasses built-in `ensemble()` to use `from_packing()` with repeated
  `solve()` on the exact DEM snapshot packing. 5 analysis modules: paired scatter (DEM vs
  Network final values ± σ), time evolution overlay (DEM trajectories vs ensemble mean),
  predictability table (R², slope, DEM/Net ratio as CSV + bar chart), network-only metrics
  (percolation onset, cluster sizes, spanning probability), and summary dashboard (cost
  comparison + R² heatmap). Comparable metrics: contacts, bridges, coordination Z,
  permeability, porosity, bridge force, locked bridges. Network-only: percolation, cluster
  size, bridge fraction.

### Bug Fixes
- **Bridge exclusion cap in stochastic model**: `compute_bridge_rates()` in
  `contact_network_model.py` now enforces a per-edge bridge limit derived from
  `bridge_exclusion_angle` and hemisphere geometry (3D: 2/θ², 2D: π/(2θ)). Previously
  the bridge count per edge was only limited by cell availability, producing ~200× too many
  bridges in 3D (14,969 vs DEM's ~2,700).
- **Periodic BC position wrapping in ContactNetwork**: `_rebuild_edges()` in
  `contact_network.py` now wraps positions into [0, L) before building the periodic cKDTree.
  Previously, particles that drifted outside the domain caused `ValueError: Some input data
  are greater than the size of the periodic box`. Also added initial wrapping in
  `from_snapshot()`.
- **LS-DEM epsilon serialization**: `save_snapshot_to_disk()` now saves `epsilon` and
  `d_epsilon` arrays to the .npz file. Previously only the in-memory snapshot dict
  included them, so resumed deformable runs would lose deformation state.

---

## [V2.5] - 2026-03-17

### Added (Mesoscale Models)
- **`analysis/spatial_pde.py`**: Spatially-resolved 1D radial PDE model for scaffold
  compaction. Extends the mean-field ODE with N_x=20 grid points ξ ∈ [0,1] (center→edge).
  Captures compaction waves, local jamming fronts, heterogeneous porosity/permeability
  profiles. State: x_f(ξ,t) + φ_tissue(ξ,t). Physics: cell traction stress vs contact
  resistance + stress-driven diffusion, with zero-flux BCs. Includes fitting to DEM data
  (η_eff, σ_0, α), DEM-snapshot initialization via radial binning, and tissue descriptor
  computation. Extracted and refactored from `parameter_sweep.py` into standalone class.
- **`analysis/contact_network.py`**: Graph-based contact network model. Nodes = granules,
  edges = contacts/near-neighbors. Tracks bridge state machine (NONE → FORMING → ACTIVE →
  LOCKED/SENESCENT), Hertz + DMT contact forces, overdamped position updates. Spheres only
  (no superellipsoid solver — the key speedup). Captures: coordination Z(t), bridge
  percolation thresholds, cluster size distributions, force chain statistics. Supports
  Monte Carlo ensemble via `run_ensemble()`. Can initialize from DEM snapshot, random
  packing, or Params alone.
- **`analysis/contact_network_model.py`**: Stochastic contact-network model with Gillespie
  SSA for exact bridge kinetics. Key differences from `contact_network.py`:
  (1) Gillespie algorithm (Bortz-Kalos-Lebowitz 1975) replaces per-timestep probability
  evaluation — every bridge formation/senescence event advances physical time exactly.
  (2) Union-find percolation tracking (Newman-Ziff 2001) with O(N) cluster statistics
  and spanning detection.
  (3) Monte Carlo ensemble with statistical summary: mean, IQR, min/max for all observables
  including bridge count, Z_bridge, spanning probability, permeability.
  (4) Self-contained sphere packing via Lubachevsky-Stillinger inflate-and-relax.
  Physics: JKR adhesive contact, motor-clutch cell traction, path-dependent bridge
  probability (contact/void/inert-blocked), bridge lock-in, secondary migration boost.
  Runs ~2.5s per 72h realization (2D, ~43 granules). Interface: `from_params(p)` →
  `solve(model)` or `ensemble(model, n_runs=1000)`.
- **`viz/mesoscale.py`**: Visualization for both mesoscale models. PDE plots: kymograph of
  x_f(ξ,t), radial profiles at selected times, domain-averaged timeseries + stress balance.
  Network plots: granule+bridge snapshot, Z(t)/bridge/percolation/cluster evolution.
  Combined 4-panel summary showing spatial PDE + network topology together.

### Design (Mesoscale Architecture)
- Three complementary models fill the gap between full DEM and mean-field ODE:
  - **SpatialPDE** answers *where* compaction happens (spatial gradients, porosity maps)
  - **ContactNetwork** (deterministic) answers *how* the packing reorganizes (force chains)
  - **ContactNetworkModel** (stochastic) answers *what variability* to expect across
    realizations (percolation thresholds, spanning probability, ensemble statistics)
- All run in seconds (vs hours for DEM), all produce tissue descriptors for organ
  distance computation via existing `arch_distance.py`.
- Cross-feeding interface: PDE local φ_f(ξ) → expected Z(ξ) for network edges;
  Network percolation state → PDE effective viscosity η_eff(ξ).

---

## [V2.4] - 2026-03-17

### Added (viz2/ — V2 Visualization System)
- **New `viz2/` package**: Complete rebuild of 2D visualization as a unified, modular system.
  10 files, 6 analysis modules, all driven from scaffold map foundation.
- **`viz2/common.py`**: Centralised color schemes, drawing helpers (`draw_granule_patches`,
  `draw_cells_on_ax`, `cell_world_positions`), phase field re-rendering, and figure utilities.
  Eliminates ~150 lines of duplicated code per module vs the old `viz/` approach.
- **`viz2/scaffold_map.py`** (Module 1): Vector-drawn scaffold map (red=functional,
  green=inert, black=void) with cell overlays. Bridge cells drawn in crimson but counted
  as functional space in all data analysis. Single-frame API for movie compositing.
- **`viz2/voronoi.py`**: Voronoi tessellation engine with two modes — center-based (standard
  Voronoi from granule centers) and boundary-based "shrink-wrap" (seeds on superellipse
  surfaces, sub-cells merged per granule via ConvexHull). Pure-numpy Sutherland-Hodgman
  domain clipping (no shapely dependency). Shape factor computation (circularity, elongation,
  aspect ratio via PCA). Local phase fraction computation via point-in-polygon on phase
  field grid.
- **`viz2/voronoi_shapes.py`** (Module 2): Voronoi overlay on scaffold map, shape factor
  spatial maps (circularity/elongation/area), distribution histograms, and mean±std
  timeseries showing compaction trends in local geometry.
- **`viz2/phase_fractions.py`** (Module 3): Global phase fractions (stacked area +
  conservation check), local phase fraction heatmaps within Voronoi cells, inner-vs-outer
  timeseries showing compaction gradient, and local-vs-global heterogeneity scatter plots.
- **`viz2/energy_stress.py`** (Module 5): Stress/strain spatial maps via Gaussian
  coarse-graining (reuses `analysis.coarse_grain`). 6 energy mode spatial maps: cell
  traction, Hertz/JKR contact, granular friction, osmotic pressure, inert frustration,
  interfacial tension. Each computed on Ngrid×Ngrid grid with proper physics. Energy
  mode timeseries showing total per mode vs time.
- **`viz2/void_percolation.py`** (Module 6): Void cluster labeling (`scipy.ndimage.label`),
  spanning-cluster percolation detection, cluster size distribution P(s) with power-law
  fits, percolation evolution timeseries (status, largest fraction, count, mean size),
  Kozeny-Carman permeability tracking.
- **`viz2/movies.py`** (Module 4): GIF animations of scaffold map, Voronoi tessellation,
  local phase fractions, stress maps, and 6-panel energy modes. Frame-by-frame rendering
  via `imageio.mimsave()` with configurable FPS and max frames.
- **`viz2/postprocess.py`**: Unified orchestrator with `--skip` / `--only` filtering.
  Supports both disk-loaded data (`-i results/default`) and live simulation data.
  CLI: `python viz2/postprocess.py -i results/default [--skip movies] [--only scaffold]`.

### Added (viz2/ — 3D Analysis Extension)
- **3D infrastructure in `viz2/common.py`**: `is_3d()` mode detection, `slice_field_z_midplane()`
  for extracting 2D slices from (Ng,Ng,Ng) phase fields, `slice_snap_z_midplane()` for projecting
  3D granules to 2D cross-sections (superellipsoid slice radii `a*(1-(dz/c)^n2)^(1/n2)`, yaw
  from quaternion), `cell_world_positions_3d()` using `superellipsoid_point` + `quat_rotate`,
  extended `ensure_phase_fields()` with `render_fields_3d` support.
- **All 6 existing modules extended for 3D**: Each module detects `is_3d` and uses z-midplane
  slicing for 2D display while preserving full 3D analysis where appropriate (e.g.
  `void_percolation` checks percolation along all 3 axes, `energy_stress` slices coarse-grained
  3D fields at midplane).
- **New `viz2/scaffold_map_3d.py`** (Module 7): PyVista off-screen 3D rendering with
  superellipsoid meshes (parametric grid + pole caps), cell spheres colored by state, bridge
  tubes, z-midplane clip plane. Multi-panel composites assembled in matplotlib. Guarded by
  `HAS_PYVISTA`; silently skips when unavailable or data is 2D.
- **Voronoi all-granule tessellation** (`viz2/voronoi.py`): Both functional AND inert granules
  seed Voronoi. Functional cells clipped to granule boundary via Sutherland-Hodgman convex
  polygon intersection (`clip_polygon_to_convex`). Inert cells keep full Voronoi region.
- **Periodic boundary rendering** (`viz2/common.py`): Ghost images for granules/cells near
  domain edges via `_periodic_offsets`, minimum-image convention for bridge vectors via
  `_min_image`.
- **Verified on DOE_3D_0001** (433 granules, 4644 cells, 200^3 fields) and regression-tested
  on DOE_2D_0001 with 0 failures.

### Documentation
- **Mean field modeling teaching guide** (`CodeLog/MeanFieldModeling/MEAN_FIELD_GUIDE.md`):
  Comprehensive rewrite of the beginner's guide to mean field modeling. Now 11 sections
  covering: (1-5) progressive toy examples (coffee cooling, logistic growth, thermostat,
  Kozeny-Carman), (6) full GELS model derivation with worked code, (7) hands-on
  tutorial for fitting real DEM data with step-by-step walkthrough, (8) DOE sweep comparison
  across 19 trials, (9) scalar ODE to spatial PDE extension, (10) energy landscape
  thermodynamic view, (11) summary with 3-level model hierarchy table. Includes appendices
  with file reference, figure generation instructions, and quick-start cheat sheet.
- **Example figure generator** (`CodeLog/MeanFieldModeling/generate_examples.py`): Expanded
  from 10 to 14 examples. New real-data examples: (11) single DEM fit with 4-panel
  diagnostic, (12) DOE parameter comparison across all trials (compaction vs stiffness,
  composition, permeability, bridging), (13) multi-run trajectory overlay colored by
  stiffness, (14) energy landscape decomposition. All 14 figures generate successfully
  from `results/LHC/` DOE data.

### Added (Visualization)
- **Vector scaffold map visualization** (`viz/postprocess.py`): New `plot_scaffold_map()`
  function draws granules as exact matplotlib patches (circles/superellipses) colored by
  type — red=functional, green=inert, black=void — with cells overlaid using cell-state
  colors. Replaces the grid-based `plot_composite()` as the primary [4/17] pipeline output
  (`scaffold_map.png`). Eliminates all aliasing artifacts since rendering is vector-based
  rather than grid-discretized. Handles periodic boundary ghost images. Legacy composite
  still saved when phi fields are available.

### Bug Fixes
- **Rendering aliasing on large domains** (`new_dem_0.py`): Phase field rendering used a
  fixed `Ngrid=200` regardless of domain size, causing severe aliasing when domain ≫ 800 µm
  (e.g. Lx=2319 µm gave 11.6 µm/pixel vs 3 µm interface width — sub-pixel tanh transitions
  rendered as binary noise). Fix: `render_fields()` auto-scales Ng to ensure grid spacing
  ≤ 2× interface_width. `render_fields_3d()` auto-scales up to 200³ cap, and widens
  interface_width to grid spacing when voxels remain coarse. Affects all phi-field-derived
  metrics (connectivity, porosity, tissue fraction).
- **Effective radii ignored periodic boundaries** (`new_dem_0.py`): `compute_effective_radii()`
  and `compute_effective_radii_3d()` used non-periodic cKDTree and raw position differences,
  missing all cross-boundary overlaps. Fix: accepts optional `p` parameter; when
  `boundary_mode='periodic'`, uses `boxsize` and `minimum_image_disp`.
- **Metric pixel area used fixed Ngrid** (`new_dem_0.py`): `compute_metrics()` computed
  `dxg = p.Lx / p.Ngrid` but the actual rendering grid may be larger (auto-scaled). Fix:
  derives `dxg` from `phi_f.shape[0]`.
- **Wall contact clip normals inverted** (`new_dem_0.py`): JKR wall clip normals used the
  force direction `sign` (pointing into domain) instead of negated sign (pointing toward
  wall). This clipped the domain-facing side of wall-adjacent particles instead of the
  wall-facing side. Fix: `float(sign)` → `-float(sign)` in all 4 wall clip paths.

### Changed
- **2D packing: Lubachevsky-Stillinger inflate-and-relax** (`new_dem_0.py`): Replaced the old
  centripetal-attraction settle with the same deflate→inflate algorithm used by 3D. RSA now
  places granules at reduced radii (α = (φ_safe/φ_target)^(1/2)), then `_settle_packing_2d()`
  inflates to target with vectorised Hertz repulsion + velocity cap + post-inflation overlap
  relaxation. Achieves jammed 2D packing at Z ≈ 3–4, up from Z ≈ 0–1 with the old method.
  Uses existing params (`packing_settle_steps`, `packing_relax_substeps`,
  `packing_inflate_phi_safe`). No new parameters.

### Added
- **Experimental comparison analysis** (`analysis/experimental_comparison.py`): New module for
  comparing Day 1 experimental packing data (hand-segmented area fractions from confocal cross-
  sections) with simulation predictions and mean-field models. Parses Excel data (36 samples:
  9 size combinations × 4 functional ratios), computes derived quantities (enrichment, packing
  efficiency), and generates 9 diagnostic plots: phase fractions vs ratio, packing efficiency,
  phase enrichment, size ratio effects, phase continuity, continuity vs fraction (percolation
  analysis), experiment vs simulation prediction, ternary composition diagram, and size heatmap.
  Supports optional DEM packing runs for parity plots. CLI: `python analysis/experimental_comparison.py`.

## [V2.4] - 2026-03-16

### Changed
- **Delayed cell senescence with stacking tolerance** (`new_dem_0.py`): Overcrowded cells no
  longer go senescent immediately. Cells can stack up to `cell_stacking_max` layers (default 3.0)
  on a granule surface — they prefer crawling on granule surfaces but will crawl on each other
  when surface area runs out. Cells in the overcrowded-but-tolerated zone accumulate
  `cell_overcrowd_age`; senescence triggers only after `overcrowd_senescence_time` hours
  (default 12.0) of sustained overcrowding. If overcrowding resolves (e.g. cells migrate or
  bridge away), the timer resets. Cells beyond the max stacking capacity still go senescent
  immediately. New params: `cell_stacking_max`, `overcrowd_senescence_time`. New per-cell
  array: `cell_overcrowd_age` (serialized to snapshots).

---

## [V2.3] - 2026-03-16

### Changed
- **Path-dependent bridge probability** (`new_dem_0.py`): Bridge formation now depends on what
  lies between the cell and its target, modelled from a fibroblast's perspective. Three regimes:
  (1) **Contact** (gap ≤ 0): cells crawl between touching surfaces, boosted by
  `bridge_contact_factor` (default 5.0). (2) **Void gap**: filopodia must probe empty space,
  probability decays as `exp(-gap / bridge_decay_length)` (default λ=10 µm). (3) **Inert-blocked**:
  inert granule in line-of-sight further penalised by `bridge_inert_factor` (default 0.1).
  Replaces old linear proximity model (`1 - gap/sense_dist`). Ray-cast via vectorised
  bounding-sphere intersection classifies each functional–functional pair. Bridges can now
  form at contact (previously blocked by `gap > L_rest`). Service of committed bridges no
  longer gated by L_rest. New params: `bridge_contact_factor`, `bridge_decay_length`,
  `bridge_inert_factor`.

### Fixed
- **Granule pass-through bug** (`new_dem_0.py`): Added post-step overlap resolution
  (`_resolve_overlaps()`) to prevent granules from interpenetrating during simulation.
  Root cause: large timestep (dt=0.5h, max 10 µm/step) combined with sharp contact
  threshold (zero force at overlap ≤ 0) allowed approaching granules to jump past contact
  detection, especially when bridge forces matured at t≈3-7h. Fix: after each position
  integration, all pairs are checked via cKDTree bounding-sphere query. Pairs with overlap
  exceeding `max_overlap_frac × min(r_i, r_j)` (default 15%) are projected apart by half
  the excess. Handles 2D/3D and periodic/wall BCs. New parameter: `Params.max_overlap_frac`.

### Added
- **LS-DEM deformable particles** (`lsdem.py`, `new_dem_0.py`): Variational level-set DEM
  implementation following Henzel & Karapiperis 2026 (arXiv:2602.12895). Each particle gets
  per-particle scalar deformation DOFs ε_α(t) that modulate fixed spatial mode shapes Φ_α(x).
  Particle geometry tracked via pre-computed signed distance fields (SDF) on body-frame grids.
  Contact detection uses surface-node-to-SDF queries (replaces Newton-Raphson for deformable
  particles), yielding multi-point contact patches. Overdamped deformation dynamics:
  γ_def dε/dt + K ε = F_ε, integrated with unconditionally stable implicit Euler.
  New module `lsdem.py` contains all LS-DEM functions; main engine hooks in via conditional
  branches when `deformable_enabled=True`. Default is `False` (zero performance impact on
  existing V2.2 rigid simulations).
  - **SDF infrastructure**: `superellipse_approx_sdf()` / `superellipsoid_approx_sdf()` using
    gradient-normalized implicit function. Pre-computed on body-frame grids via `init_sdf_grids()`.
    Bilinear/trilinear interpolation (`query_sdf_2d/3d`) and gradient queries for contact normals.
  - **Surface node discretization**: Uniform parametric sampling (2D) or Fibonacci sphere
    projection (3D). Per-node area weights for force integration. Hemisphere culling eliminates
    ~50% of SDF queries.
  - **Deformation modes**: Mode 0 = axial compression (volume-preserving), Mode 1 = prolate-oblate
    shape change. Mode 2 (3D only) = volumetric (very stiff for ν≈0.49). Elastic stiffness
    K_αβ derived from linear elasticity with shear modulus G and bulk modulus K.
  - **Semi-Lagrangian SDF update**: Deformed geometry evaluated on-demand via inverse displacement
    mapping: φ(x,t) = φ₀(x − Σ ε_α Φ_α(x)). No grid recomputation needed.
  - **Deformation metrics**: `def_strain_mean/max/std`, per-mode statistics tracked in metrics.
  - New Params: `deformable_enabled`, `n_def_modes`, `sdf_resolution`, `n_surface_nodes`,
    `n_surface_nodes_3d`, `sdf_padding`, `def_drag_scale`, `def_eps_max`.
  - MC-DEM automatically disabled when `deformable_enabled=True` (LS-DEM captures multi-contact
    stiffening geometrically).
  - **Deformed rendering** (`new_dem_0.py`): `render_fields()` and `render_fields_3d()` now
    stamp deformed particle shapes when `deformable_enabled=True`, using semi-Lagrangian
    pull-back through the mode-shape displacement field. Grid points are mapped to each
    particle's body frame, the inverse deformation u(x) = Σ ε_α Φ_α(x) is subtracted, and
    the analytical implicit function is evaluated at the pulled-back coordinate. New functions
    `_stamp_deformed_2d()` and `_stamp_deformed_3d()`. Shows physically flattened contacts
    and shape changes in rendered phase fields.
  - **Deformation visualization** (`viz/postprocess.py`): Two new plot functions.
    `plot_deformation()` draws particles colored by total deformation strain |ε| using the
    inferno colourmap with a shared colourbar across timepoints. `plot_deformation_timeseries()`
    shows mean/max strain over time and per-mode amplitude breakdown with ±1σ bands. Both
    produce output only when deformation data is present (gracefully skip for rigid runs).
    Postprocess orchestrator updated from 15 to 17 steps. New skip key: `'deformation'`.
  - **JKR adhesive contact model** (`new_dem_0.py`): Replaced Hertz + DMT adhesion with
    Johnson-Kendall-Roberts (JKR) model throughout all force computation paths (2D rigid,
    3D rigid, 2D walls, 3D walls). JKR is correct for soft hydrogels where the Tabor
    parameter μ_T >> 1. New functions: `jkr_force_from_overlap()` (Newton-Raphson solve for
    contact radius from overlap), `jkr_contact_radius()` (analytical), `jkr_pulloff_force()`.
    JKR reduces exactly to Hertz when W_adhesion = 0 (verified numerically). Force:
    F = (4/3)E*a³/R* − √(8πWE*a³). All existing adhesion parameters (W_adh_ff, W_adh_if,
    W_adh_ii) now drive JKR instead of DMT. No new parameters required.
  - **Contact-face half-plane clipping** (`new_dem_0.py`): Per-particle contact clip planes
    stored during force computation (`gs.contact_clips`). Each contact generates a half-plane
    (2D) or half-space (3D) constraint: `clip_d = √(R² − a²)` where `a` is the JKR contact
    radius. During rendering, clip planes are applied as `sdf = max(sdf, plane_sdf)` to create
    flat faces at contact regions, eliminating unrealistic overlapping circles. New helper
    functions `_apply_clips_2d()` and `_apply_clips_3d()`. All stamp functions updated:
    `_stamp_circle_2d()`, `_stamp_superellipse_2d()`, `_stamp_deformed_2d()`,
    `_stamp_granule_3d()`, `_stamp_deformed_3d()`. Rendering functions `render_fields()` and
    `render_fields_3d()` pass per-particle clips to stamp calls.
  - **LS-DEM adhesion** (`new_dem_0.py`): JKR adhesion for LS-DEM deformable contacts applied
    as translational-only correction at the pair level in `compute_forces()` / `compute_forces_3d()`.
    After LS-DEM computes pure repulsive penalty forces (which drive deformation DOFs), the
    adhesion correction F_adh = F_JKR − F_Hertz is applied to translational forces only, keeping
    it out of deformation generalized forces F_ε. This prevents adhesion from driving unphysical
    deformation while correctly pulling particles together. Surface nodes in `lsdem.py` remain
    purely repulsive.
  - **Multi-stage bridge formation** (`new_dem_0.py`, `viz/cells.py`): New `MIGRATING` state
    (CellState value 5) implements directed cell crawling before bridge formation. Lifecycle:
    PROLIFERATING → MIGRATING (directed crawl on granule surface) → BRIDGING (force ramp) →
    mature bridge. **Cell spatial awareness**: hemisphere check ensures only cells on the
    target-facing side of a granule can attempt bridges. Probability decays from cell centroid
    position (not granule center) using cell-specific gap computation. **Directed migration**:
    MIGRATING cells move along the host granule surface toward the contact point at
    `cell_migration_speed × bridge_directed_speed_mult` (default 2×). Cells within
    `bridge_commit_angle` (default 0.5 rad ≈ 29°) of the contact azimuth fast-path directly
    to BRIDGING. Abandonment: if target leaves sensing range, cell reverts to PROLIFERATING.
    2D: arc stepping via `cell_theta_local`. 3D: slerp on parametric unit sphere via
    `cell_eta_local`/`cell_omega_local`. New helper functions: `_cell_world_pos_2d/3d()`,
    `_compute_gap()`, `_angular_distance_to_target_2d/3d()`. New Params:
    `bridge_commit_angle=0.5`, `bridge_directed_speed_mult=2.0`. New metric:
    `n_migrating_cells`. Visualization: royal blue (#4169E1) for MIGRATING cells.
  - **Bridge exclusion zone** (`new_dem_0.py`): Cells avoid forming bridges near existing
    BRIDGING or MIGRATING cells on the same granule→target pair. Before attempting a bridge,
    each candidate cell's angular position is compared to all existing bridge cells on the
    same granule targeting the same neighbor. If any are within `bridge_exclusion_angle`
    (default 0.8 rad ≈ 46°), the cell skips and prefers to crawl around the granule to find
    an unoccupied region. Works in both 2D (theta arc distance) and 3D (great-circle distance
    on parametric sphere). New Param: `bridge_exclusion_angle=0.8`.
  - **Post-bridge stress fiber alignment** (`new_dem_0.py`): After entering BRIDGING state,
    fibroblasts progressively align their stress fibers and elongate along the bridge axis,
    increasing force output over time. New per-cell `cell_alignment` field (0–1) initialized
    at `bridge_alignment_min` (default 0.3) on BRIDGING entry, growing exponentially toward
    1.0 at rate `bridge_alignment_rate` (default 0.3 /h). Bridge force modulated as
    `F = F_mc × maturity × alignment`, giving a two-stage buildup: initial adhesion ramp
    (maturity, 0→1 over 2h) followed by continued force amplification as alignment improves
    (0.3→1.0 over ~6–8h more). Total time to full force ≈ 8–10h, matching real fibroblast
    remodeling timescales. Alignment resets on bridge rupture or senescence. New Params:
    `bridge_alignment_rate=0.3`, `bridge_alignment_min=0.3`. New per-cell array:
    `cell_alignment` (serialized to snapshots).

---

## [V2.2] - 2026-03-16

### Added
- **Multi-Contact DEM (MC-DEM) stiffening** (`new_dem_0.py`): Stress-based multi-contact
  correction for Hertzian contact forces (Giannis et al. 2021). When a soft granule has
  multiple simultaneous contacts, each contact sees a stiffer response due to volumetric
  confinement. Computes per-particle overlap strain ε_V = Σ δ/(2R), then scales Hertz
  repulsion by κ = 1 + ν/(1−2ν) × ε_V. For hydrogels at ν=0.49, this gives ≈2× stiffening
  for typical jammed packings (7 contacts at 1% overlap). Adhesion and friction unaffected.
  New function `mc_dem_correction()` applied as post-correction in both 2D `compute_forces()`
  and 3D `compute_forces_3d()`. New Params: `mc_dem_enabled=True`, `mc_dem_kappa_max=5.0`.
- **Comprehensive literature references** (`CodeLog/References/REFERENCES.md`): Added §20
  (superellipsoid packing & jamming: Donev 2004, Delaney/Cleary 2010, Yuan/O'Hern 2019,
  Jiao/Torquato 2009/2010), §21 (MC-DEM: Giannis 2021, Brodu 2015, Ghods 2022, Hirsch 2022),
  §22 (contact detection: Wellmann 2008, Canelas 2025, Lai 2022/2024, Gouveia 2025, Feng 2023),
  §23 (deformable DEM future: Rojek 2018/2021, Henzel/Karapiperis 2026, Feng 2025).
- **V3.0 LS-DEM upgrade roadmap** (`CodeLog/ClaudesPlan/V2.2_mc_dem_and_future_ls_dem.md`):
  Phased plan for deformable particle simulation via variational level-set DEM
  (Henzel & Karapiperis 2026). Four phases: SDF contact → level-set representation →
  deformation DOFs → new contact formation.

---

## [V2.1] - 2026-03-16

### Added
- **Cell surface coverage parameter** (`new_dem_0.py`): New `Params.cell_surface_coverage`
  (float, 0.5–1.5 typical) specifies cell loading as a fraction of granule surface area.
  At coverage=1.0 cells form a full monolayer; values >1.0 allow stacking (cells on top
  of each other). When set (>0), overrides `n_cells_per_granule` and `cell_coverage`.
  New helper `cells_from_surface_coverage()` computes per-granule cell count from 3D
  surface area (4piR^2) or 2D projected area (piR^2). Packing and overcrowding logic
  both respect the new parameter.
- **Two-stage adaptive DOE** (`Trials/generate_doe.py`, `Trials/generate_doe_stage2.py`):
  Replaced 5^7=78,125 full factorial with a two-stage adaptive strategy.
  - **Stage 1** (`generate_doe.py`): 500-run maximin Latin Hypercube Sampling across 7
    continuous factors: E_modulus (2–50 kPa), phi_solid_target (0.55–0.85), func_ratio
    (0.2–1.0), cell_surface_coverage (0.5–1.5), cell_sense_distance (20–80 µm),
    R_func_mean (30–150 µm), R_inert_mean (30–150 µm). ~9k CPU-hours (6% Puma monthly).
  - **Stage 2** (`generate_doe_stage2.py`): ~300 adaptive refinement runs generated after
    Stage 1 analysis. Fits quadratic response surfaces, computes variance-based sensitivity
    indices, and allocates points via four strategies: near-optimal (35%), steep-gradient
    (25%), high-uncertainty (20%), space-filling (20%). ~5.4k CPU-hours (4% Puma monthly).
  - Total budget: ~800 runs, ~14.4k CPU-hours (10% monthly allocation) vs 78,125 runs prior.
- **Tissue volume tracking** (`analysis/parameter_sweep.py`): New spatially-resolved state
  variable `phi_tissue(xi, t)` tracks cell + ECM volume fraction that grows within the
  functional zone of the scaffold. Cells adhered to functional granules occupy void space,
  and bridges between granules scaffold further tissue growth.
  - **Logistic growth ODE**: `d(phi_tissue)/dt = k_tissue × min(1, n_cells/n_ref) × maturity(t)
    × max(alpha_0, f_bridge × neighbor_factor) × max(0, alpha_fill × phi_void − phi_tissue)`.
    Driven by cell count, FA maturity, and bridge formation. No feedback into compaction
    mechanics (tissue is soft).
  - **TISSUE_PARAMS**: `k_tissue=0.05 h⁻¹` (base rate), `alpha_tissue_fill=0.6` (max void
    fill fraction), `alpha_tissue_0=0.1` (minimum growth without bridges),
    `n_cells_tissue_ref=10.0` (reference cell count).
  - **Tissue-corrected architecture descriptors**: `BV_TV_eff = phi_f + phi_tissue_global`,
    `porosity_eff = 1 − BV_TV_eff`, `K_f_tissue` via Kozeny-Carman with tissue-reduced void.
    Organ distance computation now uses these corrected values.
  - **Updated `_integrate_trajectory()`**: Returns 6-tuple (added tissue_traj, tissue_spatial)
    for single-sample kinetics visualization.
  - **Updated plots**: `plot_best_kinetics()` expanded to 1×3 panels (tissue volume vs time),
    `plot_radial_profiles()` expanded to 1×3 panels (tissue spatial profile).
  - **New `plot_tissue_effects()`**: 2×2 figure — (a) BV_TV_eff vs phi_f, (b) K_f_tissue vs
    K_f, (c) phi_tissue vs n_cells, (d) phi_tissue vs func_ratio.
  - **Updated recommendation table**: Added phi_tissue_global and BV_TV_eff columns.
  - **Updated radar chart**: Uses BV_TV_eff, porosity_eff, K_eff_tissue for organ comparison.
  - **Physical insight**: `n_cells_per_func` now affects architecture (not just compaction
    force). Dense organs (cardiac, liver) match better with tissue; porous organs (lung, bone)
    require few cells to avoid over-filling void space.
- **Lubachevsky-Stillinger jammed packing** (`new_dem_0.py`): Rewrote 3D packing initialization
  to guarantee fully jammed initial state (Z ≈ 6–8 contacts/granule). RSA places granules at
  deflated radii (α ≈ 0.6–0.7), then inflate-and-relax algorithm grows them to target size
  over 400 steps × 15 sub-steps with vectorized Hertz-like repulsion. Quadratic ease-in
  inflation schedule spends more time near final size where jamming is hardest. Replaces old
  centripetal-attraction approach that couldn't achieve high packing fractions.
  - New Params: `packing_inflate_phi_safe=0.20` (initial deflated packing fraction for RSA).
  - Updated defaults: `packing_settle_steps=400`, `packing_relax_substeps=15`.
- **Persistent bridge lock-in** (`new_dem_0.py`): Bridges that exceed `bridge_lock_force_threshold`
  now set a persistent `cell_bridge_locked` boolean flag. Once locked, bridges persist
  indefinitely regardless of instantaneous force fluctuations. Previously, force drops below
  threshold on any timestep would reset lock status, causing unrealistic bridge cycling.
  Lock flag is reset only on gap rupture (bridge physically breaks). Serialized in snapshots.
- **DOE periodic BC optimization** (`Trials/generate_doe.py`): Rewrote DOE generator for
  periodic boundary conditions. Key changes:
  - Switched BASE config to `boundary_mode: "periodic"`, `boundary_exclusion: 0.0`.
  - Reduced N_TARGET from 500 to 250 (periodic BCs eliminate boundary exclusion waste).
  - Added N_MAX=600 cap to prevent compute blowup from extreme size ratios.
  - Tightened R bounds from [30,150] to [40,120] µm to avoid heavy-tail compute distribution.
  - New `compute_domain_size()` enforces three constraints: N_target (particle count),
    minimum image (L > 2×cutoff for cKDTree periodic correctness), and RVE (L > 4×d_max
    for bulk packing statistics). Takes max of all three.
  - Reduced `Ngrid_3d` from 80 to 60 (3.4× faster field rendering).
  - DOE metadata now includes `_doe_n_target` and `_doe_constraint` per run.
  - Estimated ~5k CPU-hours for 500 runs (down from ~9k).
- **References** (`CodeLog/References/REFERENCES.md`): Added §18 Parisi & Zamponi (2010)
  "Mean-field theory of hard sphere glasses and jamming" and §19 Campello & Cassares (2016)
  "Rapid generation of particle packs at high packing ratios for DEM simulations."
- **Two-compartment volume conservation analysis** (`analysis/volume_conservation.py`):
  New module implementing Voronoi shrink-wrap compartment analysis for verifying global
  volume conservation. Assigns each voxel to the functional or inert compartment via
  nearest-granule cKDTree tessellation (periodic-BC aware). Uses exact granule geometry
  volumes (not rendered phase fields) for conservation accounting — `phi_solid_true` is
  provably constant to machine precision. Key functions: `voronoi_compartments()`,
  `compartment_metrics()`, `conservation_from_run()`. Generates 4-panel conservation
  figure (global conservation, compartment volumes, local void redistribution, cross-check)
  and compartment boundary slice overlay. Integrated as step 15/15 in `viz/postprocess.py`.
- **True granule-based volume metrics** (`new_dem_0.py`): Added `phi_f_true`, `phi_i_true`,
  `phi_solid_true` to `compute_metrics()` output. Computed from exact granule geometry
  (sphere: 4/3 pi R^3, superellipsoid: `superellipsoid_volume(a,b,c,n1,n2)`) — invariant
  during simulation since radii/shapes do not change. Provides a rendering-independent
  ground truth for volume conservation verification.
- **Inline Voronoi two-compartment metrics** (`new_dem_0.py`): Added per-timestep compartment
  tracking (x_f, x_i, phi_v_in_func, phi_v_in_inert, phi_solid_func, phi_solid_inert,
  phi_v_crosscheck) to `compute_metrics()` using the same Voronoi nearest-granule approach
  on the phase field grid.
- **Additive phase field blending** (`new_dem_0.py`): Changed 2D and 3D field stamping from
  `np.maximum()` (max-blending) to `+=` (additive blending). Max-blending loses volume when
  same-type granules overlap; additive blending preserves total deposited volume.
- **Iterative volume normalization** (`new_dem_0.py`): `render_fields_3d()` and `render_fields()`
  now apply iterative scale-and-cap normalization (up to 20 iterations) to match rendered
  phase field totals to true granule volumes. Residual error is <1% for typical packings
  (fundamental limit from per-voxel cap at 1.0 in dense overlap regions).

### Bug Fixes
- **Periodic boundary bridge visualization** (`viz/cells.py`): Fixed `_nearest_surface_point()`
  to apply minimum image convention when computing bridge target surface points. Previously,
  bridges crossing periodic boundaries were drawn spanning the entire domain width. Now correctly
  renders short bridges that visually terminate at domain edges.
- **Critical: phi_solid_target not applied at runtime** (`new_dem_0.py`): Fixed a bug where
  `phi_solid_target` and `func_ratio` overrides loaded from trial JSON configs were not
  propagated to `phi_f_target`/`phi_i_target`. The `Params.__post_init__` derivation runs at
  construction time, but trial configs apply overrides via `setattr()` after construction,
  leaving phi targets at defaults (0.25/0.20 = 0.45 total) instead of the intended values
  (0.55–0.85). Fix: added re-derivation at the start of `run()` so phi targets always
  reflect the current `phi_solid_target × func_ratio`. This affected all DOE runs using the
  `phi_solid_target` specification path.

---

## [V1.12] - 2026-03-16

### Documentation
- **REFERENCES.md**: Added 28 new literature references for energy landscape framework,
  organized into 9 new sections (§10–§18): Jamming Physics & Yield Stress (O'Hern, van Hecke,
  Olsson & Teitel, Tighe), Random Close Packing (Torquato, Farr & Groot, Donev, Yuan),
  Energy Landscape Theory (Wales, Kramers, Bi et al., Shi et al.), Poroelasticity (Biot,
  Brinkman), Interfacial Tension (Princen 1979/1983/1986, Durian), Polymer Dynamics
  (Doi & Edwards), Granular Hydrogel Mechanics (Cai et al., Di Caprio et al.), Tissue
  Architecture (Harrigan & Mann, Hildebrand & Ruegsegger, Hollister, Jaklic & Leonardis).
  Summary table §18 added for energy landscape parameter sources.
- **MATHEMATICAL_MODEL.docx**: Regenerated from updated markdown source with all new
  sections 11–14 (volume-conserving two-zone model, free energy landscape, computational
  results, design rules). Analysis figures inserted (110 total).

### Added
- **Jamming constraint** (`analysis/parameter_sweep.py`): Added physical requirement that
  initial solid packing must be jammed (`phi_solid >= phi_RCP`) to justify ignoring
  gravitational effects. Without jamming, granular scaffolds would collapse under gravity.
  - `phi_solid` sampling range narrowed from [0.10, 0.92] to [0.75, 0.95] to focus on the
    jammed regime and reduce wasted samples.
  - Feasibility filter in `run_sweep()` now computes `phi_RCP` before filtering and rejects
    samples below the shape-dependent jamming threshold.
  - New plot: `jamming_phase_space.png` — three-panel figure showing (a) phi_solid vs phi_RCP
    scatter with jamming boundary, (b) accessible BV/TV range with organ target lines, and
    (c) jamming margin distribution histogram.
  - Jamming boundary annotations (dashed line at phi_RCP=0.82) added to `compaction_heatmaps`
    and `organ_landscapes` plots where phi_solid is an axis.
  - Solid phase balance plot slices updated to reflect jammed phi_solid range [0.80-0.92].
  - Key physical insight: low-BV/TV organs (lung, bone) remain accessible in the jammed
    regime via low functional ratio (mostly inert granules provide structural jamming while
    sparse functional granules define tissue architecture).

---

## [V1.11] - 2026-03-16

### Added
- **Energy landscape analysis** (`analysis/energy_landscape.py`): New `EnergyLandscape`
  class decomposes the free energy of cell-driven scaffold compaction into six physically
  motivated terms as a function of compaction coordinate ξ = 1 − x_f/x_{f,0}:
  (1) G_cell — cell traction + bridge adhesion (negative, drives compaction, density-
  enhanced via ln(1−ξ));
  (2) G_elastic — Hertzian elastic contact (ODE-consistent parabolic onset at ξ_c where
  φ_local = φ_RCP);
  (3) G_yield — Herschel-Bulkley yield barrier near jamming (σ_y ~ σ_0·(φ/φ_J−1)^1.5);
  (4) G_void — osmotic void redistribution (quadratic in Δφ_v);
  (5) G_inert — geometric frustration from inert obstacles (quadratic in ξ);
  (6) G_surface — interfacial tension at functional/inert boundary (γ depends on modulus
  mismatch, favours compact round zones, penalises trapped inert granules).
  Provides: `find_equilibrium()`, `find_barrier()`, `solve_kinetics()` (overdamped
  dynamics with time-dependent bridge formation), `dimensionless_groups()` (β, Ca, Φ_r,
  Ψ, Γ), and `energy_decomposition_at_eq()`. Energies normalised by σ_cell for O(1)
  landscape structure. Calibrated to reproduce mean-field ODE equilibrium.
- **Energy landscape visualization** (`viz/energy_landscape.py`): Four publication-quality
  figures from the energy landscape analysis:
  (1) `energy_landscape_organs.png` — 2×2 grid showing G̃(ξ) with all 6 decomposed terms
  for 4 organ targets, marking equilibrium ξ* and barriers;
  (2) `energy_landscape_evolution.png` — time-evolving landscape as bridges form (t=0→72h)
  with kinetics trajectory ξ(t);
  (3) `energy_decomposition.png` — stacked bar chart of driving vs resisting energy terms
  at ξ* per organ, plus governing dimensionless numbers;
  (4) `energy_design_space.png` — heat maps of ξ* over (E_func, φ_f), (φ_i, R_func),
  and (E_func, E_inert) parameter planes for design rule extraction.
  CLI: `python viz/energy_landscape.py -i results/parameter_sweep`.
- **Timelapse temporal coherence** (`viz/scaffold_evolution.py`): Replaced per-frame
  independent position computation with frame-to-frame state propagation using damped
  motion (`_advance_positions()`) and pre-computed rendering (`_render_frame_from_pos()`).
  Each frame inherits positions from the previous frame and moves 18% toward the target,
  with gentle overlap resolution (10 iterations), eliminating unnatural inter-frame
  jostling of granules.

---

## [V1.10] - 2026-03-16

### Added
- **Periodic boundary conditions** (`new_dem_0.py`): New `Params.boundary_mode` parameter
  (`"walls"` default or `"periodic"`). When periodic, granule interactions use the minimum
  image convention via `scipy.spatial.cKDTree(boxsize=...)`, positions wrap via modulo, and
  wall forces are disabled. Eliminates artificial void accumulation near boundaries during
  compaction, enabling bulk scaffold property studies.
  - Packing (RSA + settle): Periodic placement in `[0, Lx)` with minimum-image overlap check.
    Settle phase uses random jitter instead of centripetal attraction.
  - Force computation: `compute_forces()` and `compute_forces_3d()` use periodic cKDTree,
    minimum-image displacements, and virtual positions for superellipse/superellipsoid contacts.
  - Step integration: Position wrapping via `wrap_positions()` instead of boundary clipping.
    Unwrapped positions (`x_unwrap`, `y_unwrap`, `z_unwrap`) track true displacement.
  - Field rendering: Ghost particle images stamped at periodic boundaries for correct phase fields.
  - Metrics: `boundary_exclusion` forced to 0 for periodic (full domain used). Contact
    diagnostics and bridge counting use periodic tree + minimum image.
  - Displacement: `compute_displacement()` uses unwrapped positions for correct RMS displacement.
  - Helper functions: `minimum_image_disp()`, `minimum_image_disp_3d()`, `wrap_positions()`.
  - Backward compatible: `boundary_mode='walls'` produces identical results to V1.9.

- **1D radial PDE mean-field model** (`analysis/parameter_sweep.py`): Replaced scalar ODE
  `x_f(t)` with spatially-resolved 1D PDE `x_f(xi, t)` on N_x=20 radial grid points,
  where xi is a normalized radial coordinate (center=0, edge=1).
  - **Neighbor-count modifier**: `neighbor_factor(xi) = 0.5*(1 + cos(pi*xi))` — cells at
    center sense all neighbors (factor=1), cells at edge sense none (factor=0). This
    naturally produces a dense core with dilute periphery.
  - **Stress-driven diffusion**: `D_eff * d²x_f/dxi²` with zero-flux BCs at xi=0 (symmetry)
    and xi=1 (edge). CFL stability clamped: `D_eff * dt / dxi² < 0.45`.
  - Domain-averaged outputs remain backward compatible with V1.9 sweep results.
  - New spatial outputs: `x_f_final_std` (radial heterogeneity), `x_f_gradient` (edge-center
    difference), `phi_v_f_local_std` (void fraction variability), `K_f_series` (harmonic mean
    permeability for radial flow).
  - `_integrate_trajectory()` updated to return spatial profiles for radial profile plotting.
  - New `plot_radial_profiles()` function: shows x_f(xi) and phi_v(xi) at final time for
    each organ's optimal scaffold.

---

## [V1.9] - 2026-03-16

### Added
- **Bridge force lock-in** (`new_dem_0.py`): Bridging fibroblasts whose force exceeds
  `bridge_lock_force_threshold` (default 20 nN) now prefer to stay in the bridge
  configuration indefinitely, bypassing the senescence timer. This models the
  biological observation that high-tension bridges stabilize rather than turn over.
  New parameter: `Params.bridge_lock_force_threshold`.
- **Secondary bridge migration** (`new_dem_0.py`): Non-bridging fibroblasts can migrate
  along existing bridges to form additional connections. When committed bridges already
  exist between two granules, the bridge attempt rate for new cells is boosted by
  `bridge_secondary_rate_mult` (default 3×). New parameter:
  `Params.bridge_secondary_rate_mult`.
- **Per-cell bridge force monitoring** (`new_dem_0.py`): Tracks force magnitude of every
  bridging cell at each save point. New history metrics: `n_bridging_cells`,
  `bridge_force_mean`, `bridge_force_max`, `bridge_force_min`, `bridge_force_std`,
  `n_locked_in_cells`. Run output now shows `bCells`, `bF_avg`, `lock` columns.
- **Bridge force diagnostic** (`new_dem_0.py`, `analysis/mean_field_model.py`): At end
  of simulation, compares measured bridge forces to `expected_bridge_force` (default
  100 nN) and warns if motor-clutch parameters may need tuning. New parameter:
  `Params.expected_bridge_force`.
- **Mean-field model updates** (`analysis/mean_field_model.py`): Bridge fraction now
  accounts for lock-in (no senescence turnover when F_cell ≥ threshold) and secondary
  migration rate boost. New constructor parameters: `bridge_senescence_time`,
  `bridge_lock_force_threshold`, `bridge_secondary_rate_mult`, `expected_bridge_force`.
- **`analysis/parameter_sweep.py`**: Mean-field parameter sweep and organ target prediction.
  Sweeps 6 physical parameters (E_modulus, phi_f_target, phi_i_target, R_func_mean,
  n_cells_per_granule, bridge_attempt_rate) through 50,400 combinations using the
  mean-field ODE model. Physics-based scaling laws derive fitting parameters (eta_eff,
  sigma_0, alpha) from granule mechanics without requiring DEM calibration data.
  Computes scaffold descriptors (BV/TV, porosity, Kozeny-Carman permeability) and
  architectural distance to all 7 organ targets for each combination.
  Outputs: `sweep_data.csv` (45K+ rows), `recommendations.json`, and 10 publication-
  quality figures including compaction heatmaps, solid phase balance diagrams,
  compaction driver analysis, organ distance landscapes, recommendation tables,
  radar comparisons, sensitivity tornado charts, closest-organ phase map,
  permeability-porosity space, and compaction kinetics for optimal scaffolds.
  CLI: `python analysis/parameter_sweep.py [-o DIR] [--quick] [--t-total H]`.
- **`viz/scaffold_evolution.py`**: 2D spatial maps showing time evolution of
  granular scaffold microstructure under cell-driven compaction. For 4 organ
  targets (trabecular bone, intestinal mucosa, kidney cortex, cardiac muscle),
  generates a packed 2D domain of superellipse granules (functional + inert)
  and applies the mean-field compaction trajectory to animate clustering over
  5 time snapshots (0, 6, 18, 36, 72 h). Shows cell bodies, cell bridges, and
  void redistribution. Companion kinetics plot shows x_f(t), local void
  fractions, and bridge fraction per organ.
  Outputs: `scaffold_evolution.png`, `scaffold_kinetics.png`.
  CLI: `python viz/scaffold_evolution.py -i results/parameter_sweep`.
- **`viz/scaffold_evolution_3d.py`**: 3-D volumetric version of the scaffold
  evolution visualization using PyVista off-screen rendering. Generates
  superellipsoid granule packings in a 3-D cubic domain, applies mean-field
  compaction trajectories, and renders semi-transparent isometric views showing
  interior clustering. Cell bridges rendered as tubes, cells as small spheres.
  Output: `scaffold_evolution_3d.png`.
  CLI: `python viz/scaffold_evolution_3d.py -i results/parameter_sweep`.
- **Packing constraint** (`new_dem_0.py`): Added `phi_solid_target` and `func_ratio`
  parameters as an alternative to setting `phi_f_target`/`phi_i_target` independently.
  When `phi_solid_target > 0`, derives `phi_f_target = phi_solid * func_ratio` and
  `phi_i_target = phi_solid * (1 - func_ratio)`, ensuring physically consistent total
  solid fractions (typically 0.55–0.75). Backward compatible: existing trials using
  `phi_f_target`/`phi_i_target` directly continue to work unchanged.
- **Parameter sweep packing physics** (`analysis/parameter_sweep.py`): Replaced independent
  `phi_f`/`phi_i` LHS sampling with constrained `phi_solid` (0.10–0.92) and `func_ratio`
  (0–1), deriving `phi_f = phi_solid * func_ratio` and `phi_i = phi_solid * (1 - func_ratio)`.
  Wide phi_solid range covers all 7 organ porosity targets: cardiac muscle (0.12) through
  lung alveoli (0.88). Initial x_f_0 clamped to deformable limit when initial packing
  already exceeds phi_max. Sweep now accepts configurable motor-clutch and bridge parameters
  via CLI.
- **Volume-conserving two-zone mean-field model** (`analysis/mean_field_model.py`,
  `analysis/parameter_sweep.py`): Fundamental physics rewrite. Granules are incompressible
  so phi_f + phi_i = phi_solid = CONSTANT. State variable changed from phi_f (which
  incorrectly decreased) to x_f (functional zone volume fraction). As cells compact,
  x_f shrinks → functional granules pack tighter (approaching RCP) → void expelled from
  functional zone → inert zone void increases. Global void fraction stays constant.
  Outputs: x_f_final, phi_v_f_local, phi_v_i_local, compaction_ratio, zone-weighted
  effective permeability. Organ distance uses heterogeneous pore structure.
  All 11 plots updated for new physics.
- **Deformable packing limit** (`analysis/parameter_sweep.py`, `analysis/mean_field_model.py`):
  Soft, deformable superellipsoid granules can now pack beyond the rigid-particle RCP. The
  hard floor `x_f_min = phi_f / phi_RCP` replaced with `x_f_min = phi_f / phi_max`, where
  `phi_max > phi_RCP` depends on granule compliance (1/E, soft = more packable) and
  superellipsoid blockiness (n2 > 2 = flatter faces = tighter interlocking). The contact
  resistance function still grows above phi_RCP, providing physical pushback. Implemented
  via `compute_phi_max_deformable()` in parameter_sweep.py and inline in CompactionModel.
  New output: `phi_max` in sweep results. Example: E=0.5 kPa, phi_RCP=0.82 → phi_max≈0.90;
  E=200 kPa → phi_max≈0.82 (near rigid limit).
- **3D volume conservation** (`new_dem_0.py`): Added `overlap_lens_volume()` for exact 3D
  sphere–sphere overlap volume, `compute_effective_radii_3d()` for volume-conserving display
  radii (3D analogue of existing 2D `compute_effective_radii()`). `render_fields_3d()` now
  uses effective radii for sphere mode, conserving total granule volume in the phase field.
  New metric: `volume_conservation` (3D analogue of `area_conservation`). Metrics distance
  calculations and force magnitudes now correctly use all 3 components in 3D mode.

- **Organ-specific phase mapping** (`analysis/organ_targets.py`, `analysis/parameter_sweep.py`):
  The three DEM phases map onto biological tissue architecture, but the mapping is
  **organ-specific**. Each organ defines `perfusive_void_fraction` (f_perf) specifying what
  fraction of non-tissue space is liquid-filled/perfusive vs structural void:
  - phi_f (functional) → tissue parenchyma → BV/TV  [universal]
  - phi_i (inert) → structural void spaces  [organ-dependent fraction]
  - phi_v (liquid) → perfusive channels      [organ-dependent fraction]
  Per-organ f_perf values: lung 0.05 (air sacs), bone 0.10 (marrow), intestine 0.15,
  pancreas 0.30, kidney 0.70, cardiac 0.85, liver 0.90 (sinusoidal blood).
  `compute_organ_distances()` rewritten with organ-specific logic:
  - Permeability: K_f (functional zone) for perfusive-dominated organs (f_perf≥0.5),
    K_i (inert zone) for structural-void-dominated organs (f_perf<0.5).
  - Void-split penalty: z-score penalising scaffolds whose phi_i/(phi_i+phi_v) deviates
    from the organ's expected (1 - f_perf).
  - Per-zone pore radii (r_pore_f, r_pore_i) now exported from sweep for zone-specific
    comparison.
  Granule radius ranges widened (R_func=[5,150] µm, R_inert=[5,300] µm)
  to cover all organ targets from liver sinusoids (8 µm) to trabecular bone spacing (600 µm).
- **Fixed parameter support** (`analysis/parameter_sweep.py`): Parameters can now be held
  constant rather than swept. `cell_sense_distance` fixed at 50 µm (filopodia sensing range).
  `FIXED_PARAMS` dict injected into sample arrays during LHS generation and trajectory
  integration. `func_ratio` constrained to [0.05, 0.95] to prevent degenerate zones.

### Bug Fixes
- **Volume conservation violation** (major): Previous mean-field model decreased phi_f
  during compaction, implying functional granules were losing volume. Fixed by switching
  to two-zone model where total solid fraction is always conserved.
- **Negative compaction ratio**: When phi_solid is high and func_ratio is high, x_f_0 could
  be less than x_f_min, causing spurious expansion. Fixed by clamping x_f_0 to
  max(x_f_0, x_f_min) in both vectorised sweep and single-trajectory integration.
- **cell_sense_distance KeyError**: After removing from swept parameters, `_integrate_trajectory`
  and `recommend_parameters` failed. Fixed by injecting FIXED_PARAMS into sample dicts.
- **Degenerate inert zone void fractions**: When func_ratio ≈ 1, phi_i is negligible and
  phi_v_i becomes meaningless (0.99). Recommendation table now shows "—" when phi_i < 0.02.
- Fixed `KeyError: 'phi_f'` in `plot_compaction_heatmaps` after packing constraint refactor.
- Fixed `KeyError` in `_integrate_trajectory` and `plot_best_kinetics` where sample dicts
  were built from `PARAM_NAMES` only, missing derived `phi_f`/`phi_i` keys.

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
- **`arch_distance.py`**: Weighted Mahalanobis-like architectural distance between GELS
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
