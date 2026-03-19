# GELS Project Guide

## Project Overview

**GELS** (Granule-Enabled Living Scaffolds)
is a 2D/3D overdamped particle dynamics simulator for modelling cell-driven rearrangement
of hydrogel granular scaffolds. The primary simulation engine is `new_dem_0.py`.

**Current version: V2.6**

## Repository Layout

```
GELS/
├── new_dem_0.py                 # PRIMARY simulation engine (V2.3, no plotting)
├── lsdem.py                     # LS-DEM deformable particle module (V2.3, Henzel 2026)
├── run_hpc_headless.py          # HPC headless runner (supports 2D/3D modes)
├── run_all_trials.py            # Batch trial runner (local / SLURM)
├── run_analysis_pipeline.py     # Full V1.7 analysis pipeline orchestrator
├── reconstruct_history.py       # Reconstruct history from snapshots
├── viz/                         # Visualization package
│   ├── __init__.py
│   ├── postprocess.py           # Unified post-processing (loads data, runs all viz)
│   ├── cells.py                 # Cell morphology & stress map visualization (V1.5.1)
│   ├── stress.py                # 3D surface stress, granule isosurfaces (V1.5.2)
│   ├── compaction.py            # Void-space & compaction visualization
│   ├── percolation.py           # Transport property analysis (Darcy, Kozeny-Carman)
│   ├── movies.py                # 3D volumetric animations (rotating, timelapse, sweep)
│   ├── phases.py                # Individual phase volume visualization
│   ├── shapes.py                # Granule shape gallery (superellipses, superellipsoids)
│   ├── doe.py                   # DOE statistical analysis & visualization
│   ├── dimensionless.py         # Dimensionless analysis, data collapse by β/Ca (V1.7)
│   ├── doe_2d_vs_3d.py         # 2D vs 3D DOE comparison & scaling analysis (V2.6)
│   ├── doe_dem_vs_network.py   # DEM vs contact network model comparison (V2.6)
│   ├── scaffold_evolution.py    # 2D microstructure evolution + timelapse (V1.11)
│   ├── scaffold_evolution_3d.py # 3D volumetric evolution (PyVista) per organ (V1.9)
│   ├── energy_landscape.py     # Energy landscape visualization (V1.11)
│   └── mesoscale.py            # Mesoscale model visualization (V2.5)
├── analysis/                    # Mathematical analysis package
│   ├── __init__.py
│   ├── mean_field_model.py      # Volume-conserving two-zone compaction model (V1.9)
│   ├── spatial_pde.py           # 1D radial PDE compaction model (V2.5)
│   ├── contact_network.py       # Graph-based contact network model (V2.5)
│   ├── contact_network_model.py # Gillespie SSA stochastic network model (V2.5)
│   ├── energy_landscape.py      # Free energy landscape decomposition (V1.11)
│   ├── coarse_grain.py          # Stress tensor, strain rate, viscosity from DEM (V1.7)
│   ├── tissue_descriptors.py    # Tissue architecture descriptor vector (V1.7)
│   ├── organ_targets.py         # Organ system target vectors (V1.7)
│   ├── arch_distance.py         # Architectural distance to organ targets (V1.7)
│   └── parameter_sweep.py      # Mean-field 1D PDE sweep & organ prediction (V1.10)
├── Trials/                      # Parameter sweep JSON configs
│   ├── generate_doe.py          # DOE config generator
│   └── default_trial.json       # Default V1.9 trial config
├── CodeLog/
│   ├── Architecture/            # Architecture documents
│   ├── Paper/                   # Publication manuscript (LaTeX)
│   ├── Readme/                  # README documents
│   ├── References/              # Literature references
│   └── Updates/                 # Changelog / update log
├── hpc/                         # HPC setup scripts and user configs
├── results/                     # Simulation output (gitignored)
└── CLAUDE.md                    # THIS FILE
```

## Key Technical Decisions

- **Units throughout**: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
- **Simulation modes** (V1.4+): `"2D"` (pure 2D), `"2D-slice"` (generate 3D packing, slice
  at z-midplane, run 2D), `"3D"` (full volumetric). Controlled by `Params.mode`.
- **3D domain** (V1.4+): `Lx × Ly × Lz` with 6-face wall boundaries. `Lz` defaults to 800 µm.
- **Superellipsoids** (V1.4+): `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1` with
  equatorial blockiness n1 and meridional blockiness n2. Spheres when a=b=c, n1=n2=2.
- **Quaternion orientation** (V1.4+): 3D rotational state stored as unit quaternions (w,x,y,z).
  Overdamped angular dynamics: `γ_rot dω/dt = Σ τ`. Integrated via `quat_integrate()`.
- **Contact model**: JKR adhesive contact (V2.3+, replaces Hertz+DMT from V1.1). Stiffness
  derived from `E_modulus` and `poisson_ratio`. JKR force:
  F = (4/3)E*a³/R* − √(8πWE*a³). Reduces exactly to Hertz when W=0. Contact radius
  solved via Newton-Raphson from overlap. Functions: `jkr_force_from_overlap()`,
  `jkr_contact_radius()`, `jkr_pulloff_force()`. Legacy `hertz_contact_force()` retained
  for reference. Per-particle contact clip planes (`gs.contact_clips`) store half-plane
  constraints at JKR contact faces for rendering with flat faces instead of overlapping
  circles.
- **Cell force model**: Motor-clutch (V1.2+, Chan & Odde 2008). Cell traction depends on
  substrate stiffness, replacing the old simple spring. See `motor_clutch_force()`.
- **Cell lifecycle** (V1.2+, updated V2.3): Cells are 20 µm spheres that attach instantly
  (default `t_attach_onset=0`, `t_attach_half=0`) → spread to 5 µm-tall ellipsoids (volume
  conserved) → overcrowded cells go senescent. **Multi-stage bridge formation** (V2.3):
  PROLIFERATING → MIGRATING → BRIDGING → SENESCENT. Cells sense targets via hemisphere
  check (only cells on the target-facing side can bridge), compute cell-specific gap decay
  from their centroid (not granule center), then either fast-path to BRIDGING (if within
  `bridge_commit_angle` of contact azimuth) or enter MIGRATING state for directed crawl.
  MIGRATING cells move along the host granule surface toward the contact point at
  `cell_migration_speed × bridge_directed_speed_mult` (default 2×), abandon if target
  leaves sensing range, and transition to BRIDGING upon arrival. **Bridge exclusion zone**:
  cells avoid bridging near existing BRIDGING/MIGRATING cells on the same granule→target
  pair (min angular separation `bridge_exclusion_angle`, default 0.8 rad ≈ 46°), forcing
  bridges to spread spatially. **Post-bridge alignment**: bridging cells progressively
  align stress fibers along bridge axis (`cell_alignment` grows from `bridge_alignment_min`
  toward 1.0 at `bridge_alignment_rate`), modulating force: `F = F_mc × maturity × alignment`.
  Persistent bridges → SENESCENT after `bridge_senescence_time`.
  See `update_cell_state()`, `_update_individual_cells()`, `_attempt_new_bridges()`.
- **Granule shape 2D** (V1.3+): Superellipses `|x/a|^n + |y/b|^n = 1` parameterised by
  semi-axes (a, b), blockiness exponent (n), and orientation (θ). Circles are the
  special case a=b, n=2. Enable with `shape_enabled=True`. See `superellipse_*()` functions.
- **Contact detection 2D** (V1.3+): Common normal method for superellipse–superellipse contact.
  Newton-Raphson iterative solver. Circle fast-path bypasses solver entirely.
- **Contact detection 3D** (V1.4+): 3D common normal method for superellipsoid–superellipsoid
  contact. 4-unknown Newton-Raphson (eta_i, omega_i, eta_j, omega_j). Sphere fast-path.
- **Transport metrics** (V1.4+): Kozeny-Carman permeability, Darcy flow, RCP fraction,
  compaction ratio, Darcy number. See `kozeny_carman_permeability()`, `rcp_fraction_superellipsoid()`.
- **Rotational dynamics**: Overdamped rotation from off-centre contact torques.
  2D: `γ_rot dθ/dt = Σ τ` (V1.3+). 3D: quaternion integration (V1.4+).
- **Shape descriptors** (V1.3+, Liu et al. 2025): Circularity, aspect ratio, elongation,
  blockiness tracked per granule and reported as population statistics.
- **Volume conservation**: Overlap lens area is tracked and redistributed via effective radii
  for rendering. Forces still use the original (undeformed) radii.
- **Integration**: Overdamped Euler (no inertia). Velocity cap prevents numerical blowup.
- **Neighbour search**: `scipy.spatial.cKDTree` with cutoff `2*max_r_bound + L_max` where
  `L_max = max(cell_sense_distance, bridge_break_gap)`.
- **Performance** (V1.4+): Optional Numba JIT for Newton-Raphson solver and superellipsoid
  geometry. Bounding-box clipping in 3D field rendering. Target: 500-1000 granules.
- **Packing** (V1.4.1+, updated V2.4): Lubachevsky-Stillinger inflate-and-relax in both
  2D and 3D. RSA places granules at deflated radii (α = (φ_safe/φ_target)^(1/d) where
  d=2 for 2D, d=3 for 3D), then settle inflates to target with Hertz repulsion + velocity
  cap over `packing_settle_steps` (400) × `packing_relax_substeps` (15), plus post-inflation
  overlap relaxation. Achieves jammed packing (Z ≈ 3–4 in 2D, Z ≈ 4–6 in 3D). No artificial
  gap (`packing_gap=0.0`). Seed is random by default (`seed=None`).
- **Packing composition** (V1.9+): Two ways to specify functional/inert fractions.
  Option A: `phi_f_target` + `phi_i_target` directly. Option B: `phi_solid_target` +
  `func_ratio` — total solid fraction (0.55–0.75 typical) split by functional ratio.
  When `phi_solid_target > 0`, derives `phi_f = phi_solid * func_ratio`,
  `phi_i = phi_solid * (1 - func_ratio)`. Ensures physically consistent packing.
- **Boundary modes** (V1.10+): `Params.boundary_mode` controls boundary conditions.
  `"walls"` (default): rigid Hertzian wall contacts, position clipping, boundary exclusion
  for metrics. `"periodic"`: minimum image convention for all pairwise interactions,
  `cKDTree(boxsize=)` for periodic neighbour search, position wrapping via modulo,
  ghost particle images for field rendering, no boundary exclusion. Unwrapped positions
  (`x_unwrap`, `y_unwrap`, `z_unwrap`) track true displacement across periodic boundaries.
- **Boundary exclusion** (V1.4.1+): `boundary_exclusion=0.2` excludes 20% from each
  domain edge when computing metrics (connectivity, porosity, permeability). Only the
  inner 60% of the domain volume is sampled. Forced to 0 for periodic boundaries.
- **Trial JSON format** (V1.4.1+): Two formats supported. **Flat**: keys are `Params` field
  names directly (e.g., `"E_modulus": 10.0`). **Legacy**: nested sections (domain,
  mechanics, shape, etc.) with translated key names. Flat format detected by `"_format": "flat"`.
- **Individual cell tracking** (V1.5+, updated V2.3): Each cell tracked individually via
  `CellState` enum (ATTACHED, SPREADING, PROLIFERATING, MIGRATING, BRIDGING, SENESCENT).
  Flat arrays in
  `GranuleSystem` indexed by `cell_offset[i]:cell_offset[i+1]`. Per-granule aggregates
  remain authoritative for force computation (physics identical to V1.4.1).
- **Data serialization** (V1.5+, updated V2.6): Per-timepoint `.npz` snapshots with granule
  + cell arrays. Scalar metrics as CSV/JSON. Params and metadata as JSON. Archived as
  `.tar.gz`. Controlled by `Params.save_data`, `save_fields` (default `True`), `output_dir`,
  `compress_archive`. **V2.6 changes**: `save_fields` default changed to `False` — field
  grids (phi_f/phi_i/phi_v) are no longer saved during simulation since they can be
  reconstructed from particle positions at plot time. This roughly halves storage per run.
  `check_disk_space()` guards against full filesystems: < 5 GB skips fields, < 1 GB skips
  save entirely. SLURM pre-flight aborts if < 2 GB free.
- **Resume from snapshot** (V2.6+): `Params.resume_from` (str, default empty). When set to
  a run directory path, `run()` loads the last snapshot via `find_last_snapshot()` +
  `restore_gs_from_snapshot()`, skips packing generation, and continues the main loop from
  the last saved timestep. Full state restored: positions, velocities, cell state, per-cell
  arrays, LS-DEM deformation (epsilon/d_epsilon). Displacement tracking resets to resume
  point. CLI: `--resume-from` in `run_hpc_headless.py`. Batch resume: `RUN_MODE=4` in
  `run_all_trials.py` scans for incomplete trials, uploads last snapshot + metadata to HPC,
  generates and submits resume SLURM scripts.
- **Visualization decoupled** (V1.6+): `new_dem_0.py` has no matplotlib dependency. All
  plotting is done via `viz/postprocess.py` which loads saved data and runs all viz scripts.
- **Data loading** (V1.5+): `load_run(run_dir)` → `(hist, snaps, p, metadata)`. Also
  `load_cells(run_dir, snap_index)` for targeted cell analysis.
- **Per-contact data** (V1.5.2+): Each contact stores point, normal, overlap, R_eff,
  F_normal, A_contact. Serialized to `.npz` as structured arrays. Used by `viz/stress.py`
  for Hertzian surface stress mapping.
- **Bridge formation kinetics** (V1.5.2+, updated V2.3): Bridges form probabilistically via
  Poisson process, ramp force over `bridge_formation_time`, persist across timesteps, and
  transition to SENESCENT after sustained load. **Bridge lock-in** (V1.9): bridges whose force
  exceeds `bridge_lock_force_threshold` bypass senescence and persist indefinitely.
  **Secondary migration** (V1.9): non-bridging cells can migrate along existing bridges
  (`bridge_secondary_rate_mult`). **Cell spatial awareness** (V2.3): hemisphere check
  ensures only target-facing cells attempt bridges; probability decays from cell centroid
  position, not granule center. **Path-dependent probability** (V2.3): gap classification
  (contact/void/inert-blocked) with `bridge_contact_factor`, `bridge_decay_length`,
  `bridge_inert_factor`. **Multi-stage formation** (V2.3): MIGRATING state for directed
  crawl before bridging; `bridge_commit_angle` (arrival threshold, default 0.5 rad),
  `bridge_directed_speed_mult` (crawl speed multiplier, default 2.0). Parameters:
  `bridge_attempt_rate`, `bridge_formation_time`, `bridge_senescence_time`,
  `min_fa_for_bridge`, `bridge_break_gap`, `bridge_lock_force_threshold`,
  `bridge_secondary_rate_mult`, `expected_bridge_force`, `bridge_contact_factor`,
  `bridge_decay_length`, `bridge_inert_factor`, `bridge_commit_angle`,
  `bridge_directed_speed_mult`, `bridge_exclusion_angle`, `bridge_alignment_rate`,
  `bridge_alignment_min`.
- **Tissue volume tracking** (V2.1+): Spatially-resolved `phi_tissue(xi, t)` on the N_x=20
  radial grid tracks cell + ECM volume fraction growing in the functional zone. Logistic
  growth ODE driven by cell count, FA maturity, and bridge formation. No feedback into
  compaction mechanics (tissue is soft). Corrected architecture descriptors (`BV_TV_eff`,
  `porosity_eff`, `K_f_tissue`) used for organ distance computation. Parameters in
  `TISSUE_PARAMS`: `k_tissue=0.05`, `alpha_tissue_fill=0.6`, `alpha_tissue_0=0.1`,
  `n_cells_tissue_ref=10.0`.
- **Cell surface coverage** (V2.1+): `Params.cell_surface_coverage` (default 0.0 = disabled)
  specifies cell loading as a fraction of granule surface area. At 1.0, cells form a full
  monolayer; >1.0 allows stacking. When set, overrides `n_cells_per_granule` and
  `cell_coverage`. Uses `cells_from_surface_coverage()` which computes
  `n = round(4πR² × coverage / A_cell_spread)` (3D) or `πR²` (2D). For R=40 µm:
  coverage 0.5→8 cells, 1.0→16, 1.5→24.
- **Multi-Contact DEM (MC-DEM)** (V2.2+): Stress-based multi-contact correction
  for Hertzian contact forces (Giannis et al. 2021). Per-particle volumetric overlap
  strain ε_V = Σ δ/(2R) drives confinement factor κ = 1 + ν/(1−2ν) × ε_V that scales
  Hertz repulsion. Captures stiffening when soft granules have many simultaneous contacts
  (jammed packings). For ν=0.49 (nearly incompressible hydrogel): c_mc=24.5, giving ~2×
  stiffening at typical packing. Capped at `mc_dem_kappa_max=5.0`. Applied as post-correction
  via `mc_dem_correction()` in both 2D and 3D force computations. Params: `mc_dem_enabled`
  (bool, default True), `mc_dem_kappa_max` (float, default 5.0).
- **LS-DEM deformable particles** (V2.3+): Variational level-set DEM following
  Henzel & Karapiperis 2026 (arXiv:2602.12895). Each particle gets scalar deformation
  DOFs ε_α(t) modulating fixed spatial mode shapes Φ_α(x). Particle geometry tracked
  via signed distance fields (SDF) on body-frame grids. Contact detection uses
  surface-node-to-SDF queries instead of Newton-Raphson, yielding multi-point contact
  patches. Overdamped deformation: γ_def dε/dt + K ε = F_ε, implicit Euler integration.
  Semi-Lagrangian SDF update: φ(x,t) = φ₀(x − Σ ε_α Φ_α(x)). Modes: axial compression
  (volume-preserving), prolate-oblate, volumetric (3D). MC-DEM auto-disabled when active.
  JKR-consistent adhesion at surface nodes: penetrating nodes get repulsion − adhesion,
  near-surface nodes (0 < SDF < 10% radius) get linearly fading attraction.
  All functions in `lsdem.py`. Params: `deformable_enabled` (bool, default False),
  `n_def_modes` (int, default 2), `sdf_resolution` (int, default 32),
  `n_surface_nodes` (int, default 128), `n_surface_nodes_3d` (int, default 512),
  `sdf_padding` (float, default 1.3), `def_drag_scale` (float, default 0.1),
  `def_eps_max` (float, default 0.3).
- **Deformed rendering** (V2.3+): `render_fields()` and `render_fields_3d()` stamp deformed
  particle shapes via semi-Lagrangian pull-back when `deformable_enabled=True`. Grid points
  are transformed to body frame, inverse mode-shape displacement is subtracted, then the
  analytical implicit function is evaluated at the pulled-back coordinate. Functions
  `_stamp_deformed_2d()` and `_stamp_deformed_3d()`. Zero overhead when deformable is off.
- **Deformation visualization** (V2.3+): `viz/postprocess.py` includes `plot_deformation()`
  (particle map colored by |ε|, inferno colourmap) and `plot_deformation_timeseries()`
  (mean/max strain + per-mode amplitudes with ±σ bands). Both gracefully skip for rigid runs.
  Skip key: `'deformation'`.
- **Mesoscale models** (V2.5+): Two complementary models between full DEM and mean-field ODE.
  **Spatial PDE** (`analysis/spatial_pde.py`): 1D radial grid ξ ∈ [0,1], N_x=20 points.
  State: x_f(ξ,t) functional zone fraction + φ_tissue(ξ,t). Physics:
  ∂x_f/∂t = −x_f·(σ_cell−σ_resist)/η_eff + D·∂²x_f/∂ξ². Cell stress modulated by
  neighbor_factor(ξ) = ½(1+cos(πξ)). Zero-flux BCs. Tissue volume logistic growth.
  Initializable from Params, DEM snapshot (radial binning), or DEM run (with fitting).
  Class: `SpatialPDE` with `solve()`, `fit_to_data()`, `descriptors()`.
  **Contact Network** (`analysis/contact_network.py`): Graph G=(V,E), spheres only (no
  superellipsoid solver). Bridge state machine: NONE→FORMING→ACTIVE→LOCKED/SENESCENT.
  Hertz+DMT contact, motor-clutch bridge force, overdamped dynamics. cKDTree neighbor
  search (periodic or wall BCs). Metrics: Z(t), Z_ff, bridge fraction, percolation
  (spanning cluster), cluster size distribution, force statistics. Monte Carlo ensembles
  via `run_ensemble()`. Class: `ContactNetwork` with `step()`, `solve()`, `metrics()`,
  `adjacency_matrix()`, `descriptors()`.
  **Stochastic Contact Network** (`analysis/contact_network_model.py`): Gillespie SSA
  (Bortz-Kalos-Lebowitz 1975) for exact stochastic bridge formation/senescence.
  Union-find percolation tracking (Newman-Ziff 2001) with O(N) spanning detection.
  JKR adhesive contact, motor-clutch cell traction, path-dependent bridge probability
  (contact/void/inert-blocked), bridge lock-in, secondary migration boost. Monte Carlo
  ensemble with full statistical summary (mean, IQR, min/max). Interface: `from_params(p)`
  → `solve(model)` or `ensemble(model, n_runs=1000)`. ~2.5s per 72h realization (2D).
  All three produce tissue descriptors for organ distance via `arch_distance.py`.
  Visualization: `viz/mesoscale.py` (kymographs, radial profiles, network snapshots,
  evolution panels, combined summary).

## Conventions

- Granule types: `0` = functional (cell-laden), `1` = inert (passive).
- Phase fields: `phi_f` = functional, `phi_i` = inert, `phi_v` = void = 1 - phi_f - phi_i.
- All parameters live in the `Params` dataclass; modify defaults there or pass overrides.
- Snapshots are dicts with keys: `phi_f`, `phi_i`, `phi_v`, `x`, `y`, `r`, `gtype`, plus
  cell state and shape arrays. In 3D mode, also includes `z`, `quat`, `c`, `n1`, `n2`.
- Serialized snapshots (V1.5) also include per-cell arrays (`cell_*`) and granule
  velocities/forces.

## Documentation Requirements

- **Update log**: Every code change must be recorded in `CodeLog/Updates/CHANGELOG.md`.
  Bug fixes go in a dedicated "Bug Fixes" subsection within the relevant version entry.
- **Architecture**: The system architecture document lives at `CodeLog/Architecture/ARCHITECTURE.md`.
  Update it when modules are added, removed, or significantly restructured.
- **README**: The user-facing README is at `CodeLog/Readme/README.md`.

## Running the Simulation

```bash
# Default 2D mode
python3 new_dem_0.py

# 3D mode
python3 -c "from new_dem_0 import Params, run; run(Params(mode='3D', Lx=400, Ly=400, Lz=400, t_total=24))"
```

Or import and call programmatically:

```python
from new_dem_0 import Params, run

# 2D (default, backward compatible)
p = Params(E_modulus=5.0, t_total=48.0)
hist, snaps, p, gs = run(p)

# 3D volumetric
p = Params(mode='3D', Lx=400, Ly=400, Lz=400, t_total=48.0)
hist, snaps, p, gs = run(p)

# 2D-slice (generate 3D packing, slice at z-midplane, run 2D)
p = Params(mode='2D-slice', Lx=800, Ly=800, Lz=800, t_total=72.0)
hist, snaps, p, gs = run(p)
```

### Post-Processing (Visualization)

All visualization is done as a separate step after the simulation completes.
The unified postprocessor loads saved data and runs all viz scripts:

```bash
# Run all visualizations on saved output:
python viz/postprocess.py -i results/default

# Skip specific modules (e.g., movies and stress):
python viz/postprocess.py -i results/default --skip movies stress

# Or run individual viz scripts:
python viz/compaction.py -i results/default
python viz/stress.py -i results/default
```

Or import programmatically:

```python
from viz.postprocess import run_all
run_all('results/default')                          # all visualizations
run_all('results/default', skip={'movies','stress'}) # selective
```

### Loading Saved Data (V1.5)

```python
from new_dem_0 import load_run, load_cells

# Load from output directory or .tar.gz archive
hist, snaps, p, metadata = load_run('results/default')

# Load per-cell data from a specific timepoint
cells = load_cells('results/default', snap_index=0)
# cells['cell_state'], cells['cell_fx'], cells['cell_bridge_target'], ...
```

## Dependencies

- Python 3.9+
- numpy, scipy, matplotlib
- Optional: pyvista (3D rendering), scikit-image (marching cubes), numba (JIT acceleration)
- Optional for post-processing: imageio, tqdm

---

## UArizona HPC Best Practices

Reference: [UA HPC Documentation](https://github.com/UA-ResearchComputing-HPC/hpc-documentation)

### Hostnames & Access

| Purpose | Hostname | Notes |
|---------|----------|-------|
| SSH login (bastion) | `hpc.arizona.edu` | Type `shell` after login to reach Puma |
| File transfers | `filexfer.hpc.arizona.edu` | 100 Gb link to shared storage |
| Web portal | `ood.hpc.arizona.edu` | Open OnDemand (browser-based) |
| VPN (off-campus) | `vpn.arizona.edu` | Required for R-DAS; use Cisco Secure Client |
| HPC VPN | `vpn.hpc.arizona.edu` | Required for X11 forwarding, port tunnels |

**SSH login flow:**
```bash
ssh netid@hpc.arizona.edu     # lands on bastion (gatekeeper)
# then type:
shell                          # connects to Puma login node (junonia or wentletrap)
```

**Critical:** The bastion host has **no shared storage** and a **10 MB quota**. Never use
`hpc.arizona.edu` for file transfers — always use `filexfer.hpc.arizona.edu`. Three failed
password attempts lock your account for 1 hour.

**For scripted automation (run_all_trials.py Mode 3):**
```bash
# Submit jobs via double-hop SSH:
ssh netid@hpc.arizona.edu "ssh shell.hpc.arizona.edu 'cd ~/GELS && sbatch script.slurm'"

# File transfers via filexfer:
rsync -ravz local_dir/ netid@filexfer.hpc.arizona.edu:~/GELS/
```

### Clusters

| Cluster | CPUs/node | RAM/node | GPUs | OS | Mem/CPU |
|---------|-----------|----------|------|----|---------|
| **Puma** (recommended) | 94 | 512 GB | V100S, A100 MIG | Rocky Linux 9 | 5 GB |
| Ocelote | 28 | 192 GB | P100 | CentOS 7 | 6 GB |
| El Gato | 16 | 64 GB | none | CentOS 7 | 4 GB |

**Use Puma for GELS.** Software compiled on CentOS 7 (Ocelote) may fail on Rocky Linux 9
(Puma) and vice versa. Always compile and test on the same cluster you'll run on.

### Allocations & Limits

**Monthly CPU-hours (Standard, free):** Puma 150,000 | Ocelote 100,000 | El Gato 7,000

**Per-group limits (Puma Standard):** 3,290 CPUs, 16,996 GB memory, 4 GPUs max simultaneous.

**Global limits:** Max 1,000 simultaneous jobs. Max walltime 10 days (240 hours).

**Windfall partition:** No hours consumed but lower priority and **preemptible** — jobs can be
killed at any time. Good for testing, not for production runs.

**Check remaining hours:** Run `va` on a login node.

### Storage

| Path | Quota | Lifetime | Backed Up? |
|------|-------|----------|------------|
| `/home/uXX/netid` | **50 GB** | Account lifetime | No |
| `/groups/pi_netid` | **500 GB** | PI account lifetime | No |
| `/xdisk/pi_netid` | 200 GB – 20 TB | **300 days then deleted** | No |
| `/tmp` (compute node) | ~1 TB | Job duration only | No |

**Nothing on HPC is backed up.** Maintain your own backups.

**Warnings:**
- Filling `/home` causes login failures and Open OnDemand errors.
- Avoid >100,000 files in a single directory (slows the parallel filesystem).
- Redirect pip/conda cache to `/groups` or `/xdisk` — they default to `~/.local` and eat `/home` quota.
- Check usage: `uquota` command. Find hidden space hogs: `du -hs $(ls -A ~)`.

**For GELS:** Store results in `/groups` (persistent) or `/xdisk` (temporary, larger).
Keep the repo clone in `/home/uXX/netid/GELS` or `/groups`. Transfer results to your
local machine via rsync before `/xdisk` expires.

### Python Environment Setup

```bash
# On a login node (after typing 'shell'):
module load python/3.11/3.11.4
python3 -m venv --system-site-packages /groups/<pi_group>/gels_env
source /groups/<pi_group>/gels_env/bin/activate
pip install numpy scipy matplotlib imageio scikit-image tqdm pyvista numba
```

**In batch scripts:**
```bash
module load python/3.11/3.11.4
source /groups/<pi_group>/gels_env/bin/activate
export MPLBACKEND=Agg  # headless matplotlib
```

**Rules:**
- Always specify module version: `python/3.11/3.11.4`, not just `python`.
- Always use `python3`, never `python` (system python is 2.7).
- Never `pip install` globally — always use a virtual environment.
- Modules are only available on compute/login nodes, not the bastion.
- `~/.local/lib/python3.X` causes cross-contamination between venvs — use `--system-site-packages`
  and install into the venv, or clean out `~/.local` if you see import conflicts.

### Batch Job Directives

**Minimum SLURM script:**
```bash
#!/bin/bash
#SBATCH --job-name=gels-run
#SBATCH --account=your_group       # omit for windfall
#SBATCH --partition=standard       # or windfall, gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # GELS is single-process
#SBATCH --cpus-per-task=4          # more CPUs = more memory (5 GB/CPU on Puma)
#SBATCH --time=04:00:00            # HHH:MM:SS, max 240:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
```

**CPU/Memory relationship on Puma:** Each CPU allocated comes with 5 GB. Requesting
`--cpus-per-task=4` gives 20 GB. Do NOT specify both `--ntasks` and `--mem` simultaneously —
specify one and let the scheduler calculate the other. Invalid ratios may redirect to
high-memory queues (longer wait).

**GELS is single-node, single-process.** Always use `--nodes=1 --ntasks=1`.
Increase `--cpus-per-task` if you need more memory (e.g., large 3D grids).

### Array Jobs (Parameter Sweeps)

```bash
#SBATCH --array=1-12               # basic range
#SBATCH --array=1-100%10           # limit to 10 concurrent subjobs
#SBATCH --array=1-3,7,10-15        # non-sequential
#SBATCH --output=slurm_logs/%x_%A_%a.out   # %A = parent job, %a = array index
```

**Environment variable:** `$SLURM_ARRAY_TASK_ID` is the unique integer for each sub-job.

**Pattern for reading trial names from a list file:**
```bash
TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/trial_list.txt)
```

**Monitoring array jobs:**
```bash
squeue -r --job=<jobid>     # -r flag expands array tasks into individual rows
squeue --me                 # all your jobs
```

Without `-r`, `squeue` shows the parent job as a single "R" line as long as ANY sub-task
is running — you cannot see per-task completion. **Always use `-r` for array jobs.**

**Never submit hundreds of jobs via a for-loop.** Use `--array` instead. Users submitting
>100 jobs via loops will be contacted by HPC staff.

### Job Monitoring & Diagnostics

| Command | Purpose |
|---------|---------|
| `squeue --me` | All your running/pending jobs |
| `squeue -r --job=JOBID` | Individual array sub-tasks |
| `scontrol show jobs JOBID` | Detailed info on running/pending job |
| `scancel JOBID` | Cancel a job (all array tasks) |
| `scancel JOBID_3` | Cancel specific array task |
| `seff JOBID` | CPU and memory efficiency (completed jobs) |
| `job-history JOBID` | Human-readable completed job info |
| `past-jobs -d 7` | Search your jobs from past 7 days |
| `job-limits GROUP` | Group resource limits and current usage |
| `va` | Check remaining allocation hours |
| `uquota` | Check storage quotas |

**After every run:** Use `seff JOBID` to check CPU and memory efficiency. If efficiency
is below ~80%, reduce resource requests for future jobs. This shortens queue wait times.

### File Transfers

| Size | Method |
|------|--------|
| < 64 MB | Open OnDemand web upload (`ood.hpc.arizona.edu`) |
| < 100 GB | rsync / scp via `filexfer.hpc.arizona.edu` |
| > 100 GB | Globus (endpoint: "UA HPC Filesystems") |

**Rsync (recommended):**
```bash
# Upload repo to cluster (trailing slash = contents only)
rsync -ravz --exclude __pycache__ --exclude .git --exclude results/ \
    ./  netid@filexfer.hpc.arizona.edu:~/GELS/

# Download results from cluster
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**SCP:**
```bash
scp -rp ./Trials/ netid@filexfer.hpc.arizona.edu:~/GELS/Trials/
scp -rp netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**Best practices:**
- Limit to 2–3 concurrent transfer sessions.
- Tar/compress many small files before transferring (parallel filesystem is slow with many small files).
- Set up SSH keys on `filexfer.hpc.arizona.edu` to avoid repeated password/2FA prompts.
- If SCP/SFTP fails with "Received message too long", check `~/.bashrc` for echo/printf
  statements and comment them out.

### GELS HPC Workflow

**Automated (run_all_trials.py Mode 3):**
```
1. Finds all .json files in Trials/
2. Generates SLURM array job script (results → /groups, staggered starts)
3. Rsyncs repo to cluster via filexfer
4. Submits via SSH: bastion → shell → sbatch
5. Polls with squeue -r (per-task progress)
6. Incremental rsync as tasks complete (delete from HPC after transfer)
7. Final rsync when all tasks done
```

**Resume incomplete trials (run_all_trials.py Mode 4, V2.6):**
```
1. Scans local results/LHC/ for incomplete trials (t_last < t_total)
2. Uploads last snapshot + params.json + history.json to HPC for each
3. Generates resume SLURM script with --resume-from
4. Rsyncs repo, submits, polls, and syncs as in Mode 3
```

**Manual workflow:**
```bash
# 1. Sync code to cluster
rsync -ravz --exclude __pycache__ --exclude .git --exclude results/ \
    ./  netid@filexfer.hpc.arizona.edu:~/GELS/

# 2. SSH in and submit
ssh netid@hpc.arizona.edu
shell
cd ~/GELS
sbatch hpc/run_all_trials.slurm

# 3. Monitor
squeue -r --me
seff JOBID   # after completion

# 4. Sync results back (from local machine)
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**Standalone result sync:**
```bash
python3 hpc/sync_results.py              # check status + sync
python3 hpc/sync_results.py --list-only  # just check what's there
```

### HPC Config File (hpc/Alex.json)

Each user's HPC settings are stored in a JSON file:
```json
{
    "netid": "mcgheealex",
    "group": "your_pi_group",
    "repo_path": "~/GELS",
    "python_module": "python/3.11/3.11.4",
    "venv_path": "/groups/your_pi_group/gels_env",
    "partition": "standard",
    "cpus": 4,
    "walltime": "04:00:00",
    "results_path": "/groups/your_pi_group/results",
    "max_concurrent": 10
}
```

**V2.6 fields**: `results_path` directs simulation output to `/groups` (500 GB) instead of
`/home` (50 GB). `max_concurrent` limits simultaneous SLURM array tasks (default 10).

### Common Gotchas

- **Never run computations on login nodes.** They will be killed without warning.
- **Puma vs Ocelote binaries are incompatible** (Rocky Linux 9 vs CentOS 7).
- **`MPLBACKEND=Agg` is required** in batch scripts — there is no display server.
- **`squeue` without `-r` hides array sub-task status** — always use `-r` for array jobs.
- **`/xdisk` data is deleted after 300 days** with no extensions. Back up before expiration.
- **Filling `/home` (50 GB) breaks login.** Keep pip cache and conda envs in `/groups`.
- **Module versions change.** Always specify exact version (e.g., `python/3.11/3.11.4`).
- **Interactive sessions for installs only.** Compile and install software from compute nodes
  via `interactive -a your_group`, not from login nodes.

### Required Acknowledgement (Publications)

> "This material is based upon High Performance Computing (HPC) resources supported by the
> University of Arizona TRIF, UITS, and Research, Innovation, and Impact (RII) and maintained
> by the UArizona Research Technologies department."
