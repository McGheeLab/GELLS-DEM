# GELS Project Guide

## Project Overview

**GELS** (Granule-Enabled Living Scaffolds)
is a 2D/3D overdamped particle dynamics simulator for modelling cell-driven rearrangement
of hydrogel granular scaffolds. The primary simulation engine is `gels/engine.py`.

**Current version: V3.2 (in progress — see `CodeLog/Updates/CHANGELOG.md` for landed phases)**

**Runs are local only.** HPC execution was retired in V2.7; simulations are driven
by the five numbered steps in `pipeline/`. See `pipeline/README.md`.

## Repository Layout

```
GELS/
├── gels/                        # SIMULATION ENGINE PACKAGE (V2.7+)
│   ├── __init__.py              # Re-exports Params, run, load_run, packing helpers
│   ├── engine.py                # Params, GranuleSystem, packing, run()/step(), serialization, dispatchers
│   ├── materials.py             # f-rules shared by loops + kernels (mixing, E*, Langmuir gain) (V3.0)
│   ├── bench.py                 # python -m gels.bench — timings vs N and threads (V3.0)
│   ├── lsdem.py                 # LS-DEM deformable particle module (Henzel 2026)
│   ├── division.py              # Cell division pass (doubling time, contact inhibition) (V3.1)
│   ├── presets.py               # Named bundles of setup overrides (V3.1)
│   ├── live/                    # Observer hook, frame builder, TkAgg viewer, tail/replay (V3.0)
│   ├── io/                      # SnapshotWriter (background, atomic, np.load-compatible) (V3.0)
│   └── kernels/                 # V3.0 compute layer (numba, parallel, thread-count independent)
│       ├── __init__.py          # numba fallbacks, configure_threads(), auto_threads()
│       ├── reference.py         # V2.7 Python loops, verbatim — oracle + LS-DEM path
│       ├── neighbors.py         # cKDTree pairs (reference order) / linked-cell pairs (sorted); CSR
│       ├── geometry2d.py, geometry3d.py  # status-tuple contact solvers
│       ├── contact2d.py, contact3d.py    # pair pass → MC-DEM → owner-writes gather (F, torque, clips, walls)
│       ├── contacts.py          # ContactSoA records, noise
│       ├── cells.py             # cell state machine + bridging (splitmix64 counter RNG)
│       ├── bridging.py          # interim Python bridging (perf_cells_backend='python', exact)
│       ├── integrate.py         # overlap resolution
│       ├── render.py            # bounding-box field stamps, effective radii
│       ├── metrics.py           # compiled metrics twin (bincount, strided Voronoi, shape cache)
│       └── packing.py           # RSA neighbour grid + compiled settle (bit-identical packing)
├── tests/                       # unittest suite (python -m unittest discover -s tests -v)
│   ├── fixtures/                # V2.7 bit-identical oracle runs (VERSIONED; see .gitignore)
│   ├── make_fixtures.py         # regenerates fixtures (only on purpose!)
│   └── test_*.py
├── pipeline/                    # STEP-BY-STEP LOCAL RUNNER (V2.7+)
│   ├── step0_new_setup.py       # Write / convert a sectioned YAML setup file (V3.0)
│   ├── step1_config.py          # Resolve Params (defaults→setup YAML / trial JSON→--set→CLI) → params.json + setup.yaml
│   ├── step2_pack.py            # Packing + cell seeding + t=0 eval → snap_0000.npz
│   ├── step3_simulate.py        # Advance to t_total; --continue resumes/extends; --live viewer; --threads
│   ├── step4_postprocess.py     # Drive viz2 (default) and/or viz suites
│   ├── step5_analysis.py        # Run analysis/ on one run → analysis/summary.json
│   ├── live_view.py             # Watch a running step 3, replay a run, or dump PNGs (V3.0)
│   ├── run_showcase.py          # Nine V3.0 showcase conditions in parallel + comparison (V3.0)
│   ├── _common.py               # Step state, guards, Params/trial-JSON loading
│   └── README.md                # Pipeline guide
├── viz/                         # Visualization package (V1.x suite)
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
├── viz2/                        # V2 visualization suite (pipeline default)
│   ├── postprocess.py           # viz2 orchestrator (run_all)
│   ├── common.py                # Shared loading + ensure_phase_fields() reconstruction
│   ├── scaffold_map.py          # Scaffold maps (2D)
│   ├── scaffold_map_3d.py       # Scaffold maps (3D, PyVista)
│   ├── voronoi.py               # Voronoi tessellation
│   ├── voronoi_shapes.py        # Voronoi + shape descriptors
│   ├── phase_fractions.py       # Global and local phase fractions
│   ├── movies.py                # GIF animations
│   ├── energy_stress.py         # Energy and stress maps
│   ├── void_percolation.py      # Void percolation, Kozeny-Carman
│   ├── parallel.py              # pmap: per-snapshot process pool for the suite (V3.0)
│   └── compare_runs.py          # Several runs on the same figures (V3.0)
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
│   ├── volume_conservation.py   # Volume/area conservation diagnostics
│   ├── experimental_comparison.py # Comparison against experimental data
│   ├── image_analysis.py        # Image-based scaffold analysis
│   └── parameter_sweep.py      # Mean-field 1D PDE sweep & organ prediction (V1.10)
├── Trials/                      # Parameter sweep JSON configs
│   ├── generate_doe.py          # DOE config generator
│   ├── default_trial.json       # Default V1.9 trial config
│   └── showcase_v3/             # Commented YAML setups of the V3.0 showcase (written by run_showcase.py)
├── CodeLog/
│   ├── Architecture/            # Architecture documents
│   ├── Paper/                   # Publication manuscript (LaTeX)
│   ├── Readme/                  # README documents
│   ├── References/              # Literature references
│   └── Updates/                 # Changelog / update log
├── old code/                    # RETIRED (V2.7) — archived, imports NOT updated
│   ├── hpc/                     # SLURM templates, Alex.json, sync_results.py
│   ├── run_all_trials.py        # Batch trial runner (local / SLURM)
│   ├── run_hpc_headless.py      # HPC headless runner
│   ├── run_analysis_pipeline.py # V1.7 analysis orchestrator (DOE_01–DOE_23)
│   ├── reconstruct_history.py   # Reconstruct history from snapshots
│   ├── test_hertz_clipping.py   # LS-DEM/JKR contact clipping demo
│   ├── HPC_NOTES.md             # Former UArizona HPC section of this file
│   └── README.md                # What was archived and why
├── robotsim/                    # SEPARATE REPO, gitignored — robot-ball framework (see below)
├── results/                     # Simulation output (gitignored)
└── CLAUDE.md                    # THIS FILE
```

**No `.py` files live at the repository root.** Anything there in an older
checkout has moved to `gels/`, `pipeline/`, or `old code/`.

## `robotsim/` — the robot-ball framework (separate repo, gitignored)

`robotsim/` is a **self-contained sibling project with its own git history**, moved here
from `~/robot-ball-framework` on 2026-09-18 so it lives next to the engine it borrows from.
It is listed in `.gitignore`: GELLS-DEM does not track it, and commits made here never touch
it. `cd robotsim` first, then use its own git.

**What it models.** Explicit **robots living on the shells of granular balls** — the
"physically real" tier of the granular-agent phase-space exploration, and the discrete
counterpart to the continuum cell agents in `gels/kernels/cells.py`. Robots are stored as
body-frame surface directions, so forces accumulate at the true surface anchor and generate
torque, and the abstraction survives the superellipsoid generalization (only
`surface_point(u)` changes).

- **State machine**: SEARCH → ANCHOR → BRIDGE, plus RIDE (robots crawl on robots when the
  first layer jams), SLEEP (activity decayed, contact-triggered reactivation), and FROZEN
  (battery dead, grip locked, bond becomes a passive latch). SLEEP/FROZEN bonds are
  non-backdrivable latches; only BRIDGE is an active force-controlled actuator.
- **Grip law**: `f_max = sigma_g * A_r * zeta1*zeta2/(zeta1+zeta2)`. With `zeta_I = 0`,
  robots can neither crawl on nor grip inert balls, so F-I bridges die on their own and
  **the inert balls ARE the barriers** — nothing is hard-coded.
- **Reach**: bridges span surface gaps up to `h_r` through a capture annulus, so
  near-contact pairs are bridgeable and anchors sit at the rim, not the contact point.
  This is the discrete form of the Aim-2 bridging question (do agents reach across voids,
  or only reinforce existing contacts?).
- **Emergent agitation**: crawling robots exert propulsion reaction forces on their hosts,
  the physical origin of `E_agit ~ a^2`. **No thermostat.**
- **Zero-pressure barostat**: the box carries no wall load, so bonded-network tension
  contracts the volume until Hertzian back-pressure balances it. Volume is an observable
  (`V/V0` and `Phi` per frame); soft contacts (`k_n = 0.05`) keep post-jamming compaction
  visible.
- **Initial condition**: verified jamming by the O'Hern protocol, shared with `gel3d`.
- **Battery is work-based, not a timer**: `e_clamp` per clamp-on, `p_bridge` per unit time
  holding a bridge, small `p_search` while crawling. Bridging robots die first.
- **Sizes**: ball diameters 30–250 (log-uniform, both species); robots are always
  two-lobe Ø10 crawlers. Code unit = 50 phys.

**Layout.**

```
robotsim/
├── code/                        # All simulators — start at code/README.md
│   ├── robotsim.py              # L3 shell-dwelling robots on balls (the main model)
│   ├── robotsim_sq.py           # Superellipsoid generalization
│   ├── superquadric.py          # Support-function MTD contact solver
│   ├── gel2d.py, gel3d.py       # Prototypes: gelation (Lu 2008 mapping); quasi-2D slab + O'Hern jamming
│   ├── run_experiment.py        # Named presets: baseline, deep_battery, short_battery,
│   │                            #   low_coverage, crowded, boulders, pore_hiders, grip_limited
│   ├── run_factorial.py         # Full-factorial coarsening study (144 cells)
│   ├── run_phase_diagram.py, run_reach_sweep.py, run_quench.py, run_quench3d.py, run_sq.py
│   ├── run_cell.py              # Headless per-cell runner (HPC entry point)
│   ├── build_landscape.py, analyze_landscape.py, compare_shapes.py, convergence.py
│   ├── collect_summary.py, enrich_summary.py, add_voronoi.py, shadow_check.py
│   └── export_viewer.py, serve.py, *_template.html   # Self-contained HTML gallery viewer
├── sections/                    # 00-07: the theory the code implements (notation,
│                                #   mean-field compaction, energy landscape, phase
│                                #   separation, simulation framework, experiments,
│                                #   literature, gelation concept)
├── docs/gells-dem-adoption.md   # WHAT THIS REPO WANTS FROM GELLS-DEM — feature-adoption
│                                #   report with file:line refs into new_dem_0.py, lsdem.py,
│                                #   run_all_trials.py, hpc/, analysis/. Read before porting
│                                #   anything in either direction.
├── hpc/                         # hpckit (adopted from VSClaude/hpc; validated on UA Puma)
├── proposal_scaffold/, sections/, Proposal_v5.md, PROPOSAL_TEMPLATE.md, tools/
│                                # Proposal drafting built on top of the simulation results
├── trials/, trials_regen/, trials_test/   # Factorial trial configs
└── results, figures, old, "granular flow channels"   # SYMLINKS — see below
```

**The data did not move.** `results/` (49 GB) and `figures/` (8.4 GB), plus `old/` and
`granular flow channels/` (275 MB), were left in `~/robot-ball-framework/` and are
**symlinked** back into `robotsim/`. Scripts that build paths relative to the repo root
therefore still resolve, but nothing large lives under GELLS-DEM. Both the symlinks and
their targets are gitignored on each side. Do not `rm -rf ~/robot-ball-framework` — that
is where the trial output actually is.

**Relationship to GELLS-DEM.** The two are independent codebases attacking the same
physics from opposite ends: GELLS-DEM treats cells as a contact-level kernel inside a
mature 2D/3D engine; robotsim treats them as explicit surface agents with batteries and
a state machine. `robotsim/docs/gells-dem-adoption.md` is the standing account of which
GELLS-DEM machinery robotsim should borrow (HPC, per-frame restart-complete npz,
params/metadata provenance, checkpoint-restart, walltime estimator) and which robotsim
pieces are already better and must not be regressed (support-function MTD superquadric
solver, O'Hern FIRE jamming, Laguerre-Voronoi, Katz-Thompson permeability, Newman-Ziff
percolation).

## Key Technical Decisions

- **Units throughout**: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
- **Simulation modes** (V1.4+): `"2D"` (pure 2D), `"2D-slice"` (generate 3D packing, slice
  at z-midplane, run 2D), `"3D"` (full volumetric). Controlled by `Params.mode`.
- **3D domain** (V1.4+): `Lx × Ly × Lz` with 6-face wall boundaries. `Lz` defaults to 800 µm.
- **Container shape** (V3.1): `boundary_shape` (`box` | `cylinder`, 3D only: axis z, radius
  min(Lx,Ly)/2 about the centre) and `boundary_top` (`wall` | `free`). With a free top, z is
  up in 3D and **y is up in 2D**, so a 2D run is a "dish slice" with a floor, two side walls
  and a free surface; `Lz` (or `Ly`) is then the CONTAINER height, about 1.4 × the bed, and
  granules that reach it are clamped and counted in `n_top_clamped` (the neighbour and render
  grids need positions inside the box). Never hand-roll `Lx*Ly*Lz`: use
  `domain_volume(p)` / `domain_base_area(p)`, which are shape-aware and reduce to the V3.0
  expressions for a box.
- **Gravity** (V3.1): `gravity_enabled` adds a buoyant weight `(ρ_gran − ρ_medium) g V` in nN
  along −z (3D) / −y (2D); per-species `density_kg_m3`. PMMA in water is 0.059 nN at
  r = 20 µm against 50–180 nN of cell traction, so in the dynamics it is a small bias — the
  bed is sedimented by the PACKER, not by the simulation.
- **Boundary chemistry** (V3.1): `boundary_functionalization` (f_wall) mixes the smooth
  wall's adhesion with `mix_pair(f_i, f_wall, …)`, and `boundary_layer_enabled` lines the
  floor and wall with immobile granules (`gs.fixed`) that cells bridge to through the normal
  machinery. This is what separates the two experimental regimes: an inert boundary lets the
  bed detach and compact, a functionalized one pins it and the functional phase coarsens.
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
  Note it saturates monotonically in stiffness (no biphasic optimum): β = k_sub/(k_sub+k_opt)
  is 0.51 at 10 kPa and ≥ 0.99 above 1 MPa, so on a rigid granule the force is 0.909 × stall
  and the calibration knob is the stall force, not the modulus.
- **Contractile bridge** (V3.1): `bridge_force_model` = `constant` (the V2.7 actuator: force
  independent of gap, strain, velocity and load) or `hill` (an active element with a stall
  force and an unloaded shortening speed `cell_contraction_speed`). The Hill factor is solved
  per PAIR, not evaluated explicitly, because the bridges' own pull is most of the closing
  speed; an explicit `1 − v/v₀` limit-cycles. A rigid environment gives factor 1 — the cell
  holds its stall force at force balance. v₀ → ∞ reduces exactly to `constant`.
- **Cell division** (V3.1): `cell_division_enabled` with `cell_doubling_time` (24 h),
  a refractory `cell_division_min_age` and contact inhibition against `cell_capacity` ×
  `cell_division_max_layers`. `cell_capacity_coverage` separates capacity from seeding
  coverage — without it a granule seeded at monolayer density is already confluent.
  `GranuleSystem.add_cells` grows the CSR cell arrays, keeping every existing cell's slot
  within its granule's range; all per-cell arrays are declared once in `CELL_ARRAYS`.
  **Reset a parent BEFORE `add_cells`**: the CSR rebuild shifts absolute cell indices, so an
  index array taken before it is stale afterwards (V3.1 zeroed the wrong cells this way).
  **`cells.division.model: cycle`** (V3.2) gives each cell its own lognormal cycle length and
  realises the stated doubling time; `poisson` is memoryless, so a refractory period adds to
  the mean cycle and 24 h behaves like 42.6 h.
  **The cycle clock is advanced by `gels.division.age_cells`**, called from both
  `update_cell_state` twins just before `divide_cells` (V3.2). Until then nothing incremented
  `gs.cell_age`, so with the default `min_age_h = 8` no cell ever divided; the kernel parameter
  that looks like it does the ageing is bound to `cell_bridge_age`, a different array.
- **Rigid granules (PMMA)** (V3.1): declare the true modulus on the species and cap only the
  CONTACT stiffness with `contact_E_cap`, plus a Coulomb `friction_mu` (additive to the
  hydrogel shear-stress law, because for a rigid contact τ₀·A is ~1e-3 nN). Be explicit that
  with an explicit overdamped integrator rigidity is enforced GEOMETRICALLY by
  `_resolve_overlaps` and the velocity cap, not by resolved Hertz forces: reported contact
  forces are overshoots, not equilibrium values.
- **Presets** (V3.1): `gels/presets.py` holds bundles of dotted overrides (`pmma_well`,
  `pmma_dish_slice`, `functionalized_wall`, `inert_wall`, `fibroblast_realistic`,
  `hydrogel_box_legacy`), applied by `step0 --preset` / `step1 --preset` before `--set` and
  recorded in `meta.presets`.
- **Which metrics to trust** (V3.1): the positional bed observables (`bed_height_mean`,
  `bed_height_p95`, `phi_bed`, `bed_radius_p95`, `wall_contact_fraction`), `n_granules_inner`
  and the analytic `*_true` fractions. The field-based `phi_solid` / `porosity` /
  `compaction_ratio` are inflated by the tanh interface halo (≈ 1.5× at R = 20 µm) and by
  overlap double counting.
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
- **Granule shape in the packer** (V3.2): `packing.shape_contact` + `contact.shape_dynamics`
  + `contact.curvature_R_cap` (preset `fragmented_granules`). Before this the packer was
  bounding-sphere throughout, so turning `shape_enabled` on made the bed **41 % taller**
  (phi 0.52 vs 0.75). Three things had to change together and therefore sit behind one flag:
  `r_bound` must be the true circumscribed radius (`shape_bound_radius`; `max(a,b,c)` is 21 %
  short at n = 3.5 and does not bound the body), the settle must use the closed-form
  DIRECTIONAL radius `lambda(u) = F(u)^(-1/n2)` with force AND torque from its gradient, and
  RSA must deflate by `shape_bound_factor` or it silently drops granules (20 % in 3D).
  **Two traps**: `lambda` is degree -1, so `grad(lambda).u = -lambda` and the radial part must
  be projected out (keeping it gives a force 2.5x too large that is not a gradient); and a
  purely radial force yields zero torque, so without the tangential tilt nothing rotates and
  the density gain vanishes. `curvature_R_cap` is not optional for blocky shapes: a flat face
  has a curvature radius of ~1e15 um and `F ~ sqrt(R_eff)`.
- **Soft granules** (V3.2): `contact.overlap_model: elastic` sizes `max_overlap_frac` from the
  contact law rather than by hand -- the volumetric ceiling it imposes is `(2/(2-f))^3`, so
  V3.1's 0.03 left only 4.6 % of headroom. `contact.semi_implicit` damps each step by the
  local contact stiffness (`vel = F/(gamma + dt*sum 2 E_s a)`): unconditionally stable in the
  diagonal part, same fixed point, and it takes stability duty off the velocity cap (fraction
  of granules pinned at the cap fell from ~1.0 to ~0.00). **Trust `phi_bed_net`**, not
  `phi_bed`: the latter double-counts overlap, which is 1-4 % once a soft bed compacts through
  interpenetration. `n_overlap_clipped` / `overlap_clip_fraction` / `frac_velocity_clipped`
  say whether a run is reporting the contact law or the numerical rails.
- **LS-DEM is not the tool for deformable granules** (assessed V3.2): its modes are global
  affine strains about a body-fixed axis, so a granule cannot flatten at a contact; wall
  contacts accumulate no deformation; it forces the whole run onto the Python reference path;
  and it has no tests. See the V3.2 changelog entry before reaching for it.
- **Packing consolidation** (V3.1): `packing_consolidation` = `centre` (the V2.7 centripetal
  pull toward the box centre — kept as the Params default for bit-identical legacy runs, but
  it consolidates a sub-RCP packing into a ball and leaves the corners empty; see the V3.1
  Bug Fixes entry), `gravity` (a downward body force during the post-relax that sediments the
  granules into a bed with a free surface), `none`, or `auto` (the template default: gravity
  when gravity is enabled, else none). `granules.bed_height_um` states the granule amount as
  a settled bed height instead of a container solid fraction.
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
- **Visualization decoupled** (V1.6+): `gels/engine.py` has no matplotlib dependency. All
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

- **Engine package** (V2.7+): The engine lives in `gels/` as `gels.engine` and
  `gels.lsdem`; `gels/__init__.py` re-exports the common API so
  `from gels import Params, run` works. Downstream modules use
  `from gels.engine import X`. Every `viz/`, `viz2/` and `analysis/` module
  already inserts the repo root on `sys.path`, so no install step is required.
- **Step-by-step execution** (V2.7+): Runs are driven by five numbered scripts
  in `pipeline/`, each operating on one run directory and checkpointing into
  `<run_dir>/pipeline_state.json`. Steps 2 and 3 are joined by the engine's
  V2.6 resume mechanism (step 2 writes `snap_0000.npz`; step 3 restores it and
  continues), which restarts the RNG stream — so stepping is deterministic for
  a given seed but not bit-for-bit equal to a monolithic `run()`.
- **Local only** (V2.7+): HPC execution is retired. No SLURM, no cluster sync.
  The archived machinery is in `old code/` with imports left unmodified.
- **Console encoding**: the engine prints non-ASCII (`ν`, `µ`, `→`). The
  pipeline reconfigures stdout/stderr to UTF-8 with `errors='replace'`, and
  since V3.0 `gels/engine.py` itself switches both streams to `errors='replace'`
  at import (encoding untouched), because a default Windows cp1252 console — or
  a pipe, whose default handler is `surrogateescape` — otherwise kills a run
  inside `print()`.
- **Live view** (V3.0): `run(p, seed, observer=None)` exposes an `Observer`
  hook (`gels/live/observer.py`). `step3 --live` spawns a TkAgg viewer in a
  separate `spawn` process fed by a bounded queue (`put_nowait`, frames dropped
  when behind) — the simulation never waits for the window. The parent stays on
  Agg; the child sets `MPLBACKEND=TkAgg` before importing matplotlib and must
  not import `viz2.common` (it pins Agg) — use `viz2.palette` /
  `viz2.snapshot_ops`. No pipeline script may import matplotlib at module
  level. Snapshots and `history.json` are written atomically so
  `pipeline/live_view.py` can tail a running step 3.
- **Numba constraints**: `np.linspace(..., endpoint=False)` is unsupported in
  nopython mode. Use `np.arange(n) * (2*np.pi/n)` inside `@njit` functions.
  A `@njit` function that can return `None` cannot be called from compiled
  code — `gels/kernels/geometry2d.py` / `geometry3d.py` hold status-tuple copies
  of the contact solvers for that reason. Keep the copies in sync with the
  originals in `gels/engine.py`.
- **Compiled kernels** (V3.0, `gels/kernels/`): `Params.use_numba` (default True)
  routes contacts, cells/bridging, overlap resolution, rendering, metrics and
  packing to numba kernels through the dispatchers in `gels/engine.py`
  (`kernels_enabled`, `cells_kernels_enabled`, `_render_kernels_ok`). LS-DEM
  always runs on the reference path. Kernels use owner-writes gathers and
  sorted pair lists, so results never depend on the thread count. Exactness:
  packing bit-identical to the reference; forces / fields / metrics to rounding;
  bridging statistically equivalent (counter-hash RNG, one `Generator` draw per
  pass). `perf_cells_backend='python'` = exact V2.7 cell machinery on compiled
  contacts; `use_numba=False` = pure reference (the fixture gate). Every kernel
  has a `tests/test_*_vs_reference.py`; the oracle is `gels/kernels/reference.py`,
  which must stay verbatim.
- **Visualization is parallel over snapshots** (V3.0): the `viz2` suite is plain
  Python/matplotlib — the compiled kernels do not touch it — and used to pin one
  core per run. Every per-snapshot loop (GIF frames, Voronoi tessellations,
  energy fields, void clusters) now goes through `viz2.parallel.pmap`, a spawn
  process pool. Worker count: `--workers N` on `step4_postprocess.py` /
  `viz2/postprocess.py`, else `$VIZ2_WORKERS`, else physical cores capped at 8;
  `--workers 1` is the serial path and is also the automatic fallback if the pool
  fails, with identical output. Workers are pinned to one numeric thread each, so
  several runs plotting side by side do not oversubscribe. Functions handed to
  `pmap` must be module level and take one picklable argument.
- **Coarse-graining is vectorised** (V3.0): `analysis/coarse_grain.py::coarse_grain_field`
  evaluates its Gaussian sums over chunks of grid cells with numpy, not a Python
  loop over cells x contacts. It is called per movie frame, per stress/strain
  panel and by step 5, so the old form (Ngrid^3 x n_contacts) dominated
  post-processing and analysis. Same weights, same 1e-10 cutoff, same
  normalisation; only summation order differs, so values agree to ~1e-15.
- **Threads / performance knobs**: `perf_threads` (0 = `auto_threads(N)` =
  clamp(N/1000, 4, physical cores) — 40 threads are slower than 8 below
  N ≈ 10⁴), `perf_threading_layer` (`omp`), `perf_neighbor_backend`
  (`cells` | `ckdtree` | `reference`), `perf_cells_backend` (`kernels` |
  `python`), `perf_metrics_voronoi_stride` (4), `perf_async_io` /
  `perf_io_queue_depth` / `perf_io_compresslevel` (background snapshot writer),
  `perf_keep_snaps_in_memory` (False: `run()` returns `snaps=[]`, read them
  with `load_run`). YAML: the `performance:` section. Measure with
  `python -m gels.bench --mode 2D --N 1000,10000 --threads 8,40 --path kernels|reference`.

## Conventions

- Granule species (V3.0): every granule has `species_id` and a functionalization `f ∈ [0,1]`
  (collagen-I coverage; 1 = fully coated, 0 = bare). `gtype` is **derived**
  (`0` where `f ≥ f_min_adhesion`, else `1`) and kept only for downstream consumers —
  never write it. Legacy two-species runs: species 0 = collagen (f=1), 1 = bare (f=0).
  Adhesion/friction pair parameters are `W_adh_cc/cb/bb`, `tau_0_cc/cb/bb`
  (collagen–collagen / collagen–bare / bare–bare); `W_adh_ff/if/ii` etc. are aliases.
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

Runs proceed as five numbered steps, each a standalone script operating on one
run directory. Each step records completion in `<run_dir>/pipeline_state.json`,
verifies its prerequisite ran, and refuses to clobber completed work without
`--force`. Every step prints the next command when it finishes.

```bash
python pipeline/step1_config.py      --name my_run --t_total 72
python pipeline/step2_pack.py        -i results/my_run
python pipeline/step3_simulate.py    -i results/my_run            # add --live to watch it
python pipeline/step4_postprocess.py -i results/my_run
python pipeline/step5_analysis.py    -i results/my_run
```

Configuration precedence in step 1 is: `Params` defaults → trial JSON
(`--trial Trials/X.json`) → individual CLI flags (one per `Params` field).

```bash
python pipeline/step1_config.py --trial Trials/DOE2_2D_0001.json
python pipeline/step1_config.py --name stiff3d --mode 3D --E_modulus 20 --t_total 48
```

Useful flags:

| Flag | Step | Effect |
|------|------|--------|
| `--force` | any | Re-run a completed step (2 and 3 also clear downstream state) |
| `--continue` | 3 | Resume an interrupted run, or extend a finished one with `--t_total` |
| `--archive` | 3 | Also write a `.tar.gz` of the run directory |
| `--suite viz\|viz2\|both` | 4 | Which visualization suite to run (default `viz2`) |
| `--workers N` | 4 | Processes for the per-snapshot plotting work (default: auto, 1 = serial) |
| `--skip` / `--only` | 4 | Select visualization modules |
| `--modules` | 5 | Subset of analysis modules |

See `pipeline/README.md` for the full guide and run-directory layout.

Presets bundle the correlated settings of a configuration:

```bash
python pipeline/step0_new_setup.py --list-presets
python pipeline/step0_new_setup.py --preset pmma_well,fibroblast_realistic -o well.yaml
python pipeline/step1_config.py --setup well.yaml --preset inert_wall --name well_run
```

Several conditions at once: `python pipeline/run_showcase.py` runs the nine V3.0
showcase conditions concurrently (each its own steps 1–5 process, thread budget =
physical cores) into `results/showcase_v3/` and then calls
`viz2/compare_runs.py`, which draws any set of finished runs on the same
figures (`compare_scaffolds.png`, `compare_timeseries.png`, `compare_species.png`,
`compare_timelapse.gif`, `compare_summary.md`). `--sweep well` (V3.1) runs the well
calibration set — container, force model, division and wall chemistry varied one at a time —
and `viz2/compare_runs.py --experiment measured.csv` overlays measured bed geometry
(`t_h, bed_height_um[, bed_height_sd_um][, bed_diameter_um]`) on the bed panels;
`compare_bed_profiles.png` is a vertical section through the container axis.

### Calling the engine directly

The pipeline is the supported path, but the engine can still be driven
programmatically:

```python
from gels import Params, run

# 2D (default)
p = Params(E_modulus=5.0, t_total=48.0)
hist, snaps, p, gs = run(p)

# 3D volumetric
p = Params(mode='3D', Lx=400, Ly=400, Lz=400, t_total=48.0)
hist, snaps, p, gs = run(p)

# 2D-slice (generate 3D packing, slice at z-midplane, run 2D)
p = Params(mode='2D-slice', Lx=800, Ly=800, Lz=800, t_total=72.0)
hist, snaps, p, gs = run(p)
```

Note that `run()` does packing and dynamics in one call, so it is **not**
equivalent to steps 2 + 3: stepping resumes between them, which restarts the
RNG stream. Both are deterministic for a given seed.

### Post-Processing (Visualization)

Step 4 drives this, but either suite can be invoked directly. `viz2` is the
newer suite and the pipeline default; `viz` is the V1.x suite.

```bash
python viz2/postprocess.py -i results/my_run
python viz2/postprocess.py -i results/my_run --only scaffold phases
python viz/postprocess.py  -i results/my_run --rerender --skip movies stress
```

Since V2.6, `save_fields` defaults to `False`, so phase-field grids are not on
disk. `viz2` rebuilds them via `viz2.common.ensure_phase_fields()`; `viz`
requires `--rerender` (the pipeline passes it automatically).

```python
from viz2 import run_all
run_all(run_dir='results/my_run')

from viz.postprocess import run_all as viz1_run_all
viz1_run_all('results/my_run', rerender=True, skip={'movies', 'stress'})
```

### Loading Saved Data (V1.5)

```python
from gels import load_run, load_cells

# Load from output directory or .tar.gz archive
hist, snaps, p, metadata = load_run('results/my_run')

# Load per-cell data from a specific timepoint
cells = load_cells('results/my_run', snap_index=0)
# cells['cell_state'], cells['cell_fx'], cells['cell_bridge_target'], ...
```

## Dependencies

- Python 3.9+
- Required: numpy, scipy, matplotlib
- Optional: numba (JIT acceleration), scikit-image (marching cubes),
  pyvista + vtk (3D rendering)
- Optional for post-processing: imageio, imageio-ffmpeg (MP4 writer), tqdm
- Optional for document builds: python-docx, openpyxl, pypandoc

Install everything:

```bash
pip install numpy scipy matplotlib numba scikit-image pyvista             imageio imageio-ffmpeg tqdm python-docx openpyxl pypandoc
```

**Known issue (Windows):** `import pyvista` can fail with
`DLL load failed while importing vtkCommonExecutionModel: An Application
Control policy has blocked this file`. This is Windows Smart App Control /
WDAC refusing to load the VTK DLLs, not a broken install. Either allow the
VTK DLLs in site-packages, or accept that the PyVista-backed 3D modules
(`viz/scaffold_evolution_3d.py`, `viz2/scaffold_map_3d.py`, parts of
`viz/movies.py` and `viz/stress.py`) will be skipped. Everything else runs.
