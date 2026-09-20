# GELS

**Granule-Enabled Living Scaffolds**

A 2D/3D particle dynamics simulator for modelling cell-driven rearrangement of
hydrogel granular scaffolds.

---

## What It Does

GELS simulates how cell-laden hydrogel granules reorganize over time within
a confined domain. Functional granules carry cells that form mechanical bridges
to neighbouring functional granules, pulling them together. Inert granules act as
passive spacers. The simulation predicts how scaffold microstructure --- void
networks, functional connectivity, tissue density, transport properties --- evolves
over 24--72 hours.

### Key Physics

- **JKR adhesive contact** (V2.3; Hertz+DMT before) with a physically meaningful Young's
  modulus (1--100 kPa) and a separate cap on the *contact* stiffness for rigid beads
- **Motor-clutch cell bridging** (Chan & Odde 2008) — substrate stiffness-dependent traction
  with probabilistic initiation, a maturity ramp, lock-in and senescence; optionally a Hill
  force-velocity element rather than a constant actuator
- **Traction as a stress over the adhesion area** (V3.6), scaled by ligand coverage — with
  the **cell strain energy** that traction force microscopy measures
- **Superellipsoid / superellipse granules**, with contact detection by the closed-form
  **support function** (V3.4): a true penetration depth, an exact `R_eff`, and forces that
  are the gradient of an energy
- **Quaternion-based 3D rotational dynamics**
- **Gravity, containers and free surfaces** (V3.1): box or cylinder, walled or open top
- **Transport metrics**: Kozeny-Carman *and* Katz-Thompson permeability, geodesic
  tortuosity, Laguerre (halo-free) local packing fraction, contact-graph percolation
- **Overdamped dynamics as gradient flow** (V3.5), with an audit that checks it
- **Overdamped Langevin dynamics** (no inertia, appropriate for viscous culture medium)

---

## What's New in V3.7

The engine now reports a **stress**, not just a force.

Everything mechanical it reported before — `F_mean`, `F_max` — was extensive, so it
was comparable to nothing: not to a measurement, and not even to another run at a
different N. `gels/stress.py` adds the Love-Weber virial stress in kPa, split into
the part the granular skeleton carries and the part the cells are generating:

```
P_contact   +0.271 kPa      the skeleton
P_active    -0.412 kPa      the cells  (negative = TENSION)
stress_active_frac  0.60    the cell-derived share
```

The sign is the physics. A contracting bridge puts the bed in tension and *relieves*
the compression the contacts carry — while simultaneously raising `P_contact`
(0.015 → 0.271 kPa over 4 h) by pulling granules into closer contact.

Alongside it, the **stress-force-fabric** decomposition (Rothenburg & Bathurst 1989):
`fabric_a_c` and `fabric_a_n`, two scalars that say how much of the shear comes from
the anisotropy of the contact network rather than from the forces on it — the form a
continuum model can consume. `sff_closure` reports how much of `q/p` those two
explain (0.82–1.04 measured), because the tangential force is not stored and pretending
otherwise would be the easy mistake.

And `stress_patch_p95` says when to stop believing any of it: Hertz and JKR are
small-strain theories, and on a gravity-loaded 3D bed the contact patch reaches 22 % of
the granule radius. That is a flag on the *contact law*, not on the stress formula —
Love-Weber itself is exact for this model.

**The visualization suite now uses the engine's physics rather than its own.** Three
modules were each rebuilding the contact law from the global `E_modulus`, because the
snapshot did not carry the per-pair values. It does now, and the correction is not
subtle: the contact-energy field was **790× too large**, and on the PMMA preset the
assumed stiffness was off by **3e4×**. The six-panel energy figure turned out to be
plotting four different unit systems on one axis; each panel now states its own.

See `CodeLog/Updates/CHANGELOG.md` for the full entry.

## What's New in V3.6

Three things the engine modelled but the solver never *felt*, plus the machinery
to say when a run is done.

- **`dt` is a coupling interval, not an integration step.** `contact.semi_implicit`
  gets the equilibrium exactly right for any `dt`, but the *rate* wrong by
  `1 + dt·k/gamma` — which at the shipped `dt = 0.5 h` averages **41** on a
  cell-seeded bed, so a 24 h compaction came out ~4x low. `dynamics.substep: auto`
  makes `dt` the interval the run is *sampled* at and substeps the mechanics
  underneath it. It declines a passive run, because a bed that ends at a fixed
  point does not care (measured: 0.05 % on a gravity bed).
- **Cell types are objects** — `gels/celltypes/`, every value carrying its source.
  `--cell-type fibroblast` (the only reviewed one) or `--cell-type msc`.
  `--describe-cell-type NAME` prints the provenance and the derived checks.
- **Traction is a stress over the contact area**, and cells now hold **strain
  energy** — the quantity traction force microscopy reports, so the model has a
  direct comparison with experiment it did not have before. It currently reads
  about **10x below** a fibroblast on flat TFM; that is a visible calibration
  target rather than an invisible one.
- **Cells crawl on cells** (`cells.stacking`): a cell above the monolayer anchors
  to a *cell*, and pulls at ~18 % of what it would on the granule. A second
  storey adds connectivity without adding much traction.
- **The wall is a real contact** — its stiffness reaches the implicit step, which
  took a bed off the overlap rail.
- **A convergence detector** (`convergence.enable`, off by default) with
  proportional windows, and `pipeline/shadow_check.py` to calibrate it against
  runs you have already paid for.

### V3.2 – V3.5 in one paragraph each

- **V3.2** — soft granules: `contact.overlap_model: elastic` sizes the overlap
  allowance from the contact law, `contact.semi_implicit` takes stability duty off
  the velocity cap, and the **numerical rails became observable**
  (`overlap_clip_fraction`, `frac_velocity_clipped`). Shape-aware packing.
- **V3.3** — structural metrics that measure the contact graph rather than a
  halo-inflated render: union-find percolation, Laguerre local packing fraction,
  Katz-Thompson permeability. Plus a minimum-image guard that had been silently
  dropping periodic pairs.
- **V3.4** — the **support-function MTD** contact solver. The old common-normal
  solver found 12 % of true contacts, over-reported penetration 9.8x, and had the
  normal *backwards* for half of 3D shaped contacts.
- **V3.5** — energy descent: `dynamics.gradient_flow` audits `dE/dt <= 0` and
  confirms the assembled force law is a gradient to **3.9e-9**; `packing.relax`
  FIRE-relaxes the packed bed under the dynamics' own force law, taking the
  handoff from 346x the driving load to 1.0x.

---

## What's New in V3.1

The container, the granule material and the cells move closer to the bench experiment:
**fibroblasts on collagen-coated PMMA granules sedimented under gravity in a well** with a
flat floor, a cylindrical side wall and a free top.

- **A container that can compact.** `boundary.shape` (`box` | `cylinder`) and `boundary.top`
  (`wall` | `free`). A closed box is fixed-volume and cannot compact at all; with a free top
  the bed surface can descend, and against an inert wall the bed can pull away from it.
- **Gravity** with per-species density (PMMA 1180 kg/m³), and a packer that **sediments** the
  bed instead of pulling it toward the domain centre (`packing.consolidation`). The old
  behaviour consolidated sub-RCP packings into a ball with empty corners — see the Bug Fixes
  entry in the changelog.
- **Rigid granules.** Declare the true modulus and cap only the contact stiffness
  (`contact.stiffness_cap_kPa`), plus Coulomb friction. Rigidity is enforced geometrically by
  the overlap projection, not by resolved Hertz forces — stated plainly because it bounds
  what the contact forces mean.
- **Cells that contract against a force balance** (`cells.bridging.force_model: hill`) instead
  of pulling with a fixed force, and **cells that divide** (`cells.division`, 24 h doubling
  with contact inhibition).
- **Boundary chemistry**: a coated wall with an immobile granule lining cells can bridge to,
  which is what separates the two regimes seen at the bench — an inert boundary lets the bed
  detach and compact, a functionalized one pins it and the functional phase coarsens.
- **Bed observables** (`bed_height_mean`, `phi_bed`, `bed_radius_p95`, `wall_contact_fraction`)
  measured from positions rather than from the rendered field, a `--sweep well` calibration
  set, and `compare_runs.py --experiment` to overlay measured bed geometry.

## What's New in V3.0

The model is now framed as **fibroblasts migrating on a bed of hydrogel
granules coated with Collagen-I**:

- **Granule species with a degree of functionalization** `f in [0, 1]` (1 = fully
  collagen-coated, 0 = bare). Any number of species can be mixed by volume
  fraction, each with its own radius distribution, shape, stiffness and colour.
  Adhesion and friction mix bilinearly in coverage; cell traction follows a
  Langmuir ligand-density law (`g(f) = f(1+k)/(f+k)`, k = K_sigma/sigma_max, defaults
  from the collagen literature — see `CodeLog/References/fibroblast_parameters.md`);
  cell seeding, bridging eligibility and the bridge lock-in threshold scale
  with the same coverage. `f in {0, 1}` reproduces the V2.7 functional / inert model exactly.
- **Sectioned YAML setup files** (`pipeline/step0_new_setup.py` writes a commented
  template; `step1_config.py --setup FILE --set path=value`) with physical
  defaults for human dermal fibroblasts.
- **Live view**: `step3_simulate.py --live` opens a real-time window (pause,
  single-step, stop, colour by species / f / speed); `pipeline/live_view.py`
  tails or replays any run.
- **Compiled parallel engine** (`gels/kernels/`, numba): contacts, cells and
  bridging, rendering, metrics and packing run as thread-count-independent
  kernels. On the 40-core workstation a 2D step at N = 3000 dropped from
  0.87 s to 0.02 s and a field render at N = 3000 from 298 s to 0.2 s; N = 10^4 to 10^5
  granules are practical. `use_numba: false` runs the pure-Python reference,
  which the compiled kernels are tested against (`tests/`).
- **Showcase and multi-run comparison**: `python pipeline/run_showcase.py` runs
  nine conditions concurrently (a functionalization ladder on one 5 mm domain,
  shaped granules, a 20 mm bed of ~51 000 granules, a 1.6 mm 3D cube) into
  `results/showcase_v3/` and draws them on the same figures with
  `viz2/compare_runs.py` (final scaffolds side by side, metrics overlaid, a
  synchronized timelapse, a summary table). The comparison tool works on any
  set of finished runs.

---

## Quick Start

### Requirements

- Python 3.9+
- numpy, scipy, matplotlib
- Recommended: numba (compiled parallel kernels; without it everything runs on the slow Python reference), pyyaml (YAML setup files)
- Optional: pyvista (3D rendering), scikit-image (marching cubes)
- Optional for post-processing: imageio, imageio-ffmpeg, tqdm

### Run a Simulation

Simulations run locally as five numbered steps, each operating on one run
directory. Every step prints the command for the next one when it finishes.

```bash
python pipeline/step0_new_setup.py   -o setup.yaml                      # optional: commented template to edit
python pipeline/step1_config.py      --setup setup.yaml --name my_run   # or: --name my_run --t_total 72
python pipeline/step2_pack.py        -i results/my_run
python pipeline/step3_simulate.py    -i results/my_run
python pipeline/step4_postprocess.py -i results/my_run
python pipeline/step5_analysis.py    -i results/my_run
```

Start from a trial config instead of flags:

```bash
python pipeline/step1_config.py --trial Trials/DOE2_2D_0001.json
```

See `pipeline/README.md` for the full guide, including resuming an interrupted
run (`step3_simulate.py --continue`) and re-running individual steps.

### 3D Volumetric Simulation

```python
from gels import Params, run

p = Params(
    mode='3D',
    Lx=400, Ly=400, Lz=400,
    t_total=48.0,
    shape_enabled=True,
)
hist, snaps, p, gs = run(p)
```

### 2D-Slice Mode

Generate 3D granule packing, slice at z-midplane, run 2D simulation:

```python
p = Params(
    mode='2D-slice',
    Lx=800, Ly=800, Lz=800,
    t_total=72.0,
)
hist, snaps, p, gs = run(p)
```

### Custom Parameters

```python
from gels import Params, run

p = Params(
    E_modulus=5.0,          # kPa — softer hydrogel
    phi_f_target=0.30,      # 30% functional
    phi_i_target=0.15,      # 15% inert
    t_total=48.0,           # 48-hour simulation
)
hist, snaps, p, gs = run(p)
```

### Understanding Stiffness

The Young's modulus `E_modulus` controls how much granules overlap under cell forces:

| E (kPa) | Equilibrium overlap | delta/R | Physical meaning |
|---------|-------------------|---------|-----------------|
| 1 | ~5.6 um | 14% | Very soft, significant deformation |
| 10 | ~1.2 um | 3% | Moderate (default) |
| 100 | ~0.26 um | 0.7% | Stiff, nearly rigid |

---

## Simulation Modes

| Mode | Description | Domain | Granule Shape |
|------|-------------|--------|---------------|
| `"2D"` | Pure 2D simulation | Lx × Ly, 4 walls | Superellipses |
| `"2D-slice"` | 3D packing → z-midplane slice → 2D sim | Lx × Ly × Lz → Lx × Ly | Superellipsoid → superellipse cross-section |
| `"3D"` | Full volumetric simulation | Lx × Ly × Lz, 6 walls | Superellipsoids |

---

## Project Structure

```
GELS/
├── gels/                         # Simulation engine package (V2.7+)
│   ├── __init__.py               # Re-exports Params, run, load_run, ...
│   ├── engine.py                 # Primary engine (was new_dem_0.py)
│   ├── config.py                 # Sectioned YAML setup <-> Params (V3.0)
│   ├── presets.py                # Named bundles of correlated overrides (V3.1)
│   ├── celltypes/                # Instantiable cell types, each value sourced (V3.6)
│   ├── convergence.py            # Proportional-window arrest detector (V3.6)
│   ├── division.py               # Cell division pass (V3.1)
│   ├── laguerre.py               # Halo-free local packing fraction (V3.3)
│   ├── pore.py                   # Katz-Thompson permeability, tortuosity (V3.3)
│   ├── kernels/                  # Compiled compute layer + its Python oracle (V3.0)
│   ├── live/                     # Observer hook, live viewer (V3.0)
│   ├── io/                       # Background snapshot writer (V3.0)
│   └── lsdem.py                  # LS-DEM deformable particles
├── pipeline/                     # Step-by-step local runner (V2.7+)
│   ├── step0_new_setup.py        # Write a sectioned YAML setup; --preset, --cell-type
│   ├── step1_config.py           # Resolve parameters -> params.json
│   ├── step2_pack.py             # Packing + cell seeding -> snap_0000
│   ├── step3_simulate.py         # Advance dynamics to t_total
│   ├── step4_postprocess.py      # Figures and movies
│   ├── step5_analysis.py         # Mathematical analysis
│   ├── _common.py                # Step state, guards, config loading
│   └── README.md                 # Pipeline guide
├── viz/                          # Visualization package (V1.x suite)
│   ├── postprocess.py            # Unified post-processing orchestrator
│   ├── cells.py                  # Cell morphology & stress maps
│   ├── stress.py                 # 3D surface stress, isosurfaces
│   ├── compaction.py             # Void-space & compaction plots
│   ├── percolation.py            # Transport property analysis
│   ├── movies.py                 # 3D volumetric animations
│   ├── phases.py                 # Individual phase volumes
│   ├── shapes.py                 # Granule shape gallery
│   ├── doe.py                    # DOE statistical analysis
│   ├── dimensionless.py          # Dimensionless analysis, data collapse
│   ├── scaffold_evolution.py     # 2D microstructure evolution timelapse
│   ├── scaffold_evolution_3d.py  # 3D volumetric evolution, PyVista
│   └── energy_landscape.py       # Energy landscape visualization
├── viz2/                         # V2 visualization suite (pipeline default)
│   ├── postprocess.py            # viz2 orchestrator
│   ├── scaffold_map.py           # Scaffold maps (2D / 3D)
│   ├── voronoi_shapes.py         # Voronoi tessellation
│   ├── phase_fractions.py        # Global and local phase fractions
│   ├── energy_stress.py          # Energy and stress maps
│   ├── void_percolation.py       # Void percolation, Kozeny-Carman
│   └── common.py                 # Shared loading + field reconstruction
├── analysis/                     # Mathematical analysis package
│   ├── mean_field_model.py       # Two-zone compaction ODE
│   ├── spatial_pde.py            # 1D radial PDE compaction model
│   ├── contact_network.py        # Graph-based contact network model
│   ├── contact_network_model.py  # Gillespie SSA stochastic network
│   ├── parameter_sweep.py        # 11-D LHS + 1D PDE sweep
│   ├── energy_landscape.py       # Free energy landscape decomposition
│   ├── coarse_grain.py           # Stress tensor, strain rate, viscosity
│   ├── tissue_descriptors.py     # Tissue architecture descriptors
│   ├── organ_targets.py          # Organ system target vectors
│   └── arch_distance.py          # Architectural distance to organs
├── Trials/                       # Parameter sweep configs (DOE)
│   ├── generate_doe.py           # DOE config generator
│   └── default_trial.json        # Default trial config
├── results/                      # Run directories (gitignored)
├── old code/                     # Retired scripts and HPC machinery (V2.7)
│   ├── hpc/                      # SLURM templates, cluster config, sync
│   └── README.md                 # What was archived and why
├── CodeLog/
│   ├── Architecture/             # Architecture documents
│   ├── Paper/                    # Publication manuscript (LaTeX)
│   ├── Readme/README.md          # This file
│   ├── References/               # Literature references
│   └── Updates/CHANGELOG.md
└── CLAUDE.md                     # Developer guide
```

---

## Post-Processing & Visualization

Visualization is decoupled from the simulation engine (V1.6+). Run post-processing
on saved output data:

```bash
# Run all visualizations on saved output:
python viz/postprocess.py -i results/default

# Skip specific modules:
python viz/postprocess.py -i results/default --skip movies stress

# Run individual viz scripts:
python viz/compaction.py -i results/default
python viz/stress.py -i results/default
```

Or programmatically:
```python
from viz.postprocess import run_all
run_all('results/default')                          # all visualizations
run_all('results/default', skip={'movies','stress'}) # selective
```

### Visualization Modules (`viz/`)

| Module | Plots |
|--------|-------|
| `viz/compaction.py` | Void fraction, packing fraction, compaction ratio, void size distribution, stacked phases |
| `viz/percolation.py` | Kozeny-Carman permeability, porosity + RCP, dimensionless groups, Darcy flow, void connectivity |
| `viz/movies.py` | Rotating 3D isosurface, time-lapse compaction, z-sweep cross-section, composite 2×2 |
| `viz/phases.py` | Phase isosurface strip, phase fractions vs time, tri-plane evolution, interface area vs time |
| `viz/cells.py` | Cell morphology patches, force-magnitude stress maps, cell timelapse GIF |
| `viz/stress.py` | 3D Hertzian surface stress fields, cross-section stress maps, granule evolution GIFs (requires PyVista) |
| `viz/shapes.py` | Granule shape gallery (superellipses, superellipsoids) |
| `viz/doe.py` | DOE statistical analysis & visualization |
| `viz/dimensionless.py` | Dimensionless analysis, data collapse by β/Ca |
| `viz/scaffold_evolution.py` | 2D microstructure evolution timelapse for organ targets (V1.9) |
| `viz/scaffold_evolution_3d.py` | 3D volumetric evolution with PyVista per organ (V1.9) |
| `viz/energy_landscape.py` | Energy landscape per organ, time evolution, decomposition, design space (V1.11) |

### Mathematical Analysis (`analysis/`)

| Module | Purpose |
|--------|---------|
| `analysis/mean_field_model.py` | Volume-conserving two-zone compaction ODE with fitting (V1.9) |
| `analysis/parameter_sweep.py` | 11-D Latin hypercube sweep with 1D PDE, organ prediction (V1.10) |
| `analysis/energy_landscape.py` | Free energy landscape decomposition (6 terms) with kinetics (V1.11) |
| `analysis/coarse_grain.py` | Stress tensor, strain rate, viscosity from DEM |
| `analysis/tissue_descriptors.py` | Tissue architecture descriptor vector |
| `analysis/organ_targets.py` | Organ system target vectors (7 organs) with phase mapping (V1.9) |
| `analysis/arch_distance.py` | Architectural distance to organ targets |

All visualization scripts support PyVista (primary) with matplotlib fallback.

---

## Transport Metrics (V1.4)

The simulation tracks transport-relevant quantities:

| Metric | Description |
|--------|-------------|
| Porosity ε(t) | Void volume fraction over time |
| Kozeny-Carman K | Permeability: K = ε³d²/[180(1-ε)²] |
| RCP fraction | Random close packing limit: φ_RCP ≈ 0.64 + 0.08*(AR-1) |
| Compaction ratio | φ_solid / φ_RCP (1.0 = at RCP) |
| Darcy number | Da = K/L² |
| Darcy flow rate | q = K/μ × dP/L |
| Peclet number | Pe = vL/D |
| Pore Reynolds | Re = ρvd/μ |

---

## Running Batches

HPC execution was retired in V2.7 — everything runs locally. To sweep a set of
trial configs, loop the pipeline over them:

```bash
for trial in Trials/DOE2_2D_*.json; do
    name=$(basename "$trial" .json)
    python pipeline/step1_config.py      --trial "$trial" --name "$name"
    python pipeline/step2_pack.py        -i "results/$name"
    python pipeline/step3_simulate.py    -i "results/$name"
    python pipeline/step4_postprocess.py -i "results/$name"
    python pipeline/step5_analysis.py    -i "results/$name"
done
```

Because each step checkpoints into `pipeline_state.json`, re-running the loop
skips work that already completed and only picks up where it left off.

`python pipeline/run_showcase.py` runs a table of conditions concurrently and
then draws them on the same figures with `viz2/compare_runs.py`.

Before enabling the V3.6 convergence detector on a sweep, calibrate it against
runs you have already finished:

```bash
python pipeline/shadow_check.py results/
python pipeline/shadow_check.py results/ --sweep eps_gel=0.005,0.01,0.02
```

The previous SLURM machinery (`hpc/`, `run_all_trials.py`,
`run_hpc_headless.py`) is archived unmodified under `old code/` — see
`old code/README.md` for what moved and how to recover it from git.

---

## Programmatic Access

The `run()` function returns:
- `hist`: list of metric dictionaries (one per save step)
- `snaps`: list of snapshot dicts with keys `phi_f`, `phi_i`, `phi_v`, `x`, `y`, `r`, `gtype`, etc.
- `p`: the Params used
- `gs`: final GranuleSystem state

---

## Citation

If you use this code in published work, please cite the McGhee Lab at the
University of Illinois at Urbana-Champaign.

---

## License

Internal research code. Contact the McGhee Lab for usage terms.
