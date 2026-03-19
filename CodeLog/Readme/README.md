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

- **Hertzian contact mechanics** with physically meaningful Young's modulus (1--100 kPa)
- **Motor-clutch cell bridging** (Chan & Odde 2008) — substrate stiffness-dependent traction with probabilistic bridge initiation, maturity ramp, and senescence
- **DMT adhesion** and **area-dependent friction** by granule pair type
- **Superellipsoid granule shapes** (3D) / superellipse shapes (2D)
- **Quaternion-based 3D rotational dynamics**
- **Transport metrics**: Kozeny-Carman permeability, Darcy flow, RCP compaction
- **Volume conservation** via effective-radius correction for overlapping granules
- **Overdamped Langevin dynamics** (no inertia, appropriate for viscous culture medium)

---

## Quick Start

### Requirements

- Python 3.9+
- numpy, scipy, matplotlib
- Optional: pyvista (3D rendering), scikit-image (marching cubes), numba (JIT acceleration)

### Run a Simulation

```bash
# Default 2D simulation
python3 new_dem_0.py
```

This runs the default configuration (800 x 800 um domain, E = 10 kPa, 72 hours)
and displays granule positions, phase fields, and metric time series.

### 3D Volumetric Simulation

```python
from new_dem_0 import Params, run

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
from new_dem_0 import Params, run

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
├── new_dem_0.py                 # Primary simulation engine (V1.12, no plotting)
├── run_hpc_headless.py          # HPC headless runner (supports 2D/3D)
├── run_all_trials.py            # Batch trial runner (local / SLURM array)
├── run_analysis_pipeline.py     # Full analysis pipeline orchestrator
├── reconstruct_history.py       # Reconstruct history from snapshots
├── viz/                         # Visualization package
│   ├── postprocess.py           # Unified post-processing orchestrator
│   ├── cells.py                 # Cell morphology & stress maps
│   ├── stress.py                # 3D surface stress, isosurfaces
│   ├── compaction.py            # Void-space & compaction plots
│   ├── percolation.py           # Transport property analysis
│   ├── movies.py                # 3D volumetric animations
│   ├── phases.py                # Individual phase volumes
│   ├── shapes.py                # Granule shape gallery
│   ├── doe.py                   # DOE statistical analysis
│   ├── dimensionless.py         # Dimensionless analysis, data collapse
│   ├── scaffold_evolution.py    # 2D microstructure evolution timelapse (V1.9)
│   ├── scaffold_evolution_3d.py # 3D volumetric evolution, PyVista (V1.9)
│   └── energy_landscape.py      # Energy landscape visualization (V1.11)
├── analysis/                    # Mathematical analysis package
│   ├── mean_field_model.py      # Two-zone compaction ODE (V1.9)
│   ├── parameter_sweep.py       # 11-D LHS + 1D PDE sweep (V1.10)
│   ├── energy_landscape.py      # Free energy landscape decomposition (V1.11)
│   ├── coarse_grain.py          # Stress tensor, strain rate, viscosity
│   ├── tissue_descriptors.py    # Tissue architecture descriptors
│   ├── organ_targets.py         # Organ system target vectors + phase mapping (V1.9)
│   └── arch_distance.py         # Architectural distance to organs
├── Trials/                      # Parameter sweep configs (DOE)
│   ├── generate_doe.py          # DOE config generator
│   └── default_trial.json       # Default V1.9 trial config
├── CodeLog/
│   ├── Architecture/            # Architecture documents
│   ├── Paper/                   # Publication manuscript (LaTeX)
│   ├── Readme/README.md         # This file
│   ├── References/              # Literature references
│   └── Updates/CHANGELOG.md
├── hpc/                         # HPC setup, user configs, sync scripts
└── CLAUDE.md                    # Developer guide + HPC best practices
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

## HPC Usage

### Run on HPC (e.g., UArizona Puma)

```bash
# Set up environment (one time)
python3 hpc/generate_hpc_scripts.py       # generates scripts, syncs repo, sets up venv

# Run a single simulation
python3 run_hpc_headless.py --t_total 48 --output-dir results/run1

# Run from a trial config
python3 run_hpc_headless.py --trial Trials/Trial15_3D.json --output-dir results/3d_test

# Run all trials as SLURM array job (auto-submit + monitor)
# Set RUN_MODE=3 in run_all_trials.py, then:
python3 run_all_trials.py

# Sync results from cluster
python3 hpc/sync_results.py
```

See `CLAUDE.md` for detailed UArizona HPC best practices.

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
