# GELLS-DEM

**Granular Encapsulated Living-cell Laden Scaffold - Discrete Element Method**

A 2D/3D particle dynamics simulator for modelling cell-driven rearrangement of
hydrogel granular scaffolds.

---

## What It Does

GELLS-DEM simulates how cell-laden hydrogel granules reorganize over time within
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
GELLS-DEM/
├── new_dem_0.py                 # Main simulation engine (V1.5.2)
├── viz_compaction.py            # Void-space & compaction plots
├── viz_percolation.py           # Transport property analysis
├── viz_movies.py                # 3D volumetric animations
├── viz_phases.py                # Individual phase volumes
├── viz_cells.py                 # Cell morphology, stress maps, GIFs (V1.5.1)
├── viz_stress.py                # 3D surface stress, isosurfaces, GIFs (V1.5.2)
├── new_dem_visualization.py     # Legacy post-processing
├── new_dem_postprocess.py       # Legacy post-processing (JSON)
├── run_hpc_headless.py          # HPC headless runner
├── run_all_trials.py            # Batch trial runner (local / SLURM array)
├── Trials/                      # Parameter sweep configs (flat JSON format)
│   ├── Trial15_3D.json          # 3D trial (800x800x500)
│   ├── Trial16_3D.json          # 3D small domain (400x400x400)
│   └── Trial17_2D.json          # 2D baseline
├── CodeLog/
│   ├── Architecture/ARCHITECTURE.md
│   ├── Readme/README.md         # This file
│   ├── References/              # Literature references
│   └── Updates/CHANGELOG.md
├── hpc/                         # HPC setup, user configs, sync scripts
├── old/                         # Archived legacy code
└── CLAUDE.md                    # Developer guide + HPC best practices
```

---

## Visualization

### Built-in Plots

When run as a script, four matplotlib figures are generated:
1. **Granule positions** at 5 time snapshots
2. **Phase fields** (functional, inert, void) at 5 time snapshots
3. **Metric time series** (3×3 grid: topology, dynamics, cell state)
4. **RGB composite** (Red=functional, Green=void, Blue=inert)

### Specialized Visualization Scripts (V1.4+)

| Script | Plots |
|--------|-------|
| `viz_compaction.py` | Void fraction, packing fraction, compaction ratio, void size distribution, stacked phases |
| `viz_percolation.py` | Kozeny-Carman permeability, porosity + RCP, dimensionless groups (Pe, Re, Da, compaction, porosity ratio), Darcy flow, void connectivity |
| `viz_movies.py` | Rotating 3D isosurface, time-lapse compaction, z-sweep cross-section, composite 2×2 |
| `viz_phases.py` | Phase isosurface strip, phase fractions vs time, tri-plane evolution (XY/XZ/YZ), interface area vs time |
| `viz_cells.py` | Cell morphology patches, force-magnitude stress maps, cell timelapse GIF (V1.5.1) |
| `viz_stress.py` | 3D Hertzian surface stress fields, cross-section stress maps, granule evolution GIFs, isosurface rendering (V1.5.2, requires PyVista) |

Usage:
```bash
python viz_compaction.py -i ./simulations/run1
python viz_percolation.py -i ./simulations/run1
python viz_movies.py -i ./simulations/run1
python viz_phases.py -i ./simulations/run1
python viz_cells.py -i ./simulations/run1
python viz_stress.py -i ./simulations/run1
```

Or programmatically:
```python
import viz_compaction, viz_percolation, viz_movies, viz_phases, viz_cells, viz_stress

viz_compaction.run_all(hist, snaps=snaps, outdir='plots/')
viz_percolation.run_all(hist, outdir='plots/')
viz_movies.run_all(snaps, hist, p, outdir='plots/')
viz_phases.run_all(hist, snaps=snaps, p=p, outdir='plots/')
viz_cells.run_all(snaps, hist, p, outdir='plots/')
viz_stress.run_all(snaps, hist, p, outdir='plots/')
```

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
