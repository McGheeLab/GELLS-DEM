# GELLS-DEM

**Granular Encapsulated Living-cell Laden Scaffold - Discrete Element Method**

A 2D particle dynamics simulator for modelling cell-driven rearrangement of
hydrogel granular scaffolds.

---

## What It Does

GELLS-DEM simulates how cell-laden hydrogel granules reorganize over time within
a confined domain. Functional granules carry cells that form mechanical bridges
to neighbouring functional granules, pulling them together. Inert granules act as
passive spacers. The simulation predicts how scaffold microstructure --- void
networks, functional connectivity, tissue density --- evolves over 24--72 hours.

### Key Physics

- **Hertzian contact mechanics** with physically meaningful Young's modulus (1--100 kPa)
- **Cell-mediated bridging forces** between functional granule pairs
- **Volume conservation** via effective-radius correction for overlapping granules
- **Overdamped Langevin dynamics** (no inertia, appropriate for viscous culture medium)
- **Active noise** on functional granules to model cell-driven fluctuations

---

## Quick Start

### Requirements

- Python 3.9+
- numpy
- scipy
- matplotlib

### Run a Simulation

```bash
python3 new_dem_0.py
```

This runs the default configuration (800 x 800 um domain, E = 10 kPa, 72 hours)
and displays granule positions, phase fields, and metric time series.

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

## Project Structure

```
GELLS-DEM/
├── new_dem_0.py                 # Main simulation engine
├── new_dem_visualization.py     # Unified post-processing & visualization
├── new_dem_postprocess.py       # Legacy-compatible post-processing
├── dem_config.json              # Default JSON config (legacy format)
├── Trials/                      # Parameter sweep configs (Trial1-12.json)
├── CodeLog/
│   ├── Architecture/            # System architecture document
│   │   └── ARCHITECTURE.md
│   ├── Readme/                  # This README
│   │   └── README.md
│   └── Updates/                 # Changelog
│       └── CHANGELOG.md
├── old/                         # Archived legacy code
└── CLAUDE.md                    # Developer guide & project conventions
```

---

## Simulation Outputs

### Console Output

Each save step prints a status line:

```
  t(h)  f_cl   f_lf  v_cl  tissue  bridges  disp_f   δ/R%  AreaCon
   0.0    12   0.12     8   0.045       18     0.0    0.0  1.0000
   2.0    11   0.14     7   0.052       22     3.2    1.4  0.9998
```

| Column | Meaning |
|--------|---------|
| `t(h)` | Simulation time in hours |
| `f_cl` | Number of functional clusters |
| `f_lf` | Largest functional cluster as fraction of total |
| `v_cl` | Number of void clusters |
| `tissue` | Fraction of domain with phi_f > 0.5 |
| `bridges` | Active cell bridges |
| `disp_f` | Mean functional granule displacement (um) |
| `delta/R%` | Maximum overlap as percentage of smaller radius |
| `AreaCon` | Area conservation metric (1.0 = perfect) |

### Figures

When run as a script, four matplotlib figures are generated:
1. **Granule positions** at 5 time snapshots
2. **Phase fields** (functional, inert, void) at 5 time snapshots
3. **Metric time series** (6 panels: clusters, connectivity, bridges, displacement, tissue, cluster area)
4. **RGB composite** (Red=functional, Green=void, Blue=inert)

### Programmatic Access

The `run()` function returns:
- `hist`: list of metric dictionaries (one per save step)
- `snaps`: list of snapshot tuples `(phi_f, phi_i, phi_v, x, y, r, gtype)`
- `p`: the Params used
- `gs`: final GranuleSystem state

---

## Configuration Parameters

All parameters are fields of the `Params` dataclass. Key groups:

### Domain
- `Lx`, `Ly` — Box dimensions (um). Default: 800 x 800.

### Granule Properties
- `R_func_mean/std` — Functional granule radius distribution (um). Default: 40 +/- 5.
- `R_inert_mean/std` — Inert granule radius distribution (um). Default: 60 +/- 8.
- `phi_f_target`, `phi_i_target` — Target area fractions. Default: 0.25, 0.20.

### Cell Properties
- `n_cells_per_granule` — Cells seeded per functional granule. Default: 8.
- `cell_diameter` — Cell size (um). Default: 15.
- `cell_coverage` — Max fraction of granule surface covered. Default: 0.6.
- `F_max_per_cell` — Maximum force per cell bridge (nN). Default: 50.

### Contact Mechanics
- `E_modulus` — Young's modulus of hydrogel (kPa). Default: 10.
- `poisson_ratio` — Poisson's ratio. Default: 0.45.

### Cell Bridging
- `k_cell` — Cell bridge spring constant (nN/um). Default: 0.8.
- `L_max` — Maximum bridging gap (um). Default: 60.
- `L_rest` — Bridge rest length (um). Default: 12.

### Time Integration
- `dt` — Timestep (hours). Default: 0.5.
- `t_total` — Total simulation time (hours). Default: 72.
- `v_max` — Velocity cap (um/hr). Default: 20.

---

## Post-Processing

### Unified Post-Processor (`new_dem_visualization.py`)

Full-featured visualisation suite that reads JSON frame files from disk:
- Z-stack evolution plots
- 3D rotating isosurface GIFs
- Metric dashboards
- Phase evolution strips

```bash
python3 new_dem_visualization.py -i ./simulations/run1
```

### Legacy Post-Processor (`new_dem_postprocess.py`)

Lighter weight, generates slice-bin plots and 3D voxel GIFs from JSON frames.

---

## Citation

If you use this code in published work, please cite the McGhee Lab at the
University of Illinois at Urbana-Champaign.

---

## License

Internal research code. Contact the McGhee Lab for usage terms.
