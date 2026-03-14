# GELLS-DEM Project Guide

## Project Overview

**GELLS-DEM** (Granular Encapsulated Living-cell Laden Scaffold - Discrete Element Method)
is a 2D overdamped particle dynamics simulator for modelling cell-driven rearrangement
of hydrogel granular scaffolds. The primary simulation engine is `new_dem_0.py`.

**Current version: V1.2**

## Repository Layout

```
GELLS-DEM/
├── new_dem_0.py                 # PRIMARY simulation engine (V1.2)
├── new_dem_visualization.py     # Unified post-processing & visualisation
├── new_dem_postprocess.py       # Legacy-compatible post-processing (JSON frames)
├── dem_config.json              # Default JSON config (legacy format)
├── Trials/                      # Parameter sweep JSON configs (Trial1-12)
├── CodeLog/
│   ├── Architecture/            # Architecture documents
│   ├── Readme/                  # README documents
│   └── Updates/                 # Changelog / update log
├── old/                         # Archived legacy code (3D superellipsoid, etc.)
└── CLAUDE.md                    # THIS FILE
```

## Key Technical Decisions

- **Units throughout**: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
- **Contact model**: Hertzian (V1.1+). Stiffness is derived from `E_modulus` and `poisson_ratio`,
  NOT set as an arbitrary spring constant. See `hertz_contact_force()`.
- **Cell force model**: Motor-clutch (V1.2+, Chan & Odde 2008). Cell traction depends on
  substrate stiffness, replacing the old simple spring. See `motor_clutch_force()`.
- **Cell lifecycle** (V1.2+): Cells are 20 µm spheres → attach at ~3 h → spread to 5 µm-tall
  ellipsoids (volume conserved) → overcrowded cells crawl on others → bridges form via
  motor-clutch adhesions. See `update_cell_state()`.
- **Volume conservation**: Overlap lens area is tracked and redistributed via effective radii
  for rendering. Forces still use the original (undeformed) radii.
- **Integration**: Overdamped Euler (no inertia). Velocity cap prevents numerical blowup.
- **Neighbour search**: `scipy.spatial.cKDTree` with a cutoff of `2*max_r + cell_sense_distance`.

## Conventions

- Granule types: `0` = functional (cell-laden), `1` = inert (passive).
- Phase fields: `phi_f` = functional, `phi_i` = inert, `phi_v` = void = 1 - phi_f - phi_i.
- All parameters live in the `Params` dataclass; modify defaults there or pass overrides.

## Documentation Requirements

- **Update log**: Every code change must be recorded in `CodeLog/Updates/CHANGELOG.md`.
  Bug fixes go in a dedicated "Bug Fixes" subsection within the relevant version entry.
- **Architecture**: The system architecture document lives at `CodeLog/Architecture/ARCHITECTURE.md`.
  Update it when modules are added, removed, or significantly restructured.
- **README**: The user-facing README is at `CodeLog/Readme/README.md`.

## Running the Simulation

```bash
python3 new_dem_0.py
```

Or import and call programmatically:

```python
from new_dem_0 import Params, run
p = Params(E_modulus=5.0, t_total=48.0)
hist, snaps, p, gs = run(p)
```

## Dependencies

- Python 3.9+
- numpy, scipy, matplotlib
- Optional for post-processing: imageio, scikit-image, tqdm
