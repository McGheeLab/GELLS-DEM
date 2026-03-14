# GELLS-DEM Architecture Document

**Version:** V1.2
**Last updated:** 2026-03-14
**Primary source file:** `new_dem_0.py`

---

## 1. System Overview

GELLS-DEM simulates the rearrangement of hydrogel granular scaffolds driven by
cell-mediated forces. It models two populations of 2D circular granules ---
functional (cell-laden) and inert (passive) --- within a confined domain, using
overdamped Langevin dynamics.

The simulator is designed to predict scaffold microstructure evolution
(void network topology, functional connectivity, tissue formation) over
experimentally relevant timescales (24--72 hours).

```
┌─────────────────────────────────────────────────────────────┐
│                        new_dem_0.py                         │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌────────────────────┐  │
│  │  Params   │──▶│ generate_    │──▶│  GranuleSystem     │  │
│  │ (config)  │   │ packing()   │   │  (state container) │  │
│  └──────────┘   └──────────────┘   └─────────┬──────────┘  │
│                                               │             │
│  ┌────────────────────────────────────────────▼──────────┐  │
│  │                    run() loop                         │  │
│  │                                                       │  │
│  │   ┌──────────────────┐    ┌─────────────────────┐    │  │
│  │   │ compute_forces() │───▶│     step()           │    │  │
│  │   │  • Hertz contact │    │  • update_cell_state │    │  │
│  │   │  • motor-clutch  │    │  • overdamped Euler  │    │  │
│  │   │  • wall repulsion│    │  • velocity cap      │    │  │
│  │   │  • active noise  │    │  • wall clamp        │    │  │
│  │   └──────────────────┘    └─────────────────────┘    │  │                                │  │
│  │                                                       │  │
│  │   ┌──────────────────┐    ┌─────────────────────┐    │  │
│  │   │ render_fields()  │───▶│ compute_metrics()   │    │  │
│  │   │  • eff. radii    │    │  • connectivity     │    │  │
│  │   │  • tanh profiles │    │  • overlap stats    │    │  │
│  │   │  • vol. conserv. │    │  • force stats      │    │  │
│  │   └──────────────────┘    └─────────────────────┘    │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                  Visualisation                         │  │
│  │  plot_granules  plot_fields  plot_timeseries           │  │
│  │  plot_composite                                        │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Module Breakdown

### 2.1 Configuration — `Params` (dataclass)

| Group | Parameters | Units | Purpose |
|-------|-----------|-------|---------|
| Domain | `Lx`, `Ly` | um | Simulation box size |
| Granule sizes | `R_func_mean/std`, `R_inert_mean/std` | um | Gaussian size distributions |
| Composition | `phi_f_target`, `phi_i_target` | -- | Target area fractions |
| Cell geometry | `n_cells_per_granule`, `cell_diameter`, `cell_height_spread`, `cell_coverage` | --, um, um, -- | Cell seeding (20 um spheres, 5 um spread height) |
| Cell timeline | `t_attach_onset`, `t_attach_half`, `t_spread_duration`, `fa_maturation_rate` | h, h, h, 1/h | Attachment/spreading/FA kinetics |
| Motor-clutch | `n_motors`, `F_motor_stall`, `n_clutches`, `k_clutch`, `k_on_clutch`, `k_off_clutch` | --, nN, --, nN/um, 1/s, 1/s | Chan & Odde 2008 force model |
| Cell bridging | `cell_sense_distance`, `F_max_per_cell`, `L_rest` | um, nN, um | Filopodia range, force cap, rest length |
| Contact mechanics | `E_modulus`, `poisson_ratio` | kPa, -- | Hertzian contact (V1.1) |
| Drag | `eta`, `drag_scale` | Pa-s, -- | Overdamped dynamics |
| Noise | `T_active` | nN-um | Active temperature (functional only) |
| Time | `dt`, `t_total`, `save_every_h`, `v_max` | h, h, h, um/h | Integration control |
| Rendering | `Ngrid`, `interface_width` | --, um | Phase field grid |

### 2.2 State Container — `GranuleSystem`

Stores all per-granule arrays:
- **Position**: `x`, `y` (float64)
- **Radius**: `r` (float64, constant throughout simulation)
- **Type**: `gtype` (int, 0=functional, 1=inert)
- **Cell count**: `n_cells` (float64, projected-area limited seeded count)
- **Cell state** (V1.2):
  - `n_attached` (float64) — cells that have attached to granule surface
  - `spread_fraction` (float64, 0–1) — cell morphology: 0=sphere, 1=fully spread ellipsoid
  - `fa_maturity` (float64, 0–1) — focal adhesion maturation state
  - `n_overcrowded` (float64) — cells crawling on top of other cells
- **Derived masks**: `func_mask`, `inert_mask` (bool arrays)
- **Particle count**: `N` (int)

### 2.3 Hertz Contact Mechanics

Added in V1.1. Three functions implement the physics:

#### `hertz_contact_force(E_star_Pa, R_eff_um, delta_um) -> float`
Computes the Hertzian normal force:

```
F = (4/3) E* sqrt(R*) delta^(3/2)
```

Unit conversion factor of 1e-3 maps (Pa, um, um) -> nN.

#### `overlap_lens_area(R1, R2, d) -> float`
Exact 2D lens-shaped intersection area between two circles. Used for
volume conservation tracking and effective radii computation.

#### `compute_effective_radii(gs) -> (r_eff, overlap_area)`
For each granule, sums its share of overlap area with all neighbours
(split proportional to r^2), then inflates the display radius:

```
r_eff = sqrt(r^2 + delta_A / pi)
```

This ensures total material area is conserved despite DEM overlaps.

#### `print_stiffness_info(p)`
Diagnostic: prints the expected equilibrium overlap for the configured
modulus and typical cell force. Helps users verify physical meaning.

### 2.4 Cell Geometry & Motor-Clutch Model (V1.2)

#### `cell_projected_area(spread_frac, cell_d, cell_h)`
Projected area of a cell transitioning from sphere to oblate ellipsoid.
Volume is conserved: `V = (4/3)pi(d/2)^3`. When spread (height `h`), the
semi-major axis `a = sqrt(d^3 / 4h)` gives a footprint ~4x larger than the
spherical projection.

#### `max_cells_on_granule(R, cell_proj_area, coverage)`
Capacity = `floor(pi*R^2 * coverage / A_cell)`. Decreases as cells spread.

#### `motor_clutch_force(E_kPa, p, fa_maturity)`
Steady-state traction force per cell from Chan & Odde (2008):
```
k_sub = pi * E * a_cell / (1-nu^2)     substrate stiffness (nN/um)
k_opt = n_clutches * k_clutch           clutch ensemble stiffness
engagement = k_on / (k_on + k_off)      clutch fraction
F_mc = F_stall * k_sub/(k_sub+k_opt) * engagement * fa_maturity
```
Gives ~2 nN on 1 kPa, ~12 nN on 10 kPa, ~21 nN on 100 kPa.

#### `update_cell_state(gs, p, t)`
Per-step cell lifecycle:
1. **Attachment**: sigmoidal kinetics after `t_attach_onset` (~3 h)
2. **Spreading**: linear ramp from sphere to ellipsoid, rate modulated by
   substrate stiffness (stiffer → faster via motor-clutch mechanotransduction)
3. **FA maturation**: ramps at `fa_maturation_rate` once spreading begins
4. **Overcrowding**: when spread footprint exceeds granule capacity, excess
   cells crawl on top of neighbours

### 2.5 Packing Generator — `generate_packing()`

Random sequential addition (RSA) with:
- Interleaved placement (inert-functional-inert-...) for spatial mixing.
- Minimum surface gap of 2 um enforced.
- Up to 800 random placement attempts per granule.
- Cell count per functional granule is `min(n_cells_input, projected_area_cap)`,
  where the cap uses the spherical cell projected area `pi*(d/2)^2`.

### 2.6 Force Computation — `compute_forces()`

Four force contributions, evaluated per timestep:

| Force | Scope | Law | Key parameters |
|-------|-------|-----|----------------|
| **Contact** | All overlapping pairs | Hertz: F = (4/3) E* sqrt(R*) delta^1.5 | `E_modulus`, `poisson_ratio` |
| **Cell bridging** | Functional-functional, gap in (0, sense_dist), attached cells | Motor-clutch: F = F_mc * n_bridges (stiffness + FA dependent) | `n_motors`, `F_motor_stall`, `n_clutches`, `k_clutch`, `cell_sense_distance` |
| **Wall** | Granules penetrating boundary | Hertz (sphere vs rigid flat): E*_wall = 2 * E*_gg | `E_modulus`, `poisson_ratio` |
| **Active noise** | Functional granules only | Gaussian white noise ~ sqrt(2 gamma T_active / dt) | `T_active` |

**Effective moduli:**
- Granule-granule: `E* = E / [2(1 - nu^2)]`
- Granule-wall (rigid limit): `E* = E / (1 - nu^2)`

**Neighbour search:** `cKDTree.query_pairs()` with cutoff `2*max_r + cell_sense_distance`.

**Cell bridging details (V1.2):**
- Only attached, non-overcrowded cells can bridge (`n_avail = n_attached - n_overcrowded`)
- Bridge count: `n_br = sqrt(n_avail_i * n_avail_j) * proximity`
- Force per cell from `motor_clutch_force()`, averaged FA maturity from both granules
- Bridges only exert force when gap > L_rest (cells under tension)

### 2.7 Time Integration — `step()`

```
update_cell_state(gs, p, t)    cell lifecycle (V1.2)
F = compute_forces(gs, p, rng)
v_i = F_i / gamma_i           gamma_i = drag_scale * r_i
|v_i| = min(|v_i|, v_max)     velocity cap for stability
x_i += v_i * dt
x_i = clamp(x_i, walls)       hard wall boundary
```

No inertial terms (overdamped regime appropriate for viscous medium).

### 2.8 Phase Field Rendering — `render_fields()`

Converts particle positions to continuous fields on an (Ngrid x Ngrid) grid:
1. Computes volume-conserving effective radii via `compute_effective_radii()`.
2. Stamps each granule as a tanh profile: `0.5 * (1 - tanh((dist - r_eff) / w))`.
3. Uses `max()` blending (not additive) to prevent double-counting.
4. Clamps `phi_f + phi_i <= 0.99` to ensure physical bounds.

Returns `(phi_f, phi_i, phi_v)` where `phi_v = 1 - phi_f - phi_i`.

### 2.9 Metrics — `compute_metrics()`

Computed at each save step:

| Metric | Description |
|--------|-------------|
| `phi_f_mean`, `phi_i_mean`, `phi_v_mean` | Mean phase fractions |
| `func_nc`, `func_lf`, `func_cov` | Functional cluster count, largest fraction, coverage |
| `void_nc`, `void_lf`, `void_cov` | Void cluster count, largest fraction, coverage |
| `inert_nc`, `inert_lf` | Inert cluster count, largest fraction |
| `tissue_frac` | Fraction of domain with phi_f > 0.5 |
| `func_max_area` | Area of largest functional cluster (um^2) |
| `packing_func_rich` | Mean packing in functional-rich regions |
| `F_mean`, `F_max`, `F_func_mean` | Force statistics (nN) |
| `n_contacts` | Number of overlapping granule pairs |
| `max_overlap_ratio` | Maximum delta/R across all contacts |
| `total_overlap_area` | Total lens overlap area (um^2) |
| `area_conservation` | 1 - (overlap_area / total_granule_area) |
| `n_bridges` | Active cell bridges (functional pairs with attached cells in sensing range) |
| `disp_func`, `disp_inert` | Mean displacement from initial positions (um) |
| `n_attached_total` | Total attached cells across all functional granules (V1.2) |
| `n_seeded_total` | Total seeded cells (V1.2) |
| `mean_spread_frac` | Mean cell spread fraction across functional granules (V1.2) |
| `mean_fa_maturity` | Mean focal adhesion maturity (V1.2) |
| `n_overcrowded_total` | Total overcrowded cells (crawling on others) (V1.2) |

Cluster analysis uses `scipy.ndimage.label` on thresholded fields.

### 2.10 Visualisation Functions

| Function | Output |
|----------|--------|
| `plot_granules()` | Circle patches at 5 time snapshots |
| `plot_fields()` | 3-row (phi_f, phi_i, phi_v) phase field heatmaps |
| `plot_timeseries()` | 3x3 grid: topology, dynamics, cell state (V1.2) |
| `plot_composite()` | RGB composite (R=functional, G=void, B=inert) |

### 2.11 Main Simulation Loop — `run()`

```
generate_packing() -> GranuleSystem
update_cell_state(t=0)   # initial cell state
for each timestep:
    t += dt
    step(gs, p, rng, t)     # update_cell_state + compute_forces + integrate
    if save_step:
        render_fields()
        compute_metrics()
        store snapshot (includes cell state arrays)
return (history, snapshots, params, final_state)
```

---

## 3. Data Flow

```
Params ──▶ generate_packing() ──▶ GranuleSystem
                                       │
                        ┌──────────────┘
                        ▼
                  ┌───────────┐
                  │  run()    │◀── rng (seeded)
                  │  loop     │
                  └─────┬─────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
     hist (list of dicts)   snaps (list of tuples)
              │                   │
              ▼                   ▼
     plot_timeseries()    plot_granules()
                          plot_fields()
                          plot_composite()
```

**Snapshot tuple format (V1.2):**
```python
(phi_f, phi_i, phi_v, x_array, y_array, r_array, gtype_array,
 n_attached_array, spread_fraction_array, fa_maturity_array, n_overcrowded_array)
```

---

## 4. External Files

### Trial Configuration JSONs (`Trials/`)

Legacy format from earlier DEM versions. Fields include domain, granule shapes
(aspect ratio, roundness, roughness), cell properties, mechanics parameters,
and time stepping. These configs are **not directly consumed** by the current
`new_dem_0.py` (which uses the `Params` dataclass), but serve as parameter
records for experimental sweeps.

### Post-Processing Scripts

| File | Purpose |
|------|---------|
| `new_dem_visualization.py` | Full-featured post-processor: z-stacks, 3D isosurfaces, metric dashboards, animated GIFs. Reads JSON frame files from disk. |
| `new_dem_postprocess.py` | Lighter post-processor for 2D/3D frame JSONs. Generates slice-bin evolution plots and rotating 3D voxel GIFs. |

### Legacy Code (`old/`)

Archived versions including 3D superellipsoid DEM, Numba-accelerated solvers,
and cell-stress visualisation. Retained for reference but not actively maintained.

---

## 5. Design Constraints and Assumptions

1. **2D only**: Current engine operates in 2D (circles, not spheres). 3D extension
   exists in `old/new_dem.py` but is not maintained.
2. **Overdamped regime**: No inertial terms. Valid for cell-culture timescales
   (hours) in viscous medium.
3. **Rigid granules**: Granule radii do not change during simulation. Deformation
   effects are captured through the Hertz contact force and volume-conserving
   effective radii for rendering.
4. **No friction**: Only normal contact forces. Tangential/rotational degrees of
   freedom are not modelled.
5. **Hertz validity**: The contact model assumes small overlaps (delta/R < ~10%).
   The `max_overlap_ratio` metric monitors this assumption.
6. **Deterministic with seed**: All randomness flows through `numpy.random.Generator`,
   seeded for reproducibility.
