# GELLS-DEM Architecture Document

**Version:** V1.4.1
**Last updated:** 2026-03-14
**Primary source file:** `new_dem_0.py`

---

## 1. System Overview

GELLS-DEM simulates the rearrangement of hydrogel granular scaffolds driven by
cell-mediated forces. It models two populations of granules --- functional (cell-laden)
and inert (passive) --- within a confined domain, using overdamped Langevin dynamics.

The simulator operates in three modes:
- **2D**: Pure 2D simulation with superellipse granules (V1.3 behavior)
- **2D-slice**: Generate 3D superellipsoid packing, slice at z-midplane, run 2D
- **3D**: Full volumetric simulation with superellipsoid granules and quaternion orientation

The simulator predicts scaffold microstructure evolution (void network topology,
functional connectivity, tissue formation, transport properties) over experimentally
relevant timescales (24--72 hours).

```
┌──────────────────────────────────────────────────────────────────────┐
│                          new_dem_0.py                                │
│                                                                      │
│  ┌──────────┐   ┌────────────────────┐   ┌────────────────────────┐ │
│  │  Params   │──▶│ generate_packing*()│──▶│    GranuleSystem       │ │
│  │ (config)  │   │  2D / 3D / slice   │   │  (mode-aware state)    │ │
│  └──────────┘   └────────────────────┘   └──────────┬─────────────┘ │
│                                                      │               │
│  ┌───────────────────────────────────────────────────▼────────────┐  │
│  │                        run() loop                              │  │
│  │                                                                │  │
│  │   ┌───────────────────────┐    ┌────────────────────────────┐  │  │
│  │   │ compute_forces*()     │───▶│     step()                 │  │  │
│  │   │  • Hertz contact      │    │  • update_cell_state       │  │  │
│  │   │  • motor-clutch       │    │  • overdamped Euler (2D/3D)│  │  │
│  │   │  • wall (4 or 6 face) │    │  • velocity cap            │  │  │
│  │   │  • friction + torques │    │  • wall clamp              │  │  │
│  │   │  • active noise       │    │  • quaternion integration  │  │  │
│  │   └───────────────────────┘    └────────────────────────────┘  │  │
│  │                                                                │  │
│  │   ┌───────────────────────┐    ┌────────────────────────────┐  │  │
│  │   │ render_fields*()      │───▶│ compute_metrics()          │  │  │
│  │   │  • 2D: tanh profiles  │    │  • connectivity            │  │  │
│  │   │  • 3D: volumetric     │    │  • overlap stats           │  │  │
│  │   │  • eff. radii         │    │  • Kozeny-Carman, Darcy    │  │  │
│  │   │  • bbox clipping (3D) │    │  • RCP, compaction ratio   │  │  │
│  │   └───────────────────────┘    └────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    Built-in Visualisation                      │  │
│  │  plot_granules  plot_fields  plot_timeseries  plot_composite   │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Visualization Scripts (V1.4)                       │
│                                                                      │
│  viz_compaction.py     Void-space evolution, packing, compaction     │
│  viz_percolation.py    Darcy, Kozeny-Carman, dimensionless groups    │
│  viz_movies.py         Rotating GIF, timelapse, z-sweep, composite   │
│  viz_phases.py         Phase isosurfaces, fractions, tri-plane       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. Module Breakdown

### 2.1 Configuration — `Params` (dataclass)

| Group | Parameters | Units | Purpose |
|-------|-----------|-------|---------|
| Mode | `mode` | -- | `"2D"`, `"2D-slice"`, or `"3D"` |
| Domain | `Lx`, `Ly`, `Lz` | um | Simulation box size (Lz used in 3D/2D-slice) |
| Granule sizes | `R_func_mean/std`, `R_inert_mean/std` | um | Gaussian size distributions |
| Composition | `phi_f_target`, `phi_i_target` | -- | Target area/volume fractions |
| Cell geometry | `n_cells_per_granule`, `cell_diameter`, `cell_height_spread`, `cell_coverage` | --, um, um, -- | Cell seeding (20 um spheres, 5 um spread height) |
| Cell timeline | `t_attach_onset`, `t_attach_half`, `t_spread_duration`, `fa_maturation_rate` | h, h, h, 1/h | Attachment/spreading/FA kinetics |
| Motor-clutch | `n_motors`, `F_motor_stall`, `n_clutches`, `k_clutch`, `k_on_clutch`, `k_off_clutch` | --, nN, --, nN/um, 1/s, 1/s | Chan & Odde 2008 force model |
| Cell bridging | `cell_sense_distance`, `F_max_per_cell`, `L_rest` | um, nN, um | Filopodia range, force cap, rest length |
| Contact mechanics | `E_modulus`, `poisson_ratio` | kPa, -- | Hertzian contact (V1.1) |
| Friction | `tau_0_ii/if/ff`, `friction_v_ref` | Pa, µm/h | Area-dependent hydrogel friction (V1.2) |
| Adhesion | `W_adh_ii/if/ff` | J/m² | DMT adhesion by pair type (V1.2) |
| Shape (2D) | `shape_enabled`, `aspect_ratio_func/inert_mean/std`, `blockiness_func/inert_mean/std` | --, --, -- | Superellipse shape (V1.3) |
| Shape (3D) | `aspect_ratio_c_func/inert_mean/std`, `blockiness_n2_func/inert_mean/std` | --, -- | Superellipsoid c-axis ratio and meridional blockiness (V1.4) |
| Drag | `eta`, `drag_scale`, `drag_scale_rot` | Pa-s, --, -- | Overdamped dynamics |
| Noise | `T_active` | nN-um | Active temperature (functional only) |
| Time | `dt`, `t_total`, `save_every_h`, `v_max` | h, h, h, um/h | Integration control |
| Rendering | `Ngrid`, `Ngrid_3d`, `interface_width` | --, --, um | Phase field grid (2D and 3D) |
| Performance | `use_numba` | -- | Optional Numba JIT acceleration |

### 2.2 State Container — `GranuleSystem`

Mode-aware container. Stores all per-granule arrays:

**Common (all modes):**
- **Position**: `x`, `y` (float64); `z` (float64, 3D only)
- **Radius**: `r` (float64, constant throughout simulation)
- **Type**: `gtype` (int, 0=functional, 1=inert)
- **Cell count**: `n_cells` (float64)
- **Cell state** (V1.2): `n_attached`, `spread_fraction`, `fa_maturity`, `n_overcrowded`
- **Velocity state**: `vx`, `vy` (float64); `vz` (float64, 3D only)
- **Shape semi-axes**: `a`, `b` (float64); `c` (float64, 3D only)
- **Blockiness**: `n_shape` (equatorial n1); `n1`, `n2` (3D mode)
- **Bounding radius**: `r_bound = max(a, b)` in 2D, `max(a, b, c)` in 3D
- **Fast-path flag**: `is_circle` (bool)

**2D only:**
- `theta` (float64) — orientation angle (rad)
- `omega` (float64) — angular velocity (rad/h)

**3D only:**
- `quat` (N,4 float64) — unit quaternion (w,x,y,z)
- `omega_3d` (N,3 float64) — angular velocity vector (rad/h)

**Properties:**
- `is_3d` — True if mode is "3D"
- `positions()` — returns (N,2) in 2D or (N,3) in 3D
- `func_mask`, `inert_mask` — boolean arrays

### 2.3 Quaternion Utilities (V1.4)

Pure functions for quaternion math:

| Function | Purpose |
|----------|---------|
| `quat_multiply(q1, q2)` | Hamilton product |
| `quat_conjugate(q)` | Inverse rotation |
| `quat_normalize(q)` | Enforce unit length |
| `quat_rotate(q, v)` | Rotate vector by quaternion |
| `quat_rotate_inv(q, v)` | Inverse rotation (world → body) |
| `quat_to_rotation_matrix(q)` | 3×3 rotation matrix |
| `quat_from_axis_angle(axis, angle)` | Create from axis-angle |
| `quat_random(rng)` | Uniform random orientation |
| `quat_integrate(q, omega, dt)` | Integrate angular velocity |

### 2.4 Superellipsoid Geometry (V1.4)

3D shape functions for `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1`:

| Function | Purpose |
|----------|---------|
| `superellipsoid_volume(a,b,c,n1,n2)` | Exact volume (Jaklic formula) |
| `superellipsoid_point(eta,omega,...)` | Parametric surface point |
| `superellipsoid_normal(eta,omega,...)` | Outward unit normal |
| `superellipsoid_curvature_radii(...)` | Principal curvature radii (for Hertz R_eff) |
| `superellipsoid_implicit(...)` | Inside/outside test in body frame |
| `superellipsoid_implicit_world(...)` | Inside/outside test in world frame (uses quaternion) |
| `superellipsoid_mesh(...)` | Triangle mesh for rendering |

### 2.5 Hertz Contact Mechanics

Three functions implement the contact physics (V1.1+):

#### `hertz_contact_force(E_star_Pa, R_eff_um, delta_um) -> float`
Hertzian normal force: `F = (4/3) E* sqrt(R*) delta^(3/2)`.
Unit conversion factor of 1e-3 maps (Pa, um, um) → nN.

#### `overlap_lens_area(R1, R2, d) -> float`
Exact 2D lens-shaped intersection area between two circles.

#### `compute_effective_radii(gs) -> (r_eff, overlap_area)`
Volume-conserving display radii: `r_eff = sqrt(r² + ΔA/π)`.

### 2.6 Cell Geometry & Motor-Clutch Model (V1.2)

#### `motor_clutch_force(E_kPa, p, fa_maturity)`
Steady-state traction per cell (Chan & Odde 2008):
`F_mc = F_stall * k_sub/(k_sub+k_opt) * engagement * fa_maturity`

#### `update_cell_state(gs, p, t)`
Per-step lifecycle: attachment → spreading → FA maturation → overcrowding

### 2.7 Packing Generators

| Generator | Mode | Description |
|-----------|------|-------------|
| `generate_packing()` | 2D | 2D RSA with superellipse shapes |
| `generate_packing_3d()` | 3D | 3D RSA with superellipsoid shapes, random quaternion orientation |
| `generate_packing_2d_slice()` | 2D-slice | Calls `generate_packing_3d()`, slices at z=Lz/2 via `slice_superellipsoid_z()` |

### 2.8 Contact Detection

**2D (V1.3):** Circle fast-path (`r_i + r_j - d`) or superellipse common normal (Newton-Raphson, 2 unknowns).

**3D (V1.4):**
- `find_contact_spheres_3d()` — analytical sphere-sphere overlap
- `find_contact_superellipsoids_3d()` — Newton-Raphson with 4 unknowns (eta_i, omega_i, eta_j, omega_j). Returns penetration depth, contact normal, contact point, local curvature radii.
- `find_contact_wall_3d()` — 6 wall faces with bounding-sphere sampling

### 2.9 Force Computation

| Force | Scope | Law | Key parameters |
|-------|-------|-----|----------------|
| **Contact (normal)** | All overlapping pairs | Hertz − DMT | `E_modulus`, `poisson_ratio`, `W_adh_*` |
| **Contact (tangential)** | All overlapping pairs | Area-dependent friction | `tau_0_*`, `friction_v_ref` |
| **Contact torque** | Non-spherical granules | τ = (contact_pt − centre) × F | Off-centre contacts |
| **Cell bridging** | Functional pairs with attached cells | Motor-clutch | `n_motors`, `F_motor_stall`, etc. |
| **Wall** | Granules penetrating boundary | Hertz (rigid flat) | 4 walls (2D) or 6 walls (3D) |
| **Active noise** | Functional granules only | Gaussian white noise | `T_active` |

Dispatch: `compute_forces()` → 2D path, `compute_forces_3d()` → 3D path.

### 2.10 Time Integration — `step()`

**2D path:**
```
update_cell_state(gs, p, t)
F, torques = compute_forces(gs, p, rng)
v = F / gamma;  |v| = min(|v|, v_max)
x += v * dt;  x = clamp(x, 4 walls)
theta += omega * dt  (non-circular granules)
```

**3D path:**
```
update_cell_state(gs, p, t)
F, torques = compute_forces_3d(gs, p, rng)
v = F / gamma;  |v| = min(|v|, v_max)
x,y,z += v * dt;  clamp to 6 walls
quat = quat_integrate(quat, omega_3d, dt)
```

### 2.11 Phase Field Rendering

**2D:** `render_fields()` — tanh-profile stamping with max-blending on Ngrid² grid.
**3D:** `render_fields_3d()` — superellipsoid implicit function with tanh profile on Ngrid_3d³ grid. Bounding-box clipping per granule for performance.

### 2.12 Transport Metrics (V1.4)

| Metric | Formula | Added to `compute_metrics()` |
|--------|---------|------------------------------|
| Porosity | `ε = phi_v_mean` | `porosity` |
| Kozeny-Carman permeability | `K = ε³d²/[180(1-ε)²]` | `K_kozeny_carman` |
| RCP fraction | `φ_RCP ≈ 0.64 + 0.08*(AR-1)` | `phi_RCP` |
| Compaction ratio | `φ_solid / φ_RCP` | `compaction_ratio` |
| Darcy number | `Da = K / L²` | `Da_number` |
| Grain diameter | Mean `2*r` | `d_grain_mean` |

### 2.13 Built-in Visualisation Functions

| Function | Output |
|----------|--------|
| `plot_granules()` | Circle/polygon patches at 5 time snapshots |
| `plot_fields()` | 3-row (phi_f, phi_i, phi_v) phase field heatmaps |
| `plot_timeseries()` | 3×3 grid: topology, dynamics, cell state |
| `plot_composite()` | RGB composite (R=functional, G=void, B=inert) |

---

## 3. Data Flow

```
Params ──▶ generate_packing*() ──▶ GranuleSystem
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
      hist (list of dicts)   snaps (list of dicts)
               │                   │
    ┌──────────┼───────────────────┼──────────────┐
    ▼          ▼                   ▼              ▼
viz_compaction  viz_percolation  viz_movies   viz_phases
plot_timeseries plot_granules    plot_fields  plot_composite
```

**Snapshot dict format (V1.4):**
```python
{
    'phi_f': ndarray,          # Phase field (Ngrid² or Ngrid_3d³)
    'phi_i': ndarray,
    'phi_v': ndarray,
    'x': ndarray,              # Granule positions
    'y': ndarray,
    'z': ndarray,              # (3D only)
    'r': ndarray,
    'gtype': ndarray,
    'n_attached': ndarray,     # Cell state
    'spread_fraction': ndarray,
    'fa_maturity': ndarray,
    'n_overcrowded': ndarray,
    'a': ndarray, 'b': ndarray,  # Shape
    'n_shape': ndarray,
    'theta': ndarray,          # (2D only)
    'c': ndarray,              # (3D only)
    'n1': ndarray, 'n2': ndarray,  # (3D only)
    'quat': ndarray,           # (3D only, N×4)
}
```

---

## 4. Visualization Scripts (V1.4)

| Script | Plots | Input |
|--------|-------|-------|
| `viz_compaction.py` | Void fraction, packing fraction, compaction ratio, void cluster size distribution, stacked phase areas | `hist`, `snaps` |
| `viz_percolation.py` | Kozeny-Carman K(t), porosity + RCP, dimensionless groups dashboard (2×3), Darcy flow rate, void connectivity | `hist` |
| `viz_movies.py` | Rotating 3D isosurface GIF, time-lapse compaction, z-sweep cross-section, composite 2×2 | `snaps`, `hist`, `p` |
| `viz_phases.py` | Phase isosurface strip, phase volume fractions, tri-plane evolution (XY/XZ/YZ), interface area vs time | `hist`, `snaps`, `p` |

All scripts: CLI (`-i input_dir`), importable (`run_all()`), PyVista primary with matplotlib fallback.

---

## 5. External Files

### Trial Configuration JSONs (`Trials/`)

Two formats supported (V1.4.1):

**Flat format** (recommended) — keys map directly to `Params` field names:
```json
{
    "_format": "flat",
    "mode": "3D",
    "Lx": 800, "Ly": 800, "Lz": 500,
    "E_modulus": 10.0,
    "t_total": 48.0,
    "shape_enabled": true
}
```

**Legacy format** — nested sections with translated key names:
```json
{
    "mode": "3D",
    "domain": { "side_length_um": 400, "Lz_um": 400 },
    "shape": { "aspect_ratio_c_func_mean": 1.0 },
    "output": { "Ngrid_3d": 80 }
}
```

Missing `"mode"` defaults to `"2D"`. See `Trial15_3D.json` for a flat format example.

### HPC Support

| File | Purpose |
|------|---------|
| `run_hpc_headless.py` | CLI-driven headless runner with `--trial` and `--mode` support |
| `run_all_trials.py` | Batch runner: local (mode 1), SLURM generate (mode 2), SLURM auto-submit (mode 3) |
| `hpc/generate_hpc_scripts.py` | Per-user SLURM script generator and automated HPC setup |
| `hpc/sync_results.py` | Standalone result sync: check job status + rsync from cluster |
| `hpc/Alex.json` | User HPC config (netid, group, cpus, walltime, paths) |

### Post-Processing Scripts (Legacy)

| File | Purpose |
|------|---------|
| `new_dem_visualization.py` | Full-featured post-processor (z-stacks, 3D isosurfaces, GIFs) |
| `new_dem_postprocess.py` | Lightweight post-processor for JSON frames |

---

## 6. Design Constraints and Assumptions

1. **Three modes**: 2D, 2D-slice, and 3D. Default is 2D for backward compatibility.
2. **Overdamped regime**: No inertial terms. Valid for cell-culture timescales
   (hours) in viscous medium.
3. **Rigid granules**: Granule shapes do not deform during simulation. Deformation
   effects captured through Hertz contact force and volume-conserving effective radii.
4. **Hertz validity**: Contact model assumes small overlaps (delta/R < ~10%).
5. **Random by default**: Seed is `None` (random) unless explicitly set. All randomness
   flows through `numpy.random.Generator`.
6. **Performance target**: 500-1000 granules in 3D with optional Numba JIT.
7. **Rendering backends**: PyVista primary for 3D (high quality, off-screen capable),
   matplotlib fallback when PyVista unavailable.
