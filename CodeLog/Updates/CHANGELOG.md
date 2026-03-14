# GELLS-DEM Changelog

All notable changes to the GELLS-DEM simulation engine are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering: `MAJOR.MINOR` where MAJOR tracks breaking API changes and
MINOR tracks feature additions and improvements.

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
- **Summary output**: Final summary now reports contact count, max overlap ratio,
  and area conservation.

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
