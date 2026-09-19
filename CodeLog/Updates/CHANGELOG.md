# GELS Changelog

All notable changes to the GELS simulation engine are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Version numbering: `MAJOR.MINOR` where MAJOR tracks breaking API changes and
MINOR tracks feature additions and improvements.

---

## [V3.4] - in progress (started 2026-09-19)

**Support-function minimum-translation-distance contact solver.** Phase 3 of the
`robotsim/` adoption plan, carried over from V3.3 under its own version because it
changes the default contact physics for non-spherical granules. Plan:
`CodeLog/ClaudesPlan/3.4.md`.

The V3.2 KNOWN BLOCKER is now quantified rather than suspected: at 2 % past first touch,
`find_contact_superellipsoids_3d` detects **25/200 = 12 %** of true contacts and
over-reports penetration by **median 9.8x, p90 43x** on the ones it does find. The cause
is structural -- `delta` is the distance between two common-normal surface points, not a
penetration depth, and the reported normal is `(p_j - p_i)/|.|`, not a surface normal, so
the force is not the gradient of any energy.

The replacement is exact rather than iterative in the shape: the support function of the
anisotropic two-exponent superellipsoid is closed form, a nested dual norm
`h(n) = ||( ||(a n_x, b n_y)||_q1 , c n_z )||_q2` with `q1 = n1/(n1-1)`, `q2 = n2/(n2-1)`,
verified against a 600x1200 brute-force surface maximisation at **4.9e-6** relative
(mesh-limited) and **3.6e-15** at the sphere anchor.

---

## [V3.3] - 2026-09-19

Adopts the machinery worth taking from `robotsim/`, the gitignored sibling repo that
attacks the same physics from the opposite end (explicit surface agents vs. a
contact-level cell kernel). `robotsim/docs/gells-dem-adoption.md` records what robotsim
should borrow from GELLS; this is the reverse direction, which had never been written
down. Plan: `CodeLog/ClaudesPlan/3.3.md`.

V3.3 lands the parts that add measurements and correctness without moving any existing
number: the prerequisite twin-drift fixes, the periodic minimum-image guard, and three
halo-free structural metrics. The contact solver, FIRE relaxation, cell stacking,
convergence detection and the HTML viewer are carried into V3.4 and beyond.

### Phase 0 -- prerequisites

**`cell_capacity_foothold` was applied on one path only.** `kernels/cells.py::_capacity`
had no `foothold` argument while `engine.cell_capacity` and `division.capacity_vector`
both applied one, so the two backends computed different per-granule capacities. Not
latent: `fibroblast_realistic` sets `capacity_foothold=0.25`, and at r = 40 um in 3D the
reference gave **cap = 64** against the kernel's **16**. Capacity sets the overcrowding
and division thresholds, so the backends disagreed on cell fate under the flagship preset.

`container_voxel_mask` moved from `viz2/void_percolation.py` to `gels/engine.py` (the
kernels cannot import from `viz2`), verified mask-identical across box/cylinder x
wall/free x 2D/3D. A dead `hasattr(gs, 'cell_diameter_offset')` branch in
`division._daughter_angles` was removed -- the attribute is never assigned anywhere.

### Cell capacity was returning geometrically impossible numbers

Exposed by the foothold fix above: with both paths finally agreeing, they agreed on a
number that cannot happen. `cells_from_surface_coverage` computed `A_surface / A_cell`,
which tiles a surface at **100 %** -- no packing achieves that. On an R = 40 um granule
with `cell_capacity_foothold = 0.25` it returned **64 cells**, against a hard ceiling of
**58** rigid 20 um discs (hexagonal, 0.9069; random close packing gives 53), and against a
confluent fibroblast monolayer of **20-40** (500-1000 um^2/cell at 1-2e5 cells/cm^2).

Two compounding causes:

1. **No packing-efficiency factor.** Added `engine.PACKING_EFFICIENCY = 0.9069` (the planar
   hexagonal maximum), applied in all three implementations of the rule --
   `engine.cells_from_surface_coverage`, the compiled `kernels/cells.py::_capacity`, and
   `division.capacity_vector`. A geometric constant, not a knob, so it is module-level
   rather than a Params field. Capacity can no longer exceed what discs can occupy.
2. **`fibroblast_realistic` sat exactly on the floor.** `A_rounded/A_spread = h/d = 0.25`
   *exactly* by volume conservation, so `capacity_foothold = 0.25` was the densest value
   the parameter can express and was identical to anything below it. Raised to **0.5**.

Together these give **~700 um^2/cell at every granule size** -- scale-invariant and
mid-band for a confluent fibroblast monolayer:

| granule | R = 20 (D 40) | R = 30 | R = 40 (D 80) | R = 60 |
|---|---|---|---|---|
| before | 16 | 36 | **64** (impossible) | 144 |
| after | 7 | 16 | **29** | 65 |
| um^2/cell | 718 | 707 | 693 | 696 |

The legacy projected-area rule (`cell_coverage`, used when no surface coverage is set) is
untouched: its 0.6 default already acts as a de-facto efficiency factor. Only
`run3d_spheres` of the four reference runs uses the surface-coverage rule, and its seeded
population moved 211 -> 192, exactly the 9.07 % the factor predicts.

Tests pin the ceiling (capacity never exceeds hexagonal packing at any foothold or
radius), the biological band, and three-way agreement between the implementations.

### Phase 1 -- periodic neighbour search was silently dropping contacts

`cKDTree(pos, boxsize=L).query_pairs(r)` returns each unordered pair **at most once**, at
its minimum image, and raises nothing for `r > L/2`. Above that a granule has two images
of a neighbour inside the cutoff and only one is reported, so the interaction is silently
**dropped** -- a missing contact, not a double count, which is why it never presented as
an obviously wrong force. Demonstrated: box L = 10, points at x = 1 and x = 9, cutoff 8
gives one pair where a 9-image enumeration finds two interactions.

`check_min_image` now guards `half_pairs`, the funnel both the linked-cell and ckdtree
backends share. No shipped configuration is affected -- the defaults and the DOE2 periodic
trials sit 8-18x clear -- but three periodic cases in `test_render_metrics_vs_reference`
were over the line (r_bound up to 82 um in a 400 um box, about four granule diameters
across) and were measuring the wrong physics; their granules were halved and the 3D
periodic box raised to 300 um.

**`perf_neighbor_skin` is deprecated, on measurement.** It promised a Verlet list that was
never implemented. Building it was measured rather than assumed, and it loses
(N = 3841, 3D, cutoff 135 um):

| skin (um) | pairs | | build | step |
|---|---|---|---|---|
| 0 | 193,988 | 1.00x | 1.96 ms | **19.1 ms** |
| 5 | 214,676 | 1.11x | 2.41 ms | |
| 10 | 236,695 | 1.22x | 2.60 ms | |
| 20 | 283,851 | 1.46x | 2.98 ms | **33.3 ms** (1.74x) |

The rebuild a skin eliminates is only ~10-14 % of a step (flat from N = 78 to N = 11197),
while the larger list it creates is traversed by the force loop on every step. At the
declared default of 20 um, even free rebuilds would land at ~29 ms against 19 ms today,
and no positive skin breaks even. The cutoff is `2*max_r_bound + L_max` and **L_max (cell
sensing) dominates**, so the candidate list is already ~50 neighbours per granule. The
optimisation the data points at is splitting the short contact-range list from the long
sensing-range one -- noted, not done.

### Phase 2 -- structural metrics that measure structure, not the renderer

Three measures that CLAUDE.md already warned were untrustworthy now have halo-free
counterparts. All are computed by ONE shared module called from both metrics twins, so
they agree by construction rather than by matching arithmetic.

**Percolation on the real contact graph.** `func_lf` thresholds the rendered tanh field at
`mean + 0.3*std` and runs `ndimage.label`: it joins granules separated by up to
2*`interface_width` of nothing, and its value moves with `Ngrid`. The new `gran_*` block
runs union-find on granules that actually touch, using the engine's own `overlap > 0`
predicate (`contact_stats_k` gained an output array; the reference records the same flag),
so there is no second definition of "touching" and no tolerance to tune. On a settled 2D
packing `func_lf = 0.899` against `gran_lf_func = 0.750` **in five separate clusters** --
the field measure both overstates continuity and hides the fragmentation. `func_lf` is
untouched; the disagreement is the deliverable.

Periodic boxes get the Newman-Ziff wrapping test; walled and cylindrical ones get a
floor-to-**bed-surface** spanning test, because nothing reaches a lid that is not there.

**Laguerre (radical) local packing fraction** (`output.metrics_laguerre`). Each granule's
own share of space from an exact partition, weighting every bisector by radius as a
polydisperse pack requires. Measured inflation of the field measure: **1.24x in 3D, 1.38x
in 2D**. More sharply, the field-based `compaction_ratio` reads **1.024 and 1.302** -- denser
than random close packing, which is impossible for a freshly settled bed; `compaction_func`
gives 0.803.

Three things the port needed that robotsim's version does not have:

1. a **verified** neighbour cutoff -- a fixed `6*r_max` gave cells totalling **4.8x the box
   volume**, because the cutoff is unrelated to the local spacing. The cell is now rebuilt
   with a grown cutoff until no granule outside it could have clipped it;
2. a **Chebyshev-centre fallback** -- a granule's centre is not always inside its own power
   cell (when a larger neighbour overlaps it deeply), though the cell is still non-empty;
3. **free-top handling** -- a loose lid bounds the tessellation and any cell touching it is
   excluded, with `laguerre_valid_frac` reported so a mean is never quoted without its
   denominator.

**Katz-Thompson permeability** (`output.metrics_pore_field`). Kozeny-Carman is blind to
channelization. Two fields with identical porosity, one dispersed into narrow pores and one
coarsened into a wide channel:

| | porosity | l_c | Katz-Thompson | Kozeny-Carman |
|---|---|---|---|---|
| dispersed | 0.111 | 1.00 | 0.0020 | 0.0039 |
| coarsened | 0.111 | 10.00 | 0.1967 | 0.0039 |
| **ratio** | 1.0x | | **100x** | **1.0x** |

The signed-distance field was already half-built in both twins and discarded --
`tree.query(voxel_pts)` returns `(distance, index)` and only the index was kept.
`K_kc_analytic` states Kozeny-Carman on the same `eps` and `S_v`, so the only difference
between the two is the channelization term. One correction to the source: the lateral
`np.roll` in the tortuosity BFS is applied only under periodic boundaries -- a wall is not
a mirror, and rolling across one invents paths and deflates the tortuosity.

### Testing -- the V2.7 fixture gate is platform-bound

`test_legacy_identity` failed all four reference runs on macOS **before any V3.3 change**.
The fixtures were blessed on Windows / CPython 3.14 / numpy 2.5; elsewhere packing
positions differ by ~1e-13 (different libm, different numpy reduction order). In the 2D
runs that stays at ~1e-12 forever, but in `run3d_spheres` it flips a discrete
bridge-formation decision at t = 1.0 and the trajectories bifurcate:

| run | t = 0 | t = 1.0 | t = 2.5 |
|---|---|---|---|
| `run2d_walls` | 1.4e-12 | 3.2e-14 | 1.9e-12 |
| `run3d_spheres` | 2.2e-14 | **6.7e-01** | **7.4e-01** |

No tolerance can bridge that, so exact equality of a **run** is only meaningful on the
machine that blessed the fixture. `test_legacy_identity` now skips the run comparison on a
platform-fingerprint mismatch (`params.json` is pure config resolution and is still
checked everywhere), and `tests/test_local_identity.py` applies the same atol = 0
comparisons against a baseline blessed on the current machine via
`make_fixtures.py --local`. Verified sensitive: it catches a 1e-12 relative perturbation.
`--force` on `tests/fixtures/` now also requires `--i-really-mean-it`, since the V2.7 code
is no longer in the tree and those files are the only record of its behaviour.

### Bug Fixes

- `kernels/cells.py::_capacity` ignored `cell_capacity_foothold` (4x capacity divergence
  between backends under `fibroblast_realistic`).
- Periodic neighbour search silently dropped pairs whenever the cutoff reached half the
  shortest box edge.
- `division._daughter_angles` branched on an attribute that is never assigned.
- `test_io_writer` asserted an "unwritable" path that `write_npz_atomic` creates via
  `os.makedirs(exist_ok=True)`; it passed on Windows only because `Z:\` does not exist.

---

## [V3.2] - in progress (started 2026-09-18)

The granules stop being rigid spheres. The lab's are irregular, roughly cuboidal hydrogel
fragments at **10-50 kPa** with surface roughness, not the 3 GPa PMMA beads V3.1 was built
around -- so contacts genuinely flatten under cell traction, and a bed of blocky fragments
can pack past the sphere limit of phi ~ 0.64. The cell population also stops being
synchronised. Plan: `CodeLog/ClaudesPlan/3.2.md`.

Three problems motivated it:

1. rigid spheres cannot compact past close packing, but superellipsoids can -- and the real
   granules are cuboidal with an aspect ratio and surface roughness;
2. the real granules are soft enough to deform at the contact, especially where cell
   traction binds and tightens it;
3. every cell started at cycle phase zero, so divisions would arrive in a burst.

### Irregular cuboidal granules -- a shape-aware packer

The lab's granules are irregular, roughly cuboidal fragments, and blocky superellipsoids jam
well past the sphere limit. **The shape path was not merely missing, it was backwards.** The
packer was bounding-sphere throughout -- `RSAChecker` and `settle_sub_k` used only `r_bound`,
and `p.shape_enabled` was stored in `packing.py` and never read -- while `r_bound =
max(a, b, c)` ignores blockiness and so **does not even bound the body** for n > 2 (measured:
21 % short at AR 1.0 / n 3.5). With semi-axes renormalised to equal volume, an AR-1.8 / n-3.5
fragment's bounding sphere claims **2.39x** its volume. Measured on a 2D gravity-settled bed:

| bed | phi_bed | height |
|---|---|---|
| spheres | 0.747 | 539 um |
| shapes, packer blind (V3.1 behaviour) | **0.522** | 761 um |
| shapes, shape-aware packer | 0.735 | 541 um |

So turning shapes on used to make the bed **41 % taller**, and the shape-aware packer fixes
that regression.

**Retraction.** An earlier draft of this entry claimed the shaped bed packs to 0.884 against
0.839 for spheres. That compared SUMS OF GRANULE AREAS, which double-count overlap, and the
shaped packings carry 4.7-7.4 % interpenetration against 0.8-1.1 % for spheres. Re-measured by
rasterising the actual superellipses -- a pixel is either inside some granule or not, no lens
formula involved -- the true covered fractions are **shapes 0.834 vs spheres 0.829**: a real
but marginal gain in 2D, not the 5 % claimed. Do not quote nominal solid fractions for shaped
beds; use the rasterised value or `phi_bed_net`.

**The packed density is an INPUT, not an output.** With a FIXED granule count in a fixed
container, the achieved fraction tracks `granules.bed_solid_fraction` at 0.97-1.00x from 0.60
to 0.85, while the coordination number rises from 2.5 to 4.1. Granules are placed inside a
height `V_solid/(request * A_base)` and inflated there, so gravity only reseats rattlers and
never finds a sedimented density of its own. `bed_solid_fraction` therefore has to be chosen
physically, and `check_bed_fraction` now warns when it sits outside the random-loose to
random-close band for the dimension (3D 0.55-0.64, 2D 0.78-0.84). The `soft_gel_*` presets
were lowered to 0.52 (3D) and 0.70 (2D) so the bed starts with room to compact.

- **`packing.shape_contact`** (default false) switches on the true bounding radius, the RSA
  correction and a shape-aware settle substep together. They ship behind ONE flag because the
  corrected radius is LARGER, so switching it on alone lowers the bounding-sphere jamming
  ceiling and makes packing worse.
- **The directional radius.** The GELS implicit function is positively homogeneous of degree
  n2 for ANY (n1, n2) -- the inner `^n1` is immediately raised to `n2/n1` -- so the distance
  from centre to surface along a body direction is closed form, `lambda(u) = F(u)^(-1/n2)`.
  Verified to 1e-11 for five (n1, n2) pairs including n1 != n2, so no Newton fallback is
  needed. The pair gap is `g = lambda_i(u) + lambda_j(-u) - d` and both force and torque are
  `-dE/dq` of that same scalar with `E = (2/5) k_rep g^{5/2}`, so the settle is gradient
  descent on a potential.
- **Two things that had to be right.** `lambda` is homogeneous of degree **-1**, so
  `grad(lambda).u = -lambda`, NOT zero: the radial part must be projected out. Verified
  against finite differences -- with the projection the analytic gradient matches
  `d(gap)/dx` to 2e-10; **without it the error is 1.37, a factor of ~2.5 and a force that is
  not the gradient of anything.** And a purely radial force produces no torque at all (the
  arm is parallel to the force), so without the tangential tilt nothing rotates and the whole
  density gain disappears.
- **Rotation in the settle**, overdamped with the second moment of the shape standing in for
  inertia, capped at 0.15 rad (2D) / 0.12 rad (3D) per substep. Those caps are module
  constants beside `dt_settle` and `k_rep`, not Params fields. The stop rule now also waits on
  the largest rotation, so the settle cannot exit while the bed is still re-orienting.
- **`contact.shape_dynamics`** makes the overlap projection, the wall clamp and the bed
  surface use the directional radius. Required: at a true phi 0.68 the BOUNDING-SPHERE
  fraction exceeds 1, so every contact reads as a deep overlap and the projection pulls the
  packing apart -- measured **7.25 % bed expansion in 6 h with no cells**, and 37 % of pairs
  clipped every step. With the shape-aware path that falls to 4.6 % and 16 %.
- **`contact.curvature_R_cap`** (recommended 2.0). The local curvature radius of a
  superellipse runs away at a flat face: measured **3.5e15 um at n = 3.5** against 48 um for
  a circle, and since `F ~ sqrt(R_eff)` that is a contact ~1e6 times too stiff.
- **`packing.shape_margin` is exposed and documented as blunt.** Measured: raising it does not
  reduce the residual relaxation and it breaks post-relax convergence (32 um residual overlap
  against a 1 um tolerance, hitting the 3000-step cap). Default 0 is correct.

### KNOWN BLOCKER: the superellipse contact solver is wrong (pre-existing, V1.3)

**A shaped bed is not yet usable for compaction studies.** From a loose start (request 0.70)
it EXPANDS 12.3 % over 72 h with no cells at all, true phi falling 0.693 -> 0.586; with the
full cell model it still expands 3.3 %. Spheres under the same conditions are stable to 0.03 %.

The cause is not the settle. Measured against the exact minimum translation distance (from the
support function, which is closed form for these shapes), the settle's directional-radius gap
is accurate -- at n = 3.5 it gives 0.783 / 1.958 / 3.917 um against a true 0.804 / 1.976 /
3.930, within 3 %. It is `se2d_contact_k`, the common-normal solver the DYNAMICS uses, that is
broken:

| n = 3.5, true overlap | settle proxy | `se2d_contact_k` |
|---|---|---|
| 0.804 um | 0.783 um | **reports no contact** |
| 1.976 um | 1.958 um | **reports no contact** |
| 3.930 um | 3.917 um | **reports no contact** |

| n = 2.0, true overlap | settle proxy | `se2d_contact_k` |
|---|---|---|
| 0.731 um | 0.802 um | **34.74 um** (47x) |
| 1.527 um | 1.527 um | **23.83 um** (16x) |
| 4.708 um | 4.010 um | **reports no contact** |

So it misses contacts entirely for blocky granules, and where it does fire it returns an
"overlap" larger than the granule itself -- because `delta` is the distance between the two
common-normal surface points, which is not a penetration depth. Detection is also
non-monotonic in overlap. On a packed bed the mean reported overlap is **21.9 um where the
settle sees 0.025 um**, a factor of 889; the p95 is 43 um on a 20 um granule. Those phantom
overlaps are what blow the bed apart, and they explain why the curvature cap barely helped.

This has been wrong since shapes were introduced in V1.3. It is invisible to
`tests/test_legacy_identity.py`, which freezes the fixture `run2d_shapes` bit-for-bit
regardless of whether the physics is right, and to `tests/test_forces_vs_reference.py`, which
only checks that the kernel and the reference agree -- they agree because they share the
solver.

**The fix** is to replace the common-normal `delta` with the support-function MTD, which is
exact, closed form for n1 == n2, and yields the normal and contact points as well. It is
specified in `CodeLog/ClaudesPlan/3.2.md`. Until it lands, do not draw physical conclusions
from any run with `shape_enabled` and n > 2.

### Bug Fixes (packing)

- **`r_bound` did not bound a blocky granule.** `max(a, b, c)` is the circumscribed radius
  only at n = 2; for n > 2 the farthest surface point is toward a corner. Measured 21 % short
  at AR 1.0 / n 3.5. Replaced by `shape_bound_radius`, exact for n1 == n2 via the interior
  stationary point of `r^2 = sum s_k^2 u_k^(2/n)` and a golden section otherwise. Only active
  under `packing.shape_contact`, because the correct (larger) bound must ship with the
  directional overlap or it makes packing worse.
- **RSA silently dropped granules.** `if not placed: pass`. With the corrected bound a 3D
  shaped packing lost **20 % of its granules (415 requested, 333 placed)** without a word.
  Fixed by `shape_bound_factor`, which deflates the RSA placement by the bounding-sphere
  volume a shaped granule actually reserves (no RNG draws, so the packing stream is
  untouched) -- all 415 are now placed. The drop is also counted and warned about. It is a
  warning rather than an error because small dense test systems legitimately lose a few
  granules to RSA.

### Soft granules that deform

At 10 kPa a 180 nN bridge equilibrates a contact at delta = 2.85 um (r = 40) or 3.59 um
(r = 20); at 3 GPa it equilibrates at **0.001 um**. The Hertz/JKR law already produces that
deformation -- V3.1 removed it twice over, by choosing PMMA and by pinning
`max_overlap_frac` at 0.03 so the geometric rail bound long before the contact law did.
**Measured on a 2D dish under identical cell traction from an identical packing**, the
interpenetrated volume grows from 0.048 % of the solid to:

| granule | lens fraction at 24 h | growth | mean bridge force |
|---|---|---|---|
| rigid 3 GPa (the V3.1 configuration) | 0.343 % | 7.1x | 174 nN |
| soft 50 kPa | 0.641 % | 13.3x | 147 nN |
| **soft 10 kPa** | **3.775 %** | **78.5x** | 92 nN |

Soft granules deform about 11x more than rigid ones under the same traction. The bridge
forces fall in the right order for a Hill element too: it holds near stall on a rigid
substrate and sheds force as it shortens faster on a soft one.

- **`contact.overlap_model: fixed | elastic`** (default `fixed` = V3.1). `elastic` sizes
  `max_overlap_frac` from the contact law -- `safety * delta_eq(expected_force_nN) / r` on
  the softest, smallest species, clamped to [0.05, 0.30] -- and bakes it into `params.json`
  so it is traceable. `max_overlap_frac` returns to its intended role as a numerical rail
  against explicit-step overshoot rather than the thing that decides the compaction: the
  volumetric ceiling it imposes is `(2/(2-f))^3`, which is **1.046 at 0.03 and 1.263 at 0.15**.
- **`contact.semi_implicit`** (default false) damps each step by the local contact stiffness,
  `vel = F / (gamma + dt * sum_contacts 2 E_s a)`. Both quantities are already on the contact
  record, so it costs one bincount and no kernel change. It is unconditionally stable in the
  diagonal part and **leaves the fixed point untouched** (at equilibrium F = 0, so the step is
  zero for any damping), i.e. it changes the transient, not the physics. Measured effect on a
  2D dish: the fraction of granules pinned at the velocity cap fell from **0.86-1.00 to
  0.000-0.004**, and spurious bed relaxation from -8.1 % to -2.2 %. Explicit overdamped Euler
  is stable only for `k < gamma/dt` = 2 nN/um at r = 20 um, and at 10 kPa that bound is
  crossed at delta ~ 0.01 um -- i.e. at every load-bearing contact -- so before this the bed
  was held by the velocity cap and the overlap projection, neither of which is a contact model.
- **The rails are now observable.** New metrics `n_overlap_clipped`, `overlap_clip_fraction`
  and `frac_velocity_clipped`. A run in which the projection fires on a large fraction of
  contacts is reporting the rail, not Hertz/JKR, and there was previously no way to tell. This
  immediately caught a real artefact: **with no cells at all the packed bed still expanded
  8.1 % over 12 h**, because the packer's settle force law (`k_rep * ov^1.5` in settle units)
  is not the JKR law the dynamics uses, so a packed bed is not a mechanical equilibrium of the
  dynamics. The semi-implicit step reduces it to 2.2 %; the remainder is recorded as a known
  limitation rather than hidden.
- **Honest volume bookkeeping.** `phi_solid_true` and `phi_bed` sum `(4/3) pi r^3` over
  nominal radii and double-count every contact lens -- 0.1-0.2 % at the 3 % overlaps of a
  rigid bed, but 1.1-4.1 % at the 10-15 % overlaps a soft bed reaches, which is exactly this
  regime. New keys `phi_bed_net`, `bed_solid_volume_net`, `phi_solid_true_net`,
  `overlap_volume` and `overlap_volume_fraction`, from a vectorised
  `overlap_solid_volume` that agrees exactly with the existing scalar `overlap_lens_area` /
  `overlap_lens_volume`. The existing keys are untouched, so `test_legacy_identity` is
  unaffected.
- **Presets `soft_gel_well` and `soft_gel_dish_slice`** are now the primary system: 10 kPa,
  nu 0.45, no stiffness cap, elastic rail, semi-implicit step, MC-DEM off, Coulomb mu 0.3,
  the V3.0 hydrogel adhesion energies restored, density 1050 kg/m3 and a looser assumed bed
  (0.52 in 3D, 0.75 in 2D). `pmma_well` / `pmma_dish_slice` are unchanged and re-described as
  the rigid-bead control.
- **Cells binding and tightening the contact.**
  `cells.bridging.contact_adhesion_nN_per_cell` (default 0, preset 20 nN) adds a mature
  bridging cell's grip to the SHEAR cap of the contact it spans:
  `F_t_cap += w_cell * sum(maturity)`. Only the tangential direction is new, and
  deliberately so -- the normal direction already has the bridge pulling the pair together
  with up to the stall force, and the contact it deepens already grows its own shear cap
  through the contact radius, so adding a normal hold would double-count the same pull. What
  was missing is the collar's resistance to SLIDING, which is what locks a compacted
  structure in place. Measured on the 2D dish at 10 kPa: bed height at 24 h 542.3 -> 533.0 um
  and interpenetration 3.78 % -> 4.16 % at 20 nN per cell.
  Read from the persistent per-cell arrays rather than from this step's bridging pass,
  because the contact pass runs first -- so it is the bridges committed before the step,
  which is the physically right set. One extra kernel argument, zero cost when disabled.

- **MC-DEM stays off** for soft-granule compaction and becomes an explicit sweep axis. It is
  physically motivated for soft gels -- that is its origin -- but it opposes compaction,
  stiffens the contact up to 5x (worsening the stability picture by the same factor), and
  `c_mc = nu/(1-2nu)` is 4.5 at nu = 0.45 and **24.5 at nu = 0.49**, so nu must be reported
  alongside E whenever it is on.
- **LS-DEM is not used, and why is recorded.** `deformable_enabled` looks like the obvious
  tool for deformable granules and is not: it forces the entire run onto the Python reference
  path, its mode shapes are global affine strains about a body-fixed axis so a granule cannot
  flatten at a contact (and for spheres rotation is never integrated, so that axis can never
  align with a contact normal), wall contacts accumulate no deformation at all -- fatal for a
  gravity-settled bed -- it bypasses the species tables for the global `p.E_modulus`, its
  linear node penalty is ~7x stiffer than Hertz at 10 kPa, and it has no tests or benchmarks.

### Asynchronous cell cycle

A seeded population is not in phase. V3.1 put every cell at cycle phase zero, so once the
`cell_age` clock was fixed the whole bed cleared the division refractory period on the same
step and divided in a single burst. Measured on a 160-cell test bed: **160 divisions in one
step at t = 23.5 h and none otherwise**, against 2-10 per step spread through the run after
this change.

- **`cells.division.model: poisson | cycle`** (default `poisson` = V3.1). Under `cycle`
  each cell carries its own cycle length `cell_cycle_time`, drawn lognormal with **median**
  `doubling_time_h` and CV `cells.division.cycle_time_cv` (0.15 typical), and divides when
  `cell_age` reaches it. The memoryless rule it replaces has exponential cycle times, i.e.
  **CV = 1** -- far wider than the 0.1-0.3 real fibroblasts show.
- **`cycle` realises the doubling time it states; `poisson` does not.** Measured over 48 h
  with `min_age_h = 8`: cycle grows **7.97x** against an ideal 2^3 = 8 and is insensitive to
  the refractory period, while poisson grows 4.17x, because a memoryless rule gated by an 8 h
  refractory has a mean cycle of `8 + T_d/ln2` = 42.6 h, not 24 h. Under `poisson`,
  `doubling_time_h` is a rate constant, not a realised doubling time.
- **Random starting phase.** With division enabled, `_initialize_cells` draws
  `cell_age ~ U(0, cell_cycle_time)` per cell. The draws sit **after** the per-granule loop,
  which is the last consumer of the packing RNG stream, so the V2.7 fixtures stay
  bit-identical; a draw inside the loop would shift every later granule's surface angles.
- **Contact inhibition is G0 arrest** under `cycle`: a cell whose granule is at capacity does
  not advance its clock, so a confluent population holds its phase and resumes staggered
  instead of bursting when room appears.
- **New per-cell array `cell_cycle_time`** in the `CELL_ARRAYS` registry, so `add_cells` and
  the snapshot round trip carry it automatically.
- **`cells.seeding.clock_jitter_h`** (default 0) gives each granule an offset on the
  attachment / spreading / focal-adhesion clock. Those are pure functions of global
  simulation time, so the whole bed matured in lockstep: measured `spread_fraction` standard
  deviation across granules **0 without the jitter, 0.24 with it at 3 h**. Applied as
  `t_i = t + gs.cell_clock_offset[i]` in `aggregates_k` and its reference twin, which is
  exactly `t` when the jitter is zero.
- **Diagnostics**: `cell_age_mean`, `cell_age_cv` and `cell_cycle_time_mean` in both metrics
  twins, from one shared `gels.division.cycle_metrics` so they cannot disagree.
  `cell_age_cv` is the instrument for this problem -- 0 for a synchronised population, about
  1/sqrt(3) = 0.577 for one drawn uniformly over the cycle. `viz2/compare_runs.py` gains a
  **Division rate** panel (differentiated from `n_divisions_cum` at plot time, so it needs no
  engine state and works on older runs) and a **Cell-cycle asynchrony** panel.
- **A granule must be big enough to hold more than one cell before it can divide.** In 2D
  the capacity is `floor(pi r^2 * coverage / A_cell)` and a spread fibroblast's footprint is
  ~1257 um^2, which is exactly the projected area of an r = 20 um granule -- so **capacity is
  1 cell** there and the bed is confluent by construction, whatever the seeding coverage.
  Measured: r = 20 -> 1 cell, r = 40 -> 4, r = 60 -> 9. The 40 um 2D dish therefore cannot
  show division at `max_layers = 1`; the 3D well at 80 um granules has room for 16.
  `fibroblast_realistic` now also lowers `cells.seeding.surface_coverage` to 0.8, because
  V3.1 set `capacity_coverage = 1.0` while leaving seeding at the template's 1.0 -- seeding
  exactly at capacity, so every cell arrested in G0 immediately.

- **Preset `asynchronous_cells`** (cycle model, CV 0.15, 2 h clock jitter) so the bundle can be
  applied on its own. `fibroblast_realistic` now uses the cycle model as well, because that is
  what makes its stated 24 h doubling time the realised one; `hydrogel_box_legacy` pins the
  V3.1 values explicitly.

### Bug Fixes

- **The parent's post-division reset landed on the wrong cell.** `GranuleSystem.add_cells`
  rebuilds the CSR layout, shifting every absolute cell index above a birth, but
  `divide_cells` computed `parents` before the rebuild and reset `cell_age` /
  `cell_bridge_age` after it. Demonstrated directly: with births on granules 0 and 2, the
  parent at old index 4 moves to index 5, and V3.1 zeroed index 4 instead. Harmless under
  the memoryless rule, fatal under the cycle model -- a parent that never resets divides
  again on every step, and the test population ran away to 250 824 cells instead of 1 914.
  Fixed by resetting before `add_cells`, so the rebuild carries the new values through.

- **V3.1 cell division never fired.** `gs.cell_age` was allocated
  (`gels/engine.py` `CELL_ARRAYS`), read by `divide_cells`, reset on division and
  serialised -- but **nothing ever incremented it**. The compiled kernel's parameter of the
  same name (`gels/kernels/cells.py` `cells_update_k`) is bound to `gs.cell_bridge_age`, a
  different array, and that collision hid the omission. With the shipped default
  `cells.division.min_age_h = 8.0` -- the value the `fibroblast_realistic` preset uses --
  `cell_age >= min_age` was false for ever, so **no cell divided in any real run**. Every
  division test passed `min_age = 0.0`, so the suite stayed green, and
  `test_min_age_delays_division` passed vacuously.

  Fixed by `gels.division.age_cells(gs, p)`, called from both `update_cell_state` twins
  immediately before `divide_cells` so the backends cannot drift, and by renaming the
  colliding kernel parameter to `cell_bridge_age` (positional, so no behaviour change).
  Measured on the test system at the default `min_age_h = 8.0` over 48 h: **0 divisions
  before, 5 after**. `test_min_age_delays_division` now asserts both halves,
  `test_default_min_age_still_divides` is the regression, and
  `test_age_cells_advances_live_cells_only` covers the clock itself.

  Ageing runs unconditionally -- it is a clock -- which is safe for the bit-identity gate
  because `cell_age` is a V3.1 key that the V2.7 fixtures do not carry and
  `test_legacy_identity` iterates the fixture's keys.

---

## [V3.1] - in progress (started 2026-09-18)

The container, the granule material and the cells all move closer to the lab's experiment:
fibroblasts on collagen-coated **PMMA** granules sedimented under gravity in a **well**
(flat floor, cylindrical side wall, free top), contracting against a force balance and
**dividing**. Plan: `CodeLog/ClaudesPlan/3.1.md`.

Three problems motivated it, all found while reading the V3.0 sweep output:

1. the 3D packings were consolidated into a ball with empty corners (see Bug Fixes);
2. nothing in the model could produce compaction — a closed box is fixed-volume, the bridge
   was a constant-force actuator that never sensed a force balance, and there was no
   gravity, no free surface and no way to represent a rigid granule;
3. the cell population was fixed at seeding.

### Container geometry
- **`boundary.shape: box | cylinder`** (3D; axis z, R = min(Lx,Ly)/2 about the centre) and
  **`boundary.top: wall | free`** (z is up in 3D; y is up in 2D, so a 2D run becomes a
  "dish slice" with a floor, two side walls and a free surface). Both require
  `boundary.mode: walls`; the defaults (`box`, `wall`) are the V3.0 six-face box.
- Sphere–cylinder contact is radial JKR against the existing wall tables; superellipsoids
  use a new arbitrary-plane routine (`wall3d_plane_k` / `find_contact_wall_plane_3d`) on the
  tangent plane at the granule's azimuth, of which the axis walls are the special case.
  The concave composite radius is approximated by the granule radius, which under-estimates
  the contact stiffness by 1–4 % over the useful size range.
- `apply_position_bounds` replaces the four `np.clip` blocks in `step()`: radial projection
  for a cylinder, a floor, and for a free top a clamp at the container height that is
  **counted** (`n_top_clamped`) rather than silently applied — the neighbour and render
  grids need positions inside the box, so `Lz` is the container height, about 1.4x the bed.
- `domain_volume` / `domain_base_area` are shape-aware and are now used by `species_counts`,
  the packing report and the metric volumes. For a box they are the V3.0 expressions verbatim.
- **`granules.bed_height_um`** states the granule amount as a settled bed height instead of a
  container solid fraction (with `granules.bed_solid_fraction`, the packing fraction assumed
  for that bed). Verified end to end: a 1500 µm bed request settles to 1499 µm at φ_bed 0.802
  against the 0.80 assumed.

### Gravity and buoyancy
- Buoyant weight `W = (ρ_granule − ρ_medium) g V` in nN on every mobile granule, along −z
  (3D) or −y (2D), added at the four force-assembly sites. PMMA in water (Δρ = 180 kg/m³) is
  0.059 nN at r = 20 µm against 50–180 nN of cell traction, so in the dynamics gravity is a
  small bias that keeps the bed on the floor; the sedimentation itself happens in the packer.
- `gravity.{enabled, g_m_per_s2, medium_density_kg_m3, granule_density_kg_m3, scale}` and a
  per-species `density_kg_m3`. Off by default.

### Packing consolidation — replaces the centripetal settle
- **`packing.consolidation: centre | gravity | none | auto`** (Params default `centre` =
  the V2.7 behaviour, bit-identical; template `auto`). `none` fills a closed box uniformly
  to the corners; `gravity` sediments the granules into a bed with a free surface.
- The granules are inflated *without* gravity (a body force on deflated granules drops them
  into a dense coplanar layer that then stays laterally jammed); the body force acts during
  the post-relax, holding while the bed still moves and then unloading geometrically.
- The force is bounded by what the settle can resolve — the deepest contact of a column
  carries half the overlap tolerance at full load — not by the granule size. Measured on a
  400 × 400 × 600 µm free-top box and cylinder: between 0.3 and 1.0 of the tolerance the
  settled bed is the same to within 1 µm of height and 0.001 of φ_bed, because the inflation
  does the compaction and the body force only reseats rattlers and flattens the surface.
  The first version of this schedule used ~0.5 mean radius instead of ~0.5 tolerance, a 67×
  larger force, and left residual overlaps at 1.5–2 µm.
- Compiled and reference settles remain bit-identical for every mode
  (`tests/test_packing_vs_reference.py` covers gravity + cylinder + free top in 2D and 3D).

### PMMA material
- **`contact.stiffness_cap_kPa`** caps the modulus entering the pair and wall E* **only**, so
  a granule can be declared at its true 3 GPa (which the cells' motor-clutch sees, saturating
  traction at 0.909 × stall) while the explicit integrator keeps a bounded contact force.
- **`contact.friction_mu`** adds a Coulomb term `µ·F_n` to the tangential capacity. Additive,
  not a `min`: for a rigid granule the JKR contact area is tiny, so τ₀·A is ~1e-3 nN and a
  `min` would switch friction off entirely. `µ = 0` skips the branch and is bit-identical.
- State plainly: with an explicit overdamped integrator (γ = drag_scale·r, dt = 0.5 h)
  contact stiffness above ~4 nN/µm is unstable, so rigidity is enforced **geometrically** by
  `_resolve_overlaps` plus the velocity cap, not by resolved Hertz forces. Forces reported on
  a loaded contact are overshoots, not equilibrium values.

### Boundary functionalization
- **`boundary.functionalization`** (f_wall) mixes the smooth wall's JKR adhesion with
  `mix_pair(f_i, f_wall, ...)`; f_wall = 0 reduces to the V3.0 bare wall exactly.
- **`boundary.layer`** lines the floor and side wall with immobile granules of that coverage,
  so cells bridge to the container through the ordinary bridging machinery — which is what
  distinguishes the two regimes the lab sees: against an inert boundary the bed detaches and
  compacts, against a functionalized one it stays pinned and the functional phase coarsens.
  Immobile granules (`gs.fixed`) are exempt from integration, gravity, the position clamp and
  the packing inflation, and their mobile partner takes the whole overlap correction.

### Contractile bridge
- **`cells.bridging.force_model: constant | hill`**. The constant model is the V2.7 actuator:
  force independent of gap, strain, velocity and load, so the cell never senses that the bed
  has stopped yielding. The Hill model makes each bridge an active element with a stall force
  and an unloaded shortening speed `contraction_speed_um_per_h` (v₀).
- The law is solved per pair rather than evaluated explicitly, because the bridges' own pull
  is most of the closing speed: `fac = (1 − v_env/v₀)/(1 + c·F_iso/v₀)` with
  `v_env = v_rel − c·F_prev`. A rigid environment gives fac = 1 (the cell holds its stall
  force — tensional homeostasis); a free pair closes at `v₀x/(1+x)` as a stable fixed point;
  a stretched bridge gets up to `1 + eccentric_gain`; and v₀ → ∞ gives exactly 1.0, the
  constant model. An explicit `1 − v/v₀` limit-cycles at these parameters (loop gain ≈ 15).
- Two per-cell records (`cell_bridge_gap_prev`, `cell_bridge_force`) carry the state between
  steps and are written by both models, so a run can switch model on resume.

### Cell division
- **`cells.division.{enabled, doubling_time_h, min_age_h, max_layers, divide_while_bridging}`**.
  Per eligible cell per step, `p = (1 − exp(−ln2·dt/T_d))·max(0, 1 − n/(capacity·max_layers))`:
  exponential growth at the population doubling time while there is room, stopping at
  confluence. A cell whose host granule is full colonises the granule it bridges to.
- **`cells.seeding.capacity_coverage`** separates capacity from seeding coverage. Without it
  a granule seeded at its monolayer density is already confluent and can never gain a cell,
  which is why division needs it (0 keeps the V3.0 meaning).
- `GranuleSystem.add_cells` rebuilds the CSR cell arrays: existing cells keep their slot
  within their granule's range and daughters are appended to it, so anything indexing a cell
  by rank keeps its meaning and `cell_bridge_target` (a granule index) is invariant. New
  arrays `cell_age`, `cell_generation` and the counter `n_divisions_cum`; all per-cell arrays
  now come from one `CELL_ARRAYS` registry used by the constructor, `add_cells` and the
  snapshot round trip.
- The pass runs once per step from both `update_cell_state` twins, after the aggregates and
  before the per-cell state machine — the one point in a step where nothing holds a live view
  into the cell arrays. The two backends are statistically equivalent rather than
  bit-identical, as for every other stochastic cell rule.
- The live viewer follows a growing population: the frame builder refreshes its static cell
  table when the count changes and the renderer adopts it.

### Bed observables
- New metric keys, all positional (from centres and radii, so free of the rendering halo and
  of pairwise-overlap double counting): `bed_height_mean`, `bed_height_p95`,
  `bed_envelope_volume`, `phi_bed`, `bed_radius_p95`, `wall_contact_fraction`, `n_floating`,
  `n_top_clamped`, `n_boundary`, plus `n_cells_total`, `n_live_cells`, `n_divisions_cum`.
  Emitted only for open or cylindrical containers, where a bed surface exists at all.
- `phi_z_profile` / `phi_z_edges` in the snapshot npz: solid fraction against height from
  exact sphere-slab volumes.
- **Which numbers to trust against a measurement**: the bed keys, `n_granules_inner` and the
  analytic `*_true` fractions. The field-based `phi_solid`, `porosity` and `compaction_ratio`
  are inflated by the tanh interface halo (≈ 1.5× at R = 20 µm with `interface_width` 3) and
  by overlap double counting — at t = 0 the sweep3d runs report `phi_solid` 0.725 for an
  analytic 0.55.

### Presets and the setup surface
- New `gels/presets.py`: `pmma_well`, `pmma_dish_slice`, `functionalized_wall`, `inert_wall`,
  `fibroblast_realistic`, `hydrogel_box_legacy`. A preset is an ordered list of dotted
  overrides, because the interesting configurations are correlated bundles.
- `step0_new_setup.py --preset NAME[,NAME]` / `--list-presets`, and `step1_config.py
  --preset` applied after the setup file and before `--set`. Names are recorded in
  `meta.presets`, and a written setup carries a header listing every value a preset changed
  (PyYAML cannot emit per-key comments; those live in `TEMPLATE_YAML`).

### Well calibration sweep and comparison plots
- `run_showcase.py --sweep well`: ten conditions varying one thing at a time — container
  (closed box control vs dish vs well), force model, division and wall chemistry — so each
  can be calibrated against the measured bed height separately. The 2D dish slices are
  ~1 900 granules and run in about a minute; the 3D well at 80 µm granules is ~10 000.
- `viz2/compare_runs.py`: bed-height (absolute and relative), `phi_bed`, bed radius, wall
  contact, cell population and cumulative divisions panels; new summary columns; a
  `compare_bed_profiles.png` vertical section through the container axis; and
  `--experiment CSV` (`t_h, bed_height_um[, bed_height_sd_um][, bed_diameter_um]`) to overlay
  measured data. The section works on closed boxes too, which is how the packing artefact
  below is visible at a glance.

### Expected physics — what this can and cannot show
Rigid monodisperse spheres cannot pack past φ ≈ 0.64, and a sedimented bed starts at
0.55–0.60, so densification alone lowers a bed by at most ≈ 7–15 %. What the new setup adds
is a free surface that can actually descend, a bed that can pull away from an inert wall
(`wall_contact_fraction`, `bed_radius_p95`), cluster coalescence, and the vertical profile.
Compaction well beyond that bound cannot come from rearrangement of rigid spheres: it needs
granule loss, a much looser initial bed (testable by lowering `bed_solid_fraction`) or
cell/ECM volume, which this model does not represent.

### Bug Fixes
- **3D packings were consolidated into a ball, leaving the corners of the box empty.** The
  settle applied a centripetal attraction toward the domain centre for the whole inflation —
  `attract = max(0.3(1−t), 0.02)`, magnitude `attract·r_i`, never switched off — while the
  walls were a one-sided clamp with nothing pushing material back out
  (`gels/kernels/packing.py`, `settle_sub_k`; reference twins in `gels/kernels/reference.py`).
  For 20 µm-radius granules that is up to 361 µm of inward drift.

  **Measured on `results/sweep3d`** (2 mm cube, 40 µm granules): the bed ends as a ball of
  radius ≈ 1400 µm; all eight 200 µm corner cubes hold 0–1 granules against 127 expected;
  the interior is 1.31× over-dense (φ_inner 0.725, above random close packing); every 40 µm
  run printed `WARNING: post-relax hit 1000 steps, max_overlap=4.0 µm`; `compaction_ratio`
  starts at 1.13; and `phi_solid` **falls** 0.725 → 0.670 over the run as the bed relaxes
  outward into the corner voids — which `disp_func` (a whole-domain average with no boundary
  exclusion) recorded as "compaction". RSA itself was never at fault: target and placed
  counts match in every run.

  **Resolution**: `packing.consolidation`, with `centre` kept as the Params default so the
  V2.7/V3.0 fixtures stay bit-identical, `none` for a closed box and `gravity` for an open
  one. Part of the reported compaction in `results/sweep3d` is relaxation of this artefact;
  those runs should not be re-read as cell-driven compaction without re-packing.
- **`compare_runs` normalised the bridging-cell fraction by the seeded count**, which exceeds
  1 once cells divide. It now divides by the live count per history entry when available.
- **Void percolation counted the space outside the container as pore space.** With a
  cylindrical wall or a free top, everything outside the wall and above the bed is void, so
  it labels as one cluster touching every face: percolation is trivially true and the
  cluster-size distribution is dominated by it. `viz2/void_percolation.py` now masks to the
  container interior and below the free surface (`container_mask`), and measures the span
  between the first and last planes that actually hold container voxels. A closed box gets
  no mask at all, so its output is unchanged.

## [V3.0] - in progress (started 2026-09-16)

Reframing from binary functional/inert granules to fibroblasts on Collagen-I-coated
hydrogel granules with a continuous degree of functionalization f ∈ [0,1], named granule
species, a sectioned YAML setup file, a real-time viewer, and a numba-parallel engine.
Plan: `~/.claude/plans/lets-update-all-of-lexical-quail.md`. Phases land incrementally;
each is recorded here as it completes.

### Phase 0 — Safety net & scaffold
- **`tests/`** (unittest, no pytest dependency): `make_fixtures.py` generated V2.7 reference
  fixtures on the pre-refactor code — `params.json` as produced by `step1_config.py --trial` for
  `DOE2_2D_0001` and `default_trial`, and four 5-step reference runs (2D walls, 2D periodic,
  2D superellipses, 3D spheres with surface-coverage seeding) with every snapshot and field
  grid. `test_legacy_identity.py` re-runs them on the current code and requires bit-identical
  results — the hard gate for the Python-path phases.
- **`gels/kernels/`** package skeleton: `HAS_NUMBA`/`njit`/`prange` fallbacks and
  `configure_threads(n, layer)` (threading layer + thread count; `omp` verified to scale 35× at
  40 threads on the dual-Xeon workstation).
- **`gels/kernels/reference.py`**: the 21 per-pair / per-cell / per-voxel Python loop functions
  (`compute_forces`, `compute_forces_3d`, `update_cell_state`, `_update_individual_cells`,
  `_service_committed_bridges`, `_classify_bridge_path`, `_attempt_new_bridges`,
  `mc_dem_correction`, `_resolve_overlaps`, the five `_stamp_*` renderers,
  `_periodic_image_offsets_2d`, `render_fields`, `render_fields_3d`, `connectivity`,
  `compute_metrics`, `_settle_packing_2d/3d` — 2,659 lines) moved **verbatim** out of
  `gels/engine.py` (5,794 → 3,199 lines). The engine keeps same-named dispatcher wrappers with
  identical signatures, so every call site and import is unchanged; the reference module is the
  oracle for the compiled kernels and the path LS-DEM keeps running on. Verified bit-identical
  against the fixtures.
- **`GranuleSystem` storage**: positions, velocities and unwrapped positions are now C-contiguous
  `(N,3)` arrays `gs.pos`, `gs.vel`, `gs.pos_unwrap` (the kernel-facing layout). `gs.x/y/z`,
  `gs.vx/vy/vz`, `gs.x_unwrap/...` are write-through column properties: reads return views,
  `gs.x[:] = ...` writes through, and `gs.x = arr` copies into the column (allocating the storage
  when the object was built with `__new__`, as the viz shims do). No behaviour change.
- **`Params` performance section**: `perf_threads` (0 = auto physical cores, -1 = all logical),
  `perf_threading_layer` (`omp` default), `perf_neighbor_backend`, `perf_neighbor_skin`,
  `perf_max_clips`, `perf_field_dtype`, `perf_field_res_um`, `perf_max_grid_2d/3d`,
  `perf_metrics_voronoi_stride`, `perf_keep_snaps_in_memory`, `perf_async_io`,
  `perf_io_queue_depth`, `perf_io_compresslevel`, `perf_fastmath`, `perf_warmup`. Only the thread
  configuration is wired so far (`run()` calls `gels.kernels.configure_threads`); the rest are
  consumed by later phases. `use_numba` will be honoured (False → reference path) once kernels exist.
- **`gels/bench.py`**: `python -m gels.bench --mode 2D --N 500,2000 --steps 5 --threads 1,40`
  times packing, `step()`, rendering and metrics per configuration.
- **`tests/test_numba_env.py`**: proves `@njit(parallel=True, nogil=True, cache=True)` + `prange`
  compiles, caches and is thread-count independent on the installed numba/Python.
- Dependencies: `pyyaml` (setup files), `tbb` (optional threading layer).

### Phase 1 — Species / functionalization data model (no behaviour change; gate bit-identical)
- **`gels/materials.py`**: the f-rules shared by the Python loops and the future kernels, all
  `@njit` leaves — `mix_pair`/`mix_wall` (bilinear coverage mixing of adhesion energy and
  friction stress: collagen–collagen with probability f_i·f_j, collagen–bare with
  f_i(1−f_j)+f_j(1−f_i), bare–bare otherwise), `pair_E_star_Pa`/`wall_E_star_Pa` (per-granule
  E, ν → E*), `ligand_factor` (normalised Langmuir `g = f(1+κ)/(f+κ)`), `traction_gain`
  (law × rule: langmuir|power × target|min|product), `blocker_factor`, and
  `build_pair_tables` (K×K species tables). Every rule reduces exactly to the V2.7 LUT/gates at
  f ∈ {0, 1} (`tests/test_materials.py`).
- **`Params`**: new `species` list (plain dicts; survives `asdict → json → list`),
  `f_min_adhesion=0.05`, `packing_count_rule`, and the functionalization → cell-response block
  `sigma_ligand_max=750`, `K_sigma_traction=160`, `K_sigma_attach=50`, `traction_f_law='langmuir'`,
  `traction_f_rule='target'`, `traction_exponent`, `cell_seeding_law='langmuir'`,
  `cell_seeding_exponent`, `bridge_lock_scales_with_ligand=True`. Adhesion/friction storage
  renamed to coating pairs `W_adh_cc/cb/bb`, `tau_0_cc/cb/bb`; the old names `W_adh_ff/if/ii`,
  `tau_0_ff/if/ii` are read/write alias properties (`Params.LEGACY_ALIASES`), honoured by
  `params.json` loading, trial JSON (`_common.load_trial_json`) and the CLI (`--W_adh_ff` maps to
  `W_adh_cc`).
- **`GranuleSystem`**: per-granule `species_id` (int32), `f`, `E_gran`, `nu_gran`, `activity`,
  `adhesive_mask`; species table `K`, `species_f/E/nu`, `species_names/colors`; precomputed
  `pair_W`, `pair_tau`, `pair_Estar` (K×K) and `wall_W`, `wall_Estar`. **`gtype`, `func_mask`,
  `inert_mask` are now derived** (`gtype = 0 where f ≥ f_min_adhesion else 1`) in
  `set_species_table()`; legacy `gtype`-only construction yields the two-species table and an
  identical `gtype`. New `GranuleSystem.from_snapshot_dict(snap, p)` for post-hoc rendering.
- **Engine helpers**: `legacy_species_from_params(p)` (collagen f=1 from `R_func_*`/`*_func_*`,
  bare f=0 from `R_inert_*`/`*_inert_*`, radius floors 15/20 µm, colours `#CC2222`/`#22AA22`),
  `resolve_species(p)` (uses/caches `p.species`), `seeding_gain(f, p)`.
  `get_pair_friction_params(f_i, f_j, p)` now applies the coverage-mixing rule (was a dead
  gtype LUT).
- **Serialization**: snapshots gain `species_id`, `f`, `E_gran`, `nu_gran` (`gtype` kept);
  `metadata.json` gains a `species` table with per-species granule and cell counts;
  `restore_gs_from_snapshot` restores species (V2.7 snapshots derive them from `gtype`);
  `load_run` resolves species for old runs.
- **`CodeLog/References/fibroblast_parameters.md`**: provenance and verification status of every
  physical default (ligand-density ladder, Gaudet 2003, Mrksich 1998, Oria 2017, Erdmann–Schwarz
  2004, Bangasser 2013/2017), the "not supported — do not implement" list, and the Langmuir
  coupling with its predicted gains (g(0.2) ≈ 0.59 traction, ≈ 0.80 attachment).
- Tests: `tests/test_species_model.py` (legacy construction, multi-species mapping, snapshot
  round-trip incl. a V2.7 fixture, aliases through params.json / trial JSON / CLI).

### Phase 2 — Physics mapping of f into the loops (gate still bit-identical for f ∈ {0,1})
- **Pair contacts** (`reference.compute_forces`, `compute_forces_3d`): the 4-entry `friction_lut`
  and wall LUTs are gone. Adhesion energy W, friction stress τ₀ and the effective modulus E*
  come from the species tables (`gs.pair_W/pair_tau/pair_Estar[si,sj]`, `wall_W/wall_Estar[si]`),
  i.e. coverage mixing of (cc, cb, bb) and per-granule E, ν. Contact records gain
  `species_i/j`, `f_i/j`, `E_star`, `W`, `tau_0`. For identical materials the tables reproduce
  the old `(E·1e3)/(2(1−ν²))` literally (Python `ν**2` is not always `ν·ν` on MSVC), keeping the
  legacy trajectories bit-identical. `mc_dem_correction` uses the recorded per-contact E* and a
  per-granule confinement factor `ν_i/(1−2ν_i)`.
- **Cell traction is two-sided and ligand-gated**: `motor_clutch_force(E, p, fa, nu=None, g=1.0)`
  takes the host granule's E and ν and the ligand gain g, applied as
  `F_stall·k_sub/(k_sub + g·k_opt)·engagement·fa·g` (bond-number ceiling and reduced clutch
  stiffness; never by scaling `n_clutches`). Each bridge pair computes `F_cell_i` (host i's E, ν,
  FA maturity, `traction_gain(f_i, f_j, law, rule, γ, κ)`) and `F_cell_j`; cells on gi use
  `F_cell_i`, cells on gj use `F_cell_j` in `_service_committed_bridges` and `_attempt_new_bridges`.
  Bridging is gated on `adhesive_mask[i] and adhesive_mask[j]` (plus at least one cell), so a bare
  target is never bridged to while a partially coated one is.
- **Bridge lock-in** threshold scales with the same ligand gain when
  `bridge_lock_scales_with_ligand` (Erdmann & Schwarz: sustainable cluster force ∝ bond number).
- **Line-of-sight blocker** (`_classify_bridge_path`): the binary inert branch is replaced by
  `min over blockers of blocker_factor(f_k, bridge_inert_factor)` — bare blocker → the old penalty,
  coated blocker → none, linear in between (piecewise so the endpoints are exact).
- **Cells and coverage**: per-host stiffness factor for spreading (`E_gran[i]`, `nu_gran[i]`);
  gates on `adhesive_mask` instead of `gtype`; new `seeded_cells(n_full, f, p)` and
  `cell_capacity(gs, i, p, A_cell)` scale seeding and the overcrowding capacity by
  `seeding_gain(f)` (both packers, `update_cell_state`, `_update_individual_cells`); active noise
  is drawn for adhesive granules with amplitude × `sqrt(activity)`.
- Tests: `tests/test_physics_mapping.py` — motor-clutch monotone in g and equal to V2.7 at g = 1
  (sign-trap guard), mixed-species contact records vs `materials`, no bridges to bare targets /
  bridges to half-coated ones, blocker penalty 0.1 / 0.55 / 1.0 for bare / half / coated blockers,
  seeding and capacity vs f. `test_legacy_identity` remains bit-identical.

### Phase 3 — Species-aware packing, per-species rendering and metrics (gate still bit-identical)
- **Packing** (`generate_packing`, `generate_packing_3d`, `generate_packing_2d_slice`): counts
  come from `species_counts(species, p)` — `n_k = round(phi_k·V_domain/V̄_k)` with per-species
  `phi_target` (legacy: the exact `phi_f/phi_i` targets) or `solid_fraction × volume_fraction`,
  and `V̄_k` by `count_rule` (`mean_radius` = the V2.7 rule, forced for legacy species;
  `mean_volume` uses E[R²]/E[R³] of the normal or lognormal radius distribution). Placement is a
  round-robin over species in ascending f (bare first) — the V2.7 inert/functional interleave for
  two species — and radius/shape draws use the species' own moments in the old order, so legacy
  packings are unchanged. Reports and `pipeline_state.json` list granules and cells per species.
  All 400 `Trials/DOE2_*.json` reproduce their V2.7 counts.
- **Rendering**: `render_fields_species(gs, p)` returns one grid per species
  `phi_s (K, Ng, Ng[, Ng])`; the five stampers take a single target grid; volume-conserving
  normalisation is shared (`_normalize_species_fields`) and sums grids in index order so the
  legacy `phi_f`/`phi_i` are bit-identical. `render_fields`/`render_fields_3d` are wrappers via
  `group_species_fields` (adhesive vs non-adhesive species); `collagen_field` gives Σ f_k·phi_k.
  `species_view(gs)` lets the lightweight viz shims (gtype only) render unchanged.
- **Metrics**: `compute_metrics(..., phi_s=None)` adds per-species keys `n_gran_sp_k`,
  `n_cells_sp_k`, `n_bridging_sp_k`, `phi_sp_k_true`, `phi_sp_k_mean`, `Z_sp_k`, the contact
  matrix `n_contacts_sp_a_b` (a ≤ b), `phi_c_true`, `phi_c_mean`, `f_solid_mean` and
  `contact_ff_weight` (mean f_i·f_j over contacts); `compute_displacement_species` adds
  `disp_sp_k`. Species are addressed by index; names live in `metadata.json`.
- **`run()`**: renders per species, passes `phi_s` to metrics, drops the grids after metrics, and
  honours `perf_keep_snaps_in_memory` (step 3 sets it False — it never used the returned snapshots).
- Tests: `tests/test_species_packing_render.py` (legacy count formula 2D/3D, volume-fraction
  counts under both rules, three-species packing with the per-granule seeding law, per-species
  rendering/grouping/collagen field, species metric keys, 2D-slice species carry-over, run()
  history keys). End-to-end pipeline smoke (steps 1–5) passes on the legacy path.

### Phase 4 — Sectioned YAML setup and pipeline integration
- **`gels/config.py`**: nested dataclasses for the sections `domain, boundary, granules
  (solid_fraction, count_rule, species[]), cells (seeding, ligand, kinetics, motor_clutch,
  migration, sensing, bridging, activity), contact, dynamics, packing, time, output, performance,
  deformable`; `load_setup` (.yaml/.yml via PyYAML, .json without), `save_setup`, `validate`
  (volume fractions sum to 1, f ∈ [0,1], enums, ≤ 8 species), `apply_overrides` for dotted paths
  with list indices (`granules.species[1].functionalization=0.5`, coerced to the field type),
  `Setup.to_params()` (one explicit `FLAT_MAP` covering every non-species Params field exactly
  once — tested — plus species dicts with `phi_target = solid × volume_fraction` and legacy
  back-fill of `R_func_*`/`R_inert_*`/`func_ratio`/`phi_*` from the extreme-f species),
  `Setup.from_params()`/`from_legacy_params()` (two species; legacy species keep
  `count_rule='mean_radius'` at the species level so converted trials reproduce V2.7 counts),
  `summarize()` (species table) and `TEMPLATE_YAML`. Section defaults are pulled from
  `Params()`, so `Setup() → to_params()` reproduces default Params; the **recommended physical
  values** (migration 30 µm/h, 200 motors × 0.5 nN, 150 nN cap, Langmuir traction with
  σ_max = 750 / K = 160 molecules/µm², attachment K = 50, spreading 2 h, FA rate 0.5/h) live in
  the commented template, not in `Params` defaults (older runs and the fixtures rely on those).
- **`pipeline/step0_new_setup.py`**: writes the commented template (default), or converts a legacy
  trial (`--from-trial`) or an existing run (`--from-run`) into a setup file; refuses to overwrite
  without `--force`.
- **`pipeline/step1_config.py`**: `--setup FILE` (mutually exclusive with `--trial`) and repeatable
  `--set PATH=VALUE`; precedence defaults → file → `--set` → flat flags. Writes `params.json`
  **and** a resolved `setup.yaml`; prints the sectioned summary with the species table. The
  legacy `--trial`/flat-flag path is byte-for-byte unchanged (it only passes through a Setup when
  `--set` is given), so the V2.7 params fixtures still match exactly.
- Tests: `tests/test_config_roundtrip.py` (FLAT_MAP coverage, default and idempotent round trips,
  legacy-trial conversion, every V2.7 params fixture survives `from_legacy_params → to_params`,
  template physical defaults, YAML ≡ JSON, unknown-key and validation errors, `--set` coercion and
  path errors). A three-species `--setup` run passes steps 1–5.

### Phase 5 — Engine observer hook, I/O hardening and the live view
- **Observer hook** (`gels/live/observer.py`; `run(p, seed, observer=None)`): `on_start(gs, p, seed,
  n_steps, resume_step)`, `on_step(s, t, gs, F, contacts) → bool(stop)`, `on_save(snap_idx, s, t,
  snap, m)`, `on_end(hist, gs, reason)`; `CallableObserver` / `as_observer` accept a plain function.
  Zero overhead when `None`. A stop request — or Ctrl+C — ends the loop gracefully: the final state
  is saved as a snapshot at the stop step, `metadata.json` records `steps_completed`, `t_reached`,
  `stopped_early`, `stop_reason`, and `step3 --continue` resumes exactly from there.
- **I/O hardening** (`gels/engine.py`): snapshots and `history.json` are written atomically
  (`.tmp` + `os.replace`); `history.json` is rewritten at every save instead of only at the end;
  `save_history_to_disk` uses the union of metric keys; `load_params(run_dir)` extracted from
  `load_run`. The engine switches stdout/stderr to `errors='replace'` at import, so its progress
  lines (α, µ, →) degrade to `?` instead of raising `UnicodeEncodeError` on a cp1252 Windows console
  (previously only the pipeline scripts were protected).
- **Frames** (`gels/live/frames.py`): vectorised `cell_world_xy` / `cell_world_xyz` (agree with the
  engine's scalar helpers to 1e-9), `bridge_segments` (end on the target surface, minimum image when
  periodic), compact `start` / `frame` messages (float32 positions, int8 cell states, `(M, 2, dim)`
  bridge segments, metrics on save steps). The same builders serve a running `GranuleSystem` and a
  saved snapshot dict.
- **`LiveViewObserver`**: bounded `multiprocessing.Queue(maxsize=4)`, `put_nowait` — frames are
  dropped, never awaited, when the window falls behind; adaptive cadence from the measured step
  rate (or a fixed `--live-every`); control channel for pause / single step / stop / cadence / fps /
  colour mode / detach; a paused run waits in 50 ms slices; a dead viewer detaches and the run
  continues.
- **Viewer process** (`gels/live/viewer.py`, TkAgg in a `spawn` child): one `EllipseCollection` for
  all granules (species colours, or `f` / speed colour maps), one marker `Line2D` per cell state, a
  `LineCollection` for bridges (locked bridges highlighted), HUD, species + cell-state legend, a
  four-panel metrics column (bridges, porosity, K, largest cluster), blitting on a Tk `after()`
  timer; 3D runs are shown as a z-slice (`[` `]` `0` move it). Keys: space, n, x, q/Esc, +/−, c, b,
  l, m, g, s, h. Headless mode writes one PNG per frame. The viewer never imports `viz2.common`.
- **Tail / replay** (`gels/live/tail.py`, `pipeline/live_view.py`): `SnapshotTailSource` follows a
  run directory (attach to a running step 3 with `-i RUN`, `--from-start` replays what exists
  first; `.tmp` files are invisible and a truncated `.npz` is retried on the next poll);
  `ReplaySource` plays back a finished run (`--replay`, `--loop`, `--fps`); `--headless --out DIR`
  dumps PNGs. Metrics for the panel are joined from `history.json` by time.
- **Step 3 flags**: `--live`, `--live-every N`, `--live-fps F`, `--no-live-hold` (otherwise step 3
  waits for the window to be closed), `--threads N` (sets `perf_threads`; `run()` applies it via
  `gels.kernels.configure_threads`).
- **viz2 split**: colours and species tables in `viz2/palette.py`; matplotlib-free snapshot helpers
  (`is_3d`, `slice_snap_z_midplane` now carrying `species_id` / `f`, vectorised
  `cell_world_positions`, `ensure_phase_fields` via `GranuleSystem.from_snapshot_dict`) in
  `viz2/snapshot_ops.py`; `viz2/common.py` (610 → 298 lines) re-exports both so existing imports
  work. `draw_granule_patches` colours by species when `colors=None`; `make_scaffold_legend(species=,
  p=)` lists every species; `draw_cells_on_ax` draws with a handful of `EllipseCollection`s instead
  of one patch per cell (same look; ~1 s → milliseconds per frame).
- Tests: `test_observer_hook.py` (callback order and counts, stop at a step writes history + final
  snapshot and `--continue` picks it up, incremental `history.json`, `load_params`),
  `test_frames.py`, `test_viz2_species.py`, `test_live_viewer.py` (headless replay of the 2D and 3D
  fixtures — one PNG per snapshot, every cell drawn, three species → three colours, z-slice subset;
  tail source ignores `.tmp` and retries a truncated file; observer never blocks on a full queue;
  pause / step / resume / detach; replay control). Legacy gate unchanged: every V2.7 fixture still
  reproduces bit-identically.

### Phase 6a — Compiled contact kernels (numba, parallel)
- **`gels/kernels/contact2d.py`, `contact3d.py`**: the per-pair Python loops of
  `reference.compute_forces[_3d]` become three compiled passes — `pair_pass_*` (`prange` over
  half pairs → per-pair records: geometry, JKR normal force, area-dependent friction),
  `mc_dem_pairs` (per-particle volumetric strain and the Giannis κ correction, accumulated in
  pair order so κ matches bit for bit) and `gather_*` (`prange` over particles → forces,
  torques, fixed-width clip planes, walls; owner writes only, no atomics, so results do not
  depend on the thread count). Cell bridging stays in Python for now
  (`gels/kernels/bridging.py`), walking the same candidate pairs in the same pair order, so
  the cell state machine and the random stream are unchanged; active noise is the
  reference's numpy code. Compiled contact + Python bridging is the interim until Phase 6c.
- **`gels/kernels/geometry2d.py`, `geometry3d.py`**: status-tuple copies of the superellipse,
  superellipsoid and wall contact solvers (numba cannot call an Optional-returning function
  from compiled code); the 2D and superellipsoid solvers are statement-for-statement the
  originals, the sphere / 3D-wall routines are scalar re-expressions of the Python originals.
- **`gels/kernels/neighbors.py`**: `half_pairs_tree` (the reference's `cKDTree` pairs, same
  order) and compiled `csr_from_pairs` (particle → its pairs). The linked-cell list with a
  Verlet skin (`perf_neighbor_backend = 'cells'`) comes in a later step.
- **`gels/kernels/contacts.py`**: `ContactSoA` — structure-of-arrays contact records that still
  iterate and index as dicts (viz, user code unchanged; `save_snapshot_to_disk` gained a
  column fast path), `clips_to_lists` (clip arrays → the renderer's tuple lists),
  `add_active_noise`.
- **Dispatch** (`gels/engine.py`): `kernels_enabled(p)` — numba present, `use_numba`, rigid
  granules (LS-DEM keeps the reference path), `perf_neighbor_backend != 'reference'`;
  `compute_forces` / `compute_forces_3d` route accordingly, nothing else in `step()` or
  `run()` changes. `python -m gels.bench --path kernels|reference` with a per-phase breakdown
  (cell state machine / forces: neighbours, contacts, bridging / render / metrics).
- **Exactness** (`tests/test_forces_vs_reference.py`): same `GranuleSystem`, same seed →
  forces and torques agree to 1e-9 (only the summation order differs), identical contact set
  with per-pair E*, W, τ₀, identical clip planes, identical cell state / bridge targets /
  lock flags / per-cell forces, identical RNG state afterwards; 1 vs 8 threads bit-identical;
  3-step `step()` trajectories agree to 1e-7; mixed-stiffness species carry `pair_E_star_Pa`.
  Covers 2D circles and superellipses (walls and periodic), 3D spheres (walls and periodic)
  and 3D superellipsoids. The bit-identical fixture gate (`tests/make_fixtures.py`) now pins
  `use_numba=False`: the fixtures define the reference path.
- Reference quirk preserved on purpose (logged for Phase 7 review): with periodic boundaries
  the torque lever arm of the second body of a superellipse pair uses its centre inside the
  box rather than the minimum image; the kernels reproduce it so trajectories match.

### Phase 6d — Compiled rendering and metrics
- **`gels/kernels/render.py`**: bounding-box stamps, `prange` over grid rows (2D) or x-slabs (3D)
  with a per-slab CSR of granule images, so every pixel accumulates its granules in a fixed
  order (thread-count independent); circles, superellipses, spheres and superellipsoids, JKR
  clip planes (fixed-width arrays built from `gs.contact_clips`), periodic ghost images with the
  reference's selection criterion. The 2D kernel truncates each tanh profile at 8 interface
  widths (< 1e-7 there; the reference stamps the whole grid) — fields agree to ~1e-6; the 3D
  kernel uses the reference's own 3w box. Effective radii (`effective_radii_2d/3d`) are compiled
  passes over the same `cKDTree` pairs in the same order (2D bit-identical; 3D to rounding, the
  reference takes `d` from BLAS `np.dot`). Normalisation is the reference's numpy code. LS-DEM
  (deformed) rendering stays on the reference path (`engine._render_kernels_ok`).
- **`gels/kernels/metrics.py`**: same keys and values as `reference.compute_metrics`; the contact
  statistics and bridge-count pair loops are sequential compiled loops over the same pairs in
  the same order (sums bit-identical), cluster sizes via `np.bincount`, the Voronoi compartment
  query on all cores (`workers=-1`), superellipse areas / perimeters / superellipsoid volumes
  computed once per `GranuleSystem` (`shape_cache`; rigid granules) instead of at every save.
- **Dispatch** (`gels/engine.py`): `render_fields`, `render_fields_3d`, `render_fields_species`
  and `compute_metrics` route to the kernels under `kernels_enabled(p)` (rendering additionally
  requires no LS-DEM state); `run()` / step 2 / step 3 / the bench pick them up unchanged.
- Tests (`tests/test_render_metrics_vs_reference.py`, 18): fields within 2e-5 of the reference for
  2D circles and superellipses (walls, periodic), 3D spheres (walls, periodic) and 3D
  superellipsoids, with clip planes present; effective radii; every metric key equal (ints exact,
  floats to 1e-9) on the same fields, including a three-species system; dispatch and the shape
  cache. Reference timings that motivated this step (2D circles, 40 threads): one render took
  30.9 s at N = 1000 and 298 s at N = 3000 against a 0.25 s / 0.87 s step.

### Phase 6c — Compiled cell state machine and bridging
- **`gels/kernels/cells.py`**: `aggregates_k` (attachment, spreading, FA maturity, overcrowding;
  `prange` over granules), `cells_update_k` (bridge ageing / ligand-scaled lock-in / senescence,
  directed migration with the 3D great-circle step, state assignment, random-walk migration;
  each granule updates only its own cells), `bridge_pairs_k` (`prange` over candidate pairs:
  gap, per-side motor-clutch traction with the ligand gain, committed-bridge count, path factor
  via the restricted ray-cast over N(i) ∪ N(j)) and `bridge_cells_k` (`prange` over granules:
  service / rupture of committed bridges, attempts by the granule's own eligible cells —
  hemisphere check, exclusion zone, cell-specific decay, fast-path bridging or migration).
  Randomness is a counter-based splitmix64 hash keyed by one 62-bit draw per call from the run's
  `Generator` plus cell / partner indices: deterministic for a seed, independent of the thread
  count, one `Generator` draw per pass instead of one per eligible cell.
- **`Params.perf_cells_backend`** (`performance.cells_backend`): `kernels` (default) or `python`
  — the exact V2.7 cell machinery (`reference.update_cell_state`, `gels/kernels/bridging.py`) on
  top of the compiled contacts. `engine.update_cell_state` and the contact kernels dispatch on it.
- Restricted ray-cast (also in the Python bridging pass): every possible blocker of a bridging
  candidate pair lies within `2·max(r_bound) + L_max` of one of the two granules, so testing the
  neighbour lists the contact kernel already built gives the reference's blocker set exactly at
  O(Z) instead of O(N) per pair (the reference scanned — and allocated — all N granules per pair).
- Semantics that differ from the reference, on purpose: the "existing bridges" boost uses the
  start-of-step committed count (the reference saw bridges formed earlier in its sequential
  sweep); with periodic boundaries the partner is always the owner's minimum image (the reference
  mixed a virtual image with real positions on the second side of a straddling pair).
- Tests (`tests/test_cells_vs_reference.py`, 13): deterministic parts exact against the reference
  (state machine with the random walk off, in 2D walls / periodic and 3D; service and rupture
  with attempts disabled), bridging statistics equivalent over 6–12 seeds (2D walls, 2D periodic,
  3D), 1 vs 8 threads bit-identical over four steps, hashed uniform / normal moments, periodic
  bridges consistent with the minimum image, backend switch. `test_forces_vs_reference` pins
  `perf_cells_backend='python'` for its RNG-exact comparison.
- Measured before this step (2D circles, N = 3000, 40 threads): step 0.55 s of which compiled
  contacts 0.014 s, Python bridging 0.35 s, Python cell state machine 0.10 s.

### Phase 6f — Packing: neighbour-grid RSA and compiled settle (bit-identical)
- **`gels/kernels/packing.py`**: `RSAChecker` answers the random-sequential-addition overlap
  question either with the reference's Python scan or with a linked-cell grid searched over an
  adaptive reach — same inequality, same operands, same decision, hence the same random draws
  (the scalar `rng.uniform` draws per attempt are kept, because attempt counts are data-dependent
  and batching them would change the stream). `settle_packing_2d/3d` run the reference's
  inflate-and-relax with a compiled substep; the pair list still comes from `cKDTree` in the
  reference order and each particle sums its i-side terms in pair order then its j-side terms —
  the order `np.add.at` used — so positions match to the bit. Result: the generated packing is
  identical on both paths (`tests/test_packing_vs_reference.py`, 9 configurations: 2D circles /
  superellipses, 3D spheres / superellipsoids, walls and periodic, three species, settle off),
  and a run's initial condition does not depend on the compute backend. `generate_packing[_3d]`
  and `_settle_packing_2d/3d` dispatch on `kernels_enabled(p)`.
- Settle pair list follows `perf_neighbor_backend`: `cells` (default) rebuilds a linked-cell list per
  substep — an order of magnitude faster than a `cKDTree` at N ≥ 10⁴, deterministic and
  thread-independent, but its sorted pair order rounds the jammed configuration differently from the
  reference; `ckdtree` (or the exact cell mode) keeps the bit-identical settle. The RSA stage is
  identical either way.

### Phase 6g — Neighbour list, clip arrays, overlap resolution, metrics and thread tuning
- **`gels/kernels/neighbors.py`**: `half_pairs_cells` — compiled linked-cell grid, parallel over
  particles, each particle's partners sorted, so the pair list is lexicographic and independent
  of the thread count; same pair set as `cKDTree.query_pairs` (`tests/test_neighbors.py`, 2D and
  3D, walls and periodic; periodic boxes narrower than three cells fall back to the tree).
  `neighbor_backend(p)` keeps the tree whenever the exact Python cell machinery runs (its random
  stream depends on the reference pair order) and for `perf_neighbor_backend='ckdtree'`.
- **`gels/kernels/integrate.py`**: `resolve_overlaps` — the post-step overlap correction on the
  configured neighbour list with a compiled pass (bit-identical to the reference with the tree
  backend, to rounding with the cell list).
- Clip planes stay as arrays on the kernel path (`gs.clip_arrays`, consumed directly by the
  compiled renderer) instead of being converted to per-granule Python lists every step;
  `step()` no longer resets N Python lists when the kernels are active.
- Metrics: the functional-cluster labelling is computed once (was twice), and the Voronoi
  compartment query samples a stride-`perf_metrics_voronoi_stride` subgrid (default 4; stride 1
  reproduces the reference exactly, stride 4 changes the compartment estimates by < 2e-2 on the
  test systems — `test_voronoi_stride_close_to_full_grid`).
- Threads: `perf_threads = 0` now means `auto_threads(N)` = clamp(N / 1000, 4, physical cores),
  chosen in `run()` once the granule count is known; the benchmark showed 40 threads slower than
  8 below N ≈ 10⁴ (fork/join overhead across many small `prange` regions).
- Cache hygiene: `gels.kernels.ensure_cache_fresh()` (run at import) hashes every kernel source
  plus `engine.py` / `materials.py` and purges numba's `.nbi` / `.nbc` files when the hash moves —
  numba does not invalidate callers when a leaf they inlined from another module changes.
- Reserved, not yet honoured: `perf_neighbor_skin` (no Verlet skin — the cell list is rebuilt every
  step), `perf_field_dtype`, `perf_field_res_um`, `perf_max_grid_2d/3d` (the grid policy is still
  the reference's `max(Ngrid, L / 2w)`), `perf_fastmath`, `perf_warmup`.
- Benchmark (2D circles, this workstation) before this step, kernels vs reference: step 251 ms →
  7 ms at N = 1000 (8 threads), 874 ms → 23 ms at N = 3000, 65 ms at N = 10 000; render at
  N = 3000 298 s → 0.2 s. After this step: 20 ms per step at N = 10⁴ and 0.18 s at N = 10⁵ (8 threads);
  full tables in `CodeLog/References/benchmarks_v3.md` (`results/` is gitignored, so the record lives there).

### Phase 6e — run() output path
- **`gels/io/writer.py`**: `write_npz_atomic` writes a dict of arrays as a zip of `.npy` members
  (the `np.savez` layout, so `np.load` reads it unchanged) at a selectable deflate level through
  a `.tmp` file renamed into place; `SnapshotWriter` runs those writes on a background thread
  behind a bounded queue (`perf_io_queue_depth`, default 2) — the run hands over already-copied
  arrays and continues, blocks only when the disk falls that many files behind, and collects
  write errors to report at the end. `run()` uses it when `perf_async_io` (default on) at level
  `perf_io_compresslevel` (default 1: 3–4× faster than numpy's default 6 for a few percent more
  bytes); snapshots and field grids go through it, `history.json` stays synchronous.
- The in-memory snapshot dict (~30 array copies per save) is now built by `_snapshot_dict()` only
  when `perf_keep_snaps_in_memory` is set or an observer is attached. **Default change:**
  `perf_keep_snaps_in_memory = False` — `run()` returns `snaps = []`; use `load_run()` for the
  saved snapshots (the pipeline already did). Tests: `tests/test_io_writer.py`.

### Phase 7 — Species in the remaining viz / analysis paths, documentation
- `viz2/scaffold_map.py`: the legend lists every species (it was built without the species
  table and showed only Functional / Inert next to three-coloured granules).
- `viz/postprocess.py`: the `__new__`-based GranuleSystem shim used for re-rendering becomes
  `GranuleSystem.from_snapshot_dict` (species, radii and shapes come back exactly as saved).
- `viz2/scaffold_map_3d.py`: granules coloured by species (`viz2.palette.species_colors`; legacy
  runs keep red / green). `viz2/energy_stress.py`: the friction dissipation field takes τ₀ per
  contact from the coverage mixing rule (`viz2.snapshot_ops.contact_friction`, equal to the old
  ff / if / ii lookup for f ∈ {0, 1}). `viz2/phase_fractions.py`: new `species_fractions.png`
  (stacked per-species fractions with the collagen-weighted solid) when the history carries
  `phi_sp_{k}_mean`; skipped for pre-V3.0 runs.
- Documentation: `CodeLog/Architecture/ARCHITECTURE.md` (kernel module table, exactness policy,
  compute layout of a step), `CLAUDE.md` (kernel layout, compiled-kernel and threading decisions,
  numba constraints), `pipeline/README.md` (performance notes), `CodeLog/Readme/README.md`
  ("What's new in V3.0", requirements, step 0 in the quick start).

### Showcase runs and multi-run comparison (2026-09-17)
- **`pipeline/run_showcase.py`**: nine conditions that exercise what V3.0 adds, run concurrently
  as separate pipeline processes and then compared on one set of figures. A six-run
  functionalization ladder on one 5 × 5 mm domain (~3 200 granules each: binary f = 1/0;
  three species 1/0.5/0; five-step ladder 1/0.75/0.5/0.25/0; low dose f = 0.2/0; uniform 0.5;
  uniform 1), shaped granules per species (rounded squares coated, ellipses half coated, circles
  bare), a 20 × 20 mm two-dimensional bed of ~51 000 granules (>10⁵ cells) and a 1.6 mm 3D cube of
  ~8 000 spheres. Each condition is written as a commented setup file in `Trials/showcase_v3/`
  (editable, runnable alone with `step1 --setup`) and driven through steps 1–5 with its own
  thread count; a weighted scheduler keeps the sum of running threads within `--budget`
  (default: physical cores). Flags: `--list`, `--only`, `--t_total`, `--budget`, `--force`,
  `--no-postprocess`, `--no-analysis`, `--compare-only`, `--write-setups`, `--dry-run`. Output:
  `results/showcase_v3/<condition>/` (standard run layout plus `showcase_log.txt` and
  `showcase.json`), `results/showcase_v3/showcase_manifest.json`, `README.md` (what each run is
  and where the figures are) and `comparison/`. Measured: the physics of all nine conditions
  (≈ 78 000 granules, ≈ 260 000 cells, 72 h each) finished within ≈ 6 min of wall time — 25–65 s
  of dynamics per 3 000-granule run, 188 s for the 51 000-granule bed, 91 s for the 8 000-sphere
  cube (table in `CodeLog/References/benchmarks_v3.md`).
- **`viz2/compare_runs.py`**: any number of finished runs on the same figures —
  `compare_scaffolds.png` (final scaffold of every run, one panel each, the same colour for the
  same f across runs, a scale bar per panel because domains differ, one shared legend),
  `compare_timeseries.png` (bridging cells per 100 granules and per seeded cell, mean bridge
  force, functional connectivity, porosity, displacement of coated granules; one line per run),
  `compare_species.png` (stacked species fractions per run), `compare_dose_response.png` (final
  cells per granule, bridging fraction, displacement and coordination number of *every species*
  against its own f — one point per species, one line per run, marker filled with the species
  colour: the Langmuir seeding law read straight off the runs, 0 cells per granule at f = 0,
  3.1 at f = 0.2, saturating at 4.1 above f = 0.5), `compare_timelapse.gif` (the scaffold grid on
  a common time axis) and `compare_summary.md/.csv` (size, cells, composition per species, wall
  times of every step, final metrics). Circles are drawn as one
  `EllipseCollection` per panel so 5 × 10⁴-granule runs render in seconds; shaped and periodic
  runs use `draw_granule_patches`; 3D runs are shown as their z-midplane slice. CLI:
  `python viz2/compare_runs.py -i RUN [RUN ...] | PARENT_DIR | "glob" -o OUT --labels ...`.
- `CodeLog/References/benchmarks_v3.md`: second benchmark pass recorded — N = 10⁵ at 16 / 40 threads
  with the cell-list settle (pack 155 s / 184 s, step 0.143 s / 0.238 s) and 3D spheres at 5 000 /
  20 000 (step 0.033 s / 0.123 s at 16 threads); 16 threads is the sweet spot for one run on this
  machine, several runs in parallel use it better than one run at 40.

### Parallel visualization (2026-09-17)
- **`viz2/parallel.py`** (new): `pmap(fn, items, workers)` — a drop-in parallel `map` over the
  per-snapshot work of the visualization suite, backed by a `spawn` `ProcessPoolExecutor`.
  `fn` must be a module-level function taking one picklable argument. Worker count comes from the
  argument, else `$VIZ2_WORKERS`, else the physical core count capped at 8; `workers=1` runs
  inline with no pool, which is also the automatic fallback if the pool cannot start or a worker
  dies, so output is identical either way. Workers pin `OMP/NUMBA/OPENBLAS/MKL/NUMEXPR_NUM_THREADS`
  to 1 (several visualization processes may already run side by side) and set
  `GELS_NO_CACHE_CHECK` (the parent has already validated the kernel cache).
- Every expensive loop in `viz2` is now a `pmap` over snapshots, because snapshots are independent:
  all five GIF builders in `movies.py` (`_frame_scaffold`, `_frame_voronoi`, `_frame_local_phases`,
  `_frame_stress`, `_frame_energy_modes`), the tessellations in `voronoi_shapes.py`
  (`_voronoi_and_shapes`, `_shape_stats`) and `phase_fractions.py` (`_local_fractions`,
  `_inner_outer_means`), the energy fields in `energy_stress.py` (`_energy_fields`) and the cluster
  analysis in `void_percolation.py` (`_percolation_stats`, `_cluster_sizes`).
- **Duplicated tessellations removed.** `voronoi_shapes.run_all` computed the same centre-based
  Voronoi of the same selected snapshots three times (overlay, shape-factor maps, histograms) and
  `phase_fractions.run_all` twice (heatmaps, local-vs-global); each now computes it once and passes
  it down via a new `precomputed=` argument. This is a saving on the serial path too.
- Plumbing: `viz2/postprocess.py::run_all(workers=...)` resolves the count once, prints it and
  passes it only to modules whose `run_all` accepts it (checked with `inspect.signature`, so a
  third-party module keeps working); `python viz2/postprocess.py --workers N`; every parallelized
  module's own CLI grew `--workers`; `pipeline/step4_postprocess.py --workers N` (recorded in
  `pipeline_state.json`); `pipeline/run_showcase.py` gives each condition a `viz_workers` count
  (4, or 6 for the two large runs).
- Measured on the 3 183-granule / 37-snapshot showcase run, idle machine: the whole suite goes
  from **1709 s to 318 s (5.4×)**, with all 25 output files byte-size identical. Per module:
  movies 890 → 100 s (8.9×), phase fractions 253 → 40 s, Voronoi shape factors 253 → 50 s,
  energy/stress 147 → 51 s, percolation 160 → 72 s. The last two gain least because their cost is
  in the parent process (large multi-panel PNGs drawn after the parallel part), which puts the
  suite's serial floor near 60 s — beyond ~16 workers there is little left to win. Full table in
  `CodeLog/References/benchmarks_v3.md`.

### 3D granule-size / composition sweep (2026-09-17)
- **`pipeline/run_showcase.py --sweep sweep3d`**: a second condition set, a full factorial over
  three factors in a 2 mm cube — functional granule size (mean diameter 40 / 80 / 150 µm, f = 1),
  inert granule size (the same three, f = 0) and functional share of the solid (0.50 / 0.75 / 1.00).
  At 100 % functional the inert size stops mattering, so the 27 nominal cells collapse to
  **21 distinct conditions** spanning 2 400 to 127 500 granules and 54 000 to 382 000 cells
  (1.02 M granules and 3.3 M cells in total). `--sweep` selects the condition set and its output
  and setup folders (`results/sweep3d/`, `Trials/sweep3d/`); `--sweep showcase` is the default.
- `Condition` gains `Ngrid_3d`, `cell_surface_coverage` and a free-form `factors` dict that is
  written into each run's `showcase.json`, so a sweep figure can key on the factor coordinates.
- Sweep choices, all recorded in the generated setup files: solid fraction **0.55** (random close
  packing of spheres is ≈ 0.64 and the V3.0 3D benchmark showed 0.62 leaves residual overlaps above
  tolerance; 0.55 packs cleanly for every size combination including the three monodisperse cells);
  cell seeding **0.8 of each granule's surface area** (3 cells on a 40 µm granule, 13 on an 80 µm
  one, 45 on a 150 µm one — bare granules take none, since the Langmuir seeding gain is 0 at f = 0);
  `Ngrid_3d = 256` (7.8 µm voxels) instead of 200, which is still only ≈ 2.6 voxels per radius at
  40 µm, so grid-derived porosity / connectivity are coarse for the smallest size while the
  analytic `*_true` metrics are unaffected; snapshots every 4 h over 72 h (19 per run).
- Result (2026-09-18): all 21 conditions completed in **8 h 10 min** of wall time under a
  40-thread budget. Compaction of the coated phase falls monotonically with functional share in
  every size combination — most strongly for large coated granules dispersed in small bare ones
  (72.7 µm mean displacement at 50 % functional against 27.1 µm at 100 %) — while coordination
  number rises with functional share and converges to ≈ 2.0 contacts per granule. The coated phase
  stays fully connected (`func_lf` ≈ 1) everywhere except that same dilute large-in-small case
  (0.79). Bridging saturates: ≈ 99 % of seeded cells end up in a locked bridge in every condition,
  so displacement and coordination, not bridge counts, are the discriminating readouts at these
  parameters. Figures and the per-condition table are in `results/sweep3d/comparison/`; timings and
  the packing-scaling caveat in `CodeLog/References/benchmarks_v3.md`.
- Cell parameters are the literature-based V3.0 template values documented in
  `CodeLog/References/fibroblast_parameters.md` (30 µm/h migration, 200 × 0.5 nN motor stall,
  75 clutches, 150 nN cap, 2 h spreading, 0.5 h⁻¹ FA maturation, 40 µm sensing, σ_max = 750
  collagen/µm²). Bridge lock-in is left **on** at 20 nN: a fully mature cell on 10 kPa granules
  generates 46.6 nN (measured, matching the predicted 47 nN), so every matured bridge locks —
  and because `SENESCENT` is a terminal state, raising the threshold above 47 nN would instead
  send every bridging cell to senescence by ~30 h and leave the scaffold inert for the rest of
  the run. Lock-in is what keeps the tissue load-bearing over 72 h.

### Bug Fixes
- **`viz2` void-cluster sizing was O(clusters × grid points).**
  `compute_void_clusters` and `cluster_size_distribution` each built their size array as
  `[np.sum(labels == k) for k in range(1, n_clusters + 1)]` — a full pass over the label array per
  cluster, and the two functions did it separately for the same snapshot. On a 1.4 M-cell grid with
  63 478 clusters that loop takes **142.6 s**; `np.bincount(labels.ravel())[1:]` gives the identical
  array in **0.007 s** (19 500×). On the real 20 mm showcase snapshot — 50 930 granules, a 3334²
  grid, 14 175 void clusters — the old loop compared 1.6 × 10¹¹ elements per call and ran twice per
  snapshot, which had the post-processing of that condition stuck for over an hour; the whole
  per-snapshot chain (field render, labelling, percolation test, size distribution) now takes
  **2.6 s**. Both functions call a shared `cluster_sizes(labels)` helper.
  The same pattern appeared a third time in `plot_void_clusters`, which tinted the cluster overlay
  with `for k in range(1, n_clusters + 1): mask = labels == k` — another full grid sweep per
  cluster, plus two masked writes. On a 700² grid with 21 920 clusters that loop takes **36.1 s**;
  colouring through a label-indexed lookup table (`lut[labels]`) gives a bit-identical image in
  **0.011 s** (3 400×), and on the real 20 mm panel it was the difference between ~25 minutes and
  well under a second per panel. End to end: post-processing the 20 mm showcase condition
  (scaffold + phases + percolation, 37 snapshots, 50 930 granules, 16 workers) now takes
  **119 s**; before these two fixes the percolation module alone ran for over an hour without
  finishing. The equivalent fix landed in
  `gels/kernels/metrics.py` during Phase 6d; `viz2/void_percolation.py` was missed.
- **`analysis/coarse_grain.py::coarse_grain_field` swept every grid cell against every contact
  in Python.** The stress branch ran `for ix, iy, iz: for k in contacts:` with several small array
  allocations in the inner body — Ngrid³ × n_contacts, i.e. 8 000 × 50 000 = 4 × 10⁸ iterations per
  call in 3D — and the strain-rate branch did the same twice over all granules. It is called once
  per frame by the stress movie, once per panel by the stress and strain maps, and by step 5, which
  is why post-processing and analysis were the two largest line items of the 3D sweep (19.8 h and
  21.6 h against 3.6 h of actual simulation). Now vectorised over chunks of grid cells
  (`_cell_centres`, `_gauss_weights`, `_chunk_rows`, `np.tensordot` / `np.einsum`), with the same
  Gaussian weights, the same 1e-10 cutoff and the same normalisation; only the summation order
  changes. Measured against the previous implementation on real snapshots:

  | case | quantity | before | after | speed-up | max relative difference |
  |---|---|---|---|---|---|
  | 2D, 3 183 granules, 3 691 contacts, Ngrid 20 | stress | 8.97 s | 0.099 s | 91× | 8.5e-16 |
  | 2D, same | strain rate | 13.80 s | 0.188 s | 74× | 7.7e-16 |
  | 3D, 2 418 granules, 3 897 contacts, Ngrid 8 | stress | 16.91 s | 0.120 s | 141× | 2.1e-15 |
  | 3D, same | strain rate | 15.82 s | 0.229 s | 69× | 1.0e-15 |

  The differences are at the level of floating-point summation order. The 3D comparison uses
  Ngrid = 8 only because the old path is unaffordable at the production Ngrid = 20, where it is
  a further ~16× more work. End to end, with this fix and the two `viz2` cluster fixes, the full
  six-module `viz2` suite on the 3D showcase condition (8 026 granules, 75 638 cells, 37 snapshots,
  all five GIFs, 16 workers) takes **300.8 s**; before the fixes the same suite ran for over an
  hour on two separate attempts without finishing.
- **viz2 local phase fractions scaled as O(granules × grid points).**
  `viz2/voronoi.py::compute_local_phase_fractions` ran a point-in-polygon test of every Voronoi
  cell against the *whole* phase-field grid (≈ 7 × 10⁵ points at 5 mm), per cell, per snapshot —
  fine for the 60–100 granules of the V2.x runs, but ~2×10⁹ tests per snapshot at N = 3 000 and
  the first showcase pass sat in `phase_fractions` for 20 minutes with no output (the
  51 000-granule run would never have finished). The test is now restricted to the cell's
  bounding-box window of the grid (identical result, total cost O(grid)).
  `viz2/phase_fractions.py` additionally skips the Voronoi-based local plots above
  `LOCAL_VORONOI_MAX_GRANULES = 20000` (one tessellation per snapshot, clipped granule by
  granule in Python, still costs minutes per snapshot there); the global and per-species panels
  are always drawn. `pipeline/run_showcase.py` holds its thread budget for steps 1–3 only, so
  post-processing of finished runs no longer blocks queued physics.
- **Displacement metric (`disp_func`, `disp_inert`, `disp_sp_k`) was wrong in two ways.**
  `GranuleSystem.pos_unwrap` — the unwrapped positions that `compute_displacement` /
  `compute_displacement_species` compare against the start positions — must equal `pos` plus an
  integer number of periodic images. (1) It was set at construction, i.e. at the RSA positions,
  and the inflate-and-relax settle moved `pos` without it, so a fresh `run()` reported the settle
  displacement (~40–55 µm) as a constant offset from t = 0 (since V1.10; resumed runs were
  unaffected because `restore_gs_from_snapshot` re-syncs, which is why a stepped run showed only a
  spurious spike at t = 0). (2) It was advanced by the integrated velocity, while wall clipping and
  the V2.1 overlap resolution move `pos` directly, so it drifted away from the real positions for
  every granule pressed against a wall or pushed apart — the reported mean displacement of the
  showcase runs was 2–4× the true |x(72 h) − x(0)| (81 vs 35 µm for the binary run, 457 vs
  105 µm for the shaped run). All three packing generators now end with `pos_unwrap = pos`, and
  `step()` advances `pos_unwrap` by the step's true displacement (`_sync_unwrapped`: position
  after all boundary handling minus position before integration, minimum-imaged for periodic
  boxes), so the invariant holds exactly on both compute paths. Only the `disp_*` history keys
  change; the reference fixtures in `tests/fixtures/` were regenerated with
  `tests/make_fixtures.py --force` (every other value bit-identical to before), and
  `test_species_model` now builds its pre-V3.0 snapshot by stripping the V3.0 keys from the
  fixture instead of assuming the fixture lacks them.

---

## [V2.7] - 2026-09-16

Local-only restructure. HPC execution is retired, the engine moves into a
proper package, and running a simulation becomes an explicit five-step process.

### Changed (Repository Layout)
- **`gels/` package**: `new_dem_0.py` → `gels/engine.py` and `lsdem.py` →
  `gels/lsdem.py`. A new `gels/__init__.py` re-exports the common API
  (`Params`, `run`, `load_run`, packing/render/snapshot helpers), so
  `from gels import Params, run` works. All ~30 import sites across `viz/`,
  `viz2/`, `analysis/` and `Trials/` were updated from `from new_dem_0 import`
  to `from gels.engine import`. Every module already inserted the repo root on
  `sys.path`, so no installation step is needed.
- **No `.py` at the repository root.** The root now holds only packages and
  documentation.
- **`old code/`**: archive for retired scripts — `run_all_trials.py`,
  `run_hpc_headless.py`, `run_analysis_pipeline.py`, `reconstruct_history.py`,
  `test_hertz_clipping.py`, and the whole `hpc/` tree (SLURM templates,
  `Alex.json`, `sync_results.py`, `generate_hpc_scripts.py`). Their imports
  were left pointing at the old module names so they remain a faithful record
  of the V2.6 state; see `old code/README.md`.

### Added (Step-by-Step Pipeline)
- **`pipeline/`**: five numbered scripts, run one at a time, all operating on a
  single run directory.
  - `step1_config.py` — resolve Params from defaults → trial JSON → CLI flags;
    writes `params.json`. One flag per Params field, as before.
  - `step2_pack.py` — RSA + Lubachevsky-Stillinger packing, cell seeding, and
    the `t = 0` force/metric evaluation; writes `snapshots/snap_0000.npz`,
    `history.json`, `metadata.json`.
  - `step3_simulate.py` — restores snapshot 0000 and advances to `t_total`.
    `--continue` resumes an interrupted run from its last snapshot and can
    extend a finished run to a later `t_total`; `--force` restarts from the
    packing; `--archive` writes a `.tar.gz`.
  - `step4_postprocess.py` — drives the `viz2` (default) and/or `viz` suites,
    rebuilding phase fields on demand so figures are correct under V2.6's
    `save_fields=False` default.
  - `step5_analysis.py` — runs `mean_field`, `coarse_grain`, `descriptors` and
    `arch_distance` against a single run, collecting `analysis/summary.json`.
    Replaces `run_analysis_pipeline.py`, which was hardcoded to `DOE_01`–`DOE_23`.
  - `pipeline/_common.py` — shared state, guards, Params/trial-JSON loading
    (ported from `run_hpc_headless.py`), and console formatting.
- **Checkpointed progress**: `<run_dir>/pipeline_state.json` records each
  step's completion time and details. A step refuses to re-run without
  `--force` (exit 0, prints the override command) and exits 1 naming the
  prerequisite if run out of order. `--force` on steps 2 and 3 clears
  downstream snapshots and truncates history so a run can never resume from a
  stale state.
- **`pipeline/README.md`**: usage guide, run-directory layout, and notes.

### Bug Fixes
- **`is_circle` lost on resume** (`restore_gs_from_snapshot`): `GranuleSystem`
  marks any system constructed with explicit `a`/`b` as non-circular, and the
  restore path always passes them — so every resumed run silently dropped the
  circle fast-path. `is_circle` gates ~26 branches including contact detection,
  so a resumed packing of circles switched to the superellipse Newton-Raphson
  solver, and began emitting `shape_*` metrics mid-run. Now recomputed from the
  restored geometry (`a == b == r`, `n1 == n2 == 2`, and `c == r` in 3D).
  Measured 4.4× speedup on a resumed 2D circle run.
- **`save_history_to_disk` crash on heterogeneous history**: CSV fieldnames
  were taken from `hist[0]` alone, so a resumed run whose earlier entries had
  different keys raised `ValueError: dict contains fields not in fieldnames`
  *after* the simulation completed — losing both `history.csv` and the
  `history.json` written after it. Now uses the union of keys across all
  entries in first-seen order, with `restval=''`.
- **Numba compilation failure in `find_contact_superellipse_wall`**:
  `np.linspace(..., endpoint=False)` is unsupported in nopython mode, so any
  run with non-circular granules and wall boundaries aborted with a
  `TypingError` once JIT was active. Replaced with an equivalent `np.arange`
  form. The four other `endpoint=False` uses are in pure-Python functions and
  are unaffected.
- **Windows console encoding**: the engine prints `ν`, `µ` and `→`, which a
  default cp1252 console cannot encode, killing a run inside `print()`. The
  pipeline reconfigures stdout/stderr to UTF-8 with `errors='replace'`.

---

## [V2.6] - 2026-03-18

### Added (2D vs 3D DOE Comparison)
- **`viz/doe_2d_vs_3d.py`**: New comparison analysis script for matched 2D/3D DOE runs
  (same LHS design matrix, seed=2025). 7 analysis modules: paired scatter plots with OLS
  regression, dimensionless collapse (φ/φ_RCP, Z/Z_iso, K/d²), factor importance heatmaps
  (Pearson r, 2D vs 3D vs Δ), time evolution overlay (2D blue vs 3D red, mean ± σ),
  scaling regression table (3D = α + β·2D with R² bar chart), summary radar chart, and
  3D/2D ratio vs factor values (parameter-dependent scaling detection).
- **Key finding**: Permeability (K) is the only metric with strong 2D→3D collapse
  (R²=0.94, slope=0.83). Connectivity, bridges, and contacts show dimension-dependent
  scaling that cannot be reduced to a simple power law.

### Added (HPC Storage Protection & Resume)
- **Results to /groups**: New `results_path` field in `hpc/Alex.json` directs simulation
  output to `/groups/mcgheealex/results` (500 GB) instead of `/home` (50 GB). SLURM
  templates use absolute paths. All sync functions (`_do_sync`, `_wait_and_sync`,
  `_wait_and_sync_multi`, `sync_results.py`) updated to read from `results_path`.
- **Disk space guard**: `check_disk_space()` function checks available space before each
  snapshot save. < 5 GB → skip fields (save particle data only). < 1 GB → skip snapshot
  entirely. SLURM pre-flight bash check aborts job if < 2 GB free.
- **Fields off by default**: `save_fields` default changed from `True` to `False`. Phase
  field grids (phi_f/phi_i/phi_v) are no longer saved during simulation — they're
  reconstructable from particle positions at plot time. Roughly halves storage per run.
- **Staggered starts**: `max_concurrent` in Alex.json (default 10, was 200). Plus 0-30s
  random sleep per task to reduce I/O storms.
- **Resume from snapshot**: `resume_from` parameter in Params. `find_last_snapshot()` and
  `restore_gs_from_snapshot()` reconstruct full GranuleSystem from saved .npz. `run()`
  supports resume branch: skips packing, loads state, continues from last saved timestep.
  `--resume-from` CLI arg in `run_hpc_headless.py`.
- **Batch resume (RUN_MODE=4)**: `run_all_trials.py` mode 4 scans local results for
  incomplete trials, uploads last snapshot to HPC via `upload_for_resume()`, generates
  resume SLURM scripts, and submits.

### Added (DEM vs Contact Network Comparison)
- **`viz/doe_dem_vs_network.py`**: Validates stochastic contact-network model as a surrogate
  for expensive 3D DEM by running both on identical initial packings from the 20-point DOE.
  Custom ensemble: bypasses built-in `ensemble()` to use `from_packing()` with repeated
  `solve()` on the exact DEM snapshot packing. 5 analysis modules: paired scatter (DEM vs
  Network final values ± σ), time evolution overlay (DEM trajectories vs ensemble mean),
  predictability table (R², slope, DEM/Net ratio as CSV + bar chart), network-only metrics
  (percolation onset, cluster sizes, spanning probability), and summary dashboard (cost
  comparison + R² heatmap). Comparable metrics: contacts, bridges, coordination Z,
  permeability, porosity, bridge force, locked bridges. Network-only: percolation, cluster
  size, bridge fraction.

### Bug Fixes
- **Bridge exclusion cap in stochastic model**: `compute_bridge_rates()` in
  `contact_network_model.py` now enforces a per-edge bridge limit derived from
  `bridge_exclusion_angle` and hemisphere geometry (3D: 2/θ², 2D: π/(2θ)). Previously
  the bridge count per edge was only limited by cell availability, producing ~200× too many
  bridges in 3D (14,969 vs DEM's ~2,700).
- **Periodic BC position wrapping in ContactNetwork**: `_rebuild_edges()` in
  `contact_network.py` now wraps positions into [0, L) before building the periodic cKDTree.
  Previously, particles that drifted outside the domain caused `ValueError: Some input data
  are greater than the size of the periodic box`. Also added initial wrapping in
  `from_snapshot()`.
- **LS-DEM epsilon serialization**: `save_snapshot_to_disk()` now saves `epsilon` and
  `d_epsilon` arrays to the .npz file. Previously only the in-memory snapshot dict
  included them, so resumed deformable runs would lose deformation state.

---

## [V2.5] - 2026-03-17

### Added (Mesoscale Models)
- **`analysis/spatial_pde.py`**: Spatially-resolved 1D radial PDE model for scaffold
  compaction. Extends the mean-field ODE with N_x=20 grid points ξ ∈ [0,1] (center→edge).
  Captures compaction waves, local jamming fronts, heterogeneous porosity/permeability
  profiles. State: x_f(ξ,t) + φ_tissue(ξ,t). Physics: cell traction stress vs contact
  resistance + stress-driven diffusion, with zero-flux BCs. Includes fitting to DEM data
  (η_eff, σ_0, α), DEM-snapshot initialization via radial binning, and tissue descriptor
  computation. Extracted and refactored from `parameter_sweep.py` into standalone class.
- **`analysis/contact_network.py`**: Graph-based contact network model. Nodes = granules,
  edges = contacts/near-neighbors. Tracks bridge state machine (NONE → FORMING → ACTIVE →
  LOCKED/SENESCENT), Hertz + DMT contact forces, overdamped position updates. Spheres only
  (no superellipsoid solver — the key speedup). Captures: coordination Z(t), bridge
  percolation thresholds, cluster size distributions, force chain statistics. Supports
  Monte Carlo ensemble via `run_ensemble()`. Can initialize from DEM snapshot, random
  packing, or Params alone.
- **`analysis/contact_network_model.py`**: Stochastic contact-network model with Gillespie
  SSA for exact bridge kinetics. Key differences from `contact_network.py`:
  (1) Gillespie algorithm (Bortz-Kalos-Lebowitz 1975) replaces per-timestep probability
  evaluation — every bridge formation/senescence event advances physical time exactly.
  (2) Union-find percolation tracking (Newman-Ziff 2001) with O(N) cluster statistics
  and spanning detection.
  (3) Monte Carlo ensemble with statistical summary: mean, IQR, min/max for all observables
  including bridge count, Z_bridge, spanning probability, permeability.
  (4) Self-contained sphere packing via Lubachevsky-Stillinger inflate-and-relax.
  Physics: JKR adhesive contact, motor-clutch cell traction, path-dependent bridge
  probability (contact/void/inert-blocked), bridge lock-in, secondary migration boost.
  Runs ~2.5s per 72h realization (2D, ~43 granules). Interface: `from_params(p)` →
  `solve(model)` or `ensemble(model, n_runs=1000)`.
- **`viz/mesoscale.py`**: Visualization for both mesoscale models. PDE plots: kymograph of
  x_f(ξ,t), radial profiles at selected times, domain-averaged timeseries + stress balance.
  Network plots: granule+bridge snapshot, Z(t)/bridge/percolation/cluster evolution.
  Combined 4-panel summary showing spatial PDE + network topology together.

### Design (Mesoscale Architecture)
- Three complementary models fill the gap between full DEM and mean-field ODE:
  - **SpatialPDE** answers *where* compaction happens (spatial gradients, porosity maps)
  - **ContactNetwork** (deterministic) answers *how* the packing reorganizes (force chains)
  - **ContactNetworkModel** (stochastic) answers *what variability* to expect across
    realizations (percolation thresholds, spanning probability, ensemble statistics)
- All run in seconds (vs hours for DEM), all produce tissue descriptors for organ
  distance computation via existing `arch_distance.py`.
- Cross-feeding interface: PDE local φ_f(ξ) → expected Z(ξ) for network edges;
  Network percolation state → PDE effective viscosity η_eff(ξ).

---

## [V2.4] - 2026-03-17

### Added (viz2/ — V2 Visualization System)
- **New `viz2/` package**: Complete rebuild of 2D visualization as a unified, modular system.
  10 files, 6 analysis modules, all driven from scaffold map foundation.
- **`viz2/common.py`**: Centralised color schemes, drawing helpers (`draw_granule_patches`,
  `draw_cells_on_ax`, `cell_world_positions`), phase field re-rendering, and figure utilities.
  Eliminates ~150 lines of duplicated code per module vs the old `viz/` approach.
- **`viz2/scaffold_map.py`** (Module 1): Vector-drawn scaffold map (red=functional,
  green=inert, black=void) with cell overlays. Bridge cells drawn in crimson but counted
  as functional space in all data analysis. Single-frame API for movie compositing.
- **`viz2/voronoi.py`**: Voronoi tessellation engine with two modes — center-based (standard
  Voronoi from granule centers) and boundary-based "shrink-wrap" (seeds on superellipse
  surfaces, sub-cells merged per granule via ConvexHull). Pure-numpy Sutherland-Hodgman
  domain clipping (no shapely dependency). Shape factor computation (circularity, elongation,
  aspect ratio via PCA). Local phase fraction computation via point-in-polygon on phase
  field grid.
- **`viz2/voronoi_shapes.py`** (Module 2): Voronoi overlay on scaffold map, shape factor
  spatial maps (circularity/elongation/area), distribution histograms, and mean±std
  timeseries showing compaction trends in local geometry.
- **`viz2/phase_fractions.py`** (Module 3): Global phase fractions (stacked area +
  conservation check), local phase fraction heatmaps within Voronoi cells, inner-vs-outer
  timeseries showing compaction gradient, and local-vs-global heterogeneity scatter plots.
- **`viz2/energy_stress.py`** (Module 5): Stress/strain spatial maps via Gaussian
  coarse-graining (reuses `analysis.coarse_grain`). 6 energy mode spatial maps: cell
  traction, Hertz/JKR contact, granular friction, osmotic pressure, inert frustration,
  interfacial tension. Each computed on Ngrid×Ngrid grid with proper physics. Energy
  mode timeseries showing total per mode vs time.
- **`viz2/void_percolation.py`** (Module 6): Void cluster labeling (`scipy.ndimage.label`),
  spanning-cluster percolation detection, cluster size distribution P(s) with power-law
  fits, percolation evolution timeseries (status, largest fraction, count, mean size),
  Kozeny-Carman permeability tracking.
- **`viz2/movies.py`** (Module 4): GIF animations of scaffold map, Voronoi tessellation,
  local phase fractions, stress maps, and 6-panel energy modes. Frame-by-frame rendering
  via `imageio.mimsave()` with configurable FPS and max frames.
- **`viz2/postprocess.py`**: Unified orchestrator with `--skip` / `--only` filtering.
  Supports both disk-loaded data (`-i results/default`) and live simulation data.
  CLI: `python viz2/postprocess.py -i results/default [--skip movies] [--only scaffold]`.

### Added (viz2/ — 3D Analysis Extension)
- **3D infrastructure in `viz2/common.py`**: `is_3d()` mode detection, `slice_field_z_midplane()`
  for extracting 2D slices from (Ng,Ng,Ng) phase fields, `slice_snap_z_midplane()` for projecting
  3D granules to 2D cross-sections (superellipsoid slice radii `a*(1-(dz/c)^n2)^(1/n2)`, yaw
  from quaternion), `cell_world_positions_3d()` using `superellipsoid_point` + `quat_rotate`,
  extended `ensure_phase_fields()` with `render_fields_3d` support.
- **All 6 existing modules extended for 3D**: Each module detects `is_3d` and uses z-midplane
  slicing for 2D display while preserving full 3D analysis where appropriate (e.g.
  `void_percolation` checks percolation along all 3 axes, `energy_stress` slices coarse-grained
  3D fields at midplane).
- **New `viz2/scaffold_map_3d.py`** (Module 7): PyVista off-screen 3D rendering with
  superellipsoid meshes (parametric grid + pole caps), cell spheres colored by state, bridge
  tubes, z-midplane clip plane. Multi-panel composites assembled in matplotlib. Guarded by
  `HAS_PYVISTA`; silently skips when unavailable or data is 2D.
- **Voronoi all-granule tessellation** (`viz2/voronoi.py`): Both functional AND inert granules
  seed Voronoi. Functional cells clipped to granule boundary via Sutherland-Hodgman convex
  polygon intersection (`clip_polygon_to_convex`). Inert cells keep full Voronoi region.
- **Periodic boundary rendering** (`viz2/common.py`): Ghost images for granules/cells near
  domain edges via `_periodic_offsets`, minimum-image convention for bridge vectors via
  `_min_image`.
- **Verified on DOE_3D_0001** (433 granules, 4644 cells, 200^3 fields) and regression-tested
  on DOE_2D_0001 with 0 failures.

### Documentation
- **Mean field modeling teaching guide** (`CodeLog/MeanFieldModeling/MEAN_FIELD_GUIDE.md`):
  Comprehensive rewrite of the beginner's guide to mean field modeling. Now 11 sections
  covering: (1-5) progressive toy examples (coffee cooling, logistic growth, thermostat,
  Kozeny-Carman), (6) full GELS model derivation with worked code, (7) hands-on
  tutorial for fitting real DEM data with step-by-step walkthrough, (8) DOE sweep comparison
  across 19 trials, (9) scalar ODE to spatial PDE extension, (10) energy landscape
  thermodynamic view, (11) summary with 3-level model hierarchy table. Includes appendices
  with file reference, figure generation instructions, and quick-start cheat sheet.
- **Example figure generator** (`CodeLog/MeanFieldModeling/generate_examples.py`): Expanded
  from 10 to 14 examples. New real-data examples: (11) single DEM fit with 4-panel
  diagnostic, (12) DOE parameter comparison across all trials (compaction vs stiffness,
  composition, permeability, bridging), (13) multi-run trajectory overlay colored by
  stiffness, (14) energy landscape decomposition. All 14 figures generate successfully
  from `results/LHC/` DOE data.

### Added (Visualization)
- **Vector scaffold map visualization** (`viz/postprocess.py`): New `plot_scaffold_map()`
  function draws granules as exact matplotlib patches (circles/superellipses) colored by
  type — red=functional, green=inert, black=void — with cells overlaid using cell-state
  colors. Replaces the grid-based `plot_composite()` as the primary [4/17] pipeline output
  (`scaffold_map.png`). Eliminates all aliasing artifacts since rendering is vector-based
  rather than grid-discretized. Handles periodic boundary ghost images. Legacy composite
  still saved when phi fields are available.

### Bug Fixes
- **Rendering aliasing on large domains** (`new_dem_0.py`): Phase field rendering used a
  fixed `Ngrid=200` regardless of domain size, causing severe aliasing when domain ≫ 800 µm
  (e.g. Lx=2319 µm gave 11.6 µm/pixel vs 3 µm interface width — sub-pixel tanh transitions
  rendered as binary noise). Fix: `render_fields()` auto-scales Ng to ensure grid spacing
  ≤ 2× interface_width. `render_fields_3d()` auto-scales up to 200³ cap, and widens
  interface_width to grid spacing when voxels remain coarse. Affects all phi-field-derived
  metrics (connectivity, porosity, tissue fraction).
- **Effective radii ignored periodic boundaries** (`new_dem_0.py`): `compute_effective_radii()`
  and `compute_effective_radii_3d()` used non-periodic cKDTree and raw position differences,
  missing all cross-boundary overlaps. Fix: accepts optional `p` parameter; when
  `boundary_mode='periodic'`, uses `boxsize` and `minimum_image_disp`.
- **Metric pixel area used fixed Ngrid** (`new_dem_0.py`): `compute_metrics()` computed
  `dxg = p.Lx / p.Ngrid` but the actual rendering grid may be larger (auto-scaled). Fix:
  derives `dxg` from `phi_f.shape[0]`.
- **Wall contact clip normals inverted** (`new_dem_0.py`): JKR wall clip normals used the
  force direction `sign` (pointing into domain) instead of negated sign (pointing toward
  wall). This clipped the domain-facing side of wall-adjacent particles instead of the
  wall-facing side. Fix: `float(sign)` → `-float(sign)` in all 4 wall clip paths.

### Changed
- **2D packing: Lubachevsky-Stillinger inflate-and-relax** (`new_dem_0.py`): Replaced the old
  centripetal-attraction settle with the same deflate→inflate algorithm used by 3D. RSA now
  places granules at reduced radii (α = (φ_safe/φ_target)^(1/2)), then `_settle_packing_2d()`
  inflates to target with vectorised Hertz repulsion + velocity cap + post-inflation overlap
  relaxation. Achieves jammed 2D packing at Z ≈ 3–4, up from Z ≈ 0–1 with the old method.
  Uses existing params (`packing_settle_steps`, `packing_relax_substeps`,
  `packing_inflate_phi_safe`). No new parameters.

### Added
- **Experimental comparison analysis** (`analysis/experimental_comparison.py`): New module for
  comparing Day 1 experimental packing data (hand-segmented area fractions from confocal cross-
  sections) with simulation predictions and mean-field models. Parses Excel data (36 samples:
  9 size combinations × 4 functional ratios), computes derived quantities (enrichment, packing
  efficiency), and generates 9 diagnostic plots: phase fractions vs ratio, packing efficiency,
  phase enrichment, size ratio effects, phase continuity, continuity vs fraction (percolation
  analysis), experiment vs simulation prediction, ternary composition diagram, and size heatmap.
  Supports optional DEM packing runs for parity plots. CLI: `python analysis/experimental_comparison.py`.

## [V2.4] - 2026-03-16

### Changed
- **Delayed cell senescence with stacking tolerance** (`new_dem_0.py`): Overcrowded cells no
  longer go senescent immediately. Cells can stack up to `cell_stacking_max` layers (default 3.0)
  on a granule surface — they prefer crawling on granule surfaces but will crawl on each other
  when surface area runs out. Cells in the overcrowded-but-tolerated zone accumulate
  `cell_overcrowd_age`; senescence triggers only after `overcrowd_senescence_time` hours
  (default 12.0) of sustained overcrowding. If overcrowding resolves (e.g. cells migrate or
  bridge away), the timer resets. Cells beyond the max stacking capacity still go senescent
  immediately. New params: `cell_stacking_max`, `overcrowd_senescence_time`. New per-cell
  array: `cell_overcrowd_age` (serialized to snapshots).

---

## [V2.3] - 2026-03-16

### Changed
- **Path-dependent bridge probability** (`new_dem_0.py`): Bridge formation now depends on what
  lies between the cell and its target, modelled from a fibroblast's perspective. Three regimes:
  (1) **Contact** (gap ≤ 0): cells crawl between touching surfaces, boosted by
  `bridge_contact_factor` (default 5.0). (2) **Void gap**: filopodia must probe empty space,
  probability decays as `exp(-gap / bridge_decay_length)` (default λ=10 µm). (3) **Inert-blocked**:
  inert granule in line-of-sight further penalised by `bridge_inert_factor` (default 0.1).
  Replaces old linear proximity model (`1 - gap/sense_dist`). Ray-cast via vectorised
  bounding-sphere intersection classifies each functional–functional pair. Bridges can now
  form at contact (previously blocked by `gap > L_rest`). Service of committed bridges no
  longer gated by L_rest. New params: `bridge_contact_factor`, `bridge_decay_length`,
  `bridge_inert_factor`.

### Fixed
- **Granule pass-through bug** (`new_dem_0.py`): Added post-step overlap resolution
  (`_resolve_overlaps()`) to prevent granules from interpenetrating during simulation.
  Root cause: large timestep (dt=0.5h, max 10 µm/step) combined with sharp contact
  threshold (zero force at overlap ≤ 0) allowed approaching granules to jump past contact
  detection, especially when bridge forces matured at t≈3-7h. Fix: after each position
  integration, all pairs are checked via cKDTree bounding-sphere query. Pairs with overlap
  exceeding `max_overlap_frac × min(r_i, r_j)` (default 15%) are projected apart by half
  the excess. Handles 2D/3D and periodic/wall BCs. New parameter: `Params.max_overlap_frac`.

### Added
- **LS-DEM deformable particles** (`lsdem.py`, `new_dem_0.py`): Variational level-set DEM
  implementation following Henzel & Karapiperis 2026 (arXiv:2602.12895). Each particle gets
  per-particle scalar deformation DOFs ε_α(t) that modulate fixed spatial mode shapes Φ_α(x).
  Particle geometry tracked via pre-computed signed distance fields (SDF) on body-frame grids.
  Contact detection uses surface-node-to-SDF queries (replaces Newton-Raphson for deformable
  particles), yielding multi-point contact patches. Overdamped deformation dynamics:
  γ_def dε/dt + K ε = F_ε, integrated with unconditionally stable implicit Euler.
  New module `lsdem.py` contains all LS-DEM functions; main engine hooks in via conditional
  branches when `deformable_enabled=True`. Default is `False` (zero performance impact on
  existing V2.2 rigid simulations).
  - **SDF infrastructure**: `superellipse_approx_sdf()` / `superellipsoid_approx_sdf()` using
    gradient-normalized implicit function. Pre-computed on body-frame grids via `init_sdf_grids()`.
    Bilinear/trilinear interpolation (`query_sdf_2d/3d`) and gradient queries for contact normals.
  - **Surface node discretization**: Uniform parametric sampling (2D) or Fibonacci sphere
    projection (3D). Per-node area weights for force integration. Hemisphere culling eliminates
    ~50% of SDF queries.
  - **Deformation modes**: Mode 0 = axial compression (volume-preserving), Mode 1 = prolate-oblate
    shape change. Mode 2 (3D only) = volumetric (very stiff for ν≈0.49). Elastic stiffness
    K_αβ derived from linear elasticity with shear modulus G and bulk modulus K.
  - **Semi-Lagrangian SDF update**: Deformed geometry evaluated on-demand via inverse displacement
    mapping: φ(x,t) = φ₀(x − Σ ε_α Φ_α(x)). No grid recomputation needed.
  - **Deformation metrics**: `def_strain_mean/max/std`, per-mode statistics tracked in metrics.
  - New Params: `deformable_enabled`, `n_def_modes`, `sdf_resolution`, `n_surface_nodes`,
    `n_surface_nodes_3d`, `sdf_padding`, `def_drag_scale`, `def_eps_max`.
  - MC-DEM automatically disabled when `deformable_enabled=True` (LS-DEM captures multi-contact
    stiffening geometrically).
  - **Deformed rendering** (`new_dem_0.py`): `render_fields()` and `render_fields_3d()` now
    stamp deformed particle shapes when `deformable_enabled=True`, using semi-Lagrangian
    pull-back through the mode-shape displacement field. Grid points are mapped to each
    particle's body frame, the inverse deformation u(x) = Σ ε_α Φ_α(x) is subtracted, and
    the analytical implicit function is evaluated at the pulled-back coordinate. New functions
    `_stamp_deformed_2d()` and `_stamp_deformed_3d()`. Shows physically flattened contacts
    and shape changes in rendered phase fields.
  - **Deformation visualization** (`viz/postprocess.py`): Two new plot functions.
    `plot_deformation()` draws particles colored by total deformation strain |ε| using the
    inferno colourmap with a shared colourbar across timepoints. `plot_deformation_timeseries()`
    shows mean/max strain over time and per-mode amplitude breakdown with ±1σ bands. Both
    produce output only when deformation data is present (gracefully skip for rigid runs).
    Postprocess orchestrator updated from 15 to 17 steps. New skip key: `'deformation'`.
  - **JKR adhesive contact model** (`new_dem_0.py`): Replaced Hertz + DMT adhesion with
    Johnson-Kendall-Roberts (JKR) model throughout all force computation paths (2D rigid,
    3D rigid, 2D walls, 3D walls). JKR is correct for soft hydrogels where the Tabor
    parameter μ_T >> 1. New functions: `jkr_force_from_overlap()` (Newton-Raphson solve for
    contact radius from overlap), `jkr_contact_radius()` (analytical), `jkr_pulloff_force()`.
    JKR reduces exactly to Hertz when W_adhesion = 0 (verified numerically). Force:
    F = (4/3)E*a³/R* − √(8πWE*a³). All existing adhesion parameters (W_adh_ff, W_adh_if,
    W_adh_ii) now drive JKR instead of DMT. No new parameters required.
  - **Contact-face half-plane clipping** (`new_dem_0.py`): Per-particle contact clip planes
    stored during force computation (`gs.contact_clips`). Each contact generates a half-plane
    (2D) or half-space (3D) constraint: `clip_d = √(R² − a²)` where `a` is the JKR contact
    radius. During rendering, clip planes are applied as `sdf = max(sdf, plane_sdf)` to create
    flat faces at contact regions, eliminating unrealistic overlapping circles. New helper
    functions `_apply_clips_2d()` and `_apply_clips_3d()`. All stamp functions updated:
    `_stamp_circle_2d()`, `_stamp_superellipse_2d()`, `_stamp_deformed_2d()`,
    `_stamp_granule_3d()`, `_stamp_deformed_3d()`. Rendering functions `render_fields()` and
    `render_fields_3d()` pass per-particle clips to stamp calls.
  - **LS-DEM adhesion** (`new_dem_0.py`): JKR adhesion for LS-DEM deformable contacts applied
    as translational-only correction at the pair level in `compute_forces()` / `compute_forces_3d()`.
    After LS-DEM computes pure repulsive penalty forces (which drive deformation DOFs), the
    adhesion correction F_adh = F_JKR − F_Hertz is applied to translational forces only, keeping
    it out of deformation generalized forces F_ε. This prevents adhesion from driving unphysical
    deformation while correctly pulling particles together. Surface nodes in `lsdem.py` remain
    purely repulsive.
  - **Multi-stage bridge formation** (`new_dem_0.py`, `viz/cells.py`): New `MIGRATING` state
    (CellState value 5) implements directed cell crawling before bridge formation. Lifecycle:
    PROLIFERATING → MIGRATING (directed crawl on granule surface) → BRIDGING (force ramp) →
    mature bridge. **Cell spatial awareness**: hemisphere check ensures only cells on the
    target-facing side of a granule can attempt bridges. Probability decays from cell centroid
    position (not granule center) using cell-specific gap computation. **Directed migration**:
    MIGRATING cells move along the host granule surface toward the contact point at
    `cell_migration_speed × bridge_directed_speed_mult` (default 2×). Cells within
    `bridge_commit_angle` (default 0.5 rad ≈ 29°) of the contact azimuth fast-path directly
    to BRIDGING. Abandonment: if target leaves sensing range, cell reverts to PROLIFERATING.
    2D: arc stepping via `cell_theta_local`. 3D: slerp on parametric unit sphere via
    `cell_eta_local`/`cell_omega_local`. New helper functions: `_cell_world_pos_2d/3d()`,
    `_compute_gap()`, `_angular_distance_to_target_2d/3d()`. New Params:
    `bridge_commit_angle=0.5`, `bridge_directed_speed_mult=2.0`. New metric:
    `n_migrating_cells`. Visualization: royal blue (#4169E1) for MIGRATING cells.
  - **Bridge exclusion zone** (`new_dem_0.py`): Cells avoid forming bridges near existing
    BRIDGING or MIGRATING cells on the same granule→target pair. Before attempting a bridge,
    each candidate cell's angular position is compared to all existing bridge cells on the
    same granule targeting the same neighbor. If any are within `bridge_exclusion_angle`
    (default 0.8 rad ≈ 46°), the cell skips and prefers to crawl around the granule to find
    an unoccupied region. Works in both 2D (theta arc distance) and 3D (great-circle distance
    on parametric sphere). New Param: `bridge_exclusion_angle=0.8`.
  - **Post-bridge stress fiber alignment** (`new_dem_0.py`): After entering BRIDGING state,
    fibroblasts progressively align their stress fibers and elongate along the bridge axis,
    increasing force output over time. New per-cell `cell_alignment` field (0–1) initialized
    at `bridge_alignment_min` (default 0.3) on BRIDGING entry, growing exponentially toward
    1.0 at rate `bridge_alignment_rate` (default 0.3 /h). Bridge force modulated as
    `F = F_mc × maturity × alignment`, giving a two-stage buildup: initial adhesion ramp
    (maturity, 0→1 over 2h) followed by continued force amplification as alignment improves
    (0.3→1.0 over ~6–8h more). Total time to full force ≈ 8–10h, matching real fibroblast
    remodeling timescales. Alignment resets on bridge rupture or senescence. New Params:
    `bridge_alignment_rate=0.3`, `bridge_alignment_min=0.3`. New per-cell array:
    `cell_alignment` (serialized to snapshots).

---

## [V2.2] - 2026-03-16

### Added
- **Multi-Contact DEM (MC-DEM) stiffening** (`new_dem_0.py`): Stress-based multi-contact
  correction for Hertzian contact forces (Giannis et al. 2021). When a soft granule has
  multiple simultaneous contacts, each contact sees a stiffer response due to volumetric
  confinement. Computes per-particle overlap strain ε_V = Σ δ/(2R), then scales Hertz
  repulsion by κ = 1 + ν/(1−2ν) × ε_V. For hydrogels at ν=0.49, this gives ≈2× stiffening
  for typical jammed packings (7 contacts at 1% overlap). Adhesion and friction unaffected.
  New function `mc_dem_correction()` applied as post-correction in both 2D `compute_forces()`
  and 3D `compute_forces_3d()`. New Params: `mc_dem_enabled=True`, `mc_dem_kappa_max=5.0`.
- **Comprehensive literature references** (`CodeLog/References/REFERENCES.md`): Added §20
  (superellipsoid packing & jamming: Donev 2004, Delaney/Cleary 2010, Yuan/O'Hern 2019,
  Jiao/Torquato 2009/2010), §21 (MC-DEM: Giannis 2021, Brodu 2015, Ghods 2022, Hirsch 2022),
  §22 (contact detection: Wellmann 2008, Canelas 2025, Lai 2022/2024, Gouveia 2025, Feng 2023),
  §23 (deformable DEM future: Rojek 2018/2021, Henzel/Karapiperis 2026, Feng 2025).
- **V3.0 LS-DEM upgrade roadmap** (`CodeLog/ClaudesPlan/V2.2_mc_dem_and_future_ls_dem.md`):
  Phased plan for deformable particle simulation via variational level-set DEM
  (Henzel & Karapiperis 2026). Four phases: SDF contact → level-set representation →
  deformation DOFs → new contact formation.

---

## [V2.1] - 2026-03-16

### Added
- **Cell surface coverage parameter** (`new_dem_0.py`): New `Params.cell_surface_coverage`
  (float, 0.5–1.5 typical) specifies cell loading as a fraction of granule surface area.
  At coverage=1.0 cells form a full monolayer; values >1.0 allow stacking (cells on top
  of each other). When set (>0), overrides `n_cells_per_granule` and `cell_coverage`.
  New helper `cells_from_surface_coverage()` computes per-granule cell count from 3D
  surface area (4piR^2) or 2D projected area (piR^2). Packing and overcrowding logic
  both respect the new parameter.
- **Two-stage adaptive DOE** (`Trials/generate_doe.py`, `Trials/generate_doe_stage2.py`):
  Replaced 5^7=78,125 full factorial with a two-stage adaptive strategy.
  - **Stage 1** (`generate_doe.py`): 500-run maximin Latin Hypercube Sampling across 7
    continuous factors: E_modulus (2–50 kPa), phi_solid_target (0.55–0.85), func_ratio
    (0.2–1.0), cell_surface_coverage (0.5–1.5), cell_sense_distance (20–80 µm),
    R_func_mean (30–150 µm), R_inert_mean (30–150 µm). ~9k CPU-hours (6% Puma monthly).
  - **Stage 2** (`generate_doe_stage2.py`): ~300 adaptive refinement runs generated after
    Stage 1 analysis. Fits quadratic response surfaces, computes variance-based sensitivity
    indices, and allocates points via four strategies: near-optimal (35%), steep-gradient
    (25%), high-uncertainty (20%), space-filling (20%). ~5.4k CPU-hours (4% Puma monthly).
  - Total budget: ~800 runs, ~14.4k CPU-hours (10% monthly allocation) vs 78,125 runs prior.
- **Tissue volume tracking** (`analysis/parameter_sweep.py`): New spatially-resolved state
  variable `phi_tissue(xi, t)` tracks cell + ECM volume fraction that grows within the
  functional zone of the scaffold. Cells adhered to functional granules occupy void space,
  and bridges between granules scaffold further tissue growth.
  - **Logistic growth ODE**: `d(phi_tissue)/dt = k_tissue × min(1, n_cells/n_ref) × maturity(t)
    × max(alpha_0, f_bridge × neighbor_factor) × max(0, alpha_fill × phi_void − phi_tissue)`.
    Driven by cell count, FA maturity, and bridge formation. No feedback into compaction
    mechanics (tissue is soft).
  - **TISSUE_PARAMS**: `k_tissue=0.05 h⁻¹` (base rate), `alpha_tissue_fill=0.6` (max void
    fill fraction), `alpha_tissue_0=0.1` (minimum growth without bridges),
    `n_cells_tissue_ref=10.0` (reference cell count).
  - **Tissue-corrected architecture descriptors**: `BV_TV_eff = phi_f + phi_tissue_global`,
    `porosity_eff = 1 − BV_TV_eff`, `K_f_tissue` via Kozeny-Carman with tissue-reduced void.
    Organ distance computation now uses these corrected values.
  - **Updated `_integrate_trajectory()`**: Returns 6-tuple (added tissue_traj, tissue_spatial)
    for single-sample kinetics visualization.
  - **Updated plots**: `plot_best_kinetics()` expanded to 1×3 panels (tissue volume vs time),
    `plot_radial_profiles()` expanded to 1×3 panels (tissue spatial profile).
  - **New `plot_tissue_effects()`**: 2×2 figure — (a) BV_TV_eff vs phi_f, (b) K_f_tissue vs
    K_f, (c) phi_tissue vs n_cells, (d) phi_tissue vs func_ratio.
  - **Updated recommendation table**: Added phi_tissue_global and BV_TV_eff columns.
  - **Updated radar chart**: Uses BV_TV_eff, porosity_eff, K_eff_tissue for organ comparison.
  - **Physical insight**: `n_cells_per_func` now affects architecture (not just compaction
    force). Dense organs (cardiac, liver) match better with tissue; porous organs (lung, bone)
    require few cells to avoid over-filling void space.
- **Lubachevsky-Stillinger jammed packing** (`new_dem_0.py`): Rewrote 3D packing initialization
  to guarantee fully jammed initial state (Z ≈ 6–8 contacts/granule). RSA places granules at
  deflated radii (α ≈ 0.6–0.7), then inflate-and-relax algorithm grows them to target size
  over 400 steps × 15 sub-steps with vectorized Hertz-like repulsion. Quadratic ease-in
  inflation schedule spends more time near final size where jamming is hardest. Replaces old
  centripetal-attraction approach that couldn't achieve high packing fractions.
  - New Params: `packing_inflate_phi_safe=0.20` (initial deflated packing fraction for RSA).
  - Updated defaults: `packing_settle_steps=400`, `packing_relax_substeps=15`.
- **Persistent bridge lock-in** (`new_dem_0.py`): Bridges that exceed `bridge_lock_force_threshold`
  now set a persistent `cell_bridge_locked` boolean flag. Once locked, bridges persist
  indefinitely regardless of instantaneous force fluctuations. Previously, force drops below
  threshold on any timestep would reset lock status, causing unrealistic bridge cycling.
  Lock flag is reset only on gap rupture (bridge physically breaks). Serialized in snapshots.
- **DOE periodic BC optimization** (`Trials/generate_doe.py`): Rewrote DOE generator for
  periodic boundary conditions. Key changes:
  - Switched BASE config to `boundary_mode: "periodic"`, `boundary_exclusion: 0.0`.
  - Reduced N_TARGET from 500 to 250 (periodic BCs eliminate boundary exclusion waste).
  - Added N_MAX=600 cap to prevent compute blowup from extreme size ratios.
  - Tightened R bounds from [30,150] to [40,120] µm to avoid heavy-tail compute distribution.
  - New `compute_domain_size()` enforces three constraints: N_target (particle count),
    minimum image (L > 2×cutoff for cKDTree periodic correctness), and RVE (L > 4×d_max
    for bulk packing statistics). Takes max of all three.
  - Reduced `Ngrid_3d` from 80 to 60 (3.4× faster field rendering).
  - DOE metadata now includes `_doe_n_target` and `_doe_constraint` per run.
  - Estimated ~5k CPU-hours for 500 runs (down from ~9k).
- **References** (`CodeLog/References/REFERENCES.md`): Added §18 Parisi & Zamponi (2010)
  "Mean-field theory of hard sphere glasses and jamming" and §19 Campello & Cassares (2016)
  "Rapid generation of particle packs at high packing ratios for DEM simulations."
- **Two-compartment volume conservation analysis** (`analysis/volume_conservation.py`):
  New module implementing Voronoi shrink-wrap compartment analysis for verifying global
  volume conservation. Assigns each voxel to the functional or inert compartment via
  nearest-granule cKDTree tessellation (periodic-BC aware). Uses exact granule geometry
  volumes (not rendered phase fields) for conservation accounting — `phi_solid_true` is
  provably constant to machine precision. Key functions: `voronoi_compartments()`,
  `compartment_metrics()`, `conservation_from_run()`. Generates 4-panel conservation
  figure (global conservation, compartment volumes, local void redistribution, cross-check)
  and compartment boundary slice overlay. Integrated as step 15/15 in `viz/postprocess.py`.
- **True granule-based volume metrics** (`new_dem_0.py`): Added `phi_f_true`, `phi_i_true`,
  `phi_solid_true` to `compute_metrics()` output. Computed from exact granule geometry
  (sphere: 4/3 pi R^3, superellipsoid: `superellipsoid_volume(a,b,c,n1,n2)`) — invariant
  during simulation since radii/shapes do not change. Provides a rendering-independent
  ground truth for volume conservation verification.
- **Inline Voronoi two-compartment metrics** (`new_dem_0.py`): Added per-timestep compartment
  tracking (x_f, x_i, phi_v_in_func, phi_v_in_inert, phi_solid_func, phi_solid_inert,
  phi_v_crosscheck) to `compute_metrics()` using the same Voronoi nearest-granule approach
  on the phase field grid.
- **Additive phase field blending** (`new_dem_0.py`): Changed 2D and 3D field stamping from
  `np.maximum()` (max-blending) to `+=` (additive blending). Max-blending loses volume when
  same-type granules overlap; additive blending preserves total deposited volume.
- **Iterative volume normalization** (`new_dem_0.py`): `render_fields_3d()` and `render_fields()`
  now apply iterative scale-and-cap normalization (up to 20 iterations) to match rendered
  phase field totals to true granule volumes. Residual error is <1% for typical packings
  (fundamental limit from per-voxel cap at 1.0 in dense overlap regions).

### Bug Fixes
- **Periodic boundary bridge visualization** (`viz/cells.py`): Fixed `_nearest_surface_point()`
  to apply minimum image convention when computing bridge target surface points. Previously,
  bridges crossing periodic boundaries were drawn spanning the entire domain width. Now correctly
  renders short bridges that visually terminate at domain edges.
- **Critical: phi_solid_target not applied at runtime** (`new_dem_0.py`): Fixed a bug where
  `phi_solid_target` and `func_ratio` overrides loaded from trial JSON configs were not
  propagated to `phi_f_target`/`phi_i_target`. The `Params.__post_init__` derivation runs at
  construction time, but trial configs apply overrides via `setattr()` after construction,
  leaving phi targets at defaults (0.25/0.20 = 0.45 total) instead of the intended values
  (0.55–0.85). Fix: added re-derivation at the start of `run()` so phi targets always
  reflect the current `phi_solid_target × func_ratio`. This affected all DOE runs using the
  `phi_solid_target` specification path.

---

## [V1.12] - 2026-03-16

### Documentation
- **REFERENCES.md**: Added 28 new literature references for energy landscape framework,
  organized into 9 new sections (§10–§18): Jamming Physics & Yield Stress (O'Hern, van Hecke,
  Olsson & Teitel, Tighe), Random Close Packing (Torquato, Farr & Groot, Donev, Yuan),
  Energy Landscape Theory (Wales, Kramers, Bi et al., Shi et al.), Poroelasticity (Biot,
  Brinkman), Interfacial Tension (Princen 1979/1983/1986, Durian), Polymer Dynamics
  (Doi & Edwards), Granular Hydrogel Mechanics (Cai et al., Di Caprio et al.), Tissue
  Architecture (Harrigan & Mann, Hildebrand & Ruegsegger, Hollister, Jaklic & Leonardis).
  Summary table §18 added for energy landscape parameter sources.
- **MATHEMATICAL_MODEL.docx**: Regenerated from updated markdown source with all new
  sections 11–14 (volume-conserving two-zone model, free energy landscape, computational
  results, design rules). Analysis figures inserted (110 total).

### Added
- **Jamming constraint** (`analysis/parameter_sweep.py`): Added physical requirement that
  initial solid packing must be jammed (`phi_solid >= phi_RCP`) to justify ignoring
  gravitational effects. Without jamming, granular scaffolds would collapse under gravity.
  - `phi_solid` sampling range narrowed from [0.10, 0.92] to [0.75, 0.95] to focus on the
    jammed regime and reduce wasted samples.
  - Feasibility filter in `run_sweep()` now computes `phi_RCP` before filtering and rejects
    samples below the shape-dependent jamming threshold.
  - New plot: `jamming_phase_space.png` — three-panel figure showing (a) phi_solid vs phi_RCP
    scatter with jamming boundary, (b) accessible BV/TV range with organ target lines, and
    (c) jamming margin distribution histogram.
  - Jamming boundary annotations (dashed line at phi_RCP=0.82) added to `compaction_heatmaps`
    and `organ_landscapes` plots where phi_solid is an axis.
  - Solid phase balance plot slices updated to reflect jammed phi_solid range [0.80-0.92].
  - Key physical insight: low-BV/TV organs (lung, bone) remain accessible in the jammed
    regime via low functional ratio (mostly inert granules provide structural jamming while
    sparse functional granules define tissue architecture).

---

## [V1.11] - 2026-03-16

### Added
- **Energy landscape analysis** (`analysis/energy_landscape.py`): New `EnergyLandscape`
  class decomposes the free energy of cell-driven scaffold compaction into six physically
  motivated terms as a function of compaction coordinate ξ = 1 − x_f/x_{f,0}:
  (1) G_cell — cell traction + bridge adhesion (negative, drives compaction, density-
  enhanced via ln(1−ξ));
  (2) G_elastic — Hertzian elastic contact (ODE-consistent parabolic onset at ξ_c where
  φ_local = φ_RCP);
  (3) G_yield — Herschel-Bulkley yield barrier near jamming (σ_y ~ σ_0·(φ/φ_J−1)^1.5);
  (4) G_void — osmotic void redistribution (quadratic in Δφ_v);
  (5) G_inert — geometric frustration from inert obstacles (quadratic in ξ);
  (6) G_surface — interfacial tension at functional/inert boundary (γ depends on modulus
  mismatch, favours compact round zones, penalises trapped inert granules).
  Provides: `find_equilibrium()`, `find_barrier()`, `solve_kinetics()` (overdamped
  dynamics with time-dependent bridge formation), `dimensionless_groups()` (β, Ca, Φ_r,
  Ψ, Γ), and `energy_decomposition_at_eq()`. Energies normalised by σ_cell for O(1)
  landscape structure. Calibrated to reproduce mean-field ODE equilibrium.
- **Energy landscape visualization** (`viz/energy_landscape.py`): Four publication-quality
  figures from the energy landscape analysis:
  (1) `energy_landscape_organs.png` — 2×2 grid showing G̃(ξ) with all 6 decomposed terms
  for 4 organ targets, marking equilibrium ξ* and barriers;
  (2) `energy_landscape_evolution.png` — time-evolving landscape as bridges form (t=0→72h)
  with kinetics trajectory ξ(t);
  (3) `energy_decomposition.png` — stacked bar chart of driving vs resisting energy terms
  at ξ* per organ, plus governing dimensionless numbers;
  (4) `energy_design_space.png` — heat maps of ξ* over (E_func, φ_f), (φ_i, R_func),
  and (E_func, E_inert) parameter planes for design rule extraction.
  CLI: `python viz/energy_landscape.py -i results/parameter_sweep`.
- **Timelapse temporal coherence** (`viz/scaffold_evolution.py`): Replaced per-frame
  independent position computation with frame-to-frame state propagation using damped
  motion (`_advance_positions()`) and pre-computed rendering (`_render_frame_from_pos()`).
  Each frame inherits positions from the previous frame and moves 18% toward the target,
  with gentle overlap resolution (10 iterations), eliminating unnatural inter-frame
  jostling of granules.

---

## [V1.10] - 2026-03-16

### Added
- **Periodic boundary conditions** (`new_dem_0.py`): New `Params.boundary_mode` parameter
  (`"walls"` default or `"periodic"`). When periodic, granule interactions use the minimum
  image convention via `scipy.spatial.cKDTree(boxsize=...)`, positions wrap via modulo, and
  wall forces are disabled. Eliminates artificial void accumulation near boundaries during
  compaction, enabling bulk scaffold property studies.
  - Packing (RSA + settle): Periodic placement in `[0, Lx)` with minimum-image overlap check.
    Settle phase uses random jitter instead of centripetal attraction.
  - Force computation: `compute_forces()` and `compute_forces_3d()` use periodic cKDTree,
    minimum-image displacements, and virtual positions for superellipse/superellipsoid contacts.
  - Step integration: Position wrapping via `wrap_positions()` instead of boundary clipping.
    Unwrapped positions (`x_unwrap`, `y_unwrap`, `z_unwrap`) track true displacement.
  - Field rendering: Ghost particle images stamped at periodic boundaries for correct phase fields.
  - Metrics: `boundary_exclusion` forced to 0 for periodic (full domain used). Contact
    diagnostics and bridge counting use periodic tree + minimum image.
  - Displacement: `compute_displacement()` uses unwrapped positions for correct RMS displacement.
  - Helper functions: `minimum_image_disp()`, `minimum_image_disp_3d()`, `wrap_positions()`.
  - Backward compatible: `boundary_mode='walls'` produces identical results to V1.9.

- **1D radial PDE mean-field model** (`analysis/parameter_sweep.py`): Replaced scalar ODE
  `x_f(t)` with spatially-resolved 1D PDE `x_f(xi, t)` on N_x=20 radial grid points,
  where xi is a normalized radial coordinate (center=0, edge=1).
  - **Neighbor-count modifier**: `neighbor_factor(xi) = 0.5*(1 + cos(pi*xi))` — cells at
    center sense all neighbors (factor=1), cells at edge sense none (factor=0). This
    naturally produces a dense core with dilute periphery.
  - **Stress-driven diffusion**: `D_eff * d²x_f/dxi²` with zero-flux BCs at xi=0 (symmetry)
    and xi=1 (edge). CFL stability clamped: `D_eff * dt / dxi² < 0.45`.
  - Domain-averaged outputs remain backward compatible with V1.9 sweep results.
  - New spatial outputs: `x_f_final_std` (radial heterogeneity), `x_f_gradient` (edge-center
    difference), `phi_v_f_local_std` (void fraction variability), `K_f_series` (harmonic mean
    permeability for radial flow).
  - `_integrate_trajectory()` updated to return spatial profiles for radial profile plotting.
  - New `plot_radial_profiles()` function: shows x_f(xi) and phi_v(xi) at final time for
    each organ's optimal scaffold.

---

## [V1.9] - 2026-03-16

### Added
- **Bridge force lock-in** (`new_dem_0.py`): Bridging fibroblasts whose force exceeds
  `bridge_lock_force_threshold` (default 20 nN) now prefer to stay in the bridge
  configuration indefinitely, bypassing the senescence timer. This models the
  biological observation that high-tension bridges stabilize rather than turn over.
  New parameter: `Params.bridge_lock_force_threshold`.
- **Secondary bridge migration** (`new_dem_0.py`): Non-bridging fibroblasts can migrate
  along existing bridges to form additional connections. When committed bridges already
  exist between two granules, the bridge attempt rate for new cells is boosted by
  `bridge_secondary_rate_mult` (default 3×). New parameter:
  `Params.bridge_secondary_rate_mult`.
- **Per-cell bridge force monitoring** (`new_dem_0.py`): Tracks force magnitude of every
  bridging cell at each save point. New history metrics: `n_bridging_cells`,
  `bridge_force_mean`, `bridge_force_max`, `bridge_force_min`, `bridge_force_std`,
  `n_locked_in_cells`. Run output now shows `bCells`, `bF_avg`, `lock` columns.
- **Bridge force diagnostic** (`new_dem_0.py`, `analysis/mean_field_model.py`): At end
  of simulation, compares measured bridge forces to `expected_bridge_force` (default
  100 nN) and warns if motor-clutch parameters may need tuning. New parameter:
  `Params.expected_bridge_force`.
- **Mean-field model updates** (`analysis/mean_field_model.py`): Bridge fraction now
  accounts for lock-in (no senescence turnover when F_cell ≥ threshold) and secondary
  migration rate boost. New constructor parameters: `bridge_senescence_time`,
  `bridge_lock_force_threshold`, `bridge_secondary_rate_mult`, `expected_bridge_force`.
- **`analysis/parameter_sweep.py`**: Mean-field parameter sweep and organ target prediction.
  Sweeps 6 physical parameters (E_modulus, phi_f_target, phi_i_target, R_func_mean,
  n_cells_per_granule, bridge_attempt_rate) through 50,400 combinations using the
  mean-field ODE model. Physics-based scaling laws derive fitting parameters (eta_eff,
  sigma_0, alpha) from granule mechanics without requiring DEM calibration data.
  Computes scaffold descriptors (BV/TV, porosity, Kozeny-Carman permeability) and
  architectural distance to all 7 organ targets for each combination.
  Outputs: `sweep_data.csv` (45K+ rows), `recommendations.json`, and 10 publication-
  quality figures including compaction heatmaps, solid phase balance diagrams,
  compaction driver analysis, organ distance landscapes, recommendation tables,
  radar comparisons, sensitivity tornado charts, closest-organ phase map,
  permeability-porosity space, and compaction kinetics for optimal scaffolds.
  CLI: `python analysis/parameter_sweep.py [-o DIR] [--quick] [--t-total H]`.
- **`viz/scaffold_evolution.py`**: 2D spatial maps showing time evolution of
  granular scaffold microstructure under cell-driven compaction. For 4 organ
  targets (trabecular bone, intestinal mucosa, kidney cortex, cardiac muscle),
  generates a packed 2D domain of superellipse granules (functional + inert)
  and applies the mean-field compaction trajectory to animate clustering over
  5 time snapshots (0, 6, 18, 36, 72 h). Shows cell bodies, cell bridges, and
  void redistribution. Companion kinetics plot shows x_f(t), local void
  fractions, and bridge fraction per organ.
  Outputs: `scaffold_evolution.png`, `scaffold_kinetics.png`.
  CLI: `python viz/scaffold_evolution.py -i results/parameter_sweep`.
- **`viz/scaffold_evolution_3d.py`**: 3-D volumetric version of the scaffold
  evolution visualization using PyVista off-screen rendering. Generates
  superellipsoid granule packings in a 3-D cubic domain, applies mean-field
  compaction trajectories, and renders semi-transparent isometric views showing
  interior clustering. Cell bridges rendered as tubes, cells as small spheres.
  Output: `scaffold_evolution_3d.png`.
  CLI: `python viz/scaffold_evolution_3d.py -i results/parameter_sweep`.
- **Packing constraint** (`new_dem_0.py`): Added `phi_solid_target` and `func_ratio`
  parameters as an alternative to setting `phi_f_target`/`phi_i_target` independently.
  When `phi_solid_target > 0`, derives `phi_f_target = phi_solid * func_ratio` and
  `phi_i_target = phi_solid * (1 - func_ratio)`, ensuring physically consistent total
  solid fractions (typically 0.55–0.75). Backward compatible: existing trials using
  `phi_f_target`/`phi_i_target` directly continue to work unchanged.
- **Parameter sweep packing physics** (`analysis/parameter_sweep.py`): Replaced independent
  `phi_f`/`phi_i` LHS sampling with constrained `phi_solid` (0.10–0.92) and `func_ratio`
  (0–1), deriving `phi_f = phi_solid * func_ratio` and `phi_i = phi_solid * (1 - func_ratio)`.
  Wide phi_solid range covers all 7 organ porosity targets: cardiac muscle (0.12) through
  lung alveoli (0.88). Initial x_f_0 clamped to deformable limit when initial packing
  already exceeds phi_max. Sweep now accepts configurable motor-clutch and bridge parameters
  via CLI.
- **Volume-conserving two-zone mean-field model** (`analysis/mean_field_model.py`,
  `analysis/parameter_sweep.py`): Fundamental physics rewrite. Granules are incompressible
  so phi_f + phi_i = phi_solid = CONSTANT. State variable changed from phi_f (which
  incorrectly decreased) to x_f (functional zone volume fraction). As cells compact,
  x_f shrinks → functional granules pack tighter (approaching RCP) → void expelled from
  functional zone → inert zone void increases. Global void fraction stays constant.
  Outputs: x_f_final, phi_v_f_local, phi_v_i_local, compaction_ratio, zone-weighted
  effective permeability. Organ distance uses heterogeneous pore structure.
  All 11 plots updated for new physics.
- **Deformable packing limit** (`analysis/parameter_sweep.py`, `analysis/mean_field_model.py`):
  Soft, deformable superellipsoid granules can now pack beyond the rigid-particle RCP. The
  hard floor `x_f_min = phi_f / phi_RCP` replaced with `x_f_min = phi_f / phi_max`, where
  `phi_max > phi_RCP` depends on granule compliance (1/E, soft = more packable) and
  superellipsoid blockiness (n2 > 2 = flatter faces = tighter interlocking). The contact
  resistance function still grows above phi_RCP, providing physical pushback. Implemented
  via `compute_phi_max_deformable()` in parameter_sweep.py and inline in CompactionModel.
  New output: `phi_max` in sweep results. Example: E=0.5 kPa, phi_RCP=0.82 → phi_max≈0.90;
  E=200 kPa → phi_max≈0.82 (near rigid limit).
- **3D volume conservation** (`new_dem_0.py`): Added `overlap_lens_volume()` for exact 3D
  sphere–sphere overlap volume, `compute_effective_radii_3d()` for volume-conserving display
  radii (3D analogue of existing 2D `compute_effective_radii()`). `render_fields_3d()` now
  uses effective radii for sphere mode, conserving total granule volume in the phase field.
  New metric: `volume_conservation` (3D analogue of `area_conservation`). Metrics distance
  calculations and force magnitudes now correctly use all 3 components in 3D mode.

- **Organ-specific phase mapping** (`analysis/organ_targets.py`, `analysis/parameter_sweep.py`):
  The three DEM phases map onto biological tissue architecture, but the mapping is
  **organ-specific**. Each organ defines `perfusive_void_fraction` (f_perf) specifying what
  fraction of non-tissue space is liquid-filled/perfusive vs structural void:
  - phi_f (functional) → tissue parenchyma → BV/TV  [universal]
  - phi_i (inert) → structural void spaces  [organ-dependent fraction]
  - phi_v (liquid) → perfusive channels      [organ-dependent fraction]
  Per-organ f_perf values: lung 0.05 (air sacs), bone 0.10 (marrow), intestine 0.15,
  pancreas 0.30, kidney 0.70, cardiac 0.85, liver 0.90 (sinusoidal blood).
  `compute_organ_distances()` rewritten with organ-specific logic:
  - Permeability: K_f (functional zone) for perfusive-dominated organs (f_perf≥0.5),
    K_i (inert zone) for structural-void-dominated organs (f_perf<0.5).
  - Void-split penalty: z-score penalising scaffolds whose phi_i/(phi_i+phi_v) deviates
    from the organ's expected (1 - f_perf).
  - Per-zone pore radii (r_pore_f, r_pore_i) now exported from sweep for zone-specific
    comparison.
  Granule radius ranges widened (R_func=[5,150] µm, R_inert=[5,300] µm)
  to cover all organ targets from liver sinusoids (8 µm) to trabecular bone spacing (600 µm).
- **Fixed parameter support** (`analysis/parameter_sweep.py`): Parameters can now be held
  constant rather than swept. `cell_sense_distance` fixed at 50 µm (filopodia sensing range).
  `FIXED_PARAMS` dict injected into sample arrays during LHS generation and trajectory
  integration. `func_ratio` constrained to [0.05, 0.95] to prevent degenerate zones.

### Bug Fixes
- **Volume conservation violation** (major): Previous mean-field model decreased phi_f
  during compaction, implying functional granules were losing volume. Fixed by switching
  to two-zone model where total solid fraction is always conserved.
- **Negative compaction ratio**: When phi_solid is high and func_ratio is high, x_f_0 could
  be less than x_f_min, causing spurious expansion. Fixed by clamping x_f_0 to
  max(x_f_0, x_f_min) in both vectorised sweep and single-trajectory integration.
- **cell_sense_distance KeyError**: After removing from swept parameters, `_integrate_trajectory`
  and `recommend_parameters` failed. Fixed by injecting FIXED_PARAMS into sample dicts.
- **Degenerate inert zone void fractions**: When func_ratio ≈ 1, phi_i is negligible and
  phi_v_i becomes meaningless (0.99). Recommendation table now shows "—" when phi_i < 0.02.
- Fixed `KeyError: 'phi_f'` in `plot_compaction_heatmaps` after packing constraint refactor.
- Fixed `KeyError` in `_integrate_trajectory` and `plot_best_kinetics` where sample dicts
  were built from `PARAM_NAMES` only, missing derived `phi_f`/`phi_i` keys.

---

## [V1.8] - 2026-03-15

### Changed
- **Project restructured**: Organized ~20 root-level Python files into `viz/` and `analysis/`
  packages. Root now contains only the simulation engine (`new_dem_0.py`) and 4 runner scripts.
- **Visualization files** moved to `viz/` package (10 files): `viz_postprocess.py` → `viz/postprocess.py`,
  `viz_cells.py` → `viz/cells.py`, `viz_stress.py` → `viz/stress.py`, etc. Import as
  `from viz.postprocess import run_all`.
- **Analysis files** moved to `analysis/` package (5 files): `mean_field_model.py` →
  `analysis/mean_field_model.py`, `coarse_grain.py` → `analysis/coarse_grain.py`, etc.
  Import as `from analysis.coarse_grain import compute_stress_tensor`.
- **HPC files consolidated**: `run_hpc.slurm` and `setup_hpc_env.sh` moved into `hpc/`.

### Removed
- `dem_config.json` (legacy, unused)
- `old/` directory (empty)

---

## [V1.7] - 2026-03-15

### Added
- **Mathematical analysis framework**: Complete continuum-scale mathematical model connecting
  DEM particle simulations to tissue-level predictions and organ comparisons.
- **`mean_field_model.py`**: Mean-field ODE model for cell-driven scaffold compaction.
  `dφ_f/dt = -φ_f(σ_cell - σ_resist)/η_eff` with motor-clutch cell stress, jamming resistance,
  and Poisson bridge formation kinetics. `CompactionModel` class with `solve()`, `fit_to_data()`,
  and `from_run()` for automatic fitting to simulation data. Extracts η_eff (effective viscosity),
  σ_0 (jamming stress), α (jamming exponent). Predicts permeability evolution via Kozeny-Carman
  and Darcy flow rate. Plotting: fit overlay, phase evolution, permeability, stress balance.
- **`coarse_grain.py`**: Coarse-graining routines extracting continuum quantities from DEM
  snapshots. `compute_stress_tensor()` via Love-Weber formula (uses per-contact data from V1.5.2).
  `compute_strain_rate()` from velocity field. `compute_effective_viscosity()` = σ_dev/(2ε̇_dev).
  `compute_coordination()` decomposed by pair type. `extract_continuum_timeseries()` for full
  time evolution. `coarse_grain_field()` for spatially-resolved stress/strain fields via
  Gaussian weighting. Handles both 2D and 3D modes.
- **`viz_dimensionless.py`**: Dimensionless analysis and data collapse across DOE runs.
  Computes β (motor-clutch engagement), Ca (cellular capillary number), jamming proximity,
  timescale ratio, composition ratio, size ratio. Seven plot types: β-collapse (rescaled
  compaction trajectories), Ca-scaling (power-law fits), jamming diagram, composition effects,
  DOE factor effects (main effects + interactions), dimensionless dashboard (3×2), phase space
  (β vs Ca contour). Includes summary statistics table output.
- **`tissue_descriptors.py`**: Comprehensive tissue architecture descriptor vector from 3D
  phase fields (inert phase treated as pore space). 18 descriptors in 8 categories: volume
  fractions (BV/TV, porosity, S/V), morphometry (Tb.Th, Tb.Sp, Tb.N via distance transform),
  topology (Euler characteristic, connectivity density, SMI), spatial statistics (two-point
  correlation S₂(r) via FFT, correlation length, chord length distribution), anisotropy (MIL
  tensor, degree of anisotropy DA, fractional anisotropy FA), transport (tortuosity via Laplace
  solver, Kozeny-Carman permeability), pore size distribution. Handles 2D gracefully.
- **`organ_targets.py`**: Literature-based target descriptor vectors for 7 native organ systems:
  trabecular bone, lung alveoli, liver, kidney cortex, cardiac muscle, pancreatic islet,
  intestinal mucosa. Each with mean, standard deviation (natural variability), description,
  and key references. Radar/spider chart for organ profile comparison.
- **`arch_distance.py`**: Weighted Mahalanobis-like architectural distance between GELS
  scaffolds and organ targets. Log-transform for scale-dependent descriptors (BV/TV, Tb.Th,
  etc.) so ratios drive comparison. `distance_trajectory()` tracks which organ the scaffold
  converges toward over time. `optimal_parameters()` finds DOE conditions minimizing distance
  to a target organ. `sensitivity_analysis()` quantifies DOE factor effects on organ matching.
  Heatmap, radar, and optimization landscape plots.
- **`CodeLog/Architecture/MATHEMATICAL_MODEL.md`**: Formal mathematical model document for
  publications. Covers: microscale DEM equations (Hertz, motor-clutch, friction, adhesion,
  bridge kinetics), mesoscale coarse-graining (Love-Weber, strain rate), macroscale continuum
  model (phase conservation, Darcy flow), mean-field ODE, 6 dimensionless groups with scaling
  laws, tissue architecture characterization framework, and experimental validation strategy.

### Changed
- **`viz/postprocess.py`** (was `viz_postprocess.py`): Pipeline expanded from 10 to 14 modules. New modules 11-14:
  mean_field_model, coarse_grain, tissue_descriptors, arch_distance. Batch DOE processing
  now auto-triggers dimensionless analysis and architectural distance. New skip names:
  'mean_field', 'coarse_grain', 'tissue', 'arch_distance', 'dimensionless'.
- **`CodeLog/Architecture/ARCHITECTURE.md`**: Updated to V1.7 with mathematical analysis
  framework section. New architecture diagram shows analysis pipeline.

---

## [V1.6] - 2026-03-15

### Changed
- **Visualization decoupled from simulation engine** (`new_dem_0.py`): All matplotlib imports,
  built-in plotting functions (`plot_granules`, `plot_fields`, `plot_timeseries`,
  `plot_composite`), and visualization script calls removed from `new_dem_0.py`. The simulation
  engine now only runs physics and saves data — no plotting dependencies.
- **`save_fields` default changed to `True`**: Phase field grids (`phi_f`, `phi_i`, `phi_v`)
  are now saved to disk by default so that postprocessing can produce all visualizations
  without re-running the simulation.
- **Removed UNATTACHED cell state**: Cells are now always attached from the start.
  `CellState` enum: ATTACHED=0, SPREADING=1, PROLIFERATING=2, BRIDGING=3, SENESCENT=4.
  Excess cells during packing go directly to SENESCENT instead of UNATTACHED.

### Added
- **`viz_postprocess.py`**: Unified post-processing script that loads saved simulation data
  and produces all visualizations. Runs the 4 built-in plots (granules, fields, timeseries,
  composite) plus all `viz_*.py` scripts (compaction, percolation, movies, phases, cells,
  stress). CLI: `python viz_postprocess.py -i results/default`. Supports `--skip` to exclude
  specific visualization modules. Programmatic API: `viz_postprocess.run_all(run_dir)`.
- **Numba JIT acceleration** (`new_dem_0.py`): `@njit(cache=True)` applied to ~20 hot-path
  functions: quaternion utilities (7), superellipsoid geometry (5), superellipse geometry (8),
  contact solvers (4: `find_contact_superellipses`, `find_contact_superellipse_wall`,
  `find_contact_spheres_3d`, `find_contact_superellipsoids_3d`), and `hertz_contact_force`.
  Identity fallback when Numba is not installed. JIT warmup (`_warmup_jit()`) runs before
  first timestep to avoid first-step compilation delay.
- **Vectorized integration** (`new_dem_0.py`): Both 2D and 3D integration paths in `step()`
  replaced per-granule Python for-loops with vectorized NumPy operations: force→velocity,
  velocity cap, position update, boundary clamping, rotational dynamics. Active noise in
  `compute_forces()` and `compute_forces_3d()` also vectorized.
- **Walltime estimation** (`run_all_trials.py`): Power-law scaling model
  `T = α * N^β * (steps/ref_steps)` with RSA efficiency correction and safety factor.
  `estimate_walltime()` reads trial JSON, predicts wall time, and `run_hpc()` uses the
  maximum estimate across all trials for SLURM `--time`.
- **Scaling benchmark trials** (`Trials/Trial22-29`): 8 trials varying domain size from
  200 µm to 2000 µm cube (6 to ~7300 granules) for compute-time benchmarking.
- **Wall time recording**: `metadata.json` now includes `wall_time_s` and `n_steps` after
  simulation completes.

---

## [V1.5.2] - 2026-03-14

### Added
- **Per-contact data storage** (`new_dem_0.py`): `compute_forces_3d()` and `compute_forces()`
  now return a third element — a list of contact dicts with per-contact fields: `i`, `j`
  (granule pair), `cx/cy/cz` (contact point), `nx/ny/nz` (contact normal), `overlap`,
  `R_eff`, `F_normal`, `A_contact`. Contacts propagated through `step()`, `save()`, and
  serialized to disk in `.npz` snapshots as structured arrays (`contact_i`, `contact_j`,
  `contact_cx`, etc.).
- **`viz_stress.py`**: New PyVista-based 3D visualization script for surface stress fields.
  - `granule_surface_mesh()`: Generates PyVista PolyData mesh from superellipsoid parametric
    surface with quaternion rotation and world-frame translation.
  - `compute_surface_stress()`: Maps Hertzian contact pressure distribution
    `p(r) = p0 * sqrt(1 - r²/a_c²)` to mesh vertices, with Gaussian smoothing tail.
  - `reconstruct_contacts()`: Fallback contact detection for older snapshots without
    stored contact data.
  - `plot_stress_cross_section()`: Z-slab cross-section (thickness = 1 functional granule
    diameter) showing continuous surface stress colormap. Top-down camera, multi-timepoint.
  - `select_interesting_granules()`: Auto-selects granules by composite score (contact count,
    force magnitude, bridge count), excluding boundary granules.
  - `granule_evolution_gif()`: Isolated granule group time evolution in isometric view with
    surface stress colormap and cell ellipsoid rendering. Fixed color scale across frames.
  - `cell_ellipsoid_mesh()`: Volume-conserving ellipsoid meshes for cells — sphere for
    unattached, oblate for spreading, prolate for bridging (elongated toward target).
  - `plot_granules_3d()`: Isosurface rendering with mid-slice opaque (`opacity=1.0`) and
    remaining granules as ghosts (`opacity=0.1`).
  - `run_all()`: Public API + CLI with `-i`/`-o`, uses `load_run()`.

### Changed
- **Instant cell attachment**: Cells now start as ATTACHED (not UNATTACHED) with
  `t_attach_onset=0.0` and `t_attach_half=0.0` (instant full attachment). When
  `t_attach_half <= 0.01`, the sigmoidal kinetics are bypassed and all cells attach
  immediately.
- **Bridge formation kinetics** (`new_dem_0.py`): Bridge formation is now a gradual,
  stochastic process instead of deterministic and instantaneous:
  - **Probabilistic initiation**: Each eligible cell has a Poisson-distributed probability
    of finding a bridge target per timestep (`bridge_attempt_rate`, default 0.3/h), modulated
    by gap proximity. Only SPREADING or PROLIFERATING cells with sufficient FA maturity
    (`min_fa_for_bridge`, default 0.3) can attempt bridges.
  - **Maturity ramp**: New bridges start weak and ramp to full motor-clutch force over
    `bridge_formation_time` (default 2 h), modeling filopodia extension, migration to gap
    edge, and adhesion formation on the target granule.
  - **Persistent bridges**: Committed BRIDGING cells retain their state across timesteps
    (no longer reset to ATTACHED each step). Bridge age tracked in `cell_bridge_age` array.
  - **Bridge senescence**: After sustained mechanical load (`bridge_senescence_time`,
    default 24 h), bridging cells transition to SENESCENT — they do not detach.
  - **Bridge rupture**: If the gap exceeds `bridge_break_gap` (default 60 µm), the bridge
    ruptures and the cell goes senescent (mechanically damaged).
  - New parameters: `bridge_attempt_rate`, `bridge_formation_time`,
    `bridge_senescence_time`, `min_fa_for_bridge`, `bridge_break_gap`.
- **Phase field isosurface rendering** (`viz_stress.plot_granules_3d()`): `granules_3d.png`
  now uses phase field isosurfaces (`phi_f`, `phi_i`) so overlapping/touching granules
  merge into a continuous solid, matching the physical scaffold. Mid z-slice is opaque,
  rest rendered as ghosts. Falls back to individual parametric meshes when phase fields
  are unavailable.
- **Bridge cell rendering** (`viz_cells.py`): Replaced dumbbell shape with volume-conserving
  elongated ellipse spanning the bridge gap. Cell width derived from area conservation
  (`A_cell / (π * a_long)`), providing realistic fibroblast morphology.
- **`plot_granules()`** (`new_dem_0.py`): 3D mode now delegates to
  `viz_stress.plot_granules_3d()` for isosurface rendering with z-slice opacity, falling
  back to 2D circle projection if PyVista is unavailable.
- **`run_all_trials.py`**: Added `viz_stress.run_all()` to local trial pipeline after
  `viz_cells`.
- **`viz_cells.run_all()`**: In 3D mode, automatically delegates stress rendering to
  `viz_stress` if PyVista is available.

### Bug Fixes
- **Watertight granule meshes** (`viz_stress.granule_surface_mesh()`): Fixed hollow shell
  rendering caused by open parametric surface seams. Omega now uses `endpoint=False` with
  modulo-wrapped face connectivity, and single-vertex pole caps replace degenerate pole rings.
  Meshes are verified manifold via `mesh.is_manifold`. `compute_normals(consistent_normals=True)`
  ensures correct outward face normals for solid rendering and clipping.
- **Watertight cell ellipsoids** (`viz_stress.cell_ellipsoid_mesh()`): Replaced open parametric
  mesh with `pv.Sphere()` primitive scaled by axis radii — guaranteed watertight with proper
  normals.

---

## [V1.5.1] - 2026-03-14

### Added
- **`viz_cells.py`**: New visualization script for cell and stress analysis (2D + 3D).
  - `plot_stress_map()`: Granules colored by net force magnitude (`YlOrRd` colormap)
    with directional arrows (quiver). Multi-panel layout at selected timepoints.
  - `plot_cell_states()`: Fibroblast-like cell morphology rendered as matplotlib patches.
    Stellate/star shapes for proliferating cells, dumbbell-shaped bridging cells that span
    surface-to-surface with contact patches at both ends, round blobs for unattached.
  - `cell_timelapse_gif()`: Animated GIF of cell migration, bridge formation, and pulling
    across all snapshot timepoints. Uses PillowWriter with imageio fallback.
  - `run_all()`: Public API matching other viz scripts, plus CLI with `-i`/`-o`.
  - 3D support via XY projection (circles for granules, quaternion-rotated cell positions).
- **Cell migration** (`new_dem_0.py`): Non-bridging attached cells (ATTACHED, SPREADING,
  PROLIFERATING) now perform a random walk on the granule surface each timestep.
  Controlled by `Params.cell_migration_speed` (default 5.0 µm/h). Applies in 2D
  (`cell_theta_local`) and 3D (`cell_eta_local`, `cell_omega_local`).

### Changed
- **`new_dem_0.py` `save()` closure**: In-memory snapshot dicts now include `force_x`,
  `force_y`, `cell_state`, `cell_granule_id`, `cell_theta_local`, `cell_fx`, `cell_fy`,
  `cell_bridge_target`, `cell_contact_area`, `cell_offset`. 3D mode also adds `force_z`,
  `cell_fz`, `cell_eta_local`, `cell_omega_local`. Previously these were only in on-disk
  `.npz` files — now available to viz scripts called in-process.
- **`update_cell_state()`**: Now accepts optional `rng` parameter, passed through to
  `_update_individual_cells()` for cell migration random walk.
- **Bridge cell rendering**: Bridging cells now render as dumbbell shapes spanning from
  the host granule surface to the nearest point on the target granule surface, with
  contact patches at both attachment points (radius proportional to contact area) and
  a thin process in the middle. Previously rendered as small fixed-size shapes with
  dashed lines to the target granule center.
- **`run_all_trials.py`**: Added `viz_cells.run_all()` to local trial pipeline.

---

## [V1.5] - 2026-03-14

### Added
- **Individual cell tracking (V1.5)**: Each cell is tracked individually with its own
  state (`CellState` enum: UNATTACHED, ATTACHED, SPREADING, PROLIFERATING, BRIDGING,
  SENESCENT), surface position on its host granule, force vector, contact area with
  host granule, and bridge target granule. Cell data stored as flat numpy arrays
  indexed by `cell_offset` per granule.
- **CellState enum**: `IntEnum` with 6 discrete states, extensible for future cell types.
  Stored as integers in numpy arrays for efficient serialization.
- **Data serialization**: Full simulation data now saved to disk during `run()`.
  Per-timepoint `.npz` snapshots include granule positions, velocities, forces, shapes,
  orientations, and all per-cell tracking arrays. Scalar metrics saved as `history.csv`
  (pandas-loadable) and `history.json` (exact fidelity). Phase fields optionally saved
  separately. All output archived as `.tar.gz` for HPC transfer.
- **`load_run()` function**: Load a complete simulation from disk (directory or `.tar.gz`).
  Returns `(hist, snaps, p, metadata)` matching the `run()` return signature.
- **`load_cells()` function**: Targeted loading of per-cell data for specific snapshots.
- **New Params fields**: `save_data` (bool, default True), `save_fields` (bool, default
  False), `output_dir` (str), `compress_archive` (bool, default True).
- **Granule velocity and forces in snapshots**: `vx`, `vy`, `vz`, `force_x`, `force_y`,
  `force_z` now included in serialized data.
- **`_record_bridge_forces()` helper**: Distributes bridging forces to individual cells
  during force computation.
- **`_initialize_cells()` helper**: Distributes cells uniformly on functional granule
  surfaces during packing.

### Changed
- **`update_cell_state()`**: Now also updates individual cell states via
  `_update_individual_cells()`. Per-granule aggregates remain authoritative for force
  computation (backward compatible, identical physics to V1.4.1).
- **`compute_forces()` / `compute_forces_3d()`**: Now record per-cell bridging forces
  and bridge targets. Resets cell force arrays at start of each call.
- **`save()` inner function in `run()`**: Now writes to disk when `p.save_data=True`.
- **Viz scripts**: All 4 viz scripts (`viz_compaction`, `viz_percolation`, `viz_movies`,
  `viz_phases`) can now load full simulation data from disk via `load_run()` in their
  `__main__` blocks.
- **`run_hpc_headless.py`**: `--output-dir` now sets both figure and data serialization paths.
- **`run_all_trials.py`**: Sets `p.output_dir` per trial for data serialization.

### Output Format
```
results/<run_name>/
  params.json          — Params dataclass as JSON dict
  metadata.json        — Version, git hash, seed, domain, cell state enum mapping
  history.csv          — Scalar metrics (one row per save point, pandas-loadable)
  history.json         — Scalar metrics (exact numeric fidelity)
  snapshots/
    snap_0000.npz      — Granule + cell arrays at t=0
    snap_0001.npz      — etc.
  fields/              — Optional (save_fields=True)
    fields_0000.npz    — phi_f, phi_i, phi_v grids
  <run_name>.tar.gz    — Archive of everything
```

---

## [V1.4.1] - 2026-03-14

### Added
- **Packing settle phase**: After RSA placement, `_settle_packing_2d()` and
  `_settle_packing_3d()` run isotropic compression micro-steps (centripetal
  attraction + Hertz repulsion) to bring granules into contact. Controlled by
  `Params.packing_settle_steps` (default 200).
- **Boundary exclusion zone**: `Params.boundary_exclusion` (default 0.2) specifies
  the fraction of the domain excluded from each edge when computing metrics. Only
  granules and field voxels in the inner region are used for connectivity, porosity,
  permeability, and compaction metrics.
- **Flat Trial JSON format (V1.4+)**: Trial configs can now use a flat key-value
  format where keys map directly to `Params` field names. Detected by `"_format": "flat"`
  or presence of Params field names at top level. Legacy nested format still supported.
- **New Params fields**: `packing_gap` (default 0.0 µm), `boundary_exclusion`
  (default 0.2), `packing_settle_steps` (default 200).
- **Example Trial configs**: `Trial15_3D.json`, `Trial16_3D.json` (flat 3D),
  `Trial17_2D.json` (flat 2D).
- **HPC result sync script**: `hpc/sync_results.py` — standalone script to check
  job status and rsync results from the cluster. Supports `--list-only` flag.

### Changed
- **Packing gap**: Default changed from 2.0 µm to 0.0 µm (`Params.packing_gap`).
  Granules now start in contact rather than with artificial spacing.
- **Seed handling**: `run()` default seed changed from 42 to `None` (random).
  `run_all_trials.py` default `SEED = None`. Each run produces a unique packing.
  For HPC, a random seed is generated once and embedded in the SLURM script.
- **`load_trial_json()`**: Now auto-detects flat vs legacy JSON format. Flat format
  maps keys directly to Params fields; unrecognized keys produce a warning.
- **`compute_metrics()`**: All field-based metrics (phi means, connectivity, porosity,
  compaction ratio, Kozeny-Carman) now computed on the inner region after boundary
  exclusion. Reports `boundary_exclusion` and `n_granules_inner` in metrics dict.
- **Legacy Trial configs removed**: Deleted Trial1–Trial12 (legacy nested format).
  Replaced with Trial15–17 using flat format.

### Fixed (HPC Best Practices Audit)
- **`squeue -r` for array jobs**: All `squeue` calls now use `-r` flag to expand
  array sub-tasks into individual rows. Without `-r`, `squeue` shows parent array
  job as single "R" line while any sub-task runs, preventing completion detection.
- **Removed `--mem` from SLURM directives**: Puma allocates 5 GB/CPU automatically
  via `--cpus-per-task`. Specifying both `--mem` and `--cpus-per-task` can cause
  invalid resource requests or redirect to high-memory queues.
- **`slurm_logs/` directory**: SLURM opens output files before script body executes.
  Directory is now created during repo sync (`_sync_repo_to_cluster`) instead of
  inside the script body.
- **`$HOME` in SLURM scripts**: `venv_path` and `repo_path` now use `$HOME` instead
  of `~` inside generated SLURM script content. `~` is kept for SSH/rsync commands
  where the remote shell expands it.
- **Trial validation**: Array job scripts now validate that `$TRIAL` is non-empty
  and exit with error if no trial found for the array task ID.
- **`seff` reminder**: All job completion messages now remind to run `seff JOBID`
  for CPU/memory efficiency checking.
- **Per-task progress tracking**: `_wait_and_sync()` rewritten to show
  `[N done, M running, K pending]` with incremental result sync.
- **Dead code cleanup**: Removed unused `mem_gb` default from `generate_hpc_scripts.py`.

---

## [V1.4] - 2026-03-14

### Added
- **Full 3D volumetric simulation**: Three simulation modes controlled by `Params.mode`:
  `"2D"` (pure V1.3 behavior), `"2D-slice"` (generate 3D packing, slice at z-midplane,
  run 2D simulation), `"3D"` (full 3D overdamped particle dynamics with volumetric
  phase fields).
- **Superellipsoid granule shapes**: 3D analog of superellipses —
  `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1` with equatorial blockiness (n1),
  meridional blockiness (n2), and three semi-axes (a, b, c). Spheres are the special
  case a=b=c, n1=n2=2.
- **Superellipsoid geometry utilities**: `superellipsoid_volume()` (Jaklic & Leonardis 2000),
  `superellipsoid_point()`, `superellipsoid_normal()`, `superellipsoid_curvature_radii()`,
  `superellipsoid_implicit()`, `superellipsoid_implicit_world()`, `superellipsoid_mesh()`.
- **Quaternion orientation system**: Unit quaternion (w,x,y,z) representation for 3D
  rotational state. Utilities: `quat_multiply()`, `quat_conjugate()`, `quat_normalize()`,
  `quat_rotate()`, `quat_rotate_inv()`, `quat_to_rotation_matrix()`, `quat_from_axis_angle()`,
  `quat_random()`, `quat_integrate()`.
- **3D packing generator**: `generate_packing_3d()` — random sequential addition in
  Lx × Ly × Lz box with volume-preserving semi-axis scaling, random quaternion
  orientation, and bounding-sphere overlap checks.
- **3D contact detection**: `find_contact_spheres_3d()` for sphere fast-path,
  `find_contact_superellipsoids_3d()` for general superellipsoid–superellipsoid contact
  via 4-unknown Newton-Raphson (eta_i, omega_i, eta_j, omega_j).
- **3D wall contact**: `find_contact_wall_3d()` for 6 wall faces with Hertzian response.
- **3D force computation**: `compute_forces_3d()` with (N,3) force array, (N,3) torque
  array, 3D tangential friction, 3D cell bridging, 3D active noise.
- **3D integration**: Quaternion rotational integration via `quat_integrate()`,
  3D translational overdamped Euler, 6-face wall clamp.
- **3D volumetric rendering**: `render_fields_3d()` on Ngrid_3d³ grid with
  bounding-box clipping per granule for performance.
- **2D-slice mode**: `slice_superellipsoid_z()` slices a 3D superellipsoid at z-midplane
  to produce a 2D superellipse cross-section. `generate_packing_2d_slice()` generates
  3D packing then slices for 2D simulation.
- **Transport metrics**: `kozeny_carman_permeability()` (K = ε³d²/[180(1−ε)²]),
  `rcp_fraction_superellipsoid()` (φ_RCP ~ 0.64 + 0.08*(AR−1)), porosity, d_grain_mean,
  phi_solid, phi_RCP, compaction_ratio, Da_number added to `compute_metrics()`.
- **Visualization script: `viz_compaction.py`**: Void fraction, packing fraction,
  compaction ratio, void size distribution, stacked phase evolution.
- **Visualization script: `viz_percolation.py`**: Kozeny-Carman permeability, Darcy flow
  rate, porosity with RCP reference, dimensionless groups dashboard (2×3: K, compaction
  ratio, Darcy number, Peclet, Reynolds, porosity ratio), void connectivity.
- **Visualization script: `viz_movies.py`**: Rotating 3D isosurface GIF, time-lapse
  compaction, z-sweep cross-section, composite 2×2. PyVista primary, matplotlib fallback.
- **Visualization script: `viz_phases.py`**: Three-panel isosurface strip at selected
  times, phase volume fractions vs time, tri-plane evolution (XY/XZ/YZ midplanes),
  phase interface area vs time.
- **Example 3D trial config**: `Trials/Trial13_3D.json` with `"mode": "3D"` and
  3D-specific fields (Lz_um, aspect_ratio_c_range, blockiness_n2_range, Ngrid_3d).
- **Optional Numba JIT**: `@njit` decorator for Newton-Raphson solver and geometry
  functions. Falls back to pure Python/NumPy when Numba unavailable. Controlled by
  `Params.use_numba`.
- **New `Params` fields**: `mode`, `Lz`, `aspect_ratio_c_func/inert_mean/std`,
  `blockiness_n2_func/inert_mean/std`, `omega_max_3d`, `Ngrid_3d`, `use_numba`.
- **New `GranuleSystem` arrays**: `z`, `vz`, `c`, `n1`, `n2`, `quat` (N,4),
  `omega_3d` (N,3), mode-aware `positions()`, `is_3d` property.

### Changed
- **`GranuleSystem`** is now mode-aware: stores `z`, `vz`, `quat`, `omega_3d` in 3D
  mode; `theta`, `omega` in 2D mode. `r_bound = max(a, b, c)` in 3D.
- **`compute_forces()`** dispatches to `compute_forces_3d()` in 3D mode. Returns
  `(F, torques)` where F is (N,3) and torques is (N,3) in 3D.
- **`step()`** dispatches to 3D integration (quaternion rotation, 6-face clamp) in 3D mode.
- **`render_fields()`** dispatches to `render_fields_3d()` in 3D mode.
- **`compute_metrics()`** adds transport metrics (porosity, K, compaction_ratio, etc.)
  in all modes.
- **`run()`** dispatches to mode-appropriate packing generator. Snapshots are now dicts
  (not tuples) with named keys.
- **`run_hpc_headless.py`** supports `mode`, `Lz`, `Ngrid_3d` from Trial JSON configs.
- **Console output** shows transport metrics (porosity, K, compaction ratio) per save step.
- **`compute_displacement()`** handles 3D positions (z0 parameter).

---

## [V1.3] - 2026-03-14

### Added
- **Superellipse granule shapes**: Granules are now parameterised as
  superellipses `|x/a|^n + |y/b|^n = 1` with per-granule semi-axes (a, b),
  blockiness exponent (n), and orientation angle (θ). Controlled by
  `shape_enabled` flag in `Params` (default: `False` for V1.2 backward
  compatibility). Per-type distributions for aspect ratio and blockiness.
- **Common normal contact detection**: Newton-Raphson iterative solver for
  superellipse–superellipse contact. Returns penetration depth, contact normal,
  contact point, and local curvature radii. Circle fast-path preserves V1.2
  performance when `shape_enabled=False`.
- **Superellipse–wall contact**: Boundary sampling method to detect penetration
  of non-circular granules against domain walls.
- **Rotational dynamics**: Overdamped angular integration `γ_rot dθ/dt = Σ τ`
  from off-centre contact forces and friction. Angular velocity cap `omega_max`.
  Only active for non-circular granules.
- **Hertz with local curvature**: Contact force uses local radius of curvature
  at the contact point rather than the global granule radius. Generalises the
  Hertz, DMT adhesion, and area-dependent friction models to non-circular shapes.
- **Shape descriptors** (following Liu et al. 2025, Table 1): Per-granule
  circularity (`4πA/P²`), aspect ratio, elongation, and blockiness. Population
  statistics (mean, std) added to metrics output.
- **Superellipse geometry utilities**: `superellipse_area()`, `superellipse_perimeter()`,
  `superellipse_point()`, `superellipse_normal_vec()`, `superellipse_curvature_radius()`,
  `superellipse_polygon_pts()`, `superellipse_implicit()`.
- **New `Params` fields**: `shape_enabled`, `aspect_ratio_func/inert_mean/std`,
  `blockiness_func/inert_mean/std`, `drag_scale_rot`, `omega_max`.
- **New `GranuleSystem` arrays**: `a`, `b`, `n_shape`, `theta`, `omega`, `r_bound`,
  `is_circle` flag.

### Changed
- **`compute_forces()`** now returns `(F, torques)` tuple instead of just `F`.
  Torque array is `(N,)` in units of nN·µm.
- **`step()`** integrates both translational and rotational dynamics.
- **Wall clamp** uses `r_bound` (bounding circle radius) instead of `r`.
- **Neighbour search** uses `r_bound` for cutoff distance.
- **`render_fields()`** uses superellipse implicit function for field profiles
  when `shape_enabled=True`.
- **`plot_granules()`** renders `Polygon` patches for superellipses, `Circle`
  patches for circles.
- **`compute_metrics()`** uses shape-aware contact detection and area
  computation for superellipses.
- **Snapshot tuple** extended with shape arrays (a, b, n_shape, theta) at
  indices 11–14.

---

## [V1.2] - 2026-03-14

### Added
- **HPC support files** for UArizona Puma cluster:
  - `run_hpc.slurm` — SLURM batch job script for headless simulation runs.
  - `setup_hpc_env.sh` — One-time environment setup (Python 3.11 venv).
  - `run_hpc_headless.py` — CLI-driven headless runner that saves figures to
    disk (no `plt.show()`), with argparse overrides for all `Params` fields.
  - `CodeLog/Readme/HPC_SETUP.md` — Step-by-step guide for VSCode Remote SSH
    to UArizona Puma, environment setup, and job submission.
  - `hpc/generate_hpc_scripts.py` — Generates per-user `setup_env.sh` and
    `run.slurm` from a JSON config file (e.g. `hpc/Alex.json`).
  - `hpc/Alex.json` — Example user config for HPC script generation.
- **Trial batch runner** (`run_all_trials.py`): Runs all Trial JSON configs
  in `Trials/` either locally (sequential) or on HPC (SLURM array job, one
  task per trial running in parallel).
- **Trial JSON loading** in `run_hpc_headless.py`: `--trial` flag maps
  the legacy nested Trial JSON format to current `Params` fields.

### Added
- **Area-dependent hydrogel friction** (Gong 2006, Pitenis et al. 2014,
  Uruena et al. 2018): Tangential friction at granule contacts uses the
  hydrogel-tribology model `F_fric = τ₀ × A_contact × tanh(|v_t|/v_ref)`,
  where `A_contact = π R* δ` is the Hertzian contact area. Three pair types
  with distinct interfacial shear stresses: inert–inert (τ₀=50 Pa, bare
  Gemini gel), inert–functional (500 Pa, bare vs collagen), functional–
  functional (2000 Pa, collagen–collagen H-bonding/entanglement).
- **DMT adhesion** (Derjaguin, Muller, Toporov 1975): Constant attractive
  normal force during contact `F_adh = 2π W R*`. Work of adhesion varies by
  pair type: W_ii=0.5, W_if=1.0, W_ff=2.0 mJ/m². Net normal force
  `F = F_Hertz − F_adh` can be negative (attractive).
- **`get_pair_friction_params()`**: Helper to look up τ₀ and W by granule
  pair surface chemistry (bare gel vs collagen-I coated).
- **Velocity state in `GranuleSystem`**: `vx`, `vy` arrays store per-granule
  velocities from the previous timestep for friction force calculation.
- **New parameters in `Params`**: `tau_0_ii/if/ff`, `W_adh_ii/if/ff`,
  `friction_v_ref`.
- **Contact-type metrics**: `n_contacts_ff`, `n_contacts_if`, `n_contacts_ii`
  tracked per save step.
- **Literature references document**: `CodeLog/References/REFERENCES.md`
  catalogues all papers, equations, assumptions, and parameter derivations.
- **Cell lifecycle model**: Cells now follow a biologically motivated lifecycle:
  seeded (spheres, d=20 um) -> attached (~3 h onset, sigmoidal kinetics) ->
  spread (volume-conserving ellipsoid, ~5 um tall) -> bridging (motor-clutch).
- **Motor-clutch force model** (Chan & Odde 2008): Replaces the simple linear
  spring (`k_cell * extension`) with a substrate-stiffness-dependent force model.
  Force per cell depends on `E_modulus` through `F_mc = F_stall * k_sub/(k_sub + k_opt)
  * engagement * FA_maturity`, giving ~2 nN on 1 kPa and ~21 nN on 100 kPa substrates.
- **Projected-area cell placement**: Cell capacity per granule is now computed from
  the cell's projected area (`pi*(d/2)^2` for spheres) divided into the granule's
  projected area (`pi*R^2 * coverage`), not circumference.
- **Cell state arrays in `GranuleSystem`**: `n_attached`, `spread_fraction`,
  `fa_maturity`, `n_overcrowded` track per-granule cell state each timestep.
- **Cell geometry functions**: `cell_projected_area()` computes the footprint as
  cells transition from sphere to oblate ellipsoid (volume conserved).
  `max_cells_on_granule()` derives capacity from projected area ratio.
- **`update_cell_state()`**: Per-step cell state evolution with sigmoidal
  attachment kinetics, stiffness-dependent spreading rate, FA maturation,
  and overcrowding detection (excess cells crawl on each other).
- **`motor_clutch_force()`**: Steady-state motor-clutch force per cell with
  correct unit handling (kPa * um = nN/um for substrate stiffness).
- **Cell state metrics**: `n_attached_total`, `n_seeded_total`, `mean_spread_frac`,
  `mean_fa_maturity`, `n_overcrowded_total` tracked at each save step.
- **Cell state in snapshots**: Snapshots now include `n_attached`, `spread_fraction`,
  `fa_maturity`, `n_overcrowded` arrays for post-processing.
- **Cell state timeseries plots**: Third row in `plot_timeseries()` shows
  attachment, spreading/FA maturity, and overcrowding evolution.
- **New parameters in `Params`**: `cell_height_spread`, `t_attach_onset`,
  `t_attach_half`, `t_spread_duration`, `fa_maturation_rate`, `n_motors`,
  `F_motor_stall`, `n_clutches`, `k_clutch`, `k_on_clutch`, `k_off_clutch`,
  `F_bond`, `cell_sense_distance`.

### Changed
- **`cell_diameter`**: Default changed from 15.0 to 20.0 um (measured cell diameter).
- **`L_rest`**: Default changed from 12.0 to 5.0 um (spread cell thickness).
- **Cell bridging force**: Only attached, non-overcrowded cells can form bridges.
  Bridge force is `F_mc * n_bridges` (motor-clutch, not spring-extension).
- **Bridge metric**: `n_bridges` now only counts pairs where both granules have
  attached cells, matching the force computation.
- **`L_max`**: Now a derived property (`= cell_sense_distance`) instead of an
  independent parameter.
- **Console output**: Status table now shows `attach`, `spread`, `FA_mat`
  columns instead of `delta/R%` and `AreaCon`.
- **`step()` signature**: Now takes `t` (simulation time) to drive cell state.
- **`plot_timeseries()`**: Expanded from 2x3 to 3x3 grid with cell state row.

### Removed
- `k_cell` parameter (replaced by motor-clutch model).
- `L_max` as independent parameter (now derived from `cell_sense_distance`).

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
- **Summary output**: Final summary now reports contact count (by pair type),
  max overlap ratio, area conservation, and friction/adhesion parameters.

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
