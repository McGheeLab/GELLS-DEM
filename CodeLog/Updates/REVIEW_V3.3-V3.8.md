# Two days, six versions: what changed in GELS between V3.2 and V3.8

**18–20 September 2026.** 40 commits, 73 files, +17,177 / −767 lines,
**17 new test files** (25 → 42) and **9 new engine modules** — five leaves
(`stress`, `convergence`, `laguerre`, `pore`, `kernels/percolation`) plus the
four-file `celltypes/` package.

This is a review document, not a changelog. `CodeLog/Updates/CHANGELOG.md` has
the per-version detail; this says what the six versions were *for*, what they
found, and which numbers you should not quote without reading the caveat next
to them.

---

## The through-line

Nearly everything below is one defect repeated in different places:

> **The model computed something the solver, the packer, or the figures never
> actually used — and nobody was measuring the gap.**

* V3.4: the contact solver reported a penetration depth that was not a
  penetration depth, and a normal that pointed the wrong way half the time.
* V3.5: the packer handed the dynamics a bed loaded 346× above the driving
  force, and no check looked at it.
* V3.6: a wall contact was a contact everywhere except in the implicit step;
  a cell standing on another cell was given a granule's stiffness.
* V3.7: three visualization modules each rebuilt the contact law from a global
  modulus, because the snapshot did not carry the per-pair one.
* V3.8: the cell's random walk had no persistence time, so it inherited the
  timestep's.

The pattern is worth naming because it predicts where to look next: anywhere a
quantity is *derived twice* — once where it is computed and once where it is
consumed — the two will drift, and nothing will say so.

---

## V3.3 — measure the structure, not the renderer

Adopted from `robotsim/`, the gitignored sibling repo.

New leaf modules, each importing nothing from `gels.engine` so the two twins,
the pipeline and the tests share one implementation: `gels/laguerre.py`
(radical Voronoi local packing), `gels/pore.py` (Katz–Thompson permeability,
geodesic tortuosity), `gels/kernels/percolation.py` (union-find on the real
contact graph).

**Why it mattered.** `func_lf` — the connectivity metric the project had been
quoting — was `scipy.ndimage.label` on a tanh field. It is resolution-dependent
and halo-inflated, and it bridges granules separated by ~6 µm of nothing.
`gran_lf_func` measures the actual contact graph. The two are both reported
deliberately; their *ratio* is the inflation.

**Bug fixes.** Periodic neighbour search was silently **dropping** contacts
when the cutoff exceeded half the box — scipy returns each pair at most once,
so a missing interaction, not a double count, which is why it never showed up
as an error. And cell capacity was returning geometrically impossible numbers
(a packing that tiles a surface at 100 %).

---

## V3.4 — the contact solver was broken, not approximate

The load-bearing version. The superellipsoid contact solver was replaced with a
**support-function minimum-translation-distance** solver.

The old common-normal solver, measured:

| | |
|---|---|
| fraction of true contacts detected | **12 %** |
| penetration over-reported by | **9.8× median, 43× p90** |
| 3D shaped contacts with the normal **backwards** | **~50 %** |

The last row is the serious one: force is applied as `−F·n`, so half of the
shaped 3D contacts were **attracting**.

The replacement is exact because the support function of the two-exponent
superellipsoid is closed form — a nested dual norm. `∇h` *is* the support
point, so the contact point is free; `R_eff = √(det ∇²h)` is exact to ~1e-11
instead of ~1e-3; and `sep(n₀) > 0` **proves** separation, which makes it
**4.5× faster** while finding **3.3× more** contacts.

Wall contact became exact in one support evaluation, replacing six brute-force
samplers that under-reported penetration by a median 0.118 µm — the size of the
overlaps being resolved.

`contact.semi_implicit` became the default: across four reference configs it
drives `frac_velocity_clipped` from 0.18–0.35 to **0.000**, so a run reports the
contact law rather than the numerical rails.

---

## V3.5 — is the force law the gradient of an energy, and is the bed balanced?

**`dynamics.gradient_flow`** audits the claim that overdamped dynamics is
gradient flow. Central-differencing the potential against `compute_forces` gives
**3.9e-9** relative for spheres with walls and gravity — the V3.4 force law
really is the gradient of an energy, end to end.

Four things break the equality and **all are intentional**: the active noise
(which injects work the *same size* as the contact work, so the audit is
uninterpretable unless you set `T_active = 0`), tangential friction, MC-DEM, and
shaped granules.

**`packing.relax = fire`** fixed the handoff. The settle stops on a *length*
tolerance that knows nothing about the contact law the dynamics will apply, so
it was handing over a bed pre-loaded far above the driving load:

| | before | after |
|---|---|---|
| shaped gravity bed, residual / granule weight | **346×** | **1.0×** |
| 3D | 131× | 3.6× |
| max penetration | 1.40 µm | 0.032 µm |

**Three negative results, all pinned as tests** — measuring the settle's overlap
tolerance better buys nothing, a length tolerance cannot be chosen without
knowing `E*`, and in 2D the loop exits on its step cap and never on the
tolerance at all.

Also: the V2.7 fixture gate was retired in favour of one locally-blessed
identity gate. Pinning every deliberate default change to a version nobody runs
had cost five pins and froze `reference.py`.

---

## V3.6 — the substrate the solver does not see

Six phases. The two that change results most:

**`dt` is a coupling interval, not an integration step.** The semi-implicit step
`vel = F/(γ + dt·k)` has an exact fixed point for any `dt` — that is what makes
it unconditionally stable — but the **rate** is wrong by `(1 + S)` with
`S = dt·k/γ`. At the shipped `dt = 0.5 h` on a cell-seeded bed, **S averages 41
and reaches 88**, and the 24 h answer was off by ~4× (`F_mean` 136 vs 29 nN).
`dynamics.substep: auto` now subdivides to `S ≈ 0.2`. It **declines a passive
run**, and that is measured: a gravity bed at `dt = 0.5` is within 0.05 % of
`dt = 0.005`, because it ends *at* a fixed point where a compaction run's whole
answer is a rate.

**Cell types became objects** (`gels/celltypes/`) — a frozen dataclass of
`Measured` values, each carrying its source and range, so no value can be added
without saying where it came from. `fibroblast` is the only reviewed type; `msc`
says so.

Plus: wall contacts reach the semi-implicit step (`dt·k/γ` was **13** at the
default `dt`); traction as a stress over contact area; cell strain energy as the
TFM-comparable observable; cells crawl on cells (stacked traction ratio
**0.178**); and a convergence detector with proportional windows.

**A negative result worth keeping:** at the fibroblast's own numbers the adhesion
ceiling never binds, and the model's cell strain energy reads **~10× below** a
fibroblast on flat TFM (0.044 pJ/cell against 0.1–10 pJ). That is the most
actionable calibration target the whole fortnight produced.

---

## V3.7 — the physics the figures were doing by themselves

A sweep of `viz/`, `viz2/` and `analysis/` for machinery stronger than the
engine's. It ran in **two directions**.

**One thing viz had that the engine did not:** a stress tensor.
`grep -rn "virial\|stress_tensor\|pressure" gels/` returned **zero hits** across
130 metric keys. Everything mechanical the engine reported was a *force* —
extensive, and therefore comparable to nothing. `gels/stress.py` adds the
Love–Weber virial stress in kPa, split into what the granular skeleton carries
and what the cells generate:

```
P_contact   +0.271 kPa      the skeleton
P_active    -0.412 kPa      the cells  (negative = TENSION)
stress_active_frac  0.60    the cell-derived share
```

The sign is physics: a contracting bridge puts the bed in **tension** and
*relieves* the compression the contacts carry, while simultaneously raising
`P_contact` by pulling granules closer.

**Three things viz was doing for itself, wrongly.** The contact-energy field was
**790× too large** — ×1000 for a missing Pa→nN/µm² conversion, ×0.75 for a
Hertz prefactor of `(2/5)` where the integral gives `(8/15)`, ×0.73 for Hertz
where the contact is JKR. On the PMMA preset the assumed stiffness was off by
**3e4×**, because a global modulus ignores per-species values and
`contact_E_cap`. Root cause: the snapshot carried ten contact columns and not
`a_contact` / `E_star` / `W` / `kappa`, so nothing downstream *could* use the
engine's own energy. Fixed for ~2 % of snapshot size.

**Stress–force–fabric** (Rothenburg & Bathurst 1989) was added as the upscaling
after the observation that Love–Weber is usually quoted as a rigid-particle
result. It is in fact **exact for this model** — the particle-centred form
telescopes to the branch-vector form whatever the contact point, and the CMN
(1981) terms beyond it are inertial and vanish overdamped. What is *not* exact
at large deformation is the layer underneath: Hertz and JKR are small-strain
theories, so `stress_patch_p95 = a_contact/min(r)` is now measured — 0.05–0.12
on 2D beds but **0.22 on a gravity-loaded 3D bed**.

Verifying the SFF identity caught a bug in V3.7's own new code: the deviator was
taken in the 3×3 zero-padded embedding, so **every isotropic 2D state reported
`q/p` = 0.577**, a shear invented entirely by the padding.

---

## V3.8 — the cell's search has a persistence time, not a timestep

The question was whether `cell_sense_distance` could be derived from the random
search distance in `dt`. It found something worse in the mechanism behind it.

The surface walk drew `N(0, v·dt/r)` — a **ballistic** displacement used as the
width of a **diffusive** step. Variance adds, so search scaled as `√(T·dt)`.
V3.6 then made `dt` a coupling interval subdivided 129–330×:

| T = 0.5 h | search arc |
|---|---|
| `substep: off` | 17.0 µm |
| `substep: auto` (n_sub 129) | **1.45 µm** |
| | **11.75× suppressed** |

V3.6's claim that "everything inside `step` is already a rate × dt" holds for
`1−exp(−rate·dt)`, `+= dt` and `speed·dt`. **It does not hold for a Gaussian
whose width is ∝ dt.** It was invisible because V3.6 re-blessed the baseline —
the known cost of the "re-bless, don't pin" rule.

The walk now carries a heading, which is the only random term and has the `√dt`
scaling. Ballistic for `t ≪ τ_p`, diffusive after, `dt`-independent either way.
**11.75× → 1.175×.**

`persistence_time_h` is a **required** cell-type value with no default.
Crowding is mean-field, with `p_climb = f_cell_cell` — so one parameter sets
both how hard a stacked cell pulls and how willing a cell is to climb.

**A negative result that redirected the question:** `cell_sense_distance` is a
hard cutoff sitting on `exp(−gap/30 µm)`, so at the fibroblast's 80 µm the bridge
probability is already **7 %** of contact value. `bridge_decay_length` is what
sets the void-spanning range; `sense_distance` was the wrong knob.

---

## What to trust, and what not to

**Trust:** the positional bed observables (`bed_height_p95`, `phi_bed_net`,
`bed_radius_p95`), the analytic `*_true` fractions, `gran_lf_*` (the real contact
graph), the Laguerre `phi_loc_*`, and the V3.7 stresses.

**Do not quote without the caveat:**

| quantity | caveat |
|---|---|
| `phi_solid` / `porosity` / `compaction_ratio` | inflated ~1.5× by the tanh halo and overlap double-counting |
| `func_lf` | resolution-dependent; use `gran_lf_func` |
| `phi_bed` | double-counts overlap by 1–4 % once a soft bed compacts |
| `max|F|` | a max over a heavy tail; use `free_force_percentile` |
| any 3D pressure | check `stress_patch_p95` — 0.22 is the edge of Hertz/JKR validity |
| cell strain energy | reads ~10× below flat-substrate TFM |
| `sff_closure` | pair with the **contact** stress, never the total |

**Off by default and uncalibrated:** the convergence detector (calibrate with
`pipeline/shadow_check.py --sweep` first), `cells.stacking`, `cells.traction`
stress ceiling, `dynamics.gradient_flow`.

---

## Open calibration targets

Not tasks — things that need a measurement or a literature review before they
can back a result.

1. **The ~10× TFM gap.** 0.044 pJ per loaded cell against a measured 0.1–10 pJ;
   21.8 Pa footprint traction against ~300 Pa. `k_cell_nN_per_um` is currently
   *calibrated* to land in the band, not predicted, so the energy is not yet an
   independent prediction.
2. **`τ_p = 1.0 h` is the short end of Gail & Boone.** Their data show
   persistence across 2.5 h intervals. `D = v²τ/2` is linear in τ, so the
   (0.5, 3.0) span is a factor of 6 in diffusivity.
3. **`f_cell_cell = 0.15` is an assumption** now doing double duty — stacked
   traction *and* climbing probability. Two observables, one unmeasured number.
4. **`gels/celltypes/msc.py` has not been reviewed**, including a traction
   stress solved backwards from a consistency check.
5. **The convergence tolerances** are module constants never calibrated against
   a real run.
6. **`gels/laguerre.py` tessellates on bounding spheres** — `phi_loc_*` on
   blocky granules divides a true shape volume by a sphere-based territory.
   `viz2/voronoi.py` already has the shape-aware alternative.

---

## Testing

| | |
|---|---|
| test files added | 17 (25 → 42) |
| tests at V3.8 | **629, no skips** |
| regression gate | one: `tests/test_identity.py`, locally blessed |

The gate was re-blessed twice in this window — V3.6 (substepping) and V3.8 (the
walk). **Both re-blessings are documented in the changelog with what moved and
why**, because after re-blessing the gate can no longer see it. That discipline
is the only thing standing between "we changed a default deliberately" and "we
lost a regression."

A note on method that earned its place: several findings in this window came
from **verifying an identity that should hold** rather than from a failing test.
The 2D deviator bug surfaced from checking the SFF relation; the walk's
`dt`-dependence surfaced from asking what `dt` the search distance implied.
Tests that pass tell you nothing about the thing you did not think to test.
