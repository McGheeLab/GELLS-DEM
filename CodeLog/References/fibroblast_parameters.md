# Physical parameters for fibroblasts on collagen-I-coated hydrogel granules

Written for: the GELS maintainers and anyone reviewing the V3.0 defaults before publication.

This file records **where every physical default in the V3.0 `cells:` / `contact:` /
`granules:` sections comes from**, how strong the evidence is, and what the literature
explicitly does *not* support. It was assembled in September 2026 from a structured
literature review on ligand surface density → fibroblast traction. Verification status
is preserved per item:

| tag | meaning |
|---|---|
| `[verified]` | primary source read and the quoted claim confirmed verbatim by ≥3 independent reviewers |
| `[numbers verified, framing corrected]` | the numbers are right; an earlier "law" reading of them was refuted |
| `[unverified]` | primary source extracted once; not independently re-checked — **verify before publication** |
| `[typical reported range]` | well-known order of magnitude; no single citation |

> **Bottom line for the model.** There is no published single law for traction vs. ligand
> density. Four independent experimental systems are consistent with a *saturating*
> response in **absolute** ligand density (molecules/µm²), not in coating fraction, with a
> half-saturation of order 10²–10³ ligands/µm². Encoding traction as linear in the coating
> fraction, or as a hard spacing threshold at 58–70 nm, is **not** supported.

---

## 1. Ligand density → fibroblast response (the empirical ladder)

Three unrelated methods (randomly grafted RGD, adsorbed collagen-I, gold nanopatterns)
land on the same absolute-density scale:

| absolute ligand density | mean spacing 1/√σ | fibroblast response | source |
|---|---|---|---|
| ~5 molecules/µm² | ~440 nm | spreading onset (~600 µm² spread area) | Massia & Hubbell 1991 `[surfaced second-hand]`; Gaudet 2003 lowest point |
| ~50 molecules/µm² | ~140 nm | focal contacts and stress fibres appear | Massia & Hubbell 1991 |
| ~150–200 molecules/µm² | ~70–80 nm | ligand supply ≈ integrin supply (~200 integrins/µm² from ~5·10⁵ per cell, Akiyama & Yamada 1985); maximum spread area | Gaudet 2003 |
| ~350–500 molecules/µm² | ~50–58 nm | mature focal adhesions / traction saturation | Cavalcanti-Adam 2007 |
| > 10³ molecules/µm² | < 30 nm | no further gain; force per bond falls ∝ 1/σ | Gaudet 2003; Cavalcanti-Adam 2007 |

### 1.1 Gaudet et al. 2003, *Biophys J* 85(5):3329 — collagen-I density titration `[unverified]`

Human fibroblasts, traction force microscopy, collagen-I surface density 4.8 → 750
molecules/µm². This is the GELS experiment.

| σ (collagen molecules/µm²) | mean traction ‖T‖ |
|---|---|
| 5–100 | 440–590 Pa |
| ~160 (transition) | 260 Pa (minimum) |
| 750 | 700 Pa |

Total force per cell ≈ 0.04 dyn = **400 nN**, ‖T‖ ≈ 300 Pa typical. Spread area rises
600 → 1300 µm² up to σ* ≈ 160/µm², then declines more than twofold by 750/µm². Migration
speed peaks separately at ~10 µm/h near σ ≈ 450/µm² — the area and speed optima are at
different densities. Mechanism: σ* ≈ 160/µm² coincides with the cell's own integrin
density; below it the cell is ligand-limited, above it integrin-limited. Force per
integrin–collagen bond falls from ~100 pN (low density, an upper bound that assumes one
integrin per collagen molecule and exceeds the ~9 pN α5β1–FN rupture force) to ~1 pN.

### 1.2 Roberts, Chen & Mrksich 1998, *JACS* — mixed RGD/(EG)₃OH SAMs `[unverified]`

The direct analogue of diluting a collagen coat with an inert (CH₃ / EG) background.
Attachment and spreading onset at RGD mole fraction χ ≥ 10⁻⁵; spreading saturates at
χ ≥ 10⁻³; at χ ≤ 0.05 the SAM adsorbs essentially no serum protein (SPR). Converting with
the alkanethiolate packing density 4.65·10⁶ chains/µm² (assumption, not the paper's):
onset ≈ 47 RGD/µm², saturation ≈ 4650 RGD/µm². A Langmuir with K_σ ≈ 450/µm² gives 9 %
and 91 % response at those two densities — independently bracketing Gaudet's σ* ≈ 160
within a factor of 3.

### 1.3 Nanospacing studies — what survives scrutiny `[numbers verified, framing corrected]`

- Cavalcanti-Adam 2007, *Biophys J* 92:2964: "cell-surface attachment is NOT sensitive to
  pattern density, whereas the formation of stable focal adhesions and persistent spreading
  is" — adhesions *nucleate* at wide spacing; they fail to *mature*. Spread area 2712 ± 752
  µm² at 58 nm vs 1559 ± 527 µm² at 108 nm: density falls 3.47×, area 1.74× (area ∝ σ^0.45,
  a sublinear dose–response, overlapping SD bands). Only two spacings were tested.
- The "critical spacing" moves >2× with linker chemistry: ~62 nm (aminohexanoic acid),
  ~110 nm (PEG), ~162 nm (polyproline) — Kim et al. 2022, *Adv Mater* 34:2110340.
- Huang et al. 2009, *Nano Lett* 9:1111: at identical mean spacing >70 nm, *ordered* patterns
  switch adhesion off and *disordered* ones switch it on — mean density alone is not the
  controlling variable.
- **Oria et al. 2017, *Nature* 552:219 `[verified 3–0]`**: "increasing the spacing between
  ligands promotes the growth of focal adhesions on low-rigidity substrates, but leads to
  adhesion collapse on more-rigid substrates." The sign of the ligand-density effect flips
  with substrate stiffness. GELS granules (kPa) are in the soft regime where the classic
  rigid-glass threshold does not transfer. (Note: the often-quoted "integrin clustering
  impaired beyond a few tens of nanometres" is Oria's *introduction* citing Arnold 2004 /
  Cavalcanti-Adam 2007, not a finding of that paper.)

### 1.4 Force per adhesion — the 5.5 nN/µm² "constant" `[numbers verified, scope corrected]`

| source | value | scope |
|---|---|---|
| Balaban 2001, *Nat Cell Biol* 3:466 | 5.5 ± 2 nN/µm² | single FAs, GFP-vinculin, micropatterned elastomer |
| Stricker 2011, *Biophys J* | 4 nN/µm² (NIH3T3), 2 (U2OS) | **growth phase only**; r = 0.73–0.87 while growing, 0.08–0.30 once mature |
| Tan 2003, *PNAS* | F ∝ A only above 1 µm² | two adhesion classes |
| Gardel 2008, *JCB* 183:999 | 65 ± 28 Pa (paxillin marker) | ~85× lower |
| Han 2012, *Biophys J* | R² = 0.2–0.4 per FA | weak at the single-adhesion level |
| Gallant 2005, *MBoC* 16:4329 | 200 nN per patch over a 100× area range | stress per area not constant |

Valid as a population-average trend for *growing* adhesions; invalid as a per-adhesion law
or in steady state. If GELS ever needs a stress, use **4 nN/µm²** (fibroblast-matched).

### 1.5 Theory used to shape the model `[unverified unless noted]`

- **Erdmann & Schwarz 2004** (adhesion cluster stability): critical cluster force
  `f_c = N_t · plog(γ/e)` is *linear in bond number*; cluster lifetime is bought by
  rebinding, not bond number. With α5β1/FN numbers (k₀ = 0.012 Hz, F_b = 9 pN, γ ≈ 0.2)
  the critical force is ≈ 0.5 pN per bond, and 10⁴ bonds/µm² reproduce Balaban's 5.5 nN/µm²
  from first principles: **σ_FA = ρ_bond · F_b · plog(γ/e)** — the "constant stress" is really
  proportional to engaged-bond density and falls when ligand density limits it.
- **Bangasser & Odde 2013** `[numbers verified, framing corrected]`: traction magnitude
  ≈ n_m·F_m; the `1/ln(n_c)` appears only in the *optimal-stiffness* expression
  κ_opt = F_m n_m k_on / (v_u ln(n_c) ln(1/ε)); motor and clutch numbers must be balanced
  (n_m ≈ n_c). The paper never mentions ligand density; n_c is intracellular clutch
  availability.
- **Bangasser et al. 2017, *Nat Commun* 8:15313**: partial integrin blockade (0.6 µM
  cyclo(RGDfV)) reduced traction strain energy ~4× **uniformly across all substrate
  stiffnesses** — the licence to treat ligand density as a multiplicative prefactor on the
  motor-clutch force, separable from the stiffness dependence.
- **Elosegui-Artola 2016, *Nat Cell Biol* 18:540**: fibronectin 1/10/100 µg/ml on 5 and 29 kPa
  gels — matrix density, contractility, integrin ligation and talin stability regulate force
  transmission "differently and nonlinearly"; talin unfolding sets a genuine stiffness
  threshold.
- **Gallant, Michael & García 2005**: bound integrins hyperbolic in adhesive area
  (half-max 77 µm², R² = 0.89); adhesion strength saturates; titrating FN 20 → 200 ng/cm²
  raised strength sevenfold while rise time was unchanged — **ligand density sets the
  ceiling, not the kinetics**. Coyer 2012 (*J Cell Sci* 125:5110): individual nanoisland area
  (≥ 0.11 µm² contiguous ECM) controls clustering, not total adhesive area.

---

## 2. What the evidence does **not** support — do not implement

1. **Linear F ∝ ligand density.** Contradicted by Gaudet (U-shaped), Cavalcanti-Adam
   (area ∝ σ^0.45), Gallant (hyperbolic), Han (logarithmic in adhesion number).
2. **A hard percolation threshold at 58–70 nm spacing.** Condition-dependent: moves with
   linker length (62 → 162 nm), inverts with substrate rigidity (Oria 2017), depends on
   ligand order at fixed density (Huang 2009). Fibroblasts form focal contacts at 140 nm and
   spread at 440 nm on randomly grafted RGD (Massia & Hubbell 1991).
3. **A density-independent 5.5 nN/µm² adhesion stress.** It equals ρ_bond·F_b·plog(γ/e) and
   scales with engaged-bond density; breaks down in mature adhesions and per adhesion.
4. **`F_per_clutch = F_total / n_clutches`.** Not in Oria 2017 and physically wrong for that
   model (force is not divided equally among clutches).
5. **`F_traction ∝ 1/ln(n_c)`.** Category error — that is the stiffness *optimum*, not the
   force magnitude.
6. **Scaling `n_clutches` down with coating fraction in the motor-clutch formula.** In
   `F = F_stall · k_sub/(k_sub + n_c k_c) · …` fewer clutches *raise* the bracket and hence
   traction. Ligand density must enter as a transmission prefactor (§3).
7. No Hill coefficient, K_d or fitted dose–response for traction vs ligand density exists in
   this literature. The Langmuir form below is a fit to Gaudet's and Mrksich's data points,
   not a published equation.

---

## 3. How GELS V3.0 encodes this (`gels/materials.py`)

Degree of functionalization f ∈ [0,1] is the coated fraction; absolute ligand density is
σ = f·σ_max. The normalised Langmuir gain (g(0) = 0, g(1) = 1):

```
g(f) = f (1 + κ) / (f + κ),   κ = K_σ / σ_max
```

enters the motor-clutch force as a bond-number ceiling **and** a reduced clutch ensemble
stiffness (Erdmann–Schwarz: f_c ∝ N_t):

```
F_mc = F_stall · k_sub / (k_sub + g·k_opt) · engagement · fa_maturity · g
```

For a bridge from a cell on host granule h to target t, `f_eff` is the **target's** f by
default (`traction_f_rule: target`) — the host anchorage is already encoded in the
f-dependent cell count — with `min` and `product` selectable. A power law `g = f^γ`
(`traction_f_law: power`) is kept for DOE exploration. Both reduce exactly to the V2.7
binary model at f ∈ {0, 1}.

Predicted gain with the defaults (σ_max = 750, K_σ = 160 → κ = 0.213):

| f | g(f) | reading |
|---|---|---|
| 1.00 | 1.000 | fully coated |
| 0.50 | 0.855 | |
| 0.20 | 0.587 | 20 % collagen-I / 80 % bare |
| 0.10 | 0.410 | |
| 0.05 | 0.256 | ≈ `f_min_adhesion`: cells can still grip |

For a saturated RGD monolayer (σ_max ≈ 4·10⁴ /µm²) κ = 0.004 and g(0.2) = 0.98 — dilution to
20 % is invisible. **Calibrating σ_max for the actual coating chemistry matters more than
the coating fraction itself.**

Cell **attachment** saturates at far lower density than traction (focal contacts at
~50/µm², spreading onset ~5/µm²), so seeding uses the same Langmuir with its own
half-saturation `K_sigma_attach = 50/µm²` (κ = 1/15): at f = 0.2 a granule keeps ~80 % of its
cells but they pull at ~59 % force. The bridge lock-in threshold scales with g as well
(`lock_scales_with_ligand`), since the sustainable cluster force is linear in bond number.

---

## 4. Default values and their provenance

### 4.1 Human dermal fibroblasts on collagen-I (`cells:` section)

| quantity | V3.0 default | typical range / source | V2.7 default |
|---|---|---|---|
| migration speed | 30 µm/h (0.5 µm/min) | 12–60 µm/h on 2D collagen `[typical reported range]`; 10–30 µm/h on curved 3D scaffolds; Gaudet 2003 peak ~10 µm/h at 450/µm² | 5 µm/h (**unphysically low**) |
| directed crawl multiplier | 2.0 | modelling assumption | 2.0 |
| motor stall `n_motors × F_motor_stall` | 200 × 0.5 nN = 100 nN | 50–500 nN total per cell on 5–30 kPa (TFM: Dembo & Wang 1999; Munevar 2001; micropillars: Ghibaudo 2008) `[verify magnitudes]`; Gaudet 2003 ≈ 400 nN | 50 × 0.5 = 25 nN (too low; trials already use 200) |
| `F_max_per_cell` | 150 nN | cap; co-tune with `bridge_lock_force_threshold` and `expected_bridge_force` | 50 |
| traction stress (implicit) | — | mean 0.5–2 kPa, peaks 2–5 kPa `[typical reported range]` | — |
| spread area (d = 20 µm, h = 5 µm, volume conserved) | 1257 µm² | 500–1500 µm² on 1–5 kPa; 1500–4000 µm² on >10 kPa (Yeung 2005) `[verify numbers]` | same |
| rounded cell diameter | 20 µm | 15–20 µm | 20 |
| spreading time | 2 h | 1–3 h `[typical reported range]` | 3 h |
| FA maturation rate | 0.5 h⁻¹ (mature in ~2 h) | nascent → mature FA 10–30 min; stress-fibre maturation 1–3 h | 0.3 h⁻¹ |
| sensing / reach distance | 40 µm | filopodia 2–10 µm (`bridge_decay_length` 10 µm); whole-cell reach 40–100 µm for elongated fibroblasts | 40 |
| monolayer density | `surface_coverage 1.0` (16 cells on R = 40 µm) | 1 cell per 1000–2000 µm² | same |
| `sigma_ligand_max` | 750 collagen/µm² | Gaudet 2003 upper point (collagen-I adsorption); RGD monolayer ≈ 4·10⁴ | — |
| `K_sigma_traction` | 160 /µm² | Gaudet σ*; Mrksich-derived ≈ 450 | — |
| `K_sigma_attach` | 50 /µm² | focal-contact onset (Massia & Hubbell 1991); spreading onset ~5 | — |
| doubling time (not modelled) | — | 20–30 h | — |
| bridge senescence time | 24 h | model parameter | 24 h |
| active noise `T_active` | 5 nN·µm | model parameter; calibrate to displacement fluctuations | 5 |

Motor-clutch sanity numbers with `n_motors = 200, n_clutches = 75, k_clutch = 5 nN/µm,
d = 20 µm, ν = 0.45`: k_opt = 375 nN/µm; at E = 2 / 10 / 50 kPa, k_sub = 79 / 394 / 1970
nN/µm, β = k_sub/(k_sub+k_opt) = 0.17 / 0.51 / 0.84, F_mc = 16 / 47 / 76 nN at full maturity.

### 4.2 Hydrogel granules (`granules:` / `contact:` sections)

| quantity | V3.0 default | typical range / source |
|---|---|---|
| Young's modulus, GelMA microgels | 10 kPa | 2–5 kPa (5 % w/v) to 10–30 kPa (10–15 %) `[typical reported range]` |
| PEG (PEG-DA / PEG-NB) microgels | — | 1–100 kPa |
| Ca-alginate microbeads | — | 1–100 kPa |
| HA-based granular hydrogels | — | 1–30 kPa |
| Poisson ratio (swollen gel) | 0.45 | 0.40–0.49; note ν = 0.49 makes the MC-DEM confinement factor c_mc = 24.5 (very sensitive) |
| work of adhesion, bare gel–gel in water (`bb`) | 0.5 mJ/m² | 0.1–5 mJ/m² for hydrated soft gels — far below JKR-in-air values `[verify]` |
| collagen–bare (`cb`) | 1 mJ/m² | assumption |
| collagen–collagen (`cc`) | 2 mJ/m² | 1–10 mJ/m² `[assumption; verify]` |
| friction shear stress, bare gel–gel (`bb`) | 50 Pa | 10–100 Pa; Gong 2006; Pitenis et al. 2014 (Gemini hydrogel friction) |
| collagen–bare (`cb`) | 500 Pa | fitted |
| collagen–collagen (`cc`) | 2000 Pa | no direct measurement known; flag as fitted |
| granule radius | 40–60 µm | 20–150 µm for microfluidic / batch microgels |
| `drag_scale` | 0.05 | model coefficient; calibrate to 24–72 h compaction |

---

## 4.3 PMMA microspheres and the well (V3.1, preset `pmma_well`)

The lab's granules are poly(methyl methacrylate) beads in a well with a flat floor, a
cylindrical side wall and a free top. PMMA is six orders of magnitude stiffer than the
hydrogels V3.0 was written for, which changes what the contact model can and cannot do.

| quantity | V3.1 preset | typical range / source |
|---|---|---|
| Young's modulus | 3.0·10⁶ kPa (3 GPa) | 2.4–3.3 GPa `[typical reported range]` |
| Poisson ratio | 0.37 | 0.35–0.40 |
| density | 1180 kg/m³ | 1.17–1.20 g/cm³ |
| medium density | 1000 kg/m³ | culture medium 1000–1007 |
| buoyant weight (r = 20 / 40 / 75 µm) | 0.059 / 0.47 / 3.1 nN | Δρ g V; compare 50–180 nN of cell traction |
| bed weight stress at 1.5 mm depth | ≈ 2 Pa | vs ≈ 40 Pa of cell contact stress |
| `contact.stiffness_cap_kPa` | 100 | **numerical**, not physical: caps the CONTACT modulus so the explicit integrator sees a bounded force. The cells still see 3 GPa. |
| `contact.max_overlap_frac` | 0.03 | 0.6 µm at r = 20 µm; rigidity is enforced by the overlap projection |
| Coulomb friction µ | 0.3 | dry PMMA–PMMA 0.3–0.5 `[typical reported range]`; lower when wet `[unverified]` |
| work of adhesion, PMMA–PMMA in water | 0.1 mJ/m² used `[assumption; calibrate]` | Hamaker-based estimate ≈ 10 mJ/m² `[unverified]`, which would give a JKR pull-off of ≈ 940 nN for two 80 µm beads — cell-force scale, i.e. the bed would be glued to the wall |
| assumed settled bed fraction | 0.60 (3D), 0.80 (2D) | random loose packing of spheres 0.55–0.60; discs ≈ 0.80 |

**Traction is insensitive to the exact modulus here.** The motor-clutch factor
β = k_sub/(k_sub + k_opt) is 0.51 at 10 kPa, 0.91 at 100 kPa and ≥ 0.99 above 1 MPa, so on
any rigid granule the per-cell force saturates at 0.909 × stall. The calibration knob is the
stall force, not the modulus.

**Rigidity is geometric, not force-resolved.** With γ = drag_scale·r and dt = 0.5 h an
explicit overdamped step is stable only for contact stiffness below ≈ 4 nN/µm, which no
contact carrying cell-scale force can satisfy. Non-penetration therefore comes from
`_resolve_overlaps` plus the velocity cap. Captured: volume exclusion to the tolerance,
sliding rearrangement, approximate jamming. Not captured: equilibrium contact forces and
force chains, wall reaction, rolling resistance, stored elastic energy.

## 4.4 Fibroblast mechanics added in V3.1 (preset `fibroblast_realistic`)

| quantity | V3.1 preset | typical range / source |
|---|---|---|
| doubling time | 24 h (`cells.division`) | 20–30 h for human dermal fibroblasts; ≈ 20 h for NIH-3T3 `[typical reported range]` |
| G1 refractory period after division | 8 h | model parameter |
| contact inhibition | divide while `n_cells < capacity × 1.0` | confluence stops division |
| contraction speed v₀ | 12 µm/h | stress-fibre / actomyosin shortening 0.1–0.5 µm/min `[unverified]`; fibroblast-populated collagen lattices contract at a comparable rate |
| eccentric gain | 0.5 | 0.5–0.8 above isometric in muscle `[typical reported range]`; a modelling assumption here |
| whole-cell reach (`sense_distance`) | 80 µm | a spread fibroblast is 100–150 µm long `[typical reported range]` |
| bridge rupture gap | 120 µm | ≈ one cell length `[assumption]` |
| gap decay length | 30 µm | protrusion scale rather than filopodial (2–10 µm) `[assumption]` |
| stall force on a rigid substrate | 400 × 0.5 = 200 nN, cap 200 nN | 50–500 nN per cell by TFM / micropillars; Gaudet 2003 ≈ 400 nN total `[unverified]`. Saturation gives ≈ 182 nN per bridge. |
| lock-in threshold | 90 nN | ≈ half the matured force, so a bridge locks once load-bearing |
| cell grip at a contact | 20 nN per mature bridging cell (`contact_adhesion_nN_per_cell`) | `[assumption; calibrate]` -- a cell spanning a contact resists shear there; cell-matrix adhesions hold tens of nN |
| cell-cycle CV | 0.15 (`cells.division.cycle_time_cv`) | 0.1-0.3 for mammalian cells in culture `[typical reported range]`; the memoryless rule it replaces implies CV = 1 |
| starting cycle phase | U(0, T_i) per cell | a seeded population is asynchronous; the growing-population form is f(a) = (2 ln2/T) 2^(-a/T), biased young `[not implemented]` |
| maturation clock jitter | 2 h (`cells.seeding.clock_jitter_h`) | model parameter: attachment and spreading are otherwise identical for every granule in the bed |

**Which division model to use (V3.2).** `poisson` is memoryless, so the refractory period adds
to the mean cycle: with `min_age_h = 8` and `doubling_time_h = 24` the realised mean cycle is
`8 + 24/ln2` = 42.6 h, and a 48 h run grows 4.2x instead of 4.0x = 2^2. Under `cycle` the cell
divides at its own drawn `T_i`, so the stated doubling time is the realised one (measured 7.97x
over 72 h against an ideal 8.0x) and the refractory period costs nothing while it sits below
`T_i`. Prefer `cycle`; `poisson` is kept because it is the V3.1 behaviour.

**Note on the force model.** Until V3.1 a bridge pulled with a force independent of gap,
strain, velocity and load, so a cell could not sense that the granules had stopped moving.
The Hill element makes the force fall with shortening speed and hold at the stall force when
the environment resists — the behaviour described as "expanding until there is a force
balance with themselves and the environment". Both models reach the same endpoint force; they
differ in the approach rate and in how load builds.

---

## 4.5 Fragmented hydrogel granules (V3.2, presets `soft_gel_*` + `fragmented_granules`)

The primary system from V3.2: irregular, roughly cuboidal hydrogel fragments, not the rigid
PMMA beads of V3.1 (which remain available as the stiff control, section 4.3).

| quantity | V3.2 preset | typical range / source |
|---|---|---|
| Young's modulus | 10 kPa (`contact.E_kPa`) | 10-50 kPa as stated by the lab; sweep the top of the range |
| Poisson ratio | 0.45 | 0.40-0.49. Report it whenever MC-DEM is on: `c_mc = nu/(1-2nu)` is 4.5 here and **24.5 at 0.49** |
| density | 1050 kg/m3 | hydrated fragmented gel 1030-1100 `[typical]` |
| aspect ratio | 1.8 +/- 0.4 per granule | fragmented / extruded gel `[assumption; measure from imaging]` |
| blockiness n | 3.5 +/- 1.0 per granule | 2 = ellipse, >2 = blocky (Liu 2025) `[assumption; measure]` |
| Coulomb friction mu | 0.3 | rough fragments; the hydrogel shear-stress law alone gives ~1e-3 nN at these contact areas |
| assumed settled bed fraction | 0.52 (3D), 0.75 (2D) | rough irregular fragments settle looser than smooth spheres, and a looser start is half the compaction budget |
| `contact.overlap_model` | `elastic` | derives `max_overlap_frac` from `delta_eq(expected_force_nN)` instead of by hand |
| `contact.curvature_R_cap` | 2.0 | **not optional for blocky shapes**: a flat face has a local curvature radius of ~1e15 um and `F ~ sqrt(R_eff)` |

**Equilibrium contact overlap under a 180 nN bridge** (the number that decides how much a bed
can compact through deformation):

| E | r = 20 um | r = 40 um |
|---|---|---|
| 10 kPa | 3.59 um (18.0 % of r) | 2.85 um (7.1 %) |
| 50 kPa | 1.23 um (6.1 %) | 0.98 um (2.4 %) |
| 3 GPa (PMMA) | 0.001 um | 0.001 um |

Measured consequence, same packing and same traction: interpenetrated volume grows 7.1x
(3 GPa), 13.3x (50 kPa) and **78.5x (10 kPa)** over 24 h. This is the single largest thing
V3.1 gave up by choosing PMMA.

**Packing fraction reached** (2D gravity-settled bed, exact superellipse areas): spheres jam
at 0.839 with a convergence failure; AR 1.8 / n 3.5 fragments reach **0.884** and still
converge, with coordination number rising from 4.2 to 5.4.

---

## 5. Open items

- Gaudet 2003, Erdmann & Schwarz 2004 and Bangasser 2017 — the three papers the traction
  defaults lean on hardest — were not independently re-verified in the review. Massia &
  Hubbell 1991 (the canonical fibroblast ligand-spacing result) was reached only
  second-hand. Re-verify all four before publication.
- Oria 2017's soft-substrate sign flip (sparser ligands → larger adhesions on soft gels) is
  a known effect in exactly the GELS stiffness regime and is **not** modelled.
- Migration speed vs ligand density (Gaudet: peak at ~450/µm², distinct from the area
  optimum) is not modelled; `speed_um_per_h` is a constant.
- `bridge_lock_force_threshold` (20 nN default; 5 nN in the DOE trials) and
  `expected_bridge_force` (100 nN) must be co-tuned with the new stall force and with
  `lock_scales_with_ligand`.
- **V3.1**: the contraction speed, the eccentric gain, the reach triplet and the PMMA work of
  adhesion are the four values most worth measuring or bounding — each is an assumption or a
  typical range rather than a measurement for this system, and each changes the compaction
  curve directly. Calibrate in that order against the measured bed height: stall force sets
  the endpoint, v₀ the early rate, the reach the late plateau, the doubling time the
  curvature after ~24 h.
- The assumed settled bed fractions (0.60 in 3D, 0.80 in 2D) only convert
  `granules.bed_height_um` into a granule count; the achieved fraction is reported as
  `phi_bed` and is the number to compare.

## 6. Sources

Ligand-density titration + force: Gaudet 2003 *Biophys J* 85:3329 · Roberts, Chen & Mrksich
1998 *JACS* · Reinhart-King 2003 *Langmuir* 19:1573 · Elosegui-Artola 2016 *Nat Cell Biol*
18:540 · Gallant 2005 *MBoC* 16:4329

Nanospacing: Oria 2017 *Nature* 552:219 · Cavalcanti-Adam 2007 *Biophys J* 92:2964 ·
Arnold 2004 / 2008 · Huang 2009 *Nano Lett* 9:1111 · Coyer 2012 *J Cell Sci* 125:5110 ·
Kim 2022 *Adv Mater* 34:2110340 · Massia & Hubbell 1991 *JCB* 114:1089

Force per adhesion: Balaban 2001 *Nat Cell Biol* 3:466 · Tan 2003 *PNAS* · Stricker 2011
*Biophys J* · Gardel 2008 *JCB* 183:999 · Han 2012 *Biophys J*

Theory: Erdmann & Schwarz 2004 *J Chem Phys* · Bangasser & Odde 2013 · Bangasser 2017
*Nat Commun* 8:15313 · Chan & Odde 2008 *Science* 322:1687 (motor-clutch)

Hydrogel friction / adhesion: Gong 2006 · Pitenis 2014 · (granule moduli: typical ranges)
