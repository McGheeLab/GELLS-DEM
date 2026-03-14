# GELLS-DEM Literature References

All literature sources used in the GELLS-DEM simulation engine, organised by
physical model. Each entry includes the full citation, the equations or
assumptions we adopt, parameter values we extract, and the rationale for
choosing that particular model.

**Units throughout the code:** µm (length), nN (force), hours (time), kPa (modulus).

---

## 1. Hertzian Contact Mechanics

**Introduced in:** V1.1

### 1.1 Primary Reference

> K. L. Johnson, *Contact Mechanics*, Cambridge University Press, 1985.

Classic treatment of elastic contact between convex bodies. Chapter 4
(normal contact of elastic solids) provides the foundational equations we use
for granule–granule and granule–wall repulsion.

### 1.2 Equations Adopted

**Two identical elastic spheres** (granule–granule):

```
R* = R_i R_j / (R_i + R_j)                   reduced radius
E* = E / [2(1 − ν²)]                          effective modulus (identical materials)
δ  = R_i + R_j − d_ij                         overlap (positive when in contact)
F  = (4/3) E* √R* · δ^{3/2}                  normal repulsive force
```

**Sphere against a rigid flat** (granule–wall):

```
E*_wall = E / (1 − ν²)                        rigid-wall limit (E_wall → ∞)
R*_wall = R_i                                  sphere-flat geometry
F_wall  = (4/3) E*_wall √R_i · pen^{3/2}
```

**Contact area** (3D Hertzian, used for friction calculation):

```
a = √(R* · δ)                                 contact radius (µm)
A_contact = π R* δ                             contact area (µm²)
```

### 1.3 Unit Conversion in Code

The Hertz formula is evaluated in mixed units (E* in Pa, R* and δ in µm).
The conversion factor to nN is:

```
F [nN] = (4/3) · E* [Pa] · √(R* [µm]) · δ^{3/2} [µm^{3/2}] · 10⁻³
```

because Pa · µm² = 10⁻¹² N = 10⁻³ nN.

### 1.4 Assumptions

- **Small overlap**: Hertz theory assumes δ/R ≪ 1 (typically < 10%).
  The simulation monitors `max_overlap_ratio` to verify this.
- **Identical materials**: All granules (functional and inert) share the same
  bulk E and ν. Surface chemistry differs but bulk stiffness does not.
- **Quasi-static contacts**: Overdamped regime; no inertial effects.
- **Frictionless contact** was assumed in V1.1. V1.2 adds tangential friction
  as a separate force (see §3).

### 1.5 Parameter Values

| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| E_modulus | 10 kPa | 1–100 kPa | Typical hydrogel Young's modulus (PEG, GelMA, agarose) |
| poisson_ratio | 0.45 | 0.4–0.5 | Hydrogels are nearly incompressible |

---

## 2. Motor-Clutch Cell Force Model

**Introduced in:** V1.2

### 2.1 Primary Reference

> C. E. Chan and D. J. Odde, "Traction dynamics of filopodia on compliant
> substrates," *Science*, vol. 322, no. 5908, pp. 1687–1691, 2008.
> doi:10.1126/science.1163595

Foundational model for how cells generate traction force on deformable
substrates through a molecular motor–clutch mechanism. Shows that traction
force is a biphasic function of substrate stiffness.

### 2.2 Supporting References

> B. L. Bangasser, S. S. Rosenfeld, and D. J. Odde, "Determinants of
> maximal force transmission in a motor-clutch model of cell traction in a
> compliant microenvironment," *Biophysical Journal*, vol. 105, no. 3,
> pp. 581–592, 2013. doi:10.1016/j.bpj.2013.06.027

Extended the Chan & Odde model to predict optimal stiffness for maximum
traction, which depends on the number of motors and clutches.

> A. W. Casey, S. H. Bhagwat, et al., "Motor-Clutch Model for Cell
> Spreading and Durotaxis," *Biophysical Journal*, vol. 114, no. 3, 2018.

Applies the motor-clutch framework to predict cell spreading dynamics on
substrates of varying stiffness — supports our stiffness-dependent spreading
rate.

### 2.3 Equations Adopted

**Steady-state traction force per cell:**

```
k_sub = π E a_cell / (1 − ν²)            substrate stiffness at cell scale (nN/µm)
k_opt = n_clutches · k_clutch             clutch ensemble stiffness (nN/µm)
engagement = k_on / (k_on + k_off)        steady-state clutch fraction
F_mc  = F_stall · k_sub/(k_sub + k_opt) · engagement · FA_maturity
```

where:
- `F_stall = n_motors × F_motor_stall` is the total motor stall force
- `a_cell = cell_diameter / 2` is the cell radius
- `FA_maturity ∈ [0, 1]` accounts for focal adhesion maturation

**Stiffness-dependent spreading rate:**

Cells spread faster on stiffer substrates (mechanotransduction). The effective
spreading duration is modulated by the same stiffness factor:

```
stiffness_factor = k_sub / (k_sub + k_opt)
eff_spread_dur = t_spread_duration / max(0.2, stiffness_factor)
```

### 2.4 Assumptions

- **Steady-state approximation**: We use the steady-state force, not the full
  stochastic dynamics of motor stepping and clutch binding/unbinding. This is
  valid for timescales ≫ clutch turnover (~seconds vs hours).
- **Rising branch only**: For hydrogel substrates (1–100 kPa), we are on the
  rising (sub-optimal) portion of the stiffness–force curve where
  F ∝ k_sub/(k_sub + k_opt). The peak and falling branch (very stiff
  substrates) are not relevant here.
- **Identical cells**: All cells on a given granule have the same motor-clutch
  parameters. Heterogeneity enters through attachment timing (sigmoidal
  kinetics) and overcrowding.

### 2.5 Parameter Values

| Parameter | Default | Biological basis |
|-----------|---------|------------------|
| n_motors | 50 | Stress fibre complexes per cell |
| F_motor_stall | 0.5 nN | ~500 pN per myosin-II minifilament |
| n_clutches | 75 | Integrin clutch clusters per cell |
| k_clutch | 5.0 nN/µm | Clutch cluster spring constant |
| k_on_clutch | 1.0 /s | Integrin binding rate |
| k_off_clutch | 0.1 /s | Baseline integrin unbinding rate |
| F_bond | 0.002 nN | 2 pN, characteristic bond rupture force |
| cell_diameter | 20 µm | Typical mesenchymal cell diameter |
| cell_height_spread | 5 µm | Measured spread cell height on hydrogels |

**Predicted forces** (with default parameters):

| Substrate E (kPa) | k_sub (nN/µm) | F_mc per cell (nN) |
|--------------------|----------------|---------------------|
| 1 | 39.7 | ~2 |
| 10 | 397 | ~12 |
| 100 | 3970 | ~21 |

---

## 3. Hydrogel Friction (Area-Dependent Tangential Force)

**Introduced in:** V1.2

### 3.1 Primary References

> J. P. Gong, "Friction and lubrication of hydrogels — its richness and
> complexity," *Soft Matter*, vol. 2, pp. 544–552, 2006.
> doi:10.1039/B603209P

Foundational review establishing two regimes of hydrogel friction:
(a) **repulsive interfaces** where friction arises from hydrodynamic shearing
of the water-swollen interfacial layer, and (b) **adsorptive interfaces**
where friction arises from elastic stretching and detachment of adsorbed
polymer chains across the interface.

> A. A. Pitenis, J. M. Uruena, K. D. Schulze, R. M. Nixon,
> A. C. Dunn, B. A. Krick, W. G. Sawyer, and T. E. Angelini,
> "Polymer fluctuation lubrication in hydrogel gemini interfaces,"
> *Soft Matter*, vol. 10, pp. 8955–8962, 2014.
> doi:10.1039/C4SM01728E

Demonstrated that self-mated ("gemini") hydrogel interfaces exhibit
**area-dependent friction** with a velocity-independent interfacial shear
stress τ₀ at low sliding speeds. This is fundamentally different from
Coulomb friction (F = µN).

> J. M. Uruena, A. A. Pitenis, R. M. Nixon, K. D. Schulze,
> T. E. Angelini, and W. G. Sawyer, "Normal load scaling of friction
> in gemini hydrogels," *Biotribology*, vol. 13, pp. 30–35, 2018.
> doi:10.1016/j.biotri.2018.01.002

Quantified the scaling law µ_eff ~ F_n^{−1/3} for gemini hydrogel contacts,
confirming that friction scales with contact area (F_f = τ₀ A) rather than
normal load. This arises because Hertzian contact area A ~ F_n^{2/3}.

### 3.2 Supporting References

> Y. Meier, R. Polderman, and M. H. Müser, "Molecular mechanisms of
> self-mated hydrogel friction," *Tribology Letters*, vol. 71, 54, 2023.
> doi:10.1007/s11249-023-01746-z

Molecular dynamics simulations confirm the constant-shear-stress model for
gemini gel interfaces and provide insight into the molecular-level origins
(polymer chain fluctuations at the interface).

> A. L. Chau, P. T. Getty, A. R. Rhode, C. M. Bates, T. E. Angelini,
> and W. G. Sawyer, "Pore-size dependence and slow relaxation of hydrogel
> friction on smooth surfaces," *PNAS*, vol. 117, no. 11247, 2020.
> doi:10.1073/pnas.1922364117

Links the hydrogel mesh/pore size to the friction coefficient. Smaller mesh
→ higher τ₀ because polymer segments are more densely packed at the
interface.

> Y. Gombert, F. Simič, A. Roncoroni, M. Dübner, T. Geue, and
> N. D. Spencer, "Structuring hydrogel surfaces for tribology,"
> *Adv. Mater. Interfaces*, vol. 6, 1901320, 2019.
> doi:10.1002/admi.201901320

Surface structuring effects on hydrogel tribology; confirms that friction
is governed by real contact area, not nominal area.

### 3.3 Equations Adopted

**Area-dependent tangential friction:**

```
A_contact = π R* δ                             Hertzian contact area (µm²)
v_t = v_rel − (v_rel · n̂) n̂                   relative tangential velocity
F_friction = τ₀ · A_contact · tanh(|v_t| / v_ref)   friction magnitude (nN)
```

Direction: opposes relative tangential sliding. The `tanh` regularisation
avoids numerical chattering at very low sliding speeds (v_ref = 1 µm/h).

**Unit conversion:**

```
F_friction [nN] = τ₀ [Pa] × A_contact [µm²] × 10⁻³
```

because Pa × µm² = 10⁻¹² N = 10⁻³ nN.

**Effective friction coefficient scaling** (emergent, not imposed):

```
µ_eff = F_friction / F_normal = τ₀ A / F_n ~ F_n^{−1/3}
```

This inverse scaling is a hallmark of hydrogel tribology (Uruena et al. 2018)
and arises naturally from combining area-dependent friction with Hertzian
contact mechanics.

### 3.4 Three Interaction Types

The functional granules are coated with collagen-I (col-1) protein, which
presents abundant amine (−NH₂) and carboxyl (−COOH) functional groups. This
creates three distinct surface-chemistry regimes:

| Pair type | Surface chemistry | Friction mechanism | τ₀ (Pa) |
|-----------|-------------------|--------------------|---------|
| **Inert–Inert** | Bare gel–bare gel (Gemini) | Polymer fluctuation lubrication; hydrated interface is very slippery | 50 |
| **Inert–Functional** | Bare gel–collagen-coated | Asymmetric chain adsorption; collagen chains adsorb onto bare gel via H-bonding | 500 |
| **Func.–Func.** | Collagen–collagen | Mutual chain interpenetration, H-bonding (NH₂/COOH), protein entanglement | 2000 |

**Rationale for τ₀ values:**

- **Inert–inert (50 Pa):** Bare hydrogel gemini contacts are among the most
  lubricious material pairs known (µ ~ 0.001–0.01). At typical contact
  pressures of ~1–10 kPa, τ₀ = µ × P gives 1–100 Pa. We use 50 Pa as a
  mid-range estimate (Pitenis et al. 2014, Uruena et al. 2018).

- **Inert–functional (500 Pa):** Collagen-I has backbone amide groups and
  abundant −NH₂ and −COOH side groups that can form hydrogen bonds with the
  bare gel polymer network. This shifts the interface from the repulsive
  (Regime A) to the adsorptive (Regime B) regime of Gong (2006), increasing
  τ₀ by roughly an order of magnitude.

- **Functional–functional (2000 Pa):** Two collagen-coated surfaces can form
  mutual hydrogen bonds (−NH₂ ··· −COOH), physical chain entanglement, and
  electrostatic interactions between charged groups at physiological pH.
  This is analogous to protein-on-protein friction, which is substantially
  higher than polymer-on-polymer (Gong 2006, §3.2). The value of 2000 Pa is
  consistent with measurements of protein-coated hydrogel interfaces.

### 3.5 Assumptions

- **Velocity-independent regime**: At the slow speeds of cell-driven
  rearrangement (~µm/h), hydrogel friction is in the constant-τ₀ regime,
  well below the viscous-drag transition (~mm/s). The `tanh(v_t/v_ref)`
  factor is purely a numerical regularisation, not a physical velocity
  dependence.
- **Previous-step velocities**: Friction forces use velocities from the
  previous timestep (explicit scheme). Valid for dt small relative to the
  relaxation timescale.
- **Rotational friction**: V1.3 adds rotational degrees of freedom for
  non-circular granules; torques arise from off-centre contact forces. See §8.
- **Same bulk modulus**: The contact area calculation uses the same E* for
  all pair types. Only the surface interaction (τ₀, W) varies by pair type.

---

## 4. DMT Adhesion (Normal Attractive Force)

**Introduced in:** V1.2

### 4.1 Primary Reference

> B. V. Derjaguin, V. M. Muller, and Yu. P. Toporov, "Effect of contact
> deformations on the adhesion of particles," *Journal of Colloid and
> Interface Science*, vol. 53, no. 2, pp. 314–326, 1975.
> doi:10.1016/0021-9797(75)90018-1

The DMT model treats adhesion as a surface force that acts within the
Hertzian contact zone but does not modify the contact profile (unlike JKR).
The pull-off force is:

```
F_pull-off = 2π W R*
```

### 4.2 Why DMT and Not JKR

> K. L. Johnson, K. Kendall, and A. D. Roberts, "Surface energy and the
> contact of elastic solids," *Proceedings of the Royal Society A*,
> vol. 324, no. 1558, pp. 301–313, 1971. doi:10.1098/rspa.1971.0141

The JKR model is more accurate for soft, adhesive contacts (large Tabor
parameter λ_T), which applies to our hydrogels:

```
λ_T = (R W² / E*² z₀³)^{1/3}
```

For W ~ 1 mJ/m², E* ~ 5 kPa, R ~ 40 µm, z₀ ~ 0.5 nm: λ_T ≫ 5, firmly
in the JKR regime.

However, the full JKR model is implicit (contact area is not a simple
function of overlap) and computationally expensive for DEM. We use the
**DMT model** as a practical approximation:

- DMT overestimates the pull-off force by ~25% compared to JKR
  (2πWR* vs 3/2 πWR*), but the qualitative behavior is the same.
- DMT is explicit: the adhesive force is a constant offset to the Hertzian
  repulsion, making it easy to implement in the DEM force loop.
- The adhesion energy W can be tuned to compensate for the DMT/JKR
  discrepancy.

> R. W. Style, R. Boltyanskiy, Y. Che, J. S. Wettlaufer, L. A. Wilen,
> and E. R. Dufresne, "Universal deformation of soft substrates near a
> contact line and the direct measurement of solid surface stresses,"
> *Physical Review Letters*, vol. 110, 066103, 2013.

Discusses adhesion mechanics at soft interfaces, supporting the use of
adhesion models for hydrogel contacts.

### 4.3 Equations Adopted

**DMT adhesion during contact** (δ > 0):

```
F_adh = 2π W R*                               pull-off force (constant)
F_normal = F_Hertz − F_adh                     net normal force
```

The net normal force can be negative (attractive), meaning the contact is
under tension. Particles separate when the overlap becomes zero and the
external driving force exceeds F_adh.

**Unit conversion:**

```
F_adh [nN] = 2π · W [J/m²] · R* [µm] · 10³
```

because J/m² × µm = 10⁻⁶ N = 10³ nN ... let us verify:
W [J/m²] × R* [m] gives [N/m × m] = [N]. With R* in µm:
F = 2π × W × R*×10⁻⁶ [N] = 2π × W × R* × 10⁻⁶ × 10⁹ [nN]
= 2π × W × R* × 10³ [nN]. ✓

### 4.4 Three Interaction Types

| Pair type | W_adh (J/m²) | F_pull-off for R*=20 µm | Physical basis |
|-----------|-------------|-------------------------|----------------|
| **Inert–Inert** | 0.0005 (0.5 mJ/m²) | ~63 nN | Bare hydrogels are weakly adhesive; van der Waals + small polymer bridging |
| **Inert–Func.** | 0.001 (1 mJ/m²) | ~126 nN | Collagen chains on functional surface adsorb onto bare gel via H-bonding |
| **Func.–Func.** | 0.002 (2 mJ/m²) | ~251 nN | Mutual H-bonding (NH₂/COOH), chain interpenetration, electrostatic patches |

**Rationale for W values:**

The chosen values are at the **conservative (low) end** of literature ranges
for hydrogel adhesion. Full-strength collagen–collagen adhesion can reach
50–500 mJ/m² (Pei et al. 2021, PNAS 2023), but such values would produce
pull-off forces exceeding 10⁴ nN, leading to >25% overlap ratios that violate
the Hertz small-overlap assumption. Our values produce equilibrium overlaps
of 1–4 µm (2–10% of R), keeping the simulation within the regime where
Hertzian contact mechanics is valid.

### 4.5 Supporting References for Adhesion Energies

> L. Pei, Z. Zhao, C. Chen, J. Jiang, and Z. Suo, "Recent progress in
> polymer hydrogel bioadhesives," *Journal of Polymer Science*, vol. 59,
> pp. 1312–1337, 2021. doi:10.1002/pol.20210249

Comprehensive review of hydrogel adhesion mechanisms: covalent bonding,
H-bonding, electrostatic interactions, chain entanglement, and topological
interlocking. Provides adhesion energy ranges for various hydrogel systems.

> S. Xia, S. Song, F. Jia, and G. Gao, "Programming hydrogel adhesion
> with engineered network topology," *PNAS*, vol. 120, e2307816120, 2023.
> doi:10.1073/pnas.2307816120

Demonstrates that hydrogel adhesion can be programmed from ~1 to ~1000 J/m²
by controlling network topology and interpenetration.

> L. Riley, L. Schirmer, and T. Segura, "Granular hydrogels: emergent
> properties of jammed hydrogel microparticles and their applications in
> tissue repair and regeneration," *Current Opinion in Biotechnology*,
> vol. 60, pp. 1–8, 2019. doi:10.1016/j.copbio.2018.11.001

Review of granular hydrogel scaffolds; discusses inter-particle adhesion
and its role in scaffold mechanical properties.

### 4.6 Assumptions

- **Instantaneous debonding**: When overlap goes to zero, the adhesive
  contact breaks immediately. No gradual pull-off or fibrillation.
- **No long-range adhesion**: DMT adhesion acts only during contact (δ > 0).
  There is no attractive force across a gap (unlike van der Waals at
  nanometre separations, which is negligible at µm scale).
- **Constant W**: The work of adhesion does not change with contact time or
  history. In reality, adhesion can increase with contact duration
  (aging/creep), but this is neglected for simplicity.

---

## 5. Cell Biology Assumptions

**Introduced in:** V1.2

### 5.1 Cell Geometry

> B. Alberts et al., *Molecular Biology of the Cell*, 6th ed.,
> Garland Science, 2014.

- Initial cell diameter: 20 µm (typical for mesenchymal/fibroblast cells)
- Spread cell height: 5 µm (measured on hydrogel substrates)
- Volume conservation during spreading: V = (4/3)π(d/2)³ = const

### 5.2 Collagen-I Surface Chemistry

> M. D. Shoulders and R. T. Raines, "Collagen structure and stability,"
> *Annual Review of Biochemistry*, vol. 78, pp. 929–958, 2009.
> doi:10.1146/annurev.biochem.77.032207.120833

Collagen-I is a triple-helix protein with:
- Abundant **hydroxyproline** and **glycine** residues in the backbone
- **Amine groups** (−NH₂) from lysine and hydroxylysine residues
- **Carboxyl groups** (−COOH) from aspartate and glutamate residues
- At physiological pH (~7.4), some amines are protonated (−NH₃⁺) and
  carboxyls are deprotonated (−COO⁻), creating charged patches

These functional groups enable:
1. **Hydrogen bonding** (NH₂···COOH, OH···COOH) → increased τ₀
2. **Electrostatic interactions** (NH₃⁺···COO⁻) → adhesion
3. **Chain interpenetration** between collagen fibrils on opposing surfaces
4. **Integrin binding** (cells recognise GFOGER sequences in collagen-I)

### 5.3 Cell Attachment Kinetics

> A. J. Engler, S. Sen, H. L. Sweeney, and D. E. Discher,
> "Matrix elasticity directs stem cell lineage specification," *Cell*,
> vol. 126, pp. 677–689, 2006. doi:10.1016/j.cell.2006.06.044

Cells attach and spread on substrates over a timescale of 1–6 hours,
depending on substrate stiffness, ligand density, and cell type. We model
this as:
- Attachment onset at 3 hours (time for cells to settle and form initial contacts)
- Sigmoidal attachment kinetics (half-time 1.5 hours after onset)
- Spread duration of 3 hours (modulated by stiffness)

### 5.4 Focal Adhesion Maturation

> B. Geiger, J. P. Spatz, and A. D. Bershadsky, "Environmental sensing
> through focal adhesions," *Nature Reviews Molecular Cell Biology*,
> vol. 10, pp. 21–33, 2009. doi:10.1038/nrm2593

Focal adhesions mature from nascent adhesions (submicron) to mature FAs
(several µm) over hours, driven by mechanical tension from the actin
cytoskeleton. Maturation rate depends on substrate stiffness.

---

## 6. Overdamped (Langevin) Dynamics

### 6.1 Reference

> E. M. Purcell, "Life at low Reynolds number," *American Journal of
> Physics*, vol. 45, no. 1, pp. 3–11, 1977. doi:10.1119/1.10903

At the microscale in viscous media, inertia is negligible (Re ≪ 1).
The equation of motion reduces to:

```
γ_i dx_i/dt = F_i^total + ξ_i(t)
```

where γ_i = drag coefficient and ξ_i is stochastic noise.

### 6.2 Assumptions

- **No inertia**: Justified for µm-scale objects in aqueous medium at
  hour timescales.
- **Stokes drag**: γ_i = 6πηR_i in principle, but we use a scaled version
  γ_i = drag_scale × R_i to account for effective friction with the
  surrounding medium (which includes other granules, ECM, etc.).
- **Euler integration**: First-order explicit; stable because overdamped
  systems have no oscillatory instabilities. Velocity cap prevents
  numerical blowup.

---

## 7. Superellipse Granule Shapes

**Introduced in:** V1.3

### 7.1 Primary Reference — Shape Descriptors

> S. Liu, Z. Nie, W. Hu, J. Gong, and P. Lei, "Particle shape effects on
> macro- and micro-mechanical behaviours of DEM-modelled granular materials,"
> *Computers and Geotechnics*, vol. 187, 107504, 2025.
> doi:10.1016/j.compgeo.2025.107504

Comprehensive study of how particle shape affects force chains, fabric
anisotropy, and macroscopic strength in granular materials. Table 1 provides
a taxonomy of shape descriptors (sphericity/circularity, roundness, aspect
ratio, convexity, elongation, roughness) that we adopt for characterising
our superellipse granules.

**Shape descriptors adopted from Table 1:**

| Descriptor | Definition | Superellipse formula |
|-----------|------------|---------------------|
| Circularity | 4πA/P² | `4π × superellipse_area / superellipse_perimeter²` |
| Aspect ratio | max(a,b)/min(a,b) | Direct from semi-axes |
| Elongation | 1 − min(a,b)/max(a,b) | Direct from semi-axes |
| Blockiness | Exponent n | Higher n = more rectangular |

### 7.2 Primary Reference — Superellipse DEM

> G. Delaney, J. E. Hilton, and P. W. Cleary, "Defining random
> reproducible granular packings," *Physica A*, vol. 389, no. 10,
> pp. 1929–1936, 2010. doi:10.1016/j.physa.2010.01.003

Demonstrates the use of superellipse/superellipsoid particles in DEM
simulations for generating reproducible random packings. Establishes
the superellipse as a practical parameterisation for non-spherical DEM
particles.

### 7.3 Supporting References

> J. R. Williams and A. P. Pentland, "Superquadrics and modal dynamics
> for discrete elements in interactive design," *Engineering Computations*,
> vol. 9, no. 2, pp. 115–127, 1992. doi:10.1108/eb023852

Early application of superquadric (including superellipse) shapes in
discrete element methods. Establishes the contact detection framework
for superquadric particles.

> G. Mollon and J. Zhao, "3D generation of realistic granular samples
> based on random fields theory and Fourier shape descriptors,"
> *Computer Methods in Applied Mechanics and Engineering*, vol. 279,
> pp. 46–65, 2014. doi:10.1016/j.cma.2014.06.022

Advanced shape representation methods for granular materials; motivates
the use of smooth parametric shapes (like superellipses) as a practical
compromise between circles and fully arbitrary outlines.

### 7.4 Equations Adopted

**Superellipse implicit equation** (body frame):

```
|x'/a|^n + |y'/b|^n = 1
```

where (a, b) are semi-axes, n is the blockiness exponent, and (x', y') are
body-frame coordinates obtained by rotating world coordinates by −θ.

**Special cases:**
- n = 2, a = b: circle (V1.2 backward compatible)
- n = 2, a ≠ b: ellipse
- n > 2: rounded rectangle (blocky hydrogels)
- n < 2: diamond / pinched shape

**Area** (exact, via Gamma functions):

```
A = 4ab · Γ(1 + 1/n)² / Γ(1 + 2/n)
```

For n = 2, a = b = R: A = 4R² · Γ(3/2)² / Γ(2) = 4R² · (√π/2)² / 1 = πR². ✓

**Equivalent radius** (area-preserving):

```
r_eq = √(A / π)
```

This preserves drag coefficients, cell capacity, and noise scaling from V1.2.

**Bounding radius:**

```
r_bound = max(a, b)
```

Used for broad-phase neighbour search and wall clamping.

**Parametric boundary point:**

```
x(t) = a · |cos t|^(2/n) · sign(cos t)
y(t) = b · |sin t|^(2/n) · sign(sin t)
```

for t ∈ [0, 2π].

**Local radius of curvature** (analytic):

```
κ(t) = ab / [(a² sin²t + b² cos²t)^{3/2}] · n · |cos t · sin t|^{2/n - 2}
```

(simplified form; full expression uses parametric derivatives). Used by Hertz
contact with local curvature (§7.6).

### 7.5 Contact Detection — Common Normal Method

**Why common normal over GJK:** The common normal method produces the exact
contact normal, penetration depth, contact point, AND local curvature radii
in a single solve. GJK gives only distance/penetration, requiring additional
work for normals and curvature.

**Algorithm:** For two superellipses i and j, find parameters (t_i, t_j) such
that the outward normals at the boundary points are anti-parallel and collinear
with the line connecting the points. This is a 2-equation nonlinear system
solved by Newton-Raphson (typically 4-8 iterations).

**Returns:** `(in_contact, δ, n_x, n_y, c_x, c_y, R_loc_i, R_loc_j)` where:
- δ = penetration depth (positive when overlapping)
- (n_x, n_y) = contact normal (i → j)
- (c_x, c_y) = contact point (midpoint of closest boundary points)
- R_loc_i, R_loc_j = local curvature radii at the contact points

**Circle fast-path:** When both particles have `is_circle = True`, the solver
is bypassed entirely and the standard `δ = r_i + r_j − d` formula is used.
This preserves V1.2 performance.

### 7.6 Hertz Contact with Local Curvature

For non-circular granules, the effective radius in the Hertz formula uses
the local curvature at the contact point rather than the global radius:

```
R*_local = R_loc_i × R_loc_j / (R_loc_i + R_loc_j)
F_Hertz = (4/3) E* √R*_local · δ^{3/2}
```

DMT adhesion and contact area for friction also use R*_local:

```
F_adh = 2π W R*_local
A_contact = π R*_local δ
```

### 7.7 Superellipse–Wall Contact

For wall contacts, a 1D optimisation finds the boundary point of the
superellipse closest to the wall. Penetration depth is the signed distance
from this point to the wall plane. Local curvature at the wall contact point
is used for the Hertz wall force.

### 7.8 Assumptions

- **Convex particles only**: Superellipses with n ≥ 1 are always convex.
  Non-convex shapes (n < 1) are not supported.
- **Small overlap**: Hertz theory still assumes δ/R ≪ 1. Local curvature
  makes this less restrictive (high-curvature contacts have smaller effective
  R*, producing stiffer response that limits penetration).
- **Rigid rotation**: Particles rotate as rigid bodies. No deformation-induced
  shape change.
- **Independent shape distributions**: Functional and inert granules can have
  different aspect ratio and blockiness distributions.

### 7.9 Parameter Values

| Parameter | Default | Range | Source |
|-----------|---------|-------|--------|
| shape_enabled | False | — | V1.2 backward compatibility |
| aspect_ratio_func_mean | 1.0 | 1.0–2.0 | Hydrogel microgel AR (Liu 2025 Table 1) |
| aspect_ratio_func_std | 0.0 | 0.0–0.3 | Polydispersity |
| blockiness_func_mean | 2.0 | 1.5–4.0 | 2=ellipse, >2=blocky (Liu 2025) |
| blockiness_func_std | 0.0 | 0.0–0.5 | Polydispersity |
| drag_scale_rot | 0.05 | 0.01–0.1 | Scaling for rotational drag |
| omega_max | 1.0 | 0.5–5.0 | rad/h, angular velocity cap |

---

## 8. Overdamped Rotational Dynamics

**Introduced in:** V1.3

### 8.1 Equations Adopted

For non-circular granules, torques arise from contact forces applied at
off-centre contact points:

```
τ_i = Σ (c − x_i) × F_contact
```

where c is the contact point and × denotes the 2D cross product (scalar).

The overdamped angular equation of motion:

```
γ_rot dθ/dt = Σ τ
```

where the rotational drag coefficient is:

```
γ_rot = drag_scale_rot × (a² + b²) / 2
```

This scales with the second moment of the particle shape (analogous to the
moment of inertia, but in the overdamped regime drag replaces inertia).

**Integration:**

```
ω_i = τ_i / γ_rot_i
ω_i = clip(ω_i, −ω_max, ω_max)
θ_i += ω_i × dt
```

### 8.2 Assumptions

- **No angular inertia**: Same overdamped justification as translational
  dynamics (§6). At the microscale in viscous media, rotational inertia
  is negligible.
- **Circles have zero torque**: When a = b and contacts pass through the
  centre, cross products vanish. The rotational integration is skipped
  for circle particles as an optimisation.
- **Angular velocity cap**: Prevents numerical instability from large
  torque impulses, analogous to the translational velocity cap.

---

## 9. Summary of All Parameters by Source

| Parameter | Value | Unit | Source | Section |
|-----------|-------|------|--------|---------|
| E_modulus | 10 | kPa | Hydrogel literature (PEG, GelMA) | §1 |
| poisson_ratio | 0.45 | — | Nearly incompressible hydrogel | §1 |
| tau_0_ii | 50 | Pa | Pitenis 2014, Uruena 2018 | §3 |
| tau_0_if | 500 | Pa | Gong 2006 (adsorptive regime) | §3 |
| tau_0_ff | 2000 | Pa | Gong 2006, protein friction literature | §3 |
| W_adh_ii | 0.0005 | J/m² | Conservative; bare gel vdW adhesion | §4 |
| W_adh_if | 0.001 | J/m² | Conservative; H-bonding from collagen | §4 |
| W_adh_ff | 0.002 | J/m² | Conservative; mutual collagen adhesion | §4 |
| n_motors | 50 | — | Stress fibre count per cell | §2 |
| F_motor_stall | 0.5 | nN | Myosin-II stall force per minifilament | §2 |
| n_clutches | 75 | — | Integrin clutch clusters per cell | §2 |
| k_clutch | 5.0 | nN/µm | Clutch spring constant | §2 |
| cell_diameter | 20 | µm | Mesenchymal cell diameter | §5 |
| cell_height_spread | 5 | µm | Spread cell on hydrogel | §5 |
| t_attach_onset | 3.0 | h | Engler 2006, cell settling time | §5 |
| shape_enabled | False | — | V1.2 backward compatibility | §7 |
| aspect_ratio_*_mean | 1.0 | — | Liu 2025 Table 1 | §7 |
| blockiness_*_mean | 2.0 | — | Liu 2025 (n exponent) | §7 |
| drag_scale_rot | 0.05 | — | Overdamped rotational scaling | §8 |
| omega_max | 1.0 | rad/h | Angular velocity cap | §8 |
