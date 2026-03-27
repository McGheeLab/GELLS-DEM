# GELS Literature References

All literature sources used in the GELS simulation engine, organised by
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

---

## 10. Jamming Physics & Yield Stress

**Introduced in:** V1.7 (mean-field model), V1.11 (energy landscape)

### 10.1 Jamming Transition

> A. J. Liu and S. R. Nagel, "Jamming is not just cool any more," *Nature*,
> vol. 396, pp. 21–22, 1998. doi:10.1038/23819

Seminal perspective establishing the jamming phase diagram unifying granular
materials, foams, emulsions, and glasses. The jamming point J defines the
critical packing fraction φ_J above which a disordered assembly of repulsive
particles develops a nonzero shear modulus. For our granular hydrogel scaffold,
φ_J sets the boundary between a flowable suspension and a mechanically stable
scaffold.

### 10.2 Zero-Temperature Jamming

> C. S. O'Hern, L. E. Silbert, A. J. Liu, and S. R. Nagel, "Jamming at
> zero temperature and zero applied stress: The epitome of disorder,"
> *Physical Review E*, vol. 68, no. 1, 011306, 2003.
> doi:10.1103/PhysRevE.68.011306

Defines the critical scaling laws near the jamming transition for frictionless
soft spheres. The key results we adopt:

**Equations adopted:**

```
σ_y ~ (φ − φ_J)^Δ        yield stress scaling near jamming
G   ~ (φ − φ_J)^α        shear modulus scaling
z − z_c ~ (φ − φ_J)^0.5  excess contacts above isostaticity
```

where Δ ≈ 1.0–1.5 depending on interaction potential (Δ = 3/2 for Hertzian
particles). We use Δ = 1.5 in the energy landscape yield stress term G_yield,
consistent with Hertzian (3/2-power) contacts.

**Rationale:** The O'Hern scaling provides a physics-based form for the yield
stress barrier in the free energy landscape, replacing ad hoc constitutive
choices. The Herschel-Bulkley yield term G_yield = c_yield · σ_0 · max(0, ξ − ξ_J)^Δ
derives directly from this scaling.

### 10.3 Jamming Review

> M. van Hecke, "Jamming of soft particles: geometry, mechanics, scaling and
> isostaticity," *Journal of Physics: Condensed Matter*, vol. 22, no. 3,
> 033101, 2010. doi:10.1088/0953-8984/22/3/033101

Comprehensive review of the physics of jammed soft-particle packings.
Establishes the geometric picture of isostaticity (z_iso = 2d for frictionless
spheres in d dimensions), the role of excess contacts in determining mechanical
properties, and the scaling of elastic moduli and yield stress with distance
from φ_J. We adopt the general framework of treating granular scaffolds as
soft jammed packings where mechanical response emerges from the network of
Hertzian contacts.

### 10.4 Herschel-Bulkley Rheology Near Jamming

> P. Olsson and S. Teitel, "Critical scaling of shear viscosity at the
> jamming transition," *Physical Review Letters*, vol. 99, 178001, 2007.
> doi:10.1103/PhysRevLett.99.178001

First demonstration that the jamming transition at finite shear rate exhibits
critical scaling with a diverging viscosity η ~ |φ − φ_J|^{−β}. Establishes
the connection between jamming and the Herschel-Bulkley constitutive law.

> P. Olsson and S. Teitel, "Herschel-Bulkley shearing rheology near the
> athermal jamming transition," *Physical Review Letters*, vol. 109, 108001,
> 2012. doi:10.1103/PhysRevLett.109.108001

Explicit derivation of Herschel-Bulkley rheology σ = σ_y + k·γ̇^n near the
athermal jamming point. Shows that the yield stress σ_y and flow exponent
emerge naturally from the jamming critical point. We adopt this framework
in the mean-field ODE where resistance stress saturates at σ_y above φ_J,
and in the energy landscape yield barrier term G_yield.

**Equations adopted:**

```
σ(γ̇) = σ_y + k · γ̇^n     Herschel-Bulkley constitutive law
σ_y   ~ (φ − φ_J)^Δ       yield stress from jamming
```

### 10.5 Stress Scaling Near Jamming

> B. P. Tighe, E. Woldhuis, and M. van Hecke, "Model for the scaling of
> stresses and fluctuations in flows near jamming," *Physical Review Letters*,
> vol. 105, 088303, 2010. doi:10.1103/PhysRevLett.105.088303

Develops a model for stress fluctuations near the jamming transition that
explains the broad stress distributions observed in simulations. Relevant to
our coarse-graining analysis (§analysis/coarse_grain.py) where we compute
stress tensors from DEM contact data and observe large spatial fluctuations
characteristic of proximity to jamming.

---

## 11. Random Close Packing & Polydispersity

**Introduced in:** V1.4 (transport metrics), V1.7 (mean-field model)

### 11.1 Definition of RCP

> S. Torquato, T. M. Truskett, and P. G. Debenedetti, "Is random close
> packing of spheres well defined?" *Physical Review Letters*, vol. 84,
> no. 10, pp. 2064–2067, 2000. doi:10.1103/PhysRevLett.84.2064

Critical analysis of the random close packing concept. Proposes replacing
the ill-defined "RCP" with the "maximally random jammed" (MRJ) state. For
monodisperse spheres, φ_RCP ≈ 0.64 (3D) or φ_RCP ≈ 0.84 (2D). We use
these values as baselines and correct for polydispersity and shape effects
via the formulas in §11.2–11.3.

### 11.2 Polydisperse Packing

> R. S. Farr and R. D. Groot, "Close packing density of polydisperse hard
> spheres," *Journal of Chemical Physics*, vol. 131, no. 24, 244104, 2009.
> doi:10.1063/1.3276799

Analytic formula for the random close packing fraction of polydisperse hard
spheres as a function of the size distribution. We adopt the first-order
correction:

**Equation adopted:**

```
φ_RCP(σ_R) ≈ φ_RCP,mono + 0.016 · CV_R²
```

where CV_R = σ_R / ⟨R⟩ is the coefficient of variation of the radius
distribution. Wider size distributions pack more efficiently (small particles
fill interstices between large ones), increasing the jamming fraction.

### 11.3 Non-Spherical Packing

> A. Donev, I. Cisse, D. Sachs, E. A. Variano, F. H. Stillinger,
> R. Connelly, S. Torquato, and P. M. Chaikin, "Improving the density of
> jammed disordered packings using ellipsoids," *Science*, vol. 303,
> no. 5660, pp. 990–993, 2004. doi:10.1126/science.1093010

Experimental and computational demonstration that slightly non-spherical
particles (ellipsoids with aspect ratio ~1.25) pack more densely than spheres
(φ ≈ 0.71 vs 0.64). We adopt the correction in `rcp_fraction_superellipsoid()`:

**Equation adopted:**

```
Δφ_shape ≈ 0.013 · (AR − 1)    for AR close to 1
```

> Y. Yuan, K. VanderWerf, M. D. Shattuck, and C. S. O'Hern, "Jammed
> packings of 3D superellipsoids with tunable packing fraction, coordination
> number, and ordering," *Soft Matter*, vol. 15, pp. 9751, 2019.
> doi:10.1039/C9SM01932D

Systematic study of jammed packings of superellipsoids as a function of
blockiness and aspect ratio. Provides φ_J values for the superellipsoid
parameter space relevant to our simulation. Confirms that blockiness > 2
increases packing fraction (more space-filling shapes).

### 11.4 Superellipsoid Packing Properties

> G. W. Delaney and P. W. Cleary, "The packing properties of
> superellipsoids," *Europhysics Letters*, vol. 89, 34002, 2010.
> doi:10.1209/0295-5075/89/34002

Comprehensive study of packing fractions, coordination numbers, and ordering
in superellipsoid packings. We use results from this paper in the function
`rcp_fraction_superellipsoid()` to correct the baseline RCP fraction for
blockiness effects.

---

## 12. Energy Landscape Theory

**Introduced in:** V1.11

### 12.1 Energy Landscape Formalism

> D. J. Wales, *Energy Landscapes: Applications to Clusters, Biomolecules
> and Glasses*, Cambridge University Press, 2003. ISBN: 978-0521814157

Foundational monograph on energy landscape theory. Establishes the mathematical
framework for analysing complex systems through their potential energy surface:
minima correspond to stable states, saddle points to transition states, and the
connectivity of minima determines kinetic pathways. We adopt this framework to
construct a one-dimensional free energy landscape G(ξ) along the compaction
coordinate ξ, where the minimum of G(ξ) gives the equilibrium compaction and
the barrier height determines the kinetic timescale.

**Key concepts adopted:**

- **Minimum = equilibrium compaction ξ***: The compaction coordinate at which
  dG/dξ = 0 and d²G/dξ² > 0.
- **Barrier height ΔG‡**: Energy difference between the maximum of G(ξ) and
  the uncompacted state G(0), determining the activation barrier for
  compaction initiation.
- **Landscape curvature κ = d²G/dξ²|_{ξ*}**: Determines the sharpness of the
  equilibrium and fluctuation amplitude δξ ~ √(k_BT/κ).
- **Overdamped descent**: In the viscous limit (cell-driven remodelling over
  hours), the system follows the steepest descent path dξ/dt = −(1/η_eff)·dG/dξ.

### 12.2 Kramers Theory for Overdamped Dynamics

> H. A. Kramers, "Brownian motion in a field of force and the diffusion model
> of chemical reactions," *Physica*, vol. 7, no. 4, pp. 284–304, 1940.
> doi:10.1016/S0031-8914(40)90098-2

Classic theory for escape rates over energy barriers in the overdamped
(high-friction) limit. The overdamped Kramers rate is:

**Equation adopted:**

```
k_escape = (ω_min · ω_barrier) / (2π γ) · exp(−ΔG‡ / k_BT)
```

We use the structure (not the thermal activation) of Kramers theory to
formulate the kinetic equation for compaction dynamics. In our system, the
"thermal energy" is replaced by the active cell traction σ_cell, so the
relevant energy scale is G/σ_cell rather than G/k_BT. The overdamped descent
equation dξ/dt = −(1/η_eff)·dG/dξ is the deterministic limit of the Kramers
framework.

### 12.3 Energy Barriers in Dense Tissues

> D. Bi, J. H. Lopez, J. M. Schwarz, and M. L. Manning, "Energy barriers
> and cell migration in densely packed tissues," *Soft Matter*, vol. 10,
> pp. 1885–1890, 2014. doi:10.1039/C3SM52893F

Computes energy barriers for cell rearrangements in confluent tissues using
a vertex model. Shows that energy barriers scale with cell–cell interfacial
tension and increase as cells become more rigid (higher shape parameter).
Directly motivates our treatment of the inert-granule frustration term G_inert,
which represents the energy cost of rearranging functional granules past
inert obstacles.

> D. Bi, J. H. Lopez, J. M. Schwarz, and M. L. Manning, "A density-independent
> rigidity transition in biological tissues," *Nature Physics*, vol. 11,
> pp. 1074–1079, 2015. doi:10.1038/nphys3471

Discovers a rigidity transition in confluent tissues that depends on cell shape
rather than packing fraction. Introduces the target shape parameter s_0 = P_0/√A_0
as the control variable. Relevant to our framework because it demonstrates that
energy landscape topology (barrier heights, number of minima) in biological
tissues depends on geometric parameters — analogous to how our landscape depends
on the dimensionless groups β, Ca, Φ_r, Ψ, and Γ.

### 12.4 Energy Landscape Decomposition in Biology

> J. Shi, K. Aihara, T. Li, and L. Chen, "Energy landscape decomposition
> for cell differentiation with proliferation effect," *National Science
> Review*, vol. 9, no. 8, nwac116, 2022. doi:10.1093/nsr/nwac116

Develops an energy landscape decomposition approach for cell fate transitions,
separating the landscape into contributions from different regulatory
interactions. We adopt the conceptual approach of decomposing G(ξ) into
physically distinct contributions (cell traction, elastic, yield, void,
inert frustration, interfacial) so that each term can be independently
varied and its effect on the equilibrium assessed.

---

## 13. Poroelasticity & Consolidation

**Introduced in:** V1.7 (mean-field model), V1.11 (energy landscape)

### 13.1 Biot Consolidation

> M. A. Biot, "General theory of three-dimensional consolidation," *Journal
> of Applied Physics*, vol. 12, no. 2, pp. 155–164, 1941.
> doi:10.1063/1.1712886

Foundational theory for coupled fluid flow and solid deformation in porous
media. Biot's consolidation model describes how interstitial fluid is expelled
from a deforming porous skeleton, producing time-dependent compaction. We adopt
the conceptual framework (not the full PDE) for the void-redistribution energy
term G_void, where osmotic pressure Π drives equilibration of void fraction
between functional and inert zones as compaction proceeds.

### 13.2 Brinkman Permeability

> H. C. Brinkman, "A calculation of the viscous force exerted by a flowing
> fluid on a dense swarm of particles," *Applied Scientific Research*,
> vol. A1, pp. 27–34, 1949. doi:10.1007/BF02120313

Extension of Darcy's law to account for viscous stress in the fluid phase,
relevant at intermediate porosities. The Brinkman equation bridges the
Stokes (dilute) and Darcy (dense) limits. We use the Kozeny-Carman
permeability model (implemented in `kozeny_carman_permeability()`) which is
the Darcy limit appropriate for our densely packed granular scaffolds (φ > 0.4).

---

## 14. Interfacial Tension in Emulsions & Foams

**Introduced in:** V1.11

### 14.1 Concentrated Emulsion Mechanics

> H. M. Princen, "Highly concentrated emulsions. I. Cylindrical systems,"
> *Journal of Colloid and Interface Science*, vol. 71, pp. 55–66, 1979.
> doi:10.1016/0021-9797(79)90222-2

First systematic treatment of the mechanics of concentrated emulsions
(φ > φ_RCP). Derives the relationship between droplet deformation, interfacial
tension, and osmotic pressure. Motivates our treatment of the granular scaffold
as a dense emulsion where functional and inert granules play the role of
deformable droplets separated by thin fluid films.

### 14.2 Foam and Emulsion Rheology

> H. M. Princen, "Rheology of foams and highly concentrated emulsions.
> I. Elastic properties and yield stress of a cylindrical model system,"
> *Journal of Colloid and Interface Science*, vol. 91, pp. 160–175, 1983.
> doi:10.1016/0021-9797(83)90323-5

Derives the elastic shear modulus and yield stress of ordered and disordered
emulsions as functions of droplet volume fraction and interfacial tension:

**Equations adopted (conceptual form):**

```
G_foam ~ (γ_s / R) · f(φ)         elastic modulus from interfacial tension
σ_y   ~ (γ_s / R) · g(φ − φ_c)   yield stress from surface energy
```

where γ_s is the interfacial tension, R the droplet radius, and f, g are
known functions of packing fraction. We use the scaling relationship
G_surface ~ γ to parameterize the interfacial energy cost of compaction in
our energy landscape.

### 14.3 Osmotic Pressure of Foams

> H. M. Princen, "Osmotic pressure of foams and highly concentrated
> emulsions. I. Theoretical considerations," *Langmuir*, vol. 2,
> pp. 519–524, 1986. doi:10.1021/la00070a023

Derives the osmotic pressure Π(φ) for concentrated emulsions, which
represents the thermodynamic cost of increasing the volume fraction by
expelling continuous-phase fluid. We adopt this concept for the void
redistribution energy G_void, where the osmotic pressure quantifies the
energetic cost of transferring void space from the compacting functional
zone to the inert zone.

### 14.4 Foam Mechanics at the Bubble Scale

> D. J. Durian, "Foam mechanics at the bubble scale," *Physical Review
> Letters*, vol. 75, no. 26, pp. 4780–4783, 1995.
> doi:10.1103/PhysRevLett.75.4780

Introduces the "bubble model" for foam dynamics where each bubble interacts
through spring-like repulsion and viscous drag. Demonstrates that macroscopic
Herschel-Bulkley rheology emerges from microscopic bubble-scale interactions
— the same paradigm we adopt in GELS where macroscopic compaction
behavior emerges from microscopic granule-granule Hertzian contacts and
cell-driven forces.

---

## 15. Polymer Dynamics & Viscosity

**Introduced in:** V1.7 (mean-field model), V1.11 (energy landscape)

### 15.1 Polymer Dynamics Reference

> M. Doi and S. F. Edwards, *The Theory of Polymer Dynamics*, Oxford
> University Press, 1986. ISBN: 978-0198520337

Standard reference for the dynamics of polymer systems. The overdamped
Langevin equation (§6) used throughout GELS derives from the same
theoretical framework applied to colloidal and polymeric systems. We adopt
the viscous-dominated, inertia-free dynamical equations that Doi and Edwards
develop for concentrated polymer solutions, adapted to our granular system
where effective viscosity η_eff governs the compaction rate.

---

## 16. Granular Hydrogel Mechanics

**Introduced in:** V1.5 (individual cell tracking), V1.11 (energy landscape)

### 16.1 Building Block Properties

> S. Cai, B. Bhatt, et al., "Building block properties govern granular
> hydrogel mechanics through contact deformations," *Science Advances*,
> vol. 8, no. 50, eadd8570, 2022. doi:10.1126/sciadv.add8570

Demonstrates that the mechanical properties of granular hydrogel scaffolds
are governed by the Hertzian contact mechanics of individual building blocks
(microgels), not by bulk material properties alone. The Young's modulus and
size of individual granules determine the macroscopic storage modulus and
yield stress through contact area and overlap. Directly supports our use of
Hertzian contact mechanics (§1) as the primary mechanical interaction and
validates our elastic energy term G_elastic.

### 16.2 Programmed Shape Transformations

> N. Di Caprio, A. J. Hughes, and J. A. Burdick, "Programmed shape
> transformations in cell-laden granular composites," *Science Advances*,
> vol. 11, no. 3, eadq5011, 2025. doi:10.1126/sciadv.adq5011

Demonstrates that cell-driven compaction in granular composites produces
programmable macroscopic shape changes. The functional (cell-laden) zones
compact while inert zones resist, generating internal stresses that drive
shape transformation. This is the primary experimental motivation for the
GELS simulation and the energy landscape framework: the competition
between cell traction (driving compaction) and mechanical resistance (elastic,
yield, geometric frustration) determines the final tissue architecture.

---

## 17. Tissue Architecture & Design Rules

**Introduced in:** V1.7 (tissue descriptors), V1.11 (energy landscape design rules)

### 17.1 Microstructural Anisotropy Tensors

> T. P. Harrigan and R. W. Mann, "Characterization of microstructural
> anisotropy in orthotropic materials using a second rank tensor," *Journal
> of Materials Science*, vol. 19, no. 3, pp. 761–767, 1984.
> doi:10.1007/BF00540446

Introduces the use of a fabric tensor (second-rank symmetric tensor) to
quantify microstructural anisotropy in porous materials. We adopt this
approach in `analysis/tissue_descriptors.py` to characterize the directional
organization of the granular scaffold through eigenvalues and eigenvectors
of the void-space fabric tensor.

### 17.2 3D Thickness Measurement

> T. Hildebrand and P. Ruegsegger, "A new method for the model-independent
> assessment of thickness in three-dimensional images," *Journal of
> Microscopy*, vol. 185, no. 1, pp. 67–75, 1997.
> doi:10.1046/j.1365-2818.1997.1340694.x

Develops the "sphere-fitting" method for measuring trabecular thickness
and spacing in 3D images without requiring a structural model. We adopt
this approach conceptually in our tissue descriptor vector, where mean
trabecular thickness (Tb.Th) and spacing (Tb.Sp) are computed from the
phase fields φ_f and φ_i.

### 17.3 Porous Scaffold Design

> S. J. Hollister, "Porous scaffold design for tissue engineering," *Nature
> Materials*, vol. 4, no. 7, pp. 518–524, 2005.
> doi:10.1038/nmat1421

Comprehensive review of computational approaches to designing tissue
engineering scaffolds with specified pore architecture, mechanical properties,
and transport characteristics. Establishes the design target framework: a
scaffold must simultaneously satisfy constraints on porosity, pore size,
connectivity, mechanical strength, and permeability. We adopt this
multi-objective design philosophy in our energy landscape analysis, where
the dimensionless groups (β, Ca, Φ_r, Ψ, Γ) define a design space and
the energy landscape minimum ξ* maps to the achievable tissue architecture.

### 17.4 Superquadric Geometry

> A. Jaklic and A. Leonardis, "Superquadrics and their geometric properties,"
> in *Segmentation and Recovery of Superquadrics*, pp. 13–39, Springer, 2000.
> doi:10.1007/0-306-46857-1_2

Mathematical reference for superquadric (including superellipse and
superellipsoid) geometric properties: volume, surface area, curvature,
and inside-outside functions. We use the analytic formulas from this
reference for computing superellipse/superellipsoid areas, bounding radii,
and local curvatures in the contact detection algorithm (§7).

---

## 18. Jamming & Glass Transition (Mean-Field Theory)

**Introduced in:** V2.1

### 18.1 Primary Reference

> G. Parisi and F. Zamponi, "Mean-field theory of hard sphere glasses and
> jamming," *Reviews of Modern Physics*, vol. 82, pp. 789–845, 2010.
> doi:10.1103/RevModPhys.82.789

Comprehensive mean-field (replica) theory connecting the glass transition and
jamming in hard-sphere systems. Derives the Edwards entropy, equation of state,
and jamming density from first principles. Provides the theoretical framework
for understanding how packing fraction, coordination number, and mechanical
stability are related near the jamming point — directly relevant to our
granular scaffold compaction physics and the interpretation of DOE results
across varying φ_solid and granule size.

### 18.2 Key Concepts Used

- **Jamming density** φ_J as a function of spatial dimension and preparation
  protocol — informs our choice of φ_solid_target bounds [0.55, 0.85].
- **Isostaticity**: the minimum coordination number z_iso = 2d for frictionless
  spheres at jamming. Our periodic-BC simulations measure z as a diagnostic.
- **Scaling near jamming**: excess contacts Δz ~ (φ − φ_J)^{1/2}, pressure
  P ~ (φ − φ_J)^{Δ−1} with Hertzian exponent Δ = 5/2. Connects to the
  energy landscape decomposition (§12).

---

## 19. DEM Particle Packing Generation

**Introduced in:** V2.1

### 19.1 Primary Reference — Lubachevsky-Stillinger Algorithm

> B. D. Lubachevsky and F. H. Stillinger, "Geometric properties of random
> disk packings," *Journal of Statistical Physics*, vol. 60, nos. 5–6,
> pp. 561–583, 1990. doi:10.1007/BF01025983

> B. D. Lubachevsky, "How to simulate billiards and similar systems,"
> *Journal of Computational Physics*, vol. 94, no. 2, pp. 255–283, 1991.
> doi:10.1016/0021-9991(91)90222-7

The Lubachevsky-Stillinger (LS) protocol is the standard method for
generating jammed packings at prescribed packing fractions. The algorithm:

1. **Place particles at reduced (deflated) radii** so that RSA can position
   all N particles without overlap (achievable up to φ_RSA ≈ 0.30–0.38 in 3D).
2. **Inflate radii gradually** from the deflated state toward target size.
3. **At each inflation step, relax overlaps** via soft repulsive forces
   (or event-driven elastic collisions in the original formulation).
4. **Result**: a mechanically stable (jammed) packing at the target φ_solid
   with coordination number Z ≈ 2d (isostatic) for monodisperse frictionless
   spheres, or Z ≈ 4–8 for bidisperse/polydisperse systems.

The key insight is that gradual inflation forces particles to explore
configuration space and find stable contact networks, unlike RSA alone
which saturates well below random close packing.

### 19.2 Equations Adopted

**Deflation factor** (initial radius fraction for RSA placement):

```
α_start = (φ_safe / φ_target)^(1/3)        3D volume scaling
```

where `φ_safe = 0.20` ensures RSA can place all particles (well below
the RSA saturation limit of ~0.38 in 3D).

**Inflation schedule** (quadratic ease-in, more time near jamming):

```
t = step / n_steps                          normalised time [0, 1]
α(t) = α_start + (1 − α_start) × [1 − (1 − t)²]
```

This spends more steps near α ≈ 1.0 where overlap resolution is hardest.

**Overlap repulsion** (Hertz-like, during relaxation sub-steps):

```
F_rep = k_rep × δ^{3/2}                    δ = R_i + R_j − d_ij
```

with `k_rep = 5.0`, applied along the centre-to-centre unit normal.
Overdamped dynamics with velocity cap `v_max = 0.3 × ⟨R_target⟩` prevents
numerical instability.

### 19.3 Parameters in Code

| Parameter | Value | Unit | Role |
|-----------|-------|------|------|
| `packing_inflate_phi_safe` | 0.20 | — | Initial deflated φ for RSA |
| `packing_settle_steps` | 400 | — | Number of inflation steps |
| `packing_relax_substeps` | 15 | — | Overlap relaxation per inflation step |
| k_rep | 5.0 | nN/µm^1.5 | Hertz-like repulsion stiffness |
| v_cap | 0.3 × ⟨R⟩ | µm/step | Max displacement per sub-step |
| dt_settle | 0.02 | — | Relaxation micro-step size |

### 19.4 Secondary Reference — RSA + Compression

> E. M. B. Campello and K. R. Cassares, "Rapid generation of particle packs
> at high packing ratios for DEM simulations of granular compacts," *Latin
> American Journal of Solids and Structures*, vol. 13, no. 1, pp. 23–50, 2016.
> doi:10.1590/1679-78251694

Presents efficient algorithms for generating dense random packings of
polydisperse spheres suitable for DEM simulation. Compares RSA (random
sequential addition) with dynamic compression methods and characterises
packing quality (coordination number, radial distribution function) as a
function of target packing fraction.

### 19.5 Relevance to GELS

- **Lubachevsky-Stillinger inflate-and-relax**: our packing protocol (V2.1)
  uses deflated RSA placement (α ≈ 0.6–0.7) followed by 400-step inflation
  with Hertz-like overlap relaxation, achieving Z ≈ 6–8 for bidisperse
  packings at φ_solid = 0.55–0.85.
- **Polydisperse & bidisperse packings**: both references analyse how size
  ratio affects achievable packing fraction and coordination number —
  directly relevant to our DOE factors R_func_mean and R_inert_mean.
- **Periodic boundary conditions**: validates that periodic packings converge
  to bulk statistics faster than wall-bounded packings at the same N,
  supporting our switch to `boundary_mode="periodic"` in the V2.1 DOE.
- **Jamming verification**: coordination number Z reported after settling
  confirms isostatic or hyperstatic state (Z ≥ 2d = 6 in 3D).

---

## 20. Superellipsoid Packing & Jamming

**Introduced in:** V2.1

### 20.1 Foundational Result — Non-Spherical Dense Packing

> A. Donev, I. Cisse, D. Sachs, E. A. Variano, F. H. Stillinger,
> R. Connelly, S. Torquato, P. M. Chaikin, "Improving the Density of
> Jammed Disordered Packings Using Ellipsoids," *Science*, 303(5660),
> 990–993, 2004. doi:10.1126/science.1093010

Demonstrated that any deviation from spherical shape increases random
packing density: ellipsoids randomly pack to φ = 0.68–0.74 vs ~0.64 for
spheres, with ~10 contacts/particle vs 6. Foundational for all subsequent
non-spherical packing work.

### 20.2 Superellipsoid DEM Expansion (Inflate-and-Relax)

> G. W. Delaney and P. W. Cleary, "The packing properties of
> superellipsoids," *EPL (Europhysics Letters)*, 89(3), 34002, 2010.

Dynamic particle expansion (Lubachevsky-Stillinger for soft DEM) applied
to superellipsoids. MRJ packings reach φ ≈ 0.72–0.74. Packing density
increases with blockiness deviation from 2 (sphere). Our `_settle_packing_3d`
inflate-and-relax algorithm is based on this approach.

### 20.3 Definitive Superellipsoid Jamming Data

> Y. Yuan, K. VanderWerf, M. D. Shattuck, C. S. O'Hern, "Jammed
> packings of 3D superellipsoids with tunable packing fraction,
> coordination number, and ordering," *Soft Matter*, 15(47), 9751–9761,
> 2019. doi:10.1039/C9SM01932D

Tested 200+ superellipsoid shapes via athermal quasi-static compression.
Key results: superellipsoid packings are **hypostatic** (Z_J < Z_iso);
φ_J depends on ≥2 independent shape parameters in 3D; packings are
tuneable in φ, Z, and orientational order via protocol choice.

### 20.4 Optimal Superballs

> Y. Jiao, F. H. Stillinger, S. Torquato, "Optimal Packings of
> Superballs," *Physical Review E*, 79, 041309, 2009.

> Y. Jiao, F. H. Stillinger, S. Torquato, "Distinctive Features
> Arising in Maximally Random Jammed Packings of Superballs,"
> *Physical Review E*, 81, 041304, 2010.

Adaptive Shrinking Cell (ASC) method for superballs. φ increases rapidly
with shape parameter p: ~0.68 at p=1.5, ~0.82 at p=5 (sphere p=2 gives
~0.64). ASC formulates packing as constrained optimisation via Sequential
Linear Programming.

### 20.5 Packing Fraction Reference Table

| Shape | φ_J | Z | Source |
|-------|-----|---|--------|
| Sphere | ~0.64 | 6 | Donev 2004 |
| Ellipsoid (AR ≈ 1.4) | 0.68–0.71 | ~10 | Donev 2004 |
| General ellipsoid | up to 0.74 | ~12 | Donev 2004 |
| Superball p = 1.5 | ~0.68 | — | Jiao/Torquato 2009 |
| Superball p = 5 | ~0.82 | — | Jiao/Torquato 2009 |
| Superellipsoid (MRJ) | 0.72–0.74 | < 2d_f | Delaney/Cleary 2010 |

---

## 21. Multi-Contact DEM (MC-DEM) for Soft Particles

**Introduced in:** V2.2

### 21.1 Primary Reference

> K. Giannis, C. Schilde, J. H. Finke, A. Kwade, M. A. Celigueta,
> K. T. Tahir, H. Wiggers, S. Luding, "Stress based multi-contact model
> for discrete-element simulations," *Granular Matter*, 23, 5, 2021.
> doi:10.1007/s10035-020-01060-8

Introduces a stress-based multi-contact DEM variant where the trace of
the per-particle stress tensor, coupled with Poisson's ratio, makes all
contacts on a particle dependent on all other contacts. Validated for
**hydrogels**, rubber, and glass beads under confined/unconfined compression.

### 21.2 Physics

In standard Hertz DEM, each contact is computed independently. For soft
particles (hydrogels, E ~ 1–50 kPa) with many simultaneous contacts,
this ignores confinement stiffening: a particle squeezed from all sides
is stiffer at each contact than the same particle with a single contact.

MC-DEM corrects this by computing a per-particle volumetric overlap strain:

```
ε_V,i = Σ_c δ_c / (2 R_i)              sum over all contacts c on particle i
κ_i   = 1 + ν/(1 − 2ν) · ε_V,i         confinement correction factor
κ_ij  = (κ_i + κ_j) / 2                 pair-averaged correction
F_corrected = κ_ij × F_Hertz            scaled repulsive force
```

For nearly incompressible materials (ν → 0.5), the correction is strong:
ν = 0.49 → ν/(1−2ν) = 24.5. A particle with 7 contacts at 1% overlap
sees κ ≈ 1.9 (nearly 2× stiffening).

### 21.3 Parameters in Code

| Parameter | Default | Unit | Role |
|-----------|---------|------|------|
| `mc_dem_enabled` | `True` | — | Enable/disable MC-DEM correction |
| `mc_dem_kappa_max` | 5.0 | — | Cap on confinement multiplier |
| `poisson_ratio` | 0.45 | — | Drives correction magnitude via ν/(1−2ν) |

### 21.4 Supporting References

> N. Brodu, J. A. Dijksman, R. P. Behringer, "Spanning the scales of
> granular materials through microscopic force imaging," *Nature
> Communications*, 6, 6361, 2015. doi:10.1038/ncomms7361

Experimental 3D force imaging in deformable hydrogel packings. Demonstrated
multi-contact nonlinear stiffening — the effect MC-DEM captures.

> N. Ghods, P. Poorsolhjouy, M. Gonzalez, S. Radl, "Discrete element
> modeling of strongly deformed particles in dense shear flows," *Powder
> Technology*, 401, 117288, 2022. doi:10.1016/j.powtec.2022.117288

Multi-contact force closure for dense soft particle shear. Calibrated
against a nonlocal formulation in the quasi-static limit.

> M. Hirsch et al., "Building block properties govern granular hydrogel
> mechanics through contact deformations," *Science Advances*, 8,
> eadd8570, 2022. doi:10.1126/sciadv.add8570

Experimental validation: microgel stiffness and size control macroscale
granular hydrogel mechanics via Hertzian contact. Direct validation data
for GELS.

---

## 22. Contact Detection for Non-Spherical DEM

**Introduced in:** V1.3 (common-normal), reviewed V2.2

### 22.1 Common-Normal Method (Used in GELS)

> C. Wellmann, C. Lillie, P. Wriggers, "A contact detection algorithm
> for superellipsoids based on the common-normal concept," *Engineering
> Computations*, 25(5), 432–461, 2008. doi:10.1108/02644400810881374

Reformulates 3D superellipsoid contact as 2D unconstrained optimisation
via Newton-Raphson with Levenberg-Marquardt. Contact points defined by
anti-parallel surface normals. Only valid for smooth convex particles.
This is the algorithm used in `find_contact_superellipsoids_3d()`.

### 22.2 Energy Conservation Proof

> R. B. Canelas et al., "A common-normal-based framework for efficient
> ellipse contact detection in discrete element modelling," *Computers
> and Geotechnics*, 188, 2025. doi:10.1016/S0266352X25004732

Compares four contact methods: intersection, geometric potential, midway,
and common-normal. Key finding: **only the common-normal method guarantees
energy conservation**. Others introduce spurious energy gain/loss. Validates
our choice of contact algorithm.

### 22.3 Signed Distance Field (SDF) Approach

> Z. Lai, S. Zhao, J. Zhao, L. Huang, "Signed distance field framework
> for unified DEM modeling of granular media with arbitrary particle
> shapes," *Computational Mechanics*, 70, 763–793, 2022.
> doi:10.1007/s00466-022-02220-8

Generic SDF-based interface for arbitrary shapes. Recovers classical shapes
(superellipsoid, polyhedron, spherical harmonics) as special cases.
Energy-conserving node-to-surface contact detection.

> Z. Lai, Y. T. Feng, J. Zhao, L. Huang, "Unifying the contact in
> signed distance field-based and conventional discrete element methods,"
> *Computers and Geotechnics*, 173, 106560, 2024.
> doi:10.1016/j.compgeo.2024.106560

Bridges SDF-DEM and conventional Hertz contact. Establishes parameter
mapping between SDF contact potentials and standard DEM parameters.

> O. R. Gouveia, J. M. Guedes, R. B. Ruben, "Contact detection in
> computational mechanics: a signed distance field approach for convex
> superelliptical bodies," *Computational Mechanics*, 2025.
> doi:10.1007/s00466-025-02666-6

Introduces Gap Distance Field (GDF) concept for 2D superellipses.
Reformulates contact as unconstrained minimisation over GDF. More robust
for near-degenerate configurations than Newton-Raphson.

### 22.4 Review

> Y. T. Feng, "Thirty years of developments in contact modelling of
> non-spherical particles in DEM: a selective review," *Acta Mechanica
> Sinica*, 39, 722343, 2023. doi:10.1007/s10409-022-22343-x

Comprehensive classification of shape representations and contact methods
for non-spherical DEM, covering all major approaches.

---

## 23. Deformable Particle DEM (Future: V3.0)

### 23.1 DDEM — Global + Local Deformation Modes

> J. Rojek, A. Zubelewicz, N. Madan, S. Nosewicz, "The discrete element
> method with deformable particles," *Int. J. Numer. Methods Eng.*,
> 114(8), 828–860, 2018. doi:10.1002/nme.5767

> J. Rojek et al., "3D formulation of the deformable discrete element
> method," *Int. J. Numer. Methods Eng.*, 122(14), 3335–3367, 2021.
> doi:10.1002/nme.6666

Deformation decomposed into **global mode** (uniform stress from all contacts)
and **local mode** (contact-specific). Particles change shape; new contacts
form as particles flatten. 2D (2018) and 3D (2021) formulations. ~2–5×
cost of rigid DEM.

### 23.2 Variational LS-DEM (Deformable Level Sets)

> T. Henzel and K. Karapiperis, "A Variational Formulation for Deformable
> Particle Simulations and its Level Set Discrete Element Method
> Implementation," arXiv:2602.12895, 2026.

Energetic variational formulation (Lagrange-d'Alembert principle) embedding
translational, rotational, and deformation degrees of freedom. Deformation
via evolving level sets. Not restricted to specific geometries. Claims
**computational cost of the same order of magnitude as rigid DEM** — the
most promising approach for a future GELS upgrade.

### 23.3 Granular Hydrogel Scaffold Design

> S. Feng et al., "Practical Guide to the Design of Granular Hydrogels
> for Customizing Complex Cellular Microenvironments," *Adv. Healthcare
> Materials*, 14(27), e01947, 2025. doi:10.1002/adhm.202501947

Review of inter- and intra-microgel design factors for granular hydrogel
scaffolds. Nondirected packing is the mainstream assembly strategy.
Provides experimental context for GELS parameter choices.

---

## 24. Summary of Energy Landscape Parameters

(Renumbered from §20)

| Parameter | Value | Unit | Source | Section |
|-----------|-------|------|--------|---------|
| σ_cell | ~0.001 | kPa | Motor-clutch at hydrogel stiffness | §2, §12 |
| σ_0 | 0.005·E_eff | kPa | ODE-consistent elastic prefactor | §12 |
| ξ_J | ~0.25 | — | Proximity to jamming (ξ_max - Ψ) | §10, §12 |
| c_yield | 0.25 | — | HB yield prefactor (Olsson & Teitel) | §10, §12 |
| Δ | 1.5 | — | O'Hern Hertzian exponent | §10, §12 |
| Π | 3·σ_cell | kPa | Osmotic void pressure | §13, §14 |
| k_frust | 1.5·σ_cell | kPa | Inert frustration prefactor | §12 |
| γ_surface | 2.5·σ_cell·(0.3+mismatch) | kPa·µm | Interfacial tension | §14, §12 |
| η_eff | ~1 | kPa·h | Effective viscosity for compaction | §6, §15 |
