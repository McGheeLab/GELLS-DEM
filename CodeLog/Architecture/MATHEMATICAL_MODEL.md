# Mathematical Framework for Cell-Driven Granular Scaffold Remodeling

**Version:** V1.6
**Last updated:** 2026-03-15
**Associated code:** `new_dem_0.py`

---

## 1. Introduction

We develop a multiscale mathematical framework for the mechanical remodeling of
granular hydrogel scaffolds driven by encapsulated living cells. The system comprises
$N$ rigid hydrogel granules --- functional (cell-laden) and inert (passive) --- confined
within a bounded domain. Three coupled processes govern scaffold evolution:

1. **Cell-driven compaction.** Cells seeded on functional granules attach, spread,
   form focal adhesions, and generate traction forces that pull neighboring granules
   together through intercellular bridges.
2. **Granular contact mechanics.** Hertzian elasticity, surface-chemistry-dependent
   friction, and DMT adhesion govern granule-granule and granule-wall interactions.
3. **Interstitial fluid transport.** The evolving void network controls hydraulic
   permeability and nutrient delivery, coupling scaffold architecture to tissue
   viability.

The discrete element model (DEM) resolves the particle scale (Sections 2--3), while
coarse-grained continuum descriptions capture tissue-scale behavior (Sections 4--5).
Dimensionless groups (Section 6) identify the dominant physics, and a tissue
architecture descriptor (Section 7) enables quantitative comparison to organ
microstructure targets.

**Units.** Throughout this document: micrometres ($\mu$m) for length, nanonewtons (nN)
for force, hours (h) for time, and kilopascals (kPa) for elastic modulus. Where SI
units are required for dimensional consistency, explicit conversion factors are noted.

---

## 2. Microscale: Discrete Element Model

### 2.1 Equations of Motion

The system is deeply overdamped: inertial forces are negligible compared to viscous
drag at the hydrogel length scale ($\text{Re} \ll 1$). Each granule $i$ obeys the
overdamped Langevin equation

$$\gamma_i \frac{d\mathbf{x}_i}{dt} = \mathbf{F}_i^{\text{contact}} + \mathbf{F}_i^{\text{cell}} + \mathbf{F}_i^{\text{wall}} + \mathbf{F}_i^{\text{noise}}$$

where the translational drag coefficient is

$$\gamma_i = \gamma_0 \, R_i$$

with $\gamma_0$ a dimensionless drag scale and $R_i$ the equivalent radius of granule
$i$. The terms on the right-hand side represent, respectively: (1) contact forces from
neighboring granules (Hertz repulsion, DMT adhesion, and tangential friction), (2)
cell-mediated traction forces transmitted through intercellular bridges, (3) repulsive
wall forces at domain boundaries, and (4) active noise representing stochastic
cell-driven fluctuations.

Time integration uses a forward Euler scheme,

$$\mathbf{x}_i(t + \Delta t) = \mathbf{x}_i(t) + \frac{\mathbf{F}_i^{\text{tot}}}{\gamma_i} \Delta t$$

with a velocity cap $|\mathbf{v}_i| \leq v_{\max}$ to prevent numerical blowup from
large transient forces.

### 2.2 Hertzian Contact Force

The normal contact force between two elastic granules follows the Hertz theory for
non-adhesive elastic spheres (Johnson, 1985):

$$F_n = \frac{4}{3} E^* \sqrt{R^*} \, \delta^{3/2}$$

where $\delta = R_i + R_j - |\mathbf{x}_j - \mathbf{x}_i|$ is the overlap
(penetration depth), and the effective elastic modulus and effective radius are

$$E^* = \frac{E}{2(1 - \nu^2)}, \qquad R^* = \frac{R_i R_j}{R_i + R_j}$$

Here $E$ is the Young's modulus of the hydrogel and $\nu$ is its Poisson ratio. The
factor of 2 in the denominator of $E^*$ accounts for equal-modulus contact between two
deformable bodies. For granule-wall contact, the wall is treated as a rigid half-space,
giving $E^*_{\text{wall}} = E / (1 - \nu^2)$ and $R^* = R_i$.

**Unit conversion.** With $E$ in kPa ($= \text{nN}/\mu\text{m}^2 \times 10^{-3}$),
$R^*$ in $\mu$m, and $\delta$ in $\mu$m, the force is obtained in nN via

$$F_n \; [\text{nN}] = \frac{4}{3} \, E^*_{\text{Pa}} \, \sqrt{R^*_{\mu\text{m}}} \; \delta_{\mu\text{m}}^{3/2} \times 10^{-3}$$

where $E^*_{\text{Pa}} = E_{\text{kPa}} \times 10^3 / [2(1 - \nu^2)]$.

For superellipsoidal granules (Section 2.8), $R^*$ is replaced by the harmonic mean of
the local radii of curvature at the contact point, computed from the parametric surface
geometry.

### 2.3 Friction (Gong 2006, Pitenis 2014)

Hydrogel surfaces exhibit area-dependent friction governed by surface chemistry rather
than Coulombic mechanics. The tangential friction force opposing relative sliding is

$$F_{\text{fric}} = \tau_0 \, A_c \, \tanh\!\left(\frac{|v_t|}{v_{\text{ref}}}\right)$$

where $\tau_0$ is the interfacial shear stress (Pa), $A_c = \pi R^* \delta$ is the
Hertzian contact area ($\mu\text{m}^2$), $v_t$ is the relative tangential velocity at
the contact point, and $v_{\text{ref}}$ is a regularization velocity that smooths the
discontinuity at zero sliding speed.

The interfacial shear stress depends on the surface chemistry of the contacting pair:

| Pair type | Surface chemistry | $\tau_0$ (Pa) |
|-----------|-------------------|---------------|
| Functional--functional | Collagen I -- collagen I | 2000 |
| Inert--functional | Bare gel -- collagen I | 500 |
| Inert--inert | Bare gel -- bare gel | 50 |

These values reflect the hierarchy of hydrogen bonding and entanglement between
collagen-coated and uncoated hydrogel surfaces (Gong, 2006; Pitenis et al., 2014).

### 2.4 DMT Adhesion

The Derjaguin-Muller-Toporov (DMT) model provides a constant adhesive contribution
during contact:

$$F_{\text{adh}} = 2\pi W R^*$$

where $W$ is the work of adhesion (J/m$^2$), which depends on the pair type as shown below.

| Pair type | $W$ (J/m$^2$) |
|-----------|---------------|
| Functional--functional | 0.002 |
| Inert--functional | 0.001 |
| Inert--inert | 0.0005 |

The net normal contact force is

$$F_n^{\text{net}} = F_n^{\text{Hertz}} - F_{\text{adh}}$$

which can become negative (attractive) when the Hertzian repulsion is small relative to
adhesion, stabilizing granule contacts at small overlaps. In simulation units, the
adhesion force is computed as $F_{\text{adh}} = 2\pi W R^* \times 10^3$ nN to convert
from the SI product $[\text{J/m}^2][\mu\text{m}]$.

### 2.5 Motor-Clutch Cell Traction (Chan & Odde 2008)

Cells encapsulated within functional granules generate traction forces via the
motor-clutch mechanism. Following Chan and Odde (2008), the steady-state traction
force per cell is

$$F_{\text{cell}} = F_{\text{stall}} \cdot \beta \cdot \xi \cdot m(t)$$

where:

- $F_{\text{stall}} = n_m F_m$ is the total motor stall force, with $n_m$ motor
  complexes (stress fibers) each generating stall force $F_m$.

- $\beta = k_{\text{sub}} / (k_{\text{sub}} + k_{\text{opt}})$ is the
  stiffness-dependent engagement ratio. On soft substrates ($k_{\text{sub}} \ll k_{\text{opt}}$),
  clutches disengage before motors stall, yielding low force. On stiff substrates
  ($k_{\text{sub}} \gg k_{\text{opt}}$), $\beta \to 1$ and cells reach their maximum
  traction capacity.

- $k_{\text{sub}} = \pi E a / (1 - \nu^2)$ is the effective substrate stiffness
  sensed by a cell of radius $a = d_{\text{cell}} / 2$ on a hydrogel of modulus $E$
  and Poisson ratio $\nu$. This represents the stiffness of a rigid punch on an elastic
  half-space.

- $k_{\text{opt}} = n_c k_c$ is the optimal (clutch ensemble) stiffness, with $n_c$
  integrin clutch clusters each of spring constant $k_c$.

- $\xi = k_{\text{on}} / (k_{\text{on}} + k_{\text{off}})$ is the steady-state
  fraction of engaged clutches, determined by the clutch binding rate $k_{\text{on}}$
  and baseline unbinding rate $k_{\text{off}}$.

- $m(t)$ is the focal adhesion maturity, which ramps from 0 to 1 at a rate
  $r_{\text{FA}}$ (h$^{-1}$) after cell spreading initiates:
  $m(t) = \min(1, \, r_{\text{FA}} \cdot t_{\text{spread}})$.

The force per cell is capped at $F_{\max}$ to prevent unphysical traction at extreme
parameter combinations. Stiffness-dependent engagement ($\beta$) also modulates the
cell spreading rate: the effective spreading duration is
$\tau_{\text{spread}}^{\text{eff}} = \tau_{\text{spread}} / \max(0.2, \beta)$, so
that cells on stiffer substrates spread faster.

### 2.6 Bridge Formation Kinetics

Cell-cell bridges form stochastically between functional granule pairs separated by a
gap $d < L_s$ (the filopodial sensing distance). Only cells that have reached the
SPREADING or PROLIFERATING state with sufficient focal adhesion maturity
($m \geq m_{\min}$) are eligible.

**Bridge initiation.** Each eligible cell attempts to form a bridge as a Poisson
process modulated by proximity:

$$P_{\text{bridge}}(\Delta t) = \left[1 - \exp\!\left(-r_b \, \Delta t\right)\right] \cdot \left(1 - \frac{d}{L_s}\right)$$

where $r_b$ is the attempt rate (h$^{-1}$), $d$ is the surface-to-surface gap, and
$L_s$ is the sensing distance. The proximity factor $(1 - d/L_s)$ reflects the
decreasing probability that a filopodium reaches a more distant target.

**Force ramp.** A newly formed bridge ramps its force linearly over a formation time
$\tau_f$:

$$F_{\text{bridge}}(t) = F_{\text{cell}} \cdot \min\!\left(1, \; \frac{t_{\text{age}}}{\tau_f}\right)$$

where $t_{\text{age}}$ is the time since bridge commitment and $F_{\text{cell}}$ is
the motor-clutch force computed from the average focal adhesion maturity of the two
granules.

**Bridge persistence and senescence.** Committed bridges persist across timesteps.
A bridge ruptures if the gap exceeds the break distance $d > d_{\text{break}}$, causing
the cell to transition to a senescent state. Bridges that sustain mechanical load for a
duration exceeding $\tau_{\text{sen}}$ (bridge senescence time) also transition to
senescence, modeling contact-inhibited quiescence.

**Bridge parameters.** The kinetic parameters are:

| Parameter | Symbol | Default | Units |
|-----------|--------|---------|-------|
| Attempt rate | $r_b$ | 0.3 | h$^{-1}$ |
| Formation time | $\tau_f$ | 2.0 | h |
| Senescence time | $\tau_{\text{sen}}$ | 24.0 | h |
| Min FA maturity | $m_{\min}$ | 0.3 | -- |
| Break gap | $d_{\text{break}}$ | 60.0 | $\mu$m |

### 2.7 Rotational Dynamics

Off-center contact forces generate torques that rotate non-spherical granules. In the
overdamped regime, angular dynamics are governed by:

**2D.** For superellipse granules with semi-axes $(a, b)$:

$$\gamma_{\text{rot}} \frac{d\theta}{dt} = \sum_c \tau_c, \qquad \gamma_{\text{rot}} = \gamma_0^{\text{rot}} \frac{a^2 + b^2}{2}$$

where $\tau_c = (\mathbf{r}_c - \mathbf{x}_i) \times \mathbf{F}_c$ is the torque from
contact $c$, and $\gamma_0^{\text{rot}}$ is the rotational drag scale.

**3D.** For superellipsoid granules with semi-axes $(a, b, c)$, orientation is
represented by unit quaternions $\mathbf{q} = (w, x, y, z)$ with $|\mathbf{q}| = 1$.
The overdamped angular dynamics are:

$$I_{\text{eff}} \frac{d\boldsymbol{\omega}}{dt} = \sum_c \boldsymbol{\tau}_c, \qquad I_{\text{eff}} = \gamma_0^{\text{rot}} \frac{a^2 + b^2 + c^2}{3}$$

Quaternion integration follows the first-order update:

$$\mathbf{q}(t + \Delta t) = \text{normalize}\!\left[\mathbf{q}(t) + \frac{\Delta t}{2} \, [0, \boldsymbol{\omega}] \otimes \mathbf{q}(t)\right]$$

where $\otimes$ denotes the Hamilton product and $[0, \boldsymbol{\omega}]$ is the
pure quaternion $(0, \omega_x, \omega_y, \omega_z)$. An angular velocity cap
$|\boldsymbol{\omega}| \leq \omega_{\max}$ prevents unphysical rapid rotation.

### 2.8 Superellipsoid Shape Model

Granule shape is parameterized as a 3D superellipsoid:

$$\left(\left|\frac{x}{a}\right|^{n_1} + \left|\frac{y}{b}\right|^{n_1}\right)^{n_2/n_1} + \left|\frac{z}{c}\right|^{n_2} = 1$$

where $(a, b, c)$ are the semi-axes, $n_1$ is the equatorial blockiness exponent, and
$n_2$ is the meridional blockiness exponent. Special cases include:

| Shape | Parameters |
|-------|-----------|
| Sphere | $a = b = c$, $n_1 = n_2 = 2$ |
| Ellipsoid | $a \neq b \neq c$, $n_1 = n_2 = 2$ |
| Cylinder-like | $n_1, n_2 > 2$ |
| Octahedron-like | $n_1, n_2 < 2$ |

In 2D, the superellipse specialization applies:

$$\left|\frac{x}{a}\right|^{n} + \left|\frac{y}{b}\right|^{n} = 1$$

The parametric surface representation is:

$$\mathbf{r}(\eta, \omega) = \begin{pmatrix} a \, \text{sgn}(\cos\eta)|\cos\eta|^{2/n_2} \, \text{sgn}(\cos\omega)|\cos\omega|^{2/n_1} \\ b \, \text{sgn}(\cos\eta)|\cos\eta|^{2/n_2} \, \text{sgn}(\sin\omega)|\sin\omega|^{2/n_1} \\ c \, \text{sgn}(\sin\eta)|\sin\eta|^{2/n_2} \end{pmatrix}$$

for $\eta \in [-\pi/2, \pi/2]$ and $\omega \in [-\pi, \pi)$.

**Contact detection** uses the common normal method: for each candidate pair, a
Newton-Raphson solver finds the parametric coordinates $(\eta_i, \omega_i, \eta_j, \omega_j)$
such that the outward normals at the closest surface points are anti-parallel and
collinear with the line connecting them. The effective radius of curvature at the contact
point is obtained from the local surface geometry, replacing the sphere-based $R^*$ in
the Hertz force law.

The volume of a superellipsoid is (Jaklic and Leonardis, 2000):

$$V = 2abc \, \epsilon_1 \epsilon_2 \, B\!\left(\frac{\epsilon_1}{2} + 1, \, \epsilon_1\right) B\!\left(\frac{\epsilon_2}{2}, \, \frac{\epsilon_2}{2}\right)$$

where $\epsilon_1 = 2/n_1$, $\epsilon_2 = 2/n_2$, and $B$ is the Beta function.

### 2.9 Active Noise

Cell-driven fluctuations on functional granules are modeled as Gaussian white noise:

$$\mathbf{F}_i^{\text{noise}} = \sqrt{\frac{2 \gamma_i T_{\text{active}}}{\Delta t}} \; \boldsymbol{\eta}_i(t)$$

where $T_{\text{active}}$ is an effective active temperature (nN$\cdot\mu$m) and
$\boldsymbol{\eta}_i(t)$ is a vector of independent standard normal random variables.
Inert granules receive no active noise ($T_{\text{active}} = 0$ for $g_i = 1$).

### 2.10 Cell Lifecycle

Each cell progresses through a discrete state machine:

$$\text{ATTACHED} \to \text{SPREADING} \to \text{PROLIFERATING} \to \text{BRIDGING} \to \text{SENESCENT}$$

**Attachment.** At time $t \geq t_{\text{onset}}$, cells attach with sigmoidal kinetics:

$$f_{\text{att}}(t) = \frac{1}{1 + \exp\!\left[-3\,(t - t_{\text{onset}} - t_{1/2}) / t_{1/2}\right]}$$

where $t_{1/2}$ is the half-time for attachment. With default parameters
($t_{\text{onset}} = 0$, $t_{1/2} = 0$), attachment is instantaneous.

**Spreading.** Attached cells transition from spheres (diameter $d_{\text{cell}} = 20\;\mu$m)
to oblate ellipsoids (height $h_{\text{spread}} = 5\;\mu$m) with volume conservation:

$$V_{\text{cell}} = \frac{4}{3}\pi \left(\frac{d}{2}\right)^3 = \frac{4}{3}\pi a_{\text{spread}}^2 \frac{h}{2}$$

$$\Rightarrow \quad A_{\text{spread}} = \pi a_{\text{spread}}^2 = \frac{\pi d^3}{4h}$$

The spread fraction ramps linearly from 0 to 1 over an effective duration
$\tau_{\text{spread}}^{\text{eff}} = \tau_{\text{spread}} / \max(0.2, \beta)$, where
$\beta$ is the motor-clutch engagement ratio. Projected area interpolates linearly
between the spherical and spread values.

**Overcrowding.** The maximum number of cells per granule is set by the ratio of
granule surface area to cell footprint, scaled by a coverage fraction:

$$n_{\max} = \left\lfloor \frac{\pi R^2 \cdot c_{\text{cov}}}{A_{\text{cell}}} \right\rfloor$$

Excess cells ($n_{\text{attached}} > n_{\max}$) transition to the SENESCENT state.

**Migration.** Mobile cells (ATTACHED, SPREADING, PROLIFERATING) undergo a random walk
on the granule surface, with angular displacement per timestep:

$$\Delta\theta \sim \mathcal{N}\!\left(0, \; \frac{v_{\text{mig}} \, \Delta t}{R_i}\right)$$

where $v_{\text{mig}}$ is the migration speed ($\mu$m/h).

---

## 3. Mesoscale: Coarse-Graining

### 3.1 Stress Tensor (Love-Weber)

The volume-averaged Cauchy stress tensor for the granular assembly is computed via the
Love-Weber formula:

$$\sigma_{\alpha\beta} = \frac{1}{V} \sum_{\text{contacts}\; c} f_\alpha^c \, \ell_\beta^c$$

where $f_\alpha^c$ is the $\alpha$-component of the contact force and
$\ell_\beta^c = x_\beta^j - x_\beta^i$ is the branch vector connecting the centers of
the two granules in contact. The summation runs over all contacts within the averaging
volume $V$.

The stress admits a decomposition into passive and active contributions:

$$\boldsymbol{\sigma} = \boldsymbol{\sigma}^{\text{contact}} + \boldsymbol{\sigma}^{\text{active}}$$

where $\boldsymbol{\sigma}^{\text{contact}}$ arises from Hertz, friction, and adhesion
forces, and $\boldsymbol{\sigma}^{\text{active}}$ arises from cell traction bridges.

### 3.2 Strain Rate Tensor

The coarse-grained strain rate tensor is

$$\dot{\varepsilon}_{\alpha\beta} = \frac{1}{2V} \sum_{i} \left(v_\alpha^i x_\beta^i + v_\beta^i x_\alpha^i\right) V_i$$

where $v_\alpha^i$ is the velocity component, $x_\beta^i$ is the position component,
and $V_i$ is the Voronoi volume of granule $i$.

### 3.3 Effective Viscosity

The effective shear viscosity of the granular assembly is defined through the ratio of
deviatoric stress to deviatoric strain rate:

$$\eta_{\text{eff}} = \frac{|\boldsymbol{\sigma}_{\text{dev}}|}{2\,|\dot{\boldsymbol{\varepsilon}}_{\text{dev}}|}$$

where $|\cdot|$ denotes the von Mises norm:
$|\boldsymbol{\sigma}_{\text{dev}}| = \sqrt{\tfrac{1}{2}\sigma_{ij}'\sigma_{ij}'}$ with
$\sigma_{ij}' = \sigma_{ij} - \tfrac{1}{d}\sigma_{kk}\delta_{ij}$ and $d$ is the
spatial dimension. This effective viscosity characterizes the scaffold's resistance to
cell-driven rearrangement and is expected to diverge near the jamming transition.

---

## 4. Macroscale: Continuum Model

### 4.1 Conservation Equations

The scaffold is treated as a three-phase mixture: functional granule phase ($\phi_f$),
inert granule phase ($\phi_i$), and interstitial void ($\phi_v$). In the absence of
mass exchange between phases, each solid phase satisfies a conservation equation:

$$\frac{\partial \phi_\alpha}{\partial t} + \nabla \cdot (\phi_\alpha \, \mathbf{v}_\alpha) = 0, \qquad \alpha \in \{f, \, i\}$$

The void fraction is determined by the saturation constraint:

$$\phi_v = 1 - \phi_f - \phi_i$$

These fields are computed on a regular Eulerian grid by stamping each granule as a
smoothed indicator function (tanh profile with interface width $w$):

$$\phi_\alpha(\mathbf{x}) = \max_{i \in \alpha} \left[\frac{1}{2}\left(1 - \tanh\frac{s_i(\mathbf{x})}{w}\right)\right]$$

where $s_i(\mathbf{x})$ is the approximate signed distance from point $\mathbf{x}$ to
the surface of granule $i$.

### 4.2 Momentum Balance (Overdamped)

In the overdamped limit, the momentum equation for phase $\alpha$ reduces to a force
balance:

$$0 = \nabla \cdot \boldsymbol{\sigma}_\alpha + \mathbf{f}_\alpha^{\text{contact}} + \mathbf{f}_\alpha^{\text{active}} - \zeta_\alpha \phi_\alpha \left(\mathbf{v}_\alpha - \mathbf{v}_f\right)$$

where $\boldsymbol{\sigma}_\alpha$ is the phase stress tensor,
$\mathbf{f}_\alpha^{\text{contact}}$ represents inter-phase contact forces,
$\mathbf{f}_\alpha^{\text{active}}$ is the cell traction body force density, and the
last term is the drag between the solid phase and the interstitial fluid, with
$\zeta_\alpha$ a drag coefficient.

### 4.3 Darcy Flow

The interstitial fluid velocity relative to the solid skeleton follows Darcy's law:

$$\phi_v \, \mathbf{v}_f = -\frac{k}{\mu} \nabla P$$

where $k$ is the hydraulic permeability ($\mu\text{m}^2$), $\mu$ is the fluid
viscosity (Pa$\cdot$s), and $P$ is the pore fluid pressure.

### 4.4 Kozeny-Carman Permeability

The permeability of the granular bed is estimated by the Kozeny-Carman relation:

$$k = \frac{\phi_v^3 \, d^2}{180 \, (1 - \phi_v)^2}$$

where $d$ is the mean granule diameter. This classical result assumes a bed of
monodisperse spheres; for polydisperse superellipsoidal packings, $d$ is taken as the
mean equivalent-sphere diameter. The strong dependence on porosity ($k \propto \phi_v^3$
for small changes) couples the transport properties tightly to compaction.

---

## 5. Mean-Field Compaction Model

The simplest useful model for scaffold compaction tracks the bulk functional volume
fraction $\phi_f(t)$. It collapses the microscale DEM into a single ordinary
differential equation that can be solved analytically in limiting cases.

### 5.1 Compaction Rate

$$\frac{d\phi_f}{dt} = -\phi_f \cdot \dot{\varepsilon}_{\text{comp}}$$

where the compaction strain rate is

$$\dot{\varepsilon}_{\text{comp}} = \frac{\sigma_{\text{cell}}(t) - \sigma_{\text{resist}}(\phi_{\text{solid}})}{\eta_{\text{eff}}}$$

### 5.2 Cell Stress (Volume-Averaged Motor-Clutch)

The cell-generated compressive stress is the volume average of all bridge forces:

$$\sigma_{\text{cell}}(t) = n_{\text{cell}} \cdot F_{\text{cell}}(E) \cdot f_{\text{bridge}}(t) \cdot m(t)$$

where $n_{\text{cell}}$ is the number density of bridging cells, $F_{\text{cell}}(E)$
is the motor-clutch force at substrate stiffness $E$, $f_{\text{bridge}}(t)$ is the
fraction of cells that have formed bridges (governed by the Poisson kinetics of
Section 2.6), and $m(t)$ is the mean focal adhesion maturity.

### 5.3 Resistance Stress (Jamming)

As the solid fraction $\phi_{\text{solid}} = \phi_f + \phi_i$ approaches the jamming
point $\phi_J$, the scaffold stiffens and resists further compaction:

$$\sigma_{\text{resist}} = \sigma_0 \left(\frac{\phi_{\text{solid}}}{\phi_J} - 1\right)^\alpha \quad \text{for } \phi_{\text{solid}} > \phi_J$$

$$\sigma_{\text{resist}} = 0 \quad \text{for } \phi_{\text{solid}} \leq \phi_J$$

The jamming fraction $\phi_J$ depends on particle shape. For random close packing of
spheres, $\phi_J \approx 0.64$ (Torquato et al., 2000). For ellipsoids,
$\phi_J \approx 0.64 + 0.08 (\text{AR} - 1)$ up to aspect ratio $\text{AR} \approx 2$
(Donev et al., 2004). The exponent $\alpha$ characterizes the stiffening law near
jamming; for frictionless spheres, $\alpha \approx 3/2$ (O'Hern et al., 2003).

### 5.4 Mechanical Equilibrium

Compaction halts when $\sigma_{\text{cell}} = \sigma_{\text{resist}}$, defining the
equilibrium solid fraction $\phi_{\text{solid}}^{\text{eq}}$. For weak cell forces
relative to scaffold stiffness (small Compaction number $\text{Ca}$, Section 6.1),
$\phi_{\text{solid}}^{\text{eq}} \approx \phi_J$ and the scaffold barely compacts. For
large $\text{Ca}$, the equilibrium approaches maximal packing.

---

## 6. Dimensionless Analysis

### 6.1 Key Dimensionless Groups

The system behavior is governed by the following dimensionless groups:

| Group | Symbol | Definition | Physical meaning | Typical range |
|-------|--------|-----------|-----------------|---------------|
| Motor-clutch engagement | $\beta$ | $k_{\text{sub}} / (k_{\text{sub}} + k_{\text{opt}})$ | Fraction of motor force transmitted to substrate | 0.01 -- 0.93 |
| Compaction number | $\text{Ca}$ | $\sigma_{\text{cell}} / E^*$ | Cell stress relative to granule stiffness | $10^{-4}$ -- $10^{-2}$ |
| Jamming proximity | $\Phi / \Phi_J$ | $\phi_{\text{solid}} / \phi_{\text{RCP}}$ | Distance from random close packing | 0.5 -- 1.2 |
| Darcy number | $\text{Da}$ | $k / L^2$ | Permeability relative to domain scale | $\sim 10^{-5}$ |
| Timescale ratio | $\tau_{\text{bio}} / \tau_{\text{mech}}$ | $\tau_{\text{bridge}} / \tau_{\text{damp}}$ | Biological vs. mechanical relaxation time | 100 -- 1000 |
| Clutch engagement | $\xi$ | $k_{\text{on}} / (k_{\text{on}} + k_{\text{off}})$ | Steady-state fraction of bound clutches | 0.5 -- 0.95 |

**Motor-clutch engagement $\beta$.** This is the master variable of the model. It
captures the nonlinear coupling between substrate stiffness and cell force output.
For soft gels ($E \sim 1$ kPa), $\beta \ll 1$ and cells generate negligible traction.
For stiff gels ($E \sim 50$ kPa), $\beta \to 1$ and cells approach their stall force.
Substituting the default parameters ($n_c = 75$, $k_c = 5.0$ nN/$\mu$m,
$a = 10\;\mu$m):

$$k_{\text{opt}} = 75 \times 5.0 = 375 \; \text{nN}/\mu\text{m}$$

$$k_{\text{sub}} = \pi \times E \times 10 / (1 - 0.45^2) \approx 40.0 \, E \; \text{nN}/\mu\text{m} \quad (E \text{ in kPa})$$

So $\beta = 40E / (40E + 375)$, giving $\beta = 0.10$ at $E = 1$ kPa and $\beta = 0.84$
at $E = 50$ kPa.

**Compaction number Ca.** This ratio determines whether cell forces are sufficient to
deform the granular packing. When $\text{Ca} \ll 1$, the scaffold is essentially rigid
on the cell force scale. Significant compaction requires $\text{Ca} \gtrsim \text{Ca}_c$
where $\text{Ca}_c$ is set by the jamming resistance.

**Timescale ratio.** The mechanical relaxation time
$\tau_{\text{damp}} = \gamma / E^* \sim 10^{-2}$ h is much shorter than the biological
timescales of bridge formation ($\tau_f \sim 2$ h) and senescence
($\tau_{\text{sen}} \sim 24$ h). This separation of scales justifies the quasistatic
(overdamped) approximation.

### 6.2 Scaling Laws

The engagement ratio $\beta$ controls the key scaling relationships:

- **Traction force:** $F_{\text{cell}} \propto \beta$, since the stall force, clutch
  engagement, and FA maturity are bounded.

- **Compaction efficiency:** The fraction of initial void space eliminated by
  cell-driven compaction scales with $\beta$: stronger engagement drives more
  compaction before equilibrium is reached.

- **Spreading rate:** $\tau_{\text{spread}}^{\text{eff}} = \tau_{\text{spread}} / \beta$
  (for $\beta > 0.2$), so stiffer substrates yield faster spreading and earlier onset
  of traction.

- **Permeability evolution:** As compaction increases $\phi_{\text{solid}}$, permeability
  decreases via the Kozeny-Carman relation. The feedback loop is:

$$\beta \uparrow \;\to\; \sigma_{\text{cell}} \uparrow \;\to\; \phi_{\text{solid}} \uparrow \;\to\; k \downarrow$$

### 6.3 Dimensionless Compaction ODE

Let $\Phi = \phi_f / \phi_{f,0}$ (normalized functional fraction) and
$T = t / \tau_{\text{bridge}}$ (normalized time). The mean-field ODE becomes:

$$\frac{d\Phi}{dT} = -\Phi \left[B(T) - R(\Phi)\right]$$

where:

- $B(T) = \sigma_{\text{cell}}(T) / \eta_{\text{eff}}$ is the dimensionless driving
  term, which ramps from zero as bridges form and saturates when all eligible cells are
  bridging.

- $R(\Phi) = \sigma_{\text{resist}}(\Phi) / \eta_{\text{eff}}$ is the dimensionless
  resistance, which is zero for $\Phi < \Phi_J / \phi_{f,0}$ and diverges as
  $\Phi \to \Phi_J / \phi_{f,0}$ from below.

Equilibrium is reached when $B(T^*) = R(\Phi^*)$. The final compaction
$\Phi^* = \phi_f^{\text{eq}} / \phi_{f,0}$ depends on $\beta$, cell density, and
granule stiffness through the dimensionless groups defined above.

---

## 7. Tissue Architecture Characterization

The goal of scaffold design is to produce a microstructure that recapitulates the
architecture of a target organ. We define a descriptor vector and an architectural
distance metric for quantitative comparison.

### 7.1 Descriptor Vector

The tissue architecture is characterized by a vector $\mathbf{d} \in \mathbb{R}^{n_d}$
whose components fall into six categories:

**Volume fractions.**
- $\text{BV/TV}$: bone (or solid) volume / total volume = $\phi_f + \phi_i$
- Porosity: $\phi_v = 1 - \text{BV/TV}$

**Morphometry (Hildebrand and Ruegsegger, 1997).**
- $\text{Tb.Th}$: trabecular (strand) thickness --- mean thickness of the solid phase
- $\text{Tb.Sp}$: trabecular separation --- mean pore diameter
- $\text{Tb.N}$: trabecular number = $\text{BV/TV} / \text{Tb.Th}$
- $\text{SMI}$: structure model index (0 = plates, 3 = rods, 4 = spheres)

**Topology.**
- $\chi$: Euler characteristic of the solid phase (genus proxy)
- Connectivity density: $\text{Conn.D} = (1 - \chi) / V$ (number of connections per
  unit volume)

**Spatial statistics.**
- $S_2(r)$: two-point correlation function of the solid phase
- $\xi$: correlation length (exponential decay constant of $S_2$)
- Chord length distribution: probability density of intercept lengths through solid
  and void phases

**Anisotropy (Harrigan and Mann, 1984).**
- MIL tensor: mean intercept length tensor
- $\text{DA}$: degree of anisotropy = $1 - \lambda_{\min} / \lambda_{\max}$
- $\text{FA}$: fractional anisotropy (analogous to diffusion tensor FA)

**Transport.**
- $\tau$: tortuosity (effective path length / straight-line distance)
- $k$: Kozeny-Carman permeability

### 7.2 Architectural Distance

Given a GELLS scaffold descriptor $\mathbf{d}^{\text{GELLS}}$ and a target organ
descriptor $\mathbf{d}^{\text{organ}}$ with population standard deviation
$\boldsymbol{\sigma}^{\text{organ}}$, the architectural distance is:

$$D_{\text{arch}} = \sqrt{\sum_{k=1}^{n_d} w_k \left(\frac{d_k^{\text{GELLS}} - d_k^{\text{organ}}}{\sigma_k^{\text{organ}}}\right)^2}$$

where $w_k$ are importance weights. Log-transforms are applied to scale-dependent
quantities (Tb.Th, Tb.Sp, $k$, $\xi$) before computing the distance, so that
fractional deviations are penalized equally across scales:

$$d_k \to \log_{10}(d_k) \quad \text{for scale-dependent descriptors}$$

### 7.3 Organ System Targets

Representative architectural parameters for potential scaffold targets are summarized
below. Values are drawn from quantitative histomorphometry and micro-CT studies.

| Organ | BV/TV | Tb.Th ($\mu$m) | Tb.Sp ($\mu$m) | Tb.N (mm$^{-1}$) | DA | $k$ ($\mu$m$^2$) |
|-------|-------|----------------|-----------------|-------------------|----|-------------------|
| Trabecular bone (vertebral) | 0.12--0.25 | 100--200 | 500--1000 | 1.0--1.8 | 0.3--0.7 | $10^1$--$10^3$ |
| Lung alveoli | 0.10--0.15 | 5--15 | 100--300 | 5--10 | 0.1--0.3 | $10^3$--$10^5$ |
| Liver (hepatic lobule) | 0.80--0.90 | 200--500 | 10--30 (sinusoid) | 2--4 | 0.1--0.2 | $10^0$--$10^1$ |
| Kidney cortex | 0.70--0.85 | 50--150 | 20--60 | 3--6 | 0.2--0.4 | $10^0$--$10^2$ |
| Cardiac muscle | 0.75--0.85 | 10--25 | 5--15 | 10--20 | 0.5--0.8 | $10^{-1}$--$10^1$ |
| Pancreatic islet | 0.60--0.75 | 30--80 | 10--30 | 5--10 | 0.1--0.2 | $10^0$--$10^2$ |
| Intestinal mucosa | 0.50--0.70 | 50--200 | 30--100 | 3--8 | 0.3--0.6 | $10^1$--$10^3$ |

These targets define the feasible design space: scaffold parameters ($E$, $\phi_f$,
$\phi_i$, granule size distribution, cell density) should be chosen to minimize
$D_{\text{arch}}$ for the desired organ.

---

## 8. Coupled Prediction Chain

The complete prediction chain links substrate stiffness to scaffold transport
properties and tissue architecture:

$$\beta(E) \;\to\; \sigma_{\text{cell}}(t; \beta) \;\to\; \phi_f(t) \;\to\; \phi_v(t) \;\to\; k(t) \;\to\; Q(t, \Delta P)$$

where $Q$ is the volumetric flow rate through the scaffold under a pressure drop
$\Delta P$. Each arrow represents a functional dependence:

1. **Stiffness sensing.** The substrate modulus $E$ determines the motor-clutch
   engagement $\beta$ and hence the cell traction force $F_{\text{cell}}$.

2. **Force generation.** $\sigma_{\text{cell}}(t)$ emerges from the combination of
   motor-clutch force, bridge kinetics, and focal adhesion maturation.

3. **Compaction.** The net stress drives compaction: $\phi_f(t)$ increases (functional
   granules pack more densely) while $\phi_v(t)$ decreases.

4. **Transport.** Permeability $k(t)$ tracks porosity through Kozeny-Carman, closing
   the loop between scaffold mechanics and nutrient transport.

5. **Architecture.** The final microstructure is characterized by the descriptor vector
   $\mathbf{d}$, which is compared to organ targets via $D_{\text{arch}}$.

The DEM simulation resolves steps 1--4 directly at the particle scale. The mean-field
model (Section 5) provides an analytical approximation that enables rapid parameter
exploration before committing to full DEM runs.

---

## 9. Experimental Validation

The model connects to experiments through four independent measurement modalities:

**Phase evolution from volumetric imaging.** Confocal or light-sheet microscopy
provides time-resolved maps of $\phi_f(\mathbf{x}, t)$ and $\phi_i(\mathbf{x}, t)$.
Spatially averaged trajectories $\phi_f(t)$ can be fit to the mean-field ODE
(Section 5) to extract the effective viscosity $\eta_{\text{eff}}$ and cell stress
$\sigma_{\text{cell}}$. The ratio $\sigma_{\text{cell}} / \eta_{\text{eff}}$ gives
the compaction rate directly.

**Interstitial flow from particle image velocimetry.** PIV with fluorescent
microspheres in the void space yields the pore-scale velocity field
$\mathbf{v}_f(\mathbf{x})$. Spatial averaging gives the Darcy flux, from which the
permeability $k(t)$ is extracted. Comparison to the Kozeny-Carman prediction
(Section 4.4) validates the relationship between compaction and transport.

**Cell traction from traction force microscopy.** Cells on hydrogel substrates of known
stiffness generate displacement fields measurable by embedded fluorescent beads.
Inversion yields the traction stress, which can be fit to the motor-clutch model
(Section 2.5) to determine $F_{\text{stall}}$, $k_{\text{opt}}$, and the engagement
curve $\beta(E)$.

**Tissue architecture from histology and micro-CT.** Histological sections or micro-CT
reconstructions of both the scaffold and native organ tissue yield the descriptor
vectors $\mathbf{d}^{\text{GELLS}}$ and $\mathbf{d}^{\text{organ}}$. The architectural
distance $D_{\text{arch}}$ (Section 7.2) quantifies how closely the scaffold
recapitulates the target organ. Minimization of $D_{\text{arch}}$ over the design
parameter space constitutes the scaffold optimization problem.

---

## 10. Computational Results: 24-Run Design of Experiments

The complete mathematical framework was applied to a 24-run fractional factorial
Design of Experiments (DOE). Each run simulates 3D granular scaffold compaction
in a $514 \times 514 \times 514\ \mu\text{m}$ domain with $\sim\!208$ superellipsoidal
granules over 10--38 hours. The five DOE factors are:

| Factor | Symbol | Low Level | High Level |
|--------|--------|-----------|------------|
| A: Modulus | $E$ | 2 kPa | 50 kPa |
| B: Cells/granule | $n_c$ | 4 | 8 |
| C: Functional radius | $R_f$ | 30 $\mu$m | 60 $\mu$m |
| D: Composition | $\phi_f/\phi_i$ | 0.2/0.4 | 0.4/0.2 |
| E: Adhesion/friction | $W_{\text{adh}}$, $\tau_0$ | low | high |

### 10.1 Mean-Field Model Fitting

The mean-field ODE (Section 5) was fitted to the $\phi_f(t)$ time series of each
run by optimizing $(\eta_{\text{eff}},\, \sigma_0,\, \alpha)$. All 23 runs with
data produced convergent fits. Runs with high substrate stiffness ($E = 50$ kPa)
exhibited the strongest compaction:

| Run | $E$ (kPa) | $R^2$ | $\eta_{\text{eff}}$ (nN$\cdot$h/$\mu$m$^2$) | $\Delta\phi_f$ |
|-----|-----------|-------|----------------------------------------------|-----------------|
| DOE_10 | 50 | 0.630 | 0.0023 | $-0.040$ |
| DOE_22 | 50 | 0.200 | 0.0054 | $-0.061$ |
| DOE_08 | 50 | 0.185 | 0.0043 | $-0.073$ |
| DOE_18 | 50 | 0.121 | 0.0049 | $-0.141$ |
| DOE_12 | 50 | 0.122 | 0.013 | $-0.035$ |

The fitted effective viscosity $\eta_{\text{eff}} \sim 10^{-3}$--$10^{-2}$
nN$\cdot$h/$\mu$m$^2$ is consistent with the overdamped granular assembly
at low volume fractions ($\phi_s/\phi_J \approx 0.92$). Runs with $E = 2$ kPa
produced $\beta < 0.2$, yielding weaker cell traction and minimal net compaction
($|\Delta\phi_f| < 0.01$), which limits the ODE's predictive power ($R^2 < 0$).

### 10.2 Coarse-Grained Continuum Quantities

The Love--Weber stress tensor (Section 3.1), strain-rate tensor (Section 3.2),
and effective viscosity (Section 3.3) were extracted from all 22 runs with
snapshot data. Key findings:

- **Contact pressure** ranges from $|P| \sim 0.01$ kPa ($E = 2$ kPa runs) to
  $|P| \sim 2$ kPa ($E = 50$ kPa runs), scaling approximately linearly with modulus.
- **Coordination number** at the final timepoint: $Z \in [0.13,\, 1.67]$.
  High-adhesion, high-modulus runs (DOE_08, DOE_18) reach $Z > 1.4$, approaching
  the frictional jamming onset $Z_J \approx 4$ from below.
- Pressure relaxation is pronounced: initial contact stresses decay by 50--90\%
  over the simulation duration as the packing equilibrates under cell-driven forces.

| Run | $P_0$ (kPa) | $P_f$ (kPa) | $Z_f$ | $\sigma_{\text{vM},f}$ (kPa) |
|-----|-------------|-------------|--------|-------------------------------|
| DOE_08 | $-1.94$ | $-0.09$ | 1.67 | 0.17 |
| DOE_18 | $-0.34$ | $-0.30$ | 1.45 | 0.52 |
| DOE_04 | $-1.21$ | $-0.29$ | 1.05 | 0.51 |
| DOE_20 | $-0.34$ | $-0.17$ | 1.36 | 0.30 |

### 10.3 Dimensionless Analysis

The motor-clutch engagement ratio $\beta$ and cellular capillary number $\text{Ca}$
were computed for all 23 runs (Section 6):

$$
\beta = \frac{k_{\text{sub}}}{k_{\text{sub}} + k_{\text{opt}}} \in [0.181,\; 0.846]
$$

$$
\text{Ca} = \frac{\sigma_{\text{cell}}}{E^*} \in [0.21,\; 17.66]
$$

The data collapse by $\beta$ reveals two distinct regimes:

1. **Low-$\beta$ regime** ($\beta < 0.3$, $E = 2$ kPa): Cell traction is
   substrate-limited. Compaction is negligible ($|\Delta\phi_f| < 0.01$) regardless
   of other parameters. The system is in a force-starved state.

2. **High-$\beta$ regime** ($\beta > 0.8$, $E = 50$ kPa): Cell traction saturates
   toward $F_{\text{stall}}$. Compaction magnitude is controlled by adhesion,
   composition ratio, and cell number rather than substrate stiffness alone.

The capillary number Ca shows an inverse relationship with compaction: low-Ca
(high-modulus) runs exhibit the largest $|\Delta\phi_f|$ because the scaffold is
stiff enough to transmit cell forces without excessive deformation at each contact.

### 10.4 Tissue Architecture Descriptors

The 22-descriptor tissue vector (Section 7.1) was computed from the final 3D
phase fields of each run. Key descriptors across the DOE:

| Descriptor | Symbol | Range | Units |
|-----------|--------|-------|-------|
| Bone volume fraction | BV/TV | 0.139 -- 0.576 | -- |
| Trabecular thickness | Tb.Th | 19.2 -- 39.1 | $\mu$m |
| Trabecular spacing | Tb.Sp | 23.8 -- 64.1 | $\mu$m |
| Trabecular number | Tb.N | 0.009 -- 0.020 | $\mu$m$^{-1}$ |
| Structure Model Index | SMI | $-2.71$ -- 1.80 | -- |
| Degree of anisotropy | DA | 1.05 -- 1.37 | -- |
| Tortuosity | $\tau$ | 14.2 -- 349.7 | -- |
| Kozeny-Carman permeability | $K$ | 2.7 -- 85.4 | $\mu$m$^2$ |

The composition factor (D) dominates BV/TV: runs with $\phi_f = 0.4$ produce
$\text{BV/TV} \in [0.35,\, 0.58]$ while $\phi_f = 0.2$ runs yield
$\text{BV/TV} \in [0.14,\, 0.20]$. SMI transitions from plate-like ($\text{SMI} < 0$)
at high BV/TV to rod-like ($\text{SMI} > 1$) at low BV/TV, consistent with
trabecular bone morphology literature.

### 10.5 Architectural Distance to Organ Targets

The weighted architectural distance $D_{\text{arch}}$ (Section 7.2) was computed
between each DOE run's final tissue state and the 7 organ targets (Section 7.3).
The closest scaffold-organ matches:

| Rank | Run | Target Organ | $D_{\text{arch}}$ | Key Parameters |
|------|-----|--------------|-------------------|----------------|
| 1 | DOE_22 | Trabecular bone | 15.1 | $E=50$, $\phi_f=0.4$, high adhesion |
| 2 | DOE_20 | Trabecular bone | 16.8 | $E=50$, $\phi_f=0.4$, low adhesion |
| 3 | DOE_18 | Trabecular bone | 16.8 | $E=50$, $\phi_f=0.4$, $R_f=60$ |
| 4 | DOE_23 | Trabecular bone | 17.3 | $E=2$, $\phi_f=0.4$, center point |

**Interpretation.** High functional volume fraction ($\phi_f = 0.4$) is the strongest
predictor of proximity to trabecular bone. Substrate stiffness ($E = 50$ kPa) provides
a secondary benefit by enabling cell-mediated compaction that increases BV/TV and
reduces Tb.Sp. The remaining distance ($D_{\text{arch}} \sim 15$) is driven primarily
by the mismatch in Tb.Th ($\sim\!20$--$40\ \mu$m in simulation vs. $\sim\!150\ \mu$m
in adult human trabecular bone) and tortuosity.

### 10.6 Summary of Key Findings

1. **Substrate stiffness is the master variable** for cell-driven compaction, acting
   through the motor-clutch engagement ratio $\beta$. Below a critical stiffness
   ($E \lesssim 5$ kPa, $\beta < 0.3$), cell traction is insufficient to drive
   measurable compaction.

2. **Composition ratio dominates tissue architecture.** The functional phase fraction
   $\phi_f$ directly controls BV/TV, Tb.Th, and SMI. The inert phase acts as a void
   template.

3. **The mean-field ODE captures compaction dynamics** in the high-$\beta$ regime
   ($R^2 \sim 0.1$--$0.6$) but underperforms at low $\beta$ where stochastic effects
   and bridge kinetics dominate.

4. **Scaffold architectures span a wide range** of tissue descriptors
   ($\text{BV/TV} \in [0.14,\, 0.58]$, $\text{SMI} \in [-2.7,\, 1.8]$), suggesting
   the GELLS system can be tuned toward multiple organ targets.

5. **Trabecular bone is the most accessible target** with $D_{\text{arch}} = 15.1$
   at optimal parameters. Liver and kidney cortex are secondary targets requiring
   higher BV/TV and lower porosity than achieved in this DOE.

---

## 11. Volume-Conserving Two-Zone Mean-Field Model

### 11.1 Physical Motivation

The mean-field model of Section 5, which tracks the bulk functional volume fraction
$\phi_f(t)$, contains a fundamental inconsistency: it implicitly assumes that granular
material is created or destroyed during compaction. Hydrogel granules are
incompressible on the timescale of cell-driven rearrangement ($\sim$10--100 h). The
total solid volume fraction $\phi_s = \phi_f + \phi_i$ must therefore remain constant
throughout the compaction process.

We resolve this by introducing a **two-zone model** in which the scaffold domain is
partitioned into a functional zone (volume fraction $x_f$ of the total domain) and an
inert zone (volume fraction $x_i = 1 - x_f$). Cell-driven compaction shrinks the
functional zone, concentrating functional granules into a smaller volume while
expanding the inert zone to accommodate displaced material. The global solid fraction
$\phi_s$ is conserved exactly; only its spatial distribution changes.

This reformulation is analogous to the poroelastic consolidation of Biot (1941), where
fluid is squeezed from a compressing region into a dilating region, and to the osmotic
compression of concentrated emulsions described by Princen (1986).

### 11.2 State Variable and Conservation Law

Let $x_f(t)$ denote the volume fraction of the domain occupied by the functional zone
at time $t$. The initial value is

$$x_{f,0} = \max\!\left(\frac{\phi_f}{\phi_s}, \; \frac{\phi_f}{\phi_{\max}}\right)$$

where $\phi_{\max}$ is the maximum deformable packing fraction (Section 11.4). The
inert zone occupies $x_i = 1 - x_f$.

**Conservation.** The total solid area (2D) or volume (3D) is

$$\phi_f \cdot A_{\text{domain}} + \phi_i \cdot A_{\text{domain}} = \phi_s \cdot A_{\text{domain}} = \text{const}$$

The **local** packing fractions within each zone are:

$$\phi_{f}^{\text{local}}(t) = \frac{\phi_f}{x_f(t)}, \qquad \phi_{i}^{\text{local}}(t) = \frac{\phi_i}{1 - x_f(t)}$$

As cells compact the functional zone ($x_f$ decreases), $\phi_f^{\text{local}}$
increases toward random close packing while $\phi_i^{\text{local}}$ decreases (inert
granules spread into a larger volume). The local void fractions are:

$$\phi_{v,f}^{\text{local}} = 1 - \phi_f^{\text{local}}, \qquad \phi_{v,i}^{\text{local}} = 1 - \phi_i^{\text{local}}$$

The global void fraction $\phi_v = 1 - \phi_s$ is exactly conserved; only its spatial
distribution between the two zones changes.

### 11.3 Compaction ODE

The functional zone fraction evolves according to the overdamped force balance:

$$\frac{dx_f}{dt} = -x_f \cdot \frac{\max\!\left(0, \; \sigma_{\text{cell}}(t) - \sigma_{\text{resist}}(x_f)\right)}{\eta_{\text{eff}}}$$

where $\sigma_{\text{cell}}$ is the volume-averaged cell traction stress (Section 5.2)
and $\sigma_{\text{resist}}$ is the elastic resistance from contact deformation.
The $x_f$ prefactor ensures that compaction rate scales with the size of the zone
being compressed, vanishing as $x_f \to 0$.

The resistance stress is

$$\sigma_{\text{resist}} = \sigma_0 \cdot \max\!\left(0, \; \frac{\phi_f^{\text{local}}}{\phi_{\text{RCP}}} - 1\right)$$

where $\sigma_0 = 0.005 \, E_{\text{eff}}$ is the elastic resistance scale derived from
Hertzian contact mechanics, and $\phi_{\text{RCP}}$ is the random close packing fraction
for the bidisperse superellipsoidal granule mixture (Section 11.5). Resistance vanishes
when $\phi_f^{\text{local}} < \phi_{\text{RCP}}$ and grows linearly with excess packing
beyond RCP.

### 11.4 Deformable Packing Limit

Rigid particles cannot exceed $\phi_{\text{RCP}}$, but soft hydrogel granules deform at
contacts and can pack beyond this limit. Following the approach of Cai et al. (2022),
who showed that Hertzian contact deformations predict the bulk modulus of granular
hydrogels, we define a deformable packing limit:

$$\phi_{\max} = \phi_{\text{RCP}} + (1 - \phi_{\text{RCP}}) \cdot \left[1 - \exp\!\left(-\frac{0.5}{E_f^{0.3}}\right)\right] \cdot (1 + 0.1(n_2 - 2))$$

Softer granules (lower $E_f$) deform more at contacts, permitting higher packing. Blockier
shapes ($n_2 > 2$) interlock more efficiently, further increasing $\phi_{\max}$. The
compaction minimum is $x_{f,\min} = \phi_f / \phi_{\max}$.

### 11.5 Random Close Packing of Bidisperse Shaped Granules

The rigid-particle RCP fraction $\phi_{\text{RCP}}$ depends on granule shape and size
distribution. We adopt the following empirical model, informed by Farr and Groot (2009)
for polydisperse spheres, Donev et al. (2004) for ellipsoids, and Yuan et al. (2019)
for superellipsoids:

$$\phi_{\text{RCP}} = \phi_{\text{RCP}}^{\text{mono}} + \Delta\phi_{\text{AR}} + \Delta\phi_n + \Delta\phi_{\text{bi}}$$

where:

- $\phi_{\text{RCP}}^{\text{mono}} = 0.82$ (2D) or $0.64$ (3D) for monodisperse circles/spheres
- $\Delta\phi_{\text{AR}} = 0.04(\text{AR} - 1) - 0.015(\text{AR} - 1)^2$ accounts for aspect
  ratio effects, peaking near $\text{AR} \approx 1.3$ (Donev et al. 2004)
- $\Delta\phi_n = 0.015(n_2 - 2)$ accounts for blockiness (squarer shapes pack denser; Delaney
  and Cleary, 2010)
- $\Delta\phi_{\text{bi}} = 0.18 \, x_s(1 - x_s)(1 - 1/r)$ is the Farr--Groot bidisperse
  correction, where $x_s$ is the volume fraction of small particles and $r$ is the size ratio

The effective contact modulus is the composition-weighted harmonic mean:

$$E_{\text{eff}} = \frac{w_{ff} E_f + w_{fi}\bar{E}_{fi} + w_{ii} E_i}{w_{ff} + w_{fi} + w_{ii}}$$

where $w_{ff} = \phi_f^2$, $w_{ii} = \phi_i^2$, $w_{fi} = 2\phi_f\phi_i$, and
$\bar{E}_{fi} = 2E_f E_i / (E_f + E_i)$ is the Hertzian reduced modulus for a
functional--inert contact.

---

## 12. Free Energy Landscape for Scaffold Compaction

### 12.1 Motivation: From Kinetics to Thermodynamics

The volume-conserving ODE (Section 11.3) describes the kinetics of compaction but does
not reveal the underlying energetic driving forces and barriers. To develop rational
design rules for tissue-specific scaffold architectures, we require a thermodynamic
framework that decomposes the total free energy into physically distinct contributions.
This approach, inspired by the energy landscape formalism in soft matter physics
(Wales, 2003) and its applications to cell rearrangements in dense tissues (Bi et al.,
2014, 2015), enables identification of the controlling mechanisms for each organ target.

We define a free energy density $G(\xi)$ as a function of a single compaction coordinate
$\xi$, such that the overdamped dynamics of the system correspond to gradient descent on
this landscape:

$$\frac{d\xi}{dt} = -\frac{1}{\eta_{\text{eff}}} \frac{dG}{d\xi}$$

This formulation, standard for overdamped systems in colloidal and granular physics
(Kramers, 1940; Doi and Edwards, 1986; Tighe et al., 2010), connects the equilibrium
scaffold architecture ($\xi^*$ at the energy minimum) to the balance of six competing
physical mechanisms.

### 12.2 Compaction Coordinate

We parameterize the degree of compaction by the dimensionless coordinate

$$\xi = 1 - \frac{x_f}{x_{f,0}} \in [0, \, \xi_{\max}]$$

where $\xi = 0$ is the initial (uncompacted) state and $\xi_{\max} = 1 - x_{f,\min}/x_{f,0}$
is the maximum compaction permitted by the deformable packing limit. The functional zone
fraction at compaction $\xi$ is

$$x_f(\xi) = x_{f,0}(1 - \xi)$$

and the local packing fractions (Section 11.2) become explicit functions of $\xi$:

$$\phi_f^{\text{local}}(\xi) = \frac{\phi_f}{x_{f,0}(1 - \xi)}, \qquad \phi_i^{\text{local}}(\xi) = \frac{\phi_i}{1 - x_{f,0}(1 - \xi)}$$

### 12.3 Decomposition of the Free Energy

The total free energy density is decomposed into six terms, each corresponding to a
distinct physical mechanism:

$$G(\xi) = G_{\text{cell}}(\xi) + G_{\text{elastic}}(\xi) + G_{\text{yield}}(\xi) + G_{\text{void}}(\xi) + G_{\text{inert}}(\xi) + G_{\gamma}(\xi)$$

The first term is the driving force (negative, favoring compaction); the remaining five
are resistance terms (positive, opposing compaction) that arise from distinct physical
origins. We derive each term below.

### 12.4 Cell Traction Energy ($G_{\text{cell}}$)

Cells encapsulated within functional granules generate contractile traction through the
motor-clutch mechanism (Chan and Odde, 2008; Section 2.5) and transmit this force to
neighboring granules through intercellular bridges (Section 2.6). The volume-averaged
cell traction stress is

$$\sigma_{\text{cell}} = \rho_{\text{cell}} \cdot F_{\text{MC}}(E_f) \cdot f_{\text{bridge}} \cdot m$$

where $\rho_{\text{cell}}$ is the cell number density (cells/$\mu$m$^2$),
$F_{\text{MC}}(E_f)$ is the motor-clutch traction force at substrate stiffness $E_f$
(Section 2.5), $f_{\text{bridge}}$ is the fraction of cells that have formed bridges
(Poisson kinetics, Section 2.6), and $m$ is the focal adhesion maturity.

As the functional zone compacts, the local cell density increases as $1/(1 - \xi)$,
enhancing the bridging probability and effective traction. The cell traction energy is
obtained by integrating the density-enhanced driving stress:

$$G_{\text{cell}}(\xi) = \sigma_{\text{cell}} \cdot \ln(1 - \xi)$$

This expression is strictly negative for $\xi > 0$, reflecting the thermodynamic driving
force for compaction. The logarithmic divergence as $\xi \to 1$ captures the physical
reality that cell density (and hence contractile stress) grows without bound as the
functional zone shrinks to zero volume --- though in practice, compaction is arrested by
the resistance terms long before this limit.

**Decision rationale.** The logarithmic form arises naturally from the density
enhancement $\rho(\xi) = \rho_0/(1 - \xi)$ and the assumption that per-cell traction
force is independent of local density (valid in the sub-saturated regime where cells do
not compete for adhesion sites). This is a conservative assumption; cooperative effects
between nearby cells could further enhance traction at high density (Di Caprio et al.,
2025).

### 12.5 Hertzian Elastic Energy ($G_{\text{elastic}}$)

When compaction drives the local functional packing fraction above the random close
packing threshold $\phi_{\text{RCP}}$, granules are forced into elastic contact and
store Hertzian elastic energy. This is the primary resistance mechanism identified by
the mean-field ODE (Section 11.3).

We derive the elastic energy by integrating the ODE-consistent resistance stress
$\sigma_{\text{resist}} = \sigma_0 \max(0, \phi_f^{\text{local}}/\phi_{\text{RCP}} - 1)$
over the compaction coordinate. Defining the elastic onset

$$\xi_c = 1 - \frac{\phi_f}{x_{f,0} \phi_{\text{RCP}}}$$

as the compaction at which $\phi_f^{\text{local}}$ first reaches $\phi_{\text{RCP}}$,
the integrated elastic energy is (to leading order near $\xi_c$):

$$G_{\text{elastic}}(\xi) = \frac{\sigma_0}{2} \max(0, \, \xi - \xi_c)^2$$

where $\sigma_0 = 0.005 \, E_{\text{eff}}$. This parabolic form is exact for the
linearized mean-field model near the elastic onset and ensures that the energy landscape
equilibrium is consistent with the ODE's force balance. The quadratic growth above $\xi_c$
captures the progressive stiffening of the granular assembly as contacts deform under
Hertzian mechanics (Johnson, 1985). Cai et al. (2022) demonstrated experimentally that
Hertzian contact deformations accurately predict the bulk mechanical response of granular
hydrogel assemblies, supporting this formulation.

**Decision rationale.** We use the ODE-derived parabolic form rather than the full
Hertz energy ($\propto \delta^{5/2}$) to maintain analytical consistency between the
energy landscape and the mean-field kinetics. The two formulations agree near the onset
$\xi_c$ and diverge only at very high compaction where additional physics (granule
plasticity, network percolation) would in any case require model extensions.

### 12.6 Herschel-Bulkley Yield Barrier ($G_{\text{yield}}$)

Dense granular materials exhibit a yield stress below which no macroscopic flow occurs
(Liu and Nagel, 1998; O'Hern et al., 2003). For packing fractions above the jamming
point $\phi_J$, the yield stress scales as a power law in the distance from jamming
(Olsson and Teitel, 2007, 2012; van Hecke, 2010):

$$\sigma_y = c_y \, \sigma_0 \cdot \max\!\left(0, \; \frac{\phi_f^{\text{local}}}{\phi_J} - 1\right)^{\!\Delta}$$

where $\phi_J = 0.92 \, \phi_{\text{RCP}}$ is the jamming fraction (slightly below RCP,
consistent with the observation that jamming onset precedes close packing for frictional
particles; van Hecke, 2010), $c_y = 0.25$ is a dimensionless yield coefficient, and
$\Delta = 3/2$ is the O'Hern scaling exponent for frictionless soft spheres (O'Hern
et al., 2003). The connection to the Herschel-Bulkley constitutive law
$\sigma = \sigma_y + K\dot{\gamma}^n$ enters through the yield stress term, which
sets the threshold for initiating granular rearrangement.

The yield stress generates an energy barrier in the compaction landscape:

$$G_{\text{yield}}(\xi) = \sigma_y(\xi) \cdot \max(0, \xi - \xi_J)$$

where $\xi_J$ is the compaction at which $\phi_f^{\text{local}}$ first reaches $\phi_J$.
Since $\phi_J < \phi_{\text{RCP}}$, the yield barrier activates before the elastic
resistance ($\xi_J < \xi_c$), creating an activation threshold that cells must overcome
before macroscopic rearrangement can proceed. This is consistent with the observation
that granular scaffolds maintain their shape under gravity but can be slowly restructured
by sustained biological forces (Riley et al., 2019).

**Decision rationale.** The Herschel-Bulkley framework is the standard constitutive model
for yield-stress fluids, encompassing dense granular suspensions, foams, and emulsions
(Olsson and Teitel, 2012). The $\Delta = 3/2$ exponent is well-established for soft
frictionless particles (O'Hern et al., 2003) and provides a physically grounded
scaling near jamming without introducing free parameters. The Durian bubble model
(Durian, 1995) and subsequent theoretical work (Tighe et al., 2010) confirm that
overdamped soft-particle systems exhibit this scaling.

### 12.7 Void Redistribution Energy ($G_{\text{void}}$)

Compaction expels interstitial fluid from the functional zone into the inert zone,
altering the local void fractions from their initial equilibrium values. This
redistribution incurs an osmotic free energy cost analogous to the osmotic pressure
in concentrated emulsions (Princen, 1986) and the consolidation resistance in
poroelastic media (Biot, 1941).

The osmotic energy is modeled as a quadratic penalty for deviation from the initial
void fractions, weighted by the zone volumes:

$$G_{\text{void}}(\xi) = \Pi \left[\left(\phi_{v,f}^{\text{local}} - \phi_{v,f}^{0}\right)^2 x_f(\xi) + \left(\phi_{v,i}^{\text{local}} - \phi_{v,i}^{0}\right)^2 (1 - x_f(\xi))\right]$$

where $\Pi = 3 \, \sigma_{\text{cell}}$ is the osmotic compressibility modulus (scaled
to the cell traction stress, consistent with the observation that biological forces, not
thermal fluctuations, drive void redistribution at the granular scale), and the
superscript 0 denotes initial values.

The quadratic form is the leading-order expansion of the free energy of mixing for a
compressible pore fluid in a granular skeleton. At the granular scale, the "osmotic
pressure" arises not from thermal fluctuations but from the elastic and geometric
constraints of the pore network, as analyzed by Brinkman (1949) for viscous flow through
dense particle swarms.

**Decision rationale.** The void redistribution cost is essential for capturing the
observation that compaction stalls before the elastic hard wall is reached: even at
packing fractions below $\phi_{\text{RCP}}$, the cost of expelling fluid from the
densifying functional zone creates a soft resistance that opposes further compaction.
Scaling $\Pi$ to $\sigma_{\text{cell}}$ (rather than $E_{\text{eff}}$) ensures that
this term acts as a perturbative correction to the cell-elastic balance, consistent with
the mean-field ODE predictions.

### 12.8 Inert Granule Frustration Energy ($G_{\text{inert}}$)

Inert granules act as geometric obstacles to the compaction of the functional zone. As
the functional zone contracts, inert granules at the zone boundary must be displaced,
creating an energetic cost that grows with the amount of compaction and the relative
abundance of inert material. This frustration mechanism is analogous to the geometric
frustration that prevents crystallization in binary hard-sphere mixtures and to the
jamming-induced rigidity in tissue cell layers with mixed cell types (Bi et al., 2015).

The frustration energy is modeled as a quadratic function of compaction, weighted by the
inert-to-functional volume ratio:

$$G_{\text{inert}}(\xi) = k_f \cdot \frac{\phi_i}{\phi_f} \cdot \xi^2$$

where $k_f = 1.5 \, \sigma_{\text{cell}}$ is the frustration coefficient (scaled to the
cell traction stress). The quadratic growth reflects the increasing difficulty of
displacing inert granules as the functional zone contracts: early compaction pushes aside
peripheral inert granules easily, while later compaction requires rearranging granules that
are more deeply embedded in the scaffold.

**Decision rationale.** The $\phi_i/\phi_f$ prefactor captures the intuition that
scaffolds with more inert content relative to functional content are harder to compact,
consistent with the DEM simulation results (Section 10) showing that high
$\phi_i/\phi_f$ runs exhibit less compaction at equal cell traction. The quadratic
scaling is the simplest monotonically increasing resistance consistent with dimensional
analysis.

### 12.9 Interfacial Tension Energy ($G_{\gamma}$)

The boundary between the functional and inert zones constitutes a material interface
with an effective surface tension $\gamma_{fi}$ arising from the modulus mismatch and
differential adhesion between the two granule types. This interfacial energy plays a
role analogous to the surface tension in foam rheology (Princen, 1979, 1983) and the
cortical tension at cell-cell boundaries in tissue mechanics (Bi et al., 2015).

The interfacial tension is estimated as

$$\gamma_{fi} = c_\gamma \, \sigma_{\text{cell}} \left(0.3 + \frac{|E_f - E_i|}{E_f + E_i}\right)$$

where $c_\gamma = 2.5$ is a dimensionless coefficient and the modulus mismatch factor
reflects the observation that interfaces between mechanically dissimilar materials have
higher effective surface tension (greater disruption of the contact force network at
the boundary).

The interfacial energy contains two competing contributions:

$$G_{\gamma}(\xi) = \gamma_{fi} \left[\sqrt{1 - \xi} - 1\right] + \frac{\gamma_{fi} \phi_i}{2} \xi$$

The first term represents the energy change due to the geometric contraction of the
zone boundary. In 2D, a circular zone of area fraction $x_f$ has perimeter
$P \propto \sqrt{x_f} = \sqrt{x_{f,0}(1 - \xi)}$, so the perimeter ratio is
$P/P_0 = \sqrt{1 - \xi}$. Since $\sqrt{1 - \xi} < 1$ for $\xi > 0$, this term is
**negative** and favors compaction (the system lowers its interfacial energy by reducing
the boundary area). This is the same driving force responsible for Ostwald ripening in
emulsions and coarsening in foams.

The second term is a **mixing penalty** that accounts for inert granules trapped at or
within the zone boundary. As compaction proceeds, the convoluted interphase boundary
contains more inert granules, increasing the total interfacial area and opposing further
compaction.

**Decision rationale.** Interfacial tension is a well-established driving force in the
mechanics of concentrated emulsions and foams (Princen, 1979, 1983). In granular
scaffolds, the "interface" between functional and inert zones is not a sharp material
boundary but an emergent mesoscale feature of the biphasic packing. We assign it an
effective surface tension using dimensional arguments and the modulus mismatch as a proxy
for the disruption of the local force network at the zone boundary. The functional form
($\sqrt{1-\xi}$ for area change, linear mixing penalty) is consistent with the scaling
of interfacial area with volume fraction in 2D biphasic systems.

### 12.10 Total Free Energy and Equilibrium

The total free energy density is

$$G(\xi) = \sigma_{\text{cell}} \ln(1 - \xi) + \frac{\sigma_0}{2}(\xi - \xi_c)_+^2 + \sigma_y(\xi)(\xi - \xi_J)_+ + \Pi \sum_\alpha (\Delta\phi_{v,\alpha})^2 x_\alpha + k_f \frac{\phi_i}{\phi_f} \xi^2 + G_\gamma(\xi)$$

where $(\cdot)_+ = \max(0, \cdot)$.

**Equilibrium.** The equilibrium compaction $\xi^*$ is the global minimum of $G(\xi)$
on $[0, \xi_{\max}]$, satisfying

$$\left.\frac{dG}{d\xi}\right|_{\xi^*} = 0, \qquad \left.\frac{d^2G}{d\xi^2}\right|_{\xi^*} > 0$$

The equilibrium condition is equivalent to a generalized force balance among all six
mechanisms: cells pull the system toward higher $\xi$, while elastic contacts, yield
stress, void redistribution, inert frustration, and interfacial mixing push back. The
position of the minimum $\xi^*$ determines the final tissue architecture.

**Barrier.** If the yield and void terms create a local maximum at $\xi_b < \xi^*$, an
activation barrier $\Delta G^* = G(\xi_b) - G(0)$ must be overcome for compaction to
initiate. This barrier is crossed when the cell traction stress exceeds the yield stress:
$\sigma_{\text{cell}} > \sigma_y(\phi_J)$.

### 12.11 Normalised Energy and Landscape Structure

To reveal the landscape structure, we normalize the free energy by the cell traction
scale:

$$\tilde{G}(\xi) = \frac{G(\xi)}{\sigma_{\text{cell}}}$$

In normalized units, the cell driving term $\tilde{G}_{\text{cell}} = \ln(1 - \xi)$ is
$O(1)$, and the resistance terms are scaled by their ratio to $\sigma_{\text{cell}}$.
The key ratio governing the landscape shape is

$$\frac{\sigma_0}{\sigma_{\text{cell}}} = \frac{0.005 \, E_{\text{eff}}}{\rho_{\text{cell}} \cdot F_{\text{MC}}(E_f) \cdot f_b \cdot m}$$

When this ratio is large (stiff scaffold, few cells), the elastic parabola rises steeply
and compaction is limited. When the ratio is small (soft scaffold, many cells), cells
drive compaction nearly to $\xi_{\max}$.

### 12.12 Overdamped Kinetics on the Landscape

The compaction dynamics follow overdamped gradient descent with time-dependent driving:

$$\frac{d\xi}{dt} = -\frac{1}{\eta_{\text{eff}}} \frac{dG}{d\xi}\bigg|_{f_b(t), \, m(t)}$$

where $f_b(t)$ and $m(t)$ evolve according to the bridge kinetics (Section 2.6) and
focal adhesion maturation (Section 2.5), respectively. At early times ($f_b \approx 0$),
the cell driving term vanishes and the landscape is flat; as bridges form ($f_b \to 1$),
the cell traction deepens the energy well and the system rolls toward $\xi^*$.

This time-dependent landscape is analogous to the evolving energy landscape in protein
folding (Wales, 2003) and cell differentiation (Shi et al., 2022), where external
parameters (temperature, signaling molecules) reshape the landscape over time, guiding
the system through a sequence of metastable states toward the final equilibrium.

### 12.13 Governing Dimensionless Groups

Five dimensionless numbers govern the shape of the energy landscape and hence the
equilibrium scaffold architecture:

| Group | Symbol | Definition | Physical meaning |
|-------|--------|-----------|-----------------|
| Cell-to-elastic ratio | $\beta$ | $\sigma_{\text{cell}} / \sigma_0$ | Driving force vs. contact resistance |
| Cell-to-yield ratio (capillary number) | $Ca$ | $\sigma_{\text{cell}} / \sigma_y$ | Can cells overcome the yield barrier? |
| Inert obstruction | $\Phi_r$ | $\phi_i / \phi_f$ | Geometric frustration from inert granules |
| Packing proximity | $\Psi$ | $\phi_f^{\text{local}}(0) / \phi_{\text{RCP}}$ | How close is initial packing to RCP? |
| Interfacial number | $\Gamma$ | $\gamma_{fi} / \sigma_{\text{cell}}$ | Surface tension vs. cell driving |

**Design interpretation.** To achieve a target compaction $\xi^*_{\text{target}}$
(corresponding to the desired organ architecture), the scaffold parameters ($E_f$,
$E_i$, $\phi_f$, $\phi_i$, $R_f$, $n_{\text{cells}}$) must be chosen such that the
energy minimum falls at $\xi^*_{\text{target}}$. The dimensionless groups provide a
reduced parameter space for this optimization:

- Increasing $\beta$ (softer scaffold or more cells) deepens the well and shifts $\xi^*$
  rightward (more compaction).
- Increasing $Ca$ above 1 ensures the yield barrier is overcome.
- Increasing $\Phi_r$ (more inert content) raises the frustration floor and shifts $\xi^*$
  leftward (less compaction).
- Increasing $\Psi$ toward 1 means the initial packing is already near RCP, leaving little
  room for compaction ($\xi_c$ is small).
- Large $\Gamma$ means interfacial effects dominate; the mixing penalty can prevent
  compaction if inert granules are densely distributed at the zone boundary.

---

## 13. Computational Results: Energy Landscape Analysis

### 13.1 Landscape Structure Across Organ Targets

The energy landscape was computed for four representative organ targets using optimal
scaffold parameters identified by the mean-field parameter sweep (Section 10). The
results reveal qualitatively distinct landscape shapes:

| Organ Target | $D_{\text{arch}}$ | $\xi_{\max}$ | $\xi^*$ | $\tilde{G}^*$ | Dominant resistance |
|-------------|-------------------|---------------|---------|---------------|---------------------|
| Trabecular bone | 5.7 | 0.589 | 0.577 | $-0.56$ | Elastic (near $\xi_{\max}$) |
| Intestinal mucosa | 7.7 | 0.382 | 0.375 | $-0.57$ | Elastic + void |
| Kidney cortex | 9.1 | 0.217 | 0.213 | $-0.19$ | Elastic + inert |
| Cardiac muscle | 9.2 | 0.218 | 0.096 | $-0.14$ | Yield + inert + elastic |

**Trabecular bone** ($D = 5.7$, the most accessible target) exhibits a deep well with
$\xi^*$ approaching $\xi_{\max}$, indicating that cell traction can drive nearly
maximal compaction. The landscape is dominated by the cell-elastic balance, with
perturbative contributions from the other terms.

**Cardiac muscle** ($D = 9.2$, the most challenging target) shows a shallow well with
$\xi^* \approx 0.10$, reflecting the strong resistance from a stiff scaffold with
high inert content. The yield barrier is significant, and the inert frustration term
contributes substantially to limiting compaction.

### 13.2 Time Evolution of the Landscape

The energy landscape evolves as cells mature and bridges form. At $t = 0$ (no bridges),
the landscape is flat (no cell driving). As $f_b$ increases from 0 to 1 over
approximately 10--20 hours, the cell traction well deepens progressively, pulling the
equilibrium toward higher $\xi$. The final landscape shape (at $f_b = 1$, $m = 1$)
determines the long-time compaction.

The kinetics on the evolving landscape show a characteristic S-shaped trajectory
$\xi(t)$, with three phases:

1. **Lag phase** (0--5 h): Cells attach and spread; $f_b \approx 0$; no compaction.
2. **Rapid compaction** (5--20 h): Bridges form rapidly; the energy well deepens; the
   system accelerates toward the minimum.
3. **Equilibration** (20--72 h): $\xi(t)$ asymptotes to $\xi^*$ as the driving stress
   equilibrates with the resistance.

### 13.3 Energy Decomposition at Equilibrium

The stacked decomposition of energy terms at $\xi^*$ reveals the relative importance of
each mechanism for each organ target. The cell traction and interfacial tension terms
are consistently negative (driving), while elastic, yield, void, and inert terms are
positive (resisting). The balance varies by organ: bone and mucosa are cell-dominated
(large negative $\tilde{G}_{\text{cell}}$), while kidney and cardiac are
resistance-dominated (large positive $\tilde{G}_{\text{elastic}} + \tilde{G}_{\text{inert}}$).

### 13.4 Design Space Maps

Two-dimensional heat maps of $\xi^*$ over key parameter pairs reveal the design
sensitivities:

- **$E_f$ vs. $\phi_f$:** Compaction increases with $\phi_f$ (more functional material
  to compact) and decreases with $E_f$ (stiffer scaffold resists deformation).
  The gradient is steeper along $\phi_f$, indicating composition is a stronger lever than
  stiffness.

- **$\phi_i$ vs. $R_f$:** Higher inert content suppresses compaction through geometric
  frustration. Larger granule radii slightly reduce compaction (fewer granules per
  domain, lower cell density).

- **$E_f$ vs. $E_i$:** The modulus mismatch affects both the elastic resistance
  ($E_{\text{eff}}$) and the interfacial tension ($\gamma_{fi}$). Along the
  equal-modulus diagonal ($E_f = E_i$), interfacial effects are minimized.

---

## 14. Design Rules from the Energy Landscape

### 14.1 General Principles

The energy landscape framework yields the following design rules for targeting specific
tissue architectures:

1. **To increase compaction** (e.g., trabecular bone): Decrease $E_f$ (softer functional
   granules), increase $n_{\text{cells}}$ (more driving force), decrease $\phi_i/\phi_f$
   (less inert obstruction), or increase cell sensing distance (more bridges).

2. **To decrease compaction** (e.g., cardiac muscle): Increase $E_f$, decrease
   $n_{\text{cells}}$, increase $\phi_i/\phi_f$, or use granule shapes that increase
   $\phi_{\text{RCP}}$ (lowering $\xi_c$).

3. **To sharpen the yield barrier** (controlled compaction onset): Increase
   $E_{\text{eff}}$ (raises $\sigma_y$) or choose compositions where $\phi_f^{\text{local}}$
   is initially close to $\phi_J$ (small $\xi_J$).

4. **To minimize interfacial effects**: Match $E_f \approx E_i$ (reduces $\gamma_{fi}$)
   and avoid trapping inert granules within the functional zone.

### 14.2 Organ-Specific Guidelines

| Organ | Required $\xi^*$ | Key design lever | Constraint |
|-------|------------------|------------------|------------|
| Trabecular bone | $\sim 0.5$--$0.6$ | Low $E_f$, high $\phi_f$ | Must maintain permeability for nutrient transport |
| Intestinal mucosa | $\sim 0.3$--$0.4$ | Moderate $E_f$, balanced $\phi_f/\phi_i$ | Intermediate void structure for absorption |
| Kidney cortex | $\sim 0.2$ | Moderate--high $E_f$ | Tight packing with controlled porosity |
| Cardiac muscle | $\sim 0.1$ | High $E_f$, high $\phi_i$ | Minimal compaction; aligned, anisotropic structure |

These guidelines constitute a first-generation set of design rules connecting scaffold
fabrication parameters to target tissue architectures through the mechanistic energy
landscape framework. Experimental validation against confocal microscopy measurements
of $\phi_f^{\text{local}}(t)$ in cell-laden granular hydrogels is a critical next step.

---

## References

- Bi, D., Lopez, J. H., Schwarz, J. M., and Manning, M. L. (2014). Energy barriers and
  cell migration in densely packed tissues. *Soft Matter*, 10:1885--1890.

- Bi, D., Lopez, J. H., Schwarz, J. M., and Manning, M. L. (2015). A density-independent
  rigidity transition in biological tissues. *Nature Physics*, 11:1074--1079.

- Biot, M. A. (1941). General theory of three-dimensional consolidation. *Journal of
  Applied Physics*, 12(2):155--164.

- Brinkman, H. C. (1949). A calculation of the viscous force exerted by a flowing fluid on
  a dense swarm of particles. *Applied Scientific Research*, A1:27--34.

- Cai, S. et al. (2022). Building block properties govern granular hydrogel mechanics
  through contact deformations. *Science Advances*, 8(50):eadd8570.

- Chan, C. E. and Odde, D. J. (2008). Traction dynamics of filopodia on compliant
  substrates. *Science*, 322(5908):1687--1691.

- Delaney, G. W. and Cleary, P. W. (2010). The packing properties of superellipsoids.
  *Europhysics Letters*, 89:34002.

- Derjaguin, B. V., Muller, V. M., and Toporov, Y. P. (1975). Effect of contact
  deformations on the adhesion of particles. *Journal of Colloid and Interface Science*,
  53(2):314--326.

- Di Caprio, N., Hughes, A. J., and Burdick, J. A. (2025). Programmed shape
  transformations in cell-laden granular composites. *Science Advances*, 11(3):eadq5011.

- Doi, M. and Edwards, S. F. (1986). *The Theory of Polymer Dynamics*. Oxford University
  Press.

- Donev, A., Cisse, I., Sachs, D., Variano, E. A., Stillinger, F. H., Connelly, R.,
  Torquato, S., and Chaikin, P. M. (2004). Improving the density of jammed disordered
  packings using ellipsoids. *Science*, 303(5660):990--993.

- Durian, D. J. (1995). Foam mechanics at the bubble scale. *Physical Review Letters*,
  75(26):4780--4783.

- Farr, R. S. and Groot, R. D. (2009). Close packing density of polydisperse hard
  spheres. *Journal of Chemical Physics*, 131(24):244104.

- Gong, J. P. (2006). Friction and lubrication of hydrogels --- its richness and
  complexity. *Soft Matter*, 2(7):544--552.

- Harrigan, T. P. and Mann, R. W. (1984). Characterization of microstructural
  anisotropy in orthotropic materials using a second rank tensor. *Journal of Materials
  Science*, 19(3):761--767.

- Hildebrand, T. and Ruegsegger, P. (1997). A new method for the model-independent
  assessment of thickness in three-dimensional images. *Journal of Microscopy*,
  185(1):67--75.

- Hollister, S. J. (2005). Porous scaffold design for tissue engineering. *Nature
  Materials*, 4(7):518--524.

- Jaklic, A. and Leonardis, A. (2000). Superquadrics and their geometric properties.
  In *Segmentation and Recovery of Superquadrics*, pages 13--39. Springer.

- Johnson, K. L. (1985). *Contact Mechanics*. Cambridge University Press.

- Kramers, H. A. (1940). Brownian motion in a field of force and the diffusion model of
  chemical reactions. *Physica*, 7(4):284--304.

- Liu, A. J. and Nagel, S. R. (1998). Jamming is not just cool any more. *Nature*,
  396:21--22.

- O'Hern, C. S., Silbert, L. E., Liu, A. J., and Nagel, S. R. (2003). Jamming at
  zero temperature and zero applied stress: The epitome of disorder. *Physical Review E*,
  68(1):011306.

- Olsson, P. and Teitel, S. (2007). Critical scaling of shear viscosity at the jamming
  transition. *Physical Review Letters*, 99:178001.

- Olsson, P. and Teitel, S. (2012). Herschel-Bulkley shearing rheology near the
  athermal jamming transition. *Physical Review Letters*, 109:108001.

- Pitenis, A. A., Uruenya, J. M., Schulze, K. D., Nixon, R. M., Dunn, A. C.,
  Krick, B. A., Sawyer, W. G., and Angelini, T. E. (2014). Polymer fluctuation
  lubrication in hydrogel gemini interfaces. *Soft Matter*, 10(44):8955--8962.

- Princen, H. M. (1979). Highly concentrated emulsions. *Journal of Colloid and
  Interface Science*, 71:55--66.

- Princen, H. M. (1983). Rheology of foams and highly concentrated emulsions I.
  Elastic properties and yield stress of a cylindrical model system. *Journal of
  Colloid and Interface Science*, 91:160--175.

- Princen, H. M. (1986). Osmotic pressure of foams and highly concentrated emulsions I.
  Theoretical considerations. *Langmuir*, 2:519--524.

- Riley, L., Schirmer, L., and Segura, T. (2019). Granular hydrogels: emergent
  properties of jammed hydrogel microparticles and their applications in tissue repair
  and regeneration. *Current Opinion in Biotechnology*, 60:1--8.

- Shi, J., Aihara, K., Li, T., and Chen, L. (2022). Energy landscape decomposition for
  cell differentiation with proliferation effect. *National Science Review*,
  9(8):nwac116.

- Tighe, B. P., Woldhuis, E., and van Hecke, M. (2010). Model for the scaling of
  stresses and fluctuations in flows near jamming. *Physical Review Letters*, 105:088303.

- Torquato, S., Truskett, T. M., and Debenedetti, P. G. (2000). Is random close
  packing of spheres well defined? *Physical Review Letters*, 84(10):2064--2067.

- van Hecke, M. (2010). Jamming of soft particles: geometry, mechanics, scaling and
  isostaticity. *Journal of Physics: Condensed Matter*, 22(3):033101.

- Wales, D. J. (2003). *Energy Landscapes: Applications to Clusters, Biomolecules and
  Glasses*. Cambridge University Press.

- Yuan, Y., VanderWerf, K., Shattuck, M. D., and O'Hern, C. S. (2019). Jammed packings
  of 3D superellipsoids with tunable packing fraction, coordination number, and ordering.
  *Soft Matter*, 15:9751.
