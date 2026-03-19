# A Beginner's Guide to Mean Field Modeling

**From Coffee Cooling to Cell-Driven Scaffold Compaction**

*A teaching document for the GELS project*

---

## Table of Contents

1. [What Is a Mean Field Model?](#1-what-is-a-mean-field-model)
2. [Your First Mean Field Model: Coffee Cooling](#2-your-first-mean-field-model-coffee-cooling)
3. [Adding Nonlinearity: Population Growth](#3-adding-nonlinearity-population-growth)
4. [Two Competing Forces: A Thermostat](#4-two-competing-forces-a-thermostat)
5. [From Particles to Fields: The Porous Medium](#5-from-particles-to-fields-the-porous-medium)
6. [Putting It All Together: The GELS Mean Field Model](#6-putting-it-all-together-the-gels-mean-field-model)
7. [Hands-On: Running the Model on Real DEM Data](#7-hands-on-running-the-model-on-real-dem-data)
8. [Comparing Across Experiments: The DOE Sweep](#8-comparing-across-experiments-the-doe-sweep)
9. [From Scalar ODE to Spatial PDE](#9-from-scalar-ode-to-spatial-pde)
10. [The Energy Landscape: A Thermodynamic View](#10-the-energy-landscape-a-thermodynamic-view)
11. [Summary and Key Takeaways](#11-summary-and-key-takeaways)

---

## 1. What Is a Mean Field Model?

### The Core Idea

Imagine you have a jar filled with 500 rubber balls, and someone is slowly squeezing
the jar from the outside. You want to predict how tightly packed the balls become over
time. You have two choices:

**Option A: Track every ball.** Compute the position, velocity, and contacts of all 500
balls at every instant. This is a **Discrete Element Method (DEM)** simulation — it's
accurate but expensive. Our `new_dem_0.py` does exactly this.

**Option B: Track the average.** Instead of 500 positions, track *one number*: the
average packing fraction. Write a simple equation for how that number changes over time.
This is a **mean field model**.

The term "mean field" means we replace the complex interactions between individual
particles with their *average* (mean) effect. Instead of asking "what force does ball #237
feel from its six neighbours?", we ask "what is the *average* force a typical ball feels
in a region with this packing density?"

### Why Bother?

| Feature | DEM Simulation | Mean Field Model |
|---------|---------------|-----------------|
| Accuracy | High (resolves individual contacts) | Approximate (averages over details) |
| Speed | Minutes to hours | Milliseconds |
| Parameters explored | 1 run at a time | 200,000 in seconds |
| Physical insight | Hard to extract (too much data) | Built into the equations |
| Design optimization | Impractical | Natural fit |

A mean field model is not a replacement for DEM — it's a **companion**. We use DEM to
validate the mean field model, then use the mean field model to explore the vast parameter
space that DEM cannot reach.

### The Recipe

Every mean field model follows the same recipe:

1. **Choose your state variable(s)** — what single number (or few numbers) captures the
   system's state?
2. **Write the rate equation** — how does that number change over time? This is almost
   always an ordinary differential equation (ODE): `dx/dt = f(x, t)`.
3. **Identify the driving forces** — what pushes the system forward? What resists?
4. **Solve the ODE** — analytically if possible, numerically if not.
5. **Validate against data** — compare to experiments or detailed simulations.

Let's build this intuition step by step with increasingly complex examples.

---

## 2. Your First Mean Field Model: Coffee Cooling

### The Physical Setup

You pour a cup of coffee at 90°C and set it on your desk. The room is 22°C. How does
the coffee temperature change over time?

You *could* simulate every water molecule, every convection current, every bit of heat
radiation. Or you could write one equation.

### Step 1: Choose the State Variable

**T(t)** = temperature of the coffee at time t (°C).

We are "averaging" over all the complex fluid dynamics inside the cup and replacing it
with a single number.

### Step 2: Write the Rate Equation

Newton's law of cooling says: the rate of temperature change is proportional to the
temperature difference between the coffee and the room.

```
dT/dt = -k × (T - T_room)
```

where:
- `k` is a cooling constant (depends on cup material, surface area, etc.)
- `T_room` = 22°C (the environment)
- The negative sign means the coffee *loses* heat when it's hotter than the room

This is our first mean field model! We replaced millions of molecular collisions with
one number `k` that captures their average effect.

### Step 3: Solve

This ODE has an exact (analytical) solution:

```
T(t) = T_room + (T_0 - T_room) × exp(-k × t)
```

### Step 4: Code It Up

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T_0 = 90.0       # initial temperature (°C)
T_room = 22.0    # room temperature (°C)
k = 0.1          # cooling rate (1/min)

# Time array
t = np.linspace(0, 60, 200)  # 0 to 60 minutes

# Analytical solution
T = T_room + (T_0 - T_room) * np.exp(-k * t)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(t, T, 'C1-', lw=2, label='Mean field: $T(t) = T_{room} + (T_0 - T_{room})e^{-kt}$')
plt.axhline(T_room, color='grey', ls='--', label=f'Room temperature ({T_room}°C)')
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Example 1: Coffee Cooling')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

> **Figure reference:** See `figures/example_01_coffee_cooling.png`

### What Did We Learn?

- **One equation** replaced a complex system of fluid dynamics.
- The single parameter `k` encodes all the microscopic physics (convection, conduction,
  radiation) into one effective number.
- The model predicts exponential decay toward equilibrium — a universal behaviour whenever
  a driving force is proportional to how far you are from equilibrium.

> **Key concept:** A mean field model trades microscopic detail for macroscopic
> predictability. The parameter `k` is an *effective* parameter — it doesn't correspond
> to any single physical process, but captures their combined effect.

---

## 3. Adding Nonlinearity: Population Growth

### The Physical Setup

A colony of bacteria doubles every hour in a petri dish. But the dish can only hold
1,000,000 bacteria. What happens?

### Step 1: State Variable

**N(t)** = number of bacteria at time t.

### Step 2: Rate Equation (First Attempt — Exponential Growth)

If bacteria just double freely:

```
dN/dt = r × N
```

where `r` = growth rate. This gives N(t) = N₀ × exp(r×t) — exponential growth forever.
But that's unrealistic. The dish has a carrying capacity.

### Step 2 (Revised): The Logistic Equation

Add a term that slows growth as N approaches the carrying capacity K:

```
dN/dt = r × N × (1 - N/K)
```

This is the **logistic equation**. When N is small, (1 - N/K) ≈ 1, so growth is nearly
exponential. When N → K, the growth rate → 0. The system saturates.

This is our first **nonlinear** mean field model — the rate depends on the state itself.

### Step 3: Code and Compare

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
N_0 = 100        # initial bacteria count
r = 0.5          # growth rate (1/hour)
K = 1_000_000    # carrying capacity

# Solve the logistic ODE numerically
def logistic_rhs(t, y):
    N = y[0]
    return [r * N * (1.0 - N / K)]

t_span = (0, 40)
t_eval = np.linspace(0, 40, 300)
sol = solve_ivp(logistic_rhs, t_span, [N_0], t_eval=t_eval, method='RK45')

# Also compute uncapped exponential for comparison
N_exp = N_0 * np.exp(r * t_eval)

# Plot both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
ax1.plot(t_eval, sol.y[0], 'C0-', lw=2, label='Logistic (mean field)')
ax1.plot(t_eval, np.clip(N_exp, 0, 5e6), 'C3--', lw=1.5, label='Uncapped exponential')
ax1.axhline(K, color='grey', ls=':', label=f'Carrying capacity K = {K:,.0f}')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Population N')
ax1.set_title('Population Growth (linear scale)')
ax1.legend()
ax1.set_ylim(0, 2e6)
ax1.grid(True, alpha=0.3)

# Log scale
ax2.semilogy(t_eval, sol.y[0], 'C0-', lw=2, label='Logistic')
ax2.semilogy(t_eval, N_exp, 'C3--', lw=1.5, label='Exponential')
ax2.axhline(K, color='grey', ls=':')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Population N (log scale)')
ax2.set_title('Population Growth (log scale)')
ax2.legend()
ax2.set_ylim(1, 1e8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

> **Figure reference:** See `figures/example_02_logistic_growth.png`

### What Did We Learn?

- **Nonlinearity creates new behaviour.** The logistic model has an S-shaped (sigmoidal)
  curve — fast early growth that slows and saturates. Neither pure exponential nor pure
  linear models capture this.
- **Two parameters** (`r` and `K`) encode all the biology. `r` sets the timescale, `K`
  sets the endpoint.
- **The (1 - N/K) factor is a resistance term.** As the system fills up, it pushes back
  against further growth. This is *exactly* the same idea we'll use for granular jamming.

> **Key concept:** Mean field models become powerful when they include competing effects —
> a driving force and a resistance. The interplay between them creates rich dynamics from
> simple equations.

---

## 4. Two Competing Forces: A Thermostat

### Why This Example Matters

The GELS mean field model is fundamentally about two competing stresses: cells
*pulling* the scaffold together, and granule contacts *resisting* further compression.
Before we get there, let's build intuition with a simpler two-force system.

### The Physical Setup

Imagine a room with both a heater and an air conditioner. The heater turns on gradually
(like cells maturing and forming bridges), and the AC resists temperature increases above
a setpoint (like granular jamming resistance).

### State Variable

**T(t)** = room temperature.

### Rate Equation

```
dT/dt = [Q_heater(t) - Q_cooling(T)] / C
```

where:
- `Q_heater(t)` = heater output, ramps up over time (like cell maturation)
- `Q_cooling(T)` = cooling that kicks in above a threshold (like jamming resistance)
- `C` = thermal mass (like granular viscosity)

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters
C = 10.0          # thermal mass (like viscosity)
T_set = 25.0      # setpoint (like RCP jamming threshold)
T_0 = 18.0        # initial temperature

# Heater: ramps up like cell maturation
def Q_heater(t):
    Q_max = 8.0
    tau = 3.0
    return Q_max * (1.0 - np.exp(-t / tau))

# Cooling: kicks in above threshold (like jamming resistance)
def Q_cooling(T):
    if T <= T_set:
        return 0.0
    sigma_0 = 2.0
    alpha = 1.5
    return sigma_0 * ((T - T_set) / T_set) ** alpha

# ODE
def rhs(t, y):
    T = y[0]
    net = Q_heater(t) - Q_cooling(T)
    if net < 0:
        net = 0.0
    return [net / C]

# Solve
t_span = (0, 30)
t_eval = np.linspace(0, 30, 500)
sol = solve_ivp(rhs, t_span, [T_0], t_eval=t_eval, method='RK45')

T = sol.y[0]
q_heat = np.array([Q_heater(t) for t in sol.t])
q_cool = np.array([Q_cooling(T[i]) for i in range(len(T))])

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(sol.t, T, 'C1-', lw=2, label='Temperature T(t)')
ax1.axhline(T_set, color='grey', ls='--', label=f'Threshold = {T_set}°C')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Temperature (°C)')
ax1.set_title('Two Competing Forces: Heater vs Cooling')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(sol.t, q_heat, 'C1-', lw=2, label='$Q_{heater}$ (driving)')
ax2.plot(sol.t, q_cool, 'C0-', lw=2, label='$Q_{cooling}$ (resistance)')
ax2.fill_between(sol.t, q_cool, q_heat,
                 where=q_heat > q_cool, color='C1', alpha=0.15, label='Net driving')
ax2.fill_between(sol.t, q_heat, q_cool,
                 where=q_cool >= q_heat, color='C0', alpha=0.15, label='Arrested')
ax2.set_xlabel('Time (hours)')
ax2.set_ylabel('Heat flux')
ax2.set_title('Stress Balance: Driving vs Resistance')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

> **Figure reference:** See `figures/example_03_competing_forces.png`

### The Three Regimes

Look at the stress balance plot (right panel). There are three distinct regimes:

1. **Early (0–5 h):** The heater is ramping up but the room is still below the threshold.
   No resistance. Temperature rises freely. *In our scaffold: cells are maturing, forming
   focal adhesions, but haven't started bridging yet.*

2. **Middle (5–15 h):** Heater is at full power, resistance is growing. Temperature still
   rises but decelerates. *In our scaffold: bridges are forming and pulling granules
   together, but contacts are stiffening as the packing approaches jamming.*

3. **Late (15+ h):** Driving equals resistance. The system reaches a **dynamic
   equilibrium** — not because forces disappeared, but because they exactly balance. *In
   our scaffold: compaction arrests. The final packing density is set by this balance.*

> **Key concept:** The final state isn't determined by either force alone — it emerges
> from their *balance*. This is the central idea of the GELS mean field model.

---

## 5. From Particles to Fields: The Porous Medium

### Why This Matters

So far our models tracked one number evolving in time. But a real scaffold has spatial
structure — the packing might be denser in the center than at the edges. And the whole
point of building a scaffold is to deliver nutrients to cells — so we need to know how
easily fluid can flow through it.

### The Kozeny-Carman Equation

One of the most useful results in porous media physics is the Kozeny-Carman equation.
It connects the void fraction (how much empty space there is between granules) to the
permeability (how easily fluid can flow through):

```
K = φ_v³ × d² / [180 × (1 - φ_v)²]
```

where:
- `K` = permeability (µm²) — higher means fluid flows more easily
- `φ_v` = void fraction (0 to 1) — fraction of space that is empty
- `d` = grain diameter (µm)

This is a *mean field* relationship: it replaces the complex pore geometry with a single
effective parameter.

### The Dramatic Nonlinearity

```python
import numpy as np
import matplotlib.pyplot as plt

def kozeny_carman(phi_v, d_grain):
    phi_v = np.clip(phi_v, 0.01, 0.99)
    return phi_v**3 * d_grain**2 / (180.0 * (1.0 - phi_v)**2)

d_grain = 80.0  # µm (typical functional granule diameter)
phi_v = np.linspace(0.05, 0.60, 200)
K = kozeny_carman(phi_v, d_grain)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(phi_v, K, 'C2-', lw=2)
ax1.set_xlabel('Void fraction $\\phi_v$')
ax1.set_ylabel('Permeability $K$ (µm²)')
ax1.set_title(f'Kozeny-Carman: d = {d_grain:.0f} µm')
ax1.grid(True, alpha=0.3)
ax1.axvspan(0.15, 0.25, alpha=0.1, color='C3', label='Typical scaffold range')
ax1.legend()

ax2.semilogy(phi_v, K, 'C2-', lw=2)
ax2.set_xlabel('Void fraction $\\phi_v$')
ax2.set_ylabel('Permeability $K$ (µm²) — log scale')
ax2.set_title('Same data, log scale — cubic sensitivity!')
ax2.grid(True, alpha=0.3)
ax2.axvspan(0.15, 0.25, alpha=0.1, color='C3', label='Typical scaffold range')
ax2.legend()

plt.tight_layout()
plt.show()
```

> **Figure reference:** See `figures/example_04_kozeny_carman.png`

### Why This Matters for Tissue Engineering

Look at the steep curve. A small change in void fraction causes a *huge* change in
permeability. Going from φ_v = 0.30 to φ_v = 0.20 (a 33% reduction in void space)
drops permeability by roughly **75%**.

This is why mean field modeling is critical for scaffold design:
- Cells compact the scaffold (reducing φ_v)
- Compaction reduces permeability (Kozeny-Carman)
- Reduced permeability starves cells of nutrients
- This creates a feedback loop that the mean field model captures

> **Key concept:** The Kozeny-Carman equation converts structural information (packing
> fraction) into functional information (transport). This is how our mean field model
> connects cell-driven compaction to scaffold performance.

---

## 6. Putting It All Together: The GELS Mean Field Model

Now we have all the building blocks. Let's assemble the actual mean field model used in
`analysis/mean_field_model.py`.

### 6.1 The Physical Picture

Our scaffold is a packed bed of hydrogel granules, seeded with living cells:

```
┌─────────────────────────────────────────┐
│  ○ ● ○ ● ○ ● ○   ← Functional (●) and │
│ ● ○ ● ○ ● ○ ●      Inert (○) granules  │
│  ○ ● ○ ● ○ ● ○                         │
│ ● ○ ●═● ○ ● ○    ← Cells form bridges  │
│  ○ ● ○ ● ○ ● ○      between granules    │
│ ● ○ ● ○ ● ○ ●      (═ = bridge)        │
│  ○ ● ○ ● ○ ● ○                         │
└─────────────────────────────────────────┘
```

Cells attach to **functional** granules, mature, form bridges to neighbours, and *pull*.
This compacts the functional region while pushing void space into the inert region.

### 6.2 The State Variable

We track one number: **x_f(t)** — the fraction of the domain occupied by the
"functional zone" (functional granules + their local void space).

```
  ┌────────────────────────────────────┐
  │      x_f          │   1 - x_f     │
  │  (functional zone) │ (inert zone)  │
  │                    │               │
  │  ● granules + void │ ○ granules    │
  │  + cells + bridges │   + void     │
  └────────────────────────────────────┘
```

**Why x_f and not packing fraction?**

Because granules are incompressible. The total solid volume `φ_f + φ_i = φ_solid` is
**constant**. What changes is how that solid is *distributed*. Compaction means
functional granules crowd together (x_f shrinks), while their void gets expelled into
the inert zone.

### 6.3 Volume Conservation

This is the most important constraint. Let's walk through it carefully.

**Before compaction (t = 0):**
- Functional granules occupy fraction φ_f of the domain
- Inert granules occupy fraction φ_i
- Void occupies fraction φ_v = 1 - φ_f - φ_i
- The functional zone fraction: x_f₀ = φ_f / φ_solid (granules uniformly mixed)

**After compaction (t > 0):**
- φ_f and φ_i are **unchanged** (granules don't disappear or appear)
- But functional granules are now packed tighter within a smaller zone x_f < x_f₀
- Local void in functional zone: φ_v,f = 1 - φ_f / x_f (less void, tighter packing)
- Local void in inert zone: φ_v,i = 1 - φ_i / (1 - x_f) (more void, looser packing)
- Total void is still 1 - φ_solid (volume is conserved globally)

Let's verify this with code:

```python
import numpy as np
import matplotlib.pyplot as plt

phi_f = 0.40    # functional solid fraction (constant)
phi_i = 0.25    # inert solid fraction (constant)
phi_solid = phi_f + phi_i  # = 0.65 (always constant)

x_f = np.linspace(phi_f / 0.95, phi_f / phi_solid + 0.15, 200)

# Local void fractions
phi_v_func = 1.0 - phi_f / x_f
phi_v_inert = 1.0 - phi_i / (1.0 - x_f)

# Global void (should be constant!)
phi_v_global = x_f * phi_v_func + (1.0 - x_f) * phi_v_inert

print(f"Global void (should be {1 - phi_solid:.2f} everywhere):")
print(f"  min = {phi_v_global.min():.6f}")
print(f"  max = {phi_v_global.max():.6f}")
# Output: Global void is exactly 0.35 everywhere. Volume is conserved!
```

> **Figure reference:** See `figures/example_05_volume_conservation.png`

### 6.4 The Driving Force: Cell Traction

Cells pull on the scaffold through a molecular mechanism called the **motor-clutch model**
(Chan & Odde 2008). Here's the intuition:

```
    Cell membrane
    ┌───────────────────┐
    │  Motor proteins   │   Motors pull inward (like tiny muscles)
    │  ↓  ↓  ↓  ↓  ↓   │
    │  Clutch springs   │   Clutches grip the substrate (like brakes)
    │  ↓  ↓  ↓  ↓  ↓   │
    └───────────────────┘
    ═════════════════════   Substrate (granule surface)
```

- **Motors** (myosin) pull with a fixed stall force F_stall
- **Clutches** (integrins) connect motors to the substrate with springs of stiffness k_clutch
- The substrate itself has a stiffness k_sub (depends on the hydrogel modulus E)
- The cell "senses" substrate stiffness through the balance of motor and clutch forces

The key insight is that cell traction force depends on substrate stiffness:

```
β = k_sub / (k_sub + k_opt)      ← motor-clutch engagement ratio
F_cell = F_stall × β × engagement ← traction force per cell
```

where:
- `k_sub = π × E × a_cell / (1 - ν²)` — substrate spring constant (from Hertz contact)
- `k_opt = n_clutches × k_clutch` — optimal clutch ensemble stiffness
- `engagement = k_on / (k_on + k_off)` — fraction of clutches that are bound

**On soft substrates** (small E): k_sub << k_opt, so β → 0, and cells barely pull.
**On stiff substrates** (large E): k_sub >> k_opt, so β → 1, and cells pull at full force.

> **Figure reference:** See `figures/example_06_motor_clutch.png`

But cells don't pull at full force from the start. Several biological processes ramp up
over time:

1. **FA maturation**: Focal adhesions (the mechanical anchors) mature over hours.
   `maturity(t) = 1 - exp(-fa_rate × (t - t_spread))`

2. **Bridge formation**: Cells must form bridges between granules before they can compact
   the scaffold. Bridges form via a Poisson process: `f_bridge(t) = 1 - exp(-rate × t × maturity)`

3. **Force ramp**: New bridges ramp their force linearly over `bridge_formation_time`.

The total cell stress is:

```
σ_cell(t) = n_cell_density × F_cell × f_bridge(t) × maturity(t) × ramp(t)
```

This is the **driving force** — it grows over time as cells mature and form bridges.

### 6.5 The Resistance: Granular Jamming

As cells compact the functional zone, the local packing fraction φ_f/x_f increases.
When it approaches the **random close packing (RCP)** fraction φ_RCP (about 0.64 for
spheres in 3D, ~0.82 in 2D, higher for non-spherical granules), the granules jam — they
can't be compressed further without deforming.

The resistance stress grows as:

```
σ_resist(x_f) = σ_0 × (φ_f/x_f / φ_RCP - 1)^α     when φ_f/x_f > φ_RCP
               = 0                                     otherwise
```

where:
- `σ_0` = jamming stress prefactor (how stiff the jammed state is)
- `α` = jamming exponent (α = 1 for Hertzian contacts, can be higher)

This is exactly like the cooling resistance in Example 4: zero below a threshold, then
growing rapidly above it.

> **Figure reference:** See `figures/example_07_jamming_resistance.png`

### 6.6 The ODE: Putting Driving and Resistance Together

Now we combine everything into the central equation of the model:

```
dx_f/dt = -x_f × (σ_cell(t) - σ_resist(x_f)) / η_eff
```

Let's break this down:

| Term | Meaning | Analogy to Example 4 |
|------|---------|---------------------|
| `dx_f/dt` | Rate of compaction | `dT/dt` |
| `x_f` | Current state | `T` |
| `σ_cell(t)` | Cell traction (grows in time) | `Q_heater(t)` |
| `σ_resist(x_f)` | Jamming resistance (grows with compaction) | `Q_cooling(T)` |
| `η_eff` | Effective granular viscosity | `C` (thermal mass) |

The negative sign means compaction *decreases* x_f (the functional zone shrinks).

**The three fitted parameters** are:
1. **η_eff** — effective granular viscosity. Controls *how fast* compaction proceeds.
   High viscosity = slow compaction.
2. **σ_0** — jamming stress prefactor. Controls *how strongly* the jammed state resists.
3. **α** — jamming exponent. Controls *how sharply* resistance turns on at jamming.

Everything else (E_modulus, cell count, bridge kinetics, geometry) is set by the
simulation parameters. Only these three are fitted to match DEM data.

### 6.7 Full Worked Example

Let's build and solve the complete GELS mean field model from scratch:

```python
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# =====================================================================
# STEP 1: Define physical parameters (from a typical simulation)
# =====================================================================

# Hydrogel properties
E_modulus = 10.0          # kPa (Young's modulus)
poisson = 0.45

# Scaffold composition
phi_f = 0.40              # functional solid fraction (CONSTANT)
phi_i = 0.25              # inert solid fraction (CONSTANT)
phi_solid = phi_f + phi_i # = 0.65 (CONSTANT)
phi_RCP = 0.64            # random close packing
phi_max = 0.72            # deformable limit (soft granules can go past RCP)

# Granule geometry
R_func = 40.0             # µm, functional granule radius
N_func = 160              # number of functional granules
n_cells_per = 8           # cells per granule
domain_area = 800**2      # µm² (2D domain)

# Cell biology timescales
t_spread = 3.0            # hours to spread
fa_rate = 0.3             # 1/h, FA maturation rate
bridge_rate = 0.3         # 1/h, bridge formation rate
bridge_form_time = 2.0    # hours to ramp bridge force

# Motor-clutch model
n_motors = 50
F_motor_stall = 0.5       # nN
n_clutches = 75
k_clutch = 5.0            # nN/µm
k_opt = n_clutches * k_clutch
k_on, k_off = 1.0, 0.1
engagement = k_on / (k_on + k_off)
F_stall = n_motors * F_motor_stall
a_cell = 10.0             # µm

# =====================================================================
# STEP 2: Compute derived quantities
# =====================================================================

k_sub = np.pi * E_modulus * a_cell / (1.0 - poisson**2)
beta = k_sub / (k_sub + k_opt)
F_cell = F_stall * beta * engagement

n_total_cells = N_func * n_cells_per
n_cell_density = n_total_cells / domain_area

x_f0 = phi_f / phi_solid       # initial functional zone fraction
x_f_min = phi_f / phi_max      # minimum (deformable limit)

print(f"Cell force: F_cell = {F_cell:.2f} nN")
print(f"Initial x_f: {x_f0:.3f}")
print(f"Minimum x_f: {x_f_min:.3f}")

# =====================================================================
# STEP 3: Define the time-dependent driving functions
# =====================================================================

def maturity(t):
    t_eff = max(0.0, t - t_spread)
    return 1.0 - np.exp(-fa_rate * t_eff)

def bridge_fraction(t):
    mat = maturity(t)
    if mat < 0.1:
        return 0.0
    t_bridge = max(0.0, t - t_spread - 1.0)
    return 1.0 - np.exp(-bridge_rate * t_bridge * mat)

def bridge_ramp(t):
    t_start = t_spread + 1.0
    t_since = max(0.0, t - t_start)
    return min(1.0, t_since / bridge_form_time)

def sigma_cell(t):
    return n_cell_density * F_cell * bridge_fraction(t) * maturity(t) * bridge_ramp(t)

def sigma_resist(x_f, sigma_0, alpha):
    phi_local = phi_f / max(x_f, 1e-6)
    ratio = phi_local / phi_RCP
    if ratio <= 1.0:
        return 0.0
    return sigma_0 * (ratio - 1.0)**alpha

# =====================================================================
# STEP 4: Write and solve the ODE
# =====================================================================

# Fitted parameters (these would come from fitting to DEM data)
eta_eff = 0.5       # effective viscosity (nN·h/µm²)
sigma_0 = 0.001     # jamming prefactor
alpha = 1.2         # jamming exponent

def compaction_ode(t, y):
    x_f = np.clip(y[0], x_f_min, 1.0 - phi_i)
    s_cell = sigma_cell(t)
    s_resist = sigma_resist(x_f, sigma_0, alpha)
    net = s_cell - s_resist
    if net <= 0:
        return [0.0]
    return [-x_f * net / eta_eff]

t_span = (0, 72)
t_eval = np.linspace(0, 72, 500)
sol = solve_ivp(compaction_ode, t_span, [x_f0], t_eval=t_eval,
                method='RK45', rtol=1e-8, atol=1e-10)

# =====================================================================
# STEP 5: Extract results and plot
# =====================================================================

x_f_traj = sol.y[0]
phi_v_func = 1.0 - phi_f / np.maximum(x_f_traj, 1e-6)
phi_v_inert = 1.0 - phi_i / np.maximum(1.0 - x_f_traj, 1e-6)

s_cell_arr = np.array([sigma_cell(t) for t in sol.t])
s_resist_arr = np.array([sigma_resist(x_f_traj[i], sigma_0, alpha)
                          for i in range(len(sol.t))])

# Kozeny-Carman permeability
d_f = 2.0 * R_func
K_func = np.clip(phi_v_func, 0.01, 0.99)**3 * d_f**2 / \
         (180.0 * (1.0 - np.clip(phi_v_func, 0.01, 0.99))**2)

# 4-panel figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# ... (plotting code, see generate_examples.py for full version)

plt.show()
```

> **Figure reference:** See `figures/example_08_full_model.png`

The 4-panel figure shows:
- **(a) Compaction trajectory:** x_f decreases from ~0.62 toward ~0.56 over 72 hours.
  The dashed line marks the jamming limit; the dotted line marks the deformable limit.
- **(b) Void redistribution:** As functional zone void decreases, inert zone void
  increases — volume is conserved.
- **(c) Stress balance:** Cell traction ramps up, resistance grows as packing tightens,
  and they converge at equilibrium. The shaded region shows net compaction stress.
- **(d) Permeability:** Permeability in the functional zone drops by roughly an order of
  magnitude as compaction proceeds.

### 6.8 The Biological Timeline

Let's visualize *why* the driving force has that particular shape:

```python
t = np.linspace(0, 48, 300)
mat = np.array([maturity(ti) for ti in t])
fb = np.array([bridge_fraction(ti) for ti in t])
ramp = np.array([bridge_ramp(ti) for ti in t])
combined = mat * fb * ramp
```

> **Figure reference:** See `figures/example_09_biological_timeline.png`

The combined curve (black) is the product of all three biological ramp functions:
- Hours 0–3: Cells are attaching and spreading. No force yet.
- Hours 3–4: FAs begin maturing, but bridges haven't started.
- Hours 4–8: Bridges form rapidly. Force ramps up steeply.
- Hours 8–48: Most bridges are formed. Force saturates near its maximum.

This S-shaped ramp is what gives the compaction trajectory its characteristic shape.

### 6.9 Parameter Sensitivity

One of the biggest advantages of a mean field model is that you can instantly see how
parameters affect the outcome. Let's sweep each key parameter:

> **Figure reference:** See `figures/example_10_parameter_sensitivity.png`

**Reading the sensitivity plots:**

- **(a) Viscosity** controls the *timescale* — higher viscosity means slower compaction, but
  the same final state. Think of honey (high η) vs water (low η) flowing through a funnel.

- **(b) Jamming prefactor** controls the *final state* — higher σ₀ means the system arrests
  earlier (at higher x_f) because resistance kicks in sooner.

- **(c) Jamming exponent** controls *how sharply* arrest occurs — higher α means a more
  abrupt stop. α = 1 gives a gradual approach; α = 2.5 gives a sudden wall.

- **(d) Substrate stiffness** controls *how hard cells can pull* — stiffer substrates
  (higher E) give cells more traction via the motor-clutch model, leading to more compaction.

---

## 7. Hands-On: Running the Model on Real DEM Data

This is where theory meets practice. We have actual DEM simulation results sitting in
`results/LHC/DOE_2D_*` — 19 completed runs with different parameter combinations from
our Design of Experiments (DOE). Let's use the mean field model to analyze them.

### 7.1 What's in a DEM Run?

Each DOE trial directory (e.g., `results/LHC/DOE_2D_0006/`) contains:

```
DOE_2D_0006/
├── params.json          # All simulation parameters
├── history.csv          # Time series of bulk metrics (49 timesteps)
├── history.json         # Same data as JSON
├── metadata.json        # Run metadata (timing, version, etc.)
├── snapshots/           # Per-timestep particle positions (.npz files)
├── fields/              # Phase field arrays (.npz files)
└── plots/               # Pre-generated visualization images
    └── descriptors.json # Final tissue architecture descriptors
```

The key file for mean field fitting is `history.csv`. Each row is one timestep, and the
columns include:

| Column | Meaning |
|--------|---------|
| `time` | Simulation time in hours |
| `phi_f_mean` | Mean functional phase fraction (coarse-grained) |
| `phi_i_mean` | Mean inert phase fraction |
| `phi_v_mean` | Mean void fraction |
| `x_f` | Functional zone fraction (our state variable!) |
| `porosity` | Same as phi_v_mean |
| `n_bridges` | Number of active cell bridges |
| `F_mean` | Mean contact force |
| `K_kozeny_carman` | Computed permeability |

### 7.2 Fitting a Single Run

Let's walk through fitting the mean field model to DOE trial 0010 (a soft hydrogel,
E = 4.0 kPa):

```python
# From the command line:
python analysis/mean_field_model.py -i results/LHC/DOE_2D_0010

# Or programmatically:
from analysis.mean_field_model import from_run, run_all

model, fit_result, data = from_run('results/LHC/DOE_2D_0010')

print(f"Trial parameters:")
print(f"  E_modulus     = {model.E_modulus:.1f} kPa")
print(f"  phi_f (const) = {model.phi_f0:.3f}")
print(f"  phi_i (const) = {model.phi_i0:.3f}")
print(f"  R_func        = {model.R_func:.1f} µm")
print(f"  F_cell        = {model.F_cell:.2f} nN")
print(f"")
print(f"Fitted parameters:")
print(f"  η_eff  = {fit_result['eta_eff']:.4g} nN·h/µm²")
print(f"  σ₀     = {fit_result['sigma_0']:.4g} nN/µm²")
print(f"  α      = {fit_result['alpha']:.3f}")
print(f"  R²     = {fit_result['R2']:.4f}")
```

This gives you the three fitted parameters and a goodness-of-fit metric (R²).

### 7.3 What the Fit Tells You

The fit produces four diagnostic plots:

**1. Compaction fit** (`compaction_fit.png`):

This is the money plot. It overlays the DEM data points (circles) with the mean field
model curve (solid line). A good fit has R² > 0.95.

Things to look for:
- **Good fit, high R²:** The mean field captures the essential physics. The three
  parameters are sufficient to describe the system.
- **Systematic deviation at early times:** The model may not capture the initial
  transient perfectly, because real cells spread at different rates.
- **Deviation at late times:** Could indicate that bridge lock-in or senescence dynamics
  aren't well captured by the simple Poisson model.

**2. Phase evolution** (`phase_evolution.png`):

Shows how the functional zone fraction x_f and local void fractions evolve. The left
panel also shows the jamming limit and deformable limit lines — the model trajectory
should approach but not cross the deformable limit.

**3. Stress balance** (`stress_balance.png`):

Shows the time-evolving competition between cell traction (σ_cell) and contact
resistance (σ_resist). The shaded region between them indicates net compaction stress.
Where the curves cross, compaction arrests.

**4. Permeability evolution** (`permeability_evolution.png`):

Shows how Kozeny-Carman permeability drops as the functional zone compacts. This is
the functional consequence that matters for tissue engineering.

> **Figure reference:** See `figures/example_11_dem_single_fit.png`

### 7.4 Understanding the Fitted Parameters

Here's what the fitted values physically mean for a real DOE run:

**Example: DOE_2D_0010 (E = 4.0 kPa, soft hydrogel)**
```
η_eff ≈ 0.1–1.0   →  Low viscosity, granules rearrange easily
σ₀    ≈ 0.001     →  Weak jamming (soft contacts)
α     ≈ 1.0–1.5   →  Gradual resistance buildup (Hertzian-ish)
```

**Example: DOE_2D_0001 (E = 48.2 kPa, stiff hydrogel)**
```
η_eff ≈ 1.0–10.0  →  Higher viscosity, stiffer matrix
σ₀    ≈ 0.01+     →  Stronger jamming resistance
α     ≈ 1.0–2.0   →  Sharper jamming transition
```

The key insight: **softer hydrogels compact more** (lower E → lower F_cell, but also
lower σ_resist), while **stiffer hydrogels compact less** but provide better structural
support.

### 7.5 Step-by-Step: Your First Fit

Here's a complete hands-on walkthrough. Run each cell in sequence:

```python
# Cell 1: Load the DEM data
import json, csv
import numpy as np

run_dir = 'results/LHC/DOE_2D_0010'

with open(f'{run_dir}/params.json') as f:
    params = json.load(f)

print(f"E_modulus = {params['E_modulus']} kPa")
print(f"phi_f_target = {params['phi_f_target']:.4f}")
print(f"phi_i_target = {params['phi_i_target']:.4f}")
print(f"R_func_mean  = {params['R_func_mean']} µm")

# Cell 2: Read the time series
with open(f'{run_dir}/history.csv') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

t_data = np.array([float(r['time']) for r in rows])
xf_data = np.array([float(r['x_f']) for r in rows])
porosity = np.array([float(r['porosity']) for r in rows])
n_bridges = np.array([float(r['n_bridges']) for r in rows])

print(f"Time range: {t_data[0]:.1f} to {t_data[-1]:.1f} hours")
print(f"x_f range:  {xf_data[0]:.4f} → {xf_data[-1]:.4f}")
print(f"Max bridges: {n_bridges.max():.0f}")

# Cell 3: Plot the raw DEM data
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(t_data, xf_data, 'ko-', ms=3)
axes[0].set_xlabel('Time (h)')
axes[0].set_ylabel('$x_f$')
axes[0].set_title('Functional zone fraction')
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_data, porosity, 'C2o-', ms=3)
axes[1].set_xlabel('Time (h)')
axes[1].set_ylabel('Porosity')
axes[1].set_title('Global porosity')
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_data, n_bridges, 'C1o-', ms=3)
axes[2].set_xlabel('Time (h)')
axes[2].set_ylabel('Bridge count')
axes[2].set_title('Active bridges')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Cell 4: Fit the mean field model
from analysis.mean_field_model import from_run
model, fit, data = from_run(run_dir)

print(f"\nFit results:")
print(f"  η_eff = {fit['eta_eff']:.4g}")
print(f"  σ₀    = {fit['sigma_0']:.4g}")
print(f"  α     = {fit['alpha']:.3f}")
print(f"  R²    = {fit['R2']:.4f}")
```

---

## 8. Comparing Across Experiments: The DOE Sweep

### 8.1 Why Compare Multiple Runs?

A single DEM run gives you one data point. The DOE (Design of Experiments) gives us 19
runs spanning different:
- Modulus (E = 4–100 kPa)
- Composition (func_ratio = 0.33–0.93)
- Granule size (R = 45–100 µm)

By fitting the mean field model to *all* of them, we can see how the fitted parameters
change across conditions and whether the model is universal.

### 8.2 Fitting All DOE Runs

```python
import json
from pathlib import Path
from analysis.mean_field_model import from_run

results = {}
for trial_dir in sorted(Path('results/LHC').glob('DOE_2D_*')):
    if not trial_dir.is_dir():
        continue
    name = trial_dir.name
    try:
        model, fit, data = from_run(str(trial_dir))
        with open(trial_dir / 'params.json') as f:
            params = json.load(f)
        results[name] = {
            'E': params['E_modulus'],
            'func_ratio': params.get('func_ratio', 0.5),
            'R_func': params['R_func_mean'],
            'eta': fit['eta_eff'],
            'sigma_0': fit['sigma_0'],
            'alpha': fit['alpha'],
            'R2': fit['R2'],
            'x_f_final': data['x_f'][-1],
            'compaction': (data['x_f'][0] - data['x_f'][-1]) / data['x_f'][0],
        }
        print(f"  {name}: E={params['E_modulus']:5.1f} kPa, "
              f"R²={fit['R2']:.3f}, "
              f"compaction={results[name]['compaction']:.1%}")
    except Exception as e:
        print(f"  {name}: FAILED - {e}")
```

### 8.3 What to Look For

When you plot the fitted parameters across all DOE runs, you should see physical trends:

**η_eff vs E_modulus:** Expect a positive correlation. Stiffer hydrogels should have
higher effective viscosity because contacts resist rearrangement more.

**σ_0 vs E_modulus:** Should scale roughly linearly. The jamming stress prefactor comes
from Hertzian contact mechanics, where F ~ E × δ^(3/2).

**Compaction ratio vs func_ratio:** More functional granules (higher func_ratio) means
more cell bridges, more driving force, potentially more compaction — but also more
material to compact, so the relationship is nonlinear.

> **Figure reference:** See `figures/example_12_doe_comparison.png`

### 8.4 The Multi-Run Overlay

The most powerful visualization is overlaying the DEM data and mean field fits for
multiple runs on the same axes. This immediately shows:

1. Whether the model captures the *range* of behaviours across conditions
2. Whether there are systematic deviations (suggesting missing physics)
3. How much compaction varies with parameters

> **Figure reference:** See `figures/example_13_multi_run_overlay.png`

---

## 9. From Scalar ODE to Spatial PDE

### 9.1 The Limitation of the Scalar Model

The ODE model assumes the entire scaffold compacts uniformly — the same amount
everywhere. But in reality, compaction varies spatially:

- **Center of scaffold:** More neighbours → higher bridge probability → more compaction
- **Edge of scaffold:** Fewer neighbours → less bridging → less compaction

This creates a **gradient** of compaction that the scalar ODE misses.

### 9.2 The 1D Radial PDE

The `analysis/parameter_sweep.py` module extends the scalar ODE to a **1D radial PDE**:

```
∂x_f/∂t = -x_f(ξ,t) × [σ_cell(ξ,t) - σ_resist(x_f)] / η_eff + D_eff × ∂²x_f/∂ξ²
```

where ξ ∈ [0, 1] is a radial coordinate (0 = center, 1 = edge).

**What's new compared to the scalar ODE:**

1. **Spatial variable ξ:** The functional zone fraction x_f now varies across the scaffold
2. **Spatially-varying bridge rate:** Center has more neighbours, so bridges form faster.
   This is encoded by a `neighbor_factor = 0.5 × (1 + cos(π × ξ))`:
   - At center (ξ=0): factor = 1.0 (all neighbours present)
   - At edge (ξ=1): factor = 0.0 (no neighbours outside the scaffold)
3. **Diffusion term:** D_eff × ∂²x_f/∂ξ² captures stress-driven redistribution of
   packing — if one region is much denser, stress pushes material toward less dense regions

### 9.3 Boundary Conditions

```
∂x_f/∂ξ(0,t) = 0    (zero flux at center — symmetry)
∂x_f/∂ξ(1,t) = 0    (zero flux at edge — no material leaves)
```

### 9.4 The Vectorised Sweep

The real power of the PDE model is that it can be **vectorised** — solving 200,000
parameter combinations simultaneously on a single laptop. This is how we explore the
design space:

```
11 dimensions swept:
  R_func:           [5, 150] µm        (granule radius)
  R_inert:          [5, 300] µm        (inert granule radius)
  phi_solid:        [0.75, 0.95]       (total solid fraction)
  func_ratio:       [0.05, 0.95]       (functional:total ratio)
  E_func:           [0.5, 200] kPa     (functional modulus)
  E_inert:          [0.5, 200] kPa     (inert modulus)
  n_cells_per_func: [1, 30]            (cell loading)
  aspect_ratio_func:    [1.0, 3.0]     (functional shape)
  aspect_ratio_inert:   [1.0, 3.0]     (inert shape)
  blockiness_n2_func:   [2.0, 6.0]     (functional blockiness)
  blockiness_n2_inert:  [2.0, 6.0]     (inert blockiness)
```

Each combination is a different scaffold design. The PDE predicts the final compaction,
permeability, tissue growth, and organ distance for *every single one*.

### 9.5 Tissue Volume Growth

The PDE model also includes tissue growth (V2.1+):

```
dφ_tissue/dt = k_tissue × (n_cells/n_ref) × maturity × driver × (φ_tissue_max - φ_tissue)
```

This is a spatially-varying logistic growth:
- Tissue fills the void space left by compaction
- More bridges → more ECM secretion → faster tissue growth
- Center grows faster than edge (more neighbours)
- Saturates at `alpha_fill × φ_void_local`

This is important because tissue volume affects the final architecture descriptors used
for organ matching.

---

## 10. The Energy Landscape: A Thermodynamic View

### 10.1 From Forces to Energy

Everything we've done so far is in the *force* picture: driving stress vs resistance
stress. But there's a complementary *energy* picture that gives deeper insight.

The `analysis/energy_landscape.py` module computes a **free energy landscape** G(ξ)
where ξ = compaction coordinate (ξ = 0 means no compaction, ξ → 1 means fully compacted).

### 10.2 Six Energy Terms

The total free energy is decomposed into six physically motivated terms:

```
G_total(ξ) = G_cell + G_elastic + G_yield + G_void + G_inert + G_surface
```

| Term | Physical Origin | Sign | Meaning |
|------|----------------|------|---------|
| G_cell | Cell traction | Negative (drives compaction) | Cells lower energy by compacting |
| G_elastic | Hertzian contacts | Positive (resists) | Contacts store elastic energy |
| G_yield | Granular friction | Positive (barrier) | Activation energy to start compaction |
| G_void | Osmotic pressure | Positive (resists) | Cost of redistributing void |
| G_inert | Inert frustration | Positive (resists) | Inert granules geometrically block |
| G_surface | Interface tension | Mixed | Interfacial energy at func/inert boundary |

### 10.3 The Energy Picture

```
G(ξ)
 │
 │     ╱╲                          ← Barrier (yield + elastic)
 │    ╱  ╲
 │   ╱    ╲
 │  ╱      ╲──────  ← Equilibrium (minimum)
 │ ╱               ╲
 │╱    Cell traction  ╲           ← Cell energy pulls system downhill
 │     drives toward   ╲
 │     minimum          ╲        ← If enough driving force, crosses barrier
 └──────────────────────────── ξ
   0            ξ*          1
```

The system evolves by rolling downhill on this energy landscape:
- **dξ/dt = -(1/η) × dG/dξ** (overdamped dynamics)
- The equilibrium is at the energy minimum (ξ*)
- If there's a barrier, the system must overcome it to start compacting

### 10.4 Dimensionless Groups

The energy landscape naturally defines dimensionless groups that collapse the parameter
space:

```
β = σ_cell / σ_0           Cell-to-jamming ratio (how hard cells push vs how stiff the packing)
Ca = σ_cell / σ_yield      Cellular capillary number (how easily cells overcome friction)
Φ_r = φ_i / φ_f            Inert obstruction ratio
Ψ = φ_f_local / φ_RCP      Proximity to jamming (1 = just jammed)
Γ = γ_fi / σ_cell           Interfacial number
```

These groups allow you to predict compaction *without* solving any equations — just
by knowing the ratios of competing effects.

---

## 11. Summary and Key Takeaways

### The Building Blocks

| Example | State Variable | Driving Force | Resistance | Key Lesson |
|---------|---------------|--------------|------------|------------|
| Coffee cooling | Temperature T | — | Proportional to (T - T_room) | Exponential decay to equilibrium |
| Population growth | Population N | Growth rate r×N | Carrying capacity (1 - N/K) | Nonlinearity creates saturation |
| Thermostat | Temperature T | Heater Q(t) | Cooling above threshold | Two competing forces → equilibrium |
| Porous medium | Void fraction φ_v | — | — | Structure → function (Kozeny-Carman) |
| **GELS** | **Zone fraction x_f** | **Cell traction σ_cell(t)** | **Jamming σ_resist(x_f)** | **All of the above combined** |

### The GELS Model at a Glance

```
                    ┌─────────────────────────────┐
                    │     dx_f/dt = ?              │
                    │                              │
  Cell biology ──→  │  -x_f × (σ_cell - σ_resist) │  ←── Granular mechanics
  (time-dependent)  │  ─────────────────────────── │      (state-dependent)
                    │           η_eff              │
                    │                              │
                    │  ← Effective viscosity        │
                    └─────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────────────────┐
                    │  x_f(t) trajectory           │
                    │  → local void fractions      │
                    │  → permeability (K-C)        │
                    │  → tissue descriptors        │
                    │  → organ distance            │
                    └─────────────────────────────┘
```

### The Three Levels of the Mean Field Model

```
┌──────────────────────────────────────────────────────────────┐
│  Level 1: SCALAR ODE  (analysis/mean_field_model.py)         │
│  - One number: x_f(t)                                        │
│  - 3 fitted parameters: η_eff, σ₀, α                        │
│  - Post-hoc fit to individual DEM runs                       │
│  - Best for: understanding, validation, quick analysis       │
├──────────────────────────────────────────────────────────────┤
│  Level 2: RADIAL PDE  (analysis/parameter_sweep.py)          │
│  - Spatial field: x_f(ξ, t) on 20-point radial grid         │
│  - 11 swept parameters, vectorised (200k samples in seconds) │
│  - Tissue growth, organ distance prediction                  │
│  - Best for: design optimization, organ targeting            │
├──────────────────────────────────────────────────────────────┤
│  Level 3: ENERGY LANDSCAPE  (analysis/energy_landscape.py)   │
│  - Free energy G(ξ) with 6 mechanistic terms                │
│  - Dimensionless groups (β, Ca, Φ_r, Ψ, Γ)                  │
│  - No fit parameters — all from physics                      │
│  - Best for: insight, scaling laws, data collapse             │
└──────────────────────────────────────────────────────────────┘
```

### The Workflow

```
1. Run DEM (new_dem_0.py)  →  Detailed particle data (hours)
                               │
2. Fit mean field              │  Validate: R² > 0.95?
   (mean_field_model.py)    ←──┘  If not, check model assumptions
                               │
3. Parameter sweep             │  Explore: 200,000 designs
   (parameter_sweep.py)     ←──┘  in seconds
                               │
4. Organ targeting             │  Predict: which scaffold
   (organ_targets.py)       ←──┘  matches which organ?
                               │
5. Design experiments          │  Validate with
   (real scaffolds)         ←──┘  confocal imaging
```

### Key Equations Reference

| Equation | What It Does |
|----------|-------------|
| `F_cell = F_stall × β × engagement` | Motor-clutch cell traction |
| `σ_cell = n_density × F_cell × f_bridge × maturity × ramp` | Total cell stress |
| `σ_resist = σ₀ × (φ_local/φ_RCP - 1)^α` | Jamming resistance |
| `dx_f/dt = -x_f × (σ_cell - σ_resist) / η_eff` | Compaction ODE |
| `K = φ_v³ × d² / [180(1-φ_v)²]` | Kozeny-Carman permeability |

### Tips for the Beginner

1. **Start simple.** If your mean field model has more than 3–5 free parameters, it's
   probably overfitting. Our model has exactly 3 free parameters.

2. **Validate against data.** A mean field model is only as good as its match to reality
   (or detailed simulations). Always compute R² and look at the residuals.

3. **Understand the limits.** Mean field models average over spatial fluctuations. They
   work well for bulk properties (average compaction, permeability) but miss local effects
   (individual bridges, stress concentrations).

4. **Use dimensional analysis.** If you can write your model in terms of dimensionless
   groups (like β, Ca, φ/φ_RCP), you reduce the parameter space and gain physical insight.

5. **The model is a tool, not the truth.** Use it to generate hypotheses, guide
   experiments, and explore design space — then verify with DEM and experiments.

6. **Run `generate_examples.py` yourself.** All the figures in this guide can be
   regenerated from `CodeLog/MeanFieldModeling/generate_examples.py`. Try modifying
   parameters and re-running to build your intuition.

---

## Appendix A: File Reference

| File | Purpose |
|------|---------|
| `analysis/mean_field_model.py` | Two-zone compaction ODE model, fitting, plotting |
| `analysis/parameter_sweep.py` | 11-D LHS sweep with 1D radial PDE + tissue growth |
| `analysis/energy_landscape.py` | Free energy decomposition (6 terms), kinetics, dimensionless |
| `analysis/tissue_descriptors.py` | 18 tissue architecture descriptors |
| `analysis/organ_targets.py` | 7 organ target vectors for design optimization |
| `viz/dimensionless.py` | Dimensionless analysis and data collapse |
| `CodeLog/MeanFieldModeling/generate_examples.py` | Script to generate all figures in this guide |

## Appendix B: Generating the Figures

All figures in this guide are generated by running:

```bash
cd /path/to/GELS
python CodeLog/MeanFieldModeling/generate_examples.py
```

This produces all figures in `CodeLog/MeanFieldModeling/figures/`:

| Figure | File | Section |
|--------|------|---------|
| Coffee Cooling | `example_01_coffee_cooling.png` | 2 |
| Logistic Growth | `example_02_logistic_growth.png` | 3 |
| Competing Forces | `example_03_competing_forces.png` | 4 |
| Kozeny-Carman | `example_04_kozeny_carman.png` | 5 |
| Volume Conservation | `example_05_volume_conservation.png` | 6.3 |
| Motor-Clutch | `example_06_motor_clutch.png` | 6.4 |
| Jamming Resistance | `example_07_jamming_resistance.png` | 6.5 |
| Full Model (4-panel) | `example_08_full_model.png` | 6.7 |
| Biological Timeline | `example_09_biological_timeline.png` | 6.8 |
| Parameter Sensitivity | `example_10_parameter_sensitivity.png` | 6.9 |
| Single DEM Fit | `example_11_dem_single_fit.png` | 7.3 |
| DOE Comparison | `example_12_doe_comparison.png` | 8.3 |
| Multi-Run Overlay | `example_13_multi_run_overlay.png` | 8.4 |
| Energy Landscape | `example_14_energy_landscape.png` | 10 |

## Appendix C: Quick-Start Cheat Sheet

```python
# === Fit mean field to one DEM run ===
from analysis.mean_field_model import from_run
model, fit, data = from_run('results/LHC/DOE_2D_0006')
# fit['eta_eff'], fit['sigma_0'], fit['alpha'], fit['R2']

# === Generate all diagnostic plots ===
from analysis.mean_field_model import run_all
model, fit, data = run_all('results/LHC/DOE_2D_0006')

# === Build a model from scratch (no DEM data needed) ===
from analysis.mean_field_model import CompactionModel, MotorClutchParams
mc = MotorClutchParams()
model = CompactionModel(
    E_modulus=10.0, phi_f0=0.40, phi_i0=0.25,
    R_func=40.0, n_cells_per_granule=8, N_func=160,
    domain_volume=800**2, mc_params=mc
)
sol = model.solve((0, 72), eta_eff=0.5, sigma_0=0.001, alpha=1.2)

# === Run a 200k parameter sweep ===
from analysis.parameter_sweep import generate_lhs, run_sweep_vectorised
samples = generate_lhs(200_000, seed=42)
outputs = run_sweep_vectorised(samples)
```

---

*Document version: V2.4 — March 2026*
*GELS Project, McGhee Lab, UIUC*
