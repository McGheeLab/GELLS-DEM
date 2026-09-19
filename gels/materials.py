"""
Material and functionalization rules shared by the engine loops and the kernels.
==================================================================================

Every dependence of the physics on the degree of collagen-I functionalization
``f ∈ [0, 1]`` of a granule lives here, as small ``@njit`` leaf functions with
no Python objects, so the pure-Python reference loops
(``gels.kernels.reference``) and the compiled kernels (``gels.kernels.*``)
use one definition. All rules reduce **exactly** to the V2.7 binary model at
``f ∈ {0, 1}`` (f = 1 ≡ the old "functional" granule, f = 0 ≡ "inert").

Rules
-----
* **Pair adhesion / friction** — bilinear *coverage mixing*: a contact patch
  between granules with coated fractions f_i, f_j is collagen–collagen with
  probability f_i·f_j, collagen–bare with f_i(1−f_j) + f_j(1−f_i) and
  bare–bare otherwise::

      X(f_i, f_j) = f_i f_j X_cc + [f_i(1−f_j) + f_j(1−f_i)] X_cb + (1−f_i)(1−f_j) X_bb

  Walls are bare: ``X_wall(f_i) = f_i X_cb + (1−f_i) X_bb``.

* **Effective modulus** from per-granule E_i, ν_i (kPa → Pa)::

      1/E* = (1−ν_i²)/E_i + (1−ν_j²)/E_j        (identical bodies: E/(2(1−ν²)))

* **Cell traction vs. ligand density** — the literature (Gaudet 2003, Roberts
  & Mrksich 1998, Erdmann & Schwarz 2004; see
  ``CodeLog/References/fibroblast_parameters.md``) supports a *saturating*
  response in absolute ligand density, not a linear one in coating fraction.
  With σ = f·σ_max ligands/µm² and half-saturation K_σ, the normalised
  Langmuir gain is::

      g(f) = f (1 + κ) / (f + κ),   κ = K_σ / σ_max,   g(0) = 0, g(1) = 1

  A power law ``g = f^γ`` is kept as an alternative (``LAW_POWER``). For a
  bridge, ``f_eff`` is chosen by ``RULE_TARGET`` (the target's coverage — the
  host anchorage is already encoded in the cell count), ``RULE_MIN`` or
  ``RULE_PRODUCT``. ``g`` multiplies the motor-clutch force **and** its clutch
  stiffness; it must never be applied by scaling ``n_clutches`` (in the
  motor-clutch formula that *raises* traction).

* **Line-of-sight blocker penalty** for bridge formation::

      b(f_k) = b_bare + (1 − b_bare) f_k       (b(0) = bridge_inert_factor, b(1) = 1)
"""

import numpy as np

from gels.kernels import njit

# Traction law / rule codes (ints so they can be passed into kernels).
LAW_LANGMUIR = 0
LAW_POWER = 1
RULE_TARGET = 0
RULE_MIN = 1
RULE_PRODUCT = 2

LAW_CODES = {'langmuir': LAW_LANGMUIR, 'power': LAW_POWER}
RULE_CODES = {'target': RULE_TARGET, 'min': RULE_MIN, 'product': RULE_PRODUCT}


def law_code(name):
    try:
        return LAW_CODES[str(name).lower()]
    except KeyError:
        raise ValueError(f"traction_f_law must be one of {sorted(LAW_CODES)}, got {name!r}")


def rule_code(name):
    try:
        return RULE_CODES[str(name).lower()]
    except KeyError:
        raise ValueError(f"traction_f_rule must be one of {sorted(RULE_CODES)}, got {name!r}")


# ──────────────────────────────────────────────────────────────────────
# Pair properties
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def mix_pair(fi, fj, x_cc, x_cb, x_bb):
    """Bilinear coverage mixing of a pair property (adhesion energy W or τ₀).

    Reduces to x_cc at (1,1), x_bb at (0,0) and x_cb at (1,0)/(0,1).
    """
    return (fi * fj * x_cc
            + (fi * (1.0 - fj) + fj * (1.0 - fi)) * x_cb
            + (1.0 - fi) * (1.0 - fj) * x_bb)


@njit(cache=True)
def mix_wall(fi, x_cb, x_bb):
    """Pair property against a bare wall: X(f_i, 0)."""
    return fi * x_cb + (1.0 - fi) * x_bb


@njit(cache=True)
def pair_E_star_Pa(Ei_kPa, nui, Ej_kPa, nuj):
    """Effective contact modulus in Pa from per-granule E (kPa) and ν.

    1/E* = (1−ν_i²)/E_i + (1−ν_j²)/E_j. For identical bodies this equals
    E/(2(1−ν²)) — the V2.7 `E_star_gg`.
    """
    return 1e3 / ((1.0 - nui * nui) / Ei_kPa + (1.0 - nuj * nuj) / Ej_kPa)


@njit(cache=True)
def wall_E_star_Pa(Ei_kPa, nui):
    """Effective modulus against a rigid wall in Pa: E_i/(1−ν_i²) (V2.7 `E_star_gw`)."""
    return 1e3 * Ei_kPa / (1.0 - nui * nui)


# ──────────────────────────────────────────────────────────────────────
# Functionalization → cell response
# ──────────────────────────────────────────────────────────────────────

@njit(cache=True)
def ligand_factor(f, kappa):
    """Normalised Langmuir gain g(f) = f(1+κ)/(f+κ), κ = K_σ/σ_max.

    g(0) = 0, g(1) = 1, strictly increasing. κ → ∞ recovers the linear law
    g = f; κ → 0 is a step (any coating saturates the response).
    """
    if f <= 0.0:
        return 0.0
    if f >= 1.0:
        return 1.0
    if kappa <= 0.0:
        return 1.0
    return f * (1.0 + kappa) / (f + kappa)


@njit(cache=True)
def traction_gain(f_host, f_target, law, rule, gamma, kappa):
    """Multiplier on per-cell traction for a cell on `f_host` gripping `f_target`.

    rule: RULE_TARGET (f_eff = f_target) | RULE_MIN | RULE_PRODUCT
    law:  LAW_LANGMUIR (ligand_factor(f_eff, kappa)) | LAW_POWER (f_eff ** gamma)
    """
    if rule == RULE_MIN:
        f_eff = min(f_host, f_target)
    elif rule == RULE_PRODUCT:
        f_eff = f_host * f_target
    else:
        f_eff = f_target
    if f_eff <= 0.0:
        return 0.0
    if f_eff >= 1.0:
        return 1.0
    if law == LAW_POWER:
        return f_eff ** gamma
    return ligand_factor(f_eff, kappa)


@njit(cache=True)
def hill_pair_factor(v_rel, F_prev, F_iso, c_pair, v0, f_ecc, at_min_gap):
    """Force-velocity multiplier shared by every bridge of one granule pair (V3.1).

    A bridging cell is an active element with stall force F_s and unloaded
    shortening speed v0. Evaluating F = F_s (1 - v/v0) on the measured
    closing speed oscillates, because the bridges' own pull is most of that
    speed; so the pair's own contribution is solved for instead:

        v_env = v_rel - c_pair F_prev        closing speed imposed by everything else
        fac   = (1 - v_env/v0) / (1 + c_pair F_iso/v0)

    with ``c_pair`` the pair mobility 1/gamma_i + 1/gamma_j, ``F_prev`` the
    force the bridges applied last step and ``F_iso`` their stall capacity.
    Limits: a rigid environment (v_rel = 0, F_prev = F_iso) gives fac = 1,
    i.e. the cell holds its stall force at force balance (tensional
    homeostasis); a free pair closes at v0 x/(1+x) < v0 with x = c F_iso/v0
    and is a stable fixed point; a bridge being stretched (v_rel < 0) gets up
    to 1 + f_ecc (eccentric); and v0 -> inf gives exactly 1.0, the constant
    actuator of V2.7.
    """
    if v0 <= 0.0 or at_min_gap:
        return 1.0
    v_env = v_rel - c_pair * F_prev
    fac = (1.0 - v_env / v0) / (1.0 + c_pair * F_iso / v0)
    if fac < 0.0:
        return 0.0
    hi = 1.0 + f_ecc
    if fac > hi:
        return hi
    return fac


@njit(cache=True)
def blocker_factor(f_blocker, bare_factor):
    """Bridge-path penalty when a granule of coverage f_blocker sits in the line of sight.

    Piecewise so the endpoints are exact in floating point (bare_factor at
    f = 0, 1.0 at f = 1 — the two V2.7 branches), linear in between.
    """
    if f_blocker <= 0.0:
        return bare_factor
    if f_blocker >= 1.0:
        return 1.0
    return bare_factor + (1.0 - bare_factor) * f_blocker


# ──────────────────────────────────────────────────────────────────────
# Species tables (plain numpy; built once per GranuleSystem)
# ──────────────────────────────────────────────────────────────────────

def build_pair_tables(species_f, species_E_kPa, species_nu,
                      W_cc, W_cb, W_bb, tau_cc, tau_cb, tau_bb,
                      E_cap_kPa=0.0, f_wall=0.0):
    """K×K pair tables and K wall tables for a species list.

    Returns dict with 'pair_W' (J/m²), 'pair_tau' (Pa), 'pair_Estar' (Pa),
    'wall_W', 'wall_Estar'. For the legacy two species (f = 1 collagen,
    f = 0 bare) the tables reproduce the V2.7 friction/adhesion LUT exactly.

    V3.1: ``E_cap_kPa`` > 0 caps the modulus entering E* (contact stiffness
    only — the per-granule E seen by the cells is untouched), so a rigid
    material such as PMMA can be declared literally while the explicit
    integrator keeps a bounded contact force. ``f_wall`` is the collagen
    coverage of the container wall: the wall adhesion becomes
    mix_pair(f_i, f_wall, ...) instead of the bare-wall mix_wall(f_i, ...).
    Both defaults (0) reproduce the V3.0 tables bit for bit.
    """
    f = [float(v) for v in species_f]
    E_true = [float(v) for v in species_E_kPa]
    E = [min(v, float(E_cap_kPa)) if E_cap_kPa > 0.0 else v for v in E_true]
    nu = [float(v) for v in species_nu]
    K = len(f)
    pair_W = np.empty((K, K))
    pair_tau = np.empty((K, K))
    pair_Estar = np.empty((K, K))
    wall_W = np.empty(K)
    wall_Estar = np.empty(K)
    # Scalar Python-float arithmetic in the V2.7 operation order, so that for
    # identical materials the tables are bit-identical to the old global
    # constants E_star_gg = (E*1e3)/(2(1-nu**2)) and E_star_gw = (E*1e3)/(1-nu**2)
    # (Python's float ** 2 is not always x*x on every platform, so the legacy
    # `nu ** 2` is reproduced literally).
    for a in range(K):
        nu2_a = nu[a] ** 2
        wall_Estar[a] = (E[a] * 1e3) / (1.0 - nu2_a)
        if f_wall > 0.0:
            wall_W[a] = mix_pair(f[a], f_wall, W_cc, W_cb, W_bb)
        else:
            wall_W[a] = mix_wall(f[a], W_cb, W_bb)
        for b in range(K):
            pair_W[a, b] = mix_pair(f[a], f[b], W_cc, W_cb, W_bb)
            pair_tau[a, b] = mix_pair(f[a], f[b], tau_cc, tau_cb, tau_bb)
            if E[a] == E[b] and nu[a] == nu[b]:
                pair_Estar[a, b] = (E[a] * 1e3) / (2.0 * (1.0 - nu2_a))
            else:
                pair_Estar[a, b] = pair_E_star_Pa(E[a], nu[a], E[b], nu[b])
    return {
        'pair_W': np.ascontiguousarray(pair_W),
        'pair_tau': np.ascontiguousarray(pair_tau),
        'pair_Estar': np.ascontiguousarray(pair_Estar),
        'wall_W': np.ascontiguousarray(wall_W),
        'wall_Estar': np.ascontiguousarray(wall_Estar),
    }


__all__ = [
    'LAW_LANGMUIR', 'LAW_POWER', 'RULE_TARGET', 'RULE_MIN', 'RULE_PRODUCT',
    'LAW_CODES', 'RULE_CODES', 'law_code', 'rule_code',
    'mix_pair', 'mix_wall', 'pair_E_star_Pa', 'wall_E_star_Pa', 'hill_pair_factor',
    'ligand_factor', 'traction_gain', 'blocker_factor', 'build_pair_tables',
]
