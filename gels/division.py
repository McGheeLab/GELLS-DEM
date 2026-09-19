"""
Cell division (V3.1).
=====================

Fibroblasts divide with a doubling time of 20-30 h, so over a 72 h run the
population roughly triples. V3.0 seeded a fixed number of cells and only ever
lost them to senescence; this module lets the count grow.

One pass per step, called from both ``update_cell_state`` twins after the
per-granule aggregates and before the per-cell state machine — the one point
in a step where nothing holds a live view into the ``cell_*`` arrays, so they
can be reallocated. Both backends call the same function and draw from the
same ``Generator``, so their division statistics are identical.

Per eligible cell the probability of dividing in a step of length dt is

    p_div = (1 - exp(-ln2 dt / T_d)) * max(0, 1 - n_cells / (capacity * max_layers))

so growth is exponential with the population doubling time while there is
room, and stops at confluence (contact inhibition). A cell whose host granule
is full colonises the granule it bridges to, if that one has room — which is
how a colony spreads through the bed rather than piling up on its seed.

Cells are never created on a granule that cells cannot grip
(``f < f_min_adhesion``).
"""

import numpy as np

from gels.engine import (CellState, PACKING_EFFICIENCY, capacity_coverage, cell_projected_area,
                         cells_from_surface_coverage)

__all__ = ['divide_cells', 'capacity_vector', 'age_cells', 'cycle_metrics']


def capacity_vector(gs, p):
    """Per-granule cell capacity (float), the same rule ``cell_capacity`` applies.

    Vectorised over granules: surface-coverage capacity when a coverage is
    set, else the legacy projected-area cap, both scaled by the seeding gain
    so a partly coated granule holds proportionally fewer cells.
    """
    N = gs.N
    cov = capacity_coverage(p)
    r = gs.r[:N]
    if cov > 0:
        # V3.2: capacity is set by the FOOTHOLD a cell needs, not by the area it
        # would cover if it had unlimited room -- see cells_from_surface_coverage.
        A_cell = cell_projected_area(1.0, p.cell_diameter, p.cell_height_spread)
        foothold = float(getattr(p, 'cell_capacity_foothold', 1.0))
        if foothold < 1.0:
            A_cell = max(cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread),
                         foothold * A_cell)
        area = 4.0 * np.pi * r ** 2 if p.mode in ('3D', '2D-slice') else np.pi * r ** 2
        cap = np.maximum(1.0, np.round(area * cov * PACKING_EFFICIENCY / A_cell))
    else:
        A_cell = cell_projected_area(0.0, p.cell_diameter, p.cell_height_spread)
        cap = np.floor(np.pi * r ** 2 * p.cell_coverage / A_cell)
    return np.maximum(0.0, cap * gs.activity[:N])


def age_cells(gs, p):
    """Advance every live cell's cycle clock by one step (V3.2).

    V3.1 allocated ``cell_age``, read it in ``divide_cells`` and reset it on
    division -- but nothing ever advanced it. The compiled kernel's parameter of
    the same name is bound to ``cell_bridge_age``, a different array, and that
    collision hid the omission. With the shipped default
    ``cell_division_min_age = 8 h`` the eligibility test was therefore false for
    ever and no cell divided in a real run; the tests passed because they all
    set ``min_age = 0``.

    Called from both ``update_cell_state`` twins immediately before
    ``divide_cells`` so the two backends cannot drift. It runs unconditionally
    -- it is a clock, and ``cell_age`` is a V3.1 key that the V2.7 fixtures do
    not carry, so legacy identity is unaffected.

    Under the ``cycle`` division model a cell whose granule is at capacity does
    not advance (G0 arrest), so a confluent population holds its phase and
    resumes staggered instead of dividing in a burst once room appears.
    """
    C = gs.total_cells
    if C == 0:
        return
    live = gs.cell_state[:C] != int(CellState.SENESCENT)
    if getattr(p, 'cell_division_model', 'poisson') == 'cycle':
        cap = capacity_vector(gs, p) * float(p.cell_division_max_layers)
        has_room = cap > gs.n_cells[:gs.N]
        live = live & has_room[gs.cell_granule_id[:C]]
    gs.cell_age[:C] += np.where(live, float(p.dt), 0.0)


def divide_cells(gs, p, rng, t):
    """One division pass. Returns the number of daughters created.

    No-op (and no random draws, so legacy runs are untouched) unless
    ``cell_division_enabled``.
    """
    if not getattr(p, 'cell_division_enabled', False) or gs.total_cells == 0 or rng is None:
        return 0

    state = gs.cell_state
    host = gs.cell_granule_id
    alive = state != int(CellState.SENESCENT)
    eligible = alive & (gs.cell_age >= p.cell_division_min_age)
    if not p.cell_divide_while_bridging:
        eligible &= gs.cell_bridge_target < 0
    idx = np.nonzero(eligible)[0]
    if idx.size == 0:
        return 0

    cap = capacity_vector(gs, p) * float(p.cell_division_max_layers)
    n_cells = gs.n_cells[:gs.N]
    room = np.where(cap > 0, np.maximum(0.0, 1.0 - n_cells / np.maximum(cap, 1e-12)), 0.0)

    h = host[idx]
    room_host = room[h]
    tgt = gs.cell_bridge_target[idx]
    has_tgt = (tgt >= 0) & (tgt < gs.N)
    tgt_safe = np.where(has_tgt, tgt, 0)
    room_tgt = np.where(has_tgt & gs.adhesive_mask[tgt_safe], room[tgt_safe], 0.0)

    # host first; a cell on a full granule colonises the one it bridges to
    use_tgt = (room_host <= 0.0) & (room_tgt > 0.0)
    room_eff = np.where(use_tgt, room_tgt, room_host)
    dest = np.where(use_tgt, tgt_safe, h)

    if getattr(p, 'cell_division_model', 'poisson') == 'cycle':
        # V3.2: each cell divides when it reaches its OWN cycle length. The
        # spread comes from the lognormal draw at seeding and after every
        # division, not from a memoryless coin flip -- whose cycle times have
        # CV = 1, far wider than the 0.1-0.3 real fibroblasts show. No draw is
        # consumed for the decision, so both backends stay in step.
        born = (gs.cell_age[idx] >= _cycle_times(gs, p, idx)) & (room_eff > 0.0)
    else:
        rate = 1.0 - np.exp(-np.log(2.0) * p.dt / max(1e-9, p.cell_doubling_time))
        p_div = rate * room_eff
        u = rng.random(idx.size)                  # one draw per eligible cell, both backends
        born = u < p_div
    if not np.any(born):
        return 0

    parents = idx[born]
    dest = dest[born].astype(int)

    # cap births per destination at the free capacity, keeping the lowest cell index
    order = np.argsort(dest, kind='stable')
    parents, dest = parents[order], dest[order]
    free = np.floor(np.maximum(0.0, cap[dest] - n_cells[dest])).astype(int)
    rank = np.zeros(dest.size, dtype=int)
    if dest.size > 1:
        same = dest[1:] == dest[:-1]
        run = 0
        for k in range(1, dest.size):
            run = run + 1 if same[k - 1] else 0
            rank[k] = run
    keep = rank < np.maximum(free, 0)
    if not np.any(keep):
        return 0
    parents, dest = parents[keep], dest[keep]

    theta, eta, omega = _daughter_angles(gs, parents, dest)
    gen = np.minimum(gs.cell_generation[parents].astype(np.int16) + 1, 127).astype(np.int8)

    # The parent restarts its cycle and a bridging parent re-ramps its force.
    # This MUST happen before add_cells: the CSR rebuild shifts every absolute
    # cell index above a birth, so `parents` is stale afterwards. V3.1 reset
    # after the rebuild and so zeroed the wrong cells -- harmless for the
    # memoryless rule, fatal for the cycle model, where a parent that never
    # resets divides again on every step.
    gs.cell_age[parents] = 0.0
    bridging = gs.cell_state[parents] == int(CellState.BRIDGING)
    if np.any(bridging):
        gs.cell_bridge_age[parents[bridging]] = 0.0
    # V3.2: mother and daughter draw fresh cycle lengths, so a lineage does not
    # stay locked in phase for ever -- both sit at age 0 here by construction.
    _draw_cycle_times(gs, p, rng, parents)

    daughters = gs.add_cells(dest, theta=theta, eta=eta, omega=omega, generation=gen)
    _draw_cycle_times(gs, p, rng, daughters)

    # hosts that gained cells: keep the per-granule aggregates consistent
    touched = np.unique(dest)
    frac = np.where(gs.n_cells[touched] > 0,
                    gs.n_attached[touched] / np.maximum(gs.n_cells[touched] - np.bincount(
                        dest, minlength=gs.N)[touched], 1.0), 1.0)
    gs.n_attached[touched] = np.minimum(gs.n_cells[touched], gs.n_attached[touched] + np.bincount(
        dest, minlength=gs.N)[touched] * np.clip(frac, 0.0, 1.0))
    cap_t = capacity_vector(gs, p)[touched]
    gs.n_overcrowded[touched] = np.maximum(0.0, gs.n_attached[touched] - cap_t)

    gs.n_divisions_cum = int(getattr(gs, 'n_divisions_cum', 0)) + int(parents.size)
    return int(parents.size)


def cycle_metrics(gs):
    """Cell-cycle diagnostics (V3.2) -- how spread the population actually is.

    ``cell_age_cv`` is the instrument for the synchrony problem: a population
    seeded all at phase zero reads 0, and one drawn uniformly over the cycle
    reads about 1/sqrt(3) = 0.577. Shared by both metrics twins so they cannot
    disagree. The per-hour division rate is deliberately NOT computed here --
    it differentiates cleanly from ``n_divisions_cum`` at plot time, which
    needs no extra state and works on runs recorded before this change.
    """
    C = int(gs.total_cells)
    if C == 0:
        return {'cell_age_mean': 0.0, 'cell_age_cv': 0.0, 'cell_cycle_time_mean': 0.0}
    age = gs.cell_age[:C]
    mu = float(age.mean())
    return {
        'cell_age_mean': mu,
        'cell_age_cv': float(age.std() / mu) if mu > 1e-12 else 0.0,
        'cell_cycle_time_mean': float(gs.cell_cycle_time[:C].mean()),
    }


def _cycle_times(gs, p, idx):
    """Each cell's own cycle length, falling back to the population doubling time."""
    T_d = max(1e-9, float(p.cell_doubling_time))
    T_i = gs.cell_cycle_time[idx]
    return np.where(T_i > 0.0, T_i, T_d)


def _draw_cycle_times(gs, p, rng, idx):
    """Give these cells a fresh cycle length (lognormal, median = T_d).

    Consumes a draw only in the ``cycle`` model with a non-zero CV, so the
    poisson path's RNG consumption is unchanged.
    """
    if idx.size == 0:
        return
    T_d = max(1e-9, float(p.cell_doubling_time))
    cv = max(0.0, float(getattr(p, 'cell_division_cv', 0.0)))
    if getattr(p, 'cell_division_model', 'poisson') == 'cycle' and cv > 0.0:
        sigma = np.sqrt(np.log1p(cv * cv))
        gs.cell_cycle_time[idx] = T_d * np.exp(sigma * rng.standard_normal(idx.size))
    else:
        gs.cell_cycle_time[idx] = T_d


def _daughter_angles(gs, parents, dest):
    """Surface coordinates of the daughters: beside the parent, or facing it on the target."""
    r_dest = np.maximum(gs.r[dest], 1e-9)
    # One cell-diameter of arc beside the parent. The 20.0 um is the default
    # cell_diameter hard-coded: this helper has no `p`, so it cannot read
    # p.cell_diameter without a signature change. Correct for the default and
    # wrong for any other cell size -- worth fixing, but it is a behaviour
    # change, so not here. (V3.3: dropped a `gs.cell_diameter_offset` hasattr
    # branch that was dead -- the attribute is never assigned anywhere.)
    offset = np.minimum(np.pi / 2.0, 20.0 / r_dest)
    same = dest == gs.cell_granule_id[parents]
    theta = np.where(same, gs.cell_theta_local[parents] + offset, 0.0)
    eta = np.where(same, gs.cell_eta_local[parents], 0.0)
    omega = np.where(same, gs.cell_omega_local[parents] + offset, 0.0)
    # colonising a new granule: land on the side facing the parent's granule
    other = ~same
    if np.any(other):
        src = gs.cell_granule_id[parents[other]]
        d = gs.pos[src] - gs.pos[dest[other]]
        nrm = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
        theta[other] = np.arctan2(d[:, 1], d[:, 0])
        eta[other] = np.arcsin(np.clip(d[:, 2] / nrm, -1.0, 1.0)) * 0.8
        omega[other] = np.arctan2(d[:, 1], d[:, 0])
    eta = np.clip(eta, -0.8 * np.pi / 2.0, 0.8 * np.pi / 2.0)
    return theta % (2.0 * np.pi), eta, omega % (2.0 * np.pi)
