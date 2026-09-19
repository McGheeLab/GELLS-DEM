"""
Colours and species tables for GELS visualisation (matplotlib-free).
====================================================================

Split out of ``viz2/common.py`` in V3.0 so the live viewer process — which
must not import anything that pins the Agg backend — can share the exact
colour conventions of the post-processing suite. ``viz2.common`` re-exports
everything here.

Granules are coloured **by species** (``species_table``); the legacy
two-species runs keep the historical red (collagen-coated / functional) and
green (bare / inert). Cells are coloured by ``CellState``.
"""

import numpy as np

from gels.engine import CellState, LEGACY_SPECIES_COLORS, resolve_species

# ── granule / void colours ───────────────────────────────────────────────
FUNC_COLOR = LEGACY_SPECIES_COLORS[0]     # '#CC2222' red   — collagen-coated (legacy "functional")
INERT_COLOR = LEGACY_SPECIES_COLORS[1]    # '#22AA22' green — bare (legacy "inert")
VOID_COLOR = '#000000'                    # black background (void space)

# ── cell state colours ───────────────────────────────────────────────────
CELL_COLORS = {
    int(CellState.ATTACHED):      '#DAA520',   # goldenrod
    int(CellState.SPREADING):     '#FF8C00',   # dark orange
    int(CellState.PROLIFERATING): '#228B22',   # forest green
    int(CellState.BRIDGING):      '#DC143C',   # crimson
    int(CellState.SENESCENT):     '#696969',   # dim gray
    int(CellState.MIGRATING):     '#4169E1',   # royal blue
}

CELL_LABELS = {
    int(CellState.ATTACHED):      'Attached',
    int(CellState.SPREADING):     'Spreading',
    int(CellState.PROLIFERATING): 'Proliferating',
    int(CellState.BRIDGING):      'Bridging',
    int(CellState.SENESCENT):     'Senescent',
    int(CellState.MIGRATING):     'Migrating',
}

# ── energy mode colours ──────────────────────────────────────────────────
ENERGY_COLORS = {
    'traction':      '#2E7D32',   # green
    'contact':       '#1565C0',   # blue
    'friction':      '#C62828',   # red
    'osmotic':       '#6A1B9A',   # purple
    'frustration':   '#795548',   # brown
    'interfacial':   '#E65100',   # orange
}

LOCKED_BRIDGE_COLOR = '#FFD700'   # gold: locked-in bridges in the live view

_LEGACY_SPECIES = [
    {'id': 0, 'name': 'Functional', 'color': FUNC_COLOR, 'f': 1.0, 'E': None},
    {'id': 1, 'name': 'Inert', 'color': INERT_COLOR, 'f': 0.0, 'E': None},
]


def species_table(p=None, snap=None):
    """List of {'id', 'name', 'color', 'f', 'E'} for the run's species.

    Uses ``p.species`` (resolving the legacy pair when empty); without a
    Params, falls back to the two legacy classes. ``snap`` is accepted for
    API symmetry (a snapshot alone cannot name its species).
    """
    if p is not None:
        try:
            species = resolve_species(p)
        except Exception:
            species = None
        if species:
            return [{'id': k, 'name': str(sp.get('name', f'species_{k}')),
                     'color': str(sp.get('color', '#888888')), 'f': float(sp['f']),
                     'E': None if sp.get('E_kPa') is None else float(sp['E_kPa'])}
                    for k, sp in enumerate(species)]
    return [dict(s) for s in _LEGACY_SPECIES]


def species_ids(snap):
    """Per-granule species index from a snapshot (falls back to gtype)."""
    if 'species_id' in snap and snap['species_id'] is not None:
        return np.asarray(snap['species_id'], dtype=int)
    return np.asarray(snap['gtype'], dtype=int)


def hex_to_rgba(color, alpha=1.0):
    """'#RRGGBB' / '#RGB' / named colour → (r, g, b, a) floats in [0,1]."""
    c = str(color).strip()
    if c.startswith('#') and len(c) in (7, 4):
        if len(c) == 4:
            c = '#' + ''.join(ch * 2 for ch in c[1:])
        r, g, b = (int(c[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
        return (r, g, b, float(alpha))
    from matplotlib.colors import to_rgba   # named colours; does not select a backend
    r, g, b, _ = to_rgba(c)
    return (r, g, b, float(alpha))


def species_colors(snap, p=None, alpha=1.0):
    """Per-granule hex colour strings by species (legacy: red / green by gtype)."""
    table = species_table(p, snap)
    sid = species_ids(snap)
    lut = [s['color'] for s in table]
    if len(lut) <= int(sid.max(initial=0)):
        lut = lut + ['#888888'] * (int(sid.max(initial=0)) + 1 - len(lut))
    return [lut[k] for k in sid]


def granule_rgba(species_id, species, alpha=1.0):
    """(N, 4) RGBA array for granule species indices given a species table."""
    sid = np.asarray(species_id, dtype=int)
    K = max(len(species), int(sid.max(initial=0)) + 1)
    lut = np.zeros((K, 4))
    lut[:, 3] = alpha
    for k in range(K):
        color = species[k]['color'] if k < len(species) else '#888888'
        lut[k] = hex_to_rgba(color, alpha)
    return lut[sid]


def cell_rgba(cell_state, alpha=0.8):
    """(C, 4) RGBA array for cell states."""
    st = np.asarray(cell_state, dtype=int)
    lut = np.zeros((max(6, int(st.max(initial=0)) + 1), 4))
    for k in range(lut.shape[0]):
        lut[k] = hex_to_rgba(CELL_COLORS.get(k, '#AAAAAA'), alpha)
    return lut[st]


def legend_handles_species(species):
    """matplotlib Line2D handles for a species legend (imported lazily)."""
    from matplotlib.lines import Line2D
    handles = []
    for s in species:
        label = s['name'] if s.get('f') is None else f"{s['name']} (f={s['f']:.2f})"
        handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=s['color'],
                              markersize=10, linestyle='None', label=label))
    return handles


__all__ = [
    'FUNC_COLOR', 'INERT_COLOR', 'VOID_COLOR', 'CELL_COLORS', 'CELL_LABELS', 'ENERGY_COLORS',
    'LOCKED_BRIDGE_COLOR', 'species_table', 'species_ids', 'species_colors', 'granule_rgba',
    'cell_rgba', 'hex_to_rgba', 'legend_handles_species',
]
