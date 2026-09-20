"""
Cell types you can instantiate (V3.6).
======================================

A cell type is the set of measured properties that distinguish one cell from
another, held with its provenance. Add one by dropping a module in this package
that defines a module-level ``CELL_TYPE``; nothing else needs to change.

    from gels.celltypes import get, available

    ct = get('fibroblast')
    print(ct.report())              # every value with its source
    ct.to_overrides()               # dotted overrides, same currency as a preset

    python pipeline/step0_new_setup.py --cell-type fibroblast -o setup.yaml
    python pipeline/step1_config.py --setup setup.yaml --cell-type msc --name run1

**Only `fibroblast` has been through the literature review** in
``CodeLog/References/fibroblast_parameters.md``. `msc` exists to keep the
abstraction honest -- an interface with one implementation is a guess -- and
says so in its own report.

A cell type is applied AFTER any ``--preset`` and BEFORE any ``--set``, so a
preset states the scaffold, the cell type states the cell, and an explicit
``--set`` always wins.
"""

import importlib
import os
import pkgutil

from gels.celltypes.base import ASSUMPTION, CellType, Measured

__all__ = ['CellType', 'Measured', 'ASSUMPTION', 'get', 'available', 'register',
           'overrides_for', 'DEFAULT_TYPE']

DEFAULT_TYPE = 'fibroblast'

_REGISTRY = {}
_SCANNED = False


def register(cell_type):
    """Add a type to the registry (the discovery path calls this for you)."""
    if not isinstance(cell_type, CellType):
        raise TypeError(f'not a CellType: {cell_type!r}')
    _REGISTRY[cell_type.name] = cell_type
    return cell_type


def _scan():
    global _SCANNED
    if _SCANNED:
        return
    _SCANNED = True
    here = os.path.dirname(os.path.abspath(__file__))
    for mod in pkgutil.iter_modules([here]):
        if mod.name.startswith('_') or mod.name == 'base':
            continue
        m = importlib.import_module(f'{__name__}.{mod.name}')
        ct = getattr(m, 'CELL_TYPE', None)
        if isinstance(ct, CellType):
            register(ct)


def available():
    """``{name: display_name}`` for every type in the folder, sorted."""
    _scan()
    return {k: _REGISTRY[k].display_name for k in sorted(_REGISTRY)}


def get(name):
    """The `CellType` called ``name``, or a KeyError naming what there is."""
    _scan()
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown cell type {name!r}; available: "
                       + ', '.join(sorted(_REGISTRY)))
    return _REGISTRY[key]


def overrides_for(name):
    """Dotted overrides for ``name``, ready for `gels.presets.apply_presets`."""
    return get(name).to_overrides()


def apply_cell_type(setup, name):
    """Apply a cell type's overrides to a `gels.config.Setup`, in place.

    Mirrors `gels.presets.apply_presets`, including recording what was applied,
    so a run directory says which cell it modelled and a reader does not have to
    reverse-engineer it from twenty numbers.
    """
    from gels.config import apply_overrides
    ct = get(name)
    apply_overrides(setup, ct.to_overrides())
    setup.meta.cell_type = ct.name
    return setup
