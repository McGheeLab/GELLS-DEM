"""
GELS — Granule-Enabled Living Scaffolds
=======================================

2D/3D overdamped particle dynamics simulator for cell-driven rearrangement of
hydrogel granular scaffolds.

This package holds the live simulation engine. It was previously a pair of
loose modules at the repository root (``new_dem_0.py`` and ``lsdem.py``); they
now live here as :mod:`gels.engine` and :mod:`gels.lsdem`.

Typical use::

    from gels import Params, run
    hist, snaps, p, gs = run(Params(mode='3D', t_total=48.0))

For the step-by-step local workflow, prefer the numbered scripts in
``pipeline/`` over calling :func:`run` directly.
"""

from gels.engine import (  # noqa: F401
    Params,
    GranuleSystem,
    CellState,
    run,
    generate_packing,
    generate_packing_3d,
    generate_packing_2d_slice,
    compute_metrics,
    render_fields,
    render_fields_3d,
    load_run,
    load_params,
    load_cells,
    find_last_snapshot,
    restore_gs_from_snapshot,
    save_snapshot_to_disk,
    save_history_to_disk,
    save_params_metadata,
    create_archive,
    print_stiffness_info,
    resolve_species,
    legacy_species_from_params,
    seeding_gain,
    species_counts,
    species_view,
    render_fields_species,
    group_species_fields,
    collagen_field,
)

__version__ = "3.0.dev"

__all__ = [
    "Params", "GranuleSystem", "CellState", "run",
    "generate_packing", "generate_packing_3d", "generate_packing_2d_slice",
    "compute_metrics", "render_fields", "render_fields_3d",
    "load_run", "load_params", "load_cells", "find_last_snapshot", "restore_gs_from_snapshot",
    "save_snapshot_to_disk", "save_history_to_disk", "save_params_metadata",
    "create_archive", "print_stiffness_info",
    "resolve_species", "legacy_species_from_params", "seeding_gain",
    "species_counts", "species_view", "render_fields_species",
    "group_species_fields", "collagen_field",
]
