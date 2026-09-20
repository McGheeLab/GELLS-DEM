"""
Setup presets (V3.1).
=====================

A preset is an ordered list of dotted ``--set``-style overrides applied to a
``gels.config.Setup``. They exist because the interesting configurations are
correlated bundles — a PMMA well is a container shape, a boundary condition,
a gravity setting, a contact treatment and a packing mode that only make
sense together — and stating them one flag at a time is where mistakes live.

    python pipeline/step0_new_setup.py --preset pmma_well,fibroblast_realistic -o setup.yaml
    python pipeline/step1_config.py --setup setup.yaml --preset fibroblast_realistic --name run1

Presets are applied in the order given, before any ``--set``, so a later
preset or an explicit ``--set`` always wins. The names applied are recorded in
``meta.presets`` of the written setup, so a run directory says how it was
configured.

Provenance for the physical values is in
``CodeLog/References/fibroblast_parameters.md``; values marked there as
assumptions rather than measurements are the ones to calibrate first.
"""

from typing import Dict, List

__all__ = ['PRESETS', 'PRESET_DESCRIPTIONS', 'resolve_presets', 'apply_presets', 'preset_deltas']


# ── the PMMA well: container, material, gravity, packing ──────────────
# 2 mm-diameter model well with a 1.5 mm bed, as agreed for the calibration
# runs: the real plate well is 6.4 mm across, which at 40 um granules is
# ~1e6 granules; the scaled well keeps the bed depth and the granule size and
# treats the wall curvature as approximate.
_PMMA_CONTACT = [
    'contact.E_kPa=3.0e6',              # PMMA 2.4-3.3 GPa
    'contact.poisson_ratio=0.37',
    'contact.stiffness_cap_kPa=100',    # CONTACT stiffness only; cells still see 3 GPa
    'contact.max_overlap_frac=0.03',    # rigidity is enforced geometrically, not by force
    'contact.mc_dem.enabled=false',     # multi-contact stiffening is a soft-gel model
    'contact.friction_mu=0.3',          # wet PMMA 0.2-0.5; without it the bed is frictionless
    'contact.adhesion_energy_J_per_m2.cc=0.0005',   # hydrogel values would glue the bed to the wall
    'contact.adhesion_energy_J_per_m2.cb=0.0002',
    'contact.adhesion_energy_J_per_m2.bb=0.0001',
]

_GRAVITY = [
    'gravity.enabled=true',
    'gravity.medium_density_kg_m3=1000',
    'gravity.granule_density_kg_m3=1180',     # PMMA
    'gravity.scale=1.0',
    'packing.consolidation=auto',             # -> gravity: the bed is sedimented, not centred
    'packing.settle_steps=600',
]

# ── soft fragmented hydrogel: the primary system from V3.2 ─────────
# The lab's granules are hydrogel at 10-50 kPa, not the 3 GPa PMMA of V3.1. At
# 10 kPa a 180 nN bridge flattens a contact by 2.9 um (r = 40) to 3.6 um
# (r = 20); at 3 GPa it flattens it by 0.001 um. That difference is most of the
# compaction the model was missing, and it costs nothing to switch on.
_SOFT_CONTACT = [
    'contact.E_kPa=10.0',                  # 10-50 kPa; sweep the top of the range
    'contact.poisson_ratio=0.45',          # report it: MC-DEM's c_mc = nu/(1-2nu) is 4.5 here, 24.5 at 0.49
    'contact.stiffness_cap_kPa=0.0',       # the cap existed only to tame 3 GPa
    'contact.overlap_model=elastic',       # size the rail from the contact law, not by hand
    'contact.overlap_safety=1.5',
    'contact.semi_implicit=true',          # takes stability duty off the velocity cap
    'contact.max_overlap_frac=0.20',       # used only if overlap_model is switched back to fixed
    'contact.mc_dem.enabled=false',        # opposes compaction and stiffens the contact up to 5x
    'contact.friction_mu=0.3',             # rough fragments; tau_0*A alone is ~1e-3 nN at these areas
    # the V3.0 hydrogel adhesion values, restored: PMMA needed them lowered to
    # stop the JKR pull-off gluing the bed to the wall, a soft gel does not.
    'contact.adhesion_energy_J_per_m2.cc=0.002',
    'contact.adhesion_energy_J_per_m2.cb=0.001',
    'contact.adhesion_energy_J_per_m2.bb=0.0005',
]

_SOFT_GRAVITY = [
    'gravity.enabled=true',
    'gravity.medium_density_kg_m3=1000',
    'gravity.granule_density_kg_m3=1050',  # hydrated fragmented gel 1030-1100
    'gravity.scale=1.0',
    'packing.consolidation=auto',
    'packing.settle_steps=600',
]


PRESETS: Dict[str, List[str]] = {

    'pmma_well': [
        'domain.mode=3D',
        'domain.size_um[0]=2000', 'domain.size_um[1]=2000', 'domain.size_um[2]=2100',
        'boundary.mode=walls',
        'boundary.shape=cylinder',
        'boundary.top=free',
        'granules.bed_height_um=1500',        # sets the granule amount; overrides solid_fraction
        'granules.bed_solid_fraction=0.60',
        'output.Ngrid_3d=256',
    ] + _PMMA_CONTACT + _GRAVITY,

    'pmma_dish_slice': [
        'domain.mode=2D',
        'domain.size_um[0]=2000', 'domain.size_um[1]=2100', 'domain.size_um[2]=800',
        'boundary.mode=walls',
        'boundary.shape=box',                 # 2D: floor at y=0, two side walls, free top
        'boundary.top=free',
        'granules.bed_height_um=1500',
        'granules.bed_solid_fraction=0.80',   # 2D random loose packing of discs
        'output.Ngrid=400',
    ] + _PMMA_CONTACT + _GRAVITY,

    'soft_gel_well': [
        'domain.mode=3D',
        'domain.size_um[0]=2000', 'domain.size_um[1]=2000', 'domain.size_um[2]=2100',
        'boundary.mode=walls',
        'boundary.shape=cylinder',
        'boundary.top=free',
        'granules.bed_height_um=1500',
        # This is the answer, not a guess: the packer places granules at this
        # density and inflates them there, so the bed STARTS here. 3D spheres
        # are 0.55-0.60 at random loose packing and 0.64 at close packing;
        # rough irregular fragments settle looser still, and a loose start is
        # half the compaction budget. A bed that starts near close packing has
        # nowhere to go.
        'granules.bed_solid_fraction=0.52',
        'dynamics.v_max_um_per_h=30.0',
        'output.Ngrid_3d=256',
    ] + _SOFT_CONTACT + _SOFT_GRAVITY,

    'soft_gel_dish_slice': [
        'domain.mode=2D',
        'domain.size_um[0]=2000', 'domain.size_um[1]=2100', 'domain.size_um[2]=800',
        'boundary.mode=walls',
        'boundary.shape=box',
        'boundary.top=free',
        'granules.bed_height_um=1500',
        # as above: 2D discs are ~0.78 at random loose and ~0.84 at close
        # packing, so 0.70 leaves real headroom for the cells to take up
        'granules.bed_solid_fraction=0.70',
        'dynamics.v_max_um_per_h=30.0',
        'output.Ngrid=400',
    ] + _SOFT_CONTACT + _SOFT_GRAVITY,

    'functionalized_wall': [
        'boundary.functionalization=1.0',     # cells grip the container as they grip a granule
        'boundary.layer.enabled=true',        # lining of immobile granules: bridges can attach
        'boundary.layer.embed_frac=0.5',
        'boundary.layer.seed_cells=true',
    ],

    'inert_wall': [
        'boundary.functionalization=0.0',
        'boundary.layer.enabled=false',
    ],

    'fibroblast_realistic': [
        # contractile element: holds its stall force once the bed stops yielding
        'cells.bridging.force_model=hill',
        'cells.bridging.contraction_speed_um_per_h=12.0',
        'cells.bridging.eccentric_gain=0.5',
        'cells.bridging.min_gap_um=0.0',
        # reach of a spread fibroblast (100-150 um long), not of a filopodium
        'cells.sensing.sense_distance_um=80.0',
        'cells.bridging.break_gap_um=120.0',
        'cells.sensing.bridge_decay_length_um=30.0',
        # traction on a rigid substrate: beta = k_sub/(k_sub+k_opt) saturates at 0.909,
        # so 400 x 0.5 nN stall gives ~182 nN per bridge (TFM 50-500 nN per cell)
        'cells.motor_clutch.n_motors=400',
        'cells.motor_clutch.F_max_per_cell_nN=200.0',
        'cells.bridging.lock_force_nN=90.0',        # ~half the matured force
        'cells.bridging.expected_force_nN=180.0',
        # cells bridging across a contact also stop the granules sliding at it
        'cells.bridging.contact_adhesion_nN_per_cell=20.0',
        # division: human dermal fibroblasts double in 20-30 h.
        # The cycle model is what makes 24 h MEAN 24 h: under the memoryless rule
        # the 8 h refractory adds to the mean cycle (8 + T_d/ln2 = 42.6 h), so the
        # measured population growth over 48 h is 4.2x instead of 4.0x -> 2^1.1.
        'cells.division.enabled=true',
        'cells.division.model=cycle',
        'cells.division.cycle_time_cv=0.15',     # real fibroblast cycle CV is 0.1-0.3, not 1.0
        'cells.division.doubling_time_h=24.0',
        'cells.division.min_age_h=8.0',
        'cells.division.max_layers=1.0',
        'cells.division.divide_while_bridging=true',
        # Capacity is a full monolayer; seed BELOW it or the population is confluent
        # at t = 0, every cell arrests in G0 and nothing ever divides. Setting
        # capacity_coverage alone (V3.1) left seeding at the template's 1.0.
        'cells.seeding.capacity_coverage=1.0',
        'cells.seeding.surface_coverage=0.8',
        # cells ball up to whatever contact area they can grip, and a cell
        # spanning two granules only needs a foothold on each -- so capacity
        # is set below the flat spread footprint. Without this a 40 um granule
        # holds exactly ONE cell and the bed is confluent at seeding, so
        # nothing ever divides.
        # V3.3: was 0.25, which is EXACTLY the floor (A_rounded/A_spread = h/d
        # = 0.25 by volume conservation) -- the densest value the parameter can
        # express, and identical to anything below it. Combined with the
        # missing packing-efficiency factor it gave 64 cells on an R = 40 um
        # granule, above the hard ceiling of 58 rigid 20 um discs and well
        # above a confluent fibroblast monolayer (20-40 at 500-1000 um^2/cell).
        # 0.5 with PACKING_EFFICIENCY gives 29 at R = 40 and 7 at R = 20, both
        # inside the confluent band.
        'cells.seeding.capacity_foothold=0.5',
        # integration: v_max must not be the thing that limits contraction
        'dynamics.v_max_um_per_h=30.0',
    ],

    # ── V3.6: a bed seeded ABOVE the monolayer, so layer 1 exists at t = 0 ──
    # `fibroblast_realistic` alone never overflows: it seeds at 0.8 of a
    # capacity of 1.0. Without a second storey the stacking substitution has
    # nothing to act on and the feature is untestable end to end.
    'stacked_monolayer': [
        'cells.stacking.enabled=true',
        'cells.stacking.substrate_E_kPa=1.0',     # a cell is ~1 kPa to another cell
        'cells.stacking.substrate_poisson=0.5',
        'cells.stacking.f_cell_cell=0.15',        # ASSUMPTION -- calibrate first
        'cells.seeding.surface_coverage=1.8',     # ~1.8 storeys at t = 0
        'cells.seeding.capacity_coverage=1.0',    # capacity is still ONE monolayer
        'cells.seeding.stacking_max=3.0',         # tolerate three before senescence
    ],

    'fragmented_granules': [
        # Irregular cuboidal fragments rather than spheres. The two packing
        # flags belong together: the corrected bounding radius is LARGER, so on
        # its own it lowers the jamming ceiling and makes packing worse.
        'packing.shape_enabled=true',
        'packing.shape_contact=true',
        'packing.shape_margin=0.0',        # measured blunt: costs convergence, buys nothing
        'contact.shape_dynamics=true',     # without it the projection pulls the packed bed apart
        'contact.curvature_R_cap=2.0',     # a flat face has an ~1e15 um curvature radius
        'packing.settle_steps=600',
        # The per-granule aspect ratio and blockiness live on each SPECIES, because
        # the species count varies; set them alongside this preset, e.g.
        #   --set granules.species[0].aspect_ratio.mean=1.8
        #   --set granules.species[0].aspect_ratio.std=0.4
        #   --set granules.species[0].blockiness.mean=3.5
        #   --set granules.species[0].blockiness.std=1.0
    ],

    'asynchronous_cells': [
        # A seeded population is not in phase. Without this every cell sits at
        # cycle phase 0, clears the refractory period on the same step and the
        # whole bed divides at once -- measured: 160 divisions in a single step
        # at t = 23.5 h, versus 2-10 per step spread through the run.
        'cells.division.model=cycle',
        'cells.division.cycle_time_cv=0.15',
        # attachment / spreading / FA maturation are functions of global time, so
        # the bed matures in lockstep without a per-granule offset
        'cells.seeding.clock_jitter_h=2.0',
    ],

    'hydrogel_box_legacy': [
        # the V3.0 defaults, stated explicitly: a closed box, no gravity, soft granules,
        # constant-force bridges and a fixed cell population.
        'boundary.shape=box', 'boundary.top=wall', 'boundary.functionalization=0.0',
        'boundary.layer.enabled=false',
        'gravity.enabled=false',
        'packing.consolidation=centre',
        'granules.bed_height_um=0.0',
        'contact.E_kPa=10.0', 'contact.poisson_ratio=0.45',
        'contact.stiffness_cap_kPa=0.0', 'contact.friction_mu=0.0',
        'contact.mc_dem.enabled=true', 'contact.max_overlap_frac=0.15',
        'cells.bridging.force_model=constant',
        'cells.division.enabled=false',
        'cells.division.model=poisson', 'cells.division.cycle_time_cv=0.0',
        'cells.seeding.clock_jitter_h=0.0',
        'cells.seeding.capacity_coverage=0.0',
        'cells.sensing.sense_distance_um=40.0',
        'cells.bridging.break_gap_um=60.0',
        'cells.sensing.bridge_decay_length_um=10.0',
        'cells.motor_clutch.n_motors=200', 'cells.motor_clutch.F_max_per_cell_nN=150.0',
        'dynamics.v_max_um_per_h=20.0',
    ],
}

PRESET_DESCRIPTIONS: Dict[str, str] = {
    'pmma_well': 'rigid-bead CONTROL: PMMA in a 2 mm model well (cylinder, free top, gravity)',
    'pmma_dish_slice': 'the rigid-bead control as a fast 2D dish slice',
    'soft_gel_well': 'soft 10 kPa hydrogel in a 2 mm model well -- the primary system (V3.2)',
    'soft_gel_dish_slice': 'the soft-gel well as a fast 2D dish slice (floor, two walls, free top)',
    'functionalized_wall': 'collagen-coated container with a lining cells can bridge to',
    'inert_wall': 'bare container (the default): the bed can detach from it',
    'fibroblast_realistic': 'Hill contraction, 80 um reach, rigid-substrate stall force, division (24 h)',
    'asynchronous_cells': 'cells spread through the cycle at seeding, with per-cell cycle lengths',
    'stacked_monolayer': 'seed 1.8 storeys so cells stand on cells; layer >= 1 anchors to a cell (V3.6)',
    'fragmented_granules': 'irregular cuboidal fragments: shape-aware packing and contact (V3.2)',
    'hydrogel_box_legacy': 'the V3.0 defaults (closed box, no gravity, 10 kPa hydrogel, no division)',
}


def resolve_presets(names) -> List[str]:
    """Flatten preset names ('a,b' or ['a', 'b']) into one ordered override list."""
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    flat = []
    for item in names:
        flat.extend(part.strip() for part in str(item).split(',') if part.strip())
    unknown = [n for n in flat if n not in PRESETS]
    if unknown:
        raise ValueError(f"unknown preset(s) {unknown}; known: {sorted(PRESETS)}")
    out = []
    for n in flat:
        out.extend(PRESETS[n])
    return out


def preset_names(names) -> List[str]:
    """The preset names in the order they are applied."""
    if names is None:
        return []
    if isinstance(names, str):
        names = [names]
    flat = []
    for item in names:
        flat.extend(part.strip() for part in str(item).split(',') if part.strip())
    return flat


def apply_presets(setup, names):
    """Apply the named presets to a Setup in order and record them in meta.presets."""
    from gels.config import apply_overrides
    ov = resolve_presets(names)
    if not ov:
        return setup
    apply_overrides(setup, ov)
    applied = preset_names(names)
    existing = list(getattr(setup.meta, 'presets', []) or [])
    setup.meta.presets = existing + [n for n in applied if n not in existing]
    return setup


def preset_deltas(base, names):
    """(preset, path, old, new) for every value a preset changes, for a file header."""
    from gels.config import get_path
    import copy as _copy
    out = []
    work = _copy.deepcopy(base)
    from gels.config import apply_overrides
    for name in preset_names(names):
        for item in PRESETS[name]:
            path, _, raw = item.partition('=')
            path = path.strip()
            old = get_path(work, path)
            apply_overrides(work, [item])
            new = get_path(work, path)
            if old != new:
                out.append((name, path, old, new))
    return out
