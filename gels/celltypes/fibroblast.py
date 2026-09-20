"""
Human dermal fibroblast on collagen-I-coated hydrogel granules.

The default, and the only type whose numbers have been through the structured
literature review in ``CodeLog/References/fibroblast_parameters.md``. Read that
file before changing anything here; it records not only the values but what the
evidence explicitly refuses to support.

Three of its findings are load-bearing for this type:

* **Traction stress is not density-independent.** ``sigma_FA = rho_bond * F_b *
  plog(gamma/e)`` scales with engaged-bond density, so the 4 nN/um^2 below is
  the value at SATURATING ligand and is scaled by the Langmuir gain and clutch
  engagement before use. Section 2, item 3.
* **4 nN/um^2 is the number to use if a stress is needed** -- Stricker 2011,
  NIH3T3, and the reference file says so in as many words. Valid as a
  population-average trend for growing adhesions; Balaban's 5.5 +- 2 nN/um^2 is
  the same quantity at single adhesions. Section 1.4.
* **Migration at 5 um/h is "unphysically low"** -- the V2.7 default. 12-60 um/h
  on 2D collagen; Gaudet 2003 peaks near 10 um/h at 450 ligands/um^2. Section 3.

The self-consistency check the type has to pass: ``sigma x A_adhesion`` must
reproduce the independently measured whole-cell traction. 4 nN/um^2 x 0.08 x
1257 um^2 = 402 nN against Gaudet's ~400 nN. That agreement is what sets the
adhesion area fraction, which is otherwise the weakest number here.
"""

from gels.celltypes.base import ASSUMPTION, CellType, Measured as M

CELL_TYPE = CellType(
    name='fibroblast',
    display_name='Human dermal fibroblast',
    description='The GELS default. Collagen-I-coated granules, 1-50 kPa.',

    diameter_um=M(20.0, 'um', 'rounded fibroblast 15-25 um [typical reported range]',
                  (15.0, 25.0)),
    spread_height_um=M(5.0, 'um', 'spread fibroblast 3-7 um thick [typical reported range]',
                       (3.0, 7.0)),
    # Set by the sigma x A = 400 nN consistency check against Gaudet 2003, and
    # consistent with the 5-15 % focal-adhesion area fraction usually reported.
    adhesion_area_fraction=M(0.08, '-', 'derived: Balaban 2001 5.5 nN/um^2 x A_FA = '
                                        'Gaudet 2003 400 nN/cell', (0.05, 0.15)),

    traction_stress_Pa=M(4000.0, 'Pa', 'Stricker 2011 (NIH3T3) 4 nN/um^2; Balaban 2001 '
                                       '5.5 +- 2 -- at saturating ligand', (2000.0, 7500.0)),
    total_traction_nN=M(400.0, 'nN', 'Gaudet 2003 ~0.04 dyn/cell; TFM 50-500 nN '
                                     '(Dembo & Wang 1999, Munevar 2001)', (50.0, 500.0)),

    n_motors=M(400, '-', 'balanced with n_clutches per Bangasser & Odde 2013 (n_m ~ n_c); '
                         'gives 200 nN stall'),
    F_motor_stall_nN=M(0.5, 'nN', '~500 pN per stress fibre [typical reported range]',
                       (0.2, 1.0)),
    n_clutches=M(400, '-', 'balanced with n_motors (Bangasser & Odde 2013)'),
    k_clutch_nN_per_um=M(5.0, 'nN/um', 'clutch cluster spring constant', (1.0, 10.0)),
    k_on_clutch_per_s=M(1.0, '1/s', 'Chan & Odde 2008'),
    k_off_clutch_per_s=M(0.1, '1/s', 'Chan & Odde 2008 baseline'),

    # Whole-cell series compliance. On a 10 kPa granule the substrate spring is
    # ~394 nN/um, so THIS is the soft element and it is what puts the stored
    # energy in the 0.1-10 pJ band traction force microscopy reports.
    k_cell_nN_per_um=M(10.0, 'nN/um', 'cortical E ~ 1 kPa over a 10 um scale; '
                                      'chosen so U = F^2/2k lands in the measured '
                                      '0.1-10 pJ TFM band', (1.0, 50.0)),

    migration_speed_um_per_h=M(30.0, 'um/h', '12-60 um/h on 2D collagen '
                                             '[typical reported range]; 10-30 on curved 3D',
                               (12.0, 60.0)),
    contraction_speed_um_per_h=M(12.0, 'um/h', '0.1-0.5 um/min unloaded shortening',
                                 (6.0, 30.0)),
    sense_distance_um=M(80.0, 'um', 'reach of a spread fibroblast (100-150 um long), '
                                    'not of a filopodium', (40.0, 150.0)),
    spread_duration_h=M(3.0, 'h', 'spreading complete in 1-6 h [typical reported range]',
                        (1.0, 6.0)),
    fa_maturation_rate_per_h=M(0.3, '1/h', ASSUMPTION),
    doubling_time_h=M(24.0, 'h', 'human dermal fibroblasts 20-30 h', (20.0, 30.0)),
    cycle_cv=M(0.15, '-', 'real fibroblast cycle CV 0.1-0.3', (0.1, 0.3)),
    division_min_age_h=M(8.0, 'h', 'G1 refractory [typical reported range]', (4.0, 12.0)),

    extra_overrides=(
        # The contractile element: holds its stall force once the bed stops
        # yielding. Reduces exactly to the V2.7 actuator as v0 -> inf.
        'cells.bridging.force_model=hill',
        'cells.bridging.eccentric_gain=0.5',
        'cells.bridging.min_gap_um=0.0',
        'cells.bridging.break_gap_um=120.0',
        'cells.sensing.bridge_decay_length_um=30.0',
        'cells.bridging.lock_force_nN=90.0',
        'cells.bridging.expected_force_nN=180.0',
        'cells.bridging.contact_adhesion_nN_per_cell=20.0',
        # The cycle model is what makes 24 h MEAN 24 h: under the memoryless
        # rule the 8 h refractory adds to the mean (8 + T_d/ln2 = 42.6 h).
        'cells.division.enabled=true',
        'cells.division.model=cycle',
        'cells.division.max_layers=1.0',
        'cells.division.divide_while_bridging=true',
        # Seed BELOW capacity or the population is confluent at t = 0, every
        # cell arrests in G0, and nothing ever divides.
        'cells.seeding.capacity_coverage=1.0',
        'cells.seeding.surface_coverage=0.8',
        'cells.seeding.capacity_foothold=0.5',
    ),
    notes="""Ligand response is a Langmuir in ABSOLUTE density, not in coating
fraction: g(f) = f(1+k)/(f+k). There is no published single law for traction vs
ligand density -- four independent systems support saturation in molecules/um^2
with a half-saturation of order 1e2-1e3. Do not encode a linear law or a hard
58-70 nm spacing threshold; see section 2 of the reference file.""",
)
