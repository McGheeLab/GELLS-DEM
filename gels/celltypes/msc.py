"""
Human mesenchymal stromal cell (hMSC).

**This type has NOT been through the structured literature review that
``CodeLog/References/fibroblast_parameters.md`` gives the fibroblast.** It
exists to prove the abstraction is not fibroblast-shaped and to give a starting
point for the second cell type the lab runs; most of its values are marked
`ASSUMPTION` or as order-of-magnitude ranges, and `CellType.report` will say so.
Do the review before quoting a result from it.

The differences that are well established, and which are the point of having a
second type at all:

* hMSCs spread markedly more than fibroblasts on stiff substrates
  (Engler 2006, McBeath 2004) -- 1500-3000 um^2 against 500-1500.
* They are more stiffness-sensitive: lineage follows substrate modulus
  (Engler 2006), which in this model shows up through the motor-clutch
  ``k_sub/(k_sub + g k_opt)`` term rather than through anything added here.
* They divide much more slowly -- 30-60 h and strongly passage-dependent,
  against 20-30 h.
"""

from gels.celltypes.base import ASSUMPTION, CellType, Measured as M

CELL_TYPE = CellType(
    name='msc',
    display_name='Human mesenchymal stromal cell',
    description='SECOND TYPE, NOT YET REVIEWED. Starting point, not a datum.',

    diameter_um=M(22.0, 'um', 'rounded hMSC 18-25 um [typical reported range]',
                  (18.0, 25.0)),
    spread_height_um=M(4.0, 'um', ASSUMPTION + ': thinner than a fibroblast because it '
                                  'spreads further at conserved volume', (3.0, 7.0)),
    adhesion_area_fraction=M(0.06, '-', ASSUMPTION + ': fibroblast value scaled by the '
                                        'larger footprint at similar whole-cell traction',
                             (0.05, 0.15)),

    # NOT the fibroblast's 4000 Pa, and the difference is the type's one real
    # prediction: an hMSC spreading to ~2100 um^2 while pulling the measured
    # ~120 nN implies a traction stress about 4x below a fibroblast's. hMSC TFM
    # does report lower mean tractions, but this number was SOLVED from the
    # consistency check rather than measured -- check it before quoting it.
    traction_stress_Pa=M(1050.0, 'Pa', ASSUMPTION + ': solved from sigma x A = the '
                                       'measured 120 nN; hMSC TFM means are 100-500 Pa',
                         (300.0, 3000.0)),
    total_traction_nN=M(120.0, 'nN', 'Fu 2010 microposts, hMSC ~50-200 nN total '
                                     '[typical reported range]', (50.0, 500.0)),

    n_motors=M(240, '-', ASSUMPTION + ': balanced with n_clutches, scaled to the '
                         'lower measured traction'),
    F_motor_stall_nN=M(0.5, 'nN', '~500 pN per stress fibre [typical reported range]',
                       (0.2, 1.0)),
    n_clutches=M(240, '-', ASSUMPTION + ': balanced with n_motors'),
    k_clutch_nN_per_um=M(5.0, 'nN/um', ASSUMPTION + ': fibroblast value', (1.0, 10.0)),
    k_on_clutch_per_s=M(1.0, '1/s', ASSUMPTION + ': fibroblast value (Chan & Odde 2008)'),
    k_off_clutch_per_s=M(0.1, '1/s', ASSUMPTION + ': fibroblast value (Chan & Odde 2008)'),

    k_cell_nN_per_um=M(8.0, 'nN/um', ASSUMPTION + ': hMSCs are softer than fibroblasts by '
                                     'AFM; scaled down from 10', (1.0, 50.0)),

    migration_speed_um_per_h=M(15.0, 'um/h', 'hMSC 10-25 um/h on 2D '
                                             '[typical reported range]', (10.0, 25.0)),
    contraction_speed_um_per_h=M(10.0, 'um/h', ASSUMPTION + ': fibroblast value, scaled',
                                 (6.0, 30.0)),
    sense_distance_um=M(90.0, 'um', ASSUMPTION + ': larger cell, longer reach',
                        (40.0, 150.0)),
    spread_duration_h=M(4.0, 'h', ASSUMPTION + ': slower than a fibroblast', (1.0, 6.0)),
    fa_maturation_rate_per_h=M(0.25, '1/h', ASSUMPTION),
    doubling_time_h=M(40.0, 'h', 'hMSC 30-60 h, strongly passage-dependent', (30.0, 60.0)),
    cycle_cv=M(0.25, '-', ASSUMPTION + ': broader than a fibroblast population',
               (0.1, 0.3)),
    division_min_age_h=M(12.0, 'h', ASSUMPTION, (4.0, 24.0)),

    extra_overrides=(
        'cells.bridging.force_model=hill',
        'cells.bridging.eccentric_gain=0.5',
        'cells.bridging.min_gap_um=0.0',
        'cells.bridging.break_gap_um=140.0',
        'cells.sensing.bridge_decay_length_um=35.0',
        'cells.bridging.lock_force_nN=60.0',
        'cells.bridging.expected_force_nN=120.0',
        'cells.bridging.contact_adhesion_nN_per_cell=20.0',
        'cells.division.enabled=true',
        'cells.division.model=cycle',
        'cells.division.max_layers=1.0',
        'cells.division.divide_while_bridging=true',
        'cells.seeding.capacity_coverage=1.0',
        'cells.seeding.surface_coverage=0.8',
        'cells.seeding.capacity_foothold=0.5',
    ),
    notes="""Most values here are ASSUMPTIONS carried over from the fibroblast.
Before this type is used for a published result it needs its own pass of the
review in CodeLog/References/fibroblast_parameters.md -- in particular the
traction stress, the adhesion area fraction and the clutch numbers, which are
the three the fibroblast review changed most.""",
)
