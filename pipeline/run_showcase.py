"""
GELS V3.0 showcase — several conditions, run in parallel, compared on one set of figures.

Nine runs demonstrate what the V3.0 engine adds: granule species with a degree of
collagen-I functionalization f in [0, 1] (a six-run ladder on one 5 x 5 mm domain),
shaped granules, a 20 x 20 mm two-dimensional bed of ~50 000 granules and a 1.6 mm
three-dimensional cube — all driven through the numbered pipeline steps and then
drawn together by viz2/compare_runs.py.

    python pipeline/run_showcase.py                         # everything -> results/showcase_v3/
    python pipeline/run_showcase.py --list                  # the condition table (with estimated N)
    python pipeline/run_showcase.py --only 01_binary 09_3d  # a subset
    python pipeline/run_showcase.py --compare-only          # redraw the comparison figures only
    python pipeline/run_showcase.py --write-setups          # (re)write Trials/showcase_v3/*.yaml and stop
    python pipeline/run_showcase.py --t_total 24 --budget 20  # shorter runs, fewer cores

Each condition is a commented setup file in Trials/showcase_v3/ (regenerated here, editable
by hand) and one run directory under the output folder with the usual layout
(params.json, setup.yaml, snapshots/, history.json, viz2_plots/, analysis/) plus
showcase_log.txt (the pipeline output) and showcase.json (label and timings). The
comparison lands in <out>/comparison/.

Conditions run concurrently as separate processes; each gets its own numba thread
count and the scheduler keeps the sum of running threads within --budget (default:
physical cores). The budget covers the physics (steps 1-3) only; post-processing then
runs outside it with its own pool of `viz_workers` plotting processes, so the next
condition's physics can start while earlier runs are still drawing.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors='replace')
    except (AttributeError, ValueError):
        pass

DEFAULT_OUT = os.path.join('results', 'showcase_v3')
DEFAULT_SETUPS = os.path.join('Trials', 'showcase_v3')

# One colour per functionalization level, shared by every condition so the same f
# is the same colour on every panel of the comparison figures.
F_COLORS = {
    1.00: '#CC2222',   # red     fully collagen-coated
    0.75: '#E8772E',   # orange
    0.50: '#2255CC',   # blue    half coated
    0.25: '#8E44AD',   # purple
    0.20: '#17A589',   # teal    low dose (Langmuir regime: g = 0.59, seeding ~80 %)
    0.00: '#22AA22',   # green   bare hydrogel
}


def species(name, f, vf, radius=(40.0, 5.0, 15.0), aspect=(1.0, 0.0), block=(2.0, 0.0),
            aspect_c=(1.0, 0.0), block_n2=(2.0, 0.0), color=None, E_kPa=None):
    """A species record for the condition table (plain dict, turned into a Setup later)."""
    return dict(name=name, f=float(f), vf=float(vf), radius=tuple(radius), aspect=tuple(aspect),
                block=tuple(block), aspect_c=tuple(aspect_c), block_n2=tuple(block_n2),
                color=color or F_COLORS[round(float(f), 2)], E_kPa=E_kPa)


@dataclass
class Condition:
    key: str
    label: str
    description: str
    species: List[dict]
    mode: str = '2D'
    size_um: Tuple[float, float, float] = (5000.0, 5000.0, 800.0)
    solid_fraction: float = 0.65
    shape_enabled: bool = False
    threads: int = 4
    viz_only: Optional[Tuple[str, ...]] = None     # step 4 --only (None = full viz2 suite)
    viz_skip: Tuple[str, ...] = ()                 # step 4 --skip
    analysis: bool = True
    viz_workers: int = 4          # step 4 plotting processes for this condition
    save_every_h: float = 2.0
    Ngrid: Optional[int] = None                    # None = template value
    Ngrid_3d: Optional[int] = None                 # None = template value
    cell_surface_coverage: Optional[float] = None  # None = template value (1.0)
    factors: Optional[dict] = None                 # sweep coordinates, recorded in showcase.json
    # ── V3.1 ──
    presets: Tuple[str, ...] = ()                  # gels.presets names, applied before the explicit fields
    setup_overrides: Tuple[str, ...] = ()          # dotted --set overrides, applied last
    boundary_shape: str = 'box'                    # box | cylinder (3D)
    boundary_top: str = 'wall'                     # wall | free
    bed_height_um: Optional[float] = None          # granule amount as a settled bed height

    def base_area(self):
        """Footprint of the container (um^2; a length in 2D)."""
        Lx, Ly, _Lz = self.size_um
        if self.boundary_shape == 'cylinder' and self.mode == '3D':
            return math.pi * (0.5 * min(Lx, Ly)) ** 2
        return Lx * Ly if self.mode == '3D' else Lx

    def solid_volume(self):
        """Total granule volume (um^3; area in 2D) the condition asks for."""
        Lx, Ly, Lz = self.size_um
        if self.bed_height_um:
            phi_bed = 0.60 if self.mode == '3D' else 0.80
            return phi_bed * self.base_area() * self.bed_height_um
        V = (math.pi * (0.5 * min(Lx, Ly)) ** 2 * Lz
             if (self.boundary_shape == 'cylinder' and self.mode == '3D')
             else (Lx * Ly * Lz if self.mode == '3D' else Lx * Ly))
        return self.solid_fraction * V

    def estimated_counts(self):
        """Expected granules per species from the mean-volume count rule."""
        V_solid = self.solid_volume()
        out = []
        for sp in self.species:
            mu, sd, _ = sp['radius']
            if self.mode == '3D':
                mean_vol = (4.0 / 3.0) * math.pi * (mu**3 + 3 * mu * sd**2)
            else:
                mean_vol = math.pi * (mu**2 + sd**2)
            out.append(int(round(V_solid * sp['vf'] / mean_vol)))
        return out


LADDER_DOMAIN = (5000.0, 5000.0, 800.0)      # ~3 200 granules of 40 um at solid fraction 0.65

CONDITIONS = [
    Condition(
        key='01_binary', label='Binary  f = 1 / 0  (60 / 40)',
        description='The V2.7 picture: fully coated granules (60 % of the solid) mixed with bare ones.',
        species=[species('coated', 1.0, 0.60), species('bare', 0.0, 0.40)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='02_three_species', label='Three species  f = 1 / 0.5 / 0  (20 / 40 / 40)',
        description='Fully coated, half coated and bare granules: the half-coated species keeps ~94 % '
                    'of its cells but they pull at ~86 % force (Langmuir g(0.5)).',
        species=[species('coated', 1.0, 0.20), species('half', 0.5, 0.40), species('bare', 0.0, 0.40)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='03_ladder', label='Five-step ladder  f = 1, 0.75, 0.5, 0.25, 0  (20 % each)',
        description='A continuous functionalization ladder; adhesion, friction, seeding and traction '
                    'all interpolate with f.',
        species=[species('f100', 1.0, 0.20), species('f075', 0.75, 0.20), species('f050', 0.5, 0.20),
                 species('f025', 0.25, 0.20), species('bare', 0.0, 0.20)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='04_low_dose', label='Low dose  f = 0.2 / 0  (60 / 40)',
        description='Sparse coating in the Langmuir regime: cells still attach (g_attach = 0.80) but '
                    'grip weakly (g = 0.59) — the regime the binary model cannot express.',
        species=[species('sparse', 0.2, 0.60), species('bare', 0.0, 0.40)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='05_uniform_half', label='Uniform  f = 0.5',
        description='Every granule half coated: no bare phase, reduced traction everywhere.',
        species=[species('half', 0.5, 1.0)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='06_uniform_full', label='Uniform  f = 1',
        description='Every granule fully coated: the maximum-traction control.',
        species=[species('coated', 1.0, 1.0)],
        size_um=LADDER_DOMAIN),
    Condition(
        key='07_shapes', label='Shaped granules  (blocky coated / elongated half / round bare)',
        description='Superellipse granules per species: rounded squares (n = 4) fully coated, '
                    'ellipses (a/b = 1.6) half coated, circles bare.',
        species=[species('blocky_coated', 1.0, 0.30, block=(4.0, 0.3)),
                 species('elongated_half', 0.5, 0.35, aspect=(1.6, 0.15)),
                 species('round_bare', 0.0, 0.35)],
        size_um=LADDER_DOMAIN, shape_enabled=True, threads=4),
    Condition(
        key='08_large_2d', label='Large 2D  20 × 20 mm  (~50 000 granules)',
        description='The three-species mix on a 20 mm square — a 12-well footprint — with ~50 000 '
                    'granules and >10^5 cells.',
        species=[species('coated', 1.0, 0.20), species('half', 0.5, 0.40), species('bare', 0.0, 0.40)],
        size_um=(20000.0, 20000.0, 800.0), threads=16, viz_workers=6,
        viz_only=('scaffold', 'phases', 'percolation')),
    Condition(
        key='09_3d', label='3D  1.6 mm cube  (~8 000 spheres)',
        description='The three-species mix as a volumetric bed of spheres (solid fraction 0.55); '
                    'drawn as its z-midplane slice.',
        species=[species('coated', 1.0, 0.20), species('half', 0.5, 0.40), species('bare', 0.0, 0.40)],
        mode='3D', size_um=(1600.0, 1600.0, 1600.0), solid_fraction=0.55, threads=8,
        viz_workers=6, viz_skip=('scaffold_3d',)),
]


# ══════════════════════════════════════════════════════════════════════
# Sweep: 3D granule size x composition (2 mm cube)
# ══════════════════════════════════════════════════════════════════════
#
# Three factors, full factorial:
#   functional granule size   mean diameter 40 / 80 / 150 um  (f = 1, fully collagen-coated)
#   inert granule size        mean diameter 40 / 80 / 150 um  (f = 0, bare)
#   functional volume share   0.50 / 0.75 / 1.00 of the solid
#
# At 100 % functional there is no inert phase, so the inert size stops mattering and the
# nine nominal cells collapse to three: 3 x 3 x 2 + 3 = 21 distinct conditions.
#
# Cells are seeded to 0.8 of each granule's surface area (Langmuir seeding gain g(f) = 1 at
# f = 1, so functional granules carry the full 0.8 coverage and bare granules carry none):
# 3 cells on a 40 um granule, 13 on an 80 um one, 45 on a 150 um one.
#
# Solid fraction 0.55: random close packing of spheres is ~0.64 and the V3.0 3D benchmark
# showed 0.62 leaves residual overlaps above tolerance, so 0.55 packs cleanly for every
# size combination — including the three monodisperse (100 % functional) cells — and keeps
# the conditions comparable.
#
# Ngrid_3d = 256 (7.8 um voxels) rather than the 200 default: at 40 um granules that is
# still only ~2.6 voxels per radius, so grid-derived quantities (porosity, connectivity,
# percolation) are coarse for the smallest size. The analytic `*_true` metrics — solid
# fraction, contacts, coordination number — are computed from radii and are unaffected.

SWEEP3D_DOMAIN = (2000.0, 2000.0, 2000.0)     # 2 mm cube
SWEEP3D_PHI = 0.55                             # total solid volume fraction
SWEEP3D_COVERAGE = 0.8                         # cells per granule surface area at t = 0
SWEEP3D_SAVE_EVERY_H = 4.0                     # 19 snapshots over 72 h
SIZE_LEVELS = {'small': 40.0, 'medium': 80.0, 'large': 150.0}    # mean DIAMETER, um
RATIO_LEVELS = (0.50, 0.75, 1.00)              # functional share of the solid volume


def _mean_granule_volume(diameter_um, rel_std=0.10):
    """E[V] of a normal radius distribution: (4/3)pi (mu^3 + 3 mu sigma^2)."""
    mu = diameter_um / 2.0
    sd = rel_std * mu
    return (4.0 / 3.0) * math.pi * (mu ** 3 + 3.0 * mu * sd ** 2)


def _sweep3d_species(diameter_um, f, volume_fraction, name):
    mu = diameter_um / 2.0
    return species(name, f, volume_fraction,
                   radius=(mu, 0.10 * mu, 0.50 * mu),
                   color=F_COLORS[1.00] if f >= 0.5 else F_COLORS[0.00])


def build_sweep3d():
    """The 21 distinct (functional size, inert size, functional share) conditions."""
    conds = []
    for ratio in RATIO_LEVELS:
        for fkey, fdia in SIZE_LEVELS.items():
            inert_levels = [(None, None)] if ratio >= 1.0 else list(SIZE_LEVELS.items())
            for ikey, idia in inert_levels:
                sp = [_sweep3d_species(fdia, 1.0, ratio, f'functional_{int(fdia)}um')]
                if ikey is not None:
                    sp.append(_sweep3d_species(idia, 0.0, 1.0 - ratio, f'inert_{int(idia)}um'))
                n_est = round(SWEEP3D_PHI * math.prod(SWEEP3D_DOMAIN) * ratio
                              / _mean_granule_volume(fdia))
                if ikey is not None:
                    n_est += round(SWEEP3D_PHI * math.prod(SWEEP3D_DOMAIN) * (1.0 - ratio)
                                   / _mean_granule_volume(idia))
                # heavier runs get more compute threads and more plotting workers
                threads = 12 if n_est > 60000 else (8 if n_est > 15000 else 4)
                viz_workers = 6 if n_est > 15000 else 4
                # the Voronoi / movie modules redraw every granule per frame: only
                # affordable for the coarse packings
                viz_only = ('scaffold', 'phases') if n_est > 20000 else \
                           ('scaffold', 'phases', 'movies')
                if ikey is None:
                    key = f'func{int(fdia)}_r{int(ratio * 100)}'
                    label = f'{int(fdia)} um functional only  (100 %)'
                    desc = (f'Monodisperse bed of fully collagen-coated {int(fdia)} um granules; '
                            f'the no-inert reference for this size.')
                else:
                    key = f'func{int(fdia)}_inert{int(idia)}_r{int(ratio * 100)}'
                    label = f'F {int(fdia)} um / I {int(idia)} um  ({int(ratio * 100)} % func)'
                    desc = (f'{int(ratio * 100)} % of the solid is {int(fdia)} um fully coated '
                            f'granules, the rest {int(idia)} um bare granules.')
                conds.append(Condition(
                    key=key, label=label, description=desc, species=sp,
                    mode='3D', size_um=SWEEP3D_DOMAIN, solid_fraction=SWEEP3D_PHI,
                    threads=threads, viz_workers=viz_workers, viz_only=viz_only,
                    save_every_h=SWEEP3D_SAVE_EVERY_H, Ngrid_3d=256,
                    cell_surface_coverage=SWEEP3D_COVERAGE,
                    factors=dict(func_diameter_um=fdia, inert_diameter_um=idia,
                                 func_size=fkey, inert_size=ikey or 'none',
                                 func_ratio=ratio, n_estimated=n_est)))
    return conds




# ══════════════════════════════════════════════════════════════════════
# V3.1 well calibration sweep
# ══════════════════════════════════════════════════════════════════════
# A scaled model of the lab's experiment: PMMA granules sedimented under
# gravity in a well with a flat floor, a cylindrical side wall and a free top,
# carrying fibroblasts that contract, reach ~80 um and divide every 24 h.
#
# The set is built to separate the four things that were missing or wrong in
# V3.0, one factor at a time, cheaply enough to iterate:
#   - the container (a closed box cannot compact at all),
#   - the force model (constant actuator vs contractile element),
#   - cell division,
#   - the boundary chemistry (an inert wall lets the bed detach; a
#     functionalized one pins it and the functional phase coarsens instead).
#
# The 2D dish slices are ~2 000 granules and run in about a minute each; the
# 3D well at 80 um granules is ~10 000 and takes a few minutes. 40 um granules
# in the same well is ~80 000 (--granule-um-3d 40), which costs roughly
# 20 min of packing plus 10 min of simulation per run.

WELL_DIAMETER_UM = 2000.0        # scaled model well (a 96-well plate is 6.4 mm)
WELL_HEIGHT_UM = 2100.0          # container height = grid extent; ~1.4 x the bed
WELL_BED_UM = 1500.0             # settled bed height -> sets the granule amount
WELL_GRANULE_UM = 40.0           # 2D dish slices
WELL_GRANULE_UM_3D = 80.0        # 3D well (40 um is ~80 000 granules)
WELL_COVERAGE = 0.8              # granules arrive pre-coated with cells
WELL_SAVE_EVERY_H = 2.0


def _well_species(diameter_um, f=1.0, vf=1.0, name=None):
    mu = diameter_um / 2.0
    return species(name or f'pmma_{int(diameter_um)}um', f, vf,
                   radius=(mu, 0.10 * mu, 0.50 * mu),
                   color=F_COLORS[1.00] if f >= 0.5 else F_COLORS[0.00])


def build_well(granule_um=WELL_GRANULE_UM, granule_um_3d=WELL_GRANULE_UM_3D):
    """The well calibration set: force model x division x wall chemistry, 2D and 3D."""
    dish = dict(mode='2D', size_um=(WELL_DIAMETER_UM, WELL_HEIGHT_UM, 800.0),
                boundary_shape='box', boundary_top='free', bed_height_um=WELL_BED_UM,
                cell_surface_coverage=WELL_COVERAGE, save_every_h=WELL_SAVE_EVERY_H,
                analysis=False, threads=4, viz_workers=4,
                viz_only=('scaffold', 'phases', 'percolation'))
    well3d = dict(mode='3D', size_um=(WELL_DIAMETER_UM, WELL_DIAMETER_UM, WELL_HEIGHT_UM),
                  boundary_shape='cylinder', boundary_top='free', bed_height_um=WELL_BED_UM,
                  cell_surface_coverage=WELL_COVERAGE, save_every_h=WELL_SAVE_EVERY_H,
                  analysis=False, threads=8, viz_workers=4, Ngrid_3d=256,
                  viz_only=('scaffold', 'phases'))
    one = [_well_species(granule_um)]
    one3d = [_well_species(granule_um_3d)]
    mix = [_well_species(granule_um, 1.0, 0.5, 'coated'),
           _well_species(granule_um, 0.0, 0.5, 'bare')]

    def F(**kw):
        base = dict(granule_um=granule_um, bed_height_um=WELL_BED_UM)
        base.update(kw)
        return base

    conds = [
        Condition(
            key='dish_const_nodiv', label='2D dish - constant force, no division',
            description='Baseline: the V3.0 cell model in the new container. Any compaction here '
                        'is rearrangement under a fixed pull.',
            species=one, presets=('pmma_dish_slice', 'inert_wall'),
            setup_overrides=('cells.bridging.force_model=constant', 'cells.division.enabled=false'),
            factors=F(force_model='constant', division=False, wall='inert', geometry='dish'), **dish),
        Condition(
            key='dish_hill_nodiv', label='2D dish - contractile, no division',
            description='Adds the Hill force-velocity element: cells hold their stall force once '
                        'the bed stops yielding, and approach at v0 rather than at the velocity cap.',
            species=one, presets=('pmma_dish_slice', 'fibroblast_realistic', 'inert_wall'),
            setup_overrides=('cells.division.enabled=false',),
            factors=F(force_model='hill', division=False, wall='inert', geometry='dish'), **dish),
        Condition(
            key='dish_const_div', label='2D dish - constant force, division',
            description='Adds cell division alone, so its effect separates from the force model.',
            species=one, presets=('pmma_dish_slice', 'fibroblast_realistic', 'inert_wall'),
            setup_overrides=('cells.bridging.force_model=constant',),
            factors=F(force_model='constant', division=True, wall='inert', geometry='dish'), **dish),
        Condition(
            key='dish_hill_div', label='2D dish - contractile + division (full cell model)',
            description='The full V3.1 cell model against an inert wall: the reference 2D condition.',
            species=one, presets=('pmma_dish_slice', 'fibroblast_realistic', 'inert_wall'),
            factors=F(force_model='hill', division=True, wall='inert', geometry='dish'), **dish),
        Condition(
            key='dish_hill_div_fwall', label='2D dish - full model, functionalized wall',
            description='The same run against a collagen-coated wall lined with immobile granules. '
                        'Expect the bed to stay pinned instead of detaching.',
            species=one, presets=('pmma_dish_slice', 'fibroblast_realistic', 'functionalized_wall'),
            factors=F(force_model='hill', division=True, wall='functionalized', geometry='dish'), **dish),
        Condition(
            key='dish_mix50_inert', label='2D dish - 50 % coated / 50 % bare, inert wall',
            description='A two-phase bed against an inert wall: void closure plus detachment.',
            species=mix, presets=('pmma_dish_slice', 'fibroblast_realistic', 'inert_wall'),
            factors=F(force_model='hill', division=True, wall='inert', geometry='dish',
                      func_ratio=0.5), **dish),
        Condition(
            key='dish_mix50_fwall', label='2D dish - 50 % coated / 50 % bare, functionalized wall',
            description='The coarsening regime: with the wall coated there is no global compaction, '
                        'and the functional phase closes voids locally instead.',
            species=mix, presets=('pmma_dish_slice', 'fibroblast_realistic', 'functionalized_wall'),
            factors=F(force_model='hill', division=True, wall='functionalized', geometry='dish',
                      func_ratio=0.5), **dish),
        Condition(
            key='box_legacy_hill_div', label='2D closed box (control) - full model',
            description='Control: the same cells in a V3.0 closed box with the centripetal packing. '
                        'A fixed-volume container cannot compact, whatever the cells do.',
            species=one, presets=('pmma_dish_slice', 'fibroblast_realistic', 'inert_wall'),
            setup_overrides=('boundary.top=wall', 'gravity.enabled=false',
                             'packing.consolidation=centre', 'granules.bed_height_um=0.0',
                             'granules.solid_fraction=0.55'),
            factors=F(force_model='hill', division=True, wall='inert', geometry='closed box'),
            **{**dish, 'boundary_top': 'wall', 'bed_height_um': None, 'solid_fraction': 0.55}),
        Condition(
            key='well3d_hill_div', label='3D well - full model, inert wall',
            description=f'{int(granule_um_3d)} um PMMA granules in a {WELL_DIAMETER_UM/1000:g} mm '
                        'cylindrical well with a free top, against an inert wall.',
            species=one3d, presets=('pmma_well', 'fibroblast_realistic', 'inert_wall'),
            factors=F(force_model='hill', division=True, wall='inert', geometry='well',
                      granule_um=granule_um_3d), **well3d),
        Condition(
            key='well3d_hill_div_fwall', label='3D well - full model, functionalized wall',
            description='The same well with a collagen-coated wall and an immobile granule lining.',
            species=one3d, presets=('pmma_well', 'fibroblast_realistic', 'functionalized_wall'),
            factors=F(force_model='hill', division=True, wall='functionalized', geometry='well',
                      granule_um=granule_um_3d), **well3d),
    ]
    for c in conds:
        c.factors['n_estimated'] = sum(c.estimated_counts())
        n = c.factors['n_estimated']
        c.threads = 12 if n > 60000 else (8 if n > 15000 else 4)
    return conds

SWEEPS = {
    'showcase': (lambda: CONDITIONS, os.path.join('results', 'showcase_v3'),
                 os.path.join('Trials', 'showcase_v3')),
    'sweep3d': (build_sweep3d, os.path.join('results', 'sweep3d'),
                os.path.join('Trials', 'sweep3d')),
    'well': (build_well, os.path.join('results', 'well_calibration'),
             os.path.join('Trials', 'well_calibration')),
}


# ──────────────────────────────────────────────────────────────────────
# Setup files
# ──────────────────────────────────────────────────────────────────────

def build_setup(cond, t_total, seed):
    """A gels.config.Setup for the condition, starting from the V3.1 template.

    Order: template -> presets -> the condition's explicit fields -> its dotted
    overrides, so the explicit fields beat a preset and an override beats
    everything (the same precedence as step 1's file -> --preset -> --set).
    """
    from gels.config import (GranuleSpecies, MomentSpec, RadiusSpec, apply_overrides,
                             template_setup, validate)
    setup = template_setup()
    if cond.presets:
        from gels.presets import apply_presets
        apply_presets(setup, list(cond.presets))
    setup.domain.mode = cond.mode
    setup.domain.size_um = [float(v) for v in cond.size_um]
    setup.granules.solid_fraction = float(cond.solid_fraction)
    setup.granules.count_rule = 'mean_volume'
    setup.granules.species = []
    for sp in cond.species:
        mu, sd, rmin = sp['radius']
        setup.granules.species.append(GranuleSpecies(
            name=sp['name'], functionalization=sp['f'], volume_fraction=sp['vf'], color=sp['color'],
            radius=RadiusSpec(mean_um=mu, std_um=sd, min_um=rmin, distribution='normal'),
            aspect_ratio=MomentSpec(*sp['aspect']), aspect_ratio_c=MomentSpec(*sp['aspect_c']),
            blockiness=MomentSpec(*sp['block']), blockiness_n2=MomentSpec(*sp['block_n2']),
            E_kPa=sp['E_kPa']))
    setup.packing.shape_enabled = bool(cond.shape_enabled)
    setup.time.t_total_h = float(t_total)
    setup.time.save_every_h = float(cond.save_every_h)
    setup.performance.threads = int(cond.threads)
    if cond.Ngrid is not None:
        setup.output.Ngrid = int(cond.Ngrid)
    if cond.Ngrid_3d is not None:
        setup.output.Ngrid_3d = int(cond.Ngrid_3d)
    if cond.cell_surface_coverage is not None:
        setup.cells.seeding.surface_coverage = float(cond.cell_surface_coverage)
    # V3.1 container geometry
    setup.boundary.shape = cond.boundary_shape
    setup.boundary.top = cond.boundary_top
    if cond.bed_height_um:
        setup.granules.bed_height_um = float(cond.bed_height_um)
        setup.granules.bed_solid_fraction = 0.60 if cond.mode == '3D' else 0.80
    if cond.setup_overrides:
        apply_overrides(setup, list(cond.setup_overrides))
    validate(setup)
    return setup


def write_setup_file(cond, path, t_total, seed):
    from gels.config import save_setup
    setup = build_setup(cond, t_total, seed)
    save_setup(setup, path)
    counts = cond.estimated_counts()
    lines = [
        f"# GELS V3.0 showcase condition: {cond.key}",
        f"# {cond.label}",
        f"# {cond.description}",
        "#",
        f"# mode {cond.mode}, domain {cond.size_um[0]:g} x {cond.size_um[1]:g}"
        + (f" x {cond.size_um[2]:g}" if cond.mode == '3D' else '') + " um, solid fraction "
        f"{cond.solid_fraction:g}, ~{sum(counts):,} granules expected",
        "# species (f, share of solid, expected count): "
        + "; ".join(f"{sp['name']} ({sp['f']:.2f}, {sp['vf']:.2f}, ~{n})"
                    for sp, n in zip(cond.species, counts)),
        "#",
        "# Written by pipeline/run_showcase.py; everything not listed there is the V3.0 template",
        "# default (gels/config.py TEMPLATE_YAML). Run alone with:",
        f"#   python pipeline/step1_config.py --setup {path.replace(os.sep, '/')} --name {cond.key} --seed {seed}",
        "",
    ]
    with open(path, encoding='utf-8') as fh:
        body = fh.read()
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines) + body)
    return path


# ──────────────────────────────────────────────────────────────────────
# Running one condition through the numbered steps
# ──────────────────────────────────────────────────────────────────────

def _state(run_dir):
    try:
        with open(os.path.join(run_dir, 'pipeline_state.json'), encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {'steps': {}}


def _is_done(run_dir, step):
    return _state(run_dir).get('steps', {}).get(step, {}).get('status') == 'done'


def _details(run_dir, step):
    return _state(run_dir).get('steps', {}).get(step, {}).get('details', {}) or {}


def _child_env():
    env = dict(os.environ)
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('PYTHONUTF8', '1')
    return env


def run_condition(cond, out_dir, setups_dir, seed, t_total, force, do_post, do_analysis,
                  dry_run=False, log_fn=print, gate=None):
    run_dir = os.path.join(out_dir, cond.key)
    setup_path = os.path.join(setups_dir, cond.key + '.yaml')
    os.makedirs(run_dir, exist_ok=True)
    manifest_path = os.path.join(run_dir, 'showcase.json')
    manifest = dict(key=cond.key, label=cond.label, description=cond.description, threads=cond.threads,
                    seed=seed, t_total_h=t_total, setup=setup_path, status='running', steps={},
                    factors=cond.factors or {})
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)

    py = [sys.executable, '-u']
    f = ['--force'] if force else []
    plan = [
        ('config', py + [os.path.join('pipeline', 'step1_config.py'), '--setup', setup_path,
                         '-o', run_dir, '--seed', str(seed)] + f),
        ('pack', py + [os.path.join('pipeline', 'step2_pack.py'), '-i', run_dir] + f),
        ('simulate', py + [os.path.join('pipeline', 'step3_simulate.py'), '-i', run_dir,
                           '--threads', str(cond.threads)] + f),
    ]
    if do_post:
        cmd = py + [os.path.join('pipeline', 'step4_postprocess.py'), '-i', run_dir,
                    '--workers', str(cond.viz_workers)] + f
        if cond.viz_only:
            cmd += ['--only', *cond.viz_only]
        if cond.viz_skip:
            cmd += ['--skip', *cond.viz_skip]
        plan.append(('postprocess', cmd))
    if do_analysis and cond.analysis:
        plan.append(('analysis', py + [os.path.join('pipeline', 'step5_analysis.py'), '-i', run_dir] + f))

    log_path = os.path.join(run_dir, 'showcase_log.txt')
    t_start = time.time()
    status = 'ok'
    holding = gate is not None          # the compute budget covers steps 1-3 only
    with open(log_path, 'a', encoding='utf-8') as log:
        log.write(f"\n===== showcase {cond.key} — {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        for step, cmd in plan:
            if holding and step in ('postprocess', 'analysis'):
                gate.release(cond.threads)     # single-threaded plotting: let the next physics run start
                holding = False
                log_fn(f"  [{cond.key}] physics done; released {cond.threads} threads for the queue")
            if _is_done(run_dir, step) and not force:
                log_fn(f"  [{cond.key}] {step}: already done, skipped")
                manifest['steps'][step] = dict(status='skipped')
                continue
            log_fn(f"  [{cond.key}] {step} ...")
            log.write(f"\n--- {step}: {' '.join(cmd)}\n")
            log.flush()
            if dry_run:
                manifest['steps'][step] = dict(status='dry-run', cmd=cmd)
                continue
            t0 = time.time()
            rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=REPO, env=_child_env())
            dt = time.time() - t0
            manifest['steps'][step] = dict(status='ok' if rc == 0 else f'failed (exit {rc})',
                                           seconds=round(dt, 1))
            if rc != 0:
                log_fn(f"  [{cond.key}] {step} FAILED (exit {rc}) after {dt:.0f} s — see {log_path}")
                status = f'failed at {step}'
                break
            extra = ''
            d = _details(run_dir, step)
            if step == 'pack':
                extra = f" — {d.get('n_granules', '?')} granules, {d.get('total_cells', '?')} cells"
            elif step == 'simulate':
                extra = f" — {d.get('n_snapshots', '?')} snapshots"
            log_fn(f"  [{cond.key}] {step} done in {dt:.0f} s{extra}")
    if holding:
        gate.release(cond.threads)
    manifest['status'] = status
    manifest['wall_seconds'] = round(time.time() - t_start, 1)
    manifest['run_dir'] = run_dir
    with open(manifest_path, 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


# ──────────────────────────────────────────────────────────────────────
# Weighted scheduler: sum of thread counts of running conditions <= budget
# ──────────────────────────────────────────────────────────────────────

class ThreadBudget:
    def __init__(self, budget):
        self.budget = max(1, int(budget))
        self.used = 0
        self.running = 0
        self.cv = threading.Condition()

    def acquire(self, weight):
        with self.cv:
            while self.running > 0 and self.used + weight > self.budget:
                self.cv.wait()
            self.used += weight
            self.running += 1

    def release(self, weight):
        with self.cv:
            self.used -= weight
            self.running -= 1
            self.cv.notify_all()

    def release_if_held(self, weight):
        """Release after an exception when it is unknown whether run_condition got that far."""
        with self.cv:
            if self.used >= weight and self.running > 0:
                self.used -= weight
                self.running -= 1
                self.cv.notify_all()


def run_many(conds, budget, **kw):
    gate = ThreadBudget(budget)
    results = {}
    lock = threading.Lock()

    def log_fn(msg):
        with lock:
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    def worker(cond):
        gate.acquire(cond.threads)
        try:
            log_fn(f"  [{cond.key}] start ({cond.threads} threads; {gate.used}/{gate.budget} in use)")
            res = run_condition(cond, log_fn=log_fn, gate=gate, **kw)   # releases the budget itself
        except Exception as exc:  # keep the other conditions going
            res = dict(key=cond.key, label=cond.label, status=f'error: {type(exc).__name__}: {exc}')
            log_fn(f"  [{cond.key}] ERROR {exc}")
            gate.release_if_held(cond.threads)
        with lock:
            results[cond.key] = res

    # heaviest first so the big runs do not end up waiting for a last small slot
    threads = [threading.Thread(target=worker, args=(c,), daemon=True, name=c.key)
               for c in sorted(conds, key=lambda c: -c.threads)]
    for th in threads:
        th.start()
        time.sleep(0.5)      # stagger the starts (numba cache index writes, console noise)
    for th in threads:
        th.join()
    return [results[c.key] for c in conds if c.key in results]


# ──────────────────────────────────────────────────────────────────────
# Warm-up, comparison, reporting
# ──────────────────────────────────────────────────────────────────────

def warm_up(log_fn=print):
    """Compile / load the 2D kernels once so the parallel children start from a warm cache."""
    t0 = time.time()
    from gels import Params, run
    p = Params(mode='2D', Lx=300.0, Ly=300.0, phi_solid_target=0.6, func_ratio=0.6,
               cell_surface_coverage=1.0, t_total=1.0, dt=0.5, save_every_h=1.0, save_data=False,
               packing_settle_steps=10, perf_threads=2)
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run(p, seed=1)
    log_fn(f"  kernels warm ({time.time() - t0:.0f} s)")


def compare(run_dirs, labels, outdir, max_frames, log_fn=print):
    cmd = [sys.executable, '-u', os.path.join('viz2', 'compare_runs.py'), '-i', *run_dirs,
           '-o', outdir, '--labels', *labels, '--max-frames', str(max_frames)]
    log_fn(f"  comparison -> {outdir}")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO, env=_child_env())
    log_fn(f"  comparison {'done' if rc == 0 else f'FAILED (exit {rc})'} in {time.time() - t0:.0f} s")
    return rc


SWEEP_TITLES = {
    'showcase': ("GELS V3.0 showcase",
                 "Nine conditions exercising what V3.0 adds: a functionalization ladder, "
                 "shaped granules, a 20 mm two-dimensional bed and a three-dimensional cube."),
    'well': ("GELS V3.1 — well calibration set",
             "A scaled model of the lab's experiment: PMMA granules sedimented under gravity in a "
             "2 mm well with a flat floor, a cylindrical side wall and a free top. The set varies "
             "one thing at a time - container, bridge force model, cell division and wall "
             "chemistry - so each can be calibrated against the measured bed height separately. "
             "The closed-box control is there to show that a fixed-volume container cannot "
             "compact at all."),
    'sweep3d': ("GELS V3.0 — 3D granule size x composition sweep",
                "Full factorial in a 2 mm cube over functional granule size (mean diameter "
                "40 / 80 / 150 um), inert granule size (the same three) and functional share of "
                "the solid (0.50 / 0.75 / 1.00). The 100 %-functional cells have no inert phase, "
                "so the 27 nominal combinations collapse to 21 distinct runs. Cells are seeded to "
                "0.8 of each granule's surface area; solid fraction 0.55; 72 h per run."),
}


# Caveats printed under "## Caveats" in each sweep's README.
SWEEP_NOTES = {
    'sweep3d': [
        "**The packings of this sweep carry a known artefact (fixed in V3.1).** Until V3.1 the "
        "settle pulled every granule toward the centre of the box for the whole inflation and "
        "the walls were a one-sided clamp, so a packing below random close packing consolidated "
        "into a ball. Re-packed on this sweep's own configuration (2 mm cube, 150 um granules, "
        "phi 0.55): with the old `centre` consolidation all eight 200 um corner cubes hold 0 "
        "granules and the bed reaches only 1384 um from the centre; with `none` they hold 1-3 "
        "each (uniform expectation 2.4) and the bed reaches 1571 um, against a corner at 1732 um. "
        "At 40 um granules the effect is larger still: the bed is a ball of radius ~1400 um, the "
        "interior is 1.31x over-dense, and `phi_solid` FALLS 0.725 -> 0.670 over the run as the "
        "bed relaxes outward into the corner voids.",
        "Consequence for reading these numbers: part of the reported `disp_func` is relaxation of "
        "the packing rather than cell-driven motion, and `phi_solid` starts above random close "
        "packing. The comparison of conditions is still informative, because every run shares the "
        "artefact, but the absolute compaction is not. Re-run with "
        "`packing.consolidation=none` (a closed box) before quoting absolute numbers.",
        "The plotting and analysis timings recorded here predate the V3.0 performance fixes and "
        "are not representative of the current code.",
    ],
    'well': [
        "Rigid monodisperse spheres cannot pack past phi ~ 0.64 and a sedimented bed starts at "
        "0.55-0.60, so densification alone lowers a bed by at most ~7-15 %. Compaction beyond "
        "that bound has to come from envelope contraction (the bed detaching from an inert wall, "
        "clusters coalescing), from a much looser initial bed (lower `bed_solid_fraction`), or "
        "from cell and ECM volume, which this model does not represent.",
        "Compare the POSITIONAL observables (`bed_height_mean`, `phi_bed`, `bed_radius_p95`, "
        "`wall_contact_fraction`) with the measurement. The field-based `phi_solid` and "
        "`porosity` are inflated by the tanh interface halo and by overlap double counting.",
        "The well is a 2 mm scaled model; a real plate well is 6.4 mm across, so wall effects "
        "are relatively stronger here. Fit the RELATIVE bed height h(t)/h(0) first.",
    ],
}

def write_readme(out_dir, conds, sweep='showcase', setups_dir=None):
    """README.md in the output folder: what each run is and where the figures are."""
    title, blurb = SWEEP_TITLES.get(sweep, (f"GELS run set: {sweep}", ""))
    setups_dir = (setups_dir or SWEEPS[sweep][2]).replace(os.sep, '/')

    def dom(c):
        d = f"{c.size_um[0]/1000:g} × {c.size_um[1]/1000:g}"
        return d + (f" × {c.size_um[2]/1000:g} mm" if c.mode == '3D' else " mm")
    lines = [
        f"# {title}",
        "",
        blurb,
        "",
        f"Generated by `python pipeline/run_showcase.py --sweep {sweep}` "
        f"({time.strftime('%Y-%m-%d %H:%M')}).",
        "Every condition is a standard pipeline run directory (`params.json`, `setup.yaml`,",
        "`snapshots/`, `history.json`, `viz2_plots/`, `analysis/`) plus `showcase_log.txt`",
        "(pipeline output) and `showcase.json` (label, thread count, per-step wall times).",
        f"The setup file of each condition is `{setups_dir}/<run>.yaml`; edit it and rerun",
        f"one condition alone with `python pipeline/step1_config.py --setup {setups_dir}/<run>.yaml`.",
        "",
        "## All conditions on the same figures — `comparison/`",
        "",
        "| file | content |",
        "|---|---|",
        "| `compare_scaffolds.png` | final scaffold of every condition; the same f has the same colour on every panel; scale bar per panel |",
        "| `compare_timeseries.png` | bridging cells, bridge force, functional connectivity, porosity, displacement vs time — one line per condition |",
        "| `compare_species.png` | stacked species fractions vs time, one panel per condition |",
        "| `compare_dose_response.png` | final cells per granule, bridging fraction, displacement and coordination number of every species against its own f |",
        "| `compare_sweep.png` | (sweeps only) final porosity, coordination, compaction and bridging against the composition axis, one line per granule-size combination |",
        "| `compare_timelapse.gif` | every condition animated on a common time axis |",
        "| `compare_summary.md` / `.csv` | sizes, cells, composition per species, wall times of each step, final metrics |",
        "",
        "Colour code: f = 1.00 red, 0.75 orange, 0.50 blue, 0.25 purple, 0.20 teal, 0.00 green;",
        "cells: proliferating green, migrating blue, bridging crimson, senescent grey; void black.",
        "",
        "## Conditions",
        "",
        "| run | condition | mode | domain | ~N | species (f: share of solid) | threads |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in conds:
        sp = ", ".join(f"{s['f']:.2f}: {s['vf']:.2f}" for s in c.species)
        if c.shape_enabled:
            sp += " (superellipse shapes)"
        lines.append(f"| `{c.key}/` | {c.label} | {c.mode} | {dom(c)} | {sum(c.estimated_counts()):,} | {sp} | {c.threads} |")
    lines += ["", "What each condition shows:", ""]
    for c in conds:
        lines.append(f"- **{c.key}** — {c.description}")
    if any(c.factors for c in conds):
        keys = sorted({k for c in conds for k in (c.factors or {})})
        lines += ["", "## Sweep coordinates", "",
                  "| run | " + " | ".join(keys) + " |",
                  "|---|" + "---|" * len(keys)]
        for c in conds:
            fac = c.factors or {}
            lines.append(f"| `{c.key}` | " + " | ".join(str(fac.get(k, '')) for k in keys) + " |")
    notes = SWEEP_NOTES.get(sweep)
    if notes:
        lines += ["", "## Caveats", ""]
        for n in notes:
            lines.append(f"- {n}")
    lines += ["", "Per-run figures are in `<run>/viz2_plots/`, the analysis summary in `<run>/analysis/summary.json`.", ""]
    path = os.path.join(out_dir, 'README.md')
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write("\n".join(lines))
    return path


def print_conditions(conds):
    wide = max([len(c.key) for c in conds] + [3])
    print(f"\n  {'key':<{wide}s} {'mode':<4s} {'domain [mm]':<16s} {'thr':>3s} {'~N':>8s} {'~cells':>9s}  species (f: share)")
    for c in conds:
        counts = c.estimated_counts()
        dom = (f"{c.size_um[0]/1000:g} x {c.size_um[1]/1000:g}"
               + (f" x {c.size_um[2]/1000:g}" if c.mode == '3D' else ''))
        sp = ", ".join(f"{s['f']:.2f}: {s['vf']:.2f}" for s in c.species)
        flags = " shapes" if c.shape_enabled else ""
        cov = c.cell_surface_coverage if c.cell_surface_coverage is not None else 1.0
        cells = 0
        for s, n in zip(c.species, counts):
            R = s['radius'][0]
            per = round(4 * math.pi * R * R * cov / 1257.0) if s['f'] >= 0.05 else 0
            cells += n * per
        print(f"  {c.key:<{wide}s} {c.mode:<4s} {dom:<16s} {c.threads:>3d} {sum(counts):>8,d} "
              f"{cells:>9,d}  {sp}{flags}")
        print(f"  {'':<{wide}s} {c.label}")
    print()


def print_results(results, out_dir):
    print("\n  Showcase summary")
    print(f"  {'condition':<18s} {'status':<16s} {'pack':>7s} {'sim':>7s} {'viz':>7s} {'analysis':>9s} {'total':>7s}")
    for r in results:
        steps = r.get('steps', {})

        def sec(name):
            v = steps.get(name, {}).get('seconds')
            return f"{v:7.0f}" if v is not None else f"{'—':>7s}"
        total = r.get('wall_seconds')
        print(f"  {r['key']:<18s} {r.get('status', '?'):<16s} {sec('pack')} {sec('simulate')} "
              f"{sec('postprocess')} {sec('analysis'):>9s} {total if total is not None else '—':>7}")
    print(f"\n  Output folder: {out_dir}")
    print(f"  Comparison:    {os.path.join(out_dir, 'comparison')}")


def main():
    parser = argparse.ArgumentParser(description="GELS V3.0 showcase runner.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__)
    parser.add_argument('--sweep', choices=sorted(SWEEPS), default='showcase',
                        help='Which condition set to run: showcase (the nine V3.0 demos) or '
                             'sweep3d (21-condition 3D granule size x composition sweep)')
    parser.add_argument('-o', '--out', default=None, help='Output folder (default: per sweep)')
    parser.add_argument('--setups-dir', default=None,
                        help='Where the condition setup files are written (default: per sweep)')
    parser.add_argument('--only', nargs='*', default=None, help='Condition keys to run (default all)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--t_total', type=float, default=72.0, help='Simulated hours (default 72)')
    parser.add_argument('--budget', type=int, default=None,
                        help='Total compute threads across concurrent runs (default: physical cores)')
    parser.add_argument('--force', action='store_true', help='Redo completed steps')
    parser.add_argument('--no-postprocess', action='store_true', help='Skip step 4 per run')
    parser.add_argument('--no-analysis', action='store_true', help='Skip step 5 per run')
    parser.add_argument('--no-warmup', action='store_true', help='Do not pre-compile the kernels')
    parser.add_argument('--frames', type=int, default=24, help='Frames in the comparison timelapse')
    parser.add_argument('--list', action='store_true', help='Print the condition table and exit')
    parser.add_argument('--write-setups', action='store_true', help='Write the setup files and exit')
    parser.add_argument('--compare-only', action='store_true', help='Only redraw the comparison figures')
    parser.add_argument('--dry-run', action='store_true', help='Print the commands without running them')
    args = parser.parse_args()

    builder, default_out, default_setups = SWEEPS[args.sweep]
    all_conds = builder()
    if args.out is None:
        args.out = default_out
    if args.setups_dir is None:
        args.setups_dir = default_setups

    conds = all_conds
    if args.only:
        unknown = [k for k in args.only if k not in {c.key for c in all_conds}]
        if unknown:
            print(f"  Unknown condition(s): {unknown}. Known: {[c.key for c in all_conds]}")
            sys.exit(1)
        conds = [c for c in all_conds if c.key in set(args.only)]

    if args.list:
        print_conditions(conds)
        return

    os.chdir(REPO)
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.setups_dir, exist_ok=True)

    if not args.compare_only:
        for c in conds:
            path = write_setup_file(c, os.path.join(args.setups_dir, c.key + '.yaml'), args.t_total, args.seed)
            print(f"  wrote {path}")
        if args.write_setups:
            return

    budget = args.budget
    if budget is None:
        try:
            from gels.kernels import physical_cores
            budget = physical_cores()
        except Exception:
            budget = os.cpu_count() or 8
    print_conditions(conds)
    print(f"  thread budget {budget}, seed {args.seed}, t_total {args.t_total:g} h, output {args.out}\n")

    results = []
    if not args.compare_only:
        if not (args.no_warmup or args.dry_run):
            warm_up()
        t0 = time.time()
        results = run_many(conds, budget, out_dir=args.out, setups_dir=args.setups_dir, seed=args.seed,
                           t_total=args.t_total, force=args.force, do_post=not args.no_postprocess,
                           do_analysis=not args.no_analysis, dry_run=args.dry_run)
        print(f"\n  all conditions finished in {time.time() - t0:.0f} s")
        with open(os.path.join(args.out, 'showcase_manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(dict(seed=args.seed, t_total_h=args.t_total, budget=budget,
                           conditions=results), fh, indent=2)
        print_results(results, args.out)
        if args.dry_run:
            return

    ok = [c for c in conds if _is_done(os.path.join(args.out, c.key), 'simulate')]
    if not ok:
        print("  No finished runs to compare.")
        return
    compare([os.path.join(args.out, c.key) for c in ok], [c.label for c in ok],
            os.path.join(args.out, 'comparison'), args.frames)
    print(f"  wrote {write_readme(args.out, conds, args.sweep, args.setups_dir)}")


if __name__ == '__main__':
    main()
