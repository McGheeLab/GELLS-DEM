"""
What a cell type is (V3.6).
===========================

A ``CellType`` is the set of measured properties that distinguish one cell from
another, held with its provenance, plus the derived quantities the engine needs.
It exists so that "run this with MSCs instead of fibroblasts" is a one-word
change rather than twenty correlated numbers, and so that every one of those
numbers says where it came from.

Two rules it is built to keep, both from
``CodeLog/References/fibroblast_parameters.md``:

* **Every value carries a source.** A `Measured` with ``source='ASSUMPTION'``
  is a calibration target, not a datum, and `CellType.report` prints it as such.
  There is no way to add a number without saying where it came from.
* **The traction stress is not a constant.** The reference file lists a
  "density-independent 5.5 nN/um^2 adhesion stress" as something the evidence
  explicitly does **not** support: the stress equals
  ``rho_bond * F_b * plog(gamma/e)`` and therefore scales with ENGAGED-BOND
  density. So `traction_stress_Pa` here is the stress at *saturating* ligand
  and full clutch engagement, and the engine multiplies it by the Langmuir
  ligand gain ``g`` and by the clutch engagement fraction before it is used.
  `adhesion_force_nN` is the only place that conversion is written down.

A type is a plain frozen dataclass, so a run can be reproduced from
``meta.cell_type`` and nothing about it is dynamic.
"""

from dataclasses import dataclass, field, fields
from typing import Optional, Tuple

import math

__all__ = ['Measured', 'CellType', 'ASSUMPTION']

ASSUMPTION = 'ASSUMPTION'


@dataclass(frozen=True)
class Measured:
    """One number, its units, its source, and the range the source reports.

    ``source`` is a citation, or `ASSUMPTION` when the value is a calibration
    target rather than a measurement. ``span`` is the range in the literature,
    which is what makes a derived quantity checkable: a model that lands
    outside the span of the things it was built from is saying something.
    """
    value: float
    units: str
    source: str
    span: Optional[Tuple[float, float]] = None

    def __float__(self):
        return float(self.value)

    @property
    def assumed(self) -> bool:
        return self.source == ASSUMPTION

    def in_span(self) -> Optional[bool]:
        if self.span is None:
            return None
        return self.span[0] <= self.value <= self.span[1]

    def __str__(self):
        s = f'{self.value:g} {self.units}'
        if self.span is not None:
            s += f' (reported {self.span[0]:g}-{self.span[1]:g})'
        return s


def _v(m):
    return float(m.value) if isinstance(m, Measured) else float(m)


@dataclass(frozen=True)
class CellType:
    """A cell type: measured properties, and the quantities derived from them.

    Everything here is per-cell and independent of the scaffold. What the cell
    meets -- granule modulus, ligand coverage, how many neighbours it has --
    belongs to the run, not the type.
    """

    name: str
    display_name: str
    description: str

    # ── geometry ──────────────────────────────────────────────────────
    diameter_um: Measured                 # rounded, before spreading
    spread_height_um: Measured            # height of the spread ellipsoid
    adhesion_area_fraction: Measured      # focal-adhesion area / projected area

    # ── traction ──────────────────────────────────────────────────────
    # At SATURATING ligand and full engagement. The engine scales it by the
    # Langmuir gain g and the clutch engagement fraction -- see the module
    # docstring, and `adhesion_force_nN` below.
    traction_stress_Pa: Measured
    total_traction_nN: Measured           # whole-cell, for the consistency check

    # ── motor-clutch (Chan & Odde 2008) ───────────────────────────────
    n_motors: Measured
    F_motor_stall_nN: Measured
    n_clutches: Measured
    k_clutch_nN_per_um: Measured
    k_on_clutch_per_s: Measured
    k_off_clutch_per_s: Measured

    # ── series elasticity: what stores the strain energy ──────────────
    # Stress fibres plus adhesions, in series with the substrate. On a stiff
    # granule this is the SOFT element and therefore sets the stored energy,
    # which is why a cell-type property and not a substrate one decides the
    # number that traction force microscopy reports.
    k_cell_nN_per_um: Measured

    # ── kinetics ──────────────────────────────────────────────────────
    migration_speed_um_per_h: Measured
    contraction_speed_um_per_h: Measured
    sense_distance_um: Measured
    spread_duration_h: Measured
    fa_maturation_rate_per_h: Measured
    doubling_time_h: Measured
    cycle_cv: Measured
    division_min_age_h: Measured

    # ── optional extras a type may or may not set ─────────────────────
    extra_overrides: tuple = field(default_factory=tuple)
    notes: str = ''

    # ── derived ───────────────────────────────────────────────────────

    def spread_area_um2(self, spread_fraction: float = 1.0) -> float:
        """Projected footprint at this spread fraction, um^2.

        Same expression as `engine.cell_projected_area`, duplicated here only so
        a type can be inspected without importing the engine. The engine remains
        the authority at run time; `tests/test_celltypes.py` holds them equal.
        """
        d = _v(self.diameter_um)
        h = _v(self.spread_height_um)
        r = d / 2.0
        A_sphere = math.pi * r ** 2
        A_spread = math.pi * (3.0 * ((4.0 / 3.0) * math.pi * r ** 3)
                              / (4.0 * math.pi * (h / 2.0)))
        return A_sphere + float(spread_fraction) * (A_spread - A_sphere)

    def adhesion_area_um2(self, spread_fraction: float = 1.0) -> float:
        """The part of the footprint that is actually focal adhesion, um^2."""
        return _v(self.adhesion_area_fraction) * self.spread_area_um2(spread_fraction)

    def adhesion_force_nN(self, spread_fraction: float = 1.0,
                          ligand_gain: float = 1.0, engagement: float = 1.0) -> float:
        """The traction the ADHESION can carry, nN.

        ``F = sigma * A``, with `sigma` scaled by the engaged-bond density --
        the Langmuir ligand gain and the clutch engagement fraction -- because a
        density-independent adhesion stress is the thing the reference file says
        not to implement.

        Pa * um^2 = 1e-12 N = 1e-3 nN, hence the factor.
        """
        sigma = _v(self.traction_stress_Pa) * float(ligand_gain) * float(engagement)
        return sigma * self.adhesion_area_um2(spread_fraction) * 1e-3

    def engagement(self) -> float:
        on = _v(self.k_on_clutch_per_s)
        off = _v(self.k_off_clutch_per_s)
        return on / (on + off) if (on + off) > 0 else 0.0

    def motor_stall_nN(self) -> float:
        return _v(self.n_motors) * _v(self.F_motor_stall_nN)

    def strain_energy_nN_um(self, force_nN: float, k_substrate_nN_per_um: float) -> float:
        """Stored elastic energy at this traction, nN.um, = 1e-15 J.

        The cell and the substrate are springs in series, so
        ``1/k = 1/k_cell + 1/k_sub`` and ``U = F^2 / 2k``. Divide by 1000 for
        the picojoules traction force microscopy reports.
        """
        kc = _v(self.k_cell_nN_per_um)
        ks = float(k_substrate_nN_per_um)
        if kc <= 0 and ks <= 0:
            return 0.0
        inv = (1.0 / kc if kc > 0 else 0.0) + (1.0 / ks if ks > 0 else 0.0)
        k = 1.0 / inv if inv > 0 else 0.0
        return 0.5 * float(force_nN) ** 2 / k if k > 0 else 0.0

    # ── how it reaches the engine ─────────────────────────────────────

    def to_overrides(self):
        """Dotted `--set`-style overrides, in `gels.presets` form.

        Deliberately the same currency as a preset, so a cell type composes with
        `pmma_well` / `fragmented_granules` and the resolution order (`--preset`,
        then `--cell-type`, then `--set`) stays one rule.
        """
        ov = [
            f'cells.seeding.diameter_um={_v(self.diameter_um):g}',
            f'cells.seeding.spread_height_um={_v(self.spread_height_um):g}',
            f'cells.motor_clutch.n_motors={int(_v(self.n_motors))}',
            f'cells.motor_clutch.F_motor_stall_nN={_v(self.F_motor_stall_nN):g}',
            f'cells.motor_clutch.n_clutches={int(_v(self.n_clutches))}',
            f'cells.motor_clutch.k_clutch_nN_per_um={_v(self.k_clutch_nN_per_um):g}',
            f'cells.motor_clutch.k_on_per_s={_v(self.k_on_clutch_per_s):g}',
            f'cells.motor_clutch.k_off_per_s={_v(self.k_off_clutch_per_s):g}',
            f'cells.traction.stress_Pa={_v(self.traction_stress_Pa):g}',
            f'cells.traction.adhesion_area_frac={_v(self.adhesion_area_fraction):g}',
            f'cells.traction.k_cell_nN_per_um={_v(self.k_cell_nN_per_um):g}',
            f'cells.motor_clutch.F_max_per_cell_nN={_v(self.total_traction_nN):g}',
            f'cells.migration.speed_um_per_h={_v(self.migration_speed_um_per_h):g}',
            f'cells.bridging.contraction_speed_um_per_h={_v(self.contraction_speed_um_per_h):g}',
            f'cells.sensing.sense_distance_um={_v(self.sense_distance_um):g}',
            f'cells.kinetics.spread_duration_h={_v(self.spread_duration_h):g}',
            f'cells.kinetics.fa_maturation_rate_per_h={_v(self.fa_maturation_rate_per_h):g}',
            f'cells.division.doubling_time_h={_v(self.doubling_time_h):g}',
            f'cells.division.cycle_time_cv={_v(self.cycle_cv):g}',
            f'cells.division.min_age_h={_v(self.division_min_age_h):g}',
            f'cells.type={self.name}',
        ]
        ov.extend(self.extra_overrides)
        return ov

    # ── what it says about itself ─────────────────────────────────────

    def report(self) -> str:
        """Every value with its source, then the derived checks.

        The derived block is the point: it is where a self-consistent type is
        told apart from a plausible-looking one. `sigma x A` against the
        independently measured whole-cell traction is the check that caught the
        area fraction.
        """
        out = [f'{self.display_name}  ({self.name})', '=' * 62, self.description, '']
        assumed = []
        for f_ in fields(self):
            m = getattr(self, f_.name)
            if not isinstance(m, Measured):
                continue
            flag = '  <- ASSUMPTION' if m.assumed else ''
            span = m.in_span()
            if span is False:
                flag += '  <- OUTSIDE the reported range'
            out.append(f'  {f_.name:<28} {str(m):<34} {m.source}{flag}')
            if m.assumed:
                assumed.append(f_.name)
        A = self.spread_area_um2()
        A_ad = self.adhesion_area_um2()
        F_ad = self.adhesion_force_nN(1.0, 1.0, self.engagement())
        F_meas = _v(self.total_traction_nN)
        out += ['', '  derived', '  ' + '-' * 60,
                f'  spread footprint             {A:.0f} um^2',
                f'  adhesion area                {A_ad:.0f} um^2'
                f'  ({_v(self.adhesion_area_fraction) * 100:.0f} % of it)',
                f'  motor stall                  {self.motor_stall_nN():.0f} nN',
                f'  clutch engagement            {self.engagement():.2f}',
                f'  sigma x A at saturation      {F_ad:.0f} nN',
                f'  measured whole-cell traction {F_meas:.0f} nN',
                f'  ratio                        {F_ad / F_meas:.2f}'
                f'{"   <- these should agree" if abs(F_ad / F_meas - 1) > 0.5 else ""}']
        for k_sub, lab in ((394.0, '10 kPa granule'), (1e5, 'rigid granule')):
            U = self.strain_energy_nN_um(F_meas, k_sub)
            out.append(f'  strain energy, {lab:<16} {U:.0f} nN.um = {U / 1000:.2f} pJ')
        if assumed:
            out += ['', f'  {len(assumed)} value(s) are ASSUMPTIONS, not measurements: '
                        + ', '.join(assumed),
                    '  Those are the ones to calibrate first.']
        if self.notes:
            out += ['', '  ' + self.notes.strip().replace('\n', '\n  ')]
        return '\n'.join(out)
