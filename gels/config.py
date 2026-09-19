"""
Sectioned simulation setup (V3.0).
==================================

A GELS run is described by a YAML (or JSON) file organised by the objects in
the system::

    domain, boundary, granules (solid fraction + a list of SPECIES), cells
    (seeding, ligand, kinetics, motor_clutch, migration, sensing, bridging,
    activity), contact, dynamics, packing, time, output, performance, deformable

This module holds the nested dataclasses for those sections (:class:`Setup`),
loads/saves the files, converts to and from the engine's flat :class:`Params`
(``Setup.to_params`` / ``Setup.from_params``) and applies dotted-path
overrides (``apply_overrides``). The engine itself never sees a Setup — it
keeps its flat ``p.X`` access — which is why the 400 legacy trial files, old
``params.json`` files and plain ``Params()`` construction all keep working.

Defaults of every section are taken from ``Params()`` so a default Setup
round-trips exactly to default Params. The **recommended physical values**
for fibroblasts on collagen-I (see ``CodeLog/References/fibroblast_parameters.md``)
live in :data:`TEMPLATE_YAML`, the commented file new setups start from —
not in ``Params`` defaults, which older runs and the regression fixtures rely on.

Usage::

    from gels.config import load_setup, TEMPLATE_YAML
    setup = load_setup('setup.yaml')
    setup = apply_overrides(setup, ['granules.species[1].functionalization=0.5'])
    p = setup.to_params()
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional

from gels.engine import Params, legacy_species_from_params, resolve_species

_P = Params()   # source of every default so Setup() ≡ Params() after to_params()


# ──────────────────────────────────────────────────────────────────────
# Section dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class Domain:
    mode: str = _P.mode                          # 2D | 2D-slice | 3D
    size_um: List[float] = field(default_factory=lambda: [_P.Lx, _P.Ly, _P.Lz])


@dataclass
class Meta:
    presets: List[str] = field(default_factory=list)   # names applied by --preset (provenance)
    notes: str = ''


@dataclass
class BoundaryLayer:
    enabled: bool = _P.boundary_layer_enabled     # immobile granule lattice lining floor + side wall
    radius_um: float = _P.boundary_layer_radius   # 0 -> mean radius of the smallest mobile species
    embed_frac: float = _P.boundary_layer_embed_frac
    seed_cells: bool = _P.boundary_layer_seed_cells


@dataclass
class Boundary:
    mode: str = _P.boundary_mode                  # walls | periodic
    shape: str = _P.boundary_shape                # box | cylinder (3D only)
    top: str = _P.boundary_top                    # wall | free
    functionalization: float = _P.boundary_functionalization   # collagen coverage of the wall (0 = inert)
    layer: BoundaryLayer = field(default_factory=BoundaryLayer)


@dataclass
class Gravity:
    enabled: bool = _P.gravity_enabled
    g_m_per_s2: float = _P.g_accel
    medium_density_kg_m3: float = _P.medium_density
    granule_density_kg_m3: float = _P.granule_density   # default for species without density_kg_m3
    scale: float = _P.gravity_scale


@dataclass
class RadiusSpec:
    mean_um: float = 40.0
    std_um: float = 0.0
    min_um: Optional[float] = None                # None → max(1, 0.25·mean)
    distribution: str = 'normal'                  # normal | lognormal


@dataclass
class MomentSpec:
    mean: float = 1.0
    std: float = 0.0


@dataclass
class GranuleSpecies:
    name: str = 'granule'
    functionalization: float = 1.0                # f ∈ [0,1]: fraction of surface coated with collagen-I
    volume_fraction: float = 1.0                  # share of the SOLID volume (area in 2D)
    color: str = '#CC2222'
    radius: RadiusSpec = field(default_factory=RadiusSpec)
    aspect_ratio: MomentSpec = field(default_factory=MomentSpec)        # a/b (2D) / equatorial (3D)
    aspect_ratio_c: MomentSpec = field(default_factory=MomentSpec)      # c/a (3D)
    blockiness: MomentSpec = field(default_factory=lambda: MomentSpec(2.0, 0.0))     # n (2D) / n1 (3D)
    blockiness_n2: MomentSpec = field(default_factory=lambda: MomentSpec(2.0, 0.0))  # n2 (3D)
    E_kPa: Optional[float] = None                 # None → contact.E_kPa
    poisson_ratio: Optional[float] = None         # None → contact.poisson_ratio
    count_rule: Optional[str] = None              # None → granules.count_rule (legacy species: mean_radius)
    density_kg_m3: Optional[float] = None         # None → gravity.granule_density_kg_m3


def _legacy_species_defaults() -> List[GranuleSpecies]:
    return [_species_from_dict(d) for d in legacy_species_from_params(_P)]


@dataclass
class Granules:
    solid_fraction: float = _P.phi_f_target + _P.phi_i_target   # total solid VOLUME fraction (AREA in 2D)
    bed_height_um: float = _P.bed_height          # > 0: amount as a settled bed height (needs boundary.top free)
    bed_solid_fraction: float = _P.bed_phi_assumed   # packing fraction assumed for that bed (count only)
    count_rule: str = _P.packing_count_rule                       # mean_volume | mean_radius
    species: List[GranuleSpecies] = field(default_factory=_legacy_species_defaults)


@dataclass
class Seeding:
    surface_coverage: float = _P.cell_surface_coverage   # monolayer reference at f = 1 (0 → n_cells_per_granule)
    n_cells_per_granule: int = _P.n_cells_per_granule    # legacy fallback when surface_coverage = 0
    coverage: float = _P.cell_coverage                   # legacy projected-area cap
    law: str = _P.cell_seeding_law                       # langmuir | power
    K_attach_per_um2: float = _P.K_sigma_attach
    seeding_exponent: float = _P.cell_seeding_exponent
    min_functionalization: float = _P.f_min_adhesion
    diameter_um: float = _P.cell_diameter
    spread_height_um: float = _P.cell_height_spread
    stacking_max: float = _P.cell_stacking_max
    overcrowd_senescence_time_h: float = _P.overcrowd_senescence_time
    capacity_coverage: float = _P.cell_capacity_coverage   # 0 -> surface_coverage (capacity = seeding)
    clock_jitter_h: float = _P.cell_clock_jitter           # V3.2: per-granule offset on the maturation clock
    capacity_foothold: float = _P.cell_capacity_foothold   # V3.2: footprint a cell needs to HOLD a place


@dataclass
class Ligand:
    sigma_max_per_um2: float = _P.sigma_ligand_max
    K_traction_per_um2: float = _P.K_sigma_traction


@dataclass
class Kinetics:
    attach_onset_h: float = _P.t_attach_onset
    attach_half_h: float = _P.t_attach_half
    spread_duration_h: float = _P.t_spread_duration
    fa_maturation_rate_per_h: float = _P.fa_maturation_rate


@dataclass
class MotorClutch:
    n_motors: int = _P.n_motors
    F_motor_stall_nN: float = _P.F_motor_stall
    n_clutches: int = _P.n_clutches
    k_clutch_nN_per_um: float = _P.k_clutch
    k_on_per_s: float = _P.k_on_clutch
    k_off_per_s: float = _P.k_off_clutch
    F_max_per_cell_nN: float = _P.F_max_per_cell
    traction_f_law: str = _P.traction_f_law
    traction_f_rule: str = _P.traction_f_rule
    traction_exponent: float = _P.traction_exponent


@dataclass
class Migration:
    speed_um_per_h: float = _P.cell_migration_speed
    directed_speed_mult: float = _P.bridge_directed_speed_mult


@dataclass
class Sensing:
    sense_distance_um: float = _P.cell_sense_distance
    bridge_decay_length_um: float = _P.bridge_decay_length


@dataclass
class Bridging:
    attempt_rate_per_h: float = _P.bridge_attempt_rate
    formation_time_h: float = _P.bridge_formation_time
    senescence_time_h: float = _P.bridge_senescence_time
    min_fa: float = _P.min_fa_for_bridge
    break_gap_um: float = _P.bridge_break_gap
    lock_force_nN: float = _P.bridge_lock_force_threshold
    lock_scales_with_ligand: bool = _P.bridge_lock_scales_with_ligand
    secondary_rate_mult: float = _P.bridge_secondary_rate_mult
    expected_force_nN: float = _P.expected_bridge_force
    contact_factor: float = _P.bridge_contact_factor
    bare_blocker_factor: float = _P.bridge_inert_factor
    commit_angle_rad: float = _P.bridge_commit_angle
    exclusion_angle_rad: float = _P.bridge_exclusion_angle
    alignment_rate_per_h: float = _P.bridge_alignment_rate
    alignment_min: float = _P.bridge_alignment_min
    force_model: str = _P.bridge_force_model                     # constant | hill
    contraction_speed_um_per_h: float = _P.cell_contraction_speed
    eccentric_gain: float = _P.bridge_eccentric_gain
    min_gap_um: float = _P.bridge_min_gap
    contact_adhesion_nN_per_cell: float = _P.cell_contact_adhesion   # V3.2: shear grip a bridging cell adds


@dataclass
class Division:
    enabled: bool = _P.cell_division_enabled
    doubling_time_h: float = _P.cell_doubling_time
    min_age_h: float = _P.cell_division_min_age
    max_layers: float = _P.cell_division_max_layers
    divide_while_bridging: bool = _P.cell_divide_while_bridging
    model: str = _P.cell_division_model        # V3.2: poisson (memoryless) | cycle (per-cell cycle length)
    cycle_time_cv: float = _P.cell_division_cv  # V3.2: lognormal CV of the per-cell cycle length


@dataclass
class Activity:
    T_active_nN_um: float = _P.T_active


@dataclass
class Cells:
    seeding: Seeding = field(default_factory=Seeding)
    ligand: Ligand = field(default_factory=Ligand)
    kinetics: Kinetics = field(default_factory=Kinetics)
    motor_clutch: MotorClutch = field(default_factory=MotorClutch)
    migration: Migration = field(default_factory=Migration)
    sensing: Sensing = field(default_factory=Sensing)
    bridging: Bridging = field(default_factory=Bridging)
    division: Division = field(default_factory=Division)
    activity: Activity = field(default_factory=Activity)


@dataclass
class PairTriplet:
    cc: float = 0.0       # collagen–collagen
    cb: float = 0.0       # collagen–bare
    bb: float = 0.0       # bare–bare


@dataclass
class McDem:
    enabled: bool = _P.mc_dem_enabled
    kappa_max: float = _P.mc_dem_kappa_max


@dataclass
class Contact:
    E_kPa: float = _P.E_modulus
    poisson_ratio: float = _P.poisson_ratio
    adhesion_energy_J_per_m2: PairTriplet = field(
        default_factory=lambda: PairTriplet(_P.W_adh_cc, _P.W_adh_cb, _P.W_adh_bb))
    friction_shear_stress_Pa: PairTriplet = field(
        default_factory=lambda: PairTriplet(_P.tau_0_cc, _P.tau_0_cb, _P.tau_0_bb))
    friction_v_ref_um_per_h: float = _P.friction_v_ref
    mc_dem: McDem = field(default_factory=McDem)
    max_overlap_frac: float = _P.max_overlap_frac
    stiffness_cap_kPa: float = _P.contact_E_cap   # 0 = none; caps the CONTACT modulus only
    friction_mu: float = _P.friction_mu           # Coulomb coefficient added to the shear-stress law (0 = off)
    overlap_model: str = _P.contact_overlap_model   # V3.2: fixed | elastic (max_overlap_frac from the contact law)
    overlap_safety: float = _P.contact_overlap_safety
    semi_implicit: bool = _P.contact_semi_implicit   # V3.2: damp the step by the local contact stiffness
    shape_dynamics: bool = _P.contact_shape_dynamics   # V3.2: shape-aware overlap projection / wall clamp
    curvature_R_cap: float = _P.curvature_R_cap        # V3.2: cap R_eff at cap*min(r_i,r_j); 0 = off
    wall_torque: bool = _P.contact_wall_torque         # V3.4: r x F at the wall contact point


@dataclass
class Dynamics:
    drag_scale: float = _P.drag_scale
    drag_scale_rot: float = _P.drag_scale_rot
    v_max_um_per_h: float = _P.v_max
    omega_max_rad_per_h: float = _P.omega_max
    omega_max_3d_rad_per_h: float = _P.omega_max_3d
    gradient_flow: str = _P.dynamics_gradient_flow   # V3.5: off | monitor | damped


@dataclass
class Packing:
    gap_um: float = _P.packing_gap
    settle_steps: int = _P.packing_settle_steps
    relax_substeps: int = _P.packing_relax_substeps
    inflate_phi_safe: float = _P.packing_inflate_phi_safe
    shape_enabled: bool = _P.shape_enabled
    consolidation: str = _P.packing_consolidation   # centre | gravity | none | auto
    shape_contact: bool = _P.packing_shape_contact   # V3.2: shape-aware settle (needs shape_enabled)
    shape_margin: float = _P.packing_shape_margin


@dataclass
class Time:
    dt_h: float = _P.dt
    t_total_h: float = _P.t_total
    save_every_h: float = _P.save_every_h


@dataclass
class Output:
    dir: str = _P.output_dir
    save_data: bool = _P.save_data
    save_fields: bool = _P.save_fields
    compress_archive: bool = _P.compress_archive
    Ngrid: int = _P.Ngrid
    Ngrid_3d: int = _P.Ngrid_3d
    interface_width_um: float = _P.interface_width
    metrics_boundary_exclusion: float = _P.boundary_exclusion
    metrics_laguerre: bool = _P.metrics_laguerre
    metrics_pore_field: bool = _P.metrics_pore_field
    keep_snaps_in_memory: bool = _P.perf_keep_snaps_in_memory


@dataclass
class Performance:
    threads: int = _P.perf_threads
    threading_layer: str = _P.perf_threading_layer
    use_numba: bool = _P.use_numba
    neighbor_backend: str = _P.perf_neighbor_backend
    cells_backend: str = _P.perf_cells_backend
    max_clips: int = _P.perf_max_clips
    field_dtype: str = _P.perf_field_dtype
    field_res_um: float = _P.perf_field_res_um
    max_grid_2d: int = _P.perf_max_grid_2d
    max_grid_3d: int = _P.perf_max_grid_3d
    metrics_voronoi_stride: int = _P.perf_metrics_voronoi_stride
    async_io: bool = _P.perf_async_io
    io_queue_depth: int = _P.perf_io_queue_depth
    io_compresslevel: int = _P.perf_io_compresslevel
    fastmath: bool = _P.perf_fastmath
    warmup: bool = _P.perf_warmup


@dataclass
class Deformable:
    enabled: bool = _P.deformable_enabled
    n_def_modes: int = _P.n_def_modes
    sdf_resolution: int = _P.sdf_resolution
    n_surface_nodes: int = _P.n_surface_nodes
    n_surface_nodes_3d: int = _P.n_surface_nodes_3d
    sdf_padding: float = _P.sdf_padding
    def_drag_scale: float = _P.def_drag_scale
    def_eps_max: float = _P.def_eps_max


@dataclass
class Setup:
    meta: Meta = field(default_factory=Meta)
    domain: Domain = field(default_factory=Domain)
    boundary: Boundary = field(default_factory=Boundary)
    gravity: Gravity = field(default_factory=Gravity)
    granules: Granules = field(default_factory=Granules)
    cells: Cells = field(default_factory=Cells)
    contact: Contact = field(default_factory=Contact)
    dynamics: Dynamics = field(default_factory=Dynamics)
    packing: Packing = field(default_factory=Packing)
    time: Time = field(default_factory=Time)
    output: Output = field(default_factory=Output)
    performance: Performance = field(default_factory=Performance)
    deformable: Deformable = field(default_factory=Deformable)

    # ── conversions ────────────────────────────────────────────────────
    def to_params(self) -> Params:
        return setup_to_params(self)

    @classmethod
    def from_params(cls, p: Params) -> 'Setup':
        return setup_from_params(p)

    @classmethod
    def from_legacy_params(cls, p: Params) -> 'Setup':
        """Two-species Setup from a V2.7-style Params (R_func_*/R_inert_*/func_ratio)."""
        q = dataclasses.replace(p, species=[])
        return setup_from_params(q)

    def to_dict(self) -> dict:
        return setup_to_dict(self)

    def validate(self) -> None:
        validate(self)


# ──────────────────────────────────────────────────────────────────────
# Flat map: dotted Setup path → Params field
# ──────────────────────────────────────────────────────────────────────
# Every non-deprecated, non-species-derived Params field appears exactly once
# (tested). Species-derived legacy fields are back-filled from the species list.

FLAT_MAP = [
    ('domain.mode', 'mode'),
    ('domain.size_um[0]', 'Lx'),
    ('domain.size_um[1]', 'Ly'),
    ('domain.size_um[2]', 'Lz'),
    ('boundary.mode', 'boundary_mode'),
    ('boundary.shape', 'boundary_shape'),
    ('boundary.top', 'boundary_top'),
    ('boundary.functionalization', 'boundary_functionalization'),
    ('boundary.layer.enabled', 'boundary_layer_enabled'),
    ('boundary.layer.radius_um', 'boundary_layer_radius'),
    ('boundary.layer.embed_frac', 'boundary_layer_embed_frac'),
    ('boundary.layer.seed_cells', 'boundary_layer_seed_cells'),
    ('gravity.enabled', 'gravity_enabled'),
    ('gravity.g_m_per_s2', 'g_accel'),
    ('gravity.medium_density_kg_m3', 'medium_density'),
    ('gravity.granule_density_kg_m3', 'granule_density'),
    ('gravity.scale', 'gravity_scale'),
    ('granules.solid_fraction', 'phi_solid_target'),
    ('granules.bed_height_um', 'bed_height'),
    ('granules.bed_solid_fraction', 'bed_phi_assumed'),
    ('granules.count_rule', 'packing_count_rule'),
    ('cells.seeding.surface_coverage', 'cell_surface_coverage'),
    ('cells.seeding.n_cells_per_granule', 'n_cells_per_granule'),
    ('cells.seeding.coverage', 'cell_coverage'),
    ('cells.seeding.law', 'cell_seeding_law'),
    ('cells.seeding.K_attach_per_um2', 'K_sigma_attach'),
    ('cells.seeding.seeding_exponent', 'cell_seeding_exponent'),
    ('cells.seeding.min_functionalization', 'f_min_adhesion'),
    ('cells.seeding.diameter_um', 'cell_diameter'),
    ('cells.seeding.spread_height_um', 'cell_height_spread'),
    ('cells.seeding.stacking_max', 'cell_stacking_max'),
    ('cells.seeding.overcrowd_senescence_time_h', 'overcrowd_senescence_time'),
    ('cells.seeding.capacity_coverage', 'cell_capacity_coverage'),
    ('cells.seeding.clock_jitter_h', 'cell_clock_jitter'),
    ('cells.seeding.capacity_foothold', 'cell_capacity_foothold'),
    ('cells.ligand.sigma_max_per_um2', 'sigma_ligand_max'),
    ('cells.ligand.K_traction_per_um2', 'K_sigma_traction'),
    ('cells.kinetics.attach_onset_h', 't_attach_onset'),
    ('cells.kinetics.attach_half_h', 't_attach_half'),
    ('cells.kinetics.spread_duration_h', 't_spread_duration'),
    ('cells.kinetics.fa_maturation_rate_per_h', 'fa_maturation_rate'),
    ('cells.motor_clutch.n_motors', 'n_motors'),
    ('cells.motor_clutch.F_motor_stall_nN', 'F_motor_stall'),
    ('cells.motor_clutch.n_clutches', 'n_clutches'),
    ('cells.motor_clutch.k_clutch_nN_per_um', 'k_clutch'),
    ('cells.motor_clutch.k_on_per_s', 'k_on_clutch'),
    ('cells.motor_clutch.k_off_per_s', 'k_off_clutch'),
    ('cells.motor_clutch.F_max_per_cell_nN', 'F_max_per_cell'),
    ('cells.motor_clutch.traction_f_law', 'traction_f_law'),
    ('cells.motor_clutch.traction_f_rule', 'traction_f_rule'),
    ('cells.motor_clutch.traction_exponent', 'traction_exponent'),
    ('cells.migration.speed_um_per_h', 'cell_migration_speed'),
    ('cells.migration.directed_speed_mult', 'bridge_directed_speed_mult'),
    ('cells.sensing.sense_distance_um', 'cell_sense_distance'),
    ('cells.sensing.bridge_decay_length_um', 'bridge_decay_length'),
    ('cells.bridging.attempt_rate_per_h', 'bridge_attempt_rate'),
    ('cells.bridging.formation_time_h', 'bridge_formation_time'),
    ('cells.bridging.senescence_time_h', 'bridge_senescence_time'),
    ('cells.bridging.min_fa', 'min_fa_for_bridge'),
    ('cells.bridging.break_gap_um', 'bridge_break_gap'),
    ('cells.bridging.lock_force_nN', 'bridge_lock_force_threshold'),
    ('cells.bridging.lock_scales_with_ligand', 'bridge_lock_scales_with_ligand'),
    ('cells.bridging.secondary_rate_mult', 'bridge_secondary_rate_mult'),
    ('cells.bridging.expected_force_nN', 'expected_bridge_force'),
    ('cells.bridging.contact_factor', 'bridge_contact_factor'),
    ('cells.bridging.bare_blocker_factor', 'bridge_inert_factor'),
    ('cells.bridging.commit_angle_rad', 'bridge_commit_angle'),
    ('cells.bridging.exclusion_angle_rad', 'bridge_exclusion_angle'),
    ('cells.bridging.alignment_rate_per_h', 'bridge_alignment_rate'),
    ('cells.bridging.alignment_min', 'bridge_alignment_min'),
    ('cells.bridging.force_model', 'bridge_force_model'),
    ('cells.bridging.contraction_speed_um_per_h', 'cell_contraction_speed'),
    ('cells.bridging.eccentric_gain', 'bridge_eccentric_gain'),
    ('cells.bridging.min_gap_um', 'bridge_min_gap'),
    ('cells.bridging.contact_adhesion_nN_per_cell', 'cell_contact_adhesion'),
    ('cells.division.enabled', 'cell_division_enabled'),
    ('cells.division.doubling_time_h', 'cell_doubling_time'),
    ('cells.division.min_age_h', 'cell_division_min_age'),
    ('cells.division.max_layers', 'cell_division_max_layers'),
    ('cells.division.divide_while_bridging', 'cell_divide_while_bridging'),
    ('cells.division.model', 'cell_division_model'),
    ('cells.division.cycle_time_cv', 'cell_division_cv'),
    ('cells.activity.T_active_nN_um', 'T_active'),
    ('contact.E_kPa', 'E_modulus'),
    ('contact.poisson_ratio', 'poisson_ratio'),
    ('contact.adhesion_energy_J_per_m2.cc', 'W_adh_cc'),
    ('contact.adhesion_energy_J_per_m2.cb', 'W_adh_cb'),
    ('contact.adhesion_energy_J_per_m2.bb', 'W_adh_bb'),
    ('contact.friction_shear_stress_Pa.cc', 'tau_0_cc'),
    ('contact.friction_shear_stress_Pa.cb', 'tau_0_cb'),
    ('contact.friction_shear_stress_Pa.bb', 'tau_0_bb'),
    ('contact.friction_v_ref_um_per_h', 'friction_v_ref'),
    ('contact.mc_dem.enabled', 'mc_dem_enabled'),
    ('contact.mc_dem.kappa_max', 'mc_dem_kappa_max'),
    ('contact.max_overlap_frac', 'max_overlap_frac'),
    ('contact.stiffness_cap_kPa', 'contact_E_cap'),
    ('contact.friction_mu', 'friction_mu'),
    ('contact.overlap_model', 'contact_overlap_model'),
    ('contact.overlap_safety', 'contact_overlap_safety'),
    ('contact.semi_implicit', 'contact_semi_implicit'),
    ('contact.shape_dynamics', 'contact_shape_dynamics'),
    ('contact.curvature_R_cap', 'curvature_R_cap'),
    ('contact.wall_torque', 'contact_wall_torque'),
    ('dynamics.drag_scale', 'drag_scale'),
    ('dynamics.drag_scale_rot', 'drag_scale_rot'),
    ('dynamics.v_max_um_per_h', 'v_max'),
    ('dynamics.omega_max_rad_per_h', 'omega_max'),
    ('dynamics.omega_max_3d_rad_per_h', 'omega_max_3d'),
    ('dynamics.gradient_flow', 'dynamics_gradient_flow'),
    ('packing.gap_um', 'packing_gap'),
    ('packing.settle_steps', 'packing_settle_steps'),
    ('packing.relax_substeps', 'packing_relax_substeps'),
    ('packing.inflate_phi_safe', 'packing_inflate_phi_safe'),
    ('packing.shape_enabled', 'shape_enabled'),
    ('packing.shape_contact', 'packing_shape_contact'),
    ('packing.shape_margin', 'packing_shape_margin'),
    ('packing.consolidation', 'packing_consolidation'),
    ('time.dt_h', 'dt'),
    ('time.t_total_h', 't_total'),
    ('time.save_every_h', 'save_every_h'),
    ('output.dir', 'output_dir'),
    ('output.save_data', 'save_data'),
    ('output.save_fields', 'save_fields'),
    ('output.compress_archive', 'compress_archive'),
    ('output.Ngrid', 'Ngrid'),
    ('output.Ngrid_3d', 'Ngrid_3d'),
    ('output.interface_width_um', 'interface_width'),
    ('output.metrics_boundary_exclusion', 'boundary_exclusion'),
    ('output.metrics_laguerre', 'metrics_laguerre'),
    ('output.metrics_pore_field', 'metrics_pore_field'),
    ('output.keep_snaps_in_memory', 'perf_keep_snaps_in_memory'),
    ('performance.threads', 'perf_threads'),
    ('performance.threading_layer', 'perf_threading_layer'),
    ('performance.use_numba', 'use_numba'),
    ('performance.neighbor_backend', 'perf_neighbor_backend'),
    ('performance.cells_backend', 'perf_cells_backend'),
    ('performance.max_clips', 'perf_max_clips'),
    ('performance.field_dtype', 'perf_field_dtype'),
    ('performance.field_res_um', 'perf_field_res_um'),
    ('performance.max_grid_2d', 'perf_max_grid_2d'),
    ('performance.max_grid_3d', 'perf_max_grid_3d'),
    ('performance.metrics_voronoi_stride', 'perf_metrics_voronoi_stride'),
    ('performance.async_io', 'perf_async_io'),
    ('performance.io_queue_depth', 'perf_io_queue_depth'),
    ('performance.io_compresslevel', 'perf_io_compresslevel'),
    ('performance.fastmath', 'perf_fastmath'),
    ('performance.warmup', 'perf_warmup'),
    ('deformable.enabled', 'deformable_enabled'),
    ('deformable.n_def_modes', 'n_def_modes'),
    ('deformable.sdf_resolution', 'sdf_resolution'),
    ('deformable.n_surface_nodes', 'n_surface_nodes'),
    ('deformable.n_surface_nodes_3d', 'n_surface_nodes_3d'),
    ('deformable.sdf_padding', 'sdf_padding'),
    ('deformable.def_drag_scale', 'def_drag_scale'),
    ('deformable.def_eps_max', 'def_eps_max'),
]

# Params fields intentionally NOT in FLAT_MAP
SPECIES_DERIVED = {
    'species', 'phi_f_target', 'phi_i_target', 'func_ratio',
    'R_func_mean', 'R_func_std', 'R_inert_mean', 'R_inert_std',
    'aspect_ratio_func_mean', 'aspect_ratio_func_std',
    'aspect_ratio_inert_mean', 'aspect_ratio_inert_std',
    'blockiness_func_mean', 'blockiness_func_std',
    'blockiness_inert_mean', 'blockiness_inert_std',
    'aspect_ratio_c_func_mean', 'aspect_ratio_c_func_std',
    'aspect_ratio_c_inert_mean', 'aspect_ratio_c_inert_std',
    'blockiness_n2_func_mean', 'blockiness_n2_func_std',
    'blockiness_n2_inert_mean', 'blockiness_n2_inert_std',
}
# Declared in Params, never read. `perf_neighbor_skin` promised a Verlet list
# ("list rebuilt when 2*max_disp >= skin") that was never implemented; V3.3
# measured what it would buy and found it NEGATIVE, so it stays unimplemented
# and is no longer reachable from a setup file. See CHANGELOG V3.3.
DEPRECATED = {'F_bond', 'L_rest', 'eta', 'perf_neighbor_skin'}
PIPELINE_MANAGED = {'resume_from'}                # set by step 3, never by a setup file


# ──────────────────────────────────────────────────────────────────────
# Dotted-path access
# ──────────────────────────────────────────────────────────────────────

_TOKEN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+)\])?')


def _split_path(path: str):
    parts = []
    for piece in path.split('.'):
        m = _TOKEN.fullmatch(piece)
        if not m:
            raise ValueError(f"bad path component {piece!r} in {path!r}")
        parts.append((m.group(1), int(m.group(2)) if m.group(2) is not None else None))
    return parts


def _walk(obj, parts):
    """Return (container, key) for the last component so callers can get/set."""
    for name, idx in parts[:-1]:
        obj = getattr(obj, name)
        if idx is not None:
            obj = obj[idx]
    return obj, parts[-1]


def get_path(setup, path: str):
    obj, (name, idx) = _walk(setup, _split_path(path))
    val = getattr(obj, name)
    return val[idx] if idx is not None else val


def set_path(setup, path: str, value) -> None:
    obj, (name, idx) = _walk(setup, _split_path(path))
    if idx is not None:
        getattr(obj, name)[idx] = value
    else:
        setattr(obj, name, value)


# ──────────────────────────────────────────────────────────────────────
# Setup ↔ Params
# ──────────────────────────────────────────────────────────────────────

def _coerce_like(current, value):
    """Coerce a parsed value to the type of the field it replaces."""
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() in ('1', 'true', 'yes', 'on')
        return bool(value)
    if isinstance(current, int) and not isinstance(current, bool):
        return int(round(float(value)))
    if isinstance(current, float):
        return float(value)
    if isinstance(current, str):
        return str(value)
    return value


def species_to_dict(sp: GranuleSpecies, solid_fraction: float) -> dict:
    """Plain-scalar species record for Params.species (JSON-safe)."""
    r = sp.radius
    rmin = r.min_um if r.min_um is not None else max(1.0, 0.25 * float(r.mean_um))
    return dict(
        name=str(sp.name), f=float(sp.functionalization),
        volume_fraction=float(sp.volume_fraction),
        phi_target=float(solid_fraction) * float(sp.volume_fraction),
        color=str(sp.color),
        radius_mean=float(r.mean_um), radius_std=float(r.std_um), radius_min=float(rmin),
        radius_distribution=str(r.distribution),
        aspect_ratio_mean=float(sp.aspect_ratio.mean), aspect_ratio_std=float(sp.aspect_ratio.std),
        aspect_ratio_c_mean=float(sp.aspect_ratio_c.mean), aspect_ratio_c_std=float(sp.aspect_ratio_c.std),
        blockiness_mean=float(sp.blockiness.mean), blockiness_std=float(sp.blockiness.std),
        blockiness_n2_mean=float(sp.blockiness_n2.mean), blockiness_n2_std=float(sp.blockiness_n2.std),
        E_kPa=None if sp.E_kPa is None else float(sp.E_kPa),
        poisson_ratio=None if sp.poisson_ratio is None else float(sp.poisson_ratio),
        count_rule=None if sp.count_rule is None else str(sp.count_rule),
        density_kg_m3=None if sp.density_kg_m3 is None else float(sp.density_kg_m3),
    )


def _species_from_dict(d: dict) -> GranuleSpecies:
    return GranuleSpecies(
        name=str(d.get('name', 'granule')),
        functionalization=float(d['f']),
        volume_fraction=float(d.get('volume_fraction', 1.0)),
        color=str(d.get('color', '#888888')),
        radius=RadiusSpec(mean_um=float(d['radius_mean']), std_um=float(d.get('radius_std', 0.0) or 0.0),
                          min_um=None if d.get('radius_min') is None else float(d['radius_min']),
                          distribution=str(d.get('radius_distribution') or 'normal')),
        aspect_ratio=MomentSpec(float(d.get('aspect_ratio_mean', 1.0)), float(d.get('aspect_ratio_std', 0.0))),
        aspect_ratio_c=MomentSpec(float(d.get('aspect_ratio_c_mean', 1.0)), float(d.get('aspect_ratio_c_std', 0.0))),
        blockiness=MomentSpec(float(d.get('blockiness_mean', 2.0)), float(d.get('blockiness_std', 0.0))),
        blockiness_n2=MomentSpec(float(d.get('blockiness_n2_mean', 2.0)), float(d.get('blockiness_n2_std', 0.0))),
        E_kPa=None if d.get('E_kPa') is None else float(d['E_kPa']),
        poisson_ratio=None if d.get('poisson_ratio') is None else float(d['poisson_ratio']),
        count_rule=None if d.get('count_rule') is None else str(d['count_rule']),
        density_kg_m3=None if d.get('density_kg_m3') is None else float(d['density_kg_m3']),
    )


def setup_to_params(setup: Setup) -> Params:
    """Flatten a Setup onto a Params (scalars via FLAT_MAP, species as dicts,
    legacy species-derived fields back-filled from the extreme-f species)."""
    validate(setup)
    p = Params()
    for path, fld in FLAT_MAP:
        set_val = get_path(setup, path)
        setattr(p, fld, _coerce_like(getattr(p, fld), set_val))

    solid = float(setup.granules.solid_fraction)
    bed_h = float(setup.granules.bed_height_um or 0.0)
    if bed_h > 0.0:
        # Amount given as a settled bed height: V_solid = phi_bed * A_base * H_bed,
        # so the container solid fraction is phi_bed * H_bed / (vertical extent);
        # the base area cancels and the engine count rule is unchanged.
        L_up = float(setup.domain.size_um[2] if setup.domain.mode in ('3D', '2D-slice')
                     else setup.domain.size_um[1])
        solid = float(setup.granules.bed_solid_fraction) * bed_h / L_up
    p.species = [species_to_dict(sp, solid) for sp in setup.granules.species]

    # Legacy back-fill so analysis modules and print_stiffness_info keep working
    f_min = p.f_min_adhesion
    adhesive = [sp for sp in setup.granules.species if sp.functionalization >= f_min]
    bare = [sp for sp in setup.granules.species if sp.functionalization < f_min]
    hi = max(setup.granules.species, key=lambda s: (s.functionalization, s.volume_fraction))
    lo = min(setup.granules.species, key=lambda s: (s.functionalization, -s.volume_fraction))
    p.func_ratio = float(sum(sp.volume_fraction for sp in adhesive)) if setup.granules.species else 0.5
    p.phi_solid_target = solid
    p.phi_f_target = solid * p.func_ratio
    p.phi_i_target = solid * (1.0 - p.func_ratio)
    p.R_func_mean, p.R_func_std = float(hi.radius.mean_um), float(hi.radius.std_um)
    p.R_inert_mean, p.R_inert_std = float(lo.radius.mean_um), float(lo.radius.std_um)
    p.aspect_ratio_func_mean, p.aspect_ratio_func_std = hi.aspect_ratio.mean, hi.aspect_ratio.std
    p.aspect_ratio_inert_mean, p.aspect_ratio_inert_std = lo.aspect_ratio.mean, lo.aspect_ratio.std
    p.aspect_ratio_c_func_mean, p.aspect_ratio_c_func_std = hi.aspect_ratio_c.mean, hi.aspect_ratio_c.std
    p.aspect_ratio_c_inert_mean, p.aspect_ratio_c_inert_std = lo.aspect_ratio_c.mean, lo.aspect_ratio_c.std
    p.blockiness_func_mean, p.blockiness_func_std = hi.blockiness.mean, hi.blockiness.std
    p.blockiness_inert_mean, p.blockiness_inert_std = lo.blockiness.mean, lo.blockiness.std
    p.blockiness_n2_func_mean, p.blockiness_n2_func_std = hi.blockiness_n2.mean, hi.blockiness_n2.std
    p.blockiness_n2_inert_mean, p.blockiness_n2_inert_std = lo.blockiness_n2.mean, lo.blockiness_n2.std
    del bare
    # V3.2: the elastic overlap rail is derived in Params.__post_init__, but this
    # function builds a default Params and then setattrs every field, so the
    # derivation would run against defaults and be overwritten. Re-derive here,
    # after the species are in place -- elastic_overlap_frac reads them.
    if getattr(p, 'contact_overlap_model', 'fixed') == 'elastic':
        from gels.engine import elastic_overlap_frac
        p.max_overlap_frac = elastic_overlap_frac(p)
    return p


def setup_from_params(p: Params) -> Setup:
    """Inverse of setup_to_params. Uses p.species when present, else the legacy pair.

    Note: a Setup always states the solid fraction explicitly, so a Params
    that gave phi_f/phi_i directly (phi_solid_target = 0) comes back with
    phi_solid_target = phi_f + phi_i and func_ratio = phi_f/(phi_f+phi_i);
    the products can differ from the originals by one ulp. The pipeline's
    legacy path therefore never round-trips through a Setup unless --set is used.
    """
    setup = Setup()
    for path, fld in FLAT_MAP:
        current = get_path(setup, path)
        set_path(setup, path, _coerce_like(current, getattr(p, fld)))
    species = resolve_species(p) if p.species else legacy_species_from_params(p)
    solid = p.phi_solid_target if p.phi_solid_target > 0 else (p.phi_f_target + p.phi_i_target)
    setup.granules.solid_fraction = float(solid)
    out = []
    for d in species:
        sp = _species_from_dict(d)
        if d.get('volume_fraction') is None and d.get('phi_target') is not None and solid > 0:
            sp.volume_fraction = float(d['phi_target']) / float(solid)
        out.append(sp)
    setup.granules.species = out   # legacy species carry count_rule='mean_radius' themselves
    return setup


# ──────────────────────────────────────────────────────────────────────
# dict / file I/O
# ──────────────────────────────────────────────────────────────────────

def setup_to_dict(setup: Setup) -> dict:
    return dataclasses.asdict(setup)


def _from_dict(cls, data, where='setup'):
    """Recursive dataclass construction with unknown-key errors."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; valid keys: {sorted(known)}")
    kwargs = {}
    for name, f in known.items():
        if name not in data:
            continue
        val = data[name]
        sub = f"{where}.{name}"
        if cls is Granules and name == 'species':
            if not isinstance(val, list):
                raise ValueError(f"{sub}: expected a list of species")
            kwargs[name] = [_species_from_yaml(v, f"{sub}[{k}]") for k, v in enumerate(val)]
            continue
        ftype = _field_dataclass(f)
        if ftype is not None and isinstance(val, dict):
            kwargs[name] = _from_dict(ftype, val, sub)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _field_dataclass(f):
    """Dataclass type of a field if it holds a (non-list) dataclass, else None."""
    default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
    return type(default) if is_dataclass(default) else None


def _species_from_yaml(v, where):
    if not isinstance(v, dict):
        raise ValueError(f"{where}: expected a mapping")
    known = {f.name for f in fields(GranuleSpecies)}
    unknown = sorted(set(v) - known)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; valid keys: {sorted(known)}")
    kw = dict(v)
    if 'radius' in kw:
        kw['radius'] = _from_dict(RadiusSpec, kw['radius'], f"{where}.radius")
    for key in ('aspect_ratio', 'aspect_ratio_c', 'blockiness', 'blockiness_n2'):
        if key in kw:
            kw[key] = _from_dict(MomentSpec, kw[key], f"{where}.{key}")
    return GranuleSpecies(**kw)


def setup_from_dict(d: dict) -> Setup:
    return _from_dict(Setup, d or {}, 'setup')


def load_setup(path: str) -> Setup:
    """Load a sectioned setup from .yaml/.yml (PyYAML) or .json."""
    ext = os.path.splitext(path)[1].lower()
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    if ext in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("Reading YAML setups needs PyYAML: pip install pyyaml") from exc
        data = yaml.safe_load(text)
    elif ext == '.json':
        data = json.loads(text)
    else:
        raise ValueError(f"setup file must be .yaml, .yml or .json, got {path!r}")
    setup = setup_from_dict(data)
    # YAML 1.1 reads bare `off` / `on` / `no` / `yes` as booleans, so
    # `gradient_flow: off` arrives as False. Normalise rather than reject: the
    # spelling is the natural one and the trap is PyYAML's, not the user's.
    if isinstance(setup.dynamics.gradient_flow, bool):
        setup.dynamics.gradient_flow = 'off' if not setup.dynamics.gradient_flow else 'monitor'
    validate(setup)
    return setup


def save_setup(setup: Setup, path: str, header=None) -> None:
    """Write the resolved setup (YAML if PyYAML is available or .json requested).

    ``header`` is a list of comment lines prepended to the YAML. PyYAML cannot
    emit per-key comments, so a written setup carries its provenance there -
    which presets were applied and what they changed; the per-key commentary
    lives in TEMPLATE_YAML.
    """
    data = setup_to_dict(setup)
    ext = os.path.splitext(path)[1].lower()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if ext == '.json':
        with open(path, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return
    try:
        import yaml
    except ImportError:
        with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        return
    with open(path, 'w', encoding='utf-8') as fh:
        for line in (header or []):
            fh.write(line if str(line).startswith('#') else '# ' + str(line))
            fh.write(chr(10))
        if header:
            fh.write(chr(10))
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)


# ──────────────────────────────────────────────────────────────────────
# Validation and overrides
# ──────────────────────────────────────────────────────────────────────

def validate(setup: Setup) -> None:
    errs = []
    if setup.domain.mode not in ('2D', '2D-slice', '3D'):
        errs.append(f"domain.mode must be 2D, 2D-slice or 3D (got {setup.domain.mode!r})")
    if len(setup.domain.size_um) != 3 or any(float(v) <= 0 for v in setup.domain.size_um):
        errs.append("domain.size_um must be three positive lengths [Lx, Ly, Lz]")
    if setup.boundary.mode not in ('walls', 'periodic'):
        errs.append(f"boundary.mode must be walls or periodic (got {setup.boundary.mode!r})")
    b = setup.boundary
    if b.shape not in ('box', 'cylinder'):
        errs.append(f"boundary.shape must be box or cylinder (got {b.shape!r})")
    if b.top not in ('wall', 'free'):
        errs.append(f"boundary.top must be wall or free (got {b.top!r})")
    if b.shape == 'cylinder':
        if setup.domain.mode != '3D':
            errs.append("boundary.shape cylinder needs domain.mode 3D")
        if b.mode != 'walls':
            errs.append("boundary.shape cylinder needs boundary.mode walls")
    if b.top == 'free' and b.mode != 'walls':
        errs.append("boundary.top free needs boundary.mode walls")
    if not (0.0 <= float(b.functionalization) <= 1.0):
        errs.append("boundary.functionalization must be in [0,1]")
    if b.layer.enabled and b.mode != 'walls':
        errs.append("boundary.layer needs boundary.mode walls")
    if float(b.layer.radius_um) < 0 or not (0.0 <= float(b.layer.embed_frac) <= 1.0):
        errs.append("boundary.layer.radius_um must be >= 0 and embed_frac in [0,1]")
    gr = setup.gravity
    if float(gr.g_m_per_s2) < 0 or float(gr.medium_density_kg_m3) <= 0 \
            or float(gr.granule_density_kg_m3) <= 0 or float(gr.scale) < 0:
        errs.append("gravity: g and scale must be >= 0, densities > 0")
    if gr.enabled and b.mode != 'walls':
        errs.append("gravity.enabled needs boundary.mode walls")
    pk = setup.packing
    if pk.consolidation not in ('centre', 'gravity', 'none', 'auto'):
        errs.append(f"packing.consolidation must be centre, gravity, none or auto (got {pk.consolidation!r})")
    if pk.consolidation == 'gravity' and b.mode != 'walls':
        errs.append("packing.consolidation gravity needs boundary.mode walls")
    g = setup.granules
    if not (0.0 < float(g.solid_fraction) < 1.0):
        errs.append(f"granules.solid_fraction must be in (0,1) (got {g.solid_fraction})")
    if float(g.bed_height_um) < 0:
        errs.append("granules.bed_height_um must be >= 0")
    if float(g.bed_height_um) > 0:
        L_up = float(setup.domain.size_um[2] if setup.domain.mode in ('3D', '2D-slice')
                     else setup.domain.size_um[1])
        if b.top != 'free':
            errs.append("granules.bed_height_um needs boundary.top free")
        if float(g.bed_height_um) > 0.85 * L_up:
            errs.append(f"granules.bed_height_um {g.bed_height_um:g} exceeds 0.85 x container height {L_up:g}; "
                        f"raise the vertical size to at least {float(g.bed_height_um) / 0.85:.0f} um")
    if not (0.3 <= float(g.bed_solid_fraction) <= 1.0):
        errs.append("granules.bed_solid_fraction must be in [0.3, 1]")
    if g.count_rule not in ('mean_volume', 'mean_radius'):
        errs.append(f"granules.count_rule must be mean_volume or mean_radius (got {g.count_rule!r})")
    if not g.species:
        errs.append("granules.species must list at least one species")
    else:
        vf = sum(float(sp.volume_fraction) for sp in g.species)
        if abs(vf - 1.0) > 1e-6:
            errs.append(f"granules.species volume_fraction values must sum to 1 (got {vf:.6f})")
        names = [sp.name for sp in g.species]
        if len(set(names)) != len(names):
            errs.append(f"granules.species names must be unique (got {names})")
        for k, sp in enumerate(g.species):
            if not (0.0 <= float(sp.functionalization) <= 1.0):
                errs.append(f"species[{k}] functionalization must be in [0,1]")
            if float(sp.radius.mean_um) <= 0 or float(sp.radius.std_um) < 0:
                errs.append(f"species[{k}] radius mean must be > 0 and std >= 0")
            if sp.radius.distribution not in ('normal', 'lognormal'):
                errs.append(f"species[{k}] radius.distribution must be normal or lognormal")
            if sp.E_kPa is not None and float(sp.E_kPa) <= 0:
                errs.append(f"species[{k}] E_kPa must be > 0")
            if sp.count_rule not in (None, 'mean_volume', 'mean_radius'):
                errs.append(f"species[{k}] count_rule must be mean_volume, mean_radius or null")
            if sp.density_kg_m3 is not None and float(sp.density_kg_m3) <= 0:
                errs.append(f"species[{k}] density_kg_m3 must be > 0")
        if len(g.species) > 8:
            errs.append(f"more than 8 species ({len(g.species)}) — rendering keeps one grid per species")
    mc = setup.cells.motor_clutch
    if mc.traction_f_law not in ('langmuir', 'power'):
        errs.append("cells.motor_clutch.traction_f_law must be langmuir or power")
    if mc.traction_f_rule not in ('target', 'min', 'product'):
        errs.append("cells.motor_clutch.traction_f_rule must be target, min or product")
    if setup.cells.seeding.law not in ('langmuir', 'power'):
        errs.append("cells.seeding.law must be langmuir or power")
    if float(setup.cells.seeding.capacity_coverage) < 0:
        errs.append("cells.seeding.capacity_coverage must be >= 0")
    br = setup.cells.bridging
    if br.force_model not in ('constant', 'hill'):
        errs.append(f"cells.bridging.force_model must be constant or hill (got {br.force_model!r})")
    if br.force_model == 'hill' and float(br.contraction_speed_um_per_h) <= 0:
        errs.append("cells.bridging.contraction_speed_um_per_h must be > 0 for the hill model")
    if float(br.eccentric_gain) < 0 or float(br.min_gap_um) < 0:
        errs.append("cells.bridging.eccentric_gain and min_gap_um must be >= 0")
    if float(br.contact_adhesion_nN_per_cell) < 0:
        errs.append("cells.bridging.contact_adhesion_nN_per_cell must be >= 0")
    dv = setup.cells.division
    if float(dv.doubling_time_h) <= 0 or float(dv.min_age_h) < 0:
        errs.append("cells.division.doubling_time_h must be > 0 and min_age_h >= 0")
    if not (0.0 < float(dv.max_layers) <= float(setup.cells.seeding.stacking_max)):
        errs.append("cells.division.max_layers must be in (0, cells.seeding.stacking_max]")
    if dv.model not in ('poisson', 'cycle'):
        errs.append("cells.division.model must be 'poisson' or 'cycle'")
    if float(dv.cycle_time_cv) < 0:
        errs.append("cells.division.cycle_time_cv must be >= 0")
    if dv.model == 'cycle' and float(dv.cycle_time_cv) <= 0:
        errs.append("cells.division.cycle_time_cv must be > 0 for the cycle model "
                    "(a zero CV leaves the population synchronised)")
    if float(setup.cells.seeding.clock_jitter_h) < 0:
        errs.append("cells.seeding.clock_jitter_h must be >= 0")
    if not (0.0 < float(setup.cells.seeding.capacity_foothold) <= 1.0):
        errs.append("cells.seeding.capacity_foothold must be in (0, 1]")
    if float(setup.contact.stiffness_cap_kPa) < 0 or float(setup.contact.friction_mu) < 0:
        errs.append("contact.stiffness_cap_kPa and friction_mu must be >= 0")
    if setup.packing.shape_contact and not setup.packing.shape_enabled:
        errs.append("packing.shape_contact needs packing.shape_enabled (it has no effect on spheres)")
    if not (0.0 <= float(setup.packing.shape_margin) <= 0.5):
        errs.append("packing.shape_margin must be in [0, 0.5]")
    if setup.packing.shape_contact and not setup.contact.shape_dynamics:
        errs.append("packing.shape_contact without contact.shape_dynamics: the packed bed "
                    "would be pulled apart by the bounding-sphere overlap projection on the "
                    "first step (measured 7 % bed expansion in 6 h with no cells)")
    if setup.contact.shape_dynamics and not setup.packing.shape_enabled:
        errs.append("contact.shape_dynamics needs packing.shape_enabled")
    if float(setup.contact.curvature_R_cap) < 0:
        errs.append("contact.curvature_R_cap must be >= 0")
    _blocky = max([max(float(sp.blockiness.mean), float(sp.blockiness_n2.mean))
                   for sp in setup.granules.species] or [2.0]) > 2.05
    if (float(setup.contact.curvature_R_cap) <= 0
            and (setup.packing.shape_contact
                 or (setup.packing.shape_enabled and _blocky))):
        # V3.4 widened this from packing.shape_contact to any blocky packing.
        # Under the support-function solver R_eff is the TRUE curvature radius
        # of the body, which at a flat face really is ~1e15 um -- where the old
        # common-normal path returned a 0.01-radian finite difference of a
        # parametric sample, i.e. noise that happened to stay bounded. Being
        # right about the geometry makes the cap load-bearing, not optional.
        errs.append("blocky granules need contact.curvature_R_cap > 0 (2.0 is the recommended "
                    "value): the local curvature radius at a flat face runs to ~1e15 um, and "
                    "F ~ sqrt(R_eff), so the contact is millions of times too stiff")
    if setup.contact.overlap_model not in ('fixed', 'elastic'):
        errs.append("contact.overlap_model must be 'fixed' or 'elastic'")
    if setup.dynamics.gradient_flow not in ('off', 'monitor', 'damped'):
        errs.append("dynamics.gradient_flow must be 'off', 'monitor' or 'damped'")
    if float(setup.contact.overlap_safety) <= 0:
        errs.append("contact.overlap_safety must be > 0")
    if not isinstance(setup.meta.presets, list) or not all(isinstance(n, str) for n in setup.meta.presets):
        errs.append("meta.presets must be a list of preset names")
    else:
        from gels.presets import PRESETS as _PRESETS
        bad = [n for n in setup.meta.presets if n not in _PRESETS]
        if bad:
            errs.append(f"meta.presets: unknown preset(s) {bad}; known: {sorted(_PRESETS)}")
    if float(setup.time.dt_h) <= 0 or float(setup.time.t_total_h) <= 0:
        errs.append("time.dt_h and time.t_total_h must be > 0")
    if float(setup.cells.ligand.sigma_max_per_um2) <= 0:
        errs.append("cells.ligand.sigma_max_per_um2 must be > 0")
    if errs:
        raise ValueError("invalid setup:\n  - " + "\n  - ".join(errs))


def parse_override(text: str):
    """'section.key[idx].sub=value' → (path, raw value string)."""
    if '=' not in text:
        raise ValueError(f"--set expects path=value, got {text!r}")
    path, value = text.split('=', 1)
    return path.strip(), value.strip()


def _parse_scalar(raw: str):
    low = raw.lower()
    if low in ('null', 'none', '~'):
        return None
    if low in ('true', 'false'):
        return low == 'true'
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw.strip('"\'')


def apply_overrides(setup: Setup, assignments) -> Setup:
    """Apply dotted-path overrides (strings 'a.b[i].c=value' or (path, value) pairs)."""
    for item in assignments or []:
        path, raw = parse_override(item) if isinstance(item, str) else item
        try:
            current = get_path(setup, path)
        except (AttributeError, IndexError, ValueError) as exc:
            raise ValueError(f"--set {path}: no such setting ({exc})") from None
        value = _parse_scalar(raw) if isinstance(raw, str) else raw
        if is_dataclass(current):
            raise ValueError(f"--set {path}: refers to a section, not a value")
        if current is None or value is None:
            coerced = value
        else:
            coerced = _coerce_like(current, value)
        set_path(setup, path, coerced)
    return setup


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────

def summarize(setup: Setup) -> str:
    d = setup.domain
    g = setup.granules
    c = setup.cells
    lines = []
    if d.mode in ('3D', '2D-slice'):
        lines.append(f"  Domain:        {d.size_um[0]:.0f} x {d.size_um[1]:.0f} x {d.size_um[2]:.0f} um ({d.mode})")
    else:
        lines.append(f"  Domain:        {d.size_um[0]:.0f} x {d.size_um[1]:.0f} um (2D)")
    b = setup.boundary
    extra = []
    if b.shape != 'box':
        extra.append(f"{b.shape} D={min(d.size_um[0], d.size_um[1]):.0f} um")
    if b.top != 'wall':
        extra.append("free top")
    if float(b.functionalization) > 0:
        extra.append(f"wall f={b.functionalization:g}")
    if b.layer.enabled:
        extra.append("boundary layer")
    lines.append(f"  Boundary:      {b.mode}" + (", " + ", ".join(extra) if extra else ""))
    gr = setup.gravity
    if gr.enabled:
        lines.append(f"  Gravity:       on (g={gr.g_m_per_s2:g} m/s^2, medium {gr.medium_density_kg_m3:g} kg/m^3, "
                     f"granules {gr.granule_density_kg_m3:g} kg/m^3 unless a species says otherwise, "
                     f"scale {gr.scale:g}); packing consolidation {setup.packing.consolidation}")
    else:
        lines.append(f"  Packing:       consolidation {setup.packing.consolidation}")
    if float(g.bed_height_um) > 0:
        L_up = float(d.size_um[2] if d.mode in ('3D', '2D-slice') else d.size_um[1])
        lines.append(f"  Bed:           {g.bed_height_um:.0f} um settled height at phi_bed {g.bed_solid_fraction:g} "
                     f"-> container solid fraction {g.bed_solid_fraction * g.bed_height_um / L_up:.3f}")
    if float(g.bed_height_um) > 0:
        L_up = float(d.size_um[2] if d.mode in ('3D', '2D-slice') else d.size_um[1])
        eff = g.bed_solid_fraction * g.bed_height_um / L_up
        lines.append(f"  Solid frac:    {eff:.3f}  (from the bed height; count rule: {g.count_rule})")
    else:
        lines.append(f"  Solid frac:    {g.solid_fraction:.3f}  (count rule: {g.count_rule})")
    lines.append("  Species:")
    lines.append(f"    {'#':>2} {'name':<18} {'colour':<8} {'f':>5} {'vol.frac':>8} {'E(kPa)':>7} {'R(um)':>12}")
    for k, sp in enumerate(g.species):
        E = sp.E_kPa if sp.E_kPa is not None else setup.contact.E_kPa
        lines.append(f"    {k:>2} {sp.name:<18} {sp.color:<8} {sp.functionalization:>5.2f} "
                     f"{sp.volume_fraction:>8.3f} {E:>7g} {sp.radius.mean_um:>5.0f} +/- {sp.radius.std_um:<4.0f}")
    lines.append(f"  Cells:         d={c.seeding.diameter_um:g} um, coverage {c.seeding.surface_coverage:g} at f=1, "
                 f"seeding {c.seeding.law} (K_attach={c.seeding.K_attach_per_um2:g}/um^2)")
    lines.append(f"  Traction:      {c.motor_clutch.n_motors} x {c.motor_clutch.F_motor_stall_nN:g} nN stall, "
                 f"cap {c.motor_clutch.F_max_per_cell_nN:g} nN, law {c.motor_clutch.traction_f_law}/"
                 f"{c.motor_clutch.traction_f_rule} (sigma_max={c.ligand.sigma_max_per_um2:g}, "
                 f"K_traction={c.ligand.K_traction_per_um2:g}/um^2)")
    lines.append(f"  Migration:     {c.migration.speed_um_per_h:g} um/h; sensing {c.sensing.sense_distance_um:g} um")
    br = c.bridging
    hill = f" (v0={br.contraction_speed_um_per_h:g} um/h, eccentric +{br.eccentric_gain:g})" \
        if br.force_model == 'hill' else ""
    dv = c.division
    div = (f"; division on: doubling {dv.doubling_time_h:g} h, min age {dv.min_age_h:g} h, "
           f"max {dv.max_layers:g} x capacity" if dv.enabled else "; division off")
    lines.append(f"  Bridges:       force model {br.force_model}{hill}{div}")
    cap = f", contact E capped at {setup.contact.stiffness_cap_kPa:g} kPa" if setup.contact.stiffness_cap_kPa > 0 else ""
    mu = f", Coulomb mu={setup.contact.friction_mu:g}" if setup.contact.friction_mu > 0 else ""
    lines.append(f"  Contact:       E={setup.contact.E_kPa:g} kPa, nu={setup.contact.poisson_ratio:g}{cap}{mu}; "
                 f"W(cc/cb/bb)={setup.contact.adhesion_energy_J_per_m2.cc:g}/"
                 f"{setup.contact.adhesion_energy_J_per_m2.cb:g}/{setup.contact.adhesion_energy_J_per_m2.bb:g} J/m^2")
    if setup.meta.presets:
        lines.append(f"  Presets:       {', '.join(setup.meta.presets)}")
    t = setup.time
    lines.append(f"  Time:          {t.t_total_h:g} h, dt={t.dt_h:g} h, save every {t.save_every_h:g} h")
    lines.append(f"  Performance:   threads={setup.performance.threads} ({setup.performance.threading_layer}), "
                 f"numba={'on' if setup.performance.use_numba else 'off'}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Commented template (PyYAML cannot emit comments, so this is a literal)
# ──────────────────────────────────────────────────────────────────────

TEMPLATE_YAML = """\
# GELS V3.2 simulation setup
# ==========================
# Fibroblasts migrating on a bed of hydrogel granules coated with Collagen-I.
# Units: micrometres, hours, nanonewtons, kilopascals unless a key says otherwise.
# Every value here is a RECOMMENDED starting point for human dermal fibroblasts on
# collagen-I microgels; provenance and typical ranges for each are recorded in
# CodeLog/References/fibroblast_parameters.md (many are literature-based — verify).
#
#   python pipeline/step1_config.py --setup setup.yaml --name my_run
#   python pipeline/step1_config.py --setup setup.yaml --set granules.species[1].functionalization=0.3

# presets (python pipeline/step0_new_setup.py --preset NAME[,NAME]; --list-presets to see them).
# step1_config.py --preset applies them after the setup file and before --set.
#   soft_gel_well         soft 10 kPa hydrogel in a 2 mm model well -- the primary system (V3.2)
#   soft_gel_dish_slice   the soft-gel well as a fast 2D dish slice (floor, two walls, free top)
#   pmma_well             rigid-bead CONTROL: PMMA in a 2 mm model well (cylinder, free top, gravity)
#   pmma_dish_slice       the rigid-bead control as a fast 2D dish slice
#   functionalized_wall   collagen-coated container with a lining cells can bridge to
#   inert_wall            bare container (the default): the bed can detach from it
#   fibroblast_realistic  Hill contraction, 80 um reach, rigid-substrate stall force, division (24 h)
#   asynchronous_cells    cells spread through the cycle at seeding, with per-cell cycle lengths
#   fragmented_granules   irregular cuboidal fragments: shape-aware packing and contact (V3.2)
#   hydrogel_box_legacy   the V3.0 defaults (closed box, no gravity, 10 kPa hydrogel, no division)

meta:
  presets: []                      # names applied with --preset (recorded for provenance; see gels/presets.py)
  notes: ""

domain:
  mode: 2D                         # 2D | 2D-slice | 3D
  size_um: [800.0, 800.0, 800.0]   # Lx, Ly, Lz (Lz used by 3D and 2D-slice); with a free top this is the CONTAINER height

boundary:
  mode: walls                      # walls | periodic
  shape: box                       # box | cylinder (3D only: axis z, R = min(Lx,Ly)/2, centre (Lx/2, Ly/2))
  top: wall                        # wall | free (open top: z is up in 3D, y is up in a 2D "dish slice")
  functionalization: 0.0           # collagen coverage f of the container wall: 0 = inert, 1 = fully coated
  layer:                           # optional lattice of immobile granules lining floor and wall (cells bridge to it)
    enabled: false
    radius_um: 0.0                 # 0 -> mean radius of the smallest mobile species
    embed_frac: 0.5                # 0.5 = hemispheres protrude into the container
    seed_cells: true

gravity:
  enabled: false                   # buoyant weight (rho_granule - rho_medium) g V; the packer sediments the bed
  g_m_per_s2: 9.81
  medium_density_kg_m3: 1000.0     # culture medium ~1000-1007
  granule_density_kg_m3: 1000.0    # default for species without density_kg_m3 (PMMA 1180, hydrogel ~1050)
  scale: 1.0                       # multiplier for numerical experiments (1 = physical)

granules:
  solid_fraction: 0.65             # total solid VOLUME fraction (AREA fraction in 2D) of the CONTAINER; 0.55-0.75 typical
  bed_height_um: 0.0               # > 0 (needs boundary.top free): amount as a settled bed height; overrides solid_fraction
  bed_solid_fraction: 0.60         # packing fraction assumed for that bed when deriving the count (3D ~0.60, 2D ~0.80)
  count_rule: mean_volume          # mean_volume (E[V] of the radius distribution) | mean_radius (V2.7 rule)
  species:                         # volume_fraction values must sum to 1
    - name: collagen_full
      functionalization: 1.0       # f: fraction of the surface coated with collagen-I (1 = fully coated)
      volume_fraction: 0.20        # share of the SOLID volume
      color: "#CC2222"
      radius: {mean_um: 40.0, std_um: 5.0, min_um: 15.0, distribution: normal}
      aspect_ratio:   {mean: 1.0, std: 0.0}   # a/b (2D) or equatorial (3D); needs packing.shape_enabled
      aspect_ratio_c: {mean: 1.0, std: 0.0}   # c/a (3D only)
      blockiness:     {mean: 2.0, std: 0.0}   # superellipse exponent n; 2 = ellipse
      blockiness_n2:  {mean: 2.0, std: 0.0}   # meridional exponent (3D only)
      E_kPa: null                  # null -> contact.E_kPa; set to mix granule stiffnesses
      poisson_ratio: null          # null -> contact.poisson_ratio
      density_kg_m3: null          # null -> gravity.granule_density_kg_m3 (PMMA 1180)
    - name: collagen_half
      functionalization: 0.5       # 50 % coated: cells keep ~94 % of their number, pull at ~86 % force
      volume_fraction: 0.40
      color: "#2255CC"
      radius: {mean_um: 40.0, std_um: 5.0, min_um: 15.0, distribution: normal}
    - name: bare
      functionalization: 0.0       # bare hydrogel: cells cannot attach or grip
      volume_fraction: 0.40
      color: "#22AA22"
      radius: {mean_um: 60.0, std_um: 8.0, min_um: 20.0, distribution: normal}

cells:                             # human dermal fibroblasts
  seeding:
    surface_coverage: 1.0          # monolayer reference at f = 1 (cells per granule = 4*pi*R^2*cov / A_cell)
    law: langmuir                  # cells per granule vs f: langmuir (K_attach) | power (seeding_exponent)
    K_attach_per_um2: 50.0         # attachment half-saturation; focal contacts appear ~50 collagen/um^2
    seeding_exponent: 1.0          # power law only
    min_functionalization: 0.05    # f below which cells cannot attach to or grip a granule
    diameter_um: 20.0              # rounded cell diameter (15-20 um)
    spread_height_um: 5.0          # spread ellipsoid height (volume conserved -> ~1257 um^2 footprint)
    stacking_max: 3.0              # layers of cells tolerated before senescence
    overcrowd_senescence_time_h: 12.0
    capacity_coverage: 0.0         # capacity for overcrowding/division; 0 -> surface_coverage (seed below it to let cells divide)
    clock_jitter_h: 0.0            # per-granule spread on attachment/spreading/FA timing (0 = whole bed matures in lockstep)
    capacity_foothold: 1.0         # fraction of the SPREAD footprint a cell needs to hold a place. 1.0 = a cell must
                                   # spread flat (V3.1); 0.25 = it may ball up to its rounded cross-section, so a 40 um
                                   # granule holds 4 rather than 1. Floored at the rounded area.
  ligand:
    sigma_max_per_um2: 750.0       # collagen-I molecules/um^2 at f = 1 (Gaudet 2003 upper point; RGD monolayer ~4e4)
    K_traction_per_um2: 160.0      # traction half-saturation (Gaudet sigma*; Mrksich-derived ~450)
  kinetics:
    attach_onset_h: 0.0
    attach_half_h: 0.0             # 0 = instant attachment
    spread_duration_h: 2.0         # 1-3 h
    fa_maturation_rate_per_h: 0.5  # focal adhesions mature in ~2 h
  motor_clutch:                    # Chan & Odde 2008
    n_motors: 200                  # 200 x 0.5 nN = 100 nN stall (50-500 nN per cell reported)
    F_motor_stall_nN: 0.5
    n_clutches: 75
    k_clutch_nN_per_um: 5.0
    k_on_per_s: 1.0
    k_off_per_s: 0.1
    F_max_per_cell_nN: 150.0       # cap; co-tune with bridging.lock_force_nN and expected_force_nN
    traction_f_law: langmuir       # traction vs coating: langmuir (ligand.K_traction) | power (traction_exponent)
    traction_f_rule: target        # which f a bridging cell grips with: target | min | product
    traction_exponent: 1.0         # power law only
  migration:
    speed_um_per_h: 30.0           # 12-60 um/h on collagen (0.2-1 um/min)
    directed_speed_mult: 2.0       # crawl speed multiplier toward a bridge target
  sensing:
    sense_distance_um: 40.0        # whole-cell reach (40-100 um for elongated fibroblasts)
    bridge_decay_length_um: 10.0   # filopodial reach across a gap (2-10 um)
  bridging:
    attempt_rate_per_h: 0.3
    formation_time_h: 2.0
    senescence_time_h: 24.0
    min_fa: 0.3
    break_gap_um: 60.0
    lock_force_nN: 20.0
    lock_scales_with_ligand: true  # sustainable cluster force is linear in bond number (Erdmann & Schwarz 2004)
    secondary_rate_mult: 3.0
    expected_force_nN: 100.0
    contact_factor: 5.0
    bare_blocker_factor: 0.1       # penalty when an f = 0 granule blocks the line of sight (f = 1 -> none)
    commit_angle_rad: 0.5
    exclusion_angle_rad: 0.8
    alignment_rate_per_h: 0.3
    alignment_min: 0.3
    force_model: constant          # constant (V2.7 actuator) | hill (force-velocity: holds the stall force at force balance)
    contraction_speed_um_per_h: 12.0   # hill only: unloaded shortening speed v0 (0.1-0.5 um/min)
    eccentric_gain: 0.5            # hill only: extra force while a bridge is stretched (up to 1 + gain)
    min_gap_um: 0.0                # hill only: gap below which active shortening stops (isometric hold)
    contact_adhesion_nN_per_cell: 0.0   # shear resistance a mature bridging cell adds to THAT contact (5-50 nN)
  division:
    enabled: false
    doubling_time_h: 24.0          # human dermal fibroblasts 20-30 h
    min_age_h: 8.0                 # refractory period after birth / last division
    max_layers: 1.0                # contact inhibition: divide only while n_cells < capacity x max_layers
    divide_while_bridging: true    # bridging cells round up transiently and re-spread; false -> ~no division
    model: poisson                 # poisson = memoryless (V3.1); cycle = each cell has its own cycle length
    cycle_time_cv: 0.0             # lognormal CV of that cycle length (cycle model needs > 0; 0.15 is typical)
  activity:
    T_active_nN_um: 5.0            # active noise temperature, scaled per granule by the seeding gain

contact:                           # hydrogel granules
  E_kPa: 10.0                      # GelMA 2-30 kPa, PEG/alginate 1-100 kPa (species may override)
  poisson_ratio: 0.45              # 0.40-0.49 (0.49 makes MC-DEM confinement very stiff)
  adhesion_energy_J_per_m2: {cc: 0.002, cb: 0.001, bb: 0.0005}   # collagen-collagen / collagen-bare / bare-bare
  friction_shear_stress_Pa:  {cc: 2000.0, cb: 500.0, bb: 50.0}   # bare gel 10-100 Pa (Gong 2006, Pitenis 2014)
  friction_v_ref_um_per_h: 1.0
  mc_dem: {enabled: true, kappa_max: 5.0}
  max_overlap_frac: 0.15           # standing overlap allowed before the projection acts; phi ceiling = (2/(2-f))^3
  overlap_model: fixed             # fixed = use max_overlap_frac as given; elastic = derive it from the contact law
  overlap_safety: 1.5              # elastic only: headroom over the overlap at which a contact carries expected_force_nN
  semi_implicit: true              # damp each step by the local contact stiffness (stable at any E; same fixed point)
  shape_dynamics: false            # shape-aware overlap projection (REQUIRED with packing.shape_contact, or the bed relaxes apart)
  curvature_R_cap: 2.0             # cap R_eff at this x min(r_i,r_j); 0 = off. REQUIRED for blocky shapes (2.0): a flat
                                   # face has an almost infinite curvature radius and F ~ sqrt(R_eff)
  stiffness_cap_kPa: 0.0           # 0 = none; caps the CONTACT modulus only (cells see the true E); PMMA preset 100
  wall_torque: true                # apply r x F at the wall contact point (V3.4). Free -- the support function
                                   #   returns the point. Identically zero for spheres (their wall contact is on
                                   #   the centre line), so this is "on for non-spherical granules".
  friction_mu: 0.0                 # Coulomb coefficient added to the shear-stress friction (0 = hydrogel law only)

dynamics:
  drag_scale: 0.05                 # calibrate to 24-72 h compaction
  drag_scale_rot: 0.05
  v_max_um_per_h: 20.0
  omega_max_rad_per_h: 1.0
  omega_max_3d_rad_per_h: 1.0
  gradient_flow: "off"             # off | monitor | damped -- audit dE/dt <= 0 (V3.5)

packing:
  gap_um: 0.0
  settle_steps: 400
  relax_substeps: 15
  inflate_phi_safe: 0.20
  shape_enabled: false             # true -> superellipse/superellipsoid granules from the species moments
  shape_contact: false             # shape-aware settle: true bounding radius, directional overlap, rotation to nest
  shape_margin: 0.0                # inflate every directional radius by (1+margin) (blunt; costs reachable phi)
  consolidation: auto              # centre (V2.7 pull toward the box centre) | gravity | none | auto (gravity if enabled, else none)

time:
  dt_h: 0.5
  t_total_h: 72.0
  save_every_h: 2.0

output:
  dir: results/default             # overridden by the pipeline run directory
  save_data: true
  save_fields: false               # phase-field grids are rebuilt from particles at plot time
  compress_archive: true
  Ngrid: 200
  Ngrid_3d: 80
  interface_width_um: 3.0
  metrics_boundary_exclusion: 0.2
  metrics_laguerre: false          # halo-free local packing fraction; ~0.3-1 s/frame at N=1000
  metrics_pore_field: false        # Katz-Thompson permeability + geodesic tortuosity (full grid)
  keep_snaps_in_memory: true

performance:
  threads: 0                       # numba threads: 0 = auto (physical cores), -1 = all logical CPUs
  threading_layer: omp             # omp | tbb | workqueue
  cells_backend: kernels           # kernels (compiled cell state machine + bridging, hashed RNG) | python (exact V2.7 cell code)
  use_numba: true
  neighbor_backend: cells          # cells | ckdtree | reference
  max_clips: 16
  field_dtype: float32
  field_res_um: 0.0                # 0 -> 2 x interface_width
  max_grid_2d: 4096
  max_grid_3d: 256
  metrics_voronoi_stride: 4
  async_io: true
  io_queue_depth: 2
  io_compresslevel: 1
  fastmath: true
  warmup: true

deformable:                        # LS-DEM deformable granules (deferred; runs on the reference path)
  enabled: false
  n_def_modes: 2
  sdf_resolution: 32
  n_surface_nodes: 128
  n_surface_nodes_3d: 512
  sdf_padding: 1.3
  def_drag_scale: 0.1
  def_eps_max: 0.3
"""


def template_setup() -> Setup:
    """The template parsed into a Setup (requires PyYAML)."""
    import yaml
    return setup_from_dict(yaml.safe_load(TEMPLATE_YAML))


__all__ = [
    'Setup', 'Meta', 'Domain', 'Boundary', 'BoundaryLayer', 'Gravity', 'Granules', 'GranuleSpecies',
    'RadiusSpec', 'MomentSpec', 'Division',
    'Cells', 'Seeding', 'Ligand', 'Kinetics', 'MotorClutch', 'Migration', 'Sensing', 'Bridging',
    'Activity', 'Contact', 'PairTriplet', 'McDem', 'Dynamics', 'Packing', 'Time', 'Output',
    'Performance', 'Deformable',
    'FLAT_MAP', 'SPECIES_DERIVED', 'DEPRECATED', 'PIPELINE_MANAGED',
    'load_setup', 'save_setup', 'setup_from_dict', 'setup_to_dict', 'setup_to_params',
    'setup_from_params', 'species_to_dict', 'validate', 'apply_overrides', 'parse_override',
    'get_path', 'set_path', 'summarize', 'TEMPLATE_YAML', 'template_setup',
]
