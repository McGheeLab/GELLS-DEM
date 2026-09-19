"""
STEP 2 of 5 - Generate the initial granule packing.
===================================================

Builds the jammed packing for the configured run and saves it as snapshot
0000, which step 3 then resumes from. Concretely this step:

    1. reads <run_dir>/params.json
    2. runs mode-aware RSA + Lubachevsky-Stillinger inflate-and-relax
    3. seeds cells on the functional granules and initialises LS-DEM state
    4. evaluates forces and metrics at t = 0
    5. writes snapshots/snap_0000.npz, history.json, metadata.json

Packing is the expensive non-dynamical part of a run, so isolating it here
means you can inspect or re-use a packing without paying for it again, and
re-run the simulation (step 3) against the exact same initial condition.

Examples
--------
    python pipeline/step2_pack.py -i results/my_run
    python pipeline/step2_pack.py -i results/my_run --force
"""

import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    banner, done, guard, mark_done, params_from_run_dir, print_progress,
    require, save_params, seed_for,
)


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 2: generate the initial packing.")
    parser.add_argument('-i', '--run-dir', required=True,
                        help='Run directory created by step 1')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the seed recorded at step 1')
    parser.add_argument('--force', action='store_true',
                        help='Discard an existing packing and rebuild it')
    args = parser.parse_args()

    run_dir = args.run_dir
    banner('pack', run_dir)
    require(run_dir, 'config')
    guard(run_dir, 'pack', args.force)

    import numpy as np
    from gels.engine import (
        bed_solid_volume, bed_surface, boundary_geometry, consolidation_mode,
        compute_displacement, compute_displacement_species, compute_forces,
        compute_forces_3d, compute_metrics, generate_packing, handoff_force_balance,
        generate_packing_2d_slice, generate_packing_3d, group_species_fields,
        render_fields_species,
        save_history_to_disk, save_params_metadata, save_snapshot_to_disk,
        update_cell_state,
    )

    p = params_from_run_dir(run_dir)
    p.output_dir = run_dir
    p.resume_from = ""
    seed = args.seed if args.seed is not None else seed_for(run_dir, 42)
    rng = np.random.default_rng(seed)

    # A rebuild must not leave stale snapshots from a previous packing behind,
    # otherwise step 3 would resume from the wrong (later) state.
    snap_dir = os.path.join(run_dir, 'snapshots')
    if args.force and os.path.isdir(snap_dir):
        shutil.rmtree(snap_dir)
        for stale in ('history.json', 'history.csv'):
            path = os.path.join(run_dir, stale)
            if os.path.exists(path):
                os.remove(path)
        print("  Cleared previous packing and snapshots")

    print(f"  Mode:          {p.mode}")
    print(f"  Seed:          {seed}")
    print("\n  Generating packing (RSA + inflate-and-relax)...")
    t0 = time.time()

    if p.mode == "3D":
        gs = generate_packing_3d(p, seed=seed)
    elif p.mode == "2D-slice":
        gs = generate_packing_2d_slice(p, seed=seed)
    else:
        gs = generate_packing(p, seed=seed)

    if p.deformable_enabled:
        from gels.lsdem import init_lsdem
        print("  Initialising LS-DEM deformable particles...")
        init_lsdem(gs, p)

    # ── Evaluate the t = 0 state exactly as run() does for a fresh start ──
    update_cell_state(gs, p, 0.0, rng)
    if gs.is_3d:
        F0, _, contacts0 = compute_forces_3d(gs, p, rng)
    else:
        F0, _, contacts0 = compute_forces(gs, p, rng)
    handoff_force_balance(gs, p, F0)     # V3.4: warn if the packer over-loaded the bed
    phi_s = render_fields_species(gs, p)
    phi_f, phi_i, phi_v = group_species_fields(gs, phi_s)

    m = compute_metrics(gs, p, phi_f, phi_i, phi_v, 0.0, F0, phi_s=phi_s)
    x0 = gs.x.copy()
    y0 = gs.y.copy()
    z0 = gs.z.copy() if gs.is_3d else None
    disp_f, disp_i = compute_displacement(gs, x0, y0, z0)
    m['disp_func'] = disp_f
    m['disp_inert'] = disp_i
    m.update(compute_displacement_species(gs, x0, y0, z0))

    elapsed = time.time() - t0

    # ── Persist: params + metadata + snapshot 0000 + one-entry history ──
    save_params(p, run_dir)
    save_params_metadata(p, gs, run_dir, seed)
    save_snapshot_to_disk(0, gs, p, 0.0, F0, run_dir,
                          save_fields=p.save_fields,
                          phi_f=phi_f, phi_i=phi_i, phi_v=phi_v,
                          contacts=contacts0)
    save_history_to_disk([m], run_dir)

    # ── Report ──
    n_func = int(np.sum(gs.func_mask))
    n_inert = int(np.sum(gs.inert_mask))
    n_contacts = int(m.get('n_contacts', 0))
    # compute_metrics reports contact counts, not coordination; Z follows
    # directly as each contact is shared by two granules.
    Z = 2.0 * n_contacts / gs.N if gs.N else float('nan')

    print(f"\n  Granules:      {gs.N} total ({n_func} adhesive, {n_inert} bare)")
    for k in range(gs.K):
        mk = gs.species_id == k
        print(f"    species {k}:  {int(np.sum(mk)):>5d} x {gs.species_names[k]:<16s} "
              f"f={gs.species_f[k]:.2f}  E={gs.species_E[k]:g} kPa  "
              f"cells={int(np.sum(gs.n_cells[mk]))}  colour {gs.species_colors[k]}")
    print(f"  Cells:         {int(gs.total_cells)} seeded")
    print(f"  Contacts:      {n_contacts}  (ff={m.get('n_contacts_ff', 0)}, "
          f"if={m.get('n_contacts_if', 0)}, ii={m.get('n_contacts_ii', 0)})")
    print(f"  Coordination:  Z = {Z:.2f}")
    # phi_solid/porosity are sampled over the inner region only (boundary
    # exclusion), so they are not directly comparable to the domain-wide
    # phi_f_target + phi_i_target printed during placement above.
    print(f"  Solid frac:    {m.get('phi_solid', float('nan')):.3f}  "
          f"(inner region, boundary_exclusion={p.boundary_exclusion:g})")
    print(f"  Porosity:      {m.get('porosity', float('nan')):.3f}")
    print(f"  Permeability:  K = {m.get('K_kozeny_carman', float('nan')):.1f} um^2")
    # V3.1: settled-bed report for open containers / gravity consolidation
    geom = boundary_geometry(p)
    bed_info = {}
    if consolidation_mode(p) == 'gravity' or geom.top_free or geom.shape_code == 1:
        bed = bed_surface(gs, p)
        V_solid = bed_solid_volume(gs)
        env = bed['bed_envelope_volume']
        phi_bed = V_solid / env if env > 0 else float('nan')
        n_fixed = int(np.sum(gs.fixed))
        print(f"  Bed:           height {bed['bed_height_mean']:.0f} um (p95 {bed['bed_height_p95']:.0f}), "
              f"envelope solid fraction {phi_bed:.3f}, {n_fixed} boundary granules; "
              f"consolidation {consolidation_mode(p)}")
        bed_info = {'bed_height_um': round(float(bed['bed_height_mean']), 1),
                    'bed_height_p95_um': round(float(bed['bed_height_p95']), 1),
                    'phi_bed': round(float(phi_bed), 4), 'n_boundary': n_fixed}
    print(f"\n  Wrote snapshot 0000 to {snap_dir}")

    mark_done(run_dir, 'pack', {
        'seed': seed,
        'n_granules': int(gs.N),
        'n_functional': n_func,
        'n_inert': n_inert,
        'total_cells': int(gs.total_cells),
        'species': [{'name': gs.species_names[k], 'f': float(gs.species_f[k]),
                     'n_granules': int(np.sum(gs.species_id == k)),
                     'n_cells': int(np.sum(gs.n_cells[gs.species_id == k]))}
                    for k in range(gs.K)],
        'n_contacts': n_contacts,
        'Z': round(float(Z), 3),
        'phi_solid': round(float(m.get('phi_solid', float('nan'))), 4),
        'consolidation': consolidation_mode(p),
        **bed_info,
        'pack_seconds': round(elapsed, 1),
    })

    print_progress(run_dir)
    done('pack', run_dir, elapsed)


if __name__ == '__main__':
    main()
