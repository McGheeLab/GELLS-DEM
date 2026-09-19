"""
STEP 1 of 5 - Configure a run.
==============================

Resolves simulation parameters and writes them to ``<run_dir>/params.json``
(the flat form the engine reads) and ``<run_dir>/setup.yaml`` (the sectioned
form for humans), creating the run directory and its ``pipeline_state.json``.
No simulation work happens here - this step is cheap and safe to re-run while
you settle on parameters.

Two ways to describe a run:

  * a **sectioned setup file** (V3.0; write one with step0_new_setup.py) —
    domain, boundary, granule species, cells, contact, packing, time, output:

        --setup setup.yaml [--set section.key=value ...]

  * a **legacy trial** or plain flags — the V2.x flat Params vocabulary, which
    becomes a two-species run (collagen f=1 / bare f=0):

        --trial Trials/DOE2_2D_0001.json [--E_modulus 5.0 ...]

Precedence (lowest first): Params defaults → setup/trial file → ``--set``
overrides → individual flat flags (one exists per Params field).

Examples
--------
    python pipeline/step1_config.py --setup setup.yaml --name my_run
    python pipeline/step1_config.py --setup setup.yaml --name f30 --set granules.species[1].functionalization=0.3
    python pipeline/step1_config.py --trial Trials/DOE2_2D_0001.json
    python pipeline/step1_config.py --name stiff3d --mode 3D --E_modulus 20 --t_total 48
    python pipeline/step1_config.py --name my_run --force --t_total 96
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402  (repo root is bootstrapped by _common)
    add_params_args, apply_params_args, banner, done, guard, load_trial_json,
    mark_done, print_progress, save_params,
)


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 1: configure a run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Examples')[-1])
    parser.add_argument('--setup', type=str, default=None,
                        help='Sectioned YAML/JSON setup file (write one with step0_new_setup.py)')
    parser.add_argument('--trial', type=str, default=None,
                        help='Legacy Trial JSON config (flat or nested format)')
    parser.add_argument('--preset', action='append', default=[],
                        help='Named preset(s) of setup overrides, applied after the setup file '
                             'and before --set (see gels/presets.py; step0 --list-presets)')
    parser.add_argument('--set', dest='overrides', action='append', default=[],
                        metavar='PATH=VALUE',
                        help='Override a setup value by dotted path, e.g. '
                             'granules.species[1].functionalization=0.5 (repeatable)')
    parser.add_argument('--name', type=str, default=None,
                        help='Run name; output goes to results/<name>')
    parser.add_argument('-o', '--run-dir', type=str, default=None,
                        help='Explicit run directory (overrides --name)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for packing and stochastic cell '
                             'processes (default: 42)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing configuration')
    add_params_args(parser)
    args = parser.parse_args()

    if args.setup and args.trial:
        print("\n  ERROR: --setup and --trial are mutually exclusive.")
        sys.exit(1)

    # ── Resolve the run directory ──
    if args.run_dir:
        run_dir = args.run_dir
    else:
        name = args.name
        if name is None:
            src = args.setup or args.trial
            name = os.path.splitext(os.path.basename(src))[0] if src else 'default'
        run_dir = os.path.join('results', name)

    banner('config', run_dir)
    guard(run_dir, 'config', args.force)

    from gels.engine import Params, print_stiffness_info
    from gels.config import Setup, apply_overrides, load_setup, save_setup, summarize

    setup = None
    trial_applied = {}
    if args.setup:
        # ── V3.0 sectioned setup: file → --set → flat flags ──
        if not os.path.exists(args.setup):
            print(f"\n  ERROR: setup file not found: {args.setup}")
            sys.exit(1)
        setup = load_setup(args.setup)
        if args.preset:
            from gels.presets import apply_presets
            setup = apply_presets(setup, args.preset)     # presets, then --set: --set wins
        setup = apply_overrides(setup, args.overrides)
        p = setup.to_params()
        print(f"  Setup file:    {args.setup} ({len(setup.granules.species)} species)")
        cli_applied = apply_params_args(p, args)
        if cli_applied:
            setup = Setup.from_params(p)      # keep the human-readable copy in sync
    else:
        # ── Legacy path: defaults → trial JSON → flat flags (unchanged from V2.7) ──
        p = Params()
        if args.trial:
            if not os.path.exists(args.trial):
                print(f"\n  ERROR: trial file not found: {args.trial}")
                sys.exit(1)
            trial_applied = load_trial_json(args.trial)
            for k, v in trial_applied.items():
                if hasattr(p, k):
                    try:
                        setattr(p, k, type(getattr(p, k))(v))
                    except (ValueError, TypeError):
                        setattr(p, k, v)
            print(f"  Trial file:    {args.trial} "
                  f"({len(trial_applied)} parameters)")
        cli_applied = apply_params_args(p, args)
        if args.overrides or args.preset:
            # Dotted overrides on a legacy configuration go through the
            # sectioned view (two species) and back.
            setup = Setup.from_params(p)
            if args.preset:
                from gels.presets import apply_presets
                setup = apply_presets(setup, args.preset)
            setup = apply_overrides(setup, args.overrides)
            p = setup.to_params()

    for k in ('output_dir', 'resume_from'):
        cli_applied.pop(k, None)

    # ── Pipeline-managed fields ──
    # The run directory is the single source of truth for output; resume is
    # driven by step 3, not by params.json.
    p.output_dir = run_dir
    p.resume_from = ""
    p.save_data = True

    # Re-derive composition targets if a solid fraction was given (mirrors
    # the same derivation inside run(), which we bypass by stepping).
    if p.phi_solid_target > 0:
        p.phi_f_target = p.phi_solid_target * p.func_ratio
        p.phi_i_target = p.phi_solid_target * (1.0 - p.func_ratio)

    if args.preset:
        from gels.presets import preset_names
        print(f"  Presets:       {', '.join(preset_names(args.preset))}")
    if args.overrides:
        print(f"  --set:         {len(args.overrides)}")
        for item in args.overrides:
            print(f"    {item}")
    if cli_applied:
        print(f"  CLI overrides: {len(cli_applied)}")
        for k, v in sorted(cli_applied.items()):
            print(f"    {k} = {v}")

    # ── Report the resolved configuration (sectioned view) ──
    setup_out = setup if setup is not None else Setup.from_params(p)
    print()
    print(summarize(setup_out))
    n_steps = int(p.t_total / p.dt)
    n_snaps = n_steps // p.save_every + 1
    print(f"  Steps:         {n_steps} steps, ~{n_snaps} snapshots")
    print(f"  Deformable:    {'LS-DEM on' if p.deformable_enabled else 'rigid'}")
    print(f"  Seed:          {args.seed}")
    print_stiffness_info(p)

    # ── Write configuration ──
    save_params(p, run_dir)
    setup_path = os.path.join(run_dir, 'setup.yaml')
    try:
        save_setup(setup_out, setup_path)
    except Exception as exc:  # the flat params.json is authoritative; the YAML is a courtesy copy
        print(f"  (could not write {setup_path}: {exc})")
    mark_done(run_dir, 'config', {
        'seed': args.seed,
        'setup': args.setup or '(none)',
        'trial': args.trial or '(none)',
        'n_species': len(setup_out.granules.species),
        'mode': p.mode,
        't_total': p.t_total,
        'n_set_overrides': len(args.overrides),
        'n_cli_overrides': len(cli_applied),
    })

    print(f"\n  Wrote {os.path.join(run_dir, 'params.json')}")
    print(f"  Wrote {setup_path}")
    print_progress(run_dir)
    done('config', run_dir)


if __name__ == '__main__':
    main()
