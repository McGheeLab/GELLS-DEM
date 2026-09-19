"""
STEP 0 (optional) - Write a setup file to start from.
=====================================================

Creates a sectioned YAML setup — domain, boundary, granule species, cells,
contact mechanics, packing, time, output, performance — that you edit and
then hand to step 1 with ``--setup``. Three sources:

    template     the fully commented V3.0 template with recommended physical
                 values for fibroblasts on collagen-I microgels (default)
    --from-trial convert a legacy Trials/*.json (two species: coated + bare)
    --from-run   reconstruct the setup of an existing run from its params.json

Examples
--------
    python pipeline/step0_new_setup.py                       # writes ./setup.yaml
    python pipeline/step0_new_setup.py -o setups/stiff3d.yaml
    python pipeline/step0_new_setup.py --from-trial Trials/DOE2_2D_0001.json -o setups/doe1.yaml
    python pipeline/step0_new_setup.py --from-run results/my_run -o setups/rerun.yaml
    python pipeline/step0_new_setup.py -o setup.yaml --force  # overwrite

Then:   python pipeline/step1_config.py --setup setup.yaml --name my_run
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import REPO_ROOT, load_trial_json, params_from_run_dir  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 0: write a sectioned setup file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('Examples')[-1])
    parser.add_argument('-o', '--output', default='setup.yaml',
                        help='Where to write the setup (default: ./setup.yaml)')
    src = parser.add_mutually_exclusive_group()
    src.add_argument('--from-trial', type=str, default=None,
                     help='Convert a legacy Trials/*.json into a two-species setup')
    src.add_argument('--from-run', type=str, default=None,
                     help='Reconstruct the setup of an existing run directory')
    parser.add_argument('--preset', action='append', default=[],
                        help='Named preset(s) of overrides to apply, e.g. '
                             '--preset pmma_well,fibroblast_realistic (repeatable; later ones win)')
    parser.add_argument('--list-presets', action='store_true',
                        help='List the available presets and exit')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing file')
    args = parser.parse_args()

    if args.list_presets:
        from gels.presets import PRESET_DESCRIPTIONS
        print("\nPresets (--preset NAME[,NAME]; applied in order, later ones win):\n")
        for name, desc in PRESET_DESCRIPTIONS.items():
            print(f"  {name:22s} {desc}")
        print()
        return 0

    out = args.output
    if os.path.exists(out) and not args.force:
        print(f"\n  {out} already exists. Use --force to overwrite.\n")
        return 1

    from gels.config import Setup, TEMPLATE_YAML, save_setup, summarize
    from gels.engine import Params

    print('=' * 68)
    print("  GELS pipeline - STEP 0: new setup file")
    print('=' * 68)

    if args.from_trial:
        path = args.from_trial
        if not os.path.exists(path):
            print(f"\n  ERROR: trial file not found: {path}")
            return 1
        p = Params()
        for k, v in load_trial_json(path).items():
            if hasattr(p, k):
                try:
                    setattr(p, k, type(getattr(p, k))(v))
                except (ValueError, TypeError):
                    setattr(p, k, v)
        if p.phi_solid_target > 0:
            p.phi_f_target = p.phi_solid_target * p.func_ratio
            p.phi_i_target = p.phi_solid_target * (1.0 - p.func_ratio)
        setup = Setup.from_legacy_params(p)
        save_setup(setup, out)
        print(f"  Source:  legacy trial {path} → two species (collagen f=1, bare f=0)")
    elif args.from_run:
        p = params_from_run_dir(args.from_run)
        setup = Setup.from_params(p)
        save_setup(setup, out)
        print(f"  Source:  run directory {args.from_run}")
    else:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(TEMPLATE_YAML)
        print("  Source:  commented V3.1 template (recommended fibroblast / collagen-I values)")
        try:
            import yaml  # noqa: F401
            from gels.config import template_setup
            setup = template_setup()
        except ImportError:
            setup = None

    # ── V3.1 presets: bundles of correlated overrides (container + material + cells) ──
    if args.preset:
        from gels.presets import apply_presets, preset_deltas, preset_names
        if setup is None:
            print("\n  ERROR: --preset needs PyYAML (the setup has to be parsed to be edited)")
            return 1
        import copy as _copy
        deltas = preset_deltas(_copy.deepcopy(setup), args.preset)
        apply_presets(setup, args.preset)
        names = preset_names(args.preset)
        header = [f"GELS setup written by step0 with presets: {', '.join(names)}",
                  "Per-key documentation: gels/config.py TEMPLATE_YAML "
                  "(PyYAML cannot emit comments, so this file has none).",
                  ""]
        header += [f"{n}: {path}: {old!r} -> {new!r}" for n, path, old, new in deltas]
        save_setup(setup, out, header=header)
        print(f"  Presets: {', '.join(names)} ({len(deltas)} values changed)")

    print(f"  Wrote:   {os.path.abspath(out)}")
    if setup is not None:
        print()
        print(summarize(setup))
    print(f"\n  Edit the file, then:\n\n      python pipeline/step1_config.py --setup {out} --name my_run\n")
    return 0


if __name__ == '__main__':
    sys.exit(main())
