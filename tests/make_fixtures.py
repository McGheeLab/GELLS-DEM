"""
Generate V2.7 reference fixtures for the V3.0 refactor.
=======================================================

Run this ONCE on the pre-refactor (V2.7) code and keep the outputs under
``tests/fixtures/``. The regression tests re-run the same configurations on
the current code and require bit-identical results for the Python path
(Phases 0-4 of the V3.0 plan), so any behaviour change is caught immediately.

Contents written:

    tests/fixtures/manifest.json                 versions, seed, configurations
    tests/fixtures/params_v27_<trial>.json       params.json as produced by step1_config.py --trial
    tests/fixtures/<run_name>/                   5-step reference run: params.json, metadata.json,
                                                 history.json, snapshots/snap_0000..0005.npz, fields/

A SECOND, PLATFORM-LOCAL baseline lives under ``tests/fixtures_local/``.
The V2.7 fixtures above were generated on Windows / CPython 3.14.7 / numpy
2.5.3 and are NOT reproducible bit-for-bit on another platform: packing
positions differ by ~1e-13 (ARM vs x86 libm, different numpy reduction
order), and in ``run3d_spheres`` that flips a discrete bridge-formation
decision at t = 1.0, after which the trajectories bifurcate to O(1). No
tolerance can bridge that, so ``test_legacy_identity`` skips when the
platform fingerprint does not match, and ``test_local_identity`` pins
behaviour against a baseline blessed on THIS machine instead.

Usage:
    python tests/make_fixtures.py                       # refuses to overwrite
    python tests/make_fixtures.py --local               # write tests/fixtures_local/
    python tests/make_fixtures.py --force --i-really-mean-it
                                # regenerate the V2.7 oracle. This DESTROYS an
                                # irreplaceable record: the V2.7 code is no
                                # longer in the tree, so the fixtures ARE the
                                # only copy of its behaviour. Almost never right.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

os.environ.setdefault('MPLBACKEND', 'Agg')
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

FIXTURES = os.path.join(REPO, 'tests', 'fixtures')
FIXTURES_LOCAL = os.path.join(REPO, 'tests', 'fixtures_local')
SEED = 7


def platform_fingerprint():
    """What has to match for an exact-equality fixture comparison to be meaningful.

    Deliberately coarse: the machine/ABI, the CPython minor version and the
    numpy minor version. Those are what change float reduction order and libm
    results; the patch levels do not.
    """
    import numpy
    return {
        'machine': platform.machine(),
        'system': platform.system(),
        'python': '.'.join(platform.python_version_tuple()[:2]),
        'numpy': '.'.join(numpy.__version__.split('.')[:2]),
    }

# Small, fast configurations that still exercise every code path the refactor
# touches: walls vs periodic, circles vs superellipses, 2D vs 3D, and both
# cell-seeding rules (n_cells_per_granule fallback vs cell_surface_coverage).
COMMON = dict(
    t_total=2.5, dt=0.5, save_every_h=0.5,   # 5 steps, a snapshot every step
    Ngrid=64, save_data=True, save_fields=True, compress_archive=False,
    # ── V2.7 defaults that later versions changed, pinned here ──
    # The fixtures exist to reproduce V2.7, so a deliberate DEFAULT change must
    # be pinned rather than superseded whenever the old code path still exists.
    # Pinning keeps the oracle testing everything else at atol = 0; superseding
    # would retire it wholesale. The shipping defaults are covered instead by
    # LOCAL_ONLY_RUNS below, so neither configuration goes unguarded.
    contact_semi_implicit=False,             # V3.4 flipped this to True
    boundary_wall_clamp='legacy',            # V3.5 moved the clip inside the wall
    packing_relax='none',                    # V3.5 made this 'auto' (= fire under a load)
    dynamics_gradient_flow='off',            # V3.5 added it; 'off' is also its default
)
REFERENCE_RUNS = {
    'run2d_walls': dict(mode='2D', Lx=400.0, Ly=400.0, boundary_mode='walls'),
    'run2d_periodic': dict(mode='2D', Lx=400.0, Ly=400.0, boundary_mode='periodic'),
    'run2d_shapes': dict(mode='2D', Lx=400.0, Ly=400.0, shape_enabled=True,
                         aspect_ratio_func_mean=1.3, aspect_ratio_inert_mean=1.2,
                         blockiness_func_mean=2.5, blockiness_inert_mean=2.2),
    'run3d_spheres': dict(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, Ngrid_3d=24,
                          cell_surface_coverage=1.0),
}
PARAMS_TRIALS = ['Trials/DOE2_2D_0001.json', 'Trials/default_trial.json']

# Runs at the CURRENT shipping defaults, with none of the V2.7 pins in COMMON.
# They have no V2.7 counterpart, so they exist only in the platform-local
# baseline and are checked only by `test_local_identity`. They are what stops a
# pin in COMMON from leaving the shipped configuration untested.
LOCAL_ONLY_RUNS = {
    'v34_2d_walls': dict(mode='2D', Lx=400.0, Ly=400.0, boundary_mode='walls'),
    'v34_3d_spheres': dict(mode='3D', Lx=250.0, Ly=250.0, Lz=250.0, Ngrid_3d=24,
                           cell_surface_coverage=1.0),
    'v34_2d_shapes': dict(mode='2D', Lx=400.0, Ly=400.0, shape_enabled=True,
                          aspect_ratio_func_mean=1.3, aspect_ratio_inert_mean=1.2,
                          blockiness_func_mean=2.5, blockiness_inert_mean=2.2),
}

# The parts of COMMON that are configuration rather than a V2.7 pin.
COMMON_BASE = dict(t_total=2.5, dt=0.5, save_every_h=0.5,
                   Ngrid=64, save_data=True, save_fields=True, compress_archive=False)

# Reference runs whose V2.7 output a later version DELIBERATELY supersedes.
# `test_legacy_identity` skips these with the reason printed; the platform-local
# baseline (`test_local_identity`) still pins them bit-for-bit, so they are not
# unguarded -- only unpinned to V2.7.
#
# Adding an entry here is a statement that the old numbers were WRONG, and it
# needs the measurement to back it up. Never add one to make a gate go green.
SUPERSEDED_RUNS = {
    'run2d_shapes':
        "V3.4: shaped contacts moved to the support-function MTD solver. The "
        "V2.7 common-normal solver found ZERO granule-granule contacts in this "
        "configuration at every frame (11 granules, 55 pairs, overlaps of "
        "0.19-1.17 um); MTD finds 1 at t=0 and 3 at t=2.5, and disp_func falls "
        "6.30 -> 4.41 um because the contacts now resist. See the V3.4 entry in "
        "CodeLog/Updates/CHANGELOG.md.",
}


def reference_params(name):
    """Params for one reference run (shared with the regression test).

    ``REFERENCE_RUNS`` carry the V2.7 pins in ``COMMON``; ``LOCAL_ONLY_RUNS``
    deliberately do not, so they exercise the current shipping defaults.
    """
    from gels.engine import Params
    if name in LOCAL_ONLY_RUNS:
        p = Params(**COMMON_BASE, **LOCAL_ONLY_RUNS[name])
    else:
        p = Params(**COMMON, **REFERENCE_RUNS[name])
    # The fixtures define the pure-Python reference path (bit-identical gate).
    # The compiled kernels are held to that path by tests/test_forces_vs_reference.py.
    p.use_numba = False
    return p


def make_run(name, out_dir):
    """Run one reference configuration into out_dir."""
    from gels.engine import run
    p = reference_params(name)
    p.output_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    run(p, seed=SEED)


def params_fixture_name(trial):
    return 'params_v27_' + os.path.splitext(os.path.basename(trial))[0] + '.json'


def make_params_fixture(trial, out_path):
    """Capture params.json exactly as step1_config.py produces it for a trial."""
    tmp = tempfile.mkdtemp(prefix='gels_fixture_')
    try:
        subprocess.run(
            [sys.executable, os.path.join('pipeline', 'step1_config.py'),
             '--trial', trial, '-o', tmp, '--force'],
            cwd=REPO, check=True, capture_output=True)
        shutil.copy(os.path.join(tmp, 'params.json'), out_path)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--force', action='store_true',
                    help='overwrite existing fixtures')
    ap.add_argument('--i-really-mean-it', dest='confirm', action='store_true',
                    help='required with --force on the V2.7 oracle (see the module docstring)')
    ap.add_argument('--local', action='store_true',
                    help='write the platform-local baseline to tests/fixtures_local/ instead')
    args = ap.parse_args()

    target = FIXTURES_LOCAL if args.local else FIXTURES

    if args.force and not args.local and not args.confirm:
        print("Refusing to regenerate the V2.7 oracle in tests/fixtures/.\n"
              "The V2.7 code is no longer in the tree, so these files are the ONLY\n"
              "record of its behaviour, and --force deletes them. If you are certain,\n"
              "pass --i-really-mean-it as well. To refresh the platform-local\n"
              "baseline instead, use:  --local --force")
        return 1
    if os.path.isdir(target) and os.listdir(target) and not args.force:
        print(f"Fixtures already exist in {target}; use --force to regenerate.")
        return 1
    if args.force and os.path.isdir(target):
        shutil.rmtree(target)
    os.makedirs(target, exist_ok=True)

    import numpy
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_hash = 'unknown'

    for trial in PARAMS_TRIALS:
        out = os.path.join(target, params_fixture_name(trial))
        print(f"params fixture: {trial} -> {os.path.relpath(out, REPO)}")
        make_params_fixture(trial, out)

    runs = list(REFERENCE_RUNS) + (list(LOCAL_ONLY_RUNS) if args.local else [])
    for name in runs:
        out_dir = os.path.join(target, name)
        print(f"\nreference run: {name} -> {os.path.relpath(out_dir, REPO)}")
        make_run(name, out_dir)

    manifest = {
        'seed': SEED,
        'common': COMMON,
        'reference_runs': REFERENCE_RUNS,
        'local_only_runs': LOCAL_ONLY_RUNS if args.local else {},
        'params_trials': PARAMS_TRIALS,
        'generated_with': {
            'git_hash': git_hash,
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'platform': platform.platform(),
        },
        'fingerprint': platform_fingerprint(),
        'kind': 'platform_local' if args.local else 'v27_oracle',
    }
    with open(os.path.join(target, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {os.path.relpath(target, REPO)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
