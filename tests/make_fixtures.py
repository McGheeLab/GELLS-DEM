"""
Bless the regression baseline for `tests/test_identity.py`.
===========================================================

Writes ``tests/fixtures_local/``: a handful of small runs at the CURRENT
shipping defaults, plus ``params.json`` as ``step1_config.py --trial``
produces it. ``test_identity`` re-runs the same configurations and requires
bit-identical results, so any unintended behaviour change is caught.

Contents written:

    tests/fixtures_local/manifest.json           versions, seed, configurations, fingerprint
    tests/fixtures_local/params_<trial>.json     params.json from step1_config.py --trial
    tests/fixtures_local/<run_name>/             5-step run: params.json, metadata.json,
                                                 history.json, snapshots/, fields/

The baseline is PLATFORM-LOCAL and gitignored. Exact equality of a run is only
meaningful on the machine it was blessed on: packing positions differ by ~1e-13
across libm / numpy reduction order, and in ``run3d_spheres`` that flips a
discrete bridge-formation decision at t = 1.0, after which the trajectories
bifurcate to O(1). No tolerance can bridge that, so `test_identity` checks the
fingerprint and skips rather than pretending.

V3.5 retired the V2.7 oracle that used to live in ``tests/fixtures/``; see the
docstring of ``tests/test_identity.py`` for why. Those files stay on disk as
the only surviving record of V2.7's behaviour, and nothing reads them.

Usage:
    python tests/make_fixtures.py --local            # first time
    python tests/make_fixtures.py --local --force    # re-bless, deliberately
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

# Small, fast configurations that still exercise every path a change is likely
# to touch: walls vs periodic, circles vs superellipses, 2D vs 3D, and both
# cell-seeding rules (n_cells_per_granule fallback vs cell_surface_coverage).
#
# NOTHING IS PINNED HERE. Until V3.5 this dict carried a growing list of V2.7
# defaults, so that the retired V2.7 oracle would stay green across deliberate
# default changes -- five of them by V3.5, each one a configuration the shipped
# defaults no longer used. The baseline is now blessed from the current tree, so
# the runs exercise what actually ships and a default change is re-blessed
# rather than pinned around.
COMMON = dict(
    t_total=2.5, dt=0.5, save_every_h=0.5,   # 5 steps, a snapshot every step
    Ngrid=64, save_data=True, save_fields=True, compress_archive=False,
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


def reference_params(name):
    """Params for one reference run (shared with the regression test)."""
    from gels.engine import Params
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
    return 'params_' + os.path.splitext(os.path.basename(trial))[0] + '.json'


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
                    help='re-bless: delete and regenerate the baseline')
    ap.add_argument('--local', action='store_true', default=True,
                    help='accepted for compatibility; the local baseline is the only target')
    args = ap.parse_args()

    # V3.5: tests/fixtures/ (the retired V2.7 oracle) is never written. It is kept
    # on disk as the only surviving record of V2.7's behaviour and nothing reads it.
    target = FIXTURES_LOCAL

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

    for name in REFERENCE_RUNS:
        out_dir = os.path.join(target, name)
        print(f"\nreference run: {name} -> {os.path.relpath(out_dir, REPO)}")
        make_run(name, out_dir)

    manifest = {
        'seed': SEED,
        'common': COMMON,
        'reference_runs': REFERENCE_RUNS,
        'params_trials': PARAMS_TRIALS,
        'generated_with': {
            'git_hash': git_hash,
            'python': platform.python_version(),
            'numpy': numpy.__version__,
            'platform': platform.platform(),
        },
        'fingerprint': platform_fingerprint(),
        'kind': 'platform_local',
    }
    with open(os.path.join(target, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote {os.path.relpath(target, REPO)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
