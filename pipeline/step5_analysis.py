"""
STEP 5 of 5 - Mathematical analysis.
====================================

Runs the analysis package against a single finished run and collects the
results into ``<run_dir>/analysis/summary.json`` alongside each module's own
plots.

Modules
-------
    mean_field    fit the two-zone volume-conserving compaction ODE
    coarse_grain  stress tensor, strain rate, effective viscosity, Z(t)
    descriptors   tissue architecture descriptor vector (BV/TV, thickness,
                  Minkowski functionals, correlation length, ...)
    arch_distance architectural distance to each organ target vector

Each module runs independently; one failing does not abort the rest, and its
error is recorded in the summary. Descriptors are computed from phase fields
rebuilt from particle positions when ``fields/`` is absent, which is the
default since V2.6.

Examples
--------
    python pipeline/step5_analysis.py -i results/my_run
    python pipeline/step5_analysis.py -i results/my_run --modules descriptors arch_distance
    python pipeline/step5_analysis.py -i results/my_run --force
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import (  # noqa: E402
    banner, done, guard, mark_done, print_progress, require,
)

MODULES = ['mean_field', 'coarse_grain', 'descriptors', 'arch_distance']


def main():
    parser = argparse.ArgumentParser(
        description="GELS pipeline step 5: mathematical analysis.")
    parser.add_argument('-i', '--run-dir', required=True,
                        help='Run directory containing the simulated run')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Analysis output directory '
                             '(default: <run_dir>/analysis)')
    parser.add_argument('--modules', nargs='*', default=None,
                        choices=MODULES,
                        help=f'Subset of modules to run (default: all of '
                             f'{", ".join(MODULES)})')
    parser.add_argument('--force', action='store_true',
                        help='Re-run even if this step already completed')
    args = parser.parse_args()

    run_dir = args.run_dir
    banner('analysis', run_dir)
    require(run_dir, 'config')
    require(run_dir, 'pack')
    require(run_dir, 'simulate')
    guard(run_dir, 'analysis', args.force)

    outdir = args.outdir or os.path.join(run_dir, 'analysis')
    os.makedirs(outdir, exist_ok=True)
    wanted = args.modules or MODULES

    summary = {}
    t0 = time.time()

    runners = {
        'mean_field': _mean_field,
        'coarse_grain': _coarse_grain,
        'descriptors': _descriptors,
        'arch_distance': _arch_distance,
    }

    for name in wanted:
        print(f"\n  --- {name} ---")
        sub = os.path.join(outdir, name)
        os.makedirs(sub, exist_ok=True)
        try:
            summary[name] = runners[name](run_dir, sub)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}")
            summary[name] = {'error': f'{type(exc).__name__}: {exc}'}
        finally:
            _close_figures()

    elapsed = time.time() - t0

    summary_path = os.path.join(outdir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    print(f"\n  Wrote {summary_path}")

    ok = [k for k, v in summary.items() if 'error' not in v]
    bad = [k for k, v in summary.items() if 'error' in v]
    print(f"  Modules succeeded: {', '.join(ok) if ok else '(none)'}")
    if bad:
        print(f"  Modules failed:    {', '.join(bad)}")

    mark_done(run_dir, 'analysis', {
        'modules_run': wanted,
        'succeeded': ok,
        'failed': bad,
        'wall_seconds': round(elapsed, 1),
    })

    print_progress(run_dir)
    done('analysis', run_dir, elapsed)


# ----------------------------------------------------------------------
# Individual analysis modules
# ----------------------------------------------------------------------

def _mean_field(run_dir, outdir):
    """Fit the mean-field two-zone compaction ODE to the run."""
    from analysis.mean_field_model import from_run, run_all

    _model, fit, data = from_run(run_dir)
    run_all(run_dir, outdir=outdir)
    return {
        'R2': float(fit['R2']),
        'eta_eff': float(fit['eta_eff']),
        'sigma_0': float(fit['sigma_0']),
        'alpha': float(fit['alpha']),
        'phi_f_initial': float(data['phi_f'][0]),
        'phi_f_final': float(data['phi_f'][-1]),
        'phi_f_change': float(data['phi_f'][-1] - data['phi_f'][0]),
        'n_timepoints': int(len(data['time'])),
    }


def _coarse_grain(run_dir, outdir):
    """Extract continuum stress, strain rate, viscosity and coordination."""
    import numpy as np
    from analysis.coarse_grain import extract_continuum_timeseries, run_all

    ts = extract_continuum_timeseries(run_dir)
    run_all(run_dir, outdir=outdir)

    eta = np.asarray(ts['eta_eff'])
    finite = eta[np.isfinite(eta)]
    return {
        'n_timepoints': int(len(ts['time'])),
        'pressure_initial': float(ts['pressure'][0]),
        'pressure_final': float(ts['pressure'][-1]),
        'von_mises_final': float(ts['von_mises'][-1]),
        'eta_eff_mean': float(np.mean(finite)) if finite.size else None,
        'Z_initial': float(ts['Z'][0]),
        'Z_final': float(ts['Z'][-1]),
        'Z_ff_final': float(ts['Z_ff'][-1]),
    }


def _descriptors(run_dir, outdir):
    """Compute the tissue architecture descriptor vector at the last snapshot.

    Phase fields are rebuilt from particle positions when the run has no
    saved ``fields/`` directory, which is the V2.6 default.
    """
    from gels.engine import load_run
    from analysis.tissue_descriptors import compute_descriptor_vector
    from viz2.common import ensure_phase_fields

    hist, snaps, p, _meta = load_run(run_dir)
    if not snaps:
        raise ValueError(f"No snapshots found in {run_dir}")

    snap = snaps[-1]
    phi_f, phi_i, phi_v = ensure_phase_fields(snap, p)

    dx = p.Lx / (phi_f.shape[2] if phi_f.ndim == 3 else phi_f.shape[1])
    bx = getattr(p, 'boundary_exclusion', 0.2)
    desc = compute_descriptor_vector(phi_f, phi_i, phi_v, dx, bx)

    clean = {k: _jsonable(v) for k, v in desc.items()}
    with open(os.path.join(outdir, 'descriptors.json'), 'w') as f:
        json.dump(clean, f, indent=2, default=_jsonable)

    bvtv = desc.get('BV_TV', desc.get('bv_tv'))
    if isinstance(bvtv, float):
        print(f"  BV/TV = {bvtv:.3f}  ({len(desc)} descriptors)")
    else:
        print(f"  Computed {len(desc)} descriptors")
    return clean


def _arch_distance(run_dir, outdir):
    """Architectural distance from this scaffold to each organ target."""
    from analysis.arch_distance import distance_trajectory, run_all

    run_all(run_dir=run_dir, outdir=outdir)

    # distance_trajectory returns (organ -> distance series, times)
    trajectories, times = distance_trajectory(run_dir)

    result = {}
    for organ, series in trajectories.items():
        if len(series):
            result[organ] = float(series[-1])

    if not result:
        return {'final_distances': {}}

    best = min(result, key=result.get)
    print(f"  Closest organ: {best} (D_arch = {result[best]:.3f})")
    return {
        'final_distances': result,
        'closest_organ': best,
        'closest_distance': result[best],
        't_final': float(times[-1]) if len(times) else None,
    }


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _close_figures():
    try:
        import matplotlib.pyplot as plt
        plt.close('all')
    except Exception:
        pass


def _jsonable(v):
    """Convert numpy scalars/arrays to JSON-serialisable Python types."""
    import numpy as np

    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


if __name__ == '__main__':
    main()
