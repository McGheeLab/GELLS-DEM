"""
GELS benchmark harness.
=======================

Times the components of a timestep (forces, cell update, integration, field
rendering, metrics) and the packing stage as a function of granule count and
thread count, so performance work can be measured instead of guessed.

    python -m gels.bench --mode 2D --N 500,2000 --steps 5
    python -m gels.bench --mode 3D --N 1000 --steps 3 --threads 1,40
    python -m gels.bench --mode 2D --N 10000 --steps 10 --json results/bench/2d.json

Each configuration builds a square/cubic domain sized for the requested N at a
fixed solid fraction, generates the packing, then times ``steps`` calls of
``step()`` plus one ``render_fields*`` and one ``compute_metrics`` call.
Thread counts are applied with ``gels.kernels.configure_threads`` before the
first parallel kernel runs; on the pure-Python reference path they make no
difference, which the table will show honestly.
"""

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# The engine prints non-ASCII (alpha, mu, arrows); a default Windows console
# is cp1252 and would abort the benchmark inside a print().
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402


def _domain_size_for(N, mode, R=40.0, phi_solid=0.62):
    """Side length giving ~N granules of radius R at the requested solid fraction."""
    if mode == '3D':
        return (N * (4.0 / 3.0) * np.pi * R**3 / phi_solid) ** (1.0 / 3.0)
    return (N * np.pi * R**2 / phi_solid) ** 0.5


def make_params(N, mode, shape=False, periodic=False, R=40.0, phi_solid=0.62,
                func_ratio=0.6):
    from gels.engine import Params
    L = float(_domain_size_for(N, mode, R, phi_solid))
    kw = dict(mode=mode, Lx=L, Ly=L, Lz=L if mode == '3D' else 800.0,
              R_func_mean=R, R_func_std=0.0, R_inert_mean=R, R_inert_std=0.0,
              phi_solid_target=phi_solid, func_ratio=func_ratio,
              cell_surface_coverage=1.0,
              boundary_mode='periodic' if periodic else 'walls',
              shape_enabled=bool(shape),
              save_data=False, compress_archive=False,
              dt=0.5, t_total=1.0)
    if shape:
        kw.update(aspect_ratio_func_mean=1.2, aspect_ratio_inert_mean=1.2,
                  blockiness_func_mean=2.5, blockiness_inert_mean=2.5)
    return Params(**kw)


def bench_one(N, mode, steps, threads, shape=False, periodic=False, seed=1, path='kernels'):
    from gels import engine
    from gels.kernels import configure_threads

    layer, n_thr = configure_threads(threads, 'omp', n_granules=N)
    p = make_params(N, mode, shape=shape, periodic=periodic)
    p.use_numba = (path == 'kernels')
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    if mode == '3D':
        gs = engine.generate_packing_3d(p, seed=seed)
    else:
        gs = engine.generate_packing(p, seed=seed)
    t_pack = time.perf_counter() - t0

    # Warm up JIT / caches on one untimed step.
    engine.update_cell_state(gs, p, 0.0, rng)
    engine.step(gs, p, rng, p.dt)

    t_steps = []
    t = p.dt
    F = None
    contacts = None
    for _ in range(steps):
        t += p.dt
        t0 = time.perf_counter()
        F, contacts = engine.step(gs, p, rng, t)
        t_steps.append(time.perf_counter() - t0)

    # Breakdown of one more step: cell state machine, forces (with the kernel
    # phase split when on the compiled path), then rendering and metrics.
    t += p.dt
    t0 = time.perf_counter()
    engine.update_cell_state(gs, p, t, rng)
    t_cells = time.perf_counter() - t0
    if engine.kernels_enabled(p):
        gs.clip_arrays = None
    else:
        for k in range(gs.N):
            gs.contact_clips[k] = []
    t0 = time.perf_counter()
    if gs.is_3d:
        F, _tq, contacts = engine.compute_forces_3d(gs, p, rng)
    else:
        F, _tq, contacts = engine.compute_forces(gs, p, rng)
    t_forces = time.perf_counter() - t0
    split = {}
    if engine.kernels_enabled(p):
        from gels.kernels import contact2d, contact3d
        split = dict(contact3d.TIMINGS if gs.is_3d else contact2d.TIMINGS)

    t0 = time.perf_counter()
    if gs.is_3d:
        pf, pi, pv = engine.render_fields_3d(gs, p)
    else:
        pf, pi, pv = engine.render_fields(gs, p)
    t_render = time.perf_counter() - t0

    t0 = time.perf_counter()
    engine.compute_metrics(gs, p, pf, pi, pv, t, F)
    t_metrics = time.perf_counter() - t0

    return {
        'mode': mode, 'path': path, 'N_target': N, 'N': int(gs.N), 'cells': int(gs.total_cells),
        'shape': bool(shape), 'periodic': bool(periodic),
        'threads': int(n_thr), 'layer': layer,
        'contacts': int(len(contacts)) if contacts is not None else None,
        'grid': int(pf.shape[0]),
        't_pack_s': round(t_pack, 3),
        't_step_s': round(float(np.median(t_steps)), 4),
        't_step_min_s': round(float(np.min(t_steps)), 4),
        't_cells_s': round(t_cells, 4),
        't_forces_s': round(t_forces, 4),
        't_forces_neighbours_s': round(split.get('neighbours', float('nan')), 4),
        't_forces_contacts_s': round(split.get('contacts', float('nan')), 4),
        't_forces_bridging_s': round(split.get('bridging', float('nan')), 4),
        't_render_s': round(t_render, 4),
        't_metrics_s': round(t_metrics, 4),
        'granule_steps_per_s': round(gs.N / float(np.median(t_steps)), 1),
    }


def _fmt_row(r):
    return (f"{r['mode']:>3} {r['N']:>7d} {r['cells']:>7d} {r['threads']:>4d} "
            f"{r['t_pack_s']:>9.2f} {r['t_step_s']:>9.4f} {r['t_cells_s']:>8.4f} "
            f"{r['t_forces_s']:>8.4f} {r['t_forces_contacts_s']:>8.4f} {r['t_forces_bridging_s']:>8.4f} "
            f"{r['t_render_s']:>9.4f} {r['t_metrics_s']:>9.4f} {r['granule_steps_per_s']:>12.0f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="GELS performance benchmark")
    ap.add_argument('--mode', default='2D', choices=['2D', '3D'])
    ap.add_argument('--N', default='500,2000',
                    help='comma-separated target granule counts')
    ap.add_argument('--steps', type=int, default=5)
    ap.add_argument('--threads', default='0',
                    help='comma-separated thread counts (0 = numba default)')
    ap.add_argument('--shape', action='store_true', help='superellipse/superellipsoid granules')
    ap.add_argument('--periodic', action='store_true')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--path', default='kernels', choices=['kernels', 'reference'],
                    help='compiled kernels (default) or the pure-Python reference loops')
    ap.add_argument('--json', default=None, help='append results to this JSON file')
    args = ap.parse_args(argv)

    Ns = [int(float(s)) for s in args.N.split(',')]
    threads = [int(s) for s in args.threads.split(',')]

    print(f"path: {args.path}")
    print(f"{'mode':>3} {'N':>7} {'cells':>7} {'thr':>4} {'pack[s]':>9} "
          f"{'step[s]':>9} {'cells':>8} {'forces':>8} {'contact':>8} {'bridge':>8} "
          f"{'render[s]':>9} {'metrics':>9} {'gran*step/s':>12}")
    rows = []
    for thr in threads:
        for N in Ns:
            r = bench_one(N, args.mode, args.steps, thr,
                          shape=args.shape, periodic=args.periodic, seed=args.seed,
                          path=args.path)
            rows.append(r)
            print(_fmt_row(r), flush=True)

    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        existing = []
        if os.path.exists(args.json):
            with open(args.json) as f:
                existing = json.load(f)
        with open(args.json, 'w') as f:
            json.dump(existing + rows, f, indent=2)
        print(f"appended {len(rows)} rows to {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
