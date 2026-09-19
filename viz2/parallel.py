"""
viz2/parallel.py — process pool for the per-snapshot work in the viz2 suite (V3.0).

Every expensive loop in `viz2` is a map over snapshots: build one GIF frame, one
Voronoi tessellation, one coarse-grained field. The snapshots are independent, so
they run in a `ProcessPoolExecutor`. The engine's compiled kernels are not used
here — this is plain Python/matplotlib/scipy, which is why the visualization stage
was the one part of a run that still pinned a single core.

Use `pmap(fn, items)` exactly like `[fn(x) for x in items]`:

    from viz2.parallel import pmap
    frames = pmap(_frame_scaffold, [(snaps[i], p, t_i) for i in indices])

Rules for `fn`:

* it must be a **module-level** function (workers import it by qualified name);
* its argument and return value must be picklable (snapshot dicts, Params and
  numpy arrays all are);
* it must not rely on state set up by the parent process.

Worker count: the `workers` argument, else `$VIZ2_WORKERS`, else the physical
core count capped at 8 (beyond that the per-task pickling of a snapshot starts to
dominate). `workers=1` runs everything inline with no pool at all — which is also
the automatic fallback if the pool cannot start, so a machine or environment that
refuses to spawn processes still produces identical output, just slower.

Workers pin every numeric library to one thread (`OMP_NUM_THREADS`,
`NUMBA_NUM_THREADS`, ...): several visualization processes may already be running
side by side (`pipeline/run_showcase.py`), and nested thread pools would oversubscribe.
"""

import os
import sys

# Nothing heavy at module level: the pool initializer lives here, and it has to run
# in the worker *before* numba / matplotlib are imported by the task function.

_ENV_SINGLE_THREAD = {
    'OMP_NUM_THREADS': '1',
    'NUMBA_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'MPLBACKEND': 'Agg',
    'GELS_NO_CACHE_CHECK': '1',      # the parent already validated the kernel cache
}

DEFAULT_MAX_WORKERS = 8


def _physical_cores():
    try:
        from gels.kernels import physical_cores
        return int(physical_cores())
    except Exception:
        return int(os.cpu_count() or 1)


def resolve_workers(workers=None, n_items=None):
    """Worker count: explicit → $VIZ2_WORKERS → physical cores (capped), ≤ n_items."""
    if workers is None:
        env = os.environ.get('VIZ2_WORKERS', '').strip()
        if env:
            try:
                workers = int(env)
            except ValueError:
                workers = None
    if workers is None or workers == 0:
        workers = min(_physical_cores(), DEFAULT_MAX_WORKERS)
    workers = max(1, int(workers))
    if n_items is not None:
        workers = min(workers, max(1, int(n_items)))
    return workers


def _init_worker():
    """Single-thread every numeric library in the worker (runs before task imports)."""
    for key, value in _ENV_SINGLE_THREAD.items():
        os.environ[key] = value


def pmap(fn, items, workers=None, label=None):
    """`[fn(x) for x in items]`, evaluated in a process pool, order preserved.

    Falls back to serial evaluation when only one worker is wanted, when the
    platform cannot spawn processes, or when the pool dies mid-map. The result is
    identical either way; only the wall time differs.
    """
    items = list(items)
    if not items:
        return []
    n = resolve_workers(workers, len(items))
    if n <= 1:
        return [fn(x) for x in items]

    try:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                                 initializer=_init_worker) as pool:
            if label:
                print(f"    {label}: {len(items)} tasks on {n} workers")
            return list(pool.map(fn, items, chunksize=1))
    except Exception as exc:                       # pool refused to start, or a worker died
        print(f"    Warning: parallel map failed ({type(exc).__name__}: {exc}); "
              f"falling back to serial", file=sys.stderr)
        return [fn(x) for x in items]


__all__ = ['pmap', 'resolve_workers', 'DEFAULT_MAX_WORKERS']
