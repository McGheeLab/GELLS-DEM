"""
GELS compiled kernels (V3.0).
=============================

Numba-parallel implementations of the per-pair / per-cell / per-voxel loops
that dominate a GELS timestep, plus the configuration helpers the engine
uses to set them up. Everything under ``gels.kernels`` (except
``reference.py``) works on plain ndarrays and scalars only: no ``Params``,
no ``GranuleSystem``, no Python containers inside a kernel.

``reference.py`` holds the original pure-Python loops. It is the oracle every
kernel is regression-tested against and the path LS-DEM (deformable granules)
still runs on.

Phase 0 of the V3.0 plan ships only this package skeleton and the thread /
threading-layer configuration; the kernels themselves arrive in Phase 6.
"""

import os
import warnings

try:
    import numba
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:  # pragma: no cover - numba is an optional dependency
    numba = None
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Identity decorator standing in for numba.njit."""
        if args and callable(args[0]):
            return args[0]
        return lambda f: f

    prange = range


# Threading layers numba knows about, in order of preference on this project.
# 'omp' (Intel/LLVM OpenMP) is verified to scale on the dual-socket Windows
# workstation; 'tbb' needs the tbb wheel; 'workqueue' is numba's fallback and
# is not safe to call from more than one Python thread.
_LAYERS = ('omp', 'tbb', 'workqueue')


def configure_threads(n_threads=0, layer='omp', n_granules=None):
    """Select the numba threading layer and thread count for this process.

    Parameters
    ----------
    n_threads : int
        Threads for parallel kernels. 0 = auto: the physical-core estimate
        ``max(1, os.cpu_count() // 2)`` (hyperthreads rarely help numeric
        kernels; ``bench.py --threads`` decides per machine). -1 = numba's
        default (all logical CPUs). Values above ``NUMBA_NUM_THREADS`` are
        clamped.
    layer : str
        'omp' | 'tbb' | 'workqueue' | 'default'. Only honoured if no parallel
        kernel has run yet in this process (numba initialises the layer
        lazily on first use and cannot switch afterwards).

    Returns
    -------
    (layer_in_use, n_threads_in_use) or (None, 0) when numba is missing.
    """
    if not HAS_NUMBA:
        return None, 0

    if layer and layer != 'default':
        if layer not in _LAYERS:
            warnings.warn(f"unknown numba threading layer {layer!r}; "
                          f"expected one of {_LAYERS}")
        else:
            active = _active_layer()
            if active is None:
                os.environ['NUMBA_THREADING_LAYER'] = layer
            elif active != layer:
                warnings.warn(f"numba threading layer already initialised as "
                              f"{active!r}; cannot switch to {layer!r} in this process")

    max_threads = numba.config.NUMBA_NUM_THREADS
    if n_threads == 0:
        n_threads = auto_threads(n_granules)
    if n_threads and n_threads > 0:
        n = min(int(n_threads), max_threads)
        numba.set_num_threads(n)
    n_in_use = numba.get_num_threads()
    return _active_layer() or os.environ.get('NUMBA_THREADING_LAYER', 'default'), n_in_use


def auto_threads(n_granules=None):
    """Thread count for a system of ``n_granules``: about one thread per 1000
    granules, at least 4, at most the physical cores (all of them when N is unknown)."""
    cores = physical_cores()
    if n_granules is None:
        return cores
    return int(min(cores, max(4, int(n_granules) // 1000)))


def physical_cores():
    """Best-effort physical core count (logical CPUs / 2, at least 1)."""
    return max(1, (os.cpu_count() or 2) // 2)


def _active_layer():
    """Name of the threading layer if numba has already launched one."""
    try:
        return numba.threading_layer()
    except ValueError:      # not initialised yet
        return None


def ensure_cache_fresh(purge=True):
    """Purge numba's on-disk cache when any kernel source changed since the last run.

    numba invalidates a cached function when its own file changes, but not
    when a function it inlined from another module (gels.materials, the
    engine's leaf solvers) changed. We hash every source the kernels depend on
    and drop the .nbi/.nbc files under gels/ when the hash moves. Returns True
    when a purge happened. Set GELS_NO_CACHE_CHECK=1 to skip.
    """
    import glob
    import hashlib
    if os.environ.get('GELS_NO_CACHE_CHECK'):
        return False
    here = os.path.dirname(os.path.abspath(__file__))
    gels_dir = os.path.dirname(here)
    sources = sorted(glob.glob(os.path.join(here, '*.py'))) + [
        os.path.join(gels_dir, 'engine.py'), os.path.join(gels_dir, 'materials.py')]
    h = hashlib.sha1()
    for src in sources:
        try:
            with open(src, 'rb') as fh:
                h.update(os.path.basename(src).encode())
                h.update(fh.read())
        except OSError:
            pass
    digest = h.hexdigest()
    cache_dir = os.path.join(here, '__pycache__')
    os.makedirs(cache_dir, exist_ok=True)
    stamp = os.path.join(cache_dir, 'gels_kernels.sha1')
    try:
        with open(stamp) as fh:
            old = fh.read().strip()
    except OSError:
        old = None
    if old == digest:
        return False
    purged = 0
    if purge and old is not None:
        for d in (cache_dir, os.path.join(gels_dir, '__pycache__')):
            for f in glob.glob(os.path.join(d, '*.nbi')) + glob.glob(os.path.join(d, '*.nbc')):
                try:
                    os.remove(f)
                    purged += 1
                except OSError:
                    pass
    with open(stamp, 'w') as fh:
        fh.write(digest)
    return purged > 0


ensure_cache_fresh()


__all__ = ['HAS_NUMBA', 'njit', 'prange', 'configure_threads', 'auto_threads', 'physical_cores',
           'ensure_cache_fresh']
