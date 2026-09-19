"""
Real-time observation of a running simulation (V3.0).
=====================================================

``gels.engine.run(p, seed, observer=...)`` accepts an :class:`Observer`. The
engine calls it at the start of the run, after every step, after every saved
snapshot and at the end; the observer may ask the run to stop. Nothing in the
engine changes when no observer is given.

Modules
-------
observer   Observer base class; LiveViewObserver — throttled, never-blocking
           producer that feeds the viewer process and honours pause/step/stop
frames     builders for the ``start`` / ``frame`` / ``end`` messages from a
           GranuleSystem or a snapshot dict; vectorised cell world positions
viewer     the TkAgg viewer process (``viewer_main``), single-artist renderer,
           metrics panel, key bindings, headless PNG mode
tail       sources that replay saved snapshots or follow a run directory

The viewer never imports ``viz2.common`` (every viz2 module pins the Agg
backend at import); it uses the matplotlib-free ``viz2.palette`` and
``viz2.snapshot_ops`` instead.
"""

from gels.live.observer import Observer, LiveViewObserver  # noqa: F401

__all__ = ['Observer', 'LiveViewObserver']
