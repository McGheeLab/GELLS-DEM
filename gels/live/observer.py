"""
Observer protocol for ``gels.engine.run`` and the live-view producer.
====================================================================

:class:`Observer` is the hook the engine calls; every method is a no-op so a
subclass overrides only what it needs. :class:`LiveViewObserver` is the
engine-side half of the live viewer: it builds compact frame messages from
the running :class:`GranuleSystem` and pushes them to the viewer process
through a bounded ``multiprocessing.Queue``.

Design rules (see the V3.0 plan, Part C):

* **Never block the simulation.** Frames are sent with ``put_nowait``; when
  the queue is full the frame is dropped and the cadence backs off. The
  viewer always draws the newest frame it has.
* **Pause lives here, not in the engine.** ``on_step`` blocks in a short
  polling loop while paused, still draining control messages so
  resume / single-step / stop work.
* **A dead viewer never kills the run.** Broken pipes disable the observer
  and the simulation continues headless.
"""

import queue
import time

import numpy as np


class Observer:
    """Base class: every callback is a no-op. Override what you need.

    ``on_step`` returns ``True`` to request a graceful stop; the engine then
    saves a final snapshot at the current step and returns normally.
    """

    def on_start(self, gs, p, seed, n_steps, resume_step):
        pass

    def on_step(self, s, t, gs, F, contacts):
        return False

    def on_save(self, snap_idx, s, t, snap, m):
        pass

    def on_end(self, hist, gs, reason):
        pass


class CallableObserver(Observer):
    """Wrap a plain ``fn(s, t, gs, F) -> bool`` as an Observer."""

    def __init__(self, fn):
        self.fn = fn

    def on_step(self, s, t, gs, F, contacts):
        return bool(self.fn(s, t, gs, F))


def as_observer(obj):
    if obj is None or isinstance(obj, Observer):
        return obj
    if callable(obj):
        return CallableObserver(obj)
    raise TypeError("observer must be an Observer, a callable or None")


class CompositeObserver(Observer):
    """Several observers on one run, in order (V3.6).

    ``on_step`` **or-accumulates and must not short-circuit**: ``any(genexpr)``
    stops calling children after the first ``True``, which would leave the
    viewer's pause loop undrained and desynchronise it from the engine. The
    convergence monitor goes first and the viewer second, so a stop request is
    still seen by the viewer.
    """

    def __init__(self, *children):
        self.children = [as_observer(c) for c in children if c is not None]

    def on_start(self, gs, p, seed, n_steps, resume_step):
        for c in self.children:
            c.on_start(gs, p, seed, n_steps, resume_step)

    def on_step(self, s, t, gs, F, contacts):
        stop = False
        for c in self.children:
            stop = bool(c.on_step(s, t, gs, F, contacts)) or stop   # no short circuit
        return stop

    def on_save(self, snap_idx, s, t, snap, m):
        for c in self.children:
            c.on_save(snap_idx, s, t, snap, m)

    def on_end(self, hist, gs, reason):
        for c in self.children:
            c.on_end(hist, gs, reason)

    @property
    def stop_reason(self):
        """The first child that owns a reason gets to name the stop."""
        for c in self.children:
            r = getattr(c, 'stop_reason', None)
            if r:
                return r
        return None


class ConvergenceObserver(Observer):
    """Runs `gels.convergence.ConvergenceMonitor` on a live run (V3.6).

    The path-length signal needs positions, which `history.json` does not carry,
    so it is accumulated here from the SAVED frames -- the same resolution
    `pipeline/shadow_check.py` replays at, deliberately, so a tolerance
    calibrated in shadow is valid live.

    The monitor's own row buffer is written to ``convergence.json`` in the run
    directory, so a resumed run re-ingests its windows instead of restarting
    them.
    """

    def __init__(self, monitor, out_dir=None):
        self.monitor = monitor
        self.out_dir = out_dir
        self.records = []
        self.stop_reason = None

    def on_save(self, snap_idx, s, t, snap, m):
        if snap is None:
            return
        import numpy as _np
        xs = [snap['x'], snap['y']] + ([snap['z']] if 'z' in snap else [])
        pos = _np.stack([_np.asarray(v, dtype=float) for v in xs], axis=1)
        func = _np.asarray(snap.get('gtype', _np.zeros(len(pos)))) == 0
        self.monitor.observe_positions(pos, func)
        rec = self.monitor.update(self.monitor.row_from_metrics(t, m))
        self.records.append(rec)
        if isinstance(m, dict):
            m.update({k: v for k, v in rec.items() if k != 'conv_blocking'})
        if self.monitor.should_stop():
            self.stop_reason = 'converged'
        self._persist()

    def on_step(self, s, t, gs, F, contacts):
        return self.monitor.should_stop()

    def _persist(self):
        if not self.out_dir:
            return
        import json
        import os
        from gels.convergence import COLS
        try:
            with open(os.path.join(self.out_dir, 'convergence.json'), 'w',
                      encoding='utf-8') as fh:
                json.dump({'rows': [dict(zip(COLS, r)) for r in self.monitor.rows],
                           'status': self.monitor.status()}, fh, default=float)
        except OSError:
            pass            # a full disk must never kill the run


class RecordingObserver(Observer):
    """Counts callbacks (used by tests); optionally stops at a given step."""

    def __init__(self, stop_at_step=None):
        self.starts = 0
        self.steps = []
        self.saves = []
        self.end = None
        self.stop_at_step = stop_at_step

    def on_start(self, gs, p, seed, n_steps, resume_step):
        self.starts += 1
        self.n_steps = n_steps

    def on_step(self, s, t, gs, F, contacts):
        self.steps.append((s, float(t)))
        return self.stop_at_step is not None and s >= self.stop_at_step

    def on_save(self, snap_idx, s, t, snap, m):
        self.saves.append((snap_idx, s, float(t), len(m)))

    def on_end(self, hist, gs, reason):
        self.end = (reason, len(hist))


# ──────────────────────────────────────────────────────────────────────
# Live-view producer
# ──────────────────────────────────────────────────────────────────────

# metric keys forwarded to the viewer's side panel
FRAME_METRIC_KEYS = ('time', 'n_bridges', 'porosity', 'K_kozeny_carman', 'func_lf',
                     'n_cells_total', 'n_divisions_cum', 'bed_height_mean',
                     'phi_solid', 'n_contacts', 'n_bridging_cells', 'tissue_frac',
                     'n_locked_in_cells', 'bridge_force_mean')


class LiveViewObserver(Observer):
    """Engine-side producer for the live viewer.

    Parameters
    ----------
    frame_q : multiprocessing.Queue (bounded)
        Carries ``start`` / ``frame`` / ``end`` messages to the viewer.
    ctrl_q : multiprocessing.Queue
        Control tuples from the viewer: ('pause',), ('resume',), ('step', n),
        ('stop',), ('every', n), ('fps', f), ('detach',), ('color', mode).
    ctrl_evt : multiprocessing.Event
        Doorbell set by the viewer whenever it puts something on ctrl_q.
    stop_evt : multiprocessing.Event
        Set by the viewer to request a graceful stop of the simulation.
    every_n_steps : int
        Send a frame every N steps; 0 = auto from the measured step rate and
        ``max_fps``.
    max_fps : float
        Frame-rate ceiling.
    send_cells : bool
        Include cell positions/states (the bulk of the payload).
    """

    def __init__(self, frame_q, ctrl_q, ctrl_evt, stop_evt,
                 every_n_steps=0, max_fps=20.0, send_cells=True, run_dir=''):
        self.frame_q = frame_q
        self.ctrl_q = ctrl_q
        self.ctrl_evt = ctrl_evt
        self.stop_evt = stop_evt
        self.every_fixed = int(every_n_steps)
        self.every = max(1, self.every_fixed) if self.every_fixed > 0 else 1
        self.max_fps = float(max_fps) if max_fps and max_fps > 0 else 20.0
        self.send_cells = bool(send_cells)
        self.send_speed = False
        self.color_mode = 'species'
        self.run_dir = run_dir

        self.paused = False
        self.step_credits = 0
        self.dead = False
        self.detached = False
        self.n_sent = 0
        self.n_dropped = 0
        self._drop_window = []          # (time, dropped?) for adaptive cadence
        self._last_sent = 0.0
        self._last_step_time = None
        self._steps_per_s = 0.0
        self._pending_metrics = None
        self._force_frame = False
        self._static = None

    # ── engine callbacks ───────────────────────────────────────────────

    def on_start(self, gs, p, seed, n_steps, resume_step):
        from gels.live import frames
        self._p = p
        msg, static = frames.start_message(gs, p, seed=seed, n_steps=n_steps,
                                           resume_step=resume_step, run_dir=self.run_dir)
        self._static = static
        self._put(msg, block=True, timeout=10.0)

    def on_save(self, snap_idx, s, t, snap, m):
        self._pending_metrics = {k: float(m[k]) for k in FRAME_METRIC_KEYS if k in m}
        self._force_frame = True         # save steps always try to ship a frame

    def on_step(self, s, t, gs, F, contacts):
        if self.dead or self.detached:
            return False
        if self.stop_evt.is_set():
            return True
        self._drain_control()
        # Pause: hold the engine here, keep listening for resume / step / stop.
        while self.paused and self.step_credits == 0 and not self.stop_evt.is_set() \
                and not self.detached:
            self._drain_control(wait=0.05)
        if self.stop_evt.is_set():
            return True
        if self.step_credits > 0:
            self.step_credits -= 1
            self._force_frame = True

        now = time.perf_counter()
        self._update_rate(now)
        if self._force_frame or (s % self.every == 0 and (now - self._last_sent) >= 1.0 / self.max_fps):
            self._send_frame(gs, s, t, now)
            self._force_frame = False
        return self.stop_evt.is_set()

    def on_end(self, hist, gs, reason):
        if self.dead:
            return
        try:
            step = int(hist[-1].get('time', 0.0) / self._p.dt) if hist else 0
        except Exception:
            step = 0
        self._put({'kind': 'end', 'reason': reason,
                   't': float(hist[-1]['time']) if hist else 0.0, 'step': step,
                   'n_sent': self.n_sent, 'n_dropped': self.n_dropped},
                  block=True, timeout=5.0)

    # ── internals ──────────────────────────────────────────────────────

    def _update_rate(self, now):
        if self._last_step_time is not None:
            dt = now - self._last_step_time
            if dt > 0:
                inst = 1.0 / dt
                self._steps_per_s = inst if self._steps_per_s == 0 else 0.9 * self._steps_per_s + 0.1 * inst
        self._last_step_time = now
        if self.every_fixed <= 0 and self._steps_per_s > 0:
            auto = max(1, int(round(self._steps_per_s / self.max_fps)))
            # back off while frames are being dropped, recover slowly
            recent = [d for (tm, d) in self._drop_window if now - tm < 1.0]
            if recent and sum(recent) > 0.5 * len(recent):
                auto = max(auto, self.every * 2)
            self.every = auto

    def _send_frame(self, gs, s, t, now):
        from gels.live import frames
        try:
            msg = frames.frame_message(gs, self._p, s, t, static=self._static,
                                       metrics=self._pending_metrics,
                                       send_cells=self.send_cells,
                                       send_speed=self.send_speed)
        except Exception as exc:      # a rendering bug must not kill the run
            self.dead = True
            print(f"  live view: frame build failed ({type(exc).__name__}: {exc}); "
                  f"simulation continues without it")
            return
        msg['stats'] = {'steps_per_s': round(self._steps_per_s, 2), 'every': self.every,
                        'n_dropped': self.n_dropped, 'n_sent': self.n_sent}
        dropped = not self._put(msg, block=False)
        self._drop_window.append((now, dropped))
        if len(self._drop_window) > 200:
            del self._drop_window[:100]
        if dropped:
            self.n_dropped += 1
        else:
            self.n_sent += 1
            self._last_sent = now
            self._pending_metrics = None

    def _put(self, msg, block, timeout=None):
        try:
            if block:
                self.frame_q.put(msg, timeout=timeout)
            else:
                self.frame_q.put_nowait(msg)
            return True
        except queue.Full:
            return False
        except (BrokenPipeError, EOFError, OSError, ValueError):
            self._disconnect()
            return False

    def _disconnect(self):
        if not self.dead:
            self.dead = True
            print("  live view: viewer disconnected; simulation continues")

    def _drain_control(self, wait=0.0):
        if wait > 0:
            self.ctrl_evt.wait(wait)
        if not self.ctrl_evt.is_set():
            return
        self.ctrl_evt.clear()
        while True:
            try:
                cmd = self.ctrl_q.get_nowait()
            except queue.Empty:
                break
            except (BrokenPipeError, EOFError, OSError):
                self._disconnect()
                break
            self._handle(cmd)

    def _handle(self, cmd):
        if not cmd:
            return
        op = cmd[0]
        if op == 'pause':
            self.paused = True
        elif op == 'resume':
            self.paused = False
            self.step_credits = 0
        elif op == 'step':
            self.paused = True
            self.step_credits += int(cmd[1]) if len(cmd) > 1 else 1
        elif op == 'stop':
            self.stop_evt.set()
            self.paused = False
        elif op == 'every':
            self.every_fixed = max(0, int(cmd[1]))
            self.every = max(1, self.every_fixed) if self.every_fixed > 0 else self.every
        elif op == 'fps':
            self.max_fps = max(0.1, float(cmd[1]))
        elif op == 'detach':
            self.detached = True
            self.paused = False
        elif op == 'color':
            self.color_mode = str(cmd[1])
            self.send_speed = (self.color_mode == 'speed')
        elif op == 'cells':
            self.send_cells = bool(cmd[1])


__all__ = ['Observer', 'CallableObserver', 'RecordingObserver', 'LiveViewObserver',
           'as_observer', 'FRAME_METRIC_KEYS']
