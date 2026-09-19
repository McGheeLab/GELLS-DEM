"""
Message sources that read saved snapshots instead of a live engine.
==================================================================

``SnapshotTailSource`` follows a run directory while step 3 is writing it
(or after it finished), turning each new ``snapshots/snap_NNNN.npz`` into a
``frame`` message; ``ReplaySource`` plays the existing snapshots back at a
fixed cadence. Both synthesise the ``start`` message from ``params.json``
and the first snapshot, and feed ``history.json`` to the metrics panel.

Half-written files are impossible since the engine writes atomically
(V3.0), but a reader can still race the rename on some filesystems, so
loading retries on the next poll instead of raising.
"""

import glob
import json
import os
import time
import zipfile

import numpy as np

from gels.engine import load_params
from gels.live import frames


def _snap_files(run_dir):
    return sorted(glob.glob(os.path.join(run_dir, 'snapshots', 'snap_*.npz')))


def _load_npz(path):
    try:
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    except (zipfile.BadZipFile, EOFError, ValueError, OSError, PermissionError):
        return None


def _load_history(run_dir):
    path = os.path.join(run_dir, 'history.json')
    try:
        with open(path) as fh:
            return json.load(fh), os.path.getmtime(path)
    except (OSError, json.JSONDecodeError):
        return [], None


class _SnapshotSourceBase:
    def __init__(self, run_dir):
        self.run_dir = os.path.abspath(run_dir)
        self.p = load_params(self.run_dir)
        self.static = None
        self.started = False
        self.finished = False
        self.closed = False
        self.paused = False
        self.step_credits = 0
        self.hist, self.hist_mtime = _load_history(self.run_dir)
        self._metrics_by_time = {round(m['time'], 6): m for m in self.hist}

    def _start_from(self, snap):
        n_steps = int(round(self.p.t_total / self.p.dt)) if self.p.dt > 0 else 0
        msg, static = frames.start_from_snap(snap, self.p, run_dir=self.run_dir, n_steps=n_steps)
        msg['history'] = self.hist
        self.static = static
        self.started = True
        return msg

    def _frame_from(self, snap):
        t = float(snap.get('time', 0.0))
        self._refresh_history()
        metrics = self._metrics_by_time.get(round(t, 6))
        keys = ('time', 'n_bridges', 'porosity', 'K_kozeny_carman', 'func_lf', 'phi_solid',
                'n_contacts', 'n_bridging_cells', 'tissue_frac', 'n_locked_in_cells', 'bridge_force_mean')
        m = {k: float(metrics[k]) for k in keys if metrics and k in metrics} if metrics else None
        msg = frames.frame_from_snap(snap, self.p, self.static, metrics=m)
        msg['stats'] = {'steps_per_s': 0.0, 'every': self.p.save_every, 'n_dropped': 0}
        return msg

    def _refresh_history(self):
        path = os.path.join(self.run_dir, 'history.json')
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return
        if self.hist_mtime is None or mtime > self.hist_mtime:
            hist, mt = _load_history(self.run_dir)
            if mt is not None:
                self.hist, self.hist_mtime = hist, mt
                self._metrics_by_time = {round(m['time'], 6): m for m in hist}


class SnapshotTailSource(_SnapshotSourceBase):
    """Follow a run directory: yield each new snapshot as it appears.

    start='latest' attaches to a running step 3 and shows only new frames;
    start='first' replays what exists and then keeps following.
    """

    def __init__(self, run_dir, poll_s=0.5, start='latest', idle_finish_s=None):
        super().__init__(run_dir)
        self.poll_s = poll_s
        self.seen = set()
        self.last_poll = 0.0
        self.idle_finish_s = idle_finish_s
        self.last_new = time.time()
        files = _snap_files(self.run_dir)
        if start == 'latest' and len(files) > 1:
            self.seen.update(files[:-1])

    def poll(self):
        now = time.time()
        if now - self.last_poll < self.poll_s:
            return []
        self.last_poll = now
        msgs = []
        for path in _snap_files(self.run_dir):
            if path in self.seen:
                continue
            snap = _load_npz(path)
            if snap is None:
                continue                      # still being written; retry next poll
            self.seen.add(path)
            self.last_new = now
            if not self.started:
                msgs.append(self._start_from(snap))
            msgs.append(self._frame_from(snap))
        if not msgs and self.idle_finish_s is not None and now - self.last_new > self.idle_finish_s:
            self.finished = True
            msgs.append({'kind': 'end', 'reason': 'idle', 't': 0.0, 'step': 0})
        return msgs

    def set_speed(self, every):
        pass


class ReplaySource(_SnapshotSourceBase):
    """Play back the snapshots of a finished run at a fixed cadence."""

    def __init__(self, run_dir, fps=10.0, loop=False, start_index=0):
        super().__init__(run_dir)
        self.files = _snap_files(self.run_dir)
        if not self.files:
            raise FileNotFoundError(f"no snapshots under {self.run_dir}/snapshots")
        self.fps = max(0.1, float(fps))
        self.loop = loop
        self.i = max(0, min(start_index, len(self.files) - 1))
        self.last_emit = 0.0
        self.stride = 1

    def set_speed(self, every):
        self.stride = max(1, int(every))

    def poll(self):
        now = time.time()
        if self.finished:
            return []
        if not self.started:
            snap = _load_npz(self.files[self.i])
            if snap is None:
                return []
            return [self._start_from(snap), self._frame_from(snap)]
        if self.paused and self.step_credits == 0:
            return []
        if self.step_credits == 0 and now - self.last_emit < 1.0 / self.fps:
            return []
        if self.step_credits > 0:
            self.step_credits -= 1
        self.i += self.stride
        if self.i >= len(self.files):
            if self.loop:
                self.i = 0
            else:
                self.finished = True
                return [{'kind': 'end', 'reason': 'replay complete', 't': 0.0, 'step': 0}]
        snap = _load_npz(self.files[self.i])
        if snap is None:
            return []
        self.last_emit = now
        return [self._frame_from(snap)]

    def seek(self, index):
        self.i = max(0, min(int(index), len(self.files) - 1))
        self.step_credits = 1


__all__ = ['SnapshotTailSource', 'ReplaySource']
