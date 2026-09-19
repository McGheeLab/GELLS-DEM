"""
Background snapshot writer (V3.0 Phase 6e).
==========================================

``write_npz_atomic`` writes a dict of arrays as a zip of ``.npy`` members —
the layout ``np.savez`` produces, so ``np.load`` reads it unchanged — with a
selectable deflate level (level 1 is 3–4× faster than numpy's default 6 for
a few percent more bytes) through a ``.tmp`` file renamed into place, so a
reader tailing the directory never sees a half-written file.

``SnapshotWriter`` runs those writes on a background thread fed by a bounded
queue: the simulation hands over already-copied arrays and continues; when
the queue is full (the disk cannot keep up) ``submit`` blocks, which is the
intended back-pressure. ``close`` drains the queue and joins the thread.
Errors are collected, not raised in the simulation thread.
"""

import os
import queue
import threading
import zipfile

import numpy as np


def write_npz_atomic(path, arrays, compresslevel=1):
    """Write ``arrays`` (name → ndarray-like) to ``path`` as an np.load-compatible zip."""
    tmp = path + '.tmp'
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if compresslevel is None or compresslevel <= 0:
        compression = zipfile.ZIP_STORED
        kw = {}
    else:
        compression = zipfile.ZIP_DEFLATED
        kw = {'compresslevel': int(compresslevel)}
    with zipfile.ZipFile(tmp, 'w', compression=compression, allowZip64=True, **kw) as zf:
        for name, arr in arrays.items():
            with zf.open(name + '.npy', 'w', force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(arr), allow_pickle=False)
    os.replace(tmp, path)


class SnapshotWriter:
    """Bounded-queue background writer of snapshot / field files."""

    def __init__(self, depth=2, compresslevel=1):
        self.compresslevel = compresslevel
        self.q = queue.Queue(maxsize=max(1, int(depth)))
        self.errors = []
        self.n_written = 0
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name='gels-snapshot-writer', daemon=True)
        self._thread.start()

    def submit(self, path, arrays):
        """Queue one file; blocks while ``depth`` files are still pending."""
        if self._closed:
            raise RuntimeError("SnapshotWriter is closed")
        self.q.put((path, arrays))

    def _loop(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break
            path, arrays = item
            try:
                write_npz_atomic(path, arrays, self.compresslevel)
                self.n_written += 1
            except Exception as exc:          # keep the simulation alive; report at close
                self.errors.append((path, repr(exc)))
            finally:
                self.q.task_done()

    def flush(self):
        """Wait until every queued file has been written."""
        self.q.join()

    def close(self):
        """Drain, stop the thread and return the list of (path, error) failures."""
        if not self._closed:
            self._closed = True
            self.q.put(None)
            self._thread.join()
        return self.errors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


__all__ = ['SnapshotWriter', 'write_npz_atomic']
