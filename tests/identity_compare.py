"""
Shared machinery for the identity gate.
=======================================

Comparison helpers and the platform fingerprint check, factored out of the
retired V2.7 gate (V3.5). See ``tests/test_identity.py``, which is now the only
identity gate: the V2.7 oracle it used to serve was retired because pinning
every deliberate default change to a version nobody runs was costing more than
it caught, and because it froze ``gels/kernels/reference.py``.

``tests/fixtures/`` is kept on disk as a historical record of V2.7's behaviour
-- the V2.7 code is no longer in the tree, so those files are the only copy --
but nothing gates on it any more.
"""

import json
import os
import platform
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_fixtures as mf  # noqa: E402

EXCLUDED_PARAM_KEYS = {'output_dir', 'resume_from'}


def fingerprint_mismatch(fixtures_dir):
    """Reason string if `fixtures_dir` was blessed on a different platform, else ''."""
    path = os.path.join(fixtures_dir, 'manifest.json')
    try:
        with open(path) as fh:
            stored = json.load(fh).get('fingerprint')
    except (OSError, ValueError):
        return ''                      # no manifest / unreadable: let the test run
    if not stored:
        # Pre-V3.3 manifest: no 'fingerprint' block. Derive what we can from
        # 'generated_with' (python/numpy minor + the OS name that leads
        # platform.platform(), e.g. 'Windows-11-...' / 'macOS-15...').
        try:
            with open(path) as fh:
                gen = json.load(fh).get('generated_with') or {}
        except (OSError, ValueError):
            return ''
        if not gen:
            return ''
        import numpy as _np
        stored = {
            'system': str(gen.get('platform', '')).split('-')[0],
            'python': '.'.join(str(gen.get('python', '')).split('.')[:2]),
            'numpy': '.'.join(str(gen.get('numpy', '')).split('.')[:2]),
        }
        here = {
            'system': platform.platform().split('-')[0],
            'python': '.'.join(platform.python_version_tuple()[:2]),
            'numpy': '.'.join(_np.__version__.split('.')[:2]),
        }
        diff = {k: f"{stored[k]} != {here[k]}" for k in here if stored[k] and stored[k] != here[k]}
        if not diff:
            return ''
        return (f"fixtures in {os.path.basename(fixtures_dir)} were blessed on a different "
                f"platform ({diff}); exact-equality run comparison is not meaningful there")
    here = mf.platform_fingerprint()
    if stored == here:
        return ''
    diff = {k: f"{stored.get(k)} != {here.get(k)}" for k in here if stored.get(k) != here.get(k)}
    return (f"fixtures in {os.path.basename(fixtures_dir)} were blessed on a different "
            f"platform ({diff}); exact-equality run comparison is not meaningful there")


def _snapshot_files(run_dir, sub='snapshots', prefix='snap_'):
    d = os.path.join(run_dir, sub)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith('.npz'))


class IdentityComparison:
    """Mixin: bit-identical comparison of a run directory against a baseline."""

    def _assert_scalar_equal(self, ref, cur, where):
        if isinstance(ref, float) and ref != ref:          # NaN
            self.assertTrue(isinstance(cur, float) and cur != cur, where)
            return
        self.assertEqual(ref, cur, where)

    def _compare_history(self, ref_dir, cur_dir):
        with open(os.path.join(ref_dir, 'history.json')) as f:
            ref = json.load(f)
        with open(os.path.join(cur_dir, 'history.json')) as f:
            cur = json.load(f)
        self.assertEqual(len(ref), len(cur), "history length")
        for k, (hr, hc) in enumerate(zip(ref, cur)):
            missing = set(hr) - set(hc)
            self.assertFalse(missing, f"history[{k}] lost keys {sorted(missing)}")
            for key, val in hr.items():
                self._assert_scalar_equal(val, hc[key], f"history[{k}][{key!r}]")

    def _compare_npz_dir(self, ref_dir, cur_dir, sub, prefix):
        ref_files = _snapshot_files(ref_dir, sub, prefix)
        cur_files = _snapshot_files(cur_dir, sub, prefix)
        self.assertEqual(ref_files, cur_files, f"{sub}/ file list")
        for name in ref_files:
            a = dict(np.load(os.path.join(ref_dir, sub, name), allow_pickle=False))
            b = dict(np.load(os.path.join(cur_dir, sub, name), allow_pickle=False))
            missing = set(a) - set(b)
            self.assertFalse(missing, f"{sub}/{name} lost keys {sorted(missing)}")
            for key, arr in a.items():
                cur = b[key]
                self.assertEqual(arr.shape, cur.shape, f"{sub}/{name}[{key!r}] shape")
                if arr.dtype.kind in 'fc':
                    same = np.array_equal(arr, cur.astype(arr.dtype, copy=False),
                                          equal_nan=True)
                else:
                    same = np.array_equal(arr, cur)
                if not same and arr.dtype.kind in 'fc':
                    diff = np.nanmax(np.abs(arr.astype(float) - cur.astype(float)))
                    self.fail(f"{sub}/{name}[{key!r}] differs (max |diff| = {diff:g})")
                self.assertTrue(same, f"{sub}/{name}[{key!r}] differs")


__all__ = ['EXCLUDED_PARAM_KEYS', 'IdentityComparison', 'fingerprint_mismatch']
