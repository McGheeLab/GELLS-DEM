"""
Regression gate: the current code must reproduce the V2.7 fixtures exactly.
============================================================================

Fixtures are generated once on the pre-refactor code by ``tests/make_fixtures.py``.
This test re-runs the same configurations on the current code and requires

  * ``params.json`` produced by ``step1_config.py --trial`` to agree on every
    key present in the fixture (new keys are allowed; ``output_dir`` and
    ``resume_from`` are pipeline-managed and excluded), and
  * every ``history.json`` value and every array in every snapshot / field
    file to be **bit-identical** (``atol = 0``) for the four reference runs.

This is the hard gate for Phases 0-4 of the V3.0 plan (the Python path).
The compiled kernels (Phase 6) are covered by the ``*_vs_reference`` tests
instead, with documented tolerances.

PLATFORM GATING (V3.3). Exact equality of a RUN is only meaningful on the
machine the fixture was blessed on. The stored fixtures were generated on
Windows / CPython 3.14 / numpy 2.5; elsewhere the packing differs by ~1e-13
(different libm, different numpy reduction order) and in ``run3d_spheres``
that flips a discrete bridge-formation decision at t = 1.0, after which the
trajectories bifurcate to O(1). No tolerance can bridge that, so
``test_reference_runs_bit_identical`` SKIPS when the fingerprint does not
match. ``tests/test_local_identity.py`` provides the same guarantee against
a baseline blessed on the current machine.

``params.json`` is pure config resolution with no float dynamics, so
``test_step1_params_json_identical`` is NOT platform-gated and always runs.

Run:  python -m unittest tests.test_legacy_identity -v
"""

import json
import os
import platform
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_fixtures as mf  # noqa: E402

FIXTURES = mf.FIXTURES
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


class TestLegacyIdentity(unittest.TestCase):

    # Runs a later version deliberately supersedes; see mf.SUPERSEDED_RUNS.
    # test_local_identity overrides this to {} -- the local baseline is blessed
    # from the CURRENT tree, so nothing is superseded relative to it.
    superseded = mf.SUPERSEDED_RUNS


    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(FIXTURES, 'manifest.json')):
            raise unittest.SkipTest(
                "No fixtures under tests/fixtures/ - generate them on the "
                "V2.7 code with `python tests/make_fixtures.py` first.")
        cls.tmp = tempfile.mkdtemp(prefix='gels_identity_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    # ── helpers ─────────────────────────────────────────────────────────

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

    # ── tests ───────────────────────────────────────────────────────────

    def test_reference_runs_bit_identical(self):
        why = fingerprint_mismatch(FIXTURES)
        if why:
            self.skipTest(why + " - see tests/test_local_identity.py")
        for name in mf.REFERENCE_RUNS:
            with self.subTest(run=name):
                if name in self.superseded:
                    self.skipTest(self.superseded[name])
                ref_dir = os.path.join(FIXTURES, name)
                self.assertTrue(os.path.isdir(ref_dir), f"missing fixture {name}")
                cur_dir = os.path.join(self.tmp, name)
                mf.make_run(name, cur_dir)
                self._compare_history(ref_dir, cur_dir)
                self._compare_npz_dir(ref_dir, cur_dir, 'snapshots', 'snap_')
                self._compare_npz_dir(ref_dir, cur_dir, 'fields', 'fields_')

    def test_step1_params_json_identical(self):
        for trial in mf.PARAMS_TRIALS:
            with self.subTest(trial=trial):
                ref_path = os.path.join(FIXTURES, mf.params_fixture_name(trial))
                self.assertTrue(os.path.isfile(ref_path), f"missing fixture for {trial}")
                cur_path = os.path.join(self.tmp, mf.params_fixture_name(trial))
                mf.make_params_fixture(trial, cur_path)
                with open(ref_path) as f:
                    ref = json.load(f)
                with open(cur_path) as f:
                    cur = json.load(f)
                from gels.engine import Params
                aliases = getattr(Params, 'LEGACY_ALIASES', {})
                for key, val in ref.items():
                    if key in EXCLUDED_PARAM_KEYS:
                        continue
                    # V3.0 renamed some fields; the value must survive under the new name
                    cur_key = key if key in cur else aliases.get(key, key)
                    self.assertIn(cur_key, cur, f"{trial}: params.json lost key {key!r}")
                    self._assert_scalar_equal(val, cur[cur_key], f"{trial}: params[{key!r}]")


if __name__ == '__main__':
    unittest.main()
