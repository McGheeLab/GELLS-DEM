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

Run:  python -m unittest tests.test_legacy_identity -v
"""

import json
import os
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


def _snapshot_files(run_dir, sub='snapshots', prefix='snap_'):
    d = os.path.join(run_dir, sub)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d)
                  if f.startswith(prefix) and f.endswith('.npz'))


class TestLegacyIdentity(unittest.TestCase):

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
        for name in mf.REFERENCE_RUNS:
            with self.subTest(run=name):
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
