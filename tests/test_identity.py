"""
Regression gate: behaviour must not drift, measured against a local baseline.
=============================================================================

The one identity gate. It re-runs a handful of small configurations at the
CURRENT shipping defaults and requires the results to be bit-identical to a
baseline blessed from this tree on this machine::

    python tests/make_fixtures.py --local            # first time
    python tests/make_fixtures.py --local --force    # re-bless, deliberately

What it guarantees is exactly what a feature branch needs: **the change I am
making right now does not alter behaviour I did not mean to alter.**

Re-bless only when a behaviour change is intended and understood, and record
why in the changelog -- after re-blessing, this gate can no longer see it.

WHY THERE IS NO LONGER A V2.7 GATE (V3.5). ``tests/test_legacy_identity.py``
pinned the tree to fixtures generated on the pre-refactor V2.7 engine. It was
retired because the project has not run a significant body of simulations on
the old behaviour, so reproducing it was not worth what it cost:

  * every deliberate default change had to be pinned in ``make_fixtures.COMMON``
    to keep the gate green -- five pins had accumulated by V3.5, each one a
    configuration the shipped defaults no longer used;
  * it froze ``gels/kernels/reference.py``, which could not be changed even
    when the kernels it is the oracle for needed to change with it;
  * and it was platform-gated anyway, so on any machine but the one it was
    blessed on it skipped and this gate did the work.

``tests/fixtures/`` is kept on disk as the only surviving record of V2.7's
behaviour. Nothing reads it.

Run:  python -m unittest tests.test_identity -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import make_fixtures as mf  # noqa: E402
from identity_compare import (  # noqa: E402
    EXCLUDED_PARAM_KEYS, IdentityComparison, fingerprint_mismatch,
)

BASELINE = mf.FIXTURES_LOCAL


class TestIdentity(IdentityComparison, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(BASELINE, 'manifest.json')):
            raise unittest.SkipTest(
                "No baseline under tests/fixtures_local/ - create one with "
                "`python tests/make_fixtures.py --local`.")
        cls.tmp = tempfile.mkdtemp(prefix='gels_identity_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reference_runs_bit_identical(self):
        why = fingerprint_mismatch(BASELINE)
        if why:
            self.skipTest(why + " - re-bless with `make_fixtures.py --local --force`")
        for name in mf.REFERENCE_RUNS:
            with self.subTest(run=name):
                ref_dir = os.path.join(BASELINE, name)
                self.assertTrue(os.path.isdir(ref_dir), f"missing baseline run {name}")
                cur_dir = os.path.join(self.tmp, name)
                mf.make_run(name, cur_dir)
                self._compare_history(ref_dir, cur_dir)
                self._compare_npz_dir(ref_dir, cur_dir, 'snapshots', 'snap_')
                self._compare_npz_dir(ref_dir, cur_dir, 'fields', 'fields_')

    def test_step1_params_json_identical(self):
        """Config resolution is pure and platform-independent, so this one is
        never fingerprint-gated."""
        for trial in mf.PARAMS_TRIALS:
            with self.subTest(trial=trial):
                ref_path = os.path.join(BASELINE, mf.params_fixture_name(trial))
                self.assertTrue(os.path.isfile(ref_path), f"missing baseline for {trial}")
                cur_path = os.path.join(self.tmp, mf.params_fixture_name(trial))
                mf.make_params_fixture(trial, cur_path)
                with open(ref_path) as f:
                    ref = json.load(f)
                with open(cur_path) as f:
                    cur = json.load(f)
                for key, val in ref.items():
                    if key in EXCLUDED_PARAM_KEYS:
                        continue
                    self.assertIn(key, cur, f"{trial}: params.json lost key {key!r}")
                    self._assert_scalar_equal(val, cur[key], f"{trial}: params[{key!r}]")


if __name__ == '__main__':
    unittest.main()
