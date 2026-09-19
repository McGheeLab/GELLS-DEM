"""
Regression gate: behaviour must not drift from the PLATFORM-LOCAL baseline.
===========================================================================

``tests/test_legacy_identity.py`` pins the current code to the V2.7 oracle in
``tests/fixtures/``, but that oracle was blessed on one specific machine and
exact equality does not survive a platform change: packing positions differ by
~1e-13 (different libm, different numpy reduction order), and in
``run3d_spheres`` that flips a discrete bridge-formation decision at t = 1.0,
after which the trajectories bifurcate to O(1). So on any other machine the
V2.7 gate skips and this one takes over.

The baseline in ``tests/fixtures_local/`` is generated from the current tree on
the current machine::

    python tests/make_fixtures.py --local            # first time
    python tests/make_fixtures.py --local --force    # re-bless, deliberately

and this test then requires bit-identical results for the same four reference
runs. What it guarantees is narrower than the V2.7 gate but is exactly what a
feature branch needs: **the change I am making right now does not alter
behaviour while its flags are off.**

Re-bless the baseline only when a behaviour change is intended and understood;
record why in the changelog, because after re-blessing this gate can no longer
see that change.

Run:  python -m unittest tests.test_local_identity -v
"""

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
# The parent is imported for its comparison helpers and `del`d at the bottom of
# this module: unittest collects ANY TestCase subclass bound in a module
# namespace, regardless of name, so leaving it bound would run the V2.7 gate a
# second time from here.
from test_legacy_identity import TestLegacyIdentity as _TestLegacyIdentity  # noqa: E402
from test_legacy_identity import fingerprint_mismatch  # noqa: E402

FIXTURES_LOCAL = mf.FIXTURES_LOCAL


class TestLocalIdentity(_TestLegacyIdentity):
    """Same comparisons as the V2.7 gate, against the platform-local baseline."""

    # The local baseline is blessed from the CURRENT tree, so no run is
    # superseded relative to it -- including the ones V3.4 superseded in V2.7.
    # This is what stops `SUPERSEDED_RUNS` from becoming a hole in the gate.
    superseded = {}

    @classmethod
    def setUpClass(cls):
        if not os.path.isfile(os.path.join(FIXTURES_LOCAL, 'manifest.json')):
            raise unittest.SkipTest(
                "No baseline under tests/fixtures_local/ - create one with "
                "`python tests/make_fixtures.py --local`.")
        cls.tmp = tempfile.mkdtemp(prefix='gels_local_identity_')

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_reference_runs_bit_identical(self):
        why = fingerprint_mismatch(FIXTURES_LOCAL)
        if why:
            self.skipTest(why + " - re-bless with `make_fixtures.py --local --force`")
        for name in list(mf.REFERENCE_RUNS) + list(mf.LOCAL_ONLY_RUNS):
            with self.subTest(run=name):
                ref_dir = os.path.join(FIXTURES_LOCAL, name)
                self.assertTrue(os.path.isdir(ref_dir), f"missing baseline run {name}")
                cur_dir = os.path.join(self.tmp, name)
                mf.make_run(name, cur_dir)
                self._compare_history(ref_dir, cur_dir)
                self._compare_npz_dir(ref_dir, cur_dir, 'snapshots', 'snap_')
                self._compare_npz_dir(ref_dir, cur_dir, 'fields', 'fields_')

    def test_step1_params_json_identical(self):
        self.skipTest("params.json is platform-independent; covered by test_legacy_identity")


# See the import comment: unbind the parent so it is collected only from its own
# module. TestLocalIdentity keeps it via __bases__, so the subclass still works.
del _TestLegacyIdentity


if __name__ == '__main__':
    unittest.main()
