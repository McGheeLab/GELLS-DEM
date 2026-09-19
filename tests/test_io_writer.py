"""
Background snapshot writer (V3.0 Phase 6e).

  * write_npz_atomic produces files np.load reads back exactly (ints, floats,
    bools, 0-d scalars), leaves no .tmp behind and honours the deflate level,
  * SnapshotWriter writes every submitted file, blocks on a full queue rather
    than dropping, and reports errors at close instead of raising,
  * run() with perf_async_io writes the expected snapshots, history and
    metadata and returns no in-memory snapshots by default.

Run:  python -m unittest tests.test_io_writer -v
"""

import glob
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault('MPLBACKEND', 'Agg')

from gels.engine import Params, load_run, run  # noqa: E402
from gels.io import SnapshotWriter, write_npz_atomic  # noqa: E402


class TestWriteNpz(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_io_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_roundtrip_and_atomic(self):
        rng = np.random.default_rng(0)
        arrays = {
            'time': np.float64(2.5),
            'x': rng.normal(size=1000),
            'ids': rng.integers(0, 10, size=1000).astype(np.int32),
            'flags': rng.random(1000) > 0.5,
            'grid': rng.random((32, 32)).astype(np.float32),
        }
        path = os.path.join(self.tmp, 'snap_0001.npz')
        write_npz_atomic(path, arrays, compresslevel=1)
        self.assertTrue(os.path.exists(path))
        self.assertFalse(os.path.exists(path + '.tmp'))
        with np.load(path, allow_pickle=False) as d:
            self.assertEqual(set(d.files), set(arrays))
            for k, v in arrays.items():
                np.testing.assert_array_equal(d[k], np.asarray(v))
                self.assertEqual(d[k].dtype, np.asarray(v).dtype)
        # level 0 (stored) is larger than level 6, level 1 in between or equal
        p0 = os.path.join(self.tmp, 'l0.npz')
        p6 = os.path.join(self.tmp, 'l6.npz')
        write_npz_atomic(p0, arrays, compresslevel=0)
        write_npz_atomic(p6, arrays, compresslevel=6)
        self.assertGreater(os.path.getsize(p0), os.path.getsize(p6))
        self.assertLessEqual(os.path.getsize(p6), os.path.getsize(path) + 64)

    def test_writer_writes_all_and_reports_errors(self):
        w = SnapshotWriter(depth=2, compresslevel=1)
        paths = [os.path.join(self.tmp, f'snap_{k:04d}.npz') for k in range(6)]
        for k, path in enumerate(paths):
            w.submit(path, {'k': np.full(10, k)})
        # an unwritable target is reported, not raised
        w.submit(os.path.join(self.tmp, 'no_such_dir', 'x', 'y.npz') if os.name != 'nt'
                 else 'Z:\\definitely\\not\\here\\y.npz', {'a': np.zeros(1)})
        errors = w.close()
        for k, path in enumerate(paths):
            with np.load(path) as d:
                self.assertEqual(int(d['k'][0]), k)
        self.assertEqual(len(errors), 1)
        self.assertEqual(w.n_written, 6)
        self.assertFalse(glob.glob(os.path.join(self.tmp, '*.tmp')))
        with self.assertRaises(RuntimeError):
            w.submit(paths[0], {'k': np.zeros(1)})


class TestRunAsyncIO(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='gels_async_')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_writes_snapshots_through_the_writer(self):
        p = Params(mode='2D', Lx=400.0, Ly=400.0, t_total=2.0, dt=0.5, save_every_h=0.5,
                   phi_solid_target=0.55, cell_surface_coverage=1.0, packing_settle_steps=20,
                   save_data=True, compress_archive=False, output_dir=self.tmp, Ngrid=40,
                   perf_async_io=True, perf_io_compresslevel=1)
        self.assertFalse(p.perf_keep_snaps_in_memory)
        hist, snaps, p_out, gs = run(p, seed=2)
        self.assertEqual(snaps, [])
        files = sorted(glob.glob(os.path.join(self.tmp, 'snapshots', 'snap_*.npz')))
        self.assertEqual(len(files), len(hist))
        self.assertFalse(glob.glob(os.path.join(self.tmp, 'snapshots', '*.tmp')))
        hist2, snaps2, p2, meta = load_run(self.tmp)
        self.assertEqual(len(hist2), len(hist))
        self.assertEqual(len(snaps2), len(files))
        with np.load(files[-1], allow_pickle=False) as d:
            self.assertAlmostEqual(float(d['time']), hist[-1]['time'])
            np.testing.assert_array_equal(d['x'], gs.x)
        with open(os.path.join(self.tmp, 'metadata.json')) as fh:
            meta_json = json.load(fh)
        self.assertEqual(meta_json['stop_reason'], 'complete')

    def test_sync_path_still_works(self):
        p = Params(mode='2D', Lx=300.0, Ly=300.0, t_total=1.0, dt=0.5, save_every_h=0.5,
                   phi_solid_target=0.5, packing_settle_steps=10, save_data=True, compress_archive=False,
                   output_dir=self.tmp, Ngrid=32, perf_async_io=False, perf_keep_snaps_in_memory=True)
        hist, snaps, p_out, gs = run(p, seed=3)
        self.assertEqual(len(snaps), len(hist))
        self.assertEqual(len(glob.glob(os.path.join(self.tmp, 'snapshots', 'snap_*.npz'))), len(hist))


if __name__ == '__main__':
    unittest.main()
