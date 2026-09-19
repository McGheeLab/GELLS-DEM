"""Output helpers (V3.0): background snapshot writer with atomic, np.load-compatible files."""

from gels.io.writer import SnapshotWriter, write_npz_atomic  # noqa: F401

__all__ = ['SnapshotWriter', 'write_npz_atomic']
