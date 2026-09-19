"""
Contact records shared by the compiled force kernels (V3.0 Phase 6a).
====================================================================

``alloc_records`` allocates the per-pair arrays the pair passes fill;
``ContactSoA`` is the structure-of-arrays replacement for the reference's
list of contact dicts (it iterates as dicts, so ``save_snapshot_to_disk``,
``viz`` and any user code that indexes ``c['overlap']`` keep working);
``clips_to_lists`` converts the fixed-width clip-plane arrays into the
per-granule tuple lists the Python renderer consumes; ``add_active_noise``
is the reference's vectorised noise term (kept in numpy so the random
stream is unchanged).
"""

import numpy as np

# Per-pair record names (float64 unless noted); 3D adds nz, cz, Ftz, dz.
_REC_2D = ('overlap', 'nx', 'ny', 'cx', 'cy', 'Reff', 'Fn', 'a', 'A', 'Ftx', 'Fty', 'd', 'dx', 'dy')
_REC_3D_EXTRA = ('nz', 'cz', 'Ftz', 'dz')


def alloc_records(M, dim):
    """Zeroed per-pair record arrays for ``M`` pairs (``hit`` is int8)."""
    rec = {name: np.zeros(M, dtype=np.float64) for name in _REC_2D}
    if dim == 3:
        for name in _REC_3D_EXTRA:
            rec[name] = np.zeros(M, dtype=np.float64)
    rec['hit'] = np.zeros(M, dtype=np.int8)     # 1 contact, 0 none, -1 skipped (coincident centres)
    return rec


class ContactSoA:
    """Contacts of one force evaluation as columns (one entry per contact with overlap > 0).

    Behaves like the reference list of dicts: ``len()``, truthiness,
    iteration and indexing yield dicts with Python scalars; ``column(name)``
    gives the array directly.
    """

    FIELDS = ('i', 'j', 'cx', 'cy', 'cz', 'nx', 'ny', 'nz', 'overlap', 'R_eff', 'F_normal',
              'A_contact', 'a_contact', 'gtype_i', 'gtype_j', 'species_i', 'species_j',
              'f_i', 'f_j', 'E_star', 'W', 'tau_0', 'kappa')

    def __init__(self, arrays):
        self.arrays = arrays
        self.n = int(len(arrays['i']))

    @classmethod
    def from_records(cls, pair_i, pair_j, rec, gs, c_Estar, c_W, c_tau, dF, c_kappa, dim):
        sel = np.nonzero((rec['hit'] == 1) & (rec['overlap'] > 0.0))[0]
        i = pair_i[sel].astype(np.int32)
        j = pair_j[sel].astype(np.int32)
        zeros = np.zeros(sel.size)
        arrays = {
            'i': i, 'j': j,
            'cx': rec['cx'][sel], 'cy': rec['cy'][sel],
            'cz': rec['cz'][sel] if dim == 3 else zeros,
            'nx': rec['nx'][sel], 'ny': rec['ny'][sel],
            'nz': rec['nz'][sel] if dim == 3 else zeros,
            'overlap': rec['overlap'][sel], 'R_eff': rec['Reff'][sel],
            'F_normal': rec['Fn'][sel] + dF[sel],          # MC-DEM correction folded in, as the reference does
            'A_contact': rec['A'][sel], 'a_contact': rec['a'][sel],
            'gtype_i': gs.gtype[i], 'gtype_j': gs.gtype[j],
            'species_i': gs.species_id[i], 'species_j': gs.species_id[j],
            'f_i': gs.f[i], 'f_j': gs.f[j],
            'E_star': c_Estar[sel], 'W': c_W[sel], 'tau_0': c_tau[sel],
            'kappa': c_kappa[sel],
        }
        return cls(arrays)

    def __len__(self):
        return self.n

    def __bool__(self):
        return self.n > 0

    def column(self, name):
        return self.arrays[name]

    def __getitem__(self, k):
        if isinstance(k, slice):
            return [self[q] for q in range(*k.indices(self.n))]
        if k < 0:
            k += self.n
        if k < 0 or k >= self.n:
            raise IndexError(k)
        out = {}
        for name, arr in self.arrays.items():
            v = arr[k]
            out[name] = v.item() if hasattr(v, 'item') else v
        return out

    def __iter__(self):
        for k in range(self.n):
            yield self[k]

    def to_dicts(self):
        return list(iter(self))

    def __repr__(self):
        return f"ContactSoA(n={self.n})"


def clips_to_lists(gs, clip_n, clip_d, clip_cnt):
    """Append the kernel clip planes to ``gs.contact_clips`` (tuples, as the renderer expects)."""
    dim = clip_n.shape[2]
    lists = gs.contact_clips
    for i in np.nonzero(clip_cnt)[0]:
        c = int(clip_cnt[i])
        if dim == 2:
            lists[i].extend((float(clip_n[i, q, 0]), float(clip_n[i, q, 1]), float(clip_d[i, q]))
                            for q in range(c))
        else:
            lists[i].extend((float(clip_n[i, q, 0]), float(clip_n[i, q, 1]), float(clip_n[i, q, 2]),
                             float(clip_d[i, q])) for q in range(c))


def add_active_noise(gs, p, rng, F):
    """Active noise on adhesive granules — the reference's vectorised code, unchanged."""
    N = gs.N
    if p.T_active > 0:
        func_mask = gs.adhesive_mask[:N]
        n_func = int(np.sum(func_mask))
        if n_func > 0:
            gamma_func = p.drag_scale * gs.r[:N][func_mask]
            noise_amp = (np.sqrt(2 * gamma_func * p.T_active / p.dt)
                         * np.sqrt(gs.activity[:N][func_mask]))
            for ax in range(F.shape[1]):
                F[:N, ax][func_mask] += noise_amp * rng.standard_normal(n_func)


__all__ = ['alloc_records', 'ContactSoA', 'clips_to_lists', 'add_active_noise']
