"""
Snapshot helpers shared by viz2 and the live viewer (matplotlib-free).
====================================================================

Moved out of ``viz2/common.py`` in V3.0: 3D detection, z-midplane slicing,
vectorised cell world positions, phase-field reconstruction and the periodic
ghost-image helpers. ``viz2.common`` re-exports them; the live viewer imports
this module directly because it must not pin the Agg backend.
"""

import numpy as np

from gels.engine import GranuleSystem, Params, render_fields, render_fields_3d


def is_3d(snap, p):
    """Detect whether the data is from a 3D simulation."""
    mode = getattr(p, 'mode', '2D')
    if mode == '3D':
        return True
    if 'z' in snap and len(snap['z']) > 0 and np.any(snap['z'] != 0):
        return True
    return False


def slice_field_z_midplane(field_3d, z_frac=0.5):
    """Extract a 2D slice from a 3D phase field at z = z_frac * Nz."""
    if field_3d.ndim == 2:
        return field_3d
    Nz = field_3d.shape[2]
    iz = min(int(z_frac * Nz), Nz - 1)
    return field_3d[:, :, iz]


def slice_snap_z_midplane(snap, p, z_frac=0.5):
    """Create a 2D-compatible snapshot by slicing 3D data at a z plane.

    Selects granules whose centre is within one bounding radius of the
    plane, projects to 2D (cross-section radii, yaw angle from the
    quaternion), carries the cells and contacts of the selected granules and
    (V3.0) their species indices and functionalization. Adds ``_z_slice``
    and ``_3d_idx`` (the original indices) to the result.
    """
    Lz = float(getattr(p, 'Lz', 0))
    z_mid = Lz * z_frac
    zs = snap['z']
    N = len(zs)

    a_arr = snap.get('a', snap['r']).astype(np.float64)
    b_arr = snap.get('b', snap['r']).astype(np.float64)
    c_arr = snap.get('c', snap['r']).astype(np.float64)
    r_bound = np.maximum(np.maximum(a_arr, b_arr), c_arr)

    dz = np.abs(zs - z_mid)
    mask = dz < r_bound
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(N)

    n2_arr = snap.get('n2', np.full(N, 2.0)).astype(np.float64)
    frac_z = np.clip(dz[idx] / np.maximum(c_arr[idx], 1e-6), 0.0, 0.999)
    shrink = np.maximum(1.0 - frac_z ** n2_arr[idx], 0.0) ** (1.0 / n2_arr[idx])

    a_eff = a_arr[idx] * shrink
    b_eff = b_arr[idx] * shrink
    r_eff = snap['r'][idx] * shrink

    if 'quat' in snap:
        quats = snap['quat'][idx]
        w, qx, qy, qz = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
        theta_eff = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    else:
        theta_eff = snap.get('theta', np.zeros(N))[idx]

    old_to_new = np.full(N, -1, dtype=int)
    old_to_new[idx] = np.arange(len(idx))

    snap_2d = {
        'x': snap['x'][idx].copy(),
        'y': snap['y'][idx].copy(),
        'r': r_eff,
        'a': a_eff,
        'b': b_eff,
        'n1': snap.get('n1', np.full(N, 2.0))[idx].copy(),
        'theta': theta_eff,
        'gtype': snap['gtype'][idx].copy(),
    }
    for key in ('species_id', 'f', 'E_gran', 'nu_gran'):     # V3.0 species arrays
        if key in snap and snap[key] is not None and len(snap[key]) == N:
            snap_2d[key] = np.asarray(snap[key])[idx].copy()

    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    if len(cell_gi) > 0:
        cell_mask = np.isin(cell_gi, idx)
        cell_keys = [k for k in snap if k.startswith('cell_') and k != 'cell_offset']
        for k in cell_keys:
            arr = snap[k]
            if len(arr) == len(cell_gi):
                snap_2d[k] = arr[cell_mask].copy()
        if 'cell_granule_id' in snap_2d:
            snap_2d['cell_granule_id'] = old_to_new[snap_2d['cell_granule_id']]
        if 'cell_bridge_target' in snap_2d:
            bt = snap_2d['cell_bridge_target'].copy()
            valid = (bt >= 0) & (bt < N)
            bt[valid] = old_to_new[bt[valid]]
            bt[bt < 0] = -1
            snap_2d['cell_bridge_target'] = bt

    ci = snap.get('contact_i', np.array([], dtype=int))
    if len(ci) > 0:
        cj = snap.get('contact_j', np.array([], dtype=int))
        contact_mask = np.isin(ci, idx) & np.isin(cj, idx)
        contact_keys = [k for k in snap if k.startswith('contact_')]
        for k in contact_keys:
            arr = snap[k]
            if len(arr) == len(ci):
                snap_2d[k] = arr[contact_mask].copy()
        if 'contact_i' in snap_2d:
            snap_2d['contact_i'] = old_to_new[snap_2d['contact_i']]
            snap_2d['contact_j'] = old_to_new[snap_2d['contact_j']]

    snap_2d['_z_slice'] = z_mid
    snap_2d['_3d_idx'] = idx
    return snap_2d


def cell_world_positions(snap):
    """World (x, y) of every cell in a 2D snapshot — vectorised (V3.0)."""
    from gels.live.frames import cell_world_xy
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    if len(cell_gi) == 0:
        return np.array([]), np.array([])
    xs, ys = snap['x'], snap['y']
    N = len(xs)
    return cell_world_xy(
        xs, ys, snap.get('theta', np.zeros(N)),
        snap.get('a', snap['r']), snap.get('b', snap['r']),
        snap.get('n1', snap.get('n_shape', np.full(N, 2.0))),
        cell_gi, snap.get('cell_theta_local', np.zeros(len(cell_gi))))


def cell_world_positions_3d(snap):
    """World (x, y, z) of every cell in a 3D snapshot — vectorised (V3.0)."""
    from gels.live.frames import cell_world_xyz
    cell_gi = snap.get('cell_granule_id', np.array([], dtype=int))
    if len(cell_gi) == 0:
        return np.array([]), np.array([]), np.array([])
    xs = snap['x']
    N = len(xs)
    return cell_world_xyz(
        xs, snap['y'], snap['z'],
        snap.get('quat', np.tile([1.0, 0.0, 0.0, 0.0], (N, 1))),
        snap.get('a', snap['r']), snap.get('b', snap['r']), snap.get('c', snap['r']),
        snap.get('n1', np.full(N, 2.0)), snap.get('n2', np.full(N, 2.0)),
        cell_gi, snap.get('cell_eta_local', np.zeros(len(cell_gi))),
        snap.get('cell_omega_local', np.zeros(len(cell_gi))))


def ensure_phase_fields(snap, p):
    """Ensure phi_f, phi_i, phi_v are in the snapshot; rebuild from particles if not.

    Uses ``GranuleSystem.from_snapshot_dict`` (V3.0), which reproduces the
    former shim's conventions (exact circle test in 2D, no clips), and the
    engine renderers. Mutates ``snap`` and returns (phi_f, phi_i, phi_v).
    """
    if 'phi_f' in snap and snap['phi_f'] is not None:
        return snap['phi_f'], snap['phi_i'], snap['phi_v']
    pp = p if p is not None else Params()
    mode = '3D' if is_3d(snap, pp) else '2D'
    gs = GranuleSystem.from_snapshot_dict(snap, pp, mode=mode)
    if mode == '3D':
        phi_f, phi_i, phi_v = render_fields_3d(gs, pp)
    else:
        phi_f, phi_i, phi_v = render_fields(gs, pp)
    snap['phi_f'] = phi_f
    snap['phi_i'] = phi_i
    snap['phi_v'] = phi_v
    return phi_f, phi_i, phi_v


def _periodic_offsets(x, r_bound, L):
    """Periodic image shifts for an object of extent r_bound at x (always includes 0)."""
    offsets = [0.0]
    if x - r_bound < 0:
        offsets.append(L)
    if x + r_bound > L:
        offsets.append(-L)
    return offsets


def _min_image(dx, L):
    """Minimum image convention: shift dx to [-L/2, L/2)."""
    return dx - L * np.round(dx / L)


__all__ = [
    'is_3d', 'slice_field_z_midplane', 'slice_snap_z_midplane', 'cell_world_positions',
    'cell_world_positions_3d', 'ensure_phase_fields', '_periodic_offsets', '_min_image',
]


def functionalization(snap):
    """Per-granule f: the saved array, or 1 / 0 from gtype for legacy snapshots."""
    if 'f' in snap:
        return np.asarray(snap['f'], dtype=np.float64)
    return np.where(np.asarray(snap['gtype']) == 0, 1.0, 0.0)


def contact_friction(snap, p, ci, cj):
    """Friction shear stress tau_0 (Pa) of each contact (ci, cj) from the coverage mixing rule.

    Reduces to the legacy ff / if / ii lookup when f is 0 or 1.
    """
    from gels.materials import mix_pair
    f = functionalization(snap)
    ci = np.asarray(ci, dtype=int)
    cj = np.asarray(cj, dtype=int)
    if ci.size == 0:
        return np.zeros(0)
    cc = float(getattr(p, 'tau_0_cc', getattr(p, 'tau_0_ff', 2000.0)))
    cb = float(getattr(p, 'tau_0_cb', getattr(p, 'tau_0_if', 500.0)))
    bb = float(getattr(p, 'tau_0_bb', getattr(p, 'tau_0_ii', 50.0)))
    return np.array([mix_pair(float(f[i]), float(f[j]), cc, cb, bb) for i, j in zip(ci, cj)])

