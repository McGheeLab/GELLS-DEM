"""
Frame and start messages for the live viewer, from a GranuleSystem or a snapshot.
=================================================================================

Also home of the **vectorised** cell world-position helpers that replace the
per-cell Python loops of ``viz2/common.py`` (``cell_world_xy`` /
``cell_world_xyz`` reproduce ``gels.engine._cell_world_pos_2d/_3d`` to
floating-point precision, ~1 ms for 10⁵ cells).

Messages are plain dicts of numpy arrays (float32 on the wire, int8/int32
indices) so they pickle cheaply through a ``multiprocessing.Queue``.

    start : static per-run data (domain, species table, radii/shapes,
            cell → granule map)          — sent once
    frame : positions, orientations, cell positions/states, bridge segments,
            optional speeds and metrics  — sent per displayed step
    end   : reason + counters            — sent once
"""

import os

import numpy as np

from gels.engine import CellState

# ──────────────────────────────────────────────────────────────────────
# Vectorised geometry (mirrors the scalar njit leaves in gels.engine)
# ──────────────────────────────────────────────────────────────────────

def sgnpow(x, e):
    """sign(x)·|x|^e, elementwise (the engine's _sgnpow)."""
    x = np.asarray(x, dtype=np.float64)
    return np.sign(x) * np.abs(x) ** e


def superellipse_points(t, a, b, n):
    """Body-frame boundary points of superellipses at parameters t (broadcasting)."""
    t = np.asarray(t, dtype=np.float64)
    ct, st = np.cos(t), np.sin(t)
    e = 2.0 / np.asarray(n, dtype=np.float64)
    return a * sgnpow(ct, e), b * sgnpow(st, e)


def superellipsoid_points(eta, omega, a, b, c, n1, n2):
    """Body-frame surface points of superellipsoids → (M, 3)."""
    e2 = 2.0 / np.asarray(n2, dtype=np.float64)
    e1 = 2.0 / np.asarray(n1, dtype=np.float64)
    ce, se = np.cos(eta), np.sin(eta)
    co, so = np.cos(omega), np.sin(omega)
    x = a * sgnpow(ce, e2) * sgnpow(co, e1)
    y = b * sgnpow(ce, e2) * sgnpow(so, e1)
    z = c * sgnpow(se, e2)
    return np.column_stack([x, y, z])


def quat_rotate_many(q, v):
    """Rotate vectors v (M,3) by unit quaternions q (M,4) as q v q* → (M,3)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
    # t = q * (0, v)
    tw = -x * vx - y * vy - z * vz
    tx = w * vx + y * vz - z * vy
    ty = w * vy - x * vz + z * vx
    tz = w * vz + x * vy - y * vx
    # r = t * conj(q)  (conj = (w, -x, -y, -z)); return the vector part
    rx = -tw * x + tx * w - ty * z + tz * y
    ry = -tw * y + tx * z + ty * w - tz * x
    rz = -tw * z - tx * y + ty * x + tz * w
    return np.column_stack([rx, ry, rz])


def cell_world_xy(x, y, theta, a, b, n1, cell_gid, cell_theta):
    """World (x, y) of cells on their host granules in 2D (vectorised).

    Cells with an invalid host index get (0, 0), as the old loop did.
    """
    gid = np.asarray(cell_gid, dtype=np.int64)
    C = gid.shape[0]
    wx = np.zeros(C)
    wy = np.zeros(C)
    if C == 0:
        return wx, wy
    N = len(x)
    ok = (gid >= 0) & (gid < N)
    g = gid[ok]
    bx, by = superellipse_points(np.asarray(cell_theta)[ok], np.asarray(a)[g],
                                 np.asarray(b)[g], np.asarray(n1)[g])
    th = np.asarray(theta)[g]
    ct, st = np.cos(th), np.sin(th)
    wx[ok] = np.asarray(x)[g] + ct * bx - st * by
    wy[ok] = np.asarray(y)[g] + st * bx + ct * by
    return wx, wy


def cell_world_xyz(x, y, z, quat, a, b, c, n1, n2, cell_gid, cell_eta, cell_omega):
    """World (x, y, z) of cells on their host granules in 3D (vectorised)."""
    gid = np.asarray(cell_gid, dtype=np.int64)
    C = gid.shape[0]
    wx = np.zeros(C)
    wy = np.zeros(C)
    wz = np.zeros(C)
    if C == 0:
        return wx, wy, wz
    N = len(x)
    ok = (gid >= 0) & (gid < N)
    g = gid[ok]
    bp = superellipsoid_points(np.asarray(cell_eta)[ok], np.asarray(cell_omega)[ok],
                               np.asarray(a)[g], np.asarray(b)[g], np.asarray(c)[g],
                               np.asarray(n1)[g], np.asarray(n2)[g])
    wp = quat_rotate_many(np.asarray(quat)[g], bp)
    wx[ok] = np.asarray(x)[g] + wp[:, 0]
    wy[ok] = np.asarray(y)[g] + wp[:, 1]
    wz[ok] = np.asarray(z)[g] + wp[:, 2]
    return wx, wy, wz


def bridge_segments(cw, cell_state, cell_bridge_target, pos, r, L, periodic):
    """Segments from bridging cells to the surface of their target granule.

    cw : (C, dim) cell world positions; pos : (N, dim) granule centres;
    L : (dim,) box lengths. Returns (segments (M, 2, dim), cell indices (M,)).
    Same construction as viz2's bridge ellipses: minimum image when periodic,
    end point = target centre pulled back by the target radius.
    """
    st = np.asarray(cell_state)
    bt = np.asarray(cell_bridge_target, dtype=np.int64)
    N = pos.shape[0]
    sel = np.where((st == int(CellState.BRIDGING)) & (bt >= 0) & (bt < N))[0]
    dim = pos.shape[1]
    if sel.size == 0:
        return np.zeros((0, 2, dim)), sel
    start = cw[sel]
    tgt = pos[bt[sel]]
    d = tgt - start
    if periodic:
        for k in range(dim):
            d[:, k] -= L[k] * np.round(d[:, k] / L[k])
    dist = np.linalg.norm(d, axis=1)
    safe = np.where(dist > 1e-6, dist, 1.0)
    end = start + d - (r[bt[sel]] / safe)[:, None] * d
    end[dist <= 1e-6] = start[dist <= 1e-6]
    seg = np.stack([start, end], axis=1)
    return seg, sel


# ──────────────────────────────────────────────────────────────────────
# Message builders (GranuleSystem or snapshot dict as the source)
# ──────────────────────────────────────────────────────────────────────

def _get(src, key, default=None):
    if isinstance(src, dict):
        v = src.get(key, default)
    else:
        v = getattr(src, key, default)
    return default if v is None else v


def _mode_of(src, p):
    m = _get(src, 'mode', None)
    if isinstance(m, np.ndarray):
        m = str(m)
    if not m:
        m = getattr(p, 'mode', '2D')
    return str(m)


def start_message(gs, p, seed=None, n_steps=0, resume_step=0, run_dir=''):
    """(start message, static dict) for a GranuleSystem or snapshot dict."""
    from viz2.palette import species_table
    mode = _mode_of(gs, p)
    N = len(_get(gs, 'x'))
    r = np.asarray(_get(gs, 'r'), dtype=np.float64)
    a = np.asarray(_get(gs, 'a', r), dtype=np.float64)
    b = np.asarray(_get(gs, 'b', r), dtype=np.float64)
    c = np.asarray(_get(gs, 'c', r), dtype=np.float64)
    n1 = np.asarray(_get(gs, 'n1', _get(gs, 'n_shape', np.full(N, 2.0))), dtype=np.float64)
    n2 = np.asarray(_get(gs, 'n2', n1), dtype=np.float64)
    gtype = np.asarray(_get(gs, 'gtype', np.zeros(N, dtype=int)), dtype=np.int8)
    sid = _get(gs, 'species_id', None)
    sid = np.asarray(sid if sid is not None else np.where(gtype == 0, 0, 1), dtype=np.int16)
    f = _get(gs, 'f', None)
    f = np.asarray(f if f is not None else np.where(gtype == 0, 1.0, 0.0), dtype=np.float32)
    cell_gid = np.asarray(_get(gs, 'cell_granule_id', np.zeros(0, dtype=int)), dtype=np.int32)
    cell_offset = _get(gs, 'cell_offset', None)
    cell_offset = (np.asarray(cell_offset, dtype=np.int32) if cell_offset is not None
                   else np.zeros(N + 1, dtype=np.int32))
    static = {
        'mode': mode, 'N': N, 'n_cells': int(len(cell_gid)),
        'r': r, 'a': a, 'b': b, 'c': c, 'n1': n1, 'n2': n2,
        'species_id': sid, 'f': f, 'gtype': gtype,
        'cell_granule_id': cell_gid, 'cell_offset': cell_offset,
        'Lx': float(p.Lx), 'Ly': float(p.Ly), 'Lz': float(p.Lz),
        'periodic': getattr(p, 'boundary_mode', 'walls') == 'periodic',
    }
    msg = {
        'kind': 'start', 'run_dir': str(run_dir or getattr(p, 'output_dir', '')),
        'mode': mode, 'boundary_mode': getattr(p, 'boundary_mode', 'walls'),
        'Lx': float(p.Lx), 'Ly': float(p.Ly), 'Lz': float(p.Lz),
        'dt': float(p.dt), 't_total': float(p.t_total), 'n_steps': int(n_steps),
        'resume_step': int(resume_step), 'save_every': int(p.save_every),
        'seed': None if seed is None else int(seed),
        'species': species_table(p),
        'N': N, 'n_cells': int(len(cell_gid)),
        'r': r.astype(np.float32), 'a': a.astype(np.float32), 'b': b.astype(np.float32),
        'c': c.astype(np.float32), 'n1': n1.astype(np.float32), 'n2': n2.astype(np.float32),
        'species_id': sid, 'f': f, 'gtype': gtype,
        'cell_granule_id': cell_gid, 'cell_offset': cell_offset,
    }
    return msg, static


def frame_message(gs, p, step, t, static, metrics=None, send_cells=True, send_speed=False,
                  wall=None):
    """Per-step frame from a GranuleSystem or a snapshot dict."""
    import time
    mode = static['mode']
    is3d = (mode == '3D')
    x = np.asarray(_get(gs, 'x'), dtype=np.float64)
    y = np.asarray(_get(gs, 'y'), dtype=np.float64)
    msg = {'kind': 'frame', 'step': int(step), 't': float(t),
           'wall': float(time.time() if wall is None else wall),
           'x': x.astype(np.float32), 'y': y.astype(np.float32)}
    if is3d:
        z = np.asarray(_get(gs, 'z'), dtype=np.float64)
        quat = _get(gs, 'quat', None)
        if quat is None:
            quat = np.tile([1.0, 0.0, 0.0, 0.0], (len(x), 1))
        quat = np.asarray(quat, dtype=np.float64)
        msg['z'] = z.astype(np.float32)
        msg['quat'] = quat.astype(np.float32)
    else:
        theta = np.asarray(_get(gs, 'theta', np.zeros(len(x))), dtype=np.float64)
        msg['theta'] = theta.astype(np.float32)

    if send_speed:
        vx = np.asarray(_get(gs, 'vx', np.zeros(len(x))), dtype=np.float64)
        vy = np.asarray(_get(gs, 'vy', np.zeros(len(x))), dtype=np.float64)
        sp2 = vx**2 + vy**2
        if is3d:
            vz = np.asarray(_get(gs, 'vz', np.zeros(len(x))), dtype=np.float64)
            sp2 = sp2 + vz**2
        msg['speed'] = np.sqrt(sp2).astype(np.float32)

    # V3.1: cells divide, so the count can grow mid-run. Refresh the static
    # cell arrays into the frame when that happens; the renderer adopts them.
    live_gid = _get(gs, 'cell_granule_id', None)
    if live_gid is not None and len(live_gid) != static['n_cells']:
        gid_new = np.asarray(live_gid, dtype=np.int32)
        off_new = _get(gs, 'cell_offset', None)
        static['n_cells'] = int(len(gid_new))
        static['cell_granule_id'] = gid_new
        if off_new is not None:
            static['cell_offset'] = np.asarray(off_new, dtype=np.int32)
        msg['n_cells'] = static['n_cells']
        msg['cell_granule_id'] = gid_new
        msg['cell_offset'] = static['cell_offset']

    C = static['n_cells']
    if send_cells and C > 0:
        gid = static['cell_granule_id']
        state = np.asarray(_get(gs, 'cell_state', np.zeros(C, dtype=int)), dtype=np.int8)
        bt = np.asarray(_get(gs, 'cell_bridge_target', np.full(C, -1)), dtype=np.int64)
        locked = np.asarray(_get(gs, 'cell_bridge_locked', np.zeros(C, dtype=bool)), dtype=bool)
        if is3d:
            cw_x, cw_y, cw_z = cell_world_xyz(
                x, y, z, quat, static['a'], static['b'], static['c'], static['n1'], static['n2'],
                gid, _get(gs, 'cell_eta_local', np.zeros(C)), _get(gs, 'cell_omega_local', np.zeros(C)))
            cw = np.column_stack([cw_x, cw_y, cw_z])
            pos = np.column_stack([x, y, z])
            L = np.array([static['Lx'], static['Ly'], static['Lz']])
            msg['cell_z'] = cw_z.astype(np.float32)
        else:
            cw_x, cw_y = cell_world_xy(x, y, theta, static['a'], static['b'], static['n1'],
                                       gid, _get(gs, 'cell_theta_local', np.zeros(C)))
            cw = np.column_stack([cw_x, cw_y])
            pos = np.column_stack([x, y])
            L = np.array([static['Lx'], static['Ly']])
        if static['periodic']:
            cw = np.mod(cw, L)
        msg['cell_x'] = cw[:, 0].astype(np.float32)
        msg['cell_y'] = cw[:, 1].astype(np.float32)
        msg['cell_state'] = state
        msg['cell_theta'] = np.asarray(_get(gs, 'cell_theta_local', np.zeros(C)), dtype=np.float32)
        seg, sel = bridge_segments(cw, state, bt, pos, static['r'], L, static['periodic'])
        msg['bridge_seg'] = seg.astype(np.float32)
        msg['bridge_locked'] = locked[sel] if sel.size else np.zeros(0, dtype=bool)
        msg['bridge_target'] = bt[sel].astype(np.int32) if sel.size else np.zeros(0, dtype=np.int32)
        msg['bridge_cell'] = sel.astype(np.int32)
    if metrics:
        msg['metrics'] = dict(metrics)
    return msg


def start_from_snap(snap, p, run_dir='', n_steps=0):
    """Start message from a saved snapshot (the tail viewer's first message)."""
    return start_message(snap, p, seed=None, n_steps=n_steps, resume_step=0, run_dir=run_dir)


def frame_from_snap(snap, p, static, step=None, t=None, metrics=None, send_cells=True):
    """Frame message from a saved snapshot dict."""
    if t is None:
        t = float(snap.get('time', 0.0))
    if step is None:
        step = int(round(t / p.dt)) if p.dt > 0 else 0
    return frame_message(snap, p, step, t, static, metrics=metrics, send_cells=send_cells)


def run_dir_of(p):
    return os.path.abspath(getattr(p, 'output_dir', '.'))


__all__ = ['sgnpow', 'superellipse_points', 'superellipsoid_points', 'quat_rotate_many',
           'cell_world_xy', 'cell_world_xyz', 'bridge_segments',
           'start_message', 'frame_message', 'start_from_snap', 'frame_from_snap']
