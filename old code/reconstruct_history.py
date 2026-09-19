"""
Reconstruct history.json and history.csv for DOE runs that only have snapshots + fields.
Computes the key scalar metrics from raw .npz data without needing a full GranuleSystem.
"""
import numpy as np
import json
import csv
import os
import sys
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.ndimage import label


def kozeny_carman_permeability(porosity, d_grain):
    """K = eps^3 * d^2 / (180 * (1-eps)^2)"""
    eps = max(porosity, 1e-6)
    return eps**3 * d_grain**2 / (180.0 * max((1.0 - eps)**2, 1e-12))


def connectivity(field, threshold=0.3):
    """Number of clusters, largest fraction, coverage."""
    b = (field > threshold).astype(int)
    lab, nc = label(b)
    if nc == 0 or b.sum() == 0:
        return 0, 0.0, 0.0
    sz = np.array([np.sum(lab == l) for l in range(1, nc + 1)])
    return nc, float(sz.max() / b.sum()), float(b.sum() / b.size)


def reconstruct_metrics(snap_path, fields_path, params):
    """Compute history metrics from a single snapshot + fields pair."""
    snap = np.load(snap_path, allow_pickle=True)
    fields = np.load(fields_path, allow_pickle=True)

    t = float(snap['time'])
    phi_f = fields['phi_f']
    phi_i = fields['phi_i']
    phi_v = fields['phi_v']

    x = snap['x']
    y = snap['y']
    z = snap['z'] if 'z' in snap else None
    r = snap['r']
    gtype = snap['gtype']
    is_3d = z is not None and len(z) > 0

    N = len(x)
    func_mask = gtype == 0
    inert_mask = gtype == 1

    # Boundary exclusion
    bx = params.get('boundary_exclusion', 0.2)
    Lx = params.get('Lx', 800.0)
    Ly = params.get('Ly', 800.0)
    Lz = params.get('Lz', 800.0)
    Ngrid = params.get('Ngrid', 80)

    if bx > 0:
        shape = phi_f.shape
        lo = [int(bx * s) for s in shape]
        hi = [int((1.0 - bx) * s) for s in shape]
        if phi_f.ndim == 3:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        else:
            inner = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
        pf = phi_f[inner]
        pi = phi_i[inner]
        pv = phi_v[inner]

        x_lo, x_hi = bx * Lx, (1.0 - bx) * Lx
        y_lo, y_hi = bx * Ly, (1.0 - bx) * Ly
        inner_gran = (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)
        if is_3d:
            z_lo, z_hi = bx * Lz, (1.0 - bx) * Lz
            inner_gran &= (z >= z_lo) & (z <= z_hi)
    else:
        pf, pi, pv = phi_f, phi_i, phi_v
        inner_gran = np.ones(N, dtype=bool)

    m = dict(time=t)
    m['phi_f_mean'] = float(np.mean(pf))
    m['phi_i_mean'] = float(np.mean(pi))
    m['phi_v_mean'] = float(np.mean(pv))
    m['boundary_exclusion'] = bx
    m['n_granules_inner'] = int(np.sum(inner_gran))

    # Connectivity
    fn, fl, fc = connectivity(pf, 0.3)
    m.update(func_nc=fn, func_lf=fl, func_cov=fc)
    vn, vl, vc = connectivity(pv, 0.3)
    m.update(void_nc=vn, void_lf=vl, void_cov=vc)
    inn, il, _ = connectivity(pi, 0.3)
    m.update(inert_nc=inn, inert_lf=il)

    # Tissue fraction
    m['tissue_frac'] = float(np.mean(pf > 0.5))

    # Max cluster area
    thr = np.mean(pf) + 0.3 * np.std(pf)
    b = (pf > thr).astype(int)
    lab, nc = label(b)
    dxg = Lx / Ngrid
    if nc > 0:
        sizes = np.array([np.sum(lab == l) for l in range(1, nc + 1)])
        m['func_max_area'] = float(np.max(sizes) * dxg**2)
    else:
        m['func_max_area'] = 0.0

    # Packing in functional-rich region
    fr = pf > thr
    pt = pf + pi
    m['packing_func_rich'] = float(np.mean(pt[fr])) if np.any(fr) else 0.0

    # Force metrics
    fx = snap['force_x']
    fy = snap['force_y']
    fz = snap.get('force_z', np.zeros_like(fx))
    if is_3d:
        F_mag = np.sqrt(fx**2 + fy**2 + fz**2)
    else:
        F_mag = np.sqrt(fx**2 + fy**2)
    m['F_mean'] = float(np.mean(F_mag))
    m['F_max'] = float(np.max(F_mag))
    m['F_func_mean'] = float(np.mean(F_mag[func_mask])) if np.any(func_mask) else 0.0

    # Contact metrics from stored contact arrays
    if 'contact_i' in snap and len(snap['contact_i']) > 0:
        ci = snap['contact_i']
        cj = snap['contact_j']
        overlap = snap['contact_overlap']
        n_contacts = len(ci)
        n_ff = int(np.sum((gtype[ci] == 0) & (gtype[cj] == 0)))
        n_ii = int(np.sum((gtype[ci] == 1) & (gtype[cj] == 1)))
        n_if = n_contacts - n_ff - n_ii
        max_overlap_ratio = float(np.max(overlap / np.minimum(r[ci], r[cj]))) if n_contacts > 0 else 0.0
    else:
        n_contacts = 0
        n_ff = n_ii = n_if = 0
        max_overlap_ratio = 0.0

    m['n_contacts'] = n_contacts
    m['n_contacts_ff'] = n_ff
    m['n_contacts_if'] = n_if
    m['n_contacts_ii'] = n_ii
    m['max_overlap_ratio'] = max_overlap_ratio

    # Cell state metrics
    m['n_attached_total'] = float(np.sum(snap['n_attached'][func_mask])) if 'n_attached' in snap else 0.0
    m['n_seeded_total'] = float(np.sum(snap['n_cells'][func_mask])) if 'n_cells' in snap else 0.0
    m['mean_spread_frac'] = float(np.mean(snap['spread_fraction'][func_mask])) if 'spread_fraction' in snap and np.any(func_mask) else 0.0
    m['mean_fa_maturity'] = float(np.mean(snap['fa_maturity'][func_mask])) if 'fa_maturity' in snap and np.any(func_mask) else 0.0
    m['n_overcrowded_total'] = float(np.sum(snap['n_overcrowded'][func_mask])) if 'n_overcrowded' in snap else 0.0

    # Transport metrics
    porosity = float(np.mean(pv))
    m['porosity'] = porosity
    inner_r = r[inner_gran] if np.any(inner_gran) else r
    d_grain = float(np.mean(2 * inner_r))
    m['d_grain_mean'] = d_grain
    m['K_kozeny_carman'] = float(kozeny_carman_permeability(porosity, d_grain))

    # Compaction ratio
    phi_solid = float(np.mean(pf) + np.mean(pi))
    m['phi_solid'] = phi_solid
    ar_mean = 1.0
    if 'a' in snap and 'b' in snap:
        a_arr = snap['a']
        b_arr = snap['b']
        if len(a_arr) > 0:
            ar_mean = float(np.mean(np.maximum(a_arr, b_arr) / np.minimum(a_arr, b_arr)))

    # RCP fraction approximation (Donev et al. 2004)
    phi_rcp = 0.64 + 0.0038 * (ar_mean - 1.0)
    m['phi_RCP'] = phi_rcp
    m['compaction_ratio'] = phi_solid / phi_rcp if phi_rcp > 0 else 0.0

    # Darcy number
    if is_3d:
        L_char = (Lx * Ly * Lz) ** (1.0 / 3.0)
    else:
        L_char = np.sqrt(Lx * Ly)
    m['Da_number'] = m['K_kozeny_carman'] / (L_char**2) if L_char > 0 else 0.0

    # Bridge count from cell data
    if 'cell_bridge_target' in snap:
        bt = snap['cell_bridge_target']
        m['n_bridges'] = int(np.sum(bt >= 0))
    else:
        m['n_bridges'] = 0

    return m


def reconstruct_run(run_dir):
    """Reconstruct history.json and history.csv for a single run."""
    run_dir = Path(run_dir)
    snap_dir = run_dir / 'snapshots'
    field_dir = run_dir / 'fields'

    if not snap_dir.exists() or not field_dir.exists():
        print(f"  SKIP {run_dir.name}: missing snapshots/ or fields/")
        return False

    # Already has history?
    if (run_dir / 'history.json').exists():
        print(f"  SKIP {run_dir.name}: history.json already exists")
        return True

    # Load params
    params_path = run_dir / 'params.json'
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
    else:
        params = {}

    # Find matching snap/field pairs
    snap_files = sorted(snap_dir.glob('snap_*.npz'))
    field_files = sorted(field_dir.glob('fields_*.npz'))

    # Match by index
    snap_indices = {f.stem.split('_')[1]: f for f in snap_files}
    field_indices = {f.stem.split('_')[1]: f for f in field_files}
    common = sorted(set(snap_indices.keys()) & set(field_indices.keys()))

    if not common:
        print(f"  SKIP {run_dir.name}: no matching snap/field pairs")
        return False

    history = []
    for idx in common:
        try:
            m = reconstruct_metrics(snap_indices[idx], field_indices[idx], params)
            history.append(m)
        except Exception as e:
            print(f"    WARNING: {run_dir.name}/snap_{idx}: {e}")
            continue

    if not history:
        print(f"  SKIP {run_dir.name}: no metrics computed")
        return False

    # Sort by time
    history.sort(key=lambda h: h['time'])

    # Save history.json
    with open(run_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Save history.csv
    keys = list(history[0].keys())
    with open(run_dir / 'history.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(history)

    print(f"  OK {run_dir.name}: {len(history)} timepoints, t=[{history[0]['time']:.1f}, {history[-1]['time']:.1f}]h")
    return True


def main():
    base = Path('results/trials/trials')
    if not base.exists():
        print(f"ERROR: {base} not found")
        sys.exit(1)

    doe_dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith('DOE_'))
    print(f"Found {len(doe_dirs)} DOE directories")

    success = 0
    for d in doe_dirs:
        if reconstruct_run(d):
            success += 1

    print(f"\nDone: {success}/{len(doe_dirs)} runs have history data")


if __name__ == '__main__':
    main()
