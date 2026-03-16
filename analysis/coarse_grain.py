"""
Continuum Coarse-Graining for GELLS-DEM
========================================
Extracts continuum-level quantities (stress tensor, strain rate tensor,
effective viscosity, coordination number) from DEM simulation snapshots
via coarse-graining.

Theory
------
Stress: Love-Weber formula  sigma_ab = (1/V) sum_c f_a^c l_b^c
Strain rate: velocity gradient  eps_dot_ab = (1/2V) sum_i [v_a^i x_b^i + v_b^i x_a^i] V_i
Viscosity: eta_eff = |sigma_dev| / (2 |eps_dot_dev|)

Units
-----
Length: um, Force: nN, Time: h, Stress: nN/um^2 = kPa,
Strain rate: 1/h, Viscosity: kPa*h

Usage
-----
    python analysis/coarse_grain.py -i results/default
    python analysis/coarse_grain.py -i results/default -o plots/

Or import programmatically::

    from analysis.coarse_grain import compute_stress_tensor, extract_continuum_timeseries
    sigma = compute_stress_tensor(snap, p)
    ts = extract_continuum_timeseries('results/default')
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ======================================================================
# Helpers
# ======================================================================

def _is_3d(snap, p):
    """Determine whether a snapshot is from a 3D simulation."""
    mode = getattr(p, 'mode', '2D')
    if mode == '3D':
        return True
    if 'z' in snap and np.any(snap['z'] != 0):
        return True
    return False


def _dim(snap, p):
    """Return spatial dimensionality (2 or 3)."""
    return 3 if _is_3d(snap, p) else 2


def _domain_volume(p, is_3d):
    """Total domain volume (um^3) or area (um^2)."""
    if is_3d:
        return p.Lx * p.Ly * p.Lz
    return p.Lx * p.Ly


def _inner_mask(snap, p):
    """Boolean mask for granules inside the boundary-exclusion region."""
    be = getattr(p, 'boundary_exclusion', 0.0)
    N = len(snap['x'])
    if be <= 0:
        return np.ones(N, dtype=bool)

    xlo = be * p.Lx
    xhi = (1 - be) * p.Lx
    ylo = be * p.Ly
    yhi = (1 - be) * p.Ly
    mask = (snap['x'] >= xlo) & (snap['x'] <= xhi) & \
           (snap['y'] >= ylo) & (snap['y'] <= yhi)

    if _is_3d(snap, p):
        zlo = be * p.Lz
        zhi = (1 - be) * p.Lz
        mask &= (snap['z'] >= zlo) & (snap['z'] <= zhi)

    return mask


def _inner_volume(p, is_3d):
    """Volume (or area) of the inner region after boundary exclusion."""
    be = getattr(p, 'boundary_exclusion', 0.0)
    frac = (1.0 - 2.0 * be)
    if is_3d:
        return p.Lx * p.Ly * p.Lz * frac**3
    return p.Lx * p.Ly * frac**2


def _granule_volume(snap, i, is_3d):
    """Volume (3D) or area (2D) of granule i."""
    r = snap['r'][i]
    if is_3d:
        return (4.0 / 3.0) * np.pi * r**3
    return np.pi * r**2


def _granule_volumes(snap, is_3d):
    """Volume (3D) or area (2D) for all granules, vectorized."""
    r = snap['r']
    if is_3d:
        return (4.0 / 3.0) * np.pi * r**3
    return np.pi * r**2


def _positions(snap, is_3d):
    """Return (N, D) position array."""
    if is_3d:
        return np.column_stack([snap['x'], snap['y'], snap['z']])
    return np.column_stack([snap['x'], snap['y']])


def _velocities(snap, is_3d):
    """Return (N, D) velocity array in um/h."""
    vx = snap.get('vx', np.zeros(len(snap['x'])))
    vy = snap.get('vy', np.zeros(len(snap['x'])))
    if is_3d:
        vz = snap.get('vz', np.zeros(len(snap['x'])))
        return np.column_stack([vx, vy, vz])
    return np.column_stack([vx, vy])


def _forces(snap, is_3d):
    """Return (N, D) force array in nN."""
    fx = snap.get('force_x', np.zeros(len(snap['x'])))
    fy = snap.get('force_y', np.zeros(len(snap['x'])))
    if is_3d:
        fz = snap.get('force_z', np.zeros(len(snap['x'])))
        return np.column_stack([fx, fy, fz])
    return np.column_stack([fx, fy])


def _tensor_to_3x3(t, D):
    """Embed a DxD tensor into a 3x3 matrix (zero-padded for 2D)."""
    out = np.zeros((3, 3))
    out[:D, :D] = t
    return out


def _von_mises(sigma):
    """Von Mises equivalent stress from 3x3 stress tensor.

    sigma_vm = sqrt(3/2 * s_ij s_ij) where s = sigma - (tr(sigma)/3)*I
    """
    p = np.trace(sigma) / 3.0
    s = sigma - p * np.eye(3)
    return np.sqrt(1.5 * np.sum(s * s))


def _dev_magnitude(t):
    """Magnitude of deviatoric part: sqrt(t_dev : t_dev / 2)."""
    p = np.trace(t) / 3.0
    dev = t - p * np.eye(3)
    return np.sqrt(0.5 * np.sum(dev * dev))


# ======================================================================
# 1. Stress tensor (Love-Weber formula)
# ======================================================================

def _reconstruct_contacts(snap, p):
    """Reconstruct contact list from granule positions when per-contact data
    is not available in the snapshot.

    Returns list of dicts with keys: i, j, cx, cy, cz, nx, ny, nz,
    overlap, R_eff, F_normal, A_contact, gtype_i, gtype_j.
    """
    is_3d = _is_3d(snap, p)
    pos = _positions(snap, is_3d)
    r = snap['r']
    N = len(r)
    gtype = snap['gtype']

    nu2 = p.poisson_ratio ** 2
    E_star_gg = (p.E_modulus * 1e3) / (2.0 * (1.0 - nu2))

    max_r = float(np.max(r))
    cutoff = 2.0 * max_r
    tree = cKDTree(pos)
    pairs = tree.query_pairs(cutoff, output_type='ndarray')

    contacts = []
    for idx in range(len(pairs)):
        i, j = pairs[idx]
        dp = pos[j] - pos[i]
        d = np.linalg.norm(dp)
        if d < 1e-6:
            continue
        overlap = r[i] + r[j] - d
        if overlap <= 0:
            continue

        n_vec = dp / d
        R_eff = r[i] * r[j] / (r[i] + r[j])

        # Hertzian force
        from new_dem_0 import hertz_contact_force
        Fc = hertz_contact_force(E_star_gg, R_eff, overlap)

        # DMT adhesion
        gi, gj = int(gtype[i]), int(gtype[j])
        if gi == 0 and gj == 0:
            W_adh = p.W_adh_ff
        elif gi == 1 and gj == 1:
            W_adh = p.W_adh_ii
        else:
            W_adh = p.W_adh_if
        F_adh = 2.0 * np.pi * W_adh * R_eff * 1e3
        F_normal = Fc - F_adh

        A_contact = np.pi * R_eff * overlap

        # Contact point
        cx = pos[i, 0] + (r[i] - overlap / 2) * n_vec[0]
        cy = pos[i, 1] + (r[i] - overlap / 2) * n_vec[1]
        cz = 0.0
        nz = 0.0
        if is_3d:
            cz = pos[i, 2] + (r[i] - overlap / 2) * n_vec[2]
            nz = n_vec[2]

        contacts.append({
            'i': i, 'j': j,
            'cx': cx, 'cy': cy, 'cz': cz,
            'nx': n_vec[0], 'ny': n_vec[1], 'nz': nz,
            'overlap': overlap, 'R_eff': R_eff,
            'F_normal': F_normal, 'A_contact': A_contact,
            'gtype_i': gi, 'gtype_j': gj,
        })

    return contacts


def _get_contacts(snap, p):
    """Extract per-contact data from snapshot, or reconstruct if absent.

    Returns arrays: ci, cj, cx, cy, cz, nx, ny, nz, F_normal, gtype_i, gtype_j
    """
    if 'contact_i' in snap and len(snap['contact_i']) > 0:
        ci = np.asarray(snap['contact_i'])
        cj = np.asarray(snap['contact_j'])
        cx = np.asarray(snap['contact_cx'])
        cy = np.asarray(snap['contact_cy'])
        cz = np.asarray(snap.get('contact_cz', np.zeros(len(ci))))
        nx = np.asarray(snap['contact_nx'])
        ny = np.asarray(snap['contact_ny'])
        nz = np.asarray(snap.get('contact_nz', np.zeros(len(ci))))
        F_normal = np.asarray(snap['contact_F_normal'])
        gtype = snap['gtype']
        gtype_i = gtype[ci].astype(int)
        gtype_j = gtype[cj].astype(int)
        return ci, cj, cx, cy, cz, nx, ny, nz, F_normal, gtype_i, gtype_j

    # Fallback: reconstruct contacts
    contacts = _reconstruct_contacts(snap, p)
    if not contacts:
        n = 0
        empty = np.array([], dtype=np.float64)
        empty_int = np.array([], dtype=np.int32)
        return (empty_int, empty_int, empty, empty, empty,
                empty, empty, empty, empty, empty_int, empty_int)

    ci = np.array([c['i'] for c in contacts], dtype=np.int32)
    cj = np.array([c['j'] for c in contacts], dtype=np.int32)
    cx = np.array([c['cx'] for c in contacts])
    cy = np.array([c['cy'] for c in contacts])
    cz = np.array([c['cz'] for c in contacts])
    nx = np.array([c['nx'] for c in contacts])
    ny = np.array([c['ny'] for c in contacts])
    nz = np.array([c['nz'] for c in contacts])
    F_normal = np.array([c['F_normal'] for c in contacts])
    gtype_i = np.array([c['gtype_i'] for c in contacts], dtype=np.int32)
    gtype_j = np.array([c['gtype_j'] for c in contacts], dtype=np.int32)
    return ci, cj, cx, cy, cz, nx, ny, nz, F_normal, gtype_i, gtype_j


def compute_stress_tensor(snap, p):
    """Compute the Cauchy stress tensor from contact forces (Love-Weber).

    The stress tensor is decomposed into contributions from Hertzian contact
    forces and cell bridging (active) forces.

    Parameters
    ----------
    snap : dict
        Snapshot dict with granule positions, forces, and (optionally)
        per-contact arrays.
    p : Params
        Simulation parameters.

    Returns
    -------
    dict
        sigma_total : (3, 3) ndarray  -- total stress tensor (kPa)
        sigma_contact : (3, 3) ndarray -- contact contribution (kPa)
        sigma_active : (3, 3) ndarray  -- cell bridging contribution (kPa)
        pressure : float  -- mean normal stress = -tr(sigma)/3 (kPa)
        deviatoric_stress : (3, 3) ndarray -- sigma - pressure*I
        von_mises : float  -- von Mises equivalent stress (kPa)
    """
    is_3d = _is_3d(snap, p)
    D = 3 if is_3d else 2
    V = _inner_volume(p, is_3d)

    pos = _positions(snap, is_3d)
    mask = _inner_mask(snap, p)

    # --- Contact stress (Love-Weber) ---
    ci, cj, cx, cy, cz, nx, ny, nz, F_normal, gtype_i, gtype_j = \
        _get_contacts(snap, p)

    sigma_contact = np.zeros((D, D))
    sigma_active = np.zeros((D, D))

    # Filter contacts: at least one granule in inner region
    if len(ci) > 0:
        contact_inner = mask[ci] | mask[cj]

        for k in range(len(ci)):
            if not contact_inner[k]:
                continue

            i, j = ci[k], cj[k]

            # Force vector (contact normal * F_normal)
            f = np.zeros(D)
            if D == 3:
                f[0] = F_normal[k] * nx[k]
                f[1] = F_normal[k] * ny[k]
                f[2] = F_normal[k] * nz[k]
            else:
                f[0] = F_normal[k] * nx[k]
                f[1] = F_normal[k] * ny[k]

            # Branch vector: center-to-center (i -> j)
            l = pos[j, :D] - pos[i, :D]

            # sigma_ab += f_a * l_b / V
            sigma_contact += np.outer(f, l)

    if V > 0:
        sigma_contact /= V

    # --- Active stress from cell bridges ---
    # Reconstruct bridging forces from per-cell data if available
    if 'cell_bridge_target' in snap and 'cell_fx' in snap:
        cell_bt = snap['cell_bridge_target']
        cell_fx = snap['cell_fx']
        cell_fy = snap['cell_fy']
        cell_fz = snap.get('cell_fz', np.zeros(len(cell_fx)))
        cell_gid = snap['cell_granule_id']
        cell_offset = snap.get('cell_offset', None)

        # For each bridging cell, the force vector is cell_f and the branch
        # vector goes from the host granule to the bridge target granule
        bridging = cell_bt >= 0
        bridge_indices = np.where(bridging)[0]

        for ci_cell in bridge_indices:
            gi = int(cell_gid[ci_cell])
            gj = int(cell_bt[ci_cell])
            if gi >= len(snap['x']) or gj >= len(snap['x']):
                continue
            if not (mask[gi] or mask[gj]):
                continue

            f = np.zeros(D)
            f[0] = cell_fx[ci_cell]
            f[1] = cell_fy[ci_cell]
            if D == 3:
                f[2] = cell_fz[ci_cell]

            l = pos[gj, :D] - pos[gi, :D]
            sigma_active += np.outer(f, l)

        if V > 0:
            sigma_active /= V

    # Embed into 3x3
    sigma_contact_3 = _tensor_to_3x3(sigma_contact, D)
    sigma_active_3 = _tensor_to_3x3(sigma_active, D)
    sigma_total = sigma_contact_3 + sigma_active_3

    pressure = -np.trace(sigma_total) / 3.0
    dev = sigma_total - (-pressure) * np.eye(3)  # dev = sigma - (tr/3)*I
    # Correct: deviatoric = sigma - (tr(sigma)/3)*I
    dev = sigma_total - (np.trace(sigma_total) / 3.0) * np.eye(3)
    vm = _von_mises(sigma_total)

    return {
        'sigma_total': sigma_total,
        'sigma_contact': sigma_contact_3,
        'sigma_active': sigma_active_3,
        'pressure': pressure,
        'deviatoric_stress': dev,
        'von_mises': vm,
    }


# ======================================================================
# 2. Strain rate tensor
# ======================================================================

def compute_strain_rate(snap, p):
    """Compute the strain rate tensor from granule velocities.

    Uses the volume-averaged velocity gradient:
        eps_dot_ab = (1/2V) sum_i [v_a^i x_b^i + v_b^i x_a^i] V_i

    where V_i is the volume of granule i (sphere approximation).

    Parameters
    ----------
    snap : dict
        Snapshot dict with granule positions and velocities.
    p : Params
        Simulation parameters.

    Returns
    -------
    dict
        eps_dot : (3, 3) ndarray  -- strain rate tensor (1/h)
        volumetric_strain_rate : float  -- tr(eps_dot) (compaction rate, 1/h)
        shear_strain_rate : float  -- sqrt(eps_dot_dev : eps_dot_dev / 2) (1/h)
    """
    is_3d = _is_3d(snap, p)
    D = 3 if is_3d else 2
    V_domain = _inner_volume(p, is_3d)

    pos = _positions(snap, is_3d)
    vel = _velocities(snap, is_3d)
    mask = _inner_mask(snap, p)
    vols = _granule_volumes(snap, is_3d)

    # Centroid of inner region
    inner_idx = np.where(mask)[0]
    if len(inner_idx) == 0:
        eps_3 = np.zeros((3, 3))
        return {
            'eps_dot': eps_3,
            'volumetric_strain_rate': 0.0,
            'shear_strain_rate': 0.0,
        }

    # Mean position and velocity for reference (remove rigid body motion)
    total_vol = np.sum(vols[mask])
    if total_vol < 1e-30:
        eps_3 = np.zeros((3, 3))
        return {
            'eps_dot': eps_3,
            'volumetric_strain_rate': 0.0,
            'shear_strain_rate': 0.0,
        }

    x_mean = np.sum(pos[mask, :D] * vols[mask, None], axis=0) / total_vol
    v_mean = np.sum(vel[mask, :D] * vols[mask, None], axis=0) / total_vol

    # Fluctuation-based strain rate
    eps_dot = np.zeros((D, D))
    for idx in inner_idx:
        dx = pos[idx, :D] - x_mean
        dv = vel[idx, :D] - v_mean
        Vi = vols[idx]
        # Symmetric part: (v_a x_b + v_b x_a) / 2
        eps_dot += Vi * (np.outer(dv, dx) + np.outer(dx, dv)) / 2.0

    if V_domain > 0:
        eps_dot /= V_domain

    eps_3 = _tensor_to_3x3(eps_dot, D)
    vol_rate = np.trace(eps_3)
    shear_rate = _dev_magnitude(eps_3)

    return {
        'eps_dot': eps_3,
        'volumetric_strain_rate': vol_rate,
        'shear_strain_rate': shear_rate,
    }


# ======================================================================
# 3. Effective viscosity
# ======================================================================

def compute_effective_viscosity(snap, p):
    """Compute effective viscosity from stress and strain rate.

    eta_eff = |sigma_dev| / (2 * |eps_dot_dev|)

    where |sigma_dev| is the von Mises stress and |eps_dot_dev| is
    the deviatoric strain rate magnitude.

    Parameters
    ----------
    snap : dict
        Snapshot dict.
    p : Params
        Simulation parameters.

    Returns
    -------
    float
        Effective viscosity in kPa*h. Returns np.inf if strain rate is
        negligible (quasi-static).
    """
    stress = compute_stress_tensor(snap, p)
    strain = compute_strain_rate(snap, p)

    vm = stress['von_mises']
    shear_rate = strain['shear_strain_rate']

    if shear_rate < 1e-30:
        return np.inf

    return vm / (2.0 * shear_rate)


# ======================================================================
# 4. Coordination number
# ======================================================================

def compute_coordination(snap, p):
    """Compute coordination number and its decomposition by granule type.

    Parameters
    ----------
    snap : dict
        Snapshot dict.
    p : Params
        Simulation parameters.

    Returns
    -------
    dict
        Z : float  -- mean coordination number (all granules)
        Z_ff : float  -- functional-functional coordination
        Z_fi : float  -- functional-inert coordination
        Z_ii : float  -- inert-inert coordination
        n_contacts : int  -- total number of contacts
        n_contacts_ff : int
        n_contacts_fi : int
        n_contacts_ii : int
    """
    ci, cj, cx, cy, cz, nx, ny, nz, F_normal, gtype_i, gtype_j = \
        _get_contacts(snap, p)

    mask = _inner_mask(snap, p)
    gtype = snap['gtype']
    N = len(gtype)
    N_func = int(np.sum(gtype == 0))
    N_inert = int(np.sum(gtype == 1))

    n_contacts = 0
    n_ff = 0
    n_fi = 0
    n_ii = 0

    # Per-granule contact count
    contacts_per_granule = np.zeros(N, dtype=int)

    for k in range(len(ci)):
        i, j = ci[k], cj[k]
        if not (mask[i] or mask[j]):
            continue
        n_contacts += 1
        contacts_per_granule[i] += 1
        contacts_per_granule[j] += 1

        gi, gj = int(gtype_i[k]), int(gtype_j[k])
        if gi == 0 and gj == 0:
            n_ff += 1
        elif gi == 1 and gj == 1:
            n_ii += 1
        else:
            n_fi += 1

    # Inner granule count
    N_inner = int(np.sum(mask))
    Z = 2.0 * n_contacts / max(1, N_inner)

    N_func_inner = int(np.sum(mask & (gtype == 0)))
    N_inert_inner = int(np.sum(mask & (gtype == 1)))

    Z_ff = 2.0 * n_ff / max(1, N_func_inner) if N_func_inner > 0 else 0.0
    Z_fi = n_fi / max(1, N_func_inner + N_inert_inner)  # each fi contact touches one of each
    Z_ii = 2.0 * n_ii / max(1, N_inert_inner) if N_inert_inner > 0 else 0.0

    return {
        'Z': Z,
        'Z_ff': Z_ff,
        'Z_fi': Z_fi,
        'Z_ii': Z_ii,
        'n_contacts': n_contacts,
        'n_contacts_ff': n_ff,
        'n_contacts_fi': n_fi,
        'n_contacts_ii': n_ii,
    }


# ======================================================================
# 5. Time series extraction
# ======================================================================

def extract_continuum_timeseries(run_dir):
    """Extract continuum quantities at every snapshot.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output directory or .tar.gz archive.

    Returns
    -------
    dict of arrays
        Keys: time, pressure, von_mises, volumetric_strain_rate,
        shear_strain_rate, eta_eff, Z, Z_ff, Z_fi, Z_ii,
        sigma_contact_pressure, sigma_active_pressure,
        n_contacts, phi_solid, porosity.
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)

    n = len(snaps)
    ts = {
        'time': np.zeros(n),
        'pressure': np.zeros(n),
        'von_mises': np.zeros(n),
        'volumetric_strain_rate': np.zeros(n),
        'shear_strain_rate': np.zeros(n),
        'eta_eff': np.zeros(n),
        'Z': np.zeros(n),
        'Z_ff': np.zeros(n),
        'Z_fi': np.zeros(n),
        'Z_ii': np.zeros(n),
        'sigma_contact_pressure': np.zeros(n),
        'sigma_active_pressure': np.zeros(n),
        'n_contacts': np.zeros(n, dtype=int),
    }

    # Scalar history data for twin-axis plots
    if hist:
        ts['phi_solid'] = np.array([h.get('phi_solid', 0) for h in hist[:n]])
        ts['porosity'] = np.array([h.get('porosity', 0) for h in hist[:n]])
    else:
        ts['phi_solid'] = np.zeros(n)
        ts['porosity'] = np.zeros(n)

    for k, snap in enumerate(snaps):
        t = float(snap.get('time', 0))
        ts['time'][k] = t

        stress = compute_stress_tensor(snap, p)
        strain = compute_strain_rate(snap, p)
        coord = compute_coordination(snap, p)

        ts['pressure'][k] = stress['pressure']
        ts['von_mises'][k] = stress['von_mises']
        ts['volumetric_strain_rate'][k] = strain['volumetric_strain_rate']
        ts['shear_strain_rate'][k] = strain['shear_strain_rate']

        shear = strain['shear_strain_rate']
        if shear > 1e-30:
            ts['eta_eff'][k] = stress['von_mises'] / (2.0 * shear)
        else:
            ts['eta_eff'][k] = np.inf

        ts['Z'][k] = coord['Z']
        ts['Z_ff'][k] = coord['Z_ff']
        ts['Z_fi'][k] = coord['Z_fi']
        ts['Z_ii'][k] = coord['Z_ii']
        ts['n_contacts'][k] = coord['n_contacts']

        # Decomposed pressures
        ts['sigma_contact_pressure'][k] = -np.trace(stress['sigma_contact']) / 3.0
        ts['sigma_active_pressure'][k] = -np.trace(stress['sigma_active']) / 3.0

    return ts


# ======================================================================
# 6. Spatial coarse-graining
# ======================================================================

def coarse_grain_field(snap, p, quantity='stress', Ngrid=20):
    """Compute spatially resolved continuum fields via Gaussian coarse-graining.

    Divides the domain into Ngrid^D grid cells and computes the stress
    or strain-rate tensor in each cell using a Gaussian weighting function.

    Parameters
    ----------
    snap : dict
        Snapshot dict.
    p : Params
        Simulation parameters.
    quantity : str
        'stress' or 'strain_rate'.
    Ngrid : int
        Number of grid cells per dimension.

    Returns
    -------
    dict
        grid_x, grid_y, grid_z : 1D coordinate arrays
        pressure : ND array of pressure field (kPa)
        von_mises : ND array of von Mises stress field (kPa)
        volumetric_strain_rate : ND array (1/h), only if quantity='strain_rate'
        shear_strain_rate : ND array (1/h), only if quantity='strain_rate'
    """
    is_3d = _is_3d(snap, p)
    D = 3 if is_3d else 2
    pos = _positions(snap, is_3d)
    N = len(snap['x'])
    r = snap['r']

    # Grid
    gx = np.linspace(0, p.Lx, Ngrid + 1)
    gy = np.linspace(0, p.Ly, Ngrid + 1)
    gx_c = 0.5 * (gx[:-1] + gx[1:])
    gy_c = 0.5 * (gy[:-1] + gy[1:])

    if is_3d:
        gz = np.linspace(0, p.Lz, Ngrid + 1)
        gz_c = 0.5 * (gz[:-1] + gz[1:])
    else:
        gz_c = np.array([0.0])

    # Gaussian width ~ mean granule radius
    w = float(np.mean(r)) * 1.5

    # Cell volume
    dx = p.Lx / Ngrid
    dy = p.Ly / Ngrid
    if is_3d:
        dz = p.Lz / Ngrid
        cell_vol = dx * dy * dz
    else:
        cell_vol = dx * dy

    shape = (Ngrid, Ngrid, Ngrid) if is_3d else (Ngrid, Ngrid)
    pressure_field = np.zeros(shape)
    vm_field = np.zeros(shape)
    vol_sr_field = np.zeros(shape) if quantity == 'strain_rate' else None
    shear_sr_field = np.zeros(shape) if quantity == 'strain_rate' else None

    if quantity == 'stress':
        ci_arr, cj_arr, cx_arr, cy_arr, cz_arr, nx_arr, ny_arr, nz_arr, \
            Fn_arr, gti, gtj = _get_contacts(snap, p)

    vel = _velocities(snap, is_3d) if quantity == 'strain_rate' else None
    vols = _granule_volumes(snap, is_3d) if quantity == 'strain_rate' else None

    def _gauss_weight(x_cell, x_part, w):
        """Gaussian weight function."""
        d2 = np.sum((x_cell - x_part)**2)
        return np.exp(-d2 / (2.0 * w**2))

    # Iterate over grid cells
    for ix in range(Ngrid):
        for iy in range(Ngrid):
            nz_range = range(Ngrid) if is_3d else range(1)
            for iz in nz_range:
                x_cell = np.array([gx_c[ix], gy_c[iy]])
                if is_3d:
                    x_cell = np.array([gx_c[ix], gy_c[iy], gz_c[iz]])

                grid_idx = (ix, iy, iz) if is_3d else (ix, iy)

                if quantity == 'stress':
                    sigma_local = np.zeros((D, D))
                    w_total = 0.0

                    for k in range(len(ci_arr)):
                        c_pt = np.array([cx_arr[k], cy_arr[k]])
                        if is_3d:
                            c_pt = np.array([cx_arr[k], cy_arr[k], cz_arr[k]])

                        wt = _gauss_weight(x_cell, c_pt, w)
                        if wt < 1e-10:
                            continue

                        f = np.zeros(D)
                        f[0] = Fn_arr[k] * nx_arr[k]
                        f[1] = Fn_arr[k] * ny_arr[k]
                        if D == 3:
                            f[2] = Fn_arr[k] * nz_arr[k]

                        i_g, j_g = ci_arr[k], cj_arr[k]
                        l = pos[j_g, :D] - pos[i_g, :D]

                        sigma_local += wt * np.outer(f, l)
                        w_total += wt

                    if w_total > 1e-30:
                        sigma_local /= (cell_vol * w_total / max(1, len(ci_arr)))

                    s3 = _tensor_to_3x3(sigma_local, D)
                    pressure_field[grid_idx] = -np.trace(s3) / 3.0
                    vm_field[grid_idx] = _von_mises(s3)

                elif quantity == 'strain_rate':
                    eps_local = np.zeros((D, D))
                    w_total = 0.0

                    # Volume-weighted mean velocity in this cell
                    v_mean_local = np.zeros(D)
                    vol_sum = 0.0
                    for i_g in range(N):
                        wt = _gauss_weight(x_cell, pos[i_g, :D], w)
                        if wt < 1e-10:
                            continue
                        v_mean_local += wt * vols[i_g] * vel[i_g, :D]
                        vol_sum += wt * vols[i_g]
                    if vol_sum > 1e-30:
                        v_mean_local /= vol_sum

                    for i_g in range(N):
                        wt = _gauss_weight(x_cell, pos[i_g, :D], w)
                        if wt < 1e-10:
                            continue
                        dx_g = pos[i_g, :D] - x_cell
                        dv_g = vel[i_g, :D] - v_mean_local
                        Vi = vols[i_g]
                        eps_local += wt * Vi * (np.outer(dv_g, dx_g) +
                                                np.outer(dx_g, dv_g)) / 2.0
                        w_total += wt

                    if w_total > 1e-30 and cell_vol > 0:
                        eps_local /= (cell_vol * w_total / max(1, N))

                    e3 = _tensor_to_3x3(eps_local, D)
                    pressure_field[grid_idx] = -np.trace(e3) / 3.0
                    vm_field[grid_idx] = _von_mises(e3)
                    if vol_sr_field is not None:
                        vol_sr_field[grid_idx] = np.trace(e3)
                    if shear_sr_field is not None:
                        shear_sr_field[grid_idx] = _dev_magnitude(e3)

    result = {
        'grid_x': gx_c,
        'grid_y': gy_c,
        'pressure': pressure_field,
        'von_mises': vm_field,
    }
    if is_3d:
        result['grid_z'] = gz_c
    if vol_sr_field is not None:
        result['volumetric_strain_rate'] = vol_sr_field
    if shear_sr_field is not None:
        result['shear_strain_rate'] = shear_sr_field

    return result


# ======================================================================
# 7. Plotting functions
# ======================================================================

def plot_stress_timeseries(run_dir, outdir=None):
    """Plot pressure and von Mises stress over time, with contact vs active
    stress decomposition.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output.
    outdir : str or None
        Output directory for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ts = extract_continuum_timeseries(run_dir)
    t = ts['time']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Panel 1: Pressure
    ax = axes[0]
    ax.plot(t, ts['pressure'], 'k-', lw=2, label='Total pressure')
    ax.plot(t, ts['sigma_contact_pressure'], 'C0--', lw=1.5,
            label='Contact pressure')
    ax.plot(t, ts['sigma_active_pressure'], 'C1--', lw=1.5,
            label='Active (cell) pressure')
    ax.set_ylabel('Pressure (kPa)')
    ax.set_title('Stress Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Panel 2: Von Mises
    ax = axes[1]
    ax.plot(t, ts['von_mises'], 'C3-', lw=2, label='von Mises stress')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Von Mises stress (kPa)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'stress_timeseries.png'), dpi=200)
    return fig


def plot_strain_rate_timeseries(run_dir, outdir=None):
    """Plot volumetric and shear strain rates over time.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output.
    outdir : str or None
        Output directory for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ts = extract_continuum_timeseries(run_dir)
    t = ts['time']

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, ts['volumetric_strain_rate'], 'C0-', lw=2,
            label='Volumetric strain rate (compaction)')
    ax.set_ylabel(r'$\dot{\varepsilon}_{vol}$ (1/h)')
    ax.set_title('Strain Rate Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', lw=0.5, ls=':')

    ax = axes[1]
    ax.plot(t, ts['shear_strain_rate'], 'C1-', lw=2,
            label='Deviatoric (shear) strain rate')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel(r'$|\dot{\varepsilon}_{dev}|$ (1/h)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'strain_rate_timeseries.png'), dpi=200)
    return fig


def plot_viscosity_evolution(run_dir, outdir=None):
    """Plot effective viscosity over time with packing fraction on twin axis.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output.
    outdir : str or None
        Output directory for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ts = extract_continuum_timeseries(run_dir)
    t = ts['time']
    eta = ts['eta_eff'].copy()

    # Cap infinite values for plotting
    finite_mask = np.isfinite(eta)
    if np.any(finite_mask):
        eta_max = np.max(eta[finite_mask]) * 10
    else:
        eta_max = 1.0
    eta[~finite_mask] = eta_max

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = 'C0'
    ax1.semilogy(t, eta, color=color1, lw=2, label=r'$\eta_{eff}$')
    ax1.set_xlabel('Time (h)')
    ax1.set_ylabel(r'Effective viscosity $\eta_{eff}$ (kPa$\cdot$h)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title('Effective Viscosity Evolution')
    ax1.grid(True, alpha=0.3)

    # Twin axis: packing fraction
    ax2 = ax1.twinx()
    color2 = 'C3'
    phi = ts.get('phi_solid', np.zeros(len(t)))
    if np.any(phi > 0):
        ax2.plot(t, phi, color=color2, lw=1.5, ls='--', label=r'$\phi_{solid}$')
        ax2.set_ylabel(r'Solid packing fraction $\phi$', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

    fig.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'viscosity_evolution.png'), dpi=200)
    return fig


def plot_coordination_evolution(run_dir, outdir=None):
    """Plot coordination number over time, decomposed by granule type.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output.
    outdir : str or None
        Output directory for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    ts = extract_continuum_timeseries(run_dir)
    t = ts['time']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, ts['Z'], 'k-', lw=2, label=r'$Z$ (total)')
    ax.plot(t, ts['Z_ff'], 'C1--', lw=1.5, label=r'$Z_{ff}$ (func-func)')
    ax.plot(t, ts['Z_fi'], 'C2-.', lw=1.5, label=r'$Z_{fi}$ (func-inert)')
    ax.plot(t, ts['Z_ii'], 'C0:', lw=1.5, label=r'$Z_{ii}$ (inert-inert)')
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Coordination number')
    ax.set_title('Coordination Number Evolution')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    if outdir:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        fig.savefig(str(Path(outdir) / 'coordination_evolution.png'), dpi=200)
    return fig


def run_all(run_dir, outdir=None):
    """Generate all coarse-graining plots.

    Parameters
    ----------
    run_dir : str
        Path to saved simulation output directory.
    outdir : str or None
        Output directory for saved figures. Defaults to run_dir/visualizations.

    Returns
    -------
    dict of matplotlib.figure.Figure
    """
    if outdir is None:
        outdir = str(Path(run_dir) / 'visualizations')

    Path(outdir).mkdir(parents=True, exist_ok=True)

    figs = {}
    figs['stress'] = plot_stress_timeseries(run_dir, outdir)
    figs['strain_rate'] = plot_strain_rate_timeseries(run_dir, outdir)
    figs['viscosity'] = plot_viscosity_evolution(run_dir, outdir)
    figs['coordination'] = plot_coordination_evolution(run_dir, outdir)

    print(f"  Coarse-graining plots saved to {outdir}")
    return figs


# ======================================================================
# 8. CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Continuum coarse-graining for GELLS-DEM simulations')
    parser.add_argument('-i', '--input', required=True,
                        help='Input run directory or .tar.gz archive')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory for plots (default: <input>/visualizations)')
    args = parser.parse_args()

    indir = args.input
    outdir = args.outdir or str(Path(indir) / 'visualizations')
    run_all(indir, outdir)
