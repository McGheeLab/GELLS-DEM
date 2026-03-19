"""
Tissue Architecture Descriptors for GELS
==============================================
Computes a comprehensive tissue architecture descriptor vector from 3D
phase fields -- used to characterise both GELS scaffolds and compare
to native organ architectures.

Descriptor categories:
    1. Volume & surface measures (BV/TV, S/V, specific surface)
    2. Thickness & spacing (Hildebrand & Ruegsegger 1997)
    3. Minkowski functionals (volume, surface, curvature, Euler)
    4. Structure model index (plates vs rods vs spheres)
    5. Spatial correlation functions (two-point, chord length)
    6. Anisotropy (mean intercept length tensor)
    7. Tortuosity (Laplace solver)
    8. Pore size distribution

Data is loaded via:
    from new_dem_0 import load_run
    hist, snaps, p, metadata = load_run(run_dir)
    # snaps[i] is a dict with 'phi_f', 'phi_i', 'phi_v' as 3D arrays

Or from .npz files:
    data = np.load('fields/fields_0000.npz')
    phi_f, phi_i, phi_v = data['phi_f'], data['phi_i'], data['phi_v']

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).

Usage:
    python analysis/tissue_descriptors.py -i results/default
    python analysis/tissue_descriptors.py -i results/default --snap -1 -o ./analysis
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, label
from pathlib import Path
import argparse
import json
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from skimage.measure import marching_cubes, euler_number
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Silence marching_cubes deprecation warnings when present
warnings.filterwarnings('ignore', category=DeprecationWarning, module='skimage')


# ======================================================================
# Helpers
# ======================================================================

def _crop_inner(arr, boundary_exclusion):
    """Crop the inner region of an array, excluding boundary_exclusion
    fraction from each edge.

    Parameters
    ----------
    arr : ndarray
        2D or 3D array.
    boundary_exclusion : float
        Fraction to exclude from each edge (e.g., 0.2 removes 20% from
        each side, leaving the inner 60%).

    Returns
    -------
    ndarray
        Cropped array.
    """
    if boundary_exclusion <= 0:
        return arr
    shape = arr.shape
    lo = [int(boundary_exclusion * s) for s in shape]
    hi = [int((1.0 - boundary_exclusion) * s) for s in shape]
    if arr.ndim == 3:
        return arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    elif arr.ndim == 2:
        return arr[lo[0]:hi[0], lo[1]:hi[1]]
    return arr


def _binarize(phi, threshold=0.5):
    """Binarize a phase field at the given threshold."""
    return (phi > threshold).astype(np.bool_)


def _triangle_areas(verts, faces):
    """Compute areas of triangles defined by vertices and face indices.

    Parameters
    ----------
    verts : ndarray, shape (N, 3)
        Vertex coordinates.
    faces : ndarray, shape (M, 3)
        Triangle face indices into verts.

    Returns
    -------
    ndarray, shape (M,)
        Area of each triangle.
    """
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _surface_area_from_binary(binary, dx):
    """Compute surface area from a binary volume via marching cubes.

    Parameters
    ----------
    binary : ndarray (3D bool)
        Binary volume.
    dx : float
        Voxel size in micrometres.

    Returns
    -------
    float
        Surface area in um^2, or 0 if marching cubes fails.
    """
    if not HAS_SKIMAGE:
        return 0.0
    if binary.sum() == 0 or binary.all():
        return 0.0
    try:
        verts, faces, _, _ = marching_cubes(
            binary.astype(np.float32), level=0.5, spacing=(dx, dx, dx))
        return float(np.sum(_triangle_areas(verts, faces)))
    except Exception:
        return 0.0


def _radial_average(corr_3d, dx, max_r=None, n_bins=50):
    """Radially average a 3D correlation function.

    Parameters
    ----------
    corr_3d : ndarray (3D)
        Correlation values (already shifted so origin is at center).
    dx : float
        Voxel size.
    max_r : float or None
        Maximum radial distance. Defaults to half the smallest dimension.
    n_bins : int
        Number of radial bins.

    Returns
    -------
    r : ndarray
        Radial distance centres.
    S2 : ndarray
        Radially averaged values.
    """
    nz, ny, nx = corr_3d.shape
    if max_r is None:
        max_r = min(nz, ny, nx) * dx / 2.0

    # Distance grid (origin at center)
    iz = np.arange(nz) - nz // 2
    iy = np.arange(ny) - ny // 2
    ix = np.arange(nx) - nx // 2
    rr = np.sqrt((iz[:, None, None] * dx) ** 2 +
                 (iy[None, :, None] * dx) ** 2 +
                 (ix[None, None, :] * dx) ** 2)

    bin_edges = np.linspace(0, max_r, n_bins + 1)
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    S2 = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (rr >= bin_edges[i]) & (rr < bin_edges[i + 1])
        if mask.any():
            S2[i] = float(np.mean(corr_3d[mask]))
    return r_centers, S2


# ======================================================================
# 1. Volume & Surface Measures
# ======================================================================

def volume_fractions(phi_f, phi_i, phi_v, dx):
    """Compute volume fractions and surface density.

    Parameters
    ----------
    phi_f : ndarray
        Functional phase field (tissue).
    phi_i : ndarray
        Inert phase field.
    phi_v : ndarray
        Void phase field.
    dx : float
        Voxel size in micrometres.

    Returns
    -------
    dict
        bv_tv : float
            Tissue (bone) volume fraction (BV/TV equivalent).
        porosity : float
            Pore volume fraction (1 - bv_tv).
        surface_density : float
            Surface area per total volume S/V_total (um^-1).
        specific_surface : float
            Surface area per tissue volume S/V_tissue (um^-1).
    """
    tissue = _binarize(phi_f, 0.5)
    bv_tv = float(np.mean(tissue))
    porosity = 1.0 - bv_tv

    S = _surface_area_from_binary(tissue, dx)

    ndim = phi_f.ndim
    if ndim == 3:
        V_total = phi_f.size * dx ** 3
    else:
        V_total = phi_f.size * dx ** 2

    surface_density = S / V_total if V_total > 0 else 0.0
    V_tissue = bv_tv * V_total
    specific_surface = S / V_tissue if V_tissue > 0 else 0.0

    return dict(
        bv_tv=bv_tv,
        porosity=porosity,
        surface_density=surface_density,
        specific_surface=specific_surface,
    )


# ======================================================================
# 2. Thickness & Spacing (Hildebrand & Ruegsegger 1997)
# ======================================================================

def thickness_spacing(tissue_binary, dx):
    """Compute mean tissue thickness and pore spacing via distance transform.

    Uses the simplified approach: Tb.Th ~ 2 * mean(EDT[tissue]) and
    Tb.Sp ~ 2 * mean(EDT[pore]).

    Parameters
    ----------
    tissue_binary : ndarray (bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.

    Returns
    -------
    dict
        tb_th : float
            Mean tissue thickness (um).
        tb_sp : float
            Mean pore spacing (um).
        tb_n : float
            Feature number = 1 / (Tb.Th + Tb.Sp) (um^-1).
        thickness_distribution : ndarray
            Local thickness values at tissue voxels (um).
        spacing_distribution : ndarray
            Local spacing values at pore voxels (um).
    """
    # Distance transform for tissue phase
    dt_tissue = distance_transform_edt(tissue_binary, sampling=dx)
    # Distance transform for pore phase
    pore_binary = ~tissue_binary
    dt_pore = distance_transform_edt(pore_binary, sampling=dx)

    # Local thickness = 2 * distance to nearest boundary
    thickness_vals = 2.0 * dt_tissue[tissue_binary]
    spacing_vals = 2.0 * dt_pore[pore_binary]

    tb_th = float(np.mean(thickness_vals)) if len(thickness_vals) > 0 else 0.0
    tb_sp = float(np.mean(spacing_vals)) if len(spacing_vals) > 0 else 0.0
    denom = tb_th + tb_sp
    tb_n = 1.0 / denom if denom > 0 else 0.0

    return dict(
        tb_th=tb_th,
        tb_sp=tb_sp,
        tb_n=tb_n,
        thickness_distribution=thickness_vals,
        spacing_distribution=spacing_vals,
    )


# ======================================================================
# 3. Minkowski Functionals
# ======================================================================

def minkowski_functionals(tissue_binary, dx):
    """Compute four Minkowski functionals in 3D.

    Parameters
    ----------
    tissue_binary : ndarray (3D bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.

    Returns
    -------
    dict
        volume : float
            Total tissue volume (um^3).
        surface_area : float
            Total interface area (um^2).
        mean_curvature_integral : float
            Integral of mean curvature over the surface.
        euler_characteristic : int
            chi = #components - #tunnels + #cavities.
        connectivity_density : float
            -chi / V_total (um^-3).
    """
    volume = float(np.sum(tissue_binary)) * dx ** 3

    surface_area = _surface_area_from_binary(tissue_binary, dx)

    # Euler characteristic
    chi = 0
    if HAS_SKIMAGE:
        try:
            chi = int(euler_number(tissue_binary, connectivity=3))
        except Exception:
            chi = 0

    V_total = tissue_binary.size * dx ** 3
    connectivity_density = -float(chi) / V_total if V_total > 0 else 0.0

    # Mean curvature integral from marching cubes mesh
    mean_curvature_integral = 0.0
    if HAS_SKIMAGE and tissue_binary.any() and not tissue_binary.all():
        try:
            verts, faces, normals, _ = marching_cubes(
                tissue_binary.astype(np.float32), level=0.5,
                spacing=(dx, dx, dx))

            # Per-vertex mean curvature estimation via angle defect / area.
            # Compute the mixed area and angle defect at each vertex.
            n_verts = len(verts)
            vertex_area = np.zeros(n_verts)
            vertex_curvature = np.zeros(n_verts)

            for fi in range(len(faces)):
                i0, i1, i2 = faces[fi]
                v0, v1, v2 = verts[i0], verts[i1], verts[i2]
                # Edge vectors
                e01 = v1 - v0
                e02 = v2 - v0
                e12 = v2 - v1
                area = 0.5 * np.linalg.norm(np.cross(e01, e02))
                # Distribute area equally to each vertex (1/3 each)
                third_area = area / 3.0
                vertex_area[i0] += third_area
                vertex_area[i1] += third_area
                vertex_area[i2] += third_area

            # Angle defect: 2*pi - sum_of_angles at each vertex
            vertex_angle_sum = np.zeros(n_verts)
            for fi in range(len(faces)):
                i0, i1, i2 = faces[fi]
                v0, v1, v2 = verts[i0], verts[i1], verts[i2]
                for vi, (va, vb, vc) in [(i0, (v0, v1, v2)),
                                          (i1, (v1, v0, v2)),
                                          (i2, (v2, v0, v1))]:
                    e1 = vb - va
                    e2 = vc - va
                    n1 = np.linalg.norm(e1)
                    n2 = np.linalg.norm(e2)
                    if n1 > 0 and n2 > 0:
                        cos_a = np.clip(np.dot(e1, e2) / (n1 * n2), -1, 1)
                        vertex_angle_sum[vi] += np.arccos(cos_a)

            angle_defect = 2.0 * np.pi - vertex_angle_sum
            # Mean curvature H at vertex ~ angle_defect / (2 * mixed_area)
            # Integral of H = sum(H * dA) ~ sum(angle_defect / 2)
            # The Gauss-Bonnet interpretation: integral of K dA = sum(angle_defect)
            # For mean curvature integral, use a different approach:
            # H_integral = sum over edges of (dihedral_angle * edge_length / 2)
            # This is more standard for discrete surfaces.

            # Edge-based mean curvature integral (Desburn et al. 1999)
            # Build edge -> face adjacency
            edge_faces = {}
            for fi in range(len(faces)):
                for ea, eb in [(faces[fi][0], faces[fi][1]),
                               (faces[fi][1], faces[fi][2]),
                               (faces[fi][2], faces[fi][0])]:
                    edge = (min(ea, eb), max(ea, eb))
                    edge_faces.setdefault(edge, []).append(fi)

            H_int = 0.0
            for (ea, eb), flist in edge_faces.items():
                if len(flist) == 2:
                    # Dihedral angle between two adjacent faces
                    # Use face normals from cross product
                    def _face_normal(fi):
                        i0, i1, i2 = faces[fi]
                        e1 = verts[i1] - verts[i0]
                        e2 = verts[i2] - verts[i0]
                        n = np.cross(e1, e2)
                        norm = np.linalg.norm(n)
                        return n / norm if norm > 0 else np.zeros(3)

                    n1 = _face_normal(flist[0])
                    n2 = _face_normal(flist[1])
                    cos_d = np.clip(np.dot(n1, n2), -1, 1)
                    dihedral = np.arccos(cos_d)
                    edge_len = np.linalg.norm(verts[eb] - verts[ea])
                    # Sign: positive for convex edges
                    # For unsigned integral:
                    H_int += dihedral * edge_len

            mean_curvature_integral = H_int / 2.0

        except Exception:
            mean_curvature_integral = 0.0

    return dict(
        volume=volume,
        surface_area=surface_area,
        mean_curvature_integral=mean_curvature_integral,
        euler_characteristic=chi,
        connectivity_density=connectivity_density,
    )


# ======================================================================
# 4. Structure Model Index (SMI)
# ======================================================================

def structure_model_index(tissue_binary, dx):
    """Compute the Structure Model Index.

    SMI distinguishes plates (0), rods (3), spheres (4).
    SMI = 6 * V * dS/dr / S^2

    dS/dr is estimated by dilating the binary by one voxel and measuring
    the surface area change.

    Parameters
    ----------
    tissue_binary : ndarray (3D bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.

    Returns
    -------
    dict
        smi : float
            Structure model index.
    """
    S = _surface_area_from_binary(tissue_binary, dx)
    if S <= 0:
        return dict(smi=0.0)

    V = float(np.sum(tissue_binary)) * dx ** 3

    # Dilate by one voxel
    dilated = binary_dilation(tissue_binary, iterations=1)
    S_prime = _surface_area_from_binary(dilated, dx)

    dS_dr = (S_prime - S) / dx
    smi = 6.0 * V * dS_dr / (S ** 2)

    return dict(smi=float(smi))


# ======================================================================
# 5. Spatial Correlation Functions
# ======================================================================

def two_point_correlation(tissue_binary, dx, max_r=None, n_bins=50):
    """Compute the two-point correlation function S2(r) via FFT.

    S2(r) = Pr[both x and x+r are in tissue]

    Parameters
    ----------
    tissue_binary : ndarray (3D bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.
    max_r : float or None
        Maximum radial distance in um. Defaults to half the smallest
        dimension.
    n_bins : int
        Number of radial bins.

    Returns
    -------
    dict
        r : ndarray
            Radial distance centres (um).
        S2 : ndarray
            Two-point correlation values.
        correlation_length : float
            Distance where S2 decays to (BV/TV)^2 + 1/e * (S2(0) - (BV/TV)^2).
    """
    I = tissue_binary.astype(np.float64)
    N = I.size

    # FFT-based autocorrelation
    F = np.fft.fftn(I)
    S2_3d = np.real(np.fft.ifftn(F * np.conj(F))) / N

    # Shift origin to center
    S2_3d = np.fft.fftshift(S2_3d)

    r, S2 = _radial_average(S2_3d, dx, max_r=max_r, n_bins=n_bins)

    # Extract correlation length
    bv_tv = float(np.mean(I))
    S2_0 = bv_tv  # S2(0) = volume fraction
    S2_inf = bv_tv ** 2
    threshold = S2_inf + (S2_0 - S2_inf) / np.e

    correlation_length = 0.0
    for i in range(len(S2)):
        if S2[i] <= threshold:
            correlation_length = float(r[i])
            break
    else:
        # Did not cross threshold within range
        correlation_length = float(r[-1]) if len(r) > 0 else 0.0

    return dict(
        r=r,
        S2=S2,
        correlation_length=correlation_length,
    )


def chord_length_distribution(tissue_binary, dx, n_rays=1000, phase='tissue',
                              rng=None):
    """Compute chord length distribution by casting random rays.

    Parameters
    ----------
    tissue_binary : ndarray (3D bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.
    n_rays : int
        Number of random rays to cast.
    phase : str
        'tissue' or 'pore' -- which phase to measure intercepts in.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    dict
        bin_centers : ndarray
            Chord length values (um).
        pdf : ndarray
            Probability density.
        mean_chord : float
            Mean chord length (um).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    target = tissue_binary if phase == 'tissue' else ~tissue_binary
    nz, ny, nx = target.shape
    chords = []

    for _ in range(n_rays):
        # Random axis: 0=z, 1=y, 2=x
        axis = rng.integers(0, 3)
        if axis == 0:
            iy = rng.integers(0, ny)
            ix = rng.integers(0, nx)
            line = target[:, iy, ix]
        elif axis == 1:
            iz = rng.integers(0, nz)
            ix = rng.integers(0, nx)
            line = target[iz, :, ix]
        else:
            iz = rng.integers(0, nz)
            iy = rng.integers(0, ny)
            line = target[iz, iy, :]

        # Find runs of True
        changes = np.diff(line.astype(np.int8))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        # Handle edge cases
        if line[0]:
            starts = np.concatenate([[0], starts])
        if line[-1]:
            ends = np.concatenate([ends, [len(line)]])

        for s, e in zip(starts, ends):
            chords.append((e - s) * dx)

    chords = np.array(chords) if len(chords) > 0 else np.array([0.0])
    mean_chord = float(np.mean(chords))

    # Histogram
    n_bins = max(10, min(50, len(chords) // 20))
    counts, bin_edges = np.histogram(chords, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return dict(
        bin_centers=bin_centers,
        pdf=counts,
        mean_chord=mean_chord,
    )


# ======================================================================
# 6. Anisotropy (Mean Intercept Length Tensor)
# ======================================================================

def mean_intercept_length(tissue_binary, dx, n_directions=50, rng=None):
    """Compute the Mean Intercept Length tensor.

    Parameters
    ----------
    tissue_binary : ndarray (3D bool)
        Binary tissue volume.
    dx : float
        Voxel size in micrometres.
    n_directions : int
        Number of directions sampled uniformly on the unit sphere.
    rng : numpy.random.Generator or None
        Random number generator.

    Returns
    -------
    dict
        da : float
            Degree of anisotropy = lambda_max / lambda_min.
        fa : float
            Fractional anisotropy (0-1 scale).
        principal_direction : ndarray, shape (3,)
            Eigenvector corresponding to the largest eigenvalue.
        mil_tensor : ndarray, shape (3, 3)
            Symmetric MIL tensor.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nz, ny, nx = tissue_binary.shape

    # Sample directions uniformly on the unit sphere (Fibonacci spiral)
    directions = np.zeros((n_directions, 3))
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n_directions):
        theta = np.arccos(1.0 - 2.0 * (i + 0.5) / n_directions)
        phi = 2.0 * np.pi * i / golden_ratio
        directions[i] = [np.sin(theta) * np.cos(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(theta)]

    mil_values = np.zeros(n_directions)
    n_lines_per_dir = max(50, min(200, min(nz, ny, nx)))

    for di, d in enumerate(directions):
        total_length = 0.0
        total_intersections = 0

        # Cast parallel lines: pick random starting points on the plane
        # perpendicular to direction d
        for _ in range(n_lines_per_dir):
            # Random starting point
            p0 = rng.uniform(0, [nz, ny, nx]).astype(np.float64)
            # Walk along direction d in voxel steps
            step = d / np.linalg.norm(d)  # unit step in voxel coords
            max_steps = int(1.5 * max(nz, ny, nx))
            prev_in_tissue = None
            n_crossings = 0
            for s in range(max_steps):
                pos = p0 + s * step
                iz, iy, ix = int(pos[0]), int(pos[1]), int(pos[2])
                if iz < 0 or iz >= nz or iy < 0 or iy >= ny or ix < 0 or ix >= nx:
                    break
                in_tissue = bool(tissue_binary[iz, iy, ix])
                if prev_in_tissue is not None and in_tissue != prev_in_tissue:
                    n_crossings += 1
                prev_in_tissue = in_tissue
            line_length = s * dx  # approximate length traversed
            total_length += line_length
            total_intersections += n_crossings

        if total_intersections > 0:
            mil_values[di] = total_length / total_intersections
        else:
            mil_values[di] = total_length  # no intersections

    # Build MIL tensor: M_ij = (1/N) sum MIL(n)^2 * n_i * n_j
    M = np.zeros((3, 3))
    for di in range(n_directions):
        n = directions[di]
        M += mil_values[di] ** 2 * np.outer(n, n)
    M /= n_directions

    # Eigendecompose
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    lam_max = eigenvalues[0]
    lam_min = eigenvalues[-1]
    da = float(lam_max / lam_min) if lam_min > 0 else 1.0

    # Fractional anisotropy (FA)
    lam_mean = np.mean(eigenvalues)
    if lam_mean > 0:
        fa = float(np.sqrt(1.5 * np.sum((eigenvalues - lam_mean) ** 2) /
                           np.sum(eigenvalues ** 2)))
    else:
        fa = 0.0

    principal_direction = eigenvectors[:, 0]

    return dict(
        da=da,
        fa=fa,
        principal_direction=principal_direction,
        mil_tensor=M,
    )


# ======================================================================
# 7. Tortuosity (Laplace solver)
# ======================================================================

def tortuosity(pore_binary, dx, direction='z', tol=1e-4, max_iter=5000):
    """Compute tortuosity via a Laplace equation solver.

    Solves nabla^2 psi = 0 in pore space with Dirichlet BCs on two
    opposing faces and Neumann (zero flux) on pore-solid interfaces.

    tau = <|grad psi|^2> / <d psi / d_dir>^2

    Parameters
    ----------
    pore_binary : ndarray (3D bool)
        Binary pore volume (True = pore).
    dx : float
        Voxel size in micrometres.
    direction : str
        Direction for the potential gradient: 'x', 'y', or 'z'.
    tol : float
        Convergence tolerance (max absolute update).
    max_iter : int
        Maximum Jacobi iterations.

    Returns
    -------
    dict
        tau : float
            Tortuosity (>= 1, higher = more tortuous).
        psi : ndarray
            Solved potential field.
    """
    nz, ny, nx = pore_binary.shape

    # Map direction to axis index
    dir_map = {'z': 0, 'y': 1, 'x': 2}
    axis = dir_map.get(direction, 0)

    # Initialize potential: linear gradient in the chosen direction
    psi = np.zeros_like(pore_binary, dtype=np.float64)
    n_along = pore_binary.shape[axis]
    for i in range(n_along):
        slc = [slice(None)] * 3
        slc[axis] = i
        psi[tuple(slc)] = float(i) / max(1, n_along - 1)

    # Apply pore mask: set solid voxels to NaN as marker
    psi[~pore_binary] = 0.0

    # Check if pore space is connected from face to face
    slc_lo = [slice(None)] * 3
    slc_lo[axis] = 0
    slc_hi = [slice(None)] * 3
    slc_hi[axis] = n_along - 1
    has_lo = np.any(pore_binary[tuple(slc_lo)])
    has_hi = np.any(pore_binary[tuple(slc_hi)])
    if not (has_lo and has_hi):
        # No connected path possible
        return dict(tau=float('inf'), psi=psi)

    # Jacobi iteration
    for iteration in range(max_iter):
        psi_old = psi.copy()

        # Average of 6 neighbours
        psi_new = np.zeros_like(psi)
        count = np.zeros_like(psi)

        for ax in range(3):
            for shift in [-1, 1]:
                shifted = np.roll(psi, shift, axis=ax)
                # Mask: only count pore neighbours
                shifted_mask = np.roll(pore_binary, shift, axis=ax)
                valid = pore_binary & shifted_mask
                psi_new[valid] += shifted[valid]
                count[valid] += 1.0

        # Update interior pore voxels
        interior = pore_binary & (count > 0)
        psi[interior] = psi_new[interior] / count[interior]

        # Enforce Dirichlet BCs on entry/exit faces
        slc_bc0 = [slice(None)] * 3
        slc_bc0[axis] = 0
        bc0_mask = pore_binary[tuple(slc_bc0)]
        face0 = psi[tuple(slc_bc0)]
        face0[bc0_mask] = 0.0
        psi[tuple(slc_bc0)] = face0

        slc_bc1 = [slice(None)] * 3
        slc_bc1[axis] = n_along - 1
        bc1_mask = pore_binary[tuple(slc_bc1)]
        face1 = psi[tuple(slc_bc1)]
        face1[bc1_mask] = 1.0
        psi[tuple(slc_bc1)] = face1

        # Zero out solid voxels
        psi[~pore_binary] = 0.0

        # Check convergence
        diff = np.abs(psi[pore_binary] - psi_old[pore_binary])
        if len(diff) > 0 and np.max(diff) < tol:
            break

    # Compute tortuosity: tau = <|grad psi|^2> / <d psi / d_dir>^2
    grad = np.gradient(psi, dx, axis=(0, 1, 2))
    grad_mag_sq = grad[0] ** 2 + grad[1] ** 2 + grad[2] ** 2

    # Only consider pore voxels (exclude boundary faces)
    interior_mask = pore_binary.copy()
    slc_rm0 = [slice(None)] * 3
    slc_rm0[axis] = 0
    interior_mask[tuple(slc_rm0)] = False
    slc_rm1 = [slice(None)] * 3
    slc_rm1[axis] = n_along - 1
    interior_mask[tuple(slc_rm1)] = False

    if np.sum(interior_mask) == 0:
        return dict(tau=1.0, psi=psi)

    mean_grad_sq = float(np.mean(grad_mag_sq[interior_mask]))
    mean_ddir = float(np.mean(grad[axis][interior_mask]))

    if mean_ddir ** 2 > 0:
        tau = mean_grad_sq / (mean_ddir ** 2)
    else:
        tau = float('inf')

    # Clamp to physical range
    tau = max(1.0, tau)

    return dict(tau=float(tau), psi=psi)


# ======================================================================
# 8. Pore Size Distribution
# ======================================================================

def pore_size_distribution(pore_binary, dx, n_bins=30):
    """Compute the distribution of inscribed sphere radii in pore space.

    Uses the Euclidean distance transform.

    Parameters
    ----------
    pore_binary : ndarray (3D bool)
        Binary pore volume.
    dx : float
        Voxel size in micrometres.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    dict
        bin_centers : ndarray
            Pore radius values (um).
        pdf : ndarray
            Probability density.
        mean_pore_radius : float
            Mean inscribed pore radius (um).
        median_pore_radius : float
            Median inscribed pore radius (um).
    """
    dt = distance_transform_edt(pore_binary, sampling=dx)
    pore_radii = dt[pore_binary]

    if len(pore_radii) == 0:
        return dict(
            bin_centers=np.zeros(n_bins),
            pdf=np.zeros(n_bins),
            mean_pore_radius=0.0,
            median_pore_radius=0.0,
        )

    mean_pore_radius = float(np.mean(pore_radii))
    median_pore_radius = float(np.median(pore_radii))

    counts, bin_edges = np.histogram(pore_radii, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    return dict(
        bin_centers=bin_centers,
        pdf=counts,
        mean_pore_radius=mean_pore_radius,
        median_pore_radius=median_pore_radius,
    )


# ======================================================================
# 9. Master Descriptor Vector
# ======================================================================

def compute_descriptor_vector(phi_f, phi_i, phi_v, dx, boundary_exclusion=0.2):
    """Compute the full tissue architecture descriptor vector.

    The tissue phase is phi_f. The pore phase is (1 - phi_f), i.e.,
    inert granules are treated as void/pore space in the tissue
    architecture analysis.

    Parameters
    ----------
    phi_f : ndarray
        3D (or 2D) functional phase field.
    phi_i : ndarray
        3D (or 2D) inert phase field.
    phi_v : ndarray
        3D (or 2D) void phase field.
    dx : float
        Voxel size in micrometres.
    boundary_exclusion : float
        Fraction to exclude from each edge before computing descriptors.

    Returns
    -------
    dict
        Descriptor dictionary with keys:
        'bv_tv', 'porosity', 'surface_density', 'specific_surface',
        'tb_th', 'tb_sp', 'tb_n',
        'euler_characteristic', 'connectivity_density',
        'smi',
        'correlation_length',
        'mean_chord_tissue', 'mean_chord_pore',
        'da', 'fa',
        'tortuosity',
        'mean_pore_radius',
        'permeability_KC'
    """
    # Crop inner region
    pf = _crop_inner(phi_f, boundary_exclusion)
    pi = _crop_inner(phi_i, boundary_exclusion)
    pv = _crop_inner(phi_v, boundary_exclusion)

    is_3d = pf.ndim == 3

    tissue = _binarize(pf, 0.5)
    pore = ~tissue

    result = {}

    # -- 1. Volume & surface --
    vf = volume_fractions(pf, pi, pv, dx)
    result.update(vf)

    # Kozeny-Carman permeability from porosity
    porosity = vf['porosity']
    # Estimate mean grain diameter from tissue features
    if porosity > 0.01 and porosity < 0.99:
        # Use 2 * mean distance transform in tissue as grain size proxy
        dt_t = distance_transform_edt(tissue, sampling=dx)
        d_grain = 2.0 * float(np.mean(dt_t[tissue])) if tissue.any() else 1.0
        eps = np.clip(porosity, 0.01, 0.99)
        permeability_KC = eps ** 3 * d_grain ** 2 / (180.0 * (1.0 - eps) ** 2)
    else:
        permeability_KC = 0.0
        d_grain = 0.0
    result['permeability_KC'] = float(permeability_KC)

    # -- 2. Thickness & spacing --
    ts = thickness_spacing(tissue, dx)
    result['tb_th'] = ts['tb_th']
    result['tb_sp'] = ts['tb_sp']
    result['tb_n'] = ts['tb_n']
    result['thickness_distribution'] = ts['thickness_distribution']
    result['spacing_distribution'] = ts['spacing_distribution']

    if is_3d:
        # -- 3. Minkowski functionals --
        mf = minkowski_functionals(tissue, dx)
        result['volume'] = mf['volume']
        result['surface_area_mink'] = mf['surface_area']
        result['mean_curvature_integral'] = mf['mean_curvature_integral']
        result['euler_characteristic'] = mf['euler_characteristic']
        result['connectivity_density'] = mf['connectivity_density']

        # -- 4. SMI --
        smi_result = structure_model_index(tissue, dx)
        result['smi'] = smi_result['smi']

        # -- 5. Spatial correlations --
        tpc = two_point_correlation(tissue, dx)
        result['correlation_length'] = tpc['correlation_length']
        result['S2_r'] = tpc['r']
        result['S2_values'] = tpc['S2']

        cld_t = chord_length_distribution(tissue, dx, phase='tissue')
        result['mean_chord_tissue'] = cld_t['mean_chord']

        cld_p = chord_length_distribution(tissue, dx, phase='pore')
        result['mean_chord_pore'] = cld_p['mean_chord']

        # -- 6. Anisotropy --
        mil = mean_intercept_length(tissue, dx)
        result['da'] = mil['da']
        result['fa'] = mil['fa']
        result['principal_direction'] = mil['principal_direction']
        result['mil_tensor'] = mil['mil_tensor']

        # -- 7. Tortuosity --
        tort = tortuosity(pore, dx)
        result['tortuosity'] = tort['tau']

        # -- 8. Pore size distribution --
        psd = pore_size_distribution(pore, dx)
        result['mean_pore_radius'] = psd['mean_pore_radius']
        result['median_pore_radius'] = psd['median_pore_radius']
        result['psd_bin_centers'] = psd['bin_centers']
        result['psd_pdf'] = psd['pdf']
    else:
        # 2D mode: skip 3D-only descriptors, provide defaults
        result['euler_characteristic'] = 0
        result['connectivity_density'] = 0.0
        result['smi'] = 0.0
        result['correlation_length'] = 0.0
        result['mean_chord_tissue'] = 0.0
        result['mean_chord_pore'] = 0.0
        result['da'] = 1.0
        result['fa'] = 0.0
        result['tortuosity'] = 1.0
        result['mean_pore_radius'] = 0.0

    return result


# ======================================================================
# 10. Convenience Functions
# ======================================================================

def from_run(run_dir, snap_index=-1):
    """Load a run and compute descriptors from a specific snapshot.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or .tar.gz archive.
    snap_index : int
        Which snapshot to analyse (-1 = last).

    Returns
    -------
    dict
        Descriptor dictionary from compute_descriptor_vector().
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)
    if not snaps:
        raise ValueError(f"No snapshots found in {run_dir}")

    snap = snaps[snap_index]
    phi_f = snap['phi_f']
    phi_i = snap['phi_i']
    phi_v = snap['phi_v']

    # Determine voxel size
    if phi_f.ndim == 3:
        dx = p.Lx / phi_f.shape[2]  # shape is (Nz, Ny, Nx)
    else:
        dx = p.Lx / phi_f.shape[1]

    bx = getattr(p, 'boundary_exclusion', 0.2)

    return compute_descriptor_vector(phi_f, phi_i, phi_v, dx, bx)


def descriptor_timeseries(run_dir):
    """Compute descriptor vector at each saved timepoint.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or .tar.gz archive.

    Returns
    -------
    list of dict
        One descriptor dictionary per snapshot.
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)
    if not snaps:
        raise ValueError(f"No snapshots found in {run_dir}")

    if snaps[0]['phi_f'].ndim == 3:
        dx = p.Lx / snaps[0]['phi_f'].shape[2]
    else:
        dx = p.Lx / snaps[0]['phi_f'].shape[1]

    bx = getattr(p, 'boundary_exclusion', 0.2)

    results = []
    for i, snap in enumerate(snaps):
        t = hist[i]['time'] if i < len(hist) else float(i)
        desc = compute_descriptor_vector(
            snap['phi_f'], snap['phi_i'], snap['phi_v'], dx, bx)
        desc['time'] = t
        desc['snap_index'] = i
        results.append(desc)

    return results


def run_all(run_dir, outdir=None):
    """Compute descriptors and make summary plots.

    Parameters
    ----------
    run_dir : str
        Path to the output directory or .tar.gz archive.
    outdir : str or None
        Output directory for plots and JSON summary. Defaults to
        ``run_dir/tissue_descriptors``.
    """
    from new_dem_0 import load_run

    hist, snaps, p, metadata = load_run(run_dir)
    if not snaps:
        print(f"  tissue_descriptors: no snapshots in {run_dir}, skipping")
        return

    if outdir is None:
        rd = run_dir[:-7] if run_dir.endswith('.tar.gz') else run_dir
        outdir = str(Path(rd) / 'tissue_descriptors')
    Path(outdir).mkdir(parents=True, exist_ok=True)

    if snaps[0]['phi_f'].ndim == 3:
        dx = p.Lx / snaps[0]['phi_f'].shape[2]
    else:
        dx = p.Lx / snaps[0]['phi_f'].shape[1]
    bx = getattr(p, 'boundary_exclusion', 0.2)

    # Compute descriptors for last snapshot
    snap = snaps[-1]
    print("  tissue_descriptors: computing descriptor vector...")
    desc = compute_descriptor_vector(
        snap['phi_f'], snap['phi_i'], snap['phi_v'], dx, bx)

    # Save scalar descriptors to JSON (exclude large arrays)
    scalar_keys = [
        'bv_tv', 'porosity', 'surface_density', 'specific_surface',
        'tb_th', 'tb_sp', 'tb_n',
        'euler_characteristic', 'connectivity_density',
        'smi', 'correlation_length',
        'mean_chord_tissue', 'mean_chord_pore',
        'da', 'fa', 'tortuosity',
        'mean_pore_radius', 'median_pore_radius',
        'permeability_KC',
    ]
    scalar_desc = {}
    for k in scalar_keys:
        if k in desc:
            v = desc[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                scalar_desc[k] = float(v)

    json_path = Path(outdir) / 'descriptors.json'
    with open(json_path, 'w') as f:
        json.dump(scalar_desc, f, indent=2)
    print(f"  tissue_descriptors: saved {json_path}")

    # Generate plots
    print("  tissue_descriptors: generating plots...")

    plot_descriptor_summary(desc, outdir=outdir)
    plot_thickness_distribution(desc, outdir=outdir)

    if 'S2_r' in desc and 'S2_values' in desc:
        plot_correlation_function(
            desc['S2_r'], desc['S2_values'],
            correlation_length=desc.get('correlation_length'),
            outdir=outdir)

    if 'psd_bin_centers' in desc and 'psd_pdf' in desc:
        plot_pore_size_distribution(desc, outdir=outdir)

    print(f"  tissue_descriptors: all outputs saved to {outdir}")


# ======================================================================
# 11. Plotting
# ======================================================================

def plot_descriptor_summary(descriptors, ax=None, outdir=None):
    """Bar chart of all scalar descriptors (normalised for comparison).

    Parameters
    ----------
    descriptors : dict
        Descriptor dictionary.
    ax : matplotlib.axes.Axes or None
        If provided, plot into this axes.
    outdir : str or None
        If provided, save figure to this directory.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Select scalar descriptors and normalise
    display_keys = [
        ('bv_tv', 'BV/TV'),
        ('porosity', 'Porosity'),
        ('surface_density', 'S/V total (um^-1)'),
        ('tb_th', 'Tb.Th (um)'),
        ('tb_sp', 'Tb.Sp (um)'),
        ('tb_n', 'Tb.N (um^-1)'),
        ('smi', 'SMI'),
        ('da', 'DA'),
        ('fa', 'FA'),
        ('tortuosity', 'Tortuosity'),
        ('mean_pore_radius', 'Pore radius (um)'),
        ('correlation_length', 'Corr. length (um)'),
        ('connectivity_density', 'Conn. density (um^-3)'),
    ]

    keys = []
    labels = []
    values = []
    for k, lab in display_keys:
        if k in descriptors:
            v = descriptors[k]
            if isinstance(v, (int, float, np.integer, np.floating)):
                if np.isfinite(v):
                    keys.append(k)
                    labels.append(lab)
                    values.append(float(v))

    if len(values) == 0:
        return None

    values = np.array(values)
    # Normalise to [0, 1] for display
    vmin = values.min()
    vmax = values.max()
    if vmax > vmin:
        normed = (values - vmin) / (vmax - vmin)
    else:
        normed = np.ones_like(values) * 0.5

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    colors = plt.cm.viridis(normed)
    bars = ax.barh(np.arange(len(labels)), values, color=colors, edgecolor='k',
                   linewidth=0.5)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel('Value')
    ax.set_title('Tissue Architecture Descriptors')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Annotate bars with values
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(bar.get_width() + 0.02 * (vmax - vmin if vmax > vmin else 1),
                bar.get_y() + bar.get_height() / 2,
                f'{val:.4g}', va='center', fontsize=8)

    plt.tight_layout()

    if outdir:
        fig.savefig(Path(outdir) / 'descriptor_summary.png', dpi=150,
                    bbox_inches='tight')
    return fig


def plot_thickness_distribution(descriptors, ax=None, outdir=None):
    """Histogram of Tb.Th and Tb.Sp distributions.

    Parameters
    ----------
    descriptors : dict
        Must contain 'thickness_distribution' and 'spacing_distribution'.
    ax : matplotlib.axes.Axes or None
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    th_dist = descriptors.get('thickness_distribution')
    sp_dist = descriptors.get('spacing_distribution')

    if th_dist is None and sp_dist is None:
        return None
    if th_dist is not None and len(th_dist) == 0:
        th_dist = None
    if sp_dist is not None and len(sp_dist) == 0:
        sp_dist = None
    if th_dist is None and sp_dist is None:
        return None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    n_bins = 40
    if th_dist is not None and len(th_dist) > 0:
        ax.hist(th_dist, bins=n_bins, alpha=0.6, label='Tb.Th (tissue)',
                color='C0', density=True, edgecolor='k', linewidth=0.3)
    if sp_dist is not None and len(sp_dist) > 0:
        ax.hist(sp_dist, bins=n_bins, alpha=0.6, label='Tb.Sp (pore)',
                color='C1', density=True, edgecolor='k', linewidth=0.3)

    ax.set_xlabel('Distance (um)')
    ax.set_ylabel('Probability density')
    ax.set_title('Thickness & Spacing Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outdir:
        fig.savefig(Path(outdir) / 'thickness_distribution.png', dpi=150,
                    bbox_inches='tight')
    return fig


def plot_correlation_function(r, S2, correlation_length=None, ax=None,
                              outdir=None):
    """Plot the two-point correlation function S2(r).

    Parameters
    ----------
    r : ndarray
        Radial distances (um).
    S2 : ndarray
        Correlation values.
    correlation_length : float or None
        If given, mark on the plot.
    ax : matplotlib.axes.Axes or None
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    ax.plot(r, S2, 'C0-', lw=2, label='$S_2(r)$')

    if correlation_length is not None and np.isfinite(correlation_length):
        ax.axvline(correlation_length, color='C3', ls='--', lw=1.5,
                   label=f'corr. length = {correlation_length:.1f} um')

    ax.set_xlabel('Radial distance r (um)')
    ax.set_ylabel('$S_2(r)$')
    ax.set_title('Two-Point Correlation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outdir:
        fig.savefig(Path(outdir) / 'correlation_function.png', dpi=150,
                    bbox_inches='tight')
    return fig


def plot_pore_size_distribution(descriptors, ax=None, outdir=None):
    """Plot the pore size distribution histogram.

    Parameters
    ----------
    descriptors : dict
        Must contain 'psd_bin_centers' and 'psd_pdf'.
    ax : matplotlib.axes.Axes or None
    outdir : str or None

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    bc = descriptors.get('psd_bin_centers')
    pdf = descriptors.get('psd_pdf')
    if bc is None or pdf is None:
        return None
    if len(bc) == 0:
        return None

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

    width = bc[1] - bc[0] if len(bc) > 1 else 1.0
    ax.bar(bc, pdf, width=width * 0.9, color='C2', edgecolor='k',
           linewidth=0.3, alpha=0.7)

    mean_r = descriptors.get('mean_pore_radius')
    median_r = descriptors.get('median_pore_radius')
    if mean_r is not None:
        ax.axvline(mean_r, color='C3', ls='--', lw=1.5,
                   label=f'mean = {mean_r:.1f} um')
    if median_r is not None:
        ax.axvline(median_r, color='C4', ls=':', lw=1.5,
                   label=f'median = {median_r:.1f} um')

    ax.set_xlabel('Inscribed pore radius (um)')
    ax.set_ylabel('Probability density')
    ax.set_title('Pore Size Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if outdir:
        fig.savefig(Path(outdir) / 'pore_size_distribution.png', dpi=150,
                    bbox_inches='tight')
    return fig


# ======================================================================
# 12. CLI
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Tissue architecture descriptors for GELS')
    parser.add_argument('-i', '--input', required=True,
                        help='Input run directory or .tar.gz archive')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory for plots and JSON')
    parser.add_argument('--snap', type=int, default=-1,
                        help='Snapshot index (default: -1 = last)')
    args = parser.parse_args()

    indir = args.input
    outdir = args.outdir

    print("=" * 65)
    print("  GELS Tissue Architecture Descriptors")
    print("=" * 65)

    if outdir is None and args.snap == -1:
        # Use run_all for full analysis
        run_all(indir, outdir=outdir)
    else:
        # Single snapshot analysis
        from new_dem_0 import load_run

        hist, snaps, p, metadata = load_run(indir)
        if not snaps:
            print(f"No snapshots found in {indir}")
        else:
            snap = snaps[args.snap]
            phi_f = snap['phi_f']
            phi_i = snap['phi_i']
            phi_v = snap['phi_v']

            if phi_f.ndim == 3:
                dx = p.Lx / phi_f.shape[2]
            else:
                dx = p.Lx / phi_f.shape[1]
            bx = getattr(p, 'boundary_exclusion', 0.2)

            desc = compute_descriptor_vector(phi_f, phi_i, phi_v, dx, bx)

            if outdir is None:
                rd = indir[:-7] if indir.endswith('.tar.gz') else indir
                outdir = str(Path(rd) / 'tissue_descriptors')
            Path(outdir).mkdir(parents=True, exist_ok=True)

            # Save scalars
            scalar_keys = [
                'bv_tv', 'porosity', 'surface_density', 'specific_surface',
                'tb_th', 'tb_sp', 'tb_n',
                'euler_characteristic', 'connectivity_density',
                'smi', 'correlation_length',
                'mean_chord_tissue', 'mean_chord_pore',
                'da', 'fa', 'tortuosity',
                'mean_pore_radius', 'median_pore_radius',
                'permeability_KC',
            ]
            scalar_desc = {}
            for k in scalar_keys:
                if k in desc:
                    v = desc[k]
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        scalar_desc[k] = float(v)

            json_path = Path(outdir) / 'descriptors.json'
            with open(json_path, 'w') as f:
                json.dump(scalar_desc, f, indent=2)

            # Print summary
            print(f"\n  Snapshot {args.snap}:")
            for k, v in sorted(scalar_desc.items()):
                print(f"    {k:30s} = {v:.6g}")

            # Plots
            plot_descriptor_summary(desc, outdir=outdir)
            plot_thickness_distribution(desc, outdir=outdir)
            if 'S2_r' in desc:
                plot_correlation_function(
                    desc['S2_r'], desc['S2_values'],
                    correlation_length=desc.get('correlation_length'),
                    outdir=outdir)
            if 'psd_bin_centers' in desc:
                plot_pore_size_distribution(desc, outdir=outdir)

            print(f"\n  Outputs saved to {outdir}")
