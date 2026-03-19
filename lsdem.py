"""
Level-Set Discrete Element Method (LS-DEM) for Deformable Particles
====================================================================
V2.3 — Henzel & Karapiperis 2026 variational formulation.

This module implements deformable particle simulation via:
  1. Signed distance field (SDF) representation per particle
  2. Surface node discretization for contact detection
  3. Modal deformation DOFs with elastic stiffness
  4. Semi-Lagrangian SDF update under deformation
  5. Implicit Euler integration for overdamped deformation dynamics

All functions are designed to be called from new_dem_0.py when
Params.deformable_enabled = True.

Units: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
"""

import numpy as np
from scipy.optimize import brentq

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


# ══════════════════════════════════════════════════════════════════════
# Signed Distance Field (SDF) Computation
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _superellipse_implicit_body(bx, by, a, b, n):
    """Superellipse implicit value in body frame. <1 inside, =1 boundary, >1 outside."""
    return (np.abs(bx / a))**n + (np.abs(by / b))**n


@njit(cache=True)
def _superellipse_grad_body(bx, by, a, b, n):
    """Gradient of the superellipse implicit function in body frame."""
    eps = 1e-30
    gx = (n / a) * np.sign(bx) * (np.abs(bx / a) + eps)**(n - 1.0)
    gy = (n / b) * np.sign(by) * (np.abs(by / b) + eps)**(n - 1.0)
    return gx, gy


@njit(cache=True)
def superellipse_approx_sdf(bx, by, a, b, n):
    """
    Approximate signed distance from a superellipse surface in body frame.

    Uses gradient-normalized implicit function:
        phi(x) ≈ (f(x) - 1) / |grad f(x)|

    Returns: negative inside, zero on surface, positive outside.
    """
    f = _superellipse_implicit_body(bx, by, a, b, n)
    gx, gy = _superellipse_grad_body(bx, by, a, b, n)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    if grad_mag < 1e-30:
        # At origin or degenerate — use f^(1/n) * min_axis as proxy
        return (f**(1.0 / n) - 1.0) * min(a, b)
    return (f - 1.0) / grad_mag


@njit(cache=True)
def _sgnpow(v, e):
    """Signed power: sign(v) * |v|^e, safe for v=0."""
    return np.sign(v) * (np.abs(v) + 1e-30)**e


@njit(cache=True)
def _superellipsoid_implicit_body(bx, by, bz, a, b, c, n1, n2):
    """Superellipsoid implicit value in body frame."""
    return ((np.abs(bx / a))**n1 + (np.abs(by / b))**n1)**(n2 / n1) + \
           (np.abs(bz / c))**n2


@njit(cache=True)
def _superellipsoid_grad_body(bx, by, bz, a, b, c, n1, n2):
    """Gradient of the superellipsoid implicit function in body frame."""
    eps = 1e-30
    abx = np.abs(bx / a) + eps
    aby = np.abs(by / b) + eps
    abz = np.abs(bz / c) + eps

    inner = abx**n1 + aby**n1
    inner_eps = inner + eps

    # df/dx = (n2/n1) * inner^(n2/n1 - 1) * n1/a * |x/a|^(n1-1) * sign(x/a)
    gx = (n2 / a) * inner_eps**(n2 / n1 - 1.0) * abx**(n1 - 1.0) * np.sign(bx)
    gy = (n2 / b) * inner_eps**(n2 / n1 - 1.0) * aby**(n1 - 1.0) * np.sign(by)
    gz = (n2 / c) * abz**(n2 - 1.0) * np.sign(bz)
    return gx, gy, gz


@njit(cache=True)
def superellipsoid_approx_sdf(bx, by, bz, a, b, c, n1, n2):
    """
    Approximate signed distance from a superellipsoid surface in body frame.
    Returns: negative inside, zero on surface, positive outside.
    """
    f = _superellipsoid_implicit_body(bx, by, bz, a, b, c, n1, n2)
    gx, gy, gz = _superellipsoid_grad_body(bx, by, bz, a, b, c, n1, n2)
    grad_mag = np.sqrt(gx * gx + gy * gy + gz * gz)
    if grad_mag < 1e-30:
        return (f**(1.0 / n2) - 1.0) * min(a, min(b, c))
    return (f - 1.0) / grad_mag


# ══════════════════════════════════════════════════════════════════════
# SDF Grid Precomputation
# ══════════════════════════════════════════════════════════════════════

def init_sdf_grids(gs, p):
    """
    Pre-compute SDF grids for all particles in body frame.

    Each particle gets a regular grid covering [-pad*R, +pad*R] per axis,
    where R = max semi-axis and pad = p.sdf_padding.

    Sets: gs.sdf_grids (list), gs.sdf_extents (N, ndim, 2)
    """
    N = gs.N
    res = p.sdf_resolution
    pad = p.sdf_padding
    ndim = 3 if gs.is_3d else 2

    gs.sdf_grids = []
    gs.sdf_extents = np.zeros((N, ndim, 2))

    for i in range(N):
        if ndim == 2:
            ai, bi, ni = gs.a[i], gs.b[i], gs.n_shape[i]
            extent = pad * max(ai, bi)
            lo, hi = -extent, extent
            gs.sdf_extents[i, 0] = [lo, hi]
            gs.sdf_extents[i, 1] = [lo, hi]

            # Build 2D SDF grid
            grid = np.empty((res, res), dtype=np.float64)
            xs = np.linspace(lo, hi, res)
            ys = np.linspace(lo, hi, res)
            for ix in range(res):
                for iy in range(res):
                    grid[ix, iy] = superellipse_approx_sdf(
                        xs[ix], ys[iy], ai, bi, ni)
            gs.sdf_grids.append(grid)
        else:
            ai, bi, ci = gs.a[i], gs.b[i], gs.c[i]
            n1i, n2i = gs.n1[i], gs.n2[i]
            extent = pad * max(ai, max(bi, ci))
            lo, hi = -extent, extent
            gs.sdf_extents[i, 0] = [lo, hi]
            gs.sdf_extents[i, 1] = [lo, hi]
            gs.sdf_extents[i, 2] = [lo, hi]

            # Build 3D SDF grid
            grid = np.empty((res, res, res), dtype=np.float64)
            xs = np.linspace(lo, hi, res)
            ys = np.linspace(lo, hi, res)
            zs = np.linspace(lo, hi, res)
            for ix in range(res):
                for iy in range(res):
                    for iz in range(res):
                        grid[ix, iy, iz] = superellipsoid_approx_sdf(
                            xs[ix], ys[iy], zs[iz], ai, bi, ci, n1i, n2i)
            gs.sdf_grids.append(grid)


# ══════════════════════════════════════════════════════════════════════
# SDF Query (Interpolation)
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx, by):
    """
    Bilinear interpolation of a 2D SDF grid at body-frame point (bx, by).
    Returns signed distance (negative inside, positive outside).
    """
    # Map to grid indices
    fx = (bx - lo_x) / (hi_x - lo_x) * (res - 1)
    fy = (by - lo_y) / (hi_y - lo_y) * (res - 1)

    # Clamp to grid bounds
    fx = max(0.0, min(fx, res - 1.001))
    fy = max(0.0, min(fy, res - 1.001))

    ix = int(fx)
    iy = int(fy)
    ix1 = min(ix + 1, res - 1)
    iy1 = min(iy + 1, res - 1)
    sx = fx - ix
    sy = fy - iy

    # Bilinear interpolation
    v00 = sdf_grid[ix, iy]
    v10 = sdf_grid[ix1, iy]
    v01 = sdf_grid[ix, iy1]
    v11 = sdf_grid[ix1, iy1]

    return (v00 * (1 - sx) * (1 - sy) +
            v10 * sx * (1 - sy) +
            v01 * (1 - sx) * sy +
            v11 * sx * sy)


@njit(cache=True)
def query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res,
                 bx, by, bz):
    """
    Trilinear interpolation of a 3D SDF grid at body-frame point (bx, by, bz).
    """
    fx = (bx - lo_x) / (hi_x - lo_x) * (res - 1)
    fy = (by - lo_y) / (hi_y - lo_y) * (res - 1)
    fz = (bz - lo_z) / (hi_z - lo_z) * (res - 1)

    fx = max(0.0, min(fx, res - 1.001))
    fy = max(0.0, min(fy, res - 1.001))
    fz = max(0.0, min(fz, res - 1.001))

    ix = int(fx); iy = int(fy); iz = int(fz)
    ix1 = min(ix + 1, res - 1)
    iy1 = min(iy + 1, res - 1)
    iz1 = min(iz + 1, res - 1)
    sx = fx - ix; sy = fy - iy; sz = fz - iz

    v000 = sdf_grid[ix, iy, iz]
    v100 = sdf_grid[ix1, iy, iz]
    v010 = sdf_grid[ix, iy1, iz]
    v110 = sdf_grid[ix1, iy1, iz]
    v001 = sdf_grid[ix, iy, iz1]
    v101 = sdf_grid[ix1, iy, iz1]
    v011 = sdf_grid[ix, iy1, iz1]
    v111 = sdf_grid[ix1, iy1, iz1]

    return (v000 * (1-sx) * (1-sy) * (1-sz) +
            v100 * sx * (1-sy) * (1-sz) +
            v010 * (1-sx) * sy * (1-sz) +
            v110 * sx * sy * (1-sz) +
            v001 * (1-sx) * (1-sy) * sz +
            v101 * sx * (1-sy) * sz +
            v011 * (1-sx) * sy * sz +
            v111 * sx * sy * sz)


@njit(cache=True)
def query_sdf_gradient_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx, by):
    """
    SDF gradient at body-frame point via central finite differences on the grid.
    Returns (gx, gy) — unnormalized gradient.
    """
    h = (hi_x - lo_x) / (res - 1)
    gx = (query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx + h, by) -
          query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx - h, by)) / (2.0 * h)
    gy = (query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx, by + h) -
          query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx, by - h)) / (2.0 * h)
    return gx, gy


@njit(cache=True)
def query_sdf_gradient_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res,
                          bx, by, bz):
    """
    SDF gradient at body-frame point via central finite differences.
    Returns (gx, gy, gz) — unnormalized gradient.
    """
    h = (hi_x - lo_x) / (res - 1)
    gx = (query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx + h, by, bz) -
          query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx - h, by, bz)) / (2.0 * h)
    gy = (query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx, by + h, bz) -
          query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx, by - h, bz)) / (2.0 * h)
    gz = (query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx, by, bz + h) -
          query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res, bx, by, bz - h)) / (2.0 * h)
    return gx, gy, gz


# ══════════════════════════════════════════════════════════════════════
# Semi-Lagrangian SDF query (deformed geometry)
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _mode_displacement_2d(bx, by, a, b, nu, n_modes):
    """
    Evaluate mode shape displacement vectors at body-frame point (bx, by).

    Mode shapes produce DISPLACEMENT in µm (not dimensionless).
    This is critical: u(x) = eps * Phi(x) must have units of µm so that the
    semi-Lagrangian SDF update (x_ref = x - u) is dimensionally correct.

    Mode 0: Axial compression (volume-preserving deviatoric)
        Phi_0 = (-nu*x, y)  -- compress along y, expand along x
    Mode 1: Prolate-oblate (in-plane shape change)
        Phi_1 = (x, -y)     -- stretch x, compress y

    Returns: (n_modes, 2) array of displacement vectors [µm].
    """
    modes = np.zeros((n_modes, 2))
    # Mode 0: axial compression along y, lateral bulge along x
    modes[0, 0] = -nu * bx
    modes[0, 1] = by
    # Mode 1: prolate-oblate
    if n_modes > 1:
        modes[1, 0] = bx
        modes[1, 1] = -by
    return modes


@njit(cache=True)
def _mode_displacement_3d(bx, by, bz, a, b, c, nu, n_modes):
    """
    Evaluate mode shape displacement vectors at body-frame point (bx, by, bz).

    Mode shapes produce DISPLACEMENT in µm.

    Mode 0: Axial compression (flatten along z, bulge laterally)
        Phi_0 = (-nu*x, -nu*y, z)
    Mode 1: Prolate-oblate (xy-plane shape change)
        Phi_1 = (x, -y, 0)
    Mode 2: Volumetric (uniform expansion — very stiff for nu~0.49)
        Phi_2 = (x, y, z)

    Returns: (n_modes, 3) array [µm].
    """
    modes = np.zeros((n_modes, 3))
    modes[0, 0] = -nu * bx
    modes[0, 1] = -nu * by
    modes[0, 2] = bz
    if n_modes > 1:
        modes[1, 0] = bx
        modes[1, 1] = -by
        modes[1, 2] = 0.0
    if n_modes > 2:
        modes[2, 0] = bx
        modes[2, 1] = by
        modes[2, 2] = bz
    return modes


@njit(cache=True)
def query_deformed_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res,
                          bx, by, epsilon, a, b, nu, n_modes):
    """
    Semi-Lagrangian evaluation of deformed SDF at body-frame point.

    Pull back to reference configuration: x_ref = x - u(x)
    where u(x) = sum_alpha epsilon_alpha * Phi_alpha(x),
    then query the undeformed SDF at x_ref.
    """
    modes = _mode_displacement_2d(bx, by, a, b, nu, n_modes)
    ux, uy = 0.0, 0.0
    for alpha in range(n_modes):
        ux += epsilon[alpha] * modes[alpha, 0]
        uy += epsilon[alpha] * modes[alpha, 1]
    bx_ref = bx - ux
    by_ref = by - uy
    return query_sdf_2d(sdf_grid, lo_x, hi_x, lo_y, hi_y, res, bx_ref, by_ref)


@njit(cache=True)
def query_deformed_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res,
                          bx, by, bz, epsilon, a, b, c, nu, n_modes):
    """Semi-Lagrangian evaluation of deformed SDF at 3D body-frame point."""
    modes = _mode_displacement_3d(bx, by, bz, a, b, c, nu, n_modes)
    ux, uy, uz = 0.0, 0.0, 0.0
    for alpha in range(n_modes):
        ux += epsilon[alpha] * modes[alpha, 0]
        uy += epsilon[alpha] * modes[alpha, 1]
        uz += epsilon[alpha] * modes[alpha, 2]
    bx_ref = bx - ux
    by_ref = by - uy
    bz_ref = bz - uz
    return query_sdf_3d(sdf_grid, lo_x, hi_x, lo_y, hi_y, lo_z, hi_z, res,
                        bx_ref, by_ref, bz_ref)


# ══════════════════════════════════════════════════════════════════════
# Surface Node Discretization
# ══════════════════════════════════════════════════════════════════════

def init_surface_nodes_2d(gs, p):
    """
    Place surface nodes on each superellipse boundary (2D).

    Uses uniform parametric sampling t in [0, 2*pi). Computes:
      - Node positions in body frame
      - Outward normal at each node
      - Arc-length weight (area element for force weighting)

    Sets:
      gs.surface_nodes_body: (N, n_nodes, 2)
      gs.surface_normals_body: (N, n_nodes, 2)
      gs.surface_node_areas: (N, n_nodes)
    """
    N = gs.N
    n_nodes = p.n_surface_nodes
    gs.surface_nodes_body = np.zeros((N, n_nodes, 2))
    gs.surface_normals_body = np.zeros((N, n_nodes, 2))
    gs.surface_node_areas = np.zeros((N, n_nodes))

    ts = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    dt = 2 * np.pi / n_nodes

    for i in range(N):
        ai, bi, ni = gs.a[i], gs.b[i], gs.n_shape[i]
        e = 2.0 / ni

        for k in range(n_nodes):
            t = ts[k]
            ct, st = np.cos(t), np.sin(t)

            # Surface point (body frame)
            px = ai * np.sign(ct) * np.abs(ct)**e
            py = bi * np.sign(st) * np.abs(st)**e
            gs.surface_nodes_body[i, k, 0] = px
            gs.surface_nodes_body[i, k, 1] = py

            # Tangent vector for normal and arc length
            dxdt = -ai * e * np.sign(ct) * (np.abs(ct) + 1e-30)**(e - 1.0) * st
            dydt = bi * e * np.sign(st) * (np.abs(st) + 1e-30)**(e - 1.0) * ct

            # Outward normal = (dydt, -dxdt) normalized
            nx, ny = dydt, -dxdt
            mag = np.sqrt(nx * nx + ny * ny)
            if mag > 1e-30:
                nx /= mag
                ny /= mag
            gs.surface_normals_body[i, k, 0] = nx
            gs.surface_normals_body[i, k, 1] = ny

            # Arc length element: |dr/dt| * dt
            ds = np.sqrt(dxdt * dxdt + dydt * dydt) * dt
            gs.surface_node_areas[i, k] = ds


def init_surface_nodes_3d(gs, p):
    """
    Place surface nodes on each superellipsoid surface (3D).

    Uses Fibonacci sphere sampling to generate approximately uniform
    directions, then projects each onto the superellipsoid surface.

    Sets:
      gs.surface_nodes_body: (N, n_nodes, 3)
      gs.surface_normals_body: (N, n_nodes, 3)
      gs.surface_node_areas: (N, n_nodes)
    """
    N = gs.N
    n_nodes = p.n_surface_nodes_3d
    gs.surface_nodes_body = np.zeros((N, n_nodes, 3))
    gs.surface_normals_body = np.zeros((N, n_nodes, 3))
    gs.surface_node_areas = np.zeros((N, n_nodes))

    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0

    # Generate Fibonacci sphere directions
    directions = np.zeros((n_nodes, 3))
    for k in range(n_nodes):
        theta = np.arccos(1.0 - 2.0 * (k + 0.5) / n_nodes)
        phi = 2.0 * np.pi * k / golden_ratio
        directions[k, 0] = np.sin(theta) * np.cos(phi)
        directions[k, 1] = np.sin(theta) * np.sin(phi)
        directions[k, 2] = np.cos(theta)

    for i in range(N):
        ai, bi, ci = gs.a[i], gs.b[i], gs.c[i]
        n1i, n2i = gs.n1[i], gs.n2[i]

        # Approximate total surface area for area weighting
        # (use sphere approximation scaled by semi-axes)
        r_eq = (ai * bi * ci)**(1.0 / 3.0)
        total_area_approx = 4.0 * np.pi * r_eq * r_eq
        area_per_node = total_area_approx / n_nodes

        for k in range(n_nodes):
            dx, dy, dz = directions[k]

            # Project direction onto superellipsoid: find s such that
            # f(s*dx, s*dy, s*dz) = 1
            # Use analytical formula for superellipsoid ray intersection
            # For the implicit function: (|sx*dx/a|^n1+|s*dy/b|^n1)^(n2/n1)+|s*dz/c|^n2=1
            # Start with sphere estimate and refine
            s_guess = 1.0 / np.sqrt((dx / ai)**2 + (dy / bi)**2 + (dz / ci)**2)

            # Newton-Raphson refinement
            s = s_guess
            for _iter in range(20):
                bx, by, bz = s * dx, s * dy, s * dz
                f = _superellipsoid_implicit_body(bx, by, bz, ai, bi, ci, n1i, n2i)
                if abs(f - 1.0) < 1e-10:
                    break
                gx, gy, gz = _superellipsoid_grad_body(bx, by, bz, ai, bi, ci, n1i, n2i)
                df_ds = gx * dx + gy * dy + gz * dz
                if abs(df_ds) < 1e-30:
                    break
                s -= (f - 1.0) / df_ds
                s = max(s, 1e-6)

            px, py, pz = s * dx, s * dy, s * dz
            gs.surface_nodes_body[i, k, 0] = px
            gs.surface_nodes_body[i, k, 1] = py
            gs.surface_nodes_body[i, k, 2] = pz

            # Outward normal from implicit gradient
            gx, gy, gz = _superellipsoid_grad_body(px, py, pz, ai, bi, ci, n1i, n2i)
            mag = np.sqrt(gx * gx + gy * gy + gz * gz)
            if mag > 1e-30:
                gx /= mag; gy /= mag; gz /= mag
            gs.surface_normals_body[i, k, 0] = gx
            gs.surface_normals_body[i, k, 1] = gy
            gs.surface_normals_body[i, k, 2] = gz

            gs.surface_node_areas[i, k] = area_per_node


# ══════════════════════════════════════════════════════════════════════
# Deformation Modes + Stiffness
# ══════════════════════════════════════════════════════════════════════

def compute_mode_stiffness_2d(a, b, n_shape, E_kPa, nu, n_modes):
    """
    Compute the deformation stiffness matrix K (n_modes x n_modes) for a 2D
    superellipse with DIMENSIONED mode shapes (Phi in µm).

    Mode shapes produce displacement u = eps * Phi(x) [µm], where Phi(x) is
    in µm. The strain from mode alpha is eps_ij = eps * d(Phi_i)/d(x_j),
    which is DIMENSIONLESS (correct).

    Stiffness K_ab = integral of C_ijkl * strain(Phi_a)_ij * strain(Phi_b)_kl dA
    Units: Pa * µm² = 1e-3 nN·µm (energy). K * eps = F_eps [nN·µm].

    Mode 0: Phi = (-nu*x, y) → strain: e_xx = -nu, e_yy = 1 (deviatoric for nu=0.5)
    Mode 1: Phi = (x, -y)   → strain: e_xx = 1, e_yy = -1 (pure deviatoric)

    Args:
        a, b: semi-axes (µm)
        n_shape: blockiness exponent
        E_kPa: Young's modulus (kPa)
        nu: Poisson's ratio
        n_modes: number of modes

    Returns: K (n_modes, n_modes) in nN·µm (energy) units
    """
    from scipy.special import gamma as _gamma_func
    e = 1.0 / n_shape
    A = 4.0 * a * b * (_gamma_func(1.0 + e)**2) / _gamma_func(1.0 + 2.0 * e)

    E_Pa = E_kPa * 1e3  # kPa -> Pa

    # Plane stress stiffness: C = E/(1-nu^2) * [[1,nu,0],[nu,1,0],[0,0,(1-nu)/2]]
    C11 = E_Pa / (1.0 - nu * nu)
    C12 = nu * C11
    C66 = E_Pa / (2.0 * (1.0 + nu))  # shear

    K = np.zeros((n_modes, n_modes))

    # Mode 0: strain = (-nu, 1, 0)
    # w = (1/2)[C11*nu^2 - 2*C12*nu + C11*1] = (1/2)*C11*(1+nu^2) - C12*nu
    #   = (1/2)*E/(1-nu^2) * (1+nu^2) - nu^2*E/(1-nu^2)
    #   = E/(2(1-nu^2)) * (1+nu^2 - 2*nu^2) = E/(2(1-nu^2)) * (1-nu^2) = E/2
    K[0, 0] = E_Pa * A * 1e-3  # analytical: K = E * A (from 2*w*A = E*A)

    # Mode 1: strain = (1, -1, 0)
    # w = (1/2)[C11*1 - 2*C12 + C11*1] = C11 - C12 = E/(1-nu^2) * (1-nu) = E/(1+nu) = 2G
    # K = 2*w*A = 2*2G*A = 4G*A ... wait, K = 2w (double the density since integral)
    # Actually K = integral 2w dA (from U = (1/2) K eps^2, and U = w*A*eps^2)
    # So K = 2*w_coeff * A where w = w_coeff * eps^2
    # Mode 1: w_coeff = 2G, so K = 4G*A
    G = E_Pa / (2.0 * (1.0 + nu))
    if n_modes > 1:
        K[1, 1] = 4.0 * G * A * 1e-3

    # Cross terms K_01: strain_0 = (-nu, 1), strain_1 = (1, -1)
    # w_01 = C11*(-nu)*1 + C12*((-nu)*(-1) + 1*1) + C11*(1)*(-1)
    # = C11*(-nu - 1) + C12*(nu + 1) = (C12 - C11)*(1+nu) = -C66*2*(1+nu)?
    # Actually: w_01 = sigma_0_ij * eps_1_ij
    # sigma_0: s_xx = C11*(-nu)+C12*1 = C11*(-nu+nu) = 0, s_yy = C12*(-nu)+C11*1 = E_Pa/(1-nu^2)*(1-nu^2) = E_Pa
    # w_01 = s_xx*1 + s_yy*(-1) = 0 - E_Pa = -E_Pa
    # But wait, this means K_01 = -E_Pa * A * 1e-3, the modes are NOT orthogonal!
    # For nu = 0.49 this is significant.
    if n_modes > 1:
        K[0, 1] = -E_Pa * A * 1e-3
        K[1, 0] = K[0, 1]

    return K


def compute_mode_stiffness_3d(a, b, c, n1, n2, E_kPa, nu, n_modes):
    """
    Compute the deformation stiffness matrix K (n_modes x n_modes) for a 3D
    superellipsoid with DIMENSIONED mode shapes (Phi in µm).

    Mode 0: Phi = (-nu*x, -nu*y, z) → strain: e_xx=e_yy=-nu, e_zz=1
    Mode 1: Phi = (x, -y, 0) → strain: e_xx=1, e_yy=-1, e_zz=0
    Mode 2: Phi = (x, y, z) → strain: e_xx=e_yy=e_zz=1 (volumetric)

    Returns: K (n_modes, n_modes) in nN·µm (energy) units
    """
    from scipy.special import gamma as _gamma_func, beta as _beta_func

    # Superellipsoid volume
    e1 = 2.0 / n1
    e2 = 2.0 / n2
    V = 2.0 * a * b * c * e1 * e2 * \
        _beta_func(e1 / 2.0 + 1.0, e1) * _beta_func(e2 / 2.0, e2 / 2.0)

    E_Pa = E_kPa * 1e3
    G = E_Pa / (2.0 * (1.0 + nu))
    K_bulk_mod = E_Pa / (3.0 * (1.0 - 2.0 * nu))

    # Isotropic stiffness tensor components
    lam = E_Pa * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # Lame lambda
    mu = G  # Lame mu = shear modulus

    K = np.zeros((n_modes, n_modes))

    # Compute K_ab = V * sum_ijkl C_ijkl * eps_a_ij * eps_b_kl
    # (uniform strain → integral = value * V)
    # C_ijkl = lam * d_ij*d_kl + mu * (d_ik*d_jl + d_il*d_jk)
    # For diagonal strains: w = lam*(tr_eps)^2/2 + mu*sum(eps_ii^2)

    # Mode 0: eps = (-nu, -nu, 1)
    tr0 = -nu - nu + 1.0
    sq0 = nu**2 + nu**2 + 1.0
    w00 = lam * tr0**2 + 2.0 * mu * sq0
    K[0, 0] = w00 * V * 1e-3

    if n_modes > 1:
        # Mode 1: eps = (1, -1, 0)
        tr1 = 0.0
        sq1 = 1.0 + 1.0
        w11 = lam * tr1**2 + 2.0 * mu * sq1
        K[1, 1] = w11 * V * 1e-3

        # Cross K_01
        tr01 = tr0 * tr1  # = 0
        cross01 = (-nu) * 1.0 + (-nu) * (-1.0) + 1.0 * 0.0  # = 0
        w01 = lam * tr01 + 2.0 * mu * cross01
        K[0, 1] = w01 * V * 1e-3
        K[1, 0] = K[0, 1]

    if n_modes > 2:
        # Mode 2: eps = (1, 1, 1)
        tr2 = 3.0
        sq2 = 3.0
        w22 = lam * tr2**2 + 2.0 * mu * sq2
        K[2, 2] = w22 * V * 1e-3

        # Cross K_02
        tr02 = tr0 * 3.0
        cross02 = (-nu) * 1.0 + (-nu) * 1.0 + 1.0 * 1.0
        w02 = lam * tr02 + 2.0 * mu * cross02
        K[0, 2] = w02 * V * 1e-3
        K[2, 0] = K[0, 2]

        # Cross K_12
        cross12 = 1.0 * 1.0 + (-1.0) * 1.0 + 0.0 * 1.0  # = 0
        w12 = lam * 0.0 + 2.0 * mu * cross12
        K[1, 2] = w12 * V * 1e-3
        K[2, 1] = K[1, 2]

    return K


def init_mode_shapes(gs, p):
    """
    Pre-compute mode shape displacement vectors at each surface node,
    and the elastic stiffness matrix K for each particle.

    Sets:
      gs.mode_shapes_at_nodes: (N, n_nodes, n_modes, ndim)
      gs.K_def: (N, n_modes, n_modes)
    """
    N = gs.N
    n_modes = p.n_def_modes
    nu = p.poisson_ratio
    ndim = 3 if gs.is_3d else 2
    n_nodes = p.n_surface_nodes_3d if gs.is_3d else p.n_surface_nodes

    gs.mode_shapes_at_nodes = np.zeros((N, n_nodes, n_modes, ndim))
    gs.K_def = np.zeros((N, n_modes, n_modes))

    for i in range(N):
        # Compute mode shapes at each surface node
        for k in range(n_nodes):
            if ndim == 2:
                bx = gs.surface_nodes_body[i, k, 0]
                by = gs.surface_nodes_body[i, k, 1]
                modes = _mode_displacement_2d(bx, by, gs.a[i], gs.b[i], nu, n_modes)
                gs.mode_shapes_at_nodes[i, k, :, :] = modes[:n_modes, :ndim]
            else:
                bx = gs.surface_nodes_body[i, k, 0]
                by = gs.surface_nodes_body[i, k, 1]
                bz = gs.surface_nodes_body[i, k, 2]
                modes = _mode_displacement_3d(bx, by, bz, gs.a[i], gs.b[i], gs.c[i],
                                              nu, n_modes)
                gs.mode_shapes_at_nodes[i, k, :, :] = modes[:n_modes, :ndim]

        # Compute stiffness matrix
        if ndim == 2:
            gs.K_def[i] = compute_mode_stiffness_2d(
                gs.a[i], gs.b[i], gs.n_shape[i], p.E_modulus, nu, n_modes)
        else:
            gs.K_def[i] = compute_mode_stiffness_3d(
                gs.a[i], gs.b[i], gs.c[i], gs.n1[i], gs.n2[i],
                p.E_modulus, nu, n_modes)


# ══════════════════════════════════════════════════════════════════════
# LS-DEM Contact Detection
# ══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _world_to_body_2d(px, py, cx, cy, cos_th, sin_th):
    """Transform world point to body frame."""
    dx, dy = px - cx, py - cy
    return cos_th * dx + sin_th * dy, -sin_th * dx + cos_th * dy


@njit(cache=True)
def _body_to_world_2d(bx, by, cx, cy, cos_th, sin_th):
    """Transform body point to world frame."""
    return cx + cos_th * bx - sin_th * by, cy + sin_th * bx + cos_th * by


def find_contacts_lsdem_2d(gs, i, j, xi, yi, xj, yj, p):
    """
    LS-DEM contact detection between two 2D particles.

    Tests surface nodes of particle i against SDF of particle j,
    and vice versa. Accumulates forces, torques, and deformation forces.

    Args:
        gs: GranuleSystem
        i, j: particle indices
        xi, yi: position of particle i (may include periodic offset)
        xj, yj: position of particle j
        p: Params

    Returns:
        dict with keys: 'in_contact', 'overlap_max', 'F_i', 'F_j' (2D force),
        'tau_i', 'tau_j' (scalar torque), 'F_eps_i', 'F_eps_j' (n_modes),
        'contact_nodes' (list of node contacts for viz),
        or None if no contact.
    """
    n_modes = p.n_def_modes
    nu = p.poisson_ratio
    res = p.sdf_resolution

    # Effective modulus for penalty force
    E_Pa = p.E_modulus * 1e3
    nu2 = nu * nu
    E_star = E_Pa / (2.0 * (1.0 - nu2))

    # Get SDF grid and extents for both particles
    sdf_j = gs.sdf_grids[j]
    lo_jx, hi_jx = gs.sdf_extents[j, 0]
    lo_jy, hi_jy = gs.sdf_extents[j, 1]

    sdf_i = gs.sdf_grids[i]
    lo_ix, hi_ix = gs.sdf_extents[i, 0]
    lo_iy, hi_iy = gs.sdf_extents[i, 1]

    # Trig for rotations
    cos_i, sin_i = np.cos(gs.theta[i]), np.sin(gs.theta[i])
    cos_j, sin_j = np.cos(gs.theta[j]), np.sin(gs.theta[j])

    # Deformation state
    eps_i = gs.epsilon[i] if gs.epsilon is not None else np.zeros(n_modes)
    eps_j = gs.epsilon[j] if gs.epsilon is not None else np.zeros(n_modes)
    has_deform = gs.epsilon is not None and (np.any(eps_i != 0) or np.any(eps_j != 0))

    # Centre-to-centre direction for hemisphere culling
    dx_c = xj - xi
    dy_c = yj - yi
    d_c = np.sqrt(dx_c * dx_c + dy_c * dy_c)
    if d_c < 1e-12:
        return None
    ux_c, uy_c = dx_c / d_c, dy_c / d_c

    # Accumulators
    F_i = np.zeros(2)
    F_j = np.zeros(2)
    tau_i = 0.0
    tau_j = 0.0
    F_eps_i = np.zeros(n_modes)
    F_eps_j = np.zeros(n_modes)
    overlap_max = 0.0
    n_contacts = 0
    contact_nodes = []

    n_nodes = p.n_surface_nodes

    # ── Test nodes of i against SDF of j ──
    for k in range(n_nodes):
        # Body-frame node position (with deformation)
        bx_i = gs.surface_nodes_body[i, k, 0]
        by_i = gs.surface_nodes_body[i, k, 1]
        if has_deform:
            # Apply deformation: x_deformed = x_ref + u(x_ref)
            for alpha in range(n_modes):
                bx_i += eps_i[alpha] * gs.mode_shapes_at_nodes[i, k, alpha, 0]
                by_i += eps_i[alpha] * gs.mode_shapes_at_nodes[i, k, alpha, 1]

        # Body-frame normal for hemisphere culling
        nx_body = gs.surface_normals_body[i, k, 0]
        ny_body = gs.surface_normals_body[i, k, 1]
        # Transform normal to world for culling
        nx_w = cos_i * nx_body - sin_i * ny_body
        ny_w = sin_i * nx_body + cos_i * ny_body
        # Cull: skip nodes facing away from j
        if nx_w * ux_c + ny_w * uy_c < -0.1:
            continue

        # Transform node to world frame
        wx = xi + cos_i * bx_i - sin_i * by_i
        wy = yi + sin_i * bx_i + cos_i * by_i

        # Transform to body frame of j
        dx_j, dy_j = wx - xj, wy - yj
        bx_j = cos_j * dx_j + sin_j * dy_j
        by_j = -sin_j * dx_j + cos_j * dy_j

        # Query SDF of j (with deformation if active)
        if has_deform and np.any(eps_j != 0):
            sdf_val = query_deformed_sdf_2d(
                sdf_j, lo_jx, hi_jx, lo_jy, hi_jy, res,
                bx_j, by_j, eps_j, gs.a[j], gs.b[j], nu, n_modes)
        else:
            sdf_val = query_sdf_2d(sdf_j, lo_jx, hi_jx, lo_jy, hi_jy, res,
                                   bx_j, by_j)

        if sdf_val >= 0:
            continue  # not penetrating

        # Contact normal from SDF gradient (in j's body frame)
        gnx, gny = query_sdf_gradient_2d(sdf_j, lo_jx, hi_jx, lo_jy, hi_jy,
                                         res, bx_j, by_j)
        g_mag = np.sqrt(gnx * gnx + gny * gny)
        if g_mag < 1e-20:
            continue
        # Normal points outward from j (direction to push i out)
        nnx_b = gnx / g_mag
        nny_b = gny / g_mag

        # Transform normal to world frame
        nnx_w = cos_j * nnx_b - sin_j * nny_b
        nny_w = sin_j * nnx_b + cos_j * nny_b

        area_k = gs.surface_node_areas[i, k]
        delta_k = -sdf_val
        if delta_k > overlap_max:
            overlap_max = delta_k

        f_mag = E_star * delta_k * area_k * 1e-3  # Pa*µm*µm -> nN

        # Force on node (pushes i out of j)
        fx = f_mag * nnx_w
        fy = f_mag * nny_w

        F_i[0] += fx
        F_i[1] += fy
        F_j[0] -= fx
        F_j[1] -= fy

        # Torques
        rx_i = wx - xi
        ry_i = wy - yi
        tau_i += rx_i * fy - ry_i * fx

        rx_j = wx - xj
        ry_j = wy - yj
        tau_j -= (rx_j * fy - ry_j * fx)

        # Deformation generalized forces
        if gs.epsilon is not None:
            # Force in body frame of i (node owner)
            fx_bi = cos_i * fx + sin_i * fy
            fy_bi = -sin_i * fx + cos_i * fy
            for alpha in range(n_modes):
                phi_x = gs.mode_shapes_at_nodes[i, k, alpha, 0]
                phi_y = gs.mode_shapes_at_nodes[i, k, alpha, 1]
                F_eps_i[alpha] += phi_x * fx_bi + phi_y * fy_bi

            # Force in body frame of j (SDF owner) — at the query point
            fx_bj = cos_j * (-fx) + sin_j * (-fy)
            fy_bj = -sin_j * (-fx) + cos_j * (-fy)
            # Evaluate mode shapes at query point in j's body frame
            modes_j = _mode_displacement_2d(bx_j, by_j, gs.a[j], gs.b[j],
                                            nu, n_modes)
            for alpha in range(n_modes):
                F_eps_j[alpha] += modes_j[alpha, 0] * fx_bj + modes_j[alpha, 1] * fy_bj

        n_contacts += 1
        contact_nodes.append({
            'wx': wx, 'wy': wy,
            'delta': delta_k,
            'nx': nnx_w, 'ny': nny_w,
            'f_mag': abs(f_mag)
        })

    # ── Test nodes of j against SDF of i (symmetric) ──
    for k in range(n_nodes):
        bx_jk = gs.surface_nodes_body[j, k, 0]
        by_jk = gs.surface_nodes_body[j, k, 1]
        if has_deform:
            for alpha in range(n_modes):
                bx_jk += eps_j[alpha] * gs.mode_shapes_at_nodes[j, k, alpha, 0]
                by_jk += eps_j[alpha] * gs.mode_shapes_at_nodes[j, k, alpha, 1]

        # Hemisphere culling (opposite direction)
        nx_body = gs.surface_normals_body[j, k, 0]
        ny_body = gs.surface_normals_body[j, k, 1]
        nx_w = cos_j * nx_body - sin_j * ny_body
        ny_w = sin_j * nx_body + cos_j * ny_body
        if nx_w * (-ux_c) + ny_w * (-uy_c) < -0.1:
            continue

        # Transform to world
        wx = xj + cos_j * bx_jk - sin_j * by_jk
        wy = yj + sin_j * bx_jk + cos_j * by_jk

        # Transform to body frame of i
        dx_i, dy_i = wx - xi, wy - yi
        bx_iq = cos_i * dx_i + sin_i * dy_i
        by_iq = -sin_i * dx_i + cos_i * dy_i

        # Query SDF of i
        if has_deform and np.any(eps_i != 0):
            sdf_val = query_deformed_sdf_2d(
                sdf_i, lo_ix, hi_ix, lo_iy, hi_iy, res,
                bx_iq, by_iq, eps_i, gs.a[i], gs.b[i], nu, n_modes)
        else:
            sdf_val = query_sdf_2d(sdf_i, lo_ix, hi_ix, lo_iy, hi_iy, res,
                                   bx_iq, by_iq)

        if sdf_val >= 0:
            continue

        gnx, gny = query_sdf_gradient_2d(sdf_i, lo_ix, hi_ix, lo_iy, hi_iy,
                                         res, bx_iq, by_iq)
        g_mag = np.sqrt(gnx * gnx + gny * gny)
        if g_mag < 1e-20:
            continue
        nnx_b = gnx / g_mag
        nny_b = gny / g_mag
        nnx_w = cos_i * nnx_b - sin_i * nny_b
        nny_w = sin_i * nnx_b + cos_i * nny_b

        area_k = gs.surface_node_areas[j, k]
        delta_k = -sdf_val
        if delta_k > overlap_max:
            overlap_max = delta_k

        f_mag = E_star * delta_k * area_k * 1e-3

        # Force pushes j out of i (normal points out of i)
        fx = f_mag * nnx_w
        fy = f_mag * nny_w

        F_j[0] += fx
        F_j[1] += fy
        F_i[0] -= fx
        F_i[1] -= fy

        rx_j = wx - xj
        ry_j = wy - yj
        tau_j += rx_j * fy - ry_j * fx

        rx_i = wx - xi
        ry_i = wy - yi
        tau_i -= (rx_i * fy - ry_i * fx)

        if gs.epsilon is not None:
            fx_bj = cos_j * fx + sin_j * fy
            fy_bj = -sin_j * fx + cos_j * fy
            for alpha in range(n_modes):
                phi_x = gs.mode_shapes_at_nodes[j, k, alpha, 0]
                phi_y = gs.mode_shapes_at_nodes[j, k, alpha, 1]
                F_eps_j[alpha] += phi_x * fx_bj + phi_y * fy_bj

            fx_bi = cos_i * (-fx) + sin_i * (-fy)
            fy_bi = -sin_i * (-fx) + cos_i * (-fy)
            modes_i = _mode_displacement_2d(bx_iq, by_iq, gs.a[i], gs.b[i],
                                            nu, n_modes)
            for alpha in range(n_modes):
                F_eps_i[alpha] += modes_i[alpha, 0] * fx_bi + modes_i[alpha, 1] * fy_bi

        n_contacts += 1

    if n_contacts == 0:
        return None

    # Halve forces to avoid double-counting (both directions tested)
    F_i *= 0.5
    F_j *= 0.5
    tau_i *= 0.5
    tau_j *= 0.5
    F_eps_i *= 0.5
    F_eps_j *= 0.5

    return {
        'in_contact': True,
        'overlap_max': overlap_max,
        'n_contact_nodes': n_contacts,
        'F_i': F_i, 'F_j': F_j,
        'tau_i': tau_i, 'tau_j': tau_j,
        'F_eps_i': F_eps_i, 'F_eps_j': F_eps_j,
        'contact_nodes': contact_nodes,
        'i': i, 'j': j,
    }


def find_contacts_lsdem_3d(gs, i, j, xi, yi, zi, xj, yj, zj, p):
    """
    LS-DEM contact detection between two 3D particles.

    Same algorithm as 2D but with 3D quaternion transforms and
    trilinear SDF interpolation.

    Returns dict or None.
    """
    # Import quaternion helpers from the main module
    from new_dem_0 import quat_rotate, quat_rotate_inv

    n_modes = p.n_def_modes
    nu = p.poisson_ratio
    res = p.sdf_resolution

    E_Pa = p.E_modulus * 1e3
    nu2 = nu * nu
    E_star = E_Pa / (2.0 * (1.0 - nu2))

    sdf_j = gs.sdf_grids[j]
    lo_jx, hi_jx = gs.sdf_extents[j, 0]
    lo_jy, hi_jy = gs.sdf_extents[j, 1]
    lo_jz, hi_jz = gs.sdf_extents[j, 2]

    sdf_i = gs.sdf_grids[i]
    lo_ix, hi_ix = gs.sdf_extents[i, 0]
    lo_iy, hi_iy = gs.sdf_extents[i, 1]
    lo_iz, hi_iz = gs.sdf_extents[i, 2]

    qi = gs.quat[i]
    qj = gs.quat[j]

    eps_i = gs.epsilon[i] if gs.epsilon is not None else np.zeros(n_modes)
    eps_j = gs.epsilon[j] if gs.epsilon is not None else np.zeros(n_modes)
    has_deform = gs.epsilon is not None and (np.any(eps_i != 0) or np.any(eps_j != 0))

    dx_c = xj - xi; dy_c = yj - yi; dz_c = zj - zi
    d_c = np.sqrt(dx_c**2 + dy_c**2 + dz_c**2)
    if d_c < 1e-12:
        return None
    ux_c = dx_c / d_c; uy_c = dy_c / d_c; uz_c = dz_c / d_c
    dir_c = np.array([ux_c, uy_c, uz_c])

    F_i = np.zeros(3)
    F_j = np.zeros(3)
    tau_i = np.zeros(3)
    tau_j = np.zeros(3)
    F_eps_i = np.zeros(n_modes)
    F_eps_j = np.zeros(n_modes)
    overlap_max = 0.0
    n_contacts = 0

    n_nodes = p.n_surface_nodes_3d

    # ── Test nodes of i against SDF of j ──
    for k in range(n_nodes):
        bx_i = gs.surface_nodes_body[i, k, 0]
        by_i = gs.surface_nodes_body[i, k, 1]
        bz_i = gs.surface_nodes_body[i, k, 2]
        if has_deform:
            for alpha in range(n_modes):
                bx_i += eps_i[alpha] * gs.mode_shapes_at_nodes[i, k, alpha, 0]
                by_i += eps_i[alpha] * gs.mode_shapes_at_nodes[i, k, alpha, 1]
                bz_i += eps_i[alpha] * gs.mode_shapes_at_nodes[i, k, alpha, 2]

        # Hemisphere culling
        n_body = gs.surface_normals_body[i, k]
        n_world = quat_rotate(qi, n_body)
        if np.dot(n_world, dir_c) < -0.1:
            continue

        # Transform to world
        body_vec = np.array([bx_i, by_i, bz_i])
        w_pos = np.array([xi, yi, zi]) + quat_rotate(qi, body_vec)

        # Transform to body frame of j
        dv = w_pos - np.array([xj, yj, zj])
        body_j = quat_rotate_inv(qj, dv)

        # Query SDF of j
        if has_deform and np.any(eps_j != 0):
            sdf_val = query_deformed_sdf_3d(
                sdf_j, lo_jx, hi_jx, lo_jy, hi_jy, lo_jz, hi_jz, res,
                body_j[0], body_j[1], body_j[2],
                eps_j, gs.a[j], gs.b[j], gs.c[j], nu, n_modes)
        else:
            sdf_val = query_sdf_3d(sdf_j, lo_jx, hi_jx, lo_jy, hi_jy, lo_jz, hi_jz,
                                   res, body_j[0], body_j[1], body_j[2])

        if sdf_val >= 0:
            continue

        gnx, gny, gnz = query_sdf_gradient_3d(
            sdf_j, lo_jx, hi_jx, lo_jy, hi_jy, lo_jz, hi_jz, res,
            body_j[0], body_j[1], body_j[2])
        g_mag = np.sqrt(gnx**2 + gny**2 + gnz**2)
        if g_mag < 1e-20:
            continue
        n_body_j = np.array([gnx / g_mag, gny / g_mag, gnz / g_mag])
        n_world_contact = quat_rotate(qj, n_body_j)

        area_k = gs.surface_node_areas[i, k]
        delta_k = -sdf_val
        if delta_k > overlap_max:
            overlap_max = delta_k

        f_mag = E_star * delta_k * area_k * 1e-3

        f_world = f_mag * n_world_contact

        F_i += f_world
        F_j -= f_world

        r_i = w_pos - np.array([xi, yi, zi])
        r_j = w_pos - np.array([xj, yj, zj])
        tau_i += np.cross(r_i, f_world)
        tau_j -= np.cross(r_j, f_world)

        if gs.epsilon is not None:
            f_body_i = quat_rotate_inv(qi, f_world)
            for alpha in range(n_modes):
                phi = gs.mode_shapes_at_nodes[i, k, alpha]
                F_eps_i[alpha] += np.dot(phi, f_body_i)

            f_body_j = quat_rotate_inv(qj, -f_world)
            modes_j = _mode_displacement_3d(
                body_j[0], body_j[1], body_j[2],
                gs.a[j], gs.b[j], gs.c[j], nu, n_modes)
            for alpha in range(n_modes):
                F_eps_j[alpha] += np.dot(modes_j[alpha], f_body_j)

        n_contacts += 1

    # ── Test nodes of j against SDF of i (symmetric) ──
    for k in range(n_nodes):
        bx_jk = gs.surface_nodes_body[j, k, 0]
        by_jk = gs.surface_nodes_body[j, k, 1]
        bz_jk = gs.surface_nodes_body[j, k, 2]
        if has_deform:
            for alpha in range(n_modes):
                bx_jk += eps_j[alpha] * gs.mode_shapes_at_nodes[j, k, alpha, 0]
                by_jk += eps_j[alpha] * gs.mode_shapes_at_nodes[j, k, alpha, 1]
                bz_jk += eps_j[alpha] * gs.mode_shapes_at_nodes[j, k, alpha, 2]

        n_body = gs.surface_normals_body[j, k]
        n_world = quat_rotate(qj, n_body)
        if np.dot(n_world, -dir_c) < -0.1:
            continue

        body_vec = np.array([bx_jk, by_jk, bz_jk])
        w_pos = np.array([xj, yj, zj]) + quat_rotate(qj, body_vec)

        dv = w_pos - np.array([xi, yi, zi])
        body_i = quat_rotate_inv(qi, dv)

        if has_deform and np.any(eps_i != 0):
            sdf_val = query_deformed_sdf_3d(
                sdf_i, lo_ix, hi_ix, lo_iy, hi_iy, lo_iz, hi_iz, res,
                body_i[0], body_i[1], body_i[2],
                eps_i, gs.a[i], gs.b[i], gs.c[i], nu, n_modes)
        else:
            sdf_val = query_sdf_3d(sdf_i, lo_ix, hi_ix, lo_iy, hi_iy, lo_iz, hi_iz,
                                   res, body_i[0], body_i[1], body_i[2])

        if sdf_val >= 0:
            continue

        gnx, gny, gnz = query_sdf_gradient_3d(
            sdf_i, lo_ix, hi_ix, lo_iy, hi_iy, lo_iz, hi_iz, res,
            body_i[0], body_i[1], body_i[2])
        g_mag = np.sqrt(gnx**2 + gny**2 + gnz**2)
        if g_mag < 1e-20:
            continue
        n_body_i = np.array([gnx / g_mag, gny / g_mag, gnz / g_mag])
        n_world_contact = quat_rotate(qi, n_body_i)

        area_k = gs.surface_node_areas[j, k]
        delta_k = -sdf_val
        if delta_k > overlap_max:
            overlap_max = delta_k

        f_mag = E_star * delta_k * area_k * 1e-3

        f_world = f_mag * n_world_contact

        F_j += f_world
        F_i -= f_world

        r_j = w_pos - np.array([xj, yj, zj])
        r_i = w_pos - np.array([xi, yi, zi])
        tau_j += np.cross(r_j, f_world)
        tau_i -= np.cross(r_i, f_world)

        if gs.epsilon is not None:
            f_body_j = quat_rotate_inv(qj, f_world)
            for alpha in range(n_modes):
                phi = gs.mode_shapes_at_nodes[j, k, alpha]
                F_eps_j[alpha] += np.dot(phi, f_body_j)

            f_body_i = quat_rotate_inv(qi, -f_world)
            modes_i = _mode_displacement_3d(
                body_i[0], body_i[1], body_i[2],
                gs.a[i], gs.b[i], gs.c[i], nu, n_modes)
            for alpha in range(n_modes):
                F_eps_i[alpha] += np.dot(modes_i[alpha], f_body_i)

        n_contacts += 1

    if n_contacts == 0:
        return None

    # Halve to avoid double counting
    F_i *= 0.5
    F_j *= 0.5
    tau_i *= 0.5
    tau_j *= 0.5
    F_eps_i *= 0.5
    F_eps_j *= 0.5

    return {
        'in_contact': True,
        'overlap_max': overlap_max,
        'n_contact_nodes': n_contacts,
        'F_i': F_i, 'F_j': F_j,
        'tau_i': tau_i, 'tau_j': tau_j,
        'F_eps_i': F_eps_i, 'F_eps_j': F_eps_j,
        'i': i, 'j': j,
    }


# ══════════════════════════════════════════════════════════════════════
# Deformation Integration (Implicit Euler)
# ══════════════════════════════════════════════════════════════════════

def integrate_deformation_implicit(gs, p):
    """
    Integrate overdamped deformation DOFs using implicit Euler.

    The overdamped equation:
        γ_def dε/dt + K ε = F_ε

    Implicit Euler:
        (γ_def/dt I + K) ε_new = γ_def/dt ε_old + F_ε
        ε_new = solve(A, rhs)

    The system is n_modes x n_modes (typically 2-3), so direct solve is trivial.
    """
    n_modes = p.n_def_modes
    dt = p.dt

    for i in range(gs.N):
        # Deformation drag: gamma = tau_relax * K, where tau_relax = def_drag_scale (hours).
        # This gives deformation relaxation time = def_drag_scale hours.
        # Units: gamma [nN·µm·h], K [nN·µm], so gamma/K = def_drag_scale [h]. ✓
        K_diag_mean = max(np.trace(gs.K_def[i]) / n_modes, 1e-10)
        gamma_def = p.def_drag_scale * K_diag_mean  # nN·µm·h

        # Build system matrix: A = (gamma_def/dt) * I + K
        A = (gamma_def / dt) * np.eye(n_modes) + gs.K_def[i]

        # RHS: (gamma_def/dt) * eps_old + F_eps
        rhs = (gamma_def / dt) * gs.epsilon[i] + gs.F_eps_accum[i]

        # Solve (tiny system — direct)
        eps_new = np.linalg.solve(A, rhs)

        # Cap deformation amplitude
        for alpha in range(n_modes):
            eps_new[alpha] = np.clip(eps_new[alpha], -p.def_eps_max, p.def_eps_max)

        # Store rate
        gs.d_epsilon[i] = (eps_new - gs.epsilon[i]) / dt
        gs.epsilon[i] = eps_new


# ══════════════════════════════════════════════════════════════════════
# Top-Level Initialization
# ══════════════════════════════════════════════════════════════════════

def init_lsdem(gs, p):
    """
    Initialize all LS-DEM data structures for deformable particles.

    Call after packing generation in run(), before the main loop.
    """
    n_modes = p.n_def_modes
    N = gs.N

    print("  Initializing LS-DEM deformable particles...")

    # Allocate deformation state
    gs.epsilon = np.zeros((N, n_modes), dtype=np.float64)
    gs.d_epsilon = np.zeros((N, n_modes), dtype=np.float64)
    gs.F_eps_accum = np.zeros((N, n_modes), dtype=np.float64)

    # Pre-compute SDF grids
    print(f"    Computing SDF grids ({p.sdf_resolution}^{'3' if gs.is_3d else '2'} per particle)...")
    init_sdf_grids(gs, p)

    # Place surface nodes
    if gs.is_3d:
        print(f"    Placing {p.n_surface_nodes_3d} surface nodes per particle (3D)...")
        init_surface_nodes_3d(gs, p)
    else:
        print(f"    Placing {p.n_surface_nodes} surface nodes per particle (2D)...")
        init_surface_nodes_2d(gs, p)

    # Compute mode shapes and stiffness
    print(f"    Computing {n_modes} deformation modes per particle...")
    init_mode_shapes(gs, p)

    # Disable MC-DEM when deformable (LS-DEM captures multi-contact stiffening)
    if p.mc_dem_enabled:
        print("    Note: MC-DEM disabled (LS-DEM captures multi-contact effects)")
        p.mc_dem_enabled = False

    # Memory report
    sdf_mem = sum(g.nbytes for g in gs.sdf_grids) / 1e6
    node_mem = (gs.surface_nodes_body.nbytes + gs.surface_normals_body.nbytes +
                gs.surface_node_areas.nbytes) / 1e6
    mode_mem = gs.mode_shapes_at_nodes.nbytes / 1e6
    print(f"    Memory: SDF={sdf_mem:.1f} MB, nodes={node_mem:.1f} MB, modes={mode_mem:.1f} MB")
    print("  LS-DEM initialization complete.")
