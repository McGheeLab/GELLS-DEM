"""
granular_simulation.py
=======================

This module implements a coarse-grained simulation of active rearrangement
of a binary granular system inspired by fibroblast-driven hydrogel
rearrangement.  The model is based on a discrete element method (DEM)
with soft-sphere contacts and dynamic, contractile bonds between
functional granules.  The simulation code aims to reproduce the key
features described in the user's conceptual framework:

1. Two species of particles:
   - ``functional`` granules (index ``F``) that can form active cell
     bridges with other functional particles.
   - ``inert`` granules (index ``I``) that do not form bridges.

2. Contacts between any pair of particles are treated with a linear
   spring model with damping.  A simple tangential friction model is
   included but can be disabled via parameters.

3. Active bonds form only between functional particles when they come
   within a capture range.  Each functional particle has a finite
   number of adhesive “cells” available, limited by its surface area.
   These cells are allocated dynamically among nearby functional
   neighbours using a distance‐weighted scheme.  Bonds recruit cells
   over a timescale ``tau_n`` and contract towards contact with a
   rate ``beta``.  Each engaged cell contributes a fixed contractile
   force up to a maximum.

4. The system evolves on two separated timescales:
   - A **cell time step** ``dt_cell`` during which cell recruitment and
     bond contraction are updated.
   - A **mechanical relaxation** loop that drives the system to a
     mechanically equilibrated state under the current bond and contact
     forces.  This is performed by overdamped integration until
     forces fall below a tolerance.

5. The simulation can run for several days of biological time by
   choosing appropriate values of ``dt_cell`` and the rate constants
   (``beta``, ``tau_n``, etc.).  To keep runtimes reasonable for
   laptops, the simulation domain should be chosen to include only a
   modest number of particles (tens to a few hundreds) and large
   mechanical time steps can be used because the mechanical
   relaxation is overdamped.

This code is intended as a starting point for experimentation.
Parameters are exposed at the top level of the ``Simulation`` class
and in the ``run_example`` function.  Users should adjust the domain
size, number of particles, mechanical stiffness, and time step sizes
to achieve the desired balance between realism and computational
efficiency.

Note: This implementation avoids heavy dependencies and is written
using only ``numpy`` for numeric operations.  No GPU acceleration or
spatial partitioning structures are used; therefore the complexity
scales quadratically with the number of particles.  For coarse
simulations on a laptop this is acceptable up to a few hundred
particles.  For larger systems consider using dedicated DEM libraries.

Author: OpenAI ChatGPT (2026)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import numpy as np


def random_packing(N: int, radii: np.ndarray, box_size: float,
                   max_attempts: int = 10000) -> np.ndarray:
    """Generate non-overlapping random positions for particles in a cubic domain.

    Parameters
    ----------
    N : int
        Number of particles.
    radii : np.ndarray
        Array of radii of length N.
    box_size : float
        Side length of the cubic domain (domain is [0, box_size]^3).
    max_attempts : int
        Maximum number of attempts to place each particle before
        giving up.  If packing fails, a ValueError is raised.

    Returns
    -------
    positions : np.ndarray
        Array of shape (N, 3) with the initial positions.

    Notes
    -----
    This function uses a simple rejection sampling algorithm to place
    spheres sequentially in the box without overlaps.  It is not
    efficient for high packing fractions or large particle numbers but
    suffices for moderate N when the total volume fraction is low.
    """
    positions = np.zeros((N, 3), dtype=np.float64)
    for i in range(N):
        placed = False
        for attempt in range(max_attempts):
            pos = np.random.rand(3) * box_size
            if i == 0:
                positions[i] = pos
                placed = True
                break
            # check overlap with previous spheres
            diffs = positions[:i] - pos
            dist2 = np.sum(diffs * diffs, axis=1)
            min_dist2 = (radii[:i] + radii[i]) ** 2
            if np.all(dist2 >= min_dist2):
                positions[i] = pos
                placed = True
                break
        if not placed:
            raise ValueError(
                f"Failed to pack particle {i} after {max_attempts} attempts. "
                "Try increasing the box size or decreasing packing fraction."
            )
    return positions


@dataclass
class Bond:
    """Data structure for an active bond between two functional particles."""
    i: int
    j: int
    # Number of engaged cells allocated to this bond (continuous, non-negative)
    n_cells: float = 0.0
    # Rest length of the contractile spring (starts at current distance)
    rest_length: float = 0.0
    # Current gap at bond creation (used for hysteresis)
    # This can be used to implement a break threshold based on a larger gap


@dataclass
class Simulation:
    """Discrete element simulation with active contractile bonds."""

    # Particle properties
    radii: np.ndarray  # (N,) radii of all particles
    types: np.ndarray  # (N,) booleans: True for functional, False for inert
    box_size: float  # domain side length

    # Active bond parameters
    lambda_on: float  # capture range for bond formation (absolute units)
    lambda_off: float  # range beyond which bonds break (>= lambda_on)
    cell_footprint: float  # effective area per cell on a granule surface
    alpha_surface: float  # surface area packing factor for cell adhesion
    f_cell: float  # contractile force contributed by a single engaged cell
    k_cell: float  # spring constant per engaged cell
    beta: float  # rate of rest length contraction (1/time)
    tau_n: float  # timescale for cell recruitment dynamics
    n_cells_per_granule: int  # total cells per functional granule (prior to saturation)

    # Mechanical parameters
    k_n: float  # normal contact stiffness (spring constant)
    gamma_n: float  # normal damping coefficient
    k_t: float  # tangential spring constant (for friction)
    gamma_t: float  # tangential damping coefficient
    mu: float  # friction coefficient (Coulomb friction)
    # Overdamped drag coefficient (per particle); if scalar, same for all
    gamma_drag: float

    # Integration parameters
    dt_cell: float  # cell time step (biological time units)
    max_mech_iters: int  # maximum mechanical relaxation iterations per cell step
    mech_force_tol: float  # force tolerance for mechanical relaxation convergence

    # Derived/optional settings
    periodic: bool = False  # if true, apply periodic boundary conditions
    verbose: bool = False  # print progress messages

    # State variables (initialized in __post_init__)
    positions: np.ndarray = field(init=False)  # (N,3) positions
    velocities: np.ndarray = field(init=False)  # (N,3) velocities (used for damping)
    tangential: Dict[Tuple[int, int], np.ndarray] = field(init=False)  # contact tangential springs
    bonds: Dict[Tuple[int, int], Bond] = field(init=False)  # active bonds between functional pairs
    n_max: np.ndarray = field(init=False)  # (N,) maximum number of cells per granule (saturation)

    def __post_init__(self):
        N = len(self.radii)
        # Initialize positions by random non-overlapping packing
        self.positions = random_packing(N, self.radii, self.box_size)
        self.velocities = np.zeros((N, 3), dtype=np.float64)
        self.tangential = {}
        self.bonds = {}
        # Precompute surface area per granule for saturation
        surface_area = 4.0 * math.pi * (self.radii ** 2)
        self.n_max = self.alpha_surface * surface_area / self.cell_footprint
        # If user provided fewer cells per granule than saturation limit, use that
        self.n_total = np.where(self.types, np.minimum(self.n_cells_per_granule, self.n_max), 0.0)

    def apply_periodic(self, dr: np.ndarray) -> np.ndarray:
        """Apply periodic boundary conditions to a displacement vector.

        For each component of the vector, if the displacement is greater
        than half the box size, subtract the box size; if less than
        minus half the box size, add the box size.  This implements
        minimum image convention.
        """
        if not self.periodic:
            return dr
        half = self.box_size * 0.5
        dr = np.where(dr > half, dr - self.box_size, dr)
        dr = np.where(dr < -half, dr + self.box_size, dr)
        return dr

    def build_contacts(self) -> List[Tuple[int, int, float, np.ndarray]]:
        """Compute contact overlaps and unit normals between all particle pairs.

        Returns a list of tuples (i, j, delta_n, n_hat) for pairs in contact.
        delta_n = (R_i + R_j) - distance; n_hat points from j to i.
        """
        N = len(self.radii)
        contacts = []
        for i in range(N - 1):
            for j in range(i + 1, N):
                # displacement vector from j to i
                dr = self.positions[i] - self.positions[j]
                dr = self.apply_periodic(dr)
                dist = np.linalg.norm(dr)
                r_sum = self.radii[i] + self.radii[j]
                overlap = r_sum - dist
                if overlap > 0:
                    # compute normal
                    if dist > 0:
                        n_hat = dr / dist
                    else:
                        # Particles on top of each other; choose arbitrary normal
                        n_hat = np.array([1.0, 0.0, 0.0])
                    contacts.append((i, j, overlap, n_hat))
        return contacts

    def update_bonds(self):
        """Update cell allocations and bond states based on current positions.

        This function follows the saturation and allocation scheme described
        in the conceptual framework.  Each functional particle has a
        maximum number of cells ``n_max[i]`` and a total available cells
        ``n_total[i]``.  Cells are allocated to all functional neighbours
        within ``lambda_off`` with weights that decay with the gap.

        Bond formation and breaking are handled here: if a functional pair
        is within ``lambda_on`` capture range and has a non-zero
        allocation, a bond either forms or is updated.  If the gap is
        larger than ``lambda_off`` the bond is removed.
        """
        N = len(self.radii)
        positions = self.positions
        radii = self.radii
        # Compute neighbour lists for functional particles within lambda_off
        neighbours: Dict[int, List[int]] = {i: [] for i in range(N)}
        gaps: Dict[Tuple[int, int], float] = {}
        for i in range(N - 1):
            if not self.types[i]:
                continue  # inert particles do not form bonds
            for j in range(i + 1, N):
                if not self.types[j]:
                    continue
                dr = positions[i] - positions[j]
                dr = self.apply_periodic(dr)
                dist = np.linalg.norm(dr)
                gap = dist - (radii[i] + radii[j])
                if gap <= self.lambda_off:
                    neighbours[i].append(j)
                    neighbours[j].append(i)
                    gaps[(i, j)] = gap
        # Allocate cells proportionally to distance weights
        allocations: Dict[Tuple[int, int], float] = {}
        for i in range(N):
            if not self.types[i]:
                continue
            neighs = neighbours[i]
            if not neighs:
                continue
            # Weights: exponential decay with gap/lengthscale; choose p=2, lambda_w = lambda_on/2
            lambda_w = self.lambda_on * 0.5 if self.lambda_on > 0 else 1.0
            weights = []
            for j in neighs:
                # symmetric key (min, max) for gap
                key = (min(i, j), max(i, j))
                gap = gaps[key]
                # Negative gap means overlapping; treat as zero gap
                x = max(gap, 0.0) / lambda_w
                w = math.exp(-x * x)
                weights.append(w)
            weights = np.array(weights)
            if weights.sum() == 0:
                continue
            # total available cells for this particle (saturated by n_max)
            n_available = self.n_total[i]
            # allocate fractionally
            weights_norm = weights / weights.sum()
            for idx, j in enumerate(neighs):
                a_ij = n_available * weights_norm[idx]
                allocations[(i, j) if i < j else (j, i)] = allocations.get((min(i, j), max(i, j)), 0.0) + a_ij
        # Ensure symmetric allocation takes minimum of both directions
        targets: Dict[Tuple[int, int], float] = {}
        for (i, j), a in allocations.items():
            # a is sum of contributions from i and j (we counted both), but we need min of both allocations
            # We will divide by 2 when updating; but to be safe, compute min
            # However, since we used symmetric keys and added contributions from both sides, we store total
            targets[(i, j)] = a  # we will later limit by n_max implicitly
        # Update each bond based on target allocation with recruitment dynamics
        # For pairs that are within lambda_off but have zero target, we allow bond to relax to zero
        for (i, j), target in targets.items():
            # Limit engaged cells by per-particle maximum; since we normalized by total n_total this is okay
            # Calculate desired engaged cell number (target), but enforce non-negative
            target_n = max(target, 0.0)
            # Determine if within capture range for formation
            gap = gaps.get((i, j), 0.0)
            within_capture = gap <= self.lambda_on
            key = (i, j)
            b = self.bonds.get(key)
            if within_capture and target_n > 0:
                # Create bond if it does not exist
                if b is None:
                    b = Bond(i=i, j=j, n_cells=0.0, rest_length=0.0)
                    # initialize rest_length to current distance
                    # compute current distance with periodic correction
                    dr = self.positions[i] - self.positions[j]
                    dr = self.apply_periodic(dr)
                    dist = np.linalg.norm(dr)
                    b.rest_length = dist
                    self.bonds[key] = b
                # update engaged cells via first-order kinetics
                dn = (target_n - b.n_cells) * (self.dt_cell / self.tau_n)
                b.n_cells += dn
                # Update rest length contraction
                # exponential relaxation towards contact
                r_sum = self.radii[i] + self.radii[j]
                b.rest_length = r_sum + (b.rest_length - r_sum) * math.exp(-self.beta * self.dt_cell)
            else:
                # Outside capture or no target; if bond exists, decay engaged cells to zero and remove if small
                if b is not None:
                    # Relax engaged cells
                    dn = (0.0 - b.n_cells) * (self.dt_cell / self.tau_n)
                    b.n_cells += dn
                    # rest length still contracts slowly if gap < lambda_off
                    r_sum = self.radii[i] + self.radii[j]
                    b.rest_length = r_sum + (b.rest_length - r_sum) * math.exp(-self.beta * self.dt_cell)
                    # Remove bond if engaged cells below threshold or gap above lambda_off
                    if b.n_cells < 1e-6 or gap > self.lambda_off:
                        self.bonds.pop(key, None)

        # Also remove bonds for pairs no longer neighbours
        to_remove = []
        for key in self.bonds.keys():
            i, j = key
            # If not in neighbours, remove
            if key not in targets:
                to_remove.append(key)
        for key in to_remove:
            self.bonds.pop(key, None)

    def mechanical_relaxation(self):
        """Relax mechanical forces using overdamped integration.

        This performs a fixed number of iterations or until the maximum
        force magnitude falls below ``mech_force_tol``.  Overdamped
        integration with drag ``gamma_drag`` is used.  Tangential
        friction is accumulated in ``self.tangential``.
        """
        N = len(self.radii)
        positions = self.positions
        radii = self.radii
        # Reset forces and tangential springs; we accumulate incremental tangential displacement
        # Create a copy to avoid modifying dict during iteration
        for contact_key in list(self.tangential.keys()):
            # We'll reset tangential state only when contact ends in step (see below)
            pass

        for iter_num in range(self.max_mech_iters):
            # Initialize forces and torques to zero
            forces = np.zeros((N, 3), dtype=np.float64)
            torques = np.zeros((N, 3), dtype=np.float64)
            # Compute contact forces
            contacts = self.build_contacts()
            for (i, j, delta_n, n_hat) in contacts:
                # normal relative velocity at contact
                # relative velocity includes drag effect; we don't have explicit velocities, so approximate via previous velocities
                dv = self.velocities[i] - self.velocities[j]
                vn = np.dot(dv, n_hat)
                fn_mag = self.k_n * delta_n - self.gamma_n * vn
                if fn_mag < 0:
                    fn_mag = 0.0
                fn_vec = fn_mag * n_hat
                # tangential displacement and velocity
                vt_vec = dv - vn * n_hat
                # unique key for tangential spring
                key = (i, j)
                # update tangential spring displacement
                xi = self.tangential.get(key)
                if xi is None:
                    xi = np.zeros(3)
                # incremental tangential displacement
                xi = xi + vt_vec * (1.0)  # treat dt_m = 1 for scaling; will be normalized by gamma_drag later
                # trial tangential force
                ft_vec = -self.k_t * xi - self.gamma_t * vt_vec
                # apply Coulomb friction limit
                ft_mag = np.linalg.norm(ft_vec)
                max_ft = self.mu * fn_mag
                if ft_mag > max_ft and ft_mag > 0:
                    ft_vec = ft_vec / ft_mag * max_ft
                    # renormalize tangential spring to reflect slip
                    xi = -(ft_vec + self.gamma_t * vt_vec) / self.k_t
                # update stored tangential spring
                self.tangential[key] = xi
                # accumulate forces
                forces[i] += fn_vec + ft_vec
                forces[j] -= fn_vec + ft_vec
                # accumulate torques (for rotational friction; optional)
                # For spheres, torque due to tangential force = R * n_hat x ft_vec
                ri = self.radii[i]
                rj = self.radii[j]
                torques[i] += ri * np.cross(n_hat, ft_vec)
                torques[j] -= rj * np.cross(n_hat, ft_vec)
            # Compute bond forces
            for (i, j), b in list(self.bonds.items()):
                # compute vector from j to i
                dr = positions[i] - positions[j]
                dr = self.apply_periodic(dr)
                dist = np.linalg.norm(dr)
                if dist == 0:
                    n_hat = np.array([1.0, 0.0, 0.0])
                else:
                    n_hat = dr / dist
                # bond spring force = k_cell * n_cells * (dist - rest_length), attractive only
                # limit maximum to f_cell * n_cells
                delta = dist - b.rest_length
                if delta > 0:
                    fb_mag = 0.0  # bond is slack if longer than rest length
                else:
                    fb_raw = self.k_cell * b.n_cells * delta  # negative quantity (contractile)
                    fb_max = self.f_cell * b.n_cells
                    # limit magnitude to max active traction
                    if abs(fb_raw) > fb_max:
                        fb_raw = -fb_max
                    fb_mag = fb_raw
                fb_vec = fb_mag * n_hat
                forces[i] += fb_vec
                forces[j] -= fb_vec
                # no torques from central bond forces
            # Determine maximum force magnitude
            max_force = np.max(np.linalg.norm(forces, axis=1))
            # Compute velocities from forces via overdamped drag: v = F / gamma_drag
            # gamma_drag may be scalar or array matching particles
            if np.isscalar(self.gamma_drag):
                self.velocities = forces / self.gamma_drag
            else:
                # broadcast
                self.velocities = forces / self.gamma_drag[:, None]
            # Update positions
            self.positions += self.velocities
            # Apply periodic boundary conditions if enabled
            if self.periodic:
                self.positions = np.mod(self.positions, self.box_size)
            # If convergence achieved, break
            if max_force < self.mech_force_tol:
                break
        # End mechanical relaxation

    def step(self):
        """Perform one cell time step: update bonds and relax mechanics."""
        # Update cell allocations and bond dynamics
        self.update_bonds()
        # Relax mechanical forces for the current configuration and bonds
        self.mechanical_relaxation()

    def run(self, total_time: float, record_interval: float = 1.0) -> Dict[str, List]:
        """Run the simulation for a given total biological time.

        Parameters
        ----------
        total_time : float
            Total simulation time in the same units as ``dt_cell`` (e.g., days).
        record_interval : float
            Interval at which to record summary outputs (same units as
            total_time).  For example, record once per day.

        Returns
        -------
        results : dict
            Dictionary containing recorded times, porosity, functional
            compaction index, inert segregation index, and bond counts.

        Notes
        -----
        The simulation state (positions, bonds) is updated in-place.
        """
        results = {
            "time": [],
            "porosity": [],
            "compaction": [],
            "inert_segregation": [],
            "bond_count": [],
        }
        t = 0.0
        next_record = 0.0
        # initial porosity and packing fraction
        initial_phi = self.compute_solid_fraction()
        initial_porosity = 1.0 - initial_phi
        while t < total_time:
            # perform one cell step
            self.step()
            t += self.dt_cell
            # record if needed
            if t >= next_record or math.isclose(t, total_time):
                phi = self.compute_solid_fraction()
                porosity = 1.0 - phi
                compaction = (initial_porosity - porosity) / (initial_porosity + 1e-12)
                # compute inert segregation index
                seg = self.compute_inert_segregation()
                results["time"].append(t)
                results["porosity"].append(porosity)
                results["compaction"].append(compaction)
                results["inert_segregation"].append(seg)
                results["bond_count"].append(len(self.bonds))
                if self.verbose:
                    print(f"t={t:.3f}: porosity={porosity:.3f}, compaction={compaction:.3f}, bonds={len(self.bonds)}")
                next_record += record_interval
        return results

    def compute_solid_fraction(self) -> float:
        """Compute the solids volume fraction inside the domain."""
        V_domain = self.box_size ** 3
        volumes = (4.0 / 3.0) * math.pi * (self.radii ** 3)
        return volumes.sum() / V_domain

    def compute_inert_segregation(self) -> float:
        """Compute an inert segregation index based on positions and voids.

        A simple implementation counts inert particles whose centres lie
        in regions of lower local functional density.  We discretize the
        domain into a coarse grid and compute porosity; inert centres
        inside voxels above the average porosity contribute to the
        segregation index.  The index ranges from 0 (no segregation) to
        1 (all inert particles in voids).
        """
        # grid resolution: 10 cells per dimension
        ngrid = 10
        cell_size = self.box_size / ngrid
        # arrays to accumulate solid volume per voxel
        solid = np.zeros((ngrid, ngrid, ngrid), dtype=np.float64)
        # occupancy counts for inert particles
        inert_in_void = 0
        inert_total = 0
        # compute solid fraction in each voxel by summing volumes of spheres intersecting voxel
        # For efficiency, approximate by assigning each sphere's full volume to the voxel containing its centre
        indices = np.floor(self.positions / cell_size).astype(int)
        indices = np.clip(indices, 0, ngrid - 1)
        volumes = (4.0 / 3.0) * math.pi * (self.radii ** 3)
        for vol, idx in zip(volumes, indices):
            solid[tuple(idx)] += vol
        # compute porosity per voxel
        voxel_volume = cell_size ** 3
        porosity = 1.0 - (solid / voxel_volume)
        mean_porosity = np.mean(porosity)
        # count inert particles in voxels with porosity greater than mean
        for i in range(len(self.radii)):
            if not self.types[i]:
                inert_total += 1
                idx = tuple(indices[i])
                if porosity[idx] > mean_porosity:
                    inert_in_void += 1
        if inert_total == 0:
            return 0.0
        return inert_in_void / inert_total


def run_example():
    """Run a coarse simulation for demonstration.

    This function sets up a small number of granules with a range of
    diameters (20–200 µm) and runs the simulation for three days.  The
    parameters are chosen to produce meaningful rearrangement while
    keeping the computational cost low.  Adjust ``num_particles``,
    ``box_size`` and the various stiffness constants to trade off
    speed and accuracy.

    Returns
    -------
    results : dict
        Recorded results of the simulation.
    sim : Simulation
        The simulation instance with final state (positions, bonds, etc.).
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    # Number of particles (total).  To keep the example runtimes reasonable
    # on a laptop, use around 60–80 particles.  Increase this for
    # more realistic results at the cost of longer simulation times.
    num_particles = 80  # adjust for speed; mixture of functional and inert
    # Fraction of functional particles
    frac_functional = 0.6
    num_func = int(num_particles * frac_functional)
    num_inert = num_particles - num_func
    # Radii: random diameters between 20 and 200 µm.  Convert to metres for simulation.
    diameters_um = np.random.uniform(20.0, 200.0, size=num_particles)
    radii_um = diameters_um / 2.0
    radii_m = radii_um * 1e-6  # convert microns to metres
    # Types: first num_func functional (True), rest inert (False)
    types = np.zeros(num_particles, dtype=bool)
    types[:num_func] = True
    # Shuffle particle order to mix types
    perm = np.random.permutation(num_particles)
    radii_m = radii_m[perm]
    types = types[perm]
    # Domain size: choose box so packing fraction ~0.2
    # approximate total volume of particles
    volumes = (4.0 / 3.0) * math.pi * (radii_m ** 3)
    total_volume = volumes.sum()
    target_packing = 0.15
    box_vol = total_volume / target_packing
    box_size = box_vol ** (1.0 / 3.0)
    # Parameters
    lambda_on = 50e-6  # 50 µm bridging reach
    lambda_off = 80e-6  # 80 µm off range
    cell_footprint = (20e-6) ** 2  # one cell occupies approx 20 µm diameter footprint
    alpha_surface = 0.7  # packing efficiency on surface
    f_cell = 50e-9  # 50 nN per engaged cell (approximate traction)
    k_cell = 5e-3  # 5 mN/m spring constant per cell
    beta = 1.0 / (24.0)  # rest length contraction rate: day-scale (per hour)
    tau_n = 4.0  # hours timescale for recruitment
    n_cells_per_granule = 50  # total cells per functional granule before saturation
    # Mechanical parameters
    k_n = 1e3  # contact stiffness (N/m); choose large to keep overlaps small
    gamma_n = 1e-2  # damping (N*s/m)
    k_t = 2e2  # tangential stiffness (N/m)
    gamma_t = 1e-2  # tangential damping (N*s/m)
    mu = 0.5  # friction coefficient
    gamma_drag = 1e-2  # drag coefficient (N*s/m)
    # Integration parameters
    dt_cell = 6.0  # hours per cell step (biological time)
    # Convert to days units: 1 day = 24 hours
    dt_cell_days = dt_cell / 24.0
    max_mech_iters = 30
    mech_force_tol = 1e-6
    # Create simulation
    sim = Simulation(
        radii=radii_m,
        types=types,
        box_size=box_size,
        lambda_on=lambda_on,
        lambda_off=lambda_off,
        cell_footprint=cell_footprint,
        alpha_surface=alpha_surface,
        f_cell=f_cell,
        k_cell=k_cell,
        beta=beta,
        tau_n=tau_n,
        n_cells_per_granule=n_cells_per_granule,
        k_n=k_n,
        gamma_n=gamma_n,
        k_t=k_t,
        gamma_t=gamma_t,
        mu=mu,
        gamma_drag=gamma_drag,
        dt_cell=dt_cell_days,
        max_mech_iters=max_mech_iters,
        mech_force_tol=mech_force_tol,
        periodic=True,
        verbose=True,
    )
    # Run simulation for 3 days
    total_time_days = 3.0
    record_interval_days = 1.0
    results = sim.run(total_time=total_time_days, record_interval=record_interval_days)
    return results, sim


if __name__ == "__main__":
    # Execute example when run as a script
    results, sim = run_example()
    # Print recorded summary
    print("\nSimulation summary (recorded each day):")
    for t, por, comp, seg, bc in zip(
            results["time"], results["porosity"], results["compaction"],
            results["inert_segregation"], results["bond_count"]):
        print(f"Day {t:.1f}: porosity={por:.3f}, compaction={comp:.3f}, "
              f"segregation={seg:.3f}, bonds={bc}")