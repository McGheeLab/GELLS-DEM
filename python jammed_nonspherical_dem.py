"""
Coarse-Grained DEM with Jammed Non-Spherical Granules
======================================================

Features:
1. Jammed initial packing via Lubachevsky-Stillinger growth algorithm
2. Non-spherical granules (superellipsoids) with:
   - Form/Sphericity: aspect ratios (a, b, c semi-axes)
   - Roundness: superellipsoid exponent (n=2 sphere, n>2 blocky)
   - Roughness: affects friction coefficient
3. Orientation tracking with rotational dynamics
4. Overdamped dynamics for long-timescale simulation

Requirements:
    pip install numpy matplotlib scipy tqdm

Usage:
    python jammed_nonspherical_dem.py
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from pathlib import Path
import json
import time as time_module

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.transforms as transforms
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from scipy.spatial import cKDTree
    from scipy.optimize import minimize_scalar, brentq
    from scipy.spatial.transform import Rotation
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found. Using fallback implementations.")


# =============================================================================
# SHAPE REPRESENTATION
# =============================================================================

@dataclass
class GranuleShape:
    """
    Shape descriptor for a granule using superellipsoid representation.
    
    Superellipsoid implicit equation:
        (|x/a|^(2/e2) + |y/b|^(2/e2))^(e2/e1) + |z/c|^(2/e1) = 1
    
    For simplicity, we use a single exponent n:
        |x/a|^n + |y/b|^n + |z/c|^n = 1
    
    Parameters:
        a, b, c: Semi-axes defining overall form (sphericity)
        n: Roundness exponent (2=ellipsoid, >2=blocky/angular, <2=pinched)
        roughness: Surface roughness parameter [0,1] affecting friction
    
    Shape metrics:
        - Sphericity Ψ = (π^(1/3) * (6V)^(2/3)) / A
          where V=volume, A=surface area
          Ψ=1 for sphere, <1 for other shapes
        
        - Roundness R = (average radius of corners) / (max inscribed radius)
          Related to exponent n: higher n = lower roundness
    """
    # Semi-axes (form/sphericity)
    a: float  # x semi-axis
    b: float  # y semi-axis  
    c: float  # z semi-axis
    
    # Roundness exponent
    n: float = 3.0  # 2=ellipsoid, >2=more angular
    
    # Surface roughness [0, 1]
    roughness: float = 0.5
    
    @property
    def equivalent_radius(self) -> float:
        """Radius of sphere with same volume."""
        V = self.volume
        return (3 * V / (4 * np.pi)) ** (1/3)
    
    @property
    def volume(self) -> float:
        """Approximate volume of superellipsoid."""
        # For ellipsoid (n=2): V = (4/3)πabc
        # For general superellipsoid, use approximation
        if abs(self.n - 2.0) < 0.01:
            return (4/3) * np.pi * self.a * self.b * self.c
        else:
            # Approximation for superellipsoid
            # V ≈ (2/n)^3 * (4/3)πabc * Γ(1+1/n)^3 / Γ(1+3/n)
            # Simplified approximation:
            base_vol = (4/3) * np.pi * self.a * self.b * self.c
            # Correction factor (empirical fit)
            correction = (2/self.n) ** (0.5 * (self.n - 2) / self.n)
            return base_vol * correction
    
    @property
    def sphericity(self) -> float:
        """
        Sphericity: ratio of surface area of equivalent sphere to actual surface area.
        Returns value in [0, 1], with 1 being a perfect sphere.
        """
        r_eq = self.equivalent_radius
        sphere_area = 4 * np.pi * r_eq**2
        
        # Approximate surface area of ellipsoid (Knud Thomsen approximation)
        p = 1.6075
        ap, bp, cp = self.a**p, self.b**p, self.c**p
        actual_area = 4 * np.pi * ((ap*bp + ap*cp + bp*cp) / 3) ** (1/p)
        
        # Adjust for roundness (blockier = more surface area)
        if self.n > 2:
            actual_area *= (1 + 0.1 * (self.n - 2))
        
        return min(1.0, sphere_area / actual_area)
    
    @property
    def bounding_radius(self) -> float:
        """Maximum distance from center to surface."""
        return max(self.a, self.b, self.c)
    
    @property
    def min_radius(self) -> float:
        """Minimum distance from center to surface."""
        return min(self.a, self.b, self.c)
    
    @property
    def aspect_ratio(self) -> float:
        """Ratio of max to min semi-axis."""
        return self.bounding_radius / self.min_radius
    
    def surface_point(self, theta: float, phi: float) -> np.ndarray:
        """
        Get point on superellipsoid surface in body frame.
        
        Args:
            theta: azimuthal angle [0, 2π]
            phi: polar angle [0, π]
        """
        # Parameterization for superellipsoid
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Sign-preserving power
        def spow(x, p):
            return np.sign(x) * np.abs(x) ** p
        
        e = 2.0 / self.n
        x = self.a * spow(cos_theta, e) * spow(sin_phi, e)
        y = self.b * spow(sin_theta, e) * spow(sin_phi, e)
        z = self.c * spow(cos_phi, e)
        
        return np.array([x, y, z])
    
    def inside(self, point: np.ndarray) -> bool:
        """Check if point is inside superellipsoid (body frame)."""
        x, y, z = point
        val = (np.abs(x/self.a)**self.n + 
               np.abs(y/self.b)**self.n + 
               np.abs(z/self.c)**self.n)
        return val <= 1.0
    
    def distance_to_surface(self, point: np.ndarray) -> float:
        """
        Approximate signed distance from point to surface (body frame).
        Negative if inside, positive if outside.
        """
        x, y, z = point
        # Implicit function value
        f = (np.abs(x/self.a)**self.n + 
             np.abs(y/self.b)**self.n + 
             np.abs(z/self.c)**self.n)
        
        # Approximate distance using gradient
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            return -self.min_radius
        
        # Scale to get approximate distance
        # For ellipsoid: d ≈ r * (f^(1/n) - 1)
        return r * (f**(1/self.n) - 1)


def create_random_shape(mean_radius: float, 
                        aspect_ratio_range: Tuple[float, float] = (1.0, 2.0),
                        roundness_range: Tuple[float, float] = (2.0, 4.0),
                        roughness_range: Tuple[float, float] = (0.0, 0.5)) -> GranuleShape:
    """
    Create a random granule shape with specified property ranges.
    
    Args:
        mean_radius: Target equivalent radius
        aspect_ratio_range: (min, max) for longest/shortest axis ratio
        roundness_range: (min, max) for superellipsoid exponent
        roughness_range: (min, max) for surface roughness
    """
    # Ensure positive radius
    mean_radius = max(5.0, mean_radius)
    
    # Random aspect ratio
    aspect_ratio = np.random.uniform(*aspect_ratio_range)
    aspect_ratio = max(1.0, min(aspect_ratio, 3.0))  # Clamp for stability
    
    # Start with equivalent sphere and stretch
    # For an ellipsoid with semi-axes a, b, c:
    # Volume = (4/3) * pi * a * b * c
    # We want this equal to (4/3) * pi * r^3
    # So a * b * c = r^3
    
    # Make c the long axis
    c = mean_radius * aspect_ratio**(1/3)
    ab = mean_radius / aspect_ratio**(1/6)
    
    # Add some asymmetry in a vs b (small)
    asymmetry = np.random.uniform(0.95, 1.05)
    a = ab * asymmetry
    b = ab / asymmetry
    
    # Ensure all axes are positive and reasonable
    a = max(3.0, min(a, mean_radius * 3))
    b = max(3.0, min(b, mean_radius * 3))
    c = max(3.0, min(c, mean_radius * 3))
    
    # Scale to achieve target volume
    target_vol = (4/3) * np.pi * mean_radius**3
    current_vol = (4/3) * np.pi * a * b * c
    if current_vol > 0:
        scale = (target_vol / current_vol)**(1/3)
        a *= scale
        b *= scale
        c *= scale
    
    # Random roundness and roughness
    n = np.random.uniform(*roundness_range)
    n = max(2.0, min(n, 5.0))  # Clamp for numerical stability
    
    roughness = np.random.uniform(*roughness_range)
    roughness = max(0.0, min(roughness, 1.0))
    
    return GranuleShape(a=a, b=b, c=c, n=n, roughness=roughness)


# =============================================================================
# ORIENTATION REPRESENTATION
# =============================================================================

class Quaternion:
    """Quaternion for 3D rotation representation."""
    
    def __init__(self, w: float = 1.0, x: float = 0.0, 
                 y: float = 0.0, z: float = 0.0):
        self.q = np.array([w, x, y, z], dtype=np.float64)
        self.normalize()
    
    def normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 1e-10:
            self.q /= norm
    
    @property
    def w(self): return self.q[0]
    @property
    def x(self): return self.q[1]
    @property
    def y(self): return self.q[2]
    @property
    def z(self): return self.q[3]
    
    @classmethod
    def from_axis_angle(cls, axis: np.ndarray, angle: float) -> 'Quaternion':
        """Create quaternion from axis-angle representation."""
        axis = np.asarray(axis, dtype=np.float64)
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        return cls(w, xyz[0], xyz[1], xyz[2])
    
    @classmethod
    def random(cls) -> 'Quaternion':
        """Generate uniformly random rotation quaternion."""
        u = np.random.random(3)
        q = cls(
            np.sqrt(1 - u[0]) * np.sin(2 * np.pi * u[1]),
            np.sqrt(1 - u[0]) * np.cos(2 * np.pi * u[1]),
            np.sqrt(u[0]) * np.sin(2 * np.pi * u[2]),
            np.sqrt(u[0]) * np.cos(2 * np.pi * u[2])
        )
        return q
    
    def to_rotation_matrix(self) -> np.ndarray:
        """Convert to 3x3 rotation matrix."""
        w, x, y, z = self.q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Rotate a vector by this quaternion."""
        R = self.to_rotation_matrix()
        return R @ v
    
    def inverse_rotate_vector(self, v: np.ndarray) -> np.ndarray:
        """Inverse rotate (world to body frame)."""
        R = self.to_rotation_matrix()
        return R.T @ v
    
    def multiply(self, other: 'Quaternion') -> 'Quaternion':
        """Quaternion multiplication (composition of rotations)."""
        w1, x1, y1, z1 = self.q
        w2, x2, y2, z2 = other.q
        return Quaternion(
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        )
    
    def integrate(self, omega: np.ndarray, dt: float) -> 'Quaternion':
        """Integrate angular velocity to update orientation."""
        # q_dot = 0.5 * q * [0, omega]
        omega_quat = Quaternion(0, omega[0], omega[1], omega[2])
        q_dot = self.multiply(omega_quat)
        q_dot.q *= 0.5
        
        new_q = Quaternion(
            self.w + q_dot.w * dt,
            self.x + q_dot.x * dt,
            self.y + q_dot.y * dt,
            self.z + q_dot.z * dt
        )
        new_q.normalize()
        return new_q


# =============================================================================
# CONTACT DETECTION FOR SUPERELLIPSOIDS
# =============================================================================

def superellipsoid_contact(pos1: np.ndarray, quat1: Quaternion, shape1: GranuleShape,
                           pos2: np.ndarray, quat2: Quaternion, shape2: GranuleShape,
                           ) -> Tuple[bool, float, np.ndarray, np.ndarray]:
    """
    Detect contact between two superellipsoids.
    
    Uses a combination of:
    1. Bounding sphere check (fast rejection)
    2. Gaussian overlap approximation for ellipsoids
    3. Iterative refinement for actual contact point
    
    Returns:
        in_contact: bool
        overlap: float (positive if overlapping)
        normal: contact normal (from 1 to 2)
        contact_point: point of contact in world frame
    """
    # Fast bounding sphere check
    r_sep = pos2 - pos1
    dist = np.linalg.norm(r_sep)
    max_extent = shape1.bounding_radius + shape2.bounding_radius
    
    if dist > max_extent * 1.1:
        return False, 0.0, np.zeros(3), np.zeros(3)
    
    if dist < 1e-10:
        # Coincident centers - push apart
        normal = np.array([1.0, 0.0, 0.0])
        overlap = shape1.min_radius + shape2.min_radius
        return True, overlap, normal, pos1
    
    # Direction from 1 to 2
    direction = r_sep / dist
    
    # Transform direction to body frames
    dir_body1 = quat1.inverse_rotate_vector(direction)
    dir_body2 = quat2.inverse_rotate_vector(-direction)
    
    # Find surface points along contact direction (approximate)
    # For superellipsoid, the surface point in direction d is approximately:
    # p = [a*sign(dx)*|dx|^(2/n-1), b*sign(dy)*|dy|^(2/n-1), c*sign(dz)*|dz|^(2/n-1)]
    # normalized to lie on surface
    
    def surface_distance(shape: GranuleShape, direction: np.ndarray) -> float:
        """Distance from center to surface along given direction."""
        dx, dy, dz = direction
        n = shape.n
        
        # Avoid division by zero
        eps = 1e-10
        dx = dx if abs(dx) > eps else eps * np.sign(dx + eps)
        dy = dy if abs(dy) > eps else eps * np.sign(dy + eps)
        dz = dz if abs(dz) > eps else eps * np.sign(dz + eps)
        
        # For superellipsoid |x/a|^n + |y/b|^n + |z/c|^n = 1
        # Point on surface in direction (dx, dy, dz) at distance r:
        # |r*dx/a|^n + |r*dy/b|^n + |r*dz/c|^n = 1
        # r^n * (|dx/a|^n + |dy/b|^n + |dz/c|^n) = 1
        
        term = (np.abs(dx/shape.a)**n + 
                np.abs(dy/shape.b)**n + 
                np.abs(dz/shape.c)**n)
        
        if term > 1e-10:
            r = term ** (-1/n)
        else:
            r = shape.bounding_radius
        
        return r
    
    r1 = surface_distance(shape1, dir_body1)
    r2 = surface_distance(shape2, dir_body2)
    
    # Overlap calculation
    overlap = r1 + r2 - dist
    
    if overlap <= 0:
        return False, 0.0, np.zeros(3), np.zeros(3)
    
    # Contact point (midpoint of overlap region)
    contact_point = pos1 + direction * (r1 - overlap/2)
    
    # Contact normal (approximate as center-to-center for stability)
    normal = direction
    
    return True, overlap, normal, contact_point


def compute_contact_force(overlap: float, normal: np.ndarray,
                          rel_velocity: np.ndarray,
                          shape1: GranuleShape, shape2: GranuleShape,
                          stiffness: float, damping: float) -> np.ndarray:
    """
    Compute contact force between two granules.
    
    Includes:
    - Normal repulsion (soft harmonic)
    - Velocity-dependent damping
    - Roughness-enhanced friction
    """
    if overlap <= 0:
        return np.zeros(3)
    
    # Normal force (soft repulsion)
    # Use geometric mean of radii for effective stiffness
    r_eff = np.sqrt(shape1.equivalent_radius * shape2.equivalent_radius)
    k_eff = stiffness * r_eff / 50.0  # Normalize by typical radius
    
    F_n_mag = k_eff * overlap
    
    # Damping (velocity dependent)
    v_n = np.dot(rel_velocity, normal)
    F_damp = -damping * v_n
    
    F_n_total = max(0, F_n_mag + F_damp)  # No adhesion
    
    # Tangential (friction) - simplified
    v_t = rel_velocity - v_n * normal
    v_t_mag = np.linalg.norm(v_t)
    
    # Friction coefficient increases with roughness
    mu_base = 0.2
    mu = mu_base + 0.5 * (shape1.roughness + shape2.roughness) / 2
    
    F_t_max = mu * F_n_total
    
    if v_t_mag > 1e-10:
        # Viscous friction (overdamped regime)
        F_t_mag = min(F_t_max, damping * v_t_mag)
        F_t = -F_t_mag * (v_t / v_t_mag)
    else:
        F_t = np.zeros(3)
    
    return F_n_total * normal + F_t


# =============================================================================
# JAMMED PACKING GENERATOR
# =============================================================================

class JammedPackingGenerator:
    """
    Generate jammed packings using Lubachevsky-Stillinger growth algorithm.
    
    Algorithm:
    1. Start with point particles at random positions
    2. Grow all particles at constant rate while running dynamics
    3. Stop when desired packing fraction / jamming is achieved
    4. Final relaxation to ensure mechanical equilibrium
    """
    
    def __init__(self, domain: np.ndarray, 
                 target_packing_fraction: float = 0.60,
                 verbose: bool = True):
        self.domain = domain
        self.target_phi = target_packing_fraction
        self.verbose = verbose
    
    def generate(self, 
                 n_granules: int,
                 shapes: List[GranuleShape],
                 growth_rate: float = 0.0005,
                 max_iterations: int = 200000) -> Tuple[np.ndarray, List[Quaternion]]:
        """
        Generate jammed packing.
        
        Args:
            n_granules: Number of granules
            shapes: List of GranuleShape for each granule
            growth_rate: Rate of size growth per iteration
            max_iterations: Maximum iterations
        
        Returns:
            positions: [n, 3] array of positions
            orientations: List of Quaternion orientations
        """
        if self.verbose:
            print(f"\nGenerating jammed packing for {n_granules} granules...")
            print(f"Target packing fraction: {self.target_phi:.2f}")
        
        # Initialize positions with good spacing using grid + noise
        positions = np.zeros((n_granules, 3))
        n_per_side = int(np.ceil(n_granules ** (1/3)))
        spacing = np.min(self.domain) / (n_per_side + 1)
        
        idx = 0
        for ix in range(n_per_side):
            for iy in range(n_per_side):
                for iz in range(n_per_side):
                    if idx >= n_granules:
                        break
                    positions[idx] = np.array([
                        (ix + 1) * spacing + np.random.uniform(-spacing*0.3, spacing*0.3),
                        (iy + 1) * spacing + np.random.uniform(-spacing*0.3, spacing*0.3),
                        (iz + 1) * spacing + np.random.uniform(-spacing*0.3, spacing*0.3)
                    ])
                    # Clamp to domain
                    positions[idx] = np.clip(positions[idx], 10, self.domain - 10)
                    idx += 1
                if idx >= n_granules:
                    break
            if idx >= n_granules:
                break
        
        # Initialize random orientations
        orientations = [Quaternion.random() for _ in range(n_granules)]
        
        # Current scale factor (start small)
        scale = 0.05
        
        # Parameters - more conservative for stability
        damping = 2.0
        stiffness = 0.5
        max_force_mag = 50.0  # Cap forces to prevent explosions
        
        # Growth loop
        iteration = 0
        current_phi = 0.0
        
        if self.verbose and HAS_TQDM:
            pbar = tqdm(total=self.target_phi, desc="Growing packing", unit="φ")
            last_phi = 0.0
        
        while current_phi < self.target_phi and iteration < max_iterations:
            # Compute forces
            forces = np.zeros((n_granules, 3))
            
            # Granule-granule contacts - use simple sphere approximation for speed
            for i in range(n_granules):
                ri = shapes[i].equivalent_radius * scale
                for j in range(i + 1, n_granules):
                    rj = shapes[j].equivalent_radius * scale
                    
                    r_ij = positions[j] - positions[i]
                    dist = np.linalg.norm(r_ij)
                    
                    if dist < 1e-6:
                        # Coincident - push apart randomly
                        r_ij = np.random.randn(3)
                        dist = np.linalg.norm(r_ij)
                    
                    overlap = ri + rj - dist
                    
                    if overlap > 0:
                        normal = r_ij / dist
                        F_mag = min(stiffness * overlap, max_force_mag)
                        F = F_mag * normal
                        forces[i] -= F
                        forces[j] += F
            
            # Wall forces
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                for d in range(3):
                    # Lower wall
                    if positions[i, d] < r:
                        overlap = r - positions[i, d]
                        forces[i, d] += min(stiffness * overlap, max_force_mag)
                    # Upper wall
                    if positions[i, d] > self.domain[d] - r:
                        overlap = positions[i, d] - (self.domain[d] - r)
                        forces[i, d] -= min(stiffness * overlap, max_force_mag)
            
            # Cap total force magnitude per particle
            for i in range(n_granules):
                f_mag = np.linalg.norm(forces[i])
                if f_mag > max_force_mag:
                    forces[i] = forces[i] * max_force_mag / f_mag
            
            # Update positions (overdamped, no velocities needed)
            dt = 0.5
            positions += forces * dt / damping
            
            # Check for NaN and fix
            nan_mask = np.any(np.isnan(positions) | np.isinf(positions), axis=1)
            if np.any(nan_mask):
                # Reset bad particles to random positions
                for i in np.where(nan_mask)[0]:
                    positions[i] = np.array([
                        np.random.uniform(50, self.domain[0] - 50),
                        np.random.uniform(50, self.domain[1] - 50),
                        np.random.uniform(50, self.domain[2] - 50)
                    ])
            
            # Clamp to domain
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                margin = max(r + 1, 5)
                positions[i] = np.clip(positions[i], margin, self.domain - margin)
            
            # Grow particles (slowly)
            scale += growth_rate
            
            # Compute current packing fraction
            total_volume = sum(s.volume * scale**3 for s in shapes)
            domain_volume = np.prod(self.domain)
            current_phi = total_volume / domain_volume
            
            # Reduce growth rate as we approach target
            if current_phi > 0.4 * self.target_phi:
                growth_rate = max(0.00005, growth_rate * 0.9995)
            
            iteration += 1
            
            # Update progress
            if self.verbose and HAS_TQDM and current_phi - last_phi > 0.005:
                pbar.update(current_phi - last_phi)
                last_phi = current_phi
        
        if self.verbose and HAS_TQDM:
            pbar.close()
        
        # Final relaxation with careful force limiting
        if self.verbose:
            print("Running final relaxation...")
        
        for relax_iter in range(2000):
            forces = np.zeros((n_granules, 3))
            max_overlap = 0.0
            
            for i in range(n_granules):
                ri = shapes[i].equivalent_radius * scale
                for j in range(i + 1, n_granules):
                    rj = shapes[j].equivalent_radius * scale
                    
                    r_ij = positions[j] - positions[i]
                    dist = np.linalg.norm(r_ij)
                    
                    if dist < 1e-6:
                        continue
                    
                    overlap = ri + rj - dist
                    
                    if overlap > 0:
                        max_overlap = max(max_overlap, overlap)
                        normal = r_ij / dist
                        F_mag = min(stiffness * overlap, max_force_mag * 0.5)
                        F = F_mag * normal
                        forces[i] -= F
                        forces[j] += F
            
            # Wall forces
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                for d in range(3):
                    if positions[i, d] < r:
                        forces[i, d] += min(stiffness * (r - positions[i, d]), max_force_mag)
                    if positions[i, d] > self.domain[d] - r:
                        forces[i, d] -= min(stiffness * (positions[i, d] - (self.domain[d] - r)), max_force_mag)
            
            # Move with small step
            positions += forces * 0.2 / damping
            
            # Clamp
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                positions[i] = np.clip(positions[i], r + 0.5, self.domain - r - 0.5)
            
            # Check convergence
            if max_overlap < 0.5 and relax_iter > 100:
                break
        
        if self.verbose:
            print(f"  Relaxation finished after {relax_iter + 1} iterations")
            print(f"  Max remaining overlap: {max_overlap:.2f} μm")
        
        # Compute final metrics
        n_contacts = 0
        for i in range(n_granules):
            ri = shapes[i].equivalent_radius * scale
            for j in range(i + 1, n_granules):
                rj = shapes[j].equivalent_radius * scale
                dist = np.linalg.norm(positions[j] - positions[i])
                if dist < ri + rj + 2.0:  # Small tolerance
                    n_contacts += 1
        
        mean_coordination = 2 * n_contacts / n_granules
        
        # Scale shapes to final size
        for i, shape in enumerate(shapes):
            shape.a *= scale
            shape.b *= scale
            shape.c *= scale
        
        if self.verbose:
            print(f"Packing complete:")
            print(f"  Final packing fraction: {current_phi:.3f}")
            print(f"  Mean coordination number: {mean_coordination:.2f}")
            print(f"  Scale factor: {scale:.3f}")
            print(f"  Iterations: {iteration}")
        
        # Final NaN check
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            raise ValueError("Packing generation failed - NaN in positions")
        
        return positions, orientations


# =============================================================================
# MAIN GRANULE SYSTEM
# =============================================================================

class GranuleType(Enum):
    FUNCTIONAL = 0
    INERT = 1


@dataclass
class NonSphericalGranuleSystem:
    """System of non-spherical granules with orientations."""
    n: int
    positions: np.ndarray          # [n, 3]
    velocities: np.ndarray         # [n, 3]
    orientations: List[Quaternion] # length n
    angular_velocities: np.ndarray # [n, 3]
    shapes: List[GranuleShape]     # length n
    types: np.ndarray              # [n] int
    n_cells: np.ndarray            # [n] int
    drag_coeffs: np.ndarray        # [n] translational drag
    rot_drag_coeffs: np.ndarray    # [n] rotational drag
    
    @classmethod
    def create(cls, n: int) -> 'NonSphericalGranuleSystem':
        return cls(
            n=n,
            positions=np.zeros((n, 3)),
            velocities=np.zeros((n, 3)),
            orientations=[Quaternion() for _ in range(n)],
            angular_velocities=np.zeros((n, 3)),
            shapes=[GranuleShape(1, 1, 1) for _ in range(n)],
            types=np.zeros(n, dtype=np.int32),
            n_cells=np.zeros(n, dtype=np.int32),
            drag_coeffs=np.ones(n),
            rot_drag_coeffs=np.ones(n)
        )
    
    @property
    def functional_mask(self) -> np.ndarray:
        return self.types == GranuleType.FUNCTIONAL.value
    
    @property
    def inert_mask(self) -> np.ndarray:
        return self.types == GranuleType.INERT.value


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class JammedSimConfig:
    """Configuration for jammed non-spherical granule simulation."""
    
    # === Granule counts ===
    n_functional: int = 100
    n_inert: int = 100
    
    # === Shape parameters ===
    # Functional granules
    functional_radius_mean: float = 40.0  # μm
    functional_radius_std: float = 8.0
    functional_aspect_ratio: Tuple[float, float] = (1.0, 1.8)
    functional_roundness: Tuple[float, float] = (2.0, 3.5)
    functional_roughness: Tuple[float, float] = (0.1, 0.4)
    
    # Inert granules
    inert_radius_mean: float = 60.0
    inert_radius_std: float = 12.0
    inert_aspect_ratio: Tuple[float, float] = (1.0, 1.5)
    inert_roundness: Tuple[float, float] = (2.0, 3.0)
    inert_roughness: Tuple[float, float] = (0.0, 0.3)
    
    # === Initial packing ===
    target_packing_fraction: float = 0.55
    auto_calculate_domain: bool = True  # If True, compute domain from granule volumes
    
    # === Cell properties ===
    cells_per_granule_mean: float = 5.0
    cell_diameter: float = 20.0
    cell_bridge_stiffness: float = 0.15
    max_bridge_gap: float = 50.0
    
    # === Mechanics ===
    repulsion_stiffness: float = 1.0
    damping: float = 1.5
    
    # === Domain (μm) - will be auto-calculated if auto_calculate_domain=True ===
    domain_x: float = 800.0
    domain_y: float = 800.0
    domain_z: float = 800.0
    wall_stiffness: float = 2.0
    
    # === Time ===
    total_time_hours: float = 72.0
    dt_initial: float = 0.01
    dt_min: float = 0.001
    dt_max: float = 0.5
    max_displacement_fraction: float = 0.03
    
    # === Output ===
    save_interval_hours: float = 2.0
    output_dir: str = "./jammed_nonspherical_output"
    verbose: bool = True
    
    @property
    def domain(self) -> np.ndarray:
        return np.array([self.domain_x, self.domain_y, self.domain_z])
    
    def calculate_domain_from_granules(self) -> Tuple[float, float, float]:
        """
        Calculate required domain size based on granule sizes and target packing fraction.
        
        Returns (domain_x, domain_y, domain_z) in μm
        """
        # Estimate total granule volume
        # V_sphere = (4/3) * π * r³
        
        # Functional granules volume
        r_func = self.functional_radius_mean
        v_func_single = (4/3) * np.pi * r_func**3
        v_func_total = self.n_functional * v_func_single
        
        # Inert granules volume
        r_inert = self.inert_radius_mean
        v_inert_single = (4/3) * np.pi * r_inert**3
        v_inert_total = self.n_inert * v_inert_single
        
        # Total granule volume
        v_total_granules = v_func_total + v_inert_total
        
        # Required domain volume for target packing fraction
        # packing_fraction = v_granules / v_domain
        # v_domain = v_granules / packing_fraction
        v_domain = v_total_granules / self.target_packing_fraction
        
        # For cubic domain: L = v_domain^(1/3)
        # Add 10% margin for wall effects
        L = v_domain ** (1/3) * 1.05
        
        return L, L, L
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {}
        for key in ['n_functional', 'n_inert', 'functional_radius_mean', 'functional_radius_std',
                    'inert_radius_mean', 'inert_radius_std', 'target_packing_fraction',
                    'cells_per_granule_mean', 'cell_diameter', 'cell_bridge_stiffness',
                    'max_bridge_gap', 'repulsion_stiffness', 'damping', 'domain_x', 'domain_y',
                    'domain_z', 'wall_stiffness', 'total_time_hours', 'dt_initial', 'dt_min',
                    'dt_max', 'max_displacement_fraction', 'save_interval_hours', 'output_dir',
                    'auto_calculate_domain']:
            d[key] = getattr(self, key)
        d['functional_aspect_ratio'] = list(self.functional_aspect_ratio)
        d['functional_roundness'] = list(self.functional_roundness)
        d['functional_roughness'] = list(self.functional_roughness)
        d['inert_aspect_ratio'] = list(self.inert_aspect_ratio)
        d['inert_roundness'] = list(self.inert_roundness)
        d['inert_roughness'] = list(self.inert_roughness)
        return d


# =============================================================================
# MAIN SIMULATION
# =============================================================================

class JammedNonSphericalSimulation:
    """Main simulation with jammed non-spherical granules."""
    
    def __init__(self, cfg: JammedSimConfig):
        self.cfg = cfg
        self.granules: Optional[NonSphericalGranuleSystem] = None
        self.time = 0.0
        self.step_count = 0
        self.dt = cfg.dt_initial
        
        self.history = {
            'time_hours': [],
            'n_contacts': [],
            'n_bridges': [],
            'mean_coordination': [],
            'max_velocity': [],
            'packing_fraction': []
        }
        
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup(self):
        """Initialize jammed packing of non-spherical granules."""
        cfg = self.cfg
        n_total = cfg.n_functional + cfg.n_inert
        
        if cfg.verbose:
            print("\n" + "="*60)
            print("Setting up Jammed Non-Spherical Granule System")
            print("="*60)
        
        # === Auto-calculate domain size if requested ===
        if cfg.auto_calculate_domain:
            new_domain = cfg.calculate_domain_from_granules()
            cfg.domain_x, cfg.domain_y, cfg.domain_z = new_domain
            if cfg.verbose:
                print(f"\nAuto-calculated domain size:")
                print(f"  Based on {cfg.n_functional} functional (R={cfg.functional_radius_mean}μm)")
                print(f"           {cfg.n_inert} inert (R={cfg.inert_radius_mean}μm)")
                print(f"  Target packing fraction: {cfg.target_packing_fraction}")
                print(f"  Domain: {cfg.domain_x:.1f} × {cfg.domain_y:.1f} × {cfg.domain_z:.1f} μm")
        
        # Create granule system
        self.granules = NonSphericalGranuleSystem.create(n_total)
        
        # === Create shapes FIRST (before assigning types) ===
        if cfg.verbose:
            print("\nGenerating granule shapes...")
        
        # Create all shapes in a list
        all_shapes = []
        all_types = []
        
        # Functional granules
        for i in range(cfg.n_functional):
            r = np.random.normal(cfg.functional_radius_mean, cfg.functional_radius_std)
            r = max(15.0, min(r, cfg.functional_radius_mean * 2))
            shape = create_random_shape(
                r,
                cfg.functional_aspect_ratio,
                cfg.functional_roundness,
                cfg.functional_roughness
            )
            all_shapes.append(shape)
            all_types.append(GranuleType.FUNCTIONAL.value)
        
        # Inert granules
        for i in range(cfg.n_inert):
            r = np.random.normal(cfg.inert_radius_mean, cfg.inert_radius_std)
            r = max(15.0, min(r, cfg.inert_radius_mean * 2))
            shape = create_random_shape(
                r,
                cfg.inert_aspect_ratio,
                cfg.inert_roundness,
                cfg.inert_roughness
            )
            all_shapes.append(shape)
            all_types.append(GranuleType.INERT.value)
        
        # === SHUFFLE to ensure well-mixed initial distribution ===
        shuffle_indices = np.random.permutation(n_total)
        shapes_shuffled = [all_shapes[i] for i in shuffle_indices]
        types_shuffled = np.array([all_types[i] for i in shuffle_indices])
        
        # Assign shuffled types
        self.granules.types = types_shuffled
        
        if cfg.verbose:
            print(f"  Shuffled granules for well-mixed initial distribution")
        
        # Print shape statistics
        if cfg.verbose:
            func_mask = types_shuffled == GranuleType.FUNCTIONAL.value
            inert_mask = types_shuffled == GranuleType.INERT.value
            
            func_shapes = [s for i, s in enumerate(shapes_shuffled) if func_mask[i]]
            inert_shapes = [s for i, s in enumerate(shapes_shuffled) if inert_mask[i]]
            
            func_sphericity = np.mean([s.sphericity for s in func_shapes])
            inert_sphericity = np.mean([s.sphericity for s in inert_shapes])
            func_aspect = np.mean([s.aspect_ratio for s in func_shapes])
            inert_aspect = np.mean([s.aspect_ratio for s in inert_shapes])
            func_radius = np.mean([s.equivalent_radius for s in func_shapes])
            inert_radius = np.mean([s.equivalent_radius for s in inert_shapes])
            
            print(f"  Functional: R_eq={func_radius:.1f}μm, sphericity={func_sphericity:.2f}, aspect={func_aspect:.2f}")
            print(f"  Inert: R_eq={inert_radius:.1f}μm, sphericity={inert_sphericity:.2f}, aspect={inert_aspect:.2f}")
        
        # === Generate jammed packing with well-mixed shapes ===
        shapes_for_packing = [
            GranuleShape(s.a, s.b, s.c, s.n, s.roughness) for s in shapes_shuffled
        ]
        
        packer = JammedPackingGenerator(
            cfg.domain,
            cfg.target_packing_fraction,
            cfg.verbose
        )
        
        positions, orientations = packer.generate(n_total, shapes_for_packing)
        
        # Use the scaled shapes from packing
        self.granules.shapes = shapes_for_packing
        self.granules.positions = positions
        self.granules.orientations = orientations
        
        # Assign cells to functional granules only
        func_mask = self.granules.types == GranuleType.FUNCTIONAL.value
        n_functional_actual = np.sum(func_mask)
        
        self.granules.n_cells = np.zeros(n_total, dtype=np.int32)
        self.granules.n_cells[func_mask] = np.maximum(
            1, np.random.poisson(cfg.cells_per_granule_mean, n_functional_actual)
        ).astype(np.int32)
        
        # Compute drag coefficients
        for i in range(n_total):
            r_eq = self.granules.shapes[i].equivalent_radius
            self.granules.drag_coeffs[i] = cfg.damping * max(r_eq, 10.0) / max(cfg.functional_radius_mean, 10.0)
            self.granules.rot_drag_coeffs[i] = cfg.damping * max(r_eq, 10.0)**3 / max(cfg.functional_radius_mean, 10.0)**3
        
        # === Calculate and report mixing quality ===
        if cfg.verbose:
            mixing_score = self._calculate_mixing_quality()
            print(f"\nInitial mixing quality: {mixing_score:.2f} (1.0 = perfectly mixed)")
        
        if cfg.verbose:
            print(f"\nSetup complete.")
            print(f"  Total granules: {n_total}")
            print(f"  Total cells: {np.sum(self.granules.n_cells)}")
    
    def _calculate_mixing_quality(self) -> float:
        """
        Calculate how well-mixed the functional and inert granules are.
        
        Returns a score from 0 to 1, where:
        - 1.0 = perfectly mixed (each granule's neighbors are random mix)
        - 0.0 = fully segregated (functional and inert in separate regions)
        
        Uses local neighborhood composition compared to global ratio.
        """
        n = self.granules.n
        positions = self.granules.positions
        types = self.granules.types
        
        # Global fraction of functional
        global_func_frac = np.sum(types == GranuleType.FUNCTIONAL.value) / n
        
        # For each granule, compute local functional fraction in neighborhood
        neighbor_radius = np.mean([s.equivalent_radius for s in self.granules.shapes]) * 4
        
        local_deviations = []
        
        for i in range(n):
            # Find neighbors within radius
            distances = np.linalg.norm(positions - positions[i], axis=1)
            neighbor_mask = (distances < neighbor_radius) & (distances > 0)
            
            if np.sum(neighbor_mask) > 0:
                neighbor_types = types[neighbor_mask]
                local_func_frac = np.sum(neighbor_types == GranuleType.FUNCTIONAL.value) / len(neighbor_types)
                deviation = abs(local_func_frac - global_func_frac)
                local_deviations.append(deviation)
        
        if len(local_deviations) == 0:
            return 1.0
        
        # Mean deviation from expected - lower is better
        mean_deviation = np.mean(local_deviations)
        
        # Convert to 0-1 score where 1 is good
        # Max possible deviation is max(global_func_frac, 1-global_func_frac)
        max_deviation = max(global_func_frac, 1 - global_func_frac)
        
        if max_deviation > 0:
            mixing_score = 1.0 - (mean_deviation / max_deviation)
        else:
            mixing_score = 1.0
        
        return max(0.0, min(1.0, mixing_score))
    
    def _find_contacts_and_bridges(self) -> Tuple[List, List]:
        """Find contact pairs and cell bridge pairs."""
        contacts = []
        bridges = []
        
        n = self.granules.n
        positions = self.granules.positions
        shapes = self.granules.shapes
        orientations = self.granules.orientations
        types = self.granules.types
        
        # Check for NaN/Inf and fix
        nan_mask = np.any(np.isnan(positions) | np.isinf(positions), axis=1)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} particles have NaN positions, resetting...")
            for i in np.where(nan_mask)[0]:
                positions[i] = np.array([
                    self.cfg.domain_x / 2 + np.random.uniform(-50, 50),
                    self.cfg.domain_y / 2 + np.random.uniform(-50, 50),
                    self.cfg.domain_z / 2 + np.random.uniform(-50, 50)
                ])
                self.granules.velocities[i] = np.zeros(3)
        
        # Use spatial hashing for efficiency
        if HAS_SCIPY:
            # Build KD-tree using bounding radii
            bounding_radii = np.array([s.bounding_radius for s in shapes])
            max_extent = 2 * np.max(bounding_radii) + self.cfg.max_bridge_gap
            
            try:
                tree = cKDTree(positions)
                pairs = tree.query_pairs(max_extent)
            except ValueError as e:
                print(f"KDTree error: {e}, falling back to brute force")
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        else:
            # Brute force
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for i, j in pairs:
            # Simple sphere-based contact for stability
            dist = np.linalg.norm(positions[j] - positions[i])
            
            if dist < 1e-6:
                continue
            
            r_i = shapes[i].equivalent_radius
            r_j = shapes[j].equivalent_radius
            overlap = r_i + r_j - dist
            
            if overlap > 0:
                normal = (positions[j] - positions[i]) / dist
                contact_pt = positions[i] + normal * (r_i - overlap/2)
                contacts.append((i, j, overlap, normal, contact_pt))
            
            # Check for cell bridge (functional-functional only)
            if (types[i] == GranuleType.FUNCTIONAL.value and
                types[j] == GranuleType.FUNCTIONAL.value):
                
                # Surface-to-surface distance
                gap = dist - r_i - r_j
                
                if gap < self.cfg.max_bridge_gap:
                    n_i = self.granules.n_cells[i]
                    n_j = self.granules.n_cells[j]
                    proximity = max(0, 1.0 - max(0, gap) / self.cfg.max_bridge_gap)
                    n_bridge = max(1, int(np.sqrt(n_i * n_j) * proximity))
                    
                    direction = (positions[j] - positions[i]) / dist
                    bridges.append((i, j, n_bridge, dist, direction))
        
        return contacts, bridges
    
    def _compute_forces_and_torques(self, contacts, bridges):
        """Compute forces and torques on all granules."""
        n = self.granules.n
        forces = np.zeros((n, 3))
        torques = np.zeros((n, 3))
        
        cfg = self.cfg
        max_force = 100.0  # Cap forces to prevent instability
        
        # Contact forces
        for i, j, overlap, normal, contact_pt in contacts:
            rel_vel = self.granules.velocities[j] - self.granules.velocities[i]
            
            # Simple harmonic repulsion with damping
            shape_i = self.granules.shapes[i]
            shape_j = self.granules.shapes[j]
            
            r_eff = np.sqrt(shape_i.equivalent_radius * shape_j.equivalent_radius)
            k_eff = cfg.repulsion_stiffness * r_eff / 50.0
            
            F_mag = k_eff * overlap
            
            # Damping
            v_n = np.dot(rel_vel, normal)
            F_damp = -cfg.damping * 0.5 * v_n
            
            F_total = max(0, F_mag + F_damp)
            F_total = min(F_total, max_force)  # Cap force
            
            F = F_total * normal
            
            forces[i] -= F
            forces[j] += F
            
            # Simplified torque (skip for stability)
            # r_i = contact_pt - self.granules.positions[i]
            # torques[i] -= np.cross(r_i, F) * 0.1  # Reduced torque coupling
        
        # Cell bridge forces
        rest_length = cfg.cell_diameter
        k_cell = cfg.cell_bridge_stiffness
        
        for i, j, n_cells, dist, direction in bridges:
            r_i = self.granules.shapes[i].equivalent_radius
            r_j = self.granules.shapes[j].equivalent_radius
            
            gap = dist - r_i - r_j
            extension = gap - rest_length
            
            # Cells pull (contract)
            F_mag = k_cell * n_cells * extension
            if extension < 0:
                F_mag *= 0.1  # Weak push resistance
            
            # Cap bridge force too
            F_mag = np.clip(F_mag, -max_force * 0.5, max_force * 0.5)
            
            # Add noise
            noise = np.clip(1.0 + 0.15 * np.random.randn(), 0.5, 1.5)
            F_mag *= noise
            
            F = F_mag * direction
            forces[i] += F
            forces[j] -= F
        
        # Wall forces
        for i in range(n):
            r = self.granules.shapes[i].bounding_radius
            pos = self.granules.positions[i]
            
            for d in range(3):
                if pos[d] < r:
                    overlap = r - pos[d]
                    forces[i, d] += min(cfg.wall_stiffness * overlap, max_force)
                if pos[d] > cfg.domain[d] - r:
                    overlap = pos[d] - (cfg.domain[d] - r)
                    forces[i, d] -= min(cfg.wall_stiffness * overlap, max_force)
        
        # Activity noise on functional granules
        func_mask = self.granules.functional_mask
        noise_scale = 0.03 * np.sqrt(self.granules.n_cells + 1)
        forces[func_mask] += noise_scale[func_mask, np.newaxis] * np.random.randn(np.sum(func_mask), 3)
        
        # Final force capping per particle
        for i in range(n):
            f_mag = np.linalg.norm(forces[i])
            if f_mag > max_force:
                forces[i] = forces[i] * max_force / f_mag
        
        return forces, torques
    
    def step(self):
        """Perform one simulation step."""
        # Find interactions
        contacts, bridges = self._find_contacts_and_bridges()
        
        # Compute forces and torques
        forces, torques = self._compute_forces_and_torques(contacts, bridges)
        
        # Overdamped dynamics: v = F / γ
        velocities = forces / self.granules.drag_coeffs[:, np.newaxis]
        angular_vels = torques / self.granules.rot_drag_coeffs[:, np.newaxis]
        
        # Cap velocities
        max_vel = 100.0  # μm/hr
        for i in range(self.granules.n):
            v_mag = np.linalg.norm(velocities[i])
            if v_mag > max_vel:
                velocities[i] = velocities[i] * max_vel / v_mag
        
        # Adaptive timestep
        max_vel_actual = np.max(np.linalg.norm(velocities, axis=1))
        if max_vel_actual > 1e-10:
            min_r = min(s.min_radius for s in self.granules.shapes)
            max_disp = self.cfg.max_displacement_fraction * min_r
            dt_vel = max_disp / max_vel_actual
            self.dt = np.clip(dt_vel, self.cfg.dt_min, self.cfg.dt_max)
        else:
            self.dt = self.cfg.dt_max
        
        # Update positions
        self.granules.positions += velocities * self.dt
        self.granules.velocities = velocities
        
        # Update orientations (skip for now - simplifies stability)
        # for i in range(self.granules.n):
        #     self.granules.orientations[i] = self.granules.orientations[i].integrate(
        #         angular_vels[i], self.dt
        #     )
        self.granules.angular_velocities = angular_vels
        
        # Boundary enforcement - hard clamp
        cfg = self.cfg
        for i in range(self.granules.n):
            r = self.granules.shapes[i].bounding_radius
            self.granules.positions[i] = np.clip(
                self.granules.positions[i],
                r + 1.0, cfg.domain - r - 1.0
            )
        
        # Check for NaN and reset if needed
        nan_mask = np.any(np.isnan(self.granules.positions) | np.isinf(self.granules.positions), axis=1)
        if np.any(nan_mask):
            print(f"Warning: {np.sum(nan_mask)} particles have NaN positions")
            for i in np.where(nan_mask)[0]:
                self.granules.positions[i] = np.array([
                    cfg.domain_x / 2 + np.random.uniform(-20, 20),
                    cfg.domain_y / 2 + np.random.uniform(-20, 20),
                    cfg.domain_z / 2 + np.random.uniform(-20, 20)
                ])
                self.granules.velocities[i] = np.zeros(3)
        
        self.time += self.dt
        self.step_count += 1
        
        return contacts, bridges
    
    def _record_history(self, contacts, bridges):
        """Record metrics."""
        n = self.granules.n
        
        # Coordination number
        coord = np.zeros(n)
        for i, j, *_ in contacts:
            coord[i] += 1
            coord[j] += 1
        
        # Packing fraction
        total_vol = sum(s.volume for s in self.granules.shapes)
        domain_vol = np.prod(self.cfg.domain)
        phi = total_vol / domain_vol
        
        self.history['time_hours'].append(self.time)
        self.history['n_contacts'].append(len(contacts))
        self.history['n_bridges'].append(len(bridges))
        self.history['mean_coordination'].append(np.mean(coord))
        self.history['max_velocity'].append(np.max(np.linalg.norm(self.granules.velocities, axis=1)))
        self.history['packing_fraction'].append(phi)
    
    def save_frame(self, label: str = None):
        """Save current state."""
        if label is None:
            label = f"t{self.time:.1f}h"
        
        filename = f"{self.cfg.output_dir}/frame_{label}.json"
        
        # Calculate mean radius for visualization
        mean_radius = np.mean([s.equivalent_radius for s in self.granules.shapes])
        
        data = {
            'time_hours': self.time,
            'n_granules': self.granules.n,
            'positions': self.granules.positions.tolist(),
            'orientations': [q.q.tolist() for q in self.granules.orientations],
            'shapes': [
                {'a': s.a, 'b': s.b, 'c': s.c, 'n': s.n, 'roughness': s.roughness}
                for s in self.granules.shapes
            ],
            'types': self.granules.types.tolist(),
            'n_cells': self.granules.n_cells.tolist(),
            'mean_radius': mean_radius,
            'domain': self.cfg.domain.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def run(self):
        """Run the simulation."""
        cfg = self.cfg
        
        if self.granules is None:
            self.setup()
        
        if cfg.verbose:
            print(f"\n{'='*60}")
            print("Starting Simulation")
            print(f"{'='*60}")
            print(f"Target time: {cfg.total_time_hours:.0f} hours ({cfg.total_time_hours/24:.1f} days)")
        
        start_wall = time_module.time()
        last_save = 0.0
        last_print = 0.0
        
        self.save_frame("initial")
        
        if HAS_TQDM and cfg.verbose:
            pbar = tqdm(total=cfg.total_time_hours, desc="Simulating", unit="hr")
            pbar_time = 0.0
        
        while self.time < cfg.total_time_hours:
            contacts, bridges = self.step()
            
            if self.time - last_print >= 0.1:
                self._record_history(contacts, bridges)
                last_print = self.time
            
            if self.time - last_save >= cfg.save_interval_hours:
                self.save_frame()
                last_save = self.time
                
                if cfg.verbose and not HAS_TQDM:
                    coord = len(contacts) * 2 / self.granules.n if self.granules.n > 0 else 0
                    print(f"  t={self.time:.1f}h | contacts={len(contacts)} | "
                          f"bridges={len(bridges)} | Z={coord:.1f}")
            
            if HAS_TQDM and cfg.verbose:
                pbar.update(self.time - pbar_time)
                pbar_time = self.time
        
        if HAS_TQDM and cfg.verbose:
            pbar.close()
        
        self.save_frame("final")
        
        with open(f"{cfg.output_dir}/history.json", 'w') as f:
            json.dump(self.history, f)
        
        # Save config
        try:
            with open(f"{cfg.output_dir}/config.json", 'w') as f:
                json.dump(cfg.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
        
        elapsed = time_module.time() - start_wall
        
        if cfg.verbose:
            print(f"\n{'='*60}")
            print("Simulation Complete")
            print(f"{'='*60}")
            print(f"Simulated: {self.time:.1f} hours ({self.time/24:.1f} days)")
            print(f"Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            if elapsed > 0:
                print(f"Speedup: {self.time*3600/elapsed:.0f}x")
        
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("JAMMED NON-SPHERICAL GRANULE DEM SIMULATION")
    print("="*70)
    
    cfg = JammedSimConfig(
        # Granule counts
        n_functional=200,
        n_inert=200,
        
        # Functional: smaller, more irregular
        functional_radius_mean=15.0,
        functional_radius_std=2.0,
        functional_aspect_ratio=(1.0, 2),
        functional_roundness=(2.0, 3.0),
        functional_roughness=(0.5, 1),
        
        # Inert: larger, more spherical
        inert_radius_mean=20.0,
        inert_radius_std=2.0,
        inert_aspect_ratio=(1.0, 2),
        inert_roundness=(2.0, 3),
        inert_roughness=(0.5, 1),
        
        # Jammed packing
        target_packing_fraction=0.52,
        auto_calculate_domain=True,  # Auto-compute domain size!
        
        # === Cell properties ===
        cells_per_granule_mean = 15.0,
        cell_diameter = 20.0,
        cell_bridge_stiffness = 0.15,
        max_bridge_gap = 50.0,

        # Time
        total_time_hours=72.0,
        save_interval_hours=1.0,  # Save every 1 hour for smooth animation
        
        # Mechanics
        repulsion_stiffness=1.0,
        damping=1.15,
        
        output_dir="./jammed_nonspherical_sim_1",
        verbose=True
    )
    
    # Show what domain will be calculated
    if cfg.auto_calculate_domain:
        predicted_domain = cfg.calculate_domain_from_granules()
        print(f"\nPredicted domain size: {predicted_domain[0]:.1f} × {predicted_domain[1]:.1f} × {predicted_domain[2]:.1f} μm")
    
    sim = JammedNonSphericalSimulation(cfg)
    history = sim.run()
    
    # Plot results
    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        t = np.array(history['time_hours']) / 24
        
        axes[0, 0].plot(t, history['n_contacts'], 'b-', label='Contacts')
        axes[0, 0].plot(t, history['n_bridges'], 'r-', label='Bridges')
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].set_title('Interactions')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, history['mean_coordination'], 'g-')
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Mean Coordination')
        axes[0, 1].set_title('Contact Network')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].semilogy(t, history['max_velocity'], 'm-')
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Max Velocity (μm/hr)')
        axes[1, 0].set_title('Activity')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t, history['packing_fraction'], 'c-')
        axes[1, 1].set_xlabel('Time (days)')
        axes[1, 1].set_ylabel('Packing Fraction')
        axes[1, 1].set_title('Packing Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{cfg.output_dir}/history.png", dpi=150)
        print(f"\nSaved plot to {cfg.output_dir}/history.png")
        plt.close()
    
    return sim, history


if __name__ == "__main__":
    sim, history = main()