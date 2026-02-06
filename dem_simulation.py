"""
Coarse-Grained DEM Simulation with JSON Configuration
======================================================

Features:
- JSON config file for all parameters
- Dialog box for config file selection
- Cell count based on surface area
- Granule counts calculated from volume requirements
- True shape storage for visualization

Requirements:
    pip install numpy scipy tqdm

Usage:
    python dem_simulation.py                    # Opens file dialog
    python dem_simulation.py config.json        # Uses specified config
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
from pathlib import Path
import json
import sys
import time as time_module

try:
    import tkinter as tk
    from tkinter import filedialog
    HAS_TK = True
except ImportError:
    HAS_TK = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# CONFIGURATION LOADER
# =============================================================================

def select_config_file() -> Optional[str]:
    """Open dialog to select config JSON file."""
    if not HAS_TK:
        print("tkinter not available. Please specify config file as argument.")
        return None
    
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filepath = filedialog.askopenfilename(
        title="Select Simulation Config File",
        filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        initialdir="."
    )
    
    root.destroy()
    return filepath if filepath else None


def load_config(filepath: str) -> Dict[str, Any]:
    """Load and validate configuration from JSON file."""
    with open(filepath) as f:
        config = json.load(f)
    
    # Set defaults for any missing values
    defaults = {
        "simulation_name": "unnamed_sim",
        "domain": {"side_length_um": 500.0, "target_packing_fraction": 0.55},
        "granule_ratio": {"functional_fraction": 0.5},
        "functional_granules": {
            "radius_mean_um": 40.0, "radius_std_um": 8.0,
            "aspect_ratio_range": [1.0, 1.5], "roundness_range": [2.0, 3.0],
            "roughness_range": [0.1, 0.3]
        },
        "inert_granules": {
            "radius_mean_um": 50.0, "radius_std_um": 10.0,
            "aspect_ratio_range": [1.0, 1.3], "roundness_range": [2.0, 2.5],
            "roughness_range": [0.0, 0.2]
        },
        "cell_properties": {
            "diameter_um": 20.0, "attachment_area_fraction": 0.5,
            "force_per_cell_nN": 5.0, "max_bridge_gap_um": 50.0
        },
        "mechanics": {"repulsion_stiffness": 1.0, "damping": 1.5, "wall_stiffness": 2.0},
        "time": {
            "total_hours": 72.0, "save_interval_hours": 2.0,
            "dt_initial_hours": 0.01, "dt_min_hours": 0.001, "dt_max_hours": 0.5
        },
        "output": {"base_directory": "./simulations", "save_true_shapes": True}
    }
    
    def merge_defaults(cfg, defs):
        for key, value in defs.items():
            if key not in cfg:
                cfg[key] = value
            elif isinstance(value, dict) and isinstance(cfg[key], dict):
                merge_defaults(cfg[key], value)
    
    merge_defaults(config, defaults)
    return config


def calculate_granule_counts(config: Dict) -> Tuple[int, int]:
    """
    Calculate number of functional and inert granules needed to fill
    the domain at the target packing fraction.
    """
    domain_side = config["domain"]["side_length_um"]
    target_phi = config["domain"]["target_packing_fraction"]
    func_fraction = config["granule_ratio"]["functional_fraction"]
    
    domain_volume = domain_side ** 3
    total_granule_volume = domain_volume * target_phi
    
    # Volume for each type
    func_volume = total_granule_volume * func_fraction
    inert_volume = total_granule_volume * (1 - func_fraction)
    
    # Mean granule volumes
    r_func = config["functional_granules"]["radius_mean_um"]
    r_inert = config["inert_granules"]["radius_mean_um"]
    
    v_func_single = (4/3) * np.pi * r_func**3
    v_inert_single = (4/3) * np.pi * r_inert**3
    
    n_functional = max(1, int(np.round(func_volume / v_func_single)))
    n_inert = max(1, int(np.round(inert_volume / v_inert_single)))
    
    return n_functional, n_inert


def calculate_cells_on_granule(granule_surface_area: float, config: Dict) -> int:
    """
    Calculate number of cells that attach to a granule based on surface area.
    
    cells = ceil(granule_area * attachment_fraction / cell_area)
    """
    cell_diameter = config["cell_properties"]["diameter_um"]
    attachment_fraction = config["cell_properties"]["attachment_area_fraction"]
    
    # Cell attachment footprint (circular)
    cell_area = np.pi * (cell_diameter / 2) ** 2
    
    # Available area for cells
    available_area = granule_surface_area * attachment_fraction
    
    # Number of cells (round up)
    n_cells = int(np.ceil(available_area / cell_area))
    return max(1, n_cells)


# =============================================================================
# SHAPE CLASSES
# =============================================================================

class GranuleType(Enum):
    FUNCTIONAL = 0
    INERT = 1


@dataclass
class GranuleShape:
    """Ellipsoid shape for simulation."""
    a: float  # x semi-axis
    b: float  # y semi-axis
    c: float  # z semi-axis
    n: float = 2.0  # roundness exponent
    roughness: float = 0.0
    
    @property
    def equivalent_radius(self) -> float:
        V = (4/3) * np.pi * self.a * self.b * self.c
        return (3 * V / (4 * np.pi)) ** (1/3)
    
    @property
    def volume(self) -> float:
        return (4/3) * np.pi * self.a * self.b * self.c
    
    @property
    def surface_area(self) -> float:
        """Approximate surface area using Knud Thomsen formula."""
        p = 1.6075
        ap, bp, cp = self.a**p, self.b**p, self.c**p
        return 4 * np.pi * ((ap*bp + ap*cp + bp*cp) / 3) ** (1/p)
    
    @property
    def bounding_radius(self) -> float:
        return max(self.a, self.b, self.c)
    
    def to_dict(self) -> Dict:
        return {"a": self.a, "b": self.b, "c": self.c, "n": self.n, "roughness": self.roughness}


@dataclass
class TrueShape:
    """
    True granule shape for visualization (superellipsoid with surface details).
    Stored separately from simulation shape.
    """
    a: float
    b: float
    c: float
    n: float  # roundness
    roughness: float
    surface_points: Optional[np.ndarray] = None  # Pre-computed surface mesh
    
    def generate_surface_mesh(self, resolution: int = 20) -> np.ndarray:
        """Generate surface points for visualization."""
        theta = np.linspace(0, 2*np.pi, resolution)
        phi = np.linspace(0, np.pi, resolution//2)
        theta, phi = np.meshgrid(theta, phi)
        
        # Superellipsoid parameterization
        def spow(x, p):
            return np.sign(x) * np.abs(x) ** p
        
        e = 2.0 / self.n
        x = self.a * spow(np.cos(theta), e) * spow(np.sin(phi), e)
        y = self.b * spow(np.sin(theta), e) * spow(np.sin(phi), e)
        z = self.c * spow(np.cos(phi), e)
        
        # Add surface roughness
        if self.roughness > 0:
            noise = np.random.randn(*x.shape) * self.roughness * min(self.a, self.b, self.c) * 0.1
            r = np.sqrt(x**2 + y**2 + z**2)
            x += noise * x / (r + 1e-6)
            y += noise * y / (r + 1e-6)
            z += noise * z / (r + 1e-6)
        
        self.surface_points = np.stack([x, y, z], axis=-1)
        return self.surface_points
    
    def to_dict(self) -> Dict:
        return {
            "a": self.a, "b": self.b, "c": self.c, 
            "n": self.n, "roughness": self.roughness,
            "has_surface_mesh": self.surface_points is not None
        }


def create_granule_shapes(mean_radius: float, config_section: Dict) -> Tuple[GranuleShape, TrueShape]:
    """Create both simulation shape and true shape for a granule."""
    aspect_range = config_section["aspect_ratio_range"]
    round_range = config_section["roundness_range"]
    rough_range = config_section["roughness_range"]
    
    aspect_ratio = np.random.uniform(*aspect_range)
    n = np.random.uniform(*round_range)
    roughness = np.random.uniform(*rough_range)
    
    # Create ellipsoid axes
    c = mean_radius * aspect_ratio**(1/3)
    ab = mean_radius / aspect_ratio**(1/6)
    asymmetry = np.random.uniform(0.95, 1.05)
    a = ab * asymmetry
    b = ab / asymmetry
    
    # Scale to target volume
    target_vol = (4/3) * np.pi * mean_radius**3
    current_vol = (4/3) * np.pi * a * b * c
    if current_vol > 0:
        scale = (target_vol / current_vol)**(1/3)
        a, b, c = a*scale, b*scale, c*scale
    
    # Simulation shape (uses equivalent sphere for contacts)
    sim_shape = GranuleShape(a=a, b=b, c=c, n=n, roughness=roughness)
    
    # True shape (stores actual geometry)
    true_shape = TrueShape(a=a, b=b, c=c, n=n, roughness=roughness)
    
    return sim_shape, true_shape


# =============================================================================
# QUATERNION
# =============================================================================

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.q = np.array([w, x, y, z], dtype=np.float64)
        self._normalize()
    
    def _normalize(self):
        norm = np.linalg.norm(self.q)
        if norm > 1e-10:
            self.q /= norm
    
    @classmethod
    def random(cls):
        u = np.random.random(3)
        return cls(
            np.sqrt(1-u[0]) * np.sin(2*np.pi*u[1]),
            np.sqrt(1-u[0]) * np.cos(2*np.pi*u[1]),
            np.sqrt(u[0]) * np.sin(2*np.pi*u[2]),
            np.sqrt(u[0]) * np.cos(2*np.pi*u[2])
        )
    
    def to_matrix(self) -> np.ndarray:
        w, x, y, z = self.q
        return np.array([
            [1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
            [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
            [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]
        ])


# =============================================================================
# GRANULE SYSTEM
# =============================================================================

class GranuleSystem:
    def __init__(self, n: int):
        self.n = n
        self.positions = np.zeros((n, 3))
        self.velocities = np.zeros((n, 3))
        self.orientations = [Quaternion() for _ in range(n)]
        self.sim_shapes: List[GranuleShape] = []
        self.true_shapes: List[TrueShape] = []
        self.types = np.zeros(n, dtype=np.int32)
        self.n_cells = np.zeros(n, dtype=np.int32)
        self.drag_coeffs = np.ones(n)


# =============================================================================
# PACKING GENERATOR
# =============================================================================

class PackingGenerator:
    def __init__(self, domain_size: float, target_phi: float, verbose: bool = True):
        self.domain = np.array([domain_size, domain_size, domain_size])
        self.target_phi = target_phi
        self.verbose = verbose
    
    def generate(self, n_granules: int, shapes: List[GranuleShape]) -> Tuple[np.ndarray, List[Quaternion]]:
        if self.verbose:
            print(f"\nGenerating jammed packing for {n_granules} granules...")
        
        # Grid initialization
        positions = np.zeros((n_granules, 3))
        n_side = int(np.ceil(n_granules ** (1/3)))
        spacing = min(self.domain) / (n_side + 1)
        
        idx = 0
        for ix in range(n_side):
            for iy in range(n_side):
                for iz in range(n_side):
                    if idx >= n_granules:
                        break
                    positions[idx] = np.array([
                        (ix + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2),
                        (iy + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2),
                        (iz + 1) * spacing + np.random.uniform(-spacing*0.2, spacing*0.2)
                    ])
                    positions[idx] = np.clip(positions[idx], 20, self.domain - 20)
                    idx += 1
                if idx >= n_granules:
                    break
            if idx >= n_granules:
                break
        
        orientations = [Quaternion.random() for _ in range(n_granules)]
        
        # Growth algorithm
        scale = 0.05
        growth_rate = 0.0005
        stiffness, damping = 0.5, 2.0
        max_force = 50.0
        current_phi = 0.0
        
        if self.verbose and HAS_TQDM:
            pbar = tqdm(total=self.target_phi, desc="Growing packing", unit="φ")
            last_phi = 0.0
        
        for iteration in range(200000):
            if current_phi >= self.target_phi:
                break
            
            forces = np.zeros((n_granules, 3))
            
            # Granule-granule repulsion
            for i in range(n_granules):
                ri = shapes[i].equivalent_radius * scale
                for j in range(i + 1, n_granules):
                    rj = shapes[j].equivalent_radius * scale
                    r_ij = positions[j] - positions[i]
                    dist = np.linalg.norm(r_ij)
                    if dist < 1e-6:
                        r_ij = np.random.randn(3)
                        dist = np.linalg.norm(r_ij)
                    overlap = ri + rj - dist
                    if overlap > 0:
                        F = min(stiffness * overlap, max_force) * r_ij / dist
                        forces[i] -= F
                        forces[j] += F
            
            # Wall forces
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                for d in range(3):
                    if positions[i, d] < r:
                        forces[i, d] += min(stiffness * (r - positions[i, d]), max_force)
                    if positions[i, d] > self.domain[d] - r:
                        forces[i, d] -= min(stiffness * (positions[i, d] - (self.domain[d] - r)), max_force)
            
            # Cap forces and update
            for i in range(n_granules):
                f_mag = np.linalg.norm(forces[i])
                if f_mag > max_force:
                    forces[i] *= max_force / f_mag
            
            positions += forces * 0.5 / damping
            
            # Clamp and check NaN
            for i in range(n_granules):
                if np.any(np.isnan(positions[i])):
                    positions[i] = self.domain / 2 + np.random.randn(3) * 10
                r = shapes[i].bounding_radius * scale
                positions[i] = np.clip(positions[i], r + 1, self.domain - r - 1)
            
            scale += growth_rate
            total_vol = sum(s.volume * scale**3 for s in shapes)
            current_phi = total_vol / np.prod(self.domain)
            
            if current_phi > 0.4 * self.target_phi:
                growth_rate = max(0.00005, growth_rate * 0.9995)
            
            if self.verbose and HAS_TQDM and current_phi - last_phi > 0.005:
                pbar.update(current_phi - last_phi)
                last_phi = current_phi
        
        if self.verbose and HAS_TQDM:
            pbar.close()
        
        # Final relaxation
        if self.verbose:
            print("Final relaxation...")
        
        for _ in range(1000):
            forces = np.zeros((n_granules, 3))
            max_overlap = 0
            
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
                        F = min(stiffness * overlap, max_force * 0.5) * r_ij / dist
                        forces[i] -= F
                        forces[j] += F
            
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                for d in range(3):
                    if positions[i, d] < r:
                        forces[i, d] += stiffness * (r - positions[i, d])
                    if positions[i, d] > self.domain[d] - r:
                        forces[i, d] -= stiffness * (positions[i, d] - (self.domain[d] - r))
            
            positions += forces * 0.2 / damping
            for i in range(n_granules):
                r = shapes[i].bounding_radius * scale
                positions[i] = np.clip(positions[i], r + 0.5, self.domain - r - 0.5)
            
            if max_overlap < 0.5:
                break
        
        # Scale shapes
        for s in shapes:
            s.a *= scale
            s.b *= scale
            s.c *= scale
        
        if self.verbose:
            print(f"  Final packing fraction: {current_phi:.3f}")
        
        return positions, orientations


# =============================================================================
# MAIN SIMULATION
# =============================================================================

class Simulation:
    def __init__(self, config: Dict, config_filepath: str):
        self.config = config
        self.config_name = Path(config_filepath).stem
        
        # Setup output directory
        base_dir = Path(config["output"]["base_directory"])
        self.output_dir = base_dir / self.config_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy config to output
        with open(self.output_dir / f"{self.config_name}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        self.domain_size = config["domain"]["side_length_um"]
        self.domain = np.array([self.domain_size] * 3)
        
        # Calculate granule counts
        self.n_functional, self.n_inert = calculate_granule_counts(config)
        self.n_total = self.n_functional + self.n_inert
        
        print(f"\nSimulation: {self.config_name}")
        print(f"  Domain: {self.domain_size:.0f}³ μm")
        print(f"  Granules: {self.n_functional} functional + {self.n_inert} inert = {self.n_total}")
        
        self.granules: Optional[GranuleSystem] = None
        self.time = 0.0
        self.step_count = 0
        self.dt = config["time"]["dt_initial_hours"]
        
        self.history = {
            'time_hours': [], 'n_contacts': [], 'n_bridges': [],
            'mean_coordination': [], 'max_velocity': [], 'packing_fraction': []
        }
    
    def setup(self):
        """Initialize granule system."""
        config = self.config
        
        self.granules = GranuleSystem(self.n_total)
        
        # Create shapes and assign types (shuffled for good mixing)
        all_sim_shapes = []
        all_true_shapes = []
        all_types = []
        all_n_cells = []
        
        # Functional granules
        func_config = config["functional_granules"]
        for _ in range(self.n_functional):
            r = np.random.normal(func_config["radius_mean_um"], func_config["radius_std_um"])
            r = max(15.0, min(r, func_config["radius_mean_um"] * 2))
            
            sim_shape, true_shape = create_granule_shapes(r, func_config)
            all_sim_shapes.append(sim_shape)
            all_true_shapes.append(true_shape)
            all_types.append(GranuleType.FUNCTIONAL.value)
            
            # Calculate cells based on surface area
            n_cells = calculate_cells_on_granule(sim_shape.surface_area, config)
            all_n_cells.append(n_cells)
        
        # Inert granules
        inert_config = config["inert_granules"]
        for _ in range(self.n_inert):
            r = np.random.normal(inert_config["radius_mean_um"], inert_config["radius_std_um"])
            r = max(15.0, min(r, inert_config["radius_mean_um"] * 2))
            
            sim_shape, true_shape = create_granule_shapes(r, inert_config)
            all_sim_shapes.append(sim_shape)
            all_true_shapes.append(true_shape)
            all_types.append(GranuleType.INERT.value)
            all_n_cells.append(0)  # No cells on inert
        
        # Shuffle for good mixing
        indices = np.random.permutation(self.n_total)
        self.granules.sim_shapes = [all_sim_shapes[i] for i in indices]
        self.granules.true_shapes = [all_true_shapes[i] for i in indices]
        self.granules.types = np.array([all_types[i] for i in indices])
        self.granules.n_cells = np.array([all_n_cells[i] for i in indices])
        
        # Generate jammed packing
        packer = PackingGenerator(
            self.domain_size,
            config["domain"]["target_packing_fraction"],
            verbose=True
        )
        positions, orientations = packer.generate(self.n_total, self.granules.sim_shapes)
        
        self.granules.positions = positions
        self.granules.orientations = orientations
        
        # Set drag coefficients
        for i in range(self.n_total):
            r_eq = self.granules.sim_shapes[i].equivalent_radius
            self.granules.drag_coeffs[i] = config["mechanics"]["damping"] * r_eq / 50.0
        
        # Generate true shape meshes if requested
        if config["output"]["save_true_shapes"]:
            print("Generating true shape meshes...")
            for ts in self.granules.true_shapes:
                ts.generate_surface_mesh(resolution=16)
        
        total_cells = np.sum(self.granules.n_cells)
        print(f"  Total cells: {total_cells}")
        print(f"  Mean cells per functional: {total_cells / self.n_functional:.1f}")
    
    def _find_contacts_and_bridges(self):
        contacts, bridges = [], []
        positions = self.granules.positions
        shapes = self.granules.sim_shapes
        types = self.granules.types
        n = self.n_total
        
        # Check for NaN
        nan_mask = np.any(np.isnan(positions) | np.isinf(positions), axis=1)
        if np.any(nan_mask):
            for i in np.where(nan_mask)[0]:
                positions[i] = self.domain / 2 + np.random.randn(3) * 10
                self.granules.velocities[i] = 0
        
        max_r = max(s.bounding_radius for s in shapes)
        max_bridge = self.config["cell_properties"]["max_bridge_gap_um"]
        cutoff = 2 * max_r + max_bridge
        
        if HAS_SCIPY:
            try:
                tree = cKDTree(positions)
                pairs = tree.query_pairs(cutoff)
            except:
                pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for i, j in pairs:
            r_ij = positions[j] - positions[i]
            dist = np.linalg.norm(r_ij)
            if dist < 1e-6:
                continue
            
            ri = shapes[i].equivalent_radius
            rj = shapes[j].equivalent_radius
            overlap = ri + rj - dist
            
            if overlap > 0:
                normal = r_ij / dist
                contacts.append((i, j, overlap, normal))
            
            if types[i] == 0 and types[j] == 0:  # Both functional
                gap = dist - ri - rj
                if gap < max_bridge:
                    ni, nj = self.granules.n_cells[i], self.granules.n_cells[j]
                    proximity = max(0, 1 - max(0, gap) / max_bridge)
                    n_bridge = max(1, int(np.sqrt(ni * nj) * proximity))
                    bridges.append((i, j, n_bridge, dist, r_ij / dist))
        
        return contacts, bridges
    
    def _compute_forces(self, contacts, bridges):
        n = self.n_total
        forces = np.zeros((n, 3))
        config = self.config
        
        k_rep = config["mechanics"]["repulsion_stiffness"]
        k_wall = config["mechanics"]["wall_stiffness"]
        damping = config["mechanics"]["damping"]
        cell_force = config["cell_properties"]["force_per_cell_nN"]
        cell_diameter = config["cell_properties"]["diameter_um"]
        max_force = 100.0
        
        # Contact forces
        for i, j, overlap, normal in contacts:
            shapes = self.granules.sim_shapes
            r_eff = np.sqrt(shapes[i].equivalent_radius * shapes[j].equivalent_radius)
            F_mag = min(k_rep * r_eff / 50 * overlap, max_force)
            
            v_rel = self.granules.velocities[j] - self.granules.velocities[i]
            v_n = np.dot(v_rel, normal)
            F_damp = -damping * 0.5 * v_n
            
            F_total = max(0, F_mag + F_damp)
            forces[i] -= F_total * normal
            forces[j] += F_total * normal
        
        # Cell bridge forces
        # Convert nN to simulation units (scale with cell_force parameter)
        k_cell = cell_force * 0.03  # Scaling factor
        
        for i, j, n_cells, dist, direction in bridges:
            ri = self.granules.sim_shapes[i].equivalent_radius
            rj = self.granules.sim_shapes[j].equivalent_radius
            gap = dist - ri - rj
            extension = gap - cell_diameter
            
            F_mag = k_cell * n_cells * extension
            if extension < 0:
                F_mag *= 0.1
            
            F_mag = np.clip(F_mag * (1 + 0.15 * np.random.randn()), -max_force/2, max_force/2)
            forces[i] += F_mag * direction
            forces[j] -= F_mag * direction
        
        # Wall forces
        for i in range(n):
            r = self.granules.sim_shapes[i].bounding_radius
            pos = self.granules.positions[i]
            for d in range(3):
                if pos[d] < r:
                    forces[i, d] += min(k_wall * (r - pos[d]), max_force)
                if pos[d] > self.domain[d] - r:
                    forces[i, d] -= min(k_wall * (pos[d] - (self.domain[d] - r)), max_force)
        
        # Activity noise on functional
        func_mask = self.granules.types == 0
        noise = 0.03 * np.sqrt(self.granules.n_cells[func_mask] + 1)
        forces[func_mask] += noise[:, np.newaxis] * np.random.randn(np.sum(func_mask), 3)
        
        # Cap forces
        for i in range(n):
            f_mag = np.linalg.norm(forces[i])
            if f_mag > max_force:
                forces[i] *= max_force / f_mag
        
        return forces
    
    def step(self):
        contacts, bridges = self._find_contacts_and_bridges()
        forces = self._compute_forces(contacts, bridges)
        
        velocities = forces / self.granules.drag_coeffs[:, np.newaxis]
        
        # Cap velocities
        for i in range(self.n_total):
            v_mag = np.linalg.norm(velocities[i])
            if v_mag > 100:
                velocities[i] *= 100 / v_mag
        
        # Adaptive timestep
        max_vel = np.max(np.linalg.norm(velocities, axis=1))
        if max_vel > 1e-10:
            min_r = min(s.equivalent_radius for s in self.granules.sim_shapes)
            self.dt = np.clip(0.03 * min_r / max_vel,
                             self.config["time"]["dt_min_hours"],
                             self.config["time"]["dt_max_hours"])
        else:
            self.dt = self.config["time"]["dt_max_hours"]
        
        self.granules.positions += velocities * self.dt
        self.granules.velocities = velocities
        
        # Boundary clamp
        for i in range(self.n_total):
            r = self.granules.sim_shapes[i].bounding_radius
            self.granules.positions[i] = np.clip(
                self.granules.positions[i], r + 1, self.domain - r - 1
            )
        
        self.time += self.dt
        self.step_count += 1
        return contacts, bridges
    
    def _record_history(self, contacts, bridges):
        coord = np.zeros(self.n_total)
        for i, j, *_ in contacts:
            coord[i] += 1
            coord[j] += 1
        
        total_vol = sum(s.volume for s in self.granules.sim_shapes)
        phi = total_vol / np.prod(self.domain)
        
        self.history['time_hours'].append(self.time)
        self.history['n_contacts'].append(len(contacts))
        self.history['n_bridges'].append(len(bridges))
        self.history['mean_coordination'].append(np.mean(coord))
        self.history['max_velocity'].append(np.max(np.linalg.norm(self.granules.velocities, axis=1)))
        self.history['packing_fraction'].append(phi)
    
    def save_frame(self, label: str = None):
        if label is None:
            label = f"t{self.time:.1f}h"
        
        filename = self.output_dir / f"{self.config_name}_frame_{label}.json"
        
        data = {
            'time_hours': self.time,
            'config_name': self.config_name,
            'n_granules': self.n_total,
            'domain': self.domain.tolist(),
            'positions': self.granules.positions.tolist(),
            'orientations': [q.q.tolist() for q in self.granules.orientations],
            'sim_shapes': [s.to_dict() for s in self.granules.sim_shapes],
            'true_shapes': [s.to_dict() for s in self.granules.true_shapes],
            'types': self.granules.types.tolist(),
            'n_cells': self.granules.n_cells.tolist()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def run(self):
        if self.granules is None:
            self.setup()
        
        total_hours = self.config["time"]["total_hours"]
        save_interval = self.config["time"]["save_interval_hours"]
        
        print(f"\nRunning simulation for {total_hours:.0f} hours...")
        
        start_wall = time_module.time()
        last_save = 0.0
        last_record = 0.0
        
        self.save_frame("initial")
        
        if HAS_TQDM:
            pbar = tqdm(total=total_hours, desc="Simulating", unit="hr")
            pbar_time = 0.0
        
        while self.time < total_hours:
            contacts, bridges = self.step()
            
            if self.time - last_record >= 0.1:
                self._record_history(contacts, bridges)
                last_record = self.time
            
            if self.time - last_save >= save_interval:
                self.save_frame()
                last_save = self.time
            
            if HAS_TQDM:
                pbar.update(self.time - pbar_time)
                pbar_time = self.time
        
        if HAS_TQDM:
            pbar.close()
        
        self.save_frame("final")
        
        # Save history
        with open(self.output_dir / f"{self.config_name}_history.json", 'w') as f:
            json.dump(self.history, f)
        
        elapsed = time_module.time() - start_wall
        print(f"\nSimulation complete!")
        print(f"  Simulated: {self.time:.1f} hours ({self.time/24:.1f} days)")
        print(f"  Wall time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Output: {self.output_dir}")
        
        return self.history


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*60)
    print("DEM Simulation with JSON Configuration")
    print("="*60)
    
    # Get config file
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        print("\nSelect configuration file...")
        config_path = select_config_file()
    
    if not config_path:
        print("No config file selected. Exiting.")
        return
    
    print(f"\nLoading config: {config_path}")
    config = load_config(config_path)
    
    # Run simulation
    sim = Simulation(config, config_path)
    sim.run()


if __name__ == "__main__":
    main()