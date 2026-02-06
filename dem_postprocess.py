"""
Visualization for Jammed Non-Spherical Granule Simulations
===========================================================

Handles superellipsoid granules with proper cross-section rendering.
Includes animation of slice evolution over time.

Requirements:
    pip install numpy matplotlib scipy pillow

Usage:
    python dem_postprocess.py --input ./simulations/trial_experiment_001
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon, Patch
from matplotlib.collections import PatchCollection, EllipseCollection
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

try:
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ShapeData:
    """Shape parameters for a granule (true shape)."""
    a: float
    b: float
    c: float
    n: float      # superellipsoid roundness exponent
    roughness: float

    @property
    def equivalent_radius(self) -> float:
        V = (4/3) * np.pi * self.a * self.b * self.c
        return (3 * V / (4 * np.pi)) ** (1/3)

    @property
    def sphericity(self) -> float:
        r_eq = self.equivalent_radius
        sphere_area = 4 * np.pi * r_eq**2
        p = 1.6075
        actual_area = 4 * np.pi * ((self.a**p * self.b**p +
                                     self.a**p * self.c**p +
                                     self.b**p * self.c**p) / 3) ** (1/p)
        return min(1.0, sphere_area / actual_area)

    @property
    def aspect_ratio(self) -> float:
        return max(self.a, self.b, self.c) / min(self.a, self.b, self.c)


@dataclass
class FrameData:
    """Data from a single simulation frame."""
    time_hours: float
    n_granules: int
    positions: np.ndarray       # [n, 3]
    orientations: np.ndarray    # [n, 4] quaternions
    shapes: List[ShapeData]     # true shapes for visualization
    types: np.ndarray           # [n]
    n_cells: np.ndarray         # [n]
    mean_radius: Optional[float] = None
    domain: Optional[List[float]] = None

    @property
    def time_days(self) -> float:
        return self.time_hours / 24.0

    @property
    def functional_mask(self) -> np.ndarray:
        return self.types == 0

    @property
    def inert_mask(self) -> np.ndarray:
        return self.types == 1


def load_frame(filepath: str) -> Optional[FrameData]:
    """Load a JSON frame file. Handles both old and new key formats."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        # Prefer true_shapes for visualization, fall back to sim_shapes, then shapes
        shape_key = None
        for key in ['true_shapes', 'sim_shapes', 'shapes']:
            if key in data:
                shape_key = key
                break

        if shape_key is None:
            print(f"Warning: no shape data in {filepath}")
            return None

        shapes = []
        for s in data[shape_key]:
            shapes.append(ShapeData(
                a=s['a'], b=s['b'], c=s['c'],
                n=s.get('n', 2.0),
                roughness=s.get('roughness', 0.0)
            ))

        frame = FrameData(
            time_hours=data['time_hours'],
            n_granules=data['n_granules'],
            positions=np.array(data['positions']),
            orientations=np.array(data['orientations']),
            shapes=shapes,
            types=np.array(data['types']),
            n_cells=np.array(data['n_cells']),
            mean_radius=data.get('mean_radius', None),
            domain=data.get('domain', None)
        )
        return frame
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


# =============================================================================
# SUPERELLIPSOID CROSS-SECTION (TRUE SHAPE)
# =============================================================================

def superellipsoid_slice_contour(
    center: np.ndarray,
    orientation: np.ndarray,
    shape: ShapeData,
    plane_axis: int,
    plane_position: float,
    n_points: int = 64
) -> Optional[np.ndarray]:
    """
    Compute the intersection contour of a superellipsoid with an axis-aligned
    plane, returning an Nx2 polygon in the in-plane coordinates.

    Uses a parametric sweep in the body frame, accounting for the roundness
    exponent n (superellipsoid) and surface roughness.
    """
    R = quaternion_to_rotation_matrix(orientation)

    # Plane normal in world frame
    plane_normal = np.zeros(3)
    plane_normal[plane_axis] = 1.0

    # Transform plane into body frame
    center_body = np.zeros(3)  # body origin is at center
    normal_body = R.T @ plane_normal
    d_body = np.dot(normal_body, R.T @ (np.array([plane_position, plane_position, plane_position])
                    * plane_normal - center))

    # Quick bounding check: max extent along normal_body
    semi = np.array([shape.a, shape.b, shape.c])
    max_ext = np.sum(np.abs(normal_body) * semi)
    if abs(d_body) > max_ext:
        return None

    # Determine in-plane world axes
    if plane_axis == 0:
        dims = [1, 2]
    elif plane_axis == 1:
        dims = [0, 2]
    else:
        dims = [0, 1]

    # We'll numerically find the contour by sweeping an angle parameter.
    # In the body frame the superellipsoid is:
    #   |x/a|^n + |y/b|^n + |z/c|^n = 1
    # The slice plane in body coords is: normal_body . p = d_body
    #
    # Strategy: parameterize by angle theta around the plane, find the
    # point on the intersection contour for each theta.

    # Build a local 2D basis in the slice plane (body frame)
    # Find two orthogonal vectors in the plane normal_body . v = 0
    if abs(normal_body[0]) < 0.9:
        u1 = np.cross(normal_body, np.array([1, 0, 0]))
    else:
        u1 = np.cross(normal_body, np.array([0, 1, 0]))
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(normal_body, u1)
    u2 /= np.linalg.norm(u2)

    # Point on plane closest to origin
    p0 = normal_body * d_body

    contour_body = []
    thetas = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    n_exp = shape.n
    eps = 1e-12

    for theta in thetas:
        direction = np.cos(theta) * u1 + np.sin(theta) * u2
        # Find t such that |p0 + t*direction| on superellipsoid surface
        # i.e.  |(p0x + t*dx)/a|^n + |(p0y + t*dy)/b|^n + |(p0z + t*dz)/c|^n = 1
        # Solve via bisection
        t_lo, t_hi = 0.0, max(semi) * 2.0

        # First check if the ray intersects at all
        p_test = p0 + t_hi * direction
        val = (abs(p_test[0] / (shape.a + eps))**n_exp +
               abs(p_test[1] / (shape.b + eps))**n_exp +
               abs(p_test[2] / (shape.c + eps))**n_exp)
        if val < 1.0:
            # Even at max t we're inside — use t_hi
            contour_body.append(p0 + t_hi * direction)
            continue

        # Bisection to find surface crossing
        for _ in range(40):
            t_mid = 0.5 * (t_lo + t_hi)
            p_mid = p0 + t_mid * direction
            val = (abs(p_mid[0] / (shape.a + eps))**n_exp +
                   abs(p_mid[1] / (shape.b + eps))**n_exp +
                   abs(p_mid[2] / (shape.c + eps))**n_exp)
            if val < 1.0:
                t_lo = t_mid
            else:
                t_hi = t_mid

        t_sol = 0.5 * (t_lo + t_hi)
        if t_sol < 1e-6:
            return None  # degenerate
        contour_body.append(p0 + t_sol * direction)

    contour_body = np.array(contour_body)  # (n_points, 3)

    # Apply roughness perturbation
    if shape.roughness > 0:
        r_scale = shape.roughness * min(semi) * 0.1
        # Deterministic roughness from a hash so it's stable across frames
        for k in range(len(contour_body)):
            seed_val = int(abs(contour_body[k, 0] * 1000 + contour_body[k, 1] * 100 + contour_body[k, 2] * 10)) % (2**31)
            rng = np.random.RandomState(seed_val)
            noise = rng.randn() * r_scale
            # Perturb radially from p0
            radial = contour_body[k] - p0
            r_len = np.linalg.norm(radial)
            if r_len > eps:
                contour_body[k] += (noise / r_len) * radial

    # Transform back to world frame
    contour_world = (R @ contour_body.T).T + center

    # Project to 2D in-plane coordinates
    contour_2d = contour_world[:, dims]
    return contour_2d


def compute_ellipsoid_plane_intersection(
    center: np.ndarray,
    orientation: np.ndarray,
    shape: ShapeData,
    plane_axis: int,
    plane_position: float
) -> Optional[Tuple[np.ndarray, float, float, float]]:
    """
    Approximate the intersection as an ellipse (for fallback / fast rendering).
    Returns (center_2d, width, height, angle_deg) or None.
    """
    R = quaternion_to_rotation_matrix(orientation)
    body_axes = R.T
    semi_axes = np.array([shape.a, shape.b, shape.c])

    plane_normal = np.zeros(3)
    plane_normal[plane_axis] = 1.0

    max_extent = 0
    for i in range(3):
        max_extent = max(max_extent, semi_axes[i] * abs(np.dot(body_axes[:, i], plane_normal)))
    max_extent = max(max_extent, max(semi_axes))

    dist_to_plane = abs(center[plane_axis] - plane_position)
    if dist_to_plane > max_extent:
        return None

    if plane_axis == 0:
        in_plane = [1, 2]
    elif plane_axis == 1:
        in_plane = [0, 2]
    else:
        in_plane = [0, 1]

    center_2d = np.array([center[in_plane[0]], center[in_plane[1]]])

    if max_extent > 1e-6:
        scale = np.sqrt(max(0, 1 - (dist_to_plane / max_extent)**2))
    else:
        scale = 0

    if scale < 0.01:
        return None

    width = height = 0
    for i in range(3):
        body_axis_world = body_axes[:, i]
        width = max(width, abs(body_axis_world[in_plane[0]]) * semi_axes[i])
        height = max(height, abs(body_axis_world[in_plane[1]]) * semi_axes[i])

    width *= scale
    height *= scale

    body_x_world = body_axes[:, 0]
    angle_deg = np.degrees(np.arctan2(body_x_world[in_plane[1]], body_x_world[in_plane[0]]))

    return center_2d, width * 2, height * 2, angle_deg


# =============================================================================
# CROSS-SECTION RENDERING
# =============================================================================

def render_cross_section(
    frame: FrameData,
    domain: np.ndarray,
    axis: int,
    position: float,
    resolution: int = 200,
    use_true_shape: bool = True
) -> Tuple[np.ndarray, List[dict]]:
    """
    Render a cross-section as a rasterized phase map and vector overlays.
    When use_true_shape=True, computes actual superellipsoid contours.

    Returns:
        image: 2D array (0=void, 1=functional, 2=inert)
        overlays: list of dicts with polygon or ellipse data
    """
    if axis == 0:
        dims = [1, 2]; extent = [domain[1], domain[2]]
    elif axis == 1:
        dims = [0, 2]; extent = [domain[0], domain[2]]
    else:
        dims = [0, 1]; extent = [domain[0], domain[1]]

    image = np.zeros((resolution, resolution), dtype=np.float32)
    dx = extent[0] / resolution
    dy = extent[1] / resolution
    overlays = []

    for i in range(frame.n_granules):
        gtype = 'functional' if frame.types[i] == 0 else 'inert'
        value = 1.0 if frame.types[i] == 0 else 2.0

        if use_true_shape:
            contour = superellipsoid_slice_contour(
                frame.positions[i], frame.orientations[i],
                frame.shapes[i], axis, position, n_points=48
            )
            if contour is None:
                continue

            overlays.append({'contour': contour, 'type': gtype})

            # Rasterize polygon
            from matplotlib.path import Path as MplPath
            poly_path = MplPath(contour)
            # Build grid points inside bounding box
            cmin = contour.min(axis=0)
            cmax = contour.max(axis=0)
            ix_min = max(0, int(cmin[0] / dx))
            ix_max = min(resolution - 1, int(cmax[0] / dx) + 1)
            iy_min = max(0, int(cmin[1] / dy))
            iy_max = min(resolution - 1, int(cmax[1] / dy) + 1)

            for ix in range(ix_min, ix_max + 1):
                for iy in range(iy_min, iy_max + 1):
                    px = (ix + 0.5) * dx
                    py = (iy + 0.5) * dy
                    if poly_path.contains_point((px, py)):
                        image[ix, iy] = value
        else:
            result = compute_ellipsoid_plane_intersection(
                frame.positions[i], frame.orientations[i],
                frame.shapes[i], axis, position
            )
            if result is None:
                continue

            center_2d, w, h, angle = result
            overlays.append({
                'center': center_2d, 'width': w, 'height': h,
                'angle': angle, 'type': gtype
            })

            cx, cy = center_2d
            rx, ry = w / 2, h / 2
            ix_min = max(0, int((cx - rx) / dx))
            ix_max = min(resolution - 1, int((cx + rx) / dx) + 1)
            iy_min = max(0, int((cy - ry) / dy))
            iy_max = min(resolution - 1, int((cy + ry) / dy) + 1)
            cos_a = np.cos(np.radians(-angle))
            sin_a = np.sin(np.radians(-angle))

            for ix in range(ix_min, ix_max):
                for iy in range(iy_min, iy_max):
                    px = (ix + 0.5) * dx
                    py = (iy + 0.5) * dy
                    dx_l = px - cx
                    dy_l = py - cy
                    x_rot = dx_l * cos_a - dy_l * sin_a
                    y_rot = dx_l * sin_a + dy_l * cos_a
                    if rx > 0 and ry > 0:
                        if (x_rot / rx)**2 + (y_rot / ry)**2 <= 1:
                            image[ix, iy] = value

    return image, overlays


# =============================================================================
# VOID SPACE ANALYSIS
# =============================================================================

class VoidAnalyzer:
    """Analyze void space for non-spherical granules."""

    def __init__(self, domain: np.ndarray, resolution: int = 50):
        self.domain = domain
        self.resolution = resolution

    def compute_void_fraction(self, frame: FrameData) -> float:
        total_vol = sum((4/3) * np.pi * s.a * s.b * s.c for s in frame.shapes)
        return 1.0 - total_vol / np.prod(self.domain)

    def compute_void_profile(self, frame: FrameData, axis: int = 2,
                             n_slices: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.linspace(0, self.domain[axis], n_slices + 1)
        positions = 0.5 * (positions[:-1] + positions[1:])
        void_fractions = []

        for pos in positions:
            slice_thickness = self.domain[axis] / n_slices
            slice_vol = 0
            for i, shape in enumerate(frame.shapes):
                center = frame.positions[i, axis]
                r_eq = shape.equivalent_radius
                if abs(center - pos) < r_eq + slice_thickness / 2:
                    d = abs(center - pos)
                    if d < r_eq:
                        h = min(slice_thickness, r_eq - d + slice_thickness / 2)
                        cap_vol = np.pi * h**2 * (r_eq - h / 3)
                        slice_vol += min(cap_vol, (4/3) * np.pi * r_eq**3)

            total_slice = self.domain[0] * self.domain[1] * self.domain[axis] / n_slices
            void_fractions.append(1.0 - slice_vol / total_slice)

        return positions, np.array(void_fractions)


# =============================================================================
# MAIN VISUALIZER
# =============================================================================

class NonSphericalVisualizer:
    """Visualizer for non-spherical granule simulations."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)

        # Detect config name from directory name
        self.config_name = self.output_dir.name

        # Load config — try {name}_config.json first, then config.json
        self.config = {}
        for pattern in [f"{self.config_name}_config.json", "config.json"]:
            cfg_path = self.output_dir / pattern
            if cfg_path.exists():
                with open(cfg_path) as f:
                    self.config = json.load(f)
                print(f"Loaded config: {cfg_path.name}")
                break

        # Load history — try {name}_history.json first, then history.json
        self.history = None
        for pattern in [f"{self.config_name}_history.json", "history.json"]:
            hist_path = self.output_dir / pattern
            if hist_path.exists():
                with open(hist_path) as f:
                    self.history = json.load(f)
                print(f"Loaded history: {hist_path.name}")
                break

        # Find frame files — try {name}_frame_*.json first, then frame_*.json
        self.frame_files = sorted(self.output_dir.glob(f"{self.config_name}_frame_*.json"))
        if not self.frame_files:
            self.frame_files = sorted(self.output_dir.glob("frame_*.json"))
        print(f"Found {len(self.frame_files)} frames")

        self.frames = []
        for fp in self.frame_files:
            frame = load_frame(str(fp))
            if frame is not None:
                self.frames.append(frame)
        # Sort frames by time so evolution plots always go t=0 → t=max
        self.frames.sort(key=lambda f: f.time_hours)
        print(f"Loaded {len(self.frames)} frames")

        if self.frames:
            print(f"Time range: {self.frames[0].time_days:.2f} – {self.frames[-1].time_days:.2f} days")

        # Get domain
        self.domain = None
        if 'domain' in self.config:
            side = self.config['domain'].get('side_length_um', None)
            if side:
                self.domain = np.array([side, side, side])
        if self.domain is None and 'domain_x' in self.config:
            self.domain = np.array([
                self.config['domain_x'], self.config['domain_y'], self.config['domain_z']
            ])
        if self.domain is None and self.frames and self.frames[0].domain is not None:
            self.domain = np.array(self.frames[0].domain)
        if self.domain is None and self.frames:
            pos = self.frames[0].positions
            max_r = max(s.equivalent_radius for s in self.frames[0].shapes)
            self.domain = np.max(pos, axis=0) + max_r * 2
            print(f"Estimated domain from positions: {self.domain}")
        if self.domain is None:
            self.domain = np.array([700.0, 700.0, 700.0])
        print(f"Domain: {self.domain[0]:.1f} × {self.domain[1]:.1f} × {self.domain[2]:.1f} μm")

        # Phase-map colormap: 0=void(white), 1=functional(red), 2=inert(blue)
        self.cmap = LinearSegmentedColormap.from_list('granules', ['white', 'red', 'blue'], N=3)

        self.void_analyzer = VoidAnalyzer(self.domain)

        if self.frames:
            self.mean_granule_radius = np.mean([s.equivalent_radius for s in self.frames[0].shapes])
        else:
            self.mean_granule_radius = 50.0

    # -----------------------------------------------------------------
    # Phase-space map (3-color solid: void / functional / inert)
    # -----------------------------------------------------------------
    def plot_phase_map(self, frame_index: int = -1, axis: int = 2,
                       resolution: int = 300, save: bool = True) -> plt.Figure:
        """
        Plot a clean 3-color phase-space cross-section.
        White = void, Red = functional, Blue = inert.
        """
        if not self.frames:
            return None

        frame = self.frames[frame_index]
        slice_pos = self.domain[axis] / 2

        image, _ = render_cross_section(
            frame, self.domain, axis, slice_pos,
            resolution=resolution, use_true_shape=True
        )

        plane_labels = [('Y', 'Z'), ('X', 'Z'), ('X', 'Y')]
        axis_labels = ['X', 'Y', 'Z']
        if axis == 2:
            ext = [0, self.domain[0], 0, self.domain[1]]
        elif axis == 1:
            ext = [0, self.domain[0], 0, self.domain[2]]
        else:
            ext = [0, self.domain[1], 0, self.domain[2]]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image.T, origin='lower', cmap=self.cmap,
                  vmin=0, vmax=2, extent=ext, aspect='equal', interpolation='nearest')
        ax.set_xlabel(f'{plane_labels[axis][0]} (μm)', fontsize=12)
        ax.set_ylabel(f'{plane_labels[axis][1]} (μm)', fontsize=12)
        ax.set_title(
            f'Phase Map — {axis_labels[axis]} = {slice_pos:.0f} μm  '
            f'(t = {frame.time_days:.2f} days)', fontsize=13
        )

        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Void'),
            Patch(facecolor='red', label='Functional'),
            Patch(facecolor='blue', label='Inert'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.tight_layout()

        if save:
            label = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"phase_map_{axis_labels[axis].lower()}_{label}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # True-shape vector cross-section
    # -----------------------------------------------------------------
    def plot_true_shape_cross_section(self, frame_index: int = -1, axis: int = 2,
                                       save: bool = True) -> plt.Figure:
        """
        Vector plot showing actual superellipsoid contours (polygons) in cross-section.
        """
        if not self.frames:
            return None

        frame = self.frames[frame_index]
        slice_pos = self.domain[axis] / 2
        _, overlays = render_cross_section(
            frame, self.domain, axis, slice_pos,
            resolution=50, use_true_shape=True  # low raster res, we only want overlays
        )

        plane_labels = [('Y', 'Z'), ('X', 'Z'), ('X', 'Y')]
        axis_labels = ['X', 'Y', 'Z']
        if axis == 2:
            ext = [0, self.domain[0], 0, self.domain[1]]
        elif axis == 1:
            ext = [0, self.domain[0], 0, self.domain[2]]
        else:
            ext = [0, self.domain[1], 0, self.domain[2]]

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_aspect('equal')
        ax.set_facecolor('white')

        for ov in overlays:
            if 'contour' in ov:
                color = '#d62728' if ov['type'] == 'functional' else '#1f77b4'
                ec = 'darkred' if ov['type'] == 'functional' else 'darkblue'
                alpha = 0.7 if ov['type'] == 'functional' else 0.5
                poly = Polygon(ov['contour'], closed=True,
                               facecolor=color, edgecolor=ec,
                               alpha=alpha, linewidth=0.6)
                ax.add_patch(poly)
            elif 'center' in ov:
                color = '#d62728' if ov['type'] == 'functional' else '#1f77b4'
                ec = 'darkred' if ov['type'] == 'functional' else 'darkblue'
                alpha = 0.7 if ov['type'] == 'functional' else 0.5
                ell = Ellipse(xy=ov['center'], width=ov['width'], height=ov['height'],
                              angle=ov['angle'], facecolor=color, edgecolor=ec,
                              alpha=alpha, linewidth=0.6)
                ax.add_patch(ell)

        ax.set_xlabel(f'{plane_labels[axis][0]} (μm)', fontsize=12)
        ax.set_ylabel(f'{plane_labels[axis][1]} (μm)', fontsize=12)
        ax.set_title(
            f'True Shape Cross-Section — {axis_labels[axis]} = {slice_pos:.0f} μm  '
            f'(t = {frame.time_days:.2f} days)', fontsize=12
        )
        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Void'),
            Patch(facecolor='#d62728', alpha=0.7, label='Functional'),
            Patch(facecolor='#1f77b4', alpha=0.5, label='Inert'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        plt.tight_layout()

        if save:
            label = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"true_shape_{axis_labels[axis].lower()}_{label}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Cross-section evolution (raster + vector rows)
    # -----------------------------------------------------------------
    def plot_cross_section_evolution(self, axis: int = 2, n_times: int = 6,
                                     save: bool = True) -> plt.Figure:
        if not self.frames:
            return None

        frame_indices = np.linspace(0, len(self.frames) - 1, n_times, dtype=int)
        slice_pos = self.domain[axis] / 2

        fig, axes_grid = plt.subplots(2, n_times, figsize=(3 * n_times, 6))
        axis_labels = ['X', 'Y', 'Z']
        plane_labels = [('Y', 'Z'), ('X', 'Z'), ('X', 'Y')]

        for col, fi in enumerate(frame_indices):
            frame = self.frames[fi]
            image, overlays = render_cross_section(
                frame, self.domain, axis, slice_pos,
                resolution=150, use_true_shape=True
            )

            if axis == 2:
                ext = [0, self.domain[0], 0, self.domain[1]]
            elif axis == 1:
                ext = [0, self.domain[0], 0, self.domain[2]]
            else:
                ext = [0, self.domain[1], 0, self.domain[2]]

            # Top: rasterized phase map
            ax = axes_grid[0, col]
            ax.imshow(image.T, origin='lower', cmap=self.cmap,
                      vmin=0, vmax=2, extent=ext, aspect='equal', interpolation='nearest')
            ax.set_title(f't = {frame.time_days:.1f} d', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'Phase Map\n{plane_labels[axis][1]} (μm)')
            ax.set_xlabel(f'{plane_labels[axis][0]} (μm)')

            # Bottom: vector true shapes
            ax = axes_grid[1, col]
            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ax.set_aspect('equal')
            ax.set_facecolor('white')

            for ov in overlays:
                is_func = ov['type'] == 'functional'
                color = '#d62728' if is_func else '#1f77b4'
                ec = 'darkred' if is_func else 'darkblue'
                alpha = 0.7 if is_func else 0.5

                if 'contour' in ov:
                    poly = Polygon(ov['contour'], closed=True,
                                   facecolor=color, edgecolor=ec,
                                   alpha=alpha, linewidth=0.4)
                    ax.add_patch(poly)
                elif 'center' in ov:
                    ell = Ellipse(xy=ov['center'], width=ov['width'],
                                  height=ov['height'], angle=ov['angle'],
                                  facecolor=color, edgecolor=ec,
                                  alpha=alpha, linewidth=0.4)
                    ax.add_patch(ell)

            if col == 0:
                ax.set_ylabel(f'True Shape\n{plane_labels[axis][1]} (μm)')
            ax.set_xlabel(f'{plane_labels[axis][0]} (μm)')

        legend_elements = [
            Patch(facecolor='white', edgecolor='gray', label='Void'),
            Patch(facecolor='#d62728', alpha=0.7, label='Functional'),
            Patch(facecolor='#1f77b4', alpha=0.5, label='Inert'),
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        plt.suptitle(
            f'Cross-Section Evolution ({axis_labels[axis]}={slice_pos:.0f} μm)', fontsize=12
        )
        plt.tight_layout()

        if save:
            fp = self.viz_dir / f"cross_section_evolution_{axis_labels[axis].lower()}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Void fraction evolution
    # -----------------------------------------------------------------
    def plot_void_fraction_evolution(self, save: bool = True) -> plt.Figure:
        if not self.frames:
            return None

        print("Computing void fractions...")
        times, void_fractions = [], []
        for frame in self.frames:
            times.append(frame.time_days)
            void_fractions.append(self.void_analyzer.compute_void_fraction(frame))

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.plot(times, void_fractions, 'ko-', markersize=6)
        if self.history and 'packing_fraction' in self.history:
            t_h = np.array(self.history['time_hours']) / 24
            vf_h = 1 - np.array(self.history['packing_fraction'])
            ax.plot(t_h, vf_h, 'b-', alpha=0.5, label='From history')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Void Fraction')
        ax.set_title('Overall Void Fraction'); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[0, 1]
        n_prof = min(5, len(self.frames))
        idxs = np.linspace(0, len(self.frames) - 1, n_prof, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_prof))
        for idx, c in zip(idxs, colors):
            z, vf = self.void_analyzer.compute_void_profile(self.frames[idx], axis=2)
            ax.plot(z, vf, '-', color=c, lw=2, label=f't = {self.frames[idx].time_days:.1f} d')
        ax.set_xlabel('Z position (μm)'); ax.set_ylabel('Local Void Fraction')
        ax.set_title('Void Profile (Z-axis)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        packing = [1 - vf for vf in void_fractions]
        ax.fill_between(times, 0, packing, alpha=0.3, color='steelblue')
        ax.plot(times, packing, 'b-', lw=2)
        ax.axhline(y=0.64, color='r', ls='--', alpha=0.7, label='Random close packing')
        ax.axhline(y=0.55, color='orange', ls='--', alpha=0.7, label='Random loose packing')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction')
        ax.set_title('Packing Density Evolution')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 0.8)

        ax = axes[1, 1]
        if self.frames:
            frame = self.frames[-1]
            fm, im = frame.functional_mask, frame.inert_mask
            sph_f = [s.sphericity for i, s in enumerate(frame.shapes) if fm[i]]
            sph_i = [s.sphericity for i, s in enumerate(frame.shapes) if im[i]]
            asp_f = [s.aspect_ratio for i, s in enumerate(frame.shapes) if fm[i]]
            asp_i = [s.aspect_ratio for i, s in enumerate(frame.shapes) if im[i]]
            x = np.arange(2); w = 0.35
            ax.bar(x - w/2, [np.mean(sph_f), np.mean(sph_i)], w,
                   label='Sphericity', color='steelblue', alpha=0.7)
            ax.bar(x + w/2, [np.mean(asp_f)/2, np.mean(asp_i)/2], w,
                   label='Aspect Ratio/2', color='coral', alpha=0.7)
            ax.set_xticks(x); ax.set_xticklabels(['Functional', 'Inert'])
            ax.set_ylabel('Value'); ax.set_title('Shape Statistics')
            ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "void_fraction_evolution.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Detailed single frame
    # -----------------------------------------------------------------
    def plot_detailed_frame(self, frame_index: int = -1, save: bool = True) -> plt.Figure:
        if not self.frames:
            return None

        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3)

        for col, ax_id in enumerate([2, 1, 0]):
            ax = fig.add_subplot(gs[0, col])
            sp = self.domain[ax_id] / 2
            image, _ = render_cross_section(
                frame, self.domain, ax_id, sp, resolution=200, use_true_shape=True
            )
            pl = [('Y', 'Z'), ('X', 'Z'), ('X', 'Y')]
            if ax_id == 2:
                ext = [0, self.domain[0], 0, self.domain[1]]
            elif ax_id == 1:
                ext = [0, self.domain[0], 0, self.domain[2]]
            else:
                ext = [0, self.domain[1], 0, self.domain[2]]
            ax.imshow(image.T, origin='lower', cmap=self.cmap, vmin=0, vmax=2,
                      extent=ext, aspect='equal', interpolation='nearest')
            ax.set_xlabel(f'{pl[ax_id][0]} (μm)'); ax.set_ylabel(f'{pl[ax_id][1]} (μm)')
            ax.set_title(f'{["YZ","XZ","XY"][ax_id]} plane (mid-slice)')

        ax = fig.add_subplot(gs[1, 0])
        for a, label, c in [(0, 'X', 'red'), (1, 'Y', 'green'), (2, 'Z', 'blue')]:
            pos, vf = self.void_analyzer.compute_void_profile(frame, a)
            ax.plot(pos, vf, '-', color=c, lw=2, label=f'{label}-axis')
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Local Void Fraction')
        ax.set_title('Void Fraction Profiles'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        fr = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.functional_mask[i]]
        ir = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.inert_mask[i]]
        ax.hist(fr, bins=15, alpha=0.6, color='red', label='Functional')
        ax.hist(ir, bins=15, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equivalent Radius (μm)'); ax.set_ylabel('Count')
        ax.set_title('Size Distribution'); ax.legend()

        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        n_f = np.sum(frame.functional_mask)
        n_i = np.sum(frame.inert_mask)
        vf = self.void_analyzer.compute_void_fraction(frame)
        mean_sph_f = np.mean([s.sphericity for i, s in enumerate(frame.shapes) if frame.functional_mask[i]])
        mean_sph_i = np.mean([s.sphericity for i, s in enumerate(frame.shapes) if frame.inert_mask[i]])
        stats = (
            f"Frame Statistics\n{'─'*28}\n"
            f"Time: {frame.time_hours:.1f} h ({frame.time_days:.2f} days)\n\n"
            f"Granule Counts:\n  Functional: {n_f}\n  Inert: {n_i}\n  Total: {frame.n_granules}\n\n"
            f"Equivalent Radii:\n  Functional: {np.mean(fr):.1f} μm\n  Inert: {np.mean(ir):.1f} μm\n\n"
            f"Shape (Sphericity):\n  Functional: {mean_sph_f:.3f}\n  Inert: {mean_sph_i:.3f}\n\n"
            f"Packing:\n  Void fraction: {vf:.3f}\n  Packing fraction: {1-vf:.3f}\n\n"
            f"Cells: {np.sum(frame.n_cells)} total\n"
            f"Domain: {self.domain[0]:.0f}×{self.domain[1]:.0f}×{self.domain[2]:.0f} μm"
        )
        ax.text(0.1, 0.95, stats, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Detailed Analysis — t = {frame.time_days:.2f} days', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            label = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"detailed_{label}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Slice animation
    # -----------------------------------------------------------------
    def create_slice_animation(self, axis: int = 2, slice_position: float = None,
                                slice_thickness: float = None, fps: int = 8,
                                dpi: int = 150, save: bool = True) -> Optional[str]:
        if not HAS_ANIMATION or not self.frames:
            return None

        print(f"Creating slice animation with {len(self.frames)} frames...")

        if slice_position is None:
            slice_position = self.domain[axis] / 2
        if slice_thickness is None:
            mean_r = np.mean([s.equivalent_radius for s in self.frames[0].shapes])
            slice_thickness = mean_r * 2.2

        axis_labels = ['X', 'Y', 'Z']
        if axis == 0:
            dim1, dim2 = 1, 2; xlabel, ylabel = 'Y (μm)', 'Z (μm)'
            ext = [0, self.domain[1], 0, self.domain[2]]
        elif axis == 1:
            dim1, dim2 = 0, 2; xlabel, ylabel = 'X (μm)', 'Z (μm)'
            ext = [0, self.domain[0], 0, self.domain[2]]
        else:
            dim1, dim2 = 0, 1; xlabel, ylabel = 'X (μm)', 'Y (μm)'
            ext = [0, self.domain[0], 0, self.domain[1]]

        fig, axes_pair = plt.subplots(1, 2, figsize=(14, 6))
        ax_slice, ax_metrics = axes_pair

        ax_slice.set_xlim(ext[0], ext[1]); ax_slice.set_ylim(ext[2], ext[3])
        ax_slice.set_aspect('equal'); ax_slice.set_xlabel(xlabel); ax_slice.set_ylabel(ylabel)
        ax_slice.set_facecolor('#f0f0f0')
        ax_slice.legend(handles=[
            Patch(facecolor='red', alpha=0.7, edgecolor='darkred', label='Functional'),
            Patch(facecolor='blue', alpha=0.5, edgecolor='darkblue', label='Inert'),
        ], loc='upper right', fontsize=9)

        if self.history:
            t_h = np.array(self.history['time_hours']) / 24
            ax_metrics.plot(t_h, self.history['mean_coordination'], 'g-', lw=2, label='Coordination')
            ax_metrics.set_xlabel('Time (days)'); ax_metrics.set_ylabel('Mean Coordination', color='g')
            ax_metrics.tick_params(axis='y', labelcolor='g'); ax_metrics.grid(True, alpha=0.3)
            ax2 = ax_metrics.twinx()
            ax2.plot(t_h, self.history['n_bridges'], 'r-', lw=2, alpha=0.7, label='Bridges')
            ax2.set_ylabel('Cell Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')

        time_line = ax_metrics.axvline(x=0, color='black', lw=2, ls='--')
        title = fig.suptitle('', fontsize=12, fontweight='bold')

        # Track granule patches so we can remove them without touching legend patches
        granule_patches = []

        def update(frame_idx):
            frame = self.frames[frame_idx]

            # Remove only the granule patches from the previous frame
            for p in granule_patches:
                p.remove()
            granule_patches.clear()

            slice_min = slice_position - slice_thickness / 2
            slice_max = slice_position + slice_thickness / 2

            for i in range(frame.n_granules):
                pos = frame.positions[i]; shape = frame.shapes[i]
                r_b = max(shape.a, shape.b, shape.c)
                if pos[axis] - r_b > slice_max or pos[axis] + r_b < slice_min:
                    continue

                r_eq = shape.equivalent_radius
                dist = abs(pos[axis] - slice_position)
                r_cross = np.sqrt(max(0, r_eq**2 - dist**2)) if dist < r_eq else r_eq * 0.3

                is_func = frame.types[i] == 0
                color = 'red' if is_func else 'blue'
                ec = 'darkred' if is_func else 'darkblue'
                alpha = max(0.2, (0.7 if is_func else 0.5) * (1 - 0.3 * dist / (r_eq + 1e-6)))

                # Approximate cross-section ellipse
                sc = r_cross / (r_eq + 1e-6)
                if axis == 2:
                    w, h = shape.a * 2 * sc, shape.b * 2 * sc
                elif axis == 1:
                    w, h = shape.a * 2 * sc, shape.c * 2 * sc
                else:
                    w, h = shape.b * 2 * sc, shape.c * 2 * sc
                w = max(3, min(w, r_b * 2.5)); h = max(3, min(h, r_b * 2.5))

                q = frame.orientations[i]; R = quaternion_to_rotation_matrix(q)
                angle = np.degrees(np.arctan2(R[dim2, dim1], R[dim1, dim1]))

                ell = Ellipse(xy=(pos[dim1], pos[dim2]), width=w, height=h, angle=angle,
                              facecolor=color, edgecolor=ec, alpha=alpha, linewidth=0.5)
                ax_slice.add_patch(ell)
                granule_patches.append(ell)

            time_line.set_xdata([frame.time_days, frame.time_days])
            title.set_text(
                f'Slice Evolution — t = {frame.time_hours:.1f}h ({frame.time_days:.2f} days)\n'
                f'{axis_labels[axis]} = {slice_position:.0f} μm ± {slice_thickness/2:.0f} μm'
            )
            return []

        print("  Rendering frames...")
        anim = FuncAnimation(fig, update, frames=len(self.frames), interval=1000 // fps, blit=False)
        plt.tight_layout()

        if save:
            fp = self.viz_dir / f"slice_animation_{axis_labels[axis].lower()}.gif"
            try:
                anim.save(str(fp), writer=PillowWriter(fps=fps), dpi=dpi)
                print(f"  Saved: {fp}")
            except Exception as e:
                print(f"  Error saving GIF: {e}")
                fp = None
            plt.close()
            return str(fp) if fp else None
        else:
            plt.show()
            return None

    # -----------------------------------------------------------------
    # History summary
    # -----------------------------------------------------------------
    def plot_history(self, save: bool = True) -> Optional[plt.Figure]:
        if not self.history:
            print("No history data")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = np.array(self.history['time_hours']) / 24

        ax = axes[0, 0]
        ax.plot(t, self.history['n_contacts'], 'b-', lw=2, label='Contacts')
        ax.plot(t, self.history['n_bridges'], 'r-', lw=2, label='Bridges')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Count')
        ax.set_title('Interactions'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(t, self.history['mean_coordination'], 'g-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean Coordination')
        ax.set_title('Contact Network'); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.semilogy(t, self.history['max_velocity'], 'm-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Max Velocity (μm/hr)')
        ax.set_title('Activity Level'); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(t, self.history['packing_fraction'], 'c-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction')
        ax.set_title('Packing Density'); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "history_summary.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight')
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Generate everything
    # -----------------------------------------------------------------
    def create_all(self):
        print("\n" + "=" * 60)
        print("Generating All Visualizations")
        print("=" * 60)

        print("\n1. History summary...")
        self.plot_history()

        print("\n2. Void fraction evolution...")
        self.plot_void_fraction_evolution()

        print("\n3. Phase map (final frame, all 3 planes)...")
        if self.frames:
            for ax in [2, 1, 0]:
                self.plot_phase_map(frame_index=-1, axis=ax)

        print("\n4. True-shape cross-section (final)...")
        if self.frames:
            self.plot_true_shape_cross_section(frame_index=-1, axis=2)

        print("\n5. Cross-section evolution (XY)...")
        self.plot_cross_section_evolution(axis=2)

        print("\n6. Cross-section evolution (XZ)...")
        self.plot_cross_section_evolution(axis=1)

        print("\n7. Detailed initial frame...")
        if self.frames:
            self.plot_detailed_frame(0)

        print("\n8. Detailed final frame...")
        if self.frames:
            self.plot_detailed_frame(-1)

        print("\n9. Slice animation...")
        if self.frames:
            self.create_slice_animation(axis=2, fps=6)

        print(f"\nAll visualizations saved to: {self.viz_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize non-spherical granule simulation')
    parser.add_argument('--input', '-i', type=str, default='./simulations/dem_config',
                        help='Path to simulation output directory')
    parser.add_argument('--all', '-a', action='store_true', help='Generate all visualizations')
    parser.add_argument('--phase-map', '-p', action='store_true', help='Phase map only')
    parser.add_argument('--true-shape', '-t', action='store_true', help='True shape cross-section only')
    parser.add_argument('--slice-animation', '-s', action='store_true', help='Slice animation')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--axis', type=int, default=2, help='Slice axis (0=X, 1=Y, 2=Z)')

    args = parser.parse_args()
    viz = NonSphericalVisualizer(args.input)

    if args.phase_map:
        viz.plot_phase_map(axis=args.axis)
    elif args.true_shape:
        viz.plot_true_shape_cross_section(axis=args.axis)
    elif args.slice_animation:
        viz.create_slice_animation(axis=args.axis, fps=args.fps)
    elif args.all:
        viz.create_all()
    else:
        viz.create_all()


if __name__ == "__main__":
    main()