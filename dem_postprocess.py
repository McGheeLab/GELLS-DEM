"""
Visualization for Jammed Non-Spherical Granule Simulations
===========================================================

All cross-sections rendered as 3-color phase maps:
  White = void,  Red = functional,  Blue = inert

Includes watch mode for live monitoring during simulation.

Requirements:
    pip install numpy matplotlib scipy pillow

Usage:
    python dem_postprocess.py -i ./simulations/dem_config          # full plots
    python dem_postprocess.py -i ./simulations/dem_config --watch   # live monitor
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import json
import argparse
import time as time_module
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


# =============================================================================
# DATA LOADING
# =============================================================================

@dataclass
class ShapeData:
    a: float; b: float; c: float; n: float; roughness: float

    @property
    def equivalent_radius(self):
        return (self.a * self.b * self.c) ** (1 / 3)

    @property
    def sphericity(self):
        r = self.equivalent_radius
        p = 1.6075
        sa = 4 * np.pi * ((self.a**p*self.b**p + self.a**p*self.c**p + self.b**p*self.c**p) / 3) ** (1/p)
        return min(1.0, 4 * np.pi * r**2 / sa)

    @property
    def aspect_ratio(self):
        return max(self.a, self.b, self.c) / min(self.a, self.b, self.c)


@dataclass
class FrameData:
    time_hours: float
    n_granules: int
    positions: np.ndarray
    orientations: np.ndarray
    shapes: List[ShapeData]
    types: np.ndarray
    n_cells: np.ndarray
    mean_radius: Optional[float] = None
    domain: Optional[List[float]] = None

    @property
    def time_days(self): return self.time_hours / 24.0
    @property
    def functional_mask(self): return self.types == 0
    @property
    def inert_mask(self): return self.types == 1


def load_frame(filepath: str) -> Optional[FrameData]:
    try:
        with open(filepath) as f:
            data = json.load(f)
        key = None
        for k in ['true_shapes', 'sim_shapes', 'shapes']:
            if k in data:
                key = k; break
        if key is None:
            return None
        shapes = [ShapeData(s['a'], s['b'], s['c'], s.get('n', 2.0), s.get('roughness', 0.0))
                  for s in data[key]]
        return FrameData(
            time_hours=data['time_hours'], n_granules=data['n_granules'],
            positions=np.array(data['positions']),
            orientations=np.array(data['orientations']),
            shapes=shapes,
            types=np.array(data['types']),
            n_cells=np.array(data['n_cells']),
            mean_radius=data.get('mean_radius'),
            domain=data.get('domain'))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


# =============================================================================
# PHASE MAP RENDERING (ellipsoid cross-section → rasterized 3-color image)
# =============================================================================

def render_phase_map(
    frame: FrameData,
    domain: np.ndarray,
    axis: int,
    position: float,
    resolution: int = 300
) -> np.ndarray:
    """
    Render a rasterized phase-map cross-section.
    Returns 2D array: 0 = void, 1 = functional, 2 = inert.
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

    for i in range(frame.n_granules):
        shape = frame.shapes[i]
        pos = frame.positions[i]
        q = frame.orientations[i]
        R = quaternion_to_rotation_matrix(q)
        semi = np.array([shape.a, shape.b, shape.c])

        # Check if ellipsoid intersects the slice plane
        plane_normal = np.zeros(3); plane_normal[axis] = 1.0
        # Max extent along plane normal
        max_ext = sum(semi[k] * abs(R[axis, k]) for k in range(3))
        dist_to_plane = abs(pos[axis] - position)
        if dist_to_plane > max_ext:
            continue

        # Compute intersection ellipse parameters
        # Transform plane into body frame, find 2D ellipse
        # Using the quadric-plane intersection formula:

        # Build the ellipsoid matrix M = R @ diag(1/a^2, 1/b^2, 1/c^2) @ R^T
        D_inv = np.diag([1.0/semi[0]**2, 1.0/semi[1]**2, 1.0/semi[2]**2])
        M = R @ D_inv @ R.T

        # Signed distance from center to plane
        h = pos[axis] - position

        # The intersection ellipse in the 2 in-plane dims is defined by:
        # [x-cx, y-cy]^T @ A_2d @ [x-cx, y-cy] <= 1
        # where A_2d comes from eliminating the normal-axis coordinate.

        # Extract the 2x2 sub-matrix and coupling terms
        d0, d1 = dims
        m_nn = M[axis, axis]
        if m_nn < 1e-12:
            continue

        # Center shift of the intersection ellipse
        cx_2d = pos[d0] - M[d0, axis] / m_nn * h
        cy_2d = pos[d1] - M[d1, axis] / m_nn * h

        # 2x2 matrix for the intersection ellipse
        A = np.array([
            [M[d0, d0] - M[d0, axis]**2 / m_nn,
             M[d0, d1] - M[d0, axis] * M[d1, axis] / m_nn],
            [M[d1, d0] - M[d1, axis] * M[d0, axis] / m_nn,
             M[d1, d1] - M[d1, axis]**2 / m_nn]
        ])

        # RHS: 1 - h^2 * m_nn  (the "budget" left after the normal component)
        rhs = 1.0 - h**2 * m_nn
        if rhs <= 0:
            continue

        # Scale A so the ellipse is:  v^T @ A_scaled @ v <= 1
        A_scaled = A / rhs

        # Get eigenvalues to find semi-axes of the 2D ellipse
        try:
            eigvals, eigvecs = np.linalg.eigh(A_scaled)
        except np.linalg.LinAlgError:
            continue

        if eigvals[0] <= 0 or eigvals[1] <= 0:
            continue

        # Semi-axes of 2D ellipse
        sa0 = 1.0 / np.sqrt(eigvals[0])
        sa1 = 1.0 / np.sqrt(eigvals[1])
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        value = 1.0 if frame.types[i] == 0 else 2.0

        # Rasterize the ellipse
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # Bounding box in pixel coords
        rx = max(sa0, sa1)
        ix_min = max(0, int((cx_2d - rx) / dx) - 1)
        ix_max = min(resolution - 1, int((cx_2d + rx) / dx) + 2)
        iy_min = max(0, int((cy_2d - rx) / dy) - 1)
        iy_max = min(resolution - 1, int((cy_2d + rx) / dy) + 2)

        for ix in range(ix_min, ix_max):
            px = (ix + 0.5) * dx - cx_2d
            for iy in range(iy_min, iy_max):
                py = (iy + 0.5) * dy - cy_2d
                xr = px * cos_a - py * sin_a
                yr = px * sin_a + py * cos_a
                if (xr / sa0)**2 + (yr / sa1)**2 <= 1.0:
                    image[ix, iy] = value

    return image


# =============================================================================
# VOID ANALYSIS
# =============================================================================

class VoidAnalyzer:
    def __init__(self, domain):
        self.domain = domain

    def compute_void_fraction(self, frame):
        vol = sum((4/3)*np.pi*s.a*s.b*s.c for s in frame.shapes)
        return 1.0 - vol / np.prod(self.domain)

    def compute_void_profile(self, frame, axis=2, n_slices=30):
        positions = np.linspace(0, self.domain[axis], n_slices + 1)
        positions = 0.5 * (positions[:-1] + positions[1:])
        voids = []
        for pos in positions:
            thick = self.domain[axis] / n_slices
            sv = 0
            for i, s in enumerate(frame.shapes):
                c = frame.positions[i, axis]
                r = s.equivalent_radius
                if abs(c - pos) < r + thick / 2:
                    d = abs(c - pos)
                    if d < r:
                        h = min(thick, r - d + thick / 2)
                        sv += min(np.pi * h**2 * (r - h/3), (4/3)*np.pi*r**3)
            total = self.domain[0] * self.domain[1] * thick
            voids.append(1.0 - sv / total)
        return positions, np.array(voids)


# =============================================================================
# VISUALIZER
# =============================================================================

# Shared colormap and legend
PHASE_CMAP = LinearSegmentedColormap.from_list('phase', ['white', 'red', 'blue'], N=3)
PHASE_LEGEND = [
    Patch(facecolor='white', edgecolor='gray', label='Void'),
    Patch(facecolor='red', label='Functional'),
    Patch(facecolor='blue', label='Inert'),
]


def _plot_phase(ax, image, extent, xlabel, ylabel, title=None):
    """Helper to plot a phase map on an axis."""
    ax.imshow(image.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
              extent=extent, aspect='equal', interpolation='nearest')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def _extent_for_axis(domain, axis):
    if axis == 2: return [0, domain[0], 0, domain[1]], 'X (μm)', 'Y (μm)'
    if axis == 1: return [0, domain[0], 0, domain[2]], 'X (μm)', 'Z (μm)'
    return [0, domain[1], 0, domain[2]], 'Y (μm)', 'Z (μm)'


class NonSphericalVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.config_name = self.output_dir.name

        # Load config
        self.config = {}
        for pat in [f"{self.config_name}_config.json", "config.json"]:
            p = self.output_dir / pat
            if p.exists():
                with open(p) as f: self.config = json.load(f)
                print(f"Loaded config: {p.name}"); break

        # Load history
        self.history = None
        for pat in [f"{self.config_name}_history.json", "history.json"]:
            p = self.output_dir / pat
            if p.exists():
                with open(p) as f: self.history = json.load(f)
                print(f"Loaded history: {p.name}"); break

        # Load frames
        self.frame_files = sorted(self.output_dir.glob(f"{self.config_name}_frame_*.json"))
        if not self.frame_files:
            self.frame_files = sorted(self.output_dir.glob("frame_*.json"))
        self.frame_files = list(self.frame_files)
        print(f"Found {len(self.frame_files)} frame files")

        self.frames = []
        for fp in self.frame_files:
            fr = load_frame(str(fp))
            if fr is not None:
                self.frames.append(fr)
        self.frames.sort(key=lambda f: f.time_hours)
        print(f"Loaded {len(self.frames)} frames")
        if self.frames:
            print(f"Time range: {self.frames[0].time_days:.2f} – {self.frames[-1].time_days:.2f} days")

        # Domain
        self.domain = None
        if 'domain' in self.config:
            s = self.config['domain'].get('side_length_um')
            if s: self.domain = np.array([s, s, s])
        if self.domain is None and self.frames and self.frames[0].domain is not None:
            self.domain = np.array(self.frames[0].domain)
        if self.domain is None and self.frames:
            self.domain = np.max(self.frames[0].positions, axis=0) + 100
        if self.domain is None:
            self.domain = np.array([700.0, 700.0, 700.0])
        print(f"Domain: {self.domain[0]:.1f} × {self.domain[1]:.1f} × {self.domain[2]:.1f} μm")

        self.void_analyzer = VoidAnalyzer(self.domain)

    # -----------------------------------------------------------------
    # Phase map: single frame, single plane
    # -----------------------------------------------------------------
    def plot_phase_map(self, frame_index=-1, axis=2, resolution=300, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        sp = self.domain[axis] / 2
        image = render_phase_map(frame, self.domain, axis, sp, resolution)
        ext, xl, yl = _extent_for_axis(self.domain, axis)

        fig, ax = plt.subplots(figsize=(8, 8))
        _plot_phase(ax, image, ext, xl, yl,
                    f'Phase Map — {"XYZ"[axis]} = {sp:.0f} μm  (t = {frame.time_days:.2f} d)')
        ax.legend(handles=PHASE_LEGEND, loc='upper right', fontsize=10)
        plt.tight_layout()
        if save:
            lab = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"phase_map_{"xyz"[axis]}_{lab}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Phase map evolution across time
    # -----------------------------------------------------------------
    def plot_phase_evolution(self, axis=2, n_times=6, resolution=200, save=True):
        if not self.frames: return None
        idxs = np.linspace(0, len(self.frames) - 1, n_times, dtype=int)
        sp = self.domain[axis] / 2
        ext, xl, yl = _extent_for_axis(self.domain, axis)

        fig, axes = plt.subplots(1, n_times, figsize=(3.5 * n_times, 4))
        if n_times == 1: axes = [axes]

        for col, fi in enumerate(idxs):
            frame = self.frames[fi]
            image = render_phase_map(frame, self.domain, axis, sp, resolution)
            _plot_phase(axes[col], image, ext, xl,
                        yl if col == 0 else '',
                        f't = {frame.time_days:.1f} d')
            if col > 0:
                axes[col].set_yticklabels([])

        fig.legend(handles=PHASE_LEGEND, loc='upper right',
                   bbox_to_anchor=(0.99, 0.99), fontsize=9)
        plt.suptitle(f'Phase Map Evolution ({"XYZ"[axis]} = {sp:.0f} μm)', fontsize=13)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f"phase_evolution_{"xyz"[axis]}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # 3-plane phase map of a single frame
    # -----------------------------------------------------------------
    def plot_three_plane(self, frame_index=-1, resolution=250, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, ax_id in enumerate([2, 1, 0]):
            sp = self.domain[ax_id] / 2
            image = render_phase_map(frame, self.domain, ax_id, sp, resolution)
            ext, xl, yl = _extent_for_axis(self.domain, ax_id)
            _plot_phase(axes[col], image, ext, xl, yl,
                        f'{"XY XZ YZ".split()[col]} ({"XYZ"[ax_id]} = {sp:.0f} μm)')
        fig.legend(handles=PHASE_LEGEND, loc='upper right',
                   bbox_to_anchor=(0.99, 0.99), fontsize=9)
        plt.suptitle(f't = {frame.time_hours:.1f}h ({frame.time_days:.2f} d)', fontsize=13)
        plt.tight_layout()
        if save:
            lab = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"three_plane_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Void fraction evolution
    # -----------------------------------------------------------------
    def plot_void_fraction_evolution(self, save=True):
        if not self.frames: return None
        print("Computing void fractions...")
        times = [f.time_days for f in self.frames]
        vfs = [self.void_analyzer.compute_void_fraction(f) for f in self.frames]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        ax = axes[0, 0]
        ax.plot(times, vfs, 'ko-', ms=4)
        if self.history and 'packing_fraction' in self.history:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, 1 - np.array(self.history['packing_fraction']), 'b-', alpha=0.5, label='history')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Void Fraction')
        ax.set_title('Overall Void Fraction'); ax.grid(True, alpha=0.3); ax.legend()

        ax = axes[0, 1]
        ni = min(5, len(self.frames))
        idxs = np.linspace(0, len(self.frames) - 1, ni, dtype=int)
        cols = plt.cm.viridis(np.linspace(0, 1, ni))
        for idx, c in zip(idxs, cols):
            z, vf = self.void_analyzer.compute_void_profile(self.frames[idx], axis=2)
            ax.plot(z, vf, '-', color=c, lw=2, label=f't={self.frames[idx].time_days:.1f}d')
        ax.set_xlabel('Z (μm)'); ax.set_ylabel('Local Void Fraction')
        ax.set_title('Void Profile (Z)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        pk = [1-v for v in vfs]
        ax.fill_between(times, 0, pk, alpha=0.3, color='steelblue')
        ax.plot(times, pk, 'b-', lw=2)
        ax.axhline(0.64, color='r', ls='--', alpha=0.7, label='RCP')
        ax.axhline(0.55, color='orange', ls='--', alpha=0.7, label='RLP')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction')
        ax.set_title('Packing Density'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.8)

        ax = axes[1, 1]
        frame = self.frames[-1]
        fm, im = frame.functional_mask, frame.inert_mask
        sf = [s.sphericity for i, s in enumerate(frame.shapes) if fm[i]]
        si = [s.sphericity for i, s in enumerate(frame.shapes) if im[i]]
        af = [s.aspect_ratio for i, s in enumerate(frame.shapes) if fm[i]]
        ai = [s.aspect_ratio for i, s in enumerate(frame.shapes) if im[i]]
        x = np.arange(2); w = 0.35
        ax.bar(x-w/2, [np.mean(sf), np.mean(si)], w, label='Sphericity', color='steelblue', alpha=0.7)
        ax.bar(x+w/2, [np.mean(af)/2, np.mean(ai)/2], w, label='AR/2', color='coral', alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(['Functional', 'Inert'])
        ax.set_title('Shape Statistics'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "void_fraction_evolution.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Detailed single frame
    # -----------------------------------------------------------------
    def plot_detailed_frame(self, frame_index=-1, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3)

        for col, ax_id in enumerate([2, 1, 0]):
            ax = fig.add_subplot(gs[0, col])
            sp = self.domain[ax_id] / 2
            image = render_phase_map(frame, self.domain, ax_id, sp, 250)
            ext, xl, yl = _extent_for_axis(self.domain, ax_id)
            _plot_phase(ax, image, ext, xl, yl,
                        f'{"XY XZ YZ".split()[col]} mid-slice')

        ax = fig.add_subplot(gs[1, 0])
        for a, lab, c in [(0,'X','red'), (1,'Y','green'), (2,'Z','blue')]:
            p, vf = self.void_analyzer.compute_void_profile(frame, a)
            ax.plot(p, vf, '-', color=c, lw=2, label=f'{lab}')
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Void Fraction')
        ax.set_title('Void Profiles'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        fr = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.functional_mask[i]]
        ir = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.inert_mask[i]]
        ax.hist(fr, bins=15, alpha=0.6, color='red', label='Functional')
        ax.hist(ir, bins=15, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equiv. Radius (μm)'); ax.set_ylabel('Count')
        ax.set_title('Size Distribution'); ax.legend()

        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        nf = int(np.sum(frame.functional_mask))
        ni = int(np.sum(frame.inert_mask))
        vf = self.void_analyzer.compute_void_fraction(frame)
        stats = (
            f"Time: {frame.time_hours:.1f}h ({frame.time_days:.2f}d)\n\n"
            f"Functional: {nf}   Inert: {ni}   Total: {frame.n_granules}\n\n"
            f"Mean R func: {np.mean(fr):.1f} μm\n"
            f"Mean R inert: {np.mean(ir):.1f} μm\n\n"
            f"Void fraction: {vf:.3f}\n"
            f"Packing fraction: {1-vf:.3f}\n\n"
            f"Total cells: {int(np.sum(frame.n_cells))}\n"
            f"Domain: {self.domain[0]:.0f}³ μm"
        )
        ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=11,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Detailed — t = {frame.time_days:.2f} d', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index == -1 else f"frame{frame_index}"
            fp = self.viz_dir / f"detailed_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # History summary
    # -----------------------------------------------------------------
    def plot_history(self, save=True):
        if not self.history:
            print("No history data"); return None
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
        if 'max_overlap_um' in self.history:
            ax2 = ax.twinx()
            ax2.plot(t, self.history['max_overlap_um'], 'm--', lw=1.5, alpha=0.7)
            ax2.set_ylabel('Max Overlap (μm)', color='m')
            ax2.tick_params(axis='y', labelcolor='m')

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "history_summary.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # -----------------------------------------------------------------
    # Slice animation (phase map version)
    # -----------------------------------------------------------------
    def create_slice_animation(self, axis=2, resolution=200, fps=6, save=True):
        if not HAS_ANIMATION or not self.frames:
            return None
        print(f"Creating phase-map animation ({len(self.frames)} frames)...")

        sp = self.domain[axis] / 2
        ext, xl, yl = _extent_for_axis(self.domain, axis)

        fig, axes_pair = plt.subplots(1, 2, figsize=(14, 6))
        ax_img = axes_pair[0]
        ax_met = axes_pair[1]

        # Pre-render first frame to set up imshow
        img0 = render_phase_map(self.frames[0], self.domain, axis, sp, resolution)
        im = ax_img.imshow(img0.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
                           extent=ext, aspect='equal', interpolation='nearest')
        ax_img.set_xlabel(xl); ax_img.set_ylabel(yl)
        ax_img.legend(handles=PHASE_LEGEND, loc='upper right', fontsize=8)

        if self.history and len(self.history.get('time_hours', [])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax_met.plot(th, self.history['mean_coordination'], 'g-', lw=2)
            ax_met.set_xlabel('Time (days)'); ax_met.set_ylabel('Coordination', color='g')
            ax_met.tick_params(axis='y', labelcolor='g'); ax_met.grid(True, alpha=0.3)
            ax2 = ax_met.twinx()
            ax2.plot(th, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
            ax2.set_ylabel('Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')

        time_line = ax_met.axvline(x=0, color='black', lw=2, ls='--')
        title = fig.suptitle('', fontsize=12, fontweight='bold')

        def update(fi):
            frame = self.frames[fi]
            img = render_phase_map(frame, self.domain, axis, sp, resolution)
            im.set_data(img.T)
            time_line.set_xdata([frame.time_days, frame.time_days])
            title.set_text(f't = {frame.time_hours:.1f}h ({frame.time_days:.2f}d)')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames),
                             interval=1000 // fps, blit=True)
        plt.tight_layout()

        if save:
            fp = self.viz_dir / f"phase_animation_{"xyz"[axis]}.gif"
            try:
                anim.save(str(fp), writer=PillowWriter(fps=fps), dpi=120)
                print(f"Saved: {fp}")
            except Exception as e:
                print(f"Error saving GIF: {e}"); fp = None
            plt.close(fig)
            return str(fp) if fp else None
        else:
            plt.show(); return None

    # -----------------------------------------------------------------
    # Live snapshot (for watch mode)
    # -----------------------------------------------------------------
    def plot_live_snapshot(self, save=True):
        if not self.frames: return None
        frame = self.frames[-1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        # Top-left: XY phase map
        sp = self.domain[2] / 2
        img = render_phase_map(frame, self.domain, 2, sp, 200)
        ext, xl, yl = _extent_for_axis(self.domain, 2)
        _plot_phase(axes[0, 0], img, ext, xl, yl,
                    f'XY Phase Map (t = {frame.time_hours:.1f}h)')

        # Top-right: XZ phase map
        sp2 = self.domain[1] / 2
        img2 = render_phase_map(frame, self.domain, 1, sp2, 200)
        ext2, xl2, yl2 = _extent_for_axis(self.domain, 1)
        _plot_phase(axes[0, 1], img2, ext2, xl2, yl2,
                    f'XZ Phase Map (t = {frame.time_hours:.1f}h)')

        # Bottom-left: coordination + bridges
        ax = axes[1, 0]
        if self.history and len(self.history.get('time_hours', [])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, self.history['mean_coordination'], 'g-', lw=2)
            ax.set_xlabel('Time (days)'); ax.set_ylabel('Coordination', color='g')
            ax.tick_params(axis='y', labelcolor='g'); ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(th, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
            ax2.set_ylabel('Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')
            ax.axvline(x=frame.time_hours / 24, color='black', ls='--', lw=1.5)
        ax.set_title('Network Evolution')

        # Bottom-right: packing + overlap
        ax = axes[1, 1]
        if self.history and len(self.history.get('time_hours', [])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, self.history['packing_fraction'], 'b-', lw=2)
            ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing φ', color='b')
            ax.tick_params(axis='y', labelcolor='b'); ax.grid(True, alpha=0.3)
            if 'max_overlap_um' in self.history:
                ax2 = ax.twinx()
                ax2.plot(th, self.history['max_overlap_um'], 'm-', lw=1.5, alpha=0.7)
                ax2.set_ylabel('Max Overlap (μm)', color='m')
                ax2.tick_params(axis='y', labelcolor='m')
        ax.set_title('Packing & Overlap')

        fig.legend(handles=PHASE_LEGEND, loc='upper center', ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 1.0))
        plt.suptitle(f'{self.config_name} — t = {frame.time_hours:.1f}h — '
                     f'{frame.n_granules} granules — {len(self.frames)} frames',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / "live_snapshot.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
        return None

    # -----------------------------------------------------------------
    # Live reload
    # -----------------------------------------------------------------
    def reload_frames(self) -> int:
        old_count = len(self.frames)
        for pat in [f"{self.config_name}_history.json", "history.json"]:
            p = self.output_dir / pat
            if p.exists():
                try:
                    with open(p) as f: self.history = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass
                break
        new_files = sorted(self.output_dir.glob(f"{self.config_name}_frame_*.json"))
        if not new_files:
            new_files = sorted(self.output_dir.glob("frame_*.json"))
        known = {str(fp) for fp in self.frame_files}
        added = 0
        for fp in new_files:
            if str(fp) not in known:
                fr = load_frame(str(fp))
                if fr is not None:
                    self.frames.append(fr)
                    self.frame_files.append(fp)
                    added += 1
        if added > 0:
            self.frames.sort(key=lambda f: f.time_hours)
            print(f"  [watch] +{added} frames → {len(self.frames)} total "
                  f"(t = {self.frames[-1].time_hours:.1f}h)")
        return added

    # -----------------------------------------------------------------
    # Watch mode
    # -----------------------------------------------------------------
    def watch(self, interval_seconds=30.0, full_plots_every=5):
        print(f"\n{'='*60}")
        print(f"WATCH MODE — polling every {interval_seconds:.0f}s")
        print(f"  Output: {self.output_dir}")
        print(f"  Snapshot: {self.viz_dir / 'live_snapshot.png'}")
        print(f"  Full plots every {full_plots_every} cycles.  Ctrl+C to stop.")
        print(f"{'='*60}\n")
        matplotlib.use('Agg')
        cycle = 0
        try:
            while True:
                n_new = self.reload_frames()
                if n_new > 0 or cycle == 0:
                    print("  [watch] Generating live snapshot...")
                    self.plot_live_snapshot(save=True)
                    if cycle % full_plots_every == 0 and len(self.frames) >= 2:
                        print("  [watch] Regenerating full plots...")
                        for fn in [self.plot_history, lambda: self.plot_phase_map(axis=2),
                                   self.plot_void_fraction_evolution,
                                   lambda: self.plot_phase_evolution(axis=2)]:
                            try: fn(save=True) if 'save' in fn.__code__.co_varnames else fn()
                            except Exception as e: print(f"    skipped: {e}")
                        print("  [watch] Done.")
                    cycle += 1
                final = list(self.output_dir.glob("*_frame_final.json"))
                if final and len(self.frames) > 2:
                    self.reload_frames()
                    print("\n  [watch] Final frame detected — generating all plots...")
                    self.create_all()
                    print("  [watch] Complete."); break
                time_module.sleep(interval_seconds)
        except KeyboardInterrupt:
            print(f"\n  [watch] Interrupted with {len(self.frames)} frames.")
            if len(self.frames) >= 2: self.create_all()

    # -----------------------------------------------------------------
    # Generate all
    # -----------------------------------------------------------------
    def create_all(self):
        print(f"\n{'='*60}\nGenerating All Visualizations\n{'='*60}")

        print("\n1. History summary...")
        self.plot_history()

        print("\n2. Void fraction evolution...")
        self.plot_void_fraction_evolution()

        print("\n3. Phase map evolution (XY, XZ)...")
        for ax in [2, 1]:
            self.plot_phase_evolution(axis=ax)

        print("\n4. Three-plane phase maps (initial + final)...")
        if self.frames:
            self.plot_three_plane(frame_index=0)
            self.plot_three_plane(frame_index=-1)

        print("\n5. Detailed frames...")
        if self.frames:
            self.plot_detailed_frame(0)
            self.plot_detailed_frame(-1)

        print("\n6. Phase map final (all planes)...")
        if self.frames:
            for ax in [2, 1, 0]:
                self.plot_phase_map(frame_index=-1, axis=ax)

        print("\n7. Slice animation...")
        if self.frames:
            self.create_slice_animation(axis=2, fps=6)

        print(f"\nAll saved to: {self.viz_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize granule simulation (phase maps)')
    parser.add_argument('--input', '-i', type=str, default='./simulations/Trial6')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--phase-map', '-p', action='store_true')
    parser.add_argument('--watch', '-w', action='store_true')
    parser.add_argument('--watch-interval', type=float, default=30.0)
    parser.add_argument('--slice-animation', '-s', action='store_true')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--axis', type=int, default=2)

    args = parser.parse_args()
    viz = NonSphericalVisualizer(args.input)

    if args.watch:
        viz.watch(interval_seconds=args.watch_interval)
    elif args.phase_map:
        viz.plot_phase_map(axis=args.axis)
    elif args.slice_animation:
        viz.create_slice_animation(axis=args.axis, fps=args.fps)
    elif args.all:
        viz.create_all()
    else:
        viz.create_all()


if __name__ == "__main__":
    main()