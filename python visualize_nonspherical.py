"""
Visualization for Jammed Non-Spherical Granule Simulations
===========================================================

Requirements:
    pip install numpy matplotlib pillow

Usage:
    python visualize_nonspherical.py --input ./jammed_nonspherical_sim
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from pathlib import Path
import json
import argparse
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAS_ANIMATION = True
except ImportError:
    HAS_ANIMATION = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ShapeData:
    a: float
    b: float
    c: float
    n: float
    roughness: float
    
    @property
    def equivalent_radius(self) -> float:
        V = (4/3) * np.pi * self.a * self.b * self.c
        return (3 * V / (4 * np.pi)) ** (1/3)


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
    def time_days(self) -> float:
        return self.time_hours / 24.0
    
    @property
    def functional_mask(self) -> np.ndarray:
        return self.types == 0
    
    @property
    def inert_mask(self) -> np.ndarray:
        return self.types == 1


def load_frame(filepath: str) -> Optional[FrameData]:
    try:
        with open(filepath) as f:
            data = json.load(f)
        
        shapes = [
            ShapeData(s['a'], s['b'], s['c'], s['n'], s['roughness'])
            for s in data['shapes']
        ]
        
        return FrameData(
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
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


# =============================================================================
# VOID ANALYSIS
# =============================================================================

class VoidAnalyzer:
    def __init__(self, domain: np.ndarray, resolution: int = 50):
        self.domain = domain
        self.resolution = resolution
    
    def compute_void_fraction(self, frame: FrameData) -> float:
        total_volume = sum((4/3) * np.pi * s.a * s.b * s.c for s in frame.shapes)
        domain_volume = np.prod(self.domain)
        return 1.0 - total_volume / domain_volume
    
    def compute_void_profile(self, frame: FrameData, axis: int = 2,
                             n_slices: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.linspace(0, self.domain[axis], n_slices + 1)
        positions = 0.5 * (positions[:-1] + positions[1:])
        void_fractions = []
        
        for pos in positions:
            slice_thickness = self.domain[axis] / n_slices
            slice_volume = 0
            
            for i, shape in enumerate(frame.shapes):
                center = frame.positions[i, axis]
                r_eq = shape.equivalent_radius
                
                if abs(center - pos) < r_eq + slice_thickness/2:
                    d = abs(center - pos)
                    if d < r_eq:
                        h = min(slice_thickness, r_eq - d + slice_thickness/2)
                        cap_vol = np.pi * h**2 * (r_eq - h/3)
                        slice_volume += min(cap_vol, (4/3) * np.pi * r_eq**3)
            
            total_slice_vol = self.domain[0] * self.domain[1] * slice_thickness
            void_fractions.append(1.0 - slice_volume / total_slice_vol)
        
        return positions, np.array(void_fractions)


# =============================================================================
# CROSS SECTION RENDERING
# =============================================================================

def render_cross_section(frame: FrameData, domain: np.ndarray, axis: int,
                         position: float, resolution: int = 200) -> np.ndarray:
    if axis == 0:
        dims = [1, 2]
        extent = [domain[1], domain[2]]
    elif axis == 1:
        dims = [0, 2]
        extent = [domain[0], domain[2]]
    else:
        dims = [0, 1]
        extent = [domain[0], domain[1]]
    
    image = np.zeros((resolution, resolution), dtype=np.float32)
    dx = extent[0] / resolution
    dy = extent[1] / resolution
    
    for i in range(frame.n_granules):
        pos = frame.positions[i]
        shape = frame.shapes[i]
        r_eq = shape.equivalent_radius
        
        dist_to_plane = abs(pos[axis] - position)
        if dist_to_plane >= r_eq:
            continue
        
        r_cross = np.sqrt(max(0, r_eq**2 - dist_to_plane**2))
        cx, cy = pos[dims[0]], pos[dims[1]]
        
        value = 1.0 if frame.types[i] == 0 else 2.0
        
        ix_min = max(0, int((cx - r_cross) / dx))
        ix_max = min(resolution - 1, int((cx + r_cross) / dx) + 1)
        iy_min = max(0, int((cy - r_cross) / dy))
        iy_max = min(resolution - 1, int((cy + r_cross) / dy) + 1)
        
        for ix in range(ix_min, ix_max):
            for iy in range(iy_min, iy_max):
                px = (ix + 0.5) * dx
                py = (iy + 0.5) * dy
                if (px - cx)**2 + (py - cy)**2 <= r_cross**2:
                    image[ix, iy] = value
    
    return image


# =============================================================================
# MAIN VISUALIZER
# =============================================================================

class NonSphericalVisualizer:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Load config
        config_path = self.output_dir / "config.json"
        self.config = {}
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
        
        # Load history
        history_path = self.output_dir / "history.json"
        self.history = None
        if history_path.exists():
            with open(history_path) as f:
                self.history = json.load(f)
        
        # Load frames
        self.frame_files = sorted(self.output_dir.glob("frame_*.json"))
        print(f"Found {len(self.frame_files)} frames")
        
        self.frames = []
        for fp in self.frame_files:
            frame = load_frame(str(fp))
            if frame is not None:
                self.frames.append(frame)
        
        self.frames.sort(key=lambda f: f.time_hours)
        print(f"Loaded {len(self.frames)} frames")
        
        if self.frames:
            print(f"Time range: {self.frames[0].time_days:.2f} - {self.frames[-1].time_days:.2f} days")
        
        # Get domain
        self.domain = None
        if 'domain_x' in self.config:
            self.domain = np.array([
                self.config['domain_x'],
                self.config['domain_y'],
                self.config['domain_z']
            ])
        elif self.frames and self.frames[0].domain:
            self.domain = np.array(self.frames[0].domain)
        elif self.frames:
            positions = self.frames[0].positions
            max_r = max(s.equivalent_radius for s in self.frames[0].shapes)
            self.domain = np.max(positions, axis=0) + max_r * 2
        else:
            self.domain = np.array([500.0, 500.0, 500.0])
        
        print(f"Domain: {self.domain[0]:.1f} × {self.domain[1]:.1f} × {self.domain[2]:.1f} μm")
        
        self.cmap = LinearSegmentedColormap.from_list('granules', ['white', 'red', 'blue'], N=3)
        self.void_analyzer = VoidAnalyzer(self.domain)
        
        if self.frames:
            self.mean_granule_radius = np.mean([s.equivalent_radius for s in self.frames[0].shapes])
        else:
            self.mean_granule_radius = 50.0
    
    def plot_history(self, save: bool = True):
        """Plot simulation history metrics."""
        if not self.history:
            print("No history data available")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = np.array(self.history['time_hours']) / 24
        
        axes[0, 0].plot(t, self.history['n_contacts'], 'b-', lw=2, label='Contacts')
        axes[0, 0].plot(t, self.history['n_bridges'], 'r-', lw=2, label='Bridges')
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title('Interactions')
        
        axes[0, 1].plot(t, self.history['mean_coordination'], 'g-', lw=2)
        axes[0, 1].set_xlabel('Time (days)')
        axes[0, 1].set_ylabel('Mean Coordination')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Contact Network')
        
        axes[1, 0].semilogy(t, self.history['max_velocity'], 'm-', lw=2)
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Max Velocity (μm/hr)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title('Activity Level')
        
        axes[1, 1].plot(t, self.history['packing_fraction'], 'c-', lw=2)
        axes[1, 1].set_xlabel('Time (days)')
        axes[1, 1].set_ylabel('Packing Fraction')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title('Packing Density')
        
        plt.suptitle('Simulation History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            filepath = self.viz_dir / "history_summary.png"
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        plt.close()
        return fig
    
    def plot_void_fraction_evolution(self, save: bool = True):
        """Plot void fraction over time."""
        if not self.frames:
            return None
        
        print("Computing void fractions...")
        times, void_fractions = [], []
        for frame in self.frames:
            times.append(frame.time_days)
            void_fractions.append(self.void_analyzer.compute_void_fraction(frame))
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].plot(times, void_fractions, 'ko-', markersize=4)
        axes[0, 0].set_xlabel('Time (days)')
        axes[0, 0].set_ylabel('Void Fraction')
        axes[0, 0].set_title('Overall Void Fraction')
        axes[0, 0].grid(True, alpha=0.3)
        
        n_profiles = min(5, len(self.frames))
        indices = np.linspace(0, len(self.frames)-1, n_profiles, dtype=int)
        colors = plt.cm.viridis(np.linspace(0, 1, n_profiles))
        for idx, color in zip(indices, colors):
            z, vf = self.void_analyzer.compute_void_profile(self.frames[idx], axis=2)
            axes[0, 1].plot(z, vf, '-', color=color, lw=2, label=f't={self.frames[idx].time_days:.1f}d')
        axes[0, 1].set_xlabel('Z position (μm)')
        axes[0, 1].set_ylabel('Local Void Fraction')
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title('Void Profile (Z-axis)')
        
        packing = [1 - vf for vf in void_fractions]
        axes[1, 0].fill_between(times, 0, packing, alpha=0.3, color='steelblue')
        axes[1, 0].plot(times, packing, 'b-', lw=2)
        axes[1, 0].axhline(y=0.64, color='r', ls='--', alpha=0.7, label='RCP')
        axes[1, 0].axhline(y=0.55, color='orange', ls='--', alpha=0.7, label='RLP')
        axes[1, 0].set_xlabel('Time (days)')
        axes[1, 0].set_ylabel('Packing Fraction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 0.8)
        axes[1, 0].set_title('Packing Evolution')
        
        if len(times) > 1:
            dt = np.diff(times)
            dvf = np.diff(void_fractions)
            rate = dvf / dt
            t_mid = [(times[i] + times[i+1])/2 for i in range(len(times)-1)]
            axes[1, 1].plot(t_mid, rate, 'g-', lw=2)
            axes[1, 1].axhline(y=0, color='k', ls='-', alpha=0.3)
            axes[1, 1].set_xlabel('Time (days)')
            axes[1, 1].set_ylabel('d(Void Fraction)/dt')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_title('Compaction Rate')
        
        plt.tight_layout()
        if save:
            filepath = self.viz_dir / "void_fraction_evolution.png"
            plt.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")
        plt.close()
        return fig
    
    def plot_cross_section_evolution(self, axis: int = 2, n_times: int = 6, save: bool = True):
        """Plot cross-sections at multiple times."""
        if not self.frames:
            return None
        
        frame_indices = np.linspace(0, len(self.frames)-1, n_times, dtype=int)
        slice_pos = self.domain[axis] / 2
        
        fig, axes = plt.subplots(1, n_times, figsize=(3*n_times, 3))
        if n_times == 1:
            axes = [axes]
        
        axis_labels = ['X', 'Y', 'Z']
        if axis == 2:
            extent = [0, self.domain[0], 0, self.domain[1]]
        elif axis == 1:
            extent = [0, self.domain[0], 0, self.domain[2]]
        else:
            extent = [0, self.domain[1], 0, self.domain[2]]
        
        for col, fi in enumerate(frame_indices):
            frame = self.frames[fi]
            image = render_cross_section(frame, self.domain, axis, slice_pos, resolution=150)
            
            axes[col].imshow(image.T, origin='lower', cmap=self.cmap, vmin=0, vmax=2, extent=extent)
            axes[col].set_title(f't = {frame.time_days:.1f} days', fontsize=10)
            axes[col].set_xlabel(f'{axis_labels[(axis+1)%3]} (μm)')
            if col == 0:
                axes[col].set_ylabel(f'{axis_labels[(axis+2)%3]} (μm)')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='white', edgecolor='black', label='Void'),
            Patch(facecolor='red', label='Functional'),
            Patch(facecolor='blue', label='Inert')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        
        plt.suptitle(f'Cross-Section Evolution ({axis_labels[axis]}={slice_pos:.0f}μm)', fontsize=12)
        plt.tight_layout()
        
        if save:
            filepath = self.viz_dir / f"cross_section_evolution_{axis_labels[axis].lower()}.png"
            plt.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")
        plt.close()
        return fig
    
    def plot_detailed_frame(self, frame_index: int = -1, save: bool = True):
        """Detailed view of a single frame."""
        if not self.frames:
            return None
        
        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3)
        
        # Three cross-sections
        for col, axis in enumerate([2, 1, 0]):
            ax = fig.add_subplot(gs[0, col])
            slice_pos = self.domain[axis] / 2
            image = render_cross_section(frame, self.domain, axis, slice_pos, resolution=150)
            
            if axis == 2:
                extent = [0, self.domain[0], 0, self.domain[1]]
                labels = ('X', 'Y')
            elif axis == 1:
                extent = [0, self.domain[0], 0, self.domain[2]]
                labels = ('X', 'Z')
            else:
                extent = [0, self.domain[1], 0, self.domain[2]]
                labels = ('Y', 'Z')
            
            ax.imshow(image.T, origin='lower', cmap=self.cmap, vmin=0, vmax=2, extent=extent)
            ax.set_xlabel(f'{labels[0]} (μm)')
            ax.set_ylabel(f'{labels[1]} (μm)')
            ax.set_title(f'{["YZ","XZ","XY"][axis]} plane')
        
        # Void profiles
        ax = fig.add_subplot(gs[1, 0])
        for axis, label, color in [(0, 'X', 'red'), (1, 'Y', 'green'), (2, 'Z', 'blue')]:
            pos, vf = self.void_analyzer.compute_void_profile(frame, axis)
            ax.plot(pos, vf, '-', color=color, lw=2, label=f'{label}-axis')
        ax.set_xlabel('Position (μm)')
        ax.set_ylabel('Local Void Fraction')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title('Void Profiles')
        
        # Size distribution
        ax = fig.add_subplot(gs[1, 1])
        func_radii = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.functional_mask[i]]
        inert_radii = [s.equivalent_radius for i, s in enumerate(frame.shapes) if frame.inert_mask[i]]
        ax.hist(func_radii, bins=12, alpha=0.6, color='red', label='Functional')
        ax.hist(inert_radii, bins=12, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equivalent Radius (μm)')
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title('Size Distribution')
        
        # Stats
        ax = fig.add_subplot(gs[1, 2])
        ax.axis('off')
        void_frac = self.void_analyzer.compute_void_fraction(frame)
        stats = f"""
Time: {frame.time_hours:.1f}h ({frame.time_days:.2f} days)

Granules:
  Functional: {np.sum(frame.functional_mask)}
  Inert: {np.sum(frame.inert_mask)}
  Total: {frame.n_granules}

Radii (mean):
  Functional: {np.mean(func_radii):.1f} μm
  Inert: {np.mean(inert_radii):.1f} μm

Packing:
  Void fraction: {void_frac:.3f}
  Packing fraction: {1-void_frac:.3f}

Cells: {np.sum(frame.n_cells)} total
        """
        ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Frame Analysis - t = {frame.time_days:.2f} days', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            label = "final" if frame_index == -1 else f"frame{frame_index}"
            filepath = self.viz_dir / f"detailed_{label}.png"
            plt.savefig(filepath, dpi=150)
            print(f"Saved: {filepath}")
        plt.close()
        return fig
    
    def create_slice_animation(self, axis: int = 2, slice_thickness: float = None,
                                fps: int = 6, save: bool = True):
        """Create animation of a slice ~1 granule thick."""
        if not HAS_ANIMATION or not self.frames:
            print("Animation not available")
            return None
        
        print(f"Creating slice animation with {len(self.frames)} frames...")
        
        slice_position = self.domain[axis] / 2
        if slice_thickness is None:
            slice_thickness = self.mean_granule_radius * 2.2
        
        print(f"  Slice at {'XYZ'[axis]}={slice_position:.0f}μm, thickness={slice_thickness:.0f}μm")
        
        axis_labels = ['X', 'Y', 'Z']
        if axis == 2:
            dim1, dim2 = 0, 1
            xlabel, ylabel = 'X (μm)', 'Y (μm)'
        elif axis == 1:
            dim1, dim2 = 0, 2
            xlabel, ylabel = 'X (μm)', 'Z (μm)'
        else:
            dim1, dim2 = 1, 2
            xlabel, ylabel = 'Y (μm)', 'Z (μm)'
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_slice, ax_metrics = axes
        
        ax_slice.set_xlim(0, self.domain[dim1])
        ax_slice.set_ylim(0, self.domain[dim2])
        ax_slice.set_aspect('equal')
        ax_slice.set_xlabel(xlabel)
        ax_slice.set_ylabel(ylabel)
        ax_slice.set_facecolor('#f0f0f0')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Functional'),
            Patch(facecolor='blue', alpha=0.5, label='Inert')
        ]
        ax_slice.legend(handles=legend_elements, loc='upper right')
        
        time_line = None
        if self.history and len(self.history.get('time_hours', [])) > 0:
            try:
                t_hist = np.array(self.history['time_hours']) / 24
                ax_metrics.plot(t_hist, self.history['mean_coordination'], 'g-', lw=2)
                ax_metrics.set_xlabel('Time (days)')
                ax_metrics.set_ylabel('Mean Coordination', color='g')
                ax_metrics.grid(True, alpha=0.3)
                ax_metrics.set_xlim(0, max(t_hist) * 1.05)
                
                if 'n_bridges' in self.history:
                    ax2 = ax_metrics.twinx()
                    ax2.plot(t_hist, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
                    ax2.set_ylabel('Cell Bridges', color='r')
                
                time_line = ax_metrics.axvline(x=0, color='black', lw=2, ls='--')
            except Exception as e:
                print(f"  Warning: Could not plot history: {e}")
        else:
            ax_metrics.text(0.5, 0.5, 'No history data', ha='center', va='center', 
                           transform=ax_metrics.transAxes, fontsize=12)
            ax_metrics.set_xlabel('Time (days)')
        
        title = fig.suptitle('', fontsize=12, fontweight='bold')
        
        def update(frame_idx):
            frame = self.frames[frame_idx]
            
            # Clear old patches properly
            while ax_slice.patches:
                ax_slice.patches[0].remove()
            
            slice_min = slice_position - slice_thickness / 2
            slice_max = slice_position + slice_thickness / 2
            count = 0
            
            for i in range(frame.n_granules):
                pos = frame.positions[i]
                shape = frame.shapes[i]
                r_eq = shape.equivalent_radius
                
                if pos[axis] - r_eq > slice_max or pos[axis] + r_eq < slice_min:
                    continue
                
                dist = abs(pos[axis] - slice_position)
                if dist < r_eq:
                    r_cross = np.sqrt(max(1, r_eq**2 - dist**2))
                else:
                    r_cross = r_eq * 0.3
                
                x2d, y2d = pos[dim1], pos[dim2]
                color = 'red' if frame.types[i] == 0 else 'blue'
                alpha = 0.7 if frame.types[i] == 0 else 0.5
                if r_eq > 0:
                    alpha *= (1 - 0.3 * min(1, dist / r_eq))
                
                circle = Circle((x2d, y2d), r_cross, facecolor=color,
                               edgecolor='black', alpha=max(0.2, alpha), lw=0.5)
                ax_slice.add_patch(circle)
                count += 1
            
            if time_line is not None:
                time_line.set_xdata([frame.time_days, frame.time_days])
            
            title.set_text(f't = {frame.time_hours:.1f}h ({frame.time_days:.2f} days) | {count} granules in slice')
            return []
        
        anim = FuncAnimation(fig, update, frames=len(self.frames), interval=1000//fps, blit=False)
        plt.tight_layout()
        
        if save:
            filepath = self.viz_dir / f"slice_animation_{axis_labels[axis].lower()}.gif"
            print(f"  Saving to {filepath}...")
            try:
                anim.save(str(filepath), writer=PillowWriter(fps=fps), dpi=120)
                print(f"  Saved: {filepath}")
            except Exception as e:
                print(f"  Error: {e}")
                filepath = None
            plt.close()
            return str(filepath) if filepath else None
        else:
            plt.show()
            return None
    
    def create_all(self):
        """Generate all visualizations."""
        print("\n" + "="*60)
        print("Generating All Visualizations")
        print("="*60)
        
        print("\n1. History summary...")
        self.plot_history()
        
        print("\n2. Void fraction evolution...")
        self.plot_void_fraction_evolution()
        
        print("\n3. Cross-section evolution (XY)...")
        self.plot_cross_section_evolution(axis=2)
        
        print("\n4. Cross-section evolution (XZ)...")
        self.plot_cross_section_evolution(axis=1)
        
        print("\n5. Detailed initial frame...")
        if self.frames:
            self.plot_detailed_frame(0)
        
        print("\n6. Detailed final frame...")
        if self.frames:
            self.plot_detailed_frame(-1)
        
        print("\n7. Slice animation...")
        if self.frames:
            self.create_slice_animation(axis=2, fps=6)
        
        print(f"\nAll visualizations saved to: {self.viz_dir}")


def main():
    parser = argparse.ArgumentParser(description='Visualize simulation results')
    parser.add_argument('--input', '-i', type=str, default='./jammed_nonspherical_sim_2')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--slice-animation', '-s', action='store_true')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--axis', type=int, default=2)
    
    args = parser.parse_args()
    
    viz = NonSphericalVisualizer(args.input)
    
    if args.slice_animation:
        viz.create_slice_animation(axis=args.axis, fps=args.fps)
    else:
        viz.create_all()


if __name__ == "__main__":
    main()