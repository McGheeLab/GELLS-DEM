"""
Visualization for Superellipsoid Granule Simulations
=====================================================

Phase maps: point-in-superellipsoid test |x/a|^n + |y/b|^n + |z/c|^n <= 1
Packing fraction: voxel-based with 1-granule border exclusion
Cell maps: stress, aspect ratio, bridging state per cell

Usage:
    python dem_postprocess.py -i ./simulations/dem_config
    python dem_postprocess.py -i ./simulations/dem_config --watch
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
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

try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY_NDI = True
except ImportError:
    HAS_SCIPY_NDI = False


# =============================================================================
# DATA
# =============================================================================

@dataclass
class ShapeData:
    a: float; b: float; c: float; n: float; roughness: float
    @property
    def equivalent_radius(self): return (self.a*self.b*self.c)**(1/3)
    @property
    def sphericity(self):
        r=self.equivalent_radius; p=1.6075
        sa=4*np.pi*((self.a**p*self.b**p+self.a**p*self.c**p+self.b**p*self.c**p)/3)**(1/p)
        return min(1.0, 4*np.pi*r**2/sa)
    @property
    def aspect_ratio(self):
        return max(self.a,self.b,self.c)/min(self.a,self.b,self.c)

@dataclass
class CellData:
    """Per-cell data loaded from frame JSON."""
    total_cells: int
    cell_diameter: float
    parent: np.ndarray          # (n_cells,) int — parent granule index
    world_pos: np.ndarray       # (n_cells, 3)
    is_bridging: np.ndarray     # (n_cells,) bool
    bridge_target: np.ndarray   # (n_cells,) int
    bridge_endpoint: np.ndarray # (n_cells, 3)
    gap_um: np.ndarray          # (n_cells,)
    stress_nN: np.ndarray       # (n_cells,)
    aspect_ratio: np.ndarray    # (n_cells,)

@dataclass
class FrameData:
    time_hours:float; n_granules:int
    positions:np.ndarray; orientations:np.ndarray
    shapes:List[ShapeData]; types:np.ndarray; n_cells:np.ndarray
    domain:Optional[List[float]]=None
    cell_data:Optional[CellData]=None
    @property
    def time_days(self): return self.time_hours/24.0
    @property
    def functional_mask(self): return self.types==0
    @property
    def inert_mask(self): return self.types==1

def load_frame(filepath):
    try:
        with open(filepath) as f: data=json.load(f)
        key=None
        for k in ['true_shapes','sim_shapes','shapes']:
            if k in data: key=k; break
        if key is None: return None
        shapes=[ShapeData(s['a'],s['b'],s['c'],s.get('n',2.0),s.get('roughness',0.0)) for s in data[key]]

        # Load cell data if present
        cd = None
        if 'cell_data' in data:
            c = data['cell_data']
            cd = CellData(
                total_cells=c['total_cells'],
                cell_diameter=c['cell_diameter_um'],
                parent=np.array(c['parent'], dtype=int),
                world_pos=np.array(c['world_pos']),
                is_bridging=np.array(c['is_bridging'], dtype=bool),
                bridge_target=np.array(c['bridge_target'], dtype=int),
                bridge_endpoint=np.array(c['bridge_endpoint']),
                gap_um=np.array(c['gap_um']),
                stress_nN=np.array(c['stress_nN']),
                aspect_ratio=np.array(c['aspect_ratio']),
            )

        return FrameData(
            time_hours=data['time_hours'],n_granules=data['n_granules'],
            positions=np.array(data['positions']),orientations=np.array(data['orientations']),
            shapes=shapes,types=np.array(data['types']),n_cells=np.array(data['n_cells']),
            domain=data.get('domain'), cell_data=cd)
    except Exception as e:
        print(f"Error loading {filepath}: {e}"); return None

def _quat_to_rot(q):
    w,x,y,z=q
    return np.array([[1-2*y*y-2*z*z,2*x*y-2*w*z,2*x*z+2*w*y],
                     [2*x*y+2*w*z,1-2*x*x-2*z*z,2*y*z-2*w*x],
                     [2*x*z-2*w*y,2*y*z+2*w*x,1-2*x*x-2*y*y]])


# =============================================================================
# SUPERELLIPSOID PHASE MAP (vectorized per granule)
# =============================================================================

def render_phase_map(frame, domain, axis, position, resolution=300):
    if axis == 0:   dims=[1,2]; ext=[domain[1],domain[2]]
    elif axis == 1:  dims=[0,2]; ext=[domain[0],domain[2]]
    else:            dims=[0,1]; ext=[domain[0],domain[1]]

    image = np.zeros((resolution, resolution), dtype=np.float32)
    dx = ext[0]/resolution
    dy = ext[1]/resolution

    for i in range(frame.n_granules):
        shape = frame.shapes[i]
        pos = frame.positions[i]
        R = _quat_to_rot(frame.orientations[i])
        rb = max(shape.a, shape.b, shape.c)
        if abs(pos[axis] - position) > rb: continue
        value = 1.0 if frame.types[i] == 0 else 2.0
        cx, cy = pos[dims[0]], pos[dims[1]]
        ix_min = max(0, int((cx - rb) / dx) - 1)
        ix_max = min(resolution, int((cx + rb) / dx) + 2)
        iy_min = max(0, int((cy - rb) / dy) - 1)
        iy_max = min(resolution, int((cy + rb) / dy) + 2)
        if ix_max <= ix_min or iy_max <= iy_min: continue
        px = (np.arange(ix_min, ix_max) + 0.5) * dx
        py = (np.arange(iy_min, iy_max) + 0.5) * dy
        PX, PY = np.meshgrid(px, py, indexing='ij')
        world = np.empty(PX.shape + (3,))
        if axis == 0:   world[...,0]=position; world[...,1]=PX; world[...,2]=PY
        elif axis == 1: world[...,0]=PX; world[...,1]=position; world[...,2]=PY
        else:            world[...,0]=PX; world[...,1]=PY; world[...,2]=position
        relative = world - pos
        body = np.einsum('ij,...j->...i', R.T, relative)
        n_exp = shape.n
        se_val = (np.abs(body[...,0]/shape.a)**n_exp +
                  np.abs(body[...,1]/shape.b)**n_exp +
                  np.abs(body[...,2]/shape.c)**n_exp)
        inside = se_val <= 1.0
        image[ix_min:ix_max, iy_min:iy_max][inside] = value
    return image


# =============================================================================
# CELL VISUALIZATION HELPERS
# =============================================================================

def _cells_in_slice(cell_data, axis, position, thickness):
    """
    Select cells within a slab of given thickness centered at position along axis.
    Returns mask into cell_data arrays.
    """
    if cell_data is None or cell_data.total_cells == 0:
        return np.zeros(0, dtype=bool)
    coords = cell_data.world_pos[:, axis]
    return np.abs(coords - position) <= thickness / 2


def _project_cells(cell_data, axis, mask):
    """Project cell positions onto the 2D plane perpendicular to axis."""
    if axis == 0:   dims = [1, 2]
    elif axis == 1: dims = [0, 2]
    else:           dims = [0, 1]
    pos2d = cell_data.world_pos[mask][:, dims]
    return pos2d


def render_cell_scatter(frame, domain, axis, position, thickness=None, ax=None,
                        color_by='type', vmin=None, vmax=None, cmap=None, s=12):
    """
    Render cell positions as scatter on a 2D slice.

    color_by: 'type' | 'stress' | 'aspect_ratio' | 'bridging'
    Returns the scatter artist and the mask used.
    """
    cd = frame.cell_data
    if cd is None or cd.total_cells == 0:
        return None, np.zeros(0, dtype=bool)

    if thickness is None:
        # Default: mean cell diameter
        thickness = cd.cell_diameter * 1.5

    mask = _cells_in_slice(cd, axis, position, thickness)
    if np.sum(mask) == 0:
        return None, mask

    pos2d = _project_cells(cd, axis, mask)

    if color_by == 'type':
        # Color by parent granule type
        parent_types = frame.types[cd.parent[mask]]
        colors = np.where(parent_types == 0, 'red', 'blue')
        sc = ax.scatter(pos2d[:, 0], pos2d[:, 1], c=colors, s=s, alpha=0.7,
                        edgecolors='k', linewidths=0.3, zorder=5)
    elif color_by == 'stress':
        vals = cd.stress_nN[mask]
        if cmap is None: cmap = 'hot'
        if vmax is None: vmax = max(np.max(vals), 1e-6)
        if vmin is None: vmin = 0
        sc = ax.scatter(pos2d[:, 0], pos2d[:, 1], c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=s, alpha=0.8,
                        edgecolors='k', linewidths=0.2, zorder=5)
    elif color_by == 'aspect_ratio':
        vals = cd.aspect_ratio[mask]
        if cmap is None: cmap = 'viridis'
        if vmax is None: vmax = max(np.max(vals), 1.01)
        if vmin is None: vmin = 1.0
        sc = ax.scatter(pos2d[:, 0], pos2d[:, 1], c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=s, alpha=0.8,
                        edgecolors='k', linewidths=0.2, zorder=5)
    elif color_by == 'bridging':
        bridging = cd.is_bridging[mask]
        colors = np.where(bridging, 'orange', 'gray')
        sc = ax.scatter(pos2d[:, 0], pos2d[:, 1], c=colors, s=s, alpha=0.7,
                        edgecolors='k', linewidths=0.3, zorder=5)
    else:
        sc = ax.scatter(pos2d[:, 0], pos2d[:, 1], c='gray', s=s, alpha=0.5, zorder=5)

    return sc, mask


def render_cell_field(cell_data, domain, axis, position, thickness, field='stress',
                      resolution=100, sigma=None):
    """
    Render a smoothed 2D heatmap of a cell scalar field (stress or aspect_ratio)
    by splatting cell values onto a grid and Gaussian-smoothing.

    Returns: 2D array (resolution x resolution), extent list.
    """
    if cell_data is None or cell_data.total_cells == 0:
        return np.full((resolution, resolution), np.nan), [0, domain[0], 0, domain[1]]

    if axis == 0:   dims=[1,2]; ext=[0,domain[1],0,domain[2]]
    elif axis == 1: dims=[0,2]; ext=[0,domain[0],0,domain[2]]
    else:           dims=[0,1]; ext=[0,domain[0],0,domain[1]]

    mask = _cells_in_slice(cell_data, axis, position, thickness)
    if np.sum(mask) == 0:
        return np.full((resolution, resolution), np.nan), ext

    pos2d = cell_data.world_pos[mask][:, dims]

    if field == 'stress':
        vals = cell_data.stress_nN[mask]
    elif field == 'aspect_ratio':
        vals = cell_data.aspect_ratio[mask]
    else:
        vals = cell_data.stress_nN[mask]

    # Splat onto grid
    dx = (ext[1] - ext[0]) / resolution
    dy = (ext[3] - ext[2]) / resolution

    weight_grid = np.zeros((resolution, resolution), dtype=np.float64)
    value_grid = np.zeros((resolution, resolution), dtype=np.float64)

    ix = np.clip(((pos2d[:, 0] - ext[0]) / dx).astype(int), 0, resolution - 1)
    iy = np.clip(((pos2d[:, 1] - ext[2]) / dy).astype(int), 0, resolution - 1)

    # Accumulate (handles multiple cells in same pixel)
    np.add.at(weight_grid, (ix, iy), 1.0)
    np.add.at(value_grid, (ix, iy), vals)

    # Average where we have data
    has_data = weight_grid > 0
    result = np.full((resolution, resolution), np.nan)
    result[has_data] = value_grid[has_data] / weight_grid[has_data]

    # Gaussian smooth for visualization (fill NaN with 0 temporarily)
    if sigma is None:
        # Auto sigma based on cell diameter relative to pixel size
        cell_px = max(cell_data.cell_diameter / dx, 2)
        sigma = cell_px * 1.5

    if HAS_SCIPY_NDI and sigma > 0:
        fill = np.where(np.isnan(result), 0, result)
        fill_w = np.where(np.isnan(result), 0, 1.0)
        smooth_val = gaussian_filter(fill, sigma=sigma)
        smooth_w = gaussian_filter(fill_w, sigma=sigma)
        valid = smooth_w > 0.01
        result = np.full_like(result, np.nan)
        result[valid] = smooth_val[valid] / smooth_w[valid]

    return result, ext


def render_bridge_lines(cell_data, axis, position, thickness, ax, domain):
    """Draw lines connecting bridging cell pairs in the slice."""
    if cell_data is None or cell_data.total_cells == 0:
        return

    if axis == 0:   dims = [1, 2]
    elif axis == 1: dims = [0, 2]
    else:           dims = [0, 1]

    mask = _cells_in_slice(cell_data, axis, position, thickness)
    bridging = mask & cell_data.is_bridging

    indices = np.where(bridging)[0]
    for ci in indices:
        p1 = cell_data.world_pos[ci, dims]
        p2 = cell_data.bridge_endpoint[ci, dims]
        stress = cell_data.stress_nN[ci]
        alpha = min(1.0, 0.3 + 0.7 * stress / max(np.max(cell_data.stress_nN[bridging]), 1e-6))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'orange', lw=0.8, alpha=alpha, zorder=4)


# =============================================================================
# VOXEL-BASED PACKING FRACTION
# =============================================================================

def compute_voxel_packing_fast(frame, domain, margin, res=50):
    gmin = np.array([margin, margin, margin])
    gmax = np.array([domain[0]-margin, domain[1]-margin, domain[2]-margin])
    if np.any(gmax <= gmin): return 0.0
    xs = np.linspace(gmin[0], gmax[0], res)
    ys = np.linspace(gmin[1], gmax[1], res)
    zs = np.linspace(gmin[2], gmax[2], res)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    grid = np.stack([X, Y, Z], axis=-1)
    occupied = np.zeros((res, res, res), dtype=bool)
    for g in range(frame.n_granules):
        shape = frame.shapes[g]; pos = frame.positions[g]
        R = _quat_to_rot(frame.orientations[g])
        rb = max(shape.a, shape.b, shape.c)
        mask_x = (xs >= pos[0]-rb) & (xs <= pos[0]+rb)
        mask_y = (ys >= pos[1]-rb) & (ys <= pos[1]+rb)
        mask_z = (zs >= pos[2]-rb) & (zs <= pos[2]+rb)
        if not (np.any(mask_x) and np.any(mask_y) and np.any(mask_z)): continue
        ix=np.where(mask_x)[0]; iy=np.where(mask_y)[0]; iz=np.where(mask_z)[0]
        sub = grid[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1]
        rel = sub - pos
        body = np.einsum('ij,...j->...i', R.T, rel)
        n_exp = shape.n
        se_val = (np.abs(body[...,0]/shape.a)**n_exp +
                  np.abs(body[...,1]/shape.b)**n_exp +
                  np.abs(body[...,2]/shape.c)**n_exp)
        occupied[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1] |= se_val <= 1.0
    return np.sum(occupied) / occupied.size


def compute_void_profile(frame, domain, margin, axis=2, n_slices=30, slice_res=80):
    gmin_ax = margin; gmax_ax = domain[axis] - margin
    if gmax_ax <= gmin_ax: return np.array([]), np.array([])
    positions = np.linspace(gmin_ax, gmax_ax, n_slices)
    voids = []
    for pos in positions:
        img = render_phase_map(frame, domain, axis, pos, resolution=slice_res)
        if axis == 0:   ext = [domain[1], domain[2]]
        elif axis == 1: ext = [domain[0], domain[2]]
        else:           ext = [domain[0], domain[1]]
        dx = ext[0]/slice_res; dy = ext[1]/slice_res
        bpx = int(margin/dx); bpy = int(margin/dy)
        interior = img[bpx:slice_res-bpx, bpy:slice_res-bpy]
        voids.append(np.sum(interior==0)/interior.size if interior.size>0 else 1.0)
    return positions, np.array(voids)


# =============================================================================
# VISUALIZER
# =============================================================================

PHASE_CMAP = LinearSegmentedColormap.from_list('phase', ['white', 'red', 'blue'], N=3)
PHASE_LEGEND = [
    Patch(facecolor='white', edgecolor='gray', label='Void'),
    Patch(facecolor='red', label='Functional'),
    Patch(facecolor='blue', label='Inert'),
]

def _plot_phase(ax, image, extent, xlabel, ylabel, title=None):
    ax.imshow(image.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
              extent=extent, aspect='equal', interpolation='nearest')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title)

def _ext(domain, axis):
    if axis==2: return [0,domain[0],0,domain[1]],'X (μm)','Y (μm)'
    if axis==1: return [0,domain[0],0,domain[2]],'X (μm)','Z (μm)'
    return [0,domain[1],0,domain[2]],'Y (μm)','Z (μm)'


class Visualizer:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.viz_dir = self.output_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        self.config_name = self.output_dir.name

        self.config = {}
        for p in [f"{self.config_name}_config.json", "config.json"]:
            fp = self.output_dir / p
            if fp.exists():
                with open(fp) as f: self.config = json.load(f)
                print(f"Config: {fp.name}"); break

        self.history = None
        for p in [f"{self.config_name}_history.json", "history.json"]:
            fp = self.output_dir / p
            if fp.exists():
                try:
                    with open(fp) as f: self.history = json.load(f)
                    print(f"History: {fp.name}")
                except: pass
                break

        self.frame_files = sorted(self.output_dir.glob(f"{self.config_name}_frame_*.json"))
        if not self.frame_files:
            self.frame_files = sorted(self.output_dir.glob("frame_*.json"))
        self.frame_files = list(self.frame_files)

        self.frames = []
        for fp in self.frame_files:
            fr = load_frame(str(fp))
            if fr is not None: self.frames.append(fr)
        self.frames.sort(key=lambda f: f.time_hours)
        print(f"Loaded {len(self.frames)} frames")
        if self.frames:
            print(f"Time: {self.frames[0].time_days:.2f} – {self.frames[-1].time_days:.2f} d")
            n_with_cells = sum(1 for f in self.frames if f.cell_data is not None)
            print(f"Frames with cell data: {n_with_cells}/{len(self.frames)}")

        self.domain = None
        if 'domain' in self.config:
            s = self.config['domain'].get('side_length_um')
            if s: self.domain = np.array([s, s, s])
        if self.domain is None and self.frames and self.frames[0].domain is not None:
            self.domain = np.array(self.frames[0].domain)
        if self.domain is None:
            self.domain = np.array([700.0]*3)
        print(f"Domain: {self.domain[0]:.0f}³ μm")

        if self.frames:
            self.margin = np.mean([s.equivalent_radius for s in self.frames[0].shapes])
        else:
            self.margin = 40.0
        print(f"Border margin: {self.margin:.1f} μm")

    # =========================================================================
    # GLOBAL RANGES — compute consistent color scales across all frames
    # =========================================================================
    def _global_cell_ranges(self):
        """Compute global max stress and max aspect ratio across all frames."""
        max_stress = 0.0
        max_ar = 1.0
        for fr in self.frames:
            if fr.cell_data is None: continue
            cd = fr.cell_data
            if cd.total_cells == 0: continue
            ms = np.max(cd.stress_nN)
            ma = np.max(cd.aspect_ratio)
            if ms > max_stress: max_stress = ms
            if ma > max_ar: max_ar = ma
        return max(max_stress, 1e-6), max(max_ar, 1.01)

    # =========================================================================
    # PHASE MAP (original)
    # =========================================================================
    def plot_phase_map(self, frame_index=-1, axis=2, resolution=300, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        sp = self.domain[axis] / 2
        img = render_phase_map(frame, self.domain, axis, sp, resolution)
        ext, xl, yl = _ext(self.domain, axis)
        fig, ax = plt.subplots(figsize=(8, 8))
        _plot_phase(ax, img, ext, xl, yl,
                    f'Phase Map — {"XYZ"[axis]}={sp:.0f}μm  t={frame.time_days:.2f}d')
        ax.legend(handles=PHASE_LEGEND, loc='upper right')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"phase_{'xyz'[axis]}_{lab}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    def plot_phase_evolution(self, axis=2, n_times=6, resolution=200, save=True):
        if not self.frames: return None
        idxs = np.linspace(0, len(self.frames)-1, n_times, dtype=int)
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        fig, axes = plt.subplots(1, n_times, figsize=(3.5*n_times, 4))
        if n_times == 1: axes = [axes]
        for col, fi in enumerate(idxs):
            fr = self.frames[fi]
            img = render_phase_map(fr, self.domain, axis, sp, resolution)
            _plot_phase(axes[col], img, ext, xl,
                        yl if col==0 else '', f't={fr.time_days:.1f}d')
            if col > 0: axes[col].set_yticklabels([])
        fig.legend(handles=PHASE_LEGEND, loc='upper right', bbox_to_anchor=(0.99,0.99))
        plt.suptitle(f'Phase Evolution ({"XYZ"[axis]}={sp:.0f}μm)')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f"phase_evo_{'xyz'[axis]}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # CELL POSITION MAP (scatter on phase background)
    # =========================================================================
    def plot_cell_map(self, frame_index=-1, axis=2, resolution=250, save=True):
        """Phase map with individual cells overlaid as dots, colored by type."""
        if not self.frames: return None
        frame = self.frames[frame_index]
        if frame.cell_data is None:
            print("No cell data in frame"); return None
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: cells colored by type on phase background
        img = render_phase_map(frame, self.domain, axis, sp, resolution)
        _plot_phase(axes[0], img, ext, xl, yl, 'Cells by Type')
        thickness = frame.cell_data.cell_diameter * 2.0
        render_cell_scatter(frame, self.domain, axis, sp, thickness,
                            ax=axes[0], color_by='type', s=8)
        render_bridge_lines(frame.cell_data, axis, sp, thickness, axes[0], self.domain)

        # Panel 2: cells colored by bridging state
        _plot_phase(axes[1], img, ext, xl, '', 'Bridging State')
        render_cell_scatter(frame, self.domain, axis, sp, thickness,
                            ax=axes[1], color_by='bridging', s=10)
        render_bridge_lines(frame.cell_data, axis, sp, thickness, axes[1], self.domain)
        axes[1].legend(handles=[
            Patch(facecolor='orange', label='Bridging'),
            Patch(facecolor='gray', label='Free'),
        ], loc='upper right', fontsize=8)

        # Panel 3: cells colored by stress
        _plot_phase(axes[2], img * 0.3, ext, xl, '', 'Cell Stress (nN)')
        gmax_stress = max(np.max(frame.cell_data.stress_nN), 1e-6)
        sc, _ = render_cell_scatter(frame, self.domain, axis, sp, thickness,
                                    ax=axes[2], color_by='stress', s=14,
                                    vmin=0, vmax=gmax_stress, cmap='hot')
        if sc is not None:
            plt.colorbar(sc, ax=axes[2], label='Stress (nN)', shrink=0.8)

        n_br = int(np.sum(frame.cell_data.is_bridging))
        plt.suptitle(f'Cell Map — t={frame.time_days:.2f}d — '
                     f'{frame.cell_data.total_cells} cells, {n_br} bridging',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"cell_map_{'xyz'[axis]}_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # STRESS HEATMAP
    # =========================================================================
    def plot_stress_map(self, frame_index=-1, axis=2, resolution=120, save=True):
        """Smoothed stress heatmap on phase background."""
        if not self.frames: return None
        frame = self.frames[frame_index]
        if frame.cell_data is None: print("No cell data"); return None

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        thickness = frame.cell_data.cell_diameter * 2.5

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Phase background
        img = render_phase_map(frame, self.domain, axis, sp, resolution)
        _plot_phase(axes[0], img, ext, xl, yl, 'Phase Map')
        axes[0].legend(handles=PHASE_LEGEND, loc='upper right', fontsize=8)

        # Stress field
        field, fext = render_cell_field(frame.cell_data, self.domain, axis, sp,
                                        thickness, field='stress', resolution=resolution)
        gmax = max(np.nanmax(field) if not np.all(np.isnan(field)) else 0, 1e-6)
        # Show phase as faint background
        axes[1].imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                       extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
        im = axes[1].imshow(field.T, origin='lower', cmap='hot', vmin=0, vmax=gmax,
                            extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
        axes[1].set_xlabel(xl); axes[1].set_ylabel('')
        axes[1].set_title(f'Stress Field (nN)')
        plt.colorbar(im, ax=axes[1], label='Stress (nN)', shrink=0.8)

        plt.suptitle(f'Stress Map — t={frame.time_days:.2f}d', fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"stress_map_{'xyz'[axis]}_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # ASPECT RATIO HEATMAP
    # =========================================================================
    def plot_aspect_ratio_map(self, frame_index=-1, axis=2, resolution=120, save=True):
        """Smoothed aspect ratio heatmap on phase background."""
        if not self.frames: return None
        frame = self.frames[frame_index]
        if frame.cell_data is None: print("No cell data"); return None

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        thickness = frame.cell_data.cell_diameter * 2.5

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        img = render_phase_map(frame, self.domain, axis, sp, resolution)
        _plot_phase(axes[0], img, ext, xl, yl, 'Phase Map')
        axes[0].legend(handles=PHASE_LEGEND, loc='upper right', fontsize=8)

        field, fext = render_cell_field(frame.cell_data, self.domain, axis, sp,
                                        thickness, field='aspect_ratio', resolution=resolution)
        gmax = max(np.nanmax(field) if not np.all(np.isnan(field)) else 1, 1.01)
        axes[1].imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                       extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
        im = axes[1].imshow(field.T, origin='lower', cmap='viridis', vmin=1.0, vmax=gmax,
                            extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
        axes[1].set_xlabel(xl); axes[1].set_ylabel('')
        axes[1].set_title('Cell Aspect Ratio (L/w)')
        plt.colorbar(im, ax=axes[1], label='Aspect Ratio', shrink=0.8)

        plt.suptitle(f'Aspect Ratio Map — t={frame.time_days:.2f}d', fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"ar_map_{'xyz'[axis]}_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # EVOLUTION PANELS (cell maps, stress, aspect ratio)
    # =========================================================================
    def _evolution_panel(self, field_type, axis=2, n_times=6, resolution=100, save=True):
        """
        Generic evolution panel for cell-level fields.
        field_type: 'cells' | 'stress' | 'aspect_ratio'
        """
        frames_with_cells = [f for f in self.frames if f.cell_data is not None]
        if not frames_with_cells:
            print(f"No frames with cell data for {field_type} evolution"); return None

        idxs = np.linspace(0, len(frames_with_cells)-1, min(n_times, len(frames_with_cells)), dtype=int)
        n_cols = len(idxs)
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)

        # Global ranges for consistent coloring
        gmax_stress, gmax_ar = self._global_cell_ranges()

        if field_type == 'cells':
            fig, axes = plt.subplots(1, n_cols, figsize=(3.5*n_cols, 4))
            title = 'Cell Position Evolution'
            cmap_name = None
        elif field_type == 'stress':
            fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4.5))
            title = 'Stress Field Evolution'
            cmap_name = 'hot'
        else:
            fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 4.5))
            title = 'Aspect Ratio Evolution'
            cmap_name = 'viridis'

        if n_cols == 1: axes = [axes]

        for col, fi in enumerate(idxs):
            ax = axes[col]
            fr = frames_with_cells[fi]
            cd = fr.cell_data
            thickness = cd.cell_diameter * 2.5

            img = render_phase_map(fr, self.domain, axis, sp, resolution)

            if field_type == 'cells':
                _plot_phase(ax, img, ext, xl, yl if col==0 else '',
                            f't={fr.time_days:.1f}d')
                render_cell_scatter(fr, self.domain, axis, sp, thickness,
                                    ax=ax, color_by='type', s=6)
                render_bridge_lines(cd, axis, sp, thickness, ax, self.domain)
            elif field_type == 'stress':
                ax.imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                          extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
                fld, fext = render_cell_field(cd, self.domain, axis, sp, thickness,
                                              field='stress', resolution=resolution)
                im = ax.imshow(fld.T, origin='lower', cmap=cmap_name, vmin=0, vmax=gmax_stress,
                               extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
                ax.set_xlabel(xl); ax.set_ylabel(yl if col==0 else '')
                ax.set_title(f't={fr.time_days:.1f}d')
            else:  # aspect_ratio
                ax.imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                          extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
                fld, fext = render_cell_field(cd, self.domain, axis, sp, thickness,
                                              field='aspect_ratio', resolution=resolution)
                im = ax.imshow(fld.T, origin='lower', cmap=cmap_name, vmin=1.0, vmax=gmax_ar,
                               extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
                ax.set_xlabel(xl); ax.set_ylabel(yl if col==0 else '')
                ax.set_title(f't={fr.time_days:.1f}d')

            if col > 0: ax.set_yticklabels([])

        # Colorbar for field plots
        if field_type in ('stress', 'aspect_ratio') and n_cols > 0:
            label = 'Stress (nN)' if field_type == 'stress' else 'Aspect Ratio (L/w)'
            fig.colorbar(im, ax=axes, label=label, shrink=0.6, pad=0.02)

        plt.suptitle(f'{title} ({"XYZ"[axis]}={sp:.0f}μm)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f"{field_type}_evo_{'xyz'[axis]}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    def plot_cell_evolution(self, axis=2, n_times=6, resolution=100, save=True):
        return self._evolution_panel('cells', axis, n_times, resolution, save)

    def plot_stress_evolution(self, axis=2, n_times=6, resolution=100, save=True):
        return self._evolution_panel('stress', axis, n_times, resolution, save)

    def plot_aspect_ratio_evolution(self, axis=2, n_times=6, resolution=100, save=True):
        return self._evolution_panel('aspect_ratio', axis, n_times, resolution, save)

    # =========================================================================
    # CELL METRICS HISTORY (time series from history dict)
    # =========================================================================
    def plot_cell_history(self, save=True):
        """Plot cell-level metrics over time from simulation history."""
        if not self.history: print("No history"); return None
        h = self.history
        needed = ['mean_stress_nN', 'max_stress_nN', 'mean_ar', 'max_ar', 'n_bridging_cells']
        if not all(k in h for k in needed):
            print("History missing cell metrics"); return None

        t = np.array(h['time_hours']) / 24

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Stress over time
        ax = axes[0, 0]
        ax.plot(t, h['mean_stress_nN'], 'r-', lw=2, label='Mean')
        ax.plot(t, h['max_stress_nN'], 'r--', lw=1.5, alpha=0.7, label='Max')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Stress (nN)')
        ax.set_title('Cell Stress'); ax.legend(); ax.grid(True, alpha=0.3)

        # Aspect ratio over time
        ax = axes[0, 1]
        ax.plot(t, h['mean_ar'], 'b-', lw=2, label='Mean')
        ax.plot(t, h['max_ar'], 'b--', lw=1.5, alpha=0.7, label='Max')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Aspect Ratio (L/w)')
        ax.set_title('Cell Aspect Ratio'); ax.legend(); ax.grid(True, alpha=0.3)

        # Bridging cells over time
        ax = axes[1, 0]
        ax.plot(t, h['n_bridging_cells'], 'orange', lw=2)
        total = h.get('total_cells', 1)
        ax.axhline(total, color='gray', ls=':', alpha=0.5, label=f'Total: {total}')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('# Bridging Cells')
        ax.set_title('Bridging Cells'); ax.legend(); ax.grid(True, alpha=0.3)

        # Bridges + contacts
        ax = axes[1, 1]
        ax.plot(t, h['n_contacts'], 'b-', lw=2, label='Contacts')
        ax.plot(t, h['n_bridges'], 'r-', lw=2, label='Bridges')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Count')
        ax.set_title('Contacts & Bridges'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('Cell Mechanics History', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / "cell_history.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # ANIMATIONS (GIF) — phase, cells, stress, aspect ratio
    # =========================================================================
    def _create_field_animation(self, field_type, axis=2, resolution=150, fps=6, save=True):
        """
        Generic GIF animation for: 'phase', 'cells', 'stress', 'aspect_ratio'
        """
        if not HAS_ANIMATION: print("No animation support"); return None

        if field_type in ('cells', 'stress', 'aspect_ratio'):
            anim_frames = [f for f in self.frames if f.cell_data is not None]
        else:
            anim_frames = self.frames

        if not anim_frames: print(f"No frames for {field_type} animation"); return None
        print(f"Creating {field_type} animation ({len(anim_frames)} frames)...")

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        gmax_stress, gmax_ar = self._global_cell_ranges()

        if field_type in ('stress', 'aspect_ratio'):
            fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(14, 6),
                                                  gridspec_kw={'width_ratios': [1.3, 1]})
        else:
            fig, (ax_img, ax_ts) = plt.subplots(1, 2, figsize=(14, 6))

        # Initialize first frame
        fr0 = anim_frames[0]

        if field_type == 'phase':
            img0 = render_phase_map(fr0, self.domain, axis, sp, resolution)
            im = ax_img.imshow(img0.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
                               extent=ext, aspect='equal', interpolation='nearest')
            ax_img.legend(handles=PHASE_LEGEND, loc='upper right', fontsize=8)
        elif field_type == 'cells':
            img0 = render_phase_map(fr0, self.domain, axis, sp, resolution)
            im = ax_img.imshow(img0.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
                               extent=ext, aspect='equal', interpolation='nearest')
        elif field_type == 'stress':
            img0 = render_phase_map(fr0, self.domain, axis, sp, resolution)
            ax_img.imshow(img0.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                          extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
            thickness = fr0.cell_data.cell_diameter * 2.5
            fld, fext = render_cell_field(fr0.cell_data, self.domain, axis, sp, thickness,
                                          field='stress', resolution=resolution)
            im = ax_img.imshow(fld.T, origin='lower', cmap='hot', vmin=0, vmax=gmax_stress,
                               extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
            plt.colorbar(im, ax=ax_img, label='Stress (nN)', shrink=0.7)
        else:  # aspect_ratio
            img0 = render_phase_map(fr0, self.domain, axis, sp, resolution)
            ax_img.imshow(img0.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                          extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
            thickness = fr0.cell_data.cell_diameter * 2.5
            fld, fext = render_cell_field(fr0.cell_data, self.domain, axis, sp, thickness,
                                          field='aspect_ratio', resolution=resolution)
            im = ax_img.imshow(fld.T, origin='lower', cmap='viridis', vmin=1.0, vmax=gmax_ar,
                               extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
            plt.colorbar(im, ax=ax_img, label='Aspect Ratio', shrink=0.7)

        ax_img.set_xlabel(xl); ax_img.set_ylabel(yl)

        # Time series panel
        if self.history and len(self.history.get('time_hours', [])) > 1:
            th = np.array(self.history['time_hours']) / 24
            if field_type == 'stress' and 'mean_stress_nN' in self.history:
                ax_ts.plot(th, self.history['mean_stress_nN'], 'r-', lw=2, label='Mean Stress')
                ax_ts.set_ylabel('Stress (nN)')
            elif field_type == 'aspect_ratio' and 'mean_ar' in self.history:
                ax_ts.plot(th, self.history['mean_ar'], 'b-', lw=2, label='Mean AR')
                ax_ts.set_ylabel('Aspect Ratio')
            elif field_type == 'cells' and 'n_bridging_cells' in self.history:
                ax_ts.plot(th, self.history['n_bridging_cells'], 'orange', lw=2, label='Bridging')
                ax_ts.set_ylabel('# Bridging Cells')
            else:
                ax_ts.plot(th, self.history.get('mean_coordination', []), 'g-', lw=2)
                ax_ts.set_ylabel('Coordination')
            ax_ts.set_xlabel('Time (days)'); ax_ts.grid(True, alpha=0.3); ax_ts.legend()

        tl = ax_ts.axvline(x=0, color='black', lw=2, ls='--')
        title = fig.suptitle('', fontsize=12, fontweight='bold')

        # Store scatter artists for cell animation
        scatter_artists = []

        def update(fi):
            nonlocal scatter_artists
            fr = anim_frames[fi]

            if field_type == 'phase':
                im.set_data(render_phase_map(fr, self.domain, axis, sp, resolution).T)
            elif field_type == 'cells':
                im.set_data(render_phase_map(fr, self.domain, axis, sp, resolution).T)
                # Remove old scatter
                for art in scatter_artists:
                    art.remove()
                scatter_artists.clear()
                # Remove old lines (bridge lines)
                while len(ax_img.lines) > 0:
                    ax_img.lines[-1].remove()
                if fr.cell_data is not None:
                    thickness = fr.cell_data.cell_diameter * 2.0
                    sc, _ = render_cell_scatter(fr, self.domain, axis, sp, thickness,
                                                ax=ax_img, color_by='type', s=6)
                    if sc is not None:
                        scatter_artists.append(sc)
                    render_bridge_lines(fr.cell_data, axis, sp, thickness, ax_img, self.domain)
            elif field_type == 'stress':
                if fr.cell_data is not None:
                    thickness = fr.cell_data.cell_diameter * 2.5
                    fld, _ = render_cell_field(fr.cell_data, self.domain, axis, sp, thickness,
                                               field='stress', resolution=resolution)
                    im.set_data(fld.T)
            elif field_type == 'aspect_ratio':
                if fr.cell_data is not None:
                    thickness = fr.cell_data.cell_diameter * 2.5
                    fld, _ = render_cell_field(fr.cell_data, self.domain, axis, sp, thickness,
                                               field='aspect_ratio', resolution=resolution)
                    im.set_data(fld.T)

            tl.set_xdata([fr.time_days, fr.time_days])
            title.set_text(f'{field_type.replace("_"," ").title()} — t={fr.time_hours:.1f}h ({fr.time_days:.2f}d)')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(anim_frames),
                             interval=1000//fps, blit=False)  # blit=False for scatter updates
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f"anim_{field_type}_{'xyz'[axis]}.gif"
            try:
                anim.save(str(fp), writer=PillowWriter(fps=fps), dpi=120)
                print(f"Saved: {fp}")
            except Exception as e:
                print(f"Animation error: {e}"); fp = None
            plt.close(fig)
            return str(fp) if fp else None
        plt.show(); return None

    def create_animation(self, axis=2, resolution=200, fps=6, save=True):
        return self._create_field_animation('phase', axis, resolution, fps, save)

    def create_cell_animation(self, axis=2, resolution=150, fps=6, save=True):
        return self._create_field_animation('cells', axis, resolution, fps, save)

    def create_stress_animation(self, axis=2, resolution=120, fps=6, save=True):
        return self._create_field_animation('stress', axis, resolution, fps, save)

    def create_ar_animation(self, axis=2, resolution=120, fps=6, save=True):
        return self._create_field_animation('aspect_ratio', axis, resolution, fps, save)

    # =========================================================================
    # COMBINED CELL DASHBOARD (single-frame overview)
    # =========================================================================
    def plot_cell_dashboard(self, frame_index=-1, axis=2, resolution=120, save=True):
        """
        6-panel dashboard: phase + cells, bridging, stress scatter, stress field,
        AR scatter, AR field.
        """
        if not self.frames: return None
        frame = self.frames[frame_index]
        if frame.cell_data is None: print("No cell data"); return None

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        cd = frame.cell_data
        thickness = cd.cell_diameter * 2.5

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Row 1: Phase+Cells, Bridging, Stress scatter
        img = render_phase_map(frame, self.domain, axis, sp, resolution)

        _plot_phase(axes[0,0], img, ext, xl, yl, 'Cells (type)')
        render_cell_scatter(frame, self.domain, axis, sp, thickness,
                            ax=axes[0,0], color_by='type', s=8)
        render_bridge_lines(cd, axis, sp, thickness, axes[0,0], self.domain)

        _plot_phase(axes[0,1], img, ext, xl, '', 'Bridging')
        render_cell_scatter(frame, self.domain, axis, sp, thickness,
                            ax=axes[0,1], color_by='bridging', s=10)
        render_bridge_lines(cd, axis, sp, thickness, axes[0,1], self.domain)

        _plot_phase(axes[0,2], img * 0.2, ext, xl, '', 'Stress (scatter)')
        gmax_s = max(np.max(cd.stress_nN), 1e-6)
        sc, _ = render_cell_scatter(frame, self.domain, axis, sp, thickness,
                                    ax=axes[0,2], color_by='stress', s=14,
                                    vmin=0, vmax=gmax_s, cmap='hot')
        if sc: plt.colorbar(sc, ax=axes[0,2], label='nN', shrink=0.7)

        # Row 2: Stress field, AR scatter, AR field
        axes[1,0].imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                         extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
        fld_s, fext = render_cell_field(cd, self.domain, axis, sp, thickness,
                                        field='stress', resolution=resolution)
        im_s = axes[1,0].imshow(fld_s.T, origin='lower', cmap='hot', vmin=0, vmax=gmax_s,
                                extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
        axes[1,0].set_xlabel(xl); axes[1,0].set_ylabel(yl); axes[1,0].set_title('Stress Field (nN)')
        plt.colorbar(im_s, ax=axes[1,0], label='nN', shrink=0.7)

        _plot_phase(axes[1,1], img * 0.2, ext, xl, '', 'Aspect Ratio (scatter)')
        gmax_a = max(np.max(cd.aspect_ratio), 1.01)
        sc2, _ = render_cell_scatter(frame, self.domain, axis, sp, thickness,
                                     ax=axes[1,1], color_by='aspect_ratio', s=14,
                                     vmin=1.0, vmax=gmax_a, cmap='viridis')
        if sc2: plt.colorbar(sc2, ax=axes[1,1], label='L/w', shrink=0.7)

        axes[1,2].imshow(img.T * 0.15, origin='lower', cmap='gray', vmin=0, vmax=2,
                         extent=ext, aspect='equal', interpolation='nearest', alpha=0.3)
        fld_a, fext = render_cell_field(cd, self.domain, axis, sp, thickness,
                                        field='aspect_ratio', resolution=resolution)
        im_a = axes[1,2].imshow(fld_a.T, origin='lower', cmap='viridis', vmin=1.0, vmax=gmax_a,
                                extent=fext, aspect='equal', interpolation='bilinear', alpha=0.85)
        axes[1,2].set_xlabel(xl); axes[1,2].set_ylabel(''); axes[1,2].set_title('AR Field (L/w)')
        plt.colorbar(im_a, ax=axes[1,2], label='L/w', shrink=0.7)

        n_br = int(np.sum(cd.is_bridging))
        plt.suptitle(f'Cell Dashboard — t={frame.time_days:.2f}d — '
                     f'{cd.total_cells} cells, {n_br} bridging',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"cell_dashboard_{'xyz'[axis]}_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # =========================================================================
    # THREE-PLANE, DETAILED, PACKING, HISTORY (preserved from original)
    # =========================================================================
    def plot_three_plane(self, frame_index=-1, resolution=250, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, ax_id in enumerate([2, 1, 0]):
            sp = self.domain[ax_id] / 2
            img = render_phase_map(frame, self.domain, ax_id, sp, resolution)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(axes[col], img, ext, xl, yl,
                        f'{"XY XZ YZ".split()[col]} ({"XYZ"[ax_id]}={sp:.0f}μm)')
        fig.legend(handles=PHASE_LEGEND, loc='upper right', bbox_to_anchor=(0.99,0.99))
        plt.suptitle(f't={frame.time_hours:.1f}h ({frame.time_days:.2f}d)')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"three_plane_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    def plot_packing_evolution(self, save=True):
        if not self.frames: return None
        print("Computing voxel packing fractions (border-excluded)...")
        times, phis = [], []
        for fi, frame in enumerate(self.frames):
            phi = compute_voxel_packing_fast(frame, self.domain, self.margin, res=40)
            times.append(frame.time_days); phis.append(phi)
            if fi % max(1, len(self.frames)//5) == 0:
                print(f"  Frame {fi}/{len(self.frames)}: φ = {phi:.3f}")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0,0]
        ax.plot(times, phis, 'bo-', ms=4, lw=2)
        ax.axhline(0.64, color='r', ls='--', alpha=0.5, label='RCP sphere')
        ax.axhline(0.55, color='orange', ls='--', alpha=0.5, label='RLP sphere')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction')
        ax.set_title(f'Packing Fraction (margin={self.margin:.0f}μm)'); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[0,1]
        ax.plot(times, [1-p for p in phis], 'ko-', ms=4, lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Void Fraction'); ax.set_title('Void Fraction'); ax.grid(True, alpha=0.3)
        ax = axes[1,0]
        if self.history and 'packing_fraction' in self.history:
            th = np.array(self.history['time_hours'])/24
            ax.plot(th, self.history['packing_fraction'], 'b-', lw=2, label='Sim history')
            ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing φ'); ax.set_title('Packing (sim history)'); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[1,1]
        frame = self.frames[-1]
        for a, lab, c in [(0,'X','red'),(1,'Y','green'),(2,'Z','blue')]:
            p, vf = compute_void_profile(frame, self.domain, self.margin, axis=a, n_slices=25, slice_res=60)
            if len(p)>0: ax.plot(p, vf, '-', color=c, lw=2, label=lab)
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Local Void Fraction')
        ax.set_title('Void Profiles (final)'); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / "packing_evolution.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved: {fp}")
        return fig

    def plot_detailed_frame(self, frame_index=-1, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(16, 12)); gs = gridspec.GridSpec(2, 3)
        for col, ax_id in enumerate([2,1,0]):
            ax = fig.add_subplot(gs[0, col])
            sp = self.domain[ax_id]/2
            img = render_phase_map(frame, self.domain, ax_id, sp, 250)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(ax, img, ext, xl, yl, f'{"XY XZ YZ".split()[col]}')
        ax = fig.add_subplot(gs[1,0])
        for a, lab, c in [(0,'X','red'),(1,'Y','green'),(2,'Z','blue')]:
            p, vf = compute_void_profile(frame, self.domain, self.margin, axis=a, n_slices=20, slice_res=50)
            if len(p)>0: ax.plot(p, vf, '-', color=c, lw=2, label=lab)
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Void Fraction'); ax.set_title('Void Profiles'); ax.legend(); ax.grid(True, alpha=0.3)
        ax = fig.add_subplot(gs[1,1])
        fr = [s.equivalent_radius for i,s in enumerate(frame.shapes) if frame.functional_mask[i]]
        ir = [s.equivalent_radius for i,s in enumerate(frame.shapes) if frame.inert_mask[i]]
        ax.hist(fr, bins=15, alpha=0.6, color='red', label='Functional')
        ax.hist(ir, bins=15, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equiv. Radius (μm)'); ax.set_ylabel('Count'); ax.set_title('Size Distribution'); ax.legend()
        ax = fig.add_subplot(gs[1,2]); ax.axis('off')
        phi = compute_voxel_packing_fast(frame, self.domain, self.margin, res=35)
        stats = (f"t = {frame.time_hours:.1f}h ({frame.time_days:.2f}d)\n\n"
                 f"Func: {int(np.sum(frame.functional_mask))}  Inert: {int(np.sum(frame.inert_mask))}  Total: {frame.n_granules}\n\n"
                 f"Mean R func: {np.mean(fr):.1f} μm\nMean R inert: {np.mean(ir):.1f} μm\n\n"
                 f"Voxel packing φ: {phi:.3f}\nVoxel void frac: {1-phi:.3f}\n\n"
                 f"Border margin: {self.margin:.1f} μm\nTotal cells: {int(np.sum(frame.n_cells))}\nDomain: {self.domain[0]:.0f}³ μm")
        ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=11, va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.suptitle(f'Detailed — t={frame.time_days:.2f}d', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"detailed_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved: {fp}")
        return fig

    def plot_history(self, save=True):
        if not self.history: print("No history"); return None
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        t = np.array(self.history['time_hours'])/24
        ax = axes[0,0]
        ax.plot(t, self.history['n_contacts'], 'b-', lw=2, label='Contacts')
        ax.plot(t, self.history['n_bridges'], 'r-', lw=2, label='Bridges')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Count'); ax.set_title('Interactions'); ax.legend(); ax.grid(True, alpha=0.3)
        ax = axes[0,1]
        ax.plot(t, self.history['mean_coordination'], 'g-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean Coordination'); ax.set_title('Contact Network'); ax.grid(True, alpha=0.3)
        ax = axes[1,0]
        ax.semilogy(t, self.history['max_velocity'], 'm-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Max Velocity (μm/hr)'); ax.set_title('Activity Level'); ax.grid(True, alpha=0.3)
        ax = axes[1,1]
        ax.plot(t, self.history['packing_fraction'], 'b-', lw=2, label='Voxel φ')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction'); ax.set_title('Packing'); ax.grid(True, alpha=0.3); ax.legend()
        plt.tight_layout()
        if save:
            fp = self.viz_dir / "history.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig); print(f"Saved: {fp}")
        return fig

    def plot_snapshot(self, save=True):
        if not self.frames: return None
        frame = self.frames[-1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for idx, (r, c, ax_id) in enumerate([(0,0,2), (0,1,1)]):
            sp = self.domain[ax_id]/2
            img = render_phase_map(frame, self.domain, ax_id, sp, 200)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(axes[r,c], img, ext, xl, yl, f'{"XY XZ".split()[idx]} (t={frame.time_hours:.1f}h)')
        ax = axes[1,0]
        if self.history and len(self.history.get('time_hours',[]))>1:
            th = np.array(self.history['time_hours'])/24
            ax.plot(th, self.history['mean_coordination'], 'g-', lw=2)
            ax.set_ylabel('Coordination', color='g'); ax.tick_params(axis='y', labelcolor='g'); ax.grid(True, alpha=0.3)
            ax2 = ax.twinx(); ax2.plot(th, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
            ax2.set_ylabel('Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')
        ax.set_xlabel('Time (days)'); ax.set_title('Network')
        ax = axes[1,1]
        if self.history and len(self.history.get('time_hours',[]))>1:
            th = np.array(self.history['time_hours'])/24
            ax.plot(th, self.history['packing_fraction'], 'b-', lw=2)
            ax.set_ylabel('Voxel φ', color='b'); ax.tick_params(axis='y', labelcolor='b'); ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time (days)'); ax.set_title('Packing')
        plt.suptitle(f'{self.config_name} — t={frame.time_hours:.1f}h', fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save:
            plt.savefig(self.viz_dir/"live_snapshot.png", dpi=150, bbox_inches='tight'); plt.close(fig)
        return None

    def reload_frames(self):
        for p in [f"{self.config_name}_history.json", "history.json"]:
            fp = self.output_dir / p
            if fp.exists():
                try:
                    with open(fp) as f: self.history = json.load(f)
                except: pass
                break
        new_files = sorted(self.output_dir.glob(f"{self.config_name}_frame_*.json"))
        if not new_files: new_files = sorted(self.output_dir.glob("frame_*.json"))
        known = {str(fp) for fp in self.frame_files}; added = 0
        for fp in new_files:
            if str(fp) not in known:
                fr = load_frame(str(fp))
                if fr is not None: self.frames.append(fr); self.frame_files.append(fp); added += 1
        if added > 0:
            self.frames.sort(key=lambda f: f.time_hours)
            print(f"  [watch] +{added} → {len(self.frames)} frames (t={self.frames[-1].time_hours:.1f}h)")
        return added

    def watch(self, interval=30.0, full_every=5):
        print(f"\n{'='*60}\nWATCH MODE — every {interval:.0f}s\n{'='*60}\n")
        matplotlib.use('Agg'); cycle = 0
        try:
            while True:
                n_new = self.reload_frames()
                if n_new > 0 or cycle == 0:
                    self.plot_snapshot(save=True)
                    if cycle % full_every == 0 and len(self.frames) >= 2:
                        print("  [watch] Full plots...")
                        for fn in [self.plot_history, self.plot_cell_history,
                                   lambda: self.plot_phase_map(axis=2),
                                   lambda: self.plot_phase_evolution(axis=2),
                                   lambda: self.plot_cell_dashboard(axis=2)]:
                            try: fn()
                            except Exception as e: print(f"    skip: {e}")
                    cycle += 1
                if list(self.output_dir.glob("*_frame_final.json")) and len(self.frames) > 2:
                    self.reload_frames()
                    print("\n  [watch] Final frame — generating all...")
                    self.create_all(); break
                time_module.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n  [watch] Interrupted with {len(self.frames)} frames")
            if len(self.frames) >= 2: self.create_all()

    # =========================================================================
    # ALL
    # =========================================================================
    def create_all(self):
        print(f"\n{'='*60}\nGenerating All Visualizations\n{'='*60}")
        print("\n1. History..."); self.plot_history()
        print("\n2. Cell history..."); self.plot_cell_history()
        print("\n3. Packing evolution..."); self.plot_packing_evolution()
        print("\n4. Phase evolution...")
        for a in [2,1]: self.plot_phase_evolution(axis=a)
        print("\n5. Cell evolution...")
        for a in [2,1]: self.plot_cell_evolution(axis=a)
        print("\n6. Stress evolution...")
        for a in [2,1]: self.plot_stress_evolution(axis=a)
        print("\n7. Aspect ratio evolution...")
        for a in [2,1]: self.plot_aspect_ratio_evolution(axis=a)
        print("\n8. Three-plane...")
        if self.frames: self.plot_three_plane(0); self.plot_three_plane(-1)
        print("\n9. Detailed...")
        if self.frames: self.plot_detailed_frame(0); self.plot_detailed_frame(-1)
        print("\n10. Cell maps (final)...")
        if self.frames:
            for a in [2,1,0]: self.plot_cell_map(-1, axis=a)
        print("\n11. Stress maps (final)...")
        if self.frames:
            for a in [2,1,0]: self.plot_stress_map(-1, axis=a)
        print("\n12. Aspect ratio maps (final)...")
        if self.frames:
            for a in [2,1,0]: self.plot_aspect_ratio_map(-1, axis=a)
        print("\n13. Cell dashboard...")
        if self.frames:
            self.plot_cell_dashboard(0); self.plot_cell_dashboard(-1)
        print("\n14. Phase maps (final)...")
        if self.frames:
            for a in [2,1,0]: self.plot_phase_map(-1, axis=a)
        print("\n15. Animations (GIF)...")
        if self.frames:
            self.create_animation(axis=2, fps=6)
            self.create_cell_animation(axis=2, fps=6)
            self.create_stress_animation(axis=2, fps=5)
            self.create_ar_animation(axis=2, fps=5)
        print(f"\nDone → {self.viz_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='./simulations/Trial4_lowstiff')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--phase-map', '-p', action='store_true')
    parser.add_argument('--cell-map', action='store_true')
    parser.add_argument('--stress-map', action='store_true')
    parser.add_argument('--ar-map', action='store_true')
    parser.add_argument('--cell-dashboard', action='store_true')
    parser.add_argument('--cell-anim', action='store_true')
    parser.add_argument('--stress-anim', action='store_true')
    parser.add_argument('--ar-anim', action='store_true')
    parser.add_argument('--watch', '-w', action='store_true')
    parser.add_argument('--watch-interval', type=float, default=30.0)
    parser.add_argument('--animation', '-s', action='store_true')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--axis', type=int, default=2)
    args = parser.parse_args()

    viz = Visualizer(args.input)
    if args.watch: viz.watch(interval=args.watch_interval)
    elif args.cell_map: viz.plot_cell_map(axis=args.axis)
    elif args.stress_map: viz.plot_stress_map(axis=args.axis)
    elif args.ar_map: viz.plot_aspect_ratio_map(axis=args.axis)
    elif args.cell_dashboard: viz.plot_cell_dashboard(axis=args.axis)
    elif args.cell_anim: viz.create_cell_animation(axis=args.axis, fps=args.fps)
    elif args.stress_anim: viz.create_stress_animation(axis=args.axis, fps=args.fps)
    elif args.ar_anim: viz.create_ar_animation(axis=args.axis, fps=args.fps)
    elif args.phase_map: viz.plot_phase_map(axis=args.axis)
    elif args.animation: viz.create_animation(axis=args.axis, fps=args.fps)
    elif args.all: viz.create_all()
    else: viz.create_all()

if __name__ == "__main__": main()