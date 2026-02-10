"""
Visualization for Superellipsoid Granule Simulations
=====================================================

Phase maps: point-in-superellipsoid test |x/a|^n + |y/b|^n + |z/c|^n <= 1
Packing fraction: voxel-based with 1-granule border exclusion

Usage:
    python dem_postprocess.py -i ./simulations/dem_config
    python dem_postprocess.py -i ./simulations/dem_config --watch
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
class FrameData:
    time_hours:float; n_granules:int
    positions:np.ndarray; orientations:np.ndarray
    shapes:List[ShapeData]; types:np.ndarray; n_cells:np.ndarray
    domain:Optional[List[float]]=None
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
        return FrameData(
            time_hours=data['time_hours'],n_granules=data['n_granules'],
            positions=np.array(data['positions']),orientations=np.array(data['orientations']),
            shapes=shapes,types=np.array(data['types']),n_cells=np.array(data['n_cells']),
            domain=data.get('domain'))
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
    """
    Rasterize cross-section using point-in-superellipsoid test.
    For each granule: transform pixel grid to body frame, evaluate
    |x/a|^n + |y/b|^n + |z/c|^n <= 1.
    Returns 2D array: 0=void, 1=functional, 2=inert.
    """
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

        # Quick check: does bounding sphere intersect slice?
        if abs(pos[axis] - position) > rb:
            continue

        value = 1.0 if frame.types[i] == 0 else 2.0

        # Bounding box in slice pixel coords
        cx, cy = pos[dims[0]], pos[dims[1]]
        ix_min = max(0, int((cx - rb) / dx) - 1)
        ix_max = min(resolution, int((cx + rb) / dx) + 2)
        iy_min = max(0, int((cy - rb) / dy) - 1)
        iy_max = min(resolution, int((cy + rb) / dy) + 2)

        if ix_max <= ix_min or iy_max <= iy_min:
            continue

        # Build pixel coordinate arrays (vectorized)
        px = (np.arange(ix_min, ix_max) + 0.5) * dx
        py = (np.arange(iy_min, iy_max) + 0.5) * dy
        PX, PY = np.meshgrid(px, py, indexing='ij')  # (nx, ny)

        # Construct world coordinates for the slice
        world = np.empty(PX.shape + (3,))
        if axis == 0:
            world[..., 0] = position; world[..., 1] = PX; world[..., 2] = PY
        elif axis == 1:
            world[..., 0] = PX; world[..., 1] = position; world[..., 2] = PY
        else:
            world[..., 0] = PX; world[..., 1] = PY; world[..., 2] = position

        # Transform to body frame
        relative = world - pos  # broadcasting (nx, ny, 3)
        body = np.einsum('ij,...j->...i', R.T, relative)  # (nx, ny, 3)

        # Superellipsoid inclusion test
        n_exp = shape.n
        se_val = (np.abs(body[..., 0] / shape.a) ** n_exp +
                  np.abs(body[..., 1] / shape.b) ** n_exp +
                  np.abs(body[..., 2] / shape.c) ** n_exp)

        inside = se_val <= 1.0
        image[ix_min:ix_max, iy_min:iy_max][inside] = value

    return image


# =============================================================================
# VOXEL-BASED PACKING FRACTION (border-excluded)
# =============================================================================

def compute_voxel_packing(frame, domain, margin, res=50):
    """
    Packing fraction from 3D voxelization, excluding a border region
    of width `margin` from each wall.
    """
    gmin = np.array([margin, margin, margin])
    gmax = np.array([domain[0]-margin, domain[1]-margin, domain[2]-margin])
    if np.any(gmax <= gmin):
        return 0.0

    xs = np.linspace(gmin[0], gmax[0], res)
    ys = np.linspace(gmin[1], gmax[1], res)
    zs = np.linspace(gmin[2], gmax[2], res)

    filled = 0
    total = res**3
    n_g = frame.n_granules

    # Pre-compute rotation matrices
    Rs = [_quat_to_rot(frame.orientations[g]) for g in range(n_g)]
    sps = [(frame.shapes[g].a, frame.shapes[g].b, frame.shapes[g].c, frame.shapes[g].n)
           for g in range(n_g)]
    rbs = [max(s.a, s.b, s.c) for s in frame.shapes]
    pos = frame.positions

    for ix, px in enumerate(xs):
        for iy, py in enumerate(ys):
            for iz, pz in enumerate(zs):
                p = np.array([px, py, pz])
                for g in range(n_g):
                    d2 = np.sum((p - pos[g])**2)
                    if d2 > rbs[g]**2:
                        continue
                    body = Rs[g].T @ (p - pos[g])
                    a, b, c, n = sps[g]
                    val = abs(body[0]/a)**n + abs(body[1]/b)**n + abs(body[2]/c)**n
                    if val <= 1.0:
                        filled += 1
                        break
    return filled / max(total, 1)


def compute_voxel_packing_fast(frame, domain, margin, res=50):
    """
    Faster voxel packing using vectorized slice-by-slice computation.
    """
    gmin = np.array([margin, margin, margin])
    gmax = np.array([domain[0]-margin, domain[1]-margin, domain[2]-margin])
    if np.any(gmax <= gmin):
        return 0.0

    xs = np.linspace(gmin[0], gmax[0], res)
    ys = np.linspace(gmin[1], gmax[1], res)
    zs = np.linspace(gmin[2], gmax[2], res)

    # Build full 3D grid
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')  # (res, res, res)
    grid = np.stack([X, Y, Z], axis=-1)  # (res, res, res, 3)
    occupied = np.zeros((res, res, res), dtype=bool)

    for g in range(frame.n_granules):
        shape = frame.shapes[g]
        pos = frame.positions[g]
        R = _quat_to_rot(frame.orientations[g])
        rb = max(shape.a, shape.b, shape.c)

        # Bounding box filter
        mask_x = (xs >= pos[0] - rb) & (xs <= pos[0] + rb)
        mask_y = (ys >= pos[1] - rb) & (ys <= pos[1] + rb)
        mask_z = (zs >= pos[2] - rb) & (zs <= pos[2] + rb)

        if not (np.any(mask_x) and np.any(mask_y) and np.any(mask_z)):
            continue

        ix = np.where(mask_x)[0]
        iy = np.where(mask_y)[0]
        iz = np.where(mask_z)[0]

        sub = grid[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1]  # (sx, sy, sz, 3)
        rel = sub - pos
        body = np.einsum('ij,...j->...i', R.T, rel)

        n_exp = shape.n
        se_val = (np.abs(body[..., 0] / shape.a) ** n_exp +
                  np.abs(body[..., 1] / shape.b) ** n_exp +
                  np.abs(body[..., 2] / shape.c) ** n_exp)

        inside = se_val <= 1.0
        occupied[ix[0]:ix[-1]+1, iy[0]:iy[-1]+1, iz[0]:iz[-1]+1] |= inside

    return np.sum(occupied) / occupied.size


# =============================================================================
# VOID PROFILE (slice-based voxel)
# =============================================================================

def compute_void_profile(frame, domain, margin, axis=2, n_slices=30, slice_res=80):
    """
    Void fraction profile along an axis using 2D voxelized slices,
    border-excluded.
    """
    gmin_ax = margin
    gmax_ax = domain[axis] - margin
    if gmax_ax <= gmin_ax:
        return np.array([]), np.array([])

    positions = np.linspace(gmin_ax, gmax_ax, n_slices)
    voids = []

    for pos in positions:
        img = render_phase_map(frame, domain, axis, pos, resolution=slice_res)
        # Only count interior pixels (exclude border in-plane too)
        if axis == 0:
            ext = [domain[1], domain[2]]
        elif axis == 1:
            ext = [domain[0], domain[2]]
        else:
            ext = [domain[0], domain[1]]

        dx = ext[0] / slice_res
        dy = ext[1] / slice_res

        # Border mask in 2D
        border_px_x = int(margin / dx)
        border_px_y = int(margin / dy)
        interior = img[border_px_x:slice_res-border_px_x,
                       border_px_y:slice_res-border_px_y]

        if interior.size == 0:
            voids.append(1.0)
        else:
            voids.append(np.sum(interior == 0) / interior.size)

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

        # Domain
        self.domain = None
        if 'domain' in self.config:
            s = self.config['domain'].get('side_length_um')
            if s: self.domain = np.array([s, s, s])
        if self.domain is None and self.frames and self.frames[0].domain is not None:
            self.domain = np.array(self.frames[0].domain)
        if self.domain is None:
            self.domain = np.array([700.0]*3)
        print(f"Domain: {self.domain[0]:.0f}³ μm")

        # Mean granule radius for border margin
        if self.frames:
            self.margin = np.mean([s.equivalent_radius for s in self.frames[0].shapes])
        else:
            self.margin = 40.0
        print(f"Border margin: {self.margin:.1f} μm (1 mean granule radius)")

    # ---- Phase map: single ----
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
            fp = self.viz_dir / f"phase_{"xyz"[axis]}_{lab}.png"
            plt.savefig(fp, dpi=200, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # ---- Phase evolution ----
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
            fp = self.viz_dir / f"phase_evo_{"xyz"[axis]}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # ---- Three-plane ----
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

    # ---- Voxel packing fraction evolution ----
    def plot_packing_evolution(self, save=True):
        if not self.frames: return None

        # Compute voxel packing for each frame
        print("Computing voxel packing fractions (border-excluded)...")
        times, phis = [], []
        for fi, frame in enumerate(self.frames):
            phi = compute_voxel_packing_fast(frame, self.domain, self.margin, res=40)
            times.append(frame.time_days)
            phis.append(phi)
            if fi % max(1, len(self.frames)//5) == 0:
                print(f"  Frame {fi}/{len(self.frames)}: φ = {phi:.3f}")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Voxel packing over time
        ax = axes[0, 0]
        ax.plot(times, phis, 'bo-', ms=4, lw=2)
        ax.axhline(0.64, color='r', ls='--', alpha=0.5, label='RCP sphere')
        ax.axhline(0.55, color='orange', ls='--', alpha=0.5, label='RLP sphere')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction (voxel)')
        ax.set_title(f'Packing Fraction (margin={self.margin:.0f}μm)')
        ax.legend(); ax.grid(True, alpha=0.3)

        # Void fraction over time
        ax = axes[0, 1]
        vfs = [1-p for p in phis]
        ax.plot(times, vfs, 'ko-', ms=4, lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Void Fraction (voxel)')
        ax.set_title('Void Fraction'); ax.grid(True, alpha=0.3)

        # From history if available
        ax = axes[1, 0]
        if self.history and 'packing_fraction' in self.history:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, self.history['packing_fraction'], 'b-', lw=2, label='Sim history')
            ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing φ')
            ax.set_title('Packing (from sim history)'); ax.legend()
            ax.grid(True, alpha=0.3)
            if 'max_overlap_um' in self.history:
                ax2 = ax.twinx()
                ax2.plot(th, self.history['max_overlap_um'], 'm--', lw=1.5, alpha=0.7)
                ax2.set_ylabel('Max Overlap (μm)', color='m')
                ax2.tick_params(axis='y', labelcolor='m')

        # Void profiles (final frame)
        ax = axes[1, 1]
        frame = self.frames[-1]
        for a, lab, c in [(0,'X','red'), (1,'Y','green'), (2,'Z','blue')]:
            p, vf = compute_void_profile(frame, self.domain, self.margin, axis=a,
                                          n_slices=25, slice_res=60)
            if len(p) > 0:
                ax.plot(p, vf, '-', color=c, lw=2, label=f'{lab}')
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Local Void Fraction')
        ax.set_title('Void Profiles (final, border-excluded)')
        ax.legend(); ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "packing_evolution.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # ---- Detailed frame ----
    def plot_detailed_frame(self, frame_index=-1, save=True):
        if not self.frames: return None
        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(2, 3)

        for col, ax_id in enumerate([2, 1, 0]):
            ax = fig.add_subplot(gs[0, col])
            sp = self.domain[ax_id] / 2
            img = render_phase_map(frame, self.domain, ax_id, sp, 250)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(ax, img, ext, xl, yl, f'{"XY XZ YZ".split()[col]}')

        ax = fig.add_subplot(gs[1, 0])
        for a, lab, c in [(0,'X','red'), (1,'Y','green'), (2,'Z','blue')]:
            p, vf = compute_void_profile(frame, self.domain, self.margin, axis=a,
                                          n_slices=20, slice_res=50)
            if len(p) > 0: ax.plot(p, vf, '-', color=c, lw=2, label=lab)
        ax.set_xlabel('Position (μm)'); ax.set_ylabel('Void Fraction')
        ax.set_title('Void Profiles'); ax.legend(); ax.grid(True, alpha=0.3)

        ax = fig.add_subplot(gs[1, 1])
        fr = [s.equivalent_radius for i,s in enumerate(frame.shapes) if frame.functional_mask[i]]
        ir = [s.equivalent_radius for i,s in enumerate(frame.shapes) if frame.inert_mask[i]]
        ax.hist(fr, bins=15, alpha=0.6, color='red', label='Functional')
        ax.hist(ir, bins=15, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equiv. Radius (μm)'); ax.set_ylabel('Count')
        ax.set_title('Size Distribution'); ax.legend()

        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        phi = compute_voxel_packing_fast(frame, self.domain, self.margin, res=35)
        stats = (f"t = {frame.time_hours:.1f}h ({frame.time_days:.2f}d)\n\n"
                 f"Func: {int(np.sum(frame.functional_mask))}  "
                 f"Inert: {int(np.sum(frame.inert_mask))}  "
                 f"Total: {frame.n_granules}\n\n"
                 f"Mean R func: {np.mean(fr):.1f} μm\n"
                 f"Mean R inert: {np.mean(ir):.1f} μm\n\n"
                 f"Voxel packing φ: {phi:.3f}\n"
                 f"Voxel void frac: {1-phi:.3f}\n\n"
                 f"Border margin: {self.margin:.1f} μm\n"
                 f"Total cells: {int(np.sum(frame.n_cells))}\n"
                 f"Domain: {self.domain[0]:.0f}³ μm")
        ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=11,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Detailed — t={frame.time_days:.2f}d', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index==-1 else f"f{frame_index}"
            fp = self.viz_dir / f"detailed_{lab}.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # ---- History ----
    def plot_history(self, save=True):
        if not self.history: print("No history"); return None
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
        ax.plot(t, self.history['packing_fraction'], 'b-', lw=2, label='Voxel φ')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Packing Fraction')
        ax.set_title('Packing (voxel, border-excluded)'); ax.grid(True, alpha=0.3)
        if 'max_overlap_um' in self.history:
            ax2 = ax.twinx()
            ax2.plot(t, self.history['max_overlap_um'], 'm--', lw=1.5, alpha=0.7)
            ax2.set_ylabel('Max Overlap (μm)', color='m')
            ax2.tick_params(axis='y', labelcolor='m')
        ax.legend()

        plt.tight_layout()
        if save:
            fp = self.viz_dir / "history.png"
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"Saved: {fp}")
        return fig

    # ---- Animation ----
    def create_animation(self, axis=2, resolution=200, fps=6, save=True):
        if not HAS_ANIMATION or not self.frames: return None
        print(f"Creating animation ({len(self.frames)} frames)...")
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)

        fig, (ax_img, ax_met) = plt.subplots(1, 2, figsize=(14, 6))
        img0 = render_phase_map(self.frames[0], self.domain, axis, sp, resolution)
        im = ax_img.imshow(img0.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
                           extent=ext, aspect='equal', interpolation='nearest')
        ax_img.set_xlabel(xl); ax_img.set_ylabel(yl)
        ax_img.legend(handles=PHASE_LEGEND, loc='upper right', fontsize=8)

        if self.history and len(self.history.get('time_hours',[])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax_met.plot(th, self.history['mean_coordination'], 'g-', lw=2)
            ax_met.set_xlabel('Time (days)'); ax_met.set_ylabel('Coordination', color='g')
            ax_met.tick_params(axis='y', labelcolor='g'); ax_met.grid(True, alpha=0.3)
            ax2 = ax_met.twinx()
            ax2.plot(th, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
            ax2.set_ylabel('Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')

        tl = ax_met.axvline(x=0, color='black', lw=2, ls='--')
        title = fig.suptitle('', fontsize=12, fontweight='bold')

        def update(fi):
            fr = self.frames[fi]
            im.set_data(render_phase_map(fr, self.domain, axis, sp, resolution).T)
            tl.set_xdata([fr.time_days, fr.time_days])
            title.set_text(f't={fr.time_hours:.1f}h ({fr.time_days:.2f}d)')
            return [im]

        anim = FuncAnimation(fig, update, frames=len(self.frames),
                             interval=1000//fps, blit=True)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f"animation_{"xyz"[axis]}.gif"
            try:
                anim.save(str(fp), writer=PillowWriter(fps=fps), dpi=120)
                print(f"Saved: {fp}")
            except Exception as e:
                print(f"Error: {e}"); fp = None
            plt.close(fig)
            return str(fp) if fp else None
        plt.show(); return None

    # ---- Live snapshot ----
    def plot_snapshot(self, save=True):
        if not self.frames: return None
        frame = self.frames[-1]
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        for idx, (r, c, ax_id) in enumerate([(0,0,2), (0,1,1)]):
            sp = self.domain[ax_id] / 2
            img = render_phase_map(frame, self.domain, ax_id, sp, 200)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(axes[r,c], img, ext, xl, yl,
                        f'{"XY XZ".split()[idx]} (t={frame.time_hours:.1f}h)')

        ax = axes[1, 0]
        if self.history and len(self.history.get('time_hours',[])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, self.history['mean_coordination'], 'g-', lw=2)
            ax.set_ylabel('Coordination', color='g'); ax.tick_params(axis='y', labelcolor='g')
            ax.grid(True, alpha=0.3)
            ax2 = ax.twinx()
            ax2.plot(th, self.history['n_bridges'], 'r-', lw=2, alpha=0.7)
            ax2.set_ylabel('Bridges', color='r'); ax2.tick_params(axis='y', labelcolor='r')
            ax.axvline(frame.time_hours/24, color='black', ls='--', lw=1.5)
        ax.set_xlabel('Time (days)'); ax.set_title('Network')

        ax = axes[1, 1]
        if self.history and len(self.history.get('time_hours',[])) > 1:
            th = np.array(self.history['time_hours']) / 24
            ax.plot(th, self.history['packing_fraction'], 'b-', lw=2)
            ax.set_ylabel('Voxel φ', color='b'); ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True, alpha=0.3)
            if 'max_overlap_um' in self.history:
                ax2 = ax.twinx()
                ax2.plot(th, self.history['max_overlap_um'], 'm-', lw=1.5, alpha=0.7)
                ax2.set_ylabel('Max Overlap (μm)', color='m')
                ax2.tick_params(axis='y', labelcolor='m')
        ax.set_xlabel('Time (days)'); ax.set_title('Packing & Overlap')

        fig.legend(handles=PHASE_LEGEND, loc='upper center', ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 1.0))
        plt.suptitle(f'{self.config_name} — t={frame.time_hours:.1f}h — '
                     f'{frame.n_granules} granules — {len(self.frames)} frames',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save:
            plt.savefig(self.viz_dir/"live_snapshot.png", dpi=150, bbox_inches='tight')
            plt.close(fig)
        return None

    # ---- Reload ----
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
        known = {str(fp) for fp in self.frame_files}
        added = 0
        for fp in new_files:
            if str(fp) not in known:
                fr = load_frame(str(fp))
                if fr is not None:
                    self.frames.append(fr); self.frame_files.append(fp); added += 1
        if added > 0:
            self.frames.sort(key=lambda f: f.time_hours)
            print(f"  [watch] +{added} → {len(self.frames)} frames "
                  f"(t={self.frames[-1].time_hours:.1f}h)")
        return added

    # ---- Watch ----
    def watch(self, interval=30.0, full_every=5):
        print(f"\n{'='*60}\nWATCH MODE — every {interval:.0f}s, full plots every {full_every} cycles\n{'='*60}\n")
        matplotlib.use('Agg')
        cycle = 0
        try:
            while True:
                n_new = self.reload_frames()
                if n_new > 0 or cycle == 0:
                    self.plot_snapshot(save=True)
                    if cycle % full_every == 0 and len(self.frames) >= 2:
                        print("  [watch] Full plots...")
                        for fn in [self.plot_history,
                                   lambda: self.plot_phase_map(axis=2),
                                   lambda: self.plot_phase_evolution(axis=2)]:
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

    # ---- All ----
    def create_all(self):
        print(f"\n{'='*60}\nGenerating All\n{'='*60}")
        print("\n1. History..."); self.plot_history()
        print("\n2. Packing evolution (voxel)..."); self.plot_packing_evolution()
        print("\n3. Phase evolution...")
        for a in [2,1]: self.plot_phase_evolution(axis=a)
        print("\n4. Three-plane...")
        if self.frames:
            self.plot_three_plane(0); self.plot_three_plane(-1)
        print("\n5. Detailed...")
        if self.frames:
            self.plot_detailed_frame(0); self.plot_detailed_frame(-1)
        print("\n6. Phase maps (final)...")
        if self.frames:
            for a in [2,1,0]: self.plot_phase_map(-1, axis=a)
        print("\n7. Animation...")
        if self.frames: self.create_animation(axis=2, fps=6)
        print(f"\nDone → {self.viz_dir}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='./simulations/Trial10')
    parser.add_argument('--all', '-a', action='store_true')
    parser.add_argument('--phase-map', '-p', action='store_true')
    parser.add_argument('--watch', '-w', action='store_true')
    parser.add_argument('--watch-interval', type=float, default=30.0)
    parser.add_argument('--animation', '-s', action='store_true')
    parser.add_argument('--fps', type=int, default=6)
    parser.add_argument('--axis', type=int, default=2)
    args = parser.parse_args()

    viz = Visualizer(args.input)
    if args.watch: viz.watch(interval=args.watch_interval)
    elif args.phase_map: viz.plot_phase_map(axis=args.axis)
    elif args.animation: viz.create_animation(axis=args.axis, fps=args.fps)
    elif args.all: viz.create_all()
    else: viz.create_all()

if __name__ == "__main__": main()