"""
Unified Post-Processing for Granular Hydrogel Simulations
==========================================================
Reads ONLY from disk:  *_frame_*.json  +  *_history.json
Works with both DEM Superellipsoid and Cahn-Hilliard outputs.

Visualizations produced (all saved into <input_dir>/visualizations/):
  1. Z-stack evolution — cross-sections color-coded by z-depth (jet)
  2. Metric time-series — contacts, coordination, stress, displacement …
  3. Phase snapshot evolution — functional / inert / void rows
  4. 3D rotating isosurface GIF — isolated phases, 360° rotation
  5. Tri-plane animated GIF — X/Y/Z mid-plane over time
  6. Cell stress / aspect-ratio maps & evolution (if cell_data present)
  7. Three-plane snapshot, detailed dashboard, phase evolution strip

Usage:
    python dem_postprocess_unified.py                        # folder picker
    python dem_postprocess_unified.py -i ./simulations/run1  # explicit path
    python dem_postprocess_unified.py -i ./simulations/run1 --zstack --rotate3d
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec
from pathlib import Path
import json, argparse, warnings, sys, glob
from typing import List, Optional, Dict
from dataclasses import dataclass

try:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    HAS_3D = True
except ImportError:
    HAS_3D = False
try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    HAS_ANIM = True
except ImportError:
    HAS_ANIM = False
try:
    from scipy.ndimage import label as sp_label, gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
try:
    from skimage.measure import marching_cubes
    HAS_MC = True
except ImportError:
    try:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
        HAS_MC = True
    except ImportError:
        HAS_MC = False
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

warnings.filterwarnings('ignore', category=UserWarning)

# =========================================================================
# CONSTANTS
# =========================================================================
FUNC_COLOR  = '#d62728'
INERT_COLOR = '#1f77b4'
VOID_COLOR  = '#2ca02c'

from matplotlib.colors import LinearSegmentedColormap
PHASE_CMAP = LinearSegmentedColormap.from_list('phase', ['white', 'red', 'blue'], N=3)
PHASE_LEGEND = [
    Patch(facecolor='white', edgecolor='gray', label='Void'),
    Patch(facecolor='red',   label='Functional'),
    Patch(facecolor='blue',  label='Inert'),
]

def _progress(it, **kw):
    return tqdm(it, **kw) if HAS_TQDM else it


# =========================================================================
# DATA STRUCTURES
# =========================================================================

@dataclass
class ShapeData:
    a: float; b: float; c: float; n: float; roughness: float = 0.0
    @property
    def eq_radius(self):      return (self.a * self.b * self.c) ** (1/3)
    @property
    def bounding_radius(self): return max(self.a, self.b, self.c)

@dataclass
class CellData:
    total_cells:  int
    parent:       np.ndarray   # (n_cells,)
    world_pos:    np.ndarray   # (n_cells, 3)
    is_bridging:  np.ndarray   # (n_cells,) bool
    gap_um:       np.ndarray
    stress_nN:    np.ndarray
    aspect_ratio: np.ndarray

@dataclass
class FrameData:
    time_hours:   float
    n_granules:   int
    positions:    np.ndarray   # (ng, 3)
    orientations: np.ndarray   # (ng, 4) quaternions
    shapes:       List[ShapeData]
    types:        np.ndarray   # (ng,)  0=func 1=inert
    n_cells:      np.ndarray
    domain:       np.ndarray   # (3,)
    cell_data:    Optional[CellData] = None
    @property
    def time_days(self):       return self.time_hours / 24.0
    @property
    def functional_mask(self): return self.types == 0
    @property
    def inert_mask(self):      return self.types == 1


# =========================================================================
# FRAME / HISTORY LOADING
# =========================================================================

def _quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*w*z,   2*x*z+2*w*y],
        [2*x*y+2*w*z,   1-2*x*x-2*z*z, 2*y*z-2*w*x],
        [2*x*z-2*w*y,   2*y*z+2*w*x,   1-2*x*x-2*y*y]])


def load_frame(filepath) -> Optional[FrameData]:
    """Load a single frame JSON (works with both old and new DEM outputs)."""
    try:
        with open(filepath) as f:
            d = json.load(f)

        # --- shapes ---
        shape_key = None
        for k in ('true_shapes', 'sim_shapes', 'shapes'):
            if k in d: shape_key = k; break
        if shape_key is None:
            print(f"  [WARN] no shape key in {Path(filepath).name}"); return None
        shapes = [ShapeData(s['a'], s['b'], s['c'],
                            s.get('n', 2.0), s.get('roughness', 0.0))
                  for s in d[shape_key]]

        # --- domain ---
        if 'domain' in d:
            domain = np.array(d['domain'], dtype=float)
        else:
            # try to infer from config or positions
            pos = np.array(d['positions'])
            side = float(np.max(pos) * 1.1)
            domain = np.array([side, side, side])

        # --- cell data ---
        cd = None
        if 'cell_data' in d:
            c = d['cell_data']
            cd = CellData(
                total_cells  = c.get('total_cells', 0),
                parent       = np.array(c.get('parent', []),       dtype=int),
                world_pos    = np.array(c.get('world_pos', []),    dtype=float),
                is_bridging  = np.array(c.get('is_bridging', []),  dtype=bool),
                gap_um       = np.array(c.get('gap_um', []),       dtype=float),
                stress_nN    = np.array(c.get('stress_nN', []),    dtype=float),
                aspect_ratio = np.array(c.get('aspect_ratio', []), dtype=float),
            )

        return FrameData(
            time_hours   = d.get('time_hours', 0.0),
            n_granules   = d.get('n_granules', len(shapes)),
            positions    = np.array(d['positions'],    dtype=float),
            orientations = np.array(d['orientations'], dtype=float),
            shapes       = shapes,
            types        = np.array(d['types'],   dtype=int),
            n_cells      = np.array(d.get('n_cells', np.zeros(len(shapes))), dtype=int),
            domain       = domain,
            cell_data    = cd,
        )
    except Exception as e:
        print(f"  [WARN] {Path(filepath).name}: {e}")
        return None


def load_history(filepath) -> List[Dict]:
    """Load history JSON.  Handles both list-of-dicts and dict-of-lists."""
    with open(filepath) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        return raw
    # dict-of-lists → list-of-dicts
    keys = list(raw.keys())
    n = len(raw[keys[0]])
    return [{k: raw[k][i] for k in keys} for i in range(n)]


def discover_files(folder: Path):
    """Auto-discover frame JSONs and history JSON inside *folder*."""
    folder = Path(folder)
    # Frame files: anything matching *_frame_*.json
    frame_files = sorted(folder.glob('*_frame_*.json'))
    # History: anything matching *_history.json
    hist_files = sorted(folder.glob('*_history.json'))
    hist_file = hist_files[0] if hist_files else None
    # Config
    cfg_files = sorted(folder.glob('*_config.json'))
    cfg_file = cfg_files[0] if cfg_files else None
    return frame_files, hist_file, cfg_file


def select_folder():
    """Open a GUI folder picker, fall back to CLI input."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        r = tk.Tk(); r.withdraw(); r.attributes('-topmost', True)
        fp = filedialog.askdirectory(title="Select simulation output folder")
        r.destroy()
        if fp: return Path(fp)
    except Exception:
        pass
    p = input("  Enter path to simulation output folder: ").strip()
    return Path(p) if p else None


# =========================================================================
# RENDERING  —  superellipsoid slice & 3-D voxelisation
# =========================================================================

def render_phase_map(frame: FrameData, axis: int, position: float,
                     resolution: int = 200) -> np.ndarray:
    """2-D phase map: 0 = void, 1 = functional, 2 = inert."""
    domain = frame.domain
    if axis == 0:   dims = [1, 2]
    elif axis == 1: dims = [0, 2]
    else:           dims = [0, 1]
    ext = [domain[dims[0]], domain[dims[1]]]
    image = np.zeros((resolution, resolution), dtype=np.float32)
    dx, dy = ext[0] / resolution, ext[1] / resolution

    for i in range(frame.n_granules):
        sh  = frame.shapes[i]
        pos = frame.positions[i]
        R   = _quat_to_rot(frame.orientations[i])
        rb  = sh.bounding_radius
        if abs(pos[axis] - position) > rb:
            continue
        lo0 = max(0, int((pos[dims[0]] - rb) / dx))
        hi0 = min(resolution, int((pos[dims[0]] + rb) / dx) + 1)
        lo1 = max(0, int((pos[dims[1]] - rb) / dy))
        hi1 = min(resolution, int((pos[dims[1]] + rb) / dy) + 1)
        if lo0 >= hi0 or lo1 >= hi1:
            continue
        gx, gy = np.meshgrid((np.arange(lo0, hi0) + 0.5) * dx,
                              (np.arange(lo1, hi1) + 0.5) * dy, indexing='ij')
        pts = np.zeros((*gx.shape, 3))
        pts[..., dims[0]] = gx; pts[..., dims[1]] = gy; pts[..., axis] = position
        body = np.einsum('ij,...j->...i', R.T, pts - pos)
        se = (np.abs(body[...,0]/sh.a)**sh.n +
              np.abs(body[...,1]/sh.b)**sh.n +
              np.abs(body[...,2]/sh.c)**sh.n)
        inside = se <= 1.0
        val = 1.0 if frame.types[i] == 0 else 2.0
        sub = image[lo0:hi0, lo1:hi1]
        sub[inside & (sub == 0)] = val
    return image


def render_3d_volume(frame: FrameData, resolution: int = 50):
    """Voxelise into boolean arrays (func, inert, void)."""
    dom = frame.domain; res = resolution
    xs = np.linspace(0, dom[0], res)
    ys = np.linspace(0, dom[1], res)
    zs = np.linspace(0, dom[2], res)
    func_v  = np.zeros((res, res, res), dtype=bool)
    inert_v = np.zeros((res, res, res), dtype=bool)

    for i in _progress(range(frame.n_granules), desc="    Voxelising"):
        sh = frame.shapes[i]; pos = frame.positions[i]
        R = _quat_to_rot(frame.orientations[i]); rb = sh.bounding_radius
        ix0 = max(0, np.searchsorted(xs, pos[0]-rb)-1)
        ix1 = min(res, np.searchsorted(xs, pos[0]+rb)+1)
        iy0 = max(0, np.searchsorted(ys, pos[1]-rb)-1)
        iy1 = min(res, np.searchsorted(ys, pos[1]+rb)+1)
        iz0 = max(0, np.searchsorted(zs, pos[2]-rb)-1)
        iz1 = min(res, np.searchsorted(zs, pos[2]+rb)+1)
        if ix0 >= ix1 or iy0 >= iy1 or iz0 >= iz1: continue
        gx, gy, gz = np.meshgrid(xs[ix0:ix1], ys[iy0:iy1], zs[iz0:iz1], indexing='ij')
        body = np.einsum('ij,...j->...i', R.T,
                         np.stack([gx, gy, gz], axis=-1) - pos)
        se = (np.abs(body[...,0]/sh.a)**sh.n +
              np.abs(body[...,1]/sh.b)**sh.n +
              np.abs(body[...,2]/sh.c)**sh.n)
        inside = se <= 1.0
        if frame.types[i] == 0:
            func_v[ix0:ix1, iy0:iy1, iz0:iz1] |= inside
        else:
            inert_v[ix0:ix1, iy0:iy1, iz0:iz1] |= inside
    return func_v, inert_v, ~(func_v | inert_v)


# =========================================================================
# PLOT HELPERS
# =========================================================================

def _ext(domain, axis):
    if axis == 2: return [0, domain[0], 0, domain[1]], 'X (μm)', 'Y (μm)'
    if axis == 1: return [0, domain[0], 0, domain[2]], 'X (μm)', 'Z (μm)'
    return [0, domain[1], 0, domain[2]], 'Y (μm)', 'Z (μm)'

def _plot_phase(ax, image, extent, xlabel, ylabel, title=None):
    ax.imshow(image.T, origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
              extent=extent, aspect='equal', interpolation='nearest')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    if title: ax.set_title(title, fontsize=10)

def _pick_frame_indices(n, count):
    return np.unique(np.linspace(0, n-1, min(count, n), dtype=int))

def _get_hist(h, key, default=0):
    return h.get(key, default)


# =========================================================================
# VISUALIZER
# =========================================================================

class Visualizer:
    """
    File-based post-processor.
    Point it at a folder containing *_frame_*.json and *_history.json.
    """

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.viz_dir = self.folder / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)

        frame_files, hist_file, cfg_file = discover_files(self.folder)

        # ── config ──
        self.config: Dict = {}
        if cfg_file:
            with open(cfg_file) as f: self.config = json.load(f)
            print(f"  Config : {cfg_file.name}")

        # ── history ──
        self.history: List[Dict] = []
        if hist_file:
            self.history = load_history(hist_file)
            print(f"  History: {hist_file.name}  ({len(self.history)} records)")

        # ── frames ──
        self.frames: List[FrameData] = []
        print(f"  Loading {len(frame_files)} frame files …")
        for ff in _progress(frame_files, desc="  Frames"):
            fr = load_frame(ff)
            if fr is not None:
                self.frames.append(fr)
        # Sort by time
        self.frames.sort(key=lambda f: f.time_hours)
        print(f"  Loaded  {len(self.frames)} valid frames")

        # ── domain ──
        if self.frames and self.frames[0].domain is not None:
            self.domain = self.frames[0].domain.copy()
        else:
            ds = self.config.get('domain', {}).get('side_length_um', 500.0)
            self.domain = np.array([ds, ds, ds])
        print(f"  Domain : {self.domain}")

        # ── has cell data? ──
        self.has_cells = any(f.cell_data is not None and f.cell_data.total_cells > 0
                            for f in self.frames)
        if self.has_cells:
            print("  Cell data detected ✓")

    # =================================================================
    # 1.  Z-STACK EVOLUTION  —  jet-coloured depth slices
    # =================================================================

    def plot_zstack_evolution(self, n_times=5, n_slices=8, resolution=150, save=True):
        """Cross-sections at multiple z-depths, jet-coded by z position."""
        print("\n  [1] Z-stack evolution …")
        if not self.frames:
            print("      No frames"); return None

        idxs = _pick_frame_indices(len(self.frames), n_times)
        z_fracs = np.linspace(0.1, 0.9, n_slices)
        jet = plt.cm.jet
        ncol = len(idxs)

        fig, axes = plt.subplots(n_slices + 1, ncol,
                                 figsize=(3.5 * ncol, 2.6 * (n_slices + 1)))
        if ncol == 1: axes = axes[:, np.newaxis]

        for col, fi in enumerate(idxs):
            frame = self.frames[fi]
            composite = np.zeros((resolution, resolution, 4))

            for row, zf in enumerate(z_fracs):
                z_pos = self.domain[2] * zf
                img = render_phase_map(frame, axis=2, position=z_pos,
                                       resolution=resolution)
                color = jet(zf)
                rgba = np.zeros((resolution, resolution, 4))
                rgba[img == 1] = [color[0],       color[1]*0.3, color[2]*0.3, 0.85]
                rgba[img == 2] = [color[0]*0.3, color[1]*0.3, color[2],       0.85]

                ax = axes[row + 1, col]
                ax.imshow(rgba.transpose(1, 0, 2), origin='lower',
                          extent=[0, self.domain[0], 0, self.domain[1]])
                ax.set_title(f'z={zf:.2f}', fontsize=7)
                ax.tick_params(labelsize=5)
                if col > 0: ax.set_yticklabels([])
                if row < n_slices - 1: ax.set_xticklabels([])

                alpha = rgba[..., 3:4]
                composite = composite * (1 - alpha * 0.35) + rgba * 0.35

            composite[..., 3] = np.clip(composite[..., 3], 0, 1)
            axes[0, col].imshow(np.clip(composite.transpose(1, 0, 2), 0, 1),
                                origin='lower',
                                extent=[0, self.domain[0], 0, self.domain[1]])
            axes[0, col].set_title(f't={frame.time_hours:.1f}h\ncomposite', fontsize=9)

        sm = ScalarMappable(cmap='jet', norm=Normalize(0, self.domain[2]))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.25, pad=0.01)
        cbar.set_label('Z depth (μm)', fontsize=9)
        plt.suptitle('Z-Stack Evolution — Depth-Coded Cross Sections',
                     fontsize=13, y=1.005)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / 'zstack_evolution.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"      → {fp}")
        return fig

    # =================================================================
    # 2.  METRIC TIME-SERIES  (contact, coordination, stress …)
    # =================================================================

    def plot_metrics(self, save=True):
        """Six-panel time-evolution summary."""
        print("\n  [2] Metric time-series …")
        if not self.history:
            print("      No history"); return None

        h = self.history
        t = np.array([_get_hist(r, 'time_hours', 0) for r in h]) / 24.0

        fig, axes = plt.subplots(2, 3, figsize=(17, 10))

        # (0,0) contacts / bridges
        ax = axes[0, 0]
        ax.plot(t, [_get_hist(r, 'n_contacts') for r in h],  'b-', lw=2, label='Contacts')
        ax.plot(t, [_get_hist(r, 'n_bridges')  for r in h],  'r-', lw=2, label='Bridges')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Count')
        ax.set_title('Interactions'); ax.legend(); ax.grid(True, alpha=0.3)

        # (0,1) coordination
        ax = axes[0, 1]
        ax.plot(t, [_get_hist(r, 'mean_coordination') for r in h], 'g-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean Coordination')
        ax.set_title('Contact Network'); ax.grid(True, alpha=0.3)

        # (0,2) cluster counts
        ax = axes[0, 2]
        fc = [_get_hist(r, 'func_nc', _get_hist(r, 'func_n_clusters', 0)) for r in h]
        vc = [_get_hist(r, 'void_nc', _get_hist(r, 'void_n_clusters', 0)) for r in h]
        ax.plot(t, fc, '-', color=FUNC_COLOR, lw=2, label='Functional')
        ax.plot(t, vc, '-', color=VOID_COLOR, lw=2, label='Void')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('# Clusters')
        ax.set_title('Phase Connectivity'); ax.legend(); ax.grid(True, alpha=0.3)

        # (1,0) velocity
        ax = axes[1, 0]
        mv = [max(_get_hist(r, 'max_velocity', 0), 1e-12) for r in h]
        ax.semilogy(t, mv, 'm-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Max Velocity (μm/hr)')
        ax.set_title('Activity Level'); ax.grid(True, alpha=0.3)

        # (1,1) displacement
        ax = axes[1, 1]
        ax.plot(t, [_get_hist(r, 'disp_func')  for r in h],
                '-', color=FUNC_COLOR, lw=2, label='Functional')
        ax.plot(t, [_get_hist(r, 'disp_inert') for r in h],
                '-', color=INERT_COLOR, lw=2, label='Inert')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean Disp. (μm)')
        ax.set_title('Displacement'); ax.legend(); ax.grid(True, alpha=0.3)

        # (1,2) stress
        ax = axes[1, 2]
        ax.plot(t, [_get_hist(r, 'mean_stress') for r in h],
                '-', color='orange', lw=2, label='Mean')
        ax.plot(t, [_get_hist(r, 'max_stress')  for r in h],
                '--', color='red', lw=1.5, label='Max')
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Stress (nN)')
        ax.set_title('Cell Stress'); ax.legend(); ax.grid(True, alpha=0.3)

        plt.suptitle('Simulation Metrics', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / 'metrics_evolution.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"      → {fp}")
        return fig

    # =================================================================
    # 3.  PHASE SNAPSHOT EVOLUTION  (func / inert / void rows)
    # =================================================================

    def plot_phase_snapshots(self, n_times=5, resolution=200, save=True):
        """Three-row panel: functional / inert / void at selected times."""
        print("\n  [3] Phase snapshot evolution …")
        if not self.frames:
            print("      No frames"); return None

        idxs = _pick_frame_indices(len(self.frames), n_times)
        ext, xl, yl = _ext(self.domain, 2)
        ncol = len(idxs)
        labels_row = ['Functional', 'Inert', 'Void']
        cmaps_row  = ['Reds', 'Blues', 'Greens']

        fig, axes = plt.subplots(3, ncol, figsize=(3.5 * ncol, 10))
        if ncol == 1: axes = axes[:, np.newaxis]

        for col, fi in enumerate(idxs):
            fr = self.frames[fi]
            img = render_phase_map(fr, axis=2, position=self.domain[2]/2,
                                   resolution=resolution)
            fields = [(img == 1).astype(float),
                      (img == 2).astype(float),
                      (img == 0).astype(float)]
            for row in range(3):
                ax = axes[row, col]
                ax.imshow(fields[row].T, origin='lower', extent=ext,
                          cmap=cmaps_row[row], vmin=0, vmax=1,
                          interpolation='nearest')
                ax.set_title(f't={fr.time_hours:.1f}h', fontsize=9)
                if col == 0: ax.set_ylabel(labels_row[row], fontsize=11)
                if col > 0:  ax.set_yticklabels([])

        plt.suptitle('Phase Snapshot Evolution (mid-Z)', fontsize=13)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / 'phase_snapshots.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"      → {fp}")
        return fig

    # =================================================================
    # 4.  3-D ROTATING ISOSURFACE GIF
    # =================================================================

    def create_3d_rotating_gif(self, resolution=50, n_rot_frames=72,
                               fps=12, save=True):
        """Rotating 360° GIF with functional / inert / void isosurfaces."""
        print("\n  [4] 3D rotating isosurface GIF …")
        if not HAS_ANIM or not HAS_3D:
            print("      Need matplotlib animation + mpl_toolkits.mplot3d"); return None
        if not self.frames:
            print("      No frames"); return None

        frame = self.frames[-1]
        func_v, inert_v, void_v = render_3d_volume(frame, resolution)

        # Smooth for nicer surfaces
        if HAS_SCIPY:
            func_s  = gaussian_filter(func_v.astype(float),  sigma=1.0)
            inert_s = gaussian_filter(inert_v.astype(float), sigma=1.0)
            void_s  = gaussian_filter(void_v.astype(float),  sigma=1.0)
        else:
            func_s, inert_s, void_s = [v.astype(float) for v in
                                        (func_v, inert_v, void_v)]

        volumes = [func_s, inert_s, void_s]
        colors  = [FUNC_COLOR, INERT_COLOR, VOID_COLOR]
        titles  = ['Functional', 'Inert', 'Void']

        use_mc = HAS_MC
        meshes = []
        if use_mc:
            for vol in volumes:
                if vol.max() < 0.3:
                    meshes.append(None); continue
                try:
                    verts, faces, _, _ = marching_cubes(vol, level=0.5)
                    meshes.append((verts, faces))
                except Exception:
                    meshes.append(None)

        fig = plt.figure(figsize=(18, 6))
        ax_list = []
        for i in range(3):
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
            ax.set_title(titles[i], fontsize=12, fontweight='bold', color=colors[i])
            ax.set_xlim(0, resolution); ax.set_ylim(0, resolution)
            ax.set_zlim(0, resolution)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.set_box_aspect([1, 1, 1])
            if use_mc and meshes[i] is not None:
                v, f = meshes[i]
                mc = Poly3DCollection(v[f], alpha=0.55,
                                      facecolor=colors[i], edgecolor='none')
                ax.add_collection3d(mc)
            elif not use_mc:
                # Voxel fallback (subsample for speed)
                vs = volumes[i] > 0.5
                vs[::2, :, :] = False; vs[:, ::2, :] = False
                ax.voxels(vs, facecolors=colors[i], edgecolor='none', alpha=0.4)
            ax_list.append(ax)
        plt.tight_layout()

        def _update(idx):
            angle = idx * (360.0 / n_rot_frames)
            for ax in ax_list:
                ax.view_init(elev=25, azim=angle)
            return ax_list

        anim = FuncAnimation(fig, _update, frames=n_rot_frames,
                             interval=1000 // fps, blit=False)
        if save:
            fp = self.viz_dir / '3d_phases_rotating.gif'
            print(f"      Rendering {n_rot_frames} frames …")
            anim.save(str(fp), writer=PillowWriter(fps=fps))
            plt.close(fig)
            print(f"      → {fp}")
        return anim

    # =================================================================
    # 5.  TRI-PLANE ANIMATED GIF  (X / Y / Z mid-planes over time)
    # =================================================================

    def create_triplane_animation(self, resolution=150, fps=4, save=True):
        """Animated GIF: XY, XZ, YZ mid-plane cross-sections evolving."""
        print("\n  [5] Tri-plane animation …")
        if not HAS_ANIM:
            print("      Need matplotlib.animation"); return None
        if len(self.frames) < 2:
            print("      Need ≥ 2 frames"); return None

        axis_ids = [2, 1, 0]
        plane_labels = ['XY (mid-Z)', 'XZ (mid-Y)', 'YZ (mid-X)']

        # Pre-render
        print(f"      Pre-rendering {len(self.frames)} × 3 planes …")
        all_imgs = []
        for fr in _progress(self.frames, desc="      Render"):
            imgs = []
            for ax_id in axis_ids:
                sp = self.domain[ax_id] / 2
                imgs.append(render_phase_map(fr, axis=ax_id, position=sp,
                                             resolution=resolution))
            all_imgs.append(imgs)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ims = []
        for col in range(3):
            ext, xl, yl = _ext(self.domain, axis_ids[col])
            im = axes[col].imshow(np.zeros((resolution, resolution)).T,
                                  origin='lower', cmap=PHASE_CMAP, vmin=0, vmax=2,
                                  extent=ext, interpolation='nearest')
            axes[col].set_xlabel(xl); axes[col].set_ylabel(yl)
            axes[col].set_title(plane_labels[col])
            ims.append(im)
        title_txt = fig.suptitle('', fontsize=13)
        fig.legend(handles=PHASE_LEGEND, loc='upper right',
                   bbox_to_anchor=(0.99, 0.95))
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        def _update(fi):
            for c in range(3):
                ims[c].set_data(all_imgs[fi][c].T)
            title_txt.set_text(f't = {self.frames[fi].time_hours:.1f} h  '
                               f'({self.frames[fi].time_days:.2f} d)')
            return ims

        anim = FuncAnimation(fig, _update, frames=len(self.frames),
                             interval=1000 // fps, blit=False)
        if save:
            fp = self.viz_dir / 'triplane_evolution.gif'
            anim.save(str(fp), writer=PillowWriter(fps=fps))
            plt.close(fig)
            print(f"      → {fp}")
        return anim

    # =================================================================
    # 6.  CELL VISUALIZATIONS  (stress, AR, bridging)
    # =================================================================

    def plot_cell_dashboard(self, frame_index=-1, axis=2,
                            resolution=200, save=True):
        """Three-panel: cells by type / stress / aspect ratio."""
        print("\n  [6a] Cell dashboard …")
        if not self.has_cells:
            print("       No cell data"); return None
        frame = self.frames[frame_index]
        cd = frame.cell_data
        if cd is None or cd.total_cells == 0:
            print("       Empty cell data"); return None

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        cell_diam = self.config.get('cell_properties', {}).get('diameter_um', 20)
        thickness = cell_diam * 2.5
        dims = [0, 1] if axis == 2 else ([0, 2] if axis == 1 else [1, 2])
        mask = np.abs(cd.world_pos[:, axis] - sp) < thickness
        if not np.any(mask):
            print("       No cells near slice"); return None

        cx = cd.world_pos[mask, dims[0]]
        cy = cd.world_pos[mask, dims[1]]
        img = render_phase_map(frame, axis=axis, position=sp, resolution=resolution)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # Panel 1 — type
        _plot_phase(axes[0], img, ext, xl, yl, 'Cells by Type')
        pt = frame.types[cd.parent[mask]]
        for val, clr in [(0, FUNC_COLOR), (1, INERT_COLOR)]:
            m = pt == val
            if np.any(m):
                axes[0].scatter(cx[m], cy[m], s=6, c=clr, alpha=0.7, zorder=3)

        # Panel 2 — stress
        _plot_phase(axes[1], img * 0.2, ext, xl, '', 'Cell Stress (nN)')
        vmax_s = max(np.max(cd.stress_nN[mask]), 1e-6)
        sc2 = axes[1].scatter(cx, cy, c=cd.stress_nN[mask], s=10,
                              cmap='hot', vmin=0, vmax=vmax_s, alpha=0.8, zorder=3)
        plt.colorbar(sc2, ax=axes[1], label='Stress (nN)', shrink=0.8)

        # Panel 3 — AR
        _plot_phase(axes[2], img * 0.2, ext, xl, '', 'Cell Aspect Ratio')
        vmax_a = max(np.max(cd.aspect_ratio[mask]), 1.01)
        sc3 = axes[2].scatter(cx, cy, c=cd.aspect_ratio[mask], s=10,
                              cmap='plasma', vmin=1, vmax=vmax_a, alpha=0.8, zorder=3)
        plt.colorbar(sc3, ax=axes[2], label='Aspect Ratio', shrink=0.8)

        n_br = int(np.sum(cd.is_bridging))
        plt.suptitle(f'Cell Dashboard — t={frame.time_hours:.1f}h — '
                     f'{cd.total_cells} cells, {n_br} bridging',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f'cell_dashboard_{"xyz"[axis]}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def plot_cell_history(self, save=True):
        """Bridging count / stress / AR time-series."""
        print("\n  [6b] Cell history …")
        if not self.history:
            print("       No history"); return None
        h = self.history
        t = np.array([_get_hist(r, 'time_hours') for r in h]) / 24.0

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        axes[0].plot(t, [_get_hist(r, 'n_bridging_cells') for r in h],
                     '-', color='orange', lw=2)
        axes[0].set_xlabel('Time (days)'); axes[0].set_ylabel('Bridging Cells')
        axes[0].set_title('Cell Bridging'); axes[0].grid(True, alpha=0.3)

        axes[1].plot(t, [_get_hist(r, 'mean_stress') for r in h],
                     '-', color='red', lw=2, label='Mean')
        axes[1].plot(t, [_get_hist(r, 'max_stress')  for r in h],
                     '--', color='darkred', lw=1.5, label='Max')
        axes[1].set_xlabel('Time (days)'); axes[1].set_ylabel('Stress (nN)')
        axes[1].set_title('Cell Stress'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        axes[2].plot(t, [_get_hist(r, 'mean_cell_ar', 1) for r in h],
                     '-', color='purple', lw=2)
        axes[2].set_xlabel('Time (days)'); axes[2].set_ylabel('Aspect Ratio')
        axes[2].set_title('Cell Elongation'); axes[2].grid(True, alpha=0.3)

        plt.suptitle('Cell Metrics Evolution', fontsize=13)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / 'cell_history.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def plot_stress_evolution(self, axis=2, n_times=6, resolution=120, save=True):
        """Stress field panels at selected times."""
        print("\n  [6c] Stress evolution …")
        if not self.has_cells or not self.frames:
            print("       No cell data"); return None

        idxs = _pick_frame_indices(len(self.frames), n_times)
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        cell_diam = self.config.get('cell_properties', {}).get('diameter_um', 20)
        thickness = cell_diam * 2.5
        dims = [0, 1] if axis == 2 else ([0, 2] if axis == 1 else [1, 2])

        gmax = 1e-6
        for fi in idxs:
            cd = self.frames[fi].cell_data
            if cd and cd.total_cells > 0: gmax = max(gmax, np.max(cd.stress_nN))

        ncol = len(idxs)
        fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4))
        if ncol == 1: axes = [axes]

        for col, fi in enumerate(idxs):
            fr = self.frames[fi]; ax = axes[col]
            img = render_phase_map(fr, axis=axis, position=sp, resolution=resolution)
            _plot_phase(ax, img * 0.2, ext, xl, yl if col == 0 else '',
                        f't={fr.time_hours:.1f}h')
            if col > 0: ax.set_yticklabels([])
            cd = fr.cell_data
            if cd is None or cd.total_cells == 0: continue
            mask = np.abs(cd.world_pos[:, axis] - sp) < thickness
            if not np.any(mask): continue
            ax.scatter(cd.world_pos[mask, dims[0]], cd.world_pos[mask, dims[1]],
                       c=cd.stress_nN[mask], s=6, cmap='hot',
                       vmin=0, vmax=gmax, alpha=0.8, zorder=3)

        sm = ScalarMappable(cmap='hot', norm=Normalize(0, gmax)); sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Stress (nN)', shrink=0.6, pad=0.02)
        plt.suptitle(f'Cell Stress Evolution ({"XYZ"[axis]}={sp:.0f}μm)', fontsize=13)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f'stress_evo_{"xyz"[axis]}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def plot_ar_evolution(self, axis=2, n_times=6, resolution=120, save=True):
        """Aspect-ratio field panels at selected times."""
        print("\n  [6d] AR evolution …")
        if not self.has_cells or not self.frames:
            print("       No cell data"); return None

        idxs = _pick_frame_indices(len(self.frames), n_times)
        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        cell_diam = self.config.get('cell_properties', {}).get('diameter_um', 20)
        thickness = cell_diam * 2.5
        dims = [0, 1] if axis == 2 else ([0, 2] if axis == 1 else [1, 2])

        gmax = 1.01
        for fi in idxs:
            cd = self.frames[fi].cell_data
            if cd and cd.total_cells > 0: gmax = max(gmax, np.max(cd.aspect_ratio))

        ncol = len(idxs)
        fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4))
        if ncol == 1: axes = [axes]

        for col, fi in enumerate(idxs):
            fr = self.frames[fi]; ax = axes[col]
            img = render_phase_map(fr, axis=axis, position=sp, resolution=resolution)
            _plot_phase(ax, img * 0.2, ext, xl, yl if col == 0 else '',
                        f't={fr.time_hours:.1f}h')
            if col > 0: ax.set_yticklabels([])
            cd = fr.cell_data
            if cd is None or cd.total_cells == 0: continue
            mask = np.abs(cd.world_pos[:, axis] - sp) < thickness
            if not np.any(mask): continue
            ax.scatter(cd.world_pos[mask, dims[0]], cd.world_pos[mask, dims[1]],
                       c=cd.aspect_ratio[mask], s=6, cmap='plasma',
                       vmin=1, vmax=gmax, alpha=0.8, zorder=3)

        sm = ScalarMappable(cmap='plasma', norm=Normalize(1, gmax)); sm.set_array([])
        fig.colorbar(sm, ax=axes, label='Aspect Ratio', shrink=0.6, pad=0.02)
        plt.suptitle(f'Cell AR Evolution ({"XYZ"[axis]}={sp:.0f}μm)', fontsize=13)
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f'ar_evo_{"xyz"[axis]}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def create_cell_animation(self, axis=2, resolution=120, fps=4, save=True):
        """Animated GIF of cell stress over time."""
        print("\n  [6e] Cell stress animation …")
        if not self.has_cells or not HAS_ANIM or len(self.frames) < 2:
            print("       Skipped"); return None

        sp = self.domain[axis] / 2
        ext, xl, yl = _ext(self.domain, axis)
        cell_diam = self.config.get('cell_properties', {}).get('diameter_um', 20)
        thickness = cell_diam * 2.5
        dims = [0, 1] if axis == 2 else ([0, 2] if axis == 1 else [1, 2])

        gmax = 1e-6
        for fr in self.frames:
            if fr.cell_data and fr.cell_data.total_cells > 0:
                gmax = max(gmax, np.max(fr.cell_data.stress_nN))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        im_bg = ax1.imshow(np.zeros((resolution, resolution)).T, origin='lower',
                           cmap=PHASE_CMAP, vmin=0, vmax=2, extent=ext,
                           interpolation='nearest', alpha=0.3)
        scat = ax1.scatter([], [], c=[], s=8, cmap='hot', vmin=0, vmax=gmax, alpha=0.8)
        ax1.set_xlim(ext[0], ext[1]); ax1.set_ylim(ext[2], ext[3])
        ax1.set_xlabel(xl); ax1.set_ylabel(yl); ax1.set_title('Cell Stress')
        plt.colorbar(scat, ax=ax1, label='Stress (nN)', shrink=0.8)

        # Time-series
        t_arr = [fr.time_hours / 24 for fr in self.frames]
        s_arr = []
        for fr in self.frames:
            cd = fr.cell_data
            if cd and cd.total_cells > 0 and np.any(cd.is_bridging):
                s_arr.append(float(np.mean(cd.stress_nN[cd.is_bridging])))
            else:
                s_arr.append(0.0)
        ax2.plot(t_arr, s_arr, 'r-', lw=1.5, alpha=0.4)
        vline = ax2.axvline(0, color='k', lw=2)
        ax2.set_xlabel('Time (days)'); ax2.set_ylabel('Mean Bridge Stress (nN)')
        ax2.set_title('Stress History'); ax2.grid(True, alpha=0.3)
        title_txt = fig.suptitle('', fontsize=13)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        def _update(fi):
            fr = self.frames[fi]
            img = render_phase_map(fr, axis=axis, position=sp, resolution=resolution)
            im_bg.set_data(img.T)
            cd = fr.cell_data
            if cd and cd.total_cells > 0:
                mask = np.abs(cd.world_pos[:, axis] - sp) < thickness
                if np.any(mask):
                    scat.set_offsets(np.c_[cd.world_pos[mask, dims[0]],
                                           cd.world_pos[mask, dims[1]]])
                    scat.set_array(cd.stress_nN[mask])
                else:
                    scat.set_offsets(np.empty((0, 2)))
            else:
                scat.set_offsets(np.empty((0, 2)))
            vline.set_xdata([t_arr[fi]])
            title_txt.set_text(f't = {fr.time_hours:.1f} h')
            return [im_bg, scat, vline]

        anim = FuncAnimation(fig, _update, frames=len(self.frames),
                             interval=1000 // fps, blit=False)
        if save:
            fp = self.viz_dir / f'cell_stress_anim_{"xyz"[axis]}.gif'
            anim.save(str(fp), writer=PillowWriter(fps=fps))
            plt.close(fig)
            print(f"       → {fp}")
        return anim

    # =================================================================
    # 7.  THREE-PLANE / DETAILED / PHASE-EVO STRIP
    # =================================================================

    def plot_three_plane(self, frame_index=-1, resolution=250, save=True):
        print("\n  [7a] Three-plane snapshot …")
        if not self.frames: print("       No frames"); return None
        frame = self.frames[frame_index]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for col, ax_id in enumerate([2, 1, 0]):
            sp = self.domain[ax_id] / 2
            img = render_phase_map(frame, axis=ax_id, position=sp,
                                   resolution=resolution)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(axes[col], img, ext, xl, yl,
                        f'{"XY  XZ  YZ".split()[col]} ({"XYZ"[ax_id]}={sp:.0f}μm)')
        fig.legend(handles=PHASE_LEGEND, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        plt.suptitle(f't = {frame.time_hours:.1f}h ({frame.time_days:.2f}d)')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index == -1 else f"f{frame_index}"
            fp = self.viz_dir / f'three_plane_{lab}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def plot_phase_evolution(self, axis=2, n_times=6, resolution=200, save=True):
        print("\n  [7b] Phase evolution strip …")
        if not self.frames: print("       No frames"); return None
        idxs = _pick_frame_indices(len(self.frames), n_times)
        sp = self.domain[axis] / 2; ext, xl, yl = _ext(self.domain, axis)
        ncol = len(idxs)
        fig, axes = plt.subplots(1, ncol, figsize=(3.5 * ncol, 4))
        if ncol == 1: axes = [axes]
        for col, fi in enumerate(idxs):
            fr = self.frames[fi]
            img = render_phase_map(fr, axis=axis, position=sp, resolution=resolution)
            _plot_phase(axes[col], img, ext, xl,
                        yl if col == 0 else '', f't={fr.time_hours:.1f}h')
            if col > 0: axes[col].set_yticklabels([])
        fig.legend(handles=PHASE_LEGEND, loc='upper right', bbox_to_anchor=(0.99, 0.99))
        plt.suptitle(f'Phase Evolution ({"XYZ"[axis]}={sp:.0f}μm)')
        plt.tight_layout()
        if save:
            fp = self.viz_dir / f'phase_evo_{"xyz"[axis]}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    def plot_detailed_frame(self, frame_index=-1, save=True):
        print("\n  [7c] Detailed dashboard …")
        if not self.frames: print("       No frames"); return None
        frame = self.frames[frame_index]
        fig = plt.figure(figsize=(16, 12)); gs = gridspec.GridSpec(2, 3)

        for col, ax_id in enumerate([2, 1, 0]):
            ax = fig.add_subplot(gs[0, col])
            sp = self.domain[ax_id] / 2
            img = render_phase_map(frame, axis=ax_id, position=sp, resolution=200)
            ext, xl, yl = _ext(self.domain, ax_id)
            _plot_phase(ax, img, ext, xl, yl, f'{"XY  XZ  YZ".split()[col]}')

        # Size histogram
        ax = fig.add_subplot(gs[1, 0])
        fr_r = [s.eq_radius for i, s in enumerate(frame.shapes) if frame.functional_mask[i]]
        ir_r = [s.eq_radius for i, s in enumerate(frame.shapes) if frame.inert_mask[i]]
        if fr_r: ax.hist(fr_r, bins=15, alpha=0.6, color='red',  label='Functional')
        if ir_r: ax.hist(ir_r, bins=15, alpha=0.6, color='blue', label='Inert')
        ax.set_xlabel('Equiv. Radius (μm)'); ax.set_ylabel('Count')
        ax.set_title('Size Distribution'); ax.legend()

        # Coordination time-series
        ax = fig.add_subplot(gs[1, 1])
        if self.history:
            t = np.array([_get_hist(r, 'time_hours') for r in self.history]) / 24
            ax.plot(t, [_get_hist(r, 'mean_coordination') for r in self.history],
                    'g-', lw=2)
        ax.set_xlabel('Time (days)'); ax.set_ylabel('Mean Coordination')
        ax.set_title('Coordination'); ax.grid(True, alpha=0.3)

        # Stats box
        ax = fig.add_subplot(gs[1, 2]); ax.axis('off')
        n_f = int(np.sum(frame.functional_mask))
        n_i = int(np.sum(frame.inert_mask))
        stats = (f"t = {frame.time_hours:.1f}h ({frame.time_days:.2f}d)\n\n"
                 f"Func: {n_f}   Inert: {n_i}   Total: {frame.n_granules}\n\n"
                 f"Mean R func:  {np.mean(fr_r):.1f} μm\n"
                 f"Mean R inert: {np.mean(ir_r):.1f} μm\n\n"
                 f"Total cells: {int(np.sum(frame.n_cells))}\n"
                 f"Domain: {self.domain[0]:.0f}³ μm")
        ax.text(0.1, 0.9, stats, transform=ax.transAxes, fontsize=11,
                va='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'Detailed — t={frame.time_days:.2f}d',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        if save:
            lab = "final" if frame_index == -1 else f"f{frame_index}"
            fp = self.viz_dir / f'detailed_{lab}.png'
            plt.savefig(fp, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"       → {fp}")
        return fig

    # =================================================================
    # MASTER
    # =================================================================

    def create_all(self):
        """Generate every visualisation and save to <folder>/visualizations/."""
        print("\n" + "=" * 65)
        print("  POST-PROCESSOR — Generating All Visualizations")
        print("=" * 65)

        self.plot_zstack_evolution()         # 1
        self.plot_metrics()                  # 2
        self.plot_phase_snapshots()          # 3
        self.create_3d_rotating_gif()        # 4
        self.create_triplane_animation()     # 5

        if self.has_cells:                   # 6
            self.plot_cell_dashboard()
            self.plot_cell_history()
            self.plot_stress_evolution()
            self.plot_ar_evolution()
            self.create_cell_animation()

        self.plot_three_plane()              # 7
        self.plot_phase_evolution()
        self.plot_detailed_frame()

        print("\n" + "=" * 65)
        print(f"  DONE — all outputs in  {self.viz_dir}")
        print("=" * 65)


# =========================================================================
# CLI ENTRY POINT
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Post-process granular hydrogel simulation outputs')
    parser.add_argument('-i', '--input', type=str, default=None,
                        help='Path to folder with frame JSONs + history JSON')
    parser.add_argument('--all',        action='store_true', help='All plots (default)')
    parser.add_argument('--zstack',     action='store_true')
    parser.add_argument('--metrics',    action='store_true')
    parser.add_argument('--snapshots',  action='store_true')
    parser.add_argument('--rotate3d',   action='store_true')
    parser.add_argument('--triplane',   action='store_true')
    parser.add_argument('--cells',      action='store_true')
    parser.add_argument('--cell-hist',  action='store_true')
    parser.add_argument('--stress-evo', action='store_true')
    parser.add_argument('--ar-evo',     action='store_true')
    parser.add_argument('--cell-anim',  action='store_true')
    parser.add_argument('--three-plane',action='store_true')
    parser.add_argument('--detailed',   action='store_true')
    parser.add_argument('--phase-evo',  action='store_true')
    parser.add_argument('--axis', type=int, default=2, help='Slice axis 0/1/2')
    parser.add_argument('--fps',  type=int, default=4,  help='Animation FPS')
    parser.add_argument('--res3d',type=int, default=50, help='3D voxel resolution')
    args = parser.parse_args()

    # ── Get folder ──
    if args.input:
        folder = Path(args.input)
    else:
        folder = select_folder()
    if folder is None or not folder.is_dir():
        print("  No valid folder selected."); return

    print(f"\n  Input: {folder}")
    viz = Visualizer(folder)

    # ── Dispatch ──
    explicit = any([args.zstack, args.metrics, args.snapshots, args.rotate3d,
                    args.triplane, args.cells, args.cell_hist, args.stress_evo,
                    args.ar_evo, args.cell_anim, args.three_plane,
                    args.detailed, args.phase_evo])

    if args.all or not explicit:
        viz.create_all(); return

    if args.zstack:      viz.plot_zstack_evolution()
    if args.metrics:     viz.plot_metrics()
    if args.snapshots:   viz.plot_phase_snapshots()
    if args.rotate3d:    viz.create_3d_rotating_gif(resolution=args.res3d)
    if args.triplane:    viz.create_triplane_animation(fps=args.fps)
    if args.cells:       viz.plot_cell_dashboard(axis=args.axis)
    if args.cell_hist:   viz.plot_cell_history()
    if args.stress_evo:  viz.plot_stress_evolution(axis=args.axis)
    if args.ar_evo:      viz.plot_ar_evolution(axis=args.axis)
    if args.cell_anim:   viz.create_cell_animation(axis=args.axis, fps=args.fps)
    if args.three_plane: viz.plot_three_plane()
    if args.detailed:    viz.plot_detailed_frame()
    if args.phase_evo:   viz.plot_phase_evolution(axis=args.axis)


if __name__ == "__main__":
    main()