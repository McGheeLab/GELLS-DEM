
"""
DEM Post-Processing (new_dem.py + legacy DEM JSON frame outputs)

Generates:
1) Granule "slice-bin" evolution plots (x-y cross-section, binned by z; jet colormap).
2) Plot styling consistent with new_dem_0.py.
3) Final-phase 3D voxel visualizations (functional/inert/void) rotating 360° → GIF.
4) Time-evolution GIF of mid-volume cross-sections (XY @ z=mid, XZ @ y=mid, YZ @ x=mid).

Works with:
- new_dem.py frame JSONs (keys: domain, positions, orientations, shapes, types, n_cells, cell_data)  [see save_frame] 
- dem_sim_speed_viz_shape_cells.py frame JSONs (keys: sim_shapes/true_shapes) [see save_frame]
- 2D/legacy outputs: if positions are 2D or z-extent is missing, falls back to XY-only visuals.

Usage:
    1) Edit USER CONFIG at the top of this file.
    2) Run: python new_new_dem_postprocess.py

"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import numpy as np


# ===========================
# USER CONFIG (EDIT THESE)
# ===========================
# Leave DATA_DIR as "" to pick a folder via GUI (macOS/Windows/Linux) or terminal prompt fallback.
DATA_DIR: str = ""
OUT_SUBDIR: str = "postprocess"

# Rendering / discretization controls
RENDER_RES: int = 160       # resolution for 2D rasterized slice plots
VOX_RES: int = 72           # voxel grid resolution per axis for 3D phase rendering
N_Z_SLICES: int = 10        # number of z-bins used in z-coded granule evolution
FPS_TIME: int = 10          # fps for time-evolution GIFs
FPS_ROTATE: int = 18        # fps for the 360° rotation GIF

# GIF duration knobs (advanced)
ROTATE_FRAMES: int = 72     # number of frames over 360° (more = smoother)

# ===========================
# END USER CONFIG
# ===========================

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio


# =============================================================================
# Styling (match new_dem_0.py plots)
# =============================================================================

FUNC_COLOR = "orangered"
INERT_COLOR = "steelblue"


def _pick_data_dir() -> str:
    """Return a folder path. Uses a GUI chooser when available; otherwise prompts in terminal."""
    if DATA_DIR and str(DATA_DIR).strip():
        return str(DATA_DIR).strip()

    # Try Tkinter folder picker
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes("-topmost", 1)
        chosen = filedialog.askdirectory(title="Select DEM simulation output folder")
        root.destroy()
        if chosen:
            return chosen
    except Exception:
        pass

    # Terminal fallback
    while True:
        chosen = input("Enter path to DEM simulation output folder: ").strip()
        if chosen:
            return chosen


def fig_to_rgb_array(fig):
    """Backend-agnostic Matplotlib figure capture → (H, W, 3) uint8 RGB."""
    fig.canvas.draw()
    try:
        rgba = np.asarray(fig.canvas.buffer_rgba())  # (H, W, 4)
        rgb = np.array(rgba, copy=True)[..., :3]
        return rgb.astype(np.uint8, copy=False)
    except Exception:
        # Fallback for backends that only expose ARGB
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        rgb = argb[..., [1, 2, 3]]
        return rgb.astype(np.uint8, copy=False)


def apply_style():
    """Small, consistent aesthetic (inspired by new_dem_0.py visualization defaults)."""
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


# =============================================================================
# Loading frames (new_dem + older dem_sim_speed_viz_shape_cells)
# =============================================================================

@dataclass
class Frame:
    time_hours: float
    dom: np.ndarray          # (3,) or (2,)
    pos: np.ndarray          # (N,3) or (N,2)
    quat: Optional[np.ndarray]   # (N,4) or None
    sp: np.ndarray           # (N,4) -> (a,b,c,n) or (N,3) radii fallback
    types: np.ndarray        # (N,)
    meta: Dict


def _as_np(x, dtype=float):
    return np.asarray(x, dtype=dtype)


def _pick_shape_list(data: Dict) -> List[Dict]:
    # new_dem.py uses "shapes"  fileciteturn1file4L7-L16
    # dem_sim_speed_viz_shape_cells.py uses "sim_shapes"/"true_shapes" fileciteturn1file1L1-L11
    if "shapes" in data:
        return data["shapes"]
    if "sim_shapes" in data:
        return data["sim_shapes"]
    if "true_shapes" in data:
        return data["true_shapes"]
    raise KeyError("No shapes field found (expected shapes/sim_shapes/true_shapes).")


def _shape_dicts_to_sp(shape_dicts: List[Dict], pos_dim: int) -> np.ndarray:
    """
    Convert shape dict(s) into a numeric 'sp' array.
    Expected for superellipsoids: a,b,c,n (4 params).
    Fallbacks:
      - If only radius exists: treat as sphere/ball with a=b=c=r, n=2
      - If 2D: treat as disk with c small (for slicing), n=2
    """
    sp = np.zeros((len(shape_dicts), 4), dtype=float)
    for i, d in enumerate(shape_dicts):
        # Try common keys
        a = d.get("a", d.get("ax", d.get("rx", d.get("r", None))))
        b = d.get("b", d.get("ay", d.get("ry", d.get("r", None))))
        c = d.get("c", d.get("az", d.get("rz", d.get("r", None))))
        n = d.get("n", d.get("p", d.get("expo", 2.0)))
        if a is None or b is None:
            r = d.get("r", d.get("radius", 10.0))
            a = b = c = r
            n = 2.0
        if c is None:
            # 2D-ish shapes: give small thickness for voxel/slice logic
            c = min(a, b) * 0.25
        sp[i] = [float(a), float(b), float(c), float(n)]
    # If 2D simulation, just keep sp but c will be small.
    if pos_dim == 2:
        sp[:, 2] = np.maximum(sp[:, 2], 1.0)
    return sp


def load_frames(out_dir: Path) -> List[Frame]:
    out_dir = Path(out_dir)
    frame_files = sorted(out_dir.glob("*_frame_*.json"))
    if not frame_files:
        raise FileNotFoundError(f"No *_frame_*.json files found in: {out_dir}")

    frames: List[Frame] = []
    for fn in frame_files:
        with open(fn, "r") as f:
            data = json.load(f)

        dom = _as_np(data.get("domain", data.get("dom", [1, 1, 1])), float)
        pos = _as_np(data.get("positions", data.get("pos")), float)
        if pos.ndim != 2:
            raise ValueError(f"positions must be 2D array; got shape {pos.shape} in {fn}")

        # Normalize 2D → 3D where helpful
        pos_dim = pos.shape[1]
        if pos_dim == 2:
            dom = dom[:2]
        elif dom.size == 2:
            # 3D positions but 2D domain? extend
            dom = np.array([dom[0], dom[1], float(np.max(pos[:, 2]) + 1.0)])

        quat = data.get("orientations", None)
        if quat is not None:
            quat = _as_np(quat, float)
            # some outputs might be nested objects; keep only lists
            if quat.ndim != 2 or quat.shape[1] != 4:
                quat = None

        shape_dicts = _pick_shape_list(data)
        sp = _shape_dicts_to_sp(shape_dicts, pos_dim=pos.shape[1])

        types = _as_np(data.get("types", np.zeros(len(pos))), int)

        t = float(data.get("time_hours", data.get("time", 0.0)))
        frames.append(Frame(time_hours=t, dom=dom, pos=pos, quat=quat, sp=sp, types=types, meta={"file": str(fn)}))

    # sort by time (filenames already usually ordered, but be safe)
    frames.sort(key=lambda fr: fr.time_hours)
    return frames


# =============================================================================
# Geometry: quaternion rotations + superellipsoid inside test
# =============================================================================

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Quaternion (w,x,y,z) or (x,y,z,w) robustness: infer by magnitude of first element."""
    q = np.asarray(q, dtype=float).copy()
    if q.shape != (4,):
        raise ValueError("quat_to_rot expects shape (4,)")

    # Heuristic: new_dem stores q.q.tolist() (likely w,x,y,z) fileciteturn1file4L11-L15
    # If |w| is smallest, assume last element is w.
    if abs(q[0]) < abs(q[3]):
        x, y, z, w = q
    else:
        w, x, y, z = q

    n = math.sqrt(w*w + x*x + y*y + z*z) + 1e-12
    w, x, y, z = w/n, x/n, y/n, z/n

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=float)


def point_inside_superellipsoid(pt: np.ndarray, center: np.ndarray, sp: np.ndarray, R: Optional[np.ndarray]) -> bool:
    """
    sp = (a,b,c,n); equation: |x/a|^n + |y/b|^n + |z/c|^n <= 1
    where (x,y,z) are in body frame.
    """
    a, b, c, n = float(sp[0]), float(sp[1]), float(sp[2]), float(sp[3])
    d = pt - center
    if d.size == 2:
        # 2D: ignore z
        x, y = d
        z = 0.0
    else:
        x, y, z = d
    if R is not None and d.size == 3:
        d_b = R.T @ d
        x, y, z = d_b[0], d_b[1], d_b[2]

    # Avoid div0
    a = max(a, 1e-6); b = max(b, 1e-6); c = max(c, 1e-6)
    v = (abs(x/a)**n) + (abs(y/b)**n) + (abs(z/c)**n)
    return v <= 1.0


def bounding_r(sp: np.ndarray) -> float:
    return float(max(sp[0], sp[1], sp[2]))


# =============================================================================
# Rendering helpers
# =============================================================================

def render_mid_slices(frame: Frame, res: int = 160) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Render mid-plane slices for XY(z=mid), XZ(y=mid), YZ(x=mid) as integer label images:
        0 void, 1 functional, 2 inert
    If 2D, returns only XY and dummy for others.
    """
    pos = frame.pos
    dom = frame.dom
    sp = frame.sp
    types = frame.types
    q = frame.quat

    if pos.shape[1] == 2 or dom.size == 2:
        # 2D: just XY occupancy
        return render_plane(frame, axis=2, frac=0.0, res=res), None, None

    xy = render_plane(frame, axis=2, frac=0.5, res=res)
    xz = render_plane(frame, axis=1, frac=0.5, res=res)
    yz = render_plane(frame, axis=0, frac=0.5, res=res)
    return xy, xz, yz


def render_plane(frame: Frame, axis: int, frac: float, res: int) -> np.ndarray:
    """
    Rasterize a phase map on a plane at fixed coordinate along 'axis'.
    Returns label image with 0 void, 1 functional, 2 inert.
    Mirrors the logic of new_dem.render_slice() for inside tests fileciteturn1file6L14-L45
    but keeps it pure-Python for post-processing flexibility.
    """
    pos = frame.pos
    dom = frame.dom
    sp = frame.sp
    types = frame.types
    quat = frame.quat

    # Determine plane dimensions
    if axis == 2:   # z fixed → x-y
        d1, d2 = 0, 1
    elif axis == 1: # y fixed → x-z
        d1, d2 = 0, 2
    else:           # x fixed → y-z
        d1, d2 = 1, 2

    slice_pos = dom[axis] * frac
    dx = dom[d1] / res
    dy = dom[d2] / res

    img = np.zeros((res, res), dtype=np.uint8)

    # Precompute rotation matrices (optional)
    Rmats = None
    if quat is not None and pos.shape[1] == 3:
        Rmats = np.stack([quat_to_rot(quat[i]) for i in range(len(pos))], axis=0)

    for ix in range(res):
        p1 = (ix + 0.5) * dx
        for iy in range(res):
            p2 = (iy + 0.5) * dy
            pt = np.zeros(3, dtype=float)
            pt[d1] = p1
            pt[d2] = p2
            pt[axis] = slice_pos

            # Search granules (bounding reject first)
            for g in range(len(pos)):
                br = bounding_r(sp[g])
                if abs(pos[g, axis] - slice_pos) > br:
                    continue
                dd = pt - pos[g]
                if float(dd @ dd) > br * br:
                    continue
                R = Rmats[g] if Rmats is not None else None
                if point_inside_superellipsoid(pt, pos[g], sp[g], R):
                    img[ix, iy] = 1 if types[g] == 0 else 2
                    break
    return img


def labels_to_rgb(img: np.ndarray) -> np.ndarray:
    """Map labels to RGB like new_dem_0.py composite: R=functional, G=void, B=inert. fileciteturn1file3L10-L17"""
    # Start with void as green.
    rgb = np.zeros(img.shape + (3,), dtype=float)
    void = (img == 0)
    func = (img == 1)
    inert = (img == 2)
    rgb[void, 1] = 1.0
    rgb[func, 0] = 1.0
    rgb[inert, 2] = 1.0
    return rgb


# =============================================================================
# Plots / GIF generation
# =============================================================================

def make_z_binned_slices_evolution_gif(frames: List[Frame], out_dir: Path, nslices: int = 10, fps: int = 10):
    """
    For each timepoint, bin granules by z and plot x-y circles per z-slice.
    Color code slice index by a jet colormap to highlight z-structure evolution.
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    cmap = plt.get_cmap("jet")
    for fr in frames:
        fig, ax = plt.subplots(1, 1, figsize=(6.2, 6.0))
        ax.set_aspect("equal")

        dom = fr.dom
        pos = fr.pos
        sp = fr.sp
        types = fr.types

        if pos.shape[1] == 2 or dom.size == 2:
            # 2D fallback: color by type only
            for i in range(len(pos)):
                col = FUNC_COLOR if types[i] == 0 else INERT_COLOR
                r = float(max(sp[i][0], sp[i][1]))  # disk-ish
                ax.add_patch(Circle((pos[i, 0], pos[i, 1]), r, fc=col, ec="k", lw=0.25, alpha=0.65))
            ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[1])
            ax.set_title(f"Granules (2D)  t={fr.time_hours:.2f} h")
        else:
            z = pos[:, 2]
            zmin, zmax = 0.0, float(dom[2])
            edges = np.linspace(zmin, zmax, nslices + 1)
            for k in range(nslices):
                z0, z1 = edges[k], edges[k+1]
                mid = 0.5 * (z0 + z1)
                sel = (z >= z0) & (z < z1)
                if not np.any(sel):
                    continue
                col = cmap((mid - zmin) / max(1e-9, (zmax - zmin)))
                # Within each slice, draw functional and inert with different alpha
                for i in np.where(sel)[0]:
                    r = float(max(sp[i][0], sp[i][1], sp[i][2]))  # coarse "footprint"
                    alpha = 0.70 if types[i] == 0 else 0.45
                    ax.add_patch(Circle((pos[i, 0], pos[i, 1]), r, fc=col, ec="k", lw=0.2, alpha=alpha))
            ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[1])
            ax.set_title(f"Z-binned XY cross-section (jet by z)  t={fr.time_hours:.2f} h")
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=zmin, vmax=zmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("z (µm)")

        ax.set_xlabel("x (µm)"); ax.set_ylabel("y (µm)")
        fig.tight_layout()

        # Convert fig to image
        img = fig_to_rgb_array(fig)
        images.append(img)
        plt.close(fig)

    gif_path = out_dir / "granule_zslice_evolution.gif"
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path


def make_midplane_time_gif(frames: List[Frame], out_dir: Path, res: int = 160, fps: int = 10):
    """Animated GIF of XY, XZ, YZ mid-planes across time."""
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = []
    for fr in frames:
        xy, xz, yz = render_mid_slices(fr, res=res)

        fig = plt.figure(figsize=(12.0, 4.0))
        if xy is not None:
            ax1 = fig.add_subplot(1, 3, 1)
            ax1.imshow(labels_to_rgb(xy).transpose(1, 0, 2), origin="lower")
            ax1.set_title(f"XY @ z=mid   t={fr.time_hours:.2f} h")
            ax1.set_axis_off()

        if xz is not None:
            ax2 = fig.add_subplot(1, 3, 2)
            ax2.imshow(labels_to_rgb(xz).transpose(1, 0, 2), origin="lower")
            ax2.set_title("XZ @ y=mid")
            ax2.set_axis_off()

        if yz is not None:
            ax3 = fig.add_subplot(1, 3, 3)
            ax3.imshow(labels_to_rgb(yz).transpose(1, 0, 2), origin="lower")
            ax3.set_title("YZ @ x=mid")
            ax3.set_axis_off()

        fig.suptitle("Mid-plane cross-sections (R=functional, G=void, B=inert)", y=1.02)
        fig.tight_layout()

        img = fig_to_rgb_array(fig)
        images.append(img)
        plt.close(fig)

    gif_path = out_dir / "midplane_xyz_time_evolution.gif"
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path


def voxelize(frame: Frame, res: int = 72) -> np.ndarray:
    """
    Voxelize volume into labels:
        0 void, 1 functional, 2 inert
    Note: O(N * affected_voxels) per frame; keep res moderate (60–90).
    """
    pos = frame.pos
    dom = frame.dom
    sp = frame.sp
    types = frame.types
    quat = frame.quat

    if pos.shape[1] == 2 or dom.size == 2:
        # 2D fallback: make a thin volume
        dom = np.array([dom[0], dom[1], max(1.0, float(np.max(sp[:,2])) * 2)], dtype=float)
        pos = np.column_stack([pos[:, 0], pos[:, 1], np.zeros(len(pos))])

    nx = ny = nz = int(res)
    dx = dom[0] / nx
    dy = dom[1] / ny
    dz = dom[2] / nz

    vol = np.zeros((nx, ny, nz), dtype=np.uint8)

    # Precompute rotations
    Rmats = None
    if quat is not None and pos.shape[1] == 3:
        Rmats = np.stack([quat_to_rot(quat[i]) for i in range(len(pos))], axis=0)

    # For each granule, stamp into local bounding box
    for g in range(len(pos)):
        br = bounding_r(sp[g])
        cx, cy, cz = pos[g]
        ix0 = max(0, int((cx - br) / dx) - 1)
        ix1 = min(nx - 1, int((cx + br) / dx) + 1)
        iy0 = max(0, int((cy - br) / dy) - 1)
        iy1 = min(ny - 1, int((cy + br) / dy) + 1)
        iz0 = max(0, int((cz - br) / dz) - 1)
        iz1 = min(nz - 1, int((cz + br) / dz) + 1)

        R = Rmats[g] if Rmats is not None else None
        lab = 1 if types[g] == 0 else 2

        for ix in range(ix0, ix1 + 1):
            x = (ix + 0.5) * dx
            for iy in range(iy0, iy1 + 1):
                y = (iy + 0.5) * dy
                for iz in range(iz0, iz1 + 1):
                    if vol[ix, iy, iz] != 0:
                        continue  # already occupied
                    z = (iz + 0.5) * dz
                    pt = np.array([x, y, z], dtype=float)
                    # quick bounding sphere
                    dd = pt - pos[g]
                    if float(dd @ dd) > br * br:
                        continue
                    if point_inside_superellipsoid(pt, pos[g], sp[g], R):
                        vol[ix, iy, iz] = lab
    return vol


def make_final_phase_rotation_gif(final_frame: Frame, out_dir: Path, vox_res: int = 72, fps: int = 18):
    """
    Create a rotating 3D isometric-view GIF with 3 subplots:
      - functional mask
      - inert mask
      - void mask
    """
    apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if final_frame.pos.shape[1] == 2 or final_frame.dom.size == 2:
        # Can't do meaningful 3D rotation; still create a single XY figure
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        dom = final_frame.dom
        for i in range(len(final_frame.pos)):
            col = FUNC_COLOR if final_frame.types[i] == 0 else INERT_COLOR
            r = float(max(final_frame.sp[i][0], final_frame.sp[i][1]))
            ax.add_patch(Circle((final_frame.pos[i, 0], final_frame.pos[i, 1]), r, fc=col, ec="k", lw=0.25, alpha=0.65))
        ax.set_xlim(0, dom[0]); ax.set_ylim(0, dom[1]); ax.set_aspect("equal")
        ax.set_title("Final state (2D fallback)")
        fig.tight_layout()
        png_path = out_dir / "final_state_2d.png"
        fig.savefig(png_path)
        plt.close(fig)
        return png_path

    vol = voxelize(final_frame, res=vox_res)
    func = (vol == 1)
    inert = (vol == 2)
    void = (vol == 0)

    # Downsample void for display (voxels are dense)
    # Keep only boundary-ish void voxels via simple erosion-like neighbor test
    if void.sum() > func.sum() + inert.sum():
        v = void.copy()
        # neighbor count test
        neigh = np.zeros_like(v, dtype=np.uint8)
        neigh[:-1,:,:] += v[1:,:,:]
        neigh[1:,:,:] += v[:-1,:,:]
        neigh[:,:-1,:] += v[:,1:,:]
        neigh[:,1:,:] += v[:,:-1,:]
        neigh[:,:,:-1] += v[:,:,1:]
        neigh[:,:,1:] += v[:,:,:-1]
        void = v & (neigh < 6)

    images = []

    import matplotlib as mpl
    for angle in np.linspace(0, 360, ROTATE_FRAMES, endpoint=False):
        fig = plt.figure(figsize=(13.5, 4.8))
        axs = [
            fig.add_subplot(1, 3, 1, projection="3d"),
            fig.add_subplot(1, 3, 2, projection="3d"),
            fig.add_subplot(1, 3, 3, projection="3d"),
        ]

        # Configure all axes similarly
        for ax in axs:
            ax.set_box_aspect((1, 1, 1))
            ax.view_init(elev=25, azim=angle)  # isometric-ish
            ax.set_axis_off()

        # Plot functional
        axs[0].voxels(func, facecolors=mpl.colors.to_rgba(FUNC_COLOR, 0.85), edgecolor="k", linewidth=0.05)
        axs[0].set_title("Functional")

        # Plot inert
        axs[1].voxels(inert, facecolors=mpl.colors.to_rgba(INERT_COLOR, 0.75), edgecolor="k", linewidth=0.05)
        axs[1].set_title("Inert")

        # Plot void (semi-transparent)
        axs[2].voxels(void, facecolors=mpl.colors.to_rgba("green", 0.08), edgecolor=None)
        axs[2].set_title("Void")

        fig.suptitle(f"Final phases (voxelized)   t={final_frame.time_hours:.2f} h", y=0.98)
        fig.tight_layout()

        img = fig_to_rgb_array(fig)
        images.append(img)
        plt.close(fig)

    gif_path = out_dir / "final_phases_rotate.gif"
    imageio.mimsave(gif_path, images, fps=fps)
    return gif_path


# =============================================================================
# CLI
# =============================================================================


def main():
    sim_dir = _pick_data_dir()
    sim_dir = os.path.abspath(sim_dir)

    frames = load_frames(sim_dir)
    if not frames:
        raise RuntimeError(f"No frame JSONs found in: {sim_dir}")

    out_dir = Path(sim_dir) / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[postprocess] Loaded {len(frames)} frames from {sim_dir}")
    print(f"[postprocess] Writing outputs to {out_dir}")

    p1 = make_z_binned_slices_evolution_gif(frames, out_dir, nslices=N_Z_SLICES, fps=FPS_TIME)
    print(f"[postprocess] Wrote: {p1}")

    p2 = make_midplane_time_gif(frames, out_dir, res=RENDER_RES, fps=FPS_TIME)
    print(f"[postprocess] Wrote: {p2}")

    p3 = make_final_phase_rotation_gif(frames[-1], out_dir, vox_res=VOX_RES, fps=FPS_ROTATE)
    print(f"[postprocess] Wrote: {p3}")

    print("[postprocess] Done.")


if __name__ == "__main__":
    main()

