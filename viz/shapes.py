#!/usr/bin/env python3
"""
viz_shapes.py — Display all granule shapes available in GELLS-DEM.

Shows 2D superellipses and 3D superellipsoids with varying aspect ratios
and blockiness exponents, annotated with the parameters that create them.

Usage:
    python viz_shapes.py                   # show both 2D and 3D panels
    python viz_shapes.py --mode 2D         # 2D superellipses only
    python viz_shapes.py --mode 3D         # 3D superellipsoids only
    python viz_shapes.py -o shapes.png     # save to file instead of displaying
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── 2D superellipse helpers ──────────────────────────────────────────

def superellipse_points(a, b, n, num_pts=256):
    """Return (x, y) arrays tracing a superellipse |x/a|^n + |y/b|^n = 1."""
    t = np.linspace(0, 2 * np.pi, num_pts, endpoint=True)
    e = 2.0 / n
    ct, st = np.cos(t), np.sin(t)
    x = a * np.sign(ct) * np.abs(ct) ** e
    y = b * np.sign(st) * np.abs(st) ** e
    return x, y


# ── 3D superellipsoid helpers ────────────────────────────────────────

def _sgnpow(x, p):
    return np.sign(x) * np.abs(x) ** p


def superellipsoid_mesh(a, b, c, n1, n2, n_pts=40):
    """Parametric surface mesh for (|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1."""
    eta = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
    omega = np.linspace(-np.pi, np.pi, n_pts)
    E, O = np.meshgrid(eta, omega, indexing='ij')
    e1 = 2.0 / n1
    e2 = 2.0 / n2
    X = a * _sgnpow(np.cos(E), e2) * _sgnpow(np.cos(O), e1)
    Y = b * _sgnpow(np.cos(E), e2) * _sgnpow(np.sin(O), e1)
    Z = c * _sgnpow(np.sin(E), e2)
    return X, Y, Z


# ── 2D gallery ───────────────────────────────────────────────────────

def plot_2d_gallery(fig, grid_spec):
    """Draw a grid of 2D superellipses varying aspect ratio and blockiness."""
    # Parameter ranges
    aspect_ratios = [1.0, 1.3, 1.8]
    blockiness_vals = [1.5, 2.0, 3.0, 5.0, 10.0]

    nrows = len(aspect_ratios)
    ncols = len(blockiness_vals)

    inner = grid_spec.subgridspec(nrows + 1, ncols, hspace=0.35, wspace=0.30,
                                  height_ratios=[0.12] + [1] * nrows)

    # Title row
    ax_title = fig.add_subplot(inner[0, :])
    ax_title.set_axis_off()
    ax_title.text(0.5, 0.3, '2D Superellipses:  $|x/a|^n + |y/b|^n = 1$',
                  ha='center', va='center', fontsize=14, fontweight='bold')

    R = 40.0  # reference equivalent radius (µm)

    for row, ar in enumerate(aspect_ratios):
        for col, n in enumerate(blockiness_vals):
            ax = fig.add_subplot(inner[row + 1, col])

            a = R * np.sqrt(ar)
            b = R / np.sqrt(ar)
            x, y = superellipse_points(a, b, n)

            ax.fill(x, y, alpha=0.25, color='C0')
            ax.plot(x, y, color='C0', linewidth=1.5)

            # Reference circle
            if not (ar == 1.0 and n == 2.0):
                xc, yc = superellipse_points(R, R, 2.0)
                ax.plot(xc, yc, color='grey', linewidth=0.7, linestyle='--', alpha=0.5)

            lim = max(a, b) * 1.25
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_aspect('equal')
            ax.tick_params(labelsize=6)
            ax.set_xticks([])
            ax.set_yticks([])

            # Labels
            label = f'a={a:.0f}  b={b:.0f}\nn={n}'
            ax.set_xlabel(label, fontsize=7, labelpad=2)

            if col == 0:
                ax.set_ylabel(f'AR = {ar}', fontsize=9, fontweight='bold')
            if row == 0:
                ax.set_title(f'n = {n}', fontsize=9, fontweight='bold')


# ── 3D gallery ───────────────────────────────────────────────────────

def plot_3d_gallery(fig, grid_spec):
    """Draw a grid of 3D superellipsoids varying aspect ratio and blockiness."""
    # Show combos of (aspect_ratio a/b, blockiness n1, c/a ratio, n2)
    shapes = [
        # (label, a/b, c/a, n1, n2)
        ('Sphere',              1.0, 1.0, 2.0, 2.0),
        ('Oblate\n(tablet)',    1.0, 0.5, 2.0, 2.0),
        ('Prolate\n(capsule)',  1.0, 1.8, 2.0, 2.0),
        ('Blocky sphere\n(cube-like)', 1.0, 1.0, 4.0, 4.0),
        ('Ellipsoid',           1.5, 1.0, 2.0, 2.0),
        ('Blocky ellipsoid',    1.5, 1.0, 3.0, 3.0),
        ('Round cylinder',      1.0, 2.0, 2.0, 5.0),
        ('Blocky tablet',       1.0, 0.5, 4.0, 4.0),
        ('Super-blocky',        1.0, 1.0, 8.0, 8.0),
        ('Mixed blockiness',    1.3, 1.0, 2.5, 4.0),
        ('Flat blocky',         1.3, 0.6, 3.5, 3.5),
        ('Elongated round',     1.0, 2.0, 2.0, 2.0),
    ]

    nrows, ncols = 3, 4
    inner = grid_spec.subgridspec(nrows + 1, ncols, hspace=0.25, wspace=0.15,
                                  height_ratios=[0.10] + [1] * nrows)

    ax_title = fig.add_subplot(inner[0, :])
    ax_title.set_axis_off()
    ax_title.text(0.5, 0.3,
                  r'3D Superellipsoids:  $(|x/a|^{n_1}+|y/b|^{n_1})^{n_2/n_1}+|z/c|^{n_2}=1$',
                  ha='center', va='center', fontsize=14, fontweight='bold')

    R = 40.0

    for idx, (name, ar_ab, ar_ca, n1, n2) in enumerate(shapes):
        row = idx // ncols
        col = idx % ncols
        ax = fig.add_subplot(inner[row + 1, col], projection='3d')

        a = R * np.sqrt(ar_ab)
        b = R / np.sqrt(ar_ab)
        c = a * ar_ca

        X, Y, Z = superellipsoid_mesh(a, b, c, n1, n2, n_pts=32)

        ax.plot_surface(X, Y, Z, alpha=0.6, color='C1', edgecolor='C1',
                        linewidth=0.15, shade=True)

        lim = max(a, b, c) * 1.3
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=-60)

        param_str = (f'{name}\n'
                     f'a={a:.0f} b={b:.0f} c={c:.0f}\n'
                     f'$n_1$={n1}  $n_2$={n2}')
        ax.set_title(param_str, fontsize=7, pad=-5)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Display GELLS-DEM granule shapes')
    parser.add_argument('--mode', choices=['2D', '3D', 'both'], default='both',
                        help='Which shape gallery to show (default: both)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Save figure to file instead of displaying')
    args = parser.parse_args()

    if args.mode == '2D':
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(1, 1)
        plot_2d_gallery(fig, gs[0, 0])
    elif args.mode == '3D':
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(1, 1)
        plot_3d_gallery(fig, gs[0, 0])
    else:
        fig = plt.figure(figsize=(16, 16))
        gs = fig.add_gridspec(2, 1, hspace=0.15, height_ratios=[1, 1.1])
        plot_2d_gallery(fig, gs[0])
        plot_3d_gallery(fig, gs[1])

    fig.suptitle('GELLS-DEM Granule Shape Catalogue', fontsize=16,
                 fontweight='bold', y=0.98)

    if args.output:
        fig.savefig(args.output, dpi=200, bbox_inches='tight')
        print(f'Saved to {args.output}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
