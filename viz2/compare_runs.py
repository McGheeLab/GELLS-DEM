"""
viz2/compare_runs.py — several runs on the same figures (V3.0).

Loads any number of finished run directories and draws them together:

* ``compare_scaffolds.png``   final scaffold map of every run, one panel each,
                              same colour code for the same functionalization,
                              a scale bar per panel (domains may differ in size),
                              one shared legend
* ``compare_timeseries.png``  bridging, bridge force, functional connectivity,
                              porosity and displacement vs time, one line per run
* ``compare_species.png``     stacked species fractions vs time, one panel per run
* ``compare_dose_response.png``  final cells per granule, bridging fraction, displacement
                              and coordination number of every species against its f
* ``compare_sweep.png``       (sweeps only) final porosity, coordination, compaction and
                              bridging against the sweep's composition axis, one line per
                              granule-size combination
* ``compare_timelapse.gif``   the scaffold grid animated on a common time axis
* ``compare_summary.md/.csv`` size, cell counts, composition, wall times and
                              final metrics of every run in one table

Usage::

    python viz2/compare_runs.py -i results/showcase_v3/01_binary results/showcase_v3/02_three_species -o out/
    python viz2/compare_runs.py -i results/showcase_v3            # every run directory below it
    python viz2/compare_runs.py -i "results/showcase_v3/0*" --labels Binary "Three species" ...

3D runs are drawn as their z-midplane slice. Large runs (10⁴–10⁵ granules) are
drawn with collections, not one patch per granule.
"""

import sys
import os
import csv
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from matplotlib.lines import Line2D

from gels.engine import load_run
from viz2.common import (
    setup_scaffold_axes, draw_granule_patches, draw_cells_on_ax, save_fig,
    render_frame_to_array, is_3d, slice_snap_z_midplane, species_table,
    granule_rgba, CELL_COLORS, CELL_LABELS, VOID_COLOR,
)
from viz2.palette import species_ids


# ======================================================================
# Loading
# ======================================================================

def discover_run_dirs(items):
    """Expand globs and parent directories into a list of run directories.

    A directory is a run when it holds ``params.json``; a directory without
    one is scanned one level down (so ``-i results/showcase_v3`` works).
    """
    out = []
    for item in items:
        paths = sorted(glob.glob(item)) if any(ch in item for ch in '*?[') else [item]
        for path in paths:
            if not os.path.isdir(path):
                print(f"    Warning: not a directory, skipped: {path}")
                continue
            if os.path.exists(os.path.join(path, 'params.json')):
                out.append(path)
                continue
            subs = sorted(d for d in glob.glob(os.path.join(path, '*'))
                          if os.path.isdir(d) and os.path.exists(os.path.join(d, 'params.json')))
            if not subs:
                print(f"    Warning: no run directories under {path}")
            out.extend(subs)
    seen, uniq = set(), []
    for d in out:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            uniq.append(d)
    return uniq


def _read_json(path, default=None):
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def load_runs(run_dirs, labels=None):
    """Load every run: list of dicts with hist, snaps, p, metadata, state, label."""
    runs = []
    for k, run_dir in enumerate(run_dirs):
        print(f"  Loading {run_dir} ...")
        hist, snaps, p, metadata = load_run(run_dir)
        if not snaps:
            print("    Warning: no snapshots, skipped")
            continue
        state = _read_json(os.path.join(run_dir, 'pipeline_state.json'), {}) or {}
        manifest = _read_json(os.path.join(run_dir, 'showcase.json'), {}) or {}
        label = None
        if labels and k < len(labels) and labels[k]:
            label = labels[k]
        elif manifest.get('label'):
            label = manifest['label']
        else:
            label = os.path.basename(os.path.normpath(run_dir))
        runs.append(dict(run_dir=run_dir, label=label, hist=hist, snaps=snaps, p=p,
                         metadata=metadata or {}, state=state, manifest=manifest))
    return runs


# ======================================================================
# Small helpers
# ======================================================================

def _n_granules(run):
    return int(len(run['snaps'][0]['x']))


def _n_cells(run):
    return int(len(run['snaps'][0].get('cell_state', [])))


def _domain_text(run):
    p = run['p']
    if getattr(p, 'mode', '2D') == '3D':
        return f"3D {p.Lx/1000:.1f} × {p.Ly/1000:.1f} × {p.Lz/1000:.1f} mm"
    return f"2D {p.Lx/1000:.1f} × {p.Ly/1000:.1f} mm"


def _panel_title(run, with_counts=True):
    title = run['label']
    if with_counts:
        title += f"\nN = {_n_granules(run):,} · {_n_cells(run):,} cells · {_domain_text(run)}"
    return title


def _grid_shape(n):
    ncols = 1 if n <= 1 else (2 if n <= 4 else 3)
    if n > 9:
        ncols = 4
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def _draw_snap(ax, run, snap, show_cells=True):
    """Granules (+ cells) of one snapshot on ``ax``; collections for circles."""
    p = run['p']
    draw_snap = slice_snap_z_midplane(snap, p) if is_3d(snap, p) else snap
    setup_scaffold_axes(ax, p, dark=True)
    a = np.asarray(draw_snap.get('a', draw_snap['r']))
    b = np.asarray(draw_snap.get('b', draw_snap['r']))
    n1 = np.asarray(draw_snap.get('n1', draw_snap.get('n_shape', np.full(len(a), 2.0))))
    circles = bool(np.all(a == b) and np.all(n1 == 2.0))
    periodic = getattr(p, 'boundary_mode', 'walls') == 'periodic'
    if circles and not periodic:
        r = np.asarray(draw_snap['r'], dtype=float)
        rgba = granule_rgba(species_ids(draw_snap), species_table(p, draw_snap), alpha=0.95)
        col = EllipseCollection(2 * r, 2 * r, np.zeros(len(r)), units='xy',
                                offsets=np.column_stack([draw_snap['x'], draw_snap['y']]),
                                offset_transform=ax.transData, facecolors=rgba,
                                edgecolors='none')
        ax.add_collection(col)
    else:
        draw_granule_patches(ax, draw_snap, p)
    if show_cells:
        draw_cells_on_ax(ax, draw_snap, p)
    _add_scale_bar(ax, p)
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    return draw_snap


def _add_scale_bar(ax, p):
    L = float(min(p.Lx, p.Ly))
    bar = 1000.0 if L >= 3000 else (500.0 if L >= 1200 else (200.0 if L >= 500 else 100.0))
    x0, y0 = 0.04 * p.Lx, 0.05 * p.Ly
    ax.plot([x0, x0 + bar], [y0, y0], color='white', lw=2.5, solid_capstyle='butt')
    label = f"{bar/1000:g} mm" if bar >= 1000 else f"{bar:g} µm"
    ax.text(x0 + bar / 2, y0 + 0.02 * p.Ly, label, color='white', ha='center',
            va='bottom', fontsize=8)


def species_union(runs):
    """Union of species over runs keyed by (f, colour), sorted by f descending.

    ``names`` lists the species names that share this (f, colour), most widely
    used first, so a legend can show the representative name.
    """
    from collections import Counter
    seen = {}
    for run in runs:
        for sp in species_table(run['p'], run['snaps'][0]):
            key = (round(float(sp['f']), 4), str(sp['color']).lower())
            seen.setdefault(key, Counter())[str(sp['name'])] += 1
    entries = []
    for (f, color), counter in sorted(seen.items(), key=lambda kv: -kv[0][0]):
        names = [n for n, _ in counter.most_common()]
        entries.append(dict(f=f, color=color, names=names))
    return entries


def _legend_handles(runs, seen_states):
    handles = []
    for e in species_union(runs):
        names = e['names']
        name_txt = names[0] if len(names) == 1 else f"{names[0]} (+{len(names)-1})"
        handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=e['color'],
                              markeredgecolor='#444444', markersize=10, linestyle='None',
                              label=f"f = {e['f']:.2f}  ·  {name_txt}"))
    handles.append(Line2D([0], [0], marker='s', color='w', markerfacecolor=VOID_COLOR,
                          markeredgecolor='#444444', markersize=10, linestyle='None', label='Void'))
    for s_val in sorted(seen_states):
        if s_val in CELL_COLORS:
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=CELL_COLORS[s_val],
                                  markersize=7, linestyle='None',
                                  label=CELL_LABELS.get(s_val, f'State {s_val}')))
    return handles


def _cell_states_seen(runs, snap_index):
    seen = set()
    for run in runs:
        snap = run['snaps'][snap_index if snap_index >= 0 else len(run['snaps']) + snap_index]
        st = snap.get('cell_state')
        if st is not None and len(st):
            seen.update(int(v) for v in np.unique(np.asarray(st, dtype=int)))
    return seen


def _snap_at_time(run, t):
    times = np.array([h.get('time', np.nan) for h in run['hist']], dtype=float)
    n = min(len(times), len(run['snaps']))
    if n == 0:
        return 0
    k = int(np.nanargmin(np.abs(times[:n] - t)))
    return k


# ======================================================================
# Figure 1: final scaffold maps side by side
# ======================================================================

def plot_final_scaffolds(runs, outdir=None, snap_index=-1, show_cells=True,
                         filename='compare_scaffolds.png', panel_in=4.6):
    n = len(runs)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, panel_in * nrows + 1.6),
                             facecolor='#111111', squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    t_txt = None
    for k, run in enumerate(runs):
        ax = axes.flat[k]
        ax.set_visible(True)
        snap = run['snaps'][snap_index]
        _draw_snap(ax, run, snap, show_cells=show_cells)
        si = snap_index if snap_index >= 0 else len(run['snaps']) + snap_index
        t = run['hist'][si].get('time', 0.0) if si < len(run['hist']) else 0.0
        t_txt = f"t = {t:.0f} h"
        ax.set_title(_panel_title(run), fontsize=9.5, color='white', linespacing=1.3)
    handles = _legend_handles(runs, _cell_states_seen(runs, snap_index))
    leg = fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6), fontsize=9,
                     facecolor='#222222', edgecolor='#888888', labelcolor='white',
                     bbox_to_anchor=(0.5, 0.0))
    leg.get_frame().set_alpha(0.9)
    fig.suptitle(f"Final scaffolds — {t_txt}" if t_txt else "Final scaffolds",
                 color='white', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.06 if nrows > 1 else 0.14, 1, 0.96), h_pad=3.0)
    if outdir:
        save_fig(fig, outdir, filename, dpi=170)
    return fig


# ======================================================================
# Figure 2: metrics vs time, all runs overlaid
# ======================================================================

# (history key, panel title, y-label, transform of (values, run) -> values)
_TS_PANELS = [
    ('n_bridging_cells', 'Bridging cells per 100 granules', 'cells in a bridge / 100 granules',
     lambda v, run: 100.0 * v / max(_n_granules(run), 1)),
    ('n_bridging_cells', 'Cells in a bridge', 'fraction of live cells',
     lambda v, run: v / np.maximum(_cells_series(run, len(v)), 1.0)),
    ('bridge_force_mean', 'Mean bridge force', 'nN', None),
    ('func_lf', 'Functional connectivity', 'largest coated cluster / coated area', None),
    ('porosity', 'Porosity (inner domain)', 'void fraction', None),
    ('disp_func', 'Displacement of coated granules', 'µm (mean)', None),
    # ── V3.1 bed observables (present only for open / cylindrical containers) ──
    ('bed_height_mean', 'Bed height', 'µm', None),
    ('bed_height_mean', 'Bed height (relative)', 'h(t) / h(0)',
     lambda v, run: v / (v[0] if len(v) and v[0] else 1.0)),
    ('phi_bed', 'Solid fraction of the bed', 'granule volume / envelope', None),
    ('bed_radius_p95', 'Bed radius (95th percentile)', 'µm', None),
    ('wall_contact_fraction', 'Perimeter granules touching the wall', 'fraction', None),
    ('n_cells_total', 'Cell population', 'N(t) / N(0)',
     lambda v, run: v / (v[0] if len(v) and v[0] else 1.0)),
    ('n_divisions_cum', 'Cell divisions', 'cumulative', None),
    # -- V3.2 cell cycle --
    ('n_divisions_cum', 'Division rate', 'divisions / h', lambda v, run: _division_rate(v, run)),
    ('cell_age_cv', 'Cell-cycle asynchrony', 'CV of cell age (0.58 = uniform)', None),
]

# history keys the measured-data overlay can be drawn on
_EXPERIMENT_PANELS = {
    'Bed height': ('bed_height_um', 'bed_height_sd_um', 1.0),
    'Bed height (relative)': ('bed_height_um', 'bed_height_sd_um', 'relative'),
    'Bed radius (95th percentile)': ('bed_diameter_um', 'bed_diameter_sd_um', 0.5),
}


def _division_rate(v, run):
    """Divisions per hour, differentiated from the cumulative count (V3.2).

    Deliberately derived here rather than recorded by the engine: it needs no
    per-step state and it works on runs saved before this panel existed. A
    synchronised population shows a spike; an asynchronous one a smooth curve.
    """
    hist = run.get('hist') or []
    t = np.array([h.get('time', np.nan) for h in hist], dtype=float)
    if len(t) != len(v) or len(v) < 2:
        return np.full(len(v), np.nan)
    dt = np.diff(t)
    rate = np.full(len(v), np.nan)
    rate[1:] = np.diff(v) / np.where(dt > 0, dt, np.nan)
    return rate


def _cells_series(run, n):
    """Live cell count per history entry, falling back to the seeded count."""
    hist = run.get('hist') or []
    if hist and 'n_live_cells' in hist[0]:
        v = np.array([h.get('n_live_cells', np.nan) for h in hist], dtype=float)
    elif hist and 'n_cells_total' in hist[0]:
        v = np.array([h.get('n_cells_total', np.nan) for h in hist], dtype=float)
    else:
        return np.full(n, float(max(_n_cells(run), 1)))
    if len(v) != n:
        v = np.resize(v, n)
    return v


def load_experiment_csv(path):
    """Measured bed geometry vs time for the overlay.

    Columns: ``t_h`` plus any of ``bed_height_um``, ``bed_height_sd_um``,
    ``bed_diameter_um``, ``bed_diameter_sd_um``. Lines starting with # are
    ignored. Returns a dict of float arrays keyed by column name.
    """
    rows = []
    with open(path, newline='', encoding='utf-8') as fh:
        lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith('#')]
    reader = csv.DictReader(lines)
    for r in reader:
        rows.append(r)
    if not rows:
        return {}
    out = {}
    for col in rows[0]:
        if col is None:
            continue
        try:
            out[col.strip()] = np.array([float(r[col]) for r in rows], dtype=float)
        except (TypeError, ValueError):
            continue
    if 't_h' not in out:
        raise ValueError(f"{path}: needs a t_h column (got {sorted(out)})")
    return out


def _run_styles(n):
    cmap = plt.get_cmap('tab10' if n <= 10 else 'tab20')
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<', '>', 'p']
    return [dict(color=cmap(k % cmap.N), marker=markers[k % len(markers)]) for k in range(n)]


def plot_timeseries(runs, outdir=None, filename='compare_timeseries.png', panels=None,
                    experiment=None):
    panels = panels or _TS_PANELS
    present = [pn for pn in panels if any(pn[0] in run['hist'][0] for run in runs if run['hist'])]
    if not present:
        print("    Warning: none of the requested history keys present; skipping time series")
        return None
    ncols = 3 if len(present) > 4 else 2
    nrows = int(np.ceil(len(present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.8 * nrows), squeeze=False)
    styles = _run_styles(len(runs))
    for ax in axes.flat[len(present):]:
        ax.set_visible(False)
    for ax, (key, title, ylabel, tf) in zip(axes.flat, present):
        for run, st in zip(runs, styles):
            hist = run['hist']
            if not hist or key not in hist[0]:
                continue
            t = np.array([h.get('time', np.nan) for h in hist], dtype=float)
            v = np.array([h.get(key, np.nan) for h in hist], dtype=float)
            if tf is not None:
                v = tf(v, run)
            every = max(1, len(t) // 12)
            ax.plot(t, v, '-', color=st['color'], lw=1.8, marker=st['marker'], markevery=every,
                    ms=4.5, label=run['label'])
        # measured data on the panels it applies to (V3.1)
        spec = _EXPERIMENT_PANELS.get(title) if experiment else None
        if spec:
            col, sd_col, scale = spec
            if col in experiment:
                ev = experiment[col]
                if scale == 'relative':
                    ev = ev / (ev[0] if len(ev) and ev[0] else 1.0)
                    err = None
                else:
                    ev = ev * scale
                    err = experiment.get(sd_col)
                    err = err * scale if err is not None else None
                ax.errorbar(experiment['t_h'], ev, yerr=err, fmt='o', color='k', ms=6,
                            capsize=3, lw=1.4, zorder=5, label='measured')
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Time (h)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(runs), 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('Metrics vs time — all conditions', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.06 + 0.02 * int(np.ceil(len(runs) / 5)), 1, 0.96))
    if outdir:
        save_fig(fig, outdir, filename)
    return fig


# ======================================================================
# Figure 3: species fractions, one stacked panel per run
# ======================================================================

def plot_species_fractions(runs, outdir=None, filename='compare_species.png'):
    runs_ok = [r for r in runs if r['hist'] and 'phi_sp_0_mean' in r['hist'][0]]
    if not runs_ok:
        print("    Warning: no per-species history keys; skipping species fractions")
        return None
    n = len(runs_ok)
    nrows, ncols = _grid_shape(n)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows), squeeze=False)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, run in zip(axes.flat, runs_ok):
        hist, p = run['hist'], run['p']
        species = species_table(p, run['snaps'][0])
        t = np.array([h['time'] for h in hist], dtype=float)
        bottom = np.zeros_like(t)
        K = 0
        while f'phi_sp_{K}_mean' in hist[0]:
            K += 1
        for k in range(K):
            vals = np.array([h.get(f'phi_sp_{k}_mean', 0.0) for h in hist], dtype=float)
            sp = species[k] if k < len(species) else {'name': f'species_{k}', 'color': '#888888', 'f': float('nan')}
            ax.fill_between(t, bottom, bottom + vals, color=sp['color'], alpha=0.8,
                            label=f"f = {sp['f']:.2f}")
            bottom = bottom + vals
        pv = np.array([h.get('phi_v_mean', 0.0) for h in hist], dtype=float)
        ax.fill_between(t, bottom, bottom + pv, color='#BBBBBB', alpha=0.5, label='void')
        if 'phi_c_mean' in hist[0]:
            pc = np.array([h.get('phi_c_mean', 0.0) for h in hist], dtype=float)
            ax.plot(t, pc, 'k--', lw=1.2, label='collagen-weighted solid')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(t.min(), t.max())
        ax.set_title(run['label'], fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (h)')
        ax.set_ylabel('fraction of domain')
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7, loc='center right', framealpha=0.85)
    fig.suptitle('Species fractions vs time', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if outdir:
        save_fig(fig, outdir, filename)
    return fig


# ======================================================================
# Figure 4: dose-response — per-species outcome vs functionalization
# ======================================================================

def _per_species_finals(run):
    """Final per-species quantities keyed by species index (V3.0 history keys)."""
    hist = run['hist']
    if not hist or 'n_gran_sp_0' not in hist[-1]:
        return None
    h = hist[-1]
    species = species_table(run['p'], run['snaps'][0])
    out = []
    for k, sp in enumerate(species):
        n_g = float(h.get(f'n_gran_sp_{k}', 0.0))
        if n_g <= 0:
            continue
        n_c = float(h.get(f'n_cells_sp_{k}', 0.0))
        out.append(dict(
            f=float(sp['f']), name=sp['name'], color=sp['color'], n_gran=n_g,
            cells_per_granule=n_c / n_g,
            bridging_frac=(float(h.get(f'n_bridging_sp_{k}', 0.0)) / n_c) if n_c > 0 else np.nan,
            disp=float(h.get(f'disp_sp_{k}', np.nan)),
            Z=float(h.get(f'Z_sp_{k}', np.nan)),
        ))
    return sorted(out, key=lambda d: d['f'])


_DOSE_PANELS = [
    ('cells_per_granule', 'Cells seeded per granule', 'cells / granule'),
    ('bridging_frac', 'Cells that formed a bridge', 'fraction of that species\' cells'),
    ('disp', 'Displacement over the run', 'µm (mean)'),
    ('Z', 'Coordination number', 'contacts / granule'),
]


def plot_dose_response(runs, outdir=None, filename='compare_dose_response.png'):
    """Final per-species outcomes against that species' functionalization f.

    Every point is one species of one run, so a run with K species contributes K
    points joined by a line. This is the V3.0 species model's payoff: within a
    single packing, granules differing only in collagen coverage take different
    numbers of cells, form bridges at different rates and move different distances.
    """
    data = [(run, _per_species_finals(run)) for run in runs]
    data = [(r, d) for r, d in data if d]
    if not data:
        print("    Warning: no per-species history keys; skipping dose-response")
        return None
    fig, axes = plt.subplots(1, len(_DOSE_PANELS), figsize=(4.4 * len(_DOSE_PANELS), 4.0))
    styles = _run_styles(len(runs))
    style_of = {id(run): st for run, st in zip(runs, styles)}
    for ax, (key, title, ylabel) in zip(np.atleast_1d(axes), _DOSE_PANELS):
        for run, rows in data:
            st = style_of[id(run)]
            xs = [r['f'] for r in rows]
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, '-', color=st['color'], lw=1.4, alpha=0.75, zorder=1,
                    label=run['label'])
            # each marker filled with that species' own colour
            for r in rows:
                ax.plot([r['f']], [r[key]], marker=st['marker'], ms=9, mfc=r['color'],
                        mec=st['color'], mew=1.6, ls='None', zorder=2)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel('Functionalization  f  (collagen-I coverage)')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.05, 1.05)
        ax.grid(True, alpha=0.25)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=min(len(runs), 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Outcome vs degree of functionalization (final state; marker fill = species colour)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.06 + 0.03 * int(np.ceil(len(runs) / 5)), 1, 0.94))
    if outdir:
        save_fig(fig, outdir, filename)
    return fig


# ======================================================================
# Figure 5: sweep factor grid (runs carrying `factors` in showcase.json)
# ======================================================================

_SWEEP_PANELS = [
    ('porosity', 'Porosity', 'void fraction (inner domain)', None),
    ('n_contacts', 'Coordination', 'contacts per granule',
     lambda v, run: v / max(_n_granules(run), 1)),
    ('disp_func', 'Compaction', 'mean displacement of coated granules [µm]', None),
    ('n_bridging_cells', 'Bridging', 'bridging cells per 100 granules',
     lambda v, run: 100.0 * v / max(_n_granules(run), 1)),
]

_SIZE_ORDER = ['small', 'medium', 'large']


def plot_sweep_factors(runs, outdir=None, filename='compare_sweep.png',
                       x_key='func_ratio', x_label='Functional share of the solid'):
    """Final metrics against the sweep's x factor, one line per size combination.

    Reads the `factors` dict each run's showcase.json carries (written by
    pipeline/run_showcase.py). Returns None for runs without it, so the plain
    showcase set simply skips this figure. Colour encodes the functional granule
    size, line style the inert granule size; the 100 %-functional runs have no
    inert phase, so their point is shared by every line of that colour.
    """
    rows = []
    for run in runs:
        fac = (run.get('manifest') or {}).get('factors') or {}
        if x_key not in fac or not run['hist']:
            continue
        rows.append((run, fac))
    if len(rows) < 2:
        return None

    func_sizes = sorted({f['func_size'] for _r, f in rows},
                        key=lambda k: _SIZE_ORDER.index(k) if k in _SIZE_ORDER else 99)
    inert_sizes = sorted({f['inert_size'] for _r, f in rows if f['inert_size'] != 'none'},
                         key=lambda k: _SIZE_ORDER.index(k) if k in _SIZE_ORDER else 99)
    cmap = plt.get_cmap('viridis')
    colour = {k: cmap(0.1 + 0.8 * i / max(len(func_sizes) - 1, 1))
              for i, k in enumerate(func_sizes)}
    dash = {k: st for k, st in zip(inert_sizes, ['-', '--', ':', '-.'])}
    mark = {k: m for k, m in zip(inert_sizes, ['o', 's', '^', 'D'])}

    fig, axes = plt.subplots(1, len(_SWEEP_PANELS), figsize=(4.6 * len(_SWEEP_PANELS), 4.2))
    for ax, (key, title, ylabel, tf) in zip(np.atleast_1d(axes), _SWEEP_PANELS):
        for fs in func_sizes:
            # the no-inert runs of this functional size: shared endpoint of every line
            shared = [(f[x_key], r) for r, f in rows
                      if f['func_size'] == fs and f['inert_size'] == 'none']
            for is_ in inert_sizes:
                pts = [(f[x_key], r) for r, f in rows
                       if f['func_size'] == fs and f['inert_size'] == is_] + shared
                if not pts:
                    continue
                pts.sort(key=lambda p: p[0])
                xs, ys = [], []
                for x, run in pts:
                    v = run['hist'][-1].get(key)
                    if v is None:
                        continue
                    xs.append(x)
                    ys.append(tf(float(v), run) if tf else float(v))
                if len(xs) < 1:
                    continue
                ax.plot(xs, ys, dash[is_], color=colour[fs], marker=mark[is_], ms=6,
                        lw=1.8, alpha=0.9)
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    handles = [Line2D([0], [0], color=colour[k], lw=3, label=f'functional: {k}')
               for k in func_sizes]
    handles += [Line2D([0], [0], color='#555555', ls=dash[k], marker=mark[k], lw=1.8,
                       label=f'inert: {k}') for k in inert_sizes]
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 6), fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Sweep: final state vs composition, by granule size',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    if outdir:
        save_fig(fig, outdir, filename)
    return fig


# ======================================================================
# Figure 6: synchronized timelapse GIF
# ======================================================================

def gif_timelapse(runs, outdir, filename='compare_timelapse.gif', max_frames=24, fps=4,
                  show_cells=True, panel_in=3.6):
    try:
        import imageio
    except ImportError:
        print("    Warning: imageio not available, skipping timelapse GIF")
        return None
    t_end = min(max(h.get('time', 0.0) for h in run['hist']) for run in runs if run['hist'])
    n_common = min(len(run['snaps']) for run in runs)
    n_frames = min(max_frames, n_common)
    times = np.linspace(0.0, t_end, n_frames)
    n = len(runs)
    nrows, ncols = _grid_shape(n)
    frames = []
    for fi, t in enumerate(times):
        fig, axes = plt.subplots(nrows, ncols, figsize=(panel_in * ncols, panel_in * nrows + 0.9),
                                 facecolor='#111111', squeeze=False)
        for ax in axes.flat:
            ax.set_visible(False)
        for k, run in enumerate(runs):
            ax = axes.flat[k]
            ax.set_visible(True)
            si = _snap_at_time(run, t)
            _draw_snap(ax, run, run['snaps'][si], show_cells=show_cells)
            ax.set_title(run['label'], fontsize=9, color='white')
        fig.suptitle(f"t = {t:5.1f} h", color='white', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        frames.append(render_frame_to_array(fig))
        plt.close(fig)
        print(f"    frame {fi + 1}/{n_frames} (t = {t:.1f} h)")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, filename)
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)
    print(f"    Saved: {path}")
    return path


# ======================================================================
# Summary table
# ======================================================================

def _final(run, key, fmt='{:.3g}'):
    hist = run['hist']
    if not hist or key not in hist[-1] or hist[-1][key] is None:
        return '—'
    try:
        return fmt.format(float(hist[-1][key]))
    except (TypeError, ValueError):
        return str(hist[-1][key])


def _composition(run):
    species = species_table(run['p'], run['snaps'][0])
    sid = species_ids(run['snaps'][0])
    parts = []
    for sp in species:
        n_k = int(np.sum(sid == sp['id']))
        parts.append(f"f={sp['f']:.2f}: {n_k}")
    return ', '.join(parts)


def _step_seconds(run, step, key):
    return run['state'].get('steps', {}).get(step, {}).get('details', {}).get(key)


def summary_rows(runs):
    rows = []
    for run in runs:
        md = run['metadata']
        pack_s = _step_seconds(run, 'pack', 'pack_seconds')
        sim_s = _step_seconds(run, 'simulate', 'wall_seconds')
        if sim_s is None:
            sim_s = md.get('wall_time_s')
        post_s = _step_seconds(run, 'postprocess', 'wall_seconds')
        fac = (run.get('manifest') or {}).get('factors') or {}
        rows.append({
            'condition': run['label'],
            'run_dir': run['run_dir'],
            **{f'factor_{k}': v for k, v in fac.items()},
            'mode': getattr(run['p'], 'mode', '2D'),
            'domain': _domain_text(run),
            'granules': _n_granules(run),
            'cells': _n_cells(run),
            'composition (granules per species)': _composition(run),
            'shapes': 'yes' if getattr(run['p'], 'shape_enabled', False) else 'no',
            't_total_h': f"{run['hist'][-1].get('time', float('nan')):.0f}" if run['hist'] else '—',
            'steps': md.get('steps_completed', md.get('n_steps', '—')),
            'pack_s': f"{pack_s:.1f}" if pack_s is not None else '—',
            'simulate_s': f"{sim_s:.1f}" if sim_s is not None else '—',
            'postprocess_s': f"{post_s:.1f}" if post_s is not None else '—',
            'final_bridgeable_pairs': _final(run, 'n_bridges', '{:.0f}'),
            'final_bridging_cells': _final(run, 'n_bridging_cells', '{:.0f}'),
            'final_locked_cells': _final(run, 'n_locked_in_cells', '{:.0f}'),
            'final_bridge_force_nN': _final(run, 'bridge_force_mean', '{:.1f}'),
            'final_func_lf': _final(run, 'func_lf', '{:.3f}'),
            'final_porosity': _final(run, 'porosity', '{:.3f}'),
            'final_disp_func_um': _final(run, 'disp_func', '{:.1f}'),
            # ── V3.1 bed observables ──
            'initial_bed_height_um': _first(run, 'bed_height_mean', '{:.0f}'),
            'final_bed_height_um': _final(run, 'bed_height_mean', '{:.0f}'),
            'bed_height_change_pct': _change_pct(run, 'bed_height_mean'),
            'final_phi_bed': _final(run, 'phi_bed', '{:.3f}'),
            'final_wall_contact': _final(run, 'wall_contact_fraction', '{:.3f}'),
            'cells_initial': _first(run, 'n_cells_total', '{:.0f}'),
            'cells_final': _final(run, 'n_cells_total', '{:.0f}'),
            'divisions_total': _final(run, 'n_divisions_cum', '{:.0f}'),
        })
    return rows


def _first(run, key, fmt='{:.3f}'):
    hist = run.get('hist') or []
    for h in hist:
        if key in h and h[key] is not None:
            try:
                return fmt.format(float(h[key]))
            except (TypeError, ValueError):
                return '—'
    return '—'


def _change_pct(run, key):
    """Percentage change of a history key from its first to its last value."""
    hist = run.get('hist') or []
    vals = [h[key] for h in hist if key in h and h[key] is not None]
    if len(vals) < 2 or not vals[0]:
        return '—'
    try:
        return f"{100.0 * (float(vals[-1]) - float(vals[0])) / float(vals[0]):+.1f}"
    except (TypeError, ValueError, ZeroDivisionError):
        return '—'


def write_summary(runs, outdir, basename='compare_summary'):
    rows = summary_rows(runs)
    if not rows:
        return None
    os.makedirs(outdir, exist_ok=True)
    cols = list(rows[0].keys())
    csv_path = os.path.join(outdir, basename + '.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md_path = os.path.join(outdir, basename + '.md')
    show = [c for c in cols if c != 'run_dir']
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write("# Run comparison\n\n")
        fh.write("| " + " | ".join(show) + " |\n")
        fh.write("|" + "---|" * len(show) + "\n")
        for r in rows:
            fh.write("| " + " | ".join(str(r[c]) for c in show) + " |\n")
        fh.write("\nRun directories:\n\n")
        for r in rows:
            fh.write(f"- {r['condition']}: `{r['run_dir']}`\n")
    print(f"    Saved: {md_path}")
    print(f"    Saved: {csv_path}")
    return md_path



# ======================================================================
# Figure 8 (V3.1): vertical section through the bed
# ======================================================================

def _snap_at_time(run, t_target):
    """Index of the snapshot nearest a wall-clock time."""
    snaps = run.get('snaps') or []
    if not snaps:
        return None
    hist = run.get('hist') or []
    times = [h.get('time', k) for k, h in enumerate(hist)][:len(snaps)]
    if not times:
        return 0
    return int(np.argmin(np.abs(np.array(times, dtype=float) - t_target)))


def _slice_xz_axis(snap, p, thickness=None):
    """A 2D-looking snapshot from a vertical section through the container axis.

    Granules within ``thickness`` of the y = Ly/2 plane are projected onto
    (x, z), with their radii shrunk to the circle the plane actually cuts, so
    the picture is a section and not a projection. Returns None in 2D, where
    the run is already a section.
    """
    if 'z' not in snap:
        return None
    y0 = 0.5 * float(p.Ly)
    r = np.asarray(snap['r'], dtype=float)
    thickness = float(thickness if thickness is not None else np.max(r))
    d = np.abs(np.asarray(snap['y'], dtype=float) - y0)
    keep = d < np.minimum(r, thickness)

    def _sel(v):
        """Subset per-granule arrays; leave scalars, strings and other lengths alone."""
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == len(r):
            return arr[keep]
        return v

    # 'z' and 'quat' are dropped so the section reads as a plain 2D snapshot:
    # keeping them would make the drawing helper slice it again at the z-midplane.
    drop = ('phi_f', 'phi_i', 'phi_v', 'z', 'quat', 'cell_z', 'a', 'b', 'c', 'n1', 'n2', 'n_shape')
    out = {k: _sel(v) for k, v in snap.items() if k not in drop}
    out['x'] = np.asarray(snap['x'], dtype=float)[keep]
    out['y'] = np.asarray(snap['z'], dtype=float)[keep]          # z becomes the vertical axis
    out['r'] = np.sqrt(np.maximum(r[keep] ** 2 - d[keep] ** 2, 1e-9))
    out['a'] = out['r']
    out['b'] = out['r']
    out['n1'] = np.full(len(out['r']), 2.0)
    out['_section'] = True
    return out


def plot_bed_profiles(runs, outdir=None, filename='compare_bed_profiles.png',
                      times=(0.0, 24.0, 48.0, 72.0), experiment=None):
    """Vertical sections of every run at a few times, with the container drawn.

    For a 3D run this is a slice through the cylinder axis; a 2D dish slice is
    already a section. Works on closed boxes too, which is how the V3.0
    packing artefact (a ball inside the cube) shows up directly.
    """
    usable = [r for r in runs if r.get('snaps')]
    if not usable:
        print("    Warning: no snapshots loaded; skipping bed profiles")
        return None
    t_end = max((r['hist'][-1].get('time', 0.0) if r['hist'] else 0.0) for r in usable)
    cols = [t for t in times if t <= t_end + 1e-9]
    if len(cols) < 2:                      # short run: spread the columns over what there is
        cols = list(np.linspace(0.0, t_end, min(4, max(2, len(usable[0].get('snaps') or [1])))))
    nrows, ncols = len(usable), len(cols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows),
                             squeeze=False, facecolor='#111111')
    for ri, run in enumerate(usable):
        p = run['p']
        is3d = getattr(p, 'mode', '2D') == '3D'
        shape = getattr(p, 'boundary_shape', 'box')
        top_free = getattr(p, 'boundary_top', 'wall') == 'free'
        L_up = float(p.Lz if is3d else p.Ly)
        for ci, t in enumerate(cols):
            ax = axes[ri][ci]
            ax.set_facecolor('#111111')
            idx = _snap_at_time(run, t)
            snap = run['snaps'][idx] if idx is not None else None
            if snap is not None:
                sec = _slice_xz_axis(snap, p) if is3d else snap
                if sec is not None and len(sec.get('x', [])):
                    # drawn directly rather than through _draw_snap: the section is
                    # already a plane, and the snapshot helpers would slice it again.
                    xs = np.asarray(sec['x'], dtype=float)
                    ys = np.asarray(sec['y'], dtype=float)
                    rs = np.asarray(sec['r'], dtype=float)
                    rgba = granule_rgba(species_ids(sec), species_table(p, sec), alpha=0.95)
                    ax.add_collection(EllipseCollection(
                        2 * rs, 2 * rs, np.zeros(len(rs)), units='xy',
                        offsets=np.column_stack([xs, ys]), offset_transform=ax.transData,
                        facecolors=rgba, edgecolors='none'))
            # container outline
            if is3d and shape == 'cylinder':
                cx, R = 0.5 * float(p.Lx), 0.5 * min(float(p.Lx), float(p.Ly))
                x_lo, x_hi = cx - R, cx + R
            else:
                x_lo, x_hi = 0.0, float(p.Lx)
            ax.plot([x_lo, x_lo], [0, L_up], color='#8899AA', lw=1.6)
            ax.plot([x_hi, x_hi], [0, L_up], color='#8899AA', lw=1.6)
            ax.plot([x_lo, x_hi], [0, 0], color='#8899AA', lw=2.2)
            if not top_free:
                ax.plot([x_lo, x_hi], [L_up, L_up], color='#8899AA', lw=2.2)
            h = None
            if run['hist'] and idx is not None and idx < len(run['hist']):
                h = run['hist'][idx].get('bed_height_mean')
            if h:
                ax.plot([x_lo, x_hi], [h, h], color='#FFD166', lw=1.4, ls='--')
            if experiment is not None and 'bed_height_um' in experiment:
                near = np.abs(experiment['t_h'] - t) <= 2.0
                if np.any(near):
                    ax.plot([x_hi], [float(np.mean(experiment['bed_height_um'][near]))],
                            marker='<', color='#FF5C5C', ms=9, clip_on=False)
            ax.set_xlim(x_lo - 0.02 * (x_hi - x_lo), x_hi + 0.02 * (x_hi - x_lo))
            ax.set_ylim(0, L_up)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t = {t:.0f} h", color='white', fontsize=11, fontweight='bold')
            if ci == 0:
                ax.set_ylabel(run['label'], color='white', fontsize=9)
    fig.suptitle('Vertical section through the bed', color='white', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if outdir:
        save_fig(fig, outdir, filename)
    return fig


# ======================================================================
# Orchestration
# ======================================================================

def run_all(run_dirs, labels=None, outdir=None, max_frames=24, fps=4, skip=(), show_cells=True,
            experiment=None):
    """Load the runs and produce every comparison figure into ``outdir``."""
    run_dirs = discover_run_dirs(run_dirs)
    if not run_dirs:
        print("  No run directories found.")
        return []
    runs = load_runs(run_dirs, labels)
    if not runs:
        print("  Nothing to compare.")
        return []
    outdir = outdir or os.path.join(os.path.dirname(os.path.normpath(runs[0]['run_dir'])), 'comparison')
    print(f"\n  Comparing {len(runs)} runs -> {outdir}")
    skip = set(skip or ())
    n_steps = 8
    if 'scaffolds' not in skip:
        print(f"  [1/{n_steps}] final scaffolds")
        plot_final_scaffolds(runs, outdir=outdir, show_cells=show_cells)
    if 'timeseries' not in skip:
        print(f"  [2/{n_steps}] time series")
        plot_timeseries(runs, outdir=outdir, experiment=experiment)
    if 'species' not in skip:
        print(f"  [3/{n_steps}] species fractions")
        plot_species_fractions(runs, outdir=outdir)
    if 'dose' not in skip:
        print(f"  [4/{n_steps}] dose-response vs f")
        plot_dose_response(runs, outdir=outdir)
    if 'sweep' not in skip:
        print(f"  [5/{n_steps}] sweep factor grid")
        if plot_sweep_factors(runs, outdir=outdir) is None:
            print("    (runs carry no sweep factors; skipped)")
    if 'summary' not in skip:
        print(f"  [6/{n_steps}] summary table")
        write_summary(runs, outdir)
    if 'timelapse' not in skip:
        print(f"  [7/{n_steps}] timelapse GIF")
        gif_timelapse(runs, outdir, max_frames=max_frames, fps=fps, show_cells=show_cells)
    # V3.1: a vertical section through the container. Most informative for an
    # open container, but it also shows how a closed box was packed, so it is
    # drawn for every run set (turn it off with --skip bed).
    if 'bed' not in skip:
        print(f"  [8/{n_steps}] bed profiles")
        plot_bed_profiles(runs, outdir=outdir, experiment=experiment)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Draw several GELS runs on the same figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('-i', '--runs', nargs='+', required=True,
                        help='Run directories, globs, or a parent directory of runs')
    parser.add_argument('-o', '--outdir', default=None,
                        help='Output directory (default: <parent of first run>/comparison)')
    parser.add_argument('--labels', nargs='*', default=None,
                        help='Panel labels, one per run (default: showcase label or directory name)')
    parser.add_argument('--max-frames', type=int, default=24, help='Frames in the timelapse GIF')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument('--no-cells', action='store_true', help='Do not draw cells on the maps')
    parser.add_argument('--experiment', default=None,
                        help='CSV of measured bed geometry to overlay '
                             '(columns t_h, bed_height_um[, bed_height_sd_um][, bed_diameter_um])')
    parser.add_argument('--skip', nargs='*', default=[],
                        choices=['scaffolds', 'timeseries', 'species', 'dose', 'sweep', 'bed',
                                 'summary', 'timelapse'],
                        help='Figures to skip')
    args = parser.parse_args()
    experiment = load_experiment_csv(args.experiment) if getattr(args, 'experiment', None) else None
    run_all(args.runs, labels=args.labels, outdir=args.outdir, max_frames=args.max_frames,
            fps=args.fps, skip=args.skip, show_cells=not args.no_cells, experiment=experiment)


if __name__ == '__main__':
    main()
