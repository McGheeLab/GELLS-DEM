"""
Live viewer process (matplotlib, TkAgg) for a running or saved GELS run.
=======================================================================

``viewer_main`` is the ``multiprocessing`` target spawned by
``step3_simulate.py --live``; ``run_viewer`` is shared with the tailing /
replay CLI (``pipeline/live_view.py``). One artist per class of thing —
an ``EllipseCollection`` for all granules, one marker ``Line2D`` per cell
state, a ``LineCollection`` for bridges — updated in place and blitted, so a
frame costs milliseconds even at 10⁴ granules.

Rules
-----
* This module must not import ``viz2.common`` (it pins the Agg backend).
  It uses ``viz2.palette`` and ``viz2.snapshot_ops`` only.
* ``MPLBACKEND`` is set before matplotlib is imported (the parent pipeline
  process exports ``Agg``).
* The viewer never blocks the simulation: it drains whatever frames are
  queued and draws the newest one.

Keys
----
  space  pause / resume      n  single step        x  stop the simulation (saves state)
  q Esc  detach (close, sim continues) / quit in tail mode
  + -    frames more / less often                  c  colour: species → f → speed
  b l m  toggle cells / bridges / metrics panel    g  toggle periodic ghost images
  [ ] 0  3D: move the z-slice down / up / midplane s  save PNG      h  help
"""

import os
import queue
import sys
import time

import numpy as np

HELP_TEXT = (
    "space pause/resume   n step   x stop sim   q/Esc detach\n"
    "+/- frame cadence    c colour (species/f/speed)\n"
    "b cells  l bridges  m metrics  g ghosts   [ ] 0 z-slice   s save png   h help")


def _utf8_console():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except (AttributeError, ValueError):
            pass


# ──────────────────────────────────────────────────────────────────────
# Sources and controls
# ──────────────────────────────────────────────────────────────────────

class QueueSource:
    """Messages from the engine process (multiprocessing.Queue)."""

    def __init__(self, frame_q):
        self.q = frame_q
        self.closed = False

    def poll(self):
        msgs = []
        while True:
            try:
                msgs.append(self.q.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError, BrokenPipeError):
                self.closed = True
                break
        return msgs


class QueueControl:
    """Control channel back to the engine (pause/step/stop/...)."""
    is_live = True

    def __init__(self, ctrl_q, ctrl_evt, stop_evt):
        self.q = ctrl_q
        self.evt = ctrl_evt
        self.stop_evt = stop_evt

    def send(self, *cmd):
        try:
            self.q.put(tuple(cmd))
            self.evt.set()
        except (EOFError, OSError, BrokenPipeError, ValueError):
            pass

    def stop(self):
        try:
            self.stop_evt.set()
        except Exception:
            pass
        self.send('stop')


class LocalControl:
    """Control for replay/tail mode: pause affects playback only."""
    is_live = False

    def __init__(self, source=None):
        self.source = source

    def send(self, *cmd):
        if self.source is None or not cmd:
            return
        op = cmd[0]
        if op == 'pause':
            self.source.paused = True
        elif op == 'resume':
            self.source.paused = False
        elif op == 'step':
            self.source.step_credits += int(cmd[1]) if len(cmd) > 1 else 1
        elif op == 'every':
            self.source.set_speed(int(cmd[1]))

    def stop(self):
        if self.source is not None:
            self.source.finished = True


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────

def _yaw_from_quat(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))


class LiveRenderer:
    """Draws frames into one axes using persistent artists."""

    def __init__(self, fig, ax, start, opts):
        from matplotlib.collections import EllipseCollection, LineCollection
        from matplotlib.lines import Line2D
        from viz2.palette import (CELL_COLORS, CELL_LABELS, LOCKED_BRIDGE_COLOR, VOID_COLOR,
                                  cell_rgba, granule_rgba, hex_to_rgba)
        self.fig, self.ax = fig, ax
        self.start = start
        self.opts = opts
        self.mode = start['mode']
        self.is3d = self.mode == '3D'
        self.N = start['N']
        self.Lx, self.Ly, self.Lz = start['Lx'], start['Ly'], start['Lz']
        self.periodic = start['boundary_mode'] == 'periodic'
        self.species = start['species']
        self.z_frac = float(opts.get('z_frac', 0.5))
        self.color_mode = opts.get('color_by', 'species')
        self.show_cells = not opts.get('no_cells', False)
        self.show_bridges = True
        self.show_ghosts = True
        self.rgba_species = granule_rgba(start['species_id'], self.species, alpha=0.95)
        self.f = np.asarray(start['f'], dtype=np.float64)
        self.locked_color = hex_to_rgba(LOCKED_BRIDGE_COLOR)
        self.bridge_color = hex_to_rgba(CELL_COLORS[3], 0.9)
        self._cell_rgba = cell_rgba
        self._hex = hex_to_rgba

        ax.set_facecolor(VOID_COLOR)
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        a, b = start['a'], start['b']
        self.granules = EllipseCollection(2 * a, 2 * b, np.zeros(self.N), units='xy',
                                          offsets=np.zeros((self.N, 2)), offset_transform=ax.transData,
                                          facecolors=self.rgba_species, edgecolors='none', animated=True)
        ax.add_collection(self.granules)
        self.ghosts = EllipseCollection([], [], [], units='xy', offsets=np.zeros((0, 2)),
                                        offset_transform=ax.transData, facecolors='none',
                                        edgecolors='none', animated=True)
        ax.add_collection(self.ghosts)
        self.cell_artists = {}
        for state, color in CELL_COLORS.items():
            ln, = ax.plot([], [], linestyle='None', marker='o', markersize=4.0,
                          markerfacecolor=color, markeredgecolor='white', markeredgewidth=0.2,
                          animated=True, label=CELL_LABELS[state])
            self.cell_artists[state] = ln
        self.bridges = LineCollection(np.zeros((0, 2, 2)), colors=[self.bridge_color],
                                      linewidths=1.4, animated=True)
        ax.add_collection(self.bridges)
        self.hud = ax.text(0.01, 0.99, '', transform=ax.transAxes, va='top', ha='left',
                           color='white', fontsize=8, family='monospace', animated=True,
                           bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))
        self.help_text = ax.text(0.5, 0.02, '', transform=ax.transAxes, va='bottom', ha='center',
                                 color='yellow', fontsize=8, family='monospace', animated=True,
                                 bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
        title = f"{os.path.basename(str(start.get('run_dir', ''))) or 'GELS'} — {self.mode}, " \
                f"N={self.N}, cells={start['n_cells']}"
        ax.set_title(title, fontsize=10)
        self._legend()
        self.frame = None
        self.paused = False
        self.bg = None
        self.help_visible = False

    def _legend(self):
        from matplotlib.lines import Line2D
        from viz2.palette import CELL_COLORS, CELL_LABELS
        handles = [Line2D([0], [0], marker='s', color='w', markerfacecolor=s['color'], markersize=9,
                          linestyle='None', label=f"{s['name']} (f={s['f']:.2f})") for s in self.species]
        for state in (0, 1, 2, 3, 4, 5):
            handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=CELL_COLORS[state],
                                  markersize=6, linestyle='None', label=CELL_LABELS[state]))
        self.ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=5,
                       fontsize=7, framealpha=0.6, borderaxespad=0.0)

    # ── frame → artists ────────────────────────────────────────────────
    def _view_2d(self, frame):
        """(indices, x, y, widths, heights, angles) of granules to draw."""
        if not self.is3d:
            idx = np.arange(self.N)
            return (idx, frame['x'].astype(float), frame['y'].astype(float),
                    2 * self.start['a'], 2 * self.start['b'], np.degrees(frame['theta'].astype(float)))
        z = frame['z'].astype(float)
        z_mid = self.Lz * self.z_frac
        a, b, c = self.start['a'], self.start['b'], self.start['c']
        rb = np.maximum(np.maximum(a, b), c)
        dz = np.abs(z - z_mid)
        idx = np.where(dz < rb)[0]
        if idx.size == 0:
            idx = np.arange(self.N)
        n2 = self.start['n2'][idx]
        frac = np.clip(dz[idx] / np.maximum(c[idx], 1e-6), 0.0, 0.999)
        shrink = np.maximum(1.0 - frac ** n2, 0.0) ** (1.0 / n2)
        ang = np.degrees(_yaw_from_quat(frame['quat'][idx].astype(float)))
        return (idx, frame['x'][idx].astype(float), frame['y'][idx].astype(float),
                2 * a[idx] * shrink, 2 * b[idx] * shrink, ang)

    def _colors(self, frame, idx):
        if self.color_mode == 'f':
            from matplotlib import colormaps
            return colormaps['viridis'](self.f[idx])
        if self.color_mode == 'speed' and 'speed' in frame:
            from matplotlib import colormaps
            sp = frame['speed'].astype(float)[idx]
            vmax = max(float(np.percentile(sp, 99)), 1e-9)
            return colormaps['magma'](np.clip(sp / vmax, 0, 1))
        return self.rgba_species[idx]

    def update(self, frame):
        # V3.1: adopt a refreshed cell table when the population has grown
        for key in ('n_cells', 'cell_granule_id', 'cell_offset'):
            if key in frame:
                self.start[key] = frame[key]
        self.frame = frame
        idx, x, y, w, h, ang = self._view_2d(frame)
        self.granules.set_offsets(np.column_stack([x, y]))
        self.granules.set_widths(w)
        self.granules.set_heights(h)
        self.granules.set_angles(ang)
        self.granules.set_facecolors(self._colors(frame, idx))
        # periodic ghost images of near-edge granules
        if self.periodic and self.show_ghosts:
            rb = np.maximum(w, h) / 2
            gx, gy, gw, gh, ga, gc = [], [], [], [], [], []
            cols = self._colors(frame, idx)
            for dx in (-self.Lx, 0.0, self.Lx):
                for dy in (-self.Ly, 0.0, self.Ly):
                    if dx == 0.0 and dy == 0.0:
                        continue
                    xi, yi = x + dx, y + dy
                    m = (xi + rb > 0) & (xi - rb < self.Lx) & (yi + rb > 0) & (yi - rb < self.Ly)
                    if np.any(m):
                        gx.append(xi[m]); gy.append(yi[m]); gw.append(w[m]); gh.append(h[m])
                        ga.append(ang[m]); gc.append(cols[m])
            if gx:
                self.ghosts.set_offsets(np.column_stack([np.concatenate(gx), np.concatenate(gy)]))
                self.ghosts.set_widths(np.concatenate(gw)); self.ghosts.set_heights(np.concatenate(gh))
                self.ghosts.set_angles(np.concatenate(ga)); self.ghosts.set_facecolors(np.concatenate(gc))
            else:
                self.ghosts.set_offsets(np.zeros((0, 2)))
        else:
            self.ghosts.set_offsets(np.zeros((0, 2)))

        # cells
        if self.show_cells and 'cell_x' in frame:
            cx, cy = frame['cell_x'].astype(float), frame['cell_y'].astype(float)
            state = frame['cell_state'].astype(int)
            if self.is3d:
                host = self.start['cell_granule_id']
                keep_g = np.zeros(self.N, dtype=bool); keep_g[idx] = True
                cell_keep = keep_g[host]
            else:
                cell_keep = np.ones(cx.shape[0], dtype=bool)
            for st, ln in self.cell_artists.items():
                m = cell_keep & (state == st)
                ln.set_data(cx[m], cy[m])
        else:
            for ln in self.cell_artists.values():
                ln.set_data([], [])

        # bridges
        if self.show_bridges and 'bridge_seg' in frame and frame['bridge_seg'].shape[0] > 0:
            seg = frame['bridge_seg'].astype(float)
            locked = frame.get('bridge_locked', np.zeros(seg.shape[0], dtype=bool))
            if self.is3d:
                host = self.start['cell_granule_id'][frame['bridge_cell']]
                keep_g = np.zeros(self.N, dtype=bool); keep_g[idx] = True
                m = keep_g[host]
                seg, locked = seg[m], locked[m]
            self.bridges.set_segments(seg[:, :, :2])
            self.bridges.set_colors([self.locked_color if lk else self.bridge_color for lk in locked])
        else:
            self.bridges.set_segments(np.zeros((0, 2, 2)))

        stats = frame.get('stats', {})
        lines = [f"step {frame['step']:>6d}   t = {frame['t']:7.2f} h",
                 f"{stats.get('steps_per_s', 0):5.1f} step/s   every {stats.get('every', '-')}   "
                 f"dropped {stats.get('n_dropped', 0)}"]
        m = frame.get('metrics')
        if m:
            lines.append(f"bridges {int(m.get('n_bridges', 0)):<5d} porosity {m.get('porosity', 0):.3f}")
            lines.append(f"K {m.get('K_kozeny_carman', 0):.1f} µm²   cluster {m.get('func_lf', 0):.2f}")
        tail = f"colour: {self.color_mode}"
        if self.is3d:
            tail += f"   z {self.z_frac:.2f}"
        if self.paused:
            tail += "   [PAUSED]"
        lines.append(tail)
        self.hud.set_text('\n'.join(lines))
        self.help_text.set_text(HELP_TEXT if self.help_visible else '')

    def animated_artists(self):
        return [self.granules, self.ghosts, *self.cell_artists.values(), self.bridges,
                self.hud, self.help_text]


class MetricsPanel:
    """Small time-series column: bridges, porosity, permeability, largest cluster."""
    KEYS = [('n_bridges', 'bridges'), ('porosity', 'porosity'),
            ('K_kozeny_carman', 'K (µm²)'), ('func_lf', 'largest cluster')]

    def __init__(self, axes):
        self.axes = axes
        self.t = []
        self._seen = set()
        self.series = {k: [] for k, _ in self.KEYS}
        self.lines = {}
        for ax, (key, label) in zip(axes, self.KEYS):
            ln, = ax.plot([], [], color='#4FC3F7', lw=1.2)
            self.lines[key] = ln
            ax.set_ylabel(label, fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)
        axes[-1].set_xlabel('t (h)', fontsize=7)
        self.dirty = False

    def add(self, metrics):
        if not metrics or 'time' not in metrics:
            return
        key = round(float(metrics['time']), 6)
        if key in self._seen:                 # history preload and per-frame metrics overlap
            return
        self._seen.add(key)
        self.t.append(float(metrics['time']))
        for key, _ in self.KEYS:
            self.series[key].append(float(metrics.get(key, np.nan)))
        self.dirty = True

    def load_history(self, hist):
        for m in hist:
            self.add(m)

    def redraw(self):
        for ax, (key, _) in zip(self.axes, self.KEYS):
            self.lines[key].set_data(self.t, self.series[key])
            ax.relim()
            ax.autoscale_view()
        self.dirty = False


# ──────────────────────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────────────────────

def run_viewer(source, ctrl, opts):
    """Drive the window (or headless PNG dumps) from a message source."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    headless = bool(opts.get('headless'))
    fps = float(opts.get('fps', 20.0))
    out_dir = opts.get('out')
    hold = opts.get('hold', True)
    show_metrics = opts.get('metrics', True)

    fig = plt.figure(figsize=(11, 7) if show_metrics else (7.5, 7.5), facecolor='#202020')
    if show_metrics:
        gs_ = GridSpec(4, 5, figure=fig, wspace=0.35, hspace=0.45)
        ax = fig.add_subplot(gs_[:, :3])
        maxes = [fig.add_subplot(gs_[i, 3:]) for i in range(4)]
        for a in maxes:
            a.set_facecolor('#303030'); a.tick_params(colors='#DDDDDD')
            for sp in a.spines.values():
                sp.set_color('#777777')
            a.yaxis.label.set_color('#DDDDDD'); a.xaxis.label.set_color('#DDDDDD')
        panel = MetricsPanel(maxes)
    else:
        ax = fig.add_subplot(111)
        panel = None
    ax.title.set_color('#EEEEEE')

    state = {'renderer': None, 'bg': None, 'done': False, 'frames_seen': 0, 'skipped': 0,
             'last_full': 0.0, 'n_saved': 0, 'quit': False, 'paused': False}
    canvas = fig.canvas

    def full_draw():
        canvas.draw()
        if not headless:
            state['bg'] = canvas.copy_from_bbox(fig.bbox)
        state['last_full'] = time.perf_counter()

    def blit():
        r = state['renderer']
        if r is None:
            return
        if headless or state['bg'] is None:
            full_draw()
            return
        canvas.restore_region(state['bg'])
        for art in r.animated_artists():
            ax.draw_artist(art)
        canvas.blit(fig.bbox)
        canvas.flush_events()

    def handle(msgs):
        r = state['renderer']
        frame = None
        for m in msgs:
            kind = m.get('kind')
            if kind == 'start':
                r = LiveRenderer(fig, ax, m, opts)
                state['renderer'] = r
                if panel is not None and m.get('history'):
                    panel.load_history(m['history'])
                full_draw()
                if not headless:
                    print(f"  viewer: window up ({m['mode']}, N={m['N']}, cells={m['n_cells']})", flush=True)
            elif kind == 'frame':
                if frame is not None:
                    state['skipped'] += 1
                frame = m
                if panel is not None and m.get('metrics'):
                    panel.add(m['metrics'])
            elif kind == 'end':
                state['done'] = True
                if not headless:
                    print(f"  viewer: run {m.get('reason', '')}; showed {state['frames_seen']} frame(s), "
                          f"{state['skipped']} superseded in the queue", flush=True)
                if r is not None:
                    r.hud.set_text(r.hud.get_text() + f"\nfinished: {m.get('reason', '')} — "
                                   + ("close the window to exit" if hold else ""))
                if panel is not None and panel.dirty:
                    panel.redraw()
                full_draw()
        if frame is not None and state['renderer'] is not None:
            r = state['renderer']
            r.paused = state['paused']
            r.update(frame)
            state['frames_seen'] += 1
            if state['frames_seen'] == 1 and not headless:
                print(f"  viewer: first frame at step {frame['step']}", flush=True)
            if panel is not None and panel.dirty and (headless or time.perf_counter() - state['last_full'] > 1.0):
                panel.redraw()
                full_draw()
            else:
                blit()
            if headless and out_dir:
                os.makedirs(out_dir, exist_ok=True)
                fig.savefig(os.path.join(out_dir, f"frame_{frame['step']:06d}.png"), dpi=100,
                            facecolor=fig.get_facecolor())
                state['n_saved'] += 1

    def on_key(event):
        r = state['renderer']
        k = event.key
        if k == ' ':
            state['paused'] = not state['paused']
            ctrl.send('pause' if state['paused'] else 'resume')
            if r is not None:
                r.paused = state['paused']
                r.update(r.frame) if r.frame else None
                blit()
        elif k == 'n':
            state['paused'] = True
            ctrl.send('step', 1)
        elif k == 'x':
            ctrl.stop()
            if not ctrl.is_live:
                state['quit'] = True
                plt.close(fig)
        elif k in ('q', 'escape'):
            ctrl.send('detach')
            state['quit'] = True
            plt.close(fig)
        elif k in ('+', '='):
            ctrl.send('every', max(1, int((r.frame or {}).get('stats', {}).get('every', 2)) // 2) if r else 1)
        elif k == '-':
            ctrl.send('every', int((r.frame or {}).get('stats', {}).get('every', 1)) * 2 if r else 2)
        elif k == 'c' and r is not None:
            modes = ['species', 'f', 'speed']
            r.color_mode = modes[(modes.index(r.color_mode) + 1) % len(modes)]
            ctrl.send('color', r.color_mode)
            if r.frame:
                r.update(r.frame); blit()
        elif k == 'b' and r is not None:
            r.show_cells = not r.show_cells
            ctrl.send('cells', r.show_cells)
            if r.frame:
                r.update(r.frame); blit()
        elif k == 'l' and r is not None:
            r.show_bridges = not r.show_bridges
            if r.frame:
                r.update(r.frame); blit()
        elif k == 'g' and r is not None:
            r.show_ghosts = not r.show_ghosts
            if r.frame:
                r.update(r.frame); blit()
        elif k == 'm' and panel is not None:
            for a in panel.axes:
                a.set_visible(not a.get_visible())
            full_draw()
        elif k in ('[', ']', '0') and r is not None and r.is3d:
            r.z_frac = 0.5 if k == '0' else float(np.clip(r.z_frac + (0.05 if k == ']' else -0.05), 0.0, 1.0))
            if r.frame:
                r.update(r.frame); blit()
        elif k == 's' and r is not None:
            run_dir = str(state['renderer'].start.get('run_dir') or '.')
            d = os.path.join(run_dir, 'live') if os.path.isdir(run_dir) else (out_dir or '.')
            os.makedirs(d, exist_ok=True)
            step = (r.frame or {}).get('step', 0)
            path = os.path.join(d, f'frame_{step:06d}.png')
            fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
            print(f"  saved {path}")
        elif k == 'h' and r is not None:
            r.help_visible = not r.help_visible
            if r.frame:
                r.update(r.frame); blit()

    def on_resize(event):
        state['bg'] = None
        full_draw()

    def on_close(event):
        if not state['quit']:
            ctrl.send('detach')
        state['quit'] = True

    if headless:
        # Plain loop: consume until the source ends, dumping PNGs.
        idle = 0
        while not state['done'] and not state['quit']:
            msgs = source.poll()
            if msgs:
                idle = 0
                handle(msgs)
            else:
                idle += 1
                if getattr(source, 'finished', False) or getattr(source, 'closed', False) or idle > 200:
                    break
                time.sleep(0.02)
        plt.close(fig)
        return state

    canvas.mpl_connect('key_press_event', on_key)
    canvas.mpl_connect('resize_event', on_resize)
    canvas.mpl_connect('close_event', on_close)
    interval_ms = max(10, int(1000.0 / fps))

    def tick():
        if state['quit']:
            return
        try:
            msgs = source.poll()
            if msgs:
                handle(msgs)
            elif getattr(source, 'closed', False) and not state['done']:
                state['done'] = True
            if state['done'] and not hold:
                plt.close(fig)
                return
        except Exception as exc:   # keep the window alive on a bad frame
            print(f"  viewer: {type(exc).__name__}: {exc}")
        try:
            canvas.manager.window.after(interval_ms, tick)
        except Exception:
            pass

    full_draw()
    try:
        canvas.manager.window.after(interval_ms, tick)
    except Exception:
        # backends without a Tk window: fall back to a timer
        timer = canvas.new_timer(interval=interval_ms)
        timer.add_callback(tick)
        timer.start()
    plt.show(block=True)
    return state


def viewer_main(frame_q, ctrl_q, ctrl_evt, stop_evt, opts):
    """multiprocessing target: TkAgg window fed by the engine's observer."""
    os.environ['MPLBACKEND'] = opts.get('backend', 'TkAgg')
    _utf8_console()
    import matplotlib
    matplotlib.use(os.environ['MPLBACKEND'])
    source = QueueSource(frame_q)
    ctrl = QueueControl(ctrl_q, ctrl_evt, stop_evt)
    try:
        run_viewer(source, ctrl, opts)
    finally:
        try:
            ctrl.send('detach')
        except Exception:
            pass


__all__ = ['viewer_main', 'run_viewer', 'LiveRenderer', 'MetricsPanel', 'QueueSource',
           'QueueControl', 'LocalControl', 'HELP_TEXT']
