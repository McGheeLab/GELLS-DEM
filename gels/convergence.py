"""
Has the run arrested? (V3.6)
============================

Decides when a scaffold has stopped evolving, so a run can stop instead of
burning the rest of ``t_total``. Ported from ``robotsim/code/convergence.py``,
whose central idea is the whole reason this can be trusted:

    **Proportional windows.** Coarsening in an arrested gel slows as a power law
    ``t^-alpha``. Over FIXED windows such a process drops below any threshold
    quickly and falsely triggers. Over proportional windows -- compare
    ``[t/2, 3t/4)`` against ``[3t/4, t]`` -- the change per window decays one
    power of ``t`` more slowly, so the false trigger is pushed far outside any
    run you would actually do.

**Measured, because the usual phrasing of this claim is too strong.** A power
law does not stay un-flat for ever under proportional windows either; it just
takes far longer. For ``x(t) = 1 - 0.5 t^-alpha``:

    alpha = 0.3    proportional first passes at t = 635 h   fixed window: 58 h
    alpha = 0.5    proportional first passes at t = 203 h   fixed window: 50 h

A GELS run is 24-72 h. So a fixed-window detector would falsely stop a still
coarsening run INSIDE it, and the proportional one does not come near. That is
the claim `tests/test_convergence.py` holds, and it is a ratio, not an
absolute.

Signals
-------
Five flatness tests and two guards, all from the flat metrics dict plus a path
length the monitor accumulates itself:

1. **path length** -- `disp_func` / `disp_inert` are NET displacement and cancel
   under creep, so they cannot be used. The monitor sums ``|dx|`` between
   consecutive SAVED frames instead. Save resolution is deliberate: it makes the
   live detector and the shadow harness compute the *identical* signal, so a
   tolerance calibrated in shadow is valid live. Flat if **quiet or steady** --
   a prestressed bed can sustain a constant-rate creep that never changes the
   structure, and steady residual motion with every structural signal flat is
   converged. Real coarsening shows a DECAYING rate and failing structural
   signals, so it still blocks.
2. **bridges** -- `n_bridges`, relative.
3. **connected functional phase** -- `gran_lf_func`, absolute. **Not**
   `func_lf`: that is `scipy.ndimage.label` on a tanh field, so it moves with
   the interface halo and with `Ngrid`, and its flatness is partly a rendering
   artefact.
4. **mixing** -- `demix_phi_loc` and the functional-inert coordination
   ``n_contacts_if / n_contacts``, relative with an absolute floor. Catches slow
   phase separation still in progress when the positions already look quiet.
5. **force** -- `F_mean`, relative. Mechanical settling.

**Guard A, division.** Differentiate `n_divisions_cum`. A constant division rate
is not a steady state however flat everything else looks. Pinned at 0 when
division is off, so it costs nothing there.

**Guard B, functional-phase compaction.** `compaction_func >= 0.97` AND
``|d phi_loc_func|`` small -- at the ceiling *and* not still rising. Both come
from `gels/laguerre.py`, so `conv_enable` requires `output.metrics_laguerre`.
Do **not** substitute `compaction_ratio`: it is halo-inflated ~1.5x, so a 0.97
threshold against it means nothing. When the Laguerre keys are absent the
monitor **refuses to converge** rather than quietly dropping the guard.

Stop rule: every flatness test passes and both guards are satisfied,
continuously from the streak start ``t0`` until ``t >= CONV_SPAN * t0`` -- the
pass streak has to span a full proportional window of its own -- and
``t >= conv_t_min_h``.

Nothing here imports from `gels.engine`: the monitor consumes the flat metrics
dict and a position array, so `pipeline/`, `tests/` and the engine can all use
it, and the shadow harness replays a finished run through the identical code.
"""

import numpy as np

__all__ = ['ConvergenceMonitor', 'COLS', 'TOLERANCES', 'path_increment']

# The raw per-frame buffer, in order. Persisted to convergence.json so a resumed
# run re-ingests its own history instead of starting the windows from scratch.
COLS = ('t', 'path_func', 'path_inert', 'n_bridges', 'gran_lf_func',
        'demix_phi_loc', 'z_if', 'F_mean', 'n_divisions_cum',
        'compaction_func', 'phi_loc_func', 'has_laguerre')

# Module constants, not config fields: there are a dozen and they are calibrated
# together by `pipeline/shadow_check.py`, not chosen one at a time. `overrides`
# on the constructor is the escape hatch the shadow sweep uses.
TOLERANCES = dict(
    span=2.0,            # the pass streak must span t0 -> span * t0
    eps_disp=0.02,       # path per granule per 100 h, in mean-radius units
    steady_frac=0.10,    # |rate_B - rate_A| <= this * rate_A counts as steady
    eps_bridge=0.02,     # relative change in n_bridges
    eps_gel=0.01,        # absolute change in gran_lf_func
    eps_mix=0.02,        # relative change in demix_phi_loc or z_if
    eps_force=0.05,      # relative change in F_mean
    div_frac=0.10,       # division rate in B, as a fraction of the run mean
    compaction_min=0.97, # compaction_func must have reached this
    eps_phi_loc=0.005,   # ... and phi_loc_func must not still be rising
    min_rows=8,          # fewer than this and no decision is made at all
)


def path_increment(pos, pos_prev, mask=None):
    """Mean straight-line distance moved between two saved frames, um.

    The signal `disp_func` cannot give: net displacement cancels under creep,
    and a bed that is quietly churning reads the same as one that has stopped.
    """
    if pos_prev is None or pos is None:
        return 0.0
    pos = np.asarray(pos, dtype=float)
    pos_prev = np.asarray(pos_prev, dtype=float)
    n = min(len(pos), len(pos_prev))
    if n == 0:
        return 0.0
    d = np.linalg.norm(pos[:n] - pos_prev[:n], axis=1)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)[:n]
        if not m.any():
            return 0.0
        d = d[m]
    return float(np.mean(d))


class ConvergenceMonitor:
    """Feed one row per saved frame with `update`; read `should_stop()`.

    ``mean_r`` sets the displacement scale, so the tolerance is in granule radii
    rather than micrometres and transfers between bed sizes.
    """

    def __init__(self, enable=False, shadow=False, t_min_h=24.0, mean_r=20.0,
                 overrides=None):
        self.enable = bool(enable)
        self.shadow = bool(shadow)
        self.t_min = float(t_min_h)
        self.mean_r = max(float(mean_r), 1e-9)
        self.tol = dict(TOLERANCES)
        if overrides:
            unknown = set(overrides) - set(self.tol)
            if unknown:
                raise KeyError(f'unknown convergence tolerance(s): {sorted(unknown)}')
            self.tol.update(overrides)
        self.rows = []
        self._path = [0.0, 0.0]        # running cumulative path, func / inert
        self._pos_prev = None
        self.streak_t0 = None
        self.converged = False
        self.reason = ''
        self.last = {}

    # ── ingestion ────────────────────────────────────────────────────

    def observe_positions(self, pos, func_mask):
        """Accumulate the path length from this frame's positions.

        Call once per SAVED frame, before `update`. The monitor keeps only the
        previous frame's positions, so the cost is one array and one norm.
        """
        if self._pos_prev is not None:
            inert = ~np.asarray(func_mask, dtype=bool)
            self._path[0] += path_increment(pos, self._pos_prev, func_mask)
            self._path[1] += path_increment(pos, self._pos_prev, inert)
        self._pos_prev = np.array(pos, dtype=float, copy=True)
        return tuple(self._path)

    def row_from_metrics(self, t, m):
        """Build one raw row from the flat metrics dict and the running path."""
        nc = float(m.get('n_contacts', 0.0) or 0.0)
        z_if = float(m.get('n_contacts_if', 0.0)) / nc if nc > 0 else 0.0
        has_lag = ('compaction_func' in m) and ('phi_loc_func' in m)
        return {
            't': float(t),
            'path_func': self._path[0],
            'path_inert': self._path[1],
            'n_bridges': float(m.get('n_bridges', 0.0) or 0.0),
            'gran_lf_func': float(m.get('gran_lf_func', 0.0) or 0.0),
            'demix_phi_loc': float(m.get('demix_phi_loc', 0.0) or 0.0),
            'z_if': z_if,
            'F_mean': float(m.get('F_mean', 0.0) or 0.0),
            'n_divisions_cum': float(m.get('n_divisions_cum', 0.0) or 0.0),
            'compaction_func': float(m.get('compaction_func', 0.0) or 0.0),
            'phi_loc_func': float(m.get('phi_loc_func', 0.0) or 0.0),
            'has_laguerre': 1.0 if has_lag else 0.0,
        }

    def load_rows(self, rows):
        """Re-ingest raw rows on resume, without evaluating.

        History carries no positions, so the monitor persists its own buffer
        (``convergence.json``); this is what reads it back.
        """
        for r in rows:
            try:
                self.rows.append(tuple(float(r[k]) for k in COLS))
            except (KeyError, TypeError, ValueError):
                continue
        if self.rows:
            self._path = [self.rows[-1][COLS.index('path_func')],
                          self.rows[-1][COLS.index('path_inert')]]
        return len(self.rows)

    def update(self, row):
        """Add one raw row (a dict with the COLS keys) and evaluate."""
        self.rows.append(tuple(float(row[k]) for k in COLS))
        self.last = self._evaluate()
        return self.last

    def should_stop(self):
        return self.enable and (not self.shadow) and self.converged

    def status(self):
        return dict(enabled=self.enable, shadow=self.shadow,
                    converged=self.converged, reason=self.reason,
                    streak_t0=self.streak_t0, n_rows=len(self.rows))

    # ── evaluation ───────────────────────────────────────────────────

    @staticmethod
    def _win(a, t0, t1):
        t = a[:, 0]
        return a[(t >= t0 - 1e-12) & (t <= t1 + 1e-12)]

    def _blank(self, blocking=''):
        """Every key, always.

        The record is merged into the metrics dict, which goes through
        `csv.DictWriter` with the field names taken from the first row -- so a
        record that sometimes omits keys would break the CSV on the frame the
        detector first has enough data. Zeros mean "no decision", which is also
        what they mean before the windows fill.
        """
        return dict(conv_pass=0, conv_converged=int(self.converged),
                    conv_streak_t0=float(self.streak_t0 or 0.0),
                    conv_path_func=0.0, conv_path_inert=0.0,
                    conv_path_steady=0, conv_path_quiet=0,
                    conv_dbridge=0.0, conv_dgel=0.0, conv_dmix=0.0,
                    conv_dforce=0.0, conv_div=0.0, conv_compaction=0.0,
                    conv_blocking=blocking)

    def _evaluate(self):
        rec = self._blank()
        if self.converged:
            rec['conv_pass'] = 1
            return rec
        a = np.asarray(self.rows, dtype=float)
        t = a[-1, 0]
        if t <= 0 or len(a) < int(self.tol['min_rows']):
            rec['conv_blocking'] = 'not enough rows'
            return rec
        A = self._win(a, 0.50 * t, 0.75 * t)
        B = self._win(a, 0.75 * t, t)
        if len(A) < 2 or len(B) < 2:
            # Not enough resolution to decide. This is why validate() checks
            # that save_every_h yields enough rows: without it the detector
            # silently never fires.
            rec['conv_blocking'] = 'window too sparse'
            return rec
        col = {k: i for i, k in enumerate(COLS)}
        T = self.tol

        def mean(w, k):
            return float(np.mean(w[:, col[k]]))

        # 1. path length: quiet OR steady (see the module docstring)
        scale = 100.0 / (0.25 * t) / self.mean_r
        rates = {}
        for key in ('path_func', 'path_inert'):
            q = np.interp([0.50 * t, 0.75 * t, t], a[:, 0], a[:, col[key]])
            rates[key] = ((q[1] - q[0]) * scale, (q[2] - q[1]) * scale)
        quiet = all(rB < T['eps_disp'] for _rA, rB in rates.values())
        steady = all(abs(rB - rA) <= T['steady_frac'] * max(rA, T['eps_disp'])
                     for rA, rB in rates.values())
        f_path = quiet or steady

        # 2-5. the structural and mechanical flatness tests
        bA, bB = mean(A, 'n_bridges'), mean(B, 'n_bridges')
        d_bridge = abs(bB - bA) / max(bA, 1.0)
        d_gel = abs(mean(B, 'gran_lf_func') - mean(A, 'gran_lf_func'))
        d_mix = max(abs(mean(B, 'demix_phi_loc') - mean(A, 'demix_phi_loc'))
                    / max(abs(mean(A, 'demix_phi_loc')), 1e-3),
                    abs(mean(B, 'z_if') - mean(A, 'z_if'))
                    / max(abs(mean(A, 'z_if')), 1e-3))
        d_force = abs(mean(B, 'F_mean') - mean(A, 'F_mean')) / max(abs(mean(A, 'F_mean')), 1e-8)

        # Guard A: division. A constant division rate is not a steady state.
        div_B = (a[-1, col['n_divisions_cum']] - mean(B, 'n_divisions_cum')) / max(0.125 * t, 1e-9)
        div_all = a[-1, col['n_divisions_cum']] / max(t, 1e-9)
        g_div = div_all <= 0.0 or (div_B / max(div_all, 1e-12)) < T['div_frac']

        # Guard B: functional-phase compaction. Refuses, rather than passes,
        # when the Laguerre keys are absent.
        has_lag = mean(B, 'has_laguerre') > 0.5
        d_phi = abs(mean(B, 'phi_loc_func') - mean(A, 'phi_loc_func'))
        g_comp = has_lag and (mean(B, 'compaction_func') >= T['compaction_min']) \
            and (d_phi < T['eps_phi_loc'])

        tests = (('path', f_path), ('bridges', d_bridge < T['eps_bridge']),
                 ('gel', d_gel < T['eps_gel']), ('mix', d_mix < T['eps_mix']),
                 ('force', d_force < T['eps_force']),
                 ('division', g_div), ('compaction', g_comp))
        ok = all(v for _k, v in tests)
        # The most useful column in the shadow report: the LAST signal to fail.
        rec['conv_blocking'] = '' if ok else ','.join(k for k, v in tests if not v)

        if ok:
            if self.streak_t0 is None:
                self.streak_t0 = t
            if t >= T['span'] * self.streak_t0 and t >= self.t_min:
                self.converged = True
                self.reason = (f'flat since t={self.streak_t0:.1f} h '
                               f'(path {rates["path_func"][1]:.2e}, '
                               f'bridges {d_bridge:.2e}, gel {d_gel:.2e}, '
                               f'mix {d_mix:.2e}, force {d_force:.2e})')
        else:
            self.streak_t0 = None

        rec.update(conv_pass=int(ok), conv_converged=int(self.converged),
                   conv_streak_t0=float(self.streak_t0 or 0.0),
                   conv_path_func=float(rates['path_func'][1]),
                   conv_path_inert=float(rates['path_inert'][1]),
                   conv_path_steady=int(steady), conv_path_quiet=int(quiet),
                   conv_dbridge=float(d_bridge), conv_dgel=float(d_gel),
                   conv_dmix=float(d_mix), conv_dforce=float(d_force),
                   conv_div=float(div_B / max(div_all, 1e-12)) if div_all > 0 else 0.0,
                   conv_compaction=float(mean(B, 'compaction_func')))
        return rec
