"""
Shared helpers for the GELS step-by-step local pipeline.
=======================================================

The pipeline splits one simulation run into five numbered steps, each a
standalone script in this directory:

    step1_config.py       resolve parameters      -> params.json
    step2_pack.py         build the packing       -> snapshots/snap_0000.npz
    step3_simulate.py     advance to t_total      -> snapshots/snap_NNNN.npz
    step4_postprocess.py  render figures/movies   -> viz2_plots/
    step5_analysis.py     mathematical analysis   -> analysis/

Every step reads and writes a single *run directory*. Progress is recorded in
``<run_dir>/pipeline_state.json`` so each step can verify its prerequisite ran,
refuse to silently clobber completed work, and be re-run on its own with
``--force``.

Steps 2 and 3 are joined by the engine's own V2.6 resume mechanism: step 2
writes snapshot 0000 (the initial packing), and step 3 restores that snapshot
and continues the main loop from it. Because resuming restarts the random
number stream, a packing + simulate pair is not bit-for-bit identical to a
single monolithic ``run()`` call, though it is fully deterministic and
reproducible for a given seed.
"""

import json
import os
import sys
from datetime import datetime

# Repo root on sys.path so `gels`, `viz`, `viz2`, `analysis` all import
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Local runs are headless by default; figures are written, never shown.
os.environ.setdefault('MPLBACKEND', 'Agg')

# Windows consoles default to a legacy codepage (cp1252) that cannot encode
# the Unicode the engine prints freely (nu, mu, arrows, box-drawing rules).
# Without this, a run dies inside a print() statement. errors='replace' keeps
# output flowing even on a console that cannot be switched to UTF-8.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):  # not a reconfigurable text stream
        pass

STATE_FILE = 'pipeline_state.json'

# Ordered pipeline steps: (key, number, script, human label)
STEPS = [
    ('config',      1, 'step1_config.py',      'Configure'),
    ('pack',        2, 'step2_pack.py',        'Generate packing'),
    ('simulate',    3, 'step3_simulate.py',    'Run simulation'),
    ('postprocess', 4, 'step4_postprocess.py', 'Post-process'),
    ('analysis',    5, 'step5_analysis.py',    'Analysis'),
]

STEP_INFO = {key: (num, script, label) for key, num, script, label in STEPS}


# ----------------------------------------------------------------------
# Console formatting
# ----------------------------------------------------------------------

def banner(step_key, run_dir):
    """Print the standard step header."""
    num, _script, label = STEP_INFO[step_key]
    print('=' * 68)
    print(f"  GELS pipeline - STEP {num}/{len(STEPS)}: {label}")
    print('=' * 68)
    print(f"  Run directory: {os.path.abspath(run_dir)}")


def done(step_key, run_dir, elapsed=None):
    """Print the standard step footer, including the next command to run."""
    num, _script, label = STEP_INFO[step_key]
    print()
    took = f" in {elapsed:.1f}s" if elapsed is not None else ""
    print(f"  Step {num} ({label}) complete{took}.")
    nxt = next((s for s in STEPS if s[1] == num + 1), None)
    if nxt:
        print(f"  Next:  python pipeline/{nxt[2]} -i {run_dir}")
    else:
        print(f"  Pipeline finished. Results in {os.path.abspath(run_dir)}")


# ----------------------------------------------------------------------
# Pipeline state
# ----------------------------------------------------------------------

def state_path(run_dir):
    return os.path.join(run_dir, STATE_FILE)


def read_state(run_dir):
    """Read pipeline_state.json, or return an empty skeleton if absent."""
    path = state_path(run_dir)
    if not os.path.exists(path):
        return {'run_dir': os.path.abspath(run_dir), 'steps': {}}
    with open(path) as f:
        return json.load(f)


def write_state(run_dir, state):
    os.makedirs(run_dir, exist_ok=True)
    with open(state_path(run_dir), 'w') as f:
        json.dump(state, f, indent=2, default=str)


def mark_done(run_dir, step_key, details=None):
    """Record a step as completed, with optional details."""
    state = read_state(run_dir)
    state.setdefault('steps', {})[step_key] = {
        'status': 'done',
        'completed': datetime.now().isoformat(timespec='seconds'),
        'details': details or {},
    }
    write_state(run_dir, state)


def is_done(run_dir, step_key):
    state = read_state(run_dir)
    return state.get('steps', {}).get(step_key, {}).get('status') == 'done'


def step_details(run_dir, step_key):
    state = read_state(run_dir)
    return state.get('steps', {}).get(step_key, {}).get('details', {})


def _dir_flag(step_key):
    """The run-directory flag a given step's CLI expects.

    Step 1 creates the directory (-o/--run-dir); every later step consumes an
    existing one (-i/--run-dir).
    """
    return '-o' if step_key == 'config' else '-i'


def require(run_dir, step_key):
    """Abort with a helpful message unless `step_key` has completed."""
    if is_done(run_dir, step_key):
        return
    num, script, label = STEP_INFO[step_key]
    flag = _dir_flag(step_key)
    print(f"\n  ERROR: step {num} ({label}) has not been run for this "
          f"run directory.")
    print(f"  Run this first:\n\n      python pipeline/{script} {flag} {run_dir}\n")
    sys.exit(1)


def guard(run_dir, step_key, force):
    """Abort if `step_key` already completed and --force was not given."""
    if not is_done(run_dir, step_key) or force:
        return
    num, script, _label = STEP_INFO[step_key]
    detail = step_details(run_dir, step_key)
    when = read_state(run_dir)['steps'][step_key].get('completed', '?')
    flag = _dir_flag(step_key)
    print(f"\n  Step {num} already completed for this run ({when}).")
    for k, v in (detail or {}).items():
        print(f"    {k}: {v}")
    print("\n  Nothing to do. Re-run it anyway with:\n")
    print(f"      python pipeline/{script} {flag} {run_dir} --force\n")
    sys.exit(0)


def print_progress(run_dir):
    """Print a checklist of pipeline progress for a run directory."""
    state = read_state(run_dir)
    steps = state.get('steps', {})
    print(f"\n  Progress for {os.path.abspath(run_dir)}:")
    for key, num, _script, label in STEPS:
        entry = steps.get(key, {})
        if entry.get('status') == 'done':
            mark, when = '[x]', entry.get('completed', '')
        else:
            mark, when = '[ ]', ''
        print(f"    {mark} {num}. {label:<18} {when}")


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

def params_from_run_dir(run_dir):
    """Rebuild a Params object from the run directory's params.json."""
    from gels.engine import Params

    path = os.path.join(run_dir, 'params.json')
    if not os.path.exists(path):
        print(f"\n  ERROR: no params.json in {run_dir}.")
        print("  Run step 1 first:\n\n"
              f"      python pipeline/step1_config.py -o {run_dir}\n")
        sys.exit(1)

    with open(path) as f:
        params_dict = json.load(f)

    p = Params()
    for k, v in params_dict.items():
        if not hasattr(p, k):
            continue
        field_type = type(getattr(p, k))
        try:
            setattr(p, k, field_type(v))
        except (ValueError, TypeError):
            setattr(p, k, v)
    return p


def save_params(p, run_dir):
    """Write Params to <run_dir>/params.json (load_run-compatible)."""
    from dataclasses import asdict

    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'params.json'), 'w') as f:
        json.dump(asdict(p), f, indent=2, default=float)


def add_params_args(parser):
    """Add one CLI flag per Params field, defaulting to None (= no override).

    Mirrors the behaviour of the archived ``run_hpc_headless.py`` so existing
    command lines keep working locally.
    """
    from gels.engine import Params

    for field_name, field_val in vars(Params()).items():
        if isinstance(field_val, bool):
            # bool is a subclass of int, so it must be checked first
            parser.add_argument(
                f"--{field_name}",
                type=lambda s: s.lower() in ('true', '1', 'yes'),
                default=None, metavar='BOOL')
        elif isinstance(field_val, (int, float)):
            parser.add_argument(f"--{field_name}",
                                type=type(field_val), default=None)
        elif isinstance(field_val, str):
            parser.add_argument(f"--{field_name}", type=str, default=None)
    # V3.0 legacy aliases (--W_adh_ff etc.) write to the renamed field.
    for old_name, new_name in getattr(Params, 'LEGACY_ALIASES', {}).items():
        parser.add_argument(f"--{old_name}", dest=new_name, type=float, default=None,
                            help=f"alias of --{new_name}")
    return parser


def apply_params_args(p, args):
    """Apply non-None CLI overrides onto a Params object. Returns changed keys."""
    changed = {}
    for field_name in vars(p):
        val = getattr(args, field_name, None)
        if val is not None:
            setattr(p, field_name, val)
            changed[field_name] = val
    return changed


def seed_for(run_dir, fallback=None):
    """Return the seed recorded at step 1, or `fallback`."""
    detail = step_details(run_dir, 'config')
    seed = detail.get('seed')
    return fallback if seed is None else int(seed)


# ----------------------------------------------------------------------
# Trial JSON loading
#
# Ported verbatim (behaviour-wise) from the archived run_hpc_headless.py so
# that every JSON in Trials/ keeps working with the local pipeline.
# ----------------------------------------------------------------------

def load_trial_json(path):
    """Load a Trial JSON and return a dict of Params field overrides.

    Supports two formats:
      - **Flat (V1.4+)**: keys are Params field names directly. Detected by
        ``"_format": "flat"`` or by any top-level key matching a Params field.
      - **Legacy (V1.3)**: nested sections (domain, mechanics, shape, ...),
        mapped to Params fields via explicit translation.
    """
    from gels.engine import Params

    with open(path) as f:
        t = json.load(f)

    params_fields = {f for f in vars(Params()) if not f.startswith('_')}
    # V3.0: renamed fields (W_adh_ff → W_adh_cc, ...) are reachable through
    # alias properties; accept the old names so every existing trial keeps
    # all of its values instead of silently dropping them.
    params_fields |= set(getattr(Params, 'LEGACY_ALIASES', {}))
    is_flat = (t.get("_format") == "flat" or
               any(k in params_fields for k in t if not k.startswith('_')))

    return _load_flat(t, params_fields) if is_flat else _load_legacy(t)


def _load_flat(t, params_fields):
    """Load flat-format Trial JSON: keys map directly to Params fields."""
    overrides = {}
    for k, v in t.items():
        if k.startswith('_'):
            continue  # metadata keys (_name, _description, _format)
        if k in params_fields:
            overrides[k] = v
        else:
            print(f"  Warning: Trial JSON key '{k}' is not a Params field (ignored)")
    return overrides


def _load_legacy(t):
    """Load legacy nested Trial JSON format (V1.3 and earlier)."""
    overrides = {}

    # Domain
    if "domain" in t:
        d = t["domain"]
        if "side_length_um" in d:
            overrides["Lx"] = d["side_length_um"]
            overrides["Ly"] = d["side_length_um"]
        if "Lz_um" in d:
            overrides["Lz"] = d["Lz_um"]

    # Composition - convert functional_fraction to area fraction targets
    if "granule_ratio" in t:
        ff = t["granule_ratio"].get("functional_fraction", 0.5)
        packing = t.get("domain", {}).get("target_packing_fraction", 0.55)
        overrides["phi_f_target"] = ff * packing
        overrides["phi_i_target"] = (1.0 - ff) * packing

    # Functional granules
    if "functional_granules" in t:
        fg = t["functional_granules"]
        if "radius_mean_um" in fg:
            overrides["R_func_mean"] = fg["radius_mean_um"]
        if "radius_std_um" in fg:
            overrides["R_func_std"] = fg["radius_std_um"]

    # Inert granules
    if "inert_granules" in t:
        ig = t["inert_granules"]
        if "radius_mean_um" in ig:
            overrides["R_inert_mean"] = ig["radius_mean_um"]
        if "radius_std_um" in ig:
            overrides["R_inert_std"] = ig["radius_std_um"]

    # Cell properties
    if "cell_properties" in t:
        cp = t["cell_properties"]
        if "diameter_um" in cp:
            overrides["cell_diameter"] = cp["diameter_um"]
        if "attachment_area_fraction" in cp:
            overrides["cell_coverage"] = cp["attachment_area_fraction"]
        if "force_per_cell_nN" in cp:
            overrides["F_max_per_cell"] = cp["force_per_cell_nN"]
        if "max_bridge_gap_um" in cp:
            overrides["cell_sense_distance"] = cp["max_bridge_gap_um"]

    # Mechanics
    if "mechanics" in t:
        m = t["mechanics"]
        if "repulsion_stiffness" in m:
            overrides["E_modulus"] = m["repulsion_stiffness"]
        if "damping" in m:
            overrides["drag_scale"] = m["damping"]

    # Time
    if "time" in t:
        tm = t["time"]
        if "total_hours" in tm:
            overrides["t_total"] = tm["total_hours"]
        if "save_interval_hours" in tm:
            overrides["save_every_h"] = tm["save_interval_hours"]
        if "dt_max_hours" in tm:
            overrides["dt"] = tm["dt_max_hours"]

    # Granule shape (V1.3)
    if "shape" in t:
        s = t["shape"]
        if "enabled" in s:
            overrides["shape_enabled"] = bool(s["enabled"])
        for key in ("aspect_ratio_func_mean", "aspect_ratio_func_std",
                    "aspect_ratio_inert_mean", "aspect_ratio_inert_std",
                    "blockiness_func_mean", "blockiness_func_std",
                    "blockiness_inert_mean", "blockiness_inert_std",
                    "drag_scale_rot", "omega_max",
                    "aspect_ratio_c_func_mean", "aspect_ratio_c_func_std",
                    "aspect_ratio_c_inert_mean", "aspect_ratio_c_inert_std",
                    "blockiness_n2_func_mean", "blockiness_n2_func_std",
                    "blockiness_n2_inert_mean", "blockiness_n2_inert_std"):
            if key in s:
                overrides[key] = s[key]

    # Simulation mode (V1.4)
    if "mode" in t:
        overrides["mode"] = t["mode"]

    # 3D grid resolution (V1.4)
    if "output" in t:
        o = t["output"]
        if "Ngrid_3d" in o:
            overrides["Ngrid_3d"] = o["Ngrid_3d"]

    return overrides
