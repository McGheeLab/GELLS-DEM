#!/usr/bin/env python3
"""
Run all trials in the Trials/ folder.

Run modes (set RUN_MODE below):
  1 = Local   — run each trial sequentially in-process
  2 = HPC     — generate SLURM scripts only (print submit command)
  3 = HPC     — generate SLURM scripts and auto-submit via sbatch
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────
RUN_MODE = 3                        # 1 = Local, 2 = HPC (generate scripts), 3 = HPC (auto-submit)
TRIALS_DIR = "Trials"               # directory containing trial .json files
OUTPUT_DIR = "results/LHC2"          # base output directory
SEED = 42                           # random seed (None = random each run)
HPC_CONFIG = "hpc/Alex.json"       # HPC user config (used by modes 2 and 3)
# ────────────────────────────────────────────────────────────────────

import glob
import json
import math
import os
import re
import subprocess
import sys
import time


# ── Wall-time estimation model ────────────────────────────────────
# Power-law fit from L-scaling benchmarks (Trial22–27, V1.6, Numba JIT,
# Puma 4-CPU, 3D superellipsoid with shape_enabled=True):
#
#   T_step ≈ 0.003 × N^1.5  seconds per timestep
#
# Calibration data (20 steps each, save_fields=True, Ngrid_3d≈80–200):
#   N=  180 → T=  205s  (model:  205s,  -0.1%)
#   N=  445 → T=  671s  (model:  623s,  -7.1%)
#   N=  912 → T= 1673s  (model: 1713s,  +2.4%)
#   N= 1540 → T= 2849s  (model: 3686s, +29.4%)
#
# Accurate to ±10% for N=100–1000, conservative for N>1000.
# RSA typically achieves 85–95% of target packing for phi_total=0.4–0.7.
_COEFF = 0.003          # power-law coefficient (seconds)
_EXPONENT = 1.5         # power-law exponent (super-linear due to contact detection)
_RSA_EFFICIENCY = 0.85  # fraction of target granules placed by RSA
_OVERHEAD_S = 60        # fixed overhead: JIT compile + packing settle + archive
_SAFETY_FACTOR = 2.0    # multiply predicted time for SLURM --time


def estimate_walltime(trial_path: str) -> tuple:
    """Estimate wall-clock time (seconds) for a trial from its JSON config.

    Uses a power-law model calibrated on L-scaling benchmarks:
        T = 0.003 × N^1.5 × n_steps + overhead

    Returns (estimated_seconds, n_granules_est, n_steps).
    """
    with open(trial_path) as f:
        t = json.load(f)

    # Read params (flat format keys map directly to Params fields)
    mode = t.get("mode", "2D")
    Lx = float(t.get("Lx", 800))
    Ly = float(t.get("Ly", 800))
    Lz = float(t.get("Lz", 800))
    R_f = float(t.get("R_func_mean", 40))
    R_i = float(t.get("R_inert_mean", 60))
    # Support both phi_f/phi_i_target and phi_solid_target + func_ratio
    phi_solid = float(t.get("phi_solid_target", 0))
    if phi_solid > 0:
        func_ratio = float(t.get("func_ratio", 0.5))
        phi_f = phi_solid * func_ratio
        phi_i = phi_solid * (1.0 - func_ratio)
    else:
        phi_f = float(t.get("phi_f_target", 0.25))
        phi_i = float(t.get("phi_i_target", 0.20))
    dt = float(t.get("dt", 0.1))
    t_total = float(t.get("t_total", 48))

    n_steps = int(round(t_total / dt))

    if mode == "3D" or mode == "2D-slice":
        domain_vol = Lx * Ly * Lz
        vol_f = (4.0 / 3.0) * math.pi * R_f ** 3
        vol_i = (4.0 / 3.0) * math.pi * R_i ** 3
    else:
        domain_vol = Lx * Ly
        vol_f = math.pi * R_f ** 2
        vol_i = math.pi * R_i ** 2

    n_target = phi_f * domain_vol / vol_f
    if phi_i > 0:
        n_target += phi_i * domain_vol / vol_i
    n_target = int(round(n_target))
    n_est = max(1, int(round(n_target * _RSA_EFFICIENCY)))

    # Power-law model: T_step = 0.003 * N^1.5
    t_sim = _COEFF * n_est ** _EXPONENT * n_steps
    t_total_est = t_sim + _OVERHEAD_S

    return t_total_est, n_est, n_steps


def seconds_to_slurm_time(seconds: int) -> str:
    """Convert seconds to HH:MM:SS format for SLURM --time."""
    seconds = max(seconds, 60)  # minimum 1 minute
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def find_trials(trials_dir: str) -> list:
    """Find all .json files in the trials directory, sorted naturally."""
    pattern = os.path.join(trials_dir, "*.json")
    files = sorted(glob.glob(pattern), key=natural_sort_key)
    return files


def natural_sort_key(path: str):
    """Sort Trial1, Trial2, ... Trial10, Trial11 in natural order."""
    name = os.path.basename(path)
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', name)]


def run_local(trials: list, output_base: str, seed=None):
    """Run all trials sequentially in the current process."""
    from run_hpc_headless import load_trial_json
    from new_dem_0 import Params, run, print_stiffness_info

    os.makedirs(output_base, exist_ok=True)

    for i, trial_path in enumerate(trials, 1):
        trial_name = os.path.splitext(os.path.basename(trial_path))[0]
        out_dir = os.path.join(output_base, trial_name)
        os.makedirs(out_dir, exist_ok=True)

        print("\n" + "=" * 65)
        print(f"  [{i}/{len(trials)}] {trial_name}")
        print("=" * 65)

        # Build Params from trial JSON
        p = Params()
        overrides = load_trial_json(trial_path)
        for k, v in overrides.items():
            if hasattr(p, k):
                setattr(p, k, type(getattr(p, k))(v))

        # V1.5: Wire output_dir for data serialization
        p.output_dir = out_dir

        mode = getattr(p, 'mode', '2D')
        if mode == '3D' or mode == '2D-slice':
            print(f"  Mode: {mode}")
            print(f"  Domain: {p.Lx:.0f} x {p.Ly:.0f} x {p.Lz:.0f} um")
        else:
            print(f"  Domain: {p.Lx:.0f} x {p.Ly:.0f} um")
        print(f"  E_modulus={p.E_modulus} kPa, t_total={p.t_total} h")
        print_stiffness_info(p)

        # Run simulation (data saved to out_dir by engine)
        hist, _, p, _ = run(p, seed=seed)

        # Post-process: viz2 (scaffold maps, Voronoi, phase fractions, etc.)
        from viz2 import run_all as postprocess_v2
        postprocess_v2(run_dir=out_dir)

        # Summary
        h0, hf = hist[0], hist[-1]
        ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
              'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
        print(f"  Result: {h0['func_nc']} -> {hf['func_nc']} clusters ({ft})")
        print(f"  Saved to {out_dir}/")

    print("\n" + "=" * 65)
    print(f"  All {len(trials)} trials complete. Results in {output_base}/")
    print("=" * 65)


_MAX_ARRAY_SIZE = 200   # Tasks per batch (~10 batches for 2000 trials)
_MAX_CONCURRENT = 200   # Concurrent array tasks: 200 × 4 CPUs = 800 CPUs (within 3290 limit)


def run_hpc(trials: list, output_base: str, seed, hpc_config_path: str):
    """Generate SLURM array job scripts for all trials.

    If len(trials) > _MAX_ARRAY_SIZE, splits into multiple batch scripts
    (batch_0, batch_1, ...) each with ≤1000 tasks. Each batch uses an
    OFFSET variable so the trial_list.txt line lookup works across batches.
    Concurrency throttled via --array=%N to stay within group CPU limits.

    Returns list of (slurm_path, n_tasks, offset) tuples.
    """
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31 - 1)
        print(f"  Using random seed: {seed}")
    with open(hpc_config_path) as f:
        hpc = json.load(f)

    repo_path = hpc["repo_path"]
    results_path = hpc.get("results_path", f"{repo_path}/results")
    max_concurrent = hpc.get("max_concurrent", _MAX_CONCURRENT)

    # Write a single trial list file (all batches share it via OFFSET)
    trial_list_path = os.path.join("hpc", "trial_list.txt")
    with open(trial_list_path, "w") as f:
        for t in trials:
            f.write(os.path.basename(t) + "\n")
    print(f"Wrote {trial_list_path} ({len(trials)} trials)")

    # ── Estimate walltime from scaling model ──
    max_est = 0
    print("\n  Walltime estimates (scaling model):")
    for t in trials:
        name = os.path.splitext(os.path.basename(t))[0]
        try:
            est_s, n_est, n_steps = estimate_walltime(t)
            est_safe = est_s * _SAFETY_FACTOR
            max_est = max(max_est, est_safe)
            print(f"    {name:30s}  ~{n_est:>6d} granules  "
                  f"{n_steps:>5d} steps  "
                  f"est {est_s/60:>6.1f} min  "
                  f"(x{_SAFETY_FACTOR:.0f} = {est_safe/60:.0f} min)")
        except Exception as e:
            print(f"    {name:30s}  estimate failed: {e}")

    default_walltime = hpc.get('walltime', '04:00:00')
    if max_est > 0:
        walltime = seconds_to_slurm_time(int(max_est))
        max_slurm = 240 * 3600
        if max_est > max_slurm:
            walltime = "240:00:00"
            print(f"\n  WARNING: Estimated time ({max_est/3600:.1f}h) exceeds "
                  f"SLURM max (240h). Clamped to 240:00:00.")
        print(f"\n  Using walltime: {walltime} "
              f"(max estimate x{_SAFETY_FACTOR:.0f} safety factor)")
    else:
        walltime = default_walltime
        print(f"\n  Using default walltime: {walltime}")

    venv_path = hpc['venv_path'].replace('~', '$HOME')

    # ── Split into batches of _MAX_ARRAY_SIZE ──
    n_batches = math.ceil(len(trials) / _MAX_ARRAY_SIZE)
    batch_info = []  # list of (slurm_path, n_tasks, offset)

    for batch_idx in range(n_batches):
        offset = batch_idx * _MAX_ARRAY_SIZE
        batch_size = min(_MAX_ARRAY_SIZE, len(trials) - offset)

        # Throttle concurrent tasks: --array=1-N%M limits to M running at once.
        # Default 10 concurrent × 4 CPUs/task = 40 CPUs (configurable via max_concurrent).
        array_spec = f"1-{batch_size}%{max_concurrent}"

        slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name=gels-b{batch_idx}
#SBATCH --account={hpc['group']}
#SBATCH --partition={hpc.get('partition', 'standard')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={hpc.get('cpus', 4)}
#SBATCH --time={walltime}
#SBATCH --array={array_spec}
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load {hpc['python_module']}
source {venv_path}/bin/activate
export MPLBACKEND=Agg

cd {repo_path}

# V2.6: Results written to /groups (500 GB) instead of /home (50 GB)
RESULTS_BASE="{results_path}/LHC"
mkdir -p "$RESULTS_BASE"

# Pre-flight disk space check (abort if < 2 GB free)
AVAIL_KB=$(df --output=avail "$RESULTS_BASE" 2>/dev/null | tail -1)
AVAIL_GB=$(( ${{AVAIL_KB:-0}} / 1048576 ))
if [ "$AVAIL_GB" -lt 2 ]; then
    echo "ABORT: Only ${{AVAIL_GB}} GB free on results filesystem. Need >= 2 GB."
    exit 1
fi

# Stagger starts to reduce I/O storms (0-30 second random delay)
SLEEP_SEC=$(( RANDOM % 30 ))
echo "  Stagger sleep: ${{SLEEP_SEC}}s"
sleep $SLEEP_SEC

# Offset into trial_list.txt for this batch
OFFSET={offset}
LINE_NUM=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

TRIAL=$(sed -n "${{LINE_NUM}}p" hpc/trial_list.txt)
TRIAL_NAME="${{TRIAL%.json}}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial found for line $LINE_NUM (task $SLURM_ARRAY_TASK_ID, offset $OFFSET)"
    exit 1
fi

echo "=== Batch {batch_idx} task $SLURM_ARRAY_TASK_ID (line $LINE_NUM): $TRIAL ==="
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE, Start: $(date)"
echo "Results: $RESULTS_BASE/$TRIAL_NAME"
echo "Free disk: ${{AVAIL_GB}} GB"

python3 run_hpc_headless.py \\
    --trial "Trials/$TRIAL" \\
    --output-dir "$RESULTS_BASE/$TRIAL_NAME" \\
    --save_fields False \\
    --seed {seed}

echo "Task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
"""

        if n_batches == 1:
            slurm_path = os.path.join("hpc", "run_all_trials.slurm")
        else:
            slurm_path = os.path.join("hpc", f"run_all_trials_batch{batch_idx}.slurm")

        with open(slurm_path, "w") as f:
            f.write(slurm_script)

        batch_info.append((slurm_path, batch_size, offset))
        print(f"Wrote {slurm_path} (tasks {offset+1}–{offset+batch_size}, "
              f"max {max_concurrent} concurrent)")

    print(f"\n{n_batches} SLURM batch(es), {len(trials)} total tasks")
    print(f"Results will be in {output_base}/<TrialName>/")

    return batch_info


def _sync_repo_to_cluster(hpc_config_path: str):
    """Rsync local repo to cluster so HPC has latest code and trial configs."""
    with open(hpc_config_path) as f:
        hpc = json.load(f)
    netid = hpc["netid"]
    repo_path = hpc["repo_path"]
    filexfer = f"{netid}@filexfer.hpc.arizona.edu"
    local_repo = os.path.dirname(os.path.abspath(__file__))

    print("  Syncing repo to cluster...")
    # Ensure remote directories exist (filexfer has shared storage access)
    subprocess.run(
        ["ssh", filexfer, f"mkdir -p {repo_path} {repo_path}/slurm_logs"],
        capture_output=True, timeout=30)
    # Rsync via filexfer (UA HPC docs: always use filexfer for transfers)
    # Exclude large data directories — only code + trial configs are needed.
    result = subprocess.run([
        "rsync", "-ravz", "--delete",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "*.pyc",
        "--exclude", "results/",
        "--exclude", "simulations/",
        "--exclude", "slurm_logs/",
        "--exclude", "Trials/trials/",
        "--exclude", "old/",
        "--exclude", "*.tar.gz",
        local_repo + "/",
        f"{filexfer}:{repo_path}/"
    ], capture_output=True, text=True, timeout=300)
    if result.returncode == 0:
        print("  Repo synced.")
    else:
        print(f"  rsync warning: {result.stderr.strip()}")


def _wait_and_sync(netid: str, repo_path: str, job_id: str, n_tasks: int,
                   results_path: str = None):
    """Poll HPC array job status per-task, continuously sync and delete results.

    Uses `squeue -r` to expand array tasks so we can track individual
    sub-job completion (UA HPC docs: -r flag shows each array element).
    Every poll cycle, completed results are transferred to the local machine
    and deleted from the cluster to free HPC storage.
    """
    ssh_base = f"{netid}@hpc.arizona.edu"
    poll_interval = 60
    last_synced_done = 0        # track how many were done at last sync
    sync_every_n_new = 10       # sync after every N newly completed tasks
    last_sync_time = 0          # epoch time of last sync

    while True:
        time.sleep(poll_interval)
        try:
            # Use -r to expand array tasks into individual rows
            r = subprocess.run(
                ["ssh", ssh_base,
                 f"ssh shell.hpc.arizona.edu 'squeue -r -j {job_id} -h 2>/dev/null'"],
                capture_output=True, text=True, timeout=30)
            output = r.stdout.strip()
            if not output:
                print(f"\n  All {n_tasks} tasks complete!")
                break

            # Parse individual array task lines
            lines = output.strip().splitlines()
            running = []
            pending = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 6:
                    task_id = parts[0]   # e.g. "21470754_1"
                    state = parts[4]     # R, PD, CG, etc.
                    elapsed = parts[5]
                    if state == "R":
                        running.append((task_id, elapsed))
                    elif state == "PD":
                        pending.append(task_id)

            n_active = len(running) + len(pending)
            n_done = n_tasks - n_active

            status_parts = []
            if n_done > 0:
                status_parts.append(f"{n_done} done")
            if running:
                status_parts.append(f"{len(running)} running")
            if pending:
                status_parts.append(f"{len(pending)} pending")

            # Show elapsed time from longest-running task
            elapsed_str = ""
            if running:
                elapsed_str = f" ({running[0][1]})"

            print(f"  [{', '.join(status_parts)}]{elapsed_str}")

            # Incremental sync+delete: transfer completed results and remove
            # from HPC every time N new tasks finish, or every 10 minutes.
            newly_done = n_done - last_synced_done
            time_since_sync = time.time() - last_sync_time
            if newly_done >= sync_every_n_new or (newly_done > 0 and time_since_sync > 600):
                print(f"  Syncing {newly_done} new results (delete from HPC)...")
                _do_sync(netid, repo_path, delete_remote=True,
                         results_path=results_path)
                last_synced_done = n_done
                last_sync_time = time.time()

        except subprocess.TimeoutExpired:
            print(f"  (poll timed out, retrying...)")
        except Exception as e:
            print(f"  (poll error: {e}, retrying...)")

    # Final sync — catch any stragglers
    print(f"\n  Final sync of all results (delete from HPC)...")
    _do_sync(netid, repo_path, delete_remote=True, results_path=results_path)
    print(f"\n  Tip: Check resource efficiency with 'seff {job_id}' on the cluster.")


def _wait_and_sync_multi(netid: str, repo_path: str, job_ids: list, n_tasks: int,
                         results_path: str = None):
    """Poll multiple SLURM array jobs, continuously sync and delete results.

    Monitors all job IDs together. When all queues are empty, we're done.
    """
    ssh_base = f"{netid}@hpc.arizona.edu"
    poll_interval = 60
    last_synced_done = 0
    sync_every_n_new = 10
    last_sync_time = 0
    jobs_csv = ",".join(job_ids)

    while True:
        time.sleep(poll_interval)
        try:
            r = subprocess.run(
                ["ssh", ssh_base,
                 f"ssh shell.hpc.arizona.edu 'squeue -r -j {jobs_csv} -h 2>/dev/null'"],
                capture_output=True, text=True, timeout=30)
            output = r.stdout.strip()
            if not output:
                print(f"\n  All {n_tasks} tasks complete!")
                break

            lines = output.strip().splitlines()
            running = []
            pending = []
            for line in lines:
                parts = line.split()
                if len(parts) >= 6:
                    task_id = parts[0]
                    state = parts[4]
                    elapsed = parts[5]
                    if state == "R":
                        running.append((task_id, elapsed))
                    elif state == "PD":
                        pending.append(task_id)

            n_active = len(running) + len(pending)
            n_done = n_tasks - n_active

            status_parts = []
            if n_done > 0:
                status_parts.append(f"{n_done} done")
            if running:
                status_parts.append(f"{len(running)} running")
            if pending:
                status_parts.append(f"{len(pending)} pending")

            elapsed_str = ""
            if running:
                elapsed_str = f" ({running[0][1]})"

            print(f"  [{', '.join(status_parts)}]{elapsed_str}")

            newly_done = n_done - last_synced_done
            time_since_sync = time.time() - last_sync_time
            if newly_done >= sync_every_n_new or (newly_done > 0 and time_since_sync > 600):
                print(f"  Syncing {newly_done} new results (delete from HPC)...")
                _do_sync(netid, repo_path, delete_remote=True,
                         results_path=results_path)
                last_synced_done = n_done
                last_sync_time = time.time()

        except subprocess.TimeoutExpired:
            print(f"  (poll timed out, retrying...)")
        except Exception as e:
            print(f"  (poll error: {e}, retrying...)")

    print(f"\n  Final sync of all results (delete from HPC)...")
    _do_sync(netid, repo_path, delete_remote=True, results_path=results_path)
    print(f"\n  Tip: Check resource efficiency with 'seff {job_ids[0]}' on the cluster.")


def _do_sync(netid: str, repo_path: str, delete_remote: bool = False,
             results_path: str = None):
    """Rsync results from cluster to local.

    If delete_remote is True, successfully transferred files are deleted
    from the cluster via rsync --remove-source-files, and empty result
    directories are pruned afterwards.

    V2.6: results_path allows reading from /groups instead of /home.
    """
    filexfer = f"{netid}@filexfer.hpc.arizona.edu"
    # Remote results: use results_path if provided, else fall back to repo_path/results/
    remote_base = results_path if results_path else f"{repo_path}/results"
    remote = f"{filexfer}:{remote_base}/"
    local = "results"
    os.makedirs(local, exist_ok=True)
    rsync_cmd = ["rsync", "-avz", remote, f"{local}/"]
    if delete_remote:
        rsync_cmd.insert(2, "--remove-source-files")
    try:
        result = subprocess.run(
            rsync_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Count transferred files
            lines = result.stdout.strip().splitlines()
            n_files = sum(1 for l in lines if not l.endswith('/') and
                         not l.startswith('sent ') and not l.startswith('total ') and
                         not l.startswith('receiving'))
            print(f"  Synced to {os.path.abspath(local)}/ ({n_files} files)")
            if delete_remote and n_files > 0:
                # Prune empty directories left behind by --remove-source-files
                subprocess.run(
                    ["ssh", filexfer,
                     f"find {remote_base} -type d -empty -delete 2>/dev/null"],
                    capture_output=True, timeout=30)
                print(f"  Cleaned up remote results.")
        else:
            print(f"  rsync warning: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"  Sync timed out — try: python3 hpc/sync_results.py")
    except Exception as e:
        print(f"  Sync failed: {e}")


# ══════════════════════════════════════════════════════════════════════
# V2.6: Resume incomplete trials
# ══════════════════════════════════════════════════════════════════════

def find_incomplete_trials(local_results_dir: str, trials_dir: str) -> list:
    """Scan local results for incomplete trials (last history time < t_total).

    Returns list of (trial_json_path, local_run_dir, t_last, t_total) tuples.
    """
    incomplete = []
    pattern = os.path.join(trials_dir, "*.json")
    trial_files = sorted(glob.glob(pattern), key=natural_sort_key)

    for trial_path in trial_files:
        trial_name = os.path.splitext(os.path.basename(trial_path))[0]
        run_dir = os.path.join(local_results_dir, "LHC", trial_name)

        # Check if results directory exists with snapshots
        snap_dir = os.path.join(run_dir, 'snapshots')
        if not os.path.isdir(snap_dir):
            continue

        # Load params to get t_total
        params_path = os.path.join(run_dir, 'params.json')
        if not os.path.exists(params_path):
            continue
        with open(params_path) as f:
            params = json.load(f)
        t_total = params.get('t_total', 72.0)

        # Find last snapshot time
        snap_files = sorted(
            f for f in os.listdir(snap_dir)
            if f.startswith('snap_') and f.endswith('.npz'))
        if not snap_files:
            continue

        # Read time from last snapshot
        import numpy as np
        last_snap = os.path.join(snap_dir, snap_files[-1])
        data = dict(np.load(last_snap, allow_pickle=False))
        t_last = float(data['time'])

        # Consider incomplete if more than 1 timestep remaining
        dt = params.get('dt', 0.5)
        if t_last < t_total - dt:
            incomplete.append((trial_path, run_dir, t_last, t_total))

    return incomplete


def upload_for_resume(hpc_config_path: str, local_run_dir: str,
                      remote_run_dir: str):
    """Upload the last snapshot + metadata to HPC for resuming a trial.

    Only uploads the minimal files needed for resume:
    - params.json, metadata.json, history.json
    - snapshots/snap_XXXX.npz (the last one only)
    """
    with open(hpc_config_path) as f:
        hpc = json.load(f)
    netid = hpc["netid"]
    filexfer = f"{netid}@filexfer.hpc.arizona.edu"

    # Find last snapshot locally
    snap_dir = os.path.join(local_run_dir, 'snapshots')
    snap_files = sorted(
        f for f in os.listdir(snap_dir)
        if f.startswith('snap_') and f.endswith('.npz'))
    if not snap_files:
        print(f"  ERROR: No snapshots in {local_run_dir}")
        return False
    last_snap = snap_files[-1]

    # Create remote directories
    subprocess.run(
        ["ssh", filexfer, f"mkdir -p {remote_run_dir}/snapshots"],
        capture_output=True, timeout=30)

    # Upload each file
    files_to_upload = [
        (os.path.join(local_run_dir, 'params.json'), f"{remote_run_dir}/params.json"),
        (os.path.join(local_run_dir, 'metadata.json'), f"{remote_run_dir}/metadata.json"),
        (os.path.join(local_run_dir, 'history.json'), f"{remote_run_dir}/history.json"),
        (os.path.join(snap_dir, last_snap), f"{remote_run_dir}/snapshots/{last_snap}"),
    ]
    for local_file, remote_file in files_to_upload:
        if os.path.exists(local_file):
            result = subprocess.run(
                ["scp", local_file, f"{filexfer}:{remote_file}"],
                capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                print(f"  ERROR uploading {local_file}: {result.stderr.strip()}")
                return False

    print(f"  Uploaded resume data to {remote_run_dir}/ ({last_snap})")
    return True


def run_resume(hpc_config_path: str, local_results_dir: str,
               trials_dir: str, seed):
    """Mode 4: Find incomplete trials, upload snapshots, and submit resume jobs.

    Returns list of (slurm_path, n_tasks, offset) tuples.
    """
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31 - 1)

    with open(hpc_config_path) as f:
        hpc = json.load(f)
    repo_path = hpc["repo_path"]
    results_path = hpc.get("results_path", f"{repo_path}/results")
    max_concurrent = hpc.get("max_concurrent", _MAX_CONCURRENT)
    venv_path = hpc['venv_path'].replace('~', '$HOME')

    # Find incomplete trials
    incomplete = find_incomplete_trials(local_results_dir, trials_dir)
    if not incomplete:
        print("  No incomplete trials found.")
        return []

    print(f"  Found {len(incomplete)} incomplete trials:")
    for trial_path, run_dir, t_last, t_total in incomplete:
        name = os.path.splitext(os.path.basename(trial_path))[0]
        print(f"    {name:30s}  t={t_last:.1f}/{t_total:.1f} h "
              f"({t_last/t_total:.0%} complete)")

    # Upload resume data for each trial
    print(f"\n  Uploading resume snapshots to {results_path}/LHC/...")
    for trial_path, run_dir, t_last, t_total in incomplete:
        trial_name = os.path.splitext(os.path.basename(trial_path))[0]
        remote_run_dir = f"{results_path}/LHC/{trial_name}"
        upload_for_resume(hpc_config_path, run_dir, remote_run_dir)

    # Write trial list for resume batch
    trial_names = []
    for trial_path, _, _, _ in incomplete:
        trial_names.append(os.path.basename(trial_path))

    trial_list_path = os.path.join("hpc", "trial_list.txt")
    with open(trial_list_path, "w") as f:
        for name in trial_names:
            f.write(name + "\n")
    print(f"\n  Wrote {trial_list_path} ({len(trial_names)} resume trials)")

    # Estimate walltime from remaining time (not full run)
    max_est = 0
    for trial_path, _, t_last, t_total in incomplete:
        try:
            est_s, n_est, n_steps_full = estimate_walltime(trial_path)
            # Scale by remaining fraction
            remaining_frac = (t_total - t_last) / t_total
            est_remaining = est_s * remaining_frac * _SAFETY_FACTOR
            max_est = max(max_est, est_remaining)
        except Exception:
            pass

    if max_est > 0:
        walltime = seconds_to_slurm_time(int(max_est))
        max_slurm = 240 * 3600
        if max_est > max_slurm:
            walltime = "240:00:00"
        print(f"  Walltime: {walltime} (remaining fraction x{_SAFETY_FACTOR:.0f})")
    else:
        walltime = hpc.get('walltime', '04:00:00')
        print(f"  Using default walltime: {walltime}")

    # Generate SLURM resume script
    n_tasks = len(incomplete)
    array_spec = f"1-{n_tasks}%{max_concurrent}"

    slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name=gels-resume
#SBATCH --account={hpc['group']}
#SBATCH --partition={hpc.get('partition', 'standard')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={hpc.get('cpus', 4)}
#SBATCH --time={walltime}
#SBATCH --array={array_spec}
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load {hpc['python_module']}
source {venv_path}/bin/activate
export MPLBACKEND=Agg

cd {repo_path}

# V2.6: Results in /groups
RESULTS_BASE="{results_path}/LHC"

# Pre-flight disk space check
AVAIL_KB=$(df --output=avail "$RESULTS_BASE" 2>/dev/null | tail -1)
AVAIL_GB=$(( ${{AVAIL_KB:-0}} / 1048576 ))
if [ "$AVAIL_GB" -lt 2 ]; then
    echo "ABORT: Only ${{AVAIL_GB}} GB free. Need >= 2 GB."
    exit 1
fi

# Stagger starts
sleep $(( RANDOM % 30 ))

TRIAL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" hpc/trial_list.txt)
TRIAL_NAME="${{TRIAL%.json}}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== RESUME task $SLURM_ARRAY_TASK_ID: $TRIAL ==="
echo "Node: $(hostname), Start: $(date)"

python3 run_hpc_headless.py \\
    --trial "Trials/$TRIAL" \\
    --output-dir "$RESULTS_BASE/$TRIAL_NAME" \\
    --resume-from "$RESULTS_BASE/$TRIAL_NAME" \\
    --save_fields False \\
    --seed {seed}

echo "Resume task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
"""

    slurm_path = os.path.join("hpc", "run_resume.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)
    print(f"  Wrote {slurm_path} ({n_tasks} tasks, max {max_concurrent} concurrent)")

    return [(slurm_path, n_tasks, 0)]


def main():
    trials = find_trials(TRIALS_DIR)
    if not trials:
        print(f"No .json files found in {TRIALS_DIR}/")
        sys.exit(1)

    print(f"Found {len(trials)} trials:")
    for t in trials:
        print(f"  {os.path.basename(t)}")
    print()

    if RUN_MODE == 1:
        print("Mode 1: Running locally (sequential)\n")
        run_local(trials, OUTPUT_DIR, SEED)
    elif RUN_MODE == 2:
        print(f"Mode 2: Generating HPC SLURM job (config: {HPC_CONFIG})\n")
        batch_info = run_hpc(trials, OUTPUT_DIR, SEED, HPC_CONFIG)
        print("\nTo submit:")
        for slurm_path, _, _ in batch_info:
            print(f"  sbatch {slurm_path}")
    elif RUN_MODE == 3:
        print(f"Mode 3: Generating and submitting HPC SLURM job (config: {HPC_CONFIG})\n")
        batch_info = run_hpc(trials, OUTPUT_DIR, SEED, HPC_CONFIG)
        _sync_repo_to_cluster(HPC_CONFIG)

        with open(HPC_CONFIG) as f:
            hpc = json.load(f)
        netid = hpc["netid"]
        repo_path = hpc["repo_path"]
        results_path = hpc.get("results_path", f"{repo_path}/results")

        # ── Submit in waves ──
        # UA HPC QOSMaxSubmitJobPerUserLimit = 1000.
        # Each batch has _MAX_ARRAY_SIZE tasks. Submit up to
        # wave_size batches at once (wave_size * _MAX_ARRAY_SIZE ≤ 1000),
        # wait for the wave to finish, then submit the next wave.
        wave_size = max(1, 1000 // _MAX_ARRAY_SIZE)  # batches per wave
        total_submitted = 0

        try:
            for wave_start in range(0, len(batch_info), wave_size):
                wave = batch_info[wave_start:wave_start + wave_size]
                wave_num = wave_start // wave_size + 1
                n_waves = math.ceil(len(batch_info) / wave_size)

                print(f"\n{'='*60}")
                print(f"  Wave {wave_num}/{n_waves}: submitting {len(wave)} batch(es)")
                print(f"{'='*60}")

                job_ids = []
                wave_tasks = 0
                for slurm_path, n_tasks, offset in wave:
                    remote_cmd = f"cd {repo_path} && sbatch {slurm_path}"
                    ssh_cmd = ["ssh", f"{netid}@hpc.arizona.edu",
                               f"ssh shell.hpc.arizona.edu '{remote_cmd}'"]
                    print(f"  Submitting {slurm_path} ({n_tasks} tasks)...")
                    try:
                        result = subprocess.run(
                            ssh_cmd, capture_output=True, text=True, timeout=30)
                        if result.stdout.strip():
                            print(f"    {result.stdout.strip()}")
                        if result.returncode != 0:
                            print(f"    sbatch error: {result.stderr.strip()}")
                            continue
                        for word in result.stdout.strip().split():
                            if word.isdigit():
                                job_ids.append(word)
                                wave_tasks += n_tasks
                                break
                    except FileNotFoundError:
                        print("    ssh not found.")
                        break
                    except subprocess.TimeoutExpired:
                        print("    SSH timed out — check VPN connection.")
                        break

                if not job_ids:
                    print("  No jobs submitted in this wave. Stopping.")
                    break

                total_submitted += wave_tasks
                print(f"\n  Wave {wave_num}: {wave_tasks} tasks submitted "
                      f"({total_submitted}/{len(trials)} total)")
                print(f"  Job IDs: {', '.join(job_ids)}")
                print(f"  Waiting for wave to complete...\n")

                _wait_and_sync_multi(netid, repo_path, job_ids, wave_tasks,
                                     results_path=results_path)

                print(f"\n  Wave {wave_num} complete. "
                      f"Progress: {total_submitted}/{len(trials)} tasks done.")

        except KeyboardInterrupt:
            print(f"\n\n  Stopped. {total_submitted} tasks submitted so far.")
            print(f"  Running jobs will continue on the cluster.")
            print(f"  Sync when ready:  python3 hpc/sync_results.py")

        if total_submitted == len(trials):
            print(f"\n{'='*60}")
            print(f"  All {len(trials)} trials complete!")
            print(f"{'='*60}")
    elif RUN_MODE == 4:
        print(f"Mode 4: Resume incomplete trials (config: {HPC_CONFIG})\n")
        batch_info = run_resume(HPC_CONFIG, "results", TRIALS_DIR, SEED)
        if not batch_info:
            return
        _sync_repo_to_cluster(HPC_CONFIG)

        with open(HPC_CONFIG) as f:
            hpc = json.load(f)
        netid = hpc["netid"]
        repo_path = hpc["repo_path"]
        results_path = hpc.get("results_path", f"{repo_path}/results")

        # Submit the resume batch
        for slurm_path, n_tasks, offset in batch_info:
            remote_cmd = f"cd {repo_path} && sbatch {slurm_path}"
            ssh_cmd = ["ssh", f"{netid}@hpc.arizona.edu",
                       f"ssh shell.hpc.arizona.edu '{remote_cmd}'"]
            print(f"  Submitting {slurm_path} ({n_tasks} tasks)...")
            try:
                result = subprocess.run(
                    ssh_cmd, capture_output=True, text=True, timeout=30)
                if result.stdout.strip():
                    print(f"    {result.stdout.strip()}")
                if result.returncode != 0:
                    print(f"    sbatch error: {result.stderr.strip()}")
                    return
                for word in result.stdout.strip().split():
                    if word.isdigit():
                        job_id = word
                        break
                else:
                    print("    Could not parse job ID.")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"    Submit failed: {e}")
                return

        print(f"\n  Resume job submitted (ID: {job_id}). Waiting for completion...\n")
        _wait_and_sync(netid, repo_path, job_id, n_tasks,
                       results_path=results_path)
        print(f"\n  Resume complete!")
    else:
        print(f"Invalid RUN_MODE={RUN_MODE}. Set to 1, 2, 3, or 4.")
        sys.exit(1)


if __name__ == "__main__":
    main()
