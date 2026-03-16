#!/usr/bin/env python3
"""
Run all trials in the Trials/ folder.

Run modes (set RUN_MODE below):
  1 = Local   — run each trial sequentially in-process
  2 = HPC     — generate SLURM scripts only (print submit command)
  3 = HPC     — generate SLURM scripts and auto-submit via sbatch
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────
RUN_MODE = 3                        # 1 = Local, 2 = HPC, 3 = HPC (custom config)
TRIALS_DIR = "Trials"               # directory containing trial .json files
OUTPUT_DIR = "results/trials"       # base output directory
SEED = None                         # random seed (None = random each run)
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
    from viz.postprocess import run_all as postprocess

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

        # Post-process: all visualizations from saved data
        postprocess(out_dir)

        # Summary
        h0, hf = hist[0], hist[-1]
        ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
              'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
        print(f"  Result: {h0['func_nc']} -> {hf['func_nc']} clusters ({ft})")
        print(f"  Saved to {out_dir}/")

    print("\n" + "=" * 65)
    print(f"  All {len(trials)} trials complete. Results in {output_base}/")
    print("=" * 65)


def run_hpc(trials: list, output_base: str, seed, hpc_config_path: str):
    """Generate and submit a SLURM array job for all trials."""
    import random as _random
    if seed is None:
        seed = _random.randint(0, 2**31 - 1)
        print(f"  Using random seed: {seed}")
    with open(hpc_config_path) as f:
        hpc = json.load(f)

    repo_path = hpc["repo_path"]

    # Write a trial list file so the array job knows which JSON to use
    trial_list_path = os.path.join("hpc", "trial_list.txt")
    with open(trial_list_path, "w") as f:
        for t in trials:
            f.write(os.path.basename(t) + "\n")
    print(f"Wrote {trial_list_path} ({len(trials)} trials)")

    # ── Estimate walltime from scaling model ──
    # SLURM array jobs share one --time, so use the max across all trials.
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

    # Use estimated walltime if it exceeds the HPC config default
    default_walltime = hpc.get('walltime', '04:00:00')
    if max_est > 0:
        walltime = seconds_to_slurm_time(int(max_est))
        # Clamp to SLURM max (240 hours)
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

    # Expand ~ to $HOME for shell compatibility in SLURM scripts
    venv_path = hpc['venv_path'].replace('~', '$HOME')

    # Generate the array SLURM script
    # NOTE: --cpus-per-task controls memory on Puma (5 GB/CPU).
    #       Do NOT specify both --mem and --cpus-per-task (UA HPC docs).
    slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name=gells-trials
#SBATCH --account={hpc['group']}
#SBATCH --partition={hpc.get('partition', 'standard')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={hpc.get('cpus', 4)}
#SBATCH --time={walltime}
#SBATCH --array=1-{len(trials)}
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load {hpc['python_module']}
source {venv_path}/bin/activate
export MPLBACKEND=Agg

cd {repo_path}

# Read the trial filename for this array task
TRIAL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" hpc/trial_list.txt)
TRIAL_NAME="${{TRIAL%.json}}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial found for array task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== Array task $SLURM_ARRAY_TASK_ID: $TRIAL ==="
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE, Start: $(date)"

python3 run_hpc_headless.py \\
    --trial "Trials/$TRIAL" \\
    --output-dir "{output_base}/$TRIAL_NAME" \\
    --seed {seed}

echo "Task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
"""

    slurm_path = os.path.join("hpc", "run_all_trials.slurm")
    with open(slurm_path, "w") as f:
        f.write(slurm_script)
    print(f"Wrote {slurm_path}")

    print(f"\nSLURM array job: {len(trials)} tasks")
    print(f"Each trial runs independently on its own node.")
    print(f"Results will be in {output_base}/<TrialName>/")

    return slurm_path


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


def _wait_and_sync(netid: str, repo_path: str, job_id: str, n_tasks: int):
    """Poll HPC array job status per-task, sync results when all done.

    Uses `squeue -r` to expand array tasks so we can track individual
    sub-job completion (UA HPC docs: -r flag shows each array element).
    """
    ssh_base = f"{netid}@hpc.arizona.edu"
    poll_interval = 30
    synced_already = False

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

            # Incremental sync: when some tasks have finished, sync partial results
            # Don't delete remote yet — other tasks may still be running
            if n_done > 0 and not synced_already:
                synced_already = True
                print(f"  Syncing {n_done} completed results...")
                _do_sync(netid, repo_path, delete_remote=False)

        except subprocess.TimeoutExpired:
            print(f"  (poll timed out, retrying...)")
        except Exception as e:
            print(f"  (poll error: {e}, retrying...)")

    # Final sync — delete remote files after successful transfer
    print(f"\n  Final sync of all results (will delete remote copies)...")
    _do_sync(netid, repo_path, delete_remote=True)
    print(f"\n  Tip: Check resource efficiency with 'seff {job_id}' on the cluster.")


def _do_sync(netid: str, repo_path: str, delete_remote: bool = False):
    """Rsync results from cluster to local.

    If delete_remote is True, successfully transferred files are deleted
    from the cluster via rsync --remove-source-files, and empty result
    directories are pruned afterwards.
    """
    filexfer = f"{netid}@filexfer.hpc.arizona.edu"
    remote = f"{filexfer}:{repo_path}/results/"
    local = OUTPUT_DIR
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
                     f"find {repo_path}/results -type d -empty -delete 2>/dev/null"],
                    capture_output=True, timeout=30)
                print(f"  Cleaned up remote results.")
        else:
            print(f"  rsync warning: {result.stderr.strip()}")
    except subprocess.TimeoutExpired:
        print(f"  Sync timed out — try: python3 hpc/sync_results.py")
    except Exception as e:
        print(f"  Sync failed: {e}")


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
        slurm_path = run_hpc(trials, OUTPUT_DIR, SEED, HPC_CONFIG)
        print(f"\nTo submit:\n  sbatch {slurm_path}")
    elif RUN_MODE == 3:
        print(f"Mode 3: Generating and submitting HPC SLURM job (config: {HPC_CONFIG})\n")
        slurm_path = run_hpc(trials, OUTPUT_DIR, SEED, HPC_CONFIG)
        # Sync repo to cluster so HPC has latest code + trial configs
        _sync_repo_to_cluster(HPC_CONFIG)
        # Submit via SSH to the cluster
        with open(HPC_CONFIG) as f:
            hpc = json.load(f)
        netid = hpc["netid"]
        repo_path = hpc["repo_path"]
        remote_cmd = f"cd {repo_path} && sbatch {slurm_path}"
        ssh_cmd = ["ssh", f"{netid}@hpc.arizona.edu",
                   f"ssh shell.hpc.arizona.edu '{remote_cmd}'"]
        print(f"\nSubmitting via SSH to {netid}@shell.hpc.arizona.edu...")
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=30)
            if result.stdout.strip():
                print(f"  {result.stdout.strip()}")
            if result.returncode != 0:
                print(f"  sbatch error: {result.stderr.strip()}")
            else:
                # Extract job ID
                job_id = None
                for word in result.stdout.strip().split():
                    if word.isdigit():
                        job_id = word
                        break

                if job_id:
                    print(f"\n  Waiting for job {job_id}... (Ctrl+C to stop waiting)")
                    print(f"  You can always sync later: python3 hpc/sync_results.py\n")
                    try:
                        _wait_and_sync(netid, repo_path, job_id, len(trials))
                    except KeyboardInterrupt:
                        print(f"\n\n  Stopped waiting. Job {job_id} is still running on the cluster.")
                        print(f"  Sync when ready:  python3 hpc/sync_results.py")
                else:
                    print(f"\n  Sync when ready:  python3 hpc/sync_results.py")
        except FileNotFoundError:
            print("  ssh not found.")
        except subprocess.TimeoutExpired:
            print("  SSH timed out — check VPN connection.")
    else:
        print(f"Invalid RUN_MODE={RUN_MODE}. Set to 1, 2, or 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
