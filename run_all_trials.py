#!/usr/bin/env python3
"""
Run all trials in the Trials/ folder.

Works both locally and on HPC:
  - Local:  python3 run_all_trials.py
  - HPC:    python3 run_all_trials.py --hpc
  - HPC:    python3 run_all_trials.py --hpc --config hpc/Alex.json

Local mode runs each trial sequentially in-process.
HPC mode generates and submits a SLURM array job (one task per trial).
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys


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


def run_local(trials: list, output_base: str, seed: int):
    """Run all trials sequentially in the current process."""
    # Import here so matplotlib backend is set
    import matplotlib
    matplotlib.use('Agg')

    from run_hpc_headless import load_trial_json
    from new_dem_0 import (
        Params, run, plot_granules, plot_fields,
        plot_timeseries, plot_composite, print_stiffness_info
    )
    import matplotlib.pyplot as plt

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

        print(f"  Domain: {p.Lx:.0f} x {p.Ly:.0f} um")
        print(f"  E_modulus={p.E_modulus} kPa, t_total={p.t_total} h")
        print_stiffness_info(p)

        # Run
        hist, snaps, p, gs = run(p, seed=seed)

        # Save figures
        for plot_fn, fname in [
            (plot_granules, "granules.png"),
            (plot_fields, "fields.png"),
            (plot_timeseries, "timeseries.png"),
            (plot_composite, "composite.png"),
        ]:
            if fname == "timeseries.png":
                fig = plot_fn(hist, p)
            else:
                fig = plot_fn(snaps, hist, p)
            fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches='tight')

        plt.close('all')

        # Summary
        h0, hf = hist[0], hist[-1]
        ft = ('CONTINUOUS' if hf['func_lf'] > 0.8 else
              'FEW LARGE CLUSTERS' if hf['func_nc'] < 6 else 'MANY ISLANDS')
        print(f"  Result: {h0['func_nc']} -> {hf['func_nc']} clusters ({ft})")
        print(f"  Saved to {out_dir}/")

    print("\n" + "=" * 65)
    print(f"  All {len(trials)} trials complete. Results in {output_base}/")
    print("=" * 65)


def run_hpc(trials: list, output_base: str, seed: int, hpc_config_path: str):
    """Generate and submit a SLURM array job for all trials."""
    with open(hpc_config_path) as f:
        hpc = json.load(f)

    repo_path = hpc["repo_path"]

    # Write a trial list file so the array job knows which JSON to use
    trial_list_path = os.path.join("hpc", "trial_list.txt")
    with open(trial_list_path, "w") as f:
        for t in trials:
            f.write(os.path.basename(t) + "\n")
    print(f"Wrote {trial_list_path} ({len(trials)} trials)")

    # Generate the array SLURM script
    slurm_script = f"""\
#!/bin/bash
#SBATCH --job-name=gells-trials
#SBATCH --account={hpc['group']}
#SBATCH --partition={hpc.get('partition', 'standard')}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={hpc.get('cpus', 4)}
#SBATCH --mem={hpc.get('mem_gb', 16)}G
#SBATCH --time={hpc.get('walltime', '04:00:00')}
#SBATCH --array=1-{len(trials)}
#SBATCH --output=trials_%A_%a.out
#SBATCH --error=trials_%A_%a.err

module load {hpc['python_module']}
source {hpc['venv_path']}/bin/activate
export MPLBACKEND=Agg

cd {repo_path}

# Read the trial filename for this array task
TRIAL=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" hpc/trial_list.txt)
TRIAL_NAME="${{TRIAL%.json}}"

echo "=== Array task $SLURM_ARRAY_TASK_ID: $TRIAL ==="

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
    print(f"\nTo submit:")
    print(f"  sbatch {slurm_path}")
    print(f"\nOr to submit now from your local machine:")
    print(f"  rsync this repo to the cluster, then:")
    print(f"  ssh {hpc['netid']}@hpc.arizona.edu")
    print(f"  cd {repo_path} && sbatch hpc/run_all_trials.slurm")
    print(f"\nMonitor: squeue --user {hpc['netid']}")
    print(f"Results will be in {output_base}/<TrialName>/")


def main():
    parser = argparse.ArgumentParser(
        description="Run all Trial JSON configs in the Trials/ folder"
    )
    parser.add_argument("--trials-dir", default="Trials",
                        help="Directory containing trial .json files")
    parser.add_argument("--output-dir", default="results/trials",
                        help="Base output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hpc", action="store_true",
                        help="Generate SLURM array job instead of running locally")
    parser.add_argument("--config", default="hpc/Alex.json",
                        help="HPC user config for --hpc mode")
    args = parser.parse_args()

    trials = find_trials(args.trials_dir)
    if not trials:
        print(f"No .json files found in {args.trials_dir}/")
        sys.exit(1)

    print(f"Found {len(trials)} trials:")
    for t in trials:
        print(f"  {os.path.basename(t)}")
    print()

    if args.hpc:
        run_hpc(trials, args.output_dir, args.seed, args.config)
    else:
        run_local(trials, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
