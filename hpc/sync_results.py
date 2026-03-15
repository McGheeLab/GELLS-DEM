#!/usr/bin/env python3
"""
Sync HPC results back to local machine.

Usage:
    python3 hpc/sync_results.py
    python3 hpc/sync_results.py --config hpc/Alex.json
    python3 hpc/sync_results.py --local-dir results/trials
"""

# ── USER CONFIGURATION ──────────────────────────────────────────────
HPC_CONFIG = "hpc/Alex.json"
LOCAL_DIR = "results"
# ────────────────────────────────────────────────────────────────────

import argparse
import json
import os
import subprocess
import sys


def check_jobs(netid: str):
    """Check if any jobs are still running.

    Uses squeue -r to expand array sub-tasks into individual rows
    (UA HPC docs: -r flag required to see per-task status).
    """
    try:
        r = subprocess.run(
            ["ssh", f"{netid}@hpc.arizona.edu",
             f"ssh shell.hpc.arizona.edu 'squeue -r -u {netid} -h 2>/dev/null'"],
            capture_output=True, text=True, timeout=30)
        jobs = r.stdout.strip()
        if jobs:
            print(f"  Active jobs for {netid}:")
            print(f"  {'JOBID':<16} {'NAME':<16} {'STATE':<8} {'TIME':<10}")
            for line in jobs.splitlines():
                parts = line.split()
                if len(parts) >= 6:
                    print(f"  {parts[0]:<16} {parts[2]:<16} {parts[4]:<8} {parts[5]:<10}")
            print()
            return True
        else:
            print("  No active jobs.\n")
            return False
    except subprocess.TimeoutExpired:
        print("  SSH timed out — check VPN connection.")
        return False
    except FileNotFoundError:
        print("  ssh not found.")
        return False


def list_remote_results(netid: str, repo_path: str):
    """List result directories on the cluster."""
    try:
        r = subprocess.run(
            ["ssh", f"{netid}@filexfer.hpc.arizona.edu",
             f"ls -la {repo_path}/results/ 2>/dev/null"],
            capture_output=True, text=True, timeout=30)
        if r.stdout.strip():
            print("  Remote results:")
            for line in r.stdout.strip().splitlines():
                print(f"    {line}")
            print()
        else:
            print("  No results found on cluster.\n")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("  Could not list remote results.\n")


def sync_results(netid: str, repo_path: str, local_dir: str):
    """Rsync results from cluster to local machine."""
    remote = f"{netid}@filexfer.hpc.arizona.edu:{repo_path}/results/"
    os.makedirs(local_dir, exist_ok=True)

    print(f"  Syncing: {remote}")
    print(f"      ->   {os.path.abspath(local_dir)}/\n")

    try:
        result = subprocess.run(
            ["rsync", "-avz", "--progress", remote, f"{local_dir}/"],
            timeout=600)
        if result.returncode == 0:
            print(f"\n  Sync complete! Results in {os.path.abspath(local_dir)}/")
        else:
            print(f"\n  rsync exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        print("  rsync timed out (10 min limit). Try again or increase timeout.")
    except FileNotFoundError:
        print("  rsync not found. Install it or sync manually:")
        print(f"    rsync -avz {remote} {local_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Sync HPC results to local machine")
    parser.add_argument("--config", default=HPC_CONFIG, help="HPC config JSON")
    parser.add_argument("--local-dir", default=LOCAL_DIR, help="Local output directory")
    parser.add_argument("--list-only", action="store_true", help="Just list remote results")
    args = parser.parse_args()

    with open(args.config) as f:
        hpc = json.load(f)

    netid = hpc["netid"]
    repo_path = hpc["repo_path"]

    print("=" * 60)
    print(f"  GELLS-DEM: Sync Results from {netid}@puma")
    print("=" * 60 + "\n")

    # Check for running jobs
    has_jobs = check_jobs(netid)
    if has_jobs:
        print("  Warning: jobs still running — results may be incomplete.\n")

    # List what's on the cluster
    list_remote_results(netid, repo_path)

    if args.list_only:
        return

    # Sync
    sync_results(netid, repo_path, args.local_dir)


if __name__ == "__main__":
    main()
