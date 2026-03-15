#!/usr/bin/env python3
"""
Generate user-specific HPC scripts and configure local/remote environment.

Automates everything possible for running GELLS-DEM on UArizona HPC (Puma):
  - Generates setup_env.sh and run.slurm from a JSON user config
  - Generates/checks SSH keys
  - Writes ~/.ssh/config entries
  - Copies SSH keys to bastion + filexfer (only if not already there)
  - Syncs the repo to the cluster (only if files are missing/stale)
  - Submits the environment setup as a batch job (only if venv doesn't exist)

Usage:
    python3 hpc/generate_hpc_scripts.py                    # uses Alex.json
    python3 hpc/generate_hpc_scripts.py hpc/Other.json     # different user
    python3 hpc/generate_hpc_scripts.py --sim-args "--E_modulus 5.0 --t_total 96"
"""

import argparse
import json
import os
import stat
import subprocess
import sys

# ── Default user config ──
USER_CONFIG = os.path.join(os.path.dirname(__file__), "Alex.json")

SSH_KEY = os.path.expanduser("~/.ssh/id_rsa")
BASTION = "hpc.arizona.edu"
FILEXFER = "filexfer.hpc.arizona.edu"
REPO_URL = "https://github.com/McGheeLab/GELLS-DEM.git"


# ══════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════

def step(n, title):
    print(f"\n{'=' * 60}")
    print(f"  STEP {n}: {title}")
    print("=" * 60)


def run_cmd(cmd, check=True, **kwargs):
    """Run a command, printing it first. Returns CompletedProcess."""
    if isinstance(cmd, list):
        display = " ".join(cmd)
    else:
        display = cmd
    print(f"  $ {display}")
    return subprocess.run(cmd, check=check, **kwargs)


def ssh_quiet(host, command, timeout=10):
    """Run a command over SSH, return (success, stdout)."""
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
         host, command],
        capture_output=True, text=True, timeout=timeout
    )
    return result.returncode == 0, result.stdout.strip()


def ask_continue(prompt="Continue?"):
    """Ask y/n, return True if yes."""
    resp = input(f"  {prompt} [Y/n] ").strip().lower()
    return resp in ("", "y", "yes")


# ══════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════

def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = json.load(f)

    required = ["netid", "group", "python_module", "repo_path", "venv_path"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        print(f"Error: missing required fields in {path}: {missing}")
        sys.exit(1)

    cfg.setdefault("cluster", "puma")
    cfg.setdefault("partition", "standard")
    cfg.setdefault("cpus", 4)
    cfg.setdefault("walltime", "04:00:00")
    return cfg


# ══════════════════════════════════════════════════════════════════════
# Script generators
# ══════════════════════════════════════════════════════════════════════

def generate_setup_env(cfg: dict) -> str:
    venv_path = cfg['venv_path'].replace('~', '$HOME')
    return f"""\
#!/bin/bash
# Auto-generated environment setup for {cfg['netid']}
# Run ONCE on a compute node (UA HPC: install software on compute nodes, not login nodes):
#   interactive -a {cfg['group']} -n {cfg['cpus']} -t 1:00:00
#   bash setup_env.sh

set -euo pipefail

echo "=== Setting up GELLS-DEM environment for {cfg['netid']} ==="

module load {cfg['python_module']}

python3 -m venv --system-site-packages {venv_path}
source {venv_path}/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib

echo ""
echo "=== Environment ready at {venv_path} ==="
echo "To activate:"
echo "  module load {cfg['python_module']} && source {venv_path}/bin/activate"
echo "To submit a job:"
echo "  sbatch run.slurm"
"""


def generate_run_slurm(cfg: dict, sim_args: str = "") -> str:
    run_cmd_str = f"python3 run_hpc_headless.py --output-dir results {sim_args}".rstrip()
    venv_path = cfg['venv_path'].replace('~', '$HOME')
    repo_path = cfg['repo_path'].replace('~', '$HOME')
    # NOTE: Do NOT specify both --mem and --cpus-per-task (UA HPC docs).
    #       Puma allocates 5 GB/CPU automatically.
    return f"""\
#!/bin/bash
#SBATCH --job-name=gells-dem
#SBATCH --account={cfg['group']}
#SBATCH --partition={cfg['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cfg['cpus']}
#SBATCH --time={cfg['walltime']}
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ── Load Python and activate virtual environment ──
module load {cfg['python_module']}
source {venv_path}/bin/activate

# ── Use non-interactive matplotlib backend (no display on HPC) ──
export MPLBACKEND=Agg

# ── Run simulation ──
cd {repo_path}
{run_cmd_str}

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Check efficiency: seff $SLURM_JOB_ID"
"""


def generate_setup_env_slurm(cfg: dict, setup_script_remote: str) -> str:
    """SLURM job that runs the environment setup on a compute node."""
    repo_path = cfg['repo_path'].replace('~', '$HOME')
    return f"""\
#!/bin/bash
#SBATCH --job-name=gells-setup
#SBATCH --account={cfg['group']}
#SBATCH --partition={cfg['partition']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p {repo_path}/slurm_logs
bash {setup_script_remote}
echo "Environment setup finished at $(date)"
"""


def generate_ssh_config(cfg: dict) -> str:
    netid = cfg["netid"]
    return f"""\
# --- UArizona HPC ({netid}) ---
Host uahpcbastion
    HostName hpc.arizona.edu
    User {netid}
    IdentityFile ~/.ssh/id_rsa

Host uahpcfxfr
    HostName filexfer.hpc.arizona.edu
    User {netid}
    IdentityFile ~/.ssh/id_rsa

Host *.puma.hpc.arizona.edu
    User {netid}
    IdentityFile ~/.ssh/id_rsa
"""


# ══════════════════════════════════════════════════════════════════════
# Detection: check what's already done
# ══════════════════════════════════════════════════════════════════════

def ssh_key_is_on_host(cfg: dict, host: str) -> bool:
    """Test if we can SSH to host in BatchMode (key-only, no password)."""
    target = f"{cfg['netid']}@{host}"
    try:
        ok, _ = ssh_quiet(target, "echo ok")
        return ok
    except (subprocess.TimeoutExpired, Exception):
        return False


def detect_ssh_keys_copied(cfg: dict) -> dict:
    """Check which hosts already have our SSH key."""
    results = {}
    for host, label in [(BASTION, "bastion"), (FILEXFER, "filexfer")]:
        print(f"  Checking {label} ({host})...", end=" ", flush=True)
        ok = ssh_key_is_on_host(cfg, host)
        results[label] = ok
        print("key present" if ok else "key needed")
    return results


def detect_repo_on_cluster(cfg: dict) -> bool:
    """Check if the repo exists on the cluster (via filexfer)."""
    target = f"{cfg['netid']}@{FILEXFER}"
    marker = f"{cfg['repo_path']}/new_dem_0.py"
    try:
        ok, _ = ssh_quiet(target, f"test -f {marker} && echo yes")
        return ok
    except (subprocess.TimeoutExpired, Exception):
        return False


def detect_venv_on_cluster(cfg: dict) -> bool:
    """Check if the Python venv already exists on the cluster."""
    target = f"{cfg['netid']}@{FILEXFER}"
    venv_marker = cfg['venv_path'].replace('~', '$HOME') + "/bin/activate"
    try:
        ok, _ = ssh_quiet(target, f"test -f {venv_marker} && echo yes")
        return ok
    except (subprocess.TimeoutExpired, Exception):
        return False


def detect_vpn_connected(cfg: dict) -> bool:
    """Check if we can reach filexfer (implies HPC VPN is up)."""
    target = f"{cfg['netid']}@{FILEXFER}"
    try:
        ok, _ = ssh_quiet(target, "echo ok")
        return ok
    except (subprocess.TimeoutExpired, Exception):
        return False


# ══════════════════════════════════════════════════════════════════════
# Automated setup steps
# ══════════════════════════════════════════════════════════════════════

def ensure_ssh_key():
    """Generate SSH key if none exists."""
    if os.path.exists(SSH_KEY):
        print(f"  SSH key found: {SSH_KEY}")
        return
    print(f"  No SSH key at {SSH_KEY} — generating one...")
    run_cmd(["ssh-keygen", "-t", "rsa", "-f", SSH_KEY, "-N", ""])


def update_ssh_config(cfg: dict):
    """Add HPC SSH entries to ~/.ssh/config if not already present."""
    ssh_dir = os.path.expanduser("~/.ssh")
    os.makedirs(ssh_dir, exist_ok=True)
    config_path = os.path.join(ssh_dir, "config")

    marker = f"# --- UArizona HPC ({cfg['netid']}) ---"
    block = generate_ssh_config(cfg)

    existing = ""
    if os.path.exists(config_path):
        with open(config_path) as f:
            existing = f.read()

    if marker in existing:
        print(f"  SSH config already has entries for {cfg['netid']}")
        return

    with open(config_path, "a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write("\n" + block)

    print(f"  Added HPC entries to {config_path}")


def copy_ssh_keys(cfg: dict, key_status: dict):
    """Run ssh-copy-id only to hosts that need it."""
    netid = cfg["netid"]
    hosts = [(BASTION, "bastion"), (FILEXFER, "filexfer")]
    all_present = True
    for host, label in hosts:
        if key_status.get(label):
            continue
        all_present = False
        target = f"{netid}@{host}"
        print(f"\n  Copying SSH key to {label} ({target})...")
        print("  (You will be prompted for your password and Duo)")
        result = run_cmd(["ssh-copy-id", "-i", SSH_KEY, target], check=False)
        if result.returncode == 0:
            print(f"  Key copied to {label} successfully")
        else:
            print(f"  WARNING: ssh-copy-id to {label} failed (rc={result.returncode})")
            print(f"  You can retry manually: ssh-copy-id {target}")
    if all_present:
        print("  SSH keys already on both hosts — nothing to do")


def sync_repo_to_cluster(cfg: dict, local_repo: str):
    """rsync the local repo to the cluster via filexfer."""
    netid = cfg["netid"]
    repo_path = cfg["repo_path"]
    remote = f"{netid}@{FILEXFER}"

    print(f"\n  Creating {repo_path} on cluster...")
    run_cmd(["ssh", remote, f"mkdir -p {repo_path}"], check=False)

    print(f"  Syncing {local_repo}/ -> {remote}:{repo_path}/")
    run_cmd([
        "rsync", "-avz", "--delete",
        "--exclude", "__pycache__",
        "--exclude", ".git",
        "--exclude", "*.pyc",
        "--exclude", "results/",
        "--exclude", "simulations/",
        local_repo + "/",
        f"{remote}:{repo_path}/"
    ], check=False)


def submit_env_setup(cfg: dict, setup_path_local: str):
    """Upload a SLURM job that runs setup_env.sh on a compute node."""
    netid = cfg["netid"]
    repo_path = cfg["repo_path"]
    remote = f"{netid}@{FILEXFER}"
    setup_remote = f"{repo_path}/setup_env.sh"

    slurm_content = generate_setup_env_slurm(cfg, setup_remote)
    tmp_slurm = os.path.join(os.path.dirname(setup_path_local), "setup_env.slurm")
    with open(tmp_slurm, "w") as f:
        f.write(slurm_content)

    print(f"\n  Uploading setup SLURM job...")
    run_cmd(["scp", tmp_slurm, f"{remote}:{repo_path}/setup_env.slurm"], check=False)

    print(f"  Submitting environment setup job...")
    result = run_cmd(
        ["ssh", f"{netid}@{BASTION}",
         f"ssh shell.hpc.arizona.edu 'cd {repo_path} && sbatch setup_env.slurm'"],
        check=False, capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        print(f"  {result.stdout.strip()}")
        print("  Environment will be ready when the job completes.")
        print(f"  Check: ssh {netid}@{BASTION} 'ssh shell.hpc.arizona.edu squeue -r -u {netid}'")
    else:
        stderr = result.stderr.strip() if result.stderr else ""
        print(f"  Could not auto-submit setup job. {stderr}")
        print(f"  To set up manually, SSH in and run:")
        print(f"    interactive -a {cfg['group']} -n {cfg['cpus']} -t 1:00:00")
        print(f"    bash {repo_path}/setup_env.sh")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate user-specific HPC setup and run scripts"
    )
    parser.add_argument("config", nargs="?", default=USER_CONFIG,
                        help="Path to user JSON config (default: Alex.json)")
    parser.add_argument(
        "--sim-args", default="",
        help="Extra arguments for run_hpc_headless.py (e.g. '--E_modulus 5.0 --t_total 96')"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Directory to write scripts (default: hpc/<netid>/)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force all steps even if autodetection says they're done"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = args.output_dir or os.path.join("hpc", cfg["netid"])
    os.makedirs(out_dir, exist_ok=True)

    local_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print(f"  GELLS-DEM HPC Setup for {cfg['netid']}")
    print(f"  Cluster: {cfg['cluster']}  |  Group: {cfg['group']}")
    print("=" * 60)

    # ── Step 1: Generate scripts (always) ──
    step(1, "Generate scripts")

    setup_path = os.path.join(out_dir, "setup_env.sh")
    with open(setup_path, "w") as f:
        f.write(generate_setup_env(cfg))
    os.chmod(setup_path, os.stat(setup_path).st_mode | stat.S_IXUSR)
    print(f"  Wrote {setup_path}")

    slurm_path = os.path.join(out_dir, "run.slurm")
    with open(slurm_path, "w") as f:
        f.write(generate_run_slurm(cfg, args.sim_args))
    os.chmod(slurm_path, os.stat(slurm_path).st_mode | stat.S_IXUSR)
    print(f"  Wrote {slurm_path}")

    # ── Step 2: SSH key + config ──
    step(2, "SSH key & config")
    ensure_ssh_key()
    update_ssh_config(cfg)

    # ── Step 3: Detect & copy SSH keys ──
    step(3, "Copy SSH keys to cluster")
    key_status = detect_ssh_keys_copied(cfg)
    need_keys = args.force or not all(key_status.values())
    if need_keys:
        copy_ssh_keys(cfg, key_status)
    else:
        print("  SSH keys already on both hosts — nothing to do")

    # ── Step 4: Check VPN ──
    step(4, "Check HPC VPN")
    vpn_ok = detect_vpn_connected(cfg)
    if vpn_ok:
        print("  VPN is connected (can reach filexfer)")
    else:
        print("""
  Cannot reach filexfer.hpc.arizona.edu.
  Open Cisco AnyConnect and connect to: vpn.hpc.arizona.edu
  (NOT the regular UA VPN)
""")
        if not ask_continue("Is the VPN connected now?"):
            print("\n  Connect the VPN and re-run. Steps 1-3 will be skipped automatically.\n")
            return
        # Re-check
        vpn_ok = detect_vpn_connected(cfg)
        if not vpn_ok:
            print("  Still cannot reach cluster. Check VPN and try again.")
            return

    # ── Step 5: Sync repo ──
    step(5, "Sync repo to cluster")
    repo_exists = detect_repo_on_cluster(cfg)
    if repo_exists and not args.force:
        print(f"  Repo already on cluster ({cfg['repo_path']}/new_dem_0.py found)")
        print("  Re-syncing to push any local changes...")
    sync_repo_to_cluster(cfg, local_repo)

    # ── Step 6: Environment setup ──
    step(6, "Python environment setup")
    venv_exists = detect_venv_on_cluster(cfg)
    if venv_exists and not args.force:
        print(f"  Venv already exists ({cfg['venv_path']}/bin/activate found)")
        print("  Skipping environment setup")
    else:
        print("  Venv not found on cluster — submitting setup job...")
        submit_env_setup(cfg, setup_path)

    # ── Step 7: Manual steps ──
    step(7, "Connect VSCode & run (manual)")
    print(f"""
  SSH in and start an interactive session:
    ssh {cfg['netid']}@{BASTION}
    # type 'shell' at the bastion prompt
    interactive -a {cfg['group']} -n {cfg['cpus']} -t 8:00:00
    hostname    # note the output, e.g. r6u24n2.puma.hpc.arizona.edu

  In VSCode:
    >< -> Connect to Host -> + Add New SSH Host
    Enter: ssh {cfg['netid']}@<hostname>.puma.hpc.arizona.edu
    Open folder: {cfg['repo_path']}

  Run interactively (from VSCode terminal):
    module load {cfg['python_module']}
    source {cfg['venv_path']}/bin/activate
    python3 run_hpc_headless.py --output-dir results

  Or submit a batch job:
    cd {cfg['repo_path']}
    sbatch run.slurm
    squeue -r --user {cfg['netid']}
""")

    print("=" * 60)
    print("  Setup complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
