Written for: the maintainer, if HPC use is ever revived.

# Archived: UArizona HPC notes

These were the HPC operating instructions from `CLAUDE.md` up to V2.6.
GELS moved to local-only execution in V2.7; this file is kept for
reference and is no longer part of the active developer guide.
---

## UArizona HPC Best Practices

Reference: [UA HPC Documentation](https://github.com/UA-ResearchComputing-HPC/hpc-documentation)

### Hostnames & Access

| Purpose | Hostname | Notes |
|---------|----------|-------|
| SSH login (bastion) | `hpc.arizona.edu` | Type `shell` after login to reach Puma |
| File transfers | `filexfer.hpc.arizona.edu` | 100 Gb link to shared storage |
| Web portal | `ood.hpc.arizona.edu` | Open OnDemand (browser-based) |
| VPN (off-campus) | `vpn.arizona.edu` | Required for R-DAS; use Cisco Secure Client |
| HPC VPN | `vpn.hpc.arizona.edu` | Required for X11 forwarding, port tunnels |

**SSH login flow:**
```bash
ssh netid@hpc.arizona.edu     # lands on bastion (gatekeeper)
# then type:
shell                          # connects to Puma login node (junonia or wentletrap)
```

**Critical:** The bastion host has **no shared storage** and a **10 MB quota**. Never use
`hpc.arizona.edu` for file transfers — always use `filexfer.hpc.arizona.edu`. Three failed
password attempts lock your account for 1 hour.

**For scripted automation (run_all_trials.py Mode 3):**
```bash
# Submit jobs via double-hop SSH:
ssh netid@hpc.arizona.edu "ssh shell.hpc.arizona.edu 'cd ~/GELS && sbatch script.slurm'"

# File transfers via filexfer:
rsync -ravz local_dir/ netid@filexfer.hpc.arizona.edu:~/GELS/
```

### Clusters

| Cluster | CPUs/node | RAM/node | GPUs | OS | Mem/CPU |
|---------|-----------|----------|------|----|---------|
| **Puma** (recommended) | 94 | 512 GB | V100S, A100 MIG | Rocky Linux 9 | 5 GB |
| Ocelote | 28 | 192 GB | P100 | CentOS 7 | 6 GB |
| El Gato | 16 | 64 GB | none | CentOS 7 | 4 GB |

**Use Puma for GELS.** Software compiled on CentOS 7 (Ocelote) may fail on Rocky Linux 9
(Puma) and vice versa. Always compile and test on the same cluster you'll run on.

### Allocations & Limits

**Monthly CPU-hours (Standard, free):** Puma 150,000 | Ocelote 100,000 | El Gato 7,000

**Per-group limits (Puma Standard):** 3,290 CPUs, 16,996 GB memory, 4 GPUs max simultaneous.

**Global limits:** Max 1,000 simultaneous jobs. Max walltime 10 days (240 hours).

**Windfall partition:** No hours consumed but lower priority and **preemptible** — jobs can be
killed at any time. Good for testing, not for production runs.

**Check remaining hours:** Run `va` on a login node.

### Storage

| Path | Quota | Lifetime | Backed Up? |
|------|-------|----------|------------|
| `/home/uXX/netid` | **50 GB** | Account lifetime | No |
| `/groups/pi_netid` | **500 GB** | PI account lifetime | No |
| `/xdisk/pi_netid` | 200 GB – 20 TB | **300 days then deleted** | No |
| `/tmp` (compute node) | ~1 TB | Job duration only | No |

**Nothing on HPC is backed up.** Maintain your own backups.

**Warnings:**
- Filling `/home` causes login failures and Open OnDemand errors.
- Avoid >100,000 files in a single directory (slows the parallel filesystem).
- Redirect pip/conda cache to `/groups` or `/xdisk` — they default to `~/.local` and eat `/home` quota.
- Check usage: `uquota` command. Find hidden space hogs: `du -hs $(ls -A ~)`.

**For GELS:** Store results in `/groups` (persistent) or `/xdisk` (temporary, larger).
Keep the repo clone in `/home/uXX/netid/GELS` or `/groups`. Transfer results to your
local machine via rsync before `/xdisk` expires.

### Python Environment Setup

```bash
# On a login node (after typing 'shell'):
module load python/3.11/3.11.4
python3 -m venv --system-site-packages /groups/<pi_group>/gels_env
source /groups/<pi_group>/gels_env/bin/activate
pip install numpy scipy matplotlib imageio scikit-image tqdm pyvista numba
```

**In batch scripts:**
```bash
module load python/3.11/3.11.4
source /groups/<pi_group>/gels_env/bin/activate
export MPLBACKEND=Agg  # headless matplotlib
```

**Rules:**
- Always specify module version: `python/3.11/3.11.4`, not just `python`.
- Always use `python3`, never `python` (system python is 2.7).
- Never `pip install` globally — always use a virtual environment.
- Modules are only available on compute/login nodes, not the bastion.
- `~/.local/lib/python3.X` causes cross-contamination between venvs — use `--system-site-packages`
  and install into the venv, or clean out `~/.local` if you see import conflicts.

### Batch Job Directives

**Minimum SLURM script:**
```bash
#!/bin/bash
#SBATCH --job-name=gels-run
#SBATCH --account=your_group       # omit for windfall
#SBATCH --partition=standard       # or windfall, gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # GELS is single-process
#SBATCH --cpus-per-task=4          # more CPUs = more memory (5 GB/CPU on Puma)
#SBATCH --time=04:00:00            # HHH:MM:SS, max 240:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
```

**CPU/Memory relationship on Puma:** Each CPU allocated comes with 5 GB. Requesting
`--cpus-per-task=4` gives 20 GB. Do NOT specify both `--ntasks` and `--mem` simultaneously —
specify one and let the scheduler calculate the other. Invalid ratios may redirect to
high-memory queues (longer wait).

**GELS is single-node, single-process.** Always use `--nodes=1 --ntasks=1`.
Increase `--cpus-per-task` if you need more memory (e.g., large 3D grids).

### Array Jobs (Parameter Sweeps)

```bash
#SBATCH --array=1-12               # basic range
#SBATCH --array=1-100%10           # limit to 10 concurrent subjobs
#SBATCH --array=1-3,7,10-15        # non-sequential
#SBATCH --output=slurm_logs/%x_%A_%a.out   # %A = parent job, %a = array index
```

**Environment variable:** `$SLURM_ARRAY_TASK_ID` is the unique integer for each sub-job.

**Pattern for reading trial names from a list file:**
```bash
TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/trial_list.txt)
```

**Monitoring array jobs:**
```bash
squeue -r --job=<jobid>     # -r flag expands array tasks into individual rows
squeue --me                 # all your jobs
```

Without `-r`, `squeue` shows the parent job as a single "R" line as long as ANY sub-task
is running — you cannot see per-task completion. **Always use `-r` for array jobs.**

**Never submit hundreds of jobs via a for-loop.** Use `--array` instead. Users submitting
>100 jobs via loops will be contacted by HPC staff.

### Job Monitoring & Diagnostics

| Command | Purpose |
|---------|---------|
| `squeue --me` | All your running/pending jobs |
| `squeue -r --job=JOBID` | Individual array sub-tasks |
| `scontrol show jobs JOBID` | Detailed info on running/pending job |
| `scancel JOBID` | Cancel a job (all array tasks) |
| `scancel JOBID_3` | Cancel specific array task |
| `seff JOBID` | CPU and memory efficiency (completed jobs) |
| `job-history JOBID` | Human-readable completed job info |
| `past-jobs -d 7` | Search your jobs from past 7 days |
| `job-limits GROUP` | Group resource limits and current usage |
| `va` | Check remaining allocation hours |
| `uquota` | Check storage quotas |

**After every run:** Use `seff JOBID` to check CPU and memory efficiency. If efficiency
is below ~80%, reduce resource requests for future jobs. This shortens queue wait times.

### File Transfers

| Size | Method |
|------|--------|
| < 64 MB | Open OnDemand web upload (`ood.hpc.arizona.edu`) |
| < 100 GB | rsync / scp via `filexfer.hpc.arizona.edu` |
| > 100 GB | Globus (endpoint: "UA HPC Filesystems") |

**Rsync (recommended):**
```bash
# Upload repo to cluster (trailing slash = contents only)
rsync -ravz --exclude __pycache__ --exclude .git --exclude results/ \
    ./  netid@filexfer.hpc.arizona.edu:~/GELS/

# Download results from cluster
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**SCP:**
```bash
scp -rp ./Trials/ netid@filexfer.hpc.arizona.edu:~/GELS/Trials/
scp -rp netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**Best practices:**
- Limit to 2–3 concurrent transfer sessions.
- Tar/compress many small files before transferring (parallel filesystem is slow with many small files).
- Set up SSH keys on `filexfer.hpc.arizona.edu` to avoid repeated password/2FA prompts.
- If SCP/SFTP fails with "Received message too long", check `~/.bashrc` for echo/printf
  statements and comment them out.

### GELS HPC Workflow

**Automated (run_all_trials.py Mode 3):**
```
1. Finds all .json files in Trials/
2. Generates SLURM array job script (results → /groups, staggered starts)
3. Rsyncs repo to cluster via filexfer
4. Submits via SSH: bastion → shell → sbatch
5. Polls with squeue -r (per-task progress)
6. Incremental rsync as tasks complete (delete from HPC after transfer)
7. Final rsync when all tasks done
```

**Resume incomplete trials (run_all_trials.py Mode 4, V2.6):**
```
1. Scans local results/LHC/ for incomplete trials (t_last < t_total)
2. Uploads last snapshot + params.json + history.json to HPC for each
3. Generates resume SLURM script with --resume-from
4. Rsyncs repo, submits, polls, and syncs as in Mode 3
```

**Manual workflow:**
```bash
# 1. Sync code to cluster
rsync -ravz --exclude __pycache__ --exclude .git --exclude results/ \
    ./  netid@filexfer.hpc.arizona.edu:~/GELS/

# 2. SSH in and submit
ssh netid@hpc.arizona.edu
shell
cd ~/GELS
sbatch hpc/run_all_trials.slurm

# 3. Monitor
squeue -r --me
seff JOBID   # after completion

# 4. Sync results back (from local machine)
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELS/results/ ./results/
```

**Standalone result sync:**
```bash
python3 hpc/sync_results.py              # check status + sync
python3 hpc/sync_results.py --list-only  # just check what's there
```

### HPC Config File (hpc/Alex.json)

Each user's HPC settings are stored in a JSON file:
```json
{
    "netid": "mcgheealex",
    "group": "your_pi_group",
    "repo_path": "~/GELS",
    "python_module": "python/3.11/3.11.4",
    "venv_path": "/groups/your_pi_group/gels_env",
    "partition": "standard",
    "cpus": 4,
    "walltime": "04:00:00",
    "results_path": "/groups/your_pi_group/results",
    "max_concurrent": 10
}
```

**V2.6 fields**: `results_path` directs simulation output to `/groups` (500 GB) instead of
`/home` (50 GB). `max_concurrent` limits simultaneous SLURM array tasks (default 10).

### Common Gotchas

- **Never run computations on login nodes.** They will be killed without warning.
- **Puma vs Ocelote binaries are incompatible** (Rocky Linux 9 vs CentOS 7).
- **`MPLBACKEND=Agg` is required** in batch scripts — there is no display server.
- **`squeue` without `-r` hides array sub-task status** — always use `-r` for array jobs.
- **`/xdisk` data is deleted after 300 days** with no extensions. Back up before expiration.
- **Filling `/home` (50 GB) breaks login.** Keep pip cache and conda envs in `/groups`.
- **Module versions change.** Always specify exact version (e.g., `python/3.11/3.11.4`).
- **Interactive sessions for installs only.** Compile and install software from compute nodes
  via `interactive -a your_group`, not from login nodes.

### Required Acknowledgement (Publications)

> "This material is based upon High Performance Computing (HPC) resources supported by the
> University of Arizona TRIF, UITS, and Research, Innovation, and Impact (RII) and maintained
> by the UArizona Research Technologies department."
