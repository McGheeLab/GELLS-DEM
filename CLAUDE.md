# GELLS-DEM Project Guide

## Project Overview

**GELLS-DEM** (Granular Encapsulated Living-cell Laden Scaffold - Discrete Element Method)
is a 2D/3D overdamped particle dynamics simulator for modelling cell-driven rearrangement
of hydrogel granular scaffolds. The primary simulation engine is `new_dem_0.py`.

**Current version: V1.4.1**

## Repository Layout

```
GELLS-DEM/
├── new_dem_0.py                 # PRIMARY simulation engine (V1.4)
├── viz_compaction.py            # Void-space & compaction visualization
├── viz_percolation.py           # Transport property analysis (Darcy, Kozeny-Carman)
├── viz_movies.py                # 3D volumetric animations (rotating, timelapse, sweep)
├── viz_phases.py                # Individual phase volume visualization
├── new_dem_visualization.py     # Unified post-processing & visualisation (legacy)
├── new_dem_postprocess.py       # Legacy-compatible post-processing (JSON frames)
├── run_hpc_headless.py          # HPC headless runner (supports 2D/3D modes)
├── run_all_trials.py            # Batch trial runner (local / SLURM)
├── dem_config.json              # Default JSON config (legacy format)
├── Trials/                      # Parameter sweep JSON configs
│   ├── Trial15_3D.json          # 3D trial config (flat format)
│   ├── Trial16_3D.json          # 3D small domain trial (flat format)
│   └── Trial17_2D.json          # 2D baseline trial (flat format)
├── CodeLog/
│   ├── Architecture/            # Architecture documents
│   ├── Readme/                  # README documents
│   ├── References/              # Literature references
│   └── Updates/                 # Changelog / update log
├── hpc/                         # HPC setup scripts and user configs
├── old/                         # Archived legacy code (3D superellipsoid, etc.)
└── CLAUDE.md                    # THIS FILE
```

## Key Technical Decisions

- **Units throughout**: micrometres (length), nanonewtons (force), hours (time), kPa (modulus).
- **Simulation modes** (V1.4+): `"2D"` (pure 2D), `"2D-slice"` (generate 3D packing, slice
  at z-midplane, run 2D), `"3D"` (full volumetric). Controlled by `Params.mode`.
- **3D domain** (V1.4+): `Lx × Ly × Lz` with 6-face wall boundaries. `Lz` defaults to 800 µm.
- **Superellipsoids** (V1.4+): `(|x/a|^n1 + |y/b|^n1)^(n2/n1) + |z/c|^n2 = 1` with
  equatorial blockiness n1 and meridional blockiness n2. Spheres when a=b=c, n1=n2=2.
- **Quaternion orientation** (V1.4+): 3D rotational state stored as unit quaternions (w,x,y,z).
  Overdamped angular dynamics: `γ_rot dω/dt = Σ τ`. Integrated via `quat_integrate()`.
- **Contact model**: Hertzian (V1.1+). Stiffness is derived from `E_modulus` and `poisson_ratio`,
  NOT set as an arbitrary spring constant. See `hertz_contact_force()`.
- **Cell force model**: Motor-clutch (V1.2+, Chan & Odde 2008). Cell traction depends on
  substrate stiffness, replacing the old simple spring. See `motor_clutch_force()`.
- **Cell lifecycle** (V1.2+): Cells are 20 µm spheres → attach at ~3 h → spread to 5 µm-tall
  ellipsoids (volume conserved) → overcrowded cells crawl on others → bridges form via
  motor-clutch adhesions. See `update_cell_state()`.
- **Granule shape 2D** (V1.3+): Superellipses `|x/a|^n + |y/b|^n = 1` parameterised by
  semi-axes (a, b), blockiness exponent (n), and orientation (θ). Circles are the
  special case a=b, n=2. Enable with `shape_enabled=True`. See `superellipse_*()` functions.
- **Contact detection 2D** (V1.3+): Common normal method for superellipse–superellipse contact.
  Newton-Raphson iterative solver. Circle fast-path bypasses solver entirely.
- **Contact detection 3D** (V1.4+): 3D common normal method for superellipsoid–superellipsoid
  contact. 4-unknown Newton-Raphson (eta_i, omega_i, eta_j, omega_j). Sphere fast-path.
- **Transport metrics** (V1.4+): Kozeny-Carman permeability, Darcy flow, RCP fraction,
  compaction ratio, Darcy number. See `kozeny_carman_permeability()`, `rcp_fraction_superellipsoid()`.
- **Rotational dynamics**: Overdamped rotation from off-centre contact torques.
  2D: `γ_rot dθ/dt = Σ τ` (V1.3+). 3D: quaternion integration (V1.4+).
- **Shape descriptors** (V1.3+, Liu et al. 2025): Circularity, aspect ratio, elongation,
  blockiness tracked per granule and reported as population statistics.
- **Volume conservation**: Overlap lens area is tracked and redistributed via effective radii
  for rendering. Forces still use the original (undeformed) radii.
- **Integration**: Overdamped Euler (no inertia). Velocity cap prevents numerical blowup.
- **Neighbour search**: `scipy.spatial.cKDTree` with cutoff `2*max_r_bound + cell_sense_distance`.
- **Performance** (V1.4+): Optional Numba JIT for Newton-Raphson solver and superellipsoid
  geometry. Bounding-box clipping in 3D field rendering. Target: 500-1000 granules.
- **Packing** (V1.4.1+): RSA placement followed by compression settle phase
  (`packing_settle_steps=200`) to achieve granule contact. No artificial gap
  (`packing_gap=0.0`). Seed is random by default (`seed=None`).
- **Boundary exclusion** (V1.4.1+): `boundary_exclusion=0.2` excludes 20% from each
  domain edge when computing metrics (connectivity, porosity, permeability). Only the
  inner 60% of the domain volume is sampled.
- **Trial JSON format** (V1.4.1+): Two formats supported. **Flat**: keys are `Params` field
  names directly (e.g., `"E_modulus": 10.0`). **Legacy**: nested sections (domain,
  mechanics, shape, etc.) with translated key names. Flat format detected by `"_format": "flat"`.

## Conventions

- Granule types: `0` = functional (cell-laden), `1` = inert (passive).
- Phase fields: `phi_f` = functional, `phi_i` = inert, `phi_v` = void = 1 - phi_f - phi_i.
- All parameters live in the `Params` dataclass; modify defaults there or pass overrides.
- Snapshots are dicts with keys: `phi_f`, `phi_i`, `phi_v`, `x`, `y`, `r`, `gtype`, plus
  cell state and shape arrays. In 3D mode, also includes `z`, `quat`, `c`, `n1`, `n2`.

## Documentation Requirements

- **Update log**: Every code change must be recorded in `CodeLog/Updates/CHANGELOG.md`.
  Bug fixes go in a dedicated "Bug Fixes" subsection within the relevant version entry.
- **Architecture**: The system architecture document lives at `CodeLog/Architecture/ARCHITECTURE.md`.
  Update it when modules are added, removed, or significantly restructured.
- **README**: The user-facing README is at `CodeLog/Readme/README.md`.

## Running the Simulation

```bash
# Default 2D mode
python3 new_dem_0.py

# 3D mode
python3 -c "from new_dem_0 import Params, run; run(Params(mode='3D', Lx=400, Ly=400, Lz=400, t_total=24))"
```

Or import and call programmatically:

```python
from new_dem_0 import Params, run

# 2D (default, backward compatible)
p = Params(E_modulus=5.0, t_total=48.0)
hist, snaps, p, gs = run(p)

# 3D volumetric
p = Params(mode='3D', Lx=400, Ly=400, Lz=400, t_total=48.0)
hist, snaps, p, gs = run(p)

# 2D-slice (generate 3D packing, slice at z-midplane, run 2D)
p = Params(mode='2D-slice', Lx=800, Ly=800, Lz=800, t_total=72.0)
hist, snaps, p, gs = run(p)
```

### Visualization Scripts

```bash
# After running a simulation that saves to ./simulations/run1:
python viz_compaction.py -i ./simulations/run1
python viz_percolation.py -i ./simulations/run1
python viz_movies.py -i ./simulations/run1
python viz_phases.py -i ./simulations/run1
```

Or import programmatically:

```python
import viz_compaction, viz_percolation, viz_movies, viz_phases

viz_compaction.run_all(hist, snaps=snaps, outdir='plots/')
viz_percolation.run_all(hist, outdir='plots/')
viz_movies.run_all(snaps, hist, p, outdir='plots/')
viz_phases.run_all(hist, snaps=snaps, p=p, outdir='plots/')
```

## Dependencies

- Python 3.9+
- numpy, scipy, matplotlib
- Optional: pyvista (3D rendering), scikit-image (marching cubes), numba (JIT acceleration)
- Optional for post-processing: imageio, tqdm

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
ssh netid@hpc.arizona.edu "ssh shell.hpc.arizona.edu 'cd ~/GELLS-DEM && sbatch script.slurm'"

# File transfers via filexfer:
rsync -ravz local_dir/ netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/
```

### Clusters

| Cluster | CPUs/node | RAM/node | GPUs | OS | Mem/CPU |
|---------|-----------|----------|------|----|---------|
| **Puma** (recommended) | 94 | 512 GB | V100S, A100 MIG | Rocky Linux 9 | 5 GB |
| Ocelote | 28 | 192 GB | P100 | CentOS 7 | 6 GB |
| El Gato | 16 | 64 GB | none | CentOS 7 | 4 GB |

**Use Puma for GELLS-DEM.** Software compiled on CentOS 7 (Ocelote) may fail on Rocky Linux 9
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

**For GELLS-DEM:** Store results in `/groups` (persistent) or `/xdisk` (temporary, larger).
Keep the repo clone in `/home/uXX/netid/GELLS-DEM` or `/groups`. Transfer results to your
local machine via rsync before `/xdisk` expires.

### Python Environment Setup

```bash
# On a login node (after typing 'shell'):
module load python/3.11/3.11.4
python3 -m venv --system-site-packages /groups/<pi_group>/gells_env
source /groups/<pi_group>/gells_env/bin/activate
pip install numpy scipy matplotlib imageio scikit-image tqdm pyvista numba
```

**In batch scripts:**
```bash
module load python/3.11/3.11.4
source /groups/<pi_group>/gells_env/bin/activate
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
#SBATCH --job-name=gells-run
#SBATCH --account=your_group       # omit for windfall
#SBATCH --partition=standard       # or windfall, gpu_standard
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # GELLS-DEM is single-process
#SBATCH --cpus-per-task=4          # more CPUs = more memory (5 GB/CPU on Puma)
#SBATCH --time=04:00:00            # HHH:MM:SS, max 240:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
```

**CPU/Memory relationship on Puma:** Each CPU allocated comes with 5 GB. Requesting
`--cpus-per-task=4` gives 20 GB. Do NOT specify both `--ntasks` and `--mem` simultaneously —
specify one and let the scheduler calculate the other. Invalid ratios may redirect to
high-memory queues (longer wait).

**GELLS-DEM is single-node, single-process.** Always use `--nodes=1 --ntasks=1`.
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
    ./  netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/

# Download results from cluster
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/results/ ./results/
```

**SCP:**
```bash
scp -rp ./Trials/ netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/Trials/
scp -rp netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/results/ ./results/
```

**Best practices:**
- Limit to 2–3 concurrent transfer sessions.
- Tar/compress many small files before transferring (parallel filesystem is slow with many small files).
- Set up SSH keys on `filexfer.hpc.arizona.edu` to avoid repeated password/2FA prompts.
- If SCP/SFTP fails with "Received message too long", check `~/.bashrc` for echo/printf
  statements and comment them out.

### GELLS-DEM HPC Workflow

**Automated (run_all_trials.py Mode 3):**
```
1. Finds all .json files in Trials/
2. Generates SLURM array job script
3. Rsyncs repo to cluster via filexfer
4. Submits via SSH: bastion → shell → sbatch
5. Polls with squeue -r (per-task progress)
6. Incremental rsync as tasks complete
7. Final rsync when all tasks done
```

**Manual workflow:**
```bash
# 1. Sync code to cluster
rsync -ravz --exclude __pycache__ --exclude .git --exclude results/ \
    ./  netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/

# 2. SSH in and submit
ssh netid@hpc.arizona.edu
shell
cd ~/GELLS-DEM
sbatch hpc/run_all_trials.slurm

# 3. Monitor
squeue -r --me
seff JOBID   # after completion

# 4. Sync results back (from local machine)
rsync -ravz netid@filexfer.hpc.arizona.edu:~/GELLS-DEM/results/ ./results/
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
    "repo_path": "~/GELLS-DEM",
    "python_module": "python/3.11/3.11.4",
    "venv_path": "/groups/your_pi_group/gells_env",
    "partition": "standard",
    "cpus": 4,
    "walltime": "04:00:00"
}
```

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
