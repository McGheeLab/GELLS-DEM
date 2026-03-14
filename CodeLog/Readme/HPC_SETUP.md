# Running GELLS-DEM on UArizona HPC (Puma)

This guide covers running the simulation on the University of Arizona
HPC cluster from VSCode on your local machine.

**Reference docs:** <https://hpcdocs.hpc.arizona.edu/>

---

## Prerequisites

- A UArizona HPC account (register at <https://hpcdocs.hpc.arizona.edu/registration_and_access/registration/>)
- Membership in a PI allocation group (needed for the `--account` flag)
- [Cisco AnyConnect](https://vpn.hpc.arizona.edu) VPN client installed
- VSCode with the **Remote - SSH** extension

---

## 1. SSH Key Setup (one-time)

Generate an SSH key if you don't already have one:

```bash
ssh-keygen -t rsa
```

Copy it to both the bastion host and the file transfer node:

```bash
ssh-copy-id YOUR_NETID@hpc.arizona.edu
ssh-copy-id YOUR_NETID@filexfer.hpc.arizona.edu
```

> The `filexfer` key is what lets VSCode connect without Duo prompts.

Add the following to `~/.ssh/config` on your local machine:

```
Host uahpcbastion
    HostName hpc.arizona.edu
    User YOUR_NETID
    IdentityFile ~/.ssh/id_rsa

Host uahpcfxfr
    HostName filexfer.hpc.arizona.edu
    User YOUR_NETID
    IdentityFile ~/.ssh/id_rsa

Host *.puma.hpc.arizona.edu
    User YOUR_NETID
    IdentityFile ~/.ssh/id_rsa
```

Replace `YOUR_NETID` with your UArizona NetID in all three blocks.

---

## 2. Connect to the HPC VPN

Open Cisco AnyConnect and connect to:

```
vpn.hpc.arizona.edu
```

> **Important:** This is the HPC-specific VPN, not the regular UArizona VPN.
> It is required so that VSCode can reach Puma compute nodes directly.

---

## 3. Clone the Repository on the Cluster

```bash
ssh YOUR_NETID@hpc.arizona.edu
# At the bastion prompt, type: shell

cd ~
git clone https://github.com/McGheeLab/GELLS-DEM.git
```

### Storage locations

| Path | Quota | Use for |
|------|-------|---------|
| `/home/uXX/NETID` | 50 GB | Code, environments |
| `/groups/PI_NETID` | 500 GB | Large output data |
| `/xdisk/PI_NETID` | up to 20 TB | Bulk results (request from PI) |

---

## 4. Set Up the Python Environment (one-time)

This must run on a **compute node**, not a login node:

```bash
interactive -a YOUR_GROUP -n 4 -t 1:00:00
bash ~/GELLS-DEM/setup_hpc_env.sh
```

This creates a virtual environment at `~/gells-dem-env` with Python 3.11,
numpy, scipy, and matplotlib.

---

## 5. Connect VSCode to a Compute Node

1. **Start an interactive session** (from an SSH terminal):

   ```bash
   interactive -a YOUR_GROUP -n 4 -t 8:00:00
   ```

2. **Get the hostname:**

   ```bash
   hostname
   # Example output: r6u24n2.puma.hpc.arizona.edu
   ```

3. **In VSCode:**
   - Click the `><` icon (bottom-left corner)
   - Select **Connect to Host...** > **+ Add New SSH Host...**
   - Enter: `ssh YOUR_NETID@r6u24n2.puma.hpc.arizona.edu`
     (use the hostname from step 2)
   - Select the config file to update (`~/.ssh/config`)
   - Click **Connect** on the new host entry

4. **Open your project folder:** `~/GELLS-DEM`

> **Notes:**
> - The compute node hostname changes each session -- you will re-enter it each time.
> - VSCode Remote SSH only works on **Puma** (not Ocelote or El Gato).
> - Do **not** connect to the bastion host (`hpc.arizona.edu`) -- it has a 10 MB
>   quota and will fill up.

---

## 6. Running Simulations

### Option A: Interactive (from VSCode terminal)

```bash
module load python/3.11/3.11.4
source ~/gells-dem-env/bin/activate

# Default parameters
python3 run_hpc_headless.py

# With parameter overrides
python3 run_hpc_headless.py --E_modulus 5.0 --t_total 96 --output-dir results_E5

# See all options
python3 run_hpc_headless.py --help
```

Figures are saved to the `--output-dir` directory (default: `results/`).

### Option B: Batch job (unattended)

First, edit `run_hpc.slurm` and replace `YOUR_GROUP` with your PI's
allocation group:

```bash
# Find your group name
va

# Edit the script
nano run_hpc.slurm   # or edit in VSCode
```

Submit and monitor:

```bash
sbatch run_hpc.slurm
squeue --user YOUR_NETID       # check status (PD=pending, R=running)
cat gells-dem_JOBID.out        # view output log
```

### Batch job defaults

| Resource | Value | Modify in `run_hpc.slurm` |
|----------|-------|---------------------------|
| Partition | `standard` | `--partition` |
| CPUs | 4 | `--cpus-per-task` |
| Memory | 16 GB | `--mem` |
| Wall time | 4 hours | `--time` |

---

## 7. Retrieving Results

From VSCode you can browse and download files directly through the
file explorer.

Alternatively, use `scp` from your local machine:

```bash
scp -r YOUR_NETID@filexfer.hpc.arizona.edu:~/GELLS-DEM/results ./results
```

---

## Quick Reference

```bash
# Connect to HPC VPN
# vpn.hpc.arizona.edu via Cisco AnyConnect

# SSH in
ssh YOUR_NETID@hpc.arizona.edu    # then type 'shell'

# Start interactive session
interactive -a YOUR_GROUP -n 4 -t 8:00:00

# Activate environment
module load python/3.11/3.11.4
source ~/gells-dem-env/bin/activate

# Run
python3 run_hpc_headless.py --output-dir results

# Submit batch job
sbatch run_hpc.slurm

# Check job status
squeue --user YOUR_NETID
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| VSCode connection hangs | Make sure you're on `vpn.hpc.arizona.edu`, not the regular UA VPN |
| Duo prompt blocks VSCode | Re-do SSH key setup for `filexfer.hpc.arizona.edu` |
| `ModuleNotFoundError` | Run `module load python/3.11/3.11.4` before activating the venv |
| `cannot open display` / matplotlib error | Use `run_hpc_headless.py` or set `export MPLBACKEND=Agg` |
| Home directory full | Move `~/.cache` and `~/.conda` to `/groups` or `/xdisk` |
| Account locked | 3 failed password attempts = 1 hour lockout; wait or contact HPC support |

---

## Files

| File | Purpose |
|------|---------|
| `setup_hpc_env.sh` | One-time Python environment setup |
| `run_hpc.slurm` | SLURM batch job script |
| `run_hpc_headless.py` | Headless simulation runner with CLI param overrides |
