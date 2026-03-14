#!/bin/bash
#SBATCH --job-name=gells-dem
#SBATCH --account=YOUR_GROUP          # <- replace with your PI's group name
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=gells-dem_%j.out
#SBATCH --error=gells-dem_%j.err

# ── Load Python and activate virtual environment ──
module load python/3.11/3.11.4

# Activate the project venv (create it first with setup_hpc_env.sh)
source ~/gells-dem-env/bin/activate

# ── Use non-interactive matplotlib backend (no display on HPC) ──
export MPLBACKEND=Agg

# ── Run simulation ──
cd ~/GELLS-DEM
python3 new_dem_0.py

echo "Job finished at $(date)"
