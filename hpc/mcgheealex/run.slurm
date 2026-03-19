#!/bin/bash
#SBATCH --job-name=gels
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# ── Load Python and activate virtual environment ──
module load python/3.11/3.11.4
source $HOME/gels-env/bin/activate

# ── Use non-interactive matplotlib backend (no display on HPC) ──
export MPLBACKEND=Agg

# ── Run simulation ──
cd $HOME/GELS
python3 run_hpc_headless.py --output-dir results

echo "Job $SLURM_JOB_ID finished at $(date)"
echo "Check efficiency: seff $SLURM_JOB_ID"
