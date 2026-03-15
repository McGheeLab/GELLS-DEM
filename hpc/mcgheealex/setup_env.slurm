#!/bin/bash
#SBATCH --job-name=gells-setup
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1             # Puma: 5 GB per CPU (1 CPU = 5 GB, enough for setup)
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p $HOME/GELLS-DEM/slurm_logs
bash $HOME/GELLS-DEM/hpc/mcgheealex/setup_env.sh
echo "Environment setup finished at $(date)"
