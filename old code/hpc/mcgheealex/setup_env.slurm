#!/bin/bash
#SBATCH --job-name=gels-setup
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p $HOME/GELS/slurm_logs
bash ~/GELS/setup_env.sh
echo "Environment setup finished at $(date)"
