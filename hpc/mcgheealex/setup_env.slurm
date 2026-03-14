#!/bin/bash
#SBATCH --job-name=gells-setup
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=gells-setup_%j.out
#SBATCH --error=gells-setup_%j.err

bash ~/GELLS-DEM/setup_env.sh
echo "Environment setup finished at $(date)"
