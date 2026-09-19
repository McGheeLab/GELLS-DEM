#!/bin/bash
#SBATCH --job-name=gels-b10
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=34:52:50
#SBATCH --array=1-1%200
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load python/3.11/3.11.4
source $HOME/gels-env/bin/activate
export MPLBACKEND=Agg

cd ~/GELS

# Offset into trial_list.txt for this batch
OFFSET=2000
LINE_NUM=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

TRIAL=$(sed -n "${LINE_NUM}p" hpc/trial_list.txt)
TRIAL_NAME="${TRIAL%.json}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial found for line $LINE_NUM (task $SLURM_ARRAY_TASK_ID, offset $OFFSET)"
    exit 1
fi

echo "=== Batch 10 task $SLURM_ARRAY_TASK_ID (line $LINE_NUM): $TRIAL ==="
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE, Start: $(date)"

python3 run_hpc_headless.py \
    --trial "Trials/$TRIAL" \
    --output-dir "results/LHC/$TRIAL_NAME" \
    --seed 1618428158

echo "Task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
