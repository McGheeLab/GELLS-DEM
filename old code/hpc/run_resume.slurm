#!/bin/bash
#SBATCH --job-name=gels-resume
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:09:45
#SBATCH --array=1-16%10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load python/3.11/3.11.4
source $HOME/gels-env/bin/activate
export MPLBACKEND=Agg

cd ~/GELLS-DEM

# V2.6: Results in /groups
RESULTS_BASE="/groups/mcgheealex/results/LHC"

# Pre-flight disk space check
AVAIL_KB=$(df --output=avail "$RESULTS_BASE" 2>/dev/null | tail -1)
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1048576 ))
if [ "$AVAIL_GB" -lt 2 ]; then
    echo "ABORT: Only ${AVAIL_GB} GB free. Need >= 2 GB."
    exit 1
fi

# Stagger starts
sleep $(( RANDOM % 30 ))

TRIAL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" hpc/trial_list.txt)
TRIAL_NAME="${TRIAL%.json}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial for task $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "=== RESUME task $SLURM_ARRAY_TASK_ID: $TRIAL ==="
echo "Node: $(hostname), Start: $(date)"

python3 run_hpc_headless.py \
    --trial "Trials/$TRIAL" \
    --output-dir "$RESULTS_BASE/$TRIAL_NAME" \
    --resume-from "$RESULTS_BASE/$TRIAL_NAME" \
    --save_fields False \
    --seed 42

echo "Resume task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
