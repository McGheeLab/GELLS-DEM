#!/bin/bash
#SBATCH --job-name=gels-b1
#SBATCH --account=mcgheealex
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=27:05:49
#SBATCH --array=1-200%10
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err

module load python/3.11/3.11.4
source $HOME/gels-env/bin/activate
export MPLBACKEND=Agg

cd ~/GELLS-DEM

# V2.6: Results written to /groups (500 GB) instead of /home (50 GB)
RESULTS_BASE="/groups/mcgheealex/results/LHC"
mkdir -p "$RESULTS_BASE"

# Pre-flight disk space check (abort if < 2 GB free)
AVAIL_KB=$(df --output=avail "$RESULTS_BASE" 2>/dev/null | tail -1)
AVAIL_GB=$(( ${AVAIL_KB:-0} / 1048576 ))
if [ "$AVAIL_GB" -lt 2 ]; then
    echo "ABORT: Only ${AVAIL_GB} GB free on results filesystem. Need >= 2 GB."
    exit 1
fi

# Stagger starts to reduce I/O storms (0-30 second random delay)
SLEEP_SEC=$(( RANDOM % 30 ))
echo "  Stagger sleep: ${SLEEP_SEC}s"
sleep $SLEEP_SEC

# Offset into trial_list.txt for this batch
OFFSET=200
LINE_NUM=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

TRIAL=$(sed -n "${LINE_NUM}p" hpc/trial_list.txt)
TRIAL_NAME="${TRIAL%.json}"

if [ -z "$TRIAL" ]; then
    echo "ERROR: No trial found for line $LINE_NUM (task $SLURM_ARRAY_TASK_ID, offset $OFFSET)"
    exit 1
fi

echo "=== Batch 1 task $SLURM_ARRAY_TASK_ID (line $LINE_NUM): $TRIAL ==="
echo "Node: $(hostname), CPUs: $SLURM_CPUS_ON_NODE, Start: $(date)"
echo "Results: $RESULTS_BASE/$TRIAL_NAME"
echo "Free disk: ${AVAIL_GB} GB"

python3 run_hpc_headless.py \
    --trial "Trials/$TRIAL" \
    --output-dir "$RESULTS_BASE/$TRIAL_NAME" \
    --save_fields False \
    --seed 42

echo "Task $SLURM_ARRAY_TASK_ID ($TRIAL) finished at $(date)"
