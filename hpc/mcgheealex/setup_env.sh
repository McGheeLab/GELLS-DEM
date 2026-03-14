#!/bin/bash
# Auto-generated environment setup for mcgheealex
# Run ONCE on a compute node:
#   interactive -a mcgheealex -n 4 -t 1:00:00
#   bash setup_env.sh

set -euo pipefail

echo "=== Setting up GELLS-DEM environment for mcgheealex ==="

module load python/3.11/3.11.4

python3 -m venv --system-site-packages ~/gells-dem-env
source ~/gells-dem-env/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib

echo ""
echo "=== Environment ready at ~/gells-dem-env ==="
echo "To activate:"
echo "  module load python/3.11/3.11.4 && source ~/gells-dem-env/bin/activate"
echo "To submit a job:"
echo "  sbatch run.slurm"
