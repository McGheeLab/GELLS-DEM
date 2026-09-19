#!/bin/bash
# Auto-generated environment setup for mcgheealex
# Run ONCE on a compute node (UA HPC: install software on compute nodes, not login nodes):
#   interactive -a mcgheealex -n 4 -t 1:00:00
#   bash setup_env.sh

set -euo pipefail

echo "=== Setting up GELS environment for mcgheealex ==="

module load python/3.11/3.11.4

python3 -m venv --system-site-packages $HOME/gels-env
source $HOME/gels-env/bin/activate

pip install --upgrade pip
pip install numpy scipy matplotlib

echo ""
echo "=== Environment ready at $HOME/gels-env ==="
echo "To activate:"
echo "  module load python/3.11/3.11.4 && source $HOME/gels-env/bin/activate"
echo "To submit a job:"
echo "  sbatch run.slurm"
