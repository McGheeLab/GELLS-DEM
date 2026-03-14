#!/bin/bash
# One-time setup script for GELLS-DEM on UArizona HPC (Puma)
# Run this ONCE after cloning the repo to a compute node interactive session.
#
# Usage:
#   interactive -a YOUR_GROUP -n 4 -t 1:00:00
#   bash setup_hpc_env.sh

set -euo pipefail

echo "=== Setting up GELLS-DEM environment on UArizona HPC ==="

# Load Python module
module load python/3.11/3.11.4

# Create virtual environment with access to system-site numpy/scipy
python3 -m venv --system-site-packages ~/gells-dem-env
source ~/gells-dem-env/bin/activate

pip install --upgrade pip

# Install dependencies (numpy/scipy are on system-site, but ensure matplotlib)
pip install numpy scipy matplotlib

echo ""
echo "=== Environment ready at ~/gells-dem-env ==="
echo "To activate:  module load python/3.11/3.11.4 && source ~/gells-dem-env/bin/activate"
echo "To run:       sbatch run_hpc.slurm"
