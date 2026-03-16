#!/bin/bash
# One-time setup script for GELLS-DEM on UArizona HPC (Puma)
#
# Run this ONCE from an interactive compute node session:
#   interactive -a YOUR_GROUP -n 4 -t 1:00:00
#   bash setup_hpc_env.sh
#
# NOTE: Do NOT run this on login nodes — software installs must
#       happen on compute nodes (UA HPC policy).

set -euo pipefail

echo "=== Setting up GELLS-DEM environment on UArizona HPC ==="

# Load Python module (always specify exact version)
module load python/3.11/3.11.4

# Create virtual environment with access to system-site numpy/scipy
# NOTE: If /home quota is tight (50 GB limit), move venv to /groups:
#   python3 -m venv --system-site-packages /groups/YOUR_GROUP/gells_env
python3 -m venv --system-site-packages $HOME/gells-dem-env
source $HOME/gells-dem-env/bin/activate

pip install --upgrade pip

# Install dependencies
pip install numpy scipy matplotlib

# Optional (for 3D rendering and performance):
# pip install pyvista scikit-image numba imageio tqdm

echo ""
echo "=== Environment ready at $HOME/gells-dem-env ==="
echo "To activate:  module load python/3.11/3.11.4 && source \$HOME/gells-dem-env/bin/activate"
echo "To run:       sbatch run_hpc.slurm"
echo ""
echo "Storage tip: Check /home usage with 'uquota'. If tight, move venv to /groups."
