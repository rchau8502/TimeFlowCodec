#!/bin/bash
# ==============================================================================
# Slurm Batch Submission Script for TimeFlowCodec HPC Worker
# ==============================================================================
#SBATCH --job-name=tfc_hpc_worker
#SBATCH --output=logs/tfc_worker_%j.out
#SBATCH --error=logs/tfc_worker_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --requeue

# 1. Environment & Module Setup (Adapt to your cluster's module hierarchy)
echo "=== Starting HPC Worker Job on $(hostname) at $(date) ==="
echo "Slurm Job ID: ${SLURM_JOB_ID}"

# Load required cluster modules (e.g. GCC, Python, CUDA if available)
module purge 2>/dev/null || true
module load python/3.11 2>/dev/null || module load anaconda3 2>/dev/null || true

# 2. Virtual Environment Setup
VENV_DIR="${HOME}/.venvs/tfc_worker_venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r requirements.txt
    # Optionally install the TimeFlowCodec research package
    pip install timeflowcodec 2>/dev/null || true
else
    source "${VENV_DIR}/bin/activate"
fi

# 3. Credentials & Environment Configuration
# Ensure secrets are sourced securely from the user's protected dotfile
if [ -f "${HOME}/.tfc_cluster.env" ]; then
    source "${HOME}/.tfc_cluster.env"
fi

mkdir -p logs

# 4. Launch Outbound Polling Worker
echo "Starting outbound queue polling loop..."
python3 -u hpc_worker.py

echo "=== HPC Worker Job exited cleanly at $(date) ==="
