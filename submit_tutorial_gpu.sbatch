#!/bin/bash
#SBATCH --job-name=bhsfp_tutorial_gpu
#SBATCH --output=tutorial_gpu_%j.out
#SBATCH --error=tutorial_gpu_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_dev

## ==========================================================================
# SLURM job script for tutorial_realdata.jl on GPU
# Submit with: sbatch submit_tutorial_gpu.sbatch
## ==========================================================================

# Load required modules (adjust to your cluster)
module load cuda 2>/dev/null || true

# Add Julia to PATH — adjust this path to match your installation
export PATH="$HOME/.juliaup/bin:$PATH"

# Verify Julia is found
if ! command -v julia &> /dev/null; then
    echo "ERROR: Julia not found in PATH. Tried: $HOME/.juliaup/bin/julia"
    exit 1
fi
echo "Julia: $(which julia) — $(julia --version)"

# Set environment variables
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Node:          $SLURM_NODELIST"
echo "  CPUs:          $SLURM_CPUS_PER_TASK"
echo "  GPU:           $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  Memory:        ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "  Julia threads: $JULIA_NUM_THREADS"
echo "  Working dir:   $(pwd)"
echo "============================================================"

# Print GPU info
nvidia-smi

echo ""
echo "[$(date '+%H:%M:%S')] Starting tutorial_realdata.jl on GPU..."
echo "─────────────────────────────────────────────────────────────"

while true; do
    echo "$(date '+%H:%M:%S') RSS=$(ps -o rss= -p $$ 2>/dev/null | awk '{print $1/1024/1024 " GB"}')" >> memory_log.txt
    sleep 10
done &
MEM_PID=$!

julia --project=. --threads=${JULIA_NUM_THREADS} docs/src/tutorial_realdata.jl

kill $MEM_PID 2>/dev/null


echo ""
echo "============================================================"
echo "  Job completed at $(date)"
echo "============================================================"