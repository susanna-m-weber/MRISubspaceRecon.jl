#!/bin/bash
#SBATCH --job-name=bhsfp_recon
#SBATCH --output=recon_%j.out
#SBATCH --error=recon_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks=1

## ==========================================================================
# SLURM job script for bhsFP GROG reconstruction
#
# Submit with:  sbatch submit_recon.sbatch
# Monitor with: squeue -u $USER
# Output:       recon_<jobid>.out / recon_<jobid>.err
#
# Uses separate Julia environments:
#   environments/inspect/        — quick metadata inspection
#   environments/recon/          — full GROG reconstruction
## ==========================================================================

## --------------------------------------------------------------------------
# Load modules (adjust to your HPC system)
## --------------------------------------------------------------------------
# Option A: If your HPC has a Julia module
# module load julia/1.11

# Option B: If using conda
module load anaconda3 2>/dev/null || module load conda 2>/dev/null || true
source activate mri_recon 2>/dev/null || conda activate mri_recon

## --------------------------------------------------------------------------
# Set environment variables
## --------------------------------------------------------------------------
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Limit BLAS threads to avoid oversubscription with Julia threads
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Navigate to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Node:          $SLURM_NODELIST"
echo "  CPUs:          $SLURM_CPUS_PER_TASK"
echo "  Memory:        ${SLURM_MEM_PER_NODE:-unknown} MB"
echo "  Julia threads: $JULIA_NUM_THREADS"
echo "  Working dir:   $(pwd)"
echo "  Julia version: $(julia --version)"
echo "============================================================"

## --------------------------------------------------------------------------
# Step 1: Inspect raw data (fast, ~1 min)
## --------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] Inspecting raw data..."
echo "─────────────────────────────────────────────"

julia --project=environments/inspect --threads=${JULIA_NUM_THREADS} inspect_rawdata.jl

## --------------------------------------------------------------------------
# Step 2: Full GROG reconstruction (slow, ~1-4 hours)
## --------------------------------------------------------------------------
echo ""
echo "[$(date '+%H:%M:%S')] Starting full GROG reconstruction..."
echo "─────────────────────────────────────────────"

julia --project=environments/recon --threads=${JULIA_NUM_THREADS} reconstruct_bhsfp_grog.jl

## --------------------------------------------------------------------------
# Done
## --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Job completed at $(date)"
echo "============================================================"