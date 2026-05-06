#!/bin/bash
## ==========================================================================
# Setup conda environment with Julia and run reconstruction scripts on HPC
#
# Uses separate Julia environments for different tasks:
#   environments/inspect/        — lightweight, just reads .dat file metadata
#   environments/test_visualize/ — runs test scripts with phantom data
#   environments/recon/          — full reconstruction from raw data
## ==========================================================================

set -e  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

## --------------------------------------------------------------------------
# 1) Create conda environment with Julia
## --------------------------------------------------------------------------
echo "============================================================"
echo "  STEP 1: Setting up conda environment"
echo "============================================================"


# Activate the environment

echo "Julia version: $(julia --version)"
echo "Julia threads: auto"
echo ""

## --------------------------------------------------------------------------
# 2) Instantiate all Julia environments
## --------------------------------------------------------------------------
echo "============================================================"
echo "  STEP 2: Installing Julia packages for all environments"
echo "============================================================"

# Core package environment
echo "--- Core package (aux/) ---"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Inspect environment (lightweight)
echo "--- Inspect environment ---"
julia --project=environments/inspect -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Test visualization environment
echo "--- Test visualization environment ---"
julia --project=environments/test_visualize -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Reconstruction environment
echo "--- Reconstruction environment ---"
julia --project=environments/recon -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

echo ""
echo "All environments instantiated and precompiled."
echo ""

## --------------------------------------------------------------------------
# 3) Inspect raw data (fast — just metadata)
## --------------------------------------------------------------------------
echo "============================================================"
echo "  STEP 3: Inspecting raw data"
echo "============================================================"

julia --project=environments/inspect --threads=auto inspect_rawdata.jl

## --------------------------------------------------------------------------
# 4) Run the grog_recon_visualize test script
## --------------------------------------------------------------------------
# echo ""
# echo "============================================================"
# echo "  STEP 4: Running grog_recon_visualize.jl (test with phantom)"
# echo "============================================================"

# julia --project=environments/test_visualize --threads=auto test/grog_recon_visualize.jl

## --------------------------------------------------------------------------
# 5) Run the full bhsFP reconstruction
## --------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  STEP 5: Running reconstruct_bhsfp_grog.jl (full recon)"
echo "============================================================"

julia --project=environments/recon --threads=auto reconstruct_bhsfp_grog.jl

echo ""
echo "============================================================"
echo "  ALL DONE"
echo "============================================================"