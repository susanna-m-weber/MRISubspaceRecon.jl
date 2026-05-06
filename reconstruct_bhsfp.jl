## Reconstruction script for meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat
## Uses GROG gridding + FFT-based subspace reconstruction

using MRISubspaceRecon
using IterativeSolvers
using LinearAlgebra
using FFTW
using JLD2
using RawData # or MRIRawData / RawDataReader — adjust to your Siemens raw data reader

## ==========================================================================
# 1) Load raw data
# ==========================================================================
# Adjust the raw data reader to match your setup. Common options include:
# - RawData.jl, MRIRawData.jl, or a custom Siemens .dat reader
# The data should be shaped as (Nr, Ncyc, Nt, Ncoil) or (Nr*Ncyc, Nt, Ncoil)

raw_file = "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat"
# data, header = read_rawdata(raw_file)  # <-- uncomment and adjust to your reader

# Example expected dimensions after loading:
# Nr   = number of ADC samples per readout (e.g., 2*Nx for oversampling)
# Ncyc = number of cycles (3 based on filename)
# Nt   = number of time frames per cycle
# Ncoil = number of receive coils

## ==========================================================================
# 2) Set reconstruction parameters
# ==========================================================================
# Adjust these based on your protocol / raw data header
Nx = 128                    # image matrix size (adjust to your protocol)
img_shape = (Nx, Nx, Nx)    # 3D kooshball; use (Nx, Nx) for 2D
Nr = 2 * Nx                 # number of ADC samples per readout (with 2x oversampling)
Ncyc = 3                    # number of cycles (from filename)
# Nt = ...                  # number of time frames per cycle (from header/protocol)
# Ncoil = ...               # number of coils (from data)

## ==========================================================================
# 3) Load subspace basis
# ==========================================================================
basis_file = "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc/aux/bases_network_3T_R01_brain.jld2"
U = load(basis_file)        # adjust key name, e.g., load(basis_file, "U") or load(basis_file, "basis")
# U should be of size (Nt, Ncoeff) or (Nt, Ncoeff, Nrep)
# where Ncoeff is the number of subspace coefficients

## ==========================================================================
# 4) Generate trajectory
# ==========================================================================
# For a 3D kooshball with golden-ratio sampling:
trj = traj_kooshball_goldenratio(Nr, Ncyc, Nt; adc_dim=true)
# trj dimensions: (3, Nr, Ncyc, Nt) with adc_dim=true

# For 2D radial golden angle:
# trj = traj_2d_radial_goldenratio(Nr, Ncyc, Nt; adc_dim=true)
# trj = trj[1:2, :, :, :]  # keep only 2 dimensions

## ==========================================================================
# 5) Reshape data for reconstruction
# ==========================================================================
# Ensure data is shaped as (Nr, Ncyc, Nt, Ncoil) for 4D input
# or (Nr*Ncyc, Nt, Ncoil) for 3D input
# data = reshape(data, Nr, Ncyc, Nt, Ncoil)

## ==========================================================================
# 6) GROG gridding (non-Cartesian → Cartesian)
# ==========================================================================
println("Performing GROG gridding...")
trj_cart = radial_grog!(data, trj, Nr, img_shape)
println("GROG gridding complete.")

## ==========================================================================
# 7) Estimate coil sensitivity maps
# ==========================================================================
println("Estimating coil maps...")
cmaps = calculate_coil_maps(data, trj_cart, img_shape; U)
println("Coil maps estimated.")

## ==========================================================================
# 8) Compute backprojection (adjoint / initial estimate)
# ==========================================================================
println("Computing backprojection...")
xbp = calculate_backprojection(data, trj_cart, cmaps; U)
println("Backprojection complete. Size: $(size(xbp))")

## ==========================================================================
# 9) Build normal operator and solve with CG
# ==========================================================================
println("Building FFT normal operator...")
AᴴA = FFTNormalOp(img_shape, trj_cart, U; cmaps)

println("Running CG reconstruction...")
Ncoeff = size(U, 2)
x_recon = cg(AᴴA, vec(xbp), maxiter=20, verbose=true)
x_recon = reshape(x_recon, img_shape..., Ncoeff)
println("Reconstruction complete. Size: $(size(x_recon))")

## ==========================================================================
# 10) Save results
# ==========================================================================
output_file = "recon_MT_bhsfp_BPJA01_3cyc.jld2"
jldsave(output_file; x_recon, cmaps, xbp, trj_cart, U)
println("Results saved to $output_file")