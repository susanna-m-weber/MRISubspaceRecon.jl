

using MRISubspaceRecon
using MRITwixTools
using IterativeSolvers
using FFTW
using NonuniformFFTs
using LinearAlgebra
using JLD2
using Plots
using CUDA

# check GPU
# Check GPU availability
if CUDA.functional()
    println("✓ GPU detected: $(CUDA.device())")
    println("  GPU memory: $(round(CUDA.total_memory() / 1e9, digits=1)) GB")
    USE_GPU = true
else
    println("✗ No GPU available — running on CPU")
    USE_GPU = false
end
using CUDA

println("CUDA functional: ", CUDA.functional())
println("GPU device: ", CUDA.functional() ? CUDA.device() : "none")

# Load raw data
raw_file = joinpath(@__DIR__, "..", "..", "..", "..", "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat")
println("Loading: $raw_file")
twix_raw = read_twix(raw_file; verbose=false)
twix = isa(twix_raw, Vector) ? twix_raw[end] : twix_raw

# Extract parameters from header
Nx = Int(twix.hdr["MeasYaps.sKSpace.lBaseResolution"])
println("Base resolution: Nx = $Nx")

# Read image data with oversampling removal
twix.image.removeOS = true
println("Reading image data (with OS removal)...")
data_raw = getdata(twix.image)
println("Raw data shape: $(size(data_raw))")

# Get dimensions
full_sz = fullSize(twix.image)
Nr = size(data_raw, 1)    # readout samples after OS removal
Ncoil = full_sz[2]        # number of coils
NLin = full_sz[3]         # lines
NAve = full_sz[6]         # averages (= spokes per time frame)
NRep = full_sz[9]         # repetitions
Nspokes_total = NLin * NAve * NRep

println("Nr=$Nr, Ncoil=$Ncoil, NLin=$NLin, NAve=$NAve, NRep=$NRep")
println("Total spokes: $Nspokes_total")

#  Load subspace basis
basis_file = joinpath(@__DIR__, "..", "..", "bases_network_3T_R01_brain.jld2")
println("\nLoading basis from: $basis_file")
basis_data = load(basis_file)
U_full = ComplexF32.(basis_data["U"])
println("Basis \"U\" size: $(size(U_full))")

# Determine Nt and Ncyc
Nt = size(U_full, 1)
Ncyc = 3

Nc = 4  # number of subspace coefficients
U = U_full[1:Nt, 1:Nc]

println("Nt=$Nt, Ncyc=$Ncyc, Nc=$Nc")
println("Subspace basis: $(size(U))")

#  Reshape data
# Data format needed: (Nr*Ncyc, Nt, Ncoil) for 3D input, or (Nr, Ncyc, Nt, Ncoil) for 4D
println("\nReshaping data...")
data_flat = reshape(data_raw, Nr, Ncoil, Nspokes_total)

# 4D format: (Nr, Ncyc, Nt, Ncoil)
data = Array{ComplexF32}(undef, Nr, Ncyc, Nt, Ncoil)

# Map flat spoke index to (icyc, it) using trajectory convention:
# traj_kooshball_goldenratio reshapes as (Nt, Ncyc) then transposes to (Ncyc, Nt)
# So flat_spoke_idx = (icyc-1)*Nt + it
for icoil in 1:Ncoil
    for it in 1:Nt
        for icyc in 1:Ncyc
            spoke_idx = (icyc - 1) * Nt + it
            data[:, icyc, it, icoil] .= ComplexF32.(@view data_flat[:, icoil, spoke_idx])
        end
    end
end

data_raw = nothing
data_flat = nothing
GC.gc()

println("Data reshaped to: $(size(data))")
println("Data memory: $(round(sizeof(data)/1e9, digits=2)) GB")

# # Generate trajectory
# 3D kooshball trajectory with golden-ratio sampling
# float values in range k ∈ [-0.5, 0.5)
img_shape = (Nx, Nx, Nx)
println("\nimg_shape: $img_shape")

trj = traj_kooshball_goldenratio(Nr, Ncyc, Nt; adc_dim=false)
trj = Float32.(trj)
println("Trajectory size: $(size(trj))")

# Reshape data and trajectory to 3D format for GPU compatibility:
# data: (Nr*Ncyc, Nt, Ncoil), trj: (3, Nr*Ncyc, Nt)
data = reshape(data, :, Nt, Ncoil)
trj = reshape(trj, size(trj, 1), :, Nt)
println("\nReshaped for reconstruction:")
println("  data: $(size(data))")
println("  trj:  $(size(trj))")

# Transfer to GPU if available
if USE_GPU
    println("\nTransferring data to GPU...")
    data = CuArray(data)
    trj = CuArray(trj)
    U = CuArray(U)
    println("  GPU memory used: $(round((CUDA.total_memory() - CUDA.available_memory()) / 1e9, digits=2)) GB")
end

println("typeof(trj) = $(typeof(trj))")
println("typeof(U) = $(typeof(U))")
println("typeof(data) = $(typeof(data))")

#  Coil sensitivity maps
# auto-calibrated from k-space using ESPIRiT
println("\nEstimating coil maps (ESPIRiT)...")
# Compute coil maps on CPU (less memory-intensive, avoids GPU OOM during calibration)
# Then transfer results to GPU for the reconstruction
println("  (Computing coil maps on CPU to avoid GPU memory pressure)")
if USE_GPU
    data_cpu = Array(data)
    trj_cpu = Array(trj)
    U_cpu = Array(U)
else
    data_cpu = data
    trj_cpu = trj
    U_cpu = U
end
t_cmaps = @elapsed cmaps = calculate_coil_maps(data_cpu, trj_cpu, img_shape; U=U_cpu, verbose=true)

# Transfer coil maps to GPU if needed
if USE_GPU
    cmaps = [CuArray(c) for c in cmaps]
    data_cpu = nothing
    trj_cpu = nothing
    U_cpu = nothing
    GC.gc()
end
println("Coil maps estimated. Time: $(round(t_cmaps, digits=1)) s")
println("Number of coil maps: $(length(cmaps)), size: $(size(cmaps[1]))")

# # Normal operator and adjoint
# Compute the NFFT-based normal operator:
println("\nBuilding NFFT normal operator...")
t_op = @elapsed AᴴA = NFFTNormalOp(img_shape, trj, U; cmaps)
println("Normal operator built. Time: $(round(t_op, digits=1)) s")
println(AᴴA)

# Compute the adjoint NUFFT (backprojection):
println("\nComputing backprojection...")
t_bp = @elapsed b = calculate_backprojection(data, trj, cmaps; U, density_compensation=:radial_3D, verbose=true)
println("Backprojection complete. Time: $(round(t_bp, digits=1)) s")
println("size(b) = $(size(b))")

# Visualize backprojection (central axial slice)
slice_z = Nx ÷ 2
b_plot = USE_GPU ? Array(b) : b
p = heatmap(abs.(b_plot[:, :, slice_z, 1])', title="Backprojection — Coeff 1",
            colorbar=true, aspect_ratio=1, color=:grays)
display(p)
savefig("tutorial_backprojection.png")
println("Saved: tutorial_backprojection.png")

#  Iterative reconstruction with CG
# Solve the inverse problem with conjugate gradient:
Niter = 20
println("\nRunning CG ($Niter iterations)...")
t_cg = @elapsed xr = cg(AᴴA, vec(b); maxiter=Niter, verbose=true)
xr = reshape(xr, Nx, Nx, Nx, Nc)

# Transfer back to CPU for plotting/saving
if USE_GPU
    xr = Array(xr)
    b = Array(b)
    cmaps = [Array(c) for c in cmaps]
end
println("Reconstruction complete. Time: $(round(t_cg, digits=1)) s")
println("Reconstructed image size: $(size(xr))")

# Visualize results
println("\nVisualizing results...")

# Axial slice through all coefficients
slice_z = Nx ÷ 2
p1 = plot(layout=(2, Nc), size=(350*Nc, 700))
for ic in 1:Nc
    heatmap!(p1, abs.(xr[:, :, slice_z, ic])', subplot=ic,
             ticks=[], colorbar=false, title="|Coeff $ic|", color=:grays)
    heatmap!(p1, angle.(xr[:, :, slice_z, ic])', subplot=Nc+ic,
             ticks=[], colorbar=false, title="∠Coeff $ic", color=:hsv)
end
display(p1)
savefig("tutorial_recon_axial.png")
println("Saved: tutorial_recon_axial.png")

# Orthogonal views of first coefficient
slice_y = Nx ÷ 2
slice_x = Nx ÷ 2
p2 = plot(layout=(1, 3), size=(1200, 400))
heatmap!(p2, abs.(xr[:, :, slice_z, 1])', subplot=1,
         ticks=[], colorbar=false, title="Axial (z=$slice_z)", color=:grays, aspect_ratio=1)
heatmap!(p2, abs.(xr[:, slice_y, :, 1])', subplot=2,
         ticks=[], colorbar=false, title="Coronal (y=$slice_y)", color=:grays, aspect_ratio=1)
heatmap!(p2, abs.(xr[slice_x, :, :, 1])', subplot=3,
         ticks=[], colorbar=false, title="Sagittal (x=$slice_x)", color=:grays, aspect_ratio=1)
display(p2)
savefig("tutorial_recon_ortho.png")
println("Saved: tutorial_recon_ortho.png")

#  Save results
output_file = joinpath(@__DIR__, "..", "..", "recon_tutorial.jld2")
jldsave(output_file; xr, cmaps, b, U)
println("\nResults saved to: $output_file")

# Summary
println("\n", "="^60)
println("RECONSTRUCTION SUMMARY")
println("="^60)
println("Image shape:         $img_shape")
println("Subspace coeffs:     $Nc")
println("Number of coils:     $Ncoil")
println("Time frames (Nt):    $Nt")
println("Spokes/frame (Ncyc): $Ncyc")
println("ADC samples (Nr):    $Nr")
println("CG iterations:       $Niter")
println("Method:              NFFT (non-Cartesian)")
println("─"^40)
println("Coil map time:       $(round(t_cmaps, digits=1)) s")
println("Operator build time: $(round(t_op, digits=1)) s")
println("Backprojection time: $(round(t_bp, digits=1)) s")
println("CG solve time:       $(round(t_cg, digits=1)) s")
println("─"^40)
println("Total time:          $(round(t_cmaps + t_op + t_bp + t_cg, digits=1)) s")
println("="^60)