## ==========================================================================
# Reconstruction script for meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat
# Uses MRITwixTools for data loading, GROG gridding + FFT-based subspace
# reconstruction, with visualization via CairoMakie.
## ==========================================================================

using MRISubspaceRecon
using MRITwixTools
using IterativeSolvers
using LinearAlgebra
using FFTW
using JLD2
using CairoMakie

## ==========================================================================
# 1) Load raw data with MRITwixTools
## ==========================================================================
raw_file = joinpath(@__DIR__, "..", "..", "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat")
println("Reading twix file: $raw_file")
println("This may take a moment...")
twix_raw = read_twix(raw_file)

# Handle multi-raid files (VD/VE/XA): read_twix returns a Vector{TwixObj}
if isa(twix_raw, Vector)
    println("Multi-raid file detected with $(length(twix_raw)) measurement(s).")
    println("Using last measurement (index $(length(twix_raw))).")
    twix = twix_raw[end]
else
    twix = twix_raw
end

# Display available scan types and header info
println("\nTwix object: ", twix)
println("Available fields: ", collect(propertynames(twix)))

# Print data dimensions
println("\nImage data full size: ", fullSize(twix.image))
println("Image data squeezed size: ", sqzSize(twix.image))
println("Image data squeezed dims: ", sqzDims(twix.image))

## ==========================================================================
# 2) Extract parameters from header
## ==========================================================================
# Try to extract matrix size from header
Nx = try
    Int(twix.hdr.MeasYaps.sKSpace.lBaseResolution)
catch
    128  # fallback
end
println("\nBase resolution: Nx = $Nx")

# Search for additional useful parameters
println("\nSearching header for relevant parameters...")
try
    results = search(twix.hdr, "lBaseResolution", "ReadoutOversamplingFactor")
    println(results)
catch e
    println("Header search: ", e)
end

## ==========================================================================
# 3) Read image data
## ==========================================================================
# Enable readout oversampling removal in MRITwixTools (does it during reading,
# so we never allocate the full oversampled array)
twix.image.removeOS = true

println("\nReading image data (with OS removal)...")
data_raw = getdata(twix.image)
println("Raw data size: $(size(data_raw))")
println("Raw data type: $(eltype(data_raw))")

# Extract dimensions from fullSize
# fullSize returns [NCol, NCha, NLin, NPar, NSli, NAve, NPhs, NEco, NRep, NSet, NSeg, ...]
full_sz = fullSize(twix.image)
Nr_raw = full_sz[1]    # NCol after OS removal (= Nx since removeOS=true)
Ncoil = full_sz[2]     # number of receive coils
Nr = Nx                # readout samples after OS removal = base resolution
Ncyc = 3               # number of cycles (from filename "3cyc")

# Determine Nt from data shape
# For radial/kooshball data: dims beyond Col and Cha encode spokes
# Total spokes = Ncyc * Nt
Nspokes_total = prod(full_sz[3:end])
Nt = Nspokes_total ÷ Ncyc

println("\nDerived parameters:")
println("  Nx (matrix size):    $Nx")
println("  Nr_raw (ADC cols):   $Nr_raw")
println("  Nr (expected):       $Nr")
println("  Ncoil:               $Ncoil")
println("  Ncyc:                $Ncyc")
println("  Nt (time frames):    $Nt")
println("  Total spokes:        $Nspokes_total")

## ==========================================================================
# 4) Reshape data for reconstruction
## ==========================================================================
# getdata returns data shaped according to sqzDims.
# We need to reshape to (Nr*Ncyc, Nt, Ncoil) for the reconstruction pipeline.
#
# Typical radial data from twix: (NCol, NCha, NLin, ...) where
#   NCol = readout samples, NCha = coils, NLin = spoke index
# We treat the spokes as being organized into Ncyc spokes per time frame × Nt frames.

println("\n  Raw data memory: $(round(sizeof(data_raw) / 1e9, digits=2)) GB")

# Reshape in-place (no copy) to collapse spoke dimensions
# After removeOS, NCol = Nx (half the original oversampled readout)
data_raw_rs = reshape(data_raw, Nr_raw, Ncoil, :)
# data_raw_rs is (NCol, Ncoil, Nspokes_total) — just a view, no allocation

# Adjust Nr if it doesn't match what we got after OS removal
if Nr_raw != Nr
    println("  Note: Nr_raw=$Nr_raw after OS removal, adjusting Nr")
    Nr = Nr_raw
end

# Permute (NCol, Ncoil, Nspokes) -> (Nr*Ncyc, Nt, Ncoil) in one allocation
# Allocate final array directly at target shape
data = Array{ComplexF32}(undef, Nr * Ncyc, Nt, Ncoil)

# Copy coil-by-coil to avoid permutedims allocation
for icoil in 1:Ncoil
    # View of this coil's data: (Nr_raw, Nspokes_total)
    coil_data = @view data_raw_rs[:, icoil, :]
    # Reshape spokes into (Nr, Ncyc*Nt) -> (Nr, Ncyc, Nt) -> (Nr*Ncyc, Nt)
    coil_reshaped = reshape(coil_data, Nr, Ncyc, Nt)
    coil_final = reshape(coil_reshaped, Nr * Ncyc, Nt)
    data[:, :, icoil] .= ComplexF32.(coil_final)
end

# Free the original raw data to reclaim memory
data_raw = nothing
data_raw_rs = nothing
GC.gc()

println("  Data reshaped to: $(size(data))")
println("  Data memory: $(round(sizeof(data) / 1e9, digits=2)) GB")

## ==========================================================================
# 5) Set reconstruction parameters
## ==========================================================================
img_shape = (Nx, Nx, Nx)    # 3D kooshball reconstruction
Ncoeff = 4                  # number of subspace coefficients

# For 2D radial, uncomment:
# img_shape = (Nx, Nx)

println("\nReconstruction parameters:")
println("  img_shape: $img_shape")
println("  Ncoeff:    $Ncoeff")

## ==========================================================================
# 6) Load subspace basis
## ==========================================================================
basis_file = joinpath(@__DIR__, "bases_network_3T_R01_brain.jld2")
println("\nLoading subspace basis from: $basis_file")
basis_data = load(basis_file)

# Print available keys to help identify the correct variable name
println("Available keys in basis file: ", keys(basis_data))

# Adjust the key below to match your file (e.g., "U", "basis", "Phi", etc.)
# U should be of size (Nt, Ncoeff)
U = first(values(basis_data))
if ndims(U) == 1
    U = reshape(U, :, 1)
end
if size(U, 1) < Nt
    error("Basis has $(size(U,1)) time frames but data has Nt=$Nt. Check parameters.")
end
U = ComplexF32.(U[1:Nt, 1:Ncoeff])
println("Subspace basis size: $(size(U)) — (Nt, Ncoeff)")

## ==========================================================================
# 7) Generate trajectory
## ==========================================================================
println("\nGenerating 3D kooshball trajectory with golden-ratio sampling...")
trj = traj_kooshball_goldenratio(Nr, Ncyc, Nt; adc_dim=true)
# trj dimensions: (3, Nr, Ncyc, Nt)
println("Trajectory size: $(size(trj))")

# For 2D radial instead, uncomment:
# trj = traj_2d_radial_goldenratio(Nr, Ncyc, Nt; adc_dim=true)
# println("2D Trajectory size: $(size(trj))")

## ==========================================================================
# 8) Reshape trajectory for GROG
## ==========================================================================
# GROG expects trj as (Ndim, Nr*Ncyc, Nt)
trj_rs = reshape(trj, size(trj, 1), Nr * Ncyc, Nt)
println("Trajectory reshaped to: $(size(trj_rs))")

## ==========================================================================
# 9) GROG gridding (non-Cartesian → Cartesian)
## ==========================================================================
println("\nPerforming GROG gridding...")
t_grog = @elapsed trj_cart = radial_grog!(data, trj_rs, Nr, img_shape)
println("GROG gridding complete. Time: $(round(t_grog, digits=1)) s")
println("Cartesian trajectory size: $(size(trj_cart)), eltype: $(eltype(trj_cart))")

## ==========================================================================
# 10) Estimate coil sensitivity maps
## ==========================================================================
println("\nEstimating coil sensitivity maps (ESPIRiT)...")
t_cmaps = @elapsed cmaps = calculate_coil_maps(data, trj_cart, img_shape; U, verbose=true)
println("Coil maps estimated. Time: $(round(t_cmaps, digits=1)) s")
println("Number of coil maps: $(length(cmaps)), each of size: $(size(cmaps[1]))")

## ==========================================================================
# 11) Compute backprojection (adjoint / initial estimate)
## ==========================================================================
println("\nComputing backprojection...")
t_bp = @elapsed xbp = calculate_backprojection(data, trj_cart, cmaps; U)
println("Backprojection complete. Time: $(round(t_bp, digits=1)) s")
println("Backprojection size: $(size(xbp))")

## ==========================================================================
# 12) Build normal operator and solve with CG
## ==========================================================================
println("\nBuilding FFT normal operator...")
t_op = @elapsed AᴴA = FFTNormalOp(img_shape, trj_cart, U; cmaps)
println("Normal operator built. Time: $(round(t_op, digits=1)) s")

Niter = 20
println("\nRunning CG reconstruction ($Niter iterations)...")
t_cg = @elapsed x_recon = cg(AᴴA, vec(xbp); maxiter=Niter, verbose=true)
x_recon = reshape(x_recon, img_shape..., Ncoeff)
println("Reconstruction complete. Time: $(round(t_cg, digits=1)) s")
println("Reconstructed image size: $(size(x_recon))")

## ==========================================================================
# 13) Save results
## ==========================================================================
output_file = joinpath(@__DIR__, "recon_MT_bhsfp_BPJA01_3cyc_grog.jld2")
jldsave(output_file; x_recon, cmaps, xbp, trj_cart, U)
println("\nResults saved to: $output_file")

## ==========================================================================
# 14) Visualization
## ==========================================================================
println("\nGenerating figures...")

# Select central slices for display
if length(img_shape) == 3
    slice_ax = img_shape[3] ÷ 2   # axial (z) slice
    slice_cor = img_shape[2] ÷ 2  # coronal (y) slice
    slice_sag = img_shape[1] ÷ 2  # sagittal (x) slice

    # --- Figure 1: Subspace coefficient maps (axial slices) ---
    fig1 = Figure(size = (1400, 800))
    Label(fig1[0, :], "GROG Reconstruction — Axial Slice (z = $slice_ax)";
          fontsize = 20, font = :bold)

    for ic in 1:Ncoeff
        # Magnitude
        ax = Axis(fig1[1, ic]; title = "Coeff $ic — |x|", aspect = DataAspect())
        heatmap!(ax, abs.(x_recon[:, :, slice_ax, ic])'; colormap = :grays)
        hidedecorations!(ax)

        # Phase
        ax = Axis(fig1[2, ic]; title = "Coeff $ic — ∠x", aspect = DataAspect())
        heatmap!(ax, angle.(x_recon[:, :, slice_ax, ic])'; colormap = :hsv)
        hidedecorations!(ax)
    end

    save("recon_bhsfp_grog_axial.png", fig1; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_axial.png")

    # --- Figure 2: Three orthogonal views of first coefficient ---
    fig2 = Figure(size = (1200, 500))
    Label(fig2[0, :], "GROG Reconstruction — Orthogonal Views (Coeff 1, Magnitude)";
          fontsize = 18, font = :bold)

    ax = Axis(fig2[1, 1]; title = "Axial (z=$slice_ax)", aspect = DataAspect())
    heatmap!(ax, abs.(x_recon[:, :, slice_ax, 1])'; colormap = :grays)
    hidedecorations!(ax)

    ax = Axis(fig2[1, 2]; title = "Coronal (y=$slice_cor)", aspect = DataAspect())
    heatmap!(ax, abs.(x_recon[:, slice_cor, :, 1])'; colormap = :grays)
    hidedecorations!(ax)

    ax = Axis(fig2[1, 3]; title = "Sagittal (x=$slice_sag)", aspect = DataAspect())
    heatmap!(ax, abs.(x_recon[slice_sag, :, :, 1])'; colormap = :grays)
    hidedecorations!(ax)

    save("recon_bhsfp_grog_ortho.png", fig2; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_ortho.png")

    # --- Figure 3: Coil sensitivity maps (axial slice) ---
    fig3 = Figure(size = (1600, 600))
    Label(fig3[0, :], "Estimated Coil Sensitivity Maps — Axial Slice (z=$slice_ax)";
          fontsize = 18, font = :bold)

    ncols = min(Ncoil, 10)
    nrows_cmaps = ceil(Int, Ncoil / ncols)
    for ic in 1:Ncoil
        row = (ic - 1) ÷ ncols + 1
        col = (ic - 1) % ncols + 1
        ax = Axis(fig3[row, col]; title = "Coil $ic", aspect = DataAspect())
        heatmap!(ax, abs.(cmaps[ic][:, :, slice_ax])'; colormap = :grays)
        hidedecorations!(ax)
    end

    save("recon_bhsfp_grog_cmaps.png", fig3; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_cmaps.png")

    # --- Figure 4: Backprojection (initial estimate) ---
    fig4 = Figure(size = (1400, 500))
    Label(fig4[0, :], "Backprojection (Initial Estimate) — Axial Slice (z=$slice_ax)";
          fontsize = 18, font = :bold)

    for ic in 1:Ncoeff
        ax = Axis(fig4[1, ic]; title = "Coeff $ic — |xbp|", aspect = DataAspect())
        heatmap!(ax, abs.(xbp[:, :, slice_ax, ic])'; colormap = :grays)
        hidedecorations!(ax)
    end

    save("recon_bhsfp_grog_backproj.png", fig4; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_backproj.png")

elseif length(img_shape) == 2
    # --- 2D case ---
    fig1 = Figure(size = (1400, 800))
    Label(fig1[0, :], "GROG Reconstruction — 2D";
          fontsize = 20, font = :bold)

    for ic in 1:Ncoeff
        ax = Axis(fig1[1, ic]; title = "Coeff $ic — |x|", aspect = DataAspect())
        heatmap!(ax, abs.(x_recon[:, :, ic])'; colormap = :grays)
        hidedecorations!(ax)

        ax = Axis(fig1[2, ic]; title = "Coeff $ic — ∠x", aspect = DataAspect())
        heatmap!(ax, angle.(x_recon[:, :, ic])'; colormap = :hsv)
        hidedecorations!(ax)
    end

    save("recon_bhsfp_grog_2d.png", fig1; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_2d.png")

    # Coil maps
    fig2 = Figure(size = (1600, 400))
    Label(fig2[0, :], "Estimated Coil Sensitivity Maps"; fontsize = 18, font = :bold)
    ncols = min(Ncoil, 10)
    for ic in 1:Ncoil
        row = (ic - 1) ÷ ncols + 1
        col = (ic - 1) % ncols + 1
        ax = Axis(fig2[row, col]; title = "Coil $ic", aspect = DataAspect())
        heatmap!(ax, abs.(cmaps[ic])'; colormap = :grays)
        hidedecorations!(ax)
    end

    save("recon_bhsfp_grog_2d_cmaps.png", fig2; px_per_unit = 2)
    println("Saved: recon_bhsfp_grog_2d_cmaps.png")
end

display(fig1)

## ==========================================================================
# 15) Print summary
## ==========================================================================
println("\n", "="^60)
println("RECONSTRUCTION SUMMARY")
println("="^60)
println("Raw data file:       $raw_file")
println("Image shape:         $img_shape")
println("Subspace coeffs:     $Ncoeff")
println("Number of coils:     $Ncoil")
println("Time frames (Nt):    $Nt")
println("Cycles (Ncyc):       $Ncyc")
println("ADC samples (Nr):    $Nr")
println("CG iterations:       $Niter")
println("─"^40)
println("GROG time:           $(round(t_grog, digits=1)) s")
println("Coil map time:       $(round(t_cmaps, digits=1)) s")
println("Backprojection time: $(round(t_bp, digits=1)) s")
println("Operator build time: $(round(t_op, digits=1)) s")
println("CG solve time:       $(round(t_cg, digits=1)) s")
println("─"^40)
println("Total time:          $(round(t_grog + t_cmaps + t_bp + t_op + t_cg, digits=1)) s")
println("="^60)