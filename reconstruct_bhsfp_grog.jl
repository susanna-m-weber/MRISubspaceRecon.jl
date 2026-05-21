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

# Extract dimensions from the actual loaded data (accounts for removeOS)
# data_raw from getdata is shaped according to sqzDims after OS removal
data_sz = size(data_raw)
println("\nLoaded data dimensions: $data_sz")

# Also get fullSize for reference (note: fullSize reports pre-OS-removal NCol)
full_sz = fullSize(twix.image)
println("fullSize (pre-OS removal): $full_sz")

# Get actual dimensions from loaded data
# First two dims are NCol (after OS removal) and NCha
Nr_raw = data_sz[1]    # NCol after OS removal
Ncoil = data_sz[2]     # number of receive coils
Nr = Nr_raw            # readout samples = what we actually have
Ncyc = 3               # number of cycles (from filename "3cyc")

# Total spokes = everything beyond the first two dimensions
Nspokes_total = prod(data_sz[3:end])

# Update Nx based on actual readout length after OS removal
if Nr_raw != Nx
    println("  Note: Adjusting Nx from $Nx to $Nr_raw based on actual data after OS removal")
    Nx = Nr_raw
end

println("\nDerived parameters (preliminary):")
println("  Nx (matrix size):    $Nx")
println("  Nr_raw (ADC cols):   $Nr_raw")
println("  Nr (readout pts):    $Nr")
println("  Ncoil:               $Ncoil")
println("  Total spokes:        $Nspokes_total")

## ==========================================================================
# 3b) Load subspace basis to determine Nt
## ==========================================================================
basis_file = joinpath(@__DIR__, "bases_network_3T_R01_brain.jld2")
println("\nLoading subspace basis from: $basis_file")
basis_data = load(basis_file)
println("Available keys in basis file: ", collect(keys(basis_data)))

# Load the subspace basis matrix "U" — size (Nt_basis, Ncoeff_max)
U = ComplexF32.(basis_data["U"])
println("Basis \"U\" size: $(size(U)) — (Nt_basis, Ncoeff_max)")

# Determine Nt and Ncyc from data and basis
# The basis has Nt_basis rows, but the data may have slightly fewer usable time frames
# due to dummy scans or preparation pulses.
# We know from the header that lRadialViews = 60 (spokes per time frame)
Nt_basis = size(U, 1)

# Try to determine Ncyc from header (lRadialViews)
Ncyc = try
    Int(twix.hdr["MeasYaps.sKSpace.lRadialViews"])
catch
    # Fallback: find largest Ncyc that divides Nspokes_total and gives Nt <= Nt_basis
    local ncyc_candidates = [n for n in 1:200 if Nspokes_total % n == 0 && Nspokes_total ÷ n <= Nt_basis]
    isempty(ncyc_candidates) ? error("Cannot determine Ncyc") : last(ncyc_candidates)
end

Nt = Nspokes_total ÷ Ncyc
@assert Nspokes_total % Ncyc == 0 "Total spokes ($Nspokes_total) not divisible by Ncyc ($Ncyc)"

if Nt > Nt_basis
    error("Data Nt ($Nt) exceeds basis Nt ($Nt_basis). Check Ncyc or basis file.")
end

if Nt < Nt_basis
    println("  Note: Basis has $Nt_basis time frames, data has $Nt. Using first $Nt rows of U.")
end

println("\nFinal parameters:")
println("  Nt (from basis):     $Nt")
println("  Ncyc (spokes/frame): $Ncyc")
println("  Ncoil:               $Ncoil")
println("  Nr:                  $Nr")

## ==========================================================================
# 4) Reshape data for reconstruction
## ==========================================================================
# getdata returns data shaped as (NCol, NCha, NLin, NAve, NRep) [squeezed]
# From the header and dimensions:
#   NLin=1140, NAve=60, NRep=3
#   Nt = NLin * NRep = 1140 * 3 = 3420 (time frames)
#   Ncyc = NAve = 60 (spokes per time frame)
#
# The trajectory (traj_kooshball_goldenratio) generates spokes sequentially
# across all time frames: spoke ordering is (Ncyc, Nt) = (60, 3420)
# i.e., for each time frame, Ncyc spokes are acquired.
#
# In the raw data: the "Ave" dimension corresponds to the Ncyc spokes per frame,
# and "Lin" × "Rep" together form the Nt time frames.

println("\n  Raw data memory: $(round(sizeof(data_raw) / 1e9, digits=2)) GB")
println("  Raw data shape: $(size(data_raw))")
println("  Squeezed dims: $(sqzDims(twix.image))")

# data_raw shape is (NCol, NCha, NLin, NAve, NRep) = (Nr, Ncoil, 1140, 60, 3)
# We want: (Nr*Ncyc, Nt, Ncoil) = (Nr*60, 3420, 20)
# where Nt = NLin*NRep and Ncyc = NAve

# Use fullSize to get dimension counts (not affected by squeezing)
# fullSize returns [NCol, NCha, NLin, NPar, NSli, NAve, NPhs, NEco, NRep, NSet, NSeg, ...]
full_sz_img = fullSize(twix.image)
NLin = full_sz_img[3]
NAve = full_sz_img[6]
NRep = full_sz_img[9]

println("  From fullSize: NLin=$NLin, NAve=$NAve, NRep=$NRep")
println("  Expected: Nt = NLin*NRep = $(NLin*NRep), Ncyc = NAve = $NAve")
println("  Actual: Nt=$Nt, Ncyc=$Ncyc")

# Verify the mapping
if NAve == Ncyc && NLin * NRep == Nt
    println("  ✓ Dimension mapping confirmed: NAve=Ncyc, NLin*NRep=Nt")
elseif NLin == Ncyc && NAve * NRep == Nt
    println("  ✓ Alternative mapping: NLin=Ncyc, NAve*NRep=Nt")
    # Swap the interpretation
    NLin, NAve = NAve, NLin
    println("  Swapped: NLin=$NLin (time within rep), NAve=$NAve (cycles)")
elseif NLin * NAve * NRep == Nspokes_total
    # Try: NLin encodes spokes within a group, figure out correct split
    println("  Trying flexible mapping...")
    # Ncyc spokes per frame, Nt frames total
    # One of the dims should be Ncyc, the product of the others should be Nt
    if NLin == Ncyc
        NAve_use = NAve * NRep
        NLin_use = NLin
        println("  Mapping: NLin=$NLin=Ncyc, NAve*NRep=$(NAve*NRep)=Nt")
    elseif NAve == Ncyc
        println("  Mapping: NAve=$NAve=Ncyc, NLin*NRep=$(NLin*NRep)=Nt")
    else
        println("  WARNING: Cannot determine spoke/time mapping automatically.")
        println("  Proceeding with flat sequential ordering.")
    end
else
    error("Dimension mismatch: NLin*NAve*NRep=$(NLin*NAve*NRep) != Nspokes_total=$Nspokes_total")
end

# Reshape data_raw into the final (Nr*Ncyc, Nt, Ncoil) format.
# getdata returns data in squeezed form following sqzDims order.
# We reshape using the actual data dimensions and the mapping determined above.

# First, reshape data_raw to separate all acquisition dimensions
# getdata shape follows sqzDims: ["Col", "Cha", "Lin", "Ave", "Rep"] (only non-singleton)
# But after removeOS, Col is halved. The squeezed data has shape data_sz.
# We need to figure out which dimension is which from the actual sizes.

# The data from getdata is ordered with fastest-varying first:
# (Col, Cha, Lin, Ave, Rep) if all are > 1
# We reshape to separate these explicitly based on what we know:
#   Nspokes_total = NLin * NAve * NRep (verified above)
#   data_raw is (Nr_raw, Ncoil, <remaining dims flattened or separated>)

# Reshape to full 5D based on fullSize ordering: Col, Cha, Lin, NPar=1, NSli=1, Ave, ..., Rep
# Since getdata squeezes singletons, the returned shape is (Nr_raw, Ncoil, NLin, NAve, NRep)
# IF all three (NLin, NAve, NRep) > 1. Let's handle this robustly:

# Figure out the shape getdata returned by checking data_sz
println("  data_raw shape: $data_sz")
println("  Attempting reshape to (Nr=$Nr_raw, Ncoil=$Ncoil, NLin=$NLin, NAve=$NAve, NRep=$NRep)")

expected_elements = Nr_raw * Ncoil * NLin * NAve * NRep
actual_elements = prod(data_sz)
println("  Expected elements: $expected_elements, Actual: $actual_elements")

if expected_elements != actual_elements
    # fullSize might report different dims than what's in the squeezed data
    # Fall back to flat sequential ordering
    println("  WARNING: Element count mismatch. Using flat sequential spoke ordering.")
    data_raw_flat = reshape(data_raw, Nr_raw, Ncoil, Nspokes_total)

    data = Array{ComplexF32}(undef, Nr * Ncyc, Nt, Ncoil)
    println("  Allocating output: $(round(sizeof(data) / 1e9, digits=2)) GB")

    # Sequential ordering: spokes are in acquisition order
    # Map to (Ncyc, Nt): spoke_idx goes 1..Nspokes_total
    # traj_kooshball_goldenratio generates (Ncyc, Nt) ordering
    for icoil in 1:Ncoil
        for it in 1:Nt
            for icyc in 1:Ncyc
                spoke_idx = (it - 1) * Ncyc + icyc
                out_row_start = (icyc - 1) * Nr + 1
                out_row_end = icyc * Nr
                data[out_row_start:out_row_end, it, icoil] .= ComplexF32.(@view data_raw_flat[:, icoil, spoke_idx])
            end
        end
    end
else
    data_5d = reshape(data_raw, Nr_raw, Ncoil, NLin, NAve, NRep)

    data = Array{ComplexF32}(undef, Nr * Ncyc, Nt, Ncoil)
    println("  Allocating output: $(round(sizeof(data) / 1e9, digits=2)) GB")

    # Fill: time frame it = (irep-1)*NLin + ilin, cycle icyc = iave
    # This matches traj_kooshball_goldenratio which generates angles as:
    #   phi = reshape(phi_all, Nt, Ncyc) then transposed to (Ncyc, Nt)
    for icoil in 1:Ncoil
        for irep in 1:NRep
            for ilin in 1:NLin
                it = (irep - 1) * NLin + ilin
                for icyc in 1:Ncyc
                    out_row_start = (icyc - 1) * Nr + 1
                    out_row_end = icyc * Nr
                    data[out_row_start:out_row_end, it, icoil] .= ComplexF32.(@view data_5d[:, icoil, ilin, icyc, irep])
                end
            end
        end
    end
end

# Free the original raw data to reclaim memory
data_raw = nothing
data_5d = nothing
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
println("  Nx:        $Nx")
println("  Nr:        $Nr")
println("  img_shape: $img_shape")
println("  Ncoeff:    $Ncoeff")
println("  Estimated image memory: $(round(sizeof(ComplexF32) * prod(img_shape) * Ncoeff / 1e9, digits=2)) GB")

## ==========================================================================
# 6) Select subspace coefficients from basis (already loaded in step 3b)
## ==========================================================================
Ncoeff = 4
U = U[1:Nt, 1:Ncoeff]
println("\nSubspace basis used: $(size(U)) — (Nt, Ncoeff)")

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