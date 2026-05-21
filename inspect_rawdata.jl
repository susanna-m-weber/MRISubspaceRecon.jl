## ==========================================================================
# Inspect raw data from meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat
# Reads the .dat file using MRITwixTools and prints scan information.
# Does NOT perform reconstruction — just reports metadata.
## ==========================================================================

using MRITwixTools
using JLD2

## ==========================================================================
# 1) Load twix file
## ==========================================================================
raw_file = joinpath(@__DIR__, "..", "..", "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat")
println("="^70)
println("  INSPECTING: $raw_file")
println("="^70)
println("\nReading twix file (header + MDH)...")
twix_raw = read_twix(raw_file)

# Handle multi-raid files (VD/VE/XA): read_twix returns a Vector{TwixObj}
if isa(twix_raw, Vector)
    println("Multi-raid file detected with $(length(twix_raw)) measurement(s).")
    println("Using last measurement (index $(length(twix_raw))).")
    twix = twix_raw[end]
else
    twix = twix_raw
end
println("Done.\n")

## ==========================================================================
# 2) General file info
## ==========================================================================
println("─"^70)
println("  GENERAL FILE INFO")
println("─"^70)
println("Available scan types: ", collect(propertynames(twix)))
println()

# Check which scan types have data
for field in propertynames(twix)
    field == :hdr && continue
    obj = getproperty(twix, field)
    if obj isa MRITwixTools.RawData && obj.meta !== nothing
        println("  📊 $field: $(obj.meta.NAcq) acquisitions, squeezed size = $(sqzSize(obj))")
    elseif obj isa MRITwixTools.RawData
        println("  📊 $field: [no data]")
    end
end

## ==========================================================================
# 3) Image data dimensions
## ==========================================================================
println()
println("─"^70)
println("  IMAGE DATA DIMENSIONS")
println("─"^70)

full_sz = fullSize(twix.image)
sqz_sz = sqzSize(twix.image)
sqz_dims = sqzDims(twix.image)

dim_names = ["NCol", "NCha", "NLin", "NPar", "NSli", "NAve",
             "NPhs", "NEco", "NRep", "NSet", "NSeg", "NIda",
             "NIdb", "NIdc", "NIdd", "NIde"]

println("\nFull size (all 16 dimensions):")
for (name, val) in zip(dim_names, full_sz)
    val > 1 && println("  $name = $val")
end
println("\n  All dims: $full_sz")

println("\nSqueezed size: $sqz_sz")
println("Squeezed dims: $sqz_dims")

Nr_raw = full_sz[1]
Ncoil = full_sz[2]
Nspokes_total = prod(full_sz[3:end])

println("\nKey values:")
println("  ADC samples per readout (NCol): $Nr_raw")
println("  Number of coils (NCha):          $Ncoil")
println("  Total spokes/lines:              $Nspokes_total")
println("  Total acquisitions:              $(twix.image.meta.NAcq)")

## ==========================================================================
# 4) Header parameters
## ==========================================================================
println()
println("─"^70)
println("  PROTOCOL PARAMETERS (from header)")
println("─"^70)

# Base resolution
try
    Nx = Int(twix.hdr.MeasYaps.sKSpace.lBaseResolution)
    println("  Base resolution (Nx):      $Nx")
catch
    println("  Base resolution:           [not found in header]")
end

# Readout oversampling
try
    ros = twix.hdr.MeasYaps.sKSpace.dReadoutOversamplingFactor
    println("  Readout oversampling:      $ros")
catch
    println("  Readout oversampling:      [not found] (typically 2x)")
end

# FOV — try multiple common paths
fov_found = false
try
    fov_read = twix.hdr.MeasYaps.sSliceArray.asSlice[1].dReadoutFOV
    fov_phase = twix.hdr.MeasYaps.sSliceArray.asSlice[1].dPhaseFOV
    println("  FOV readout:               $fov_read mm")
    println("  FOV phase:                 $fov_phase mm")
    fov_found = true
catch; end
if !fov_found
    try
        fov_read = twix.hdr.MeasYaps.sSliceArray.lSize
        println("  sSliceArray.lSize:         $fov_read")
    catch; end
    try
        fov_read = twix.hdr.Phoenix.sSliceArray.asSlice[1].dReadoutFOV
        fov_phase = twix.hdr.Phoenix.sSliceArray.asSlice[1].dPhaseFOV
        println("  FOV readout (Phoenix):     $fov_read mm")
        println("  FOV phase (Phoenix):       $fov_phase mm")
        fov_found = true
    catch; end
end
if !fov_found
    println("  FOV:                       [not found]")
end

# Slice thickness
try
    thickness = twix.hdr.MeasYaps.sSliceArray.asSlice[1].dThickness
    println("  Slice thickness:           $thickness mm")
catch
    try
        thickness = twix.hdr.Phoenix.sSliceArray.asSlice[1].dThickness
        println("  Slice thickness (Phoenix): $thickness mm")
    catch
        println("  Slice thickness:           [not found]")
    end
end

# TR / TE — try multiple paths
try
    TR = twix.hdr.MeasYaps.alTR[1]
    println("  TR:                        $TR μs ($(TR/1000) ms)")
catch
    try
        TR = twix.hdr.Phoenix.alTR[1]
        println("  TR (Phoenix):              $TR μs ($(TR/1000) ms)")
    catch
        # Try searching
        try
            results = search(twix.hdr, "alTR")
            if !isempty(results)
                println("  TR path found:             $(first(results))")
            else
                println("  TR:                        [not found]")
            end
        catch
            println("  TR:                        [not found]")
        end
    end
end

try
    TE = twix.hdr.MeasYaps.alTE[1]
    println("  TE:                        $TE μs ($(TE/1000) ms)")
catch
    try
        TE = twix.hdr.Phoenix.alTE[1]
        println("  TE (Phoenix):              $TE μs ($(TE/1000) ms)")
    catch
        println("  TE:                        [not found]")
    end
end

# Flip angle
try
    fa = twix.hdr.MeasYaps.adFlipAngleDegree[1]
    println("  Flip angle:                $fa°")
catch
    try
        fa = twix.hdr.Phoenix.adFlipAngleDegree[1]
        println("  Flip angle (Phoenix):      $fa°")
    catch
        println("  Flip angle:                [not found]")
    end
end

# Bandwidth / Dwell time
try
    bw = twix.hdr.MeasYaps.sRXSPEC.alDwellTime[1]
    println("  Dwell time:                $bw ns")
    println("  Bandwidth/pixel:           $(round(1e9 / (bw * Nr_raw), digits=1)) Hz/px")
catch
    try
        bw = twix.hdr.Phoenix.sRXSPEC.alDwellTime[1]
        println("  Dwell time (Phoenix):      $bw ns")
    catch
        println("  Bandwidth:                 [not found]")
    end
end

# Sequence name
try
    seq = twix.hdr.MeasYaps.tSequenceFileName
    println("  Sequence:                  $seq")
catch
    try
        seq = twix.hdr.Phoenix.tSequenceFileName
        println("  Sequence (Phoenix):        $seq")
    catch
        println("  Sequence:                  [not found]")
    end
end

# Protocol name
try
    prot = twix.hdr.MeasYaps.tProtocolName
    println("  Protocol name:             $prot")
catch
    try
        prot = twix.hdr.Phoenix.tProtocolName
        println("  Protocol name (Phoenix):   $prot")
    catch
        println("  Protocol name:             [not found]")
    end
end

# Diagnostic: show available top-level header sections
println("\n  Available header sections: ", collect(keys(twix.hdr)))

## ==========================================================================
# 5) Trajectory & timing info
## ==========================================================================
println()
println("─"^70)
println("  TRAJECTORY & ACQUISITION INFO")
println("─"^70)

Ncyc = 3  # from filename
Nr = Nr_raw ÷ 2  # assuming 2x oversampling
Nt_possible = Nspokes_total ÷ Ncyc

println("  Assumed Ncyc (from filename): $Ncyc")
println("  Nr (after removing 2x OS):    $Nr")
println("  Nt (= total_spokes / Ncyc):   $Nt_possible")
println("  Spokes per time frame:        $Ncyc")

# Timing from MDH
meta = twix.image.meta
timestamps = meta.timestamp
time_since_rf = meta.timeSinceRF

println("\n  Timestamp range:     $(minimum(timestamps)) – $(maximum(timestamps))")
println("  Time since last RF:  min=$(minimum(time_since_rf)), max=$(maximum(time_since_rf)) μs")

# Check for reflected readouts
n_reflected = sum(meta.IsReflected)
println("  Reflected readouts:  $n_reflected / $(meta.NAcq) ($(round(100*n_reflected/meta.NAcq, digits=1))%)")

## ==========================================================================
# 6) Subspace basis info (if available)
## ==========================================================================
println()
println("─"^70)
println("  SUBSPACE BASIS")
println("─"^70)

basis_file = joinpath(@__DIR__, "bases_network_3T_R01_brain.jld2")
if isfile(basis_file)
    println("  Basis file: $basis_file")
    basis_data = load(basis_file)
    println("  Keys: ", collect(keys(basis_data)))
    for (k, v) in basis_data
        println("  \"$k\": type = $(typeof(v))")
        if v isa AbstractArray
            println("       size = $(size(v))")
            if ndims(v) >= 2
                println("       → Nt (rows) = $(size(v,1)), max Ncoeff (cols) = $(size(v,2))")
            end
        elseif v isa NamedTuple || v isa Tuple
            println("       (NamedTuple/Tuple — likely a neural network model)")
            println("       fieldnames: $(keys(v))")
        else
            println("       value: $v")
        end
    end
else
    println("  Basis file not found: $basis_file")
end

## ==========================================================================
# 7) Suggested reconstruction parameters
## ==========================================================================
println()
println("─"^70)
println("  SUGGESTED RECONSTRUCTION PARAMETERS")
println("─"^70)

Nx_est = Nr_raw ÷ 2
println("  Nx (matrix size):     $Nx_est")
println("  img_shape (3D):       ($Nx_est, $Nx_est, $Nx_est)")
println("  img_shape (2D):       ($Nx_est, $Nx_est)")
println("  Nr:                   $(Nr_raw ÷ 2) (with 2x OS removal)")
println("  Ncyc:                 $Ncyc")
println("  Nt:                   $Nt_possible")
println("  Ncoil:                $Ncoil")
println("  Ncoeff:               4 (adjust based on basis SVD)")
println("  CG iterations:        20")

println("\n  Estimated data memory: $(round(sizeof(ComplexF32) * Nr_raw * Nspokes_total * Ncoil / 1e9, digits=2)) GB")
println("  Estimated image memory (3D): $(round(sizeof(ComplexF32) * Nx_est^3 * 4 / 1e9, digits=2)) GB")

## ==========================================================================
# 8) Header search (useful fields)
## ==========================================================================
println()
println("─"^70)
println("  HEADER SEARCH")
println("─"^70)

search_terms = ["BaseResolution", "Radial", "Spokes", "Repetitions",
                "FlipAngle", "Bandwidth", "FieldStrength"]

for term in search_terms
    println("\n  Search: \"$term\"")
    try
        results = search(twix.hdr, term)
        if isempty(results)
            println("    [no results]")
        else
            for (i, r) in enumerate(results)
                i > 5 && (println("    ... ($(length(results) - 5) more)"); break)
                println("    $r")
            end
        end
    catch e
        println("    [error: $e]")
    end
end

println()
println("="^70)
println("  INSPECTION COMPLETE")
println("="^70)