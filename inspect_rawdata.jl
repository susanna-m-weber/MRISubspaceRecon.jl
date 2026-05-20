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
println("  INSPECTING: $(basename(raw_file))")
println("="^70)

twix_raw = read_twix(raw_file; verbose=false)

# Handle multi-raid files (VD/VE/XA): read_twix returns a Vector{TwixObj}
if isa(twix_raw, Vector)
    twix = twix_raw[end]
else
    twix = twix_raw
end

## ==========================================================================
# 2) General file info
## ==========================================================================
println("\n─"^70)
println("  SCAN TYPES")
println("─"^70)
for field in propertynames(twix)
    field == :hdr && continue
    obj = getproperty(twix, field)
    if obj isa MRITwixTools.RawData && obj.meta !== nothing
        println("  $field: $(obj.meta.NAcq) acquisitions, size = $(sqzSize(obj))")
    end
end

## ==========================================================================
# 3) Image data dimensions
## ==========================================================================
println("\n─"^70)
println("  IMAGE DATA DIMENSIONS")
println("─"^70)

full_sz = fullSize(twix.image)
sqz_dims = sqzDims(twix.image)

dim_names = ["NCol", "NCha", "NLin", "NPar", "NSli", "NAve",
             "NPhs", "NEco", "NRep", "NSet", "NSeg", "NIda",
             "NIdb", "NIdc", "NIdd", "NIde"]

for (name, val) in zip(dim_names, full_sz)
    val > 1 && println("  $name = $val")
end

Nr_raw = full_sz[1]
Ncoil = full_sz[2]
Nspokes_total = prod(full_sz[3:end])

println("  Total spokes:  $Nspokes_total")
println("  Acquisitions:  $(twix.image.meta.NAcq)")

## ==========================================================================
# 4) Header parameters
## ==========================================================================
println("\n─"^70)
println("  PROTOCOL PARAMETERS")
println("─"^70)

try; println("  Base resolution:   $(Int(twix.hdr.MeasYaps.sKSpace.lBaseResolution))"); catch; end
try; println("  Readout OS:        $(twix.hdr.MeasYaps.sKSpace.dReadoutOversamplingFactor)"); catch; end
try; println("  FOV readout:       $(twix.hdr.MeasYaps.sSliceArray.asSlice[1].dReadoutFOV) mm"); catch; end
try; println("  FOV phase:         $(twix.hdr.MeasYaps.sSliceArray.asSlice[1].dPhaseFOV) mm"); catch; end
try; println("  Slice thickness:   $(twix.hdr.MeasYaps.sSliceArray.asSlice[1].dThickness) mm"); catch; end
try; TR = twix.hdr.MeasYaps.alTR[1]; println("  TR:                $(TR/1000) ms"); catch; end
try; TE = twix.hdr.MeasYaps.alTE[1]; println("  TE:                $(TE/1000) ms"); catch; end
try; println("  Flip angle:        $(twix.hdr.MeasYaps.adFlipAngleDegree[1])°"); catch; end
try; bw = twix.hdr.MeasYaps.sRXSPEC.alDwellTime[1]; println("  Dwell time:        $bw ns"); catch; end
try; println("  Sequence:          $(twix.hdr.MeasYaps.tSequenceFileName)"); catch; end
try; println("  Protocol:          $(twix.hdr.MeasYaps.tProtocolName)"); catch; end

## ==========================================================================
# 5) Trajectory & timing info
## ==========================================================================
println("\n─"^70)
println("  TRAJECTORY INFO")
println("─"^70)

Ncyc = 3
Nr = Nr_raw ÷ 2
Nt_possible = Nspokes_total ÷ Ncyc
meta = twix.image.meta

println("  Ncyc (from filename): $Ncyc")
println("  Nr (after 2x OS):     $Nr")
println("  Nt (spokes / Ncyc):   $Nt_possible")
println("  Reflected readouts:   $(sum(meta.IsReflected)) / $(meta.NAcq)")

## ==========================================================================
# 6) Subspace basis info (if available)
## ==========================================================================
println("\n─"^70)
println("  SUBSPACE BASIS")
println("─"^70)

basis_file = joinpath(@__DIR__, "bases_network_3T_R01_brain.jld2")
if isfile(basis_file)
    basis_data = load(basis_file)
    for (k, v) in basis_data
        if v isa AbstractArray
            println("  \"$k\": $(typeof(v)), size $(size(v))")
        elseif v isa NamedTuple || v isa Tuple
            println("  \"$k\": NamedTuple (neural network), fields: $(keys(v))")
        else
            println("  \"$k\": $(typeof(v))")
        end
    end
else
    println("  [not found]")
end

## ==========================================================================
# 7) Suggested reconstruction parameters
## ==========================================================================
println("\n─"^70)
println("  SUGGESTED RECON PARAMETERS")
println("─"^70)

Nx_est = Nr_raw ÷ 2
println("  Nx:          $Nx_est")
println("  img_shape:   ($Nx_est, $Nx_est, $Nx_est)")
println("  Nr:          $(Nr_raw ÷ 2)")
println("  Ncyc:        $Ncyc")
println("  Nt:          $Nt_possible")
println("  Ncoil:       $Ncoil")
println("  Ncoeff:      4")
println("  Data memory: $(round(sizeof(ComplexF32) * Nr_raw * Nspokes_total * Ncoil / 1e9, digits=2)) GB")
println("  Image memory (3D): $(round(sizeof(ComplexF32) * Nx_est^3 * 4 / 1e9, digits=2)) GB")

println("\n", "="^70)
println("  DONE")
println("="^70)