## ==========================================================================

using MRITwixTools
using JLD2

## ==========================================================================
# Load twix file
raw_file = joinpath(@__DIR__, "..", "..", "meas_MID00158_FID02539_MT_bhsfp_BPJA01_3cyc.dat")
println("="^70)
println("  INSPECTING: $raw_file")
println("="^70)
println("\nReading twix file (header + MDH)...")
twix_raw = read_twix(raw_file)

# Handle multi-raid files (VD/VE/XA)
if isa(twix_raw, Vector)
    println("Multi-raid file detected with $(length(twix_raw)) measurement(s).")
    println("Using last measurement (index $(length(twix_raw))).")
    twix = twix_raw[end]
else
    twix = twix_raw
end
println("Done.\n")

## ==========================================================================
#  General file info
println("─"^70)
println(" FILE INFO")
println("─"^70)
println("Available scan types: ", collect(propertynames(twix)))
println()

# Check which scan types have data
for field in propertynames(twix)
    field == :hdr && continue
    obj = getproperty(twix, field)
    if obj isa MRITwixTools.RawData && obj.meta !== nothing
        println(" $field: $(obj.meta.NAcq) acquisitions, squeezed size = $(sqzSize(obj))")
    elseif obj isa MRITwixTools.RawData
        println(" $field: [no data]")
    end
end

## ==========================================================================
# Image data dimensions
println()
println("─"^70)
println(" DATA DIMENSIONS")
println("─"^70)

full_sz = fullSize(twix.image)
sqz_sz = sqzSize(twix.image)
sqz_dims = sqzDims(twix.image)

dim_names = ["NCol", "NCha", "NLin", "NPar", "NSli", "NAve",
             "NPhs", "NEco", "NRep", "NSet", "NSeg", "NIda",
             "NIdb", "NIdc", "NIdd", "NIde"]

println("\nFull size (all 16 dimensions):")
for (name, val) in zip(dim_names, full_sz)
    println("  $name = $val")
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


NPar = full_sz[4]
NSli = full_sz[5]
thickness = twix.hdr["MeasYaps.sSliceArray.asSlice.0.dThickness"]  # 192.0

println()
print("Trajectory: ")
if NPar > 1
    print("3D Cartesian/Stack-of-stars (NPar = $NPar)")
elseif NSli > 1
    print("2D multi-slice (NSli = $NSli)")
elseif thickness > 50  # thick slab
    print("3D radial (kooshball) — single thick slab ($thickness mm)")
else
    print("2D single-slice ($thickness mm)")
end

println()
## ==========================================================================
#  Header parameters
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

# Helper function to safely get a header value by path
function hdr_get(hdr, path)
    try
        return hdr[path]
    catch
        return nothing
    end
end

# Readout oversampling
ros = hdr_get(twix.hdr, "MeasYaps.sKSpace.dReadoutOversamplingFactor")
if ros !== nothing
    println("  Readout oversampling:      $ros")

else
    println("  Readout oversampling:      2x (assumed, NCol/Nx = $(Nr_raw ÷ Int(twix.hdr["MeasYaps.sKSpace.lBaseResolution"])))")
end

# FOV
fov_read = hdr_get(twix.hdr, "MeasYaps.sSliceArray.asSlice.0.dReadoutFOV")
fov_phase = hdr_get(twix.hdr, "MeasYaps.sSliceArray.asSlice.0.dPhaseFOV")
if fov_read !== nothing
    println("  FOV readout:               $fov_read mm")
    println("  FOV phase:                 $fov_phase mm")
else
    println("  FOV:                       [not found]")
end

# Slice thickness
thickness = hdr_get(twix.hdr, "MeasYaps.sSliceArray.asSlice.0.dThickness")
if thickness !== nothing
    println("  Slab thickness:            $thickness mm")
else
    println("  Slice thickness:           [not found]")
end

# TR
TR = hdr_get(twix.hdr, "MeasYaps.alTR.0")
if TR !== nothing
    println("  TR:                        $TR μs ($(TR/1000) ms)")

else
    println("  TR:                        [not found]")
end

# TE
TE = hdr_get(twix.hdr, "MeasYaps.alTE.0")
if TE !== nothing
    println("  TE:                        $TE μs ($(TE/1000) ms)")
else
    println("  TE:                        [not found]")
end

# Dwell time / Bandwidth
bw = hdr_get(twix.hdr, "MeasYaps.sRXSPEC.alDwellTime.0")
if bw !== nothing
    println("  Dwell time:                $bw ns")
    println("  Bandwidth/pixel:           $(round(1e9 / (bw * Nr_raw), digits=1)) Hz/px")

else
    println("  Bandwidth:                 [not found]")
end

# Field strength
B0 = hdr_get(twix.hdr, "Meas.flMagneticFieldStrength")
if B0 !== nothing
    println("  Field strength:            $(round(B0, digits=2)) T")
end

# Radial views
rad_views = hdr_get(twix.hdr, "MeasYaps.sKSpace.lRadialViews")
if rad_views !== nothing
    println("  Radial views:              $(Int(rad_views))")
end

# Sequence name
seq = hdr_get(twix.hdr, "MeasYaps.tSequenceFileName")
if seq !== nothing
    println("  Sequence:                  $seq")

end

# Protocol name
prot = hdr_get(twix.hdr, "MeasYaps.tProtocolName")
if prot !== nothing
    println("  Protocol name:             $prot")

end

# Available header sections
println("\n  Header sections: ", collect(keys(twix.hdr)))

## ==========================================================================
# Trajectory & timing info
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

## ==========================================================================
# Subspace basis info 
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
        end
    end
else
    println("  Basis file not found: $basis_file")
end

## ==========================================================================
# Estimate memory load
println("\n  Estimated data memory: $(round(sizeof(ComplexF32) * Nr_raw * Nspokes_total * Ncoil / 1e9, digits=2)) GB")
println("  Estimated image memory (3D): $(round(sizeof(ComplexF32) * Nx^3 * 4 / 1e9, digits=2)) GB")

## ==========================================================================
# Header search 
println()
println("─"^70)
println("  HEADER SEARCH")
println("─"^70)

search_terms = ["BaseResolution", "Radial", "Spokes", "Repetitions",

                "FlipAngle", "Bandwidth", "FieldStrength",
                "alTR", "alTE", "FOV", "DwellTime", "Thickness"]

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
