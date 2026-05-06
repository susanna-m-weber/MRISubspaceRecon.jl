using MRISubspaceRecon
using ImagePhantoms
using LinearAlgebra
using IterativeSolvers
using FFTW
using CairoMakie

## set parameters
T = Float32
Tint = Int32
Nx = 128
Nc = 4
Nt = 40
Ncoil = 9
img_shape = (Nx, Nx)

## create test image
x = zeros(Complex{T}, Nx, Nx, Nc)
x[:,:,1] = transpose(shepp_logan(Nx))
x[1:end÷2,:,1] .*= exp(1im * π/3)
x[:,:,2] = shepp_logan(Nx)

## coil maps
cmaps = ones(Complex{T}, Nx, Nx, Ncoil)
[cmaps[i,:,2] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,3] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,4] .*= exp( 1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,5] .*= exp(-1im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[i,:,6] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[i,:,7] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,1)]
[cmaps[:,i,8] .*= exp( 2im * π * i/Nx) for i ∈ axes(cmaps,2)]
[cmaps[:,i,9] .*= exp(-2im * π * i/Nx) for i ∈ axes(cmaps,2)]

for i ∈ CartesianIndices(@view cmaps[:,:,1])
    cmaps[i,:] ./= norm(cmaps[i,:])
end
cmaps = [cmaps[:,:,ic] for ic=1:Ncoil]

## set up basis functions
U = randn(Complex{T}, Nt, Nc)
U,_,_ = svd(U)

## simulate data
data = Array{Complex{T}}(undef, Nx, Nx, Nt, Ncoil)
for icoil = 1:Ncoil
    Threads.@threads for i ∈ CartesianIndices(@view x[:,:,1])
        data[i,:,icoil] .= U * x[i,:] .* cmaps[icoil][i]
    end
end

data .= ifftshift(data, (1, 2))
data = reshape(data, size(data,1), size(data,2), size(data,3)*size(data,4))
fft!(data, [1, 2])
data = fftshift(data, (1, 2))
data = reshape(data, Nx*Nx, Nt, Ncoil)

sample_mask = rand(Nx, Nx, Nt) .< 0.8
trj = collect(Iterators.product(1:Nx, 1:Nx, 1:Nt))
kx = reshape(getindex.(trj, 1), (1, Nx*Nx, Nt))
ky = reshape(getindex.(trj, 2), (1, Nx*Nx, Nt))
trj = Tint.(cat(kx, ky; dims=1))
sample_mask = reshape(sample_mask, Nx*Nx, Nt)

## build normal operator and reconstruct
A = FFTNormalOp((Nx,Nx), trj, U; cmaps, sample_mask)

xbp = calculate_backprojection(data, trj, cmaps; U, sample_mask)
xr = cg(A, vec(xbp), maxiter=20)
xr = reshape(xr, Nx, Nx, Nc)

## ---------- Visualization ----------

fig = Figure(size = (1400, 900))

# Plot original image (magnitude and phase for each subspace coefficient)
for ic in 1:Nc
    # Magnitude
    ax = Axis(fig[1, ic]; title = "Original |x| (coeff $ic)", aspect = DataAspect())
    heatmap!(ax, abs.(x[:,:,ic])'; colormap = :grays)
    hidedecorations!(ax)

    # Phase
    ax = Axis(fig[2, ic]; title = "Original ∠x (coeff $ic)", aspect = DataAspect())
    heatmap!(ax, angle.(x[:,:,ic])'; colormap = :hsv)
    hidedecorations!(ax)
end

# Plot reconstructed image (magnitude and phase for each subspace coefficient)
for ic in 1:Nc
    ax = Axis(fig[3, ic]; title = "Recon |x| (coeff $ic)", aspect = DataAspect())
    heatmap!(ax, abs.(xr[:,:,ic])'; colormap = :grays)
    hidedecorations!(ax)

    ax = Axis(fig[4, ic]; title = "Recon ∠x (coeff $ic)", aspect = DataAspect())
    heatmap!(ax, angle.(xr[:,:,ic])'; colormap = :hsv)
    hidedecorations!(ax)
end

# Plot error (magnitude difference)
for ic in 1:Nc
    ax = Axis(fig[5, ic]; title = "Error |x - xr| (coeff $ic)", aspect = DataAspect())
    heatmap!(ax, abs.(x[:,:,ic] .- xr[:,:,ic])'; colormap = :inferno)
    hidedecorations!(ax)
end

Label(fig[0, :], "Cartesian Subspace Reconstruction: Original vs. Reconstructed";
      fontsize = 20, font = :bold)

save("reconstruct_cart_trj_result.png", fig; px_per_unit = 2)
display(fig)

println("\nMax absolute error: ", maximum(abs.(x .- xr)))
println("Relative error:     ", norm(x .- xr) / norm(x))
println("\nFigure saved to: reconstruct_cart_trj_result.png")