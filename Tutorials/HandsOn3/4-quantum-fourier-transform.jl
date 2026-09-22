using ITensors
using ITensorMPS
using Plots: plot, plot!
using FFTW: fft

include("resources/tensor_cross/tensor_cross.jl")
include("resources/qtt_utils.jl")
include("resources/quantum_fourier_transform.jl")

"""
    extract_function_values(M::MPS, m::Integer)

Extract 2^m values of a function encoded as an MPS `M`
in the QTT format, on a coarse grid of 2^m evenly spaced
points in [0,1). (Fixes all but the first m bits to zero.)
"""
function extract_function_values(M::MPS, m::Integer)
    sites = siteinds(M)
    n = length(M)
    R = ITensor(1.)
    for j=reverse(m+1:n)
        R *= M[j]*ITensor([1,0],sites[j])
    end
    T = prod([M[i] for i=1:m])*R
    A = Array(T,reverse(sites[1:m])...)
    return reshape(A,2^m)
end

"""
    extract_fourier_transform_values(Mk::MPS, m::Integer)

Given the MPS `Mk` resulting from applying the QFT to a function f(x),
extract values of the continuous Fourier transform

    f̂(k) = ∫f(x) exp(ikx) dx

Returns `(ks, vals)` where `ks` are the 2^(m+1) wavevectors of smallest |k|,
in increasing order, and `vals` the corresponding values of f̂(k). The spacing 
of the `ks` is 2π because f(x) is defined on an interval of length 1.

The bits of `Mk` encode an integer j = 0,1,...,N-1 (N=2^n) with the first
bit the most significant, and the value of `Mk` for that integer is 

    1/√N ∑ₓ f(x) exp(-2πijx) ≈ √N f̂(k)   with  k = -2πj 

Small positive j have all of their leading bits equal to 0. Negative 
integers -j are stored at N-j, so have all of their leading bits equal to 1.
"""
function extract_fourier_transform_values(Mk::MPS, m::Integer)
    sites = siteinds(Mk)
    n = length(Mk)
    function low_bits(leading_bit)
        L = ITensor(1.)
        for j=1:n-m
            L *= Mk[j]*onehot(sites[j]=>leading_bit)
        end
        T = L*prod([Mk[i] for i=n-m+1:n])
        A = Array(T,reverse(sites[n-m+1:n])...)
        return reshape(A,2^m)
    end
    js = -2^m:2^m-1
    vals = vcat(low_bits(2),low_bits(1))
    # Convert to k = -2πj and f̂(k), reversing so that k is increasing
    ks = reverse(-2π*js)
    vals = reverse(vals)/√(2.0^n)
    return ks, vals
end

function main(;
              n = 16,           # number of bits used to encode function
              a = 100,          # frequency you can adjust
              W = 1E-2,         # width of peak
              log_npoints = 10, # number of grid points for plotting f(x)
              log_nfreqs = 5    # plot 2*2^log_nfreqs values of k, centered on k=0
             )

  # Function to be loaded
  f(x) = exp(-(x-0.5)^2/W)*cos(a*x)
  println("Loading function: f(x) = exp(-(x-0.5)^2/$W)*cos($a*x)")

  println("Performing tensor cross interpolation:")
  M, info = tensor_cross(siteinds("Qubit",n),
                        (bits...)->f(b2c(bits...));
                         nsweep=5,
                         cutoff=1E-10,
                         outputlevel=1)
  println("Max rank of f(x): χ=$(maxlinkdim(M))")

  #
  # Fourier transform by applying the QFT as an MPO of low rank,
  # constructed following Chen and Lindsey, arXiv:2404.03182.
  # Computes  1/√N ∑ₓ f(x) exp(-2πijx)  for all integers j=0,1,...,N-1
  # and all N=2^n grid points x at once
  #
  println("\nPerforming quantum Fourier transform (QFT):")
  fourier_transform(M; cutoff=1E-12) # run once first to exclude compilation time
  Mk, qft_time = @timed fourier_transform(M; cutoff=1E-12)
  println("Max rank of f̂(k): χ=$(maxlinkdim(Mk))")

  #
  # For comparison, do a fast Fourier transform (FFT)
  # of the vector of all N=2^n values of f(x)
  #
  println("\nPerforming fast Fourier transform (FFT):")
  N = 2^n
  fvec = [f(j/N) for j=0:N-1]
  fft(fvec) # run once first to exclude compilation time
  fft_values, fft_time = @timed fft(fvec)

  println()
  @printf("QFT took %.3E seconds\n",qft_time)
  @printf("FFT took %.3E seconds\n",fft_time)

  #
  # Plot f(x) and f̂(k)
  #
  fx = extract_function_values(M,log_npoints)
  xs = range(0, 1-1/2^log_npoints, length=2^log_npoints)

  # Extract values of f̂(k) = ∫f(x) exp(ikx) dx for the wavevectors k of smallest |k|
  ks, fk = extract_fourier_transform_values(Mk,log_nfreqs)

  # The FFT computes ∑ₓ f(x) exp(-2πijx) for j=0,1,...,N-1 which is N*f̂(k) for k = -2πj.
  # Negative integers -j are stored at N-j.
  fk_fft = [fft_values[1+mod(round(Int,-k/2π),N)] for k in ks]/N

  #
  # Exact result: Gaussians of width ∼1/√W centered at k = ±a
  # NOTE:
  # (Only correct for the f(x) defined above: update or remove if you change f(x))
  #
  exact(k) = √(π*W)/2*(exp(-W*(k-a)^2/4) + exp(-W*(k+a)^2/4))
  kc = range(first(ks), last(ks), length=1000)

  #
  # Plot results
  #
  plt_x = plot(xs, fx; xlabel="x", ylabel="f(x)", label="", color=:blue)
  plt_k = plot(kc, exact.(kc); xlabel="k", ylabel="|f̂(k)|", label="Exact", color=:black)
  plot!(plt_k, ks, abs.(fk_fft); seriestype=:scatter, label="FFT", markercolor=:white,
        markerstrokecolor=:blue, markersize=6)
  plot!(plt_k, ks, abs.(fk); seriestype=:scatter, label="QFT", color=:red,
        markersize=3, markerstrokewidth=0)

  display(plot(plt_x, plt_k; layout=(2,1), size=(800,700)))

  return (; M, Mk)
end
