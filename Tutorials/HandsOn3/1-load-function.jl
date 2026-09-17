using ITensorMPS: siteinds, maxlinkdim
using Plots: plot

include("resources/tensor_cross/tensor_cross.jl")
include("resources/integrate.jl")
include("resources/qtt_utils.jl")

"""
    extract_qtt_values(M::MPS, n::Integer)

Extract N=2^n values of a function encoded as an
MPS `M` in the QTT format.
Returns a vector of N=2^n values.
"""
function extract_function_values(M::MPS, n::Integer)
    Npoints = 2^n
    sites = siteinds(M)
    L = length(M)
    if n > L-1
        error("MPS has L=$L indices, maximum n is $(L-1)")
    end
    R = ITensor(1.)
    for j=reverse(n+1:L)
        R *= M[j]*ITensor([1,0],sites[j])
    end
    T = prod([M[i] for i=1:n])*R
    A = Array(T,reverse(sites[1:n])...)
    vals = reshape(A,2^n)
    return vals
end

function main(;
              n = 32,          # number of bits used to encode function
              a = 100,         # frequency you can adjust
              W = 1E-2,        # width of peak
              log_npoints = 10 # number of grid points for plotting
             )

  # Function to be loaded
  f(x) = exp(-(x-0.5)^2/W)*cos(a*x)
  println("Loading function: f(x) = exp(-(x-0.5)^2/$W)*cos($a*x)")
  @printf("W = %.3E\n",W)
  @printf("a = %.3E\n",a)

  println("Performing tensor cross interpolation:")
  M, info = tensor_cross(siteinds("Qubit",n),
                        (bits...)->f(b2c(bits...));
                         nsweep=5, 
                         cutoff=1E-10, 
                         outputlevel=1)

  println()
  println("Interpolated function onto 2^$n = $(2^n) virtual grid points")
  println("Tensor cross performed $(info.function_calls) calls to the function")

  χ = maxlinkdim(M)
  println("Max rank χ=$χ")

  vals = extract_function_values(M,log_npoints)
  display(plot_grid_function(vals))

  res = (;)
  return res
end

function plot_grid_function(vals)
  return plot(vals; marker=:circle, markersize=2, markercolor=:blue, markerstrokecolor=:blue)
end
