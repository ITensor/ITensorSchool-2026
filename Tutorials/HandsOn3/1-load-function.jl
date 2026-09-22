using ITensors
using ITensorMPS
using LinearAlgebra: norm
using Plots: plot, grid

include("resources/tensor_cross/tensor_cross.jl")
include("resources/qtt_utils.jl")

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

function main(;
              n = 32,           # number of bits used to encode function
              a = 100,          # frequency you can adjust
              W = 1E-2,         # width of peak
              log_npoints = 10, # number of grid points for plotting
              # Tensor cross parameters
              nsweep = 5,
              cutoff = 1E-10,
             )

  # Function to be loaded
  f(x) = exp(-(x-0.5)^2/W)*cos(a*x)
  println("Loading function: f(x) = exp(-(x-0.5)^2/$W)*cos($a*x)")
  @printf("W = %.3E\n",W)
  @printf("a = %.3E\n",a)

  println("Performing tensor cross interpolation:")
  M, info = tensor_cross(siteinds("Qubit",n),
                        (bits...)->f(b2c(bits...));
                         nsweep, 
                         cutoff, 
                         outputlevel=1)

  println()
  println("Interpolated function onto 2^$n = $(2^n) virtual grid points")
  println("Tensor cross performed $(info.function_calls) calls to the function")

  χ = maxlinkdim(M)
  println("Max rank χ=$χ")

  vals = extract_function_values(M,log_npoints)

  Δ = 2.0^(-log_npoints)
  extracted_grid_points = 0:Δ:(1-Δ)
  exact_vals = f.(extracted_grid_points)
  @printf("\nError on extracted (plotted) points = %.4E\n",norm(vals-exact_vals))

  display(plot_grid_function(extracted_grid_points, vals, exact_vals))

  res = (;)
  return res
end

function plot_grid_function(x, vals, exact_vals)
  plt_f = plot(x, vals; 
               marker=:circle, markersize=2, markercolor=:blue, markerstrokecolor=:blue,
               xlabel="x", ylabel="f(x)", label="")

  # Error versus the true function values. Use a small
  # floor value so that zero error can be shown on a log scale
  err = max.(abs.(vals-exact_vals), 1E-16)
  plt_err = plot(x, err; 
                 color=:red, yscale=:log10,
                 xlabel="x", ylabel="error", label="")

  return plot(plt_f, plt_err; layout=grid(2,1,heights=[2/3,1/3]), size=(700,650))
end
