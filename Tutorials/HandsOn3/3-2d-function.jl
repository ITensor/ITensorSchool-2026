using ITensors
using ITensorMPS
using Random: Random
using Plots: contourf, plot!, scatter!, RGB, mm

include("resources/tensor_cross/tensor_cross.jl")
include("resources/qtt_utils.jl")

function main(; nbits = 16,  # number of bits per dimension for QTT encoding
                # Parameters controlling f(x,y)
                Ngaussian = 50, # number of Gaussians to sum to make f(x,y)
                width = 0.01,  # typical width of each Gaussian
                Nsamples = 1000, 
                # Tensor cross related parameters
                maxdim = 60,
                cutoff = 1E-5,
                nsweep = 5,
                random_seed=1)

  Random.seed!(random_seed)

  #
  # Define f(x,y) to be a
  # sum of Ngaussian random Gaussians
  # of typical width W
  #
  ws = [width*rand() for g=1:Ngaussian]
  xcs = [rand() for g=1:Ngaussian]
  ycs = [rand() for g=1:Ngaussian]
  signs = [rand((+1,-1)) for g=1:Ngaussian]
  heights = [rand() for g=1:Ngaussian]
  function f(x,y)
    val = 0.0
    for g=1:Ngaussian
      val += heights[g]*signs[g]*exp(-((x-xcs[g])^2+(y-ycs[g])^2)/ws[g])
    end
    return val
  end

  #
  # Call tensor_cross to load the function
  #
  println("Performing tensor cross interpolation:")
  M, info = tensor_cross(siteinds("Qubit",2*nbits),(i...)->f(b2c(i[1:nbits]...),b2c(i[nbits+1:end]...));
                               nsweep, maxdim, cutoff, outputlevel=1)
  println("Bond dimensions χ's = \n",linkdims(M))

  #
  # Sample the MPS. Each sample is a vector of bits like [1,2,2,1,...]
  # drawn with probability |M(bits)|^2. The first nbits are the
  # bits of x and the remaining nbits are the bits of y.
  #
  Ms = normalize(orthogonalize(M,1))
  sample_x = Float64[]
  sample_y = Float64[]
  for ns in 1:Nsamples
    s = sample(Ms)
    push!(sample_x, b2c(s[1:nbits]))
    push!(sample_y, b2c(s[nbits+1:end]))
  end

  # 
  # Plot
  #

  # Do a brute-force search for minimum and maximum
  (; x, y, z, max_x_x, max_y_x, min_x_x, min_y_x) = brute_force_search(f; nn = 10)

  plt = contourf(x, y, z'; levels=20, color=:lajolla, linewidth=0,
                 aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), size=(900,850),
                 top_margin=10mm, legend=:outertop, legend_columns=3,
                 legendfontsize=12, foreground_color_legend=nothing)

  #
  # Show location of each sample as a small blue point
  #
  bright_blue = RGB(0, 150/255, 1)
  scatter!(plt, sample_x, sample_y; color=bright_blue, markersize=2.5,
           markerstrokewidth=0, label="Samples")

  #
  # Mark global max (green) and min (red) found by brute-force search
  #
  bright_green = RGB(1/255, 220/255, 0)
  bright_red = RGB(240/255, 0, 0)
  Δ = 0.05
  linewidth = 3.0
  plot!(plt, [max_x_x-Δ,max_x_x+Δ], [max_y_x,max_y_x]; color=bright_green, linewidth, label="Global maximum")
  plot!(plt, [max_x_x,max_x_x], [max_y_x-Δ,max_y_x+Δ]; color=bright_green, linewidth, label="")

  plot!(plt, [min_x_x-Δ,min_x_x+Δ], [min_y_x,min_y_x]; color=bright_red, linewidth, label="Global minimum")
  plot!(plt, [min_x_x,min_x_x], [min_y_x-Δ,min_y_x+Δ]; color=bright_red, linewidth, label="")

  display(plt)

  return (;)
end

#
# Evaluate f on a 2^nn × 2^nn grid, recording the
# location and value of its minimum and maximum.
# Returns the grid points x,y and values z[i,j] = f(x[i],y[j])
#
function brute_force_search(f; nn = 10)
  x = range(0, 1-1/2^nn, length=2^nn)
  y = range(0, 1-1/2^nn, length=2^nn)
  z = zeros(2^nn,2^nn)
  max_f_x, max_x_x, max_y_x = -Inf,-Inf,-Inf
  min_f_x, min_x_x, min_y_x = +Inf,+Inf,+Inf
  for (i,x_) in enumerate(x), (j,y_) in enumerate(y)
    val = f(x_,y_)
    if val > max_f_x
      max_f_x, max_x_x, max_y_x = val,x_,y_
    end
    if val < min_f_x
      min_f_x, min_x_x, min_y_x = val,x_,y_
    end
    z[i,j] = val
  end
  return (; x, y, z, max_f_x, max_x_x, max_y_x, min_f_x, min_x_x, min_y_x)
end

