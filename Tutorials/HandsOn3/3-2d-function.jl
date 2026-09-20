using ITensorMPS: siteinds, linkdims, maxlinkdim
using Random: Random
using Plots: contourf, plot!, RGB, mm


include("resources/tensor_cross/tensor_cross.jl")
include("resources/qtt_utils.jl")

function main(; nbits = 16, random_seed=1)
  Random.seed!(random_seed)

  show_pivots = false


  #
  # Sum of random Gaussians
  #
  Ng = 50
  ω = 0.01
  ws = [ω*rand() for g=1:Ng]
  xcs = [rand() for g=1:Ng]
  ycs = [rand() for g=1:Ng]
  signs = [rand((+1,-1)) for g=1:Ng]
  heights = [rand() for g=1:Ng]
  function f(x,y)
    val = 0.0
    for g=1:Ng
      val += heights[g]*signs[g]*exp(-((x-xcs[g])^2+(y-ycs[g])^2)/ws[g])
    end
    return val
  end


  println("Performing tensor cross interpolation:")
  maxdim = 60
  cutoff = 1E-5
  nsweep = 5
  M,info = tensor_cross(siteinds("Qubit",2*nbits),(i...)->f(b2c(i[1:nbits]...),b2c(i[nbits+1:end]...));
                               nsweep, maxdim, cutoff, outputlevel=1)
  println("Bond dimensions χ's = \n",linkdims(M))

  #N = 2^(2n)
  #Nc = length(cache)
  #@printf("\n%d queries versus %d grid points. %.8f%% of grid points used.\n",Nc,N,100*Nc/N)

  #Nsample = 1000
  #avg_err = 0.0
  #Ms = copy(M)
  #orthogonalize!(Ms,1)
  #normalize!(Ms)
  #for ns in 1:Nsample
  #  sa = sample!(Ms)
  #  x,y = b2c(sa[1:n]),b2c(sa[n+1:end])
  #  avg_err += abs(f(x,y)-evaluate(M,sa))/Nsample
  #end
  #@printf("Average error (sampled) = %.4E\n",avg_err)

  # 
  # Plot
  #

  # Do a brute-force search for minimum and maximum
  (; x, y, z, max_x_x, max_y_x, min_x_x, min_y_x) = brute_force_search(f; nn = 10)

  plt = contourf(x, y, z'; levels=20, color=:lajolla, linewidth=0,
                 aspect_ratio=:equal, xlims=(0,1), ylims=(0,1), size=(900,850),
                 top_margin=10mm, legend=:outertop, legend_columns=2,
                 legendfontsize=12, foreground_color_legend=nothing)

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

  F = Figure(size = (1200, 900))
  #white = RGBAf(255, 255, 255)
  #green = RGBf(1/255, 113/255, 0)
  #blue = RGBf(0, 84/255, 155/255)
  #red = RGBf(174/255, 42/255, 22/255)
  #bright_green = RGBf(1/255, 220/255, 0)
  #bright_red = RGBf(240/255, 0, 0)
  #ax = Axis(F[1, 1]; xgridcolor=white, ygridcolor=white)
  #contourf!(ax,x,y,z; colormap=:lajolla)

  linewidth = 0.5

  if show_pivots
    px = Float64[]
    py = Float64[]
    for k in keys(cache)
      x = b2c(k[1:n])
      y = b2c(k[n+1:end])
      push!(px,x)
      push!(py,y)
    end
    scatter!(ax,px,py; color=white, markersize=2.0)
  end
  
  if find_optima
    Δ = 0.05
    linewidth = 2.0
    lines!(ax,[max_x-Δ,max_x+Δ],[max_y ,max_y]; color=bright_green,linewidth)
    lines!(ax,[max_x,max_x],[max_y-Δ,max_y+Δ]; color=bright_green,linewidth)

    lines!(ax,[min_x-Δ,min_x+Δ],[min_y ,min_y]; color=bright_red,linewidth)
    lines!(ax,[min_x,min_x],[min_y-Δ,min_y+Δ]; color=bright_red,linewidth)
  end

  display(F)

  return
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

