using ITensors
using ITensorMPS
using Plots: plot

include("resources/tensor_cross/tensor_cross.jl")
include("resources/qtt_utils.jl")

function integrate(M::MPS)
    sites = siteinds(M) # array of site indices of M
    n = length(sites) # number of qubits / MPS indices
    I = ITensor(1.0) # scalar ITensor to accumulate integral product
    for j=1:n
        # 
        # (1) Fill in code below to compute and contract
        #     the diagram that integrates the QTT function
        #     on [0,1)
        # (2) Tip: to make an single-index ITensor with 
        #     index s and elements [a,b], use ITensor([a,b], s)
        #
        # I = I * ...
    end
    return scalar(I)
end


function main(; 
               n=32,   # number of bits encoding the function
               W=1E-2) # width W of the Cauchy distribution

  # Unnormalized Cauchy distribution
  # centered at 0.5 of width W
  f(x) = W/((x-0.5)^2 + W^2)

  @printf("\nWidth W = %.3E\n\n",W)

  println("Performing cross interpolation:")
  M,info = tensor_cross(siteinds("Qubit",n),
                        (bits...)->f(b2c(bits...));
                         nsweep=5, 
                         cutoff=1E-10, 
                         outputlevel=1)
  χ = maxlinkdim(M)
  println("Max rank χ=$χ")

  I = @timed integrate(M)
  println("\nIntegration took $(I.time) seconds")

  println()
  @printf("π = %.12f\n",π)
  @printf("I = %.12f\n",I.value)
  @printf("err = %.4E\n",abs(π-I.value))

  res = (; integral=I)
  return res
end
