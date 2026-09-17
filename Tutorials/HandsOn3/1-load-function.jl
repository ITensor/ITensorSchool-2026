using ITensorMPS: siteinds, maxlinkdim

include("resources/tensor_cross/tensor_cross.jl")
include("resources/binary_fractions.jl")
include("resources/integrate.jl")

function main()
  n = 32
  c = 1E-5 # Good values to take are between 1E-4 to 1E-9

  # Unnormalized Cauchy distribution centered at 0.5
  f(x) = c/((x-0.5)^2 + c^2)

  @printf("\nc = %.3E\n\n",c)

  println("Performing cross interpolation:")
  M,info = tensor_cross(siteinds("Qubit",n),(xs...)->f(b2c(xs...));
                             nsweep=5, initial_pivot=c2b(1/2,n), cutoff=1E-10, outputlevel=1)
  χ = maxlinkdim(M)
  println("Max rank χ=$χ")

  I = @timed integrate(M)
  println("\nIntegration took $(I.time) seconds")

  println()
  @printf("π = %.12f\n",π)
  @printf("I = %.12f\n",I.value)
  @printf("err = %.4E\n",abs(π-I.value))

  res = (;)
  return res
end
