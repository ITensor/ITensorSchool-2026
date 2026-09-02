# Load the Plots library for plotting results
using Plots: Plots, plot

"""
    approx_pi(nterm::Int)

Approximate `pi` using the
[Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80)
using `nterm` terms in the series.
"""
function approx_pi(nterm::Int)
    pi_by_4 = 1.0
    for k in 1:nterm
        pi_by_4 += (-1)^k / (2k + 1)
    end
    return 4 * pi_by_4
end

"""
    main(; kwargs...)

Approximate `pi` using the
[Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80) and compute the
error.

# Keywords
- `nterm::Int = 10^5`: The number of terms to use in the approximation of `pi`.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `pi_approx::Float64`: The approximate value of `pi`.
- `error::Float64`: The absolute error in the approximation of `pi`.
- `nterm::Int`: Same as above.
"""
function main(; nterm::Int = 10^5, outputlevel::Int = 1)
    # Print the number of terms to use to approximate `pi`.
    if outputlevel > 0
        println("Number of terms: ", nterm)
    end

    # Approximate `pi` by calling the function defined above.
    pi_approx = approx_pi(nterm)
    if outputlevel > 0
        println("Approximate pi: ", pi_approx)
    end

    # Compute the error in the approximation.
    # `pi` is a built-in constant in Julia that represents
    # the value of pi to arbitrary precision.
    error = abs(pi_approx - pi)
    if outputlevel > 0
        println("Error: ", error)
    end

    # Return the approximation to `pi`, the absolute error,
    # and the number of terms used in the approximation.
    return (; pi_approx, error, nterm)
end
