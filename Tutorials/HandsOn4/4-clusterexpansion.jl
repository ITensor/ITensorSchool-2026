using NamedGraphs: named_grid

# Load the Plots library for plotting results
using Plots: Plots, plot

include("ising_tensornetwork.jl")
include("belief_propagation.jl")

"""
    main(; kwargs...)

Create a periodic 5x5 square grid, build the Ising tensor network on it, and compute the
free energy density both with the belief propagation you completed in Tutorial 2 and with
the first order loop correction added.

# Keywords
- `beta::Number = 0.2`: The inverse temperature parameter.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `phi_bp_tn::Number`: The free energy density from belief propagation.
- `phi_bp_corrected_tn::Number`: The same with the loop correction added.
- `phi_exact::Number`: The exact free energy density in the thermodynamic limit, from Onsager's solution.
- `niters`: The number of belief propagation iterations taken, or `nothing` if it did not converge.
- `tn::Dict`: The Ising tensor network.
- `g::NamedGraph`: The graph of the tensor network.
- `messages::Dict`: The converged belief propagation messages.
"""
function main(; beta::Number = 0.2, outputlevel::Int = 1)
    g = named_grid((5, 5); periodic = true)

    tn = ising_tensornetwork(g, beta)
    messages, niters = belief_propagation(tn, g; niters = 1000, outputlevel)

    phi_bp_tn = phi_bp(tn, g, messages)
    smallest_loop_size = 4
    phi_bp_corrected_tn = phi_bp_tn + phi_cluster_correction(tn, g, messages; smallest_loop_size)
    phi_exact = ising_phi(beta)

    return (; phi_bp_tn, phi_bp_corrected_tn, phi_exact, tn, g, messages, niters)
end
