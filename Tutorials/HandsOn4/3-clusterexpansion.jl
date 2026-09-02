using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics: mean

# Load the Plots library for plotting results
using Plots: Plots, plot

include("ising_tensornetwork.jl")
include("belief_propagation.jl")

"""
    main(; kwargs...)  

Creates a periodic 5x5 grid graph, constructs the Ising tensor network on it, and computes
both the Bethe-Peierls free energy density using belief propagation and a loop-corrected
free energy density.

# Keywords
- `beta::Number = 0.2`: The inverse temperature parameter.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `phi_bp_tn::Number`: The Bethe-Peierls free energy density computed via belief propagation.
- `phi_bp_corrected_tn::Number`: The loop-corrected free energy density.
- `phi_exact::Number`: The exact free energy density from Onsager's solution in the thermodynamic limit.
- `niters::Int`: The number of iterations taken for convergence in belief propagation.
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
