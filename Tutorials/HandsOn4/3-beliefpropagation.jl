using NamedGraphs: named_grid

# Load the Plots library for plotting results
using Plots: Plots, plot

include("ising_tensornetwork.jl")
include("belief_propagation.jl")

"""
    main(; kwargs...)

Create an `Lx` by `Ly` square grid, build the Ising tensor network on it, and compute the
free energy density with the belief propagation you completed in Tutorial 2.

# Keywords
- `Lx::Int = 3`: The number of columns in the grid.
- `Ly::Int = 3`: The number of rows in the grid.
- `beta::Number = 0.2`: The inverse temperature parameter.
- `periodic::Bool = false`: Whether to use periodic boundary conditions.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `phi_bp_tn::Number`: The free energy density from belief propagation.
- `phi_exact::Number`: The exact free energy density in the thermodynamic limit, from Onsager's solution.
- `tn::Dict`: The Ising tensor network.
- `g::NamedGraph`: The graph of the tensor network.
- `messages::Dict`: The converged belief propagation messages.
- `niters`: The number of belief propagation iterations taken, or `nothing` if it did not converge.
"""
function main(; Lx::Int = 3, Ly::Int = 3, beta::Number = 0.2, periodic = false, outputlevel::Int = 1)
    g = named_grid((Lx, Ly); periodic)

    tn = ising_tensornetwork(g, beta)
    messages, niters = belief_propagation(tn, g; niters = 1000, outputlevel)

    phi_bp_tn = phi_bp(tn, g, messages)
    phi_exact = ising_phi(beta)
    return (; phi_bp_tn, phi_exact, tn, g, messages, niters)
end
