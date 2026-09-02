using NamedGraphs.NamedGraphGenerators: named_grid
using Statistics: mean

# Load the Plots library for plotting results
using Plots: Plots, plot

include("ising_tensornetwork.jl")
include("belief_propagation.jl")

"""
    main(; kwargs...)

Creates a grid graph of size `Lx` by `Ly`, constructs the Ising tensor network on it, and
computes the Bethe-Peierls free energy density using belief propagation.

# Keywords
- `Lx::Int = 3`: The number of columns in the grid.
- `Ly::Int = 3`: The number of rows in the grid.
- `beta::Number = 0.2`: The inverse temperature parameter.
- `periodic::Bool = false`: Whether to use periodic boundary conditions.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `phi_bp_tn::Number`: The Bethe-Peierls free energy density computed via belief propagation.
- `phi_exact::Number`: The exact free energy density from Onsager's solution in the thermodynamic limit.
- `tn::Dict`: The Ising tensor network.
- `g::NamedGraph`: The graph of the tensor network.
- `messages::Dict`: The converged belief propagation messages.
- `niters::Int`: The number of iterations taken for convergence in belief propagation.
"""
function main(; Lx::Int = 3, Ly::Int = 3, beta::Number = 0.2, periodic = false, outputlevel::Int = 1)
    g = named_grid((Lx, Ly); periodic)

    tn = ising_tensornetwork(g, beta)
    messages, niters = belief_propagation(tn, g; niters = 1000, outputlevel)

    phi_bp_tn = phi_bp(tn, g, messages)
    phi_exact = ising_phi(beta)
    return (; phi_bp_tn, phi_exact, tn, g, messages, niters)
end
