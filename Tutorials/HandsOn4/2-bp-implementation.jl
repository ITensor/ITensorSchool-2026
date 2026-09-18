using Graphs: is_tree, nv
using NamedGraphs: named_path_graph

include("ising_tensornetwork.jl")
include("contract_network.jl")
include("belief_propagation.jl")

"""
    main(; kwargs...)

Build the Ising tensor network on an open path graph of `L` vertices, run your belief
propagation from `belief_propagation.jl` on it, and compare the resulting free energy
density `ϕ = log(Z) / L` to the value from contracting the network exactly.

A path graph is a tree, and on a tree belief propagation is exact, so the two agree once
your implementation is complete. The pass/fail check is only performed on a tree. On a
graph with loops belief propagation is approximate, and the script just prints both numbers.

# Keywords
- `L::Int = 6`: The number of vertices in the path graph.
- `beta::Number = 0.3`: The inverse temperature parameter.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `phi_bp_tn::Number`: The free energy density from belief propagation.
- `phi_contracted_tn::Number`: The free energy density from exact contraction.
- `tn::Dict`: The Ising tensor network.
- `g::NamedGraph`: The graph of the tensor network.
- `messages::Dict`: The belief propagation messages returned by your implementation.
- `niters`: The number of iterations taken for convergence, or `nothing` if BP did not converge.
"""
function main(; L::Int = 6, beta::Number = 0.3, outputlevel::Int = 1)
    g = named_path_graph(L)
    tn = ising_tensornetwork(g, beta)

    # Run your belief propagation implementation (top of `belief_propagation.jl`)
    messages, niters = belief_propagation(tn, g; niters = 100, outputlevel)
    phi_bp_tn = phi_bp(tn, g, messages)

    # Exact contraction of the network to check against
    phi_contracted_tn = log(contract_network(tn, g)[]) / nv(g)

    if outputlevel > 0
        println("BP free energy density:    ", phi_bp_tn)
        println("Exact free energy density: ", phi_contracted_tn)
        if !is_tree(g)
            println("Graph has loops: BP is only approximate here, so no agreement check is performed")
        elseif abs(phi_bp_tn - phi_contracted_tn) < 1.0e-8
            println("BP free energy density AGREES with exact contraction (as it should on a tree)")
        else
            println("BP free energy density DOES NOT agree with exact contraction (it should on a tree)")
        end
    end

    return (; phi_bp_tn, phi_contracted_tn, tn, g, messages, niters)
end
