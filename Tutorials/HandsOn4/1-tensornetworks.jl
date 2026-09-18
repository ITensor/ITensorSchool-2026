using Graphs: add_edge!
using NamedGraphs: NamedGraph, NamedEdge
using NamedGraphs: named_path_graph

include("ising_tensornetwork.jl")
include("contract_network.jl")

"""
    main(; kwargs...)

Create a simple path graph, constructs the Ising tensor network on it, and computes the
partition function Z by contracting the tensor network.

# Keywords
- `beta::Number = 0.2`: The inverse temperature parameter.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `tn::Dict{Any, ITensor}`: The Ising tensor network, one tensor per vertex of `g`.
- `g::NamedGraph`: The created graph.
- `z::Number`: The partition function, from contracting the network exactly.
- `beta::Number`: The inverse temperature, as passed in.
"""
function main(; beta::Number = 0.2, outputlevel::Int = 1)
    # Create a simple graph
    g = NamedGraph([1, 2, 3])
    es = [1 => 2, 2 => 3]
    for e in es
        add_edge!(g, e)
    end

    # Construct the tensor network of the classical Ising partition function on the graph
    tn = ising_tensornetwork(g, beta)

    # Contract the tensor network to compute the partition function Z
    z = contract_network(tn, g)[]

    return (; tn, g, z, beta)
end
