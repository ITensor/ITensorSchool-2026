using Graphs: add_edge!
using NamedGraphs: NamedGraph, NamedEdge
using NamedGraphs: named_path_graph

include("../../Tutorials/HandsOn4/ising_tensornetwork.jl")
include("../../Tutorials/HandsOn4/contract_network.jl")

"""
    main(; kwargs...)

Create a path graph on `L` vertices (optionally periodic), constructs the Ising tensor
network on it, and computes the partition function Z by contracting the tensor network.

This is the completed solution to exercise 1 of Tutorial 1.

# Keywords
- `L::Int = 3`: The number of vertices in the path graph.
- `periodic::Bool = false`: Whether to add an edge between vertex `L` and vertex `1`.
- `beta::Number = 0.2`: The inverse temperature parameter.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `tn::Dict{Any, ITensor}`: The Ising tensor network, one tensor per vertex of `g`.
- `g::NamedGraph`: The created graph.
- `z::Number`: The partition function, from contracting the network exactly.
- `beta::Number`: The inverse temperature, as passed in.
"""
function main(; L::Int = 3, periodic::Bool = false, beta::Number = 0.2, outputlevel::Int = 1)
    # Create a path graph on L vertices
    g = NamedGraph(collect(1:L))
    es = [i => i + 1 for i in 1:(L - 1)]
    if periodic
        push!(es, L => 1)
    end
    for e in es
        add_edge!(g, e)
    end

    # Construct the tensor network of the classical Ising partition function on the graph
    tn = ising_tensornetwork(g, beta)

    # Contract the tensor network to compute the partition function Z
    z = contract_network(tn, g)[]

    if outputlevel > 0
        z_exact = periodic ? cosh(beta)^L + sinh(beta)^L : 2 * cosh(beta)^(L - 1)
        println("Tensor network Z: ", z)
        println("Exact Z:          ", z_exact)
    end

    return (; tn, g, z, beta)
end
