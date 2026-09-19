using Graphs: is_tree, vertices
using NamedGraphs: named_path_graph
using ITensors: apply, dim

include("aklt_tensornetwork.jl")
include("quantum_belief_propagation.jl")

"""
    main(; kwargs...)

Build the AKLT state on a graph, run your quantum belief propagation from
`quantum_belief_propagation.jl` on its norm network, and compare `⟨(Sᶻ)²⟩` on one vertex to
the value from contracting the network exactly.

The default graph is an open path, which is a tree, and on a tree belief propagation is
exact, so the two agree once your implementation is complete. The pass/fail check is only
performed on a tree. On a graph with loops belief propagation is approximate, and the script
just prints both numbers.

# Keywords
- `g`: The graph to build the AKLT state on. Defaults to an open path of six vertices.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `sz2_bp::Number`: `⟨(Sᶻ)²⟩` on one vertex, from belief propagation.
- `sz2_exact::Number`: The same quantity, from exact contraction.
- `state`: The AKLT state, as returned by `aklt_tensornetwork`.
- `messages::Dict`: The messages returned by your implementation.
- `niters`: The number of iterations taken for convergence.
"""
function main(; g = named_path_graph(6), outputlevel::Int = 1)
    state = aklt_tensornetwork(g)

    # Run your quantum belief propagation (in `quantum_belief_propagation.jl`)
    messages, niters = belief_propagation(state; outputlevel)

    # A vertex of maximal degree, so that we look at a bulk spin rather than a chain end
    v = argmax(u -> length(link_indices(state, u)), collect(vertices(g)))
    ops = spin_operators(state.sites[v])
    sz2 = apply(ops.sz, ops.sz)

    sz2_bp = expect(state, messages, v, sz2)
    sz2_exact = expect_exact(state, v, sz2)

    if outputlevel > 0
        println("Physical spin on vertex $v: S = ", (dim(state.sites[v]) - 1) / 2)
        println("⟨(Sᶻ)²⟩ BP    = ", sz2_bp)
        println("⟨(Sᶻ)²⟩ exact = ", sz2_exact)
        if !is_tree(g)
            println("Graph has loops: BP is only approximate here, so no agreement check is performed")
        elseif abs(sz2_bp - sz2_exact) < 1.0e-8
            println("Quantum BP AGREES with exact contraction (as it should on a tree)")
        else
            println("Quantum BP DOES NOT agree with exact contraction (it should on a tree)")
        end
    end

    return (; sz2_bp, sz2_exact, state, messages, niters)
end
