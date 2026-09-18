using Graphs: edges, src, dst, vertices
using NamedGraphs: named_grid, boundary_edges

include("belief_propagation.jl")
include("../../Tutorials/HandsOn4/aklt_tensornetwork.jl")

"""
    expect_bp(state, messages, v, O::ITensor)

The expectation value of the operator `O` on vertex `v`, using the converged belief
propagation `messages` on the norm network as the environment around `v`.

This is `phi_factor` with an operator inserted: the numerator contracts the vertex tensor
carrying `O` with all of its incoming messages, and the denominator does the same without
the operator.
"""
function expect_bp(state, messages, v, O::ITensor)
    g = state.g
    messages = binormalized_messages(g, messages)
    incoming = [messages[e] for e in boundary_edges(g, [v]; dir = :in)]
    numerator = contract_network([[operator_tensor(state, v, O)]; incoming])[]
    denominator = contract_network([[state.tn[v]]; incoming])[]
    return numerator / denominator
end

"""
    expect_bond_bp(state, messages, v, w, Ov::ITensor, Ow::ITensor)

The expectation value of `Ov` on vertex `v` times `Ow` on vertex `w`, where `v` and `w` are
neighbours, using belief propagation messages on the boundary of the pair.

This is the same idea as `phi_cluster_correction`, which contracts a whole loop of tensors
with the messages incident to it, applied to a cluster of just two vertices.
"""
function expect_bond_bp(state, messages, v, w, Ov::ITensor, Ow::ITensor)
    g = state.g
    messages = binormalized_messages(g, messages)
    incoming = [messages[e] for e in boundary_edges(g, [v, w]; dir = :in)]
    numerator = contract_network(
        [[operator_tensor(state, v, Ov), operator_tensor(state, w, Ow)]; incoming]
    )[]
    denominator = contract_network([[state.tn[v], state.tn[w]]; incoming])[]
    return numerator / denominator
end

"""
    expect_bond_exact(state, v, w, Ov::ITensor, Ow::ITensor)

The same bond expectation value, obtained by contracting the whole finite network exactly
rather than using belief propagation.
"""
function expect_bond_exact(state, v, w, Ov::ITensor, Ow::ITensor)
    tn_numerator = copy(state.tn)
    tn_numerator[v] = operator_tensor(state, v, Ov)
    tn_numerator[w] = operator_tensor(state, w, Ow)
    return contract_network(tn_numerator, state.g)[] / contract_network(state.tn, state.g)[]
end

"""
    heisenberg_bond(state, messages, v, w; exact = false)

The Heisenberg bond energy `⟨S_v · S_w⟩`, assembled from `SᶻSᶻ` and the two flip-flop terms.
Set `exact = true` to contract the finite network exactly instead of using belief propagation.
"""
function heisenberg_bond(state, messages, v, w; exact = false)
    ops_v = spin_operators(state.sites[v])
    ops_w = spin_operators(state.sites[w])
    f = if exact
        (Ov, Ow) -> expect_bond_exact(state, v, w, Ov, Ow)
    else
        (Ov, Ow) -> expect_bond_bp(state, messages, v, w, Ov, Ow)
    end
    return f(ops_v.sz, ops_w.sz) +
        0.5 * (f(ops_v.sp, ops_w.sm) + f(ops_v.sm, ops_w.sp))
end

"""
    main(; kwargs...)

Build the AKLT state on a graph, run belief propagation on its norm network, and compare
local expectation values to exact contraction of the same finite network.

On a periodic ring every vertex has degree two, so this is the spin-1 AKLT chain, for which
the nearest neighbour correlations are known exactly in the thermodynamic limit:

    ⟨Sᶻ⟩ = 0,   ⟨(Sᶻ)²⟩ = 2/3,   ⟨SᶻSᶻ⟩ = -4/9,   ⟨S·S⟩ = -4/3.

# Keywords
- `g`: The graph to build the AKLT state on. Defaults to a periodic ring of six vertices.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `sz::Number`: `⟨Sᶻ⟩` on one vertex, from belief propagation.
- `sz2::Number`: `⟨(Sᶻ)²⟩` on one vertex, from belief propagation.
- `ss_bp::Number`: `⟨S·S⟩` on one bond, from belief propagation.
- `ss_exact::Number`: `⟨S·S⟩` on the same bond, from exact contraction.
- `state`: The AKLT state and its norm network.
- `messages::Dict`: The converged belief propagation messages.
- `niters`: The number of iterations taken for convergence.
"""
function main(; g = named_grid((6, 1); periodic = true), outputlevel::Int = 1)
    state = aklt_tensornetwork(g)
    messages, niters = belief_propagation(state.tn, g; niters = 500, outputlevel)

    v = first(vertices(g))
    ops = spin_operators(state.sites[v])
    sz = expect_bp(state, messages, v, ops.sz)
    sz2 = expect_bp(state, messages, v, apply(ops.sz, ops.sz))

    e = first(edges(g))
    ss_bp = heisenberg_bond(state, messages, src(e), dst(e))
    ss_exact = heisenberg_bond(state, messages, src(e), dst(e); exact = true)

    if outputlevel > 0
        println("Physical spin S on each vertex: ", (dim(state.sites[v]) - 1) / 2)
        println("⟨Sᶻ⟩       BP    = ", sz)
        println("⟨(Sᶻ)²⟩    BP    = ", sz2)
        println("⟨S·S⟩      BP    = ", ss_bp)
        println("⟨S·S⟩      exact = ", ss_exact, "  (this finite network)")
    end

    return (; sz, sz2, ss_bp, ss_exact, state, messages, niters)
end
