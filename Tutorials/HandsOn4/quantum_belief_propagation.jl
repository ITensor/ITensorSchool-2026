using Graphs: is_tree, nv, vertices
using NamedGraphs: NamedGraph, named_path_graph, all_edges, boundary_edges
using ITensors: ITensor, dag, delta, inds, normalize, prime
using LinearAlgebra: dot
using Statistics: mean

include("contract_network.jl")
include("aklt_tensornetwork.jl")

"""
Stretch goal: quantum belief propagation.

You are asked to run belief propagation on the norm network `⟨ψ|ψ⟩` of a quantum state and
use it to compute expectation values. For each numbered step (1), (2), (3) below, fill in
the missing code.

The norm network has *two* tensors per vertex, the ket `ψ_v` and the bra `conj(ψ_v)`, joined
over the physical index. Each edge therefore carries two indices, the ket leg `l` and the bra
leg `l'`, so a message here is a matrix rather than a vector.

The whole point of this exercise is to never multiply the ket and the bra at a vertex
together. That product is a tensor with `2z` virtual legs and costs `χ^(2z)` to store, which
dwarfs everything else in the calculation. Instead, absorb the incoming messages into the
ket one at a time and only then close with the bra, which costs about `χ^(z+1)`.

Unlike the classical free energy, expectation values are ratios of two contractions that
share the same messages, so the normalization of the messages cancels and there is no need
for anything like `binormalized_messages` here.
"""

"""
    quantum_initial_messages(state)

Initial messages: the identity matrix on the ket and bra legs of each edge.
"""
function quantum_initial_messages(state)
    return Dict(
        e => normalize(delta(state.links[e], prime(state.links[e])))
            for e in all_edges(state.g)
    )
end

"""
    quantum_updated_message(state, messages, e)

The new message along the directed edge `e`, from `src(e)` to `dst(e)`.
"""
function quantum_updated_message(state, messages, e)
    # The directed edges pointing into `src(e)`, excluding the one coming from `dst(e)`
    incoming_es = setdiff(boundary_edges(state.g, [src(e)]; dir = :in), [reverse(e)])

    # (1) Starting from `ket_tensor(state, src(e))`, multiply in the message on each edge of
    #     `incoming_es` one at a time, then close with `bra_tensor(state, src(e))` and
    #     normalize. Do NOT contract the ket with the bra first: that is the expensive
    #     double layer tensor this exercise exists to avoid.
    # ...

    return messages[e]
end

function quantum_update_messages(state, messages)
    updated_messages = copy(messages)
    for e in all_edges(state.g)
        updated_messages[e] = quantum_updated_message(state, messages, e)
    end
    return updated_messages
end

function quantum_message_distance(state, messages, old_messages)
    return mean([1 - dot(messages[e], old_messages[e])^2 for e in all_edges(state.g)])
end

"""
    quantum_belief_propagation(state[, messages]; kwargs...)

Run belief propagation on the norm network of `state`.

# Keywords
- `niters::Int = 2000`: The maximum number of iterations to perform.
- `tol::Float64 = 1e-14`: The tolerance for convergence. This is tighter than the default
  used for the Ising model because the error in an expectation value goes roughly like the
  square root of this measure, so a loose tolerance costs you several digits.
- `outputlevel::Int = 1`: The verbosity level of the output.

# Returns
- `messages::Dict`: The converged messages.
- `niters`: The number of iterations taken, or `nothing` if it did not converge.
"""
function quantum_belief_propagation(
        state, messages = quantum_initial_messages(state);
        niters::Int = 2000, tol::Float64 = 1.0e-14, outputlevel::Int = 1
    )
    for i in 1:niters
        old_messages = messages
        messages = quantum_update_messages(state, messages)
        if quantum_message_distance(state, messages, old_messages) < tol
            outputlevel >= 1 && println("Quantum BP converged after $i iterations")
            return messages, i
        end
    end
    outputlevel >= 1 && println("Quantum BP did NOT converge after $niters iterations")
    return messages, nothing
end

"""
    quantum_expect(state, messages, v, O::ITensor)

The expectation value `⟨ψ|O_v|ψ⟩ / ⟨ψ|ψ⟩` of the operator `O` on vertex `v`.
"""
function quantum_expect(state, messages, v, O::ITensor)
    incoming = [messages[e] for e in boundary_edges(state.g, [v]; dir = :in)]

    # (2) Build the numerator and the denominator. Both start from a ket layer, with the
    #     operator applied for the numerator via `ket_tensor(state, v, O)`, absorb every
    #     message in `incoming`, and then close with `bra_tensor(state, v)`. Each contraction
    #     gives a scalar `ITensor`, so use `[]` to get a number, and return their ratio.
    # ...

    return 0.0
end

"""
    quantum_expect_bond(state, messages, v, w, Ov::ITensor, Ow::ITensor)

The expectation value of `Ov` on vertex `v` times `Ow` on the neighbouring vertex `w`.
"""
function quantum_expect_bond(state, messages, v, w, Ov::ITensor, Ow::ITensor)
    incoming = [messages[e] for e in boundary_edges(state.g, [v, w]; dir = :in)]

    # (3) The same idea for a cluster of two neighbouring vertices. Absorb each message in
    #     `incoming` into whichever of the two ket layers shares an index with it, contract
    #     the two kets together over the bond they share, and then close with both bras.
    #     Again form the numerator and the denominator and return their ratio.
    # ...

    return 0.0
end

"""
    quantum_expect_exact(state, v, O::ITensor)

The same single vertex expectation value, obtained by contracting the whole norm network
exactly. Only usable on small graphs, and provided here so that you have something to check
your belief propagation answers against.
"""
function quantum_expect_exact(state, v, O::ITensor)
    vs = collect(vertices(state.g))
    bras = [bra_tensor(state, u) for u in vs]
    denominator = contract_network([[ket_tensor(state, u) for u in vs]; bras])[]
    numerator = contract_network(
        [[ket_tensor(state, u, u == v ? O : nothing) for u in vs]; bras]
    )[]
    return numerator / denominator
end

"""
    main(; kwargs...)

Build the AKLT state on a graph, run your quantum belief propagation on its norm network and
compare `⟨(Sᶻ)²⟩` on one vertex to exact contraction of the same network.

The default graph is an open path, which is a tree, so belief propagation is exact there and
the two should agree once your implementation is complete. On a graph with loops belief
propagation is only approximate, so the check is skipped.

# Keywords
- `g`: The graph to build the AKLT state on. Defaults to an open path of six vertices.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `sz2_bp::Number`: `⟨(Sᶻ)²⟩` on one vertex, from belief propagation.
- `sz2_exact::Number`: The same quantity, from exact contraction.
- `state`: The AKLT state.
- `messages::Dict`: The messages returned by your implementation.
- `niters`: The number of iterations taken for convergence.
"""
function main(; g = named_path_graph(6), outputlevel::Int = 1)
    state = aklt_tensornetwork(g)
    messages, niters = quantum_belief_propagation(state; outputlevel)

    # a vertex in the bulk, where the physical spin is set by the coordination number
    v = argmax(u -> length(link_indices(state, u)), collect(vertices(g)))
    ops = spin_operators(state.sites[v])
    sz2 = apply(ops.sz, ops.sz)

    sz2_bp = quantum_expect(state, messages, v, sz2)
    sz2_exact = quantum_expect_exact(state, v, sz2)

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
