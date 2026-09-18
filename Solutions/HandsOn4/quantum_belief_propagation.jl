using Graphs: is_tree, nv, vertices
using NamedGraphs: NamedGraph, named_path_graph, all_edges, boundary_edges
using ITensors: ITensor, dag, delta, inds, normalize, prime
using LinearAlgebra: dot
using Statistics: mean

include("../../Tutorials/HandsOn4/contract_network.jl")
include("../../Tutorials/HandsOn4/aklt_tensornetwork.jl")

"""
Stretch goal: quantum belief propagation. This is the completed solution.

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

    # (1) Absorb the incoming messages into the ket layer one at a time, and only then
    #     close with the bra. The ket never meets the bra until every message is already in,
    #     so the expensive double layer tensor is never formed.
    t = ket_tensor(state, src(e))
    for e_in in incoming_es
        t = t * messages[e_in]
    end
    t = t * bra_tensor(state, src(e))
    return normalize(t)
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

    # (2) Numerator and denominator differ only by the operator on the ket layer.
    numerator = ket_tensor(state, v, O)
    denominator = ket_tensor(state, v)
    for m in incoming
        numerator = numerator * m
        denominator = denominator * m
    end
    bra = bra_tensor(state, v)
    return (numerator * bra)[] / (denominator * bra)[]
end

"""
    quantum_expect_bond(state, messages, v, w, Ov::ITensor, Ow::ITensor)

The expectation value of `Ov` on vertex `v` times `Ow` on the neighbouring vertex `w`.
"""
function quantum_expect_bond(state, messages, v, w, Ov, Ow)
    incoming = [messages[e] for e in boundary_edges(state.g, [v, w]; dir = :in)]

    # (3) The same idea on a cluster of two vertices. Each boundary message belongs to
    #     whichever of the two kets carries its index.
    function value(operator_v, operator_w)
        kv = ket_tensor(state, v, operator_v)
        kw = ket_tensor(state, w, operator_w)
        for m in incoming
            if isempty(intersect(inds(m), inds(kw)))
                kv = kv * m
            else
                kw = kw * m
            end
        end
        # contract the two kets over the bond they share, then close with both bras
        return (kv * kw * bra_tensor(state, v) * bra_tensor(state, w))[]
    end
    return value(Ov, Ow) / value(nothing, nothing)
end

"""
    heisenberg_terms(state, v, w)

`S_v · S_w` written as a list of `(O_v, O_w, coefficient)` triples, using

    S_v · S_w = SᶻSᶻ + (S⁺S⁻ + S⁻S⁺) / 2.

Every term is a product of one operator on `v` and one on `w`, which is the form
`quantum_expect_bond` takes.
"""
function heisenberg_terms(state, v, w)
    ops_v = spin_operators(state.sites[v])
    ops_w = spin_operators(state.sites[w])
    return [
        (ops_v.sz, ops_w.sz, 1.0),
        (ops_v.sp, ops_w.sm, 0.5),
        (ops_v.sm, ops_w.sp, 0.5),
    ]
end

"""
    heisenberg_bond(state, messages, v, w)

The Heisenberg bond energy `⟨S_v · S_w⟩`.
"""
function heisenberg_bond(state, messages, v, w)
    return sum(
        c * quantum_expect_bond(state, messages, v, w, Ov, Ow)
            for (Ov, Ow, c) in heisenberg_terms(state, v, w)
    )
end

"""
    heisenberg_bond_squared(state, messages, v, w)

The expectation value of `(S_v · S_w)²`.

Squaring the sum of three terms gives nine, each still a product of one operator on `v` and
one on `w`, because `(A ⊗ B)(A' ⊗ B') = AA' ⊗ BB'`.
"""
function heisenberg_bond_squared(state, messages, v, w)
    terms = heisenberg_terms(state, v, w)
    return sum(
        c1 * c2 * quantum_expect_bond(
            state, messages, v, w, apply(Ov1, Ov2), apply(Ow1, Ow2)
        )
            for (Ov1, Ow1, c1) in terms, (Ov2, Ow2, c2) in terms
    )
end

"""
    aklt_bond_energy(state, messages, v, w)

The energy of one bond of the spin-1 AKLT parent Hamiltonian, the projector onto total spin
2 on that bond,

    P₂ = 1/3 + (S_v · S_w) / 2 + (S_v · S_w)² / 6.

The AKLT state is annihilated by every such projector, so this is zero for the AKLT state.

This is the spin-1 projector, so it only applies where both vertices have degree two, such
as in the bulk of a chain or anywhere on a ring. On a lattice of higher coordination number
the parent Hamiltonian projects onto a different maximal spin.
"""
function aklt_bond_energy(state, messages, v, w)
    ss = heisenberg_bond(state, messages, v, w)
    ss2 = heisenberg_bond_squared(state, messages, v, w)
    return 1 / 3 + ss / 2 + ss2 / 6
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
