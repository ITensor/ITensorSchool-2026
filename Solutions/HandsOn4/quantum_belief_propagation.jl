using Graphs: vertices
using NamedGraphs: all_edges, boundary_edges
using ITensors: ITensor, delta, inds, normalize, prime
using LinearAlgebra: dot
using Statistics: mean

include("../../Tutorials/HandsOn4/contract_network.jl")
include("../../Tutorials/HandsOn4/aklt_tensornetwork.jl")

"""
Stretch Tutorial: quantum belief propagation. This is the completed solution.

You are asked to run belief propagation on the norm network `⟨ψ|ψ⟩` of a tensor network
state and use it to compute expectation values. For each numbered step (1), (2), (3) below,
fill in the missing code, then run `main` in `Tutorials/HandsOn4/5-quantumbp.jl`.

The norm network has *two* tensors per vertex, the ket `ψ_v` and the bra `conj(ψ_v)`, joined
over the physical index. Each edge therefore carries two indices, the ket leg `l` and the bra
leg `l'`, so a message here is a matrix rather than a vector. Multiply the incoming messages
into the ket one at a time, then multiply by the bra.

Expectation values are ratios of two contractions that share the same messages, so the
normalization of the messages cancels. Nothing like `binormalized_messages` is needed here.

Nothing in this file is specific to the AKLT state. It works for any state stored the way
`aklt_tensornetwork` returns it, and every function takes that `state` as its first
argument, which is what keeps these names apart from the ones in `belief_propagation.jl`.
"""

"""
    initial_messages(state::NamedTuple)

The messages to start from: on each directed edge, the identity matrix between the ket leg
and the bra leg, normalized.
"""
function initial_messages(state::NamedTuple)
    return Dict(
        e => normalize(delta(state.links[e], prime(state.links[e])))
            for e in all_edges(state.g)
    )
end

"""
    updated_message(state::NamedTuple, messages, e)

The new message along the directed edge `e`, sent from `src(e)` to `dst(e)`: the ket and bra
on `src(e)` contracted with every message arriving at `src(e)` except the one from `dst(e)`.
"""
function updated_message(state::NamedTuple, messages, e)
    # The directed edges pointing into `src(e)`, excluding the one coming from `dst(e)`
    incoming_es = setdiff(boundary_edges(state.g, [src(e)]; dir = :in), [reverse(e)])

    # (1) Multiply the incoming messages into the ket one at a time, then multiply by the bra.
    t = ket_tensor(state, src(e))
    for e_in in incoming_es
        t = t * messages[e_in]
    end
    t = t * bra_tensor(state, src(e))
    return normalize(t)
end

function update_messages(state::NamedTuple, messages)
    updated_messages = copy(messages)
    for e in all_edges(state.g)
        updated_messages[e] = updated_message(state, messages, e)
    end
    return updated_messages
end

function message_distance(state::NamedTuple, messages, old_messages)
    return mean([1 - dot(messages[e], old_messages[e])^2 for e in all_edges(state.g)])
end

"""
    belief_propagation(state::NamedTuple[, messages]; kwargs...)

Run belief propagation on the norm network of `state`: update every message from the current
ones, repeatedly, until they stop changing.

# Keywords
- `niters::Int = 2000`: The maximum number of iterations to perform.
- `tol::Float64 = 1e-14`: Stop once `message_distance` drops below this. It is tighter than
  the default used for the Ising model because the error in an expectation value goes
  roughly like the square root of this measure, so a loose tolerance costs several digits.
- `outputlevel::Int = 1`: Set to `0` to suppress the printed convergence message.

# Returns
- `messages::Dict`: The converged messages, one per directed edge.
- `niters`: The number of iterations taken, or `nothing` if `niters` was reached first.
"""
function belief_propagation(
        state::NamedTuple, messages = initial_messages(state);
        niters::Int = 2000, tol::Float64 = 1.0e-14, outputlevel::Int = 1
    )
    for i in 1:niters
        old_messages = messages
        messages = update_messages(state, messages)
        if message_distance(state, messages, old_messages) < tol
            outputlevel >= 1 && println("BP Algorithm Converged after $i iterations")
            return messages, i
        end
    end
    outputlevel >= 1 && println("BP Algorithm did NOT converge after $niters iterations")
    return messages, nothing
end

"""
    expect(state::NamedTuple, messages, v, O::ITensor)

The expectation value `⟨ψ|O_v|ψ⟩ / ⟨ψ|ψ⟩` of the operator `O` on vertex `v`.
"""
function expect(state::NamedTuple, messages, v, O::ITensor)
    incoming = [messages[e] for e in boundary_edges(state.g, [v]; dir = :in)]

    # (2) The numerator and the denominator differ only by the operator on the ket.
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
    expect_bond(state::NamedTuple, messages, v, w, Ov, Ow)

The expectation value `⟨ψ|O_v O_w|ψ⟩ / ⟨ψ|ψ⟩` of an operator on vertex `v` times one on the
neighbouring vertex `w`. Either operator may be `nothing`, meaning the identity.
"""
function expect_bond(state::NamedTuple, messages, v, w, Ov, Ow)
    incoming = [messages[e] for e in boundary_edges(state.g, [v, w]; dir = :in)]

    # (3) The same idea on a cluster of two vertices. Each boundary message is absorbed into
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
    heisenberg_terms(state::NamedTuple, v, w)

`S_v · S_w` written as a list of `(O_v, O_w, coefficient)` triples, using

    S_v · S_w = SᶻSᶻ + (S⁺S⁻ + S⁻S⁺) / 2.

Every term is a product of one operator on `v` and one on `w`, which is the form `expect_bond`
takes.
"""
function heisenberg_terms(state::NamedTuple, v, w)
    ops_v = spin_operators(state.sites[v])
    ops_w = spin_operators(state.sites[w])
    return [
        (ops_v.sz, ops_w.sz, 1.0),
        (ops_v.sp, ops_w.sm, 0.5),
        (ops_v.sm, ops_w.sp, 0.5),
    ]
end

"""
    heisenberg_bond(state::NamedTuple, messages, v, w)

The Heisenberg bond energy `⟨S_v · S_w⟩`.
"""
function heisenberg_bond(state::NamedTuple, messages, v, w)
    return sum(
        c * expect_bond(state, messages, v, w, Ov, Ow)
            for (Ov, Ow, c) in heisenberg_terms(state, v, w)
    )
end

"""
    heisenberg_bond_squared(state::NamedTuple, messages, v, w)

The expectation value of `(S_v · S_w)²`.

Squaring the sum of three terms gives nine, each still a product of one operator on `v` and
one on `w`, because `(A ⊗ B)(A' ⊗ B') = AA' ⊗ BB'`.
"""
function heisenberg_bond_squared(state::NamedTuple, messages, v, w)
    terms = heisenberg_terms(state, v, w)
    return sum(
        c1 * c2 * expect_bond(state, messages, v, w, apply(Ov1, Ov2), apply(Ow1, Ow2))
            for (Ov1, Ow1, c1) in terms, (Ov2, Ow2, c2) in terms
    )
end

"""
    aklt_bond_energy(state::NamedTuple, messages, v, w)

The energy of one bond of the spin-1 AKLT parent Hamiltonian, the projector onto total spin
2 on that bond,

    P₂ = 1/3 + (S_v · S_w) / 2 + (S_v · S_w)² / 6.

The AKLT state is annihilated by every such projector, so this is zero for the AKLT state.

This is the spin-1 projector, so it only applies where both vertices have degree two, such
as in the bulk of a chain or anywhere on a ring. On a lattice of higher coordination number
the parent Hamiltonian projects onto a different maximal spin.
"""
function aklt_bond_energy(state::NamedTuple, messages, v, w)
    ss = heisenberg_bond(state, messages, v, w)
    ss2 = heisenberg_bond_squared(state, messages, v, w)
    return 1 / 3 + ss / 2 + ss2 / 6
end

"""
    expect_exact(state::NamedTuple, operators::Dict)
    expect_exact(state::NamedTuple, v, O::ITensor)
    expect_exact(state::NamedTuple, v, w, Ov::ITensor, Ow::ITensor)

The same expectation values, from contracting the whole norm network exactly with
`contract_network`. `operators` maps vertices to the operators to insert there.

Only usable on small graphs. It is provided so that you have something to check your belief
propagation answers against.
"""
function expect_exact(state::NamedTuple, operators::Dict)
    vs = collect(vertices(state.g))
    bras = [bra_tensor(state, u) for u in vs]
    denominator = contract_network([[ket_tensor(state, u) for u in vs]; bras])[]
    numerator = contract_network(
        [[ket_tensor(state, u, get(operators, u, nothing)) for u in vs]; bras]
    )[]
    return numerator / denominator
end
expect_exact(state::NamedTuple, v, O::ITensor) = expect_exact(state, Dict(v => O))
function expect_exact(state::NamedTuple, v, w, Ov::ITensor, Ow::ITensor)
    return expect_exact(state, Dict(v => Ov, w => Ow))
end
