using Graphs: vertices
using NamedGraphs: all_edges, boundary_edges
using ITensors: ITensor, delta, inds, normalize, prime
using LinearAlgebra: dot
using Statistics: mean

include("contract_network.jl")
include("aklt_tensornetwork.jl")

"""
Stretch Tutorial: quantum belief propagation.

You are asked to run belief propagation on the norm network `⟨ψ|ψ⟩` of a tensor network
state and use it to compute expectation values. For each numbered step (1), (2), (3) below,
fill in the missing code, then run `main` in `5-quantumbp.jl`.

The norm network has *two* tensors per vertex, the ket `ψ_v` and the bra `conj(ψ_v)`, joined
over the physical index. Each edge therefore carries two indices, the ket leg `l` and the bra
leg `l'`, so a message here is a matrix rather than a vector. Multiply the incoming messages
into the ket one at a time, then multiply by the bra.

Expectation values are ratios of two contractions that share the same messages, so the
normalization of the messages cancels.

Nothing in this file is specific to the AKLT state. It works for any tensor network state stored the way
`aklt_tensornetwork` returns it, and every function takes that `state` as its first
argument, which is what keeps these names apart from the ones in `belief_propagation.jl`.
"""

"""
    initial_messages(state::NamedTuple)

The messages to start from: on each directed edge, the identity matrix between the ket leg
and the bra leg, normalized is typically a good choice.
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

    # (1) Starting from `ket_tensor(state, src(e))`, multiply in the message on each edge of
    #     `incoming_es` one at a time, then close with `bra_tensor(state, src(e))` and
    #     normalize.
    # ...

    return messages[e]
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

    # (2) Build the numerator and the denominator. Both start from a ket layer, with the
    #     operator applied for the numerator via `ket_tensor(state, v, O)`, absorb every
    #     message in `incoming`, and then close with `bra_tensor(state, v)`. Each contraction
    #     gives a scalar `ITensor`, so use `[]` to get a number, and return their ratio.
    # ...

    return 0.0
end

"""
    expect_bond(state::NamedTuple, messages, v, w, Ov, Ow)

The expectation value `⟨ψ|O_v O_w|ψ⟩ / ⟨ψ|ψ⟩` of an operator on vertex `v` times one on the
neighbouring vertex `w`. Either operator may be `nothing`, meaning the identity.
"""
function expect_bond(state::NamedTuple, messages, v, w, Ov, Ow)
    incoming = [messages[e] for e in boundary_edges(state.g, [v, w]; dir = :in)]

    # (3) The same idea for a cluster of two neighbouring vertices. Absorb each message in
    #     `incoming` into whichever of the two ket layers shares an index with it, contract
    #     the two kets together over the bond they share, and then close with both bras.
    #     Again form the numerator and the denominator and return their ratio.
    # ...

    return 0.0
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
