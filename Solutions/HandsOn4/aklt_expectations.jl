using Graphs: edges, src, dst, vertices
using ITensors: apply
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
    heisenberg_terms(state, v, w)

`S_v · S_w` written as a list of `(O_v, O_w, coefficient)` triples, using

    S_v · S_w = SᶻSᶻ + (S⁺S⁻ + S⁻S⁺) / 2.

Writing it this way means every term is a product of one operator on `v` and one on `w`,
which is what `expect_bond_bp` and `expect_bond_exact` take.
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

# Evaluate a bond expectation value either with belief propagation or by exact contraction.
function bond_evaluator(state, messages, v, w; exact)
    return if exact
        (Ov, Ow) -> expect_bond_exact(state, v, w, Ov, Ow)
    else
        (Ov, Ow) -> expect_bond_bp(state, messages, v, w, Ov, Ow)
    end
end

"""
    heisenberg_bond(state, messages, v, w; exact = false)

The Heisenberg bond energy `⟨S_v · S_w⟩`. Set `exact = true` to contract the finite network
exactly instead of using belief propagation.
"""
function heisenberg_bond(state, messages, v, w; exact = false)
    f = bond_evaluator(state, messages, v, w; exact)
    return sum(c * f(Ov, Ow) for (Ov, Ow, c) in heisenberg_terms(state, v, w))
end

"""
    heisenberg_bond_squared(state, messages, v, w; exact = false)

The expectation value of `(S_v · S_w)²`.

Squaring the sum above gives nine terms, each still a product of one operator on `v` and one
on `w`, because `(A ⊗ B)(A' ⊗ B') = AA' ⊗ BB'`.
"""
function heisenberg_bond_squared(state, messages, v, w; exact = false)
    f = bond_evaluator(state, messages, v, w; exact)
    terms = heisenberg_terms(state, v, w)
    return sum(
        c1 * c2 * f(apply(Ov1, Ov2), apply(Ow1, Ow2))
            for (Ov1, Ow1, c1) in terms, (Ov2, Ow2, c2) in terms
    )
end

"""
    aklt_bond_energy(state, messages, v, w; exact = false)

The energy of one bond of the spin-1 AKLT parent Hamiltonian, which is the projector onto
total spin 2 on that bond,

    P₂ = 1/3 + (S_v · S_w) / 2 + (S_v · S_w)² / 6.

The AKLT state is annihilated by every such projector, so this is zero for the AKLT state.

This particular projector is the spin-1 one, so it only applies where both vertices have
degree two, such as on a ring. On a lattice of higher coordination number the parent
Hamiltonian projects onto a different maximal spin.
"""
function aklt_bond_energy(state, messages, v, w; exact = false)
    ss = heisenberg_bond(state, messages, v, w; exact)
    ss2 = heisenberg_bond_squared(state, messages, v, w; exact)
    return 1 / 3 + ss / 2 + ss2 / 6
end

"""
    main(; kwargs...)

Build the AKLT state on a graph, run belief propagation on its norm network, and compare
local expectation values to exact contraction of the same finite network.

On a periodic ring every vertex has degree two, so this is the spin-1 AKLT chain, for which
the nearest neighbour correlations are known exactly in the thermodynamic limit:

    ⟨Sᶻ⟩ = 0,   ⟨(Sᶻ)²⟩ = 2/3,   ⟨SᶻSᶻ⟩ = -4/9,   ⟨S·S⟩ = -4/3,   ⟨(S·S)²⟩ = 2.

Those last two are exactly the values that make the AKLT parent Hamiltonian bond energy
vanish, which is reported as `energy` when the vertices have degree two.

# Keywords
- `g`: The graph to build the AKLT state on. Defaults to a periodic ring of six vertices.
- `tol::Float64 = 1e-14`: The belief propagation convergence tolerance. This is tighter than
  the default used for the Ising model, because the error in an expectation value goes
  roughly like the square root of the message convergence measure.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `sz::Number`: `⟨Sᶻ⟩` on one vertex, from belief propagation.
- `sz2::Number`: `⟨(Sᶻ)²⟩` on one vertex, from belief propagation.
- `ss_bp::Number`: `⟨S·S⟩` on one bond, from belief propagation.
- `ss_exact::Number`: `⟨S·S⟩` on the same bond, from exact contraction.
- `energy`: The AKLT parent Hamiltonian bond energy from belief propagation, which should be
  zero, or `nothing` when the bond is not a spin-1 one.
- `state`: The AKLT state and its norm network.
- `messages::Dict`: The converged belief propagation messages.
- `niters`: The number of iterations taken for convergence.
"""
function main(; g = named_grid((6, 1); periodic = true), tol::Float64 = 1.0e-14, outputlevel::Int = 1)
    state = aklt_tensornetwork(g)
    messages, niters = belief_propagation(state.tn, g; niters = 2000, tol, outputlevel)

    v = first(vertices(g))
    ops = spin_operators(state.sites[v])
    sz = expect_bp(state, messages, v, ops.sz)
    sz2 = expect_bp(state, messages, v, apply(ops.sz, ops.sz))

    e = first(edges(g))
    ss_bp = heisenberg_bond(state, messages, src(e), dst(e))
    ss_exact = heisenberg_bond(state, messages, src(e), dst(e); exact = true)

    # The spin-1 parent Hamiltonian projector only applies when both ends have degree two.
    spin_one_bond = dim(state.sites[src(e)]) == 3 && dim(state.sites[dst(e)]) == 3
    energy = spin_one_bond ? aklt_bond_energy(state, messages, src(e), dst(e)) : nothing

    if outputlevel > 0
        println("Physical spin S on each vertex: ", (dim(state.sites[v]) - 1) / 2)
        println("⟨Sᶻ⟩       BP    = ", sz)
        println("⟨(Sᶻ)²⟩    BP    = ", sz2)
        println("⟨S·S⟩      BP    = ", ss_bp)
        println("⟨S·S⟩      exact = ", ss_exact, "  (this finite network)")
        if energy !== nothing
            println("AKLT parent Hamiltonian bond energy = ", energy, "  (exactly zero)")
        end
    end

    return (; sz, sz2, ss_bp, ss_exact, energy, state, messages, niters)
end
