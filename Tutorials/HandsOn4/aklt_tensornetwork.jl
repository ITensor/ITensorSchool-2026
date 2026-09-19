using Graphs: degree, edges, src, dst, vertices
using NamedGraphs: NamedGraph, incident_edges
using ITensors: ITensor, Index, apply, dag, dim, prime

"""
    aklt_tensornetwork(g::NamedGraph)

Build the AKLT (valence bond solid) state on the graph `g`.

Each edge of `g` carries a singlet of two spin-1/2s. At a vertex of degree `z` those `z`
spin-1/2s are projected onto their maximal total spin `S = z/2`, so the physical index there
has dimension `z + 1`. On a ring every vertex has `z = 2` and the state is the familiar
spin-1 AKLT chain; on a square lattice `z = 4` and it is the spin-2 AKLT state.

The result is the state itself: one tensor per vertex, each carrying one physical index and
one virtual index per incident edge. The norm network `⟨ψ|ψ⟩` is never built explicitly.
Its two layers are reached through `ket_tensor` and `bra_tensor`.

# Returns
A named tuple containing:
- `g::NamedGraph`: The graph, as passed in.
- `psi::Dict`: The state, one tensor per vertex.
- `sites::Dict`: The physical index on each vertex.
- `links::Dict`: The virtual index on each (directed) edge.
"""
function aklt_tensornetwork(g::NamedGraph)
    links = Dict(e => Index(2, "l$(src(e))_$(dst(e))") for e in edges(g))
    links = merge(links, Dict(reverse(e) => links[e] for e in edges(g)))
    sites = Dict(v => Index(degree(g, v) + 1, "s$(v)") for v in vertices(g))

    # Project the z virtual spin-1/2s on each vertex onto total spin S = z/2. Virtual index
    # value 1 is "up" and 2 is "down". A configuration of the virtual spins with k downs
    # belongs to the physical state with S^z = S - k, stored as index value k + 1, and every
    # such configuration enters with the same weight, normalized over the binomial(z, k) of them.
    psi = Dict{Any, ITensor}()
    for v in vertices(g)
        es = incident_edges(g, v)
        z = length(es)
        t = ITensor(sites[v], [links[e] for e in es]...)
        for config in Iterators.product(ntuple(_ -> 1:2, z)...)
            k = count(==(2), config)
            t[sites[v] => k + 1, (links[es[i]] => config[i] for i in 1:z)...] =
                1 / sqrt(binomial(z, k))
        end
        psi[v] = t
    end

    # Turn the shared virtual index on each edge into a singlet by absorbing the
    # antisymmetric tensor ε into the tensor at one end of the edge.
    for e in edges(g)
        l = links[e]
        epsilon = ITensor(prime(l), l)
        epsilon[prime(l) => 1, l => 2] = 1.0
        epsilon[prime(l) => 2, l => 1] = -1.0
        psi[src(e)] = apply(epsilon, psi[src(e)])
    end

    return (; g, psi, sites, links)
end

"""
    link_indices(state, v)

The virtual indices of the tensor on vertex `v`, one per incident edge.

This and the functions below work for any tensor network state stored the way
`aklt_tensornetwork` returns it: a named tuple with the graph `g`, the tensors `psi`, the
physical indices `sites` and the virtual indices `links`.
"""
function link_indices(state, v)
    return [state.links[e] for e in incident_edges(state.g, v)]
end

"""
    ket_tensor(state, v)
    ket_tensor(state, v, O::ITensor)

The ket layer of the norm network at vertex `v`, which is just the tensor `ψ_v` of the state,
optionally with the operator `O` applied to its physical index.

`O` is applied with `apply`, so the physical index of the result is unprimed and still
contracts with `bra_tensor(state, v)`. Passing `nothing` for `O` returns the plain ket.
"""
ket_tensor(state, v) = state.psi[v]
ket_tensor(state, v, O::ITensor) = apply(O, state.psi[v])
ket_tensor(state, v, ::Nothing) = ket_tensor(state, v)

"""
    bra_tensor(state, v)

The bra layer of the norm network at vertex `v`: the complex conjugate of `ψ_v` with its
*virtual* indices primed.

The physical index is left unprimed, so `ket_tensor(state, v) * bra_tensor(state, v)`
contracts over it while the virtual indices stay distinct. Each edge of the norm network
therefore carries two indices, the ket leg `l` and the bra leg `l'`.

The two layers are kept as separate tensors. When contracting with messages, multiply the
messages into the ket first and then multiply by the bra.
"""
bra_tensor(state, v) = dag(prime(state.psi[v], link_indices(state, v)))

"""
    spin_operators(s::Index)

The spin operators `Sᶻ`, `S⁺` and `S⁻` for a spin `S`, on a physical index `s` of dimension
`2S + 1`, as `ITensor`s with indices `(s', s)`. Index value `i` holds `Sᶻ = S - (i - 1)`, so
value `1` is the fully polarized "up" state, matching the convention of `aklt_tensornetwork`.
"""
function spin_operators(s::Index)
    d = dim(s)
    S = (d - 1) / 2
    m(i) = S - (i - 1)   # index value i holds S^z eigenvalue m
    sz = zeros(d, d)
    sp = zeros(d, d)
    sm = zeros(d, d)
    for i in 1:d
        sz[i, i] = m(i)
        i > 1 && (sp[i - 1, i] = sqrt(S * (S + 1) - m(i) * (m(i) + 1)))
        i < d && (sm[i + 1, i] = sqrt(S * (S + 1) - m(i) * (m(i) - 1)))
    end
    return (;
        sz = ITensor(sz, prime(s), s),
        sp = ITensor(sp, prime(s), s),
        sm = ITensor(sm, prime(s), s),
    )
end
