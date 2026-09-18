using Graphs: degree, edges, src, dst, vertices
using NamedGraphs: NamedGraph, incident_edges
using ITensors: ITensor, Index, apply, combiner, dag, dim, noprime, prime

"""
    aklt_tensornetwork(g::NamedGraph)

Build the AKLT (valence bond solid) state on the graph `g`, together with the tensor network
representing its norm `⟨ψ|ψ⟩`.

Each edge of `g` carries a singlet of two spin-1/2s. At a vertex of degree `z` those `z`
spin-1/2s are projected onto their maximal total spin `S = z/2`, so the physical index there
has dimension `z + 1`. On a chain every bulk vertex has `z = 2` and the state is the familiar
spin-1 AKLT chain; on a square lattice `z = 4` and it is the spin-2 AKLT state.

The norm network `tn` has one tensor per vertex of `g`, namely `ψ_v` contracted with its own
conjugate over the physical index. It therefore lives on the *same* graph `g` and can be
passed straight to `belief_propagation`.

# Returns
A named tuple containing:
- `g::NamedGraph`: The graph, as passed in.
- `psi::Dict`: The state, one tensor per vertex, each with one physical and `z` virtual indices.
- `sites::Dict`: The physical index on each vertex.
- `links::Dict`: The virtual index on each (directed) edge.
- `combiners::Dict`: The combiner used on each edge of the norm network (see below).
- `tn::Dict`: The norm network `⟨ψ|ψ⟩`, one tensor per vertex of `g`.

Each edge of the norm network would naturally carry two indices, one from the bra and one
from the ket. They are fused into a single index with a combiner so that the network has
exactly one index per edge, as `belief_propagation` expects. The combiners are returned
because inserting an operator later has to reuse the very same ones: building a fresh
combiner would produce a new index that no longer matches the messages.
"""
function aklt_tensornetwork(g::NamedGraph)
    links = Dict(e => Index(2, "l$(src(e))_$(dst(e))") for e in edges(g))
    links = merge(links, Dict(reverse(e) => links[e] for e in edges(g)))
    sites = Dict(v => Index(degree(g, v) + 1, "s$(v)") for v in vertices(g))

    # Project the z virtual spin-1/2s on each vertex onto total spin S = z/2. A virtual index
    # value of 1 is "up" and 2 is "down"; a configuration with k downs contributes to the
    # physical basis state k + 1, with the normalization of a symmetrized state.
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

    # Place the singlet on each edge, absorbed into the source endpoint.
    for e in edges(g)
        l = links[e]
        epsilon = ITensor(prime(l), l)
        epsilon[prime(l) => 1, l => 2] = 1.0
        epsilon[prime(l) => 2, l => 1] = -1.0
        psi[src(e)] = noprime(psi[src(e)] * epsilon)
    end

    # The norm network, on the same graph, with the bra and ket indices of each edge fused.
    tn = Dict{Any, ITensor}()
    for v in vertices(g)
        incident = [links[e] for e in incident_edges(g, v)]
        tn[v] = psi[v] * dag(prime(psi[v], incident))
    end
    combiners = Dict(e => combiner(links[e], prime(links[e])) for e in edges(g))
    combiners = merge(combiners, Dict(reverse(e) => combiners[e] for e in edges(g)))
    for e in edges(g)
        tn[src(e)] = tn[src(e)] * combiners[e]
        tn[dst(e)] = tn[dst(e)] * combiners[e]
    end

    return (; g, psi, sites, links, combiners, tn)
end

"""
    spin_operators(s::Index)

The spin operators `Sᶻ`, `S⁺` and `S⁻` on a physical index `s` of dimension `2S + 1`,
returned as `ITensor`s with indices `(s', s)`.
"""
function spin_operators(s::Index)
    d = dim(s)
    S = (d - 1) / 2
    m(i) = S - (i - 1)   # the index value i holds magnetic quantum number m
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

"""
    operator_tensor(state, v, O::ITensor)

The vertex tensor of the norm network with the operator `O` inserted at vertex `v`.

Replacing `state.tn[v]` with this tensor turns the norm network `⟨ψ|ψ⟩` into the network
`⟨ψ|O_v|ψ⟩`, so the ratio of the two contractions is the expectation value of `O` on `v`.
"""
function operator_tensor(state, v, O::ITensor)
    psi_v = state.psi[v]
    s = state.sites[v]
    incident = [state.links[e] for e in incident_edges(state.g, v)]
    t = psi_v * O * dag(prime(prime(psi_v, incident), s))
    for e in incident_edges(state.g, v)
        t = t * state.combiners[e]
    end
    return t
end
