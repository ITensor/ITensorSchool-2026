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

This returns the state itself, one tensor per vertex, each carrying one physical index and
one virtual index per incident edge. It does *not* build the norm network `⟨ψ|ψ⟩`: see
`ket_tensor` and `bra_tensor`, and the note there about why the two layers are kept apart.

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
        psi[src(e)] = apply(epsilon, psi[src(e)])
    end

    return (; g, psi, sites, links)
end

"""
    link_indices(state, v)

The virtual indices on the edges incident to vertex `v`.
"""
function link_indices(state, v)
    return [state.links[e] for e in incident_edges(state.g, v)]
end

"""
    ket_tensor(state, v)
    ket_tensor(state, v, O::ITensor)

The ket layer at vertex `v`, optionally with the operator `O` applied to its physical index.

`O` is applied with `apply`, so the physical index of the result is unprimed and still
contracts with `bra_tensor(state, v)`.
"""
ket_tensor(state, v) = state.psi[v]
ket_tensor(state, v, O::ITensor) = apply(O, state.psi[v])
ket_tensor(state, v, ::Nothing) = ket_tensor(state, v)

"""
    bra_tensor(state, v)

The bra layer at vertex `v`: the conjugate of the ket with its *virtual* indices primed.

The physical index is left unprimed so that `ket_tensor(state, v) * bra_tensor(state, v)`
contracts over it, while the virtual indices stay distinct. Each edge of the norm network
therefore carries two indices, the ket leg `l` and the bra leg `l'`.

Note that the two layers are deliberately never multiplied together to form a single
"double layer" tensor per vertex. Doing so would produce a tensor with `2z` virtual indices,
costing `χ^(2z)` memory at bond dimension `χ`, which is by far the most expensive object in
sight. Contracting the messages into the ket first and only then closing with the bra keeps
the cost down to roughly `χ^(z+1)`. For a degree four vertex at `χ = 10` that is the
difference between 763 MiB and under a megabyte.
"""
bra_tensor(state, v) = dag(prime(state.psi[v], link_indices(state, v)))

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
