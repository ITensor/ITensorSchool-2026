using Graphs: add_edge!, dst, edges, neighbors, nv, rem_vertex!, src, vertices
using ITensors: dim, inds
using NamedGraphs: NamedGraph

"""
    contract_network(tn::Dict, g::NamedGraph)
    contract_network(tn::Dict)
    contract_network(tn::Vector)

Contract a whole tensor network down to a single `ITensor`.

The contraction is greedy: at each step the pair of neighbouring tensors that is cheapest to
multiply together is contracted, and the graph is updated to merge those two vertices. This
is fine for the small networks in these tutorials, but the cost still grows exponentially
with the size of the network, which is why we turn to belief propagation.

The tensor network can be given as a dictionary `tn` from the vertices of `g` to tensors, as
a dictionary alone, in which case tensors that share an index are treated as neighbours, or
simply as a vector of tensors.
"""
function contract_network(tn::Dict, g::NamedGraph)
    if nv(g) == 1
        return only(values(tn))
    end
    if isempty(edges(g))
        # Nothing left to contract over: the remaining tensors share no indices, so the
        # result is just their product.
        return prod(values(tn))
    end
    min_e = argmin(edges(g)) do e
        t1 = tn[src(e)]
        t2 = tn[dst(e)]
        # Contraction cost.
        cost = prod(dim, symdiff(inds(t1), inds(t2)); init = 1)
        cost *= prod(dim, intersect(inds(t1), inds(t2)); init = 1)
        return cost
    end
    v1, v2 = src(min_e), dst(min_e)
    tn = copy(tn)
    tn[v2] = tn[v1] * tn[v2]
    delete!(tn, v1)
    g = copy(g)
    neighbors_src = setdiff(neighbors(g, v1), [v2])
    neighbors_dst = setdiff(neighbors(g, v2), [v1])
    rem_vertex!(g, v1)
    for n_src in neighbors_src
        add_edge!(g, v2 => n_src)
    end
    for n_dst in neighbors_dst
        add_edge!(g, v2 => n_dst)
    end
    return contract_network(tn, g)
end
function contract_network(tn::Dict)
    g = NamedGraph(collect(keys(tn)))
    for v1 in vertices(g)
        for v2 in vertices(g)
            if v1 ≠ v2
                if !isempty(intersect(inds(tn[v1]), inds(tn[v2])))
                    add_edge!(g, v1, v2)
                end
            end
        end
    end
    return contract_network(tn, g)
end
function contract_network(tn::Vector)
    return contract_network(Dict(i => tn[i] for i in 1:length(tn)))
end
