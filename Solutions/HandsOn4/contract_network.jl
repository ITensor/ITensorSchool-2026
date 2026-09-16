using Graphs: add_edge!, edges, nv, rem_vertex!, vertices
using ITensors: dim, inds
using NamedGraphs: NamedGraph

"""
    contract_network(tn::Dict, g::NamedGraph)

Contract the tensor network using an eager contraction sequence.

# Arguments
- `tn::Dict`: A dictionary representing the tensor network, where keys are vertices and values are tensors.
- `g::NamedGraph`: The named graph representing the structure of the tensor network.
"""
function contract_network(tn::Dict, g::NamedGraph)
    if nv(g) == 1
        return only(values(tn))
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
