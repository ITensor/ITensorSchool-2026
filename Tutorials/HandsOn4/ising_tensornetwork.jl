using ITensors: ITensor, Index, apply, delta, prime
using NamedGraphs: NamedGraph, edges, vertices, dst, src, neighbors, incident_edges
using QuadGK: quadgk

"""
    ising_tensornetwork(g::NamedGraph, β::Real)

The tensor network whose full contraction is the partition function of the ferromagnetic
Ising model on the graph `g` at inverse temperature `β`, with the Boltzmann weight on every
edge scaled by `1/2`.

The tensor on a vertex is a copy tensor (`delta`), which forces the spin to take the same
value on every incident edge, with the symmetric square root of the Boltzmann matrix
`W = exp(β s s') / 2` absorbed into each of its legs. Two neighbouring tensors then share one
full `W` across their common edge.

# Arguments
- `g::NamedGraph`: The graph whose vertices carry the spins and whose edges carry the couplings.
- `β::Real`: The inverse temperature.

# Returns
- `tn::Dict{Any, ITensor}`: A dictionary mapping each vertex of `g` to its tensor.
"""
function ising_tensornetwork(g::NamedGraph, β::Real)
    links = Dict(e => Index(2, "e$(src(e))_$(dst(e))") for e in edges(g))
    links = merge(links, Dict(reverse(e) => links[e] for e in edges(g)))

    # symmetric square root of the Boltzmann matrix W = exp(β s s') / 2, whose eigenvalues
    # are cosh(β) and sinh(β)
    λ1, λ2 = cosh(β), sinh(β)
    α = 0.5 * (sqrt(λ1) + sqrt(λ2))
    ϕ = 0.5 * (sqrt(λ1) - sqrt(λ2))
    sqrt_W = [α ϕ; ϕ α]

    ts = map(vertices(g)) do v
        es = incident_edges(g, v)
        t = delta([links[e] for e in es])
        for e in es
            t = apply(t, ITensor(sqrt_W, links[e], prime(links[e])))
        end
        return v => t
    end
    return Dict(ts)
end

"""
    ising_phi(β::Real)

The exact free energy density `ϕ(β) = -β f(β) = log(Z) / N` of the 2D Ising model on the
square lattice in the thermodynamic limit, from Onsager's solution. The `-log(2)` accounts
for the `1/2` scaling of the Boltzmann weights used in `ising_tensornetwork`.

# Arguments
- `β::Real`: The inverse temperature.

# Returns
- `phi::Real`: The free energy density.
"""
function ising_phi(β)
    g(θ1, θ2) = log(
        cosh(2β) * cosh(2β) -
            sinh(2β) * cos(θ1) -
            sinh(2β) * cos(θ2)
    )
    inner(θ2) = quadgk(θ1 -> g(θ1, θ2), 0, 2π)[1]
    return -log(2) + (1 / (8π^2)) * quadgk(inner, 0, 2π)[1]
end
