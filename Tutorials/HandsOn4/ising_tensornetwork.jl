using ITensors: ITensor, Index, apply, delta, prime
using NamedGraphs: NamedGraph, edges, vertices, dst, src, neighbors, incident_edges
using QuadGK: quadgk

"""
    ising_tensornetwork(g::NamedGraph, β::Real)

Constructs the tensor network representation of the Ising model on a given graph.

# Arguments
- `g::NamedGraph`: The graph representing the lattice structure of the Ising model.
- `β::Real`: The inverse temperature parameter.

# Returns
- `tn::Dict{Any, ITensor}`: A dictionary mapping each vertex to its corresponding tensor in the network.
"""
function ising_tensornetwork(g::NamedGraph, β::Real)
    links = Dict(e => Index(2, "e$(src(e))_$(dst(e))") for e in edges(g))
    links = merge(links, Dict(reverse(e) => links[e] for e in edges(g)))

    # symmetric sqrt of Boltzmann matrix W = exp(β σσ')
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

Computes `ϕ(β) = - β * f(β)` for the 2D Ising model in the thermodynamic limit using Onsager's solution.

# Arguments
- `β::Real`: The inverse temperature parameter.

# Returns
- `phi::Real`: `ϕ(β) = - β * f(β)` where `f(β)`` is the exact free energy density.
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
