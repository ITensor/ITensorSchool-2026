using Graphs: vertices, edges, src, dst, neighbors, simplecycles_limited_length
using NamedGraphs: NamedEdge, NamedGraph
using NamedGraphs.GraphsExtensions: all_edges, boundary_edges
using LinearAlgebra: dot, normalize
using ITensors: ITensor, Index, inds, onehot

include("contract_network.jl")

function updated_message(tn::Dict, g::NamedGraph, messages::Dict, e::NamedEdge)
    incoming_es = setdiff(boundary_edges(g, [src(e)]; dir = :in), [reverse(e)])
    local_tensor = tn[src(e)]
    messages = copy(messages)
    incoming_messages = [messages[e] for e in incoming_es]
    return normalize(contract_network([[local_tensor]; incoming_messages]))
end

function update_messages(tn::Dict, g::NamedGraph, messages::Dict)
    updated_messages = copy(messages)
    for e in all_edges(g)
        updated_messages[e] = updated_message(tn, g, messages, e)
    end
    return updated_messages
end

function initial_messages(tn::Dict, g::NamedGraph)
    linkind(tn, e) = only(intersect(inds(tn[src(e)]), inds(tn[dst(e)])))
    return Dict(e => onehot(linkind(tn, e) => 1) for e in all_edges(g))
end

function binormalized_messages(g::NamedGraph, messages::Dict)
    binorm_messages = copy(messages)
    for e in edges(g)
        n = (messages[e] * messages[reverse(e)])[]
        binorm_messages[e] = sign(n) * messages[e] / sqrt(abs(n))
        binorm_messages[reverse(e)] = messages[reverse(e)] / sqrt(abs(n))
    end
    return binorm_messages
end

"""
    phi_bp(tn::Dict, g::NamedGraph, messages::Dict)

Computes the Bethe-Peierls free energy density.

# Arguments
- `tn::Dict`: A dictionary representing the tensor network, where keys are vertices and values are tensors.
- `g::NamedGraph`: The named graph representing the structure of the tensor network.
- `messages::Dict`: A dictionary containing the messages for each edge in the graph.

# Returns
- `phi::Number`: The free energy estimate per vertex.
"""
function phi_bp(tn::Dict, g::NamedGraph, messages::Dict)
    messages = binormalized_messages(g, messages)
    return sum([log(phi_factor(tn, g, messages, v)) for v in vertices(g)]) / nv(g)
end

"""
    belief_propagation(tn::Dict, g::NamedGraph[, messages::Dict]; kwargs...)

Performs the Belief Propagation algorithm on a given tensor network defined over a named graph.

# Arguments
- `tn::Dict`: A dictionary representing the tensor network, where keys are vertices and values are tensors.
- `g::NamedGraph`: The named graph representing the structure of the tensor network.
- `messages::Dict = initial_messages(tn, g)`: Initial messages for each edge in the graph.

# Keywords
- `niters::Int`: The maximum number of iterations to perform.
- `tol::Float64 = 1e-10`: The tolerance for convergence.
- `outputlevel::Int = 1`: The verbosity level of the output.

# Returns
- `messages::Dict`: A dictionary containing the converged messages for each edge in the graph.
- `niters::Int`: The number of iterations taken to converge, or `nothing` if not converged within `niters`.
"""
function belief_propagation(
        tn::Dict, g::NamedGraph, messages::Dict = initial_messages(tn, g); niters::Int,
        tol::Float64 = 1.0e-10, outputlevel::Int = 1
    )
    for i in 1:niters
        old_messages = copy(messages)
        messages = update_messages(tn, g, messages)
        if mean([1 - dot(messages[e], old_messages[e])^2 for e in all_edges(g)]) < tol
            outputlevel >= 1 && println("BP Algorithm Converged after $i iterations")
            return messages, i
            break
        end
    end
    return messages, nothing
end

function phi_factor(tn::Dict, g::NamedGraph, messages::Dict, v)
    incoming_messages = [messages[e] for e in boundary_edges(g, [v]; dir = :in)]
    return contract_network([[tn[v]]; incoming_messages])[]
end

"""
    phi_cluster_correction(tn::Dict, g::NamedGraph, messages::Dict; smallest_loop_size::Int)

Computes the first order correction to the Bethe-Peierls free energy.

# Arguments
- `tn::Dict`: A dictionary representing the tensor network, where keys are vertices and values are tensors.
- `g::NamedGraph`: The named graph representing the structure of the tensor network.
- `messages::Dict`: A dictionary containing the messages for each edge in the graph.

# Keywords
- `smallest_loop_size::Int`: The size of the smallest loops to consider for the correction.

# Returns
- `phi::Number`: The corrected free energy estimate per vertex.
"""
function phi_cluster_correction(tn::Dict, g::NamedGraph, messages::Dict; smallest_loop_size::Int)
    messages = binormalized_messages(g, messages)
    rescaled_tn = Dict(v => tn[v] / phi_factor(tn, g, messages, v) for v in vertices(g))
    cycles = filter(c -> length(c) > 2, simplecycles_limited_length(g, smallest_loop_size))
    cycles = unique(Set.(cycles))
    isempty(cycles) && error("No cycles found with length $smallest_loop_size")
    cycle_weights = []
    for cycle in cycles
        incoming_messages = [messages[e] for e in boundary_edges(g, cycle; dir = :in)]
        local_tensors = [rescaled_tn[v] for v in cycle]
        weight = contract_network([local_tensors; incoming_messages])[]
        push!(cycle_weights, weight)
    end
    return sum(log.(cycle_weights)) / length(vertices(g))
end
