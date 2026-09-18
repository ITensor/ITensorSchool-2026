using Graphs: vertices, edges, src, dst, neighbors, nv, simplecycles_limited_length
using NamedGraphs: NamedEdge, NamedGraph
using NamedGraphs: all_edges, boundary_edges
using LinearAlgebra: dot, normalize
using Statistics: mean
using ITensors: ITensor, Index, inds, onehot

include("../../Tutorials/HandsOn4/contract_network.jl")

"""
Tutorial 2:

You are asked to complete the implementation of the belief propagation (BP)
algorithm in this file. For each numbered step (1), (2), (3) below, fill in the
missing code.

This is the completed solution.
"""

"""
    updated_message(tn::Dict, g::NamedGraph, messages::Dict, e::NamedEdge)

Compute the new message to send along the directed edge `e`, from `src(e)` to `dst(e)`.
"""
function updated_message(tn::Dict, g::NamedGraph, messages::Dict, e::NamedEdge)
    # The directed edges of `g` pointing into `src(e)`, excluding the edge coming from `dst(e)`
    incoming_es = setdiff(boundary_edges(g, [src(e)]; dir = :in), [reverse(e)])

    # (1) Contract the tensor `tn[src(e)]` with the messages living on `incoming_es`
    #     and normalize the result. `contract_network` accepts a vector of tensors.
    local_tensor = tn[src(e)]
    incoming_messages = [messages[e_in] for e_in in incoming_es]
    return normalize(contract_network([[local_tensor]; incoming_messages]))
end

"""
    update_messages(tn::Dict, g::NamedGraph, messages::Dict)

Compute a new message on every directed edge of `g`, using the current `messages`.
"""
function update_messages(tn::Dict, g::NamedGraph, messages::Dict)
    updated_messages = copy(messages)
    for e in all_edges(g)
        updated_messages[e] = updated_message(tn, g, messages, e)
    end
    return updated_messages
end

"""
    message_distance(g::NamedGraph, messages::Dict, old_messages::Dict)

A measure of how much the messages changed in the last update. It is zero when every
new message is parallel to the corresponding old message.
"""
function message_distance(g::NamedGraph, messages::Dict, old_messages::Dict)
    # (2) For each directed edge `e` in `all_edges(g)`, compute 1 - dot(m_new, m_old)^2
    #     where `m_new = messages[e]` and `m_old = old_messages[e]` are both normalized.
    #     Return the mean of these numbers over all directed edges.
    return mean([1 - dot(messages[e], old_messages[e])^2 for e in all_edges(g)])
end

"""
    initial_messages(tn::Dict, g::NamedGraph)

Initial messages: a `onehot` vector on the index shared by the two tensors of each edge.
"""
function initial_messages(tn::Dict, g::NamedGraph)
    linkind(tn, e) = only(intersect(inds(tn[src(e)]), inds(tn[dst(e)])))
    return Dict(e => onehot(linkind(tn, e) => 1) for e in all_edges(g))
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
        old_messages = messages
        messages = update_messages(tn, g, messages)
        if message_distance(g, messages, old_messages) < tol
            outputlevel >= 1 && println("BP Algorithm Converged after $i iterations")
            return messages, i
        end
    end
    outputlevel >= 1 && println("BP Algorithm did NOT converge after $niters iterations")
    return messages, nothing
end

"""
    phi_factor(tn::Dict, g::NamedGraph, messages::Dict, v)

The scalar obtained by contracting the tensor on vertex `v` with all of its incoming messages.
"""
function phi_factor(tn::Dict, g::NamedGraph, messages::Dict, v)
    # (3) Gather the messages on every directed edge pointing into `v`, contract them with
    #     `tn[v]` and return the resulting scalar (use `[]` to extract it from the ITensor).
    incoming_messages = [messages[e] for e in boundary_edges(g, [v]; dir = :in)]
    return contract_network([[tn[v]]; incoming_messages])[]
end

"""
    binormalized_messages(g::NamedGraph, messages::Dict)

Rescale the messages so that the pair of messages on each edge contract to one.
"""
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

Computes the Bethe-Peierls free energy density `ϕ = log(Z_BP) / nv(g)`.

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
    phi_cluster_correction(tn::Dict, g::NamedGraph, messages::Dict; smallest_loop_size::Int)

Computes the first order correction to the Bethe-Peierls free energy.

# Arguments
- `tn::Dict`: A dictionary representing the tensor network, where keys are vertices and values are tensors.
- `g::NamedGraph`: The named graph representing the structure of the tensor network.
- `messages::Dict`: A dictionary containing the messages for each edge in the graph.

# Keywords
- `smallest_loop_size::Int`: The size of the smallest loops to consider for the correction.

# Returns
- `phi::Number`: The correction to the free energy estimate per vertex.
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
    return sum(log.(cycle_weights)) / nv(g)
end
