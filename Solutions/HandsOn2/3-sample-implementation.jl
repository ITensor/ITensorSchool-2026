using ITensorMPS: MPS, random_mps, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: dag, expect, sim_linkinds
# Functions for building the tensors used when sampling
using ITensors: ITensor, dim, onehot, scalar
using LinearAlgebra: normalize
using Random: AbstractRNG, default_rng
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
using Statistics: mean
# Load the Plots package for plotting
using Plots: Plots, plot, plot!

"""
    sample_state(rng::AbstractRNG, psi::MPS)
    sample_state(psi::MPS)

Draw one product state from the probability distribution |⟨state|ψ⟩|² defined by the MPS
`psi`, returned as a vector holding the state that was drawn on each site.

Each site is drawn in turn, conditioned on the states already drawn for the sites to its
left, which is what makes the states that come out independent samples of |⟨state|ψ⟩|².
"""
function sample_state(rng::AbstractRNG, psi::MPS)
    nsite = length(psi)
    sites = siteinds(psi)

    # psid is a copy of psi with new internal link indices and with all of the tensors
    # conjugated, so psi and psid together form the norm network ⟨ψ|ψ⟩
    psid = dag(sim_linkinds(psi))

    # Rs[j] is the part of the norm network from site j to the end of the chain contracted
    # together, so Rs[nsite + 1] is a trivial scalar environment
    Rs = Vector{ITensor}(undef, nsite + 1)
    Rs[nsite + 1] = ITensor(1.0)
    for j in reverse(1:nsite)
        Rs[j] = Rs[j + 1] * psid[j] * psi[j]
    end

    # L is the part of the norm network to the left of site j, projected onto the states
    # that have been drawn so far. It starts out trivial and grows one site at a time.
    L = ITensor(1.0)
    result = zeros(Int, nsite)
    for j in 1:nsite
        s = sites[j]

        # (1)
        Ls = [L * (psi[j] * onehot(s => n)) * (psid[j] * onehot(s => n)) for n in 1:dim(s)]

        # (2)
        probabilities = [real(scalar(Ln * Rs[j + 1])) for Ln in Ls]
        probabilities /= sum(probabilities)
        n = searchsortedfirst(cumsum(probabilities), rand(rng))

        # (3)
        result[j] = n
        L = Ls[n]
    end
    return result
end

sample_state(psi::MPS) = sample_state(default_rng(), psi)

# Answers to the follow-up questions in the Hands-On 2 README

"""
    sample_state(rng::AbstractRNG, psi::MPS, psid::MPS, Rs::Vector{ITensor})

Draw one product state from |⟨state|ψ⟩|² using right environments `Rs` that were contracted
from `psi` and `psid` beforehand.

`psid` is passed along with `Rs` so that the link indices match up properly.
"""
function sample_state(rng::AbstractRNG, psi::MPS, psid::MPS, Rs::Vector{ITensor})
    nsite = length(psi)
    sites = siteinds(psi)
    L = ITensor(1.0)
    result = zeros(Int, nsite)
    for j in 1:nsite
        s = sites[j]
        # Closing L with Rs[j] gives the weight of all states of site j added together,
        # which is what the probabilities below would be normalized by, so the state can be
        # drawn against a running sum and the states after it never have to be computed
        r = rand(rng) * real(scalar(L * Rs[j]))
        cumulative = 0.0
        n = dim(s)
        Ln = L
        for m in 1:dim(s)
            n = m
            Ln = L * (psi[j] * onehot(s => m)) * (psid[j] * onehot(s => m))
            # Whatever weight is left over belongs to the last state, so it is drawn
            # without closing its environment at all
            m == dim(s) && break
            cumulative += real(scalar(Ln * Rs[j + 1]))
            cumulative > r && break
        end
        result[j] = n
        L = Ln
    end
    return result
end

"""
    sample_states(rng::AbstractRNG, psi::MPS, nsample::Int)

Draw `nsample` product states from |⟨state|ψ⟩|².

The right environments do not depend on the states that are drawn, so they are contracted
once and reused for every sample.
"""
function sample_states(rng::AbstractRNG, psi::MPS, nsample::Int)
    nsite = length(psi)
    psid = dag(sim_linkinds(psi))
    Rs = Vector{ITensor}(undef, nsite + 1)
    Rs[nsite + 1] = ITensor(1.0)
    for j in reverse(1:nsite)
        Rs[j] = Rs[j + 1] * psid[j] * psi[j]
    end
    return [sample_state(rng, psi, psid, Rs) for _ in 1:nsample]
end

"""
    sampled_sz(states::Vector{Vector{Int}})

Average ⟨Szⱼ⟩ on each site j over a collection of sampled product states, where state 1 of a
spin-1/2 site is up and state 2 is down.
"""
function sampled_sz(states::Vector{Vector{Int}})
    return [mean(state[j] == 1 ? 1 / 2 : -1 / 2 for state in states) for j in 1:length(first(states))]
end

"""
    main(; kwargs...)

Draw many product states from a random MPS with your `sample_state` function and compare the
magnetization they give against `expect`.

# Keywords
- `nsite::Int = 20`: Number of sites in the spin chain.
- `nsample::Int = 2000`: Number of product states to draw.
- `linkdim::Int = 4`: Bond dimension of the random MPS that is sampled.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `psi::MPS`: The MPS that was sampled.
- `states::Vector{Vector{Int}}`: The product states that were drawn.
- `sz::Vector{Float64}`: Vector of ⟨Szⱼ⟩ from your samples.
- `sz_reference::Vector{Float64}`: Vector of ⟨Szⱼ⟩ from `expect`, for checking.
- `nsite::Int`: Same as above.
- `nsample::Int`: Same as above.
- `linkdim::Int`: Same as above.
"""
function main(; nsite = 20, nsample = 2000, linkdim = 4, outputlevel = 1)
    rng = StableRNG(1234)
    sites = siteinds("S=1/2", nsite)
    psi = normalize(random_mps(rng, sites; linkdims = linkdim))

    states = sample_states(rng, psi, nsample)
    sz = sampled_sz(states)
    # Sampling ⟨Szⱼ⟩ is a Monte Carlo estimate, so it only agrees with `expect` to within
    # the statistical error of the mean of nsample draws
    sz_reference = expect(psi, "Sz")
    max_diff = maximum(abs, sz - sz_reference)
    tolerance = 5 * 1 / 2 / sqrt(nsample)
    if max_diff < tolerance
        if outputlevel > 0
            @info "Your sampled ⟨Szⱼ⟩ values agree with `expect` (maximum difference = $max_diff)"
        end
    else
        @warn "Your sampled ⟨Szⱼ⟩ values DO NOT agree with `expect` (maximum difference = $max_diff)"
    end

    res = (; psi, states, sz, sz_reference, nsite, nsample, linkdim)
    if outputlevel > 0
        display(plot_sampled_sz(res))
    end
    return res
end

# Plotting functions

"""
    plot_sampled_sz(res::NamedTuple)

Plot ⟨Szⱼ⟩ on each site j averaged over the states drawn by your `sample_state` function
(solid blue line, filled markers) on top of the values from `expect` (dashed green line, open
markers). `res` is expected to be a `NamedTuple` with fields `sz`, `sz_reference`, and
`nsite`, such as the results of `main`.
"""
function plot_sampled_sz(res::NamedTuple)
    (; sz, sz_reference, nsite) = res
    p = plot(
        1:nsite, sz_reference;
        label = "reference (expect)", color = :green, linestyle = :dash,
        marker = :circle, markersize = 6, markercolor = :white, markerstrokecolor = :green,
        xlim = (1, nsite), ylim = (-0.5, 0.5), xlabel = "Site j", ylabel = "⟨Szⱼ⟩",
    )
    plot!(
        p, 1:nsite, sz;
        label = "your sample", color = :blue, linestyle = :solid,
        marker = :circle, markersize = 4, markercolor = :blue, markerstrokecolor = :blue,
    )
    return p
end
