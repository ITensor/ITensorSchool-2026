using ITensorMPS: MPS, random_mps, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: dag, expect, linkinds
# Functions for building the tensors used when sampling
using ITensors: ITensor, dim, onehot, prime, scalar
using LinearAlgebra: normalize
using Statistics: mean
# Load the Plots package for plotting
using Plots: Plots, plot, plot!

"""
    sample_state(psi::MPS)

Sample one product state from the probability distribution |⟨state|ψ⟩|² defined by the MPS
`psi`, returned as a vector holding the sampled state on each site.

The sites are sampled in a sweep from left to right, each one conditioned on the states
already sampled to its left, so the result is a sample of |⟨state|ψ⟩|².
"""
function sample_state(psi::MPS)
    nsite = length(psi)
    sites = siteinds(psi)

    # psid is a copy of psi with primed link indices and with all of the tensors conjugated,
    # so psi and psid together form the norm network ⟨ψ|ψ⟩
    psid = dag(prime(linkinds, psi))

    # Rs[j] is the part of the norm network from site j to the end of the chain contracted
    # together, so Rs[nsite + 1] is a trivial scalar environment
    Rs = Vector{ITensor}(undef, nsite + 1)
    Rs[nsite + 1] = ITensor(1.0)
    for j in reverse(1:nsite)
        Rs[j] = Rs[j + 1] * psid[j] * psi[j]
    end

    # L is the part of psi to the left of site j, projected onto the states that have been
    # sampled so far, so `dag(prime(L))` is the matching part of psid. It starts out trivial
    # and grows one site at a time.
    L = ITensor(1.0)
    result = zeros(Int, nsite)
    for j in 1:nsite
        s = sites[j]

        # (1)
        Ls = [L * (psi[j] * onehot(s => n)) for n in 1:dim(s)]

        # (2)
        probabilities = [real(scalar(Ls[n] * Rs[j + 1] * dag(prime(Ls[n])))) for n in 1:dim(s)]
        probabilities /= sum(probabilities)
        n = searchsortedfirst(cumsum(probabilities), rand())

        # (3)
        result[j] = n
        L = Ls[n]
    end
    return result
end

# Answers to the follow-up questions in the Hands-On 2 README

"""
    sample_state(psi::MPS, psid::MPS, Rs::Vector{ITensor})

Sample one product state from |⟨state|ψ⟩|² using right environments `Rs` that were
contracted from `psi` and `psid` beforehand.

`psid` is taken as an argument rather than derived from `psi`, so that this works whatever
was done to keep the link indices of the two copies apart.
"""
function sample_state(psi::MPS, psid::MPS, Rs::Vector{ITensor})
    nsite = length(psi)
    sites = siteinds(psi)
    L = ITensor(1.0)
    Ld = ITensor(1.0)
    result = zeros(Int, nsite)
    for j in 1:nsite
        s = sites[j]
        # Closing L and Ld around Rs[j] gives the weight of all states of site j added
        # together, which is what the probabilities below would be normalized by, so the state
        # can be sampled against a running sum and the states after it never have to be computed
        r = rand() * real(scalar(L * Rs[j] * Ld))
        cumulative = 0.0
        n = dim(s)
        Ln = L
        Lnd = Ld
        for m in 1:dim(s)
            n = m
            Ln = L * (psi[j] * onehot(s => m))
            Lnd = Ld * (psid[j] * onehot(s => m))
            # Whatever weight is left over belongs to the last state, so it is sampled
            # without closing it against Rs at all
            m == dim(s) && break
            cumulative += real(scalar(Ln * Rs[j + 1] * Lnd))
            cumulative > r && break
        end
        result[j] = n
        L = Ln
        Ld = Lnd
    end
    return result
end

"""
    sample_states(psi::MPS, nsample::Int)

Sample `nsample` product states from |⟨state|ψ⟩|².

The right environments do not depend on the states that are sampled, so they are contracted
once and reused for every sample.
"""
function sample_states(psi::MPS, nsample::Int)
    nsite = length(psi)
    psid = dag(prime(linkinds, psi))
    Rs = Vector{ITensor}(undef, nsite + 1)
    Rs[nsite + 1] = ITensor(1.0)
    for j in reverse(1:nsite)
        Rs[j] = Rs[j + 1] * psid[j] * psi[j]
    end
    return [sample_state(psi, psid, Rs) for _ in 1:nsample]
end

"""
    sampled_sz(states::Vector{Vector{Int}})

Average ⟨Szⱼ⟩ on each site j over a collection of sampled product states. Assumes the sites
are spin-1/2.
"""
function sampled_sz(states::Vector{Vector{Int}})
    spin(n) = n == 1 ? 1 / 2 : -1 / 2
    return [mean(spin(state[j]) for state in states) for j in 1:length(first(states))]
end

"""
    main(; kwargs...)

Sample many product states from a random MPS with your `sample_state` function and compare
the magnetization they give against `expect`.

# Keywords
- `nsite::Int = 20`: Number of sites in the spin chain.
- `nsample::Int = 2000`: Number of product states to sample.
- `linkdim::Int = 4`: Bond dimension of the random MPS that is sampled.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `psi::MPS`: The MPS that was sampled.
- `states::Vector{Vector{Int}}`: The product states that were sampled.
- `sz::Vector{Float64}`: Vector of ⟨Szⱼ⟩ from your samples.
- `sz_reference::Vector{Float64}`: Vector of ⟨Szⱼ⟩ from `expect`, for checking.
- `nsite::Int`: Same as above.
- `nsample::Int`: Same as above.
- `linkdim::Int`: Same as above.
"""
function main(; nsite = 20, nsample = 2000, linkdim = 4, outputlevel = 1)
    sites = siteinds("S=1/2", nsite)
    psi = normalize(random_mps(sites; linkdims = linkdim))

    states = sample_states(psi, nsample)
    sz = sampled_sz(states)
    # Sampling ⟨Szⱼ⟩ is a Monte Carlo estimate, so it only agrees with `expect` to within
    # the statistical error of the mean of nsample samples
    sz_reference = expect(psi, "Sz")
    max_diff = maximum(abs, sz - sz_reference)
    # Each ⟨Szⱼ⟩ averages nsample samples of ±1/2, so its standard error is at most this
    standard_error = (1 / 2) / sqrt(nsample)
    tolerance = 5 * standard_error
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

Plot ⟨Szⱼ⟩ on each site j averaged over the states sampled by your `sample_state` function
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
