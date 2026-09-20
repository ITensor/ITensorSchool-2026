# ITensorMPS itself is loaded so that its own `ITensorMPS.sample!` is available to compare
# against
using ITensorMPS: ITensorMPS, MPS, random_mps, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: dag, expect, linkinds
# Functions for building the tensors used when sampling
using ITensors: ITensor, dim, onehot, prime, scalar
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

Sample one product state from the probability distribution |⟨state|ψ⟩|² defined by the MPS
`psi`, returned as a vector holding the sampled state on each site.

The sites are sampled in a sweep from left to right, each one conditioned on the states
already sampled to its left, so the result is a sample of |⟨state|ψ⟩|².
"""
function sample_state(rng::AbstractRNG, psi::MPS)

    @warn "`sample_state` is not implemented yet, so it will always return the same state. Fill in the steps below." maxlog = 1

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
    # sampled so far. The matching part of psid is its conjugate, with primed links. L starts
    # out trivial and grows one site at a time.
    L = ITensor(1.0)
    # Your implementation should overwrite this state with a sample from the MPS. Until you
    # implement that, this function will return this trivial all up product state every time.
    result = ones(Int, nsite)
    for j in 1:nsite
        s = sites[j]

        # (1)
        # For each state n of site j, project the tensor of psi on that site onto the state
        # with `onehot(s => n)` and contract it with L
        #
        # Ls = [... for n in 1:dim(s)]
        #

        #TODO remove
        Ls = [L * (psi[j] * onehot(s => n)) for n in 1:dim(s)]

        # (2)
        # Closing each of those, along with its counterpart from psid, against the right
        # environment Rs[j + 1] gives a number, the probability of sampling that state of site j
        # given the states already sampled. Compute those probabilities, normalize them, and
        # sample a state n from them.
        #
        # probabilities = [... for Ln in Ls]
        # n = ...
        #
        n = 1

        #TODO remove
        probabilities = [real(scalar(Ln * Rs[j + 1] * dag(prime(Ln)))) for Ln in Ls]
        probabilities /= sum(probabilities)
        n = searchsortedfirst(cumsum(probabilities), rand(rng))

        # (3)
        # Record the sampled state. The tensor you built for it in step (1) is the L for
        # the next site.
        #
        # result[j] = ...
        # L = ...
        #

        #TODO remove
        result[j] = n
        L = Ls[n]
    end
    return result
end

sample_state(psi::MPS) = sample_state(default_rng(), psi)

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
- `rng::AbstractRNG = default_rng()`: Random number generator. Pass a seeded one, such as
  `StableRNG(1234)`, to get the same samples every run.
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
function main(; nsite = 20, nsample = 2000, linkdim = 4, rng = default_rng(), outputlevel = 1)
    sites = siteinds("S=1/2", nsite)
    psi = normalize(random_mps(rng, sites; linkdims = linkdim))

    states = [sample_state(rng, psi) for _ in 1:nsample]
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
