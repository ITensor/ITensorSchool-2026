using ITensors: ITensors, ITensor, apply, commoninds, noprime, uniqueinds
using ITensorMPS: MPS, maxlinkdim, normalize!, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner, orthogonalize
# Functions for time evolution (for checking)
using ITensorMPS: op
using LinearAlgebra: norm, normalize, svd
# Load the Plots package for plotting
using Plots: Plots, plot, plot!

include("resources/reference_tebd.jl")

"""
    tebd_step(gate::ITensor, A::ITensor, B::ITensor; truncation_keyword_args...)

Apply `gate` to the neighboring MPS tensors `A` and `B` and split the result back into two
tensors, returned as a tuple. Keyword arguments such as `cutoff` and `maxdim` are passed on
to `svd` to truncate the bond between them.

This is the exercise of this tutorial: until you fill it in, `A` and `B` come back unchanged.
"""
function tebd_step(gate::ITensor, A::ITensor, B::ITensor; truncation_keyword_args...)

    @warn "`tebd_step` is not implemented yet, so the state will not evolve. Fill in the steps below." maxlog = 1

    # (1)
    # Add code to apply the gate to the portion of the MPS
    # consisting of tensors A and B
    #
    # gate_AB = ...
    #

    # (2)
    # Use the ITensor `svd` function to factorize and truncate
    # resulting tensor to restore an internal low-rank
    # bond structure
    #
    #                Determine correct arguments...
    #                | 
    #                v 
    # U, S, V = svd(...; truncation_keyword_args...)

    # (3) 
    # Use the results of `svd` to assemble new MPS tensors
    #
    # A = ...
    # B = ...

    return A, B
end

"""
    tebd(gates, psi::MPS; truncation_keyword_args...)

Evolve `psi` by one time step with your `tebd_step`, sweeping left to right and back again.
`gates` must be ordered to match that sweep, one gate per bond forwards followed by one per
bond backwards, which is how `make_heisenberg_gates` returns them. Keyword arguments are
passed on to `tebd_step`.
"""
function tebd(gates, psi::MPS; truncation_keyword_args...)
    psi = copy(psi)
    nsite = length(psi)
    fwd_sweep = [j => (j + 1) for j in 1:(nsite - 1)]
    rev_sweep = [j => (j - 1) for j in reverse(2:nsite)]
    bonds = vcat(fwd_sweep, rev_sweep)
    for (bond, gate) in zip(bonds, gates)
        s1, s2 = bond
        psi = orthogonalize(psi, s1)
        psi[s1], psi[s2] = tebd_step(gate, psi[s1], psi[s2]; truncation_keyword_args...)
    end
    return psi
end

"""
    make_heisenberg_gates(sites, step)

Build the gates for one TEBD step of the 1D Heisenberg model on `sites`, ordered for the
forward and backward sweep `tebd` makes. Each gate is `exp(step / 2 * hj)` for its bond term
`hj`, half a step because every bond is visited twice per sweep.

Real time evolution by `timestep` passes `step = -im * timestep`, imaginary time evolution by
`betastep` passes `step = -betastep`.
"""
function make_heisenberg_gates(sites, step)
    # Make gates (1, 2), (2, 3), (3, 4), ...
    gates = map(1:(length(sites) - 1)) do j
        si, sj = sites[j], sites[j + 1]
        hj = 1 / 2 * op("S+", si) * op("S-", sj) +
            1 / 2 * op("S-", si) * op("S+", sj) +
            op("Sz", si) * op("Sz", sj)
        return exp(step / 2 * hj)
    end
    # Include gates in reverse order too
    # (N, N - 1), (N - 1, N - 2), ...
    append!(gates, reverse(gates))
    return gates
end

"""
    main(; kwargs...)

Evolve a Néel state under the 1D Heisenberg Hamiltonian with your `tebd` and compare ⟨Szⱼ⟩
against a reference implementation at every time step.

# Keywords
- `nsite::Int = 20`: Number of sites in the spin chain.
- `time::Float64 = 5.0`: Total time for evolution.
- `timestep::Float64 = 0.1`: Time step for each TEBD application.
- `cutoff::Float64 = 1.0e-8`: Cutoff for truncation during TEBD.
- `maxdim::Int = typemax(Int)`: Largest bond dimension kept during TEBD.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `times::Vector{Float64}`: Vector of time points at which measurements were taken.
- `szs::Vector{Vector{Float64}}`: Vector of ⟨Szⱼ⟩ from your `tebd` at each time point.
- `szs_reference::Vector{Vector{Float64}}`: The same from the reference, for checking.
- `nsite::Int`: Same as above.
- `time::Float64`: Same as above.
- `timestep::Float64`: Same as above.
- `cutoff::Float64`: Same as above.
- `maxdim::Int`: Same as above.
"""
function main(;
        nsite = 20,
        cutoff = 1.0e-8,
        maxdim = typemax(Int),
        timestep = 0.1,
        time = 5.0,
        outputlevel = 1,
    )
    # Make an array of 'site' indices
    sites = siteinds("S=1/2", nsite)

    # Obtain time-evolution gates / circuit
    gates = make_heisenberg_gates(sites, -im * timestep)

    # Initialize psi to be a product state (alternating up and down)
    psi_init = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    psi = copy(psi_init)

    # Compute and record <Sz> on every site at each time step
    # then apply the gates to go to the next time
    times = 0.0:timestep:time
    szs = Vector{Float64}[]
    for t in times
        sz = expect(psi, "Sz")
        push!(szs, sz)
        if outputlevel > 0
            println("t = $t  ⟨Sz_$(nsite ÷ 2)⟩ = $(sz[nsite ÷ 2])")
        end
        t ≈ time && break
        # Evolve psi with your implementation of tebd
        psi = tebd(gates, psi; cutoff, maxdim)
        normalize!(psi)
    end

    # Run the reference TEBD calculation for comparison
    szs_reference = reference_tebd(gates, times, psi_init; cutoff, maxdim)

    res = (; times = collect(times), szs, szs_reference, nsite, time, timestep, cutoff, maxdim)

    max_diff = maximum(maximum(abs, sz - sz_ref) for (sz, sz_ref) in zip(szs, szs_reference))
    if max_diff < 1.0e-6
        if outputlevel > 0
            @info "Your ⟨Szⱼ⟩ values agree with the reference (maximum difference = $max_diff)"
        end
    else
        @warn "Your ⟨Szⱼ⟩ values DO NOT agree with the reference (maximum difference = $max_diff)"
    end
    if outputlevel > 0
        # Plot results at the final time
        display(plot_szs(res))
    end
    return res
end

# Plotting functions

"""
    plot_szs(res::NamedTuple; step::Int = length(res.times))

Plot ⟨Szⱼ⟩ on each site j at time step `step` (by default the final time) from your `tebd`
implementation (solid blue line, filled markers) on top of the reference calculation (dashed
green line, open markers). `res` is expected to be a `NamedTuple` with fields `times`, `szs`,
`szs_reference`, and `nsite`, such as the results of `main`.
"""
function plot_szs(res::NamedTuple; step::Int = length(res.times))
    (; times, szs, szs_reference, nsite) = res
    p = plot(
        1:nsite, szs_reference[step];
        label = "correct reference", color = :green, linestyle = :dash,
        marker = :circle, markersize = 6, markercolor = :white, markerstrokecolor = :green,
        xlim = (1, nsite), ylim = (-0.5, 0.5), xlabel = "Site j", ylabel = "⟨Szⱼ⟩",
        title = "t = $(times[step])",
    )
    plot!(
        p, 1:nsite, szs[step];
        label = "your tebd", color = :blue, linestyle = :solid,
        marker = :circle, markersize = 4, markercolor = :blue, markerstrokecolor = :blue,
    )
    return p
end
