using ITensorMPS: MPS, MPO, OpSum, dmrg, maxlinkdim, random_mps, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner
# Functions for time evolution
using ITensorMPS: apply, op
using LinearAlgebra: normalize
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
# Load the Plots package for plotting
using Plots: Plots, plot

include("../src/animate.jl")

function plot_tebd_sz(res; step::Int)
    return plot(
        res.szs[step]; xlim = (1, res.nsite), ylim = (-0.5, 0.5), xlabel = "Site j",
        ylabel = "⟨Szⱼ(β=$(res.betas[step]))⟩", legend = false
    )
end

function animate_tebd_sz(res; fps = res.nsite)
    return animate(i -> plot_tebd_sz(res; step = i); nframes = length(res.szs), fps)
end

"""
   main(; kwargs...)

Perform time-evolving block decimation (TEBD) on a 1D Heisenberg spin-1/2 chain to perform
imaginary time evolution to find the ground state.

# Keywords
- `nsite::Int = 30`: Number of sites in the spin chain.
- `beta::Float64 = 20.0`: Total inverse temperature for evolution.
- `betastep::Float64 = 0.2`: Time step for each TEBD application.
- `cutoff::Float64 = 1.0e-10`: Cutoff for truncation during TEBD.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `H::MPO`: The Hamiltonian as an MPO.
- `psit::MPS`: The final wavefunction after time evolution as an MPS.
- `psi_ground_state::MPS`: The ground state from DMRG.
- `energy_ground_state::Float64`: The ground state energy from DMRG.
- `betas::Vector{Float64}`: Vector of imaginary time points at which measurements were taken.
- `szs::Vector{Vector{Float64}}`: Vector of ⟨Sz⟩ measurements at each time point.
- `energies::Vector{Float64}`: Vector of energy measurements at each time point.
- `nsite::Int`: Same as above.
- `beta::Float64`: Same as above.
- `betastep::Float64`: Same as above.
- `cutoff::Float64`: Same as above.
"""
function main(;
        # Number of sites
        nsite = 30,
        # TEBD parameters
        beta = 20.0,
        betastep = 0.2,
        cutoff = 1.0e-10,
        outputlevel = 1,
    )
    # Build the physical indices for nsite spins (spin 1/2)
    sites = siteinds("S=1/2", nsite)

    terms = OpSum()
    for j in 1:(nsite - 1)
        terms += 1 / 2, "S+", j, "S-", j + 1
        terms += 1 / 2, "S-", j, "S+", j + 1
        terms += "Sz", j, "Sz", j + 1
    end
    H = MPO(terms, sites)

    if outputlevel > 0
        println("Run DMRG to get a reference energy for imaginary time evolution")
    end
    psi0 = random_mps(sites; linkdims = 10)
    energy_ground_state, psi_ground_state = dmrg(
        H, psi0; nsweeps = 5, maxdim = [10, 20, 100, 100, 200],
        cutoff = [1.0e-10], outputlevel = min(outputlevel, 1)
    )

    # Make gates (1, 2), (2, 3), (3, 4), ...
    function gate(j)
        si, sj = sites[j], sites[j + 1]
        hj = 1 / 2 * op("S+", si) * op("S-", sj) +
            1 / 2 * op("S-", si) * op("S+", sj) +
            op("Sz", si) * op("Sz", sj)
        return exp(-betastep / 2 * hj)
    end
    gates = [gate(j) for j in 1:(nsite - 1)]
    # Include gates in reverse order too
    # (N, N - 1), (N - 1, N - 2), ...
    append!(gates, reverse(gates))

    # Make starting state
    rng = StableRNG(123)
    psit = random_mps(rng, sites)

    if outputlevel > 0
        println("\nStarting imaginary time evolution")
    end
    szs = [expect(psit, "Sz")]
    energies = [inner(psit', H, psit)]
    betas = 0.0:betastep:beta
    print_every = 5
    for current_beta in betas[2:end]
        psit = normalize(apply(gates, psit; cutoff))
        energy_t = inner(psit', H, psit)
        sz_t = expect(psit, "Sz")
        push!(szs, sz_t)
        push!(energies, energy_t)
        if floor(current_beta - betastep + 10eps()) ≠ floor(current_beta) &&
                floor(current_beta) % print_every == 0
            if outputlevel > 0
                println("beta: ", current_beta)
                println("Bond dimension: ", maxlinkdim(psit))
                println("⟨ψₜ|Szⱼ|ψₜ⟩: ", sz_t[nsite ÷ 2])
                println("∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: ", sum(sz_t))
                println("⟨ψₜ|H|ψₜ⟩: ", energy_t)
                println()
            end
        end
    end

    res = (;
        H, psit, psi_ground_state, energy_ground_state, betas, szs, energies, nsite, beta,
        betastep, cutoff,
    )
    if outputlevel > 1
        animate_tebd_sz(res)
    end
    return res
end
