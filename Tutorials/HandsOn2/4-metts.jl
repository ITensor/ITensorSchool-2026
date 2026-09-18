# ITensorMPS itself is loaded so that `ITensorMPS.sample!` can stand in for your `sample_state`
using ITensorMPS: ITensorMPS, MPS, MPO, OpSum, dmrg, maxlinkdim, random_mps, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner
# Functions for time evolution
using ITensorMPS: apply, op
using LinearAlgebra: norm, normalize
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
using Statistics: mean
# Load the Plots package for plotting
using Plots: Plots, plot

using Printf: @printf

# Load the `tebd` function from the TEBD implementation tutorial
include("1-tebd-implementation.jl")
# Load the `sample_state` function from the sampling tutorial
include("3-sample-implementation.jl")

"""
    mean_and_sem(v::Vector)

Given a Vector of numbers, returns
the mean (average) and the standard error
of the mean (= the width of distribution of the numbers).
"""
function mean_and_sem(v::Vector)
    mn = mean(v)
    # Summing the squared deviations rather than ⟨v²⟩ - ⟨v⟩² keeps this from going slightly
    # negative, and taking the square root of a negative number, when every entry is the same
    return mn, √(sum(abs2, v .- mn)) / length(v)
end

"""
   main(; kwargs...)

Perform METTS (minimally entangled typical thermal states) algorithm on a 1D 
Heisenberg spin-1/2 chain to compute thermal expectation values at finite temperature.

# Keywords
- `nsite::Int = 10`: Number of sites in the spin chain.
- `beta::Float64 = 4.0`: Inverse temperature β = 1/T.
- `betastep::Float64 = 0.1`: Time step for imaginary time evolution.
- `cutoff::Float64 = 1.0e-8`: Cutoff for truncation during imaginary time evolution.
- `NMETTS::Int = 100`: Number of METTS samples to generate for averaging.
- `Nwarm::Int = 10`: Number of warmup METTS to generate before collecting measurements.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `H::MPO`: The Hamiltonian as an MPO.
- `psi::MPS`: The final METTS state.
- `betas::StepRangeLen`: Vector of β/2 time points used in imaginary time evolution.
- `energies::Vector{Float64}`: Vector of energy measurements from each METTS sample.
- `energy_dmrg::Float64`: Ground state energy computed by DMRG for reference.
- `nsite::Int`: Same as above.
- `beta::Float64`: Same as above.
- `betastep::Float64`: Same as above.
- `cutoff::Float64`: Same as above.
- `NMETTS::Int`: Same as above.
- `Nwarm::Int`: Same as above.
"""
function main(;
        # Number of sites
        nsite = 10,
        # TEBD parameters
        beta = 4.0,
        betastep = 0.1,
        cutoff = 1.0e-8,
        # METTS parameters
        NMETTS = 100,
        Nwarm = 10,
        outputlevel = 1,
    )
    # Build the physical indices for nsite spins (spin 1/2)
    sites = siteinds("S=1/2", nsite)

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

    # Make H for measuring the energy
    terms = OpSum()
    for j in 1:(nsite - 1)
        terms += 1 / 2, "S+", j, "S-", j + 1
        terms += 1 / 2, "S-", j, "S+", j + 1
        terms += "Sz", j, "Sz", j + 1
    end
    H = MPO(terms, sites)

    # Compute the DMRG energy as a reference.
    energy_dmrg, _ = dmrg(
        H, random_mps(sites; linkdims = 10); nsweeps = 5, maxdim = [10, 20, 100, 100, 200],
        cutoff = [1.0e-10], outputlevel = 0
    )

    # Make starting state
    rng = StableRNG(123)
    psi = random_mps(rng, sites)

    # Make y-rotation gates to use in METTS collapses
    Ry_gates = [op("Ry", sites[j]; θ = π / 2) for j in 1:nsite]

    betas = betastep:betastep:(beta / 2)
    if norm(length(betas) * betastep - beta / 2) > 1.0e-10
        error("Time step betastep=$betastep not commensurate with beta/2=$(beta / 2)")
    end

    energies = Float64[]
    print_every = 10
    for step in 1:(Nwarm + NMETTS)
        if outputlevel > 0 && step % print_every == 0
            if step <= Nwarm
                println("Making warmup METTS number $step")
            else
                println("Making METTS number $(step - Nwarm)")
            end
        end

        # Do the time evolution by applying the gates
        for _ in betas
            psi = normalize(tebd(gates, psi; cutoff))
        end

        # Measure properties after >= Nwarm
        # METTS have been made
        if step > Nwarm
            energy = inner(psi', H, psi)
            push!(energies, energy)
            if outputlevel > 0 && step % print_every == 0
                @printf("  Energy of METTS %d = %.4f\n", step - Nwarm, energy)
                @printf("  Energy of ground state from DMRG %.4f\n", energy_dmrg)
                mean_energy, sem_energy = mean_and_sem(energies)
                @printf(
                    "  Estimated Energy = %.4f +- %.4f  [%.4f,%.4f]\n",
                    mean_energy,
                    sem_energy,
                    mean_energy - sem_energy,
                    mean_energy + sem_energy
                )
            end
        end

        # Measure in X or Z basis on alternating steps
        if step % 2 == 1
            psi = apply(Ry_gates, psi)
            samp = sample_state(rng, psi)
            state = [samp[j] == 1 ? "X+" : "X-" for j in 1:nsite]
        else
            samp = sample_state(rng, psi)
            state = [samp[j] == 1 ? "Z+" : "Z-" for j in 1:nsite]
        end
        if outputlevel > 0 && step % print_every == 0
            println("  Sampled state: ", state)
        end
        psi = MPS(sites, state)
    end

    return (;
        H, psi, betas, energies, energy_dmrg, nsite, beta, betastep, cutoff,
        NMETTS, Nwarm,
    )
end
