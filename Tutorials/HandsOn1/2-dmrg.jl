using ITensorMPS: MPO, OpSum, dmrg, maxlinkdim, random_mps, siteinds
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
# Load the Plots library for plotting results
using Plots: Plots, plot

"""
    main(; kwargs...)

Run the density matrix renormalization group (DMRG) algorithm to compute the ground state
energy and wavefunction of a 1D Heisenberg spin-1/2 chain.

This function constructs the spin chain Hamiltonian as a matrix product operator (MPO),
initializes a random matrix product state (MPS) as the starting point, and performs a series
of DMRG sweeps to optimize the ground state energy and wavefunction.

# Keywords
- `nsite::Int = 30`: Number of sites in the spin chain.
- `nsweeps::Int = 5`: Number of DMRG sweeps to perform.
- `maxdim::Vector{Int} = [10, 20, 100, 100, 200]`: Maximum bond dimensions for each sweep.
- `cutoff::Vector{Float64} = [1.0e-10]`: Cutoff for truncation.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `energy::Float64`: The optimized ground state energy.
- `H::MPO`: The Hamiltonian as an MPO.
- `psi::MPS`: The optimized ground state wavefunction as an MPS.
- `nsite::Int`: Same as above.
- `nsweeps::Int`: Same as above.
- `maxdim::Vector{Int}`: Same as above.
- `cutoff::Vector{Float64}`: Same as above.
"""
function main(;
        # Number of sites
        nsite = 30,
        # DMRG parameters
        nsweeps = 5,
        maxdim = [10, 20, 100, 100, 200],
        cutoff = [1.0e-10],
        outputlevel = 1,
    )

    if outputlevel > 0
        println("nsite: ", nsite)
        println("nsweeps: ", nsweeps)
        println("maxdim: ", maxdim)
        println("cutoff: ", cutoff)
    end

    # Build the physical indices for nsite spins (spin 1/2)
    sites = siteinds("S=1/2", nsite)

    # Build the Heisenberg Hamiltonian as an MPO
    terms = OpSum()
    for j in 1:(nsite - 1)
        terms += 1 / 2, "S+", j, "S-", j + 1
        terms += 1 / 2, "S-", j, "S+", j + 1
        terms += "Sz", j, "Sz", j + 1
    end
    H = MPO(terms, sites)

    # It has bond dimension 5
    if outputlevel > 0
        println("MPO bond dimension: ", maxlinkdim(H))
    end

    # Initial state for DMRG
    rng = StableRNG(123)
    psi0 = random_mps(rng, sites; linkdims = 10)

    # It starts with a bond dimension 10
    if outputlevel > 0
        println("Initial MPS bond dimension: ", maxlinkdim(psi0))
    end

    # Run DMRG
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff, outputlevel)

    if outputlevel > 0
        println("Optimized MPS bond dimension: ", maxlinkdim(psi))
        println("DMRG energy: ", energy)
    end

    return (; energy, H, psi, nsite, nsweeps, maxdim, cutoff)
end
