using LinearAlgebra: norm
using ITensors: ITensors, ITensor, apply, inds, pause
using ITensorMPS: MPO, MPS, OpSum, apply, dag, dmrg, maxlinkdim, op, random_mps, sim_linkinds, siteind, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: ITensorMPS, AbstractObserver, correlation_matrix, inner
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
# Load the Plots package for plotting
using Plots: Plots, plot

include("resources/animate.jl")

"""
Tutorial 3:

You are asked to complete the implementation of this
`expect` (expectation value) function.
For each numbered step (1),(2),(3) below, fill in
the missing code.
"""
function expect(psi::MPS, opname::String, j::Int)

    # psid is a copy of psi with internal link indices replaced
    # with new ones of the same dimension (different ids) and
    # with all tensor Hermitian-conjugated
    psid = dag(sim_linkinds(psi))

    L = ITensor(1.)
    # (1) Add a loop building the left environment or 'message' L
    # ...

    R = ITensor(1.)
    # (2) Add a loop building the right environment or 'message' R
    # ...

    O = op(opname,siteinds(psi)[j]) # operator ITensor sⱼ'--O--sⱼ

    # (3) Use `apply` to apply O to the jth tensor of the MPS
    #     psi and contract with the conjugated tensor
    #     then contract with L and R and obtain the scalar value
    value = 0.0

    return value
end

expect(psi::MPS, opname::String) = [expect(psi,opname,j) for j=1:length(psi)]


"""
    main(; kwargs...)

Perform DMRG on a Heisenberg spin-1 chain and measure ⟨Sz⟩.

# Keywords
- `nsite::Int = 30`: Number of sites in the spin chain.
- `nsweeps::Int = 5`: Number of DMRG sweeps.
- `maxdim::Vector{Int} = [10, 20, 100, 100, 200]`: Maximum bond dimensions for each sweep.
- `cutoff::Vector{Float64} = [1.0e-10]`: Cutoff values for each sweep.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Results
A named tuple containing:
- `energy::Float64`: The optimized ground state energy.
- `H::MPO`: The Hamiltonian as an MPO.
- `psi::MPS`: The optimized ground state wavefunction as an MPS.
- `sz::Vector{Float64}`: Vector of ⟨Sz⟩ measurements.
- `sz_frames::Vector{Vector{Float64}}`: Vector of ⟨Sz⟩ measurements at each DMRG step.
- `nsite::Int`: Same as above.
- `nsweeps::Int`: Same as above.
- `maxdim::Vector{Int}`: Same as above.
- `cutoff::Vector{Float64}`: Same as above.
"""
function main(;
        # Number of sites
        nsite = 40,
        # DMRG parameters
        nsweeps = 6,
        maxdim = [10, 20, 100, 100, 200],
        cutoff = [1.0e-10],
        outputlevel = 1,
    )
    if outputlevel > 0
        println("Number of sites: ", nsite)
    end
    # Build the physical indices for nsite spins
    sites = siteinds("S=1", nsite)

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
    psi0 = random_mps(sites; linkdims = 10)

    # It starts with a bond dimension 10
    if outputlevel > 0
        println("Initial MPS bond dimension: ", maxlinkdim(psi0))
    end

    # Run DMRG
    observer = SzObserver() # see later in this file for definition
    energy, psi = dmrg(
        H, psi0; nsweeps, maxdim, cutoff, observer, outputlevel = min(outputlevel, 1)
    )
    sz_frames = observer.szs

    if outputlevel > 0
        println("Optimized MPS bond dimension: ", maxlinkdim(psi))
        println("Energy: ", energy)
        println("⟨ψ|ψ⟩: ", inner(psi, psi))
        println("⟨ψ|H|ψ⟩: ", inner(psi', H, psi))
    end

    # Obtain expected ⟨Sᶻ⟩ from your own `expect` function (top of this file)
    sz = expect(psi, "Sz")

    # Use ITensorMPS.expect to check
    sz_correct = ITensorMPS.expect(psi,"Sz")
    error = norm(sz-sz_correct)
    if error < 1E-6
        println("Expected ⟨Sᶻ⟩ values agree with ITensorMPS.expect")
    else
        println("Expected ⟨Sᶻ⟩ values DO NOT agree with ITensorMPS.expect")
    end

    res = (; energy, H, psi, sz, sz_frames, nsite, nsweeps, maxdim, cutoff)
    if outputlevel > 0
        display(plot_dmrg_sz(res))
    end
    if outputlevel > 1
        animate_dmrg_sz(res)
    end
    return res
end

#
# SzObserver type for collecting measurements from DMRG
#

@kwdef struct SzObserver <: AbstractObserver
    szs = Vector{Float64}[]
end
function ITensorMPS.measure!(obs::SzObserver; psi, kwargs...)
    push!(obs.szs, ITensorMPS.expect(psi, "Sz"))
end

#
# Plotting and animation functions
#

function plot_dmrg_sz(sz::Vector{Float64}, nsite::Int; title = "")
    return plot(
        sz; xlim = (1, nsite), ylim = (-0.25, 0.25),
        xlabel = "Site j", ylabel = "⟨Szⱼ⟩", legend = false, title
    )
end
plot_dmrg_sz(res; kwargs...) = plot_dmrg_sz(res.sz, res.nsite; kwargs...)


function animate_dmrg_sz(res; fps = res.nsite)
    return animate(; nframes = length(res.sz_frames), fps) do i
        return plot_dmrg_sz(res.sz_frames[i], res.nsite; title = "Sweep = $(i ÷ (2 * res.nsite) + 1)")
    end
end


