using ITensorMPS: MPS, MPO, OpSum, dmrg, maxlinkdim, random_mps, siteinds, linkinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner, orthogonalize
# Functions for time evolution
using ITensorMPS: apply, op
using LinearAlgebra: normalize, diag, svd
# Load the Plots package for plotting
using Plots: Plots, plot

# Load the `tebd` and `make_heisenberg_gates` functions from the TEBD implementation tutorial
include("1-tebd-implementation.jl")
include("resources/animate.jl")

"""
    plot_tebd_sz(res::NamedTuple; step::Int)

Plot ⟨Szⱼ⟩ on each site j at time step `step`. `res` is expected to be a `NamedTuple` with
fields `szs`, `times`, and `nsite`, such as the results of `main`.
"""
function plot_tebd_sz(res::NamedTuple; step::Int)
    (; szs, times, nsite) = res
    return plot(
        szs[step]; xlim = (1, nsite), ylim = (-0.5, 0.5), xlabel = "Site j",
        ylabel = "⟨Szⱼ(t=$(times[step]))⟩", legend = false
    )
end

"""
    animate_tebd_sz(res::NamedTuple; fps = res.nsite)

Animate `plot_tebd_sz` over all time steps. `res` is expected to be a `NamedTuple` with the
fields `plot_tebd_sz` uses, such as the results of `main`.
"""
function animate_tebd_sz(res::NamedTuple; fps = res.nsite)
    return animate(i -> plot_tebd_sz(res; step = i); nframes = length(res.szs), fps)
end

function entanglement_entropy(ψ::MPS, bond::Int = length(ψ) ÷ 2; cutoff = 1.0e-12)
    ψ = normalize(ψ)
    ψ = orthogonalize(ψ, bond)
    U, S, V = svd(ψ[bond], (linkinds(ψ, bond - 1)..., siteinds(ψ, bond)...))
    Sd = Array(diag(S))
    Sd2 = Sd .^ 2
    return sum(d -> (d > cutoff) ? -d * log(d) : 0.0, Sd2)
end

"""
   main(; kwargs...)

Perform time-evolving block decimation (TEBD) on a 1D Heisenberg spin-1/2 chain to perform
real time evolution of the Heisenberg ground state with a spin flip in the center of the
chain.

# Keywords
- `nsite::Int = 30`: Number of sites in the spin chain.
- `time::Float64 = 6.0`: Total time for evolution.
- `timestep::Float64 = 0.1`: Time step for each TEBD application.
- `cutoff::Float64 = 1.0e-10`: Cutoff for truncation during TEBD.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `H::MPO`: The Hamiltonian as an MPO.
- `psit::MPS`: The final wavefunction after time evolution as an MPS.
- `times::Vector{Float64}`: Vector of time points at which measurements were taken.
- `szs::Vector{Vector{Float64}}`: Vector of ⟨Sz⟩ measurements at each time point.
- `energies::Vector{ComplexF64}`: Vector of energy measurements at each time point.
- `entanglements::Vector{Float64}`: Vector of half chain entanglement entropies at each time point.
- `nsite::Int`: Same as above.
- `time::Float64`: Same as above.
- `timestep::Float64`: Same as above.
- `cutoff::Float64`: Same as above.
"""
function main(;
        # Number of sites
        nsite = 30,
        # TEBD parameters
        time = 6.0,
        timestep = 0.1,
        cutoff = 1.0e-10,
        outputlevel = 1,
    )
    # Build the physical indices for nsite spins (spin 1/2)
    sites = siteinds("S=1/2", nsite)

    # Build the Heisenberg Hamiltonian as an MPO for computing energies,
    # running DMRG, etc.
    terms = OpSum()
    for j in 1:(nsite - 1)
        terms += 1 / 2, "S+", j, "S-", j + 1
        terms += 1 / 2, "S-", j, "S+", j + 1
        terms += "Sz", j, "Sz", j + 1
    end
    H = MPO(terms, sites)

    if outputlevel > 0
        println("Constructing the starting state for time evolution")
    end
    # Site whose ⟨Szⱼ⟩ is reported while the simulation runs
    j = nsite ÷ 2

    # --- Initial state ---
    # Run DMRG to get a starting state for time evolution
    psi0 = random_mps(sites; linkdims = 10)
    _, psi = dmrg(
        H, psi0; nsweeps = 5, maxdim = [10, 20, 100, 100, 200],
        cutoff = [1.0e-10], outputlevel = min(outputlevel, 1)
    )
    # Make the starting state by applying `S+` to the center of the chain
    psit = normalize(apply(op("S+", sites[j]), psi))
    # --- End initial state ---

    gates = make_heisenberg_gates(sites, -im * timestep)

    if outputlevel > 0
        println("\nStarting real time evolution")
    end
    szs = [expect(psit, "Sz")]
    energies = ComplexF64[inner(psit', H, psit)]
    entanglements = [entanglement_entropy(psit, nsite ÷ 2)]
    times = 0.0:timestep:time
    print_every = 1
    for current_time in times[2:end]
        psit = normalize(tebd(gates, psit; cutoff))
        energy_t = inner(psit', H, psit)
        sz_t = expect(psit, "Sz")
        push!(szs, sz_t)
        push!(energies, energy_t)
        push!(entanglements, entanglement_entropy(psit, nsite ÷ 2))
        if floor(current_time - timestep + 10eps()) ≠ floor(current_time) &&
                floor(current_time) % print_every == 0
            if outputlevel > 0
                println("time: ", current_time)
                println("Bond dimension: ", maxlinkdim(psit))
                println("⟨ψₜ|Szⱼ|ψₜ⟩: ", sz_t[j])
                println("∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: ", sum(sz_t))
                println("⟨ψₜ|H|ψₜ⟩: ", energy_t)
                println()
            end
        end
    end

    # A blank `tebd_step` hands the state back untouched, which still plots as a perfectly
    # plausible flat time series
    if szs[end] ≈ szs[1]
        @warn "The state did not change over the evolution, is `tebd_step` implemented?"
    end

    res = (;
        H, psit, times, szs, energies, entanglements, nsite, time, timestep, cutoff,
    )
    if outputlevel > 1
        animate_tebd_sz(res)
    end
    return res
end
