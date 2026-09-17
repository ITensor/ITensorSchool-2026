# Regenerates the figures in `resources/images/` used by the HandsOn2 README.
#
# Run from the `Tutorials/HandsOn2` directory with the HandsOn2 project activated:
#
#     julia --project=. resources/make_plots.jl
#
# Some figures show the result of an exercise, so this script loads the completed METTS file
# from the `Solutions` folder rather than the one in this directory. Each tutorial is loaded
# into its own module because they all define a function called `main`.

using ITensorMPS: MPO, MPS, OpSum, inner, random_mps, siteinds
using LinearAlgebra: normalize
using StableRNGs: StableRNG
using Plots: Plots, plot, savefig

const HANDSON_DIR = joinpath(@__DIR__, "..")
const SOLUTIONS_DIR = joinpath(HANDSON_DIR, "..", "..", "Solutions", "HandsOn2")
const IMAGE_DIR = joinpath(@__DIR__, "images")

module SpinChain
include(joinpath(@__DIR__, "..", "2-tebd-spin-chain.jl"))
end

module METTS
include(joinpath(@__DIR__, "..", "..", "..", "Solutions", "HandsOn2", "4-metts.jl"))
end

Plots.default(; linewidth = 2, markersize = 5, markerstrokewidth = 0, size = (600, 400), dpi = 150)

heisenberg_mpo(sites) = MPO(
    let terms = OpSum()
        for j in 1:(length(sites) - 1)
            terms += 1 / 2, "S+", j, "S-", j + 1
            terms += 1 / 2, "S-", j, "S+", j + 1
            terms += "Sz", j, "Sz", j + 1
        end
        terms
    end, sites
)

# Tutorial 2: the local quench of the Heisenberg ground state, at the README defaults
function plot_quench(; nsite = 30, time = 6.0)
    res = SpinChain.main(; nsite, time, outputlevel = 0)
    # A blank `tebd_step` still runs and returns the state unchanged, which would quietly
    # produce a figure set showing no dynamics at all
    if res.szs[end] ≈ res.szs[1]
        error("the state did not evolve, is `tebd_step` implemented?")
    end
    nstep = length(res.szs)
    for (name, step) in ("initial" => 1, "mid" => (nstep + 1) ÷ 2, "final" => nstep)
        savefig(
            SpinChain.plot_tebd_sz(res; step),
            joinpath(IMAGE_DIR, "2-sz_$name.png")
        )
    end
    p = plot(
        res.times, res.entanglements;
        xlabel = "Time", ylabel = "Entanglement", legend = false
    )
    savefig(p, joinpath(IMAGE_DIR, "2-entanglement_local_quench.png"))
    return p
end

# Tutorial 2, exercise 3: the same quantity starting from an anti-ferromagnetic state
function plot_neel_entanglement(; nsite = 30, time = 6.0, timestep = 0.1, cutoff = 1.0e-10)
    sites = siteinds("S=1/2", nsite)
    gates = SpinChain.make_heisenberg_gates(sites, timestep)
    psit = MPS(sites, [iseven(j) ? "Z+" : "Z-" for j in 1:nsite])
    times = 0.0:timestep:time
    entanglements = [SpinChain.entanglement_entropy(psit, nsite ÷ 2)]
    for _ in times[2:end]
        psit = normalize(SpinChain.tebd(gates, psit; cutoff))
        push!(entanglements, SpinChain.entanglement_entropy(psit, nsite ÷ 2))
    end
    p = plot(
        times, entanglements; xlabel = "Time", ylabel = "Entanglement", legend = false
    )
    savefig(p, joinpath(IMAGE_DIR, "2-entanglement_neel.png"))
    return p
end

# Tutorial 3, exercise 2: the energy variance along the imaginary time evolution
function plot_energy_variance(; nsite = 30, beta = 20.0, betastep = 0.2, cutoff = 1.0e-10)
    sites = siteinds("S=1/2", nsite)
    H = heisenberg_mpo(sites)
    # Imaginary time evolution is real time evolution with dt -> -i dβ
    gates = SpinChain.make_heisenberg_gates(sites, -im * betastep)
    psit = random_mps(StableRNG(123), sites)
    betas = 0.0:betastep:beta
    variance(psi) = real(inner(H, psi, H, psi) - inner(psi', H, psi)^2)
    energy_vars = [variance(psit)]
    for _ in betas[2:end]
        psit = normalize(SpinChain.tebd(gates, psit; cutoff))
        push!(energy_vars, variance(psit))
    end
    p = plot(
        betas, energy_vars;
        xlabel = "Imaginary Time", ylabel = "Energy Variance", legend = false
    )
    savefig(p, joinpath(IMAGE_DIR, "3-energy_variance.png"))
    return p
end

# Tutorial 4, exercises 2 and 3: the specific heat at low and at high temperature
function plot_specific_heat(;
        betas = 0.2:0.2:8.0, high_temperature_betas = 0.1:0.1:0.5, nsite = 15, NMETTS = 40
    )
    results = [METTS.main(; beta, betastep = 0.1, NMETTS, nsite, outputlevel = 0) for beta in betas]
    p = plot(
        betas, METTS.specific_heat.(results);
        xlabel = "Beta", ylabel = "Specific Heat", legend = false
    )
    savefig(p, joinpath(IMAGE_DIR, "4-specific_heat.png"))

    high_temperature_results = [
        METTS.main(; beta, betastep = 0.01, NMETTS, nsite, outputlevel = 0)
            for beta in high_temperature_betas
    ]
    p_high_temperature = plot(
        high_temperature_betas .^ 2, METTS.specific_heat.(high_temperature_results);
        xlabel = "Beta Squared", ylabel = "Specific Heat", legend = false
    )
    savefig(p_high_temperature, joinpath(IMAGE_DIR, "4-specific_heat_high_temperature.png"))
    return p, p_high_temperature
end

function make_all_plots()
    mkpath(IMAGE_DIR)
    plot_quench()
    plot_neel_entanglement()
    plot_energy_variance()
    plot_specific_heat()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_all_plots()
end
