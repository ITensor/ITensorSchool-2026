# Regenerates the figures in `resources/images/` used by the HandsOn1 README that come from
# running the tutorials.
#
# Run from the `Tutorials/HandsOn1` directory with the HandsOn1 project activated:
#
#     julia --project=. resources/make_plots.jl
#
# The diagrams in `resources/images/` are drawn by hand and are not produced here.

using Plots: Plots, plot, savefig

const HANDSON_DIR = joinpath(@__DIR__, "..")
const IMAGE_DIR = joinpath(@__DIR__, "images")

module DMRG
include(joinpath(@__DIR__, "..", "2-dmrg.jl"))
end

Plots.default(; linewidth = 2, markersize = 5, markerstrokewidth = 0, size = (600, 400), dpi = 150)

# Tutorial 2: ground state energy per site of the spin-1/2 chain versus system size
function plot_energy_per_site(; nsites = 10:10:60)
    energies = [DMRG.main(; nsite, outputlevel = 0).energy for nsite in nsites]
    p = plot(
        nsites, energies ./ nsites;
        xlabel = "Number of sites", ylabel = "Energy per site", legend = false
    )
    savefig(p, joinpath(IMAGE_DIR, "2-energy_per_site.png"))
    return p
end

function make_all_plots()
    mkpath(IMAGE_DIR)
    plot_energy_per_site()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_all_plots()
end
