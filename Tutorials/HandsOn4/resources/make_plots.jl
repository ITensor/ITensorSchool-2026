# Regenerates the figures in `resources/images/` used by the HandsOn4 README.
#
# Run from the `Tutorials/HandsOn4` directory with the HandsOn4 project activated:
#
#     julia --project=. resources/make_plots.jl
#
# The figures need a working belief propagation implementation, so this script loads the
# completed `belief_propagation.jl` from the `Solutions` folder rather than the partially
# blank one in this directory.

using Graphs: nv
using NamedGraphs: named_grid
using Plots: Plots, plot, savefig

const HANDSON_DIR = joinpath(@__DIR__, "..")
const SOLUTIONS_DIR = joinpath(HANDSON_DIR, "..", "..", "Solutions", "HandsOn4")
const IMAGE_DIR = joinpath(@__DIR__, "images")

include(joinpath(HANDSON_DIR, "ising_tensornetwork.jl"))
include(joinpath(SOLUTIONS_DIR, "belief_propagation.jl"))

Plots.gr()
Plots.default(; linewidth = 2, markersize = 5, markerstrokewidth = 0, size = (600, 400), dpi = 150)

function bp_phi(g, beta)
    tn = ising_tensornetwork(g, beta)
    messages, niters = belief_propagation(tn, g; niters = 1000, outputlevel = 0)
    return phi_bp(tn, g, messages), niters, tn, messages
end

phi_1d_obc(L, beta) = log(2 * cosh(beta)^(L - 1)) / L
phi_1d_pbc(L, beta) = log(cosh(beta)^L + sinh(beta)^L) / L

# Tutorial 3, exercise 2: BP error on a periodic ring versus system size
function plot_1d_pbc_error(; beta = 0.2, Lxs = 3:20)
    errs = [abs(bp_phi(named_grid((Lx, 1); periodic = true), beta)[1] - phi_1d_pbc(Lx, beta)) for Lx in Lxs]
    p = plot(
        Lxs, errs; yscale = :log10, marker = :circle, legend = false,
        xlabel = "System size Lx", ylabel = "|ϕ_BP - ϕ_exact|",
        title = "1D periodic ring, β = $beta",
    )
    savefig(p, joinpath(IMAGE_DIR, "3-bp_error_1d_pbc.png"))
    return p
end

# Tutorial 3, exercise 3: BP iterations to converge versus β on an open square grid
function plot_niters_vs_beta(; L = 15, betas = [0.05 * (i - 1) for i in 1:21])
    niters = [bp_phi(named_grid((L, L); periodic = false), beta)[2] for beta in betas]
    p = plot(
        betas, niters; marker = :circle, legend = false,
        xlabel = "β", ylabel = "BP iterations to converge",
        title = "$(L)x$(L) open square grid",
    )
    savefig(p, joinpath(IMAGE_DIR, "3-bp_niters_vs_beta.png"))
    return p
end

# Tutorial 3, exercise 5: BP error versus L on an open square grid at small β
function plot_2d_obc_error(; beta = 0.1, Ls = 3:20)
    exact = ising_phi(beta)
    errs = [abs(bp_phi(named_grid((L, L); periodic = false), beta)[1] - exact) for L in Ls]
    p = plot(
        Ls, errs; marker = :circle, legend = false,
        xlabel = "System size L", ylabel = "|ϕ_BP - ϕ_Onsager|",
        title = "LxL open square grid, β = $beta",
    )
    savefig(p, joinpath(IMAGE_DIR, "3-bp_error_2d_obc.png"))
    return p
end

# Tutorial 3, exercise 7: BP error versus β on a periodic square grid
function plot_2d_pbc_error(; L = 5, betas = [0.01 * (i - 1) for i in 1:101])
    g = named_grid((L, L); periodic = true)
    errs = [abs(bp_phi(g, beta)[1] - ising_phi(beta)) for beta in betas]
    p = plot(
        betas, errs; legend = false,
        xlabel = "β", ylabel = "|ϕ_BP - ϕ_Onsager|",
        title = "Periodic square lattice",
    )
    savefig(p, joinpath(IMAGE_DIR, "3-bp_error_2d_pbc.png"))
    return p
end

# Tutorial 4, exercise 8: BP error and cluster-corrected BP error versus β
function plot_cluster_correction(; L = 5, betas = [0.01 * (i - 1) for i in 1:101])
    g = named_grid((L, L); periodic = true)
    bp_errs = Float64[]
    corrected_errs = Float64[]
    for beta in betas
        phi, _, tn, messages = bp_phi(g, beta)
        correction = phi_cluster_correction(tn, g, messages; smallest_loop_size = 4)
        exact = ising_phi(beta)
        push!(bp_errs, abs(phi - exact))
        push!(corrected_errs, abs(phi + correction - exact))
    end
    p = plot(
        betas, [bp_errs, corrected_errs]; label = ["BP" "BP + loop correction"],
        xlabel = "β", ylabel = "Absolute error vs Onsager",
        title = "Periodic square lattice",
    )
    savefig(p, joinpath(IMAGE_DIR, "4-bp_cluster_correction.png"))
    return p
end

function make_all_plots()
    mkpath(IMAGE_DIR)
    plot_1d_pbc_error()
    plot_niters_vs_beta()
    plot_2d_obc_error()
    plot_2d_pbc_error()
    plot_cluster_correction()
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_all_plots()
end
