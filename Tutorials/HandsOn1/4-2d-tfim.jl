using ITensorMPS: MPO, OpSum, dmrg, maxlinkdim, random_mps, siteinds
# Functions for performing 2D DMRG
using ITensorMPS: square_lattice
# Functions for performing measurements of MPS
using ITensorMPS: ITensorMPS, AbstractObserver, expect
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
# Used for computing average magnetization
using Statistics: mean
# Load the Plots package for plotting
using Plots: Plots, @layout, plot, plot!, quiver

include("../src/animate.jl")

function plot_spins(res, i::Int = length(res.szs))
    points = vec(reverse.(Tuple.(CartesianIndices((res.ny, res.nx)))))
    xs = first.(points)
    ys = last.(points)
    sxs = vec(res.sxs[i])
    szs = vec(res.szs[i])
    quiver_data = (sxs, szs)
    xlims = (0.2, res.nx + 0.4)
    ylims = (0.2, res.ny + 0.4)
    q_plt = quiver(xs, ys, quiver = quiver_data; title = "Magnetization", color = :blue, arrow = :closed, xlims, ylims)
    h_plt = plot(1:res.nx, res.fields; color = :red, label = "h(x)")
    layout = @layout [a{0.8h}; b{0.2h}]
    plt = plot(q_plt, h_plt; layout)
    return plt
end

function animate_spins(res; fps = res.nsite)
    return animate(i -> plot_spins(res, i); nframes = length(res.szs), fps)
end

@kwdef struct SxSzObserver <: AbstractObserver
    nx::Int
    ny::Int
    sxs::Vector{Matrix{Float64}} = Vector{Float64}[]
    szs::Vector{Matrix{Float64}} = Vector{Float64}[]
end
function ITensorMPS.measure!(obs::SxSzObserver; psi, kwargs...)
    push!(obs.sxs, reshape(expect(psi, "Sx"), (obs.ny, obs.nx)))
    push!(obs.szs, reshape(expect(psi, "Sz"), (obs.ny, obs.nx)))
    return nothing
end

function plot_rescaled(pairs::Pair...)
    isempty(pairs) && return
    res1, fac1 = pairs[1]
    az1 = vec(mean(res1.szs[end]; dims = 1))
    plt = plot(fac1 * collect(1:res1.nx), az1)
    for p in 2:length(pairs)
        res_p, fac_p = pairs[p]
        az_p = vec(mean(res_p.szs[end]; dims = 1))
        plt = plot!(plt, fac_p * collect(1:res_p.nx), az_p)
    end
    return plt
end


"""
    main(; kwargs...)
    
Perform DMRG on a 2D transverse-field Ising model (TFIM) and measure ⟨Sz⟩ and site densities.
    
# Keywords
- `nx::Int = 30`: Number of sites in x direction.
- `ny::Int = 3`: Number of sites in y direction.
- `h_max::Float64 = 1.5`: maximum on-site transverse field (reached at right-hand side of system)
- `ramp_width::Float64 = 4.0`: width of ramp function for magnetic field
- `nsweeps::Int = 4`: Number of DMRG sweeps.
- `maxdim::Vector{Int} = [2, 4, 16, 30, 60]`: Maximum bond dimensions for each sweep.
- `cutoff::Vector{Float64} = [1.0e-6]`: Cutoff values for each sweep.
- `noise::Vector{Float64} = [1.0e-6, 1.0e-7, 0.0, 0.0]`: Noise values for each sweep.
- `outputlevel::Int = 1`: Controls how much information will be printed by the script.

# Returns
A named tuple containing:
- `energy::Float64`: The optimized ground state energy.
- `fields::Vector{Float64}`: Transverse magnetic fields applied at each column `x`.
- `H::MPO`: The Hamiltonian as an MPO.
- `psi::MPS`: The ground state wavefunction as an MPS.
- `nx::Int`: Same as above.
- `ny::Int`: Same as above.
- `nsite::Int`: Total number of sites.
- `sxs::Vector{Vector{Float64}}`: Vector of ⟨Sx⟩ measurements at each DMRG step.
- `szs::Vector{Vector{Float64}}`: Vector of ⟨Sz⟩ measurements at each DMRG step.
- `nsweeps::Int`: Same as above.
- `maxdim::Vector{Int}`: Same as above.
- `cutoff::Vector{Float64}`: Same as above.
- `noise::Vector{Float64}`: Same as above.
"""
function main(;
        # Number of sites in x and y
        nx = 30,
        ny = 3,
        # Model parameters
        h_max = 1.5,
        ramp_width = 4.0,
        # DMRG parameters
        nsweeps = 4,
        maxdim = [2, 4, 16, 30, 60],
        cutoff = [1.0e-6],
        noise = [1.0e-6, 1.0e-7, 0.0, 0.0],
        outputlevel = 1,
    )
    nsite = nx * ny

    # Magnetic field profile at each x value
    field(x) = h_max * (1 / 2 + 1 / 2 * tanh((x - nx ÷ 2) / ramp_width))

    # Build the physical indices
    sites = siteinds("S=1/2", nsite)

    # Build the lattice
    lattice = square_lattice(nx, ny; yperiodic = true)

    # Build the transverse-field Ising model as an MPO
    terms = OpSum()
    for b in lattice
        i, j = b.s1, b.s2
        xi, xj = Int(b.x1), Int(b.x2)
        terms -= "Z", i, "Z", j
        terms -= field(xi) / 2, "X", i
        terms -= field(xj) / 2, "X", j
    end
    H = MPO(terms, sites)

    if outputlevel > 0
        println("MPO bond dimension: ", maxlinkdim(H))
    end

    # Initial state for DMRG
    rng = StableRNG(123)
    psi0 = random_mps(rng, sites; linkdims = 4)

    # It starts with a bond dimension 10
    if outputlevel > 0
        println("Initial MPS bond dimension: ", maxlinkdim(psi0))
    end

    # Run DMRG, measuring Sz and site densities during the sweep
    observer = SxSzObserver(; nx, ny)
    energy, psi = dmrg(
        H, psi0; nsweeps, maxdim, cutoff, observer, outputlevel = min(outputlevel, 1)
    )
    sxs = observer.sxs
    szs = observer.szs
    fields = field.(1:nx)

    res = (; energy, fields, H, psi, nx, ny, nsite, sxs, szs, nsweeps, maxdim, cutoff, noise)
    if outputlevel > 1
        animate_spins(res)
    end
    return res
end
