using ITensors: ITensors, ITensor, apply, commoninds, noprime, uniqueinds
using ITensorMPS: MPS, maxlinkdim, normalize!, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner, orthogonalize
# Functions for time evolution (for checking)
using ITensorMPS: op
using LinearAlgebra: norm, normalize, svd
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG
# Load the Plots package for plotting
using Plots: Plots, plot, plot!

include("resources/reference_tebd.jl")

function tebd_step(gate::ITensor, A::ITensor, B::ITensor; truncation_keyword_args...)

    # (1)
    # Add code to apply the gate to the portion of the MPS
    # consisting of tensors A and B
    #
    # gate_AB = ...
    #

    #TODO remove
    gate_AB = apply(gate,A*B)

    # (2)
    # Use the ITensor `svd` function to factorize and truncate
    # resulting tensor to restore an internal low-rank
    # bond structure
    #
    #                Determine correct arguments...
    #                | 
    #                v 
    # U, S, V = svd(...; truncation_keyword_args...)

    #TODO remove
    ui = uniqueinds(A,B)
    U,S,V = svd(gate_AB,ui; truncation_keyword_args...)
    

    # (3) 
    # Use the results of `svd` to assemble new MPS tensors
    #
    # A = ...
    # B = ...

    #TODO remove
    A = U*S
    B = V

    return A, B
end

function tebd(gates, psi; keyword_args...)
    psi = copy(psi)
    N = length(psi)
    fwd_sweep = [j=>(j+1) for j=1:N-1]
    rev_sweep = [j=>(j-1) for j=reverse(2:N)]
    bonds = vcat(fwd_sweep,rev_sweep)
    for (bond,gate) in zip(bonds,gates)
        s1, s2 = bond
        psi = orthogonalize(psi,s1)
        psi[s1], psi[s2] = tebd_step(gate,psi[s1],psi[s2]; keyword_args...)
    end
    return psi
end

function make_heisenberg_gates(sites, dt)
    N = length(sites)
    # Make gates (1,2),(2,3),(3,4),...
    gates = ITensor[]
    for j in 1:(N - 1)
        s1,s2 = sites[j], sites[j+1]
        hj =
          op("Sz", s1) * op("Sz", s2) +
          1 / 2 * op("S+", s1) * op("S-", s2) +
          1 / 2 * op("S-", s1) * op("S+", s2)
        Gj = exp(-im * dt / 2 * hj)
        push!(gates, Gj)
    end
    # Include gates in reverse order too
    # (N,N-1),(N-1,N-2),...
    append!(gates, reverse(gates))
    return gates
end

function main(; N=20,
                cutoff = 1E-8,
                maxdim = typemax(Int),
                dt = 0.1,
                ttotal = 5.0,
                outputlevel = 1
                )
    # Make an array of 'site' indices
    sites = siteinds("S=1/2", N)

    # Obtain time-evolution gates / circuit
    gates = make_heisenberg_gates(sites, dt)

    # Initialize psi to be a product state (alternating up and down)
    psi_init = MPS(sites, n -> isodd(n) ? "Up" : "Dn")
    psi = copy(psi_init)

    # Compute and record <Sz> on every site at each time step
    # then apply the gates to go to the next time
    time_range = 0.0:dt:ttotal
    szs = Vector{Float64}[]
    for t in time_range
        sz = expect(psi, "Sz")
        push!(szs, sz)
        if outputlevel > 0
            println("t = $t  ⟨Sz_$(N÷2)⟩ = $(sz[N÷2])")
        end
        t≈ttotal && break
        # Evolve psi with your implementation of tebd
        psi = tebd(gates, psi; cutoff, maxdim)
        normalize!(psi)
    end

    # Run the reference TEBD calculation for comparison
    szs_reference = reference_tebd(gates, time_range, psi_init; cutoff, maxdim)

    res = (; times=collect(time_range), szs, szs_reference, N, dt, ttotal, cutoff, maxdim)

    if outputlevel > 0
        max_diff = maximum(maximum(abs, sz - sz_ref) for (sz, sz_ref) in zip(szs, szs_reference))
        println("\nMaximum difference between your ⟨Szⱼ⟩ and the reference: $max_diff")
        # Plot results at the final time
        display(plot_szs(res))
    end
    return res
end

# Plotting functions

"""
    plot_szs(sz::Vector{Float64}, sz_reference::Vector{Float64}; title = "")
    plot_szs(res; step = length(res.times))

Plot ⟨Szⱼ⟩ on each site j from your `tebd` implementation (solid blue line,
filled markers) on top of the reference calculation (dashed green line,
open markers). Given the results `res` of `main`, plots the values at time
step `step` (by default the final time).
"""
function plot_szs(sz::Vector{Float64}, sz_reference::Vector{Float64}; title = "")
    nsite = length(sz)
    p = plot(
        1:nsite, sz_reference;
        label = "correct reference", color = :green, linestyle = :dash,
        marker = :circle, markersize = 6, markercolor = :white, markerstrokecolor = :green,
        xlim = (1, nsite), ylim = (-0.5, 0.5), xlabel = "Site j", ylabel = "⟨Szⱼ⟩", title,
    )
    plot!(
        p, 1:nsite, sz;
        label = "your tebd", color = :blue, linestyle = :solid,
        marker = :circle, markersize = 4, markercolor = :blue, markerstrokecolor = :blue,
    )
    return p
end

function plot_szs(res; step::Int = length(res.times))
    return plot_szs(res.szs[step], res.szs_reference[step]; title = "t = $(res.times[step])")
end
