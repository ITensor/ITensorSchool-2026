#
# Three ways to use a CallbackObserver during a DMRG calculation.
#
using ITensors, ITensorMPS
using Printf

include("callback_observer.jl")

"Heisenberg Hamiltonian for a chain of `N` spins."
function heisenberg(N)
    os = OpSum()
    for j in 1:(N - 1)
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
        os += "Sz", j, "Sz", j + 1
    end
    return os
end

function main()
    N = 20
    sites = siteinds("S=1/2", N; conserve_qns = true)
    H = MPO(heisenberg(N), sites)
    psi0 = random_mps(sites, n -> isodd(n) ? "Up" : "Dn"; linkdims = 4)
    dmrg_kwargs = (; nsweeps = 8, maxdim = [10, 20, 50, 100, 200], cutoff = 1.0e-10, outputlevel = 0)

    ## 1. A callback can be as simple as printing whatever you care about.

    function report(; sweep, energy, psi, kwargs...)
        @printf("  sweep %2d   energy = %.10f   maxlinkdim = %3d\n", sweep, energy, maxlinkdim(psi))
        return nothing
    end

    println("1. Reporting after each sweep:")
    dmrg(H, psi0; observer = CallbackObserver(report), dmrg_kwargs...)

    ## 2. Because a callback is just a function, a closure can accumulate a history
    ##    of anything you like -- here the energy, bond dimension, truncation error
    ##    and wall time of every sweep -- with none of it living in the observer.

    history = []
    t0 = time()
    function record!(; sweep, energy, psi, spec, kwargs...)
        push!(
            history,
            (; sweep, energy, maxlinkdim = maxlinkdim(psi), truncerr = truncerror(spec), time = time() - t0)
        )
        return nothing
    end

    println("\n2. Accumulating a history in a closure:")
    energy, psi = dmrg(H, psi0; observer = CallbackObserver(record!), dmrg_kwargs...)
    for h in history
        @printf(
            "  sweep %2d   E - E₀ = %9.2e   maxlinkdim = %3d   truncerr = %8.1e   t = %5.2fs\n",
            h.sweep, h.energy - energy, h.maxlinkdim, h.truncerr, h.time
        )
    end

    ## 3. With `each_bond=true` the callback runs after every local update instead,
    ##    which is enough to watch the sweeps travel back and forth in real time.

    function show_sweep(; sweep, bond, half_sweep, energy, psi, kwargs...)
        chain = fill('-', length(psi) - 1)
        chain[bond] = half_sweep == 1 ? '>' : '<'
        @printf("\r  sweep %2d  |%s|  E = %.8f", sweep, String(chain), energy)
        flush(stdout)
        return nothing
    end

    println("\n3. Watching the orthogonality center sweep along the chain:")
    dmrg(H, psi0; observer = CallbackObserver(show_sweep; each_bond = true), dmrg_kwargs...)

    @printf("\n\nGround state energy: %.10f\n", energy)
end
