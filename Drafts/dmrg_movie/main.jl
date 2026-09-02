# Make a movie of a DMRG calculation, using a CallbackObserver to grab a
# snapshot of the MPS after every local update.
#
# The system is a spin-1 Heisenberg chain with open ends, in the total Sᶻ = +1
# sector. Its ground state is in the Haldane phase, whose signature is a spin-1/2
# degree of freedom living at each end of the chain -- so watch the staggered
# magnetization of the starting state melt away in the bulk of the chain while
# surviving, exponentially decaying, at the two edges.
#

using ITensors, ITensorMPS

include(joinpath(@__DIR__, "..", "callback", "callback_observer.jl"))
include(joinpath(@__DIR__, "plotting.jl"))

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

"""
    measure_chain(psi)

⟨Sᶻⱼ⟩ on every site and the von Neumann entanglement entropy of every bond,
both obtained in a single left-to-right pass over `psi`.
"""
function measure_chain(psi::MPS)
    psi = orthogonalize(psi, 1)  # returns a copy, so DMRG's own MPS is untouched
    N = length(psi)
    sz, entropy = zeros(N), zeros(N - 1)
    for b in 1:(N - 1)
        s = siteind(psi, b)
        sz[b] = real(inner(psi[b], apply(op("Sz", s), psi[b])))
        # Split the orthogonality center to read off the entanglement spectrum
        # of the cut between sites 1:b and b+1:N, then move the center to b+1.
        U, S, V = svd(psi[b], b == 1 ? (s,) : (linkind(psi, b - 1), s))
        for n in 1:dim(S, 1)
            p = S[n, n]^2
            p > 1.0e-14 && (entropy[b] -= p * log(p))
        end
        psi[b] = U
        psi[b + 1] *= S * V
    end
    s = siteind(psi, N)
    sz[N] = real(inner(psi[N], apply(op("Sz", s), psi[N])))
    return sz, entropy
end

function main()
    N = 24
    sites = siteinds("S=1", N; conserve_qns = true)
    H = MPO(heisenberg(N), sites)

    initial_state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    initial_state[N ÷ 2] = "Z0"
    psi0 = random_mps(sites, initial_state; linkdims = 4)

    # Callback function called by DMRG after every local update.
    frames = []
    function snapshot(; psi, energy, sweep, half_sweep, bond, kwargs...)
        sz, entropy = measure_chain(psi)
        push!(frames, (; sweep, half_sweep, bond, energy, sz, entropy, bonddims = linkdims(psi)))
        return nothing
    end

    energy, psi = dmrg(
        H, psi0;
        nsweeps = 6, maxdim = [10, 20, 40, 80, 160], cutoff = 1.0e-10,
        observer = CallbackObserver(snapshot; each_bond = true)
    )

    println("\nGround state energy: ", energy, " (Sᶻ = +1 sector, open ends)")
    println("Recorded $(length(frames)) frames, rendering movie...")
    println("Wrote ", dmrg_movie(frames; filename = joinpath(@__DIR__, "dmrg.mp4")))
end
