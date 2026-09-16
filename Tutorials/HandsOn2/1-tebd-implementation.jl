using ITensors: ITensors, ITensor, commoninds, uniqueinds
using ITensorMPS: MPS, maxlinkdim, normalize!, siteinds
# Functions for performing measurements of MPS
using ITensorMPS: expect, inner, orthogonalize
# Functions for time evolution (for checking)
using ITensorMPS: apply, op
using LinearAlgebra: norm, normalize, svd
# Use to set the RNG seed for reproducibility
using StableRNGs: StableRNG

function tebd_step(gate::ITensor, A::ITensor, B::ITensor; svd_keyword_args...)

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
    # U, S, V = svd(...; svd_keyword_args...)

    #TODO remove
    ui = uniqueinds(A,B)
    U,S,V = svd(gate_AB,ui; svd_keyword_args...)
    

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
                ttotal = 5.0
                )
    # Make an array of 'site' indices
    sites = siteinds("S=1/2", N)

    # Obtain time-evolution gates / circuit
    gates = make_heisenberg_gates(sites, dt)

    # Initialize psi to be a product state (alternating up and down)
    psi = MPS(sites, n -> isodd(n) ? "Up" : "Dn")

    # Compute and print <Sz> at each time step
    # then apply the gates to go to the next time
    time_range = 0.0:dt:ttotal
    szs = Float64[]
    for t in time_range
        sz = expect(psi, "Sz"; sites=N÷2)
        push!(szs,sz)
        println("t = $t  ⟨Sz_$(N÷2)⟩ = $sz")
        t≈ttotal && break
        psi = tebd(gates, psi; cutoff, maxdim)
        normalize!(psi)
    end
    res = (; times=collect(time_range), szs)
    return res
end
