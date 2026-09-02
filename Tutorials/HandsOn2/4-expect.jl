using ITensorMPS: MPS, random_mps, expect, correlation_matrix, op, siteinds, linkinds
using ITensors: ITensor, inds, sim, prime, noprime, apply
using StableRNGs: StableRNG
using Plots: Plots, plot

function main(; outputlevel = 1)
    N = 10
    s = siteinds("S=1/2", N)
    rng = StableRNG(123)
    psi = random_mps(rng, s; linkdims = 3)
    # Make a new version of `psi` with randomized link indices.
    dpsi = sim(linkinds, psi)

    O = "X"
    Ls = Vector{ITensor}(undef, N)
    L = ITensor(1.0)
    for j in 1:N
        L = L * psi[j] * dpsi[j]
        Ls[j] = L
    end

    # Check the result of the full contraction, i.e. `L[]` (`[]`) extracts a scalar
    # from an 0-dimensional tensor). Why do you get that result?

    # Make a similar vector of right environments `Rs`
    # by contracting right to left.

    # Use `Ls` and `Rs` as environments to compute the expectation value
    # of operator `O` on every site. As a reminder, you can construct
    # the operator on site `j` with `op(O, s[j])`, which will have indices
    # `(s[j]', s[j])` (i.e. pairs of primed and unprimed indices).

    # Check the results against `expect(psi, O)` for each site `j`.
    Os = expect(psi, O)
    if outputlevel > 0
        println("<$(O)ⱼ>: ", Os)
    end

    return (; Os, psi, L, Ls)
end
