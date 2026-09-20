using ITensors: Index, ITensor, dim, onehot, prime, replaceind, swapprime
using ITensorMPS: MPO, MPS, apply, siteinds

#
# Construction of the quantum Fourier transform (QFT) as an MPO
# of low rank by interpolation on a Chebyshev grid, following:
#
#   Jielun Chen and Michael Lindsey, "Direct interpolative construction
#   of the discrete Fourier transform as a matrix product operator",
#   arXiv:2404.03182
#

default_qft_rank() = 14

"""
Lagrange interpolating polynomial for the points X(i)
i=0,1,...K evaluated at the position x
"""
function L(j::Integer, X, K::Integer, x::Number)
    val = 1.0
    for i = 0:K
        (j == i) && continue
        val *= (x - X(i)) / (X(j) - X(i))
    end
    return val
end

"""
Chebyshev grid points on [0,1]
"""
c(j, K) = (1 - cos(j * π / K)) / 2

P(j, K, x) = L(j, i -> c(i, K), K, x)

function qft_mpo(
    s::Vector{<:Index},
    t::Vector{<:Index};
    rank = default_qft_rank(),
)
    n = length(s)
    @assert n == length(t)

    K = rank - 1

    l = [Index(rank, "Link,QFT,j=$j") for j = 1:n+1]

    Q = MPO(n)
    for j = 1:n
        A = ITensor(l[j], s[j], t[j], l[j+1])
        for a = 0:K, σ = 0:1, τ = 0:1, b = 0:K
            A[1+a, 1+σ, 1+τ, 1+b] =
                P(a, K, (σ + c(b, K)) / 2) * exp(-π * im * (σ + c(b, K)) * τ)
        end
        # Put in normalization that
        # multiplies to become 1/√N
        Q[j] = A / √(dim(s[j]))
    end

    Q[1] *= ITensor(ones(rank), l[1])
    Q[n] *= onehot(l[n+1] => 1)

    return Q
end

qft_mpo(s::Vector{<:Index}; kws...) = qft_mpo(s, prime(s); kws...)

function fourier_transform(psi::MPS; inverse=false, rank = default_qft_rank(), kws...)
    s = siteinds(psi)
    n = length(s)
    Q = qft_mpo(s, prime(s); rank)
    if inverse
      for j=1:n
        Q[j] = swapprime(Q,0=>1)
      end
    end
    psi_q = apply(Q, psi; kws...)
    rev_psi_q = [replaceind(psi_q[j], s[j] => reverse(s)[j]) for j = 1:n]
    return MPS(reverse(rev_psi_q))
end
