using ITensors: ITensors, Index, ITensor, dag, dim
using LinearAlgebra: Diagonal

function eye(Elt, R::Int, C::Int)
    M = zeros(Elt, R, C)
    for j = 1:min(R, C)
        M[j, j] = 1.0
    end
    return M
end

eye(R::Int, C::Int) = eye(Float64, R, C)

"""
    backsolveL(L::AbstractMatrix)

Compute the inverse of a lower-triangular matrix L whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
function backsolveL(L::AbstractMatrix)
    N = size(L, 1)
    (size(L, 2) == N) ||
        error("backsolveL and backsolveU only supported for square matrices")
    iL = eye(eltype(L), N, N)
    for j = 1:(N-1), i = (j+1):N
        iL[i, j] = -L[i, j]
        for m = (j+1):(i-1)
            iL[i, j] -= L[i, m] * iL[m, j]
        end
    end
    return iL
end

"""
    backsolveU(U::AbstractMatrix)

Compute the inverse of an upper-triangular matrix U whose
diagonal entries are all equal to 1.0 using a stable
back-solving algorithm.
"""
backsolveU(U::Matrix) = transpose(backsolveL(transpose(U)))

"""
    prrldu(M::Matrix; cutoff, maxdim)

Compute the pivoted, rank-revealing LDU decomposition of an
arbitrary matrix M (M can be non-invertible and/or rectangular).
Returns matrices L,D,U and permutations pr and pc such that
L and U are lower- and upper-triangular matrices with diagonal
values equal to 1 and L[pr,:]*D*U[:,pc] ≈ M. The diagonal matrix
D will have size (k,k) with diagonal entries of decreasing
absolute value such that norm(L[pr,:]*D*U[:,pc]-M,Inf) <= abs(D[k,k]).
(Note that this inequality uses the infinity norm.)
The value of k is determined dynamically such that both `k <= maxdim`
and `abs(D[k,k]) < cutoff`.
"""
function prrldu(M_::Matrix; cutoff::Real = 0.0, maxdim::Int = typemax(Int), mindim::Int = 1)
    @assert maxdim > 0
    @assert mindim > 0
    mindim = min(maxdim, mindim)
    Elt = eltype(M_)
    M = copy(M_)
    Nr, Nc = size(M)
    k = min(Nr, Nc)

    # Determine pivots
    rps = collect(1:Nr)
    cps = collect(1:Nc)

    inf_error = 0.0
    for s = 1:k
        Mabs_max, piv = findmax(abs, M)
        if Mabs_max < cutoff
            inf_error = Mabs_max
            break
        end
        Base.swaprows!(M, 1, piv[1])
        Base.swapcols!(M, 1, piv[2])
        M = M[2:end, 2:end] - M[2:end, 1] * transpose(M[1, 2:end]) / M[1, 1]
        rps[s], rps[piv[1]+s-1] = rps[piv[1]+s-1], rps[s]
        cps[s], cps[piv[2]+s-1] = cps[piv[2]+s-1], cps[s]
    end
    M = M_[rps, cps]

    L = eye(Elt, Nr, k)
    d = zeros(Elt, k)
    U = eye(Elt, k, Nc)
    rank = 0
    for s = 1:min(k, maxdim)
        P = M[s, s]
        d[s] = P

        if rank < mindim
            # then proceed
        elseif (iszero(P) || (abs(P) < cutoff && rank + 1 > mindim))
            break
        end
        iszero(P) && (P = one(Elt))
        rank += 1

        piv_col = M[(s+1):end, s]
        L[(s+1):end, s] = piv_col / P

        piv_row = M[s, (s+1):end]
        U[s, (s+1):end] = piv_row / P

        if s < k
            M[(s+1):end, (s+1):end] =
                M[(s+1):end, (s+1):end] - piv_col * transpose(piv_row) / P
        end
    end
    L = L[:, 1:rank]
    d = d[1:rank]
    U = U[1:rank, :]

    return L, d, U, invperm(rps), invperm(cps), inf_error
end

"""
    interpolative(M::Matrix; cutoff, maxdim)

Returns `C, Z, piv_cols, inf_error` where
C and Z are matrices such that `C*Z ≈ M`.
The matrix C consists of columns of M, and
which column is given by the integer entries of
the array `piv_cols`. The number of columns of
C is controlled by the approximate (SVD) rank of M,
which is controlled by the parameters `cutoff`
and `maxdim`.

"""
function interpolative(M::Matrix; kws...)
    # Compute interpolative decomposition (ID) from PRRLU
    L, d, U, pr, pc, inf_error = prrldu(M; kws...)
    U11 = U[:, 1:length(d)]
    iU11 = backsolveU(U11)
    ZjJ = iU11 * U
    CIj = L * Diagonal(d) * U11
    C = CIj[pr, :]
    Z = ZjJ[:, pc]
    # Compute mapping of pivot columns to column indices
    piv_cols = invperm(pc)[1:length(d)]
    return C, Z, piv_cols, inf_error
end

"""
    interpolative(T::ITensor, row_inds; kws...)

Compute the interpolative decomposition of an ITensor, treated as a 
matrix with row indices given by the collection `row_inds`. 

Return a tuple of the following:
* C - ITensor containing specific columns of `M`
* Z - ITensor such that `C*Z ≈ M`
* pivs - array of arrays specifying the settings of the column indices of `M`
  corresponding to the columns contained in C
* inf_error - maximum elementwise (infinity norm) error between `C*Z` and `M`

Internally uses the pivoted, rank-revealing LDU matrix decomposition.

Optional keyword arguments:
* maxdim::Int - maximum number of columns to keep in factorization
* mindim::Int - minimum number of columns to keep in factorization
* cutoff::Float64 - keep only as many columns such that the value of the infinity (max) norm difference from the original tensor is below this value
* tags="Link" - tags to use for the Index connecting `C` to `Z`
"""
function interpolative(
    T::ITensor,
    row_inds;
    cutoff = 0.0,
    maxdim = typemax(Int),
    mindim = 1,
    tags = "Link",
)
    # Matricize T
    col_inds = setdiff(inds(T), row_inds)
    Cmb_row, Cmb_col = combiner(row_inds), combiner(col_inds)
    cr, cc = combinedind(Cmb_row), combinedind(Cmb_col)
    t = matrix(Cmb_row * T * Cmb_col, cr, cc)

    c, z, piv_cols, inf_error = interpolative(t; cutoff, maxdim, mindim)

    # Make ITensors from C and Z matrices
    r = length(piv_cols)
    b = Index(r; tags)
    C = ITensor(c, cr, b) * dag(Cmb_row)
    Z = ITensor(z, b, cc) * dag(Cmb_col)

    # Compute mapping of pivot columns to column indices
    col_ranges = [1:dim(i) for i in col_inds]
    pivs = [zeros(Int, length(col_ranges)) for c = 1:r]
    for (col, vals) in enumerate(Iterators.product(col_ranges...))
        loc = findfirst(==(col), piv_cols)
        if !isnothing(loc)
            pivs[loc] = [v for v in vals]
        end
    end

    return C, Z, pivs, inf_error
end
