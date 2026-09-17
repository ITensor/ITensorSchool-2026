include("binary_fractions.jl")

"""
    evaluate(M::MPS, v::Vector{Int})

Given a vector of "bit" settings (values 1 or 2, 1-indexed)
evaluate the scalar value of the MPS for the specified
bit settings.
"""
function evaluate(M::MPS, v)
    V = ITensor(1.0)
    s = siteinds(M)
    for (j, sj) in enumerate(s)
        V *= M[j] * onehot(s[j] => v[j])
    end
    return scalar(V)
end
"""
    evaluate(M::MPS, x::Float64)

Evaluate the function encoded by the MPS `M`
at the point `x` in [0,1) (rounded to the nearest grid point).
"""
evaluate(M::MPS, x::Float64) = evaluate(M, c2b(x, length(M)))
