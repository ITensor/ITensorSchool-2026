function b2i(xs...)
    n = length(xs)
    i = 0
    for j = 1:n
        i += (xs[j] - 1) * 2^(n - j)
    end
    return i
end

"""
Convert integer arguments (1,2,2,1,2,...)
into a continuous variable in [0,1), understood
as a binary fraction.
"""
function b2c(xs...)
    n = length(xs)
    x = 0.0
    for j = 1:n
        x += (xs[j] - 1) / 2^j
    end
    return x
end
b2c(xs::Vector) = b2c(xs...)
b2c(xs::Tuple) = b2c(xs...)

"""
Convert a real number x in [0,1) to
a vector of integers [1,2,1,2,...]
representing the bits (1-indexed) of 
a binary fraction representation of x.
"""
function c2b(x::Real, n::Int)
    b = fill(1, n)
    val = 0.0
    δ = 1.0
    for j = 1:n
        δ /= 2
        nextval = val + δ
        if nextval <= x
            b[j] = 2
            val = nextval
        end
    end
    return b
end
