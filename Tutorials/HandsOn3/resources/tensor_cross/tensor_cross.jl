using Printf: @printf
using ITensors: ITensors, Index, ITensor, combiner, combinedind, commonind, dim, matrix, hastags, onehot, permute, inds, uniqueind, val
using ITensorMPS: ITensorMPS, MPS, sweepnext

include("lu.jl")

function init_cross(sites, f::Function, piv::Vector)
    N = length(sites)
    M = MPS(sites)
    fp = f(piv...)
    #if(iszero(fp)) 
    #  println("Initial pivot = ",piv)
    #  error("Value of function is zero on initial pivot, please choose another pivot")
    #end

    Js = [Index(1; tags = "Link,J$j") for j = 1:N+1]

    T1 = ITensor(sites[1], Js[2])
    for sv in sites[1]
        T1[sv, Js[2]=>1] = f(val(sv), piv[2:end]...)
    end
    M[1] = T1

    for j = 2:N
        Tj = ITensor(Js[j], sites[j], Js[j+1])
        for sv in sites[j]
            Tj[Js[j]=>1, sv, Js[j+1]=>1] = f(piv[1:j-1]..., val(sv), piv[j+1:end]...)
        end
        M[j] = Tj / fp
    end
    M[N] *= onehot(Js[N+1] => 1)

    # Make initial pivots data structure
    pivs = fill([Int[]], N - 1)
    for j = 1:(N-1)
        pivs[j] = [piv[j+1:end]]
    end

    return M, pivs
end

"""
Helper function for cross_interpolate
"""
function ilinkind(M::MPS, b::Int)
    N = length(M)
    (b < 0 || b > N) && error("b=$b out of range in ilinkind")
    if 1 <= b < N
        return commonind(M[b], M[b+1]; tags = "Link")
    elseif b == 0
        return uniqueind(M[1], M[2]; tags = "Link")
    elseif b == N
        return uniqueind(M[N], M[N-1]; tags = "Link")
    end
end


"""
    tensor_cross(sites::Vector{<:Index}, f::Function; kwargs...)

Perform the tensor cross interpolation (TCI) / TT-cross algorithm on the function `f`,
returning an MPS approximating `f` together with information about the interpolation.

The function is sampled only at the points needed to build the interpolation, so the
number of evaluations of `f` is typically far smaller than the total number of grid points
`prod(dim.(sites))`. Each distinct set of arguments is evaluated at most once (values are
cached).

# Arguments
- `sites::Vector{<:Index}`: Array or container of `Index` objects for each bit encoding `f`. 
  These become the site indices of the returned MPS.
- `f::Function`: The function to interpolate. It is called as `f(i₁, i₂, ..., iₙ)` with
  `n = length(sites)` integer arguments, where `iⱼ` ranges over `1:dim(sites[j])`, and
  must return a number (real or complex).

# Keywords
- `nsweep::Int = 1`: Number of back-and-forth sweeps over the bonds of the MPS.
- `outputlevel::Int = 0`: Controls how much information is printed. `1` prints the
  maximum interpolation error after each sweep and `2` additionally prints the rank
  and truncation error at each bond.
- `initial_pivot::Vector{Int}`: Index values `[i₁, ..., iₙ]`, one per site, of the point
  used to start the interpolation. Ideally this is near an extremum of `f`, and `f` must be
  nonzero there. Defaults to a random point.
- `cutoff::Real = 0.0`: Truncate the rank of each bond so that the estimated maximum
  (infinity norm) difference from the true function values remains below this value.
- `maxdim::Int = typemax(Int)`: Maximum rank (bond dimension) to keep at each bond.
- `mindim::Int = 1`: Minimum rank to keep at each bond.

# Returns
A tuple `(M, info)` where:
- `M::MPS`: An MPS with site indices `sites` such that contracting `M` with the index
  values `(i₁, ..., iₙ)` (for example by setting each site index to a `onehot` vector)
  approximates `f(i₁, ..., iₙ)`.
- `info::NamedTuple`: Information about the interpolation with the fields
  - `pivots::Vector{Vector{Vector{Int}}}`: For each bond `b` between sites `b` and `b+1`,
    a vector with one entry per value of that bond index, each entry giving the settings
    of the site indices on one side of the bond (the sites `b+1, ..., n` after a completed
    sweep) at the pivot point selected for that value.
  - `function_calls::Int`: The number of distinct points at which `f` was evaluated.

  If `f` returns real values the following fields are also included, giving the extreme
  values of `f` found among the points sampled during the interpolation (so they are
  estimates of the true extrema of `f`):
  - `max_value`, `max_inds::Vector{Int}`: The largest value found and the index values
    `[i₁, ..., iₙ]` at which it occurred.
  - `min_value`, `min_inds::Vector{Int}`: The smallest value found and the index values
    at which it occurred.
"""
function tensor_cross(
    sites::Vector{<:Index},
    f::Function;
    nsweep = 1,
    outputlevel = 0,
    initial_pivot = [rand(1:dim(sj)) for sj in sites],
    kwargs...,
)

    val_i = f(initial_pivot...)
    compute_min_max = isreal(val_i)
    fcache = Dict{Vector{Int},typeof(val_i)}()
    M, pivs = init_cross(sites, f, initial_pivot)

    # Define pivot helper function
    pivot(i, val) = (i < 1 || i >= N) ? Int[] : pivs[i][val]

    # Attach edge indices for convenience
    N = length(sites)
    l0 = Index(1; tags = "Link,I0")
    lN1 = Index(1; tags = "Link,J$(N+1)")
    M[1] *= onehot(l0 => 1)
    M[N] *= onehot(lN1 => 1)

    maxval = -Inf
    minval = +Inf
    maxarg = zeros(Int, N)
    minarg = zeros(Int, N)

    for sw = 1:nsweep
        max_inf_norm = 0.0
        for (n, ha) in sweepnext(N; ncenter = 2)
            (outputlevel >= 2) && println("sw=$sw, bond=$n,$(n+1), half=$ha")

            i = ilinkind(M, n - 1)
            j = ilinkind(M, n + 1)

            # Fill up Pi tensor with actual values from f
            Pi = M[n] * M[n+1]
            Pi = permute(Pi, i, sites[n], sites[n+1], j)
            inf_norm = 0.0
            for iv = 1:dim(i), sv1 = 1:dim(sites[n]), sv2 = 1:dim(sites[n+1]), jv = 1:dim(j)
                arg = vcat(pivot(n - 1, iv), [sv1, sv2], pivot(n + 1, jv))
                @assert length(arg) == N

                if haskey(fcache, arg)
                    val = fcache[arg]
                else
                    val = f(arg...)
                    fcache[arg] = val
                end

                if compute_min_max
                    if val > maxval
                        maxval = val
                        maxarg = arg
                    end
                    if val < minval
                        minval = val
                        minarg = arg
                    end
                end

                prev_val = Pi[iv, sv1, sv2, jv]
                inf_norm = max(inf_norm, abs(val - prev_val))

                Pi[iv, sv1, sv2, jv] = val
            end

            if ha == 1
                tags = "Link,I$n"
                row_inds = (sites[n+1], j)
                M[n+1], M[n], npivs, id_error = interpolative(Pi, row_inds; tags, kwargs...)
            elseif ha == 2
                tags = "Link,J$n"
                row_inds = (i, sites[n])
                M[n], M[n+1], npivs, id_error = interpolative(Pi, row_inds; tags, kwargs...)
            end

            if outputlevel >= 2
                println("  Truncated to rank χ = ", dim(commonind(M[n], M[n+1])))
                @printf("  Truncation max error = %.3E\n", id_error)
                @printf("  Max interpolation difference = %s\n", inf_norm)
            end
            max_inf_norm = max(max_inf_norm, inf_norm)

            # Update pivs
            b = commonind(M[n], M[n+1])
            sl = 1 # location of site index
            Z = (ha == 1) ? M[n] : M[n+1]
            for i in inds(Z)
                (i == b) && continue
                hastags(i, "Site") && break
                sl += 1
            end
            ll = (sl == 1) ? 2 : 1 # location of previous link index

            resize!(pivs[n], 0)
            for p in npivs
                li = p[ll]
                si = p[sl]
                if ha == 1
                    new = vcat(pivot(n - 1, li), [si])
                else
                    new = vcat([si], pivot(n + 1, li))
                end
                push!(pivs[n], new)
            end

        end
        (outputlevel >= 1) && @printf("After sweep %d, max error = %s\n", sw, max_inf_norm)
    end
    M[1] *= onehot(l0 => 1)
    M[N] *= onehot(lN1 => 1)
    if compute_min_max
        return M,
        (;
            pivots = pivs,
            function_calls = length(fcache),
            max_inds = maxarg,
            max_value = maxval,
            min_inds = minarg,
            min_value = minval,
        )
    end
    return M, (; pivots = pivs, function_calls = length(fcache))
end
