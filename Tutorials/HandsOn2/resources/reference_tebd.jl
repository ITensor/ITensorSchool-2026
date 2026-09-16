using ITensorMPS: ITensorMPS, MPS, expect, normalize!

#
# This function runs a reference ITensorMPS
# version of TEBD to provide data to check against
# your implementation of TEBD. It returns ⟨Sz⟩ on
# every site at each time in `time_range`.
#
function reference_tebd(gates, time_range, psi_init::MPS; truncation_keyword_args...)
    psi = copy(psi_init)
    szs = Vector{Float64}[]
    for t in time_range
        push!(szs, expect(psi, "Sz"))
        t≈last(time_range) && break
        psi = ITensorMPS.apply(gates, psi; truncation_keyword_args...)
        normalize!(psi)
    end
    return szs
end
