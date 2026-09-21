using ITensors: ITensor, scalar
using ITensorMPS: MPS, siteinds

function integrate(M::MPS)
    I = ITensor(1.0)
    for (m, s) in zip(M, siteinds(M))
        I *= (m * ITensor([1 / 2, 1 / 2], s))
    end
    return scalar(I)
end
