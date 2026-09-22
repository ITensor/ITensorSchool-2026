using Pkg
cd(@__DIR__)
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Revise
include(joinpath(@__DIR__, "resources", "suppress_julia_banner.jl"))
include(joinpath(@__DIR__, "resources", "itensor_man.jl"))
println("\n")
print_itensor_man()
printstyled("\n  Welcome to the ITensor School 2026!\n"; bold = true, color = :blue)
println("\nHandsOn4 environment ready. Use includet(\"file.jl\") to load tutorial files.\n")
