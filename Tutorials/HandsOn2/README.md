# Day 2 Hands-On Tutorials

## Table of Contents

- [Tutorial 1: Real Time Evolution](#tutorial-1)
- [Tutorial 2: Imaginary Time Evolution](#tutorial-2)
- [Tutorial 3: Finite Temperature](#tutorial-3)
- [Tutorial 4: Expect](#tutorial-4)
- [Stretch Goals](#stretch-goals)

<a id="tutorial-1"></a>
<details>
  <summary><h2>Tutorial 1: Real Time Evolution</h2></summary>
  <hr>

In this tutorial we will simulate the time evolution of several initial states under the 1D spin-1/2 Heisenberg
Hamiltonian using the time evolving block decimation (TEBD) algorithm. See
the [ITensorMPS.jl tutorial on TEBD](https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html)
for more background on the algorithm. We will work off of the script [1-tebd.jl](./1-tebd.jl).

To get started with today's tutorials, first make sure you are in the correct directory (`Tutorials/Day2`). Once you are, activate the project for the day and instantiate the dependencies:
```julia
julia> pwd()
"[...]/ITensorCCQSchool/Tutorials/Day2"

julia> readdir()
6-element Vector{String}:
 "1-tebd.jl"
 "2-imaginary-time.jl"
 "3-metts.jl"
[...]

julia> ]

(@v1.12) pkg> activate .
  Activating project at `[...]/ITensorCCQSchool/Tutorials/Day2`

(Day2) pkg> instantiate
    Updating registry at `~/.julia/registries/General.toml`
    Updating `[...]/ITensorCCQSchool/Tutorials/Day2/Project.toml`
  [0d1a4710] + ITensorMPS v0.3.23
  [9136182c] + ITensors v0.9.14
[...]
```

The initial state constructed in `main` is the ground state of the Hamiltonian with the central spin excited. Running this with `main()` simulates the dynamics up until time `time = 5.0`:
```julia
julia> include("1-tebd.jl")
main

julia> res = main();
Constructing the starting state for time evolution
After sweep 1 energy=-13.090090463121994  maxlinkdim=10 maxerr=2.65E-03 time=8.142
After sweep 2 energy=-13.1112893979339  maxlinkdim=20 maxerr=2.81E-07 time=0.049
After sweep 3 energy=-13.111355728893775  maxlinkdim=46 maxerr=1.00E-10 time=0.091
After sweep 4 energy=-13.111355752032505  maxlinkdim=47 maxerr=9.98E-11 time=0.131
After sweep 5 energy=-13.111355752014125  maxlinkdim=47 maxerr=9.40E-11 time=0.129

Starting real time evolution
time: 1.0
Bond dimension: 40
⟨ψₜ|Szⱼ|ψₜ⟩: 0.35502158815958673
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.0000000000048608
⟨ψₜ|H|ψₜ⟩: -11.92934637704218 + 4.030838518672647e-15im

time: 2.0
Bond dimension: 47
⟨ψₜ|Szⱼ|ψₜ⟩: 0.08355269930503985
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.0000000000049247
⟨ψₜ|H|ψₜ⟩: -11.929346333266217 - 7.372917490718186e-16im

time: 3.0
Bond dimension: 61
⟨ψₜ|Szⱼ|ψₜ⟩: -0.05027844121579299
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.0000000000049203
⟨ψₜ|H|ψₜ⟩: -11.929346353859174 - 6.7265084051248846e-15im

time: 4.0
Bond dimension: 81
⟨ψₜ|Szⱼ|ψₜ⟩: 0.00558010630047976
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.000000000004993
⟨ψₜ|H|ψₜ⟩: -11.929346384584218 + 5.261764273111527e-16im

time: 5.0
Bond dimension: 99
⟨ψₜ|Szⱼ|ψₜ⟩: 0.06961335623586766
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.000000000010171
⟨ψₜ|H|ψₜ⟩: -11.929346264044392 + 7.06646111921064e-15im

time: 6.0
Bond dimension: 119
⟨ψₜ|Szⱼ|ψₜ⟩: 0.027820350947381323
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: 1.00000000001218
⟨ψₜ|H|ψₜ⟩: -11.929345107661792 - 4.71650358907334e-15im


julia> Plots.unicodeplots(); # Plot in the terminal

julia> plot_tebd_sz(res; step = 1) # S⁺|ψ⟩
            ┌────────────────────────────────────────┐
         0.5│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⡇⠀⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠀⠀⡀⠀⢀⡆⠀⡇⠀⠀⠀⢣⠀⡿⡀⢀⢆⠀⢠⡀⠀⢀⠀⠀⡀⠀⠀⠀⠀⠀│
⟨Szⱼ(t=0.0)⟩│⡤⠦⢤⠴⠵⢤⠮⢦⢤⠮⢦⡼⠼⡤⡼⢼⢤⠧⠤⠤⠤⢼⢼⠤⢧⡼⠼⣤⠧⠵⣤⠧⠧⡴⠭⢦⠴⠵⠤⠴│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠀⠁⠀⠱⠁⠈⣾⠀⠀⠀⠀⢸⡎⠀⠸⠁⠀⠈⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        -0.5│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            └────────────────────────────────────────┘
            ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site j⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀30⠀

julia> res.energies # Energy is approximately conserved
61-element Vector{ComplexF64}:
 -11.929346427893112 + 0.0im
 -11.929346421423265 + 4.790079870662175e-16im
 -11.929346417479275 - 6.6539534252930864e-15im
 -11.929346413789068 - 2.9342313727697586e-15im
 -11.929346409376699 + 1.8253653289020973e-15im
                     ⋮
 -11.929345696534325 + 9.66864767894011e-16im
 -11.929345533661307 - 5.5223249006699915e-15im
 -11.929345339070496 - 1.971017777958375e-15im
 -11.929345107661792 - 4.71650358907334e-15im

julia> sum.(res.szs) # Total spin at each time is approximately conserved
61-element Vector{Float64}:
 0.9999999999785e02
 1.0000000000049107
 1.000000000004897
 1.0000000000048772
 1.0000000000048648
 ⋮
 1.0000000000114306
 1.000000000011622
 1.0000000000118427
 1.00000000001218

julia> animate_tebd_sz(res) # Animation of ⟨Szⱼ⟩ as a function of time
            ┌────────────────────────────────────────┐
         0.5│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢰⡄⠀⣰⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⡀⠀⢠⠊⠉⠒⠉⠑⢄⠀⠀⠀⠀⠀⠀⠀⣀⠀⠀⠀⠀⠀⠀⡴⠁⠸⣠⠃⡇⢀⠷⡀⠀⣄⠀⠀│
⟨Szⱼ(t=6.0)⟩│⡤⠷⢤⠴⠭⠵⠥⠤⠤⠤⠤⠤⠤⠧⠴⠭⠭⠭⠭⠭⠤⠵⠶⠶⠦⠶⠮⠤⠤⠤⠯⠤⢼⡼⠤⢧⡼⠬⢦⠮│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠁⠀⠈⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        -0.5│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            └────────────────────────────────────────┘
            ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site j⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀30⠀

```
The animation lets us visualize how the excitation propagates through the system as a function of time.

1. Included in `main()` is a function `entanglement_entropy(ψ::MPS, bond::Int = length(ψ) ÷ 2)` to compute the von Neumann entanglement entropy between sites `1..bond` and `bond+1...N` of the MPS. The vector of half-chain entanglement entropies is output by `main` as `entanglements`.
Plot this half chain entanglement entropy as a function of time, how does it behave?

```julia
julia> Plots.unicodeplots()

julia> plot(res.times, res.entanglements; xlabel = "Time", ylabel = "Entanglement", legend = false)
            ┌────────────────────────────────────────┐
     1.28615│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠔⠒⠲⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠉⠒⠢⣄⣀⠀⠀⠀⢀⣀⠤⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠉⠁⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
Entanglement│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⢀⠴⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            │⠀⡇⠀⠀⠀⣀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     0.43001│⠀⡧⠔⠒⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
            └────────────────────────────────────────┘
            ⠀-0.18⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Time⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀6.18⠀

```
Is this what you would expect for a local quench? Why or why not? What happens around time `t ~ 5.0`? Try increasing the time of the simulation to `time = 8.0` to resolve the long-time behavior better. Notice that the simulation time per time step increases as a function of time, why is that the case?

2. We can change the initial state to something different. Let's try a state where all the spins are polarised along the z-axis. This can be done by commenting out the part of the code where the initial state was created by DMRG and then excited (lines 83-91) and substitute them for:
```julia
    psit = MPS(sites, ["Z+" for i in 1:nsite])
```
What do you notice about the dynamics of the quench now? Hint: think about the symmetries of the model.

3. Now try initializing the system in an anti-ferromagnetic state instead:
```julia
    psit = MPS(sites, [iseven(j) ? "Z+" : "Z-" for j in 1:nsite])
```

Plot the entanglement entropy as a function of time.

```julia
julia> plot(res.times, res.entanglements; xlabel = "Time", ylabel = "Entanglement", legend = false)
            ┌────────────────────────────────────────┐  
      3.0758│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠀│
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠴⠊⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠚⠁⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠖⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
Entanglement│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⢀⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⢀⠜⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
  -0.0895865│⠤⡷⠮⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│  
            └────────────────────────────────────────┘  
            ⠀-0.18⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Time⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀6.18⠀ 
``` 

How does this differ to the first initial state we used (the locally excited ground state)? What does this imply for the growth of the bond dimension of the function of time to maintain accuracy? Hint: for an arbitrary MPS of bond dimension $\chi$, $S_{\rm Von Neumann} \leq log_{2}(\chi)$.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-2"></a>
<details>
  <summary><h2>Tutorial 2: Imaginary Time Evolution</h2></summary>
  <hr>

Now we are going to switch from real time to imaginary time evolution. This is incredibly easy with tensor networks, as we can just perform the substitution $dt \rightarrow - {\rm i} d \beta$.

We will be working off the script [2-imaginary-time.jl](./2-imaginary-time.jl) which does this for you and implements the imaginary time dynamics of a random initial state under the Heisenberg Hamiltonian.


```julia
julia> res = main();
Run DMRG to get a reference energy for imaginary time evolution
After sweep 1 energy=-13.10580711255933  maxlinkdim=10 maxerr=2.04E-03 time=0.029
After sweep 2 energy=-13.111348929097458  maxlinkdim=20 maxerr=1.41E-07 time=0.040
After sweep 3 energy=-13.11135575001343  maxlinkdim=45 maxerr=9.81E-11 time=0.085
After sweep 4 energy=-13.111355751942149  maxlinkdim=47 maxerr=1.00E-10 time=0.118
After sweep 5 energy=-13.111355751949796  maxlinkdim=47 maxerr=1.00E-10 time=0.112

Starting imaginary time evolution
beta: 5.0
Bond dimension: 24
⟨ψₜ|Szⱼ|ψₜ⟩: -0.07015198148930198
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: -0.3554642454935465
⟨ψₜ|H|ψₜ⟩: -12.918726195417213

beta: 10.0
Bond dimension: 38
⟨ψₜ|Szⱼ|ψₜ⟩: -0.0007049850288560583
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: -0.12498385296119857
⟨ψₜ|H|ψₜ⟩: -13.082551163963094

beta: 15.0
Bond dimension: 40
⟨ψₜ|Szⱼ|ψₜ⟩: 0.007906094151056576
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: -0.035190911135993715
⟨ψₜ|H|ψₜ⟩: -13.105243727446057

beta: 20.0
Bond dimension: 40
⟨ψₜ|Szⱼ|ψₜ⟩: 0.005455558345839978
∑ⱼ⟨ψₜ|Szⱼ|ψₜ⟩: -0.010092112111528002
⟨ψₜ|H|ψₜ⟩: -13.109727125683662


julia> res.energies .- res.energy_ground_state
101-element Vector{Float64}:
 13.344740259203546
 11.607416095173846
  9.788767514056136
  8.027055433275628
  6.478091389576792
  5.219661141730995
  ⋮
  0.00210452773845482
  0.0019989336674051117
  0.001898845407890093
  0.0018039541390795222
  0.0017139718695293737
  0.0016286262661342477

```

1. Notice how the energy is converging to that of the DMRG calculation. You can show an animation of the local $Sz$ of each spin in the chain by calling:
```julia
julia> animate_tebd_sz(res)
[...]
```
Observe how the system relaxes to a state with no local magnetization, not unlike what we saw in similar animations of DMRG optimization (though note the convergence to the ground state is slower than DMRG in computation time.)

2. We can calculate the variance of `psit` to observe how close it is to an eigenstate of `H`. The variance for an operator $H$ is defined as $\langle H^2 \rangle - \langle H \rangle^2$. In ITensor, we can compute it as follows:
```julia
julia> inner(res.H, res.psit, res.H, res.psit) - inner(res.psit', res.H, res.psit)^2
0.00020948820113630973

```
Edit the `main` function in the file `2-imaginary-time.jl` to calculate the variance of the energy as a function of time in your simulation and have `main` return it as a new output `energy_vars`. As a reference, see how the `energies` are saved and computed, and note that as an optimization you could use the energy that was already computed at each step in the second term of the variance. Once you get that working, rerun the `main` function to compute the energy variance at each imaginary time step and plot them as follows:
```julia
julia> plot(res.betas, res.energy_vars; xlabel = "Imaginary Time", ylabel = "Energy Variance", legend = false)
               ┌────────────────────────────────────────┐  
        4.67397│⠀⡷⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
               │⠀⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
Energy Variance│⠀⡇⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⠀⢧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
               │⠀⡇⠀⠀⠀⠀⠣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
      -0.135919│⠤⡧⠤⠤⠤⠤⠤⠬⠶⠶⠶⠶⠦⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│  
               └────────────────────────────────────────┘  
               ⠀-0.6⠀⠀⠀⠀⠀⠀⠀⠀Imaginary Time⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀20.6⠀  
```

3. The initial state we used is a random `MPS`constructed via the lines
```julia
    rng = StableRNG(123)
    psit = random_mps(rng, sites)
```

Try changing the seed of the random number generator (the number `123` above) to generate a different random initial state. Does the result still converge to the ground state? Can you think of what initial states might prevent this happening? Hint: think about the symmetries of the model. Try to construct some. Does the variance still go to zero?


This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-3"></a>
<details>
  <summary><h2>Tutorial 3: Finite Temperature</h2></summary>
  <hr>

We are now going to run the METTS (minimally entangled thermal states) algorithm to extract finite temperature properties of the system while remaining in the pure state picture. This is done in the file [3-metts.jl](./3-metts.jl).

1. Run the `main` function from `3-metts.jl` to get an estimate of the energy of the 1D Heisenberg chain at finite temperature (by default, `nsite = 10` and `beta = 4.0`):
```julia
julia> include("3-metts.jl")

julia> res = main();
Making warmup METTS number 10
  Sampled state: ["Z-", "Z+", "Z+", "Z+", "Z-", "Z-", "Z+", "Z-", "Z+", "Z-"]
Making METTS number 10
  Energy of METTS 10 = -3.8574
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -4.0282 +- 0.0470  [-4.0751,-3.9812]
  Sampled state: ["Z-", "Z-", "Z+", "Z+", "Z-", "Z+", "Z-", "Z+", "Z+", "Z-"]
Making METTS number 20
  Energy of METTS 20 = -3.8099
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -4.0215 +- 0.0353  [-4.0567,-3.9862]
  Sampled state: ["Z-", "Z-", "Z+", "Z-", "Z-", "Z+", "Z+", "Z-", "Z-", "Z+"]
Making METTS number 30
  Energy of METTS 30 = -3.7525
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9553 +- 0.0403  [-3.9956,-3.9149]
  Sampled state: ["Z-", "Z-", "Z+", "Z-", "Z+", "Z+", "Z-", "Z-", "Z+", "Z-"]
Making METTS number 40
  Energy of METTS 40 = -3.7525
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9531 +- 0.0325  [-3.9856,-3.9207]
  Sampled state: ["Z-", "Z-", "Z+", "Z-", "Z+", "Z-", "Z+", "Z-", "Z-", "Z+"]
Making METTS number 50
  Energy of METTS 50 = -3.8910
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9307 +- 0.0306  [-3.9613,-3.9001]
  Sampled state: ["Z-", "Z+", "Z+", "Z-", "Z+", "Z-", "Z+", "Z-", "Z-", "Z+"]
Making METTS number 60
  Energy of METTS 60 = -4.1383
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9222 +- 0.0350  [-3.9572,-3.8872]
  Sampled state: ["Z+", "Z-", "Z-", "Z+", "Z-", "Z+", "Z-", "Z+", "Z+", "Z-"]
Making METTS number 70
  Energy of METTS 70 = -3.7682
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9442 +- 0.0313  [-3.9755,-3.9129]
  Sampled state: ["Z-", "Z+", "Z-", "Z+", "Z-", "Z-", "Z+", "Z-", "Z+", "Z-"]
Making METTS number 80
  Energy of METTS 80 = -3.6767
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9308 +- 0.0286  [-3.9594,-3.9022]
  Sampled state: ["Z-", "Z+", "Z+", "Z-", "Z+", "Z-", "Z+", "Z-", "Z-", "Z+"]
Making METTS number 90
  Energy of METTS 90 = -3.4279
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9253 +- 0.0268  [-3.9521,-3.8984]
  Sampled state: ["Z+", "Z-", "Z+", "Z-", "Z+", "Z-", "Z+", "Z+", "Z-", "Z+"]
Making METTS number 100
  Energy of METTS 100 = -4.1412
  Energy of ground state from DMRG -4.2580
  Estimated Energy = -3.9331 +- 0.0247  [-3.9578,-3.9084]
  Sampled state: ["Z-", "Z-", "Z+", "Z+", "Z+", "Z-", "Z-", "Z+", "Z-", "Z+"]

```

2. Next we will approximate the specific heat as a function of $\beta$. The specific heat can be approximated from the METTS algorithm via the following formula:

$$C_{v}(\beta) = \frac{\beta^2}{\rm nsite}\left(\overline{\langle H^2 \rangle} - \left(\overline{\langle H \rangle}\right)^2 \right)$$

where $\overline{X} = \frac{1}{\rm NMETTS}\sum_{i=1}^{\rm NMETTS}\langle X\rangle_{i}$ denotes the METTS ensemble average.

Start by modifying `main()` to keep track of the mean square energy ($\langle H^2 \rangle$) of each METTS after it has been evolved in a vector of values `square_energies` that you output from `main`. Use `energies` as a reference how to do that, and remember that you can compute $\langle H^2 \rangle$ for an operator `H` and a state `psi` in ITensor using:
```julia
inner(H, psi, H, psi)
```

Include the updated file and run `main` again to get the mean square energies of each METTS:
```julia
julia> include("3-metts.jl")
main

julia> res = main(; outputlevel = 0);

julia> res.energies
100-element Vector{Float64}:
 -4.074445803221543
 -4.09784222998449
 -4.097842229984583
 -4.212509962722872
 -4.143542986283167
  ⋮
 -4.138309287997185
 -4.188511217371247
 -4.141192907671604
 -4.1411719927464885

julia> res.square_energies
100-element Vector{Float64}:
 16.71674724578452
 16.900519950999097
 16.900519950999833
 17.803180810357997
 17.278567742109843
  ⋮
 17.16646036669021
 17.61775065669934
 17.214861545581314
 17.214678861292388

```
Use the formula above to compute the specific heat $C_{v}(\beta)$ using the energies and square energies. Note that Julia's [Statistics.jl](https://docs.julialang.org/en/v1/stdlib/Statistics/) standard library function `mean` is of use here, and we've already loaded it in the script for convenience. Specifically, define a function `specific_heat(res)` using `res.beta`, `res.nsite`, `res.energies`, and `res.square_energies` which you can call on results `res` to compute the specific heat:
```julia
julia> specific_heat(res) = [...]

julia> specific_heat(res)
0.2563153342962835
```
For the default parameters ($\beta = 4.0$, NMETTS $=100, nsite = 10$) provided you should find $C_{v}(\beta = 4.0) \approx 0.26$ (the random number generator (RNG) for the initial state and sampling is seeded so that the results are numerically reproducable).
Next we are going to measure the specific heat as a function of inverse temperature.

2. Construct an array of $\beta$ values:
```julia
julia> betas = 0.4:0.4:8.0;

```
and then create a vector of simulation outputs for these `betas`:
```julia
julia> results = [main(; beta, betastep = 0.1, NMETTS = 10, nsite = 15) for beta in betas];
[...]

```
This might take a few minutes to run. Also feel free to play around with the setting of `NMETTS`, which will trade off speed for better or worse statistical noise. We suggest setting `NMETTS = 10` and `nsite = 15` to get a coarse grained result on a slightly bigger system. Try plotting the result using the function `specific_heat(res)` you defined above by calling it on each of the results obtained:
```julia
julia> plot(betas, specific_heat.(results); xlabel = "Beta", ylabel = "Specific Heat", legend = false)
             ┌────────────────────────────────────────┐
     0.332206│⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠶⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⠀⠀⢀⡠⠃⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⠀⢸⠁⠀⠀⠀⠀⠈⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⠀⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⢢⠀⠀⡠⠒⢤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
Specific Heat│⠀⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡔⠁⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⠀⡎⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⢰⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠢⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠒⠒⠒⠤⠤⠤⢄⣀⡀⠀│
    0.0269131│⠀⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             └────────────────────────────────────────┘
             ⠀0.16⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Beta⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀8.64⠀

```
Feel free to enable another plotting backend by executing `Plots.gr()` if you want a clearer look at the data (and you can switch back to plotting in the REPL with `Plots.unicodeplots()`). The specific heat of the spin 1/2 antiferromagnetic Heisenberg model is known to display a broad peak at $T = 0.48J$ (here we have $J = 1$) with a maximum value of $~0.35J$. Do your results agree with this?

3. The high temperature regime should display an inverse square dependence of the specific heat with temperature, i.e $C_{v} \propto \frac{1}{T^{2}}$. Use a range $0 \leq \beta \leq 0.4$ to try to confirm this. When using a finer range of betas, make sure to adjust the `betastep` input of `main` to be commensurate with the chosen `betas` or you won't be able to reach the desired `betas` given the step size and the script will error. For example, you may want to use `betas = 0.1:0.1:0.5` and `betastep = 0.01`. You should be able to reproduce a plot like:
```julia
julia> plot(betas .^ 2, specific_heat.(results); xlabel = "Beta Squared", ylabel = "Specific Heat", legend = false)
             ┌────────────────────────────────────────┐
    0.0365955│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠊⠁⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠔⠊⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠔⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
Specific Heat│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⠀⠀⠀⢀⠤⠒⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⠀⠀⢀⡠⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
             │⠀⡇⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
  -0.00106589│⠤⡷⠭⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│
             └────────────────────────────────────────┘
             ⠀-0.0048⠀⠀⠀⠀⠀⠀Beta Squared⠀⠀⠀⠀⠀⠀⠀⠀⠀0.1648⠀

```

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-4"></a>
<details>
  <summary><h2>Tutorial 4: Expect</h2></summary>
  <hr>

In this tutorial, the goal is to write your own `expect` function. See the hints in the file "4-expect.jl".

Use the ["ITensors.jl Under the Hood" slides from today](https://itensor.org/school/04_ITensors.jl_Under_the_Hood.pdf)
as a reference for how to apply the operator to a site of the MPS.

For debugging your code, you can print the indices of the tensors
you are contracting to make sure they match up as expected (pay attention to the Index
ids!), and also print the tensors themselves to see their values. You can use
`println(t)`, `@show t`, `@show inds(t)`, etc.

Also it may be helpful to output intermediate objects from `main` as part of the
NamedTuple so you can inspect them in the REPL.

How does your implementation scale with system size? Try constructing other states using
the MPS constructor (see previous tutorials and slides as a reference) and try computing other
expectation values.

As an extra challenge, try writing your own `correlation_matrix` function.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="stretch-goals"></a>
<details>
  <summary><h2>Stretch Goals</h2></summary>
  <hr>

If you completed all the tutorials and would like more of a challenge, you can try the following "stretch goal".

In the low temperature regime the spin 1/2 1D Heisenberg model is known to be a gapless Luttinger Liquid which is a phase of matter characterised by a specific heat $C_{v} \propto T$. See if you can confirm this by running the METTS code in the low temperature regime (say $8.0 \leq \beta \leq 10.0$) and measuring the specific heat capacity. Note that in this low-temperature regime, finite size effects will be more significant and the imaginary time evolution needed to reach the lower temperatures will take longer, so you will have to be careful about the parameters you choose and simulations could take some time. It can help to take a large enough `betastep` (say `betastep = O(0.1)`) so your simulations run in reasonable time.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>
