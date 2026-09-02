# Day 1 Hands-On Tutorials

## Table of Contents

- [Installation Instructions](#installation-instructions)
- [Tutorial 1: Julia Intro](#tutorial-1)
- [Tutorial 2: DMRG](#tutorial-2)
- [Tutorial 3: DMRG Measurments](#tutorial-3)
- [Tutorial 4: 2D Ising Model](#tutorial-4)
- [Stretch Goals](#stretch-goals)

<a id="installation-instructions"></a>
<details>
  <summary><h2>Installation Instructions</h2></summary>
  <hr>

To run the tutorials:

1. Download and install the latest release (v1.12.1) of Julia following the official
instructions here: https://julialang.org/install/. If you already have Julia installed,
please upgrade to Julia v1.12.1, which you can install by using the same installation
instructions which will install `juliaup`, which you can use from the command line to
upgrade to the latest version of Julia with:
```
$ juliaup update
```

2. Navigate to a folder on your machine, such as your home folder, where you want the tutorials to be located. (They will be put into a subfolder.)

3. Start the Julia REPL by executing the `julia` command, which should now be available
on your computer if you followed the installation instructions in step 1.:
```
$ julia
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.12.0 (2025-10-07)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org release
|__/                   |


julia> 1 + 1
2

```
Try typing a command (such as `1 + 1` shown above) to get a feel for how it works. A number of math operations are available out-of-the-box, such as `sin`, `cos`, etc., while other functionality (such as [linear algebra](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/)) requires loading packages. The surface level syntax is similar to other high level interactive languages like Python and MATLAB. The Julia documentation provides a helpful guide [comparing Julia to other comparable languages](https://docs.julialang.org/en/v1/manual/noteworthy-differences/).

4. Create a local copy of the tutorial code in a new directory `ITensorCCQSchool` in your current directory by running:
```julia
julia> using LibGit2: clone

julia> clone("https://github.com/ITensor/ITensorCCQSchool", "ITensorCCQSchool")
```
Here we use Julia's  [LibGit2 standard library](https://docs.julialang.org/en/v1/stdlib/LibGit2/) to clone the repository containing the tutorials. Alternatively you can execute `git clone https://github.com/ITensor/ITensorCCQSchool` directly from the command line (outside of the Julia REPL).

5. Now that you have Julia installed and the tutorial code available, we will give an introduction to running the first tutorial for day 1 ([1-julia-intro.jl](./1-julia-intro.jl)). Enter the `ITensorCCQSchool/Tutorials/Day1` directory using Julia's [`cd`](https://docs.julialang.org/en/v1/base/file/#Base.Filesystem.cd-Tuple{AbstractString}) function and install the dependencies from the Julia REPL:
```julia
julia> cd("ITensorCCQSchool/Tutorials/Day1")

julia> ]

(Day1) pkg> activate .
  Activating project at `[...]/ITensorCCQSchool/Tutorials/Day1/`

(Day1) pkg> instantiate
    Updating registry at `~/.julia/registries/General.toml`
    Updating `[...]/ITensorCCQSchool/Tutorials/Day1/Project.toml`
  [0d1a4710] + ITensorMPS v0.3.22
  [9136182c] + ITensors v0.9.13
  [...]

```
Executing `]` at the REPL enables the Pkg REPL, which is more convenient for entering Pkg commands. Press delete/backspace to exit the Pkg REPL and go back to the standard Julia REPL prompt. `activate .` enables the local environment/project in `Tutorials/Day1` (your current directory), where the package dependencies for the tutorials on the first day of the school are defined (in case you are curious, they are defined in the [Project.toml](./Project.toml)). `instantiate` installs those dependencies and performs some compilation. It may take some time but it will only need to be done once for each project (so in our case, once for each day of the school).

6. Use `include` to load the [first tutorial](./Day1/1-julia-intro.jl) into the REPL. That will introduce the function `main` which you can execute to run the tutorial:
```julia
julia> include("1-julia-intro.jl")
main

julia> main();
nterm = 100000
pi_approx = 3.1416026534897203
error = 9.999899927226608e-6

```
The script is approximating the value of `pi` using the [Leibniz formula](https://en.wikipedia.org/wiki/Leibniz_formula_for_%CF%80) using `nterm` terms in the series. To access the values that are returned from the `main` function so you can analyze them interactively, you can call the script like this:
```julia
julia> res = main();
nterm = 100000
pi_approx = 3.1416026534897203
error = 9.999899927226608e-6

julia> res
(pi_approx = 3.1416026534897203, error = 9.999899927226608e-6, nterm = 100000)

```
`res = main()` captures the output of the script to the [NamedTuple](https://docs.julialang.org/en/v1/base/base/#Core.NamedTuple) `res`, which you can think of as an anonymous struct, i.e. like a struct it has fields that can store different kinds of objects but that's all it is. You can access values from `res` as follows:
```julia
julia> res.pi_approx
3.1416026534897203

julia> res.error
9.999899927226608e-6

julia> keys(res)
(:pi_approx, :error, :nterm)

julia> (; pi_approx, error) = res

julia> pi_approx
3.1416026534897203

julia> error
9.999899927226608e-6

```

7. You can run the script with different parameters as follows:
```julia
julia> res = main(; nterm = 10^7);
Number of terms: 10000000
Approximate pi: 3.1415927535897814
Error: 9.999998829002266e-8

julia> res.pi_approx
3.1415927535897814

julia> pi # Julia's built-in definition of pi
π = 3.1415926535897...

```
and you can learn about other parameters by printing the documentation using Julia's [help mode](https://docs.julialang.org/en/v1/stdlib/REPL/#Help-mode):
```julia
julia> ?

help?> main
search: main @main min Main Pair join map mark asin wait max map! sin in tan

  main(; kwargs...)

  Approximate pi using the Leibniz formula and compute the error.

  Keywords
  ≡≡≡≡≡≡≡≡

    •  nterm::Int = 10^5: The number of terms to use in the approximation of pi.
    •  outputlevel::Int = 1: Controls how much information will be printed by the
       script.

  Returns
  ≡≡≡≡≡≡≡

  A named tuple containing:

    •  pi_approx::Float64: The approximate value of pi.
    •  error::Float64: The absolute error in the approximation of pi.
    •  nterm::Int: Same as above.

julia>

```

8. Note that you can analyze which directory you are in and what tutorial files are available directly from the Julia REPL using functions such as [`pwd`](https://docs.julialang.org/en/v1/base/file/#Base.Filesystem.pwd) and [`readdir`](https://docs.julialang.org/en/v1/base/file/#Base.Filesystem.readdir):
```julia
julia> pwd()
"[...]/ITensorCCQSchool/Tutorials/Day1"

julia> readdir()
7-element Vector{String}:
 "1-julia-intro.jl"
 "2-dmrg.jl"
 "3-dmrg-measure.jl"
[...]

```
Of course, you can also use your terminal and/or file manager as usual. Additionally, if you want to modify the script itself, one way to do that is:
```julia
julia> edit("1-julia-intro.jl")

```
which will open your file in a text editor determined by Julia (see the [documentation for `edit`](https://docs.julialang.org/en/v1/stdlib/InteractiveUtils/#InteractiveUtils.edit-Tuple{AbstractString,%20Integer}) for more details). Otherwise, open the file with your text editor or IDE of choice, such as Vim, Emacs, VS Code, etc. A convenient way to do that is by entering Julia's shell mode by executing `;` at the REPL:
```julia
julia> ;

shell> vi 1-julia-intro.jl

```
and press delete/backspace to go back to the Julia REPL. When you are done editing the tutorial script, simply save the file and `include` the file again to get a new `main` function to execute in your existing Julia session to see your changes reflected in the output:
```julia
julia> include("1-julia-intro.jl")
main

julia> res = main();
[...]

```
Note that if you don't call `include` again, you won't see the changes you make to the file reflected when you call the `main` function. (For advanced users, note that you can use [`Revise.includet`](https://timholy.github.io/Revise.jl/stable/cookbook/#includet-usage) as an alternative to `include` which would automatically track changes to the file and update `main` without having to call `include` each time.)

**Note:** We highly recommend keeping your Julia session open throughout each day of the tutorial, which will keep your package environment active and ensure you don't incur re-compilation of precompiled code. If at some point you close your Julia session, make sure to enter the directory corresponding to the tutorial day (i.e. `Tutorials/Day1`) and execute:
```julia
julia> ]

pkg> activate .
```
to activate the environment, which will ensure you have the correct dependencies available to run the tutorial scripts for that day.

Also note that if you want to cancel a calculation that is in-progress, you can execute Control-C on your keyboard, which will cancel the calculation and return you to the Julia REPL prompt.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-1"></a>
<details>
  <summary><h2>Tutorial 1: Julia Intro</h2></summary>
  <hr>

Tutorial 1 of day 1 is based on the script [1-julia-intro.jl](./1-julia-intro.jl).

1. Run the script like you did as part of the [installation instructions](#installation-instructions):
```julia
julia> include("1-julia-intro.jl")
main

julia> main();
nterm = 100000
pi_approx = 3.1416026534897203
error = 9.999899927226608e-6

```

2. Get the errors as a function of iterations by running the script in a loop:
```julia
julia> nterms = [10^k for k in 3:7]
5-element Vector{Int64}:
     1000
    10000
   100000
  1000000
 10000000

julia> results = [main(; nterm, outputlevel=0) for nterm in nterms]
5-element Vector{@NamedTuple{pi_approx::Float64, error::Float64, nterm::Int64}}:
 (pi_approx = 3.1425916543395442, error = 0.0009990007497511222, nterm = 1000)
 (pi_approx = 3.1416926435905346, error = 9.99900007414567e-5, nterm = 10000)
 (pi_approx = 3.1416026534897203, error = 9.999899927226608e-6, nterm = 100000)
 (pi_approx = 3.1415936535887745, error = 9.999989813991306e-7, nterm = 1000000)
 (pi_approx = 3.1415927535897814, error = 9.999998829002266e-8, nterm = 10000000)

julia> errors = [res.error for res in results]
5-element Vector{Float64}:
 0.0009990007497511222
 9.99900007414567e-5
 9.999899927226608e-6
 9.999989813991306e-7
 9.999998829002266e-8

```
Here we suppress the printing from within the script with `outputlevel = 0`. 

3. Plot the errors in the REPL as a function of inverse number of terms to see a linear relationship:
```julia
julia> Plots.unicodeplots(); # Enable the UnicodePlots backend to plot in the terminal

julia> plot(inv.(nterms), errors; legend = false)
          ┌────────────────────────────────────────┐
0.00102897│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⠀⠀⠀⡠⠒⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⠀⠀⠀⣀⠔⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
          │⠀⡇⢀⠤⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
-2.9867e⁻⁵│⠤⡷⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│
          └────────────────────────────────────────┘
          ⠀-2.9897e⁻⁵⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀0.00103⠀

```

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-2"></a>
<details>
  <summary><h2>Tutorial 2: DMRG</h2></summary>
  <hr>

In this tutorial you will run DMRG on the 1D Heisenberg model.

1. Run the `main` function provided in the file [2-dmrg.jl](./2-dmrg.jl):
```julia
julia> include("2-dmrg.jl")
main

julia> res = main();
nsite: 30
nsweeps: 5
maxdim: [10, 20, 100, 100, 200]
cutoff: [1.0e-10]
MPO bond dimension: 5
Initial MPS bond dimension: 10
After sweep 1 energy=-13.096407053604542  maxlinkdim=10 maxerr=1.76E-03 time=7.821
After sweep 2 energy=-13.11131477426564  maxlinkdim=20 maxerr=3.09E-07 time=0.053
After sweep 3 energy=-13.111355746388746  maxlinkdim=47 maxerr=9.99E-11 time=0.101
After sweep 4 energy=-13.11135575201415  maxlinkdim=47 maxerr=9.63E-11 time=0.168
After sweep 5 energy=-13.111355752020133  maxlinkdim=47 maxerr=9.40E-11 time=0.130
Optimized MPS bond dimension: 47
DMRG energy: -13.111355752020133

```
Note that the first sweep takes a lot longer than the subsequent sweeps. This is because Julia is just-in-time compiled, so it compiles functions the first time they are run in a new Julia session, but then repeated calls to the same function with the same types of inputs don't need to be compiled again. (In future versions of ITensors.jl/ITensorMPS.jl we will automate some of that precompilation so it is performed when those packages are installed but that is a work-in-progress.)

You can see the energy converges rapidly to a fixed value with the number of sweeps. That isn't always the case for DMRG, in particular the convergence of fermionic systems and 2D systems can be much slower, depend on the initial state used, and even get stuck in local minimima if the optimization isn't performed carefully.

2. Try changing the number of sites and analyze the energy per site as a function of system size:
```julia
julia> res = main(; nsite = 40);
nsite: 40
nsweeps: 5
maxdim: [10, 20, 100, 100, 200]
cutoff: [1.0e-10]
MPO bond dimension: 5
Initial MPS bond dimension: 10
After sweep 1 energy=-17.519729740637832  maxlinkdim=10 maxerr=2.22E-03 time=0.045
After sweep 2 energy=-17.54130281611256  maxlinkdim=20 maxerr=1.20E-06 time=0.050
After sweep 3 energy=-17.541473181198704  maxlinkdim=58 maxerr=9.89E-11 time=0.134
After sweep 4 energy=-17.541473289639278  maxlinkdim=58 maxerr=9.99E-11 time=0.214
After sweep 5 energy=-17.541473289665372  maxlinkdim=58 maxerr=9.54E-11 time=0.209
Optimized MPS bond dimension: 58
DMRG energy: -17.541473289665372

julia> res.energy / res.nsite
-0.4385368322416343

julia> res = main(; nsite = 50);
nsite: 50
nsweeps: 5
maxdim: [10, 20, 100, 100, 200]
cutoff: [1.0e-10]
MPO bond dimension: 5
Initial MPS bond dimension: 10
After sweep 1 energy=-21.90903851718557  maxlinkdim=10 maxerr=1.79E-03 time=0.042
After sweep 2 energy=-21.96988537799032  maxlinkdim=20 maxerr=6.14E-07 time=0.069
After sweep 3 energy=-21.972103405447253  maxlinkdim=57 maxerr=9.99E-11 time=0.221
After sweep 4 energy=-21.97211026485518  maxlinkdim=71 maxerr=1.00E-10 time=0.687
After sweep 5 energy=-21.972110267055612  maxlinkdim=69 maxerr=1.00E-10 time=0.441
Optimized MPS bond dimension: 69
DMRG energy: -21.972110267055612

julia> res.energy / res.nsite
-0.4394422053411122

```
Note that the bond dimension DMRG needs to represent the ground state accurately increases with system size. This is because the state is gapless, which means the entanglement and correlations increase with system size. You can see that the energy per site increases with system size. That is a reflection of the fact that we are studying finite systems with open boundary conditions, and the energy hasn't converged to the thermodynamic limit yet.

3. Let's look at the energy as a function of system size:
```julia
julia> nsites = 10:10:60
10:10:60

julia> energies = [main(; nsite, outputlevel = 0).energy for nsite in nsites]
6-element Vector{Float64}:
  -4.258035206805344
  -8.682473330911792
 -13.111355752020133
 -17.541473289665372
 -21.972110267055612
 -26.40301513042684

```
Note this will take a few seconds since you are running DMRG multiple times.

We can look at the energy per site as a function of system size:
```julia
julia> energies ./ nsites
6-element Vector{Float64}:
 -0.4258035206805344
 -0.4341236665455896
 -0.4370451917340044
 -0.4385368322416343
 -0.4394422053411122
 -0.4400502521737807

```
where we use Julia's [broadcasting syntax](https://docs.julialang.org/en/v1/manual/functions/#man-vectorized) to vectorize the division operation (`/`) over the two vectors. If we plot the results we can see the energy per site converging towards something:
```julia
julia> plot(nsites, energies ./ nsites; legend = false)
         ┌────────────────────────────────────────┐
-0.425376│⠀⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠈⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠱⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠒⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠒⠤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
         │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠒⠒⠢⠤⢄⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀│
-0.440478│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠉⠑⠒⠒⠒⠤⠀│
         └────────────────────────────────────────┘
         ⠀8.5⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀61.5⠀

```
The convergence is pretty slow, since we are looking at a critical system. It is actually more helpful to look at the energy _differences_ between system sizes:
```julia
julia> diff(energies) ./ 10
5-element Vector{Float64}:
 -0.4424438124106448
 -0.4428882421108341
 -0.44301175376452395
 -0.443063697739024
 -0.4430904863371229

```
where we use the Julia [`diff`](https://docs.julialang.org/en/v1/base/arrays/#Base.diff) function to get the differences between adjacent energies and divide by `10` since the system sizes between DMRG runs differ by `10` sites. We can see these averaged differences start to approach the exact result for the energy in the thermodynamic limit from the Bethe ansatz:
```julia
julia> energy_exact = 1 / 4 - log(2)
-0.4431471805599453

julia> abs.((diff(energies) ./ 10) .- energy_exact)
5-element Vector{Float64}:
 0.0007033681493004984
 0.000258938449111179
 0.00013542679542133396
 8.3482820921299e-5
 5.6694222822395446e-5

```
which is pretty impressive considering the largest system size we ran was only 60 sites! The reason why this is more accurate is that we can think of computing energies differences as subtracting out boundary effects, and more generally can be thought of as a 1D version of a cluster expansion method such as [numerical linked cluster expansion (NLCE)](https://arxiv.org/abs/1401.3504).

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-3"></a>
<details>
  <summary><h2>Tutorial 3: DMRG Measurements</h2></summary>
  <hr>

In this tutorial you will explore measurements of MPS ground states, and use them to visualize a DMRG calculation.

1. Run the `main` function provided in the file [3-dmrg-measure.jl](./3-dmrg-measure.jl). DMRG will run and you will see a plot of the expected value of Sz on each site and the ⟨SzⱼSz⟩ correlator between the central site "j" and all other sites. As expected, you can see that the expected value of Sz on each site for the ground state is approximately zero, while the ⟨SzSz⟩ decays as a function of distance and the sign is different for even and odd distances.

```julia
julia> include("3-dmrg-measure.jl")
main

julia> res = main();
Number of sites: 30
MPO bond dimension: 5
Initial MPS bond dimension: 10
After sweep 1 energy=-13.107422228239033  maxlinkdim=10 maxerr=2.26E-03 time=0.230
After sweep 2 energy=-13.111347876235168  maxlinkdim=20 maxerr=1.40E-07 time=0.261
After sweep 3 energy=-13.111355749980916  maxlinkdim=45 maxerr=9.92E-11 time=0.439
After sweep 4 energy=-13.111355751929354  maxlinkdim=47 maxerr=9.99E-11 time=0.540
After sweep 5 energy=-13.111355751940831  maxlinkdim=47 maxerr=9.99E-11 time=0.521
Optimized MPS bond dimension: 47
Energy: -13.111355751940831
⟨ψ|ψ⟩: 1.0000000000000038
⟨ψ|H|ψ⟩: -13.111355751940852
     ┌────────────────────────────────────────┐
 0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
⟨Szⱼ⟩│⠶⠤⠶⠶⠴⠶⠶⠶⠶⠦⠴⠶⠤⠶⠦⠤⠶⠦⠴⠶⠤⠴⠶⠤⠶⠦⠤⠶⠦⠴⠶⠤⠴⠶⠤⠤⠤⠴⠶⠤│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
-0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     └────────────────────────────────────────┘
     ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site j⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀30⠀
        ┌────────────────────────────────────────┐
    0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⢻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⢸⢸⠀⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⡀⠀⡰⡀⢀⢿⠀⡸⠈⡆⢸⢇⠀⢰⡀⠀⢀⠀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
⟨SzⱼSzₖ⟩│⠵⠴⠮⠦⡤⠾⢤⠼⠼⢤⠮⢵⢴⠥⢧⢼⠼⡤⡧⠤⡧⢼⢼⠤⡮⢧⢤⠯⡦⡴⠽⡤⡴⠧⣤⠼⠦⡤⠶⠦│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠁⠀⠈⠇⠀⡇⡇⠀⡇⡜⠀⣷⠁⠈⠎⠀⠘⠁⠀⠈⠀⠀⠁⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢣⡇⠀⡇⡇⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠇⠀⢇⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   -0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        └────────────────────────────────────────┘
        ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site k⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀30⠀

```

2. Try changing the number of sites and sweeps. By saving the results `res` then passing them into the provided `animate_dmrg_sz` function, you can see a live animation or replay of the calculation! You can see that Sz starts out nonzero but quickly decays to zero as expected.

```julia
julia> res = main(; nsweeps = 4, nsite = 50);
MPO bond dimension: 5
Initial MPS bond dimension: 10
After sweep 1 energy=-21.945792072338246  maxlinkdim=10 maxerr=1.79E-03 time=0.571
After sweep 2 energy=-21.97174042201767  maxlinkdim=20 maxerr=1.08E-06 time=0.712
After sweep 3 energy=-21.97210988196606  maxlinkdim=63 maxerr=9.94E-11 time=1.356
After sweep 4 energy=-21.972110267624643  maxlinkdim=70 maxerr=9.99E-11 time=2.080
Optimized MPS bond dimension: 70
Energy: -21.972110267624643
⟨ψ|ψ⟩: 1.0000000000000064
⟨ψ|H|ψ⟩: -21.972110267625013
     ┌────────────────────────────────────────┐
 0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
⟨Szⱼ⟩│⠦⠶⠴⠦⠴⠤⠶⠴⠦⠶⠤⠦⠴⠦⠶⠤⠦⠴⠤⠶⠴⠦⠴⠤⠦⠴⠦⠶⠤⠦⠴⠦⠶⠴⠦⠴⠤⠶⠤⠦│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
-0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
     └────────────────────────────────────────┘
     ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site j⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀50⠀
        ┌────────────────────────────────────────┐
    0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠀⣿⠀⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⢀⠀⣆⢸⡇⡇⡇⣇⢰⡀⡀⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
⟨SzⱼSzₖ⟩│⠦⠶⠴⠦⠼⠦⠾⡤⢧⠼⢦⠷⡼⢧⡾⣴⢽⡼⡧⡧⣧⢿⡼⡧⡿⣴⢧⡼⣤⠷⡼⢧⠾⡤⠦⡴⢤⠴⠤⠦│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠙⠈⡇⣷⠁⣿⢸⡇⠸⠀⠃⠈⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⠀⣿⠀⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⠀⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
   -0.25│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│
        └────────────────────────────────────────┘
        ⠀1⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Site k⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀50⠀

julia> animate_dmrg_sz(res)
[...]

```

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>


<a id="tutorial-4"></a>
<details>
  <summary><h2>Tutorial 4: 2D Ising Model</h2></summary>
  <hr>

In this tutorial, you will explore 2D DMRG calculations, using the transverse-field Ising model on a square-lattice cylinder. 


1. Run the `main` function provided in the file [4-2d-tfim.jl](./4-2d-tfim.jl). DMRG will run and you will see information about each sweep.
```julia
julia> include("4-2d-tfim.jl")
main

julia> res = main();
MPO bond dimension: 5
Initial MPS bond dimension: 4
After sweep 1 energy=-223.74572376546493  maxlinkdim=2 maxerr=2.08E-02 time=12.120
After sweep 2 energy=-224.38416688578914  maxlinkdim=4 maxerr=4.31E-04 time=3.093
After sweep 3 energy=-224.79019989335737  maxlinkdim=9 maxerr=9.93E-07 time=3.096
After sweep 4 energy=-224.79244529239656  maxlinkdim=10 maxerr=9.40E-07 time=3.163
```

2. You can explore the results by displaying the expected ⟨Sz⟩ values stored in `res.szs` or expected ⟨Sx⟩ values stored in `res.sxs`. 
```julia
julia> res.szs[end]
3×30 Matrix{Float64}:
 -0.5  -0.5  -0.499999  -0.499998  -0.499994  …  -0.0423845  -0.0328509  -0.0259664  -0.0222705
 -0.5  -0.5  -0.499999  -0.499998  -0.499994     -0.0423846  -0.032851   -0.0259668  -0.0222707
 -0.5  -0.5  -0.499999  -0.499998  -0.499994     -0.0423839  -0.0328507  -0.025967   -0.0222708
```

3. Plot the results by calling `plot_spins(res)` or see an animation of the entire calculation by calling `animate_spins(res)`.
```julia

julia> Plots.gr(); # Enable the GR backend to plot in a window

julia> plot_spins(res)
[...]

julia> animate_spins(res)
[...]
```

4. Try changing some of the key parameters such as `nx`, `ny`, `h_max`, and `ramp_width`. You can see all the parameters of `main` by viewing the docstring for `main`:
```julia
julia> ?

help?> main
[...]
```

In particular, try raising `ny` to larger and larger values (staying below 20) and pass a `maxdim` array that allows the maximum bond dimension to grow, such as `maxdim=[50,100,200,400,800]`. How large of an `ny` can you do in a practical amount of time? Why do the resources needed grow?

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="stretch-goals"></a>
<details>
  <summary><h2>Stretch Goals</h2></summary>
  <hr>

If you completed all the tutorials and would like more of a challenge, you can try this "stretch goal" exploring topological physics through the spin-1 Heisenberg chain.

The S=1 version of the 1D Heisenberg chain is in a topological phase (the "Haldane phase") which is characterized by emergent S=1/2 edge states on each end and an associated four-fold ground state degeneracy. (Another model in the same phase is the exactly solvable "AKLT" model.)

We can explore this phase using ITensor DMRG by making the following changes to the Tutorial 3 file `3-dmrg-measure.jl` and re-running the main function in this file.

1. First, change the local Hilbert space type to "S=1". This is the first argument to the `siteinds` function which appears near the top of the `main` function.

2. Now, include your changed file and rerun the calculation. We recommend using `nsite = 100` and calling `main` for this part as
```julia
julia> include("3-dmrg-measure.jl")
main

julia> res = main(; nsweeps = 6, nsite = 100);
```
If the calculation is too slow, try passing a larger `cutoff` parameter such as `cutoff = 1e-6`. (As a reminder, to cancel an ongoing calculation without exiting the REPL you can execute Control-C on you keyboard.)

In the plot of ⟨Sz⟩ shown after the calculation runs, what do you notice about the shape of ⟨Sz⟩? Note that you can manually plot the Sz results with:
```julia
julia> plot_dmrg_sz(res)
```
and you can enable plotting to the REPL with `Plots.unicodeplots()` or to a window with `Plots.gr()`.

3. Now we will attempt to 'quench' one of the emergent S=1/2 edge states by placing an actual S=1/2 spin at the left edge. The idea is that the Heisenberg coupling between this spin and the edge state will form a singlet and quench any non-zero magnetization at that edge.

To make this change, after the line defining the `sites` array, prepend a `"S=1/2"` site by doing
```julia
sites = [[siteind("S=1/2")]; sites[2:end]]
```
Include your changed file and rerun DMRG. What do you notice about the shape of ⟨Sz⟩ now?

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>
