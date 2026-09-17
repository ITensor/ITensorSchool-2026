# Hands-On Tutorial 3

## Table of Contents

- [Tutorial 1: Load and Plot a QTT Function](#tutorial-1)
- [Tutorial 2: Integrate a QTT Function](#tutorial-2)

To get started with today's tutorials, first make sure you are in the correct directory (`Tutorials/HandsOn3`). Once you are, activate the project for this hands-on session and instantiate the dependencies:
```julia
julia> pwd()
"[...]/ITensorSchool-2026/Tutorials/HandsOn3"

julia> ]

(@v1.13) pkg> activate .
  Activating project at `[...]/ITensorSchool-2026/Tutorials/HandsOn3`

(HandsOn3) pkg> instantiate
    Updating registry at `~/.julia/registries/General.toml`
    Updating `[...]/ITensorSchool-2026/Tutorials/HandsOn3/Project.toml`
    ...
```

<a id="tutorial-1"></a>
<details>
  <summary><h2>Tutorial 1: Load and Plot a QTT Function</h2></summary>

In this introductory tutorial, you will load and plot a one-dimensional
function encoded as an MPS in the quantics tensor train (QTT) format.

</details>

<a id="tutorial-2"></a>
<details>
  <summary><h2>Tutorial 2: Integrate a QTT Function</h2></summary>

In this tutorial, you will complete a Julia function that integrates a
one-dimensional function encoded as an MPS in the quantics tensor train (QTT)
format.

The function to be integrated is an unnormalized Cauchy distribution (or Lorentzian)
which has an integral of $\pi$ in the limit of its width $W$ approaching zero.

<p align="center">
  <img src="resources/images/2-cauchy-distribution.png" alt="Cauchy distribution" width="800">
</p>

The task of integration on $[0,1)$ can be adapted to the QTT tensor network setting 
by the following mathematical steps:

<p align="center">
  <img src="resources/images/2-qtt-integration.png" alt="Integration of a QTT" width="800">
</p>

**Your task** is to complete an implementation of code that performs the above integration
method by contracting single-index tensors with components $[1/2, 1/2]$ onto every open
index of an MPS.

1. Open the file `2-integrate.jl` and load this file using `include("2-integrate.jl");` in the Julia terminal.
Run the `main` function as `res = main();` and read the output. The initial implementation just returns the (incorrect) number 1.0 from the `integrate` function which results in a large error.

2. Read the function `integrate` at the top. The ITensor `I` is provided for you to contract with other
ITensors making up the integration diagram above, ultimately resulting in a scalar ITensor.
Add the missing line or lines of code inside the provided loop to create the vectors (black dots)
in the diagram above and contract them with each MPS tensor and accumulate (contract) the result into `I`.

To make a single-index ITensor with index `s` and elements `[a,b]`, use `ITensor([a,b], s)`.

3. Once you have a working `integrate` function, rerun `res = main();` in the Julia terminal and see if you now get a reasonable approximation of $\pi$. Adjust the width $W$ to smaller values to see how much you can improve the approximation.

4. As optional "stretch goals", try changing the function to another one you believe is challenging to integrate (e.g. highly oscillatory or multi-scale functions). Does the loading and integration process continue to work or eventually break? 

Another optional goal is to modify the `integrate` function to integrate over a different region besides $[0,1)$.

</details>
