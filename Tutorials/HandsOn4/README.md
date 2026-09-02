# Day 3 Hands-On Tutorials

## Table of Contents

- [Tutorial 1: Tensor Networks](#tutorial-1)
- [Tutorial 2: Belief Propagation](#tutorial-2)
- [Tutorial 3: BP Cluster Expansion](#tutorial-3)
- [Stretch Goals](#stretch-goals)

<a id="tutorial-1"></a>
<details>
  <summary><h2>Tutorial 1: Tensor Networks</h2></summary>
  <hr>

We are going to combine the `NamedGraphs.jl` and `ITensors.jl` packages to build tensor networks of varying topology. 

To get started with today's tutorials, first make sure you are in the correct directory (`Tutorials/Day3`). Once you are, activate the project for the day and instantiate the dependencies:
```julia
julia> pwd()
"[...]/ITensorCCQSchool/Tutorials/Day3"

julia> readdir()
8-element Vector{String}:
 "1-tensornetworks.jl"
 "2-beliefpropagation.jl"
 "3-clusterexpansion.jl"
[...]

julia> ]

(@v1.12) pkg> activate .
  Activating project at `[...]/ITensorCCQSchool/Tutorials/Day3`

(Day3) pkg> instantiate
    Updating registry at `~/.julia/registries/General.toml`
    Updating `[...]/ITensorCCQSchool/Tutorials/Day3/Project.toml`
  [86223c79] + Graphs v1.13.1
  [9136182c] + ITensors v0.9.14
  [678767b0] + NamedGraphs v0.7.3
[...]
```

A simple graph `g` is just a series of vertices and edges between pairs of those vertices. There are no multiedges or self edges. The package `NamedGraphs.jl` is built around the `NamedGraph` object `g`, which can be constructed using either the pre-built graph constructors or our own via code like 

```julia
julia> using NamedGraphs: NamedGraph, NamedEdge

julia> g = NamedGraph([1,2,3]);

julia> es = [1 => 2, 2 => 3];

julia> for e in es
           add_edge!(g, e)
       end
```

First, lets run the  script [1-tensornetworks.jl](./1-tensornetworks.jl)

```julia
julia> include("1-tensornetworks.jl")
main
```

By looking inside it you will see that it builds the 3-site path graph, which can be accessed and viewed via

```julia
julia> res = main();

julia> res.g
NamedGraph{Int64} with 3 vertices:
3-element NamedGraphs.OrderedDictionaries.OrderedIndices{Int64}:
 1
 2
 3

and 2 edge(s):
1 => 2
2 => 3
```

1: Modify the graph construction in `main()` to create a path graph on `L` vertices, where `L` is an integer variable that can be specified as a keyword argument to main. Compare the output to the pre-written constructor `named_path_graph(L::Int)` in `NamedGraphs.jl`. Add in a `periodic` flag to your constructor to add a periodic boundary if the flag is true.
 
With this you should be able to do
```julia
julia> res = main(; L = 5, periodic = true);

julia> res.g
NamedGraph{Int64} with 5 vertices:
5-element NamedGraphs.OrderedDictionaries.OrderedIndices{Int64}:
 1
 2
 3
 4
 5

and 5 edge(s):
1 => 2
1 => 5
2 => 3
3 => 4
4 => 5
```

We can build a tensor network as a dictionary of tensors, one for each vertex of the `NamedGraph` `g`. The edges of the graph `g` (which are of the  type `NamedEdge`) dictate which tensors share indices to be contracted over. 

Provided in [ising_tensornetwork.jl](./ising_tensornetwork.jl) is a pre-built constructor for the tensor network representing the partition function of the ising model on a given `NamedGraph` g at a given inverse temperature `β`. The partition function reads 

$$Z(\beta) = \frac{1}{2}\sum_{s_{1} \in {-1, 1}}\sum_{s_{2} \in {-1, 1}} ... \sum_{s_{L}\in {-1, 1}}\exp(-\beta \sum_{ij}s_{i}.s_{j}),$$

where we have scaled by a factor of 1/2 for convenience.

This object is returned by `main()`.You can inspect the individual tensors on each vertex of the constructed tensor network via `res.tn[v]` where `v` is the name of the vertex.
```julia
julia> res = main(; L = 3, periodic = false, beta = 0.2);

julia> show(res.tn[1])
ITensor ord=1
Dim 1: (dim=2|id=290|"e1_2")
NDTensors.Dense{Float64, Vector{Float64}}
 2-element
 1.0099835422515933
 1.0099835422515933
```

This tensor network can be contracted by multiplying all the tensors together. This contraction is pre-computed for you in `main()`

```julia
julia> res = main(; n = 3, periodic = false);

julia> res.z
2.081072371838455
```

In 1D the partition function of the Ising model is analytically computable for any system size L and both Periodic and Open Boundaries. The results are

$$Z_{OBC}(\beta) = 2\cosh^{L-1}(\beta)$$

for open boundaries and

$$Z_{PBC}(\beta) = \cosh^{L}(\beta) + \sinh^{L}(\beta)$$

for periodic boundaries.

2. Compare the output of `res.z` with these values for both periodic and open boundaries. Do they agree? If they do, then congratulations, you just solved the 1D PBC and OBC Ising model with a tensor network approach.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-2"></a>
<details>
  <summary><h2>Tutorial 2: Belief Propagation</h2></summary>
  <hr>

In the previous tutorial, we contracted the tensor network exactly by multiplying the tensors together, vertex by vertex. This can only be done efficiently for tree-like networks (those composed of no loops, or a small number of loops) and only when taking careful care over the order of contraction.

In this tutorial we are going to contract tensor networks in an efficient, but approximate manner via belief propagation. The core belief propagation functions are contained in the script [belief_propagation.jl](./beliefpropagationfunctions.jl).

The function `main` in [2-beliefpropagation.jl](./2-beliefpropagation.jl) now builds an $L_{x} \times L_{y}$ square grid tensornetwork representing the partition function of the Ising model in 2D. Inverse temperature is set via the `beta` kwarg and periodic boundaries (in both directions) can be added with the kwarg `periodic`. Returned is the number of iterations BP took to converge (`niters`), and the rescaled free energy density (`phi_bp_tn`)

$$\phi(\beta) = -\beta f(\beta) = \frac{1}{L_{x}L_{y}}\ln(Z(\beta))$$

We can do the following to get the BP computed value for $\phi$ on a 10x1 OBC square grid. This is just a path graph, like in the previous example.
```julia
julia> include("2-beliefpropagation.jl")
main

julia> res = main(; Lx = 3, Ly = 1, beta = 0.2, periodic = false);
BP Algorithm Converged after 3 iterations

julia> res.phi_bp_tn
0.24429444141332002
```
1. Compare the result to the analytical value for 1D OBC

$$\phi_{OBC}(\beta) = \frac{1}{L_{x}}\ln(2\cosh^{Lx-1}(\beta))$$

They agree, even though we used BP to compute it. Why?

2. We can also get the bp approximated free energy density for a periodic ring `g`. 
```julia
julia> res = main(; Lx =  3, Ly = 1, periodic = true);
BP Algorithm Converged after 8 iterations

julia> res.phi_bp_tn
0.019868071835749606
```

2. Compare the result to the 1D scaled free energy density on PBC, 

$$\phi_{OBC}(\beta) = \frac{1}{L_{x}}\ln(\cosh^{Lx}(\beta) + \sinh^{L_{x}}(\beta))$$

They don't agree. Why? Pick a finite value of $\beta$ between $0$ and $1$ and compute both the exact PBC free energy vs $Lx$ for $Lx = 3,4,...20$ and the `bp` free energy using the `main` function (set $Ly = 1$, `periodic = true`). You can also pass `outputlevel=0` as a `kwarg` to `main` to suppress the output from running BP.

Plot the error between the bp approximated `phi` and
the exact `phi` as a function of $L_{x}$ on a log scale. What's the scaling? Why?

```julia
julia> Plots.unicodeplots(); # Enable the UnicodePlots backend to plot in the terminal

julia> plot(Lxs, bp_abs_errs; yscale = :log, xlabel = "System Size Lx", ylabel = "abs error")
          ┌────────────────────────────────────────┐  
10⁻²⸱³²⁹⁵⁷│⠀⢢⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│y1
          │⠀⠀⠑⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠉⢆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
 abs error│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠢⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
          │⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
10⁻¹¹⸱⁶³⁴¹│⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠑⠒⠢⠤⠤⠤⠤⠤⠤⠤⠤⠤⠀│  
          └────────────────────────────────────────┘  
          ⠀2.49⠀⠀⠀⠀⠀⠀⠀⠀System Size Lx⠀⠀⠀⠀⠀⠀⠀⠀⠀20.51⠀  
```

Inspect the values for `phi_bp_tn` returned by `main` versus system size? Do you notice something odd? Why are they all the same value?

Now we're going to move fully into 2D. Let's compute the BP approximate free energy density on a OBC square grid with $L_{x} = L$ and $L_{y} = L$ as a function of $\beta$.

```julia
julia> betas = [0.05 * (i - 1) for i in 1:21]

julia> phi_bps = [main(; Lx = 15, Ly = 15, periodic = false, beta, outputlevel = 0).phi_bp_tn for beta in betas]
```

Congratulations. You just approximately solved the 2D Ising model on a 15x15 square lattice for twenty different inverse temperatures in about 10 seconds.

3. How does the number of iterations that BP took to converge depend on the inverse temperature? Plot this. Where's the peak? Is it near the critical point of the 2D model? Or somewhere different?

Included in `[2-beliefpropagation.jl](./2-beliefpropagation.jl)` is a function for computing the exact rescaled free energy of the 2D model in the thermodynamic limit via Onsager's famous result. This is returned by `main` as `phi_exact`.

$$\phi(\beta) = -\beta f(\beta) = -\ln 2 + \frac{1}{8\pi^{2}}\int_{0}^{2\pi}\int_{0}^{2\pi}\ln\left[\cosh\left(2\beta \right)\cosh\left(2\beta \right)-\sinh\left(2\beta \right)\cos\left(\theta_{1}\right)-\sinh\left(2\beta \right)\cos\left(\theta_{2}\right)\right]d\theta_{1}, d\theta_{2}.$$

Lets compare our results to that.

5. Pick a small value for $\beta$ (say $\beta = 0.1$) and plot the error between `bp` and the `exact` result as a function of graph size $L$ for $L_{x} = L$ and $L_{y} = L$. How does it scale?

Now lets move to periodic boundary conditions. 
```julia
julia> res = main(; Lx = 5, Ly = 5, periodic = true, beta = 0.2)
BP Algorithm Converged after 21 iterations
(phi_bp_tn = -0.6534110369600732, phi_exact = -0.6517635488435647, niters = 21)
```
6. What do you notice about the dependence of `phi_bp_tn` on $L$?


As BP is letting us work directly in the thermodynamic limit with periodic boundaries, we can pick a small $L >= 3$ and a fine-range of betas and rapidly get the BP answer in the thermodynamic limit.

```julia
julia> betas = [0.01 * (i - 1) for i in 1:101]
```

7. Plot the absolute error between BP and Onsager's result. Where does it peak? 

```julia
julia> plot(betas, abs_errs, xlabel = "Beta", ylabel = "Abs Error")
            ┌────────────────────────────────────────┐  
   0.0181705│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│y1
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
   Abs Error│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠸⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠘⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⠀⠀⠀⠀⠀⢣⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
            │⠀⡇⠀⠀⠀⠀⠀⢠⠊⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│  
-0.000550323│⠤⡧⠤⠤⠴⠶⠯⠥⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠬⠽⠶⠶⠶⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│  
            └────────────────────────────────────────┘  
            ⠀-0.03⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Beta⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀1.03⠀ 
```

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-3"></a>
<details>
  <summary><h2>Tutorial 3: BP Cluster Expansion</h2></summary>
  <hr>

Now we are going to try to correct our BP results with a first order cluster expansion.

To first order, the correction to the partition function via a cluster expansion is a multiplicative rescaling

$$Z \approx Z_{BP} \prod_{l}Z_{l}$$

where $Z_{\rm BP}$ is the BP approximation of the partition function and the product is over the smallest loops $l$ in the lattice, with $Z_{l}$ defined as the contraction of the loop of tensors, with bp messages incident to it.

This formula is implemented in `[3-clusterexpansion.jl](./3-clusterexpansion.jl)` at the level of the rescaled free energy $\phi(\beta) = -\beta f(\beta)$. We use the `NamedGraphs.simple_cycles_limited_length` function to enumerate these loops. 

For the periodic square lattice, setting $L >= 5$ will give us a first order cluster expanded result for $\phi(\beta)$ directly in the thermodynamic limit. This is due to the homogenity of the tensor network and that there is exactly one loop of size $4$ per vertex when $L >= 5$. The parameters $L_{x} = 5, L_{y} = 5$ and `periodic = true` have all been set for you and `main` returns the bp value for `phi` (`phi_bp_tn`), the corrected value for `phi` (`phi_bp_corrected_tn`) and Onsager's exact result (`phi_exact`) - all in the thermodynamic limit for your choice of $\beta$.

8. Calculate the bp error and the cluster corrected bp error, with respect to the exact solution, for a range of `betas`. Plot these.

```julia
julia> plot(betas, [bp_errs, bp_corrected_errs], xlabel = "Beta", ylabel = "Absolute Err", label = ["BP Err", "BP_Corrected_Err"])
            ┌────────────────────────────────────────┐                
     0.01817│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡰⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│BP Err          
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│BP_Corrected_Err
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⢱⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⡷⣸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
Absolute Err│⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⢸⠀⡿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⠇⠀⢸⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⡸⠀⠀⡎⠀⢸⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⢠⠃⠀⠀⡇⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⢀⠇⠀⠀⢻⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⠀⠀⡜⠀⠀⠀⡞⠀⠀⠀⠸⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
            │⠀⡇⠀⠀⠀⠀⠀⢠⠔⠁⠀⣀⠎⠀⠀⠀⠀⠀⢇⠣⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀│                
-0.000532222│⠤⡧⠤⠤⠴⠶⠮⠥⠶⠶⠮⠤⠤⠤⠤⠤⠤⠤⠤⠭⠾⠿⠶⠶⠦⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤⠤│                
            └────────────────────────────────────────┘                
            ⠀-0.03⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀Beta⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀1.03⠀     
```

Using cluster expanded results to improve tensor network contraction is an active research area. Just last week two papers appeared on the arXiv about this (https://arxiv.org/abs/2510.05647 and https://arxiv.org/abs/2510.02290) and we used the expansion written in Eq. (5) of the former - so you are now at the bleeding edge of research in this area.


This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="stretch-goals"></a>
<details>
  <summary><h2>Stretch Goals</h2></summary>
  <hr>

If you completed all the tutorials and would like more of a challenge, choose from among the following "stretch goal" activities.

1. You can use the `named_grid((nx,ny,nz,...); periodic)` to construct any hypercubic lattice in your choice of dimension. Try using the code to use BP (and the cluster expansion if you're feeling confident) to solve the 3D Ising model. Do you think the errors are better or worse than in 2D? Why? What about in 4D?

2. Try writing a function to construct a tensornetwork on a graph `g` with some bond dimension $\chi$ and random entries in the tensors. Use the function `ising_tensornetwork` in `[ising_tensornetwork.jl](./ising_tensornetwork.jl)` for a template. The `phi_bp` function, when passed converged `messages`, approximates the logarithm of the contraction of the given tensornetwork you pass it. Compare that result to exact contraction of the network, for which a function is provided in `[contract_network.jl](./contract_network.jl)`

Study how the error from the BP contraction depends on the geometry of the tensor network.

You might find it useful to know that you can import various pre-defined constructors for your favourite lattices such as
```julia
julia> using NamedGraphs.NamedGraphGenerators
g1 = named_hexagonal_lattice_graph(4,4; periodic = true)
g2 = named_comb_tree((4,3))
g3 = named_grid((4,4,4))
```

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>