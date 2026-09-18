# Hands-On Tutorial 4

## Table of Contents

- [Tutorial 1: Tensor Networks](#tutorial-1)
- [Tutorial 2: Complete a Belief Propagation Implementation](#tutorial-2)
- [Tutorial 3: Belief Propagation on the 2D Ising Model](#tutorial-3)
- [Tutorial 4: BP Cluster Expansion](#tutorial-4)
- [Stretch Goals](#stretch-goals)

<a id="tutorial-1"></a>
<details>
  <summary><h2>Tutorial 1: Tensor Networks</h2></summary>
  <hr>

We are going to combine the `NamedGraphs.jl` and `ITensors.jl` packages to build tensor networks of varying topology.

To get started with today's tutorials, first make sure you are in the correct directory (`Tutorials/HandsOn4`). Once you are, activate the project for this hands-on session and instantiate the dependencies:
```julia
julia> pwd()
"[...]/ITensorSchool-2026/Tutorials/HandsOn4"

julia> readdir()
14-element Vector{String}:
 "1-tensornetworks.jl"
 "2-bp-implementation.jl"
 "3-beliefpropagation.jl"
 "4-clusterexpansion.jl"
 "5-quantumbp.jl"
[...]

julia> ]

(@v1.13) pkg> activate .
  Activating project at `[...]/ITensorSchool-2026/Tutorials/HandsOn4`

(HandsOn4) pkg> instantiate
    Updating registry at `~/.julia/registries/General.toml`
    Updating `[...]/ITensorSchool-2026/Tutorials/HandsOn4/Project.toml`
  [86223c79] + Graphs v1.15.0
  [9136182c] + ITensors v0.9.31
  [678767b0] + NamedGraphs v0.14.0
[...]
```

A simple graph `g` is just a series of vertices and edges between pairs of those vertices. There are no multiedges or self edges. The package `NamedGraphs.jl` is built around the `NamedGraph` object `g`, which can be constructed using either the pre-built graph constructors or our own via code like

```julia
julia> using Graphs: add_edge!

julia> using NamedGraphs: NamedGraph, NamedEdge

julia> g = NamedGraph([1,2,3]);

julia> es = [1 => 2, 2 => 3];

julia> for e in es
           add_edge!(g, e)
       end
```

First, let's run the script [1-tensornetworks.jl](./1-tensornetworks.jl)

```julia
julia> include("1-tensornetworks.jl")
main
```

By looking inside it you will see that it builds the 3-site path graph, which can be accessed and viewed via

```julia
julia> res = main();

julia> res.g
NamedGraph{Int64} with 3 vertices:
3-element Dictionaries.Indices{Int64}:
 1
 2
 3

and 2 edge(s):
2-element Vector{NamedEdge{Int64}}:
 1 => 2
 2 => 3
```

1. Modify the graph construction in `main()` to create a path graph on `L` vertices, where `L` is an integer variable that can be specified as a keyword argument to `main`. Compare the output to the pre-written constructor `named_path_graph(L::Int)` in `NamedGraphs.jl`. Add in a `periodic` flag to your constructor to add a periodic boundary if the flag is true.

With this you should be able to do
```julia
julia> res = main(; L = 5, periodic = true);

julia> res.g
NamedGraph{Int64} with 5 vertices:
5-element Dictionaries.Indices{Int64}:
 1
 2
 3
 4
 5

and 5 edge(s):
5-element Vector{NamedEdge{Int64}}:
 1 => 2
 1 => 5
 2 => 3
 3 => 4
 4 => 5
```

We can build a tensor network as a dictionary of tensors, one for each vertex of the `NamedGraph` `g`. The edges of the graph `g` (which are of the type `NamedEdge`) dictate which tensors share indices to be contracted over. Below is the periodic path on 5 vertices from exercise 1, and the tensor network that lives on it: one tensor $T_{v}$ per vertex, and one shared index per edge.

<p align="center">
  <img src="resources/images/1-graph_to_network.png" alt="A graph and the tensor network built on it" width="700">
</p>

Provided in [ising_tensornetwork.jl](./ising_tensornetwork.jl) is a pre-built constructor for the tensor network representing the partition function of the ferromagnetic Ising model on a given `NamedGraph` `g` at a given inverse temperature `β`. The partition function reads

$$Z(\beta) = \sum_{s_{1} \in \lbrace -1, 1\rbrace}\sum_{s_{2} \in \lbrace -1, 1\rbrace} \cdots \sum_{s_{L}\in \lbrace -1, 1\rbrace}\prod_{\langle ij \rangle}\frac{\exp(\beta s_{i}s_{j})}{2},$$

where the product runs over the edges $\langle ij \rangle$ of the graph. For convenience, the Boltzmann weight on each edge has been scaled by a factor of $1/2$. This simplifies the analytic formulas below and removes an additive constant from the free energy density.

The tensor on each vertex is a copy tensor $\delta$, which forces the spin to take the same value on every edge leaving that vertex, with the symmetric square root of the Boltzmann matrix $W_{ss'} = \tfrac{1}{2} e^{\beta s s'}$ absorbed into each of its legs. Two neighbouring tensors then share exactly one full $W$ across their common edge, so contracting the whole network sums every spin configuration with the right weight.

<p align="center">
  <img src="resources/images/1-ising_tensor.png" alt="The Ising vertex tensor as a copy tensor with square-root Boltzmann weights on its legs" width="800">
</p>

This object is returned by `main()`. You can inspect the individual tensors on each vertex of the constructed tensor network via `res.tn[v]` where `v` is the name of the vertex.
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

This tensor network can be contracted by multiplying all the tensors together. This contraction is pre-computed for you in `main()` using the function `contract_network` from [contract_network.jl](./contract_network.jl).

```julia
julia> res = main(; L = 3, periodic = false);

julia> res.z
2.081072371838455
```

In 1D the partition function of the Ising model is analytically computable for any system size L and both periodic and open boundaries. With the $1/2$ scaling on each edge the results are

$$Z_{OBC}(\beta) = 2\cosh^{L-1}(\beta)$$

for open boundaries and

$$Z_{PBC}(\beta) = \cosh^{L}(\beta) + \sinh^{L}(\beta)$$

for periodic boundaries.

2. Compare the output of `res.z` with these values for both periodic and open boundaries. Do they agree? If they do, then congratulations, you just solved the 1D PBC and OBC Ising model with a tensor network approach.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-2"></a>
<details>
  <summary><h2>Tutorial 2: Complete a Belief Propagation Implementation</h2></summary>
  <hr>

In the previous tutorial, we contracted the tensor network exactly by multiplying the tensors together, vertex by vertex. This can only be done efficiently for tree-like networks (those with no loops, or only a few) and only if care is taken over the order of contraction.

In this tutorial we are going to contract tensor networks in an efficient, but approximate manner via belief propagation (BP). The BP code lives in the file [belief_propagation.jl](./belief_propagation.jl). It is not fully implemented, and you are asked to finish implementing it. Tutorials 3 and 4 both use this file, so they will only give correct answers once you have completed it.

**The algorithm.** BP associates a *message* to every directed edge $v \to w$ of the graph. A message $m_{v \to w}$ is a vector (an `ITensor` with a single index) living on the index shared by the tensors $T_v$ and $T_w$. You can think of it as an approximation to the environment that vertex $w$ sees when it looks towards $v$, exactly like the `L` and `R` environment tensors you built in Hands-On 1 when computing an expectation value of an MPS. The messages are determined self-consistently by the update rule

$$m_{v \to w} \propto T_{v} \prod_{u \in \partial v,\, u \neq w} m_{u \to v},$$

i.e. the new message out of $v$ towards $w$ is the tensor $T_{v}$ contracted with all the messages coming *into* $v$ except the one coming from $w$.

<p align="center">
  <img src="resources/images/2-message_update.png" alt="The belief propagation message update rule" width="750">
</p>

Starting from some initial guess, all the messages are updated repeatedly until they stop changing. Once converged, the contraction of $T_{v}$ with *all* of its incoming messages gives a scalar $Z_{v}$,

<p align="center">
  <img src="resources/images/2-phi_factor.png" alt="The scalar Z_v from contracting a tensor with all of its incoming messages" width="450">
</p>

and the BP approximation to the free energy density is

$$\phi_{BP} = \frac{1}{N}\sum_{v} \ln Z_{v},$$

after the messages have been suitably normalized (that part is done for you). On a tree the messages are exactly the environments and BP is exact. On a path graph, for example, the message $m_{2 \to 3}$ is everything to the left of that edge contracted together, playing the same role as the `L` environment you built in Hands-On 1. On a graph with loops BP is an approximation.

<p align="center">
  <img src="resources/images/2-messages_are_environments.png" alt="On a path graph a message is the contraction of everything on one side of the edge" width="700">
</p>

1. First, run the `main` function provided in [2-bp-implementation.jl](./2-bp-implementation.jl). It builds the Ising tensor network on a path graph of `L` sites, runs BP on it and compares the result to exact contraction. Since a path graph is a tree the two should agree, but they do not yet because the implementation is incomplete.

```julia
julia> include("2-bp-implementation.jl")
main

julia> res = main();
BP Algorithm Converged after 1 iterations
BP free energy density:    0.0
Exact free energy density: 0.1524751716982743
BP free energy density DOES NOT agree with exact contraction (it should on a tree)
```

2. Open [belief_propagation.jl](./belief_propagation.jl). The three numbered steps `(1)`, `(2)`, `(3)` mark the missing pieces. A few things you will need:
   - `tn[v]` is the `ITensor` on vertex `v` and `messages[e]` is the message on the directed edge `e`. `src(e)` and `dst(e)` give the two ends of `e` and `reverse(e)` the edge pointing the other way.
   - `boundary_edges(g, [v]; dir = :in)` returns the directed edges of `g` pointing into `v`. `all_edges(g)` returns every directed edge (both directions for each edge of `g`).
   - `contract_network(ts::Vector)` contracts a vector of `ITensor`s together and returns the resulting `ITensor`. `normalize` rescales an `ITensor` to have unit norm. A scalar `ITensor` `t` can be converted to a number with `t[]`.

3. Step `(1)` is the function `updated_message`, which should compute the new message along the directed edge `e`. The directed edges pointing into `src(e)`, excluding the one coming from `dst(e)`, are already collected for you in `incoming_es`. Build the vector of tensors `[tn[src(e)], messages[e_1], messages[e_2], ...]`, contract it and normalize. Once you have done this run `main()` again. BP will now take more than one iteration but the answer will still be wrong.

4. Step `(2)` is the function `message_distance`, which is used to decide whether BP has converged. For each directed edge `e` in `all_edges(g)`, compute `1 - dot(messages[e], old_messages[e])^2`, which vanishes when the new and old (normalized) messages are parallel, and return the mean over all directed edges.

5. Step `(3)` is the function `phi_factor`, which should contract the tensor `tn[v]` with *all* of the messages pointing into `v` and return the resulting scalar $Z_{v}$. Once complete, `main()` should report agreement:

```julia
julia> res = main();
BP Algorithm Converged after 6 iterations
BP free energy density:    0.15247517169827435
Exact free energy density: 0.1524751716982743
BP free energy density AGREES with exact contraction (as it should on a tree)
```

Note that this pass/fail check is only meaningful on a tree, where BP is exact. The script checks that the graph is a tree before judging the result, so if you swap in a graph with loops it will just print the two numbers side by side. That is the situation we move to in the next tutorial.

6. Try different values of `L` and `beta`. How does the number of iterations BP takes to converge on a path graph depend on `L`? Why?

You can inspect the converged messages via `res.messages`, which is a dictionary keyed by `NamedEdge`. For example, on the path graph the message from vertex 1 to vertex 2 is
```julia
julia> using NamedGraphs: NamedEdge

julia> res.messages[NamedEdge(1 => 2)]
```
Compare it to the tensor `res.tn[1]`. Why are they parallel?

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-3"></a>
<details>
  <summary><h2>Tutorial 3: Belief Propagation on the 2D Ising Model</h2></summary>
  <hr>

Now that you have a working BP implementation, let's use it on graphs with loops. The function `main` in [3-beliefpropagation.jl](./3-beliefpropagation.jl) builds an $L_{x} \times L_{y}$ square grid tensor network representing the partition function of the Ising model in 2D. Inverse temperature is set via the `beta` kwarg and periodic boundaries (in both directions) can be added with the kwarg `periodic`. Returned is the number of iterations BP took to converge (`niters`), and the rescaled free energy density (`phi_bp_tn`)

$$\phi(\beta) = -\beta f(\beta) = \frac{1}{L_{x}L_{y}}\ln(Z(\beta)).$$

We can do the following to get the BP computed value for $\phi$ on a 3x1 OBC square grid. This is just a path graph, like in the previous tutorial.
```julia
julia> include("3-beliefpropagation.jl")
main

julia> res = main(; Lx = 3, Ly = 1, beta = 0.2, periodic = false);
BP Algorithm Converged after 3 iterations

julia> res.phi_bp_tn
0.24429444141332002
```
1. Compare the result to the analytical value for 1D OBC

$$\phi_{OBC}(\beta) = \frac{1}{L_{x}}\ln\left(2\cosh^{L_{x}-1}(\beta)\right).$$

They agree, even though we used BP to compute it. Why?

2. We can also get the BP approximated free energy density for a periodic ring.
```julia
julia> res = main(; Lx =  3, Ly = 1, periodic = true);
BP Algorithm Converged after 8 iterations

julia> res.phi_bp_tn
0.019868071835749606
```
Compare the result to the 1D free energy density on PBC,

$$\phi_{PBC}(\beta) = \frac{1}{L_{x}}\ln\left(\cosh^{L_{x}}(\beta) + \sinh^{L_{x}}(\beta)\right).$$

They don't agree. Why? Pick a finite value of $\beta$ between $0$ and $1$ and compute both the exact PBC free energy density vs $L_{x}$ for $L_{x} = 3, 4, \ldots, 20$ and the BP free energy density using the `main` function (set $L_{y} = 1$, `periodic = true`). You can also pass `outputlevel = 0` as a kwarg to `main` to suppress the output from running BP.

Plot the absolute error between the BP approximated $\phi$ and the exact $\phi$ as a function of $L_{x}$ on a log scale. What's the scaling? Why? The `Plots.jl` package is loaded by the script, so you can do something like

```julia
julia> plot(Lxs, bp_abs_errs; yscale = :log10, xlabel = "System size Lx", ylabel = "abs error")
```
and you should see something like the following (here $\beta = 0.2$).

<p align="center">
  <img src="resources/images/3-bp_error_1d_pbc.png" alt="BP error on a periodic ring versus system size" width="500">
</p>

Inspect the values for `phi_bp_tn` returned by `main` versus system size. Do you notice something odd? Why are they all the same value?

Now we're going to move fully into 2D. Let's compute the BP approximate free energy density on a OBC square grid with $L_{x} = L$ and $L_{y} = L$ as a function of $\beta$.

```julia
julia> betas = [0.05 * (i - 1) for i in 1:21];

julia> results = [main(; Lx = 15, Ly = 15, periodic = false, beta, outputlevel = 0) for beta in betas];

julia> phi_bps = [res.phi_bp_tn for res in results];
```

Congratulations. You just approximately solved the 2D Ising model on a 15x15 square lattice for twenty one different inverse temperatures in a matter of seconds.

3. How does the number of iterations that BP took to converge (`res.niters`) depend on the inverse temperature? Plot this. Where's the peak? Is it near the critical point $\beta_{c} = \ln(1 + \sqrt{2})/2 \approx 0.4407$ of the 2D model? Or somewhere different?

<p align="center">
  <img src="resources/images/3-bp_niters_vs_beta.png" alt="BP iterations to converge versus beta" width="500">
</p>

Included in [ising_tensornetwork.jl](./ising_tensornetwork.jl) is a function `ising_phi` for computing the exact rescaled free energy of the 2D model in the thermodynamic limit via Onsager's famous result. This is returned by `main` as `phi_exact`. With our $1/2$ rescaling of the Boltzmann weights it reads

$$\phi(\beta) = -\beta f(\beta) = -\ln 2 + \frac{1}{8\pi^{2}}\int_{0}^{2\pi}\int_{0}^{2\pi}\ln\left[\cosh^{2}\left(2\beta \right)-\sinh\left(2\beta \right)\cos\left(\theta_{1}\right)-\sinh\left(2\beta \right)\cos\left(\theta_{2}\right)\right]d\theta_{1} d\theta_{2}.$$

Let's compare our results to that.

4. Pick a small value for $\beta$ (say $\beta = 0.1$) and plot the absolute error between the BP result and the exact result as a function of graph size $L$ for $L_{x} = L$ and $L_{y} = L$ with open boundaries. How does it scale? Is this error coming from BP, or from somewhere else?

<p align="center">
  <img src="resources/images/3-bp_error_2d_obc.png" alt="BP error on an open square grid versus system size" width="500">
</p>

Now let's move to periodic boundary conditions.
```julia
julia> res = main(; Lx = 5, Ly = 5, periodic = true, beta = 0.2);
BP Algorithm Converged after 21 iterations

julia> res.phi_bp_tn, res.phi_exact
(-0.653411036960073, -0.6517635488435647)
```
5. What do you notice about the dependence of `phi_bp_tn` on $L$ with periodic boundaries?

As BP is letting us work directly in the thermodynamic limit with periodic boundaries, we can pick a small $L \geq 3$ and a fine range of betas and rapidly get the BP answer in the thermodynamic limit.

```julia
julia> betas = [0.01 * (i - 1) for i in 1:101];
```

6. Plot the absolute error between BP and Onsager's result as a function of $\beta$. Where does it peak? Compare to the location of the peak in the number of BP iterations from exercise 3.

<p align="center">
  <img src="resources/images/3-bp_error_2d_pbc.png" alt="BP error versus beta on the periodic square lattice" width="500">
</p>

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="tutorial-4"></a>
<details>
  <summary><h2>Tutorial 4: BP Cluster Expansion</h2></summary>
  <hr>

Now we are going to try to correct our BP results with a first order cluster expansion.

To first order, the correction to the partition function via a cluster expansion is a multiplicative rescaling

$$Z \approx Z_{BP} \prod_{l}Z_{l}$$

where $Z_{\rm BP}$ is the BP approximation of the partition function and the product is over the smallest loops $l$ in the lattice, with $Z_{l}$ defined as the contraction of the loop of tensors, closed off by the BP messages arriving from outside the loop.

<p align="center">
  <img src="resources/images/4-loop_correction.png" alt="A loop of four tensors closed by the messages arriving from outside it" width="750">
</p>

This formula is implemented in the function `phi_cluster_correction` in [belief_propagation.jl](./belief_propagation.jl) at the level of the rescaled free energy $\phi(\beta) = -\beta f(\beta)$, and is called by `main` in [4-clusterexpansion.jl](./4-clusterexpansion.jl). We use the `simplecycles_limited_length` function from `Graphs.jl` to enumerate the loops. Take a look at `phi_cluster_correction` and notice that it uses the `phi_factor` function you wrote in Tutorial 2.

For the periodic square lattice, setting $L \geq 5$ will give us a first order cluster expanded result for $\phi(\beta)$ directly in the thermodynamic limit. This is due to the homogeneity of the tensor network and that there is exactly one loop of size $4$ per vertex when $L \geq 5$. The parameters $L_{x} = 5, L_{y} = 5$ and `periodic = true` have all been set for you and `main` returns the BP value for `phi` (`phi_bp_tn`), the corrected value for `phi` (`phi_bp_corrected_tn`) and Onsager's exact result (`phi_exact`), all in the thermodynamic limit for your choice of $\beta$.

```julia
julia> include("4-clusterexpansion.jl")
main

julia> res = main(; beta = 0.2);
BP Algorithm Converged after 21 iterations

julia> res.phi_bp_tn, res.phi_bp_corrected_tn, res.phi_exact
(-0.653411036960073, -0.6518945381687995, -0.6517635488435647)
```

1. Calculate the BP error and the cluster corrected BP error, with respect to the exact solution, for a range of `betas`. Plot these.

```julia
julia> plot(betas, [bp_errs, bp_corrected_errs]; xlabel = "β", ylabel = "Absolute error", label = ["BP" "BP + loop correction"])
```

<p align="center">
  <img src="resources/images/4-bp_cluster_correction.png" alt="BP error and cluster corrected BP error versus beta" width="500">
</p>

2. By how much does the correction reduce the error at its peak? Where does the correction help the most and where does it help the least? What do you think happens at higher orders in the expansion?

Using cluster expanded results to improve tensor network contraction is an active research area. In October 2025 two papers appeared on the arXiv about this (https://arxiv.org/abs/2510.05647 and https://arxiv.org/abs/2510.02290) and we used the expansion written in Eq. (5) of the former, so you are now at the bleeding edge of research in this area.

This is the end of the current tutorial, continue on to the next tutorial or click [here](#table-of-contents) to return to the table of contents.

</details>

<a id="stretch-goals"></a>
<details>
  <summary><h2>Stretch Goals</h2></summary>
  <hr>

If you completed all the tutorials and would like more of a challenge, choose from among the following "stretch goal" activities.

1. You can use `named_grid((nx, ny, nz, ...); periodic)` to construct any hypercubic lattice in your choice of dimension. Try using the code to use BP (and the cluster expansion if you're feeling confident) to solve the 3D Ising model. Do you think the errors are better or worse than in 2D? Why? What about in 4D?

2. Try writing a function to construct a tensor network on a graph `g` with some bond dimension $\chi$ and random entries in the tensors. Use the function `ising_tensornetwork` in [ising_tensornetwork.jl](./ising_tensornetwork.jl) for a template. The `phi_bp` function, when passed converged `messages`, approximates the logarithm of the contraction of the given tensor network you pass it (divided by the number of vertices). Compare that result to exact contraction of the network, for which a function is provided in [contract_network.jl](./contract_network.jl).

Study how the error from the BP contraction depends on the geometry of the tensor network.

You might find it useful to know that you can import various pre-defined constructors for your favourite lattices such as
```julia
julia> using NamedGraphs: named_hexagonal_lattice_graph, named_comb_tree, named_grid

julia> g1 = named_hexagonal_lattice_graph(4, 4; periodic = true);

julia> g2 = named_comb_tree((4, 3));

julia> g3 = named_grid((4, 4, 4));
```

3. The BP implementation you wrote updates every message simultaneously from the previous set of messages (a "parallel" or Jacobi-style schedule). Modify `update_messages` so that each new message is immediately used when computing the following ones (a "sequential" or Gauss-Seidel-style schedule). How does the number of iterations to converge change? Does the order in which you visit the edges matter?

4. Everything so far has been a classical partition function. Belief propagation works just as well on a quantum state, and this stretch goal is a larger, more open one: you will write your own quantum belief propagation from a template and use it to study the AKLT state.

A tensor network state puts a tensor on every vertex carrying one extra *physical* index. Its norm $\langle \psi | \psi \rangle$ is a tensor network containing **two** tensors per vertex, the ket $\psi_{v}$ and the bra $\overline{\psi_{v}}$, joined over the physical index $s_{v}$. Every edge therefore carries two indices, the ket leg $\ell$ and the bra leg $\ell'$, so a message here is a matrix rather than a vector.

<p align="center">
  <img src="resources/images/5-norm_network.png" alt="The two-layer norm network of a tensor network state, with a matrix-valued message" width="800">
</p>

It is tempting to multiply the ket and the bra at each vertex together first, producing a single "double layer" tensor, so that the network looks exactly like the classical ones and your Tutorial 2 code runs on it untouched. Do not do this. At a vertex of degree $z$ and bond dimension $\chi$ that tensor has $2z$ virtual legs and costs $\chi^{2z}$ to store, which is by far the largest object in the calculation. Absorbing the messages into the ket one at a time and only then closing with the bra costs around $\chi^{z+1}$ instead. That order is the whole content of the message update here:

<p align="center">
  <img src="resources/images/5-lazy_contraction.png" alt="Absorb the messages into the ket first, then close with the bra" width="900">
</p>

On a degree four vertex the difference is already stark:

| Bond dimension | Double layer tensor | Absorbing messages first |
| --- | --- | --- |
| 6 | 12.8 MiB, 8.8 ms | 0.1 MiB, 0.1 ms |
| 8 | 128 MiB, 81 ms | 0.3 MiB, 0.1 ms |
| 10 | 763 MiB, 275 ms | 0.6 MiB, 0.2 ms |

Keeping the two layers apart until a message has been applied is what every serious tensor network code does, and it is the point of this exercise.

The AKLT (valence bond solid) state is provided for you in [aklt_tensornetwork.jl](./aklt_tensornetwork.jl), along with the spin operators. Every edge of the graph carries a singlet of two spin-1/2s, and at a vertex of degree $z$ those $z$ spin-1/2s are projected onto their maximal total spin $S = z/2$. On a ring every vertex has $z = 2$, so this is the spin-1 AKLT chain. On a square lattice $z = 4$ and it is the spin-2 AKLT state. The file gives you `ket_tensor(state, v)`, `ket_tensor(state, v, O)` with an operator applied, `bra_tensor(state, v)` and `spin_operators(state.sites[v])`.

The AKLT state is not just a convenient tensor network, it is the exact ground state of a physical Hamiltonian. Because each bond carries a single shared singlet, the two spins on any bond can never combine into their maximum total spin, so the state is annihilated by the projector onto that maximum. The parent Hamiltonian is the sum of those projectors,

$$H = \sum_{\langle ij \rangle} P^{(ij)}_{2S},$$

a sum of positive terms, so any state it annihilates is a ground state of energy exactly zero. For the spin-1 chain the projector onto total spin 2 is a polynomial in the Heisenberg coupling,

$$P_{2} = \frac{1}{3} + \frac{1}{2}\mathbf{S}_{i} \cdot \mathbf{S}_{j} + \frac{1}{6}\left(\mathbf{S}_{i} \cdot \mathbf{S}_{j}\right)^{2},$$

which is where the biquadratic term in the usual AKLT Hamiltonian comes from. Affleck, Kennedy, Lieb and Tasaki introduced the model in 1987 as a rigorous example of the Haldane gap, and its spin-1/2 edge states are the standard first example of a symmetry protected topological phase.

The code you have to write is in [quantum_belief_propagation.jl](./quantum_belief_propagation.jl), and the driver that exercises it is [5-quantumbp.jl](./5-quantumbp.jl), which works the same way as the numbered scripts in the tutorials above. The message passing loop, the initial messages and an exact contraction routine to check against are written for you; the three numbered steps are yours. Running the driver before you start will tell you how far off you are:

```julia
julia> include("5-quantumbp.jl")
main

julia> res = main();
BP Algorithm Converged after 1 iterations
Physical spin on vertex 2: S = 1.0
⟨(Sᶻ)²⟩ BP    = 0.0
⟨(Sᶻ)²⟩ exact = 0.6666666666666666
Quantum BP DOES NOT agree with exact contraction (it should on a tree)
```

Step (1) is `updated_message`, step (2) is `expect` for a single vertex and step (3) is `expect_bond` for a neighbouring pair, each built by absorbing messages into the ket layer before touching the bra. Note that expectation values are ratios of two contractions sharing the same messages, so the normalization cancels and nothing like `binormalized_messages` is needed. The default graph is an open path, which is a tree, so belief propagation is exact there and `main` checks it for you.

These functions share their names with the ones in `belief_propagation.jl`, but they all take the `state` as their first argument, so the two sets stay apart.

Once it passes, the physics is yours to explore. Some suggestions:

- On a ring, check $\langle S^{z} \rangle = 0$ and $\langle (S^{z})^{2} \rangle = 2/3$. Be warned that these are not a real test: the single site reduced density matrix of the AKLT state is maximally mixed by symmetry, so any method that respects the symmetry gets them right.
- Assemble the Heisenberg bond energy from your two site function, using $\mathbf{S}_{v} \cdot \mathbf{S}_{w} = S^{z}S^{z} + (S^{+}S^{-} + S^{-}S^{+})/2$, and compare to the exact thermodynamic limit value $-4/3$ for the spin-1 chain, which follows from the famous correlation function $\langle S^{z}_{i} S^{z}_{j} \rangle = \frac{4}{3}\left(-\frac{1}{3}\right)^{|i-j|}$. Compare also to exact contraction of the finite ring, for which `expect_exact` takes a pair of vertices and operators. Belief propagation gives $-4/3$ at *every* ring size, while exact contraction of the finite ring only approaches it as the ring grows, the same effect you saw for the periodic Ising chain in Tutorial 3.
- Evaluate the parent Hamiltonian itself and confirm the bond energy vanishes to machine precision. Squaring the sum of three terms above gives nine, each still a product of one operator per vertex since $(A \otimes B)(A' \otimes B') = AA' \otimes BB'$. Unlike the correlations, this is a statement about the state alone, so it holds at any ring size.
- Move to a periodic square lattice, where the state becomes the spin-2 AKLT state and belief propagation is genuinely approximate. How large is the discrepancy against exact contraction, and how does it compare to the Ising errors from Tutorial 3? Note the $P_{2}$ formula above is the spin-1 projector and no longer applies; the parent Hamiltonian there projects onto total spin 4.

This is the end of the tutorials, click [here](#table-of-contents) to return to the table of contents.

</details>
