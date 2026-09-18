# Hands-On 4 Solutions

Completed versions of the files in [Tutorials/HandsOn4](../../Tutorials/HandsOn4) that are
intentionally left partially blank.

- [belief_propagation.jl](./belief_propagation.jl): the completed belief propagation
  implementation from Tutorial 2. Steps (1), (2) and (3) are filled in. Copy this file over
  `Tutorials/HandsOn4/belief_propagation.jl` to run Tutorials 3 and 4 if you get stuck.
- [1-tensornetworks.jl](./1-tensornetworks.jl): `main` with the `L` and `periodic` keyword
  arguments asked for in exercise 1 of Tutorial 1.
- [aklt_expectations.jl](./aklt_expectations.jl): expectation values of the AKLT state from
  belief propagation, asked for in stretch goal 4. It provides `expect_bp` for a single
  vertex, `expect_bond_bp` for a pair of neighbours, `heisenberg_bond` for the bond energy,
  and a `main` that compares all of them to exact contraction of the finite network.

Each file here loads whatever it needs from `Tutorials/HandsOn4`, so you can `include` it
from this folder as it is. The figures in the Hands-On 4 README that show the result of an
exercise are generated from these files by
[Tutorials/HandsOn4/resources/make_plots.jl](../../Tutorials/HandsOn4/resources/make_plots.jl).
