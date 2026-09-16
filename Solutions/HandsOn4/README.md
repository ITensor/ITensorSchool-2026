# Hands-On 4 Solutions

Completed versions of the files in [Tutorials/HandsOn4](../../Tutorials/HandsOn4) that are
intentionally left partially blank.

- [belief_propagation.jl](./belief_propagation.jl): the completed belief propagation
  implementation from Tutorial 2. Steps (1), (2) and (3) are filled in. Copy this file over
  `Tutorials/HandsOn4/belief_propagation.jl` to run Tutorials 3 and 4 if you get stuck.
- [1-tensornetworks.jl](./1-tensornetworks.jl): `main` with the `L` and `periodic` keyword
  arguments asked for in exercise 1 of Tutorial 1.
- [contract_network.jl](./contract_network.jl): an unmodified copy of the tutorial file of the
  same name, included here so that `belief_propagation.jl` can be loaded directly from this
  folder (for example by `Tutorials/HandsOn4/resources/make_plots.jl`).

To use a solution, copy it into `Tutorials/HandsOn4` and `include` it from there so that
the other files it depends on are found.
