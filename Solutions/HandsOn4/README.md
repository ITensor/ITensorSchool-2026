# Hands-On 4 Solutions

Completed versions of the files in [Tutorials/HandsOn4](../../Tutorials/HandsOn4) that are
intentionally left partially blank.

- [belief_propagation.jl](./belief_propagation.jl): the completed belief propagation
  implementation from Tutorial 2. Steps (1), (2) and (3) are filled in. Copy this file over
  `Tutorials/HandsOn4/belief_propagation.jl` to run Tutorials 3 and 4 if you get stuck.
- [1-tensornetworks.jl](./1-tensornetworks.jl): `main` with the `L` and `periodic` keyword
  arguments asked for in exercise 1 of Tutorial 1.
- [quantum_belief_propagation.jl](./quantum_belief_propagation.jl): the completed quantum
  belief propagation from the Stretch Tutorial, with steps (1), (2) and (3) filled in, followed by
  the Heisenberg bond energy and the AKLT parent Hamiltonian projector asked for at the end
  of that tutorial. Copy this file over
  `Tutorials/HandsOn4/quantum_belief_propagation.jl` to run `5-quantumbp.jl` if you get
  stuck.

Each file here loads whatever it needs from `Tutorials/HandsOn4`, so you can `include` it
from this folder as it is. The figures in the Hands-On 4 README that show the result of an
exercise are generated from these files by
[Tutorials/HandsOn4/resources/make_plots.jl](../../Tutorials/HandsOn4/resources/make_plots.jl).
