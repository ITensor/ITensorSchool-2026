# Hands-On 2 Solutions

Completed versions of the files in [Tutorials/HandsOn2](../../Tutorials/HandsOn2) whose
exercises ask you to add code.

- [1-tebd-implementation.jl](./1-tebd-implementation.jl): the completed `tebd_step` from
  Tutorial 1, with steps (1), (2) and (3) filled in. Copy this file over
  `Tutorials/HandsOn2/1-tebd-implementation.jl` to run Tutorials 2 to 4 if you get stuck.
- [3-sample-implementation.jl](./3-sample-implementation.jl): the completed `sample_state` from
  Tutorial 3, with steps (1), (2) and (3) filled in, followed by the answers to the two
  questions at the end of that tutorial: a `sample_state` that draws without computing every
  probability, and a `sample_states` that contracts the environments once for all of the states it
  draws. Copy this file over `Tutorials/HandsOn2/3-sample-implementation.jl` to run Tutorial 4
  if you get stuck.
- [4-metts.jl](./4-metts.jl): `main` also records the mean square energy of each METTS as
  `square_energies`, and the file defines the `specific_heat(res)` function asked for in
  Tutorial 4.

Each file here loads whatever it needs from `Tutorials/HandsOn2`, so you can `include` it
from this folder as it is. The figures in the Hands-On 2 README that show the result of an
exercise are generated from these files by
[Tutorials/HandsOn2/resources/make_plots.jl](../../Tutorials/HandsOn2/resources/make_plots.jl).
