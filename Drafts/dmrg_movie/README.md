# A movie of a DMRG calculation

```
julia --project=. main.jl        # ~1 minute, writes dmrg.mp4
```

Uses the [`CallbackObserver`](../callback) from the previous demo to take a
snapshot of the MPS after *every* local update of a DMRG calculation, then
animates the snapshots with Plots.jl.

The system is a 24-site spin-1 Heisenberg chain with open ends, in the total
Sᶻ = +1 sector. Four things are plotted:

- **⟨Sᶻⱼ⟩ along the chain**, with the bond currently being optimized highlighted
  in orange, and a triangle showing which way the sweep is going.
- **Entanglement entropy of each bond** — flat in the bulk, as it should be for a
  gapped 1D ground state.
- **Bond dimension of each bond**, which DMRG grows on its own as the truncation
  cutoff demands.
- **E − E_min**, drawn as the calculation proceeds, on a log scale.

The physics to watch for: the staggered magnetization of the starting product
state melts away over the first sweep or two, leaving magnetization localized at
the two ends of the chain. Those are the spin-1/2 edge modes of the Haldane
phase — the ground state of the spin-1 chain behaves as though each end carries
half a spin, and here the two of them are locked into Sᶻ = +1 together.

## The code

- [`main.jl`](main.jl) — the physics: build the Hamiltonian, run DMRG, and
  measure ⟨Sᶻⱼ⟩ and the bond entropies in the callback. Both measurements are
  done in a single pass over the MPS, since the callback runs a few hundred
  times.
- [`plotting.jl`](plotting.jl) — all the Plots.jl code, and nothing else.

The dashboard is built out of a few reusable "instruments" — `bar_panel`,
`scatter_panel`, `trace_panel` and `header_panel` — each of which draws one
panel from a plain vector of numbers and passes everything else (labels, limits,
colors) straight through to Plots. `dmrg_figure` then only has to say which
instrument goes where:

```julia
plot(header, magnetization, entropy, bonddims, convergence;
     layout = @layout([t{0.06h}; a{0.38h}; b c; d]))
```

The line of text along the top is a panel of its own rather than a `plot_title`,
so that each field can be left-anchored at a fixed fraction of the width. That
keeps `sweep`, `bond`, `E` and `χ` in the same place all through the movie
instead of sliding back and forth as the numbers in them grow a digit.

Every frame is drawn from scratch, so rendering any single snapshot of the
calculation is just an index:

```julia
frames = ...          # collected by the callback in main.jl
dmrg_figure(frames, 137)
```

The things that must *not* jump around from frame to frame — the entropy and
bond-dimension axis ranges, and the convergence curve — are computed once by
`movie_scales` and handed to every frame, so that only the data moves.
`dmrg_movie` loops over the frames, collecting them into a `Plots.Animation`,
and writes the result out with `mp4`.
