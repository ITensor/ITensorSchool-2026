# CallbackObserver

DMRG in ITensorMPS lets you pass an *observer* object, and calls two of its
methods as the calculation proceeds: `measure!` after every local update, and
`checkdone!` after every sweep. Writing a new observer type for each thing you
want to watch gets old quickly.

`CallbackObserver` (in [`callback_observer.jl`](callback_observer.jl)) is an
observer that just forwards to a function you supply:

```julia
function report(; sweep, energy, psi, kwargs...)
    println("sweep $sweep: energy = $energy, maxlinkdim = $(maxlinkdim(psi))")
end

energy, psi = dmrg(H, psi0; nsweeps=5, maxdim, cutoff, observer=CallbackObserver(report))
```

Your callback receives, as keyword arguments, everything DMRG passes to
`measure!`:

| keyword | what it is |
| --- | --- |
| `psi` | the current MPS |
| `energy` | energy after the most recent local update |
| `sweep` | which sweep we are on (1-based) |
| `half_sweep` | 1 while sweeping left-to-right, 2 right-to-left |
| `bond` | the bond `(bond, bond+1)` that was just updated |
| `spec` | `Spectrum` of the truncation done at that bond |
| `projected_operator` | the projected Hamiltonian (a `ProjMPO`) |
| `outputlevel` | the `outputlevel` DMRG was called with |
| `sweep_is_done` | `true` on the last bond update of a sweep |

Take the ones you need and end the argument list with `kwargs...` so your
callback keeps working if DMRG grows new keywords.

By default the callback runs **once at the end of each sweep**. Pass
`each_bond=true` to run it after **every local update** instead, which is enough
to watch the orthogonality center travel along the chain.

`CallbackObserver` does not implement `checkdone!`, so a callback cannot stop
DMRG early; use `DMRGObserver` (or your own observer type) for that.

## Running the demo

```
julia --project=. main.jl
```

[`main.jl`](main.jl) runs DMRG on a 20-site spin-1/2 Heisenberg chain three
times, with a callback that (1) prints the energy each sweep, (2) closes over a
`history` vector and accumulates one entry per sweep, and (3) draws the sweep
position as ASCII, redrawn in place after every local update — that last one
wants a real terminal, since it relies on `\r`.

See [`../dmrg_movie`](../dmrg_movie) for a callback that turns the same
information into a movie.
