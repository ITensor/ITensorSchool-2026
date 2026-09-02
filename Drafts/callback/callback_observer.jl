using ITensorMPS: ITensorMPS, AbstractObserver

"""
    CallbackObserver(callback; each_bond=false)

An ITensorMPS observer that hands control back to *you*: `callback` is a
function of your own choosing which gets run during a DMRG calculation.

The callback is called with the same keyword arguments that `dmrg` passes to
`measure!`:

  - `psi`                – the current MPS
  - `energy`             – energy after the most recent local update
  - `sweep`              – which sweep we are on (1-based)
  - `half_sweep`         – 1 while sweeping left-to-right, 2 right-to-left
  - `bond`               – the bond `(bond, bond+1)` that was just updated
  - `spec`               – `Spectrum` of the truncation done at that bond
  - `projected_operator` – the projected Hamiltonian (a `ProjMPO`)
  - `outputlevel`        – the `outputlevel` DMRG was called with
  - `sweep_is_done`      – `true` on the last bond update of a sweep

Take only the ones you need and end the argument list with `kwargs...`, so that
your callback keeps working if DMRG grows new keywords:

```julia
function report(; sweep, energy, psi, kwargs...)
    println("sweep \$sweep: energy = \$energy, maxlinkdim = \$(maxlinkdim(psi))")
end
energy, psi = dmrg(H, psi0; nsweeps=5, maxdim, cutoff, observer=CallbackObserver(report))
```

By default the callback runs once at the end of each sweep. Pass
`each_bond=true` to run it after every local (bond) update instead, e.g. to
watch the orthogonality center travel along the chain.

Note that `dmrg` calls a *second* observer method, `checkdone!`, to decide
whether to stop early; `CallbackObserver` leaves that at its default (never
stop early), so the callback cannot end the calculation.
"""
struct CallbackObserver{F} <: AbstractObserver
    callback::F
    each_bond::Bool
end

CallbackObserver(callback; each_bond = false) = CallbackObserver(callback, each_bond)

function ITensorMPS.measure!(observer::CallbackObserver; kwargs...)
    if observer.each_bond || kwargs[:sweep_is_done]
        observer.callback(; kwargs...)
    end
    return nothing
end
