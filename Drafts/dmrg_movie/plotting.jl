# All of the Plots.jl-specific code lives here, so that main.jl stays about DMRG.

using Plots
using Printf

gr()  # the GR backend renders frames fast and writes mp4 through FFMPEG

const SWEEP_COLOR = :darkorange

"""
    bond_marker!(p, bond_x; linewidth = 8)

Draw the translucent vertical bar that marks the bond currently being optimized.
`bond_x` is a half-integer, since bond `b` sits between sites `b` and `b+1`.
"""
function bond_marker!(p, bond_x; linewidth = 8)
    vline!(p, [bond_x]; color = SWEEP_COLOR, alpha = 0.4, linewidth, label = "")
    return p
end

"""
    bar_panel(y; bond_x = nothing, marker_width = 8, kwargs...)

One panel of the dashboard: a bar chart of `y` against its index, with the
current bond marked if `bond_x` is given. Everything else -- axis labels and
limits, colors -- is passed through `kwargs` to `Plots.bar` at the call site.
"""
function bar_panel(y; bond_x = nothing, marker_width = 8, kwargs...)
    p = plot(; legend = false, colorbar = false)
    isnothing(bond_x) || bond_marker!(p, bond_x; linewidth = marker_width)
    bar!(p, eachindex(y), y; label = "", kwargs...)
    return p
end

"""
    scatter_panel(y; bond_x = nothing, marker_width = 8, kwargs...)

Same idea as [`bar_panel`](@ref), but drawing `y` as markers joined by a line.
"""
function scatter_panel(y; bond_x = nothing, marker_width = 8, kwargs...)
    p = plot(; legend = false)
    isnothing(bond_x) || bond_marker!(p, bond_x; linewidth = marker_width)
    plot!(p, eachindex(y), y; marker = :circle, label = "", kwargs...)
    return p
end

"""
    trace_panel(y, n; marks = Int[], color = :crimson, kwargs...)

A panel that fills in as the calculation proceeds: the first `n` entries of `y`
drawn as a curve with a dot at its leading end. Vertical guide lines are drawn
at each index in `marks` (here, the first update of each sweep).
"""
function trace_panel(y, n; marks = Int[], color = :crimson, kwargs...)
    p = plot(; legend = false, kwargs...)
    isempty(marks) || vline!(p, marks; color = :gray, alpha = 0.35, label = "")
    plot!(p, 1:n, view(y, 1:n); color, linewidth = 2, label = "")
    scatter!(p, [n], [y[n]]; color, markersize = 5, label = "")
    return p
end

"""
    header_panel(fields; fontsize = 17)

The strip of text across the top of the dashboard. `fields` is a list of
`x => text` pairs, and each `text` is *left*-anchored at the fraction `x` of the
width, so a field stays put from frame to frame instead of sliding around as the
number in it grows a digit.
"""
function header_panel(fields; fontsize = 17)
    p = plot(; framestyle = :none, legend = false, xlims = (0, 1), ylims = (0, 1))
    for (x, text) in fields
        annotate!(p, x, 0.35, Plots.text(text, fontsize, :left, "helvetica bold"))
    end
    return p
end

"""
    movie_scales(frames)

Everything about the movie that must *not* change from frame to frame: the axis
limits shared by every frame, the energy excess curve, and where each sweep
starts. Computed once and handed to every call of [`dmrg_figure`](@ref), so that
the axes stay put while the data moves.
"""
function movie_scales(frames)
    energy = [f.energy for f in frames]
    # How far each frame still sits above the best energy DMRG ever reached
    # (floored, since the last frames are converged to machine precision).
    excess = max.(energy .- minimum(energy), 1.0e-12)
    return (;
        excess,
        sweep_start = findall(n -> n == 1 || frames[n].sweep != frames[n - 1].sweep, eachindex(frames)),
        max_entropy = 1.1 * maximum(maximum(f.entropy) for f in frames),
        max_bonddim = 1.1 * maximum(maximum(f.bonddims) for f in frames),
    )
end

"""
    dmrg_figure(frames, n, scales = movie_scales(frames))

Draw frame `n` of the movie and return the `Plots.Plot`, so rendering any single
snapshot of the calculation is just

```julia
dmrg_figure(frames, 137)
```

`frames` is a vector of snapshots taken during a DMRG calculation, each a
NamedTuple with the fields `sweep`, `half_sweep`, `bond`, `energy`, `sz`,
`entropy` and `bonddims`. `scales` holds the frame-independent axis limits; pass
the same one to every frame of a movie (see [`movie_scales`](@ref)).
"""
function dmrg_figure(frames, n, scales = movie_scales(frames))
    frame = frames[n]
    N = length(frame.sz)
    bond_x = frame.bond + 0.5   # the current bond being optimized by DMRG

    ## Text header with sweep number and global information
    header = header_panel(
        [
            0.02 => @sprintf("sweep %d %s", frame.sweep, frame.half_sweep == 1 ? "→" : "←"),
            0.24 => @sprintf("bond (%d,%d)", frame.bond, frame.bond + 1),
            0.46 => @sprintf("E = %.8f", frame.energy),
            0.79 => @sprintf("max χ = %d", maximum(frame.bonddims)),
        ]
    )

    ## Local magnetization, with the bond being optimized marked
    magnetization = bar_panel(
        frame.sz; bond_x, marker_width = 12,
        fill_z = frame.sz, color = :balance, clims = (-1, 1),
        linecolor = :black, linewidth = 1,
        xlabel = "site j", ylabel = "⟨Sᶻⱼ⟩",
        xlims = (0.3, N + 0.7), ylims = (-1.08, 1.08)
    )

    ## Entanglement entropy of each bond
    entropy = scatter_panel(
        frame.entropy; bond_x,
        color = :seagreen, markersize = 3, markerstrokewidth = 0,
        xlabel = "bond b", ylabel = "entanglement entropy",
        xlims = (0.3, N - 0.3), ylims = (0, scales.max_entropy)
    )

    ## Bond dimension of each bond
    bonddims = bar_panel(
        frame.bonddims; bond_x, color = :slateblue, linewidth = 0,
        xlabel = "bond b", ylabel = "bond dimension χ_b",
        xlims = (0.3, N - 0.3), ylims = (0, scales.max_bonddim)
    )

    ## Energy convergence trace
    convergence = trace_panel(
        scales.excess, n; marks = scales.sweep_start,
        color = :crimson, yscale = :log10,
        xlabel = "local update", ylabel = "E - E_min",
        xlims = (0, length(frames) + 1),
        ylims = (0.5 * minimum(scales.excess), 2 * maximum(scales.excess))
    )

    return plot(
        header, magnetization, entropy, bonddims, convergence;
        layout = @layout([t{0.06h}; a; b c; d]), size = (900, 720),
        left_margin = 5Plots.mm, bottom_margin = 3Plots.mm
    )
end

"""
    dmrg_movie(frames; filename="dmrg.mp4", framerate=30)

Draw every frame in order and save the result as a movie.
"""
function dmrg_movie(frames; filename = "dmrg.mp4", framerate = 30)
    scales = movie_scales(frames)
    animation = Animation()
    for n in eachindex(frames)
        Plots.frame(animation, dmrg_figure(frames, n, scales))
    end
    mp4(animation, filename; fps = framerate, show_msg = false)
    return filename
end
