using Plots: Plots

"""
    animate(f; nframes::Int, fps::Int = 30)

Call `f(i)` for each frame `i in 1:nframes`, where `f(i)` returns an object to print to the
terminal at each frame. Renders each frame in-place in the terminal at `fps` frames per
second.
"""
animate(f; nframes::Int, fps::Real = 30) = animate(Plots.backend(), f; nframes, fps)

function animate(::Plots.AbstractBackend, f; nframes::Int, fps::Real)
    return try
        for i in 1:nframes
            display(f(i))
            sleep(1 / fps)
        end
    finally
        # Show cursor again
        print(stdout, "\x1b[?25h")
    end
    return nothing
end
