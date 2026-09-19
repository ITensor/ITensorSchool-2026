# Regenerates the diagrams in `resources/images/` from the TikZ sources in
# `resources/diagrams/`. The figures produced from the tutorial code are made by
# `make_plots.jl` instead.
#
# Needs `pdflatex` (with TikZ) and `pdftoppm` (from poppler) on the PATH. Run with
#
#     julia resources/make_diagrams.jl

const DIAGRAM_DIR = joinpath(@__DIR__, "diagrams")
const IMAGE_DIR = joinpath(@__DIR__, "images")
const DPI = 300

function make_diagram(texfile::AbstractString)
    name = splitext(basename(texfile))[1]
    mktempdir() do build
        cp(joinpath(DIAGRAM_DIR, "common.tex"), joinpath(build, "common.tex"))
        cp(texfile, joinpath(build, "$name.tex"))
        latex = `pdflatex -interaction=batchmode -halt-on-error $name.tex`
        success(Cmd(latex; dir = build)) || error("pdflatex failed on $name.tex, see $build/$name.log")
        # `-singlefile` writes exactly `<prefix>.png`
        run(`pdftoppm -png -r $DPI -singlefile $(joinpath(build, "$name.pdf")) $(joinpath(IMAGE_DIR, name))`)
    end
    println("wrote $(joinpath(IMAGE_DIR, "$name.png"))")
    return nothing
end

function make_all_diagrams()
    for tool in ("pdflatex", "pdftoppm")
        isnothing(Sys.which(tool)) && error("`$tool` not found on the PATH")
    end
    mkpath(IMAGE_DIR)
    for f in sort(readdir(DIAGRAM_DIR; join = true))
        endswith(f, ".tex") && basename(f) != "common.tex" && make_diagram(f)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_all_diagrams()
end
