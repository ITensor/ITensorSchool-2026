# Suppress the REPL banner (must run before the REPL starts, e.g. from `julia --load`)
let i = findfirst(==(:banner), fieldnames(Base.JLOptions))
  T = fieldtype(Base.JLOptions, i)
  p = Ptr{T}(cglobal(:jl_options, T) + fieldoffset(Base.JLOptions, i))
  unsafe_store!(p, T(0))
end
