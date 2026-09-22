"""
    print_itensor_man([io::IO])

Print an ASCII-art version of the ITensor logo ("ITensor Man")
with a yellow head, using only ANSI colors via `printstyled`.
"""
function print_itensor_man(io::IO = stdout)
  o(s) = printstyled(io, s; bold = true)                            # outline
  y(s) = printstyled(io, s; color = :light_yellow, reverse = true)  # yellow fill
  o("           \\ | /\n")
  o("         +-------+\n")
  o("         |"); y("       "); o("|\n")
  o("         |"); y("       "); o("|\n")
  o("         |"); y("       "); o("|\n")
  o("         +-------+\n")
  o("             |\n")
  o("         +-------+\n")
  o("   ------|       |------\n")
  o("         |       |\n")
  o("         |       |\n")
  o("         +-------+\n")
  o("          /     \\\n")
  o("         /       \\\n")
end
