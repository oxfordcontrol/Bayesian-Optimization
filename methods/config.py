from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Main
cosmo_solver = Main.include("solver_explicit.jl")
