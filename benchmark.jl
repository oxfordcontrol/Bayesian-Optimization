using LinearAlgebra
using BenchmarkTools

n = 200*200*200
x = randn(n)
y = randn(n)
z = randn(n)

function first!(z, x, y)
    a = 1.5
    b = 2.0
    return @.z = a*x + b*y
end

function second!(z, x, y)
    a = 1.5
    b = 2.0
    # copyto!(z, x)
    # rmul!(z, a)
    return axpy!(b, y, z)
end

@btime first!(z, x, y)
@btime second!(z, x, y)
nothing