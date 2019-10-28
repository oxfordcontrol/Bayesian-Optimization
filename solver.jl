using JuMP, SCS #, MosekTools
using LinearAlgebra, SparseArrays
include("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/COSMO_original/src/COSMO.jl")
using Main.COSMO
# using COSMO
using MathOptInterface
using Statistics
using DataStructures

function populate_upper_triangle(x)
	n = Int(1/2*(sqrt(8*length(x) + 1) - 1)) # Solution of (n^2 + n)/2 = length(x) obtained by WolframAlpha
	A = zeros(n, n)
    k = 0
    for i in 1:n, j in i:n
        k += 1
        if i != j
            A[i, j] = x[k]
        else
            A[i, j] = x[k]
        end
    end
	return (A + A')/2
end

function solve_dual_jump(omega, ymin, solver; kwargs...)
    model = Model(with_optimizer(solver; kwargs...))
    n = size(omega, 1)

    Y = [@variable(model, [1:n, 1:n], PSD) for i = 1:n]
    cost = dot(zeros(n, n), Y[1])
    for matrix_idx = 2:n
        add_to_expression!(cost, Y[matrix_idx][matrix_idx - 1, end])
        add_to_expression!(cost, -ymin*Y[matrix_idx][end, end]) 
    end
    @objective(model, Min, cost)
    for j = 1:n, i = j:n
        sum_ij = Y[1][i, j] + Y[2][i, j]
        for matrix_idx = 3:n
            add_to_expression!(sum_ij, Y[matrix_idx][i, j])
        end
        @constraint(model, sum_ij == omega[i, j])
    end
    JuMP.optimize!(model)

    return model
end

function solve_primal_jump(omega, ymin, solver; kwargs...)
    model = Model(with_optimizer(solver; kwargs...))
    n = size(omega, 1)
    @variable(model, X[1:n, 1:n], PSD)
    for i = 1:n-1
        C = spzeros(n, n)
        C[end, end] = -ymin
        C[end, i] = 1/2
        C[i, end] = 1/2
        @constraint(model, Symmetric(C + X) in PSDCone())
    end
    @objective(model, Min, dot(omega, X))
    JuMP.optimize!(model)

    return model
end
# k = 5
omegas = CircularBuffer{Matrix{Float64}}(10)
models = CircularBuffer{Main.COSMO.Workspace{Float64}}(10)
#=
push!(omegas, Matrix(1.0*I, k, k))
push!(models, 
solve_primal(omegas[end], 1.0, COSMO.Optimizer,
    verbose=true, check_termination=40,
    scaling=0, lanczos=true, max_iter=10000)[2].moi_backend.optimizer.model.optimizer.inner
)
=#

function modify_primal_problem!(model, omega, ymin)
    k = size(omega, 1)
    for i = 2:k
        idx = Int(i*k*(k + 1)/2)
        # @assert model.p.b[idx] == -ymin
        model.p.b[idx] = -ymin
    end
    counter = 1
    for j in 1:k, i in 1:k
        if j == i
            # @assert model.p.q[counter] == omega[i, j]
            model.p.q[counter] = omega[i, j]
            counter += 1
        elseif j < i
            # @assert model.p.q[counter] == 2*omega[i, j]
            model.p.q[counter] = 2*omega[i, j]
            counter += 1
        end
    end
    return model
end

function modify_dual_problem!(model, omega, ymin)
    k = size(omega, 1)
    for i = 2:k
        idx = Int(i*k*(k + 1)/2)
        # @assert model.p.q[idx] == -ymin
        model.p.q[idx] = -ymin
    end
    counter = 1
    # @show model.p.b
    for j in 1:k, i in 1:k
        if j == i
            #@assert model.p.b[counter] == omega[i, j]
            model.p.b[counter] = omega[i, j]
            counter += 1
        elseif j < i
            # @assert model.p.b[counter] == omega[i, j]
            model.p.b[counter] = omega[i, j]
            counter += 1
        end
    end
    return model
end 

function solve_primal_wrapper(omega, ymin, lanczos=true)
    k = size(omega, 1)
    if !isempty(omegas)
        min_value = Inf
        min_idx = -1
        for (index, matrix) in enumerate(omegas)
            if norm(matrix - omega) < min_value
                min_idx = index
                min_value = norm(matrix - omega)
            end
        end
        model = deepcopy(models[min_idx])
        modify_primal_problem!(model, omega, ymin)
    else
        model = solve_primal_jump(omega, ymin, COSMO.Optimizer, verbose=true,
            check_termination=40, scaling=0, lanczos=lanczos, max_iter=10000, eps_abs = 1e-4, eps_rel = 1e-4
        ).moi_backend.optimizer.model.optimizer.inner
    end
    @show model.times
    results = COSMO.optimize!(model)
    push!(omegas, omega)
    push!(models, model)

    M = populate_upper_triangle(model.vars.x)
    return dot(M, omega), M
end

function solve_dual_wrapper(omega, ymin, lanczos=true)
    k = size(omega, 1)
    if !isempty(omegas)
        min_value = Inf
        min_idx = -1
        for (index, matrix) in enumerate(omegas)
            if norm(matrix - omega) < min_value
                min_idx = index
                min_value = norm(matrix - omega)
            end
        end
        model = deepcopy(models[min_idx])
        modify_dual_problem!(model, omega, ymin)
    else
        model = solve_dual_jump(omega, ymin, COSMO.Optimizer, verbose=true,
            check_termination=40, scaling=0, lanczos=lanczos, max_iter=10000, eps_abs = 1e-4, eps_rel = 1e-4,
            adaptive_rho=true,
        ).moi_backend.optimizer.model.optimizer.inner
    end
    results = COSMO.optimize!(model)
    @show model.times
    push!(omegas, omega)
    push!(models, model)

    M = populate_upper_triangle(model.vars.μ[1:Int(k*(k + 1)/2)])
    return dot(M, omega), M
end

solve_primal_wrapper
#=
k = 200
S = Symmetric(randn(k - 1, k - 1));
S += -minimum(eigvals(S))*I;
@show eigvals(S);
S = 0*S + I
mu = randn(k - 1, 1);
mu ./= mean(mu)
omega = [S + mu*mu' mu; mu' 1.0]
opt_val, M = solve_dual_wrapper(omega, 1.0, true)
omegas = CircularBuffer{Matrix{Float64}}(5)
models = CircularBuffer{Main.COSMO.Workspace{Float64}}(5)
opt_val, M = solve_dual_wrapper(omega, 1.0, false)
=#