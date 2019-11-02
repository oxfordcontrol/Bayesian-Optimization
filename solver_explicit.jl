using JuMP
using LinearAlgebra, SparseArrays
include("../ApproximateCOSMO.jl/src/COSMO.jl")
using Main.COSMO
using DataStructures
using DataFrames, CSV

function solve_dual_cosmo(omega, ymin; lobpcg=true, kwargs...)
    k = size(omega, 1)
    psd_size = Int(k*(k + 1)/2)
    model = COSMO.Model()
    A1 = kron(ones(1, k), SparseMatrixCSC(-1.0*I, psd_size, psd_size))
    b1 = COSMO.extract_upper_triangle(-omega)
    constraints = [COSMO.Constraint(A1, b1, COSMO.ZeroSet)]
    q = zeros(psd_size*k)
    for i = 1:k
        A = [spzeros(psd_size, (i - 1)*psd_size) SparseMatrixCSC(-1.0*I, psd_size, psd_size) spzeros(psd_size, (k - i)*psd_size)]
        b = zeros(psd_size)
        
        if lobpcg
            push!(constraints, COSMO.Constraint(A, b, COSMO.PsdConeTriangleLOBPCG(size(A, 1), buffer_size=1)))
        else
            push!(constraints, COSMO.Constraint(A, b, COSMO.PsdConeTriangle))
        end

        if i > 1
            q[i*psd_size] = ymin
            q[i*psd_size - (k - i + 1)] = -1.0/sqrt(2)
        end
    end
    settings = COSMO.Settings(kkt_solver=COSMO.CustomBOSolver; kwargs...)
    COSMO.assemble!(model, spzeros(psd_size*k, psd_size*k), q, constraints, settings=settings)
    COSMO.optimize!(model)
    return model
end

df = DataFrame(Iterations = Int[], TotalTime = Float64[], ProjectionTime = Float64[], LinearSolveTime = Float64[])
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

function modify_dual_problem!(model, omega, ymin)
    b1 = COSMO.extract_upper_triangle(-omega)
    psd_size = length(b1)
    model.p.b[1:psd_size] .= b1

    k = size(omega, 1)
    for i = 2:k
        model.p.q[i*psd_size] = ymin
        model.p.q[i*psd_size - (k - i + 1)] = -1.0/sqrt(2)
    end

    return model
end 


function solve_dual_wrapper(omega, ymin; lobpcg=true)
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
        model = solve_dual_cosmo(omega, ymin, lobpcg=lobpcg, verbose=true,
            check_termination=40, scaling=0, max_iter=10000, eps_abs = 1e-4, eps_rel = 1e-4,
            adaptive_rho=true,
        )
    end
    results = COSMO.optimize!(model)
    # @show model.times
    push!(omegas, omega)
    push!(models, model)

    push!(df, [model.iterations, model.times.solver_time, model.times.proj_time, model.times.sol_time])
    lobpcg_string = lobpcg ? "lobpcg" : "exact"
    df |> CSV.write(string("timings_", k - 1, "_", lobpcg_string, ".csv"))

    M = COSMO.populate_upper_triangle(model.vars.μ[end-Int(k*(k + 1)/2)+1:end])
    return dot(M, omega), M
end

(omega, ymin) -> solve_dual_wrapper(omega, ymin, lobpcg=true)
#=
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

function extract_primal_solution_jump(x)
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

k = 4
S = Symmetric(randn(k - 1, k - 1));
S += -minimum(eigvals(S))*I;
@show eigvals(S);
S = S + I
mu = randn(k - 1, 1);
mu ./= sum(mu)/length(mu)
omega = [S + mu*mu' mu; mu' 1.0]
model_jump = solve_dual_jump(omega, 1.0, COSMO.Optimizer).moi_backend.optimizer.model.optimizer.inner
model_pure = solve_dual_cosmo(omega, 1.0, verbose=true)
show(stdout, "text/plain", extract_primal_solution_jump(model_jump.vars.μ[1:Int(k*(k + 1)/2)])); println()
show(stdout, "text/plain", COSMO.populate_upper_triangle(model_pure.vars.μ[end-Int(k*(k + 1)/2)+1:end])); println()
=#