module CRNAnalysis

using HomotopyContinuation, Distributions, Random, LinearAlgebra

export find_stationary_points, filter_systems, analyze_stability, construct_system_from_matrices, isbistable


function find_stationary_points(F::System, n₀::Vector{Float64}, k₀::Vector{Float64}, nr_models::Int, nr_initial_concentration_sets::Int=10)
    Random.seed!(1234)
    dist = LogUniform(0.01, 10.0)

    result_nk₀ = solve(F, target_parameters = [n₀; k₀])

    # Generate nr_models sets of rate constants
    k_data = reshape(rand(dist, length(k₀) * nr_models), (nr_models, length(k₀)))
    # For each model generate nr_initial_concentration_sets sets of initial concentrations that all add up to 1. Sample them log uniformly initially
    n_data = reshape(rand(dist, length(n₀) * nr_models * nr_initial_concentration_sets), (nr_models, nr_initial_concentration_sets, length(n₀)))
    n_data = n_data ./ sum(n_data, dims=3)
    n_data = reshape(n_data, (nr_models * nr_initial_concentration_sets, length(n₀)))

    # Duplicate the rate constants for each set of initial concentrations
    k_data = repeat(k_data, inner = (nr_initial_concentration_sets, 1))
    
    # Concat the rate constants and initial concentrations along the rows
    nk_data = hcat(n_data, k_data)
    
    data_points = solve(
        F,
        solutions(result_nk₀),
        start_parameters = [n₀; k₀],
        target_parameters = eachrow(nk_data),
        transform_result = (result, params) -> (real(solutions(result, only_real=true, only_nonsingular=true)), params),
    )

    return data_points
end

function filter_systems(data_points::Vector{Tuple{Vector{Vector{Float64}}, AbstractArray{Float64}}}, tol::Float64=1e-8)::Vector{Tuple{Vector{Vector{Float64}}, AbstractArray{Float64}}}
    eligible_data_points = []
    for (solutions, params) in data_points
        # Bistable systems need at least 3 non-negative solutions (2 stable and >=1 saddle)
        nonneg_solutions = [solution for solution in solutions if all(solution .>= -tol)]
        println("Number of non-negative solutions: $(length(nonneg_solutions))")
        if length(nonneg_solutions) >= 3
            push!(eligible_data_points, (nonneg_solutions, params))
        end 
    end
    return eligible_data_points
end

function analyze_stability(F::System, data_points::Vector{<:Tuple{Vector{Vector{Float64}}, AbstractArray{Float64}}}, tol::Float64=1e-8)::Nothing
    for (sols, params) in data_points
        println("-----------------------------------")
        for sol in sols
            J = jacobian(F, sol)
            vars = parameters(F)
            J = evaluate(J, vars => params)
            λ = real(eigvals(J))
            if all(λ .< tol)
                println("Stable solution found at $(sol) with parameters $(params)")
                println("Eigenvalues: $(λ)")
            elseif all(λ .> -tol)
                println("Unstable solution found at $(sol) with parameters $(params)")
                println("Eigenvalues: $(λ)")
            else
                println("Saddle point found at $(sol) with parameters $(params)")
                println("Eigenvalues: $(λ)")
            end
        end
    end 
end

function analyze_stability(F::System, data_points::Vector{<:Tuple{Vector{Vector{Float64}}, AbstractArray{Float64}}}, nr_initial_concentration_sets::Int, len_n::Int, len_k::Int, tol::Float64=1e-8)::Vector{Dict{String, Any}}

    # Group solutions into models defined by the rate constants
    model_solutions = []
    
    for (i, (sols, params)) in enumerate(data_points)
        index = floor(Int, (i - 1e-10) / nr_initial_concentration_sets) + 1
        if i % nr_initial_concentration_sets == 1
            push!(model_solutions, Dict("k" => params[len_n + 1:end], "stable" => [], "saddle" => [], "unstable" => []))
        end
        for sol in sols
            if any(sol .< -tol)
                continue
            end
            J = jacobian(F, sol)
            vars = parameters(F)
            #print(J)
            #print(vars)
            #print(params)
            J = evaluate(J, vars => params)
            λ = real(eigvals(J))
            if all(λ .< tol)
                push!(model_solutions[index]["stable"], (sol, params[begin:len_n], λ))
            elseif all(λ .> -tol)
                push!(model_solutions[index]["unstable"], (sol, params[begin:len_n], λ))
            else
                push!(model_solutions[index]["saddle"], (sol, params[begin:len_n], λ))
            end
        end
    end 
    return model_solutions
end

function analyze_stability(F::System, sol::Vector{Float64}, params::Vector{Float64}, tol::Float64=1e-8)::Nothing
    J = jacobian(F, sol)
    J = evaluate(J, variables(J) => params[length(variables(J)):end])
    λ = real(eigvals(J))
    if all(λ .< tol)
        println("Stable solution found at $(sol) with parameters $(params)")
        println("Eigenvalues: $(λ)")
    elseif all(λ .> -tol)
        println("Unstable solution found at $(sol) with parameters $(params)")
        println("Eigenvalues: $(λ)")
    else
        println("Saddle point found at $(sol) with parameters $(params)")
        println("Eigenvalues: $(λ)")
    end
end

function construct_system_from_matrices(M::AbstractArray{Int}, c::AbstractArray{Int}, T::AbstractArray{Float64})::System
    # Number variables is number of rows in M
    M = convert(Matrix{Int}, M)
    T = convert(Matrix{Float64}, T)
    nr_species = size(M, 1) 
    x = [Variable(:x, i) for i in 1:nr_species]
    # Number of rate constants is number of columns in M (number of reactions)
    nr_k = size(M, 2) 
    k = [Variable(:k, i) for i in 1:nr_k]
    # Number of initial concentrations needed is all species || number independent species (number of species - number of rows in T)
    nr_ind_species = nr_species - size(T, 1)
    n = [Variable(:n, i) for i in 1:nr_species]

    # Multiply the rate constants onto the coefficient matrix
    c = c .* k'
    c = Vector{Expression}[eachrow(c)...]

    # Calculate constant Expression for conservation laws
    conservation_laws = T * n .- T[begin:end, begin:nr_ind_species] * x[begin:nr_ind_species]
    substitution_vars = T[begin:end, nr_ind_species + 1:end] * x[nr_ind_species + 1:end]
    substitution_vars = [variables(substitution_vars[i])[1] for i in eachindex(substitution_vars)] # this is to retrieve only the variables and not the coefficient which is always 1

    print(conservation_laws)
    print(substitution_vars)

    # First construct the whole System and then substitute the dependent species
    f = []
    for i in 1:nr_ind_species
        # Construct the polynomial for species i
        poly = poly_from_exponents_coefficients(M, c[i], x)
        # Substitute the conservation laws for the dependent species
        for j in eachindex(substitution_vars)
            poly = subs(poly, substitution_vars[j] => conservation_laws[j])
        end
        push!(f, poly)
    end

    return System(f, parameters = [n; k])
end

function isbistable(soldict)
    return length(soldict["stable"]) >= 2 && length(soldict["saddle"]) >= 1
end

end # module