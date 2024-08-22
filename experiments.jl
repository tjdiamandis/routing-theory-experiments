using Pkg
Pkg.activate(joinpath(@__DIR__))

using LinearAlgebra, Random, StatsBase, Printf
using Clarabel, JuMP


abstract type CFMM end
# ----- Uniswap edge -----
struct Uniswap{T} <: CFMM
    R::Vector{T}
    γ::T
    Ai::Vector{Int}
    q::T

    function Uniswap(R::Vector{T}, γ::T, Ai::Vector{Int}, q::T) where T <: AbstractFloat
        length(R) != 2 && ArgumentError("R must be of length 2")
        length(Ai) != 2 && ArgumentError("Ai must be of length 2")
        return new{T}(R, γ, Ai, q)
    end
end


struct Balancer{T} <: CFMM
    R::Vector{T}
    γ::T
    Ai::Vector{Int}
    w::T
    q::T

    function Balancer(R::Vector{T}, γ::T, Ai::Vector{Int}, w::T, q::T) where T <: AbstractFloat
        length(R) != 2 && ArgumentError("R must be of length 2")
        length(Ai) != 2 && ArgumentError("Ai must be of length 2")
        !(w > 0 && w < 1) && ArgumentError("w must be in (0, 1)")

        return new{T}(R, γ, Ai, w, q)
    end
end

function add_cfmm(optimizer, cfmm::Uniswap, Δ, Λ, λi)
    R, γ = cfmm.R, cfmm.γ
    ϕR = sqrt(R[1]*R[2])
    @constraint(optimizer, vcat(λi*R + γ * Δ - Λ, λi*ϕR) in MOI.PowerCone(0.5))
    @constraint(optimizer, Δ .≥ 0)
    @constraint(optimizer, Λ .≥ 0)
    return nothing
end

function add_cfmm(optimizer, cfmm::Balancer, Δ, Λ, λi)
    R, γ, w = cfmm.R, cfmm.γ, cfmm.w
    ϕR = R[1]^w*R[2]^(1-w)
    @constraint(optimizer, vcat(λi*R + γ * Δ - Λ, λi*ϕR) in MOI.PowerCone(w))
    @constraint(optimizer, Δ .≥ 0)
    @constraint(optimizer, Λ .≥ 0)
    return nothing
end

function build_pools(n_pools, n_tokens; q, rseed=1)
    cfmms = Vector{CFMM}()
    Random.seed!(rseed)

    threshold = 0.5
    rns = rand(n_pools)
    for i in 1:n_pools
        rn = rns[i] 
        γ = 0.997
        if rn ≤ threshold
            Ri = 100 * rand(2) .+ 100
            Ai = sample(collect(1:n_tokens), 2, replace=false)
            push!(cfmms, Uniswap(Ri, γ, Ai, q))
        else
            Ri = 100 * rand(2) .+ 100
            Ai = sample(collect(1:n_tokens), 2, replace=false)
            w = 0.8
            push!(cfmms, Balancer(Ri, γ, Ai, w, q))
        end
    end
    return cfmms
end


function build_model(cfmms, c, μ)
    n = length(c)
    m = length(cfmms)

    model = Model(() -> Clarabel.Optimizer())

    @variable(model, y[1:n])
    Δs = [@variable(model, [1:2]) for cfmm in cfmms]
    Λs = [@variable(model, [1:2]) for cfmm in cfmms]
    λ = @variable(model, [1:m])

    # edge constraints xᵢ = Λᵢ - Δᵢ ∈ Tᵢ
    for (i, cfmm) in enumerate(cfmms)
        add_cfmm(model, cfmm, Δs[i], Λs[i], λ[i])
    end

    # net flow constraint
    net_flow = zeros(AffExpr, n)
    for (i, cfmm) in enumerate(cfmms)
        @. net_flow[cfmm.Ai] += Λs[i] - Δs[i]
    end
    @constraint(model, y .== net_flow)

    # objective
    @constraint(model, λ .<= 1)
    @constraint(model, λ .>= 0)
    @objective(model, Max, dot(c, y) - μ * sum(abs2, y) - sum(λ[i]*cfmms[i].q for i in 1:m))

    return model, y, Δs, Λs, λ
end

function run_trial(n, q, μ; tol=1e-2)
    
    # build pools
    m = round(Int, n^2 / 4)
    cfmms = build_pools(m, n; q=q)

    # Define objective function
    Random.seed!(1)
    min_price = 1e-2
    max_price = 1.0
    c = rand(n) .* (max_price - min_price) .+ min_price

    # solve problem
    model, y, Δs, Λs, λ = build_model(cfmms, c, μ)
    set_silent(model)
    optimize!(model)

    # check termination status
    status = termination_status(model)
    status ∉ (MOI.OPTIMAL, MOI.SLOW_PROGRESS) && @info "\t\tMosek termination status: $status"
    
    # get results
    pstar = objective_value(model)
    λv = value.(λ)
    yv = value.(y)
    Δvs = [value.(Δ) for Δ in Δs]
    Λvs = [value.(Λ) for Λ in Λs]
    xs = [Λvs[i] - Δvs[i] for i in 1:m]

    λ_frac = λv[(λv .< 1.0 - tol) .& (λv .> tol)]
    violations = length(λ_frac)

    y_rounded = zeros(n)
    for (i, cfmm) in enumerate(cfmms)
        if λv[i] > tol
            @. y_rounded[cfmm.Ai] += xs[i]
        end
    end
    phat = dot(c, y_rounded) - μ * sum(abs2, y_rounded) - sum(λv[i] > tol ? cfmms[i].q : 0.0 for i in 1:m)


    rel_obj_diff = (pstar - phat) / min(pstar, phat)

    return violations, rel_obj_diff
end

ns = 10 .^ range(1, 3, length=10) .|> x -> round(Int, x)
μs = [0., 1e-4, 1e-2]
q_low = 1e-2
q_high = 1.
qs = (q_low, q_high)

violations = zeros(Int, length(qs), length(ns), length(μs))
rel_obj_diffs = zeros(length(qs), length(ns), length(μs))

for (iq, q) in enumerate(qs), (in, n) in enumerate(ns), (iμ, μ) in enumerate(μs)
    @info "Running trial for n=$n, q=$q, μ=$μ"
    violations[iq, in, iμ], rel_obj_diffs[iq, in, iμ] = run_trial(n, q, μ)
    @info "\t$(violations[iq, in, iμ]), $(round(rel_obj_diffs[iq, in, iμ], digits=4))"
end


# Violations table
println("Violations table")
println("n\tμ=0.0\tμ=1e-4\tμ=1e-2\tμ=0.0\tμ=1e-4\tμ=1e-2")
for (in, n) in enumerate(ns)
    print("$n")
    m = round(Int, n^2 / 4)
    print("&\t$m")
    for iμ in 1:length(μs)
        print("\t&$(round(Int, violations[1, in, iμ]))")
    end
    for iμ in 1:length(μs)
        print("\t&$(round(Int, violations[2, in, iμ]))")
    end
    print("\\\\")
    println()
end


# Relative objective difference table
println("Relative objective difference table")
println("n\tμ=0.0\tμ=1e-4\tμ=1e-2\tμ=0.0\tμ=1e-4\tμ=1e-2")
for (in, n) in enumerate(ns)
    print("$n")
    m = round(Int, n^2 / 4)
    print("\t&$m")
    for iμ in 1:length(μs)
        @printf("\t&\\texttt{%.2e}", abs(rel_obj_diffs[1, in, iμ]))
    end
    for iμ in 1:length(μs)
        @printf("\t&\\texttt{%.2e}", abs(rel_obj_diffs[2, in, iμ]))
    end
    print("\\\\")
    println()
end
