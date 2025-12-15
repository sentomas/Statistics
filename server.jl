using Oxygen
using HTTP
using Distributions
using HypothesisTests
using DataFrames

# 1. Basic Study Statistics
struct StudyData
    a::Int # Case Exposed
    b::Int # Control Exposed
    c::Int # Case Unexposed
    d::Int # Control Unexposed
end

function calc_study_stats(data::StudyData)
    # Haldane Correction (add 0.5 if any cell is 0)
    if data.a == 0 || data.b == 0 || data.c == 0 || data.d == 0
        a, b, c, d = data.a + 0.5, data.b + 0.5, data.c + 0.5, data.d + 0.5
    else
        a, b, c, d = float(data.a), float(data.b), float(data.c), float(data.d)
    end

    # Odds Ratio
    or_val = (a * d) / (b * c)
    log_or = log(or_val)
    
    # Standard Error (SE)
    se = sqrt(1/a + 1/b + 1/c + 1/d)
    
    # 95% CI
    ci_lower = exp(log_or - 1.96 * se)
    ci_upper = exp(log_or + 1.96 * se)

    # Chi-Square Test (using HypothesisTests.jl)
    # Create contingency table
    t = FisherExactTest(data.a, data.b, data.c, data.d)
    p_val = pvalue(t)

    return (or=or_val, log_or=log_or, se=se, ci=(ci_lower, ci_upper), p=p_val)
end

# 2. Meta-Analysis Logic (Fixed Effects - Inverse Variance Method)
function perform_meta_analysis(studies::Vector{StudyData})
    results = map(calc_study_stats, studies)
    
    # Extract Log ORs and Variances (Variance = SE^2)
    yi = [r.log_or for r in results]
    vi = [r.se^2 for r in results]
    
    # Weights (Inverse Variance)
    wi = 1 ./ vi
    
    # Weighted Average (Pooled Log OR)
    weighted_avg = sum(wi .* yi) / sum(wi)
    
    # SE of pooled estimate
    se_pooled = sqrt(1 / sum(wi))
    
    # Convert back to OR
    pooled_or = exp(weighted_avg)
    pooled_ci_lower = exp(weighted_avg - 1.96 * se_pooled)
    pooled_ci_upper = exp(weighted_avg + 1.96 * se_pooled)
    
    return Dict(
        "Pooled_OR" => pooled_or,
        "Pooled_CI" => (pooled_ci_lower, pooled_ci_upper),
        "Method" => "Fixed Effects (Inverse Variance)"
    )
end

# API Endpoints
@post "/analyze_study" function(req::HTTP.Request)
    data = json(req, StudyData)
    return calc_study_stats(data)
end

@post "/meta_analysis" function(req::HTTP.Request)
    # Expects a list of study objects
    data = json(req, Vector{StudyData})
    return perform_meta_analysis(data)
end

# Serve
serve()