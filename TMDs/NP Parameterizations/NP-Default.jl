const b0 = 1.12292
const bmax = b0

#function bstar_func(; b, Q)
#
#    #bmin = b0/Q
#
#    return(
#        bmax*(
#        (1-exp(-(b/bmax)^2))#/(1-exp(-(b/bmin)^4))
#        )^(1/2)
#    )
#end

function bstar_func(; b, Q)

   bstar = b/(1+(b/bmax)^4)^(1/4)

    return bstar
end

const xhat = 0.1f0
const Q0 = 1.0f0

#const g2 = 0.248f0
#const λ = 1.82f0
#const λ2 = 0.0215f0

#const N1 = 0.316f0
#const N2 = 0.134f0
#const N3 = 0.013f0

#const α1 = 1.29f0
#const α2 = 4.27f0
#const α3 = 4.27f0

#const σ1 = 0.68f0
#const σ2 = 0.455f0
#const σ3 = 12.71f0

@inline exp32(x::Float32) = exp(x)

function NP_f_func(x_64::Float64, b_64::Float64)

    g2 = Float32(NP_g2)
    λ  = Float32(NP_λ)
    λ2 = Float32(NP_λ2)
    N1 = Float32(NP_N1);  N2 = Float32(NP_N2);  N3 = Float32(NP_N3)
    α1 = Float32(NP_α1);  α2 = Float32(NP_α2);  α3 = Float32(NP_α3)
    σ1 = Float32(NP_σ1);  σ2 = Float32(NP_σ2);  σ3 = Float32(NP_σ3)

    denom1 = 1 / (xhat^σ1*(1-xhat)^(α1^2))
    denom2 = 1 / (xhat^σ2*(1-xhat)^(α2^2))
    denom3 = 1 / (xhat^σ3*(1-xhat)^(α3^2))

    x = Float32(x_64)
    b = Float32(b_64)

    g1x = N1*(x^σ1*(1-x)^(α1^2)) * denom1
    g2x = N2*(x^σ2*(1-x)^(α2^2)) * denom2
    g3x = N3*(x^σ3*(1-x)^(α3^2)) * denom3

    b2 = b*b

    Sudakov_num = g1x*exp32(-g1x*b2/4) + λ^2*g2x^2*(1-g2x*b2/4)*exp32(-g2x*b2/4) + λ2^2*g3x*exp32(-g3x*b2/4)
    Sudakov_denom = g1x + λ^2*g2x^2 + λ2^2*g3x

    gK = -g2^2*b2/2/(1+(b/bmax)^4)^(1/4)  #-g2^2*b2/2 #-g2^2*b2/2/sqrt(1+(b/bmax)^2) 

    SNP_μ = (Sudakov_num/Sudakov_denom)
    SNP_ζ = (gK/2)

    return SNP_μ, SNP_ζ
end

read_csv(path) = DataFrame(CSV.File(joinpath(@__DIR__, path)))
df_CS = read_csv("../../Grids/$fit_name/CS_Pert.csv")

initialize_interpolator(
    df = df_CS,
    interpolator_name = "CS_grid",
    variable_names = ["b","Q"],
    target_names = ["CS_Pert"],
)

let itp = interpolators[:CS_grid]
    global CS_grid
    @inline CS_grid(b::Real, Q::Real) = itp(b, Q)[1]
end

function CS_total_func(b, Q)

    g2 = Float32(NP_g2)

    gK = -g2^2*b^2/2/(1+(b/bmax)^4)^(1/4)
    CS = CS_grid(b, Q) + gK

    return CS
end