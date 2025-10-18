using QuadGK
using SpecialFunctions
using StaticArrays
using LinearAlgebra
using Serialization
using ProgressBars

@inline function DY_NP_func(; b, xp, xN, Q)::Float32

    #if b >= b_thres
    #    return 0f0
    #end

    p = if if_grid
        NP_f_grid(xp, b)
    else
        NP_f_func(xp, b)
    end
    n = if if_grid
        NP_f_grid(xN, b)
    else
        NP_f_func(xN, b)
    end

    u_p = Float32(p[1]);  z_p = Float32(p[2])
    u_n = Float32(n[1]);  z_n = Float32(n[2])

    #logQratio = log(Float32(Q) * INV_Q0_F32)
    
    bstar = bstar_func(b=b,Q=Q)
    logQratio = log(Float32(Q) * bstar/b0)

    w = exp(2f0 * (z_p + z_n) * logQratio)

    return (u_p * u_n * w)
end


function DY_integrated_xsec_table_FI(table)::Float32
    total::Float32 = 0f0
    #Main.global_isoscalarity = table[2]

    @inbounds for block_entry in table[1]
        (xp, xN, Q) = block_entry[1]
        bm          = block_entry[2]  # Matrix{Float32}, 2 columns

        # cast once per block (avoid per-row promotions)
        xp32 = Float32(xp); xN32 = Float32(xN); Q32 = Float32(Q)

        n = size(bm, 1)
        @inbounds @simd for i in 1:n
            # coeff * NP_FI(b)
            coeff = bm[i, 1]
            b     = bm[i, 2]
            np    = DY_NP_func(; b=b, xp=xp32, xN=xN32, Q=Q32)  # scalar
            total = muladd(coeff, np, total)
        end
    end

    return total
end

const table_type = Tuple{Vector{Tuple{Tuple{Float32, Float32, Float32}, Matrix{Float32}}}, Float64}
const tables = Dict{String, table_type}() 

function DY_table_read(path)
    table_path = joinpath(@__DIR__, path)
    table = open(table_path, "r") do io
        deserialize(io)::table_type
    end
    tables[path] = table
    return nothing
end

predictions = Dict{String, Float32}()

# Build once on pid 1, then install everywhere (idempotent).
function broadcast_NP_grid!()
    itp = _build_NP_itp!()                               # pid 1
    @sync for p in (1, workers()...)
        @async remotecall_wait(_install_or_update_NP!, p, itp)
    end
    nothing
end

function DY_xsec_all_pmap()

    t = @elapsed begin

    broadcast_NP_grid!()

    # ship ONLY the per-key table, not the whole dict
    pairs_vec = collect(tables)  # ::Vector{Pair{String, table_type}}

    local outs

        outs = pmap(pairs_vec; batch_size=1) do (key, table)
            key => DY_integrated_xsec_table_FI(table)  # safe: each worker has its own global
        end
    end

    predictions = Dict(outs)
    
    return predictions, t
end