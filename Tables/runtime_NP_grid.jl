using CSV, DataFrames, Interpolations, StaticArrays

function array_generator(; a0::Float64, an::Float64, n::Integer, method::String, 
    base::Float64=exp(1.0), power::Float64=2.0)

    if a0 > an || n < 2 || base <= 1.0 || power <= 0.0 
        error("Input Invalid")
    elseif method == "linear"
        array = range(a0, an; length=n)
    elseif method == "log"
        if a0 <= 0.0 || an <= 0.0
            error("For log method, a0 and an must be positive")
        end
        a0_lin = log(a0)/log(base) 
        an_lin = log(an)/log(base)
        array = base .^ range(a0_lin, an_lin; length=n)
    elseif method == "power"
        u = range(0.0, 1.0; length=n)
        array = a0 .+ (an - a0) .* (u .^ power)
    else
        error("Choose method from linear, log, or power; log has base option, power has power option")
    end
    return collect(array)
end

function array_assembler(; variable_ranges, variable_settings)

    N_dim = length(variable_ranges)

    arrays = [
        array_generator(
            a0    = float(variable_ranges[d][1]),
            an    = float(variable_ranges[d][2]),
            n     = variable_settings[d][1],
            method= variable_settings[d][2],
            base  = variable_settings[d][3],
            power = variable_settings[d][4]
        ) for d in 1:N_dim
    ]

    return vec(collect(Iterators.product(arrays...)))
end

function grid_generator_raw(; variables_grid, variable_names, target_func, target_names, precision)

    N_variables  = length(first(variables_grid))
    df = DataFrame((Symbol(variable_names[i]) => getindex.(variables_grid, i) for i in 1:N_variables)...)

    predictions = [target_func(t...) for t in variables_grid]

    predictions = map(v -> v isa Number ? (v,) : v, predictions)  

    for (j, nm) in enumerate(target_names)
        df[!, Symbol(nm)] = getindex.(predictions, j)
    end

    transform!(df, names(df, Number) .=> ByRow(x -> round(x; sigdigits=precision)) .=> names(df, Number))

    return df
end

interpolators = Dict{Symbol,Any}()

function initialize_interpolator(; df, interpolator_name, variable_names, target_names)
    # axes & values in Float32 → less memory traffic at query time
    variable_axes = map(v -> Float32.(sort(unique(df[!, v]))), variable_names)
    shape = map(length, variable_axes)
    N_targets = length(target_names)

    target_vec = Array{SVector{N_targets,Float32}}(undef, shape...)
    for row in eachrow(df)
        grid_index = ntuple(d -> searchsortedfirst(variable_axes[d], Float32(row[variable_names[d]])),
                            length(variable_names))
        target_vec[grid_index...] = SVector{N_targets,Float32}((Float32(row[t]) for t in target_names)...)
    end

    vector_interpolant = interpolate(Tuple(variable_axes), target_vec, Gridded(Linear()))
    interpolators[Symbol(interpolator_name)] = vector_interpolant
    return nothing
end

#----------------------------------------------------------------------------
# Settings for grid generation
#----------------------------------------------------------------------------

# independent variables
const grid_variable_names = ["x", "b"]

const grid_eps_x = 1e-5
const grid_eps_b = 1e-4
const grid_variable_ranges = [
    [grid_eps_x,1-grid_eps_x],
    [grid_eps_b,30.0],
]

const grid_variable_settings = [
    [250, "power", exp(1), 2.5],
    [250, "power", exp(1), 2.5],
]

# outputs

const grid_target_names = ["SNP_μ", "SNP_ζ"]

target_func(x, b) = NP_f_func(x, b)

const grid_precision = 6

const grid_variables_grid = array_assembler(variable_ranges=grid_variable_ranges, variable_settings=grid_variable_settings)

# Safe placeholder (no itp at include-time)
if !isdefined(Main, :NP_f_grid)
    @inline function NP_f_grid(x::Real, b::Real)
        throw(ArgumentError("NP_f_grid not initialized; call broadcast_NP_grid!() or install_NP_grid_everywhere!()"))
    end
end

# Build the interpolator on THIS process and return it (no globals touched)
function _build_NP_itp!()
    df_SNP = grid_generator_raw(
        variables_grid = grid_variables_grid,
        variable_names = grid_variable_names,
        target_func    = target_func,
        target_names   = grid_target_names,
        precision      = grid_precision,
    )
    initialize_interpolator(
        df                = df_SNP,
        interpolator_name = "NP_f_grid",
        variable_names    = grid_variable_names,
        target_names      = grid_target_names,
    )
    return interpolators[:NP_f_grid]
end

# Install or update NP_f_grid on THIS process; no `itp` at top-level.
function _install_or_update_NP!(itp)
    # stash in a temp global so eval'd code can see it (no $itp)
    global __np_tmp_itp__ = itp
    if !isdefined(Main, :_NP_ITP_REF)
        Core.eval(Main, :(begin
            const _NP_ITP_REF = Ref{typeof(__np_tmp_itp__)}(__np_tmp_itp__)
            @inline NP_f_grid(x::Real, b::Real) = (_NP_ITP_REF[])(x, b)
        end))
    else
        _NP_ITP_REF[] = __np_tmp_itp__   # update for new fit params (same type)
    end
    __np_tmp_itp__ = nothing
    nothing
end