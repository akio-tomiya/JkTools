module JkTools

using Statistics  # Provides functions for statistical calculations

# Functions to export
export jk_index_set, jk_index, jk_meanerror
export jk_block_index_set, jk_block_index, jk_block_meanerror

#=
# Example Usage:
data = [12.3, 15.6, 14.2, 11.8, 13.7, 16.4, 14.8, 13.1, 12.9, 15.2]
# (14.0, 0.47469288317114383)  # Expected output for mean and standard error
jk_meanerror(data)
jk_meanerror(data,"mean") 
=#

"""
    jk_index(input_data::AbstractVector) -> Vector{Vector{Int}}

Generates Jackknife index subsets directly from the input data.

# Arguments
- `input_data::AbstractVector`: A one-dimensional collection of input data.

# Returns
- `Vector{Vector{Int}}`: A list of subsets of indices for Jackknife resampling, each excluding one element at a time.

# Notes
This function combines the length of the input data with `jk_index_set` to generate the subsets.
"""
function jk_index(input_data::AbstractVector)
    Ndat = length(input_data)
    index = 1:Ndat
    return jk_index_set(index)
end

"""
    jk_index_set(index::AbstractVector{<:Integer}) -> Vector{Vector{Int}}

Generates subsets of indices for Jackknife resampling from the given array of indices.

# Arguments
- `index::AbstractVector{<:Integer}`: A vector containing the indices of the original data.

# Returns
- `Vector{Vector{Int}}`: A list of subsets of indices, each excluding one element at a time.
"""
function jk_index_set(index::AbstractVector{<:Integer})
    N = length(index)  # Total number of indices
    subsets = Vector{Vector{Int}}()  # Container for subsets
    sizehint!(subsets, N)

    # Create subsets by excluding one index at a time
    for i in 1:N
        subset = Vector{Int}(vcat(index[1:i-1], index[i+1:end]))
        push!(subsets, subset)
    end
    return subsets
end

"""
    jk_index_set(index::UnitRange{Int}) -> Vector{Vector{Int}}

Generates subsets of indices for Jackknife resampling from the given range of indices.

# Arguments
- `index::UnitRange{Int}`: A range of indices for the original data.

# Returns
- `Vector{Vector{Int}}`: A list of subsets of indices, each excluding one element at a time.
"""
function jk_index_set(index::UnitRange{Int})
    return jk_index_set(collect(index))  # Convert range to vector and reuse the function
end

function _observable_function(KEY::String)
    return if KEY in ["mean", "average"]
        mean
    elseif KEY in ["sus", "susceptibility"]
        x -> var(x, corrected=false)
    elseif KEY in ["binder", "bin"]
        x -> mean(x .^ 4) / mean(x .^ 2)^2
    else
        error("Unsupported KEY: $KEY")
    end
end

function _validate_jackknife_input(input_data::AbstractVector)
    length(input_data) >= 2 || throw(ArgumentError("jackknife error requires at least two data points"))
    return nothing
end

function _jackknife_meanerror(input_data::AbstractVector{<:Real}, statistic::Function)
    _validate_jackknife_input(input_data)

    subsets = jk_index_set(1:length(input_data))
    NJK = length(subsets)
    JK_values = Float64[]
    sizehint!(JK_values, NJK)

    for subset in subsets
        sample = input_data[subset]
        push!(JK_values, statistic(sample))
    end

    return statistic(input_data), std(JK_values, corrected=false) * sqrt(NJK - 1)
end

"""
    jk_meanerror(input_data::AbstractVector{<:Real}) -> Tuple{Real, Float64}

Calculates the mean and Jackknife error for the given data array.

# Arguments
- `input_data::AbstractVector{<:Real}`: A one-dimensional array of input data.

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error of the input data.
"""
function jk_meanerror(input_data::AbstractVector{<:Real})
    return _jackknife_meanerror(input_data, mean)
end

"""
    jk_meanerror(input_data::AbstractVector{<:Real}, func::Function) -> Tuple{Real, Float64}

Calculates the mean and Jackknife error for the given data array using a custom function.

# Arguments
- `input_data::AbstractVector{<:Real}`: A one-dimensional array of input data.
- `func::Function`: A function that maps a sample vector to one statistical value
  (for example, `x -> mean(x .^ 2)`).

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error of the computed values.
"""
function jk_meanerror(input_data::AbstractVector{<:Real}, func::Function)
    return _jackknife_meanerror(input_data, func)
end

"""
    jk_meanerror(input_data::AbstractVector{<:Real}, KEY::String) -> Tuple{Real, Float64}

Calculates the mean and Jackknife error for statistical observables based on a specified keyword.

# Arguments
- `input_data::AbstractVector{<:Real}`: A one-dimensional array of input data.
- `KEY::String`: A keyword specifying the observable:
  - `"mean"` or `"average"`: Mean value.
  - `"sus"` or `"susceptibility"`: Variance (susceptibility).
  - `"binder"` or `"bin"`: Binder cumulant.

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error for the specified observable.

# Notes
An error is raised if an unsupported `KEY` is provided.
"""
function jk_meanerror(input_data::AbstractVector{<:Real}, KEY::String)
    return _jackknife_meanerror(input_data, _observable_function(KEY))
end

function _block_first_kept(n::Integer, block_size::Integer)
    block_size >= 1 || throw(ArgumentError("block_size must be positive"))

    remainder = mod(n, block_size)
    first_kept = remainder + 1
    nblocks = div(n - remainder, block_size)
    nblocks >= 2 || throw(ArgumentError("block jackknife requires at least two blocks"))

    return first_kept, nblocks
end

"""
    jk_block_index(input_data::AbstractVector, block_size::Integer) -> Vector{Vector{Int}}

Generates block Jackknife index subsets directly from the input data.

If the input length is not divisible by `block_size`, the initial remainder is
dropped before making equal-size blocks.
"""
function jk_block_index(input_data::AbstractVector, block_size::Integer)
    Ndat = length(input_data)
    index = 1:Ndat
    return jk_block_index_set(index, block_size)
end

"""
    jk_block_index_set(index::AbstractVector{<:Integer}, block_size::Integer) -> Vector{Vector{Int}}

Generates leave-one-block-out index subsets for block Jackknife resampling.

If `length(index)` is not divisible by `block_size`, the initial remainder is
dropped before making equal-size blocks.
"""
function jk_block_index_set(index::AbstractVector{<:Integer}, block_size::Integer)
    N = length(index)
    first_kept, nblocks = _block_first_kept(N, block_size)

    subsets = Vector{Vector{Int}}()
    sizehint!(subsets, nblocks)

    for block_start in first_kept:block_size:N
        block_end = block_start + block_size - 1
        subset = Vector{Int}(vcat(index[first_kept:block_start-1], index[block_end+1:N]))
        push!(subsets, subset)
    end

    return subsets
end

function _block_jackknife_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer, statistic::Function)
    N = length(input_data)
    first_kept, nblocks = _block_first_kept(N, block_size)

    JK_values = Float64[]
    sizehint!(JK_values, nblocks)

    for block_start in first_kept:block_size:N
        block_end = block_start + block_size - 1
        sample = input_data[vcat(first_kept:block_start-1, block_end+1:N)]
        push!(JK_values, statistic(sample))
    end

    return statistic(input_data[first_kept:N]), std(JK_values, corrected=false) * sqrt(nblocks - 1)
end

"""
    jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer) -> Tuple{Real, Float64}

Calculates the mean and block Jackknife error for the given data array.
"""
function jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer)
    return _block_jackknife_meanerror(input_data, block_size, mean)
end

"""
    jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer, func::Function) -> Tuple{Real, Float64}

Calculates a custom statistic and its block Jackknife error.
"""
function jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer, func::Function)
    return _block_jackknife_meanerror(input_data, block_size, func)
end

"""
    jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer, KEY::String) -> Tuple{Real, Float64}

Calculates a predefined observable and its block Jackknife error.
"""
function jk_block_meanerror(input_data::AbstractVector{<:Real}, block_size::Integer, KEY::String)
    return _block_jackknife_meanerror(input_data, block_size, _observable_function(KEY))
end

end
