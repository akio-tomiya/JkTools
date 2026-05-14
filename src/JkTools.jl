"""
    JkTools

Small tools for ordinary Jackknife, block Jackknife, and histogram error
estimates. Use Julia help mode, for example `?jk_meanerror`, or call
`jk_help()` for a short overview.
"""
module JkTools

using Statistics  # Provides functions for statistical calculations

# Functions to export
export jk_index_set, jk_index, jk_meanerror
export jk_block_index_set, jk_block_index, jk_block_meanerror
export jk_histogram, jk_block_histogram
export jk_hist, jk_block_hist
export jk_help

const _HELP_EN = """
JkTools quick help

Ordinary Jackknife:
  jk_meanerror(data)
  jk_meanerror(data, key)
  jk_meanerror(data, func)

Block Jackknife:
  jk_block_meanerror(data, block_size)
  jk_block_meanerror(data, block_size, key)
  jk_block_meanerror(data, block_size, func)

Histogram with error bars:
  jk_hist(samples; bins=20)
  jk_hist(samples; bins=edges)
  jk_block_hist(samples, block_size; bins=20)

Supported keys:
  "mean", "average", "sus", "susceptibility", "binder", "bin"

Histogram results have:
  hist.edges, hist.centers, hist.values, hist.errors
"""

"""
    jk_help([io::IO=stdout]) -> nothing

Prints a short JkTools help message. `lang` can be `:en`, `:ja`, or `:both`.
"""
function jk_help(io::IO=stdout)
    print(io, _HELP_EN)
    return nothing
end

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

Returns:
`Vector{Vector{Int}}`. Each element is an index set with one data point removed.

Example:
```julia
jk_index([10.0, 20.0, 30.0])
# [[2, 3], [1, 3], [1, 2]]
```
"""
function jk_index(input_data::AbstractVector)
    Ndat = length(input_data)
    index = 1:Ndat
    return jk_index_set(index)
end

"""
    jk_index_set(index::AbstractVector{<:Integer}) -> Vector{Vector{Int}}

Generates subsets of indices for Jackknife resampling from the given array of indices.

Example:
```julia
jk_index_set(1:4)
# [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
```
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

Returns:
`(central_value, error)`.
"""
function jk_meanerror(input_data::AbstractVector{<:Real})
    return _jackknife_meanerror(input_data, mean)
end

"""
    jk_meanerror(input_data::AbstractVector{<:Real}, func::Function) -> Tuple{Real, Float64}

Calculates the mean and Jackknife error for the given data array using a custom function.
`func` must map one sample vector to one scalar observable.

Example:
```julia
jk_meanerror(data, x -> mean(x .^ 2))
jk_meanerror(data, x -> var(x, corrected=false))
```
"""
function jk_meanerror(input_data::AbstractVector{<:Real}, func::Function)
    return _jackknife_meanerror(input_data, func)
end

"""
    jk_meanerror(input_data::AbstractVector{<:Real}, KEY::String) -> Tuple{Real, Float64}

Calculates the mean and Jackknife error for statistical observables based on a specified keyword.

Supported keys:
- `"mean"` or `"average"`: `mean(x)`
- `"sus"` or `"susceptibility"`: `var(x, corrected=false)`
- `"binder"` or `"bin"`: `mean(x .^ 4) / mean(x .^ 2)^2`

Unsupported keys throw an error.
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
`func` must map one sample vector to one scalar observable.
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

function _validate_histogram_edges(edges::AbstractVector{<:Real})
    length(edges) >= 2 || throw(ArgumentError("histogram edges require at least two values"))

    for i in 1:length(edges)-1
        edges[i] < edges[i+1] || throw(ArgumentError("histogram edges must be strictly increasing"))
    end

    return nothing
end

function _bin_centers(edges::AbstractVector{<:Real})
    return [(edges[i] + edges[i+1]) / 2 for i in 1:length(edges)-1]
end

function _histogram_counts(values, edges::AbstractVector{<:Real})
    counts = zeros(Float64, length(edges) - 1)
    data = values isa Real ? (values,) : values

    for x in data
        bin = searchsortedlast(edges, x)
        if bin == length(edges) && x == edges[end]
            bin -= 1
        end

        if 1 <= bin <= length(counts)
            counts[bin] += 1
        end
    end

    return counts
end

function _sample_histogram_counts(samples::AbstractVector, edges::AbstractVector{<:Real})
    length(samples) >= 2 || throw(ArgumentError("histogram jackknife requires at least two samples"))
    _validate_histogram_edges(edges)

    counts = zeros(Float64, length(samples), length(edges) - 1)
    for (i, sample) in enumerate(samples)
        counts[i, :] .= _histogram_counts(sample, edges)
    end

    return counts
end

function _append_histogram_values!(values::Vector{Float64}, sample)
    if sample isa Real
        push!(values, Float64(sample))
    else
        for x in sample
            push!(values, Float64(x))
        end
    end

    return values
end

function _histogram_data_range(samples::AbstractVector)
    values = Float64[]
    for sample in samples
        _append_histogram_values!(values, sample)
    end

    isempty(values) && throw(ArgumentError("histogram bins cannot be inferred from empty samples"))
    return minimum(values), maximum(values)
end

function _histogram_edges_from_bins(samples::AbstractVector, bins::Integer)
    bins >= 1 || throw(ArgumentError("bins must be positive"))

    xmin, xmax = _histogram_data_range(samples)
    if xmin == xmax
        half_width = xmin == 0 ? 0.5 : abs(xmin) / 2
        xmin -= half_width
        xmax += half_width
    end

    return collect(range(xmin, xmax; length=bins + 1))
end

function _histogram_edges_from_bins(samples::AbstractVector, bins::AbstractVector{<:Real})
    _validate_histogram_edges(bins)
    return collect(Float64, bins)
end

function _scale_histogram_values(values::AbstractVector, edges::AbstractVector{<:Real}, scale::Real, density::Bool)
    scaled = Float64.(values) .* Float64(scale)

    if density
        widths = [edges[i+1] - edges[i] for i in 1:length(edges)-1]
        scaled ./= Float64.(widths)
    end

    return scaled
end

function _histogram_result(edges::AbstractVector{<:Real}, values::AbstractVector, errors::AbstractVector)
    return (
        edges=Float64.(edges),
        centers=Float64.(_bin_centers(edges)),
        values=Float64.(values),
        errors=Float64.(errors),
    )
end

function _jackknife_histogram_from_counts(counts::Matrix{Float64}, edges::AbstractVector{<:Real}, scale::Real, density::Bool)
    nsamples, nbins = size(counts)
    nsamples >= 2 || throw(ArgumentError("histogram jackknife requires at least two samples"))

    central = _scale_histogram_values(vec(mean(counts, dims=1)), edges, scale, density)
    JK_values = zeros(Float64, nsamples, nbins)

    for i in 1:nsamples
        rows = vcat(1:i-1, i+1:nsamples)
        JK_values[i, :] .= _scale_histogram_values(vec(mean(counts[rows, :], dims=1)), edges, scale, density)
    end

    errors = [
        std(JK_values[:, bin], corrected=false) * sqrt(nsamples - 1)
        for bin in 1:nbins
    ]

    return _histogram_result(edges, central, errors)
end

function _block_jackknife_histogram_from_counts(counts::Matrix{Float64}, edges::AbstractVector{<:Real}, block_size::Integer, scale::Real, density::Bool)
    nsamples, nbins = size(counts)
    first_kept, nblocks = _block_first_kept(nsamples, block_size)

    central = _scale_histogram_values(vec(mean(counts[first_kept:nsamples, :], dims=1)), edges, scale, density)
    JK_values = zeros(Float64, nblocks, nbins)

    for (block_index, block_start) in enumerate(first_kept:block_size:nsamples)
        block_end = block_start + block_size - 1
        rows = vcat(first_kept:block_start-1, block_end+1:nsamples)
        JK_values[block_index, :] .= _scale_histogram_values(vec(mean(counts[rows, :], dims=1)), edges, scale, density)
    end

    errors = [
        std(JK_values[:, bin], corrected=false) * sqrt(nblocks - 1)
        for bin in 1:nbins
    ]

    return _histogram_result(edges, central, errors)
end

"""
    jk_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}; scale::Real=1.0, density::Bool=false)

Calculates histogram bin values and Jackknife errors.

Each element of `samples` is treated as one Jackknife sample. An element may be a
single real number or a vector of real numbers, such as eigenvalues measured on
one configuration. Bins use `[edge[i], edge[i+1])`, with the final right edge
included. Values outside the edges are ignored.

The result is a named tuple with `edges`, `centers`, `values`, and `errors`.
`scale` multiplies both values and errors. If `density=true`, values and errors
are also divided by the bin widths.
"""
function jk_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}; scale::Real=1.0, density::Bool=false)
    counts = _sample_histogram_counts(samples, edges)
    return _jackknife_histogram_from_counts(counts, edges, scale, density)
end

"""
    jk_histogram(samples::AbstractVector; bins=10, scale::Real=1.0, density::Bool=false)

Histogram-like keyword API for `jk_histogram`. `bins` may be a positive integer
or a vector of bin edges.
"""
function jk_histogram(samples::AbstractVector; bins=10, scale::Real=1.0, density::Bool=false)
    edges = _histogram_edges_from_bins(samples, bins)
    return jk_histogram(samples, edges; scale=scale, density=density)
end

"""
    jk_hist(args...; kwargs...)

Short alias for `jk_histogram`, intended for hist-like usage.
"""
function jk_hist(args...; kwargs...)
    return jk_histogram(args...; kwargs...)
end

"""
    jk_block_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}, block_size::Integer; scale::Real=1.0, density::Bool=false)

Calculates histogram bin values and block Jackknife errors.

If the number of samples is not divisible by `block_size`, the initial remainder
is dropped before making equal-size blocks.
"""
function jk_block_histogram(samples::AbstractVector, edges::AbstractVector{<:Real}, block_size::Integer; scale::Real=1.0, density::Bool=false)
    counts = _sample_histogram_counts(samples, edges)
    return _block_jackknife_histogram_from_counts(counts, edges, block_size, scale, density)
end

"""
    jk_block_histogram(samples::AbstractVector, block_size::Integer; bins=10, scale::Real=1.0, density::Bool=false)

Histogram-like keyword API for block Jackknife histograms. `bins` may be a
positive integer or a vector of bin edges.
"""
function jk_block_histogram(samples::AbstractVector, block_size::Integer; bins=10, scale::Real=1.0, density::Bool=false)
    edges = _histogram_edges_from_bins(samples, bins)
    return jk_block_histogram(samples, edges, block_size; scale=scale, density=density)
end

"""
    jk_block_hist(args...; kwargs...)

Short alias for `jk_block_histogram`, intended for hist-like usage.
"""
function jk_block_hist(args...; kwargs...)
    return jk_block_histogram(args...; kwargs...)
end

end
