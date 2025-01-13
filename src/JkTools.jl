module JkTools

using Statistics  # Provides functions for statistical calculations

# Functions to export
export jk_index_set, jk_index, jk_meanerror

#=
# Example Usage:
data = [12.3, 15.6, 14.2, 11.8, 13.7, 16.4, 14.8, 13.1, 12.9, 15.2]
# (14.0, 0.47469288317114383)  # Expected output for mean and standard error
jk_meanerror(data)
jk_meanerror(data,"mean") 
=#

"""
    jk_index(input_data::Array{Float64, 1}) -> Vector{Vector{Int}}

Generates Jackknife index subsets directly from the input data.

# Arguments
- `input_data::Array{Float64, 1}`: A one-dimensional array of input data.

# Returns
- `Vector{Vector{Int}}`: A list of subsets of indices for Jackknife resampling, each excluding one element at a time.

# Notes
This function combines the length of the input data with `jk_index_set` to generate the subsets.
"""
function jk_index(input_data::Array{Float64, 1})
    Ndat = length(input_data)
    index = 1:Ndat
    return jk_index_set(index)
end

"""
    jk_index_set(index::Vector{Int}) -> Vector{Vector{Int}}

Generates subsets of indices for Jackknife resampling from the given array of indices.

# Arguments
- `index::Vector{Int}`: A vector containing the indices of the original data.

# Returns
- `Vector{Vector{Int}}`: A list of subsets of indices, each excluding one element at a time.
"""
function jk_index_set(index::Vector{Int})
    N = length(index)  # Total number of indices
    subsets = Vector{Vector{Int}}()  # Container for subsets

    # Create subsets by excluding one index at a time
    for i in 1:N
        subset = vcat(index[1:i-1], index[i+1:end])
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

"""
    jk_meanerror(input_data::Array{Float64, 1}) -> Tuple{Float64, Float64}

Calculates the mean and Jackknife error for the given data array.

# Arguments
- `input_data::Array{Float64, 1}`: A one-dimensional array of input data.

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error of the input data.
"""
function jk_meanerror(input_data::Array{Float64, 1})
    # Generate Jackknife index subsets
    subsets = jk_index_set(1:length(input_data))
    NJK = length(subsets)  # Number of subsets
    JK_means = Float64[]   # Store means of subsets

    # Calculate mean for each subset
    for subset in subsets
        sample = input_data[subset]  # Extract subset data
        mean_val = mean(sample)     # Compute mean
        push!(JK_means, mean_val)
    end

    # Calculate and return mean and Jackknife error
    return mean(JK_means), std(JK_means, corrected=false) * sqrt(NJK - 1)
end

"""
    jk_meanerror(input_data::Array{Float64, 1}, func::Function) -> Tuple{Float64, Float64}

Calculates the mean and Jackknife error for the given data array using a custom function.

# Arguments
- `input_data::Array{Float64, 1}`: A one-dimensional array of input data.
- `func::Function`: A function to compute the central value for each subset (e.g. func(x)=x^2).

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error of the computed values.
"""
function jk_meanerror(input_data::Array{Float64, 1}, func::Function)
    # Generate Jackknife index subsets
    subsets = jk_index_set(1:length(input_data))
    NJK = length(subsets)  # Number of subsets
    JK_values = Float64[]  # Store function results for subsets

    # Apply function to each subset and compute mean
    for subset in subsets
        sample = input_data[subset]
        func_val = mean(func.(sample))  # Apply function element-wise and compute mean
        push!(JK_values, func_val)
    end

    # Calculate and return mean and Jackknife error
    return mean(JK_values), std(JK_values, corrected=false) * sqrt(NJK - 1)
end

"""
    jk_meanerror(input_data::Array{Float64, 1}, KEY::String) -> Tuple{Float64, Float64}

Calculates the mean and Jackknife error for statistical observables based on a specified keyword.

# Arguments
- `input_data::Array{Float64, 1}`: A one-dimensional array of input data.
- `KEY::String`: A keyword specifying the observable:
  - `"mean"` or `"average"`: Mean value.
  - `"sus"` or `"susceptibility"`: Variance (susceptibility).
  - `"binder"` or `"bin"`: Binder cumulant.

# Returns
- `Tuple{Float64, Float64}`: A tuple containing the mean and the Jackknife error for the specified observable.

# Notes
An error is raised if an unsupported `KEY` is provided.
"""
function jk_meanerror(input_data::Array{Float64, 1}, KEY::String)
    # Generate Jackknife index subsets
    subsets = jk_index_set(1:length(input_data))
    NJK = length(subsets)  # Number of subsets
    JK_values = Float64[]  # Store computed values for subsets

    # Define the function based on the key
    func = if KEY in ["mean", "average"]
        mean
    elseif KEY in ["sus", "susceptibility"]
        x -> var(x, corrected=false)
    elseif KEY in ["binder", "bin"]
        x -> mean(x .^ 4) / mean(x .^ 2)^2
    else
        error("Unsupported KEY: $KEY")
    end

    # Apply function to each subset
    for subset in subsets
        sample = input_data[subset]
        func_val = func(sample)  # Apply the function
        push!(JK_values, func_val)
    end

    # Calculate and return mean and Jackknife error
    return mean(JK_values), std(JK_values, corrected=false) * sqrt(NJK - 1)
end

end