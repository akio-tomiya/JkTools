# JkTools

[![Build Status](https://github.com/akio/JkTools.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/akio/JkTools.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/akio/JkTools.jl.svg?branch=main)](https://app.travis-ci.com/akio/JkTools.jl)
[![Coverage](https://codecov.io/gh/akio/JkTools.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/akio/JkTools.jl)
[![Coverage](https://coveralls.io/repos/github/akio/JkTools.jl/badge.svg?branch=main)](https://coveralls.io/github/akio/JkTools.jl?branch=main)

# JkTools

**Author**: A. Tomiya

JkTools is a simple Julia package for performing Jackknife resampling and estimating statistical errors. 

---

## Installation

To use this package, copy the `JkTools` module into your project. It does not require additional dependencies other than the Julia `Statistics` standard library.

---

## Features

- **Jackknife index generation**
  - `jk_index`: Generate Jackknife index subsets from input data.
  - `jk_index_set`: Generate Jackknife index subsets from indices or ranges.
- **Mean and error estimation**
  - `jk_meanerror`: Compute the mean and Jackknife error for input data.
  - Supports custom functions and predefined statistical observables (e.g., mean, susceptibility, Binder cumulant).

---

## Usage

### Example

```julia
using JkTools

# Example data
data = [12.3, 15.6, 14.2, 11.8, 13.7, 16.4, 14.8, 13.1, 12.9, 15.2]

# Compute mean and Jackknife error
mean_val, error_val = jk_meanerror(data)
println("Mean: $mean_val, Error: $error_val")

# Compute susceptibility using predefined keys
susceptibility, error_sus = jk_meanerror(data, "sus")
println("Susceptibility: $susceptibility, Error: $error_sus")

# Custom function
# custom_func = x -> x^2 # this definition is also acceptable.
custom_func(x) = x^2
custom_mean, custom_error = jk_meanerror(data, custom_func)
println("Custom Mean: $custom_mean, Custom Error: $custom_error")
```

---

## License
This package is distributed under the MIT License.

