# JkTools

**Author**: A. Tomiya

[Japanese manual](README.ja.md)

JkTools is a small Julia package for Jackknife resampling and statistical error
estimation. It is intended for simple analysis workflows such as Monte Carlo
measurements, where one wants central values and Jackknife errors for primary or
secondary observables.

The package depends only on Julia's standard `Statistics` library.
The package test suite is run by GitHub Actions CI on pushes and pull requests.

## Installation

From the Julia package prompt:

```julia
pkg> add https://github.com/akio-tomiya/JkTools
```

or from Julia code:

```julia
using Pkg
Pkg.add(url="https://github.com/akio-tomiya/JkTools")
```

## Features

- Ordinary leave-one-out Jackknife indices:
  - `jk_index(data)`
  - `jk_index_set(index)`
- Block Jackknife indices:
  - `jk_block_index(data, block_size)`
  - `jk_block_index_set(index, block_size)`
- Jackknife error estimates:
  - `jk_meanerror(data)`
  - `jk_meanerror(data, key)`
  - `jk_meanerror(data, func)`
- Block Jackknife error estimates:
  - `jk_block_meanerror(data, block_size)`
  - `jk_block_meanerror(data, block_size, key)`
  - `jk_block_meanerror(data, block_size, func)`
- Histogram estimates with error bars:
  - `jk_histogram(samples, edges)`
  - `jk_histogram(samples; bins=10)`
  - `jk_hist(samples; bins=10)`
  - `jk_block_histogram(samples, edges, block_size)`
  - `jk_block_histogram(samples, block_size; bins=10)`
  - `jk_block_hist(samples, block_size; bins=10)`
- English help:
  - `?jk_meanerror`
  - `?jk_hist`
  - `jk_help()`

Supported observable keys are:

```julia
"mean", "average"          # mean(x)
"sus", "susceptibility"    # var(x, corrected=false)
"binder", "bin"            # mean(x .^ 4) / mean(x .^ 2)^2
```

Custom functions should map one sample vector to one scalar observable:

```julia
x -> mean(x .^ 2)
x -> var(x, corrected=false)
```

## Help

Use Julia help mode to read English docstrings:

```julia
?jk_meanerror
?jk_block_meanerror
?jk_hist
```

For a short overview:

```julia
jk_help()
```

## Basic Usage

```julia
using JkTools
using Statistics

data = [12.3, 15.6, 14.2, 11.8, 13.7, 16.4, 14.8, 13.1, 12.9, 15.2]

mean_val, mean_err = jk_meanerror(data)
println("mean = $mean_val +/- $mean_err")

sus_val, sus_err = jk_meanerror(data, "sus")
println("susceptibility = $sus_val +/- $sus_err")

square_val, square_err = jk_meanerror(data, x -> mean(x .^ 2))
println("mean(x^2) = $square_val +/- $square_err")
```

For nonlinear observables, the central value is computed from the full sample,
while the error is computed from the leave-one-out Jackknife samples.

## Jackknife Indices

```julia
jk_index_set(1:4)
# [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]

jk_index([10.0, 20.0, 30.0])
# [[2, 3], [1, 3], [1, 2]]
```

## Block Jackknife

Block Jackknife removes one contiguous block at a time. This is useful when the
data have autocorrelation, for example in Markov-chain Monte Carlo measurements.

```julia
block_size = 2

block_mean, block_err = jk_block_meanerror(data, block_size)
println("block mean = $block_mean +/- $block_err")

block_sus, block_sus_err = jk_block_meanerror(data, block_size, "sus")
println("block susceptibility = $block_sus +/- $block_sus_err")
```

If `length(data)` is not divisible by `block_size`, JkTools drops the initial
remainder before making equal-size blocks. This is useful when the earliest data
may be affected by thermalization.

```julia
jk_block_index_set(1:5, 2)
# [[4, 5], [2, 3]]
```

In this example, `1` is dropped, then the kept data are split into `[2, 3]` and
`[4, 5]`.

At least two full blocks are required after the initial remainder is dropped.

## Histograms With Error Bars

`jk_histogram` and `jk_block_histogram` compute one central histogram and one
Jackknife error for each bin. The input is organized by sample. For example, for
Dirac eigenvalues, each element can be the eigenvalues measured on one gauge
configuration.

```julia
eigenvalues_by_config = [
    [0.002, 0.011, 0.018],
    [0.004, 0.010, 0.026],
    [0.001, 0.016, 0.021],
]

hist = jk_hist(eigenvalues_by_config; bins=collect(0.0:0.01:0.03))

hist.centers  # bin centers
hist.values   # bin heights
hist.errors   # Jackknife error for each bin
```

Like Julia plotting histogram functions, `bins` may also be an integer:

```julia
hist = jk_hist(eigenvalues_by_config; bins=20)
```

Bins use `[edge[i], edge[i+1])`, with the final right edge included. Values
outside the edges are ignored.

For density-like plots, use `density=true` to divide by bin width, and `scale`
for any additional normalization factor such as `1 / volume`.

```julia
volume = 32^3 * 8
hist = jk_block_hist(eigenvalues_by_config, 2; bins=20, scale=1 / volume, density=true)
```

JkTools does not depend on a plotting package. With Plots.jl, one possible plot
is:

```julia
using Plots

edges = hist.edges

bar(
    hist.centers,
    hist.values;
    yerror=hist.errors,
    bar_width=edges[2] - edges[1],
    label=false,
    xlabel="lambda",
    ylabel="rho(lambda)",
)
```

See also:

```text
examples/histogram_errorbars.jl
```

That example uses Plots.jl and contains the core pattern:

```julia
hist = jk_block_hist(eigenvalues_by_config, block_size; bins=bins, density=true)
bar(hist.centers, hist.values; yerror=hist.errors)
```

## Input Validation

Ordinary Jackknife requires at least two data points. Block Jackknife requires a
positive `block_size` and at least two full blocks after dropping the initial
remainder. Invalid inputs throw `ArgumentError`.

## License

This package is distributed under the MIT License.
