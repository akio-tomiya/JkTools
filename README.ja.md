# JkTools

**Author**: A. Tomiya

[English manual](README.md)

JkTools は、Jackknife resampling と統計誤差評価のための小さな Julia
パッケージです。Monte Carlo 測定などで、プライマリな物理量や
セカンダリな物理量の中心値と Jackknife 誤差を求める用途を想定しています。

依存は Julia 標準ライブラリの `Statistics` だけです。
package のテストは、push と pull request 時に GitHub Actions CI で実行します。

## インストール

Julia の package prompt から:

```julia
pkg> add https://github.com/akio-tomiya/JkTools
```

Julia code から:

```julia
using Pkg
Pkg.add(url="https://github.com/akio-tomiya/JkTools")
```

## 機能

- 通常の 1 点抜き Jackknife index:
  - `jk_index(data)`
  - `jk_index_set(index)`
- ブロック Jackknife index:
  - `jk_block_index(data, block_size)`
  - `jk_block_index_set(index, block_size)`
- Jackknife 誤差評価:
  - `jk_meanerror(data)`
  - `jk_meanerror(data, key)`
  - `jk_meanerror(data, func)`
- ブロック Jackknife 誤差評価:
  - `jk_block_meanerror(data, block_size)`
  - `jk_block_meanerror(data, block_size, key)`
  - `jk_block_meanerror(data, block_size, func)`
- 誤差棒つきヒストグラム用の値:
  - `jk_histogram(samples, edges)`
  - `jk_histogram(samples; bins=10)`
  - `jk_hist(samples; bins=10)`
  - `jk_block_histogram(samples, edges, block_size)`
  - `jk_block_histogram(samples, block_size; bins=10)`
  - `jk_block_hist(samples, block_size; bins=10)`
- 英語ヘルプ:
  - `?jk_meanerror`
  - `?jk_hist`
  - `jk_help()`

通常 API は、`block` keyword を指定するとブロック Jackknife に切り替わります。

```julia
jk_meanerror(data; block=2)
jk_meanerror(data, "sus"; block=2)
jk_hist(eigenvalues_by_config; bins=20, block=2)
```

使える observable key:

```julia
"mean", "average"          # mean(x)
"sus", "susceptibility"    # var(x, corrected=false)
"binder", "bin"            # Binder ratio: mean(x .^ 4) / mean(x .^ 2)^2
```

JkTools では `binder` / `bin` は上の Binder ratio を意味します。

任意関数を渡す場合、その関数は 1 つの sample vector から 1 つの
スカラー observable を返す必要があります。

```julia
x -> mean(x .^ 2)
x -> var(x, corrected=false)
```

## ヘルプ

Julia の help mode で英語 docstring を確認できます。

```julia
?jk_meanerror
?jk_block_meanerror
?jk_hist
```

短い概要を表示するには:

```julia
jk_help()
```

## 基本的な使い方

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

非線形 observable の場合、中心値は全 sample から計算し、誤差は
leave-one-out の Jackknife sample から計算します。

## Jackknife Index

```julia
jk_index_set(1:4)
# [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]

jk_index([10.0, 20.0, 30.0])
# [[2, 3], [1, 3], [1, 2]]
```

## ブロック Jackknife

ブロック Jackknife は、1 点ずつではなく連続した block を 1 つずつ抜きます。
Markov-chain Monte Carlo のようにデータに自己相関がある場合に有用です。

```julia
block_size = 2

block_mean, block_err = jk_meanerror(data; block=block_size)
println("block mean = $block_mean +/- $block_err")

block_sus, block_sus_err = jk_meanerror(data, "sus"; block=block_size)
println("block susceptibility = $block_sus +/- $block_sus_err")
```

これまでの明示的な書き方もそのまま使えます。

```julia
jk_block_meanerror(data, block_size)
jk_block_meanerror(data, block_size, "sus")
```

`length(data)` が `block_size` で割り切れない場合、JkTools は先頭側の
余りを drop してから同じ大きさの block を作ります。これは、初期の
データが熱化の影響を受けている可能性がある場合に便利です。

```julia
jk_block_index_set(1:5, 2)
# [[4, 5], [2, 3]]
```

この例では `1` を drop し、残ったデータを `[2, 3]` と `[4, 5]` に
分けています。

先頭の余りを drop したあと、少なくとも 2 つの full block が必要です。

## 誤差棒つきヒストグラム

`jk_histogram` と `jk_block_histogram` は、中心となるヒストグラムと
各 bin の Jackknife 誤差を計算します。入力は sample ごとに整理します。
例えば Dirac eigenvalue の場合、各要素を 1 つの gauge configuration で
測った eigenvalue の列にできます。

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

Julia の plotting histogram に近く、`bins` には整数も指定できます。

```julia
hist = jk_hist(eigenvalues_by_config; bins=20)
```

bin は `[edge[i], edge[i+1])` として扱い、最後の右端だけ含めます。
範囲外の値は無視します。

密度のように表示したい場合は、`density=true` で bin 幅で割り、
`scale` で `1 / volume` などの規格化因子を掛けます。

```julia
volume = 32^3 * 8
hist = jk_hist(eigenvalues_by_config; bins=20, block=2, scale=1 / volume, density=true)
```

これまでの明示的な `jk_block_hist(eigenvalues_by_config, 2; bins=20)` も
そのまま使えます。

JkTools は plotting package に依存しません。Plots.jl を使う場合は、
例えば次のように描画できます。

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

例は次にも置いてあります。

```text
examples/histogram_errorbars.jl
```

この例では Plots.jl を使い、中心となる書き方は次です。

```julia
hist = jk_hist(eigenvalues_by_config; bins=bins, block=block_size, density=true)
bar(hist.centers, hist.values; yerror=hist.errors)
```

## 入力の検証

通常の Jackknife には少なくとも 2 点のデータが必要です。ブロック
Jackknife では、正の `block_size` と、先頭の余りを drop したあとの
少なくとも 2 つの full block が必要です。不正な入力では
`ArgumentError` を投げます。

## ライセンス

このパッケージは MIT License で配布されています。
