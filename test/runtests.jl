using JkTools
using Statistics
using Test

function jackknife_expected(data::AbstractVector, statistic::Function)
    n = length(data)
    n >= 2 || throw(ArgumentError("jackknife error requires at least two data points"))

    jk_values = [
        statistic(data[vcat(1:i-1, i+1:n)])
        for i in 1:n
    ]

    return statistic(data), std(jk_values, corrected=false) * sqrt(n - 1)
end

function block_jackknife_expected(data::AbstractVector, block_size::Integer, statistic::Function)
    n = length(data)
    block_size >= 1 || throw(ArgumentError("block_size must be positive"))

    first_kept = mod(n, block_size) + 1
    block_ranges = [
        start:start + block_size - 1
        for start in first_kept:block_size:n
    ]
    length(block_ranges) >= 2 || throw(ArgumentError("block jackknife requires at least two blocks"))

    jk_values = [
        statistic(data[vcat(first_kept:first(block)-1, last(block)+1:n)])
        for block in block_ranges
    ]

    return statistic(data[first_kept:n]), std(jk_values, corrected=false) * sqrt(length(block_ranges) - 1)
end

function test_result(got, expected)
    @test got[1] ≈ expected[1]
    @test got[2] ≈ expected[2]
end

@testset "JkTools.jl" begin
    @testset "jackknife indices" begin
        @test jk_index_set([1, 2, 3, 4]) == [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
        @test jk_index_set(1:4) == [[2, 3, 4], [1, 3, 4], [1, 2, 4], [1, 2, 3]]
        @test jk_index_set([2, 4, 6]) == [[4, 6], [2, 6], [2, 4]]

        @test jk_index([10.0, 20.0, 30.0]) == [[2, 3], [1, 3], [1, 2]]
        @test jk_index([10, 20, 30]) == [[2, 3], [1, 3], [1, 2]]
    end

    @testset "mean jackknife error" begin
        data = [1.0, 2.0, 3.0, 4.0]
        expected = jackknife_expected(data, mean)

        test_result(jk_meanerror(data), expected)
        test_result(jk_meanerror(data, "mean"), expected)
        test_result(jk_meanerror(data, "average"), expected)
    end

    @testset "observable keys" begin
        data = [1.0, 2.0, 3.0, 4.0]

        susceptibility = x -> var(x, corrected=false)
        susceptibility_expected = jackknife_expected(data, susceptibility)
        test_result(jk_meanerror(data, "sus"), susceptibility_expected)
        test_result(jk_meanerror(data, "susceptibility"), susceptibility_expected)

        binder_ratio = x -> mean(x .^ 4) / mean(x .^ 2)^2
        binder_expected = jackknife_expected(data, binder_ratio)
        test_result(jk_meanerror(data, "binder"), binder_expected)
        test_result(jk_meanerror(data, "bin"), binder_expected)

        @test_throws ErrorException jk_meanerror(data, "unknown")
    end

    @testset "custom statistic" begin
        data = [1.0, 2.0, 3.0, 4.0]

        square_mean = x -> mean(x .^ 2)
        test_result(jk_meanerror(data, square_mean), jackknife_expected(data, square_mean))

        susceptibility = x -> var(x, corrected=false)
        test_result(jk_meanerror(data, susceptibility), jackknife_expected(data, susceptibility))
    end

    @testset "real-valued inputs" begin
        data = [1, 2, 3, 4]
        test_result(jk_meanerror(data), jackknife_expected(data, mean))
        test_result(jk_meanerror(data, "sus"), jackknife_expected(data, x -> var(x, corrected=false)))
    end

    @testset "input validation" begin
        @test_throws ArgumentError jk_meanerror(Float64[])
        @test_throws ArgumentError jk_meanerror([1.0])
        @test_throws ArgumentError jk_meanerror(Float64[], mean)
        @test_throws ArgumentError jk_meanerror([1.0], mean)
        @test_throws ArgumentError jk_meanerror(Float64[], "mean")
        @test_throws ArgumentError jk_meanerror([1.0], "mean")
    end

    @testset "block jackknife indices" begin
        @test jk_block_index_set(1:6, 2) == [[3, 4, 5, 6], [1, 2, 5, 6], [1, 2, 3, 4]]
        @test jk_block_index_set(1:5, 2) == [[4, 5], [2, 3]]
        @test jk_block_index_set([2, 4, 6, 8, 10], 2) == [[8, 10], [4, 6]]

        @test jk_block_index([10.0, 20.0, 30.0, 40.0, 50.0], 2) == [[4, 5], [2, 3]]
        @test jk_block_index([10, 20, 30, 40, 50], 2) == [[4, 5], [2, 3]]

        @test_throws ArgumentError jk_block_index_set(1:4, 0)
        @test_throws ArgumentError jk_block_index_set(1:4, -1)
        @test_throws ArgumentError jk_block_index_set(1:4, 4)
        @test_throws ArgumentError jk_block_index_set(1:3, 5)
    end

    @testset "block mean jackknife error" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        expected = block_jackknife_expected(data, 2, mean)

        test_result(jk_block_meanerror(data, 2), expected)
        test_result(jk_block_meanerror(data, 2, "mean"), expected)
        test_result(jk_block_meanerror(data, 2, "average"), expected)

        uneven_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        test_result(jk_block_meanerror(uneven_data, 2), block_jackknife_expected(uneven_data, 2, mean))
    end

    @testset "block observable keys" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        susceptibility = x -> var(x, corrected=false)
        susceptibility_expected = block_jackknife_expected(data, 2, susceptibility)
        test_result(jk_block_meanerror(data, 2, "sus"), susceptibility_expected)
        test_result(jk_block_meanerror(data, 2, "susceptibility"), susceptibility_expected)

        binder_ratio = x -> mean(x .^ 4) / mean(x .^ 2)^2
        binder_expected = block_jackknife_expected(data, 2, binder_ratio)
        test_result(jk_block_meanerror(data, 2, "binder"), binder_expected)
        test_result(jk_block_meanerror(data, 2, "bin"), binder_expected)

        @test_throws ErrorException jk_block_meanerror(data, 2, "unknown")
    end

    @testset "block custom statistic" begin
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

        square_mean = x -> mean(x .^ 2)
        test_result(jk_block_meanerror(data, 2, square_mean), block_jackknife_expected(data, 2, square_mean))

        susceptibility = x -> var(x, corrected=false)
        test_result(jk_block_meanerror(data, 2, susceptibility), block_jackknife_expected(data, 2, susceptibility))
    end

    @testset "block real-valued inputs and validation" begin
        data = [1, 2, 3, 4, 5, 6]
        test_result(jk_block_meanerror(data, 2), block_jackknife_expected(data, 2, mean))
        test_result(jk_block_meanerror(data, 2, "sus"), block_jackknife_expected(data, 2, x -> var(x, corrected=false)))

        @test_throws ArgumentError jk_block_meanerror(Float64[], 1)
        @test_throws ArgumentError jk_block_meanerror([1.0], 1)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 0)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], -1)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2)
        @test_throws ArgumentError jk_block_meanerror(Float64[], 1, mean)
        @test_throws ArgumentError jk_block_meanerror([1.0], 1, mean)
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2, mean)
        @test_throws ArgumentError jk_block_meanerror(Float64[], 1, "mean")
        @test_throws ArgumentError jk_block_meanerror([1.0], 1, "mean")
        @test_throws ArgumentError jk_block_meanerror([1.0, 2.0], 2, "mean")
    end
end
