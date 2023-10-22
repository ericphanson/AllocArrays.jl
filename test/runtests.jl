using AllocArrays, Bumper
using Test, Aqua

function some_allocating_function(a)
    b = similar(a)
    b .= a
    c = similar(a)
    c .= a
    return (; b, c)
end

function basic_reduction(a)
    (; b, c) = some_allocating_function(a)
    return sum(b .+ c)
end

function bumper_reduction(a)
    buf = AllocBuffer(2^24) # 16 MiB
    with_locked_bumper(buf) do
        @no_escape buf begin
            basic_reduction(a)
        end
    end
end

@testset "AllocArrays.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(AllocArrays)
    end

    a = AllocArray(ones(Float64, 1000))

    (; b, c) = some_allocating_function(a)
    @test b isa AllocArray
    @test c isa AllocArray

    @test bumper_reduction(a) == 2000.0
    @test basic_reduction(a) == 2000.0

    include("flux.jl")
end
