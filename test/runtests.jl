using AllocArrays
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
    b = BumperAllocator()
    with_allocator(b) do
        ret = basic_reduction(a)
        reset!(b)
        return ret
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
    include("checked.jl")
    include("autoscaling_alloc_buffer.jl")

    # Bug reported here:
    # https://julialang.zulipchat.com/#narrow/stream/137791-general/topic/AllocArrays.2Ejl/near/398698500
    a = AllocArray(1:4)
    @test a[1:2] .+ a[3:4]' isa AllocArray

    c = CheckedAllocArray(1:4)
    @test c[1:2] .+ c[3:4]' isa CheckedAllocArray

    # Constructor does not recurse
    a = AllocArray(1:4)
    @test AllocArray(a).arr === a.arr

    b = BumperAllocator()
    @test sprint(show, b) == "BumperAllocator(AutoscalingAllocBuffer())"
    @test sprint(show, MIME"text/plain"(), b) == "BumperAllocator(AutoscalingAllocBuffer())"
    x = with_allocator(b) do
        similar(c, 100, 100)
    end
    @test sprint(show, b) == "BumperAllocator(AutoscalingAllocBuffer())"
    @test sprint(show, MIME"text/plain"(), b) == "BumperAllocator(AutoscalingAllocBuffer()) (tracking 1 CheckedAllocArray memory location)"

    b = UncheckedBumperAllocator()
    @test sprint(show, b) == "UncheckedBumperAllocator(AutoscalingAllocBuffer())"
    @test sprint(show, MIME"text/plain"(), b) == "UncheckedBumperAllocator(AutoscalingAllocBuffer())"

end
