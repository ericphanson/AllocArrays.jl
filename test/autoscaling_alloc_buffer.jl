using AllocArrays: DEFAULT_INITIAL_BUFFER_SIZE, AutoscalingAllocBuffer, amount_used,
                   total_capacity
using Bumper: alloc!, reset_buffer!

@testset "AutoscalingAllocBuffer" begin
    # test cases:
    # - many small allocations with too-small initial buffer should result in sublinear additional buffers
    # - downscale after workload drops for awhile

    b = AutoscalingAllocBuffer(10^3)
    @test total_capacity(b) == 10^3
    @test amount_used(b) == 0

    m = alloc!(b, Float32, 100, 100)
    @test eltype(m) == Float32
    @test size(m) == (100, 100)

    n = 100 * 100 * sizeof(Float32)
    @test amount_used(b) == n
    @test total_capacity(b) > 2n

    # allocate 80x more than initial capacity
    for i in 1:200
        m = alloc!(b, Float32, 100, 100)
        @test eltype(m) == Float32
        @test size(m) == (100, 100)
    end
    @test total_capacity(b) > 80n
    @test amount_used(b) > 80n
    # still not too many additional buffers
    @test length(b.additional_buffers) < 10

    # now check reset: we should have more capacity than used last time,
    # while having no additional buffers
    used = amount_used(b)
    reset_buffer!(b)
    @test total_capacity(b) > used
    @test total_capacity(b) > 80n
    @test amount_used(b) == 0
    @test isempty(b.additional_buffers)

    # now test downscaling; only allocating `n` for awhile means we end up
    # with total capacity below 2n
    for i in 1:6
        alloc!(b, Float32, 100, 100)
        reset_buffer!(b)
    end
    @test total_capacity(b) < 2n
end
