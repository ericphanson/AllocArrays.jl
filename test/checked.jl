
function checked_bumper_run(model, data)
    buf = AllocBuffer(2^25) # 32 MiB
    # should be safe here because we don't allocate concurrently
    with_checked_bumper_no_escape(buf) do
        sum(model, data)
    end
end

@testset "Basic model" begin
    model = Chain(Dense(1 => 23, tanh), Dense(23 => 1; bias=true), only)

    data = [[x] for x in -2:0.001f0:2]

    alloc_data = CheckedAllocArray.(data)

    @test sum(model, data) ≈ checked_bumper_run(model, alloc_data) atol = 1e-3
    # @test sum(model, data) ≈ sum(model, alloc_data) atol = 1e-3

    # # Show some timing info
    # @showtime sum(model, data)
    # @showtime sum(model, alloc_data)
    # @showtime checked_bumper_run(model, alloc_data)
end

@testset "escape" begin
    input = CheckedAllocArray([1.0])
    buf = AllocBuffer(2^25) # 32 MiB
    y = with_checked_bumper_no_escape(buf) do
        y = similar(input)
        y .= 2
    end

    # TODO- exception type
    @test_throws Any y[1]

end
