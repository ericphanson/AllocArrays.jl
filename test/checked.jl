function checked_bumper_run(model, data)
    b = BumperAllocator(2^25) # 32 MiB
    # should be safe here because we don't allocate concurrently
    with_allocator(b) do
        result = sum(model, data)
        reset!(b)
        return result
    end
end

@testset "Basic model" begin
    model = Chain(Dense(1 => 23, tanh), Dense(23 => 1; bias=true), only)

    data = [[x] for x in -2:0.001f0:2]

    checked_alloc_data = CheckedAllocArray.(data)

    @test sum(model, data) ≈ checked_bumper_run(model, checked_alloc_data) atol = 1e-3
    @test sum(model, data) ≈ sum(model, checked_alloc_data) atol = 1e-3

    # # Show some timing info
    @showtime checked_bumper_run(model, checked_alloc_data)
end

@testset "basic escape" begin
    input = CheckedAllocArray([1.0])
    b = BumperAllocator(2^23) # 8 MiB
    y = with_allocator(b) do
        y = similar(input)
        y .= 2
        reset!(b)
        return y
    end
    @test_throws InvalidMemoryException y[1]
end

function bad_function_1(a)
    b = BumperAllocator(2^23) # 8 MiB
    output = []
    with_allocator(b) do
        result = some_allocating_function(a)
        push!(output, result.b) # wrong! `b` is escaping `reset`!
        reset!(b)
    end
    return sum(output...)
end

function bad_function_2(a)
    b = BumperAllocator(2^23) # 8 MiB
    output = Channel(Inf)
    with_allocator(b) do
        @sync for _ = 1:10
            Threads.@spawn begin
                scalar = basic_reduction(a)
                put!(output, scalar)
                reset!(b) # wrong! we cannot reset here as `b` is being used on other tasks
            end
        end
    end
    close(output)
    return sum(collect(output))
end

@testset begin
    @test_throws InvalidMemoryException bad_function_1(CheckedAllocArray([1]))

    # This is not guaranteed to throw:
    # @test_throws InvalidMemoryException bad_function_2(CheckedAllocArray([1]))

end
