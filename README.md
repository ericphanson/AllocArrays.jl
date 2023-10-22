# AllocArrays

[![Build Status](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ericphanson/AllocArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ericphanson/AllocArrays.jl)

Prototype that attempts to allow usage of [Bumper.jl](https://github.com/MasonProtter/Bumper.jl) (or potentially other allocation strategies) through code that doesn't know about Bumper.

This is accomplished by creating a wrapper type `AllocArray` which dispatches `similar` dynamically to an allocator depending on the contextual scope (using [ScopedValues.jl](https://github.com/vchuravy/ScopedValues.jl)).

Demo:
```julia
using AllocArrays, Bumper

# Some function, possibly in some package,
# that doesn't know about bumper, but happens to only use `similar`
# to allocate memory
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

arr = ones(Float64, 100_000)
@time basic_reduction(a) #  0.129436 seconds (181.10 k allocations: 14.228 MiB, 99.11% compilation time)
@time basic_reduction(a) #  0.000494 seconds (21 allocations: 2.289 MiB)


function bumper_reduction(a)
    with_bumper() do
        @no_escape begin
            basic_reduction(a)
        end
    end
end

a = AllocArray(arr);

@time bumper_reduction(a) #  0.010638 seconds (16.28 k allocations: 1.129 MiB, 89.93% compilation time)
@time bumper_reduction(a) #  0.000528 seconds (25 allocations: 800 bytes)
```
We can see we brought allocations down from 2.289 MiB to 800 bytes.

For a less-toy example, in `test/flux.jl` we test inferencne over a Flux model:
```julia
# Baseline: Array
infer!(predictions, model, data): 2.053936 seconds (59.49 k allocations: 2.841 GiB, 11.71% gc time)
# Baseline: StrideArray
stride_data = StrideArray.(data)
infer!(predictions, model, stride_data): 1.960467 seconds (59.49 k allocations: 2.841 GiB, 11.92% gc time)
# Using AllocArray:
alloc_data = AllocArray.(data)
infer!(predictions, model, alloc_data): 1.630521 seconds (118.34 k allocations: 28.843 MiB)
```
We can see in this example, we got ~100x less allocation, and slight runtime improvement.
