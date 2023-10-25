# AllocArrays

[![Build Status](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ericphanson/AllocArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ericphanson/AllocArrays.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ericphanson.github.io/AllocArrays.jl/dev/)

Prototype that attempts to allow usage of [Bumper.jl](https://github.com/MasonProtter/Bumper.jl) (or potentially other allocation strategies) through code that doesn't know about Bumper.

This is accomplished by creating a wrapper type `AllocArray` which dispatches `similar` dynamically to an allocator depending on the contextual scope (using [ScopedValues.jl](https://github.com/vchuravy/ScopedValues.jl)).

This package also provides a much safer `CheckedAllocArray` which keeps track of the validity
of each allocated array, to provide an error in case of access to an invalid array. This
can be used to test and develop code, before switching to `AllocArray` in case the overhead
of these checks is prohibitive.

Demo:

```julia
using AllocArrays

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
    b = BumperAllocator(2^24) # 16 MiB
    with_allocator(b) do
        ret = basic_reduction(a)
        reset!(b)
        return ret
    end
end

a = AllocArray(arr);

@time bumper_reduction(a) #  0.010638 seconds (16.28 k allocations: 1.129 MiB, 89.93% compilation time)
@time bumper_reduction(a) #  0.000528 seconds (25 allocations: 800 bytes)
```

We can see we brought allocations down from 2.289 MiB to 800 bytes.

For a less-toy example, in `test/flux.jl` we test inference over a Flux model:

```julia
# Baseline: Array
infer!(predictions, model, data): 1.824578 seconds (59.50 k allocations: 2.841 GiB, 10.10% gc time)
# Baseline: StrideArray
stride_data = StrideArray.(data)
infer!(predictions, model, stride_data): 1.741713 seconds (59.50 k allocations: 2.841 GiB, 11.00% gc time)
# Using AllocArray:
alloc_data = AllocArray.(data)
infer!(predictions, model, alloc_data): 1.773365 seconds (150.89 k allocations: 30.338 MiB, 0.53% gc time)
```

We can see in this example, we got ~100x less allocation, and similar runtime.

## Design notes

The user is responsible for constructing buffers (via `AllocBuffer` or the constructors `BumperAllocator` and `UncheckedBumperAllocator`) and for resetting them (`reset!`).
No implicit buffers are used, and `reset!` is never called in the package. These choices are deliberate: the caller must construct the buffer, pass it to AllocArrays.jl to be used when appropriate, and reset it when they are done.

In particular, the caller must:
- ...not reset a buffer in active use. E.g., do not call `reset!` on a buffer that may be used by another task
- ...not allow memory allocated with a buffer to be live after the underlying buffer has been reset
- ...reset their buffers before it runs out of memory

## Safety

Before using a bump allocator (`BumperAllocator`, or `UncheckedBumperAllocator`) it is recommended the user read the [Bumper.jl README](https://github.com/MasonProtter/Bumper.jl#bumperjl) to understand how it works and what the limitations are.

It is also recommended to start with `CheckedAllocArray` (with `BumperAllocator`)
and move to using `AllocArray` only when necessary on well-tested code.

Note also:

- Just as with all usage of Bumper.jl, the user is responsible for only using `reset!` when newly allocated arrays truly will not escape
  - with AllocArrays this can be slightly more subtle, because within a `with_allocator` with a bump allocator  block, `similar` calls on `AllocArray` or `CheckedAllocArray`s will allocate using the bump allocator. Thus, one must be sure that none of those allocations leak past a `reset!`. The simplest way to do so is to be sure no allocations of any kind escape from a `reset!` block. You can pass in pre-allocated memory and fill that.
- Using an `UncheckedBumpAllocator` is not safe if allocations may occur concurrently. Since you may not know all the `similar` calls present in the code, this is a-priori dangerous to use.
- Using `BumpAllocator` provides a semi-safe alternative simply by using a lock to control access to `buf`. In this way, the single buffer `buf` will be used to allocate for all `similar` calls (even across threads/tasks).
    - However, `reset!` must be called outside of the threaded region, since deallocation in the bump allocator (via `reset!`) on one task will interfere with allocations on others.
