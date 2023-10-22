# AllocArrays

[![Build Status](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ericphanson/AllocArrays.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ericphanson/AllocArrays.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ericphanson/AllocArrays.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://ericphanson.github.io/AllocArrays.jl/dev/)

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
    buf = AllocBuffer()
    with_locked_bumper(buf) do
        @no_escape buf begin
            basic_reduction(a)
        end
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

This package does not create any Bumper.jl buffers, does not use any implicit ones, and does not reset any buffer that is handed to it. These choices are deliberate: the caller must create the buffer, pass it to AllocArrays.jl as desired, and reset it when they are done.

In particular, the caller must:
- ...not reset a buffer in active use. E.g., do not call `@no_escape` on a buffer that may be used by another task
- ...not allow memory allocated with a buffer to be live after the underlying buffer has been reset
- ...reset their buffers before it runs out of memory

## Safety

Before using a bump allocator (`unsafe_with_bumper`, or `with_locked_bumper`) it is recommended the user read the [Bumper.jl README](https://github.com/MasonProtter/Bumper.jl#bumperjl) to understand how it works and what the limitations are.

Note also:

- Just as with all usage of Bumper.jl, the user is responsible for only using Bumper's `@no_escape` when newly allocated arrays truly will not escape
  - with AllocArrays this can be slightly more subtle, because within `unsafe_with_bumper` or `with_locked_bumper` block, `similar` calls on `AllocArray`s will allocate using the bump allocator. Thus, one must be sure that none of those allocations leak past a `@no_escape` block. The simplest way to do so is to be sure no allocations of any kind escape from a `@no_escape` block. You can pass in pre-allocated memory and fill that.
- Calling `unsafe_with_bumper(f, buf)` is not safe if allocations may occur concurrently. Since you may not know all the `similar` calls present in the code, this is a-priori dangerous to use.
- Calling `with_locked_bumper(f, buf)` provides a safe alternative simply by using a lock to control access to `buf`. In this way, the single buffer `buf` will be used to allocate for all `similar` calls (even across threads/tasks).
    - However, `@no_escape` must be called outside of the threaded region, since deallocation in the bump allocator (via `@no_escape`) on one task will interfere with allocations on others.

## Examples of improper usage

In the following, we will use our functions:

```julia
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
```
as above.

### Wrong: allowing escapes

Allocations inside `@no_escape` must not escape!

```julia
function bad_function_1(a)
    buf = AllocBuffer()
    output = []
    with_locked_bumper(buf) do
        @no_escape buf begin
            result = some_allocating_function(a)
            push!(output, result.b) # wrong! `b` is escaping `@no_escape`!
        end
    end
    return sum(output...)
end
```

Here is a corrected version:

```julia
function good_function_1(a)
    buf = AllocBuffer()

    # note, we are not inside `with_locked_bumper`, so we are not making buffer-backed memory
    output = similar(a)

    with_locked_bumper(buf) do
        @no_escape buf begin
            result = some_allocating_function(a)
            output .= result.b # OK! we are copying buffer-backed memory into our heap-allocated memory
        end
    end
    return sum(result)
end
```

### Wrong: resetting a buffer in active use

```julia
function bad_function_2(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_locked_bumper(buf) do
        @sync for _ = 1:10
            Threads.@spawn begin
                @no_escape buf begin
                    scalar = basic_reduction(a)
                    put!(output, scalar)
                end # wrong! we cannot reset here as `buf` is being used on other tasks
            end
        end
    end
    close(output)
    return sum(collect(output))
end
```

Here is a corrected version:

```julia
function good_function_2(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_locked_bumper(buf) do
        @no_escape buf begin
            @sync for _ = 1:10
                Threads.@spawn begin
                    scalar = basic_reduction(a)
                    put!(output, scalar)
                end
            end
        end # OK! resetting once we no longer need the allocations
    end
    close(output)
    return sum(collect(output))
end
```

Or, if we need to reset multiple times as we process the data, we could do a serial version:

```julia
function good_function_2b(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_locked_bumper(buf) do
        for _ = 1:10
            @no_escape buf begin
                scalar = basic_reduction(a)
                put!(output, scalar)
            end # OK to reset here! buffer-backed memory is not being used
        end
    end
    close(output)
    return sum(collect(output))
end
```

We could also do something in-between, by launching tasks in batches of `n`, then resetting the buffer between them. Or we could use multiple buffers.

### Wrong: neglecting to reset the buffer

As shown above, we must be careful about when we reset the buffer. However, if we never reset it (analogous to never garbage collecting), we run into another problem which is we will run out memory!

```julia
function bad_function_3(a, N)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_locked_bumper(buf) do
        for _ = 1:N # bad! we are going to allocate `N` times without resetting!
            # if `N` is very large, we will run out of memory.
            scalar = basic_reduction(a)
            put!(output, scalar)
        end
    end
    close(output)
    return sum(collect(output))
end
```

This can be fixed by resetting appropriately, as in e.g. `good_function_2b` above.
