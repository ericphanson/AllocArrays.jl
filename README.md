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
@time basic_reduction(arr) # 0.068864 seconds (377.41 k allocations: 21.811 MiB, 99.47% compilation time)
@time basic_reduction(arr) # 0.007774 seconds (10 allocations: 2.289 MiB, 94.21% gc time)


function bumper_reduction!(b, a)
    with_allocator(b) do
        ret = basic_reduction(a)
        reset!(b)
        return ret
    end
end

b = BumperAllocator(2^24) # 16 MiB
a = AllocArray(arr);

@time bumper_reduction!(b, a) #  0.205106 seconds (893.40 k allocations: 44.941 MiB, 2.62% gc time, 99.67% compilation time)
@time bumper_reduction!(b, a) #  0.000217 seconds (27 allocations: 800 bytes)
```

We can see we brought allocations down from 2.289 MiB to ~1 KiB.

For a less-toy example, in `test/flux.jl` we test inference over a Flux model:

```julia
# Baseline: Array
infer!(b, predictions, model, data): 0.492020 seconds (6.77 k allocations: 221.508 MiB, 4.57% gc time)
alloc_data = AllocArray.(data)
infer!(b, predictions, model, alloc_data): 0.326735 seconds (10.09 k allocations: 843.047 KiB)
# see "Usage with Flux" below for `recursive_alloc_arrays` and `infer!`
aa_model = recursive_alloc_arrays(model)
infer!(b, predictions, aa_model, alloc_data): 0.329056 seconds (10.54 k allocations: 855.547 KiB)
# checked example (use for testing)
checked_alloc_data = CheckedAllocArray.(data)
infer!(b, predictions, model, checked_alloc_data): 15.190550 seconds (22.61 k allocations: 1.363 MiB)
```

We can see in this example, we got 200x less allocation (and no GC time), and similar runtime, for `AllocArray`s. We also can reduce allocations more with `aa_model` than `model` in some cases; not here though. We see `CheckedAllocArrays` are far slower.

## Usage with Flux

For reducing allocations as much as possible with Flux models, we can use `recursive_alloc_arrays` below to convert a model to use `AllocArray`s. This will convert all arrays in the model to `AllocArray`s, and will also convert any arrays in the model's parameters to `AllocArray`s. This way, any layers during the forward pass of the model which use `similar` calls based on the layer's parameters will use the bump allocator, when the forward pass is invoked within `with_allocator`.

```julia
using Functors

function recursive_alloc_arrays(obj)
    return fmap(x -> begin
        x isa AbstractArray || return x
        isbitstype(eltype(x)) || return x
        return AllocArray(x)
    end, obj; exclude=x -> x isa AbstractArray{<:Number} || x isa Function)
end

function infer!(b::BumpAllocator, predictions, model, data)
    # Here we use a locked bumper for thread-safety, since NNlib multithreads
    # some of it's functions. However we are sure to only deallocate outside of the threaded region. (All concurrency occurs within the `model` call itself).
    with_allocator(b) do
        for (idx, x) in enumerate(data)
            predictions[idx] .= model(x)
            reset!(b) # reset `b` after each batch
        end
    end
    # don't escape bump-allocated memory!
    return predictions
end
```

The specific function to use here may need to be adjusted depending on the details of the model and Functors.jl. This one is tested in `test/flux.jl` on a simple model.

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
