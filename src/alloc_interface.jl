# Allocation interface

"""
    abstract type Allocator end

Alloactors need to subtype `Alloactor` and implement two methods of `alloc_similar`:

- `AllocArrays.alloc_similar(::Allocator, arr, ::Type{T}, dims::Dims)`
- `AllocArrays.alloc_similar(::Allocator, ::Type{Arr}, dims::Dims) where {Arr<:AbstractArray}`

where the latter is used by broadcasting.
"""
abstract type Allocator end

#####
##### Default allocator
#####

# Just dispatches to `similar`

struct DefaultAllocator <: Allocator end

const DEFAULT_ALLOCATOR = DefaultAllocator()

function alloc_similar(::DefaultAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    return similar(Array{T}, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return similar(Arr, dims)
end

function alloc_similar(::DefaultAllocator, ::CheckedAllocArray, ::Type{T}, dims::Dims) where {T}
    return similar(Array{T}, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return similar(Arr, dims)
end

#####
##### Bumper.jl
#####

# Could be moved to a package extension?


"""
    TODO-fixme
    unsafe_with_bumper(f, buf::AllocBuffer)

Runs `f()` in the context of using a `UncheckedBumperAllocator{typeof(buf)}` to
allocate memory to `similar` calls on [`AllocArray`](@ref)s.

All such allocations should occur within an `@no_escape` block,
and of course, no such allocations should escape that block.

!!! warning
    Not thread-safe. `f` must not allocate memory using `similar` calls on `AllocArray`'s
    across multiple threads or tasks.

Remember: if `f` calls into another package, you might not know if they use concurrency
or not! It is safer to use [`with_locked_bumper`](@ref) for this reason.

## Example

```
using AllocArrays, Bumper
using AllocArrays: unsafe_with_bumper

input = AllocArray([1,2,3])
buf = AllocBuffer()
unsafe_with_bumper(buf) do
     @no_escape buf begin
        # ...code with must not allocate AllocArrays on multiple tasks via `similar` nor escape or return newly-allocated AllocArrays...
        sum(input .* 2)
     end
end

# output
12
```
"""
struct UncheckedBumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
end

function reset!(B::UncheckedBumperAllocator)
    Bumper.reset_buffer!(B.buf)
    return nothing
end

function alloc_similar(B::UncheckedBumperAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end

function alloc_similar(B::UncheckedBumperAllocator, ::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T, N, Arr}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end

# TODO-
# This should wrap a `UncheckedBumperAllocator`
# Then if get an `AllocArray`, we just forward to that and don't do any checking
# If we get a `CheckedAllocArray`, we go through the checking route
# Then there are 2 options:
# - UncheckedBumperAllocator: no protection
# - BumperAllocator: wraps UncheckedBumperAllocator, adds both concurrency and memory safety
# ...
# this interface means the central object the user creates is these allocator types
# then they call our `reset!`. Not `@no_escape`.
# Should we have a `try/finally` interface?
struct BumperAllocator{B} <: Allocator
    bumper::B
    mems::Vector{MemValid}
    # this lock protects access to `mems` and `buf`, and ensures
    # we can make atomic "transactions" to add new memory or invalidate all memory
    lock::ReentrantLock
end

function BumperAllocator(B::AllocBuffer)
    return BumperAllocator(UncheckedBumperAllocator(B), MemValid[], ReentrantLock())
end

BumperAllocator(n::Int) = BumperAllocator(AllocBuffer(n))

Base.lock(B::BumperAllocator) = lock(B.lock)
Base.unlock(B::BumperAllocator) = unlock(B.lock)

function alloc_similar(B::BumperAllocator, c::CheckedAllocArray, ::Type{T},
                       dims::Dims) where {T}
    @lock B begin
        # I think we do still need C's read lock, bc it could have been
        # allocated by a DIFFERENT bumper which could be deallocating as we speak
        # so to safely access `c.alloc_array` we should still use the read lock
        inner = @lock(c, alloc_similar(B.bumper, _get_inner(c), T, dims))
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end

function alloc_similar(B::BumperAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    @lock B begin
        inner = alloc_similar(B.bumper, Arr, dims)
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end

# If we have a `BumperAllocator` and are asked to allocate an unchecked array
# then we can do that by dispatching to the inner bumper. We will still
# get the lock for concurrency-safety.
function alloc_similar(B::BumperAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    return @lock(B, alloc_similar(B.bumper, AllocArray{T,N,Arr}, dims))
end

function alloc_similar(B::BumperAllocator, a::AllocArray, ::Type{T},
                       dims::Dims) where {T}
    return @lock(B, alloc_similar(B.bumper, a, T, dims))
end

function reset!(B::BumperAllocator)
    @lock B begin
        # Invalidate all memory first
        for mem in B.mems
            invalidate!(mem)
        end
        # Then empty the tracked memory
        empty!(B.mems)

        # Then reset the inner bumper
        reset!(B.bumper)
    end
    return nothing
end

function with_allocator(f, b)
    return with(f, CURRENT_ALLOCATOR => b)
end
