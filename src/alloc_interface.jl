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

struct BumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
end

function alloc_similar(B::BumperAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end

function alloc_similar(B::BumperAllocator, ::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T, N, Arr}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end

"""
    unsafe_with_bumper(f, buf::AllocBuffer)

Runs `f()` in the context of using a `BumperAllocator{typeof(buf)}` to
allocate memory to `similar` calls on [`AllocArray`](@ref)s.

All such allocations should occur within an `@no_escape` block,
and of course, no such allocations should escape that block.

!!! warning
    Not thread-safe. `f` must not allocate memory using `similar` calls on `AllocArray`'s
    across multiple threads or tasks.

Remember: if `f` calls into another package, you might not know if they use concurrency
or not! It is safer to use [`with_locked_bumper`](@ref) for this reason.

## Example

```jldoctest
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
function unsafe_with_bumper(f, buf::AllocBuffer)
    return with(f, CURRENT_ALLOCATOR => BumperAllocator(buf))
end

#####
##### LockedBumperAllocator
#####

# An alternative route to thread safety: just lock the allocator before using it.
# This helps with short-lived tasks (which shouldn't each get their own buffer)

struct LockedBumperAllocator{A} <: Allocator
    bumper::A
    lock::ReentrantLock
end

function LockedBumperAllocator(buf::AllocBuffer)
    return LockedBumperAllocator(BumperAllocator(buf), ReentrantLock())
end

function LockedBumperAllocator(b)
    return LockedBumperAllocator(b, ReentrantLock())
end
Base.lock(f::Function, B::LockedBumperAllocator) = lock(f, B.lock)
Base.lock(B::LockedBumperAllocator) = lock(B.lock)
Base.unlock(B::LockedBumperAllocator) = unlock(B.lock)

function alloc_similar(B::LockedBumperAllocator, args...)
    return @lock(B, alloc_similar(B.bumper, args...))
end

"""
    with_locked_bumper(f, buf::AllocBuffer)

Runs `f()` in the context of using a `LockedBumperAllocator` to
allocate memory to `similar` calls on [`AllocArray`](@ref)s.

All such allocations should occur within an `@no_escape` block,
and of course, no such allocations should escape that block.

Thread-safe: `f` may spawn multiple tasks or threads, which may each allocate memory using `similar` calls on `AllocArray`'s. However:

!!! warning
    `f` must call `@no_escape` only outside of the threaded region, since deallocation in the bump allocator (via `@no_escape`) on one task will interfere with allocations on others.

## Example

```
using AllocArrays, Bumper

buf = AllocBuffer()
input = AllocArray([1,2,3])
c = Channel(Inf)
with_locked_bumper(buf) do
    # ...code with may be multithreaded but which must not escape or return newly-allocated AllocArrays...
    @no_escape buf begin # called outside of threaded region
        @sync for i = 1:10
            Threads.@spawn put!(c, sum(input .+ i))
        end
    end
    close(c)
end
sum(collect(c))

# output
225
```
"""
function with_locked_bumper(f, buf::AllocBuffer)
    return with(f, CURRENT_ALLOCATOR => LockedBumperAllocator(buf))
end
