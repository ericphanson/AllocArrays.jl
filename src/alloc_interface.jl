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

function alloc_similar(::DefaultAllocator, arr, ::Type{T}, dims::Dims) where {T}
    return similar(arr, T, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{Arr},
                       dims::Dims) where {Arr<:AbstractArray}
    return similar(Arr, dims)
end

#####
##### Bumper.jl
#####

# Could be moved to a package extension?

struct BumperAllocator{B<:Union{Nothing,AllocBuffer}} <: Allocator
    buf::B
end

BumperAllocator() = BumperAllocator(nothing)

function alloc_similar(B::BumperAllocator, arr, ::Type{T}, dims::Dims) where {T}
    # ignore arr type for now
    return Bumper.alloc(T, @something(B.buf, default_buffer()), dims...)
end

function alloc_similar(B::BumperAllocator, ::Type{Arr}, dims::Dims) where {Arr}
    return Bumper.alloc(eltype(Arr), @something(B.buf, default_buffer()), dims...)
end

"""
    with_bumper(f)

Runs `f()` in the context of using a `BumperAllocator{Nothing}` to
allocate memory to `similar` calls on [`AllocArray`](@ref)s.

All such allocations should occur within an `@no_escape` block,
and of course, no such allocations should escape that block.

Thread-safe: `f` may spawn multiple tasks or threads, which may each allocate memory using `similar` calls on `AllocArray`'s.

## Example

```jldoctest
using AllocArrays, Bumper
input = AllocArray([1,2,3])
c = Channel(Inf)
with_bumper() do
    Threads.@threads for i = 1:10
        @no_escape begin
            # ...code with may be multithreaded but which must not escape or return newly-allocated AllocArrays...
            put!(c, sum(input .+ i))
        end
     end
     close(c)
end
sum(collect(c))

# output
225
```
"""
function with_bumper(f)
    return with(f, CURRENT_ALLOCATOR => BumperAllocator())
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

## Example

```jldoctest
using AllocArrays, Bumper
using AllocArrays: unsafe_with_bumper

input = AllocArray([1,2,3])
buf = default_buffer()
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

struct LockedBumperAllocator{B<:AllocBuffer} <: Allocator
    bumper::BumperAllocator{B}
    lock::ReentrantLock
end
function LockedBumperAllocator(buf::AllocBuffer)
    return LockedBumperAllocator(BumperAllocator(buf), ReentrantLock())
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

Thread-safe: `f` may spawn multiple tasks or threads, which may each allocate memory using `similar` calls on `AllocArray`'s.

## Example

```jldoctest
using AllocArrays, Bumper

buf = default_buffer()
input = AllocArray([1,2,3])
c = Channel(Inf)
with_locked_bumper(buf) do
    # ...code with may be multithreaded but which must not escape or return newly-allocated AllocArrays...
    @sync for i = 1:10
        @no_escape buf begin
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
