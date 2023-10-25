# Allocation interface

"""
    abstract type Allocator end

Allocators need to subtype `Allocator` and implement two methods of `alloc_similar`:

- `AllocArrays.alloc_similar(::Allocator, arr, ::Type{T}, dims::Dims)`
- `AllocArrays.alloc_similar(::Allocator, ::Type{Arr}, dims::Dims) where {Arr<:AbstractArray}`

where the latter is used by broadcasting.
"""
abstract type Allocator end

#####
##### DefaultAllocator
#####

# Just dispatches to `similar`

"""
    DefaultAllocator()

Represents the default Julia allocator.

Used to dispatch `similar` calls
for [`AllocArray`](@ref)s and [`CheckedAllocArray`](@ref)s
to allocate using the the default Julia allocator.

This allocator is used by default if another one is not
used via [`with_allocator`](@ref).
"""
struct DefaultAllocator <: Allocator end

const DEFAULT_ALLOCATOR = DefaultAllocator()

function alloc_similar(::DefaultAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    return AllocArray(similar(Array{T}, dims))
end

function alloc_similar(::DefaultAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return AllocArray(similar(Arr, dims))
end

function alloc_similar(::DefaultAllocator, ::CheckedAllocArray, ::Type{T}, dims::Dims) where {T}
    return CheckedAllocArray(similar(Array{T}, dims))
end

function alloc_similar(::DefaultAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    # We know the memory is valid since it was allocated with the
    # default allocator
    return CheckedAllocArray(similar(Arr, dims), MemValid(true))
end

#####
##### UncheckedBumperAllocator
#####

# Naive use of Bumper.jl

"""
    UncheckedBumperAllocator(b::AllocBuffer)

Use with [`with_allocator`](@ref) to dispatch `similar` calls
for [`AllocArray`](@ref)s to allocate using the buffer `b`,
an `AllocBuffer` provided by Bumper.jl.

Does not support [`CheckedAllocArray`](@ref).

This provides a naive & direct interface to allocating on the buffer
with no safety checks or locks.

This is unsafe to use if multiple tasks may be allocating simultaneously,
and using [`BumperAllocator`](@ref) is recommended in general.

Used with [`reset!`](@ref) to deallocate.

See also: [`BumperAllocator`](@ref).

## Example

```jldoctest
using AllocArrays, Bumper

input = AllocArray([1,2,3])
b = UncheckedBumperAllocator(AllocBuffer(2^24)) # 16 MiB
with_allocator(b) do
    # ...code with must not allocate AllocArrays on multiple tasks via `similar` nor escape or return newly-allocated AllocArrays...
    ret = sum(input .* 2)
    reset!(b)
    return ret
end

# output
12
```
"""
struct UncheckedBumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
end

"""
    UncheckedBumperAllocator(n_bytes::Int)

Shorthand for `UncheckedBumperAllocator(AllocBuffer(n_bytes))`.
"""
UncheckedBumperAllocator(n_bytes::Int) = UncheckedBumperAllocator(AllocBuffer(n_bytes))

"""
    reset!(B::UncheckedBumperAllocator)

Resets the `UncheckedBumperAllocator`, deallocating all of the arrays
created by it.

This must only be used if those arrays will not be accessed again.

It is not safe to deallocate on one task while using the allocator
to allocate on another task. Therefore this should only be called outside
of threaded regions of code.
"""
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

#####
##### BumperAllocator
#####

# Has safety checks
# - concurrecy safety: protects access to buffer with a lock
#   - this does not protect against `reset!` being called while a task is using the buffer!
#   - but this allow allocations to occur on multiple tasks
# - memory safety, when opted into with `CheckedAllocArray`
#   - when `CheckedAllocArray` is used, we keep a reference to the `MemValid` associated to that array
#   - on `reset!` we invalidate that object (using a write-lock)
#   - every access to a `CheckedAllocArray` uses a read-lock to `MemValid` to ensure the memory is still valid

"""
    BumperAllocator(b::AllocBuffer)

Use with [`with_allocator`](@ref) to dispatch `similar` calls
for [`AllocArray`](@ref)s and [`CheckedAllocArray`](@ref)s
to allocate using the buffer `b`, an `AllocBuffer` provided by
Bumper.jl.

Uses a lock to serialize allocations to the buffer `b`,
which should allow safe concurrent usage.

Used with [`reset!`](@ref) to deallocate. Note it is not safe
to deallocate while another task may be allocating, except with
[`CheckedAllocArray`](@ref)s which will error appropriately.

See also: [`UncheckedBumperAllocator`](@ref).

## Example

```jldoctest
using AllocArrays, Bumper

b = BumperAllocator(AllocBuffer(2^24)) # 16 MiB
input = AllocArray([1,2,3])
c = Channel(Inf)
with_allocator(b) do
    # ...code with may be multithreaded but which must not escape or return newly-allocated AllocArrays...
    @sync for i = 1:10
        Threads.@spawn put!(c, sum(input .+ i))
    end
    reset!(b) # called outside of threaded region
    close(c)
end
sum(collect(c))

# output
225
```
"""
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

"""
    BumperAllocator(n_bytes::Int)

Shorthand for `BumperAllocator(AllocBuffer(n_bytes))`.
"""
BumperAllocator(n_bytes::Int) = BumperAllocator(AllocBuffer(n_bytes))

Base.lock(B::BumperAllocator) = lock(B.lock)
Base.unlock(B::BumperAllocator) = unlock(B.lock)

# `CheckedAllocArray`

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

# `AllocArray`

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

"""
    reset!(b::BumperAllocator)

Resets the `BumperAllocator`, deallocating all of the arrays
created by it.

This must only be used if those arrays will not be accessed again.
However, [`CheckedAllocArray`](@ref)s allocated by this allocator will be marked
invalid, causing future accesses to them to error, as a safety feature.
[`AllocArray`](@ref)s have no such safety feature, and access to them
after `reset!` is unsafe.

It is also not safe to deallocate on one task while using the allocator
to allocate on another task. Therefore this should only be called outside
of threaded regions of code.
"""
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

"""
    with_allocator(f, allocator)

Run `f` within a dynamic scope such that `similar` calls to
[`AllocArray`](@ref)s and [`CheckedAllocArray`](@ref)s dispatch
to allocator `allocator`.

Used with allocators [`DefaultAllocator`](@ref), [`BumperAllocator`](@ref),
and [`UncheckedBumperAllocator`](@ref).
"""
function with_allocator(f, allocator)
    return with(f, CURRENT_ALLOCATOR => allocator)
end
