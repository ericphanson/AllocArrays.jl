# Allocation interface
# Alloactors need to subtype `Alloactor`
# and implement two methods of `alloc_similar`:
# `alloc_similar(::Allocator, arr, ::Type{T}, dims::Dims)`
# and
# `alloc_similar(::Allocator, ::Type{Arr}, dims::Dims) where {Arr<:AbstractArray}`
# where the latter is used by broadcasting.

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

function with_bumper(f, buf...)
    return with(f, CURRENT_ALLOCATOR => BumperAllocator(buf...))
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
function LockedBumperAllocator(buf=default_buffer())
    return LockedBumperAllocator(BumperAllocator(buf), ReentrantLock())
end
Base.lock(f::Function, B::LockedBumperAllocator) = lock(f, B.lock)
Base.lock(B::LockedBumperAllocator) = lock(B.lock)
Base.unlock(B::LockedBumperAllocator) = unlock(B.lock)

function alloc_similar(B::LockedBumperAllocator, args...)
    return @lock(B, alloc_similar(B.bumper, args...))
end
function with_locked_bumper(f, buf...)
    return with(f, CURRENT_ALLOCATOR => LockedBumperAllocator(buf...))
end
