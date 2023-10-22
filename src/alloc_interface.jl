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

struct BumperAllocator{B} <: Allocator
    buf::AllocBuffer{B}
end

BumperAllocator() = BumperAllocator(Bumper.default_buffer())

function alloc_similar(B::BumperAllocator, arr, ::Type{T}, dims::Dims) where {T}
    # ignore arr type for now
    return Bumper.alloc(T, B.buf, dims...)
end

function alloc_similar(B::BumperAllocator, ::Type{Arr}, dims::Dims) where {Arr}
    return Bumper.alloc(eltype(Arr), B.buf, dims...)
end

function with_bumper(f, buf...)
    return with(f, CURRENT_ALLOCATOR => BumperAllocator(buf...))
end
