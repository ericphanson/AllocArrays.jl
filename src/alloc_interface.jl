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

# Could be moved to a package extension

# We could add a field for the buffer instead of relying
# on the default buffer. That might be important for working with CuArrays etc
struct BumperAllocator <: Allocator end

function alloc_similar(::BumperAllocator, arr, ::Type{T}, dims::Dims) where {T}
    # ignore arr type for now
    return Bumper.alloc(T, dims...)
end

function alloc_similar(::BumperAllocator, ::Type{Arr}, dims::Dims) where {Arr}
    return Bumper.alloc(eltype(Arr), dims...)
end

function with_bumper(f)
    return scoped(f, CURRENT_ALLOCATOR => BumperAllocator())
end
