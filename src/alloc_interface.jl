abstract type Allocator end

struct DefaultAllocator <: Allocator end

const DEFAULT_ALLOCATOR = DefaultAllocator()

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

function alloc_similar(::DefaultAllocator, arr, ::Type{T}, dims::Dims) where {T}
    return similar(arr, T, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{Arr},
                       dims::Dims) where {Arr<:AbstractArray}
    return similar(Arr, dims)
end

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
