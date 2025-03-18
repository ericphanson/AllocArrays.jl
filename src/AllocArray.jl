using Base: Base.Broadcast
using Base: Dims
using Base.Broadcast: Broadcasted, ArrayStyle
using UnsafeArrays: UnsafeArray

"""
    struct AllocArray{T,N,R} <: DenseArray{T,N}
        arr::UnsafeArray{T,N}
        gcref::R
    end

    AllocArray(arr::AbstractArray)

Wrapper type which forwards most array methods to the inner array `arr`,
but dispatches `similar` to special allocation methods.

The inner array `arr` is always represented as an `UnsafeArray`,
but the field `gcref` may hold a materialized `AbstractArray` corresponding
to the same data, which is held to preserve a reference to the data to prevent GC.

Use the constructor `AllocArray(arr)` to construct an `AllocArray`. Note that `arr`
must be able to be represented as an `UnsafeArray`, meaning it must be a bits-type
and have a pointer. To support `UnitRange` and similar, `collect` it first.

Typically this constructor is only used at the entrypoint of a larger set of code
which is expected to use `similar` based on this input for further allocations.
When inside a `with_allocator` block, `similar` can be dispatched to a
(dynamically-scoped) bump allocator.
"""
struct AllocArray{T,N,R} <: DenseArray{T,N}
    arr::UnsafeArray{T,N}
    gcref::R

    function AllocArray(gcref::AbstractArray)
        arr = UnsafeArray(pointer(gcref), size(gcref))
        return new{eltype(arr),ndims(arr),typeof(gcref)}(arr, gcref)
    end

    # already allocated with Bumper, no gcref needed
    function AllocArray(arr::UnsafeArray)
        return new{eltype(arr),ndims(arr),Nothing}(arr, nothing)
    end
end

AllocMatrix{T} = AllocArray{T,2}
AllocVector{T} = AllocArray{T,1}

@inline Base.parent(a::AllocArray) = getfield(a, :arr)

@inline Base.@propagate_inbounds function Base.setindex!(a::AllocArray{T,N}, value,
                                                         I::Vararg{Int,N}) where {T,N}
    return setindex!(getfield(a, :arr), value, I...)
end

@inline Base.@propagate_inbounds function Base.setindex!(a::AllocArray{T,N}, value,
                                                         I::Int) where {T,N}
    return setindex!(getfield(a, :arr), value, I)
end

@inline Base.@propagate_inbounds function Base.getindex(a::AllocArray{T,N},
                                                        I::Vararg{Int,N}) where {T,N}
    return getindex(getfield(a, :arr), I...)
end

@inline Base.@propagate_inbounds function Base.getindex(a::AllocArray{T,N},
                                                        I::Int) where {T,N}
    return getindex(getfield(a, :arr), I)
end

Base.size(a::AllocArray) = size(getfield(a, :arr))

Base.IndexStyle(::Type{<:AllocArray{T,N}}) where {T,N} = Base.IndexStyle(UnsafeArray{T,N})

# used only by broadcasting?
function Base.similar(::Type{<:AllocArray{T}}, dims::Dims) where {T}
    return alloc_similar(CURRENT_ALLOCATOR[], AllocArray{T}, dims)
end

function Base.similar(a::AllocArray, ::Type{T}, dims::Dims) where {T}
    return alloc_similar(CURRENT_ALLOCATOR[], a, T, dims)
end

#####
##### Broadcasting
#####

function Base.BroadcastStyle(::Type{<:AllocArray})
    return ArrayStyle{AllocArray}()
end

function Base.similar(bc::Broadcasted{ArrayStyle{AllocArray}},
                      ::Type{T}) where {T}
    return similar(AllocArray{T}, axes(bc))
end

#####
##### Adapt.jl
#####

# Adapt.adapt_structure(to, x::AllocArray) = AllocArray(adapt(to, parent(x)))

#####
##### StridedArray interface
#####

function Base.unsafe_convert(::Type{Ptr{T}}, a::AllocArray) where {T}
    return Base.unsafe_convert(Ptr{T}, getfield(a, :arr))
end

Base.elsize(::Type{<:AllocArray{T,N}}) where {T,N} = Base.elsize(UnsafeArray{T,N})

Base.strides(a::AllocArray) = strides(getfield(a, :arr))

#####
##### Other
#####

# Avoid reshaped arrays; saves quite a bit of time
Base.reshape(a::AllocArray, args::Int...) = AllocArray(reshape(getfield(a, :arr), args...))
Base.reshape(a::AllocArray, dims::Dims) = AllocArray(reshape(getfield(a, :arr), dims))

# Also helps avoid reshaped arrays
function Base.view(a::AllocArray{T,N},
                   i::Vararg{Union{Integer,AbstractRange,Colon},N}) where {T,N}
    return AllocArray(view(getfield(a, :arr), i...))
end
