using Base: Base.Broadcast
using Base: Dims
using Base.Broadcast: Broadcasted, ArrayStyle

"""
    struct AllocArray{T, N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
        arr::A
    end

Wrapper type which forwards most array methods to the inner array `arr`,
but dispatches `similar` to special allocation methods.
"""
struct AllocArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
    arr::A
end

@inline Base.parent(a::AllocArray) = getfield(a, :arr)

@inline Base.@propagate_inbounds function Base.setindex!(a::AllocArray{T,N}, value,
                                                         I::Vararg{Int,N}) where {T,N}
    return setindex!(getfield(a, :arr), value, I...)
end

@inline Base.@propagate_inbounds function Base.getindex(a::AllocArray{T,N},
                                                        I::Vararg{Int,N}) where {T,N}
    return getindex(getfield(a, :arr), I...)
end
Base.size(a::AllocArray) = size(getfield(a, :arr))

# used only by broadcasting?
function Base.similar(::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T,N,Arr}
    # TODO- I think this shouldn't be `Array{T}`,
    # but if we do `Arr`, then we are passing dimensionality info we don't necessarily
    # want to respect here. We want some `generic_type(Arr)` where is `Array{T}`
    # for `Vector` etc, but would be `CuArray` for `CuVector` etc.
    inner = alloc_similar(CURRENT_ALLOCATOR[], Array{T}, dims)
    return AllocArray(inner)
end

function Base.similar(a::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = alloc_similar(CURRENT_ALLOCATOR[], getfield(a, :arr), T, dims)
    return AllocArray(inner)
end

#####
##### Broadcasting
#####

function Base.BroadcastStyle(::Type{AllocArray{T,N,Arr}}) where {T,N,Arr}
    return Broadcast.ArrayStyle{AllocArray{T,N,Arr}}()
end

function Base.similar(bc::Broadcasted{ArrayStyle{AllocArray{T,N,Arr}}},
                      ::Type{ElType}) where {T,N,Arr,ElType}
    return similar(AllocArray{T,N,Arr}, axes(bc))
end

#####
##### Adapt.jl
#####

Adapt.adapt_structure(to, x::AllocArray) = AllocArray(adapt(to, parent(x)))

#####
##### StridedArray interface
#####

function Base.unsafe_convert(::Type{Ptr{T}}, a::AllocArray) where {T}
    return Base.unsafe_convert(Ptr{T}, getfield(a, :arr))
end

Base.elsize(::Type{<:AllocArray{T,N,Arr}}) where {T,N,Arr} = Base.elsize(Arr)

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
