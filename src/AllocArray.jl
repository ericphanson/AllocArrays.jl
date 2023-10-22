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

Base.parent(a::AllocArray) = a.arr

function Base.setindex!(a::AllocArray{T,N}, value, I::Vararg{Int,N}) where {T,N}
    return setindex!(parent(a), value, I...)
end
Base.getindex(a::AllocArray{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(parent(a), I...)
Base.size(a::AllocArray) = size(parent(a))

# used only by broadcasting?
function Base.similar(::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T,N,Arr}
    # TODO- I think this shouldn't be `Array{T}`,
    # but if we do `Arr`, then we are passing dimensionality info we don't necessarily
    # want to respect here. We want some `generic_type(Arr)` where is `Array{T}`
    # for `Vector` etc, but would be `CuArray` for `CuVector` etc.
    inner = alloc_similar(CURRENT_ALLOCATOR[], Array{T}, dims)
    return AllocArray(inner)
end

function Base.similar(A::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = alloc_similar(CURRENT_ALLOCATOR[], parent(A), T, dims)
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
    return Base.unsafe_convert(Ptr{T}, parent(a))
end

Base.elsize(::Type{<:AllocArray{T,N,Arr}}) where {T,N,Arr} = Base.elsize(Arr)

# Piracy - shouldn't this be defined on the type level in StrideArraysCore?
@inline Base.elsize(::Type{<:StrideArraysCore.AbstractStrideArray{T}}) where {T} = sizeof(T)

Base.strides(a::AllocArray) = strides(parent(a))
