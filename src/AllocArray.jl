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

function Base.setindex!(a::AllocArray{T,N}, value, I::Vararg{Int,N}) where {T,N}
    return setindex!(a.arr, value, I...)
end
Base.getindex(a::AllocArray{T,N}, I::Vararg{Int,N}) where {T,N} = getindex(a.arr, I...)
Base.size(a::AllocArray) = size(a.arr)

function Base.BroadcastStyle(::Type{AllocArray{T,N,Arr}}) where {T,N,Arr}
    return Broadcast.ArrayStyle{AllocArray{T,N,Arr}}()
end

function Base.similar(bc::Broadcasted{ArrayStyle{AllocArray{T,N,Arr}}},
                      ::Type{ElType}) where {T,N,Arr,ElType}
    return similar(AllocArray{T,N,Arr}, axes(bc))
end

function Base.similar(::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T,N,Arr}
    inner = alloc_similar(CURRENT_ALLOCATOR[], Arr, dims)
    return AllocArray{T,N,Arr}(inner)
end

function Base.similar(A::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = alloc_similar(CURRENT_ALLOCATOR[], A.arr, T, dims)
    return AllocArray(inner)
end
