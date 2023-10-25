using Base: Base.Broadcast
using Base: Dims
using Base.Broadcast: Broadcasted, ArrayStyle

# This is a separate mutable struct so it has its own identity
# If we create another `CheckedAllocArray` backed by the same memory
# as this one (e.g. via `reshape`), it gets a reference to the same
# `MemValid` object, so it has the same validity lifetime
mutable struct MemValid
    const lock::ReadWriteLock # from ConcurrentUtilities
    valid::Bool
end

function MemValid(valid)
    return MemValid(ReadWriteLock(), valid)
end

"""
    CheckedAllocArray(arr::AbstractArray)

Wrapper type which forwards most array methods to the inner array,
but dispatches `similar` to special allocation methods.

Keeps track of whether or not its memory is valid with `valid`.
All accesses to the array first check if the memory is still valid,
and throw an `InvalidMemoryException` if not.

If the array has memory allocated via a [`BumperAllocator`](@ref),
when the `BumperAllocator` is reset via [`reset!`](@ref), its memory
will be marked invalid.

Uses locks to ensure concurrency safety to avoid races between
deallocating memory on one task (with `reset!`) while accessing it on another task
(e.g. `getindex`). However, this array is as unsafe as any other to
write and read its contents simultaneously with multiple tasks. (i.e. locks
are used only to ensure validity of the memory backing the array when the
memory is accessed, not to remove data races when using the array as usual).

See also: [`AllocArray`](@ref).
"""
struct CheckedAllocArray{T,N,A<:AllocArray{T,N}} <: AbstractArray{T,N}
    alloc_array::A
    valid::MemValid
end

CheckedAllocArray(arr::AbstractArray) = CheckedAllocArray(AllocArray(arr), MemValid(true))

function CheckedAllocArray(::AllocArray)
    throw(ArgumentError("""
                        Cannot construct `CheckedAllocArray` from `AllocArray`, since
                        cannot be sure of the lifetime of the memory backing the object.
                        Why do you need this? File an issue.
                        """))
end

# idempotent I guess
CheckedAllocArray(c::CheckedAllocArray) = c

struct InvalidMemoryException <: Base.Exception end

function Base.showerror(io::IO, ::InvalidMemoryException)
    print(io, "InvalidMemoryException:")
    return print(io, " Array accessed after its memory has been deallocated.")
end

# We must be already inside the read lock
function _get_inner(a::CheckedAllocArray)
    is_valid = a.valid.valid
    is_valid || throw(InvalidMemoryException())
    return a.alloc_array
end

# Read lock
Base.lock(c::CheckedAllocArray) = readlock(c.valid.lock)
Base.unlock(c::CheckedAllocArray) = readunlock(c.valid.lock)

function invalidate!(valid::MemValid)
    # Writer lock
    @lock valid.lock begin
        valid.valid = false
    end
    return nothing
end

@inline function Base.parent(a::CheckedAllocArray)
    @lock(a, parent(_get_inner(a)))
end

@inline Base.@propagate_inbounds function Base.setindex!(a::CheckedAllocArray{T,N}, value,
                                                         I::Vararg{Int,N}) where {T,N}
    return @lock(a, setindex!(_get_inner(a), value, I...))
end

@inline Base.@propagate_inbounds function Base.setindex!(a::CheckedAllocArray{T,N}, value,
                                                         I::Int) where {T,N}
    return @lock(a, setindex!(_get_inner(a), value, I))
end

@inline Base.@propagate_inbounds function Base.getindex(a::CheckedAllocArray{T,N},
                                                        I::Vararg{Int,N}) where {T,N}
    return @lock(a, getindex(_get_inner(a), I...))
end

@inline Base.@propagate_inbounds function Base.getindex(a::CheckedAllocArray{T,N},
                                                        I::Int) where {T,N}
    return @lock(a, getindex(_get_inner(a), I))
end

function Base.size(a::CheckedAllocArray)
    return @lock(a, size(_get_inner(a)))
end

Base.IndexStyle(::Type{<:CheckedAllocArray{T,N,Arr}}) where {T,N,Arr} = Base.IndexStyle(Arr)

# used only by broadcasting?
function Base.similar(::Type{CheckedAllocArray{T,N,Arr}}, dims::Dims) where {T,N,Arr}
    return alloc_similar(CURRENT_ALLOCATOR[], CheckedAllocArray{T,N,Arr}, dims)
end

function Base.similar(a::CheckedAllocArray, ::Type{T}, dims::Dims) where {T}
    return alloc_similar(CURRENT_ALLOCATOR[], a, T, dims)
end

#####
##### Broadcasting
#####

function Base.BroadcastStyle(::Type{CheckedAllocArray{T,N,Arr}}) where {T,N,Arr}
    return Broadcast.ArrayStyle{CheckedAllocArray{T,N,Arr}}()
end

function Base.similar(bc::Broadcasted{ArrayStyle{CheckedAllocArray{T,N,Arr}}},
                      ::Type{ElType}) where {T,N,Arr,ElType}
    return similar(CheckedAllocArray{T,N,Arr}, axes(bc))::CheckedAllocArray
end

#####
##### Adapt.jl
#####

function Adapt.adapt_structure(to, a::CheckedAllocArray)
    inner = @lock(a, Adapt.adapt_structure(to, _get_inner(a)))
    return CheckedAllocArray(inner, a.valid)
end

#####
##### StridedArray interface
#####

function Base.unsafe_convert(::Type{Ptr{T}}, a::CheckedAllocArray) where {T}
    return @lock(a, Base.unsafe_convert(Type{Ptr{T}}, _get_inner(a)))
end

Base.elsize(::Type{<:CheckedAllocArray{T,N,Arr}}) where {T,N,Arr} = Base.elsize(Arr)

function Base.strides(a::CheckedAllocArray)
    return @lock(a, strides(_get_inner(a)))
end

#####
##### Other
#####

# Avoid reshaped arrays; saves quite a bit of time
function Base.reshape(a::CheckedAllocArray, args::Int...)
    inner = @lock(a, reshape(_get_inner(a), args...))
    return CheckedAllocArray(inner, a.valid)
end

function Base.reshape(a::CheckedAllocArray, dims::Dims)
    inner = @lock(a, reshape(_get_inner(a), dims))
    return CheckedAllocArray(inner, a.valid)
end

# Also helps avoid reshaped arrays
function Base.view(a::CheckedAllocArray{T,N},
                   i::Vararg{Union{Integer,AbstractRange,Colon},N}) where {T,N}
    inner = @lock(a, view(_get_inner(a), i))
    return CheckedAllocArray(inner, a.valid)
end
