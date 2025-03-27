# AllocArrays should be seen as "storage" like CuArray, not a wrapper like Transpose,
# since we want to recurse through models and convert arrays to AllocArrays, so that
# in the forward pass we can use our bump allocator
Adapt.adapt_storage(::Type{<:AllocArray}, xs::AbstractArray) = copy_to_alloc(xs)

function copy_to_alloc(xs::AbstractArray{T}) where {T}
    arr = similar(AllocArray{T}, size(xs))
    copyto!(arr, xs)
    return arr
end

struct CachedUpdatelessFunction{T,F,A}
    f::F
    allocator::A
end
function CachedUpdatelessFunction{T}(f; allocator=BumperAllocator()) where {T}
    return CachedUpdatelessFunction{T,typeof(f),typeof(allocator)}(f, allocator)
end
function CachedUpdatelessFunction(f; allocator=BumperAllocator())
    return CachedUpdatelessFunction{AllocArray}(f; allocator)
end

function (c::CachedUpdatelessFunction{T})(args...; kw...) where {T}
    with_allocator(c.allocator) do
        try
            tmp_f = Adapt.adapt(T, c.f)
            args = Adapt.adapt(T, args)
            kw = Adapt.adapt(T, kw)
            return Adapt.adapt(Array, tmp_f(args...; kw...))
        finally
            reset!(c.allocator)
        end
    end
end

function Base.show(io::IO, mime::MIME"text/plain", c::CachedUpdatelessFunction{T}) where {T}
    print(io, CachedUpdatelessFunction)
    if T != AllocArray
        print(io, "{$T}")
    end
    print(io, " with function\n")
    show(io, mime, c.f)
    print(io, "\nand allocator ")
    show(io, mime, c.allocator)
    return nothing
end

function Base.show(io::IO, c::CachedUpdatelessFunction{T}) where {T}
    print(io, CachedUpdatelessFunction)
    if T != AllocArray
        print(io, "{$T}")
    end
    print(io, "(")
    show(io, c.f)
    print(io, ", ")
    show(io, c.allocator)
    print(io, ")")
    return nothing
end
