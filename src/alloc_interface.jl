# Allocation interface

"""
    abstract type Allocator end

Alloactors need to subtype `Alloactor` and implement two methods of `alloc_similar`:

- `AllocArrays.alloc_similar(::Allocator, arr, ::Type{T}, dims::Dims)`
- `AllocArrays.alloc_similar(::Allocator, ::Type{Arr}, dims::Dims) where {Arr<:AbstractArray}`

where the latter is used by broadcasting.
"""
abstract type Allocator end

#####
##### Default allocator
#####

# Just dispatches to `similar`

struct DefaultAllocator <: Allocator end

const DEFAULT_ALLOCATOR = DefaultAllocator()

function alloc_similar(::DefaultAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    return similar(Array{T}, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return similar(Arr, dims)
end

function alloc_similar(::DefaultAllocator, ::CheckedAllocArray, ::Type{T}, dims::Dims) where {T}
    return similar(Array{T}, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return similar(Arr, dims)
end

#####
##### Bumper.jl
#####

# Could be moved to a package extension?


"""
    TODO-fixme
    unsafe_with_bumper(f, buf::AllocBuffer)

Runs `f()` in the context of using a `UncheckedBumperAllocator{typeof(buf)}` to
allocate memory to `similar` calls on [`AllocArray`](@ref)s.

All such allocations should occur within an `@no_escape` block,
and of course, no such allocations should escape that block.

!!! warning
    Not thread-safe. `f` must not allocate memory using `similar` calls on `AllocArray`'s
    across multiple threads or tasks.

Remember: if `f` calls into another package, you might not know if they use concurrency
or not! It is safer to use [`with_locked_bumper`](@ref) for this reason.

## Example

```jldoctest
using AllocArrays, Bumper
using AllocArrays: unsafe_with_bumper

input = AllocArray([1,2,3])
buf = AllocBuffer()
unsafe_with_bumper(buf) do
     @no_escape buf begin
        # ...code with must not allocate AllocArrays on multiple tasks via `similar` nor escape or return newly-allocated AllocArrays...
        sum(input .* 2)
     end
end

# output
12
```
"""
struct UncheckedBumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
end

function reset!(B::UncheckedBumperAllocator)
    Bumper.reset_buffer!(B.buf)
    return nothing
end

function alloc_similar(B::UncheckedBumperAllocator, ::AllocArray, ::Type{T}, dims::Dims) where {T}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end

function alloc_similar(B::UncheckedBumperAllocator, ::Type{AllocArray{T,N,Arr}}, dims::Dims) where {T, N, Arr}
    inner = Bumper.alloc(T, B.buf, dims...)
    return AllocArray(inner)
end
