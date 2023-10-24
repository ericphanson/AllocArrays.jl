# TODO-
# This should wrap a `UncheckedBumperAllocator`
# Then if get an `AllocArray`, we just forward to that and don't do any checking
# If we get a `CheckedAllocArray`, we go through the checking route
# Then there are 3 options:
# - UncheckedBumperAllocator: no protection
# - LockedBumperAllocator: wraps UncheckedBumperAllocator, adds concurrency protection
# - CheckedBumperAllocator: wraps UncheckedBumperAllocator, adds both concurrency and memory safety
# ...
# this interface means the central object the user creates is these allocator types
# then they call our `reset!`. Not `@no_escape`.
# Should we have a `try/finally` interface?
struct CheckedBumperAllocator{B} <: Allocator
    bumper::B
    mems::Vector{MemValid}
    # this lock protects access to `mems` and `buf`, and ensures
    # we can make atomic "transactions" to add new memory or invalidate all memory
    lock::ReentrantLock
end

function CheckedBumperAllocator(B::AllocBuffer)
    return CheckedBumperAllocator(UncheckedBumperAllocator(B), MemValid[], ReentrantLock())
end

Base.lock(B::CheckedBumperAllocator) = lock(B.lock)
Base.unlock(B::CheckedBumperAllocator) = unlock(B.lock)

function alloc_similar(B::CheckedBumperAllocator, c::CheckedAllocArray, ::Type{T},
                       dims::Dims) where {T}
    @lock B begin
        # I think we do still need C's read lock, bc it could have been
        # allocated by a DIFFERENT bumper which could be deallocating as we speak
        # so to safely access `c.alloc_array` we should still use the read lock
        inner = @lock(c, alloc_similar(B.bumper, _get_inner(c), T, dims))
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end

function alloc_similar(B::CheckedBumperAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    @lock B begin
        inner = alloc_similar(B.bumper, Arr, dims)
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end

function reset!(B::LockedBumperAllocator)
    @lock B begin
        reset!(B.bumper)
    end
    return nothing
end

function reset!(B::UncheckedBumperAllocator)
    Bumper.reset_buffer!(B.buf)
    return nothing
end

function reset!(B::CheckedBumperAllocator)
    @lock B begin
        # Invalidate all memory first
        for mem in B.mems
            invalidate!(mem)
        end
        # Then empty the tracked memory
        empty!(B.mems)

        # Then reset the inner bumper
        reset!(B.bumper)
    end
    return nothing
end

# If we have a `CheckedBumperAllocator` and are asked to allocate an unchecked array
# then we can do that by dispatching to the inner bumper. We will still
# get the lock for concurrency-safety.
# I.e. we are acting just like a `LockedBumperAllocator` in this case.
function alloc_similar(B::CheckedBumperAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    return @lock(B, alloc_similar(B.bumper, AllocArray{T,N,Arr}, dims))
end

function alloc_similar(B::CheckedBumperAllocator, a::AllocArray, ::Type{T},
                       dims::Dims) where {T}
    return @lock(B, alloc_similar(B.bumper, a, T, dims))
end

bumper(b::AllocBuffer) = CheckedBumperAllocator(b)

function with_allocator(f, b)
    return with(f, CURRENT_ALLOCATOR => b)
end


function alloc_similar(::DefaultAllocator, ::CheckedAllocArray, ::Type{T}, dims::Dims) where {T}
    return similar(Array{T}, dims)
end

function alloc_similar(::DefaultAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T, N, Arr}
    return similar(Arr, dims)
end
