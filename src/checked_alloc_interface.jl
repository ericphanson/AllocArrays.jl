# TODO-
# This should wrap a `UncheckedBumperAllocator`
# Then if get an `AllocArray`, we just forward to that and don't do any checking
# If we get a `CheckedAllocArray`, we go through the checking route
# Then there are 3 options:
# - UncheckedBumperAllocator: no protection
# - OnlyLockedBumperAllocator: wraps UncheckedBumperAllocator, adds concurrency protection
# - BumperAllocator: wraps UncheckedBumperAllocator, adds both concurrency and memory safety
# ...
# this interface means the central object the user creates is these allocator types
# then they call our `reset!`. Not `@no_escape`.
# Should we have a `try/finally` interface?
struct BumperAllocator{B} <: Allocator
    bumper::B
    mems::Vector{MemValid}
    # this lock protects access to `mems` and `buf`, and ensures
    # we can make atomic "transactions" to add new memory or invalidate all memory
    lock::ReentrantLock
end

function BumperAllocator(B::AllocBuffer)
    return BumperAllocator(UncheckedBumperAllocator(B), MemValid[], ReentrantLock())
end

Base.lock(B::BumperAllocator) = lock(B.lock)
Base.unlock(B::BumperAllocator) = unlock(B.lock)

function alloc_similar(B::BumperAllocator, c::CheckedAllocArray, ::Type{T},
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

function alloc_similar(B::BumperAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    @lock B begin
        inner = alloc_similar(B.bumper, Arr, dims)
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end


function reset!(B::BumperAllocator)
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

# If we have a `BumperAllocator` and are asked to allocate an unchecked array
# then we can do that by dispatching to the inner bumper. We will still
# get the lock for concurrency-safety.
# I.e. we are acting just like a `OnlyLockedBumperAllocator` in this case.
function alloc_similar(B::BumperAllocator, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    return @lock(B, alloc_similar(B.bumper, AllocArray{T,N,Arr}, dims))
end

function alloc_similar(B::BumperAllocator, a::AllocArray, ::Type{T},
                       dims::Dims) where {T}
    return @lock(B, alloc_similar(B.bumper, a, T, dims))
end

bumper(b::AllocBuffer) = BumperAllocator(b)

function with_allocator(f, b)
    return with(f, CURRENT_ALLOCATOR => b)
end
