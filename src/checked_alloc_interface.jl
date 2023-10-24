# TODO-
# This should wrap a `BumperAllocator`
# Then if get an `AllocArray`, we just forward to that and don't do any checking
# If we get a `CheckedAllocArray`, we go through the checking route
# Then there are 3 options:
# - BumperAllocator: no protection
# - LockedBumperAllocator: wraps BumperAllocator, adds concurrency protection
# - CheckedBumperAllocator: wraps BumperAllocator, adds both concurrency and memory safety
struct CheckedBumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
    mems::Vector{MemValid}
    # this lock protects access to `mems` and `buf`, and ensures
    # we can make atomic "transactions" to add new memory or invalidate all memory
    lock::ReentrantLock
end

function CheckedBumperAllocator(B::AllocBuffer)
    return CheckedBumperAllocator(B, MemValid[], ReentrantLock())
end

Base.lock(B::CheckedBumperAllocator) = lock(B.lock)
Base.unlock(B::CheckedBumperAllocator) = unlock(B.lock)

function alloc_similar(B::CheckedBumperAllocator, ::CheckedAllocArray, ::Type{T},
                       dims::Dims) where {T}
    @lock B begin
        inner = Bumper.alloc(T, B.buf, dims...)
        valid = MemValid(true)
        push!(B.mems, valid)
        return CheckedAllocArray(inner, valid)
    end
end

function alloc_similar(B::CheckedBumperAllocator, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    @lock B begin
        inner = Bumper.alloc(T, B.buf, dims...)
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

function reset!(B::GenericBumper)
    reset!(B.checked_bumper) # first, in case they share memory
    reset!(B.bumper)
    return nothing
end

function reset!(B::BumperAllocator)
    Bumper.reset_buffer!(B.buf)
    return nothing
end

function reset!(B::CheckedBumperAllocator)
    @lock B begin
        # Invalidate all memory first
        for mem in B.mems
            invalidate!(mem)
        end
        # Then reset the buffer
        Bumper.reset_buffer!(B.buf)

        # Then empty the tracked memory
        empty!(B.mems)
    end
    return nothing
end

"TODO"
function with_checked_bumper_no_escape(f, buf::AllocBuffer)
    B = CheckedBumperAllocator(buf)
    ret = with(f, CURRENT_ALLOCATOR => B)
    reset!(B)
    return ret
end

struct GenericBumper{B1<:BumperAllocator,B2<:CheckedBumperAllocator}
    bumper::B1
    checked_bumper::B2
end

function bumper(buf::AllocBuffer)
    return LockedBumperAllocator(GenericBumper(BumperAllocator(buf),
                                               CheckedBumperAllocator(buf)))
end

function alloc_similar(B::GenericBumper, ::Type{CheckedAllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    return alloc_similar(B.checked_bumper, CheckedAllocArray{T,N,Arr}, dims)
end

function alloc_similar(B::GenericBumper, ::Type{AllocArray{T,N,Arr}},
                       dims::Dims) where {T,N,Arr}
    return alloc_similar(B.bumper, AllocArray{T,N,Arr}, dims)
end

function alloc_similar(B::GenericBumper, a::CheckedAllocArray, ::Type{T},
                       dims::Dims) where {T}
    return alloc_similar(B.checked_bumper, a, T, dims)
end

function alloc_similar(B::GenericBumper, a::AllocArray, ::Type{T},
                       dims::Dims) where {T}
    return alloc_similar(B.bumper, a, T, dims)
end

# function with_bumper(f, buf::AllocBuffer)
#     return with(f, CURRENT_ALLOCATOR => bumper(buf))
# end


function with_bumper(f, b)
    return with(f, CURRENT_ALLOCATOR => b)
end
