
struct CheckedBumperAllocator{B<:AllocBuffer} <: Allocator
    buf::B
    mems::Vector{MemValid}
    lock::ReentrantLock
end

function CheckedBumperAllocator(B::AllocBuffer)
    return CheckedBumperAllocator(B, MemValid[], ReentrantLock())
end

Base.lock(B::CheckedBumperAllocator) = lock(B.lock)
Base.unlock(B::CheckedBumperAllocator) = unlock(B.lock)

function alloc_checked_similar(B::CheckedBumperAllocator, arr, ::Type{T}, dims::Dims) where {T}
    @lock B begin
        # ignore arr type for now
        inner = Bumper.alloc(T, B.buf, dims...)
        valid = MemValid(true)
        push!(B.mems, valid)
        return inner, valid
    end
end

function alloc_checked_similar(B::CheckedBumperAllocator, ::Type{Arr}, dims::Dims) where {Arr}
    @lock B begin
        inner = Bumper.alloc(eltype(Arr), B.buf, dims...)
        valid = MemValid(true)
        push!(B.mems, valid)
        return inner, valid
    end
end

function reset!(B::CheckedBumperAllocator)
    @lock B begin
        # Invalidate all memory first
        for mem in B.mems
            invalidate!(mem)
        end
        # Then reset the buffer
        Bumper.reset_buffer!(B.buf)
    end
    return nothing
end

function with_checked_bumper_no_escape(f, buf::AllocBuffer)
    B = CheckedBumperAllocator(buf)
    ret = with(f, CURRENT_ALLOCATOR => B)
    reset!(B)
    return ret
end
