const DEFAULT_NEW_BUFFER_SIZE = 2^27 # 128 MB
const DEFAULT_HISTORY_SIZE = 5
const DEFAULT_DESIRED_MAX_BUFFERS = 1

mutable struct VectorOfAllocBuffer
    const v::Vector{AllocBuffer{Vector{UInt8}}}
    new_buffer_size::Int
    const sz_history::Vector{Int}
    max_history_size::Int
    desired_max_buffers::Int
end

"""
    VectorOfAllocBuffer(new_buffer_size::Int=$DEFAULT_NEW_BUFFER_SIZE;
                        max_history_size=$DEFAULT_HISTORY_SIZE,
                        desired_max_buffers=$DEFAULT_DESIRED_MAX_BUFFERS)

Construct a `VectorOfAllocBuffer`. This uses an `AllocBuffer` of size `sz`. If an allocation
of size `allocation_size` is requested and no buffer has enough space, it will allocate a new
buffer of size `max(sz, 2*allocation_size)`, and set the internal `new_buffer_size` parameter to that amount.

When `reset_buffer!` is called, if there are more than `desired_max_buffers` buffers, it will
create a new buffer whose size is the maximum of the amount of actual space used (times 1.1) and the
maximum of the previous buffer size (stored in the history, which stores the last `max_history_size` sizes).
Additionally, the `new_buffer_size` parameter will be set to half of this size, meaning any additional allocations on the next run may increase the total memory usage by 50% (unless they are large!).

This means:

- `VectorOfAllocBuffer` does not run out of memory (unlike `AllocBuffer`), since new memory will allocated on-demand when necessary
- for repeated runs of the same size, a single contiguous buffer will be used (when `desired_max_buffers==1`), which should approximately match the performance of a tuned `AllocBuffer`
- `VectorOfAllocBuffer` can reuse allocated memory between runs like `AllocBuffer`, but without potentially OOMing like `SlabBuffer`. Additionally, `VectorOfAllocBuffer` separately tracks the memory used by the first buffer vs the additional buffers allocated dynamically, so small unexpected additional allocations don't double the memory consumption (unlike a second slab being allocated).
"""
function VectorOfAllocBuffer(sz::Int=DEFAULT_NEW_BUFFER_SIZE;
                             max_history_size=DEFAULT_HISTORY_SIZE,
                             desired_max_buffers=DEFAULT_DESIRED_MAX_BUFFERS)
    return VectorOfAllocBuffer([AllocBuffer(sz)], sz, [sz], max_history_size,
                               desired_max_buffers)
end

# Record the size to the history without letting the history get too big
function record_size!(v::VectorOfAllocBuffer, new_sz::Int)
    if length(v.sz_history) > v.max_history_size
        popfirst!(v.sz_history)
    end
    push!(v.sz_history, new_sz)
    return nothing
end

function total_capacity(b::AllocBuffer)
    return length(b.buf)
end

function amount_used(b::AllocBuffer)
    return b.offset
end

function total_capacity(v::VectorOfAllocBuffer)
    return sum(total_capacity, v.v)
end

function amount_used(v::VectorOfAllocBuffer)
    return sum(amount_used, v.v)
end

amount_left(b) = total_capacity(b) - amount_used(b)

function Bumper.alloc_ptr!(v::VectorOfAllocBuffer, sz::Int)::Ptr{Cvoid}
    # fastest path: use first buffer if possible
    b = first(v.v)
    amount_left(b) >= sz && return Bumper.alloc_ptr!(b, sz)
    # medium path: use first available buffer
    for b in v.v
        if amount_left(b) >= sz
            return Bumper.alloc_ptr!(b, sz)
        end
    end
    # slow path: allocate new memory
    return slow_path_alloc_ptr!(v, sz)
end

@noinline function slow_path_alloc_ptr!(v::VectorOfAllocBuffer, sz::Int)::Ptr{Cvoid}
    # allocate a new buffer, defaulting to `v.new_buffer_size`, but
    # tuning the size based on the allocatin coming in
    new_sz = max(v.new_buffer_size, 2 * sz)
    v.new_buffer_size = new_sz
    b = AllocBuffer(v.new_buffer_size)
    pushfirst!(v.v, b) # start of the queue
    @debug "[AllocArray.slow_path_alloc_ptr!] Fulfilling allocation of $(Base.format_bytes(sz)) with additional buffer of size $(Base.format_bytes(new_sz))"
    return Bumper.alloc_ptr!(b, sz)
end

function Bumper.reset_buffer!(v::VectorOfAllocBuffer)
    if length(v.v) > v.desired_max_buffers
        reallocate!(v)
    end
    foreach(Bumper.reset_buffer!, v.v)
    return nothing
end

# we will replace all our existing buffers with 1 big buffer
function reallocate!(v::VectorOfAllocBuffer)
    old_sz = v.new_buffer_size
    old_n_buffers = length(v.v)
    t = @elapsed begin
        proposed_new_size = ceil(Int, amount_used(v) * 1.1)
        new_size = max(proposed_new_size, maximum(v.sz_history))
        record_size!(v, new_size)
        empty!(v.v)
        push!(v.v, AllocBuffer(new_size))
        # if we need to allocate additional buffers still, lets do a smaller amount
        # so that we don't way over-allocate
        v.new_buffer_size = cld(new_size, 2)
    end
    @debug "[AllocArray.reallocate!] Had $(old_n_buffers) buffers (desired max: $(v.desired_max_buffers)), so reallocated from size $(Base.format_bytes(old_sz)) to $(Base.format_bytes(v.new_buffer_size)) in $t seconds"
    return nothing
end

function Bumper.checkpoint_save(v::VectorOfAllocBuffer)
    return error("VectorOfAllocBuffer does not support checkpointing")
end
