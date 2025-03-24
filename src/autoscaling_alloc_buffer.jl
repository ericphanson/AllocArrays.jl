const DEFAULT_INITIAL_BUFFER_SIZE = 2^27 # 128 MB
const DEFAULT_HISTORY_SIZE = 5

mutable struct AutoscalingAllocBuffer
    main_buffer::AllocBuffer{Vector{UInt8}}
    const additional_buffers::Vector{AllocBuffer{Vector{UInt8}}}
    new_buffer_size::Int
    const total_used_history::Vector{Int}
    max_history_size::Int
end

"""
    AutoscalingAllocBuffer(initial_buffer_size::Int=$DEFAULT_INITIAL_BUFFER_SIZE;
                           max_history_size=$DEFAULT_HISTORY_SIZE)

Construct a `AutoscalingAllocBuffer`. This constructs an `AllocBuffer` of size `initial_buffer_size`. If an allocation (of some size `allocation_size`) is requested and there is not enough space, it will allocate a new buffer (whose size is determined by internal heuristics).

When `reset_buffer!` is called, if there are additional buffers, a new larger main buffer will be created so that if similar sized allocations occur on the next run no additional buffers will be needed. It will remember the amount of allocations used in the past `max_history_size` runs to inform the size of the new main buffer, to ensure there is at least enough room for each of those past runs.

This means:

- `AutoscalingAllocBuffer` does not run out of memory (unlike `AllocBuffer`), since new memory will allocated on-demand when necessary
- for repeated runs of the same size, a single contiguous buffer will be used, which should approximately match the performance of a tuned `AllocBuffer`
- `AutoscalingAllocBuffer` can reuse allocated memory between runs like `AllocBuffer`, but without potentially OOMing like `AutoscalingAllocBuffer`. Additionally, `AutoscalingAllocBuffer` separately tracks the memory used by the main buffer vs the additional buffers allocated dynamically, so small unexpected additional allocations don't double the memory consumption (unlike a second slab being allocated).
"""
function AutoscalingAllocBuffer(sz::Int=DEFAULT_INITIAL_BUFFER_SIZE;
                                max_history_size=DEFAULT_HISTORY_SIZE)
    return AutoscalingAllocBuffer(AllocBuffer(sz), [], cld(sz, 2), [sz], max_history_size)
end

# Record the size to the history without letting the history get too big
function record_size!(v::AutoscalingAllocBuffer, new_sz::Int)
    if length(v.total_used_history) > v.max_history_size
        popfirst!(v.total_used_history)
    end
    push!(v.total_used_history, new_sz)
    return nothing
end

function total_capacity(b::AllocBuffer)
    return length(b.buf)
end

function amount_used(b::AllocBuffer)
    return Int(b.offset)
end

function total_capacity(v::AutoscalingAllocBuffer)
    return total_capacity(v.main_buffer) + sum(total_capacity, v.additional_buffers; init=0)
end

function amount_used(v::AutoscalingAllocBuffer)
    return amount_used(v.main_buffer) + sum(amount_used, v.additional_buffers; init=0)
end

amount_left(b) = total_capacity(b) - amount_used(b)

function Bumper.alloc_ptr!(v::AutoscalingAllocBuffer, sz::Int)::Ptr{Cvoid}
    # fastest path: use main buffer if possible
    amount_left(v.main_buffer) >= sz && return Bumper.alloc_ptr!(v.main_buffer, sz)
    # medium path: use another available buffer
    for b in v.additional_buffers
        if amount_left(b) >= sz
            return Bumper.alloc_ptr!(b, sz)
        end
    end
    # slow path: allocate new memory
    return slow_path_alloc_ptr!(v, sz)
end

@noinline function slow_path_alloc_ptr!(v::AutoscalingAllocBuffer, sz::Int)::Ptr{Cvoid}
    # the amount we choose here will determine how many additional buffers we need.
    # We want to choose at least `2 * v.new_buffer_size` so we exponentially grow
    # the amount per buffer so we don't need to allocate too many buffers.
    # we also want to allocate at least enough for `sz` to fulfill this one, and
    # we choose 2x so we still have some room after fulfilling this allocation.
    new_sz = max(2 * v.new_buffer_size, 2 * sz)
    v.new_buffer_size = new_sz
    b = AllocBuffer(new_sz)
    pushfirst!(v.additional_buffers, b) # start of the queue
    @debug "[AllocArray.slow_path_alloc_ptr!] Fulfilling allocation of $(Base.format_bytes(sz)) with additional buffer of size $(Base.format_bytes(new_sz))."
    return Bumper.alloc_ptr!(b, sz)
end

function propose_realloc_size(v::AutoscalingAllocBuffer)
    return ceil(Int, maximum(v.total_used_history) * 1.1)
end

function Bumper.reset_buffer!(v::AutoscalingAllocBuffer)
    record_size!(v, amount_used(v))
    proposed_sz = propose_realloc_size(v)
    # reallocate if we had additional buffers (means main buffer is not big enough),
    # or we want to downscale by more than 50%
    if !isempty(v.additional_buffers) ||
       (proposed_sz < cld(total_capacity(v.main_buffer), 2))
        reallocate!(v, proposed_sz)
    end
    Bumper.reset_buffer!(v.main_buffer)
    # if we had additional buffers left we'd have to reset them, but we don't
    return nothing
end

# we will replace all our existing buffers with 1 big buffer
function reallocate!(v::AutoscalingAllocBuffer, to_size::Int)
    old_main_buffer_size = total_capacity(v.main_buffer)
    old_new_buffer_size = v.new_buffer_size
    n_additional_buffers = length(v.additional_buffers)
    t = @elapsed begin
        empty!(v.additional_buffers)
        v.main_buffer = AllocBuffer(to_size)
        # if we need to allocate additional buffers still, lets do a smaller amount
        # so that we don't way over-allocate. If we need more buffers, we'll scale exponentially,
        # so we can start relatively small here at 25%
        v.new_buffer_size = cld(to_size, 4)
    end
    @debug "[AllocArray.reallocate!] Had $(n_additional_buffers) additional buffers. Reallocated main buffer from $(Base.format_bytes(old_main_buffer_size)) to $(Base.format_bytes(to_size)) in $t seconds. For future additional buffers, set `new_buffer_size` to $(Base.format_bytes(v.new_buffer_size)) (from $(Base.format_bytes(old_new_buffer_size)))."
    return nothing
end

function Bumper.checkpoint_save(v::AutoscalingAllocBuffer)
    return error("AutoscalingAllocBuffer does not support checkpointing")
end
