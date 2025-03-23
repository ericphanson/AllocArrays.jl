const DEFAULT_NEW_BUFFER_SIZE = 2^27 # 128 MB
const DEFAULT_HISTORY_SIZE = 5
const DEFAULT_DESIRED_MAX_BUFFERS = 1

mutable struct AutoscalingAllocBuffer
    main_buffer::AllocBuffer{Vector{UInt8}}
    const additional_buffers::Vector{AllocBuffer{Vector{UInt8}}}
    new_buffer_size::Int
    const main_buffer_size_history::Vector{Int}
    max_history_size::Int
    desired_max_buffers::Int
end

"""
    AutoscalingAllocBuffer(new_buffer_size::Int=$DEFAULT_NEW_BUFFER_SIZE;
                           max_history_size=$DEFAULT_HISTORY_SIZE,
                           desired_max_buffers=$DEFAULT_DESIRED_MAX_BUFFERS)

Construct a `AutoscalingAllocBuffer`. This constructs an `AllocBuffer` of size `new_buffer_size`. If an allocation (of some size `allocation_size`) is requested and there is not enough space, it will allocate a new buffer of size `max(new_buffer_size, 2*allocation_size)`, and set the internal `new_buffer_size` parameter to that amount. Thus, over the course of the run, additional buffers will be allocated as necessary, the size of which will adapt to the allocations incoming.

When `reset_buffer!` is called, if there are more than `desired_max_buffers` buffers, a new main buffer will be created, whose size is the maximum of the amount of actual space used in all buffers (times 1.1) and the
maximum of the previous main buffer sizes (stored in the history, which stores the last `max_history_size` sizes).
Additionally, the `new_buffer_size` parameter will be set to half of this size, meaning any additional allocations on the next run may increase the total memory usage by 50% (unless they are large!).

This means:

- `AutoscalingAllocBuffer` does not run out of memory (unlike `AllocBuffer`), since new memory will allocated on-demand when necessary
- for repeated runs of the same size, a single contiguous buffer will be used (when `desired_max_buffers==1`), which should approximately match the performance of a tuned `AllocBuffer`
- `AutoscalingAllocBuffer` can reuse allocated memory between runs like `AllocBuffer`, but without potentially OOMing like `SlabBuffer`. Additionally, `AutoscalingAllocBuffer` separately tracks the memory used by the main buffer vs the additional buffers allocated dynamically, so small unexpected additional allocations don't double the memory consumption (unlike a second slab being allocated).
"""
function AutoscalingAllocBuffer(sz::Int=DEFAULT_NEW_BUFFER_SIZE;
                                max_history_size=DEFAULT_HISTORY_SIZE,
                                desired_max_buffers=DEFAULT_DESIRED_MAX_BUFFERS)
    return AutoscalingAllocBuffer(AllocBuffer(sz), [], sz, [sz], max_history_size,
                                  desired_max_buffers)
end

# Record the size to the history without letting the history get too big
function record_size!(v::AutoscalingAllocBuffer, new_sz::Int)
    if length(v.main_buffer_size_history) > v.max_history_size
        popfirst!(v.main_buffer_size_history)
    end
    push!(v.main_buffer_size_history, new_sz)
    return nothing
end

function total_capacity(b::AllocBuffer)
    return length(b.buf)
end

function amount_used(b::AllocBuffer)
    return b.offset
end

function total_capacity(v::AutoscalingAllocBuffer)
    return total_capacity(v.main_buffer) + sum(total_capacity, v.additional_buffers)
end

function amount_used(v::AutoscalingAllocBuffer)
    return amount_used(v.main_buffer) + sum(amount_used, v.additional_buffers)
end

amount_left(b) = total_capacity(b) - amount_used(b)

function Bumper.alloc_ptr!(v::AutoscalingAllocBuffer, sz::Int)::Ptr{Cvoid}
    # fastest path: use main buffer if possible
    b = v.main_buffer
    amount_left(b) >= sz && return Bumper.alloc_ptr!(b, sz)
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
    old_new_buffer_size = v.new_buffer_size
    # allocate a new buffer, defaulting to `v.new_buffer_size`, but
    # tuning the size based on the allocatin coming in
    new_sz = max(v.new_buffer_size, 2 * sz)
    v.new_buffer_size = new_sz
    b = AllocBuffer(new_sz)
    pushfirst!(v.additional_buffers, b) # start of the queue
    @debug "[AllocArray.slow_path_alloc_ptr!] Fulfilling allocation of $(Base.format_bytes(sz)) with additional buffer of size $(Base.format_bytes(new_sz)). Set `new_buffer_size` to $(Base.format_bytes(v.new_buffer_size)) from $(Base.format_bytes(old_new_buffer_size))."
    return Bumper.alloc_ptr!(b, sz)
end

function Bumper.reset_buffer!(v::AutoscalingAllocBuffer)
    if length(v.additional_buffers) + 1 > v.desired_max_buffers
        reallocate!(v)
    end
    Bumper.reset_buffer!(v.main_buffer)
    foreach(Bumper.reset_buffer!, v.additional_buffers)
    return nothing
end

# we will replace all our existing buffers with 1 big buffer
function reallocate!(v::AutoscalingAllocBuffer)
    old_main_buffer_size = total_capacity(v.main_buffer)
    old_new_buffer_size = v.new_buffer_size
    old_n_buffers = length(v.additional_buffers) + 1
    t = @elapsed begin
        main_buffer_size = max(ceil(Int, amount_used(v) * 1.1),
                               maximum(v.main_buffer_size_history))
        record_size!(v, main_buffer_size)
        empty!(v.additional_buffers)
        v.main_buffer = AllocBuffer(main_buffer_size)
        # if we need to allocate additional buffers still, lets do a smaller amount
        # so that we don't way over-allocate
        v.new_buffer_size = cld(main_buffer_size, 2)
    end
    @debug "[AllocArray.reallocate!] Had $(old_n_buffers) buffers (desired max: $(v.desired_max_buffers)), so reallocated from main buffer size $(Base.format_bytes(old_main_buffer_size)) to $(Base.format_bytes(main_buffer_size)) in $t seconds. Set `new_buffer_size` to $(Base.format_bytes(v.new_buffer_size)) from $(Base.format_bytes(old_new_buffer_size))."
    return nothing
end

function Bumper.checkpoint_save(v::AutoscalingAllocBuffer)
    return error("AutoscalingAllocBuffer does not support checkpointing")
end
