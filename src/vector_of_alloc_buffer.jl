const DEFAULT_NEW_BUFFER_SIZE = 2^27 # 128 MB
const DEFAULT_HISTORY_SIZE = 10
const DEFAULT_DESIRED_MAX_BUFFERS = 3

mutable struct VectorOfAllocBuffer
    const v::Vector{AllocBuffer{Vector{UInt8}}}
    new_buffer_size::Int
    const sz_history::Vector{Int}
    max_history_size::Int
    desired_max_buffers::Int
end

function update_size!(v::VectorOfAllocBuffer, new_sz::Int)
    if length(v.sz_history) > v.max_history_size
        popfirst!(v.sz_history)
    end
    push!(v.sz_history, v.new_buffer_size)
    v.new_buffer_size = new_sz
    return nothing
end

function VectorOfAllocBuffer(sz::Int)
    return VectorOfAllocBuffer([AllocBuffer(sz)], sz, [sz], DEFAULT_HISTORY_SIZE,
                               DEFAULT_DESIRED_MAX_BUFFERS)
end
VectorOfAllocBuffer() = VectorOfAllocBuffer(DEFAULT_NEW_BUFFER_SIZE)

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

amount_left(b) = total_capacity(b) - amount_used(b) # TODO check for off-by-one errors

function Bumper.alloc_ptr!(v::VectorOfAllocBuffer, sz::Int)::Ptr{Cvoid}
    # see if any buffer has enough space, using the first we find
    for b in v.v
        if amount_left(b) >= sz
            return Bumper.alloc_ptr!(b, sz)
        end
    end
    # allocate a new buffer, defaulting to `v.new_buffer_size`, but
    # making it larger if necessary to accommodate `sz`
    b = AllocBuffer(max(v.new_buffer_size, sz))
    pushfirst!(v.v, b) # start of the queue
    return Bumper.alloc_ptr!(b, sz)
end

function Bumper.reset_buffer!(v::VectorOfAllocBuffer)
    if length(v.v) > v.desired_max_buffers
        reallocate!(v)
    else
        foreach(Bumper.reset_buffer!, v.v)
    end
    return nothing
end

function reallocate!(v::VectorOfAllocBuffer)
    old_sz = v.new_buffer_size
    old_n_buffers = length(v.v)
    t = @elapsed begin
        proposed_new_size = ceil(Int, amount_used(v) * 1.05)
        new_size = max(proposed_new_size, maximum(v.sz_history))
        update_size!(v, new_size)
        empty!(v.v)
        push!(v.v, AllocBuffer(v.new_buffer_size))
    end
    @debug "[AllocArray.reallocate!] Had $(old_n_buffers) buffers (desired max: $(v.desired_max_buffers)), so reallocated from size $(Base.format_bytes(old_sz)) to $(Base.format_bytes(v.new_buffer_size)) in $t seconds"
    return nothing
end

function Bumper.checkpoint_save(v::VectorOfAllocBuffer)
    error("VectorOfAllocBuffer does not support checkpointing")
end
