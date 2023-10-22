module AllocArrays

using ScopedValues: ScopedValue, scoped
using Bumper
using Adapt

# for some piracy
using StrideArraysCore: StrideArraysCore

export AllocArray, with_bumper

include("alloc_interface.jl")

# make more type stable...
# const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)
const CURRENT_ALLOCATOR = Ref{BumperAllocator{Vector{UInt8}}}()

function __init__()
    CURRENT_ALLOCATOR[] = BumperAllocator(Bumper.default_buffer())
    return nothing
end

include("AllocArray.jl")

end # module
