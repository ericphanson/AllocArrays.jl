module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using Adapt
using ConcurrentUtilities

export AllocArray, with_locked_bumper
export CheckedAllocArray, with_checked_bumper_no_escape, InvalidMemoryException
export bumper, with_bumper, reset!

include("AllocArray.jl")
include("alloc_interface.jl")

include("CheckedAllocArray.jl")
include("checked_alloc_interface.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

end # module
