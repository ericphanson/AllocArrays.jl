module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using Adapt

export AllocArray, with_bumper, with_locked_bumper

include("alloc_interface.jl")
include("AllocArray.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

end # module
