module AllocArrays

using ScopedValues: ScopedValue, scoped
using Bumper

export AllocArray, with_bumper

include("alloc_interface.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

include("AllocArray.jl")

end # module
