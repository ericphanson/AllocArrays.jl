module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using Adapt

# for some piracy
using StrideArraysCore: StrideArraysCore

export AllocArray, with_bumper

include("alloc_interface.jl")
include("AllocArray.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

end # module
