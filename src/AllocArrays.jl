module AllocArrays

using ScopedValues: ScopedValue, scoped
using Bumper

export AllocArray, with_bumper

include("alloc_interface.jl")
include("AllocArray.jl")

end # module
