module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using Adapt
using ConcurrentUtilities

# Two array types
export AllocArray
export CheckedAllocArray, InvalidMemoryException

# how to use an allocator
export with_allocator

# default allocator
export DefaultAllocator

# bump allocator support
export BumperAllocator, UncheckedBumperAllocator, reset!

include("AllocArray.jl")
include("CheckedAllocArray.jl")
include("alloc_interface.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

end # module
