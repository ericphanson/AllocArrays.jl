module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using ConcurrentUtilities
using PrecompileTools
using Adapt

# Two array types
export AllocArray
export CheckedAllocArray, InvalidMemoryException

# how to use an allocator
export with_allocator

# default allocator
export DefaultAllocator

# bump allocator support
export BumperAllocator, UncheckedBumperAllocator, reset!

# our default allocator
export AutoscalingAllocBuffer

include("AllocArray.jl")
include("CheckedAllocArray.jl")
include("alloc_interface.jl")
include("autoscaling_alloc_buffer.jl")
include("adapt.jl")

const CURRENT_ALLOCATOR = ScopedValue{Allocator}(DEFAULT_ALLOCATOR)

# PrecompileTools workload
@setup_workload begin
    @compile_workload begin
        for alloc in (() -> BumperAllocator(), () -> BumperAllocator(2^10)) # (AutoscalingAllocBuffer, 1 KiB)
            b = alloc()
            a = AllocArray([1.0f0])
            c = CheckedAllocArray(a, MemValid(true))
            with_allocator(b) do
                similar(a) .= 1
                similar(c) .= 1
                reset!(b)
            end
        end
    end
end

end # module
