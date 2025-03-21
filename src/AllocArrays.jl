module AllocArrays

using ScopedValues: ScopedValue, with
using Bumper
using ConcurrentUtilities
using PrecompileTools

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

# PrecompileTools workload
@setup_workload begin
    @compile_workload begin
        for alloc in (() -> BumperAllocator(), () -> BumperAllocator(2^10)) # (SlabBuffer, 1 KiB)
            b = alloc()
            a = AllocArray([1])
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
