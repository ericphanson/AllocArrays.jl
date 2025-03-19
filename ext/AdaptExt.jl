module AdaptExt
using Adapt, AllocArrays

# AllocArrays should be seen as "storage" like CuArray, not a wrapper like Transpose,
# since we want to recurse through models and convert arrays to AllocArrays, so that
# in the forward pass we can use our bump allocator
Adapt.adapt_storage(::Type{<:AllocArray}, xs::AbstractArray) = AllocArray(xs)

end # AdaptExt
