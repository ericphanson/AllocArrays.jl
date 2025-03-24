```@meta
CurrentModule = AllocArrays
```

# AllocArrays

API Documentation for [AllocArrays](https://github.com/ericphanson/AllocArrays.jl).
See also the README at that link for more examples and notes.

## Public API

### Array types

```@docs
AllocArray
CheckedAllocArray
```

### Allocators

```@docs
BumperAllocator
with_allocator
reset!
```

We also provide an unsafe option.

```@docs
UncheckedBumperAllocator
```

```@docs
AllocArrays.DefaultAllocator
```

### Buffers

Here we provide `AutoscalingAllocBuffer` which is used by `BumperAllocator` by default. This builds upon Bumper.jl's `AllocBuffer` to grow as needed.

```@docs
AutoscalingAllocBuffer
```
