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
