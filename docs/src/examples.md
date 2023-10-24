
## Examples of improper usage

In the following, we will use our functions:

```@repl ex
using AllocArrays, Bumper

function some_allocating_function(a)
    b = similar(a)
    b .= a
    c = similar(a)
    c .= a
    return (; b, c)
end

function basic_reduction(a)
    (; b, c) = some_allocating_function(a)
    return sum(b .+ c)
end
```

### Wrong: allowing escapes

Allocations inside `@no_escape` must not escape!

```@repl ex
function bad_function_1(a)
    b = bumper(AllocBuffer(2^25))
    output = []
    with_bumper(b) do
        result = some_allocating_function(a)
        push!(output, result.b) # wrong! `b` is escaping `@no_escape`!
        reset!(b)
    end
    return sum(output...)
end

bad_function_1(CheckedAllocArray([1]))
```

Here is a corrected version:

```@repl ex
function good_function_1(a)
    buf = AllocBuffer()

    # note, we are not inside `with_bumper`, so we are not making buffer-backed memory
    output = similar(a)

    with_bumper(buf) do
        @no_escape buf begin
            result = some_allocating_function(a)
            output .= result.b # OK! we are copying buffer-backed memory into our heap-allocated memory
        end
    end
    return sum(result)
end

good_function_1(AllocArray([1]))

```

### Wrong: resetting a buffer in active use

```@repl ex
function bad_function_2(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_bumper(buf) do
        @sync for _ = 1:10
            Threads.@spawn begin
                @no_escape buf begin
                    scalar = basic_reduction(a)
                    put!(output, scalar)
                end # wrong! we cannot reset here as `buf` is being used on other tasks
            end
        end
    end
    close(output)
    return sum(collect(output))
end

bad_function_2(CheckedAllocArray([1]))
```

Here is a corrected version:

```@repl ex
function good_function_2(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_bumper(buf) do
        @no_escape buf begin
            @sync for _ = 1:10
                Threads.@spawn begin
                    scalar = basic_reduction(a)
                    put!(output, scalar)
                end
            end
        end # OK! resetting once we no longer need the allocations
    end
    close(output)
    return sum(collect(output))
end

good_function_2(AllocArray([1]))
```

Or, if we need to reset multiple times as we process the data, we could do a serial version:

```@repl ex
function good_function_2b(a)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_bumper(buf) do
        for _ = 1:10
            @no_escape buf begin
                scalar = basic_reduction(a)
                put!(output, scalar)
            end # OK to reset here! buffer-backed memory is not being used
        end
    end
    close(output)
    return sum(collect(output))
end

good_function_2b(AllocArray([1]))
```

We could also do something in-between, by launching tasks in batches of `n`, then resetting the buffer between them. Or we could use multiple buffers.

### Wrong: neglecting to reset the buffer

As shown above, we must be careful about when we reset the buffer. However, if we never reset it (analogous to never garbage collecting), we run into another problem which is we will run out memory!

```julia
function bad_function_3(a, N)
    buf = AllocBuffer()
    output = Channel(Inf)
    with_bumper(buf) do
        for _ = 1:N # bad! we are going to allocate `N` times without resetting!
            # if `N` is very large, we will run out of memory.
            scalar = basic_reduction(a)
            put!(output, scalar)
        end
    end
    close(output)
    return sum(collect(output))
end
```

This can be fixed by resetting appropriately, as in e.g. `good_function_2b` above.
