using AllocArrays, BenchmarkTools, UnsafeArrays

function mycopy(dest, src, iter)
    # precondition: dest and src do not alias
    # precondition: the iterators of dest and src are equal
    for i in iter
        @inbounds dest[i] = src[i]
    end
    return dest
end

b = BumperAllocator(2^30);
arr = rand(Float32, 50000000);
arr2 = similar(arr);
a = AllocArray(arr);

println("With mycopy(arr2, arr, eachindex(arr2)):")
display(@benchmark mycopy(arr2, arr, eachindex(arr2)))

println("With a = AllocArray(a)")
display(@benchmark with_allocator(b) do
    c = similar(a)
    mycopy(c, a, eachindex(c))
end setup=(reset!(b)) evals=1)

# julia> include("bench.jl")
# With mycopy(arr2, arr, eachindex(arr2)):
# BenchmarkTools.Trial: 627 samples with 1 evaluation per sample.
#  Range (min … max):  6.662 ms … 38.670 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     7.151 ms              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   7.950 ms ±  2.168 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

#   █▇                                                          
#   ███▅▄▄▃▄▄▃▃▃▃▃▃▃▃▄▆▄▃▃▃▂▃▃▃▃▃▃▂▂▁▂▁▂▂▁▁▂▁▂▁▁▂▂▁▁▁▁▁▁▂▁▁▁▁▂ ▃
#   6.66 ms        Histogram: frequency by time          14 ms <

#  Memory estimate: 16 bytes, allocs estimate: 1.
# With a = AllocArray(a)
# BenchmarkTools.Trial: 698 samples with 1 evaluation per sample.
#  Range (min … max):  6.659 ms …  17.031 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     6.789 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   7.166 ms ± 941.846 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

#   █▇▄▄▂ ▁   ▁▁    ▁                                            
#   ██████████████▇▇██▆▇▇▆▇█▅▇▇▇▅▆▆▁▄▁▄▆▁▄▁▆▄▅▅▁▆▅▄▆▅▄▁▆▅▅▁▁▄▅▄ ▇
#   6.66 ms      Histogram: log(frequency) by time      10.1 ms <

#  Memory estimate: 384 bytes, allocs estimate: 13.
