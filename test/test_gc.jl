using AllocArrays, Test

function foo()
    arr = zeros(1000, 1000)
    a = AllocArray(arr)
    GC.gc()
    rand(1000, 1000)
    rand(1000, 1000)
    return a[1, 1]
end

function loop(verbose=false)
    for i in 1:20
        verbose && println(i)
        f = foo()
        if !iszero(f)
            @show f
        end
        @test f == 0.0
    end
end

loop()

GC.gc()
GC.gc()

@test isempty(AllocArrays.GC_ROOTS)

if Threads.nthreads() > 1
    println("Running multithreaded gc test")
    Threads.@threads for i=1:2
        @info "Running on $(Threads.threadid()) for iteration $i"
        loop()
    end
end
