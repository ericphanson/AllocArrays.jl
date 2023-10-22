using AllocArrays
using Documenter

DocMeta.setdocmeta!(AllocArrays, :DocTestSetup, :(using AllocArrays); recursive=true)

makedocs(;
         modules=[AllocArrays],
         authors="Eric P. Hanson",
         repo=Remotes.GitHub("ericphanson", "AllocArrays.jl"),
         sitename="AllocArrays.jl",
         format=Documenter.HTML(;
                                prettyurls=get(ENV, "CI", "false") == "true",
                                edit_link="main",
                                assets=String[]),
         pages=["Home" => "index.md",
                "Allocator interface" => "interface.md"])

deploydocs(; repo="github.com/ericphanson/AllocArrays.jl.git",
           push_preview=true,
           devbranch="main")
