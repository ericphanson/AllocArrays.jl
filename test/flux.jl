using Flux, Random, StrideArraysCore

function bumper_run(model, data)
    buf = Bumper.default_buffer()
    with_bumper(buf) do
        @no_escape buf begin
            sum(model, data)
        end
    end
end

@testset "Basic model" begin
    model = Chain(Dense(1 => 23, tanh), Dense(23 => 1; bias=true), only)

    data = [[x] for x in -2:0.001f0:2]

    alloc_data = AllocArray.(data)

    @test sum(model, data) ≈ bumper_run(model, alloc_data) atol = 1e-3
    @test sum(model, data) ≈ sum(model, alloc_data) atol = 1e-3

    # Show some timing info
    @showtime sum(model, data)
    @showtime sum(model, alloc_data)
    @showtime bumper_run(model, alloc_data)
end

#####
##### More complicated model
#####

# from https://github.com/beacon-biosignals/LegolasFlux.jl/blob/27acd1bd20e5b335794fbf4c5cade46cc75c54c6/examples/digits.jl

# This should store all the information needed
# to construct the model.
Base.@kwdef struct DigitsConfigV1
    seed::Int = 5
    dropout_rate::Float32 = 0.0f1
end

# Here's our model object itself, just a `DigitsConfig` and
# a `chain`. We keep the config around so it's easy to save out
# later.
struct DigitsModel
    chain::Chain
    config::DigitsConfigV1
end

# Ensure Flux can recurse into our model to find params etc
Flux.@functor DigitsModel (chain,)

# Construct the actual model from a config object. This is the only
# constructor that should be used, to ensure the model is created just
# from the config object alone.
function DigitsModel(config::DigitsConfigV1=DigitsConfigV1())
    dropout_rate = config.dropout_rate
    Random.seed!(config.seed)
    D = Dense(10, 10)
    chain = Chain(Dropout(dropout_rate),
                  Conv((3, 3), 1 => 32, relu),
                  BatchNorm(32, relu),
                  MaxPool((2, 2)),
                  Dropout(dropout_rate),
                  Conv((3, 3), 32 => 16, relu),
                  Dropout(dropout_rate),
                  MaxPool((2, 2)),
                  Dropout(dropout_rate),
                  Conv((3, 3), 16 => 10, relu),
                  Dropout(dropout_rate),
                  x -> reshape(x, :, size(x, 4)),
                  Dropout(dropout_rate),
                  Dense(90, 10),
                  D,
                  D, # same layer twice, to test weight-sharing
                  softmax)
    return DigitsModel(chain, config)
end

# Our model acts on input just by applying the chain.
(m::DigitsModel)(x) = m.chain(x)

function infer!(predictions, model, data)
    buf = Bumper.default_buffer()
    with_bumper(buf) do
        for (idx, x) in enumerate(data)
            @no_escape buf begin
                predictions[idx] .= model(x)
            end
        end
    end
    return predictions
end

@testset "More complicated model" begin
    model = DigitsModel()

    # Setup some fake data
    N = 10_000
    data_arr = rand(Float32, 28, 28, N)

    # Partition into batches of size 32
    batch_size = 32
    data = [reshape(data_arr[:, :, I], 28, 28, 1, :)
            for I in Iterators.partition(1:N, batch_size)]

    n_class = 10
    fresh_predictions() = [Matrix{Float32}(undef, n_class, size(x, 4)) for x in data]

    alloc_data = AllocArray.(data)

    preds_data = fresh_predictions()
    infer!(preds_data, model, data)

    preds_alloc = fresh_predictions()
    infer!(preds_alloc, model, alloc_data)

    preds_stride = fresh_predictions()
    stride_data = StrideArray.(data)
    infer!(preds_stride, model, stride_data)

    @test preds_data ≈ preds_alloc
    @test preds_data ≈ preds_stride

    predictions = fresh_predictions()
    @showtime infer!(predictions, model, data)
    @showtime infer!(predictions, model, stride_data)
    @showtime infer!(predictions, model, alloc_data)
end
