using Test

include("main.jl")

function test_trainClassANN()
    # Define test data
    topology = [2, 3, 2]
    inputs_train = [0.1 0.2 0.3; 0.4 0.5 0.6]
    targets_train = [true false true ; true false true]
    inputs_val = [0.7 0.8 0.9; 1.0 1.1 1.2]
    targets_val = [false true false ; true false true]
    inputs_test = [1.3 1.4 1.5; 1.6 1.7 1.8]
    targets_test = [true false true ; true false true]

    # Call the function under test
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology, (inputs_train, targets_train),
        validationDataset=(inputs_val, targets_val),
        testDataset=(inputs_test, targets_test)
    )

    # Perform assertions
    @test length(train_losses) > 0
    @test length(val_losses) > 0
    @test length(test_losses) > 0
    @test length(train_losses) == length(val_losses)
    @test length(train_losses) == length(test_losses)
    # @test typeof(best_model) == Flux.Chain
end

# Run the tests
@testset "Main Tests" begin
    @testset "trainClassANN" begin
        test_trainClassANN()
    end
end