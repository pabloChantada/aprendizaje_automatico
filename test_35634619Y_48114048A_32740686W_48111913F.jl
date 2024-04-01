using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: Training with empty datasets
function test_trainClassANN_empty_datasets()
    topology = [2, 3, 1]
    trainingDataset = (Array{Float64}(undef, 0, 0), falses(0, 0))
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset
    )
    @test length(train_losses) == 1
    @test length(val_losses) == 1
    @test length(test_losses) == 1
    @test best_model isa Flux.Chain
end

# Test case 2: Training with non-empty datasets
function test_trainClassANN_non_empty_datasets()
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    testDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    transferFunctions = [σ, σ, σ]
    maxEpochs = 1000
    minLoss = 0.0
    learningRate = 0.01
    maxEpochsVal = 20

    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal
    )

    @test length(train_losses) > 1
    @test length(val_losses) > 1
    @test length(test_losses) > 1
    @test best_model isa Flux.Chain
end

# Test case 3: Training with 2 classes
function test_trainClassANN_2_classes()
    topology = [2, 3, 2]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    testDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    transferFunctions = [σ, σ, σ]
    maxEpochs = 1000
    minLoss = 0.0
    learningRate = 0.01
    maxEpochsVal = 20

    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal
    )

    @test length(train_losses) > 1
    @test length(val_losses) > 1
    @test length(test_losses) > 1
    @test best_model isa Flux.Chain
end

# Test case 4: DimensionMismatch handling in trainClassANN
function test_trainClassANN_dimension_mismatch()
    # Create simulated input data that mimics the problematic case
    inputs_train = Array{Float32}(undef, 0, 0) # Empty matrix to simulate the error
    targets_train = falses(0, 0) # Empty label vector
    trainingDataset = (inputs_train, targets_train)
    
    # Define a simple topology for the network
    topology = [3, 8, 2] # Assume a network with layers expecting 3 inputs and ending with 2 outputs
    transferFunctions = [tanh, tanh, σ] # Activation functions for each layer
    
    # Try to train the network with the empty dataset
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        transferFunctions=transferFunctions
    )
    
    # Verify that the lengths of the loss vectors are greater than 0
    @test length(train_losses) > 0
    @test length(val_losses) > 0
    @test length(test_losses) > 0
    @test best_model isa Flux.Chain
    # Add specific checks for the expected error handling
end

# Test case: Error handling for DimensionMismatch in trainClassANN
function test_trainClassANN_dimension_mismatch()
    # Create simulated input data that mimics the problematic case
    inputs_train = Array{Float32}(undef, 0, 0) # Empty matrix to simulate the error
    targets_train = falses(0, 0) # Empty label vector
    trainingDataset = (inputs_train, targets_train)
    
    # Define a simple topology for the network
    topology = [3, 8, 2] # Assume a network with layers expecting 3 inputs and ending with 2 outputs
    transferFunctions = [tanh, tanh, σ] # Activation functions for each layer
    
    # Try to train the network with the empty dataset
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        transferFunctions=transferFunctions
    )
    
    # Verify that the lengths of the loss vectors are greater than 0
    @test length(train_losses) > 0
    @test length(val_losses) > 0
    @test length(test_losses) > 0
    @test best_model isa Flux.Chain
    # Add specific checks for the expected error handling
end



# Test case 5: Training with empty validation and test datasets
function test_trainClassANN_empty_validation_test_datasets()
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (Array{Float64}(undef, 0, 0), falses(0, 0))
    testDataset = (Array{Float64}(undef, 0, 0), falses(0, 0))
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset
    )
    @test length(train_losses) >= 1
    @test length(val_losses) == 1
    @test length(test_losses) == 1
    @test best_model isa Flux.Chain
end

# Test case 6: Training with custom transfer functions
function test_trainClassANN_custom_transfer_functions()
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    testDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    transferFunctions = [x -> x^2, x -> x^3, σ]
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions
    )
    @test length(train_losses) >= 1
    @test length(val_losses) >= 1
    @test length(test_losses) >= 1
    @test best_model isa Flux.Chain
end

# Test case 7: Training with maximum number of epochs reached
function test_trainClassANN_max_epochs_reached()
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    testDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    maxEpochs = 10
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        maxEpochs=maxEpochs
    )
    @test length(train_losses) == maxEpochs
    @test length(val_losses) == maxEpochs
    @test length(test_losses) == maxEpochs
    @test best_model isa Flux.Chain
end

# Test case 8: Training with minimum loss reached
function test_trainClassANN_min_loss_reached()
    topology = [2, 3, 1]
    trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    validationDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    testDataset = (rand(2, 100), rand(2, 100) .> 0.5)
    minLoss = 0.01
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        minLoss=minLoss
    )
    @test length(train_losses) >= 1
    @test length(val_losses) >= 1
    @test length(test_losses) >= 1
    @test best_model isa Flux.Chain
end

# Run the tests
@testset "trainClassANN Tests" begin
    @testset "Empty Datasets" begin
        test_trainClassANN_empty_datasets()
    end

    @testset "Non-Empty Datasets" begin
        test_trainClassANN_non_empty_datasets()
    end
    
    @testset "2 Classes" begin
        test_trainClassANN_2_classes()
    end
    
    @testset "DimensionMismatch Handling" begin
        test_trainClassANN_dimension_mismatch()
    end
    @testset "Empty Validation and Test Datasets" begin
        test_trainClassANN_empty_validation_test_datasets()
    end

    @testset "Custom Transfer Functions" begin
        test_trainClassANN_custom_transfer_functions()
    end
    
    @testset "Max Epochs Reached" begin
        test_trainClassANN_max_epochs_reached()
    end
    
    @testset "Min Loss Reached" begin
        test_trainClassANN_min_loss_reached()
    end
end

using Test
using Flux

# Asumiendo que buildClassANN ya está definido
# y que se tiene un conjunto de datos de ejemplo (inputs y targets)

# Test 1: Verificar tipos de retorno
@testset "trainClassANN Return Types" begin
    topology = [100, 10, 2] # Ejemplo de topología
    trainingDataset = (rand(Float32, 100, 50), rand(Bool, 2, 50)) # Datos de entrenamiento de ejemplo
    # Conjuntos vacíos para validación y test (no se usarán en este test)
    validationDataset = (Array{Float32,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0))
    testDataset = (Array{Float32,2}(undef, 0, 0), Array{Bool,2}(undef, 0, 0))

    model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        maxEpochs=10,
        learningRate=0.01,
        maxEpochsVal=5
    )
    
    @test model isa Flux.Chain
    @test train_losses isa Vector{Float64}
    @test val_losses isa Vector{Float64}
    @test test_losses isa Vector{Float64}
end

# Test 2: Verificar comportamiento de la longitud de los vectores de pérdida
@testset "Loss Vectors Length" begin
    topology = [100, 10, 2] # Ejemplo de topología
    trainingDataset = (rand(Float32, 100, 100), rand(Bool, 2, 100)) # Datos de entrenamiento de ejemplo
    validationDataset = (rand(Float32, 100, 20), rand(Bool, 2, 20)) # Datos de validación de ejemplo
    testDataset = (rand(Float32, 100, 20), rand(Bool, 2, 20)) # Datos de test de ejemplo
    
    model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        maxEpochs=10,
        learningRate=0.01,
        maxEpochsVal=5
    )
    
    @test length(train_losses) == 10 # Incluye el estado inicial antes del entrenamiento
    @test length(val_losses) == 10
    @test length(test_losses) == 10
end

# Test 3: Verificar mejora del modelo
@testset "Model Improvement" begin
    topology = [100, 10, 2]
    trainingDataset = (rand(Float32, 100, 100), rand(Bool, 2, 100))
    validationDataset = (rand(Float32, 100, 20), rand(Bool, 2, 20))
    testDataset = (rand(Float32, 100, 20), rand(Bool, 2, 20))
    
    model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        maxEpochs=100,
        learningRate=0.01,
        maxEpochsVal=20)
    # Asegurarse de que el modelo mejora con el tiempo (la pérdida disminuye)
    @test train_losses[end] < train_losses[1]
end
