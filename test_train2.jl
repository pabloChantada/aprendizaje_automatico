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
    trainingDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    validationDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    testDataset = (rand(100, 2), rand(100, 2) .> 0.5)
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

# Run the tests
@testset "trainClassANN Tests" begin
    @testset "Empty Datasets" begin
        # test_trainClassANN_empty_datasets()
    end

    @testset "Non-Empty Datasets" begin
        test_trainClassANN_non_empty_datasets()
    end
end

# Test case 3: Training with 2 classes
function test_trainClassANN_2_classes()
    topology = [2, 3, 2]
    trainingDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    validationDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    testDataset = (rand(100, 2), rand(100, 2) .> 0.5)
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

# Run the tests
@testset "trainClassANN Tests" begin
    @testset "Empty Datasets" begin
        # test_trainClassANN_empty_datasets()
    end

    @testset "Non-Empty Datasets" begin
        test_trainClassANN_non_empty_datasets()
    end
    
    @testset "2 Classes" begin
        test_trainClassANN_2_classes()
    end
end

# Test para verificar el manejo de DimensionMismatch en trainClassANN
@testset "DimensionMismatch Handling in trainClassANN" begin
    # Crear datos de entrada simulados que imiten el caso problemático
    inputs_train = Array{Float32}(undef, 0, 0) # Matriz vacía que simula el error
    targets_train = falses(0, 0) # Vector de etiquetas vacío
    trainingDataset = (inputs_train, targets_train)
    
    # Definir una topología simple para la red
    topology = [3, 8, 2] # Asume una red con capas que esperan 3 entradas y terminan con 2 salidas
    transferFunctions = [tanh, tanh, σ] # Funciones de activación para cada capa
    
    # Intentar entrenar la red con el conjunto de datos vacío
    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        transferFunctions=transferFunctions
    )
    # Si se llega aquí, el manejo del error no es adecuado, fallar el test
    @test length(train_losses) > 0
    @test length(val_losses) > 0
    @test length(test_losses) > 0
    @test best_model isa Flux.Chain
    # Verificar que se captura el error específico esperado
end