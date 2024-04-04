using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

topology = [2, 3, 1]
inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
targets = [0, 1, 0, 1]
crossValidationIndices = [1, 2]
numExecutions = 10
transferFunctions = [σ, σ, σ]
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.01
validationRatio = 0
maxEpochsVal = 20

ANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
    validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)
    
using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
@testset "ANNCrossValidation Tests" begin
    topology = [2, 3, 1]
    inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1,2]
    numExecutions = 10
    maxEpochs = 100
    minLoss = 0.0
    learningRate = 0.01
    validationRatio = 0
    maxEpochsVal = 20

    # Llama a ANNCrossValidation con los parámetros definidos
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices,
        numExecutions=numExecutions,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        validationRatio=validationRatio,
        maxEpochsVal=maxEpochsVal)

    # Verifica que el resultado sea una tupla de 7 elementos
    @test length(result) == 7

    # Para cada elemento de la tupla resultado, verifica que contenga una tupla de dos elementos (media y desviación estándar)
    for metric_result in result
        @test length(metric_result) == 2
        @test metric_result[1] isa Float64
        @test metric_result[2] isa Float64
    end
end

using Test;
include("35634619Y_48114048A_32740686W_48111913F.jl")
inputs = rand(100, 10)

# Mock targets: Integers representing class labels for 100 samples
targets = rand(1:3, 100)

# Mock cross-validation indices: Integers indicating which fold each sample belongs to
# For simplicity, let's assume 5-fold cross-validation
crossValidationIndices = repeat(1:5, inner = 20)
# Define ANN topology: Example [10, 5, 3] means input layer of 10 neurons, 
# one hidden layer of 5 neurons, and an output layer of 3 neurons
topology = [10, 5, 3]

# Call the function with the mock data and topology
results = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

@testset "ANNCrossValidation Functionality Tests" begin
    @testset "Basic Functionality Test" begin
        topology = [2, 3, 1]
        inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
        targets = [0, 1, 0, 1]
        crossValidationIndices = [1, 2]
        numExecutions = 10
        maxEpochs = 100
        minLoss = 0.0
        learningRate = 0.01
        validationRatio = 0
        maxEpochsVal = 20

        result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices,
            numExecutions=numExecutions,
            maxEpochs=maxEpochs,
            minLoss=minLoss,
            learningRate=learningRate,
            validationRatio=validationRatio,
            maxEpochsVal=maxEpochsVal)

        @test length(result) == 7
        for metric_result in result
            @test length(metric_result) == 2
            @test metric_result[1] isa Number
            @test metric_result[2] isa Number
        end
    end

    @testset "Edge Case: Small Dataset" begin
        topology = [2, 2, 1]
        inputs = [0.1 0.2; 0.3 0.4]
        targets = [0, 1]
        crossValidationIndices = [1, 2]
        # Remaining parameters as before...

        result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices,
            numExecutions=5,  # Reduced for brevity
            maxEpochs=100, minLoss=0.0, learningRate=0.01,
            validationRatio=0, maxEpochsVal=20)

        @test length(result) == 7
        for metric_result in result
            @test length(metric_result) == 2
            @test metric_result[1] isa Number
            @test metric_result[2] isa Number
        end
    end

    # Add more @testset blocks here for different scenarios
end
