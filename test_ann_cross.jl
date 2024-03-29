using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

topology = [2, 3, 1]
inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
targets = [0, 1, 0, 1]
crossValidationIndices = [1, 1, 2, 2]
numExecutions = 10
transferFunctions = [σ, σ, σ]
maxEpochs = 100
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
    crossValidationIndices = [1, 1, 2, 2]
    numExecutions = 10
    transferFunctions = [σ, σ, σ]
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