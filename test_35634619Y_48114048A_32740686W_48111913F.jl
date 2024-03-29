using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

topology = [2, 3, 1]
inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
targets = [0, 1, 0, 1]
crossValidationIndices = [1, 1, 2, 2]
numExecutions = 50
transferFunctions = [σ, σ, σ]
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.01
validationRatio = 0
maxEpochsVal = 20

ANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
    transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
    validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)