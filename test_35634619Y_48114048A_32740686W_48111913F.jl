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

# Test case 1: Basic test case with 2-fold cross validation
function test_ANNCrossValidation_1()
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

    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices, numExecutions=numExecutions,
        transferFunctions=transferFunctions, maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

    # Assert the mean accuracy is within an acceptable range
    @test 0.9 <= result[1][1] <= 1.0
    # Assert the standard deviation of accuracy is within an acceptable range
    @test 0.0 <= result[1][2] <= 0.1
    # Assert the mean fail rate is within an acceptable range
    @test 0.0 <= result[2][1] <= 0.1
    # Assert the standard deviation of fail rate is within an acceptable range
    @test 0.0 <= result[2][2] <= 0.1
    # Assert the mean sensitivity is within an acceptable range
    @test 0.9 <= result[3][1] <= 1.0
    # Assert the standard deviation of sensitivity is within an acceptable range
    @test 0.0 <= result[3][2] <= 0.1
    # Assert the mean specificity is within an acceptable range
    @test 0.9 <= result[4][1] <= 1.0
    # Assert the standard deviation of specificity is within an acceptable range
    @test 0.0 <= result[4][2] <= 0.1
    # Assert the mean VPP is within an acceptable range
    @test 0.9 <= result[5][1] <= 1.0
    # Assert the standard deviation of VPP is within an acceptable range
    @test 0.0 <= result[5][2] <= 0.1
    # Assert the mean VPN is within an acceptable range
    @test 0.9 <= result[6][1] <= 1.0
    # Assert the standard deviation of VPN is within an acceptable range
    @test 0.0 <= result[6][2] <= 0.1
    # Assert the mean F1 score is within an acceptable range
    @test 0.9 <= result[7][1] <= 1.0
    # Assert the standard deviation of F1 score is within an acceptable range
    @test 0.0 <= result[7][2] <= 0.1
end

# Run the test cases
@testset "ANNCrossValidation Tests" begin
    @testset "Test Case 1" begin
        test_ANNCrossValidation_1()
    end
end