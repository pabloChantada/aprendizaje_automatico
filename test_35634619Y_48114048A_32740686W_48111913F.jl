using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: ANNCrossValidation with validationRatio > 0
function test_ANNCrossValidation_case1()
    topology = [2, 1]
    inputs = [1 2 3 4; 5 6 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 2, 1, 2]
    numExecutions = 50
    transferFunctions = [σ, σ]
    maxEpochs = 1000
    minLoss = 0.0
    learningRate = 0.01
    validationRatio = 0.2
    maxEpochsVal = 20

    expected_precision = (0.5, 0.0)
    expected_errorRate = (0.5, 0.0)
    expected_sensitivity = (0.5, 0.0)
    expected_specificity = (0.5, 0.0)
    expected_VPP = (0.5, 0.0)
    expected_VPN = (0.5, 0.0)
    expected_F1 = (0.5, 0.0)

    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices;
        numExecutions=numExecutions, transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

    @test result[1] ≈ expected_precision
    @test result[2] ≈ expected_errorRate
    @test result[3] ≈ expected_sensitivity
    @test result[4] ≈ expected_specificity
    @test result[5] ≈ expected_VPP
    @test result[6] ≈ expected_VPN
    @test result[7] ≈ expected_F1
end

# Test case 2: ANNCrossValidation with validationRatio = 0
function test_ANNCrossValidation_case2()
    topology = [2, 1]
    inputs = [1 2 3 4; 5 6 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 2, 1, 2]
    numExecutions = 50
    transferFunctions = [σ, σ]
    maxEpochs = 1000
    minLoss = 0.0
    learningRate = 0.01
    validationRatio = 0.0
    maxEpochsVal = 20

    expected_precision = (0.5, 0.0)
    expected_errorRate = (0.5, 0.0)
    expected_sensitivity = (0.5, 0.0)
    expected_specificity = (0.5, 0.0)
    expected_VPP = (0.5, 0.0)
    expected_VPN = (0.5, 0.0)
    expected_F1 = (0.5, 0.0)

    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices;
        numExecutions=numExecutions, transferFunctions=transferFunctions,
        maxEpochs=maxEpochs, minLoss=minLoss, learningRate=learningRate,
        validationRatio=validationRatio, maxEpochsVal=maxEpochsVal)

    @test result[1] ≈ expected_precision
    @test result[2] ≈ expected_errorRate
    @test result[3] ≈ expected_sensitivity
    @test result[4] ≈ expected_specificity
    @test result[5] ≈ expected_VPP
    @test result[6] ≈ expected_VPN
    @test result[7] ≈ expected_F1
end

# Run the tests
@testset "ANNCrossValidation Tests" begin
    test_ANNCrossValidation_case1()
    test_ANNCrossValidation_case2()
end