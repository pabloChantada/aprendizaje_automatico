using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: Testing ANN model
function test_modelCrossValidation_ann()
    modelType = :ANN
    modelHyperparameters = Dict(:topology => [10, 5, 2],
                                :numExecutions => 5,
                                :transferFunctions => ["sigmoid", "sigmoid", "sigmoid"],
                                :maxEpochs => 100,
                                :minLoss => 0.01,
                                :learningRate => 0.1,
                                :validationRatio => 0.2,
                                :maxEpochsVal => 50)
    inputs = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30]
    targets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    crossValidationIndices = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    result = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    @test length(result) == 5
    @test all(0 <= value <= 1 for value in result)
end

# Test case 2: Testing SVM model
function test_modelCrossValidation_svm()
    modelType = :SVM
    modelHyperparameters = Dict(:C => 1.0,
                                :kernel => "rbf",
                                :degree => 3,
                                :gamma => 2.0,
                                :coef0 => 0.0)
    inputs = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30]
    targets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    crossValidationIndices = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    result = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    @test length(result) == 10
    @test all(0 <= value <= 1 for value in result)
end

# Test case 3: Testing Decision Tree Classifier model
function test_modelCrossValidation_decision_tree()
    modelType = :DecisionTreeClassifier
    modelHyperparameters = Dict(:max_depth => 4)
    inputs = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30]
    targets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    crossValidationIndices = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    result = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    @test length(result) == 10
    @test all(0 <= value <= 1 for value in result)
end

# Test case 4: Testing K Neighbors Classifier model
function test_modelCrossValidation_k_neighbors()
    modelType = :KNeighborsClassifier
    modelHyperparameters = Dict(:n_neighbors => 3)
    inputs = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30]
    targets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    crossValidationIndices = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]

    result = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    @test length(result) == 10
    @test all(0 <= value <= 1 for value in result)
end

# Run the tests
@testset "modelCrossValidation tests" begin
    test_modelCrossValidation_ann()
    test_modelCrossValidation_svm()
    test_modelCrossValidation_decision_tree()
    test_modelCrossValidation_k_neighbors()
end