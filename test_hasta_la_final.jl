using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Test case 1: Single output node, all correct predictions
@testset "Accuracy with single output node, all correct predictions" begin
    outputs = [true true; true true]
    targets = [true true; true true]
    @test accuracy(outputs, targets) == 1.0
end

# Test case 2: Single output node, all incorrect predictions
@testset "Accuracy with single output node, all incorrect predictions" begin
    outputs = [false false false; false false false]
    targets = [true true true; true true true]
    @test accuracy(outputs, targets) == 0.0
end

# Test case 3: Multiple output nodes, some correct and some incorrect predictions
@testset "Accuracy with multiple output nodes, some correct and some incorrect predictions" begin
    outputs = [true false; 
                true false]
    targets = [true true; 
                false false]
    @test accuracy(outputs, targets) == 0
end


# Test case 5: Different sizes of outputs and targets
@testset "Accuracy with different sizes of outputs and targets" begin
    outputs = [true, true, true]
    targets = [true, true, true, true, true]
    @test_throws AssertionError accuracy(outputs, targets)
end

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")


# Test case 1: N is divisible by k
N1 = 15
k1 = 3
partitions1 = crossvalidation(N1, k1)
println("Test Case 1:")
println("Number of partitions:", length(unique(partitions1)))
println("Lengths of partitions:", [sum(partitions1 .== i) for i in 1:k1])
println("Partitions:")
println(partitions1)
@assert length(unique(partitions1)) == k1
# Test case 2: N is not divisible by k
N2 = 14
k2 = 7
partitions2 = crossvalidation(N2, k2)
println("Test Case 2:")
println("Number of partitions:", length(unique(partitions2)))
println("Lengths of partitions:", [sum(partitions2 .== i) for i in 1:k2])
println("Partitions:")
println(partitions2)
@assert length(unique(partitions2)) == k2
# Test case 3: N is smaller than k
N3 = 3
k3 = 5
partitions3 = crossvalidation(N3, k3)
println("Test Case 3:")
println("Number of partitions:", length(unique(partitions3)))
println("Lengths of partitions:", [sum(partitions3 .== i) for i in 1:k3])
println("Partitions:")
println(partitions3)
@assert length(unique(partitions3)) == 3

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: Confusion matrix with all correct predictions
@testset "Confusion matrix with all correct predictions" begin
    outputs = [true, true, true]
    targets = [true, true, true]
    expected_matrix = [0 0; 
                        0 3]
    expected_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_sensitivity = 1.0
    expected_specificity = 0.0
    expected_positive_predictive_value = 1.0
    expected_negative_predictive_value = 0.0
    expected_f_score = 1.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
end

# Test case 2: Confusion matrix with all incorrect predictions
@testset "Confusion matrix with all incorrect predictions" begin
    outputs = [false, false, false]
    targets = [true, true, true]
    expected_matrix = [0 3; 0 0]
    expected_accuracy = 0.0
    expected_fail_rate = 1.0
    expected_sensitivity = 0.0
    expected_specificity = 0.0
    expected_positive_predictive_value = 0.0
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

end

# Test case 3: Confusion matrix with mixed predictions
@testset "Confusion matrix with mixed predictions" begin
    outputs = [true, false, true]
    targets = [true, true, false]
    expected_matrix = [1 1; 1 0]
    expected_accuracy = 0.3333333333333333
    expected_fail_rate = 0.6666666666666666
    expected_sensitivity = 0.0
    expected_specificity = 0.5
    expected_positive_predictive_value = 0.5
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)


end


using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: Confusion matrix with all correct predictions
@testset "Confusion matrix with all correct predictions" begin
    outputs = [true, true, true]
    targets = [true, true, true]
    expected_matrix = [0 0; 0 3]
    expected_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_sensitivity = 1.0
    expected_specificity = 0.0
    expected_positive_predictive_value = 1.0
    expected_negative_predictive_value = 0.0
    expected_f_score = 1.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score


end

# Test case 2: Confusion matrix with all incorrect predictions
@testset "Confusion matrix with all incorrect predictions" begin
    outputs = [false, false, false]
    targets = [true, true, true]
    expected_matrix = [0 0; 
                    3 0]
    expected_accuracy = 0.0
    expected_fail_rate = 1.0
    expected_sensitivity = 0.0
    expected_specificity = 0.0
    expected_positive_predictive_value = 0.0
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score


end

# Test case 3: Confusion matrix with mixed predictions
@testset "Confusion matrix with mixed predictions" begin
    outputs = [true, false, true]
    targets = [true, true, false]
    expected_matrix = [0 1; 
                        1 1]
    expected_accuracy = 0.3333333333333333
    expected_fail_rate = 0.6666666666666666
    expected_sensitivity = 0.5
    expected_specificity = 0
    expected_positive_predictive_value = 0.5
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.5

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score


end

@testset "Confusion matrix with Real" begin
    outputs = [0.6, 0.3, 0.7]
    targets = [true, true, false]
    expected_matrix = [0 1; 
                        1 1]
    expected_accuracy = 0.3333333333333333
    expected_fail_rate = 0.6666666666666666
    expected_sensitivity = 0.5
    expected_specificity = 0
    expected_positive_predictive_value = 0.5
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.5

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets, threshold=0.6)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score


end

@testset "Confusion matrix with Real" begin
    
    outputs = [true, true, true, true, true, false, false, true, true, false] # true -> 80% false -> 20%
    targets = [true, true, false, true, true, false, true, false, true, false] # true -> 60% false -> 40%
    expected_matrix = [2 2; 
                        1 5]
    expected_accuracy = 0.7
    expected_fail_rate = 0.3
    expected_sensitivity = 0.833333333333333
    expected_specificity = 0.5
    expected_positive_predictive_value = 0.7142857142857143
    expected_negative_predictive_value = 0.6666666666666666
    expected_f_score = 0.769230769230769

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets, threshold=0.6)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score

end

# Test case 4: Confusion matrix with binary outputs
@testset "Confusion matrix with binary outputs" begin
    outputs = [0.7, 0.8, 0.9, 0.6, 0.8, 0.3, 0.1, 0.9, 0.6, 0.2] # true -> 80% false -> 20%
    targets = [true, true, false, true, true, false, true, false, true, false] # true -> 60% false -> 40%
    expected_matrix = [2 2; 
                        1 5]
    expected_accuracy = 0.7
    expected_fail_rate = 0.3
    expected_sensitivity = 0.833333333333333
    expected_specificity = 0.5
    expected_positive_predictive_value = 0.7142857142857143
    expected_negative_predictive_value = 0.6666666666666666
    expected_f_score = 0.769230769230769

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets, threshold=0.6)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
end


using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case for trainClassANN function
@testset "trainClassANN function" begin
    # Define the input and target datasets
    inputs_train = [0.1 0.2 0.3; 0.4 0.5 0.6]
    targets_train = [true false true; false true false]
    inputs_val = [0.7 0.8 0.9; 1.0 1.1 1.2]
    targets_val = [true false true; false true false]
    inputs_test = [1.3 1.4 1.5; 1.6 1.7 1.8]
    targets_test = [true false true; false true false]

    # Call the trainClassANN function
    model, train_losses, val_losses, test_losses = trainClassANN([4, 2], (inputs_train, targets_train),
        validationDataset=(inputs_val, targets_val), testDataset=(inputs_test, targets_test), maxEpochs=0)

    # Perform the tests
    println("Model: ", model)
    println("Train losses: ", train_losses)
    println("Val losses: ", val_losses)
    println("Test losses: ", test_losses)
    model, train_losses, val_losses, test_losses = trainClassANN([4, 2], (inputs_train, targets_train),
    validationDataset=(inputs_val, targets_val), testDataset=(inputs_test, targets_test))

    # Perform the tests
    println("Model: ", model)
    println("Train losses: ", train_losses)
    println("Val losses: ", val_losses)
    println("Test losses: ", test_losses)
end
#=

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case for confusionMatrix function with single output node
# @testset "Confusion matrix with single output node" begin
    outputs = [true true false; true true false; true true false]
    targets = [true true false; true true false; true true false]
    expected_matrix = [3 0;
                         0 6]
    expected_accuracy = 1.0
    expected_fail_rate = 0
    expected_sensitivity = 1.0
    expected_specificity = 1.0
    expected_positive_predictive_value = 1.0
    expected_negative_predictive_value = 1.0
    expected_f_score = 1.0

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets, weighted=false)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
# end

# Test case for confusionMatrix function with multiple output nodes
@testset "Confusion matrix with multiple output nodes" begin
    outputs = [true false; true false]
    targets = [true true; false false]
    expected_matrix = [1 1; 1 0]
    expected_accuracy = 0.5
    expected_fail_rate = 0.5
    expected_sensitivity = 0.5
    expected_specificity = 0.0
    expected_positive_predictive_value = 0.5
    expected_negative_predictive_value = 0.0
    expected_f_score = 0.3333333333333333

    matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)

    @test matrix == expected_matrix
    @test matrix_accuracy ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivity ≈ expected_sensitivity
    @test specificity ≈ expected_specificity
    @test positive_predictive_value ≈ expected_positive_predictive_value
    @test negative_predictive_value ≈ expected_negative_predictive_value
    @test f_score ≈ expected_f_score
end

# Test case for confusionMatrix function with different sizes of outputs and targets
@testset "Confusion matrix with different sizes of outputs and targets" begin
    outputs = [true, true, true]
    targets = [true, true, true, true, true]
    @test_throws AssertionError confusionMatrix(outputs, targets)
end

# Test case for confusionMatrix function with weighted option
@testset "Confusion matrix with weighted option" begin
    outputs = [true, true, true]
    targets = [true, true, true]
    expected_matrix = [0 0; 0 3]
    expected_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_weighted_sensitivity = 1.0
    expected_weighted_specificity = 0.0
    expected_weighted_positive_predictive_value = 1.0
    expected_weighted_negative_predictive_value = 0.0
    expected_weighted_f_score = 1.0

    acc, fail_rate, weighted_sensitivity, weighted_specificity, weighted_positive_predictive_value, weighted_negative_predictive_value, weighted_f_score, matrix = confusionMatrix(outputs, targets, weighted=true)

    @test matrix == expected_matrix
    @test acc ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test weighted_sensitivity ≈ expected_weighted_sensitivity
    @test weighted_specificity ≈ expected_weighted_specificity
    @test weighted_positive_predictive_value ≈ expected_weighted_positive_predictive_value
    @test weighted_negative_predictive_value ≈ expected_weighted_negative_predictive_value
    @test weighted_f_score ≈ expected_weighted_f_score
end

# Test case for confusionMatrix function without weighted option
@testset "Confusion matrix without weighted option" begin
    outputs = [true, true, true]
    targets = [true, true, true]
    expected_matrix = [0 0; 0 3]
    expected_accuracy = 1.0
    expected_fail_rate = 0.0
    expected_macro_sensitivity = 1.0
    expected_macro_specificity = 0.0
    expected_macro_positive_predictive_value = 1.0
    expected_macro_negative_predictive_value = 0.0
    expected_macro_f_score = 1.0

    acc, fail_rate, macro_sensitivity, macro_specificity, macro_positive_predictive_value, macro_negative_predictive_value, macro_f_score, matrix = confusionMatrix(outputs, targets, weighted=false)

    @test matrix == expected_matrix
    @test acc ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test macro_sensitivity ≈ expected_macro_sensitivity
    @test macro_specificity ≈ expected_macro_specificity
    @test macro_positive_predictive_value ≈ expected_macro_positive_predictive_value
    @test macro_negative_predictive_value ≈ expected_macro_negative_predictive_value
    @test macro_f_score ≈ expected_macro_f_score
end

include("35634619Y_48114048A_32740686W_48111913F.jl")
outputs = [true true true;
            false false false;
            true true true]

targets = [true true true;
            false false false;
            true true true]

sensitivity = zeros(size(outputs, 2))
specificity = zeros(size(outputs, 2))
positive_predictive_value = zeros(size(outputs, 2))
negative_predictive_value = zeros(size(outputs, 2))
f_score = zeros(size(outputs, 2))

for i = 1:(size(outputs, 2))
    #matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix
    _, _, sensitivity[i], specificity[i], positive_predictive_value[i], negative_predictive_value[i], f_score[i], _ = confusionMatrix(outputs[:,i], targets[:,i])
end       
# esto esta mal
# Define the unique classes

# Create the confusion matrix
matrix = zeros(Int, size(outputs, 2), size(outputs, 2))
for real in 1:size(outputs, 1)
    for predicted in 1:size(outputs, 2)
        if outputs[real, predicted] == true && targets[real, predicted] == true
            matrix[real, predicted] += 1
        end
    end
end
=#