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

# Test case 5: Confusion matrix with multiple classes
@testset "Confusion matrix with multiple classes" begin
    outputs = [0.7 0.2 0.1; 0.3 0.8 0.9; 0.6 0.4 0.5] # true -> 80% false -> 20%
    targets = [true false false; false true true; true false true] # true -> 60% false -> 40%
    expected_matrix = [1 1 1; 
                        1 1 1; 
                        1 1 1]
    expected_accuracy = 0.3333333333333333
    expected_fail_rate = 0.6666666666666666
    expected_sensitivity = 0.5
    expected_specificity = 0.0
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

# Test case 6: Confusion matrix with empty outputs
@testset "Confusion matrix with empty outputs" begin
    outputs = []
    targets = [true, false, true]
    expected_matrix = [0 0; 
                        0 0]
    expected_accuracy = 0.0
    expected_fail_rate = 0.0
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