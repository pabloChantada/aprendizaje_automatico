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