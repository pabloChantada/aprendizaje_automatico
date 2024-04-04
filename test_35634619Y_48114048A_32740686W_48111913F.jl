using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test ANNCrossValidation function
@testset "ANNCrossValidation function" begin
    # Define test inputs
    topology = [2, 3, 1]
    inputs = [1 2; 3 4; 5 6; 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 1, 2, 2]

    # Define expected outputs
    expected_mean_acc = 0.5
    expected_std_acc = 0.0
    expected_mean_fail_rate = 0.5
    expected_std_fail_rate = 0.0
    expected_mean_sensitivity = 0.0
    expected_std_sensitivity = 0.0
    expected_mean_specificity = 1.0
    expected_std_specificity = 0.0
    expected_mean_vpp = 0.0
    expected_std_vpp = 0.0
    expected_mean_vpn = 1.0
    expected_std_vpn = 0.0
    expected_mean_f1 = 0.0
    expected_std_f1 = 0.0

    # Call the ANNCrossValidation function
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Check the results
    @test result == ((expected_mean_acc, expected_std_acc), (expected_mean_fail_rate, expected_std_fail_rate),
                     (expected_mean_sensitivity, expected_std_sensitivity), (expected_mean_specificity, expected_std_specificity),
                     (expected_mean_vpp, expected_std_vpp), (expected_mean_vpn, expected_std_vpn), (expected_mean_f1, expected_std_f1))
end

using Test

# Test ANNCrossValidation function
@testset "ANNCrossValidation function" begin
    # Define test inputs
    topology = [2, 3, 1]
    inputs = [1 2; 3 4; 5 6; 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 1, 2, 2]

    # Define expected outputs
    expected_mean_acc = 0.0
    expected_std_acc = 0.0
    expected_mean_fail_rate = 0.0
    expected_std_fail_rate = 0.0
    expected_mean_sensitivity = 0.0
    expected_std_sensitivity = 0.0
    expected_mean_specificity = 0.0
    expected_std_specificity = 0.0
    expected_mean_vpp = 0.0
    expected_std_vpp = 0.0
    expected_mean_vpn = 0.0
    expected_std_vpn = 0.0
    expected_mean_f1 = 0.0
    expected_std_f1 = 0.0

    # Call the ANNCrossValidation function
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Check the results
    @test length(result) == 7

    topology = [2, 3, 1]
    inputs = [1 2; 3 4; 5 6; 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 1, 2, 2]
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)
    @test length(result) == 7
end# Test ANNCrossValidation function

@testset "ANNCrossValidation function" begin
    # Define test inputs
    topology = [2, 4, 1]
    inputs = [2 4; 6 8; 10 12; 14 16]
    targets = [1, 0, 1, 0]
    crossValidationIndices = [1, 2, 2, 1]
    
    # Define expected outputs (Estos valores son hipotéticos y para fines de prueba)
    expected_mean_acc = 0.75
    expected_std_acc = 0.25
    expected_mean_fail_rate = 0.25
    expected_std_fail_rate = 0.25
    expected_mean_sensitivity = 0.75
    expected_std_sensitivity = 0.25
    expected_mean_specificity = 0.75
    expected_std_specificity = 0.25
    expected_mean_vpp = 0.7
    expected_std_vpp = 0.3
    expected_mean_vpn = 0.7
    expected_std_vpn = 0.3
    expected_mean_f1 = 0.72
    expected_std_f1 = 0.28
    

    # Call the ANNCrossValidation function
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Check the results
    @test result == ((expected_mean_acc, expected_std_acc), (expected_mean_fail_rate, expected_std_fail_rate),
                     (expected_mean_sensitivity, expected_std_sensitivity), (expected_mean_specificity, expected_std_specificity),
                     (expected_mean_vpp, expected_std_vpp), (expected_mean_vpn, expected_std_vpn), (expected_mean_f1, expected_std_f1))
end
# Test ANNCrossValidation function
@testset "ANNCrossValidation function" begin
    # Define test inputs
    topology = [2, 3, 1]
    inputs = [1 2; 3 4; 5 6; 7 8]
    targets = [0, 1, 0, 1]
    crossValidationIndices = [1, 1, 2, 2]

    # Define expected outputs
    expected_mean_acc = 0.5
    expected_std_acc = 0.0
    expected_mean_fail_rate = 0.5
    expected_std_fail_rate = 0.0
    expected_mean_sensitivity = 0.0
    expected_std_sensitivity = 0.0
    expected_mean_specificity = 1.0
    expected_std_specificity = 0.0
    expected_mean_vpp = 0.0
    expected_std_vpp = 0.0
    expected_mean_vpn = 1.0
    expected_std_vpn = 0.0
    expected_mean_f1 = 0.0
    expected_std_f1 = 0.0

    # Call the ANNCrossValidation function
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Check the results
    @test result == ((expected_mean_acc, expected_std_acc), (expected_mean_fail_rate, expected_std_fail_rate),
                     (expected_mean_sensitivity, expected_std_sensitivity), (expected_mean_specificity, expected_std_specificity),
                     (expected_mean_vpp, expected_std_vpp), (expected_mean_vpn, expected_std_vpn), (expected_mean_f1, expected_std_f1))
end

# Test ANNCrossValidation function with different inputs
@testset "ANNCrossValidation function with different inputs" begin
    # Define test inputs
    topology = [2, 4, 1]
    inputs = [2 4; 6 8; 10 12; 14 16]
    targets = [1, 0, 1, 0]
    crossValidationIndices = [1, 2, 2, 1]
    
    # Define expected outputs
    expected_mean_acc = 0.75
    expected_std_acc = 0.25
    expected_mean_fail_rate = 0.25
    expected_std_fail_rate = 0.25
    expected_mean_sensitivity = 0.75
    expected_std_sensitivity = 0.25
    expected_mean_specificity = 0.75
    expected_std_specificity = 0.25
    expected_mean_vpp = 0.7
    expected_std_vpp = 0.3
    expected_mean_vpn = 0.7
    expected_std_vpn = 0.3
    expected_mean_f1 = 0.72
    expected_std_f1 = 0.28

    # Call the ANNCrossValidation function
    result = ANNCrossValidation(topology, inputs, targets, crossValidationIndices)

    # Check the results
    @test result == ((expected_mean_acc, expected_std_acc), (expected_mean_fail_rate, expected_std_fail_rate),
                     (expected_mean_sensitivity, expected_std_sensitivity), (expected_mean_specificity, expected_std_specificity),
                     (expected_mean_vpp, expected_std_vpp), (expected_mean_vpn, expected_std_vpn), (expected_mean_f1, expected_std_f1))
end# Test modelCrossValidation function with ANN model




# Simulación de datos de entrada y objetivos
inputs = rand(100, 10) # 100 filas de datos, 10 características
targets = [rand(["Clase1", "Clase2"]) for _ in 1:100] # 100 objetivos aleatorios
crossValidationIndices = repeat(1:5, 20) # 5-folds

# Hiperparámetros para un modelo SVC como ejemplo
modelHyperparameters = Dict(
    "C" => 1.0,
    "kernel" => "linear",
    "degree" => 3,
    "gamma" => "scale",
    "coef0" => 0.0
)

# Llamada de prueba a la función modelCrossValidation para el modelo SVC
results = modelCrossValidation(:SVC, modelHyperparameters, inputs, targets, crossValidationIndices)

println("Resultados de la prueba: ", results)

# Asume que la generación de datos y la configuración de índices ya se ha realizado
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Hiperparámetros para ANN
modelHyperparameters = Dict("topology" => [10, 5, 2]) # Ejemplo: 10 neuronas en la capa de entrada, 5 en la oculta, 2 en la salida

# Llamada a la función modelCrossValidation para ANN
results = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices)

println("Resultados de la prueba con ANN: ", results)

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Test modelCrossValidation function with ANN model
@testset "modelCrossValidation function with ANN model" begin
    # Define test inputs
    modelType = :ANN
    modelHyperparameters = Dict("topology" => [10, 5, 2])
    inputs = rand(100, 10)
    targets = [rand(["Clase1", "Clase2"]) for _ in 1:100]
    crossValidationIndices = repeat(1:5, 20)

    # Call the modelCrossValidation function
    results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    # Check the results
    @test length(results) == 7
end

# Test modelCrossValidation function with SVC model
@testset "modelCrossValidation function with SVC model" begin
    # Define test inputs
    modelType = :SVC
    modelHyperparameters = Dict(
        "C" => 1.0,
        "kernel" => "linear",
        "degree" => 3,
        "gamma" => "scale",
        "coef0" => 0.0
    )
    inputs = rand(100, 10)
    targets = [rand(["Clase1", "Clase2"]) for _ in 1:100]
    crossValidationIndices = repeat(1:5, 20)

    # Call the modelCrossValidation function
    results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    # Check the results
    @test length(results) == 7
end

# Test modelCrossValidation function with DecisionTreeClassifier model
@testset "modelCrossValidation function with DecisionTreeClassifier model" begin
    # Define test inputs
    modelType = :DecisionTreeClassifier
    modelHyperparameters = Dict("max_depth" => 5)
    inputs = rand(100, 10)
    targets = [rand(["Clase1", "Clase2"]) for _ in 1:100]
    crossValidationIndices = repeat(1:5, 20)

    # Call the modelCrossValidation function
    results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    # Check the results
    @test length(results) == 7
end

# Test modelCrossValidation function with KNeighborsClassifier model
@testset "modelCrossValidation function with KNeighborsClassifier model" begin
    # Define test inputs
    modelType = :KNeighborsClassifier
    modelHyperparameters = Dict("n_neighbors" => 3)
    inputs = rand(100, 10)
    targets = [rand(["Clase1", "Clase2"]) for _ in 1:100]
    crossValidationIndices = repeat(1:5, 20)

    # Call the modelCrossValidation function
    results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

    # Check the results
    @test length(results) == 7
end