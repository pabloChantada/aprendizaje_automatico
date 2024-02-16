#=
TESTEAR PARTES:
    6 - modificar
    8
    9
=#

include("main.jl")

dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
inputs = convert(Array{Float32,2},inputs);
targets = dataset[:,5];

# PARTE 1
# --------------------------------------------------------------------------
#= uncoment to use
# Test 1: Dos clases
feature1 = [1, 2, 1, 2, 1, 2, 2, 1, 2]
classes1 = [1, 2]
result1 = oneHotEncoding(feature1, classes1)
println("Resultado del Test 1:")
println(result1)

# Test 2: Tres clases
feature2 = [1, 2, 3, 1, 2, 3, 3, 2, 1]
classes2 = [1, 2, 3]
result2 = oneHotEncoding(feature2, classes2)
println("\nResultado del Test 2:")
println(result2)

# Test 3: sobrecarga 1
feature3 = [1, 2, 3, 1, 2, 3, 3, 2, 1]
result3 = oneHotEncoding(feature3)
@assert result2 == result3
println("\nResultado del Test 3:")
println(result3)

# Test 4: sobrecarga 2
function test_oneHotEncoding_bool_vector()
    # Datos de prueba
    feature = [true, false, true, true, false]

    # Llamada a la función oneHotEncoding
    result = oneHotEncoding(feature)

    # Verificar si la matriz resultante tiene el tamaño correcto
    @assert size(result) == (length(feature), 1)

    # Verificar si la matriz resultante es igual al vector original convertido en una columna
    @assert result == reshape(feature, :, 1)

    println("Test para oneHotEncoding con un vector booleano pasó correctamente.")
end

# Ejecutar el test
test_oneHotEncoding_bool_vector()

# Seleccionar una columna para codificar en one-hot

encoded = oneHotEncoding(targets)

# Mostrar el resultado
println(encoded)
=#

# PARTE 2
# --------------------------------------------------------------------------
#= uncoment to use
min_val, max_val = calculateMinMaxNormalizationParameters(inputs)
#= Solo para el print sin el Float32
min_val_float64 = convert(Array{Float64}, min_val)
max_val_float64 = convert(Array{Float64}, max_val)=#
println("Minimum values per column: ", min_val)
println("Maximum values per column: ", max_val)

mean_col, std_col = calculateZeroMeanNormalizationParameters(inputs)
#= Solo para el print sin el Float32
mean_col_float64 = convert(Array{Float64}, mean_col)
std_col_float64 = convert(Array{Float64}, std_col)=#
println("Mean values per column: ", mean_col)
println("Standart Deviation values per column: ", std_col)
=#

# PARTE 3
# --------------------------------------------------------------------------
#= uncoment to use
normalize_values = normalizeMinMax!(inputs, calculateMinMaxNormalizationParameters(inputs))
println("Completa: ",normalize_values[1:15])
normalize_values = normalizeMinMax!(inputs)
println("Incompleta: ",normalize_values[1:15])
normalize_values = normalizeMinMax(inputs, calculateMinMaxNormalizationParameters(inputs))
println("Completa con copy: ",normalize_values[1:15])
normalize_values = normalizeMinMax(inputs)
println("Incompleta con copy: ",normalize_values[1:15])
=#

# PARTE 4
# --------------------------------------------------------------------------
#= uncoment to use
new_normalize_values = normalizeZeroMean!(inputs, calculateZeroMeanNormalizationParameters(inputs))
println("Completa: ",new_normalize_values[1:15])
new_normalize_values = normalizeZeroMean!(inputs)
println("Incompleta: ",new_normalize_values[1:15])
new_normalize_values = normalizeZeroMean(inputs, calculateZeroMeanNormalizationParameters(inputs))
println("Completa con copy: ",new_normalize_values[1:15])
new_normalize_values = normalizeZeroMean(inputs)
println("Incompleta con copy: ",new_normalize_values[1:15])
=#

# PARTE 5
# --------------------------------------------------------------------------
#= uncoment to use
test = [0.1,0.5,0.1,0.9,0.8]
classified = classifyOutputs(test)
println("Classified outputs: ", classified)


# Test para la función classifyOutputs con vector de salidas
function test_classifyOutputs_vector()
    outputs = [0.2, 0.6, 0.4, 0.8]
    threshold = 0.5
    expected_results = [false, true, false, true]
    results = classifyOutputs(outputs; threshold)
    @assert results == expected_results
    println("Test para classifyOutputs con vector de salidas pasó correctamente.")
end

# Test para la función classifyOutputs con matriz de salidas
function test_classifyOutputs_matrix()
    outputs = [0.2 0.6; 0.4 0.8]
    threshold = 0.5
    expected_results = [false true; false true]
    results = classifyOutputs(outputs; threshold)
    @assert results == expected_results
    println("Test para classifyOutputs con matriz de salidas pasó correctamente.")
end

# Test para la función classifyOutputs con un solo elemento
function test_classifyOutputs_single_element()
    outputs = [0.8]
    threshold = 0.5
    expected_results = [true]
    results = classifyOutputs(outputs; threshold)
    @assert results == expected_results
    println("Test para classifyOutputs con un solo elemento pasó correctamente.")
end

# Ejecutar los tests
test_classifyOutputs_vector()
test_classifyOutputs_matrix()
test_classifyOutputs_single_element()
=#

# PARTE 6
# --------------------------------------------------------------------------
#= uncoment to use
# Test para la función accuracy con vectores de valores booleanos
function test_accuracy_bool_vectors()
    # Datos de prueba
    targets = [true, true, false, false, true]
    outputs = [true, true, true, false, false]

    # Llamada a la función accuracy
    result = accuracy(targets, outputs)

    # Verificar si el resultado es correcto
    @assert result == 0.6

    println("Test para accuracy con vectores de valores booleanos pasó correctamente.")
end

# Test para la función accuracy con matrices de valores booleanos (1 columna)
function test_accuracy_bool_matrices_single_column()
    # Datos de prueba
    targets = [true, false, true, false, true]
    outputs = [true, false, false, false, true]

    # Llamada a la función accuracy
    result = accuracy([targets], [outputs])

    # Verificar si el resultado es correcto
    @assert result == 0.6

    println("Test para accuracy con matrices de valores booleanos (1 columna) pasó correctamente.")
end

# Test para la función accuracy con vectores de valores reales y umbral por defecto
function test_accuracy_real_vector_default_threshold()
    # Datos de prueba
    targets = [true, true, false, false, true]
    outputs = [0.8, 0.9, 0.2, 0.4, 0.7]

    # Llamada a la función accuracy
    result = accuracy(targets, outputs)

    # Verificar si el resultado es correcto
    @assert result == 0.6

    println("Test para accuracy con vectores de valores reales y umbral por defecto pasó correctamente.")
end

# Test para la función accuracy con matrices de valores reales y umbral especificado
function test_accuracy_real_matrices_custom_threshold()
    # Datos de prueba
    targets = [true, false, true, false, true]
    outputs = [0.8 0.6; 0.9 0.1; 0.2 0.3; 0.4 0.5; 0.7 0.8]

    # Llamada a la función accuracy
    result = accuracy(targets, outputs, threshold=0.7)

    # Verificar si el resultado es correcto
    @assert result == 0.4

    println("Test para accuracy con matrices de valores reales y umbral especificado pasó correctamente.")
end

# Ejecutar los tests
test_accuracy_bool_vectors()
# test_accuracy_bool_matrices_single_column()
# test_accuracy_real_vector_default_threshold()
# test_accuracy_real_matrices_custom_threshold()
=#

# PARTE 7
# --------------------------------------------------------------------------
#= uncoment to use
# topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
topology = [10]
topology2 = [20]
topology3 = [100]

ann = buildClassANN(2, topology, 2)
ann2 = buildClassANN(2, topology2, 2)
ann3 = buildClassANN(2, topology3, 2)

println("Red: ", ann)
println("Red2: ", ann2)
println("Red3: ", ann3)
=#

# PARTE 8
# --------------------------------------------------------------------------
#= uncoment to use
topology = [5]
dataset = oneHotEncoding(targets)
ann, loss = trainClassANN(topology, dataset)
println(ann)
println(loss)
=#

# PARTE 9
# --------------------------------------------------------------------------
#=
using Test
function test_holdOut(N::Int, P::Real)
    @test begin
        # Check for valid percentage
        if P <= 0 || P >= 1
            throw(ArgumentError("P must be between 0 and 1"))
        end

        # Call the function
        (train_indices, test_indices) = holdOut(N, P)

        # Check sizes
        @assert length(train_indices) + length(test_indices) == N
        # No duplicates
        @assert all(unique(train_indices) == train_indices)
        @assert all(unique(test_indices) == test_indices)
        # Indices within range
        @assert all(i >= 1 && i <= N for i in train_indices)
        @assert all(i >= 1 && i <= N for i in test_indices)
    end
end
function test_holdOut3(N::Int, Pval::Real, Ptest::Real)
    @test begin
        # Check for valid percentages
        if Pval <= 0 || Pval >= 1
            throw(ArgumentError("Pval must be between 0 and 1"))
        end
        if Ptest <= 0 || Ptest >= 1
            throw(ArgumentError("Ptest must be between 0 and 1"))
        end

        # Call the function
        (train_indices, val_indices, test_indices) = holdOut(N, Pval, Ptest)

        # Check sizes
        @assert length(train_indices) + length(val_indices) + length(test_indices) == N
        # No duplicates within each set
        @assert all(unique(train_indices) == train_indices)
        @assert all(unique(val_indices) == val_indices)
        @assert all(unique(test_indices) == test_indices)
        # Indices within range
        @assert all(i >= 1 && i <= N for i in train_indices)
        @assert all(i >= 1 && i <= N for i in val_indices)
        @assert all(i >= 1 && i <= N for i in test_indices)
    end
end

# Run tests for various cases
test_holdOut3(100, 0.1, 0.2)
test_holdOut3(1000, 0.3, 0.1)
test_holdOut3(50, 0.25, 0.25)
# Run tests for various cases
test_holdOut(100, 0.2)
test_holdOut(1000, 0.1)
test_holdOut(50, 0.5)

run_tests()
=#