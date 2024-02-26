#=
TESTEAR PARTES:
    8 -> Función no definida para argumentos de tipo
    9 -> Error al ejecutar con argumentos de tipo (Int, Real): MethodError: no method matching +(::Tuple{Int64}, ::Tuple{Int64})
=#
include("main.jl")
#=


dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
inputs = convert(Array{Float32,2},inputs);
targets = dataset[:,5];
=#
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

    # Test case 1: Single column matrix

# PARTE 5
# --------------------------------------------------------------------------
#=
include("main.jl")

# Caso 1: Una matriz con una sola columna
outputs1 = [0.2; 0.7; 0.4]
expected_output1 = [false; true; false]
result1 = classifyOutputs(outputs1)
@assert result1 == expected_output1

# Caso 2: Una matriz con múltiples columnas
outputs2 = [0.2 0.6; 0.7 0.8; 0.4 0.9]
expected_output2 = [false true; false true; false true]
result2 = classifyOutputs(outputs2)
println(result2)
println(expected_output2)
@assert result2 == expected_output2

println("¡Todos los casos de prueba pasaron!")


# Llamar a la función de prueba
test_classifyOutputs()

rows = 5
cols = 5
random_matrix = rand(rows, cols)
for i in 1:rows
    println(join(round.(random_matrix[i, :], digits=2), "\t"))
end
classified = classifyOutputs(random_matrix)
println(classified)
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

    println("Test para accuracy con vectores de valores booleanos pasó correctamente: ", result)
end

# Test para la función accuracy con matrices de valores booleanos (1 columna)
function test_accuracy_bool_matrices_single_column()
    # Datos de prueba
    targets = [true; false; false; false]
    outputs = [true; true; false; true]
    targets2 = [true false true; false false false; true false false; false false true]
    outputs2 = [true true false; true false true; true false true; true false true]
    # Llamada a la función accuracy
    result = accuracy(outputs2, targets2)

    # Verificar si el resultado es correcto
    @assert result == 0.5

    println("Test para accuracy con matrices de valores booleanos (1 columna) pasó correctamente.", result)
end

# Test para la función accuracy con vectores de valores reales y umbral por defecto
function test_accuracy_real_vector_default_threshold()
    # Datos de prueba
    targets = [true, true, false, false, true]
    outputs = [0.8, 0.9, 0.5, 0.4, 0.2]

    # Llamada a la función accuracy
    result = accuracy(outputs, targets)

    # Verificar si el resultado es correcto
    @assert result == 0.6

    println("Test para accuracy con vectores de valores reales y umbral por defecto pasó correctamente.", result)
end

# Test para la función accuracy con matrices de valores reales y umbral especificado
function test_accuracy_real_matrices_custom_threshold()
    # Datos de prueba
    targets = [true; false; false; false]
    outputs = [0.8; 0.9; 0.5; 0.2]
    targets2 = [true false true; false false true; true false false; false false true]
    outputs2 = [0.7 0.2 0.5; 0.3 0.3 0.9; 0.8 0.5 1; 0.2 0.1 0.8]

    # Llamada a la función accuracy
    result = accuracy(outputs2, targets2; threshold=0.7)

    # Verificar si el resultado es correcto
    @assert result == 0.75

    println("Test para accuracy con matrices de valores reales y umbral especificado pasó correctamente.", result)
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

#Nuevo test usado para arreglar la función:
function prueba_error_accuracy()
    # Datos de prueba
    outputs = [0.6, 0.3, 0.8, 0.9]
    targets = [true, false, true, true]

    # Llamada a la función accuracy con matrices de una sola columna y umbral especificado
    accuracy(outputs, targets, threshold=0.5)
end;
=#
# Ejecutar los tests
#test_accuracy_bool_vectors()
# test_accuracy_bool_matrices_single_column()
# test_accuracy_real_vector_default_threshold()
# test_accuracy_real_matrices_custom_threshold()
#prueba_error_accuracy()

#=
# Caso de prueba para la función de precisión con vectores booleanos
outputs_vec = [true, false, true, true]
targets_vec = [true, true, false, true]
println("Precisión (vectores booleanos): ", accuracy(outputs_vec, targets_vec))  # Salida esperada: 0.5

# Caso de prueba para la función de precisión con matrices booleanas
outputs_mat = [true false; false true; true true; true false]
targets_mat = [true false; true true; false true; true false]
println("Precisión (matrices booleanas): ", accuracy(outputs_mat, targets_mat))  # Salida esperada: 0.75

# Caso de prueba para la función de precisión con vectores de valores reales
outputs_real_vec = [0.6, 0.3, 0.8, 0.9]
println("Precisión (vectores de valores reales): ", accuracy(outputs_real_vec, targets_vec))  # Salida esperada: 0.5

# Caso de prueba para la función de precisión con matrices de valores reales
outputs_real_mat = [0.6 0.4; 0.3 0.7; 0.8 0.2; 0.9 0.1]
println("Precisión (matrices de valores reales): ", accuracy(outputs_real_mat, targets_mat))  # Salida esperada: 0.625
# Ejecutar los tests
test_accuracy_bool_vectors()
test_accuracy_bool_matrices_single_column()
test_accuracy_real_vector_default_threshold()
test_accuracy_real_matrices_custom_threshold()
=#
# PARTE 7
# --------------------------------------------------------------------------
# topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
function test_buildClassANN()
    # Caso 1: Red neuronal con una capa oculta
    numInputs1 = 3
    topology1 = [4, 7, 3]
    numOutputs1 = 2
    ann1 = buildClassANN(numInputs1, topology1, numOutputs1)
    # Verifica que la red neuronal tenga la estructura esperada
    println(ann1)
    @assert length(ann1) == 4  # Número de capas
    # Opcionalmente, podrías verificar más propiedades de la red neuronal, como el número de neuronas en cada capa, etc.

    # Caso 2: Red neuronal con múltiples capas ocultas
    numInputs2 = 5
    topology2 = [7, 3, 2, 4]
    numOutputs2 = 3
    ann2 = buildClassANN(numInputs2, topology2, numOutputs2)
    # Verifica que la red neuronal tenga la estructura esperada
    
    println(ann2)
    @assert length(ann2) == 6 # Número de capas
    # Otras verificaciones podrían incluir el número de neuronas en cada capa, las funciones de activación utilizadas, etc.

    println("¡Todos los casos de prueba pasaron!")
end

# Llamar a la función de prueba
test_buildClassANN()
# PARTE 8
# --------------------------------------------------------------------------
include("main.jl")
topology = [2, 3, 1]
# Define the dataset
# Inputs: 2D array of real numbers
inputs = rand(2, 100) # 2 features, 100 samples
# Targets: 2D array of booleans
targets = rand(Bool, 2, 100) # 1 target variable, 100 samples
dataset = (inputs, targets)
# Train the neural network
ann, losses = trainClassANN(topology, dataset)

# Print the losses
println(losses)
println(ann)
#=

using Test

# Definir una función para generar un conjunto de datos de prueba
function generate_test_dataset(n_samples::Int, n_features::Int, n_classes::Int)
    inputs = rand(n_samples, n_features)
    targets = rand(Bool, n_samples)
    return (inputs, targets)
end

# Generar algunos conjuntos de datos de prueba
training_dataset = generate_test_dataset(100, 4, 2)
validation_dataset = generate_test_dataset(50, 4, 2)
test_dataset = generate_test_dataset(50, 4, 2)

# Definir una topología de prueba y otras opciones
topology = [4, 2]
max_epochs = 10
min_loss = 0.0
learning_rate = 0.01
max_epochs_val = 5

best_model, train_losses, val_losses, test_losses = trainClassANN(topology, training_dataset, validationDataset=validation_dataset, testDataset=test_dataset, maxEpochs=max_epochs, minLoss=min_loss, learningRate=learning_rate, maxEpochsVal=max_epochs_val)
println(best_model, train_losses, val_losses, test_losses)
=#
# PARTE 9
# --------------------------------------------------------------------------
#= uncoment to use
using Random;
x = 3
N = 10
for _ in 1:x
    Ptest = round(rand(), digits=2)     # Porcentaje para el conjunto de prueba
    index_train, index_test = holdOut(N, Ptest)
    println()
    println("Test: ", Ptest)
    println("Tamaño del conjunto de entrenamiento:", length(index_train)," -> ", index_train)
    println("Tamaño del conjunto de test: ", length(index_test)," -> ", index_test)
end;
Ptest = 0.3
Pval = 0.2
index_train, index_val, index_test = holdOut(N, Pval, Ptest)
println("Test: ", Ptest)
println("Validacion: ", Pval)
println()
# Verificar el tamaño de los conjuntos resultantes
println("Tamaño del conjunto de entrenamiento:", length(index_train)," -> ", index_train)
println("Tamaño del conjunto de validación:", length(index_val)," -> ", index_val)
println("Tamaño del conjunto de test: ", length(index_test)," -> ", index_test)
=#