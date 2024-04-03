using Test
using Flux;
using Flux.Losses;
using FileIO;
using DelimitedFiles;
using Statistics;
using Random;
using ScikitLearn;
using LinearAlgebra;
using ScikitLearn: fit!, predict;
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Convertimos el vector de salidas deseada a texto para evitar errores con la librería de Python
targets = string.(targets)
# Creamos vectores para almacenar los resultados de las métricas en cada fold
acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
# Comenzamos la validación cruzada
# for fold in unique(crossValidationIndices)
test_indices = 1
    # Obtenemos los índices de entrenamiento
    train_indices = filter(x -> !(x in test_indices), 1:size(inputs, 1))
    # Convertimos el rango en un vector de índices
    test_indices = collect(test_indices)

    # Dividimos los datos en entrenamiento y prueba
    train_inputs = inputs[train_indices, :]
    train_targets = targets[train_indices]
    test_inputs = inputs[test_indices, :]
    test_targets = targets[test_indices]

    # Creamos el modelo según el tipo especificado
    # model = nothing
    if modelType == :SVC
        model = SVC(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
                    degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
                    coef0=modelHyperparameters["coef0"])
    elseif modelType == :DecisionTreeClassifier
        model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"])
    elseif modelType == :KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"])
    end

    # Entrenamos el modelo
    model = fit!(model, train_inputs, train_targets)
    # Problema aqui
    test_inputs = reshape(test_inputs, 1, :)  # Convierte a 1 fila y múltiples columnas
    predictions = predict(model, test_inputs)
    # ni puta idea de que es un array{String, 0} tbh




function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    # Matrices cuadradadas = clases x clases
    # Muestra la distribucion de los patrones y la clasificacion que hace el modelo
    # Filas -> como ha efectuado el modelo
    # Columnas -> valores reales
    #=
    [VN FP;
     FN VP]
    =#
    matrix = [
    sum((outputs .== false) .& (targets .== false)) sum((outputs .== true) .& (targets .== false));
    sum((outputs .== false) .& (targets .== true)) sum((outputs .== true) .& (targets .== true))
    ]

    vn, fp, fn, vp = matrix[1,1], matrix[1,2], matrix[2,1], matrix[2,2]
    matrix_accuracy = (vn + vp) / (vn + vp + fn + fp)
    fail_rate = (fn + fp) / (vn + vp + fn + fp)
    
    sensitivity = vp / (fn + vp) |> x -> isnan(x) ? 1.0 : x
    specificity = vn / (vn + fp) |> x -> isnan(x) ? 1.0 : x
    positive_predictive_value = vp / (vp + fp) |> x -> isnan(x) ? 1.0 : x
    negative_predictive_value = vn / (vn + fn) |> x -> isnan(x) ? 1.0 : x
    if (sensitivity + positive_predictive_value) == 0
        f_score = 0
    else
        f_score = 2 * (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity)
    end
    return matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix 
end

function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}, method::String)
    # Check that the number of columns of both matrices is equal and not 2
    @assert size(outputs, 2) == size(targets, 2) && size(outputs, 2) != 2

    num_classes = size(outputs, 2)

    # Initialize vectors for sensitivity, specificity, PPV, NPV, and F1 score
    sensitivities = zeros(num_classes)
    specificities = zeros(num_classes)
    ppvs = zeros(num_classes)
    npvs = zeros(num_classes)
    f1s = zeros(num_classes)

    # Iterate over each class
    for i in 1:num_classes
        # Call the confusionMatrix function with the columns corresponding to the current class
        sensitivities[i], specificities[i], ppvs[i], npvs[i], f1s[i] = confusionMatrix(outputs[:, i], targets[:, i])
    end

    confusion_matrix = [sum((outputs[:, i] .== 1) .& (targets[:, j] .== 1)) for i in 1:num_classes, j in 1:num_classes]
    # Calculate the macro or weighted average of the metrics
    if method == "macro"
        sensitivity = mean(sensitivities)
        specificity = mean(specificities)
        ppv = mean(ppvs)
        npv = mean(npvs)
        f1 = mean(f1s)
    elseif method == "weighted"
        weights = sum(targets, dims=1) ./ size(targets, 1)
        sensitivity = dot(sensitivities, weights)
        specificity = dot(specificities, weights)
        ppv = dot(ppvs, weights)
        npv = dot(npvs, weights)
        f1 = dot(f1s, weights)
    else
        error("Invalid method: $method")
    end

    # Calculate the accuracy and error rate
    acc = accuracy(outputs, targets)
    fail_rate = 1 - acc

    return acc, fail_rate, sensitivity, specificity, ppv, npv, f1, confusion_matrix
end

@testset "Test" begin
    outputs = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0]
    targets = Bool[1 0 0; 0 1 0; 0 0 1; 0 1 0; 1 0 0]
    acc, fail_rate, sensitivity, specificity, ppv, npv, f1, confusion_matrix = confusionMatrix(outputs, targets, "macro")
    expected_confusion_matrix = [1 1 0; 1 1 0; 0 0 1]
    @test confusion_matrix == expected_confusion_matrix

    # Test 1
    outputs = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0]
    targets = Bool[1 0 0; 0 1 0; 0 0 1; 0 1 0; 1 0 0]
    _, _, _, _, _, _, _, confusion_matrix = confusionMatrix(outputs, targets, "macro")
    expected_confusion_matrix = [1 1 0; 1 1 0; 0 0 1]
    @test confusion_matrix == expected_confusion_matrix

    # Test 3
    outputs = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0]
    targets = Bool[1 0 0; 0 1 0; 0 0 1; 1 0 0; 0 1 0]
    _, _, _, _, _, _, _, confusion_matrix = confusionMatrix(outputs, targets, "macro")
    expected_confusion_matrix = [2 0 0; 0 2 0; 0 0 1]
    @test confusion_matrix == expected_confusion_matrix

end