using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
using ScikitLearn

modelType = :KNeighborsClassifier
modelHyperparameters = Dict("n_neighbors" => 1)
inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
targets = [0, 1, 0, 1]
crossValidationIndices = [1, 1, 2, 2]
results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)
println(results)


include("35634619Y_48114048A_32740686W_48111913F.jl")
modelType = :KNeighborsClassifier
modelHyperparameters = Dict("n_neighbors" => 3)
inputs = [0.1 0.2 0.3 0.4; 0.5 0.6 0.7 0.8]
targets = [0, 1, 0, 1]
crossValidationIndices = [1, 1, 2, 2]
# Convertimos el vector de salidas deseada a texto para evitar errores con la librería de Python
targets = string.(targets)

# Creamos vectores para almacenar los resultados de las métricas en cada fold
acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
# Comenzamos la validación cruzada
for test_indices in unique(crossValidationIndices)
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
    model = nothing
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
    # Problema aqui
    model = fit!(model, train_inputs, train_targets)
    predictions = predict(model, test_inputs)
    metrics = confusionMatrix(predictions, test_targets)
    push!(acc, metrics[1])
    push!(fail_rate, metrics[2])
    push!(sensitivity, metrics[3])
    push!(specificity, metrics[4])
    push!(VPP, metrics[5])
    push!(VPN, metrics[6])
    push!(F1, metrics[7])
end
return ((mean(acc), std(acc)), 
        (mean(fail_rate), std(fail_rate)),
        (mean(sensitivity), std(sensitivity)), 
        (mean(specificity), std(specificity)),
        (mean(VPP), std(VPP)), 
        (mean(VPN), std(VPN)), 
        (mean(F1), std(F1)))
