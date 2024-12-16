using DelimitedFiles
dataset = readdlm("iris.data", ',')
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Prepara los inputs y los targets
inputs = dataset[:, 1:2]
inputs = convert(Array{Float32,2}, inputs)
targets = dataset[:, 5]

# Define los índices de validación cruzada, topología de la red y otros parámetros si es necesario
crossValidationIndices = [1, 2, 3]  # Esto es solo un placeholder. Necesitas una matriz real para la validación cruzada.
topology = [2, 3, 2, 4, 1]

# Llama a la función ANNCrossValidation
ANNCrossValidation(topology, inputs, targets, crossValidationIndices)  # Asegúrate de pasar todos los argumentos necesarios


targets = oneHotEncoding(targets)

train_index, val_index, test_index = holdOut(size(inputs,1), 0.25, 0.25)

inputs_train = inputs[train_index, :]
targets_train = targets[train_index, :]

inputs_val = inputs[val_index, :]
targets_val = targets[val_index, :]

inputs_test = inputs[test_index, :]
targets_test = targets[test_index, :]

max_vals, min_vals = calculateMinMaxNormalizationParameters(inputs)
mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)

normalizeMinMax!(inputs_train, (min_vals, max_vals))
normalizeMinMax!(inputs_val, (min_vals, max_vals))
normalizeMinMax!(inputs_test, (min_vals, max_vals))

topology = [2, 3, 2, 4, 1]
best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology, (inputs_train, targets_train),
        validationDataset=(inputs_val, targets_val),
        testDataset=(inputs_test, targets_test),
    )

best_model, train_losses, val_losses, test_losses = trainClassANN(
    topology, (inputs_train, targets_train),
    validationDataset=(inputs_val, targets_val),
    testDataset=(inputs_test, targets_test),
)









dataset = readdlm("iris.data", ',')
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Prepara los inputs y los targets
inputs = dataset[:, 1:2]
inputs = convert(Array{Float32,2}, inputs)
targets = dataset[:, 5]
# Define algunos datos de entrada de prueba (deberás sustituir esto con datos reales o simulados)
inputs = rand(Float32, 100, 10) # 100 ejemplos, 10 características cada uno
targets = rand([true, false], 100) # 100 targets binarios
crossValidationIndices = repeat(1:5, 20) # 5-fold cross-validation

modelHyperparameters = Dict("C" => 1.0, "kernel" => "linear", "degree" => 3, "gamma" => "auto", "coef0" => 0.0)
result = modelCrossValidation(:SVC, modelHyperparameters, inputs, targets, crossValidationIndices)
println("Resultados de la validación cruzada: $result")

modelHyperparameters = Dict("max_depth" => 5)
resultDecisionTree = modelCrossValidation(:DecisionTreeClassifier, modelHyperparameters, inputs, targets, crossValidationIndices)
println("Resultados de validación cruzada para DecisionTreeClassifier: $resultDecisionTree")

modelHyperparameters = Dict("n_neighbors" => 3)
resultKNeighbors = modelCrossValidation(:KNeighborsClassifier, modelHyperparameters, inputs, targets, crossValidationIndices)
println("Resultados de validación cruzada para KNeighborsClassifier: $resultKNeighbors")

modelHyperparameters = Dict("topology" => [1,3,5])
resultKNeighbors = modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices)
println("Resultados de validación cruzada para KNeighborsClassifier: $resultKNeighbors")

using Plots;
# Plotly, PyPlot, PlotlyJS y GR -> los mas generales
# backend()


plotly()

train_loss = plot(1:length(train_losses), train_losses, label="Train loss", title="Losses")
val_loss = plot(1:length(val_losses), val_losses, label="Validation loss")
test_loss = plot(1:length(test_losses), test_losses, label="Test loss")
graph = plot()

plot!(graph, 1:length(train_losses), train_losses, label="Train loss")
plot!(graph, 1:length(val_losses), val_losses, label="Validation loss")
plot!(graph, 1:length(test_losses), test_losses, label="Test loss")
display(graph)