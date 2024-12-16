include("35634619Y_48114048A_32740686W_48111913F.jl")
topology = [2, 3, 2]
inputs = [0 0 1 1; 0 1 0 1]
targets = [1, 2, 3, 4, 0]
transferFunctions = [σ, σ, σ]
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.01
maxEpochsVal = 20
crossValidationIndices = [1, 2, 5]

# Calcular el número de folds

numFolds = maximum(crossValidationIndices)
println(numFolds)
targets_onehot = oneHotEncoding(targets)
transpose(targets_onehot)
# Variables para almacenar las métricas
# Comunes para cualquier modelo
acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
# Realizar la validación cruzada
fold = 1
validationRatio = 0.02
    # Separar los datos de entrenamiento y test
    train_indices = findall(i -> i != fold, crossValidationIndices)
    test_indices = findall(x -> x == fold, crossValidationIndices)

    # crossValidation indices tiene que ser menor que el numero de columnas?
    train_inputs = inputs[: , train_indices]
    train_targets = reshape(targets_onehot[train_indices], 1, :)
    test_inputs = inputs[:, test_indices]
    test_targets = reshape(targets_onehot[test_indices], 1, :)

    
        # Entrenar la red neuronal
            # Determinar el tamaño del conjunto de validación
            # N = lenght(train_indices)
            # (train_idx, val_idx) = holdOut(N, validationRatio)

            total_size = size(inputs, 2) + size(targets_onehot, 2)
            size_train = size(train_inputs, 2)

            validationRatio = (size_train * validationRatio) / total_size
            P = (1 - validationRatio)
            N = size_train
            # FALLA DESDE AQUI
            (train_idx, val_idx) = holdOut(N, P)

            # Seleccionar conjuntos de entrenamiento y validación
            validation_inputs = train_inputs[:,val_idx]
            validation_targets = reshape(train_targets[val_idx], 1, :)

            train_inputs_adj = train_inputs[: , train_idx]
            train_targets_adj = reshape(train_targets[train_idx], 1, :)
            
            # Entrenar con conjuntos ajustados
            ann_trained = trainClassANN(topology, (train_inputs_adj, train_targets_adj),
                                        validationDataset=(validation_inputs, validation_targets), testDataset=(test_inputs, test_targets),
                                        transferFunctions=transferFunctions,
                                        maxEpochs=maxEpochs, minLoss=minLoss,
                                        learningRate=learningRate,
                                        maxEpochsVal=maxEpochsVal)

                                        # Modificar esto si es necesario que sea multiclase
        confusion_matrix = confusionMatrix(vec(ann_trained[1](test_inputs)), vec(test_targets))

        push!(acc, confusion_matrix[1])
        push!(fail_rate, confusion_matrix[2])
        push!(sensitivity, confusion_matrix[3])
        push!(specificity, confusion_matrix[4])
        push!(VPP, confusion_matrix[5])
        push!(VPN, confusion_matrix[6])
        push!(F1, confusion_matrix[7])

include("35634619Y_48114048A_32740686W_48111913F.jl")
function test_trainClassANN_2_classes()
    topology = [2, 3, 2]
    trainingDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    validationDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    testDataset = (rand(100, 2), rand(100, 2) .> 0.5)
    transferFunctions = [σ, σ, σ]
    maxEpochs = 0
    minLoss = 0.0
    learningRate = 0.01
    maxEpochsVal = 20

    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal
    )
    println(train_losses)
    println(val_losses)
    println(test_losses)
    @test length(train_losses)== 1
    @test length(val_losses) == 0
    @test length(test_losses) == 0
    @test best_model isa Flux.Chain
end

test_trainClassANN_2_classes()