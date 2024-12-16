using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

num_samples = 100
num_features = 10
dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
inputs = convert(Array{Float32,2},inputs);
targets = dataset[:,5];
crossValidationIndices = rand(1:5, num_samples) # Índices para validación cruzada, asumiendo 5 folds

# Topología de la red neuronal simulada
topology = [num_features, 5, 2] # Ejemplo: capa de entrada, una capa oculta, capa de salida

# Llamando a la función de validación cruzada
results = ANNCrossValidation(topology, inputs, targets, crossValidationIndices,
    numExecutions=5, # Para hacer el test más rápido
    transferFunctions=fill(σ, length(topology)),
    maxEpochs=100, minLoss=0.01, learningRate=0.01, validationRatio=0.2, maxEpochsVal=10)

println("Resultados de la validación cruzada: ", results)


numFolds = maximum(crossValidationIndices)
    targets_onehot = oneHotEncoding(targets)
    # Variables para almacenar las métricas
    # Comunes para cualquier modelo
    acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
    # Realizar la validación cruzada
    fold = 1
        # Separar los datos de entrenamiento y test
        train_indices = findall(i -> i != fold, crossValidationIndices)
        test_indices = findall(x -> x == fold, crossValidationIndices)

        train_inputs = inputs[:, train_indices]
        train_targets = reshape(targets_onehot'[train_indices], 1, :)
        test_inputs = inputs[:, test_indices]
        test_targets = reshape(targets_onehot'[test_indices], 1, :)
        vec(test_targets)
            # Entrenar la red neuronal
                # Determinar el tamaño del conjunto de validación
                # N = lenght(train_indices)
                # (train_idx, val_idx) = holdOut(N, validationRatio)

                total_size = size(inputs, 2) + size(targets_onehot, 2)
                size_train = size(train_inputs, 2)

                validationRatio = (size_train * 0.2) / total_size
                P = (1 - 0.2)
                N = size_train
                # FALLA DESDE AQUI
                (train_idx, val_idx) = holdOut(N, P)

                # Seleccionar conjuntos de entrenamiento y validación
                validation_inputs = train_inputs[:, val_idx]
                validation_targets = reshape(train_targets[val_idx], 1, :)

                train_inputs_adj = train_inputs[:, train_idx]
                train_targets_adj = reshape(train_targets[train_idx], 1, :)

                # Entrenar con conjuntos ajustados
                ann_trained = trainClassANN(topology, (train_inputs_adj, train_targets_adj),
                                            validationDataset=(validation_inputs, validation_targets), testDataset=(test_inputs, test_targets))



            numFolds = maximum(crossValidationIndices)
            targets_onehot = oneHotEncoding(targets)
            # Variables para almacenar las métricas
            # Comunes para cualquier modelo
            acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
            # Realizar la validación cruzada
            fold = 1
                # Separar los datos de entrenamiento y test
                train_indices = findall(i -> i != fold, crossValidationIndices)
                test_indices = findall(x -> x == fold, crossValidationIndices)
        
                train_inputs = inputs[:, train_indices]
                train_targets = reshape(targets_onehot[train_indices], 1, :)
                test_inputs = inputs[:, test_indices]
                test_targets = reshape(targets_onehot[test_indices], 1, :)
        
                    # Entrenar la red neuronal
                    validationRatio = 0.2
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
                        validation_inputs = train_inputs[:, val_idx]
                        validation_targets = reshape(train_targets[val_idx], 1, :)
        
                        train_inputs_adj = train_inputs[:, train_idx]
                        train_targets_adj = reshape(train_targets[train_idx], 1, :)
                        
                        # Entrenar con conjuntos ajustados
                        ann_trained = trainClassANN(topology, (train_inputs_adj, train_targets_adj),
                                                    validationDataset=(validation_inputs, validation_targets), testDataset=(test_inputs, test_targets))
