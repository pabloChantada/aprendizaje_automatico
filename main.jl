# Numero Maximo de capas -> 2
# Usar funcion Dense para los Perceptrones Multicapa:
#   Numero de entradas
#   Salidas
#   Funcion de transferencia
# Transponer las matrices creadas
# Usar Float32 casi siempre, para las practicas usarlo siempre

using Flux;
using Flux.Losses;
using FileIO;
using DelimitedFiles;
using Statistics;
using Random;

# PARTE 1
# --------------------------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
    numClasses = length(classes)
    @assert numClasses > 1 "solo hay una clase"
    if numClasses == 2
        # Si solo hay dos clases, se devuelve una matriz con una columna.
        one_col_matrix = reshape(feature .== classes[1], :, 1)
        return one_col_matrix
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase.
        oneHot = Array{Bool,2}(undef, length(feature), numClasses)
        for numClass = 1:numClasses
            oneHot[:, numClass] .= (feature .== classes[numClass])
        end
        return oneHot
    end
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    oneHotEncoding(feature, unique(feature))
end;

function oneHotEncoding(feature::AbstractArray{Bool,1})
    one_col_matrix = reshape(feature, :, 1)
    return one_col_matrix
end;

# PARTE 2
# --------------------------------------------------------------------------

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) # vectores de ints y floats
    # recibe una matriz
    # duvuelve una tupla con
    # matriz de 1 fila -> min de cada columna
    # matriz de 1 fila -> max de cada columna
    min_col = minimum(dataset, dims=1)
    max_col = maximum(dataset, dims=1)
    return (min_col, max_col)
end;

function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
    mean_col = mean(dataset, dims=1)
    std_col = std(dataset, dims=1)
    return (mean_col, std_col)
end;


# PARTE 3
# --------------------------------------------------------------------------

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # Normalizar entre el max y el min
    # Matriz de valores a normalizar y parametros de normalizacion
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    dataset ./= (range_values)
    dataset[:, vec(min_values .== max_values)] .= 0
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # MEJOR COPY O DEEPCOPY EN ESTE CASO?
    new_dataset = copy(dataset)
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    new_dataset ./= (range_values)
    new_dataset[:, vec(min_values .== max_values)] .= 0
    return new_dataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax(dataset, normalizationParameters)
end;

# PARTE 4
# --------------------------------------------------------------------------

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= avg_values
    dataset ./= std_values
    dataset[:, vec(std_values .== 0)] .= 0
    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    new_dataset = copy(dataset)
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= avg_values
    new_dataset ./= std_values
    new_dataset[:, vec(std_values .== 0)] .= 0
    return new_dataset
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    normalizeZeroMean(dataset, normalizationParameters)
end;


# PARTE 5
# --------------------------------------------------------------------------

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    # outputs -> vector de salidas, no necesariamente un una RNA
    # threshold -> opcional
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    # Recibe una matriz en vez de un vector
    if size(outputs, 2) == 1
        reshape(outputs, :, 1)
        vector_outputs = classifyOutputs(outputs[:]; threshold)
        return vector_outputs
        # (maximo cada fila/col, coordenadas del maximo)
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        outputs = falses(size(outputs))
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end
end;

# PARTE 6
# --------------------------------------------------------------------------

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Las matrices targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    #Divide el número de coincidencias entre el tamaño del vector targets para saber la media de aciertos.
    return sum(targets .== outputs) / length(targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    #Las matrices targets y outputs deben tener las mismas dimensiones.
    @assert size(targets) == size(outputs)
    # Si ambas matrices tienen una sola columna:
    if size(targets, 2) == 1 && size(outputs, 2) == 1
        #Llama a la función accuracy creada enteriormente.
        return accuracy(vec(targets), vec(outputs))
        # Si ambas matrices tienen más de una columna
    else
        #calcula la cantidad de diferencias entre las dos matrices, fila por fila.
        mismatches = sum(targets .!= outputs, dims=2)
        #Cuenta el número de filas con al menos una diferencia, y lo divide entre 
        #el número total de filas, valor el cual se resta de 1 para obtener la precisión.
        # print(1.0 - count(mismatches .> 0) / size(targets, 1)) -> 0
        return 1.0 - (sum(mismatches) / (size(targets)[1] * size(targets)[2]))
    end
end;

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #Los vectores targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    #compara cada elemento del vector outputs con el umbral especificado y devuelve un vector
    #cuyos elementos indican si el valor es mayor o igual al umbral.
    outputs_bool = outputs .>= threshold
    #Llamamos a la función creada antes y esta se encargará de comparar los vectores booleanos targets y outputs_bool y calcular la precisión del modelo.
    return accuracy(targets, outputs_bool)
end;

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
    #Las matrices targets y outputs deben tener las mismas dimensiones
    @assert size(targets) == size(outputs)
    #Comprueba si la matriz outputs tiene una sola columna.
    if size(outputs, 2) == 1
        # outputs tiene una sola columna, llamamos a la función accuracy creada anteriormente.
        return accuracy(outputs[:, 1], targets[:, 1])
    else
        #Llamamos a la función classifyOutputs que convierte la matriz de valores reales outputs 
        #en una matriz de valores booleanos.
        outputs_bool = classifyOutputs(outputs; threshold)
        #Llamamos a la función creada antes y esta se encargará de comparar los vectores booleanos targets y outputs_bool y calcular la precisión del modelo.
        return accuracy(outputs_bool, targets)
    end
end;

# PARTE 7
# --------------------------------------------------------------------------
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    # global ann, numInputsLayer
    @assert !isempty(topology) "No hay capas ocultas"

    ann = Chain()
    numInputsLayer = numInputs
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, σ))
        numInputsLayer = numOutputsLayer
    end
    # Ultima capa
    if numOutputs > 2
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity) );
        ann = Chain(ann..., softmax ); 
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs,transferFunctions[end]))
    end;
    return ann
end;

#PARTE 8 - Correguir
# --------------------------------------------------------------------------

function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 
    
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    # Extract inputs and targets from dataset
    inputs, targets = dataset

    # Transpose inputs and targets to match Flux's expectations
    inputs = transpose(inputs)
    targets = transpose(targets)
    # Create the neural network
    ann = buildClassANN(size(inputs,1), topology, size(targets,1); transferFunctions=transferFunctions)

    # Define the loss function
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Define the optimizer
    opt_state = Flux.setup(Adam(learningRate), ann)
    # Train the neural network
    losses = Float64[]
    for epoch in 1:maxEpochs
        Flux.train!(loss, ann, [(inputs, targets)], opt_state)
        current_loss = loss(ann, inputs, targets)
        push!(losses, current_loss)
        if current_loss <= minLoss
            break
        end
    end

    # Return the trained neural network and the losses
    return ann, losses
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

    # targets = reshape(dataset[2], (length(dataset[2]), 1))
    dataset = (inputs, reshape(dataset[2], (length(dataset[2]), 1)))
    trainClassANN(topology, dataset; transferFunctions, maxEpochs, minLoss, learningRate)
end;

# PARTE 9
# --------------------------------------------------------------------------

function holdOut(N::Int, P::Real)
    # N -> numero de patrones
    # P -> valor entre 0 y 1, indica el porcentaje para el test
    # numero de patrones para el test
    @assert P < 1 "Valores de test fuera de rango"
    test_size = Int(floor(N * P))
    # permutamos los datos
    index_perm = randperm(N)
    index_test = index_perm[1:test_size]
    index_train = index_perm[test_size+1:end]
    @assert length(index_test) + length(index_train) == N
    return (index_train, index_test)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # N -> numero de patrones
    # Pval -> valor entre 0 y 1, tasa de patrones del conjunto de validacion
    # Ptest -> valor entre 0 y 1, tasa de patrones del conjunto de prueba
    @assert Pval < 1 "Valores de validacion fuera de rango"
    @assert Ptest < 1 "Valores de test fuera de rango"
    # Permutacion aleatoria de los indices
    #indexes = randperm(N)
    #=
    index_val = holdOut(N, Pval)
    index_test = holdOut(N, Ptest)
    index_train = setdiff(1:N, union(index_test[2], index_val[2]))
    #while any(x -> x in index_val[2], index_test[2])
    #    index_test = holdOut(length(indexes), Ptest)
    =#
    # Obtenemos el vecto de entrenamiento con los indices restantes
    # index_train = setdiff(indexes, vcat(index_val[2], index_test[2]))
    # Calculamos los tamaños de los conjuntos
    Nval = round(Int, N * Pval)
    Ntest = round(Int, N * Ptest)
    Ntrain = N - Nval - Ntest
    # Permutación aleatoria de los índices
    indexes = randperm(N)
    # Obtenemos los índices de los conjuntos
    index_train = indexes[1:Ntrain]
    index_val = indexes[Ntrain + 1:Ntrain + Nval]
    index_test = indexes[Ntrain + Nval + 1:end]
    # Comprobamos que los vectores resultantes sean igual a N
    @assert (length(index_train) + length(index_test) + length(index_val)) == N
    return (index_train, index_val, index_test)
end

# PARTE 10
# --------------------------------------------------------------------------
# 4.1 - Devuelve matriz y  metrica
function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    # Codigo a desarrollar
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
end;

# 4.2
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
end;
