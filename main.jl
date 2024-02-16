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
    numClasses = length(classes);
    @assert numClasses>1 "solo hay una clase";
    if numClasses==2
        # Si solo hay dos clases, se devuelve una matriz con una columna.
        one_col_matrix = reshape(feature.==classes[1], :, 1);
        return one_col_matrix
    else
        # Si hay mas de dos clases se devuelve una matriz con una columna por clase.
        oneHot = Array{Bool,2}(undef, length(feature), numClasses);
        for numClass = 1:numClasses
            oneHot[:,numClass] .= (feature.==classes[numClass]);
        end;
        return oneHot
    end;
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

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    # Normalizar entre el max y el min
    # Matriz de valores a normalizar y parametros de normalizacion
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    dataset ./= (range_values)
    dataset[:, vec(min_values.==max_values)] .= 0
    return dataset
end;

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    return normalizeMinMax!(dataset, normalizationParameters)
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    # MEJOR COPY O DEEPCOPY EN ESTE CASO?
    new_dataset = copy(dataset)
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= min_values
    range_values = max_values .- min_values
    # Caso de que los valores sean 0
    # range_values[range_values .== 0] .= 1
    new_dataset ./= (range_values)
    new_dataset[:, vec(min_values.==max_values)] .= 0
    return new_dataset
end;

function normalizeMinMax(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateMinMaxNormalizationParameters(dataset)
    normalizeMinMax(dataset, normalizationParameters)
end;

# PARTE 4
# --------------------------------------------------------------------------

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= avg_values
    dataset ./= std_values
    dataset[:, vec(std_values .== 0)] .= 0;
    return dataset
end;

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
    normalizationParameters = calculateZeroMeanNormalizationParameters(dataset)
    return normalizeZeroMean!(dataset, normalizationParameters)
end;

function normalizeZeroMean(dataset::AbstractArray{<:Real,2},normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
    new_dataset = copy(dataset)
    avg_values, std_values = normalizationParameters[1], normalizationParameters[2]
    new_dataset .-= avg_values
    new_dataset ./= std_values
    new_dataset[:, vec(std_values .== 0)] .= 0;
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
    results = outputs .>= threshold
    return results
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    # Recibe una matriz en vez de un vector
    if size(outputs) == 1
        vector_outputs = classifyOutputs(outputs[:]; threshold)
        return reshape(vector_outputs, : ,1)
    else
        # (maximo cada fila/col, coordenadas del maximo)
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2); 
        outputs = falses(size(outputs)); 
        outputs[indicesMaxEachInstance] .= true
        return outputs
    end;
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
        return 1.0 - count(mismatches .> 0) / size(targets, 1)
    end;
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
        return accuracy(targets[:, 1], outputs[:, 1])
    else
        #Llamamos a la función classifyOutputs que convierte la matriz de valores reales outputs 
        #en una matriz de valores booleanos.
        outputs_bool = classifyOutputs(outputs)
        #Llamamos a la función creada antes y esta se encargará de comparar los vectores booleanos targets y outputs_bool y calcular la precisión del modelo.
        return accuracy(targets, outputs_bool)
    end;
end;

# PARTE 7
# --------------------------------------------------------------------------

function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    # global ann, numInputsLayer para usar fuera de la funcion
    @assert !isempty(topology) "No hay capas ocultas"
    
    ann = Chain()
    numInputsLayer = numInputs
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[1]))
        numInputsLayer = numOutputsLayer
    end;
    # Ultima capa
    if numOutputs > 2
        ann = Chain(ann..., numOutputs, softmax)
    else
        ann = Chain(ann..., numOutputs, sigmoid)
    end;
    return ann
end;

#PARTE 8
# --------------------------------------------------------------------------

# MIRAR COMO TESTEAR
function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    #=
    topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    Criterios de Parada
        maxIterations
        minLoss
        learningRate
    =#
    # Creacion de la neurona
    inputs = dataset[1];
    num_inputs = size(inputs)
    targets = dataset[2];
    num_targets = size(targets)
    ann = buildClassANN(num_inputs, topology, num_targets; transferFunctions=transferFunctions)
    
    opt_state = Flux.setup(Adam(learningRate), ann)
    loss_data = []
    counter = 0
    while counter != maxEpochs
        Flux.train!(loss, ann, [(inputs', targets')], opt_state);
        loss(ann, inputs, targets) = Losses.binarycrossentropy(ann(inputs), targets) 
        push!(loss_data, (counter, loss))
        counter += 1
    end;
    return (ann, loss_data)
end;

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 

    if size(targets, 2) > 1
        targets = reshape(targets, (:, 1))
    end;
    dataset = (inputs, targets)
    trainClassANN(topology, dataset; transferFunctions, maxEpochs, minLoss, learningRate)
end;

# PARTE 9
# --------------------------------------------------------------------------

function holdOut(N::Int, P::Real)
    # N -> numero de patrones
    # P -> valor entre 0 y 1, indica el porcentaje para el test
    # numero de patrones para el teset
    @assert P > 0 && P < 1 "Valores fuera de rango"
    test_size = Int(floor(N * P))
    # permutamos los datos
    index_perm = randperm(N)
    index_test = index_perm[1:test_size]
    index_train = index_perm[test_size+1:end]
    @assert size(index_test) + size(index_train) == N
    return(index_train, index_test)
end;

function holdOut(N::Int, Pval::Real, Ptest::Real)
    # N -> numero de patrones
    # Pval -> valor entre 0 y 1, tasa del conjunto validacion
    # Ptest -> valor entre 0 y 1, tasa del conjunto validacion
    index_test = holdOut(N, Ptest)
    index_val = holdOut(N, Pval)
    index_train = index_test[1] + index_val[1]
    @assert size(index_test) + size(index_train) + size(index_val) == N
    return (index_train, index_val[2], index_test[2])
end;