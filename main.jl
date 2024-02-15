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

# PARTE 1
# --------------------------------------------------------------------------

function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})

    numClasses = length(classes);
    @assert(numClasses>1);
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
        matriz = oneHot;
    end;
    return matriz
end;

function oneHotEncoding(feature::AbstractArray{<:Any,1})
    classes = unique(feature)
    @assert(numClasses>1);
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
        matriz = oneHot;
    end;
    return matriz
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

#=
# PARTE 5
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
=#

# PARTE 6
#=
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)
=#

# PARTE 7
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    # global ann, numInputsLayer
    @assert !isempty(topology) "No hay capas ocultas"
    
    ann = Chain()
    numInputsLayer = numInputs
    for numOutputsLayer = topology
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[1]))
        numInputsLayer = numOutputsLayer
    end;
    # Ultima capa
    ann = Chain(ann..., numOutputs, softmax)
    return ann
end;

#PARTE 8
#=
# BUCLE WHILE SOLO AQUI PARA ITERAR EL ENTRENAMIENTO
# EL OPTIMIZADOR SE CREA FUERA DEL BUCLE
function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
=#