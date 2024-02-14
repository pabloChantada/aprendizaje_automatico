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

#=
# PARTE 3
function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

function normalizeMinMax( dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

function normalizeMinMax!(dataset::AbstractArray{<:Real,2})

function normalizeMinMax( dataset::AbstractArray{<:Real,2})

# PARTE 4
function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})

function normalizeZeroMean( dataset::AbstractArray{<:Real,2},normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})

function normalizeZeroMean( dataset::AbstractArray{<:Real,2})

# PARTE 5
function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)

# PARTE 6
function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})

function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)

function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; threshold::Real=0.5)

# BUCLE WHILE SOLO AQUI PARA ITERAR EL ENTRENAMIENTO
# EL OPTIMIZADOR SE CREA FUERA DEL BUCLE
# PARTE 7
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology))) 

#PARTE 8
function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)

function trainClassANN(topology::AbstractArray{<:Int,1},
    (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) =#