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


# PARTE 5 - Preguntar
# --------------------------------------------------------------------------

function classifyOutputs(outputs::AbstractArray{<:Real,1}; threshold::Real=0.5)
    # outputs -> vector de salidas, no necesariamente un una RNA
    # threshold -> opcional
    return outputs .>= threshold
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    if size(outputs, 2) == 1
        vector_outputs = classifyOutputs(outputs[:]; threshold)
        matrix_outputs = reshape(vector_outputs, :, 1)
        return matrix_outputs
    else
        (_, indicesMaxEachInstance) = findmax(outputs, dims=2)
        matrix_outputs = falses(size(outputs))
        matrix_outputs[CartesianIndex.(indicesMaxEachInstance)] .= true
        return matrix_outputs
    end
end

# PARTE 6
# --------------------------------------------------------------------------

function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    #Las matrices targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    #Divide el número de coincidencias entre el tamaño del vector targets para saber la media de aciertos.
    return sum(targets .== outputs) / length(targets)
end;

function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
    @assert size(targets) == size(outputs)
    if size(targets, 2) == 1 && size(outputs, 2) == 1
        return accuracy(vec(outputs), vec(targets))
    else
        mismatches = sum(targets .!= outputs, dims=2)
        return 1.0 - (sum(mismatches .> 0) / size(targets, 1))
    end
end


function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #Los vectores targets y outputs deben tener la misma longitud.
    @assert length(targets) == length(outputs)
    #compara cada elemento del vector outputs con el umbral especificado y devuelve un vector
    #cuyos elementos indican si el valor es mayor o igual al umbral.
    outputs_bool = outputs .>= threshold
    #Llamamos a la función creada antes y esta se encargará de comparar los vectores booleanos targets y outputs_bool y calcular la precisión del modelo.
    return accuracy(outputs_bool, targets)
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
    for i::Int = 1:length(topology)    
        numOutputsLayer = topology[i]    
        ann = Chain(ann..., Dense(numInputsLayer, numOutputsLayer, transferFunctions[i]))
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

#PARTE 8 - comprobnar
# --------------------------------------------------------------------------
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0,0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20)
     
    
    # Separamos los datos de los datasets
    inputs_train, targets_train = trainingDataset
    inputs_val, targets_val = validationDataset
    inputs_test, targets_test = testDataset

    # Transponemos los datos para poder usarlos con Flux
    inputs_train = transpose(inputs_train)
    targets_train = transpose(targets_train)
    inputs_val = transpose(inputs_val)
    targets_val = transpose(targets_val)
    inputs_test = transpose(inputs_test)
    targets_test = transpose(targets_test)

    # Creamos la RNA:
    ann = buildClassANN(size(inputs_train,1), topology, size(targets_train,1); transferFunctions=transferFunctions)

    # Creamos loss, función que calcula la pérdida de la red neuronal durante el entrenamiento. (la del enunciado)
    loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

    # Configuramos el estado del optimizador Adam (Adaptive Moment Estimation) 
    #Ponemos básicamente el ritmo ideal de aprendizaje de nuestra ann RNA.
    opt_state = Flux.setup(Adam(learningRate), ann)

    # Creamos las variables para guardar el mejor modelo y su loss.
    best_model = deepcopy(ann)
    best_val_loss = 0
    epochs_since_best = 0

    # Creamos los vectores que se utilizan para almacenar los valores de pérdida (loss) durante el 
    #entrenamiento de la red neuronal en los conjuntos de datos de entrenamiento, validación y prueba, respectivamente.
    train_losses = Float64[]
    val_losses = Float64[]
    test_losses = Float64[]

    for epoch in 1:maxEpochs

        #Iniciamos el entrenamiento.
        Flux.train!(loss, ann, [(inputs_train, targets_train)], opt_state)
        
        # Calculamos los valores de pérdida de cada conjunto.
        train_loss = loss(ann, inputs_train, targets_train)
        val_loss = loss(ann, inputs_val, targets_val)
        test_loss = loss(ann, inputs_test, targets_test)
        
        # Llevamos los datos recogidos a su vector correspondiente mediante push.
        push!(train_losses, train_loss)
        push!(val_losses, val_loss)
        push!(test_losses, test_loss)
        if train_loss <= minLoss
            break
        end
        # Si el valor de pérdida de el modelo actual es menor que el de la variable 'mejor modelo' definida 
        #anteriormente cambiamos las variables antiguas por las de nuetro modelo actual.
        if val_loss < best_val_loss
            best_val_loss = val_loss
            best_model = deepcopy(ann)
            epochs_since_best = 0           #inicializamos y reiniciamos a la vez un contador de hace cuantos modelos que surgió el mejor.
        
        #si el valor de pérdida no es mejor le sumamos una al contador, y 
        #si este contador sobrepasa o iguala el máximo permitido paramos.
        else
            epochs_since_best += 1
            if epochs_since_best >= maxEpochsVal
                break
            end
        end
        #Criterio de parada temprana: verificamos que el valor de pérdida actual no sea menor que el permitido.

    end
    #Devolvemos los valores del mejor modelo.
    println("Best model: ", typeof(best_model))
    return best_model, train_losses, val_losses, test_losses
end

# seguramente falle aqui
function trainClassANN(topology::AbstractArray{<:Int,1},
    trainingDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
    validationDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    testDataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}}=
    (Array{eltype(trainingDataset[1]),2}(undef,0,0), falses(0)),
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    maxEpochsVal::Int=20) 

    # Convertimos las salidas deseadas a vectores si es necesario
    # dataset = (inputs, reshape(dataset[2], (length(dataset[2]), 1)))
    # Convertimos las salidas deseadas a vectores si es necesario
    if size(trainingDataset[2], 2) > 1
        trainingDataset = (trainingDataset[1], reshape(trainingDataset[2], :, 1))
        validationDataset = (validationDataset[1], reshape(validationDataset[2], :, 1))
        testDataset = (testDataset[1], reshape(testDataset[2], :, 1))
    end

    # Llamamos a la otra versión de la función trainClassANN
    return trainClassANN(topology, trainingDataset; validationDataset, testDataset, transferFunctions, maxEpochs, minLoss, learningRate, maxEpochsVal)
end

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
    # Matrices cuadradadas = clases x clases
    # Muestra la distribucion de los patrones y la clasificacion que hace el modelo
    # Filas -> como ha efectuado el modelo
    # Columnas -> valores reales
    #=
    [VN FP;
     FN VP]
    =#
    matrix = [sum((outputs .== 0) .& (targets .== 0)) sum((outputs .== 0) .& (targets .== 1));
              sum((outputs .== 1) .& (targets .== 0)) sum((outputs .== 1) .& (targets .== 1))] 
    vn, fp, fn, vp = matrix[1,1], matrix[1,2], matrix[2,1], matrix[2,2]
    matrix_accuracy = (vn + vp) / (vn + vp + fn + fp)
    fail_rate = (fn + fp) / (vn + vp + fn + fp)

    # fn + vp > 0 ? 1 : 0 ->if fn + vp > 0; true = 1, false = 0 * ecuacion
    sensitivity = vp / (fn + vp) * (fn + vp > 0 ? 1 : 0)
    specificity = vn / (vn + fp) * (vn + fp > 0 ? 1 : 0)
    positive_predictive_value = vp / (vp + fp) * (vp + fp > 0 ? 1 : 0)
    negative_predictive_value = vn / (vn + fn) * (vn + fn > 0 ? 1 : 0)
    f_score = (positive_predictive_value * sensitivity) / (positive_predictive_value + sensitivity) * (positive_predictive_value + sensitivity > 0 ? 1 : 0)
    return matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix 
end;

function confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    new_outputs = classifyOutputs(outputs; threshold)
    return confusionMatrix(new_outputs, targets)
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,1},
    targets::AbstractArray{Bool,1})
    #
    matrix = confusionMatrix(outputs, targets)[8]
    println("Matriz de confusión: \n")
    println("VN: ", matrix[1,1], " FP: ", matrix[1,2])
    println("FN: ", matrix[2,1], " VP: ", matrix[2,2])
    #
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,1},
    targets::AbstractArray{Bool,1}; threshold::Real=0.5)
    #
    matrix = confusionMatrix(outputs, targets; threshold=threshold)[8]
    println("Matriz de confusión: \n")
    println("VN: ", matrix[1,1], " FP: ", matrix[1,2])
    println("FN: ", matrix[2,1], " VP: ", matrix[2,2])
    #
end;

# 4.2
function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    # outputs = matriz de salidas
    # targets = matriz de salidas deseadas
    @assert size(outputs) == size(targets) "Las matrices deben tener las mismas dimensiones"
    @assert size(outputs, 2) != 2 "Las no pueden ser de dimension 2 (caso bineario)" 
    if size(outputs, 2) == 1
        confusionMatrix(outputs[:], targets[:])
    else
        sensitivity = zeros(size(outputs, 2))
        specificity = zeros(size(outputs, 2))
        positive_predictive_value = zeros(size(outputs, 2))
        negative_predictive_value = zeros(size(outputs, 2))
        f_score = zeros(size(outputs, 2))
        matrix = []

        for i = eachindex(size(outputs, 2))
            _, _, sensitivity[i], specificity[i], positive_predictive_value[i], negative_predictive_value[i], f_score[i], _ = confusionMatrix(outputs[:,i], targets[:,i])
        end
        # doble bucle ???¿?¿?¿?
        for i = eachindex(size(outputs, 2))
            matrix[i,:] = [sensitivity[i], specificity[i], positive_predictive_value[i], negative_predictive_value[i], f_score[i]]    
        end
        acc = accuracy(outputs, targets)
        fail_rate = 1 - acc

        @assert(all([in(output, unique(targets)) for output in outputs])) 
        return acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix
    end;
end;

function confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    #
    new_outputs = classifyOutputs(outputs)
    confusionMatrix(new_outputs, targets; weighted=weighted)
    #
end;

function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    #
    # Codigo a desarrollar mirar bien
    #
end;

function printConfusionMatrix(outputs::AbstractArray{Bool,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; threshold=threshold)[8]
    println("Matriz de confusión: \n")
    println(matrix)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Real,2},
    targets::AbstractArray{Bool,2}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; threshold=threshold)[8]
    println("Matriz de confusión: \n")
    println(matrix)
end;

function printConfusionMatrix(outputs::AbstractArray{<:Any,1},
    targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    matrix = confusionMatrix(outputs, targets; threshold=threshold)[8]
    println("Matriz de confusión: \n")
    println(matrix)
end;

# PARTE 11
# --------------------------------------------------------------------------

function crossvalidation(N::Int64, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int64)
    #
    # Codigo a desarrollar
    #
end;

function ANNCrossValidation(topology::AbstractArray{<:Int,1},
    inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1},
    crossValidationIndices::Array{Int64,1};
    numExecutions::Int=50,
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01,
    validationRatio::Real=0, maxEpochsVal::Int=20)
    #
    # Codigo a desarrollar
    #
end; 

# PARTE 12
# --------------------------------------------------------------------------
# modelCrossValidation
