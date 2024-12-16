# Error al ejecutar la función con 2 clases: DimensionMismatch: layer Dense(3 => 8, tanh) expects size(input, 1) == 3, but got 0×0 adjoint(::Matrix{Float32}) with eltype Float32
using Flux;
using Flux.Losses;
using FileIO;
using DelimitedFiles;
using Statistics;
using Random;
using ScikitLearn;
using LinearAlgebra;
using ScikitLearn: fit!, predict;
using Test;
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier 

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

topology = [2, 3, 1]
trainingDataset = (rand(2, 100), rand(2, 100) .> 0.5)
validationDataset = (rand(2, 50), rand(2, 50) .> 0.5)
testDataset = (rand(2, 50), rand(2, 50) .> 0.5)
transferFunctions = [σ, σ, σ]
maxEpochs = 1000
minLoss = 0.0
learningRate = 0.01
maxEpochsVal = 20


# Separamos los datos de los datasets
inputs_train = trainingDataset[1]
targets_train = trainingDataset[2]
inputs_val = validationDataset[1]
targets_val = validationDataset[2]
inputs_test = testDataset[1]
targets_test = testDataset[2]

inputs_train = transpose(inputs_train)
targets_train = transpose(targets_train)
inputs_val = transpose(inputs_val)
targets_val = transpose(targets_val)
inputs_test = transpose(inputs_test)
targets_test = transpose(targets_test)


# Si falla es posiblemente por la diferencia de tamaño entre validaiton/test y train
# Si tienen el mismo tamaño funciona, con distinto falla

# Creamos la RNA:
println(size(inputs_train,2))

ann = buildClassANN(size(inputs_train,2), topology, size(targets_train,2); transferFunctions=transferFunctions)

# Creamos loss, función que calcula la pérdida de la red neuronal durante el entrenamiento. (la del enunciado)
loss(model, x, y) = (size(y, 1) == 1) ? Losses.binarycrossentropy(model(x), y) : Losses.crossentropy(model(x), y)

# Configuramos el estado del optimizador Adam (Adaptive Moment Estimation) 
#Ponemos básicamente el ritmo ideal de aprendizaje de nuestra ann RNA.
opt_state = Flux.setup(Adam(learningRate), ann)

# Creamos las variables para guardar el mejor modelo y su loss.
best_model = deepcopy(ann)
best_val_loss = Inf
epochs_since_best = 0

# Creamos los vectores que se utilizan para almacenar los valores de pérdida (loss) durante el 
train_losses = Float64[]
val_losses = Float64[]
test_losses = Float64[]

train_loss = loss(ann, inputs_train', targets_train')
val_loss = loss(ann, inputs_val', targets_val')
test_loss = loss(ann, inputs_test', targets_test')

push!(train_losses, train_loss)
push!(val_losses, val_loss)
push!(test_losses, test_loss)
epoch = 1
while epoch < maxEpochs && val_loss > minLoss

    #Iniciamos el entrenamiento.
    # MIRAR AQUI
    Flux.train!(loss, ann, [(inputs_train', targets_train')], opt_state)
    
    # Calculamos los valores de pérdida de cada conjunto.
    train_loss = loss(ann, inputs_train', targets_train')
    val_loss = loss(ann, inputs_val', targets_val')
    test_loss = loss(ann, inputs_test', targets_test')

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
    epoch += 1
end
return best_model, train_losses, val_losses, test_losses

@testset "trainClassANN Tests" begin


    best_model, train_losses, val_losses, test_losses = trainClassANN(
        topology,
        trainingDataset,
        validationDataset=validationDataset,
        testDataset=testDataset,
        transferFunctions=transferFunctions,
        maxEpochs=maxEpochs,
        minLoss=minLoss,
        learningRate=learningRate,
        maxEpochsVal=maxEpochsVal
    )

    @test length(train_losses) > 1
    @test length(val_losses) > 1
    @test length(test_losses) > 1
    @test best_model isa Flux.Chain
end