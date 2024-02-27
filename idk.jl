function trainClassANN(topology::AbstractArray{<:Int,1},
    dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
    transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
    maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
    # topology = [numero capas ocultas, numero de neuronas, (opcional) funciones de transferencia]
    inputs, targets = dataset
    ann = buildClassANN(size(inputs,2), topology, size(targets,2); transferFunctions=transferFunctions)

    opt_state = Flux.setup(Adam(learningRate), ann)
    counter = 1
    loss(model, x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(model(x),y) : Losses.crossentropy(model(x),y);
    # training loss > loss
    while counter != maxEpochs+1
        #=
        FALLA EN EL CALCULO DE LOSS AL ENTRENAR
        =#
        Flux.train!(loss, ann, [(inputs', targets')], opt_state)
        counter += 1
    end
    return ann, loss
end;