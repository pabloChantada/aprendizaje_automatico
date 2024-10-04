using ScikitLearn: @sk_import, fit!, predict
@sk_import svm: SVC

Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}}


function selectInstances(batch::Batch, indices::Any)
    
    selected_inputs = batchInputs(batch)[indices, :]
    selected_targets = batchTargets(batch)[indices]
    return Batch(selected_inputs, selected_targets)    
    
end;

function joinBatches(batch1::Batch, batch2::Batch)
    new_inputs = vcat(batchInputs(batch1), batchInputs(batch2))
    new_targets = vcat(batchTargets(batch1), batchTargets(batch2))
    return Batch(new_inputs, new_targets)
end;


function divideBatches(dataset::Batch, batchSize::Int; shuffleRows::Bool=false)
    
    inputs = batchInputs(dataset)
    targets = batchTargets(dataset)

    if shuffleRows
        shuffled_indices = shuffle(1:batchLength(dataset))
        inputs = inputs[shuffled_indices, :]
        targets = targets[shuffled_indices]
    end

    total_length = batchLength(dataset)
    indices = 1:total_length
    partitions = collect(Base.Iterators.partition(indices, batchSize))

    return [selectInstances(dataset, partition) for partition in partitions]

end;