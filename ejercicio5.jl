function initializeStreamLearningData(datasetFolder::String, windowSize::Int, batchSize::Int)

    fullData = loadStreamLearningDataset(datasetFolder)
    # Crear la memoria inicial
    memory = selectInstances(fullData, 1:windowSize)
    # Dividir el resto de los datos en batches con tamaño batchSize, sin desordenar (shuffleRows=false)
    remainingData = selectInstances(fullData, (windowSize+1):size(fullData[1], 1))
    batches = divideBatches(remainingData, batchSize; shuffleRows=false)
    
    return memory, batches
end;

function addBatch!(memory::Batch, newBatch::Batch)
  
    # Desempaquetar la memoria actual y el nuevo lote de datos
    memoryInputs, memoryOutputs = memory
    newInputs, newOutputs = newBatch
    # Número de instancias del nuevo lote
    batchSize = size(newInputs, 1)
    # Desplazar la memoria hacia adelante, eliminando los datos más antiguos
    memoryInputs[:, 1:end-batchSize] = memoryInputs[:, batchSize+1:end]
    memoryOutputs[1:end-batchSize] = memoryOutputs[batchSize+1:end]
    
    # Añadir los nuevos datos al final de la memoria
    memoryInputs[:, end-batchSize+1:end] = newInputs
    memoryOutputs[end-batchSize+1:end] = newOutputs
end;
