using Flux;
using Flux.Losses;
using FileIO;
using JLD2;
using Images;
using DelimitedFiles;
using Test;
using Statistics
using LinearAlgebra;
using StatsBase;


# Definición del tipo Batch
Batch = Tuple{AbstractArray{<:Real,2}, AbstractArray{<:Any,1}} 

function batchInputs(batch::Batch) 
    matriz = batch[1] 
    return matriz      
end


function euclideanDistances(memory::Batch, instance::AbstractArray{<:Real,1}) 
    matriz_entradas = batchInputs(memory)  
    instance_transpuesta = instance'        
    diferencia = matriz_entradas .- instance_transpuesta  
    diferencia_cuadrado = diferencia .^ 2   
    suma_filas = sum(diferencia_cuadrado, dims=2)  
    distancias = sqrt.(suma_filas)          
    distancias_vector = vec(distancias)      

    return distancias_vector
end

# Definición de un batch
memory = (
    [1 2 3; 4 5 6; 7 8 9],  # Matriz de entradas (3 filas, 3 columnas)
    ["a", "b", "c"]          # Vector de etiquetas o información adicional
)

# Instancia a comparar
instance = [1, 2, 3]  

# Cálculo de las distancias euclídeas
distancias = euclideanDistances(memory, instance)
println(distancias)


function predictKNN(memory::Batch, instance::AbstractArray{<:Real,1}, k::Int) 
    distance = euclideanDistances(memory,instance)
    indices_vecinos = partialsortperm(distance, k) 
    salidas_vecinos = memory[2][indices_vecinos]
    valor_prediccion = mode(salidas_vecinos)
    return valor_prediccion 

end;

# Definición de un batch
memory = (
    [1 2 3; 4 5 6; 7 8 9],  # Matriz de entradas
    ["a", "b", "c"]          # Vector de etiquetas
)

# Instancia a predecir
instance = [5, 5, 5]  

# Número de vecinos
k = 2  

# Predicción
prediccion = predictKNN(memory, instance, k)
println("La predicción es: ", prediccion)


function predictKNN(memory::Batch, instances::AbstractArray{<:Real,2}, k::Int)
    
    predicciones = [predictKNN(memory, instance, k) for instance in eachrow(instances)]  
    return predicciones  

end;