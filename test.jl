include("main.jl")

outputs = [true; false; true; false; true;true; false; true; false; true]
reshaped_outputs = reshape(outputs, :, 5)
targets = [true; true; false; false; true; true; false; true; false; true]
reshaped_targets = reshape(targets, :, 5)
acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix = confusionMatrix(outputs, targets)
@assert(all([in(output, unique(targets)) for output in outputs]))
println(acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix)

# Clase = Columna / Patron = Fila
outputs = reshape(rand(Bool, 16), 4, 4)
targets = reshape(rand(Bool, 16), 4, 4)


sensitivity = zeros(size(outputs, 2))
specificity = zeros(size(outputs, 2))
positive_predictive_value = zeros(size(outputs, 2))
negative_predictive_value = zeros(size(outputs, 2))
f_score = zeros(size(outputs, 2))

for i = 1:(size(outputs, 2))
    #matrix_accuracy, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix
    _, _, sensitivity[i], specificity[i], positive_predictive_value[i], negative_predictive_value[i], f_score[i], _ = confusionMatrix(outputs[:,i], targets[:,i])
end

println(sensitivity)
println(specificity)
println(positive_predictive_value)
println(negative_predictive_value)
println(f_score)

# Reservar memoria para la matriz
matrix = zeros(size(outputs, 1), size(outputs, 2))
# oneHotEncoding(f_score)
# doble bucle
positive_predictive_value
for i = 1:(size(outputs, 2))
    for j = 1:(size(outputs, 2))
        if i == j
            matrix[i, j] = positive_predictive_value[i]
        end
    end
end

if weighted == 1
    combined = sum(data .* size(outputs, 2)) / sum(size(outputs, 2))
    println(combined)
else
    combined = mean(data)
end

acc = accuracy(outputs, targets)
fail_rate = 1 - acc
@assert (all([in(output, unique(targets)) for output in outputs])) "Error: Los valores de salida no están en el conjunto de valores posibles de salida."
return acc, fail_rate, sensitivity, specificity, positive_predictive_value, negative_predictive_value, f_score, matrix