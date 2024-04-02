
#=

n_samples = 30
outputs = rand(1:3, n_samples)
targets = rand(1:3, n_samples)

confusion_matrix = [
    sum(Bool(outputs[i] == predicted_class) & Bool(targets[i] == actual_class) for i in 1:n_samples)
    for actual_class in 1:3, predicted_class in 1:3
]

# conf_matrix = confusmat(3, outputs, targets) # Aquí, 3 indica el número de clases
acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, confusion_matrix = confusionMatrix(outputs, targets)

num_classes = size(confusion_matrix, 1)
for i in 1:num_classes
    for j in 1:num_classes
        print(confusion_matrix[i, j], " ")
    end
    println()
end
=#
# Test case for binary classification


# Mock data
include("35634619Y_48114048A_32740686W_48111913F.jl")
using Test;


# This setup results in:
# Class 1: 1 TP, 1 FN, 1 FP (Prediction correct for 1 instance, 1 misclassified as Class 2, 1 instance from Class 2 misclassified as Class 1)
# Class 2: 1 TP, 1 FN, 1 FP
# Class 3: 2 TP, 0 FN, 0 FP
# Expected confusion matrix:
# 1 1 0
# 1 1 0
# 0 0 2
using MLBase

outputs = [true false false; false true false; false false true; true false false; false true false; false false true]
targets = [true false false; false true false; false false true; false true false; true false false; false false true]
_,_,_,_,_,_,_,confusion_matrix = confusionMatrix(outputs, targets)
@test confusion_matrix == [1 1 0; 1 1 0; 0 0 2]

n_samples = 6
outputs = rand(1:3, n_samples)
targets = rand(1:3, n_samples)
acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, confusion_matrix = confusionMatrix(outputs, targets)
x = confusmat(3, outputs, targets)
num_classes = size(confusion_matrix, 1)

for i in 1:num_classes
    for j in 1:num_classes
        print(confusion_matrix[i, j], " ")
    end
    println()
end
num_classes = size(x, 1)
for i in 1:num_classes
    for j in 1:num_classes
        print(x[i, j], " ")
    end
    println()
end
@test confusion_matrix == x
