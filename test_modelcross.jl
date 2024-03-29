
include("35634619Y_48114048A_32740686W_48111913F.jl")
using ScikitLearn
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier
println("BAGUETTE")

# Definimos los hiperparámetros del modelo SVM
svm_hyperparameters = Dict("C" => 1.0, "kernel" => "rbf", "gamma" => 0.1, "degree" => 3, "coef0" => 0.0)

# Creamos datos de ejemplo
inputs = rand(100, 4)  # 100 muestras, 4 características
targets = rand(["A", "B", "C"], 100)  # Etiquetas aleatorias

# Definimos los índices de validación cruzada (por ejemplo)
crossValidationIndices = [1:20; 21:40; 41:60; 61:80; 81:100]

# Realizamos la validación cruzada para SVM
results_svm = modelCrossValidation(:SVC, svm_hyperparameters, inputs, targets, crossValidationIndices)

# Mostramos los resultados
println("Resultados de SVM:")
println("Precisión media: ", results_svm[1])
println("Tasa de error media: ", results_svm[2])
println("Sensibilidad media: ", results_svm[3])
println("Especificidad media: ", results_svm[4])
println("VPP media: ", results_svm[5])
println("VPN media: ", results_svm[6])
println("F1 media: ", results_svm[7])



#Ejemplo arboldecision:

# Definimos los hiperparámetros del modelo Árbol de decisión
dt_hyperparameters = Dict("max_depth" => 5)

# Creamos datos de ejemplo (mantenemos los mismos datos para una comparación justa)
inputs = rand(100, 4)  # 100 muestras, 4 características
targets = rand(["A", "B", "C"], 100)  # Etiquetas aleatorias

# Realizamos la validación cruzada para Árbol de decisión
results_dt = modelCrossValidation(:DecisionTreeClassifier, dt_hyperparameters, inputs, targets, crossValidationIndices)

# Mostramos los resultados
println("Resultados de Árbol de decisión:")
println("Precisión media: ", results_dt[1])
println("Tasa de error media: ", results_dt[2])
println("Sensibilidad media: ", results_dt[3])
println("Especificidad media: ", results_dt[4])
println("VPP media: ", results_dt[5])
println("VPN media: ", results_dt[6])
println("F1 media: ", results_dt[7])

#Ejemplo KNN:
# Definimos los hiperparámetros del modelo kNN
knn_hyperparameters = Dict("n_neighbors" => 3)

# Creamos datos de ejemplo (mantenemos los mismos datos para una comparación justa)
inputs = rand(100, 4)  # 100 muestras, 4 características
targets = rand(["A", "B", "C"], 100)  # Etiquetas aleatorias

# Realizamos la validación cruzada para kNN
results_knn = modelCrossValidation(:KNeighborsClassifier, knn_hyperparameters, inputs, targets, crossValidationIndices)

# Mostramos los resultados
println("Resultados de kNN:")
println("Precisión media: ", results_knn[1])
println("Tasa de error media: ", results_knn[2])
println("Sensibilidad media: ", results_knn[3])
println("Especificidad media: ", results_knn[4])
println("VPP media: ", results_knn[5])
println("VPN media: ", results_knn[6])
println("F1 media: ", results_knn[7])
