#=
9. Cree los siguientes clasificadores con los conjuntos resultantes del paso anterior. 
o MLP con al menos las siguientes arquitecturas: [50], [100] [100, 50] 
o KNN con valores de vecindario entre 1, 10 y 20 
o SVM con el parámetro C con valores 0.1, 0.5 y 1.0
=#
include("tecnicas_reduccion.jl")

@sk_import neural_network: MLPClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import svm: SVC

using ScikitLearn
using ScikitLearn.Pipelines
using ScikitLearn: fit!, predict

# ===============================
# 1. MLP Classifier
# ===============================
MLP_50 = MLPClassifier(hidden_layer_sizes=[50], max_iter=1000, random_state=42)
MLP_100 = MLPClassifier(hidden_layer_sizes=[100], max_iter=1000, random_state=42)
MLP_100_50 = MLPClassifier(hidden_layer_sizes=[100, 50], max_iter=1000, random_state=42)

function apply_mlp(X_train, X_test, y_train, y_test, model)
    fit!(model, X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = sum(y_pred .== y_test) / length(y_test)
    return accuracy
end

# ===============================
# 2. KNN Classifier
# ===============================
KNN_1 = KNeighborsClassifier(n_neighbors=1)
KNN_10 = KNeighborsClassifier(n_neighbors=10)
KNN_20 = KNeighborsClassifier(n_neighbors=20)

function apply_knn(X_train, X_test, y_train, y_test, model)
    fit!(model, X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = sum(y_pred .== y_test) / length(y_test)
    return accuracy
end


# ===============================
# 3. SVM Classifier
# ===============================
SVM_C0_1 = SVC(C=0.1, kernel="linear", random_state=42)
SVM_C0_5 = SVC(C=0.5, kernel="linear", random_state=42)
SVM_C1_0 = SVC(C=1.0, kernel="linear", random_state=42)

function apply_svm(X_train, X_test, y_train, y_test, model)
    fit!(model, X_train, y_train)
    y_pred = predict(model, X_test)
    accuracy = sum(y_pred .== y_test) / length(y_test)
    return accuracy
end



# ===============================
# APPLY MODELS
# ===============================

# ===============================
# 1. PCA 
# ===============================

#Conseguimos el conjunto de datos
accuracy_pca, model_pca = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, "PCA")

# Aplicamos algunos modelos de ejemplo
accuracy_mlp_50_pca = apply_mlp(X_train, X_test, y_train, y_test, MLP_50)
accuracy_knn_1_pca = apply_knn(X_train, X_test, y_train, y_test, KNN_1)
accuracy_svm_c0_1_pca = apply_svm(X_train, X_test, y_train, y_test, SVM_C0_1)

println("Precisión MLP (50) PCA: $accuracy_mlp_50_pca")
println("Precisión KNN (1) PCA: $accuracy_knn_1_pca")
println("Precisión SVM (C=0.1) PCA: $accuracy_svm_c0_1_pca")


# ===============================
# 2. ICA
# ===============================

#Conseguimos el conjunto de datos
accuracy_ica, model_ica = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, "ICA")

# Aplicamos algunos modelos de ejemplo
accuracy_mlp_50_ica = apply_mlp(X_train, X_test, y_train, y_test, MLP_50)
accuracy_knn_1_ica = apply_knn(X_train, X_test, y_train, y_test, KNN_1)
accuracy_svm_c0_1_ica = apply_svm(X_train, X_test, y_train, y_test, SVM_C0_1)

println("Precisión MLP (50) ICA: $accuracy_mlp_50_ica")
println("Precisión KNN (1) ICA: $accuracy_knn_1_ica")
println("Precisión SVM (C=0.1) ICA: $accuracy_svm_c0_1_ica")


# ===============================
# 3. LDA
# ===============================

#Conseguimos el conjunto de datos
accuracy_lda, model_lda = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, "LDA")

# Aplicamos algunos modelos de ejemplo
accuracy_mlp_50_lda = apply_mlp(X_train, X_test, y_train, y_test, MLP_50)
accuracy_knn_1_lda = apply_knn(X_train, X_test, y_train, y_test, KNN_1)
accuracy_svm_c0_1_lda = apply_svm(X_train, X_test, y_train, y_test, SVM_C0_1)

println("Precisión MLP (50) LDA: $accuracy_mlp_50_lda")
println("Precisión KNN (1) LDA: $accuracy_knn_1_lda")
println("Precisión SVM (C=0.1) LDA: $accuracy_svm_c0_1_lda")


# ===============================
# 4. ISOMAP
# ===============================

#Conseguimos el conjunto de datos
accuracy_isomap, model_isomap = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, "Isomap")

# Aplicamos algunos modelos de ejemplo
accuracy_mlp_50_isomap = apply_mlp(X_train, X_test, y_train, y_test, MLP_50)
accuracy_knn_1_isomap = apply_knn(X_train, X_test, y_train, y_test, KNN_1)
accuracy_svm_c0_1_isomap = apply_svm(X_train, X_test, y_train, y_test, SVM_C0_1)

println("Precisión MLP (50) Isomap: $accuracy_mlp_50_isomap")
println("Precisión KNN (1) Isomap: $accuracy_knn_1_isomap")
println("Precisión SVM (C=0.1) Isomap: $accuracy_svm_c0_1_isomap")


# ===============================
# 5. LLE
# ===============================

#Conseguimos el conjunto de datos
accuracy_lle, model_lle = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, "LLE")

# Aplicamos algunos modelos de ejemplo
accuracy_mlp_50_lle = apply_mlp(X_train, X_test, y_train, y_test, MLP_50)
accuracy_knn_1_lle = apply_knn(X_train, X_test, y_train, y_test, KNN_1)
accuracy_svm_c0_1_lle = apply_svm(X_train, X_test, y_train, y_test, SVM_C0_1)

println("Precisión MLP (50) LLE: $accuracy_mlp_50_lle")
println("Precisión KNN (1) LLE: $accuracy_knn_1_lle")
println("Precisión SVM (C=0.1) LLE: $accuracy_svm_c0_1_lle")

