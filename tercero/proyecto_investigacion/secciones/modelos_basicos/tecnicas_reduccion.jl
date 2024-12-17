# 8. Sobre el filtrado, aplicar alguna de las siguientes técnicas de reducción de la dimensionalidad:
# NO aplicar nada, PCA, LDA, ICA, Isomap, LLE
# Para cada una de estas técnicas represente mediante las dos primeras características conservadas el conjunto de datos 

include("preparacion.jl")
include("modelos_basicos.jl")
PATH = "Datos_Práctica_Evaluación_1.csv"

using DataFrames
using ScikitLearn
using Distributions
using ManifoldLearning
using Plots
using Random
using MLJ
using ScikitLearn
using MLJScikitLearnInterface 
import DataFrames: DataFrame
using ScikitLearn.Pipelines
using ScikitLearn.CrossValidation: train_test_split
using CSV

using ScikitLearn: fit!
using ScikitLearn: predict
using ScikitLearn: transform

@sk_import manifold: LocallyLinearEmbedding
@sk_import preprocessing: StandardScaler
@sk_import discriminant_analysis: LinearDiscriminantAnalysis

@sk_import decomposition: PCA
@sk_import decomposition: FastICA
@sk_import manifold: Isomap
@sk_import manifold: LocallyLinearEmbedding
@sk_import preprocessing: MinMaxScaler
@sk_import linear_model: LogisticRegression
@sk_import discriminant_analysis: LinearDiscriminantAnalysis

data2 = CSV.read(PATH, DataFrame)
println(first(data2, 1))  # Muestra las primeras 5 filas

# Preprocesar y filtrar características con ANOVA (ejemplo)
target_column = :Activity
data2_clean = preprocessData(data2)
selected_features = anova_filter(data2, target_column, alpha=0.05)
println("Características seleccionadas: ", selected_features)

# Crear `X_filtered` con las características seleccionadas
X_filtered = select(data2, selected_features)
y_filtered = data2[:, target_column]

# Codificar etiquetas
label_encoder = LabelEncoder()
y_filtered_encoded = fit_transform!(label_encoder, y_filtered)

# Dividir en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(Matrix(X_filtered), y_filtered_encoded, test_size=0.2, random_state=42)

# Función para aplicar una técnica de reducción dimensional y evaluar el modelo
function apply_dimensionality_reduction(X_train, X_test, y_train, y_test, technique::String)
    # Técnicas de reducción de dimensionalidad
    techniques = Dict(
        "no_reduction" => SelectKBest(score_func=f_classif, k=2),
        "PCA" => PCA(n_components=2),
        "ICA" => FastICA(n_components=2),
        "LDA" => LinearDiscriminantAnalysis(n_components=2),
        "Isomap" => Isomap(n_components=2),
        "LLE" => LocallyLinearEmbedding(n_components=2)
    )
    
    # Seleccionar la técnica
    reducer = techniques[technique]
    
    # Crear pipeline: estandarización + reducción dimensional + regresión logística
    model = Pipeline([
        ("scaler", MinMaxScaler()),  # Estandarización de los datos
        ("reducer", reducer),        # Reducción de dimensionalidad
        ("classifier", LogisticRegression())  # Modelo de clasificación
    ])

    # Ajustar el modelo (entrenar el modelo)
    fit!(model, X_train, y_train)
    
    # Predecir en el conjunto de prueba
    y_pred = predict(model, X_test)
    
    # Calcular la precisión del modelo
    accuracy = sum(y_pred .== y_test) / length(y_test)
    
    println("Técnica: $technique, Precisión: $accuracy")
    return accuracy, model
end

function plot_transformed_data(technique::String, X, y)
    techniques = Dict(
        "no_reduction" => SelectKBest(score_func=f_classif, k=2),
        "PCA" => PCA(n_components=2),
        "ICA" => FastICA(n_components=2),
        "LDA" => LinearDiscriminantAnalysis(n_components=2),
        "Isomap" => Isomap(n_components=2),
        "LLE" => LocallyLinearEmbedding(n_components=2)
    )
    reducer = techniques[technique]
    # Estandarizar los datos antes de aplicar la reducción de dimensionalidad
    scaler = MinMaxScaler()
    X_scaled = fit_transform!(scaler, X) 
    # Reducir la dimensionalidad
    X_reduced = fit_transform!(reducer, X_scaled, y)
    # Crear el gráfico de dispersión de las dos primeras características conservadas
    scatter(X_reduced[:, 1], X_reduced[:, 2], group=y, legend=:topright, title=selected_technique,
            xlabel="Componente 1", ylabel="Componente 2", markersize=5)
end

selected_technique = "PCA"  
accuracy, model = apply_dimensionality_reduction(X_train, X_test, y_train, y_test, selected_technique)
plot_transformed_data(selected_technique, Matrix(X_filtered), y_filtered, selected)
