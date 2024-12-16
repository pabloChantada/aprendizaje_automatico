using DataFrames
using MLJ
using Statistics
using StatsBase
using StatsModels
using MLJLinearModels  
using FeatureSelectors  
using GLM
using ScikitLearn
using Distributions
using Test

@sk_import feature_selection: mutual_info_classif
include("../modelos_basicos.jl")
include("../preparacion.jl")

function test_anova_filter()
    # Generar datos de ejemplo
    N = 120
    class_labels = ["X", "Y", "Z"]
    X1 = vcat(randn(40) .+ 2, randn(40) .+ 6, randn(40) .+ 10)  # Característica 1 con diferentes medias
    X2 = vcat(randn(40) .+ 3, randn(40) .+ 7, randn(40) .+ 12)  # Característica 2 con diferentes medias
    X3 = vcat(randn(40) .+ 5, randn(40) .+ 8, randn(40) .+ 11)  # Característica 3 con diferentes medias
    y = vcat(repeat(["X"], 40), repeat(["Y"], 40), repeat(["Z"], 40))  # Etiquetas de clase

    # Crear el DataFrame con las características y etiquetas
    data = DataFrame(Feature1 = X1, Feature2 = X2, Feature3 = X3, Class = y)
    
    # Aplicar el filtro ANOVA
    selected_features = anova_filter(data, :Class, alpha=0.05)
    
    # Mostrar las características seleccionadas
    println("Características seleccionadas para el modelo:", selected_features)
end

function test_mutual_information_filter_sklearn()
    # Generar datos de ejemplo
    N = 120
    class_labels = ["X", "Y", "Z"]
    X1 = vcat(randn(40) .+ 2, randn(40) .+ 6, randn(40) .+ 10)  # Característica 1 con diferentes medias
    X2 = vcat(randn(40) .+ 3, randn(40) .+ 7, randn(40) .+ 12)  # Característica 2 con diferentes medias
    X3 = randn(120)  # Característica irrelevante
    y = vcat(repeat(["X"], 40), repeat(["Y"], 40), repeat(["Z"], 40))  # Etiquetas de clase

    # Crear el DataFrame
    data = DataFrame(Feature1 = X1, Feature2 = X2, Feature3 = X3, Class = y)

    # Aplicar el filtro de Información Mutua
    selected_features = mutual_information_filter_sklearn(data, :Class, threshold=0.05)
    println("Características seleccionadas para el modelo:", selected_features)
end

function test_rfe_logistic_regression()
    # Crear un DataFrame de ejemplo
    data = DataFrame(
        Feature1 = randn(100),
        Feature2 = randn(100),
        Feature3 = randn(100),
        Feature4 = randn(100),
        Class = rand([0, 1], 100)  
    )
    
    selected_features = rfe_logistic_regression(data, :Class, threshold=0.5)
    
    println("Características seleccionadas para el modelo: ", selected_features)
end

# Ejecutar la prueba

@testset "Modelos Basicos" begin
    test_anova_filter()     
    test_mutual_information_filter_sklearn()   
    test_rfe_logistic_regression() 
end