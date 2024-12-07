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

@sk_import feature_selection: mutual_info_classif

include("mine.jl")
PATH = "Datos_Práctica_Evaluación_1.csv"
data2 = CSV.read(PATH,DataFrame)

#FILTRADO

# 1. Sin reducción de dimensionalidad
function no_reduction(data::DataFrame)
    println("No se aplica reducción de dimensionalidad")
    return data
end

# 2. Filtrado ANOVA
# Función ANOVA para filtrar características
function anova_filter(data::DataFrame, target_col::Symbol; alpha::Float64=0.05)
    
    X = select(data, Not(target_col))   
    y = data[!, target_col]             


    p_values = []


    for col in names(X)
   
        grouped_data = groupby(data, target_col)
        groups = [df[!, col] for df in grouped_data]  

        f_statistic, p_value = oneway_anova(groups)

        # Guardamos el p-valor en la lista
        push!(p_values, (col, p_value))
    end

    # Filtramos las características basándonos en el p-valor (usando el umbral alpha)
    selected_features = [col for (col, p_value) in p_values if p_value < alpha]

    # Mostrar resultados
    println("Características seleccionadas basadas en ANOVA:")
    println(selected_features)

    return selected_features
end

# Función que realiza ANOVA (de un solo factor)
function oneway_anova(groups::Vector)
    # Usamos la función anova de Julia, calculamos la estadística F y el p-valor
    n_groups = length(groups)
    means = map(mean, groups)
    grand_mean = mean(vcat(groups...))
    
    # Cálculo de la variabilidad entre los grupos
    ss_between = sum(length(g) * (mean(g) - grand_mean)^2 for g in groups)
    df_between = n_groups - 1
    
    # Cálculo de la variabilidad dentro de los grupos
    ss_within = sum(sum((x .- mean(g)).^2) for (x, g) in zip(groups, groups))
    df_within = sum(length(g) - 1 for g in groups)
    
    # Cálculo de la estadística F
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_statistic = ms_between / ms_within
    
    # Calcular el p-valor usando la distribución F de `Distributions`
    f_dist = FDist(df_between, df_within)  
    p_value = 1 - cdf(f_dist, f_statistic)  

    return f_statistic, p_value
end

# Función de test
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

# Llamar a la función de test
test_anova_filter()     # Este funciona



# 3. Filtrado Mutual Information

function mutual_information_filter_sklearn(data::DataFrame, target_col::Symbol; threshold::Float64=0.05)
    X = Matrix(select(data, Not(target_col)))  
    y = data[!, target_col]                    
    
    # Calcular la información mutua
    mi_scores = mutual_info_classif(X, y)
    
    # Seleccionar características con información mutua por encima del umbral
    selected_features = names(select(data, Not(target_col)))[mi_scores .> threshold]
    
    println("Puntajes de Información Mutua:", mi_scores)
    println("Características seleccionadas:", selected_features)
    return selected_features
end


# Función de prueba
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

    println("Primeras filas del DataFrame:")
    println(first(data, 5))

    # Aplicar el filtro de Información Mutua
    selected_features = mutual_information_filter_sklearn(data, :Class, threshold=0.05)
    println("Características seleccionadas para el modelo:", selected_features)
end

# Llamar a la función de prueba
test_mutual_information_filter_sklearn()    #Este funciona



# 4. RFE con Logistic Regression

function rfe_logistic_regression(data::DataFrame, target_col::Symbol; threshold::Float64=0.5)
    X = select(data, Not(target_col))  
    y = data[:, target_col]  
    
    # Convertir a matrices adecuadas para GLM
    X_matrix = Matrix(X)
    
    # Número de características en el dataset
    num_features = size(X_matrix, 2)
    
    selected_features = names(X)
    
    while size(X_matrix, 2) > 1  
        # Ajustar un modelo de regresión logística
        model = glm(X_matrix, y, Binomial())
        
        # Obtener los coeficientes del modelo (sin intercepto)
        coeffs = coef(model)[2:end]  
        
        # Calcular la importancia (absoluto de los coeficientes)
        importance = abs.(coeffs)
        
        # Ordenar las características por importancia (de mayor a menor)
        sorted_indices = sortperm(importance, rev=true)
        
        # Calcular cuántas características eliminar
        num_to_remove = floor(Int, size(X_matrix, 2) * threshold)
        
        features_to_remove = sorted_indices[end-num_to_remove+1:end]
        
        # Eliminar las características menos importantes
        X_matrix = hcat([X_matrix[:, i] for i in setdiff(1:size(X_matrix, 2), features_to_remove)]...)
        
        # Actualizar las características seleccionadas
        selected_features = selected_features[setdiff(1:length(selected_features), features_to_remove)]
        
        println("Características seleccionadas en esta iteración: ", selected_features)
    end
    
    return selected_features
end

# Ejemplo de uso de la función con un dataset de prueba
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
test_rfe_logistic_regression() # Esta funciona

