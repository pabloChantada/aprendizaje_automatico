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

# FILTRADO
# ====================================
# 1. Sin reducción de dimensionalidad
# ====================================

function no_reduction(data::DataFrame)
    println("No se aplica reducción de dimensionalidad")
    return data
end

# ==================
# 2. Filtrado ANOVA
# ==================

# Función ANOVA para filtrar características
function anova_filter(data::DataFrame, target_col::Symbol; alpha::Float64=0.05, info::Bool=false)
    # Separar las variables predictoras (X) y la variable objetivo (Y)
    X = select(data, Not([:subject, :Activity]))
    X = X[:, names(X, eltype(Float64))]  # Filtra solo las columnas numéricas

    # Crear una lista para almacenar los p-valores de cada característica
    p_values = []

    # Realizamos ANOVA para cada característica
    for col in names(X)
        # Agrupar los valores de la característica por cada clase en la variable objetivo
        grouped_data = groupby(data, target_col)
        groups = [df[!, col] for df in grouped_data]  # Extraer las columnas para ANOVA

        # Realizamos el ANOVA y calculamos el p-valor
        f_statistic, p_value = oneway_anova(groups)

        # Guardamos el p-valor en la lista
        push!(p_values, (col, p_value))
    end

    # Filtramos las características basándonos en el p-valor (usamos el umbral alpha)
    selected_features = [col for (col, p_value) in p_values if p_value < alpha]

    # Mostrar resultados
    if info
        println("Características totales: ", length(names(X)))
        println("Características seleccionadas basadas en ANOVA: ", length(selected_features))
    end
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
    f_dist = FDist(df_between, df_within)  # Usar `FDist`
    p_value = 1 - cdf(f_dist, f_statistic)  # Usar la función CDF para obtener el p-valor

    return f_statistic, p_value
end

# ==============================
# 3. Filtrado Mutual Information
# ==============================

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

# ===============================
# 4. RFE con Logistic Regression
# ===============================

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

