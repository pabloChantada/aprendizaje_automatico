using DataFrames
using CSV
using Random
using DataFrames, Statistics, StatsBase
Random.seed!(172)

# =======================================
# 1. Funcion de carga directa del DataSet
# =======================================

function loadDataset()
    PATH = "tercero/proyecto_investigacion/Datos_Práctica_Evaluación_1.csv"
    return CSV.read(PATH,DataFrame)
end

# ============================
# 2. Analisis leve del DataSet
# ============================

function csvDescription(csv)
    println("Variables (No subject; No Activity): ", size(csv, 2) - 2)    # Número de columnas
    println("Attributes: ", size(csv, 1))   # Número de filas
    println("Individuals: ", length(unique(csv[:, 1])))     # Valores únicos en la primera columna
    println("Classes: ", length(unique(csv[:, end])), " | ", unique(csv[:, end]))   # Última columna
end

csvDescription(loadDataset())
# ==========================
# 3. Preparacion del DataSet
# ==========================

# Devuelve el porcetaje de nulos en el dataset
function getNullValues(data::AbstractDataFrame, info::Bool)
    if info == true
        println("Columnas detectadas: ", names(data)) # Confirmar nombres de columnas
        println("Número de filas: ", nrow(data))      # Confirmar tamaño del DataFrame
    end
    total_nulls = 0  # Inicializar contador total de nulos
    
    for col in names(data)
        # println("Procesando columna: ", col)     # Confirmar iteración columna por columna
        
        # Calcular valores faltantes por columna
        n_nulls = sum(ismissing.(data[:, col]))
        pct_nulls = (n_nulls / nrow(data)) * 100
        total_nulls += n_nulls  # Acumular valores nulos
        
        # Mostrar resultados por columna
        if (pct_nulls > 0 && info == true)
            println("$col: $(round(pct_nulls, digits=4))%")
        end
    end
    
    # Calcular porcentaje total de nulos
    total_elements = nrow(data) * ncol(data)
    total_pct_nulls = (total_nulls / total_elements) * 100
    
    # Mostrar porcentaje total de nulos
    println("Porcentaje total de valores nulos en el DataFrame: $(round(total_pct_nulls, digits=4))%")
end

# Rellena los nulos con la media
function preprocessData(data::DataFrame)
    # Separar columnas numéricas y categóricas
    num_cols = [col for col in names(data) if eltype(data[!, col]) <: Real]
    cat_cols = setdiff(names(data), num_cols)
    
    # Imputar valores faltantes
    for col in num_cols
        data[!, col] = coalesce.(data[!, col], median(skipmissing(data[!, col])))
    end

    for col in cat_cols
        data[!, col] = coalesce.(data[!, col], mode(skipmissing(data[!, col])))
    end
    
    return data  # Devuelve el DataFrame procesado
end

# =================================
# HOLD-OUT dejando un 10% fuera
# =================================

# TODO: enseñar cuales se dejan fuera
function HoldOut(data::DataFrame)
    # Assuming your data is in a DataFrame called 'data'
    n_individuals = length(unique(data.subject))  # Adjust 'ID' to your individual identifier column
    test_size = floor(Int, 0.1 * n_individuals)

    # Get unique individuals and shuffle
    individuals = shuffle(unique(data.subject))
    test_individuals = individuals[1:test_size]

    # Split data
    # Si se quiere se puede eliminar Ref(Set()) al ser solo para optimizacion 
    test = data[in.(data.subject, Ref(Set(test_individuals))), :]
    train = data[.!in.(data.subject, Ref(Set(test_individuals))), :]

    return train, test
end

# ==================================
# 5. 5-CV con los datos del HOLD-OUT
# ==================================

function stratified_group_k_fold(data::DataFrame, k::Int=5)
    # Obtener individuos únicos
    subjects = unique(data.subject)
    n_subjects = length(subjects)
    
    # Mezclar aleatoriamente los sujetos
    shuffled_subjects = shuffle(subjects)
    
    # Calcular tamaño aproximado de cada fold
    fold_size = ceil(Int, n_subjects/k)
    
    # Crear los folds
    folds = Vector{Vector{Int}}(undef, k)
    for i in 1:k
        # Obtener los sujetos para este fold
        start_idx = (i-1)*fold_size + 1
        end_idx = min(i*fold_size, n_subjects)
        fold_subjects = shuffled_subjects[start_idx:end_idx]
        
        # Obtener todos los índices de las instancias de estos sujetos
        fold_indices = findall(x -> x in fold_subjects, data.subject)
        folds[i] = fold_indices
    end
    
    return folds
end

# =============================
# 6. MinMax sobre los conjuntos
# =============================

function normalizeMinMax!(dataset::AbstractArray{<:Real,2}, normalizationParameters::NTuple{2,AbstractArray{<:Real,2}})
    # Normalizar entre el max y el min
    min_values, max_values = normalizationParameters[1], normalizationParameters[2]
    dataset .-= min_values
    range_values = max_values .- min_values
    dataset ./= (range_values)
    dataset[:, vec(min_values .== max_values)] .= 0
    return dataset
end;

function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2}) # vectores de ints y floats
    min_col = minimum(dataset, dims=1)
    max_col = maximum(dataset, dims=1)
    return (min_col, max_col)
end;


function applyMinMax(dataset)
    min_col, max_col = calculateMinMaxNormalizationParameters(dataset)    
    normalizeMinMax(dataset, (min_col, max_col))
end;


# =========================
# =========================
# PIPELINE PARA USAR DE UNA 
# =========================
# =========================

function preparation_pipeline(info::Bool)
    # 1. Cargar datos y realizar descripción básica
    data = loadDataset()
    if info == true
        csvDescription(data)
    end
    
    # 2. Calcular porcentaje de nulos
    if info == true
        getNullValues(data, true) 
    else 
        getNullValues(data, false) 
    end 
    
    # 3. Preparar datos (rellenar nulos)
    clean_data = preprocessData(data)
    
    # 4. Realizar holdout del 10%
    train_data, test_data = HoldOut(clean_data)
    
    # 5. Preparar cross-validation
    cv_indices = stratified_group_k_fold(train_data, 5)
    
    # 6. Normalizar datos para cada fold
    normalized_folds = []
    for (i, fold_indices) in enumerate(cv_indices)
        # Separar índices de train y test para este fold
        train_idx = vcat([idx for (j,idx) in enumerate(cv_indices) if j != i]...)
        test_idx = fold_indices
        
        # Obtener datos de train y test para este fold
        fold_train = train_data[train_idx, :]
        fold_test = train_data[test_idx, :]
        
        # Calcular parámetros de normalización usando solo los datos de train
        train_matrix = Matrix(fold_train[:, Not([:subject, :Activity])])
        norm_params = calculateMinMaxNormalizationParameters(train_matrix)
        
        # Normalizar tanto train como test con los mismos parámetros
        train_normalized = normalizeMinMax!(copy(train_matrix), norm_params)
        test_matrix = Matrix(fold_test[:, Not([:subject, :Activity])])
        test_normalized = normalizeMinMax!(copy(test_matrix), norm_params)
        
        # Crear DataFrames normalizados
        train_df = DataFrame(train_normalized, names(fold_train[:, Not([:subject, :Activity])]))
        train_df[!, :subject] = fold_train.subject
        train_df[!, :Activity] = fold_train.Activity
        
        test_df = DataFrame(test_normalized, names(fold_test[:, Not([:subject, :Activity])]))
        test_df[!, :subject] = fold_test.subject
        test_df[!, :Activity] = fold_test.Activity
        
        # Guardar los datos normalizados de este fold
        push!(normalized_folds, (train=train_df, test=test_df))
    end
    
    if info 
        return Dict(
            "data_original" => data,
            "data_cleaned" => clean_data,
            "train_data" => train_data,
            "test_data" => test_data,
            "cv_indices" => cv_indices,
            "normalized_folds" => normalized_folds
        )
    else
        return Dict(
            "normalized_folds" => normalized_folds,  # Para validación cruzada
            "test_data" => test_data    # Para evaluación final
        )
    end
end