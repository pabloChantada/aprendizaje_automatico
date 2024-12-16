using DataFrames
using CSV
using Random
using DataFrames, Statistics, StatsBase, MLJ
Random.seed!(172)

# Load the data 
PATH = "Datos_Práctica_Evaluación_1.csv"
data = CSV.read(PATH,DataFrame)

function csvDescription(csv)
    println("Variables: ", size(csv, 2))    # Número de columnas
    println("Attributes: ", size(csv, 1))   # Número de filas
    println("Individuals: ", length(unique(csv[:, 1])))     # Valores únicos en la primera columna
    println("Classes: ", length(unique(csv[:, end])), " | ", unique(csv[:, end]))   # Última columna
end

csvDescription(data)
describe(data)


# Check percentage of NULL values  in the sistem
function getNullValues(data::AbstractDataFrame)
    println("Columnas detectadas: ", names(data)) # Confirmar nombres de columnas
    println("Número de filas: ", nrow(data))      # Confirmar tamaño del DataFrame
    
    total_nulls = 0  # Inicializar contador total de nulos
    
    for col in names(data)
        println("Procesando columna: ", col)     # Confirmar iteración columna por columna
        
        # Calcular valores faltantes por columna
        n_nulls = sum(ismissing.(data[:, col]))
        pct_nulls = (n_nulls / nrow(data)) * 100
        total_nulls += n_nulls  # Acumular valores nulos
        
        # Mostrar resultados por columna
        println("$col: $(round(pct_nulls, digits=4))%")
    end
    
    # Calcular porcentaje total de nulos
    total_elements = nrow(data) * ncol(data)
    total_pct_nulls = (total_nulls / total_elements) * 100
    
    # Mostrar porcentaje total de nulos
    println("Porcentaje total de valores nulos en el DataFrame: $(round(total_pct_nulls, digits=4))%")
end

getNullValues(data)


# Fill the empty data or anything else
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




# HOLD-OUT del 10% -> individual-wise | Seed 172
function holdOut(N::Int, P::Real)
    # N -> numero de patrones
    # P -> valor entre 0 y 1, indica el porcentaje para el test
    # numero de patrones para el test
    @assert P < 1 "Valores de test fuera de rango"
    test_size = Int(floor(N * P))
    # permutamos los datos
    index_perm = randperm(N)
    index_test = index_perm[1:test_size]
    index_train = index_perm[test_size+1:end]
    @assert length(index_test) + length(index_train) == N
    return (index_train, index_test)
end;


function applyHoldOut(arguments)
    # Assuming your data is in a DataFrame called 'data'
    n_individuals = length(unique(data.ID))  # Adjust 'ID' to your individual identifier column
    test_size = floor(Int, 0.1 * n_individuals)

    # Get unique individuals and shuffle
    individuals = shuffle(unique(data.ID))
    test_individuals = individuals[1:test_size]

    # Split data
    # Si se quiere se puede eliminar Ref(Set()) al ser solo para optimizacion 
    test = data[in.(data.ID, Ref(Set(test_individuals))), :]
    train = data[.!in.(data.ID, Ref(Set(test_individuals))), :]
end


# 5-CROSS-VALIDATION -> SEED 172
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;


# Vector
function crossvalidation(targets::AbstractArray{Bool,1}, k::Int64)
    indices = Array{Int64,1}(undef, length(targets));
    indices[  targets] = crossvalidation(sum(  targets), k);
    indices[.!targets] = crossvalidation(sum(.!targets), k);
    return indices;
end;


# Matriz
function crossvalidation(targets::AbstractArray{Bool,2}, k::Int64)
    numClasses = size(targets,2);
    @assert(numClasses!=2);
    if numClasses==1
        return crossvalidation(vec(targets), k);
    end;
    indices = Array{Int64,1}(undef, size(targets,1));
    for numClass in 1:numClasses
        indices[targets[:,numClass]] = crossvalidation(sum(targets[:,numClass]), k);
    end;
    return indices;
end;


function apllyCrossValidation(data)
  return crossvalidation(data, 5)  
end


# MinMax normalization
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


using DataFrames
using Random
using Statistics


# Función ANOVA para filtrar características
function anova_filter(data::DataFrame, target_col::Symbol; alpha::Float64=0.05)
    # Separar las variables predictoras (X) y la variable objetivo (Y)
    X = select(data, Not(target_col))   # Excluimos la columna de la clase
    y = data[!, target_col]             # Columna de la clase

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
    f_dist = FDist(df_between, df_within)  # Usar `FDist`
    p_value = 1 - cdf(f_dist, f_statistic)  # Usar la función CDF para obtener el p-valor

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
test_anova_filter()



