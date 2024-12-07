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
function preprocessData(arguments)

    # 1. Separate numeric and categorical columns
    num_cols = names(select(data, Real))
    cat_cols = setdiff(names(data), num_cols)

    # 2. Handle missing values
    # For numeric: impute with median
    for col in num_cols
       data[!, col] = coalesce.(data[:, col], median(skipmissing(data[:, col])))
    end

    # For categorical: impute with mode
    for col in cat_cols
       data[!, col] = coalesce.(data[:, col], mode(skipmissing(data[:, col])))
    end

    # 3. Encode categorical variables
    encoder = CategoricalEncoder()
    data_encoded = transform(fit!(machine(encoder, data)), data)

    # 4. Scale numeric features
    scaler = StandardScaler()
    data_scaled = transform(fit!(machine(scaler, data_encoded)), data_encoded) 
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


