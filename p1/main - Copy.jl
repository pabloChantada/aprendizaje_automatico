# ------------------------------------------------------------------
# Dependencias
# ------------------------------------------------------------------
include("functions.jl")
using XLSX
using DataFrames

# ------------------------------------------------------------------
# Procesado de los datos
# ------------------------------------------------------------------
# Leer el archivo
xf = XLSX.readxlsx("Dry_Bean_Dataset.xlsx")
data = XLSX.gettable(xf["Dry_Beans_Dataset"]) |> DataFrame
select!(data, Not([:AspectRation, :Eccentricity, :Extent, :Solidity, :roundness, :Compactness, :ShapeFactor1, :ShapeFactor2, :ShapeFactor3, :ShapeFactor4]))  # Update the column names as needed

# Assume inputs_df and targets_df have been defined as shown
inputs_df = data[2:end, 1:6]  # Data is assumed to be numeric
targets_df = data[2:end, 7]  # Assuming the last column is the target
# Convert DataFrame to Matrix{Float32}
inputs = Matrix{Float32}(inputs_df)
targets = vec((targets_df))

# Normalizacion
max_vals, min_vals = calculateMinMaxNormalizationParameters(inputs)
mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)
normalize_data = normalizeMinMax!(inputs, (min_vals, max_vals))
# [29324.096 214.2897 85.69418 44.97009 0.24667846 0.09200176 29774.916 59.17712 0.049086366 0.004660379 0.05951989 0.061713465 0.0011279982 0.00059587485 0.09899615 0.004366458]

# ------------------------------------------------------------------
# Genrate the results
# ------------------------------------------------------------------

# Modelos que usamos
modeltype_array = [:ANN, :SVC, :DecisionTreeClassifier, :KNeighborsClassifier]
modelHyperparameters = Dict(
    :ANN => Dict("topology" => [10, 5, 3]),
    # Usar todos los kernerl
    :SVC => Dict("kernel" => "linear", "C" => 1),
    :DecisionTreeClassifier => Dict("max_depth" => 3),
    :KNeighborsClassifier => Dict("n_neighbors" => 5)
)
num_folds = 2
crossValidationIndices = crossvalidation(targets, num_folds)

# Run models with cross-validation
for (i, modeltype) in enumerate(modeltype_array)
    hyperparams = modelHyperparameters[modeltype]
    result = modelCrossValidation(modeltype, hyperparams, inputs, targets, crossValidationIndices)
    println(result)
end