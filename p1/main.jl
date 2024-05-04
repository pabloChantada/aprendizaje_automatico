# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------
include("functions.jl")
using XLSX
using DataFrames
using Dates
# usar seed 42

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------
# Read the file
xf = XLSX.readxlsx("Dry_Bean_Dataset.xlsx")
data = XLSX.gettable(xf["Dry_Beans_Dataset"]) |> DataFrame
# Update the column selection as needed
select!(data, Not([:AspectRation, :Eccentricity, :Extent, :Solidity, :roundness, :Compactness, :ShapeFactor1, :ShapeFactor2, :ShapeFactor3, :ShapeFactor4]))

inputs_df = data[2:end, 1:6]  # Assumes numeric data
targets_df = data[2:end, 7]  # Assuming the last column is the target
inputs = Matrix{Float32}(inputs_df)
targets = vec((targets_df))

# Normalization
max_vals, min_vals = calculateMinMaxNormalizationParameters(inputs)
mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)
normalize_data = normalizeMinMax!(inputs, (min_vals, max_vals))
# [29324.096 214.2897 85.69418 44.97009 0.24667846 0.09200176 29774.916 59.17712 0.049086366 0.004660379 0.05951989 0.061713465 0.0011279982 0.00059587485 0.09899615 0.004366458]

# ------------------------------------------------------------------
# Generate the results using parallelization
# ------------------------------------------------------------------
# Cross-validation setup
num_folds = 5
crossValidationIndices = crossvalidation(targets, num_folds)

# Define different model configurations
model_configurations = Dict(
    # Generar yo las topologias
    :ANN => [Dict("topology" => [rand(5:15), rand(3:10)]) for _ in 1:8],  # 8 random configurations between 1 and 2 layers
    :SVC => [Dict("kernel" => k, "C" => c,  "gamma" => "auto", "coef0" => 0.5) for k in ["linear", "rbf", "poly", "sigmoid"] for c in [0.1, 1]],  # 8 configurations
    :DecisionTreeClassifier => [Dict("max_depth" => d) for d in 4:12],  # 8 depths
    :KNeighborsClassifier => [Dict("n_neighbors" => k) for k in 4:12]  # 8 different k values
)

start = now()
# Parallel processing setup
model_keys = collect(keys(model_configurations))
Threads.@threads for modeltype in model_keys
    configs = model_configurations[modeltype]
    for config in configs
        println("Current Model: $modeltype, Params: $config")
        result = modelCrossValidation(modeltype, config, inputs, targets, crossValidationIndices)
        println("Result: $result\n")
    end
end

end_time = now()
total_duration = (end_time - start) / Millisecond(60000)  # Convert to minutes

println("Total Runtime: $total_duration minutes")