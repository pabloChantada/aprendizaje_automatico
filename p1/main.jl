# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------
include("functions.jl")
using XLSX
using DataFrames
using Dates
using Plots
using StatsPlots
using Random
using StatsBase

Random.seed!(42)

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------
# Read the file and get data
xf = XLSX.readxlsx("Dry_Bean_Dataset.xlsx")
df = XLSX.gettable(xf["Dry_Beans_Dataset"]) |> DataFrame
# Update the column selection as needed
select!(df, Not([:AspectRation, :Eccentricity, :Extent, :Solidity, :roundness, :Compactness, :ShapeFactor1, :ShapeFactor2, :ShapeFactor3, :ShapeFactor4]))
# Keep column names before conversion
inputs = df[2:end, 1:6]
targets = df[2:end, 7]
input_names = names(inputs)
# Convert DataFrame to Matrix for numerical operations
inputs = Matrix{Float32}(inputs)
targets = vec(String.(targets))

class_counts = countmap(targets)
println("Counts for each class: ", class_counts)

function manual_undersample(features, targets)
    counts = countmap(targets)
    min_count = minimum(values(counts))
    
    indices = Int[]
    for class in keys(counts)
        class_indices = findall(t -> t == class, targets)
        sampled_indices = sample(class_indices, min_count, replace=false)
        append!(indices, sampled_indices)
    end
    
    return features[indices, :], targets[indices]
end

inputs, targets = manual_undersample(inputs, targets)
println("Counts after manual undersampling: ", countmap(targets))
inputs
# Create histograms for each column
p1 = plot(layout = (6, 1))
for (i, col_name) in enumerate(input_names)
    histogram!(p1, inputs[:, i], bins=30, title="Histogram of $(col_name)", label=col_name)
end
# Create boxplots for each column
p2 = plot(layout = (6, 1))
for (i, col_name) in enumerate(input_names)
    boxplot!(p2, inputs[:, i], title="Boxplot of $(col_name)", label=col_name)
end
# Display plots
plot(p1, p2, layout = (2, 1), size = (600, 1200))

# Normalization
min_vals, max_vals = calculateMinMaxNormalizationParameters(inputs)
mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)
normalize_data = normalizeMinMax!(inputs, (min_vals, max_vals))

# Correlation matrix of normalized data
cor_matrix = cor(inputs)
# Plot the correlation matrix
heatmap(cor_matrix, title = "Feature Correlation Matrix",
        xticks = (1:length(input_names), input_names),
        yticks = (1:length(input_names), input_names),
        color = :coolwarm)
# ------------------------------------------------------------------
# Model Evaluation Setup
# ------------------------------------------------------------------
num_folds = 5
crossValidationIndices = crossvalidation(targets, num_folds)

model_configurations = Dict(
    # Generar yo las topologias
    # :ANN => [Dict("topology" => [rand(5:15), rand(3:10)]) for _ in 1:8],  # 8 random configurations between 1 and 2 layers
    :SVC => [Dict("kernel" => k, "C" => c,  "gamma" => "auto", "coef0" => 0.5, "degree" => 3) for k in ["linear", "rbf", "poly", "sigmoid"] for c in [2, 3]],  # 8 configurations
    :DecisionTreeClassifier => [Dict("max_depth" => d) for d in 4:12],  # 8 depths
    :KNeighborsClassifier => [Dict("n_neighbors" => k) for k in 4:12]  # 8 different k values
)

# ------------------------------------------------------------------
# Model Training and Evaluation
# ------------------------------------------------------------------
all_results = Dict()

start = now()
model_configuration_array = collect(pairs(model_configurations))
Threads.@threads for (modeltype, configs) in model_configuration_array
    model_results = []
    for config in configs
        result = modelCrossValidation(modeltype, config, inputs, targets, crossValidationIndices)
        push!(model_results, (config, result))
        println("Current Model: $modeltype, Params: $config, Result: $result\n")
    end
    all_results[modeltype] = model_results
end
end_time = now()
total_duration = (end_time - start) / Millisecond(60000)  # Convert to minutes
println("Total Runtime: $total_duration minutes")

# ------------------------------------------------------------------
# Finding Best Configurations and Plotting
# ------------------------------------------------------------------
best_configs = Dict()
for (modeltype, results) in all_results
    best_result = sort(results, by=x -> x[2][1][1], rev=true)[1]  # Sort by mean accuracy, assuming result structure as before
    best_configs[modeltype] = best_result
end

# Function to format configuration details for readability
function format_config_label(config)
    return join(["$key=$(isa(value, Number) ? round(value, digits=3) : value)" for (key, value) in config], ", ")
end


# Extracting and formatting best configuration results
mean_accuracies = [round(res[2][1][1], digits=5) for res in values(best_configs)]  # Round the accuracies for higher precision
formatted_labels = [string(key, " - ", format_config_label(val[1])) for (key, val) in best_configs]

# Create the plot
bar_plot = bar(mean_accuracies, label=formatted_labels, title="Best Model Configuration Comparison",
           xlabel="Models", ylabel="Mean Accuracy", size=(800, 600))