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
using Statistics
using CSV
Random.seed!(42)

# ------------------------------------------------------------------
# Data Processing
# ------------------------------------------------------------------
# Leer el archivo y convertilo a DF para eliminar filas
xf = XLSX.readxlsx("Dry_Bean_Dataset.xlsx")
df = XLSX.gettable(xf["Dry_Beans_Dataset"]) |> DataFrame
# Eliminamos las que tengan una std < 1
select!(df, Not([:AspectRation, :Eccentricity, :Extent, :Solidity, :roundness, :Compactness, :ShapeFactor1, :ShapeFactor2, :ShapeFactor3, :ShapeFactor4]))
inputs = df[2:end, 1:6]
targets = df[2:end, 7]
input_names = names(inputs)
# Convertir a los valores que usan las funciones
inputs = Matrix{Float32}(inputs)
targets = vec(String.(targets))

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
colors = [:lightblue, :red, :green, :yellow, :orange, :purple, :cyan]
# Generamos el plot de la distribucion base de las variables
class_counts = countmap(targets)
class_labels = collect(keys(class_counts))
class_instances = collect(values(class_counts))
original_distribution = bar(class_labels, class_instances, color=colors,
                              legend=false, ylabel="Number of Instances",
                              xlabel="Class", title="Class Distribution in Dataset")

# Balanceamos las variables (usamos undersample al haber una gran diferencia entre los valores)
inputs, targets = manual_undersample(inputs, targets)  # Valor minimo = 522
class_counts = countmap(targets)
class_labels = collect(keys(class_counts))
class_instances = collect(values(class_counts))
balance_distribution = bar(class_labels, class_instances, color=colors,
                              legend=false, ylabel="Number of Instances",
                              xlabel="Class", title="Class Distribution in Dataset")

# Histograma de cada variable
histograms = [histogram(inputs[:, i], bins=30, title="Histogram of $(input_names[i])", label=input_names[i], legend=:outerright) for i in 1:(length(input_names))]
# Boxplot de cada variable
boxplots = [boxplot([input_names[i]], inputs[:, i], title="Boxplot of $(input_names[i])", label=input_names[i], legend=:outerright) for i in 1:(length(input_names))]
# Matriz de correlacion
cor_matrix = cor(inputs)
# Heatmap
hm = heatmap(cor_matrix, title = "Feature Correlation Matrix",
        xticks = (1:length(input_names), input_names),
        yticks = (1:length(input_names), input_names),
        color = :coolwarm,
        size=(1000,1000))


# ------------------------------------------------------------------
# Nomralization
# ------------------------------------------------------------------
min_vals, max_vals = calculateMinMaxNormalizationParameters(inputs)
# mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)
normalize_data = normalizeMinMax!(inputs, (min_vals, max_vals))

# ------------------------------------------------------------------
# Model Setup
# ------------------------------------------------------------------
num_folds = 5
crossValidationIndices = crossvalidation(targets, num_folds)
topologies = [[20,10],
            [10, 10],
            [10, 15],
            [30, 20],
            [15, 15],
            [50, 25],
            [40, 20],
            [25, 15]]
model_configurations = Dict(
    :ANN => [Dict("topology" => t) for t in topologies],  # 8 configuraciones
    :SVC => [Dict("kernel" => k, "C" => c,  "gamma" => "auto", "coef0" => 0.5, "degree" => 3) for k in ["linear", "rbf", "poly", "sigmoid"] for c in [2, 3]],  # 8 configuraciones
    :DecisionTreeClassifier => [Dict("max_depth" => d) for d in 4:11],  # 8 profundidades
    :KNeighborsClassifier => [Dict("n_neighbors" => k) for k in 4:11]  # 8 valores de vecinos
)

# ------------------------------------------------------------------
# Model Training and Evaluation
# ------------------------------------------------------------------
all_results = Dict()
model_configuration_array = collect(pairs(model_configurations))
# Utilizamos threads para acelerar levemente la ejecucion
Threads.@threads for (modeltype, configs) in model_configuration_array
    model_results = []
    for config in configs
        println("Current Model: $modeltype, Params: $config")
        result = modelCrossValidation(modeltype, config, inputs, targets, crossValidationIndices)
        println("Result: $result")
        push!(model_results, (config, result))
    end
    all_results[modeltype] = model_results
end

# Mostrar los resultados en un df
column_names = ["Model", "Params", "Mean Accuracy", "Std Accuracy"]
df_result = DataFrame(Model=String[], Params=Any[], Mean_Accuracy=Float64[], Std_Accuracy=Float64[])
for (modeltype, results) in all_results
    for result in results
        config = result[1]
        mean_acc = result[2][1][1]
        std_acc = result[2][1][2]
        push!(df_result, (String(modeltype), config, mean_acc, std_acc))
    end
end
# Ordenar por Mean Accuracy
sorted_df = sort(df_result, :Mean_Accuracy, rev=true)

# ------------------------------------------------------------------
# Accuracy Comparation
# ------------------------------------------------------------------
best_configs = Dict()
for (modeltype, results) in all_results
    # Ordenamos los resultados para cada modelo y cojemos el mejor
    best_result = sort(results, by=x -> x[2][1][1], rev=true)[1]
    best_configs[modeltype] = best_result
end

# Obtenemos los modelos del Dict para poder graficar
model_types = [string(k) for k in keys(best_configs)]
# Hacemos lo mismo con las accuracies
accuracies = [v[2][1][1] for v in values(best_configs)]

# Arreglamos las graficas para que se vean bien
min_accuracy, max_accuracy = minimum(accuracies), maximum(accuracies)
padding = (max_accuracy - min_accuracy) * 0.1
ylims_range = (min_accuracy - padding, max_accuracy + padding * 2)

acc_comparation = bar(model_types, accuracies, legend=false,
    ylabel="Accuracy (%)", xlabel="Model Type",
    title="Comparison of Model Accuracies",
    ylims=ylims_range,  # Adjusted ylims based on data
    yticks=round(min_accuracy - padding, digits=3):0.005:round(max_accuracy + padding * 2, digits=3),
    bar_width=0.5,  # Wider bars for better visibility
    color=[:lightblue, :lightgreen, :lightcoral],  # Different colors for each bar
    size=(800, 600))

# Para cada barra añadimos el porcentaje    
annotate!([(i, accuracies[i] + 0.0002, text(string(round(accuracies[i] * 100, digits=2)) * "%", 10)) for i in 1:(length(accuracies))])

# ------------------------------------------------------------------
# Save the Plots
# ------------------------------------------------------------------
CSV.write("results.csv", sorted_df)

savefig(original_distribution, "photos\\original_distribution.png")
savefig(balance_distribution, "photos\\balance_distribution.png")
savefig(hm, "photos\\heatmap.png")
savefig(acc_comparation, "photos\\acc_comparation.png")

# Histograms
for (i, histogram_plot) in enumerate(histograms)
    savefig(histogram_plot, "photos\\histogram_$(input_names[i]).png")
end

# Boxplots
for (i, boxplot_plot) in enumerate(boxplots)
    savefig(boxplot_plot, "photos\\boxplot_$(input_names[i]).png")
end
