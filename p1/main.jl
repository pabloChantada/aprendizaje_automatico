include("functions.jl")

using  XLSX; 
xf = XLSX.readxlsx("p1\\Dry_Bean_Dataset.xlsx")
dataset = xf["Dry_Beans_Dataset"]
# inputs = dataset[2:1001, 1:7]
inputs = dataset[2:end, 1:16]
inputs = convert(Array{Float32,2},inputs);
targets = vec(dataset[2:end, 17])

# targets = oneHotEncoding(targets)

train_index, test_index = holdOut(size(inputs,1), 0.3)

max_vals, min_vals = calculateMinMaxNormalizationParameters(inputs)
print(max_vals)
print(min_vals)
mean_vals, std_vals = calculateZeroMeanNormalizationParameters(inputs)
inputs2 = copy(inputs)
x = normalizeMinMax!(inputs, (min_vals, max_vals))
x2 = normalizeZeroMean(inputs2, (mean_vals, std_vals))
# Assuming x is already normalized
total_ones = sum(x .>= 0.9)
total_zero = sum(x .<= 0.1)
total_ones = sum(x .== 1)
total_ones = sum(x .== 0)

total_ones = sum(x2 .>= 0.9)
total_zero = sum(x2 .<= 0.1)
total_ones = sum(x2 .== 1)
total_ones = sum(x2 .== 0)

println("Total number of ones in the normalized data: ", total_ones)
println("Total number of zeros in the normalized data: ", total_zero)

# normalizeMinMax!(inputs_train, (min_vals, max_vals))
# normalizeMinMax!(inputs_val, (min_vals, max_vals))
# normalizeMinMax!(inputs_test, (min_vals, max_vals))

topology = [3, 5]
modelType = :ANN  # Or :DecisionTreeClassifier, :KNeighborsClassifier
modelType2 = :SVC  # Or :DecisionTreeClassifier, :KNeighborsClassifier
modelType3 = :DecisionTreeClassifier  # Or :DecisionTreeClassifier, :KNeighborsClassifier
modelType4 = :KNeighborsClassifier  # Or :DecisionTreeClassifier, :KNeighborsClassifier
modelHyperparameters = Dict("topology" => [10, 5])
modelHyperparameters2 = Dict("kernel" => "linear", "C" => 1)
modelHyperparameters3 = Dict("max_depth" => 3)
modelHyperparameters4 = Dict("n_neighbors" => 5)

# Assume inputs and targets are predefined
num_samples = size(inputs, 1)
num_folds = 2  # Number of folds for cross-validation

# Generate crossValidationIndices
crossValidationIndices = Vector{Int}(undef, num_samples)
for i in 1:num_samples
    # Assign fold index to each sample
    crossValidationIndices[i] = i % num_folds + 1
end

results_ann = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)
results_svm = modelCrossValidation(modelType2, modelHyperparameters2, inputs, targets, crossValidationIndices)
print(results_svm)
results_tree = modelCrossValidation(modelType3, modelHyperparameters3, inputs, targets, crossValidationIndices)
results_knn = modelCrossValidation(modelType4, modelHyperparameters4, inputs, targets, crossValidationIndices)