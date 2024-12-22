using DataFrames
using CSV
using XGBoost
using LightGBM
using ScikitLearn
using Random
using Statistics

# Set random seed
Random.seed!(172)

# Load and prepare data function
function load_sample_data(n_rows=5000)
    # Load data
    data = CSV.read("Datos_Práctica_Evaluación_1.csv", DataFrame)
    
    # Take first n_rows
    data = first(data, n_rows)
    
    # Handle missing values before conversion to matrix
    numeric_cols = names(select(data, Not([:subject, :Activity])))
    for col in numeric_cols
        if any(ismissing.(data[:, col]))
            col_mean = mean(skipmissing(data[:, col]))
            data[!, col] = coalesce.(data[:, col], col_mean)
        end
    end
    
    # Split features and target
    X = Matrix{Float64}(select(data, Not([:subject, :Activity])))
    y = data.Activity
    
    # Convert target to numeric starting from 0
    unique_labels = unique(y)
    label_map = Dict(label => idx-1 for (idx, label) in enumerate(unique_labels))
    y_numeric = Int32[label_map[label] for label in y]
    
    return X, y_numeric, label_map
end

# Test XGBoost
function test_xgboost(X, y)
    println("\nTesting XGBoost...")
    try
        num_class = length(unique(y))
        bst = xgboost((X, y); 
            num_round = 10,
            objective = "multi:softmax", 
            num_class = num_class,
            max_depth = 6,
            eta = 0.3
        )
    
        y_pred = XGBoost.predict(bst, X)
        accuracy = mean(y_pred .== y)
        println("XGBoost Accuracy: ", round(accuracy, digits=4))
        
        return bst, accuracy
    catch e
        println("Error in XGBoost: ", e)
        return nothing, 0.0
    end
end

# Test LightGBM
# Test LightGBM
function test_lightgbm(X, y)
    println("\nTesting LightGBM...")
    try
        num_class = length(unique(y))
        
        # Create the estimator
        estimator = LGBMClassification(
            objective = "multiclass",
            num_class = num_class,
            num_iterations = 100,
            learning_rate = 0.1,
            early_stopping_round = 5,
            feature_fraction = 0.8,
            bagging_fraction = 0.9,
            bagging_freq = 1,
            num_leaves = 31,
            metric = ["multi_logloss"]  # Ahora es un vector de strings
        )
        LightGBM.fit!(estimator, X, y)
        # Make predictions
        y_pred = LightGBM.predict(estimator, X)
        
        if size(y_pred, 2) == 1
            y_pred_class = Int32.(round.(y_pred))
        else
            y_pred_class = Int32[argmax(y_pred[i,:]) - 1 for i in 1:size(y_pred,1)]
        end
        
        accuracy = mean(y_pred_class .== y)
        println("LightGBM Accuracy: ", round(accuracy, digits=4))
        println("Prediction shape: ", size(y_pred))
        println("Unique predictions: ", unique(y_pred_class))
        println("Unique actual values: ", unique(y))
        
        return estimator, accuracy
    catch e
        println("Error in LightGBM: ")
        println(e)
        println("Stacktrace:")
        for (exception, backtrace) in Base.catch_stack()
            showerror(stdout, exception, backtrace)
            println()
        end
        return nothing, 0.0
    end
end

# Main testing function
function test_all_boosting()
    # Load sample data
    X, y, label_map = load_sample_data(100)
    println("Loaded data shape: ", size(X))
    println("Number of classes: ", length(unique(y)))
    println("Class labels: ", sort(unique(y)))
    
    # Test each model
    _, xgb_acc = test_xgboost(X, y)
    _, lgb_acc = test_lightgbm(X, y)
    
    # Print summary
    println("\nSummary of Results:")
    println("==================")
    println("XGBoost Accuracy: ", round(xgb_acc * 100, digits=2), "%")
    println("LightGBM Accuracy: ", round(lgb_acc * 100, digits=2), "%")
    
end

# Run tests
models = test_all_boosting()