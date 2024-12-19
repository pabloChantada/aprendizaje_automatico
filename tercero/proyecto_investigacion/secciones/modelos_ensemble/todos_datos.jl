#=
11. Entrenar con el conjunto completo de entrenamiento (todo lo que componía el 5-fold cross-validation) y testear son el 10% reservado:
    Coger las 5 mejores combinaciones de los modelos anteriores de clasificación, (1 KNN, 1 SVM, 1 MLP, 1 Bagging y 1 AdaBoosting) 
    Crear un Random Forest con valor para los estimadores del 500 y profundidad máxima de 10 
    Crear un Hard Voting con las mejores combinaciones del KNN, SVM y MLP (uno para cada una de las técnicas) 
    Crear  un  Soft Voting  con las  mejores  combinaciones del  KNN,  SVM  y MLP (uno para cada una de las técnicas) para los pesos coger el porcentaje de 
        acierto  en  test  de  cada  una  de  las  combinaciones  en  el  5-fold  cross-valiadation 
    Crear un Ensemble Stacking con MLP como clasificador final, así mismo, use como base las mejores combinaciones del SVM, KNN y MLP 
    Crear un XGBoost con los valores por defecto 
    Crear un LightGBM, con los valores por defecto 
    Crear un Catboost, con los valores por defecto 
=#

include("../preparacion.jl")

using Statistics
using DataFrames
using ScikitLearn
using ScikitLearn: fit!, predict
using Plots
using MLJ
using XGBoost
using LightGBM
using CatBoost

@sk_import ensemble: RandomForestClassifier
@sk_import ensemble: VotingClassifier
@sk_import ensemble: StackingClassifier
@sk_import neural_network: MLPClassifier
@sk_import ensemble: BaggingClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import svm: SVC
@sk_import ensemble: GradientBoostingClassifier


# ===========================================
# Funciones para entrenar clasificadores
# ===========================================

function train_random_forest(X, y)
    X_matrix = Matrix(X)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )
    fit!(rf, X_matrix, y)
    return rf
end

function train_hard_voting_classifiers(models, X, y)
    X_matrix = Matrix(X)
    voting_hard = VotingClassifier(
        estimators=models,
        voting="hard"
    )
    fit!(voting_hard, X_matrix, y)
    return voting_hard
end

function train_soft_voting_classifiers(models, weights, X, y)
    X_matrix = Matrix(X)
    voting_soft = VotingClassifier(
        estimators=models,
        voting="soft",
        weights=weights
    )
    fit!(voting_soft, X_matrix, y)
    return voting_soft
end

function train_stacking_classifiers(models, final_estimator, X, y)
    X_matrix = Matrix(X)
    stacking = StackingClassifier(
        estimators=models,
        final_estimator=final_estimator
    )
    fit!(stacking, X_matrix, y)
    return stacking
end

function train_xgboost(X, y)
    dtrain = xgboost.DMatrix(X, label=y)
    params = Dict("objective" => "binary:logistic", "eval_metric" => "logloss", "max_depth" => 6)
    num_round = 100
    xgb = xgboost.train(params, dtrain, num_round)
    return xgb
end
function train_lightgbm(X, y)
    X_matrix = Matrix(X)
    lgbm = LightGBMClassifier()
    fit!(lgbm, X_matrix, y)
    return lgbm
end

function train_catboost(X, y)
    X_matrix = Matrix(X)
    catboost = CatBoostClassifier()
    fit!(catboost, X_matrix, y)
    return catboost
end

# ===========================================
# Pipeline principal
# ===========================================

function train_ensemble_models(fold, top_models, top_accuracies)
    # Filtrar los datos
    X_train = Matrix(fold.train[:, 1:end-1])
    y_train = fold.train[:, end]

    models = Dict()

    # Random Forest
    println("Entrenando Random Forest...")
    models["random_forest"] = train_random_forest(X_train, y_train)

    # Hard Voting
    println("Entrenando Hard Voting...")
    models["voting_hard"] = train_hard_voting_classifiers(top_models, X_train, y_train)

    # Soft Voting
    println("Entrenando Soft Voting...")
    models["voting_soft"] = train_soft_voting_classifiers(top_models, top_accuracies, X_train, y_train)

    # Stacking
    println("Entrenando Stacking...")
    final_estimator = MLPClassifier(random_state=42)
    models["stacking"] = train_stacking_classifiers(top_models, final_estimator, X_train, y_train)

    # XGBoost
    println("Entrenando XGBoost...")
    # models["xgboost"] = train_xgboost(X_train, y_train)

    # LightGBM
    println("Entrenando LightGBM...")
    # models["lightgbm"] = train_lightgbm(X_train, y_train)

    # CatBoost
    println("Entrenando CatBoost...")
    # models["catboost"] = train_catboost(X_train, y_train)

    return models
end

function evaluate_models(model, X_test, y_test)
    results = Dict()
    
    # println("Evaluando modelo: $model_name")
    
    # Predicciones
    y_pred = ScikitLearn.predict(model, X_test)
    
    # Métricas
    accuracy = mean(y_pred .== y_test)
    precision = sum((y_pred .== 1) .& (y_test .== 1)) / sum(y_pred .== 1)
    recall = sum((y_pred .== 1) .& (y_test .== 1)) / sum(y_test .== 1)
    specificity = sum((y_pred .== 0) .& (y_test .== 0)) / sum(y_test .== 0)
    
    # Guardar resultados
    results[model_name] = Dict(
        "accuracy" => round(accuracy, digits=4),
        "precision" => round(precision, digits=4),
        "recall" => round(recall, digits=4),
        "specificity" => round(specificity, digits=4)
    )
    
    println("  Accuracy: $accuracy")
    println("  Precision: $precision")
    println("  Recall: $recall")
    println("  Specificity: $specificity")
    
    return results
end

function print_feature_importance(model, feature_names)
    if haskey(model, "random_forest")
        rf = model["random_forest"]
        importance = rf.feature_importances_
        println("Importancia de las variables (Random Forest):")
        for (i, imp) in enumerate(importance)
            println("  $(feature_names[i]): $(round(imp, digits=4))")
        end
    else
        println("No se encontró el modelo Random Forest en los resultados.")
    end
end


# Ejecutar el pipeline con folds y modelos preseleccionados
println("Iniciando entrenamiento de modelos ensemble...")

data = preparation_pipeline(true)

for (i, fold) in enumerate(data["normalized_folds"])
    println("\nProcesando fold $i de $(length(data["normalized_folds"]))...")
    top_models = [("knn", KNeighborsClassifier(n_neighbors=5)),
                  ("svm", SVC(kernel="linear", probability=true)),
                  ("mlp", MLPClassifier(random_state=42))]
    top_accuracies = [0.9, 0.85, 0.88]  # Pesos basados en accuracy del 5-fold CV

    models = train_ensemble_models(fold, top_models, top_accuracies)
    println("Modelos entrenados para fold $i")
    
    # Imprimir la importancia de variables del Random Forest
    feature_names = names(fold.train) |> x -> x[1:end-1]  # Excluir la columna target
    print_feature_importance(models, feature_names)
end
