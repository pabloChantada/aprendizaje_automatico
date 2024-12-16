#=
10. Adicionalmente, con los datos sólo con el tratamiento de Filtrado ANOVA, recrear las siguientes técnicas:
    BaggingClassifier con clasificador base KNN con número de vecinos 5 y número de estimadores 10 y 50 
    AdaBoosting  con  estimadores  SVM  con  kernel  lineal  siendo  el  número  de  estimadores 5. 
    GBM (GradientBoostingClasifier), con 50 estimadores y un learning_rate de 0.2 
=#

include("preparacion.jl")
using MLJ
using MLJScikitLearnInterface
using Statistics
using DataFrames

# Registrar modelos

KNNClassifier = @load KNeighborsClassifier pkg=MLJScikitLearnInterface
AdaBoostClassifier = @load AdaBoostClassifier pkg=MLJScikitLearnInterface
GradientBoostingClassifier = @load GradientBoostingClassifier pkg=MLJScikitLearnInterface

# ===========================================
# Función para entrenar BaggingClassifier con KNN
# ===========================================
function train_bagging_knn(X, y; n_estimators=10)
    # Definir el modelo base KNN
    knn = KNNClassifier(n_neighbors=5)
    
    # Crear el modelo de bagging
    bagging = EnsembleModel(
        model=knn,
        n=n_estimators,
        rng=42
    )
    
    # Entrenar el modelo
    model = machine(bagging, X, y)
    fit!(model)
    
    return model
end

# ===========================================
# Función para entrenar AdaBoost con SVM
# ===========================================
function train_adaboost_svm(X, y)
    # Definir el modelo base SVM compatible
    svm = SVMClassifier(kernel="linear")  

    # Crear el modelo AdaBoost con SVM como estimador
    adaboost = AdaBoostClassifier(
        estimator=svm,
        n_estimators=5,
        random_state=42
    )
    
    # Entrenar el modelo
    model = machine(adaboost, X, y)
    fit!(model)
    
    return model
end

# ===========================================
# Función para entrenar GradientBoosting
# ===========================================
function train_gbm(X, y)
    # Crear el modelo GBM
    gbm = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.2,
        random_state=42
    )
    
    # Entrenar el modelo
    model = machine(gbm, X, y)
    fit!(model)
    
    return model
end

# ===========================================
# Función principal para entrenar todos los modelos
# ===========================================
function train_all_ensemble_models(fold)
    # Extraer datos de entrenamiento y test del fold
    X_train = select(fold.train, Not([:subject, :Activity]))
    y_train = categorical(fold.train.Activity)
    X_test = select(fold.test, Not([:subject, :Activity]))
    y_test = categorical(fold.test.Activity)
    
    # Entrenar modelos
    models = Dict()
    # Bagging con KNN (10 estimadores)
    models["bagging_knn_10"] = train_bagging_knn(X_train, y_train, n_estimators=10)
    
    # Bagging con KNN (50 estimadores)
    models["bagging_knn_50"] = train_bagging_knn(X_train, y_train, n_estimators=50)
    
    # AdaBoost con SVM
    # models["adaboost_svm"] = train_adaboost_svm(X_train, y_train)
    
    # GradientBoosting
    # models["gbm"] = train_gbm(X_train, y_train)
    
    # Evaluar modelos
    results = Dict()
    for (name, model) in models
        y_pred = predict(model, X_test)
        accuracy = mean(y_pred .== y_test)
        results[name] = accuracy
    end
    
    return models, results
end

# ===========================================
# Pipeline principal
# ===========================================
function ensemble_pipeline(folds)

    all_results = []
    
    for (i, fold) in enumerate(folds["normalized_folds"])
        println("Procesando fold $i...")
        models, results = train_all_ensemble_models(fold)
        push!(all_results, results)
    end
    
    # Calcular medias de accuracy por modelo
    mean_results = Dict()
    for model_name in keys(all_results[1])
        accuracies = [result[model_name] for result in all_results]
        mean_results[model_name] = (
            mean = mean(accuracies),
            std = std(accuracies)
        )
    end
    
    return mean_results
end

folds = preparation_pipeline(false)
# Ejecutar el pipeline
results = ensemble_pipeline(folds)

# Mostrar resultados
for (model, metrics) in results
    println("$model: Accuracy = $(round(metrics.mean, digits=4)) ± $(round(metrics.std, digits=4))")
end


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