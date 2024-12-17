#=
10. Adicionalmente, con los datos sólo con el tratamiento de Filtrado ANOVA, recrear las siguientes técnicas:
    BaggingClassifier con clasificador base KNN con número de vecinos 5 y número de estimadores 10 y 50 
    AdaBoosting  con  estimadores  SVM  con  kernel  lineal  siendo  el  número  de  estimadores 5. 
    GBM (GradientBoostingClasifier), con 50 estimadores y un learning_rate de 0.2 
=#

include("preparacion.jl")
include("modelos_basicos/reduccion_dimensionalidad.jl")

using Statistics
using DataFrames
using ScikitLearn: fit!, predict
using ScikitLearn

# Registrar modelos
@sk_import ensemble: BaggingClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import svm: SVC
@sk_import ensemble: GradientBoostingClassifier

# ===========================================
# Función para entrenar BaggingClassifier con KNN
# ===========================================
function train_bagging_knn(X, y; n_estimators=10)
    # Convertir el DataFrame a matriz
    X_matrix = Matrix(X)
    
    # Definir el modelo base KNN
    base_knn = KNeighborsClassifier(n_neighbors=5)
    
    # Crear el modelo de bagging
    bagging = BaggingClassifier(
        estimator=base_knn,
        n_estimators=n_estimators,
        random_state=42
    )
    
    # Entrenar el modelo
    fit!(bagging, X_matrix, y)
    
    return bagging
end

# ===========================================
# Función para entrenar AdaBoost con SVM
# ===========================================
function train_adaboost_svm(X, y)
    # Convertir el DataFrame a matriz
    X_matrix = Matrix(X)
    
    # Definir el modelo base SVM con kernel lineal
    base_svm = SVC(kernel="linear", probability=true)
    
    # Crear el modelo AdaBoost
    ada = AdaBoostClassifier(
        estimator=base_svm,
        n_estimators=5,
        random_state=42
    )
    
    # Entrenar el modelo
    fit!(ada, X_matrix, y)
    
    return ada
end

# ===========================================
# Función para entrenar GradientBoosting
# ===========================================
function train_gbm(X, y)
    X_matrix = Matrix(X)
    
    gbm = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.2,
        random_state=42
    )
    
    ScikitLearn.fit!(gbm, X_matrix, y)
    return gbm
end

# ===========================================
# Función para entrenar todos los modelos
# ===========================================

function train_ensemble_models(fold)

    # Aplicar ANOVA SOLO sobre el conjunto de entrenamiento
    println("\nAplicando Filtrado ANOVA en el conjunto de entrenamiento...")
    selected_features = anova_filter(fold.train, :Activity)
    
    # Filtrar los datos de entrenamiento
    X_train_filtered = select(fold.train, [selected_features...])
    y_train = fold.train[:, :Activity]
    
    # Filtrar los datos de prueba usando las mismas características seleccionadas
    X_test_filtered = select(fold.test, [selected_features...])
    y_test = fold.test[:, :Activity]

    models = Dict()
    results = Dict()
    tiempos = Dict()
    
    # Bagging KNN 10
    println("\nEntrenando Bagging KNN (10 estimadores)...")
    t_start = time()
    models["bagging_knn_10"] = train_bagging_knn(X_train_filtered, y_train, n_estimators=10)
    tiempos["bagging_knn_10"] = round(time() - t_start, digits=2)
    
    # Bagging KNN 50
    println("Entrenando Bagging KNN (50 estimadores)...")
    t_start = time()
    models["bagging_knn_50"] = train_bagging_knn(X_train_filtered, y_train, n_estimators=50)
    tiempos["bagging_knn_50"] = round(time() - t_start, digits=2)
    
    #= AdaBoost
    println("Entrenando AdaBoost SVM...")
    t_start = time()
    models["adaboost_svm"] = train_adaboost_svm(X_train_filtered, y_train)
    tiempos["adaboost_svm"] = round(time() - t_start, digits=2)
    =#

    # GBM
    println("Entrenando GBM...")
    t_start = time()
    models["gbm"] = train_gbm(X_train_filtered, y_train)
    tiempos["gbm"] = round(time() - t_start, digits=2)

    # Evaluación
    println("\nEvaluando modelos...")
    for (name, model) in models
        t_start = time()
        y_pred = ScikitLearn.predict(model, Matrix(X_test_filtered))
        tiempo_pred = round(time() - t_start, digits=2)
        accuracy = mean(y_pred .== y_test)
        results[name] = accuracy
        println("  - $name: Accuracy = $(round(accuracy, digits=4))")
        println("    Tiempo entrenamiento: $(tiempos[name])s")
        println("    Tiempo predicción: $(tiempo_pred)s")
    end
    
    return models, results
end

# ===========================================
# Pipeline principal
# ===========================================

# Ejecutar el pipeline
println("\nIniciando pipeline de entrenamiento...")
println("======================================")

folds = preparation_pipeline(false)
println("Número total de folds: $(length(folds["normalized_folds"]))")

# Añadir verbose al pipeline principal
function ensemble_pipeline(folds; verbose=true)
    all_results = []
    
    for (i, fold) in enumerate(folds["normalized_folds"])
        if verbose
            println("\nProcesando fold $i de $(length(folds["normalized_folds"]))...")
            println("----------------------------------------")
            println("Tamaño conjunto entrenamiento: $(size(fold.train, 1)) muestras")
            println("Tamaño conjunto test: $(size(fold.test, 1)) muestras")
        end
        
        # Tiempo de entrenamiento
        tiempo_inicio = time()
        models, results = train_ensemble_models(fold)
        tiempo_total = round(time() - tiempo_inicio, digits=2)
        
        if verbose
            println("\nResultados fold $i:")
            for (model, acc) in results
                println("  - $model: $(round(acc, digits=4))")
            end
            println("Tiempo de entrenamiento fold $i: $tiempo_total segundos")
        end
        
        push!(all_results, results)
    end
    
    # Calcular medias de accuracy por modelo
    mean_results = Dict()
    if verbose
        println("\nCalculando estadísticas finales...")
    end
    
    for model_name in keys(all_results[1])
        accuracies = [result[model_name] for result in all_results]
        mean_results[model_name] = (
            mean = mean(accuracies),
            std = std(accuracies),
            min = minimum(accuracies),
            max = maximum(accuracies)
        )
    end
    
    return mean_results
end

# Ejecutar con verbose
println("\nIniciando entrenamiento de modelos...")
tiempo_inicio_total = time()
results = ensemble_pipeline(folds, verbose=true)

# Mostrar resultados finales
tiempo_total = round(time() - tiempo_inicio_total, digits=2)
println("\nResultados finales:")
println("==================")
for (model, metrics) in results
    println("$model:")
    println("  - Accuracy = $(round(metrics.mean, digits=4)) ± $(round(metrics.std, digits=4))")
    println("  - Rango = [$(round(metrics.min, digits=4)), $(round(metrics.max, digits=4))]")
end

println("\nTiempo total de ejecución: $(tiempo_total / 60) minutos")

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