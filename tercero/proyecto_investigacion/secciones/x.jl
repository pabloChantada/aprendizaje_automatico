using ScikitLearn
using RDatasets
# Registrar modelos
@sk_import ensemble: BaggingClassifier
@sk_import neighbors: KNeighborsClassifier
@sk_import ensemble: AdaBoostClassifier
@sk_import svm: SVC
@sk_import ensemble: GradientBoostingClassifier

# Cargamos los datos de ejemplo (iris dataset)
iris = dataset("datasets", "iris")
X = Matrix(iris[:, 1:4])
y = Vector{String}(iris.Species)

gbm = GradientBoostingClassifier(
    n_estimators=50,
    learning_rate=0.2,
    random_state=42
)

# Entrenamos el modelo
fit!(gbm, X, y)


# Hacemos predicciones con ambos modelos
y_pred_10 = predict(ada, X)

using Statistics
# Calculamos la precisión
accuracy_10 = mean(y_pred_10 .== y)

println("Precisión con 10 estimadores: ", round(accuracy_10, digits=4))