
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier 

model = SVC(kernel="rbf", degree=3, gamma=2, C=1);
model = DecisionTreeClassifier(max_depth=4, random_state=1)
model = KNeighborsClassifier(3); 

# Train the model
trainingInputs = [1 2 3; 4 5 6; 7 8 9; 10 11 12; 13 14 15; 16 17 18; 19 20 21; 22 23 24; 25 26 27; 28 29 30];
trainingTargets = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0];
fit!(model, trainingInputs, trainingTargets); 
# vector donde cada elemento es la etiqueta
# patrones se suele suponer que están dispuestos en filas, y por lo tanto cada columna en la matriz de
# entradas se corresponde con un atributo

testOutputs = predict(model, testInputs); 
println(keys(model));

distances = decision_function(model, inputs); 
# Por último, es necesario tener en cuenta que estos modelos suelen recibir entradas y salidas
# preprocesadas, siendo el preprocesado más común la normalización ya descrita en una práctica
# anterior