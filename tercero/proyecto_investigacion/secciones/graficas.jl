#=
Te ayudaré a identificar las gráficas y estadísticas que deberías generar según cada sección del trabajo. Iré desglosándolo por partes:

1. Preparación de los datos (20%):

- Estadísticas descriptivas básicas:
  - Número de variables (561)
  - Número de instancias (10299)
  - Número de individuos (30)
  - Número de clases de salida (6 actividades)

- Gráficas para datos faltantes:
  - Mapa de calor o gráfico de barras mostrando el porcentaje de nulos por variable
  - Gráfico circular del porcentaje total de nulos en el dataset

- Visualización del holdout:
  - Gráfico de barras mostrando la distribución de clases en el conjunto de entrenamiento (90%) vs test (10%)
  - Gráfico mostrando qué individuos quedaron en cada conjunto

- Para el 5-fold cross validation:
  - Gráfico mostrando la distribución de individuos en cada fold
  - Distribución de clases en cada fold

2. Modelos básicos (30%):

- Para cada técnica de reducción de dimensionalidad (NO reducción, ANOVA, Mutual Information, RFE):
  - Gráficos de dispersión de las dos primeras características seleccionadas
  - Gráfico de importancia de variables (para ANOVA, MI y RFE)

- Para cada técnica de reducción adicional (PCA, LDA, ICA, Isomap, LLE):
  - Scatter plots de las dos primeras componentes/características conservadas
  - Gráfico de varianza explicada acumulada (especialmente para PCA)

- Para los clasificadores (MLP, KNN, SVM):
  - Gráficos de barras comparando accuracy por modelo
  - Matrices de confusión
  - Curvas ROC/AUC para cada clase
  - Gráficos de validación cruzada mostrando media y desviación estándar

3. Modelos de ensemble (30%):

- Para cada técnica (Bagging, AdaBoost, GBM):
  - Curvas de aprendizaje
  - Matrices de confusión
  - Gráficos de importancia de variables (especialmente para Random Forest y XGBoost)
  - Comparativa de métricas (accuracy, precision, recall, f1-score)
  - Gráficos de calibración de probabilidades para Soft Voting

4. Conclusiones (10%):

- Gráfico comparativo final de todos los modelos
- Gráfico de ranking de importancia de variables (Random Forest y XGBoost)
- Diagrama de cajas (boxplot) para comparar la distribución de resultados entre modelos
- Gráficos de los resultados del contraste de hipótesis
=#