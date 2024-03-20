-------------------------------------------------------------------------------------------------------------------------
Ejercicio 2

oneHotEncoding:
   Nota: 0.02

calculateMinMaxNormalizationParameters:
   Nota: 0.02

calculateZeroMeanNormalizationParameters:
   Nota: 0.02

normalizeMinMax!:
   Nota: 0.02

normalizeMinMax:
   Nota: 0.02

normalizeZeroMean!:
   Nota: 0.02

normalizeZeroMean:
   Nota: 0.02

classifyOutputs:
   Nota: 0.25

# ARREGLADA, SON ACC POR ATRIBUTO NO POR ELEMENTO DE LA MATRIZ
accuracy:
   Salidas incorrectas al hacer el calculo con parametros (AbstractArray{Bool,2},AbstractArray{Bool,2} al usar valores booleanos de mas de una columna)
   Salidas incorrectas con parametros (AbstractArray{<:Real,2}, AbstractArray{Bool,2}; threshold::Real=0.5) al hacer el calculo con una matriz de valores reales como salidas y una matriz de valores booleanos como salidas deseadas, ambas de mas de una columna

buildClassANN:
   Nota: 0.25


-------------------------------------------------------------------------------------------------------------------------
Ejercicio 3

holdOut:
   Nota: 0.1

# Arreglao creo
__trainClassANN:__
      Valores de loss de entrenamiento incorrectos al ejecutar la función con 2 clases con conjunto de validacion al entrenar un maximo de 0 ciclos

-------------------------------------------------------------------------------------------------------------------------
Ejercicio 4

# Arreglao
confusionMatrix:
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases): sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases): especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases): VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases): VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases): matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: F1 incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: F1 incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como positivos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como positivos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: F1 incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: F1 incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: F1 incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: matriz de confusion incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de salidas reales (2 clases): sensibilidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de salidas reales (2 clases): especificidad incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de salidas reales (2 clases): VPP incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de salidas reales (2 clases): VPN incorrecta
   Salida incorrecta al ejecutar la funcion con un vector de salidas reales (2 clases): matriz de confusion incorrecta



   Salidas incorrectas al ejecutar la funcion con una matriz (multiclase) de valores booleanos y weighted=true: la salida no es una tupla de 8 elementos, con 7 valores reales y una matriz
   Salidas incorrectas al ejecutar la funcion con una matriz (multiclase) de valores booleanos y weighted=false: la salida no es una tupla de 8 elementos, con 7 valores reales y una matriz
   Salidas incorrectas al ejecutar la funcion con una matriz (multiclase) de valores reales y weighted=true: la salida no es una tupla de 8 elementos, con 7 valores reales y una matriz
   Salidas incorrectas al ejecutar la funcion con una matriz (multiclase) de valores reales y weighted=false: la salida no es una tupla de 8 elementos, con 7 valores reales y una matriz
   Error al ejecutar la funcion con un vector de valores de tipo Any (multiclase): BoundsError: attempt to access NTuple{7, Float64} at index [8]

Closest candidates are:
iterate(!Matched::Union{LinRange, StepRangeLen})
@ Base range.jl:880
iterate(!Matched::Union{LinRange, StepRangeLen}, !Matched::Integer)
@ Base range.jl:880
iterate(!Matched::T) where T<:Union{Base.KeySet{<:Any, <:Dict}, Base.ValueIterator{<:Dict}}
@ Base dict.jl:698
...



-------------------------------------------------------------------------------------------------------------------------
Ejercicio 5
# ARREGLAR (CREO QUE YA ESTA)
__crossvalidation:__
   Error al ejecutar la funcion con una sola columna: AssertionError: Numero de particiones debe ser mayor o igual que 10
   Error al ejecutar la funcion con una sola columna: AssertionError: Numero de particiones debe ser mayor o igual que 10
   Error al ejecutar la funcion con una sola columna: AssertionError: Numero de particiones debe ser mayor o igual que 10

# ARREGLAR
__ANNCrossValidation:__
   Error al ejecutar la funcion: BoundsError: attempt to access 30×3 Matrix{Float64} at index [1:30, [5, 8, 9, 7, 4, 7, 9, 8, 4, 4, 3, 5, 2, 3, 3, 10, 8, 2, 6, 7, 10, 10, 2, 6, 5, 6, 9]]


modelCrossValidation:
   Resultados incorrectos: no devuelve una tupla de 7 valores donde cada uno es una tupla de 2 valores al entrenar el modelo ANN
   Resultados incorrectos: no devuelve una tupla de 7 valores donde cada uno es una tupla de 2 valores al entrenar el modelo SVC
   Resultados incorrectos: no devuelve una tupla de 7 valores donde cada uno es una tupla de 2 valores al entrenar el modelo DecisionTreeClassifier
   Resultados incorrectos: no devuelve una tupla de 7 valores donde cada uno es una tupla de 2 valores al entrenar el modelo KNeighborsClassifier
-------------------------------------------------------------------------------------------------------------------------
Nota del archivo entregado: 0.74
