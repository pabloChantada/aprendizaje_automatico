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

   accuracy:
      Nota: 0.2

   buildClassANN:
      Nota: 0.25


   -------------------------------------------------------------------------------------------------------------------------
   Ejercicio 3

   holdOut:
      Nota: 0.1

   trainClassANN:
      Error al ejecutar la función con 2 clases: DimensionMismatch: layer Dense(3 => 8, tanh) expects size(input, 1) == 3, but got 0×0 transpose(::Matrix{Float32}) with eltype Float32

   -------------------------------------------------------------------------------------------------------------------------
   Ejercicio 4

   confusionMatrix:
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como positivos: VPN incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones positivos: especificidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), clasificados todos como negativos: VPP incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos patrones negativos: sensibilidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como positivos: especificidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como positivos: VPN incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: sensibilidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos positivos y clasificados como negativos: VPN incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: sensibilidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: VPP incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como negativos: F1 incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: especificidad incorrecta
      Salida incorrecta al ejecutar la funcion con un vector de valores booleanos (2 clases), todos negativos y clasificados como positivos: VPP incorrecta
      Error al ejecutar la funcion con una matriz (multiclase) de valores booleanos: BoundsError: attempt to access 4×4 Matrix{Int64} at index [4, 500×4 BitMatrix]
      Error al ejecutar la funcion con una matriz (multiclase) de valores reales: BoundsError: attempt to access 4×4 Matrix{Int64} at index [4, 500×4 BitMatrix]
      Error al ejecutar la funcion con un vector de valores de tipo Any (multiclase): BoundsError: attempt to access 3×3 Matrix{Int64} at index [3, 4×3 BitMatrix]


   -------------------------------------------------------------------------------------------------------------------------
   Ejercicio 5

   # crossvalidation: solucionado
      Error al ejecutar la funcion con parametros de tipo (AbstractArray{Bool,2},Int64): BoundsError: attempt to access 100-element Vector{Int64} at index [[2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57, 62, 67, 72, 77, 82, 87, 92, 97], 2]
      Error al ejecutar la funcion con parametros de tipo (AbstractArray{<:Any,1},Int64): BoundsError: attempt to access 30-element Vector{Int64} at index [[2, 6, 9, 10, 15, 17, 18, 23, 24, 30], 2]

   ANNCrossValidation:
      Error al ejecutar la funcion: BoundsError: attempt to access 30×3 Matrix{Float64} at index [1:30, [9, 5, 2, 10, 7, 8, 6, 5, 3, 6, 10, 2, 2, 4, 3, 8, 7, 4, 8, 5, 7, 4, 10, 6, 9, 9, 3]]


   -------------------------------------------------------------------------------------------------------------------------
   Ejercicio 6

   modelCrossValidation:
      Funcion no definida


   -------------------------------------------------------------------------------------------------------------------------
   Nota del archivo entregado: 0.94
