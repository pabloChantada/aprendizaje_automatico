
include("prac1.jl")


#¿¿¿¿pruebas unitarias??????
# Crear un vector de ejemplo con valores booleanos
feature_bool = [true, false, true, true, false]

# Probar la función oneHotEncoding para un único parámetro feature de tipo AbstractArray{Bool,1}
matriz_resultante = oneHotEncoding(feature_bool)
println("Matriz resultante para la función oneHotEncoding con un solo parámetro:")
println(matriz_resultante)

# Crear un vector de ejemplo con valores de diferentes clases
feature_clases = ["A", "B", "A", "C", "B", "C"]

# Probar la función oneHotEncoding para dos parámetros feature y classes de tipo AbstractArray{<:Any,1}
matriz_resultante = oneHotEncoding(feature_clases, unique(feature_clases))
println("\nMatriz resultante para la función oneHotEncoding con dos parámetros:")
println(matriz_resultante)