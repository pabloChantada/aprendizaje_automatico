#=
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
=#

include("main.jl")

dataset = readdlm("iris.data",',');
inputs = dataset[:,1:4];
inputs = convert(Array{Float32,2},inputs);
targets = dataset[:,5];

# PARTE 1
# --------------------------------------------------------------------------

# PARTE 2
# --------------------------------------------------------------------------

min_val, max_val = calculateMinMaxNormalizationParameters(inputs)
#= Solo para el print sin el Float32
min_val_float64 = convert(Array{Float64}, min_val)
max_val_float64 = convert(Array{Float64}, max_val)=#
println("Minimum values per column: ", min_val)
println("Maximum values per column: ", max_val)

mean_col, std_col = calculateZeroMeanNormalizationParameters(inputs)
#= Solo para el print sin el Float32
mean_col_float64 = convert(Array{Float64}, mean_col)
std_col_float64 = convert(Array{Float64}, std_col)=#
println("Mean values per column: ", mean_col)
println("Standart Deviation values per column: ", std_col)