include("35634619Y_48114048A_32740686W_48111913F.jl")
# Simulando datos de entrada
targets_matrix = Bool[false true false; 
                      false true false; 
                      false true false; 
                      false true false; 
                      false true false;
                      true false false; 
                      true false false; 
                      true false false; 
                      true false false; 
                      true false false; 
                      false false true; 
                      false false true; 
                      false false true; 
                      false false true; 
                      false false true]
k = 3  # Número de particiones

# Llamada a la función de validación cruzada para matrices
crossvalidation_indices_matrix = crossvalidation(targets_matrix, k)
println(crossvalidation_indices_matrix)
# Simulando datos de entrada
targets_vector = ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C", "A", "B", "C"]
crossvalidation_indices_vector = crossvalidation(targets_vector, k)

targets_vector = ["perro", 1, true, 1, 1, 1, true, true, "perro", "perro", 1, 1, true, true, "perro"]

k = 3  # Número de particiones

# Llamada a la función de validación cruzada para vectores de clases heterogéneas
crossvalidation_indices_vector = crossvalidation(targets_vector, k)
println(crossvalidation_indices_vector)