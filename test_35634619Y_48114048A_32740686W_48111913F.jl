using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
targets = [1 0 0 0; 
            1 0 0 0;
            0 0 1 0; 
            0 0 1 0;
            1 0 0 0;
            0 0 0 1;
            0 1 0 0;
            1 0 0 0;]
k = 4

index_vector = Vector{Any}(undef, size(targets, 1))

for i = 1:(size(targets, 2))
    # Numero de elementos en cada particion
    elements = sum(targets[:, i] .== 1)
    col_positions = crossvalidation(elements, k)
    index_vector[findall(targets[:, i] .== 1)] .= col_positions
end
println(index_vector)


using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: targets with 4 columns and 8 rows
targets1 = [true false true;
            false true false;
            true false true;
            true false true;
            true false true;
            false true false;
            true false true;
            true false true]
k1 = 4
println(crossvalidation(targets1, k1))

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

# Test case 1: targets with 4 columns and 8 rows
targets1 = [true false true;
            false true false;
            true false true;
            true false true;
            true false true;
            false true false;
            true false true;
            true false true]
k1 = 4
crossvalidation(targets1, k1)