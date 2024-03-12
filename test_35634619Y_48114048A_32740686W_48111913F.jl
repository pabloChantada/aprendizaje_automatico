using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Test case 1: Single output node, all correct predictions
@testset "Accuracy with single output node, all correct predictions" begin
    outputs = [true true; true true]
    targets = [true true; true true]
    @test accuracy(outputs, targets) == 1.0
end

# Test case 2: Single output node, all incorrect predictions
@testset "Accuracy with single output node, all incorrect predictions" begin
    outputs = [false false false; false false false]
    targets = [true true true; true true true]
    @test accuracy(outputs, targets) == 0.0
end

# Test case 3: Multiple output nodes, some correct and some incorrect predictions
@testset "Accuracy with multiple output nodes, some correct and some incorrect predictions" begin
    outputs = [true false; true false]
    targets = [true true; false false]
    @test accuracy(outputs, targets) == 0.5
end


# Test case 5: Different sizes of outputs and targets
@testset "Accuracy with different sizes of outputs and targets" begin
    outputs = [true, true, true]
    targets = [true, true, true, true, true]
    @test_throws AssertionError accuracy(outputs, targets)
end

using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")


# Test case 1: N is divisible by k
N1 = 15
k1 = 3
partitions1 = crossvalidation(N1, k1)
println("Test Case 1:")
println("Number of partitions:", length(unique(partitions1)))
println("Lengths of partitions:", [sum(partitions1 .== i) for i in 1:k1])
println("Partitions:")
println(partitions1)
@assert length(unique(partitions1)) == k1
# Test case 2: N is not divisible by k
N2 = 14
k2 = 7
partitions2 = crossvalidation(N2, k2)
println("Test Case 2:")
println("Number of partitions:", length(unique(partitions2)))
println("Lengths of partitions:", [sum(partitions2 .== i) for i in 1:k2])
println("Partitions:")
println(partitions2)
@assert length(unique(partitions2)) == k2
# Test case 3: N is smaller than k
N3 = 3
k3 = 5
partitions3 = crossvalidation(N3, k3)
println("Test Case 3:")
println("Number of partitions:", length(unique(partitions3)))
println("Lengths of partitions:", [sum(partitions3 .== i) for i in 1:k3])
println("Partitions:")
println(partitions3)
@assert length(unique(partitions3)) == 3