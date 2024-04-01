using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

@testset "Tests de la práctica" begin
    # Datos de prueba
    outputs_multiclase = Bool[true false false; false true false; false false true]
    targets_multiclase = Bool[true false false; false true false; false false true]

    # Test de la función confusion_matrix
    @testset "Test de confusion_matrix con datos multiclase" begin
        cm = confusionMatrix(outputs_multiclase, targets_multiclase)
        @test cm[8] == [1 0 0; 0 1 0; 0 0 1]
    end

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end

@testset "Tests de la práctica" begin
    # Datos de prueba
    outputs_incorrectos = Bool[false true false; true false false; false false true]
    targets_correctos = Bool[true false false; false true false; false false true]

    # Test de la función confusion_matrix
    @testset "Test de confusion_matrix con clasificaciones incorrectas" begin
        cm = confusionMatrix(outputs_incorrectos, targets_correctos)
        @test cm[8] == [0 1 0; 1 1 0; 0 0 2]
    end

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end

@testset "Tests de la práctica" begin
    # Datos de prueba
    outputs_complejos = Bool[true false true; false true false; true false false]
    targets_complejos = Bool[true true false; false false true; true false true]

    # Test de la función confusion_matrix
    @testset "Test de confusion_matrix con datos complejos" begin
        cm = confusionMatrix(outputs_complejos, targets_complejos)
        @test cm[8] == [2 0 1; 0 1 0; 0 1 0]
    end

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end

@testset "Test de la práctica" begin
    # Datos de prueba
    outputs_ponderados = Bool[true false true; false true false; true false true]
    targets_ponderados = Bool[true true false; false false true; true false true]

    # Test de la función confusion_matrix
    @testset "Test de confusion_matrix con datos ponderados" begin
        cm = confusionMatrix(outputs_ponderados, targets_ponderados)
        @test cm[8] == [2 0 1; 0 1 0; 0 1 0]
    end

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end

@testset "Test"