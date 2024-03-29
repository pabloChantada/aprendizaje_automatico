using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")

@testset "Tests de la práctica" begin
    # Datos de prueba
    outputs = Bool[
        1 0 0; # Clase A
        0 1 0; # Clase B
        0 0 1; # Clase C

        1 0 0; # Clase A
        0 1 0; # Clase B
        1 0 0; # Clase A incorrectamente clasificada como Clase A (debería ser Clase C)

        0 0 1; # Clase C
        0 0 1; # Clase C
        0 1 0; # Clase B incorrectamente clasificada como Clase B (debería ser Clase A)

        0 1 0  # Clase B
    ]
    targets = Bool[
        1 0 0;
        0 1 0;
        0 0 1;

        1 0 0;
        0 1 0;
        0 0 1; # Clase C correcta

        0 0 1;
        0 0 1;
        1 0 0; # Clase A correcta

        0 1 0
    ]

    # Resultados esperados
    expected_matrix = [2 0 1;
                        1 3 0;
                        0 0 3]

    expected_accuracy = 8 / 10 # 70% de precisión
    expected_fail_rate = 0.2

    # Ejecutar el test
    acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, matrix = confusionMatrix(outputs, targets, weighted=false)

    @test matrix == expected_matrix
    @test acc ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivities ≈ mean([0.6666666666666666, 1, 0.75])
    @test specificities ≈ mean([0.8571428571428571, 1.00, 0.8571428571428571])
    @test ppvs ≈ mean([0.6666666666666666, 0.75, 1.00])
    @test f1s ≈ mean([0.6666666666666666, 0.8571428571428571, 0.8571428571428571])

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end

@testset "Tests de la práctica" begin
    # Datos de prueba
    outputs = Bool[
        1 0 0; # Clase A
        0 1 0; # Clase B
        0 0 1; # Clase C

        1 0 0; # Clase A
        0 1 0; # Clase B
        1 0 0; # Clase A incorrectamente clasificada como Clase A (debería ser Clase C)

        0 0 1; # Clase C
        0 0 1; # Clase C
        0 1 0; # Clase B incorrectamente clasificada como Clase B (debería ser Clase A)

        0 1 0  # Clase B
    ]
    targets = Bool[
        1 0 0;
        0 1 0;
        0 0 1;

        1 0 0;
        0 1 0;
        0 0 1; # Clase C correcta

        0 0 1;
        0 0 1;
        1 0 0; # Clase A correcta

        0 1 0
    ]

    # Resultados esperados
    expected_matrix = [2 0 1;
                        1 3 0;
                        0 0 3]

    expected_accuracy = 8 / 10 # 70% de precisión
    expected_fail_rate = 0.2

    # Ejecutar el test
    acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, matrix = confusionMatrix(outputs, targets)

    # Si falla es por el calculo de las ponderadas
    @test matrix == expected_matrix
    @test acc ≈ expected_accuracy
    @test fail_rate ≈ expected_fail_rate
    @test sensitivities ≈ 0.8
    @test specificities ≈ 0.9142857142857143
    @test ppvs ≈ 0.825
    @test f1s ≈ 0.7999999999999999

    # Para simplificar, no incluimos tests para sensibilidad, especificidad, PPV, NPV, y F1 debido a que requieren un cálculo detallado para cada clase
end