using Flux  # Asegúrate de tener Flux instalado para redes neuronales
using Random  # Para la generación de números aleatorios
include("35634619Y_48114048A_32740686W_48111913F.jl")
# Generar datos sintéticos para clasificación binaria

using Test

# Test para el caso binario con predicciones perfectas
@testset "Confusion Matrix Tests" begin
    @testset "Binary Classification - Perfect Predictions" begin
        targets = [1; 1; 0]
        outputs = [1; 1; 0]
        acc, fail_rate, sensitivities, specificities, ppvs, npvs, f1s, confusion_matrix = confusionMatrix(outputs, targets; weighted=false)
        @test acc == 1.0
        @test fail_rate == 0.0
        @test sensitivities == 1.0
        @test specificities == 1.0
        @test ppvs == 1.0
        @test npvs == 1.0
        @test f1s == 1.0
        @test confusion_matrix == [1 0; 0 2]
    end

    # Test para el caso multiclase con predicciones mixtas
    @testset "Multiclass Classification - Mixed Predictions" begin
        targets = Flux.onehotbatch([1, 2, 3], 1:3)
        outputs = Flux.onehotbatch([1, 3, 2], 1:3)
        acc, fail_rate, _, _, _, _, _, confusion_matrix = confusionMatrix(outputs, targets; weighted=true)
        @test acc < 1.0
        @test fail_rate > 0.0
        @test confusion_matrix == [1 0 0;
                                    0 0 1;
                                    0 1 0]
       
    end
end


@testset "Multiclass Classification - Additional Tests" begin
    # Prueba con Predicciones Completamente Correctas
    targets_correct = Flux.onehotbatch([1, 2, 3], 1:3)
    outputs_correct = Flux.onehotbatch([1, 2, 3], 1:3)
    acc_correct, fail_rate_correct, _, _, _, _, _, confusion_matrix_correct = confusionMatrix(outputs_correct, targets_correct; weighted=true)
    @test acc_correct == 1.0
    @test fail_rate_correct == 0.0
    @test confusion_matrix_correct == [1 0 0; 0 1 0; 0 0 1]
    
    # Prueba con Predicciones Parcialmente Correctas (Otro Caso)
    targets_partial = Flux.onehotbatch([3, 1, 2], 1:3)
    outputs_partial = Flux.onehotbatch([2, 1, 3], 1:3)
    acc_partial, fail_rate_partial, _, _, _, _, _, confusion_matrix_partial = confusionMatrix(outputs_partial, targets_partial; weighted=true)
    @test acc_partial < 1.0
    @test fail_rate_partial > 0.0
    @test confusion_matrix_partial == [0 0 1; 0 1 0; 1 0 0]
end

using Test
# Asumiendo que la función confusion_matrix ya está definida.

@testset "Confusion Matrix Type Tests" begin
    @testset "With weighted = $weighted" for weighted in [true, false]
        # Test para matrices OneHot (booleanas)
        targets_boolean = Flux.onehotbatch([1, 2, 3], 1:3)
        outputs_boolean = Flux.onehotbatch([3, 1, 2], 1:3)
        @test let (_,_,_,_,_,_,_, confusion_matrix) = confusion_matrix(targets_boolean, outputs_boolean, weighted=weighted)
            println("Confusion Matrix", confusion_matrix)
        end

        # Test para matrices de valores reales
        targets_real = [0.1 0.9 0.0; 0.8 0.2 0.0; 0.3 0.4 0.3]
        outputs_real = [0.2 0.7 0.1; 0.9 0.1 0.0; 0.2 0.5 0.3]
        @test let (_,_,_,_,_,_,_, confusion_matrix) = confusion_matrix(targets_real, outputs_real, weighted=weighted)
            println("Confusion Matrix", confusion_matrix)

        end

        # Test para vectores de tipo Any
        targets_any = ["a", "b", "c"]
        outputs_any = ["c", "a", "b"]
        @test let (_,_,_,_,_,_,_, confusion_matrix) = confusion_matrix(targets_any, outputs_any, weighted=weighted)
            println("Confusion Matrix", confusion_matrix)

        end
    end
end

using Test;
include("35634619Y_48114048A_32740686W_48111913F.jl")

modelType = :SVC  # Or :DecisionTreeClassifier, :KNeighborsClassifier
modelHyperparameters = Dict(
    "C" => 1.0, "kernel" => "rbf", "gamma" => 0.1, "degree" => 3, "coef0" => 0.0
    # Add other hyperparameters as needed
)
inputs, targets = (rand(100, 4), rand(100) .> 0.5)
# Assume inputs and targets are predefined
crossValidationIndices = [1, 1, 2, 2]  # Example indices for 2-fold cross-validation

results = modelCrossValidation(modelType, modelHyperparameters, inputs, targets, crossValidationIndices)

println("Cross-validation results: ", results)
