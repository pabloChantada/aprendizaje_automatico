using Test
include("35634619Y_48114048A_32740686W_48111913F.jl")
@testset "modelCrossValidation tests" begin
    # Test case for ANNCrossValidation
    #=
    @testset "ANNCrossValidation" begin
        topology = [2, 2, 1]
        inputs = [0 0; 0 1; 1 0; 1 1]
        targets = [1, 0, 0, 0, 1, 1, 0]
        crossValidationIndices = [1, 2]

        results = modelCrossValidation(:ANN, Dict("topology" => topology), inputs, targets, crossValidationIndices)

        @test length(results) == 7
        @test all(length(r) == 2 for r in results)
        @test !isnan(results[1][1])
        @test !isnan(results[2][1])
        @test !isnan(results[3][1])
        @test !isnan(results[4][1])
        @test !isnan(results[5][1])
        @test !isnan(results[6][1])
        @test !isnan(results[7][1])
        @test !isnan(results[1][2])
        @test !isnan(results[2][2])
        @test !isnan(results[3][2])
        @test !isnan(results[4][2])
        @test !isnan(results[5][2])
        @test !isnan(results[6][2])
        @test !isnan(results[7][2])
        @test results[1][2] >= 0
        @test results[2][2] >= 0
        @test results[3][2] >= 0
        @test results[4][2] >= 0
        @test results[5][2] >= 0
        @test results[6][2] >= 0
        @test results[7][2] >= 0
    end
    =#
    # Test case for other models
    @testset "Other models" begin
        inputs = [1 2; 3 4; 5 6; 7 8]
        targets = [1, 0, 1, 0]
        crossValidationIndices = [1, 2]

        # Test case for SVC
        @testset "SVC" begin
            modelHyperparameters = Dict("C" => 1.0, "kernel" => "linear", "degree" => 3, "gamma" => "auto", "coef0" => 0.0)
            results = modelCrossValidation(:SVC, modelHyperparameters, inputs, targets, crossValidationIndices)

            # Add assertions for the results of SVC model
        end

        # Test case for DecisionTreeClassifier
        @testset "DecisionTreeClassifier" begin
            modelHyperparameters = Dict("max_depth" => 3)
            results = modelCrossValidation(:DecisionTreeClassifier, modelHyperparameters, inputs, targets, crossValidationIndices)

            # Add assertions for the results of DecisionTreeClassifier model
        end

        # Test case for KNeighborsClassifier
        @testset "KNeighborsClassifier" begin
            modelHyperparameters = Dict("n_neighbors" => 3)
            results = modelCrossValidation(:KNeighborsClassifier, modelHyperparameters, inputs, targets, crossValidationIndices)

            # Add assertions for the results of KNeighborsClassifier model
        end
    end
end


    # Convertimos el vector de salidas deseada a texto para evitar errores con la librería de Python
    targets = string.(targets)

    # Creamos vectores para almacenar los resultados de las métricas en cada fold
    acc, fail_rate, sensitivity, specificity, VPP, VPN, F1 = [], [], [], [], [], [], []
    # Comenzamos la validación cruzada
    # for fold in unique(crossValidationIndices)
    fold = unique(crossValidationIndices)[1]
        #for test_indices in crossValidationIndices
            # Obtenemos los índices de entrenamiento
            train_indices = filter(x -> !(x in fold), 1:size(inputs, 1))
            # Convertimos el rango en un vector de índices
            test_indices = collect(fold)
        # Dividimos los datos en entrenamiento y prueba
        train_inputs = inputs[train_indices, :]
        train_targets = targets[train_indices]
        test_inputs = inputs[test_indices, :]
        test_targets = targets[test_index]

        # Creamos el modelo según el tipo especificado
        # model = nothing
        if modelType == :SVC
            model = SVC(C=modelHyperparameters["C"], kernel=modelHyperparameters["kernel"],
                        degree=modelHyperparameters["degree"], gamma=modelHyperparameters["gamma"],
                        coef0=modelHyperparameters["coef0"])
        elseif modelType == :DecisionTreeClassifier
            model = DecisionTreeClassifier(max_depth=modelHyperparameters["max_depth"])
        elseif modelType == :KNeighborsClassifier
            model = KNeighborsClassifier(n_neighbors=modelHyperparameters["n_neighbors"])
        end

        # Entrenamos el modelo
        model = fit!(model, train_inputs, train_targets)
        # Problema aqui
        predictions = predict(model, reshape(test_inputs, 1, :))
        # ni puta idea de que es un array{String, 0} tbh
        println(predictions)
        println(test_targets)
        metrics = confusionMatrix(vec(test_targets), vec(predictions))
        push!(acc, metrics[1])
        push!(fail_rate, metrics[2])
        push!(sensitivity, metrics[3])
        push!(specificity, metrics[4])
        push!(VPP, metrics[5])
        push!(VPN, metrics[6])
        push!(F1, metrics[7])
    end
    # Devolvemos los resultados como una tupla de tuplas
    return ((mean(acc), std(acc)), 
            (mean(fail_rate), std(fail_rate)),
            (mean(sensitivity), std(sensitivity)), 
            (mean(specificity), std(specificity)),
            (mean(VPP), std(VPP)), 
            (mean(VPN), std(VPN)), 
            (mean(F1), std(F1)))
end