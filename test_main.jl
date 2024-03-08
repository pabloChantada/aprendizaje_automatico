using Test

include("35634619Y_48114048A_32740686W_48111913F.jl")
# Assuming you have defined the trainClassANN function

# Step 1: Prepare your datasets
# Example datasets (replace with your actual data)
training_data = (rand(10, 5), rand(Bool, 10))
validation_data = (rand(5, 5), rand(Bool, 5))
test_data = (rand(5, 5), rand(Bool, 5))
typeof(training_data)
typeof(validation_data)
typeof(test_data)
# Step 2: Define network topology and other parameters
topology = [5, 3, 2]  # Example: 5 input nodes, 3 hidden nodes, 2 output nodes
max_epochs = 1000
min_loss = 0.01
learning_rate = 0.01
max_epochs_val = 20

# Step 3: Call the trainClassANN function
trainClassANN(topology, training_data, validationDataset=validation_data,
              testDataset=test_data, transferFunctions=fill(σ, length(topology)),
              maxEpochs=max_epochs, minLoss=min_loss,
              learningRate=learning_rate, maxEpochsVal=max_epochs_val)
