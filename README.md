# Multi-Layer Perceptron (MLP) Project
This project implements a Multi-Layer Perceptron (MLP) neural network with support for multiple hidden layers and output neurons. The neural network is trained using the backpropagation algorithm. The code includes functions for network initialization, forward pass, backward pass, training, and evaluation.

Requirements
numpy
matplotlib
scipy
Usage
python
Copy code
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Define the MLP project functions

# Load data from MLP_1 to MLP_4
mlp_1 = scipy.io.loadmat('MLP_1.mat')['MLP_1']
mlp_2 = scipy.io.loadmat('MLP_2.mat')['MLP_2']
mlp_3 = scipy.io.loadmat('MLP_3.mat')['MLP_3']
mlp_4 = scipy.io.loadmat('MLP_4.mat')['MLP_4']

# Combine data from MLP_1 to MLP_4 into one learning set
learning_set = np.concatenate((mlp_1, mlp_2, mlp_3, mlp_4), axis=0)

print(learning_set.shape)

# Save the learning set to a new MATLAB file
scipy.io.savemat('learning_set.mat', {'learning_set': learning_set})

# Load data from MLP_5 as the test set
test_set = scipy.io.loadmat('MLP_5.mat')['MLP_5']

# Save the test set to a new MATLAB file
scipy.io.savemat('test_set.mat', {'test_set': test_set})

error_limit = 0.1
iterations_limit = 100
output_flag = 10

# Only change the middle indices corresponding to the number of neurons in the hidden layers. Can add as many layers as you like
layer_sizes = [4, 4, 32, 3]

weights, iteration, RMSE = train_nn(learning_set, layer_sizes, error_limit, iterations_limit, output_flag)
classif_rate = evaluate_model_performance(test_set, weights, layer_sizes)
Functions
backpropagation(network_weights, input_values, target_values, neural_layers, rate_learning=0.01)
Performs backpropagation to update the network weights based on the given input values, target values, and neural layer configuration.
initialize_nn(neural_layers)
Initializes the neural network weights based on the given neural layer configuration.
evaluate_nn(input_values, network_weights)
Evaluates the neural network output for a given set of input values and weights.
train_nn(training_data, neural_layers, limit_error, limit_iterations, flag_output)
Trains the neural network using the provided training data, neural layer configuration, error limit, iteration limit, and output flag.
reassignment(input_data, noutputs)
Modifies the input data to include space for desired outputs.
evaluate_model_performance(test_dataset, network_weights, neural_layers)
Evaluates the performance of the trained model on a test dataset, including classification accuracy and a plot of true vs predicted classes.
