import numpy as np
import matplotlib.pyplot as plt

def backpropagation(network_weights, input_values, target_values, neural_layers, rate_learning=0.01):
    # Forward Pass
    layer_activations = [np.append(input_values, 1)]  # Include bias in inputs
    for layer_index in range(len(neural_layers) - 1):
        layer_input = np.dot(layer_activations[-1], network_weights[f'Layer{layer_index}'])
        layer_output = np.tanh(layer_input)
        layer_output = np.append(layer_output, 1) if layer_index < len(neural_layers) - 2 else layer_output
        layer_activations.append(layer_output)

    # Compute Output Error
    output_errors = (1.0 - layer_activations[-1] ** 2) * (target_values - layer_activations[-1])

    # Backward Pass
    for layer_index in reversed(range(len(neural_layers) - 1)):
        if layer_index != len(neural_layers) - 2:  # Not the output layer
            layer_error = (1.0 - layer_activations[layer_index + 1][:-1] ** 2) * np.dot(output_errors, network_weights[f'Layer{layer_index+1}'].T)[:-1]
        else:
            layer_error = output_errors

        # Compute Gradient and Update Weights
        weight_change = np.outer(layer_activations[layer_index], layer_error)
        network_weights[f'Layer{layer_index}'] += rate_learning * weight_change

        # Update output_errors for next iteration
        output_errors = layer_error

    return network_weights

def initialize_nn(neural_layers):
    network_weights = {}
    for layer_index in range(len(neural_layers) - 1):
        network_weights[f'Layer{layer_index}'] = np.random.rand(neural_layers[layer_index] + 1, neural_layers[layer_index + 1]) - 0.5  # +1 for bias
    return network_weights

def evaluate_nn(input_values, network_weights):
    layer_output = np.append(input_values, 1)  # Adding bias term
    for layer_index in range(len(network_weights)):
        layer_input = np.dot(layer_output, network_weights[f'Layer{layer_index}'])
        layer_output = np.tanh(layer_input)
        layer_output = np.append(layer_output, 1)  # Adding bias term for next layer
    return layer_output[:-1]  # Remove the last bias term added

def train_nn(training_data, neural_layers, limit_error, limit_iterations, flag_output):
    num_samples, num_features = training_data.shape
    num_inputs = num_features - neural_layers[-1]  # Number of input features minus number of output neurons

    inputs = training_data[:, :num_inputs]
    targets = training_data[:, num_inputs:]

    network_weights = initialize_nn(neural_layers)

    RMSE = float('inf')
    current_iteration = 0
    iteration_errors = []  # Store errors for plotting

    while current_iteration < limit_iterations and RMSE > limit_error:
        current_iteration += 1
        total_error = 0

        for sample_index in range(num_samples):
            predicted_output = evaluate_nn(inputs[sample_index, :], network_weights)
            network_weights = backpropagation(network_weights, inputs[sample_index, :], targets[sample_index, :], neural_layers)
            total_error += np.sum((predicted_output - targets[sample_index, :]) ** 2)

        RMSE = np.sqrt(total_error / (num_samples * neural_layers[-1]))
        iteration_errors.append(RMSE)  # Store the error for plotting

        if current_iteration % flag_output == 0:
            print(f'Iteration {current_iteration}, Error {RMSE}')

    # Plot learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, current_iteration + 1), iteration_errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.show()

    return network_weights, current_iteration, RMSE

def reassignment(input_data, noutputs):
    N, m = input_data.shape
    class_labels = input_data[:, m - 1]
    
    input_data = input_data[:, :-1]
    desired_outputs = np.zeros((N, noutputs))
        
    input_data = np.hstack((input_data, desired_outputs))
    
    return input_data

def evaluate_model_performance(test_dataset, network_weights, neural_layers):
    num_input_neurons = neural_layers[0]
    num_samples = test_dataset.shape[0]

    input_features = test_dataset[:, :num_input_neurons]
    target_values = test_dataset[:, num_input_neurons:]

    count_correct_predictions = 0
    predicted_classes = []

    for sample_index in range(num_samples):
        model_output = evaluate_nn(input_features[sample_index, :], network_weights)
        predicted_label = np.argmax(model_output)
        actual_label = np.argmax(target_values[sample_index, :])
        predicted_classes.append(predicted_label)

        if predicted_label == actual_label:
            count_correct_predictions += 1

    accuracy_classification = (count_correct_predictions / num_samples) * 100
    print(f'Classification Accuracy = {accuracy_classification}%')

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(target_values.argmax(axis=1), 'or', label='True Classes')
    plt.plot(predicted_classes, '*b', label='Predicted Classes')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend()
    plt.title('True vs Predicted Classes')
    plt.show()

    return accuracy_classification

import scipy.io

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

# Only change the middle indices corresponding to number of neurons in the hidden layers. Can add as many layers as you like
layer_sizes = [4,4,32,3]


weights, iteration, RMSE = train_nn(learning_set, layer_sizes, error_limit, iterations_limit, output_flag)
classif_rate = evaluate_model_performance(test_set, weights, layer_sizes)
