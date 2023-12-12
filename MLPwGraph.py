import numpy as np
import matplotlib.pyplot as plt

def backprop(weights, inputs, targets, layer_sizes, learning_rate=0.01):
    # Forward Pass
    activations = [np.append(inputs, 1)]  # Include bias in inputs
    for i in range(len(layer_sizes) - 1):
        net_input = np.dot(activations[-1], weights[f'W{i}'])
        activation = np.tanh(net_input)
        activation = np.append(activation, 1) if i < len(layer_sizes) - 2 else activation
        activations.append(activation)

    # Compute Output Error
    output_errors = (1.0 - activations[-1] ** 2) * (targets - activations[-1])

    # Backward Pass
    for i in reversed(range(len(layer_sizes) - 1)):
        if i != len(layer_sizes) - 2:  # Not the output layer
            # Calculate error for hidden layers
            error = (1.0 - activations[i + 1][:-1] ** 2) * np.dot(output_errors, weights[f'W{i+1}'].T)[:-1]
        else:
            # Error for the output layer
            error = output_errors

        # Compute Gradient and Update Weights
        change = np.outer(activations[i], error)
        weights[f'W{i}'] += learning_rate * change

        # Update output_errors for next iteration
        output_errors = error

    return weights

def create_nn(layer_sizes):
    weights = {}
    for i in range(len(layer_sizes) - 1):
        weights[f'W{i}'] = np.random.rand(layer_sizes[i] + 1, layer_sizes[i + 1]) - 0.5  # +1 for bias
    return weights

def eval_nn(inputs, weights):
    activation = np.append(inputs, 1)  # Adding bias term
    for i in range(len(weights)):
        net_input = np.dot(activation, weights[f'W{i}'])
        activation = np.tanh(net_input)
        activation = np.append(activation, 1)  # Adding bias term for next layer
    return activation[:-1]  # Remove the last bias term added

def learning_ml_pk_out(input_data, layer_sizes, error_limit, iterations_limit, output_flag):
    N, nfeatures = input_data.shape
    ninputs = nfeatures - layer_sizes[-1]  # Number of input features minus number of output neurons

    inputs = input_data[:, :ninputs]
    targets = input_data[:, ninputs:]

    weights = create_nn(layer_sizes)

    RMSE = float('inf')
    iteration = 0
    errors = []  # Store errors for plotting

    while iteration < iterations_limit and RMSE > error_limit:
        iteration += 1
        total_error = 0

        for j in range(N):
            output = eval_nn(inputs[j, :], weights)
            weights = backprop(weights, inputs[j, :], targets[j, :], layer_sizes)
            total_error += np.sum((output - targets[j, :]) ** 2)

        RMSE = np.sqrt(total_error / (N * layer_sizes[-1]))
        errors.append(RMSE)  # Store the error for plotting

        if iteration % output_flag == 0:
            print(f'Iteration {iteration}, Error {RMSE}')

    # Plot learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, iteration + 1), errors, marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Root Mean Squared Error')
    plt.title('Learning Curve')
    plt.grid(True)
    plt.show()

    return weights, iteration, RMSE

def reassignment(input_data, noutputs):
    N, m = input_data.shape
    class_labels = input_data[:, m - 1]
    
    input_data = input_data[:, :-1]
    desired_outputs = np.zeros((N, noutputs))
        
    input_data = np.hstack((input_data, desired_outputs))
    
    return input_data

def testing_ml_pk_out(input_data, weights, layer_sizes):
    ninputs = layer_sizes[0]
    N = input_data.shape[0]

    inputs = input_data[:, :ninputs]
    targets = input_data[:, ninputs:]

    correct_predictions = 0
    actual_outputs = []

    for j in range(N):
        output = eval_nn(inputs[j, :], weights)
        predicted_class = np.argmax(output)
        actual_class = np.argmax(targets[j, :])
        actual_outputs.append(predicted_class)

        if predicted_class == actual_class:
            correct_predictions += 1

    classification_rate = (correct_predictions / N) * 100
    print(f'Classification Rate = {classification_rate}%')

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(targets.argmax(axis=1), 'or', label='True Classes')
    plt.plot(actual_outputs, '*b', label='Predicted Classes')
    plt.xlabel('Sample Index')
    plt.ylabel('Class')
    plt.legend()
    plt.title('True vs Predicted Classes')
    plt.show()

    return classification_rate

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

# Example usage:
# Load your data here and preprocess it as needed
# Then, call the functions accordingly

# For example:
error_limit = 0.1
iterations_limit = 100
output_flag = 10

layer_sizes = [4,4,32,3]


weights, iteration, RMSE = learning_ml_pk_out(learning_set, layer_sizes, error_limit, iterations_limit, output_flag)
classif_rate = testing_ml_pk_out(test_set, weights, layer_sizes)
