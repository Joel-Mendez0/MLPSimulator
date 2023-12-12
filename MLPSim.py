import numpy as np

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
    
    while iteration < iterations_limit and RMSE > error_limit:
        iteration += 1
        total_error = 0

        for j in range(N):
            output = eval_nn(inputs[j, :], weights)
            weights = backprop(weights, inputs[j, :], targets[j, :], layer_sizes)
            total_error += np.sum((output - targets[j, :]) ** 2)
        
        RMSE = np.sqrt(total_error / (N * layer_sizes[-1]))

        if iteration % output_flag == 0:
            print(f'Iteration {iteration}, Error {RMSE}')

    return weights, iteration, RMSE

def reassignment(input_data, noutputs):
    N, m = input_data.shape
    class_labels = input_data[:, m - 1]
    
    input_data = input_data[:, :-1]
    desired_outputs = np.zeros((N, noutputs))
    
    for i in range(N):
        if class_labels[i] == 0:
            desired_outputs[i, :] = [1, -1, -1]
        elif class_labels[i] == 1:
            desired_outputs[i, :] = [-1, 1, -1]
        elif class_labels[i] == 2:
            desired_outputs[i, :] = [-1, -1, 1]
    
    input_data = np.hstack((input_data, desired_outputs))
    
    return input_data

def testing_ml_pk_out(input_data, weights, layer_sizes):
    ninputs = layer_sizes[0]
    N = input_data.shape[0]

    inputs = input_data[:, :ninputs]
    targets = input_data[:, ninputs:]

    correct_predictions = 0

    for j in range(N):
        output = eval_nn(inputs[j, :], weights)
        predicted_class = np.argmax(output)
        actual_class = np.argmax(targets[j, :])
        if predicted_class == actual_class:
            correct_predictions += 1

    classification_rate = (correct_predictions / N) * 100
    print(f'Classification Rate = {classification_rate}%')
    
    return classification_rate

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def generate_iris_dataset():
    """
    Generate a compatible Iris dataset for the demo_mlp_k_iris function.
    The dataset will be split into training and testing datasets.
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    X = iris.data  # Input features
    y = iris.target  # Class labels

    # One-hot encoding of the class labels
    encoder = OneHotEncoder(sparse=False)
    y_encoded = encoder.fit_transform(y.reshape(-1, 1))

    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

    # Combine inputs and outputs to match the expected format
    training_data = np.hstack((X_train, y_train))
    testing_data = np.hstack((X_test, y_test))

    return training_data, testing_data
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

layer_sizes = [4, 2, 6, 100, 3]

#netw, iteration, RMSE = learning_ml_pk_out(learning_set, nhiddenneurons, noutputs, error_limit, iterations_limit, output_flag)
#classif_rate = testing_ml_pk_out(test_set, nhiddenneurons, noutputs, netw)

# Generating the Iris dataset
learning_data, test_data = generate_iris_dataset()

# Displaying a small part of the generated dataset for verification
learning_data[:5], test_data[:5]

weights, iteration, RMSE = learning_ml_pk_out(learning_data, layer_sizes, error_limit, iterations_limit, output_flag)
classif_rate = testing_ml_pk_out(test_set, weights, layer_sizes)
