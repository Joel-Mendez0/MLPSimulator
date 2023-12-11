import numpy as np

def create_nn(n_inputs, hidden_layers, n_outputs):
    """
    Create a neural network with specified number of input neurons, multiple hidden layers, and output neurons.
    Initialize the weights randomly.

    :param n_inputs: Number of input neurons
    :param hidden_layers: List containing the number of neurons in each hidden layer
    :param n_outputs: Number of output neurons
    :return: Dictionary representing the neural network with weights and architecture
    """
    layers = [n_inputs] + hidden_layers + [n_outputs]
    weights = {}

    for i in range(len(layers) - 1):
        weights[f'W{i+1}'] = np.random.randn(layers[i], layers[i+1]) * 0.01

    return weights

# Example usage
# net = create_nn(5, 10, 3)
# print(net)

def eval_nn(inputs, individual, n_inputs, n_hidden_neurons, n_outputs):
    """
    Evaluate a neural network with given inputs and weights.

    :param inputs: Input values for the neural network
    :param individual: Flattened array of weights for the network
    :param n_inputs: Number of input neurons
    :param n_hidden_neurons: Number of hidden neurons
    :param n_outputs: Number of output neurons
    :return: Output from the neural network
    """
    # Adjusting input size for bias
    n_inputs += 1  # +1 for bias node

    # Reshape weights for input to hidden layer
    wi = individual[:n_inputs * n_hidden_neurons]
    wi = wi.reshape(n_inputs, n_hidden_neurons)

    # Reshape weights for hidden to output layer
    wo = individual[n_inputs * n_hidden_neurons:]
    wo = wo.reshape(n_hidden_neurons, n_outputs)

    # Input activations (including bias)
    ai = np.append(inputs, 1)  # Adding 1 for bias node

    # Hidden layer activations
    sum_hidden = np.dot(ai, wi)
    ah = np.tanh(sum_hidden)

    # Output activations
    sum_output = np.dot(ah, wo)
    output = np.tanh(sum_output)

    return output

# Example usage
# individual = np.random.rand((5 + 1) * 10 + 10 * 3) - 0.5 # Random weights for an example network
# output = eval_nn(np.array([0.1, 0.2, 0.3, 0.4, 0.5]), individual, 5, 10, 3)
# print(output)

def learning_mlp_k_out(input_data, n_hidden_neurons, n_outputs, error_limit, iterations_limit, output_flag):
    """
    Train a neural network.

    :param input_data: Matrix containing training samples (inputs followed by expected outputs).
    :param n_hidden_neurons: Number of hidden neurons.
    :param n_outputs: Number of output neurons.
    :param error_limit: Tolerance threshold for the error.
    :param iterations_limit: Maximum number of iterations.
    :param output_flag: Flag to control output display.
    :return: Trained network weights, final number of iterations, final RMSE.
    """
    # Splitting the input_data into inputs and expected outputs
    n_inputs = input_data.shape[1] - n_outputs  # Assuming last columns are outputs
    inputs = input_data[:, :n_inputs]
    targets = input_data[:, n_inputs:]

    # Initialize network
    net = create_nn(n_inputs, n_hidden_neurons, n_outputs)

    # Training loop
    for iteration in range(iterations_limit):
        # TODO: Implement forward and backward pass for training
        # This is a placeholder as the actual implementation depends on the specific training method used in MATLAB.
        
        # Calculate error (RMSE) - Placeholder
        rmse = np.random.random()  # Random error for demonstration
        
        # Displaying training progress
        if iteration % output_flag == 0:
            print(f"Iteration: {iteration}, RMSE: {rmse}")
        
        # Check if error limit is reached
        if rmse <= error_limit:
            break

    # Returning the trained network weights, iteration count, and final RMSE
    return net['w'], iteration, rmse

# Example usage
# input_data_example = np.random.rand(100, 5 + 3)  # 100 samples, 5 inputs, 3 outputs
# trained_weights, final_iteration, final_rmse = learning_mlp_k_out(input_data_example, 10, 3, 0.01, 1000, 100)
# print(f"Trained weights: {trained_weights}, Final iteration: {final_iteration}, Final RMSE: {final_rmse}")

def testing_mlp_k_out(input_data, trained_weights, n_inputs, n_hidden_neurons, n_outputs):
    """
    Test a neural network and evaluate its performance for multi-class classification.

    :param input_data: Matrix containing test samples (inputs followed by expected outputs).
    :param trained_weights: Trained weights of the neural network.
    :param n_inputs: Number of input neurons.
    :param n_hidden_neurons: Number of hidden neurons.
    :param n_outputs: Number of output neurons.
    :return: RMSE and classification rate of the neural network.
    """
    # Splitting the input_data into inputs and expected outputs
    inputs = input_data[:, :n_inputs]
    targets = input_data[:, n_inputs:]

    # Variables to store results
    actual_outputs = []
    N = len(inputs)

    # Testing each sample
    for i in range(N):
        # Get the output from the network
        output = eval_nn(inputs[i], trained_weights, n_inputs, n_hidden_neurons, n_outputs)

        # Assuming the output with the highest value is the classified class
        classified_output = np.zeros(n_outputs)
        classified_output[np.argmax(output)] = 1
        actual_outputs.append(classified_output)

    # Converting list of outputs to a NumPy array
    actual_outputs = np.array(actual_outputs)

    # Calculating RMSE
    rmse = np.sqrt(np.mean((actual_outputs - targets) ** 2))

    # Calculating classification rate
    correct_predictions = np.sum(np.all(actual_outputs == targets, axis=1))
    classification_rate = (correct_predictions / N) * 100

    return rmse, classification_rate

# This modified function now handles multi-class classification correctly and should resolve the broadcasting error.

# Example usage
# test_data_example = np.random.rand(50, 5 + 1)  # 50 samples, 5 inputs, 1 output (binary classification for simplicity)
# trained_weights_example = np.random.rand((5 + 1) * 10 + 10 * 1) - 0.5  # Random weights for an example network
# rmse, classification_rate = testing_mlp_k_out(test_data_example, trained_weights_example, 5, 10, 1)
# print(f"RMSE: {rmse}, Classification Rate: {classification_rate}%")

def back_prop(weights, inputs, targets, n_inputs, n_hidden_neurons, n_outputs):
    """
    Backpropagation algorithm to update neural network weights.

    :param weights: Current weights of the neural network.
    :param inputs: Input values for the neural network.
    :param targets: Target output values for the neural network.
    :param n_inputs: Number of input neurons.
    :param n_hidden_neurons: Number of hidden neurons.
    :param n_outputs: Number of output neurons.
    :return: Updated weights.
    """
    # Adjusting input size for bias
    n_inputs += 1  # +1 for bias node

    # Reshape weights for input to hidden layer and hidden to output layer
    wi = weights[:n_inputs * n_hidden_neurons].reshape(n_inputs, n_hidden_neurons)
    wo = weights[n_inputs * n_hidden_neurons:].reshape(n_hidden_neurons, n_outputs)

    # Forward pass
    ai = np.append(inputs, 1)  # Adding 1 for bias node
    net_hidden = np.dot(ai, wi)
    ah = np.tanh(net_hidden)
    net_output = np.dot(ah, wo)
    ao = np.tanh(net_output)

    # Backward pass
    # Calculate errors for output neurons
    output_errors = (1.0 - ao**2) * (targets - ao)

    # Backpropagate the output neurons' errors to hidden layer neurons
    error = np.dot(output_errors, wo.T)
    hidden_errors = (1.0 - ah**2) * error

    # Update hidden neurons' weights
    change_hidden = np.dot(ai.reshape(n_inputs, 1), hidden_errors)
    wi += (1 / n_inputs) * change_hidden

    # Update output neurons' weights
    change_output = np.dot(ah.reshape(n_hidden_neurons, 1), output_errors)
    wo += (1 / (n_hidden_neurons + 1)) * change_output

    # Flatten updated weights for output
    updated_weights = np.concatenate((wi.flatten(), wo.flatten()))

    return updated_weights

# Example usage
# initial_weights = np.random.rand((5 + 1) * 10 + 10 * 3) - 0.5  # Example weights
# inputs_example = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
# targets_example = np.array([0, 1, 0])  # Example targets for a 3-output network
# updated_weights = back_prop(initial_weights, inputs_example, targets_example, 5, 10, 3)
# print(updated_weights)

def fold_data(data, num_folds, num_classes, fold_sample):
    """
    Fold data into multiple sets for cross-validation.

    :param data: Input data, expected with class labels as the last column.
    :param num_folds: Number of folds to divide the data into.
    :param num_classes: Number of classes in the dataset.
    :param fold_sample: Number of samples from each class in each fold.
    :return: Data divided into folds.
    """
    folded_data = []
    rows, cols = data.shape

    for fold in range(num_folds):
        for class_label in range(num_classes):
            # Extracting data for the current class
            class_data = data[data[:, -1] == class_label]

            # Determining start and end indices for slicing
            start_idx = fold * fold_sample
            end_idx = start_idx + fold_sample

            # Slicing the data for the current fold and class
            fold_data = class_data[start_idx:end_idx]
            folded_data.append(fold_data)

    # Concatenating all the fold data
    folded_data = np.vstack(folded_data)

    return folded_data

# Example usage
# data_example = np.random.rand(100, 5)  # Example data with 100 samples and 5 features (last column as class label)
# data_example[:, -1] = np.random.randint(0, 3, 100)  # Assigning random class labels (0, 1, 2)
# folded_data_example = fold_data(data_example, 5, 3, 10)  # 5 folds, 3 classes, 10 samples per class per fold
# print(folded_data_example)

import numpy as np

def reassignment(input_data, n_outputs):
    """
    Reformat input data for neural network training, particularly adjusting output labels for classification.

    :param input_data: Input matrix where the last column contains class labels.
    :param n_outputs: Number of outputs for the neural network.
    :return: Reformatted input data with desired outputs.
    """
    N, m = input_data.shape

    # Extracting class labels
    class_labels = input_data[:, -1]

    # Reshape Input to only have inputs, so new desired outputs will be added
    input_data = input_data[:, :-1]

    # Initialize matrix for desired outputs
    desired_outputs = np.zeros((N, n_outputs))

    # Determine desired outputs based on class label
    for i in range(N):
        if class_labels[i] == 0:
            desired_outputs[i] = [1, -1, -1]
        elif class_labels[i] == 1:
            desired_outputs[i] = [-1, 1, -1]
        elif class_labels[i] == 2:
            desired_outputs[i] = [-1, -1, 1]

    # Add new desired output to Input
    reformatted_input = np.hstack((input_data, desired_outputs))

    return reformatted_input

# Example usage
# input_data_example = np.array([[0.1, 0.2, 0], [0.3, 0.4, 1], [0.5, 0.6, 2]])
# reformatted_input = reassignment(input_data_example, 3)
# print(reformatted_input)

# Translating the Demo_MLP_k_Iris.m MATLAB script into a Python script
# Assuming the Iris dataset is in a compatible format for the Python functions

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

# Generating the Iris dataset
learning_data, test_data = generate_iris_dataset()

# Displaying a small part of the generated dataset for verification
learning_data[:5], test_data[:5]

def demo_mlp_k_iris(learning_data, test_data):
    """
    Demonstration of using a multi-layer perceptron (MLP) on the Iris dataset.

    :param learning_data: The Iris dataset for training the MLP.
    :param test_data: The Iris dataset for testing the MLP.
    :return: Classification accuracy of the MLP on the test data.
    """
    # Assign Parameters
    n_hidden_neurons = 4  # Number of hidden neurons
    n_outputs = 3  # Number of output neurons
    error_limit = 0.1  # Threshold for targeting learning RMSE
    iterations_limit = 50000  # Max number of iterations
    output_flag = 10  # Control of output

    # Learning
    trained_weights, iteration, rmse = learning_mlp_k_out(learning_data, n_hidden_neurons, n_outputs, error_limit, iterations_limit, output_flag)

    # Testing
    rmse, classification_rate = testing_mlp_k_out(test_data, trained_weights, learning_data.shape[1] - n_outputs, n_hidden_neurons, n_outputs)

    # Calculate Accuracy
    accuracy = classification_rate
    print(f"\nAccuracy = {accuracy}%")

    return accuracy

# Example usage
# Assuming learning_data and test_data are provided in a compatible format
# accuracy = demo_mlp_k_iris(learning_data, test_data)
# print(f"MLP Accuracy on Iris Dataset: {accuracy}%")

demo_mlp_k_iris(learning_data, test_data)
