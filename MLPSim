import numpy as np

def backprop(weights, inputs, targets, ninputs, nhiddenneurons, noutputs):
    ninputs += 1  # +1 for bias node
    N = len(weights)
    
    wi = weights[:ninputs * nhiddenneurons].reshape((ninputs, nhiddenneurons))
    wo = weights[ninputs * nhiddenneurons:N].reshape((nhiddenneurons, noutputs))
    
    ai = np.append(inputs, 1)  # 1 for bias node
    
    netW = np.dot(ai, wi)
    ah = np.tanh(netW)
    
    netW = np.dot(ah, wo)
    ao = np.tanh(netW)
    
    output_errors = (1.0 - ao ** 2) * (targets - ao)
    error = np.dot(output_errors, wo.T)
    hidden_errors = (1.0 - ah ** 2) * error
    
    change = np.outer(ai, hidden_errors)
    wi += (1 / ninputs) * change
    
    netW = np.dot(ai, wi)
    ah = np.tanh(netW)
    
    change = np.outer(ah, output_errors)
    wo += (1 / (nhiddenneurons + 1)) * change
    
    weights = np.concatenate((wi.flatten(), wo.flatten()))
    
    return weights

def create_nn(ninputs, nhiddenneurons, noutputs):
    wsize = ((ninputs + 1) * nhiddenneurons) + (nhiddenneurons * noutputs)  # +1 for bias
    np.random.seed(42)  # for reproducibility
    weights = np.random.rand(1, wsize) - 0.5
    return {'w': weights, 'ni': ninputs, 'nh': nhiddenneurons, 'no': noutputs}

def eval_nn(inputs, individual, ninputs, nhiddenneurons, noutputs):
    ninputs += 1
    
    wi = individual[:ninputs * nhiddenneurons].reshape((ninputs, nhiddenneurons))
    wo = individual[ninputs * nhiddenneurons:].reshape((nhiddenneurons, noutputs))
    
    ai = np.append(inputs, 1)
    
    sum_wi = np.dot(ai, wi)
    ah = np.tanh(sum_wi)
    
    sum_wo = np.dot(ah, wo)
    output = np.tanh(sum_wo)
    
    return output

def learning_ml_pk_out(input_data, nhiddenneurons, noutputs, error_limit, iterations_limit, output_flag):
    A = input_data.copy()
    
    N, ninputs = A.shape
    ninputs -= noutputs
    
    actual_outputs = np.zeros((noutputs, N))
    
    inputs = A[:, :ninputs]
    targets = A[:, ninputs:]
    
    wsize = ((ninputs + 1) * nhiddenneurons) + (nhiddenneurons * noutputs)
    
    net = create_nn(ninputs, nhiddenneurons, noutputs)
    netw = net['w'].flatten()
    
    RMSE = 10
    iteration = 0
    
    while iteration <= iterations_limit and RMSE > error_limit:
        iteration += 1
        
        for j in range(N):
            output = eval_nn(inputs[j, :], netw, ninputs, nhiddenneurons, noutputs)
            actual_outputs[:, j] = output
        
        error = np.sum((actual_outputs - targets.T) ** 2) / noutputs
        error1 = np.sum(error) / N
        RMSE = np.sqrt(error1)
        
        if iteration % output_flag == 0:
            print(f'Iteration {iteration}  Error {RMSE}')
        
        if RMSE <= error_limit:
            break
        
        for j in range(N):
            netw = backprop(netw, inputs[j, :], targets[j, :], ninputs, nhiddenneurons, noutputs)
    
    print(f'Iterations = {iteration}')
    
    for j in range(N):
        output = eval_nn(inputs[j, :], netw, ninputs, nhiddenneurons, noutputs)
        actual_outputs[:, j] = output
    
    return netw, iteration, RMSE

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

def testing_ml_pk_out(input_data, nhiddenneurons, noutputs, netw):
    noutputs = 3
    N, ninputs = input_data.shape
    
    targets = input_data[:, -3:]
    
    for i in range(N):
        if np.array_equal(targets[i, :], [1, -1, -1]):
            targets[i, :] = [1, 0, 0]
        elif np.array_equal(targets[i, :], [-1, 1, -1]):
            targets[i, :] = [0, 1, 0]
        elif np.array_equal(targets[i, :], [-1, -1, 1]):
            targets[i, :] = [0, 0, 1]
    
    targets = np.argmax(targets, axis=1)  # Convert one-hot encoding to class labels
    ninputs -= 3
    
    inputs = input_data[:, :ninputs]
    actual_outputs = np.zeros(N)
    
    for j in range(N):
        output = eval_nn(inputs[j, :], netw, ninputs, nhiddenneurons, noutputs)
        output_distance = np.abs(output - 1)
        ind_output_neuron = np.argmin(output_distance)
        actual_outputs[j] = ind_output_neuron
    
    error = np.sum((actual_outputs - targets) ** 2) / N
    RMSE = np.sqrt(error)
    
    results = (actual_outputs == targets)
    num_of_correct_outputs = np.sum(results)
    classif_rate = (num_of_correct_outputs / N) * 100
    
    print(f'Classification Rate = {classif_rate}')
    
    return classif_rate

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
mlp_1 = scipy.io.loadmat('C://Users//Joel//Desktop//MLPSimulator//MLP_1.mat')['MLP_1']
mlp_2 = scipy.io.loadmat('C://Users//Joel//Desktop//MLPSimulator//MLP_2.mat')['MLP_2']
mlp_3 = scipy.io.loadmat('C://Users//Joel//Desktop//MLPSimulator//MLP_3.mat')['MLP_3']
mlp_4 = scipy.io.loadmat('C://Users//Joel//Desktop//MLPSimulator//MLP_4.mat')['MLP_4']

# Combine data from MLP_1 to MLP_4 into one learning set
learning_set = np.concatenate((mlp_1, mlp_2, mlp_3, mlp_4), axis=0)

print(learning_set.shape)

# Save the learning set to a new MATLAB file
scipy.io.savemat('learning_set.mat', {'learning_set': learning_set})

# Load data from MLP_5 as the test set
test_set = scipy.io.loadmat('C://Users//Joel//Desktop//MLPSimulator//MLP_5.mat')['MLP_5']


# Save the test set to a new MATLAB file
scipy.io.savemat('test_set.mat', {'test_set': test_set})

# Example usage:
# Load your data here and preprocess it as needed
# Then, call the functions accordingly

# For example:
noutputs = 3
nhiddenneurons = 4
error_limit = 0.1
iterations_limit = 100
output_flag = 10

#netw, iteration, RMSE = learning_ml_pk_out(learning_set, nhiddenneurons, noutputs, error_limit, iterations_limit, output_flag)
#classif_rate = testing_ml_pk_out(test_set, nhiddenneurons, noutputs, netw)


# Generating the Iris dataset
learning_data, test_data = generate_iris_dataset()

# Displaying a small part of the generated dataset for verification
learning_data[:5], test_data[:5]

netw, iteration, RMSE = learning_ml_pk_out(learning_data, nhiddenneurons, noutputs, error_limit, iterations_limit, output_flag)
classif_rate = testing_ml_pk_out(test_set, nhiddenneurons, noutputs, netw)
