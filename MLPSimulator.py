import numpy as np

class MultiLayerPerceptron:
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        self.layers = len(hidden_layer_sizes) + 1
        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        for i in range(self.layers):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]))
            self.biases.append(np.random.randn(layer_sizes[i + 1]))

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward_pass(self, x):
        activations = [x]
        for i in range(self.layers - 1):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            activations.append(a)

        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        output = self.softmax(z)
        return output

# Example usage
mlp = MultiLayerPerceptron(input_size=10, hidden_layer_sizes=[20, 15], output_size=5)
input_data = np.random.randn(1, 10)
output = mlp.forward_pass(input_data)
print("Output:", output)
