# save_model.py
import numpy as np
import pandas as pd

# Initialize neural network parameters
def initialize_parameters():
    hidden_layer_weights = np.random.rand(10, 784) - 0.5
    hidden_layer_biases = np.random.rand(10, 1) - 0.5
    output_layer_weights = np.random.rand(10, 10) - 0.5
    output_layer_biases = np.random.rand(10, 1) - 0.5
    return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases

# Activation functions
def relu(linear_output):
    return np.maximum(linear_output, 0)

def softmax(linear_output):
    exp_output = np.exp(linear_output - np.max(linear_output, axis=0, keepdims=True))
    return exp_output / np.sum(exp_output, axis=0, keepdims=True)

# Forward propagation
def forward_propagation(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, features):
    hidden_layer_linear_output = hidden_layer_weights.dot(features) + hidden_layer_biases
    hidden_layer_activation = relu(hidden_layer_linear_output)
    output_layer_linear_output = output_layer_weights.dot(hidden_layer_activation) + output_layer_biases
    output_layer_activation = softmax(output_layer_linear_output)
    return hidden_layer_linear_output, hidden_layer_activation, output_layer_linear_output, output_layer_activation

def relu_derivative(linear_output):
    return linear_output > 0

def one_hot_encode(labels):
    one_hot_matrix = np.zeros((labels.size, labels.max() + 1))
    one_hot_matrix[np.arange(labels.size), labels] = 1
    return one_hot_matrix.T

def backward_propagation(hidden_layer_linear_output, hidden_layer_activation, output_layer_activation, output_layer_weights, features, labels, num_training_samples):
    one_hot_labels = one_hot_encode(labels)
    output_error = output_layer_activation - one_hot_labels
    gradient_output_layer_weights = 1 / num_training_samples * output_error.dot(hidden_layer_activation.T)
    gradient_output_layer_biases = 1 / num_training_samples * np.sum(output_error, axis=1, keepdims=True)
    hidden_error = output_layer_weights.T.dot(output_error) * relu_derivative(hidden_layer_linear_output)
    gradient_hidden_layer_weights = 1 / num_training_samples * hidden_error.dot(features.T)
    gradient_hidden_layer_biases = 1 / num_training_samples * np.sum(hidden_error, axis=1, keepdims=True)
    return gradient_hidden_layer_weights, gradient_hidden_layer_biases, gradient_output_layer_weights, gradient_output_layer_biases

def update_parameters(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, gradient_hidden_layer_weights, gradient_hidden_layer_biases, gradient_output_layer_weights, gradient_output_layer_biases, learning_rate):
    hidden_layer_weights -= learning_rate * gradient_hidden_layer_weights
    hidden_layer_biases -= learning_rate * gradient_hidden_layer_biases
    output_layer_weights -= learning_rate * gradient_output_layer_weights
    output_layer_biases -= learning_rate * gradient_output_layer_biases
    return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases

def get_predictions(output_layer_activation):
    return np.argmax(output_layer_activation, axis=0)

def calculate_accuracy(predictions, labels):
    return np.sum(predictions == labels) / labels.size

def gradient_descent(features, labels, learning_rate, iterations, num_training_samples):
    hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases = initialize_parameters()
    for i in range(iterations):
        hidden_layer_linear_output, hidden_layer_activation, _, output_layer_activation = forward_propagation(
            hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, features)
        gradient_hidden_layer_weights, gradient_hidden_layer_biases, gradient_output_layer_weights, gradient_output_layer_biases = backward_propagation(
            hidden_layer_linear_output, hidden_layer_activation, output_layer_activation,
            output_layer_weights, features, labels, num_training_samples)
        hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases = update_parameters(
            hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, gradient_hidden_layer_weights,
            gradient_hidden_layer_biases, gradient_output_layer_weights, gradient_output_layer_biases, learning_rate)
        if i % 50 == 0:
            predictions = get_predictions(output_layer_activation)
            print("Iteration:", i, "Accuracy:", calculate_accuracy(predictions, labels))
    return hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases


# --- Main script execution part ---
print("Loading data from train.csv...")
data = pd.read_csv('train.csv')  
data = np.array(data)
num_samples, num_features = data.shape
np.random.shuffle(data)

training_data = data.T
training_labels = training_data[0]
training_features = training_data[1:num_features] / 255.
_, num_training_samples = training_features.shape

print("Starting model training (this might take a minute)...")
W1, b1, W2, b2 = gradient_descent(training_features, training_labels, 0.10, 500, num_training_samples)

print("Training finished. Saving parameters to model_parameters.npz")
np.savez('model_parameters.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Step 1 Complete! You now have a saved model.")