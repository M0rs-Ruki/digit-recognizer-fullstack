# prediction.py
import numpy as np

# Load the parameters from the file
params = np.load('model_parameters.npz')
W1 = params['W1']
b1 = params['b1']
W2 = params['W2']
b2 = params['b2']

# --- These are the necessary functions from your training script ---
def relu(linear_output):
    return np.maximum(linear_output, 0)

def softmax(linear_output):
    exp_output = np.exp(linear_output - np.max(linear_output, axis=0, keepdims=True))
    return exp_output / np.sum(exp_output, axis=0, keepdims=True)

def forward_propagation(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, features):
    hidden_layer_linear_output = hidden_layer_weights.dot(features) + hidden_layer_biases
    hidden_layer_activation = relu(hidden_layer_linear_output)
    output_layer_linear_output = output_layer_weights.dot(hidden_layer_activation) + output_layer_biases
    output_layer_activation = softmax(output_layer_linear_output)
    return output_layer_activation

def get_predictions(output_layer_activation):
    return np.argmax(output_layer_activation, axis=0)

# This is our main prediction function
def make_prediction(image_data):
    # The image_data will be a flattened 784x1 numpy array from our server
    # Run it through the network
    A2 = forward_propagation(W1, b1, W2, b2, image_data)
    # Get the prediction
    prediction = get_predictions(A2)
    return prediction