import os
import numpy as np

# Load the parameters from the file using an absolute path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_this_dir, 'model_parameters.npz')

print(f"Loading model from: {_model_path}")
params = np.load(_model_path)
W1 = params['W1']
b1 = params['b1']
W2 = params['W2']
b2 = params['b2']
print(f"Model loaded - W1 shape: {W1.shape}, W2 shape: {W2.shape}")

def relu(linear_output):
    return np.maximum(linear_output, 0)

def softmax(linear_output):
    exp_output = np.exp(linear_output - np.max(linear_output, axis=0, keepdims=True))
    return exp_output / np.sum(exp_output, axis=0, keepdims=True)

def forward_propagation(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, features):
    print(f"Forward prop - input shape: {features.shape}")
    
    hidden_layer_linear_output = hidden_layer_weights.dot(features) + hidden_layer_biases
    print(f"Hidden linear output shape: {hidden_layer_linear_output.shape}")
    
    hidden_layer_activation = relu(hidden_layer_linear_output)
    print(f"Hidden activation shape: {hidden_layer_activation.shape}")
    
    output_layer_linear_output = output_layer_weights.dot(hidden_layer_activation) + output_layer_biases
    print(f"Output linear shape: {output_layer_linear_output.shape}")
    
    output_layer_activation = softmax(output_layer_linear_output)
    print(f"Final output shape: {output_layer_activation.shape}")
    print(f"Output probabilities: {output_layer_activation.flatten()}")
    
    return output_layer_activation

def get_predictions(output_layer_activation):
    prediction = np.argmax(output_layer_activation, axis=0)
    print(f"Argmax result: {prediction} (shape: {prediction.shape})")
    return prediction

def make_prediction(image_data):
    print(f"Making prediction for image data shape: {image_data.shape}")
    
    # Ensure input is correct shape
    if image_data.shape != (784, 1):
        print(f"ERROR: Expected shape (784, 1), got {image_data.shape}")
        return None
    
    # Check for valid input
    if np.any(np.isnan(image_data)):
        print("ERROR: Input contains NaN values")
        return None
    
    # Run forward propagation
    A2 = forward_propagation(W1, b1, W2, b2, image_data)
    
    # Get prediction
    prediction_array = get_predictions(A2)
    print(f"Prediction array: {prediction_array}")
    
    # Convert to scalar int
    if prediction_array.size == 1:
        result = int(prediction_array.item())
    else:
        result = int(prediction_array[0])
    
    print(f"Final result: {result}")
    return result
