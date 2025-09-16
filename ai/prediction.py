import os
import numpy as np

# --- Robust Model Loading with Error Handling ---
try:
    # Get the absolute path to the directory containing this script
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the model file
    _model_path = os.path.join(_this_dir, 'model_parameters.npz')

    print(f"PYTHON: Attempting to load model from: {_model_path}")

    # Check if the file actually exists before trying to load it
    if not os.path.exists(_model_path):
        raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at path: {_model_path}")

    params = np.load(_model_path)
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    print(f"PYTHON ✓: Model loaded successfully - W1 shape: {W1.shape}, W2 shape: {W2.shape}")

except Exception as e:
    print(f"PYTHON ✗: CRITICAL ERROR while loading model: {e}")
    # Re-raise the exception to ensure the application fails clearly
    raise e
# --- End of Model Loading ---


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

def make_prediction(image_data):
    A2 = forward_propagation(W1, b1, W2, b2, image_data)
    prediction_array = get_predictions(A2)
    
    # Convert numpy int to a standard Python int
    return int(prediction_array.item())