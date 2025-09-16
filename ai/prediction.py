import os
import numpy as np

# Load model parameters
_this_dir = os.path.dirname(os.path.abspath(__file__))
_model_path = os.path.join(_this_dir, 'model_parameters.npz')

print(f"Loading model from: {_model_path}")
try:
    params = np.load(_model_path)
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    print(f"Model loaded successfully - W1: {W1.shape}, W2: {W2.shape}")
except Exception as e:
    print(f"ERROR loading model: {e}")
    raise

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
    print(f"make_prediction called with shape: {image_data.shape}")
    
    try:
        # Validate input
        if image_data.shape != (784, 1):
            print(f"ERROR: Invalid input shape {image_data.shape}, expected (784, 1)")
            return None
            
        # Forward propagation
        A2 = forward_propagation(W1, b1, W2, b2, image_data)
        print(f"Forward prop output shape: {A2.shape}")
        
        # Get prediction
        prediction_array = get_predictions(A2)
        print(f"Prediction array: {prediction_array}")
        
        # Convert to scalar
        if hasattr(prediction_array, 'item'):
            result = prediction_array.item()
        else:
            result = int(prediction_array[0]) if hasattr(prediction_array, '__getitem__') else prediction_array
            
        print(f"Final prediction: {result}")
        return result
        
    except Exception as e:
        print(f"ERROR in make_prediction: {e}")
        traceback.print_exc()
        return None
