from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from prediction import make_prediction
    print("✓ Successfully imported make_prediction")
except ImportError as e:
    print(f"✗ Import error: {e}")

app = Flask(__name__)
CORS(app)

@app.route('/api/ai')
def health_check():
    return jsonify({"status": "Python AI server is alive"})

@app.route('/api/ai/predict', methods=['POST'])
def predict():
    print("=== PREDICTION REQUEST RECEIVED ===")
    
    # Check if file exists
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    print(f"File received: {file.filename}")

    try:
        # Process the image
        image_bytes = file.read()
        print(f"Image size: {len(image_bytes)} bytes")
        
        if len(image_bytes) == 0:
            print("ERROR: Empty file")
            return jsonify({'error': 'empty file'}), 400
            
        # Open and preprocess image
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image opened: {image.size}, {image.mode}")

        # Convert to grayscale and resize
        image = image.convert('L')
        image = image.resize((28, 28))
        
        # Convert to numpy array
        image_array = np.array(image)
        print(f"Array shape: {image_array.shape}")
        
        # Invert colors (MNIST is white on black)
        image_array = 255.0 - image_array
        
        # Normalize to 0-1
        image_array = image_array / 255.0
        
        # Reshape to (784, 1) for the model
        image_vector = image_array.reshape(784, 1)
        print(f"Final vector shape: {image_vector.shape}")

        # Make prediction
        print("Calling make_prediction...")
        prediction_result = make_prediction(image_vector)
        print(f"Prediction result: {prediction_result} (type: {type(prediction_result)})")

        # Ensure we have a valid prediction
        if prediction_result is None:
            print("ERROR: Prediction returned None")
            return jsonify({'error': 'prediction failed - returned None'}), 500

        # Convert to int and return
        final_prediction = int(prediction_result)
        print(f"Returning prediction: {final_prediction}")
        
        response_data = {'prediction': final_prediction}
        print(f"Response JSON: {response_data}")
        
        return jsonify(response_data)

    except Exception as e:
        print(f"ERROR in predict(): {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500

# This is important for Vercel
if __name__ == '__main__':
    app.run(debug=True)
