# ai/app.py
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import traceback

# Import your prediction function
from prediction import make_prediction

app = Flask(__name__)

@app.route('/api/ai/predict', methods=['POST'])
def predict():
    print("=== PYTHON: PREDICTION REQUEST RECEIVED ===")
    
    if 'file' not in request.files:
        print("PYTHON ERROR: No file in request")
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    print(f"PYTHON: File received: {file.filename}")

    try:
        image_bytes = file.read()
        if not image_bytes:
            print("PYTHON ERROR: Empty file uploaded")
            return jsonify({'error': 'empty file received'}), 400

        image = Image.open(io.BytesIO(image_bytes))
        
        # --- Image Preprocessing ---
        image = image.convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = 255.0 - image_array
        image_array = image_array / 255.0 
        image_vector = image_array.reshape(784, 1)

        # --- Make Prediction ---
        print("PYTHON: Calling make_prediction...")
        prediction_result = make_prediction(image_vector)
        print(f"PYTHON: Prediction result: {prediction_result}")

        if prediction_result is None:
            print("PYTHON ERROR: Prediction returned None")
            return jsonify({'error': 'model failed to return a prediction'}), 500
        
        # --- THIS IS THE CRITICAL FIX ---
        final_prediction = int(prediction_result)
        response_data = {'prediction': final_prediction}
        
        print(f"PYTHON: Sending response: {response_data}")
        return jsonify(response_data)

    except Exception as e:
        print(f"PYTHON ERROR: An exception occurred: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500

# This health check is useful for debugging
@app.route('/api/ai')
def health_check():
    return jsonify({"status": "Python AI server is alive and well!"})