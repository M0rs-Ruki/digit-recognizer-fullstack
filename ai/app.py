from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction import make_prediction

app = Flask(__name__)
CORS(app)

@app.route('/api/ai')
def health_check():
    return jsonify({"status": "Python AI server is alive"})

@app.route('/api/ai/predict', methods=['POST'])
def predict():
    print("=== AI Prediction Request ===")
    
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    print(f"File: {file.filename}, Type: {file.content_type}")

    try:
        image_bytes = file.read()
        print(f"Image bytes: {len(image_bytes)}")
        
        if len(image_bytes) == 0:
            return jsonify({'error': 'empty file'}), 400
            
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image loaded: {image.size}, {image.mode}")

        # Preprocessing with debug info
        image = image.convert('L')
        print(f"After grayscale: {image.mode}")
        
        image = image.resize((28, 28))
        print(f"After resize: {image.size}")
        
        image_array = np.array(image)
        print(f"Array shape: {image_array.shape}, dtype: {image_array.dtype}")
        print(f"Array min/max before inversion: {image_array.min()}/{image_array.max()}")
        
        image_array = 255.0 - image_array
        print(f"Array min/max after inversion: {image_array.min()}/{image_array.max()}")
        
        image_array = image_array / 255.0
        print(f"Array min/max after normalization: {image_array.min()}/{image_array.max()}")
        
        image_vector = image_array.reshape(784, 1)
        print(f"Final vector shape: {image_vector.shape}")
        print(f"Sample of vector values: {image_vector[:10].flatten()}")

        # Make prediction with debug
        print("Calling make_prediction...")
        prediction = make_prediction(image_vector)
        print(f"Raw prediction result: {prediction} (type: {type(prediction)})")

        # Ensure it's a valid integer
        if prediction is None:
            print("ERROR: Prediction is None!")
            return jsonify({'error': 'prediction returned None'}), 500
        
        if np.isnan(prediction):
            print("ERROR: Prediction is NaN!")
            return jsonify({'error': 'prediction is NaN'}), 500

        final_prediction = int(prediction)
        print(f"Final prediction: {final_prediction}")

        return jsonify({'prediction': final_prediction})

    except Exception as e:
        print(f"ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500
