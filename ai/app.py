from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback

# Import our prediction function
from prediction import make_prediction

app = Flask(__name__)
CORS(app)  # Enable CORS

@app.route('/api/ai')
def health_check():
    return jsonify({"status": "Python AI server is alive", "service": "digit-recognition"})

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
            return jsonify({'error': 'empty file received'}), 400
            
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image loaded: {image.size}, {image.mode}")

        # Preprocess for MNIST model
        image = image.convert('L')          # Grayscale
        image = image.resize((28, 28))      # 28x28 pixels
        image_array = np.array(image)       # Convert to numpy
        image_array = 255.0 - image_array  # Invert colors
        image_array = image_array / 255.0   # Normalize
        image_vector = image_array.reshape(784, 1)  # Flatten
        
        print("Preprocessing complete, making prediction...")

        # Make prediction
        prediction = make_prediction(image_vector)
        print(f"Prediction: {prediction}")

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print(f"PREDICTION ERROR: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500

# For local testing
if __name__ == '__main__':
    print("Starting AI service on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
