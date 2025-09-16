from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import io
import traceback
import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import prediction function
try:
    from prediction import make_prediction
    print("✓ Successfully imported make_prediction")
except ImportError as e:
    print(f"✗ Failed to import make_prediction: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

@app.route('/api/ai')
def health_check():
    return jsonify({
        "status": "Python AI server is alive", 
        "service": "digit-recognition",
        "python_version": sys.version
    })

@app.route('/api/ai/predict', methods=['POST'])
def predict():
    print("=== AI Prediction Request Started ===")
    
    if 'file' not in request.files:
        print("ERROR: No file in request")
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']
    print(f"File received: {file.filename}, Content-Type: {file.content_type}")

    try:
        image_bytes = file.read()
        print(f"Image bytes length: {len(image_bytes)}")
        
        if len(image_bytes) == 0:
            print("ERROR: Empty file received")
            return jsonify({'error': 'empty file received'}), 400
            
        # Process image
        image = Image.open(io.BytesIO(image_bytes))
        print(f"Image opened: size={image.size}, mode={image.mode}")

        # Preprocess for MNIST
        image = image.convert('L')
        image = image.resize((28, 28))
        image_array = np.array(image)
        image_array = 255.0 - image_array
        image_array = image_array / 255.0
        image_vector = image_array.reshape(784, 1)
        
        print("Image preprocessing completed")

        # Make prediction
        prediction = make_prediction(image_vector)
        print(f"Prediction successful: {prediction}")

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        print(f"ERROR in prediction: {str(e)}")
        print("Full traceback:")
        traceback.print_exc()
        return jsonify({'error': f'prediction failed: {str(e)}'}), 500

# Vercel entry point
def handler(request):
    return app(request.environ, lambda *args: None)
