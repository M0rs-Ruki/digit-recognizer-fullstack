# app.py
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io

# Import our prediction function from the other file
from prediction import make_prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure a file was sent
    if 'file' not in request.files:
        return jsonify({'error': 'no file provided'}), 400

    file = request.files['file']

    try:
        # Read the image file from the request
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))

        # --- PREPROCESS THE IMAGE TO MATCH YOUR MODEL'S INPUT ---
        # 1. Convert to grayscale ('L' mode)
        image = image.convert('L')

        # 2. Resize to 28x28 pixels
        image = image.resize((28, 28))

        # 3. Convert to a numpy array
        image_array = np.array(image)

        # 4. Invert colors (your model was trained on white digits on a black background)
        image_array = 255.0 - image_array

        # 5. Normalize the pixel values to be between 0 and 1
        image_array = image_array / 255.0

        # 6. Flatten the 28x28 array into a 784x1 vector
        image_vector = image_array.reshape(784, 1)

        # Make the prediction using the function from prediction.py
        prediction = make_prediction(image_vector)

        # Return the result as a JSON object
        return jsonify({'predicted_digit': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(debug=True, port=5000)