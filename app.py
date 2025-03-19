import os
from flask import Flask, request, jsonify, render_template_string
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = load_model('model_xray1.h5')
except Exception as e:
    print(f"Failed to load model: {e}")
    exit()

# Function to classify an image
def classify_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.
        predictions = model.predict(img_array)
        return "Pneumonia" if predictions[0][0] > 0.5 else "Normal"
    except Exception as e:
        return f"Error classifying image: {e}"

# Load the index.html template as a string
with open('index.html', 'r') as f:
    index_template = f.read()

# Route for index page
@app.route('/')
def index():
    return render_template_string(index_template)

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file
        img_file = request.files['file']
        img_path = 'uploaded_image.jpg'
        img_file.save(img_path)
        
        # Classify the image
        result = classify_image(img_path)
        
        return jsonify({'prediction': result})

if __name__ == '__main__':
    # Update the app to bind to the correct port and disable debug mode for production
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
