import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the model
model = load_model("mnist_cnn_model.h5")

def preprocess_image(image):
    image = image.convert("L")  # Grayscale
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert: white background, dark digit
    image_array = image_array / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def predict_digit(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    return np.argmax(prediction)
