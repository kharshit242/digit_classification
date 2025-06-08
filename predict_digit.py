import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("mnist_cnn_model.h5")

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).astype("float32") / 255.0
    image = 1 - image  # Invert colors (black digit on white bg)
    image = image.reshape(1, 28, 28, 1)
    return image

def predict_digit(image):
    processed = preprocess_image(image)
    prediction = model.predict(processed)
    return np.argmax(prediction), np.max(prediction)
