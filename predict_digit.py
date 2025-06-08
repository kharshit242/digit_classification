import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from PIL import Image

# Rebuild the model architecture (same as in Colab)
def create_model():
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, kernel_size=3, activation="relu")(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(64, kernel_size=3, activation="relu")(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(10, activation="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Create model and load weights
model = create_model()
try:
    model.load_weights("mnist_cnn_model.h5")
    print("Model weights loaded successfully!")
except:
    print("Could not load weights - using untrained model")

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