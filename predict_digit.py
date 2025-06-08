import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image
import streamlit as st
import os

@st.cache_resource
def create_model():
    """Create the Sequential CNN model"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@st.cache_resource
def load_trained_model():
    """Load the pre-trained model"""
    model = create_model()
    
    # Try both filename formats for compatibility
    weights_files = ["mnist_cnn_model.weights.h5", "mnist_cnn_model.h5"]
    
    for weights_file in weights_files:
        if os.path.exists(weights_file):
            try:
                model.load_weights(weights_file)
                st.success(f"✅ Pre-trained model loaded from {weights_file}!")
                return model, True
            except Exception as e:
                st.warning(f"Could not load {weights_file}: {e}")
                continue
    
    st.error("❌ No model weights file found!")
    st.info("Please upload 'mnist_cnn_model.weights.h5' to the repository.")
    return model, False

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    image = image.convert("L")
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def predict_digit(image):
    """Predict digit from uploaded image"""
    model, weights_loaded = load_trained_model()
    
    if not weights_loaded:
        return -1, 0.0
    
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return int(predicted_digit), float(confidence)