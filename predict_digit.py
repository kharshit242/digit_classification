import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

def create_model():
    """Create the Sequential CNN model (same as Colab)"""
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

def train_and_save_model():
    """Train the model and save weights"""
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    
    # Preprocess data
    train_images = train_images.reshape((60000, 28, 28, 1))
    train_images = train_images.astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1))
    test_images = test_images.astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    # Create and train model
    model = create_model()
    print("Training model...")
    model.fit(train_images, train_labels, epochs=5, batch_size=64)
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model weights
    model.save_weights("mnist_cnn_model.h5")
    print("Model weights saved as 'mnist_cnn_model.h5'")
    
    return model

# Initialize model
model = create_model()
weights_loaded = False

# Try to load existing weights, otherwise train new model
if os.path.exists("mnist_cnn_model.h5"):
    try:
        model.load_weights("mnist_cnn_model.h5")
        weights_loaded = True
        print("✅ Model weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        print("Training new model...")
        model = train_and_save_model()
        weights_loaded = True
else:
    print("❌ Model weights not found. Training new model...")
    model = train_and_save_model()
    weights_loaded = True

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Convert to grayscale
    image = image.convert("L")
    # Resize to 28x28
    image = image.resize((28, 28))
    # Convert to numpy array
    image_array = np.array(image)
    # Invert colors (MNIST has white digits on black background)
    image_array = 255 - image_array
    # Normalize to 0-1
    image_array = image_array.astype('float32') / 255.0
    # Reshape for model input
    image_array = image_array.reshape(1, 28, 28, 1)
    return image_array

def predict_digit(image):
    """Predict digit from uploaded image"""
    if not weights_loaded:
        return -1, 0.0
    
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return int(predicted_digit), float(confidence)