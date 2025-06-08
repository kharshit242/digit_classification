import streamlit as st
from PIL import Image
import numpy as np
from predict_digit import predict_digit

# Set page config
st.set_page_config(
    page_title="MNIST Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

# Title and description
st.title("🔢 MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9) and let the AI predict it!")

# Create two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["png", "jpg", "jpeg"],
        help="Upload a clear image of a handwritten digit"
    )

with col2:
    st.subheader("Drawing Tips")
    st.info("""
    📝 **For best results:**
    - Draw digits clearly
    - Use dark pen/pencil on white paper
    - Make digits large and centered
    - Avoid extra markings
    """)

# Display uploaded image and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display original image
    st.subheader("Uploaded Image")
    st.image(image, caption="Your uploaded image", use_column_width=True)
    
    # Predict button
    if st.button("🔍 Predict Digit", type="primary"):
        with st.spinner("Analyzing image..."):
            digit, confidence = predict_digit(image)
            
            if digit == -1:
                st.error("⚠️ Model not loaded properly. Please refresh the page.")
            else:
                # Display results
                st.subheader("Prediction Results")
                
                # Create metrics display
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Digit", f"{digit}")
                with col2:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
                
                # Success message with color coding
                if confidence > 0.8:
                    st.success(f"🎯 High confidence: The digit is **{digit}**")
                elif confidence > 0.5:
                    st.warning(f"⚡ Medium confidence: Likely **{digit}**")
                else:
                    st.info(f"🤔 Low confidence: Possibly **{digit}**")

# Add some example images or instructions
st.markdown("---")
st.subheader("💡 How it works")
st.write("""
This app uses a Convolutional Neural Network (CNN) trained on the MNIST dataset 
to recognize handwritten digits. The model achieves high accuracy on standard 
handwritten digits.
""")

# Add footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and TensorFlow")