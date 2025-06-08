import streamlit as st
from PIL import Image
from predict_digit import predict_digit

st.title("MNIST Digit Classifier")
st.write("Upload an image of a handwritten digit (0–9).")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        digit, confidence = predict_digit(image)
        st.success(f"Predicted Digit: {digit} with {confidence*100:.2f}% confidence")
