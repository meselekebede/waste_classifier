import streamlit as st
import torch
from PIL import Image
import predict_waste
import os

# UI Setup
st.set_page_config(page_title="♻️ Waste Classifier", layout="centered")
st.title("♻️ Waste Classification App")
st.markdown("Upload an image of waste, and our AI model will classify it into one of six categories.")

# Load model once at startup
@st.cache_resource
def load_model():
    return predict_waste.load_model()

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Show image
    image = Image.open(temp_path)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Predict
    with st.spinner("🔍 Analyzing..."):
        image_tensor = predict_waste.preprocess_image(temp_path)
        prediction, confidence_dict = predict_waste.predict_with_confidence(model, image_tensor)

    # Display result
    st.markdown("### 🧠 Prediction Result")
    st.markdown(f"<h3 style='text-align:center; color:green;'>{prediction}</h3>", unsafe_allow_html=True)

    # Confidence bar chart
    st.markdown("### 🔍 Confidence Scores")
    st.bar_chart(confidence_dict)

    # Optional: Clean up temp file
    os.remove(temp_path)