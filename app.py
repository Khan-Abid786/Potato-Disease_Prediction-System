import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model.h5")
    return model

model = load_model()

# Define class names based on training labels
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🥔 Potato Disease Prediction App")
st.write("Upload a potato leaf image to predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show prediction
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: {confidence:.2f}%")

st.write("🔬 Model by [Your Name]")
