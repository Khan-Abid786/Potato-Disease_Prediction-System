import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("Potato_Disease_Prediction1.h5")
    return model

model = load_model()

# Define class labels (update based on your dataset)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize image to model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Potato Disease Detection 🍂")
st.write("Upload an image of a potato leaf to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.write(f"**Prediction:** {predicted_class}")
