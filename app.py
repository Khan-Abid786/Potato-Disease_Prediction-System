import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load trained model
MODEL_PATH = "Potato_Disease_Prediction2.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Ensure this matches the model's output)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]  # Adjust based on your training labels

# Function to preprocess image
def preprocess_image(image):
    try:
        # Convert image to RGB (if not already)
        image = image.convert("RGB")
        
        # Resize image to match model's expected input shape
        target_size = (224, 224)  # Change this to 224, 224 if your model expects that
        image = image.resize(target_size)

        # Convert to numpy array
        img_array = np.array(image)

        # Convert to proper format
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# ---- Streamlit UI ----

# Create a boundary around the entire UI
st.markdown(
    """
    <style>
        .main {
            border: 3px solid #4CAF50;
            padding: 20px;
            border-radius: 15px;
        }
    </style>
    <div class="main">
    """,
    unsafe_allow_html=True,
)

# Welcome heading on the top-left
st.markdown("<h2 style='text-align: left;'>Welcome</h2>", unsafe_allow_html=True)

# Navigation bar (fake navbar with subheading)
st.markdown("---")
st.markdown("<h3 style='text-align: center; color: green;'>🥔 Potato Disease Prediction</h3>", unsafe_allow_html=True)
st.markdown("---")

# Centered file uploader with a square-shaped clickable drag-and-drop area
st.markdown("<h4 style='text-align: center;'>Upload an image of a potato leaf:</h4>", unsafe_allow_html=True)

# Custom clickable upload area
st.markdown(
    """
    <style>
        .upload-box {
            border: 2px dashed gray;
            width: 300px;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            text-align: center;
        }
    </style>
    <div class="upload-box">
        <label for="file_uploader">Click or Drag & Drop to Upload</label>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Submission button outside the square
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Process image
    processed_image = preprocess_image(Image.open(uploaded_file))

    # Button for making predictions
    if st.button("🔍 Predict"):
        if processed_image is not None:
            try:
                # Make prediction
                prediction = model.predict(processed_image)

                # Debugging outputs
                st.write(f"Model prediction raw output: {prediction}")

                if prediction.shape[1] != len(CLASS_NAMES):
                    st.error(f"Model output shape {prediction.shape} does not match CLASS_NAMES length {len(CLASS_NAMES)}")
                else:
                    predicted_index = np.argmax(prediction)
                    st.write(f"Predicted index: {predicted_index}")

                    if predicted_index >= len(CLASS_NAMES):
                        st.error("Error: Predicted index is out of range.")
                    else:
                        predicted_class = CLASS_NAMES[predicted_index]
                        st.success(f"**Prediction:** {predicted_class}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Failed to preprocess image.")

# Close the boundary div
st.markdown("</div>", unsafe_allow_html=True)
