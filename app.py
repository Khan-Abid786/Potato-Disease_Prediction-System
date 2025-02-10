import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
MODEL_PATH = "Potato_Disease_Prediction2.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Ensure this matches the model's output)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Function to preprocess image
def preprocess_image(image):
    try:
        image = image.convert("RGB")
        image = image.resize((224, 224))  # Resize to match model's input
        img_array = np.array(image) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# ---- Streamlit UI ----

# Custom CSS for styling
st.markdown(
    """
    <style>
        /* Full boundary enclosing everything */
        .main-container {
            border: 4px solid #4CAF50;
            padding: 20px;
            border-radius: 15px;
            width: 80%;
            margin: auto;
            text-align: center;
            background-color: #f8f8f8;
        }
        
        /* Small rectangular box for "Welcome" */
        .welcome-box {
            border: 2px solid #4CAF50;
            padding: 8px 15px;
            border-radius: 10px;
            display: inline-block;
            font-size: 20px;
            font-weight: bold;
            background-color: white;
        }

        /* Centered file upload box */
        .upload-box {
            border: 2px dashed gray;
            width: 300px;
            height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            margin: auto;
            text-align: center;
            font-size: 16px;
            color: #555;
        }

        /* Hide Streamlit default top bar */
        header {display: none;}

        /* Submission button */
        .submit-btn {
            margin-top: 20px;
        }

    </style>
    """,
    unsafe_allow_html=True,
)

# Start boundary
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Welcome box (Top-left)
st.markdown('<div class="welcome-box">Welcome</div>', unsafe_allow_html=True)

# Navigation Heading
st.markdown("<h2 style='color: green;'>🥔 Potato Disease Prediction</h2>", unsafe_allow_html=True)

# Upload Section
st.markdown("<h4>Upload an image of a potato leaf:</h4>", unsafe_allow_html=True)

# Clickable Upload Box
st.markdown('<div class="upload-box">Click or Drag & Drop to Upload</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

# Display uploaded image and predict
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Submission button
    if st.button("🔍 Predict", key="predict_btn"):
        processed_image = preprocess_image(image)
        if processed_image is not None:
            try:
                prediction = model.predict(processed_image)
                predicted_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_index]
                st.success(f"**Prediction:** {predicted_class}")
            except Exception as e:
                st.error(f"Prediction error: {e}")
        else:
            st.error("Failed to preprocess image.")

# Close the boundary
st.markdown("</div>", unsafe_allow_html=True)
