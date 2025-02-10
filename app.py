import tensorflow as tf
import numpy as np
import cv2
import streamlit as st
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
        target_size = (224, 224)  # Adjust based on your model input size
        image = image.resize(target_size)

        # Convert to numpy array
        img_array = np.array(image)

        # Normalize image
        img_array = img_array / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        return img_array
    except Exception as e:
        st.error(f"Image preprocessing error: {e}")
        return None

# Streamlit UI
st.title("🍃 Potato Disease Prediction System")

uploaded_file = st.file_uploader("Upload an image of a potato leaf...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    processed_image = preprocess_image(image)

    if processed_image is not None:
        try:
            # Make prediction
            prediction = model.predict(processed_image)
            
            # Get confidence scores
            confidence_scores = prediction[0]  # Extract scores
            predicted_index = np.argmax(confidence_scores)
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = confidence_scores[predicted_index] * 100  # Convert to percentage
            
            # Display prediction and confidence
            st.success(f"**Prediction:** {predicted_class} (Confidence: {confidence:.2f}%)")
            
            # Show confidence for all classes
            st.write("### Confidence Scores:")
            for i, class_name in enumerate(CLASS_NAMES):
                st.write(f"{class_name}: {confidence_scores[i] * 100:.2f}%")
        except Exception as e:
            st.error(f"Prediction error: {e}")
    else:
        st.error("Failed to preprocess image.")
