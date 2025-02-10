import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
MODEL_PATH = "Potato_Disease_Prediction2.h5"  # Ensure this path is correct
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names (Ensure this matches the model's output)
CLASS_NAMES = ["Early Blight","Late Blight","Healthy"  ]  # Adjust based on your training labels

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
