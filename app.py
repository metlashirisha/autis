import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Get the expected input shape of the model
input_shape = model.input_shape  # Typically (None, height, width, channels)
image_size = input_shape[1:3]  # Extract (height, width)

# Define image preprocessing function
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure RGB format
    image = image.resize(image_size)  # Resize to model's expected size
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)

# Streamlit UI
st.title("Autism Facial Recognition Model")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    try:
        # Preprocess image
        processed_image = preprocess_image(image)

        # Check the shape before passing to the model
        st.write(f"Processed Image Shape: {processed_image.shape}")

        # Make prediction
        prediction = model.predict(processed_image)

        # Display the result
        st.write(f"Prediction: {prediction}")
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
