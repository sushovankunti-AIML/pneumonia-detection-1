import streamlit as st
import tensorflow as tf
import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Configure the Streamlit page
st.set_page_config(page_title="Pneumonia Detection App", layout="centered")

# Load the model once and cache it to prevent reloading on every interaction
@st.cache_resource
def load_model():
    # Load your trained ResNet50 model
    return tf.keras.models.load_model('ResNet50_model.keras')

model = load_model()

st.title("Pneumonia Detection from Chest X-Rays")
st.write("Upload a DICOM (.dcm) or standard image file to predict the probability of pneumonia.")

# 1. Allow users to upload a DICOM or image file
uploaded_file = st.file_uploader("Choose a file...", type=["dcm", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Determine file type and extract pixel array
        if uploaded_file.name.lower().endswith(".dcm"):
            ds = pydicom.dcmread(uploaded_file)
            img_raw = ds.pixel_array
        else:
            image = Image.open(uploaded_file).convert('L') # Convert to grayscale
            img_raw = np.array(image)

        # 5. Display the original image
        st.image(img_raw, caption='Uploaded X-Ray', use_container_width=True, clamp=True)

        # 2. Preprocess the uploaded image
        # Resize to 512x512
        img_resized = cv2.resize(img_raw, (512, 512))
        
        # Normalize pixel values
        img_normalized = img_resized / 255.0
        
        # Expand dimensions to shape (1, 512, 512, 1)
        # Note: If your ResNet50 model expects 3 channels, use cv2.cvtColor to convert Grayscale to RGB instead.
        img_input = np.expand_dims(img_normalized, axis=-1) 
        img_input = np.expand_dims(img_input, axis=0)

        # 4. Make predictions
        with st.spinner('Analyzing image...'):
            prediction = model.predict(img_input)
            probability = float(prediction[0][0])
            predicted_class = 1 if probability > 0.5 else 0

        # Display results
        st.subheader("Prediction Results")
        if predicted_class == 1:
            st.error(f"**Diagnosis:** Pneumonia Present")
            st.write(f"**Confidence:** {probability:.4f} ({probability*100:.2f}%)")
        else:
            st.success(f"**Diagnosis:** Pneumonia Absent")
            st.write(f"**Confidence:** {(1 - probability):.4f} ({(1 - probability)*100:.2f}%)")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
