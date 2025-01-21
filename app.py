
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# Use st.cache_resource for caching models
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_model.h5')  # Ensure the correct model path
    return model

model = load_model()
model=model.load_weights('brain_tumor_model.weights.h5')

# App headers and descriptions
st.markdown("<h1 style='text-align: center; color: Black;'>Brain Tumor Classifier</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: Black;'>Upload an MRI scan, and the model will predict the likelihood of a brain tumor or generate segmentation masks.</h3>", unsafe_allow_html=True)

# File uploader
file = st.file_uploader("Please upload your MRI Scan", type=["jpg", "png", "tif"])

# Prediction function
def import_and_predict(image_data, model):
    size = (256, 256)  # Updated to match model's expected input size
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)  # Resizes image
    image = image.convert("RGB")  # Ensures the image has 3 color channels
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values to [0, 1]
    img_reshape = img.reshape(1, 256, 256, 3)  # Adjusted input shape to match model's expected input
    prediction = model.predict(img_reshape)
    pred_mask = np.squeeze(prediction, axis=0)  # Squeeze prediction to remove batch dimension
    return pred_mask

# Function to display the mask using Matplotlib
def display_mask(pred_mask):
    fig, ax = plt.subplots()
    ax.imshow(pred_mask, cmap='gray')
    ax.set_title("Predicted Output")
    ax.axis('off')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# Handle file upload and predictions
if file is None:
    st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a File</h5>", unsafe_allow_html=True)
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)  # Updated to use use_container_width
    pred_mask = import_and_predict(image, model)

    # Display the predicted mask
    if pred_mask is not None:
        mask_buf = display_mask(pred_mask)
        st.image(mask_buf, caption="Predicted Output", use_container_width=True)
    else:
        st.error("Error: Model predictions are invalid. Please check the model and input image.")
