import tensorflow as tf
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from PIL import Image,ImageOps
from io import BytesIO
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import  img_to_array
import matplotlib.pyplot as plt


model =load_model('my_model.h5')

model=model.load_weights('brain_tumor_model.weights.h5')

st.markdown("<h1 style='text-align: center; color:Black;'>Brain Tumor Detection<h1/>",unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color:Black;'>Upload MRI scan and the model will predict the likelihood of a brain tumor or generate segmentation mask.<h1/>",unsafe_allow_html=True)

file=st.file_uploader("Please upload your MRI Scan",type=['jpg','png','jpeg','tif'])
def predict(image, model):
    # Resize the image directly using PIL
    size = (256, 256) 
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)  # Resizes image
    image = image.convert("RGB")  # Ensures the image has 3 color channels
    img_arr = img_to_array(image) / 255.0  # Normalize the image
    img_arr = np.expand_dims(img_arr, axis=0)  # Add batch dimension
    pred = model.predict(img_arr)
    pred_mask = np.squeeze(pred, axis=0)  # Remove batch dimension
    return pred_mask
def display_img_mask(image,pred_mask):
    fig,ax =plt.subplots(1,1 ,figsize=(10,10))
    ax[0].imshow(image,cmap='gray')
    ax[0].set_title('Given Image')
    ax[0].axis('off')
    ax[1].imshow(pred_mask,cmap='gray')
    ax[1].set_title("Predicted Output")
    ax[1].axis('off')
    plt.show()
if file is None:
    st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a File</h5>", unsafe_allow_html=True)
else:
    image=Image.open(file)
    pred=predict(image,model)
    display_img_mask(image,pred)