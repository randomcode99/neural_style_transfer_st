import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'
from PIL import Image
import numpy as np

import tensorflow as tf
import streamlit as st
import tensorflow_hub as hub

hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

content_image = st.file_uploader("Upload the content image")
content_uploaded = False

style_image = st.file_uploader("Upload the style image")
style_uploaded = False

if content_image:
    content_image = Image.open(content_image)
    content_image = content_image.resize((512, 512))
    content_image = tf.keras.preprocessing.image.img_to_array(content_image) / 255.0
    content_image = np.array(content_image)[:, :, 0:3]
    st.image(content_image)
    content_uploaded = True

if style_image:
    style_image = Image.open(style_image)
    style_image = style_image.resize((512, 512))
    style_image = tf.keras.preprocessing.image.img_to_array(style_image)/255.0
    style_image = np.array(style_image)[:, :, 0:3]
    st.image(style_image)
    style_uploaded = True

if content_uploaded and style_uploaded:
    info = st.info("Loading Model")
    content_image = tf.constant(content_image)[np.newaxis, :, :, :]
    style_image = tf.constant(style_image)[np.newaxis, :, :, :]
    stylized_image = hub_model(content_image, style_image)[0]
    st.image(tensor_to_image(stylized_image))
    info.empty()


