import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.title("🧠 Handwritten Character Recognition (MNIST + EMNIST)")

# Load models safely
@st.cache_resource
def load_models():
    mnist_model = tf.keras.models.load_model("mnist_model.keras")
    emnist_model = tf.keras.models.load_model("emnist_model.keras")
    return mnist_model, emnist_model

mnist_model, emnist_model = load_models()

option = st.selectbox(
    "Choose Model",
    ["MNIST (Digits 0-9)", "EMNIST (Letters A-Z)"]
)

uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

def preprocess(img):
    img = img.convert("L")
    img = img.resize((28, 28))
    img = np.array(img) / 255.0
    img = img.reshape(1, 28, 28, 1)
    return img

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, width=200)

    processed = preprocess(image)

    if option == "MNIST (Digits 0-9)":
        pred = mnist_model.predict(processed)
        result = np.argmax(pred)
        conf = np.max(pred)

        st.success(f"Digit: {result}")
        st.info(f"Confidence: {conf*100:.2f}%")

    else:
        pred = emnist_model.predict(processed)
        result = np.argmax(pred) + 1
        letter = chr(result + 64)
        conf = np.max(pred)

        st.success(f"Letter: {letter}")
        st.info(f"Confidence: {conf*100:.2f}%")
