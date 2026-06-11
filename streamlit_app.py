import keras
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained models
mnist_model = keras.models.load_model("mnist_model.keras")
emnist_model = keras.models.load_model("emnist_model.keras")

st.set_page_config(page_title="Handwritten Recognition", layout="centered")

st.title("🧠 Handwritten Character Recognition System")
st.write("Upload a handwritten digit or letter image and get predictions instantly.")

# Sidebar option
option = st.sidebar.selectbox(
    "Choose Model",
    ("MNIST - Digit Recognition (0-9)", "EMNIST - Letter Recognition (A-Z)")
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image (PNG/JPG/JPEG)",
    type=["png", "jpg", "jpeg"]
)

# Preprocess image
def preprocess_image(image):
    image = image.convert("L")  # grayscale
    image = image.resize((28, 28))  # MNIST/EMNIST size
    image = np.array(image)
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    processed_image = preprocess_image(image)

    if option == "MNIST - Digit Recognition (0-9)":
        prediction = mnist_model.predict(processed_image)
        result = np.argmax(prediction)
        confidence = np.max(prediction)

        st.success(f"Predicted Digit: {result}")
        st.info(f"Confidence: {confidence * 100:.2f}%")

    else:
        prediction = emnist_model.predict(processed_image)
        result = np.argmax(prediction) + 1
        confidence = np.max(prediction)

        letter = chr(result + 64)  # Convert to A-Z

        st.success(f"Predicted Letter: {letter}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
