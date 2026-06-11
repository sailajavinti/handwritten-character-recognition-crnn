import os
import gdown
import tensorflow as tf

# Download from Google Drive or external link
if not os.path.exists("mnist_model.keras"):
    url = "YOUR_MNIST_MODEL_LINK"
    gdown.download(url, "mnist_model.keras", quiet=False)

if not os.path.exists("emnist_model.keras"):
    url = "YOUR_EMNIST_MODEL_LINK"
    gdown.download(url, "emnist_model.keras", quiet=False)

mnist_model = tf.keras.models.load_model("mnist_model.keras")
emnist_model = tf.keras.models.load_model("emnist_model.keras")
