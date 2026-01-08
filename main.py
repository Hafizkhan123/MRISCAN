import os
import sys
import warnings
import numpy as np
import streamlit as st
from PIL import Image

# =========================
# Suppress warnings
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

# =========================
# Load TensorFlow / Keras
# =========================
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.vgg16 import preprocess_input
except ImportError as e:
    st.error("TensorFlow is not installed properly.")
    st.stop()

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="MRI Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection (MRI)")
st.write("Upload an MRI image to detect tumor type.")

# =========================
# Project paths
# =========================
project_root = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(project_root, "models", "model.keras")

# =========================
# Load model (cached)
# =========================
@st.cache_resource
def load_trained_model():
    try:
        # Patch Flatten issue (same as Flask)
        from keras.src.layers import Flatten as KerasFlatten

        original_compute_output_spec = KerasFlatten.compute_output_spec

        def patched_compute_output_spec(self, inputs):
            if isinstance(inputs, list) and len(inputs) == 1:
                inputs = inputs[0]
            return original_compute_output_spec(self, inputs)

        KerasFlatten.compute_output_spec = patched_compute_output_spec

        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_trained_model()

if model is None:
    st.stop()

# =========================
# Class labels
# =========================
class_labels = ['pituitary', 'notumor', 'glioma', 'meningioma']

# =========================
# Prediction function
# =========================
def predict_tumor(image: Image.Image):
    IMAGE_SIZE = 256
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(image).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)

    # Convert logits → probabilities if needed
    try:
        is_prob = np.allclose(np.sum(prediction, axis=1), 1.0)
    except Exception:
        is_prob = False

    if not is_prob:
        prediction = tf.nn.softmax(prediction, axis=1).numpy()

    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[0][predicted_index])
    label = class_labels[predicted_index]

    display_label = "NO TUMOR" if label == "notumor" else f"TUMOR: {label.upper()}"
    return display_label, confidence

# =========================
# File uploader UI
# =========================
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing MRI scan..."):
            result, confidence = predict_tumor(image)

        st.success(f"Result: {result}")
        st.info(f"Confidence: {confidence * 100:.2f}%")
