import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

st.title("Facial Emotion Recognition")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
    "model/best_model.h5",
    compile=False
)

model = load_model()

with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)

emotion_labels = {v: k for k, v in class_indices.items()}

picture = st.camera_input("Take a photo")

if picture:
    image = Image.open(picture).convert("L")
    image = image.resize((48, 48))

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    prediction = model.predict(img_array, verbose=0)

    pred_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.image(picture)

    st.success(
        f"Emotion: {emotion_labels[pred_class]} ({confidence*100:.1f}%)"
    )