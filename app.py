import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(page_title="Facial Emotion Recognition", page_icon="😊")
st.title("😊 Facial Emotion Recognition")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/best_model.h5", compile=False)

@st.cache_data
def load_labels():
    with open("model/class_indices.json") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}

model = load_model()
emotion_labels = load_labels()

class EmotionProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Preprocess
        gray = np.mean(img, axis=2).astype("float32")          # to grayscale
        resized = np.array(Image.fromarray(gray.astype("uint8")).resize((48, 48)))
        inp = resized.astype("float32") / 255.0
        inp = inp.reshape(1, 48, 48, 1)

        # Predict
        prediction = model.predict(inp, verbose=0)
        pred_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))
        label = f"{emotion_labels[pred_class]} ({confidence*100:.1f}%)"

        # Draw label on frame
        import cv2
        cv2.putText(img, label, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.write("### Live Camera Feed")
webrtc_streamer(
    key="emotion",
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)