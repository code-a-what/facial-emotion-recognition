import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json

st.set_page_config(
    page_title="Facial Emotion Recognition",
    page_icon="😊"
)

st.title("😊 Facial Emotion Recognition")
st.write("Capture a photo and detect emotions.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/best_model.h5",
        compile=False
    )

# -----------------------------
# Load Face Detector
# -----------------------------
@st.cache_resource
def load_face_detector():
    return cv2.CascadeClassifier(
        "haarcascade/haarcascade_frontalface_default.xml"
    )

# -----------------------------
# Load Labels
# -----------------------------
@st.cache_data
def load_labels():
    with open("model/class_indices.json", "r") as f:
        class_indices = json.load(f)

    return {v: k for k, v in class_indices.items()}

model = load_model()
face_detector = load_face_detector()
emotion_labels = load_labels()

EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "fear": (128, 0, 128),
    "happy": (0, 200, 0),
    "neutral": (0, 200, 200),
    "sad": (255, 100, 0),
    "surprise": (0, 165, 255),
}

# -----------------------------
# Camera Input
# -----------------------------
camera_image = st.camera_input("📸 Take a photo")

if camera_image is not None:

    file_bytes = np.asarray(
        bytearray(camera_image.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48)
    )

    if len(faces) == 0:
        st.warning("No face detected. Try again with better lighting.")
    else:

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                (48, 48)
            )

            face = face.astype("float32") / 255.0

            face = face.reshape(
                1,
                48,
                48,
                1
            )

            preds = model.predict(
                face,
                verbose=0
            )[0]

            idx = int(np.argmax(preds))
            emotion = emotion_labels[idx]
            confidence = float(preds[idx]) * 100

            color = EMOTION_COLORS.get(
                emotion,
                (255, 255, 255)
            )

            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.rectangle(
                img,
                (x, y - 35),
                (x + w, y),
                color,
                -1
            )

            cv2.putText(
                img,
                f"{emotion.upper()} {confidence:.1f}%",
                (x + 4, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            st.success(
                f"Detected Emotion: {emotion.upper()} ({confidence:.1f}%)"
            )

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            img_rgb,
            caption="Prediction Result",
            use_column_width=True
        )