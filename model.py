# model.py

import numpy as np
import cv2
import json
import tensorflow as tf
from PIL import Image

# ── Load model and class labels once at startup ──────────────────────
MODEL_PATH     = 'model/best_model_continued.h5'
LABELS_PATH    = 'model/class_indices.json'
CASCADE_PATH   = 'haarcascade/haarcascade_frontalface_default.xml'

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class mapping and flip it → {0: 'angry', 1: 'fear', ...}
with open(LABELS_PATH, 'r') as f:
    class_indices = json.load(f)
index_to_label = {v: k for k, v in class_indices.items()}

# Load face detector
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Emotion colors for bounding boxes (BGR format for OpenCV)
EMOTION_COLORS = {
    'angry':    (0,   0,   255),
    'fear':     (128, 0,   128),
    'happy':    (0,   200, 0  ),
    'neutral':  (200, 200, 0  ),
    'sad':      (255, 100, 0  ),
    'surprise': (0,   165, 255),
}

# ── Preprocess a single face crop ────────────────────────────────────
def preprocess_face(face_gray):
    """
    Takes a grayscale face crop (numpy array),
    returns a preprocessed tensor ready for model input.
    """
    face = cv2.resize(face_gray, (48, 48))
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)   # (48,48) → (48,48,1)
    face = np.expand_dims(face, axis=0)    # (48,48,1) → (1,48,48,1)
    return face

# ── Predict emotion from a preprocessed face ─────────────────────────
def predict_emotion(face_tensor):
    """
    Returns predicted emotion label and confidence percentage.
    """
    predictions = model.predict(face_tensor, verbose=0)[0]
    emotion_idx  = np.argmax(predictions)
    emotion_name = index_to_label[emotion_idx]
    confidence   = float(predictions[emotion_idx]) * 100
    all_scores   = {index_to_label[i]: float(predictions[i]) * 100
                    for i in range(len(predictions))}
    return emotion_name, confidence, all_scores

# ── Run full detection pipeline on one image ─────────────────────────
def detect_emotions(image_rgb):
    """
    Takes an RGB image (numpy array),
    returns annotated image + list of detection results.
    """
    image_bgr  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    gray       = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(48, 48)
    )

    results = []

    for (x, y, w, h) in faces:
        face_crop   = gray[y:y+h, x:x+w]
        face_tensor = preprocess_face(face_crop)
        emotion, confidence, all_scores = predict_emotion(face_tensor)
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color, 2)

        # Draw label background
        label = f'{emotion.upper()}  {confidence:.1f}%'
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(image_bgr,
                      (x, y - th - 14),
                      (x + tw + 10, y),
                      color, -1)

        # Draw label text
        cv2.putText(image_bgr, label,
                    (x + 5, y - 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        results.append({
            'emotion':    emotion,
            'confidence': confidence,
            'all_scores': all_scores,
            'bbox':       (x, y, w, h)
        })

    # Convert back to RGB for Streamlit display
    annotated_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return annotated_rgb, results