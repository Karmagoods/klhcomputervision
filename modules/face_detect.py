import streamlit as st
import cv2
import numpy as np
from PIL import Image

def render():
    # Lazy import fixes the AttributeError
    import mediapipe as mp

    st.header("🙂 Face Detection")

    file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if not file:
        return

    image = np.array(Image.open(file).convert("RGB"))

    # Initialize face detection inside the function
    mp_face = mp.solutions.face_detection.FaceDetection()
    results = mp_face.process(image)

    if results.detections:
        for d in results.detections:
            b = d.location_data.relative_bounding_box
            h, w, _ = image.shape
            x, y = int(b.xmin * w), int(b.ymin * h)
            bw, bh = int(b.width * w), int(b.height * h)
            cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    st.image(image, use_container_width=True)
