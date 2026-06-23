import streamlit as st
import numpy as np
from PIL import Image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None


def _process_image(image_array):
    """Run MediaPipe face detection on a numpy RGB array. Returns annotated array + count."""
    output = image_array.copy()
    h, w, _ = output.shape
    face_count = 0

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,           # 1 = full-range model (better for varied distances)
        min_detection_confidence=0.5
    ) as detector:
        results = detector.process(image_array)

        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box
                x = max(0, int(box.xmin * w))
                y = max(0, int(box.ymin * h))
                bw = int(box.width * w)
                bh = int(box.height * h)

                cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 100), 2)

                score = detection.score[0] * 100
                label = f"Face {score:.1f}%"
                cv2.putText(
                    output, label,
                    (x, max(15, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 100), 2
                )
                face_count += 1

    return output, face_count


def render():
    st.header("🙂 Face Detection")

    if cv2 is None or mp is None:
        st.error(
            "Face detection requires **OpenCV** and **MediaPipe**.\n\n"
            "These libraries are not available in this cloud environment. "
            "Run locally or in Docker."
        )
        st.stop()

    st.caption("Upload an image or take a photo to detect faces using MediaPipe.")

    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)

    image_array = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if file:
            image_array = np.array(Image.open(file).convert("RGB"))
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image_array = np.array(Image.open(cam).convert("RGB"))

    if image_array is None:
        return

    with st.spinner("Detecting faces…"):
        annotated, count = _process_image(image_array)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image_array, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption=f"Detected — {count} face(s)", use_container_width=True)

    if count:
        st.success(f"✅ {count} face(s) detected.")
    else:
        st.warning("No faces detected. Try a clearer image or adjust lighting.")