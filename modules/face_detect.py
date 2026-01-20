import streamlit as st
import numpy as np
from PIL import Image

# --- Guard OpenCV ---
try:
    import cv2
except Exception:
    cv2 = None

# --- Guard MediaPipe ---
try:
    import mediapipe as mp
except Exception:
    mp = None


def render():
    st.header("🙂 Face Detection")

    # --- Environment safety check ---
    if cv2 is None or mp is None:
        st.error(
            "Face detection requires OpenCV and MediaPipe.\n\n"
            "These libraries are not available in this environment.\n"
            "Please run locally, in Docker, or via a backend service."
        )
        st.stop()

    st.caption("Upload an image to detect faces.")

    file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if not file:
        return

    # --- Load image safely ---
    image = np.array(Image.open(file).convert("RGB"))
    output = image.copy()
    h, w, _ = output.shape

    # --- Initialize MediaPipe face detector ---
    with mp.solutions.face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detector:

        results = face_detector.process(image)

        if results.detections:
            for detection in results.detections:
                box = detection.location_data.relative_bounding_box

                x = int(box.xmin * w)
                y = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)

                # Clamp values (safety)
                x = max(0, x)
                y = max(0, y)

                cv2.rectangle(
                    output,
                    (x, y),
                    (x + bw, y + bh),
                    (0, 255, 0),
                    2
                )

                score = detection.score[0] * 100
                label = f"Face {score:.1f}%"

                cv2.putText(
                    output,
                    label,
                    (x, max(15, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

    st.image(output, caption="Detected Faces", use_container_width=True)
    st.success("Detection complete ✅")
