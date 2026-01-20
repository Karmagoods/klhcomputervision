import streamlit as st
import numpy as np
from PIL import Image

# --- OpenCV guard (prevents hard crash on cloud imports) ---
try:
    import cv2
except Exception:
    cv2 = None


def render():
    st.header("🏃 Motion Detection (Frame Difference)")

    if cv2 is None:
        st.error(
            "OpenCV is not available in this environment.\n\n"
            "Motion detection requires a local or Docker-based deployment."
        )
        st.stop()

    st.caption("Upload two images sequentially to detect motion between them.")

    file = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png"]
    )

    if not file:
        return

    # --- Load image safely ---
    image = np.array(Image.open(file).convert("RGB"))

    # --- Preprocessing ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # --- First frame ---
    if "prev_frame" not in st.session_state:
        st.session_state.prev_frame = blur
        st.info("First frame stored. Upload another image to detect motion.")
        st.image(image, caption="Reference Frame", use_container_width=True)
        return

    # --- Motion detection ---
    diff = cv2.absdiff(st.session_state.prev_frame, blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Optional cleanup
    thresh = cv2.dilate(thresh, None, iterations=2)

    # --- Display results ---
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Current Frame", use_container_width=True)

    with col2:
        st.image(thresh, caption="Motion Mask", use_container_width=True)

    # --- Reset / update state ---
    st.session_state.prev_frame = blur

    if st.button("🔄 Reset Reference Frame"):
        del st.session_state.prev_frame
        st.success("Reference frame reset. Upload a new image.")

    st.success("Motion detection complete ✅")
