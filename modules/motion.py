import streamlit as st
import numpy as np
from PIL import Image

_import_error = None
try:
    import cv2
except Exception as e:
    cv2 = None
    _import_error = str(e)


def _to_gray_blur(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def _detect_motion(frame_a, frame_b, threshold=25):
    """
    Compare two blurred grayscale frames.
    Returns: diff image, binary mask, contour-annotated colour image, motion %
    """
    diff = cv2.absdiff(frame_a, frame_b)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, None, iterations=2)

    # Motion percentage
    motion_pct = (np.count_nonzero(mask) / mask.size) * 100

    # Draw contours on a blank colour canvas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    cv2.drawContours(canvas, contours, -1, (0, 255, 100), 2)

    return diff, mask, canvas, motion_pct, contours


def render():
    st.header("🏃 Motion Detection")

    if cv2 is None:
        st.error("Motion detection requires OpenCV.")
        if _import_error:
            st.code(_import_error, language="text")
            st.info("Ensure `packages.txt` contains `libgl1-mesa-glx` and `libglib2.0-0`.")
        st.stop()

    st.caption("Compare two frames to detect motion using frame differencing.")

    # --- Settings ---
    with st.expander("⚙️ Settings", expanded=False):
        threshold = st.slider("Pixel difference threshold", 5, 100, 25,
                              help="Lower = more sensitive to subtle motion")
        min_contour_area = st.slider("Min contour area (px²)", 100, 5000, 500,
                                     help="Filter out tiny noise blobs")

    # --- Input mode ---
    mode = st.radio("Input mode", ["Upload Two Images", "Webcam Sequence"], horizontal=True)

    frame_a = frame_b = None

    if mode == "Upload Two Images":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Frame A — Reference**")
            file_a = st.file_uploader("Reference frame", type=["jpg", "jpeg", "png"],
                                      key="motion_a")
            if file_a:
                frame_a = np.array(Image.open(file_a).convert("RGB"))
                st.image(frame_a, use_container_width=True)

        with col2:
            st.markdown("**Frame B — Comparison**")
            file_b = st.file_uploader("Comparison frame", type=["jpg", "jpeg", "png"],
                                      key="motion_b")
            if file_b:
                frame_b = np.array(Image.open(file_b).convert("RGB"))
                st.image(frame_b, use_container_width=True)

    else:
        st.info("Take two photos in sequence. The first becomes the reference frame.")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Frame A — Reference**")
            cam_a = st.camera_input("Reference photo", key="cam_a")
            if cam_a:
                frame_a = np.array(Image.open(cam_a).convert("RGB"))

        with col2:
            st.markdown("**Frame B — Comparison**")
            cam_b = st.camera_input("Comparison photo", key="cam_b")
            if cam_b:
                frame_b = np.array(Image.open(cam_b).convert("RGB"))

    # --- Run detection ---
    if frame_a is None or frame_b is None:
        st.info("Provide both frames to run motion detection.")
        return

    # Resize B to match A if needed
    if frame_a.shape != frame_b.shape:
        pil_b = Image.fromarray(frame_b).resize(
            (frame_a.shape[1], frame_a.shape[0]), Image.LANCZOS
        )
        frame_b = np.array(pil_b)

    with st.spinner("Analysing motion…"):
        blur_a = _to_gray_blur(frame_a)
        blur_b = _to_gray_blur(frame_b)
        diff, mask, contour_canvas, motion_pct, contours = _detect_motion(
            blur_a, blur_b, threshold
        )

        # Filter small contours
        significant = [c for c in contours if cv2.contourArea(c) >= min_contour_area]

    st.divider()
    st.subheader("Results")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(diff, caption="Pixel Difference", use_container_width=True, clamp=True)
    with c2:
        st.image(mask, caption="Motion Mask", use_container_width=True)
    with c3:
        st.image(contour_canvas, caption="Motion Contours", use_container_width=True)

    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Motion Coverage", f"{motion_pct:.2f}%")
    m2.metric("Significant Motion Regions", len(significant))

    if motion_pct < 1:
        st.success("✅ Minimal motion detected between frames.")
    elif motion_pct < 10:
        st.warning(f"⚠️ Moderate motion detected ({motion_pct:.1f}% of frame).")
    else:
        st.error(f"🚨 High motion detected ({motion_pct:.1f}% of frame).")