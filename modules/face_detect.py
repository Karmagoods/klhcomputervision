import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()

_import_error = None
try:
    import cv2
except Exception as e:
    cv2 = None
    _import_error = str(e)


def _detect_faces(image: Image.Image, scale_factor: float, min_neighbors: int, min_size: int):
    """
    Run OpenCV Haar Cascade face detection.
    Uses frontalface_default + profileface cascades for better coverage.
    Returns annotated PIL image and list of detected face rects.
    """
    img_array = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)  # Improves detection in varied lighting

    frontal = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    alt     = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    profile = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    params = dict(
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=(min_size, min_size),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces_f = frontal.detectMultiScale(gray, **params)
    faces_a = alt.detectMultiScale(gray, **params)
    faces_p = profile.detectMultiScale(gray, **params)

    # Merge all detections, deduplicate overlapping boxes
    all_faces = []
    for group in [faces_f, faces_a, faces_p]:
        if len(group):
            all_faces.extend(group.tolist())

    unique = _deduplicate(all_faces, overlap_thresh=0.3)

    # Draw on a PIL image
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    colors = ["#00FF64", "#00BFFF", "#FFD700", "#FF6B6B", "#DA70D6"]

    for i, (x, y, w, h) in enumerate(unique):
        color = colors[i % len(colors)]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        label = f"Face {i + 1}"
        text_y = max(0, y - 20)
        draw.rectangle([x, text_y, x + len(label) * 8, text_y + 17], fill=color)
        draw.text((x + 2, text_y + 1), label, fill="black", font=FONT)

    return annotated, unique


def _iou(a, b):
    """Intersection over union for two (x, y, w, h) boxes."""
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0


def _deduplicate(boxes, overlap_thresh=0.3):
    """Remove duplicate overlapping boxes."""
    unique = []
    for box in boxes:
        if not any(_iou(box, u) > overlap_thresh for u in unique):
            unique.append(box)
    return unique


def render():
    st.header("🙂 Face Detection")
    st.caption("Powered by OpenCV Haar Cascades — runs entirely free, no API key needed.")

    if cv2 is None:
        st.error("OpenCV is not available.")
        if _import_error:
            st.code(_import_error)
        st.stop()

    # Settings
    with st.expander("⚙️ Detection Settings", expanded=False):
        scale_factor  = st.slider("Scale factor", 1.05, 1.5, 1.1, 0.05,
                                  help="Lower = more thorough but slower")
        min_neighbors = st.slider("Min neighbours", 1, 10, 5,
                                  help="Higher = fewer false positives")
        min_size      = st.slider("Min face size (px)", 20, 150, 30,
                                  help="Ignore faces smaller than this")

    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image = None

    if mode == "Upload Image":
        f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if f:
            image = Image.open(f)
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam)

    if image is None:
        return

    with st.spinner("Detecting faces…"):
        annotated, faces = _detect_faces(image, scale_factor, min_neighbors, min_size)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption=f"Detected — {len(faces)} face(s)", use_container_width=True)

    if faces:
        st.success(f"✅ {len(faces)} face(s) detected.")
        st.subheader("Face Locations")
        cols = st.columns(min(len(faces), 4))
        for i, (x, y, w, h) in enumerate(faces):
            cols[i % len(cols)].markdown(f"**Face {i+1}**  \nPos: ({x}, {y})  \nSize: {w}×{h}px")
    else:
        st.warning(
            "No faces detected. Try:\n"
            "- Lowering **Min neighbours** (reduces strictness)\n"
            "- Lowering **Scale factor** (more thorough scan)\n"
            "- Ensuring the face is well-lit and frontal"
        )

    # Download annotated image
    buf = io.BytesIO()
    annotated.save(buf, format="PNG")
    buf.seek(0)
    st.download_button("⬇ Download Annotated Image", buf, file_name="faces_detected.png")