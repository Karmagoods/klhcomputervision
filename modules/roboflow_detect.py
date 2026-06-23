import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile, io, os, random
from collections import Counter

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

# ----------------------
# Font
# ----------------------
try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()

ROBOFLOW_URL = "https://serverless.roboflow.com"

# Curated public Roboflow models (model_id format: workspace/model/version)
PRESET_MODELS = {
    "COCO (YOLOv8n) — 80 classes": "coco/3",
    "People Detection": "people-detection-o4rdr/9",
    "Face Detection": "face-detection-mik1i/18",
    "Vehicle Detection": "vehicle-detection-3mmwj/1",
    "Custom (enter below)": "__custom__",
}


def _draw_predictions(image, predictions, threshold, label_map=None):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    colors = {}
    filtered = []

    for pred in predictions:
        conf = pred.get("confidence", 0) * 100
        if conf < threshold:
            continue

        cls = pred.get("class", "object")
        display = label_map.get(cls, cls) if label_map else cls
        filtered.append(pred)

        if cls not in colors:
            colors[cls] = tuple(random.choices(range(50, 230), k=3))

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top     = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline=colors[cls], width=3)
        draw.text(
            (left, max(0, top - 16)),
            f"{display} {conf:.1f}%",
            fill=colors[cls],
            font=FONT
        )

    return draw_image, filtered


def _save_temp(image):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path, quality=95)
    return path


def _download_button(obj, filename, label):
    if isinstance(obj, (dict, list)):
        import json
        buf = io.BytesIO(json.dumps(obj, indent=2).encode())
    else:
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(label, buf, file_name=filename)


def render():
    st.header("🔍 Roboflow Object Detection")
    st.caption("Run any Roboflow model via the Inference API.")

    if InferenceHTTPClient is None:
        st.error("The `inference-sdk` package is not installed. Add it to requirements.txt.")
        st.stop()

    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_KEY:
        st.error("Missing `ROBOFLOW_API_KEY` secret.")
        st.stop()

    client = InferenceHTTPClient(api_url=ROBOFLOW_URL, api_key=ROBOFLOW_KEY)

    # --- Sidebar ---
    st.sidebar.subheader("Model Settings")

    model_label = st.sidebar.selectbox("Preset model", list(PRESET_MODELS.keys()))
    model_id = PRESET_MODELS[model_label]

    if model_id == "__custom__":
        model_id = st.sidebar.text_input(
            "Custom model ID",
            placeholder="workspace/model/version",
            help="e.g. klhinnovation/my-model/1"
        )

    threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 40)

    # --- Input ---
    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image = None

    if mode == "Upload Image":
        uploaded = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if not image:
        return

    st.image(image, caption="Input image", use_container_width=True)

    if not model_id or model_id == "__custom__":
        st.info("Enter a model ID in the sidebar to continue.")
        return

    if not st.button("Run Detection 🎯"):
        return

    with st.spinner(f"Running `{model_id}`…"):
        path = _save_temp(image)
        try:
            result = client.infer(path, model_id=model_id)
        except Exception as e:
            st.error(f"Inference error: {e}")
            return
        finally:
            os.remove(path)

    predictions = result.get("predictions", [])

    if not predictions:
        st.warning("No objects detected above the threshold. Try a lower confidence setting.")
        return

    annotated, filtered = _draw_predictions(image, predictions, threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Annotated", use_container_width=True)

    if filtered:
        st.subheader("Detection Summary")
        counts = Counter(p["class"] for p in filtered)
        cols = st.columns(min(len(counts), 4))
        for i, (cls, cnt) in enumerate(counts.items()):
            cols[i % len(cols)].metric(cls.title(), cnt)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        _download_button(annotated, "annotated.png", "⬇ Download Annotated Image")
    with c2:
        _download_button(filtered, "predictions.json", "⬇ Download Predictions JSON")

    st.success("✅ Detection complete.")