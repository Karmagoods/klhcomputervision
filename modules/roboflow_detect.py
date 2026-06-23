import streamlit as st
import requests
import base64
import io
import os
import random
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()

ROBOFLOW_INFER_URL = "https://detect.roboflow.com"

PRESET_MODELS = {
    "COCO YOLOv8n — 80 classes": {"model": "coco", "version": "3"},
    "People Detection":           {"model": "people-detection-o4rdr", "version": "9"},
    "Face Detection":             {"model": "face-detection-mik1i", "version": "18"},
    "Vehicle Detection":          {"model": "vehicle-detection-3mmwj", "version": "1"},
    "Custom (enter below)":       None,
}


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _run_inference(image: Image.Image, model: str, version: str, api_key: str, confidence: int) -> list:
    """Call Roboflow infer REST endpoint directly — no SDK required."""
    url = f"{ROBOFLOW_INFER_URL}/{model}/{version}"
    b64 = _image_to_base64(image)

    resp = requests.post(
        url,
        params={"api_key": api_key, "confidence": confidence / 100},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=b64,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("predictions", [])


def _draw_predictions(image: Image.Image, predictions: list, threshold: int) -> tuple:
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    colors = {}
    filtered = []

    for pred in predictions:
        conf = pred.get("confidence", 0) * 100
        if conf < threshold:
            continue

        cls = pred.get("class", "object")
        filtered.append(pred)

        if cls not in colors:
            colors[cls] = tuple(random.choices(range(50, 230), k=3))

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top     = int(x - w / 2), int(y - h / 2)
        right, bottom = int(x + w / 2), int(y + h / 2)

        draw.rectangle([left, top, right, bottom], outline=colors[cls], width=3)
        label = f"{cls} {conf:.1f}%"
        text_y = max(0, top - 18)
        draw.rectangle([left, text_y, left + len(label) * 7, text_y + 16], fill=colors[cls])
        draw.text((left + 2, text_y), label, fill="black", font=FONT)

    return annotated, filtered


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
    st.caption("Run any Roboflow model via direct REST API — no SDK required.")

    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    if not ROBOFLOW_KEY:
        st.error("Missing `ROBOFLOW_API_KEY` secret.")
        st.stop()

    # Sidebar
    st.sidebar.subheader("Model Settings")
    model_label = st.sidebar.selectbox("Preset model", list(PRESET_MODELS.keys()))
    model_cfg = PRESET_MODELS[model_label]

    custom_model = custom_version = None
    if model_cfg is None:
        custom_input = st.sidebar.text_input("Model ID", placeholder="my-model")
        custom_version = st.sidebar.text_input("Version", placeholder="1")
        if custom_input and custom_version:
            custom_model = custom_input

    threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 40)

    # Input
    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image = None

    if mode == "Upload Image":
        f = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if f:
            image = Image.open(f).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if image is None:
        return

    st.image(image, caption="Input image", use_container_width=True)

    # Resolve model/version
    if model_cfg:
        model, version = model_cfg["model"], model_cfg["version"]
    elif custom_model and custom_version:
        model, version = custom_model, custom_version
    else:
        st.info("Enter a model ID and version in the sidebar.")
        return

    if not st.button("Run Detection 🎯"):
        return

    with st.spinner(f"Running `{model}/{version}`…"):
        try:
            predictions = _run_inference(image, model, version, ROBOFLOW_KEY, threshold)
        except requests.HTTPError as e:
            st.error(f"Roboflow API error {e.response.status_code}: {e.response.text[:300]}")
            return
        except Exception as e:
            st.error(f"Request failed: {e}")
            return

    if not predictions:
        st.warning("No objects detected. Try lowering the confidence threshold.")
        return

    annotated, filtered = _draw_predictions(image, predictions, threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption="Annotated", use_container_width=True)

    counts = Counter(p["class"] for p in filtered)
    st.subheader("Detection Summary")
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