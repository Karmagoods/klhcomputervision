import streamlit as st
import requests
import base64
import io
import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
from collections import Counter

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()

WORKFLOW_ID  = "playground-gemini-3-flash-od"
WORKSPACE    = "klhinnovation"
ROBOFLOW_URL = "https://serverless.roboflow.com"


def _image_to_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _run_workflow(image: Image.Image, roboflow_key: str, google_key: str,
                  classes: list, use_cache: bool) -> list:
    """Call Roboflow Workflow REST endpoint directly."""
    url = f"{ROBOFLOW_URL}/{WORKSPACE}/workflows/{WORKFLOW_ID}"
    b64 = _image_to_base64(image)

    payload = {
        "api_key": roboflow_key,
        "inputs": {
            "image": {"type": "base64", "value": b64},
            "classes": classes,
            "model_api_key": google_key,
        },
        "use_cache": use_cache,
    }

    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    # Roboflow workflow responses: {"outputs": [...]} or {"predictions": [...]}
    if "outputs" in data:
        outputs = data["outputs"]
        if isinstance(outputs, list) and outputs:
            item = outputs[0]
            for key in ("predictions", "detections", "output"):
                if key in item:
                    val = item[key]
                    if isinstance(val, dict):
                        return val.get("predictions", [])
                    if isinstance(val, list):
                        return val
    if "predictions" in data:
        return data["predictions"]
    return []


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
            colors[cls] = tuple(random.choices(range(60, 220), k=3))

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
        buf = io.BytesIO(json.dumps(obj, indent=2).encode())
    else:
        buf = io.BytesIO()
        obj.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(label, buf, file_name=filename)


def render():
    st.header("✨ Gemini 3 Flash Object Detection")
    st.caption("Google Gemini 3 Flash via Roboflow Workflows — direct REST API.")

    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    GOOGLE_KEY   = st.secrets.get("GOOGLE_API_KEY")   or os.getenv("GOOGLE_API_KEY")

    missing = [k for k, v in [("ROBOFLOW_API_KEY", ROBOFLOW_KEY), ("GOOGLE_API_KEY", GOOGLE_KEY)] if not v]
    if missing:
        st.error(f"Missing secrets: {', '.join(missing)}")
        st.stop()

    st.sidebar.subheader("Detection Settings")
    classes   = st.sidebar.multiselect(
        "Classes to detect",
        ["fish", "people", "eyes", "car", "animal", "text"],
        default=["fish", "people", "eyes"]
    )
    threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 50)
    use_cache = st.sidebar.checkbox("Cache workflow results", True)

    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image = None

    if mode == "Upload Image":
        f = st.file_uploader("Upload an image", ["jpg", "jpeg", "png"])
        if f:
            image = Image.open(f).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if image is None:
        return

    st.image(image, caption="Input image", use_container_width=True)

    if not st.button("Run Detection 🎯"):
        return

    with st.spinner("Running Gemini 3 Flash workflow…"):
        try:
            predictions = _run_workflow(image, ROBOFLOW_KEY, GOOGLE_KEY, classes, use_cache)
        except requests.HTTPError as e:
            st.error(f"Roboflow API error {e.response.status_code}: {e.response.text[:300]}")
            return
        except Exception as e:
            st.error(f"Request failed: {e}")
            return

    if not predictions:
        st.warning("No predictions returned. Check your workflow configuration or try a different image.")
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