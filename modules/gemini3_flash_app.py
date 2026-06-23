import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import tempfile, io, os, random
from collections import Counter

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

# ----------------------
# Font setup
# ----------------------
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()

# ----------------------
# Configuration
# ----------------------
WORKFLOW_ID  = "playground-gemini-3-flash-od"
WORKSPACE    = "klhinnovation"
ROBOFLOW_URL = "https://serverless.roboflow.com"


# ----------------------
# Helpers
# ----------------------
def _extract_predictions(result):
    """
    Roboflow workflow results are a list of dicts.
    Dig out the predictions list regardless of nesting.
    """
    if isinstance(result, list) and result:
        item = result[0]
        # Common keys returned by Roboflow workflows
        for key in ("predictions", "output", "outputs", "detections"):
            if key in item:
                val = item[key]
                # val may itself be a dict with a nested predictions list
                if isinstance(val, dict):
                    return val.get("predictions", [])
                if isinstance(val, list):
                    return val
        # Fallback: return first item values if they look like predictions
        return []
    if isinstance(result, dict):
        return result.get("predictions", [])
    return []


def _draw_predictions(image, predictions, threshold=50):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    colors = {}
    filtered = []

    for pred in predictions:
        conf = pred.get("confidence", 0) * 100
        if conf < threshold:
            continue

        cls = pred.get("class", "object")
        filtered.append(pred)

        if cls not in colors:
            colors[cls] = tuple(random.choices(range(60, 240), k=3))

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top     = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline=colors[cls], width=3)
        draw.text(
            (left, max(0, top - 16)),
            f"{cls} {conf:.1f}%",
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


# ----------------------
# render()
# ----------------------
def render():
    st.header("✨ Gemini 3 Flash Object Detection")
    st.caption("Google Gemini 3 Flash via Roboflow Workflows — multi-class detection")

    if InferenceHTTPClient is None:
        st.error("The `inference-sdk` package is not installed. Add it to requirements.txt.")
        st.stop()

    # --- Secrets ---
    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    GOOGLE_KEY   = st.secrets.get("GOOGLE_API_KEY")   or os.getenv("GOOGLE_API_KEY")

    missing = [k for k, v in [("ROBOFLOW_API_KEY", ROBOFLOW_KEY), ("GOOGLE_API_KEY", GOOGLE_KEY)] if not v]
    if missing:
        st.error(f"Missing required secrets: {', '.join(missing)}")
        st.stop()

    client = InferenceHTTPClient(api_url=ROBOFLOW_URL, api_key=ROBOFLOW_KEY)

    # --- Sidebar settings ---
    st.sidebar.subheader("Detection Settings")
    classes   = st.sidebar.multiselect(
        "Classes to detect",
        ["fish", "people", "eyes", "car", "animal", "text"],
        default=["fish", "people", "eyes"]
    )
    threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 50)
    use_cache = st.sidebar.checkbox("Cache workflow results", True)

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

    if not st.button("Run Detection 🎯"):
        return

    with st.spinner("Running Gemini 3 Flash workflow…"):
        path = _save_temp(image)
        try:
            result = client.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": path},
                parameters={"classes": classes, "model_api_key": GOOGLE_KEY},
                use_cache=use_cache
            )
        except Exception as e:
            st.error(f"Workflow error: {e}")
            return
        finally:
            os.remove(path)

    predictions = _extract_predictions(result)

    if not predictions:
        st.warning("No predictions returned. Check your workflow configuration or try a different image.")
        st.json(result)   # Show raw for debugging
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