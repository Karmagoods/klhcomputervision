import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import tempfile, io, os, random
from collections import Counter

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
WORKFLOW_ID = "playground-gemini-3-flash-od"
WORKSPACE = "klhinnovation"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"

# ----------------------
# Helper functions
# ----------------------
def draw_predictions(image, predictions, threshold=50):
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
            colors[cls] = tuple(random.choices(range(256), k=3))

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline=colors[cls], width=3)
        draw.text(
            (left, max(0, top - 14)),
            f"{cls} {conf:.1f}%",
            fill=colors[cls],
            font=FONT
        )

    return draw_image, filtered


def save_temp_image(image):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path)
    return path


def download_button(obj, filename, label):
    if isinstance(obj, (dict, list)):
        import json
        buf = io.BytesIO(json.dumps(obj, indent=2).encode())
    else:
        buf = io.BytesIO()
        obj.save(buf, format="PNG")

    buf.seek(0)
    st.download_button(label, buf, file_name=filename)


# ----------------------
# REQUIRED render()
# ----------------------
def render():
    st.title("🤖 Gemini 3 Flash Object Detection")
    st.caption("Google Gemini 3 Flash via Roboflow Workflows")

    # 🔑 Load secrets
    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    GOOGLE_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

    missing = []
    if not ROBOFLOW_KEY:
        missing.append("ROBOFLOW_API_KEY")
    if not GOOGLE_KEY:
        missing.append("GOOGLE_API_KEY")

    if missing:
        st.error(f"Missing required secrets: {', '.join(missing)}")
        st.stop()

    # ✅ Roboflow client (REQUIRED even for Gemini workflows)
    client = InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_KEY
    )

    # ----------------------
    # Sidebar settings
    # ----------------------
    st.sidebar.header("Detection Settings")

    classes = st.sidebar.multiselect(
        "Classes to detect",
        ["fish", "people", "eyes"],
        default=["fish", "people", "eyes"]
    )

    threshold = st.sidebar.slider(
        "Confidence threshold (%)",
        min_value=0,
        max_value=100,
        value=50
    )

    use_cache = st.sidebar.checkbox("Cache workflow results", True)

    # ----------------------
    # Input selection
    # ----------------------
    mode = st.radio("Input type", ["Upload Image", "Webcam"])
    image = None

    if mode == "Upload Image":
        uploaded = st.file_uploader(
            "Upload an image",
            ["jpg", "jpeg", "png"]
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if not image:
        return

    st.image(image, use_container_width=True)

    # ----------------------
    # Run workflow
    # ----------------------
    if st.button("Run Detection 🎯"):
        with st.spinner("Running Gemini 3 Flash workflow..."):
            path = save_temp_image(image)

            try:
                result = client.run_workflow(
                    workspace_name=WORKSPACE,
                    workflow_id=WORKFLOW_ID,
                    images={"image": path},
                    parameters={
                        "classes": classes,
                        "model_api_key": GOOGLE_KEY
                    },
                    use_cache=use_cache
                )
            finally:
                os.remove(path)

        predictions = result.get("predictions", [])

        annotated, filtered = draw_predictions(
            image,
            predictions,
            threshold
        )

        st.image(annotated, use_container_width=True)

        if filtered:
            st.subheader("Summary")
            counts = Counter(p["class"] for p in filtered)
            for cls, count in counts.items():
                st.write(f"**{cls}**: {count}")

        download_button(annotated, "annotated.png", "⬇ Download Annotated Image")
        download_button(filtered, "predictions.json", "⬇ Download Predictions JSON")

        st.success("Detection complete ✅")
