import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont
import tempfile, io
from collections import Counter
import random
import os

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

# Load Google API key from Streamlit secrets or environment variable
GOOGLE_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GOOGLE_KEY:
    st.error("Please set your Google API key in Streamlit secrets or .env as GOOGLE_API_KEY")
    st.stop()

# ----------------------
# Roboflow Gemini 3 Flash client
# ----------------------
# Using an empty api_key because we rely entirely on Google API
client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key="")

# ----------------------
# Helper functions
# ----------------------
def draw_predictions(image: Image.Image, predictions: list, threshold: float = 50):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    colors = {}
    filtered = []

    for pred in predictions:
        conf = pred.get("confidence", 0) * 100
        if conf < threshold:
            continue
        filtered.append(pred)
        cls = pred.get("class", "object")

        if cls not in colors:
            colors[cls] = tuple(random.choices(range(256), k=3))

        color = colors[cls]

        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]
        left, top = x - w/2, y - h/2
        right, bottom = x + w/2, y + h/2

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, max(0, top - 14)), f"{cls} {conf:.1f}%", fill=color, font=FONT)

    return draw_image, filtered

def save_temp_image(image: Image.Image):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path)
    return path

def download_button(obj, filename, label):
    if isinstance(obj, (dict, list)):
        import json
        b = io.BytesIO(json.dumps(obj, indent=2).encode())
    else:
        b = io.BytesIO()
        obj.save(b, format="PNG")
    b.seek(0)
    st.download_button(label, b, file_name=filename)

# ----------------------
# Streamlit app
# ----------------------
st.title("🤖 Gemini 3 Flash Object Detection (Google-only)")

st.sidebar.header("Settings")
classes = st.sidebar.multiselect("Classes to detect", ["fish","people","eyes"], default=["fish","people","eyes"])
threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 50)
use_cache = st.sidebar.checkbox("Cache workflow definition", True)

st.header("Upload an Image or Use Webcam")
mode = st.radio("Input type", ["Upload Image", "Webcam"])

image = None

if mode == "Upload Image":
    uploaded = st.file_uploader("Select an image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
elif mode == "Webcam":
    cam_image = st.camera_input("Take a photo")
    if cam_image:
        image = Image.open(cam_image).convert("RGB")

if image:
    st.image(image, caption="Input Image", use_container_width=True)

    if st.button("Run Detection 🎯"):
        with st.spinner("Running inference..."):
            path = save_temp_image(image)

            # Run workflow using Google API key
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

            os.remove(path)

            predictions = result.get("predictions", [])
            annotated, filtered = draw_predictions(image, predictions, threshold)

            st.subheader("Annotated Image")
            st.image(annotated, use_container_width=True)

            if filtered:
                st.subheader("Prediction Summary")
                counts = Counter(p["class"] for p in filtered)
                for cls, count in counts.items():
                    st.write(f"**{cls}**: {count}")

            download_button(annotated, "annotated_image.png", "Download Image")
            download_button(filtered, "predictions.json", "Download JSON")

            st.success("Detection complete ✅")
