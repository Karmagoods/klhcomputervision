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
    st.title("🤖 Gemini 3 Flash Object Detection (Google-only)")
    st.caption("Powered by Google Gemini 3 Flash via Roboflow Workflows")

    GOOGLE_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_KEY:
        st.error("GOOGLE_API_KEY not set in secrets or .env")
        st.stop()

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=""  # Google handles auth
    )

    st.sidebar.header("Settings")
    classes = st.sidebar.multiselect(
        "Classes to detect",
        ["fish", "people", "eyes"],
        default=["fish", "people", "eyes"]
    )
    threshold = st.sidebar.slider("Confidence threshold (%)", 0, 100, 50)
    use_cache = st.sidebar.checkbox("Cache workflow", True)

    mode = st.radio("Input type", ["Upload Image", "Webcam"])
    image = None

    if mode == "Upload Image":
        uploaded = st.file_uploader("Upload image", ["jpg", "jpeg", "png"])
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if not image:
        return

    st.image(image, use_container_width=True)

    if st.button("Run Detection 🎯"):
        with st.spinner("Running Gemini 3 Flash..."):
            path = save_temp_image(image)

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

        st.image(annotated, use_container_width=True)

        if filtered:
            st.subheader("Summary")
            counts = Counter(p["class"] for p in filtered)
            for k, v in counts.items():
                st.write(f"**{k}**: {v}")

        download_button(annotated, "annotated.png", "Download Image")
        download_button(filtered, "predictions.json", "Download JSON")

        st.success("Done ✅")
