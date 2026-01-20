import streamlit as st
import os
import io
import tempfile
import random
from collections import Counter

from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# =======================
# Font setup
# =======================
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except Exception:
    FONT = ImageFont.load_default()


# =======================
# Helper functions
# =======================
def draw_predictions(image, predictions, threshold=50):
    """
    Draw bounding boxes and labels on a PIL image.
    Returns annotated image and filtered predictions.
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)
    colors = {}
    filtered = []

    for pred in predictions:
        confidence = pred.get("confidence", 0) * 100
        if confidence < threshold:
            continue

        filtered.append(pred)

        cls = pred.get("class", "object")
        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]

        if cls not in colors:
            colors[cls] = tuple(random.randint(0, 255) for _ in range(3))

        color = colors[cls]

        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        label = f"{cls} {confidence:.1f}%"
        draw.text((left, max(0, top - 14)), label, fill=color, font=FONT)

    return output, filtered


def save_temp_image(image):
    """Save PIL image to a temp file (Streamlit Cloud safe)."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path)
    return path


def download_button(data, filename, label):
    """Download image or JSON data."""
    if isinstance(data, (dict, list)):
        import json
        buffer = io.BytesIO(json.dumps(data, indent=2).encode())
    else:
        buffer = io.BytesIO()
        data.save(buffer, format="PNG")

    buffer.seek(0)
    st.download_button(label, buffer, file_name=filename)


# =======================
# Main app
# =======================
def render():
    st.title("📷 Roboflow Computer Vision Portal")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("ROBOFLOW_API_KEY environment variable not set")
        st.stop()

    # --- Roboflow setup ---
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("your-project")   # 🔁 change if needed
    model = project.version(1).model                  # 🔁 change if needed

    threshold = st.slider("Minimum confidence (%)", 0, 100, 50)
    mode = st.radio("Input type", ["Image(s)", "Webcam"])

    # =======================
    # IMAGE MODE
    # =======================
    if mode == "Image(s)":
        files = st.file_uploader(
            "Upload image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        for file in files or []:
            image = Image.open(file).convert("RGB")
            st.subheader(file.name)
            st.image(image, use_container_width=True)

            if st.button(f"Run Detection 🎯 ({file.name})", key=file.name):
                with st.spinner("Running inference..."):
                    path = save_temp_image(image)
                    result = model.predict(path).json()
                    os.remove(path)

                predictions = result.get("predictions", [])
                annotated, filtered = draw_predictions(
                    image, predictions, threshold
                )

                st.image(annotated, use_container_width=True)

                if filtered:
                    st.subheader("Prediction Summary")
                    counts = Counter(p["class"] for p in filtered)
                    for cls, count in counts.items():
                        st.write(f"**{cls}**: {count}")

                download_button(
                    annotated,
                    f"{file.name}_annotated.png",
                    "Download annotated image"
                )
                download_button(
                    filtered,
                    f"{file.name}_predictions.json",
                    "Download predictions (JSON)"
                )

    # =======================
    # WEBCAM MODE
    # =======================
    else:
        cam = st.camera_input("Take a photo")

        if cam:
            image = Image.open(cam).convert("RGB")
            st.image(image, use_container_width=True)

            with st.spinner("Running inference..."):
                path = save_temp_image(image)
                result = model.predict(path).json()
                os.remove(path)

            predictions = result.get("predictions", [])
            annotated, _ = draw_predictions(
                image, predictions, threshold
            )

            st.image(annotated, caption="Predictions", use_container_width=True)

    st.success("Ready ✅")


# =======================
# Entry point
# =======================
if __name__ == "__main__":
    render()
