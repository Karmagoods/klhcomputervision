import streamlit as st
import streamlit as st
import os
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import tempfile, random, io
from collections import Counter
import cv2
import numpy as np

# -----------------------
# Font
# -----------------------
try:
    FONT = ImageFont.truetype("arial.ttf", 16)
except:
    FONT = ImageFont.load_default()

# -----------------------
# Helpers
# -----------------------
def draw_predictions(image, predictions, threshold=50):
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    colors = {}
    filtered = []

    for pred in predictions:
        conf = pred.get("confidence", 0) * 100
        if conf < threshold:
            continue

        filtered.append(pred)

        x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
        cls = pred.get("class", "object")

        if cls not in colors:
            colors[cls] = tuple(random.choices(range(256), k=3))

        color = colors[cls]

        left, top = x - w / 2, y - h / 2
        right, bottom = x + w / 2, y + h / 2

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text((left, top - 15), f"{cls} {conf:.1f}%", fill=color, font=FONT)

    return draw_image, filtered


def download_button(obj, filename, label):
    if isinstance(obj, (dict, list)):
        import json
        b = io.BytesIO(json.dumps(obj, indent=2).encode())
    else:
        b = io.BytesIO()
        obj.save(b, format="PNG")

    b.seek(0)
    st.download_button(label, b, file_name=filename)


def save_temp_image(image):
    """Windows-safe temp image writer"""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path)
    return path

# -----------------------
# App
# -----------------------
def render():
    st.header("📷 Roboflow Computer Vision Portal")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("ROBOFLOW_API_KEY not set")
        st.stop()

    # ✅ Stable Roboflow usage (same pattern that worked)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("your-project")   # ← change if needed
    model = project.version(1).model                   # ← change if needed

    threshold = st.slider("Minimum confidence %", 0, 100, 50)
    mode = st.radio("Input type", ["Image(s)", "Video", "Webcam"])

    # -----------------------
    # IMAGE MODE
    # -----------------------
    if mode == "Image(s)":
        files = st.file_uploader(
            "Upload image(s)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        for file in files or []:
            image = Image.open(file)
            st.subheader(file.name)
            st.image(image, use_container_width=True)

            if st.button(f"Run Detection 🎯 ({file.name})", key=file.name):
                with st.spinner("Running inference..."):
                    path = save_temp_image(image)
                    result = model.predict(path).json()
                    os.remove(path)

                preds = result.get("predictions", [])
                draw_img, filtered = draw_predictions(image, preds, threshold)
                st.image(draw_img, use_container_width=True)

                counts = Counter(p["class"] for p in filtered)
                if counts:
                    st.subheader("Prediction Summary")
                    for k, v in counts.items():
                        st.write(f"{k}: {v}")

                download_button(draw_img, f"{file.name}_annotated.png", "Download Image")
                download_button(filtered, f"{file.name}_predictions.json", "Download JSON")

    # -----------------------
    # VIDEO MODE (first frame)
    # -----------------------
    elif mode == "Video":
        video = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

        if video:
            fd, vpath = tempfile.mkstemp()
            os.close(fd)
            with open(vpath, "wb") as f:
                f.write(video.read())

            cap = cv2.VideoCapture(vpath)
            success, frame = cap.read()
            cap.release()
            os.remove(vpath)

            if success:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                st.image(image, caption="First frame", use_container_width=True)

                with st.spinner("Running inference..."):
                    path = save_temp_image(image)
                    result = model.predict(path).json()
                    os.remove(path)

                preds = result.get("predictions", [])
                draw_img, _ = draw_predictions(image, preds, threshold)
                st.image(draw_img, caption="Predictions", use_container_width=True)

    # -----------------------
    # WEBCAM MODE
    # -----------------------
    elif mode == "Webcam":
        cam = st.camera_input("Take a photo")

        if cam:
            image = Image.open(cam)
            st.image(image, use_container_width=True)

            with st.spinner("Running inference..."):
                path = save_temp_image(image)
                result = model.predict(path).json()
                os.remove(path)

            preds = result.get("predictions", [])
            draw_img, _ = draw_predictions(image, preds, threshold)
            st.image(draw_img, caption="Predictions", use_container_width=True)

    st.success("Ready ✅")


if __name__ == "__main__":
    render()
