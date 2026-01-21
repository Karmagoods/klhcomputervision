import streamlit as st
import os
import io
import tempfile
from collections import Counter

from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont

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
def draw_predictions(image, predictions, class_colors, show_boxes=True):
    output = image.copy()
    draw = ImageDraw.Draw(output)

    if not show_boxes:
        return output

    for pred in predictions:
        cls = pred["class"]
        confidence = pred["confidence"] * 100

        x, y = pred["x"], pred["y"]
        w, h = pred["width"], pred["height"]

        color = class_colors.get(cls, (255, 0, 0))

        left = x - w / 2
        top = y - h / 2
        right = x + w / 2
        bottom = y + h / 2

        draw.rectangle([left, top, right, bottom], outline=color, width=3)
        draw.text(
            (left, max(0, top - 14)),
            f"{cls} {confidence:.1f}%",
            fill=color,
            font=FONT,
        )

    return output


def save_temp_image(image):
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    image.save(path)
    return path


def download_button(data, filename, label):
    if isinstance(data, (dict, list)):
        import json
        buffer = io.BytesIO(json.dumps(data, indent=2).encode())
    else:
        buffer = io.BytesIO()
        data.save(buffer, format="PNG")

    buffer.seek(0)
    st.download_button(label, buffer, file_name=filename)


def explain_results(counts):
    parts = []
    for cls, count in counts.items():
        parts.append(f"{count} {cls}{'' if count == 1 else 's'}")
    return "I detected " + " and ".join(parts) + "."

# =======================
# Main app
# =======================
def render():
    st.title("📷 Roboflow Computer Vision Portal")
    st.caption("Interactive object detection & analysis")

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        st.error("ROBOFLOW_API_KEY environment variable not set")
        st.stop()

    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project("your-project")   # 🔁 change
    model = project.version(1).model                  # 🔁 change

    mode = st.radio("Input type", ["Image(s)", "Webcam"])
    show_boxes = st.toggle("👁 Show bounding boxes", value=True)

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
                if not predictions:
                    st.warning("No objects detected.")
                    continue

                # -----------------------
                # CLASS CONTROLS
                # -----------------------
                classes = sorted({p["class"] for p in predictions})

                selected_classes = st.multiselect(
                    "Select objects",
                    classes,
                    default=classes,
                    key=f"classes_{file.name}"
                )

                class_colors = {}
                class_thresholds = {}

                st.subheader("🎛 Class Controls")
                for cls in selected_classes:
                    col1, col2 = st.columns(2)
                    with col1:
                        class_colors[cls] = st.color_picker(
                            f"{cls} color",
                            key=f"color_{cls}_{file.name}"
                        )
                    with col2:
                        class_thresholds[cls] = st.slider(
                            f"{cls} confidence %",
                            0, 100, 50,
                            key=f"conf_{cls}_{file.name}"
                        )

                # -----------------------
                # FILTER PREDICTIONS
                # -----------------------
                filtered = [
                    p for p in predictions
                    if p["class"] in selected_classes
                    and p["confidence"] * 100 >= class_thresholds[p["class"]]
                ]

                annotated = draw_predictions(
                    image,
                    filtered,
                    class_colors,
                    show_boxes
                )

                st.image(annotated, use_container_width=True)

                # -----------------------
                # SUMMARY + AI EXPLANATION
                # -----------------------
                if filtered:
                    counts = Counter(p["class"] for p in filtered)
                    st.subheader("📊 Summary")
                    for cls, count in counts.items():
                        st.write(f"**{cls}**: {count}")

                    st.info("🤖 " + explain_results(counts))

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
            classes = sorted({p["class"] for p in predictions})

            class_colors = {cls: "#ff0000" for cls in classes}
            annotated = draw_predictions(
                image,
                predictions,
                class_colors,
                show_boxes
            )

            st.image(annotated, use_container_width=True)

    st.success("Ready ✅")

# =======================
# Entry point
# =======================
if __name__ == "__main__":
    render()
