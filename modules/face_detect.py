import streamlit as st
import anthropic
import base64
import json
import io
import re
from PIL import Image, ImageDraw, ImageFont

try:
    FONT = ImageFont.truetype("arial.ttf", 15)
except Exception:
    FONT = ImageFont.load_default()


def _image_to_base64(image: Image.Image) -> tuple[str, str]:
    """Convert PIL image to base64 string. Returns (b64_data, media_type)."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    buf.seek(0)
    return base64.standard_b64encode(buf.read()).decode("utf-8"), "image/jpeg"


def _detect_faces_with_claude(image: Image.Image, api_key: str) -> dict:
    """
    Send image to Claude Vision and ask for face detection as structured JSON.
    Returns dict with 'faces' list, each with bounding box as fraction of image size.
    """
    client = anthropic.Anthropic(api_key=api_key)
    b64, media_type = _image_to_base64(image)
    w, h = image.size

    prompt = f"""Analyse this image for human faces. 
The image is {w}px wide and {h}px tall.

For each face you detect, provide:
- x, y: top-left corner of bounding box in PIXELS
- width, height: size of bounding box in PIXELS  
- confidence: your confidence score 0.0 to 1.0
- expression: brief description (e.g. "smiling", "neutral", "surprised")
- approximate age range: e.g. "20s", "child", "elderly"

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "face_count": <integer>,
  "faces": [
    {{
      "x": <int>,
      "y": <int>, 
      "width": <int>,
      "height": <int>,
      "confidence": <float>,
      "expression": "<string>",
      "age_range": "<string>"
    }}
  ],
  "scene_description": "<brief overall description>"
}}

If no faces are detected, return face_count: 0 and an empty faces array."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt}
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


def _draw_faces(image: Image.Image, result: dict) -> Image.Image:
    """Draw bounding boxes and labels on a copy of the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    colors = ["#00FF64", "#00BFFF", "#FF6B6B", "#FFD700", "#DA70D6"]

    for i, face in enumerate(result.get("faces", [])):
        color = colors[i % len(colors)]
        x, y = face["x"], face["y"]
        x2, y2 = x + face["width"], y + face["height"]

        # Clamp to image bounds
        x, y = max(0, x), max(0, y)
        x2, y2 = min(image.width, x2), min(image.height, y2)

        draw.rectangle([x, y, x2, y2], outline=color, width=3)

        conf = face.get("confidence", 0) * 100
        expr = face.get("expression", "")
        age  = face.get("age_range", "")
        label = f"Face {i+1} · {conf:.0f}%"
        if expr:
            label += f" · {expr}"
        if age:
            label += f" ({age})"

        # Label background
        text_y = max(0, y - 20)
        draw.rectangle([x, text_y, x + len(label) * 7, text_y + 18], fill=color)
        draw.text((x + 2, text_y + 1), label, fill="black", font=FONT)

    return annotated


def render():
    st.header("🙂 Face Detection")
    st.caption("AI-powered face detection using Claude Vision — no local dependencies required.")

    ANTHROPIC_KEY = st.secrets.get("ANTHROPIC_API_KEY") or __import__("os").getenv("ANTHROPIC_API_KEY")
    if not ANTHROPIC_KEY:
        st.error("Missing `ANTHROPIC_API_KEY` secret. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()

    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
        if file:
            image = Image.open(file).convert("RGB")
    else:
        cam = st.camera_input("Take a photo")
        if cam:
            image = Image.open(cam).convert("RGB")

    if image is None:
        return

    # Resize if very large to keep API costs low
    max_dim = 1280
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        image = image.resize(
            (int(image.width * ratio), int(image.height * ratio)),
            Image.LANCZOS
        )

    st.image(image, caption="Input image", use_container_width=True)

    if not st.button("Detect Faces 🔍"):
        return

    with st.spinner("Analysing with Claude Vision…"):
        try:
            result = _detect_faces_with_claude(image, ANTHROPIC_KEY)
        except json.JSONDecodeError as e:
            st.error(f"Could not parse Claude's response as JSON: {e}")
            return
        except Exception as e:
            st.error(f"API error: {e}")
            return

    count = result.get("face_count", 0)
    faces = result.get("faces", [])
    scene = result.get("scene_description", "")

    if count == 0:
        st.warning("No faces detected in this image.")
        if scene:
            st.info(f"Scene: {scene}")
        return

    annotated = _draw_faces(image, result)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    with col2:
        st.image(annotated, caption=f"Detected — {count} face(s)", use_container_width=True)

    st.success(f"✅ {count} face(s) detected.")

    if scene:
        st.info(f"**Scene:** {scene}")

    if faces:
        st.subheader("Face Details")
        cols = st.columns(min(len(faces), 3))
        for i, face in enumerate(faces):
            with cols[i % len(cols)]:
                st.markdown(f"**Face {i+1}**")
                st.write(f"Confidence: {face.get('confidence', 0)*100:.0f}%")
                st.write(f"Expression: {face.get('expression', '—')}")
                st.write(f"Age range: {face.get('age_range', '—')}")
                st.write(f"Position: ({face['x']}, {face['y']})")