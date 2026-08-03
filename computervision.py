import os
import datetime
import streamlit as st
from pathlib import Path
from PIL import Image
from collections import Counter

from services.supabase_client import supabase
from services.roboflow_workflow_client import (
    run_bar_pub_workflow,
    WorkflowResult,
    WorkflowError,
    RoboflowAPIError,
    NetworkError,
)


def render():
    st.subheader("✨ Gemini Flash Bar & Pub AI")
    st.write(
        "COCO object detection + Google Gemini conversational analysis, "
        "powered by Roboflow Serverless Workflows."
    )
    st.divider()

    # ------------------------------------------------------------------
    # API keys — follow repo pattern: st.secrets first, then os.getenv
    # ------------------------------------------------------------------
    ROBOFLOW_KEY = st.secrets.get("ROBOFLOW_API_KEY") or os.getenv("ROBOFLOW_API_KEY")
    GOOGLE_KEY   = st.secrets.get("GOOGLE_API_KEY")   or os.getenv("GOOGLE_API_KEY")

    missing = [k for k, v in [("ROBOFLOW_API_KEY", ROBOFLOW_KEY), ("GOOGLE_API_KEY", GOOGLE_KEY)] if not v]
    if missing:
        st.error(f"Missing secrets: {', '.join(missing)}")
        st.stop()

    # ------------------------------------------------------------------
    # User controls
    # ------------------------------------------------------------------
    user_prompt = st.text_input(
        "Ask Gemini about the scene:",
        placeholder="Are the bottles running low? Any spills or hazards?",
        key="gemini_bar_prompt",
    )

    mode = st.radio("Input source", ["Upload Image", "Webcam"], horizontal=True)
    image: Image.Image | None = None

    if mode == "Upload Image":
        f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if f:
            image = Image.open(f).convert("RGB")
    else:
        cam = st.camera_input("Take a snapshot of your bar shelves or counter")
        if cam:
            image = Image.open(cam).convert("RGB")

    if image is None:
        return

    st.image(image, caption="Input image", use_container_width=True)

    if not st.button("🚀 Run Smart Bar Analysis", type="primary"):
        return

    # ------------------------------------------------------------------
    # Run workflow
    # ------------------------------------------------------------------
    with st.spinner("Processing via Roboflow Serverless Workflow…"):
        try:
            result: WorkflowResult = run_bar_pub_workflow(
                image=image,
                roboflow_api_key=ROBOFLOW_KEY,
                google_api_key=GOOGLE_KEY,
                gemini_prompt=user_prompt or None,  # None → client uses workflow default
            )
        except RoboflowAPIError as e:
            st.error(f"Roboflow API error {e.status_code}: {e.body[:300]}")
            return
        except NetworkError as e:
            st.error(f"Network error: {e}")
            return
        except WorkflowError as e:
            st.error(f"Workflow error: {e}")
            return

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    st.success("✅ Analysis complete!")

    # Counts from bounding-box predictions
    predictions = result.predictions
    counts = Counter(p.get("class", "object") for p in predictions)

    drink_classes = {"bottle", "wine glass", "cup", "bowl", "vase"}
    bottle_count = sum(cnt for cls, cnt in counts.items() if cls.lower() in drink_classes)

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="🍾 Drinks / Vessels Detected", value=int(bottle_count))
        st.metric(label="🔍 Total Objects", value=len(predictions))
    with col2:
        st.info(f"**Gemini Analysis:** {result.gemini_analysis_output or 'No response returned.'}")

    # Annotated image (decoded to disk by the client)
    if result.output_image_path and result.output_image_path.exists():
        try:
            annotated_img = Image.open(result.output_image_path)
            st.image(annotated_img, caption="📷 Annotated Detection View", use_container_width=True)
        finally:
            # Remove temp file — never hold base64 blobs longer than needed
            try:
                result.output_image_path.unlink(missing_ok=True)
            except Exception:
                pass

    # Detection summary breakdown
    if counts:
        st.subheader("Detection Summary")
        cols = st.columns(min(len(counts), 4))
        for i, (cls, cnt) in enumerate(counts.items()):
            cols[i % len(cols)].metric(cls.title(), cnt)

    # ------------------------------------------------------------------
    # Log to Supabase
    # ------------------------------------------------------------------
    try:
        log_entry = {
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "bottle_count": int(bottle_count),
            "gemini_analysis": result.gemini_analysis_output or "",
            "image_url": None,
        }
        supabase.table("pub_inventory_logs").insert(log_entry).execute()
        st.toast("📊 Record logged to pub_inventory_logs!", icon="💾")
    except Exception as db_err:
        st.sidebar.warning(f"⚠️ Supabase log failed: {db_err}")