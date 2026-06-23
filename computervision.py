import streamlit as st
from modules import (
    roboflow_detect,
    motion,
    face_detect,
    gemini3_flash_app
)

TOOLS = {
    "🔍 Roboflow Detect": roboflow_detect,
    "✨ Gemini 3 Flash": gemini3_flash_app,
    "🏃 Motion Detection": motion,
    "🙂 Face Detection": face_detect,
}

def render():
    st.sidebar.header("🧪 CV Lab")
    st.sidebar.markdown("---")

    tool_name = st.sidebar.radio(
        "Choose a module",
        list(TOOLS.keys()),
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with MediaPipe · OpenCV · Roboflow · Google Gemini")

    TOOLS[tool_name].render()