import streamlit as st
from modules import (
    roboflow_detect,
    motion,
    face_detect,
    gemini3_flash_app
)

def render():
    st.sidebar.header("Select Tool")

    tool = st.sidebar.radio(
        "Choose a module",
        [
            "Roboflow Detect",
            "Gemini 3 Flash (Google)",
            "Motion Detection",
            "Face Detection"
        ]
    )

    st.divider()

    if tool == "Roboflow Detect":
        roboflow_detect.render()
    elif tool == "Gemini 3 Flash (Google)":
        gemini3_flash_app.render()
    elif tool == "Motion Detection":
        motion.render()
    elif tool == "Face Detection":
        face_detect.render()
