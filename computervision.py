import streamlit as st
from modules import roboflow_detect, motion, face_detect

def render():
    st.sidebar.header("Experiments")

    mode = st.sidebar.selectbox(
        "Select experiment",
        [
            "Roboflow Object Detection",
            "Motion Detection",
            "Face Detection"
        ]
    )

    st.divider()

    if mode == "Roboflow Object Detection":
        roboflow_detect.render()

    elif mode == "Motion Detection":
        motion.render()

    elif mode == "Face Detection":
        face_detect.render()
