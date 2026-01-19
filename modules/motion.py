import streamlit as st
import cv2
import numpy as np
from PIL import Image

def render():
    st.header("🏃 Motion Detection")

    file = st.file_uploader("Upload image", type=["jpg","png","jpeg"])
    if not file:
        return

    image = np.array(Image.open(file).convert("RGB"))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (21,21), 0)

    if "prev" not in st.session_state:
        st.session_state.prev = blur
        st.info("Upload another image to detect motion")
        return

    diff = cv2.absdiff(st.session_state.prev, blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    st.image(thresh, caption="Motion Mask", use_container_width=True)
    st.session_state.prev = blur
