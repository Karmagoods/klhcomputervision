import streamlit as st
from dotenv import load_dotenv
import computervision

load_dotenv()

st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")
st.caption("Experiment with pretrained computer vision models.")

st.divider()

computervision.render()
