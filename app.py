import streamlit as st
import computervision

from services.supabase_client import supabase

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")
st.caption("Experiment with pretrained computer vision models.")
st.divider()

# --------------------------------------------------
# Supabase Status
# --------------------------------------------------
try:
    supabase.table("profiles").select("id").limit(1).execute()
    st.sidebar.success("🟢 Supabase Connected")

except Exception:
    st.sidebar.error("🔴 Supabase Offline")

# --------------------------------------------------
# Launch Application
# --------------------------------------------------
try:
    if hasattr(computervision, "render"):
        computervision.render()
    else:
        st.error("The computervision module is missing the render() function.")

except Exception as e:
    st.exception(e)