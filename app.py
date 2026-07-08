import streamlit as st
import computervision

from services.supabase_client import supabase

st.set_page_config(
    page_title="Computer Vision Lab",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Computer Vision Lab")
st.caption("Experiment with pretrained computer vision models.")
st.divider()

# --------------------------------------------------
# Test Supabase Connection
# --------------------------------------------------
try:
    result = (
        supabase
        .table("profiles")
        .select("id")
        .limit(1)
        .execute()
    )

    st.success("✅ Connected to Supabase")

except Exception as e:
    st.error("❌ Failed to connect to Supabase")
    st.exception(e)

st.divider()

# --------------------------------------------------
# Launch Computer Vision App
# --------------------------------------------------
try:
    if hasattr(computervision, "render"):
        computervision.render()
    else:
        st.error("computervision module does not have a render() function.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")