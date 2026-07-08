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
# Debug Supabase Configuration
# --------------------------------------------------
st.subheader("🔍 Supabase Debug")

try:
    st.write("**Project URL:**", st.secrets["SUPABASE_URL"])

    key = st.secrets["SUPABASE_KEY"]
    st.write("**API Key Prefix:**", key[:20] + "...")

except Exception as e:
    st.error("❌ Could not read Streamlit secrets.")
    st.exception(e)

st.divider()

# --------------------------------------------------
# Test Supabase Connection
# --------------------------------------------------
st.subheader("🗄️ Supabase Connection Test")

try:
    result = (
        supabase
        .table("profiles")
        .select("*")
        .limit(1)
        .execute()
    )

    st.success("✅ Connected to Supabase")
    st.write("Profiles table response:")
    st.json(result.data)

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