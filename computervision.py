import io
import datetime
import base64
import requests
import streamlit as st
from PIL import Image
from google import genai
from services.supabase_client import supabase  # Global client

def render():
    st.subheader("✨ Gemini 3 Flash Bar & Pub AI")
    st.write("Combine high-speed object tracking with contextual conversational vision reasoning.")
    st.divider()

    user_question = st.text_input(
        "Ask Gemini about what the camera can see:", 
        value="You are a bar inventory assistant. Count all visible bottles, glasses, and drinks. Identify any spills or hazards. Give a concise operational summary.",
        key="gemini_bar_prompt"
    )

    img_file = st.camera_input("Take a snapshot of your bar shelves or counter")

    if img_file:
        image_bytes = img_file.getvalue()
        pil_image = Image.open(io.BytesIO(image_bytes))

        if st.button("🚀 Run Smart Bar Analysis", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # 1. ROBOFLOW DIRECT MODEL INFERENCE
                    rf_api_key = st.secrets["ROBOFLOW_API_KEY"]
                    
                    # Update with your actual Roboflow project ID
                    project_id = "your-project-id"  
                    model_version = "1"
                    
                    rf_url = f"https://detect.roboflow.com/{project_id}/{model_version}?api_key={rf_api_key}"
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    
                    rf_response = requests.post(
                        rf_url, 
                        data=base64_image, 
                        headers={"Content-Type": "application/x-www-form-urlencoded"}
                    )
                    
                    if rf_response.status_code == 200:
                        rf_result = rf_response.json()
                        preds_list = rf_result.get("predictions", [])
                        bottle_count = len(preds_list)
                    else:
                        bottle_count = 0

                    # 2. GOOGLE GEMINI SDK (Using active model ID)
                    google_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    
                    context_prompt = (
                        f"{user_question}\n\n"
                        f"[Context: Object detection model identified {bottle_count} items in frame]."
                    )

                    # Updated model parameter
                    gemini_response = google_client.models.generate_content(
                        model='gemini-2.5-flash',
                        contents=[pil_image, context_prompt]
                    )
                    gemini_report = gemini_response.text

                    # 3. DISPLAY RESULTS & SUPABASE LOG
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(label="🍾 Items Identified", value=int(bottle_count))
                    with col2:
                        st.info(f"**Gemini Analysis Summary:**\n\n{gemini_report}")

                    log_entry = {
                        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "bottle_count": int(bottle_count),
                        "gemini_analysis": gemini_report,
                        "image_url": None
                    }
                    supabase.table("pub_inventory_logs").insert(log_entry).execute()
                    st.toast("📊 Record successfully added to database!", icon="💾")

                except Exception as e:
                    st.error(f"Processing error encountered: {e}")