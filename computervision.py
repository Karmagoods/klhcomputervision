import requests
import datetime
import streamlit as st
from services.supabase_client import supabase  # Reuses your global validated client

def render():
    st.subheader("✨ Gemini 3 Flash Bar & Pub AI")
    st.write("Combine high-speed object tracking with contextual conversational vision reasoning.")
    st.divider()

    # 1. Accept dynamic context questions for Gemini
    user_question = st.text_input(
        "Ask Gemini about what the camera can see:", 
        placeholder="Are the bottles running low? Any drink spills or hazards on the counter?",
        key="gemini_bar_prompt"
    )

    # 2. Capture mobile smartphone camera input snapshot
    img_file = st.camera_input("Take a snapshot of your bar shelves or counter")

    if img_file:
        image_bytes = img_file.getvalue()
        
        # Trigger explicit deployment action button to prevent multiple API fires on change
        if st.button("🚀 Run Smart Bar Analysis", type="primary"):
            with st.spinner("Processing image via Roboflow Serverless Workflows..."):
                try:
                    # Pull verified global API keys from st.secrets
                    rf_api_key = st.secrets["ROBOFLOW_API_KEY"]
                    
                    # 🔧 REPLACE THESE WITH YOUR ROBOFLOW WORKSPACE & WORKFLOW IDS
                    workspace_id = "your-workspace-id"
                    workflow_id = "your-workflow-id"
                    
                    # Correctly formatted Serverless Workflow Endpoint
                    workflow_url = f"https://detect.roboflow.com/infer/workflows/{workspace_id}/{workflow_id}?api_key={rf_api_key}"
                    
                    # Package the camera frames and inputs
                    files = {"image": ("image.jpg", image_bytes, "image/jpeg")}
                    data = {"gemini_prompt": user_question}
                    
                    # Dispatch to Roboflow workflow broker engine
                    response = requests.post(workflow_url, files=files, data=data)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Extract variables safely matching your defined output structure
                    bottle_count = result.get("bottle_count_output", 0)
                    gemini_report = result.get("gemini_analysis_output", "No conversational response returned.")
                    
                    # 3. Present UI layout outputs
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(label="🍾 Bottles Identified", value=int(bottle_count))
                    with col2:
                        st.info(f"**Gemini Analysis Summary:** {gemini_report}")
                    
                    # 4. Insert directly into your verified Supabase Schema columns
                    with st.spinner("Logging transaction data into cloud server tables..."):
                        log_entry = {
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "bottle_count": int(bottle_count),
                            "gemini_analysis": gemini_report,
                            "image_url": None  # Explicitly passed as None as verified via schema checks
                        }
                        
                        supabase.table("pub_inventory_logs").insert(log_entry).execute()
                        st.toast("📊 Record successfully added to pub_inventory_logs table!", icon="💾")
                        
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Network Pipeline Error: Could not resolve serverless route. Verify your Workflow layout ID is published inside your Roboflow dashboard. ({http_err})")
                except Exception as e:
                    st.error(f"Processing error encountered: {e}")