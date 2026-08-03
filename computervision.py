import requests
import datetime
import base64
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
        value="You are a bar inventory assistant. Count all visible bottles, glasses, and drinks. Identify any spills or hazards. Give a concise operational summary.",
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
                    
                    # 🛠️ Roboflow Workspace & Workflow details
                    workspace_id = "klhinnovation"
                    workflow_id = "gemini-flash-bar-and-pub-object-detection-with-count-1785756763153"
                    
                    # Correctly formatted Serverless Workflow Endpoint
                    workflow_url = f"https://detect.roboflow.com/infer/workflows/{workspace_id}/{workflow_id}?api_key={rf_api_key}"
                    
                    # Encode image bytes to base64 string
                    base64_image = base64.b64encode(image_bytes).decode("utf-8")
                    
                    # Prepare workflow payload input parameters
                    payload = {
                        "inputs": {
                            "image": {
                                "type": "base64",
                                "value": base64_image
                            },
                            "gemini_prompt": user_question
                        }
                    }
                    
                    # Dispatch to Roboflow workflow engine via JSON post
                    response = requests.post(workflow_url, json=payload)
                    response.raise_for_status()
                    result = response.json()
                    
                    # Handle response output array/dictionary
                    output_data = result[0] if isinstance(result, list) and len(result) > 0 else result
                    
                    # Extract variables matching your defined output structure
                    predictions_data = output_data.get("predictions", {})
                    preds_list = predictions_data.get("predictions", []) if isinstance(predictions_data, dict) else []
                    bottle_count = len(preds_list)
                    
                    gemini_report = output_data.get("gemini_analysis_output", "No conversational response returned.")
                    
                    # 3. Present UI layout outputs
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(label="🍾 Items Identified", value=int(bottle_count))
                    with col2:
                        st.info(f"**Gemini Analysis Summary:**\n\n{gemini_report}")
                    
                    # 4. Insert directly into your verified Supabase Schema columns
                    with st.spinner("Logging transaction data into cloud server tables..."):
                        log_entry = {
                            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            "bottle_count": int(bottle_count),
                            "gemini_analysis": gemini_report,
                            "image_url": None
                        }
                        
                        supabase.table("pub_inventory_logs").insert(log_entry).execute()
                        st.toast("📊 Record successfully added to pub_inventory_logs table!", icon="💾")
                        
                except requests.exceptions.HTTPError as http_err:
                    st.error(f"Roboflow API error {http_err.response.status_code}: {http_err.response.text}")
                except Exception as e:
                    st.error(f"Processing error encountered: {e}")