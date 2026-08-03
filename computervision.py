import requests
import base64
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
                    google_api_key = st.secrets.get("GOOGLE_API_KEY", "")

                    # 🛠️ Roboflow Workspace & Workflow details
                    workspace_id = "klhinnovation"
                    workflow_id = "playground-gemini-3-flash-object-detection"

                    # Correctly formatted Serverless Workflow Endpoint
                    workflow_url = f"https://detect.roboflow.com/infer/workflows/{workspace_id}/{workflow_id}"

                    # Encode image as base64 for JSON payload (required by Roboflow Workflows)
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                    # Build JSON payload — Roboflow Workflows use JSON, not multipart
                    payload = {
                        "api_key": rf_api_key,
                        "inputs": {
                            "image": {
                                "type": "base64",
                                "value": image_b64
                            },
                            "gemini_prompt": user_question or "Count all visible bottles and drinks. Note any hazards or spills.",
                            "google_api_key": google_api_key
                        }
                    }

                    headers = {"Content-Type": "application/json"}

                    # Dispatch to Roboflow workflow broker engine
                    response = requests.post(workflow_url, json=payload, headers=headers)
                    response.raise_for_status()
                    result = response.json()

                    # Roboflow Workflows return results under outputs[0]
                    outputs = result.get("outputs", [{}])[0]

                    # Extract variables safely matching workflow output names
                    predictions = outputs.get("predictions", {})
                    gemini_report = outputs.get("gemini_analysis_output", "No conversational response returned.")
                    output_image = outputs.get("output_image")

                    # Count detected objects (bottles, cups, etc.) from predictions
                    detections = predictions.get("predictions", []) if isinstance(predictions, dict) else []
                    bottle_count = len([d for d in detections if d.get("class", "").lower() in [
                        "bottle", "wine glass", "cup", "bowl", "vase"
                    ]])

                    # 3. Present UI layout outputs
                    st.success("Analysis Complete!")

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.metric(label="🍾 Bottles / Drinks Identified", value=int(bottle_count))
                    with col2:
                        st.info(f"**Gemini Analysis Summary:** {gemini_report}")

                    # Show annotated image if available
                    if output_image:
                        annotated_bytes = base64.b64decode(output_image.get("value", ""))
                        if annotated_bytes:
                            st.image(annotated_bytes, caption="📷 Annotated Detection View", use_container_width=True)

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