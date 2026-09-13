import io
import json
import datetime
import base64
import requests
import numpy as np
from PIL import Image
import streamlit as st
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from services.supabase_client import supabase  # Global client

# Try importing DeepFace for local facial emotion detection
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    HAS_DEEPFACE = False


# --------------------------------------------------
# Pydantic Schemas for Gemini Structured Outputs
# --------------------------------------------------
class LabelData(BaseModel):
    product_name: Optional[str] = Field(description="Name or title of the product")
    brand: Optional[str] = Field(description="Brand or manufacturer name")
    category: Optional[str] = Field(description="Category e.g., Beverage, Food, Cosmetics, Medicine")
    key_ingredients_or_details: List[str] = Field(description="List of ingredients, key specifications, or materials")
    warnings_or_notes: List[str] = Field(description="Safety warnings, directions for use, or expiration notes")
    raw_text: str = Field(description="Full extracted raw text from the label")


class InvoiceItem(BaseModel):
    description: str = Field(description="Item or service description")
    quantity: float = Field(description="Quantity purchased")
    unit_price: float = Field(description="Price per unit")
    total_price: float = Field(description="Total price for this line item")


class InvoiceData(BaseModel):
    vendor_name: str = Field(description="Name of the business/vendor issuing the invoice")
    invoice_number: Optional[str] = Field(description="Invoice or receipt reference number")
    date: Optional[str] = Field(description="Date of transaction/invoice")
    line_items: List[InvoiceItem] = Field(description="Detailed list of items/services purchased")
    subtotal: Optional[float] = Field(description="Subtotal before taxes")
    tax_amount: Optional[float] = Field(description="Calculated tax amount")
    total_amount: float = Field(description="Final total balance due or paid")


# --------------------------------------------------
# Main UI Render Function
# --------------------------------------------------
# -------------------------------------------------------
# AudioLab text-to-speech
# -------------------------------------------------------
def render_audio_reader(text: str, key: str, label: str = "Read results aloud"):
    """Render a secure, on-demand AudioLab reader for a completed result."""
    audio_state_key = f"audiolab_audio_{key}"

    if st.button(label, key=f"audiolab_button_{key}"):
        api_key = st.secrets.get("AUDIOLAB_API_KEY")

        if not api_key:
            st.error(
                "Audio reading is not configured. "
                "Add AUDIOLAB_API_KEY to Streamlit secrets."
            )
        else:
            api_key = api_key.strip()

            st.info(
                f"AudioLab key loaded: "
                f"{api_key[:5]}...{api_key[-4:]}"
            )

            try:
                with st.spinner("Creating audio..."):
                    response = requests.post(
                        "https://api.tryaudiolab.ai/v1/audio/speech",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={
                            "model": "tts/auto",
                            "voice": "auto",
                            "input": text,
                            "response_format": "mp3",
                        },
                        timeout=60,
                    )
                    response.raise_for_status()
                    st.session_state[audio_state_key] = response.content
            except requests.RequestException as exc:
                st.error(f"Could not create audio: {exc}")

    if audio_state_key in st.session_state:
        st.audio(st.session_state[audio_state_key], format="audio/mpeg")


def render():
    tab1, tab2, tab3, tab4 = st.tabs([
        "🍻 Bar & Pub AI",
        "🎭 Emotion Recognition", 
        "🏷️ Label Reader", 
        "🧾 Invoice Reader"
    ])

    # Initialize Gemini Client safely
    try:
        google_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    except Exception:
        google_client = None

    # ==================================================
    # TAB 1: BAR & PUB AI (Original Module)
    # ==================================================
    with tab1:
        st.subheader("✨ Gemini Flash Bar & Pub AI")
        st.write("Combine high-speed object tracking with contextual conversational vision reasoning.")
        st.divider()

        user_question = st.text_input(
            "Ask Gemini about what the camera can see:", 
            value="You are a bar inventory assistant. Count all visible bottles, glasses, and drinks. Identify any spills or hazards. Give a concise operational summary.",
            key="gemini_bar_prompt"
        )

        img_file = st.camera_input("Take a snapshot of your bar shelves or counter", key="bar_cam")

        if img_file:
            image_bytes = img_file.getvalue()
            pil_image = Image.open(io.BytesIO(image_bytes))

            if st.button("🚀 Run Smart Bar Analysis", type="primary", key="btn_bar"):
                with st.spinner("Analyzing bar image..."):
                    try:
                        # 1. ROBOFLOW DIRECT MODEL INFERENCE
                        rf_api_key = st.secrets.get("ROBOFLOW_API_KEY", "")
                        project_id = "your-project-id"  # Update with your Roboflow project ID
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

                        # 2. GOOGLE GEMINI SDK
                        if not google_client:
                            st.error("Gemini API Client is not configured properly.")
                        else:
                            context_prompt = (
                                f"{user_question}\n\n"
                                f"[Context: Object detection model identified {bottle_count} items in frame]."
                            )

                            gemini_response = google_client.models.generate_content(
                                model='gemini-3.6-flash',
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

                            render_audio_reader(
                                f"{bottle_count} items were identified. {gemini_report}",
                                key="bar_result",
                            )

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

    # ==================================================
    # TAB 2: FACIAL EMOTION RECOGNITION
    # ==================================================
    with tab2:
        st.subheader("🎭 Facial Emotion Detection")
        st.write("Detect faces and analyze emotional expressions in real time.")

        face_file = (
            st.camera_input("Capture facial image", key="face_cam")
            or st.file_uploader(
                "Or upload image containing faces",
                type=["jpg", "jpeg", "png"],
                key="face_upload"
            )
        )

        if face_file:
            face_img = Image.open(face_file)
            st.image(face_img, caption="Input Image", width="stretch")

            if st.button("Analyze Emotion", type="primary", key="btn_emotion"):
                with st.spinner("Analyzing facial expressions..."):

                    if HAS_DEEPFACE:
                        try:
                            # Run local facial expression analysis
                            img_np = np.array(face_img)

                            results = DeepFace.analyze(
                                img_np,
                                actions=["emotion"],
                                enforce_detection=False
                            )

                            dominant = results[0]["dominant_emotion"]
                            emotions = results[0]["emotion"]

                            # Save result so it survives Streamlit reruns
                            st.session_state["emotion_result_text"] = (
                                f"The dominant emotion detected is {dominant}. "
                                f"Confidence scores: {json.dumps(emotions)}"
                            )

                            st.session_state["emotion_dominant"] = dominant
                            st.session_state["emotion_scores"] = emotions

                        except Exception as e:
                            st.error(f"DeepFace processing error: {e}")

                    else:
                        st.warning(
                            "Local `deepface` library not available. "
                            "Falling back to Gemini Multimodal analysis."
                        )

                        if google_client:
                            try:
                                response = google_client.models.generate_content(
                                    model="gemini-3.6-flash",
                                    contents=[
                                        "Identify the main faces in this photo. "
                                        "Describe their facial expressions, emotional "
                                        "state (e.g. Happy, Surprised, Neutral, Sad, Angry), "
                                        "and confidence level.",
                                        face_img
                                    ]
                                )

                                # Save Gemini result so it survives reruns
                                st.session_state["emotion_result_text"] = response.text
                                st.session_state["emotion_dominant"] = None
                                st.session_state["emotion_scores"] = None

                            except Exception as e:
                                st.error(f"Gemini emotion analysis failed: {e}")

                        else:
                            st.error("Gemini API Key is required for fallback.")

            # --------------------------------------------------
            # DISPLAY SAVED EMOTION RESULT
            # --------------------------------------------------
            if "emotion_result_text" in st.session_state:

                dominant = st.session_state.get("emotion_dominant")
                emotions = st.session_state.get("emotion_scores")
                emotion_text = st.session_state["emotion_result_text"]

                if dominant:
                    st.success(
                        f"**Dominant Emotion:** {dominant.capitalize()}"
                    )

                if emotions:
                    st.subheader("Emotion Confidence Distribution")
                    st.bar_chart(emotions)

                st.markdown(emotion_text)

                # AudioLab reader is now OUTSIDE the Analyze button
                render_audio_reader(
                    emotion_text,
                    key="emotion_result"
                )

    # ==================================================
    # TAB 3: LABEL READING
    # ==================================================
    with tab3:
        st.subheader("🏷️ Packaging & Product Label OCR")
        st.write("Extract structured product details, ingredients, warnings, and textual info.")

        label_file = st.file_uploader("Upload product/package label", type=["jpg", "jpeg", "png"], key="label_upload")

        if label_file:
            label_img = Image.open(label_file)
            st.image(label_img, caption="Uploaded Label", width="stretch")

            if st.button("Extract Label Information", type="primary", key="btn_label"):
                if not google_client:
                    st.error("Gemini API Client is missing.")
                else:
                    with st.spinner("Processing label details..."):
                        try:
                            response = google_client.models.generate_content(
                                model="gemini-3.6-flash",
                                contents=["Extract all product information, brand names, lists, and warnings from this label image.", label_img],
                                config={
                                    "response_mime_type": "application/json",
                                    "response_schema": LabelData,
                                }
                            )
                            data = json.loads(response.text)

                            st.success("Label Extracted Successfully!")
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Product Name", data.get("product_name") or "N/A")
                            col2.metric("Brand", data.get("brand") or "N/A")
                            col3.metric("Category", data.get("category") or "N/A")

                            st.divider()

                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("##### 📝 Key Ingredients / Specifications")
                                for item in data.get("key_ingredients_or_details", []):
                                    st.write(f"• {item}")

                            with c2:
                                st.markdown("##### ⚠️ Warnings & Usage Instructions")
                                for warn in data.get("warnings_or_notes", []):
                                    st.write(f"• {warn}")

                            with st.expander("📄 View Full Detected Raw Text"):
                                st.code(data.get("raw_text", ""))

                        except Exception as e:
                            st.error(f"Failed to extract label: {e}")

    # ==================================================
    # TAB 4: INVOICE & RECEIPT PARSING
    # ==================================================
    with tab4:
        st.subheader("🧾 Invoice & Receipt Parser")
        st.write("Parse line items, subtotal, taxes, and vendor details into structured table data.")

        invoice_file = st.file_uploader("Upload an invoice or receipt image", type=["jpg", "jpeg", "png"], key="invoice_upload")

        if invoice_file:
            inv_img = Image.open(invoice_file)
            st.image(inv_img, caption="Uploaded Invoice", width="stretch")

            if st.button("Parse Financial Data", type="primary", key="btn_invoice"):
                if not google_client:
                    st.error("Gemini API Client is missing.")
                else:
                    with st.spinner("Parsing structured invoice details..."):
                        try:
                            response = google_client.models.generate_content(
                                model="gemini-3.6-flash",
                                contents=["Extract line items, prices, tax, vendor name, invoice date, and total financial amounts from this image.", inv_img],
                                config={
                                    "response_mime_type": "application/json",
                                    "response_schema": InvoiceData,
                                }
                            )
                            invoice = json.loads(response.text)

                            st.success("Invoice Parsed Successfully!")

                            # Metrics Summary
                            c1, c2, c3 = st.columns(3)
                            c1.metric("Vendor", invoice.get("vendor_name", "N/A"))
                            c2.metric("Invoice #", invoice.get("invoice_number", "N/A"))
                            c3.metric("Date", invoice.get("date", "N/A"))

                            st.divider()

                            # Line Items Dataframe
                            st.markdown("##### 📋 Line Items")
                            if invoice.get("line_items"):
                                st.dataframe(invoice["line_items"], width="stretch")

                            st.divider()

                            # Subtotal / Tax / Total Breakdown
                            t1, t2, t3 = st.columns(3)
                            t1.metric("Subtotal", f"${invoice.get('subtotal', 0.0):,.2f}" if invoice.get('subtotal') is not None else "N/A")
                            t2.metric("Tax Amount", f"${invoice.get('tax_amount', 0.0):,.2f}" if invoice.get('tax_amount') is not None else "N/A")
                            t3.metric("Total Amount", f"${invoice.get('total_amount', 0.0):,.2f}")

                        except Exception as e:
                            st.error(f"Failed to parse invoice: {e}")
