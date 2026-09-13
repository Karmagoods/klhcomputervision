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

from services.supabase_client import supabase


# ============================================================
# TRY IMPORTING DEEPFACE
# ============================================================

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    HAS_DEEPFACE = False


# ============================================================
# PYDANTIC SCHEMAS FOR GEMINI STRUCTURED OUTPUTS
# ============================================================

class LabelData(BaseModel):
    product_name: Optional[str] = Field(
        description="Name or title of the product"
    )
    brand: Optional[str] = Field(
        description="Brand or manufacturer name"
    )
    category: Optional[str] = Field(
        description="Category e.g., Beverage, Food, Cosmetics, Medicine"
    )
    key_ingredients_or_details: List[str] = Field(
        description="List of ingredients, key specifications, or materials"
    )
    warnings_or_notes: List[str] = Field(
        description="Safety warnings, directions for use, or expiration notes"
    )
    raw_text: str = Field(
        description="Full extracted raw text from the label"
    )


class InvoiceItem(BaseModel):
    description: str = Field(
        description="Item or service description"
    )
    quantity: float = Field(
        description="Quantity purchased"
    )
    unit_price: float = Field(
        description="Price per unit"
    )
    total_price: float = Field(
        description="Total price for this line item"
    )


class InvoiceData(BaseModel):
    vendor_name: str = Field(
        description="Name of the business/vendor issuing the invoice"
    )
    invoice_number: Optional[str] = Field(
        description="Invoice or receipt reference number"
    )
    date: Optional[str] = Field(
        description="Date of transaction/invoice"
    )
    line_items: List[InvoiceItem] = Field(
        description="Detailed list of items/services purchased"
    )
    subtotal: Optional[float] = Field(
        description="Subtotal before taxes"
    )
    tax_amount: Optional[float] = Field(
        description="Calculated tax amount"
    )
    total_amount: float = Field(
        description="Final total balance due or paid"
    )


# ============================================================
# AUDIOLAB TEXT-TO-SPEECH
# ============================================================

def render_audio_reader(
    text: str,
    key: str,
    label: str = "Read results aloud"
):
    """
    Render an on-demand AudioLab TTS reader.

    The generated audio is stored in Streamlit session state so
    it survives Streamlit reruns.
    """

    audio_state_key = f"audiolab_audio_{key}"
    button_key = f"audiolab_button_{key}"

    # --------------------------------------------------------
    # Generate audio
    # --------------------------------------------------------

    if st.button(label, key=button_key):

        api_key = st.secrets.get("AUDIOLAB_API_KEY")

        if not api_key:
            st.error(
                "Audio reading is not configured. "
                "Add AUDIOLAB_API_KEY to Streamlit secrets."
            )
            return

        api_key = str(api_key).strip()

        # Safe debugging - NEVER display the full key
        st.info(
            f"AudioLab key loaded: "
            f"{api_key[:5]}...{api_key[-4:]}"
        )

        try:
            with st.spinner("Creating audio..."):

                response = requests.post(
                    "https://api.tryaudiolab.ai/v1/audio/speech",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Accept": "audio/mpeg",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "tts/auto",
                        "voice": "auto",
                        "input": text,
                        "response_format": "mp3",
                    },
                    timeout=60,
                )

            # ------------------------------------------------
            # IMPORTANT:
            # Do NOT use raise_for_status() here.
            #
            # We want to see AudioLab's actual error message.
            # ------------------------------------------------

            if response.status_code != 200:

                error_body = response.text.strip()

                if not error_body:
                    error_body = "(AudioLab returned an empty error response.)"

                st.error(
                    f"AudioLab error {response.status_code}: "
                    f"{error_body}"
                )

                return

            # ------------------------------------------------
            # Successful audio response
            # ------------------------------------------------

            if not response.content:
                st.error(
                    "AudioLab returned HTTP 200 but no audio data."
                )
                return

            st.session_state[audio_state_key] = response.content

            st.success("Audio created successfully.")

        except requests.Timeout:
            st.error(
                "AudioLab request timed out. "
                "Please try again."
            )

        except requests.RequestException as exc:
            st.error(
                f"Could not connect to AudioLab: {exc}"
            )

        except Exception as exc:
            st.error(
                f"Unexpected AudioLab error: {exc}"
            )

    # --------------------------------------------------------
    # PLAY SAVED AUDIO
    # --------------------------------------------------------

    if audio_state_key in st.session_state:

        st.audio(
            st.session_state[audio_state_key],
            format="audio/mpeg"
        )


# ============================================================
# MAIN UI RENDER FUNCTION
# ============================================================

def render():

    tab1, tab2, tab3, tab4 = st.tabs([
        "Bar & Pub AI",
        "Emotion Recognition",
        "Label Reader",
        "Invoice Reader",
    ])


    # ========================================================
    # INITIALIZE GEMINI CLIENT
    # ========================================================

    try:
        google_client = genai.Client(
            api_key=st.secrets["GEMINI_API_KEY"]
        )
    except Exception:
        google_client = None


    # ========================================================
    # TAB 1: BAR & PUB AI
    # ========================================================

    with tab1:

        st.subheader("Gemini Flash Bar & Pub AI")

        st.write(
            "Combine high-speed object tracking with "
            "contextual conversational vision reasoning."
        )

        st.divider()

        user_question = st.text_input(
            "Ask Gemini about what the camera can see:",
            value=(
                "You are a bar inventory assistant. "
                "Count all visible bottles, glasses, and drinks. "
                "Identify any spills or hazards. "
                "Give a concise operational summary."
            ),
            key="gemini_bar_prompt"
        )

        img_file = st.camera_input(
            "Take a snapshot of your bar shelves or counter",
            key="bar_cam"
        )

        if img_file:

            image_bytes = img_file.getvalue()

            pil_image = Image.open(
                io.BytesIO(image_bytes)
            )

            st.image(
                pil_image,
                caption="Bar Image",
                width="stretch"
            )

            # ------------------------------------------------
            # RUN ANALYSIS
            # ------------------------------------------------

            if st.button(
                "Run Smart Bar Analysis",
                type="primary",
                key="btn_bar"
            ):

                with st.spinner(
                    "Analyzing bar image..."
                ):

                    try:

                        # ====================================
                        # 1. ROBOFLOW
                        # ====================================

                        rf_api_key = st.secrets.get(
                            "ROBOFLOW_API_KEY",
                            ""
                        )

                        project_id = "your-project-id"
                        model_version = "1"

                        rf_url = (
                            f"https://detect.roboflow.com/"
                            f"{project_id}/"
                            f"{model_version}"
                            f"?api_key={rf_api_key}"
                        )

                        base64_image = base64.b64encode(
                            image_bytes
                        ).decode("utf-8")

                        rf_response = requests.post(
                            rf_url,
                            data=base64_image,
                            headers={
                                "Content-Type":
                                    "application/x-www-form-urlencoded"
                            },
                            timeout=60
                        )

                        if rf_response.status_code == 200:

                            rf_result = rf_response.json()

                            preds_list = rf_result.get(
                                "predictions",
                                []
                            )

                            bottle_count = len(
                                preds_list
                            )

                        else:

                            bottle_count = 0


                        # ====================================
                        # 2. GEMINI
                        # ====================================

                        if not google_client:

                            st.error(
                                "Gemini API Client is not "
                                "configured properly."
                            )

                        else:

                            context_prompt = (
                                f"{user_question}\n\n"
                                f"[Context: Object detection model "
                                f"identified {bottle_count} "
                                f"items in frame]."
                            )

                            gemini_response = (
                                google_client.models.generate_content(
                                    model="gemini-3.6-flash",
                                    contents=[
                                        pil_image,
                                        context_prompt
                                    ]
                                )
                            )

                            gemini_report = (
                                gemini_response.text
                            )


                            # =================================
                            # 3. SAVE RESULT
                            # =================================

                            st.session_state[
                                "bar_result_text"
                            ] = (
                                f"{bottle_count} items were "
                                f"identified. "
                                f"{gemini_report}"
                            )

                            st.session_state[
                                "bar_bottle_count"
                            ] = int(bottle_count)


                            # =================================
                            # 4. SUPABASE LOG
                            # =================================

                            log_entry = {
                                "created_at": (
                                    datetime.datetime.now(
                                        datetime.timezone.utc
                                    ).isoformat()
                                ),
                                "bottle_count": int(
                                    bottle_count
                                ),
                                "gemini_analysis": (
                                    gemini_report
                                ),
                                "image_url": None
                            }

                            try:

                                supabase.table(
                                    "pub_inventory_logs"
                                ).insert(
                                    log_entry
                                ).execute()

                                st.toast(
                                    "Record successfully "
                                    "added to database!"
                                )

                            except Exception as db_error:

                                st.warning(
                                    f"Supabase logging failed: "
                                    f"{db_error}"
                                )


                    except Exception as e:

                        st.error(
                            f"Processing error encountered: {e}"
                        )


            # ------------------------------------------------
            # DISPLAY SAVED BAR RESULT
            # ------------------------------------------------

            if "bar_result_text" in st.session_state:

                bottle_count = st.session_state.get(
                    "bar_bottle_count",
                    0
                )

                bar_result_text = st.session_state[
                    "bar_result_text"
                ]

                st.success("Analysis Complete!")

                col1, col2 = st.columns([1, 3])

                with col1:

                    st.metric(
                        label="Items Identified",
                        value=int(bottle_count)
                    )

                with col2:

                    st.info(
                        f"**Gemini Analysis Summary:**\n\n"
                        f"{bar_result_text}"
                    )

                # IMPORTANT:
                # Audio button is outside the analysis button
                render_audio_reader(
                    bar_result_text,
                    key="bar_result"
                )


    # ========================================================
    # TAB 2: FACIAL EMOTION RECOGNITION
    # ========================================================

    with tab2:

        st.subheader("Facial Emotion Detection")

        st.write(
            "Detect faces and analyze emotional "
            "expressions in real time."
        )

        face_file = (
            st.camera_input(
                "Capture facial image",
                key="face_cam"
            )
            or
            st.file_uploader(
                "Or upload image containing faces",
                type=[
                    "jpg",
                    "jpeg",
                    "png"
                ],
                key="face_upload"
            )
        )

        if face_file:

            face_img = Image.open(face_file)

            st.image(
                face_img,
                caption="Input Image",
                width="stretch"
            )


            # ------------------------------------------------
            # ANALYZE EMOTION
            # ------------------------------------------------

            if st.button(
                "Analyze Emotion",
                type="primary",
                key="btn_emotion"
            ):

                with st.spinner(
                    "Analyzing facial expressions..."
                ):

                    # ========================================
                    # DEEPFACE
                    # ========================================

                    if HAS_DEEPFACE:

                        try:

                            img_np = np.array(
                                face_img
                            )

                            results = DeepFace.analyze(
                                img_np,
                                actions=["emotion"],
                                enforce_detection=False
                            )

                            dominant = results[0][
                                "dominant_emotion"
                            ]

                            emotions = results[0][
                                "emotion"
                            ]


                            # Save result so it survives
                            # Streamlit reruns

                            st.session_state[
                                "emotion_result_text"
                            ] = (
                                f"The dominant emotion "
                                f"detected is {dominant}. "
                                f"Confidence scores: "
                                f"{json.dumps(emotions)}"
                            )

                            st.session_state[
                                "emotion_dominant"
                            ] = dominant

                            st.session_state[
                                "emotion_scores"
                            ] = emotions


                        except Exception as e:

                            st.error(
                                f"DeepFace processing error: {e}"
                            )


                    # ========================================
                    # GEMINI FALLBACK
                    # ========================================

                    else:

                        st.warning(
                            "Local `deepface` library is not "
                            "available. Falling back to "
                            "Gemini Multimodal analysis."
                        )

                        if google_client:

                            try:

                                response = (
                                    google_client.models
                                    .generate_content(
                                        model="gemini-3.6-flash",
                                        contents=[
                                            (
                                                "Identify the main "
                                                "faces in this photo. "
                                                "Describe their facial "
                                                "expressions, emotional "
                                                "state (e.g. Happy, "
                                                "Surprised, Neutral, "
                                                "Sad, Angry), and "
                                                "confidence level."
                                            ),
                                            face_img
                                        ]
                                    )
                                )

                                # Save Gemini result

                                st.session_state[
                                    "emotion_result_text"
                                ] = response.text

                                st.session_state[
                                    "emotion_dominant"
                                ] = None

                                st.session_state[
                                    "emotion_scores"
                                ] = None


                            except Exception as e:

                                st.error(
                                    "Gemini emotion analysis "
                                    f"failed: {e}"
                                )

                        else:

                            st.error(
                                "Gemini API Key is required "
                                "for fallback."
                            )


            # ------------------------------------------------
            # DISPLAY SAVED EMOTION RESULT
            # ------------------------------------------------

            if "emotion_result_text" in st.session_state:

                dominant = st.session_state.get(
                    "emotion_dominant"
                )

                emotions = st.session_state.get(
                    "emotion_scores"
                )

                emotion_text = st.session_state[
                    "emotion_result_text"
                ]


                if dominant:

                    st.success(
                        f"**Dominant Emotion:** "
                        f"{dominant.capitalize()}"
                    )


                if emotions:

                    st.subheader(
                        "Emotion Confidence Distribution"
                    )

                    st.bar_chart(emotions)


                st.markdown(emotion_text)


                # IMPORTANT:
                # AudioLab reader is OUTSIDE the
                # Analyze Emotion button.

                render_audio_reader(
                    emotion_text,
                    key="emotion_result"
                )


    # ========================================================
    # TAB 3: LABEL READER
    # ========================================================

    with tab3:

        st.subheader(
            "Packaging & Product Label OCR"
        )

        st.write(
            "Extract structured product details, "
            "ingredients, warnings, and textual info."
        )

        label_file = st.file_uploader(
            "Upload product/package label",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="label_upload"
        )

        if label_file:

            label_img = Image.open(label_file)

            st.image(
                label_img,
                caption="Uploaded Label",
                width="stretch"
            )


            if st.button(
                "Extract Label Information",
                type="primary",
                key="btn_label"
            ):

                if not google_client:

                    st.error(
                        "Gemini API Client is missing."
                    )

                else:

                    with st.spinner(
                        "Processing label details..."
                    ):

                        try:

                            response = (
                                google_client.models
                                .generate_content(
                                    model="gemini-3.6-flash",
                                    contents=[
                                        (
                                            "Extract all product "
                                            "information, brand names, "
                                            "lists, and warnings from "
                                            "this label image."
                                        ),
                                        label_img
                                    ],
                                    config={
                                        "response_mime_type":
                                            "application/json",
                                        "response_schema":
                                            LabelData,
                                    }
                                )
                            )

                            data = json.loads(
                                response.text
                            )


                            # Save result for reruns

                            st.session_state[
                                "label_result"
                            ] = data


                        except Exception as e:

                            st.error(
                                f"Failed to extract label: {e}"
                            )


            # ------------------------------------------------
            # DISPLAY SAVED LABEL RESULT
            # ------------------------------------------------

            if "label_result" in st.session_state:

                data = st.session_state[
                    "label_result"
                ]

                st.success(
                    "Label Extracted Successfully!"
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Product Name",
                    data.get(
                        "product_name"
                    ) or "N/A"
                )

                col2.metric(
                    "Brand",
                    data.get(
                        "brand"
                    ) or "N/A"
                )

                col3.metric(
                    "Category",
                    data.get(
                        "category"
                    ) or "N/A"
                )

                st.divider()

                c1, c2 = st.columns(2)

                with c1:

                    st.markdown(
                        "##### Key Ingredients / Specifications"
                    )

                    for item in data.get(
                        "key_ingredients_or_details",
                        []
                    ):

                        st.write(
                            f"• {item}"
                        )

                with c2:

                    st.markdown(
                        "##### Warnings & Usage Instructions"
                    )

                    for warn in data.get(
                        "warnings_or_notes",
                        []
                    ):

                        st.write(
                            f"• {warn}"
                        )


                with st.expander(
                    "View Full Detected Raw Text"
                ):

                    st.code(
                        data.get(
                            "raw_text",
                            ""
                        )
                    )


                # --------------------------------------------
                # AUDIO VERSION
                # --------------------------------------------

                label_audio_text = (
                    f"Product name: "
                    f"{data.get('product_name') or 'Not identified'}. "
                    f"Brand: "
                    f"{data.get('brand') or 'Not identified'}. "
                    f"Category: "
                    f"{data.get('category') or 'Not identified'}. "
                )

                details = data.get(
                    "key_ingredients_or_details",
                    []
                )

                if details:

                    label_audio_text += (
                        "Key details: "
                        + ", ".join(details)
                        + ". "
                    )

                warnings = data.get(
                    "warnings_or_notes",
                    []
                )

                if warnings:

                    label_audio_text += (
                        "Warnings and notes: "
                        + ", ".join(warnings)
                        + "."
                    )

                render_audio_reader(
                    label_audio_text,
                    key="label_result"
                )


    # ========================================================
    # TAB 4: INVOICE & RECEIPT PARSING
    # ========================================================

    with tab4:

        st.subheader(
            "Invoice & Receipt Parser"
        )

        st.write(
            "Parse line items, subtotal, taxes, "
            "and vendor details into structured table data."
        )

        invoice_file = st.file_uploader(
            "Upload an invoice or receipt image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="invoice_upload"
        )

        if invoice_file:

            inv_img = Image.open(
                invoice_file
            )

            st.image(
                inv_img,
                caption="Uploaded Invoice",
                width="stretch"
            )


            if st.button(
                "Parse Financial Data",
                type="primary",
                key="btn_invoice"
            ):

                if not google_client:

                    st.error(
                        "Gemini API Client is missing."
                    )

                else:

                    with st.spinner(
                        "Parsing structured invoice details..."
                    ):

                        try:

                            response = (
                                google_client.models
                                .generate_content(
                                    model="gemini-3.6-flash",
                                    contents=[
                                        (
                                            "Extract line items, "
                                            "prices, tax, vendor name, "
                                            "invoice date, and total "
                                            "financial amounts from "
                                            "this image."
                                        ),
                                        inv_img
                                    ],
                                    config={
                                        "response_mime_type":
                                            "application/json",
                                        "response_schema":
                                            InvoiceData,
                                    }
                                )
                            )

                            invoice = json.loads(
                                response.text
                            )


                            # Save for reruns

                            st.session_state[
                                "invoice_result"
                            ] = invoice


                        except Exception as e:

                            st.error(
                                f"Failed to parse invoice: {e}"
                            )


            # ------------------------------------------------
            # DISPLAY SAVED INVOICE RESULT
            # ------------------------------------------------

            if "invoice_result" in st.session_state:

                invoice = st.session_state[
                    "invoice_result"
                ]

                st.success(
                    "Invoice Parsed Successfully!"
                )


                # ============================================
                # SUMMARY
                # ============================================

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Vendor",
                    invoice.get(
                        "vendor_name",
                        "N/A"
                    )
                )

                c2.metric(
                    "Invoice #",
                    invoice.get(
                        "invoice_number",
                        "N/A"
                    )
                )

                c3.metric(
                    "Date",
                    invoice.get(
                        "date",
                        "N/A"
                    )
                )


                st.divider()


                # ============================================
                # LINE ITEMS
                # ============================================

                st.markdown(
                    "##### Line Items"
                )

                if invoice.get(
                    "line_items"
                ):

                    st.dataframe(
                        invoice["line_items"],
                        width="stretch"
                    )


                st.divider()


                # ============================================
                # FINANCIAL BREAKDOWN
                # ============================================

                subtotal = invoice.get(
                    "subtotal"
                )

                tax_amount = invoice.get(
                    "tax_amount"
                )

                total_amount = invoice.get(
                    "total_amount",
                    0.0
                )


                t1, t2, t3 = st.columns(3)

                t1.metric(
                    "Subtotal",
                    (
                        f"${subtotal:,.2f}"
                        if subtotal is not None
                        else "N/A"
                    )
                )

                t2.metric(
                    "Tax Amount",
                    (
                        f"${tax_amount:,.2f}"
                        if tax_amount is not None
                        else "N/A"
                    )
                )

                t3.metric(
                    "Total Amount",
                    f"${total_amount:,.2f}"
                )


                # ============================================
                # AUDIO VERSION
                # ============================================

                invoice_audio_text = (
                    f"Invoice from "
                    f"{invoice.get('vendor_name', 'unknown vendor')}. "
                )

                if invoice.get("invoice_number"):

                    invoice_audio_text += (
                        f"Invoice number "
                        f"{invoice['invoice_number']}. "
                    )

                if invoice.get("date"):

                    invoice_audio_text += (
                        f"Date "
                        f"{invoice['date']}. "
                    )

                line_items = invoice.get(
                    "line_items",
                    []
                )

                if line_items:

                    invoice_audio_text += (
                        f"There are "
                        f"{len(line_items)} line items. "
                    )

                    for item in line_items:

                        invoice_audio_text += (
                            f"{item.get('description', 'Unknown item')}, "
                            f"quantity {item.get('quantity', 0)}, "
                            f"unit price "
                            f"{item.get('unit_price', 0):.2f}, "
                            f"line total "
                            f"{item.get('total_price', 0):.2f}. "
                        )

                if subtotal is not None:

                    invoice_audio_text += (
                        f"Subtotal "
                        f"{subtotal:.2f}. "
                    )

                if tax_amount is not None:

                    invoice_audio_text += (
                        f"Tax amount "
                        f"{tax_amount:.2f}. "
                    )

                invoice_audio_text += (
                    f"Total amount "
                    f"{total_amount:.2f}."
                )


                render_audio_reader(
                    invoice_audio_text,
                    key="invoice_result"
                )