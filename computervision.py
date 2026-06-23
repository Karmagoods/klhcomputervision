import streamlit as st


def render():
    st.sidebar.header("🧪 CV Lab")
    st.sidebar.markdown("---")

    tool_name = st.sidebar.radio(
        "Choose a module",
        [
            "🔍 Roboflow Detect",
            "✨ Gemini 3 Flash",
            "🏃 Motion Detection",
            "🙂 Face Detection",
        ],
        label_visibility="collapsed"
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with MediaPipe · OpenCV · Roboflow · Google Gemini")

    # Lazy imports — so a failed cv2/mediapipe import only breaks THAT module,
    # not the entire app. Each module handles its own import errors gracefully.
    if tool_name == "🔍 Roboflow Detect":
        from modules import roboflow_detect
        roboflow_detect.render()

    elif tool_name == "✨ Gemini 3 Flash":
        from modules import gemini3_flash_app
        gemini3_flash_app.render()

    elif tool_name == "🏃 Motion Detection":
        from modules import motion
        motion.render()

    elif tool_name == "🙂 Face Detection":
        from modules import face_detect
        face_detect.render()