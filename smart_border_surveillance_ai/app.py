import streamlit as st
import cv2
import tempfile

from motion.motion_detector import MotionDetector

st.title("Smart Border Surveillance AI")

st.success("✅ OpenCV loaded successfully!")
st.write("OpenCV version:", cv2.__version__)

try:
    detector = MotionDetector()
    st.success("✅ MotionDetector loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload a test image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        frame = cv2.imdecode(
            __import__("numpy").frombuffer(file_bytes, dtype="uint8"),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            st.error("❌ Could not read image")
        else:
            motion_detected = detector.detect(frame)

            st.image(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                caption="Test Frame"
            )

            if motion_detected:
                st.warning("🚨 Motion detected!")
            else:
                st.success("🟢 No significant motion detected.")

except Exception as e:
    st.error("❌ Test failed")
    st.code(str(e))
