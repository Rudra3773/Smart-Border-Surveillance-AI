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

except Exception as e:
    st.error("❌ MotionDetector failed")
    st.code(str(e))
