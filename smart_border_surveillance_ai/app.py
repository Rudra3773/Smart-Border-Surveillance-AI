import streamlit as st

st.title("OpenCV Deployment Test")

try:
    import cv2

    st.success("✅ OpenCV imported successfully!")
    st.write("OpenCV version:", cv2.__version__)

except Exception as e:
    st.error("❌ OpenCV import failed")
    st.code(str(e))
