import streamlit as st
import cv2
import tempfile

from motion.motion_detector import MotionDetector
from motion.thermal import thermal_view
from detection.object_detector import ObjectDetector

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    layout="wide"
)

st.title("🛡️ Smart Border Surveillance AI")
st.caption("Human vs Animal Intrusion Detection | Defense Surveillance System")

# Sidebar
st.sidebar.header("System Controls")
thermal_mode = st.sidebar.checkbox("Enable Night / Thermal Mode", value=True)
uploaded_video = st.sidebar.file_uploader(
    "Upload Surveillance Video",
    type=["mp4", "avi"]
)

motion_detector = MotionDetector()
object_detector = ObjectDetector()

frame_placeholder = st.empty()
alert_placeholder = st.empty()

if uploaded_video is not None:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video.name)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        motion_detected = motion_detector.detect(frame)

        display_frame = thermal_view(frame) if thermal_mode else frame

        if motion_detected:
            detections = object_detector.detect(frame)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                label = det["category"]
                conf = det["confidence"]

                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    display_frame,
                    f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                if label == "HUMAN":
                    alert_placeholder.error("🚨 ALERT: HUMAN INTRUSION DETECTED")

        frame_placeholder.image(display_frame, channels="BGR")

    cap.release()
