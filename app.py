import streamlit as st
import cv2
import tempfile

from motion.motion_detector import MotionDetector
from motion.thermal import thermal_view, is_night
from detection.object_detector import ObjectDetector


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    layout="wide"
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("🛡️ Smart Border Surveillance AI")

st.caption(
    "Human & Animal Intrusion Detection | "
    "Automatic Day/Night Surveillance System"
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("System Controls")

uploaded_video = st.sidebar.file_uploader(
    "Upload Surveillance Video",
    type=["mp4", "avi", "mov"]
)


# --------------------------------------------------
# INITIALIZE DETECTORS
# --------------------------------------------------

motion_detector = MotionDetector()

object_detector = ObjectDetector(
    "models/yolov8n.pt"
)


# --------------------------------------------------
# PLACEHOLDERS
# --------------------------------------------------

frame_placeholder = st.empty()

alert_placeholder = st.empty()

status_placeholder = st.empty()


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

if uploaded_video is not None:

    temp_video = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_video.write(
        uploaded_video.read()
    )

    temp_video.close()

    cap = cv2.VideoCapture(
        temp_video.name
    )

    if not cap.isOpened():

        st.error(
            "❌ Unable to open video."
        )

        st.stop()


    # --------------------------------------------------
    # PROCESS VIDEO
    # --------------------------------------------------

    frame_count = 0
    FRAME_SKIP = 2

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % FRAME_SKIP != 0:
            continue


        # --------------------------------------------------
        # DAY / NIGHT DETECTION
        # --------------------------------------------------

        night_mode = is_night(frame)


        # --------------------------------------------------
        # MOTION DETECTION
        # --------------------------------------------------

        motion_detected = motion_detector.detect(
            frame
        )


        # --------------------------------------------------
        # OBJECT DETECTION
        # IMPORTANT:
        # Run YOLO independently of motion detection
        # --------------------------------------------------

        detections = object_detector.detect(
            frame
        )


        # --------------------------------------------------
        # DISPLAY FRAME
        # --------------------------------------------------

        if night_mode:

            display_frame = thermal_view(
                frame
            )

            status_placeholder.warning(
                "🌙 NIGHT MODE — Thermal Visualization Active"
            )

        else:

            display_frame = frame

            status_placeholder.success(
                "☀️ DAY MODE — Normal Surveillance Active"
            )


        # --------------------------------------------------
        # DRAW DETECTIONS
        # --------------------------------------------------

        human_detected = False
        animal_detected = False


        for det in detections:

            x1, y1, x2, y2 = det["bbox"]

            category = det["category"]

            label = det["label"]

            confidence = det["confidence"]


            # --------------------------------------------------
            # HUMAN
            # --------------------------------------------------

            if category == "HUMAN":

                human_detected = True

                box_color = (0, 255, 0)


            # --------------------------------------------------
            # ANIMAL
            # --------------------------------------------------

            else:

                animal_detected = True

                box_color = (0, 165, 255)


            # Bounding box
            cv2.rectangle(
                display_frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # Label
            text = (
                f"{category} | "
                f"{label} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                display_frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


        # --------------------------------------------------
        # ALERT SYSTEM
        # --------------------------------------------------

        if human_detected:

            alert_placeholder.error(
                "🚨 ALERT: HUMAN INTRUSION DETECTED"
            )

        elif animal_detected:

            alert_placeholder.warning(
                "🐾 ALERT: ANIMAL INTRUSION DETECTED"
            )

        elif motion_detected:

            alert_placeholder.info(
                "⚠️ MOTION DETECTED — No classified object"
            )

        else:

            alert_placeholder.empty()


        # --------------------------------------------------
        # DISPLAY
        # --------------------------------------------------

        frame_placeholder.image(
            display_frame,
            channels="BGR",
            use_container_width=True
        )


    cap.release()

else:

    st.info(
        "📹 Upload a surveillance video to start detection."
    )