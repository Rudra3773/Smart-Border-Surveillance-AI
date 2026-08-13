import streamlit as st
import cv2
import tempfile
from pathlib import Path
from ultralytics import YOLO


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Smart Border Surveillance AI")
st.subheader("Human vs Animal Intrusion Detection")


# -----------------------------
# Load YOLO Model
# -----------------------------
MODEL_PATH = Path(__file__).parent / "yolov8n.pt"

try:
    model = YOLO(str(MODEL_PATH))
    st.success("✅ YOLOv8 model loaded successfully!")

except Exception as e:
    st.error("❌ Failed to load YOLOv8 model")
    st.code(str(e))
    st.stop()


# -----------------------------
# Video Upload
# -----------------------------
uploaded_video = st.file_uploader(
    "🎥 Upload a surveillance video",
    type=["mp4", "avi", "mov", "mkv"]
)


# -----------------------------
# Detection
# -----------------------------
if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("🚀 Start Detection"):

        # Save uploaded video temporarily
        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(uploaded_video.read())
        input_file.close()

        input_path = input_file.name

        # Open video
        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open video.")
            st.stop()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps <= 0:
            fps = 25

        # Output video
        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        output_path = output_file.name
        output_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        out = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            (width, height)
        )

        # COCO classes
        animal_classes = {
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "bird"
        }

        human_detected = 0
        animal_detected = 0

        progress = st.progress(0)
        status = st.empty()

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frame_number = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1

            # YOLO detection
            results = model(
                frame,
                verbose=False
            )

            for result in results:

                boxes = result.boxes

                for box in boxes:

                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    class_name = model.names[cls_id]

                    # Ignore low-confidence detections
                    if confidence < 0.40:
                        continue

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )

                    # -------------------------
                    # HUMAN DETECTION
                    # -------------------------
                    if class_name == "person":

                        human_detected += 1

                        label = (
                            f"PERSON "
                            f"{confidence:.2f}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )

                    # -------------------------
                    # ANIMAL DETECTION
                    # -------------------------
                    elif class_name in animal_classes:

                        animal_detected += 1

                        label = (
                            f"ANIMAL: {class_name} "
                            f"{confidence:.2f}"
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 165, 255),
                            2
                        )

                        cv2.putText(
                            frame,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )

            # Write processed frame
            out.write(frame)

            # Progress
            if total_frames > 0:

                progress_value = (
                    frame_number / total_frames
                )

                progress.progress(
                    min(progress_value, 1.0)
                )

            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )

        cap.release()
        out.release()

        progress.progress(1.0)

        status.success(
            "✅ Video processing completed!"
        )

        # -----------------------------
        # Results
        # -----------------------------
        st.subheader("📊 Detection Summary")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "👤 Human Detections",
                human_detected
            )

        with col2:
            st.metric(
                "🐾 Animal Detections",
                animal_detected
            )

        # -----------------------------
        # Display processed video
        # -----------------------------
        st.subheader("🎥 Processed Surveillance Video")

        with open(output_path, "rb") as video_file:

            video_bytes = video_file.read()

        st.video(video_bytes)

        st.success(
            "🛡️ Surveillance analysis completed successfully."
        )
