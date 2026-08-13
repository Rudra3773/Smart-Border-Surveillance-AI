import streamlit as st
import cv2
import tempfile
import subprocess
import os
from pathlib import Path

from ultralytics import YOLO


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Smart Border Surveillance AI",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# UI
# =========================================================

st.title("🛡️ Smart Border Surveillance AI")
st.subheader("Human vs Animal Intrusion Detection")


# =========================================================
# LOAD YOLO MODEL
# =========================================================

MODEL_PATH = Path(__file__).parent / "yolov8n.pt"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


try:
    model = load_model()

except Exception as e:
    st.error("❌ Unable to load detection model.")
    st.code(str(e))
    st.stop()


# =========================================================
# ANIMAL CLASSES
# =========================================================

ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe"
}


# =========================================================
# VIDEO UPLOAD
# =========================================================

uploaded_video = st.file_uploader(
    "🎥 Upload a surveillance video",
    type=["mp4", "avi", "mov", "mkv"]
)


# =========================================================
# PROCESS VIDEO
# =========================================================

if uploaded_video is not None:

    # -----------------------------------------------------
    # Display uploaded video
    # -----------------------------------------------------

    st.markdown("### 🎬 Uploaded Surveillance Video")

    st.video(
        uploaded_video,
        width=800
    )

    st.markdown("")

    # -----------------------------------------------------
    # Start Detection
    # -----------------------------------------------------

    start_detection = st.button(
        "🚀 Start Detection",
        type="primary"
    )

    if start_detection:

        # =================================================
        # SAVE INPUT VIDEO
        # =================================================

        input_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        input_file.write(uploaded_video.getbuffer())
        input_file.close()

        input_path = input_file.name


        # =================================================
        # OPEN VIDEO
        # =================================================

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("❌ Could not open uploaded video.")
            os.remove(input_path)
            st.stop()


        # =================================================
        # VIDEO INFORMATION
        # =================================================

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 25


        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )


        # =================================================
        # CREATE TEMPORARY RAW OUTPUT
        # =================================================

        raw_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".avi"
        )

        raw_output.close()

        raw_output_path = raw_output.name


        # =================================================
        # VIDEO WRITER
        # =================================================

        fourcc = cv2.VideoWriter_fourcc(
            *"XVID"
        )

        out = cv2.VideoWriter(
            raw_output_path,
            fourcc,
            fps,
            (width, height)
        )


        if not out.isOpened():

            cap.release()

            os.remove(input_path)
            os.remove(raw_output_path)

            st.error(
                "❌ Could not create processed video."
            )

            st.stop()


        # =================================================
        # DETECTION COUNTERS
        # =================================================

        human_detections = 0
        animal_detections = 0


        # =================================================
        # PROGRESS UI
        # =================================================

        progress = st.progress(0)

        status = st.empty()


        # =================================================
        # FRAME PROCESSING
        # =================================================

        frame_number = 0


        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1


            # -------------------------------------------------
            # YOLO INFERENCE
            # -------------------------------------------------

            results = model(
                frame,
                verbose=False,
                conf=0.40
            )


            # -------------------------------------------------
            # DRAW DETECTIONS
            # -------------------------------------------------

            for result in results:

                if result.boxes is None:
                    continue


                for box in result.boxes:

                    # -----------------------------------------
                    # CLASS
                    # -----------------------------------------

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = model.names[
                        class_id
                    ]


                    # -----------------------------------------
                    # CONFIDENCE
                    # -----------------------------------------

                    confidence = float(
                        box.conf[0]
                    )


                    # -----------------------------------------
                    # BOUNDING BOX
                    # -----------------------------------------

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0]
                    )


                    # =================================================
                    # HUMAN
                    # =================================================

                    if class_name == "person":

                        human_detections += 1

                        label = (
                            f"PERSON "
                            f"{confidence:.2f}"
                        )

                        # Green bounding box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )

                        # Label
                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(y1 - 10, 25)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA
                        )


                    # =================================================
                    # ANIMAL
                    # =================================================

                    elif class_name in ANIMAL_CLASSES:

                        animal_detections += 1

                        label = (
                            f"ANIMAL: "
                            f"{class_name.upper()} "
                            f"{confidence:.2f}"
                        )

                        # Orange bounding box
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 165, 255),
                            2
                        )

                        # Label
                        cv2.putText(
                            frame,
                            label,
                            (
                                x1,
                                max(y1 - 10, 25)
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2,
                            cv2.LINE_AA
                        )


            # =================================================
            # WRITE PROCESSED FRAME
            # =================================================

            out.write(frame)


            # =================================================
            # UPDATE PROGRESS
            # =================================================

            if total_frames > 0:

                progress_value = (
                    frame_number /
                    total_frames
                )

                progress.progress(
                    min(progress_value, 1.0)
                )

                status.write(
                    f"Processing video: "
                    f"{frame_number}/{total_frames} frames"
                )


        # =================================================
        # RELEASE VIDEO
        # =================================================

        cap.release()
        out.release()


        progress.progress(1.0)

        status.success(
            "✅ Video processing completed!"
        )


        # =================================================
        # CONVERT TO BROWSER-FRIENDLY MP4
        # =================================================

        st.info(
            "🎞️ Preparing processed video for browser playback..."
        )


        final_output = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        final_output.close()

        final_output_path = final_output.name


        try:

            import imageio_ffmpeg

            ffmpeg_path = (
                imageio_ffmpeg.get_ffmpeg_exe()
            )


            command = [
                ffmpeg_path,

                "-y",

                "-i",
                raw_output_path,

                "-c:v",
                "libx264",

                "-preset",
                "veryfast",

                "-crf",
                "23",

                "-pix_fmt",
                "yuv420p",

                "-movflags",
                "+faststart",

                final_output_path
            ]


            subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )


        except Exception as e:

            st.error(
                "❌ Could not prepare the processed video."
            )

            st.code(str(e))

            cap.release()

            if os.path.exists(input_path):
                os.remove(input_path)

            if os.path.exists(raw_output_path):
                os.remove(raw_output_path)

            if os.path.exists(final_output_path):
                os.remove(final_output_path)

            st.stop()


        # =================================================
        # DETECTION SUMMARY
        # =================================================

        st.markdown("## 📊 Detection Summary")

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "👤 Person Detection Instances",
                human_detections
            )


        with col2:

            st.metric(
                "🐾 Animal Detection Instances",
                animal_detections
            )


        # =================================================
        # PROCESSED VIDEO
        # =================================================

        st.markdown(
            "## 🎥 Processed Surveillance Video"
        )


        with open(
            final_output_path,
            "rb"
        ) as video_file:

            processed_video = video_file.read()


        st.video(
            processed_video,
            width=800
        )


        # =================================================
        # DOWNLOAD OPTION
        # =================================================

        st.download_button(
            label="⬇️ Download Processed Video",
            data=processed_video,
            file_name="smart_border_surveillance_result.mp4",
            mime="video/mp4"
        )


        # =================================================
        # FINAL STATUS
        # =================================================

        st.success(
            "🛡️ Surveillance analysis completed successfully."
        )


        # =================================================
        # CLEAN TEMP FILES
        # =================================================

        try:

            os.remove(input_path)

            os.remove(raw_output_path)

            os.remove(final_output_path)

        except Exception:
            pass
