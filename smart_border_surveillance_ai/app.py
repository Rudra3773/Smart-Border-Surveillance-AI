import streamlit as st
import cv2
import tempfile
import subprocess
import os
from pathlib import Path
from collections import Counter

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
# CUSTOM PAGE STYLE
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 22px;
        font-weight: 500;
        margin-bottom: 25px;
    }

    .summary-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-bottom: 15px;
        background-color: #fafafa;
    }

    .summary-number {
        font-size: 34px;
        font-weight: 700;
    }

    .summary-label {
        font-size: 16px;
        font-weight: 500;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛡️ Smart Border Surveillance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Human vs Animal Intrusion Detection</div>',
    unsafe_allow_html=True
)


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
# MAIN TWO-COLUMN LAYOUT
# =========================================================

left_col, right_col = st.columns(
    [1, 2],
    gap="large"
)


# =========================================================
# LEFT SIDE - SYSTEM CONTROLS
# =========================================================

with left_col:

    st.header("⚙️ System Controls")

    st.markdown("### 🎥 Upload Surveillance Video")

    uploaded_video = st.file_uploader(
        "Choose a surveillance video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv"
        ],
        label_visibility="collapsed"
    )

    st.markdown("")


    # -----------------------------------------------------
    # START DETECTION BUTTON
    # -----------------------------------------------------

    if uploaded_video is not None:

        start_detection = st.button(
            "🚀 Start Detection",
            type="primary",
            use_container_width=True
        )

    else:

        start_detection = False


    # -----------------------------------------------------
    # PLACEHOLDER FOR RESULTS
    # -----------------------------------------------------

    summary_placeholder = st.empty()


# =========================================================
# RIGHT SIDE - VIDEO AREA
# =========================================================

with right_col:

    if uploaded_video is not None:

        st.subheader("🎬 Uploaded Surveillance Video")

        st.video(
            uploaded_video,
            width=700
        )


# =========================================================
# DETECTION
# =========================================================

if uploaded_video is not None and start_detection:

    # =====================================================
    # SAVE INPUT VIDEO
    # =====================================================

    input_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    input_file.write(
        uploaded_video.getbuffer()
    )

    input_file.close()

    input_path = input_file.name


    # =====================================================
    # OPEN VIDEO
    # =====================================================

    cap = cv2.VideoCapture(
        input_path
    )

    if not cap.isOpened():

        st.error(
            "❌ Could not open uploaded video."
        )

        os.remove(input_path)

        st.stop()


    # =====================================================
    # VIDEO INFORMATION
    # =====================================================

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 25


    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    # =====================================================
    # TEMPORARY RAW OUTPUT
    # =====================================================

    raw_output = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".avi"
    )

    raw_output.close()

    raw_output_path = raw_output.name


    # =====================================================
    # VIDEO WRITER
    # =====================================================

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
            "❌ Could not create output video."
        )

        st.stop()


    # =====================================================
    # UNIQUE TRACK IDS
    # =====================================================

    human_ids = set()

    animal_ids = set()


    # =====================================================
    # ANIMAL SPECIES TRACKING
    # =====================================================

    animal_species = Counter()


    # =====================================================
    # PROGRESS
    # =====================================================

    progress = st.progress(0)

    status = st.empty()


    # =====================================================
    # FRAME PROCESSING
    # =====================================================

    frame_number = 0


    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # =================================================
        # YOLO TRACKING
        # =================================================

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            verbose=False
        )


        # =================================================
        # PROCESS DETECTIONS
        # =================================================

        for result in results:

            if result.boxes is None:
                continue


            boxes = result.boxes


            for box in boxes:

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


                # -----------------------------------------
                # TRACK ID
                # -----------------------------------------

                track_id = None

                if box.id is not None:

                    track_id = int(
                        box.id[0]
                    )


                # =================================================
                # HUMAN
                # =================================================

                if class_name == "person":

                    if track_id is not None:

                        human_ids.add(
                            track_id
                        )


                    label = (
                        f"HUMAN | person "
                        f"{confidence:.2f}"
                    )


                    # Green bounding box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )


                    # Label

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(
                                y1 - 10,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )


                # =================================================
                # ANIMAL
                # =================================================

                elif class_name in ANIMAL_CLASSES:

                    if track_id is not None:

                        animal_ids.add(
                            track_id
                        )

                        animal_species[
                            class_name
                        ] += 1


                    label = (
                        f"ANIMAL | "
                        f"{class_name} "
                        f"{confidence:.2f}"
                    )


                    # Orange bounding box

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 165, 255),
                        3
                    )


                    # Label

                    cv2.putText(
                        frame,
                        label,
                        (
                            x1,
                            max(
                                y1 - 10,
                                25
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA
                    )


        # =================================================
        # WRITE FRAME
        # =================================================

        out.write(frame)


        # =================================================
        # PROGRESS UPDATE
        # =================================================

        if total_frames > 0:

            progress_value = (
                frame_number /
                total_frames
            )

            progress.progress(
                min(
                    progress_value,
                    1.0
                )
            )

            status.write(
                f"Processing frame "
                f"{frame_number}/{total_frames}"
            )


    # =====================================================
    # RELEASE VIDEO
    # =====================================================

    cap.release()
    out.release()


    progress.progress(1.0)

    status.success(
        "✅ Video processing completed!"
    )


    # =====================================================
    # CONVERT TO H264 MP4
    # =====================================================

    status.info(
        "🎞️ Preparing processed video..."
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
            "❌ Could not prepare processed video."
        )

        st.code(str(e))

        st.stop()


    # =====================================================
    # RESULTS
    # =====================================================

    human_count = len(
        human_ids
    )

    animal_count = len(
        animal_ids
    )


    # =====================================================
    # LEFT SIDE - DETECTION SUMMARY
    # =====================================================

    with left_col:

        st.markdown("---")

        st.header("📊 Detection Summary")


        # -------------------------------------------------
        # HUMAN
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="summary-card">

            <div class="summary-label">
            👤 Humans Detected
            </div>

            <div class="summary-number">
            {human_count}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # ANIMAL
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="summary-card">

            <div class="summary-label">
            🐾 Animals Detected
            </div>

            <div class="summary-number">
            {animal_count}
            </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -------------------------------------------------
        # SPECIES BREAKDOWN
        # -------------------------------------------------

        if animal_species:

            st.markdown(
                "### 🐾 Animal Breakdown"
            )


            unique_species = set(
                animal_species.keys()
            )


            for species in sorted(
                unique_species
            ):

                st.write(
                    f"• {species.title()}"
                )


        # -------------------------------------------------
        # DOWNLOAD
        # -------------------------------------------------

        with open(
            final_output_path,
            "rb"
        ) as video_file:

            processed_video = (
                video_file.read()
            )


        st.download_button(
            label="⬇️ Download Processed Video",
            data=processed_video,
            file_name=(
                "smart_border_surveillance_result.mp4"
            ),
            mime="video/mp4",
            use_container_width=True
        )


        st.success(
            "🛡️ Surveillance analysis completed."
        )


    # =====================================================
    # RIGHT SIDE - PROCESSED VIDEO
    # =====================================================

    with right_col:

        st.markdown("---")

        st.subheader(
            "🎥 Processed Surveillance Video"
        )


        st.video(
            processed_video,
            width=700
        )


        st.success(
            "✅ Bounding boxes and object tracking applied."
        )


    # =====================================================
    # CLEAN TEMPORARY FILES
    # =====================================================

    try:

        os.remove(input_path)

        os.remove(raw_output_path)

        os.remove(final_output_path)

    except Exception:
        pass
